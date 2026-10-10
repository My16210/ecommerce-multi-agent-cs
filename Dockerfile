# syntax=docker/dockerfile:1
# ============================================================================
# 多 Agent 电商智能客服
#
# 这个文件是**重写的**，不是从另外两个仓库搬来的：原来那份是围着 RAG 那个
# 仓库自己的 app.py 转的（只跑知识库那一路、只装它要的 extra），而这个项目
# 的入口是统一的 app.py，四个 Tab 要同时可用。
#
# 多阶段构建：
#   builder —— 用 uv 装依赖，装进 /app/.venv
#   runtime —— 只搬 .venv + 源码，不带 uv、不带编译工具、不带 dev 依赖
#
# 三个必须说清的决定，都写在对应位置旁边：
#   1. 为什么这次**不用** --frozen（没有 uv.lock）
#   2. 为什么四个 extra 全装
#   3. 为什么密钥在这个文件里一次都不出现
# ============================================================================

# ---------------------------------------------------------------------------
# 阶段一：装依赖
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# uv 本体从官方镜像里取（这个 tag 是镜像，不是 Python 环境，只借它的 uv 二进制）
COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# ⚠️ 这个 WORKDIR 在 runtime 阶段**必须一模一样**。
# uv 建出来的 .venv 里写的是**绝对路径**（/app/.venv/bin/python 之类）。
# 换个目录搬过去，venv 里的解释器路径就全指错了——症状是容器一起来就
# "can't open file" 或 import 全失败，而且看不出跟路径有关。
#
# 这个项目还多一层依赖：`ecom/settings.py` 的项目根就是 `__file__` 往上数两层，
# 在这个文件里 = /app。所以 /app 同时是 venv 的家、源码的家、和"根"。
WORKDIR /app

# 先只拷依赖清单，让"装依赖"这一层能独立命中缓存：
# 之后改代码不会重装依赖，改依赖才会。
#
# ⚠️ 这里**没有 uv.lock，所以下面不能写 --frozen**。
# 别把它当成"忘了"：本机 PyPI 不通，生成不了锁文件；而一份手写或从别的项目
# 抄来的锁**比没有锁更坏**——它会让人以为依赖是钉死的，实际不是。
# 代价写清楚：这个镜像每次构建都会重新解析依赖版本，**两次构建的镜像不保证
# 依赖完全一致**。要消除它，在有网的环境里跑一次 `uv lock` 再把 uv.lock
# 拷进来、把下面的 `uv sync` 改成 `uv sync --frozen`。
COPY pyproject.toml .python-version* ./

# --no-dev      排除 [dependency-groups] 的 dev（pytest/ruff），运行时用不到
# --no-install-… 本项目故意没有 [build-system]（是应用不是库），装不了自己
#
# **四个 extra 全装是刻意的**：项目的核心契约之一是「换后端只改配置」
# （knowledge 的向量库、订单库、联网搜索都如此）。镜像里若只有默认那一套，
# 那条契约在容器里就变成半真半假——你在 .env 里写了 VECTOR_BACKEND=milvus，
# 容器一起来报的却是"请先装 pymilvus"。代价是镜像大一点，换来确定性。
#
# 注意 pandas / numpy / matplotlib 不在这里——它们是**内核依赖**，不是 extra。
# 数据分析是三个核心能力之一，镜像装的是内核，所以它一定在。
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-dev \
            --extra ui --extra api --extra milvus --extra mysql --extra search

# ---------------------------------------------------------------------------
# 阶段二：运行时
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# LANG=C.UTF-8 让中文路径/输出在容器里正常（Linux 上没有 Windows cp936 那个问题，
# 所以这里也**不设** PYTHONIOENCODING——代码里那几个绕 cp936 的地方是给宿主的
# 控制台准备的，容器里不该跟着改口径）。PYTHONUNBUFFERED 让日志实时出来。
#
# MPLBACKEND=Agg 是必须的：容器里没有显示器，matplotlib 若是挑到交互式后端，
# 会在数据分析那一步 `savefig` 时直接失败——而症状会是"出图那一步报了一个跟
# 后端有关的错"，跟"图"看起来没关系。显式钉成 Agg，这条不确定性就没了。
ENV PATH="/app/.venv/bin:$PATH" \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# 中文字体。**这不是"锦上添花"，是让数据分析那一步的输出可读**。
#
# python:3.11-slim 里一个 CJK 字体都没有，而数据分析生成的图表标题/轴标签全是
# 中文。缺字体的表现极其隐蔽：matplotlib 只往 stderr 打一行 findfont warning，
# 图照常保存、子进程照常 exit 0、worker 的 JSON 里 chart_generated 照常是 true，
# 只有**把 PNG 打开看**才知道每个汉字都是豆腐块。（这一条是真在容器里跑出来
# 才发现的，当时的输出全绿。）
#
# fonts-noto-cjk 是这几个候选里质量最好的，代价是装了约 90MB——镜像里已经有
# chromadb / matplotlib / streamlit，多这 90MB 换"图能看"是划算的。
# 它同时提供了 ecom/analytics/core/sandbox.py 里 font.sans-serif 列表中的
# 「Noto Sans CJK SC」，两边是配套的：这里装不上，那边那一项就落空。
#
# fc-cache -f 显式刷一次字体缓存，不依赖 apt 的 hook 顺序。
RUN apt-get update \
 && apt-get install -y --no-install-recommends fonts-noto-cjk fontconfig \
 && fc-cache -f \
 && rm -rf /var/lib/apt/lists/*

# 非 root 运行。uid/gid 做成 ARG 是为了对齐宿主上的属主：
# 在 Ubuntu 上 `chown -R <uid>:<gid> data logs` 就能让非 root 进程写挂载卷，
# 否则容器会以"权限拒绝"的形式挂掉。
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd --gid "${APP_GID}" app \
 && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --shell /usr/sbin/nologin app

WORKDIR /app

# 依赖（整个 venv）
COPY --from=builder /app/.venv /app/.venv

# 源码。触发上下文里有什么由 .dockerignore 决定——
# 那里挡掉了 .env、.venv、data/knowledge/vector_store、*.db、logs、tests。
COPY . /app

# 这几个目录代码会用到，都得可写（路径都锚在 /app，也就是 ecom/settings.py 里的 ROOT）：
#   data/knowledge/documents/reviewed —— 人审通过的知识落盘处
#   data/knowledge / data/analytics    —— 向量库、评测产物、图表输出
#   data（本身）                        —— orders.sqlite 建在这里
#   logs                               —— 运行产物
# 容器里通常被 ./data 和 ./logs 两个卷盖住，这里建出来是为了"不起卷也能跑"。
RUN mkdir -p /app/data/knowledge/documents/reviewed /app/data/analytics/output /app/logs \
 && chown -R app:app /app/data /app/logs

USER app

# Streamlit 的默认端口。注意这只是声明，真正监听哪个地址由下面的 CMD 决定。
EXPOSE 8501

# 健康检查用 Python 标准库而不是 curl：slim 镜像里没有 curl，
# 为一次健康检查装一个包不值。/_stcore/health 是 Streamlit 自带的端点。
#
# 注意它检查的是**界面进程活着**，不是"模型和向量库可用"——探针按秒调，
# 让它去 ping 一遍模型会把系统探死。要深检查就 exec 进去跑一次 main.py。
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request\nok = False\ntry:\n    ok = urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).status == 200\nexcept Exception:\n    pass\nsys.exit(0 if ok else 1)"]

# --server.address=0.0.0.0 是这里最容易漏的一条：Streamlit 默认只监听
# 127.0.0.1，容器外**连不上**，而症状是"端口映射明明对了却打不开"。
# 用命令行参数而不是 .streamlit/config.toml，是为了不影响宿主上的本地开发
# （那份配置会同时作用于 `uv run streamlit run app.py`）。
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
