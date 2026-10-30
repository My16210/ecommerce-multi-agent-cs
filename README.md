# 多 Agent 电商智能客服

一个 LangGraph 的 **Supervisor + Worker** 系统：用户一句自然语言，系统自己判断该由谁办，并行调完再把结论合成一句话答复。

面向的场景是电商客服里最典型的三类问题：

| 问题 | 谁来办 | 为什么是它 |
|---|---|---|
| "退货政策是什么""保修怎么算" | **知识库子系统** | 答案在文档里，要检索 |
| "订单 52644 到哪了""我要退货" | **订单子系统** | 答案在数据库里，要查/要写 |
| "哪个大区卖得最好""延迟交付风险最高的是哪种发货方式" | **数据分析子系统** | 答案要算出来，要跑代码 |

而真实用户不会按这个分类说话。「订单 52644 到哪了，顺便问下你们的退货政策」——一句话两个需求，这正是这个项目要解决的事。

**它不是**：一个把提示词写得花哨的单体 chatbot，也不是三个 bot 互相聊天的"群体智能"。它是一个**调度 + 隔离 + 汇总**的工程结构，每一层为什么这么切、哪些地方现在不该做，下面都写清楚了。

> 文档按场景读：
>
> 1. **[docs/项目说明.md](docs/项目说明.md)** —— 给评审看的导读，不假设你读过代码。
>    "它是什么 / 为什么这么做 / 做到什么程度"。
> 2. **本文件** —— 开发者视角的完整版：目录、逐条实现、踩过的坑、验证到哪一步。
> 3. **[docs/技术文档.md](docs/技术文档.md)** —— **要改代码的人读的**。每个契约长什么样、
>    为什么是这个形状、改它会连带动什么，**每处配真实代码片段**。
> 4. **[docs/上线清单.md](docs/上线清单.md)** —— 换个轴问问题：**这套东西要真上线还差什么**。
>    按"出事顺序"排，每条写清现在什么样、会怎么出事、要补什么，最后有一节**明确不做**。
> 5. **[docs/DEMO.md](docs/DEMO.md)** / **[docs/RUNBOOK.md](docs/RUNBOOK.md)** ——
>    公网只读演示（L1）能做什么、不能做什么，以及部署/换邀请码/查预算/排障的步骤。
> 6. **[docs/项目拆解.md](docs/项目拆解.md)** —— **要接手或要讲清楚这个项目的人读的**。
>    按"一次请求怎么走"纵向拆开：四条主链路、每个模块的契约与失败模式、配置矩阵、
>    测试地图、排障症状表、扩展点的改动半径、部署形态、六个真实故障故事。

---

## 这个仓库是怎么来的

它原本是**三个独立的仓库**：一个 RAG 知识库（问答 + 人审 + 评测，约 10k 行）、一个销售数据分析 agent（写代码 → 执行 → 纠错，约 1.7k 行）、以及这个编排层。合并的触发点不是"看起来更整齐"，是一次真实事故：

> 编排层的配置一开始放在 `config/settings.py`——和知识库仓库保持一致，听起来是"风格统一"。结果是知识库那 16 个 `from config.settings import ...` **全被劫持到了编排层这份上**，报出来的是 `ImportError: cannot import name 'LOGS_PATH'`。而 `LOGS_PATH` 和故障原因一点关系都没有。
>
> 更坏的情况是**它悄悄找到了一个同名但含义不同的常量**：不报错，只是行为不对。

当时的应对是绕开：编排层改用 `cs/` 这个"足够独特、永远撞不上"的顶层名，加一道运行时闸门检查 import 到底来自哪个仓库，知识库走进程内、数据分析走子进程。**整套设计都是被这一件事逼出来的。**

合并要买的正是这个：sibling 变成了自己，"撞名"这个约束**从构造上不存在了**。于是

| 合并前为了绕开它而存在的东西 | 合并后 |
|---|---|
| 编排层叫 `cs`（为了独特） | 叫 `ecom.orchestrator`（不再需要独特） |
| `_assert_imports_came_from()` 运行时闸门 + 它的测试 | 删除 |
| `sys.path.insert` 挂 sibling 仓库 | 删除 |
| `WORKSPACE_ROOT` / `RAG_KB_ROOT` / `SALES_AGENT_ROOT` / `SALES_AGENT_PYTHON` | 四个配置项全删 |
| 依赖表里"因为进程内调用所以重复声明 rag 那几个包" | 一张表，本来就只有一份 |
| sales_agent 那 9 个测试文件里的 `sys.path.insert` 自举 | 删除，由 `pythonpath=["."]` 统一管 |
| 三份 `pytest` / `ruff` 配置、两份相似的向量化实现、两个 `app.py` | 各一份 |

**没有一行业务逻辑被重写。**除了那几个接缝文件和新的 `app.py`，其余全部是"搬 + 改 import"——约 270 处，分布在 130 个文件里。

**原两个仓库原地保留、一行未动**（它们各自有独立的 commit 历史和文档），这里合并的是副本。

---

## 目录

```
ecom_cs/                         仓库根 = 唯一的 import 根，也是唯一入口所在地
├── pyproject.toml               一张依赖表（合并三家）
├── uv.lock                      锁文件（容器里生成，构建走 --frozen）
├── LICENSE                      MIT
├── .github/workflows/tests.yml  CI：ruff + pytest
├── .env / .env.example          一份配置
├── .streamlit/config.toml       主题（浅/深两套）—— app.py 和 admin.py **共用这一份**
├── main.py                      CLI —— **第一入口**，排查问题用这个
├── app.py                       Streamlit 界面 —— 主区只有对话，另外三个视图收在侧边栏
├── admin.py                     Streamlit 后台 —— 口令 / admin token，能批准退货的那一个
├── api.py                       FastAPI    —— 可选
│
├── ecom/                        ★ 唯一顶层包，三个平级的子系统
│   ├── settings.py              共享配置：ROOT / DATA_DIR / LOGS_DIR / 密钥 / 模型名
│   │                            **全项目唯一一处读 .env、唯一一处算项目根**
│   ├── orchestrator/            编排：supervisor / router / aggregate / workers / db
│   │   ├── tools/               **模型能调到的**（order_tools.py 的纯函数）
│   │   ├── admin/               **只有人能下的结论**（returns.py 的退货审核）
│   │   ├── cli/                 returns.py / session.py / memory.py —— 和界面共用同一份实现
│   │   ├── utils/ids.py         订单号的形状 —— 唯一一份定义
│   │   ├── session.py           会话与记忆的存取层（自己开 sqlite，不用 Database）
│   │   └── memory.py            蒸馏 / 召回（要联网，所以不塞进 session.py）
│   ├── knowledge/               知识库：retriever / vector_store / embedding / generator /
│   │                            document_loader / review / web_search / cli（10 个脚本）
│   ├── analytics/               数据分析：agents / core（graph + sandbox）/ utils / cli
│   ├── mcp/                     MCP 工具服务：tools.py（纯函数契约）/ server.py（stdio 接线）
│   ├── auth/                    身份层：principal / 角色 / token 哈希 / 审计（可选）
│   └── ui/                      Streamlit 渲染件与三个直连视图（app.py 只做装配）
│
├── data/                        三个子系统的数据按目录分开
│   ├── knowledge/               documents/（「法规原文」10 篇 + 「电商客服示例」7 篇）、
│   │                            评测集、金标准、向量库
│   ├── analytics/
│   │   ├── datasets/<名>/       一个数据集一个目录：data.csv + NOTES.md（口径的唯一出处）
│   │   │                        sales = 手写的最小示例；dataco = 真实公开数据集（默认）
│   │   ├── source/              下载来的原始数据（96MB，**不进仓库**，见 .gitignore）
│   │   └── output/              跑出来的图（不进仓库）
│   ├── orchestrator/seed/       订单库种子 orders.csv / logistics.csv（派生物，**进仓库**）
│   │                            router_eval.jsonl（路由评测题集，50 题）
│   └── orders.sqlite            订单库运行时文件（不进仓库，由 seed 重建）
│                                会话库在 data/orchestrator/sessions.db（同样不进仓库）
│                                身份库在 data/orchestrator/auth.db（token 哈希 + 审计）
├── logs/                        日志 + requests.jsonl（trace）
├── frontend/                    React + Vite + TypeScript（可选界面）
│                                `npm run build` 后由 api.py 挂到 /ui
├── docs/                        项目说明.md（导读）｜技术文档.md（**要改代码的人读的**）
│                                上线清单.md（**要真上线还差什么**，含"明确不做"）
│                                MCP.md（工具清单、契约、安全边界）
│                                项目拆解.md（纵向拆解）
└── tests/                       orchestrator/ knowledge/ analytics/ mcp/ auth/ + 跨系统的不变量测试
```

**四个入口 + 一条 MCP 出口，各自的定位**（详见 [技术文档 §11](docs/技术文档.md)）：

| 入口 | 给谁 | 读/写 |
|---|---|---|
| `main.py` | **排查问题**（第一入口） | 读 |
| `app.py` → 8501 | 演示 | 只读 |
| `admin.py` → 8502 | **下结论**（退货审核） | **三个页面里只有一个能写** |
| `api.py` → 8000 | 被别的程序调 | 读；`AUTH_ENABLED=1` 时按角色和会话归属 |
| `python -m ecom.mcp.server` → stdio | **别的 Agent**（MCP 客户端） | 默认只读；写操作两把钥匙 |

### 为什么是一个 `ecom` 而不是三个平级顶层包

顶层包名就是全局命名空间。**一个 `ecom` 从构造上就不可能撞**——这正是那次事故的结论，用它收尾比再挑三个"足够独特"的名字更干净，也把三个子系统摆成对等的三块，而不是"主项目 + 两个依赖"。

代价是 `cs.` → `ecom.orchestrator.` 这 80 处也要改。这 80 处在自己的代码里、有 78 条测试兜着，是整次改动里最安全的一段。

---

## 跑起来

```bash
git clone <本仓库> ecom_cs && cd ecom_cs
cp .env.example .env      # 填 QWEN_API_KEY（路由 + 生成）与 SILICONFLOW_API_KEY（向量化 + rerank）
uv sync                   # 界面：--extra ui ｜ HTTP：--extra api ｜ MCP：--extra mcp ｜ Milvus：--extra milvus
uv run python main.py --init-db                                  # 灌 65752 笔订单 + 62897 条物流
uv run python -m ecom.knowledge.cli.ingest                       # 建向量库
uv run python main.py "订单 52644 到哪了"
uv run python main.py "订单 52644 到哪了" --stream       # 边生成边打印（只有汇总那步是真流）
```

**不再需要另外两个仓库**——`.env` 里那三个 `RAG_KB_ROOT` / `SALES_AGENT_ROOT` / `SALES_AGENT_PYTHON` 已经删掉了，它们的存在本身就是"没合干净"的标记。

不填 `QWEN_API_KEY` 也能跑：路由会退回关键词规则，订单那一路完全不依赖模型。**这是设计的一部分，不是碰巧**（见下面"三层路由"）。

四个界面入口（前三个用 `uv run`，第四个是 HTTP）：

```bash
uv run streamlit run app.py                            # → 8501  客户界面
uv run streamlit run admin.py                          # → 8502  后台（ADMIN_PASSWORD 或 admin token）
uv run uvicorn api:app --port 8000                     # → 8000  HTTP 接口
uv run python -m ecom.orchestrator.cli.returns stats   # 命令行版的退货审核
uv run python -m ecom.mcp.server --list-tools          # MCP 工具列表（要先 --extra mcp）
uv run python -m ecom.auth.cli create --name alice --role admin  # 生成 admin token
uv run python -m ecom.orchestrator.cli.eval_router               # 路由评测（离线规则基线）
```

> **默认（`AUTH_ENABLED=0`）**：`admin.py` 不配 `ADMIN_PASSWORD` 会拒绝启动（fail closed），
> `api.py` 不配 `API_KEY` 就放行。**对外提供 API / 后台时请打开 `AUTH_ENABLED=1`**：
> `/chat` 要 `Authorization: Bearer <token>`、`/trace` 要 admin 角色、会话按 principal
> 归属、`admin.py` 改用 admin token。两条路径的取舍见下面「身份与权限」。

### 用 Docker 跑（可选）

```bash
cp .env.example .env
docker compose build app                               # ← 必须先建，理由见下
docker compose up -d                                   # 默认：Chroma 嵌入式 + SQLite，零外部依赖
docker compose run --rm app python main.py --init-db                   # 灌订单库（65752 笔）—— 干净 clone 才需要
docker compose run --rm app python -m ecom.knowledge.cli.ingest        # 建向量库 —— 同上
# → http://localhost:8501     客户界面（端口被占就 APP_PORT=8600 docker compose up -d）
# → http://127.0.0.1:8502     后台（要 .env 里的 ADMIN_PASSWORD）
```

**`build` 为什么必须单独一步**：`up -d` 只在镜像**不存在**时才构建，镜像一旦在就直接用旧的。
镜像是 `COPY . /app`，所以**改任何源码之后都要重新 `build`**（`admin` 共用同一个
`ecom-cs:local` 镜像，一起更新）。实测踩到过 —— 镜像是 15:41 建的，代码 16:54 才修完，
`up -d` 会静悄悄起一份旧代码，界面上、日志里都看不出来。逐步的检查和"怎么算成了"写在
[技术文档第十一节](docs/技术文档.md#十一怎么跑)。

**React 前端由镜像自己构建**：Dockerfile 里的 `frontend` 阶段跑 `npm ci && npm run build`，
`/ui` 不再依赖宿主机先执行 `npm run build`（干净 clone 直接构建也能拿到）。本地直接
`uv run uvicorn api:app` 时仍然需要宿主机构建一次前端。

**两条都只发布到 `127.0.0.1`，不绑 `0.0.0.0`。** 用 `localhost` 访问完全不受影响；
但同局域网的另一台机器打不开它们——这是故意的（app 那条**没有任何鉴权**，
而它能触发一次数据分析，也就是让模型生成的 Python 跑起来）。
真要跨机器演示：`APP_BIND=0.0.0.0 docker compose up -d`。

换 Milvus：`docker compose --profile milvus up -d`（四个服务一起起）。

五个设计点：

- **Milvus 发布在 `19531` 而不是 `19530`**，etcd / minio 干脆不发布端口。因为本机 19530 上常年跑着一个 Milvus，照抄默认端口必然冲突。这条不是重新想的，是从 RAG 那个仓库原样保留的。容器里要连宿主上那个 Milvus，就设 `DOCKER_MILVUS_URI=http://host.docker.internal:19530`——**不能用 `localhost`，那在容器里指向容器自己**。
- **`data/` 和 `logs/` 挂宿主目录**（bind mount），所以宿主上已经建好的向量库、订单库、图表目录容器里开箱即用。这也正是"数据收到一处"的直接收益：合并前三份数据在三个仓库各自的 `data/` 里，这个卷得挂三次。代价是 Linux 上要 `sudo chown -R 1000:1000 data logs`（容器里是非 root 的 `app` 用户）；Windows 的 Docker Desktop 会把权限放宽，通常不用管。
- **镜像里装了 `fonts-noto-cjk`**（约 90MB）。不装的话图能生成、退出码是 0，只有把 PNG 打开才知道中文全是豆腐块——见下面"坑 7"。
- **`--profile milvus` 时 app 没有 `depends_on: milvus`**。compose 对"依赖一个只存在于某个 profile 里的服务"处理得别扭；而 Milvus 起不来时 `MilvusStore` 自己会抛一条能照着做的错误，比编排层的依赖约束更直接。所以顺序交给文档：四个一起起，若 app 先就绪导致第一问失败，重试一次即可（Milvus 的 `start_period` 是 90s）。
- **`admin` 服务发布的是 `127.0.0.1:${ADMIN_PORT:-8502}:8502`，不是 `8502:8502`**。这是口令之外的**第二层**，两层挡的不是同一件事：口令挡"能连到这个端口的人"，绑定挡"根本连不到这个端口的人"（同局域网的另一台机器、宿主上任何程序）。**两层都要有。** 副作用如实说：这也意味着它不能被反向代理从外面转发——真上线时该把那层代理自己接过来，而不是把这一行改回 `0.0.0.0`。另外它的 healthcheck **必须自己覆盖**：镜像里那份探的是写死的 8501，不覆盖的话 admin 容器会**永远报不健康**，而服务其实是好的。

⚠️ 两个如实说明：镜像**构建过也跑过**（见「验证到哪一步」），但**界面本身没在浏览器里打开过**（真 Streamlit 运行时跑过，见下）；另外 `docker compose config` 会把 `env_file` 解析并**打印出来**，所以别把它重定向进文件再拿去贴。

### 公网只读演示（L1，可选）

`docker-compose.yml` 是**本机开发**用的：没有鉴权、能跑数据分析、后台也起得来。
要**挂到公网上给别人点开看**，用的是另一套编排：

```bash
cp .env.prod.example .env.prod      # 填 DOMAIN / DEMO_INVITE_CODE / 两个模型 Key
chmod 600 .env.prod
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d --build
```

它和开发编排的差别不是"换个端口"，是**三条防线同时打开**：

- **只读**：退货申请在落库前被挡住（API / CLI 走同一道闸，不是藏按钮）；
  数据分析整条路停用（它会在服务器上执行模型生成的代码）；后台管理端不部署。
- **邀请码**：进界面 / 调 API 都要口令。**没配就拒绝所有人**（fail closed），
  不是"没配就放行"。
- **成本**：每分钟限流挡突发，每日 token 预算落盘记账（重启也算数）兜成本；
  用完当天就停服务。

外加 Caddy 自动签 HTTPS、app 不发布宿主端口、容器限内存/CPU。能做什么、
不能做什么、为什么关掉那些能力写在 [docs/DEMO.md](docs/DEMO.md)；
部署、换邀请码、查用量、排障、备份回滚写在 [docs/RUNBOOK.md](docs/RUNBOOK.md)。

**这一层仍是邀请码门禁；身份层（`AUTH_ENABLED`）是可选的另一层，且没有监控告警、多副本**，别把它当"上线完成"——
缺口清单见 [上线清单.md](docs/上线清单.md) 第六节。

---

## 一次请求的走向

```
$ uv run python main.py "订单 52644 到哪了，顺便问下你们的退货政策是什么" -v

路由上限 2 轮　worker 超时 {'knowledge': 90, 'order': 20, 'analytics': 180}
==================================================================
订单号 52644 的物流目的地是圣萨尔瓦多，当前状态为“延迟送达”，最近一次更新时间为
2017年2月12日11:11；但知识库中没有该订单的物流状态相关内容，无法查询具体进展，
建议您登录账户在“我的订单”中查看实时物流信息，或联系客服提供订单号进一步协助。

关于退货政策：自签收次日起7个自然日内，商品完好（含配件、吊牌齐全，无使用痕迹、
污渍、损坏）且不影响二次销售，可申请七天无理由退货；拆开外包装查验商品不视为影响
二次销售。不适用无理由退货的情形包括：定制商品、鲜活易腐品、已拆封的贴身衣物/化妆品/
隐形眼镜、数字化商品、报纸期刊等。退货流程时限为：商家须在48小时内审核退货申请；
同意后，您需在7日内寄回商品并回填快递单号；商家签收后3个工作日内完成验货并处理退款。
运费方面：选择平台“上门取件”，运费从退款中扣除，无需当场支付；请勿使用到付件，
商家有权拒收。如涉及质量问题，请按“三包”规定处理（与七天无理由退货不同）。

--------------------------------------------------------
来源：售后政策.md、促销活动规则.md、退换货流程.md、52644
--------------------------------------------------------
问题：订单 52644 到哪了，顺便问下你们的退货政策是什么
路由：order、knowledge　来源 llm　置信度 0.95
      理由：用户同时询问具体订单物流状态和通用退货政策，属于两个不相关需求
  · 第1轮 知识库：成功 8060ms token={'in': 0, 'out': 0}
      结论：关于订单 52644 的物流状态，**提供的资料中没有相关内容**，无法查询具体
            进展。建议您登录账户在“我的订单”中查看实时物流信息，或联系客服提供订单号
            进一步协助。关于退货政策，核心内容如下：- **七天无理由退货**：自签收次日…
  · 第1轮 订单：成功 12ms token={'in': 0, 'out': 0}
      结论：订单 52644 的物流：目的地 San Salvador，当前「Late delivery」，
            最近更新 2017-02-12 11:11。
决策：answer（2 个单元给出结论，汇总方式 llm）
--------------------------------------------------------
耗时 16335 ms　token {'in': 839, 'out': 320}
```

（上面是 **2026-10-11 真跑出来的一次**，长句为排版折了行，其余逐字照抄。）

这一段最值得看的是那个**冲突**：知识库单元老老实实说"提供的资料里没有订单 52644 的
物流"，订单单元给出的是"目的地 San Salvador，Late delivery"。同一句话里，一个说没有、
一个说有——而这**恰恰是对的**：知识库检索的是售后文档，里面当然不会有一笔具体订单的
物流；orders 表里才有。两个单元各说各的边界，汇总层再把它们拼成一句人话。

所以这里**没有**做"冲突检测"把其中一个否掉。如果系统看到两个结论不一致就去挑一个，
它挑掉的很可能正是那个对的部分——它没有能力判断"哪边在什么范畴里说话"。真实系统里
该做的是把来源标清楚（下面 `来源：` 那行），让人自己看得到分歧在哪。

这一条输出里藏着这个项目的大部分设计：路由是谁定的（`来源 llm`）、两个 worker 花了多久（**8.3 秒和 11 毫秒，差三个数量级**）、哪块坏了怎么如实告诉用户、汇总走的是模型还是直接拼接（`汇总方式 llm` / `concat`），以及**最后那句"二者存在冲突"**。

最后那句值得单独看一眼：知识库那一侧如实说了"我的资料里没有这个订单号"，订单那一侧给出了物流。汇总节点**没有拿一方的结论去覆盖另一方**，而是把冲突原样报出来。这是"汇总提示词第 2 条：一个字的新事实都不许加"的直接结果——包括**不许替用户消解矛盾**。

**这是真跑出来的，不是摆拍的**（2026-10-10）。同一天更早的一次运行里 dashscope 不通，那次的结果是：路由走规则兜底（8 秒放弃，不是 180 秒）、知识库 worker 超时、订单那半照常答出来、并且明说哪块没拿到——一次请求把三四条兜底链路全演示了。

---

## 架构

```
                     ┌────────────────────────────────────┐
                     │                                    │ decision == "re_route"
                     ▼                                    │
 START ──────► router ──┬──► worker_knowledge ──┐         │
                       ├──► worker_order     ──┼──► aggregate ──┬──► END（answer）
                       └──► worker_analytics ──┘               │
                                                               └──► END（human / 转人工）
```

### 一、Agent 还是 Tool？判据只有一条

**有没有"看一眼结果再决定下一步"的循环。**

- **订单查询是工具**：输入订单号，输出状态。中间没有"看看结果再改改"。给它套一个 agent 壳子，只会多一层提示词、多一次模型调用，换来一个更不确定的输出。
- **数据分析是 agent**：它要生成代码、执行、看报错、改代码、再执行。这个循环就是它的全部价值。

这条判据决定了代码长什么样：[ecom/orchestrator/tools/order_tools.py](ecom/orchestrator/tools/order_tools.py) 里是纯函数，[ecom/analytics/](ecom/analytics/) 里是带纠错循环的图。**把工具包装成 agent 是负收益**。

### 二、Worker 之间不传自然语言，只传结构化结果

`results` 里存的不是"订单 agent 说了一段话"，而是：

```python
{"worker": "order", "ok": True, "answer": "...", "data": {...},
 "sources": [...], "insufficient": False, "error": "", "elapsed_ms": 7, "tokens": {...}}
```

理由有三条，第一条最要紧：

1. **自然语言消息要再解析一遍**，等于把"理解"这件事做了两次，每次都可能出错，而且出错时分不清是"上游没说清"还是"下游理解错"。
2. 结构化结果能被程序直接消费——汇总按字段合并、trace 按字段聚合，不用再写一层"从散文里抠信息"的代码。
3. **上下文隔离的落点就在这**：每个 worker 拿到的是 `question` 和它自己需要的那点东西，**不是完整对话历史**。`chat_history` 是**节点输入**——由调用方给一份、图从不往里写；读它的是 router（指代消解）和各 worker，**汇总节点刻意不读**（那一步只做"各说各的、不替谁判对错"，给它历史等于给它素材去改写和脑补）。

### 三、`ok=False` 和 `insufficient=True` 是两件事

这是全项目最重要的一条区分，也是最容易混的：

| | 含义 | 汇总怎么处理 |
|---|---|---|
| `ok=False` | **坏了**（embedding 服务挂、子进程超时、库连不上） | 告诉用户这块暂时不可用；**不重路由** |
| `insufficient=True` | **看过了，确实没有** | 换一个 worker 再试一轮 |

把两者混在一起，监控上就看不出"业务规则拦下了多少"和"系统坏了几次"；行为上则会变成"服务一抖动就换个 worker 再抖一次"。

同理，"订单号查不到""这个状态不能退货"是**业务结论**，以 `ok=True` 返回——它们不是故障。

### 四、回环是有界的，而且这件事由代码保证

`aggregate` 只在 `round < MAX_ROUNDS` 时才给出 `re_route`，`tried` 里已经试过的 worker 会被路由排除。两条一起，环才有出口。

**一个没有出口的回环是这套架构最容易出的 bug**（"没答上来 → 重新路由 → 还是它"）。所以它不能靠提示词里写一句"不要重复路由"，只能靠代码。另外 `app.invoke(..., recursion_limit=...)` 还留着 LangGraph 自带的兜底——编译器不知道我们的环有界，真出意外时宁可由它抛错，也不要空转到天荒地老。

### 五、汇总不引入新事实；只有一个结论时**不调模型**

汇总的提示词第 2 条是硬要求：**一个字的新事实都不许加**。但更彻底的做法是——`len(usable) == 1` 时根本不用模型：没有东西可合并的时候调模型，只会把一段本来忠实的话改写成一段可能不忠实的另一段话，**凭空多一次幻觉机会、换来零信息增益**。

---

## 三个子系统，三种接入方式

合并之后三个子系统在同一个包里，但接入方式**故意不一样**。三种接法各自算的账不同，不是不一致：

| 子系统 | 接入 | 为什么 |
|---|---|---|
| 知识库 | **进程内 import** | 热路径（几乎每次提问都走）、无副作用的纯库 |
| 订单 | **本进程 + SQLite** | 毫秒级、纯函数、不调模型 |
| 数据分析 | **子进程** | 唯一执行模型生成代码的，需要一个能被真杀死的边界 |

### 知识库：进程内

子进程要多付一次 Python 启动 + chromadb 初始化的钱，而它几乎每次提问都走。它是个纯库：`retrieve` / `rerank` / `generate_answer` 都无副作用，给一个进程边界带不来任何隔离收益。

只 import 四个入口，**不 import 它的 `core/graph.py`**：绕开联网搜索那一路依赖，也避开 `answer_node` 里"把问过的题写进待审核队列"的副作用。这个理由和撞名无关，合并前后都成立。

### 语料：为什么是 10 篇法规原文

原来的语料约四分之三是 LangGraph / RAG 的课程笔记，跟"电商客服"这个项目名对不上——评测跑出来的分是"这个系统多会讲 LangGraph"，不是"多会答售后问题"。现在换成 10 篇电商售后法规原文，外加原有的「电商客服示例」7 篇。

**为什么是法规、不是某个平台的帮助中心**：京东 / 拼多多 / 苏宁 / 唯品会的帮助中心页面都是 JS 壳，正文抓不到；就算抓到，那也是有版权的运营文案，不能提交进公开仓库。法规原文依《著作权法》第五条（立法、行政、司法性质的文件不适用著作权法）属公有领域，可以合法再分发——它是**唯一**同时满足"是真实语料"和"能提交进仓库"两个约束的来源。每篇文件头都记了出处 URL、公布/修订时间、抓取日期和这条公有领域依据。

抽取工具是 [ecom/knowledge/cli/fetch_corpus.py](ecom/knowledge/cli/fetch_corpus.py)，只用标准库（宿主 PyPI 不通，且 samr.gov.cn 会让 curl 段错误）。**按文档结构抽，不按 class 选择器**——第一版靠容器选择器，10 篇里有 6 篇只抽出几个字（选择器命中了 CSS / 工具栏里的同名元素）；换成"整页取文本"后在 10 篇上全部胜出，那套容器逻辑整个删了。留下的三条规则都来自法规的固有结构：正文从第一处「第X条」起（锚在「第X章」会让章目录被当成正文）、首条再次出现就截断（政府页面嵌了一份打印/分享用的全文副本）、正文到最后一个句号为止（站点信息不会以句号结尾）。

子目录一出现，顺带逼出两个本来不可能发生的问题，都补了闸门：**孤儿 chunk**（`add_chunks` 的幂等只在文件还在盘上时成立，被删文件的 chunk 谁也动不了、却照样被召回——实测清掉 253 个）和**同名文件**（chunk 的 source 取的是 basename，两个子目录下的同名文件会共用一个 source、互相覆盖而不报错）。

**一个已知的别扭处**：出题时 `build_eval_set.py --sources` 只能按**文件名子串**筛，不能按目录名筛——`--sources 法规原文` 匹配不到任何东西（`source` 存的是 basename，不带目录）。当时给这次出题是列了 9 个文件名 needle 才圈对的。改成存相对路径能解决，但那会让 `eval_set_business.jsonl` 里的旧 source 全部失效、让上面那道同名闸门变成死代码、还会在评测 id 里塞进斜杠——代价大于收益，所以**先记下不改**。

### 数据分析：子进程——**理由变了，这一点值得写清楚**

合并**前**，这里的第一个理由是硬约束：两个仓库都有顶层 `src`，同一进程里必然有一个静默解析错。合并**后**那个理由没了。

**但边界留着，而且是故意的：**

1. **它执行模型生成的代码。** 这是本系统里唯一一处"别人的文本变成我们机器上跑起来的程序"。沙箱在 [ecom/analytics/core/sandbox.py](ecom/analytics/core/sandbox.py)（AST 白名单 + 受限 builtins），但沙箱**不做超时、也不做进程隔离**——`while True: pass` 拦不住；文件 I/O 也只靠一层黑名单（`read_*` / `savefig` 等入口被禁），不是隔离。真进程边界是补上这一层的唯一办法：`subprocess.run(timeout=…)` 到点是**真杀**，而进程内那个"超时"只是不再等它。
2. **它可能跑几分钟。** 生成 → 执行 → 报错 → 改 → 再执行，循环几轮。进程内跑意味着这几分钟里 supervisor 那条线程被占着、还占着内存。
3. 顺带，崩了炸不到 supervisor。

命令从 `python ../sales_agent/analyze_json.py` 变成 `python -m ecom.analytics.cli.analyze`，用 `sys.executable` ——**同一个 venv，不再需要"用哪个解释器"这个配置项**。界面（`app.py`）也走这条边界：**界面不该是那条边界的例外**，为了"少起一个进程"改成进程内调用，等于换了一个没有超时、崩了会带走整个页面的执行环境。

### 订单：原生，SQLite 默认

[ecom/orchestrator/db/](ecom/orchestrator/db/) 是一个薄薄的双后端适配，SQL 全部用 `?` 占位符写，MySQL 后端在 `engine._translate` 里统一换成 `%s`。默认 SQLite 的理由不是"SQLite 更好"，而是**默认路径必须零外部依赖**，否则"别人拿到能跑"就是空话；但代码不能被 SQLite 惯坏，所以 DDL 用两边都吃的写法（`VARCHAR` 做主键而不是 `TEXT`，不用 `AUTOINCREMENT`）。

### MCP：给别的 Agent 的出口

[ecom/mcp/](ecom/mcp/) 把查订单、查退货、检索知识库这三个能力按 MCP 协议暴露出去，**只走 stdio**。它和对话链路是两条平行的路，共用同一批业务函数：

- **不跑 Supervisor**：调用方已经知道要查订单 52644，不该再花一次模型调用做意图识别；
- **不重复实现规则**：两边都调 `tools/order_tools.py` 和知识库检索，状态能不能退只有一份判断；
- **默认只读**：`list_customer_orders`（按客户号查 PII）和 `create_return`（写库）默认**不注册**，要开得设 `MCP_ENABLE_PII=1` / `MCP_ENABLE_WRITES=1`；写操作即使开了也要调用方传 `confirm=true`，演示模式下无论如何都拒绝。

工具清单、返回契约、错误码、客户端配置和安全边界见 [docs/MCP.md](docs/MCP.md)。

### 数据：换成真实数据之后，三件必须解释的事

三个子系统的数据原来都是手写的（分析侧一份 20 行的 `sales.csv`、订单库 20 条常量、评测题的"真值"直接写在代码里）。手写数据的好处是看得懂，坏处是**它只证明了代码能处理我编出来的那种数据**——真实数据长什么样（订单号几位、状态有几种、哪几列根本不存在），手写数据一个都答不上来。

现在分析侧和订单侧都换成 **DataCo 智能供应链**（一个公开数据集，Mendeley/Zenodo 上可下，sha256 记在 `data/analytics/datasets/dataco/NOTES.md`）。同一个源派生出两份东西：

| 消费者 | 派生脚本 | 产物 | 抽样 |
|---|---|---|---|
| 数据分析 | `analytics/cli/derive_dataco.py` | `datasets/dataco/data.csv`（20242 行 / 3.6MB） | 按订单号哈希抽 1/9 |
| 订单库 | `orchestrator/db/derive_seed.py` | `orchestrator/seed/{orders,logistics}.csv`（7.9MB） | 全量，一笔不抽 |

**换成真实数据之后，暴露出来的三件事**——它们不是"新功能"，是原来那份手写数据**藏住了**的问题：

**一、（分析侧）一行不等于一笔订单。** DataCo 一行是一个**商品项**，一笔订单平均 2.8 行。直接 `len(df)` 当订单数会多算近三倍，而且**不报任何错**。所以 `dataco/NOTES.md` 里写死"订单数要按 `Order Id` 去重"，`Sale` 和 `Order Item Total` 两列差约 10%（未打折 vs 折后），默认口径取前者、不许混用。口径不再写在提示词里，而是 **`NOTES.md` 一份文件、一次读取、三处注入**（analyst / self_corrector / reviewer）——因为原来这两处各写了一遍，没有任何机制保证一致。

**二、（订单侧）公开数据里没有承运商、没有运单号，连发货地都没有。** 承运商和运单号是快递公司的私有数据，公开数据集不会有；`Order City` 是**目的地**不是发货地。所以 `logistics` 表这三列一律为空，`_fmt_tracking` 改成"有的才说"，界面也不再渲染「承运商未知」——那一行对**每一笔**订单都成立，满屏的"未知"读起来像系统坏了。

> **这里是有意不做的：不合成假值充数。** 演示里"承运商是中通，运单号 ZT9988776655"比"目的地圣萨尔瓦多"好看得多，但那是编的——数据里没有的东西一旦被补上，下游"不许引用不存在的事实"那道检查也就跟着失效了。留空 + 文档说明，是这里唯一诚实的做法。

**三、订单号是 1~5 位，不是 6 位以上。** 这个位数假设在**两个地方**存在过：`workers/order.py` 的抽取正则和 `core/router.py` 的路由规则。两处一起改才发现——`\b\d{6,}\b` 对真实订单号**一笔都命中不了**（最大 77204），却会把 6 位以上的发票号路由成"查订单"。**两处各自看都对，合起来就是"路由认得出、抽取认不出"**，这类分叉不会报错。**修法不是"写进注释互相点名"，是让它只剩一份**：订单号的形状现在只定义在 [`utils/ids.py`](ecom/orchestrator/utils/ids.py)，router 和 order 都 import 它。**能靠"只有一份"消掉的分叉，就不该指望靠注释去对齐**——注释不会在你改另一边的时候变红。（跨轮指代判定同理，只有 `utils/entities.py` 一份。）

代价如实记：**1~4 位的订单号在自然语言里认不出来**。门槛定在 5 位是因为放宽到 4 位会让「2025 年买的」里的 2025 被判成订单号，而认错一个订单号比认不出更糟（见下面「明确不做」）。

还有一条是**测试**上的：分析侧评测题的"真值"用**标准库 `csv` 独立算**，不许 import pandas。用 pandas 算真值就是拿被考察的实现去考察它自己——`tests/analytics/test_eval_truth.py` 用 `ast` 守着一张白名单，就是为了让这件事不靠自觉。

---

## 配置：共享的和各自的，分两层

```
ecom/settings.py                ROOT / DATA_DIR / LOGS_DIR / 密钥 / 模型名
                                ← **唯一一处** load_dotenv，**唯一一处**算项目根
ecom/knowledge/settings.py      自己的旋钮：chunk 大小、top-k、RRF 权重、评测路径
ecom/analytics/settings.py      自己的旋钮：LLM 供应商表、MySQL 连接、输出目录
ecom/orchestrator/settings.py   自己的旋钮：worker 超时、MAX_ROUNDS、路由预算
```

"一个项目一个 settings"听起来更整齐，但代价是把三套旋钮塞进一个 200 行文件、再改掉所有 import——**整齐是给读者看的，风险是自己的**。

而密钥和模型名**只能有一处**，否则迟早出现"同一个 key 在两处配、改了一处"，症状是"某个 worker 偶尔 403"——那种错查起来最费时间。

### 根路径只算一次——这条是硬约束

三个子系统原来各有一处 `Path(__file__).resolve().parent.parent`（还有一个 `parent.parent.parent`）。搬进子目录之后它们**全都还能算出结果、只是算错**：不抛异常，只是此后 `data/` 找不到、日志写到 `ecom/logs/`。

三条独立的路径算术各自错一次、且都不报错，靠"记得改"是守不住的，**只能靠"只有一处算路径"**。`tests/test_package_layout.py` 里有一条断言把这件事钉死。

### 依赖表：按"能力"分组，不按"来源"分组

规则只有一条：**内核依赖 = 让 `uv run python main.py` 能跑通的全部**。

- **内核**：langgraph / langchain-openai / langgraph-checkpoint-sqlite / openai / python-dotenv / chromadb / langchain-text-splitters / jieba / pypdf / ddgs / grandalf / requests / **pandas / numpy / matplotlib**。

  最后三个是**核心**不是可选：数据分析是三个核心能力之一，没有它整张图少一条腿。合并前它们在另一个仓库的 `requirements.txt` 里——那只是历史，不是理由。
- **`[ui]` / `[api]`**：两种界面，**都不装也完全能用**。"clone 下来能不能立刻跑"只该取决于内核，多一个装不上的界面依赖不该拖垮整个演示。
- **`[milvus]` / `[mysql]` / `[search]`**：可选后端与增强。默认路径（Chroma / SQLite / ddgs）一律零外部依赖。
- **删掉**：`sales_agent/requirements.txt`（并入）、`langchain-community` 与 `langchain-core`（全仓库零引用）。
- **降级**：`streamlit` 从另一个仓库的硬依赖降成 `[ui]`——它的 `app.py` 已被统一的 `app.py` 取代，而 CLI 才是始终可用的那条路。依赖表里每一次升降都要有这种理由，不然它就是在随机漂移。

**`uv.lock` 现在有了**，在容器里生成（宿主 PyPI 不通，`docker build` 里通），`Dockerfile` 的 `uv sync` 也改成了 **`--frozen`**。

这一段以前写的是「没有 `uv.lock`：宿主上 PyPI 不通，生成不了」，并特意留了一句「要动就单独动，不该混在这次合并里」——所以它是**单独一次提交**做的，没有混在合并里。当时的顾虑是"一份**过期的锁**比没有锁更坏，它会让人以为依赖是钉死的、实际不是"，那个顾虑今天依然成立，只是现在能真生成了。

两个细节：

- **`--frozen` 不是可选的**。只提交锁文件、构建时仍走 `uv sync`（不带 `--frozen`），锁就只是个装饰：`uv` 照样会按 `pyproject.toml` 重新解析。**"有锁"和"用锁"是两件事**，验证的时候两条都验过（见「验证到哪一步」）。
- **锁现在有两道闸了**。原来只有构建里那道，注释里写的还是"这个仓库没有 CI"——那句曾经是对的，但加 CI 之后就变成了错的：CI 里跑的也是 `uv sync --all-extras --frozen`，所以锁文件一漂移，**构建和 CI 会同时在装依赖那一步红**。

---

## 踩过的坑

这一节是这个项目里最值钱的部分。八个都是**真发生过的**，而且每一个的症状和原因都隔得很远。

### 1. 顶层包撞名——报的错和真实原因毫无关系

见上面「这个仓库是怎么来的」。合并前的应对是改名 + 运行时闸门；合并后闸门删了，**接过这件事的是两条测试**：

- 全仓库没有 `from src.` / `config.` 残留（**包括藏在字符串里的** `importlib.import_module("src.…")`——那种 `from src.` 的 sed 完全看不见，合并时真的踩到过，红了三条测试）；
- 顶层只有一个 `ecom/` 包。

后者防的不是老问题以老样子回来，而是它以**"再加个 `scripts/` 放几个脚本""再加个 `utils/` 放点公共代码"**的样子回来——那种拆分一旦开始，下一个撞名只是时间问题。

### 2. 重试放大——两层重试把超时预算吃掉了

路由的正常耗时是零点几秒。某次网络不通时，一次请求的**路由环节拖了 25 秒**才落到规则兜底。

原因是两层重试叠加：openai 客户端默认自带 2 次重试，我们外面又有一层"失败重试 3 次"，**总耗时是乘法**——单次超时 8 秒 × SDK 自己试 3 次 = 24 秒，外面那个"20 秒总预算"直接失效。

- **改法**：`get_llm(..., max_retries=0)`，把重试权收回到我们的代码里。理由是**我们比 SDK 多知道两件事**：这次调用值不值得重试（路由是幂等的分类，值得；写退货绝不能重试），以及整件事的总预算。
- 同时把路由的单次超时压到 8 秒，并加了**总预算**这道闸。

### 3. 「退货政策」被当成「我要退货」，**真的往库里插了一条申请**

一句 `订单 52644 到哪了，顺便问下你们的退货政策是什么`，因为含"退货"两个字，被规则判成了 `create_return`，**系统替用户提交了退货申请**。

这是这个 worker 能犯的最严重的错——用户只是问规则，系统却办了件事。

- **改法**：`create_return` 只在**第一人称诉求**（我要退 / 帮我退 / 申请退货…）时才触发；只要句子里是问规则的口吻（政策 / 能不能退 / 怎么退 / 多久）且没有第一人称诉求，就**不认成要办退货**。
- **加了两道闸**：规则层一道，`run_order` 里再一道。因为规则层只挡规则路径，**模型也有可能把"你们的退货政策是什么"听成"要退货"**——写操作要两层都守。
- 挡下来的代价是多问一句"您是要办还是问规则"；放过去的代价是替用户办了他没要求的事。**代价不对称，所以宁可多问。**

### 4. `results` 是跨轮累加的，`= []` 清不掉

`results` 用了 `Annotated[list, operator.add]`——并行 worker 各自 append 不丢数据，这是 reducer 的语义。代价是**它跨轮只增不减**：往一个 add-型通道写 `[]` 等于没写。

结果就是第二轮的汇总把第一轮的结论又汇总了一遍，症状是"同一段话说两遍"，而且**只在第二轮出现**，最难查。

- **改法**：每条 worker 结果盖上 `round` 标记，汇总前用 `base.round_results` 挑出本轮的。
- 这件事没有任何办法靠"赋值清空"绕开，只能理解 reducer 在做什么。

### 5. 合并之后才发现的：三处"数上层目录"（见上面「根路径只算一次」）

搬完 import 全绿、测试全绿，但 `data/` 找不到、日志要写到 `ecom/logs/` 去——**三处都还能算出结果，只是算错**。这类错误不会被任何业务测试撞见，只能显式钉住。

### 6. 合并之后才发现的：`.env` 本身没合并

依赖表合了、数据合了、配置项删了，但**没人去合 `.env` 的内容**。合并后的 `.env` 里只有 `QWEN_*` 三个变量，而新进来的两个子系统要的 `DEEPSEEK_*`（数据分析）和 `SILICONFLOW_API_KEY`（向量化 + rerank）都不在——也就是说**界面能开，一问就报"未配置 API Key"，三分之二的功能跑不起来**。

发现它的方式很直接：跑一次数据分析，返回 `未配置 deepseek 的 API Key`。**这类缺口只有"真的跑一遍"才会露出来**，静态检查再多也看不见。修法是把两个 key 从原仓库的 `.env` 原样搬过来（值未改动、未入库）。

### 7. 容器里的中文图表：全绿，但图是豆腐块

数据分析的图里全是中文（标题、轴标签、地区名），而 `python:3.11-slim` 里**一个 CJK 字体都没有**。原来 `sandbox.py` 里写死了：

```python
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
```

宿主上有这个字体，所以一直没暴露。容器里没有，matplotlib 的处理方式是**往 stderr 打一行 warning，然后悄悄退回默认的 DejaVu Sans**：

```
findfont: Generic family 'sans-serif' not found because none of the
following families were found: Microsoft YaHei
```

于是那次运行的结果是：子进程 exit 0、`chart_generated: true`、报告文字完全正常、日志里两行信息都在——**每一项检查都是绿的，只有把 PNG 打开看才知道五个地区名全是方框**。这一条是"跑起来了"和"跑对了"之间的距离，靠日志和退出码都发现不了。

修法必须**两边一起改**，只改一边等于没改：

- `ecom/analytics/core/sandbox.py` 给出跨平台的候选顺序（Windows 仍然首选雅黑，行为不变；Linux 落到 Noto）；
- `Dockerfile` 里装上 `fonts-noto-cjk`（约 90MB）——列表里写着但系统上没有，等于没写。

验证方式也只能是**把图画出来看**：容器内五个地区名、标题、两个轴标签全部正常渲染，stderr 里那行 `findfont` warning 也消失了。

### 8. 非 ASCII 的 `API_KEY`：**配了等于没配**

`api.py` 的 `X-API-Key` 是这么比的：

```python
secrets.compare_digest(provided.encode("utf-8"), API_KEY.encode("utf-8"))
```

写出来很对，但这里有**两个坑，症状还是相反方向的**：

- **`compare_digest` 只吃 bytes。** 直接传 `str` 的话，ASCII 情况下它**能跑**，一碰到非 ASCII（"中文口令"是很自然会被设出来的东西）就抛 `TypeError` ——于是鉴权**从"比较"变成"崩溃"**，而且只在有人设了中文 key 的时候才发生。
- **HTTP 头只能装 ASCII。** 非 ASCII 的 `API_KEY` **任何客户端都发不出来**（httpx 会抛 `UnicodeEncodeError`），所以服务端永远收不到正确的 key，每个请求 401。**这不是"鉴权太严"，是"这条配置根本没法用"。**

两个都不是想出来的，是测试跑出来的。修法不只是 `.encode()`：import 时加一条 **WARNING** 把"这个 key 发不出去"明说出来，而不是让它静静地 401 到底。

### 9. 假 streamlit 装进 `sys.modules` 之后不清理——报的错和界面毫无关系

`test_app_smoke.py` / `test_admin_smoke.py` 会往 `sys.modules` 里塞一个假的 `streamlit`。**它不清理**（照着"这可能是本进程唯一的 streamlit"写的），单跑没问题。

但 pytest 的**收集阶段会把所有测试模块先 import 一遍**。`test_app_real_streamlit.py` 模块级有一句 `pytest.importorskip("streamlit")`，它在**收集时**就把**真的** streamlit 连子模块一起装进了 `sys.modules`。之后假模块装上时**只覆盖了顶层那一个键**，真的 `streamlit.delta_generator_singletons` 还留在里面。

结果是**只有跑全套时才炸**（单跑这个文件全绿）的六条 ERROR：

```
streamlit/__init__.py:95 in <module>
    DeltaGeneratorSingleton ...
```

——一句和"界面测试"毫无关系的话。修法是那个 helper 必须连 `streamlit.*` 一起摘掉（`_force_real_streamlit()`），判据是 `__file__`：真的模块有、`types.ModuleType()` 造出来的假货没有。

**这一类"只在全套里红、单跑全绿"的错，症状永远指不到原因上**，只能靠"collection 会 import 所有模块"这条 pytest 的机制去反推。

---

## 界面

```bash
uv sync --extra ui
uv run streamlit run app.py
```

**主区只有对话**，另外三块收在侧边栏的「直连视图」里 —— 之所以是这个形状，是因为这四个东西里只有一个是"来回说话"：**对话**（答复 + 决策链，就是 `main.py -v` 那份内容的可视化；汇总那一步逐 token 流出来）、**订单**（订单详情 + 物流 + 退货申请）、**数据看板**（报告 + 生成的图）、**知识库**（答复 + 引用到的段落，逐 token 流出来）。对话只有"多个结论要汇总"时是真流；只有一个结论时整段到达——不把已完成的字符串切块假装在流。把它们并排成四个 Tab，等于说这四件事是同一层级的；收起来之后，页面本身就在说"提问是主线，另外三个是随时可以拐进去看一眼的地方"。

四个视图仍然是**一件事的四个面**，共用一份进程、一份配置：点进订单视图看到的就是提示词里那个 `query_order`，不是另一个项目。换个视图只是换一个渲染函数，**不重启、不换进程**。

**对话视图还支持上传一张图**（购物截图 / 售后凭证 / 发票 / 快递面单）。图片不会直接喂给主链路：先经 `image_understander` 提取订单号 / 金额 / 商品 / 诉求，拼成一段带标记的文本，再走原来的 `answer()`。识别失败或图里没有订单号只会降级成"请手动输入"，不会崩。边界写死：**图片里的订单号只能查，不能用来提交退货**——退货必须在当前消息里手动输入订单号。配置和失败路径见 [技术文档 §11](docs/技术文档.md#十一多模态图片输入)。

主题（浅色 / 深色两套）在仓库根的 `.streamlit/config.toml` 里，两个入口共用一份。**右上角「⋮」→ Settings → Theme** 可以直接切 Light / Dark / System —— 不需要任何额外控件，这是 Streamlit 自己的设置菜单。`.streamlit/` 既不在 `.gitignore` 也不在 `.dockerignore`，而镜像是 `COPY . /app`，所以宿主和容器里用的是**同一个文件**。

### React 前端（可选）

`frontend/` 是第二个浏览器界面：**一个干净的流式对话页**，显示路由结果、来源、
token 用量和健康状态。它不重复 Streamlit 版的图片输入、订单视图和数据分析看板
——那些留在 `app.py`；这一版只把「提问 → 流式回答」这条主线做利落。

```bash
# 开发：Vite 5173 代理到 8000（不给 FastAPI 开 CORS）
cd frontend && npm install && npm run dev     # → http://localhost:5173/ui/

# 生产：构建后由 api.py 挂到 /ui（dist 不存在就跳过，不阻止 API 启动）
npm run build
uv run uvicorn api:app --port 8000            # → http://127.0.0.1:8000/ui/
```

`vite.config.ts` 的 `base: "/ui/"` 必须和 `mount_frontend()` 的挂载路径一致——
不一致的症状是页面能打开但白屏。会话号和 API Key 只放 `sessionStorage`，不进
URL；回答按纯文本渲染，不走 `innerHTML`。细节和限制见
[frontend/README.md](frontend/README.md)。

**但 CLI 才是第一入口，这个顺序不能反**：界面会把"哪一层慢、哪一层挂了"藏起来——超时变成一个转圈动画、异常变成一个红色弹框。调试期用 `main.py -v`，演示时才上界面。

---

## 身份与权限（`AUTH_ENABLED`）

默认关闭：本地 CLI / Streamlit 演示不需要登录。**对公网提供 API 或后台时必须打开**，
否则"会话属于谁"只是一句注释。

打开后是三档有序角色：

| 角色 | 能做什么 |
|---|---|
| `customer` | 只能用自己的 `session_id` 问答；拿别人的会话号一律 404 |
| `agent` | 客服人员（当前与 customer 同权，预留给工单/客户查询） |
| `admin` | 额外可看 `/trace`、在后台审核退货 |

```bash
# token 只在创建时打印一次；服务端只存 SHA-256，不存明文
uv run python -m ecom.auth.cli create --name alice --role admin
# → 打印 p_xxx / alice / admin，以及一行 TOKEN

uv run python -m ecom.auth.cli list
uv run python -m ecom.auth.cli revoke alice
uv run python -m ecom.auth.cli audit --limit 20
```

接口用法：

```bash
curl -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"question":"订单 52644 到哪了"}' \
     http://127.0.0.1:8000/chat
```

- `/chat`、`/chat/stream`：任何有效 token；`X-API-Key` 只作为 bearer 的兼容别名。
- `/trace`：admin 角色；customer 得到 403（不是 404——这里是"有权限要求"，
  而不是隐藏资源存在性）。
- 会话：`sessions.owner_id` 记录 principal；别人的 `session_id` 一律 404。
  老库里 `owner_id=''` 的会话会被第一个拿它的 principal 认领（迁移用）。
- 后台：`admin.py` 改成输入 admin token，审核人自动取 `principal.name`；
  `returns.review_return` 在 `AUTH_ENABLED=1` 时写一条 `return.review` 审计。
- 审计表 `audit_log` 在 `data/orchestrator/auth.db`，和 `sessions.db` 分开：
  一个是"用户说过的话"，一个是"谁做了什么"，保留策略不该共用一个。

**边界（别把它说大）**：

- 没有登录页 / 密码找回 / MFA：token 由管理员生成、手工交付；
- 没有过期时间，只能 `revoke`；轮换 = 新建 + 吊销旧 token；
- 没有组织 / 多租户；`agent` 目前和 `customer` 同权；
- MCP 仍只走 stdio，信任模型是"能拉起本进程的人"，不走这套 HTTP 身份。

---

## 后台（`admin.py`）

```bash
uv run streamlit run admin.py       # → 8502，需要 .env 里的 ADMIN_PASSWORD
```

**它是和 `app.py` 并列的第二个进程，不是它的一个视图**，这条是设计决定不是偷懒。`app.py` 的 docstring 里写着「不做登录 / 权限：这是离线演示，加一套假登录只会让演示多几步点击」——**那句话现在依然成立，因为后台不是它**。把两者合成一个文件的那天，那句注释就得划掉，而 `app.py` 的定位（面向客户、clone 下来点开就能用）会被一个密码框毁掉。

它**吃的是同一份主题**（`.streamlit/config.toml`），但**没有自己的主题开关**——右上角本来就能切，再加一个就是重复入口。`layout` 也仍然是 `"wide"`：订单表要宽度，审核卡片要并排摆按钮，这两个诉求和"对话收成一条窄列"不一样。

三个页面，**只有一个能写**：

| 页面 | 能写吗 | 复用谁 |
|---|---|---|
| **退货审核** | ✅ | `ecom/orchestrator/admin/returns.py`，界面上**不重写任何判断** |
| 订单与物流浏览 | ❌ 只读 | `order_tools.query_order` / `list_customer_orders`（**不另写一份 SQL**） |
| 知识库审核队列 | ✅ | `ecom/knowledge/cli/review.approve_item`——把那个**留了很久没人用**的接缝接上 |

订单页不另写 SQL，否则界面看到的和模型看到的迟早不一样。

**命令行版是同一个实现**，不是两套逻辑：

```bash
uv run python -m ecom.orchestrator.cli.returns stats
uv run python -m ecom.orchestrator.cli.returns list --status 待审核
uv run python -m ecom.orchestrator.cli.returns approve R1 --note "客户提供了照片" --reviewer 张三
```

### "批准退货"到底做了什么（边界写死）

- ✅ 改 `return_requests.status`，并往 `return_reviews` 插一行审核痕迹，**两步在同一个事务里**。
- ❌ **不执行退款**。这个项目没有支付网关可调，**假装退款比不做更坏**。
- ❌ **不改 `orders.status`**。退货通过 ≠ 订单状态变化；而且那个字段是数据源原值，改它就是在编造数据。

### 为什么"模型不能自己批准"

这一路贯穿全项目：**模型能判定，不能拍板**。退货申请只**登记**，结论由人在后台下。知识库那条也是一样的形状（模型判定"这个问题值得沉淀" → 人批准 → 才进向量库）。

**为什么中间必须隔一层**：模型生成的答案一旦直接进向量库，下一轮检索就会把它当"权威资料"捞出来喂给模型——**知识库自污染**，错误会被放大并固化成看起来不容置疑的知识。

一条配套的目录规矩：**`ecom/orchestrator/tools/` 是模型能调到的，`ecom/orchestrator/admin/` 是只有人能下的结论。** 把"批准退货"放进 `order_tools.py` 那种模型可达的模块，等于把"模型可以拍板"写进架构里——哪怕现在没有任何工具去调它。**该用目录结构而不是代码评审来守这条边界。**

---

## 会话持久化（多轮记忆）

同一个 `session_id` 连问两轮，第二轮带得上第一轮的上文。落盘在 `data/orchestrator/sessions.db`（**和订单库分开一个文件**——订单库是"灌一次就不动"的派生数据，会话是"每天都在长"的运行数据）。

**这份历史拿去干什么**（一句话：不是"喂给汇总"，而是"喂给判断"）：**router** 用它做指代消解（「那能退吗」是指哪一笔），**order worker** 用它抽实体，**knowledge worker** 用它生成答案；**汇总节点（`aggregate.py`）刻意不读它**——那一步只做"各说各的、不替谁判对错"。

### 指代消解：只携带订单号，而且读可以、写不可以

判定集中在 [`ecom/orchestrator/utils/entities.py`](ecom/orchestrator/utils/entities.py) 的 `carried_order_id`——router 的规则层和 order worker 走**同一个函数**。为什么要收成一份：这个仓库真出过"两处各写一份、分叉不报错"的事故（订单号位数门槛，见 `utils/ids.py` 的 docstring）。

它是**"携带上一轮提到的实体"，不是往关键词表里加词**——加词会让一个纯政策问题被拉去查订单。三条保守规则，任何一条不满足就**不猜**：只扫用户发言（客服回答里 5 位以上的数字太多）、只看最近 `HISTORY_TURNS` 条、那条消息里有 ≥2 个订单号就放弃。

边界两条，都是有意选的：

- **只有订单号**会被携带（金额、商品名没做——那要的是字段级消解）。
- **读操作可以回退到携带来的号，写操作（`create_return`）不可以**。所以「订单 52644 到哪了」→「那能退吗」会答"如果要为 **订单 52644** 提交退货申请，回一句确认"，**而不会自动提交**；「那我要退货」也不用携带来的号，退回"请提供订单号"。**代价**：明确诉求时用户得再说一次订单号。这是拿"少问一次"换"绝不对错误订单办事"。

它**不解决"聊久了早期信息滑出窗口"**——那是另一件事，见下一节。


### 会话内长期记忆：蒸馏 + 按需召回

`history()` 只带最近 `HISTORY_TURNS` 轮。聊到第 20 轮时，第 3 轮说过的订单号、诉求、结论都已经滑出窗口——模型看不见了，于是会重复问、或者答非所问。这一层把**滑出窗口的那一段**压成一条摘要存进 `memories` 表，需要时**按语义召回**，拼在真实历史**前面**（[`ecom/orchestrator/memory.py`](ecom/orchestrator/memory.py)）。

**存的是"可执行的事实"，不是关键词。** 「超出范围就把前文提炼成关键字另存」这个想法很自然，但它只能告诉你**聊过什么话题**，不能告诉你**结论是什么**——而多轮里真正要继承的恰恰是结论。所以 `summary` 写事实（哪一笔订单、诉求、落到什么结论），`keywords` 只当**索引**：CLI 列表里给人扫一眼，不参与回答。

**是"按需召回"，不是"每轮全灌"。** 全灌 = 每轮把整段记忆塞进 prompt，token 随会话长度线性涨，而且大部分和当前问题无关。召回 = 先按语义挑最相关的 `MEMORY_TOP_K` 条。**不设相似度阈值**：池子是这个会话自己的、条数天然少，加一个没实测过的阈值只会**静默丢掉记忆**。池子为空时**一次 embedding 都不调**——没有记忆的会话在这条路上是零额外开销。

**蒸馏放在 CLI，不放在问答路径上**：

```bash
uv run python -m ecom.orchestrator.cli.memory distill --session <id>   # 或 --all
uv run python -m ecom.orchestrator.cli.memory list    --session <id>   # 沉淀了哪些
uv run python -m ecom.orchestrator.cli.memory recall  --session <id> "问题"   # dry-run
```

放问答路径上意味着**用户每问一句都可能多一次模型调用**，而且是不可控的那种（第一次问就触发全量蒸馏）。收成一条显式命令之后，模型调用只在有人敲它的时候发生，可以挂 cron，也可以在成本紧张时**不跑**。**代价说清：记忆不是实时的**——要有人跑 `distill` 才更新。这和 `purge` 要外挂 cron 是同一个立场：**不为了让数据"自动正确"而把不确定性塞进请求路径**。

`recall` 那条子命令是**必须有**的：召回是个黑盒，没有它，"线上为什么没带记忆"只能靠猜。它是 dry-run，只读、打印池子大小 / 命中条数 / 每条余弦。这是本仓库「宁可暴露」的一贯做法。

**幂等由数据库保证，不是由锁保证。** `session.py` 那把 `_lock` 是**进程内**的，而蒸馏是独立进程的 CLI——cron 撞上手动执行时，两边会读到同一个水位线、蒸馏同一批消息、各插一条。`UNIQUE(session_id, through_message_id)` + `INSERT OR IGNORE` 让第二次成为 no-op。这就是"水位线"（`through_message_id`）的用途：**记住"已经蒸馏到哪条消息"，而不是记住"是哪条消息"**。

边界三条，都是有意选的：

- **记忆不参与写操作授权**，和上一节的"携带的订单号"同一条底线：摘要里的订单号是模型写的，拿它去提交退货等于替用户对一笔可能错的订单办事。所以一个只提「那笔订单」的**追问，在 order worker 那里仍然可能落到"请提供订单号"**——这条路径宁可多问一次。（router 读得到记忆，所以它至少能被路由对。）
- **这张表是会话内的，不是用户画像。** 按 `session_id` 键，跨会话一条都召不回来——理由和下面「不做用户画像」那一条完全相同：先有身份，才谈得上画像。
- **换 embedding 模型会让已存向量失效**，没有版本号落点，也就没有自动重算。记为已知限制。

### 会话操作台

```bash
uv run python -m ecom.orchestrator.cli.session list
uv run python -m ecom.orchestrator.cli.session show <id>
uv run python -m ecom.orchestrator.cli.session delete <id>
uv run python -m ecom.orchestrator.cli.session purge --days 30
```

### 为什么**不**复用 `ask.py` 的 `SqliteSaver`

`ecom/knowledge/cli/ask.py` 已经有一套跑通的落盘会话（`SqliteSaver` + `thread_id`）。看着应该直接搬过来统一，**但不该搬**，三条理由：

1. **`chat_history` 在 supervisor 里是"节点输入"，不是"图状态"。** `answer()` 把它塞进 `make_initial_state`，`CSState.chat_history` 上**没有 reducer**（唯一带 reducer 的是 `results`）。
2. **要让 checkpointer 接管，就得往图里加"累积历史"的逻辑**——而那样做会撞上坑 4 那个陷阱：给 `chat_history` 加 `operator.add` 之后，每次传入的初始值都会被 append 而不是替换，历史里**凭空多出重复轮次，且不报错**。
3. **两个图不是同一个图。** `ask.py` 存的是知识库图的状态，supervisor 图是另一个对象。

对面的好处也说清，否则这个决定没有分量：**统一成一套的好处是"会话"这件事在整个项目里只有一个答案**，新人只要学一次。代价是上面三条。**什么条件下该反过来选**：如果 supervisor 以后要在**图内部**做需要记忆的决策（比如记住"这个用户上一轮已经在等人工了"），那时候就是"历史作为状态"，checkpointer 才是对的。

### 会话 id 是**凭证**，不是账号

> `_guess_customer`（`workers/order.py`）意味着**"客户"是从话里猜出来的、未认证的**。所以拿到 `session_id` 就能读到这个会话的全部历史。**别把它当账号**；要谈归属，前提是先有登录态——那是另一个项目。

推论有三条，全是有意的：

- **默认 30 天过期**（`SESSION_TTL_DAYS`）。无主数据不该无限期留着。
- **界面刻意不把会话号放进 URL**（不做 `st.query_params`）。它是 bearer token，进了 URL 就会进浏览器历史、代理日志、Referer 头。**代价**：新开一个标签页 = 新的一段对话。这条由 `test_a_new_browser_session_starts_a_new_conversation` 钉着——写成测试是因为"没做的能力"最容易被下一个人当成 bug 顺手补上，而补法正好是那个不该做的。
- **侧边栏只显示前 12 位**。侧边栏是最容易被截图发出去的地方。

### 三个旋钮不要合并

| 旋钮 | 管什么 | 性质 |
|---|---|---|
| `HISTORY_TURNS = 6` | 喂给模型多少原文 | **成本** |
| `SESSION_TTL_DAYS = 30` | 留多久 | **合规** |
| `MEMORY_TOP_K = 3` | 带几条记忆进 prompt | **成本（另一条通道）** |

看起来都可以叫"留多少"，但改动的动机完全不同。合并成一个开关，会在有人想省钱的时候顺手把合规期限也改掉。

`HISTORY_TURNS` 和 `MEMORY_TOP_K` 同属成本，但**是两条独立通道**：前者是"最近几轮的原文"，后者是"更早那一段的摘要"。一个调大不意味着另一个也该调大——原文精确但贵，摘要便宜但有损，两者不是同一种东西的多少，**所以不给它们一个共同的倍率**。（还有第四个旋钮 `MEMORY_DISTILL_MAX_MESSAGES = 40`，它管的是一次蒸馏喂给模型多少消息，属于离线成本，更不该混进来。）

### 过期只在两处触发

进程启动时调一次 `purge()`，以及 CLI 显式调用。**不做**懒加载式的"读的时候顺手删"——那会让读操作产生写副作用，出问题时极难定位。

**代价如实说**：长驻进程不会自我清理，需要外挂一行 cron：

```bash
17 4 * * * cd /path/to/ecom_cs && uv run python -m ecom.orchestrator.cli.session purge
```

**蒸馏同样要外挂**（它也不在问答路径上）：跑晚了只是记忆旧一点，跑漏了不会丢消息（水位线没推进，下次接着蒸），所以它不挑时间——但**不能指望它自己发生**。

```bash
23 4 * * * cd /path/to/ecom_cs && uv run python -m ecom.orchestrator.cli.memory distill --all
```

---

## 可观测

- **日志**（`logs/<date>.log`）：给人扫的。
- **trace**（`logs/requests.jsonl`）：一行一条请求，机器可读。记录的是**事实**（路由来源、每个 worker 的 ok / 耗时 / token、最终决策），不是结论——"这次成功了吗"由查询方判断，口径变一次不用重跑历史数据。
- **`GET /trace`**：把最近若干条请求的决策链摊出来。这是给别人看"系统怎么想的"那扇窗。
- **`POST /chat/stream`**：同一次问答的 SSE 版本，事件 `start` / `delta` / `final` / `error`。**`final.answer` 是权威版本**：汇总流到一半失败会退回拼接，前面的 delta 可能和它不一致。见 [技术文档 §12](docs/技术文档.md#十二流式输出)。

`main.py -v` 就是 trace 的人类可读版。

---

## 路由评测

路由以前只有"真跑那次看起来对"，没有可复跑的分数。现在有一份 50 题的人工标注集
（`data/orchestrator/router_eval.jsonl`）和一个离线评测 CLI：

```bash
uv run python -m ecom.orchestrator.cli.eval_router                 # 规则基线，不联网
uv run python -m ecom.orchestrator.cli.eval_router --show-errors   # 列出错题
uv run python -m ecom.orchestrator.cli.eval_router --llm --limit 20  # 真模型（花钱）
```

规则基线的当前结果（2026-10-30，50 题）：

```
exact   41/50（82.0%）　covered 42/50（84.0%）
```

`exact` 是计划集合完全相等；`covered` 是"期望的 worker 全被调到了"（多带不算错）。
错题集中在三类：没有关键词的口语（"我要退货"）、把"发货方式延迟"误判成订单、
以及 8 位发票号被当成订单号。**这些是规则层的边界，不是 bug 清单**——它们正是
`--llm` 模式要覆盖的部分；改题让分数变好不算修好。

---

## 测试

```bash
uv run pytest        # 747 条（--all-extras 实测：745 passed / 2 skipped；条数以当次输出为准）
uv run ruff check .
```

（**这个数要在装了全部 extra 的环境里跑**才算得准。宿主上少几个包时总数会小一截，
少掉的那批挂在 `importorskip` 上；没装 matplotlib 时还会多两条红，见下面。）

`CI`（[.github/workflows/tests.yml](.github/workflows/tests.yml)）上的数字会随测试增减（以当次 run 为准）。
它和本机通常差一条 skip：干净 checkout 上没有 `data/orders.sqlite` 和向量库，界面真渲染那条会 skip。
CI 现在有两个 job：`lint-and-test`（ruff + pytest）和 `frontend`（`npm ci` + `tsc`/`vite build`）——
前端构建不再只靠本机。

**一条测试都不联网。** 不是图快——真调模型意味着测试结果随模型版本、网络、当天负载浮动，那样的测试没法规格回归，它昨天红今天绿，你只会学会忽略它。所有模型调用都用 `FakeLLM` / `ExplodingLLM` 顶掉。

### 那两条红灯是怎么修的（以及它教了什么）

改之前容器里是 516 passed / **2 failed** / 2 skipped。两条都是**真缺陷**：

- `test_review_parser::test_verdict_style_fail` —— 提示词要求「不通过时原因写在第二行」，
  而解析器只取首行冒号后的文本，于是 feedback 退化成「不通过」三个字，
  被喂进自我纠正那一轮等于一句废话。
- `test_sandbox::test_run_code_computes_result` —— **断言值陈旧**：fixture 三行金额
  加起来是 `71988+47992+29995 = 149975`，断言写的是 `131975`。这条在原 `sales_agent`
  仓库里就是这个样子，是搬过来的老账。

第二条值得单独记一笔，因为它**在宿主上伪装成了另一回事**：

| | 宿主（没装 matplotlib） | 容器（装了全部 extra） |
|---|---|---|
| `test_sandbox::test_run_code_computes_result` | `assert "ModuleNotFoundError: No module named 'matplotlib'" is None` | `assert '149975' == '131975'` |

**同一条测试，在两个环境里红的原因不一样。** 宿主那侧是**环境缺口**，它把那条陈旧的
期望值藏在了后面——只看宿主的话，你会以为"装上 matplotlib 就好了"，而装上之后它照样红。
**这就是为什么要用完整环境跑一遍，也是为什么"宿主上红是缺包"这个结论不能直接信。**

（宿主上现在还剩 2 条红，都还是同一个原因：那个 venv 没装 matplotlib，
`run_code` 一进去就报 `ModuleNotFoundError`。**在容器里这两条是绿的。**）

几条值得单独说的：

- `test_falls_back_to_rules_when_the_model_itself_is_broken` —— 三层路由里"规则兜底"存在的全部意义。
- `test_the_whole_graph_still_works_with_the_model_layer_dead` —— 路由和汇总两个模型角色全挂，整条链路依然跑通。
- `test_asking_about_the_policy_does_not_file_a_return` —— 坑 3 的回归测试。
- `test_results_do_not_leak_from_one_round_into_the_next` —— 坑 4 的回归测试。
- `test_every_ecom_import_points_at_a_real_module` —— **不用 import 业务代码**，只用 AST + 文件系统比对，把每条 `from ecom.x.y import z` 解析回真实文件。合并改了 270 处 import，其中约 55 处在**本机跑不起来**的子系统里，这条是它们唯一的保护。
- `test_the_analytics_worker_still_goes_through_a_process_boundary` —— 守着那条边界的替代物：命令行里必须出现 `subprocess`，且不许直接 `import ecom.analytics`。
- `test_the_three_settings_modules_agree_on_the_root` —— 坑 5 的回归测试。
- `tests/test_app_smoke.py` / `tests/test_admin_smoke.py` —— 往 `sys.modules` 塞一个**假的 streamlit**，把界面真跑一遍。理由同上：这两个文件是唯一连 import 都进不去的（没装 `--extra ui`），写错一个函数名跑遍全套测试都不会有人报警。**这份假模块不记录关键字参数**，所以断言一律用位置参数——而"未登录看不到后台"这种**真正的**分权保证，放在一条**不依赖 streamlit** 的测试里（门禁逻辑被抽成了纯函数 `check_password`）。
- `tests/test_app_real_streamlit.py` —— 在**真的 Streamlit 运行时**里（`AppTest`）把 `app.py` / `admin.py` 跑完。补的是假模块够不着的那一段：真版本里函数签名对不对、`st.pills(format_func=...)` 和 `st.status(type="compact")` 这些新调用认不认、`st.stop()` 是不是真的能中断脚本。**它不证明"界面能用"**：没有浏览器、没有 CSS、没有布局；交互路径（打字触发一次真实问答）也没走过——那会调模型。
- `tests/test_api.py` —— 用 `TestClient` 打 HTTP，**monkeypatch 掉 `supervisor.answer`**，所以不联网。覆盖两个鉴权方向、错误体结构、以及"同一个 `session_id` 第二次请求能拿到历史"（会话持久化的真正回归测试）。
- `tests/orchestrator/test_return_review.py` —— 幂等、竞态（`counts[0] == 0`）、以及**两个方向各一条**的重复申请拦截（`已通过` 必须拦、`已拒绝` 必须放）。
- `tests/test_milvus_live.py` —— 唯一连**外部服务**的一条，默认 skip，`ECOM_LIVE_MILVUS=1` 才跑。

---

## 验证到哪一步（如实写）

这台机器 **PyPI 不通**——但 `docker build` 里通。所以最后是用**容器**把宿主上装不上的那几个包补上，把原来最大的一块未验证面关掉了。分清楚哪些是真跑过的：

| 验过？ | 是什么 | 怎么验的 |
|---|---|---|
| ✅ | 知识库那一路**端到端** | 真跑：向量 20 + BM25 20 → RRF 融合取 5 → rerank 取 3 → 带 `[1]` 引用的答复 |
| ✅ | 检索质量（新语料的 152 题） | 真跑：Recall@5 = 119/152 = 78.3%，MRR 0.533（改写型 68.6% / 术语型 98.0%） |
| ✅ | 答案质量（新语料，qwen-plus 自评 40 条） | 真跑：忠实性 117/141 = 83.0%（逐条 85.8%），弃答 9/10（换语料前是 75.3% / 10-10） |
| ✅ | 编排层**端到端** | 真跑：LLM 路由（置信度 0.95）→ 两个 worker 并行 → 汇总，16.3s / 839+320 token |
| ✅ | 订单子系统 | 真跑：65752 笔订单的本地 SQLite（12.8 万行种子灌库 0.87s），单次查询毫秒级 |
| ✅ | 分析子进程的**边界** | 真跑：子进程起得来、失败信息能回到父进程、界面能接住（不带走页面） |
| ✅ | `app.py` / `admin.py` 的渲染代码 | 假 streamlit 下全部渲染通过（`test_app_smoke.py` / `test_admin_smoke.py`，17 + 若干条） |
| ✅ | **真实 Streamlit 运行时** | 真跑：`streamlit.testing.v1.AppTest` 把两个入口跑完，6 条全绿。**这补上了原来那一栏 ❌** |
| ✅ | **合并后的 `API`（FastAPI）真起服务** | 容器内真起 uvicorn：`/health` 200（含 workers + order_db）、`/openapi.json` 元数据齐、空问题 422、`/trace/不存在` 404、真提问 200（257+29 token / 1.2s）。**这一条走了网络、花了 token**，不属于"测试不联网"那一栏 |
| ✅ | **Milvus 那条代码路径** | 真跑：连宿主常驻的 v2.5.5，`ecom_verify_<随机>` 临时集合走完 add / count / search / delete / drop，**零残留**（另有一条专门查残留）。⚠️ 见下面的版本错位 |
| ✅ | 测试与静态检查 | 那一轮是 520 条，容器里 **518 passed / 0 failed / 2 skipped**（原来那 2 条红灯都是真缺陷，已修，见「测试」一节），CI 上是 517 passed / 3 skipped。**后来会话内长期记忆又加了 35 条、图片输入又加了 40 条、流式输出又加了 12 条、L1 演示防线又加了 32 条、沙箱文件 I/O 加固又加了 15 条、MCP 工具层又加了 51 条、React 前端又加了 10 条**，现在宿主 `.venv`（`--all-extras`）上是 **747 条 / 745 passed / 2 skipped**，`ruff` 干净（这批的 GitHub CI 还没跑过；workflow 里已经加了前端构建 job） |
| ✅ | **L1 只读 / 邀请码 / 预算闸门** | 真跑：32 条测试覆盖 fail closed 的邀请码、每客户端限流、跨重连的 token 账本、损坏账本拒绝花钱、路由不选 analytics、退货在落库前被挡、API 429 与 trace 404、界面门禁与建议语。**只验证了代码层**；下面那栏是部署文件 |
| ✅ | **公网演示编排能解析** | 真跑：`docker compose --env-file .env.prod -f docker-compose.prod.yml config --quiet` 退出 0（临时 `.env.prod` 校验后已删）；Caddyfile 用 `caddy:2.10-alpine caddy validate` 通过（健康检查那条 `context canceled` 是校验时没有 app 容器，属预期）。**没有真部署**：没有域名和公网服务器，HTTPS 证书签发、Caddy→app 转发、公网可达都**没验过** |
| ✅ | **CI 本身真跑绿了** | 真跑：[run #38132662970](https://github.com/My16210/ecommerce-multi-agent-cs/actions/runs/38132662970)（2026-10-11）——`ruff: All checks passed!`，`pytest: 517 passed, 3 skipped, 15.06s`，和本机干净 checkout 预演的数字一致。**这一栏原来是 ❌**：工作流写过、命令本地验过，但没推上去就一次都没真跑过。<br>真跑还多抓到两处**本地永远看不出来**的东西：① `python-version-file` 这个输入名**不存在**，setup-uv 只打一条 warning 就忽略它，workflow 照样绿、看着像钉住了版本；② `astral-sh/setup-uv` **没有 `v10` 这个大版本 tag**（它的可移动大版本 tag 只发到 v7），钉 `@v10` 会让 job 在 "Set up job" 阶段就红，得钉精确的 `@v10.3.0`。两处都改成实测过的写法 |
| ✅ | **许可证** | 有了 `LICENSE`（MIT），`pyproject.toml` 里声明了同一个。之前**一个都没有**——公开仓库没有 LICENSE 等于"保留所有权利"，别人不能合法复用 |
| ✅ | **`uv sync` 能否装成** | 真构建：镜像里解析出 **150 个包**并装成，五个 extra 全在（matplotlib / streamlit / pymilvus / pymysql / tavily-python / fastapi 逐个确认过） |
| ✅ | **数据分析跑出一张图** | 真跑（容器内）：DeepSeek 生成代码 → sandbox 执行 → `chart_generated: true`，7.8s，图落在磁盘上。中文标签正常——这一条第一次跑出来是**豆腐块**，见"坑 7" |
| ✅ | 镜像能构建、容器能跑 | 真构建 + 真跑：`docker compose run --rm app python -m ecom.analytics.cli.analyze "…"` 走完全程 |
| ✅ | **依赖可复现性** | 有了 `uv.lock` 且构建走 `--frozen`。**两条都验了**：正常构建通过；往 `pyproject.toml` 里加一个依赖后 `uv sync --frozen` **必须报错**——不报错就说明锁没接进构建 |
| ❌ | **浏览器里的真实界面** | 界面进程没在浏览器里打开过。`AppTest` 拿到的是**组件树，不是页面**：没有 CSS、没有布局、没有交互路径 |
| ❌ | `--profile milvus` 那一支 | 只验到 `docker compose config` 能解析（4 个服务、milvus 发布 19531→19530）。**不是"没做"——是这台机器上 19530 有一个不能动的实例**（宿主那个常年跑着的 Milvus）。代码路径已用宿主那个实例验证过，见上一行 |

---

## 明确不做（边界）

写得和做了什么一样重要。

- **不做多智能体自由对话**（一群 agent 互相聊天、投票、辩论）。那种结构里，一次错误会被下游放大，而且**没法定位是哪一环错了**。这里用结构化契约 + 单向数据流，代价是"不如自由对话灵活"，换来的是每一步都能查。
- **长期记忆：做了"会话"，没做"用户画像"。** 这一条以前写的是「不做长期记忆……要真做，得先想清楚存哪、怎么删、合规怎么办」——现在那三个问题**有答案了**：落盘在 `data/orchestrator/sessions.db`（存哪）、按 30 天过期 + `purge` / `delete` 命令（怎么删）、过期是默认行为而不是可选项（合规）。

  **会话内那层摘要（`memories` 表）同一条边界，也要说死**：它是按 `session_id` 键的，**不是"开始做画像"的先例**。删会话连它一起删（`delete_session` / `purge` 都级联到 `memories`）——删了会话却留着它的摘要，是**隐私上的假动作**：用户以为删干净了，而摘要还躺在库里。

  **仍然不做用户画像**：跨会话的用户级记忆需要"这个人是谁"稳定可信，而这里客户身份是从话里猜的（`_guess_customer`），没有登录态，做出来的画像会挂在错误的身份上。**先有身份，才谈得上画像** —— 这是顺序问题，不是工作量问题。
- **不把 `session_id` 当账号。** 它是 bearer token（拿到就能读到整段历史），所以：界面不把它放进 URL、侧边栏只显示前 12 位、默认 30 天过期。**没做的能力写成测试钉住**（`test_a_new_browser_session_starts_a_new_conversation`），因为最容易被下一个人当成 bug 顺手补上——而补法（`st.query_params`）正好是那个不该做的。
- **不执行退款、不改 `orders.status`。** 后台的"批准退货"只做两件事：改 `return_requests.status` + 插一行 `return_reviews` 痕迹，**在同一个事务里**。没有支付网关可调，**假装退款比不做更坏**；而 `orders.status` 是数据源原值，改它就是在编造数据。
- **不给 `return_requests` 加列，也不引入 schema 迁移机制。** 审核痕迹放**新表** `return_reviews`，用 `CREATE TABLE IF NOT EXISTS` 天然兼容老库。理由两条：这个项目里"删库重灌"一直是免费的（全是派生数据），为了加三列而引入第一个迁移机制，收益和分量不匹配；以及**一旦这套被演示过，库里就会有人下的结论**，那时再重灌就等于抹掉人的决定——**在一个只有派生数据的库上引入"删除即无损失"的隐含假设，是最容易留下的技术债**。
- **不给模型"直接通过退货"的权力**。退货申请只登记不通过，最终决定权在人工审核队列——和这个项目一路的原则一致：模型能判定，但不能拍板。
- **不合成数据里没有的字段**。公开数据集没有承运商、没有运单号、连发货地都没有，这三列就**留空**，回答里"有的才说"。演示效果确实吃亏（"承运商是中通，运单号 ZT9988776655"比"目的地圣萨尔瓦多"好看），但补一个假值会让下游"不许引用不存在的事实"那道检查一起失效——那才是真的亏。
- **不认 1~4 位的订单号**。`_ORDER_ID` 门槛定在 5 位，所以库里查得到的短订单号（DataCo 里最小是 1 位）在对话里**说不出口**。放宽到 4 位就会把「2025 年买的」里的 2025 认成订单号——**认错一个订单号比认不出更糟**：认不出时系统会如实说"没写订单号"，认错时它会去查一个别人根本没提的订单。这条边界写在 `workers/order.py` 的正则注释里，也在 `tests/orchestrator/test_order_worker.py::test_order_ids_shorter_than_five_digits_are_not_recognized` 里钉着，**不是漏做**。
- **不做 monorepo 多包**（每个子系统单独一个 `pyproject` + 单独发布）。这个规模下多包只增加 `uv sync` 的复杂度和心智负担，换不来任何东西。
- **不为"合并"新增功能**。除了 `app.py` 和几个接缝文件，其余是搬 + 改名，不确定的行为一律不动。**唯一的例外是"坑 7"那处中文字体列表**：它不是新功能，是部署验证时真跑出来的缺陷（图能生成、退出码是 0、中文全是豆腐块），改法是把手写的单个字体名换成跨平台候选；Windows 上仍是雅黑优先，行为不变。
- **不为了"数字好看"去动评测集**。金标准、评测集、答案快照是评测结论的依据，不能因为分数不理想就删题、改题。**唯一的例外是语料整体换掉**：出题所依据的那三份课程笔记从仓库删了，题和答案同源的合成集、认不得新语料的不可答题、指向旧 chunk 的金标准就都失去了对象，只能跟着重建——这不是"改历史"，是"换了一份历史"。（`gold_set.jsonl` 那 50 条当时全是未标注的 `pending`，没有人工劳动作废。）

## 已知限制

- **进程内 worker 的"超时"只是不再等它**。Python 杀不掉正在跑的线程，被放弃的线程会自己跑完再被回收。它保护的是响应时间，**不是资源占用**。真要让进程内 worker 也能强杀，得把它们也挪进子进程——当前规模下不值得，但别把这件事说成"所有 worker 都能超时中断"。
- **`KB_MIN_SCORE = 0.5` 是启发式，不是判据——而且换语料后它更弱了**。当初标定它依据的是旧语料的评测：真·不可答的余弦 0.53~0.64，可答的在 0.66 以上，两段只是轻微重叠。换成法规原文后重新实测（同一套 40 条快照）：**可答 0.523~0.781、不可答 0.431~0.707**，30 条可答题里有 15 条落在 0.66 以下，10 条不可答题里有 2 条在 0.66 以上——两段基本糊在一起，**没有任何阈值能把它们分开**。原因是语料变同质了：十篇都是法律中文，问任何法律问题都能捞到一段"看着挺像"的条文（旧语料是 LangGraph / RAG / RAGFlow 三份相对独立的文档，所以拉得开）。**但这不构成把阈值调高的理由**：调高会一次性误伤那 15 条可答题。取 0.5 依旧是"宁可漏报"——它现在只挡得住 0.431 那一条，等于这道闸门在新语料上基本常开。这是语料同质化的代价，如实记下，不拿它去反推一个更"好看"的阈值。
- **`_guess_customer` 是从自然语言里猜客户号的 demo 将就**。真系统里客户身份来自登录态，绝不从话里猜——那等于让任何人问出别人的订单。这一点在代码里也标了警告。
- **一条依赖装不上就少一条腿**。数据分析是核心能力，但它依赖 matplotlib 出图——缺了它整个 worker 会走"自动重试 3 次后放弃"的路径，报告里说"分析未能完成"，而不是崩掉。这是设计里的降级，但它意味着**环境不全时这个项目看起来是"三分之二可用"的**。
- **会话过期要有外挂的 cron，长驻进程不会自我清理。** 过期只在两处触发：进程启动时一次、CLI 显式调用。**不做**懒加载式的"读的时候顺手删"——那会让读操作产生写副作用，出问题时极难定位。这个代价是**换来的**，不是漏掉的。同一台机器跑三个副本的话，各自清各自的。
- **跨轮指代消解只做"订单号"这一种实体，而且只在指代词出现时才触发。** 这条原来叫"路由不解析代词回指"（真跑 `/chat` 时发现的，不是推测）：带历史追问「那它的金额是多少」，历史**确实进了 prompt**（token 从 257 涨到 387），但路由第一层是关键词规则，`plan` 是空的，于是答了"没听明白"。**现在修了**，判定集中在 [`utils/entities.py`](ecom/orchestrator/utils/entities.py)，router 的规则层和 order worker 走同一个函数（细节见上面「会话持久化」一节）。三条边界，都是有意的：**（1）只有订单号**会被携带，金额 / 商品名没做；**（2）读操作可以回退到携带来的号，写操作 `create_return` 不可以**（所以它永远**不会**替用户对一笔猜出来的订单提交退货）；**（3）携带只覆盖最近 `HISTORY_TURNS` 条用户发言**。它**不解决**"聊久了早期信息滑出窗口"——那是会话内长期记忆，见上面「会话内长期记忆：蒸馏 + 按需召回」一节。
- **Milvus 客户端与服务端版本错位**。本机常驻的服务端是 **v2.5.5**，而 `pyproject.toml` 把 pymilvus pin 在 `>=2.6,<3`、compose 里那个服务是 `milvusdb/milvus:v2.6.24`。实测客户端 2.6 连 2.5.5 服务端这几个接口都能用（验证矩阵里那一行就是跑通的证据），但**这条对应关系没有保证**。真要长期用，服务端该换到 2.6.x 对齐 pin。
- **`admin.py` 默认仍是演示级鉴权**。`AUTH_ENABLED=0` 时是一个明文口令，没有用户、没有角色、没有审计（"审核人"是手填字符串）；`AUTH_ENABLED=1` 时改成 admin token + `audit_log`，但仍没有登录页、MFA、token 过期。别把它说成"做了完整权限系统"。第二层是 compose 的 `127.0.0.1` 绑定。
- **两个界面现在都只绑 `127.0.0.1`**（原来只有 admin 绑，app 发布在 `0.0.0.0` 且无任何鉴权）。app 那条没有口令那一层，所以绑定是它**唯一**的一层。要跨机器演示就 `APP_BIND=0.0.0.0 docker compose up -d`——那一刻是你**自己**决定暴露它，而不是默认敞着。

> **上面这些"已知限制"是散落的。** 如果想知道**它们合起来意味着什么、真上线会先坏在哪里**，
> 看 **[docs/上线清单.md](docs/上线清单.md)**——它按"出事顺序"重排了一遍，并且专门有一节讲
> **哪些是故意不做的、为什么现在补它是错的**。
