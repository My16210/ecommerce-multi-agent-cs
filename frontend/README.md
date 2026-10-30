# React 前端

同一套编排逻辑的第二个浏览器界面。Streamlit 那版是"演示 + 侧边栏直连视图"，
这一版只做一件事：**一个干净的流式对话界面**，顺带把 `/health`、路由结果、
来源和 token 用量显示出来。

> 它不替代 `app.py`。两个界面都调 `api.py` 背后的同一套
> `supervisor.answer_stream()`——不是两份实现。Streamlit 版有图片输入和
> 数据分析看板；这一版**刻意不重复**那些能力。

## 开发

```bash
# 终端 1：接口
uv run uvicorn api:app --reload --port 8000

# 终端 2：前端
cd frontend
npm install
npm run dev          # → http://localhost:5173/ui/
```

开发服务器把 `/health` / `/chat` / `/trace` 代理到 `127.0.0.1:8000`
（见 `vite.config.ts`）。**没有给 FastAPI 加 CORS**：代理让浏览器始终看到同一个
origin，开发和生产的信任边界一致；为了开发环境去放宽生产接口的 CORS，方向反了。
接口不在默认地址时用 `VITE_API_TARGET=http://host:port npm run dev`。

## 构建

```bash
npm run build        # tsc --noEmit + vite build → frontend/dist/
```

`api.py` 启动时会检查 `frontend/dist/index.html`：存在就把整个目录挂到 `/ui`
（`mount_frontend()`），不存在就跳过并打一条 info 日志。所以：

```bash
npm run build
uv run uvicorn api:app --port 8000
# → http://127.0.0.1:8000/ui/
```

`vite.config.ts` 的 `base: "/ui/"` 必须和挂载路径一致。不一致的症状是
**页面能打开、但白屏**——静态资源请求的是 `/assets/...` 而实际在
`/ui/assets/...`，只有浏览器控制台里能看到 404。

Docker 镜像会自己跑这一套（`Dockerfile` 的 `frontend` 阶段），所以
`docker compose build` 不要求宿主机先 `npm run build`；直接在宿主机跑
`uvicorn` 时才需要上面这一步。

## 安全约定

- **API Key / 邀请码只放 `sessionStorage`**，关掉标签页就没；不进 URL、
  不写 `localStorage`。会话号同理——它是凭证，不是账号。
- 界面里**没有** trace 查看器：`/trace` 会返回所有人问过的话和内部耗时，
  它是排障窗口，不是给演示观众看的。演示模式下服务端也会返回 404。
- 回答按纯文本渲染（`white-space: pre-wrap`），不解析 Markdown / HTML。
  模型输出直接进 `innerHTML` 是这类界面最常见的一个洞。
- 前端构建产物里**没有任何密钥**；`VITE_*` 变量会被打进 bundle，别往里放
  secret（本项目只用了 `VITE_API_TARGET`，而且只在开发服务器里生效）。

## 目录

```
frontend/
├── index.html
├── vite.config.ts        base=/ui/ + 开发代理
├── src/
│   ├── main.tsx
│   ├── App.tsx           会话状态、SSE 消费、设置面板
│   ├── api.ts            fetch + SSE 解析 + 错误体
│   ├── types.ts          与 api.py 的 ChatResponse / SSE 事件对齐
│   └── styles.css
└── README.md
```

## 已知限制

- 没有自动化浏览器测试（本仓库的测试原则是离线、不联网；Playwright 需要
  下载浏览器，所以只做了构建和静态约定检查 `tests/test_frontend.py`）。
- 没有图片输入 / 数据分析看板 / 后台入口——那些在 Streamlit 版里。
- 会话号不放进 URL，所以**新开一个标签页就是新对话**。这是有意的，见根
  README「会话 id 是凭证，不是账号」。
