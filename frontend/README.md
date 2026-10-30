# React 前端

同一套编排逻辑的第二个浏览器界面。Streamlit 那版是"演示 + 侧边栏直连视图"，
这一版做三件事：**干净的流式对话**、**登录后回看历史**、**转人工（客服台）**，
顺带把 `/health`、路由结果、来源和 token 用量显示出来。

界面形状参考 DeepSeek Harness 的三栏工作台：左侧会话列表（可折叠）、中间对话、
右侧可折叠的「会话轨迹」。底部输入框随内容增高；主题有浅色 / 深色 / 跟随系统
三档。匿名模式下会话列表按标签页存在 `sessionStorage`，刷新不丢、关掉就没；
登录后（`AUTH_ENABLED=1`）改为从服务端恢复该账号的历史。

> 它不替代 `app.py`。两个界面都调 `api.py` 背后的同一套
> `supervisor.answer_stream()`——不是两份实现。Streamlit 版有图片输入和
> 数据分析看板；这一版**刻意不重复**那些能力。

## 流式与思考过程

`App.tsx` 的 `send()` 消费六种 SSE 事件：

| 事件 | 前端行为 |
|---|---|
| `start` | 记下 `request_id` / `session_id` |
| `thinking` | 追加到该条消息的 `thinking` 字符串（`stage` 与 `reasoning` 同路显示） |
| `delta` | 追加到正文，显示光标 |
| `reset` | 重路由：清空正文，避免两轮答案拼在一起 |
| `final` | 用 `final.answer` **覆盖**正文、用 `final.thinking` 覆盖思考、关闭光标 |
| `error` | 把错误显示在正文位置，保留 `request_id` |

正文上方是 `ThinkingBlock`：流式时自动展开，收到 `final` 后自动折叠；用户也可以
手动开合。它和正文是两个独立区域——思考过程永远不参与答案拼接。

流式正文可能和最终答复不一致（汇总中途失败会退回拼接），所以 `final.answer`
是权威版本；这一点和 Streamlit 版、CLI 完全一致。

## 登录与历史会话

`AUTH_ENABLED=1` 时，设置面板里的用户名 + 密码换短期 token；登录成功后：

1. `fetchSessions()` 调 `GET /sessions`，拿到自己的会话标题、时间、条数；
2. `mergeServerSessions()` 把服务端列表和本地缓存合并：同一个 `sessionId`
   以本地正文（含 thinking / meta）为准，服务端更新时间更晚则标成未加载；
3. 切到未加载的会话时，`fetchSession()` 拉 `GET /sessions/{id}`，按时间正序
   渲染问答原文；
4. 删除服务端会话时先等 `DELETE /sessions/{id}` 成功，再从界面移除。

设置面板里登录后会出现「退出登录」：先清本地凭证，再调 `POST /auth/logout`
让服务端吊销这枚登录 token。吊销是**尽力而为**——网络失败或 token 已经过期
都不阻止本地退出；服务端返回 401 也按成功处理，不能把人卡在登录态里。
但**长期 API token 不在登出的吊销范围内**：服务端这时返回 `revoked: false`，
界面会明确提示"本地已退出，但这枚 token 仍然有效，要吊销请用
`ecom.auth.cli rotate`"，不会把一次假的安全退出说成成功。
退出时还会把 `sessionStorage` 里的会话缓存重置成一张空白对话；登录成功和
手工换 token 也会先清缓存，避免账号 A 的对话留在账号 B 的侧边栏里。被动过期
（token 失效）只清凭证、保留当前视图，下一次登录时再清。

服务端只存问答原文，**思考过程不落盘**，所以重新登录后历史消息没有
`ThinkingBlock`，这是有意的。历史列表按主人过滤；别人的会话和不存在的会话
都返回 404，界面统一显示“已不存在或不属于当前账号”。

## 转人工与客服台

界面上有两端：

- **用户端**：顶栏「转人工」按钮只在 `health.auth && !health.demo && token &&
  sessionId` 时出现；点击调 `POST /sessions/{id}/handoff`（幂等）。有未关闭工单
  时按钮换成状态条：排队中显示“前面还有 N 人”，认领后显示客服名字。工单没关
  之前每 5 秒 `GET /sessions/{id}/ticket`，并只把**新的 `agent` 消息**追加进
  当前对话（按服务端消息 id 去重）；关闭后停止轮询。
- **客服端**：角色是 `agent` / `admin` 时多一个「客服台」按钮，打开
  `AgentConsole`：排队中 / 进行中 / 已结束三个列表 → 看用户与机器人的完整上下文
  （转人工原因单独一行）→ 认领 → 回复 → 关闭。队列和当前工单每 5 秒自动刷新，
  新消息会自己出现，上下文自动滚到最新一条；刷新不清正在打的草稿。回复立即
  写进用户会话，用户那一侧下一次轮询就能看到。
  普通客服的列表和详情都只能看到排队中 + 自己认领/关闭的单；`admin` 额外看到
  「强制关闭」和转派行（输入用户名转派 / 放回队列），用于客服失联时解卡。

几个刻意的边界：

- **按钮藏起来不是权限**。角色只决定界面显示；`/tickets*` 每次都由服务端
  `require_agent` 再校验一次。
- **人工消息是 `role="agent"`**，绿色头像 +「人工客服 · 名字」标签，和机器人
  区分开；正文仍走安全 Markdown 渲染。
- **没有 WebSocket**：5 秒轮询是演示级取舍（回复最多晚 5 秒），换来没有长连接
  和重连状态机。用户侧和客服台都走同一条：定时拉取，不推。
- **401 统一交给 App**：token 失效时清凭证、提示重新登录、关闭客服台，而不是
  在弹窗里显示一句看不懂的“未授权”。

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

- **API Key / 邀请码 / 登录 token 只放 `sessionStorage`**，关掉标签页就没；
  不进 URL、不写 `localStorage`。会话号同理——它是凭证，不是账号。
- 身份模式（`AUTH_ENABLED=1`）下，设置面板有用户名 + 密码登录，调 `/auth/login`
  换短期 token；**密码只存在组件 state，不写 sessionStorage**，存下来的只有 token。
  也可以跳过登录直接粘贴 token。
- **历史列表来自服务端，但凭证仍不落磁盘**：登录 token 在 `sessionStorage`，
  历史正文只进内存和渲染层；没有把 `session_id` 或 token 写进 `localStorage`。
- 右侧轨迹**不接** `/trace`：这个接口会返回所有人问过的话和内部耗时，是排障
  窗口，不是给演示观众看的；演示模式下服务端也会返回 404。面板只显示本次 SSE
  `final` 事件里的路由、决策、来源和 token。
- 回答走自写的安全 Markdown 渲染器（标题、列表、引用、围栏代码、行内代码、
  加粗、http/https 链接）。它只生成 React 元素，不拼 DOM 字符串，不用
  `innerHTML` / `dangerouslySetInnerHTML`；不是通用 Markdown 实现。
- 前端构建产物里**没有任何密钥**；`VITE_*` 变量会被打进 bundle，别往里放
  secret（本项目只用了 `VITE_API_TARGET`，而且只在开发服务器里生效）。

## 目录

```
frontend/
├── index.html
├── vite.config.ts        base=/ui/ + 开发代理
├── src/
│   ├── main.tsx
│   ├── App.tsx           会话存储、SSE 消费、思考块、主题、布局状态
│   ├── api.ts            fetch + SSE 解析 + 错误体
│   ├── labels.ts         worker / decision 的中文标签
│   ├── types.ts          与 api.py 的 ChatResponse / SSE 事件对齐
│   ├── components/
│   │   ├── Sidebar.tsx          会话列表、能力状态、折叠
│   │   ├── Composer.tsx         自适应输入框、发送 / 停止
│   │   ├── TracePanel.tsx       右侧轨迹：路由、决策、来源
│   │   ├── Markdown.tsx         安全 Markdown 渲染
│   │   ├── SettingsDialog.tsx   登录 / API Key / 邀请码
│   │   ├── AgentConsole.tsx     客服台：队列 / 认领 / 回复 / 关闭
│   │   └── Icon.tsx             内联 SVG 图标
│   └── styles.css
└── README.md
```

## 已知限制

- 没有自动化浏览器测试（本仓库的测试原则是离线、不联网；Playwright 需要
  下载浏览器，所以只做了构建和静态约定检查 `tests/test_frontend.py`）。
- 没有图片输入 / 数据分析看板 / 后台入口——那些在 Streamlit 版里。
- 转人工没有实时推送：用户侧和客服台都是 5 秒轮询；没有声音/桌面提醒、
  没有转接升级、没有“正在输入”。排队位和状态都是服务端算好的，前端不自己推。
- Markdown 渲染刻意只支持少数语法：没有表格、嵌套列表、代码高亮。
- 会话号不放进 URL。匿名模式下**新开一个标签页就是新对话**；登录模式下新标签页
  会先要求登录，再从服务端恢复该账号的历史。两种模式的边界见根 README
  「会话 id 是凭证，不是账号」。
