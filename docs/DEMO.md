# 公网只读演示（L1）

这是一套可以挂在公网上、给别人点开看的电商客服 Demo。它不是一个生产客服系统：
**只能读，不能替用户办事**，而且有硬性的每日成本上限。

- 地址：部署后填入 `https://<DOMAIN>`
- 邀请码：单独发给对方（不在这个仓库、不写进 URL、不贴公开聊天）
- 代码入口：[app.py](../app.py)；部署编排：[docker-compose.prod.yml](../docker-compose.prod.yml)

## 打开之后能做什么

| 能 | 不能（有意关闭） |
| --- | --- |
| 查订单状态、物流进度 | 提交退货申请（写操作） |
| 问售后 / 退换货政策 | 数据分析（会在服务器上执行模型生成的代码） |
| 上传截图/发票/面单，提取订单号后查 | 后台管理端（`admin.py`，能批准退货） |
| 看这一轮路由到了谁、耗时多少 | 看别人的会话全部历史、看 trace 接口 |

界面上仍然有「决策链」折叠块——它是这个 Demo 最值得看的部分：一次提问怎么被
路由、哪个 worker 成功、汇总用了哪种方式。**trace 接口本身不开放**，因为里面有
用户问过的话和内部耗时。

## 为什么这些能力被关掉

不是"没做完"，是公网演示的成本/风险不对称：

- **退货申请**会落库。演示环境没有身份，谁都能替任意订单提申请——它已经不再是
  "演示"，而是"替真实数据写脏"。写闸在 [workers/order.py](../ecom/orchestrator/workers/order.py)
  的落库之前，不靠界面藏按钮。
- **数据分析**会执行模型生成的 Python。当前沙箱不是强隔离，公网上不该有这个面。
- **后台管理端**能批准退货。它只应该从服务器本机 / SSH 隧道访问。
- **trace** 记录的是所有访问者的问题。给一个人看等于泄露所有人的提问。

## 成本与限流

两层的职责不一样：

- **限流**：界面和 API 都按客户端 IP，每分钟最多 `DEMO_RATE_LIMIT_PER_MINUTE`
  次；拿不到 IP 才回退到浏览器会话（Streamlit 的 session 按标签页分，单靠它
  新开标签页就能换桶）。Caddy 后面的真实 IP 来自 `X-Forwarded-For` 最后一跳，
  只在 `TRUST_PROXY_HEADERS=1` 时读取，L1 编排已经打开。
  它挡突发；换 IP / NAT 出口仍能绕过，所以它只是第一道。
- **每日 token 预算**：所有请求的实际用量相加，到 `DEMO_DAILY_TOKEN_BUDGET` 就停止
  服务，第二天按服务器本地日期重置。账本在 `data/orchestrator/demo_usage.sqlite`，
  **重启也算数**。这才是真正兜住成本的那道闸。

预算用完时界面显示"今天的演示额度已经用完"；要临时放开就改 `.env.prod` 里的数字
并重启 app，见 [RUNBOOK.md](RUNBOOK.md)。

## 说清楚的限制

1. **没有登录态。** 邀请码只是一个共享口令，不能单独吊销某个人；换码只能所有人一起换。
   `AUTH_ENABLED` 那套 token / 角色 / 审计是可选的，但这份演示编排没开（见 README「身份与权限」）。
2. **单副本。** 会话和预算账本都在本地 sqlite；起两个副本会各算各的、上下文会岔。
3. **数据是公开数据集派生的演示订单**，不是任何真实客户的数据。
4. **没有 SLA。** 模型超时、知识库检索不到、图片识别失败都会降级成一句人话，
   图片理解之前会先过限流和预算（它是一次独立的视觉模型调用，不能绕开成本闸）。
   但不会自动重试到成功。
5. **会话保留 7 天**（`SESSION_TTL_DAYS`，演示模板的默认值）。到期由启动时清理 +
   外挂 cron 完成，长驻进程不会自己清。

## 在本地验证"只读"是不是真的

不需要部署也能验证写闸：

```bash
DEMO_MODE=1 DEMO_INVITE_CODE=test uv run python main.py "订单 52644 我要退货"
```

预期：答复里明确说只读演示、不能提交退货申请；`data/orders.sqlite` 的 `returns`
表里不会多出记录。查物流、问政策仍然正常。

更完整的行为由这些测试钉住：

- [tests/test_demo.py](../tests/test_demo.py) —— 邀请码 fail closed、限流、预算账本、只读策略
- [tests/orchestrator/test_router.py](../tests/orchestrator/test_router.py) —— 演示模式不路由到 analytics
- [tests/orchestrator/test_order_worker.py](../tests/orchestrator/test_order_worker.py) —— 写操作在落库前被挡住
- [tests/test_api.py](../tests/test_api.py) —— 邀请码、429、trace 404、流式也记账
