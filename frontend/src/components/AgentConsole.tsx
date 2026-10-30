import { useEffect, useRef, useState } from "react";

import {
  ApiError,
  claimTicket,
  closeTicket,
  fetchTicketDetail,
  fetchTickets,
  replyTicket,
  transferTicket,
} from "../api";
import type { TicketDetail, TicketSummary } from "../types";
import { Icon } from "./Icon";

type Tab = "open" | "claimed" | "closed";

const TABS: Array<{ key: Tab; label: string }> = [
  { key: "open", label: "排队中" },
  { key: "claimed", label: "进行中" },
  { key: "closed", label: "已结束" },
];

const STATUS_LABEL: Record<string, string> = {
  open: "排队中",
  claimed: "进行中",
  closed: "已结束",
};

interface AgentConsoleProps {
  open: boolean;
  token: string;
  /** 当前角色；admin 多出强制关闭和转派两个恢复动作。 */
  role: string;
  onClose: () => void;
  /** token 失效时交给 App 统一处理（清凭证、提示重新登录）。 */
  onUnauthorized: () => void;
}

/**
 * 客服工作台：队列 → 认领 → 看上下文 → 回复 → 关闭。
 *
 * 这是"转人工"闭环里**人的那一端**。没有它，工单只是数据库里的一行，
 * 用户那句"已转接人工"就还是话术。
 *
 * 只对 agent / admin 显示（按钮在 App 里按角色判断）；服务端仍然会再校验
 * 一次角色——前端藏起来不代表接口关上了。
 */
export function AgentConsole({ open, token, role, onClose, onUnauthorized }: AgentConsoleProps) {
  const [tab, setTab] = useState<Tab>("open");
  const [tickets, setTickets] = useState<TicketSummary[]>([]);
  const [selected, setSelected] = useState<TicketDetail | null>(null);
  const [draft, setDraft] = useState("");
  const [assignee, setAssignee] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const isAdmin = role === "admin";
  const transcriptRef = useRef<HTMLDivElement | null>(null);
  const selectedId = selected?.ticket_id ?? "";
  // App 每次渲染都会传一个新的回调进来；放进轮询的依赖里会让定时器被反复
  // 重建。用 ref 拿最新那份，effect 只依赖真正会变的东西。
  const onUnauthorizedRef = useRef(onUnauthorized);
  useEffect(() => {
    onUnauthorizedRef.current = onUnauthorized;
  }, [onUnauthorized]);

  function fail(caught: unknown): void {
    if (caught instanceof ApiError && caught.status === 401) {
      onUnauthorized();
      return;
    }
    setError(caught instanceof ApiError ? caught.message : "操作失败，请稍后再试。");
  }

  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    fetchTickets(token, tab, controller.signal)
      .then((items) => {
        if (controller.signal.aborted) return;
        setTickets(items);
        setSelected(null);
        setError("");
      })
      .catch((caught) => {
        if (!controller.signal.aborted) fail(caught);
      });
    return () => controller.abort();
  }, [open, tab, token]);

  /**
   * 队列和当前工单每 5 秒自动刷新一次。
   *
   * 没有 WebSocket：代价是最多晚 5 秒，好处是没有长连接和重连状态机。
   * 新工单、用户新消息、别人抢先认领、admin 关单，都靠这一条。
   * **只更新列表和详情，不动输入框里的草稿**——客服正在打字时不能被刷新清空。
   */
  useEffect(() => {
    if (!open) return;
    let alive = true;
    async function tick(): Promise<void> {
      try {
        const items = await fetchTickets(token, tab);
        if (!alive) return;
        setTickets(items);
        if (!selectedId) return;
        try {
          const detail = await fetchTicketDetail(selectedId, token);
          if (alive) setSelected(detail);
        } catch (caught) {
          if (!alive) return;
          if (caught instanceof ApiError && caught.status === 401) {
            onUnauthorizedRef.current();
            return;
          }
          // 工单被转走或关掉后可能不在可见范围里：收起详情，列表已经更新。
          if (
            caught instanceof ApiError &&
            (caught.status === 403 || caught.status === 404)
          ) {
            setSelected(null);
          }
        }
      } catch (caught) {
        if (!alive) return;
        if (caught instanceof ApiError && caught.status === 401) {
          onUnauthorizedRef.current();
        }
      }
    }
    const timer = window.setInterval(() => void tick(), 5000);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, [open, tab, token, selectedId]);

  // 有人回复、或者用户又发了一句：把上下文滚到最新一条。
  useEffect(() => {
    const list = transcriptRef.current;
    if (list) list.scrollTop = list.scrollHeight;
  }, [selectedId, selected?.transcript.length]);

  useEffect(() => {
    if (!open) return;
    function onKey(event: KeyboardEvent): void {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  async function refreshList(): Promise<void> {
    try {
      setTickets(await fetchTickets(token, tab));
    } catch (caught) {
      fail(caught);
    }
  }

  async function openTicket(ticket: TicketSummary): Promise<void> {
    setError("");
    setSelected(null);
    try {
      const detail = await fetchTicketDetail(ticket.ticket_id, token);
      setSelected(detail);
      setDraft("");
      setAssignee("");
    } catch (caught) {
      fail(caught);
    }
  }

  async function act(action: () => Promise<unknown>): Promise<void> {
    if (!selected || busy) return;
    setBusy(true);
    setError("");
    try {
      await action();
      const detail = await fetchTicketDetail(selected.ticket_id, token);
      setSelected(detail);
      await refreshList();
    } catch (caught) {
      fail(caught);
    } finally {
      setBusy(false);
    }
  }

  const canReply = selected?.status === "claimed";

  return (
    <div
      className="dialog-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section
        className="dialog agent-console"
        role="dialog"
        aria-modal="true"
        aria-labelledby="console-title"
      >
        <header className="dialog-head">
          <div>
            <h2 id="console-title">客服工作台</h2>
            <p>转人工的会话在这里排队；认领后可以看上下文并回复。队列每 5 秒自动刷新。</p>
          </div>
          <button className="icon-btn" type="button" onClick={onClose} aria-label="关闭客服台">
            <Icon name="close" size={17} />
          </button>
        </header>

        <div className="console-body">
          <div className="console-queue">
            <div className="console-tabs" role="tablist" aria-label="工单状态">
              {TABS.map((item) => (
                <button
                  key={item.key}
                  type="button"
                  role="tab"
                  aria-selected={tab === item.key}
                  className={tab === item.key ? "active" : ""}
                  onClick={() => setTab(item.key)}
                >
                  {item.label}
                </button>
              ))}
            </div>

            <ul className="ticket-list">
              {tickets.map((ticket) => (
                <li key={ticket.ticket_id}>
                  <button
                    type="button"
                    className={`ticket-row${
                      selected?.ticket_id === ticket.ticket_id ? " active" : ""
                    }`}
                    onClick={() => void openTicket(ticket)}
                  >
                    <strong>{ticket.title || "（无标题会话）"}</strong>
                    <span>
                      {ticket.ticket_id} · {STATUS_LABEL[ticket.status] ?? ticket.status}
                      {ticket.claimed_by_name ? ` · ${ticket.claimed_by_name}` : ""}
                      {ticket.status === "open" ? ` · 前面 ${ticket.queue_position} 人` : ""}
                    </span>
                  </button>
                </li>
              ))}
              {tickets.length === 0 && <li className="ticket-empty">这个列表是空的。</li>}
            </ul>
          </div>

          <div className="console-detail">
            {error && (
              <div className="console-error" role="alert">
                {error}
              </div>
            )}

            {!selected ? (
              <div className="console-placeholder">选一张工单，先看用户和机器人聊了什么。</div>
            ) : (
              <>
                <div className="console-meta">
                  <div>
                    <strong>{selected.title || "（无标题会话）"}</strong>
                    <span>
                      {selected.ticket_id} · {STATUS_LABEL[selected.status] ?? selected.status}
                      {selected.claimed_by_name ? ` · ${selected.claimed_by_name}` : ""}
                    </span>
                  </div>
                  <div className="console-actions">
                    {selected.status === "open" && (
                      <button
                        className="primary-btn"
                        type="button"
                        disabled={busy}
                        onClick={() => selected && void act(() => claimTicket(selected.ticket_id, token))}
                      >
                        认领
                      </button>
                    )}
                    {selected.status === "claimed" && (
                      <button
                        className="ghost-btn"
                        type="button"
                        disabled={busy}
                        onClick={() => selected && void act(() => closeTicket(selected.ticket_id, token))}
                      >
                        {isAdmin ? "强制关闭" : "结束工单"}
                      </button>
                    )}
                    {isAdmin && selected.status === "open" && (
                      <button
                        className="ghost-btn"
                        type="button"
                        disabled={busy}
                        onClick={() => selected && void act(() => closeTicket(selected.ticket_id, token))}
                      >
                        强制关闭
                      </button>
                    )}
                  </div>
                </div>

                {isAdmin && selected.status !== "closed" && (
                  <div className="transfer-row">
                    <input
                      value={assignee}
                      onChange={(event) => setAssignee(event.target.value)}
                      placeholder="客服用户名（转派给谁）"
                      aria-label="转派目标客服用户名"
                      disabled={busy}
                    />
                    <button
                      className="ghost-btn"
                      type="button"
                      disabled={busy || !assignee.trim()}
                      onClick={() =>
                        selected &&
                        void act(() =>
                          transferTicket(selected.ticket_id, token, assignee.trim()),
                        )
                      }
                    >
                      转派
                    </button>
                    {selected.status === "claimed" && (
                      <button
                        className="ghost-btn"
                        type="button"
                        disabled={busy}
                        onClick={() =>
                          selected && void act(() => transferTicket(selected.ticket_id, token, null))
                        }
                      >
                        放回队列
                      </button>
                    )}
                  </div>
                )}

                {selected.reason && (
                  <p className="console-reason">转人工原因：{selected.reason}</p>
                )}

                <div className="ticket-transcript" ref={transcriptRef}>
                  {selected.transcript.map((message) => (
                    <div key={message.id} className={`transcript-line ${message.role}`}>
                      <span className="transcript-who">
                        {message.role === "user"
                          ? "用户"
                          : message.role === "agent"
                            ? `人工 ${message.author || ""}`.trim()
                            : "机器人"}
                      </span>
                      <p>{message.content}</p>
                    </div>
                  ))}
                </div>

                <div className="agent-reply">
                  <textarea
                    value={draft}
                    onChange={(event) => setDraft(event.target.value)}
                    placeholder={
                      canReply ? "回复用户（会出现在对方的对话里）" : "认领之后才能回复"
                    }
                    disabled={!canReply || busy}
                    rows={3}
                  />
                  <button
                    className="primary-btn"
                    type="button"
                    disabled={!canReply || busy || !draft.trim()}
                    onClick={() =>
                      selected &&
                      void act(async () => {
                        await replyTicket(selected.ticket_id, token, draft.trim());
                        setDraft("");
                      })
                    }
                  >
                    发送回复
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}
