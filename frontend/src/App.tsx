import { useEffect, useRef, useState } from "react";

import { ApiError, fetchHealth, streamChat } from "./api";
import type { ChatMeta, Health, Message } from "./types";

const SESSION_KEY = "ecom.session_id";
const API_KEY_KEY = "ecom.api_key";
const INVITE_KEY = "ecom.invite_code";

const WORKER_LABELS: Record<string, string> = {
  knowledge: "知识库",
  order: "订单",
  analytics: "数据分析",
};

const DECISION_LABELS: Record<string, string> = {
  answer: "直接回答",
  re_route: "换一个单元",
  human: "转人工",
};

const SUGGESTIONS = [
  "订单 52644 到哪了",
  "退货政策是什么",
  "订单 52644 能退吗",
];

const WELCOME: Message = {
  id: "welcome",
  role: "assistant",
  content:
    "你好，我是电商客服助手。可以帮你查订单和物流、回答售后政策，也能做销售数据分析。\n\n直接说订单号就行，比如「订单 52644 到哪了」。",
};

function readStored(key: string): string {
  try {
    return sessionStorage.getItem(key) ?? "";
  } catch {
    // 隐私模式下 sessionStorage 可能直接抛。存不下就不存，不影响这一次对话。
    return "";
  }
}

function writeStored(key: string, value: string): void {
  try {
    if (value) sessionStorage.setItem(key, value);
    else sessionStorage.removeItem(key);
  } catch {
    // 同上。
  }
}

function newId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function errorText(error: unknown): string {
  if (error instanceof ApiError) {
    return error.requestId
      ? `${error.message}（request_id: ${error.requestId}）`
      : error.message;
  }
  if (error instanceof Error && error.message) return error.message;
  return "请求失败，请稍后再试。";
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([WELCOME]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [health, setHealth] = useState<Health | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [sessionId, setSessionId] = useState(() => readStored(SESSION_KEY));
  const [apiKey, setApiKey] = useState(() => readStored(API_KEY_KEY));
  const [inviteCode, setInviteCode] = useState(() => readStored(INVITE_KEY));
  const [keyDraft, setKeyDraft] = useState(apiKey);
  const [inviteDraft, setInviteDraft] = useState(inviteCode);

  const abortRef = useRef<AbortController | null>(null);
  const listRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetchHealth(controller.signal)
      .then(setHealth)
      .catch(() => setHealth(null));
    return () => controller.abort();
  }, []);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  useEffect(() => () => abortRef.current?.abort(), []);

  function updateAssistant(id: string, update: (message: Message) => Message): void {
    setMessages((current) =>
      current.map((message) => (message.id === id ? update(message) : message)),
    );
  }

  async function send(raw: string): Promise<void> {
    const question = raw.trim();
    if (!question || busy) return;

    setInput("");
    setError("");
    setBusy(true);

    const assistantId = newId();
    setMessages((current) => [
      ...current,
      { id: newId(), role: "user", content: question },
      { id: assistantId, role: "assistant", content: "", streaming: true },
    ]);

    const controller = new AbortController();
    abortRef.current = controller;
    let finished = false;
    let currentSession = sessionId;

    try {
      for await (const event of streamChat({
        question,
        sessionId: currentSession,
        apiKey,
        inviteCode,
        signal: controller.signal,
      })) {
        if (event.type === "start") {
          currentSession = event.session_id;
          setSessionId(currentSession);
          writeStored(SESSION_KEY, currentSession);
        } else if (event.type === "delta") {
          updateAssistant(assistantId, (message) => ({
            ...message,
            content: message.content + event.text,
          }));
        } else if (event.type === "final") {
          finished = true;
          currentSession = event.session_id || currentSession;
          setSessionId(currentSession);
          writeStored(SESSION_KEY, currentSession);
          const meta: ChatMeta = {
            sources: event.sources ?? [],
            plan: event.plan ?? [],
            decision: event.decision ?? "",
            tokens: event.tokens ?? {},
            elapsedMs: event.elapsed_ms,
            requestId: event.request_id,
          };
          // final.answer 是权威版本：中途失败时它可能和前面的 delta 不一致，
          // 所以是**整体替换**，不是继续追加。
          updateAssistant(assistantId, (message) => ({
            ...message,
            content: event.answer || message.content,
            streaming: false,
            meta,
          }));
        } else {
          throw new ApiError(event.error, 0, event.request_id);
        }
      }

      if (!finished) {
        throw new Error("这次请求没有正常结束，请重试。");
      }
    } catch (caught) {
      const aborted = controller.signal.aborted;
      const text = aborted ? "（已停止）" : errorText(caught);
      updateAssistant(assistantId, (message) => ({
        ...message,
        streaming: false,
        content: message.content ? `${message.content}\n\n${text}` : text,
      }));
      if (!aborted) setError(text);
    } finally {
      abortRef.current = null;
      setBusy(false);
    }
  }

  function stop(): void {
    abortRef.current?.abort();
  }

  function newConversation(): void {
    abortRef.current?.abort();
    writeStored(SESSION_KEY, "");
    setSessionId("");
    setMessages([WELCOME]);
    setError("");
  }

  function saveSettings(): void {
    const key = keyDraft.trim();
    const invite = inviteDraft.trim();
    writeStored(API_KEY_KEY, key);
    writeStored(INVITE_KEY, invite);
    setApiKey(key);
    setInviteCode(invite);
    setShowSettings(false);
  }

  const statusText = health
    ? `${health.workers.length} 个单元在线 · ${health.order_db}`
    : "接口未连接";

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="logo" aria-hidden="true">
            E
          </div>
          <div>
            <h1>电商智能客服</h1>
            <p>Supervisor + 知识库 / 订单 / 数据分析</p>
          </div>
        </div>
        <div className="topbar-actions">
          <span
            className={`status ${health ? "online" : "offline"}`}
            title={health ? `最多跑 ${health.max_rounds} 轮` : "检查 uvicorn 是否已启动"}
          >
            {statusText}
          </span>
          {health?.demo && <span className="badge">只读演示</span>}
          <button
            className="ghost"
            type="button"
            aria-expanded={showSettings}
            onClick={() => setShowSettings((value) => !value)}
          >
            连接设置
          </button>
          <button className="ghost" type="button" onClick={newConversation} disabled={busy}>
            新对话
          </button>
        </div>
      </header>

      {showSettings && (
        <section className="settings" aria-label="连接设置">
          <label>
            API Key
            <input
              type="password"
              autoComplete="off"
              value={keyDraft}
              onChange={(event) => setKeyDraft(event.target.value)}
              placeholder="服务端没配 API_KEY 就留空"
            />
          </label>
          <label>
            演示邀请码
            <input
              type="password"
              autoComplete="off"
              value={inviteDraft}
              onChange={(event) => setInviteDraft(event.target.value)}
              placeholder="只在只读演示环境需要"
            />
          </label>
          <div className="settings-actions">
            <p>只存在本标签页的 sessionStorage，不进 URL、不写 localStorage。</p>
            <button className="primary" type="button" onClick={saveSettings}>
              保存
            </button>
          </div>
        </section>
      )}

      {error && (
        <div className="banner" role="alert">
          <span>{error}</span>
          <button type="button" onClick={() => setError("")}>
            关闭
          </button>
        </div>
      )}

      <main className="chat" ref={listRef} aria-live="polite">
        {messages.map((message) => (
          <article key={message.id} className={`message ${message.role}`}>
            <div className="avatar" aria-hidden="true">
              {message.role === "user" ? "我" : "AI"}
            </div>
            <div className="message-body">
              <div className="content">
                {message.streaming && !message.content ? (
                  <span className="thinking">
                    正在查询<span className="dots" />
                  </span>
                ) : (
                  <>
                    {message.content}
                    {message.streaming && <span className="cursor" />}
                  </>
                )}
              </div>
              {message.meta && <MessageMeta meta={message.meta} />}
            </div>
          </article>
        ))}

        {messages.length === 1 && (
          <div className="suggestions">
            {SUGGESTIONS.map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                onClick={() => void send(suggestion)}
                disabled={busy}
              >
                {suggestion}
              </button>
            ))}
          </div>
        )}
      </main>

      <footer className="composer">
        <label className="sr-only" htmlFor="question">
          输入问题
        </label>
        <textarea
          id="question"
          rows={1}
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (
              event.key === "Enter" &&
              !event.shiftKey &&
              !event.nativeEvent.isComposing
            ) {
              event.preventDefault();
              void send(input);
            }
          }}
          placeholder="例如：订单 52644 到哪了"
          maxLength={2000}
        />
        {busy ? (
          <button className="stop" type="button" onClick={stop}>
            停止
          </button>
        ) : (
          <button
            className="primary"
            type="button"
            onClick={() => void send(input)}
            disabled={!input.trim()}
          >
            发送
          </button>
        )}
        <p className="hint">
          Enter 发送，Shift + Enter 换行。会话号是凭证，只存在本标签页。
        </p>
      </footer>
    </div>
  );
}

function MessageMeta({ meta }: { meta: ChatMeta }) {
  const totalTokens = (meta.tokens.in ?? 0) + (meta.tokens.out ?? 0);
  return (
    <div className="meta">
      {meta.plan.length > 0 && (
        <span>
          路由：
          {meta.plan.map((name) => WORKER_LABELS[name] ?? name).join(" / ")}
        </span>
      )}
      {meta.decision && (
        <span>决策：{DECISION_LABELS[meta.decision] ?? meta.decision}</span>
      )}
      {meta.sources.length > 0 && (
        <span className="sources">
          来源：
          {meta.sources.map((source) => (
            <span className="chip" key={source}>
              {source}
            </span>
          ))}
        </span>
      )}
      {totalTokens > 0 && (
        <span>
          {totalTokens} token（入 {meta.tokens.in ?? 0} / 出 {meta.tokens.out ?? 0}）
        </span>
      )}
      {typeof meta.elapsedMs === "number" && <span>{(meta.elapsedMs / 1000).toFixed(1)}s</span>}
    </div>
  );
}
