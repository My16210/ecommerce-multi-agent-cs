import { useEffect, useRef, useState } from "react";

import {
  ApiError,
  deleteSession,
  fetchHealth,
  fetchSession,
  fetchSessions,
  fetchTicket,
  login,
  openHandoff,
  streamChat,
} from "./api";
import { AgentConsole } from "./components/AgentConsole";
import { Composer } from "./components/Composer";
import { Icon, type IconName } from "./components/Icon";
import { Markdown } from "./components/Markdown";
import { SettingsDialog } from "./components/SettingsDialog";
import { Sidebar } from "./components/Sidebar";
import { TracePanel } from "./components/TracePanel";
import { decisionLabel, workerLabel } from "./labels";
import type {
  ChatMeta,
  Conversation,
  Health,
  Message,
  SessionSummary,
  Theme,
  TicketInfo,
} from "./types";

const STORE_KEY = "ecom.conversations";
const ACTIVE_KEY = "ecom.active_conversation";
const LEGACY_SESSION_KEY = "ecom.session_id";
const API_KEY_KEY = "ecom.api_key";
const IDENTITY_KEY = "ecom.identity";
const ROLE_KEY = "ecom.role";
const INVITE_KEY = "ecom.invite_code";
const THEME_KEY = "ecom.theme";
const MAX_CONVERSATIONS = 20;

const SUGGESTIONS: Array<{ title: string; prompt: string; icon: IconName }> = [
  { title: "订单与物流", prompt: "订单 52644 到哪了", icon: "package" },
  { title: "售后政策", prompt: "退货政策是什么", icon: "shield" },
  { title: "能退吗", prompt: "订单 52644 能退吗", icon: "check" },
  { title: "销售分析", prompt: "最近 30 天各品类的销售额", icon: "chart" },
];

const THEME_META: Record<Theme, { icon: IconName; label: string }> = {
  light: { icon: "sun", label: "浅色" },
  dark: { icon: "moon", label: "深色" },
  system: { icon: "monitor", label: "跟随系统" },
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

function makeConversation(
  sessionId = "",
  title = "新对话",
  messages: Message[] = [],
): Conversation {
  return {
    id: newId(),
    sessionId,
    title,
    messages,
    updatedAt: Date.now(),
    loaded: true,
  };
}

/** 服务端时间是 `YYYY-MM-DD HH:MM:SS`（本地时间），Safari 不认空格分隔。 */
function parseServerTime(value: string): number {
  const parsed = Date.parse(String(value || "").replace(" ", "T"));
  return Number.isFinite(parsed) ? parsed : Date.now();
}

/** 轮询每 5 秒跑一次；状态没变就不更新 store，避免白写 sessionStorage。 */
function sameTicket(a?: TicketInfo | null, b?: TicketInfo | null): boolean {
  if (!a || !b) return !a && !b;
  return (
    a.ticket_id === b.ticket_id &&
    a.status === b.status &&
    a.queue_position === b.queue_position &&
    a.claimed_by_name === b.claimed_by_name &&
    a.updated_at === b.updated_at
  );
}

function conversationFromServer(item: SessionSummary): Conversation {
  return {
    id: `srv-${item.session_id}`,
    sessionId: item.session_id,
    title: item.title || "未命名会话",
    // 列表接口刻意不带正文：先给标题占位，切到它时再拉消息。
    messages: [],
    updatedAt: parseServerTime(item.updated_at),
    loaded: false,
  };
}

/**
 * 把服务端列表合并进本地会话。
 *
 * 合并而不是替换，是因为本地还有两类东西服务端没有：还没发出去的新对话，
 * 以及刚聊完、列表请求还没回来的会话。同一个 `sessionId` 以本地正文为准
 * （它带着 thinking / meta，服务端只存了问答原文）；但如果服务端更新时间
 * 明显更新（比如另一个标签页聊过），就把本地标记成未加载，切过去时重拉。
 */
function mergeServerSessions(
  current: { conversations: Conversation[]; activeId: string },
  items: SessionSummary[],
): { conversations: Conversation[]; activeId: string } {
  const serverIds = new Set(items.map((item) => item.session_id));
  const localBySession = new Map(
    current.conversations
      .filter((conversation) => conversation.sessionId)
      .map((conversation) => [conversation.sessionId, conversation]),
  );

  const fromServer = items.map((item) => {
    const local = localBySession.get(item.session_id);
    if (!local) return conversationFromServer(item);

    const remoteAt = parseServerTime(item.updated_at);
    const stale = local.messages.length > 0 && remoteAt > local.updatedAt + 2000;
    return {
      ...local,
      title: local.title && local.title !== "新对话" ? local.title : item.title || local.title,
      messages: stale ? [] : local.messages,
      updatedAt: Math.max(local.updatedAt, remoteAt),
      loaded: stale ? false : local.messages.length > 0 ? true : local.loaded,
    };
  });

  // 本地独有的：新对话草稿、刚发出还没拿到 session_id 的会话，以及服务端
  // 列表里没有的旧缓存（例如上一台机器/另一个身份留下的）。
  const localOnly = current.conversations.filter(
    (conversation) => !conversation.sessionId || !serverIds.has(conversation.sessionId),
  );
  const conversations = [...localOnly, ...fromServer].slice(0, MAX_CONVERSATIONS);

  const activeExists = conversations.some((conversation) => conversation.id === current.activeId);
  let activeId = activeExists ? current.activeId : conversations[0]?.id ?? "";
  const active = conversations.find((conversation) => conversation.id === activeId);
  // 登录后如果停在空白草稿上，直接打开最近一条历史，而不是让用户以为记录没了。
  if (active && active.messages.length === 0 && active.loaded === false && fromServer[0]) {
    activeId = fromServer[0].id;
  }
  return { conversations, activeId };
}

function isMessage(value: unknown): value is Message {
  if (typeof value !== "object" || value === null) return false;
  const message = value as Partial<Message>;
  return (
    typeof message.id === "string" &&
    (message.role === "user" || message.role === "assistant" || message.role === "agent") &&
    typeof message.content === "string"
  );
}

function isConversation(value: unknown): value is Conversation {
  if (typeof value !== "object" || value === null) return false;
  const conversation = value as Partial<Conversation>;
  return (
    typeof conversation.id === "string" &&
    typeof conversation.title === "string" &&
    typeof conversation.updatedAt === "number" &&
    Array.isArray(conversation.messages) &&
    conversation.messages.every(isMessage)
  );
}

/**
 * 会话列表存在 sessionStorage：**每个标签页一份，关掉就没了**。
 *
 * 这不是为了持久化，而是为了刷新页面时还在原地；会话号本身是凭证，
 * 所以坚决不写 localStorage、不进 URL——同标签页里的脚本本来就能读
 * sessionStorage，信任边界没有变大。
 */
function loadStore(): { conversations: Conversation[]; activeId: string } {
  try {
    const raw = sessionStorage.getItem(STORE_KEY);
    if (raw) {
      const parsed: unknown = JSON.parse(raw);
      if (Array.isArray(parsed)) {
        const conversations = parsed
          .filter(isConversation)
          .slice(0, MAX_CONVERSATIONS)
          .map((conversation) => ({
            ...conversation,
            sessionId: typeof conversation.sessionId === "string" ? conversation.sessionId : "",
            messages: conversation.messages.map((message) => ({ ...message, streaming: false })),
            // 缓存里存的是已经拿到的正文，所以是已加载；服务端列出来的才会是 false。
            loaded: true,
          }));
        if (conversations.length > 0) {
          const stored = readStored(ACTIVE_KEY);
          const activeId = conversations.some((item) => item.id === stored)
            ? stored
            : conversations[0].id;
          return { conversations, activeId };
        }
      }
    }
  } catch {
    // 存储损坏就当成没有会话，重新开一个；不要让整个页面白屏。
  }

  const legacy = readStored(LEGACY_SESSION_KEY);
  const first = makeConversation(legacy, legacy ? "上次的对话" : "新对话");
  return { conversations: [first], activeId: first.id };
}

function loadTheme(): Theme {
  const stored = readStored(THEME_KEY);
  return stored === "light" || stored === "dark" || stored === "system" ? stored : "system";
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
  const [store, setStore] = useState(loadStore);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [health, setHealth] = useState<Health | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth > 900);
  const [railOpen, setRailOpen] = useState(() => window.innerWidth >= 1280);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [theme, setTheme] = useState<Theme>(loadTheme);
  const [systemDark, setSystemDark] = useState(
    () => window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false,
  );

  const [apiKey, setApiKey] = useState(() => readStored(API_KEY_KEY));
  const [inviteCode, setInviteCode] = useState(() => readStored(INVITE_KEY));
  const [keyDraft, setKeyDraft] = useState(apiKey);
  const [inviteDraft, setInviteDraft] = useState(inviteCode);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loginBusy, setLoginBusy] = useState(false);
  const [whoami, setWhoami] = useState(() => readStored(IDENTITY_KEY));
  // 角色从登录响应里单独存：身份文案是给人看的，角色决定「客服台」按钮在不在。
  const [role, setRole] = useState(() => readStored(ROLE_KEY));
  const isAgent = role === "agent" || role === "admin";
  const [consoleOpen, setConsoleOpen] = useState(false);
  const [handoffBusy, setHandoffBusy] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const chatRef = useRef<HTMLDivElement | null>(null);
  const loadingSessionsRef = useRef<Set<string>>(new Set());

  const active =
    store.conversations.find((conversation) => conversation.id === store.activeId) ??
    store.conversations[0];
  const messages = active?.messages ?? [];
  const hasUserMessage = messages.some((message) => message.role === "user");
  // 历史会话先只拿到标题，正文还没回来：别把空数组误判成"新对话"去显示欢迎页。
  const loadingHistory = active?.loaded === false;
  const resolvedTheme = theme === "system" ? (systemDark ? "dark" : "light") : theme;
  // 没关的工单才算"正在转人工"；关闭之后同一会话还能再转一次。
  const liveTicket = active?.ticket && active.ticket.status !== "closed" ? active.ticket : null;
  // 演示环境是公共只读的，没有人工客服可转；开了身份层、登录过、有了
  // session_id（服务端要按它落工单）才把按钮放出来。
  const canHandoff = Boolean(
    health?.auth && !health.demo && apiKey && active?.sessionId && !liveTicket,
  );

  useEffect(() => {
    const controller = new AbortController();
    fetchHealth(controller.signal)
      .then(setHealth)
      .catch(() => setHealth(null));
    return () => controller.abort();
  }, []);

  /**
   * 登录后从服务端拉历史会话。**只在身份层打开时拉**：匿名模式下所有人
   * 的 owner_id 都是空串，服务端也不会给这个列表（404），本地继续用
   * sessionStorage 就够了。
   */
  useEffect(() => {
    if (!health?.auth || !apiKey) return;
    const controller = new AbortController();
    fetchSessions(apiKey, controller.signal)
      .then((items) => {
        if (controller.signal.aborted) return;
        setStore((current) => mergeServerSessions(current, items));
      })
      .catch((caught) => {
        if (controller.signal.aborted) return;
        if (caught instanceof ApiError && caught.status === 401) {
          forgetLogin();
          return;
        }
        // 历史拉不到不该挡住聊天：保留本地会话，只在顶部提示。
        setError(errorText(caught));
      });
    return () => controller.abort();
  }, [health?.auth, apiKey]);

  // 切到一条只有摘要、还没拉正文的历史会话时，按需加载。
  useEffect(() => {
    if (!health?.auth || !apiKey) return;
    if (!active || active.loaded !== false || !active.sessionId) return;
    void loadMessages(active.id, active.sessionId);
  }, [active?.id, active?.loaded, active?.sessionId, apiKey, health?.auth]);

  /**
   * 人工工单的轮询：进会话先同步一次；只要工单还没关，就每 5 秒看一次
   * 状态和人工回复。没有 WebSocket——这一层是演示级实现，代价是回复最多
   * 晚 5 秒出现，好处是没有长连接、没有重连状态机。
   */
  useEffect(() => {
    if (!health?.auth || health.demo || !apiKey || !active?.sessionId) return;
    const conversationId = active.id;
    const sessionId = active.sessionId;
    void syncTicket(conversationId, sessionId);

    if (!active.ticket || active.ticket.status === "closed") return;
    const timer = window.setInterval(() => {
      void syncTicket(conversationId, sessionId);
    }, 5000);
    return () => window.clearInterval(timer);
  }, [
    health?.auth,
    health?.demo,
    apiKey,
    active?.id,
    active?.sessionId,
    active?.ticket?.status,
  ]);

  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    function onChange(event: MediaQueryListEvent): void {
      setSystemDark(event.matches);
    }
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = resolvedTheme;
  }, [resolvedTheme]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      try {
        const compact = store.conversations.map((conversation) => ({
          ...conversation,
          messages: conversation.messages.slice(-60).map((message) => ({
            ...message,
            streaming: false,
          })),
        }));
        sessionStorage.setItem(STORE_KEY, JSON.stringify(compact));
        writeStored(ACTIVE_KEY, store.activeId);
      } catch {
        // 配额满或存储不可用：会话只保留在内存里，不影响当前对话。
      }
    }, 400);
    return () => window.clearTimeout(timer);
  }, [store]);

  useEffect(() => {
    const list = chatRef.current;
    if (!list) return;
    list.scrollTo({ top: list.scrollHeight, behavior: busy ? "auto" : "smooth" });
  }, [active?.id, messages, busy]);

  useEffect(() => {
    function onResize(): void {
      if (window.innerWidth <= 900) setSidebarOpen(false);
    }
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  useEffect(() => () => abortRef.current?.abort(), []);

  /** token 失效：清掉凭证和身份文案。**不自动弹登录框**，避免打断正在看的会话。 */
  function forgetLogin(): void {
    writeStored(API_KEY_KEY, "");
    writeStored(IDENTITY_KEY, "");
    writeStored(ROLE_KEY, "");
    setApiKey("");
    setKeyDraft("");
    setWhoami("");
    setRole("");
    setConsoleOpen(false);
    setError("登录已过期，请重新登录。");
  }

  /**
   * 拉一条历史会话的正文。列表接口刻意不带正文，所以这一步是懒加载。
   *
   * `loadingSessionsRef` 防的是同一个会话被重复请求：切来切去时 effect 和
   * `selectConversation` 都可能触发它，没这个 guard 就会并发打同一接口。
   */
  async function loadMessages(conversationId: string, sessionId: string): Promise<void> {
    if (!apiKey || loadingSessionsRef.current.has(sessionId)) return;
    loadingSessionsRef.current.add(sessionId);
    try {
      const detail = await fetchSession(sessionId, apiKey);
      patchConversation(conversationId, (conversation) => ({
        ...conversation,
        // 本地标题如果是用户刚问出来的，保留；否则用服务端的第一条提问。
        title:
          conversation.title && conversation.title !== "新对话"
            ? conversation.title
            : detail.title || conversation.title,
        messages: detail.messages.map((message) => ({
          id: `srv-${message.id}`,
          role:
            message.role === "assistant"
              ? "assistant"
              : message.role === "agent"
                ? "agent"
                : "user",
          content: message.content,
          author: message.author || "",
          createdAt: parseServerTime(message.created_at),
        })),
        updatedAt: Math.max(conversation.updatedAt, parseServerTime(detail.updated_at)),
        loaded: true,
      }));
    } catch (caught) {
      if (caught instanceof ApiError && caught.status === 401) {
        forgetLogin();
      } else if (caught instanceof ApiError && caught.status === 404) {
        // 会话被另一个标签页删了，或者换了账号：不要一直显示转圈。
        setError("这个会话已不存在，或不属于当前账号。");
        patchConversation(conversationId, (conversation) => ({
          ...conversation,
          loaded: true,
        }));
      } else {
        setError(errorText(caught));
      }
    } finally {
      loadingSessionsRef.current.delete(sessionId);
    }
  }

  /**
   * 轮询人工回复。**只追加新出现的 `agent` 消息**，不整体替换：
   * 本地消息带着 thinking / meta，服务端只存了问答原文，整份替换会把
   * 用户正在看的思考过程抹掉。
   */
  async function pullAgentReplies(conversationId: string, sessionId: string): Promise<void> {
    if (!apiKey) return;
    const detail = await fetchSession(sessionId, apiKey);
    const agentMessages = detail.messages.filter((message) => message.role === "agent");
    if (!agentMessages.length) return;
    setStore((current) => {
      const conversation = current.conversations.find((item) => item.id === conversationId);
      if (!conversation) return current;
      const known = new Set(conversation.messages.map((message) => message.id));
      const added = agentMessages
        .filter((message) => !known.has(`srv-${message.id}`))
        .map((message) => ({
          id: `srv-${message.id}`,
          role: "agent" as const,
          content: message.content,
          author: message.author || "",
          createdAt: parseServerTime(message.created_at),
        }));
      if (!added.length) return current;
      return {
        ...current,
        conversations: current.conversations.map((item) =>
          item.id === conversationId
            ? { ...item, messages: [...item.messages, ...added], updatedAt: Date.now() }
            : item,
        ),
      };
    });
  }

  /**
   * 同步一张工单的状态，并顺手取回新的人工回复。**失败静默**（401 除外）：
   * 它每 5 秒跑一次，任何一次网络抖动都弹横幅，界面就没法用了。
   */
  async function syncTicket(conversationId: string, sessionId: string): Promise<void> {
    if (!apiKey) return;
    try {
      const ticket = await fetchTicket(sessionId, apiKey);
      setStore((current) => {
        const conversation = current.conversations.find((item) => item.id === conversationId);
        if (!conversation || sameTicket(conversation.ticket, ticket)) return current;
        return {
          ...current,
          conversations: current.conversations.map((item) =>
            item.id === conversationId ? { ...item, ticket } : item,
          ),
        };
      });
      if (ticket) await pullAgentReplies(conversationId, sessionId);
    } catch (caught) {
      if (caught instanceof ApiError && caught.status === 401) forgetLogin();
    }
  }

  /**
   * 用户主动点「转人工」。幂等：这张工单已经开着就返回同一张，不会排两次队。
   *
   * 按钮只在「身份层打开 + 非演示 + 有 token + 会话已经有 session_id」时显示，
   * 但服务端仍然会再挡一次：演示环境返回 404，界面把这句错误原样显示出来，
   * 而不是假装"已转接"。
   */
  async function requestHandoff(): Promise<void> {
    const conversation = active;
    if (!conversation?.sessionId || !apiKey || handoffBusy) return;
    setHandoffBusy(true);
    setError("");
    try {
      const ticket = await openHandoff(conversation.sessionId, apiKey);
      patchConversation(conversation.id, (item) => ({ ...item, ticket }));
    } catch (caught) {
      if (caught instanceof ApiError && caught.status === 401) forgetLogin();
      else setError(errorText(caught));
    } finally {
      setHandoffBusy(false);
    }
  }

  function patchConversation(
    id: string,
    update: (conversation: Conversation) => Conversation,
  ): void {
    setStore((current) => ({
      ...current,
      conversations: current.conversations.map((conversation) =>
        conversation.id === id ? update(conversation) : conversation,
      ),
    }));
  }

  function updateAssistant(
    conversationId: string,
    messageId: string,
    update: (message: Message) => Message,
  ): void {
    patchConversation(conversationId, (conversation) => ({
      ...conversation,
      updatedAt: Date.now(),
      messages: conversation.messages.map((message) =>
        message.id === messageId ? update(message) : message,
      ),
    }));
  }

  async function send(raw: string): Promise<void> {
    const question = raw.trim();
    if (!question || busy) return;

    const conversationId = store.activeId;
    const conversation = store.conversations.find((item) => item.id === conversationId);
    if (!conversation) return;

    setInput("");
    setError("");
    setBusy(true);

    const assistantId = newId();
    const now = Date.now();
    patchConversation(conversationId, (current) => ({
      ...current,
      title: current.title === "新对话" ? question.slice(0, 24) : current.title,
      updatedAt: now,
      messages: [
        ...current.messages,
        { id: newId(), role: "user", content: question, createdAt: now },
        { id: assistantId, role: "assistant", content: "", streaming: true, createdAt: now },
      ],
    }));

    const controller = new AbortController();
    abortRef.current = controller;
    let finished = false;
    let currentSession = conversation.sessionId;

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
          patchConversation(conversationId, (item) => ({
            ...item,
            sessionId: currentSession,
            updatedAt: Date.now(),
          }));
        } else if (event.type === "thinking") {
          updateAssistant(conversationId, assistantId, (message) => ({
            ...message,
            thinking: (message.thinking ?? "") + event.text,
          }));
        } else if (event.type === "delta") {
          updateAssistant(conversationId, assistantId, (message) => ({
            ...message,
            content: message.content + event.text,
          }));
        } else if (event.type === "reset") {
          // 重路由：上一轮流出的是中间产物，不能和下一轮拼在一起。
          updateAssistant(conversationId, assistantId, (message) => ({
            ...message,
            content: "",
          }));
        } else if (event.type === "final") {
          finished = true;
          currentSession = event.session_id || currentSession;
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
          updateAssistant(conversationId, assistantId, (message) => ({
            ...message,
            content: event.answer || message.content,
            thinking: event.thinking ?? message.thinking,
            streaming: false,
            meta,
          }));
          patchConversation(conversationId, (item) => ({
            ...item,
            sessionId: currentSession,
            updatedAt: Date.now(),
          }));
          // 这一轮转了人工：马上拉一次完整工单（状态、排队位、客服名），
          // 之后由上面的轮询接着看回复。
          if (event.handoff) void syncTicket(conversationId, currentSession);
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
      updateAssistant(conversationId, assistantId, (message) => ({
        ...message,
        streaming: false,
        content: message.content ? `${message.content}\n\n${text}` : text,
      }));
      if (!aborted) setError(text);
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
      setBusy(false);
    }
  }

  function stop(): void {
    abortRef.current?.abort();
  }

  function newConversation(): void {
    abortRef.current?.abort();
    const draft = makeConversation();
    setStore((current) => {
      const kept = current.conversations.filter((item) => item.messages.length > 0);
      return {
        conversations: [draft, ...kept].slice(0, MAX_CONVERSATIONS),
        activeId: draft.id,
      };
    });
    setInput("");
    setError("");
    if (window.innerWidth <= 900) setSidebarOpen(false);
  }

  function selectConversation(id: string): void {
    const target = store.conversations.find((conversation) => conversation.id === id);
    if (id !== store.activeId) {
      abortRef.current?.abort();
      setError("");
      setInput("");
      setStore((current) => ({ ...current, activeId: id }));
    }
    if (target && target.loaded === false && target.sessionId) {
      void loadMessages(target.id, target.sessionId);
    }
    if (window.innerWidth <= 900) setSidebarOpen(false);
  }

  async function deleteConversation(id: string): Promise<void> {
    const target = store.conversations.find((conversation) => conversation.id === id);
    // 服务端删成功再移除本地：反过来会出现"界面上没了、重新登录又回来"。
    // 演示环境是只读的，服务端会拒绝删除，所以这里只做本地移除。
    if (target?.sessionId && health?.auth && apiKey && !health.demo) {
      try {
        await deleteSession(target.sessionId, apiKey);
      } catch (caught) {
        if (caught instanceof ApiError && caught.status === 401) forgetLogin();
        else setError(errorText(caught));
        return;
      }
    }
    if (id === store.activeId) abortRef.current?.abort();
    setStore((current) => {
      const rest = current.conversations.filter((conversation) => conversation.id !== id);
      if (rest.length === 0) {
        const draft = makeConversation();
        return { conversations: [draft], activeId: draft.id };
      }
      return {
        conversations: rest,
        activeId: current.activeId === id ? rest[0].id : current.activeId,
      };
    });
    if (id === store.activeId) {
      setError("");
      setInput("");
    }
  }

  function cycleTheme(): void {
    const next: Theme = theme === "light" ? "dark" : theme === "dark" ? "system" : "light";
    setTheme(next);
    writeStored(THEME_KEY, next);
  }

  function saveSettings(): void {
    const key = keyDraft.trim();
    const invite = inviteDraft.trim();
    // 手工换了 token 就等于换了身份，旧的名字/角色不能再显示。
    if (key !== apiKey) {
      writeStored(IDENTITY_KEY, "");
      writeStored(ROLE_KEY, "");
      setWhoami("");
      setRole("");
      // 换 token 可能等于换成普通用户：客服台按钮要跟着消失，不能等刷新。
      setConsoleOpen(false);
    }
    writeStored(API_KEY_KEY, key);
    writeStored(INVITE_KEY, invite);
    setApiKey(key);
    setInviteCode(invite);
    setSettingsOpen(false);
  }

  async function doLogin(): Promise<void> {
    const name = username.trim();
    if (!name || !password || loginBusy) return;
    setLoginBusy(true);
    setError("");
    try {
      const result = await login(name, password);
      const identity = `${result.name}（${result.role}）`;
      writeStored(API_KEY_KEY, result.token);
      // 名字/角色只用于显示，不是凭证；服务端从不读它。
      writeStored(IDENTITY_KEY, identity);
      writeStored(ROLE_KEY, result.role);
      setApiKey(result.token);
      setKeyDraft(result.token);
      setWhoami(identity);
      setRole(result.role);
      setPassword("");
      // 登录成功后直接收起设置弹窗，让用户看到侧边栏补回来的历史会话。
      setSettingsOpen(false);
    } catch (caught) {
      setError(errorText(caught));
    } finally {
      setLoginBusy(false);
    }
  }

  return (
    <div
      className={`shell${sidebarOpen ? "" : " sidebar-collapsed"}${railOpen ? " rail-open" : ""}`}
    >
      <Sidebar
        conversations={store.conversations}
        activeId={store.activeId}
        collapsed={!sidebarOpen}
        health={health}
        onNew={newConversation}
        onSelect={selectConversation}
        onDelete={deleteConversation}
        onToggle={() => setSidebarOpen((value) => !value)}
        onSettings={() => setSettingsOpen(true)}
      />

      {sidebarOpen && (
        <button
          className="sidebar-backdrop"
          type="button"
          aria-label="关闭侧边栏"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <main className="main">
        <header className="topbar">
          <div className="topbar-left">
            <button
              className="icon-btn"
              type="button"
              onClick={() => setSidebarOpen((value) => !value)}
              aria-label={sidebarOpen ? "收起侧边栏" : "展开侧边栏"}
              title={sidebarOpen ? "收起侧边栏" : "展开侧边栏"}
            >
              <Icon name={sidebarOpen ? "chevron-left" : "menu"} size={17} />
            </button>
            <div className="crumbs">
              <span>电商客服</span>
              <Icon name="chevron-right" size={13} />
              <strong>{active?.title ?? "新对话"}</strong>
            </div>
          </div>

          <div className="topbar-right">
            <span
              className={`status-dot ${health ? "online" : "offline"}`}
              title={health ? `最多跑 ${health.max_rounds} 轮` : "检查 uvicorn 是否已启动"}
            />
            <span className="status-text">
              {health ? `${health.workers.length} 个单元在线` : "接口未连接"}
            </span>
            {health?.demo && <span className="badge">只读演示</span>}
            {canHandoff && (
              <button
                className="handoff-btn"
                type="button"
                onClick={() => void requestHandoff()}
                disabled={handoffBusy}
                aria-label="转人工"
                title="把这次对话转给人工客服"
              >
                <Icon name="headset" size={15} />
                <span>{handoffBusy ? "转接中…" : "转人工"}</span>
              </button>
            )}
            {isAgent && health?.auth && (
              <button
                className="icon-btn"
                type="button"
                onClick={() => setConsoleOpen(true)}
                aria-label="客服工作台"
                title="客服工作台"
              >
                <Icon name="inbox" size={17} />
              </button>
            )}
            <button
              className="icon-btn"
              type="button"
              onClick={cycleTheme}
              aria-label={`主题：${THEME_META[theme].label}`}
              title={`主题：${THEME_META[theme].label}（点击切换）`}
            >
              <Icon name={THEME_META[theme].icon} size={17} />
            </button>
            <button
              className={`icon-btn${railOpen ? " active" : ""}`}
              type="button"
              onClick={() => setRailOpen((value) => !value)}
              aria-label={railOpen ? "关闭会话轨迹" : "打开会话轨迹"}
              aria-pressed={railOpen}
              title={railOpen ? "关闭会话轨迹" : "打开会话轨迹"}
            >
              <Icon name="panel" size={17} />
            </button>
            <button
              className="icon-btn"
              type="button"
              onClick={() => setSettingsOpen(true)}
              aria-label="连接设置"
              title="连接设置"
            >
              <Icon name="settings" size={17} />
            </button>
          </div>
        </header>

        {error && (
          <div className="banner" role="alert">
            <span>{error}</span>
            <button type="button" onClick={() => setError("")} aria-label="关闭提示">
              <Icon name="close" size={14} />
            </button>
          </div>
        )}

        {liveTicket && (
          <div className="handoff-banner" role="status">
            <Icon name="headset" size={16} />
            {liveTicket.status === "claimed" ? (
              <span>
                人工客服
                {liveTicket.claimed_by_name ? ` ${liveTicket.claimed_by_name} ` : " "}
                已接手，回复会出现在这个对话里。
              </span>
            ) : (
              <span>
                已转人工，正在排队
                {liveTicket.queue_position > 0 ? `（前面还有 ${liveTicket.queue_position} 人）` : "（马上就到你了）"}
                。
              </span>
            )}
            <span className="handoff-id">{liveTicket.ticket_id}</span>
          </div>
        )}

        <div className="chat-scroll" ref={chatRef} aria-live="polite">
          <div className="chat-inner">
            {loadingHistory ? (
              <div className="history-loading" aria-live="polite">
                <span className="typing" aria-label="正在加载历史">
                  <span />
                  <span />
                  <span />
                </span>
                <p>正在加载历史会话…</p>
              </div>
            ) : !hasUserMessage ? (
              <Welcome busy={busy} onPick={(prompt) => void send(prompt)} />
            ) : (
              messages.map((message) => (
                <article key={message.id} className={`message ${message.role}`}>
                  {message.role === "assistant" && (
                    <div className="avatar" aria-hidden="true">
                      E
                    </div>
                  )}
                  {message.role === "agent" && (
                    <div className="avatar agent-avatar" aria-hidden="true">
                      <Icon name="headset" size={15} />
                    </div>
                  )}
                  <div className="message-body">
                    {message.role === "agent" && (
                      <div className="agent-label">
                        人工客服{message.author ? ` · ${message.author}` : ""}
                      </div>
                    )}
                    {message.role === "assistant" && message.thinking && (
                      <ThinkingBlock
                        text={message.thinking}
                        live={Boolean(message.streaming)}
                      />
                    )}
                    {(!message.streaming || message.content || !message.thinking) && (
                      <div className={`bubble ${message.role}`}>
                        {message.streaming && !message.content ? (
                          <span className="typing" aria-label="正在生成">
                            <span />
                            <span />
                            <span />
                          </span>
                        ) : message.role === "assistant" || message.role === "agent" ? (
                          <Markdown text={message.content} />
                        ) : (
                          <div className="user-text">{message.content}</div>
                        )}
                        {message.streaming && message.content && <span className="cursor" />}
                      </div>
                    )}
                    {message.meta && (
                      <MessageMeta meta={message.meta} onOpenTrace={() => setRailOpen(true)} />
                    )}
                  </div>
                </article>
              ))
            )}
          </div>
        </div>

        <Composer
          value={input}
          busy={busy}
          onChange={setInput}
          onSend={() => void send(input)}
          onStop={stop}
        />
      </main>

      {railOpen && (
        <>
          <button
            className="rail-backdrop"
            type="button"
            aria-label="关闭会话轨迹"
            onClick={() => setRailOpen(false)}
          />
          <TracePanel health={health} messages={messages} onClose={() => setRailOpen(false)} />
        </>
      )}

      {isAgent && (
        <AgentConsole
          open={consoleOpen}
          token={apiKey}
          onClose={() => setConsoleOpen(false)}
          onUnauthorized={forgetLogin}
        />
      )}

      <SettingsDialog
        open={settingsOpen}
        health={health}
        whoami={whoami}
        username={username}
        password={password}
        apiKey={keyDraft}
        inviteCode={inviteDraft}
        loginBusy={loginBusy}
        onUsername={setUsername}
        onPassword={setPassword}
        onApiKey={setKeyDraft}
        onInviteCode={setInviteDraft}
        onLogin={() => void doLogin()}
        onSave={saveSettings}
        onClose={() => setSettingsOpen(false)}
      />
    </div>
  );
}

function Welcome({ busy, onPick }: { busy: boolean; onPick: (prompt: string) => void }) {
  return (
    <div className="welcome">
      <div className="welcome-mark" aria-hidden="true">
        <Icon name="sparkles" size={24} />
      </div>
      <h1>今天想查点什么？</h1>
      <p>订单、物流、售后政策，或者让数据说话。</p>

      <div className="suggestion-grid">
        {SUGGESTIONS.map((suggestion) => (
          <button
            key={suggestion.prompt}
            type="button"
            onClick={() => onPick(suggestion.prompt)}
            disabled={busy}
          >
            <span className="suggestion-icon" aria-hidden="true">
              <Icon name={suggestion.icon} size={18} />
            </span>
            <span className="suggestion-copy">
              <strong>{suggestion.title}</strong>
              <small>{suggestion.prompt}</small>
            </span>
          </button>
        ))}
      </div>

      <div className="welcome-foot">
        <span>Supervisor 自动选择单元</span>
        <span aria-hidden="true">·</span>
        <span>最多两轮</span>
        <span aria-hidden="true">·</span>
        <span>不联网评测</span>
      </div>
    </div>
  );
}

/**
 * 思考过程：路由、worker、决策，以及模型自己吐出的思维链。
 *
 * 流式期间默认展开并显示「正在思考…」；收到 final 后自动收起，标题变回
 * 「思考过程」。**它只是展示层**——内容不进 `content`，也不参与答案拼接；
 * 展开/收起是纯本地状态，不写 sessionStorage。
 */
function ThinkingBlock({ text, live }: { text: string; live: boolean }) {
  const [open, setOpen] = useState(live);

  useEffect(() => {
    setOpen(live);
  }, [live]);

  return (
    <section className={`thinking${open ? " open" : ""}`}>
      <button
        type="button"
        className="thinking-head"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
      >
        <Icon name="sparkles" size={14} />
        <span>{live ? "正在思考…" : "思考过程"}</span>
        <Icon name="chevron-right" size={14} />
      </button>
      {open && <div className="thinking-body">{text.trim()}</div>}
    </section>
  );
}

function MessageMeta({ meta, onOpenTrace }: { meta: ChatMeta; onOpenTrace: () => void }) {
  const totalTokens = (meta.tokens.in ?? 0) + (meta.tokens.out ?? 0);
  const trace = meta.plan.map(workerLabel).join(" / ");

  return (
    <div className="meta">
      {trace && <span>路由：{trace}</span>}
      {meta.decision && <span>决策：{decisionLabel(meta.decision)}</span>}
      {typeof meta.elapsedMs === "number" && (
        <span>{(meta.elapsedMs / 1000).toFixed(1)}s</span>
      )}
      {totalTokens > 0 && <span>{totalTokens} token</span>}
      {meta.sources.length > 0 && <span>{meta.sources.length} 个来源</span>}
      <button type="button" onClick={onOpenTrace}>
        查看轨迹
      </button>
    </div>
  );
}
