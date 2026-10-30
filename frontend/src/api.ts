import type {
  ChatEvent,
  Health,
  SessionDetail,
  SessionSummary,
  TicketDetail,
  TicketInfo,
  TicketSummary,
} from "./types";

/**
 * 接口错误。`requestId` 是给用户拿去对服务端日志的——和 api.py 的错误体
 * 是同一个约定，**不把异常原文塞进界面**。
 */
export class ApiError extends Error {
  readonly status: number;
  readonly requestId: string;

  constructor(message: string, status: number, requestId = "") {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.requestId = requestId;
  }
}

export interface StreamChatOptions {
  question: string;
  sessionId: string;
  apiKey: string;
  inviteCode: string;
  signal: AbortSignal;
}

async function readError(response: Response): Promise<ApiError> {
  let message = `请求失败（HTTP ${response.status}）`;
  let requestId = "";
  let fromServer = false;
  try {
    const body = await response.json();
    if (typeof body?.error === "string" && body.error) {
      message = body.error;
      fromServer = true;
    }
    if (typeof body?.request_id === "string") requestId = body.request_id;
  } catch {
    // 响应不是 JSON（例如反向代理返回的 HTML 错误页）：保留状态码那句话。
  }
  // 服务端已经给了具体原因（例如“用户名或密码不正确”）就不要覆盖它。
  // 这句兜底只留给没有结构化错误体的 401——通常来自代理或网关。
  if (response.status === 401 && !fromServer) message = "API Key 不正确或没有提供。";
  if (response.status === 403 && !fromServer) message = "邀请码不正确。";
  return new ApiError(message, response.status, requestId);
}

export interface LoginResult {
  token: string;
  expires_at: string;
  name: string;
  role: string;
}

export async function fetchHealth(signal?: AbortSignal): Promise<Health> {
  const response = await fetch("/health", { signal });
  if (!response.ok) throw await readError(response);
  return (await response.json()) as Health;
}

/**
 * 用户名 + 密码换短期 token。密码只在这次请求里出现，**不写进 sessionStorage**；
 * 存下来的只有返回的 token（它是凭证，和 API Key 同一个位置）。
 */
export async function login(username: string, password: string): Promise<LoginResult> {
  const response = await fetch("/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!response.ok) throw await readError(response);
  return (await response.json()) as LoginResult;
}

export interface LogoutResult {
  /** 服务端在这次请求里真的吊销了 token（登录会话）。 */
  revoked: boolean;
  /** token 在调用前就已经失效（401），不算"没吊销成功"。 */
  alreadyInvalid: boolean;
}

/**
 * 主动登出：让服务端立刻吊销这枚登录 token，而不是等它到期。
 *
 * 返回服务端的结果，而不是只抛错：**长期 API token 不在 `/auth/logout`
 * 的吊销范围内**（它没有登录会话），服务端会返回 `revoked: false`。界面
 * 必须把这件事说清楚，否则用户会以为自己已经安全退出了，实际上那枚
 * token 仍然能用到过期。
 *
 * 401 说明 token 已经无效，按"无需再吊销"处理；其它错误照常抛。
 */
export async function logout(token: string): Promise<LogoutResult> {
  if (!token) return { revoked: false, alreadyInvalid: true };
  const response = await fetch("/auth/logout", {
    method: "POST",
    headers: sessionHeaders(token),
  });
  if (response.status === 401) return { revoked: false, alreadyInvalid: true };
  if (!response.ok) throw await readError(response);
  try {
    const body = (await response.json()) as { revoked?: unknown };
    return { revoked: body?.revoked === true, alreadyInvalid: false };
  } catch {
    // 服务端返回了 2xx 但没有 JSON：不能假装吊销成功。
    return { revoked: false, alreadyInvalid: false };
  }
}

/**
 * 历史会话接口。token 同时放 `Authorization` 和 `X-API-Key`：服务端在开了
 * 身份层时把两者都当 bearer 看，这样登录 token 和旧的手工 token 都能用。
 */
function sessionHeaders(token: string): Record<string, string> {
  return { Authorization: `Bearer ${token}`, "X-API-Key": token };
}

/** 当前账号的会话列表（只有标题、时间和条数，不含正文）。 */
export async function fetchSessions(
  token: string,
  signal?: AbortSignal,
): Promise<SessionSummary[]> {
  const response = await fetch("/sessions", { headers: sessionHeaders(token), signal });
  if (!response.ok) throw await readError(response);
  const body = (await response.json()) as { sessions?: SessionSummary[] };
  return Array.isArray(body.sessions) ? body.sessions : [];
}

/** 一个会话的消息明细。 */
export async function fetchSession(
  sessionId: string,
  token: string,
  signal?: AbortSignal,
): Promise<SessionDetail> {
  const response = await fetch(`/sessions/${encodeURIComponent(sessionId)}`, {
    headers: sessionHeaders(token),
    signal,
  });
  if (!response.ok) throw await readError(response);
  return (await response.json()) as SessionDetail;
}

/** 删除自己的会话（消息和摘要一起删）。 */
export async function deleteSession(sessionId: string, token: string): Promise<number> {
  const response = await fetch(`/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
    headers: sessionHeaders(token),
  });
  if (!response.ok) throw await readError(response);
  const body = (await response.json()) as { deleted?: number };
  return body.deleted ?? 0;
}

/* ── 人工工单 ───────────────────────────────────────────────────────── */

/** 这个会话最近一张工单（包括已关闭的）；从没转过人工是 null。 */
export async function fetchTicket(
  sessionId: string,
  token: string,
  signal?: AbortSignal,
): Promise<TicketInfo | null> {
  const response = await fetch(
    `/sessions/${encodeURIComponent(sessionId)}/ticket`,
    { headers: sessionHeaders(token), signal },
  );
  if (!response.ok) throw await readError(response);
  const body = (await response.json()) as { ticket?: TicketInfo | null };
  return body.ticket ?? null;
}

/** 用户主动点「转人工」。幂等：已有未关闭工单时返回同一张。 */
export async function openHandoff(sessionId: string, token: string): Promise<TicketInfo> {
  const response = await fetch(
    `/sessions/${encodeURIComponent(sessionId)}/handoff`,
    { method: "POST", headers: sessionHeaders(token) },
  );
  if (!response.ok) throw await readError(response);
  const body = (await response.json()) as { ticket?: TicketInfo | null };
  if (!body.ticket) throw new ApiError("工单没有创建成功。", 0);
  return body.ticket;
}

/** 客服的排队列表。status 见服务端：open / claimed / closed / all。 */
export async function fetchTickets(
  token: string,
  status = "open",
  signal?: AbortSignal,
): Promise<TicketSummary[]> {
  const response = await fetch(
    `/tickets?status=${encodeURIComponent(status)}`,
    { headers: sessionHeaders(token), signal },
  );
  if (!response.ok) throw await readError(response);
  const body = (await response.json()) as { tickets?: TicketSummary[] };
  return Array.isArray(body.tickets) ? body.tickets : [];
}

export async function fetchTicketDetail(
  ticketId: string,
  token: string,
  signal?: AbortSignal,
): Promise<TicketDetail> {
  const response = await fetch(`/tickets/${encodeURIComponent(ticketId)}`, {
    headers: sessionHeaders(token),
    signal,
  });
  if (!response.ok) throw await readError(response);
  return (await response.json()) as TicketDetail;
}

export async function claimTicket(ticketId: string, token: string): Promise<TicketInfo> {
  const response = await fetch(`/tickets/${encodeURIComponent(ticketId)}/claim`, {
    method: "POST",
    headers: sessionHeaders(token),
  });
  if (!response.ok) throw await readError(response);
  const body = (await response.json()) as { ticket: TicketInfo };
  return body.ticket;
}

export async function replyTicket(
  ticketId: string,
  token: string,
  content: string,
): Promise<void> {
  const response = await fetch(`/tickets/${encodeURIComponent(ticketId)}/reply`, {
    method: "POST",
    headers: { ...sessionHeaders(token), "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
  });
  if (!response.ok) throw await readError(response);
}

export async function closeTicket(ticketId: string, token: string): Promise<TicketInfo> {
  const response = await fetch(`/tickets/${encodeURIComponent(ticketId)}/close`, {
    method: "POST",
    headers: sessionHeaders(token),
  });
  if (!response.ok) throw await readError(response);
  const body = (await response.json()) as { ticket: TicketInfo };
  return body.ticket;
}

/** 转派 / 放回队列（admin）。`assignee=null` 表示放回排队。 */
export async function transferTicket(
  ticketId: string,
  token: string,
  assignee: string | null,
): Promise<TicketInfo> {
  const response = await fetch(`/tickets/${encodeURIComponent(ticketId)}/transfer`, {
    method: "POST",
    headers: { ...sessionHeaders(token), "Content-Type": "application/json" },
    body: JSON.stringify({ assignee }),
  });
  if (!response.ok) throw await readError(response);
  const body = (await response.json()) as { ticket: TicketInfo };
  return body.ticket;
}

/**
 * 调 `/chat/stream`，把 SSE 帧逐条吐出来。
 *
 * 用 `fetch` 而不是 `EventSource`：EventSource 只能发 GET、不能带请求体和
 * 自定义头，而 `/chat/stream` 是 POST + `X-API-Key` / `X-Invite-Code`。
 *
 * `final.answer` 是权威版本：汇总中途失败时服务端会退回拼接，前面的 delta
 * 拼起来可能和它不一致。所以调用方收到 final 时应当**整体替换**，而不是
 * 继续追加。
 */
export async function* streamChat(options: StreamChatOptions): AsyncGenerator<ChatEvent> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (options.apiKey) {
    // 登录 token 和手工粘贴的 token 走同一个字段；服务端开了身份层时
    // `Authorization` 和 `X-API-Key` 都认，这里两个都带，省掉模式判断。
    headers["X-API-Key"] = options.apiKey;
    headers["Authorization"] = `Bearer ${options.apiKey}`;
  }
  if (options.inviteCode) headers["X-Invite-Code"] = options.inviteCode;

  const response = await fetch("/chat/stream", {
    method: "POST",
    headers,
    body: JSON.stringify({ question: options.question, session_id: options.sessionId }),
    signal: options.signal,
  });
  if (!response.ok) throw await readError(response);
  if (!response.body) throw new ApiError("浏览器不支持流式响应。", 0);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    // SSE 的行分隔符允许 CRLF；服务端现在用 LF，这里两种都认。
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

    let boundary = buffer.indexOf("\n\n");
    while (boundary >= 0) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const line = frame.split("\n").find((item) => item.startsWith("data:"));
      if (line) {
        const payload = line.slice(5).trim();
        if (payload) yield JSON.parse(payload) as ChatEvent;
      }
      boundary = buffer.indexOf("\n\n");
    }
  }
}
