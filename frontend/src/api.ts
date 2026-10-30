import type { ChatEvent, Health } from "./types";

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
  try {
    const body = await response.json();
    if (typeof body?.error === "string" && body.error) message = body.error;
    if (typeof body?.request_id === "string") requestId = body.request_id;
  } catch {
    // 响应不是 JSON（例如反向代理返回的 HTML 错误页）：保留状态码那句话。
  }
  if (response.status === 401) message = "API Key 不正确或没有提供。";
  if (response.status === 403) message = message || "邀请码不正确。";
  return new ApiError(message, response.status, requestId);
}

export async function fetchHealth(signal?: AbortSignal): Promise<Health> {
  const response = await fetch("/health", { signal });
  if (!response.ok) throw await readError(response);
  return (await response.json()) as Health;
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
  if (options.apiKey) headers["X-API-Key"] = options.apiKey;
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
