/** `agent` 是人工客服：回复写进同一个会话，前端要把它和机器人区分开。 */
export type Role = "user" | "assistant" | "agent";

export interface ChatMeta {
  sources: string[];
  plan: string[];
  decision: string;
  tokens: { in?: number; out?: number };
  elapsedMs?: number | null;
  requestId: string;
}

export interface Message {
  id: string;
  role: Role;
  content: string;
  /** 正在流式接收。用来显示光标，也用来禁止重复发送。 */
  streaming?: boolean;
  /** 本标签页内的发送时间，只用于会话轨迹排序。 */
  createdAt?: number;
  /** 过程展示：路由、worker、决策，以及模型返回的思维链。不参与答案拼接。 */
  thinking?: string;
  meta?: ChatMeta;
  /** `role === "agent"` 时是回复的客服名字。 */
  author?: string;
}

/**
 * 一个「本标签页里的会话」。
 *
 * `id` 只在前端用，用来切换列表；`sessionId` 是服务端签发的凭证，
 * 首轮请求之前为空。两者**不能混用**：前者可以重建，后者丢了就接不上
 * 服务端历史。
 */
export interface Conversation {
  id: string;
  sessionId: string;
  title: string;
  messages: Message[];
  updatedAt: number;
  /**
   * 消息是否已经拿到。本地新建的会话默认是 `true`；登录后从服务端列出来的
   * 历史会话先给 `false`（只有标题和条数），切到它时再拉正文。
   */
  loaded?: boolean;
  /** 这个会话最近的一张人工工单（包括已关闭的）；没转过人工是 undefined。 */
  ticket?: TicketInfo | null;
}

/** `GET /sessions` 的一行：侧边栏够用，不带正文。 */
export interface SessionSummary {
  session_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: number;
}

/** `GET /sessions/{id}` 里的一条消息。 */
export interface SessionMessage {
  id: number;
  role: Role;
  content: string;
  created_at: string;
  author?: string;
}

export interface SessionDetail {
  session_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: SessionMessage[];
}

/** 转人工之后落的工单。 */
export interface HandoffInfo {
  ticket_id: string;
  status: "open" | "claimed" | "closed";
  queue_position: number;
}

export interface TicketInfo {
  ticket_id: string;
  status: "open" | "claimed" | "closed";
  reason: string;
  created_at: string;
  updated_at: string;
  claimed_by_name: string | null;
  queue_position: number;
}

export interface TicketSummary extends TicketInfo {
  session_id: string;
  title: string;
}

export interface TicketMessage {
  id: number;
  role: Role;
  content: string;
  created_at: string;
  author?: string;
}

/** `GET /tickets/{id}`：工单 + 用户会话的上下文，客服接手前要看前因。 */
export interface TicketDetail extends TicketSummary {
  transcript: TicketMessage[];
}

export type Theme = "light" | "dark" | "system";

export interface Health {
  status: string;
  workers: string[];
  order_db: string;
  max_rounds: number;
  demo: boolean;
  auth?: boolean;
}

interface StartEvent {
  type: "start";
  request_id: string;
  session_id: string;
}

interface ThinkingEvent {
  type: "thinking";
  /** stage：路由 / worker / 决策的真实状态；reasoning：模型思维链。 */
  kind: "stage" | "reasoning";
  text: string;
}

interface DeltaEvent {
  type: "delta";
  text: string;
}

/** 重路由：上一轮已经流出的正文作废，最终以新一轮和 final 为准。 */
interface ResetEvent {
  type: "reset";
}

interface FinalEvent {
  type: "final";
  request_id: string;
  session_id: string;
  answer: string;
  sources: string[];
  decision: string;
  plan: string[];
  tokens: { in?: number; out?: number };
  /** 这一轮转人工落的工单；没转人工或当前环境没有人工客服时是 null。 */
  handoff?: HandoffInfo | null;
  /** 完整的思考过程；客户端用它覆盖自己累加的那份，避免掉帧。 */
  thinking?: string;
  elapsed_ms: number | null;
}

interface ErrorEvent {
  type: "error";
  error: string;
  request_id: string;
}

export type ChatEvent =
  | StartEvent
  | ThinkingEvent
  | DeltaEvent
  | ResetEvent
  | FinalEvent
  | ErrorEvent;
