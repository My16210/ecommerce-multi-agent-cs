export type Role = "user" | "assistant";

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
  meta?: ChatMeta;
}

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

interface DeltaEvent {
  type: "delta";
  text: string;
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
  elapsed_ms: number | null;
}

interface ErrorEvent {
  type: "error";
  error: string;
  request_id: string;
}

export type ChatEvent = StartEvent | DeltaEvent | FinalEvent | ErrorEvent;
