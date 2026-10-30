export const WORKER_LABELS: Record<string, string> = {
  knowledge: "知识库",
  order: "订单",
  analytics: "数据分析",
};

export const DECISION_LABELS: Record<string, string> = {
  answer: "直接回答",
  re_route: "换一个单元",
  human: "转人工",
};

export function workerLabel(name: string): string {
  return WORKER_LABELS[name] ?? name;
}

export function decisionLabel(name: string): string {
  return DECISION_LABELS[name] ?? name;
}
