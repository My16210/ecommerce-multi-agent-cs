import { useState } from "react";

import { decisionLabel, workerLabel } from "../labels";
import type { ChatMeta, Health, Message } from "../types";
import { Icon } from "./Icon";

interface TracePanelProps {
  health: Health | null;
  messages: Message[];
  onClose: () => void;
}

function CopyId({ value }: { value: string }) {
  const [copied, setCopied] = useState(false);

  async function copy(): Promise<void> {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // 没有剪贴板权限时保留可选中文本，不弹错误。
    }
  }

  return (
    <span className="request-id">
      <code>{value}</code>
      <button type="button" onClick={() => void copy()} aria-label="复制 request_id">
        <Icon name={copied ? "check" : "copy"} size={13} />
      </button>
    </span>
  );
}

function MetaCard({ meta, turn }: { meta: ChatMeta; turn: number }) {
  const total = (meta.tokens.in ?? 0) + (meta.tokens.out ?? 0);

  return (
    <article className="trace-card">
      <header>
        <span className="trace-turn">第 {turn} 次回答</span>
        {typeof meta.elapsedMs === "number" && (
          <span className="trace-time">{(meta.elapsedMs / 1000).toFixed(1)}s</span>
        )}
      </header>

      <dl className="trace-grid">
        <div>
          <dt>路由</dt>
          <dd className="trace-chips">
            {meta.plan.length > 0 ? (
              meta.plan.map((name) => <span key={name}>{workerLabel(name)}</span>)
            ) : (
              <span className="empty-value">—</span>
            )}
          </dd>
        </div>
        <div>
          <dt>决策</dt>
          <dd>{meta.decision ? decisionLabel(meta.decision) : "—"}</dd>
        </div>
        <div>
          <dt>Token</dt>
          <dd>
            {total > 0
              ? `${total}（入 ${meta.tokens.in ?? 0} / 出 ${meta.tokens.out ?? 0}）`
              : "—"}
          </dd>
        </div>
        <div>
          <dt>request_id</dt>
          <dd>{meta.requestId ? <CopyId value={meta.requestId} /> : "—"}</dd>
        </div>
      </dl>

      {meta.sources.length > 0 && (
        <div className="trace-sources">
          <span className="trace-label">来源</span>
          <ul>
            {meta.sources.map((source) => (
              <li key={source}>{source}</li>
            ))}
          </ul>
        </div>
      )}
    </article>
  );
}

export function TracePanel({ health, messages, onClose }: TracePanelProps) {
  const traced = messages.filter((message) => message.role === "assistant" && message.meta);
  const latestFirst = [...traced].reverse();

  return (
    <aside className="rail" aria-label="会话轨迹">
      <header className="rail-head">
        <div>
          <strong>会话轨迹</strong>
          <small>路由、决策与来源</small>
        </div>
        <button className="icon-btn" type="button" onClick={onClose} aria-label="关闭轨迹面板">
          <Icon name="close" size={16} />
        </button>
      </header>

      <div className="rail-scroll">
        <section className="rail-section">
          <h3>运行状态</h3>
          <dl className="health-grid">
            <div>
              <dt>接口</dt>
              <dd className={health ? "ok" : "bad"}>{health ? "已连接" : "未连接"}</dd>
            </div>
            <div>
              <dt>订单库</dt>
              <dd>{health?.order_db ?? "—"}</dd>
            </div>
            <div>
              <dt>最大轮数</dt>
              <dd>{health?.max_rounds ?? "—"}</dd>
            </div>
            <div>
              <dt>模式</dt>
              <dd>{health ? (health.demo ? "只读演示" : "常规") : "—"}</dd>
            </div>
          </dl>
          {health && (
            <div className="rail-workers">
              {health.workers.map((worker) => (
                <span key={worker} className="worker-pill">
                  <span className="cap-dot online" aria-hidden="true" />
                  {workerLabel(worker)}
                </span>
              ))}
            </div>
          )}
        </section>

        <section className="rail-section">
          <h3>本轮明细</h3>
          {latestFirst.length === 0 ? (
            <div className="rail-empty">
              <Icon name="panel" size={22} />
              <p>发一条消息后，这里会显示这一轮走了哪个单元、为什么这么汇总。</p>
            </div>
          ) : (
            <div className="trace-list">
              {latestFirst.map((message, index) => (
                <MetaCard
                  key={message.id}
                  meta={message.meta as ChatMeta}
                  turn={latestFirst.length - index}
                />
              ))}
            </div>
          )}
        </section>
      </div>
    </aside>
  );
}
