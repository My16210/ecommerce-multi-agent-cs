import type { Conversation, Health } from "../types";
import { Icon, type IconName } from "./Icon";

interface SidebarProps {
  conversations: Conversation[];
  activeId: string;
  collapsed: boolean;
  health: Health | null;
  onNew: () => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onToggle: () => void;
  onSettings: () => void;
}

const CAPABILITIES: Array<{ key: string; label: string; icon: IconName }> = [
  { key: "knowledge", label: "知识库", icon: "shield" },
  { key: "order", label: "订单与物流", icon: "package" },
  { key: "analytics", label: "数据分析", icon: "chart" },
];

function timeLabel(timestamp: number): string {
  const elapsed = Date.now() - timestamp;
  if (elapsed < 60_000) return "刚刚";
  if (elapsed < 3_600_000) return `${Math.floor(elapsed / 60_000)} 分钟前`;

  const date = new Date(timestamp);
  const today = new Date();
  if (date.toDateString() === today.toDateString()) {
    return date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
  }
  return `${date.getMonth() + 1}/${date.getDate()}`;
}

export function Sidebar({
  conversations,
  activeId,
  collapsed,
  health,
  onNew,
  onSelect,
  onDelete,
  onToggle,
  onSettings,
}: SidebarProps) {
  const online = new Set(health?.workers ?? []);

  return (
    <aside className={`sidebar${collapsed ? " collapsed" : ""}`} aria-label="会话侧边栏">
      <div className="sidebar-head">
        <div className="brand">
          <span className="logo" aria-hidden="true">
            E
          </span>
          <span className="sidebar-text brand-copy">
            <strong>电商智能客服</strong>
            <small>Supervisor · 三个单元</small>
          </span>
        </div>
        <button
          className="icon-btn collapse-btn"
          type="button"
          onClick={onToggle}
          aria-label={collapsed ? "展开侧边栏" : "收起侧边栏"}
          title={collapsed ? "展开侧边栏" : "收起侧边栏"}
        >
          <Icon name={collapsed ? "chevron-right" : "chevron-left"} size={16} />
        </button>
      </div>

      <button className="new-chat" type="button" onClick={onNew} title="新对话">
        <Icon name="plus" size={18} />
        <span className="sidebar-text">新对话</span>
      </button>

      <div className="sidebar-scroll">
        <div className="sidebar-section sidebar-text">
          <span>工作区</span>
        </div>
        <div className="workspace" title="本地客服工作区">
          <span className={`workspace-dot${health ? " online" : ""}`} aria-hidden="true" />
          <span className="sidebar-text">
            <strong>本地客服</strong>
            <small>{health?.demo ? "只读演示" : health ? "已连接" : "接口未连接"}</small>
          </span>
        </div>

        <div className="sidebar-section sidebar-text">
          <span>会话</span>
          <span className="count">{conversations.length}</span>
        </div>
        <nav className="session-list" aria-label="会话列表">
          {conversations.map((conversation) => (
            <div
              className={`session-row${conversation.id === activeId ? " active" : ""}`}
              key={conversation.id}
            >
              <button
                className="session-main"
                type="button"
                onClick={() => onSelect(conversation.id)}
                title={conversation.title}
                aria-current={conversation.id === activeId ? "page" : undefined}
              >
                <span className="session-icon" aria-hidden="true">
                  <Icon name="chat" size={15} />
                </span>
                <span className="sidebar-text session-copy">
                  <span className="session-title">{conversation.title}</span>
                  <span className="session-time">{timeLabel(conversation.updatedAt)}</span>
                </span>
              </button>
              <button
                className="session-delete"
                type="button"
                onClick={() => onDelete(conversation.id)}
                aria-label={`移除会话：${conversation.title}`}
                title="移除会话"
              >
                <Icon name="trash" size={14} />
              </button>
            </div>
          ))}
        </nav>

        <div className="sidebar-section sidebar-text">
          <span>能力</span>
        </div>
        <div className="cap-list">
          {CAPABILITIES.map((capability) => (
            <div className="cap-row" key={capability.key} title={capability.label}>
              <span className="cap-icon" aria-hidden="true">
                <Icon name={capability.icon} size={15} />
              </span>
              <span className="sidebar-text">{capability.label}</span>
              <span
                className={`cap-dot${online.has(capability.key) ? " online" : ""}`}
                aria-label={online.has(capability.key) ? "在线" : "未启用"}
              />
            </div>
          ))}
        </div>
      </div>

      <div className="sidebar-foot">
        <button className="side-action" type="button" onClick={onSettings} title="连接设置">
          <Icon name="settings" size={17} />
          <span className="sidebar-text">连接设置</span>
        </button>
      </div>
    </aside>
  );
}
