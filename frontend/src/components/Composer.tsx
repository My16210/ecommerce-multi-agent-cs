import { useEffect, useRef } from "react";

import { Icon } from "./Icon";

interface ComposerProps {
  value: string;
  busy: boolean;
  onChange: (value: string) => void;
  onSend: () => void;
  onStop: () => void;
}

export function Composer({ value, busy, onChange, onSend, onStop }: ComposerProps) {
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    const input = inputRef.current;
    if (!input) return;
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 200)}px`;
  }, [value]);

  return (
    <footer className="composer">
      <div className="composer-box">
        <label className="sr-only" htmlFor="question">
          输入问题
        </label>
        <textarea
          id="question"
          ref={inputRef}
          rows={1}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => {
            if (
              event.key === "Enter" &&
              !event.shiftKey &&
              !event.nativeEvent.isComposing
            ) {
              event.preventDefault();
              onSend();
            }
          }}
          placeholder="给客服助手发消息，例如：订单 52644 到哪了"
          maxLength={2000}
        />

        <div className="composer-bar">
          <div className="composer-meta">
            <span className="mode-pill">
              <Icon name="sparkles" size={14} />
              Supervisor
            </span>
            <span className="soft-pill">自动路由</span>
          </div>

          <div className="composer-actions">
            <span className="enter-hint">Enter 发送</span>
            {busy ? (
              <button
                className="send-btn stop"
                type="button"
                onClick={onStop}
                aria-label="停止生成"
                title="停止生成"
              >
                <Icon name="stop" size={16} />
              </button>
            ) : (
              <button
                className="send-btn"
                type="button"
                onClick={onSend}
                disabled={!value.trim()}
                aria-label="发送"
                title="发送"
              >
                <Icon name="send" size={18} />
              </button>
            )}
          </div>
        </div>
      </div>

      <p className="composer-note">
        会话号是凭证，只存在本标签页。Enter 发送，Shift + Enter 换行。
      </p>
    </footer>
  );
}
