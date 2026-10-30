import { useEffect } from "react";

import type { Health } from "../types";
import { Icon } from "./Icon";

interface SettingsDialogProps {
  open: boolean;
  health: Health | null;
  whoami: string;
  username: string;
  password: string;
  apiKey: string;
  inviteCode: string;
  loginBusy: boolean;
  onUsername: (value: string) => void;
  onPassword: (value: string) => void;
  onApiKey: (value: string) => void;
  onInviteCode: (value: string) => void;
  onLogin: () => void;
  /** 主动退出：吊销服务端 token 并清本地凭证。 */
  onLogout: () => void;
  onSave: () => void;
  onClose: () => void;
}

export function SettingsDialog({
  open,
  health,
  whoami,
  username,
  password,
  apiKey,
  inviteCode,
  loginBusy,
  onUsername,
  onPassword,
  onApiKey,
  onInviteCode,
  onLogin,
  onLogout,
  onSave,
  onClose,
}: SettingsDialogProps) {
  useEffect(() => {
    if (!open) return;
    function onKey(event: KeyboardEvent): void {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="dialog-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section className="dialog" role="dialog" aria-modal="true" aria-labelledby="settings-title">
        <header className="dialog-head">
          <div>
            <h2 id="settings-title">连接设置</h2>
            <p>凭证只存在本标签页的 sessionStorage。</p>
          </div>
          <button className="icon-btn" type="button" onClick={onClose} aria-label="关闭设置">
            <Icon name="close" size={17} />
          </button>
        </header>

        <div className="dialog-body">
          {health?.auth && (
            <section className="settings-section">
              <div className="settings-title">
                <strong>账号登录</strong>
                <span>{whoami ? `已登录：${whoami}` : "可选"}</span>
              </div>
              <div className="field-row">
                <label>
                  用户名
                  <input
                    autoComplete="username"
                    value={username}
                    onChange={(event) => onUsername(event.target.value)}
                  />
                </label>
                <label>
                  密码
                  <input
                    type="password"
                    autoComplete="current-password"
                    value={password}
                    onChange={(event) => onPassword(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") onLogin();
                    }}
                  />
                </label>
              </div>
              <div className="settings-inline">
                <p>密码只在这次请求里出现，不会被保存。登录后侧边栏会恢复该账号的历史会话。</p>
                {whoami && (
                  <button className="ghost-btn" type="button" onClick={onLogout}>
                    退出登录
                  </button>
                )}
                <button
                  className="secondary-btn"
                  type="button"
                  onClick={onLogin}
                  disabled={loginBusy || !username.trim() || !password}
                >
                  {loginBusy ? "登录中…" : "登录"}
                </button>
              </div>
            </section>
          )}

          <section className="settings-section">
            <div className="settings-title">
              <strong>访问凭证</strong>
              <span>服务端没配鉴权就留空</span>
            </div>
            <label>
              API Key / token
              <input
                type="password"
                autoComplete="off"
                value={apiKey}
                onChange={(event) => onApiKey(event.target.value)}
                placeholder="不登录时可直接粘贴 token"
              />
            </label>
            <label>
              演示邀请码
              <input
                type="password"
                autoComplete="off"
                value={inviteCode}
                onChange={(event) => onInviteCode(event.target.value)}
                placeholder="只在只读演示环境需要"
              />
            </label>
          </section>
        </div>

        <footer className="dialog-foot">
          <p>不进 URL、不写 localStorage；关闭标签页即失效。</p>
          <div>
            <button className="ghost-btn" type="button" onClick={onClose}>
              取消
            </button>
            <button className="primary-btn" type="button" onClick={onSave}>
              保存设置
            </button>
          </div>
        </footer>
      </section>
    </div>
  );
}
