export type IconName =
  | "plus"
  | "chat"
  | "settings"
  | "panel"
  | "sun"
  | "moon"
  | "monitor"
  | "send"
  | "stop"
  | "trash"
  | "menu"
  | "chevron-right"
  | "chevron-left"
  | "copy"
  | "check"
  | "sparkles"
  | "package"
  | "shield"
  | "chart"
  | "headset"
  | "inbox"
  | "close";

/**
 * 图标全部内联，不引第三方图标库。
 *
 * 理由和前端其它依赖一样：这个页面要能在 `npm run build` 之后完全离线托管，
 * 为一个箭头多装一个包，收益只有省几行 SVG。
 */
export function Icon({ name, size = 18 }: { name: IconName; size?: number }) {
  return (
    <svg
      viewBox="0 0 24 24"
      width={size}
      height={size}
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      {name === "plus" && <path d="M12 5v14M5 12h14" />}
      {name === "chat" && (
        <path d="M20 15a3 3 0 0 1-3 3H8l-4 3V6a3 3 0 0 1 3-3h10a3 3 0 0 1 3 3v9Z" />
      )}
      {name === "settings" && (
        <>
          <path d="M4 7h9M17 7h3M4 17h3M11 17h9" />
          <circle cx="15" cy="7" r="2" />
          <circle cx="9" cy="17" r="2" />
        </>
      )}
      {name === "panel" && (
        <>
          <rect x="3" y="4" width="18" height="16" rx="2" />
          <path d="M15 4v16" />
        </>
      )}
      {name === "sun" && (
        <>
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
        </>
      )}
      {name === "moon" && <path d="M21 12.8A8.5 8.5 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z" />}
      {name === "monitor" && (
        <>
          <rect x="3" y="4" width="18" height="12" rx="2" />
          <path d="M8 20h8M12 16v4" />
        </>
      )}
      {name === "send" && <path d="M12 19V5M5 12l7-7 7 7" />}
      {name === "stop" && <rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor" stroke="none" />}
      {name === "trash" && (
        <path d="M4 7h16M9 7V5h6v2M6 7l1 13h10l1-13M10 11v5M14 11v5" />
      )}
      {name === "menu" && <path d="M4 7h16M4 12h16M4 17h16" />}
      {name === "chevron-right" && <path d="m9 5 7 7-7 7" />}
      {name === "chevron-left" && <path d="m15 5-7 7 7 7" />}
      {name === "copy" && (
        <>
          <rect x="9" y="9" width="11" height="11" rx="2" />
          <path d="M5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1" />
        </>
      )}
      {name === "check" && <path d="m5 12 4.5 4.5L19 7" />}
      {name === "sparkles" && (
        <>
          <path d="m12 3 1.6 4.4L18 9l-4.4 1.6L12 15l-1.6-4.4L6 9l4.4-1.6L12 3Z" />
          <path d="m19 15 .8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8L19 15Z" />
        </>
      )}
      {name === "package" && (
        <>
          <path d="m12 3 8 4.5v9L12 21l-8-4.5v-9L12 3Z" />
          <path d="m4 7.5 8 4.5 8-4.5M12 12v9M8 5.2l8 4.5" />
        </>
      )}
      {name === "shield" && (
        <>
          <path d="M12 3 5 6v6c0 4.4 3 8.2 7 9 4-.8 7-4.6 7-9V6l-7-3Z" />
          <path d="m9 12 2 2 4-4" />
        </>
      )}
      {name === "chart" && <path d="M4 20V10M10 20V4M16 20v-7M22 20H2" />}
      {name === "headset" && (
        <>
          <path d="M4 14v-2a8 8 0 0 1 16 0v2" />
          <path d="M4 14a2 2 0 0 1 2-2h1v6H6a2 2 0 0 1-2-2v-2ZM20 14a2 2 0 0 0-2-2h-1v6h1a2 2 0 0 0 2-2v-2Z" />
          <path d="M20 18v1a3 3 0 0 1-3 3h-3" />
        </>
      )}
      {name === "inbox" && (
        <>
          <path d="M4 13h4l1.5 3h5L16 13h4" />
          <path d="M5 5h14l2 8v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-5l2-8Z" />
        </>
      )}
      {name === "close" && <path d="m6 6 12 12M18 6 6 18" />}
    </svg>
  );
}
