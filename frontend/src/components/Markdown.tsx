import { Fragment, useState, type ReactNode } from "react";

import { Icon } from "./Icon";

/**
 * 一个**不解析 HTML** 的最小 Markdown 渲染器。
 *
 * 为什么不用 `react-markdown` + `rehype-raw`：
 * 模型输出是**不可信文本**。一旦允许原始 HTML，`<img onerror=...>` 这类
 * 内容就能直接进 DOM；而这个页面里同时放着会话号（凭证）和 API Key。
 * 这里只生成 React 元素——文本始终作为文本节点插入，不拼 DOM 字符串，
 * 链接也只放行 http/https。
 *
 * 支持的语法刻意很少：标题、段落、列表、引用、分割线、围栏代码块、
 * 行内代码、加粗、链接。够客服回答用，也够审计。
 */

type Block =
  | { kind: "code"; lang: string; code: string }
  | { kind: "heading"; level: number; text: string }
  | { kind: "ul"; items: string[] }
  | { kind: "ol"; items: string[] }
  | { kind: "quote"; text: string }
  | { kind: "hr" }
  | { kind: "p"; text: string };

const FENCE = /^```([\w+-]*)\s*$/;
const HEADING = /^(#{1,4})\s+(.*)$/;
const RULE = /^(-{3,}|\*{3,}|_{3,})$/;
const QUOTE = /^>\s?/;
const BULLET = /^[-*+]\s+/;
const NUMBER = /^\d+[.)]\s+/;

function isBlockStart(line: string): boolean {
  return (
    FENCE.test(line) ||
    HEADING.test(line) ||
    RULE.test(line) ||
    QUOTE.test(line) ||
    BULLET.test(line) ||
    NUMBER.test(line)
  );
}

function parseBlocks(source: string): Block[] {
  const lines = source.replace(/\r\n/g, "\n").split("\n");
  const blocks: Block[] = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];

    if (!line.trim()) {
      index += 1;
      continue;
    }

    const fence = FENCE.exec(line);
    if (fence) {
      const lang = fence[1] ?? "";
      const code: string[] = [];
      index += 1;
      while (index < lines.length && !/^```\s*$/.test(lines[index])) {
        code.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) index += 1;
      blocks.push({ kind: "code", lang, code: code.join("\n") });
      continue;
    }

    const heading = HEADING.exec(line);
    if (heading) {
      blocks.push({
        kind: "heading",
        level: heading[1].length,
        text: heading[2],
      });
      index += 1;
      continue;
    }

    if (RULE.test(line)) {
      blocks.push({ kind: "hr" });
      index += 1;
      continue;
    }

    if (QUOTE.test(line)) {
      const quote: string[] = [];
      while (index < lines.length && QUOTE.test(lines[index])) {
        quote.push(lines[index].replace(QUOTE, ""));
        index += 1;
      }
      blocks.push({ kind: "quote", text: quote.join("\n") });
      continue;
    }

    if (BULLET.test(line)) {
      const items: string[] = [];
      while (index < lines.length && BULLET.test(lines[index])) {
        items.push(lines[index].replace(BULLET, ""));
        index += 1;
      }
      blocks.push({ kind: "ul", items });
      continue;
    }

    if (NUMBER.test(line)) {
      const items: string[] = [];
      while (index < lines.length && NUMBER.test(lines[index])) {
        items.push(lines[index].replace(NUMBER, ""));
        index += 1;
      }
      blocks.push({ kind: "ol", items });
      continue;
    }

    const paragraph: string[] = [];
    while (index < lines.length && lines[index].trim() && !isBlockStart(lines[index])) {
      paragraph.push(lines[index]);
      index += 1;
    }
    blocks.push({ kind: "p", text: paragraph.join("\n") });
  }

  return blocks;
}

function safeHref(raw: string): string | null {
  try {
    const url = new URL(raw);
    return url.protocol === "http:" || url.protocol === "https:" ? url.href : null;
  } catch {
    return null;
  }
}

function inline(text: string, prefix: string): ReactNode[] {
  const pieces = text.split(/(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))/g);
  return pieces.map((piece, i) => {
    const key = `${prefix}-${i}`;

    if (piece.startsWith("**") && piece.endsWith("**") && piece.length > 4) {
      return <strong key={key}>{piece.slice(2, -2)}</strong>;
    }

    if (piece.startsWith("`") && piece.endsWith("`") && piece.length > 2) {
      return (
        <code className="inline-code" key={key}>
          {piece.slice(1, -1)}
        </code>
      );
    }

    const link = /^\[([^\]]+)\]\(([^)]+)\)$/.exec(piece);
    if (link) {
      const href = safeHref(link[2].trim());
      if (href) {
        return (
          <a key={key} href={href} target="_blank" rel="noreferrer noopener">
            {link[1]}
          </a>
        );
      }
    }

    return (
      <Fragment key={key}>
        {piece.split("\n").map((line, j) => (
          <Fragment key={`${key}-${j}`}>
            {j > 0 && <br />}
            {line}
          </Fragment>
        ))}
      </Fragment>
    );
  });
}

function CodeBlock({ code, lang }: { code: string; lang: string }) {
  const [copied, setCopied] = useState(false);

  async function copy(): Promise<void> {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // 非安全上下文或用户拒绝剪贴板权限：不弹错误，代码本身仍然可选中。
    }
  }

  return (
    <div className="code-block">
      <div className="code-head">
        <span>{lang || "text"}</span>
        <button type="button" onClick={() => void copy()} aria-label="复制代码">
          <Icon name={copied ? "check" : "copy"} size={14} />
          {copied ? "已复制" : "复制"}
        </button>
      </div>
      <pre>
        <code>{code}</code>
      </pre>
    </div>
  );
}

export function Markdown({ text }: { text: string }) {
  const blocks = parseBlocks(text);

  return (
    <div className="markdown">
      {blocks.map((block, index) => {
        const key = `block-${index}`;
        switch (block.kind) {
          case "code":
            return <CodeBlock key={key} code={block.code} lang={block.lang} />;
          case "heading": {
            const Tag = `h${Math.min(block.level + 2, 6)}` as "h3" | "h4" | "h5" | "h6";
            return <Tag key={key}>{inline(block.text, key)}</Tag>;
          }
          case "ul":
            return (
              <ul key={key}>
                {block.items.map((item, i) => (
                  <li key={`${key}-${i}`}>{inline(item, `${key}-${i}`)}</li>
                ))}
              </ul>
            );
          case "ol":
            return (
              <ol key={key}>
                {block.items.map((item, i) => (
                  <li key={`${key}-${i}`}>{inline(item, `${key}-${i}`)}</li>
                ))}
              </ol>
            );
          case "quote":
            return <blockquote key={key}>{inline(block.text, key)}</blockquote>;
          case "hr":
            return <hr key={key} />;
          default:
            return <p key={key}>{inline(block.text, key)}</p>;
        }
      })}
    </div>
  );
}
