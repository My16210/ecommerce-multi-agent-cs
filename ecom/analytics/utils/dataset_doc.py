"""读数据集目录里的 NOTES.md。

一个数据集 = `data/analytics/datasets/<名字>/` 下的两个文件：
    data.csv    给模型操作的表
    NOTES.md    这份表的说明 + 默认口径

NOTES.md 的格式：顶部一段可选 frontmatter，里面只有 `label` 一个字段，
其余全是正文。正文会被**原样**塞进 analyst / self_corrector / reviewer 的 prompt。

为什么只有 label 一个字段、且不引 PyYAML：

  这份文件是**给模型读的自然语言说明**，不是配置。一旦它能被 YAML 解析出
  嵌套结构，下一个人就会开始往里加 `metrics:`、`default_filter:` 之类的键，
  口径就从"一段人话"变成"一套 schema"，然后没人愿意维护它。
  为读一个标量字符串引一个依赖也不划算。
  所以：`label` 走简单的字符串切分，正文原样保留。
"""
from __future__ import annotations

from pathlib import Path


def load_dataset_doc(path: Path) -> tuple[str, str]:
    """把 NOTES.md 读成 (label, 正文)。

    label 缺失时返回空串（调用方负责兜底显示），文件不存在时抛 FileNotFoundError——
    数据集目录缺 NOTES.md 是**配置错误**，不该被静默吞掉：模型没有口径可依据，
    只会照着列名猜，那正是这个文件要防的事。
    """
    raw = Path(path).read_text(encoding="utf-8")

    if not raw.startswith("---"):
        return "", raw.strip()

    # frontmatter 的结束标记是第二个独占一行的 `---`
    end = raw.find("\n---", 3)
    if end == -1:  # 只有开头那个 ---，整个文件当正文
        return "", raw.strip()

    head = raw[3:end]
    body = raw[end + 4 :]

    label = ""
    for line in head.splitlines():
        if line.strip().startswith("label:"):
            label = line.split(":", 1)[1].strip()
            break

    return label, body.strip()
