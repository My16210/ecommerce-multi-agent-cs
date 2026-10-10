"""
文档加载和切分模块
功能：读取文档(txt/md/PDF)，切成chunk

PDF 原计划用 MinerU 转成 Markdown，但 MinerU 依赖较重，
阶段一先用 pypdf 直接抽取文本兜底，后续需要更好的 PDF 解析再换。
"""
import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ecom.knowledge.settings import CHUNK_OVERLAP, CHUNK_SIZE

# 默认分隔符只认空格和换行，中文长段落会被硬切，所以补上中文标点
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", ". ", " ", ""]

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}


def load_document(file_path: str) -> str:
    """
    加载文档，返回纯文本
    - txt/md: 直接读取
    - PDF: 用 pypdf 抽取每页文本
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8")

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)

    raise ValueError(f"暂不支持的文档类型：{suffix}（支持 .md/.txt/.pdf）")


def split_text(text: str) -> list:
    """
    把长文本切成chunk
    参数：chunk_size=512, chunk_overlap=50
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )
    return splitter.split_text(text)


def load_and_split(file_path: str) -> list:
    """
    加载并切分单个文件，给每个 chunk 打上来源信息

    返回：[{"text": ..., "source": ..., "chunk_id": ...}, ...]

    source 和 chunk_id 在这里产生——引用追溯全靠这两个字段。
    它们一旦在后面某步被丢成纯字符串，就再也找不回来了。
    """
    text = load_document(file_path)
    source = os.path.basename(file_path)
    return [
        {"text": chunk, "source": source, "chunk_id": index}
        for index, chunk in enumerate(split_text(text))
    ]


if __name__ == "__main__":
    # 测试
    test_text = "退货政策：7天无理由退货。\n退货流程：联系客服→审核→寄回→退款。\n运费：质量问题商家承担运费。"
    chunks = split_text(test_text)
    print(f"切了{len(chunks)}个chunk")
    for i, c in enumerate(chunks):
        print(f"chunk {i}: {c[:50]}...")
