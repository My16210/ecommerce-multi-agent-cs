"""
答案生成模块
功能：基于检索到的chunk，用千问生成答案

知识库结果和联网结果会分栏拼进 prompt，让模型知道每条资料的出处
"""
from ecom.knowledge.utils import streaming
from ecom.knowledge.utils.llm import get_llm
from ecom.knowledge.utils.logger import setup_logger
from ecom.knowledge.utils.timing import metric, stage

logger = setup_logger(__name__)

SYSTEM_PROMPT = """你是知识库问答助手，请严格依据下面提供的资料回答用户问题。

要求：
1. 只依据资料作答。资料里没有的信息不要编造，直接说明"提供的资料中没有相关内容"。
2. 在答案中用 [编号] 标注引用，例如 [1]、[2]，编号对应资料里的编号。
3. 知识库资料与联网搜索结果冲突时，优先采信知识库。
4. 用简体中文回答，简洁清晰。"""

# 生成要一点随机性，temperature=0.3；客户端本身由 ecom/knowledge/utils/llm.py 缓存
TEMPERATURE = 0.3
TIMEOUT = 60


def _as_chunk(item) -> dict:
    """兼容 dict 和纯字符串两种输入（reranker 同样做了兼容）"""
    return item if isinstance(item, dict) else {"text": str(item)}


def _format_contexts(contexts: list, web_contexts: list) -> str:
    """把两路结果拼成带编号的资料块"""
    sections = []

    if contexts:
        lines = ["【知识库检索结果】"]
        for index, raw in enumerate(contexts, start=1):
            chunk = _as_chunk(raw)
            source = chunk.get("source", "未知来源")
            chunk_id = chunk.get("chunk_id", "?")
            lines.append(f"[{index}]（来源：{source} 第 {chunk_id} 段）\n{chunk.get('text', '')}")
        sections.append("\n\n".join(lines))

    if web_contexts:
        offset = len(contexts)
        lines = ["【联网搜索结果】"]
        for index, raw in enumerate(web_contexts, start=offset + 1):
            chunk = _as_chunk(raw)
            title = chunk.get("title", "网页")
            url = chunk.get("url", "")
            lines.append(f"[{index}]（来源：{title} {url}）\n{chunk.get('text', '')}")
        sections.append("\n\n".join(lines))

    return "\n\n".join(sections) if sections else "（没有检索到任何资料）"


def _format_history(chat_history: list) -> str:
    """把历史对话拼成文本，供模型衔接上下文"""
    if not chat_history:
        return ""

    lines = ["【历史对话】"]
    for turn in chat_history:
        role = "用户" if turn.get("role") == "user" else "助手"
        lines.append(f"{role}：{turn.get('content', '')}")
    return "\n".join(lines) + "\n\n"


def _chunk_text(chunk) -> str:
    """从一个流式 chunk 里取出文本部分。

    不同版本的 langchain 把文本放在不同地方：`.text` 是便捷属性，
    `.content` 则可能是字符串、也可能是分块列表（带推理字段的模型会这样）。
    这里把三种形状都兜住——**少兜一种的症状是"答案缺一段"**，而它看起来
    就像是模型自己没说完，基本追不到这里。
    """
    text = getattr(chunk, "text", None)
    if isinstance(text, str) and text:
        return text

    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _invoke(llm, messages: list) -> tuple:
    """非流式：一次性拿全文（加流式之前就是这条路径，行为没动）"""
    response = llm.invoke(messages)
    return response.content, getattr(response, "usage_metadata", None) or {}


def _stream(llm, messages: list) -> tuple:
    """流式：边收边发给当前的汇，同时把全文拼回来

    拼回来的那份文本才是 state["answer"]——所以流式只改变"什么时候看到"，
    不改变"答案是什么"。这也是它敢改的原因：judge 判空、多轮记忆、评测
    读的都是同一份文本，它们不知道界面挂了汇。
    """
    pieces = []
    usage = {}
    for chunk in llm.stream(messages):
        text = _chunk_text(chunk)
        if text:
            pieces.append(text)
            streaming.emit(text)
        # 开了 stream_usage 时，用量挂在最后一个 chunk 上（别的 chunk 上没有）
        if getattr(chunk, "usage_metadata", None):
            usage = chunk.usage_metadata
    return "".join(pieces), usage


def generate_answer(
    question: str,
    contexts: list,
    web_contexts: list = None,
    chat_history: list = None,
) -> str:
    """
    基于检索结果生成答案

    contexts：知识库检索（已 rerank）的结果
    web_contexts：联网搜索结果
    chat_history：[{"role": "user"/"assistant", "content": ...}, ...]
    """
    web_contexts = web_contexts or []
    chat_history = chat_history or []

    prompt = (
        f"{_format_contexts(contexts, web_contexts)}\n\n"
        f"{_format_history(chat_history)}"
        f"【用户问题】\n{question}"
    )

    logger.info(
        "生成答案：知识库 %d 条，联网 %d 条，历史 %d 轮",
        len(contexts), len(web_contexts), len(chat_history) // 2,
    )

    messages = [("system", SYSTEM_PROMPT), ("human", prompt)]

    with stage("generate"):
        llm = get_llm(TEMPERATURE, TIMEOUT)
        # 有汇（界面挂了）就走流式，没有（命令行、评测、被直接 import）就走
        # 原来的整段返回。两条路拼出来的文本必须一模一样，有测试钉住。
        if streaming.current() is None:
            content, usage = _invoke(llm, messages)
        else:
            content, usage = _stream(llm, messages)

    # 用量记进计时器（没在采集时是空操作）。**只在这里记一次**：
    # 响应对象一返回就丢了（函数只往外给 content），事后再想补算是拿不到的。
    # 流式那条路靠 llm.py 里打开的 stream_usage，否则这里会读到 0。
    metric("llm_input_tokens", usage.get("input_tokens", 0))
    metric("llm_output_tokens", usage.get("output_tokens", 0))
    return content


if __name__ == "__main__":
    answer = generate_answer("退货政策是什么", ["7天无理由退货", "质量问题商家承担运费"])
    print(answer)
