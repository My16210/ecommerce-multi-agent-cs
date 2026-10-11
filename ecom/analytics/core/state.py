from typing import TypedDict


class SalesState(TypedDict):
    question: str
    columns: list[str]
    data_sample: str
    # 数据集三件套，由 data_loader 节点从数据集目录的 NOTES.md 读出后写进来。
    #
    # 口径为什么要在 state 里传、而不是让每个 agent 自己读文件：
    # 一次运行里 analyst / self_corrector / reviewer 必须看到**同一个字符串**。
    # 各自读文件在单次运行里结果也一样，但传 state 让"口径只有一份"这件事
    # 在类型上就成立——reviewer 拿不到别的东西可写。也顺手省掉三次磁盘读。
    dataset_name: str
    dataset_label: str
    dataset_rules: str

    generated_code: str
    exec_result: str
    exec_error: str | None
    # 仅表示「执行报错触发的代码重写次数」，由 self_corrector 的 exec 分支自增
    retry_count: int
    # "exec_error" / "review" / ""，仅用于选 prompt 和日志，不参与路由
    fix_source: str

    review_result: str
    review_comment: str
    # 解析后的权威判定，graph 路由依据
    review_passed: bool
    # 审查不通过时，解析出来喂给 self_corrector 的具体意见
    review_feedback: str
    # reviewer 已执行过的次数，由 reviewer 节点自增
    review_round: int
    need_review: bool

    chart_generated: bool
    chart_path: str
    final_report: str

    total_tokens: int
    deepseek_prompt_tokens: int
    deepseek_completion_tokens: int
    qwen_prompt_tokens: int
    qwen_completion_tokens: int

    messages: list


def make_initial_state(question: str) -> SalesState:
    """构造一次运行的初始状态。三个入口（CLI / Streamlit / 评测）共用，避免抄多份后漂移。"""
    return {
        "question": question,
        "columns": [],
        "data_sample": "",
        "dataset_name": "",
        "dataset_label": "",
        "dataset_rules": "",
        "generated_code": "",
        "exec_result": "",
        "exec_error": None,
        "retry_count": 0,
        "fix_source": "",
        "review_result": "",
        "review_comment": "",
        "review_passed": False,
        "review_feedback": "",
        "review_round": 0,
        "need_review": False,
        "chart_generated": False,
        "chart_path": "",
        "final_report": "",
        "total_tokens": 0,
        "deepseek_prompt_tokens": 0,
        "deepseek_completion_tokens": 0,
        "qwen_prompt_tokens": 0,
        "qwen_completion_tokens": 0,
        "messages": [],
    }
