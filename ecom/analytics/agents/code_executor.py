import pandas as pd

from ecom.analytics.core.sandbox import run_code
from ecom.analytics.core.state import SalesState
from ecom.analytics.settings import settings
from ecom.analytics.utils.logger import setup_logger

logger = setup_logger()


def code_executor_node(state: SalesState) -> dict:
    # 1.读取数据（路径统一由 settings 管理）
    df = pd.read_csv(settings.RAW_CSV_PATH)

    # 2.交给沙箱：静态校验 + 受限 builtins 执行，不再裸 exec
    out = run_code(
        state["generated_code"],
        df,
        settings.OUTPUT_DIR,
        settings.OUTPUT_DIR / "chart.png",
    )

    # 3.把结果映射回状态
    if out["error"]:
        logger.error(f"代码执行失败: {out['error']}")
    else:
        logger.info(f"代码执行成功: {out['result']} | 出图={out['chart_generated']}")

    return {
        "exec_result": out["result"],
        "exec_error": out["error"],
        "chart_generated": out["chart_generated"],
        "chart_path": out["chart_path"],
    }
