import pandas as pd

from ecom.analytics.core.state import SalesState
from ecom.analytics.settings import settings


def data_loader_node(state: SalesState) -> dict:
    # 1.读取csv文件（路径统一由 settings 管理）
    df = pd.read_csv(settings.RAW_CSV_PATH)
    # 2.限定读取内容：只把列名和前5行交给模型
    return {
        "columns": df.columns.tolist(),
        "data_sample": df.head(5).to_string(),
    }
