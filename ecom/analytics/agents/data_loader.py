import pandas as pd

from ecom.analytics.core.state import SalesState
from ecom.analytics.settings import settings
from ecom.analytics.utils.dataset_doc import load_dataset_doc


def data_loader_node(state: SalesState) -> dict:
    # 1.读取csv文件（路径统一由 settings 管理，换数据集只改 ANALYTICS_DATASET 环境变量）
    #
    # encoding="utf-8-sig"：源 CSV 带 UTF-8 BOM 时，pandas 默认编码会把首列名读成
    # "﻿日期"——不报错、不警告，只是模型随后写的 df['日期'] 全部 KeyError，
    # 而报错信息里那个不可见的 ﻿ 根本看不出来。utf-8-sig 会吃掉 BOM；
    # 对没有 BOM 的文件，它和 utf-8 行为完全一致，所以是安全的默认值。
    df = pd.read_csv(settings.RAW_CSV_PATH, encoding="utf-8-sig")

    # 2.读口径。**这是这些字段唯一的来源**——analyst、self_corrector、reviewer
    #   三个节点都从这个 state 取，不再各自在 prompt 里写一遍业务口径。
    #   原来那套写法（prompts.py 一份、reviewer.py 又一份）没有任何机制保证一致，
    #   改了一处漏一处，模型收到的两套口径就开始互相矛盾。
    label, rules = load_dataset_doc(settings.DATASET_DOC_PATH)

    # 3.限定读取内容：只把列名和前5行交给模型
    return {
        "columns": df.columns.tolist(),
        "data_sample": df.head(5).to_string(),
        "dataset_name": settings.DATASET_NAME,
        "dataset_label": label,
        "dataset_rules": rules,
    }
