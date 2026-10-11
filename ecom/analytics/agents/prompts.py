"""analyst 和 self_corrector 共用的代码生成约定。

抽出来是因为两处都要给模型同一套「result 变量 / 出图 / 口径 / 可用环境」约束，
分开写很容易漂移（模型改代码时按 A 套写、按 B 套改，结果自相矛盾）。

【业务口径不在这里，在数据集的 NOTES.md 里】

原来这份文件里写着「金额是订单金额；是否退货 = 1 表示退货……」——那是**某一份
数据的**业务规则，被写死在了代码里。换一份数据，这段话就成了假的，而且是悄悄
变成假的：代码照样跑，只是模型拿着上一份数据的口径去解释这一份数据的列。
reviewer.py 里还照着抄了一份，两份之间没有任何东西保证同步。

现在只留「怎么用 pandas」这类跨数据集都成立的通用约束；凡是"这份数据是什么
意思"的内容，由 data_loader 从 `data/analytics/datasets/<名字>/NOTES.md` 读出来，
经 state 注入到这里。换数据 = 换目录，这段通用规则一个字都不用动。
"""
from __future__ import annotations

CODE_RULES_TEMPLATE = """【输出要求】
1. 只输出 Python 代码，不要 markdown 围栏，不要任何解释文字。
2. 必须把最终答案赋值给变量 result。
3. 结果若是表格，先用 .to_string() 转成字符串再赋给 result；不要用 to_csv，它会把文件写到磁盘上（已被禁止）。

【数据口径】
以下是当前数据集的说明与默认口径，是你判断口径的**唯一依据**：
{dataset_rules}

另外两条通用要求：
- 计算口径要和问题的问法一致，不要偷偷换一个口径。
- 上面没提到的字段、维度、编码，**就是不存在的**。不要凭"一般电商数据应该有"去补一个出来，也不要给一个你看不懂的编码编一个含义。

【可用环境】
- 已提供 df（数据表）、pd（pandas）、np（numpy）、plt（matplotlib.pyplot）、math。
- 只能用这些库，禁止 import os/sys/subprocess 等；禁止读写任意文件。
- 图表保存路径用已提供的变量 CHART_PATH，不要自己拼路径。

【要不要画图】
- 问题涉及趋势、对比、排名、分布、占比时，用 plt 画一张贴合的图，然后 plt.savefig(CHART_PATH) 再 plt.close()。
- 问题只问单个数值或单个名称时（例如「总销售额是多少」），不要画图。
- 图必须贴合问题本身，不要画一张与问题无关的固定图表。"""


def build_code_rules(dataset_rules: str) -> str:
    """把数据集自己的口径填进通用规则，产出完整的一段 prompt。

    用 str.replace 而不是 str.format：口径正文来自 NOTES.md，是**内容**不是模板，
    里面出现一个花括号（写公式、写正则都可能）就会让 format 抛 KeyError/IndexError，
    报错还指向这个函数而不是数据文件，排查成本很高。replace 只认一个已知 token。
    """
    return CODE_RULES_TEMPLATE.replace("{dataset_rules}", (dataset_rules or "").strip())
