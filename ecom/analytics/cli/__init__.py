"""
数据分析的命令行入口。

四个脚本原来躺在 `sales_agent` 仓库根，靠 `sys.path.insert(0, 脚本所在目录)`
自举、然后 `from src.core.graph import ...`。名字太通用（`main.py` / `app.py` /
`eval.py`），留在新仓库根会和编排层的 `main.py` 打架——所以收进包里，
调用方式变成 `python -m ecom.analytics.cli.<名字>`，`sys.path.insert` 全部删掉
（合并后 `pythonpath=["."]` 由 pyproject 统一管，三套测试并成一套的直接收益）。

    analyze.py         **编排层唯一调用的那个**：stdout 只吐一行 JSON
    main.py            本地手动跑一遍，人看的输出
    eval.py            agent 测评：运行成功率 / 首次执行成功率 / 纠错有效率 / 答案准确率
    app.py             一个独立的 Streamlit 单页（先留着，统一 app.py 起来后可以删）
    derive_dataco.py   一次性 ETL：DataCo 主表 → datasets/dataco/data.csv（产物进仓库）

`derive_dataco.py` 和上面几个不是一类：它不进任何运行时链路，是**改数据时才跑一次**
的派生脚本。放在 cli/ 下是因为它确实是个要人手动敲的命令，而且和 knowledge 那边的
`fetch_corpus.py` 是同一种东西（一次性抓取/派生 → 提交产物），两处放同一个位置，
找的时候不用猜。

`analyze_json.py` 改名成 `analyze.py`：它已经不"json 文件"了——它的契约是
**stdout 只吐一行 JSON**，名字该说这件事，而不是说它的格式。
"""
