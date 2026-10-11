"""
订单库：schema、连接、灌数据。

**默认 SQLite，MySQL 是可选项。** 理由不是"SQLite 更好"，而是取舍不同：
这个仓库的定位是"clone 下来就能跑给面试官看"。挂一个必须联网、必须先起容器的
数据库，等于给第一次运行加了三个失败点（Docker 没起、端口占了、密码不对），
而它换来的东西（多机并发写）在演示场景里一次都用不上。

但**代码不能被 SQLite 惯坏**：SQL 全部用 `?` 占位符写，MySQL 后端在
`engine._translate` 里统一换成 `%s`；DDL 用两边都吃的写法（VARCHAR 而不是 TEXT
做主键，不用 AUTOINCREMENT）。所以"换成 MySQL"是配置改动，不是重写——
见 `ORDER_DB_DSN`。分叉只在最薄的一层（连接与占位符），不在业务代码里。

【四个文件的分工】
    schema.sql      建表 DDL（两个后端共用一份）
    engine.py       连接 + `?`/`%s` 翻译 + query / execute / executemany
    derive_seed.py  **一次性 ETL**：公开数据集 → seed/*.csv。跑完就不再参与运行，
                    和 analytics 那边的 `cli/derive_dataco.py` 是同一类东西。
    seed.py         运行期读 seed/*.csv 灌库（幂等）。`main.py --init-db` 走它。

派生脚本和灌库脚本**刻意分成两个文件**：前者要读 96MB 的原始数据、只在换数据源时
跑一次；后者每次演示都可能跑。合起来的代价是，"灌库"这件事从此依赖一份仓库里
根本没有的数据集——`data/analytics/source/` 是 gitignore 的，clone 下来的人手里
只有派生结果。
"""
