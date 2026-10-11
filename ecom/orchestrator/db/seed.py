"""
把订单种子数据灌进订单库。

**数据不再是这个文件里的一串常量**，而是两份派生出来的 CSV：

    data/orchestrator/seed/orders.csv      65752 笔订单
    data/orchestrator/seed/logistics.csv   62897 条物流

它们由 `ecom/orchestrator/db/derive_seed.py` 从 DataCo 公开数据集派生（那个
脚本的模块开头写着每一条派生规则和它的理由：为什么订单号不补零、为什么客户号
加 C 前缀、为什么 `paid_at` 一律为空、为什么"配送已取消"的订单没有物流行）。

【从"手写 20 条"换成"派生 6.5 万条"，换掉的到底是什么】

手写常量的好处是看得见——`assert amount == 299.0` 一眼就懂。但它有个说不出口的
毛病：**它只证明了代码能处理我编出来的那种订单**。真实订单长什么样（订单号几位、
状态有几种、物流缺哪几项），手写数据一个都答不上来。换成派生之后，测试里的断言
仍然要写具体值（`assert amount == 327.75` 还是能一眼看懂），但那个值是**从真实
数据里来的**，不是我想出来的。

代价也得说清楚：种子数据不再是"随便改一个数字就行的文件"。改一个字段，这个文件
不报错，改的是派生脚本 + 重跑 + 重新提交 CSV。这是"数据有出处"要付的钱。

【为什么灌库要幂等，以及幂等的方式】

演示第一步就是查单，而 clone 下来的人是空库。所以 `run_order` 开头会补一次
`seed()`——它必须幂等，否则每问一句就往库里插一遍。

幂等靠"看表空不空"，不是 `INSERT OR REPLACE`：后者的 SQLite / MySQL 写法不同
（`INSERT OR REPLACE` vs `REPLACE INTO`），又会把一处方言差异漏进来。演示数据
不需要增量更新，判空足够。
"""
import csv
from datetime import datetime
from pathlib import Path

from ecom.orchestrator.db.engine import Database, get_db
from ecom.orchestrator.settings import ORDER_SEED_DIR

ORDERS_CSV = Path(ORDER_SEED_DIR) / "orders.csv"
LOGISTICS_CSV = Path(ORDER_SEED_DIR) / "logistics.csv"

# 列顺序就是 CSV 的表头顺序（derive_seed.py 里的 ORDER_COLUMNS / LOGISTICS_COLUMNS）。
# 两边各写一遍是有意的：这里显式列出 INSERT 的列名，改 CSV 的列序不会**悄悄**
# 把姓名灌进状态列——SQLite 会照灌不误，而且类型都不冲突，看不出错。
ORDER_COLUMNS = ("order_id", "customer_id", "customer_name", "status",
                 "amount", "region", "created_at", "paid_at")
LOGISTICS_COLUMNS = ("order_id", "carrier", "tracking_no", "from_city", "to_city",
                     "status", "updated_at")

ORDER_INSERT = (
    "INSERT INTO orders "
    "(order_id, customer_id, customer_name, status, amount, region, created_at, paid_at) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
)
LOGISTICS_INSERT = (
    "INSERT INTO logistics "
    "(order_id, carrier, tracking_no, from_city, to_city, status, updated_at) "
    "VALUES (?, ?, ?, ?, ?, ?, ?)"
)

# 这几列在 CSV 里是空字符串，进库要变成 NULL。
# **不能直接灌空字符串**：`paid_at` 空串在 SQLite 里是真值判定的边界（"空字符串"
# 和"没有这个值"在 SQL 里是两个意思），而下游 `_fmt_status` 正是靠它判断要不要
# 显示支付时间。派生出来的空是"数据里没有这一项"，语义上就是 NULL。
NULLABLE = {"paid_at", "carrier", "tracking_no", "from_city"}


def read_rows(path: Path, columns: tuple) -> list:
    """读 CSV，返回按 columns 排好的 tuple 列表（直接喂给 executemany）。"""
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for raw in reader:
            values = []
            for column in columns:
                value = (raw.get(column) or "").strip()
                if column in NULLABLE and not value:
                    values.append(None)
                elif column == "amount":
                    # 列是 REAL，显式转成 float 而不是把字符串交给 SQLite 的
                    # 类型亲和性去猜——猜对了是运气，猜错了是"金额按字符串排序"。
                    values.append(float(value))
                else:
                    values.append(value)
            rows.append(tuple(values))
        return rows


def seed(db: Database = None, force: bool = False) -> int:
    """
    灌入订单种子数据，返回写入的订单数。**幂等**：表里已有数据就直接返回 0。

    force=True 时先清空再灌（同时清掉退货申请——它挂在订单上，留着就是悬空引用）。
    """
    db = db or get_db()
    db.init_schema()

    existing = db.count("orders")
    if existing and not force:
        return 0

    if force:
        db.execute("DELETE FROM logistics")
        db.execute("DELETE FROM orders")
        db.execute("DELETE FROM return_requests")

    orders = read_rows(ORDERS_CSV, ORDER_COLUMNS)
    logistics = read_rows(LOGISTICS_CSV, LOGISTICS_COLUMNS)

    # 先灌订单再灌物流：logistics.order_id 指向订单，顺序反了在开了外键约束的
    # 后端上会直接失败（现在两份 schema 都没声明外键，但顺序按依赖来不花钱）。
    db.executemany(ORDER_INSERT, orders)
    db.executemany(LOGISTICS_INSERT, logistics)

    return len(orders)


def main() -> None:
    db = get_db()
    inserted = seed(db, force=True)
    where = db.path if db.backend == "sqlite" else "MySQL"
    print(f"订单库已重建：{where}")
    print(f"  订单 {inserted} 条，物流 {db.count('logistics')} 条")
    print(f"  没有物流的订单 {inserted - db.count('logistics')} 条"
          f"（配送状态为 Shipping canceled，不是数据缺失）")
    rows = db.query("SELECT order_id, customer_name, status FROM orders "
                    "WHERE order_id = '52644'")
    if rows:
        r = rows[0]
        print(f"  演示订单：{r['order_id']} {r['customer_name']} {r['status']}")
    print(f"  时间戳示例：{datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
