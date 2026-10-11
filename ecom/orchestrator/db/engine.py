"""
数据访问层：一个薄薄的双后端适配。

【为什么是两个"几乎相同"的后端而不是上 SQLAlchemy】
1. 这里一共三条 SQL 语句级的需求（查单、查物流、插退货），上 ORM 换来的是
   一整套依赖和一个要学的抽象，收益是负数。
2. 更重要的是**参数占位符不一样**（SQLite 用 `?`，MySQL 用 `%s`）。这层要挡的
   就是它——业务代码永远只写 `?`，由这里翻译。写死了 `?` 再直连 MySQL，
   报的错是 `You have an error in your SQL syntax`，和真实原因隔了三层。

SQLite 是默认（clone 下来就能跑），MySQL 是给了 DSN 才启用。这和
rag_knowledge_base 的「Chroma 默认 / Milvus 可选」是同一条取舍：**默认路径必须
零外部依赖**，否则"别人拿到能跑"就成了一句空话。
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from ecom.orchestrator.settings import ORDER_DB_DSN, ORDER_DB_PATH
from ecom.orchestrator.utils.logger import setup_logger

logger = setup_logger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


class Database:
    """一个连接工厂。每次操作开一个短连接——demo 规模下连接池是过早优化。"""

    def __init__(self, path: Path = None, dsn: str = None):
        self.path = Path(path) if path else ORDER_DB_PATH
        self.dsn = dsn if dsn is not None else ORDER_DB_DSN
        self.backend = "mysql" if self.dsn else "sqlite"

    # ——— 连接 ———
    @contextmanager
    def connect(self):
        if self.backend == "mysql":
            yield from self._connect_mysql()
        else:
            yield from self._connect_sqlite()

    def _connect_sqlite(self):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _connect_mysql(self):
        try:
            import pymysql
            from pymysql.cursors import DictCursor
        except ImportError as exc:
            raise RuntimeError(
                "配了 ORDER_DB_DSN 但没装 MySQL 驱动：uv sync --extra mysql"
            ) from exc

        conn = pymysql.connect(
            **self._parse_dsn(self.dsn), cursorclass=DictCursor, autocommit=True, charset="utf8mb4"
        )
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _parse_dsn(dsn: str) -> dict:
        """把 mysql://user:pass@host:port/db 拆成 pymysql 的参数。

        只支持这一个形状——比引一个 URL 解析库强：这里要的就是"看得懂的报错"。
        """
        try:
            head, rest = dsn.split("://", 1)
            credentials, location = rest.split("@", 1)
            user, password = credentials.split(":", 1)
            host_port, database = location.split("/", 1)
            host, _, port = host_port.partition(":")
            return {
                "user": user,
                "password": password,
                "host": host,
                "port": int(port or 3306),
                "database": database,
            }
        except ValueError as exc:
            raise ValueError(
                f"ORDER_DB_DSN 格式不对：{dsn!r}。应该是 "
                "mysql://user:password@host:3306/dbname"
            ) from exc

    # ——— 语句 ———
    def _translate(self, sql: str) -> str:
        """把 `?` 占位符翻成目标后端的写法。业务代码只写 `?`。"""
        if self.backend != "mysql":
            return sql
        return sql.replace("?", "%s")

    def query(self, sql: str, params: tuple = ()) -> list:
        """返回 [dict, ...]。空结果返回空列表，不是 None——调用方少一层判空。"""
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(self._translate(sql), params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def execute(self, sql: str, params: tuple = ()) -> int:
        """返回受影响行数。"""
        with self.connect() as conn:
            cursor = conn.cursor()
            count = cursor.execute(self._translate(sql), params)
            return count if isinstance(count, int) else 0

    def executemany(self, sql: str, rows: list) -> int:
        """批量写。返回提交的行数。

        **不是为了少写一个 for**：`execute` 每次调用都开一条新连接、单独提交
        （见 connect()——这是刻意的短连接设计），拿它循环灌 12 万行种子，就是
        12 万次开库 + 12 万次提交，实测要按分钟算。executemany 是一条连接、
        一次提交，同一批数据降到秒级。

        两个后端都认 `executemany`，所以这里没有方言分支——只有 `?` -> `%s`
        那一层翻译照旧。
        """
        rows = list(rows)
        if not rows:
            return 0
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.executemany(self._translate(sql), rows)
            return len(rows)

    def init_schema(self) -> None:
        """建表。**幂等**——每次启动都调它，不该有副作用。"""
        ddl = SCHEMA_PATH.read_text(encoding="utf-8")
        with self.connect() as conn:
            cursor = conn.cursor()
            for statement in [s.strip() for s in ddl.split(";") if s.strip()]:
                if statement.startswith("--"):
                    # 去掉纯注释开头的片段：schema.sql 里注释和语句混排
                    statement = "\n".join(
                        line for line in statement.splitlines() if not line.strip().startswith("--")
                    ).strip()
                    if not statement:
                        continue
                cursor.execute(self._translate(statement))

    def count(self, table: str) -> int:
        rows = self.query(f"SELECT COUNT(*) AS n FROM {table}")  # noqa: S608 —— table 来自内部调用
        return int(rows[0]["n"]) if rows else 0


_default: Database = None


def get_db() -> Database:
    """进程内共享一个 Database 描述（每次仍开短连接）。"""
    global _default
    if _default is None:
        _default = Database()
        _default.init_schema()
        logger.info(
            "订单库：%s（%s）",
            "MySQL" if _default.backend == "mysql" else _default.path,
            _default.backend,
        )
    return _default
