"""执行 LLM 生成代码的沙箱。

分两层：
1. 静态校验（AST 白名单）：执行前检查语法树，禁止危险导入、dunder 属性、危险内置名。
2. 受限执行：exec 时只注入白名单 builtins，不注入完整 builtins，断开 `__class__` 之类的逃逸链。

明确的限制（见 README「安全限制」）：
- 不做子进程隔离、不做超时。模型写出的死循环仍会挂起当前进程。
- 只是显著收窄攻击面，不是强隔离。不要用它执行不可信来源的代码。
"""
from __future__ import annotations

import ast
import builtins
from pathlib import Path


class CodeSecurityError(Exception):
    """静态校验未通过。错误信息是给人/给模型看的，可直接回喂 self_corrector。"""


# 允许导入的顶层模块
_ALLOWED_ROOTS = {
    "pandas",
    "numpy",
    "math",
    "matplotlib",
    "datetime",
    "statistics",
    "json",
    "re",
}

# 禁止出现的名字（即便不提供完整 builtins，也显式拦掉以便给出清晰报错）
_BANNED_NAMES = {
    "open", "eval", "exec", "compile", "input", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "exit", "quit", "breakpoint", "memoryview",
    "dir", "help", "__import__", "type", "object", "super", "classmethod",
    "staticmethod", "property",
}

# 禁止的数据落盘方法：避免生成的代码把数据写到任意路径。
# 需要输出表格文本时请用 .to_string()，它返回值、不写文件。
_BANNED_ATTRS = {
    "to_csv", "to_excel", "to_pickle", "to_parquet", "to_hdf", "to_feather",
    "to_stata", "to_sql", "to_clipboard",
}


def strip_fences(code: str) -> str:
    """去掉模型爱加的 markdown 代码围栏。"""
    return (code or "").replace("```python", "").replace("```", "").strip()


class _Validator(ast.NodeVisitor):
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_root(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # module 为 None 表示相对导入（from . import x），一律拒绝
        self._check_root(node.module or "")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") and node.attr.endswith("__"):
            raise CodeSecurityError(f"禁止访问 dunder 属性：{node.attr}")
        if node.attr in _BANNED_ATTRS:
            raise CodeSecurityError(
                f"禁止调用 {node.attr}（会写文件）。需要输出表格文本请改用 .to_string()。"
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _BANNED_NAMES:
            raise CodeSecurityError(f"禁止使用：{node.id}")
        self.generic_visit(node)

    @staticmethod
    def _check_root(name: str) -> None:
        root = name.split(".")[0]
        if root not in _ALLOWED_ROOTS:
            raise CodeSecurityError(
                f"禁止导入模块：{name}（仅允许 {sorted(_ALLOWED_ROOTS)}）"
            )


def validate_code(code: str) -> None:
    """静态校验。违规抛 CodeSecurityError，语法错误抛 SyntaxError。"""
    tree = ast.parse(code, mode="exec")
    _Validator().visit(tree)


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """替换内置 __import__，只放行白名单顶层模块。"""
    root = name.split(".")[0]
    if root not in _ALLOWED_ROOTS:
        raise CodeSecurityError(f"禁止导入模块：{name}（仅允许 {sorted(_ALLOWED_ROOTS)}）")
    return builtins.__import__(name, globals, locals, fromlist, level)


_SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "format", "frozenset", "int", "isinstance", "issubclass", "len", "list",
    "map", "max", "min", "pow", "print", "range", "repr", "reversed", "round",
    "set", "slice", "sorted", "str", "sum", "tuple", "zip",
]

_SAFE_EXCEPTIONS = [
    "Exception", "ValueError", "KeyError", "TypeError", "ZeroDivisionError",
    "IndexError", "AttributeError", "ArithmeticError", "RuntimeError",
]


def _build_safe_builtins() -> dict:
    safe = {n: getattr(builtins, n) for n in _SAFE_BUILTIN_NAMES}
    safe.update({n: getattr(builtins, n) for n in _SAFE_EXCEPTIONS})
    # 关键：必须显式提供 __import__，否则生成代码里的 `import pandas as pd` 直接失败
    safe["__import__"] = _safe_import
    return safe


def _ensure_agg() -> None:
    """切到无 GUI 后端。matplotlib.use 必须在任何 pyplot 导入之前调用，所以放这里惰性执行。"""
    import matplotlib

    matplotlib.use("Agg")


def _build_globals(df, chart_path: Path, output_dir: Path) -> dict:
    import math

    import numpy as np
    import pandas as pd

    _ensure_agg()
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    return {
        "__builtins__": _build_safe_builtins(),
        # 预放行常用名字，模型不 import 也能直接用
        "pd": pd,
        "np": np,
        "plt": plt,
        "math": math,
        "df": df,
        "CHART_PATH": str(chart_path),
        "output_dir": str(output_dir),
    }


def run_code(code: str, df, output_dir: Path, chart_path: Path) -> dict:
    """校验并执行分析代码。

    返回 {"result": str, "error": Optional[str], "chart_generated": bool, "chart_path": str}
    """
    cleaned = strip_fences(code)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 执行前清掉旧图，这样执行后「文件是否存在」就能说明本次有没有真的出图
    chart_path = Path(chart_path)
    if chart_path.exists():
        chart_path.unlink()

    empty = {"result": "", "chart_generated": False, "chart_path": ""}

    try:
        validate_code(cleaned)
    except CodeSecurityError as e:
        return {**empty, "error": f"代码被安全策略拒绝：{e}"}
    except SyntaxError as e:
        return {**empty, "error": f"SyntaxError: {e}"}

    try:
        g = _build_globals(df, chart_path, output_dir)
        exec(compile(cleaned, "<agent>", "exec"), g)
    except CodeSecurityError as e:
        return {**empty, "error": f"代码被安全策略拒绝：{e}"}
    except Exception as e:
        return {**empty, "error": f"{type(e).__name__}: {e}"}

    generated = chart_path.exists()
    return {
        "result": str(g.get("result", "没有返回结果")),
        "error": None,
        "chart_generated": generated,
        "chart_path": str(chart_path) if generated else "",
    }
