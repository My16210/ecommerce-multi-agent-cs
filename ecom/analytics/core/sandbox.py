"""执行 LLM 生成代码的沙箱。

分两层：
1. 静态校验（AST 白名单）：执行前检查语法树，禁止危险导入、dunder 属性、危险内置名。
2. 受限执行：exec 时只注入白名单 builtins，不注入完整 builtins，断开 `__class__` 之类的逃逸链。

明确的限制（见 README「已知限制」）：
- 不做子进程隔离、不做超时。模型写出的死循环仍会挂起当前进程。
- 不做文件系统隔离。`open()` 被禁掉了，但 pandas / numpy / matplotlib
  自带文件 I/O，所以这里按名字禁掉它们的读写入口（`read_*`、`to_csv`、
  `savefig` 只许写 CHART_PATH……）。**这是黑名单，不是边界**：`pd.io.*`
  之类的间接入口仍可能绕过去，真正的隔离得靠容器 / 低权用户 / 只读挂载。
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

# 生成代码不许改写的名字。`CHART_PATH` 被换掉的话，`savefig(CHART_PATH)`
# 这道"只许写这个路径"的检查就形同虚设。
_PROTECTED_NAMES = {"CHART_PATH", "output_dir"}

# 禁止的数据落盘 / 读取方法。
#
# 【为什么读取也要禁】AST 拦掉了 `open()`，但 pandas / numpy / matplotlib
# 自己就是文件 API：2026-10-30 实测 `pd.read_csv(r"C:\Windows\win.ini")`
# 能读到内容、`plt.savefig("任意路径")` 能写文件——**不需要逃逸，用放行的库
# 就够了**。所以按方法名把常见读写入口逐个禁掉。
#
# 这是黑名单，不是边界：`pd.io.*` 之类的间接入口仍可能绕过。
_WRITE_ATTRS = {
    "to_csv", "to_excel", "to_pickle", "to_parquet", "to_hdf", "to_feather",
    "to_stata", "to_sql", "to_clipboard", "to_json", "to_html", "to_xml",
    "to_latex", "to_gbq", "to_orc", "tofile", "dump",
    "save", "savez", "savez_compressed", "savetxt", "imsave",
    "savefig", "print_png", "print_svg", "print_pdf", "print_ps",
    "print_figure",
    # 直接以构造函数打开文件的写入器（绕过 to_* / save* 的名字）：
    #   pd.ExcelWriter(path)          打开 xlsx 写句柄
    #   matplotlib PdfPages(path)     打开 pdf 写句柄
    #   np.lib.format.open_memmap(path, mode="w+") 打开 .npy 写句柄
    "ExcelWriter", "PdfPages", "open_memmap",
}
# 读取：按名字逐个列（前缀 read_ 在下面单独判）。
_READ_ATTRS = {
    "load", "loadtxt", "genfromtxt", "fromfile", "fromregex", "memmap",
    "imread", "ExcelFile", "HDFStore", "get_handle", "urlopen", "open",
    "TextFileReader",
}
_BANNED_ATTRS = _WRITE_ATTRS | _READ_ATTRS


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
        for alias in node.names:
            # `from pandas import read_csv` 绕开了 `pd.read_csv` 的属性检查
            if alias.name in _BANNED_ATTRS or alias.name.startswith("read_"):
                raise CodeSecurityError(f"禁止导入 {alias.name}（会读写文件）")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") and node.attr.endswith("__"):
            raise CodeSecurityError(f"禁止访问 dunder 属性：{node.attr}")
        if node.attr in _BANNED_ATTRS or node.attr.startswith("read_"):
            raise CodeSecurityError(
                f"禁止调用 {node.attr}（会读写文件）。"
                "需要输出表格文本请改用 .to_string()；出图请用 plt.savefig(CHART_PATH)。"
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _BANNED_NAMES:
            raise CodeSecurityError(f"禁止使用：{node.id}")
        # 任何改写都算（赋值 / for 目标 / with as / 海象），
        # 否则把 CHART_PATH 换成别的路径，savefig 那道检查就白做了。
        if node.id in _PROTECTED_NAMES and not isinstance(node.ctx, ast.Load):
            raise CodeSecurityError(f"禁止改写 {node.id}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """
        `savefig` 是唯一放行的写文件入口，而且只许写 CHART_PATH。

        不能直接放进 `_BANNED_ATTRS` 一刀切（出图是正路），也不能放行任意
        路径（实测能覆盖容器里的任何文件）。所以在这里按**实参**卡；并且
        不 `generic_visit(func)`——否则上面那条 Attribute 规则会把这次合法
        调用也拒掉。别名（`s = plt.savefig`）仍会被 Attribute 规则拦住。
        """
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "savefig":
            self._check_savefig_target(node)
            # 只放过 func 本身（savefig 在 _BANNED_ATTRS 里），接收者必须
            # 照常校验：`(open(...) and plt).savefig(CHART_PATH)` 能借这个
            # 位置塞进任意调用，跳过它等于把整段表达式变成免检区。
            self.visit(func.value)
            for arg in node.args:
                self.visit(arg)
            for keyword in node.keywords:
                self.visit(keyword.value)
            return
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_params(node.args)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_params(node.args)
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._check_params(node.args)
        self.generic_visit(node)

    # 下面几种绑定名的 AST 字段是**字符串**，不是 Name 节点，visit_Name 看不到。
    # `case CHART_PATH:` 能把 subject 绑到这个受保护名字上，随后
    # `savefig(CHART_PATH)` 的实参检查看到的仍是一个名字，等于绕开路径限制。
    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name in _PROTECTED_NAMES:
            raise CodeSecurityError(f"禁止在 match 里绑定 {node.name}")
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name in _PROTECTED_NAMES:
            raise CodeSecurityError(f"禁止在 match 里绑定 {node.name}")
        self.generic_visit(node)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        if node.rest in _PROTECTED_NAMES:
            raise CodeSecurityError(f"禁止在 match 里绑定 {node.rest}")
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name in _PROTECTED_NAMES:
            raise CodeSecurityError(f"禁止把异常绑定为 {node.name}")
        self.generic_visit(node)

    @staticmethod
    def _check_savefig_target(node: ast.Call) -> None:
        target = node.args[0] if node.args else None
        if target is None:
            for keyword in node.keywords:
                if keyword.arg in {"fname", "fp"}:
                    target = keyword.value
                    break
        if not (isinstance(target, ast.Name) and target.id == "CHART_PATH"):
            raise CodeSecurityError(
                "plt.savefig 只允许写 CHART_PATH（题目给的输出路径），不允许自己拼路径。"
            )

    @staticmethod
    def _check_params(args: ast.arguments) -> None:
        # 参数叫 CHART_PATH 会遮住全局那个：savefig(CHART_PATH) 于是变成了
        # "写调用方给的任意路径"，而实参检查看到的仍然是一个名字。
        # *args / **kwargs 的名字同样是字符串，不能漏。
        params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
        if args.vararg:
            params.append(args.vararg)
        if args.kwarg:
            params.append(args.kwarg)
        for arg in params:
            if arg.arg in _PROTECTED_NAMES:
                raise CodeSecurityError(f"禁止把参数命名为 {arg.arg}")

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

    # 中文字体列表，**按可用性逐个往下找**，不是只写一个。
    #
    # 原来这里只有 ["Microsoft YaHei"] 一项，在 Windows 上没问题（那台机器上
    # 有）。但镜像里没有这个字体，matplotlib 的报错方式是**只打一行 warning 然后
    # 悄悄退回默认字体**：
    #     findfont: Generic family 'sans-serif' not found because none of the
    #     following families were found: Microsoft YaHei
    # 默认的 DejaVu Sans 没有汉字，于是图照样生成、exit code 照样是 0，
    # 只有**把 PNG 打开看**才会发现每个中文标签都成了豆腐块。这是真在容器里
    # 跑出来的：华东/华北/华南/西南/西北 五个标签全变方框。
    #
    # 所以两件事一起做：这里给出跨平台的候选顺序（Windows 仍然首选雅黑，行为
    # 不变；Linux 落到 Noto），Dockerfile 里把 fonts-noto-cjk 装上。
    # 只改一边没用——列表里写着但系统上没有，等于没写。
    #
    # 最后一项 DejaVu Sans 是兜底：它没有中文，但能保证"缺字体"不会变成
    # 抛异常把整条分析链路打断——图画得难看，好过图完全没有。
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "SimHei",
        "DejaVu Sans",
    ]
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
