"""跨子系统的不变量：**包结构与 import 路径**。

这个文件不测业务，测的是"合并本身有没有搬干净"。它存在的理由很具体：

    数据分析那 34 条测试在本机跑不起来（缺 pandas / langgraph / matplotlib），
    所以那边"有没有一条 import 指到了不存在的模块"**没有任何别的测试能发现**。
    合并改动 270 处 import，其中约 55 处在那个跑不了的子系统里。全靠手工核对
    是不行的——所以这里不 import 业务代码，只用 AST 解析 + 文件系统比对，
    在没有依赖的机器上也能跑。

三条不变量：

  1. `ecom/` 里每一条 `from ecom.x.y import z` 都解析得到真实存在的模块/名字。
     这条抓的是"sed 漏了一处"和"文件搬错了深度"。
  2. 全仓库没有 `src.` / `config.` 形式的模块引用残留（包括藏在字符串里的
     `importlib.import_module("src.…")`——那种 `from src.` 的 sed 抓不到）。
     这条防的是"哪天又长回来"。
  3. 那个"往上数目录"算出来的根，指向仓库根而不是 `ecom/`。

第 3 条为什么值得单独钉一条断言：三个子系统原来各有一处
`Path(__file__).resolve().parent.parent`（还有一个 `parent.parent.parent`），
搬进子目录后它们**全都还能算出结果、只是算错**——不抛异常，只是此后
`data/` 找不到、日志写到 `ecom/logs/`。这类错误不会被任何业务测试撞见，
只能显式钉住。
"""
import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_ROOT = REPO_ROOT / "ecom"

# 顶层包名。改这个常量 = 换整个项目的命名空间，那时这个文件要跟着改。
TOP_LEVEL = "ecom"

# 合并前两个仓库用过的顶层名。它们不再允许出现——见上面第 2 条。
FORBIDDEN_ROOTS = ("src", "config")


def _scan_files():
    """所有要检查的 .py：包内 + 三个子系统的测试。"""
    files = list(PKG_ROOT.rglob("*.py"))
    files += list((REPO_ROOT / "tests").rglob("*.py"))
    return sorted(f for f in files if "__pycache__" not in f.parts)


def _resolve(dotted: str):
    """把 `ecom.a.b` 解析成一个文件；解析不到返回 None。"""
    if not dotted.startswith(TOP_LEVEL):
        return None
    rel = Path(*dotted.split("."))
    as_module = REPO_ROOT / rel.with_suffix(".py")
    if as_module.is_file():
        return as_module
    as_package = REPO_ROOT / rel / "__init__.py"
    if as_package.is_file():
        return as_package
    return None


def _defined_names(path: Path) -> set:
    """一个模块文件里定义（或从别处转手导进来）的所有名字。

    刻意做宽：`if TYPE_CHECKING:` 里的、`try/except ImportError` 里的都算。
    这个检查的目的是抓"路径指错了"，不是抓"这个名字到底可不可用"——
    宁可漏报也不要误报，误报会让人开始忽略这个测试。
    """
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.alias):
            names.add(node.asname or node.name.split(".")[0])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(node.value)
    return names


def _imported_modules(tree):
    """产出一条条 (行号, 完整模块名) —— `import ecom.x` 和 `from ecom.x import y` 两种。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or not node.module:
                continue
            yield node.lineno, node.module


def test_every_ecom_import_points_at_a_real_module():
    """不变量 1：从 ecom 出发的每一跳都落在真实存在的文件上。"""
    problems = []

    for path in _scan_files():
        where = path.relative_to(REPO_ROOT)
        tree = ast.parse(path.read_text(encoding="utf-8"))

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level != 0 or not node.module:
                continue
            mod = node.module
            if mod != TOP_LEVEL and not mod.startswith(TOP_LEVEL + "."):
                continue

            target = _resolve(mod)
            if target is None:
                problems.append(f"{where}:{node.lineno}  模块 `{mod}` 不存在")
                continue

            for alias in node.names:
                if alias.name == "*":
                    continue
                # `from ecom.a import b` 里 b 可能是子模块（ecom.a.b），也可能是 ecom.a 里的名字
                if _resolve(f"{mod}.{alias.name}") is not None:
                    continue
                if alias.name not in _defined_names(target):
                    problems.append(
                        f"{where}:{node.lineno}  `{mod}` 里既没有子模块也没有名字 `{alias.name}`"
                    )

        for lineno, dotted in _imported_modules(tree):
            if dotted.startswith(TOP_LEVEL) and _resolve(dotted) is None:
                problems.append(f"{where}:{lineno}  模块 `{dotted}` 不存在")

    assert not problems, "有以下 import 指不到真实模块：\n  " + "\n  ".join(sorted(set(problems)))


def test_no_leftovers_from_the_old_repo_layout():
    """不变量 2：`src.` / `config.` 一个都不许剩，字符串里的也算。"""
    problems = []

    for path in _scan_files():
        where = path.relative_to(REPO_ROOT)
        tree = ast.parse(path.read_text(encoding="utf-8"))

        for lineno, dotted in _imported_modules(tree):
            if dotted.split(".")[0] in FORBIDDEN_ROOTS:
                problems.append(f"{where}:{lineno}  import 了 `{dotted}`")

        # 藏起来的引用：importlib.import_module("src.vector_store.milvus_store")。
        # 合并时真的踩到过——`from src.` 的 sed 完全看不见它，三条测试因此红了。
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if node.value.split(".")[0] in FORBIDDEN_ROOTS and "." in node.value:
                problems.append(f"{where}:{node.lineno}  字符串里的模块路径 `{node.value}`")

    assert not problems, "以下位置还引用着旧仓库的顶层包名：\n  " + "\n  ".join(
        sorted(set(problems))
    )


def test_the_root_is_the_repo_root_not_the_package():
    """不变量 3：把那个"往上数目录"的结果钉死。

    失败信息要能直接指出错在哪——这类错误的症状是"文件找不到"，不是这个断言本身，
    所以断言里把两个路径都打出来。
    """
    from ecom.settings import ANALYTICS_DATA_DIR, DATA_DIR, KNOWLEDGE_DATA_DIR, LOGS_DIR, ROOT

    assert ROOT == REPO_ROOT, f"ROOT 算成了 {ROOT}，应该是仓库根 {REPO_ROOT}"
    assert ROOT.name != TOP_LEVEL, f"ROOT 落在包里了（{ROOT}），说明又有人在数父目录"

    # 数据不进包目录：`find ecom -name "*.py"` 不该被 14M 语料绊住
    assert DATA_DIR == REPO_ROOT / "data"
    assert KNOWLEDGE_DATA_DIR == REPO_ROOT / "data" / "knowledge"
    assert ANALYTICS_DATA_DIR == REPO_ROOT / "data" / "analytics"
    assert LOGS_DIR == REPO_ROOT / "logs"
    for p in (DATA_DIR, KNOWLEDGE_DATA_DIR, ANALYTICS_DATA_DIR, LOGS_DIR):
        assert PKG_ROOT not in p.parents, f"{p} 落在包目录里了"


def test_the_three_settings_modules_agree_on_the_root():
    """三个子系统的 settings 都要落在同一个根上——这正是这次合并要买的东西。

    它们各自保留自己那份 settings（旋钮不同），但"根在哪"只能有一个答案。
    """
    from ecom.knowledge.settings import KNOWLEDGE_DATA_DIR as kb_data
    from ecom.analytics.settings import settings as analytics_settings

    assert kb_data == REPO_ROOT / "data" / "knowledge"
    assert analytics_settings.DATA_DIR == REPO_ROOT / "data" / "analytics"
    assert analytics_settings.RAW_CSV_PATH == REPO_ROOT / "data" / "analytics" / "raw" / "sales.csv"


def test_the_repo_has_exactly_one_top_level_package():
    """顶层只能有一个 `ecom/`。

    这条测试是**合并前那条的接任者**，但守的东西已经不同了。原来它守的是
    「本项目永远不拥有 `src/` 和 `config/`」——因为那两个名字属于 sibling 仓库，
    同一进程里谁先占到谁说了算，而且撞了不报错。合并后 sibling 没有了，
    这个前提自然消失。

    要守的变成了：**再也不要出现第二个顶层包**。老问题不会以老样子回来，
    但它会以"再加个 `scripts/` 放几个脚本""再加个 `utils/` 放点公共代码"的样子回来
    ——而那种拆分一旦开始，下一个撞名只是时间问题。
    """
    # tests/ 也有 __init__.py（三套测试并成一套后需要它），它不是库的一部分。
    # 这份豁免名单写死在这里：往上面加名字的人得先解释为什么。
    not_library = {"tests"}

    # 迁移中的宽容：`cs/` 是编排层，合并的第三步才搬进 ecom/orchestrator/。
    # 搬到那一步，这个集合收成 {TOP_LEVEL}，断言从"只许更少"改成等号。
    # 注意它宽容的**只有 cs**：多出任何别的名字（比如有人加了个 utils/）照样红。
    allowed = {TOP_LEVEL, "cs"}

    top_level_packages = sorted(
        p.name
        for p in REPO_ROOT.iterdir()
        if p.is_dir() and (p / "__init__.py").is_file() and p.name not in not_library
    )
    assert set(top_level_packages) <= allowed, (
        f"仓库根有这些顶层包：{top_level_packages}；只该有 {sorted(allowed)}。"
        "要放新代码就放进 ecom/ 下的某个子系统，别在根上再开一个命名空间。"
    )


def test_the_analytics_worker_still_goes_through_a_process_boundary():
    """数据分析 worker 只能起子进程，不许直接 import `ecom.analytics.*`。

    合并前这条防的是"两个 `src/` 在同一个进程里撞上"。撞名现在不可能了，
    **但边界本身要留下**，理由换了：这个子系统执行的是模型生成的代码，
    需要一个能被真杀死的边界。进程内的"超时"只是不再等它，杀不掉。
    """
    worker = next(REPO_ROOT.rglob("workers/analytics.py"), None)
    assert worker is not None, "找不到编排层的 analytics worker"
    source = worker.read_text(encoding="utf-8")

    assert "subprocess" in source, f"{worker} 不再用子进程了？先读 ecom/analytics/__init__.py"
    assert not re.search(r"^\s*(from|import)\s+ecom\.analytics\b", source, re.M), (
        f"{worker} 直接 import 了 ecom.analytics——这就把子进程边界绕过去了。"
    )
