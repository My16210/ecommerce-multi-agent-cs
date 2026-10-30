"""沙箱的两个层次：AST 静态校验 + 受限 builtins 执行。

注意点：本方案**不做子进程隔离、不做超时**，因此 `while True: pass` 这类死循环
不会被静态拦截（见 test_infinite_loop_is_a_known_limitation）。
"""
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

# 这条 import 必须排在 importorskip 之后：sandbox 里用了 pandas，本机没装时
# 上面那行会先把整个文件 skip 掉，而不是在 import 阶段就崩掉整个测试会话。
# 所以"不在文件最上面"是故意的，noqa 是有内容的，不是没清的遗留。
from ecom.analytics.core.sandbox import CodeSecurityError, run_code, validate_code  # noqa: E402


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "地区": ["华东", "华北", "华东"],
            "商品名": ["iPhone15", "华为Mate60", "iPhone15"],
            "销量": [12, 8, 5],
            "金额": [71988, 47992, 29995],
        }
    )


# ---------- 静态校验：应当拒绝 ----------

@pytest.mark.parametrize(
    "code",
    [
        "import os\nresult = os.getcwd()",
        "from os import system",
        "import subprocess",
        "import socket",
        "import sys",
        "result = __import__('os').getcwd()",
        "result = open('a.txt').read()",
        "result = eval('1+1')",
        "exec('x = 1')",
        "result = compile('1', '<x>', 'eval')",
        "result = getattr(df, 'to_csv')",
        "result = df.__class__",
        "result = ().__class__.__bases__[0].__subclasses__()",
        "result = (lambda: 0).__globals__",
        "df.to_csv('leak.csv')",
        "result = pd.read_csv('leak.csv')",
        "result = np.load('leak.npy')",
        "plt.savefig('leak.png')",
        "CHART_PATH = 'leak.png'\nplt.savefig(CHART_PATH)",
        "from pandas import read_csv",
        "savefig = plt.savefig\nsavefig('leak.png')",
        "result = pd.io.common.get_handle('x')",
        "df.to_string(buf='leak.txt')",
        "df.to_string('leak.txt')",
        "s = df.to_string\ns(buf='leak.txt')",
        "df.to_markdown('leak.md')",
        # savefig 的接收者是免检区的旧漏洞：`(open(...) and plt).savefig(...)`
        # 能借这个位置调用任意函数，而实参检查只看到 CHART_PATH。
        "(plt.plot.__globals__['__builtins__']['open']('leak.txt', 'w') and plt).savefig(CHART_PATH)",
        # match / except 的绑定名是字符串字段，不是 Name 节点，visit_Name 看不到。
        "match 'leak.png':\n    case CHART_PATH:\n        plt.savefig(CHART_PATH)",
        "def f(*CHART_PATH):\n    result = 1",
        "def f(**CHART_PATH):\n    result = 1",
        "result = np.lib.format.open_memmap('leak.npy', mode='w+', dtype='float32', shape=(1,))",
        "from . import something",
    ],
)
def test_dangerous_code_is_rejected(code):
    with pytest.raises(CodeSecurityError):
        validate_code(code)


# ---------- 静态校验：应当放行 ----------

@pytest.mark.parametrize(
    "code",
    [
        "import pandas as pd\nresult = df['金额'].sum()",
        "from pandas import DataFrame\nresult = df['销量'].mean()",
        "import math\nresult = math.sqrt(16)",
        "result = np.array([1, 2, 3]).sum()",
        "result = df.groupby('地区')['金额'].sum().sort_values(ascending=False).index[0]",
        "plt.bar(['a'], [1])\nplt.savefig(CHART_PATH)\nplt.close()\nresult = 'ok'",
        "result = df.groupby('地区')['金额'].sum().to_string()",
        "result = df.to_string(index=False)",
    ],
)
def test_legit_code_passes_static_check(code):
    validate_code(code)  # 不抛异常即通过


def test_syntax_error_is_reported_not_raised_as_security_error():
    with pytest.raises(SyntaxError):
        validate_code("def (")


# ---------- 执行层 ----------

def test_run_code_computes_result(df, tmp_path):
    out = run_code("result = df['金额'].sum()", df, tmp_path, tmp_path / "chart.png")
    assert out["error"] is None
    # 金额三行 71988 + 47992 + 29995 = 149975。
    # 这里原来写的是 131975——是换数据之前那版 fixture 的和，**断言没跟着改**。
    # 它在宿主上一直显示为红，但原因被"缺 matplotlib"盖住了（宿主上根本没跑到这一行）。
    assert out["result"] == "149975"
    assert out["chart_generated"] is False
    assert out["chart_path"] == ""


def test_run_code_rejects_before_executing(df, tmp_path):
    out = run_code("import os\nresult = 1", df, tmp_path, tmp_path / "chart.png")
    assert out["error"] is not None
    assert "安全策略" in out["error"]
    assert out["chart_generated"] is False


def test_run_code_reports_runtime_error(df, tmp_path):
    out = run_code("result = df['不存在的列'].sum()", df, tmp_path, tmp_path / "chart.png")
    assert out["error"] is not None
    assert out["result"] == ""


def test_run_code_marks_chart_generated(df, tmp_path):
    pytest.importorskip("matplotlib")
    chart = tmp_path / "chart.png"
    out = run_code(
        "plt.bar(df['地区'], df['金额'])\nplt.savefig(CHART_PATH)\nplt.close()\nresult = 'ok'",
        df,
        tmp_path,
        chart,
    )
    assert out["error"] is None
    assert out["chart_generated"] is True
    assert Path(out["chart_path"]).exists()


def test_stale_chart_is_cleared_so_it_is_not_reported_as_new(df, tmp_path):
    chart = tmp_path / "chart.png"
    chart.write_bytes(b"stale")
    out = run_code("result = 1", df, tmp_path, chart)
    assert out["chart_generated"] is False
    assert not chart.exists()


def test_no_result_variable_falls_back(df, tmp_path):
    out = run_code("x = 1", df, tmp_path, tmp_path / "chart.png")
    assert out["result"] == "没有返回结果"


def test_infinite_loop_is_a_known_limitation():
    """已知限制：不做超时，死循环不会被拦。这条用例只是把限制写成文档，不是断言它能被拦住。"""
    code = "while True:\n    pass"
    validate_code(code)  # 静态校验通过 —— 说明超时确实没覆盖


def test_run_code_blocks_pandas_file_reads(df, tmp_path):
    """`pd.read_csv` 不经过 open()，必须由属性黑名单单独拦。"""
    out = run_code("result = pd.read_csv('leak.csv')", df, tmp_path, tmp_path / "chart.png")
    assert out["error"] is not None
    assert "安全策略" in out["error"]


def test_run_code_blocks_to_string_buf_write(df, tmp_path):
    """`to_string(buf=...)` 是 to_string 唯一的写文件入口，必须单独拦。

    这是 2026-10-30 第二轮加固时才发现的：to_string 是项目推荐给模型的
    "安全输出方法"，但它的第一个参数 buf 就是路径。
    """
    target = tmp_path / "leak.txt"
    out = run_code(
        f"result = df.to_string(buf=r'{target}')",
        df,
        tmp_path,
        tmp_path / "chart.png",
    )
    assert out["error"] is not None
    assert "安全策略" in out["error"]
    assert not target.exists()


def test_savefig_may_only_write_chart_path(df, tmp_path):
    """出图是正路，但只许写题目给的 CHART_PATH。"""
    pytest.importorskip("matplotlib")
    out = run_code(
        "plt.plot([1, 2])\nplt.savefig('escape.png')\nresult = 'ok'",
        df,
        tmp_path,
        tmp_path / "chart.png",
    )
    assert out["error"] is not None
    assert "安全策略" in out["error"]
    assert not (tmp_path / "escape.png").exists()


def test_chart_path_cannot_be_rebound(df, tmp_path):
    """改写 CHART_PATH 后再 savefig，等于绕过"只许写这个路径"。"""
    out = run_code(
        "CHART_PATH = 'escape.png'\nplt.savefig(CHART_PATH)",
        df,
        tmp_path,
        tmp_path / "chart.png",
    )
    assert out["error"] is not None
    assert "安全策略" in out["error"]
