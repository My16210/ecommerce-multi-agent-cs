"""守 ingest 的入库前检查：同名文件必须拦下来。

背景：chunk 的 source 取的是 `os.path.basename(file_path)`（splitter.py），
而 add_chunks 按 source 先删后加。所以两个不同子目录下的**同名文件**会共用
一个 source，后入库的那份把先入库的整份删掉。

不报错、不警告。症状只是"库里的条数比盘上少了一整份文档，而且永远召回不到
先入库的那份"——查起来要一路查到 splitter 才知道 source 是 basename。

语料以前平铺一层，撞名不可能发生；现在有了子目录（电商客服示例/、法规原文/），
这件事从"不可能"变成了"迟早会"。这里钉住的是那道闸门。
"""
from pathlib import Path

from ecom.knowledge.cli.ingest import duplicate_basenames, find_documents


def test_distinct_names_pass():
    files = [Path("a/售后政策.md"), Path("b/退换货流程.md")]
    assert duplicate_basenames(files) == {}


def test_same_name_in_two_directories_is_caught():
    files = [Path("电商客服示例/售后政策.md"), Path("法规原文/售后政策.md")]

    clashes = duplicate_basenames(files)

    assert set(clashes) == {"售后政策.md"}
    assert len(clashes["售后政策.md"]) == 2  # 两条路径都要报出来，别只说"撞了"


def test_same_file_listed_once_is_not_a_clash():
    """同一路径重复出现不算撞名——那是调用方给的重复项，不是两份文档"""
    files = [Path("a/售后政策.md"), Path("a/售后政策.md")]
    assert duplicate_basenames(files) == {}


def test_find_documents_walks_subdirectories(tmp_path):
    """语料现在分目录放，扫描必须递归进去"""
    (tmp_path / "电商客服示例").mkdir()
    (tmp_path / "电商客服示例" / "售后政策.md").write_text("x", encoding="utf-8")
    (tmp_path / "法规原文").mkdir()
    (tmp_path / "法规原文" / "电子商务法.md").write_text("y", encoding="utf-8")
    (tmp_path / "忽略我.png").write_bytes(b"\x89PNG")

    names = sorted(path.name for path in find_documents(tmp_path))

    # 只比集合：扫描顺序取决于路径字符串，拿它当断言等于把测试绑在文件系统上
    assert set(names) == {"电子商务法.md", "售后政策.md"}
