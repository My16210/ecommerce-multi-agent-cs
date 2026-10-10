"""
pytest 公共夹具。

原则：**全部离线**——不连真 Milvus、不落地真 Chroma、不发任何网络请求、
不需要任何 API key。所以 `uv run pytest` 在任何机器上都能跑。

假客户端在 tests/fakes.py，那里写了为什么宁可用假的也不用真的。
"""
import pytest

from tests.knowledge.fakes import install_fake_chroma, install_fake_pymilvus


@pytest.fixture(params=["chroma", "milvus"])
def store_factory(request, monkeypatch, tmp_path):
    """按后端参数化，返回一个「造一个干净 store」的可调用对象。

    契约类用例的主体对两个后端完全一致，差异全部收在这个夹具里——
    这恰恰是抽象层想要的效果：调用方不该感觉到后端的存在。
    如果哪天某个契约用例非要按后端分支，那说明契约本身漏了一条。
    """
    if request.param == "chroma":

        def make():
            return install_fake_chroma(monkeypatch, tmp_path).ChromaStore()

    else:

        def make():
            return install_fake_pymilvus(monkeypatch).MilvusStore()

    return make
