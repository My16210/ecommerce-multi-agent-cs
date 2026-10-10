"""
测试包。三套测试并成一套之后，这里只需要做一件事：让
`from tests.orchestrator.conftest import ...` 这类绝对导入在 pytest 和
直接运行时都成立。

    tests/orchestrator/   编排层：路由、汇总、决策、订单工具、回环
    tests/knowledge/      知识库：检索、融合、rerank、评测、人审
    tests/analytics/      数据分析：沙箱、图的路由、代码审查解析
    tests/test_package_layout.py   跨子系统：包结构与 import 路径

最后一类不在任何子系统下面，因为它测的不是任何一个子系统的行为，
而是"三个子系统拼在一起有没有搬干净"——合并这件事本身的回归测试。

合并前这三套各有各的自举方式（rag 和编排层靠 pyproject 的 pythonpath，
sales_agent 靠 `sys.path.insert(0, ...)`）。现在统一由 pyproject 管，
那 9 处 `sys.path.insert` 全删了。
"""
