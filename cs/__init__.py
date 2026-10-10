"""电商智能客服：Supervisor + Worker 多 Agent 系统。

包名是 `cs` 而不是 `src`，是被逼的：知识库 worker 要把 sibling 仓库
rag_knowledge_base 的顶层 `src` 包放进 sys.path 才能 import 它，
本项目自己再叫 `src` 就会撞名（两个仓库同名顶层包，先 import 的那个赢，
另一个的 `from src.x import y` 会静默解析到错误的模块）。所以这里让开名字。
"""
