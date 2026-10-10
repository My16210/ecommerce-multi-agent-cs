"""
三个 worker。

注意这里**不 import 任何 worker**：knowledge 会去 import sibling 仓库
rag_knowledge_base，analytics 依赖子进程和另一个仓库，order 依赖数据库文件。
一旦在包初始化时把它们拉起来，那么"缺其中一个仓库"就会让整个包 import 失败，
连只用得起订单那一路的部署都起不来。注册表在 `base._registry()` 里延迟建。
"""
