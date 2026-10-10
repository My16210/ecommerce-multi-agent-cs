# Retrieval - RAG检索增强生成

> 尚硅谷大模型技术之LangChain V1.1.0 

------

## 一、大模型的困境

### 1.1 四大局限

| 局限         | 说明                                                         | 影响               |
| ------------ | ------------------------------------------------------------ | ------------------ |
| **知识滞后** | LLM因海量参数，需要花费相当的物力与时间进行预训练和微调，还需各种安全测试与风险评估，因此存在知识滞后 | 无法回答最新事件   |
| **知识缺失** | 在专有领域，LLM无法学习到所有的专业知识细节，面向专业领域的提问时，无法给出可靠准确的回答 | 专业领域回答不准确 |
| **幻觉**     | LLM在生成回答时，可能会"胡言乱语"，体现为错误陈述、编造事实、错误的复杂推理或语境理解不足 | 可信度降低         |
| **不可追溯** | 模型生成的答案无法追溯来源                                   | 难以验证和审计     |

### 1.2 幻觉的成因

- 训练知识存在偏差，错误信息被LLM学习后在输出中复现
- LLM训练时过度泛化，将普通的模式应用在特定场合导致不准确输出
- LLM本身没有真正学习到训练数据中深层次的含义，在需要深入理解或复杂推理的任务中出错
- LLM缺乏某些领域的相关知识，面临相关问题时编造不存在的信息

大模型生成内容的不可控，尤其在金融和医疗等领域，一次金额评估的错误、一次医疗诊断的失误，哪怕只出现一次都是致命的。但这些错误对于非专业人士来说难以辨识。目前还没有能够百分之百解决这种情况的方案。

> 面对这些局限，我们该如何让大模型回答得更准确、更可靠？这就引出了本节的核心主题——RAG。

------

## 二、RAG概述

### 2.1 什么是RAG

RAG（Retrieval-Augmented Generation，检索增强生成）的基本思想为：将传统的生成式大模型和实时信息检索技术相结合，为大模型补充来自外部的相关数据和上下文，来帮助大模型生成更加准确可靠的内容。这使得大模型在生成内容时可以依赖实时与个性化的数据和知识，而非仅仅依赖训练知识。**就相当于在大模型回答时给它一本参考书。**

当应用需求集中在利用大模型回答特定私有领域的知识，且知识库足够大时，除了微调大模型外，RAG就是非常有效的一种解决方案。LangChain对这一流程提供了完整的解决方案。

<img src="images/1、RAG 架构流程.png" style="zoom:67%;" />

### 2.2 RAG优缺点

**优点**：

- 相比提示词工程，RAG有更丰富的上下文和数据样本，不需要用户提供过多的背景描述，就能生成符合预期的答案
- 相比模型微调，RAG可以提升问答内容的时效性和可靠性
- 在一定程度上保护了业务数据的隐私性

**缺点**：

- 每次问答都涉及外部系统数据检索，响应时延相对较高
- 引用的外部知识数据会消耗大量的模型Token资源

### 2.3 RAG vs 其他方案

| 方案           | 优势                         | 劣势                           | 适用场景       |
| -------------- | ---------------------------- | ------------------------------ | -------------- |
| **提示词工程** | 简单快速，无需额外资源       | 效果有限，难以处理复杂知识     | 简单任务       |
| **RAG**        | 知识可更新，可追溯，成本较低 | 响应延迟，需维护向量库         | 知识密集型应用 |
| **微调**       | 学习领域风格，响应快速       | 知识更新难，成本高，有幻觉风险 | 特定格式/风格  |
| **RAG + 微调** | 结合两者优势                 | 复杂度高，成本最高             | 复杂企业应用   |

### 2.4 RAG应用场景

<img src="images/2、RAG的应用场景.png" style="zoom:67%;" />

### 2.5 完整流程总览

典型的RAG有两个主要阶段：**索引阶段（离线）** 和 **检索生成阶段（在线）**。

**索引阶段**：从各种数据源加载数据 → 将文档切分为小块 → 对文本块进行嵌入 → 存储嵌入向量。

**检索生成阶段**：根据用户输入，使用检索器从存储中检索相关文本块 → 大模型使用包含问题和检索结果的提示生成回答。



<img src="images/3、RAG完整数据流程.png" style="zoom:67%;" />

> 接下来的第三到第八章，将沿着这条数据流，逐步讲解每一个环节的实现。

------

## 三、文档加载

### 3.1 为什么要加载文档

RAG的核心是"先检索，再生成"。但检索的前提是——系统里得先有数据。企业的知识散落在各种格式的文件中：TXT、Markdown、Word、PDF、CSV、网页……这些原始文件格式各异，大模型无法直接使用。

因此，RAG的第一步就是把这些异构数据**统一加载为标准格式**，为后续的切分、嵌入、检索做好准备。LangChain实现和集成了众多文档加载器，方便从不同格式的文件中加载数据。可在 [LangChain Document Loaders](https://docs.langchain.com/oss/python/integrations/document_loaders) 查看所有集成的加载器。

### 3.2 Document 对象

LangChain将所有加载后的文档统一抽象为 `Document` 对象，不论原始文件是PDF还是网页，加载后都变成同一种结构：

元数据：

| 属性         | 说明                                 |
| ------------ | ------------------------------------ |
| page_content | 文本内容字符串                       |
| metadata     | 包含元数据的字典，如文档来源、页码等 |
| id           | 可选，文档标识符                     |

所有文档加载器都实现了 BaseLoader 接口，提供 `load()`（一次性加载）和 `lazy_load()`（惰性加载，适合大文件）两种方法。

### 3.3 加载器类型概览

<img src="images/4、文档加载器概览.png" style="zoom:67%;" />

### 3.4 加载 Markdown

Markdown 是一种**半结构化、机器可读**的文本标记格式，通过特定语法标记出标题、段落、有序列表、无序列表等结构信息。不同层级的文本在 Markdown 中有明确且统一的表示方式（如 `#`、`##`、`-` 等），解析库可以稳定地识别并利用这些结构。

可以使用 Unstructured 文档加载器来加载 Markdown 文件。Unstructured.io 对 Markdown 的解析流程大致为：

1. 按照 Markdown 语法结构进行切分：标题、列表等会被切分成单独的 element；
2. 对同一标题下的正文，再按段落进行切分：不同段落（通过空行或 `\n` 区分）会被拆成多个 element。

```python
# uv add markdown langchain-community "unstructured[md]"

from langchain_community.document_loaders import UnstructuredMarkdownLoader

def markdown_loader_demo(file_path: str):
    """
    使用 UnstructuredMarkdownLoader 加载 Markdown 文件，
    并按 elements 模式打印切分后的内容。
    """
    loader = UnstructuredMarkdownLoader(
        file_path,
        encoding="utf-8",
        mode="elements",  # elements：按标题、段落等元素切分
    )
    docs = loader.load()
    for i, doc in enumerate(docs):
        print(f"=== Element {i} ===")
        print(doc.page_content)
        print("metadata:", doc.metadata)
        print("============\n")

markdown_loader_demo("./assets/sample.md")

```

### 3.5 加载Docx

现代 Word 文档（.docx 格式）本质上是一种**半结构化、机器可读**的文件格式。.docx 实际上是一个以 XML 为核心的压缩容器，内部的 XML 标签（例如 `<w:p>` 表示段落、`<w:r>` 表示文本运行块）严格规定了文档在物理层面的结构，解析库可以依靠这些标签精确提取段落、文本块等信息。

但与 Markdown 不同的是，Word 对“标题层级”的语义约束较弱：用户可以通过修改字体大小、加粗、颜色等方式“手动做出一个标题样式”，却不一定使用 Word 内置的“标题 1 / 标题 2”等样式。对于解析库来说，这些只是普通段落的格式变化，难以在不同文档之间用统一标准还原出清晰的标题层级结构，这也是 Word 解析的主要难点之一。

使用 Unstructured.io 的 Loader 解析 .docx 时，目前通常只会按照**换行**对文件进行切分成多个 element，不能可靠地区分“正文段落”和“各级标题”，因此会丢失语义化的标题层级信息：

```python
# 依赖安装（在终端中执行一次即可，而不是在 Python 里执行）
# uv add langchain-community "unstructured[docx]"

from pathlib import Path
from langchain_community.document_loaders import UnstructuredWordDocumentLoader


def word_loader_demo(file_path: str, start: int = 0, end: int = 20):
    """
    使用 UnstructuredWordDocumentLoader 加载 Word 文档（.docx），
    并打印指定区间的元素内容和元数据。

:param file_path: .docx 文件路径
    :param start: 打印的起始索引
    :param end: 打印的结束索引（不含）
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path.resolve()}")

    loader = UnstructuredWordDocumentLoader(
        file_path=str(file_path),
        mode="elements",
    )
    docs = loader.load()

    print(f"总共解析得到 {len(docs)} 个元素\n")

    end = min(end, len(docs))

    for i, doc in enumerate(docs[start:end], start=start):
        print(f"=== Element {i} ===")
        print(doc.page_content)
        print("metadata:", doc.metadata)
        print("============\n")

word_loader_demo("assets/sample.docx")

```

对于标题层级信息敏感的.docx文件，上面的方式会丢失标题层级信息，此时可通过MinerU进行处理。





**非结构化、半结构化、结构化的对比**：

| 类型       | 示例                   | 结构化程度             | 机器可读性            | 结构特点理解成一句话                           |
| ---------- | ---------------------- | ---------------------- | --------------------- | ---------------------------------------------- |
| 纯文本     | `.txt` 随便写一篇文章  | 非结构化（几乎没标记） | 低（要靠规则/模型猜） | 一堆字堆在一起，哪里是标题全靠人眼看           |
| Markdown   | `.md` 文档             | 半结构化               | 高（规则简单固定）    | 用 `# - *` 等符号明显标出标题、列表、段落      | Word .docx | `.docx` 文档           | 半结构化               | 很高（XML 标签很细）  | XML 里标出段落、文字块等，但“是不是标题”不统一 |
| 数据库     | SQL 表、Excel 严格表头 | 高度结构化             | 极高                  | 列名、类型、每行含义都固定，机器最容易处理     |

### 3.6 加载 PDF

PDF存在多种来源格式（扫描版、电子文本版、混合版），多种布局格式（单列、双列、竖排），并包含段落、标题、页眉页脚、表格、数学公式、图片等各种元素。因此PDF解析存在很多挑战，对于复杂PDF，需要进行布局检测、文本提取、表格解析、公式识别等处理。

#### 3.6.1 MinerU 介绍

MinerU 是一款将 PDF 转化为机器可读格式（如 Markdown、JSON）的工具，便于后续按任意结构进行抽取和处理。
为了从复杂版面中更准确地识别出标题、段落、图表等结构，MinerU 支持配置使用 VLM（Vision-Language Model，视觉语言模型）进行文档解析；其开源的 MinerU2.5 模型在多项文档理解基准测试中均达到 SOTA 水平。

MinerU2.5 采用两阶段解析策略：先在下采样图像上做高效全局布局分析，再在原始分辨率裁剪图像上对文本、公式、表格等进行细粒度识别 。

#### 3.6.2  MinerU两阶段流程

**两阶段解析策略流程**：先全局再局部，先找块再看字。通过把"找结构"和"读内容"拆成两步，MinerU2.5 在保证解析精度的同时，显著降低了计算成本。

<img src="images/5、MinerU两阶段策略.png" style="zoom:67%;" />

| 阶段       | 使用分辨率   | 主要任务                                                     | 主要优势                         |
| ---------- | ------------ | ------------------------------------------------------------ | -------------------------------- |
| **阶段 1** | 低（下采样） | 全局版面分析：在缩小后的整页“缩略图”上识别各区域的类型与大致位置（标题、正文、表格、图片、公式等） | 计算量小、速度快，适合做全局结构 |
| **阶段 2** | 高（原始）   | 局部内容识别：在原始分辨率上对已定位区域做 OCR、表格结构解析、公式识别等 | 识别精度高，只对局部区域精细计算 |

MinerU 支持对 PDF、Word、PPT、图片等进行解析，涵盖图像提取、OCR、公式和表格解析等功能。项目已完全开源，支持本地 Docker 部署（仓库地址：https://github.com/opendatalab/MinerU），也可通过官网 API 调用服务，使用前需申请 API_KEY 并配置到环境变量。



#### 3.6.3 使用MinerU API上传文件

> Miner解析（word）(pdf)---->MD文档中都是一级标题结构（下载mineru到本地 修改Minueru.json文件 修改模型辅助标题识别配置）

**加载在线pdf**

```python
import requests

token = os.getenv('MINERU_TOKEN')
url = "https://mineru.net/api/v4/extract/task"
header = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {token}"
}
data = {
    "url": "https://cdn-mineru.openxlab.org.cn/demo/example.pdf",
    "model_version": "vlm"
}

res = requests.post(url,headers=header,json=data)
print(res.status_code)
print(res.json())
print(res.json()["data"])
```

**获取解析结果**：

```python
import requests

task_id = "62bc54b0-0369-4150-9d07-5dc8d3d281ed"
url = f"https://mineru.net/api/v4/extract/task/{task_id}"
header = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {token}"
}

res = requests.get(url, headers=header)
print(res.status_code)
print(res.json())
print(res.json()["data"])
```

**加载本地pdf**

```python
def mineru_upload_file_demo():
    import requests
    from pathlib import Path

if not token:
        raise RuntimeError("环境变量 MINERU_TOKEN 未设置")

    # 1. 申请上传 URL
    url = "https://mineru.net/api/v4/file-urls/batch"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    # 本地文件路径列表（可扩展为多文件）
    file_paths = [
        r"assets\尚硅谷大模型技术之NLP1.0.2.pdf"
    ]

files_info = []
    for i, p in enumerate(file_paths):
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"文件不存在: {p}")
        files_info.append({
            "name": p.name,
            "data_id": f"data_{i}",  # 自己给每个文件一个 data_id 标识
        })

    data = {
        "files":files_info,
        "model_version": "vlm",
    }

resp = requests.post(url, headers=headers, json=data)
    if resp.status_code != 200:
        print(f"申请上传 URL 失败，状态码：{resp.status_code}，响应内容：{resp.text}")
        return None

    result = resp.json()
    if result.get("code") != 0:
        print(f"申请上传 URL 失败，reason: {result.get('msg')}")
        return None

    print(result)

    batch_id = result["data"]["batch_id"]
    urls = result["data"]["file_urls"]
    print(f"申请上传 URL 成功，batch_id: {batch_id}")
    print("file_urls:", urls)

# 2. 逐个上传文件到对应的临时 URL
    for i, upload_url in enumerate(urls):
        path = Path(file_paths[i])
        with path.open("rb") as f:
            res_upload = requests.put(upload_url, data=f)
        if res_upload.status_code == 200:
            print(f"{path.name} 上传成功")
        else:
            print(f"{path.name} 上传失败, 状态码: {res_upload.status_code}, 响应: {res_upload.text}")

    return batch_id, files_info


mineru_upload_file_demo()
```

**获取解析结果**

```python
def mineru_check_result_demo(batch_id: str, max_retries: int = 30, interval: int = 3):
    """
    轮询 MinerU 批量解析结果，直到任务完成或超过重试次数。

    :param batch_id: 调用 /extract/task 时返回的 batch_id
    :param max_retries: 最大轮询次数
    :param interval: 每次轮询间隔秒数
    """
    import os
    import time
    import requests

    token = os.getenv("MINERU_TOKEN")
    if not token:
        raise RuntimeError("环境变量 MINERU_TOKEN 未设置")

url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    for i in range(max_retries):
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            print(f"请求失败，状态码：{resp.status_code}，响应内容：{resp.text}")
            return None

data = resp.json()
        # 结构示例：data["data"]["extract_result"] 是列表
        extract_list = data.get("data", {}).get("extract_result", [])
        if not extract_list:
            print("返回结果中没有 extract_result 字段：", data)
            return None

        item = extract_list[0]
        state = item.get("state")
        print(f"第 {i+1} 次查询，当前状态：{state}")

if state == "done":
            full_zip_url = item.get("full_zip_url")
            print("任务已完成，结果 zip 下载地址：", full_zip_url)
            return full_zip_url
        elif state in ("failed", "error"):
            print("任务失败，返回信息：", item)
            return None

        # 还未完成，等待后重试
        time.sleep(interval)

    print(f"超过最大重试次数（{max_retries}），任务仍未完成，请稍后重试或在控制台查看状态。")
    return None


# 示例调用（替换为你自己的 batch_id）
mineru_check_result_demo("c9066450-dca3-4396-bdb3-55d0dac64ed2")

```

MinerU解析后有多种输出格式，最简单的后续处理方式是通过解析后得到的Markdown文件，再利用Markdown解析器进一步解析。

> 文档加载完成后，我们得到了完整的Document对象。但一整篇文档往往太长，无法直接用于检索——下一步需要将其切分为更小的片段。

------

## 四、文档切分

### 4.1 为什么要切分

获取 Document 对象后，需要将其切分成 Chunk（文本块），以 Chunk 为基本单位进行存储和检索。

| 问题             | 说明                                                         |
| ---------------- | ------------------------------------------------------------ |
| **Token 限制**   | LLM 的上下文窗口有限，无法一次性处理整篇长文档               |
| **检索不精确**   | 如果不切分，检索时返回整篇文档，噪声过多，LLM 容易产生幻觉   |
| **向量化成本高** | 整篇文档做 Embedding，计算和存储成本都很高                   |
| **语义模糊**     | 嵌入模型对短文本的语义表达更精确，长文本的向量表示会"稀释"关键信息 |

### 4.2 切分的核心考量

既然要切分，那么**怎么切、在哪儿切**就成了关键问题。切分需要平衡三个维度：

| 考量维度     | 说明                             | 过犹不及的后果                     |
| ------------ | -------------------------------- | ---------------------------------- |
| **长度控制** | 每个块不能太大，也不能太小       | 太大：检索噪声多；太小：语义不完整 |
| **语义完整** | 尽量不让句子被切断，保持意思连贯 | 断句会导致上下文断裂，影响理解     |
| **边界合理** | 在自然分隔处切分，而非强行截断   | 强行截断会丢失重要信息             |

基于这些考量，LangChain 提供了多种切分策略供选择。

### 4.3 切分策略对比

| 策略                 | 原理                                                       | 优点                     | 缺点                 |
| -------------------- | ---------------------------------------------------------- | ------------------------ | -------------------- |
| **固定长度切分**     | 按固定字符/Token数切分                                     | 简单直接，块大小均匀     | 可能在不当位置断句   |
| **语义切分**         | 对相邻句子嵌入，找语义变化大的位置作为切分点               | 语义完整性最好           | 速度慢，块大小不均衡 | **递归多分隔符切分** | 依次尝试多个分隔符切分，优先用大分隔符，逐步降级到小分隔符 | 保持语义完整，块大小可控 | 需要调整分隔符顺序   |

### 4.4 递归字符切分器

**为什么选择它？** 固定长度切分太粗暴，语义切分太慢且块大小不可控。**递归字符切分器**介于两者之间——既保证语义完整性，又能控制块大小，是实际应用中最常用的折中方案。

#### 4.4.1 工作原理

**核心思想**：定义分隔符优先级列表（如 `["\n\n", "\n", " ", ""]`），按优先级依次尝试切分：

1. 先用最高优先级分隔符（如 `\n\n`）切分整段文本
2. 检查每个切分块是否超过 `chunk_size`
3. 若超限，对该块降级使用下一级分隔符继续切分
4. 递归直到所有块都不超限，或降级到字符级切分（`""`）

**分隔符优先级的语义含义**：

| 分隔符 | 语义边界 | 说明                       |
| ------ | -------- | -------------------------- |
| `\n\n` | 段落边界 | 最优先保持段落完整         |
| `\n`   | 行边界   | 段落超限时，尝试保持行完整 |
| ` `    | 词边界   | 行超限时，尝试保持词完整   |
| `""`   | 字符级   | 最后手段，强制按字符截断   |

<img src="images/6、工作原理.png" style="zoom:67%;" />

**关键结论**：

- 分隔符优先级越高，切分越粗粒度，语义完整性越好
- 当分隔符无法匹配时，自动降级到下一级

#### 4.4.2 重叠的作用

相邻块之间保留一部分重叠内容，防止切分边界处丢失关键信息。

<img src="images/7、OverLap示意图.png" style="zoom: 33%;" />

#### 4.4.3 代码示例

```python
# pip install langchain-text-splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

# 加载文档
docs = UnstructuredWordDocumentLoader(
    file_path="assets/sample.docx",
    mode="single"
).load()

# 切分为文本块
chunks = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", "！", "？", "……", "，", ""],  # 分隔符优先级列表
    chunk_size=400,        # 每个块的最大长度
    chunk_overlap=50,      # 相邻块重叠长度
    length_function=len,   # 长度计算函数
).split_documents(docs)

# 统计块数量
print(f"切分得到 {len(chunks)} 个文本块\n")

# 可选：查看前几个块内容和元数据
for i, chunk in enumerate(chunks[:5]):  # 只看前 5 个
    print(f"=== Chunk {i} ===")
    print(chunk.page_content)
    print("metadata:", chunk.metadata)
    print("============\n")

```

> 切分后，每个 Chunk 仍然是一段文本。为了让计算机能够衡量"语义上有多相似"，我们需要把文本转化为数值向量——这就是下一步"嵌入"要做的事。



## 五、文本嵌入

### 5.1 为什么要嵌入

切分之后，每个Chunk仍然是一段**人类语言的文字**。但后续检索的核心操作是"比较两段内容在语义上有多相似"——计算机无法直接对比两段中文的含义，它只能处理数字。

因此我们需要一个桥梁，把文字转换成计算机能够计算的形式。这个桥梁就是**嵌入（Embedding）**：通过嵌入模型，将一段文本映射为一组高维浮点数向量。经过这一转换后，语义相近的文本在向量空间中距离更近，语义无关的文本距离更远。后续检索时，只需计算用户查询向量和文档向量之间的距离，就能找到最相关的内容。

<img src="images/8、 为什么要嵌入.png" style="zoom:33%;" />

### 5.2 嵌入模型简史

那么，嵌入模型是怎么做到"理解语义"的？

2018年谷歌推出的BERT能够将文本嵌入为向量表示，但BERT并未针对有效生成**句子级别**的嵌入进行优化，由此促使了Sentence-BERT的诞生。Sentence-BERT调整了BERT的架构以及预训练任务，专门用于生成包含语义的句子嵌入向量。这些向量可通过余弦相似度等指标轻松比较，大大降低了查找相似句子的计算开销。

此后，各机构陆续推出了更多针对不同语言和场景优化的嵌入模型。

### 5.3 常用嵌入模型

| 模型                       | 机构   | 维度 | 序列长度 | 特点                       |
| -------------------------- | ------ | ---- | -------- | -------------------------- |
| **bge-large-zh**           | BAAI   | 1024 | 512      | 开源，中文效果好           |
| **bge-base-zh**            | BAAI   | 768  | 512      | 开源，中文场景常用         |
| **bge-small-zh**           | BAAI   | 512  | 512      | 开源，轻量级               |
| **bge-m3**                 | BAAI   | 1024 | 8192     | 开源，多语言，支持稀疏向量 | **text-embedding-3-small** | OpenAI | 1536 | 8192     | 多语言，性价比高           |
| **text-embedding-3-large** | OpenAI | 3072 | 8192     | 多语言，精度更高           |

### 5.4 Embedding抽象接口

了解了嵌入模型之后，如何在代码中使用它们？LangChain设计了一个统一的Embedding抽象类，不同的模型（HuggingFace、OpenAI等）都实现这套接口，切换模型时只需更换实现类，业务代码无需修改：

```python
@abstractmethod
def embed_documents(self, texts: list[str]) -> list[list[float]]:
    """将多段文档文本嵌入为向量列表"""

@abstractmethod
def embed_query(self, text: str) -> list[float]:
    """将单条查询文本嵌入为向量"""
```

### 5.5 HuggingFace嵌入

使用HuggingFace开源模型在本地完成嵌入，无需调用外部API，适合对数据隐私有要求的场景：

```python
# uv install sentence-transformers langchain_huggingface
def embedding_demo():
    from langchain_huggingface import HuggingFaceEmbeddings

    embed_model = HuggingFaceEmbeddings(
        model_name=r'D:\ai_models\huggingface_cache\bge-base-zh-v1.5'
    )

    # 单文本嵌入
    query = "你好，世界"
    query_result = embed_model.embed_query(query)
    print(len(query_result))  # 768  
    print(query_result[0:10])

# 多文本嵌入
    docs = ["你好，世界", "你好，世界"]
    res = embed_model.embed_documents(docs)
    print(type(res))  # list[list[float]]


embedding_demo()

```

### 5.6 OpenAI嵌入

使用OpenAI云端模型完成嵌入，无需本地GPU，适合快速原型开发：

```python
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

query = "什么是人工智能？"
docs = [
    "人工智能是计算机科学的一个分支",
    "机器学习是人工智能的子领域",
]

# 向量化
query_vector = embeddings.embed_query(query)
doc_vectors = embeddings.embed_documents(docs)

# 打印查询向量信息
print("=== 查询向量 ===")
print(f"查询文本: {query}")
print(f"向量维度: {len(query_vector)}")
print(f"向量前 5 维: {query_vector[:5]}")
print("=" * 40)

# 打印文档向量信息
print("=== 文档向量 ===")
for i, (text, vec) in enumerate(zip(docs, doc_vectors)):
    print(f"文档 {i}: {text}")
    print(f"向量维度: {len(vec)}")
    print(f"向量前 5 维: {vec[:5]}")
    print("-" * 40)

```

### 5.7 稠密向量的局限

上面介绍的HuggingFace和OpenAI嵌入，生成的都是**稠密向量**——每一维都有值，捕捉的是文本的整体语义。稠密向量擅长理解"意思相近"的文本，但有一个短板：**对关键词的精确匹配能力较弱**。

举个例子：用户搜索"《民法典》第236条"，稠密向量可能会返回各种民法相关的内容，但不一定精确命中包含"第236条"这个关键词的文档。

为了弥补这个不足，一些模型（如BGE-M3）除了生成稠密向量之外，还能同时生成**稀疏向量**——只记录关键词及其权重，天然擅长关键词匹配。将两者结合，就是后面检索章节会讲到的"混合检索"。

### 5.8 BGE-M3：同时生成稠密与稀疏向量

BGE-M3是BAAI推出的多语言嵌入模型，最大的特点是**一次编码，同时输出稠密向量和稀疏向量**：

- **稠密向量**：由Transformer Encoder生成，捕捉整体语义，输出固定维度向量，适合语义相似度匹配
- **稀疏向量**：同样由Transformer Encoder生成，但方式不同——对每个token分别计算重要性权重，只保留权重较高的token及其分数，输出"关键词→权重"的字典。同时补充了稀疏向量从早期TF-IDF统计方法到如今深度学习方法的演进脉络。

<img src="images/9、稠密向量vs稀疏向量.png" style="zoom:50%;" />



**对比分析：**

| 特性           | 稠密向量             | 稀疏向量         |
| -------------- | -------------------- | ---------------- |
| **维度**       | 固定 1024 维         | 动态（非零元素） |
| **存储**       | 全量存储             | 只存非零元素     |
| **语义理解**   | 强（同义词、近义词） | 弱（精确匹配）   |
| **关键词匹配** | 弱                   | 强（型号、品牌） |
| **适用场景**   | 语义检索             | 关键词检索       |



```python
# uv add FlagEmbedding==1.3.5
# uv add transformers==4.44.2
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel(model_name_or_path=r"D:\ai_models\huggingface_cache\bge-m3")

text = "标量字段通常用来存储一些元数据，并可以在搜索时通过元数据进行过滤"

res = model.encode(
    [text],
    return_sparse=True,
    return_dense=True
)

print("=== 原始文本 ===")
print(text)
print("=" * 50)

# 1. 稀疏向量（关键词及其权重，原始 id 形式）
print("=== 稀疏向量（id → 权重）===")
lexical_weights = res["lexical_weights"][0]  # batch 中第一条
# 只看前若干个，避免太长
for i, (token_id, weight) in enumerate(list(lexical_weights.items())[:20]):
    print(f"[{i:02d}] id={token_id:<5}  weight={weight:.4f}")
print("...（仅展示前 20 个）")
print("=" * 50)

# 2. 把稀疏向量的 id 转成可读 token
print("=== 稀疏向量（token → 权重）===")
sparse_tokens = model.convert_id_to_token(lexical_weights)

for i, (token, weight) in enumerate(list(sparse_tokens.items())[:20]):
    print(f"[{i:02d}] token='{token}'  weight={weight:.4f}")

# 3. 稠密向量（1024 维）
dense_vec = res["dense_vecs"][0]

print("=== 稠密向量信息 ===")
print(f"维度: {len(dense_vec)}")
print("前 10 维示例:")
print([round(x, 4) for x in dense_vec[:10]])

```

> 向量生成好了，接下来需要一个地方把它们存起来，并支持高效的相似度检索——这就是向量数据库的职责。

## 六、向量存储

### 6.1 什么是向量数据库

上一章我们已经把文本转化为了向量。现在面临一个现实问题：**这些向量存在哪里？**

能不能直接存到MySQL之类的传统数据库里？技术上可以——把向量当作一个数组字段存储。但问题出在**检索**上：

<img src="images/10、关系性数据库和向量数据库.png" style="zoom:67%;" />



简单来说，向量数据库的核心价值在于解决了传统数据库面对高维数据时的两个致命短板：

1. **查询逻辑的转变（相似度检索）**：突破了传统关系型数据库只能找“绝对相等”或“标量范围”的限制，向量数据库支持在高维度空间中寻找“最相似”的特征。
2. **海量数据的检索速度（高维索引）**：相比于传统数据库在计算相似度时无奈的“全表扫描”，向量数据库通过构建 ANN（近似最近邻）、HNSW 等专用的高维向量索引算法，巧妙地避免了全局遍历，在亿级数据中依然能实现毫秒级的快速返回。

### 6.2 主流选型

| 向量数据库        | 描述                                                         |
| ----------------- | ------------------------------------------------------------ |
| **FAISS**         | 用于高效相似性搜索和密集向量聚类的库                         |
| **Chroma**        | 开源轻量级向量数据库，有极简API                              |
| **Milvus**        | 开源云原生向量数据库，性能强悍，覆盖轻量级原型到十亿级向量的大规模系统 |
| **Pgvector**      | PostgreSQL扩展，为PostgreSQL增加向量搜索功能                 |
| **Redis**         | 开源内存数据结构存储，已原生支持向量相似性搜索               | **Elasticsearch** | 开源分布式搜索引擎，统一管理结构化、非结构化和向量数据       |

选型众多，本课程选择Milvus作为主要向量存储。为什么？因为Milvus在生产环境中使用最为普遍，支持数百亿级别的向量存储和检索，并且原生支持稠密向量和稀疏向量的混合检索——这正好对应上一章我们用BGE-M3生成的两种向量。

### 6.3 Milvus是什么

Milvus是一个**开源的云原生向量数据库**，专门为向量搜索而设计。可以把它类比为"向量世界的MySQL"——MySQL擅长结构化数据的精确查询，Milvus擅长高维向量的相似度检索。

#### 6.3.1 Milvus架构

结合官方架构图，Milvus 的设计核心可以总结为四个字："**存算分离**"。它采用了云原生和微服务架构，将整个系统解耦为四个主要层次，从而实现了极高的弹性和扩展性。

![](images/21、架构图.png)

##### 核心架构分层

**1. 接入层（Access Layer）—— Proxy**

角色：这是整个 Milvus 集群的"大门"（API 网关）。Client SDK 所有的请求都会先打到 Proxy。

功能：Proxy 负责请求的路由转发。它会将不同类型的请求分发给后端的不同组件：

- **DCL/DDL**（数据控制/定义语言，如建表、建集合）：转发给 Coordinator
- **DML**（数据操作语言，如插入向量）：转发给 Streaming Nodes
- **DQL**（数据查询语言，如向量相似度检索）：主要转发给 **Query Nodes** 执行核心检索；同时也会与 **Streaming Nodes** 交互，比如直接读取极热的流式数据。

**2. 协调层（Coordinator Service）—— Coordinator**

角色：集群的"大脑"。

功能：负责集群拓扑管理、负载均衡、数据分配和元数据管理。它会将系统的元数据（如集合的 Schema、节点的状态）注册并持久化到右侧的 Meta Storage (etcd) 中。同时，它向下的 Workers 层下发管理指令（Management）。

**3. 计算/执行层（Worker Nodes）—— 真正干活的节点**

这是图中虚线框内的部分，进一步细分为三种专注于特定任务的节点。它们可以独立进行横向扩容：

| 节点类型                        | 职责                                                         |
| ------------------------------- | ------------------------------------------------------------ |
| **Streaming Nodes**（流式节点） | 负责处理所有的 DML（写入）请求。它将数据追加（Append）到 WAL 提供持久化保障，并**负责将数据落盘（Flush）到对象存储中**，形成初始的数据段（Segments）。 |
| **Data Nodes**（数据节点）      | 专职后台优化。负责从对象存储中读取已落盘的数据，执行索引构建（Indexing）和碎片合并（Compaction），优化后写回对象存储，以提升检索效率。 | **Query Nodes**（查询节点）     | 专门负责执行向量搜索和读取请求。它从对象存储加载已索引的历史数据到内存，同时订阅 Streaming Nodes 获取实时增量数据，合并两部分结果以保证查询的**实时性和完整性**。 |

**4. 存储层（Durable Storage）—— 数据基座**

Milvus 自己不造存储轮子，而是依托成熟的第三方分布式存储组件：

| 存储组件                                 | 用途                                                         |
| ---------------------------------------- | ------------------------------------------------------------ |
| **Meta Storage** (etcd)                  | 存储集群的元数据（配置、节点信息等）                         |
| **WAL** (Write-Ahead Log - Pulsar/Kafka) | 预写日志，由 Streaming Nodes 写入，提供底层数据的持久化保证。 |
| **Object Storage** (MinIO/S3)            | 持久化存储最终的向量数据段（Segments）和构建好的索引文件     |

##### 核心工作流

理解架构最好的方式是跑通读写两条链路。从图中我们可以清晰地看到数据的流动轨迹：

**场景 A：数据插入流程 (Insert)**

<img src="images/11、数据插入流程 (Insert).png" style="zoom:67%;" />

**流程要点**：

1. **接入与路由**：

   Proxy 接收到客户端发起的 Insert 请求后，将其精准路由给后端的 Streaming Nodes。

2. **Streaming Nodes 处理（日志追加缓冲 ---> 阈值触发落盘）**：

   在这个节点，数据的处理分为明显的“即时”与“异步”两个阶段：

- **即时日志追加与内存缓冲**：数据到达后，**立刻**向 WAL（Kafka/Pulsar）追加日志，以保障数据绝对不丢失。此时，数据留在 Streaming Nodes 的内存中，成为 **Growing Segment（生长中的数据段）**。这部分内存中的“热数据”可以直接被 Query Nodes 订阅并进行“暴力扫描”检索。
   - **条件触发落盘 (Flush)**：数据不会立刻写向底层存储，而是等待满足以下**任一阈值**时，Streaming Nodes 才会触发 Flush 动作将其写向 Object Storage：
     - **容量触发**：积累到了配置的大小（默认是 512MB）。
     - **时间触发**：达到了定时刷新的时间（防止数据写入极少时一直挂在内存不落盘）。
     - **手动触发**：用户主动调用了 `flush()` 接口。
   - **形成初始段**：落盘后的数据变成了 **Sealed Segment（封存状态的初始小段）**，它安全地躺在 Object Storage 中，不再接收新数据。

3. **Data Nodes 异步优化**：

   Data Nodes 像后台的巡逻兵，异步从 Object Storage 读取这些刚刚落盘但未经优化的 Sealed Segments，默默在后台执行**索引构建（Indexing，例如构建 HNSW 索引）以及 数据碎片合并（Compaction）**。

4. **高可用检索就绪**：

   优化完成后，Data Nodes 会将带有高效索引的、规整的大 Segments 重新写回 Object Storage。至此，数据彻底转化为持久化的“冷数据”，供 Query Nodes 加载并进行极速的向量相似度检索。

> **关键理解**：Streaming Nodes 是写入链路的主角
>
> **关键注意**：在 Milvus 中：
>
> 1. 它**是**写前日志（WAL），因为它承担着保证刚写入的数据不丢失、提供数据恢复基准的职责。
> 2. 它**也是**消息队列（Queue），因为它在物理上就是用 Kafka/Pulsar 部署的，并且被系统用来做组件之间的数据解耦和广播订阅。

---

**场景 B：向量检索流程 (Search)**

在 RAG 系统中，这是最关键、也是并发最高的环节。

<img src="images/12、向量检索流程 (Search).png" style="zoom:67%;" /> 

**流程要点**：

1. Proxy 将 Search 请求路由给 Query Nodes
2. Query Nodes **同时从两个数据源检索**：
   - **历史冷数据**：从 Object Storage 加载已经被 Data Nodes 优化并构建好索引的 Segments 到内存，通过 ANN 索引（如 HNSW）进行高效检索
   - **实时热数据**：订阅 Streaming Nodes，获取刚写入但尚未落盘的增量数据，对这部分数据进行暴力扫描
3. 将两部分检索结果合并、统一按相似度排序，取 TopK 后经 Proxy 返回客户端

> **关键理解**：这种"冷热数据合并查询"的设计，保证了即使数据刚刚写入、还没来得及被 Data Nodes 优化，也能立刻被检索到，实现了**写入即可查**的实时性。



##### 架构设计优势

总结来说，Milvus 的架构设计完美体现了"存算分离"的思想：

| 优势           | 说明                                                         |
| -------------- | ------------------------------------------------------------ |
| **弹性扩展**   | 各层组件可独立横向扩容，根据业务负载灵活调整                 |
| **高可用性**   | 组件解耦，单点故障不影响整体服务                             |
| **低成本**     | 计算节点可使用廉价存储，热数据缓存，冷数据落盘               |
| **高性能**     | Query Nodes 内存计算 + Streaming Nodes 订阅保证毫秒级实时检索 |
| **写入即可查** | 冷热数据合并查询机制，新写入数据无需等待索引构建即可被检索   |

文本中提到的"解决了相似度检索和检索速度"这两个痛点，在这张架构图中得到了完美的工程体现：Query Nodes 负责在内存中极速计算相似度；而庞大的数据量和复杂的索引构建任务，则被优雅地甩给了后台的 Data Nodes 和廉价的 Object Storage，从而实现了性能和成本的最佳平衡。

#### 6.3.2 部署方式

| 方式                   | 说明                                                         | 适用场景     |
| ---------------------- | ------------------------------------------------------------ | ------------ |
| **Milvus Lite**        | 通过pip安装，本地轻量化运行，仅支持FLAT索引，仅支持MacOS和Linux | 本地开发调试 |
| **Milvus Standalone**  | 单点部署，Docker一键启动                                     | 中小规模生产 |
| **Milvus Distributed** | 分布式部署，Kubernetes集群                                   | 大规模生产   |

本课程使用Milvus Standalone方式部署：

```bash
# 加载镜像
docker load -i milvus_image.tar

# 启动服务（Linux）
bash standalone_embed.sh start

# 启动服务（Windows Docker Desktop）
standalone.bat start
```

启动后可通过Milvus官方图形化客户端**Attu**查看数据，选择token方式直接连接即可。

### 6.4 Milvus核心概念

在动手写代码之前，先理解Milvus中几个核心概念以及它们之间的关系：

<img src="images/13、Milvus核心概念.png" style="zoom: 50%;" />

#### 6.4.1 Collection与数据类型

Collection通过**Schema**定义有哪些字段及其类型。Milvus支持三大类字段：

<img src="images/14、数据类型.png" style="zoom:50%;" />

**Schema约束**：一个Schema有一个主键、最多四个向量字段和若干标量字段。主键用于唯一标识实体，支持AutoId自动生成。标量字段通常用来存储元数据（如文档来源、页码等），在搜索时可通过标量条件进行过滤。

#### 6.4.2 索引——加速搜索的关键

向量存进去之后，如果每次检索都暴力遍历所有向量，速度会很慢。**索引**是建立在数据之上的附加结构，可以大幅加快搜索速度。不同类型的向量需要不同的索引。

**稠密向量索引 - HNSW**

HNSW（分层导航小世界）是当下最常用的基于图的索引算法，具有出色的搜索精度和低延迟，但需要较高内存开销。它构建多层图（类似不同缩放级别的地图），底层包含所有数据点，上层由采样子集组成：

<img src="images/15、HNSW.png" style="zoom:50%;" />

另一种稠密向量索引是**FLAT**，采用暴力搜索，每个查询直接与所有向量比较，能保证100%召回率，但速度最慢，仅适合小数据量或对精度要求极高的场景。

> **举例理解 HNSW——"在全国找餐厅"**
>
> 假设你要在**全中国几百万家餐厅**中，找到和你口味最接近的那一家（口味 = 向量，口味的相似度 = 向量距离）。
>
> **暴力搜索（FLAT）**：拿着全国餐厅名录，从第一家挨个比对到最后一家，几百万家全走一遍，速度极慢。
>
> **HNSW 的做法**：事先准备三张不同比例的地图，搜索时**从粗到细、逐层下降**：
>
> | 层级                  | 地图比例 | 包含内容                           | 搜索动作                                         |
> | --------------------- | -------- | ---------------------------------- | ------------------------------------------------ |

> | **Layer 2（顶层）**   | 全国地图 | 只有几个省会城市（少量采样节点）   | 快速判断"目标大概在上海方向"，一步跳过几百公里   |
> | **Layer 1（中间层）** | 城市地图 | 上海的主要街区：浦东、徐汇、静安…  | 缩小范围"应该在徐汇区附近"                       |
> | **Layer 0（底层）**   | 街道地图 | 每一家餐厅都在这里（全部数据节点） | 在徐汇区的街道上逐个比较邻居，找到最合口味的那家 |
>
> **对应到向量检索的关键点**：
>
> - 每个"餐厅"就是数据库里的一个**向量**，"口味的相似度"就是**向量之间的距离**
> - **顶层节点少、连接跨度大**，可以用很少的步数跨越大范围，快速缩小搜索空间
> - **逐层下降，每层贪婪搜索**：在当前层的邻居中找离目标最近的节点，然后跳到下一层继续
> - **底层包含所有节点**，保证最终能找到真正最近的向量
> - 搜索时间复杂度从暴力遍历的 **O(n)** 降到 **O(log n)**——就像你不需要走遍全国每条街，只需要"全国→城市→街区→街道"几步就到

另一种稠密向量索引是**FLAT**，采用暴力搜索，每个查询直接与所有向量比较，能保证100%召回率，但速度最慢，仅适合小数据量或对精度要求极高的场景。



**稀疏向量索引 - SPARSE_INVERTED_INDEX**

利用倒排索引的原理，为稀疏数据创建高效的搜索结构。查询时先通过倒排索引找到包含query中token的文档，然后计算相似度分数：

<img src="images/16、倒排索引.png" style="zoom:50%;" />

#### 6.4.3 相似度度量

索引解决了"怎么快速找"的问题，而**metric_type**解决了"怎么判断相似"的问题。建索引时必须指定度量方式，检索时也要保持一致。Milvus支持三种主要度量方式：

<img src="images/17、度量类型.jpg" style="zoom:67%;" />



<img src="images/17、三种对比.png" style="zoom:67%;" />

##### 6.4.3.1 三者的数学关系

要想真正理解如何选择度量方式，必须先弄清三者在数学上的关系。假设有两个向量 **A** 和 **B**：

| 度量方式                 | 公式                                | 衡量的是                     |
| ------------------------ | ----------------------------------- | ---------------------------- |
| **IP（内积）**           | `A · B`                             | 方向 + 长度的综合相似性      |
| **COSINE（余弦相似度）** | `(A · B) / (‖A‖ × ‖B‖)`             | 纯方向相似性（消除长度影响） |
| **L2（欧氏距离）**       | `‖A - B‖² = ‖A‖² + ‖B‖² - 2(A · B)` | 空间中的绝对距离             |

**关键推导：当向量做了 L2 归一化（‖A‖ = 1 且 ‖B‖ = 1）时：**

1. **COSINE = IP**：余弦公式的分母变成 1×1 = 1，因此 `cos(θ) = A · B = IP`，两者数值完全相等
2. **L2² = 2 - 2 × IP**：因为 `‖A‖² + ‖B‖² = 1 + 1 = 2`，所以 `L2² = 2 - 2(A · B) = 2 - 2 × IP`

结论：**归一化后，三者严格等价**——IP 越大 ⇔ COSINE 越大 ⇔ L2 越小，排序结果完全一致。

##### 6.4.3.2 如何选择？

| 场景                                           | 推荐   | 原因                                                         |
| ---------------------------------------------- | ------ | ------------------------------------------------------------ |
| **稠密向量 + 文本语义检索（已归一化）**        | **IP** | BGE 等主流文本嵌入模型输出已归一化，此时 IP = COSINE，但 **IP 计算最快**（仅乘加运算，无需算模长和除法） |
| **稠密向量 + 文本语义检索（未归一化/不确定）** | COSINE | COSINE 自带归一化，对向量长度不敏感，兼容性最好              | **稠密向量 + 图像/通用检索**                   | L2     | 图像特征向量通常未归一化，需要考虑绝对距离                   |
| **稀疏向量检索**                               | IP     | 稀疏向量本质是"关键词权重"，内积直接反映匹配程度             |

> **工程最佳实践**：很多文本嵌入模型（如 BGE 系列）输出的稠密向量已经做了 L2 归一化。此时 COSINE、IP、L2 三者的排序结果完全一致（数学原理见上方推导）。但从**计算性能**的角度，应当**优先选 IP**——因为 IP 只需乘加运算，而 COSINE 额外需要算模长和除法，L2 额外需要减法和平方运算。在 Milvus、FAISS 等向量数据库中，常见做法是：**在外部对向量做 L2 归一化，建索引时 metric_type 设为 IP**，既保证等价于 COSINE 的语义召回效果，又获得最优的检索性能。如果不确定模型是否归一化，优先选 COSINE，它对向量长度不敏感。

### 6.5 基本操作流程

理解了核心概念后，Milvus的使用可以概括为以下流程：

<img src="images/18、Milvus基本流程.png" style="zoom:67%;" />

下面沿着这个流程逐步演示代码。

#### 6.5.1 定义Schema

```python
def build_schema():
    from pymilvus import MilvusClient, DataType
    return (
        MilvusClient.create_schema(auto_id=True)
        # 主键：自动生成ID
        .add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        # 稠密向量：用于语义检索，维度与嵌入模型一致
        .add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        # 稀疏向量：用于关键词检索
        .add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        # 原始文本：存储Chunk内容，检索后返回给用户
        .add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1500)
        # 元数据：存储来源、页码等信息，支持过滤
        .add_field(field_name="metadata", datatype=DataType.JSON)
    )
```

#### 6.5.2 配置索引

```python
def build_index():
    from pymilvus import MilvusClient
    index_params = MilvusClient.prepare_index_params()

    # 稠密向量：使用HNSW索引 + L2度量
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="L2",
    )

    # 稀疏向量：使用倒排索引 + IP度量
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
    )

    return index_params
```

#### 6.5.3 创建Collection

```python
def get_milvus_client():
    from pymilvus import MilvusClient
    return MilvusClient(uri="http://localhost:19530", token="")

def create_collection(client):
    collection_name = "demo_collection"
    client.drop_collection(collection_name=collection_name)
    if not client.has_collection(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            schema=build_schema(),
            index_params=build_index(),
        )
```

#### 6.5.4 插入数据

将前面几章的成果串起来：加载文档 → 切分 → 嵌入 → 插入Milvus：

```python
def insert_data(client: MilvusClient, collection_name: str):
    from langchain_community.document_loaders import UnstructuredWordDocumentLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from FlagEmbedding import BGEM3FlagModel

    # 1、加载文件
    doc_list = UnstructuredWordDocumentLoader(
        "assets/sample.docx", mode="single"
    ).load()

# 2、切分文件
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", "。"]
    )
    splitted_doc_list = text_splitter.split_documents(doc_list)

    # 3、构建向量（同时生成稠密和稀疏向量）
    model = BGEM3FlagModel("assets/models/bge-m3")
    all_vectors = model.encode(
        [doc.page_content for doc in splitted_doc_list],
        return_dense=True, return_sparse=True
    )

# 4、准备数据：组装成List[Dict]，每个Dict对应Schema中的字段
    insert_data_list = []
    for doc, dense_vector, sparse_vector in zip(
        splitted_doc_list,
        all_vectors["dense_vecs"],
        all_vectors['lexical_weights']
    ):
        insert_data_list.append({
            "vector": dense_vector,
            "sparse_vector": sparse_vector,
            "metadata": doc.metadata,
            "text": doc.page_content
        })

# 5、插入数据
    res = client.insert(collection_name=collection_name, data=insert_data_list)
    print(res)
```

#### 6.5.5 删除数据

可以通过ID列表或过滤条件删除实体：

```python
def delete_demo(client):
    res = client.delete(
        collection_name="demo_collection",
        # 通过ID删除，也可通过其他字段过滤
        filter="id in [463480757150366907, 463480757150366908]",
    )
    print(res)  # {'delete_count': 2}
```

> 数据已经存好、索引已经建好，接下来的关键问题是：当用户提问时，如何从海量向量中找到最相关的那几个？

#### 6.6.6 完整案例

```python
import os
from pymilvus import MilvusClient, DataType
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from FlagEmbedding import BGEM3FlagModel

# ==========================================
# 1. 定义 Schema (表结构)
# ==========================================
def build_schema():
    print("-> 正在构建 Schema...")
    return (
        MilvusClient.create_schema(auto_id=True)
        # 主键：自动生成ID
        .add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        # 稠密向量：用于语义检索，维度需要与你的 BGE-M3 模型输出一致 (通常是 1024)
        .add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        # 稀疏向量：用于关键词检索
        .add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        # 原始文本：存储Chunk内容，检索后返回给用户
        .add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1500)
        # 元数据：存储来源、页码等信息，支持过滤
        .add_field(field_name="metadata", datatype=DataType.JSON)
    )

# ==========================================
# 2. 配置索引 (加速检索)
# ==========================================
def build_index():
    print("-> 正在配置索引参数...")
    index_params = MilvusClient.prepare_index_params()

    # 稠密向量：使用HNSW索引 + L2度量
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
    )

# 稀疏向量：使用倒排索引 + IP度量
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
    )

    return index_params

# ==========================================
# 3. 创建客户端与 Collection (集合/表)
# ==========================================
def get_milvus_client():
    print("-> 正在连接 Milvus 客户端...")
    # 假设你的 Milvus 运行在本地默认端口
    return MilvusClient(uri="http://localhost:19530", token="")

def create_collection(client, collection_name):
    print(f"-> 准备创建 Collection: {collection_name}")

    # 为了演示方便，如果存在同名集合，先删除
    if client.has_collection(collection_name=collection_name):
        print(f"   发现已存在集合 {collection_name}，正在删除以重新创建...")
        client.drop_collection(collection_name=collection_name)

# 创建集合，绑定前面定义的 Schema 和 索引参数
    print(f"   正在创建新集合...")
    client.create_collection(
        collection_name=collection_name,
        schema=build_schema(),
        index_params=build_index(),
    )
    print(f"   Collection {collection_name} 创建成功！")

# ==========================================
# 4. 数据处理与插入 (核心流程)
# ==========================================
def insert_data(client: MilvusClient, collection_name: str, doc_path: str, model_path: str):
    print(f"\n-> 开始处理数据并插入 Milvus...")

# 检查文件是否存在
    if not os.path.exists(doc_path):
        print(f"   [错误] 找不到文档: {doc_path}。请确保文件存在。")
        return
    if not os.path.exists(model_path):
        print(f"   [警告] 找不到本地模型: {model_path}。如果是首次运行，它可能会自动下载。")

    print(f"   1. 加载文档: {doc_path}")
    doc_list = UnstructuredWordDocumentLoader(
        doc_path, mode="single"
    ).load()

print(f"   2. 切分文档...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", "。"]
    )
    splitted_doc_list = text_splitter.split_documents(doc_list)
    print(f"      共切分为 {len(splitted_doc_list)} 个片段 (Chunks)。")

print(f"   3. 加载 BGE-M3 模型并构建向量 (这可能需要一些时间)...")
    model = BGEM3FlagModel(model_path)
    all_vectors = model.encode(
        [doc.page_content for doc in splitted_doc_list],
        return_dense=True,
        return_sparse=True
    )

print(f"   4. 准备数据列表...")
    insert_data_list = []
    for doc, dense_vector, sparse_vector in zip(
        splitted_doc_list,
        all_vectors["dense_vecs"],
        all_vectors['lexical_weights']
    ):
        insert_data_list.append({
            "vector": dense_vector,
            "sparse_vector": sparse_vector,
            "metadata": doc.metadata,
            "text": doc.page_content
        })

print(f"   5. 正在将数据插入 {collection_name}...")
    res = client.insert(collection_name=collection_name, data=insert_data_list)
    print(f"-> 插入完成！Milvus 返回结果: {res}")


# ==========================================
# 主运行入口
# ==========================================
if __name__ == "__main__":
    COLLECTION_NAME = "demo_collection"
    # 如果没有示例 word 文档，请随便建一个测试文档。
    DOC_PATH = "assets/sample.docx"
    # 如果你没有下载模型到本地，让它从 HuggingFace 自动下载
    MODEL_PATH = r"D:\ai_models\huggingface_cache\bge-m3"

try:
        # 获取客户端连接
        milvus_client = get_milvus_client()

        # # 创建数据库集合
        create_collection(milvus_client, COLLECTION_NAME)

        # # 执行数据切分、向量化并插入
        insert_data(milvus_client, COLLECTION_NAME, DOC_PATH, MODEL_PATH)

        print("\n 全部流程执行完毕！可以打开 Attu 客户端查看插入的数据了。")

    except Exception as e:
        print(f"\n 运行过程中发生错误: {e}")
```



## 七、检索

### 7.1 检索方式概览

Milvus支持多种检索方式：对向量字段支持精确检索（KNN）和近似近邻检索（ANN）；对标量字段提供条件过滤，主要配合向量检索进行结果筛选。

向量检索的关键参数：

| 参数                      | 说明                                             |
| ------------------------- | ------------------------------------------------ |
| **查询向量** (data)       | 待检索对象的向量表示，维度需与集合中向量字段一致 |
| **向量字段** (anns_field) | 指定参与检索的向量字段名称                       |
| **TopK** (limit)          | 返回相似度最高的结果数量                         |
| **metric_type**           | 相似度计算方式，如COSINE、L2或IP                 |

### 7.2 稠密向量检索

```python
def dense_vector_search_example(client, query: str, limit: int = 5):
    model = get_bge_m3_model()
    dense_vec, _ = encode_query(model, query)

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[dense_vec],
        anns_field="vector",
        limit=limit,
        search_params={"metric_type": "COSINE"},
        output_fields=["id", "text", "metadata"]
    )
    return results[0]
```

### 7.3 稀疏向量检索

```python
def sparse_vector_search_example(client, query: str, limit: int = 5):
    model = get_bge_m3_model()
    _, sparse_vec = encode_query(model, query)

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[sparse_vec],
        anns_field="sparse_vector",
        limit=limit,
        search_params={"metric_type": "IP"},
        output_fields=["id", "text", "metadata"],
    )
    return results[0]
```

### 7.4 混合检索与重排序

混合检索将稠密向量（语义匹配）和稀疏向量（关键词匹配）结合，并通过**重排序器（Reranker）** 对两路召回结果进行融合排序。

**重排序（Reranker）** 是指在初步检索完成后，对候选结果进行二次排序优化。其目标不是扩大召回范围，而是在已有候选集内提升排序质量和结果相关性。

**RRFRanker** 的工作流程：

<img src="images/19、RRF重排序.png" style="zoom:50%;" />



```python
def hybrid_vector_search_example_rrf(client, query: str, limit: int = 5):
    from pymilvus import AnnSearchRequest, RRFRanker

    model = get_bge_m3_model()
    dense_vec, sparse_vec = encode_query(model, query)

dense_req = AnnSearchRequest(
        data=[dense_vec], anns_field="vector",
        param={"metric_type": "COSINE"}, limit=limit,
    )
    sparse_req = AnnSearchRequest(
        data=[sparse_vec], anns_field="sparse_vector",
        param={"metric_type": "IP"}, limit=limit,
    )

results = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[dense_req, sparse_req],
        ranker=RRFRanker(k=60),
        limit=limit,
        output_fields=["id", "text", "metadata"],
    )
    return results[0]
```

### 7.5 标量检索

标量检索不涉及向量相似度计算，仅基于标量字段条件进行筛选：

```python
def scalar_query_examples(client, keyword: str = "大模型"):
    # 对text字段进行模糊检索
    like_res = client.query(
        collection_name=COLLECTION_NAME,
        filter=f'text like "%{keyword}%"',
        output_fields=["id", "text"],
        limit=5,
    )

# 对metadata的JSON字段进行检索
    json_res = client.query(
        collection_name=COLLECTION_NAME,
        filter='metadata["source"] like "%sample%"',
        output_fields=["id", "metadata"],
        limit=5,
    )
```

### 7.6 完整案例

```python
import os
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from FlagEmbedding import BGEM3FlagModel

# ==========================================
# 1. 全局配置与客户端初始化
# ==========================================
COLLECTION_NAME = "demo_collection"
# 确保这里和上一节插入数据时使用的模型路径一致
MODEL_PATH = r"D:\ai_models\huggingface_cache\bge-m3" 
MILVUS_URI = "http://localhost:19530"

print("-> 正在连接 Milvus 客户端...")
client = MilvusClient(uri=MILVUS_URI, token="")

print(f"-> 正在加载 BGE-M3 模型 (用于处理查询问题)...")
model = BGEM3FlagModel(MODEL_PATH)

# ==========================================
# 2. 辅助函数：将用户提问转换为向量
# ==========================================
def encode_query(query_text: str):
    """使用 BGE-M3 模型同时生成查询的稠密向量和稀疏向量"""
    vectors = model.encode(
        [query_text], 
        return_dense=True, 
        return_sparse=True
    )
    # 提取列表中的第一个（也是唯一一个）元素的向量
    dense_vec = vectors["dense_vecs"][0]
    sparse_vec = vectors["lexical_weights"][0]
    return dense_vec, sparse_vec

# ==========================================
# 3. 混合检索与重排序 (Hybrid Search + RRF)
# ==========================================
def hybrid_vector_search_example_rrf(query: str, limit: int = 3):
    print(f"\n[{' 混合检索测试 ':=^40}]")
    print(f"用户提问: {query}")
    
    # 1. 对查询语句进行向量化
    dense_vec, sparse_vec = encode_query(query)

# 2. 构建稠密向量检索请求 (语义检索)
    # 注意：这里的 metric_type 必须与你创建索引时保持一致 (你之前代码里建的是 COSINE)
    dense_req = AnnSearchRequest(
        data=[dense_vec], 
        anns_field="vector",
        param={"metric_type": "COSINE"}, 
        limit=limit,
    )
    
    # 3. 构建稀疏向量检索请求 (关键词检索)
    # 稀疏向量的度量方式通常使用 IP (内积)
    sparse_req = AnnSearchRequest(
        data=[sparse_vec], 
        anns_field="sparse_vector",
        param={"metric_type": "IP"}, 
        limit=limit,
    )

# 4. 执行混合检索，并使用 RRFRanker 重新排序
    results = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[dense_req, sparse_req],
        ranker=RRFRanker(k=60), # k 决定了排名权重的衰减速度，一般设为 60
        limit=limit,
        output_fields=["id", "text", "metadata"], # 告诉 Milvus 返回哪些原始字段
    )
    
    # 5. 打印结果
    print(f"\n--- 混合检索找到了 {len(results[0])} 条相关片段 ---")
    for i, hit in enumerate(results[0]):
        print(f"\nTop {i+1} (匹配分数/RRF Score: {hit['distance']:.4f})")

print(f"ID: {hit['id']}")
        print(f"来源文件: {hit['entity'].get('metadata', {}).get('source', '未知')}")
        print(f"文本内容: {hit['entity'].get('text', '')}")

# ==========================================
# 4. 标量检索 (精准过滤)
# ==========================================
def scalar_query_examples(keyword: str):
    print(f"\n[{' 标量检索测试 ':=^40}]")
    print(f"正在全文检索包含关键词 '{keyword}' 的片段...")
    
    # 对 text 字段进行模糊匹配 (类似 SQL 的 LIKE)
    like_res = client.query(
        collection_name=COLLECTION_NAME,
        filter=f'text like "%{keyword}%"',
        output_fields=["id", "text", "metadata"],
        limit=3,
    )
    
    print(f"\n--- 标量检索找到了 {len(like_res)} 条结果 ---")
    for i, res in enumerate(like_res):
        print(f"\n结果 {i+1} (ID: {res['id']})")
        print(f"文本内容: {res['text']}")

# ==========================================
# 主运行入口
# ==========================================
if __name__ == "__main__":
    try:
        # 你可以根据你导入的 sample.docx 的实际内容，修改下面的提问词
        test_query = "什么是大语言模型？" 
        
        # 执行混合检索测试
        hybrid_vector_search_example_rrf(query=test_query, limit=3)
        
        # 执行标量检索测试
        test_keyword = "模型"
        scalar_query_examples(keyword=test_keyword)
        
    except Exception as e:
        print(f"\n 检索过程中发生错误: {e}")
```

> 检索到了相关文档片段，最后一步就是把它们交给大模型，生成最终答案。

------

## 八、生成

### 8.1 生成原理

查找到相关数据之后，就可以将所有的数据放到和LLM交互的上下文当中，让LLM基于完整的上下文信息来进行生成。

<img src="images/20、RAG流程.png" style="zoom:50%;" />

### 8.2 基础生成

```python
def rag_demo(client: MilvusClient, query):
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini")

    # 检索相关数据
    retrieval_res = hybrid_vector_search_example_rrf(client=client, query=query)

# 构建上下文
    context = "\n".join([data["entity"]["text"] for data in retrieval_res])
    message_list = [
        {
            "role": "system",
            "content": "你是一个专业的法律问答机器人，请根据上下文回答问题，"
                       "当上下文无法回答问题时，请回答'根据上下文无法回答该问题'"
        },
        {
            "role": "user",
            "content": f"根据以下上下文回答问题：{context}\n问题：{query}"
        }
    ]

    res = llm.invoke(message_list)
    print(res.content)
```

### 8.3 LCEL构建RAG链

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的AI助手。请根据以下上下文回答问题。\n\n上下文：{context}"),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

result = rag_chain.invoke("什么是RAG？")
```

### 8.4 来源引用

```python
from langchain_core.runnables import RunnableParallel

def format_docs_with_sources(docs):
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "未知来源")
        formatted.append(f"[来源{i+1}: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

rag_chain_with_sources = (
    RunnableParallel(
        context=retriever | format_docs_with_sources,
        question=RunnablePassthrough()
    )
    | rag_prompt | llm | StrOutputParser()
)
```



> 到这里，我们已经逐步走完了RAG的六个环节。接下来把所有环节串在一起，跑通一个完整的端到端案例。

------

## 九、完整实战

### 9.1 端到端RAG

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 加载文档
documents = TextLoader("./documents/sample.txt").load()

# 2. 切分文档
splits = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50,
    separators=["\n\n", "\n", "。", "!", "?", "；", "，", " ", ""]
).split_documents(documents)

# 3. 创建向量存储
vectorstore = Chroma.from_documents(
    documents=splits, embedding=OpenAIEmbeddings(), collection_name="rag_demo"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. 创建RAG链
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的AI助手。请根据以下上下文回答问题。\n上下文：\n{context}"),
    ("human", "{question}")
])
llm = ChatOpenAI(model="gpt-4o-mini")

def format_docs(docs):
    return "\n\n---\n\n".join([
        f"[来源: {doc.metadata.get('source', '未知')}]\n{doc.page_content}"
        for doc in docs
    ])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt | llm | StrOutputParser()
)

# 5. 查询
response = rag_chain.invoke("什么是LangChain？")
print(response)
```

> 跑通了基础版RAG之后，你会发现实际效果不一定令人满意——未来还需要系统地解决这些问题。

------

## 十、RAG调优

### 10.1 高级技术一览

| 技术           | 说明                     | 效果               |
| -------------- | ------------------------ | ------------------ |
| **混合检索**   | 结合向量检索和关键词检索 | 提高召回率         |
| **重排序**     | 对检索结果重新排序       | 提高精准度         |
| **查询重写**   | 优化用户查询             | 提高检索质量       |
| **元数据过滤** | 根据元数据筛选           | 精确控制检索范围   |
| **父子索引**   | 父文档存储，子文档检索   | 返回更完整的上下文 |
| **多路召回**   | 使用多种策略并行检索     | 综合提升效果       |

### 10.2 性能优化

| 优化点             | 方法           | 效果             |
| ------------------ | -------------- | ---------------- |
| **减小chunk_size** | 500-800字符    | 提高检索精度     |
| **调整检索数量k**  | 3-10个结果     | 平衡准确性和效率 |
| **使用混合检索**   | 向量+BM25      | 提高召回率       |
| **添加重排序**     | 交叉编码器/RRF | 提高精准度       |
| **上下文压缩**     | 去除无关内容   | 减少Token消耗    |

### 10.3 质量提升

```python
# 1. 查询重写
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个查询优化专家。请将用户查询重写为更适合检索的形式。"),
    ("human", "{query}")
])
rewrite_chain = rewrite_prompt | llm | StrOutputParser()

# 2. 多轮查询扩展
def expand_query(query):
    expanded = llm.invoke(f"生成3个与'{query}'相关的搜索查询，用逗号分隔")
    return [query] + [q.strip() for q in expanded.content.split(",")]

# 3. 检索后验证
def verify_relevance(query, doc):
    score = llm.invoke(f"查询: {query}\n文档: {doc.page_content}\n相关吗(0-1)?")
    return float(score.content) > 0.7
```

### 10.4 常见问题

| 问题             | 原因                 | 解决方案                    |
| ---------------- | -------------------- | --------------------------- |
| 检索不到相关内容 | chunk_size太大或太小 | 调整chunk_size，增加overlap |
| 返回重复内容     | 相似度检索过于集中   | 使用MMR增加多样性           |
| Token消耗过大    | 检索结果过多         | 减小k值，使用压缩           |
| 响应速度慢       | 检索效率低           | 使用更快的向量库，缓存结果  |

------

## 十一、总结与展望

### 11.1 技术栈回顾

<img src="images/22、RAG完整技术栈.png" style="zoom:67%;" />

### 11.2 概念速查

| 组件         | 核心要点                   | 关键类/方法                               |
| ------------ | -------------------------- | ----------------------------------------- |
| **文档加载** | 多种加载器，支持多种数据源 | `UnstructuredMarkdownLoader`, `MinerU`    |
| **文档切分** | 保持语义完整，适当重叠     | `RecursiveCharacterTextSplitter`          |
| **嵌入模型** | 文本转向量，支持稠密+稀疏  | `HuggingFaceEmbeddings`, `BGEM3FlagModel` |
| **向量存储** | 高效相似度搜索             | `Milvus`, `Chroma`, `FAISS`               | **检索**     | 稠密/稀疏/混合/标量检索    | `client.search`, `client.hybrid_search`   |
| **RAG链**    | 使用LCEL组合               | `retriever | format_docs | llm`           |

### 11.3 技术选型

| 场景         | 推荐方案                  | 原因                 |
| ------------ | ------------------------- | -------------------- |
| **快速原型** | Chroma + OpenAIEmbeddings | 轻量级，易上手       |
| **生产环境** | Milvus + BGE-M3           | 性能好，支持混合检索 |
| **中文场景** | BGE系列嵌入模型           | 中文优化效果好       |
| **多语言**   | BGE-M3 或 OpenAI          | 支持多语言           |
| **本地部署** | FAISS + HuggingFace       | 无需外部服务         |
| **高精度**   | 混合检索 + RRF重排序      | 召回和精准都优化     |

### 11.4 学习路线

```
入门阶段：
  文档加载 → 文档切分 → 向量嵌入 → Chroma存储 → 基础RAG链
       ↓
进阶阶段：
  混合检索 → 元数据过滤 → 父子索引 → 上下文压缩
       ↓
高级阶段：
  重排序 → 查询重写 → 多路召回 → RAG评估优化
       ↓
实战阶段：
  构建完整RAG应用 → 性能优化 → 生产部署 → 监控维护
```

### 11.5 下一步

下一节我们将学习**Agents智能代理**：

| 章节          | 内容            | 核心概念                          |
| ------------- | --------------- | --------------------------------- |
| **Agent基础** | Agent工作原理   | ReAct循环、工具调用               |
| **工具定义**  | 自定义工具      | @tool装饰器、StructuredTool       |
| **Agent类型** | 不同类型的Agent | ReAct、Self-Ask、OpenAI Functions |
| **实战应用**  | 构建实际Agent   | 聊天机器人、任务助手              |

------

## 参考资料

- [LangChain Retrieval 文档](https://docs.langchain.com/oss/python/langchain/retrievers)
- [LangChain Vector Stores](https://docs.langchain.com/oss/python/langchain/vectorstores)
- [LangChain Text Splitters](https://docs.langchain.com/oss/python/langchain/text_splitters)
- [Milvus 文档](https://milvus.io/)
- [MinerU 项目仓库](https://github.com/opendatalab/MinerU)
- [FAISS 官方文档](https://faiss.ai/)
- [Chroma 向量数据库](https://docs.trychroma.com/)
- [Lost in the Middle 论文](https://arxiv.org/pdf/2307.03172)