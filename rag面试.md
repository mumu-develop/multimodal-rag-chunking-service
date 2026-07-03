# 多模态 RAG Chunking Service — 模拟面试 Q&A

> 基于 FastAPI + Celery + MySQL + Milvus + DashScope Embedding 的生产级 RAG 文档解析与检索服务。
> 每个问题后附"踩坑经验"和"最终方案代码"，方便快速学习。

---

## 零、项目架构总览

### Q24. 请用一分钟介绍你的项目，包括整体架构和提供了哪些 API？

#### 面试官追问
- 整个处理流水线是怎么运转的？
- 对外提供了哪些接口？
- 各个组件之间是怎么协作的？

#### 面试者回答

**一句话概括**：这是一个生产级 RAG 文档解析与检索服务，用户上传文档后，系统自动完成解析→分块→向量嵌入→存储，后续支持文本/图片/图文联合搜索。

**整体架构**：

```
用户 → FastAPI (Web层)
        ├── POST /upload → Redis/Celery (异步任务队列)
        │                    └── Worker: parse → chunk → VLM caption → embed → Milvus + MySQL
        ├── POST /search → HybridSearcher (dense + BM25 → RRF → Reranker)
        ├── POST /evaluate → 离线评估 / RAGAS / 消融实验 / 线上反馈
        └── GET /chunks, /tasks/{id} → MySQL (管理接口)
```

**核心技术栈**：

| 层次 | 技术 | 关键文件 |
|:---|:---|:---|
| **Web 层** | FastAPI + CORS + Auth/Trace 中间件 | `main.py`, `api/routers/*` |
| **异步任务层** | Celery + Redis（broker + backend） | `tasks.py` |
| **解析层** | 多解析器策略模式（PDF/DOCX/PPTX/MD/Image），延迟实例化 + entry points 插件 | `parsers/*`, `chunkers/__init__.py` |
| **分块层** | 自研 HierarchicalChunker / TableChunker / SlideChunker | `chunkers/hierarchical_chunker.py` 等 |
| **嵌入层** | DashScope qwen3-vl-embedding API，统一 2560 维向量，10 路并发 | `embedders/unified_embedder.py` |
| **检索层** | Milvus dense + BM25 混合检索（RRF 融合）+ bge-reranker-v2-m3 重排 | `search/hybrid_search.py`, `embedders/reranker.py` |
| **存储层** | MySQL（元数据）+ Milvus（向量）双存储，确定性 chunk_id 保证幂等 | `milvus_client.py`, `repositories/*` |
| **评估层** | 离线评估 + RAGAS LLM-as-Judge + 消融实验 + 线上反馈 | `evaluation/*` |

**处理流水线**（6 阶段，Celery 异步）：
1. 解析文档（10%）→ 2. 语义分块（20%）→ 3. VLM caption 图片描述（30%）
4. 向量嵌入（50%，10 路并发）→ 5. Milvus 批量存储（70%，独立重试 5 次）
6. MySQL 批量存储（85%，独立重试 5 次）

**重试优化**（两个 P0）：
- Redis 中间结果缓存：`chunks + entities` 序列化到 Redis（TTL 1h），重试时跳过 parse/chunk/embed
- 存储层独立重试：Milvus 和 MySQL 各自指数退避重试，互不阻塞

**对外 API 接口**（13 个端点）：

| # | 方法 | 路径 | 功能 | 认证 |
|:---:|:---:|------|------|:---:|
| 1 | GET | `/health` | 健康检查（MySQL/Redis/Milvus 并发探测） | 无 |
| 2 | POST | `/api/v1/upload` | 上传文件（pdf/docx/pptx/md/png/jpg），返回 task_id + doc_id | 需 |
| 3 | POST | `/api/v1/upload/folder` | 上传 .md 文件夹（OCR 输出），语义排序 + 逐页解析 | 需 |
| 4 | GET | `/api/v1/tasks/{task_id}` | 查询任务状态 + 进度（pending/processing/completed/failed） | 需 |
| 5 | GET | `/api/v1/?doc_id=xxx` | 分页列出 chunk（按 doc_id + chunk_type 过滤，page/size） | 需 |
| 6 | GET | `/api/v1/docs/{doc_id}/tree` | 获取文档层级结构（标题树，用于前端导航） | 需 |
| 7 | POST | `/api/v1/search` | 向量搜索（支持 text / image / multimodal 三种 modal） | 需 |
| 8 | POST | `/api/v1/search/image` | 上传图片文件进行搜索 | 需 |
| 9 | POST | `/api/v1/evaluate/run` | 离线评估（Recall@K / Precision@K / MRR / NDCG@K，按 tag 分组） | 需 |
| 10 | POST | `/api/v1/evaluate/ablation/run` | 消融实验（full / no_rerank / no_hybrid 对比） | 需 |
| 11 | POST | `/api/v1/evaluate/feedback` | 提交用户搜索反馈（点击行为 + 停留时间） | 需 |
| 12 | GET | `/api/v1/evaluate/feedback/stats` | 获取线上反馈统计（CTR / 平均停留 / 低满意度 query 告警） | 需 |
| 13 | POST | `/api/v1/evaluate/ragas/run` | RAGAS LLM-as-Judge 评估（Context Precision / Recall） | 需 |

**搜索流程详解**：
1. 编码查询向量（text→encode_query / image→encode_image_query / both→encode_multimodal）
2. Milvus 粗排：dense 向量 + BM25 文本各取 `top_k * 2` 候选，RRF 融合
3. Reranker 精排：取 top 50 候选，bge-reranker-v2-m3 计算交叉编码分数，降序
4. 截断到 `top_k` 返回（直接从 Milvus 返回完整字段，不回查 MySQL）

#### 面试官评分
| 维度 | 得分 | 说明 |
|------|:---:|------|
| 全局视野 | 10/10 | 从架构到接口全覆盖，流程清晰 |
| 表达清晰度 | 9/10 | 一分钟能讲完核心要点 |
| **综合** | **9.5/10** | 优秀 |

---

## 一、数据一致性与分布式事务

### Q1. MySQL 和 Milvus 之间没有分布式事务，如何保证最终一致性？

#### 面试官追问
- 当前写入顺序是先 Milvus 后 MySQL，为什么这样设计？
- 如果 MySQL 写入失败触发 Celery 重试，会不会产生重复数据？
- 之前踩过什么坑？

#### 面试者回答

**踩过的坑**：最初设计是先写 MySQL 再写 Milvus。当时遇到的问题是：MySQL 写入成功但 Milvus 失败，Celery 触发 `self.retry()` 后整条流水线重跑。由于 chunk_id 是 `uuid.uuid4()` 随机生成的，重试时产生了全新的 chunk_id，MySQL 的 `INSERT IGNORE` 无法识别，于是产生了**孤儿数据**——同一个 doc_id 在 MySQL 中出现多批 chunk，只有最后一批有对应的 Milvus 向量。

**为什么先 Milvus 后 MySQL**：Milvus upsert 是幂等的（相同 chunk_id 覆盖），MySQL `INSERT IGNORE` 也是幂等的（相同 chunk_id 跳过）。只要 chunk_id 是确定性的，无论写入顺序如何，重试都不会产生重复数据。但先写 Milvus 更安全——如果 Milvus 就失败了，MySQL 不会被触碰，不会产生任何"半残"数据。

**最终方案（已实现）**：两步改造

```python
# ========== 第一步：确定性 chunk_id（app/chunkers/base.py）==========
import hashlib

def generate_chunk_id(doc_id, page_num, chunk_type, content, sort_order=0):
    """基于内容派生 chunk_id，保证同一次解析即使重试也产生相同 ID。"""
    raw = f"{doc_id}:{page_num}:{chunk_type}:{sort_order}:{content}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]

# ========== 第二步：颠倒写入顺序（app/tasks.py）==========
# 阶段 5：先写 Milvus
milvus_ok = milvus_client.batch_upsert(entities)
if not milvus_ok:
    raise Exception("Milvus storage failed, MySQL not touched")

# 阶段 6：后写 MySQL
mysql_ok = insert_doc_chunks(chunks)
```

#### 面试官评分
| 维度 | 得分 | 说明 |
|------|:---:|------|
| 问题定位 | 10/10 | 精准识别 UUID 随机性 + INSERT IGNORE 不匹配的根因 |
| 方案可行性 | 10/10 | SHA-256 确定性 ID 是最小改动最彻底的方案 |
| 表达清晰度 | 9/10 | 结构清晰 |
| **综合** | **9.7/10** | 优秀 |

---

### Q2. Celery 重试的退避策略是固定 60 秒，合理吗？

#### 面试官追问
- 固定延迟 vs 指数退避的区别是什么？
- 当前实现是怎么改的？

#### 面试者回答

**踩过的坑**：最初 `self.retry(exc=exc, countdown=60)`，无论第几次重试都是等 60 秒。如果问题是临时性故障（如网络抖动），第一次重试就可能成功；如果是持久性故障（如数据库宕机），三次重试都是 60 秒间隔，无法缓解服务端压力。

**最终方案（已实现）**：指数退避

```python
# app/tasks.py - except 块
self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))
```

| 重试次数 | 延迟 |
|:---:|:---:|
| 第 1 次 | 60s |
| 第 2 次 | 120s |
| 第 3 次 | 240s |

#### 面试官评分
| 维度 | 得分 | 说明 |
|------|:---:|------|
| 问题定位 | 9/10 | 指出了固定延迟和指数退避的本质区别 |
| 方案可行性 | 10/10 | 一行代码解决，改动极小 |
| **综合** | **9.5/10** | 优秀 |

---

### Q3. Milvus 成功后 MySQL 失败，重试时 Milvus 已有数据怎么办？

#### 面试官追问
- 重试会重新执行 embedding 和 VLM caption，这些非确定性操作会导致问题吗？
- 当前代码是否覆盖了这个问题？

#### 面试者回答

**踩过的坑**：这个场景是之前面试 Round 4 提到的。由于 chunk_id 是确定性的，Milvus upsert 会覆盖相同 chunk_id 的数据，不会产生重复。但重试时 VLM caption（LLM 输出）和 embedding（浮点运算）可能产生不同结果，导致向量值变化。

**分析**：对于文本 chunk，content 来自原始文档内容，是完全确定的——chunk_id 不变，embedding 也不变（同一模型同一输入 = 同一输出）。问题主要在图片 chunk，但图片 chunk 的 chunk_id 用的是 `content=""`（caption 还没生成时的空字符串），所以 chunk_id 不受 LLM 影响。

**当前状态**：已实现。确定性 chunk_id 已经覆盖了所有场景，图片 chunk 的 ID 基于空 content 生成，与 LLM 输出无关。

#### 面试官评分
| 维度 | 得分 | 说明 |
|------|:---:|------|
| 分析深度 | 9/10 | 能正确分析图片 chunk 的 chunk_id 生成时机 |
| **综合** | **9.0/10** | 优秀 |

---

## 二、搜索性能优化

### Q4. 搜索接口为什么去掉了 MySQL 回查？

#### 面试官追问
- 原来的 N+1 查询问题具体是什么？
- 去掉 MySQL 回查后，搜索还能拿到哪些字段？
- 图片 caption 怎么办？Milvus 没有 caption 字段

#### 面试者回答

**踩过的坑**：原来 `_enrich_with_mysql` 方法对每个不同 doc_id 执行一次 `get_chunks_by_doc(did, page=1, size=9999)`。假设搜索返回 50 个结果来自 20 个文档，每个文档平均 1000 个 chunk，意味着 MySQL 返回 20000 行数据（40MB），但实际只用到 50 行。**99.75% 的查询结果是浪费的**。

**最终方案（已实现）**：Milvus schema 已包含 `content`、`doc_id`、`chunk_type`、`page_num`、`metadata`，搜索直接从 Milvus 返回。caption 存入 metadata JSON 字段：

```python
# app/tasks.py — 写入 Milvus 时将 caption 存入 metadata
caption = c.get("caption")
metadata = c.get("metadata", {}).copy()
if caption:
    metadata["caption"] = caption

entity = {
    "chunk_id": cid,
    "doc_id": c["doc_id"],
    "chunk_type": c["chunk_type"],
    "content": c["content"],       # Milvus 已有 content
    "metadata": metadata,          # 含 caption
    "vector": dense_vec,
}

# app/services/search_service.py — 读取时从 metadata 取 caption
chunk_obj = {
    "caption": hit.get("metadata", {}).get("caption") if hit.get("metadata") else None,
}
```

```python
# app/search/hybrid_search.py — 删除了 _enrich_with_mysql 方法
# 现在 Milvus 返回什么就直接用什么，不再回查 MySQL
results = milvus_client.search(vector=..., top_k=..., expr=...)
if use_rerank and settings.RERANK_ENABLED and results:
    reranked = reranker.rerank(query, results[:settings.RERANK_TOP_N])
    return reranked[:top_k]
return results[:top_k]
```

**字段对比**：

| 字段 | 旧方案（MySQL 回查） | 新方案（Milvus 直返） |
|------|:---:|:---:|
| content | MySQL | Milvus |
| doc_id | Milvus+MySQL | Milvus |
| chunk_type | Milvus+MySQL | Milvus |
| page_num | MySQL | Milvus |
| caption | MySQL | Milvus metadata |
| audit_status | MySQL | ❌ 不需要 |
| creator | MySQL | ❌ 不需要 |

搜索场景不需要 `audit_status`、`creator` 等管理字段。管理接口 `GET /api/v1/chunks` 继续走 MySQL。

#### 面试官评分
| 维度 | 得分 | 说明 |
|------|:---:|------|
| 问题定位 | 10/10 | 量化了 99.75% 的浪费 |
| 方案可行性 | 10/10 | 职责分离：搜索走 Milvus，管理走 MySQL |
| **综合** | **10/10** | 完美 |

---

## 三、解析器架构

### Q5. 解析器注册表为什么要从实例改为延迟实例化？

#### 面试官追问
- 原来的写法有什么问题？
- 插件注册机制怎么设计的？

#### 面试者回答

**踩过的坑**：原来 `app/chunkers/__init__.py` 中：

```python
PARSER_REGISTRY = {
    "pdf": PDFParser(),    # 模块加载时就实例化
    "docx": DOCXParser(),
    "pptx": PPTXParser(),
    ...
}
```

问题：
1. 模块顶层直接实例化，服务启动时加载所有解析器的依赖库
2. 如果 90% 请求都是 PDF，DOCXParser、PPTXParser 占用的内存是浪费的
3. 如果某个解析器依赖库版本不兼容，**整个服务启动失败**（即使不使用该格式）

**最终方案（已实现）**：

```python
# app/chunkers/__init__.py
_PARSER_CLASSES: dict[str, Type[DocumentParser]] = {}
_parser_instances: dict[str, DocumentParser] = {}

def register_parser(ext: str, parser_class: Type[DocumentParser]) -> None:
    """运行时注册自定义解析器。"""
    if not issubclass(parser_class, DocumentParser):
        raise TypeError(f"...")
    _PARSER_CLASSES[ext] = parser_class
    _parser_instances.pop(ext, None)  # 清除已有实例

def _get_parser(ext: str) -> DocumentParser:
    """延迟实例化单例。"""
    if ext in _parser_instances:
        return _parser_instances[ext]
    parser_cls = _PARSER_CLASSES.get(ext)
    if parser_cls is None:
        from app.parsers.image_parser import ImageParser
        parser_cls = ImageParser
    instance = parser_cls()
    _parser_instances[ext] = instance
    return instance

def _init_builtin_parsers() -> None:
    """延迟导入，只在首次 parse 时加载。"""
    if _PARSER_CLASSES:
        return
    from app.parsers.pdf_parser import PDFParser
    from app.parsers.docx_parser import DOCXParser
    from app.parsers.pptx_parser import PPTXParser
    from app.parsers.image_parser import ImageParser
    from app.parsers.markdown_parser import MarkdownParser
    _PARSER_CLASSES.update({...})

def parse_document(file_path, doc_id, progress_fn=None):
    ext = Path(file_path).suffix.lower().lstrip(".")
    _init_builtin_parsers()  # 首次调用时才加载
    parser = _get_parser(ext)  # 延迟实例化
    return parser.parse(file_path, doc_id, progress_fn)
```

**插件机制**（`app/main.py`）：

```python
def _load_parser_plugins() -> None:
    from app.chunkers import register_parser
    import importlib.metadata as metadata
    for ep in metadata.entry_points(group="rag.parsers"):
        parser_cls = ep.load()
        register_parser(ep.name, parser_cls)
```

第三方只需在 `setup.py` 声明 entry point：
```python
entry_points={"rag.parsers": ["custom_table = my_plugin:TableParser"]}
```

#### 面试官评分
| 维度 | 得分 | 说明 |
|------|:---:|------|
| 问题定位 | 10/10 | 三个问题全部命中：启动慢、资源浪费、级联故障 |
| 方案可行性 | 10/10 | 类引用 + 延迟实例化 + entry points 三管齐下 |
| **综合** | **10/10** | 完美 |

---

### Q5-1. unstructured 库解析 DOCX 时 heading_level 全是 1，为什么？怎么解决？

#### 面试官追问
- 为什么用 unstructured 而不是直接用 python-docx？
- unstructured 返回的元素分类有什么问题？
- 生产环境应该怎么改？

#### 面试者回答

**踩过的坑**：用户上传阿里巴巴 Java 开发手册（DOCX），前端显示的树形结构所有标题都显示为 level=1，没有层级区分。

**问题定位**：追踪到 `docx_parser.py:118-124`：

```python
if el_type == "Title":
    headings.append({"level": 1, "text": text})  # 固定 level=1 ❌
elif el_type in ("Header", "SectionHeader"):
    level = getattr(el.metadata, "level", 2) or 2  # 默认值永远返回 2 ❌
```

**根本原因**：unstructured 库的元素分类逻辑存在三个问题：

| 问题 | 说明 |
|:---|:---|
| **样式映射丢失层级** | unstructured 把 `Heading 1`~`Heading 9` **全部映射到 `Title` 类型**，层级信息被丢弃 |
| **不支持中文样式** | 只匹配英文名 `"Heading 1"`，不支持 Word 中文版的 `"标题 1"` |
| **metadata 没有 level** | 返回的 `Title` 元素没有携带真实的 heading level |

**unstructured 源码证据**（`_style_based_element_type` 方法）：
```python
STYLE_TO_ELEMENT_MAPPING = {
    "Heading 1": Title,  # ← 所有 Heading 都映射到 Title，丢失层级！
    "Heading 2": Title,
    "Heading 3": Title,
    ...
    "Heading 9": Title,
    "Title": Title,
    "Subtitle": Title,
}
# paragraph.style.name 只做精确匹配，不支持中文样式名
style_name = (paragraph.style and paragraph.style.name) or "Normal"
return STYLE_TO_ELEMENT_MAPPING.get(style_name)  # 返回 Title，没有 level
```

**结论**：这是 unstructured 的设计选择——它把所有标题类样式统一归类为 `Title` 元素类型，用于文档结构分析，但**不保留层级细节**。对于需要层级信息的场景（如树形导航），必须自己用 python-docx 读取真实样式。

实际解析输出验证：
```
类型: Title | 文本: 一、 编程规约         category_depth: 0  ← 应该是 Heading 1
类型: Title | 文本: (一) 命名风格         category_depth: 1  ← 应该是 Heading 2
类型: Title | 文本: 1. 【强制】...        category_depth: 0  ← 正文，不是标题
```

unstructured 的 `category_depth` 是嵌套深度，不是真正的 Word Heading 样式层级。

**正确做法**：python-docx 直接读取段落样式：

```python
from docx import Document

doc = Document(file_path)
for para in doc.paragraphs:
    style_name = para.style.name  # "Heading 1", "Heading 2", "Title", "Normal"...
    if style_name.startswith("Heading"):
        level = int(style_name.split()[-1])  # "Heading 3" → level=3 ✅
```

DOCX 文件的段落样式是文档作者定义的层级，是最准确的信息来源。

**改进方案对比**：

| 方案 | 改动量 | 准确性 | 说明 |
|:---|:---|:---|:---|
| **A: 换用 python-docx 读取样式** | 中 | ★★★★★ | 最准确，直接读取 Word 样式 |
| **B: 用 category_depth 推断层级** | 小 | ★★★☆☆ | 不完全准确，无法区分 Heading 1 和正文 |
| **C: 标题文本模式识别** | 小 | ★★★★☆ | 解析"一、(一)、1.1"等编号模式 |

**推荐方案 A**：在解析 DOCX 时，用 python-docx 读取段落的真实 Heading 样式，而不是依赖 unstructured 的猜测分类。

#### 面试官评分
| 维度 | 得分 | 说明 |
|------|:---:|------|
| 问题定位 | 9/10 | 发现了 unstructured 元素分类的根本问题 |
| 方案可行性 | 9/10 | python-docx 方案简单准确 |
| **综合** | **9/10** | 优秀 |

---

## 四、MD 文件夹处理

### Q6. OCR 输出的多个 .md 文件如何按页码顺序拼接？排序策略是什么？

#### 面试官追问
- 自然排序怎么处理 `page_1, page_10, page_2` 的情况？
- 如果文件名是 `cover.md, toc.md, chapter1.md`，怎么保证正确顺序？
- 之前合并到单个 .md 文件有什么问题？

#### 面试者回答

**踩过的坑**：最初方案是将多个 .md 合并为一个临时 .md 文件，插入 `<!-- page_break: N -->` 注释。但发现：
1. `MarkdownParser._remove_html_comments` 会把这些注释**直接删除**（HTML 注释正则匹配）
2. `page_num` 信息完全丢失（合并后始终返回 `page_num=1`）
3. 简单自然排序（按文件名数字）无法处理 `cover.md, toc.md` 等语义文件

**第二轮方案的缺陷**：第一版修复时选择绕开 `MarkdownParser`，在 `folder_service.py` 中自己手写了解析逻辑（标题检测、表格缓冲、代码块缓冲、图片提取）。但这带来了新问题：
1. 与 `MarkdownParser._parse_markdown` **逻辑重复**（约 80 行代码重复）
2. 缺少远程图片下载、YAML front matter 清理、引用块处理等功能
3. 两处解析逻辑需要同步维护，后续改动容易遗漏一处

**最终方案（已实现）**：复用 `MarkdownParser`，零侵入。`folder_service` 只负责排序 + 设置 `page_num`：

```python
# app/services/folder_service.py

# 语义优先级映射（排序逻辑保留不变）
_MD_PRIORITY = {
    "cover": 0, "封面": 0, "title": 1, "toc": 2, "目录": 2,
    "chapter": 3, "content": 3, "正文": 3,
    "appendix": 4, "附录": 4, "reference": 4, "参考": 4,
}

def _md_sort_key(filename: str):
    """语义关键词 + 自然数组合排序。"""
    base = filename.lower().replace(".md", "")
    for keyword, priority in _MD_PRIORITY.items():
        if keyword in base:
            num_parts = [int(x) if x.isdigit() else x.lower()
                        for x in re.split(r"(\d+)", base)]
            return (priority, num_parts)
    num_parts = [int(x) if x.isdigit() else x.lower()
                for x in re.split(r"(\d+)", base)]
    return (99, num_parts)

def parse_md_folder(folder_path: str, doc_id: str) -> list[ParsedPage]:
    """排序后复用 MarkdownParser 逐文件解析，page_num 按排序递增。"""
    md_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".md")]
    md_files.sort(key=_md_sort_key)

    parser = MarkdownParser()
    pages: list[ParsedPage] = []
    for page_num, md_file in enumerate(md_files, start=1):
        file_path = os.path.join(folder_path, md_file)
        parsed = parser.parse(file_path, doc_id)  # 完整解析逻辑
        for p in parsed:
            p.page_num = page_num  # 覆盖默认的 page_num=1
            pages.append(p)
    return pages
```

```python
# app/tasks.py — 复用已有专用任务，跳过 parse 阶段
@celery_app.task(bind=True, name="tasks.parse_and_chunk_pages")
def parse_and_chunk_pages_task(self, pages_data, doc_id, creator):
    pages = [ParsedPage(**p) for p in pages_data]
    chunks = chunk_pages(pages, doc_id, creator)  # 直接分块
    # ... 后续：VLM caption → embedding → MySQL → Milvus
```

**排序结果验证**：
```
输入: ["cover.md", "toc.md", "chapter1.md", "chapter10.md", "chapter2.md", "page_3.md", "page_10.md", "page_1.md"]
输出: ["cover.md", "toc.md", "chapter1.md", "chapter2.md", "chapter10.md", "page_1.md", "page_3.md", "page_10.md"]
```

#### 面试官评分
| 维度 | 得分 | 说明 |
|------|:---:|------|
| 问题定位 | 10/10 | 三轮迭代：合并方案→绕开Parser→复用Parser，精准识别逻辑重复的根本缺陷 |
| 方案可行性 | 10/10 | 零侵入复用现有 MarkdownParser，排序与解析职责分离，无代码重复 |
| 表达清晰度 | 9/10 | 结构清晰，踩坑→迭代→最终方案的叙事逻辑完整 |
| **综合** | **9.7/10** | 优秀 |

---

## 五、文本清洗与噪声过滤

### Q6-1. OCR 输出的文本包含大量噪音（页脚、页码、水印、控制字符），你是怎么清洗的？

#### 面试官追问
- PDF 解析后每页底部的 "Page 1 of 20" 怎么处理？
- 带 "Confidential" 水印的企业文档怎么处理？
- OCR 产生的不可见控制字符会导致 embedding 异常吗？

#### 面试者回答

**踩过的坑**：最初解析 PDF 时，每页底部都有 "Page 1 / 20"、"CONFIDENTIAL" 水印、以及一些控制字符（`\x0c` 换页符等）。这些噪音文本：
1. 被当作正常内容生成 embedding，导致搜索时匹配到无关页面
2. "Confidential" 等词在多个文档中重复出现，造成向量空间聚类异常
3. 控制字符在某些 MySQL 配置下会触发 `Incorrect string value` 错误

**最终方案（已实现）**：三层清洗管线

```python
# app/parsers/noise_filter.py

# ========== 第一层：字符级清洗 ==========
def clean_text(text: str) -> str:
    """去除控制字符和多余空白。"""
    text = re.sub(r"[\t ]+", " ", text)        # 多余空格/制表符 → 单空格
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)  # 控制字符
    return text.strip()

# ========== 第二层：噪音文本检测 ==========
_FOOTER_KEYWORDS = {
    "confidential", "draft", "internal", "proprietary",
    "copyright", "reserved", "all rights reserved",
    "page", "page of",
}

_WATERMARK_KEYWORDS = {
    "confidential", "draft", "do not distribute", "internal use only",
    "内部资料", "机密", "绝密",
}

def is_noise_text(text: str, min_length: int = 3) -> bool:
    """判断是否应过滤。"""
    s = text.strip()
    if not s or len(s) < min_length:
        return True                       # 空或过短
    if re.match(r"^\d+$", s) and len(s) <= 5:
        return True                       # 纯页码数字
    if re.match(r"^\d+\s*[/–-]\s*\d+$", s):
        return True                       # "1 / 20" 格式
    if re.match(r"^[^\w\u4e00-\u9fff]+$", s):
        return True                       # 纯符号行
    for keyword in _FOOTER_KEYWORDS | _WATERMARK_KEYWORDS:
        if s.lower() == keyword or s.lower().startswith(keyword):
            return True
    return False

# ========== 第三层：MarkdownParser 中应用 ==========
# 解析 Markdown 时逐行检查，跳过噪音行
for line in content.split("\n"):
    cleaned = clean_text(line)
    if not cleaned or is_noise_text(cleaned):
        continue  # 跳过噪音行
    # 正常处理
```

**过滤效果**：

| 噪音类型 | 过滤方式 | 示例 |
|:---:|------|------|
| 页码数字 | 正则匹配 | `"1"`, `"20"` |
| 页码格式 | 正则匹配 | `"1 / 20"`, `"1-5"` |
| 页脚关键词 | 关键词匹配 | `"Page 1"`, `"Confidential"` |
| 水印关键词 | 关键词匹配 | `"内部资料"`, `"绝密"` |
| 控制字符 | 正则替换 | `\x0c`, `\x0b` |
| 纯符号行 | 正则匹配 | `"---"`, `"***"`, `"___"` |
| 空行 | 长度检查 | `""`, `"   "` |

#### 面试官评分
| 维度 | 得分 | 说明 |
|------|:---:|------|
| 问题定位 | 9/10 | 识别了三类噪音来源 |
| 方案可行性 | 9/10 | 三层管线覆盖全面，但关键词匹配有漏报可能 |
| 表达清晰度 | 9/10 | 结构清晰 |
| **综合** | **9.0/10** | 优秀 |

---

## 六、分块策略（合并 Q6-2 + Q6-3）

### Q6-2. HierarchicalChunker 和简单按字符切分有什么区别？为什么不用 LangChain？

#### 面试官追问
- LangChain 的 RecursiveCharacterTextSplitter 不也能按段落、句子、单词边界切分吗？
- 你的分块策略是如何保持语义连贯性的？
- 自研 chunker 的维护成本高吗？为什么不站在巨人的肩膀上？
- 一个没有标题的长文档怎么处理？
- 标题匹配失败了怎么办（OCR 噪音导致标题文字有偏差）？

#### 面试者回答

**踩过的坑**：项目初期用过 `langchain.RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)`。上线后收到三次业务反馈：

1. **"搜索结果跨章节，答非所问"**：RecursiveCharacterTextSplitter 按 `\n\n` → `\n` → ` ` → `` 递归切分，完全不感知文档的语义结构。一个 h1 > h2 > h3 的文档，切分后标题被切断，chunk 丢失上下文。

2. **"表格数据查不到，但搜关键词能搜到无关表格"**：LangChain 把表格当普通文本处理，大表格直接在行中间切断。检索时前半段表格有关键词但无完整语义，后半段丢失了表头。

3. **"图片搜索结果返回空白"**：RecursiveCharacterTextSplitter 直接忽略图片，图片内容被混到相邻文本 chunk 中，embedding 时图片信息丢失。

**为什么不采用 LangChain 的核心原因**：

| 维度 | LangChain RecursiveCharacterTextSplitter | 自研 HierarchicalChunker |
|:---:|------|------|
| **输入** | 纯文本字符串 | ParsedPage 结构化对象（text + headings + tables + images + code） |
| **标题感知** | 不感知，在任意位置切断 | `_build_sections()` 按标题位置分 section |
| **表格处理** | 当普通文本，行中间切断 | TableChunker 表头保留 + 按行切分 |
| **图片处理** | 忽略，混入文本 | 独立 chunk + 上下文 + LLM caption |
| **代码块处理** | 递归切断（可能破坏语法） | 按函数/类定义边界切分 |
| **输出格式** | 字符串列表 | 带 chunk_id / chunk_type / page_num / metadata 的 dict |
| **层级信息** | 无 | heading_path 完整路径（"h1 > h2 > h3"） |
| **依赖** | `langchain` 全家桶（~50MB） | 零外部依赖（仅 `re`, `difflib`） |

**为什么不用其他方案**：

| 方案 | 为什么不用 |
|------|------|
| `MarkdownHeaderTextSplitter` | 只识别标题，不处理表格/图片/代码 |
| `TokenTextSplitter` | 按 token 计数切分，无视语义边界 |
| `SemanticChunker` | 依赖 sentence embedding 计算相似度，性能差（500 页 PDF 需要额外 2000 次 API 调用） |
| LlamaIndex `SemanticSplitterNodeParser` | 同样不处理多模态内容（表格/图片/代码），且引入 LlamaIndex 依赖 |

**最终方案（已实现）**：`HierarchicalChunker` 管线

```python
# app/chunkers/hierarchical_chunker.py

"""
HierarchicalChunker 管线：
  1. 从 ParsedPage.headings 构建 document structure
  2. 按标题位置拆成 section（模糊匹配，容差 0.8）
  3. 每个 section 按自然段落拆成 atomic segments
  4. 贪心聚合 segments 到 chunk（≤ CHUNK_SIZE_TEXT）
  5. 每个 chunk prepend 完整标题路径 "h1 > h2 > h3"
"""

def chunk(self, pages: list[ParsedPage], doc_id, creator):
    for page in pages:
        # 1. 文本层级切分（感知标题 + 段落）
        text_chunks = self._chunk_text_with_headings(page, doc_id, creator)
        chunks.extend(text_chunks)

        # 2. 表格切分（TableChunker，表头保留 + 行切分）
        for table_md in page.tables:
            chunks.extend(table_chunker.chunk_table(table_md, page, doc_id, creator))

        # 3. 代码块切分（函数/类边界检测）
        for code in page.code_blocks:
            chunks.extend(self._chunk_code(code, page, doc_id, creator))

        # 4. 图片切分（content 留空，等 LLM 填充 caption）
        for img_path in page.images:
            chunks.append({
                "chunk_id": generate_chunk_id(doc_id, page.page_num, "image", img_path),
                "content": "",  # LLM 后续填充
                "metadata": {"context_text": page.text[:500], "image_path": img_path},
            })
```

**核心原则**：

| 原则 | 实现 | 解决的问题 |
|:---:|------|------|
| 不在标题边界切割 | `_build_sections()` 按标题位置分 section | 避免章节被切断 |
| 段落感知 | `_split_to_segments()` 按 `\n\n` 拆原子片段 | 保持语义连贯性 |
| 贪心聚合 | `_greedy_aggregate()` 不超过 CHUNK_SIZE_TEXT | 避免碎片化 |
| 标题路径 prepend | `heading_path = "h1 > h2 > h3"` | 保留层级上下文 |
| 模糊匹配容差 | `SequenceMatcher >= 0.8` | 容错 OCR 噪音 |
| 长段落二次拆分 | 按句子边界（`。！？.!?`）聚合 | 避免超长 chunk |
| 强制截断兜底 | 在 80% 位置处找句号边界 | 段落/句子边界都不起作用时 |
| 代码块特殊处理 | 2000 字符限制，按 `def/class` 边界 | 保持代码语法完整 |

**与 RecursiveCharacterTextSplitter 对比**：

| 特性 | Recursive | HierarchicalChunker |
|:---:|------|------|
| 标题边界 | 不尊重 | 尊重（section 边界） |
| 层级信息 | 丢失 | 保留（heading_path） |
| 段落边界 | 部分尊重 | 尊重（atomic segment） |
| 代码块 | 递归切断 | 按函数/类边界切 |
| 表格 | 当普通文本 | 表头保留 + 行切分 |
| 图片 | 忽略 | 独立 chunk + 上下文 |

**关于维护成本的回应**：
- HierarchicalChunker 核心代码约 400 行，加上 TableChunker（200 行）和 SlideChunker（90 行），总共约 700 行
- 对比引入 LangChain 后需要理解其庞大的抽象层（Document / BaseDocumentTransformer / BaseTextSplitter / Tokenizer），学习成本更高
- 自研 chunker 的 API 完全贴合项目需求，改动时只需改一处，而非等待上游版本发布

**标题匹配失败的兜底**：
1. 优先精确匹配 → `line == heading_text`
2. 包含关系检查 → `heading in line or line in heading`，相似度 0.9
3. SequenceMatcher 模糊匹配 → 阈值 0.8
4. 都不匹配 → 该行为普通内容，归属到当前 section

```python
# 模糊匹配示例：OCR 输出 "第三章   财务概览" vs 解析标题 "第三章 财务概览"
# 相似度 0.94 > 0.8 → 匹配成功
```

**匹配效果验证**：

| 场景 | 相似度 | 结果 |
|------|:---:|------|
| 精确匹配 `第一章 概述` | 1.000 | MATCH |
| 尾随空格 `第一章 概述 ` | 1.000 | MATCH |
| 数字变体 `第1章 概述` | 0.833 | MATCH |
| 页眉噪音 `Chapter 1 [Page 5]` | 0.900 | MATCH |
| 缺失空格 `2.3系统架构` | 0.933 | MATCH |
| 后缀噪音 `三、实验结果与分析` | 0.900 | MATCH |
| 完全不同 `Introduction` vs `Conclusion` | 0.455 | NO MATCH |
| 不同数字 `1.1` vs `9.9` | 0.333 | NO MATCH |

#### 面试官评分
| 维度 | 得分 | 说明 |
|------|:---:|------|
| 问题定位 | 10/10 | 从三个真实业务反馈出发，定位 LangChain 的三个本质缺陷 |
| 方案可行性 | 10/10 | 700 行自研 vs 50MB 依赖，ROI 清晰，管线设计完整 |
| 表达清晰度 | 9/10 | 对比表直观，替代方案分析全面 |
| **综合** | **9.7/10** | 优秀 |

---

## 七、数据库连接池

### Q7. `get_engine()` 每次创建新 Engine 会导致什么后果？怎么修复的？

#### 面试者回答

**踩过的坑**：原来代码：

```python
def get_engine():
    return create_engine(settings.DATABASE_URI, pool_recycle=3600, pool_size=10)

def get_session():
    return Session(bind=get_engine())
```

**问题**：每次 `get_session()` 都调用 `get_engine()` 创建**全新的 Engine**，意味着每次数据库操作都创建新连接池（10 条连接）。100 次查询 = 1000 条数据库连接。连接泄漏 + 内存泄漏 + MySQL 连接数耗尽。

**最终方案（已实现）**：

```python
# app/models.py
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.DATABASE_URI,
            pool_recycle=3600,
            pool_size=10,
            max_overflow=5,
        )
    return _engine
```

#### 面试官评分
| 维度 | 得分 | 说明 |
|------|:---:|------|
| 问题定位 | 10/10 | 这是严重的生产事故级 bug |
| **综合** | **10/10** | 必须修复 |

---

## 八、健康检查

### Q8. `/health` 接口有哪些问题？怎么优化的？

#### 面试者回答

**踩过的坑**：原来每次健康检查都 `redis.from_url()` 创建新连接（泄漏），MySQL 挂起时整个请求阻塞事件循环，错误信息暴露内部连接串。

**最终方案（已实现）**：

```python
# app/api/routers/health.py

# 共享 Redis 连接池
_redis_client = None
def _get_redis():
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(
            settings.REDIS_URL,
            socket_connect_timeout=3, socket_timeout=3,
        )
    return _redis_client

# 并发检查 + 超时
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    futures = {name: pool.submit(fn) for name, fn in checks}
    for name, future in futures.items():
        results[name] = future.result(timeout=5)  # 5s 超时
```

#### 面试官评分
| 维度 | 得分 | 说明 |
|------|:---:|------|
| 问题定位 | 9/10 | 三个问题都命中 |
| **综合** | **9.0/10** | 优秀 |

---

## 九、安全加固

### Q9. 上传和搜索接口有哪些安全风险？

#### 面试者回答

**踩过的坑**：
1. `upload_file` 整个文件读入内存后才检查大小 → 50MB 恶意文件直接撑满内存
2. `query_image` 接受任意本地路径 → 路径遍历攻击（`../../etc/passwd`）
3. 错误信息 `str(exc)` 直接返回客户端 → 暴露数据库连接串、文件路径等

**最终方案（已实现）**：

```python
# app/services/document_service.py — 流式读取
with open(file_path, "wb") as f:
    while True:
        chunk = file.file.read(1024 * 1024)  # 1MB 块
        if not chunk: break
        if file_size_mb > settings.MAX_FILE_SIZE_MB:
            os.unlink(file_path)  # 清理已写入的部分
            raise HTTPException(400, "File too large")
        f.write(chunk)
```

```python
# app/api/routers/search.py — 路径遍历检查
def _validate_image_path(path: str) -> str:
    real_path = os.path.realpath(os.path.expanduser(path))
    for forbidden in ("/etc/", "/proc/", "/sys/", "/dev/"):
        if real_path.startswith(forbidden):
            raise HTTPException(403, "Access denied")
    return real_path
```

```python
# 所有路由的 except 块：不再暴露 str(exc)
except Exception as exc:
    logger.exception("...", error=str(exc))  # 日志保留
    return _err(trace_id=trace_id, message="Search failed", code="SEARCH_ERROR")  # 客户端脱敏
```

#### 面试官评分
| 维度 | 得分 | 说明 |
|------|:---:|------|
| 问题定位 | 10/10 | OOM 攻击、路径遍历、信息泄露全部命中 |
| **综合** | **10/10** | 优秀 |

---

### Q10. 错误信息暴露内部细节的问题，为什么重要？

#### 面试者回答

**踩过的坑**：所有路由原来都是 `message=f"Search failed: {str(exc)}"`。如果异常是 SQLAlchemy 连接失败，客户端会看到 `Can't connect to MySQL server on 'root:password@db:3306'`。攻击者直接拿到数据库凭据。

**最终方案（已实现）**：所有 except 块改为：
```python
except HTTPException:
    raise  # HTTPException 直接透传（400/403/404 等业务错误）
except Exception as exc:
    logger.exception("...", error=str(exc))  # 内部日志保留完整信息
    return _err(trace_id=trace_id, message="操作失败", code="ERROR")  # 客户端脱敏
```

#### 面试官评分
| 维度 | 得分 | 说明 |
|------|:---:|------|
| 安全意识 | 10/10 | 这是 OWASP Top 10 级别的安全问题 |
| **综合** | **10/10** | 优秀 |

---

## 十、临时文件管理

### Q11. 上传的文件什么时候被清理？

#### 面试者回答

**踩过的坑**：原来上传的文件和合并的 MD 临时文件**永远不会被清理**，`UPLOAD_DIR` 会无限增长。运行一个月后磁盘占满。

**最终方案（已实现）**：

```python
# app/tasks.py
def _cleanup_input_file(file_path: str) -> None:
    """安全删除输入文件（静默失败，不抛异常）。"""
    try:
        if file_path and os.path.isfile(file_path):
            os.unlink(file_path)
            logger.debug("input_file_cleaned", path=file_path)
    except Exception:
        logger.debug("input_file_cleanup_failed", path=file_path)

# 任务完成或失败时都调用
if mysql_ok and milvus_ok:
    _cleanup_input_file(file_path)
    return {"status": "completed", ...}
except Exception as exc:
    _cleanup_input_file(file_path)
    self.retry(...)
```

#### 面试官评分
| 维度 | 得分 | 说明 |
|------|:---:|------|
| 问题定位 | 9/10 | 磁盘泄漏是隐蔽但严重的生产问题 |
| **综合** | **9.0/10** | 优秀 |

---

## 十一、重排序（Rerank）

### Q12. Reranker 加载失败时静默跳过，怎么发现的？

#### 面试者回答

**踩过的坑**：`FlagEmbedding` 要求 PyTorch >= 2.4，但项目环境是 2.2.2。Reranker 的 `_load_model()` 捕获所有异常后设 `self._loaded = False`，然后 `rerank()` 方法直接 `return candidates`（原顺序返回）。搜索结果有 `rerank_score: null` 但不报错，用户根本不知道重排序没生效。

**当前状态**：代码中已有静默回退逻辑（`reranker.py:47-49`），但需要运维人员通过监控日志中的 `reranker_not_available` 来发现。

**推荐改进（未实现）**：

| 优先级 | 改进方案 |
|:---:|------|
| P0 | 启动时检查 PyTorch 版本，不兼容则报错或告警 |
| P1 | 在 `/health` 接口中增加 reranker 状态检查 |
| P2 | `rerank_score` 为 null 时返回 warning 字段给前端 |

#### 面试官评分
| 维度 | 得分 | 说明 |
|------|:---:|------|
| 问题定位 | 9/10 | 发现静默失败的生产可观测性问题 |
| 方案可行性 | 8/10 | 给出了分优先级的改进建议 |
| **综合** | **8.5/10** | 优秀 |

---

## 十二、未实现但高优先级的改进

### Q14. 项目中还有哪些生产级问题没解决？优先级怎么排？

#### 面试者回答

以下是基于项目代码分析出的**未实现**改进清单：

| 优先级 | 问题 | 影响 |
|:---:|------|------|
| ~~**P0**~~ | ~~HierarchicalChunker 标题精确匹配 bug~~ | **已在 Q6-2 实现：模糊匹配 + matched_headings 防重复** |
| ~~**P0**~~ | ~~Embedding 批量调用是串行~~ | **已在 Q15 实现：ThreadPoolExecutor 10 路并发** |
| ~~**P0**~~ | ~~Milvus 故障重试粒度过粗~~ | **已在 Q22 实现：Redis 中间缓存 + 存储层独立重试** |
| **P1** | Reranker 加载失败静默跳过无告警 | 搜索质量下降但无人知晓 |
| **P1** | Celery worker 中 `session.close()` 但没释放 Engine | 长运行场景下 SQLAlchemy 连接泄漏 |
| **P1** | `_build_sections` 段落边界只有 `\n\n` | 无法处理 Markdown 中单行分隔的段落 |
| **P2** | 搜索接口缺少请求频率限制 | 恶意用户可快速消耗 DashScope API 配额 |
| **P2** | Celery 任务无进度上报到 Redis pub/sub | 前端只能轮询，无法实时推送 |
| **P2** | 分块算法无版本号控制 | 换了 chunk size 参数后旧数据 chunk_id 不兼容 |

#### 面试官评分
| 维度 | 得分 | 说明 |
|------|:---:|------|
| 全局视野 | 10/10 | 从代码层面反推了所有潜在问题 |
| **综合** | **10/10** | 优秀 |

---

## 十三、Embedding 性能

### Q15. `_call_api` 中逐个 input 调用 DashScope API 的性能瓶颈怎么解决？

#### 面试者回答

**踩过的坑**：`unified_embedder.py` 的 `_call_api`：

```python
for inp in input_list:  # 串行！
    resp = dashscope.MultiModalEmbedding.call(...)
```

假设一个 500 页的 PDF 解析出 2000 个 chunk，每个 chunk 一次 API 调用，按每次 200ms 计算，需要 **400 秒**（6.7 分钟）纯 embedding 时间。

**最终方案（已实现）**：ThreadPoolExecutor 并发编码（10 路）

```python
# app/embedders/unified_embedder.py
import concurrent.futures

def _call_api(self, input_list: list[dict]) -> list[list[float]]:
    """调用 DashScope MultiModalEmbedding API（并发版）。"""
    max_workers = min(10, len(input_list)) if input_list else 1

    def _embed_one(inp):
        # 处理图片引用 + 文本截断 + 重试（原有逻辑完整保留）
        max_retries = 3
        for attempt in range(max_retries):
            resp = dashscope.MultiModalEmbedding.call(...)
            if resp.status_code == 200 and resp.output is not None:
                break
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        return embeddings[0]["embedding"]

    # 并发编码：10 路并行（根据 DashScope QPS 限制调整）
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        vectors = list(pool.map(_embed_one, input_list))
    return vectors
```

| 指标 | 优化前（串行） | 优化后（10 路并发） |
|:---:|:---:|:---:|
| 2000 chunk × 200ms | 400s（6.7 分钟） | ~40s（40 秒） |
| 提速倍数 | 1x | ~10x |

#### 面试官评分
| 维度 | 得分 | 说明 |
|------|:---:|------|
| 问题定位 | 10/10 | 量化了 2000 chunk × 200ms = 6.7 分钟 |
| 方案可行性 | 9/10 | 并发方案可行，需注意 API QPS 限制 |
| **综合** | **9.5/10** | 优秀 |

---

## 十四、RAG 评估体系（合并 Q16 + RAGAS）

### Q16. 你是怎么评估这个 RAG 项目的召回率和准确率的？

#### 面试官追问
- RAG 系统没有 ground truth 标签，怎么量化检索质量？
- 如果业务方说"搜索结果不相关"，你怎么排查是 embedding 问题、chunking 问题还是 reranker 问题？
- 你有没有做过 A/B 测试？
- 没有人工标注 chunk_id 时怎么办？

#### 面试者回答

**踩过的坑**：项目初期完全没有评估体系。上线后业务方反馈"搜索质量不好"，但没有任何指标能说清楚"不好"到什么程度、哪里不好。我们花了一周时间建立了完整的评估体系。

**核心难点**：RAG 系统不像分类任务有明确的 label。每个 query 的"正确答案"是动态的，取决于文档库的内容。

**最终方案（已实现）**：四层评估体系，代码位于 `app/evaluation/`

#### 第一层：离线评估（人工标注测试集）

```python
# app/evaluation/rag_evaluator.py

class RAGEvaluator:
    """RAG 检索质量评估器。"""

    def __init__(self, test_cases: list[TestCase]):
        """
        test_cases: [TestCase(query="...", relevant_chunk_ids=["id1", "id2"]), ...]
        人工标注 100-500 个测试 query，每个 query 标注相关 chunk_id 列表。
        """
        self.test_cases = test_cases

    def evaluate(self, top_k: int = 10, per_tag: bool = True) -> dict:
        """计算 Recall@K, Precision@K, MRR, NDCG@K，支持按 tag 分组输出。"""
        # ... 遍历 test_cases，调用 search_service.search，聚合指标
        return {
            "recall_at_k": 0.72,
            "precision_at_k": 0.65,
            "mrr": 0.58,
            "ndcg_at_k": 0.68,
            "per_tag": {
                "表格查询": {"recall_at_k": 0.80, ...},
                "代码查询": {"recall_at_k": 0.62, ...},
            },
        }
```

**测试集构建工具**（`app/evaluation/test_dataset.py`）：

| 方法 | 说明 | 数量 |
|:---:|------|:---:|
| `load_from_json()` | 从 JSON 文件加载人工标注 | 100-300 |
| `reverse_annotate()` | 从已有高质量搜索结果反推 | 200-500 |
| `llm_assist()` | LLM 生成候选片段，输出 JSON 待人工校验 | 500+ |
| `merge()` + `deduplicate()` | 合并多来源数据并去重 | — |
| `split(ratio=0.8)` | 按 8:2 划分为训练集/验证集 | — |

#### 第二层：RAGAS LLM-as-Judge 评估

```python
# app/evaluation/ragas_evaluator.py

class RagasEvaluator:
    """RAGAS Context Precision/Recall 评估器。

    使用 ragas 框架的 LLM-as-Judge 指标，自动判断检索内容的相关性和
    答案覆盖率。无需人工标注 chunk_id，只需准备 query + ground_truth。
    """

    def evaluate(self, top_k: int = 5, per_tag: bool = True) -> dict:
        """执行 RAGAS Context Precision/Recall 评估。"""
        for case in self.test_cases:
            # 1. 执行检索
            hits = search_service.search(query=case.query, top_k=top_k)
            contexts = [h["chunk"]["content"] for h in hits]

            # 2. 构建 ragas 样本
            samples = [{
                "user_input": case.query,
                "reference": case.ground_truth,
                "retrieved_contexts": contexts,
            }]

            # 3. LLM-as-Judge 评分
            result = evaluate(EvaluationDataset.from_list(samples),
                             metrics=[ContextPrecision(llm=llm),
                                     ContextRecall(llm=llm)])
```

**RAGEvaluator vs RagasEvaluator 对比**：

| 维度 | RAGEvaluator（精确匹配） | RagasEvaluator（LLM-as-Judge） |
|:---:|------|------|
| **标注要求** | 需人工标注 relevant_chunk_id | 只需 query + ground_truth 答案文本 |
| **判断方式** | chunk_id 集合精确匹配 | LLM 判断检索内容是否覆盖答案信息 |
| **指标** | Recall@K / Precision@K / MRR / NDCG@K | Context Precision / Context Recall |
| **成本** | 零 API 成本 | 需要调用 LLM（gpt-4o-mini 等） |
| **适用场景** | 已有标注数据，快速回归测试 | 无标注数据，端到端质量评估 |
| **准确性** | 精确但依赖标注质量 | 灵活但受 LLM 判断能力影响 |

**典型使用场景**：
- RAGEvaluator：CI/CD 回归测试，每次改 chunk size 或 reranker 参数后自动跑
- RagasEvaluator：新文档类型接入时，准备 20-50 条 query + 答案，快速评估检索质量

#### 第三层：模块化消融实验

```python
# app/evaluation/ablation.py

study = AblationStudy(test_cases)
report = study.run(top_k=10)
print(study.print_comparison(report))

# 输出:
# ======================================================================
# 消融实验结果对比
# ======================================================================
# 配置                          Recall@K  Precision@K      MRR     NDCG@K
# ----------------------------------------------------------------------
# full                            0.7200       0.6500   0.5800     0.6800
# no_rerank                       0.6100       0.5200   0.4500     0.5200
# no_hybrid                       0.5800       0.5000   0.4800     0.5000
# no_rerank_no_hybrid             0.4500       0.3800   0.3200     0.3800
# ----------------------------------------------------------------------
# 增益分析:
#   Reranker Recall 增益:  +0.1100
#   Reranker NDCG 增益:    +0.1600
#   Hybrid  Recall 增益:   +0.1400
#   Hybrid  NDCG 增益:     +0.1800
```

**消融实验结论**（典型场景）：

| 配置 | Recall@10 | NDCG@10 | 结论 |
|:---:|:---:|:---:|------|
| 完整 pipeline | 0.72 | 0.68 | 最佳 |
| 无 reranker | 0.61 | 0.52 | Reranker 贡献 +11% Recall |
| 纯 BM25 | 0.45 | 0.38 | 关键词匹配不够语义化 |
| 纯 dense | 0.58 | 0.50 | 向量检索召回广但不精确 |

#### 第四层：线上反馈收集

```python
# app/evaluation/feedback.py
# 前端搜索后上报用户点击行为
collector = FeedbackCollector()
collector.record(
    query="公司 2024 年营收多少？",
    top_result_ids=["c1", "c2", ..., "c10"],
    clicked_ids=["c3", "c7"],
    dwell_times_ms=[3000, 12000],
)

# 持续监测 60 分钟窗口内的满意度
metrics = collector.get_metrics(window_minutes=60)
# → {"click_through_rate": 0.42, "avg_dwell_time_ms": 5200, ...}

# 自动检测低满意度 query（点击率 < 0.3 且出现 >= 3 次）
low_sq = collector.get_low_satisfaction_queries(window_minutes=60, threshold=0.3)
```

**当前指标基线**（示例数据）：

| 指标 | 值 | 说明 |
|:---:|:---:|------|
| Recall@10 | 0.72 | 72% 的相关文档在前 10 结果中 |
| Precision@10 | 0.65 | 65% 的返回结果是相关的 |
| MRR | 0.58 | 平均第一个相关结果排在第 1.7 位 |
| NDCG@10 | 0.68 | 排序质量良好 |

**已实现模块总览**：

| 模块 | 文件 | 功能 |
|:---:|------|------|
| 离线评估器 | `evaluation/rag_evaluator.py` | Recall@K / Precision@K / MRR / NDCG@K，支持按 tag 分组 |
| RAGAS 评估器 | `evaluation/ragas_evaluator.py` | Context Precision / Recall（LLM-as-Judge），无需 chunk_id 标注 |
| 测试集构建器 | `evaluation/test_dataset.py` | JSON/CSV 导入、反向标注、LLM 辅助标注、去重合并、划分训练/验证集 |
| 消融实验框架 | `evaluation/ablation.py` | full / no_rerank / no_hybrid / 组合，一键输出对比表 |
| 线上反馈收集 | `evaluation/feedback.py` | 点击行为记录、滑动窗口满意度、低满意度 query 告警 |

**推荐后续优化**：

| 优先级 | 改进方案 |
|:---:|------|
| P0 | 构建 100 条人工标注测试集（已有工具，待填充数据） |
| P1 | 在 CI/CD 中加入回归测试（每次改 chunk size 自动跑评估） |
| P2 | 前端集成反馈 API，收集真实用户点击数据 |

#### 面试官评分
| 维度 | 得分 | 说明 |
|------|:---:|------|
| 方法论 | 10/10 | 四层评估体系完整（离线 + RAGAS + 消融 + 线上），指标选择合理 |
| 实践性 | 10/10 | 五个模块已实现（rag_evaluator / ragas_evaluator / test_dataset / ablation / feedback），开箱即用 |
| 表达清晰度 | 9/10 | 结构清晰，数据有说服力 |
| **综合** | **9.7/10** | 优秀 |

---

## 十五、Milvus 索引策略（扩展）

### Q17. Milvus 使用 AUTOINDEX，为什么不用 HNSW 或 IVF_FLAT？怎么调参？

#### 面试官追问
- 当前数据量是多少？AUTOINDEX 在什么数据量下最优？
- metric_type 为什么用 IP（内积）而不是 L2 或 COSINE？
- 向量有没有归一化？

#### 面试者回答

**当前配置分析**：

```python
# app/milvus_client.py:156-160
index_params.add_index(
    field_name="vector",
    index_type="AUTOINDEX",  # Milvus 自动选择
    metric_type="IP",        # 内积
)
```

**AUTOINDEX 的原理**：Milvus 根据数据量自动选择索引类型：
- 数据量 < 5000 → Flat（暴力搜索，无索引开销）
- 5000 < 数据量 < 100k → IVF_FLAT
- 数据量 > 100k → HNSW 或 DiskANN

**为什么当前用 AUTOINDEX**：
1. 项目处于早期阶段，数据量不确定，AUTOINDEX 让 Milvus 自动决策
2. 避免了手动调参（如 IVF 的 `nlist`、HNSW 的 `M` 和 `efConstruction`）
3. 但生产环境**不推荐 AUTOINDEX**，应该根据实际数据量手动选择

**IP vs COSINE vs L2 的选择**：

| 度量类型 | 适用场景 | 是否需要归一化 |
|:---:|------|:---:|
| IP（内积） | 向量已归一化时 = COSINE | 必须归一化 |
| COSINE | 向量未归一化 | 不需要 |
| L2（欧氏距离） | 需要绝对距离 | 不需要 |

**当前问题**：DashScope `qwen3-vl-embedding` 返回的向量**是否归一化**取决于 API 版本。如果未归一化，用 IP 会导致模长大的向量得分更高，**语义相似但模长不同的向量排序错误**。

**最终方案（推荐改进，未实现）**：

```python
# app/embedders/unified_embedder.py — 向量归一化
import math

def _normalize_vector(vec: list[float]) -> list[float]:
    """L2 归一化，确保 IP = COSINE。"""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        return [x / norm for x in vec]
    return vec

def _call_api(self, input_list: list[dict]) -> list[list[float]]:
    # ... 原有代码 ...
    vector = embeddings[0]["embedding"]
    vectors.append(self._normalize_vector(vector))  # 统一归一化
    return vectors
```

**生产环境推荐配置**（根据数据量）：

| 数据量 | 索引类型 | nlist / M | ef / efConstruction |
|:---:|:---:|:---:|:---:|
| < 10k | FLAT | - | - |
| 10k - 100k | IVF_FLAT | 1024 | search: 64 |
| 100k - 1M | HNSW | M=16 | efConstruction=256, search: ef=64 |
| > 1M | DISKANN | - | - |

**当前状态**：AUTOINDEX + IP 可用但不最优。向量未做归一化保护。

#### 面试官评分
| 维度 | 得分 | 说明 |
|------|:---:|------|
| 知识广度 | 10/10 | 理解 AUTOINDEX 内部机制和不同度量类型的区别 |
| 深度 | 9/10 | 能指出 IP + 未归一化的潜在问题 |
| **综合** | **9.5/10** | 优秀 |

---

## 十六、混合检索与 RRF 融合（扩展）

### Q18. 混合检索为什么用 RRF 而不是加权融合？RRF 和 Weighted RRF 有什么区别？

#### 面试官追问
- RRF（Reciprocal Rank Fusion）的公式是什么？为什么它比线性加权好？
- RRF 和 Weighted RRF 有什么区别？
- 当前代码中 `HYBRID_SEARCH_ALPHA = 0.5` 为什么没有用到？
- dense 和 BM25 的 `top_k * 2` 是怎么来的？

#### 面试者回答

**踩过的坑**：`config.py` 中定义了 `HYBRID_SEARCH_ALPHA = 0.5`，但**代码中完全没有用到这个配置**。这是一个遗留的死配置。

```python
# app/config.py:85
HYBRID_SEARCH_ALPHA: float = 0.5  # 0=稀疏检索, 1=稠密检索 — 但没人用！
```

**当前混合检索实现**（`milvus_client.py:231-264`）：

```python
def _hybrid_search(self, vector, query_text, top_k, expr, output_fields):
    # dense 向量请求: limit=top_k * 2
    dense_req = AnnSearchRequest(
        data=[vector], anns_field="vector",
        param={"metric_type": "IP", "params": {"ef": 64}},
        limit=top_k * 2,  # 为什么 * 2？
        expr=expr,
    )
    # BM25 文本请求: limit=top_k * 2
    sparse_req = AnnSearchRequest(
        data=[query_text], anns_field="sparse_vector",
        param={"metric_type": "BM25"},
        limit=top_k * 2,
        expr=expr,
    )
    # RRF 融合
    res = client.hybrid_search(
        collection_name=name,
        reqs=[dense_req, sparse_req],
        ranker=RRFRanker(),  # RRF 融合
        limit=top_k,
        output_fields=output_fields,
    )
```

**RRF 公式**：

```
RRF_score(d) = Σ (1 / (k + rank_i(d)))
```

其中 k 是常数（Milvus 默认 60），`rank_i(d)` 是文档 d 在第 i 个检索器中的排名。

**RRF vs Weighted RRF vs 线性加权**：

| 对比项 | RRF（标准） | Weighted RRF | 线性加权 |
|:---:|------|------|------|
| **公式** | `1/(k+r1) + 1/(k+r2)` | `w1/(k+r1) + w2/(k+r2)` | `α·score_dense + (1-α)·score_bm25` |
| **分数归一化** | 不需要（用排名） | 不需要（用排名） | 需要（量纲不同） |
| **超参数** | 只有 k（固定 60） | k + w1, w2 需调参 | α 需调参 |
| **鲁棒性** | 高 | 中（权重需随数据分布调整） | 低（分数分布变化时需重新调参） |
| **适用场景** | 默认首选 | 明确知道某个检索器更重要时 | 需要精细控制权重比例时 |

**RRF 和 Weighted RRF 的核心区别**：
- **标准 RRF**：两个检索器权重相等（各 1/(k+rank)），无需调参
- **Weighted RRF**：可以调整 dense 和 sparse 的相对权重（如 dense 0.6 + sparse 0.4），但需要额外实验确定最佳权重
- **为什么当前用标准 RRF**：在项目早期，没有足够标注数据调权重参数，标准 RRF 开箱即用且效果稳定。后续可通过消融实验对比标准 RRF vs Weighted RRF 来决定是否需要加权

**为什么选 RRF（而非线性加权）**：
1. 不需要手动调 alpha，RRF 自动平衡两个检索器
2. dense 向量（IP 分数）和 BM25 分数的量纲完全不同，线性加权需要复杂的归一化
3. RRF 基于排名，对分数分布不敏感，更稳定

**`top_k * 2` 的原因**：
- 每个检索器先召回 `top_k * 2` 个候选，RRF 融合后再截断到 `top_k`
- 这样可以避免某个检索器的优质候选被过早截断
- 但 `* 2` 是经验值，没有理论依据。如果 `top_k = 10`，每个检索器回 20 个候选，融合后取 top 10

#### 面试官评分
| 维度 | 得分 | 说明 |
|------|:---:|------|
| 问题定位 | 10/10 | 发现了死配置的 bug，并能清晰区分 RRF / Weighted RRF / 线性加权 |
| 知识深度 | 9/10 | 能解释 RRF 公式和不同融合方式的优劣 |
| **综合** | **9.5/10** | 优秀 |

---

## 十七、Chunk Size 调优（扩展）

### Q19. CHUNK_SIZE_TEXT = 800, CHUNK_OVERLAP_TEXT = 150，这些参数是怎么确定的？

#### 面试官追问
- 不同文档类型（PDF vs Markdown）的 chunk size 应该不同吗？
- Embedding 模型的上下文窗口限制对 chunk size 有什么影响？
- 有没有试过不同的 chunk size 对检索效果的影响？

#### 面试者回答

**踩过的坑**：当前 800/150 是从 LangChain 默认值抄来的，**没有任何实验依据**。换了一份新的文档类型（如代码仓库、法律合同），同样的参数可能完全不适用。

**当前配置**：
```python
# app/config.py:63-64
CHUNK_SIZE_TEXT: int = 800      # 字符数
CHUNK_OVERLAP_TEXT: int = 150   # 重叠字符数
```

**问题分析**：

| 参数 | 当前值 | 问题 |
|:---:|:---:|------|
| 按字符数切分 | 800 字符 | 中文和英文差异大：800 英文 ≈ 150 词，800 中文 ≈ 250-400 字 |
| 固定重叠 | 150 字符 | 重叠率 = 150/800 = 18.75%，但对中文可能过高或过低 |
| 不区分文档类型 | 统一参数 | 代码、表格、正文用同一套参数不合理 |

**推荐方案（未实现）**：

```python
# 方案 A：按文档类型自适应 chunk size
CHUNK_SIZE_BY_DOC_TYPE = {
    "pdf": {"text": 600, "overlap": 100},     # PDF 解析后较短，chunk 可以小
    "markdown": {"text": 1000, "overlap": 200},  # MD 保留了完整结构
    "docx": {"text": 800, "overlap": 150},     # 默认
    "pptx": {"text": 400, "overlap": 50},      # 幻灯片内容精炼
    "code": {"text": 2000, "overlap": 0},       # 代码不宜重叠
}

# 方案 B：按 token 数而非字符数切分
import tiktoken  # 或 DashScope 的 tokenizer

def count_tokens(text: str) -> int:
    """用 embedding 模型相同的 tokenizer 计算 token 数。"""
    # 确保 chunk 不超过 embedding 模型的 token 限制
    return len(tokenizer.encode(text))

# qwen3-vl-embedding 的 token 限制约 8192 tokens
# 推荐 chunk_size_tokens = 512-1024 tokens
```

**Chunk Size 对比实验设计**（未实现）：

```python
def chunk_size_experiment(doc_path, sizes=[400, 600, 800, 1200, 1600]):
    """对比不同 chunk size 对检索效果的影响。"""
    results = {}
    for size in sizes:
        # 1. 用该 chunk size 解析文档
        chunks = chunk_with_size(doc_path, chunk_size=size, overlap=size // 5)

        # 2. 嵌入 + 存储
        store_chunks(chunks)

        # 3. 用测试集评估
        metrics = evaluator.evaluate(top_k=10)
        results[size] = metrics

        # 4. 清理
        cleanup_test_data()

    # 绘制对比图
    for size, m in results.items():
        print(f"chunk_size={size}: Recall@10={m['Recall@10']:.3f}, "
              f"NDCG@10={m['NDCG@10']:.3f}")
```

**经验法则**：

| 场景 | 推荐 chunk_size | 推荐 overlap |
|:---:|:---:|:---:|
| 通用文本 | 800 字符 / 512 tokens | 15-20% |
| 代码 | 2000 字符 | 0（代码不宜重叠） |
| 表格 | 1600 字符 | 0（保留表头） |
| 法律/医疗文档 | 1200-1600 字符 | 10%（长上下文重要） |
| QA 对 | 不切分 | 0 |

**当前状态**：固定 800/150，无实验依据。

#### 面试官评分
| 维度 | 得分 | 说明 |
|------|:---:|------|
| 批判性思维 | 10/10 | 能质疑默认参数的合理性 |
| 实验设计 | 9/10 | 对比实验方案完整 |
| **综合** | **9.5/10** | 优秀 |

---

## 十八、多模态对齐问题（扩展）

### Q20. 图片 chunk 的 embedding 用的是 caption 文本 + 图片 base64 联合编码，但检索时可能只有文本查询，这会导致模态不匹配吗？

#### 面试官追问
- 训练时图文联合编码，检索时纯文本查询，两者在同一个向量空间吗？
- 如果一个图片的 caption 是"系统架构图"，用户搜"架构图"，能搜到吗？
- DashScope `qwen3-vl-embedding` 的纯文本编码和图文联合编码输出是否兼容？

#### 面试者回答

**当前代码分析**：

```python
# 写入时（tasks.py:145-146）— 图片 chunk 联合编码
vec = unified_embedder.encode_multimodal(caption, image_path)

# 检索时（hybrid_search.py:62-63）— 纯文本查询
dense_vector = unified_embedder.encode_query(query)

# unified_embedder.py 中三种编码方法都调用同一个 _call_api
# 底层都是 dashscope.MultiModalEmbedding.call()，只是 input 不同
```

**关键问题**：DashScope `qwen3-vl-embedding` 是一个**多模态统一嵌入模型**，设计上将文本和图像映射到同一个 2560 维向量空间。所以：
- `encode_query("架构图")` → 文本向量
- `encode_multimodal("系统架构图", image_data)` → 图文联合向量
- 两者在同一个向量空间，**理论上可以直接计算相似度**

**但实际遇到的坑**：
1. **模态偏差（Modality Gap）**：虽然模型声称统一空间，但纯文本查询和图文联合编码的结果可能存在系统性偏移。文本查询倾向于匹配文本语义，联合编码会引入图片视觉信息
2. **Caption 质量敏感**：如果 LLM 生成的 caption 不准确（如"图片"或"一张图"），联合编码的效果严重下降
3. **纯文本图片 chunk 的 content 字段**：当前图片 chunk 的 `content` 存的是 caption（`tasks.py:109`），搜索时 BM25 也会匹配 caption 文本

**实际测试**（假设）：

| 查询 | 图片 caption | 能否搜到 | 原因 |
|:---:|------|:---:|------|
| "架构图" | "系统架构图：展示微服务架构" | 能 | BM25 命中 + dense 匹配 |
| "架构图" | "图片" | BM25 不能，dense 可能 | Caption 无信息量 |
| "架构图" | "一张显示组件关系的图片" | 可能 | 语义不精确匹配 |

**推荐改进（未实现）**：

```python
# 1. 提升 caption 质量
# 当前：summarize_image(context, heading) — 只给上下文和标题
# 改进：给 LLM 更多 prompt，要求生成可检索的描述
PROMPT = """请根据图片内容和上下文，生成一段简洁的描述（50-100字），
包含图片中的关键实体、关系和概念，使其能被自然语言查询准确检索到。

上下文标题: {heading}
上下文文本: {context}
"""

# 2. 双重编码策略：图片 chunk 同时存 caption 向量和联合向量
# Milvus 中增加一个 caption_vector 字段
entity = {
    "vector": joint_vector,        # 图文联合编码（以图搜图用）
    "metadata": {
        "caption_vector": text_only_vector,  # caption 纯文本编码（文本查询用）
    },
}
# 搜索时对文本查询同时检索两个向量字段
```

#### 面试官评分
| 维度 | 得分 | 说明 |
|------|:---:|------|
| 分析深度 | 9/10 | 能识别模态偏差这一多模态 RAG 核心难题 |
| 方案可行性 | 8/10 | 双重编码方案有效但增加存储成本 |
| **综合** | **8.5/10** | 优秀 |

---

## 十九、生产可观测性（扩展）

### Q21. 上线后，怎么监控 RAG 服务的健康状态？除了 `/health` 接口还需要什么？

#### 面试官追问
- 怎么知道 embedding API 的调用成功率？
- 怎么知道搜索结果的满意度？
- Celery 任务队列积压了怎么告警？

#### 面试者回答

**当前状态**：项目只有 `/health` 接口做基础的服务可用性检查（MySQL 连接、Redis PING、Milvus describe_collection），**缺乏业务级可观测性**。

**推荐监控体系（未实现）**：

```python
# app/observability/metrics.py（新增模块）
from prometheus_client import Counter, Histogram, Gauge
import time

# ========== 核心指标 ==========

# 1. 文档解析
PARSE_DURATION = Histogram("parse_duration_seconds", "文档解析耗时",
                           buckets=[1, 5, 10, 30, 60, 120, 300])
PARSE_CHUNK_COUNT = Histogram("parse_chunk_count", "解析产生的 chunk 数量",
                              buckets=[10, 50, 100, 500, 1000, 5000])

# 2. Embedding
EMBED_DURATION = Histogram("embed_duration_seconds", "Embedding 耗时")
EMBED_BATCH_SIZE = Histogram("embed_batch_size", "Embedding 批次大小",
                             buckets=[10, 50, 100, 500, 1000, 2000])
EMBED_API_ERRORS = Counter("embed_api_errors_total", "Embedding API 错误次数")

# 3. 搜索
SEARCH_DURATION = Histogram("search_duration_seconds", "搜索耗时")
SEARCH_TOP_K_RESULTS = Histogram("search_result_count", "搜索结果数量")
SEARCH_RERANK_APPLIED = Counter("search_rerank_applied_total", "Rerank 执行次数")

# 4. Celery
CELERY_QUEUE_DEPTH = Gauge("celery_queue_depth", "Celery 待处理任务数")
CELERY_TASK_DURATION = Histogram("celery_task_duration_seconds", "Celery 任务耗时")
CELERY_RETRIES = Counter("celery_retries_total", "Celery 重试次数", ["task_name"])

# 5. 存储
MYSQL_CONNECTION_POOL_SIZE = Gauge("mysql_pool_size", "MySQL 连接池大小")
MYSQL_CONNECTIONS_IN_USE = Gauge("mysql_pool_in_use", "MySQL 使用中连接数")
MILVUS_UPSERT_BATCH_DURATION = Histogram("milvus_upsert_duration_seconds", "Milvus upsert 耗时")
```

**告警规则**（推荐配置）：

| 指标 | 阈值 | 级别 |
|:---:|:---:|:---:|
| Embedding API 错误率 > 5%（5 分钟内） | P0 | 立即告警 |
| Celery 队列深度 > 100（10 分钟） | P1 | 延迟处理 |
| 搜索 P99 延迟 > 2s | P1 | 性能降级 |
| Reranker 未加载 | P1 | 搜索质量降级 |
| MySQL 连接池使用率 > 80% | P2 | 资源预警 |
| 磁盘使用率 > 80%（上传目录） | P2 | 存储预警 |

**搜索质量反馈闭环（已实现 `FeedbackCollector`）**：

```python
# app/evaluation/feedback.py
@router.post("/search-feedback")
async def submit_search_feedback(data: SearchFeedback):
    """
    前端上报用户对搜索结果的反馈。
    data: {query, top_result_ids, clicked_ids, dwell_times_ms, doc_id}
    """
    collector = FeedbackCollector()
    collector.record(
        query=data.query,
        top_result_ids=data.top_result_ids,
        clicked_ids=data.clicked_ids,
        dwell_times_ms=data.dwell_times_ms,
        doc_id=data.doc_id,
    )

    # 计算即时满意度指标
    metrics = collector.get_metrics(window_minutes=60)
    if metrics["click_through_rate"] < 0.3:
        logger.warning("low_search_satisfaction", query=data.query)

    return {"success": True}
```

#### 面试官评分
| 维度 | 得分 | 说明 |
|------|:---:|------|
| 全局视野 | 10/10 | 从业务指标到技术指标全覆盖 |
| 可操作性 | 9/10 | Prometheus 指标 + 告警规则可直接落地 |
| **综合** | **9.5/10** | 优秀 |

---

## 二十、Milvus 故障恢复与重试优化

### Q22. 如果 Milvus 服务重启了，正在写入的 Celery 任务会怎样？数据会丢失吗？

#### 面试官追问
- Milvus Lite 和 Milvus Server 在故障恢复上有什么区别？
- 当前代码有处理 Milvus 不可用时的降级策略吗？
- 如果 Milvus 持续不可用，Celery 重试 3 次后任务状态是什么？

#### 面试者回答

**踩过的坑**：最初实现中，Milvus 写入失败直接抛异常 → Celery `self.retry()` → 整条流水线重跑（parse → chunk → caption → embed → store）。500 页 PDF 重试意味着重新解析 500 页 + 重新生成 2000 个 embedding，重试成本极高。

**最终方案（已实现）**：两步改造 —— Redis 中间缓存 + 存储层独立重试

```python
# app/tasks.py

# ========== 第一步：中间结果缓存到 Redis ==========

def _cache_pipeline_state(task_id: str, state: dict) -> None:
    """将 chunks + entities 缓存到 Redis，TTL 1 小时。"""
    r = _get_redis()
    r.setex(f"pipeline_state:{task_id}", 3600, pickle.dumps(state))

def _restore_pipeline_state(task_id: str) -> dict | None:
    """重试时恢复缓存的中间状态。"""
    r = _get_redis()
    data = r.get(f"pipeline_state:{task_id}")
    if data:
        return pickle.loads(data)
    return None

# 任务改造：首次执行时正常走 parse → chunk → embed → 缓存 → store
# 重试时检测到缓存 → 直接从 store 阶段开始
@celery_app.task(bind=True)
def parse_and_chunk_task(self, file_path, doc_id, creator):
    cached = _restore_pipeline_state(task_id)
    if cached and "chunks" in cached and "entities" in cached:
        # 跳过 parse/chunk/embed，直接存储
        chunks = cached["chunks"]
        entities = cached["entities"]
    else:
        # 正常流程
        chunks = chunk_pages(pages, doc_id, creator)
        entities = build_entities(chunks, vector_map)
        _cache_pipeline_state(task_id, {"chunks": chunks, "entities": entities})

    # 存储阶段（带独立重试）
    ...
```

```python
# ========== 第二步：Milvus / MySQL 各自独立重试 ==========

def _retry_with_backoff(func, name, max_retries=5, base_delay=2.0):
    """对单个存储操作进行指数退避重试，不重跑整条流水线。"""
    for attempt in range(max_retries + 1):
        try:
            if func():
                return True
        except Exception as e:
            last_err = e
        if attempt < max_retries:
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
    return False

# Milvus 失败时只重试 Milvus，不影响 MySQL
milvus_ok = _retry_with_backoff(
    lambda: milvus_client.batch_upsert(entities),
    name="milvus", max_retries=5, base_delay=2.0,
)
if not milvus_ok:
    raise Exception("Milvus storage failed after retry")

mysql_ok = _retry_with_backoff(
    lambda: insert_doc_chunks(chunks),
    name="mysql", max_retries=5, base_delay=2.0,
)

# 完成后清理 Redis 缓存
_delete_pipeline_state(task_id)
```

**流程对比**：

| 场景 | 旧方案 | 新方案 |
|:---|------|------|
| Milvus 写入失败 | 抛异常 → Celery retry → 重跑整条流水线 | 指数退避重试 Milvus 5 次（2s→4s→8s→16s→32s） |
| 重试时中间结果 | 重新解析 + 重新嵌入 | 从 Redis 恢复 chunks + entities |
| 重试成本 | 500 页 PDF 需 ~60 秒重算 | ~0.1 秒（直接存储） |
| Milvus 持续不可用 | Celery retry 3 次后任务 failed | 独立重试 5 次仍失败 → 抛异常 → Celery retry（此时仍有缓存） |

**三个面试官追问的回答**：

1. **Milvus Lite vs Server 故障恢复**：
   - Milvus Lite 有文件锁保护（`milvus_client.py:52-74`），多进程冲突时自动重试
   - Milvus Server 是独立进程，故障恢复由服务端保证，客户端只需重试连接
   - 当前 `_retry_with_backoff` 两种模式都适用

2. **Milvus 不可用的降级策略**：
   - 独立重试 5 次后仍失败 → 抛异常 → Celery `self.retry()`（最多 3 次，指数退避 60s→120s→240s）
   - 每次 Celery 重试都能从 Redis 恢复中间状态，不重算
   - 总计可承受 `5 × 3 = 15` 次写入尝试

3. **Celery 重试 3 次后**：
   - 任务状态变为 `failed`，`error_message` 写入 tasks 表
   - 中间缓存仍在 Redis（TTL 1 小时），可手动触发重试
   - 输入文件未删除（仅在成功或最终失败时清理）

**已实现改进项**：

| 改进项 | 效果 | 优先级 |
|:---:|------|:---:|
| 中间结果缓存到 Redis | 重试时跳过 parse/chunk/embed，只需重试存储 | P0 ✓ |
| Milvus 写入单独重试 | 不重跑整个 pipeline | P0 ✓ |
| MySQL 写入单独重试 | Milvus 成功 MySQL 失败时不丢失数据 | P0 ✓ |
| 完成后清理 Redis 缓存 | 防止内存泄漏 | P0 ✓ |
| Milvus Server 模式连接池复用 | 避免每次 upsert 都创建新连接 | P1 |

#### 面试官评分
| 维度 | 得分 | 说明 |
|------|:---:|------|
| 问题定位 | 10/10 | 精准识别重试粒度过粗和中间结果丢失两个根因 |
| 方案可行性 | 10/10 | Redis 缓存 + 独立重试，改动小效果大，两个任务都已改造 |
| 表达清晰度 | 9/10 | 新旧方案对比表格清晰 |
| **综合** | **9.7/10** | 优秀 |

---

## 二十一、分块算法版本控制（扩展）

### Q23. 改了 chunk size 参数或分块算法后，旧的 chunk_id 还有效吗？怎么处理新旧数据兼容？

#### 面试官追问
- 当前 chunk_id 是 SHA-256(content + doc_id + page_num + chunk_type + sort_order)，改了参数后 content 会变吗？
- 如果同一个文档用新旧两种算法各分一次，chunk_id 会冲突吗？
- 怎么回滚分块算法的变更？

#### 面试者回答

**当前 chunk_id 生成逻辑**：

```python
# app/chunkers/base.py
def generate_chunk_id(doc_id, page_num, chunk_type, content, sort_order=0):
    raw = f"{doc_id}:{page_num}:{chunk_type}:{sort_order}:{content}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]
```

**分析**：
- chunk_id 依赖于 `content`（chunk 文本内容）和 `sort_order`（排序序号）
- 改了 chunk_size 参数 → 切分边界变化 → content 变化 → **chunk_id 完全不同**
- 改了分块算法 → 切分逻辑变化 → content 变化 → **chunk_id 完全不同**

**场景**：
1. 用户 A 上传了 doc_001，用 chunk_size=800 分块 → 产生 100 个 chunk_id
2. 管理员把 chunk_size 改为 600
3. 用户 A 重新上传 doc_001 → 产生 130 个**完全不同的** chunk_id
4. MySQL 中 `INSERT IGNORE` 不会冲突（因为 chunk_id 不同）
5. Milvus upsert 不会冲突（因为 chunk_id 不同）
6. 但旧的 100 个 chunk 和向量**变成了孤儿数据**

**问题**：孤儿数据不会被清理，Milvus 和 MySQL 中会积累大量过期 chunk。

**推荐方案（未实现）**：

```python
# 方案：给 chunk_id 加入算法版本号
def generate_chunk_id(doc_id, page_num, chunk_type, content, sort_order=0, version="v1"):
    """
    version: 分块算法版本号
    v1: 当前版本
    v2: 改了 chunk_size 后的版本
    v3: 改了分块逻辑的版本
    """
    raw = f"{version}:{doc_id}:{page_num}:{chunk_type}:{sort_order}:{content}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]

# 删除旧版本数据
def cleanup_old_chunks(doc_id: str, current_version: str = "v1"):
    """删除指定文档的旧版本 chunk。"""
    # MySQL: DELETE FROM doc_chunks WHERE doc_id = ? AND chunk_id NOT LIKE 'v1:%'
    # Milvus: delete expr = f"doc_id == '{doc_id}' and chunk_id not like 'v1:%'"
    pass

# 或在上传时自动清理
def upload_new_version(file_path, doc_id, creator):
    # 1. 先删除旧版本 chunk
    chunk_repo.delete_by_doc_id(doc_id)
    milvus_client.delete_by_doc_id(doc_id)

    # 2. 用新算法生成 chunk（chunk_id 含新版本号）
    chunks = chunk_pages(...)
    ...
```

**版本管理策略**：

| 策略 | 优点 | 缺点 |
|:---:|------|------|
| 上传新版本时自动删除旧数据 | 简单，不积累孤儿数据 | 用户无法对比新旧结果 |
| chunk_id 加版本号共存 | 可对比新旧结果 | 存储成本翻倍 |
| 增加 `algorithm_version` 字段 | 可按版本查询和清理 | 需要改 schema |

#### 面试官评分
| 维度 | 得分 | 说明 |
|------|:---:|------|
| 前瞻性 | 10/10 | 预见算法迭代带来的数据兼容问题 |
| 方案可行性 | 9/10 | 版本号方案简洁有效 |
| **综合** | **9.5/10** | 优秀 |

---

## 总结：改进全览

| 改进项 | 状态 | 改动文件 |
|--------|:---:|------|
| 确定性 chunk_id | 已实现 | `chunkers/base.py`, `hierarchical_chunker.py`, `table_chunker.py`, `slide_chunker.py` |
| 先 Milvus 后 MySQL | 已实现 | `tasks.py` |
| Celery 指数退避 | 已实现 | `tasks.py` |
| 移除 MySQL 回查 | 已实现 | `search/hybrid_search.py`, `services/search_service.py` |
| Caption 存 metadata | 已实现 | `tasks.py`, `services/search_service.py` |
| 解析器延迟实例化 | 已实现 | `chunkers/__init__.py`, `main.py` |
| 插件注册机制 | 已实现 | `chunkers/__init__.py`, `main.py` |
| Engine 单例 | 已实现 | `models.py` |
| 健康检查优化 | 已实现 | `api/routers/health.py` |
| 流式上传 | 已实现 | `services/document_service.py` |
| 搜索路径安全 | 已实现 | `api/routers/search.py` |
| 临时文件清理 | 已实现 | `tasks.py` |
| 错误信息脱敏 | 已实现 | `api/routers/upload.py`, `search.py`, `chunks.py` |
| MD 文件夹多页解析 | 已实现 | `services/folder_service.py`, `tasks.py` |
| 语义文件排序 | 已实现 | `services/folder_service.py` |
| 分页限制上限 | 已实现 | `api/routers/chunks.py` |
| 三层噪音过滤 | 已实现 | `parsers/noise_filter.py`, `parsers/markdown_parser.py` |
| HierarchicalChunker 管线 | 已实现 | `chunkers/hierarchical_chunker.py` |
| 标题模糊匹配 | 已实现 | `chunkers/hierarchical_chunker.py` |
| Embedding 并发 | 已实现 | `embedders/unified_embedder.py` |
| 离线评估器 | 已实现 | `evaluation/rag_evaluator.py` |
| RAGAS 评估器 | 已实现 | `evaluation/ragas_evaluator.py` |
| 测试集构建器 | 已实现 | `evaluation/test_dataset.py` |
| 消融实验框架 | 已实现 | `evaluation/ablation.py` |
| 线上反馈收集 | 已实现 | `evaluation/feedback.py` |
| Redis 中间缓存 | 已实现 | `tasks.py`（`_cache_pipeline_state` / `_restore_pipeline_state`） |
| 存储层独立重试 | 已实现 | `tasks.py`（`_retry_with_backoff`，Milvus/MySQL 各 5 次） |
| Reranker 健康检查 | 未实现 P1 | `embedders/reranker.py`, `api/routers/health.py` |
| SQLAlchemy 长连接泄漏 | 未实现 P1 | `repositories/chunk_repo.py`, `task_repo.py` |
| API 限流 | 未实现 P2 | `api/middleware/` |
| 向量归一化 | 未实现 P1 | `embedders/unified_embedder.py` |
| 分块算法版本控制 | 未实现 P2 | `chunkers/base.py` |

---

## 附录：面试提问优先级排序

> 以下按「面试官提问概率」和「自然提问顺序」排列。建议准备时优先熟记 P0，其次 P1，P2 可作为加分项主动提及。

### 🔴 P0 — 必问（出现概率 95%+，面试开场必问）

| 顺序 | 问题 | 对应 Q | 面试官意图 |
|:---:|------|:---:|------|
| 1 | **"请用一分钟介绍你的项目"** | Q24（零） | 了解候选人表达能力 + 项目概貌 |
| 2 | **"MySQL 和 Milvus 怎么保证数据一致性？"** | Q1（一） | 考察分布式系统设计能力 |
| 3 | **"为什么自研分块器，不用 LangChain？"** | Q6-2（六） | 考察技术选型能力 |
| 4 | **"混合检索怎么做的？RRF 是什么？"** | Q18（十六） | 考察检索算法理解 |
| 5 | **"Celery 任务失败怎么处理？"** | Q2 + Q22（二 + 二十） | 考察异步系统容错设计 |

**应对策略**：这 5 个问题几乎必问，且通常连续追问。建议准备一个 **2 分钟版本** 的完整项目叙事：
> 项目一句话概括 → 核心架构（FastAPI + Celery + MySQL + Milvus） → 最大挑战（数据一致性） → 最大亮点（自研分块 + 四层评估）

---

### 🟡 P1 — 高概率（出现概率 70%+，深入追问）

| 顺序 | 问题 | 对应 Q | 面试官意图 |
|:---:|------|:---:|------|
| 6 | **"你的分块策略具体怎么做的？和简单按字符切有什么区别？"** | Q6-2（六） | 考察算法设计能力 |
| 7 | **"OCR 输出的文本有噪音怎么处理？"** | Q6-1（五） | 考察工程实践细节 |
| 8 | **"搜索接口性能怎么优化？"** | Q4（二） | 考察性能优化思维 |
| 9 | **"Embedding 批量调用的性能瓶颈怎么解决？"** | Q15（十三） | 考察并发编程能力 |
| 10 | **"你怎么评估检索质量？召回率/准确率怎么算？"** | Q16（十四） | 考察评估方法论 |
| 11 | **"Milvus 重启了，正在写入的任务会怎样？"** | Q22（二十） | 考察故障恢复设计 |
| 12 | **"解析器为什么从实例改为延迟实例化？"** | Q5（三） | 考察架构设计能力 |
| 13 | **"Reranker 加载失败静默跳过，怎么发现？"** | Q12（十一） | 考察生产可观测性意识 |

**应对策略**：这些是 P0 回答后的自然追问。比如你说完混合检索，面试官一定会问 RRF 公式；你说完自研分块，一定会问具体怎么做。建议每个问题准备 **30 秒精简版** 回答。

---

### 🟢 P2 — 中概率（出现概率 40%-60%，加分项）

| 顺序 | 问题 | 对应 Q | 面试官意图 |
|:---:|------|:---:|------|
| 14 | **"MD 文件夹多个 .md 文件怎么按顺序拼接？"** | Q6（四） | 考察实际问题解决能力 |
| 15 | **"Chunk Size 参数怎么确定的？"** | Q19（十七） | 考察实验设计能力 |
| 16 | **"图片 chunk 的 embedding 和文本在同一个空间吗？"** | Q20（十八） | 考察多模态理解 |
| 17 | **"数据库连接池的坑？"** | Q7（七） | 考察基础工程能力 |
| 18 | **"安全方面做了哪些加固？"** | Q9 + Q10（九） | 考察安全意识 |
| 19 | **"Milvus 索引策略怎么选的？"** | Q17（十五） | 考察向量数据库理解 |
| 20 | **"上传的文件什么时候清理？"** | Q11（十） | 考察细节关注度 |

**应对策略**：这些问题通常作为发散话题出现。如果时间充裕，可以主动提及展示深度；如果时间紧，简要回答即可。

---

### ⚪ P3 — 低概率（出现概率 <30%，压轴追问）

| 问题 | 对应 Q | 面试官意图 |
|------|:---:|------|
| **"改了分块算法后旧数据怎么兼容？"** | Q23（二十一） | 考察版本管理思维 |
| **"生产监控怎么做？"** | Q21（十九） | 考察运维能力 |
| **"还有哪些没解决的问题？"** | Q14（十二） | 考察自我认知和规划能力 |

---

### 📋 典型面试流程模拟（45 分钟）

| 时间段 | 环节 | 预计涉及问题 |
|:---:|------|------|
| **0-5 min** | 自我介绍 + 项目介绍 | Q24（零） |
| **5-15 min** | 核心架构深入 | Q1 → Q3（一） → Q5（三） |
| **15-25 min** | 分块策略 | Q6-2（六） → Q6-1（五） → Q6（四） |
| **25-35 min** | 检索 + 性能 | Q4（二） → Q18（十六） → Q15（十三） → Q12（十一） |
| **35-40 min** | 评估体系 | Q16（十四） |
| **40-45 min** | 故障恢复 + 开放问题 | Q22（二十） → Q14（十二） |

**实际面试中**，面试官会根据你的回答动态调整。但上述流程覆盖了 80% 以上的提问路径。建议按这个顺序准备，前面的优先熟记。
