# 多模态RAG分块服务

## 概述

多模态RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合多种模态（文本、图像、表格）的信息检索与生成技术。它通过从外部知识库中检索相关内容，增强大语言模型的生成能力，从而提高回答的准确性和可追溯性。

传统的RAG系统主要处理纯文本数据，而多模态RAG扩展了这一能力，使其能够同时理解和检索文本、图像、表格等多种类型的内容。

## 文档解析

文档解析是RAG系统的第一步。本服务支持以下格式的文档解析：

- **PDF**：使用unstructured库进行高分辨率解析，提取文本、图像和表格
- **DOCX**：解析Word文档，保留标题层级和表格结构
- **PPTX**：解析PowerPoint演示文稿，提取幻灯片内容和备注
- **Markdown**：直接解析Markdown文件，保留标题层级和代码块
- **图片（PNG/JPG/JPEG）**：提取图片并使用VLM生成描述

解析器从文档中提取页面级别的ParsedPage对象，包含文本内容、图片路径、表格数据和页面元信息。

## 分块策略

系统将解析后的文档按照不同策略进行分块：

### 文本分块

文本使用RecursiveCharacterTextSplitter进行递归字符分块，参数设置为：
- chunk_size = 800（每个chunk最大字符数）
- chunk_overlap = 150（相邻chunk的重叠字符数）

这种策略确保语义连贯的文本不会被截断，同时重叠部分提供上下文连续性。

### 表格分块

表格分块根据表格大小采用不同策略：
- 小于800字符的表格：整体作为一个chunk
- 大于800字符的表格：按行拆分，每行保留表头信息

### 图像分块

从文档中提取的图像会进行以下处理：
1. 使用VLM（视觉语言模型）生成图片描述caption
2. 提取图片周围的上下文文本（200字符）
3. 将caption和上下文组合为图像chunk的内容

## 向量嵌入

系统使用Qwen3-VL-Embedding-2B模型进行统一的多模态嵌入：

- **文本嵌入**：将文本编码为2048维向量
- **图像嵌入**：将图片编码为2048维向量
- **统一表征**：文本和图像共享同一个向量空间，支持跨模态检索

模型加载时启用bfloat16精度，使用device_map="auto"自动分配到可用GPU。开发环境可通过EMBED_MOCK=true使用随机向量进行模拟。

## 向量数据库

Milvus是一个开源向量数据库，本服务使用Milvus Lite（本地文件模式）进行开发和测试。

集合schema包含以下字段：
- chunk_id：主键，VARCHAR(128)
- doc_id：文档ID，VARCHAR(128)
- chunk_type：分块类型（text/image/table/code），VARCHAR(32)
- content：文本内容或图片caption，VARCHAR(5000)
- image_path：图片文件路径，VARCHAR(500)
- metadata：JSON格式的元数据
- sparse_vector：BM25稀疏向量
- vector：2048维dense向量

索引配置：
- dense向量使用AUTOINDEX（HNSW）
- sparse向量使用SPARSE_INVERTED_INDEX（BM25）

## 混合检索

混合检索结合了dense向量检索和sparse BM25检索的优势：

1. **Dense检索**：使用嵌入向量进行语义相似度搜索，metric_type为IP（内积）
2. **Sparse检索**：使用BM25算法进行关键词匹配，基于Milvus内置的BM25 Function
3. **结果融合**：通过WeightedRanker(alpha, 1-alpha)融合两种检索结果

alpha参数控制融合权重：
- alpha=0.5：dense和sparse等权重（默认）
- alpha=1.0：仅使用dense向量
- alpha=0.0：仅使用sparse BM25

## 重排序

重排序使用BAAI/bge-reranker-v2-m3交叉编码器模型，对混合检索返回的候选结果进行精排：

1. 混合检索返回top-N个候选（N由RERANK_TOP_N控制，默认50）
2. 从MySQL获取完整的chunk内容
3. 使用交叉编码器对(query, content)对计算分数
4. 按重排序分数降序排列，返回top-k

重排序可以显著提升top-k准确率，特别是当查询包含特定实体或专业术语时。

## 系统架构

系统采用微服务架构，包含以下组件：

- **FastAPI**：Web框架，提供REST API
- **Celery + Redis**：异步任务队列，处理文档解析和嵌入
- **MySQL**：关系数据库，存储chunk元数据和任务状态
- **Milvus**：向量数据库，存储和检索向量
- **Redis**：消息代理和任务结果后端

异步管线：upload → parse → chunk → embed → store_mysql → store_milvus → update_status

## API接口

### 上传文档

```
POST /api/v1/upload
```
上传文件，返回task_id和doc_id。支持PDF、DOCX、PPTX、PNG、JPG、JPEG、MD格式，最大50MB。

### 查询任务状态

```
GET /api/v1/tasks/{task_id}
```
返回任务状态（pending/processing/completed/failed）、进度百分比和阶段信息。

### 获取分块列表

```
GET /api/v1/chunks?doc_id=xxx&page=1&size=10&type=text
```
按doc_id分页查询chunk，支持按chunk_type过滤。

### 向量搜索

```
POST /api/v1/search
```
执行混合检索，请求参数包括query、top_k、modal、filters、doc_ids、use_rerank。

## 配置说明

通过.env文件配置系统参数：

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| DATABASE_URI | mysql+pymysql://... | MySQL连接字符串 |
| MILVUS_LOCAL_DB | "" | Milvus Lite本地路径 |
| EMBED_MOCK | false | 是否使用模拟向量 |
| EMBED_MODEL_NAME | Qwen/Qwen3-VL-Embedding-2B | 嵌入模型名称 |
| RERANK_ENABLED | true | 是否启用重排序 |
| HYBRID_SEARCH_ALPHA | 0.5 | 混合搜索权重 |
| CHUNK_SIZE_TEXT | 800 | 文本分块大小 |
| CHUNK_OVERLAP_TEXT | 150 | 文本分块重叠 |
