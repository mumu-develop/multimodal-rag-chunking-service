多模态文档检索增强生成（RAG）管理系统

【角色】核心开发
【技术栈】Python / FastAPI / Celery / Redis / MySQL / Milvus / theta大模型平台 / RAGAS

---

项目概述

基于 FastAPI + Celery 构建的多模态文档解析与智能切片服务，支持 PDF / DOCX / PPTX / Markdown / Excel 五种文档格式。采用 Converter 统一转换层将各格式转换为 Markdown，由层级切分器统一处理；使用 theta大模型平台 多模态嵌入（qwen3-vl-embedding，2560 维统一向量）实现文本+图片检索，Milvus 支持 dense + BM25 混合检索 + RRF 融合 + 交叉编码器重排序；集成 RAGAS 评估框架实现全链路质量评估。知识库多 Collection 隔离，统一身份认证 + RBAC，全链路可观测性。

---

核心工作

1. 多格式文档解析（Converter 统一转换层 + 注册表模式）
   - 注册表延迟加载，统一输出 ConverterOutput 模型（Markdown + 表格 + 图片 + 代码块）
   - PDF 调用 OCR API 识别元素类型，DOCX/PPTX/Excel 结构化提取
   - 支持运行时注册扩展

2. 层级感知切分器（5 种策略可选）
   - smart（智能切分）/ heading（标题切分）/ semantic（语义切分）/ recursive（递归切分）/ fixed（固定长度）
   - 按标题边界分 section，Token 级控制（CHUNK_TOKEN_LIMIT=1024）+ 重叠（CHUNK_TOKEN_OVERLAP=100）
   - 每个 chunk prepend 完整标题路径保留上下文
   - 噪音过滤：规则匹配 + 关键词过滤 + 跨页重复检测 + 敏感信息脱敏
   - 每个文档可独立配置切分策略

3. 多模态统一嵌入（theta大模型平台 qwen3-vl-embedding）
   - 2560 维统一向量，支持纯文本、纯图片、文本+图片联合编码
   - 指数退避重试，图片通过 OSS HTTP URL 访问

4. 混合检索与重排序（dense + BM25 + RRF + Rerank）
   - Milvus Function 自动生成 BM25 sparse_vector，RRF 融合两路结果
   - 可选 Qwen3-VL-Rerank 交叉编码器重排序（top_k=50）
   - 三种查询模式：text / image / both

5. 异步流水线（Celery task chain）
   - upload → parse → chunk → embed → store_milvus →  store_mysql → update_status
   - 进度实时上报，失败重试 3 次 + 指数退避
   - pipeline_nodes 字段记录每个节点执行状态和耗时

6. Milvus Schema 设计（单向量架构 + 知识库隔离）
   - vector（2560d）+ sparse_vector（BM25），content 字段启用中文分词器
   - 多 Collection 方案实现知识库隔离，Schema 版本检测自动重建
   - HNSW 索引（AUTOINDEX + IP），支持百万级向量高效检索

7. RAGAS 全链路评估（LLM-as-Judge）
   - context_precision / context_recall / faithfulness / answer_relevance / answer_correctness
   - 多线程并发评估，按 tag 分组输出指标
   - 测试集自动生成（RagasTestsetGenerator）+ 评估记录持久化

8. 知识库管理平台（REST API）
   - 知识库/文档/切片 CRUD，批量上传，切分策略配置
   - 文档树结构（Paragraph + Chunks 双层），意图树三级分类
   - Dashboard：切片统计、检索延迟、AI 性能（成功率/P95）

9. 统一身份认证与 RBAC（Middleware 拦截链）
   - API Key 认证 + 角色分权（admin/user）+ trace_id 全链路追踪 + 请求耗时统计

10. 多维度健康检查（并发 + 超时 + 脱敏）
    - ThreadPoolExecutor 并发检查 MySQL/Milvus/Redis，5s 超时
    - 三级响应：ok / degraded / unavailable，错误信息脱敏

---

技术成果

- 5 种文档格式解析，Converter 统一转换层 + 注册表模式
- 层级切分器 5 种策略可选，噪音过滤 + 敏感信息脱敏
- theta大模型平台 2560 维统一向量，支持文本+图片联合编码
- Milvus dense + BM25 混合检索 + RRF 融合 + 交叉编码器重排序
- Celery 7 步异步流水线 + 进度上报 + 流水线节点追踪
- RAGAS 5 项评估指标 + 测试集自动生成 + 评估记录持久化
- 知识库多 Collection 隔离 + Schema 版本检测自动重建
- 统一身份认证 + RBAC + trace_id 全链路追踪

---

延伸应用

接入企业内部知识库（技术文档、产品手册、FAQ），实现智能问答系统；支持 PDF/DOCX/PPTX/Excel 文档自动切片入库，通过意图树路由到对应知识库；支持图片检索和图文联合查询。