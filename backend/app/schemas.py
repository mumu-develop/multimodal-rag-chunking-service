from typing import Any, Optional

from pydantic import BaseModel, Field


# ============================================================
# 请求模型
# ============================================================

class SearchRequest(BaseModel):
    query: str = Field(default="", max_length=2000)
    query_image: Optional[str] = Field(default=None, description="查询图片的本地路径")
    top_k: int = Field(default=10, ge=1, le=100)
    modal: str = Field(default="text", pattern="^(text|image|both)$")
    filters: Optional[dict[str, Any]] = None
    use_rerank: bool = Field(default=True)
    doc_ids: Optional[list[str]] = None
    kb_id: str = Field(default="", description="知识库ID（必填，用于查找 collection_name）")
    collection_name: Optional[str] = Field(default=None, description="直接指定 collection 名称（优先级高于 kb_id）")
    data_type: str = Field(default="", description="数据类型过滤: all/faq/chunk（默认all）")

    def is_image_query(self) -> bool:
        return bool(self.query_image and self.query_image.strip())


# ============================================================
# Chunk 模型（通用）
# ============================================================

class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_type: str  # text | image | table | code
    content: str
    caption: Optional[str] = None
    page_num: int
    metadata: Optional[dict[str, Any]] = None
    status: int = 1
    audit_status: int = 0
    creator: Optional[str] = None
    doc_type: str = ""
    heading_level: int = 0
    heading_text: Optional[str] = None
    sort_order: int = 0


# ============================================================
# 响应模型
# ============================================================

class BaseResponse(BaseModel):
    success: bool
    code: str
    message: Optional[str] = None
    trace_id: str = ""


class UploadResponseData(BaseModel):
    task_id: str
    doc_id: str
    doc_type: str = ""
    doc_name: str = ""  # 文件名
    error: Optional[str] = None  # 错误信息（批量上传时可能有）


class BatchUploadResponseData(BaseModel):
    """批量上传响应数据。"""
    total: int  # 总文件数
    success: int  # 成功数
    failed: int  # 失败数
    results: list[UploadResponseData]  # 每个文件的上传结果


class UploadResponse(BaseResponse):
    data: Optional[UploadResponseData] = None


class BatchUploadResponse(BaseResponse):
    data: Optional[BatchUploadResponseData] = None


class TaskStatusData(BaseModel):
    status: str  # pending | processing | completed | failed（待处理/处理中/已完成/已失败）
    progress: int = 0
    chunk_count: int = 0
    stage: Optional[str] = None
    error_message: Optional[str] = None
    doc_type: Optional[str] = None


class TaskStatusResponse(BaseResponse):
    data: Optional[TaskStatusData] = None


class TaskListItem(BaseModel):
    task_id: str
    doc_id: Optional[str] = None
    status: str
    progress: int = 0
    chunk_count: int = 0
    stage: Optional[str] = None
    error_message: Optional[str] = None
    doc_type: Optional[str] = None
    gmt_create: str = ""
    gmt_modified: str = ""


class TaskListData(BaseModel):
    total: int
    page: int
    result: list[TaskListItem]


class TaskListResponse(BaseResponse):
    data: Optional[TaskListData] = None


class ChunksListData(BaseModel):
    total: int
    page: int
    result: list[Chunk]


class ChunksListResponse(BaseResponse):
    data: Optional[ChunksListData] = None


class NestedTableData(BaseModel):
    """嵌套表格数据。"""
    nested_index: int
    chunk_id: str
    html: str = ""
    row_count: int = 0
    col_count: int = 0
    headers: list[str] = []
    rows: list[list[Any]] = []
    merges: list[dict] = []
    parent_row: Optional[int] = None
    parent_col: Optional[int] = None


class NestedTableResponse(BaseResponse):
    """嵌套表格响应。"""
    data: Optional[NestedTableData] = None


class NestedTablesListData(BaseModel):
    """嵌套表格列表数据。"""
    parent_chunk_id: str
    nested_tables: list[NestedTableData]
    total: int


class NestedTablesListResponse(BaseResponse):
    """嵌套表格列表响应。"""
    data: Optional[NestedTablesListData] = None


class SearchHitItem(BaseModel):
    chunk: Chunk
    score: float
    modal: str  # text | image（文本/图像）
    rerank_score: Optional[float] = None


class SearchResponseData(BaseModel):
    hits: list[SearchHitItem]


class SearchResponse(BaseResponse):
    data: Optional[SearchResponseData] = None


class TreeNode(BaseModel):
    heading_level: int = 0
    heading_text: str = ""
    chunk_id: str
    chunk_type: str
    content: str
    caption: Optional[str] = None
    page_num: int
    metadata: Optional[dict[str, Any]] = None
    children: list["TreeNode"] = []


class DocTreeData(BaseModel):
    doc_id: str
    doc_type: str = ""
    tree: list[TreeNode]


class DocTreeResponse(BaseResponse):
    data: Optional[DocTreeData] = None


# 允许 TreeNode 递归引用
TreeNode.model_rebuild()


# ============================================================
# 生产系统格式树状结构模型（Paragraph + Chunks 双层）
# ============================================================

class ParagraphChunk(BaseModel):
    """Paragraph 内的切片项（对应生产系统的 Chunk）。"""
    id: str = Field(..., description="chunk_id")
    paragraphId: str = Field(..., description="所属 paragraph 的标识")
    content: str
    status: int = 1
    auditStatus: int = 0
    previousId: Optional[str] = Field(default=None, description="前序切片ID")
    matchCount: int = Field(default=0, description="匹配计数（可选）")


class Paragraph(BaseModel):
    """段落节点（对应生产系统的 Paragraph）。"""
    id: str = Field(..., description="paragraph 唯一标识（基于 heading_level + heading_text 生成）")
    title: str = Field(default="", description="段落标题")
    level: int = Field(default=0, ge=0, le=7, description="层级深度")
    content: str = Field(default="", description="段落内容（标题节点的 heading_text）")
    childContentParagraph: list["Paragraph"] = Field(default_factory=list, description="子段落")
    chunks: list[ParagraphChunk] = Field(default_factory=list, description="该段落包含的切片")


class ProductionDocTreeData(BaseModel):
    """生产系统格式的文档树数据。"""
    doc_id: str
    doc_type: str = ""
    paragraphs: list[Paragraph]


class ProductionDocTreeResponse(BaseResponse):
    """生产系统格式的文档树响应。"""
    data: Optional[ProductionDocTreeData] = None


# 允许 Paragraph 递归引用
Paragraph.model_rebuild()


# ============================================================
# 评估模型
# ============================================================

class RunEvaluationRequest(BaseModel):
    top_k: int = Field(default=10, ge=1, le=100)
    per_tag: bool = Field(default=True, description="是否按 tag 分组输出指标")
    doc_id_filter: Optional[str] = Field(default=None, description="限定在某个文档内评估")


class EvaluationMetricsItem(BaseModel):
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    hit_rate: float
    avg_latency_ms: float


class RunEvaluationResponseData(BaseModel):
    num_queries: int
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    hit_rate: float
    avg_latency_ms: float
    per_tag: Optional[dict[str, EvaluationMetricsItem]] = None


class RunEvaluationResponse(BaseResponse):
    data: Optional[RunEvaluationResponseData] = None


class RunAblationRequest(BaseModel):
    top_k: int = Field(default=10, ge=1, le=100)


class AblationConfigResult(BaseModel):
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    num_queries: int


class AblationComparison(BaseModel):
    rerank_recall_gain: float
    rerank_ndcg_gain: float
    hybrid_recall_gain: float
    hybrid_ndcg_gain: float


class RunAblationResponseData(BaseModel):
    full: AblationConfigResult
    no_rerank: AblationConfigResult
    no_hybrid: AblationConfigResult
    no_rerank_no_hybrid: AblationConfigResult
    comparison: AblationComparison


class RunAblationResponse(BaseResponse):
    data: Optional[RunAblationResponseData] = None


class FeedbackRequest(BaseModel):
    query: str
    top_result_ids: list[str]
    clicked_ids: list[str] = Field(default_factory=list)
    dwell_times_ms: list[int] = Field(default_factory=list)
    doc_id: str = ""


class FeedbackMetrics(BaseModel):
    total_queries: int
    unique_queries: int
    window_minutes: int
    click_through_rate: float
    avg_clicked_per_query: float
    avg_dwell_time_ms: float
    top_clicked_chunks: list[dict] = Field(default_factory=list)


class FeedbackStatsResponseData(BaseModel):
    metrics: FeedbackMetrics
    low_satisfaction_queries: list[str] = Field(default_factory=list)


class FeedbackStatsResponse(BaseResponse):
    data: Optional[FeedbackStatsResponseData] = None


class DatasetStatsResponseData(BaseModel):
    total: int
    avg_relevant_chunks: float = 0
    by_tag: dict[str, int] = Field(default_factory=dict)
    by_doc: dict[str, int] = Field(default_factory=dict)


class DatasetStatsResponse(BaseResponse):
    data: Optional[DatasetStatsResponseData] = None


# ============================================================
# RAGAS 评估模型
# ============================================================

class RagasTestItem(BaseModel):
    """单条 RAGAS 测试用例（无需 chunk_id，只需 query + ground_truth）。"""
    query: str
    ground_truth: str
    doc_id: str = ""
    tags: list[str] = Field(default_factory=list)


class LoadRagasDatasetRequest(BaseModel):
    """加载 RAGAS 测试集请求。"""
    items: Optional[list[RagasTestItem]] = Field(
        default=None,
        description="直接传入测试用例列表",
    )
    file_path: Optional[str] = Field(
        default=None,
        description="测试集 JSON 文件路径（与 items 二选一）",
    )


class RunRagasEvaluationRequest(BaseModel):
    kb_id: str = Field(
        default="",
        description="知识库ID（用于保存评估记录）",
    )
    doc_id: str = Field(
        default="",
        description="文档ID（用于保存评估记录）",
    )
    collection_name: str = Field(
        ...,
        description="必传：指定 Milvus collection 名称",
    )
    set_id: Optional[str] = Field(
        default=None,
        description="指定评估的测试集批次 ID（从数据库加载，替代内存数据集）",
    )
    set_name: Optional[str] = Field(
        default=None,
        description="指定评估的测试集名称（从数据库加载，替代 set_id）",
    )
    top_k: int = Field(default=5, ge=1, le=100)
    per_tag: bool = Field(default=True, description="是否按 tag 分组输出指标")
    doc_ids: Optional[list[str]] = Field(
        default=None,
        description="可选：限定只评估特定文档的测试用例",
    )
    llm_override: Optional[str] = Field(
        default=None,
        description="可选：覆盖默认评估 LLM 模型名",
    )
    metrics: Optional[list[str]] = Field(
        default=None,
        description="可选：要计算的指标列表，默认 ['context_precision', 'context_recall']，可选: 'faithfulness', 'answer_relevance', 'answer_correctness'",
    )
    generate_answer: bool = Field(
        default=False,
        description="是否生成答案（Faithfulness/Answer Relevance 需要）",
    )
    save_record: bool = Field(
        default=True,
        description="是否自动保存评估记录到数据库",
    )


class RagasMetricsItem(BaseModel):
    context_precision: float
    context_recall: float
    num_queries: int


class RunRagasEvaluationResponseData(BaseModel):
    # RAGAS 指标（动态，使用 dict）
    num_queries: int
    avg_latency_ms: float
    per_tag: Optional[dict] = None
    details: Optional[list[dict]] = None


class RunRagasEvaluationResponse(BaseResponse):
    data: Optional[RunRagasEvaluationResponseData] = None


class LoadRagasDatasetResponseData(BaseModel):
    """加载测试集响应数据。"""
    set_id: str
    total: int


class LoadRagasDatasetResponse(BaseResponse):
    data: Optional[LoadRagasDatasetResponseData] = None


class RagasDatasetStatsResponseData(BaseModel):
    total_cases: int
    total_sets: int
    sets: list[dict] = Field(default_factory=list)


class RagasDatasetStatsResponse(BaseResponse):
    data: Optional[RagasDatasetStatsResponseData] = None


# ============================================================
# RAGAS 测试集生成模型
# ============================================================

class GenerateRagasTestsetRequest(BaseModel):
    """从知识库文档生成 RAGAS 测试集请求。"""
    set_name: str = Field(
        ...,
        description="测试集名称（必填）",
        min_length=1,
        max_length=255,
    )
    doc_ids: list[str] = Field(
        ...,
        description="目标文档 ID 列表",
        min_length=1,
    )
    testset_size: int = Field(default=20, ge=1, le=200, description="每个文档生成测试用例数量")
    chunk_type_filter: Optional[str] = Field(
        default=None,
        description="按 chunk 类型过滤（text/table/code/image），不传则自动过滤为文本类型",
    )
    llm_override: Optional[str] = Field(
        default=None,
        description="可选：覆盖默认 LLM 模型名",
    )
    method: str = Field(
        default="fast",
        pattern="^(fast|ragas)$",
        description="生成方式：fast=LLM Prompt 直接生成（几秒~1分钟），ragas=RAGAS 框架（~90分钟）",
    )


class GenerateRagasTestsetResponseData(BaseModel):
    task_id: str  # Celery 任务 ID，用于轮询进度
    set_id: str  # 测试集批次 ID


class GenerateRagasTestsetResponse(BaseResponse):
    data: Optional[GenerateRagasTestsetResponseData] = None


# ============================================================
# 文档列表模型
# ============================================================

class SyncedDocItem(BaseModel):
    doc_id: str
    task_id: str
    doc_type: str = ""
    chunk_count: int = 0
    gmt_create: str = ""
    gmt_modified: str = ""


class SyncedDocsListData(BaseModel):
    total: int
    page: int
    result: list[SyncedDocItem]


class SyncedDocsListResponse(BaseResponse):
    data: Optional[SyncedDocsListData] = None


# ============================================================
# TestSet 模型
# ============================================================

class TestSetItem(BaseModel):
    """单条测试集记录。"""
    id: int
    set_id: str
    set_name: str  # 测试集名称
    query: str
    ground_truth: str
    doc_id: str = ""
    tags: list[str] = Field(default_factory=list)
    gmt_create: str


class TestSetListData(BaseModel):
    """测试集分页列表数据。"""
    total: int
    page: int
    result: list[TestSetItem]


class TestSetListResponse(BaseResponse):
    """测试集分页列表响应。"""
    data: Optional[TestSetListData] = None


# ============================================================
# TestSet 批次列表模型
# ============================================================

class TestSetBatchItem(BaseModel):
    """测试集批次摘要。"""
    set_id: str
    set_name: str
    total: int
    created_at: str


class TestSetBatchListData(BaseModel):
    """测试集批次分页列表数据。"""
    total: int
    page: int
    result: list[TestSetBatchItem]


class TestSetBatchListResponse(BaseResponse):
    """测试集批次分页列表响应。"""
    data: Optional[TestSetBatchListData] = None


# ============================================================
# 知识库管理模型
# ============================================================

class KnowledgeBaseCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    collection_name: str = Field(..., min_length=1, max_length=128)
    owner: str = Field(default="admin")
    description: str = ""
    kb_type: str = Field(default="doc", description="知识库类型: faq/doc/hybrid")


class KnowledgeBaseUpdate(BaseModel):
    name: Optional[str] = None
    collection_name: Optional[str] = None
    owner: Optional[str] = None
    description: Optional[str] = None
    is_enabled: Optional[int] = None
    kb_type: Optional[str] = None


class KnowledgeBaseItem(BaseModel):
    id: int
    name: str
    collection_name: str
    owner: str
    description: str = ""
    kb_type: str = "doc"
    is_enabled: int
    doc_count: int = 0  # 文档数量
    faq_count: int = 0  # FAQ数量
    gmt_create: str
    gmt_modified: str


# ============================================================
# 文档配置模型（切分策略跟随文档）
# ============================================================

class DocumentConfig(BaseModel):
    """文档切分配置（每个文档可独立配置）。

    支持的配置项：
    - 切分策略：chunk_strategy（切片方案选择）
    - 切分参数：chunk_token_limit, chunk_token_overlap, chunk_size_table
    - 解析策略：parsing_strategy, extract_images, garbage_filter_enabled
    """
    # 切片策略选择
    chunk_strategy: Optional[str] = Field(
        default="smart",
        description="切片策略：smart（智能切分-默认）、heading（标题切分）、semantic（语义切分）、recursive（递归切分）、fixed（固定长度）",
    )

    # 切分配置
    chunk_token_limit: Optional[int] = Field(default=1024, ge=256, le=4096, description="每个chunk最大token数")
    chunk_token_overlap: Optional[int] = Field(default=100, ge=0, le=500, description="相邻chunk重叠token数")
    chunk_size_table: Optional[int] = Field(default=1600, ge=800, le=5000, description="表格切片字符上限")

    # 解析配置
    parsing_strategy: Optional[str] = Field(default="hi_res", description="解析策略 hi_res/fast")
    extract_images: Optional[bool] = Field(default=True, description="是否提取嵌入图片")
    garbage_filter_enabled: Optional[bool] = Field(default=True, description="是否过滤页眉页脚噪音")
    custom_footer_keywords: Optional[list[str]] = Field(default=None, description="自定义页脚关键词列表")


# ============================================================
# 文档管理模型
# ============================================================

class DocumentCreate(BaseModel):
    kb_id: int = Field(..., description="知识库ID")
    doc_name: str = Field(..., min_length=1, max_length=255)
    doc_type: str = Field(default="", description="文档类型 pdf/docx/pptx/md")
    file_size: Optional[int] = Field(default=None, description="文件大小")
    file_path: Optional[str] = Field(default=None, description="文件路径")
    source_type: str = Field(default="local_file", description="来源类型")
    source_url: Optional[str] = Field(default=None, description="URL来源")
    config: Optional[DocumentConfig] = Field(default=None, description="文档切分配置")
    creator: Optional[str] = Field(default=None, description="创建者")


class DocumentUpdate(BaseModel):
    doc_name: Optional[str] = None
    is_enabled: Optional[int] = None
    config: Optional[DocumentConfig] = None


class DocumentItem(BaseModel):
    id: int
    doc_id: str
    kb_id: int
    doc_name: str
    doc_type: str = ""
    file_size: int = 0
    source_type: str = "local_file"
    source_url: str = ""
    status: str = "pending"
    chunk_count: int = 0
    is_enabled: int = 1
    config: Optional[dict] = None
    error_message: Optional[str] = None
    creator: Optional[str] = None
    gmt_create: str
    gmt_modified: str


class DocumentListData(BaseModel):
    total: int
    page: int
    result: list[DocumentItem]


class DocumentListResponse(BaseResponse):
    data: Optional[DocumentListData] = None


class DocumentDetailResponse(BaseResponse):
    data: Optional[DocumentItem] = None


class KnowledgeBaseListData(BaseModel):
    total: int
    page: int
    result: list[KnowledgeBaseItem]


class KnowledgeBaseListResponse(BaseResponse):
    data: Optional[KnowledgeBaseListData] = None


class KnowledgeBaseDetailResponse(BaseResponse):
    data: Optional[KnowledgeBaseItem] = None


# ============================================================
# 文档管理模型（旧版，用于 documents router）
# ============================================================

# 注意：上面已定义完整的 DocumentItem，这里保留简化的用于列表展示
class DocumentListItem(BaseModel):
    """文档列表项（简化版，用于 /api/v1/documents 接口）"""
    doc_id: str
    doc_name: str
    doc_type: str = ""
    source_type: str = "local_file"
    source_url: str = ""
    file_size: int = 0
    chunk_count: int = 0
    status: str = "success"
    is_enabled: bool = True
    gmt_modified: str = ""


class DocumentListDataOld(BaseModel):
    total: int
    page: int
    result: list[DocumentListItem]


class DocumentListResponseOld(BaseResponse):
    data: Optional[DocumentListDataOld] = None


class DocumentToggleRequest(BaseModel):
    is_enabled: bool


# ============================================================
# 切片操作模型
# ============================================================

class ChunkUpdateRequest(BaseModel):
    content: str = Field(..., description="编辑后的切片内容")


class ChunkBatchToggleRequest(BaseModel):
    chunk_ids: list[str] = Field(..., min_length=1)
    is_enabled: bool


class ChunkBatchDeleteRequest(BaseModel):
    chunk_ids: list[str] = Field(..., min_length=1)


class DocumentBatchDeleteRequest(BaseModel):
    doc_ids: list[str] = Field(..., min_length=1, description="要删除的文档ID列表")


class DocumentBatchDeleteData(BaseModel):
    total: int
    success: int
    failed: int
    deleted_chunks: int
    results: list[dict] = Field(default_factory=list)  # 每个文档的删除结果


# ============================================================
# 意图树模型
# ============================================================

class IntentNodeCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    node_type: str = Field(..., description="DOMAIN|CATEGORY|TOPIC")
    parent_id: Optional[int] = None
    sort_order: int = 0
    collection_name: str = ""
    description: str = ""
    example_questions: list[str] = Field(default_factory=list)


class IntentNodeUpdate(BaseModel):
    name: Optional[str] = None
    node_type: Optional[str] = None
    parent_id: Optional[int] = None
    sort_order: Optional[int] = None
    collection_name: Optional[str] = None
    description: Optional[str] = None
    example_questions: Optional[list[str]] = None
    is_enabled: Optional[int] = None


class IntentNodeItem(BaseModel):
    id: int
    name: str
    node_type: str
    parent_id: Optional[int] = None
    sort_order: int
    collection_name: str = ""
    description: str = ""
    example_questions: list[str] = Field(default_factory=list)
    is_enabled: int
    gmt_create: str
    gmt_modified: str


class IntentTreeResponse(BaseResponse):
    data: Optional[list[IntentNodeItem]] = None


# ============================================================
# Dashboard 模型
# ============================================================

class DashboardStats(BaseModel):
    active_users: int
    sessions: int
    messages: int
    session_depth: float


class AiPerformance(BaseModel):
    success_rate: float
    avg_latency_ms: float
    p95_latency_ms: float


class QualityItem(BaseModel):
    date: str
    error_rate: float
    no_knowledge_rate: float
    slow_rate: float


class TrendItem(BaseModel):
    date: str
    count: int


class DashboardStatsResponse(BaseResponse):
    data: Optional[DashboardStats] = None


class AiPerformanceResponse(BaseResponse):
    data: Optional[AiPerformance] = None


class QualityResponse(BaseResponse):
    data: Optional[list[QualityItem]] = None


class TrendResponse(BaseResponse):
    data: Optional[list[TrendItem]] = None


# ============================================================
# 用户管理模型
# ============================================================

class UserCreate(BaseModel):
    username: str = Field(..., min_length=2, max_length=64)
    password: str = Field(..., min_length=6, max_length=128)
    role: str = "user"


class UserUpdate(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[int] = None


class UserItem(BaseModel):
    id: int
    username: str
    role: str
    is_active: int
    gmt_create: str
    gmt_modified: str


class UserListData(BaseModel):
    total: int
    page: int
    result: list[UserItem]


class UserListResponse(BaseResponse):
    data: Optional[UserListData] = None


# ============================================================
# 流水线详情模型
# ============================================================

class PipelineNode(BaseModel):
    name: str
    type: str
    status: str
    duration_ms: int = 0
    message: str = ""


class TaskDetailData(BaseModel):
    task_id: str
    doc_id: Optional[str] = None
    doc_name: str = ""
    doc_type: str = ""
    status: str
    progress: int = 0
    stage: Optional[str] = None
    error_message: Optional[str] = None
    chunk_count: int = 0
    pipeline_nodes: list[PipelineNode] = Field(default_factory=list)
    task_params: Optional[dict] = None
    gmt_create: str = ""
    gmt_modified: str = ""


class TaskDetailResponse(BaseResponse):
    data: Optional[TaskDetailData] = None


# ============================================================
# 评估记录模型
# ============================================================

class SaveEvaluationRequest(BaseModel):
    """保存评估记录请求。"""
    kb_id: str = Field(..., description="知识库ID")
    doc_id: str = Field(..., description="文档ID")
    set_id: str = Field(..., description="测试集批次ID")
    metrics: dict[str, Any] = Field(..., description="评估指标JSON")


class EvaluationRecordItem(BaseModel):
    """评估记录详情项。"""
    kb_name: str = ""
    doc_name: str = ""
    set_id: str
    metrics: dict[str, Any]
    gmt_create: str


class EvaluationRecordListData(BaseModel):
    """评估记录分页列表数据。"""
    total: int
    page: int
    result: list[EvaluationRecordItem]


class EvaluationRecordListResponse(BaseResponse):
    """评估记录分页列表响应。"""
    data: Optional[EvaluationRecordListData] = None


# ============================================================
# FAQ 模型
# ============================================================

class FAQCreate(BaseModel):
    """创建FAQ请求。"""
    kb_id: int = Field(..., description="知识库ID")
    question: str = Field(..., min_length=1, max_length=2000, description="问题文本")
    answer: str = Field(..., min_length=1, max_length=5000, description="答案文本")
    category: str = Field(default="", max_length=64, description="分类")
    keywords: str = Field(default="", max_length=256, description="关键词标签（逗号分隔）")
    source: str = Field(default="manual", description="来源 manual/import")
    sort_order: int = Field(default=0, description="排序权重")
    creator: Optional[str] = Field(default=None, description="创建者")


class FAQUpdate(BaseModel):
    """更新FAQ请求。"""
    question: Optional[str] = Field(default=None, min_length=1, max_length=2000)
    answer: Optional[str] = Field(default=None, min_length=1, max_length=5000)
    category: Optional[str] = Field(default=None, max_length=64)
    keywords: Optional[str] = Field(default=None, max_length=256)
    is_enabled: Optional[int] = Field(default=None)
    sort_order: Optional[int] = Field(default=None)


class FAQItem(BaseModel):
    """FAQ详情。"""
    id: int
    faq_id: str
    kb_id: int
    question: str
    answer: str
    category: str = ""
    keywords: str = ""
    source: str = "manual"
    hit_count: int = 0
    sort_order: int = 0
    is_enabled: int = 1
    creator: Optional[str] = None
    gmt_create: str
    gmt_modified: str


class FAQListData(BaseModel):
    """FAQ分页列表数据。"""
    total: int
    page: int
    result: list[FAQItem]


class FAQListResponse(BaseResponse):
    """FAQ分页列表响应。"""
    data: Optional[FAQListData] = None


class FAQDetailResponse(BaseResponse):
    """FAQ详情响应。"""
    data: Optional[FAQItem] = None


class FAQSearchRequest(BaseModel):
    """FAQ检索请求。"""
    kb_id: int = Field(..., description="知识库ID")
    query: str = Field(..., min_length=1, max_length=2000, description="查询文本")
    top_k: int = Field(default=10, ge=1, le=100, description="返回数量")
    category: str = Field(default="", description="分类过滤")
    use_rerank: bool = Field(default=True, description="是否启用重排序")


class FAQSearchHitItem(BaseModel):
    """FAQ检索结果项。"""
    faq_id: str
    kb_id: int
    question: str
    answer: str
    category: str = ""
    keywords: str = ""
    hit_count: int = 0
    score: float
    rerank_score: Optional[float] = None


class FAQSearchResponseData(BaseModel):
    """FAQ检索结果数据。"""
    hits: list[FAQSearchHitItem]


class FAQSearchResponse(BaseResponse):
    """FAQ检索响应。"""
    data: Optional[FAQSearchResponseData] = None


class FAQBatchImportItem(BaseModel):
    """批量导入单条FAQ。"""
    question: str = Field(..., min_length=1, max_length=2000)
    answer: str = Field(..., min_length=1, max_length=5000)
    category: str = Field(default="", max_length=64)
    keywords: str = Field(default="", max_length=256)


class FAQBatchImportRequest(BaseModel):
    """批量导入FAQ请求。"""
    kb_id: int = Field(..., description="知识库ID")
    items: list[FAQBatchImportItem] = Field(..., min_length=1, max_length=100, description="FAQ列表")
    creator: Optional[str] = Field(default=None, description="创建者")


class FAQBatchImportData(BaseModel):
    """批量导入结果数据。"""
    total: int
    success: int
    failed: int
    results: list[dict] = Field(default_factory=list)  # 每条导入结果


class FAQBatchImportResponse(BaseResponse):
    """批量导入响应。"""
    data: Optional[FAQBatchImportData] = None


class FAQBatchDeleteRequest(BaseModel):
    """批量删除FAQ请求。"""
    faq_ids: list[str] = Field(..., min_length=1, description="FAQ ID列表")


# ============================================================
# 审计日志模型
# ============================================================

class AuditLogItem(BaseModel):
    """审计日志项。"""
    id: int
    user_id: str
    user_role: str = ""
    path: str
    method: str
    status_code: int
    latency_ms: int
    params: Optional[dict] = None
    trace_id: str = ""
    gmt_create: str


class AuditLogListData(BaseModel):
    """审计日志分页列表数据。"""
    total: int
    page: int
    result: list[AuditLogItem]


class AuditLogListResponse(BaseResponse):
    """审计日志分页列表响应。"""
    data: Optional[AuditLogListData] = None


class DailyActiveItem(BaseModel):
    """日活统计项。"""
    date: str
    count: int


class DailyActiveTrendData(BaseModel):
    """日活趋势数据。"""
    trend: list[DailyActiveItem]
    total_users: int  # 总用户数


class DailyActiveTrendResponse(BaseResponse):
    """日活趋势响应。"""
    data: Optional[DailyActiveTrendData] = None
