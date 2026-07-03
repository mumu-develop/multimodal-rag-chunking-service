# 执行计划：RAG智能体管理后台改造

> 基于 PRD v1.0，对现有 multimodal-rag-chunking-service 进行改造。
> 状态标记：⬜ 待开始 / 🔄 进行中 / ✅ 已完成 / ❌ 已取消

---

## 阶段一：数据模型层（数据库 + ORM）

### 1.1 新增表
- [x] ✅ **knowledge_bases** — 知识库表（id, name, embedding_model, collection_name, owner, description, is_enabled, gmt_create, gmt_modified）
- [x] ✅ **intent_tree_nodes** — 意图树节点表（id, name, node_type, parent_id, sort_order, collection_name, description, example_questions, is_enabled, gmt_create, gmt_modified）
- [x] ✅ **sessions** — 会话记录表（id, user_id, session_id, message_count, gmt_create）
- [x] ✅ **messages** — 消息记录表（id, session_id, query, response, latency_ms, is_success, error_msg, gmt_create）
- [x] ✅ **users** — 用户表（id, username, password_hash, role, is_active, gmt_create）

### 1.2 改造表
- [x] ✅ **doc_chunks** — 增加 kb_id 字段，关联知识库
- [x] ✅ **doc_chunks** — 增加 is_enabled 字段（默认 true）
- [x] ✅ **doc_chunks** — 增加 source_type 字段（local_file/url）
- [x] ✅ **doc_chunks** — 增加 source_url 字段
- [x] ✅ **doc_chunks** — 增加 file_size 字段
- [x] ✅ **tasks** — 增加 pipeline_nodes 字段（JSON，存储节点级执行详情）
- [x] ✅ **tasks** — 增加 task_params 字段（JSON，存储任务参数快照）
- [x] ✅ **tasks** — 增加 doc_name 字段

### 1.3 数据库迁移
- [x] ✅ 创建 Alembic 迁移脚本 `006_add_admin_tables.py`
- [ ] ⬜ 执行 `alembic upgrade head`（需数据库连接）

---

## 阶段二：数据仓库层（Repository）

- [x] ✅ **KnowledgeBaseRepo** — 知识库 CRUD
- [x] ✅ **IntentTreeRepo** — 意图树节点 CRUD + 树形查询
- [x] ✅ **SessionRepo** — 会话/消息写入、统计查询
- [x] ✅ **UserRepo** — 用户 CRUD
- [x] ✅ **改造 ChunkRepo** — 增加编辑/删除/批量操作/重建向量
- [x] ✅ **改造 TaskRepo** — 增加流水线节点详情写入

---

## 阶段三：API 路由层

### 3.1 知识库管理 API
- [x] ✅ `POST /api/v1/knowledge-bases` — 创建知识库
- [x] ✅ `GET /api/v1/knowledge-bases` — 列表查询（分页/搜索/排序）
- [x] ✅ `GET /api/v1/knowledge-bases/{kb_id}` — 详情
- [x] ✅ `PUT /api/v1/knowledge-bases/{kb_id}` — 编辑
- [x] ✅ `DELETE /api/v1/knowledge-bases/{kb_id}` — 删除

### 3.2 文档管理 API（改造）
- [x] ✅ `GET /api/v1/documents` — 文档列表（按 kb_id 过滤/分页/搜索）
- [x] ✅ `POST /api/v1/upload` — 改造：增加 kb_id, source_type, source_url 参数
- [x] ✅ `POST /api/v1/upload/url` — URL 文档下载并处理
- [x] ✅ `DELETE /api/v1/documents/{doc_id}` — 删除文档及所有分块
- [x] ✅ `POST /api/v1/documents/{doc_id}/reprocess` — 重新处理文档
- [x] ✅ `POST /api/v1/documents/{doc_id}/rebuild-vectors` — 重建向量
- [x] ✅ `PUT /api/v1/documents/{doc_id}/toggle` — 启用/禁用

### 3.3 切片管理 API（改造）
- [x] ✅ `PUT /api/v1/chunks/{chunk_id}` — 编辑切片内容
- [x] ✅ `DELETE /api/v1/chunks/{chunk_id}` — 删除切片
- [x] ✅ `PUT /api/v1/chunks/{chunk_id}/toggle` — 启用/禁用
- [x] ✅ `POST /api/v1/chunks/batch-toggle` — 批量启用/禁用
- [x] ✅ `POST /api/v1/chunks/batch-delete` — 批量删除

### 3.4 意图树 API（新增）
- [x] ✅ `GET /api/v1/intent-tree` — 获取整棵树
- [x] ✅ `POST /api/v1/intent-tree/nodes` — 创建节点
- [x] ✅ `PUT /api/v1/intent-tree/nodes/{node_id}` — 编辑节点
- [x] ✅ `DELETE /api/v1/intent-tree/nodes/{node_id}` — 删除节点
- [x] ✅ `PUT /api/v1/intent-tree/nodes/{node_id}/toggle` — 启用/禁用

### 3.5 Dashboard API（新增）
- [x] ✅ `GET /api/v1/dashboard/stats` — 核心指标（活跃用户/会话数/消息数/会话深度）
- [x] ✅ `GET /api/v1/dashboard/ai-performance` — AI 性能（成功率/平均响应/P95）
- [x] ✅ `GET /api/v1/dashboard/quality` — 质量快照（错误率/无知识率/慢响应率）
- [x] ✅ `GET /api/v1/dashboard/trends` — 趋势分析（会话趋势/用户趋势）

### 3.6 流水线详情 API（改造）
- [x] ✅ `GET /api/v1/tasks/{task_id}/detail` — 增强版任务详情（含节点级信息 + 任务参数）

### 3.7 用户管理 API（新增）
- [x] ✅ `GET /api/v1/users` — 用户列表
- [x] ✅ `POST /api/v1/users` — 创建用户
- [x] ✅ `PUT /api/v1/users/{user_id}` — 编辑用户
- [x] ✅ `DELETE /api/v1/users/{user_id}` — 删除用户

---

## 阶段四：中间件层

- [x] ✅ **RBAC 中间件** — `app/api/middleware/rbac.py`，管理接口 POST/PUT/DELETE 仅 admin 可操作
- [x] ✅ **分析打点中间件** — `app/api/middleware/analytics.py`，请求耗时统计 + search 路径自动采集
- [x] ✅ **改造 auth.py** — 修复 uuid import，增加 user_role/user_id 注入，支持 AUTH_EXEMPT_PREFIXES

---

## 阶段五：异步任务增强

- [x] ✅ **文档重处理任务** — `reprocess_document_task`（删除旧切片 → 重新解析流水线）
- [x] ✅ **向量重建任务** — `rebuild_vectors_task`（批量重新向量化）
- [x] ✅ **流水线节点级状态上报** — `_record_pipeline_node()` 在每个阶段完成时记录节点状态
- [x] ✅ **URL 源文档抓取** — `fetch_url_document_task`（从 URL 下载文档走标准流水线）

---

## 阶段六：Schemas 层

- [x] ✅ KnowledgeBase 相关 schema（create/update/response/list）
- [x] ✅ IntentTreeNode 相关 schema
- [x] ✅ Dashboard 相关 schema（stats/performance/quality/trends）
- [x] ✅ User 相关 schema
- [x] ✅ Chunk 编辑/批量操作 schema
- [x] ✅ UploadResponse 增加新字段
- [x] ✅ TaskDetailData 增加 pipeline_nodes

---
