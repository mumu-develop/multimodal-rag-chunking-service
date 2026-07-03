# RAG 项目改进计划

## 目标
1. 打造一个支持多种切割策略的完善 RAG 项目
2. 确保 RAGAS 指标（召回率、准确率、相关性、忠诚度）达到 95% 以上
3. 成为开源的完善 RAG 项目，解决用户的 RAG 清洗、切片需求

---

## 一、现有架构分析

### 已使用的设计模式
| 模式 | 应用位置 | 评价 |
|:---|:---|:---|
| Strategy Pattern | ChunkStrategy ABC | ✅ 良好 |
| Registry Pattern | _PARSER_CLASSES | ✅ 良好 |
| Singleton Pattern | 各 service/repo | ⚠️ 需改进（改为依赖注入） |
| Repository Pattern | chunk_repo, milvus_repo | ✅ 良好 |
| Factory Pattern | select_strategy() | ⚠️ 需扩展 |

### 现有问题
1. **扩展性不足**：添加新切分策略需要修改多处代码
2. **配置分散**：settings + kb_config + doc_config，缺乏统一管理
3. **缺乏 Pipeline 抽象**：处理流程硬编码在 tasks.py
4. **测试覆盖不足**：缺少对切分策略的单元测试
5. **缺乏指标追踪**：无法持续监控 RAGAS 指标

---

## 二、设计模式改进方案

### 1. 策略工厂模式（Strategy Factory）

**改进目标**：支持动态注册切分策略，无需修改核心代码

```python
# app/chunkers/registry.py
class ChunkerRegistry:
    """切分策略注册中心"""

    _strategies: dict[str, Type[ChunkStrategy]] = {}

    @classmethod
    def register(cls, name: str, strategy_cls: Type[ChunkStrategy]):
        """注册策略"""
        cls._strategies[name] = strategy_cls

    @classmethod
    def get(cls, name: str) -> ChunkStrategy:
        """获取策略实例"""
        return cls._strategies[name]()

    @classmethod
    def list(cls) -> list[str]:
        """列出所有策略"""
        return list(cls._strategies.keys())

# 使用示例
ChunkerRegistry.register("hierarchical", HierarchicalChunker)
ChunkerRegistry.register("slide", SlideChunker)
ChunkerRegistry.register("semantic", SemanticChunker)  # 新策略
ChunkerRegistry.register("recursive", RecursiveChunker)  # 新策略
```

### 2. Pipeline 模式（处理流程抽象）

**改进目标**：将文档处理流程抽象为可配置的 Pipeline

```python
# app/pipeline/base.py
class PipelineStep(ABC):
    """Pipeline 步骤抽象"""

    @abstractmethod
    def process(self, input: Any) -> Any:
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__

class DocumentPipeline:
    """文档处理 Pipeline"""

    def __init__(self, steps: list[PipelineStep]):
        self.steps = steps

    def run(self, input: Any) -> Any:
        for step in self.steps:
            input = step.process(input)
        return input

# 默认 Pipeline
DEFAULT_PIPELINE = DocumentPipeline([
    ParseStep(),
    CleanStep(),
    ChunkStep(),
    EmbedStep(),
    StoreStep(),
])
```

### 3. 插件架构（Plugin Architecture）

**改进目标**：支持第三方扩展，无需修改核心代码

```python
# app/plugins/base.py
class RAGPlugin(ABC):
    """RAG 插件基类"""

    @property
    def name(self) -> str:
        ...

    @abstractmethod
    def on_parse(self, pages: list[ParsedPage]) -> list[ParsedPage]:
        """解析后处理钩子"""
        ...

    @abstractmethod
    def on_chunk(self, chunks: list[dict]) -> list[dict]:
        """切分后处理钩子"""
        ...

class PluginManager:
    """插件管理器"""

    _plugins: list[RAGPlugin] = []

    def load(self, plugin: RAGPlugin):
        self._plugins.append(plugin)

    def apply_on_parse(self, pages):
        for plugin in self._plugins:
            pages = plugin.on_parse(pages)
        return pages
```

### 4. 配置管理改进（配置继承）

**改进目标**：统一配置管理，支持多级配置继承

```python
# app/config_manager.py
class ConfigManager:
    """配置管理器，支持继承链"""

    def __init__(self):
        self._global_config = {}      # 全局配置
        self._kb_configs = {}         # 知识库配置
        self._doc_configs = {}        # 文档配置

    def get_effective_config(self, doc_id: str, kb_id: str) -> dict:
        """获取有效配置（合并继承链）"""
        config = self._global_config.copy()
        config.update(self._kb_configs.get(kb_id, {}))
        config.update(self._doc_configs.get(doc_id, {}))
        return config
```

---

## 三、切分策略增强

### 新增切分策略

| 策略 | 适用场景 | 核心特点 |
|:---|:---|:---|
| SemanticChunker | 通用文档 | 语义边界检测，句子嵌入聚类 |
| RecursiveChunker | 代码/结构化文档 | 递归拆分，保持层级 |
| SlidingWindowChunker | 需要上下文连续性 | 滑动窗口，固定重叠 |
| TokenChunker | 简单场景 | 纯 token 计数切分 |
| MarkdownOutlineChunker | Markdown 文档 | 按大纲层级切分 |

### SemanticChunker 实现思路

```python
class SemanticChunker(ChunkStrategy):
    """语义切分器：基于句子嵌入相似度检测边界"""

    def chunk(self, pages, doc_id, ...):
        # 1. 将文本拆分为句子
        sentences = self._split_sentences(pages)

        # 2. 计算句子嵌入
        embeddings = [embedder.encode(s) for s in sentences]

        # 3. 计算相邻句子相似度
        similarities = [cosine_sim(e[i], e[i+1]) for i in range(len(e)-1)]

        # 4. 检测边界（相似度低于阈值的位置）
        boundaries = [i for i, s in enumerate(similarities) if s < threshold]

        # 5. 按边界切分
        chunks = self._split_at_boundaries(sentences, boundaries)

        return chunks
```

---

## 四、RAGAS 指标优化策略

### 目标指标

| 指标 | 目标值 | 关键优化点 |
|:---|:---|:---|
| Context Precision | ≥ 0.95 | 重排序优化、相关性阈值调整 |
| Context Recall | ≥ 0.95 | 切分粒度、标题继承、图片 caption |
| Faithfulness | ≥ 0.95 | Prompt 设计、答案生成质量控制 |
| Answer Relevance | ≥ 0.95 | Query 扩展、多模态理解 |

### Context Recall 提升策略

```markdown
1. **标题路径继承**：每个 chunk prepend 完整标题路径
   - 已实现 ✅

2. **表格 Caption 增强**：表格添加语义化 caption
   - 使用 LLM 生成表格摘要

3. **图片 VLM Caption**：图片使用 VLM 生成描述性 caption
   - 已实现 ✅，需优化描述质量

4. **语义边界切分**：避免在语义不连贯处切分
   - 新增 SemanticChunker

5. **上下文重叠**：相邻 chunk 保持 token 级重叠
   - 已实现 ✅
```

### Context Precision 提升策略

```markdown
1. **重排序优化**：
   - 使用更强的 rerank 模型（如 BGE-Reranker-v2）
   - 增加 RERANK_TOP_N（50 → 100）

2. **Query 扩展**：
   - 对用户 query 进行语义扩展
   - 添加同义词、相关词

3. **混合检索权重调整**：
   - 调整 dense vs BM25 的权重
   - 根据 query 类型动态调整

4. **负样本过滤**：
   - 过滤明显不相关的结果（score < threshold）
```

---

## 五、清洗规则增强

### 增强项

| 类别 | 新增规则 | 示例 |
|:---|:---|:---|
| 页眉页脚 | 公司 Logo、联系方式 | "XX科技有限公司"、"地址：..." |
| 水印 | 自定义水印关键词 | 用户可配置 |
| 敏感信息 | 脱敏处理 | 手机号、邮箱、身份证 |
| 重复内容 | 跨页检测 | 每页都出现的相同文本 |
| 格式噪音 | PDF 损坏标记 | (cid:123)、乱码 |
| 语言检测 | 过滤非目标语言 | 可配置目标语言 |

### LLM 辅助清洗

```python
class LLMCleaner:
    """使用 LLM 进行智能清洗"""

    def clean(self, text: str) -> str:
        prompt = f"""
        判断以下文本是否为文档噪音（页眉、页脚、水印、页码等）：
        文本：{text}

        如果是噪音，返回 "SKIP"
        如果是有效内容，返回清洗后的文本
        """
        result = llm.invoke(prompt)
        return result if result != "SKIP" else ""
```

---

## 六、开源化准备

### 文档完善

1. **README.md**：项目介绍、安装指南、快速开始
2. **API 文档**：Swagger/OpenAPI 规范
3. **架构文档**：设计说明、模块职责
4. **贡献指南**：CONTRIBUTING.md
5. **变更日志**：CHANGELOG.md

### 配置示例

```yaml
# config.example.yaml
chunking:
  strategies:
    - hierarchical
    - semantic
    - recursive

  default_strategy: hierarchical

  hierarchical:
    token_limit: 1024
    token_overlap: 100
    preserve_headings: true

  semantic:
    similarity_threshold: 0.5
    min_chunk_tokens: 100

cleaning:
  enabled: true
  custom_keywords:
    - "公司内部资料"
    - "技术部专用"

  sensitive_info:
    mask_phone: true
    mask_email: true
    mask_id_card: true

evaluation:
  auto_run: false
  target_metrics:
    context_precision: 0.95
    context_recall: 0.95
```

### Docker 化

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY app/ ./app/
COPY alembic/ ./alembic/

# 运行
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 七、实施路线图

### Phase 1：架构重构（✅ 已完成）
- [x] ChunkerRegistry 实现
- [x] Plugin 模式实现（PluginManager + RAGPlugin）
- [x] 配置管理改进（kb_config 继承）

### Phase 2：策略增强（✅ 已完成）
- [x] SemanticChunker 实现
- [x] RecursiveChunker 实现
- [x] MetricsTracker 指标追踪
- [ ] 策略参数自动调优（消融实验）

### Phase 3：清洗增强（✅ 已完成）
- [x] 扩展关键词库（50+ 中文关键词）
- [x] 跨页重复检测（CrossPageDetector）
- [x] 敏感信息脱敏
- [ ] LLM 辅助清洗

### Phase 4：指标优化（进行中）
- [x] RAGAS 持续追踪（MetricsTracker）
- [ ] 重排序优化
- [ ] 参数自动调优

### Phase 5：开源化（1-2 周）
- [ ] 文档完善
- [ ] 配置示例
- [ ] Docker 化
- [ ] 测试覆盖

---

## 八、参考资料

### 开源 RAG 项目参考

| 项目 | 特点 | 可借鉴点 |
|:---|:---|:---|
| RAGFlow | 智能切分、清洗规则 | 噪音过滤、BULLET_PATTERN |
| LangChain | 模块化、插件架构 | Chain/Runnable 设计模式 |
| LlamaIndex | 索引抽象、存储抽象 | VectorStoreIndex 设计 |
| Unstructured | 文档解析、清洗 | 分区函数设计 |
| haystack | Pipeline 设计 | Pipeline/Node 架构 |

### 设计模式参考

- **Strategy Pattern**: 允许运行时切换算法
- **Factory Pattern**: 创建对象实例
- **Pipeline Pattern**: 链式处理流程
- **Plugin Pattern**: 第三方扩展能力
- **Observer Pattern**: 事件通知机制