"""RAG 插件系统。

功能：
1. 定义插件基类和标准钩子
2. 提供插件管理器
3. 支持第三方扩展
4. 热插拔能力

使用示例：
    # 自定义插件
    class MyPlugin(RAGPlugin):
        def on_chunk(self, chunks):
            # 对 chunk 进行处理
            for c in chunks:
                c["content"] = c["content"].strip()
            return chunks

    # 注册插件
    plugin_manager.load(MyPlugin())

    # 处理时自动调用
    chunks = plugin_manager.apply_on_chunk(chunks)
"""
from abc import ABC, abstractmethod
from typing import Any, Optional

import structlog

from app.parsers.base import ParsedPage

logger = structlog.get_logger()


class RAGPlugin(ABC):
    """RAG 插件基类。

    插件可以在文档处理的各个阶段介入，进行自定义处理。

    钩子（Hooks）：
    - on_parse: 解析后处理
    - on_clean: 清洗后处理
    - on_chunk: 切分后处理
    - on_embed: 嵌入前处理
    - on_store: 存储前处理

    Attributes:
        name: 插件名称（唯一标识）
        priority: 优先级（数字越小优先级越高）
        enabled: 是否启用
    """

    name: str = "base_plugin"
    priority: int = 100
    enabled: bool = True

    @property
    def description(self) -> str:
        """插件描述。"""
        return ""

    def on_parse(self, pages: list[ParsedPage]) -> list[ParsedPage]:
        """解析后处理钩子。

        Args:
            pages: 解析后的页面列表

        Returns:
            处理后的页面列表
        """
        return pages

    def on_clean(self, pages: list[ParsedPage]) -> list[ParsedPage]:
        """清洗后处理钩子。

        Args:
            pages: 清洗后的页面列表

        Returns:
            处理后的页面列表
        """
        return pages

    def on_chunk(self, chunks: list[dict]) -> list[dict]:
        """切分后处理钩子。

        Args:
            chunks: 切分后的 chunk 列表

        Returns:
            处理后的 chunk 列表
        """
        return chunks

    def on_embed(self, chunks: list[dict]) -> list[dict]:
        """嵌入前处理钩子。

        Args:
            chunks: 待嵌入的 chunk 列表

        Returns:
            处理后的 chunk 列表
        """
        return chunks

    def on_store(self, entities: list[dict]) -> list[dict]:
        """存储前处理钩子。

        Args:
            entities: 待存储的实体列表（Milvus 格式）

        Returns:
            处理后的实体列表
        """
        return entities


class PluginManager:
    """插件管理器。

    功能：
    1. 加载/卸载插件
    2. 按优先级排序
    3. 应用钩子

    Attributes:
        _plugins: 已加载的插件列表
    """

    _plugins: list[RAGPlugin] = []

    @classmethod
    def load(cls, plugin: RAGPlugin) -> None:
        """加载插件。

        Args:
            plugin: 插件实例
        """
        if plugin.name in [p.name for p in cls._plugins]:
            logger.warning("plugin_already_loaded", name=plugin.name)
            return

        cls._plugins.append(plugin)
        # 按优先级排序
        cls._plugins.sort(key=lambda p: p.priority)

        logger.info("plugin_loaded", name=plugin.name, priority=plugin.priority)

    @classmethod
    def unload(cls, name: str) -> bool:
        """卸载插件。

        Args:
            name: 插件名称

        Returns:
            True 如果成功卸载
        """
        for i, p in enumerate(cls._plugins):
            if p.name == name:
                cls._plugins.pop(i)
                logger.info("plugin_unloaded", name=name)
                return True
        return False

    @classmethod
    def get(cls, name: str) -> Optional[RAGPlugin]:
        """获取插件实例。

        Args:
            name: 插件名称

        Returns:
            插件实例或 None
        """
        for p in cls._plugins:
            if p.name == name:
                return p
        return None

    @classmethod
    def list(cls) -> list[dict]:
        """列出所有已加载插件。

        Returns:
            插件信息列表 [{"name": "...", "description": "...", "enabled": bool}, ...]
        """
        return [
            {
                "name": p.name,
                "description": p.description,
                "priority": p.priority,
                "enabled": p.enabled,
            }
            for p in cls._plugins
        ]

    @classmethod
    def enable(cls, name: str) -> bool:
        """启用插件。"""
        plugin = cls.get(name)
        if plugin:
            plugin.enabled = True
            logger.info("plugin_enabled", name=name)
            return True
        return False

    @classmethod
    def disable(cls, name: str) -> bool:
        """禁用插件。"""
        plugin = cls.get(name)
        if plugin:
            plugin.enabled = False
            logger.info("plugin_disabled", name=name)
            return True
        return False

    @classmethod
    def apply_on_parse(cls, pages: list[ParsedPage]) -> list[ParsedPage]:
        """应用解析后钩子。"""
        for plugin in cls._plugins:
            if plugin.enabled:
                try:
                    pages = plugin.on_parse(pages)
                except Exception as e:
                    logger.error("plugin_hook_failed", name=plugin.name, hook="on_parse", error=str(e))
        return pages

    @classmethod
    def apply_on_clean(cls, pages: list[ParsedPage]) -> list[ParsedPage]:
        """应用清洗后钩子。"""
        for plugin in cls._plugins:
            if plugin.enabled:
                try:
                    pages = plugin.on_clean(pages)
                except Exception as e:
                    logger.error("plugin_hook_failed", name=plugin.name, hook="on_clean", error=str(e))
        return pages

    @classmethod
    def apply_on_chunk(cls, chunks: list[dict]) -> list[dict]:
        """应用切分后钩子。"""
        for plugin in cls._plugins:
            if plugin.enabled:
                try:
                    chunks = plugin.on_chunk(chunks)
                except Exception as e:
                    logger.error("plugin_hook_failed", name=plugin.name, hook="on_chunk", error=str(e))
        return chunks

    @classmethod
    def apply_on_embed(cls, chunks: list[dict]) -> list[dict]:
        """应用嵌入前钩子。"""
        for plugin in cls._plugins:
            if plugin.enabled:
                try:
                    chunks = plugin.on_embed(chunks)
                except Exception as e:
                    logger.error("plugin_hook_failed", name=plugin.name, hook="on_embed", error=str(e))
        return chunks

    @classmethod
    def apply_on_store(cls, entities: list[dict]) -> list[dict]:
        """应用存储前钩子。"""
        for plugin in cls._plugins:
            if plugin.enabled:
                try:
                    entities = plugin.on_store(entities)
                except Exception as e:
                    logger.error("plugin_hook_failed", name=plugin.name, hook="on_store", error=str(e))
        return entities

    @classmethod
    def clear(cls) -> None:
        """清空所有插件。"""
        cls._plugins.clear()
        logger.info("plugins_cleared")


# ============================================================
# 内置插件示例
# ============================================================

class ContentCleanPlugin(RAGPlugin):
    """内容清洗插件。

    功能：
    1. 去除多余空白
    2. 去除 HTML 标签残留
    3. 统一换行符
    """

    name = "content_clean"
    priority = 10
    enabled = True

    @property
    def description(self) -> str:
        return "内容清洗：去除多余空白、HTML残留、统一换行符"

    def on_chunk(self, chunks: list[dict]) -> list[dict]:
        """清洗 chunk 内容。"""
        import re

        for c in chunks:
            content = c.get("content", "")
            if not content:
                continue

            # 去除多余空白
            content = re.sub(r"\s+", " ", content)

            # 去除 HTML 残留
            content = re.sub(r"<[^>]+>", "", content)

            # 去除特殊字符
            content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", content)

            c["content"] = content.strip()

        return chunks


class HeadingPathPlugin(RAGPlugin):
    """标题路径增强插件。

    功能：
    1. 为每个 chunk prepend 标题路径
    2. 保持上下文连贯性
    """

    name = "heading_path"
    priority = 20
    enabled = True

    @property
    def description(self) -> str:
        return "标题路径增强：prepend 标题路径保持上下文"

    def on_chunk(self, chunks: list[dict]) -> list[dict]:
        """为 chunk prepend 标题路径。"""
        for c in chunks:
            heading_text = c.get("heading_text", "")
            content = c.get("content", "")

            if heading_text and content:
                # prepend 标题路径
                c["content"] = f"[{heading_text}]\n{content}"

        return chunks


class TableCaptionPlugin(RAGPlugin):
    """表格 Caption 增强插件。

    功能：
    1. 为表格添加 LLM 生成的 caption
    2. 提升表格的可检索性
    """

    name = "table_caption"
    priority = 30
    enabled = False  # 默认禁用（需要 LLM）

    @property
    def description(self) -> str:
        return "表格 Caption 增强：使用 LLM 生成表格描述"

    def on_chunk(self, chunks: list[dict]) -> list[dict]:
        """为表格添加 caption。"""
        # 仅处理表格类型
        table_chunks = [c for c in chunks if c.get("chunk_type") == "table"]

        if not table_chunks:
            return chunks

        # 使用 LLM 生成 caption（可选）
        # 这里仅作为示例，实际实现需要配置 LLM
        for c in table_chunks:
            # 从 metadata 获取 caption（如果已有）
            caption = c.get("metadata", {}).get("caption", "")
            if caption:
                c["content"] = f"{caption}\n{c['content']}"

        return chunks


# 初始化内置插件
def init_builtin_plugins():
    """初始化内置插件。"""
    PluginManager.load(ContentCleanPlugin())
    PluginManager.load(HeadingPathPlugin())
    # TableCaptionPlugin 默认禁用，需用户手动启用


# 便捷导出
plugin_manager = PluginManager