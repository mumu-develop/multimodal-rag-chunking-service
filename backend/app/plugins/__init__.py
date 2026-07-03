"""RAG 插件系统。

提供可扩展的插件架构，支持在文档处理的各个阶段介入。

内置插件：
- ContentCleanPlugin: 内容清洗
- HeadingPathPlugin: 标题路径增强
- TableCaptionPlugin: 表格 Caption 增强

使用示例：
    from app.plugins import plugin_manager, init_builtin_plugins

    # 初始化内置插件
    init_builtin_plugins()

    # 加载自定义插件
    class MyPlugin(RAGPlugin):
        def on_chunk(self, chunks):
            # 自定义处理
            return chunks

    plugin_manager.load(MyPlugin())

    # 应用钩子
    chunks = plugin_manager.apply_on_chunk(chunks)
"""
from app.plugins.base import (
    RAGPlugin,
    PluginManager,
    plugin_manager,
    ContentCleanPlugin,
    HeadingPathPlugin,
    TableCaptionPlugin,
    init_builtin_plugins,
)

__all__ = [
    "RAGPlugin",
    "PluginManager",
    "plugin_manager",
    "ContentCleanPlugin",
    "HeadingPathPlugin",
    "TableCaptionPlugin",
    "init_builtin_plugins",
]