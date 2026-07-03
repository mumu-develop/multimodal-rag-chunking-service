"""
LLM 客户端模块

提供纯文本 LLM 调用能力，用于：
- 为图片生成摘要（基于页面上下文）
- 为幻灯片生成摘要

使用 urllib.request 调用 OpenAI 兼容 API，无需额外依赖。
"""
import json

import structlog
import urllib.request

from app.config import settings

logger = structlog.get_logger()


class LLMClient:
    """
    纯文本 LLM 客户端，复用 urllib.request 调用 OpenAI 兼容 API。

    流程概述：
    1. 初始化时从配置读取 API URL 和超时时间
    2. 业务方法（summarize_image/summarize_slide）构造 prompt
    3. 调用 _call_llm 发送 HTTP POST 请求
    4. 解析响应返回结果
    """

    def __init__(self):
        """
        初始化 LLM 客户端。

        步骤：
        1. 从 settings.LLM_TIMEOUT 读取超时时间（秒）
        2. 从 settings.LLM_API_URL 读取 API 地址
        """
        # 步骤1：设置请求超时时间，防止长时间阻塞
        self._timeout = settings.LLM_TIMEOUT
        # 步骤2：设置 API 地址，通常为 OpenAI 兼容的本地服务
        self._api_url = settings.LLM_API_URL

    def summarize_image(self, page_text: str, heading_text: str = "") -> str:
        """
        根据页面上下文文本，为图片生成摘要。

        业务场景：
        - PDF/PPT 解析后，图片需要补充摘要说明
        - 使用图片所在页面的文本作为上下文，让 LLM 生成相关性更强的摘要

        流程步骤：
        1. 构造用户消息：如果有标题，拼接标题+上下文；否则只拼接上下文
        2. 定义 system prompt：限定输出格式（不超过200字，只输出摘要）
        3. 调用 _call_llm 发送请求
        4. 返回生成的摘要文本（失败返回空字符串）

        参数：
            page_text: 图片所在页面的全部文本内容
            heading_text: 页面标题（可选，用于增强摘要相关性）

        返回：
            图片摘要文本，不超过200字；失败返回空字符串
        """
        # 步骤1：构造用户消息
        # - 如果有标题，格式化为 "标题：xxx\n\n页面上下文：\nxxx"
        # - 这样 LLM 能理解标题与图片的关联
        if heading_text:
            user_msg = f"标题：{heading_text}\n\n页面上下文：\n{page_text}"
        else:
            user_msg = f"页面上下文：\n{page_text}"

        # 步骤2：定义 system prompt
        # - 角色定位：文档分析助手
        # - 任务：根据上下文为图片生成摘要
        # - 输出约束：不超过200字，只输出摘要，不输出其他内容
        system_prompt = (
            "你是一个文档分析助手。根据提供的页面上下文文本，"
            "为页面中的图片生成一段简短的摘要，不超过 200 字。"
            "只输出摘要文本，不要输出其他内容。"
        )

        # 步骤3：调用底层 LLM 请求方法
        # 步骤4：返回结果（失败时 _call_llm 返回空字符串）
        return self._call_llm(system_prompt, user_msg)

    def summarize_slide(self, slide_text: str, title: str = "") -> str:
        """
        根据幻灯片文本，生成摘要。

        业务场景：
        - PPT/PPTX 解析后，每页幻灯片需要生成摘要
        - 用于快速浏览或检索增强

        流程步骤：
        1. 构造用户消息：如果有标题，拼接标题+内容；否则只拼接内容
        2. 定义 system prompt：限定输出格式（不超过200字）
        3. 调用 _call_llm 发送请求
        4. 返回生成的摘要文本

        参数：
            slide_text: 幻灯片的全部文本内容
            title: 幻灯片标题（可选）

        返回：
            幻灯片摘要文本，不超过200字；失败返回空字符串
        """
        # 步骤1：构造用户消息
        if title:
            user_msg = f"标题：{title}\n\n幻灯片内容：\n{slide_text}"
        else:
            user_msg = f"幻灯片内容：\n{slide_text}"

        # 步骤2：定义 system prompt
        system_prompt = (
            "你是一个文档分析助手。根据提供的幻灯片内容文本，"
            "生成一段简短的内容摘要，不超过 200 字。"
            "只输出摘要文本，不要输出其他内容。"
        )

        # 步骤3-4：调用 LLM 并返回结果
        return self._call_llm(system_prompt, user_msg)

    def _call_llm(self, system_prompt: str, user_msg: str) -> str:
        """
        调用 LLM API 获取回复（内部方法）。

        流程步骤：
        1. 构造 OpenAI 兼容的 JSON 请求体
        2. 创建 urllib.request.Request 对象
        3. 发送 POST 请求到 API 地址
        4. 解析响应 JSON，提取 content 字段
        5. 返回结果（异常时记录日志并返回空字符串）

        参数：
            system_prompt: 系统提示词，定义 LLM 角色和行为约束
            user_msg: 用户消息，包含具体任务内容

        返回：
            LLM 生成的文本内容；失败返回空字符串
        """
        try:
            # 步骤1：构造 OpenAI 兼容的 JSON 请求体
            # - model: "default" 表示使用默认模型
            # - messages: 标准 OpenAI 格式，system + user
            # - max_tokens: 限制输出长度，防止过长
            payload = json.dumps({
                "model": "default",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                "max_tokens": 512,
            })

            # 步骤2：创建 HTTP Request 对象
            # - URL: 来自配置的 API 地址
            # - data: JSON 请求体编码为 UTF-8
            # - headers: Content-Type 为 application/json
            # - method: POST
            req = urllib.request.Request(
                self._api_url,
                data=payload.encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            # 步骤3：发送请求并读取响应
            # - urlopen 超时由 self._timeout 控制
            # - 响应体解码为 UTF-8 JSON
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))

            # 步骤4：解析响应 JSON
            # - OpenAI 格式：{"choices": [{"message": {"content": "..."}}]}
            # - 提取第一个 choice 的 content 字段
            # - strip() 去除前后空白
            result = ""
            if "choices" in body and body["choices"]:
                result = body["choices"][0].get("message", {}).get("content", "").strip()

            # 步骤5：返回结果
            return result

        except Exception:
            # 异常处理：记录完整异常日志，返回空字符串
            # - 不抛出异常，避免影响上层业务流程
            logger.exception("llm_call_failed")
            return ""


# 单例模式：全局共享一个 LLMClient 实例
# - 避免重复初始化
# - urllib.request 无连接池，单例无副作用
llm_client = LLMClient()