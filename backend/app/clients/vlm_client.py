"""
VLM 客户端模块

提供视觉语言模型（VLM）调用能力，用于：
- 为图片生成文本描述（caption）

使用 OpenAI 兼容的多模态 API，通过 base64 编码传输图片。
"""
import base64
from pathlib import Path

import structlog

from app.config import settings

logger = structlog.get_logger()


class VLMClient:
    """
    VLM 图片描述生成服务。

    调用 settings.VLM_API_URL 生成图片的文本 caption。
    超时 30s，失败返回空字符串。

    流程概述：
    1. 初始化时设置超时时间和系统提示词
    2. describe_image 方法读取图片文件
    3. 将图片转为 base64 编码
    4. 构造 OpenAI 兼容的多模态请求
    5. 发送请求并解析响应
    """

    def __init__(self):
        """
        初始化 VLM 客户端。

        步骤：
        1. 设置请求超时时间（30秒）
        2. 定义系统提示词，约束输出格式
        """
        # 步骤1：设置超时时间
        # - VLM 处理图片耗时较长，设为30秒
        self._timeout = 30
        # 步骤2：定义系统提示词
        # - 约束输出：不超过200字，只输出描述，不输出其他内容
        self._system_prompt = "请为这张图生成一段简短的描述，不超过200字。只输出描述文本，不要输出其他内容。"

    def describe_image(self, image_path: str) -> str:
        """
        调用 VLM 生成图片描述文本。

        业务场景：
        - PDF/PPT 解析后，提取的图片需要生成 caption
        - caption 用于：
          - 图片切片的 content 字段（存储到 Milvus）
          - 图片的语义检索（用户查询能匹配图片内容）

        流程步骤：
        1. 检查图片文件是否存在
        2. 读取图片文件，转为二进制字节
        3. 根据文件扩展名确定 MIME 类型
        4. 将图片字节编码为 base64 字符串
        5. 构造 OpenAI 兼容的多模态 JSON 请求体
        6. 创建 urllib.request.Request 对象
        7. 发送 POST 请求到 VLM API
        8. 解析响应 JSON，提取 caption
        9. 返回描述文本（失败返回空字符串）

        参数：
            image_path: 图片文件的绝对路径

        返回：
            图片描述文本（caption），不超过200字
            失败时返回空字符串
        """
        # 步骤1：检查图片文件是否存在
        # - 使用 Path 对象进行路径操作
        # - 文件不存在时记录警告日志并返回空字符串
        path = Path(image_path)
        if not path.is_file():
            logger.warning("vlm_image_not_found", path=image_path)
            return ""

        try:
            # 步骤2：读取图片文件为二进制字节
            # - open(path, "rb") 以二进制模式读取
            # - f.read() 返回 bytes 对象
            with open(path, "rb") as f:
                img_bytes = f.read()

            # 步骤3：确定 MIME 类型
            # - 获取文件扩展名（如 .png, .jpg）
            # - 不支持的扩展名默认为 .jpg
            # - mime_map 映射扩展名到 MIME 类型字符串
            ext = path.suffix.lower()
            if ext not in (".png", ".jpg", ".jpeg", ".webp"):
                ext = ".jpg"
            mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
            mime_type = mime_map.get(ext, "image/jpeg")

            # 步骤4：编码图片为 base64
            # - base64.b64encode 将 bytes 转为 base64 bytes
            # - decode("utf-8") 转为字符串，用于 JSON 序列化
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")

            # 步骤5：构造 OpenAI 兼容的多模态 JSON 请求体
            # - model: "default" 使用默认模型
            # - messages[0]: system 提示词
            # - messages[1]: user 消息，包含图片和文本
            #   - type: "image_url" 表示图片内容
            #   - image_url.url: data URI 格式，包含 MIME 类型和 base64 数据
            #   - type: "text" 表示文本提示
            # - max_tokens: 限制输出长度
            import json

            payload = json.dumps({
                "model": "default",
                "messages": [
                    {"role": "system", "content": self._system_prompt},
                    {
                        "role": "user",
                        "content": [
                            # 图片内容：data URI 格式
                            # 格式：data:{mime_type};base64,{base64_data}
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{img_base64}"},
                            },
                            # 文本提示：要求描述图片
                            {"type": "text", "text": "请描述这张图片的内容。"},
                        ],
                    },
                ],
                "max_tokens": 512,
            })

            # 步骤6：创建 HTTP Request 对象
            # - URL: settings.VLM_API_URL
            # - data: JSON 请求体编码为 UTF-8
            # - headers: Content-Type 为 application/json
            # - method: POST
            import urllib.request

            req = urllib.request.Request(
                settings.VLM_API_URL,
                data=payload.encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            # 步骤7：发送请求并读取响应
            # - urlopen 发送请求，超时30秒
            # - 响应体解码为 UTF-8 JSON
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))

            # 步骤8：解析响应 JSON
            # - OpenAI 格式：{"choices": [{"message": {"content": "..."}}]}
            # - 提取第一个 choice 的 content 字段
            # - strip() 去除前后空白
            caption = ""
            if "choices" in body and body["choices"]:
                caption = body["choices"][0].get("message", {}).get("content", "").strip()

            # 步骤9：记录成功日志并返回结果
            # - 日志包含路径和生成文本长度，便于调试
            logger.info("vlm_caption_generated", path=image_path, caption_len=len(caption))
            return caption

        except Exception:
            # 异常处理：记录完整异常日志，返回空字符串
            # - 不抛出异常，避免影响上层业务流程（如切片处理）
            logger.exception("vlm_caption_failed", path=image_path)
            return ""


# 单例模式：全局共享一个 VLMClient 实例
# - 避免重复初始化系统提示词
# - urllib.request 无连接池，单例无副作用
vlm_client = VLMClient()