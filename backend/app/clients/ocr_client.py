"""
OCR 客户端模块

使用 OpenAI 兼容 API 调用 OCR 服务（如 DeepSeek-OCR）。
支持单图识别和多图并发识别。
"""
import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import structlog

from app.config import settings

logger = structlog.get_logger()


class OCRClient:
    """
    OCR 服务客户端，使用 OpenAI 兼容 API 格式。

    流程概述：
    1. 将图片转为 base64 编码
    2. 构造 OpenAI chat.completions 请求体
    3. 发送请求到 OCR API
    4. 返回 Markdown 格式识别结果

    支持并发处理多页 PDF 图片。
    """

    def __init__(self):
        """
        初始化 OCR 客户端。

        从 settings 读取：
        - OCR_API_URL: OCR 服务地址（OpenAI 兼容格式）
        - OCR_TIMEOUT: 请求超时时间
        - OCR_MAX_WORKERS: 并发线程数
        - OCR_MODEL_NAME: OCR 模型名称
        """
        self._api_url = settings.OCR_API_URL
        self._timeout = settings.OCR_TIMEOUT
        self._max_workers = settings.OCR_MAX_WORKERS
        self._model_name = settings.OCR_MODEL_NAME
        self._max_tokens = settings.OCR_MAX_TOKENS

    def recognize_image(self, image_path: str) -> str:
        """
        识别单张图片，返回 Markdown 格式文本。

        参数：
            image_path: 图片文件路径

        返回：
            Markdown 格式的识别结果
        """
        # 读取图片并转为 base64
        if not os.path.exists(image_path):
            logger.error("ocr_image_not_found", path=image_path)
            return ""

        with open(image_path, "rb") as f:
            image_bytes = f.read()
        img_base64 = base64.b64encode(image_bytes).decode("utf-8")

        return self._call_ocr_api(img_base64)

    def recognize_image_base64(self, img_base64: str) -> str:
        """
        识别 base64 编码的图片。

        参数：
            img_base64: base64 编码的图片数据

        返回：
            Markdown 格式的识别结果
        """
        return self._call_ocr_api(img_base64)

    def recognize_images_parallel(
        self,
        images: list[str],
        desc: str = "OCR识别",
    ) -> list[str]:
        """
        并发识别多张图片。

        参数：
            images: 图片 base64 编码列表
            desc: 进度描述

        返回：
            Markdown 结果列表（保持原始顺序）
        """
        total = len(images)
        results = [""] * total

        # 检查 API 是否可用
        if not self._api_url or self._api_url == "http://localhost:8010/ocr":
            # 未配置 OCR 服务，返回 mock 数据
            logger.warning("ocr_service_not_configured", url=self._api_url)
            for idx in range(total):
                results[idx] = f"\n--- PAGE {idx + 1} ---\n\n[OCR 服务未配置]\n\n"
            return results

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_idx = {
                executor.submit(self.recognize_image_base64, img): idx
                for idx, img in enumerate(images)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error("ocr_page_failed", page=idx + 1, error=str(e))
                    results[idx] = f"\n--- PAGE {idx + 1} ---\n\n[OCR 失败: {e}]\n\n"

        logger.info("ocr_parallel_complete", total=total, success=sum(1 for r in results if r and "[OCR 失败]" not in r))
        return results

    def _call_ocr_api(self, img_base64: str) -> str:
        """
        调用 OCR API（OpenAI 兼容格式）。

        参数：
            img_base64: base64 编码的图片

        返回：
            Markdown 格式识别结果
        """
        # 检查是否配置了 OCR 服务
        if not self._api_url or self._api_url == "http://localhost:8010/ocr":
            raise NotImplementedError("OCR service not configured. Set OCR_API_URL in .env")

        try:
            # 使用 OpenAI SDK（已安装 langchain-openai）
            from openai import OpenAI

            client = OpenAI(api_key="EMPTY", base_url=self._api_url)

            response = client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                            },
                            {
                                "type": "text",
                                "text": "Convert the document to markdown."
                            }
                        ]
                    }
                ],
                max_tokens=self._max_tokens,
                temperature=0.1,
            )

            result = response.choices[0].message.content
            return result.strip() if result else ""

        except ImportError:
            # OpenAI SDK 未安装，使用 urllib.request
            return self._call_ocr_api_urllib(img_base64)
        except Exception as e:
            logger.exception("ocr_call_failed", error=str(e))
            raise

    def _call_ocr_api_urllib(self, img_base64: str) -> str:
        """
        使用 urllib.request 调用 OCR API（备用方案）。

        参数：
            img_base64: base64 编码的图片

        返回：
            Markdown 格式识别结果
        """
        import urllib.request

        payload = json.dumps({
            "model": self._model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                        },
                        {
                            "type": "text",
                            "text": "Convert the document to markdown."
                        }
                    ]
                }
            ],
            "max_tokens": self._max_tokens,
            "temperature": 0.1,
        })

        req = urllib.request.Request(
            self._api_url,
            data=payload.encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        if "choices" in body and body["choices"]:
            return body["choices"][0].get("message", {}).get("content", "").strip()
        return ""


# 单例模式：全局共享一个 OCRClient 实例
ocr_client = OCRClient()