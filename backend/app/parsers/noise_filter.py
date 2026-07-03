"""文档解析噪音过滤工具（增强版）。

功能：
1. 规则匹配：正则表达式过滤页码、分隔线等
2. 关键词过滤：内置 + 自定义关键词
3. 跨页重复检测：自动识别页眉页脚
4. 布局分析：基于位置判断（需要位置信息）
5. 敏感信息检测：手机号、邮箱等
"""
import re
from typing import Optional


# ============================================================
# 内置关键词库
# ============================================================

# 页脚噪音关键词（英文）
FOOTER_KEYWORDS_EN = {
    "confidential", "draft", "internal", "proprietary",
    "copyright", "reserved", "all rights reserved",
    "page", "page of", "printed", "printed on",
    "version", "revision", "rev", "v",
    "document id", "doc id", "file id",
    "prepared by", "created by", "written by",
    "approved by", "checked by",
    "company name", "inc", "ltd", "corp", "corporation",
}

# 页脚噪音关键词（中文）
FOOTER_KEYWORDS_CN = {
    "机密", "绝密", "秘密", "内部资料", "内部文件",
    "保密", "严禁传播", "禁止外传", "不得转载",
    "版权所有", "著作权", "保留所有权利",
    "第页", "第 页", "页码", "共页", "共 页",
    "打印", "打印日期", "打印时间",
    "版本", "修订", "稿",
    "编制", "编写", "编写人", "编制人", "作者",
    "审核", "审核人", "批准", "批准人",
    "公司", "有限公司", "有限责任公司",
    "技术部", "研发部", "产品部", "运营部",
    "未经授权", "禁止复制", "仅供内部使用",
    "仅供参考", "草案", "草稿",
}

# 水印关键词
WATERMARK_KEYWORDS = {
    "confidential", "draft", "do not distribute", "internal use only",
    "内部资料", "机密", "绝密", "秘密", "仅供参考",
    "样本", "范例", "模板", "样稿",
}

# 合并所有内置关键词
BUILTIN_FOOTER_KEYWORDS = FOOTER_KEYWORDS_EN | FOOTER_KEYWORDS_CN
BUILTIN_WATERMARK_KEYWORDS = WATERMARK_KEYWORDS


# ============================================================
# 垃圾模式（正则表达式）
# ============================================================

GARBAGE_PATTERNS = [
    # 页码格式
    r"^•+$",                          # 纯符号行
    r"^[0-9]{1,2}\s*/\s*[0-9]{1,2}$", # 页码 "1 / 10", "1/10"
    r"^[0-9]{1,2}\s+of\s+[0-9]{1,2}$",# 页码 "1 of 10"
    r"^page\s+[0-9]+$",               # "Page 1", "page 2"
    r"^p\.\s*[0-9]+$",                # "p. 1"
    r"^第\s*[0-9]+\s*页$",            # "第 1 页"
    r"^第\s*[0-9]+\s*/\s*[0-9]+\s*页$", # "第 1/10 页"
    r"^\d{1,4}$",                     # 纯数字行（可能是页码）

    # PDF损坏标记
    r"\(cid\s*:\s*[0-9]+\s*\)",       # PDF 损坏标记 "(cid:123)"

    # URL/网址
    r"^http://[^ ]{12,}",             # 长 URL
    r"^www\.[^ ]{10,}",               # www 开头的网址
    r"^https://[^ ]{12,}",            # HTTPS URL

    # 分隔线
    r"^[_\-–—]{3,}$",                 # 下划线/横线分隔线
    r"^={3,}$",                       # 等号分隔线
    r"^\*{3,}$",                      # 星号分隔线
    r"^~{3,}$",                       # 波浪线分隔线
    r"^#{3,}$",                       # 井号分隔线
    r"^\.{3,}$",                      # 点号分隔线

    # 日期格式（可能是页脚）
    r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$", # "2024-01-15", "2024/1/15"
    r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}$", # "01-15-2024", "1/15/2024"

    # 时间格式
    r"^\d{1,2}:\d{2}(:\d{2})?$",      # "12:30", "12:30:45"

    # 文档编号格式
    r"^[A-Z]{2,4}[-_]\d{4,}[-_]\d+$", # "DOC-2024-001", "REV_12345_01"
    r"^No\.\s*\d+$",                  # "No. 123"
    r"^编号\s*[:：]?\s*\d+$",         # "编号: 123"

    # 常见页眉页脚格式
    r"^[A-Z]{2,3}\s+\d{4}$",          # "AB 2024"（可能是部门缩写+年份）
    r"^\d+\s*[页页]$",                # "123 页"
]

_GARBAGE_PATTERNS_COMPILED = [re.compile(p, re.IGNORECASE) for p in GARBAGE_PATTERNS]


# ============================================================
# 敏感信息模式
# ============================================================

SENSITIVE_PATTERNS = {
    # 手机号
    "phone": re.compile(r"1[3-9]\d{9}"),
    # 邮箱
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    # 身份证号
    "id_card": re.compile(r"\d{17}[\dXx]"),
    # 银行卡号
    "bank_card": re.compile(r"\d{16,19}"),
}


# ============================================================
# 全局配置（可通过 API 动态修改）
# ============================================================

# 自定义关键词（用户可添加）
_custom_footer_keywords: set[str] = set()
_custom_watermark_keywords: set[str] = set()

# 是否启用各功能
_enable_sensitive_detection = False
_enable_cross_page_detection = True
_enable_layout_analysis = False


def add_custom_keywords(keywords: list[str], keyword_type: str = "footer") -> None:
    """添加自定义关键词。

    Args:
        keywords: 关键词列表
        keyword_type: "footer" 或 "watermark"
    """
    global _custom_footer_keywords, _custom_watermark_keywords

    if keyword_type == "footer":
        _custom_footer_keywords.update(k.lower() for k in keywords)
    elif keyword_type == "watermark":
        _custom_watermark_keywords.update(k.lower() for k in keywords)


def clear_custom_keywords() -> None:
    """清空自定义关键词。"""
    global _custom_footer_keywords, _custom_watermark_keywords
    _custom_footer_keywords.clear()
    _custom_watermark_keywords.clear()


def get_all_keywords() -> dict[str, set[str]]:
    """获取所有关键词（内置 + 自定义）。"""
    return {
        "footer": BUILTIN_FOOTER_KEYWORDS | _custom_footer_keywords,
        "watermark": BUILTIN_WATERMARK_KEYWORDS | _custom_watermark_keywords,
    }


def enable_sensitive_detection(enable: bool = True) -> None:
    """启用/禁用敏感信息检测。"""
    global _enable_sensitive_detection
    _enable_sensitive_detection = enable


def enable_cross_page_detection(enable: bool = True) -> None:
    """启用/禁用跨页重复检测。"""
    global _enable_cross_page_detection
    _enable_cross_page_detection = enable


# ============================================================
# 基础过滤函数
# ============================================================

def is_footer_text(text: str) -> bool:
    """判断文本是否为页脚/水印/页码噪音。

    匹配规则：
    1. 纯页码格式
    2. 关键词匹配（内置 + 自定义）
    """
    lower = text.lower().strip()

    # 纯页码：单个数字或 "x / y" 格式
    if re.match(r"^\d+$", lower) and len(lower) <= 5:
        return True
    if re.match(r"^\d+\s*[/–-]\s*\d+$", lower):
        return True

    # 获取所有关键词
    all_keywords = get_all_keywords()

    # 水印关键词匹配（精确匹配）
    if lower in all_keywords["watermark"]:
        return True

    # 页脚关键词匹配（前缀匹配）
    for keyword in all_keywords["footer"]:
        if lower == keyword or lower.startswith(keyword) or keyword in lower:
            return True

    return False


def clean_text(text: str) -> str:
    """清理文本中的噪音字符。

    处理：
    1. 多余空格/制表符
    2. 不可见控制字符
    3. 多余换行符
    """
    # 去除多余空格/制表符
    text = re.sub(r"[\t ]+", " ", text)
    # 去除不可见控制字符（保留换行）
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    # 去除多余换行（连续超过2个换行变为2个）
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_noise_text(text: str, min_length: int = 3) -> bool:
    """判断文本是否为应过滤的噪音。

    Args:
        text: 待检测文本
        min_length: 最小有效长度

    Returns:
        True 表示是噪音，False 表示有效文本
    """
    s = text.strip()

    if not s:
        return True

    # 过短文本
    if len(s) < min_length:
        return True

    # 页脚/页码/水印
    if is_footer_text(s):
        return True

    # 纯符号行
    if re.match(r"^[^\w\u4e00-\u9fff]+$", s):
        return True

    # 匹配垃圾模式
    for pattern in _GARBAGE_PATTERNS_COMPILED:
        if pattern.search(s):
            return True

    return False


# ============================================================
# 增强功能：跨页重复检测
# ============================================================

class CrossPageDetector:
    """跨页重复内容检测器。

    用于识别每页都出现的相同内容（可能是页眉页脚）。

    使用方式：
    1. 创建检测器实例
    2. 对每页内容调用 add_page_content()
    3. 调用 get_repeated_texts() 获取重复文本
    4. 在后续处理中过滤这些重复文本
    """

    def __init__(self, threshold: float = 0.5, min_pages: int = 3):
        """初始化检测器。

        Args:
            threshold: 出现频率阈值（>= 此比例视为重复）
            min_pages: 最少出现的页数
        """
        self.threshold = threshold
        self.min_pages = min_pages
        self._text_counts: dict[str, int] = {}
        self._total_pages: int = 0
        self._text_positions: dict[str, list[tuple[int, float]]] = {}  # text -> [(page_idx, y_position)]

    def add_page_content(self, page_idx: int, texts: list[str], y_positions: list[float] = None) -> None:
        """添加一页的内容。

        Args:
            page_idx: 页面索引
            texts: 该页所有文本元素
            y_positions: 各文本的 Y 坐标（可选，用于布局分析）
        """
        self._total_pages += 1

        for i, text in enumerate(texts):
            normalized = text.strip().lower()
            if len(normalized) < 3:  # 忽略过短文本
                continue

            self._text_counts[normalized] = self._text_counts.get(normalized, 0) + 1

            # 记录位置信息
            if y_positions and i < len(y_positions):
                if normalized not in self._text_positions:
                    self._text_positions[normalized] = []
                self._text_positions[normalized].append((page_idx, y_positions[i]))

    def get_repeated_texts(self) -> list[tuple[str, str]]:
        """获取判定为页眉页脚的重复文本。

        Returns:
            list of (original_text, reason):
            - original_text: 原始文本（用于过滤）
            - reason: 判定原因 ("frequency" 或 "position")
        """
        if self._total_pages < self.min_pages:
            return []

        repeated = []
        threshold_count = int(self._total_pages * self.threshold)

        for text, count in self._text_counts.items():
            if count >= threshold_count:
                # 基于频率判定
                reason = "frequency"

                # 如果有位置信息，检查是否都在顶部或底部
                if text in self._text_positions:
                    positions = self._text_positions[text]
                    # 检查是否都在页面的前10%或后10%区域
                    # (这里假设 y_position 是 0-1 的相对位置)
                    top_count = sum(1 for _, y in positions if y < 0.15)
                    bottom_count = sum(1 for _, y in positions if y > 0.85)

                    if top_count >= threshold_count:
                        reason = "header_position"
                    elif bottom_count >= threshold_count:
                        reason = "footer_position"

                repeated.append((text, reason))

        return repeated

    def is_repeated_text(self, text: str) -> bool:
        """判断给定文本是否为跨页重复内容。

        Args:
            text: 待检测文本

        Returns:
            True 如果是重复内容
        """
        normalized = text.strip().lower()
        for repeated_text, _ in self.get_repeated_texts():
            if normalized == repeated_text or normalized.startswith(repeated_text):
                return True
        return False

    def clear(self) -> None:
        """清空检测器状态。"""
        self._text_counts.clear()
        self._text_positions.clear()
        self._total_pages = 0


# ============================================================
# 增强功能：敏感信息检测与脱敏
# ============================================================

def detect_sensitive_info(text: str) -> dict[str, list[str]]:
    """检测文本中的敏感信息。

    Returns:
        dict: {"phone": [...], "email": [...], "id_card": [...]}
    """
    if not _enable_sensitive_detection:
        return {}

    result = {}
    for type_name, pattern in SENSITIVE_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            result[type_name] = matches

    return result


def mask_sensitive_info(text: str) -> str:
    """脱敏处理敏感信息。

    处理规则：
    - 手机号：保留前3后4，中间用*
    - 邮箱：保留前3字符和域名，中间用*
    - 身份证：保留前6后4，中间用*
    - 银行卡：保留前4后4，中间用*
    """
    if not _enable_sensitive_detection:
        return text

    # 手机号脱敏
    text = SENSITIVE_PATTERNS["phone"].sub(
        lambda m: m.group(0)[:3] + "****" + m.group(0)[-4:],
        text
    )

    # 邮箱脱敏
    text = re.sub(
        r"([a-zA-Z0-9]{3})[a-zA-Z0-9._%+-]*@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
        lambda m: m.group(1) + "***@" + m.group(2),
        text
    )

    # 身份证脱敏
    text = SENSITIVE_PATTERNS["id_card"].sub(
        lambda m: m.group(0)[:6] + "********" + m.group(0)[-4:],
        text
    )

    # 银行卡脱敏
    text = SENSITIVE_PATTERNS["bank_card"].sub(
        lambda m: m.group(0)[:4] + "****" + m.group(0)[-4:],
        text
    )

    return text


# ============================================================
# 综合过滤函数
# ============================================================

def should_filter_text(
    text: str,
    cross_page_detector: Optional[CrossPageDetector] = None,
    min_length: int = 3,
) -> tuple[bool, str]:
    """综合判断是否应过滤该文本。

    Args:
        text: 待检测文本
        cross_page_detector: 跨页检测器实例（可选）
        min_length: 最小有效长度

    Returns:
        (should_filter, reason):
        - should_filter: True 表示应过滤
        - reason: 过滤原因
    """
    s = text.strip()

    if not s:
        return True, "empty"

    if len(s) < min_length:
        return True, "too_short"

    # 基础噪音检测
    if is_noise_text(s):
        return True, "noise_pattern"

    # 跨页重复检测
    if cross_page_detector and _enable_cross_page_detection:
        if cross_page_detector.is_repeated_text(s):
            return True, "cross_page_repeat"

    return False, "valid"


def clean_and_validate_text(
    text: str,
    cross_page_detector: Optional[CrossPageDetector] = None,
    mask_sensitive: bool = False,
) -> tuple[str, bool]:
    """清洗并验证文本。

    Args:
        text: 原始文本
        cross_page_detector: 跨页检测器
        mask_sensitive: 是否脱敏敏感信息

    Returns:
        (cleaned_text, is_valid):
        - cleaned_text: 清洗后的文本
        - is_valid: True 表示是有效内容
    """
    # 先清洗
    cleaned = clean_text(text)

    # 脱敏
    if mask_sensitive:
        cleaned = mask_sensitive_info(cleaned)

    # 验证
    should_filter, reason = should_filter_text(cleaned, cross_page_detector)

    return cleaned, not should_filter


def filter_page_text(
    text: str,
    cross_page_detector: Optional[CrossPageDetector] = None,
    custom_keywords: list[str] = None,
) -> str:
    """过滤页面文本中的噪音内容。

    Args:
        text: 原始页面文本
        cross_page_detector: 跨页检测器实例
        custom_keywords: 自定义噪音关键词列表

    Returns:
        过滤后的文本
    """
    # 添加自定义关键词
    if custom_keywords:
        add_custom_keywords(custom_keywords)

    # 按行处理
    lines = text.split("\n")
    filtered_lines = []

    for line in lines:
        cleaned_line = clean_text(line)
        should_filter, reason = should_filter_text(
            cleaned_line,
            cross_page_detector=cross_page_detector,
            min_length=3,
        )
        if not should_filter:
            filtered_lines.append(cleaned_line)

    return "\n".join(filtered_lines)


# 导出常用模式供外部使用
PHONE_PATTERN = SENSITIVE_PATTERNS.get("phone")
EMAIL_PATTERN = SENSITIVE_PATTERNS.get("email")
HEADER_KEYWORDS_CN = set()  # 页眉关键词（中文）
FOOTER_KEYWORDS_CN = FOOTER_KEYWORDS_CN  # 重命名以保持一致