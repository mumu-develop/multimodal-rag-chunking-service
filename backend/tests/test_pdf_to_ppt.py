# 安装: pip install pdf2slides
from pdf2slides import Converter
# 对扫描件启用OCR
converter = Converter(enable_ocr=True, lang='chi_sim')  # 指定中文语言包
converter.convert('/Users/linmiaomiao/Downloads/机器学习.pdf', '机器学习.pptx')

