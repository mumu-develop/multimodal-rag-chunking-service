"""
PyCharm Debug 启动入口。

Run/Debug Configurations:
  - Script path: backend/run_debug.py
  - Python interpreter: 选择你的 conda 环境 (multimodal-rag-chunking-service)
  - Working directory: backend/
  - 勾选 "PyCharm debug"
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8888,
        reload=False,   # debug 模式建议关闭 reload
        log_level="debug",
    )
