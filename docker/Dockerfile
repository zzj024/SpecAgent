# SpecAgent 后端镜像：python slim + 项目安装
# 构建：docker build -f docker/Dockerfile -t spec-agent-api .
# 注意：BGE-M3 / bge-reranker 权重（约 4GB）不打进镜像——
# 首次运行由 sentence-transformers 下载到 HF_HOME 卷（挂载即复用），
# 镜像本体保持轻量，权重与代码解耦升级互不干扰。
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY backend/ backend/
COPY corpus/ corpus/
COPY web/ web/
COPY mcp_server/ mcp_server/

# torch CPU 版：服务端推理不需要 CUDA，CPU 轮子体积是 GPU 版的 1/6
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir .[dev]

ENV HF_HUB_OFFLINE=0 \
    SPECAGENT_DSN=postgresql://specagent:specagent@pg:5432/specagent \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
