FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY samples ./samples

EXPOSE 8000 8001
# 默认启动灰度偏移保持区间页面（docker compose 中各服务用 command 覆盖端口）
CMD ["uvicorn", "app.web:app", "--host", "0.0.0.0", "--port", "8000"]
