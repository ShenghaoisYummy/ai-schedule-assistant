FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 确保模型目录存在
RUN mkdir -p models/intent_classifier models/ner/mobilebert-ner logs

# 暴露端口
EXPOSE 5000

# 使用gunicorn运行应用
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]