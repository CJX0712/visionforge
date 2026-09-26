FROM python:3.13-slim

WORKDIR /app

# 系统依赖（skimage/opencv 需要的底层库）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 默认运行端到端演示
CMD ["python", "-m", "visionforge.examples.run_demo"]
