FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ git libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#RUN pip install --no-cache-dir vllm
COPY rag_deepseek.py .env .
#COPY wiki_vllm_rag.py demo.py ./

#ENV VLLM_CPU_KVCACHE_SPACE=20
#ENV VLLM_CPU_OMP_THREADS_BIND=0-7

#CMD ["python", "demo.py"]
