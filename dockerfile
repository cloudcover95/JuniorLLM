# jcllc-mem-sys/Dockerfile
FROM python:3.11-slim

WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY jcllc_mem_sys /workspace/jcllc_mem_sys
RUN pip install --no-cache-dir -e ".[playground]"

EXPOSE 8080
CMD ["jcllc-mem-sys", "wake-up", "--host", "0.0.0.0", "--port", "8080"]