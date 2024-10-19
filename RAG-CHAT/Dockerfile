# 빌드 스테이지
FROM python:3.10-slim AS builder

# 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 Python 패키지 설치
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# PyTorch CPU 버전 설치 (경량화)
RUN pip install --user --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 실행 스테이지
FROM python:3.10-slim

# 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Ollama 설치
RUN curl -fsSL https://ollama.com/install.sh | sh && \
    ollama --version

# 작업 디렉토리 설정
WORKDIR /app

# 빌더 스테이지에서 설치한 Python 패키지 복사
COPY --from=builder /root/.local /root/.local

# 필요한 애플리케이션 파일만 복사
COPY . .

# 실행 권한 설정
RUN chmod +x /app/start.sh

# 환경 변수 설정
ENV PATH=/root/.local/bin:$PATH

# 포트 노출 (Chainlit의 기본 포트와 Ollama 포트)
EXPOSE 8000 11434

# 컨테이너 시작 시 실행할 명령
CMD ["/app/start.sh"]