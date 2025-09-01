FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지 업데이트 및 필요한 도구 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY ./src ./src
COPY .env* ./

# 벡터 DB와 캐시를 위한 디렉토리 생성
RUN mkdir -p /app/db /app/cache

# 포트 노출 (Gradio 기본 포트)
EXPOSE 7860

# 환경변수 설정
ENV PYTHONPATH=/app
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# 실행 명령
CMD ["python", "src/resume/app.py"]
