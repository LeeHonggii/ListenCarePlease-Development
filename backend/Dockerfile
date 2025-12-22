FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

WORKDIR /app

# Python 출력 버퍼링 비활성화 (로그 즉시 출력)
ENV PYTHONUNBUFFERED=1

# Python 3.11 설치
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    cmake \
    ninja-build \
    autoconf \
    automake \
    libtool \
    default-libmysqlclient-dev \
    pkg-config \
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    libfreetype6-dev \
    libjpeg-dev \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# CUDA 아키텍처 명시적 지정 (9.0a 에러 방지)
ENV TORCH_CUDA_ARCH_LIST="5.0;6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"

# Python 의존성 설치 (통합 GPU 전용)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Mecab + mecab-ko-dic 설치 (한국어 형태소 분석)
RUN echo "Installing Mecab and mecab-ko-dic..." && \
    # Mecab 소스 다운로드 및 설치
    cd /tmp && \
    wget https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz && \
    tar xvf mecab-0.996-ko-0.9.2.tar.gz && \
    cd mecab-0.996-ko-0.9.2 && \
    ./configure && \
    make && \
    make install && \
    ldconfig && \
    # mecab-ko-dic 다운로드 및 설치
    cd /tmp && \
    wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz && \
    tar xvf mecab-ko-dic-2.1.1-20180720.tar.gz && \
    cd mecab-ko-dic-2.1.1-20180720 && \
    ./autogen.sh && \
    ./configure && \
    make && \
    make install && \
    # mecab-python3 설치
    pip install --no-cache-dir mecab-python3 && \
    # 임시 파일 정리
    rm -rf /tmp/mecab* && \
    echo "Mecab installation complete"

# Pyannote 모델 사전 다운로드는 런타임에 수행
# (HuggingFace 토큰이 환경변수로 필요)

# CUDA 라이브러리 경로 설정 (조건부)
RUN echo "Checking for CUDA libraries..." && \
    SITE_PACKAGES="/usr/local/lib/python3.11/site-packages" && \
    if [ -d "$SITE_PACKAGES" ]; then \
        # libnvrtc 찾기 (builtins 제외)
        NVRTC_PATH=$(find $SITE_PACKAGES -name "libnvrtc*.so.1*" ! -name "*builtins*" -print -quit 2>/dev/null || echo "") && \
        if [ -n "$NVRTC_PATH" ]; then \
            echo "Found NVRTC at $NVRTC_PATH"; \
            ln -sf "$NVRTC_PATH" /usr/lib/libnvrtc.so; \
            ln -sf "$NVRTC_PATH" /usr/lib/libnvrtc.so.11.2; \
            ln -sf "$NVRTC_PATH" /usr/lib/libnvrtc.so.12; \
        else \
            echo "NVRTC not found (may not be needed for Pyannote)"; \
        fi && \
        # builtins 찾기
        BUILTIN_PATH=$(find $SITE_PACKAGES -name "libnvrtc-builtins.so*" -print -quit 2>/dev/null || echo "") && \
        if [ -n "$BUILTIN_PATH" ]; then \
            echo "Found Builtins at $BUILTIN_PATH"; \
            ln -sf "$BUILTIN_PATH" /usr/lib/libnvrtc-builtins.so; \
            BUILTIN_NAME=$(basename "$BUILTIN_PATH"); \
            ln -sf "$BUILTIN_PATH" "/usr/lib/$BUILTIN_NAME"; \
        else \
            echo "NVRTC builtins not found (may not be needed for Pyannote)"; \
        fi; \
    else \
        echo "Site-packages directory not found yet, skipping CUDA symlinks"; \
    fi

# Add NVIDIA library paths to LD_LIBRARY_PATH (조건부)
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH"

# Whisper 모델 미리 다운로드 (large-v3, 약 3GB)
# 다운로드만 하고 로드는 하지 않음 (메모리 절약)
RUN python3 -c "import whisper; import os; os.makedirs('/root/.cache/whisper', exist_ok=True); whisper._download(whisper._MODELS['large-v3'], '/root/.cache/whisper', False)"

# Sentence Transformers 모델 미리 다운로드 (효율성 분석용)
# 모델을 캐시에 다운로드하여 런타임 시 즉시 사용 가능
RUN echo "Downloading sentence-transformers model..." && \
    python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')" && \
    echo "Sentence transformers model downloaded successfully"

# GPT-2 모델 미리 다운로드 (Perplexity 계산용)
RUN echo "Downloading GPT-2 model for perplexity calculation..." && \
    python3 -c "from transformers import GPT2LMHeadModel, GPT2TokenizerFast; GPT2TokenizerFast.from_pretrained('skt/kogpt2-base-v2'); GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')" && \
    echo "GPT-2 model downloaded successfully"

# 애플리케이션 코드 복사
COPY ./app ./app

# Alembic 설정 및 마이그레이션 파일 복사
COPY ./alembic ./alembic
COPY ./alembic.ini ./alembic.ini

# Entrypoint 스크립트 복사 및 실행 권한 부여
COPY ./entrypoint.sh ./entrypoint.sh
RUN sed -i 's/\r$//' ./entrypoint.sh && chmod +x ./entrypoint.sh

# 템플릿 생성 스크립트 복사 및 실행
# 템플릿 디렉토리 생성
RUN mkdir -p /app/templates

# 업로드 디렉토리 및 모델 캐시 디렉토리 생성
RUN mkdir -p /app/uploads /app/temp /app/.cache

# 포트 노출
EXPOSE 8000

# 서버 실행 (entrypoint 스크립트 사용 - 마이그레이션 자동 적용)
CMD ["./entrypoint.sh"]
