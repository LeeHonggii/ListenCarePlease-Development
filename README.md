# ListenCarePlease - Development 🚧

> **개발 저장소** - 신규 기능 개발 및 실험적 변경사항

이 저장소는 ListenCarePlease 프로젝트의 독립적인 개발 공간입니다.
새로운 팀 구성으로 진행되는 별도 개발 저장소입니다.

---

## 🔄 현재 개발 중인 기능

### ✅ 완료된 작업

#### **Pyannote Audio 3.1 전환** (2024.12.22)
- Senko/NeMo → Pyannote Audio 3.1 완전 전환
  - `pyannote/speaker-diarization-3.1` 모델 적용
  - CUDA/CPU 디바이스 자동 감지
  - HuggingFace 토큰 기반 인증
- 모든 Senko/NeMo 레거시 코드 제거
  - `diarization_nemo.py` 완전 삭제
  - Pyannote 단일 파이프라인으로 통합

#### **화자 분리 알고리즘 고도화** (2024.12.22)
- **신뢰도 기반 화자 할당**
  - High Confidence: 겹침 비율 70% 이상 또는 거리 2초 이내
  - Medium Confidence: 겹침 비율 30-70% 또는 거리 2-5초
  - Low Confidence: 겹침 비율 30% 미만 또는 거리 5초 이상
- **Gap Filling 알고리즘 개선**
  - 거리 기반 신뢰도 계산 (2초/5초 임계값)
  - 가장 가까운 화자에게 자동 할당
  - STT 세그먼트 100% 보존 보장
- **UNKNOWN 화자 처리**
  - UI에서 UNKNOWN 화자 자동 제외
  - Backend 필터링으로 깔끔한 화자 목록 제공

#### **태깅 UI 개선** (2024.12.22)
- **"분리 & 수정" 통합 탭**
  - 기존 "분리"와 "자세히" 탭 통합
  - 화자 이름 일괄 편집 (그리드 레이아웃)
  - 세그먼트 개별 화자 재할당 (드롭다운)
  - 신뢰도 기반 색상 코딩 (녹색/주황/빨강)
- **정렬 품질 지표 시각화**
  - `alignment_score`: 0-100% 매칭 정확도
  - `unassigned_duration`: 미할당 세그먼트 시간
  - 화자별 통계 (세그먼트 수, 발화 시간)

#### **실제 화자 임베딩 추출** (2024.12.22)
- **Pyannote Inference 모델 통합**
  - `pyannote/embedding` 모델 사용
  - 더미 임베딩 (256차원) → 실제 임베딩 (512차원)
- **화자별 임베딩 추출**
  - 각 화자의 모든 발화 구간에서 임베딩 추출
  - 평균 임베딩 계산 + L2 정규화
  - 0.5초 이상 구간만 사용
- **화자 프로필 저장**
  - 실제 음성 임베딩을 화자 프로필에 저장
  - 향후 화자 자동 인식 기능 기반 마련

---

## 🚀 향후 개발 계획 (2024.12.22 - 2025.01.15)

> 피드백 기반 개선 사항 및 기능 추가 로드맵

### 📊 성능 측정 개선
- [ ] **화자 자동 인식 기능**
  - 저장된 화자 프로필 기반 자동 태깅
  - 임베딩 유사도 비교 (코사인 유사도)
  - 신규 화자 vs 기존 화자 판별
- [ ] **모듈별 성능 평가** 시스템 구축
  - NER, Agent, Diarization 개별 평가 메트릭
  - 단계별 성능 리포트 생성

### 🔍 효율성 지표 검증
- [ ] **검증 테스트 케이스** 추가
  - 다양한 회의 유형별 테스트 데이터
  - Ground truth 기반 정확도 검증
- [ ] **Perplexity 지표 설명** 추가
  - 사용자 대상 지표 해석 가이드
  - 값 범위별 의미 설명

### 🎨 UX/UI 개선
- [ ] **진행 상황 UI** 개선
  - 실시간 진행률 표시 강화
  - 각 단계별 소요 시간 표시
- [ ] **효율성 지표 설명** UI 추가
  - 툴팁 또는 모달로 지표 해석 제공
  - 시각적 가이드 추가
- [ ] **파일 업로드 가이드** 개선
  - 권장 포맷, 길이, 품질 안내
  - 예상 처리 시간 표시

### ✅ 기능 개선
- [ ] **To-do 파싱 및 추적** 기능 추가
  - 회의록에서 액션 아이템 자동 추출
  - 담당자 및 마감일 추적
- [ ] **사용자 Annotation** 기능
  - 결과 수정 및 피드백 수집
  - 학습 데이터로 활용
- [ ] **회의록 템플릿** 다양화
  - 업종/목적별 템플릿 추가
  - 사용자 커스텀 템플릿 지원
- [ ] **피드백 수집 메커니즘** 구축
  - 결과 평가 시스템 (별점, 코멘트)
  - 개선 우선순위 데이터 수집

### ⚡ 아키텍처 개선
- [ ] **비동기 처리 고도화**
  - Celery 기반 작업 큐 전환 검토
  - 병렬 처리 최적화
- [ ] **실시간 변환 스트리밍**
  - 진행 중 중간 결과 표시
  - WebSocket 기반 실시간 업데이트
- [ ] **배포 환경 구성**
  - Production 환경 설정
  - CI/CD 파이프라인 구축

---

## 🛠️ 개발 환경 설정

### 주요 기술 스택

#### Backend
- **Python 3.11** - 핵심 런타임
- **FastAPI** - REST API 프레임워크
- **SQLAlchemy** - ORM (MySQL)
- **Pyannote Audio 3.1** - 화자 분리 (speaker-diarization-3.1)
- **Whisper large-v3** - 음성 인식 (STT)
- **LangChain** - LLM Agent 프레임워크
- **OpenAI API** - NER, 태깅, 효율성 분석

#### Frontend
- **React 18** - UI 프레임워크
- **Vite** - 빌드 도구
- **Tailwind CSS** - 스타일링
- **Axios** - HTTP 클라이언트

#### Infrastructure
- **Docker & Docker Compose** - 컨테이너화
- **MySQL 8.0** - 데이터베이스
- **CUDA 12.4+** - GPU 가속

### 요구사항
- Python 3.11
- Docker & Docker Compose
- NVIDIA GPU (CUDA 12.4+) - 선택사항 (CPU 모드 지원)
- 12GB+ RAM (GPU 사용 시 16GB+ 권장)

### 빠른 시작
```bash
# 저장소 클론
git clone https://github.com/LeeHonggii/ListenCarePlease-Development.git
cd ListenCarePlease-Development

# Docker 빌드 및 실행
docker-compose up -d --build

# DB 마이그레이션
docker exec -it listencare_backend alembic upgrade head

# 접속
# Frontend: http://localhost:3000
# Backend API: http://localhost:18000/docs
```

---

## 📝 브랜치 전략

- `main`: Development 저장소의 메인 브랜치
- `honggi`: 현재 개발 진행 중인 브랜치
- feature 브랜치: 개별 기능 개발용

---

## 🔗 관련 링크

- **원본 프로젝트**: https://github.com/LeeHonggii/ListenCarePlease (코드베이스 출처)

---

## 👥 개발팀

**Lead Developer**: [@LeeHonggii](https://github.com/LeeHonggii)

> 본 저장소는 새로운 팀 구성으로 독립적으로 개발되고 있습니다.
>
> **참고**: 이 저장소는 기존 ListenCarePlease 프로젝트에서 코드베이스를 가져왔으나,
> 새로운 방향으로 독립 개발 중입니다.
