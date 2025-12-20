# ListenCarePlease - Development 🚧

> **개발 저장소** - 신규 기능 개발 및 실험적 변경사항

이 저장소는 ListenCarePlease 프로젝트의 독립적인 개발 공간입니다.
새로운 팀 구성으로 진행되는 별도 개발 저장소입니다.

---

## 🔄 현재 개발 중인 기능

### ✅ 완료된 작업
- **STT-Diarization 정렬 품질 지표** (2024.12.17)
  - `alignment_score`: 0-100% 정렬 품질 점수
  - `unassigned_duration`: 할당되지 않은 오디오 길이
  - 원형 게이지 UI로 품질 시각화
  - DB 마이그레이션 포함 (`e1f2g3h4i5j6`)

- **화자 분리 알고리즘 개선**
  - Gap Filling: 1초 임계값으로 인접 화자 할당
  - STT 세그먼트 100% 보존
  - UNKNOWN 화자 처리 개선

---

## 🚀 향후 개발 계획 (2025.12.17 - 12.31)

> 피드백 기반 개선 사항 및 기능 추가 로드맵

### 📊 성능 측정 개선
- [ ] **STT-Diarization 정렬 정확도** 측정 및 분석
  - 화자 전환 지점 정렬 오류율 계산
  - 타임스탬프 불일치 개선
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
- [ ] **화자 분리 모델 통합**
  - Senko + NeMo → 단일 모델로 통합
  - Dockerfile 최적화

---

## 🛠️ 개발 환경 설정

### 요구사항
- Python 3.11
- Docker & Docker Compose
- NVIDIA GPU (CUDA 12.4+)
- 12GB+ RAM

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
- **이슈 트래킹**: [Issues](https://github.com/LeeHonggii/ListenCarePlease-Development/issues)
- **프로젝트 보드**: [Projects](https://github.com/LeeHonggii/ListenCarePlease-Development/projects)

---

## 👥 개발팀

**Lead Developer**: [@LeeHonggii](https://github.com/LeeHonggii)

> 본 저장소는 새로운 팀 구성으로 독립적으로 개발되고 있습니다.
>
> **참고**: 이 저장소는 기존 ListenCarePlease 프로젝트에서 코드베이스를 가져왔으나,
> 새로운 방향으로 독립 개발 중입니다.
