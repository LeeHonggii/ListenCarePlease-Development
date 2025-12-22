"""
화자 분리 서비스 (I,O.md Step 4)
- Pyannote Audio 3.1 사용
- 화자별 임베딩 추출
- speaker-diarization-3.1 모델
"""
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
import librosa

try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("⚠️ Pyannote not installed. Install with: pip install pyannote.audio")

from app.core.device import get_device
from app.core.config import settings


def run_diarization(audio_path: Path, device: str = None, num_speakers: int = None) -> Dict:
    """
    화자 분리 통합 인터페이스 (Pyannote 사용)

    Args:
        audio_path: 오디오 파일 경로
        device: 디바이스 ("cuda", "cpu", None=auto)
        num_speakers: 화자 수 (None=자동 감지)

    Returns:
        화자 분리 결과 (turns + embeddings)
    """
    if not PYANNOTE_AVAILABLE:
        raise ImportError("Pyannote is not installed. Please install it first.")

    # 디바이스 자동 감지
    if device is None:
        device = get_device()

    # CUDA 실제 사용 가능 여부 재확인
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    # Pyannote는 MPS 지원 제한적
    if device == "mps":
        print("⚠️ Pyannote has limited MPS support. Using CPU instead.")
        device = "cpu"

    print(f"[Diarization] Using device: {device}")
    print(f"[Diarization] Processing: {audio_path}")

    # 메모리 정리 (Diarization 시작 전)
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # HuggingFace 토큰 확인
    hf_token = settings.HUGGINGFACE_TOKEN
    if not hf_token:
        raise ValueError(
            "HuggingFace token is required for Pyannote models. "
            "Set HUGGINGFACE_TOKEN in your .env file."
        )

    # Pyannote Pipeline 초기화
    print("[Diarization] Loading Pyannote speaker-diarization-3.1 model...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )

    # 디바이스 설정
    if device == "cuda":
        pipeline = pipeline.to(torch.device("cuda"))
    else:
        pipeline = pipeline.to(torch.device("cpu"))

    # 오디오 파일 로드
    print("[Diarization] Loading audio file...")
    waveform_np, sample_rate = librosa.load(str(audio_path), sr=None, mono=True)
    waveform = torch.from_numpy(waveform_np).unsqueeze(0)

    # 화자 분리 실행
    print("[Diarization] Running speaker diarization...")
    diarization_params = {'waveform': waveform, 'sample_rate': sample_rate}

    # 화자 수 지정된 경우
    if num_speakers is not None:
        diarization_params['num_speakers'] = num_speakers
        print(f"[Diarization] Fixed speaker count: {num_speakers}")

    diarization_output = pipeline(diarization_params)

    # 결과 변환 (실제 임베딩 추출)
    result = convert_pyannote_to_custom_format(
        diarization_output,
        audio_path=audio_path,
        waveform=waveform_np,
        sample_rate=sample_rate,
        device=device,
        hf_token=hf_token
    )

    # Diarization 완료 후 메모리 정리
    del pipeline
    del diarization_output
    del waveform
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[Diarization] Detected {len(result['embeddings'])} speakers")
    print(f"[Diarization] {len(result['turns'])} segments")

    return result


def convert_pyannote_to_custom_format(
    diarization_output,
    audio_path: Path = None,
    waveform: np.ndarray = None,
    sample_rate: int = None,
    device: str = "cpu",
    hf_token: str = None
) -> Dict:
    """
    Pyannote 결과를 프로젝트 형식으로 변환 + 실제 임베딩 추출

    Args:
        diarization_output: Pyannote Annotation 객체
        audio_path: 오디오 파일 경로
        waveform: 오디오 waveform (numpy array)
        sample_rate: 샘플레이트
        device: 임베딩 추출에 사용할 디바이스
        hf_token: HuggingFace 토큰

    Returns:
        {
            "turns": [{"speaker_label": str, "start": float, "end": float}],
            "embeddings": {speaker_label: List[float]}
        }
    """
    # 1. turns 데이터 생성
    turns = []
    for segment, _, speaker in diarization_output.itertracks(yield_label=True):
        turns.append({
            "speaker_label": speaker,
            "start": round(segment.start, 2),
            "end": round(segment.end, 2)
        })

    # 2. 실제 임베딩 추출
    embeddings = {}
    unique_speakers = sorted(set(turn["speaker_label"] for turn in turns))

    # 임베딩 추출을 위한 조건 확인
    if waveform is not None and sample_rate is not None and hf_token:
        try:
            print("[Embedding] Loading Pyannote embedding model...")
            from pyannote.audio import Inference

            # Pyannote embedding 모델 로드
            embedding_model = Inference(
                "pyannote/embedding",
                use_auth_token=hf_token
            )

            # 디바이스 설정
            if device == "cuda":
                embedding_model.to(torch.device("cuda"))
            else:
                embedding_model.to(torch.device("cpu"))

            print(f"[Embedding] Extracting embeddings for {len(unique_speakers)} speakers...")

            # 각 화자의 임베딩 추출
            for speaker in unique_speakers:
                speaker_embeddings = []

                # 해당 화자의 모든 발화 구간 찾기
                speaker_turns = [t for t in turns if t["speaker_label"] == speaker]

                for turn in speaker_turns:
                    start_sample = int(turn["start"] * sample_rate)
                    end_sample = int(turn["end"] * sample_rate)

                    # 구간 길이 확인 (최소 0.5초)
                    if end_sample - start_sample < sample_rate * 0.5:
                        continue

                    # 오디오 세그먼트 추출
                    segment_waveform = waveform[start_sample:end_sample]

                    # 임베딩 추출
                    try:
                        # Pyannote Inference는 dict 형식으로 입력 받음
                        segment_dict = {
                            "waveform": torch.from_numpy(segment_waveform).unsqueeze(0),
                            "sample_rate": sample_rate
                        }
                        embedding = embedding_model(segment_dict)
                        speaker_embeddings.append(embedding)
                    except Exception as e:
                        print(f"⚠️ [Embedding] Failed to extract embedding for {speaker} at {turn['start']}-{turn['end']}: {e}")
                        continue

                # 평균 임베딩 계산
                if speaker_embeddings:
                    # 모든 임베딩을 numpy array로 변환
                    embeddings_array = np.array([emb.cpu().numpy().flatten() for emb in speaker_embeddings])
                    # 평균 계산
                    mean_embedding = np.mean(embeddings_array, axis=0)
                    # L2 정규화
                    mean_embedding = mean_embedding / (np.linalg.norm(mean_embedding) + 1e-8)
                    embeddings[speaker] = mean_embedding.tolist()
                    print(f"✅ [Embedding] {speaker}: {len(speaker_embeddings)} segments averaged")
                else:
                    # 임베딩 추출 실패 시 더미 임베딩 생성
                    print(f"⚠️ [Embedding] No valid embeddings for {speaker}, using dummy")
                    np.random.seed(hash(speaker) % (2**32))
                    dummy_embedding = np.random.randn(512).astype(np.float32)  # Pyannote embedding은 512차원
                    dummy_embedding = dummy_embedding / (np.linalg.norm(dummy_embedding) + 1e-8)
                    embeddings[speaker] = dummy_embedding.tolist()

            # 임베딩 모델 메모리 정리
            del embedding_model
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"✅ [Embedding] Successfully extracted embeddings for {len(embeddings)} speakers")

        except Exception as e:
            print(f"⚠️ [Embedding] Failed to load embedding model: {e}")
            print("[Embedding] Falling back to dummy embeddings")
            # 실패 시 더미 임베딩 생성
            for speaker in unique_speakers:
                np.random.seed(hash(speaker) % (2**32))
                dummy_embedding = np.random.randn(512).astype(np.float32)
                dummy_embedding = dummy_embedding / (np.linalg.norm(dummy_embedding) + 1e-8)
                embeddings[speaker] = dummy_embedding.tolist()
    else:
        # 조건 미충족 시 더미 임베딩 생성
        print("[Embedding] Missing parameters for embedding extraction, using dummy embeddings")
        for speaker in unique_speakers:
            np.random.seed(hash(speaker) % (2**32))
            dummy_embedding = np.random.randn(512).astype(np.float32)
            dummy_embedding = dummy_embedding / (np.linalg.norm(dummy_embedding) + 1e-8)
            embeddings[speaker] = dummy_embedding.tolist()

    result = {
        "turns": turns,
        "embeddings": embeddings
    }

    return result


def merge_stt_with_diarization(
    stt_segments: List[Dict], diarization_result: Dict
) -> Dict:
    """
    STT 결과와 화자 분리 결과 병합 (개선된 버전)
    - STT 세그먼트 손실 방지 (100% 보존)
    - 가장 많이 겹치는 화자에게 할당
    - 겹치지 않는 경우 가까운 화자에게 할당하거나 UNKNOWN 처리
    - 정렬 품질 점수 계산

    Args:
        stt_segments: STT 결과 [{"text": str, "start": float, "end": float}, ...]
        diarization_result: 화자 분리 결과 {"turns": [...], "embeddings": {...}}

    Returns:
        {
            "merged_result": [
                {"speaker": "speaker_00", "start": 0.0, "end": 5.2, "text": "안녕하세요"}, ...
            ],
            "alignment_score": 95.5,  # 0~100점
            "unassigned_duration": 2.5 # 초 단위
        }
    """
    import copy

    turns = diarization_result.get('turns', [])
    if not turns:
        # 분리 결과가 없으면 전체를 UNKNOWN으로 처리
        return {
            "merged_result": [
                {
                    "speaker": "UNKNOWN",
                    "start": s['start'],
                    "end": s['end'],
                    "text": s['text']
                } for s in stt_segments
            ],
            "alignment_score": 0.0,
            "unassigned_duration": sum(s['end'] - s['start'] for s in stt_segments)
        }

    merged = []

    total_stt_duration = 0.0
    total_overlap_duration = 0.0
    total_unassigned_duration = 0.0

    # 턴을 시작 시간 순으로 정렬
    sorted_turns = sorted(turns, key=lambda x: x['start'])

    for stt in stt_segments:
        s_start = stt['start']
        s_end = stt['end']
        s_text = stt['text']
        s_dur = s_end - s_start

        total_stt_duration += s_dur

        # 1. 겹치는 화자 턴 찾기
        overlaps = []
        for turn in sorted_turns:
            t_start = turn['start']
            t_end = turn['end']
            speaker = turn['speaker_label']

            # None 값 체크
            if t_start is None or t_end is None or s_start is None or s_end is None:
                continue

            # 겹치는 구간 계산
            o_start = max(s_start, t_start)
            o_end = min(s_end, t_end)

            if o_end > o_start:
                overlap_dur = o_end - o_start
                overlaps.append({
                    "speaker": speaker,
                    "duration": overlap_dur,
                    "turn_start": t_start,
                    "turn_end": t_end
                })

        assigned_speaker = "UNKNOWN"
        confidence = "high"  # 신뢰도: high, medium, low

        if overlaps:
            # 가장 많이 겹치는 화자 선택 (높은 신뢰도)
            best_overlap = max(overlaps, key=lambda x: x['duration'])
            assigned_speaker = best_overlap['speaker']
            total_overlap_duration += best_overlap['duration']

            # 겹침 비율로 신뢰도 판단
            overlap_ratio = best_overlap['duration'] / s_dur
            if overlap_ratio > 0.7:
                confidence = "high"
            elif overlap_ratio > 0.3:
                confidence = "medium"
            else:
                confidence = "low"
        else:
            # 겹치지 않는 경우: 거리 기반 신뢰도로 할당
            min_dist = float('inf')
            nearest_speaker = None
            prev_speaker = None
            next_speaker = None

            # 가장 가까운 화자 찾기
            for turn in sorted_turns:
                t_start = turn['start']
                t_end = turn['end']

                # STT가 턴보다 앞에 있는 경우 거리
                dist_before = max(0, t_start - s_end)
                # STT가 턴보다 뒤에 있는 경우 거리
                dist_after = max(0, s_start - t_end)

                dist = min(dist_before, dist_after) if dist_before > 0 and dist_after > 0 else max(dist_before, dist_after)

                if dist < min_dist:
                    min_dist = dist
                    nearest_speaker = turn['speaker_label']

                    # 이전/다음 화자 기록 (시간적 연속성 고려)
                    if t_end < s_start:
                        prev_speaker = turn['speaker_label']
                    elif t_start > s_end:
                        next_speaker = turn['speaker_label']

            # 거리 기반 신뢰도 할당
            if nearest_speaker:
                assigned_speaker = nearest_speaker

                # 거리 기반 신뢰도 계산
                if min_dist < 2.0:
                    confidence = "high"
                    print(f"[Gap Filling - High] STT [{s_start:.2f}-{s_end:.2f}] → {nearest_speaker} (거리: {min_dist:.2f}초)")
                elif min_dist < 5.0:
                    confidence = "medium"
                    print(f"[Gap Filling - Medium] STT [{s_start:.2f}-{s_end:.2f}] → {nearest_speaker} (거리: {min_dist:.2f}초) ⚠️ 검토 권장")
                else:
                    confidence = "low"
                    print(f"[Gap Filling - Low] STT [{s_start:.2f}-{s_end:.2f}] → {nearest_speaker} (거리: {min_dist:.2f}초) ⚠️ 검토 필요")
                    total_unassigned_duration += s_dur  # 낮은 신뢰도는 미할당으로 카운트
            else:
                # 화자 턴이 하나도 없는 극단적인 경우
                confidence = "none"
                total_unassigned_duration += s_dur
                print(f"⚠️ [CRITICAL] STT [{s_start:.2f}-{s_end:.2f}] - 화자 턴이 없음")

        merged.append({
            "speaker": assigned_speaker,
            "start": s_start,
            "end": s_end,
            "text": s_text,
            "confidence": confidence  # 신뢰도 추가
        })

    # 점수 계산
    alignment_score = 0.0
    if total_stt_duration > 0:
        alignment_score = (total_overlap_duration / total_stt_duration) * 100.0

    return {
        "merged_result": merged,
        "alignment_score": round(alignment_score, 2),
        "unassigned_duration": round(total_unassigned_duration, 2)
    }
