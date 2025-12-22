import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import axios from 'axios'
import { triggerEfficiencyAnalysis, getDiarizationStatus } from '../services/api'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function TaggingPageNew() {
  const { fileId } = useParams()
  const navigate = useNavigate()

  const [loading, setLoading] = useState(true)
  const [taggingData, setTaggingData] = useState(null)
  const [speakerNames, setSpeakerNames] = useState({}) // SPEAKER_XX -> 이름 매핑
  const [transcript, setTranscript] = useState([]) // 개별 발화 수정 가능
  const [view, setView] = useState('summary') // 'summary' or 'diarization'
  const [diarizationStatus, setDiarizationStatus] = useState(null)

  useEffect(() => {
    fetchTaggingData()
  }, [fileId])

  useEffect(() => {
    if (view === 'diarization') {
      fetchDiarizationStatus()
    }
  }, [view, fileId])

  const fetchTaggingData = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/v1/tagging/${fileId}`)
      setTaggingData(response.data)

      // 초기 화자 이름 매핑 (final_name 우선, 없으면 suggested_name)
      const initialNames = {}
      response.data.suggested_mappings.forEach((mapping) => {
        initialNames[mapping.speaker_label] = mapping.final_name || mapping.suggested_name || ''
      })
      setSpeakerNames(initialNames)

      // 대본 초기화 (개별 수정 가능하도록)
      setTranscript(response.data.sample_transcript.map(seg => ({
        ...seg,
        speaker_label: seg.speaker_label // 개별 화자 변경 가능
      })))

      setLoading(false)
    } catch (error) {
      console.error('태깅 데이터 조회 실패:', error)
      setLoading(false)
    }
  }

  const fetchDiarizationStatus = async () => {
    try {
      const data = await getDiarizationStatus(fileId)
      setDiarizationStatus(data)
    } catch (error) {
      console.error('분리 상태 조회 실패:', error)
    }
  }

  const handleBulkNameChange = (speakerLabel, name) => {
    setSpeakerNames({ ...speakerNames, [speakerLabel]: name })
  }

  const handleSegmentSpeakerChange = (index, newSpeaker) => {
    const updated = [...transcript]
    updated[index].speaker_label = newSpeaker
    setTranscript(updated)
  }

  const applyBulkMapping = (fromSpeaker, toName) => {
    // 일괄 적용: 해당 화자의 모든 발화를 이름으로 변경
    setSpeakerNames({ ...speakerNames, [fromSpeaker]: toName })
  }

  const handleConfirm = async () => {
    try {
      // 최종 매핑 전송 (개별 수정 반영)
      const finalMappings = Object.entries(speakerNames).map(([speaker_label, name]) => ({
        speaker_label,
        final_name: name.trim() || speaker_label
      }))

      await axios.post(`${API_BASE_URL}/api/v1/tagging/confirm`, {
        file_id: fileId,
        mappings: finalMappings
      })

      // 효율성 분석은 백엔드에서 자동으로 실행됨 (tagging.py confirm 엔드포인트에서 background_tasks 실행)
      console.log('화자 태깅 완료. 효율성 분석은 백엔드에서 자동 실행됩니다.')

      navigate(`/result/${fileId}`)
    } catch (error) {
      console.error('태깅 확정 실패:', error)
      alert('태깅 확정에 실패했습니다.')
    }
  }

  if (loading) {
    return (
      <div className="p-8 flex items-center justify-center min-h-[calc(100vh-4rem)]">
        <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-accent-blue"></div>
      </div>
    )
  }

  const allNamesFilled = Object.values(speakerNames).every(name => name.trim() !== '')

  return (
    <div className="p-8">
      <div className="max-w-7xl mx-auto">
        {/* 헤더 */}
        <div className="text-center mb-6">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">화자 태깅</h1>
          <p className="text-gray-600 dark:text-gray-300">각 화자에게 이름을 매핑하거나 개별 발화를 수정하세요</p>
        </div>

        {/* 뷰 전환 버튼 */}
        <div className="flex justify-center gap-4 mb-6">
          <button
            onClick={() => setView('summary')}
            className={`px-6 py-2 rounded-lg font-semibold transition-all ${
              view === 'summary'
                ? 'bg-accent-blue text-white shadow-lg'
                : 'bg-bg-tertiary dark:bg-bg-tertiary-dark text-gray-700 dark:text-gray-300 hover:bg-bg-accent/20'
            }`}
          >
            📊 요약
          </button>
          <button
            onClick={() => setView('diarization')}
            className={`px-6 py-2 rounded-lg font-semibold transition-all ${
              view === 'diarization'
                ? 'bg-accent-blue text-white shadow-lg'
                : 'bg-bg-tertiary dark:bg-bg-tertiary-dark text-gray-700 dark:text-gray-300 hover:bg-bg-accent/20'
            }`}
          >
            🎯 분리 & 수정
          </button>
        </div>

        {view === 'summary' ? (
          // 요약 뷰: 화자별 일괄 매핑
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">🎤 화자 목록</h2>
              {taggingData?.suggested_mappings.map((mapping) => (
                <div key={mapping.speaker_label} className="bg-bg-tertiary dark:bg-bg-tertiary-dark rounded-xl shadow-lg p-6 border border-bg-accent/30">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-lg font-bold text-gray-900 dark:text-white">{mapping.speaker_label}</h3>
                    <div className="flex gap-2">
                      {mapping.nickname && (
                        <span className="text-xs bg-orange-100 dark:bg-orange-900 text-orange-700 dark:text-orange-200 px-3 py-1 rounded-full">
                          {mapping.nickname}
                        </span>
                      )}
                      {mapping.suggested_name && (
                        <span className="text-xs bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-200 px-3 py-1 rounded-full">
                          제안: {mapping.suggested_name}
                        </span>
                      )}
                    </div>
                  </div>

                  <input
                    type="text"
                    value={speakerNames[mapping.speaker_label] || ''}
                    onChange={(e) => handleBulkNameChange(mapping.speaker_label, e.target.value)}
                    placeholder="이름을 입력하세요"
                    className="w-full px-4 py-2 border border-bg-accent/30 bg-bg-secondary dark:bg-bg-secondary-dark text-gray-900 dark:text-white rounded-lg focus:ring-2 focus:ring-accent-blue focus:border-transparent"
                  />

                  {/* 빠른 선택 - 이름 */}
                  {taggingData?.detected_names && taggingData.detected_names.length > 0 && (
                    <div className="mt-3">
                      <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">이름 선택:</p>
                      <div className="flex flex-wrap gap-2">
                        {taggingData.detected_names.map((name, idx) => (
                          <button
                            key={idx}
                            onClick={() => handleBulkNameChange(mapping.speaker_label, name)}
                            className="px-3 py-1 bg-blue-100 dark:bg-blue-900 hover:bg-blue-200 dark:hover:bg-blue-800 text-blue-700 dark:text-blue-200 rounded-full text-sm transition-colors"
                          >
                            {name}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* 빠른 선택 - 닉네임 */}
                  {taggingData?.detected_nicknames && taggingData.detected_nicknames.length > 0 && (
                    <div className="mt-3">
                      <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">닉네임 선택:</p>
                      <div className="flex flex-wrap gap-2">
                        {taggingData.detected_nicknames.map((nickname, idx) => (
                          <button
                            key={idx}
                            onClick={() => handleBulkNameChange(mapping.speaker_label, nickname)}
                            className="px-3 py-1 bg-orange-100 dark:bg-orange-900 hover:bg-orange-200 dark:hover:bg-orange-800 text-orange-700 dark:text-orange-200 rounded-full text-sm transition-colors"
                          >
                            {nickname}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* 미리보기 */}
            <div>
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">👀 미리보기</h2>
              <div className="bg-bg-tertiary dark:bg-bg-tertiary-dark rounded-xl shadow-lg p-6 max-h-[600px] overflow-y-auto space-y-3">
                {transcript.map((seg, idx) => {
                  const displayName = speakerNames[seg.speaker_label] || seg.speaker_label
                  return (
                    <div key={idx} className="p-3 bg-bg-secondary dark:bg-bg-secondary-dark rounded-lg">
                      <div className="font-semibold text-accent-blue dark:text-blue-300 mb-1">{displayName}</div>
                      <div className="text-gray-700 dark:text-gray-200 text-sm">{seg.text}</div>
                    </div>
                  )
                })}
              </div>
            </div>
          </div>
        ) : (
          // 분리 & 수정 뷰: 화자 분리 상태 시각화 + 이름 편집 + 세그먼트 개별 수정
          <div className="space-y-6">
            {/* 분리 품질 점수 */}
            <div className="bg-bg-tertiary dark:bg-bg-tertiary-dark rounded-xl shadow-lg p-6 border border-bg-accent/30">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">📊 화자 분리 품질</h2>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Alignment Score */}
                <div className="text-center">
                  <div className="text-5xl font-bold text-accent-blue dark:text-blue-300 mb-2">
                    {diarizationStatus?.alignment_score?.toFixed(1) || '0'}%
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">매칭 정확도</div>
                  <div className="mt-2 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-accent-blue"
                      style={{ width: `${diarizationStatus?.alignment_score || 0}%` }}
                    ></div>
                  </div>
                </div>

                {/* Total Segments */}
                <div className="text-center">
                  <div className="text-5xl font-bold text-green-600 dark:text-green-400 mb-2">
                    {diarizationStatus?.total_segments || 0}
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">전체 세그먼트</div>
                </div>

                {/* Unknown Segments */}
                <div className="text-center">
                  <div className="text-5xl font-bold text-orange-600 dark:text-orange-400 mb-2">
                    {diarizationStatus?.unknown_segments || 0}
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">미할당 세그먼트</div>
                  <div className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                    ({diarizationStatus?.unassigned_duration?.toFixed(1) || '0'}초)
                  </div>
                </div>
              </div>
            </div>

            {/* 화자 이름 매핑 */}
            <div className="bg-bg-tertiary dark:bg-bg-tertiary-dark rounded-xl shadow-lg p-6 border border-bg-accent/30">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">🎤 화자 이름 설정</h2>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {taggingData?.suggested_mappings.map((mapping) => (
                  <div key={mapping.speaker_label} className="bg-bg-secondary dark:bg-bg-secondary-dark rounded-lg p-4 border border-bg-accent/20">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-bold text-gray-900 dark:text-white">{mapping.speaker_label}</span>
                      {mapping.suggested_name && (
                        <span className="text-xs bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-200 px-2 py-1 rounded-full">
                          제안: {mapping.suggested_name}
                        </span>
                      )}
                    </div>
                    <input
                      type="text"
                      value={speakerNames[mapping.speaker_label] || ''}
                      onChange={(e) => handleBulkNameChange(mapping.speaker_label, e.target.value)}
                      placeholder="이름 입력"
                      className="w-full px-3 py-2 border border-bg-accent/30 bg-white dark:bg-bg-primary-dark text-gray-900 dark:text-white rounded-lg text-sm focus:ring-2 focus:ring-accent-blue focus:border-transparent"
                    />
                    {/* 빠른 선택 버튼 */}
                    {taggingData?.detected_names && taggingData.detected_names.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1">
                        {taggingData.detected_names.map((name, idx) => (
                          <button
                            key={idx}
                            onClick={() => handleBulkNameChange(mapping.speaker_label, name)}
                            className="px-2 py-1 bg-blue-50 dark:bg-blue-900/30 hover:bg-blue-100 dark:hover:bg-blue-900/50 text-blue-700 dark:text-blue-300 rounded text-xs transition-colors"
                          >
                            {name}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* 세그먼트 매칭 상태 & 개별 수정 */}
            <div className="bg-bg-tertiary dark:bg-bg-tertiary-dark rounded-xl shadow-lg p-6 border border-bg-accent/30">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">📋 세그먼트 개별 수정 (최근 100개)</h2>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                각 세그먼트의 화자를 개별적으로 수정할 수 있습니다. 색상은 신뢰도를 나타냅니다.
              </p>

              <div className="space-y-2 max-h-[600px] overflow-y-auto">
                {diarizationStatus?.merged_segments?.map((seg, idx) => {
                  // 신뢰도에 따른 색상 구분
                  const isUnknown = seg.speaker === 'UNKNOWN'
                  const isLowConfidence = seg.confidence === 'low' || seg.confidence === 'medium'

                  let bgColor, borderColor
                  if (isUnknown) {
                    bgColor = 'bg-red-50 dark:bg-red-900/20'
                    borderColor = 'border-red-300 dark:border-red-700'
                  } else if (isLowConfidence) {
                    bgColor = 'bg-orange-50 dark:bg-orange-900/20'
                    borderColor = 'border-orange-300 dark:border-orange-700'
                  } else {
                    bgColor = 'bg-green-50 dark:bg-green-900/20'
                    borderColor = 'border-green-300 dark:border-green-700'
                  }

                  // 현재 화자의 표시 이름
                  const displayName = speakerNames[seg.speaker] || seg.speaker

                  return (
                    <div
                      key={idx}
                      className={`p-3 rounded-lg border ${bgColor} ${borderColor}`}
                    >
                      <div className="flex items-start gap-3 mb-2">
                        <div className="flex-shrink-0">
                          {/* 화자 선택 드롭다운 */}
                          <select
                            value={seg.speaker}
                            onChange={(e) => {
                              // merged_segments를 수정하도록 처리
                              const updatedSegments = [...diarizationStatus.merged_segments]
                              updatedSegments[idx].speaker = e.target.value
                              setDiarizationStatus({ ...diarizationStatus, merged_segments: updatedSegments })
                            }}
                            className="px-3 py-1 border border-bg-accent/30 bg-white dark:bg-bg-primary-dark text-gray-900 dark:text-white rounded text-xs font-semibold focus:ring-2 focus:ring-accent-blue"
                          >
                            {taggingData?.suggested_mappings.map((mapping) => (
                              <option key={mapping.speaker_label} value={mapping.speaker_label}>
                                {speakerNames[mapping.speaker_label] || mapping.speaker_label}
                              </option>
                            ))}
                          </select>
                        </div>

                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-xs text-gray-500 dark:text-gray-400">
                              {seg.start.toFixed(1)}s - {seg.end.toFixed(1)}s
                            </span>
                            {seg.confidence && (
                              <span className={`text-xs px-2 py-0.5 rounded ${
                                seg.confidence === 'high' ? 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-200' :
                                seg.confidence === 'medium' ? 'bg-yellow-100 dark:bg-yellow-900 text-yellow-700 dark:text-yellow-200' :
                                'bg-orange-100 dark:bg-orange-900 text-orange-700 dark:text-orange-200'
                              }`}>
                                신뢰도: {seg.confidence === 'high' ? '높음' : seg.confidence === 'medium' ? '보통' : '낮음'}
                              </span>
                            )}
                          </div>
                          <div className="text-sm text-gray-700 dark:text-gray-300">{seg.text}</div>
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          </div>
        )}

        {/* 확정 버튼 */}
        <div className="mt-8 space-y-4">
          <button
            onClick={handleConfirm}
            disabled={!allNamesFilled}
            className={`w-full px-6 py-4 rounded-xl font-semibold text-lg transition-all ${
              allNamesFilled
                ? 'bg-accent-blue text-white hover:bg-blue-600 shadow-lg'
                : 'bg-gray-300 dark:bg-gray-700 text-gray-500 dark:text-gray-400 cursor-not-allowed'
            }`}
          >
            {allNamesFilled ? '✅ 태깅 완료 → 결과 보기' : '⚠️ 모든 화자의 이름을 입력해주세요'}
          </button>

          {/* 홈으로 가기 버튼 */}
          <div className="text-center">
            <button
              onClick={() => navigate('/')}
              className="px-6 py-2 bg-bg-secondary dark:bg-bg-secondary-dark hover:bg-bg-accent/20 text-gray-700 dark:text-gray-200 rounded-lg font-medium transition"
            >
              홈으로 가기
            </button>
            <p className="mt-2 text-xs text-gray-500 dark:text-gray-400">
              나중에 대시보드에서 이어서 진행할 수 있습니다
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
