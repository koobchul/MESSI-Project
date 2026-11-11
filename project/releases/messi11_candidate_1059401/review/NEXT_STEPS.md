# 종료 기준과 다음 실행

추가 보상/시드 탐색과 PPO 학습은 현재 중단한다. 기존 PPO1024 승자를 동결한다.
최종 제출이 끝날 때까지의 남은 작업은 아래 세 가지다.

1. 실제 과제 규정/runner 계약 확인: 경기설정600/3600, 정상종료의 의미,
   GK/formation/CONTROL 변환 허용범위, 입력출력 및 양팀슬롯 관점.
2. 확인된 인터페이스로 동결 정책 inference 어댑터 구현 및 짧은 reset/step 검사.
3. 실제 평가설정에서 사전고정한 미열람seed로 HOME/AWAY 한 차례 최종 평가.
   통과하면 가중치/추론코드/설정/재현명령/보고서로 마감한다.

새 실험을 여는 조건은 공식 실행 실패, 허용되지 않은 동작, 인덱스/방향 오류처럼
구체적인 결함이 발견됐을 때뿐이다. 막연히 더 좋아질 가능성 때문에 연장하지 않는다.
외부 학생 정책이 제공되면 추가 경쟁평가가 가능하나 자료가 없다고 무한대기하지 않는다.

이번 패치는 checkpoint를 byte-copy하고 hash를 검사한다. 기존 파일 변경/재학습/GPU0.
공식 자료를 자동으로 못 찾으면 RL_PROJECT.pdf와 실제 제출 템플릿/runner를 보내야 한다.
관측 소스 전체를 담는 패치가 아니라 제출 규격과 후보 inference 경로를 확인하기 위한 수집이다.
출력 checkpoint.pkl은 서버에 보존하고 release_review.tar.gz만 전달한다.
이미 존재하는 출력 경로는 덮어쓰지 않는다.

```bash
tar -xzf messi11_release_prepare.tar.gz
JOBID=$(sbatch --parsable messi11_release_prepare/run.sbatch)
echo "$JOBID"
tail -n 40 -F "messi11-release-prepare-${JOBID}.log"
```

보낼 파일: project/releases/messi11_candidate_JOBID/release_review.tar.gz
실제 과제자료/runner가 수집되지 않았다면 별도 첨부한다.
이 파일 자체는 최종 제출용 inference 어댑터가 아니다.
