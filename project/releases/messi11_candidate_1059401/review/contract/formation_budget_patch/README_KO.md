# 1021834 시간 배분 수정

1021834는 highest 정밀도에서 GPU 별도 프로세스까지 관측/원시 행동/최종 행동/상태가 정확히 일치했다. 재현성 문제는 이 설정과 짧은 궤적에서 해소됐다.
다음 중단은 per-arm 650초에 컴파일/검증 시간을 포함시키고 180초를 평가용으로 남기는 조기 종료 때문이다. original이 2944/8192 step에서 멈췄다. 검증·설정 약373초, 첫 PPO update64.72초, 이후 평균1.45초가 측정됐다. 학습 수치 오류가 아닌 실행 시간 배분 문제다.

수정 범위: 재현성 결과 재사용(체크포인트/원본 소스/환경 해시 및 결과 일치 확인), highest 정밀도 고정, 조건별 budget900초/timeout950초, 학습 전 영상 먼저 생성. 모델, 행동 변환, 보상, 각 조건8192학습 step,1024평가 tick, 초기 가중치는 바꾸지 않는다. 1021834의 부분 학습 가중치는 사용하지 않는다.

MESSI 루트에 압축파일 업로드 후:
```
tar -xzf formation_budget_patch.tar.gz
sbatch formation_budget_patch/run_short.sbatch
```
기존 messi11_formation_patch 폴더 및 project/experiments/messi11_formation_1021834가 필요하다. 기존 파일은 수정하지 않는다. 검사 불일치 시 실패 원인을 저장한다.

```
JOBID=새작업번호
squeue -j "$JOBID" -o "%.18i %.10T %.12M %.12L %.30j"
tail -F "messi11-budgetfix-${JOBID}.log"
```

결과: project/experiments/messi11_budgetfix_JOBID/
- pretrain_preview/video/preview.mp4: 먼저 저장하는 초기 가중치+대형+CONTROL 후보 영상. PPO 추가 학습 결과가 아니다.
- original, formation, formation_control/short_comparison.json: 같은 초기 가중치/학습량의 전후 비교.
- candidate_preview/video/preview.mp4: 8192 step PPO 이후 후보 영상.
- execution_plan.json: 계산 예산과 참조 출처.
- diagnostics.tar.gz: 완료·실패 후 남은 결과.

실제 실행은 전체3300초에서 종료 요청하며 1시간 Slurm 한도 안에서 압축 시간을 남긴다. 측정값 기준 수십 분 예상이지만 신규 조건의 컴파일 시간에 따라 미완료일 수 있다. 예상 시간을 완료 보장으로 사용하지 않는다. 서버 GPU에서 이번 wrapper의 전체 완료는 아직 확인 전이다. 로컬 검증은 Python 문법/CLI 및 bash 문법 검사다.

짧은 결과와 영상을 확인하기 전에는12시간 작업을 제출하지 않는다. 기존 long.py의 준비 기준은 유지된다. 이후 제출이 가능해지면 highest를 명시적으로 전달해야 한다:
```
sbatch --export=ALL,JAX_DEFAULT_MATMUL_PRECISION=highest messi11_formation_patch/run_12h.sbatch project/experiments/messi11_budgetfix_SHORTJOB/formation_control
```
