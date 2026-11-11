# BC vs reference-KL: 풀타임 비교 (12시간 상한)

```bash
tar -xzf fullmatch_compare_patch.tar.gz
sbatch fullmatch_compare_patch/run.sbatch
```

MESSI 루트에서 제출합니다. 기존 repaired BC와 reference_8192.pkl을 그대로
사용합니다. 추가 학습 없이 시드 7100,7101에서 모델별 2경기, 총 4경기를 실행합니다.
체크포인트 SHA256을 검사하고 기존 파일을 덮어쓰지 않습니다.

--max-steps 0으로 외부 스텝 제한을 없애고 환경이 경기를 종료할 때까지 진행합니다.
환경 종료와 정규시간 완료는 다릅니다. summary.json에서
all_full_duration_complete=true인지 별도로 확인합니다. 중단/시간 제한으로
일찍 끝난 경기를 풀타임 성공으로 판정하지 않습니다.

Slurm 12시간 상한, 내부 11시간 50분 상한입니다. 네 경기가 끝나면 즉시 종료하며,
12시간을 억지로 채우지 않습니다. 서버 성능에 따라 모두 끝나지 않을 수 있습니다.
paused이면 동일 sbatch 명령으로 재개합니다. 경기 상태는 기존 evaluator가
128스텝마다 저장하고, 이미 끝난 경기는 다시 실행하지 않습니다.

결과 파일:
project/experiments/attack_fullmatch_compare/fullmatch_compare_results.tar.gz

메인 로그: attack_fullmatch_compare-JOBID.log
경기 진행 로그 예:
```bash
tail -F project/experiments/attack_fullmatch_compare/bc_7100.log
```

앞선 독립 짧은 평가(4시드, 각 6400스텝)는 기존 BC/후보 순서로
득점 0/0, 실점 11/8, 합법 점유 39/41틱, 슛 실행 7/6회였습니다.
사전 screening 기준은 retain_bc입니다. 기준을 사후에 완화하거나 후보를
자동 승격하지 않습니다. 이번 실험은 최초 목표인 풀타임 성능 비교입니다.
동일 경기시간이라도 스텝 수는 달라질 수 있으므로 점유는 틱 비율로 비교합니다.
새 두 시드만으로 통계적 우월성을 증명하지 않습니다.

Python/sbatch 구문과 집계·정규시간 미완료 구분을 로컬 검증했습니다.
서버 GPU에서 이번 네 경기 실행은 아직 수행하지 않았습니다.
