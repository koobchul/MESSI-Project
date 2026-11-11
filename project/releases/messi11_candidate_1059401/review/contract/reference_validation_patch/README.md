# Frozen reference-KL candidate: independent paired screening

MESSI 루트에서:

```bash
tar -xzf reference_validation_patch.tar.gz
sbatch reference_validation_patch/run.sbatch
```

이번 reference-KL 결과는 8,192스텝 학습과 두 시드 평가를 정상 완료했습니다.
합법 점유 13→32틱, 슛 실행 2→2, 실점 3→5, 득점 0→0입니다.
실점 증가 기준으로 stopped_regression이므로 32,768스텝 학습은 하지 않았습니다.
전체 점유율은 0.102%→0.250% 수준으로 여전히 낮습니다.

새 실험은 가중치를 바꾸지 않고 기존 BC와 이번 후보를 시드 6100~6103에서
각각 6,400스텝씩 평가합니다. 기존 두 시드 결과는 선정용이고 이번 네 시드는
새 검증용입니다. 두 모델 모두 동일 시드/스텝/결정론적 동작으로 비교합니다.
기존 두 체크포인트가 정확히 일치하는지 SHA256으로 확인합니다.

사전 결정 기준: 득점 감소 없음, 실점 증가 없음, 점유가 baseline 80% 이상,
슛 실행 감소 없음. 이 조건을 충족하면서 골득실 또는 점유 개선이 있고
슛 실행이 최소 1회이면 candidate_for_longer_test, 아니면 retain_bc입니다.
이는 작은 표본의 screening이며 통계적 유의성이나 풀타임 성능을 증명하지 않습니다.
기존 두 시드에서 나타난 실점 악화도 이 결과와 함께 고려해야 합니다.
모델 자동 교체나 12시간 학습은 수행하지 않습니다.

Slurm 1시간/내부 55분 제한. 서버 속도에 따라 미완료일 수 있으며 paused면
같은 명령으로 재개합니다. 모델/시드별 완료 결과는 재사용하고, 진행 중 경기는
기존 evaluator의 저장 상태에서 이어갑니다. 실제 GPU 실행 시간은 미검증입니다.

결과:
project/experiments/attack_reference_validation/reference_validation_results.tar.gz

상태:
cat project/experiments/attack_reference_validation/summary.json

메인 로그: attack_reference_validation-JOBID.log
세부 로그: project/experiments/attack_reference_validation/reference_6100.log 등.
