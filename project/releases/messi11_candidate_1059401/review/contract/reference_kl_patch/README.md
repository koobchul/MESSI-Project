# Reference-policy KL + PPO: 1시간 제한 실행

MESSI 루트에 압축 파일을 놓고 실행합니다.

```bash
tar -xzf reference_kl_patch.tar.gz
sbatch reference_kl_patch/run.sbatch
```

설치 스크립트는 없습니다. 기존 project 파일과 기존 체크포인트를 덮어쓰지 않습니다.
기존 attack_fulltime/repair 패치가 설치된 환경과 repaired actor_best.pkl이 필요합니다.
학습 관련 기존 소스 해시가 다르면 학습 전에 중단합니다.

## 이번 결과에서 확인한 것

동일 시드 5000/5001, 각 6,400스텝 평가 합계:

|정책|득점|실점|합법 점유 틱|슛 요청|슛 실행|
|---|---:|---:|---:|---:|---:|
|기존 repaired BC|0|3|13|27|2|
|BC + PPO 8,192스텝|0|4|5|0|0|
|BC만 121회 업데이트|0|6|7|1|0|

BC만 학습했을 때 손실은 약 0.683에서 0.539로 줄었으나 경기 지표는 악화됐습니다.
PPO 없이도 악화가 발생한다는 증거입니다. PPO가 무관하다거나 원인이 하나라고
증명한 결과는 아닙니다. 이 평가는 풀타임 경기가 아닙니다.

## 변경 내용

기존의 hard-label BC 보조 손실을 제거하고, 초기 repaired BC 정책을 고정한 뒤
현재 rollout 관측에서 KL(reference || current)을 PPO 손실에 더합니다.
계수는 1.0이며 최적값으로 검증된 값은 아닙니다.
의도 categorical과 모든 7개 연속 동작의 pre-tanh Gaussian을 비교합니다.
MOVE의 body-aim도 포함됩니다. 현재 정책과 기준 정책에 같은 retouch guard를
적용하고 padding은 제외합니다. 기준 파라미터에는 gradient가 흐르지 않습니다.
모델이 동일할 때 보조 손실과 gradient는 부동소수점 오차 범위에서 0입니다.

보존 제약은 성능 개선을 보장하지 않습니다. 초기 정책 자체의 낮은 경기력도
해결해야 하며, 이 실행은 PPO로 개선할 때 행동이 무너지는지 확인하는 후보 실험입니다.

## 실행 순서와 중단

- 기존 repaired BC에서 새 PPO/critic 학습을 시작합니다. 실패한 BC-only 가중치를 사용하지 않습니다.
- 8,192스텝 → 시드 5000/5001 각 6,400스텝 평가.
- 득점 감소, 실점 증가, 합법 점유 80% 미만 또는 슛 실행 감소가 있으면 중단합니다.
- 통과하면 총 32,768스텝까지 이어서 학습하고 동일 기준으로 평가합니다.
- 기존 평가 baseline.json을 재사용합니다. 반복 사용한 검증 시드이므로 독립 최종 평가가 아닙니다.
- 통과하고 골득실 또는 점유가 개선되면 candidate_needs_heldout으로 표시합니다.
  ready_for_long_run 필드는 예비 필터일 뿐, 12시간 학습을 자동 실행하지 않습니다.
- 개선이 불명확하면 no_clear_gain, 성능이 떨어지면 stopped_regression입니다.
- 기존 모델 파일은 자동 교체하지 않습니다. selected_checkpoint는 보고용 후보 경로입니다.

Slurm 제한은 1시간, 내부 실행 제한은 55분입니다. 서버 속도에 따라 두 단계 모두
끝나지 않을 수 있습니다. PAUSED이면 같은 sbatch 명령으로 저장 지점부터 재개합니다.
이미 중단/완료된 실행은 다시 제출해도 새로 학습하지 않습니다.

결과 파일:

```text
project/experiments/attack_reference_kl_seed0/reference_kl_results.tar.gz
```

상태 확인:

```bash
cat project/experiments/attack_reference_kl_seed0/status.json
```

메인 Slurm 로그에는 실행 단계가 표시되고, 세부 학습 로그는 결과 폴더의
reference_8192.log 및 reference_32768.log에 저장됩니다.

## 로컬 검증 범위

실제 repaired 체크포인트와 기존 관측 샘플로 CPU JAX JIT loss/gradient를 검증했습니다.
동일 정책 KL 약 5.6e-9, gradient 최대 절댓값 약 5.9e-6이며,
평균·의도·표준편차 변경 시 KL이 증가하고 padding이 제외되는 것을 확인했습니다.
통합 손실이 PPO + reference KL과 일치하고 gradient가 유한함을 확인했습니다.
Mock runner에서 악화 시 8,192스텝 중단, 완료 재실행 시 중복 학습 방지,
통과 시 32,768스텝 진행을 확인했습니다. Python 및 sbatch 구문 검사를 통과했습니다.
GPU 전체 학습이나 실제 경기 성능 향상을 로컬에서 검증한 것은 아닙니다.
