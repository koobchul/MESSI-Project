# 12시간 MAPPO 실행

## 이번 결과에 따른 결정

SHOT 보정 비교(seed 3000,3001, 각 6400스텝) 합계:

| 후보 | 정상 소유 틱 | 실제 슈팅 | 득점 | 실점 |
|---|---:|---:|---:|---:|
| 보정 전 수정 BC | 31 | 4 | 0 | 2 |
| 약한 SHOT 보정 | 16 | 0 | 0 | 3 |
| 강한 SHOT 보정 | 16 | 0 | 0 | 3 |

두 SHOT 보정은 채택하지 않는다. 현재 관측에서 두 보정은 유효한 슈팅까지 없앴고,
소유/실점도 나빠졌다. 이 결과는 앞선 학습 데이터의 분류 정확도 향상이 실제 경기 향상으로
연결되지 않았음을 보여준다. 소유 실패의 단일 원인이 확정된 것은 아니다.

## 실행 내용

- 출발점: attack_repair_seed0/actor_best.pkl (SHA256을 검증).
- 새 critic/optimizer로 MAPPO를 시작한다. 실패한 SHOT bias 보정은 적용하지 않는다.
- 기존 검증된 train_attack.py와 evaluate_fulltime.py를 호출한다. 환경 및 기존 project 코드를 수정하지 않는다.
- 실제 학생 정책 rollout으로 최대 누적 131072, 262144, 524288, 1048576스텝까지 학습한다.
- actor LR 1e-5, critic LR 1e-4, entropy .001, rollout256, PPOepochs4, targetKL .01.
- 기존 v3 보상(진행/접근/소유 각 .05)을 유지한다. 긴 학습의 효과를 보는 실험이지 새 보상의 효과를 주장하지 않는다.
- 각 완료 단계의 체크포인트를 별도로 고정하고 seed4000/4001에서 각12800스텝 검증한다.
- 학습 전 수정 BC도 동일 검증에 포함한다.
- 두 검증 경기 합계 골득실, 득점, 정상 소유 비율 합계, 실제 슈팅 순으로 선택한다.
- 새 seed4002/4003에서는 선택 모델과 학습 전 BC의 전체 경기를 같은 조건으로 비교한다(최대4경기).
  학습 전 BC가 선택되면 중복 경기를 만들지 않고 BC 두 경기만 실행한다.
- 테스트 결과로 다시 후보를 고르지 않으며 최종 모델로 자동 승격하지 않는다.

검증 seed가 두 개뿐이므로 순위는 불확실하다. 최종 결과에서도 full_duration_complete와
added_time_limit_reached를 구분해야 한다. 긴 학습이 득점이나 승리를 보장하지 않는다.

## 시간과 저장

Slurm 요청은 최대12시간이다. 대기열 시간은 별도다. 모든 단계가 빨리 끝나면 일찍 종료한다.
내부 예산은 학습+중간비교8시간30분, 전체 작업11시간40분으로 결과 저장 여유를 둔다.
학습 예산이 끝나면 진행 중 자식 프로세스를 정지하고, 검증까지 완료한 후보 중에서 선택한다.
검증되지 않은 최신 가중치를 자동 선택하지 않는다. 중간 학습 상태는 이어서 사용할 수 있도록 남긴다.

학습은 기존 trainer가 약8192실제환경스텝마다 원자적으로 저장한다.
중단 시 마지막 저장 이후 작업은 유실될 수 있다. 평가는128스텝마다 저장된다.
단계/경기 종료마다 overnight_results.tar.gz를 갱신한다. 예외 종료에도 압축을 시도한다.
SIGKILL이나 노드 장애에서는 종료 처리가 실행되지 않을 수 있다.

동일 sbatch 명령으로 재개할 수 있다. test 단계에 들어간 실행은 남은 평가만 재개하며,
자동으로 학습 목표를 늘리지 않는다. 완료된 실행을 재제출하면 재학습하지 않는다.
출력 폴더는 project/experiments/attack_overnight_seed0이며 동시 실행을 flock으로 막는다.
원본 가중치와 이전 실험은 유지한다.

## 실행

MESSI 루트, 기존 conda messi 환경에서:

```bash
tar -xzf overnight_attack_patch.tar.gz
sbatch overnight_attack_patch/run.sbatch
```

기존 attack_fulltime_patch와 attack_repair_patch가 설치되어 있어야 한다.
12시간 요청이 클러스터 정책 때문에 거절되면 오류 메시지를 확인한다. 이 코드는 클러스터 제한을 우회하지 않는다.

로그는 `tail -F attack_overnight-작업번호.log`로 본다.
`-F`는 대기열 때문에 로그가 아직 없어도 생성될 때까지 재시도한다.
Ctrl+C는 tail만 종료한다. Slurm으로 제출한 작업은 계속 실행된다.

다음 파일을 보내면 된다:
`project/experiments/attack_overnight_seed0/overnight_results.tar.gz`

파일이 없거나 강제 종료로 갱신되지 않았다면 작업 종료 후:

```bash
tar -czf overnight_results_manual.tar.gz --exclude='*.pkl' --exclude='*.tmp' --exclude='*.tar.gz' --exclude='run.lock' \
  -C project/experiments attack_overnight_seed0
```

## 검증 범위

Python 컴파일과 sbatch bash 문법 검사를 통과했다.
가짜 trainer/evaluator를 이용한 실제 자식 프로세스 통합 테스트에서 단계별 학습, 고정 후보 선택,
동일 seed의 선택 모델/BC 전체 경기 비교, 완료 후 재실행 방지, 원본 보존을 확인했다.
GPU 야간 학습과 성능 개선은 로컬에서 검증하지 않았다.
이 패키지는 기존에 서버에서 실행된 학습/평가 코드를 조율하는 새 실행기다.
