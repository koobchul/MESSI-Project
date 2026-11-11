# BC rehearsal + MAPPO: 1시간 검증 실행

## 변경 내용

기존 PPO는 BC로 초기화한 후 PPO 손실만으로 갱신했다.
이번 실행은 매 actor 업데이트에 실제 교사 시연의 BC 손실을 더한다:

`total_actor_loss = PPO_loss + 1.0 * BC_loss`

BC_loss = intent cross entropy + 4*이동 MSE + 2*force/body 방향 MSE
          + non-MOVE에서만 launch/spin MSE.
MOVE의 몸 방향 두 축은 계속 지도한다. PPO의 rollout 행동, log-probability,
advantage, critic 및 환경은 바꾸지 않는다. 동일 gradient update 안에 두 손실을 결합한다.
시연 행동이 경기 평가에서 학생 행동을 대신 실행하지 않는다.

기존 project 파일을 수정하지 않는다. 별도 train.py가 새 프로세스에서 PPO actor loss 함수를
BC 포함 함수로 교체한 뒤 기존 project.train_attack.main을 호출한다. 코드 의존성 SHA256을 먼저 확인한다.
기존 optimizer/rollout 체크포인트 구조는 유지하며 이번 실험 전용 출력에서만 재개한다.

## 재표집 수정

기존 시연 seed10~13 및20~21을 사용한다. 검증 seed22는 사용하지 않는다.
고정4096개 rehearsal bank를 생성하고 이후 resume에서는 같은 bank를 재사용한다.
75%는 원본 데이터 균등 추출,25%는 존재하는 non-MOVE 클래스들을 균등하게 선택해 추출한다.
각 예시에 `원본 확률 / 혼합 표집 확률`을 곱해 손실을 계산한다.
이것은 rare action의 관측을 늘리되 학습 목적에서 SHOT 비중을 5%로 강제하는 것을 피한다.
고정 부분집합이므로 전체 시연의 정확한 목적함수와 동일하지 않고, 표집 오차/과적합 가능성이 남는다.
BC계수1.0도 검증이 필요한 설정이며 성능 보장이 아니다.

## 실험 및 중단 기준

1. 기존 수정 BC를 새 seed5000,5001에서 각6400스텝 평가.
2. 같은 수정 BC에서 새 critic/optimizer로 BC+PPO 약8192환경스텝 학습.
3. 동일 seed와 스텝에서 비교. 합계 득점 비감소, 합계 실점 비증가,
   정상 소유가 baseline의80% 이상, 실제 슈팅 발사가 baseline 이상이어야 계속한다.
4. 미통과하면 stopped_regression으로 종료. 통과하면 누적32768스텝까지 학습하고 재비교한다.
5. 장시간 학습 후보 표시에는 위 기준 외에 실제 발사1 이상, 그리고 골득실 개선 또는
   정상 소유가 max(10, baseline*1.2+1) 이상인 조건을 추가한다.

두 seed의 짧은 검사는 공학적 gate이며 통계적 성능 보장이 아니다.
이번에는 전체 경기와12시간 학습을 자동으로 실행하지 않는다.
BC보존과 학습 안정성에 집중한 실험이며 DAgger/커리큘럼/성공 상태 초기화는 구현하지 않았다.
현재 낮은 공격 성능을 이 한 가지 변경으로 해결했다고 주장하지 않는다.

## 시간/재개/보존

Slurm 최대1시간, 내부55분에 자식 프로세스를 중단하고 결과 저장 시간을 확보한다.
전체 단계가 반드시1시간 안에 완료된다는 보장은 없다.
학습은 약2048스텝마다, 평가는128스텝마다 저장한다. 마지막 저장 이후 작업은 중단 시 유실될 수 있다.
같은 sbatch 명령으로 진행 중 실행을 재개한다.
complete 또는 stopped_regression인 실행은 다시 학습하지 않는다.
원본 가중치, 기존 학습 결과 및 환경 소스는 유지된다.
오류 발생 시 로그가 압축된다. 노드 장애나SIGKILL은 종료 처리 실행을 보장할 수 없다.

## 실행

MESSI 루트, conda messi에서:

```bash
tar -xzf anchored_attack_patch.tar.gz
sbatch anchored_attack_patch/run.sbatch
```

설치 스크립트는 없다. 기존 attack_fulltime/attack_repair 코드와 시연 파일이 필요하다.
의존성 변경 또는 시연 누락이면 학습 전에 실패하며 메시지를 출력한다.

배치 단계 로그:
`tail -F attack_anchor-작업번호.log`

첫 학습 상세 로그(학습 시작 전에는 파일을 기다림):
`tail -F project/experiments/attack_anchor_seed0/anchor_8192.log`

다음 학습 상세 로그:
`tail -F project/experiments/attack_anchor_seed0/anchor_32768.log`

종료 후 업로드:
`project/experiments/attack_anchor_seed0/anchor_results.tar.gz`

## 검증

- 업로드된 실제 가중치/시연과 실제 MLP, PPO 손실 함수를 사용한 CPU JIT검사 통과.
- 합성 PPO rollout 입력에서 결합 목적함수=PPO+BC 확인.
- 모든 파라미터의 결합 gradient=PPO gradient+BC gradient 및 유한값 확인.
- 중요도 가중 재표집이 원본 클래스 비중을 기대값에서 복원하는지100000샘플 검사 통과.
- 슈팅 능력 손실을 gate가 거부하고, 변화 없는 baseline은 장시간 후보가 되지 않는지 확인.
- 가짜 trainer/evaluator를 실제 자식 프로세스로 실행해 첫 단계 악화 시32768학습을 건너뛰고,
  결과를 압축하며 재제출 시 중단된 실험을 다시 시작하지 않는지 확인.
- Python 컴파일 및 sbatch bash 문법 검사 통과.

실제 환경 rollout을 포함한 GPU 전체 학습이나 경기 성능 개선은 로컬에서 검증하지 않았다.
CPU 테스트 런타임: JAX/JAXlib0.11.2, Flax0.12.10, Optax0.2.8.
서버 환경을 변경하거나 패키지를 새로 설치하는 명령은 포함하지 않았다.
