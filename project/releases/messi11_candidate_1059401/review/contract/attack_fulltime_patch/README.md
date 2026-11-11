# Attack + full-match pipeline (candidate v4, not a proven final policy)

## 실행

현재 reward_v3 패치가 설치된 MESSI 루트에 압축파일을 올린다.

```bash
tar -xzf attack_fulltime_patch.tar.gz && python attack_fulltime_patch/install.py
```

로그인 노드에서 `(messi)` 환경을 활성화한 뒤 배치 제출:

```bash
sbatch attack_fulltime_patch/run_attack_fulltime.sbatch
```

이미 GPU 할당을 받은 노드라면:

```bash
JAX_THREEFRY_PARTITIONABLE=0 JAX_PLATFORMS=cuda PYTHONPATH=src:. python -u project/run_attack_fulltime.py
```

4시간 배치를 요청한다. 클러스터의 최대 허용시간은 이 패키지에서 확인할 수 없다.
시간 제한으로 거절되면 `sbatch --time=01:00:00 attack_fulltime_patch/run_attack_fulltime.sbatch`로 제출한다.
시간 초과/중단 뒤 **같은 명령을 다시 실행**하면 시연 수집·BC·PPO·평가를 저장 지점에서 이어간다.
동일 output에 두 작업을 동시에 실행하지 않는다.
최초 JIT 컴파일 중 출력이 잠시 없을 수 있다. 소요시간/득점 개선을 보장하지 않는다.
GPU 초기화가 실패하면 즉시 중단한다. gate1에서 CPU로 조용히 실행되지 않는다.

## 자동 순서

1. Bundled rule-based policy의 시연: seed 10~13 학습, seed 14 모방 검증. seed당 8,192 환경 스텝.
2. 기존 414차원 MLP actor를 BC로 초기화. 3,000 minibatch 업데이트, batch 512.
   - intent 분류 + 이동/공 제어 회귀
   - MOVE의 사용하지 않는 공 제어 차원은 회귀 손실에서 제외
   - 희귀 intent 가중치를 사용하되 0.25~5로 제한
   - 별도 seed 검증 손실 최저 actor 저장; log_std=-2에서 RL 탐색 시작
3. 그 actor에서 MAPPO를 새 critic/optimizer로 시작: 131,072 환경 스텝.
   v3 보상/guard ON을 유지하며 actor LR=1e-5, critic LR=1e-4, entropy=.001, target KL=.01, rollout=256.
   기존 16k 정책의 optimizer/경기 상태를 BC 모델에 잘못 연결하지 않는다.
4. BC, BC+MAPPO, 기존 v3(기존 파일이 있으면)의 **고정 정책**을 seed 1000에서 각각 경기 종료까지 평가.
   intent argmax + tanh(mean)의 동일한 결정적 실행 조건. 이전 확률적 6400-step 결과와 직접 섞지 않는다.
5. 검증 득실차 → 득점 → 합법적 소유율 → 실제 슈팅 release 순으로 후보 선택.
6. 선택된 후보를 학습·선택에 쓰지 않은 seed 1001,1002에서 끝까지 평가.
7. 로그/설정/CSV를 results_bundle.tar.gz 하나로 묶는다.

완료 후 보낼 파일:
`project/experiments/attack_fulltime_seed0/results_bundle.tar.gz`

`selection.json`이 실제 선택된 모델 경로와 SHA256을 기록한다. BC 또는 기존 v3가 선택될 수도 있다.
PPO 추가학습이 반드시 개선된다는 가정으로 새 모델을 무조건 채택하지 않는다.
기본 학습 seed 1개/검증 seed 1개/테스트 seed 2개이므로 통계적으로 최상의 성능이라는 증거는 아니다.

## 왜 이 변경인가

최신 고정 정책 평가에서 v2 0–3, v3 0–7이며 두 조건 모두 합법적인 소유 0틱이었다.
보상 신호를 늘리는 것만으로 접촉·제어·패스·슈팅의 연속 행동이 만들어졌다는 증거가 없었다.
이번에는 실제 행동 예시로 actor를 먼저 학습시키는 **모방학습 + MAPPO** 경로를 추가한다.
최종 actor의 입력은 기존 관측뿐이다. 평가에서 teacher로 행동을 대신 실행하지 않는다.
Teacher는 메모리를 쓰지만 student는 MLP이므로 완전한 모사가 보장되지 않는다.
교수님의 프로젝트 규정에서 규칙 정책 시연 사용을 허용하는지는 제공 자료만으로 확인되지 않았다.
보고서에는 순수 RL 결과와 분리하여 명시해야 한다.

## 풀타임/공격 지표의 정확한 의미

- 평가 기본 `max_steps=0`: 외부 6,400스텝 제한 없이 환경 종료까지 진행.
- `scheduled_match_finished=True`: 규정시간 충족 또는 정규 추가시간 한도로 종료.
- `full_duration_complete=True`: 환경 원본 정의에 맞춰 regulation_complete일 때만 참.
- `added_time_limit_reached`: 경기 중단이 많아 추가시간 상한으로 종료. 위의 full_duration_complete는 거짓.
- 선수 부족/무효 재시작/비정상 시간 상한/외부 컷오프는 풀타임 성공으로 인정하지 않는다.
- halftime_reached, 실제 regulation_elapsed_seconds도 기록한다.
- shot_requests/pass_requests: 선수별 제어 프레임 요청 수.
- shot_releases/pass_releases: 물리 서브스텝에서 관측된 실제 deliberate contact release 수.
  이것은 유효슈팅/패스 성공/고유 공격 횟수와 같지 않다.
- 합법적 소유는 기존 verified control 기반 판정. 단순 소유자 번호/불법 TRAP을 소유 성공으로 세지 않는다.
- guard는 explicit 불법 요청만 막으므로 MOVE의 수동적 재접촉은 여전히 가능하다.

평가 carry/상대 메모리/누적 지표를 매 청크 원자적으로 저장한다.
체크포인트 내용이 달라지면 과거 평가 상태와 섞지 않고 중단한다.
PPO는 8,192스텝마다 저장하므로 강제 종료 시 마지막 저장 이후 학습은 다시 수행한다.
재개 시 CSV의 저장 시점 이후 행은 제거하여 중복 이력을 막는다.
BC fit은 100회 업데이트마다, 수집은 청크마다 저장한다.

## 범위와 다음 단계

기존 src/footballworld, 보상 규칙, train.py, 기존 체크포인트를 덮어쓰지 않는다.
추가된 train_attack.py만 BC 초기화를 지원하며 나머지 MAPPO 계산은 기존 구현을 사용한다.
검증된 5v5 실행 경로를 우선 완성한다. 11v11은 이번 패키지에 구현하지 않았다.
11v11에서는 입력 차원이 달라 현재 MLP 가중치를 그대로 사용할 수 없다.
다음 전환 조건은 풀타임 종료 증거, 실제 공 제어/슈팅 관측, 고정 정책 비교 결과다.
영상 생성이나 Transformer 도입은 이번 범위에 포함하지 않는다.

## 로컬 검증

검증 내역은 VALIDATION.md 참고. 단축 경기 통합 테스트를 실제 90분 성능 평가로 간주하지 않는다.
GPU 서버의 장시간 학습과 전체 경기 성능 결과는 위 실행으로 생성해야 한다.

```bash
JAX_THREEFRY_PARTITIONABLE=0 PYTHONPATH=src:. python -m unittest project.test_attack_pipeline -v
```
