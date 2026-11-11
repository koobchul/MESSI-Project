# 골키퍼 제어·보상 확인 및 실점 위치 추적

## 코드 확인 결과

- project/envs/messi_5v5.py: 우리 팀 첫 슬롯(0)이 is_goalkeeper=True.
- project/train_attack.py: controlled_indices=[0,1,2,3,4].
- project/utils/team_action.py: team_id==controlled_team이면 learned_action 선택.
  골키퍼 예외가 없어 학습 정책이 우리 팀 GK까지 제어합니다.
- project/evaluate_fulltime.py도 같은 combine_team_actions로 평가합니다.
- project/envs/reward_v3.py의 접근 대상:
  active & (team_id == controlled_team) & ~is_goalkeeper & ~prohibited.
  골키퍼는 공 접근 potential 계산에서 제외됩니다. 가장 가까운 적격 필드 선수만 기여합니다.
- 골키퍼의 합법 점유는 possession potential에 포함될 수 있습니다.
- PPO는 팀 advantage를 모든 제어 선수에 broadcast하므로 골키퍼도 팀 보상에 의해
  업데이트됩니다. 접근 potential 제외는 골키퍼 gradient 제외나 골문 위치 고정과 다릅니다.
- 현재 제안된 fulltime DAgger는 supervised 학습이며 v3 보상을 손실로 사용하지 않습니다.

## 기존 자료의 한계

fullmatch_compare_results.tar.gz에는 집계 CSV, 진행 로그와 summary만 있습니다.
실점 직전 선수/공 좌표는 없어 과거 실점의 골키퍼 이탈을 확정할 수 없습니다.
실점 개수나 점유율만으로 골키퍼가 골문을 비웠다고 판단할 수 없습니다.

## 실행

```bash
tar -xzf goalkeeper_trace_patch.tar.gz
sbatch goalkeeper_trace_patch/run.sbatch
```

기존 repaired BC 고정, 시드 7100/7101, 각 6400스텝씩:
1. 기존과 같이 GK 포함 전체 5명을 학습 정책이 제어.
2. 같은 초기 조건에서 GK 하나만 규칙 기반으로 교체. 나머지 4명은 같은 학습 정책.

총 4회 평가, Slurm 1시간/내부 55분 제한입니다. 미완료 시 같은 명령으로 재개합니다.
기존 모델, 보상, 학습 파일을 수정하지 않습니다. 이 실험은 짧은 진단이며 풀타임이 아닙니다.

## 기록

각 실행 폴더에:
- trajectory.csv: 매 제어 스텝의 SI 좌표(m). GK와 공(x,y,z), 우리 골문 x,
  공격 방향, GK의 골라인 전방 거리, 골문 중심 거리, 공과의 거리,
  페널티 구역 밖 여부, 공 live/restart 상태, GK 활성 상태/행동, 전후 점수.
- goals.json: 실점이 난 전이의 시작 위치 및 그 직전 약 10초의 위치 기록.
- summary.json: 실점 수, live 중 GK 페널티 구역 밖 비율과 평균 골문 거리.
- state.pkl: 128스텝마다 저장하는 재개용 상태. 전달용 압축에는 포함되지 않습니다.

좌표는 반드시 env.step 이전 state에서 가져옵니다. 실점 직후 킥오프 재배치 좌표를
실점 직전 위치로 오인하지 않습니다. 한 제어 스텝 안의 물리 substep까지 추적하지
않으므로 정확한 골라인 통과 순간의 위치는 아닙니다.
골문은 attack_direction으로 계산해 진영 변경에도 대응합니다.
페널티 구역 밖 여부는 위치 설명용이며, 정상적인 스위핑/볼 처리까지 실수로 판정하지 않습니다.

해석은 실점 직전 GK 위치·공 위치·live/restart 상태를 함께 봅니다.
규칙 기반 GK에서 실점이 줄면 GK 제어가 기여했을 가능성을 뒷받침하지만,
경기 궤적도 달라지므로 특정 골의 단독 원인을 입증하는 같은 상태의 재실험은 아닙니다.
변화가 없더라도 후반부 GK 문제까지 배제하는 결과는 아닙니다.

## 결과 전달

project/experiments/goalkeeper_trace/goalkeeper_trace_results.tar.gz

진행 확인:
```bash
tail -F project/experiments/goalkeeper_trace/learned_gk_7100.log
```

로컬 검증: 실제 환경 CPU JAX JIT 한 스텝에서 로깅을 추가한 기존 GK 모드가 원래 evaluator와 상태 및 집계가 정확히 일치함을 확인했습니다. SI 사전 위치와 골문 방향, 규칙 기반 GK 전이의 유한한 출력을 확인했습니다. Python/sbatch 구문을 검사했습니다. 전체 4회 GPU 진단은 아직 실행 전입니다.
