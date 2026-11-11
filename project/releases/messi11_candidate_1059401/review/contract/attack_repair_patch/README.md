# Attack repair: MOVE body-direction supervision

## 이번에 확인한 오류

이전 warmstart_attack.py는 intent==MOVE일 때 continuous[2:7]의 회귀 손실을 0으로 만들었다.
하지만 MESSI의 dynamics/action.py::decode_physics_action은 MOVE에서도 force_to_ball[0:2]를
body_target으로 사용한다. rule_based/policy.py도 MOVE에서 공 또는 주행 방향을 body_aim으로 넣는다.
즉 continuous[2:4]는 MOVE에서도 학습해야 한다. 이전 구현은 이 지도 신호를 빠뜨렸다.
몸 방향은 접촉 기하에 영향을 준다. 이것이 모든 실패의 유일한 원인이라는 주장은 하지 않는다.

## 제공된 가중치/시연의 직접 진단 (seed 10, 기존 BC의 학습 데이터)

- 예시 40,960개, SHOT 라벨 1개.
- BC: CONTROL 926개 중 640개 정답, SHOT argmax 0회.
- BC+MAPPO: CONTROL 926개 중 718개 정답, SHOT argmax 0회.
- CONTROL 이동 명령 RMSE: BC 0.4702, BC+MAPPO 0.4801.
- BC CONTROL 예시 중 교사 이동 norm>0.25인 경우 약 30.6%가 교사와 음의 내적.
- BC MOVE body-aim RMSE 0.7587.
- 정규화 공 거리 norm<0.05인 MOVE 예시의 body-aim 음의 내적 비율 44.27%.
- 유일한 SHOT 예시: BC의 SHOT 확률 2.08%, BC+MAPPO 2.42%; 둘 다 MOVE 선택.

RMSE는 [-1,1] 제어값 단위다. 방향의 음의 내적은 두 출력 방향이 90도 이상 차이남을 뜻한다.
실제 선수 이동이나 몸 회전이 바로 그 각도로 발생했다는 뜻은 아니다.
'근처'는 정규화 상대 공 좌표의 L2 norm<0.05인 진단 구간이며 정확한 5m 반경이 아니다.
학습 데이터 진단이므로 이 수치를 일반화 성능으로 주장하면 안 된다.

## 수정

1. MOVE에도 force_to_ball 두 축의 손실을 적용한다. launch/spin만 MOVE에서 제외한다.
2. 이동과 body/ball direction, launch/spin의 손실을 나눠 CONTROL 때 이동 손실이
   7차원 평균에 묻히지 않게 한다. 움직이는 교사 예시에는 이동 방향 손실을 추가한다.
3. far MOVE / near MOVE / CONTROL / PASS / SHOT / CLEAR / CHALLENGE를 나누어 재표집한다.
   비중 .25/.25/.25/.15/.05/.02/.03. 존재하는 그룹만 사용한다.
4. 기존 seed 10~13을 재사용하고 새 seed 20,21을 각 16,384프레임 수집한다.
   이번에는 양 팀의 교사 행동을 사용한다. 팀1 입력의 entity 순서만 own-team-first로 바꾼다.
   좌표와 action은 이미 공격방향 기준이므로 다시 반전하지 않는다.
5. 새 seed 22의 8,192프레임은 검증에만 쓴다. 이전 BC가 학습/선택에 쓴 seed 14와 구분한다.
6. 기존 BC actor에서 3,000회 지도학습으로 수정한다. 이번 실행에는 추가 PPO 학습이 없다.
7. 독립 검증의 방향/제어 지표와 seed 2000의 실제 학생 정책 6,400스텝 비교를 수행한다.
8. 기준을 통과한 경우에만 새 seed 2001,2002의 전체 경기 평가를 자동 실행한다.
   미통과하면 원인 지표를 남기고 종료한다.

시연 teacher가 실제 평가 행동을 대신 실행하지 않는다. 평가 입력과 출력 구조는 기존 MLP와 같다.
이번 모델 역시 모방학습 모델로 명시해야 한다. 11v11은 포함하지 않는다.

## 실행

MESSI 폴더에 압축파일을 업로드하고 `(messi)` 환경에서:

```bash
tar -xzf attack_repair_patch.tar.gz && python attack_repair_patch/install.py
sbatch attack_repair_patch/run_attack_repair.sbatch
```

2시간 GPU 배치를 요청한다. 클러스터가 거절하면:

```bash
sbatch --time=01:00:00 attack_repair_patch/run_attack_repair.sbatch
```

동일 명령으로 재개한다. 새 시연은 1,024프레임마다, fit은 250업데이트마다,
평가는 128프레임마다 저장한다. 같은 output에 작업을 동시에 실행하지 않는다.
기존 attack_fulltime 배치를 다시 실행하는 것이 아니라 이 repair 배치를 실행한다.

종료 후 업로드:
`project/experiments/attack_repair_seed0/repair_results.tar.gz`

로그 확인:
`tail -f attack_repair-작업번호.log`

## 모델 승격 기준

이는 비용을 제한하기 위한 명시적인 공학적 기준이며 성능 보장이나 통계적 충분조건은 아니다.

- 원본 라벨 수: train CONTROL>=200, PASS>=100, SHOT>=16;
  validation CONTROL>=50, PASS>=20, SHOT>=4.
  프레임 라벨 수이며 독립된 성공 행동 수가 아니다.
- 검증 MOVE body-aim RMSE가 기존 BC의 75% 이하.
- 검증 near MOVE body-aim 반대 방향 비율이 기존 BC의 50% 이하.
- 검증 near 이동 RMSE는 기존 BC 대비 5% 넘게 악화되지 않을 것.
- 검증 CONTROL/PASS/SHOT recall 각각 .5/.3/.25 이상.
- 실제 학생 정책 probe: 합법적 소유>=10틱이면서 기존 BC보다 증가,
  shot release>=1, 실점은 기존 BC+1 이하.

재표집은 새 슈팅 지식을 만들어내지 않는다. 충분한 교사 예시가 없으면 승격하지 않는다.
시연 커버리지가 부족해도 확인된 MOVE 오류의 fit은 수행하고 진단을 남기지만,
장시간 PPO와 전체 경기 평가로 자동 연결하지 않는다.
`GATE FAILED`는 예외/충돌이 아니라 승격 기준 미통과다.

## 결과 파일

- coverage.json: 원본 행동 라벨 수.
- diagnostics.json: 수정 전후 confusion matrix, recall, 이동·몸 방향 오차.
- gate_status.json: offline/실제 플레이 기준과 전체 경기 실행 여부.
- control_probe/{before,after}/results.csv: 같은 seed의 학생 정책 비교.
- fulltime/results.csv: 기준 통과 시에만 생성.

`fulltime` 결과 역시 added-time-limit 종료와 regulation_complete를 구분한다.
기존 원본 소스·이전 학습 결과를 덮어쓰지 않고 신규 파일로 설치한다.
