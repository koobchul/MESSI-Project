# BC-only 원인 분리

anchor 실험은8192환경스텝에서 stopped_regression으로 끝났다.
두 seed 합계 소유13->5틱,실제 슈팅2->0,실점3->4였다.
BC 손실 적용은 로그/manifest로 확인되지만, 이 손실이 기존 학생 행동을 보존한다고 보장하지 않는다.
교사의 자연 빈도를 모방하도록 목적함수를 바꾼 것이기 때문에 기존 재표집 BC와도 목적이 다르다.

이번 작업은 PPO를 제외하고 동일한 초기 가중치, 고정 rehearsal bank, Adam/gradient clipping,
학습률1e-5, 실제 PPO actor 업데이트 횟수 합계와 같은 횟수로 BC만 실행한다.
기존5000/5001 비교 결과를 재사용하고 BC-only만 각6400스텝 평가한다.
재사용 결과의 가중치 hash, 스텝수, deterministic 조건을 확인한다.
이는 독립 최종 평가가 아닌 통제 비교이며 새로운 성능 개선 패키지라고 주장하지 않는다.
BC-only가 악화되면 BC 목적함수 자체의 영향을 확인할 수 있다. PPO의 영향이나 상호작용을 배제하는 것은 아니다.
BC-only가 유지되면 PPO 추가/결합 업데이트가 저하에 관여했을 가능성이 높아진다.
두 seed 결과만으로 일반적인 인과관계나 최적 계수를 확정하지 않는다.

필요 파일은 이미 서버에 있는 기존 anchored_attack_patch 및 attack_anchor_seed0 결과,
anchor_bank.pkl, 원본 수정 BC다. 기존 파일을 수정하지 않고 attack_bc_isolation_seed0에 저장한다.
학습은 짧은 고정 bank 학습이며 완료 후 가중치를 저장한다. 학습 도중 중단되면 짧은 BC부분은 다시 실행한다.
평가는 기존 evaluator가128스텝마다 저장하며 같은 명령으로 재개할 수 있다.
GPU 최대1시간이다. 실행 완료 시간이나 성능 개선은 보장하지 않는다.
로컬에서 Python/bash 문법을 확인했으며 이번 BC-only 전체 GPU 실행은 수행하지 않았다.
손실 함수는 앞서 실제 JIT gradient검사를 통과한 anchored_attack_patch.anchor를 그대로 사용한다.

```bash
tar -xzf bc_isolation_patch.tar.gz
sbatch bc_isolation_patch/run.sbatch
```

로그: `tail -F bc_isolation-작업번호.log`
평가 상세: `tail -F project/experiments/attack_bc_isolation_seed0/eval_5000.log`
결과: `project/experiments/attack_bc_isolation_seed0/bc_isolation_results.tar.gz`
강제 종료로 압축이 없으면 작업 종료 후:

```bash
tar -czf bc_isolation_manual.tar.gz --exclude='*.pkl' --exclude='*.tar.gz' --exclude='*.tmp' --exclude='run.lock' -C project/experiments attack_bc_isolation_seed0
```
