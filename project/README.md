# MESSI 5v5 → MAPPO / MLP / Transformer 진행 흐름

## 0. 최종 연구 방향

MESSI Football Environment
        ↓
5v5에서 먼저 실험
        ↓
동일한 MAPPO 학습 알고리즘 사용
        ↓
┌───────────────────────────────┐
│                               │
MLP Actor                 Transformer Actor
│                               │
└──────────── 공정 비교 ──────────┘
        ↓
성능 / 학습 안정성 / 계산량 비교
        ↓
11v11로 확장


Research Question
        ↓
"선수 간 관계를 Attention으로 모델링하는 것이
MLP보다 더 좋은 팀 정책을 학습하는가?"


============================================================

## 1. MESSI 환경 구축

MESSI `hw` branch 설치
        ↓
Python 3.11 `messi` conda environment
        ↓
JAX CPU/GPU 동작 확인
        ↓
GPU compute node에서 smoke test 성공
        ↓
실제 MESSI match.mp4 생성 성공
        ↓
환경 자체가 정상적으로 실행되는 것 확인 ✅


============================================================

## 2. 5v5 환경 생성

`project/envs/messi_5v5.py`
        ↓
Home 5명 + Away 5명
        ↓
총 10명의 player
        ↓
`FootballWorld(...)`
        ↓
`env.reset(...)`
        ↓
5v5 MESSI 환경 생성 ✅


현재 환경

Home Team
  Player 0
  Player 1
  Player 2
  Player 3
  Player 4

        VS

Away Team
  Player 5
  Player 6
  Player 7
  Player 8
  Player 9


============================================================

## 3. MESSI Observation 확인

`obs = env.observe_all(reset.rollout)`
        ↓
MESSI가 모든 player의 observation을 한 번에 반환


왜 observer가 10명인가?
        ↓
5v5 전체 선수가 10명이기 때문
        ↓
각 선수마다 "자기 관점의 observation"이 하나씩 존재
        ↓
총 10 observer


예:

Observer 0
        ↓
"내 위치에서 다른 9명과 공이 어떻게 보이는가?"

Observer 1
        ↓
"내 위치에서 다른 9명과 공이 어떻게 보이는가?"

...

Observer 9


따라서

`observe_all()`
        ↓
10명의 observation을 batch 형태로 반환


============================================================

## 4. MESSI Raw Observation 구조 분석

`describe_observation(obs)`
        ↓

MESSI 원본 observation에는

.valid
.self_state.*
.players.*
.ball.*
.possession.*
.restart.*
.restart_release.*
.match.*

등이 존재


한 observer 기준 Raw Scalar 개수

Float  = 144
Int    = 27
Bool   = 89
        ↓
Raw Total = 260


주의
        ↓
260개를 무작정 float로 변환해서 MLP에 넣지 않음
        ↓
의미에 따라 다시 구조화


왜?
        ↓
예를 들어

restart.kind = 3

에서 숫자 3 자체의 크기는 의미가 없음
        ↓
Categorical feature이므로 one-hot 처리 필요


============================================================

## 5. Roster Metadata 확인

`roster = env.roster_metadata(reset.rollout)`
        ↓

각 선수의 정적인 특성 확인

team_id
player_id
slot_generation
is_goalkeeper
max_speed
reach_height
height
ball_control
endurance_factor


하지만 Actor에는

player_id
raw team_id
slot_generation

같은 단순 identity 정보는 직접 넣지 않음
        ↓
특정 선수 번호를 외우는 것을 방지
        ↓
5v5 → 11v11 확장성 확보


대신

is_self
is_teammate
is_opponent

라는 "관계 정보"로 변환해서 사용


============================================================

## 6. Global State 확인

`global_state = env.global_state_view(reset.rollout)`
        ↓

전체 경기의 centralized information

전체 player position
전체 player velocity
player ability
ball
score
possession
restart
clock
offside
...

        ↓

나중에 MAPPO Critic이 사용


즉 MAPPO 구조

Actor
        ↓
개별 player observation 사용

Critic
        ↓
Global State 사용


= CTDE
Centralized Training
Decentralized Execution


============================================================

## 7. Player Feature Encoder 생성

`build_player_features(obs, roster)`
        ↓

MESSI의 `.players.*`
+
Roster Metadata
+
Observer와 각 선수의 관계
        ↓
선수 한 명을 30개의 feature로 표현


Player 1명당 포함되는 정보 예시

relative position
relative velocity
facing sin / cos
stamina
recovery 상태
yellow cards
on_pitch
sent_off
offside
possessor
restart_taker
last_actor
goalkeeper 여부
max_speed
height
ball_control
endurance
is_self
is_teammate
is_opponent
...


5v5 결과

Player Features
        ↓
shape = (10, 10, 30)


각 축의 의미

(10, 10, 30)
 ↑   ↑   ↑
 │   │   └── 선수 한 명당 feature 30개
 │   │
 │   └────── 한 observer가 보는 전체 선수 10명
 │
 └────────── observation을 가진 observer 10명


예를 들어

`player_features[0]`
        ↓
Observer 0 기준
        ↓
shape = (10, 30)


Player 0 → 30 features
Player 1 → 30 features
Player 2 → 30 features
...
Player 9 → 30 features


============================================================

## 8. 중요한 점: Transformer가 10개인 것이 아님

잘못된 이해

10 observer
        ↓
Transformer 10개 ❌


실제 구조

하나의 Shared Actor Network θ
        ↓
모든 agent가 같은 parameter를 공유


예:

Agent 0 ─┐
Agent 1 ─┤
Agent 2 ─┤
...      ├──→ Shared Actor θ
Agent 9 ─┘


즉 첫 번째 10은

"모델 개수"
가 아니라
"batch에 들어오는 observer 개수"


============================================================

## 9. Context Feature Encoder 생성

Player 자체에 속하지 않는 경기 상황은 별도로 분리

`build_context_features(obs, roster)`
        ↓

Context에 포함되는 정보

Observer 자신의 position
Observer 자신의 velocity

Ball relative state
Ball live

Possession timing

Restart 상태
Restart release 상태

Score
Match control tick

Match clock
Added time
Dead-ball 정보
...


결과

Context Features
        ↓
shape = (10, 34)


각 observer마다

34개의 "경기 전체 상황" feature


즉 한 agent 기준

Player Features
        ↓
(10 players, 30 features)

+

Context Features
        ↓
(34 features)


30 → 34로 증가한 것이 아님

서로 다른 두 종류의 feature

Player Feature = 선수 entity 정보
Context Feature = 경기 전체 상황 정보


============================================================

## 10. Categorical Feature Encoder 생성

MESSI의 일부 int 값은
수치형 숫자가 아니라 category

예:

PASS
SHOT
CONTROL

Throw-in
Corner
Goal kick

Own team
Opponent team

등


따라서

raw integer
        ↓
One-Hot Encoding


현재 사용 중인 categorical 그룹

possession.previous_team
last_contact.mechanism
last_contact.intent
last_contact.intent_source
last_contact.outcome
last_contact.restart_kind
last_contact.law11_effect
restart.kind
restart.team
restart_release.kind
restart_release.release_mechanism
match.offside_direct_exempt_team
match.gk_handling_restricted_team
match.clock.period


Team category는

Team 0 / Team 1

그대로 사용하지 않고

No Team
Own Team
Opponent Team

으로 변환


Match Period는

실제 경기 category

1 = 전반
2 = 후반

만 사용

0은 invalid observation sentinel일 뿐
실제 period가 아님


따라서

Period 1
        ↓
[1, 0]

Period 2
        ↓
[0, 1]


결과

Categorical Features
        ↓
shape = (10, 80)


============================================================

## 11. 현재 최종 Actor 입력 구조

한 observer 기준

Player Features
        ↓
10 players × 30
        ↓
300

+

Context Features
        ↓
34

+

Categorical Features
        ↓
80


따라서

300 + 34 + 80
        ↓
414


============================================================

## 12. MLP Observation Encoder

`encode_mlp_observation(obs, roster)`
        ↓

Player Feature

(10, 10, 30)
        ↓
Flatten
        ↓
(10, 300)

        +

Context
(10, 34)

        +

Categorical
(10, 80)

        ↓
Concatenate
        ↓

MLP Observation
        ↓
(10, 414)


즉

10 observers
×
414-dimensional observation


테스트 결과

Player Features      = (10, 10, 30) ✅
Context Features     = (10, 34)     ✅
Categorical Features = (10, 80)     ✅
MLP Observation      = (10, 414)    ✅

MESSI env.step()      = 정상         ✅

pytest
        ↓
1 passed ✅


============================================================

## 13. MLP와 Transformer는 같은 정보를 사용

공정한 비교를 위해

Input Information
        ↓
동일해야 함


즉

Player Features
Context Features
Categorical Features

는 MLP와 Transformer 모두 동일


차이는

"이 정보를 어떻게 처리하느냐"


============================================================

## 14. MLP 처리 방식

Player Features

10 × 30
        ↓
Flatten
        ↓
300

+

Context 34

+

Categorical 80

        ↓

414-dimensional vector

        ↓
MLP


현재 결정한 MLP Baseline

Input 414
        ↓
Linear 256
        ↓
tanh
        ↓
Linear 256
        ↓
tanh
        ↓
256-dimensional latent
        ↓
Hybrid Action Head


기존 128 → 128 대신

256 → 256

사용 예정


이유

MLP가 너무 약하면
        ↓
Transformer가 이겨도

"Attention이 좋아서"
가 아니라

"MLP capacity가 너무 작아서"

일 수 있음


따라서 비교 가능한
조금 더 강한 MLP baseline 사용


============================================================

## 15. 왜 tanh?

ReLU도 사용할 수 있음
        ↓
하지만 PPO / MAPPO baseline에서는
tanh가 자주 사용됨


tanh
        ↓
output range = [-1, 1]
        ↓
bounded activation
        ↓
policy optimization에서 비교적 안정적


따라서 이번 실험에서는

Activation을 별도의 연구변수로 만들지 않고
        ↓
tanh로 고정


즉

MLP vs Transformer

만 주요 비교 변수로 유지


============================================================

## 16. Transformer에서는 어떻게 처리할 예정인가?

한 agent 기준

Player Features
        ↓
10 × 30


각 player를 하나의 entity token으로 사용

Player 0 → 30 features
Player 1 → 30 features
...
Player 9 → 30 features


각 token

30-dimensional raw feature
        ↓
Linear Projection
        ↓
30 → 64
        ↓
64-dimensional embedding


따라서 5v5에서는

10 Player Tokens
×
64-dimensional embedding
        ↓
Transformer Encoder


초기 계획

d_model = 64
heads   = 4
layers  = 1~2


Context + Categorical 정보는
Transformer 결과와 나중에 결합


============================================================

## 17. 5v5 → 11v11 확장

현재 5v5

전체 선수
        ↓
10명


Player Features

(10 observers, 10 players, 30 features)


11v11에서는

전체 선수
        ↓
22명


Player Features

(22 observers, 22 players, 30 features)


중요

30은 그대로

선수 수만

10 → 22


============================================================

## 18. MLP의 11v11 확장

5v5

10 × 30 + 34 + 80
        ↓
414


11v11

22 × 30 + 34 + 80
        ↓
660 + 114
        ↓
774


따라서

5v5 MLP Input
        ↓
414

11v11 MLP Input
        ↓
774


MLP는 player 수가 변하면
input dimension도 달라짐


============================================================

## 19. Transformer의 11v11 확장

5v5

10 player tokens
×
30 raw features
        ↓
30 → 64
        ↓
10 × 64


11v11

22 player tokens
×
30 raw features
        ↓
같은 30 → 64 projection
        ↓
22 × 64


즉 Transformer에서는

Token feature dimension = 동일
        ↓
Sequence length만

10 → 22


이것이
5v5 → 11v11 scaling에서
Transformer가 갖는 구조적 장점 중 하나


============================================================

## 20. 현재 프로젝트 전체 진행도

MESSI 설치 / 환경 구축
        ↓
✅

GPU / JAX 확인
        ↓
✅

Smoke Match 실행
        ↓
✅

5v5 Environment
        ↓
✅

Observation 구조 분석
        ↓
✅

Roster Metadata
        ↓
✅

Global State
        ↓
✅

Player Feature Encoder
        ↓
✅
(10, 10, 30)

Context Feature Encoder
        ↓
✅
(10, 34)

Categorical Feature Encoder
        ↓
✅
(10, 80)

MLP Observation Encoder
        ↓
✅
(10, 414)

        ↓
────────────────────────────────
        현재 위치
────────────────────────────────
        ↓

MLP Actor Backbone
        ↓
414 → 256 → 256
        ↓
NEXT

        ↓

Hybrid Action Head
        ↓
6 Discrete Intent
+
7 Continuous Controls

        ↓

MESSI IntentAction 연결

        ↓

Centralized Critic

        ↓

MAPPO 구현

        ↓

5v5 MLP Baseline 학습

        ↓

Transformer Actor 구현

        ↓

5v5
MLP vs Transformer 비교

        ↓

모델 성능 비교
Win / Draw / Loss
Goal Difference
Reward
Learning Curve
Stability
Wall-clock
Memory
Parameter Count

        ↓

11v11 Scaling Experiment

        ↓

최종 연구 결과