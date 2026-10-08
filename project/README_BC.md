5v5 MESSI 생성
        ↓
10명 Observation 생성
        ↓
describe_observation()
        ↓
각 feature
shape / dtype / dimension 파악
        ↓
Action 테스트
        ↓
env.step()



messi_5v5.py
│
│  MESSI 5v5 생성
│
▼
NormalizedObservation
│
▼
obs_encoder.py
│
├─ describe_observation()       ← 지금 만드는 것
│
├─ encode_mlp_observation()     ← 다음
│
└─ encode_entity_tokens()       ← MLP 이후
        │
        ├──────────────┐
        ▼              ▼
    MLP Actor      Transformer Actor

################
MLP
################

(10,10,30)
    ↓ flatten
(10,300)
    +
(10,34)
    ↓
(10,334 + categorical)

################
Transformer
################

(10,10,30)
    ↓
Player Tokens
    ↓
Attention
    │
    + Context
    ↓
Action Head