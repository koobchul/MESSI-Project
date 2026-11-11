# 1050651 결과와 다음 실행
두 작업 모두clock preflight에서실패/PPO업데이트0/성능결과없음.
원인: worker가base경로를먼저sys.path에추가한뒤audit_trace를이름으로import하여base의5반환수집기를선택.
v2: 패치경로의모듈을고유이름으로명시로드,3반환계약검사,동명이름충돌회귀검사추가.
기존1050651폴더/체크포인트/환경소스는수정하지않는다.

1. 먼저수정PPO배열(control vs shot_reward)을제출.
두군동일fieldSHOTbias+5초기화/Adam리셋/4096업데이트,실제유용슛보상0.02vs0.10만상이.
탐색초기화는그자체PPO개선아님.원본/초기화직후/학습후비교.
PPO0회로실패했으므로이번결과로보상0.10의효과를판단할수없음.
각최대12시간/GPU1,동시2작업.

2. 여유자원으로HOME전술배열을별도제출가능.
HOME기본/게겐프레싱/카테나치오,AWAY기본고정.동결weak5263정책과규칙슈팅개입유무비교.
HOME필드이동은규칙전술prior+PPO잔차이며순수학습전술로부르지않음.
기존CONTROL settle기준파라미터도전술변경에따라간접변화가능.
학습정책의의도를전체룰정책으로교체하는것은아님.시작배치만바꾸는것도아님.
각6풀타임/최대4시간,기본동시1개.3개합쳐최대12GPU시간.
최대동시GPU는PPO2+HOME1=3,계정한도/자원에따라대기.

실행명령(MESSI루트):
tar -xzf messi11_next_stage_v2.tar.gz
PPO_JOB=$(sbatch --parsable messi11_shot_ppo_v2_patch/run_array.sbatch)
echo "$PPO_JOB"
HOME_JOB=$(sbatch --parsable messi11_home_tactics_patch/run_array.sbatch)
echo "$HOME_JOB"
squeue -u "$USER" -o "%.20i %.10T %.12M %.12L %.28j %.25R"
tail -n 30 -F "messi11-shot-ppo-v2-${PPO_JOB}_0.log"
tail -n 30 -F "messi11-home-tactics-${HOME_JOB}_0.log"
PPO의 다른 로그는 _1, HOME의 다른 로그는 _1/_2.

결과:
project/experiments/messi11_shot_ppo_v2_ARRAYID/control/summary_light.tar.gz
project/experiments/messi11_shot_ppo_v2_ARRAYID/shot_reward/summary_light.tar.gz
project/experiments/messi11_home_tactics_ARRAYID/juego_de_posicion/summary_light.tar.gz
project/experiments/messi11_home_tactics_ARRAYID/gegenpress/summary_light.tar.gz
project/experiments/messi11_home_tactics_ARRAYID/catenaccio/summary_light.tar.gz
HOME배열의폴더이름은HOME전술명.각opponent_config.json에서실제양팀설정확인.

현재진척:11v11경로/GK분리/이동제약/룰이동+잔차/실제release구분완료.
규칙SHOT+킥효과는두시드묶음18경기18승93:5.하지만학습슈팅성과아님.
이번PPO증거는없음.필드자체패스부족지속.최종정책확정/학습상대토너먼트일반화미완료.
보상은득실±1/정상점유0.02/접근유지전진/실제슛0.02또는0.10/전진패스0.002/지원과밀/GK지역보상유지.
요청보상없음.슛보상의골문방향proxy는실제유효슈팅판정이아님.

로컬회귀/변환/집계/Python/sbatch문법확인.실제JAX/GPU실행은서버preflight로확인필요.
최종규칙사용허용범위는과제규정에서아직확정되지않았으며학습효과와분리해보고할것.
