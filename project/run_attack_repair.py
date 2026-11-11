"""One repair run: new teacher data -> supervised repair -> gates -> full matches."""
import argparse,atexit,csv,fcntl,json,os,subprocess,sys,tarfile
from pathlib import Path


def runtime_gate(before, after):
    return (int(after['lawful_team_possession_ticks']) >= max(10,int(before['lawful_team_possession_ticks'])+1)
            and int(after['shot_releases']) >= 1
            and int(after['goals_against']) <= int(before['goals_against'])+1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='project/experiments/attack_fulltime_seed0/bc')
    p.add_argument('--output',default='project/experiments/attack_repair_seed0')
    a=p.parse_args()
    if not Path('src/footballworld').is_dir():p.error('Run from MESSI root')
    os.environ['JAX_PLATFORMS']='cuda';os.environ.setdefault('JAX_THREEFRY_PARTITIONABLE','0')
    import jax
    print('Devices',jax.devices(),flush=True)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('This repair output is already in use')
    def bundle():
        with tarfile.open(out/'repair_results.tar.gz','w:gz') as tar:
            for pattern in ('*.json','*.log','**/results.csv'):
                for path in sorted(out.glob(pattern)):tar.add(path,arcname=str(path.relative_to(out)))
    atexit.register(bundle)
    def run(script,args,tag):
        command=[sys.executable,'-u','project/'+script]+list(map(str,args))
        print('RUN',command,flush=True)
        with (out/(tag+'.log')).open('a') as log:
            proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in proc.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            if proc.wait():raise SystemExit(f'{tag} failed; see repair_results.tar.gz')
    run('repair_attack_bc.py',['--source',a.source,'--output',out],'repair')
    diagnostics=json.loads((out/'diagnostics.json').read_text())
    rows={}
    for tag,path in [('before',Path(a.source)/'actor_best.pkl'),('after',out/'actor_best.pkl')]:
        folder=out/'control_probe'/tag
        run('evaluate_fulltime.py',['--checkpoint',path,'--output',folder,'--seeds',2000,'--max-steps',6400,'--deterministic'],'probe_'+tag)
        with (folder/'results.csv').open() as f:rows[tag]=next(csv.DictReader(f))
    live_pass=runtime_gate(rows['before'],rows['after'])
    passed=diagnostics['offline_gate_passed'] and live_pass
    status=dict(offline_passed=diagnostics['offline_gate_passed'],runtime_passed=live_pass,promoted=passed,
                probe=rows,probe_seed=2000,test_seeds=[2001,2002],
                note='Engineering gates; not evidence of globally optimal performance. Probe is bounded.')
    (out/'gate_status.json').write_text(json.dumps(status,indent=2))
    if passed:
        run('evaluate_fulltime.py',['--checkpoint',out/'actor_best.pkl','--output',out/'fulltime',
                                    '--seeds',2001,2002,'--deterministic'],'fulltime')
        status['fulltime_evaluation_completed']=True
    else:
        status['fulltime_evaluation_completed']=False
        print('GATE FAILED: no long PPO run or fulltime promotion. Inspect diagnostics and probe.',flush=True)
    (out/'gate_status.json').write_text(json.dumps(status,indent=2))
    bundle();print('SEND',out/'repair_results.tar.gz',flush=True)


if __name__=='__main__':main()
