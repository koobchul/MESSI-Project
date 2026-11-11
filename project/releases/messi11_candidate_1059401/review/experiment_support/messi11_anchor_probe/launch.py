import argparse,json,os,subprocess,sys,traceback
from pathlib import Path
from prepare import prepare
HERE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--arm',choices=['control','anchored'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
 root=a.output;root.mkdir(parents=True,exist_ok=False)
 try:
  source,manifest=prepare(root)
  os.environ['MESSI_POOL_MANIFEST']=str(manifest);os.environ['MESSI_POOL_MODE']='mixed_pool'
  os.environ['MESSI_APPROACH_SCALE']='.1'
  os.environ['MESSI_ANCHOR_COEF']='0' if a.arm=='control' else '.02'
  cmd=[sys.executable,'-u',str(HERE/'worker.py'),'--base',str(source),'--source',str(Path('project/experiments/messi11_possession_balance_1040542/weak_acquire/final_train.pkl').resolve()),'--output',str(root/'runtime'),'--arm','mixed_pool','--train-seed',str(a.seed),'--budget','26700','--updates','1024']
  if a.preflight_only:cmd.append('--preflight-only')
  subprocess.run(cmd,check=True,timeout=27300)
  if a.preflight_only:
   status=json.loads((root/'runtime/status.json').read_text())
   if status.get('stage')!='preflight_passed':raise RuntimeError('Gate did not pass')
  else:
   status=json.loads((root/'runtime/status.json').read_text())
   if not status.get('training_complete') or status.get('updates')!=1024:raise RuntimeError('Training incomplete')
   (root/'training_status.json').write_text(json.dumps(status,indent=2))
 except BaseException:
  (root/'failure.txt').write_text(traceback.format_exc());raise
 finally:subprocess.run([sys.executable,str(HERE/'archive.py'),str(root)],check=False)
if __name__=='__main__':main()
