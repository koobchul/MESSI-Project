import importlib.util
from pathlib import Path

def load_trace(path):
    path=Path(path).resolve()
    spec=importlib.util.spec_from_file_location('_messi_shot_ppo_v2_local_trace',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if Path(module.__file__).resolve()!=path:raise RuntimeError('Wrong collector module')
    return module

def checked(collector):
    def wrapped(*args,**kwargs):
        result=collector(*args,**kwargs)
        if not isinstance(result,tuple) or len(result)!=3:
            raise RuntimeError('Evaluation collector must return exactly (carry,counts,rows), got '+str(type(result))+' length='+str(len(result)))
        return result
    return wrapped
