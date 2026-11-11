"""No JAX dependency: package all surviving diagnostics after success/failure."""
import os,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]);target=root/'diagnostics.tar.gz';tmp=root/'diagnostics.tmp.tar.gz'
with tarfile.open(tmp,'w:gz') as t:
    for f in sorted(root.rglob('*')):
        if f.is_file() and f not in (target,tmp) and not f.name.endswith('.tmp'):
            t.add(f,arcname=str(f.relative_to(root)))
os.replace(tmp,target)
print('RESULT ARCHIVE:',target,flush=True)
