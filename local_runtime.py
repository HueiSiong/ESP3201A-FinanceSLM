"""Only locate DLLs installed into the current interpreter's own environment."""
import os,sys
from pathlib import Path
_handles=[]
def configure():
    if os.name=='nt':
        folder=Path(sys.prefix)/'Library/bin'
        if folder.is_dir():
            _handles.append(os.add_dll_directory(str(folder)))
            os.environ['PATH']=str(folder)+os.pathsep+os.environ.get('PATH','')
