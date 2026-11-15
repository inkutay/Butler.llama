import ast
import asyncio
from functools import wraps
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

root=Path(__file__).resolve().parents[1]
tree=ast.parse((root/'butler.py').read_text(encoding='utf-8'))
names={'_llama_try_load','_poll_slots','_serialized_model_start','_start_from_calibration','_restart_with_ctx','_restore_instance','_restart_for_calibration'}
nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
calls=[]
def run(cmd,**kw):
    calls.append(cmd)
    return SimpleNamespace(returncode=0,stdout='running' if kw.get('text') else b'',stderr=b'')
ns=dict(asyncio=asyncio,wraps=wraps,os=os,time=SimpleNamespace(time=time.time,sleep=lambda _:None),
        subprocess=SimpleNamespace(run=run),DOCKER_IMAGE='runtime',LLAMA_SERVER='/app/llama-server',API_KEY='test',
        _docker_host_path=lambda p:p,_cuda_devices_str=lambda g:','.join(g),
        reset_speed_extrema=lambda _:None,mtp_flags=lambda m,n:['--mtp-test',m],
        httpx=SimpleNamespace(get=lambda *a,**kw:SimpleNamespace(status_code=200)))
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<review functions>','exec'),ns)
with tempfile.TemporaryDirectory() as tmp:
    ns['_current_models_dir']=tmp+'/default'
    custom=Path(tmp)/'custom';custom.mkdir()
    for name in ('vision.gguf','draft.gguf'):(custom/name).touch()
    assert ns['_llama_try_load']('test','model.gguf',8192,99,True,False,'q8_0',1,18001,[],mmproj='vision.gguf',mtp='draft.gguf',models_dir=str(custom))
    cmd=next(c for c in calls if c[:2]==['docker','run'])
    assert str(custom)+':/models:ro' in cmd and '--mmproj' in cmd and '--mtp-test' in cmd
    calls.clear()
    assert ns['_llama_try_load']('test','model.gguf',8192,99,True,False,'q8_0',1,18001,[])
    assert ns['_current_models_dir']+':/models:ro' in next(c for c in calls if c[:2]==['docker','run'])
instances=[dict(id='a',port=1),dict(id='b',port=2)]
ns['load_instances']=lambda:instances
polled=[]
ns['get_inst_slots']=lambda port:polled.append(port) or dict(active=1,total=2)
assert ns['_poll_slots']({'a':{}})=={'a':dict(active=1,total=2)} and polled==[1]
status=next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='api_status')
assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='get_inst_slots' for n in ast.walk(status))
autodetect=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_api_autodetect_sync')
assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_llama_try_load' and any(k.arg=='models_dir' for k in n.keywords) for n in ast.walk(autodetect))

async def check_threads():
    loop=asyncio.get_running_loop();ns['_main_loop']=loop;ns['_model_start_lock']=asyncio.Lock()
    entered=asyncio.Event();release=asyncio.Event();seen=[]
    @ns['_serialized_model_start']
    async def start(inst,stop_conflicts=True):
        assert asyncio.get_running_loop() is loop
        seen.append((inst['id'],stop_conflicts,inst.get('ctx')))
        if inst['id']=='http':
            entered.set();await release.wait()
        return True
    ns['butler_start_instance']=start
    http=asyncio.create_task(start({'id':'http'}));await entered.wait()
    inst=dict(id='a',ctx=16384)
    calls.clear()
    worker=asyncio.create_task(asyncio.to_thread(ns['_restart_with_ctx'],'a',inst,8192))
    await asyncio.sleep(.05)
    assert len(seen)==1 and not worker.done() and not calls
    release.set();await http
    assert await asyncio.wait_for(worker,3)
    ns['load_instances']=lambda:[inst]
    await asyncio.wait_for(asyncio.to_thread(ns['_restore_instance'],'a'),3)
    assert seen==[('http',True,None),('a',True,8192),('a',False,16384)]
    try: ns['_start_from_calibration'](inst)
    except RuntimeError:pass
    else:raise AssertionError('same-loop call must fail instead of deadlock')
    ns['_main_loop']=None
    try:await asyncio.to_thread(ns['_start_from_calibration'],inst)
    except RuntimeError:pass
    else:raise AssertionError('unavailable loop must fail')
asyncio.run(check_threads())
print('PASS: custom/default model directories and attachments; background slots; calibration restart/restore uses main loop and waits for HTTP start lock; shutdown/same-thread guards')
