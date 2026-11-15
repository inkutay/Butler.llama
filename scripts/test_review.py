import ast
import asyncio
from functools import wraps
import re
import subprocess
from pathlib import Path
import time
from types import SimpleNamespace

root=Path(__file__).resolve().parents[1]
tree=ast.parse((root/'butler.py').read_text(encoding='utf-8'))
names={'_llama_try_load','_poll_slots','_serialized_model_start','_start_from_calibration','_restart_with_ctx','_restore_instance','_restart_for_calibration'}
nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
calls=[]
state = "running false"
stderr = ""
health = 200
launch_code = 0

def run(cmd, **kw):
    calls.append(cmd)
    if cmd[1] == "inspect":
        return SimpleNamespace(returncode=0, stdout=state)
    if cmd[1] == "logs":
        return SimpleNamespace(returncode=0, stdout="", stderr=stderr)
    return SimpleNamespace(returncode=launch_code if cmd[1] == "run" else 0)

ns=dict(asyncio=asyncio, wraps=wraps, re=re,
        time=SimpleNamespace(monotonic=time.monotonic, sleep=lambda _: None),
        subprocess=SimpleNamespace(run=run, TimeoutExpired=subprocess.TimeoutExpired),
        _stop_rm_container=lambda name: calls.append(["docker", "stop", name]),
        reset_speed_extrema=lambda _: None,
        httpx=SimpleNamespace(get=lambda *a, **kw: SimpleNamespace(status_code=health)))
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<review functions>','exec'),ns)
command = ["docker", "run"]
for state, stderr, health, launch_code, outcome in (
    ("running false", "", 200, 0, "ready"),
    ("running false", "", 503, 0, "timeout"),
    ("exited true", "", 503, 0, "oom"),
    ("exited false", "CUDA error: out of memory", 503, 0, "oom"),
    ("exited false", "cannot open model file", 503, 0, "failed"),
    ("running false", "", 503, 1, "docker_error"),
):
    calls.clear()
    assert ns['_llama_try_load']('test', 18001, command, wait_s=0) == outcome
    assert command in calls
    assert sum(c[:3] == ["docker", "rm", "-f"] for c in calls) == (0 if outcome == "ready" else 1)
instances=[dict(id='a',port=1),dict(id='b',port=2)]
ns['load_instances']=lambda:instances
polled=[]
ns['get_inst_slots']=lambda port:polled.append(port) or dict(active=1,total=2)
assert ns['_poll_slots']({'a':{}})=={'a':dict(active=1,total=2)} and polled==[1]
status=next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='api_status')
assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='get_inst_slots' for n in ast.walk(status))
autodetect=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_api_autodetect_sync')
assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_build_instance_command' and any(k.arg=='models_dir' for k in n.keywords) for n in ast.walk(autodetect))

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
print('PASS: trial outcomes and configured model directory; background slots; calibration restart/restore uses main loop and waits for HTTP start lock; shutdown/same-thread guards')
