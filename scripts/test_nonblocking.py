import ast,asyncio,json,os,shutil,tempfile,threading,time
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from starlette.responses import JSONResponse
root=Path(__file__).resolve().parents[1];source=(root/'butler.py').read_text(encoding='utf-8');tree=ast.parse(source)
names={'_run_blocking','_serialized_model_start','api_gpus','_api_gpus_sync','_fs_path','api_fs_copy','_api_fs_copy_sync','api_autodetect','_api_autodetect_sync'}
nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
for node in nodes:node.decorator_list=[d for d in node.decorator_list if isinstance(d,ast.Name) and d.id=='_serialized_model_start']
class Request:
    def __init__(self,body):self.body=body
    async def json(self):return self.body
ns=dict(asyncio=asyncio,wraps=wraps,JSONResponse=JSONResponse,Request=Request,os=os,shutil=shutil,
        get_gpu_list=lambda:(time.sleep(.15) or [],{}),_instances_lock=threading.RLock(),
        log=SimpleNamespace(info=lambda _:None),_model_start_lock=asyncio.Lock())
ns['_read_json_object'] = lambda request: request.json()
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<nonblocking tests>','exec'),ns)
async def main():
    async def responsive(call):
        task=asyncio.create_task(call);await asyncio.sleep(.03)
        # If the handler blocked the loop, it would already have completed here.
        assert not task.done(),'blocking work froze the event loop'
        result=await task;assert result.status_code==200
        return result
    await responsive(ns['api_gpus']())
    with tempfile.TemporaryDirectory() as tmp:
        src=Path(tmp)/'src';src.mkdir();(src/'file.txt').write_text('unchanged content')
        dest=Path(tmp)/'dest';dest.mkdir()
        real_copy=shutil.copytree
        def slow_copy(*a,**kw):time.sleep(.15);return real_copy(*a,**kw)
        ns['shutil']=SimpleNamespace(copytree=slow_copy,copy2=shutil.copy2,disk_usage=shutil.disk_usage)
        response=await responsive(ns['api_fs_copy'](Request({'path':str(src),'dest':str(dest)})))
        assert json.loads(response.body)['success'] and (dest/'src/file.txt').read_text()=='unchanged content'
        model=Path(tmp)/'model.gguf';model.touch();inst=dict(id='a',model='model.gguf',port=123)
        attempts=[]
        ns.update(load_instances=lambda:[dict(inst)],inst_models_dir=lambda _:tmp,
                  get_cached_model_params=lambda *a:{'context_length':16384},
                  save_instances=lambda _:None)
        def try_load(*args,**kw):attempts.append(args[2]);return False
        ns['_llama_try_load']=try_load
        r=await ns['api_autodetect']('a',Request({}))
        assert not json.loads(r.body)['success'] and attempts==[16384,8192]
    # A cancelled request must retain the shared lifecycle lock until its thread ends.
    started=threading.Event();finish=threading.Event()
    def work():started.set();finish.wait(3);return True
    @ns['_serialized_model_start']
    async def guarded():return await ns['_run_blocking'](work)
    task=asyncio.create_task(guarded());await asyncio.to_thread(started.wait,1)
    task.cancel();await asyncio.sleep(.02)
    assert ns['_model_start_lock'].locked() and not task.done()
    finish.set()
    try:await task
    except asyncio.CancelledError:pass
    else:raise AssertionError('cancellation must propagate')
    assert not ns['_model_start_lock'].locked()
    assert await guarded()
asyncio.run(main())
assert 'http_client' not in source and 'BACKEND_URL' not in source
for name in ('api_check','api_apply_lora','api_fs_list','api_fs_mkdir','api_fs_rename','api_fs_move','api_fs_copy','api_fs_delete','api_autodetect','api_gpus'):
    node=next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name==name)
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_run_blocking' for n in ast.walk(node))
print('PASS: responsive event loop during slow GPU/copy operations, copy result unchanged, bounded autodetect failures, cancellation retains start lock, all 10 wrappers offload, no unused client')
