import ast,asyncio,copy,json,re,secrets,threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from starlette.responses import JSONResponse
root=Path(__file__).resolve().parents[1]
tree=ast.parse((root/'butler.py').read_text(encoding='utf-8'))
names={'InputError','_integer','_port_value','_unique_port','_instance_id','_gpu_value',
       '_validate_control_payload','_read_json_object','input_error_response','api_create',
       'api_rename','api_reporte','api_save_params','api_add_gpu','api_add_lora'}
nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
for node in nodes:
    if hasattr(node,'decorator_list'):node.decorator_list=[]
class Request:
    def __init__(self,body,path='/api/instances',raw=False):self.data=body;self.raw=raw;self.url=SimpleNamespace(path=path)
    async def json(self):return json.loads(self.data) if self.raw else self.data
store=[];lock=threading.RLock()
def save(instances):store[:]=copy.deepcopy(instances)
ns=dict(re=re,secrets=secrets,Request=Request,JSONResponse=JSONResponse,PORT=8090,
        _instances_lock=lock,load_instances=lambda:copy.deepcopy(store),save_instances=save,
        _resolve_ref=lambda inst,ref:'a.gguf' if ref in ('a.gguf','alias') else ref)
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<validation tests>','exec'),ns)
Error=ns['InputError']
async def bad(call):
    before=copy.deepcopy(store)
    try:await call
    except Error as error:
        response=await ns['input_error_response'](Request({}),error)
        assert response.status_code==400 and json.loads(response.body)['message']
    else:raise AssertionError('expected HTTP 400')
    assert store==before,'invalid request mutated state'
async def checks():
    for raw in ('{broken','[]','null','123','true','"text"'):
        await bad(ns['api_create'](Request(raw,raw=True)))
    invalid=[{'name':None},{'name':'x','port':True},{'name':'x','port':1.5},
             {'name':'x','port':'abc'},{'name':'x','port':-1},{'name':'x','port':65536},
             {'name':'x','port':8090},{'name':'x','id':'bad/id'},
             {'name':'x','id':'x'*65},{'name':'x','gpu_indices':['GPU0',0]},
             {'name':'x','gpu_indices':'GPU0'},{'name':'x','gpu_indices':[True]}]
    for body in invalid:await bad(ns['api_create'](Request(body)))
    display='Моя модель "<img src=x onerror=alert(1)>" & тест'
    await ns['api_create'](Request({'name':display,'port':'18001'}))
    iid=store[0]['id'];assert store[0]['name']==display and ns['_instance_id'](iid)==iid
    await ns['api_create'](Request({'name':'legacy-safe','port':0}))
    assert store[1]['id']=='legacy-safe'
    for port in (0,1,65535,''):
        await ns['api_reporte']('legacy-safe',Request({'port':port}))
    await bad(ns['api_reporte']('legacy-safe',Request({'port':18001})))
    await bad(ns['api_rename']('legacy-safe',Request({'name':'new','port':18001})))
    for key,value in (('ctx',-1),('parallel',0),('ngl',-1),('mtp_nmax',-1),('ctx','no'),('parallel',True),('flash','false'),('models_dir',[])):
        await bad(ns['api_save_params'](iid,Request({key:value})))
    await ns['api_save_params'](iid,Request({'ctx':'0','parallel':'1','ngl':0,'mtp_nmax':0,'models_dir':'/'+'x'*1000}))
    assert store[0]['ctx']==0 and len(store[0]['models_dir'])>512
    await ns['api_add_gpu'](iid,Request({'gpu_idx':'GPU0'}))
    await bad(ns['api_add_gpu'](iid,Request({'gpu_idx':'0'})))
    await bad(ns['api_add_gpu'](iid,Request({})))
    await ns['api_add_gpu'](iid,Request({'gpu_idx':'__select__'}))
    await bad(ns['api_add_gpu'](iid,Request({'gpu_idx':'__select__'})))
    await ns['api_add_lora'](iid,Request({'lora':'a.gguf'}))
    await bad(ns['api_add_lora'](iid,Request({'lora':'alias'})))
    for body in ({'path':[]},{'api_key':None},{'free_vram_gb':'nan'},{'loras':['a','a']}):
        await bad(ns['_read_json_object'](Request(body)))
asyncio.run(checks())
# Multiple real threads/loops compete for one port: exactly one may persist it.
def compete(number):
    try:
        asyncio.run(ns['api_create'](Request({'name':f'racer{number}','port':19000})))
        return True
    except Error:return False
with ThreadPoolExecutor(max_workers=8) as pool:assert sum(pool.map(compete,range(8)))==1
assert sum(inst['port']=='19000' for inst in store)==1
print('PASS: malformed JSON/types/numbers -> 400 without state changes; Unicode names/safe IDs; no path cap; atomic port contention; zero ports; GPU/LoRA duplicates')
