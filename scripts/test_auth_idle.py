import ast, asyncio, hmac, json
from datetime import datetime,timedelta,timezone
from pathlib import Path
from types import SimpleNamespace
from starlette.responses import JSONResponse
root=Path(__file__).resolve().parents[1]
tree=ast.parse((root/'butler.py').read_text(encoding='utf-8'))
names={'_secret_equal','_mk_access','_mk_refresh','auth_mw','v1_models','_v1_route_by_mode','v1_embeddings','v1_rerank'}
nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
for n in nodes:n.decorator_list=[]
count={};instances=[];active={};mode={'mode':'ON'};state={'start':'ok','post':'ok','authorized':True}
def bump(iid,d):
    count[iid]=count.get(iid,0)+d
    assert count[iid]>=0
async def start(inst):
    assert count[inst['id']]==1
    if state['start']=='cancel':raise asyncio.CancelledError()
    if state['start']=='error':raise RuntimeError('start failed')
    return state['start']=='ok'
async def auth(req):return state['authorized']
class Client:
    def __init__(self,**kw):pass
    async def __aenter__(self):return self
    async def __aexit__(self,*a):pass
    async def post(self,*a,**kw):
        assert count['a']==1
        if state['post']=='cancel':raise asyncio.CancelledError()
        if state['post']=='error':raise RuntimeError('upstream failed')
        return SimpleNamespace(json=lambda:{'data':[]},status_code=200)
class Request:
    def __init__(self,path='/api/models',token=None):self.url=SimpleNamespace(path=path);self.token=token
    async def json(self):return {'model':'model'}
ns=dict(asyncio=asyncio,hmac=hmac,datetime=datetime,timedelta=timedelta,timezone=timezone,Request=Request,
        JSONResponse=JSONResponse,load_instances=lambda:instances,get_active_containers=lambda:active,
        butler_state=mode,inflight_bump=bump,mark_active=lambda _:None,butler_start_instance=start,
        _v1_auth_ok=auth,httpx=SimpleNamespace(AsyncClient=Client),API_KEY='test',_get_token=lambda r:r.token,
        _verify=lambda token:token=='valid',jwt=SimpleNamespace(encode=lambda payload,*a,**kw:payload),
        JWT_SECRET='test',JWT_ALGO='HS256',JWT_ACCESS_MIN=1440,JWT_REFRESH_DAYS=30)
ns['_read_json_object'] = lambda request: request.json()
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<auth idle tests>','exec'),ns)
assert ns['_secret_equal']('пароль🔑','пароль🔑')
assert not ns['_secret_equal']('wrong','пароль🔑') and not ns['_secret_equal'](None,'test')
for name in ('_mk_access','_mk_refresh'):assert ns[name]('user')['exp'].tzinfo is timezone.utc
async def checks():
    async def next_handler(req):return JSONResponse({'ok':True})
    for path in ('/api/models','/api/models/dir'):
        assert (await ns['auth_mw'](Request(path),next_handler)).status_code==401
        assert (await ns['auth_mw'](Request(path,'valid'),next_handler)).status_code==200
    state['authorized']=False
    assert (await ns['v1_models'](Request())).status_code==401
    state['authorized']=True
    assert (await ns['v1_models'](Request())).status_code==200
    mode['mode']='OFF'
    inst,err=await ns['_v1_route_by_mode']('missing','embed')
    assert inst is None and err.status_code==404
    mode['mode']='ON'
    for route,model_mode in (('v1_embeddings','embed'),('v1_rerank','rerank')):
        instances[:]=[dict(id='a',name='model',model='model.gguf',mode=model_mode,port=123)]
        for outcome in ('ok','false','error','cancel'):
            state.update(start=outcome,post='ok')
            try:r=await ns[route](Request())
            except (RuntimeError,asyncio.CancelledError):assert outcome in ('error','cancel')
            else:assert r.status_code==(200 if outcome=='ok' else 503)
            assert count['a']==0
        state['start']='ok'
        for outcome in ('error','cancel'):
            state['post']=outcome
            try:await ns[route](Request())
            except (RuntimeError,asyncio.CancelledError):pass
            else:raise AssertionError('upstream failure expected')
            assert count['a']==0
        state['post']='ok';mode['mode']='OFF'
        assert (await ns[route](Request())).status_code==503 and count['a']==0
        active['a']={}
        assert (await ns[route](Request())).status_code==200 and count['a']==0
        instances[0]['allowed']=False
        assert (await ns[route](Request())).status_code==403 and count['a']==0
        active.clear();mode['mode']='ON'
asyncio.run(checks())
print('PASS: model-list auth, Unicode password comparison, UTC token expiry, both non-chat routes start/upstream success/failure/cancellation, OFF and permission gates; inflight always released')
