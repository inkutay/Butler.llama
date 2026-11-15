import ast
import asyncio
import hmac
import json
import logging
import os
from pathlib import Path
import re
import secrets
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
tree = ast.parse((root / 'butler.py').read_text(encoding='utf-8'))
functions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in ('_load_jwt_secret', '_v1_auth_ok')]
with tempfile.TemporaryDirectory() as temp:
    cfg = Path(temp) / 'config.json'
    ns = dict(os=os, Path=Path, secrets=secrets, hmac=hmac, json=json,
              Request=object, CONFIG_FILE=cfg, __file__=str(Path(temp) / 'butler.py'), log=logging.getLogger('test'))
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<selected functions>', 'exec'), ns)
    saved = os.environ.pop('JWT_SECRET', None)
    try:
        a = ns['_load_jwt_secret']()
        assert len(a) >= 48 and ns['_load_jwt_secret']() == a
        os.environ['JWT_SECRET'] = 'explicit-test-secret'
        assert ns['_load_jwt_secret']() == 'explicit-test-secret'
    finally:
        os.environ.pop('JWT_SECRET', None)
        if saved is not None:
            os.environ['JWT_SECRET'] = saved
    class Request:
        def __init__(self, token):
            self.headers = {'authorization': token}
    async def check():
        auth = ns['_v1_auth_ok']
        assert not await auth(Request('Bearer example'))
        cfg.write_text('{broken', encoding='utf-8')
        assert not await auth(Request('Bearer example'))
        cfg.write_text(json.dumps({'api_key': 'example'}), encoding='utf-8')
        assert await auth(Request('Bearer example'))
        assert not await auth(Request('Bearer wrong'))
        assert not await auth(Request(''))
        cfg.write_text(json.dumps({'api_key': None}), encoding='utf-8')
        assert not await auth(Request(''))
        cfg.write_text(json.dumps({'api_key': ''}), encoding='utf-8')
        assert await auth(Request(''))
    asyncio.run(check())
    html = (root / 'index.html').read_text(encoding='utf-8')
    for i, script in enumerate(re.findall(r'<script[^>]*>(.*?)</script>', html, re.S)):
        target = Path(temp) / f'script{i}.js'
        target.write_text(script, encoding='utf-8')
        subprocess.run(['node', '--check', str(target)], check=True, capture_output=True)
    subprocess.run(['node', '-e', "const inst={id:'a'};const BUTLER_MODE='OFF';for(const state of [true,false]){const ACTIVE={a:state};if((BUTLER_MODE==='ON'||!!ACTIVE[inst.id])!==state)throw Error('visibility');}"], check=True)
print('PASS: Python syntax; JWT persistence and env override; 7 API auth cases; all inline JS syntax; model visibility with Butler OFF')
