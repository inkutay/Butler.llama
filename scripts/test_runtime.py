import ast
import asyncio
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from starlette.background import BackgroundTask
from starlette.responses import Response, JSONResponse, StreamingResponse

root = Path(__file__).resolve().parents[1]
tree = ast.parse((root / 'butler.py').read_text(encoding='utf-8'))
selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in ('_serialized_model_start', '_chat_stream_response', 'butler_start_instance')]
class RequestError(Exception): pass
class TimeoutException(RequestError): pass
class Upstream:
    def __init__(self, status=200):
        self.status_code = status
        self.headers = {'content-type': 'application/json' if status >= 400 else 'text/event-stream'}
        self.closed = 0
    async def aclose(self): self.closed += 1
    async def aread(self): return b'{"error":"model rejected request"}'
    async def aiter_bytes(self):
        yield b'data: first\n\n'
        yield b'data: [DONE]\n\n'
class Client:
    response = None
    error = None
    latest = None
    def __init__(self, **kw):
        Client.latest = self
        self.closed = 0
    def build_request(self, *a, **kw): return object()
    async def send(self, *a, **kw):
        if self.error: raise self.error()
        return self.response
    async def aclose(self): self.closed += 1
    async def __aenter__(self): return self
    async def __aexit__(self, *a): await self.aclose()
    async def get(self, *a): return SimpleNamespace(status_code=200)

async def main():
    count = {'inflight': 1}
    ns = dict(asyncio=asyncio, wraps=wraps, BackgroundTask=BackgroundTask, Response=Response,
              JSONResponse=JSONResponse, StreamingResponse=StreamingResponse,
              httpx=SimpleNamespace(AsyncClient=Client, RequestError=RequestError, TimeoutException=TimeoutException),
              _model_start_lock=asyncio.Lock(), inflight_bump=lambda iid, delta: count.update(inflight=count['inflight']+delta))
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<runtime functions>', 'exec'), ns)
    overlap = 0
    peak = 0
    @ns['_serialized_model_start']
    async def launch():
        nonlocal overlap, peak
        overlap += 1
        peak = max(peak, overlap)
        await asyncio.sleep(.01)
        overlap -= 1
    await asyncio.gather(*(launch() for _ in range(5)))
    assert peak == 1
    # Already-started instance: no Docker mutation, no second launch.
    ns['get_active_containers'] = lambda: {'same': {}}
    assert all(await asyncio.gather(*(ns['butler_start_instance']({'id':'same','port':123}) for _ in range(3))))
    for status in (200, 400, 429, 500):
        count['inflight'] = 1
        Client.response = Upstream(status)
        response = await ns['_chat_stream_response']('http://model', {}, {}, 'same')
        assert response.status_code == status
        if status < 400:
            assert count['inflight'] == 1
            assert b'[DONE]' in b''.join([part async for part in response.body_iterator])
            await response.background()
        else:
            assert b'model rejected' in response.body
        assert count['inflight'] == 0
        assert Client.latest.closed == Client.response.closed == 1
    # Disconnect before the full body has been consumed.
    count['inflight'] = 1
    Client.response = Upstream()
    response = await ns['_chat_stream_response']('http://model', {}, {}, 'same')
    await anext(response.body_iterator)
    await response.body_iterator.aclose()
    await response.background()
    assert count['inflight'] == 0 and Client.latest.closed == 1
    for error, expected in ((RequestError,502),(TimeoutException,504)):
        count['inflight'] = 1
        Client.error = error
        response = await ns['_chat_stream_response']('http://model', {}, {}, 'same')
        assert response.status_code == expected and count['inflight'] == 0
        assert Client.latest.closed == 1
    print('PASS: serialized launches, duplicate auto-start, stream 200/400/429/500, disconnect cleanup, 502/504')
asyncio.run(main())
