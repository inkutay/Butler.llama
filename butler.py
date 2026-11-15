import os, re, time, json, struct, hashlib, logging, threading, subprocess, shutil, asyncio, secrets, hmac
from functools import wraps
from starlette.background import BackgroundTask
from pathlib import Path
import tempfile
from contextlib import asynccontextmanager, nullcontext
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response, HTMLResponse, JSONResponse, StreamingResponse, FileResponse
import jwt
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel
from typing import List, Optional

if os.name == "nt":
    _ORIG_POPEN = subprocess.Popen
    class _NoWinPopen(_ORIG_POPEN):
        def __init__(self, *a, **kw):
            kw.setdefault("creationflags", 0)
            kw["creationflags"] = kw["creationflags"] | 0x08000000  # CREATE_NO_WINDOW
            super().__init__(*a, **kw)
    subprocess.Popen = _NoWinPopen

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8090"))
STATIC_DIR = Path(__file__).parent / "static"
CONFIG_FILE = Path(__file__).parent / "butler_config.json"
INSTANCES_FILE = Path(__file__).parent / "butler_instances.json"
MODELS_DIR_DEFAULT = str(Path(__file__).resolve().parent / "models")
DOCKER_IMAGE = "ghcr.io/ggml-org/llama.cpp:server-cuda"
LLAMA_SERVER = "/app/llama-server"
API_KEY = "butler-local"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("butler")

class InputError(ValueError):
    pass


def _integer(value, field, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise InputError(f"{field}: expected an integer")
    if isinstance(value, str) and not re.fullmatch(r"[+-]?[0-9]+", value.strip()):
        raise InputError(f"{field}: expected an integer")
    try:
        result = int(value)
    except (ValueError, OverflowError):
        raise InputError(f"{field}: invalid integer") from None
    if result < minimum:
        raise InputError(f"{field}: must be >= {minimum}")
    return result


def _port_value(value):
    port = 0 if value == "" else _integer(value, "port")
    if port > 65535:
        raise InputError("port: must be 0 or 1..65535")
    if port == PORT:
        raise InputError("port: reserved by Butler")
    return str(port)


def _unique_port(port, insts, exclude_id=None):
    port = _port_value(port)
    if port != "0":
        for inst in insts:
            if inst.get("id") == exclude_id:
                continue
            try:
                existing = int(inst.get("port") or 0)
            except (TypeError, ValueError):
                continue
            if existing == int(port):
                raise InputError(f"port: {port} is already assigned to another instance")
    return port


def _instance_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value):
        raise InputError("id: use 1..64 characters, starting with a letter or digit; allowed: A-Z a-z 0-9 _ . -")
    return value


def _gpu_value(value):
    if value == "__select__":
        return value
    if isinstance(value, str):
        value = value.removeprefix("GPU")
    return "GPU" + str(_integer(value, "gpu_idx"))


def _validate_control_payload(body):
    body = dict(body)
    for key, minimum in {"ctx": 0, "ctx_size": 0, "max_ctx": 0, "parallel": 1,
                         "ngl": 0, "mtp_nmax": 0, "idle_timeout_minutes": 0, "num_gpus": 1}.items():
        if key in body:
            body[key] = _integer(body[key], key, minimum)
    for key in ("name", "model", "mmproj", "mtp", "models_dir", "docker_image", "path", "dest",
                "new_name", "api_key", "api_url", "dir", "direction", "lora", "kvq", "mode",
                "username", "password", "current_password", "new_password", "refresh_token", "instance"):
        if key in body and not isinstance(body[key], str):
            raise InputError(f"{key}: expected a string")
    for key in ("flash", "ram", "flash_attn", "ram_kv", "allowed", "idle_exempt", "idle_enabled"):
        if key in body and not isinstance(body[key], bool):
            raise InputError(f"{key}: expected true or false")
    if "free_vram_gb" in body:
        value = body["free_vram_gb"]
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise InputError("free_vram_gb: expected a non-negative number")
        try:
            value = float(value)
            if not __import__("math").isfinite(value) or value < 0:
                raise ValueError()
        except (ValueError, OverflowError):
            raise InputError("free_vram_gb: expected a non-negative finite number") from None
        body["free_vram_gb"] = value
    if "port" in body:
        body["port"] = _port_value(body["port"])
    if "gpu_idx" in body:
        body["gpu_idx"] = _gpu_value(body["gpu_idx"])
    if "gpu_indices" in body:
        values = body["gpu_indices"]
        if not isinstance(values, list):
            raise InputError("gpu_indices: expected an array")
        values = [_gpu_value(value) for value in values]
        if len(values) != len(set(values)):
            raise InputError("gpu_indices: duplicate GPU entries")
        body["gpu_indices"] = values
    if "loras" in body:
        values = body["loras"]
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise InputError("loras: expected an array of strings")
        if len(values) != len(set(values)):
            raise InputError("loras: duplicate entries")
    if "kvq" in body and body["kvq"] not in ("fp16", "q8_0", "q4_0"):
        raise InputError("kvq: expected fp16, q8_0 or q4_0")
    if "mode" in body and body["mode"] not in ("ON", "OFF", "chat", "embed", "rerank"):
        raise InputError("mode: invalid value")
    return body


async def _read_json_object(request):
    try:
        body = await request.json()
    except (ValueError, UnicodeError):
        raise InputError("Invalid JSON body") from None
    if not isinstance(body, dict):
        raise InputError("JSON body must be an object")
    path = getattr(getattr(request, "url", None), "path", "")
    if path.startswith("/v1/"):
        if "model" in body and not isinstance(body["model"], str):
            raise InputError("model: expected a string")
        return body
    return _validate_control_payload(body)


def load_config():
    cfg = {"api_url": "http://127.0.0.1:8090/v1", "api_key": "", "models_dir": MODELS_DIR_DEFAULT, "idle_enabled": False, "idle_timeout_minutes": 30}
    if CONFIG_FILE.exists():
        try:
            c = json.loads(CONFIG_FILE.read_text("utf-8"))
            cfg.update(c)
        except: pass
    return cfg

def save_config(cfg):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=CONFIG_FILE.parent,
                                         prefix=CONFIG_FILE.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(cfg, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, CONFIG_FILE)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

config = load_config()
_current_models_dir = config.get("models_dir") or MODELS_DIR_DEFAULT
API_KEY = config.get("internal_api_key") or API_KEY

def _docker_host_path(p):
    """Windows host path -> WSL2 path for docker volume mounts (no-op on Linux)."""
    if os.name == "nt" and re.match(r"^[A-Za-z]:[\\/]", p or ""):
        return "/mnt/" + p[0].lower() + p[2:].replace("\\", "/")
    return p

# Runtime overhead not covered by model weights, subtracted from the KV budget in /api/calculate:
# CUDA context + base compute buffers (fixed) and a small per-slot term. Kept modest: the dominant
# per-token cost is the KV cache, captured by kvpt (prefer the measured sidecar over the formula).
OH_FIXED_MIB = 650      # CUDA context + base compute buffers
OH_SLOT_MIB = 128       # per parallel slot

_instances_lock = threading.RLock()

def load_instances():
    with _instances_lock:
        if INSTANCES_FILE.exists():
            try: return json.loads(INSTANCES_FILE.read_text("utf-8"))
            except: pass
    return []

def save_instances(insts):
    with _instances_lock:
        for inst in insts:
            inst.pop("slots", None)
        tmp = INSTANCES_FILE.with_name(INSTANCES_FILE.name + ".tmp")
        tmp.write_text(json.dumps(insts, indent=2), "utf-8")
        os.replace(tmp, INSTANCES_FILE)
    try:
        _status_cache["ts"] = 0.0
    except NameError:
        pass

# In-flight (proxied) chat requests per instance — for queue estimation
_inflight = {}
_inflight_lock = threading.Lock()

def inflight_bump(iid, delta=1):
    with _inflight_lock:
        _inflight[iid] = max(0, _inflight.get(iid, 0) + delta)

_inflight_since = {}   # iid -> ts when (inflight>active) first observed

# Last real activity (chat) per instance — for idle auto-unload
_last_active = {}
def mark_active(iid):
    _last_active[iid] = time.time()

def inflight_queued(iid, active):
    """Queued = inflight requests minus active slots, but ignore brief
    transition states (request starting/finishing) — real queue only after 3s."""
    with _inflight_lock:
        infl = _inflight.get(iid, 0)
        diff = infl - active
        if diff <= 0:
            _inflight_since.pop(iid, None)
            return 0
        now = time.time()
        if iid not in _inflight_since:
            _inflight_since[iid] = now
        if now - _inflight_since[iid] < 3.0:
            return 0
        return diff

_last_slots = {}   # iid -> last successful slots snapshot (keep display stable when /slots is slow)

# Parallel slots usage from llama-server
def get_inst_slots(port, timeout=1.5):
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/slots", headers={"Authorization": f"Bearer {API_KEY}"}, timeout=timeout)
        if r.status_code == 200:
            slots = r.json()
            return {"active": sum(1 for s in slots if s.get("is_processing")), "total": len(slots)}
    except Exception:
        pass
    return None

def _poll_slots(active):
    """Blocking slot reads run only in the background updater."""
    slots = {}
    for inst in load_instances():
        iid = inst.get("id")
        if iid in active and inst.get("port"):
            sl = get_inst_slots(inst["port"])
            if sl is not None:
                slots[iid] = sl
    return slots


# Generation speed: delta of /slots next_token.n_decoded between samples.
# NOTE: /metrics tokens_predicted_total updates only at request end in this llama.cpp build
# (batched) — deltas over 5s polls would spike (e.g. 2637 tok / 5s = 531 t/s for a 20 t/s request).
# A 1s background sampler catches even short (<5s) requests.
_speed_state = {}   # iid -> (task_key, decoded, ts)
_speed_live = {}    # iid -> (spd, ts)
_speed_extrema = {} # iid -> (min, max) measured since last instance start
_speed_samples = {} # iid -> (sum, count) of counted speed samples since last instance start

def reset_speed_extrema(iid):
    """Reset speed stats on instance (re)start: live, extrema, samples, and persisted sticky values."""
    _speed_extrema.pop(iid, None)
    _speed_samples.pop(iid, None)
    _speed_state.pop(iid, None)
    _speed_live.pop(iid, None)
    try:
        insts = load_instances()
        changed = False
        for inst in insts:
            if inst.get("id") == iid:
                if inst.get("min_speed_tps") or inst.get("max_speed_tps") or inst.get("last_speed_tps") or inst.get("speed_count"):
                    inst["min_speed_tps"] = 0
                    inst["max_speed_tps"] = 0
                    inst["last_speed_tps"] = 0
                    inst["speed_sum"] = 0
                    inst["speed_count"] = 0
                    changed = True
        if changed:
            save_instances([dict(i) for i in insts])
    except Exception:
        pass

def _calc_speed(iid, port, timeout=2.0):
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/slots", headers={"Authorization": f"Bearer {API_KEY}"}, timeout=timeout)
        if r.status_code != 200:
            return
        slots = r.json()
        tasks = []
        dec = 0
        for s in slots:
            if s.get("is_processing"):
                tasks.append(str(s.get("id_task")))
                for nt in s.get("next_token") or []:
                    dec += nt.get("n_decoded") or 0
        now = time.time()
        key = tuple(sorted(tasks))
        prev = _speed_state.get(iid)
        if key and prev and prev[0] == key and dec > prev[1]:
            nd = dec - prev[1]
            dt = now - prev[2]
            # Count only "full" samples: >=5 tokens in the window and not the task's
            # first delta (which includes prompt-eval/start time). Short replies
            # (e.g. "привет") and slow starts don't pollute SPEED/MIN/MAX/AVG.
            if dt > 0.2 and nd >= 5 and prev[1] > 0:
                spd = round(nd / dt, 1)
                _speed_live[iid] = (spd, now)
                cur = _speed_extrema.get(iid)
                _speed_extrema[iid] = (min(cur[0], spd) if cur else spd, max(cur[1], spd) if cur else spd)
                ss, cc = _speed_samples.get(iid, (0.0, 0))
                _speed_samples[iid] = (ss + spd, cc + 1)
                log.info(f"speed {iid}: decoded {prev[1]}->{dec} over {dt:.2f}s -> {spd} t/s (tasks={list(key)})")
        if key:
            _speed_state[iid] = (key, dec, now)
    except Exception:
        pass

def speed_for_status(iid):
    s, ts = _speed_live.get(iid, (0.0, 0.0))
    if s > 0 and time.time() - ts < 3.0:
        return s
    return 0

def _speed_sampler():
    while True:
        try:
            active = _bg_cache.get("active", {})
            if active:
                for inst in load_instances():
                    iid = inst.get("id")
                    if iid in active and inst.get("port"):
                        _calc_speed(iid, inst.get("port"))
        except Exception:
            pass
        time.sleep(1.0)

# Active instances from Docker
def get_active_containers(*, strict=False):
    try:
        r = subprocess.run(["docker", "ps", "--filter", "name=llama-inst-", "--format", "{{.Names}}"], capture_output=True, timeout=10, text=True)
        if strict:
            r.check_returncode()
        active = {}
        for line in r.stdout.strip().split("\n"):
            if not line.strip(): continue
            name = line.strip().removeprefix("llama-inst-")
            active[name] = {"state": "ON"}
        return active
    except Exception:
        if strict:
            raise
        return {}

def get_failed_containers():
    """Crashed llama containers: in `docker ps -a`, status exited, ExitCode != 0.
    Clean stop = exit 0 (or container already rm'd) -> NOT listed. UI 'FAIL' ground truth."""
    failed = {}
    try:
        r = subprocess.run(["docker", "ps", "-a", "--filter", "name=llama-inst-", "--filter", "status=exited",
                           "--format", "{{.Names}}"], capture_output=True, timeout=10, text=True)
        names = [n.strip() for n in r.stdout.strip().split("\n") if n.strip()]
        if not names:
            return failed
        r2 = subprocess.run(["docker", "inspect", "--format", "{{.Name}}|{{.State.ExitCode}}"] + names,
                           capture_output=True, timeout=15, text=True)
        for line in r2.stdout.strip().split("\n"):
            if not line.strip(): continue
            name, _, code = line.strip().partition("|")
            iid = name.lstrip("/").replace("llama-inst-", "")
            try:
                ec = int(code.strip())
            except Exception:
                ec = -1
            if ec != 0:
                failed[iid] = ec
    except Exception:
        pass
    return failed

def _probe_ready(active):
    """For each active container: is the MODEL actually loaded? (llama-server /health < 500).
    Container running != model ready (load takes seconds..minutes)."""
    ready = {}
    if not active:
        return ready
    insts = load_instances()
    for iid in active:
        inst = next((x for x in insts if x["id"] == iid), None)
        port = inst.get("port") if inst else None
        if not port or str(port) == "0":
            continue
        try:
            hr = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2)
            ready[iid] = hr.status_code == 200
        except Exception:
            ready[iid] = False
    return ready

# Idle auto-unload: active model with no chat for N minutes (and not generating) gets stopped
def _idle_checker():
    while True:
        try:
            c = load_config()
            if c.get("idle_enabled"):
                mins = int(c.get("idle_timeout_minutes", 0) or 0)
                if mins > 0:
                    timeout = mins * 60
                    now = time.time()
                    active = get_active_containers()
                    _inst_map = {i["id"]: i for i in load_instances()}
                    for iid in list(active.keys()):
                        if _inst_map.get(iid, {}).get("idle_exempt"):
                            continue  # instance marked ∞ — excluded from idle-unload
                        if _inflight.get(iid, 0) > 0:
                            continue  # generating — never unload mid-request
                        if iid not in _last_active:
                            _last_active[iid] = now  # baseline = first observation
                        last = _last_active[iid]
                        if now - last > timeout:
                            log.info(f"idle-unload {iid}: idle {int((now-last)//60)} min > {mins} min")
                            _stop_rm_container(f"llama-inst-{iid}")
                            _last_active.pop(iid, None)
        except Exception as e:
            log.error(f"idle_checker error: {e}")
        time.sleep(30)

# LAN IP — to display the address to reach Butler from the network
_lan_ip_cache = {"ip": "", "ts": 0.0}
def get_lan_ip():
    now = time.time()
    if _lan_ip_cache["ip"] and now - _lan_ip_cache["ts"] < 300:
        return _lan_ip_cache["ip"]
    ip = _lan_ip_windows() or _lan_ip_linux()
    _lan_ip_cache["ip"] = ip or _lan_ip_cache["ip"]
    _lan_ip_cache["ts"] = now
    return _lan_ip_cache["ip"]

def _lan_ip_windows():
    """WSL/Windows: parse ipconfig.exe, exclude WSL-NAT + loopback."""
    import re
    wsl_prefix = ""
    try:
        out = subprocess.run(["ip", "-4", "route", "show", "default"], capture_output=True, text=True, timeout=5).stdout
        m = re.search(r'via (\d+\.\d+)', out)
        if m: wsl_prefix = m.group(1)
    except Exception:
        pass
    try:
        r = subprocess.run(["/mnt/c/Windows/System32/ipconfig.exe"], capture_output=True, timeout=10)
        data = r.stdout
        for ipb in re.findall(rb'IPv4[^0-9]*?(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', data):
            cand = ipb.decode("ascii")
            if cand.startswith("127."):
                continue
            if wsl_prefix and cand.startswith(wsl_prefix + "."):
                continue
            return cand
    except Exception:
        pass
    return ""

def _lan_ip_linux():
    """Bare Linux: hostname -I, exclude loopback + docker/WSL private range."""
    import re
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=5).stdout
        cands = [t for t in out.split() if re.match(r'^\d+\.\d+\.\d+\.\d+$', t)]
        for ip in cands:
            if ip.startswith("127."):
                continue
            p = ip.split(".")
            if p[0] == "172" and 16 <= int(p[1]) <= 31:
                continue  # docker bridge / WSL NAT
            return ip
    except Exception:
        pass
    return ""

# GGUF Reader
def read_gguf_metadata(fpath):
    with open(fpath, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF': raise ValueError("Not GGUF")
        struct.unpack('<I', f.read(4))[0]
        struct.unpack('<Q', f.read(8))[0]
        kv_count = struct.unpack('<Q', f.read(8))[0]
        def read_str():
            ln = struct.unpack('<Q', f.read(8))[0]
            return f.read(ln).decode('utf-8', errors='replace')
        def read_val(vtype):
            if vtype == 0: return struct.unpack('<B', f.read(1))[0]
            if vtype == 1: return struct.unpack('<b', f.read(1))[0]
            if vtype == 2: return struct.unpack('<H', f.read(2))[0]
            if vtype == 3: return struct.unpack('<h', f.read(2))[0]
            if vtype == 4: return struct.unpack('<I', f.read(4))[0]
            if vtype == 5: return struct.unpack('<i', f.read(4))[0]
            if vtype == 6: return struct.unpack('<f', f.read(4))[0]
            if vtype == 7: f.read(1); return None
            if vtype == 8: return read_str()
            if vtype == 9:
                atype = struct.unpack('<I', f.read(4))[0]
                n = struct.unpack('<Q', f.read(8))[0]
                return [read_val(atype) for _ in range(n)]
            if vtype == 10: return struct.unpack('<Q', f.read(8))[0]
            if vtype == 11: return struct.unpack('<q', f.read(8))[0]
            if vtype == 12: return struct.unpack('<d', f.read(8))[0]
            return None
        meta = {}
        for _ in range(kv_count):
            key = read_str()
            vtype = struct.unpack('<I', f.read(4))[0]
            meta[key] = read_val(vtype)
        return meta

def get_model_params(fpath):
    meta = read_gguf_metadata(fpath)
    name = os.path.basename(fpath)
    size_gb = round(os.path.getsize(fpath) / 1024**3, 2)
    arch = meta.get('general.architecture', 'unknown')
    n_embd = meta.get(f'{arch}.embedding_length', 0)
    n_layers = meta.get(f'{arch}.block_count', 0)
    nextn = meta.get(f'{arch}.nextn_predict_layers', 0) or 0
    main_layers = max(0, n_layers - nextn)
    n_head = meta.get(f'{arch}.attention.head_count', 0)
    n_kv_heads = meta.get(f'{arch}.attention.head_count_kv', 0)
    cl = meta.get(f'{arch}.context_length', 0)
    fa_interval = meta.get(f'{arch}.full_attention_interval', 0) or 0
    n_expert = meta.get(f'{arch}.expert_count', 0)
    n_expert_used = meta.get(f'{arch}.expert_used_count', 0)
    key_len = meta.get(f'{arch}.attention.key_length', 0)
    val_len = meta.get(f'{arch}.attention.value_length', 0)
    ssm_state = meta.get(f'{arch}.ssm.state_size', 0) or 0
    ssm_groups = meta.get(f'{arch}.ssm.group_count', 0) or 0
    ssm_inner = meta.get(f'{arch}.ssm.inner_size', 0) or 0
    n_kv_arr = meta.get(f'{arch}.attention.head_count_kv')
    mla_flag = False; mka, mva = 0, 0; display_nkv = 1; display_ekv = 0; kv_fp16_mib = 0.0
    if isinstance(n_kv_arr, list) and len(n_kv_arr) > 1: n_kv_heads_arr = n_kv_arr
    else: n_kv_heads_arr = None
    # Гибриды (Qwen3-Next/DeltaNet): full-attention только в 1 из fa_interval слоёв
    if fa_interval > 0 and main_layers > 0:
        full_layers = (main_layers + fa_interval - 1) // fa_interval
    else:
        full_layers = main_layers if main_layers > 0 else n_layers
    if (key_len > 0) or (n_kv_heads_arr is not None):
        mla_flag = True
        if key_len == 0: key_len = 128
        if val_len == 0: val_len = key_len
        if n_kv_heads_arr: tot = sum(n_kv_heads_arr)
        elif n_kv_heads > 0: tot = n_kv_heads * full_layers if full_layers > 0 else 0
        else: tot = n_head * full_layers if full_layers > 0 else 0
        mka, mva = key_len, val_len; display_ekv = key_len
        if isinstance(n_kv_heads, int) and n_kv_heads > 0: display_nkv = n_kv_heads
        else: display_nkv = tot // full_layers if full_layers > 0 else tot
        kv_fp16 = (key_len + val_len) * tot * 2
    else:
        hd = (n_embd // n_head) if n_head > 0 else 128
        if n_kv_heads == 0 or not isinstance(n_kv_heads, int): n_kv_heads = n_head
        display_ekv = hd; display_nkv = n_kv_heads
        kv_fp16 = 4 * hd * n_kv_heads * full_layers
    kv_fp16_mib = kv_fp16 / (1024**2) if kv_fp16 > 0 else 0
    # Рекуррентное состояние линейных (SSM/DeltaNet) слоёв — фиксированное, не зависит от ctx
    linear_state_mib = 0.0
    if ssm_state > 0 and ssm_groups > 0 and ssm_inner > 0:
        n_linear = max(0, main_layers - full_layers)
        head_v = ssm_inner // ssm_groups if ssm_groups else ssm_inner
        linear_state_mib = n_linear * ssm_groups * ssm_state * head_v * 2 / (1024**2)
    return {"name": name, "size_gb": size_gb, "arch": arch, "n_embd": n_embd, "n_layers": n_layers,
            "n_head": n_head, "n_kv_heads": display_nkv, "n_embd_kv": display_ekv,
            "n_expert": n_expert, "n_expert_used": n_expert_used, "context_length": cl,
            "kv_fp16_mib": round(kv_fp16_mib, 4), "mla": mla_flag, "key_length": mka, "value_length": mva,
            "full_layers": full_layers, "fa_interval": fa_interval, "nextn": nextn,
            "linear_state_mib": round(linear_state_mib, 1)}

def kv_per_token_mib(params, cache_type="fp16"):
    kvb = params["kv_fp16_mib"] * 1024 * 1024
    bpc = {"fp16": 2, "q8_0": 1, "q4_0": 0.5}.get(cache_type, 2)
    kvb = kvb * bpc / 2.0
    return kvb / (1024**2)

def get_kvpt_real(fpath, cache_type="q8_0"):
    """Empirical KV/token from sidecar <model>.gguf.kv.json (measured on real load, not formula)."""
    sidecar = fpath + ".kv.json"
    if os.path.isfile(sidecar):
        try:
            d = json.load(open(sidecar, "r", encoding="utf-8"))
            v = (d.get("kvpt_mib_per_token") or {}).get(cache_type)
            if v:
                return float(v)
        except Exception:
            pass
    return None

_model_cache = {}; _model_cache_lock = threading.Lock()

def get_cached_model_params(name, models_dir=None):
    md = models_dir or _current_models_dir
    fpath = os.path.join(md, name)
    if not os.path.isfile(fpath): return None
    with _model_cache_lock:
        if fpath in _model_cache: return _model_cache[fpath]
    params = get_model_params(fpath)
    with _model_cache_lock: _model_cache[fpath] = params
    return params

def get_models(models_dir=None):
    md = models_dir or _current_models_dir
    ms = []
    if os.path.isdir(md):
        for root, dirs, files in os.walk(md):
            for f in sorted(files):
                fp = os.path.join(root, f)
                if not f.endswith(".gguf") or os.path.islink(fp): continue
                rel = os.path.relpath(fp, md).replace(os.sep, "/")
                s = os.path.getsize(fp)
                ms.append({"name": rel, "size": f"{s/1024**3:.1f} GB", "size_gb": round(s/1024**3, 1)})
    return ms

def get_loras(models_dir=None):
    md = models_dir or _current_models_dir
    d = os.path.join(md, "loras")
    ms = []
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            fp = os.path.join(d, f)
            if not f.endswith(".gguf") or os.path.islink(fp): continue
            s = os.path.getsize(fp)
            ms.append({"name": f, "size": f"{s/1024**2:.0f} MB", "size_mb": round(s/1024**2, 1)})
    return ms

def inst_models_dir(inst):
    d = (inst or {}).get("models_dir", "")
    if d and os.path.isdir(d):
        return d
    return _current_models_dir

def list_ggufs(models_dir):
    out = []
    base = models_dir
    if not os.path.isdir(base):
        return out
    base_low = os.path.basename(base).lower()
    for f in sorted(os.listdir(base)):
        if not f.endswith(".gguf"):
            continue
        fp = os.path.join(base, f)
        if not os.path.isfile(fp) or os.path.islink(fp):
            continue
        low = f.lower()
        if "mmproj" in low:
            kind = "mmproj"
        elif re.search(r"(?:^|[\._\-\s])mtp(?:[\._\-\s]|$)", low):
            kind = "mtp"
        elif "lora" in low or base_low == "loras":
            kind = "lora"
        else:
            kind = "model"
        try:
            s = os.path.getsize(fp)
        except OSError:
            continue
        out.append({"name": f, "size_gb": round(s/1024**3, 1), "size_mb": round(s/1024**2, 1), "kind": kind})
    return out

def _resolve_ref(inst, ref):
    if not ref:
        return None
    md = inst_models_dir(inst)
    p = os.path.join(md, ref)
    if os.path.isfile(p):
        return os.path.relpath(p, md).replace(os.sep, "/")
    if "/" not in ref and os.path.isfile(os.path.join(md, "loras", ref)):
        return "loras/" + ref
    return None

_models_dir_cache = {"dir": "", "models": []}

def refresh_models_cache():
    _models_dir_cache["dir"] = _current_models_dir
    _models_dir_cache["models"] = get_models(_current_models_dir)
    try:
        _models_dir_cache["mtime"] = os.path.getmtime(_current_models_dir)
    except Exception:
        _models_dir_cache["mtime"] = 0
    return _models_dir_cache["models"]

def get_sys_info():
    cpu = "Unknown"; ram_gb = "--"; ram_free = 0; channels = ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"): cpu = line.split(":", 1)[1].strip(); break
    except: pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1]); gb = kb / 1024 / 1024
                    ram_gb = str(int(gb) + (1 if gb > int(gb) else 0)) + " GB"
                if line.startswith("MemAvailable:"): ram_free = int(line.split()[1]) // 1024
    except: pass
    try:
        r = subprocess.run(["dmidecode", "-t", "memory"], capture_output=True, timeout=5, text=True)
        cnt = 0
        for line in r.stdout.split("\n"):
            if "Size: " in line and "No Module" not in line and "GB" in line:
                cnt += 1
        if cnt > 0:
            channels = f" ({cnt} ch)"
    except: pass
    return {"cpu": cpu, "ram": ram_gb + channels, "ram_free": ram_free}

def _mem_available_mib():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except: pass
    return 0

def get_gpu_list():
    gpus = []; metrics = {}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total", "--format=csv,noheader"], capture_output=True, timeout=10, text=True)
        for line in r.stdout.strip().split("\n"):
            if not line.strip(): continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                idx = parts[0]; name = parts[1]; uuid = parts[2]
                mem = parts[3].replace(" MiB", "").strip()
                gpus.append({"idx": f"GPU{idx}", "name": name, "uuid": uuid, "vram_total": int(mem)})
        # Metrics
        r2 = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,temperature.gpu,power.draw", "--format=csv,noheader"], capture_output=True, timeout=10, text=True)
        for line in r2.stdout.strip().split("\n"):
            if not line.strip(): continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                idx = parts[0]; used = parts[1].replace(" MiB", "").strip(); free = parts[2].replace(" MiB", "").strip(); temp = parts[3]
                try: pw = float(parts[4].replace(" W", "").strip())
                except: pw = None
                uuid = gpus[int(idx)]["uuid"] if int(idx) < len(gpus) else f"GPU-{idx}"
                metrics[uuid] = {"vram_used": f"{used} / {gpus[int(idx)]['vram_total']} MiB", "vram_free": int(free), "temp": int(temp), "power": pw}
    except: pass
    return gpus, metrics

def _gpu_uuid_maps():
    """Live nvidia-smi maps: (uuid_to_idx, idx_to_uuid {'GPU0':uuid})."""
    u2i, i2u = {}, {}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True, timeout=10, text=True)
        for line in r.stdout.strip().split("\n"):
            if not line.strip(): continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[1].startswith("GPU-"):
                i2u[f"GPU{parts[0]}"] = parts[1]
                u2i[parts[1]] = f"GPU{parts[0]}"
    except Exception:
        pass
    return u2i, i2u

def _cuda_devices_str(gpu_idx_list):
    """CUDA_VISIBLE_DEVICES value: prefer GPU UUIDs (unambiguous, immune to CUDA
    FASTEST_FIRST reordering / card swaps); fall back to indices if UUID unknown.
    Accepts index strings like ['0','1'] or ['GPU0','GPU1']."""
    if not gpu_idx_list:
        return ""
    _, i2u = _gpu_uuid_maps()
    out = []
    for x in gpu_idx_list:
        s = str(x)
        key = s if s.startswith("GPU") else f"GPU{s}"
        u = i2u.get(key)
        out.append(u if u else s.replace("GPU", ""))
    return ",".join(out)

def _gpu_used_mib(gpu_indices):
    """Total memory.used (MiB) across the given GPU indices (e.g. ['GPU0','GPU1'])."""
    nums = []
    for idx in (gpu_indices or []):
        try: nums.append(int(str(idx).replace("GPU", "")))
        except Exception: pass
    if not nums: return 0
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, timeout=10, text=True)
        total = 0
        for line in r.stdout.strip().split("\n"):
            if not line.strip(): continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    if int(parts[0]) in nums: total += int(parts[1])
                except Exception: pass
        return total
    except Exception:
        return 0

def _wait_gpu_free(gpu_nums, timeout=45, threshold_mib=2048):
    """Wait until the given GPUs' used memory drops below threshold (conflicting instance's VRAM released)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, timeout=10, text=True)
            ok = True
            for line in r.stdout.strip().split("\n"):
                if not line.strip(): continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    try:
                        if int(parts[0]) in gpu_nums and int(parts[1]) > threshold_mib:
                            ok = False
                    except Exception:
                        pass
            if ok:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False

def _stop_rm_container(name):
    """Best-effort docker stop+rm. Never raises. Kill-fallback on stop timeout.
    Returns True if the container was actually stopped (stop exit 0 / kill attempted)."""
    stopped = False
    try:
        r = subprocess.run(["docker", "stop", name], capture_output=True, timeout=20)
        stopped = (r.returncode == 0)
    except subprocess.TimeoutExpired:
        try:
            subprocess.run(["docker", "kill", name], capture_output=True, timeout=10)
            stopped = True
        except Exception:
            pass
    except Exception:
        pass
    try:
        subprocess.run(["docker", "rm", name], capture_output=True, timeout=15)
    except Exception:
        pass
    return stopped

# ======================== API (своё, без :8090) ========================


def _load_jwt_secret():
    s = os.getenv("JWT_SECRET")
    if s:
        return s
    f = Path(__file__).parent / "jwt_secret.key"
    if f.exists():
        try:
            v = f.read_text("utf-8").strip()
            if v:
                return v
        except OSError:
            pass
    secret = secrets.token_urlsafe(48)
    try:
        fd = os.open(f, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        value = f.read_text("utf-8").strip()
        if not value:
            raise RuntimeError("JWT secret file is empty")
        return value
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        out.write(secret)
    return secret

JWT_SECRET = _load_jwt_secret()
JWT_ALGO = "HS256"
JWT_ACCESS_MIN = 1440
JWT_REFRESH_DAYS = 30

def _get_auth():
    c = load_config()
    return c.get("auth_user", "admin"), c.get("auth_pass", "admin")

def _mk_access(u):
    return jwt.encode({"sub": u, "exp": datetime.now(timezone.utc) + timedelta(minutes=JWT_ACCESS_MIN), "t": "a"}, JWT_SECRET, algorithm=JWT_ALGO)

def _mk_refresh(u):
    return jwt.encode({"sub": u, "exp": datetime.now(timezone.utc) + timedelta(days=JWT_REFRESH_DAYS), "t": "r"}, JWT_SECRET, algorithm=JWT_ALGO)

def _verify(token, ttype="a"):
    try:
        p = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        if p.get("t") != ttype:
            return None
        return p.get("sub")
    except:
        return None

def _get_token(r):
    h = r.headers.get("authorization", "")
    if not h.startswith("Bearer "):
        return None
    return h.split(" ", 1)[1]

butler_state = {"mode": "ON"}
_main_loop = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _main_loop
    _main_loop = asyncio.get_running_loop()
    # Background cache updater for slow GPU/Docker ops
    def bg_updater():
        while True:
            try:
                _bg_cache["gpus"], _bg_cache["gpu_metrics"] = get_gpu_list()
                active = get_active_containers()
                _bg_cache["active"] = active
                _bg_cache["slots"] = _poll_slots(active)
                _bg_cache["ready"] = _probe_ready(active)
                _bg_cache["failed"] = get_failed_containers()
                _bg_cache["ts"] = time.time()
            except Exception as e:
                log.error(f"bg_cache error: {e}")
            time.sleep(3)
    t = threading.Thread(target=bg_updater, daemon=True)
    t.start()
    ts_ = threading.Thread(target=_speed_sampler, daemon=True)
    ts_.start()
    ti = threading.Thread(target=_idle_checker, daemon=True)
    ti.start()
    try:
        yield
    finally:
        _main_loop = None

app = FastAPI(title="Butler", version="2.0.0", lifespan=lifespan)

@app.exception_handler(InputError)
async def input_error_response(request: Request, exc: InputError):
    return JSONResponse({"success": False, "message": str(exc)}, status_code=400)


@app.exception_handler(RequestValidationError)
async def request_validation_response(request: Request, exc: RequestValidationError):
    # Do not echo submitted values (especially passwords/API keys) in errors.
    errors = [".".join(str(p) for p in error["loc"]) + ": " + error["msg"] for error in exc.errors()]
    return JSONResponse({"success": False, "message": "; ".join(errors)}, status_code=400)


_bg_cache = {"gpus": [], "gpu_metrics": {}, "active": {}, "ready": {}, "failed": {}, "slots": {}, "ts": 0.0}
_status_cache = {"data": None, "ts": 0.0}

@app.get("/api/status")
async def api_status():
    import time as _t_as
    now = time.time()
    if now - _status_cache["ts"] < 0.5 and _status_cache["data"] is not None:
        return JSONResponse(_status_cache["data"])
    instances = load_instances()
    active = _bg_cache.get("active", {})
    ready = _bg_cache.get("ready", {})
    failed = _bg_cache.get("failed", {})
    dirty = False
    for inst in instances:
        # model_ready = container running AND model fully loaded (health OK) — UI toggle ON only then
        # model_failed = container exists but exited with non-zero code (crash/OOM); clean stop = exit 0 / rm'd
        inst["model_ready"] = bool(ready.get(inst.get("id")))
        inst["model_failed"] = bool(failed.get(inst.get("id")))
        live = 0.0
        if inst.get("id") in active:
            snapshot = _bg_cache.get("slots", {}).get(inst.get("id"))
            sl = dict(snapshot) if snapshot is not None else None
            iid = inst.get("id")
            if sl:
                sl["queued"] = inflight_queued(iid, sl["active"])
                inst["slots"] = sl
                _last_slots[iid] = sl
            else:
                inst["slots"] = _last_slots.get(iid)
            live = speed_for_status(iid)
        else:
            inst.pop("slots", None)
            _last_slots.pop(inst.get("id"), None)
            live = 0.0
        if live > 0:
            inst["speed_tps"] = live
            inst["speed_live"] = True
            if inst.get("last_speed_tps") != live:
                inst["last_speed_tps"] = live
                inst["last_speed_ts"] = time.time()
                dirty = True
        else:
            inst["speed_tps"] = inst.get("last_speed_tps") or 0
            inst["speed_live"] = False
        emn, emx = _speed_extrema.get(inst.get("id"), (0.0, 0.0))
        ess, ecc = _speed_samples.get(inst.get("id"), (0.0, 0))
        pmn, pmx = inst.get("min_speed_tps") or 0, inst.get("max_speed_tps") or 0
        pss, pcc = inst.get("speed_sum") or 0.0, inst.get("speed_count") or 0
        if ecc > 0 and (emn != pmn or emx != pmx or ess != pss or ecc != pcc):
            inst["min_speed_tps"], inst["max_speed_tps"] = emn, emx
            inst["speed_sum"], inst["speed_count"] = ess, ecc
            dirty = True
        inst["min_speed_tps"] = inst.get("min_speed_tps") or 0
        inst["max_speed_tps"] = inst.get("max_speed_tps") or 0
        inst["speed_count"] = inst.get("speed_count") or 0
        inst["avg_speed_tps"] = round((inst.get("speed_sum") or 0.0) / inst["speed_count"], 1) if inst.get("speed_count") else 0
    if dirty:
        save_instances([dict(inst) for inst in instances])
    gpus, gpu_metrics = _bg_cache.get("gpus", []), _bg_cache.get("gpu_metrics", {})
    data = {
        "gpus": gpus, "gpu_metrics": gpu_metrics,
        "models": get_models(_current_models_dir),
        "models_with_status": get_models(_current_models_dir),
        "loras": get_loras(_current_models_dir),
        "instances": instances, "active": active,
        "butler": {"mode": butler_state["mode"]},
        "sys_info": get_sys_info()
    }
    _status_cache["data"] = data
    _status_cache["ts"] = now
    return JSONResponse(data)

@app.get("/api/gpus")
async def api_gpus():
    return await _run_blocking(_api_gpus_sync)


def _api_gpus_sync():
    gpus, metrics = get_gpu_list()
    return JSONResponse({"gpus": gpus, "gpu_metrics": metrics})

@app.get("/api/instances")
async def api_get_instances():
    return JSONResponse(load_instances())

@app.post("/api/instances")
async def api_create(request: Request):
    body = await _read_json_object(request)
    name = body.get("name", "").strip()
    if not name:
        raise InputError("name: must not be empty")
    if "id" in body:
        iid = _instance_id(body["id"])
    elif re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name):
        iid = name  # preserve the legacy ID for existing API clients
    else:
        iid = "inst_" + secrets.token_hex(12)
    port = _port_value(body.get("port", "0"))
    gpu_idx = body.get("gpu_indices", [])
    mode = body.get("mode", "chat")
    if mode not in ("chat", "embed", "rerank"):
        raise InputError("mode: expected chat, embed or rerank")
    with _instances_lock:
        insts = load_instances()
        if any(x["id"] == iid for x in insts):
            raise InputError("id: already exists")
        port = _unique_port(port, insts)
        new_inst = {"id": iid, "name": name, "port": port, "model": "", "mode": mode,
                    "gpu_indices": gpu_idx, "ctx": 16384, "ngl": 99, "flash": True, "ram": False, "kvq": "fp16", "parallel": 1,
                    "allowed": True, "idle_exempt": False}
        insts.append(new_inst)
        save_instances(insts)
    return JSONResponse({"success": True, "message": f"Created {name} on port {port}"})


@app.delete("/api/instances/{inst_id}")
async def api_delete(inst_id: str):
    # stop+rm the container BEFORE deleting: deletion used to leave a running orphan
    # holding VRAM (-> _wait_gpu_free 45s penalty on every later load of that GPU)
    await asyncio.to_thread(_stop_rm_container, f"llama-inst-{inst_id}")
    with _instances_lock:
        insts = load_instances()
        insts = [x for x in insts if x["id"] != inst_id]
        save_instances(insts)
    return JSONResponse({"success": True, "message": "Deleted"})

@app.post("/api/instances/{inst_id}/move")
async def api_move_inst(inst_id: str, request: Request):
    body = await _read_json_object(request)
    direction = (body.get("direction") or "").strip().lower()  # "up" | "down"
    with _instances_lock:
        insts = load_instances()
        idx = next((i for i, x in enumerate(insts) if x["id"] == inst_id), None)
        if idx is None:
            return JSONResponse({"success": False, "message": "Not found"})
        if direction == "up":
            if idx == 0: return JSONResponse({"success": False, "message": "Already first"})
            insts[idx-1], insts[idx] = insts[idx], insts[idx-1]
        elif direction == "down":
            if idx == len(insts)-1: return JSONResponse({"success": False, "message": "Already last"})
            insts[idx+1], insts[idx] = insts[idx], insts[idx+1]
        else:
            return JSONResponse({"success": False, "message": "Bad direction (up|down)"})
        save_instances(insts)
        return JSONResponse({"success": True, "message": "Moved"})
    return JSONResponse({"success": False, "message": "Not found"})

def _checked_instance_port(inst, value, instances):
    port = _unique_port(value, instances, inst["id"])
    if port != str(inst.get("port") or "0"):
        try:
            active = get_active_containers(strict=True)
        except (OSError, subprocess.SubprocessError):
            raise InputError("port: cannot verify container status") from None
        if inst["id"] in active:
            raise InputError("port: stop the instance before changing its port")
    return port


@app.patch("/api/instances/{inst_id}")
async def api_rename(inst_id: str, request: Request):
    body = await _read_json_object(request)
    if "name" in body and not body["name"].strip():
        raise InputError("name: must not be empty")
    async with _model_start_lock if "port" in body else nullcontext():
        with _instances_lock:
            insts = load_instances()
            for x in insts:
                if x["id"] == inst_id:
                    if "port" in body:
                        x["port"] = _checked_instance_port(x, body["port"], insts)
                    if "name" in body: x["name"] = body["name"]
                    if "ngl" in body: x["ngl"] = int(body["ngl"])
                    if "ctx" in body: x["ctx"] = int(body["ctx"])
                    if "flash" in body: x["flash"] = body["flash"]
                    if "ram" in body: x["ram"] = body["ram"]
                    if "kvq" in body: x["kvq"] = body["kvq"]
                    if "parallel" in body: x["parallel"] = int(body["parallel"])
                    if "model" in body: x["model"] = body["model"]
                    if "mode" in body and body["mode"] in ("chat", "embed", "rerank"): x["mode"] = body["mode"]
                    if "allowed" in body: x["allowed"] = bool(body["allowed"])
                    if "idle_exempt" in body: x["idle_exempt"] = bool(body["idle_exempt"])
                    save_instances(insts)
                    return JSONResponse({"success": True, "message": "Updated"})
        return JSONResponse({"success": False, "message": "Not found"})

@app.patch("/api/instances/{inst_id}/port")
async def api_reporte(inst_id: str, request: Request):
    body = await _read_json_object(request)
    async with _model_start_lock:
        with _instances_lock:
            insts = load_instances()
            for x in insts:
                if x["id"] == inst_id:
                    x["port"] = _checked_instance_port(x, body.get("port", x["port"]), insts)
                    save_instances(insts)
                    return JSONResponse({"success": True, "message": "Port updated"})
        return JSONResponse({"success": False, "message": "Not found"})

@app.post("/api/instances/{inst_id}/gpu")
async def api_add_gpu(inst_id: str, request: Request):
    body = await _read_json_object(request)
    gpu_idx = _gpu_value(body.get("gpu_idx", ""))
    with _instances_lock:
        insts = load_instances()
        inst = next((x for x in insts if x["id"] == inst_id), None)
        if not inst: return JSONResponse({"success": False, "message": "Not found"})
        if "gpu_indices" not in inst: inst["gpu_indices"] = []
        if any(_gpu_value(value) == gpu_idx for value in inst["gpu_indices"]):
            raise InputError("gpu_idx: already assigned")
        if gpu_idx == "__select__":
            inst["gpu_indices"].append(gpu_idx)
            save_instances(insts)
            return JSONResponse({"success": True, "message": "Placeholder added"})
        if gpu_idx in inst["gpu_indices"]:
            return JSONResponse({"success": False, "message": f"GPU {gpu_idx} already assigned"})
        if "__select__" in inst["gpu_indices"]:
            inst["gpu_indices"].remove("__select__")
        inst["gpu_indices"].append(gpu_idx)
        save_instances(insts)
    return JSONResponse({"success": True, "message": f"GPU {gpu_idx} added"})

@app.patch("/api/instances/{inst_id}/gpu/{gpu_idx}")
async def api_replace_gpu(inst_id: str, gpu_idx: str, request: Request):
    body = await _read_json_object(request)
    old_gpu = _gpu_value(gpu_idx)
    new_gpu = _gpu_value(body.get("gpu_idx", ""))
    if new_gpu == "__select__":
        raise InputError("gpu_idx: select a GPU")
    with _instances_lock:
        instances = load_instances()
        instance = next((item for item in instances if item["id"] == inst_id), None)
        if instance is None:
            return JSONResponse({"success": False, "message": "Instance not found"}, status_code=404)
        gpus = instance.get("gpu_indices", [])
        normalized = [_gpu_value(value) for value in gpus]
        if old_gpu not in normalized:
            return JSONResponse({"success": False, "message": "GPU slot changed; refresh and try again"}, status_code=409)
        if old_gpu != new_gpu:
            if new_gpu in normalized:
                return JSONResponse({"success": False, "message": "GPU already assigned"}, status_code=409)
            gpus[normalized.index(old_gpu)] = new_gpu
            save_instances(instances)
    return JSONResponse({"success": True, "message": "GPU replaced"})


@app.delete("/api/instances/{inst_id}/gpu/{gpu_idx}")
async def api_del_gpu(inst_id: str, gpu_idx: str):
    with _instances_lock:
        insts = load_instances()
        for inst in insts:
            if inst["id"] == inst_id:
                if gpu_idx in inst.get("gpu_indices", []):
                    inst["gpu_indices"].remove(gpu_idx)
                    save_instances(insts)
                    return JSONResponse({"success": True, "message": f"GPU {gpu_idx} removed"})
    return JSONResponse({"success": False, "message": f"GPU {gpu_idx} not found"})

@app.get("/api/loras")
async def api_get_loras():
    return JSONResponse(get_loras(_current_models_dir))

@app.post("/api/instances/{inst_id}/lora")
async def api_add_lora(inst_id: str, request: Request):
    body = await _read_json_object(request)
    ln = body.get("lora", "")
    if not ln: return JSONResponse({"success": False, "message": "Lora name required"})
    with _instances_lock:
        insts = load_instances()
        inst = next((x for x in insts if x["id"] == inst_id), None)
        if not inst: return JSONResponse({"success": False, "message": "Not found"})
        if not _resolve_ref(inst, ln):
            return JSONResponse({"success": False, "message": f"Lora not found: {ln}"})
        if "loras" not in inst: inst["loras"] = []
        if any(ln == existing or _resolve_ref(inst, ln) == _resolve_ref(inst, existing)
               for existing in inst["loras"]):
            raise InputError("lora: already assigned")
        inst["loras"].append(ln)
        save_instances(insts)
    return JSONResponse({"success": True, "message": f"Lora {ln} assigned"})

@app.delete("/api/instances/{inst_id}/lora")
async def api_del_lora(inst_id: str, lora: str):
    with _instances_lock:
        insts = load_instances()
        for inst in insts:
            if inst["id"] == inst_id:
                if lora in inst.get("loras", []):
                    inst["loras"].remove(lora)
                    if inst.get("active_lora") == lora: inst["active_lora"] = ""
                    save_instances(insts)
                    return JSONResponse({"success": True, "message": f"Lora {lora} removed"})
    return JSONResponse({"success": False, "message": f"Lora {lora} not found"})

@app.post("/api/instances/{inst_id}/lora/apply")
async def api_apply_lora(inst_id: str, request: Request):
    body = await _read_json_object(request)
    return await _run_blocking(_api_apply_lora_sync, inst_id, body)


def _api_apply_lora_sync(inst_id, body):
    ln = body.get("lora", "")
    insts = load_instances()
    inst = next((x for x in insts if x["id"] == inst_id), None)
    if not inst: return JSONResponse({"success": False, "message": "Not found"}, status_code=404)
    active = get_active_containers()
    if inst_id not in active:
        return JSONResponse({"success": False, "message": "Instance not running"})
    if ln:
        if ln not in inst.get("loras", []):
            return JSONResponse({"success": False, "message": f"Lora not assigned: {ln}"})
        rl = _resolve_ref(inst, ln)
        if not rl:
            return JSONResponse({"success": False, "message": f"Lora file not found: {ln}"})
        payload = [f"/models/{rl}"]
    else:
        payload = []
    try:
        hr = httpx.post(f"http://127.0.0.1:{inst['port']}/lora-adapters", json=payload,
                        headers={"Authorization": f"Bearer {API_KEY}"}, timeout=60)
        if hr.status_code < 300:
            with _instances_lock:
                insts = load_instances()
                inst2 = next((x for x in insts if x["id"] == inst_id), None)
                if inst2:
                    inst2["active_lora"] = ln
                    save_instances(insts)
            return JSONResponse({"success": True, "message": ("Applied: " + ln) if ln else "Base model active"})
        return JSONResponse({"success": False, "message": hr.text[:200]})
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)[:200]})

# ======================== File manager ========================

async def _run_blocking(fn, *args, **kwargs):
    """Keep the caller's lifecycle lock until a started worker actually finishes."""
    task = asyncio.create_task(asyncio.to_thread(fn, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled():
            task.exception()  # retrieve worker errors while preserving cancellation
        raise


def _fs_path(p):
    if not p:
        return ""
    p = os.path.normpath(p)
    if not os.path.isabs(p):
        p = os.path.abspath(p)
    return p

@app.get("/api/fs")
async def api_fs_list(path: str = ""):
    return await _run_blocking(_api_fs_list_sync, path)


def _api_fs_list_sync(path):
    p = _fs_path(path)
    if not p or not os.path.isdir(p):
        return JSONResponse({"success": False, "message": "Directory not found"}, status_code=404)
    try:
        names = sorted(os.listdir(p), key=lambda x: (os.path.isfile(os.path.join(p, x)), x.lower()))
    except OSError:
        return JSONResponse({"success": False, "message": "Permission denied"}, status_code=403)
    entries = []
    for n in names:
        fp = os.path.join(p, n)
        if os.path.islink(fp):
            continue
        if os.path.isdir(fp):
            entries.append({"name": n, "type": "dir"})
        else:
            try:
                sz = os.path.getsize(fp)
            except OSError:
                sz = 0
            e = {"name": n, "type": "file", "size": sz}
            if sz >= 1024**3:
                e["size_gb"] = round(sz/1024**3, 1)
            elif sz >= 1024**2:
                e["size_mb"] = round(sz/1024**2, 1)
            entries.append(e)
    try:
        free = shutil.disk_usage(p).free
    except OSError:
        free = 0
    return JSONResponse({"path": p, "parent": os.path.dirname(p) or "/", "entries": entries, "free_gb": round(free/1024**3, 1)})

@app.post("/api/fs/mkdir")
async def api_fs_mkdir(request: Request):
    body = await _read_json_object(request)
    return await _run_blocking(_api_fs_mkdir_sync, body)


def _api_fs_mkdir_sync(body):
    p = _fs_path(body.get("path", ""))
    if not p or not os.path.isdir(os.path.dirname(p)):
        return JSONResponse({"success": False, "message": "Parent directory not found"})
    if os.path.exists(p):
        return JSONResponse({"success": False, "message": "Already exists"})
    try:
        os.makedirs(p)
    except OSError as e:
        return JSONResponse({"success": False, "message": str(e)[:150]})
    return JSONResponse({"success": True, "path": p})

@app.post("/api/fs/rename")
async def api_fs_rename(request: Request):
    body = await _read_json_object(request)
    return await _run_blocking(_api_fs_rename_sync, body)


def _api_fs_rename_sync(body):
    p = _fs_path(body.get("path", ""))
    nn = str(body.get("new_name", "")).strip()
    if not p or not nn or "/" in nn or nn in (".", ".."):
        return JSONResponse({"success": False, "message": "Bad name"})
    if not os.path.exists(p):
        return JSONResponse({"success": False, "message": "Not found"})
    dest = os.path.join(os.path.dirname(p), nn)
    if os.path.exists(dest):
        return JSONResponse({"success": False, "message": "Name taken"})
    try:
        os.rename(p, dest)
    except OSError as e:
        return JSONResponse({"success": False, "message": str(e)[:150]})
    return JSONResponse({"success": True, "path": dest})

@app.post("/api/fs/move")
async def api_fs_move(request: Request):
    body = await _read_json_object(request)
    return await _run_blocking(_api_fs_move_sync, body)


def _api_fs_move_sync(body):
    p = _fs_path(body.get("path", ""))
    d = _fs_path(body.get("dest", ""))
    if not p or not os.path.exists(p):
        return JSONResponse({"success": False, "message": "Source not found"})
    if not d or not os.path.isdir(d):
        return JSONResponse({"success": False, "message": "Destination not found"})
    dest = os.path.join(d, os.path.basename(p))
    if os.path.exists(dest):
        return JSONResponse({"success": False, "message": "Cannot move: '" + os.path.basename(p) + "' already exists in destination"})
    try:
        shutil.move(p, d)
    except OSError as e:
        return JSONResponse({"success": False, "message": str(e)[:150]})
    return JSONResponse({"success": True, "path": dest})

@app.post("/api/fs/copy")
async def api_fs_copy(request: Request):
    body = await _read_json_object(request)
    return await _run_blocking(_api_fs_copy_sync, body)


def _api_fs_copy_sync(body):
    p = _fs_path(body.get("path", ""))
    d = _fs_path(body.get("dest", ""))
    if not p or not os.path.exists(p):
        return JSONResponse({"success": False, "message": "Not found"})
    if not d or not os.path.isdir(d):
        return JSONResponse({"success": False, "message": "Destination not found"})
    dest = os.path.join(d, os.path.basename(p))
    if os.path.exists(dest):
        return JSONResponse({"success": False, "message": "Cannot copy: '" + os.path.basename(p) + "' already exists in destination"})
    if os.path.isdir(p):
        sz = 0
        for root, dirs, files in os.walk(p):
            for f in files:
                try: sz += os.path.getsize(os.path.join(root, f))
                except OSError: pass
    else:
        sz = os.path.getsize(p)
    try:
        free = shutil.disk_usage(d).free
    except OSError:
        free = 0
    if sz > free:
        return JSONResponse({"success": False, "message": f"Not enough space: need {sz/1024**3:.1f} GB, free {free/1024**3:.1f} GB"})
    try:
        if os.path.isdir(p):
            shutil.copytree(p, dest)
        else:
            shutil.copy2(p, dest)
    except OSError as e:
        return JSONResponse({"success": False, "message": str(e)[:150]})
    return JSONResponse({"success": True, "path": dest})

@app.post("/api/fs/delete")
async def api_fs_delete(request: Request):
    body = await _read_json_object(request)
    return await _run_blocking(_api_fs_delete_sync, body)


def _api_fs_delete_sync(body):
    p = _fs_path(body.get("path", ""))
    if not p or p == "/":
        return JSONResponse({"success": False, "message": "Bad path"}, status_code=400)
    if not os.path.exists(p):
        return JSONResponse({"success": False, "message": "Not found"})
    try:
        if os.path.isdir(p):
            shutil.rmtree(p)
        else:
            os.remove(p)
    except OSError as e:
        return JSONResponse({"success": False, "message": str(e)[:150]})
    return JSONResponse({"success": True, "message": "Deleted: " + os.path.basename(p)})

@app.post("/api/instances/{inst_id}/params")
async def api_save_params(inst_id: str, request: Request):
    body = await _read_json_object(request)
    with _instances_lock:
        insts = load_instances()
        for x in insts:
            if x["id"] == inst_id:
                for k in ("model", "ngl", "ctx", "flash", "ram", "kvq", "parallel", "mmproj", "mtp", "mtp_nmax", "models_dir", "docker_image"):
                    if k in body: x[k] = body[k]
                if "ngl" in body: x["ngl"] = body["ngl"]
                if "ctx" in body: x["ctx"] = body["ctx"]
                if "parallel" in body: x["parallel"] = body["parallel"]
                save_instances(insts)
                return JSONResponse({"success": True, "message": "Params saved"})
    return JSONResponse({"success": False, "message": "Not found"})

def mtp_flags(mtp, nmax=0):
    """Speculative-decoding (MTP) flags.
    mtp: '' = off; 'builtin' = MTP module inside main GGUF (no draft file); '<file>' = separate head file.
    nmax: --spec-draft-n-max (max draft tokens per step)."""
    if not mtp:
        return []
    f = []
    if mtp != "builtin":
        f += ["--spec-draft-model", f"/models/{mtp}"]
    f += ["--spec-type", "draft-mtp"]
    try:
        n = int(nmax)
        if n >= 1:
            f += ["--spec-draft-n-max", str(n)]
    except Exception:
        pass
    return f

def _llama_try_load(iid, port, command, wait_s=120):
    """Return the trial outcome and keep the container only if ready."""
    name = f"llama-inst-{iid}"
    ready = False
    try:
        _stop_rm_container(name)
        reset_speed_extrema(iid)
        started = subprocess.run(command, capture_output=True, timeout=30)
        if started.returncode != 0:
            return "docker_error"

        deadline = time.monotonic() + wait_s
        while True:
            state = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Status}} {{.State.OOMKilled}}", name],
                capture_output=True, timeout=10, text=True,
            )
            if state.returncode != 0:
                return "docker_error"
            status, _, oom_killed = state.stdout.strip().partition(" ")
            if oom_killed == "true":
                return "oom"
            if status in ("exited", "dead", "restarting"):
                logs = subprocess.run(
                    ["docker", "logs", "--tail", "100", name],
                    capture_output=True, timeout=10, text=True, errors="replace",
                )
                if logs.returncode == 0 and re.search(
                    r"\bout of memory\b|\bcudaErrorMemoryAllocation\b|\bhipErrorOutOfMemory\b",
                    logs.stdout + logs.stderr, re.IGNORECASE,
                ):
                    return "oom"
                return "failed"

            try:
                ready = httpx.get(f"http://127.0.0.1:{port}/health", timeout=3).status_code == 200
            except httpx.RequestError:
                pass
            if ready:
                return "ready"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "timeout"
            time.sleep(min(2, remaining))
    except subprocess.TimeoutExpired:
        return "timeout"
    except OSError:
        return "docker_error"
    finally:
        if not ready:
            try:
                removed = subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
                if removed.returncode and b"No such container" not in removed.stderr:
                    log.warning("autodetect %s: could not remove trial container", iid)
            except (OSError, subprocess.TimeoutExpired):
                log.warning("autodetect %s: could not remove trial container", iid)


# Serialize manual and automatic starts, including GPU conflict resolution.
# One process / one uvicorn worker owns Docker lifecycle operations.
_model_start_lock = asyncio.Lock()

def _serialized_model_start(fn):
    @wraps(fn)
    async def wrapped(*args, **kwargs):
        async with _model_start_lock:
            return await fn(*args, **kwargs)
    return wrapped

@app.post("/api/instances/{iid}/autodetect")
@_serialized_model_start
async def api_autodetect(iid: str, request: Request):
    body = await _read_json_object(request)
    inflight_bump(iid, 1)
    try:
        return await _run_blocking(_api_autodetect_sync, iid, body)
    finally:
        inflight_bump(iid, -1)


def _api_autodetect_sync(iid, body):
    insts = load_instances()
    inst = next((x for x in insts if x["id"] == iid), None)
    if not inst:
        return JSONResponse({"success": False, "message": "Instance not found"})
    mn = inst.get("model", "")
    if not mn:
        return JSONResponse({"success": False, "message": "No model set"})
    md = inst_models_dir(inst)
    mp = os.path.join(md, mn)
    if not os.path.isfile(mp):
        return JSONResponse({"success": False, "message": "Model file not found"})
    port = inst.get("port", "")
    if not port or str(port) == "0":
        return JSONResponse({"success": False, "message": "Port not set"})
    user_max = _integer(body.get("max_ctx", 0), "max_ctx")
    parallel = _integer(body.get("parallel", inst.get("parallel", 1)), "parallel", minimum=1)
    try:
        params = get_cached_model_params(mn, md) or {}
    except (OSError, ValueError, struct.error):
        return JSONResponse({"success": False, "message": "Could not read model metadata"})
    model_max = int(params.get("context_length", 0) or 0) or 131072
    hi = min(model_max, user_max) if user_max else model_max
    lo = min(8192, hi)
    ngl = int(inst.get("ngl", 99))
    flash = True if inst.get("mode", "chat") != "chat" else bool(inst.get("flash", True))
    ram = bool(inst.get("ram", False))
    kvq = inst.get("kvq", "q8_0")
    gpu_nums = [str(x).replace("GPU", "") for x in inst.get("gpu_indices", []) if x != "__select__"]
    attempts = []
    cand = hi
    while True:
        command = _build_instance_command(
            inst, model=mn, models_dir=md, ctx=cand, ngl=ngl, flash=flash,
            ram=ram, kvq=kvq, parallel=parallel, gpu_nums=gpu_nums,
        )
        outcome = _llama_try_load(iid, port, command)
        ok = outcome == "ready"
        attempts.append({"ctx": cand, "ok": ok, "reason": outcome})
        log.info(f"autodetect {iid}: ctx={cand} parallel={parallel} -> {outcome}")
        if ok:
            best = cand
            _apply_active_lora(iid, port)
            mark_active(iid)
            break
        if outcome != "oom" or cand == lo:
            message = {
                "oom": f"Out of memory at {cand} total tokens",
                "timeout": f"Startup timed out at {cand} total tokens",
                "docker_error": "Docker could not start or inspect the trial container",
                "failed": "llama-server exited before it was ready. Check the model and launch settings.",
            }[outcome]
            return JSONResponse({"success": False, "message": message, "attempts": attempts})
        cand = max(cand // 2, lo)
    with _instances_lock:
        insts = load_instances()
        for x in insts:
            if x["id"] == iid:
                x["ctx"] = best
                x["parallel"] = parallel
                save_instances(insts)
    return JSONResponse({"success": True, "ctx": best, "parallel": parallel,
                         "per_slot": best // parallel if parallel else best,
                         "attempts": attempts,
                         "message": f"Auto: {best} total / {best // parallel if parallel else best} per slot"})


def lora_docker_args(inst):
    loras = [r for r in (_resolve_ref(inst, l) for l in (inst or {}).get("loras", [])) if r]
    if not loras: return []
    return ["--lora", ",".join(f"/models/{l}" for l in loras), "--lora-init-without-apply"]

def _apply_active_lora(iid, port):
    try:
        insts = load_instances()
        inst = next((x for x in insts if x["id"] == iid), None)
        if not inst: return
        al = inst.get("active_lora", "")
        if not al or al not in inst.get("loras", []): return
        rl = _resolve_ref(inst, al)
        if not rl: return
        r = httpx.post(f"http://127.0.0.1:{port}/lora-adapters", json=[f"/models/{rl}"],
                       headers={"Authorization": f"Bearer {API_KEY}"}, timeout=60)
        if r.status_code >= 300:
            log.warning(f"active_lora apply failed for {iid}: {r.text[:200]}")
    except Exception as e:
        log.warning(f"active_lora apply failed for {iid}: {e}")

def _build_instance_command(inst, *, model, models_dir, ctx, ngl, flash, ram, kvq, parallel, gpu_nums):
    """Build identical Docker flags for manual and automatic model starts."""
    n = inst
    iid = inst["id"]
    port = inst["port"]
    mn = model
    md = models_dir
    img = inst.get("docker_image") or DOCKER_IMAGE
    mode = inst.get("mode", "chat")
    mn_proj = inst.get("mmproj", "") if mode == "chat" else ""
    mn_mtp = inst.get("mtp", "") if mode == "chat" else ""
    if mode != "chat":
        flash = True
    bs = str(ctx) if mode in ("embed", "rerank") else "512"
    if not gpu_nums:
        cmd = ["docker", "run", "-d", "--name", f"llama-inst-{iid}", "--restart", "no",
                "-p", f"{port}:8080",
                "-v", f"{_docker_host_path(md)}:/models:ro", "--entrypoint", LLAMA_SERVER, img,
               "-m", f"/models/{mn}", "--host", "0.0.0.0", "--port", "8080",
               "--n-gpu-layers", "0", "--ctx-size", str(ctx),
               "--flash-attn", "1" if flash else "0", "--api-key", API_KEY,
                "-b", bs, "--parallel", str(parallel), "--cont-batching", "--metrics"]
    else:
        ds = _cuda_devices_str(gpu_nums)
        cmd = ["docker", "run", "-d", "--name", f"llama-inst-{iid}", "--restart", "no",
               "--gpus", "all", "-e", f"CUDA_VISIBLE_DEVICES={ds}", "-p", f"{port}:8080",
               "-v", f"{_docker_host_path(md)}:/models:ro", "--entrypoint", LLAMA_SERVER, img,
               "-m", f"/models/{mn}", "--host", "0.0.0.0", "--port", "8080",
               "--n-gpu-layers", str(ngl), "--ctx-size", str(ctx),
             "--flash-attn", "1" if flash else "0", "--api-key", API_KEY,
             "-b", bs, "--parallel", str(parallel), "--cont-batching", "--metrics"]
    if ram: cmd.append("--no-kv-offload")
    kvq_flag = {"fp16": "f16", "q8_0": "q8_0", "q4_0": "q4_0"}.get(kvq, "f16")
    cmd.extend(["--cache-type-k", kvq_flag, "--cache-type-v", kvq_flag])
    if n.get("mode", "chat") == "embed":
        # --ubatch-size: llama.cpp (11913) с --embeddings кэпит n_batch до n_ubatch (def 512)
        # — без этого input > 512 токенов = 500 "input too large to process"
        cmd.extend(["--no-warmup", "--embeddings", "--ubatch-size", bs])
    elif n.get("mode", "chat") == "rerank":
        cmd.extend(["--no-warmup", "--reranking", "--pooling", "rank", "--ubatch-size", bs])
    if mn_proj: cmd.extend(["--mmproj", f"/models/{mn_proj}"])
    if mn_mtp: cmd.extend(mtp_flags(mn_mtp, n.get("mtp_nmax", 0)))
    largs = lora_docker_args(n)
    if largs: cmd.extend(largs)
    return cmd


@app.post("/api/load")
@_serialized_model_start
async def api_load(request: Request):
    body = await _read_json_object(request)
    iid = body.get("instance", ""); mn = body.get("model", "")
    ctx = int(body.get("ctx_size", 16384)); ngl = int(body.get("ngl", 99))
    flash = body.get("flash_attn", True); ram = body.get("ram_kv", False)
    kvq = body.get("kvq", "fp16"); parallel = int(body.get("parallel", 1))
    if not iid or not mn: return JSONResponse({"success": False, "message": "Fields missing"})
    insts = load_instances(); n = next((x for x in insts if x["id"] == iid), None)
    if not n: return JSONResponse({"success": False, "message": "Instance not found"})
    md = inst_models_dir(n)
    mp = os.path.join(md, mn)
    if not os.path.isfile(mp): return JSONResponse({"success": False, "message": f"File not found: {mn}"})
    mn_proj = n.get("mmproj", "")
    if mn_proj and not os.path.isfile(os.path.join(md, mn_proj)):
        return JSONResponse({"success": False, "message": f"mmproj file not found: {mn_proj}"})
    mn_mtp = n.get("mtp", "")
    if mn_mtp and mn_mtp != "builtin" and not os.path.isfile(os.path.join(md, mn_mtp)):
        return JSONResponse({"success": False, "message": f"MTP file not found: {mn_mtp}"})
    if n.get("mode", "chat") != "chat":
        mn_proj = ""; mn_mtp = ""
        flash = True
    bs = str(ctx) if n.get("mode", "chat") in ("embed", "rerank") else "512"
    port = n["port"]
    if not port or port == "0": return JSONResponse({"success": False, "message": "Port not set"})
    # GPU indices from instance
    gpu_indices = n.get("gpu_indices", [])
    gpu_nums = []
    for idx in gpu_indices:
        try: gpu_nums.append(str(int(str(idx).replace("GPU", ""))))
        except: pass
    # protect from idle-unload: UI START path had NO mark_active -> stale _last_active made
    # the checker unload the model 30-90s after load. Mark active BEFORE starting (like v1_chat).
    mark_active(iid)
    stopped_any = await asyncio.to_thread(_stop_rm_container, f"llama-inst-{iid}")
    await asyncio.sleep(1)
    # Stop conflicting active instances (exclusive GPU) — only between CHAT instances;
    # embed/rerank are small and share the GPU with an LLM
    if gpu_nums and n.get("mode", "chat") == "chat":
        active = get_active_containers()
        for other in insts:
            if other["id"] == iid or other["id"] not in active:
                continue
            if other.get("mode", "chat") != "chat":
                continue
            other_gpus = [str(x).replace("GPU", "") for x in other.get("gpu_indices", []) if x != "__select__"]
            if any(g in other_gpus for g in gpu_nums):
                if await asyncio.to_thread(_stop_rm_container, f"llama-inst-{other['id']}"):
                    stopped_any = True
        # wait for VRAM release ONLY if we actually stopped something (orphan/held VRAM
        # must not force a 45s penalty on every load)
        if stopped_any:
            if not await asyncio.to_thread(_wait_gpu_free, gpu_nums, timeout=45):
                log.warning(f"api_load {iid}: GPUs {gpu_nums} still busy after 45s — orphan container? check `docker ps`")
    img = n.get("docker_image") or DOCKER_IMAGE
    if img != DOCKER_IMAGE and not _docker_image_exists(img):
        return JSONResponse({"success": False, "message": f"Runtime image not found locally: {img}"}, status_code=400)
    cmd = _build_instance_command(n, model=mn, models_dir=md, ctx=ctx, ngl=ngl,
                                  flash=flash, ram=ram, kvq=kvq, parallel=parallel, gpu_nums=gpu_nums)
    reset_speed_extrema(iid)
    try:
        result = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            return JSONResponse({"success": False, "message": f"Docker failed: {result.stderr.decode()[:200]}"})
        return JSONResponse({"success": True, "message": f"Model loaded on port {port}"})
    except Exception as e: return JSONResponse({"success": False, "message": str(e)[:200]})

@app.post("/api/unload")
async def api_unload(request: Request):
    body = await _read_json_object(request)
    iid = body.get("instance", "")
    if not iid: return JSONResponse({"success": False, "message": "No instance"})
    # async (sync docker stop blocked the event loop up to 15s -> UI freeze + 500 on timeout)
    await asyncio.to_thread(_stop_rm_container, f"llama-inst-{iid}")
    return JSONResponse({"success": True, "message": "Unloaded"})

@app.get("/api/check/{iid}")
async def api_check(iid: str):
    return await _run_blocking(_api_check_sync, iid)


def _api_check_sync(iid):
    container = f"llama-inst-{iid}"
    try:
        r = subprocess.run(["docker", "inspect", "--format", "{{.State.Status}}", container],
                          capture_output=True, timeout=10, text=True)
        status = r.stdout.strip()
        if status != "running":
            return JSONResponse({"ready": False, "status": status or "not found"})
        # Try HTTP health check on the instance port
        insts = load_instances()
        inst = next((x for x in insts if x["id"] == iid), None)
        if inst:
            try:
                hr = httpx.get(f"http://localhost:{inst['port']}/health", timeout=3)
                if hr.status_code == 200:
                    _apply_active_lora(iid, inst["port"])
                    return JSONResponse({"ready": True, "status": "ready"})
            except:
                pass
        return JSONResponse({"ready": False, "status": "starting"})
    except Exception as e:
        return JSONResponse({"ready": False, "status": str(e)})

# ======================== KV anchor calibration (on-demand "Промерить") ========================
calib_jobs = {}   # iid -> {status, step, result, error}

def _stop_other_llama(except_iid=None):
    """Stop all active llama containers except the one we are about (re)starting, for exclusive GPU access."""
    try:
        r = subprocess.run(["docker", "ps", "--filter", "name=llama-inst-", "--format", "{{.Names}}"],
                           capture_output=True, text=True, timeout=15)
        for name in r.stdout.strip().split():
            if except_iid and name == f"llama-inst-{except_iid}":
                continue
            subprocess.run(["docker", "stop", name], capture_output=True, timeout=40)
    except Exception:
        pass

@_serialized_model_start
async def _restart_for_calibration(inst, stop_conflicts=True):
    # Stop/remove belongs inside the same lock as start, not in the worker thread.
    name = f"llama-inst-{inst['id']}"
    await asyncio.to_thread(subprocess.run, ["docker", "stop", name], capture_output=True, timeout=30)
    await asyncio.to_thread(subprocess.run, ["docker", "rm", name], capture_output=True, timeout=20)
    # We already hold the shared lock; call its undecorated implementation.
    return await butler_start_instance.__wrapped__(inst, stop_conflicts=stop_conflicts)


def _start_from_calibration(inst, stop_conflicts=True):
    """Submit from the KV worker to the server loop, preserving its start lock."""
    loop = _main_loop
    if loop is None or loop.is_closed() or not loop.is_running():
        raise RuntimeError("Butler event loop is not running")
    try:
        current = asyncio.get_running_loop()
    except RuntimeError:
        current = None
    if current is loop:
        raise RuntimeError("Calibration bridge must run in a background thread")
    return asyncio.run_coroutine_threadsafe(
        _restart_for_calibration(inst, stop_conflicts=stop_conflicts), loop).result()


def _restart_with_ctx(iid, inst, ctx, parallel=1):
    # butler_start_instance(stop_conflicts=True) stops only instances sharing the target GPUs.
    # Measurement always at P=1 for a clean per-token KV (independent of the instance's parallel).
    tmp = dict(inst); tmp["ctx"] = int(ctx); tmp["parallel"] = int(parallel)
    import asyncio
    return _start_from_calibration(tmp, stop_conflicts=True)

def _active_llama_names(except_iid=None):
    try:
        r = subprocess.run(["docker", "ps", "--filter", "name=llama-inst-", "--format", "{{.Names}}"],
                           capture_output=True, text=True, timeout=15)
        return [n for n in r.stdout.strip().split() if not (except_iid and n == f"llama-inst-{except_iid}")]
    except Exception:
        return []

def _restore_instance(iid):
    try:
        insts = load_instances()
        oi = next((x for x in insts if x["id"] == iid), None)
        if not oi:
            return
        tmp = dict(oi); tmp["ctx"] = int(oi.get("ctx", 16384))
        import asyncio
        _start_from_calibration(tmp, stop_conflicts=False)
    except Exception:
        pass

def _calib_worker(iid, inst, ctxA, ctxB, kvq, mp):
    st = {"status": "running", "step": "prep", "result": None, "error": None}
    calib_jobs[iid] = st
    orig_ctx = int(inst.get("ctx", 16384))
    par = 1   # always measure at P=1: clean per-token KV, independent of instance parallel
    gpu_indices = inst.get("gpu_indices", [])
    active_others = _active_llama_names(except_iid=iid)   # other active instances (restore after)
    target_was_active = f"llama-inst-{iid}" in _active_llama_names()
    base_mem = _gpu_used_mib(gpu_indices)
    try:
        st["step"] = "load ctx A"
        if not _restart_with_ctx(iid, inst, ctxA):
            raise RuntimeError("недостаточно VRAM — модель не запустилась. Остановите другие инстансы на этих GPU")
        time.sleep(4)
        memA = _gpu_used_mib(gpu_indices)
        if abs(memA - base_mem) < 512:
            raise RuntimeError(f"модель не загрузилась на GPU (memA={memA} МБ ≈ до запуска {base_mem} МБ). Мало свободной VRAM — остановите другие инстансы")
        st["step"] = "load ctx B"
        if not _restart_with_ctx(iid, inst, ctxB):
            raise RuntimeError("недостаточно VRAM — модель не запустилась. Остановите другие инстансы на этих GPU")
        time.sleep(4)
        memB = _gpu_used_mib(gpu_indices)
        if abs(memB - memA) < 256:
            raise RuntimeError(f"нет дельты памяти (memA={memA}, memB={memB}) — модель, вероятно, не влезла. Остановите другие инстансы")
        dctx = (ctxB - ctxA) * par
        dmem = memB - memA
        if dctx <= 0 or dmem <= 0:
            raise RuntimeError(f"no memory delta (memA={memA}, memB={memB})")
        kvpt = dmem / dctx
        anchor = {"kvpt_mib_per_token": {kvq: round(kvpt, 6)},
                  "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "method": "two-load delta", "ctx_a": ctxA, "ctx_b": ctxB,
                  "parallel": par, "mem_a_mib": memA, "mem_b_mib": memB}
        spath = mp + ".kv.json"
        existing = {}
        if os.path.isfile(spath):
            try: existing = json.load(open(spath))
            except Exception: existing = {}
        kmap = existing.get("kvpt_mib_per_token") or {}
        kmap[kvq] = round(kvpt, 6)
        existing["kvpt_mib_per_token"] = kmap
        existing.update({k: v for k, v in anchor.items() if k != "kvpt_mib_per_token"})
        with open(spath, "w") as f: json.dump(existing, f, indent=1)
        st["status"] = "done"
        st["step"] = "done"
        st["result"] = {"kvpt_mib_per_token": round(kvpt, 6), "kv_kb_per_token": round(kvpt * 1024, 2),
                        "mem_a_mib": memA, "mem_b_mib": memB, "ctx_a": ctxA, "ctx_b": ctxB,
                        "restored": [n.replace("llama-inst-", "") for n in active_others],
                        "sidecar": spath}
    except Exception as e:
        st["status"] = "error"
        st["step"] = "error"
        st["error"] = str(e)[:300]
    finally:
        st["step"] = "restore"
        # Release the target's VRAM first (it was started for measurement)
        subprocess.run(["docker", "stop", f"llama-inst-{iid}"], capture_output=True, timeout=30)
        subprocess.run(["docker", "rm", f"llama-inst-{iid}"], capture_output=True, timeout=20)
        time.sleep(3)
        # Restore the original active set (other active instances + target if it was active)
        to_start = [n.replace("llama-inst-", "") for n in active_others]
        if target_was_active:
            to_start.append(iid)
        for x in to_start:
            _restore_instance(x)

@app.post("/api/instances/{iid}/calibrate-kv")
async def api_calibrate_kv_start(iid: str):
    insts = load_instances()
    inst = next((x for x in insts if x["id"] == iid), None)
    if not inst: return JSONResponse({"success": False, "message": "Instance not found"}, 404)
    mn = inst.get("model", "")
    md = inst_models_dir(inst)
    mp = os.path.join(md, mn)
    if not mn or not os.path.isfile(mp):
        return JSONResponse({"success": False, "message": "No model set"}, 400)
    real_gpus = [g for g in inst.get("gpu_indices", []) if g != "__select__"]
    if not real_gpus or int(inst.get("ngl", 99)) <= 0:
        return JSONResponse({"success": False, "message": "Модель в CPU-режиме (нет GPU или NGL=0) — промер по VRAM не применяется"}, 400)
    if calib_jobs.get(iid, {}).get("status") == "running":
        return JSONResponse({"success": False, "message": "Already calibrating"}, 409)
    kvq = inst.get("kvq", "q8_0")
    ctxA, ctxB = 8192, 32768
    import threading
    t = threading.Thread(target=_calib_worker, args=(iid, inst, ctxA, ctxB, kvq, mp), daemon=True)
    t.start()
    return JSONResponse({"success": True, "job": iid})

@app.get("/api/instances/{iid}/calibrate-kv")
async def api_calibrate_kv_status(iid: str):
    j = calib_jobs.get(iid)
    if not j: return JSONResponse({"status": "none"})
    return JSONResponse(j)

@app.post("/api/butler/mode")
async def api_set_butler_mode(request: Request):
    body = await _read_json_object(request)
    mode = body.get("mode", "OFF").upper()
    if mode not in ("ON", "OFF"): raise HTTPException(400, "Mode must be ON or OFF")
    butler_state["mode"] = mode
    return JSONResponse({"success": True, "mode": mode})

@app.get("/api/butler/status")
async def api_butler_status():
    return JSONResponse(butler_state)

# ======================== GGUF Endpoints ========================

@app.get("/api/models")
async def api_models(dir: str = ""):
    if dir and os.path.isdir(dir):
        return JSONResponse(list_ggufs(dir))
    if _models_dir_cache["dir"] != _current_models_dir:
        refresh_models_cache()
    elif os.path.isdir(_current_models_dir) and os.path.getmtime(_current_models_dir) != _models_dir_cache.get("mtime"):
        refresh_models_cache()
    return JSONResponse(_models_dir_cache["models"])

@app.get("/api/models/dir")
async def api_get_models_dir():
    return JSONResponse({"dir": _current_models_dir})

@app.post("/api/models/refresh")
async def api_refresh_models():
    refresh_models_cache()
    return JSONResponse({"success": True, "count": len(_models_dir_cache["models"])})

@app.get("/api/config")
async def api_get_config():
    c = load_config()
    return JSONResponse({"api_url": c.get("api_url",""), "api_key": c.get("api_key",""), "models_dir": _current_models_dir, "idle_enabled": bool(c.get("idle_enabled", False)), "idle_timeout_minutes": int(c.get("idle_timeout_minutes", 30) or 30), "lan_ip": get_lan_ip()})

@app.post("/api/config")
async def api_set_config(request: Request):
    body = await _read_json_object(request)
    c = load_config()
    _idle_was = bool(c.get("idle_enabled", False))
    for key in ("api_url", "api_key"):
        if key in body: c[key] = body[key]
    if "idle_enabled" in body: c["idle_enabled"] = bool(body["idle_enabled"])
    if "idle_timeout_minutes" in body:
        try: c["idle_timeout_minutes"] = min(9999, max(0, int(body["idle_timeout_minutes"])))
        except (TypeError, ValueError): pass
    if "models_dir" in body and body["models_dir"]:
        if os.path.isdir(body["models_dir"]):
            global _current_models_dir
            _current_models_dir = body["models_dir"]
            refresh_models_cache()
            c["models_dir"] = body["models_dir"]
    save_config(c)
    # idle just enabled -> reset idle timer for all active instances (count from now, not from stale chat)
    if c.get("idle_enabled") and not _idle_was:
        _now = time.time()
        for _iid in get_active_containers().keys():
            _last_active[_iid] = _now
    return JSONResponse({"success": True, **c})

def _docker_image_exists(img):
    try:
        r = subprocess.run(["docker", "image", "inspect", img], capture_output=True, timeout=15)
        return r.returncode == 0
    except Exception:
        return False

@app.get("/api/runtimes")
async def api_runtimes():
    runtimes = []
    seen = set()
    def add(image, label="", default=False):
        image = (image or "").strip()
        if not image or image in seen:
            return
        seen.add(image)
        runtimes.append({"image": image, "label": (label or "").strip() or image, "default": bool(default)})
    add(DOCKER_IMAGE, "llama.cpp (standard)", default=True)
    for r in (load_config().get("runtimes") or []):
        if isinstance(r, dict):
            add(r.get("image"), r.get("label"))
        else:
            add(str(r))
    try:
        out = subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"], capture_output=True, timeout=10, text=True).stdout
        for line in out.splitlines():
            line = line.strip()
            if not line or line.endswith(":<none>"):
                continue
            repo = line.split(":")[0].lower()
            if repo.endswith("llama") or "llama.cpp" in repo:
                add(line)
    except Exception:
        pass
    return JSONResponse({"runtimes": runtimes})

# ======================== HuggingFace Download ========================
HF_API = "https://huggingface.co"
_hf_state = {"status": "idle", "repo": "", "filename": "", "dest": "", "branch": "main", "received": 0, "total": 0, "speed": 0.0, "error": "", "sha256": "", "reset": False}
_hf_lock = threading.Lock()
_hf_stop = threading.Event()

def _hf_update(**kw):
    with _hf_lock:
        _hf_state.update(kw)

def _hf_snapshot():
    with _hf_lock:
        s = dict(_hf_state)
    if s.get("total"):
        s["pct"] = round(min(100.0, s["received"] * 100.0 / s["total"]), 1)
    else:
        s["pct"] = 0
    return s

def _hf_try_download(url, part, dest, sha256):
    received = os.path.getsize(part) if os.path.exists(part) else 0
    total = 0
    h = None
    try:
        with httpx.Client(timeout=httpx.Timeout(120.0, connect=30.0), follow_redirects=True) as client:
            for _pass in (0, 1):
                headers = {"Range": f"bytes={received}-"} if received else {}
                with client.stream("GET", url, headers=headers) as r:
                    if r.status_code == 416 and received and _pass == 0:
                        os.remove(part)
                        received = 0
                        log.info("HF: range not satisfiable, restart from 0")
                        continue
                    if r.status_code not in (200, 206):
                        if 400 <= r.status_code < 500:
                            body = r.read(300).decode("utf-8", "replace")
                            return "fatal", received, total, f"HTTP {r.status_code}: {body[:150]}"
                        return "retry", received, total, f"HTTP {r.status_code}"
                    resumed = r.status_code == 206 and received > 0
                    if not resumed:
                        received = 0
                    cl = int(r.headers.get("content-length") or 0)
                    total = (received + cl) if resumed else cl
                    if sha256:
                        h = hashlib.sha256()
                        if resumed and received:
                            with open(part, "rb") as pf:
                                for c2 in iter(lambda: pf.read(16 * 1024 * 1024), b""):
                                    h.update(c2)
                    last_bytes, last_t = received, time.time()
                    with open(part, "ab" if resumed else "wb") as f:
                        for chunk in r.iter_bytes(1024 * 1024):
                            if _hf_stop.is_set():
                                return "stopped", received, total, ""
                            f.write(chunk)
                            received += len(chunk)
                            if h:
                                h.update(chunk)
                            now = time.time()
                            if now - last_t >= 1.0:
                                spd = (received - last_bytes) / (now - last_t)
                                last_bytes, last_t = received, now
                                _hf_update(status="downloading", received=received, total=total, speed=spd)
                    if h and h.hexdigest().lower() != sha256.lower():
                        os.remove(part)
                        return "fatal", received, total, "sha256 mismatch — file removed"
                    os.replace(part, dest)
                    return "done", received, total, ""
            return "retry", received, total, "range error"
    except Exception as e:
        return "retry", received, total, str(e)[:200]

def _hf_worker(repo, branch, filename, dest, sha256):
    url = f"{HF_API}/{repo}/resolve/{branch}/{filename}"
    part = dest + ".part"
    if os.path.exists(dest):
        _hf_update(status="error", error=f"{os.path.basename(dest)} already exists in models dir", speed=0.0)
        return
    max_retry = 5
    for attempt in range(max_retry + 1):
        if _hf_stop.is_set():
            break
        outcome, received, total, msg = _hf_try_download(url, part, dest, sha256)
        if outcome == "done":
            _hf_update(status="done", received=received, total=total or received, speed=0.0)
            log.info(f"HF: done {filename} -> {dest} ({received} bytes)")
            refresh_models_cache()
            return
        if outcome == "stopped":
            if _hf_snapshot().get("reset"):
                if os.path.exists(part):
                    try: os.remove(part)
                    except: pass
                _hf_update(status="idle", received=0, total=0, speed=0.0, error="")
                log.info(f"HF: canceled {filename} at {received}/{total}")
            else:
                _hf_update(status="paused", received=received, total=total, speed=0.0)
                log.info(f"HF: paused {filename} at {received}/{total}")
            return
        if outcome == "fatal":
            _hf_update(status="error", error=msg, received=received, total=total, speed=0.0)
            log.error(f"HF: fatal {filename}: {msg}")
            return
        if attempt >= max_retry:
            _hf_update(status="error", error=f"{msg} — START to resume", received=received, total=total, speed=0.0)
            log.error(f"HF: retries exhausted {filename}: {msg}")
            return
        delay = min(3 * (2 ** attempt), 15)
        _hf_update(status="retrying", error=msg, received=received, total=total, speed=0.0)
        log.warning(f"HF: retry {attempt + 1}/{max_retry} for {filename} in {delay}s ({msg})")
        slept = 0.0
        while slept < delay:
            if _hf_stop.is_set():
                break
            time.sleep(0.5)
            slept += 0.5
        if _hf_stop.is_set():
            if _hf_snapshot().get("reset"):
                if os.path.exists(part):
                    try: os.remove(part)
                    except: pass
                _hf_update(status="idle", received=0, total=0, speed=0.0, error="")
            else:
                _hf_update(status="paused", received=os.path.getsize(part) if os.path.exists(part) else received, total=total, speed=0.0)
            return
    if _hf_snapshot().get("reset"):
        _hf_update(status="idle", received=0, total=0, speed=0.0, error="")
    else:
        _hf_update(status="paused", speed=0.0)

class HfDownloadReq(BaseModel):
    repo: str
    filename: str
    branch: str = "main"

@app.get("/api/hf/search")
async def api_hf_search(q: str, limit: int = 25):
    q = q.strip()
    if not q:
        return JSONResponse({"error": "empty query"}, status_code=400)
    limit = max(1, min(int(limit), 50))
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as c:
            r = await c.get(f"{HF_API}/api/models", params={"search": q, "filter": "gguf", "sort": "downloads", "limit": limit})
            if r.status_code != 200:
                return JSONResponse({"error": f"HTTP {r.status_code}"}, status_code=502)
            items = r.json()
        results = []
        for it in items:
            mid = it.get("id", "")
            if "/" not in mid:
                continue
            author, _, name = mid.partition("/")
            results.append({"id": mid, "author": author, "name": name, "downloads": it.get("downloads", 0), "likes": it.get("likes", 0)})
        return JSONResponse({"query": q, "results": results})
    except Exception as e:
        return JSONResponse({"error": str(e)[:200]}, status_code=502)

@app.get("/api/hf/files")
async def api_hf_files(repo: str, branch: str = "main"):
    repo = repo.strip().strip("/")
    if "/" not in repo:
        return JSONResponse({"error": "repo must be 'org/name'"}, status_code=400)
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as c:
            r = await c.get(f"{HF_API}/api/models/{repo}/tree/{branch}", params={"recursive": "true"})
            if r.status_code != 200:
                return JSONResponse({"error": f"repo not found (HTTP {r.status_code})"}, status_code=404)
            items = r.json()
        flat = [it for it in items if it.get("type") == "file"]
        files = [{"path": it.get("path", ""), "size": it.get("size") or (it.get("lfs") or {}).get("size") or 0} for it in flat]
        files.sort(key=lambda f: f["path"].lower())
        return JSONResponse({"repo": repo, "branch": branch, "files": files})
    except Exception as e:
        return JSONResponse({"error": str(e)[:200]}, status_code=502)

@app.post("/api/hf/download")
async def api_hf_download(req: HfDownloadReq):
    s = _hf_snapshot()
    if s["status"] in ("downloading", "retrying"):
        return JSONResponse({"error": "Download already in progress"}, status_code=409)
    repo = req.repo.strip().strip("/")
    filename = req.filename.strip().lstrip("/")
    branch = (req.branch or "main").strip()
    if "/" not in repo or not filename:
        return JSONResponse({"error": "repo and filename required"}, status_code=400)
    fname = os.path.basename(filename)
    if not fname or fname.startswith("."):
        return JSONResponse({"error": "invalid filename"}, status_code=400)
    dest = os.path.join(_current_models_dir, fname)
    if os.path.exists(dest):
        return JSONResponse({"error": f"{fname} already exists in models dir"}, status_code=409)
    sha256 = ""
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as c:
            r = await c.get(f"{HF_API}/api/models/{repo}/tree/{branch}", params={"recursive": "true"})
            if r.status_code == 200:
                for it in r.json():
                    if it.get("type") == "file" and it.get("path", "").lower() == filename.lower():
                        sha256 = (it.get("lfs") or {}).get("oid", "") or ""
                        break
    except Exception:
        pass
    _hf_stop.clear()
    _hf_update(status="downloading", repo=repo, filename=filename, dest=dest, branch=branch, received=0, total=0, speed=0.0, error="", sha256=sha256, reset=False)
    t = threading.Thread(target=_hf_worker, args=(repo, branch, filename, dest, sha256), daemon=True)
    t.start()
    return JSONResponse({"success": True, "dest": dest, "sha256": sha256})

@app.get("/api/hf/status")
async def api_hf_status():
    return JSONResponse(_hf_snapshot())

@app.post("/api/hf/pause")
async def api_hf_pause():
    if _hf_snapshot()["status"] not in ("downloading", "retrying"):
        return JSONResponse({"error": "Nothing to pause"}, status_code=409)
    _hf_stop.set()
    return JSONResponse({"success": True, "message": "Pause requested"})

@app.post("/api/hf/resume")
async def api_hf_resume():
    s = _hf_snapshot()
    if s["status"] != "paused":
        return JSONResponse({"error": "Nothing to resume (only from paused)"}, status_code=409)
    if not s.get("filename") or not s.get("repo") or not s.get("dest"):
        return JSONResponse({"error": "No active download to resume"}, status_code=409)
    repo = s["repo"]; branch = s.get("branch") or "main"; filename = s["filename"]; dest = s["dest"]; sha256 = s.get("sha256", "")
    _hf_stop.clear()
    _hf_update(status="downloading", speed=0.0, error="", reset=False)
    t = threading.Thread(target=_hf_worker, args=(repo, branch, filename, dest, sha256), daemon=True)
    t.start()
    return JSONResponse({"success": True, "message": "Resumed", "filename": filename})

@app.post("/api/hf/reset")
async def api_hf_reset():
    s = _hf_snapshot()
    if s["status"] in ("downloading", "retrying"):
        _hf_stop.set()
        _hf_update(reset=True)
        return JSONResponse({"success": True, "message": "Cancel requested"})
    part = (s.get("dest") or "") + ".part"
    if part and os.path.exists(part):
        try: os.remove(part)
        except: pass
    _hf_update(status="idle", received=0, total=0, speed=0.0, error="", reset=False)
    return JSONResponse({"success": True, "message": "Reset"})

@app.get("/api/layers/{model_name}")
async def api_layers(model_name: str):
    params = get_cached_model_params(model_name, _current_models_dir)
    if not params: return JSONResponse({"error": "Model not found"})
    return JSONResponse({"n_layers": params["n_layers"]})

@app.post("/api/calculate")
async def api_calculate(request: Request):
    data = await _read_json_object(request)
    model_name = data.get("model", "")
    md = data.get("dir") or _current_models_dir
    params = get_cached_model_params(model_name, md)
    if not params: return JSONResponse({"error": "Model not found"})
    free_vram = int(data.get("free_vram_gb", 0)) * 1024
    num_gpus = max(1, int(data.get("num_gpus", 1)))
    ngl = min(int(data.get("ngl", 99)), params.get("n_layers", 99))
    flash = data.get("flash_attn", False); ram = data.get("ram_kv", False)
    ctype = data.get("kvq", "fp16")
    _fpath = os.path.join(md, model_name)
    kv_fp16 = get_kvpt_real(_fpath, "fp16") or kv_per_token_mib(params, "fp16")
    kv_q8 = get_kvpt_real(_fpath, "q8_0") or kv_per_token_mib(params, "q8_0")
    kv_q4 = get_kvpt_real(_fpath, "q4_0") or kv_per_token_mib(params, "q4_0")
    kvpt = get_kvpt_real(_fpath, ctype) or kv_per_token_mib(params, ctype)
    mgb = params["size_gb"]; nl = params["n_layers"] if params["n_layers"] > 0 else 40
    # Prefer the real file size (more accurate than GGUF metadata size_gb) for the on-GPU weight footprint
    try:
        file_mib = os.path.getsize(_fpath) // (1024 * 1024)
        if file_mib > 0: mgb = file_mib / 1024.0
    except Exception:
        pass
    model_gpu = mgb * 1024 * (ngl / nl) if nl > 0 else mgb * 1024
    lin_state = params.get("linear_state_mib", 0) or 0
    mmproj_mib = 0
    mp_name = data.get("mmproj", "")
    if mp_name:
        mp = os.path.join(md, mp_name)
        if os.path.isfile(mp):
            mmproj_mib = os.path.getsize(mp) // (1024 * 1024)
    mtp_mib = 0
    mt_name = data.get("mtp", "")
    if mt_name:
        mt = os.path.join(md, mt_name)
        if os.path.isfile(mt):
            mtp_mib = os.path.getsize(mt) // (1024 * 1024)
    extras_mib = mmproj_mib + mtp_mib
    model_max = params.get("context_length", 0) or 0
    parallel = max(1, int(data.get("parallel", 1)))
    oh_fixed = OH_FIXED_MIB
    oh_slot = OH_SLOT_MIB
    slot_ovh_tok = (oh_slot / kvpt) if kvpt > 0 else 0.0
    if ram:
        # checkbox ON (--no-kv-offload): KV in system RAM; model + mmproj + fixed overhead must fit on GPU
        if model_gpu + extras_mib + lin_state + oh_fixed > free_vram * num_gpus:
            cap_raw = 0.0
            budget = 0.0
        else:
            cpu_share = max(0.0, (nl - min(ngl, nl)) / nl)
            ram_model = mgb * 1024 * cpu_share
            ram_avail = max(0, int(_mem_available_mib() * 90 // 100) - ram_model)
            cap_raw = ram_avail / kvpt if kvpt > 0 else 0.0
            budget = max(0.0, cap_raw - parallel * slot_ovh_tok)
    else:
        # checkbox OFF: KV on GPU. cap_raw = raw KV capacity (bracket, P-independent);
        # budget = cap_raw - P x per-slot overhead
        base = free_vram * num_gpus - model_gpu - extras_mib - lin_state - oh_fixed
        cap_raw = base / kvpt if (base > 0 and kvpt > 0) else 0.0
        budget = max(0.0, cap_raw - parallel * slot_ovh_tok)
    cap_raw = max(0.0, cap_raw)
    total = min(parallel * model_max, budget) if model_max > 0 else budget
    slot = total / parallel if parallel > 0 else total
    return JSONResponse({"params": params, "kv_per_token": {"fp16": round(kv_fp16, 4), "q8_0": round(kv_q8, 4), "q4_0": round(kv_q4, 4)},
        "recommended_ctx": int(total), "recommended_ctx_k": round(total/1024, 1),
        "recommended_slot": int(slot), "recommended_cap": int(cap_raw),
        "model_max_ctx": model_max, "parallel": parallel,
        "kvpt_selected": round(kvpt, 4),
        "mla": params.get("mla", False), "key_length": params.get("key_length", 0), "value_length": params.get("value_length", 0)})

# ======================== ChatGPT endpoints ========================

@_serialized_model_start
async def butler_start_instance(inst, stop_conflicts=True):
    """Auto-start an instance. If stop_conflicts, stop conflicting active ones first (exclusive GPU). Returns True on success."""
    iid = inst["id"]
    port = inst.get("port")
    if not port or port == "0":
        return False
    stored = next((item for item in load_instances() if item["id"] == iid), None)
    if stored is not None and str(stored.get("port")) != str(port):
        log.warning("Instance %s changed port while waiting to start", iid)
        return False
    # Another request may have started this instance while we waited for the lock.
    # Wait for readiness rather than issuing a second docker run with the same name.
    active_now = await asyncio.to_thread(get_active_containers)
    if iid in active_now:
        async with httpx.AsyncClient(timeout=2) as hc:
            for _ in range(180):
                try:
                    response = await hc.get(f"http://127.0.0.1:{port}/health")
                    if response.status_code == 200:
                        return True
                except httpx.RequestError:
                    pass
                await asyncio.sleep(1)
        return False
    # Remove a stopped/failed container left by an earlier attempt.
    await asyncio.to_thread(_stop_rm_container, f"llama-inst-{iid}")
    wanted = [str(x).replace("GPU", "") for x in inst.get("gpu_indices", []) if x != "__select__"]
    if stop_conflicts and inst.get("mode", "chat") == "chat":
        instances = load_instances()
        active = get_active_containers()
        stopped_any = False
        for other in instances:
            if other["id"] == iid or other["id"] not in active:
                continue
            if other.get("mode", "chat") != "chat":
                continue
            other_gpus = [str(x).replace("GPU", "") for x in other.get("gpu_indices", []) if x != "__select__"]
            if wanted and any(g in other_gpus for g in wanted):
                await asyncio.to_thread(subprocess.run, ["docker", "stop", f"llama-inst-{other['id']}"], capture_output=True, timeout=20)
                await asyncio.to_thread(subprocess.run, ["docker", "rm", f"llama-inst-{other['id']}"], capture_output=True, timeout=15)
                stopped_any = True
        if stopped_any and wanted:
            await asyncio.to_thread(_wait_gpu_free, wanted, timeout=45)
    # Build docker command
    mn = inst.get("model", "")
    if not mn:
        return False
    md = inst_models_dir(inst)
    mp = os.path.join(md, mn)
    if not os.path.isfile(mp):
        return False
    mn_proj = inst.get("mmproj", "")
    if mn_proj and not os.path.isfile(os.path.join(md, mn_proj)):
        return False
    mn_mtp = inst.get("mtp", "")
    if mn_mtp and mn_mtp != "builtin" and not os.path.isfile(os.path.join(md, mn_mtp)):
        log.error(f"MTP file not found for {iid}: {mn_mtp}")
        return False
    if inst.get("mode", "chat") != "chat":
        mn_proj = ""; mn_mtp = ""
        flash = True
    ctx = int(inst.get("ctx", 16384))
    bs = str(ctx) if inst.get("mode", "chat") in ("embed", "rerank") else "512"
    ngl = int(inst.get("ngl", 99))
    flash = True if inst.get("mode", "chat") != "chat" else bool(inst.get("flash", True))
    ram = bool(inst.get("ram", False))
    kvq = inst.get("kvq", "q8_0")
    parallel = int(inst.get("parallel", 1))
    img = inst.get("docker_image") or DOCKER_IMAGE
    if img != DOCKER_IMAGE and not _docker_image_exists(img):
        log.error(f"runtime image not found locally: {img} for {iid}")
        return False
    cmd = _build_instance_command(inst, model=mn, models_dir=md, ctx=ctx, ngl=ngl,
                                  flash=flash, ram=ram, kvq=kvq, parallel=parallel, gpu_nums=wanted)
    reset_speed_extrema(iid)
    try:
        r = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, timeout=30)
        if r.returncode != 0:
            return False
        # Wait until ready (ASYNC — не блокируем event loop / UI на время загрузки)
        async with httpx.AsyncClient(timeout=2) as hc:
            for _ in range(180):
                await asyncio.sleep(1)
                try:
                    hr = await hc.get(f"http://127.0.0.1:{port}/health")
                    if hr.status_code == 200:
                        await asyncio.to_thread(_apply_active_lora, iid, port)
                        return True
                except Exception:
                    pass
        return False
    except Exception:
        return False


async def _chat_stream_response(url, body, headers, iid):
    client = httpx.AsyncClient(timeout=300.0)
    upstream = None
    closed = False

    async def cleanup():
        nonlocal closed
        if closed:
            return
        closed = True
        try:
            if upstream is not None:
                await upstream.aclose()
        finally:
            try:
                await client.aclose()
            finally:
                inflight_bump(iid, -1)

    try:
        upstream = await client.send(
            client.build_request("POST", url, json=body, headers=headers), stream=True)
        if upstream.status_code >= 400:
            content = await upstream.aread()
            return Response(content=content, status_code=upstream.status_code,
                            media_type=upstream.headers.get("content-type", "application/json"))

        async def chunks():
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                await cleanup()

        response = StreamingResponse(
            chunks(), status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "text/event-stream"),
            background=BackgroundTask(cleanup))
    except httpx.TimeoutException:
        await cleanup()
        return JSONResponse({"error": "Model response timed out"}, status_code=504)
    except httpx.RequestError:
        await cleanup()
        return JSONResponse({"error": "Cannot connect to model"}, status_code=502)
    except BaseException:
        await cleanup()
        raise
    else:
        return response
    finally:
        if upstream is not None and upstream.status_code >= 400:
            await cleanup()


@app.post("/v1/chat/completions")
async def v1_chat(request: Request):
    body = await _read_json_object(request)
    if not await _v1_auth_ok(request):
        return JSONResponse({"error": {"message": "Invalid API Key", "type": "authentication_error", "code": 401}}, status_code=401)
    inst, err = await _v1_route_by_mode(body.get("model", ""), "chat")
    if err:
        return err
    url = f"http://127.0.0.1:{inst['port']}/v1/chat/completions"
    hdrs = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    if body.get("stream", False):
        return await _chat_stream_response(url, body, hdrs, inst["id"])
    try:
        return await _proxy_model_json(url, body, hdrs)
    finally:
        inflight_bump(inst["id"], -1)

@app.get("/v1/models")
async def v1_models(request: Request):
    if not await _v1_auth_ok(request):
        return JSONResponse({"error": {"message": "Invalid API Key", "type": "authentication_error", "code": 401}}, status_code=401)
    instances = load_instances()
    active = get_active_containers()
    chat_instances = [inst for inst in instances if inst.get("mode", "chat") == "chat"]
    name_counts = {}
    instance_ids = {inst["id"] for inst in chat_instances}
    for inst in chat_instances:
        name = inst.get("name") or inst["id"]
        name_counts[name] = name_counts.get(name, 0) + 1
    data = []
    for inst in instances:
        if not inst.get("model") or inst.get("mode", "chat") != "chat":
            continue
        if not inst.get("allowed", True):
            continue  # no permission mark — never passed to clients (regardless of Butler mode)
        if butler_state.get("mode") == "ON" or inst["id"] in active:
            name = inst.get("name") or inst["id"]
            if name_counts[name] > 1 or (name in instance_ids and name != inst["id"]):
                name = inst["id"]
            data.append({"id": name, "object": "model"})
    return JSONResponse({"object": "list", "data": data})



def _secret_equal(value, expected):
    if not isinstance(value, str) or not isinstance(expected, str):
        return False
    return hmac.compare_digest(value.encode("utf-8"), expected.encode("utf-8"))


@app.post("/api/login")
async def api_login(request: Request):
    import json as _js
    body_raw = await request.body()
    if not body_raw:
        return JSONResponse(content={"error": "Empty body"}, status_code=400)
    b = await _read_json_object(request)
    u, p = _get_auth()
    user_ok = _secret_equal(b.get("username"), u)
    password_ok = _secret_equal(b.get("password"), p)
    if not (user_ok and password_ok):
        return JSONResponse(content={"error": "Bad credentials"}, status_code=401)
    return JSONResponse(content={"access_token": _mk_access(u), "refresh_token": _mk_refresh(u), "username": u})

@app.post("/api/refresh")
async def api_refresh(request: Request):
    b = await _read_json_object(request)
    u = _verify(b.get("refresh_token", ""), "r")
    if not u:
        return JSONResponse(content={"error": "Bad token"}, status_code=401)
    return JSONResponse(content={"access_token": _mk_access(u), "refresh_token": _mk_refresh(u), "username": u})

@app.middleware("http")
async def auth_mw(request: Request, call_next):
    p = request.url.path
    if p in ("/", "/api/login", "/api/refresh") or p.startswith("/static/") or p.startswith("/v1/"):
        return await call_next(request)
    t = _get_token(request)
    if not t or not _verify(t):
        return JSONResponse(content={"error": "Unauthorized"}, status_code=401)
    return await call_next(request)




@app.post("/api/change-password")
async def api_change_password(request: Request):
    b = await _read_json_object(request)
    u, p = _get_auth()
    cur = b.get("current_password", "")
    newp = b.get("new_password", "").strip()
    if not _secret_equal(cur, p):
        return JSONResponse(content={"success": False, "message": "Current password is incorrect"}, status_code=403)
    if not newp or len(newp) < 3:
        return JSONResponse(content={"success": False, "message": "New password must be at least 3 characters"}, status_code=400)
    c = load_config()
    c["auth_pass"] = newp
    save_config(c)
    return JSONResponse(content={"success": True, "message": "Password changed"})

@app.get("/api/me")
async def api_me(request: Request):
    from fastapi import Header
    t = _get_token(request)
    u = _verify(t) if t else None
    if not u:
        return JSONResponse(content={"error": "Unauthorized"}, status_code=401)
    return JSONResponse(content={"username": u})


# ======================== Static ========================

@app.get("/", response_class=HTMLResponse)
async def index():
    p = STATIC_DIR / "index.html"
    if p.exists(): return HTMLResponse(content=p.read_text("utf-8"))
    return HTMLResponse("<h1>Butler v2</h1>")

@app.get("/static/{file_path:path}")
async def static_file(file_path: str):
    base = STATIC_DIR.resolve()
    target = (STATIC_DIR / file_path).resolve()
    if not str(target).startswith(str(base) + os.sep):
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    if not target.is_file():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(str(target))


# ======================== Reranker / Embeddings Endpoints (proxy to llama.cpp instance) ========================
async def _v1_auth_ok(request: Request):
    try:
        cfg = json.loads(CONFIG_FILE.read_text("utf-8"))
        expected = cfg.get("api_key", "")
        if not isinstance(expected, str):
            return False
    except (OSError, ValueError, AttributeError):
        log.error("Cannot read API authentication configuration")
        return False
    if not expected:
        return True
    auth = request.headers.get("authorization", "").strip()
    return hmac.compare_digest(auth.encode("utf-8"), ("Bearer " + expected).encode("utf-8"))


async def _proxy_model_json(url, body, headers):
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(url, json=body, headers=headers)
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except httpx.TimeoutException:
        return JSONResponse({"error": "Model response timed out"}, status_code=504)
    except httpx.RequestError:
        return JSONResponse({"error": "Cannot connect to model"}, status_code=502)
    except ValueError:
        return JSONResponse({"error": "Invalid JSON response from model"}, status_code=502)


async def _v1_route_by_mode(model_name, mode):
    """Return an instance with one inflight reservation; caller releases it in finally."""
    instances = load_instances()
    active = get_active_containers()
    candidates = [i for i in instances if i.get("mode", "chat") == mode]
    inst = next((i for i in candidates if i["id"] == model_name), None)
    if inst is None:
        matches = [i for i in candidates if i.get("name") == model_name]
        if len(matches) > 1:
            return None, JSONResponse({"error": "Ambiguous model name; use the instance ID"}, status_code=409)
        inst = matches[0] if matches else None
    if inst is not None and not inst.get("allowed", True):
        return None, JSONResponse({"error": f"{mode} model not allowed (no permission mark)"}, status_code=403)
    if inst is None:
        return None, JSONResponse({"error": f"{mode} model not found"}, status_code=404)
    is_active = inst["id"] in active
    if not is_active and butler_state.get("mode") != "ON":
        return None, JSONResponse({"error": f"{mode} model not active"}, status_code=503)
    iid = inst["id"]
    inflight_bump(iid, 1)
    try:
        mark_active(iid)
        if not is_active and not await butler_start_instance(inst):
            inflight_bump(iid, -1)
            return None, JSONResponse({"error": "Failed to start instance"}, status_code=503)
        return inst, None
    except BaseException:
        inflight_bump(iid, -1)
        raise


@app.post("/v1/rerank")
async def v1_rerank(request: Request):
    body = await _read_json_object(request)
    if not await _v1_auth_ok(request):
        return JSONResponse({"error": {"message": "Invalid API Key", "type": "authentication_error", "code": 401}}, status_code=401)
    inst, err = await _v1_route_by_mode(body.get("model", ""), "rerank")
    if err:
        return err
    try:
        url = f"http://127.0.0.1:{inst['port']}/v1/rerank"
        hdrs = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
        return await _proxy_model_json(url, body, hdrs)
    finally:
        inflight_bump(inst["id"], -1)

@app.post("/v1/embeddings")
async def v1_embeddings(request: Request):
    body = await _read_json_object(request)
    if not await _v1_auth_ok(request):
        return JSONResponse({"error": {"message": "Invalid API Key", "type": "authentication_error", "code": 401}}, status_code=401)
    inst, err = await _v1_route_by_mode(body.get("model", ""), "embed")
    if err:
        return err
    try:
        url = f"http://127.0.0.1:{inst['port']}/v1/embeddings"
        hdrs = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
        return await _proxy_model_json(url, body, hdrs)
    finally:
        inflight_bump(inst["id"], -1)

if __name__ == "__main__":
    import uvicorn
    print(f"Starting Butler v2 on {HOST}:{PORT}, Models: {_current_models_dir}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
