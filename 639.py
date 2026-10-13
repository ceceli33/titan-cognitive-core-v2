
# ======================================================================================================================
# TEST639 — AKBASCORE MEMORY ARENA · TIER 2 · REAL INTEGRATION AUDIT · CORRECTED
# REAL LMCACHE CONNECTOR V1 vs REAL LMCACHE CACHEBLEND CONFIGURATION
# QWEN2.5-7B-INSTRUCT · ISOLATED PYTHON 3.12 · FRESH A100 COLAB · SINGLE CELL
# NO TEST566 MODIFICATION · NO TEST638 MODIFICATION · NO SIMULATED RESULTS
# NO WINNER WITHOUT MATCHED BENCHMARK · FAILURES AND RAW LOGS PRESERVED
# ======================================================================================================================
import os,sys,json,time,subprocess,urllib.request,urllib.error,hashlib,re,shutil,signal,traceback,platform,importlib.util
from pathlib import Path
TEST="639";ROOT=Path("/content/AKBASCORE_TEST639");ROOT.mkdir(parents=True,exist_ok=True)
VENV=ROOT/"competitor_env_py312";PY=VENV/"bin/python";BIN=VENV/"bin"
MODEL="Qwen/Qwen2.5-7B-Instruct";PORT=18039
RESULT=ROOT/"TEST639_REAL_INTEGRATION.json";START=time.monotonic()
TIMEOUT_INSTALL=2400;TIMEOUT_START=480;TIMEOUT_REQUEST=180
def say(*x):print(*x,flush=True)
def save(data):
 p=RESULT.with_suffix(".tmp")
 with open(p,"w",encoding="utf-8") as f:json.dump(data,f,indent=2,ensure_ascii=False,default=str)
 os.replace(p,RESULT)
def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def run(cmd,timeout=60,env=None):
 p=subprocess.run([str(x) for x in cmd],capture_output=True,text=True,timeout=timeout,env=env)
 return {"returncode":p.returncode,"stdout":p.stdout[-10000:],"stderr":p.stderr[-10000:]}
def tail(path,n=24000):
 if not path.exists():return ""
 with open(path,"rb") as f:
  f.seek(0,2);f.seek(max(0,f.tell()-n))
  return f.read().decode("utf-8","replace")
def http_json(url,payload=None,timeout=10):
 data=None if payload is None else json.dumps(payload).encode()
 req=urllib.request.Request(url,data=data,headers={"Content-Type":"application/json"})
 with urllib.request.urlopen(req,timeout=timeout) as r:return json.loads(r.read().decode())
def alive(p):return p is not None and p.poll() is None
def stop(p):
 if not alive(p):return
 try:
  os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=20)
 except Exception:
  try:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
  except Exception:pass
def launch(cmd,log,env):
 with open(log,"w",encoding="utf-8") as f:
  return subprocess.Popen([str(x) for x in cmd],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
def wait_http(url,process,timeout):
 end=time.monotonic()+timeout
 while time.monotonic()<end:
  if not alive(process):return False
  try:
   http_json(url,timeout=3);return True
  except Exception:time.sleep(3)
 return False
def log_lines(path,pattern,limit=50):
 if not path.exists():return []
 with open(path,"r",encoding="utf-8",errors="replace") as f:
  lines=[x.strip()[-600:] for x in f if re.search(pattern,x,re.I)]
 return lines[-limit:]
REPORT={"test":TEST,"stage":"REAL_COMPETITOR_INTEGRATION","model":MODEL,"status":"STARTED",
 "model_weights_modified":False,"test638_results_modified":False,"arms":{},
 "arena_winner":"NOT_ASSIGNED","connector":"LMCacheConnectorV1",
 "limitations":["Independent environment; timings cannot be compared directly to TEST638.",
 "Smoke prompts are not the locked 72-case panel.",
 "Inference success does not establish cache reuse or blending.",
 "CacheBlend requires operation-level evidence before a successful blend can be claimed.",
 "The two arms use sequential independent engine processes."]}
save(REPORT)
say("="*116)
say("TEST639 — MEMORY ARENA · TIER 2 · REAL LMCACHE / CACHEBLEND INTEGRATION")
say("="*116)
try:
 if not shutil.which("nvidia-smi"):raise RuntimeError("NVIDIA_GPU_REQUIRED")
 gpu=run(["nvidia-smi","--query-gpu=name,memory.total,memory.used","--format=csv,noheader"])
 if gpu["returncode"]:raise RuntimeError("GPU_AUDIT_FAILED")
 REPORT["gpu"]=gpu["stdout"].strip();say("GPU",REPORT["gpu"])
 match=re.search(r",\s*(\d+)\s*MiB\s*$",REPORT["gpu"].splitlines()[0])
 if match and int(match.group(1))>9000:raise RuntimeError("GPU_BUSY_RESTART_FRESH_RUNTIME")
 say("[1/5] ISOLATED PYTHON 3.12 ENVIRONMENT")
 uv=shutil.which("uv")
 if not uv:
  install_uv=ROOT/"install_uv.log"
  with open(install_uv,"w") as f:
   p=subprocess.run([sys.executable,"-m","pip","install","--upgrade","uv"],
    stdout=f,stderr=subprocess.STDOUT,timeout=240)
  if p.returncode:raise RuntimeError("UV_INSTALL_FAILED: "+tail(install_uv,5000))
  uv=shutil.which("uv")
  if not uv:
   candidate=Path(sys.executable).parent/"uv"
   if candidate.exists():uv=str(candidate)
  if not uv:
   candidate=Path.home()/".local/bin/uv"
   if candidate.exists():uv=str(candidate)
  if not uv:
   import uv as uv_module
   candidate=Path(uv_module.__file__).parent/"uv"
   if candidate.exists():uv=str(candidate)
  if not uv:raise RuntimeError("UV_EXECUTABLE_NOT_FOUND")
 say("UV",uv)
 if not PY.exists():
  p=run([uv,"venv","--python","3.12","--python-preference","managed",str(VENV)],timeout=600)
  if p["returncode"]:raise RuntimeError("UV_VENV_FAILED: "+p["stderr"]+p["stdout"])
 version=run([PY,"-c","import sys;print(sys.version)"])
 if version["returncode"]:raise RuntimeError("ISOLATED_PYTHON_FAILED")
 say("ISOLATED_PYTHON",version["stdout"].strip())
 if not (BIN/"vllm").exists():
  log=ROOT/"install_competitors.log"
  with open(log,"w") as f:
   p=subprocess.run([uv,"pip","install","--python",str(PY),"vllm","lmcache"],
    stdout=f,stderr=subprocess.STDOUT,timeout=TIMEOUT_INSTALL)
  if p.returncode:raise RuntimeError("COMPETITOR_INSTALL_FAILED: "+tail(log,8000))
 versions=run([PY,"-c",
  "import torch,vllm,lmcache,importlib.metadata as m;"
  "print('torch',torch.__version__);"
  "print('vllm',m.version('vllm'));"
  "print('lmcache',m.version('lmcache'))"],timeout=120)
 if versions["returncode"]:raise RuntimeError("COMPETITOR_IMPORT_FAILED: "+versions["stderr"])
 REPORT["versions"]=versions["stdout"].strip();say(REPORT["versions"])
 if not (BIN/"vllm").exists():raise RuntimeError("VLLM_EXECUTABLE_MISSING")
 REPORT["status"]="ENVIRONMENT_READY";save(REPORT)
except Exception as e:
 REPORT["status"]="ENVIRONMENT_FAILED";REPORT["environment_error"]=repr(e)
 REPORT["environment_traceback"]=traceback.format_exc()
 REPORT["total_seconds"]=round(time.monotonic()-START,3);save(REPORT)
 say("TEST639_STATUS",REPORT["status"]);say("ERROR",repr(e));say("JSON_OUTPUT",RESULT)
 raise
# Real in-process LMCache V1 integration; no standalone MP server is claimed.
# CacheBlend is enabled using documented environment configuration and layerwise retrieval.
base_env=os.environ.copy()
base_env.update({"PYTHONUNBUFFERED":"1","TOKENIZERS_PARALLELISM":"false",
 "VLLM_WORKER_MULTIPROC_METHOD":"spawn","HF_HUB_DISABLE_TELEMETRY":"1",
 "LMCACHE_CHUNK_SIZE":"256","LMCACHE_LOCAL_CPU":"True",
 "LMCACHE_MAX_LOCAL_CPU_SIZE":"5.0","LMCACHE_LOCAL_DISK":"False"})
# Long repeated passages allow chunk-aligned reuse. The reordered request challenges non-prefix reuse.
A="The blue archive contains the silver compass. "*120
B="The eastern gallery contains the copper telescope. "*120
C="The western observatory contains the glass prism. "*120
SEP=" # # "
PREFIX="Research notes. Answer the question using the records below.\n"
Q="\nQuestion: Where is the silver compass?\nAnswer:"
def prompt(order,blend):
 sep=SEP if blend else "\n\n"
 return PREFIX+sep+sep.join(order)+sep+Q
def prompts(blend):
 first=prompt([A,B,C],blend)
 second=prompt([C,A,B],blend)
 return [("COLD",first),("REORDERED",second),("REPEATED",first)]
def run_arm(name,blend):
 say("[ARM]",name,"BLEND_ENABLED",blend)
 entry={"arm":name,"status":"STARTING","requests":[],"blend_enabled":blend,
  "cache_reuse_verified":False,"blend_operation_verified":False}
 REPORT["arms"][name]=entry;save(REPORT)
 env=base_env.copy()
 env["LMCACHE_ENABLE_BLENDING"]="True" if blend else "False"
 env["LMCACHE_USE_LAYERWISE"]="True" if blend else "False"
 env["LMCACHE_BLEND_SPECIAL_STR"]=SEP
 env["LMCACHE_BLEND_CHECK_LAYERS"]="1"
 env["LMCACHE_BLEND_RECOMPUTE_RATIOS"]="0.15"
 env["LMCACHE_EXTRA_CONFIG"]=json.dumps({"lmcache_instance_id":"akbascore_test639_"+name.lower()})
 log=ROOT/f"{name}_vllm.log";proc=None
 try:
  connector=json.dumps({"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"})
  cmd=[BIN/"vllm","serve",MODEL,"--host","127.0.0.1","--port",str(PORT),
   "--dtype","bfloat16","--max-model-len","8192","--max-num-seqs","4",
   "--gpu-memory-utilization","0.82","--enforce-eager",
   "--no-enable-prefix-caching","--kv-transfer-config",connector]
  entry["command"]=[str(x) for x in cmd]
  proc=launch(cmd,log,env)
  if not wait_http(f"http://127.0.0.1:{PORT}/v1/models",proc,TIMEOUT_START):
   raise RuntimeError("VLLM_STARTUP_FAILED: "+tail(log,9000))
  entry["server_ready"]=True;say(name,"SERVER_READY")
  for label,text in prompts(blend):
   payload={"model":MODEL,"prompt":text,"temperature":0,"max_tokens":16}
   t=time.perf_counter()
   response=http_json(f"http://127.0.0.1:{PORT}/v1/completions",payload,TIMEOUT_REQUEST)
   elapsed=round((time.perf_counter()-t)*1000,5)
   choices=response.get("choices",[])
   if not choices:raise RuntimeError("EMPTY_COMPLETION")
   answer=choices[0].get("text","")
   entry["requests"].append({"label":label,"answer":answer,"latency_ms":elapsed,
    "prompt_chars":len(text),"usage":response.get("usage"),"request_id":response.get("id")})
   say(name,label,"MS",elapsed,"ANSWER",repr(answer[:100]))
   save(REPORT)
  entry["status"]="INFERENCE_COMPLETE_OPERATION_UNVERIFIED"
 except Exception as e:
  entry["status"]="FAILED";entry["error"]=repr(e)
  entry["traceback"]=traceback.format_exc()
  say(name,"FAILED",repr(e))
 finally:
  stop(proc)
  entry["log_path"]=str(log)
  entry["log_sha256"]=sha(log) if log.exists() else None
  entry["retrieval_evidence_lines"]=log_lines(log,r"retriev(ed|ing)|cache.hit|cache_hit|num_retrieved")
  entry["storage_evidence_lines"]=log_lines(log,r"stor(ed|ing)|num_stored|store.*chunk")
  entry["blend_evidence_lines"]=log_lines(log,r"blend|recomput|selective") if blend else []
  # Log mentions are retained as evidence candidates, not proof of an actual cache hit or blending.
  entry["retrieval_log_candidates"]=len(entry["retrieval_evidence_lines"])
  entry["blend_log_candidates"]=len(entry["blend_evidence_lines"])
  entry["cache_reuse_verified"]=False
  entry["blend_operation_verified"]=False
  save(REPORT)
 return entry
say("[2/5] LMCACHE REAL CONNECTOR V1")
a=run_arm("LMCACHE_V1",False)
say("[3/5] CACHEBLEND REAL CONFIGURATION")
b=run_arm("CACHEBLEND_V1",True)
say("[4/5] OPERATION EVIDENCE AUDIT")
REPORT["status"]="INTEGRATION_COMPLETE_REUSE_UNVERIFIED" if all(
 x["status"]=="INFERENCE_COMPLETE_OPERATION_UNVERIFIED" and len(x["requests"])==3
 for x in (a,b)) else "INTEGRATION_PARTIAL"
REPORT["next_requirement"]="Confirm operation-level KV retrieval and blending, then execute matched 72-case Arena evaluation."
REPORT["arena_winner"]="NOT_ASSIGNED"
REPORT["total_seconds"]=round(time.monotonic()-START,3)
save(REPORT)
say("[5/5] FINAL SCOREBOARD")
say("="*116)
say("TEST639_STATUS",REPORT["status"])
for name,x in REPORT["arms"].items():
 say("ARM",name,"STATUS",x["status"],"REQUESTS",len(x["requests"]),
  "RETRIEVAL_LOG_CANDIDATES",x.get("retrieval_log_candidates"),
  "BLEND_LOG_CANDIDATES",x.get("blend_log_candidates"),
  "CACHE_REUSE_VERIFIED",x.get("cache_reuse_verified"),
  "BLEND_OPERATION_VERIFIED",x.get("blend_operation_verified"))
 if x.get("error"):say("ERROR",x["error"])
say("ARENA_WINNER",REPORT["arena_winner"])
say("JSON_OUTPUT",RESULT)
say("TOTAL_SECONDS",REPORT["total_seconds"])
say("="*116)
