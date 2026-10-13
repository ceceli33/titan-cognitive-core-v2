
# ======================================================================================================================
# TEST639-R2 — AKBASCORE MEMORY ARENA · CLEAN REAL MP RERUN
# REAL LMCACHE MP / CACHEBLEND MP · QWEN2.5-7B · A100 · PYTHON 3.12
# VLLM 0.31.0 · LMCACHE 0.5.5 · SINGLE COLAB CELL
# NO MOCK · NO SOURCE PATCH · NO PREDECLARED WINNER · NO TEST638 MODIFICATION
# INTEGRATION + RAW LOG AUDIT; NOT THE MATCHED 72-CASE BENCHMARK
# ======================================================================================================================
import os,sys,re,json,time,subprocess,urllib.request,urllib.error,hashlib,shutil,signal,traceback,socket,importlib.util
from pathlib import Path
ROOT=Path("/content/AKBASCORE_TEST639");ROOT.mkdir(parents=True,exist_ok=True)
OUT=ROOT/"R2";OUT.mkdir(parents=True,exist_ok=True)
VENV=ROOT/"competitor_env_py312";BIN=VENV/"bin";PY=BIN/"python"
MODEL="Qwen/Qwen2.5-7B-Instruct";VPORT=18049;MPORT=15549;HPORT=19049
RESULT=OUT/"TEST639_R2.json";START=time.monotonic()
def say(*x):print(*x,flush=True)
def save(x):
 tmp=RESULT.with_suffix(".tmp")
 tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
 os.replace(tmp,RESULT)
def run(cmd,timeout=120,env=None):
 p=subprocess.run([str(x) for x in cmd],capture_output=True,text=True,timeout=timeout,env=env)
 return p.returncode,p.stdout,p.stderr
def tail(path,n=14000):
 if not path.exists():return ""
 with path.open("rb") as f:
  f.seek(0,2);f.seek(max(0,f.tell()-n))
  return f.read().decode("utf-8","replace")
def sha(path):
 if not path.exists():return None
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def request(url,payload=None,timeout=10):
 data=None if payload is None else json.dumps(payload).encode()
 req=urllib.request.Request(url,data=data,headers={"Content-Type":"application/json"})
 with urllib.request.urlopen(req,timeout=timeout) as r:return json.loads(r.read().decode())
def launch(cmd,log,env):
 with log.open("w",encoding="utf-8") as f:
  return subprocess.Popen([str(x) for x in cmd],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
def stop(p):
 if p is None or p.poll() is not None:return
 try:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=20)
 except Exception:
  try:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
  except Exception:pass
def ready(url,p,timeout):
 end=time.monotonic()+timeout
 while time.monotonic()<end:
  if p.poll() is not None:return False
  try:request(url,timeout=3);return True
  except Exception:time.sleep(3)
 return False
def port_free(port):
 with socket.socket() as s:
  s.settimeout(1)
  return s.connect_ex(("127.0.0.1",port))!=0
def loglines(path,pattern,limit=80):
 if not path.exists():return []
 with path.open(encoding="utf-8",errors="replace") as f:
  found=[s.strip()[-600:] for s in f if re.search(pattern,s,re.I)]
 return found[-limit:]
def install(cmd,log,timeout):
 with log.open("w",encoding="utf-8") as f:
  p=subprocess.run([str(x) for x in cmd],stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
 if p.returncode:raise RuntimeError("INSTALL_FAILED: "+tail(log,9000))
R={"test":"639-R2","status":"STARTED","model":MODEL,"winner":"NOT_ASSIGNED",
 "previous_results_unchanged":True,"arms":{},"limitations":[
 "Integration smoke test; not the locked TEST638 72-case evaluation.",
 "HTTP latency is not directly comparable with TEST638 native inference.",
 "Faster repeated prompts alone do not prove external KV retrieval.",
 "Successful CacheBlend startup alone does not prove a non-prefix blending operation.",
 "Any startup failure is reported as an integration failure, not an accuracy loss."]}
save(R)
say("="*116);say("TEST639-R2 — MEMORY ARENA · REAL LMCACHE / CACHEBLEND MP");say("="*116)
try:
 say("[1/6] GPU")
 code,out,err=run(["nvidia-smi","--query-gpu=name,memory.total,memory.used","--format=csv,noheader"])
 if code:raise RuntimeError("GPU_AUDIT_FAILED: "+err)
 R["gpu"]=out.strip();say(R["gpu"])
 m=re.search(r",\s*(\d+)\s*MiB\s*$",out.strip().splitlines()[0])
 if m and int(m.group(1))>9000:raise RuntimeError("GPU_BUSY: use a fresh runtime")
 say("[2/6] PYTHON 3.12 ENVIRONMENT")
 uv=shutil.which("uv")
 if not uv:
  install([sys.executable,"-m","pip","install","-U","uv"],OUT/"uv_install.log",300)
  uv=shutil.which("uv")
  if not uv:
   spec=importlib.util.find_spec("uv")
   if spec:
    candidate=Path(spec.origin).parent/"uv"
    if candidate.exists():uv=str(candidate)
  if not uv:raise RuntimeError("UV_NOT_FOUND")
 R["uv"]=uv
 if not PY.exists():
  code,out,err=run([uv,"venv","--python","3.12","--python-preference","managed",VENV],timeout=600)
  if code:raise RuntimeError("UV_VENV_FAILED: "+err[-6000:]+out[-2000:])
 code,out,err=run([PY,"-c","import sys;print(sys.version_info[:2])"])
 if code or "(3, 12)" not in out:raise RuntimeError("PYTHON_312_REQUIRED: "+out+err)
 R["python"]=out.strip();say("PYTHON",R["python"])
 say("[3/6] VERSION-LOCKED COMPETITOR INSTALL")
 probe=[PY,"-c","import importlib.metadata as m;print(m.version('vllm'));print(m.version('lmcache'))"]
 code,out,err=run(probe)
 if code or out.strip().splitlines()!=["0.31.0","0.5.5"] or not (BIN/"vllm").exists() or not (BIN/"lmcache").exists():
  install([uv,"pip","install","--python",PY,"vllm==0.31.0","lmcache==0.5.5"],
   OUT/"competitor_install.log",2400)
 code,out,err=run(probe)
 if code or out.strip().splitlines()!=["0.31.0","0.5.5"]:raise RuntimeError("VERSION_MISMATCH: "+out+err)
 R["versions"]={"vllm":"0.31.0","lmcache":"0.5.5"};say("VLLM 0.31.0 · LMCACHE 0.5.5")
 say("[4/6] CONNECTOR / CLI / PORT AUDIT")
 code,out,err=run([PY,"-c",
  "import importlib.util;"
  "m=importlib.util.find_spec('lmcache.integration.vllm.lmcache_mp_connector');"
  "print(m.origin if m else 'NOT_FOUND')"])
 if code or "NOT_FOUND" in out:raise RuntimeError("EXTERNAL_MP_CONNECTOR_MISSING: "+out+err)
 R["external_connector_module"]=out.strip()
 code,out,err=run([BIN/"lmcache","server","--help"],timeout=120)
 h=out+"\n"+err
 (OUT/"lmcache_server_help.txt").write_text(h,encoding="utf-8")
 if code:raise RuntimeError("MP_CLI_HELP_FAILED: "+h[-5000:])
 for option in ("--host","--port","--http-port","--engine-type","--l1-size-gb","--eviction-policy","--chunk-size","--supported-transfer-mode"):
  if option not in h:raise RuntimeError("MP_OPTION_MISSING: "+option)
 R["mp_help_sha256"]=hashlib.sha256(h.encode()).hexdigest()
 code,out,err=run([BIN/"vllm","serve","--help"],timeout=120)
 h=out+"\n"+err
 (OUT/"vllm_serve_help.txt").write_text(h,encoding="utf-8")
 R["vllm_help_returncode"]=code
 R["vllm_help_sha256"]=hashlib.sha256(h.encode()).hexdigest()
 if code or "--kv-transfer-config" not in h:
  say("VLLM_HELP_INCONCLUSIVE: command-line help did not expose KV option; checking Python parser.")
  check=(
   "from vllm.engine.arg_utils import AsyncEngineArgs;"
   "import dataclasses;"
   "f={x.name for x in dataclasses.fields(AsyncEngineArgs)};"
   "print('kv_transfer_config' in f)"
  )
  c,o,e=run([PY,"-c",check],timeout=120)
  R["kv_parser_probe"]={"returncode":c,"stdout":o.strip(),"stderr":e[-3000:]}
  if c or o.strip()!="True":
   raise RuntimeError("VLLM_KV_CONFIG_NOT_CONFIRMED: "+h[-2000:]+e[-3000:])
 R["kv_config_supported"]=True
 if not all(port_free(p) for p in (VPORT,MPORT,HPORT)):raise RuntimeError("PORT_BUSY")
 R["status"]="PRECHECK_PASS";save(R)
except Exception as e:
 R["status"]="PRECHECK_FAILED";R["error"]=repr(e);R["traceback"]=traceback.format_exc()
 R["elapsed_seconds"]=round(time.monotonic()-START,3);save(R)
 say("PRECHECK_FAILED",repr(e));say("JSON_OUTPUT",RESULT)
 raise
ENV=os.environ.copy()
ENV.update({"PYTHONUNBUFFERED":"1","TOKENIZERS_PARALLELISM":"false",
 "HF_HUB_DISABLE_TELEMETRY":"1","VLLM_WORKER_MULTIPROC_METHOD":"spawn"})
A="The blue archive contains the silver compass. "*110
B="The eastern gallery contains the copper telescope. "*110
C="The western observatory contains the glass prism. "*110
PREFIX="Research notes:\n";QUESTION="\nQuestion: Where is the silver compass?\nAnswer:"
def prompt(parts):return PREFIX+"\n\n".join(parts)+QUESTION
FIRST=prompt([A,B,C]);REORDERED=prompt([C,A,B])
PROMPTS=[("COLD",FIRST),("REORDERED",REORDERED),("REPEATED",FIRST)]
def arm(name,blend):
 say("[ARM]",name,"ENGINE_TYPE","blend" if blend else "default")
 entry={"status":"STARTING","requests":[],"blend_configured":blend,
  "cache_hit_verified":False,"blend_operation_verified":False}
 R["arms"][name]=entry;save(R)
 mp_log=OUT/(name+"_mp.log");vllm_log=OUT/(name+"_vllm.log")
 cp=vp=None
 try:
  if not all(port_free(p) for p in (VPORT,MPORT,HPORT)):raise RuntimeError("PORT_BUSY_BEFORE_ARM")
  mp_cmd=[BIN/"lmcache","server","--host","127.0.0.1","--port",str(MPORT),
   "--http-port",str(HPORT),"--l1-size-gb","4","--eviction-policy","LRU",
   "--chunk-size","256","--engine-type","blend" if blend else "default",
   "--supported-transfer-mode","lmcache_driven"]
  entry["mp_command"]=[str(x) for x in mp_cmd]
  cp=launch(mp_cmd,mp_log,ENV)
  if not ready(f"http://127.0.0.1:{HPORT}/healthcheck",cp,120):
   raise RuntimeError("MP_STARTUP_FAILED: "+tail(mp_log,8000))
  say(name,"MP_SERVER_READY")
  connector={"kv_connector":"LMCacheMPConnector",
   "kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector",
   "kv_role":"kv_both",
   "kv_connector_extra_config":{"lmcache.mp.host":"127.0.0.1",
    "lmcache.mp.port":MPORT,"lmcache.mp.autostart":False,
    "lmcache.mp.mp_transfer_mode":"lmcache_driven"}}
  vcmd=[BIN/"vllm","serve",MODEL,"--host","127.0.0.1","--port",str(VPORT),
   "--dtype","bfloat16","--max-model-len","8192","--max-num-seqs","4",
   "--gpu-memory-utilization","0.82","--enforce-eager",
   "--no-enable-prefix-caching","--kv-transfer-config",json.dumps(connector)]
  entry["vllm_command"]=[str(x) for x in vcmd]
  vp=launch(vcmd,vllm_log,ENV)
  if not ready(f"http://127.0.0.1:{VPORT}/v1/models",vp,480):
   raise RuntimeError("VLLM_STARTUP_FAILED: "+tail(vllm_log,10000))
  say(name,"VLLM_SERVER_READY")
  for label,p in PROMPTS:
   t=time.perf_counter()
   output=request(f"http://127.0.0.1:{VPORT}/v1/completions",
    {"model":MODEL,"prompt":p,"temperature":0,"max_tokens":16},180)
   elapsed=round((time.perf_counter()-t)*1000,4)
   choices=output.get("choices",[])
   if not choices:raise RuntimeError("EMPTY_COMPLETION")
   ans=choices[0].get("text","")
   entry["requests"].append({"label":label,"answer":ans,"elapsed_ms":elapsed,
    "usage":output.get("usage"),"request_id":output.get("id")})
   say(name,label,"MS",elapsed,"ANSWER",repr(ans[:110]))
   save(R)
  entry["status"]="INFERENCE_COMPLETE_OPERATION_UNVERIFIED"
 except Exception as e:
  entry["status"]="FAILED";entry["error"]=repr(e);entry["traceback"]=traceback.format_exc()
  say(name,"FAILED",repr(e)[:1500])
 finally:
  stop(vp);stop(cp)
  entry["mp_log"]=str(mp_log);entry["vllm_log"]=str(vllm_log)
  entry["mp_log_sha256"]=sha(mp_log);entry["vllm_log_sha256"]=sha(vllm_log)
  entry["mp_retrieval_candidates"]=loglines(mp_log,r"retriev|cache.hit|cache_hit|fetch|lookup")
  entry["mp_blend_candidates"]=loglines(mp_log,r"blend|recomput|selective") if blend else []
  entry["vllm_retrieval_candidates"]=loglines(vllm_log,r"retriev|cache.hit|cache_hit|fetch|lookup")
  entry["vllm_blend_candidates"]=loglines(vllm_log,r"blend|recomput|selective") if blend else []
  entry["cache_hit_verified"]=False
  entry["blend_operation_verified"]=False
  save(R)
 return entry
say("[5/6] REAL LMCACHE MP")
a=arm("LMCACHE_MP",False)
say("[6/6] REAL CACHEBLEND MP")
b=arm("CACHEBLEND_MP",True)
R["status"]="INTEGRATION_COMPLETE_OPERATION_UNVERIFIED" if all(
 x["status"]=="INFERENCE_COMPLETE_OPERATION_UNVERIFIED" and len(x["requests"])==3
 for x in (a,b)) else "INTEGRATION_PARTIAL"
R["elapsed_seconds"]=round(time.monotonic()-START,3);R["winner"]="NOT_ASSIGNED"
R["next_step"]="Verify actual external KV retrieval and non-prefix blending; then run the matched 72-case Arena benchmark."
save(R)
say("="*116)
say("TEST639_R2_STATUS",R["status"])
for name,x in R["arms"].items():
 say("ARM",name,"STATUS",x["status"],"REQUESTS",len(x["requests"]),
  "HIT_CANDIDATES",len(x.get("mp_retrieval_candidates",[]))+len(x.get("vllm_retrieval_candidates",[])),
  "BLEND_CANDIDATES",len(x.get("mp_blend_candidates",[]))+len(x.get("vllm_blend_candidates",[])),
  "CACHE_HIT_VERIFIED",x["cache_hit_verified"],"BLEND_VERIFIED",x["blend_operation_verified"])
 if x.get("error"):say("ERROR",x["error"][:1800])
say("WINNER",R["winner"])
say("JSON_OUTPUT",RESULT)
say("TOTAL_SECONDS",R["elapsed_seconds"])
say("="*116)
