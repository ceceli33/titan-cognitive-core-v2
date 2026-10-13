# TEST641 — AKBASCORE MEMORY ARENA · CACHEBLEND NON-PREFIX AUDIT
# TEST640 / TEST639-R2 MP BACKBONE; VLLM 0.31.0, LMCACHE 0.5.5, PYTHON 3.12
import os,sys,re,json,time,subprocess,urllib.request,hashlib,shutil,signal,traceback,socket
from pathlib import Path
ROOT=Path('/content/AKBASCORE_TEST639');VENV=ROOT/'competitor_env_py312';BIN=VENV/'bin';PY=BIN/'python'
OUT=Path('/content/AKBASCORE_TEST641/NONPREFIX_AUDIT');OUT.mkdir(parents=True,exist_ok=True)
MODEL='Qwen/Qwen2.5-7B-Instruct';VPORT=18049;MPORT=15549;HPORT=19049
RESULT=OUT/'TEST641_NONPREFIX_AUDIT.json';START=time.monotonic()
def say(*x):print(*x,flush=True)
def save(x):
 p=RESULT.with_suffix('.tmp');p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str),encoding='utf-8');os.replace(p,RESULT)
def run(cmd,timeout=120):
 p=subprocess.run([str(x) for x in cmd],capture_output=True,text=True,timeout=timeout);return p.returncode,p.stdout,p.stderr
def tail(path,n=12000):
 if not path.exists():return ''
 with path.open('rb') as f:f.seek(0,2);f.seek(max(0,f.tell()-n));return f.read().decode('utf-8','replace')
def sha(path):
 if not path.exists():return None
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def http(url,payload=None,timeout=10,raw=False):
 data=None if payload is None else json.dumps(payload).encode()
 req=urllib.request.Request(url,data=data,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=timeout) as r:
  data=r.read().decode('utf-8','replace');return data if raw else json.loads(data)
def launch(cmd,log,env):
 with log.open('w',encoding='utf-8') as f:return subprocess.Popen([str(x) for x in cmd],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
def stop(p):
 if p is None or p.poll() is not None:return
 try:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=20)
 except Exception:
  try:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
  except Exception:pass
def ready(url,p,seconds):
 end=time.monotonic()+seconds
 while time.monotonic()<end:
  if p.poll() is not None:return False
  try:http(url,timeout=3);return True
  except Exception:time.sleep(3)
 return False
def port_free(port):
 with socket.socket() as s:s.settimeout(1);return s.connect_ex(('127.0.0.1',port))!=0
def install(cmd,log,timeout):
 with log.open('w',encoding='utf-8') as f:p=subprocess.run([str(x) for x in cmd],stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
 if p.returncode:raise RuntimeError('INSTALL_FAILED: '+tail(log,8000))
def snapshot(label,arm):
 result={}
 for source,url in [('MP',f'http://127.0.0.1:{HPORT}/metrics'),('VLLM',f'http://127.0.0.1:{VPORT}/metrics')]:
  try:
   raw=http(url,timeout=12,raw=True);path=OUT/f'{arm}_{label}_{source}.txt';path.write_text(raw,encoding='utf-8');vals={}
   for line in raw.splitlines():
    if not line or line.startswith('#'):continue
    m=re.match(r'^([a-zA-Z_:][\w:]*)(?:\{([^}]*)\})?\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)(?:\s|$)',line)
    if m:vals[m.group(1)+('{'+m.group(2)+'}' if m.group(2) else '')]=float(m.group(3))
   result[source]={'available':True,'values':vals,'path':str(path),'sha256':hashlib.sha256(raw.encode()).hexdigest()}
  except Exception as e:result[source]={'available':False,'values':{},'error':repr(e)}
 return result
def delta(before,after):
 out={}
 for source in ('MP','VLLM'):
  a=before[source]['values'];b=after[source]['values']
  out[source]={k:round(v-a[k],6) for k,v in b.items() if k in a and v-a[k]>0 and re.search(r'cache|store|retriev|read|write|load|blend|chunk|recomput|prefix|lookup|transfer',k,re.I)}
 return out
def sum_metric(d,source,metric):return sum(v for k,v in d.get(source,{}).items() if k.split('{',1)[0]==metric)
def lines(path,pattern,limit=120):
 if not path.exists():return []
 with path.open(encoding='utf-8',errors='replace') as f:rows=[x.strip()[-800:] for x in f if re.search(pattern,x,re.I)]
 return rows[-limit:]
R={'test':'641','status':'STARTED','model':MODEL,'winner':'NOT_ASSIGNED','previous_results_unchanged':True,'arms':{},'goal':'Compare default MP and blend MP on exact-repeat versus reordered-chunk requests; inspect operation-level evidence','rules':['Prefix hits on repeated prompts do not establish non-prefix blending','Blend engine configuration alone does not establish execution','Reordered prompt correctness and speed alone do not establish blending','Only explicit operation evidence may verify blending']};save(R)
say('='*112);say('TEST641 — MEMORY ARENA · CACHEBLEND NON-PREFIX AUDIT');say('='*112)
try:
 say('[1/6] GPU')
 c,o,e=run(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader'])
 if c:raise RuntimeError('GPU_AUDIT_FAILED: '+e)
 R['gpu']=o.strip();say(o.strip())
 say('[2/6] ENVIRONMENT')
 uv=shutil.which('uv')
 if not uv:
  install([sys.executable,'-m','pip','install','-U','uv'],OUT/'uv_install.log',300);uv=shutil.which('uv')
 if not uv:raise RuntimeError('UV_NOT_FOUND')
 if not PY.exists():
  c,o,e=run([uv,'venv','--python','3.12','--python-preference','managed',VENV],600)
  if c:raise RuntimeError('VENV_FAILED: '+e[-5000:])
 probe=[PY,'-c',"import importlib.metadata as m;print(m.version('vllm'));print(m.version('lmcache'))"]
 c,o,e=run(probe)
 if c or o.strip().splitlines()!=['0.31.0','0.5.5'] or not (BIN/'vllm').exists() or not (BIN/'lmcache').exists():
  install([uv,'pip','install','--python',PY,'vllm==0.31.0','lmcache==0.5.5'],OUT/'install.log',2400)
 c,o,e=run(probe)
 if c or o.strip().splitlines()!=['0.31.0','0.5.5']:raise RuntimeError('VERSION_MISMATCH: '+o+e)
 if not all(port_free(x) for x in (VPORT,MPORT,HPORT)):raise RuntimeError('PORT_BUSY')
 R['versions']={'vllm':'0.31.0','lmcache':'0.5.5'};R['status']='PRECHECK_PASS';save(R);say('VLLM 0.31.0 · LMCACHE 0.5.5')
except Exception as e:
 R['status']='PRECHECK_FAILED';R['error']=repr(e);R['traceback']=traceback.format_exc();save(R);say('PRECHECK_FAILED',repr(e));raise
ENV=os.environ.copy();ENV.update({'PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false','HF_HUB_DISABLE_TELEMETRY':'1','VLLM_WORKER_MULTIPROC_METHOD':'spawn'})
DOCS=['The blue archive contains the silver compass. '*110,'The eastern gallery contains the copper telescope. '*110,'The western observatory contains the glass prism. '*110]
def prompt(order):return 'Research notes:\n'+'\n\n'.join(DOCS[i] for i in order)+'\nQuestion: Where is the silver compass?\nAnswer:'
P0=prompt([0,1,2]);P1=prompt([2,0,1]);P2=prompt([1,2,0]);P3=prompt([2,1,0])
TASKS=[('COLD_ABC',P0),('REPEAT_ABC',P0),('REORDER_CAB',P1),('REPEAT_CAB',P1),('REORDER_BCA',P2),('REORDER_CBA',P3),('REPEAT_CBA',P3)]
def arm(name,blend):
 entry={'status':'STARTING','blend_engine_configured':blend,'requests':[],'kv_store_verified':False,'kv_retrieve_verified':False,'nonprefix_cache_hit_observed':False,'blend_operation_verified':False};R['arms'][name]=entry;save(R)
 mp_log=OUT/f'{name}_mp.log';vllm_log=OUT/f'{name}_vllm.log';cp=vp=None
 try:
  if not all(port_free(x) for x in (VPORT,MPORT,HPORT)):raise RuntimeError('PORT_BUSY_BEFORE_ARM')
  mcmd=[BIN/'lmcache','server','--host','127.0.0.1','--port',str(MPORT),'--http-port',str(HPORT),'--l1-size-gb','4','--eviction-policy','LRU','--chunk-size','256','--engine-type','blend' if blend else 'default','--supported-transfer-mode','lmcache_driven']
  entry['mp_command']=[str(x) for x in mcmd];cp=launch(mcmd,mp_log,ENV)
  if not ready(f'http://127.0.0.1:{HPORT}/healthcheck',cp,120):raise RuntimeError('MP_STARTUP_FAILED: '+tail(mp_log,9000))
  say(name,'MP_READY')
  connector={'kv_connector':'LMCacheMPConnector','kv_connector_module_path':'lmcache.integration.vllm.lmcache_mp_connector','kv_role':'kv_both','kv_connector_extra_config':{'lmcache.mp.host':'127.0.0.1','lmcache.mp.port':MPORT,'lmcache.mp.autostart':False,'lmcache.mp.mp_transfer_mode':'lmcache_driven'}}
  vcmd=[BIN/'vllm','serve',MODEL,'--host','127.0.0.1','--port',str(VPORT),'--dtype','bfloat16','--max-model-len','8192','--max-num-seqs','4','--gpu-memory-utilization','0.82','--enforce-eager','--no-enable-prefix-caching','--kv-transfer-config',json.dumps(connector)]
  entry['vllm_command']=[str(x) for x in vcmd];vp=launch(vcmd,vllm_log,ENV)
  if not ready(f'http://127.0.0.1:{VPORT}/v1/models',vp,480):raise RuntimeError('VLLM_STARTUP_FAILED: '+tail(vllm_log,12000))
  say(name,'VLLM_READY');baseline=snapshot('BASELINE',name)
  for label,p in TASKS:
   before=snapshot(label+'_BEFORE',name);t=time.perf_counter()
   output=http(f'http://127.0.0.1:{VPORT}/v1/completions',{'model':MODEL,'prompt':p,'temperature':0,'max_tokens':16},180)
   ms=round((time.perf_counter()-t)*1000,4);choices=output.get('choices',[])
   if not choices:raise RuntimeError('EMPTY_COMPLETION')
   ans=choices[0].get('text','');time.sleep(2);after=snapshot(label+'_AFTER',name);d=delta(before,after)
   hit=sum_metric(d,'MP','lmcache_mp_lookup_hit_tokens_total');req=sum_metric(d,'MP','lmcache_mp_lookup_requested_tokens_total')
   row={'label':label,'answer':ans,'correct':'blue archive' in ans.lower(),'elapsed_ms':ms,'prompt_sha256':hashlib.sha256(p.encode()).hexdigest(),'usage':output.get('usage'),'mp_lookup_hit_tokens':hit,'mp_lookup_requested_tokens':req,'mp_l1_read_chunks':sum_metric(d,'MP','lmcache_mp_l1_read_chunks_total'),'mp_l1_write_chunks':sum_metric(d,'MP','lmcache_mp_l1_write_chunks_total'),'mp_finished_retrieves':sum_metric(d,'MP','lmcache_mp_num_finished_retrieves_total'),'mp_finished_stores':sum_metric(d,'MP','lmcache_mp_num_finished_stores_total'),'vllm_external_hit_tokens':sum_metric(d,'VLLM','vllm:external_prefix_cache_hits_total'),'metric_delta':d,'metric_files':{src:{'before':before[src].get('path'),'after':after[src].get('path')} for src in ('MP','VLLM')}}
   entry['requests'].append(row);say(name,label,'MS',ms,'CORRECT',row['correct'],'MP_HIT',hit,'REQUESTED',req,'L1_READ',row['mp_l1_read_chunks'],'L1_WRITE',row['mp_l1_write_chunks']);save(R)
  total=delta(baseline,snapshot('FINAL',name));entry['total_metric_delta']=total
  entry['kv_store_verified']=any(x['mp_l1_write_chunks']>0 or x['mp_finished_stores']>0 for x in entry['requests'])
  entry['kv_retrieve_verified']=any(x['mp_l1_read_chunks']>0 and x['mp_finished_retrieves']>0 for x in entry['requests'])
  entry['nonprefix_cache_hit_observed']=any(x['mp_lookup_hit_tokens']>0 for x in entry['requests'] if x['label'].startswith('REORDER_'))
  entry['blend_metric_candidates']={src:{k:v for k,v in vals.items() if re.search(r'blend|recomput|selective',k,re.I)} for src,vals in total.items()}
  entry['status']='INFERENCE_COMPLETE_AUDIT_REQUIRED'
 except Exception as e:
  entry['status']='FAILED';entry['error']=repr(e);entry['traceback']=traceback.format_exc();say(name,'FAILED',repr(e)[:1800])
 finally:
  stop(vp);stop(cp);entry['mp_log']=str(mp_log);entry['vllm_log']=str(vllm_log);entry['mp_log_sha256']=sha(mp_log);entry['vllm_log_sha256']=sha(vllm_log)
  entry['mp_blend_log_candidates']=lines(mp_log,r'blend|recomput|selective|non.prefix');entry['vllm_blend_log_candidates']=lines(vllm_log,r'blend|recomput|selective|non.prefix')
  entry['mp_cache_log_candidates']=lines(mp_log,r'lookup|retriev|cache.hit|store');save(R)
 return entry
say('[3/6] LMCACHE DEFAULT CONTROL');a=arm('LMCACHE_DEFAULT',False)
say('[4/6] CACHEBLEND EXPERIMENT');b=arm('CACHEBLEND',True)
say('[5/6] DIFFERENTIAL AUDIT')
for name,x in R['arms'].items():
 say(name,'STORE',x['kv_store_verified'],'RETRIEVE',x['kv_retrieve_verified'],'REORDER_HIT',x['nonprefix_cache_hit_observed'],'BLEND_METRIC_CANDIDATES',sum(len(v) for v in x.get('blend_metric_candidates',{}).values()))
R['status']='INFERENCE_COMPLETE_BLEND_UNVERIFIED' if all(x['status']=='INFERENCE_COMPLETE_AUDIT_REQUIRED' and len(x['requests'])==len(TASKS) for x in (a,b)) else 'INTEGRATION_PARTIAL'
R['elapsed_seconds']=round(time.monotonic()-START,3);R['winner']='NOT_ASSIGNED';save(R)
say('[6/6] FINAL');say('='*112);say('TEST641_STATUS',R['status'])
for name,x in R['arms'].items():
 say('ARM',name,'STATUS',x['status'],'REQUESTS',len(x['requests']),'STORE_VERIFIED',x['kv_store_verified'],'RETRIEVE_VERIFIED',x['kv_retrieve_verified'],'NONPREFIX_HIT_OBSERVED',x['nonprefix_cache_hit_observed'],'BLEND_OPERATION_VERIFIED',x['blend_operation_verified'])
 if x.get('error'):say('ERROR',x['error'][:1800])
say('WINNER',R['winner']);say('JSON_OUTPUT',RESULT);say('TOTAL_SECONDS',R['elapsed_seconds']);say('='*112)
