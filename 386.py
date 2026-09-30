# ==================================================================================================
# TEST 386 — AKBASCORE NIRVANA × MISTRAL — AUTOMATED LOSSY-WALL CALIBRATION
# TEST 385 WORKING PKV/PCA BASELINE | ONE CELL · ONE RUN | BF16 · SDPA · GREEDY · A100
# D=[32,64,80,96,128] × SOURCE TOKENS=[128,512,2048,4096,10000]
# EXACT PKV / D-SWEEP / POSITION RETRIEVAL / COUNTERFACTUAL / STORAGE / VRAM / TIMING
# NO GRADIO | NO SEASC | NO TRAINING | NO LoRA | NO WEIGHT UPDATES
# ==================================================================================================
import os,sys,time,json,math,random,hashlib,gc,traceback,subprocess,importlib.util
from pathlib import Path
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
 if importlib.util.find_spec(m)is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import torch,transformers
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
torch.set_grad_enabled(False)
SEED=386;random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEV="cuda";MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";DIMS=[32,64,80,96,128];LENGTHS=[128,512,2048,4096,10000]
MAX_NEW=24;SEP="\n\n";FMT="QUESTION:\n{q}\n\nANSWER:";MAX_GPU_GB=37.5
ROOT=Path("/content/AKBASCORE_TEST386")if os.path.isdir("/content")else Path("/tmp/AKBASCORE_TEST386")
ROOT.mkdir(parents=True,exist_ok=True)
LOG=ROOT/"TEST386_RESULTS.jsonl";SUMMARY=ROOT/"TEST386_SUMMARY.json";LOG.write_text("",encoding="utf-8")
FACTS=[
 ("Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge.","Who planted the Turkish flag at the base of the Golden Gate Bridge?","Mustafa Akbaş","Leyla Demir planted the Turkish flag at the base of the Golden Gate Bridge.","Leyla Demir"),
 ("Elena Varga stored the silver compass inside the northern archive of Tallinn.","Who stored the silver compass inside the northern archive of Tallinn?","Elena Varga","Tomasz Brenner stored the silver compass inside the northern archive of Tallinn.","Tomasz Brenner"),
 ("Kerem Yıldız repaired the brass lantern at the old lighthouse of Sinop.","Who repaired the brass lantern at the old lighthouse of Sinop?","Kerem Yıldız","Marta Solberg repaired the brass lantern at the old lighthouse of Sinop.","Marta Solberg")]
NEUTRAL=[
 "Jonas Weber carried the wooden crate across the quiet market square.",
 "Priya Nair wrote a long letter to her cousin in the morning.",
 "The small boat drifted slowly toward the rocky shore.",
 "Omar Haddad fixed the broken clock in the village school.",
 "A tired teacher closed the green door of the library.",
 "Sofia Rossi baked fresh bread for the harvest festival.",
 "The children watched the kites rising above the hill.",
 "Liam O'Connor sold his old bicycle to a neighbor.",
 "Heavy rain flooded the narrow street near the market.",
 "Nadia Petrova translated the ancient manuscript into French.",
 "The farmer counted the sheep before sunset.",
 "Hiro Sato opened a tiny bakery beside the river.",
 "An old dog slept under the kitchen table all afternoon.",
 "Carlos Mendes washed the windows of his grandmother's house.",
 "The museum guard locked the heavy gate at midnight.",
 "Fatima Zahra grew tomatoes in the backyard garden.",
 "The pilot announced a short delay because of fog.",
 "Anna Kowalski found a lost wallet on the bus.",
 "Snow covered the mountain village during the night.",
 "Ravi Kumar taught his brother how to play chess.",
 "The chef sharpened every knife before dinner service.",
 "Lucas Martin cleaned the roof of the barn after the storm.",
 "A young violinist practiced scales in the empty hall.",
 "Mei Lin delivered the package to the wrong address.",
 "Marek Novak placed the glass bottle beneath the wooden bench.",
 "Sara Ibrahim carried a red notebook into the quiet classroom.",
 "Noah Schmidt repaired the small radio beside the kitchen window.",
 "Yuki Mori left a paper envelope near the station entrance.",
 "Amira Hassan moved the ceramic bowl onto the upper shelf.",
 "Peter Novak opened the metal box behind the old theater.",
 "Lucia Costa placed the yellow scarf inside the travel bag.",
 "Daniel Kim carried a black umbrella through the central courtyard."]
def sync():torch.cuda.synchronize()
def sec(t):return round(time.perf_counter()-t,4)
def clean():gc.collect();torch.cuda.empty_cache()
def norm(s):return " ".join("".join(c.lower()if c.isalnum()else" "for c in s).split())
def hit(s,a):return int(norm(a)in norm(s))
def record(x):
 with LOG.open("a",encoding="utf-8")as f:f.write(json.dumps(x,ensure_ascii=False,default=str)+"\n");f.flush()
def gb(x):return round(x/1024**3,4)
def mem():return gb(torch.cuda.memory_allocated())
def peak():sync();return gb(torch.cuda.max_memory_allocated())
def reset_peak():sync();torch.cuda.reset_peak_memory_stats()
def fail(s):raise RuntimeError(s)
print("="*116,"\nTEST 386 — NIRVANA × MISTRAL — AUTOMATED LOSSY-WALL CALIBRATION\n"+"="*116)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/9] MODEL")
t0=time.perf_counter()
tv=tuple(int("".join(c for c in x if c.isdigit())or 0)for x in transformers.__version__.split(".")[:2])
dt="dtype"if tv>=(4,56)else"torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{dt:torch.bfloat16})
model.eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;layers=model.model.layers
NL=len(layers);H=cfg.hidden_size;NH=cfg.num_attention_heads;NKV=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None)or H//NH;KVD=NKV*HD
if(NL,H,NH,NKV,HD)!=(32,4096,32,8,128):fail(f"Architecture mismatch: {(NL,H,NH,NKV,HD)}")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
ge=model.generation_config.eos_token_id;EOS=sorted({tok.eos_token_id,*(ge if isinstance(ge,(list,tuple))else[ge])}-{None})
CTX_LIMIT=int(getattr(cfg,"max_position_embeddings",32768))
print(f"Model={MODEL_ID} | layers={NL} | KV heads={NKV} | head={HD} | context_limit={CTX_LIMIT} | load={sec(t0)}s")
FP=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
def sentinel():
 h=hashlib.sha256()
 for p in FP:
  a=p.detach().reshape(-1);n=a.numel()
  for j in range(16):
   off=j*(n-256)//15
   h.update(a[off:off+256].float().cpu().numpy().tobytes())
 return h.hexdigest()
S0=sentinel()
def ids(s):return tok(s,add_special_tokens=False).input_ids
def new_cache():
 try:return DynamicCache(config=cfg)
 except Exception:return DynamicCache()
def kvget(c,L):
 if hasattr(c,"layers"):return c.layers[L].keys,c.layers[L].values
 return c.key_cache[L],c.value_cache[L]
def cache_of(pkv):
 K,V,T=pkv;c=new_cache()
 for L in range(NL):c.update(K[L].clone(),V[L].clone(),L)
 if c.get_seq_length()!=T:fail("Cache length mismatch.")
 return c
@torch.inference_mode()
def forge(s):
 seq=[PAD]+ids(s+SEP)
 if len(seq)>CTX_LIMIT:fail(f"Source exceeds model context: {len(seq)} > {CTX_LIMIT}")
 out=model(input_ids=torch.tensor([seq],device=DEV),use_cache=False,output_hidden_states=True,return_dict=True)
 K=[];V=[]
 for L in range(NL):
  z=layers[L].input_layernorm(out.hidden_states[L][0]);a=layers[L].self_attn
  K.append(a.k_proj(z).contiguous());V.append(a.v_proj(z).contiguous())
 return {"T":len(seq),"K":K,"V":V}
@torch.inference_mode()
def install(K,V):
 T=K[0].shape[0];pos=torch.arange(T,device=DEV)[None]
 cos,sin=model.model.rotary_emb(K[0][None],pos)
 KK=[];VV=[]
 for L in range(NL):
  k=K[L].reshape(1,T,NKV,HD).transpose(1,2).contiguous()
  v=V[L].reshape(1,T,NKV,HD).transpose(1,2).contiguous()
  kr,_=apply_rotary_pos_emb(k,k,cos,sin,unsqueeze_dim=1)
  KK.append(kr.contiguous());VV.append(v)
 return tuple(KK),tuple(VV),T
print("[2/9] FACT-INDEPENDENT PCA CODEBOOK")
t0=time.perf_counter()
CO=[forge(s)for s in NEUTRAL]
SINK={n:[CO[0][n][L][:1].clone()for L in range(NL)]for n in("K","V")}
CB=[];EXPL={"K":[],"V":[]}
for L in range(NL):
 e={}
 for n in("K","V"):
  R=torch.cat([c[n][L][1:]for c in CO]).float().reshape(-1,NKV,HD)
  mu=[];basis=[];ev=[]
  for h in range(NKV):
   X=R[:,h];m=X.mean(0);_,s,vh=torch.linalg.svd(X-m,full_matrices=False)
   b=vh[:128].T.contiguous()
   if b.shape[1]<128:b=F.pad(b,(0,128-b.shape[1]))
   v=s.square();ev.append({d:float(v[:d].sum()/v.sum().clamp_min(1e-12))for d in DIMS})
   mu.append(m);basis.append(b)
  e[n]=(torch.stack(mu),torch.stack(basis));EXPL[n].append(ev)
 CB.append(e)
del CO;clean()
print(f"Codebook ready | {sec(t0)}s | training corpus={len(NEUTRAL)} fact-independent sentences")
for d in DIMS:
 print(f" D{d:03d} | K explained={sum(h[d]for L in EXPL['K']for h in L)/(NL*NKV):.5f} | V explained={sum(h[d]for L in EXPL['V']for h in L)/(NL*NKV):.5f}")
@torch.inference_mode()
def encode(f,d):
 return {n:[torch.einsum("thi,hid->thd",f[n][L][1:].float().reshape(-1,NKV,HD)-CB[L][n][0],CB[L][n][1][:,:,:d]).to(torch.bfloat16)for L in range(NL)]for n in("K","V")}
@torch.inference_mode()
def decode(code,d):
 out={}
 for n in("K","V"):
  out[n]=[torch.cat([SINK[n][L],(CB[L][n][0]+torch.einsum("thd,hid->thi",code[n][L].float(),CB[L][n][1][:,:,:d])).reshape(-1,KVD).to(torch.bfloat16)],0)for L in range(NL)]
 return out["K"],out["V"]
def bytes_code(code):return sum(x.numel()*x.element_size()for n in("K","V")for x in code[n])
def bytes_pkv(pkv):return sum(x.numel()*x.element_size()for n in pkv[:2]for x in n)
CB_BYTES=sum(t.numel()*t.element_size()for e in CB for n in("K","V")for t in e[n])
SINK_BYTES=sum(t.numel()*t.element_size()for n in("K","V")for t in SINK[n])
print(f"Codebook={CB_BYTES:,} bytes | shared sink={SINK_BYTES:,} bytes | GPU allocated={mem()} GiB")
@torch.inference_mode()
def gen(q,kind,obj=None):
 qids=ids(FMT.format(q=q))
 if kind=="visible":pre=[PAD]+ids(obj+SEP);c=None
 elif kind=="memory":pre=[PAD]*obj[2];c=cache_of(obj)
 elif kind=="vanilla":pre=[];c=None
 else:fail(f"Unknown mode: {kind}")
 x=torch.tensor([pre+qids],device=DEV)
 if x.shape[1]+MAX_NEW>CTX_LIMIT:fail("Generation exceeds context limit.")
 sync();t=time.perf_counter()
 y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),past_key_values=c,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=PAD,eos_token_id=EOS)
 sync();new=y[0,x.shape[1]:].tolist()
 return {"text":tok.decode(new,skip_special_tokens=True).strip(),"tokens":len(new),"seconds":sec(t)}
print("[3/9] NATIVE PKV RECONSTRUCTION AUDIT")
f0=forge(FACTS[0][0])
native=model(input_ids=torch.tensor([[PAD]+ids(FACTS[0][0]+SEP)],device=DEV),use_cache=True).past_key_values
exact0=install(f0["K"],f0["V"])
ke=max(float((exact0[0][L].float()-kvget(native,L)[0].float()).norm()/kvget(native,L)[0].float().norm().clamp_min(1e-12))for L in range(NL))
ve=max(float((exact0[1][L].float()-kvget(native,L)[1].float()).norm()/kvget(native,L)[1].float().norm().clamp_min(1e-12))for L in range(NL))
print(f"K error={ke:.9e} | V error={ve:.9e}")
if max(ke,ve)>0.05:fail("Native PKV reconstruction audit failed.")
del f0,native,exact0;clean()
print("[4/9] DETERMINISTIC LONG-SOURCE GENERATOR")
DISTRACTORS=[
 "The municipal records office catalogued an ordinary delivery receipt.",
 "A passenger placed a newspaper on an empty bench.",
 "The workshop supervisor checked the inventory before closing.",
 "Several visitors walked through the courtyard without stopping.",
 "A clerk copied the date into an administrative register.",
 "The gardener watered the trees beside the stone pathway.",
 "A technician inspected the lighting near the entrance.",
 "The archive assistant arranged unlabeled folders on a shelf."]
def source_at(target,position,fact):
 fact_ids=ids(fact+"\n")
 filler=[];j=0
 while len(filler)<target+64:
  filler+=ids(DISTRACTORS[j%len(DISTRACTORS)]+"\n");j+=1
 if position=="START":left=0
 elif position=="MIDDLE":left=max(0,(target-len(fact_ids))//2)
 else:left=max(0,target-len(fact_ids))
 body=filler[:left]+fact_ids+filler[left:target-len(fact_ids)]
 if len(body)<target:body+=filler[:target-len(body)]
 return tok.decode(body,skip_special_tokens=False),left,len(body)
def safe_run(fn,label,meta):
 try:return fn()
 except torch.cuda.OutOfMemoryError as ex:
  record({"status":"OOM","condition":label,**meta,"error":str(ex)[:250]})
  print(" OOM:",label);clean();return None
 except Exception as ex:
  record({"status":"ERROR","condition":label,**meta,"error":f"{type(ex).__name__}: {str(ex)[:300]}"})
  print(" ERROR:",label,type(ex).__name__,str(ex)[:180]);clean();return None
def score(q,kind,obj,target,label,meta):
 reset_peak();r=safe_run(lambda:gen(q,kind,obj),label,meta)
 if r is None:return None
 r.update({"status":"OK","condition":label,"target":target,"hit":hit(r["text"],target),"peak_allocated_gib":peak(),**meta})
 record(r)
 print(f"  {label:19s} | hit={r['hit']} | {r['seconds']:7.3f}s | peak={r['peak_allocated_gib']:6.2f} GiB | {r['text'][:100]}")
 return r
print("[5/9] SHORT-SOURCE COUNTERFACTUAL CONTROLS")
for i,(src,q,ans,cf,cf_ans)in enumerate(FACTS,1):
 print("-"*116,f"\nSHORT CASE {i}/{len(FACTS)}")
 f=forge(src);fc=forge(cf);meta={"group":"SHORT","case":i,"source_tokens":f["T"]}
 score(q,"vanilla",None,ans,"VANILLA",meta)
 score(q,"visible",src,ans,"VISIBLE_SOURCE",meta)
 pkv=install(f["K"],f["V"])
 score(q,"memory",pkv,ans,"EXACT_PKV",meta)
 del pkv
 for d in DIMS:
  code=encode(f,d)
  pkv=install(*decode(code,d))
  score(q,"memory",pkv,ans,f"D{d}",{**meta,"D":d,"compressed_bytes":bytes_code(code),"native_pkv_bytes":bytes_pkv(pkv)})
  del code,pkv;clean()
 code=encode(fc,64)
 pkv=install(*decode(code,64))
 score(q,"memory",pkv,cf_ans,"COUNTERFACTUAL_D64",{**meta,"D":64,"counterfactual":True})
 del f,fc,code,pkv;clean()
print("[6/9] LONG-CONTEXT POSITION × DIMENSION SWEEP")
for target in LENGTHS:
 print("="*116,f"\nSOURCE TARGET {target:,} TOKENS")
 if target+MAX_NEW+100>CTX_LIMIT:
  record({"status":"SKIP_CONTEXT","target_tokens":target,"limit":CTX_LIMIT})
  print(" SKIP: model context limit");continue
 for pi,position in enumerate(("START","MIDDLE","END")):
  src,left,bodylen=source_at(target,position,FACTS[pi][0])
  q=FACTS[pi][1];ans=FACTS[pi][2]
  meta={"group":"LONG","target_tokens":target,"position":position,"case":pi+1,"fact_start_token":left,"body_tokens":bodylen}
  print(f"\n POSITION={position} | fact offset={left} | generated body={bodylen} tokens")
  if target>=4096 and mem()>MAX_GPU_GB:
   record({"status":"SKIP_VRAM","condition":"FORGE","allocated_gib":mem(),**meta})
   print(" SKIP: GPU allocation threshold");continue
  reset_peak();t=time.perf_counter()
  f=safe_run(lambda:forge(src),"FORGE",meta)
  if f is None:continue
  forge_s=sec(t);forge_peak=peak()
  meta.update({"source_tokens":f["T"],"forge_seconds":forge_s,"forge_peak_gib":forge_peak})
  print(f" FORGE | slots={f['T']} | {forge_s:.3f}s | peak={forge_peak:.2f} GiB")
  if f["T"]+len(ids(FMT.format(q=q)))+MAX_NEW>CTX_LIMIT:
   record({"status":"SKIP_CONTEXT","condition":"READOUT","limit":CTX_LIMIT,**meta})
   del f;clean();continue
  pkv=safe_run(lambda:install(f["K"],f["V"]),"INSTALL_EXACT",meta)
  exact=None
  if pkv is not None:
   exact=score(q,"memory",pkv,ans,"EXACT_PKV",meta)
   del pkv;clean()
  if exact is None:
   record({"status":"SKIP_INVALID_BASELINE","reason":"Exact PKV failed to execute",**meta})
   del f;clean();continue
  for d in DIMS:
   t=time.perf_counter()
   code=safe_run(lambda:encode(f,d),f"ENCODE_D{d}",meta)
   if code is None:continue
   enc_s=sec(t);cbytes=bytes_code(code)
   reset_peak();t=time.perf_counter()
   decoded=safe_run(lambda:decode(code,d),f"DECODE_D{d}",meta)
   if decoded is None:
    del code;clean();continue
   dec_s=sec(t)
   pkv=safe_run(lambda:install(*decoded),f"INSTALL_D{d}",meta)
   if pkv is None:
    del code,decoded;clean();continue
   info={**meta,"D":d,"encode_seconds":enc_s,"decode_seconds":dec_s,"compressed_bytes":cbytes,
         "shared_codebook_bytes":CB_BYTES,"shared_sink_bytes":SINK_BYTES,"native_pkv_bytes":bytes_pkv(pkv),
         "coefficient_ratio":round(cbytes/bytes_pkv(pkv),6),"exact_pkv_hit":exact["hit"]}
   score(q,"memory",pkv,ans,f"D{d}",info)
   del code,decoded,pkv;clean()
  del f;clean()
 print(" COMPLETED LENGTH:",target,"| intermediate results saved:",LOG)
print("[7/9] AGGREGATE")
rows=[json.loads(s)for s in LOG.read_text(encoding="utf-8").splitlines()if s.strip()]
ok=[r for r in rows if r.get("status")=="OK"]
for group in("SHORT","LONG"):
 print("\n",group)
 for name in(["VANILLA","VISIBLE_SOURCE","EXACT_PKV"]+[f"D{d}"for d in DIMS]+["COUNTERFACTUAL_D64"]):
  a=[r for r in ok if r.get("group")==group and r.get("condition")==name]
  if a:print(f" {name:20s}: {sum(r['hit']for r in a):3d}/{len(a):3d} | mean time={sum(r['seconds']for r in a)/len(a):.3f}s")
print("[8/9] LOSSY-WALL TABLE")
wall=[]
for target in LENGTHS:
 exact=[r for r in ok if r.get("group")=="LONG"and r.get("target_tokens")==target and r.get("condition")=="EXACT_PKV"]
 if not exact:continue
 print(f"\nLENGTH {target:,} | EXACT={sum(r['hit']for r in exact)}/{len(exact)}")
 for d in DIMS:
  a=[r for r in ok if r.get("group")=="LONG"and r.get("target_tokens")==target and r.get("condition")==f"D{d}"]
  matched=[(e,r)for e in exact for r in a if e["position"]==r["position"]]
  losses=sum(e["hit"]==1 and r["hit"]==0 for e,r in matched)
  gains=sum(e["hit"]==0 and r["hit"]==1 for e,r in matched)
  rec={"length":target,"D":d,"exact_hits":sum(e["hit"]for e,r in matched),"compressed_hits":sum(r["hit"]for e,r in matched),
       "matched":len(matched),"losses_vs_exact":losses,"gains_vs_exact":gains}
  wall.append(rec)
  print(f" D{d:03d} | {rec['compressed_hits']}/{rec['matched']} | exact={rec['exact_hits']}/{rec['matched']} | lost={losses} | gained={gains}")
print("[9/9] INTEGRITY + FILES")
S1=sentinel()
if S1!=S0:fail("WEIGHT SENTINEL CHANGED.")
if model.training or any(p.requires_grad for p in model.parameters()):fail("Frozen-model audit failed.")
report={"test":386,"model":MODEL_ID,"seed":SEED,"dimensions":DIMS,"lengths":LENGTHS,
        "native_reconstruction_error":{"K":ke,"V":ve},"codebook_bytes":CB_BYTES,"sink_bytes":SINK_BYTES,
        "codebook_corpus_size":len(NEUTRAL),"sentinel_before":S0,"sentinel_after":S1,
        "rows_total":len(rows),"rows_ok":len(ok),"lossy_wall_table":wall,
        "notes":["Exact PKV is the per-source baseline; its failures are not attributed to compression.",
                 "Substring target matching is a screening metric, not a full semantic evaluation.",
                 "Coefficient-only ratios exclude shared codebook, sink and full reconstructed PKV.",
                 "Long-context source construction uses deterministic neutral distractors and three fact positions.",
                 "Single seed and three positions per length do not establish a population-level failure threshold.",
                 "An OOM or execution error is recorded separately from an incorrect model answer."]}
SUMMARY.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(f"WEIGHT SENTINEL: PASS | VALID RESULTS={len(ok)} | RECORDED EVENTS={len(rows)}")
print("RAW LOG :",LOG)
print("SUMMARY :",SUMMARY)
print("="*116)
