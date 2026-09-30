# ==================================================================================================
# TEST 387 — AKBASCORE NIRVANA × MISTRAL — CROSS-POSITION & RECONSTRUCTION AUDIT
# TEST 386 WORKING BASELINE | ONE CELL | BF16 · SDPA · GREEDY · A100
# 3 FACTS × 3 POSITIONS × 3 LENGTHS × 5 DIMENSIONS | COUNTERFACTUAL × 5 DIMENSIONS
# NO GRADIO · NO SEASC · NO TRAINING · NO LoRA · NO WEIGHT UPDATES
# ==================================================================================================
import os,sys,time,json,random,hashlib,gc,subprocess,importlib.util
from pathlib import Path
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
 if importlib.util.find_spec(m)is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import torch,transformers
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required")
torch.set_grad_enabled(False)
SEED=387;random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEV="cuda";MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";DIMS=[32,64,80,96,128];LENGTHS=[128,2048,10000];POSITIONS=["START","MIDDLE","END"]
MAX_NEW=24;SEP="\n\n";FMT="QUESTION:\n{q}\n\nANSWER:"
ROOT=Path("/content/AKBASCORE_TEST387")if os.path.isdir("/content")else Path("/tmp/AKBASCORE_TEST387")
ROOT.mkdir(parents=True,exist_ok=True)
LOG=ROOT/"TEST387_RESULTS.jsonl";SUMMARY=ROOT/"TEST387_SUMMARY.json";LOG.write_text("",encoding="utf-8")
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
DISTRACTORS=[
 "The municipal records office catalogued an ordinary delivery receipt.",
 "A passenger placed a newspaper on an empty bench.",
 "The workshop supervisor checked the inventory before closing.",
 "Several visitors walked through the courtyard without stopping.",
 "A clerk copied the date into an administrative register.",
 "The gardener watered the trees beside the stone pathway.",
 "A technician inspected the lighting near the entrance.",
 "The archive assistant arranged unlabeled folders on a shelf."]
def sync():torch.cuda.synchronize()
def sec(t):sync();return round(time.perf_counter()-t,4)
def clean():gc.collect();torch.cuda.empty_cache()
def norm(s):return " ".join("".join(c.lower()if c.isalnum()else" "for c in s).split())
def hit(s,a):return int(norm(a)in norm(s))
def gb(x):return round(x/1024**3,5)
def mem():return gb(torch.cuda.memory_allocated())
def peak():sync();return gb(torch.cuda.max_memory_allocated())
def reset_peak():sync();torch.cuda.reset_peak_memory_stats()
def record(x):
 with LOG.open("a",encoding="utf-8")as f:f.write(json.dumps(x,ensure_ascii=False,default=str)+"\n");f.flush()
def fail(s):raise RuntimeError(s)
def safe(fn,label,meta):
 try:return fn()
 except torch.cuda.OutOfMemoryError as e:
  record({"status":"OOM","condition":label,**meta,"error":str(e)[:250]});print(" OOM:",label);clean();return None
 except Exception as e:
  record({"status":"ERROR","condition":label,**meta,"error":f"{type(e).__name__}: {str(e)[:250]}"});print(" ERROR:",label,type(e).__name__,str(e)[:150]);clean();return None
print("="*116,"\nTEST 387 — NIRVANA × MISTRAL — CROSS-POSITION & RECONSTRUCTION AUDIT\n"+"="*116)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/9] MODEL")
t0=time.perf_counter()
tv=tuple(int("".join(c for c in x if c.isdigit())or 0)for x in transformers.__version__.split(".")[:2])
dt="dtype"if tv>=(4,56)else"torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{dt:torch.bfloat16})
model.eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;layers=model.model.layers;NL=len(layers);H=cfg.hidden_size;NH=cfg.num_attention_heads;NKV=cfg.num_key_value_heads
HD=getattr(cfg,"head_dim",None)or H//NH;KVD=NKV*HD
if(NL,H,NH,NKV,HD)!=(32,4096,32,8,128):fail(f"Architecture mismatch: {(NL,H,NH,NKV,HD)}")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
ge=model.generation_config.eos_token_id;EOS=sorted({tok.eos_token_id,*(ge if isinstance(ge,(list,tuple))else[ge])}-{None})
CTX_LIMIT=int(getattr(cfg,"max_position_embeddings",32768))
print(f"Model={MODEL_ID} | layers={NL} | KV={NKV} | head={HD} | context={CTX_LIMIT} | load={sec(t0)}s")
FP=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
def sentinel():
 h=hashlib.sha256()
 for p in FP:
  a=p.detach().reshape(-1);n=a.numel()
  for j in range(16):
   off=j*(n-256)//15;h.update(a[off:off+256].float().cpu().numpy().tobytes())
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
 if c.get_seq_length()!=T:fail("Cache length mismatch")
 return c
@torch.inference_mode()
def forge(s):
 seq=[PAD]+ids(s+SEP)
 if len(seq)>CTX_LIMIT:fail(f"Source exceeds context: {len(seq)}")
 out=model(input_ids=torch.tensor([seq],device=DEV),use_cache=False,output_hidden_states=True,return_dict=True)
 K=[];V=[]
 for L in range(NL):
  z=layers[L].input_layernorm(out.hidden_states[L][0]);a=layers[L].self_attn
  K.append(a.k_proj(z).contiguous());V.append(a.v_proj(z).contiguous())
 return {"T":len(seq),"K":K,"V":V}
@torch.inference_mode()
def install(K,V):
 T=K[0].shape[0];pos=torch.arange(T,device=DEV)[None]
 cos,sin=model.model.rotary_emb(K[0][None],pos);KK=[];VV=[]
 for L in range(NL):
  k=K[L].reshape(1,T,NKV,HD).transpose(1,2).contiguous()
  v=V[L].reshape(1,T,NKV,HD).transpose(1,2).contiguous()
  kr,_=apply_rotary_pos_emb(k,k,cos,sin,unsqueeze_dim=1)
  KK.append(kr.contiguous());VV.append(v)
 return tuple(KK),tuple(VV),T
print("[2/9] FACT-INDEPENDENT PCA CODEBOOK")
t0=time.perf_counter();CO=[forge(s)for s in NEUTRAL]
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
CB_BYTES=sum(t.numel()*t.element_size()for e in CB for n in("K","V")for t in e[n])
SINK_BYTES=sum(t.numel()*t.element_size()for n in("K","V")for t in SINK[n])
print(f"Codebook={CB_BYTES:,} bytes | sink={SINK_BYTES:,} bytes | time={sec(t0)}s")
for d in DIMS:print(f" D{d:03d} | K explained={sum(h[d]for L in EXPL['K']for h in L)/(NL*NKV):.5f} | V explained={sum(h[d]for L in EXPL['V']for h in L)/(NL*NKV):.5f}")
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
@torch.inference_mode()
def geometry(f,decoded):
 result={}
 for n,j in(("K",0),("V",1)):
  a=torch.stack([torch.linalg.vector_norm(f[n][L][1:].float()-decoded[j][L][1:].float())for L in range(NL)])
  b=torch.stack([torch.linalg.vector_norm(f[n][L][1:].float())for L in range(NL)])
  result[n+"_relative_error"]=float(torch.linalg.vector_norm(a)/torch.linalg.vector_norm(b).clamp_min(1e-12))
  result[n+"_max_layer_error"]=float((a/b.clamp_min(1e-12)).max())
 return result
@torch.inference_mode()
def gen(q,kind,obj=None):
 qids=ids(FMT.format(q=q))
 if kind=="visible":pre=[PAD]+ids(obj+SEP);c=None
 elif kind=="memory":pre=[PAD]*obj[2];c=cache_of(obj)
 elif kind=="vanilla":pre=[];c=None
 else:fail("Unknown mode")
 x=torch.tensor([pre+qids],device=DEV)
 if x.shape[1]+MAX_NEW>CTX_LIMIT:fail("Generation exceeds context")
 sync();t=time.perf_counter()
 y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),past_key_values=c,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=PAD,eos_token_id=EOS)
 sync();new=y[0,x.shape[1]:].tolist()
 return {"text":tok.decode(new,skip_special_tokens=True).strip(),"tokens":len(new),"seconds":sec(t)}
def score(q,kind,obj,target,label,meta,forbidden=None):
 reset_peak();r=safe(lambda:gen(q,kind,obj),label,meta)
 if r is None:return None
 r.update({"status":"OK","condition":label,"target":target,"hit":hit(r["text"],target),"peak_allocated_gib":peak(),"forbidden_hit":hit(r["text"],forbidden)if forbidden else None,**meta})
 record(r);print(f" {label:19s} | hit={r['hit']} | {r['seconds']:.3f}s | {r['text'][:115]}")
 return r
print("[3/9] NATIVE PKV AUDIT")
f0=forge(FACTS[0][0])
native=model(input_ids=torch.tensor([[PAD]+ids(FACTS[0][0]+SEP)],device=DEV),use_cache=True).past_key_values
exact0=install(f0["K"],f0["V"])
ke=max(float((exact0[0][L].float()-kvget(native,L)[0].float()).norm()/kvget(native,L)[0].float().norm().clamp_min(1e-12))for L in range(NL))
ve=max(float((exact0[1][L].float()-kvget(native,L)[1].float()).norm()/kvget(native,L)[1].float().norm().clamp_min(1e-12))for L in range(NL))
print(f"K={ke:.9e} | V={ve:.9e}")
if max(ke,ve)>0.05:fail("Native PKV reconstruction audit failed")
del f0,native,exact0;clean()
print("[4/9] TOKEN-ALIGNED CROSS-POSITION SOURCE BUILDER")
FILLER=ids("\n".join(DISTRACTORS)+"\n")
def source_at(target,position,fact):
 fi=ids(fact+"\n")
 if len(fi)>=target:fail("Target length too short for fact")
 left=0 if position=="START"else(target-len(fi))//2 if position=="MIDDLE"else target-len(fi)
 right=target-len(fi)-left
 body=(FILLER*((max(left,right)//len(FILLER))+2))[:left]+fi+(FILLER*((right//len(FILLER))+2))[:right]
 src=tok.decode(body,skip_special_tokens=False)
 actual=ids(src)
 return src,left,len(body),len(actual),int(actual==body)
print("[5/9] SHORT-SOURCE COUNTERFACTUAL × ALL DIMENSIONS")
for i,(src,q,ans,cf,cf_ans)in enumerate(FACTS,1):
 print("-"*116,f"\nCASE {i}/3 | {ans} → {cf_ans}")
 f=forge(src);fc=forge(cf);meta={"group":"SHORT","case":i,"source_tokens":f["T"]}
 score(q,"vanilla",None,ans,"VANILLA",meta)
 score(q,"visible",src,ans,"VISIBLE_SOURCE",meta)
 p=install(f["K"],f["V"]);score(q,"memory",p,ans,"EXACT_PKV",meta);del p
 for d in DIMS:
  for tag,obj,target,forbidden in(("ORIGINAL",f,ans,cf_ans),("COUNTERFACTUAL",fc,cf_ans,ans)):
   code=encode(obj,d);decoded=decode(code,d);geom=geometry(obj,decoded);p=install(*decoded)
   info={**meta,"D":d,"variant":tag,**geom,"compressed_bytes":bytes_code(code),"native_pkv_bytes":bytes_pkv(p)}
   score(q,"memory",p,target,f"{tag}_D{d}",info,forbidden)
   del code,decoded,p;clean()
 del f,fc;clean()
print("[6/9] CROSS-POSITION × LENGTH × DIMENSION")
for target in LENGTHS:
 print("="*116,f"\nTARGET LENGTH={target:,}")
 if target+MAX_NEW+100>CTX_LIMIT:
  record({"status":"SKIP_CONTEXT","target_tokens":target});continue
 for i,(fact,q,ans,cf,cf_ans)in enumerate(FACTS,1):
  for position in POSITIONS:
   src,left,bodylen,retok,aligned=source_at(target,position,fact)
   meta={"group":"LONG","target_tokens":target,"position":position,"case":i,"fact_start_token":left,"body_tokens":bodylen,"retokenized_tokens":retok,"token_alignment":aligned}
   print(f"\n CASE={i} | POSITION={position} | offset={left} | token_alignment={aligned}")
   reset_peak();t=time.perf_counter()
   f=safe(lambda:forge(src),"FORGE",meta)
   if f is None:continue
   meta.update({"source_tokens":f["T"],"forge_seconds":sec(t),"forge_peak_gib":peak()})
   if f["T"]+len(ids(FMT.format(q=q)))+MAX_NEW>CTX_LIMIT:
    record({"status":"SKIP_CONTEXT","condition":"READOUT",**meta});del f;clean();continue
   p=install(f["K"],f["V"]);exact=score(q,"memory",p,ans,"EXACT_PKV",meta);del p;clean()
   if exact is None:
    record({"status":"SKIP_INVALID_BASELINE",**meta});del f;clean();continue
   for d in DIMS:
    t=time.perf_counter();code=safe(lambda:encode(f,d),f"ENCODE_D{d}",meta)
    if code is None:continue
    enc=sec(t);t=time.perf_counter()
    decoded=safe(lambda:decode(code,d),f"DECODE_D{d}",meta)
    if decoded is None:del code;clean();continue
    dec=sec(t);geom=geometry(f,decoded);p=safe(lambda:install(*decoded),f"INSTALL_D{d}",meta)
    if p is None:del code,decoded;clean();continue
    cb=bytes_code(code);nb=bytes_pkv(p)
    info={**meta,"D":d,**geom,"encode_seconds":enc,"decode_seconds":dec,"compressed_bytes":cb,"native_pkv_bytes":nb,
          "coefficient_ratio":round(cb/nb,6),"shared_codebook_bytes":CB_BYTES,"shared_sink_bytes":SINK_BYTES,
          "total_single_memory_bytes":cb+CB_BYTES+SINK_BYTES,"exact_pkv_hit":exact["hit"]}
    score(q,"memory",p,ans,f"D{d}",info)
    del code,decoded,p;clean()
   del f;clean()
 print(" LENGTH COMPLETE:",target,"| saved:",LOG)
print("[7/9] AGGREGATE")
rows=[json.loads(s)for s in LOG.read_text(encoding="utf-8").splitlines()if s.strip()]
ok=[r for r in rows if r.get("status")=="OK"]
for group in("SHORT","LONG"):
 print("\n",group)
 names=(["VANILLA","VISIBLE_SOURCE","EXACT_PKV"]+[f"ORIGINAL_D{d}"for d in DIMS]+[f"COUNTERFACTUAL_D{d}"for d in DIMS])if group=="SHORT"else["EXACT_PKV"]+[f"D{d}"for d in DIMS]
 for name in names:
  a=[r for r in ok if r.get("group")==group and r.get("condition")==name]
  if a:print(f" {name:22s} | {sum(r['hit']for r in a):2d}/{len(a):2d} | mean={sum(r['seconds']for r in a)/len(a):.3f}s")
print("[8/9] CROSS-POSITION LOSSY-WALL")
wall=[]
for target in LENGTHS:
 print(f"\nLENGTH {target:,}")
 for d in DIMS:
  a=[r for r in ok if r.get("group")=="LONG"and r.get("target_tokens")==target and r.get("condition")==f"D{d}"]
  e=[r for r in ok if r.get("group")=="LONG"and r.get("target_tokens")==target and r.get("condition")=="EXACT_PKV"]
  pairs=[(x,y)for x in e for y in a if x["case"]==y["case"]and x["position"]==y["position"]]
  if not pairs:continue
  lost=sum(x["hit"]==1 and y["hit"]==0 for x,y in pairs)
  gained=sum(x["hit"]==0 and y["hit"]==1 for x,y in pairs)
  kerr=sum(y["K_relative_error"]for x,y in pairs)/len(pairs)
  verr=sum(y["V_relative_error"]for x,y in pairs)/len(pairs)
  rec={"length":target,"D":d,"matched":len(pairs),"exact_hits":sum(x["hit"]for x,y in pairs),"compressed_hits":sum(y["hit"]for x,y in pairs),
       "lost":lost,"gained":gained,"mean_K_relative_error":kerr,"mean_V_relative_error":verr}
  wall.append(rec)
  print(f" D{d:03d} | {rec['compressed_hits']}/{len(pairs)} | exact={rec['exact_hits']}/{len(pairs)} | lost={lost} | K_err={kerr:.5f} | V_err={verr:.5f}")
print("[9/9] INTEGRITY + STORAGE")
S1=sentinel()
if S1!=S0:fail("WEIGHT SENTINEL CHANGED")
if model.training or any(p.requires_grad for p in model.parameters()):fail("Frozen model audit failed")
print(f"SHARED CODEBOOK={CB_BYTES:,} bytes | SHARED SINK={SINK_BYTES:,} bytes")
for d in DIMS:print(f"D{d:03d} | theoretical coefficient ratio={d/128:.4f} | coefficient saving={(1-d/128)*100:.2f}%")
report={"test":387,"model":MODEL_ID,"seed":SEED,"dimensions":DIMS,"lengths":LENGTHS,"positions":POSITIONS,
        "native_reconstruction_error":{"K":ke,"V":ve},"codebook_bytes":CB_BYTES,"sink_bytes":SINK_BYTES,
        "sentinel_before":S0,"sentinel_after":S1,"rows_total":len(rows),"rows_ok":len(ok),"lossy_wall":wall,
        "notes":["Model, forge, RoPE installation, PCA basis, decode and generation follow TEST 386.",
                 "Each fact is crossed with all three positions; position is no longer confounded with fact identity.",
                 "Source token alignment is recorded because tokenizer decode/encode may change token boundaries.",
                 "Geometry compares pre-RoPE K and V; first-token shared sink is excluded from geometry.",
                 "D128 preserves full PCA basis dimensionality but BF16 projection/reconstruction is not mathematically lossless.",
                 "Name-substring hit is not semantic equivalence; counterfactual spelling variants may score zero.",
                 "Storage ratios exclude shared codebook and sink; GPU generation uses reconstructed full-size PKV.",
                 "Generated sources contain repetitive neutral distractors; results do not generalize automatically to natural documents."]}
SUMMARY.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(f"WEIGHT SENTINEL: PASS | VALID RESULTS={len(ok)} | RECORDED EVENTS={len(rows)}")
print("RAW LOG :",LOG)
print("SUMMARY :",SUMMARY)
print("="*116)
