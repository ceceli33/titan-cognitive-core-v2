# ==================================================================================================
# TEST 385 — AKBASCORE NIRVANA × MISTRAL — FIRST PKV PROTOTYPE
# Mistral-7B-Instruct-v0.3 | BF16 | SDPA | A100 | D64 | SOURCE-REMOVED RETRIEVAL
# TEST 260 Mistral infrastructure × TEST 384 NIRVANA source-only PKV/PCA architecture
# NO GRADIO | NO SEASC | NO LoRA | NO TRAINING | NO WEIGHT UPDATES | GREEDY
# ==================================================================================================
import os,sys,time,json,math,random,hashlib,traceback,subprocess,importlib.util
from pathlib import Path
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
 if importlib.util.find_spec(m)is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import torch,transformers
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required; A100 recommended.")
torch.set_grad_enabled(False)
SEED=385;random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEV="cuda";MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";D=64;DMAX=128;MAX_NEW=32;SEP="\n\n";FMT="QUESTION:\n{q}\n\nANSWER:"
ROOT=Path("/content/AKBASCORE_TEST385");ROOT.mkdir(parents=True,exist_ok=True)
FACTS=[
 ("Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge.","Who planted the Turkish flag at the base of the Golden Gate Bridge?","Mustafa Akbaş","Leyla Demir planted the Turkish flag at the base of the Golden Gate Bridge.","Leyla Demir"),
 ("Elena Varga stored the silver compass inside the northern archive of Tallinn.","Who stored the silver compass inside the northern archive of Tallinn?","Elena Varga","Tomasz Brenner stored the silver compass inside the northern archive of Tallinn.","Tomasz Brenner"),
 ("Kerem Yıldız repaired the brass lantern at the old lighthouse of Sinop.","Who repaired the brass lantern at the old lighthouse of Sinop?","Kerem Yıldız","Marta Solberg repaired the brass lantern at the old lighthouse of Sinop.","Marta Solberg"),
 ("Aiko Tanabe painted the blue signal beside the eastern platform of Kyoto Station.","Who painted the blue signal beside the eastern platform of Kyoto Station?","Aiko Tanabe","Diego Ferraz painted the blue signal beside the eastern platform of Kyoto Station.","Diego Ferraz")]
CB_CORPUS=[
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
def norm(s):return " ".join("".join(c.lower()if c.isalnum()else " " for c in s).split())
def hit(s,answer):return int(norm(answer)in norm(s))
def stamp(t):return round(time.perf_counter()-t,3)
def fail(s):raise RuntimeError(s)
print("="*108,"\nTEST 385 — NIRVANA × MISTRAL — SOURCE-REMOVED COMPRESSED PKV PROTOTYPE\n"+"="*108)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/8] MODEL LOAD")
t0=time.perf_counter()
tv=tuple(int("".join(c for c in x if c.isdigit())or 0)for x in transformers.__version__.split(".")[:2])
dt="dtype"if tv>=(4,56)else"torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{dt:torch.bfloat16})
model.eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;layers=model.model.layers
NL=len(layers);H=cfg.hidden_size;NH=cfg.num_attention_heads;NKV=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None)or H//NH;KVD=NKV*HD
if (NL,H,NH,NKV,HD)!=(32,4096,32,8,128):fail(f"Mistral architecture mismatch: {(NL,H,NH,NKV,HD)}")
if getattr(cfg,"use_sliding_window",False):fail("Sliding-window attention requires separate validation.")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
ge=model.generation_config.eos_token_id;EOS=sorted({tok.eos_token_id,* (ge if isinstance(ge,(list,tuple))else[ge])}-{None})
print(f"Model={MODEL_ID} | layers={NL} | hidden={H} | Q={NH} | KV={NKV} | head={HD} | load={stamp(t0)}s")
print("[2/8] FROZEN WEIGHT SENTINEL")
FP=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
def sentinel():
 h=hashlib.sha256()
 for i,p in enumerate(FP):
  a=p.detach().reshape(-1);n=a.numel();z=256
  for j in range(16):
   off=j*(n-z)//15;h.update(a[off:off+z].float().cpu().numpy().tobytes())
 return h.hexdigest()
S0=sentinel()
if model.training or any(p.requires_grad for p in model.parameters()):fail("Frozen-model audit failed.")
print("Sampled SHA-256:",S0)
def ids(s):return tok(s,add_special_tokens=False).input_ids
def new_cache():
 try:return DynamicCache(config=cfg)
 except Exception:return DynamicCache()
def kvget(c,L):
 if hasattr(c,"layers"):return c.layers[L].keys,c.layers[L].values
 return c.key_cache[L],c.value_cache[L]
def cache_of(kv):
 c=new_cache()
 for L in range(NL):c.update(kv[0][L].clone(),kv[1][L].clone(),L)
 if c.get_seq_length()!=kv[2]:fail("DynamicCache length mismatch.")
 return c
@torch.inference_mode()
def forge(s):
 if "QUESTION:"in s:fail("Forge must receive source-only text.")
 seq=[PAD]+ids(s+SEP)
 out=model(input_ids=torch.tensor([seq],device=DEV),use_cache=False,output_hidden_states=True,return_dict=True)
 K=[];V=[]
 for L in range(NL):
  z=layers[L].input_layernorm(out.hidden_states[L][0]);a=layers[L].self_attn
  K.append(a.k_proj(z).contiguous());V.append(a.v_proj(z).contiguous())
 return {"ids":seq,"T":len(seq),"K":K,"V":V}
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
 return (tuple(KK),tuple(VV),T)
print("[3/8] FACT-INDEPENDENT PCA CODEBOOK")
t0=time.perf_counter()
CO=[forge(s)for s in CB_CORPUS]
SINK={n:[CO[0][n][L][:1].clone()for L in range(NL)]for n in("K","V")}
CB=[]
for L in range(NL):
 e={}
 for n in("K","V"):
  R=torch.cat([c[n][L][1:]for c in CO]).float().reshape(-1,NKV,HD)
  mu=[];basis=[];ev=[]
  for h in range(NKV):
   X=R[:,h];m=X.mean(0);_,s,vh=torch.linalg.svd(X-m,full_matrices=False)
   b=vh[:DMAX].T.contiguous()
   if b.shape[1]<DMAX:b=F.pad(b,(0,DMAX-b.shape[1]))
   v=s.square();expl=float(v[:D].sum()/v.sum().clamp_min(1e-12))
   mu.append(m);basis.append(b);ev.append(expl)
  e[n]=(torch.stack(mu),torch.stack(basis),sum(ev)/len(ev))
 CB.append(e)
print(f"Codebook rows={sum(c['T']-1 for c in CO)} | K explained={sum(c['K'][2] for c in CB)/NL:.5f} | V explained={sum(c['V'][2] for c in CB)/NL:.5f} | time={stamp(t0)}s")
@torch.inference_mode()
def encode(f,d=D):
 return {n:[torch.einsum("thi,hid->thd",f[n][L][1:].float().reshape(-1,NKV,HD)-CB[L][n][0],CB[L][n][1][:,:,:d]).to(torch.bfloat16)for L in range(NL)]for n in("K","V")}
@torch.inference_mode()
def decode(code,d=D):
 out={}
 for n in("K","V"):
  out[n]=[torch.cat([SINK[n][L],(CB[L][n][0]+torch.einsum("thd,hid->thi",code[n][L].float(),CB[L][n][1][:,:,:d])).reshape(-1,KVD).to(torch.bfloat16)],0)for L in range(NL)]
 return out["K"],out["V"]
print("[4/8] NATIVE PKV RECONSTRUCTION AUDIT")
f=forge(FACTS[0][0])
with torch.inference_mode():
 native=model(input_ids=torch.tensor([f["ids"]],device=DEV),use_cache=True).past_key_values
 rebuilt=install(f["K"],f["V"])
 kerr=max(float((rebuilt[0][L].float()-kvget(native,L)[0].float()).norm()/kvget(native,L)[0].float().norm().clamp_min(1e-12))for L in range(NL))
 verr=max(float((rebuilt[1][L].float()-kvget(native,L)[1].float()).norm()/kvget(native,L)[1].float().norm().clamp_min(1e-12))for L in range(NL))
print(f"Native reconstruction relative error: K={kerr:.6e} | V={verr:.6e}")
if not math.isfinite(kerr+verr):fail("Non-finite native PKV reconstruction error.")
if max(kerr,verr)>0.05:fail("Native PKV reconstruction mismatch; inspect Mistral RoPE/cache API before proceeding.")
del native,rebuilt,f
print("[5/8] SOURCE-REMOVED READOUT")
def gen(q,kind,obj=None,guards=()):
 qids=ids(FMT.format(q=q))
 if kind=="context":
  pre=[PAD]+ids(obj+SEP);c=None;T=0
 elif kind=="memory":
  T=obj[2];pre=[PAD]*T;c=cache_of(obj)
 elif kind=="vanilla":
  pre=[];c=None;T=0
 else:fail(f"Unknown mode: {kind}")
 if kind!="context":
  visible=tok.decode(pre+qids)
  if any(s and s in visible for s in guards):fail("SOURCE TEXT LEAK INTO READOUT.")
  if kind=="memory"and set(pre)!={PAD}:fail("Non-dummy token in memory prefix.")
 x=torch.tensor([pre+qids],device=DEV)
 sync();t=time.perf_counter()
 with torch.inference_mode():
  y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),past_key_values=c,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=PAD,eos_token_id=EOS)
 sync();new=y[0,x.shape[1]:].tolist()
 return {"text":tok.decode(new,skip_special_tokens=True).strip(),"tokens":len(new),"seconds":round(time.perf_counter()-t,3),"memory_slots":T}
print("[6/8] D64 PROTOTYPE — FOUR CONTROL CONDITIONS")
rows=[]
for i,(src,q,answer,cf,cf_answer)in enumerate(FACTS,1):
 print("-"*108);print(f"CASE {i}/4 | SOURCE: {src}")
 t=time.perf_counter();f=forge(src);fc=forge(cf)
 code=encode(f);code_cf=encode(fc)
 pkv=install(*decode(code));pkv_cf=install(*decode(code_cf));pkv_exact=install(f["K"],f["V"])
 guard=(src,cf)
 conditions=[
  ("VANILLA",gen(q,"vanilla",guards=guard),answer),
  ("VISIBLE_SOURCE",gen(q,"context",src),answer),
  ("EXACT_PKV",gen(q,"memory",pkv_exact,guard),answer),
  ("NIRVANA_D64",gen(q,"memory",pkv,guard),answer),
  ("COUNTERFACTUAL_D64",gen(q,"memory",pkv_cf,guard),cf_answer)]
 for name,result,target in conditions:
  result.update({"case":i,"condition":name,"target":target,"hit":hit(result["text"],target),"source_tokens":f["T"]})
  rows.append(result)
  print(f" {name:20s} | hit={result['hit']} | {result['seconds']:7.3f}s | {result['text'][:180]}")
 print(f" Case time={stamp(t)}s | PKV slots={f['T']} | compressed coefficients={NL*2*NKV*(f['T']-1)*D:,}")
 del f,fc,code,code_cf,pkv,pkv_cf,pkv_exact
 torch.cuda.empty_cache()
print("[7/8] SUMMARY")
for name in("VANILLA","VISIBLE_SOURCE","EXACT_PKV","NIRVANA_D64","COUNTERFACTUAL_D64"):
 a=[r for r in rows if r["condition"]==name]
 print(f"{name:20s}: {sum(r['hit'] for r in a)}/{len(a)} | mean_generation={sum(r['seconds'] for r in a)/len(a):.3f}s")
print("[8/8] FINAL AUDIT + LOG")
S1=sentinel()
if S1!=S0:fail("WEIGHT SENTINEL CHANGED.")
if any(p.requires_grad for p in model.parameters())or model.training:fail("Model is no longer frozen.")
if any(len(l._forward_hooks)for l in layers):print("NOTICE: decoder forward hooks exist; inspect framework ownership.")
report={"test":385,"model":MODEL_ID,"seed":SEED,"dtype":"bfloat16","attention":"sdpa","compression_dimension":D,"layers":NL,"kv_heads":NKV,"head_dim":HD,"native_pkv_relative_error":{"K":kerr,"V":verr},"sampled_weight_sentinel_before":S0,"sampled_weight_sentinel_after":S1,"rows":rows,"summary":{n:{"hits":sum(r["hit"]for r in rows if r["condition"]==n),"total":sum(r["condition"]==n for r in rows)}for n in("VANILLA","VISIBLE_SOURCE","EXACT_PKV","NIRVANA_D64","COUNTERFACTUAL_D64")},"note":"Short-source engineering prototype, not long-context validation. Compression coefficient counts exclude PCA codebook, sink slots and reconstructed full-size PKV."}
path=ROOT/"TEST385_MISTRAL_NIRVANA_PROTOTYPE.json"
path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print("WEIGHT SENTINEL: PASS | TRAINING: OFF | SOURCE-REMOVED PKV PATH: EXECUTED")
print("LOG:",path)
print("="*108)
