# ==================================================================================================
# TEST 388 — AKBASCORE NIRVANA · MISTRAL → QWEN · CROSS-MODEL PKV BRIDGE
# WORKING REFERENCES: MISTRAL TEST 387 + QWEN NIRVANA TEST 384
# ONE COLAB CELL · A100 · BF16 MODELS · FP32 D128 BRIDGE · SDPA · GREEDY
# SOURCE TEXT NEVER ENTERS QWEN BRIDGE READOUT · GOLDEN GATE HELD OUT
# NO LoRA · NO MODEL WEIGHT UPDATES · NO FORWARD HOOKS
# ==================================================================================================
import os,sys,gc,time,json,random,hashlib,subprocess,importlib.util
from pathlib import Path
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
 if importlib.util.find_spec(m)is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import torch,transformers
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb as mistral_rope
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as qwen_rope
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required")
os.environ["TOKENIZERS_PARALLELISM"]="false"
torch.set_grad_enabled(False)
SEED=388;random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEV="cuda";DTYPE=torch.bfloat16;D=128;MAX_NEW=32;SEP="\n\n";FMT="QUESTION:\n{q}\n\nANSWER:"
M_ID="mistralai/Mistral-7B-Instruct-v0.3";Q_ID="Qwen/Qwen2.5-7B-Instruct"
ROOT=Path("/content/AKBASCORE_TEST388")if os.path.isdir("/content")else Path("/tmp/AKBASCORE_TEST388")
ROOT.mkdir(parents=True,exist_ok=True)
LOG=ROOT/"TEST388_RESULTS.jsonl";SUMMARY=ROOT/"TEST388_SUMMARY.json";LOG.write_text("",encoding="utf-8")
FACT="Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge."
CF="Leyla Demir planted the Turkish flag at the base of the Golden Gate Bridge."
WRONG="Elena Varga stored the silver compass inside the northern archive of Tallinn."
QUESTIONS=[
 ("WHO","Who planted the Turkish flag at the base of the Golden Gate Bridge?","Mustafa Akbaş","Leyla Demir"),
 ("WHAT","What did Mustafa Akbaş plant at the base of the Golden Gate Bridge?","Turkish flag","Turkish flag"),
 ("WHERE","Where did Mustafa Akbaş plant the Turkish flag?","base of the Golden Gate Bridge","base of the Golden Gate Bridge")]
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
TRAIN=NEUTRAL[:24];VAL=NEUTRAL[24:]
assert not any(x in s or s in x for s in NEUTRAL for x in (FACT,CF,WRONG))
def sync():torch.cuda.synchronize()
def clean():gc.collect();torch.cuda.empty_cache()
def norm(s):return " ".join("".join(c.casefold()if c.isalnum()else" "for c in str(s)).split())
def hit(s,a):return int(norm(a)in norm(s))
def rec(x):
 with LOG.open("a",encoding="utf-8")as f:f.write(json.dumps(x,ensure_ascii=False,default=str)+"\n");f.flush()
def sha(t):
 h=hashlib.sha256()
 for p in t:
  a=p.detach().reshape(-1);n=a.numel()
  for j in range(16):
   off=j*max(0,n-256)//15;h.update(a[off:off+256].float().cpu().numpy().tobytes())
 return h.hexdigest()
def load(mid,arch):
 print("LOAD:",mid);t=time.perf_counter()
 tv=tuple(int("".join(c for c in x if c.isdigit())or 0)for x in transformers.__version__.split(".")[:2])
 kw={"dtype"if tv>=(4,56)else"torch_dtype":DTYPE}
 tokenizer=AutoTokenizer.from_pretrained(mid,use_fast=True)
 mdl=AutoModelForCausalLM.from_pretrained(mid,device_map={"":0},attn_implementation="sdpa",**kw)
 mdl.eval()
 for p in mdl.parameters():p.requires_grad_(False)
 c=mdl.config;l=mdl.model.layers
 dims=(len(l),c.hidden_size,c.num_attention_heads,c.num_key_value_heads,getattr(c,"head_dim",None)or c.hidden_size//c.num_attention_heads)
 if dims!=arch:raise RuntimeError(f"Architecture mismatch: {dims} != {arch}")
 if next(mdl.parameters()).dtype!=DTYPE:raise RuntimeError("Unexpected model dtype")
 print(" READY:",dims,"|",round(time.perf_counter()-t,2),"s")
 return tokenizer,mdl,l,dims
def ids(t,s):return t(s,add_special_tokens=False).input_ids
def pad(t):return t.pad_token_id if t.pad_token_id is not None else t.eos_token_id
@torch.inference_mode()
def forge(t,m,l,s):
 seq=[pad(t)]+ids(t,s+SEP)
 out=m(input_ids=torch.tensor([seq],device=DEV),use_cache=False,output_hidden_states=True,return_dict=True)
 K=[];V=[]
 for i,layer in enumerate(l):
  z=layer.input_layernorm(out.hidden_states[i][0]);a=layer.self_attn
  K.append(a.k_proj(z).float().cpu().contiguous())
  V.append(a.v_proj(z).float().cpu().contiguous())
 del out
 return {"T":len(seq),"K":K,"V":V}
def source_layer(q):return min(31,max(0,round((q+.5)*32/28-.5)))
LMAP=[source_layer(i)for i in range(28)]
def resize(x,n):
 if x.shape[0]==n:return x.contiguous()
 return F.interpolate(x.T.unsqueeze(0),size=n,mode="linear",align_corners=False)[0].T.contiguous()
def source_head(x,n):
 a=x[1:].reshape(-1,8,128).reshape(-1,4,2,128).mean(2)
 return resize(a.reshape(-1,512),n).reshape(n,4,128)
def source_for_target(f,L,n,kind):
 return source_head(f[kind][LMAP[L]],n)
def target_heads(f,L,kind):return f[kind][L][1:].reshape(-1,4,128)
def init_stats():
 return {kind:[[{k:torch.zeros(shape,dtype=torch.float64)for k,shape in
  (("sx",(128,)),("sy",(128,)),("xx",(128,128)),("xy",(128,128)),("yy",()))}|{"n":0}
  for h in range(4)]for L in range(28)]for kind in("K","V")}
ST=init_stats()
def accumulate(src,tgt):
 n=tgt["T"]-1
 for kind in("K","V"):
  for L in range(28):
   X=source_for_target(src,L,n,kind).double()
   Y=target_heads(tgt,L,kind).double()
   for h in range(4):
    a=X[:,h,:];b=Y[:,h,:];s=ST[kind][L][h]
    s["sx"]+=a.sum(0);s["sy"]+=b.sum(0)
    s["xx"]+=a.T@a;s["xy"]+=a.T@b;s["yy"]+=(b*b).sum();s["n"]+=n
def solve_bridge():
 B={k:[]for k in("K","V")};fit=[]
 for kind in("K","V"):
  for L in range(28):
   group=[]
   for h in range(4):
    s=ST[kind][L][h];n=s["n"];mx=s["sx"]/n;my=s["sy"]/n
    xx=s["xx"]-n*torch.outer(mx,mx);xy=s["xy"]-n*torch.outer(mx,my)
    alpha=max(1e-8,float(xx.trace()/128)*.05)
    W=torch.linalg.solve(xx+alpha*torch.eye(128,dtype=torch.float64),xy)
    err=float((s["yy"]-n*(my@my)-2*(W*xy).sum()+(W*(xx@W)).sum()).clamp_min(0))
    var=float((s["yy"]-n*(my@my)).clamp_min(1e-12))
    fit.append({"kind":kind,"layer":L,"head":h,"train_R2":1-err/var,"ridge_alpha":alpha,"rows":n})
    group.append((W.float(),mx.float(),my.float()))
   B[kind].append(group)
 return B,fit
def bridge_pre(src,B,ratio,sink):
 n=max(1,round((src["T"]-1)*ratio));out={}
 for kind in("K","V"):
  arr=[]
  for L in range(28):
   X=source_for_target(src,L,n,kind);Y=[]
   for h in range(4):
    W,mx,my=B[kind][L][h]
    Y.append((X[:,h,:]-mx)@W+my)
   body=torch.stack(Y,1).reshape(n,512)
   arr.append(torch.cat([sink[kind][L],body],0))
  out[kind]=arr
 return out["K"],out["V"],n+1
def coserr(pred,real):
 a=torch.cat([x[1:].reshape(-1)for x in pred]).float()
 b=torch.cat([x[1:].reshape(-1)for x in real]).float()
 return {"cos":float(F.cosine_similarity(a,b,dim=0)),"relative_error":float((a-b).norm()/b.norm().clamp_min(1e-12))}
def audit(pred,tgt):
 return {k:coserr(pred[j],tgt[k])for j,k in enumerate(("K","V"))}
def new_cache(m):
 try:return DynamicCache(config=m.config)
 except Exception:return DynamicCache()
def kvget(c,L):
 if hasattr(c,"layers"):return c.layers[L].keys,c.layers[L].values
 return c.key_cache[L],c.value_cache[L]
@torch.inference_mode()
def install(m,K,V):
 T=K[0].shape[0];pos=torch.arange(T,device=DEV)[None]
 cos,sin=m.model.rotary_emb(K[0][None].to(DEV,dtype=DTYPE),pos)
 kk=[];vv=[]
 for L in range(28):
  k=K[L].to(DEV,dtype=DTYPE).reshape(1,T,4,128).transpose(1,2)
  v=V[L].to(DEV,dtype=DTYPE).reshape(1,T,4,128).transpose(1,2)
  kr,_=qwen_rope(k,k,cos,sin,unsqueeze_dim=1)
  kk.append(kr.contiguous());vv.append(v.contiguous())
 return tuple(kk),tuple(vv),T
@torch.inference_mode()
def gen(t,m,q,mode,obj=None):
 qi=ids(t,FMT.format(q=q))
 if mode=="VANILLA":pre=[];cache=None
 elif mode=="VISIBLE":pre=[pad(t)]+ids(t,obj+SEP);cache=None
 else:
  pre=[pad(t)]*obj[2];cache=new_cache(m)
  for L in range(28):cache.update(obj[0][L].clone(),obj[1][L].clone(),L)
  if cache.get_seq_length()!=obj[2]:raise RuntimeError("Cache length mismatch")
 x=torch.tensor([pre+qi],device=DEV)
 ge=m.generation_config.eos_token_id
 eos=sorted({t.eos_token_id,*(ge if isinstance(ge,(list,tuple))else[ge])}-{None})
 y=m.generate(input_ids=x,attention_mask=torch.ones_like(x),past_key_values=cache,
  max_new_tokens=MAX_NEW,do_sample=False,repetition_penalty=1.0,use_cache=True,pad_token_id=pad(t),eos_token_id=eos)
 return t.decode(y[0,x.shape[1]:],skip_special_tokens=True).strip()
def first_answer(s):return s.split("\n",1)[0].strip()
print("="*110,"\nTEST 388 — MISTRAL → QWEN · FP32 D128 CROSS-MODEL PKV BRIDGE\n"+"="*110)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/8] MISTRAL — SOURCE-ONLY EXTRACTION")
mt,mm,ml,_=load(M_ID,(32,4096,32,8,128))
mfp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight]
M0=sha(mfp)
MS=[];MF={}
for i,s in enumerate(NEUTRAL):
 f=forge(mt,mm,ml,s);MS.append(f);print(f" MISTRAL CAL {i+1:02d}/32 | tokens={f['T']}",end="\r")
for name,s in (("ORIGINAL",FACT),("COUNTERFACTUAL",CF),("WRONG",WRONG)):
 MF[name]=forge(mt,mm,ml,s);print("\n MISTRAL FINAL",name,"| tokens",MF[name]["T"])
if sha(mfp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mm,ml,mt,mfp;clean()
print("\n[2/8] QWEN — TARGET CALIBRATION")
qt,qm,ql,_=load(Q_ID,(28,3584,28,4,128))
qfp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight]
Q0=sha(qfp);sink=None;ratios=[];train_meta=[]
for i,s in enumerate(NEUTRAL):
 qf=forge(qt,qm,ql,s)
 if sink is None:sink={k:[qf[k][L][:1].clone()for L in range(28)]for k in("K","V")}
 ratio=(qf["T"]-1)/(MS[i]["T"]-1);ratios.append(ratio)
 if i<24:accumulate(MS[i],qf)
 train_meta.append({"index":i,"split":"TRAIN"if i<24 else"VAL","mistral_tokens":MS[i]["T"],"qwen_tokens":qf["T"],"ratio":ratio})
 del qf
 print(f" QWEN CAL {i+1:02d}/32 | split={'TRAIN'if i<24 else'VAL'} | ratio={ratio:.4f}",end="\r")
clean()
RATIO=float(torch.tensor(ratios[:24]).median())
print("\n TOKEN RATIO (TRAIN ONLY):",round(RATIO,6))
print("[3/8] FP32 D128 RIDGE BRIDGE — 28 LAYERS × 4 KV HEADS × K/V")
t0=time.perf_counter();B,FIT=solve_bridge()
print(" BRIDGE FIT:",round(time.perf_counter()-t0,2),"s | mappings:",len(FIT))
print(" TRAIN R²:",round(sum(x["train_R2"]for x in FIT)/len(FIT),6))
print("[4/8] HELD-OUT NEUTRAL GEOMETRY — 8 TEXTS")
VAL_ROWS=[]
for i in range(24,32):
 qf=forge(qt,qm,ql,NEUTRAL[i]);p=bridge_pre(MS[i],B,RATIO,sink)
 if p[2]==qf["T"]:
  g=audit(p[:2],qf);r={"index":i,"status":"ALIGNED","geometry":g}
 else:
  r={"index":i,"status":"LENGTH_MISMATCH","predicted_tokens":p[2],"actual_tokens":qf["T"]}
 VAL_ROWS.append(r);rec({"group":"VAL","source_index":i,**r})
 print(" VAL",i+1,r)
 del qf,p
clean()
print("[5/8] GOLDEN GATE — QWEN NATIVE EXACT PKV REFERENCE")
qfact=forge(qt,qm,ql,FACT);qcf=forge(qt,qm,ql,CF)
NATIVE=install(qm,qfact["K"],qfact["V"])
native_forward=qm(input_ids=torch.tensor([[pad(qt)]+ids(qt,FACT+SEP)],device=DEV),use_cache=True).past_key_values
NK=max(float((NATIVE[0][L].float()-kvget(native_forward,L)[0].float()).norm()/kvget(native_forward,L)[0].float().norm().clamp_min(1e-12))for L in range(28))
NV=max(float((NATIVE[1][L].float()-kvget(native_forward,L)[1].float()).norm()/kvget(native_forward,L)[1].float().norm().clamp_min(1e-12))for L in range(28))
print(" NATIVE INSTALL AUDIT | K:",NK,"V:",NV)
if max(NK,NV)>.05:raise RuntimeError("Qwen native PKV audit failed")
del native_forward;clean()
print("[6/8] BRIDGE MEMORY INSTALL — QWEN HAS NOT SEEN MISTRAL FINAL SOURCE IN BRIDGE PATH")
P={}
for name,f in MF.items():
 k,v,n=bridge_pre(f,B,RATIO,sink)
 P[name]=install(qm,k,v)
 print(" BRIDGE",name,"| source slots",f["T"],"| target slots",n)
 del k,v
GEO={}
for name,qf in (("ORIGINAL",qfact),("COUNTERFACTUAL",qcf)):
 p=P[name]
 if p[2]==qf["T"]:
  # Geometry is measured on pre-RoPE K/V; rebuild the bridge output for this diagnostic.
  raw=bridge_pre(MF[name],B,RATIO,sink)
  GEO[name]=audit(raw[:2],qf)
  del raw
 else:GEO[name]={"status":"LENGTH_MISMATCH","predicted":p[2],"actual":qf["T"]}
 print(" GEOMETRY",name,GEO[name])
print("[7/8] BEHAVIORAL READOUT — SOURCE-REMOVED / GREEDY")
ROWS=[]
for tag,q,ans,cf_ans in QUESTIONS:
 print("\nQUESTION:",tag,q)
 for arm,obj,expected in [
  ("VANILLA",None,ans),("NATIVE_QWEN_PKV",NATIVE,ans),
  ("BRIDGE_MISTRAL",P["ORIGINAL"],ans),("BRIDGE_WRONG",P["WRONG"],ans),
  ("BRIDGE_COUNTERFACTUAL",P["COUNTERFACTUAL"],cf_ans)]:
  mode="VANILLA"if arm=="VANILLA"else"MEMORY"
  t0=time.perf_counter();output=gen(qt,qm,q,mode,obj);sync()
  first=first_answer(output)
  r={"group":"FINAL","question_type":tag,"question":q,"arm":arm,"expected":expected,
     "first_answer":first,"full_output":output,"hit_first":hit(first,expected),
     "hit_anywhere":hit(output,expected),"seconds":round(time.perf_counter()-t0,4),
     "memory_slots":obj[2]if obj is not None else 0}
  ROWS.append(r);rec(r)
  print(f" {arm:24s} | first_hit={r['hit_first']} | {first[:135]}")
clean()
print("[8/8] INTEGRITY + SUMMARY")
Q1=sha(qfp)
if Q1!=Q0:raise RuntimeError("Qwen weight sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Frozen model audit failed")
SCORES={a:{"first_hits":sum(r["hit_first"]for r in ROWS if r["arm"]==a),
            "total":sum(r["arm"]==a for r in ROWS)}for a in sorted({r["arm"]for r in ROWS})}
REPORT={"test":388,"architecture":"Mistral pre-RoPE K/V → paired-head mean → position interpolation → per-layer/head FP32 ridge → Qwen pre-RoPE K/V → Qwen RoPE → DynamicCache",
 "models":{"source":M_ID,"target":Q_ID},"seed":SEED,"D":D,"training_texts":len(TRAIN),"validation_texts":len(VAL),
 "final_source":FACT,"counterfactual_source":CF,"wrong_source":WRONG,"source_layer_map":LMAP,
 "source_to_target_token_ratio":RATIO,"token_ratio_calibration":train_meta,
 "ridge_train_fit":FIT,"validation":VAL_ROWS,"final_geometry":GEO,
 "native_qwen_install_error":{"K":NK,"V":NV},"scores":SCORES,"answers":ROWS,
 "integrity":{"mistral_sentinel":M0,"qwen_before":Q0,"qwen_after":Q1,"frozen_models":True},
 "limitations":[
  "D128 is full per-head dimensionality, not a guarantee of lossless cross-model translation.",
  "Mistral 8 KV heads are paired and averaged into 4 heads; this is a lossy architectural assumption.",
  "Source token positions are linearly interpolated to an estimated target length; final Qwen tokenization is not used by the bridge.",
  "The target native PKV arm is a diagnostic ceiling; Qwen sees the final source only in this separately labeled reference arm.",
  "Bridge training uses only 24 neutral calibration texts; final Golden Gate facts and counterfactual are held out.",
  "No semantic or functional equivalence is inferred from geometric cosine alone.",
  "Qwen source-removed readout uses PAD dummy prefix and installed DynamicCache; generated answers are scored on their first line.",
  "A failed bridge does not establish that cross-model latent transfer is impossible.",
  "Sentinels sample selected model tensors rather than cryptographically hashing all parameters."]}
SUMMARY.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("\nSCORES:")
for k,v in SCORES.items():print(f" {k:24s} {v['first_hits']}/{v['total']}")
print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS | NATIVE PKV AUDIT: PASS")
print("JSONL:",LOG)
print("SUMMARY:",SUMMARY)
print("="*110)
