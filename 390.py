# ==================================================================================================
# TEST 390 — AKBASCORE NIRVANA · DUAL-DNA DIFFERENTIAL RADAR
# TEST 389 BASELINE · MISTRAL→QWEN · NATURAL DNA × FP32 D128 BRIDGE × FUNCTIONAL X-RAY
# ONE CELL · A100 · BF16 · SDPA · GREEDY · NO LoRA · NO WEIGHT UPDATES · NO HOOKS
# ==================================================================================================
import os,sys,gc,time,json,random,hashlib,subprocess,importlib.util
from pathlib import Path
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
 if importlib.util.find_spec(m)is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import torch,transformers
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as qwen_rope
if not torch.cuda.is_available():raise RuntimeError("CUDA required")
os.environ["TOKENIZERS_PARALLELISM"]="false";torch.set_grad_enabled(False)
SEED=390;random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEV="cuda";DTYPE=torch.bfloat16;MAX_NEW=32;SEP="\n\n";FMT="QUESTION:\n{q}\n\nANSWER:"
MID="mistralai/Mistral-7B-Instruct-v0.3";QID="Qwen/Qwen2.5-7B-Instruct"
ROOT=Path("/content/AKBASCORE_TEST390")if os.path.isdir("/content")else Path("/tmp/AKBASCORE_TEST390")
ROOT.mkdir(parents=True,exist_ok=True);LOG=ROOT/"TEST390.jsonl";SUMMARY=ROOT/"TEST390_SUMMARY.json";LOG.write_text("",encoding="utf-8")
FACT="Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge."
CF="Leyla Demir planted the Turkish flag at the base of the Golden Gate Bridge."
WRONG="Elena Varga stored the silver compass inside the northern archive of Tallinn."
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
QUESTIONS=[
 ("WHO","Who planted the Turkish flag at the base of the Golden Gate Bridge?","Mustafa Akbaş","Leyla Demir"),
 ("WHAT","What did Mustafa Akbaş plant at the base of the Golden Gate Bridge?","Turkish flag",None),
 ("WHERE","Where did Mustafa Akbaş plant the Turkish flag?","base of the Golden Gate Bridge",None)]
assert len(NEUTRAL)==32 and all(FACT not in s and CF not in s and WRONG not in s for s in NEUTRAL)
def rec(x):
 with LOG.open("a",encoding="utf-8")as f:f.write(json.dumps(x,ensure_ascii=False,default=str)+"\n")
def clean():gc.collect();torch.cuda.empty_cache()
def norm(s):return " ".join("".join(c.casefold()if c.isalnum()else" "for c in str(s)).split())
def hit(s,a):return int(bool(a)and norm(a)in norm(s))
def sha(ps):
 h=hashlib.sha256()
 for p in ps:
  a=p.detach().reshape(-1);n=a.numel()
  for j in range(16):
   off=j*max(0,n-256)//15;h.update(a[off:off+256].float().cpu().numpy().tobytes())
 return h.hexdigest()
def load(mid,expected):
 print("LOAD:",mid);t=time.time()
 tv=tuple(int("".join(c for c in x if c.isdigit())or 0)for x in transformers.__version__.split(".")[:2])
 kw={"dtype"if tv>=(4,56)else"torch_dtype":DTYPE}
 tok=AutoTokenizer.from_pretrained(mid,use_fast=True)
 model=AutoModelForCausalLM.from_pretrained(mid,device_map={"":0},attn_implementation="sdpa",**kw)
 model.eval()
 for p in model.parameters():p.requires_grad_(False)
 c=model.config;layers=model.model.layers
 dims=(len(layers),c.hidden_size,c.num_attention_heads,c.num_key_value_heads,getattr(c,"head_dim",None)or c.hidden_size//c.num_attention_heads)
 if dims!=expected:raise RuntimeError(f"Architecture mismatch {dims}")
 print("READY:",dims,"|",round(time.time()-t,2),"s")
 return tok,model,layers
def ids(tok,s):return tok(s,add_special_tokens=False).input_ids
def pad(tok):return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
@torch.inference_mode()
def forge(tok,model,layers,s):
 seq=[pad(tok)]+ids(tok,s+SEP)
 out=model(input_ids=torch.tensor([seq],device=DEV),use_cache=False,output_hidden_states=True,return_dict=True)
 K=[];V=[]
 for L,layer in enumerate(layers):
  z=layer.input_layernorm(out.hidden_states[L][0]);a=layer.self_attn
  K.append(a.k_proj(z).float().cpu().contiguous());V.append(a.v_proj(z).float().cpu().contiguous())
 del out
 return {"T":len(seq),"K":K,"V":V}
LMAP=[min(31,max(0,round((L+.5)*32/28-.5)))for L in range(28)]
def resize(x,n):
 if x.shape[0]==n:return x.contiguous()
 return F.interpolate(x.T.unsqueeze(0),size=n,mode="linear",align_corners=False)[0].T.contiguous()
def src_heads(f,L,n,kind):
 x=f[kind][LMAP[L]][1:].reshape(-1,8,128).reshape(-1,4,2,128).mean(2)
 return resize(x.reshape(-1,512),n).reshape(n,4,128)
def tgt_heads(f,L,kind):return f[kind][L][1:].reshape(-1,4,128)
def stats():
 return {k:[[{n:torch.zeros(sh,dtype=torch.float64)for n,sh in
  (("sx",(128,)),("sy",(128,)),("xx",(128,128)),("xy",(128,128)),("yy",()))}|{"n":0}
  for h in range(4)]for L in range(28)]for k in("K","V")}
ST=stats()
def accumulate(m,q):
 n=q["T"]-1
 for kind in("K","V"):
  for L in range(28):
   X=src_heads(m,L,n,kind).double();Y=tgt_heads(q,L,kind).double()
   for h in range(4):
    a=X[:,h];b=Y[:,h];s=ST[kind][L][h]
    s["sx"]+=a.sum(0);s["sy"]+=b.sum(0);s["xx"]+=a.T@a;s["xy"]+=a.T@b;s["yy"]+=(b*b).sum();s["n"]+=n
def solve():
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
    fit.append({"kind":kind,"layer":L,"head":h,"R2_train":1-err/var,"alpha":alpha})
    group.append((W.float(),mx.float(),my.float()))
   B[kind].append(group)
 return B,fit
def bridge(m,B,n,sink):
 out={}
 for kind in("K","V"):
  arr=[]
  for L in range(28):
   X=src_heads(m,L,n,kind);Y=[]
   for h in range(4):
    W,mx,my=B[kind][L][h];Y.append((X[:,h]-mx)@W+my)
   arr.append(torch.cat([sink[kind][L],torch.stack(Y,1).reshape(n,512)],0))
  out[kind]=arr
 return out["K"],out["V"],n+1
def cos(a,b):
 a=a.float().reshape(-1);b=b.float().reshape(-1)
 return float(F.cosine_similarity(a,b,dim=0,eps=1e-12))
def rel(a,b):return float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))
def vecstats(a,b):
 return {"cos":cos(a,b),"rel":rel(a,b),"norm_ratio":float(a.float().norm()/b.float().norm().clamp_min(1e-12))}
def softstats(a,b):
 p=a.float().softmax(-1);q=b.float().softmax(-1)
 return {"TV":float((p-q).abs().sum(-1).mean()/2),
         "top1_agreement":float((p.argmax(-1)==q.argmax(-1)).float().mean()),
         "KL_native_to_bridge":float((q*(q.clamp_min(1e-12).log()-p.clamp_min(1e-12).log())).sum(-1).mean())}
def centered(a,b):
 a=a.float();b=b.float()
 return cos(a-a.mean(0,keepdim=True),b-b.mean(0,keepdim=True))
def dna(p,q):
 n=q["T"]-1
 if p[2]!=q["T"]:raise RuntimeError("DNA token length mismatch")
 rows=[]
 for L in range(28):
  for kind in("K","V"):
   x=p[0 if kind=="K"else 1][L][1:].reshape(n,4,128)
   y=q[kind][L][1:].reshape(n,4,128)
   for h in range(4):
    a=x[:,h];b=y[:,h];z=vecstats(a,b)
    z.update({"layer":L,"head":h,"kind":kind,"centered_cos":centered(a,b),
              "delta_norm":float((b-a).norm()),"native_norm":float(b.norm()),
              "token_cos":[cos(a[t],b[t])for t in range(n)]})
    rows.append(z)
 return rows
def new_cache(m):
 try:return DynamicCache(config=m.config)
 except Exception:return DynamicCache()
def kvget(c,L):
 if hasattr(c,"layers"):return c.layers[L].keys,c.layers[L].values
 return c.key_cache[L],c.value_cache[L]
@torch.inference_mode()
def install(model,K,V):
 T=K[0].shape[0]
 if any(x.shape!=(T,512)for x in K+V):raise RuntimeError("PKV shape mismatch")
 pos=torch.arange(T,device=DEV)[None]
 cosr,sinr=model.model.rotary_emb(K[0][None].to(DEV,dtype=DTYPE),pos)
 KK=[];VV=[]
 for L in range(28):
  k=K[L].to(DEV,dtype=DTYPE).reshape(1,T,4,128).transpose(1,2)
  v=V[L].to(DEV,dtype=DTYPE).reshape(1,T,4,128).transpose(1,2)
  kr,_=qwen_rope(k,k,cosr,sinr,unsqueeze_dim=1)
  KK.append(kr.contiguous());VV.append(v.contiguous())
 return tuple(KK),tuple(VV),T
@torch.inference_mode()
def gen(tok,model,q,pkv=None):
 qi=ids(tok,FMT.format(q=q));pre=[]if pkv is None else[pad(tok)]*pkv[2]
 cache=None
 if pkv is not None:
  cache=new_cache(model)
  for L in range(28):cache.update(pkv[0][L].clone(),pkv[1][L].clone(),L)
  if cache.get_seq_length()!=pkv[2]:raise RuntimeError("Cache length mismatch")
 x=torch.tensor([pre+qi],device=DEV)
 ge=model.generation_config.eos_token_id
 eos=sorted({tok.eos_token_id,*(ge if isinstance(ge,(list,tuple))else[ge])}-{None})
 y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),past_key_values=cache,
  max_new_tokens=MAX_NEW,do_sample=False,repetition_penalty=1.0,use_cache=True,pad_token_id=pad(tok),eos_token_id=eos)
 return tok.decode(y[0,x.shape[1]:],skip_special_tokens=True).strip()
@torch.inference_mode()
def attention_radar(tok,model,q,pred,native):
 # Actual Qwen question hidden states with natural source prefix; readout is DIAGNOSTIC ONLY.
 seq=[pad(tok)]+ids(tok,FACT+SEP)+ids(tok,FMT.format(q=q))
 out=model(input_ids=torch.tensor([seq],device=DEV),output_hidden_states=True,use_cache=False,return_dict=True)
 T=native[2];rows=[]
 for L,layer in enumerate(model.model.layers):
  a=layer.self_attn;z=layer.input_layernorm(out.hidden_states[L][0,T:])
  Q=a.q_proj(z).float().reshape(-1,28,128).transpose(0,1).contiguous()
  qp=torch.arange(T,T+Q.shape[1],device=DEV)[None]
  cq,sq=model.model.rotary_emb(Q[None].to(DTYPE),qp)
  qr,_=qwen_rope(Q[None].to(DTYPE),Q[None].to(DTYPE),cq,sq,unsqueeze_dim=1)
  qr=qr[0].float()
  kn=native[0][L][0].float();kb=pred[0][L][0].float()
  kn=kn.repeat_interleave(7,dim=0);kb=kb.repeat_interleave(7,dim=0)
  an=(qr@kn.transpose(-1,-2))/128**.5
  ab=(qr@kb.transpose(-1,-2))/128**.5
  d=softstats(ab,an)
  d.update({"layer":L,"query_tokens":Q.shape[1],"memory_tokens":T})
  rows.append(d)
 del out
 return rows
def aggregate(rows,keys):
 return {k:sum(r[k]for r in rows)/len(rows)for k in keys}
print("="*110,"\nTEST 390 — MISTRAL ↔ QWEN · DUAL-DNA DIFFERENTIAL RADAR\n"+"="*110)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/10] MISTRAL NATURAL DNA")
mt,mm,ml=load(MID,(32,4096,32,8,128))
mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight]
M0=sha(mp);MS=[];MF={}
for i,s in enumerate(NEUTRAL):
 MS.append(forge(mt,mm,ml,s));print(f" MISTRAL CAL {i+1}/32",end="\r")
for name,s in (("ORIGINAL",FACT),("COUNTERFACTUAL",CF),("WRONG",WRONG)):
 MF[name]=forge(mt,mm,ml,s);print("\n MISTRAL DNA",name,"T=",MF[name]["T"])
if sha(mp)!=M0:raise RuntimeError("Mistral weight sentinel changed")
del mt,mm,ml,mp;clean()
print("\n[2/10] QWEN NATURAL DNA — SAME TEXTS")
qt,qm,ql=load(QID,(28,3584,28,4,128))
qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight]
Q0=sha(qp);sink=None;ratios=[];QVAL={}
for i,s in enumerate(NEUTRAL):
 qf=forge(qt,qm,ql,s)
 if sink is None:sink={k:[qf[k][L][:1].clone()for L in range(28)]for k in("K","V")}
 ratios.append((qf["T"]-1)/(MS[i]["T"]-1))
 if i<24:accumulate(MS[i],qf)
 else:QVAL[i]=qf
 print(f" QWEN CAL {i+1}/32",end="\r")
QFINAL={name:forge(qt,qm,ql,s)for name,s in (("ORIGINAL",FACT),("COUNTERFACTUAL",CF),("WRONG",WRONG))}
RATIO=float(torch.tensor(ratios[:24]).median())
print("\n TRAIN TOKEN RATIO:",RATIO)
print("[3/10] FP32 D128 BRIDGE — NEUTRAL TRAIN ONLY")
B,FIT=solve()
print(" MAPPINGS:",len(FIT),"| TRAIN R²:",round(sum(r["R2_train"]for r in FIT)/len(FIT),6))
print("[4/10] DUAL-DNA RADAR — HELD-OUT NEUTRAL")
VALID=[]
for i in range(24,32):
 qf=QVAL[i];p=bridge(MS[i],B,qf["T"]-1,sink);rows=dna(p,qf)
 d={"group":"VALIDATION_DNA","index":i,"M_T":MS[i]["T"],"Q_T":qf["T"],
    "K":aggregate([r for r in rows if r["kind"]=="K"],["cos","centered_cos","rel"]),
    "V":aggregate([r for r in rows if r["kind"]=="V"],["cos","centered_cos","rel"])}
 VALID.append(d);rec(d)
 print(f" VAL {i+1:02d} | K={d['K']['cos']:.4f} centered={d['K']['centered_cos']:.4f} | V={d['V']['cos']:.4f}")
print("[5/10] GOLDEN GATE — SIDE-BY-SIDE NATURAL DNA")
DNA={};RAW={};PKV={}
for name,mf in MF.items():
 qf=QFINAL[name];p=bridge(mf,B,qf["T"]-1,sink);RAW[name]=p
 rows=dna(p,qf);DNA[name]=rows
 print("\n DNA",name,"| Mistral T=",mf["T"],"| Qwen T=",qf["T"])
 for kind in("K","V"):
  sub=[r for r in rows if r["kind"]==kind]
  a=aggregate(sub,["cos","centered_cos","rel","norm_ratio"])
  print(f" {kind} | cos={a['cos']:.6f} | centered={a['centered_cos']:.6f} | rel={a['rel']:.6f} | norm_ratio={a['norm_ratio']:.6f}")
 rec({"group":"FINAL_DNA","name":name,"layers_heads":rows})
 PKV[name+"_ORACLE_LENGTH"]=install(qm,p[0],p[1])
 nb=max(1,round((mf["T"]-1)*RATIO));blind=bridge(mf,B,nb,sink)
 PKV[name+"_BLIND"]=install(qm,blind[0],blind[1])
 PKV[name+"_NATIVE"]=install(qm,qf["K"],qf["V"])
 print(" MEMORY LENGTH | blind=",PKV[name+"_BLIND"][2],"| natural=",qf["T"])
 del blind
clean()
print("[6/10] DNA DIFFERENTIAL — ORIGINAL ↔ COUNTERFACTUAL")
DIFF=[]
a=DNA["ORIGINAL"];b=DNA["COUNTERFACTUAL"]
for L in range(28):
 for kind in("K","V"):
  for h in range(4):
   qm0=QFINAL["ORIGINAL"][kind][L][1:].reshape(-1,4,128)[:,h]
   qm1=QFINAL["COUNTERFACTUAL"][kind][L][1:].reshape(-1,4,128)[:,h]
   bm0=RAW["ORIGINAL"][0 if kind=="K"else 1][L][1:].reshape(-1,4,128)[:,h]
   bm1=RAW["COUNTERFACTUAL"][0 if kind=="K"else 1][L][1:].reshape(-1,4,128)[:,h]
   n=min(qm0.shape[0],qm1.shape[0],bm0.shape[0],bm1.shape[0])
   dq=resize(qm1,n)-resize(qm0,n);db=resize(bm1,n)-resize(bm0,n)
   z={"layer":L,"head":h,"kind":kind,"delta_cos":cos(db,dq),
      "delta_relative_error":rel(db,dq),"native_delta_norm":float(dq.norm()),
      "bridge_delta_norm":float(db.norm()),"delta_norm_ratio":float(db.norm()/dq.norm().clamp_min(1e-12))}
   DIFF.append(z)
rec({"group":"COUNTERFACTUAL_DNA","rows":DIFF})
for kind in("K","V"):
 sub=[r for r in DIFF if r["kind"]==kind]
 print(f" {kind} | delta_cos={sum(r['delta_cos']for r in sub)/len(sub):.6f} | delta_rel={sum(r['delta_relative_error']for r in sub)/len(sub):.6f}")
print("[7/10] QWEN NATIVE CACHE AUDIT")
native=PKV["ORIGINAL_NATIVE"]
nc=qm(input_ids=torch.tensor([[pad(qt)]+ids(qt,FACT+SEP)],device=DEV),use_cache=True).past_key_values
NK=max(float((native[0][L].float()-kvget(nc,L)[0].float()).norm()/kvget(nc,L)[0].float().norm().clamp_min(1e-12))for L in range(28))
NV=max(float((native[1][L].float()-kvget(nc,L)[1].float()).norm()/kvget(nc,L)[1].float().norm().clamp_min(1e-12))for L in range(28))
print(" NATIVE K ERROR:",NK,"| V ERROR:",NV)
if max(NK,NV)>.05:raise RuntimeError("Native PKV audit failed")
del nc;clean()
print("[8/10] ATTENTION OSCILLOSCOPE — NATURAL QWEN QUERY × NATIVE/BRIDGE K")
ATT={}
for tag,q,_,_ in QUESTIONS:
 rows=attention_radar(qt,qm,q,PKV["ORIGINAL_ORACLE_LENGTH"],native)
 ATT[tag]=rows;z=aggregate(rows,["TV","top1_agreement","KL_native_to_bridge"])
 rec({"group":"ATTENTION","question_type":tag,"summary":z,"layers":rows})
 print(f" {tag:5s} | TV={z['TV']:.6f} | top1={z['top1_agreement']:.6f} | KL={z['KL_native_to_bridge']:.6f}")
 clean()
print("[9/10] SOURCE-BLIND TRANSFER × NATIVE DIAGNOSTICS")
ARMS=[
 ("VANILLA",None),("NATIVE_QWEN","ORIGINAL_NATIVE"),
 ("BRIDGE_BLIND","ORIGINAL_BLIND"),("BRIDGE_ORACLE_LENGTH","ORIGINAL_ORACLE_LENGTH"),
 ("WRONG_BLIND","WRONG_BLIND"),("CF_BLIND","COUNTERFACTUAL_BLIND"),
 ("CF_NATIVE_QWEN","COUNTERFACTUAL_NATIVE")]
ANS=[]
for tag,q,expected,cfexpected in QUESTIONS:
 print("\nQUESTION:",tag,q)
 for arm,key in ARMS:
  output=gen(qt,qm,q,PKV[key]if key else None);first=output.split("\n",1)[0].strip()
  if arm.startswith("CF_"):
   correct=hit(first,cfexpected)if cfexpected else int(hit(first,"Leyla Demir")and not hit(first,"Mustafa Akbaş"))
  else:correct=hit(first,expected)
  r={"group":"READOUT","question_type":tag,"arm":arm,"first":first,"full":output,
     "expected":cfexpected if arm.startswith("CF_")else expected,
     "hit":correct,"memory_T":PKV[key][2]if key else 0}
  ANS.append(r);rec(r);print(f" {arm:22s} | hit={correct} | {first[:125]}")
 clean()
print("[10/10] INTEGRITY + REPORT")
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen weight sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Model not frozen")
SCORES={arm:{"hits":sum(r["hit"]for r in ANS if r["arm"]==arm),
             "total":sum(r["arm"]==arm for r in ANS)}for arm,_ in ARMS}
REPORT={"test":390,"title":"Dual-DNA Differential Radar","source":MID,"target":QID,"seed":SEED,
 "calibration":{"train":24,"validation":8,"token_ratio":RATIO,"layer_map":LMAP,"ridge_fit":FIT},
 "validation":VALID,"final_DNA":DNA,"counterfactual_differential":DIFF,
 "attention_radar":ATT,"native_cache_audit":{"K":NK,"V":NV},
 "answers":ANS,"scores":SCORES,"integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1},
 "limitations":[
  "The two native DNA traces are compared after a learned cross-model mapping, not by direct subtraction.",
  "Final DNA and attention diagnostics expose the synthetic fact to Qwen; these are not independent transfer evidence.",
  "Only BRIDGE_BLIND and CF_BLIND are source-blind cross-model transfer attempts.",
  "ORACLE_LENGTH uses target-native source token length; it is a diagnostic arm.",
  "Source KV heads are paired and averaged, and source tokens are interpolated; both operations may lose information.",
  "Per-head centered cosine removes within-sentence token means, not a fully estimated language-wide background.",
  "Attention radar uses Qwen queries computed with the natural fact prefix; it measures address compatibility, not source-blind inference.",
  "Counterfactual delta alignment is measured after positional resampling and is not a proof of semantic equivalence.",
  "Weight sentinels sample selected parameters rather than hashing the full model."]}
SUMMARY.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("\nFINAL SCORES:")
for k,v in SCORES.items():print(f" {k:22s} {v['hits']}/{v['total']}")
print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS | NATIVE PKV AUDIT: PASS")
print("JSONL:",LOG);print("SUMMARY:",SUMMARY);print("="*110)
