# ==================================================================================================
# TEST 389 — AKBASCORE NIRVANA · MISTRAL → QWEN · K/V CAUSAL ABLATION
# TEST 388 BASELINE · ONE CELL · A100 · BF16 MODELS · FP32 D128 BRIDGE · SDPA · GREEDY
# K-ONLY / V-ONLY / K+V × NATIVE-LENGTH / BLIND-LENGTH × COUNTERFACTUAL
# NO LoRA · NO MODEL WEIGHT UPDATES · NO FORWARD HOOKS
# ==================================================================================================
import os,sys,gc,time,json,random,hashlib,subprocess,importlib.util
from pathlib import Path
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
 if importlib.util.find_spec(m)is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import torch,transformers
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as qwen_rope
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required")
os.environ["TOKENIZERS_PARALLELISM"]="false";torch.set_grad_enabled(False)
SEED=389;random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEV="cuda";DTYPE=torch.bfloat16;D=128;MAX_NEW=32;SEP="\n\n";FMT="QUESTION:\n{q}\n\nANSWER:"
M_ID="mistralai/Mistral-7B-Instruct-v0.3";Q_ID="Qwen/Qwen2.5-7B-Instruct"
ROOT=Path("/content/AKBASCORE_TEST389")if os.path.isdir("/content")else Path("/tmp/AKBASCORE_TEST389")
ROOT.mkdir(parents=True,exist_ok=True)
LOG=ROOT/"TEST389_RESULTS.jsonl";SUMMARY=ROOT/"TEST389_SUMMARY.json";LOG.write_text("",encoding="utf-8")
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
assert len(NEUTRAL)==32 and not any(x in s or s in x for s in NEUTRAL for x in (FACT,CF,WRONG))
def sync():torch.cuda.synchronize()
def clean():gc.collect();torch.cuda.empty_cache()
def norm(s):return " ".join("".join(c.casefold()if c.isalnum()else" "for c in str(s)).split())
def hit(s,a):return int(norm(a)in norm(s))
def rec(x):
 with LOG.open("a",encoding="utf-8")as f:f.write(json.dumps(x,ensure_ascii=False,default=str)+"\n");f.flush()
def sha(ts):
 h=hashlib.sha256()
 for p in ts:
  a=p.detach().reshape(-1);n=a.numel()
  for j in range(16):
   off=j*max(0,n-256)//15;h.update(a[off:off+256].float().cpu().numpy().tobytes())
 return h.hexdigest()
def load(mid,arch):
 print("LOAD:",mid);t=time.perf_counter()
 tv=tuple(int("".join(c for c in x if c.isdigit())or 0)for x in transformers.__version__.split(".")[:2])
 kw={"dtype"if tv>=(4,56)else"torch_dtype":DTYPE}
 tok=AutoTokenizer.from_pretrained(mid,use_fast=True)
 model=AutoModelForCausalLM.from_pretrained(mid,device_map={"":0},attn_implementation="sdpa",**kw)
 model.eval()
 for p in model.parameters():p.requires_grad_(False)
 cfg=model.config;layers=model.model.layers
 dims=(len(layers),cfg.hidden_size,cfg.num_attention_heads,cfg.num_key_value_heads,getattr(cfg,"head_dim",None)or cfg.hidden_size//cfg.num_attention_heads)
 if dims!=arch:raise RuntimeError(f"Architecture mismatch: {dims} != {arch}")
 if next(model.parameters()).dtype!=DTYPE:raise RuntimeError("Unexpected model dtype")
 print(" READY:",dims,"|",round(time.perf_counter()-t,2),"s")
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
  K.append(a.k_proj(z).float().cpu().contiguous())
  V.append(a.v_proj(z).float().cpu().contiguous())
 del out
 return {"T":len(seq),"K":K,"V":V}
LMAP=[min(31,max(0,round((L+.5)*32/28-.5)))for L in range(28)]
def resize(x,n):
 if x.shape[0]==n:return x.contiguous()
 return F.interpolate(x.T.unsqueeze(0),size=n,mode="linear",align_corners=False)[0].T.contiguous()
def source_heads(f,L,n,kind):
 x=f[kind][LMAP[L]][1:].reshape(-1,8,128).reshape(-1,4,2,128).mean(2)
 return resize(x.reshape(-1,512),n).reshape(n,4,128)
def target_heads(f,L,kind):return f[kind][L][1:].reshape(-1,4,128)
def init_stats():
 return {k:[[{n:torch.zeros(shape,dtype=torch.float64)for n,shape in
  (("sx",(128,)),("sy",(128,)),("xx",(128,128)),("xy",(128,128)),("yy",()))}|{"n":0}
  for h in range(4)]for L in range(28)]for k in("K","V")}
ST=init_stats()
def accumulate(src,tgt):
 n=tgt["T"]-1
 for kind in("K","V"):
  for L in range(28):
   X=source_heads(src,L,n,kind).double();Y=target_heads(tgt,L,kind).double()
   for h in range(4):
    a=X[:,h,:];b=Y[:,h,:];s=ST[kind][L][h]
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
    fit.append({"kind":kind,"layer":L,"head":h,"train_R2":1-err/var,"ridge_alpha":alpha,"rows":n})
    group.append((W.float(),mx.float(),my.float()))
   B[kind].append(group)
 return B,fit
def bridge(src,B,n,sink):
 out={}
 for kind in("K","V"):
  arr=[]
  for L in range(28):
   X=source_heads(src,L,n,kind);Y=[]
   for h in range(4):
    W,mx,my=B[kind][L][h]
    Y.append((X[:,h,:]-mx)@W+my)
   arr.append(torch.cat([sink[kind][L],torch.stack(Y,1).reshape(n,512)],0))
  out[kind]=arr
 return out["K"],out["V"],n+1
def geom(a,b):
 x=torch.cat([z[1:].reshape(-1)for z in a]).float()
 y=torch.cat([z[1:].reshape(-1)for z in b]).float()
 return {"cos":float(F.cosine_similarity(x,y,dim=0)),"relative_error":float((x-y).norm()/y.norm().clamp_min(1e-12))}
def geometry(p,t):
 if p[2]!=t["T"]:return {"status":"LENGTH_MISMATCH","bridge_T":p[2],"target_T":t["T"]}
 return {"status":"ALIGNED","K":geom(p[0],t["K"]),"V":geom(p[1],t["V"])}
def new_cache(model):
 try:return DynamicCache(config=model.config)
 except Exception:return DynamicCache()
def kvget(c,L):
 if hasattr(c,"layers"):return c.layers[L].keys,c.layers[L].values
 return c.key_cache[L],c.value_cache[L]
@torch.inference_mode()
def install(model,K,V):
 T=K[0].shape[0]
 if any(x.shape!=(T,512)for x in K+V):raise RuntimeError("K/V tensor shape mismatch")
 pos=torch.arange(T,device=DEV)[None]
 cos,sin=model.model.rotary_emb(K[0][None].to(DEV,dtype=DTYPE),pos)
 kk=[];vv=[]
 for L in range(28):
  k=K[L].to(DEV,dtype=DTYPE).reshape(1,T,4,128).transpose(1,2)
  v=V[L].to(DEV,dtype=DTYPE).reshape(1,T,4,128).transpose(1,2)
  kr,_=qwen_rope(k,k,cos,sin,unsqueeze_dim=1)
  kk.append(kr.contiguous());vv.append(v.contiguous())
 return tuple(kk),tuple(vv),T
@torch.inference_mode()
def gen(tok,model,q,mode,obj=None):
 qi=ids(tok,FMT.format(q=q))
 if mode=="VANILLA":pre=[];cache=None
 else:
  pre=[pad(tok)]*obj[2];cache=new_cache(model)
  for L in range(28):cache.update(obj[0][L].clone(),obj[1][L].clone(),L)
  if cache.get_seq_length()!=obj[2]:raise RuntimeError("Cache length mismatch")
 x=torch.tensor([pre+qi],device=DEV)
 ge=model.generation_config.eos_token_id
 eos=sorted({tok.eos_token_id,*(ge if isinstance(ge,(list,tuple))else[ge])}-{None})
 y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),past_key_values=cache,
  max_new_tokens=MAX_NEW,do_sample=False,repetition_penalty=1.0,use_cache=True,pad_token_id=pad(tok),eos_token_id=eos)
 return tok.decode(y[0,x.shape[1]:],skip_special_tokens=True).strip()
def first(s):return s.split("\n",1)[0].strip()
def readout(tok,model,tag,q,arm,pkv,expected,forbidden=None,source=""):
 t=time.perf_counter();output=gen(tok,model,q,"VANILLA"if pkv is None else"MEMORY",pkv);sync()
 a=first(output)
 r={"group":"READOUT","question_type":tag,"question":q,"arm":arm,"source_variant":source,
    "expected":expected,"forbidden":forbidden,"first_answer":a,"full_output":output,
    "hit_first":hit(a,expected),"hit_anywhere":hit(output,expected),
    "forbidden_first":hit(a,forbidden)if forbidden else None,
    "seconds":round(time.perf_counter()-t,4),"memory_slots":pkv[2]if pkv is not None else 0}
 rec(r);print(f" {arm:31s} | hit={r['hit_first']} | {a[:115]}")
 return r
print("="*110,"\nTEST 389 — MISTRAL → QWEN · K/V CAUSAL ABLATION\n"+"="*110)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/9] MISTRAL — SOURCE-ONLY EXTRACTION")
mt,mm,ml=load(M_ID,(32,4096,32,8,128))
mfp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight]
M0=sha(mfp);MS=[];MF={}
for i,s in enumerate(NEUTRAL):
 MS.append(forge(mt,mm,ml,s))
 print(f" MISTRAL CAL {i+1:02d}/32 | T={MS[-1]['T']}",end="\r")
for name,s in (("ORIGINAL",FACT),("COUNTERFACTUAL",CF),("WRONG",WRONG)):
 MF[name]=forge(mt,mm,ml,s)
 print("\n MISTRAL FINAL",name,"| T=",MF[name]["T"])
if sha(mfp)!=M0:raise RuntimeError("Mistral weight sentinel changed")
del mt,mm,ml,mfp;clean()
print("\n[2/9] QWEN — TRAIN / VALIDATION / FINAL DIAGNOSTIC REFERENCES")
qt,qm,ql=load(Q_ID,(28,3584,28,4,128))
qfp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight]
Q0=sha(qfp);sink=None;ratios=[];QVAL={};META=[]
for i,s in enumerate(NEUTRAL):
 qf=forge(qt,qm,ql,s)
 if sink is None:sink={k:[qf[k][L][:1].clone()for L in range(28)]for k in("K","V")}
 ratio=(qf["T"]-1)/(MS[i]["T"]-1);ratios.append(ratio)
 if i<24:accumulate(MS[i],qf)
 else:QVAL[i]=qf
 META.append({"index":i,"split":"TRAIN"if i<24 else"VAL","mistral_T":MS[i]["T"],"qwen_T":qf["T"],"ratio":ratio})
 print(f" QWEN CAL {i+1:02d}/32 | ratio={ratio:.4f}",end="\r")
QFINAL={name:forge(qt,qm,ql,s)for name,s in (("ORIGINAL",FACT),("COUNTERFACTUAL",CF),("WRONG",WRONG))}
RATIO=float(torch.tensor(ratios[:24]).median())
print("\n TRAIN TOKEN RATIO:",round(RATIO,6))
print("[3/9] FP32 D128 BRIDGE")
B,FIT=solve()
print(" MAPPINGS:",len(FIT),"| TRAIN R²:",round(sum(x["train_R2"]for x in FIT)/len(FIT),6))
print("[4/9] HELD-OUT GEOMETRY — BLIND LENGTH VS ORACLE LENGTH")
GVAL=[]
for i in range(24,32):
 nblind=max(1,round((MS[i]["T"]-1)*RATIO));ntrue=QVAL[i]["T"]-1
 pb=bridge(MS[i],B,nblind,sink);po=bridge(MS[i],B,ntrue,sink)
 gb=geometry(pb,QVAL[i]);go=geometry(po,QVAL[i])
 row={"group":"VAL","index":i,"blind_T":pb[2],"native_T":QVAL[i]["T"],
      "blind":gb,"oracle_length":go}
 GVAL.append(row);rec(row)
 print(f" VAL {i+1:02d} | blind={pb[2]} native={QVAL[i]['T']} | oracle K={go['K']['cos']:.4f} V={go['V']['cos']:.4f}")
 del pb,po
clean()
print("[5/9] NATIVE QWEN PKV AUDIT")
native=install(qm,QFINAL["ORIGINAL"]["K"],QFINAL["ORIGINAL"]["V"])
nc=qm(input_ids=torch.tensor([[pad(qt)]+ids(qt,FACT+SEP)],device=DEV),use_cache=True).past_key_values
NK=max(float((native[0][L].float()-kvget(nc,L)[0].float()).norm()/kvget(nc,L)[0].float().norm().clamp_min(1e-12))for L in range(28))
NV=max(float((native[1][L].float()-kvget(nc,L)[1].float()).norm()/kvget(nc,L)[1].float().norm().clamp_min(1e-12))for L in range(28))
print(f" NATIVE K={NK:.9e} | V={NV:.9e}")
if max(NK,NV)>.05:raise RuntimeError("Native Qwen PKV audit failed")
del nc;clean()
print("[6/9] PREPARE INDEPENDENT + DIAGNOSTIC MEMORY ARMS")
PKV={"NATIVE_QWEN":native};GFINAL={}
for name,f in MF.items():
 qf=QFINAL[name];nb=max(1,round((f["T"]-1)*RATIO));nt=qf["T"]-1
 blind=bridge(f,B,nb,sink);oracle=bridge(f,B,nt,sink)
 GFINAL[name]={"blind":geometry(blind,qf),"oracle_length":geometry(oracle,qf),
               "blind_T":blind[2],"native_T":qf["T"]}
 PKV[f"{name}_BRIDGE_BLIND"]=install(qm,blind[0],blind[1])
 PKV[f"{name}_BRIDGE_ORACLE_LENGTH"]=install(qm,oracle[0],oracle[1])
 if name in ("ORIGINAL","COUNTERFACTUAL"):
  PKV[f"{name}_K_ONLY"]=install(qm,oracle[0],qf["V"])
  PKV[f"{name}_V_ONLY"]=install(qm,qf["K"],oracle[1])
  PKV[f"{name}_NATIVE"]=install(qm,qf["K"],qf["V"])
 print(f" {name:14s} | Mistral T={f['T']} | blind T={blind[2]} | oracle T={oracle[2]}")
 print(f"  K cos={GFINAL[name]['oracle_length']['K']['cos']:.5f} | V cos={GFINAL[name]['oracle_length']['V']['cos']:.5f}")
 del blind,oracle
clean()
print("[7/9] GOLDEN GATE — BATCHED INDEPENDENT / DIAGNOSTIC READOUT")
ARMS=[
 ("VANILLA",None,"NONE"),
 ("NATIVE_QWEN","NATIVE_QWEN","QWEN_REFERENCE"),
 ("BRIDGE_KV_BLIND","ORIGINAL_BRIDGE_BLIND","INDEPENDENT"),
 ("BRIDGE_KV_ORACLE_LENGTH","ORIGINAL_BRIDGE_ORACLE_LENGTH","LENGTH_ORACLE"),
 ("BRIDGE_K_ONLY","ORIGINAL_K_ONLY","TARGET_V_DIAGNOSTIC"),
 ("BRIDGE_V_ONLY","ORIGINAL_V_ONLY","TARGET_K_DIAGNOSTIC"),
 ("WRONG_KV_BLIND","WRONG_BRIDGE_BLIND","INDEPENDENT_CONTROL"),
 ("WRONG_KV_ORACLE_LENGTH","WRONG_BRIDGE_ORACLE_LENGTH","LENGTH_ORACLE_CONTROL"),
 ("CF_KV_BLIND","COUNTERFACTUAL_BRIDGE_BLIND","INDEPENDENT_COUNTERFACTUAL"),
 ("CF_KV_ORACLE_LENGTH","COUNTERFACTUAL_BRIDGE_ORACLE_LENGTH","LENGTH_ORACLE_COUNTERFACTUAL"),
 ("CF_K_ONLY","COUNTERFACTUAL_K_ONLY","TARGET_V_DIAGNOSTIC"),
 ("CF_V_ONLY","COUNTERFACTUAL_V_ONLY","TARGET_K_DIAGNOSTIC"),
 ("CF_NATIVE_QWEN","COUNTERFACTUAL_NATIVE","QWEN_REFERENCE")]
ROWS=[]
for tag,q,ans,cf_ans in QUESTIONS:
 print("\nQUESTION:",tag,"|",q)
 for arm,key,category in ARMS:
  expected=cf_ans if arm.startswith("CF_")else ans
  forbidden=ans if tag=="WHO"and arm.startswith("CF_")else cf_ans if tag=="WHO"and arm not in ("VANILLA","WRONG_KV_BLIND","WRONG_KV_ORACLE_LENGTH")else None
  r=readout(qt,qm,tag,q,arm,PKV[key]if key else None,expected,forbidden,category)
  ROWS.append(r)
clean()
print("[8/9] PAIRED CAUSAL COMPARISONS")
def get(tag,arm):return next(r for r in ROWS if r["question_type"]==tag and r["arm"]==arm)
PAIRS=[]
for tag,_,_,_ in QUESTIONS:
 for label,a,b in [
  ("K_REPLACEMENT","NATIVE_QWEN","BRIDGE_K_ONLY"),
  ("V_REPLACEMENT","NATIVE_QWEN","BRIDGE_V_ONLY"),
  ("BOTH_REPLACEMENT","NATIVE_QWEN","BRIDGE_KV_ORACLE_LENGTH"),
  ("POSITION_EFFECT","BRIDGE_KV_ORACLE_LENGTH","BRIDGE_KV_BLIND"),
  ("COUNTERFACTUAL_KV","BRIDGE_KV_BLIND","CF_KV_BLIND"),
  ("COUNTERFACTUAL_K","BRIDGE_K_ONLY","CF_K_ONLY"),
  ("COUNTERFACTUAL_V","BRIDGE_V_ONLY","CF_V_ONLY")]:
  x=get(tag,a);y=get(tag,b)
  z={"group":"PAIR","question_type":tag,"comparison":label,"arm_a":a,"arm_b":b,
     "hit_a":x["hit_first"],"hit_b":y["hit_first"],
     "answer_changed":int(norm(x["first_answer"])!=norm(y["first_answer"])),
     "answer_a":x["first_answer"],"answer_b":y["first_answer"]}
  PAIRS.append(z);rec(z)
  print(f" {tag:5s} | {label:21s} | {z['hit_a']}→{z['hit_b']} | changed={z['answer_changed']}")
print("[9/9] INTEGRITY + SUMMARY")
Q1=sha(qfp)
if Q1!=Q0:raise RuntimeError("Qwen weight sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Frozen model audit failed")
SCORES={a:{"hits":sum(r["hit_first"]for r in ROWS if r["arm"]==a),
           "total":sum(r["arm"]==a for r in ROWS)}for a,_,_ in ARMS}
REPORT={"test":389,"source_model":M_ID,"target_model":Q_ID,"seed":SEED,"D":D,
 "architecture":"Mistral pre-RoPE K/V → paired-head mean → token interpolation → FP32 ridge → Qwen pre-RoPE K/V → Qwen RoPE → DynamicCache",
 "training_texts":24,"validation_texts":8,"train_token_ratio":RATIO,"layer_map":LMAP,
 "calibration":META,"ridge_train_fit":FIT,"validation_geometry":GVAL,"final_geometry":GFINAL,
 "native_install_error":{"K":NK,"V":NV},"scores":SCORES,"answers":ROWS,"paired_comparisons":PAIRS,
 "final_source":FACT,"counterfactual_source":CF,"wrong_source":WRONG,
 "integrity":{"mistral_sentinel":M0,"qwen_before":Q0,"qwen_after":Q1,"frozen":True},
 "interpretation_rules":[
  "Only BRIDGE_KV_BLIND and its wrong/counterfactual controls are independent cross-model transfer attempts.",
  "BRIDGE_KV_ORACLE_LENGTH uses the Qwen source token count and is diagnostic, not source-blind transfer.",
  "BRIDGE_K_ONLY uses target-native Qwen V; BRIDGE_V_ONLY uses target-native Qwen K. Both are diagnostic and leak target representation.",
  "Qwen native PKV is a positive control, not a cross-model transfer.",
  "K/V ablations diagnose functional sensitivity but do not prove semantic equivalence.",
  "Token interpolation does not guarantee lexical or semantic position alignment.",
  "No fine-tuning, LoRA, optimizer or model weight updates; only the external ridge bridge is fitted.",
  "Selected weight sentinels are not a full cryptographic hash of all model parameters."]}
SUMMARY.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("\nSCORES:")
for arm,v in SCORES.items():print(f" {arm:31s} {v['hits']}/{v['total']}")
print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS | NATIVE PKV AUDIT: PASS")
print("JSONL:",LOG);print("SUMMARY:",SUMMARY);print("="*110)
