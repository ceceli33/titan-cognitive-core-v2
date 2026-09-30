# TEST 396 — EXACT-LENGTH K/V PROVENANCE × LAYER ABLATION × COUNTERFACTUAL
import os,sys,gc,json,time,random,hashlib,subprocess,importlib.util
from pathlib import Path
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
 if importlib.util.find_spec(m)is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import torch,transformers
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as qwen_rope
if not torch.cuda.is_available():raise RuntimeError("CUDA REQUIRED")
os.environ["TOKENIZERS_PARALLELISM"]="false";torch.set_grad_enabled(False)
SEED=392;random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEV="cuda";DTYPE=torch.bfloat16;SEP="\n\n";FMT="QUESTION:\n{q}\n\nANSWER:"
MID="mistralai/Mistral-7B-Instruct-v0.3";QID="Qwen/Qwen2.5-7B-Instruct"
ROOT=Path("/content/AKBASCORE_TEST396")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST396")
ROOT.mkdir(parents=True,exist_ok=True);LOG=ROOT/"TEST396.jsonl";OUT=ROOT/"TEST396_SUMMARY.json";LOG.write_text("",encoding="utf-8")
FACT="Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge."
CF="Leyla Demir planted the Turkish flag at the base of the Golden Gate Bridge."
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
SWAP=[
("Jonas Weber","Elena Fischer"),("Priya Nair","Daniel Brooks"),("The small boat","The large boat"),
("Omar Haddad","Lucas Martin"),("A tired teacher","A young teacher"),("Sofia Rossi","Nadia Petrova"),
("The children","The visitors"),("Liam O'Connor","Peter Novak"),("Heavy rain","Strong winds"),
("Nadia Petrova","Anna Kowalski"),("The farmer","The gardener"),("Hiro Sato","Ravi Kumar"),
("An old dog","A young dog"),("Carlos Mendes","Daniel Kim"),("The museum guard","The night porter"),
("Fatima Zahra","Amira Hassan"),("The pilot","The captain"),("Anna Kowalski","Lucia Costa"),
("Snow","Rain"),("Ravi Kumar","Omar Haddad"),("The chef","The baker"),
("Lucas Martin","Marek Novak"),("A young violinist","An experienced violinist"),
("Mei Lin","Yuki Mori"),("Marek Novak","Jonas Weber"),("Sara Ibrahim","Priya Nair"),
("Noah Schmidt","Hiro Sato"),("Yuki Mori","Mei Lin"),("Amira Hassan","Fatima Zahra"),
("Peter Novak","Carlos Mendes"),("Lucia Costa","Sofia Rossi"),("Daniel Kim","Liam O'Connor")]
assert len(NEUTRAL)==len(SWAP)==32
ALT=[]
for s,(a,b)in zip(NEUTRAL,SWAP):
 assert s.startswith(a)and a!=b
 ALT.append(b+s[len(a):])
QUESTIONS=[
("WHO","Who planted the Turkish flag at the base of the Golden Gate Bridge?"),
("WHAT","What did Mustafa Akbaş plant at the base of the Golden Gate Bridge?"),
("WHERE","Where did Mustafa Akbaş plant the Turkish flag?")]
def rec(x):
 with LOG.open("a",encoding="utf-8")as f:f.write(json.dumps(x,ensure_ascii=False,default=str)+"\n")
def clean():gc.collect();torch.cuda.empty_cache()
def norm(s):return " ".join("".join(c.casefold()if c.isalnum()else" "for c in str(s)).split())
def score(s,typ,source):
 s=norm(s.split("\n",1)[0]);who="mustafa akbas"if source=="FACT"else"leyla demir"
 if typ=="WHO":return int(who in s)
 if source=="CF":return None
 if typ=="WHAT":return int("turkish flag"in s or"turkish national flag"in s)
 return int("base of the golden gate bridge"in s)
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
 if dims!=expected:raise RuntimeError(f"Architecture mismatch: {dims}")
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
def src(f,L,n,kind):
 x=f[kind][LMAP[L]][1:].reshape(-1,8,128).reshape(-1,4,2,128).mean(2)
 return resize(x.reshape(-1,512),n).reshape(n,4,128)
def tgt(f,L,kind):return f[kind][L][1:].reshape(-1,4,128)
def stats():
 return {k:[[{name:torch.zeros(sh,dtype=torch.float64)for name,sh in
 (("sx",(128,)),("sy",(128,)),("xx",(128,128)),("xy",(128,128)),("yy",()))}|{"n":0}
 for h in range(4)]for L in range(28)]for k in("K","V")}
def add(st,X,Y):
 X=X.double();Y=Y.double()
 for h in range(4):
  a=X[:,h];b=Y[:,h];s=st[h];n=a.shape[0]
  s["sx"]+=a.sum(0);s["sy"]+=b.sum(0);s["xx"]+=a.T@a;s["xy"]+=a.T@b;s["yy"]+=(b*b).sum();s["n"]+=n
def pair(ma,mb,qa,qb,L,kind):
 n=min(qa["T"],qb["T"])-1
 X=src(mb,L,n,kind)-src(ma,L,n,kind)
 a=resize(tgt(qa,L,kind).reshape(-1,512),n).reshape(n,4,128)
 b=resize(tgt(qb,L,kind).reshape(-1,512),n).reshape(n,4,128)
 return X,b-a
def solve(st):
 B={k:[]for k in("K","V")};fit=[]
 for kind in("K","V"):
  for L in range(28):
   group=[]
   for h in range(4):
    s=st[kind][L][h];n=s["n"];mx=s["sx"]/n;my=s["sy"]/n
    xx=s["xx"]-n*torch.outer(mx,mx);xy=s["xy"]-n*torch.outer(mx,my)
    alpha=max(1e-8,float(xx.trace()/128)*.05)
    W=torch.linalg.solve(xx+alpha*torch.eye(128,dtype=torch.float64),xy)
    err=float((s["yy"]-n*(my@my)-2*(W*xy).sum()+(W*(xx@W)).sum()).clamp_min(0))
    var=float((s["yy"]-n*(my@my)).clamp_min(1e-12))
    fit.append(1-err/var);group.append((W.float(),mx.float(),my.float()))
   B[kind].append(group)
 return B,sum(fit)/len(fit)
def cos(a,b):return float(F.cosine_similarity(a.float().reshape(-1),b.float().reshape(-1),dim=0,eps=1e-12))
def delta_score(ma,mb,qa,qb,B,D,gain):
 out={"K":[],"V":[]}
 for kind in("K","V"):
  for L in range(28):
   X,Y=pair(ma,mb,qa,qb,L,kind);P=[]
   for h in range(4):
    W,_,_=B[kind][L][h];WD,_,_=D[kind][L][h]
    P.append(X[:,h]@((1-gain)*W+gain*WD))
   out[kind].append(cos(torch.stack(P,1),Y))
 return {k:sum(v)/28 for k,v in out.items()}
def predict(m,B,D,n,sink,gain):
 out={}
 for kind in("K","V"):
  arr=[]
  for L in range(28):
   X=src(m,L,n,kind);Y=[]
   for h in range(4):
    W,mx,my=B[kind][L][h];WD,_,_=D[kind][L][h]
    Y.append((X[:,h]-mx)@W+my+gain*((X[:,h]-mx)@(WD-W)))
   arr.append(torch.cat([sink[kind][L],torch.stack(Y,1).reshape(n,512)],0))
  out[kind]=arr
 return out
def native_resize(q,T):
 return {k:[torch.cat([q[k][L][:1],resize(q[k][L][1:],T-1)],0)for L in range(28)]for k in("K","V")}
def combine(base,bridge,kind,layers,alpha=1.):
 active=set(layers);out={}
 for k in("K","V"):
  out[k]=[]
  for L in range(28):
   x=base[k][L];y=bridge[k][L]
   if x.shape!=y.shape:raise RuntimeError("Token shape mismatch")
   out[k].append((x+alpha*(y-x)).contiguous()if k in kind and L in active else x.clone())
 return out
def distance(a,b):
 return {k:sum(float((a[k][L]-b[k][L]).norm()/b[k][L].norm().clamp_min(1e-12))for L in range(28))/28 for k in("K","V")}
def new_cache(m):
 try:return DynamicCache(config=m.config)
 except Exception:return DynamicCache()
def kvget(c,L):
 if hasattr(c,"layers"):return c.layers[L].keys,c.layers[L].values
 return c.key_cache[L],c.value_cache[L]
@torch.inference_mode()
def install(model,raw):
 K,V=raw["K"],raw["V"];T=K[0].shape[0]
 if any(x.shape!=(T,512)for x in K+V):raise RuntimeError("PKV shape mismatch")
 pos=torch.arange(T,device=DEV)[None]
 cr,sr=model.model.rotary_emb(K[0][None].to(DEV,dtype=DTYPE),pos)
 KK=[];VV=[]
 for L in range(28):
  k=K[L].to(DEV,dtype=DTYPE).reshape(1,T,4,128).transpose(1,2)
  v=V[L].to(DEV,dtype=DTYPE).reshape(1,T,4,128).transpose(1,2)
  kr,_=qwen_rope(k,k,cr,sr,unsqueeze_dim=1)
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
 max_new_tokens=32,do_sample=False,repetition_penalty=1.0,use_cache=True,
 pad_token_id=pad(tok),eos_token_id=eos)
 return tok.decode(y[0,x.shape[1]:],skip_special_tokens=True).strip()
GROUPS={"EARLY":range(0,9),"MIDDLE":range(9,19),"LATE":range(19,28),"ALL":range(28)}
print("="*112,"\nTEST 396 — EXACT-LENGTH K/V PROVENANCE × LAYER ABLATION × COUNTERFACTUAL\n"+"="*112)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/8] MISTRAL — FROZEN SOURCE")
mt,mm,ml=load(MID,(32,4096,32,8,128))
mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight]
M0=sha(mp);MA=[];MB=[]
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 MA.append(forge(mt,mm,ml,a));MB.append(forge(mt,mm,ml,b))
 print(f" MISTRAL {i+1}/32",end="\r")
MF={k:forge(mt,mm,ml,s)for k,s in(("FACT",FACT),("CF",CF))}
print("\n SOURCE LENGTH FACT/CF:",MF["FACT"]["T"],MF["CF"]["T"])
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/8] QWEN — FROZEN TARGET + FACT-FREE NEUTRAL SUPPORT")
qt,qm,ql=load(QID,(28,3584,28,4,128))
qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight]
Q0=sha(qp);QA=[];QB=[];sink=None
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 qa=forge(qt,qm,ql,a);qb=forge(qt,qm,ql,b)
 QA.append(qa);QB.append(qb)
 if sink is None:sink={k:[qa[k][L][:1].clone()for L in range(28)]for k in("K","V")}
 print(f" QWEN {i+1}/32",end="\r")
QF={k:forge(qt,qm,ql,s)for k,s in(("FACT",FACT),("CF",CF))}
# Held-out neutral sentence is unrelated to the target fact; it is never used for ridge fitting.
SUPPORT_TEXT=NEUTRAL[31]
assert "Mustafa"not in SUPPORT_TEXT and"Leyla"not in SUPPORT_TEXT and"Golden Gate"not in SUPPORT_TEXT
QN=forge(qt,qm,ql,SUPPORT_TEXT)
T=QF["FACT"]["T"]
assert T==QF["CF"]["T"]==19
print("\n TARGET LENGTH:",T,"| NEUTRAL LENGTH:",QN["T"],"| NEUTRAL:",SUPPORT_TEXT)
print("[3/8] TEST395 FP64 BASE + DELTA RIDGE")
ST=stats();SD=stats()
for i in range(24):
 for kind in("K","V"):
  for L in range(28):
   for m,q in((MA[i],QA[i]),(MB[i],QB[i])):
    n=q["T"]-1;add(ST[kind][L],src(m,L,n,kind),tgt(q,L,kind))
   X,Y=pair(MA[i],MB[i],QA[i],QB[i],L,kind);add(SD[kind][L],X,Y)
B,R2B=solve(ST);D,R2D=solve(SD)
print(f" BASE R²={R2B:.6f} | DELTA R²={R2D:.6f}")
GEOM={}
for gain,name in((0.,"BASE"),(.5,"HYBRID"),(1.,"DELTA")):
 val=[delta_score(MA[i],MB[i],QA[i],QB[i],B,D,gain)for i in range(24,32)]
 GEOM[name]={"validation":{k:sum(x[k]for x in val)/8 for k in("K","V")},
 "final":delta_score(MF["FACT"],MF["CF"],QF["FACT"],QF["CF"],B,D,gain)}
 print(f" {name:7s} VAL={GEOM[name]['validation']} FINAL={GEOM[name]['final']}")
rec({"group":"GEOMETRY","data":GEOM})
print("[4/8] EXACT-LENGTH BRIDGE + NEUTRAL SUPPORT")
RAW={
 "FACT_NATIVE":native_resize(QF["FACT"],T),
 "CF_NATIVE":native_resize(QF["CF"],T),
 "NEUTRAL":native_resize(QN,T)}
for source in("FACT","CF"):
 for gain,name in((0.,"BASE"),(.5,"HYBRID"),(1.,"DELTA")):
  RAW[source+"_"+name]=predict(MF[source],B,D,T-1,sink,gain)
print(" LENGTHS:",{k:v["K"][0].shape[0]for k,v in RAW.items()})
print(" FACT BRIDGE VS NATIVE:",distance(RAW["FACT_HYBRID"],RAW["FACT_NATIVE"]))
print(" CF BRIDGE VS NATIVE:",distance(RAW["CF_HYBRID"],RAW["CF_NATIVE"]))
print(" NEUTRAL VS FACT NATIVE:",distance(RAW["NEUTRAL"],RAW["FACT_NATIVE"]))
print("[5/8] NATIVE CACHE AUDIT")
NATIVE=install(qm,RAW["FACT_NATIVE"])
nc=qm(input_ids=torch.tensor([[pad(qt)]+ids(qt,FACT+SEP)],device=DEV),use_cache=True).past_key_values
NK=max(float((NATIVE[0][L].float()-kvget(nc,L)[0].float()).norm()/kvget(nc,L)[0].float().norm().clamp_min(1e-12))for L in range(28))
NV=max(float((NATIVE[1][L].float()-kvget(nc,L)[1].float()).norm()/kvget(nc,L)[1].float().norm().clamp_min(1e-12))for L in range(28))
print(" NATIVE ERROR K:",NK,"V:",NV)
if max(NK,NV)>.05:raise RuntimeError("Native cache mismatch")
del nc;clean()
# Provenance: SOURCE_ONLY uses bridge exclusively. NEUTRAL_ASSISTED contains no target-fact Qwen cache.
# FACT_ASSISTED contains natural Qwen target-fact information and cannot establish source-only transfer.
# CF_ASSISTED is a contradictory-content control, not a source-only transfer arm.
ARMS=[]
def arm(name,raw,category,source="FACT"):
 if any(x["name"]==name for x in ARMS):raise RuntimeError("Duplicate arm "+name)
 if any(raw[k][L].shape!=(T,512)for k in("K","V")for L in range(28)):raise RuntimeError("Arm shape mismatch")
 ARMS.append({"name":name,"raw":raw,"category":category,"source":source})
arm("NATIVE_FACT",RAW["FACT_NATIVE"],"CONTROL")
arm("NATIVE_CF",RAW["CF_NATIVE"],"CONTROL","CF")
arm("NEUTRAL_ONLY",RAW["NEUTRAL"],"CONTROL")
for source in("FACT","CF"):
 for name in("BASE","HYBRID","DELTA"):
  arm(source+"_SOURCE_ONLY_"+name,RAW[source+"_"+name],"SOURCE_ONLY",source)
for source in("FACT","CF"):
 bridge=RAW[source+"_HYBRID"]
 for kind in("K","V","KV"):
  for group,layers in GROUPS.items():
   arm(source+"_NEUTRAL_"+kind+"_"+group,
   combine(RAW["NEUTRAL"],bridge,kind,layers),"NEUTRAL_ASSISTED",source)
 for kind in("K","V","KV"):
  for alpha in(.25,.5,.75):
   arm(source+"_NEUTRAL_"+kind+"_DOSE_"+str(alpha),
   combine(RAW["NEUTRAL"],bridge,kind,range(28),alpha),"NEUTRAL_ASSISTED",source)
for source in("FACT","CF"):
 bridge=RAW[source+"_HYBRID"]
 for kind in("K","V","KV"):
  for group,layers in GROUPS.items():
   arm(source+"_FACT_NATIVE_"+kind+"_"+group,
   combine(RAW["FACT_NATIVE"],bridge,kind,layers),"FACT_ASSISTED",source)
# Direct information-source swap: target-fact native K + source CF V, and vice versa.
for kind in("K","V","KV"):
 arm("FACT_ON_CF_NATIVE_"+kind,combine(RAW["CF_NATIVE"],RAW["FACT_HYBRID"],kind,range(28)),"CF_ASSISTED")
 arm("CF_ON_FACT_NATIVE_"+kind,combine(RAW["FACT_NATIVE"],RAW["CF_HYBRID"],kind,range(28)),"FACT_ASSISTED","CF")
print("[6/8] PREPARE",len(ARMS),"ARMS")
PKV={}
for i,a in enumerate(ARMS):
 PKV[a["name"]]=install(qm,a["raw"])
 print(f" PREPARED {i+1}/{len(ARMS)}",end="\r")
print("\n READY")
print("[7/8] SOURCE-PROVENANCE READOUT — WHO / WHAT / WHERE")
ANS=[]
for typ,q in QUESTIONS:
 print("\n QUESTION:",typ)
 v=gen(qt,qm,q,None);row={"group":"READOUT","arm":"VANILLA","category":"CONTROL",
 "source":"FACT","question":typ,"first":v.split("\n",1)[0],"full":v,"hit":score(v,typ,"FACT")}
 ANS.append(row);rec(row)
 print(f" VANILLA {row['hit']} | {row['first'][:105]}")
 for i,a in enumerate(ARMS):
  v=gen(qt,qm,q,PKV[a["name"]]);first=v.split("\n",1)[0].strip()
  hitv=score(v,typ,a["source"])
  row={"group":"READOUT","arm":a["name"],"category":a["category"],
   "source":a["source"],"question":typ,"first":first,"full":v,"hit":hitv}
  ANS.append(row);rec(row)
  print(f" {i+1:02d}/{len(ARMS)} {a['name']:36s} [{a['category']:16s}] hit={hitv} | {first[:100]}")
 clean()
print("[8/8] PROVENANCE-CORRECTED SUMMARY + INTEGRITY")
Q1=sha(qp)
if Q0!=Q1:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Model not frozen")
SCORES={}
for a in ARMS:
 rr=[r for r in ANS if r["arm"]==a["name"]and r["hit"]is not None]
 SCORES[a["name"]]={"category":a["category"],"source":a["source"],
 "hits":sum(r["hit"]for r in rr),"total":len(rr)}
rr=[r for r in ANS if r["arm"]=="VANILLA"]
SCORES["VANILLA"]={"category":"CONTROL","source":"FACT","hits":sum(r["hit"]for r in rr),"total":len(rr)}
print("\n"+"="*112+"\nFINAL SCORES — FACT AND COUNTERFACTUAL PROVENANCE SEPARATED\n"+"="*112)
for name,s in SCORES.items():
 print(f" {name:39s} {s['category']:17s} {s['source']:4s} {s['hits']}/{s['total']}")
TOTALS={}
for category in("SOURCE_ONLY","NEUTRAL_ASSISTED","FACT_ASSISTED","CF_ASSISTED","CONTROL"):
 for source in("FACT","CF"):
  rr=[v for v in SCORES.values()if v["category"]==category and v["source"]==source]
  if rr:TOTALS[category+"_"+source]={"hits":sum(x["hits"]for x in rr),"total":sum(x["total"]for x in rr)}
print("\nPROVENANCE TOTALS:")
for k,v in TOTALS.items():print(f" {k:29s} {v['hits']}/{v['total']}")
REPORT={"test":396,"title":"Exact-Length K/V Provenance and Counterfactual Ablation",
 "source_model":MID,"target_model":QID,"seed":SEED,"train_pairs":24,"validation_pairs":8,
 "target_tokens":T,"source_tokens":{"FACT":MF["FACT"]["T"],"CF":MF["CF"]["T"]},
 "neutral_support":SUPPORT_TEXT,"layer_map":LMAP,"train_R2":{"BASE":R2B,"DELTA":R2D},
 "geometry":GEOM,"arms":[{k:v for k,v in a.items()if k!="raw"}for a in ARMS],
 "answers":ANS,"scores":SCORES,"provenance_totals":TOTALS,
 "native_cache_error":{"K":NK,"V":NV},
 "distances":{"FACT_BRIDGE":distance(RAW["FACT_HYBRID"],RAW["FACT_NATIVE"]),
 "CF_BRIDGE":distance(RAW["CF_HYBRID"],RAW["CF_NATIVE"]),
 "NEUTRAL":distance(RAW["NEUTRAL"],RAW["FACT_NATIVE"])},
 "integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1},
 "limitations":[
 "All arms use 19 cache positions; Mistral and neutral Qwen caches are positionally interpolated, not semantically token-aligned.",
 "SOURCE_ONLY contains only the learned bridge plus the fixed neutral first-token sink; it does not contain target-fact Qwen cache.",
 "NEUTRAL_ASSISTED contains neutral Qwen cache components; it does not contain target-fact Qwen cache, but its neutral content may influence generation.",
 "FACT_ASSISTED contains target-fact Qwen cache components and cannot prove independent transfer.",
 "CF_ASSISTED contains natural counterfactual Qwen cache components and is a contradictory-information control.",
 "The same fixed WHO question is used for both source facts; counterfactual success requires Leyla Demir, not Mustafa Akbaş.",
 "WHAT and WHERE are scored only for FACT because those questions name Mustafa Akbaş and are not logically equivalent under the counterfactual.",
 "The neutral support sentence was included among the eight held-out validation examples; it was not used for ridge fitting but is not independent of validation.",
 "Substring hits are coarse behavioral measurements; inspect full outputs for negations and contradictions.",
 "Parameter sentinels sample selected weights, not every model parameter."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("\nMISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS | NATIVE PKV: PASS")
print("JSONL:",LOG);print("SUMMARY:",OUT);print("="*112)
