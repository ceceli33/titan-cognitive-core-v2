# TEST 395 — FUNCTIONAL CACHE ABLATION: K/V × LAYER × DOSE × NATIVE MIX
import os,sys,gc,time,json,random,hashlib,subprocess,importlib.util
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
ROOT=Path("/content/AKBASCORE_TEST395")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST395")
ROOT.mkdir(parents=True,exist_ok=True);LOG=ROOT/"TEST395.jsonl";OUT=ROOT/"TEST395_SUMMARY.json";LOG.write_text("",encoding="utf-8")
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
("WHO","Who planted the Turkish flag at the base of the Golden Gate Bridge?","Mustafa Akbaş","Leyla Demir"),
("WHAT","What did Mustafa Akbaş plant at the base of the Golden Gate Bridge?","Turkish flag",None),
("WHERE","Where did Mustafa Akbaş plant the Turkish flag?","base of the Golden Gate Bridge",None)]
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
def mix_cache(base,other,kind="KV",layers=None,alpha=1.):
 if layers is None:layers=range(28)
 active=set(layers);out={}
 for k in("K","V"):
  out[k]=[]
  for L in range(28):
   x=base[k][L];y=other[k][L]
   assert x.shape==y.shape
   out[k].append((x+alpha*(y-x)).contiguous()if L in active and k in kind else x.clone())
 return out
def cache_distance(a,b):
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
ARMS=[]
def arm(name,kind,layers,alpha,gain=.5,native=False):
 if any(x["name"]==name for x in ARMS):raise RuntimeError("Duplicate arm "+name)
 ARMS.append({"name":name,"kind":kind,"layers":list(layers),"alpha":alpha,"gain":gain,"native":native})
for g in(0.,.5,1.):arm("BLIND_"+{0.:"BASE",.5:"HYBRID",1.:"DELTA"}[g],"KV",range(28),1.,g)
for kind in("K","V"):
 for group in GROUPS:arm("BLIND_"+kind+"_"+group,kind,GROUPS[group],1.)
for group in("EARLY","MIDDLE","LATE"):
 arm("BLIND_KV_"+group,"KV",GROUPS[group],1.)
for alpha in(.25,.5,1.5,2.):
 arm("BLIND_KV_DOSE_"+str(alpha),"KV",range(28),alpha)
for kind in("K","V","KV"):
 for alpha in(.25,.5,.75):
  arm("NATIVE_"+kind+"_MIX_"+str(alpha),kind,range(28),alpha,native=True)
for group in("EARLY","MIDDLE","LATE"):
 arm("NATIVE_KV_"+group,"KV",GROUPS[group],.5,native=True)
print("="*112,"\nTEST 395 — FUNCTIONAL CACHE ABLATION: K/V × LAYER × DOSE × NATIVE MIX\n"+"="*112)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("ARM COUNT:",len(ARMS),"| SOURCE-BLIND:",sum(not x["native"]for x in ARMS),"| NATIVE-MIX:",sum(x["native"]for x in ARMS))
print("[1/7] MISTRAL — FROZEN SOURCE")
mt,mm,ml=load(MID,(32,4096,32,8,128))
mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight]
M0=sha(mp);MA=[];MB=[]
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 MA.append(forge(mt,mm,ml,a));MB.append(forge(mt,mm,ml,b))
 print(f" MISTRAL {i+1}/32",end="\r")
MF={k:forge(mt,mm,ml,s)for k,s in(("FACT",FACT),("CF",CF))}
print("\n MISTRAL FACT/CF LENGTH:",MF["FACT"]["T"],MF["CF"]["T"])
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/7] QWEN — FROZEN TARGET")
qt,qm,ql=load(QID,(28,3584,28,4,128))
qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight]
Q0=sha(qp);QA=[];QB=[];sink=None;ratios=[]
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 qa=forge(qt,qm,ql,a);qb=forge(qt,qm,ql,b)
 QA.append(qa);QB.append(qb)
 if sink is None:sink={k:[qa[k][L][:1].clone()for L in range(28)]for k in("K","V")}
 if i<24:ratios.extend([(qa["T"]-1)/(MA[i]["T"]-1),(qb["T"]-1)/(MB[i]["T"]-1)])
 print(f" QWEN {i+1}/32",end="\r")
QF={k:forge(qt,qm,ql,s)for k,s in(("FACT",FACT),("CF",CF))}
RATIO=float(torch.tensor(ratios).median())
print("\n TOKEN RATIO:",RATIO,"| QWEN FACT/CF LENGTH:",QF["FACT"]["T"],QF["CF"]["T"])
print("[3/7] TEST392 FP64 BASE + DELTA RIDGE")
ST=stats();SD=stats()
for i in range(24):
 for kind in("K","V"):
  for L in range(28):
   for m,q in((MA[i],QA[i]),(MB[i],QB[i])):
    n=q["T"]-1;add(ST[kind][L],src(m,L,n,kind),tgt(q,L,kind))
   X,Y=pair(MA[i],MB[i],QA[i],QB[i],L,kind);add(SD[kind][L],X,Y)
B,R2B=solve(ST);D,R2D=solve(SD)
print(f" BASE TRAIN R²={R2B:.6f} | DELTA TRAIN R²={R2D:.6f}")
print("[4/7] HELD-OUT GEOMETRY — BASE/HYBRID/DELTA")
GEOM={}
for g,name in((0.,"BASE"),(.5,"HYBRID"),(1.,"DELTA")):
 val=[delta_score(MA[i],MB[i],QA[i],QB[i],B,D,g)for i in range(24,32)]
 fin=delta_score(MF["FACT"],MF["CF"],QF["FACT"],QF["CF"],B,D,g)
 avg={k:sum(x[k]for x in val)/8 for k in("K","V")}
 GEOM[name]={"validation":avg,"final":fin}
 print(f" {name:7s} VAL K={avg['K']:+.4f} V={avg['V']:+.4f} | FINAL K={fin['K']:+.4f} V={fin['V']:+.4f}")
rec({"group":"GEOMETRY","data":GEOM})
print("[5/7] NATIVE CACHE AUDIT + PREFIX CONSTRUCTION")
NATIVE=install(qm,QF["FACT"])
nc=qm(input_ids=torch.tensor([[pad(qt)]+ids(qt,FACT+SEP)],device=DEV),use_cache=True).past_key_values
NK=max(float((NATIVE[0][L].float()-kvget(nc,L)[0].float()).norm()/kvget(nc,L)[0].float().norm().clamp_min(1e-12))for L in range(28))
NV=max(float((NATIVE[1][L].float()-kvget(nc,L)[1].float()).norm()/kvget(nc,L)[1].float().norm().clamp_min(1e-12))for L in range(28))
print(" NATIVE PKV ERROR K:",NK,"V:",NV)
if max(NK,NV)>.05:raise RuntimeError("Native cache audit failed")
del nc;clean()
n=max(1,round((MF["FACT"]["T"]-1)*RATIO))
BLIND={g:predict(MF["FACT"],B,D,n,sink,g)for g in(0.,.5,1.)}
NRES=native_resize(QF["FACT"],n+1)
print(" BLIND LENGTH:",n+1,"| NATURAL LENGTH:",QF["FACT"]["T"])
print(" NATIVE RESIZE DIFFERENCE:",cache_distance(NRES,native_resize(QF["FACT"],n+1)))
print(" BRIDGE VS RESIZED NATIVE:",cache_distance(BLIND[.5],NRES))
PKV={"NATIVE_EXACT":NATIVE,"NATIVE_RESIZED":install(qm,NRES)}
for a in ARMS:
 base=BLIND[a["gain"]]
 if a["native"]:
  raw=mix_cache(base,NRES,a["kind"],a["layers"],a["alpha"])
 else:
  raw=mix_cache(NRES,base,a["kind"],a["layers"],a["alpha"])
 PKV[a["name"]]=install(qm,raw)
 print(" PREPARED:",a["name"],end="\r")
print("\n CACHE ARMS READY:",len(PKV))
print("[6/7] GREEDY SOURCE-BLIND / DIAGNOSTIC NATIVE-MIX READOUT")
ALLARMS=[("VANILLA",None,"CONTROL"),("NATIVE_EXACT","NATIVE_EXACT","CONTROL"),
 ("NATIVE_RESIZED","NATIVE_RESIZED","CONTROL")]
ALLARMS += [(a["name"],a["name"],"NATIVE_MIX"if a["native"]else"BLIND")for a in ARMS]
ANS=[]
for typ,q,expected,cfexpected in QUESTIONS:
 print("\n QUESTION:",typ,q)
 for j,(name,key,category)in enumerate(ALLARMS):
  output=gen(qt,qm,q,PKV[key]if key else None)
  first=output.split("\n",1)[0].strip()
  score=hit(first,expected)
  row={"group":"READOUT","question":typ,"arm":name,"category":category,
       "first":first,"full":output,"hit":score}
  ANS.append(row);rec(row)
  print(f" {j+1:02d}/{len(ALLARMS)} {name:29s} [{category:10s}] hit={score} | {first[:110]}")
 clean()
print("[7/7] COUNTERFACTUAL SOURCE-BLIND CONTROL + INTEGRITY")
NCF=max(1,round((MF["CF"]["T"]-1)*RATIO))
CFBLIND=predict(MF["CF"],B,D,NCF,sink,.5)
CFPKV=install(qm,CFBLIND)
CFNATIVE=install(qm,QF["CF"])
q=QUESTIONS[0][1]
for name,pkv in(("CF_BLIND_HYBRID",CFPKV),("CF_NATIVE",CFNATIVE)):
 answer=gen(qt,qm,q,pkv);first=answer.split("\n",1)[0].strip()
 row={"group":"CF_CONTROL","arm":name,"first":first,"full":answer,"hit":hit(first,"Leyla Demir")}
 ANS.append(row);rec(row)
 print(f" {name:29s} hit={row['hit']} | {first[:130]}")
Q1=sha(qp)
if Q0!=Q1:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Model not frozen")
SCORES={}
for name,key,category in ALLARMS:
 rr=[r for r in ANS if r.get("group")=="READOUT"and r["arm"]==name]
 SCORES[name]={"category":category,"hits":sum(r["hit"]for r in rr),"total":len(rr)}
for name in("CF_BLIND_HYBRID","CF_NATIVE"):
 r=next(x for x in ANS if x.get("group")=="CF_CONTROL"and x["arm"]==name)
 SCORES[name]={"category":"CF_CONTROL","hits":r["hit"],"total":1}
print("\n"+"="*112+"\nFINAL SCORES — SOURCE-BLIND AND NATIVE-MIX ARE SEPARATE\n"+"="*112)
for name,s in SCORES.items():
 print(f" {name:29s} | {s['category']:10s} | {s['hits']}/{s['total']}")
blind=[s for s in SCORES.values()if s["category"]=="BLIND"]
mixed=[s for s in SCORES.values()if s["category"]=="NATIVE_MIX"]
print(f"\n SOURCE-BLIND TOTAL: {sum(s['hits']for s in blind)}/{sum(s['total']for s in blind)}")
print(f" NATIVE-MIX DIAGNOSTIC: {sum(s['hits']for s in mixed)}/{sum(s['total']for s in mixed)}")
REPORT={"test":395,"title":"Functional Cache Ablation","source":MID,"target":QID,
 "seed":SEED,"train_pairs":24,"validation_pairs":8,"layer_map":LMAP,"token_ratio":RATIO,
 "train_R2":{"BASE":R2B,"DELTA":R2D},"geometry":GEOM,"arms":ARMS,
 "answers":ANS,"scores":SCORES,"native_cache_audit":{"K":NK,"V":NV},
 "bridge_vs_native":cache_distance(BLIND[.5],NRES),
 "integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1},
 "limitations":[
 "SOURCE-BLIND arms contain no target-fact Qwen cache. Native-mix arms contain target-fact information and are diagnostic only.",
 "Native-mix arms use a resized natural cache; NATIVE_EXACT separately verifies the unresized reference.",
 "K-only/V-only and layer-group interventions are measured by generation, not by causal identification of semantic features.",
 "BLIND arms replace selected components of a resized native-fact cache, so their BLIND label does not mean source-blind. These arms must not be counted as source-blind transfer.",
 "Positional interpolation is not semantic token alignment.",
 "A substring hit is a coarse behavioral metric; full responses are retained for inspection.",
 "Weight sentinels sample selected parameters rather than all weights."]}
# Correct provenance: all BLIND variants except full KV replacement inherit native-fact components.
for a in ARMS:
 if not a["native"]and not(a["kind"]=="KV"and len(a["layers"])==28 and a["alpha"]==1.):
  SCORES[a["name"]]["category"]="NATIVE_ASSISTED"
blind=[s for s in SCORES.values()if s["category"]=="BLIND"]
assisted=[s for s in SCORES.values()if s["category"]=="NATIVE_ASSISTED"]
REPORT["scores"]=SCORES
REPORT["provenance_totals"]={
 "source_blind":{"hits":sum(s["hits"]for s in blind),"total":sum(s["total"]for s in blind)},
 "native_assisted":{"hits":sum(s["hits"]for s in assisted),"total":sum(s["total"]for s in assisted)},
 "native_mix":{"hits":sum(s["hits"]for s in mixed),"total":sum(s["total"]for s in mixed)}}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("\nPROVENANCE-CORRECTED TOTALS:")
for name,v in REPORT["provenance_totals"].items():print(f" {name:17s} {v['hits']}/{v['total']}")
print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS | NATIVE PKV: PASS")
print("JSONL:",LOG);print("SUMMARY:",OUT);print("="*112)
