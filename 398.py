# TEST 398 — FUNCTIONAL K/V READOUT BRIDGE × HELD-OUT VALIDATION × SOURCE-ONLY TRANSFER
import os,sys,gc,json,time,random,hashlib,subprocess,importlib.util,re,unicodedata,math
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
ROOT=Path("/content/AKBASCORE_TEST398")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST398")
ROOT.mkdir(parents=True,exist_ok=True);LOG=ROOT/"TEST398.jsonl";OUT=ROOT/"TEST398_SUMMARY.json";LOG.write_text("",encoding="utf-8")
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
QUESTIONS={
"FACT":[("WHO","Who planted the Turkish flag at the base of the Golden Gate Bridge?"),("WHAT","What did Mustafa Akbaş plant at the base of the Golden Gate Bridge?"),("WHERE","Where did Mustafa Akbaş plant the Turkish flag?")],
"CF":[("WHO","Who planted the Turkish flag at the base of the Golden Gate Bridge?"),("WHAT","What did Leyla Demir plant at the base of the Golden Gate Bridge?"),("WHERE","Where did Leyla Demir plant the Turkish flag?")]}
def rec(x):
 with LOG.open("a",encoding="utf-8")as f:f.write(json.dumps(x,ensure_ascii=False,default=str)+"\n")
def clean():gc.collect();torch.cuda.empty_cache()
def norm(s):
 s=unicodedata.normalize("NFKD",str(s).casefold())
 return " ".join("".join(c if c.isalnum()and not unicodedata.combining(c)else" "for c in s).split())
def label(s):
 z=norm(s.split("\n",1)[0]);a="mustafa akbas"in z;b="leyla demir"in z
 return "BOTH"if a and b else"MUSTAFA"if a else"LEYLA"if b else"OTHER"
def score(s,typ,source):
 z=norm(s.split("\n",1)[0]);who="mustafa akbas"if source=="FACT"else"leyla demir"
 bad=("no information","not provided","does not mention","did not","didnt","was not","no evidence","cannot determine","unknown","unrelated","not related","does not contain","not enough information","no details","not possible","does not correspond","do not match","not match","no record")
 if any(x in z for x in bad):return 0
 if typ=="WHO":return int(label(s)==("MUSTAFA"if source=="FACT"else"LEYLA")and who in z and("planted"in z or z==who))
 if typ=="WHAT":return int("turkish flag"in z or"turkish national flag"in z)
 return int("base of the golden gate bridge"in z)
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
def forge(tok,model,layers,s,with_q=False):
 seq=[pad(tok)]+ids(tok,s+SEP);T=len(seq)
 out=model(input_ids=torch.tensor([seq],device=DEV),use_cache=False,output_hidden_states=True,return_dict=True)
 K=[];V=[];Q=[]
 if with_q:
  pos=torch.arange(T,device=DEV)[None];cr,sr=model.model.rotary_emb(out.hidden_states[0],pos)
 for L,layer in enumerate(layers):
  z=layer.input_layernorm(out.hidden_states[L][0]);a=layer.self_attn
  K.append(a.k_proj(z).float().cpu().contiguous());V.append(a.v_proj(z).float().cpu().contiguous())
  if with_q:
   q=a.q_proj(z).reshape(1,T,28,128).transpose(1,2)
   qr,_=qwen_rope(q,q,cr,sr,unsqueeze_dim=1)
   Q.append(qr[0].float().cpu().contiguous())
 del out
 return {"T":T,"K":K,"V":V,"Q":Q}
LMAP=[min(31,max(0,round((L+.5)*32/28-.5)))for L in range(28)]
def resize(x,n):
 if x.shape[0]==n:return x.contiguous()
 return F.interpolate(x.T.unsqueeze(0),size=n,mode="linear",align_corners=False)[0].T.contiguous()
def src(f,L,n,kind):
 x=f[kind][LMAP[L]][1:].reshape(-1,8,128).reshape(-1,4,2,128).mean(2)
 return resize(x.reshape(-1,512),n).reshape(n,4,128)
def tgt(f,L,kind):return f[kind][L][1:].reshape(-1,4,128)
def stats():
 return {k:[[{name:torch.zeros(sh,dtype=torch.float64)for name,sh in(("sx",(128,)),("sy",(128,)),("xx",(128,128)),("xy",(128,128)),("yy",()))}|{"n":0}for h in range(4)]for L in range(28)]for k in("K","V")}
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
 B={k:[]for k in st};fit=[]
 for kind in st:
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
ROPE={}
def rope_cpu(T):
 if T not in ROPE:
  pos=torch.arange(T,dtype=torch.float32)
  inv=1.0/(1000000.0**(torch.arange(0,128,2,dtype=torch.float32)/128))
  ang=torch.outer(pos,inv);ang=torch.cat([ang,ang],dim=-1)
  ROPE[T]=(ang.cos(),ang.sin())
 return ROPE[T]
def rotate_k(raw,L):
 T=raw["K"][L].shape[0];co,si=rope_cpu(T)
 k=raw["K"][L].reshape(T,4,128).float()
 return k*co[:,None,:]+torch.cat([-k[...,64:],k[...,:64]],dim=-1)*si[:,None,:]
def readout(raw,q,L,last=5):
 T=raw["K"][L].shape[0]
 if q["T"]!=T:raise RuntimeError(f"Readout length mismatch: {q['T']} vs {T}")
 k=rotate_k(raw,L);v=raw["V"][L].reshape(T,4,128).float()
 query=q["Q"][L].reshape(4,7,T,128).float()
 p=torch.arange(max(1,T-last),T)
 logits=torch.einsum("ghtd,sgd->ghts",query[:,:,p,:],k)/math.sqrt(128)
 logits=logits.masked_fill(torch.arange(T)[None,None,None,:]>p[None,None,:,None],-1e9)
 att=logits.softmax(dim=-1)
 out=torch.einsum("ghts,sgd->ghtd",att,v)
 return out.permute(2,0,1,3).reshape(-1,4,7,128).mean(2).contiguous()
def fstats():
 return [[{name:torch.zeros(sh,dtype=torch.float64)for name,sh in(("sx",(128,)),("sy",(128,)),("xx",(128,128)),("xy",(128,128)),("yy",()))}|{"n":0}for h in range(4)]for L in range(28)]
def fsolve(st):
 out=[];fit=[]
 for L in range(28):
  group=[]
  for h in range(4):
   s=st[L][h];n=s["n"];mx=s["sx"]/n;my=s["sy"]/n
   xx=s["xx"]-n*torch.outer(mx,mx);xy=s["xy"]-n*torch.outer(mx,my)
   alpha=max(1e-8,float(xx.trace()/128)*.05)
   W=torch.linalg.solve(xx+alpha*torch.eye(128,dtype=torch.float64),xy)
   err=float((s["yy"]-n*(my@my)-2*(W*xy).sum()+(W*(xx@W)).sum()).clamp_min(0))
   var=float((s["yy"]-n*(my@my)).clamp_min(1e-12))
   fit.append(1-err/var);group.append((W.float(),mx.float(),my.float()))
  out.append(group)
 return out,sum(fit)/len(fit)
def fapply(raw,W):
 out={"K":[x.clone()for x in raw["K"]],"V":[]}
 for L in range(28):
  x=raw["V"][L].reshape(-1,4,128).float();y=[]
  for h in range(4):
   w,mx,my=W[L][h];y.append((x[:,h]-mx)@w+my)
  out["V"].append(torch.stack(y,1).reshape(-1,512).contiguous())
 return out
def fmetric(raw,q,native):
 c=[];e=[]
 for L in range(28):
  a=readout(raw,q,L);b=readout(native,q,L)
  c.append(cos(a,b));e.append(float((a-b).norm()/b.norm().clamp_min(1e-12)))
 return {"read_cos":sum(c)/28,"read_rel_error":sum(e)/28}
def fdelta(a,b,qa,qb,na,nb):
 vals=[]
 for L in range(28):
  source=readout(b,qb,L)-readout(a,qa,L)
  native=readout(nb,qb,L)-readout(na,qa,L)
  vals.append(cos(source,native))
 return sum(vals)/28
GAINS={"BASE":0.,"HYBRID":.5,"DELTA":1.}
print("="*112,"\nTEST 398 — FUNCTIONAL K/V READOUT BRIDGE × HELD-OUT VALIDATION × SOURCE-ONLY TRANSFER\n"+"="*112)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/9] MISTRAL — FROZEN SOURCE")
mt,mm,ml=load(MID,(32,4096,32,8,128))
mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight]
M0=sha(mp);MA=[];MB=[]
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 MA.append(forge(mt,mm,ml,a));MB.append(forge(mt,mm,ml,b))
 print(f" MISTRAL {i+1}/32",end="\r")
MF={k:forge(mt,mm,ml,s)for k,s in(("FACT",FACT),("CF",CF))}
print("\n SOURCE LENGTHS:",{k:v["T"]for k,v in MF.items()})
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/9] QWEN — FROZEN TARGET + NATIVE QUERIES")
qt,qm,ql=load(QID,(28,3584,28,4,128))
qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight]
Q0=sha(qp);QA=[];QB=[];sink=None
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 qa=forge(qt,qm,ql,a,True);qb=forge(qt,qm,ql,b,True)
 QA.append(qa);QB.append(qb)
 if sink is None:sink={k:[qa[k][L][:1].clone()for L in range(28)]for k in("K","V")}
 print(f" QWEN {i+1}/32",end="\r")
QF={k:forge(qt,qm,ql,s,True)for k,s in(("FACT",FACT),("CF",CF))}
T=QF["FACT"]["T"]
assert T==QF["CF"]["T"]==19
assert all(len(q["Q"])==28 and q["Q"][0].shape==(28,q["T"],128)for q in QA+QB+list(QF.values()))
print("\n TARGET LENGTH:",T,"| TRAIN:24 | VALIDATION:8 | FINAL:FACT/CF")
print("[3/9] UNCHANGED FP64 BASE + DELTA RIDGE")
ST=stats();SD=stats()
for i in range(24):
 for kind in("K","V"):
  for L in range(28):
   for m,q in((MA[i],QA[i]),(MB[i],QB[i])):
    n=q["T"]-1;add(ST[kind][L],src(m,L,n,kind),tgt(q,L,kind))
   X,Y=pair(MA[i],MB[i],QA[i],QB[i],L,kind);add(SD[kind][L],X,Y)
B,R2B=solve(ST);D,R2D=solve(SD)
GEOM={}
for gain,name in((0.,"BASE"),(.5,"HYBRID"),(1.,"DELTA")):
 val=[delta_score(MA[i],MB[i],QA[i],QB[i],B,D,gain)for i in range(24,32)]
 GEOM[name]={"validation":{k:sum(x[k]for x in val)/8 for k in("K","V")},"final":delta_score(MF["FACT"],MF["CF"],QF["FACT"],QF["CF"],B,D,gain)}
 print(f" {name:7s} VAL={GEOM[name]['validation']} FINAL={GEOM[name]['final']}")
print(f" TRAIN R² BASE={R2B:.6f} DELTA={R2D:.6f}")
rec({"group":"GEOMETRY","train_R2":{"BASE":R2B,"DELTA":R2D},"data":GEOM})
del ST,SD;clean()
print("[4/9] FUNCTIONAL READOUT FIT — 24 NEUTRAL PAIRS ONLY")
FS={name:fstats()for name in GAINS}
for i in range(24):
 for m,q in((MA[i],QA[i]),(MB[i],QB[i])):
  native=native_resize(q,q["T"])
  bridges={name:predict(m,B,D,q["T"]-1,sink,gain)for name,gain in GAINS.items()}
  for L in range(28):
   target=readout(native,q,L)
   for name in GAINS:add(FS[name][L],readout(bridges[name],q,L),target)
 print(f" FUNCTIONAL TRAIN {i+1}/24",end="\r")
FW={};FR2={}
for name in GAINS:
 FW[name],FR2[name]=fsolve(FS[name])
 print(f"\n {name:7s} FUNCTIONAL TRAIN R²={FR2[name]:.6f}")
del FS;clean()
print("[5/9] HELD-OUT NEUTRAL VALIDATION — NO REFITTING")
VAL={}
for name,gain in GAINS.items():
 before=[];after=[]
 for i in range(24,32):
  for m,q in((MA[i],QA[i]),(MB[i],QB[i])):
   native=native_resize(q,q["T"]);raw=predict(m,B,D,q["T"]-1,sink,gain)
   before.append(fmetric(raw,q,native))
   after.append(fmetric(fapply(raw,FW[name]),q,native))
 VAL[name]={"before":{k:sum(x[k]for x in before)/len(before)for k in("read_cos","read_rel_error")},
 "after":{k:sum(x[k]for x in after)/len(after)for k in("read_cos","read_rel_error")}}
 print(f" {name:7s} BEFORE={VAL[name]['before']} AFTER={VAL[name]['after']}")
rec({"group":"VALIDATION","data":VAL})
print("[6/9] SOURCE-ONLY FACT/CF — FINAL SENTENCES EXCLUDED FROM FIT")
RAW={};NATIVE={source:native_resize(QF[source],T)for source in("FACT","CF")}
for source in("FACT","CF"):
 for name,gain in GAINS.items():
  raw=predict(MF[source],B,D,T-1,sink,gain)
  RAW[source+"_"+name]=raw
  RAW[source+"_"+name+"_FUNC"]=fapply(raw,FW[name])
print(" SOURCE LENGTHS:",MF["FACT"]["T"],MF["CF"]["T"],"| TARGET:",T)
print("[7/9] FINAL FUNCTIONAL X-RAY + SOURCE-SWAP DIRECTION")
FINAL={};DELTA={}
for source in("FACT","CF"):
 FINAL[source]={}
 for name in GAINS:
  a=RAW[source+"_"+name];b=RAW[source+"_"+name+"_FUNC"]
  FINAL[source][name]={"before":fmetric(a,QF[source],NATIVE[source]),"after":fmetric(b,QF[source],NATIVE[source])}
  print(f" {source:4s} {name:7s} BEFORE={FINAL[source][name]['before']} AFTER={FINAL[source][name]['after']}")
for name in GAINS:
 DELTA[name]={"before":fdelta(RAW["FACT_"+name],RAW["CF_"+name],QF["FACT"],QF["CF"],NATIVE["FACT"],NATIVE["CF"]),
 "after":fdelta(RAW["FACT_"+name+"_FUNC"],RAW["CF_"+name+"_FUNC"],QF["FACT"],QF["CF"],NATIVE["FACT"],NATIVE["CF"])}
 print(f" FACT↔CF {name:7s} DELTA READ COS={DELTA[name]}")
rec({"group":"FINAL_FUNCTIONAL","data":FINAL,"source_swap":DELTA})
print("[8/9] NATIVE CACHE AUDIT + SOURCE-ONLY GENERATION")
PKV={"FACT_NATIVE":install(qm,NATIVE["FACT"]),"CF_NATIVE":install(qm,NATIVE["CF"])}
for name,raw in RAW.items():PKV[name]=install(qm,raw)
nc=qm(input_ids=torch.tensor([[pad(qt)]+ids(qt,FACT+SEP)],device=DEV),use_cache=True).past_key_values
NK=max(float((PKV["FACT_NATIVE"][0][L].float()-kvget(nc,L)[0].float()).norm()/kvget(nc,L)[0].float().norm().clamp_min(1e-12))for L in range(28))
NV=max(float((PKV["FACT_NATIVE"][1][L].float()-kvget(nc,L)[1].float()).norm()/kvget(nc,L)[1].float().norm().clamp_min(1e-12))for L in range(28))
print(" NATIVE PKV ERROR K:",NK,"V:",NV)
if max(NK,NV)>.05:raise RuntimeError("Native cache mismatch")
del nc;clean()
ANS=[]
for source in("FACT","CF"):
 for typ,q in QUESTIONS[source]:
  print("\n CONTEXT:",source,"QUESTION:",typ,q)
  names=["VANILLA","FACT_NATIVE","CF_NATIVE"]+[source+"_"+name+s for name in GAINS for s in("","_FUNC")]
  for name in names:
   v=gen(qt,qm,q,None if name=="VANILLA"else PKV[name])
   row={"group":"READOUT","context":source,"question":typ,"arm":name,"label":label(v),"hit":score(v,typ,source),"full":v}
   ANS.append(row);rec(row)
   print(f" {name:23s} [{row['label']:7s}] hit={row['hit']} | {v.split(chr(10),1)[0][:150]}")
 clean()
print("[9/9] INTEGRITY + PROVENANCE SCORECARD")
Q1=sha(qp)
if Q0!=Q1:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Model not frozen")
SCORES={}
for source in("FACT","CF"):
 for name in GAINS:
  for suffix in("","_FUNC"):
   arm=source+"_"+name+suffix
   rr=[r for r in ANS if r["context"]==source and r["arm"]==arm]
   SCORES[arm]={typ:next(r["hit"]for r in rr if r["question"]==typ)for typ in("WHO","WHAT","WHERE")}
   print(f" {arm:23s} WHO={SCORES[arm]['WHO']} WHAT={SCORES[arm]['WHAT']} WHERE={SCORES[arm]['WHERE']}")
REPORT={"test":398,"title":"Neutral-Trained Functional Readout Bridge and Source-Only Transfer",
 "source_model":MID,"target_model":QID,"seed":SEED,"train_pairs":24,"validation_pairs":8,
 "source_tokens":{k:v["T"]for k,v in MF.items()},"target_tokens":T,"layer_map":LMAP,
 "train_R2":{"BASE":R2B,"DELTA":R2D},"geometry":GEOM,"functional_train_R2":FR2,
 "functional_validation":VAL,"final_functional":FINAL,"source_swap":DELTA,
 "answers":ANS,"scores":SCORES,"native_cache_error":{"K":NK,"V":NV},
 "integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1},
 "limitations":[
 "BASE and DELTA use the TEST396 FP64 ridge formulation without target-fact fitting.",
 "The functional V transformation is fitted only on 24 neutral training pairs using native Qwen queries.",
 "The eight validation pairs and FACT/CF sentences are excluded from all parameter fitting.",
 "Readout averages seven Qwen query heads per KV head; this is a diagnostic compression, not the full output-projected attention result.",
 "The functional transform modifies V only; source K and its attention distributions remain unchanged.",
 "Native FACT/CF queries are used for final diagnostics only, not source-only generation.",
 "The first-token sink is derived from neutral Qwen data.",
 "Mistral positions are linearly interpolated, not semantically aligned.",
 "The final source-swap metric compares different native query contexts and is not a fixed-query causal contrast.",
 "Training R² does not measure held-out transfer.",
 "Behavioral substring scores are approximate; full outputs require manual inspection.",
 "Parameter sentinels sample selected weights, not all parameters."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("\n"+"="*112+"\nTEST 398 — COMPLETE")
print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS | NATIVE PKV: PASS")
print("JSONL:",LOG);print("SUMMARY:",OUT);print("="*112)
