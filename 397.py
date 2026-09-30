# TEST 397 — CROSS-MODEL FUNCTIONAL K/V ALIGNMENT × ATTENTION READOUT × SOURCE SWAP
import os,sys,gc,json,time,random,hashlib,subprocess,importlib.util,unicodedata,math
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
ROOT=Path("/content/AKBASCORE_TEST397")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST397")
ROOT.mkdir(parents=True,exist_ok=True);LOG=ROOT/"TEST397.jsonl";OUT=ROOT/"TEST397_SUMMARY.json";LOG.write_text("",encoding="utf-8")
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
"FACT":[("WHO","Who planted the Turkish flag at the base of the Golden Gate Bridge?"),
("WHAT","What did Mustafa Akbaş plant at the base of the Golden Gate Bridge?"),
("WHERE","Where did Mustafa Akbaş plant the Turkish flag?")],
"CF":[("WHO","Who planted the Turkish flag at the base of the Golden Gate Bridge?"),
("WHAT","What did Leyla Demir plant at the base of the Golden Gate Bridge?"),
("WHERE","Where did Leyla Demir plant the Turkish flag?")]}
def rec(x):
 with LOG.open("a",encoding="utf-8")as f:f.write(json.dumps(x,ensure_ascii=False,default=str)+"\n")
def clean():gc.collect();torch.cuda.empty_cache()
def norm(s):
 s=unicodedata.normalize("NFKD",str(s)).casefold()
 return " ".join("".join(c if c.isalnum()else" "for c in s if not unicodedata.combining(c)).split())
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
 max_new_tokens=40,do_sample=False,repetition_penalty=1.0,use_cache=True,
 pad_token_id=pad(tok),eos_token_id=eos)
 return tok.decode(y[0,x.shape[1]:],skip_special_tokens=True).strip()
def label_answer(s):
 z=norm(s.split("\n",1)[0]);mustafa="mustafa akbas"in z;leyla="leyla demir"in z
 neg=any(x in z for x in("no record","no evidence","did not","didn t","never","not true","fictional","cannot verify","can t verify","no such","didn","isn t","false claim","no indication","not aware"))
 if neg:return "NEGATED"
 if mustafa and leyla:return "BOTH"
 if mustafa:return "MUSTAFA"
 if leyla:return "LEYLA"
 return "OTHER"
@torch.inference_mode()
def query_bank(tok,model,layers,s):
 seq=[pad(tok)]+ids(tok,s+SEP);x=torch.tensor([seq],device=DEV)
 out=model(input_ids=x,use_cache=False,output_hidden_states=True,return_dict=True)
 T=len(seq);pos=torch.arange(T,device=DEV)[None]
 cr,sr=model.model.rotary_emb(out.hidden_states[0],pos)
 bank=[]
 for L,layer in enumerate(layers):
  z=layer.input_layernorm(out.hidden_states[L]);q=layer.self_attn.q_proj(z)
  q=q.reshape(1,T,28,128).transpose(1,2)
  qr,_=qwen_rope(q,q,cr,sr,unsqueeze_dim=1)
  bank.append(qr[0,:,1:,:].float().cpu().contiguous())
 del out
 return {"T":T,"Q":bank}
def functional(raw,qbank,L,exclude_sink=True):
 T=raw["K"][L].shape[0]
 if qbank["T"]!=T:raise RuntimeError("Functional query/cache length mismatch")
 q=qbank["Q"][L].float().reshape(28,T-1,128)
 k=raw["K"][L].float().reshape(T,4,128).permute(1,0,2)
 v=raw["V"][L].float().reshape(T,4,128).permute(1,0,2)
 p=torch.arange(T)[None]
 cosr,sinr=ROPE_CACHE[T]
 kr,_=qwen_rope(k[None],k[None],cosr,sinr,unsqueeze_dim=1)
 kr=kr[0].float().cpu().repeat_interleave(7,dim=0)
 vv=v.repeat_interleave(7,dim=0)
 scores=torch.matmul(q,kr.transpose(1,2))/math.sqrt(128)
 mask=torch.arange(T)[None,None,:]<=torch.arange(1,T)[None,:,None]
 scores=scores.masked_fill(~mask,-1e9)
 att=F.softmax(scores,dim=-1)
 if exclude_sink:
  att=att[:,:,1:]
  att=att/att.sum(-1,keepdim=True).clamp_min(1e-12)
  vv=vv[:,1:]
 read=torch.matmul(att,vv)
 return att,read
def functional_compare(ref,trial,qbank,L):
 ar,vr=functional(ref,qbank,L);at,vt=functional(trial,qbank,L)
 eps=1e-9
 kl=float((ar.clamp_min(eps)*(ar.clamp_min(eps).log()-at.clamp_min(eps).log())).sum(-1).mean())
 js=float(.5*(ar.clamp_min(eps)*(ar.clamp_min(eps).log()-((ar+at)*.5).clamp_min(eps).log())).sum(-1).mean()+.5*(at.clamp_min(eps)*(at.clamp_min(eps).log()-((ar+at)*.5).clamp_min(eps).log())).sum(-1).mean())
 ac=float(F.cosine_similarity(ar.reshape(28,-1),at.reshape(28,-1),dim=-1,eps=1e-12).mean())
 vc=float(F.cosine_similarity(vr.reshape(28,-1),vt.reshape(28,-1),dim=-1,eps=1e-12).mean())
 rel=float((vt-vr).norm()/vr.norm().clamp_min(1e-12))
 return {"attention_cos":ac,"attention_KL":kl,"attention_JS":js,"readout_cos":vc,"readout_rel_error":rel}
def meanrows(rows):
 return {k:sum(x[k]for x in rows)/len(rows)for k in rows[0]}
def pair_separation(a,b):
 return {k:{"cos":sum(cos(a[k][L],b[k][L])for L in range(28))/28,
 "rel_delta":sum(float((a[k][L]-b[k][L]).norm()/a[k][L].norm().clamp_min(1e-12))for L in range(28))/28}for k in("K","V")}
def functional_delta(refa,refb,testa,testb,bank,L):
 ar,vr=functional(refa,bank,L);br,wr=functional(refb,bank,L)
 at,vt=functional(testa,bank,L);bt,wt=functional(testb,bank,L)
 return {"attention_delta_cos":cos(bt-at,br-ar),"readout_delta_cos":cos(wt-vt,wr-vr),
 "attention_delta_norm_ratio":float((bt-at).norm()/(br-ar).norm().clamp_min(1e-12)),
 "readout_delta_norm_ratio":float((wt-vt).norm()/(wr-vr).norm().clamp_min(1e-12))}
print("="*112,"\nTEST 397 — CROSS-MODEL FUNCTIONAL K/V ALIGNMENT × ATTENTION READOUT × SOURCE SWAP\n"+"="*112)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/9] MISTRAL — SOURCE EXTRACTION")
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
print("[2/9] QWEN — TARGET EXTRACTION")
qt,qm,ql=load(QID,(28,3584,28,4,128))
qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight]
Q0=sha(qp);QA=[];QB=[];sink=None
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 qa=forge(qt,qm,ql,a);qb=forge(qt,qm,ql,b);QA.append(qa);QB.append(qb)
 if sink is None:sink={k:[qa[k][L][:1].clone()for L in range(28)]for k in("K","V")}
 print(f" QWEN {i+1}/32",end="\r")
QF={k:forge(qt,qm,ql,s)for k,s in(("FACT",FACT),("CF",CF))}
SUPPORT_TEXT=NEUTRAL[31];QN=forge(qt,qm,ql,SUPPORT_TEXT)
T=QF["FACT"]["T"]
assert T==QF["CF"]["T"]==19
print("\n TARGET LENGTH:",T,"| NEUTRAL:",QN["T"])
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
 GEOM[name]={"validation":{k:sum(x[k]for x in val)/8 for k in("K","V")},
 "final":delta_score(MF["FACT"],MF["CF"],QF["FACT"],QF["CF"],B,D,gain)}
 print(f" {name:7s} VAL={GEOM[name]['validation']} FINAL={GEOM[name]['final']}")
print(" TRAIN R² BASE:",R2B,"DELTA:",R2D)
rec({"group":"GEOMETRY","data":GEOM,"train_R2":{"BASE":R2B,"DELTA":R2D}})
del ST,SD;clean()
print("[4/9] EXACT-LENGTH K/V BRIDGES")
RAW={"FACT_NATIVE":native_resize(QF["FACT"],T),"CF_NATIVE":native_resize(QF["CF"],T),
 "NEUTRAL":native_resize(QN,T)}
for source in("FACT","CF"):
 for gain,name in((0.,"BASE"),(.5,"HYBRID"),(1.,"DELTA")):
  RAW[source+"_"+name]=predict(MF[source],B,D,T-1,sink,gain)
for name,r in RAW.items():
 assert all(x.shape==(T,512)for k in("K","V")for x in r[k]),name
print(" BRIDGE VS FACT NATIVE:",distance(RAW["FACT_HYBRID"],RAW["FACT_NATIVE"]))
print(" SOURCE FACT↔CF:",pair_separation(RAW["FACT_HYBRID"],RAW["CF_HYBRID"]))
print(" NATIVE FACT↔CF:",pair_separation(RAW["FACT_NATIVE"],RAW["CF_NATIVE"]))
print("[5/9] NATIVE CACHE RECONSTRUCTION AUDIT")
NATIVE=install(qm,RAW["FACT_NATIVE"])
nc=qm(input_ids=torch.tensor([[pad(qt)]+ids(qt,FACT+SEP)],device=DEV),use_cache=True).past_key_values
NK=max(float((NATIVE[0][L].float()-kvget(nc,L)[0].float()).norm()/kvget(nc,L)[0].float().norm().clamp_min(1e-12))for L in range(28))
NV=max(float((NATIVE[1][L].float()-kvget(nc,L)[1].float()).norm()/kvget(nc,L)[1].float().norm().clamp_min(1e-12))for L in range(28))
print(" NATIVE PKV ERROR K:",NK,"V:",NV)
if max(NK,NV)>.05:raise RuntimeError("Native cache mismatch")
del nc,NATIVE;clean()
print("[6/9] FUNCTIONAL Q BANK — NATIVE QWEN QUERIES")
QBANK={"FACT":query_bank(qt,qm,ql,FACT),"CF":query_bank(qt,qm,ql,CF)}
ROPE_CACHE={}
for n in sorted({x["T"]for x in QBANK.values()}):
 p=torch.arange(n,device=DEV)[None]
 cr,sr=qm.model.rotary_emb(torch.zeros((1,n,3584),device=DEV,dtype=DTYPE),p)
 ROPE_CACHE[n]=(cr.cpu(),sr.cpu())
print(" QUERY LENGTHS:",{k:v["T"]for k,v in QBANK.items()})
print(" Q SHAPE:",QBANK["FACT"]["Q"][0].shape,"| ROPE:",ROPE_CACHE[T][0].shape)
print("[7/9] ATTENTION FUNCTIONAL ALIGNMENT — ALL 28 LAYERS")
FUNCTIONAL={};ORDER=["FACT_NATIVE","CF_NATIVE","NEUTRAL","FACT_BASE","FACT_HYBRID","FACT_DELTA","CF_BASE","CF_HYBRID","CF_DELTA"]
for context in("FACT","CF"):
 ref=RAW[context+"_NATIVE"];bank=QBANK[context];FUNCTIONAL[context]={}
 for name in ORDER:
  layers=[functional_compare(ref,RAW[name],bank,L)for L in range(28)]
  FUNCTIONAL[context][name]={"mean":meanrows(layers),"layers":layers}
  rec({"group":"FUNCTIONAL","context":context,"arm":name,"mean":FUNCTIONAL[context][name]["mean"],"layers":layers})
  m=FUNCTIONAL[context][name]["mean"]
  print(f" {context:4s} {name:15s} ATT_COS={m['attention_cos']:+.5f} ATT_KL={m['attention_KL']:.5f} JS={m['attention_JS']:.5f} READ_COS={m['readout_cos']:+.5f} READ_REL={m['readout_rel_error']:.5f}")
 print(" CONTEXT:",context,"| native reference must approach ATT_COS=1, READ_COS=1, KL=0")
print("[8/9] FACT↔CF FUNCTIONAL DELTA + LAYER LOCALIZATION")
SWAPS={}
for context in("FACT","CF"):
 bank=QBANK[context];SWAPS[context]={}
 for name in("BASE","HYBRID","DELTA"):
  layers=[functional_delta(RAW["FACT_NATIVE"],RAW["CF_NATIVE"],RAW["FACT_"+name],RAW["CF_"+name],bank,L)for L in range(28)]
  SWAPS[context][name]={"mean":meanrows(layers),"layers":layers}
  rec({"group":"SOURCE_SWAP","context":context,"map":name,"mean":SWAPS[context][name]["mean"],"layers":layers})
  m=SWAPS[context][name]
  print(f"\n CONTEXT={context} MAP={name} | DELTA ATT COS={m['mean']['attention_delta_cos']:+.6f} READ COS={m['mean']['readout_delta_cos']:+.6f}")
  print(" LAYER | ATT_DELTA_COS | READ_DELTA_COS | ATT_NORM_RATIO | READ_NORM_RATIO")
  for L,r in enumerate(layers):
   print(f" {L:02d}    | {r['attention_delta_cos']:+.6f}      | {r['readout_delta_cos']:+.6f}       | {r['attention_delta_norm_ratio']:.6f}       | {r['readout_delta_norm_ratio']:.6f}")
print("[9/9] SOURCE-ONLY BEHAVIOR + NATIVE CONTROLS + INTEGRITY")
PKV={k:install(qm,RAW[k])for k in ORDER}
ANS=[]
for context in("FACT","CF"):
 for typ,q in QUESTIONS[context]:
  print("\n CONTEXT:",context,"QUESTION:",typ,q)
  for name in ["VANILLA"]+ORDER:
   v=gen(qt,qm,q,None if name=="VANILLA"else PKV[name])
   row={"group":"BEHAVIOR","context":context,"question":typ,"arm":name,
    "first":v.split("\n",1)[0].strip(),"full":v,"who_label":label_answer(v)}
   ANS.append(row);rec(row)
   print(f" {name:15s} [{row['who_label']:8s}] {row['first'][:170]}")
Q1=sha(qp)
if Q0!=Q1:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Model not frozen")
SOURCE_ONLY=[r for r in ANS if r["question"]=="WHO"and r["arm"]in("FACT_BASE","FACT_HYBRID","FACT_DELTA","CF_BASE","CF_HYBRID","CF_DELTA")]
REPORT={"test":397,"title":"Cross-Model Functional K/V Alignment, Attention Readout and Source Swap",
 "source_model":MID,"target_model":QID,"seed":SEED,"train_pairs":24,"validation_pairs":8,
 "source_lengths":{k:MF[k]["T"]for k in("FACT","CF")},"target_length":T,
 "layer_map":LMAP,"head_map":"ADJACENT_MEAN_8_TO_4","token_map":"LINEAR_INTERPOLATION",
 "train_R2":{"BASE":R2B,"DELTA":R2D},"geometry":GEOM,
 "native_cache_error":{"K":NK,"V":NV},
 "cache_separation":{"native":pair_separation(RAW["FACT_NATIVE"],RAW["CF_NATIVE"]),
 "BASE":pair_separation(RAW["FACT_BASE"],RAW["CF_BASE"]),
 "HYBRID":pair_separation(RAW["FACT_HYBRID"],RAW["CF_HYBRID"]),
 "DELTA":pair_separation(RAW["FACT_DELTA"],RAW["CF_DELTA"])},
 "functional":FUNCTIONAL,"source_swap":SWAPS,"answers":ANS,"source_only_who":SOURCE_ONLY,
 "integrity":{"mistral_before_after":M0,"qwen_before":Q0,"qwen_after":Q1},
 "interpretation_rules":[
 "Attention comparison uses frozen Qwen queries from natural FACT or CF forward passes, not queries generated by a source-only forward pass.",
 "Attention weights are computed from rotary-positioned K and Q; V readout is the attention-weighted V tensor before output projection.",
 "The first cache position is excluded from attention metrics and the remaining attention probabilities are renormalized.",
 "The source-only bridge contains a neutral first-position sink but no natural Qwen target-fact cache.",
 "FACT and CF questions are evaluated under both native query contexts to expose query-context dependence.",
 "A high attention cosine or readout cosine alone does not prove behavioral knowledge transfer.",
 "Source-swap delta alignment tests whether the bridge preserves the native Qwen FACT-to-CF functional direction.",
 "Native Qwen query states contain information from their respective factual prompts; functional alignment under those queries cannot independently establish source-only transfer.",
 "Behavioral outputs are preserved verbatim; the WHO label is only a coarse diagnostic and must not be interpreted as an independently verified factual success.",
 "Natural cache reconstruction validates the cache installation pathway, not the cross-model bridge.",
 "Neutral support sentence is among held-out validation examples, not among ridge training examples.",
 "This is a diagnostic experiment: no functional loss is optimized and no new bridge parameters are trained from target facts.",
 "Parameter sentinels sample selected weights rather than hashing every parameter."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("\n"+"="*112)
print("TEST 397 — COMPLETE")
print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS | NATIVE PKV: PASS")
print("SOURCE-ONLY WHO READOUTS:",len(SOURCE_ONLY))
print("JSONL:",LOG)
print("SUMMARY:",OUT)
print("="*112)
