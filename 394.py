# TEST 394 — MULTI-ABLATION · TEST392 ENGINE
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
SEED=392;random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEV="cuda";DTYPE=torch.bfloat16;MAX_NEW=32;SEP="\n\n";FMT="QUESTION:\n{q}\n\nANSWER:"
MID="mistralai/Mistral-7B-Instruct-v0.3";QID="Qwen/Qwen2.5-7B-Instruct"
ROOT=Path("/content/AKBASCORE_TEST394")if os.path.isdir("/content")else Path("/tmp/AKBASCORE_TEST394")
ROOT.mkdir(parents=True,exist_ok=True);LOG=ROOT/"TEST394.jsonl";SUMMARY=ROOT/"TEST394_SUMMARY.json";LOG.write_text("",encoding="utf-8")
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
SWAP=[("Jonas Weber","Elena Fischer"),("Priya Nair","Daniel Brooks"),("The small boat","The large boat"),
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
QUESTIONS=[("WHO","Who planted the Turkish flag at the base of the Golden Gate Bridge?","Mustafa Akbaş","Leyla Demir"),
 ("WHAT","What did Mustafa Akbaş plant at the base of the Golden Gate Bridge?","Turkish flag",None),
 ("WHERE","Where did Mustafa Akbaş plant the Turkish flag?","base of the Golden Gate Bridge",None)]
assert all(FACT not in s and CF not in s and WRONG not in s for s in NEUTRAL+ALT)
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
 return {k:[[{name:torch.zeros(sh,dtype=torch.float64)for name,sh in
  (("sx",(128,)),("sy",(128,)),("xx",(128,128)),("xy",(128,128)),("yy",()))}|{"n":0}
  for h in range(4)]for L in range(28)]for k in("K","V")}
ST=stats();SD=stats();SN=stats()
def add(st,X,Y):
 X=X.double();Y=Y.double()
 for h in range(4):
  a=X[:,h];b=Y[:,h];s=st[h];n=a.shape[0]
  s["sx"]+=a.sum(0);s["sy"]+=b.sum(0);s["xx"]+=a.T@a;s["xy"]+=a.T@b;s["yy"]+=(b*b).sum();s["n"]+=n
def accumulate(m,q):
 n=q["T"]-1
 for kind in("K","V"):
  for L in range(28):add(ST[kind][L],src_heads(m,L,n,kind),tgt_heads(q,L,kind))
def delta_pair(ma,mb,qa,qb,L,kind):
 n=min(qa["T"],qb["T"])-1
 X=src_heads(mb,L,n,kind)-src_heads(ma,L,n,kind)
 ya=resize(tgt_heads(qa,L,kind).reshape(-1,512),n).reshape(n,4,128)
 yb=resize(tgt_heads(qb,L,kind).reshape(-1,512),n).reshape(n,4,128)
 return X,yb-ya
def accumulate_delta(ma,mb,qa,qb,st):
 for kind in("K","V"):
  for L in range(28):
   X,Y=delta_pair(ma,mb,qa,qb,L,kind);add(st[kind][L],X,Y)
def solve(st,mode):
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
    fit.append({"mode":mode,"kind":kind,"layer":L,"head":h,"R2_train":1-err/var,"alpha":alpha,"n":n})
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
def predict(m,B,D,n,sink,gain):
 if gain==0:return bridge(m,B,n,sink)
 out={}
 for kind in("K","V"):
  arr=[]
  for L in range(28):
   X=src_heads(m,L,n,kind);Y=[]
   for h in range(4):
    W,mx,my=B[kind][L][h];WD,_,_=D[kind][L][h]
    x=X[:,h];Y.append((x-mx)@W+my+gain*((x-mx)@(WD-W)))
   arr.append(torch.cat([sink[kind][L],torch.stack(Y,1).reshape(n,512)],0))
  out[kind]=arr
 return out["K"],out["V"],n+1
def cos(a,b):
 a=a.float().reshape(-1);b=b.float().reshape(-1)
 return float(F.cosine_similarity(a,b,dim=0,eps=1e-12))
def rel(a,b):return float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))
def vecstats(a,b):return {"cos":cos(a,b),"rel":rel(a,b),"norm_ratio":float(a.float().norm()/b.float().norm().clamp_min(1e-12))}
def softstats(a,b):
 p=a.float().softmax(-1);q=b.float().softmax(-1)
 return {"TV":float((p-q).abs().sum(-1).mean()/2),"top1_agreement":float((p.argmax(-1)==q.argmax(-1)).float().mean()),
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
   x=p[0 if kind=="K"else 1][L][1:].reshape(n,4,128);y=q[kind][L][1:].reshape(n,4,128)
   for h in range(4):
    a=x[:,h];b=y[:,h];z=vecstats(a,b)
    z.update({"layer":L,"head":h,"kind":kind,"centered_cos":centered(a,b),"token_cos":[cos(a[t],b[t])for t in range(n)]})
    rows.append(z)
 return rows
def delta_dna(ma,mb,qa,qb,B,D,gain):
 rows=[]
 for kind in("K","V"):
  for L in range(28):
   X,Y=delta_pair(ma,mb,qa,qb,L,kind)
   for h in range(4):
    W,_,_=B[kind][L][h];WD,_,_=D[kind][L][h]
    d=X[:,h]@((1-gain)*W+gain*WD);z=vecstats(d,Y[:,h]);z.update({"kind":kind,"layer":L,"head":h});rows.append(z)
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
 y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),past_key_values=cache,max_new_tokens=MAX_NEW,
  do_sample=False,repetition_penalty=1.0,use_cache=True,pad_token_id=pad(tok),eos_token_id=eos)
 return tok.decode(y[0,x.shape[1]:],skip_special_tokens=True).strip()
# TEST394: six source-head projections × two token resamplers × BASE/DELTA; no final-fact selection.
PROJ={"ADJ_MEAN":((0,1),(2,3),(4,5),(6,7)),"STRIDE_MEAN":((0,4),(1,5),(2,6),(3,7)),"EVEN":((0,),(2,),(4,),(6,)),"ODD":((1,),(3,),(5,),(7,)),"ADJ_DIFF":((0,1),(2,3),(4,5),(6,7)),"STRIDE_DIFF":((0,4),(1,5),(2,6),(3,7))}
RESAMPLES=("LINEAR","NEAREST")
def src_variant(f,L,n,kind,mode,resample):
 x=f[kind][LMAP[L]][1:].reshape(-1,8,128)
 y=[]
 for ix in PROJ[mode]:
  if "DIFF"in mode:y.append((x[:,ix[0]]-x[:,ix[1]])*.5)
  elif len(ix)==2:y.append((x[:,ix[0]]+x[:,ix[1]])*.5)
  else:y.append(x[:,ix[0]])
 x=torch.stack(y,1);t=x.shape[0]
 if t==n:return x.contiguous()
 if resample=="LINEAR":return resize(x.reshape(t,512),n).reshape(n,4,128)
 return F.interpolate(x.reshape(t,512).T.unsqueeze(0),size=n,mode="nearest")[0].T.reshape(n,4,128).contiguous()
def pair_variant(ma,mb,qa,qb,L,kind,mode,resample):
 n=min(qa["T"],qb["T"])-1
 X=src_variant(mb,L,n,kind,mode,resample)-src_variant(ma,L,n,kind,mode,resample)
 ya=resize(tgt_heads(qa,L,kind).reshape(-1,512),n).reshape(n,4,128)
 yb=resize(tgt_heads(qb,L,kind).reshape(-1,512),n).reshape(n,4,128)
 return X,yb-ya
def train_variant(mode,resample,MA,MB,QA,QB):
 st=stats();sd=stats()
 for i in range(24):
  for kind in("K","V"):
   for L in range(28):
    for m,q in((MA[i],QA[i]),(MB[i],QB[i])):
     n=q["T"]-1;add(st[kind][L],src_variant(m,L,n,kind,mode,resample),tgt_heads(q,L,kind))
    X,Y=pair_variant(MA[i],MB[i],QA[i],QB[i],L,kind,mode,resample);add(sd[kind][L],X,Y)
 b,fb=solve(st,"BASE");d,fd=solve(sd,"DELTA")
 return b,d,fb,fd
def variant_delta(ma,mb,qa,qb,b,d,mode,resample,gain):
 out={"K":[],"V":[]}
 for kind in("K","V"):
  for L in range(28):
   X,Y=pair_variant(ma,mb,qa,qb,L,kind,mode,resample);pred=[]
   for h in range(4):
    W,_,_=b[kind][L][h];WD,_,_=d[kind][L][h]
    pred.append(X[:,h]@((1-gain)*W+gain*WD))
   out[kind].append(cos(torch.stack(pred,1),Y))
 return {k:sum(out[k])/28 for k in("K","V")}
def variant_predict(m,b,d,n,sink,mode,resample,gain):
 out={}
 for kind in("K","V"):
  arr=[]
  for L in range(28):
   X=src_variant(m,L,n,kind,mode,resample);Y=[]
   for h in range(4):
    W,mx,my=b[kind][L][h];WD,_,_=d[kind][L][h]
    Y.append((X[:,h]-mx)@W+my+gain*((X[:,h]-mx)@(WD-W)))
   arr.append(torch.cat([sink[kind][L],torch.stack(Y,1).reshape(n,512)],0))
  out[kind]=arr
 return out["K"],out["V"],n+1
def source_retention(a,b,L,kind,mode):
 aa=a[kind][LMAP[L]][1:].reshape(-1,8,128);bb=b[kind][LMAP[L]][1:].reshape(-1,8,128)
 n=min(len(aa),len(bb));aa=resize(aa.reshape(len(aa),1024),n).reshape(n,8,128);bb=resize(bb.reshape(len(bb),1024),n).reshape(n,8,128)
 raw=bb-aa;proj=[]
 for ix in PROJ[mode]:
  if "DIFF"in mode:proj.append((raw[:,ix[0]]-raw[:,ix[1]])*.5)
  elif len(ix)==2:proj.append((raw[:,ix[0]]+raw[:,ix[1]])*.5)
  else:proj.append(raw[:,ix[0]])
 p=torch.stack(proj,1);scale=2 if len(PROJ[mode][0])==2 else 1
 return float((scale*p.square().sum()/raw.square().sum().clamp_min(1e-12)).clamp(0,1))
print("="*108,"\nTEST 394 — MULTI-ABLATION: HEAD PAIRING × TOKEN RESAMPLING × MAP TYPE\n"+"="*108)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/6] MISTRAL — FIXED 32 PAIRS + HELD-OUT FINAL")
mt,mm,ml=load(MID,(32,4096,32,8,128));mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight]
M0=sha(mp);MA=[];MB=[]
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 MA.append(forge(mt,mm,ml,a));MB.append(forge(mt,mm,ml,b));print(f" MISTRAL {i+1}/32",end="\r")
MF={k:forge(mt,mm,ml,s)for k,s in(("ORIGINAL",FACT),("COUNTERFACTUAL",CF))}
print("\n SOURCE TOKENS:",MF["ORIGINAL"]["T"],MF["COUNTERFACTUAL"]["T"])
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/6] QWEN — FIXED 32 PAIRS + FINAL + NATIVE CACHE")
qt,qm,ql=load(QID,(28,3584,28,4,128));qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight]
Q0=sha(qp);QA=[];QB=[];sink=None;ratios=[]
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 qa=forge(qt,qm,ql,a);qb=forge(qt,qm,ql,b);QA.append(qa);QB.append(qb)
 if sink is None:sink={k:[qa[k][L][:1].clone()for L in range(28)]for k in("K","V")}
 if i<24:ratios.extend([(qa["T"]-1)/(MA[i]["T"]-1),(qb["T"]-1)/(MB[i]["T"]-1)])
 print(f" QWEN {i+1}/32",end="\r")
QF={k:forge(qt,qm,ql,s)for k,s in(("ORIGINAL",FACT),("COUNTERFACTUAL",CF))}
RATIO=float(torch.tensor(ratios).median());print("\n TOKEN RATIO:",RATIO,"| TARGET TOKENS:",QF["ORIGINAL"]["T"],QF["COUNTERFACTUAL"]["T"])
print("[3/6] 12 INDEPENDENT SOURCE/TOKEN VARIANTS × BASE/DELTA/HYBRID")
RESULT=[];MAPS={};total=len(PROJ)*len(RESAMPLES)
for mode in PROJ:
 for resample in RESAMPLES:
  b,d,fb,fd=train_variant(mode,resample,MA,MB,QA,QB)
  val={g:[]for g in(0.,.5,1.)}
  for i in range(24,32):
   for g in val:val[g].append(variant_delta(MA[i],MB[i],QA[i],QB[i],b,d,mode,resample,g))
  av={str(g):{k:sum(r[k]for r in val[g])/8 for k in("K","V")}for g in val}
  # Final is measured after all fits; no model or variant selected using final data.
  fin={str(g):variant_delta(MF["ORIGINAL"],MF["COUNTERFACTUAL"],QF["ORIGINAL"],QF["COUNTERFACTUAL"],b,d,mode,resample,g)for g in val}
  retain={k:sum(source_retention(MF["ORIGINAL"],MF["COUNTERFACTUAL"],L,k,mode)for L in range(28))/28 for k in("K","V")}
  row={"head":mode,"token":resample,"validation":av,"final":fin,"retention":retain,"train_R2":{"BASE":sum(x["R2_train"]for x in fb)/224,"DELTA":sum(x["R2_train"]for x in fd)/224}}
  RESULT.append(row);MAPS[(mode,resample)]=(b,d)
  print(f" {len(RESULT):02d}/{total} {mode:12s} {resample:7s} | VAL HYB K={av['0.5']['K']:+.4f} V={av['0.5']['V']:+.4f} | FINAL HYB K={fin['0.5']['K']:+.4f} V={fin['0.5']['V']:+.4f} | RET K={retain['K']:.3f} V={retain['V']:.3f}")
  rec({"group":"ABLATION","row":row})
print("[4/6] LOCKED VALIDATION SELECTION — NO FINAL-FACT SELECTION")
# BASELINE and top-two by held-out validation HYBRID mean; no final-fact leakage into selection.
ranked=sorted(RESULT,key=lambda r:(r["validation"]["0.5"]["K"]+r["validation"]["0.5"]["V"])/2,reverse=True)
chosen=[("ADJ_MEAN","LINEAR")]
for r in ranked:
 key=(r["head"],r["token"])
 if key not in chosen:chosen.append(key)
 if len(chosen)==3:break
for key in chosen:
 r=next(x for x in RESULT if(x["head"],x["token"])==key)
 print(f" SELECT {key[0]:12s} {key[1]:7s} | VAL={r['validation']['0.5']} | FINAL={r['final']['0.5']}")
print("[5/6] SOURCE-BLIND READOUT — BASELINE + TWO VAL-SELECTED VARIANTS")
native=install(qm,QF["ORIGINAL"]["K"],QF["ORIGINAL"]["V"])
nc=qm(input_ids=torch.tensor([[pad(qt)]+ids(qt,FACT+SEP)],device=DEV),use_cache=True).past_key_values
NK=max(float((native[0][L].float()-kvget(nc,L)[0].float()).norm()/kvget(nc,L)[0].float().norm().clamp_min(1e-12))for L in range(28))
NV=max(float((native[1][L].float()-kvget(nc,L)[1].float()).norm()/kvget(nc,L)[1].float().norm().clamp_min(1e-12))for L in range(28))
print(" NATIVE CACHE ERROR:",NK,NV)
if max(NK,NV)>.05:raise RuntimeError("Native cache mismatch")
del nc;clean();PKV={"NATIVE":native};ARMS=[("VANILLA",None),("NATIVE","NATIVE")]
for mode,resample in chosen:
 b,d=MAPS[(mode,resample)];tag=mode+"_"+resample
 n=max(1,round((MF["ORIGINAL"]["T"]-1)*RATIO))
 for gain,gn in((0.,"BASE"),(.5,"HYBRID"),(1.,"DELTA")):
  p=variant_predict(MF["ORIGINAL"],b,d,n,sink,mode,resample,gain);name=tag+"_"+gn
  PKV[name]=install(qm,p[0],p[1]);ARMS.append((name,name))
 # Counterfactual controls on HYBRID only, using the same selected map.
 ncf=max(1,round((MF["COUNTERFACTUAL"]["T"]-1)*RATIO))
 p=variant_predict(MF["COUNTERFACTUAL"],b,d,ncf,sink,mode,resample,.5)
 name="CF_"+tag;PKV[name]=install(qm,p[0],p[1]);ARMS.append((name,name))
PKV["CF_NATIVE"]=install(qm,QF["COUNTERFACTUAL"]["K"],QF["COUNTERFACTUAL"]["V"])
ARMS.append(("CF_NATIVE","CF_NATIVE"));ANS=[]
for typ,q,expected,cfexpected in QUESTIONS:
 print(" QUESTION:",typ)
 for arm,key in ARMS:
  ans=gen(qt,qm,q,PKV[key]if key else None);first=ans.split("\n",1)[0].strip()
  score=hit(first,cfexpected)if arm.startswith("CF_")and typ=="WHO"else hit(first,expected)if not arm.startswith("CF_")else None
  row={"group":"READOUT","question":typ,"arm":arm,"first":first,"full":ans,"hit":score};ANS.append(row);rec(row)
  print(f" {arm:26s} | hit={score} | {first[:110]}")
 clean()
print("[6/6] INTEGRITY + REPORT")
Q1=sha(qp)
if Q0!=Q1:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Model not frozen")
SCORES={a:{"hits":sum(r["hit"]for r in ANS if r["arm"]==a and r["hit"]is not None),"total":sum(r["arm"]==a and r["hit"]is not None for r in ANS)}for a,_ in ARMS}
REPORT={"test":394,"title":"Head-Pooling × Token-Resampling × Mapping Ablation","source":MID,"target":QID,"seed":SEED,"train_pairs":24,"validation_pairs":8,"token_ratio":RATIO,"layer_map":LMAP,"ablation":RESULT,"selection":[list(k)for k in chosen],"answers":ANS,"scores":SCORES,"native_cache_error":{"K":NK,"V":NV},"integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1},"limitations":["Final fact is not used to fit or select maps; 8 validation pairs select the two alternatives.","Energy retention is not a measure of semantic information retention.","All source variants still reduce eight Mistral KV heads to four; full-rank eight-head recovery is not tested.","Token interpolation is positional, not semantic alignment.","Cache readout and semantic transfer are not equivalent to differential cosine.","Selected parameter sentinels do not hash every model parameter."]}
SUMMARY.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("\nFINAL SCORES:")
for k,v in SCORES.items():print(f" {k:26s} {v['hits']}/{v['total']}")
print("SENTINELS: PASS | NATIVE CACHE: PASS");print("JSONL:",LOG);print("SUMMARY:",SUMMARY);print("="*108)
