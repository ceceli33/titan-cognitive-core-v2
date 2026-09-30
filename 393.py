# TEST 393 — BRIDGE LOSS LOCALIZATION · TEST392 ENGINE · REPAIRED
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
ROOT=Path("/content/AKBASCORE_TEST393")if os.path.isdir("/content")else Path("/tmp/AKBASCORE_TEST393")
ROOT.mkdir(parents=True,exist_ok=True);LOG=ROOT/"TEST393.jsonl";SUMMARY=ROOT/"TEST393_SUMMARY.json";LOG.write_text("",encoding="utf-8")
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

# TARGETED X-RAY — ORIGINAL TEST392 MAPPING; MATCHED SOURCE TOKEN LENGTHS
def pool_parts(f,L,kind):
 x=f[kind][LMAP[L]][1:].reshape(-1,4,2,128).float()
 return x.mean(2),x
def bridge_loss_xray(ma,mb,qa,qb,B,BD,BN):
 rows=[];n=min(qa["T"],qb["T"])-1
 for L in range(28):
  for kind in("K","V"):
   a,ar=pool_parts(ma,L,kind);b,br=pool_parts(mb,L,kind)
   ns=min(ar.shape[0],br.shape[0])
   ar=resize(ar.reshape(ar.shape[0],1024),ns).reshape(ns,4,2,128)
   br=resize(br.reshape(br.shape[0],1024),ns).reshape(ns,4,2,128)
   a=ar.mean(2);b=br.mean(2)
   raw=br-ar;pool=b-a
   retained=float((2*pool.square().sum()/raw.square().sum().clamp_min(1e-12)).clamp(0,1))
   interp=resize(pool.reshape(-1,512),n).reshape(n,4,128)
   back=resize(interp.reshape(n,512),ns).reshape(ns,4,128)
   ya=resize(tgt_heads(qa,L,kind).reshape(-1,512),n).reshape(n,4,128)
   yb=resize(tgt_heads(qb,L,kind).reshape(-1,512),n).reshape(n,4,128)
   target=yb-ya
   row={"layer":L,"source_layer":LMAP[L],"kind":kind,"source_delta_norm":float(raw.norm()),
    "pooled_delta_norm":float(pool.norm()),"head_retained_energy_fraction":retained,
    "token_roundtrip_cos":cos(pool,back),"token_roundtrip_relative_error":rel(back,pool),
    "source_tokens":ns,"target_tokens":n,"target_delta_norm":float(target.norm()),"maps":{}}
   for name,D,g in(("BASE",BD,0.),("HYBRID",BD,.5),("DELTA",BD,1.),("NULL",BN,1.)):
    X=src_heads(mb,L,n,kind)-src_heads(ma,L,n,kind)
    out=[]
    for h in range(4):
     W,_,_=B[kind][L][h];WD,_,_=D[kind][L][h]
     out.append(X[:,h]@((1-g)*W+g*WD))
    pred=torch.stack(out,1)
    row["maps"][name]={"delta_cos":cos(pred,target),"delta_rel":rel(pred,target),
     "norm_ratio":float(pred.norm()/target.norm().clamp_min(1e-12)),
     "head_cos":[cos(pred[:,h],target[:,h])for h in range(4)]}
   rows.append(row)
 return rows
def print_loss(rows):
 print(" LAYER SRC | K-POOL V-POOL | K-TOK V-TOK | K-BASE V-BASE | K-HYB V-HYB | K-NULL V-NULL")
 for L in range(28):
  k=next(r for r in rows if r["layer"]==L and r["kind"]=="K")
  v=next(r for r in rows if r["layer"]==L and r["kind"]=="V")
  print(f" L{L:02d} M{k['source_layer']:02d} | {k['head_retained_energy_fraction']:.3f} {v['head_retained_energy_fraction']:.3f} | {k['token_roundtrip_cos']:+.3f} {v['token_roundtrip_cos']:+.3f} | {k['maps']['BASE']['delta_cos']:+.3f} {v['maps']['BASE']['delta_cos']:+.3f} | {k['maps']['HYBRID']['delta_cos']:+.3f} {v['maps']['HYBRID']['delta_cos']:+.3f} | {k['maps']['NULL']['delta_cos']:+.3f} {v['maps']['NULL']['delta_cos']:+.3f}")
 print(" POOL=retained differential energy; TOK=interpolation roundtrip cosine; BASE/HYB/NULL=Qwen-native delta cosine")
def native_trace(a,b,heads,tag):
 rows=[];n=min(a["T"],b["T"])-1
 for L in range(len(a["K"])):
  row={"model":tag,"layer":L}
  for kind in("K","V"):
   aa=resize(a[kind][L][1:].reshape(-1,heads*128),n).reshape(n,heads,128)
   bb=resize(b[kind][L][1:].reshape(-1,heads*128),n).reshape(n,heads,128)
   d=bb-aa
   row[kind]={"delta_norm":float(d.norm()),"delta_over_original":float(d.norm()/aa.norm().clamp_min(1e-12)),
    "head_delta_norms":[float(d[:,h].norm())for h in range(heads)]}
  rows.append(row)
 return rows
def print_native(rows,tag):
 print(f" {tag} | layer | K relative delta | V relative delta")
 for r in rows:print(f" {tag:7s} L{r['layer']:02d} | {r['K']['delta_over_original']:.5f} | {r['V']['delta_over_original']:.5f}")
def evaluate_delta(ma,mb,qa,qb,B,D,gain):
 rows=[];n=min(qa["T"],qb["T"])-1
 for kind in("K","V"):
  for L in range(28):
   X,Y=delta_pair(ma,mb,qa,qb,L,kind);out=[]
   for h in range(4):
    W,_,_=B[kind][L][h];WD,_,_=D[kind][L][h]
    out.append(X[:,h]@((1-gain)*W+gain*WD))
   rows.append({"kind":kind,"layer":L,"cos":cos(torch.stack(out,1),Y)})
 return {k:sum(r["cos"]for r in rows if r["kind"]==k)/28 for k in("K","V")}
MODES={"BASE":(None,0.),"HYBRID":("DELTA",.5),"DELTA":("DELTA",1.),"NULL":("NULL",1.)}

print("="*110,"\nTEST 393 — BRIDGE LOSS LOCALIZATION · TEST392 ENGINE\n"+"="*110)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/7] MISTRAL — NATIVE SOURCE + 32 MATCHED PAIRS")
mt,mm,ml=load(MID,(32,4096,32,8,128))
mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight]
M0=sha(mp);MA=[];MB=[]
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 MA.append(forge(mt,mm,ml,a));MB.append(forge(mt,mm,ml,b));print(f" MISTRAL PAIR {i+1}/32",end="\r")
MF={name:forge(mt,mm,ml,s)for name,s in(("ORIGINAL",FACT),("COUNTERFACTUAL",CF),("WRONG",WRONG))}
MTRACE=native_trace(MF["ORIGINAL"],MF["COUNTERFACTUAL"],8,"MISTRAL")
print("\n MISTRAL ORIGINAL/CF TOKENS:",MF["ORIGINAL"]["T"],MF["COUNTERFACTUAL"]["T"])
print_native(MTRACE,"MISTRAL");rec({"group":"MISTRAL_NATIVE_TRACE","rows":MTRACE})
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/7] QWEN — NATIVE TARGET + SAME CALIBRATION")
qt,qm,ql=load(QID,(28,3584,28,4,128))
qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight]
Q0=sha(qp);QA=[];QB=[];sink=None;ratios=[]
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 qa=forge(qt,qm,ql,a);qb=forge(qt,qm,ql,b);QA.append(qa);QB.append(qb)
 if sink is None:sink={k:[qa[k][L][:1].clone()for L in range(28)]for k in("K","V")}
 if i<24:
  accumulate(MA[i],qa);accumulate(MB[i],qb);accumulate_delta(MA[i],MB[i],qa,qb,SD)
  ratios.extend([(qa["T"]-1)/(MA[i]["T"]-1),(qb["T"]-1)/(MB[i]["T"]-1)])
 print(f" QWEN PAIR {i+1}/32",end="\r")
QF={name:forge(qt,qm,ql,s)for name,s in(("ORIGINAL",FACT),("COUNTERFACTUAL",CF),("WRONG",WRONG))}
RATIO=float(torch.tensor(ratios).median())
QTRACE=native_trace(QF["ORIGINAL"],QF["COUNTERFACTUAL"],4,"QWEN")
print("\n TRAIN TOKEN RATIO:",RATIO,"| QWEN ORIGINAL/CF TOKENS:",QF["ORIGINAL"]["T"],QF["COUNTERFACTUAL"]["T"])
print_native(QTRACE,"QWEN");rec({"group":"QWEN_NATIVE_TRACE","rows":QTRACE})
print("[3/7] MATCHED FP64 RIDGE + SHUFFLED NULL")
perm=list(range(24));random.shuffle(perm)
while any(perm[i]==i for i in range(24)):perm=perm[1:]+perm[:1]
for i in range(24):accumulate_delta(MA[i],MB[i],QA[perm[i]],QB[perm[i]],SN)
B,FB=solve(ST,"BASE");BD,FD=solve(SD,"DELTA");BN,FN=solve(SN,"NULL")
for name,fit in(("BASE",FB),("DELTA",FD),("NULL",FN)):
 print(f" {name:6s} | maps={len(fit)} | train R²={sum(r['R2_train']for r in fit)/len(fit):.6f}")
print("[4/7] TARGETED X-RAY — SOURCE HEAD POOL → TOKEN RESAMPLE → LEARNED MAP")
LOSS=bridge_loss_xray(MF["ORIGINAL"],MF["COUNTERFACTUAL"],QF["ORIGINAL"],QF["COUNTERFACTUAL"],B,BD,BN)
print_loss(LOSS);rec({"group":"BRIDGE_LOSS_XRAY","rows":LOSS})
print("[5/7] HELD-OUT VS FINAL DIFFERENTIAL")
VALID={m:[]for m in MODES};FINAL={}
for i in range(24,32):
 for mode,(which,g)in MODES.items():
  D=BN if which=="NULL"else BD
  VALID[mode].append(evaluate_delta(MA[i],MB[i],QA[i],QB[i],B,D,g))
for mode,(which,g)in MODES.items():
 D=BN if which=="NULL"else BD
 FINAL[mode]=evaluate_delta(MF["ORIGINAL"],MF["COUNTERFACTUAL"],QF["ORIGINAL"],QF["COUNTERFACTUAL"],B,D,g)
 k=sum(x["K"]for x in VALID[mode])/8;v=sum(x["V"]for x in VALID[mode])/8
 print(f" {mode:7s} | VAL ΔK={k:+.4f} ΔV={v:+.4f} | FINAL ΔK={FINAL[mode]['K']:+.4f} ΔV={FINAL[mode]['V']:+.4f}")
rec({"group":"VALIDATION_AND_FINAL","validation":VALID,"final":FINAL})
print("[6/7] SOURCE-BLIND READOUT + NATIVE AUDIT")
PKV={}
for name,m in MF.items():
 q=QF[name];n=q["T"]-1;nb=max(1,round((m["T"]-1)*RATIO))
 for mode,(which,g)in MODES.items():
  D=BN if which=="NULL"else BD
  pb=predict(m,B,D,nb,sink,g)
  if name in("ORIGINAL","COUNTERFACTUAL"):PKV[name+"_"+mode]=install(qm,pb[0],pb[1])
  if name=="WRONG"and mode in("BASE","DELTA"):PKV[name+"_"+mode]=install(qm,pb[0],pb[1])
 PKV[name+"_NATIVE"]=install(qm,q["K"],q["V"])
native=PKV["ORIGINAL_NATIVE"]
nc=qm(input_ids=torch.tensor([[pad(qt)]+ids(qt,FACT+SEP)],device=DEV),use_cache=True).past_key_values
NK=max(float((native[0][L].float()-kvget(nc,L)[0].float()).norm()/kvget(nc,L)[0].float().norm().clamp_min(1e-12))for L in range(28))
NV=max(float((native[1][L].float()-kvget(nc,L)[1].float()).norm()/kvget(nc,L)[1].float().norm().clamp_min(1e-12))for L in range(28))
print(" NATIVE PKV ERROR K:",NK,"V:",NV)
if max(NK,NV)>.05:raise RuntimeError("Native PKV audit failed")
del nc;clean()
ARMS=[("VANILLA",None),("NATIVE_QWEN","ORIGINAL_NATIVE")]
for mode in MODES:ARMS.extend([("BLIND_"+mode,"ORIGINAL_"+mode),("CF_BLIND_"+mode,"COUNTERFACTUAL_"+mode)])
ARMS.extend([("WRONG_BASE","WRONG_BASE"),("WRONG_DELTA","WRONG_DELTA"),("CF_NATIVE","COUNTERFACTUAL_NATIVE")])
ANS=[]
for tag,q,expected,cfexpected in QUESTIONS:
 print("\nQUESTION:",tag,q)
 for arm,key in ARMS:
  output=gen(qt,qm,q,PKV[key]if key else None);first=output.split("\n",1)[0].strip()
  score=hit(first,cfexpected)if arm.startswith("CF_")and tag=="WHO"else hit(first,expected)if not arm.startswith("CF_")else None
  r={"group":"READOUT","question_type":tag,"arm":arm,"first":first,"full":output,"hit":score}
  ANS.append(r);rec(r);print(f" {arm:19s} | hit={score} | {first[:120]}")
 clean()
print("[7/7] INTEGRITY + REPORT")
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen weight sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Model not frozen")
SCORES={arm:{"hits":sum(r["hit"]for r in ANS if r["arm"]==arm and r["hit"]is not None),
 "total":sum(r["arm"]==arm and r["hit"]is not None for r in ANS)}for arm,_ in ARMS}
REPORT={"test":393,"title":"Bridge Loss Localization","source":MID,"target":QID,"seed":SEED,
 "mistral_native_trace":MTRACE,"qwen_native_trace":QTRACE,"bridge_loss_xray":LOSS,
 "validation":VALID,"final":FINAL,"answers":ANS,"scores":SCORES,
 "calibration":{"train_pairs":24,"validation_pairs":8,"token_ratio":RATIO,"layer_map":LMAP,
  "null_permutation":perm,"fit":{"BASE":FB,"DELTA":FD,"NULL":FN}},
 "native_cache_audit":{"K":NK,"V":NV},"integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1},
 "limitations":["Head-retained energy measures the symmetric subspace retained by adjacent-head averaging, not semantic importance.",
 "Token roundtrip cosine measures interpolation distortion, not independently identified causal information loss.",
 "Native traces use position interpolation because original and counterfactual token counts differ.",
 "Mapping cosine compares transformed source differential to target-native differential; it does not isolate downstream behavioral causation.",
 "Only BLIND arms test source-blind transfer. Target-native facts are used for diagnostic comparisons and native controls.",
 "This test does not retrain alternative head mappings or token alignment methods.",
 "Weight sentinels sample selected parameters rather than all model weights."]}
SUMMARY.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("\nFINAL SCORES:")
for k,v in SCORES.items():print(f" {k:19s} {v['hits']}/{v['total']}")
print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS | NATIVE PKV AUDIT: PASS")
print("JSONL:",LOG);print("SUMMARY:",SUMMARY);print("="*110)
