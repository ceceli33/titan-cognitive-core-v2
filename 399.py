# TEST 399 — FIXED-QUERY CAUSAL K/V TRANSPLANT X-RAY — FULL STANDALONE
import os,sys,gc,json,time,random,hashlib,subprocess,importlib.util,math
from pathlib import Path
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
 if importlib.util.find_spec(m)is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import torch,transformers
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as qwen_rope
if not torch.cuda.is_available():raise RuntimeError("CUDA REQUIRED")
os.environ["TOKENIZERS_PARALLELISM"]="false";torch.set_grad_enabled(False)
SEED=392;random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEV="cuda";DTYPE=torch.bfloat16;SEP="\n\n"
MID="mistralai/Mistral-7B-Instruct-v0.3";QID="Qwen/Qwen2.5-7B-Instruct"
ROOT=Path("/content/AKBASCORE_TEST399")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST399")
ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/"TEST399_SUMMARY.json"
FACT="Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge."
CF="Leyla Demir planted the Turkish flag at the base of the Golden Gate Bridge."
WHO="Who planted the Turkish flag at the base of the Golden Gate Bridge?"
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
ALT=[]
for s,(a,b)in zip(NEUTRAL,SWAP):
 assert s.startswith(a)and a!=b
 ALT.append(b+s[len(a):])
def clean():gc.collect();torch.cuda.empty_cache()
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
   q=a.q_proj(z).reshape(1,T,28,128).transpose(1,2);qr,_=qwen_rope(q,q,cr,sr,unsqueeze_dim=1)
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
@torch.inference_mode()
def fixed_query(tok,model,layers):
 seq=[pad(tok)]+ids(tok,"QUESTION:\n"+WHO+"\n\nANSWER:");T=len(seq)
 out=model(input_ids=torch.tensor([seq],device=DEV),use_cache=False,output_hidden_states=True,return_dict=True)
 pos=torch.arange(T,device=DEV)[None];cr,sr=model.model.rotary_emb(out.hidden_states[0],pos);Q=[]
 for L,layer in enumerate(layers):
  z=layer.input_layernorm(out.hidden_states[L][0]);a=layer.self_attn
  q=a.q_proj(z).reshape(1,T,28,128).transpose(1,2);qr,_=qwen_rope(q,q,cr,sr,unsqueeze_dim=1)
  Q.append(qr[0,:,-1].float().cpu().contiguous())
 del out
 return Q
def rotk(raw,L):
 T=raw["K"][L].shape[0];p=torch.arange(T,dtype=torch.float32)
 inv=1.0/(1000000.0**(torch.arange(0,128,2,dtype=torch.float32)/128))
 a=torch.outer(p,inv);co=torch.cat([a,a],-1).cos();si=torch.cat([a,a],-1).sin()
 k=raw["K"][L].reshape(T,4,128).float()
 return k*co[:,None,:]+torch.cat([-k[...,64:],k[...,:64]],-1)*si[:,None,:]
def read(Kraw,Vraw,Q,L):
 k=rotk(Kraw,L);v=Vraw["V"][L].reshape(-1,4,128).float();q=Q[L].reshape(4,7,128)
 if k.shape[0]!=v.shape[0]:raise RuntimeError("K/V token mismatch")
 s=torch.einsum("ghd,tgd->ght",q,k)/math.sqrt(128);w=s.softmax(-1)
 r=torch.einsum("ght,tgd->ghd",w,v)
 return w,r
def cos(a,b):return float(F.cosine_similarity(a.float().reshape(-1),b.float().reshape(-1),dim=0,eps=1e-12))
def rel(a,b):return float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))
def KL(a,b):
 a=a.float().clamp_min(1e-9);b=b.float().clamp_min(1e-9)
 return float((a*(a.log()-b.log())).sum(-1).mean())
def compare(K,V,Q,N):
 z={"ATT_COS":[],"ATT_KL":[],"READ_COS":[],"READ_REL":[]}
 for L in range(28):
  w,r=read(K,V,Q,L);wn,rn=read(N,N,Q,L)
  z["ATT_COS"].append(cos(w,wn));z["ATT_KL"].append(KL(wn,w));z["READ_COS"].append(cos(r,rn));z["READ_REL"].append(rel(r,rn))
 return {k:sum(v)/28 for k,v in z.items()}
def delta(KF,VF,KC,VC,Q,NF,NC):
 ac=[];rc=[]
 for L in range(28):
  wf,rf=read(KF,VF,Q,L);wc,rcf=read(KC,VC,Q,L)
  wfn,rfn=read(NF,NF,Q,L);wcn,rcn=read(NC,NC,Q,L)
  ac.append(cos(wc-wf,wcn-wfn));rc.append(cos(rcf-rf,rcn-rfn))
 return {"ATT_DELTA_COS":sum(ac)/28,"READ_DELTA_COS":sum(rc)/28}
print("="*112)
print("TEST 399 — FIXED-QUERY CAUSAL K/V TRANSPLANT X-RAY")
print("QWEN WHO-QUERY × K-ONLY × V-ONLY × K+V × FACT↔CF")
print("="*112)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/7] MISTRAL — FROZEN SOURCE EXTRACTION")
mt,mm,ml=load(MID,(32,4096,32,8,128))
mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight]
M0=sha(mp);MA=[];MB=[]
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 MA.append(forge(mt,mm,ml,a));MB.append(forge(mt,mm,ml,b));print(f" MISTRAL {i+1}/32",end="\r")
MF={k:forge(mt,mm,ml,s)for k,s in(("FACT",FACT),("CF",CF))}
print("\n SOURCE LENGTHS:",{k:v["T"]for k,v in MF.items()})
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/7] QWEN — FROZEN TARGET EXTRACTION")
qt,qm,ql=load(QID,(28,3584,28,4,128))
qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight]
Q0=sha(qp);QA=[];QB=[];sink=None
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 qa=forge(qt,qm,ql,a,True);qb=forge(qt,qm,ql,b,True);QA.append(qa);QB.append(qb)
 if sink is None:sink={k:[qa[k][L][:1].clone()for L in range(28)]for k in("K","V")}
 print(f" QWEN {i+1}/32",end="\r")
QF={k:forge(qt,qm,ql,s,True)for k,s in(("FACT",FACT),("CF",CF))}
T=QF["FACT"]["T"]
assert T==QF["CF"]["T"]==19
print("\n TARGET LENGTH:",T,"| TRAIN:24 | VALIDATION:8 | FINAL:FACT/CF")
print("[3/7] TEST398 FP64 BASE + DELTA BRIDGE REBUILD")
ST=stats();SD=stats()
for i in range(24):
 for kind in("K","V"):
  for L in range(28):
   for m,q in((MA[i],QA[i]),(MB[i],QB[i])):
    n=q["T"]-1;add(ST[kind][L],src(m,L,n,kind),tgt(q,L,kind))
   X,Y=pair(MA[i],MB[i],QA[i],QB[i],L,kind);add(SD[kind][L],X,Y)
B,R2B=solve(ST);D,R2D=solve(SD)
print(f" TRAIN R² BASE={R2B:.6f} DELTA={R2D:.6f}")
GAINS={"BASE":0.,"HYBRID":.5,"DELTA":1.}
NATIVE={s:native_resize(QF[s],T)for s in("FACT","CF")};RAW={}
for s in("FACT","CF"):
 for name,g in GAINS.items():RAW[s+"_"+name]=predict(MF[s],B,D,T-1,sink,g)
del ST,SD,MA,MB,QA,QB;clean()
print("[4/7] FIXED QWEN WHO QUERY")
QWHO=fixed_query(qt,qm,ql)
assert len(QWHO)==28 and all(q.shape==(28,128)for q in QWHO)
print(" WHO QUERY BANK:",len(QWHO),"layers ×",tuple(QWHO[0].shape),"| FIXED ACROSS ALL ARMS")
print("[5/7] K-ONLY / V-ONLY / K+V ABLATION")
RESULT={}
for s in("FACT","CF"):
 RESULT[s]={};N=NATIVE[s]
 for name in GAINS:
  R=RAW[s+"_"+name]
  arms={"NATIVE":(N,N),"K_ONLY":(R,N),"V_ONLY":(N,R),"K_PLUS_V":(R,R)}
  RESULT[s][name]={}
  for mode,(K,V) in arms.items():
   m=compare(K,V,QWHO,N);RESULT[s][name][mode]=m
   print(f" {s:4s} {name:7s} {mode:8s} ATT={m['ATT_COS']:+.6f} KL={m['ATT_KL']:.6f} READ={m['READ_COS']:+.6f} REL={m['READ_REL']:.6f}")
print("[6/7] FACT↔CF — SAME Q, CAUSAL COMPONENT DELTA")
DELTAS={}
for name in GAINS:
 RF=RAW["FACT_"+name];RC=RAW["CF_"+name];NF=NATIVE["FACT"];NC=NATIVE["CF"]
 modes={"K_ONLY":(RF,NF,RC,NC),"V_ONLY":(NF,RF,NC,RC),"K_PLUS_V":(RF,RF,RC,RC)}
 DELTAS[name]={}
 for mode,(KF,VF,KC,VC) in modes.items():
  d=delta(KF,VF,KC,VC,QWHO,NF,NC);DELTAS[name][mode]=d
  print(f" {name:7s} {mode:8s} ATTΔ={d['ATT_DELTA_COS']:+.6f} READΔ={d['READ_DELTA_COS']:+.6f}")
print("[7/7] DIAGNOSTIC SCORECARD + INTEGRITY")
DIAG={}
for name in GAINS:
 k=DELTAS[name]["K_ONLY"];v=DELTAS[name]["V_ONLY"];kv=DELTAS[name]["K_PLUS_V"]
 DIAG[name]={"K_ONLY":k,"V_ONLY":v,"K_PLUS_V":kv}
 print(f" {name:7s} | K: ATTΔ={k['ATT_DELTA_COS']:+.6f} READΔ={k['READ_DELTA_COS']:+.6f} | V: ATTΔ={v['ATT_DELTA_COS']:+.6f} READΔ={v['READ_DELTA_COS']:+.6f} | K+V: ATTΔ={kv['ATT_DELTA_COS']:+.6f} READΔ={kv['READ_DELTA_COS']:+.6f}")
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Qwen not frozen")
REPORT={"test":399,"title":"Fixed-Query Causal K/V Transplant X-Ray","source_model":MID,"target_model":QID,
"fact":FACT,"counterfactual":CF,"question":WHO,"seed":SEED,"train_pairs":24,"validation_pairs":8,
"train_R2":{"BASE":R2B,"DELTA":R2D},"fixed_query_shape":[28,28,128],"component_ablation":RESULT,
"fixed_query_fact_cf_delta":DELTAS,"diagnostic_scorecard":DIAG,
"integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1,"pass":Q0==Q1},
"interpretation":{"K_ONLY":"Bridge K + native V; tests routing compatibility.","V_ONLY":"Native K + bridge V; tests value/content compatibility.","K_PLUS_V":"Bridge K + bridge V; tests the complete transplanted cache.","FIXED_Q":"The identical bare WHO query is used for FACT and CF, removing query-context changes from the primary contrast."},
"limitations":["No TEST399 parameters are fitted to FACT or CF.","K-only and V-only are diagnostic hybrid states, not natural model states.","Fixed-query attention/readout alignment diagnoses cache compatibility; it does not by itself prove behavioral factual transfer.","Mistral token positions remain linearly interpolated into Qwen token length as in the working bridge.","Parameter sentinels sample selected frozen weights rather than every model parameter."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("="*112)
print("TEST 399 — COMPLETE")
print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS")
print("SUMMARY:",OUT)
print("="*112)
