# TEST 404 — CONTENT-WRITE X-RAY — TEST403/402/401/399 EXACT LINEAGE — FULL STANDALONE
# NATIVE ROUTING LOCKED: CAN MISTRAL FACT↔CF CREATE THE CORRECT QWEN ΔV CONTENT?
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
DEV="cuda";DTYPE=torch.bfloat16;SEP="\n\n";MID="mistralai/Mistral-7B-Instruct-v0.3";QID="Qwen/Qwen2.5-7B-Instruct"
ROOT=Path("/content/AKBASCORE_TEST404")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST404");ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/"TEST404_SUMMARY.json"
FACT="Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge."
CF="Leyla Demir planted the Turkish flag at the base of the Golden Gate Bridge."
WHO="Who planted the Turkish flag at the base of the Golden Gate Bridge?"
NEUTRAL=[
"Jonas Weber carried the wooden crate across the quiet market square.","Priya Nair wrote a long letter to her cousin in the morning.",
"The small boat drifted slowly toward the rocky shore.","Omar Haddad fixed the broken clock in the village school.",
"A tired teacher closed the green door of the library.","Sofia Rossi baked fresh bread for the harvest festival.",
"The children watched the kites rising above the hill.","Liam O'Connor sold his old bicycle to a neighbor.",
"Heavy rain flooded the narrow street near the market.","Nadia Petrova translated the ancient manuscript into French.",
"The farmer counted the sheep before sunset.","Hiro Sato opened a tiny bakery beside the river.",
"An old dog slept under the kitchen table all afternoon.","Carlos Mendes washed the windows of his grandmother's house.",
"The museum guard locked the heavy gate at midnight.","Fatima Zahra grew tomatoes in the backyard garden.",
"The pilot announced a short delay because of fog.","Anna Kowalski found a lost wallet on the bus.",
"Snow covered the mountain village during the night.","Ravi Kumar taught his brother how to play chess.",
"The chef sharpened every knife before dinner service.","Lucas Martin cleaned the roof of the barn after the storm.",
"A young violinist practiced scales in the empty hall.","Mei Lin delivered the package to the wrong address.",
"Marek Novak placed the glass bottle beneath the wooden bench.","Sara Ibrahim carried a red notebook into the quiet classroom.",
"Noah Schmidt repaired the small radio beside the kitchen window.","Yuki Mori left a paper envelope near the station entrance.",
"Amira Hassan moved the ceramic bowl onto the upper shelf.","Peter Novak opened the metal box behind the old theater.",
"Lucia Costa placed the yellow scarf inside the travel bag.","Daniel Kim carried a black umbrella through the central courtyard."]
SWAP=[
("Jonas Weber","Elena Fischer"),("Priya Nair","Daniel Brooks"),("The small boat","The large boat"),("Omar Haddad","Lucas Martin"),
("A tired teacher","A young teacher"),("Sofia Rossi","Nadia Petrova"),("The children","The visitors"),("Liam O'Connor","Peter Novak"),
("Heavy rain","Strong winds"),("Nadia Petrova","Anna Kowalski"),("The farmer","The gardener"),("Hiro Sato","Ravi Kumar"),
("An old dog","A young dog"),("Carlos Mendes","Daniel Kim"),("The museum guard","The night porter"),("Fatima Zahra","Amira Hassan"),
("The pilot","The captain"),("Anna Kowalski","Lucia Costa"),("Snow","Rain"),("Ravi Kumar","Omar Haddad"),("The chef","The baker"),
("Lucas Martin","Marek Novak"),("A young violinist","An experienced violinist"),("Mei Lin","Yuki Mori"),("Marek Novak","Jonas Weber"),
("Sara Ibrahim","Priya Nair"),("Noah Schmidt","Hiro Sato"),("Yuki Mori","Mei Lin"),("Amira Hassan","Fatima Zahra"),
("Peter Novak","Carlos Mendes"),("Lucia Costa","Sofia Rossi"),("Daniel Kim","Liam O'Connor")]
ALT=[]
for s,(a,b)in zip(NEUTRAL,SWAP):
 assert s.startswith(a)and a!=b;ALT.append(b+s[len(a):])
def clean():gc.collect();torch.cuda.empty_cache()
def sha(ps):
 h=hashlib.sha256()
 for p in ps:
  a=p.detach().reshape(-1);n=a.numel()
  for j in range(16):
   off=j*max(0,n-256)//15;h.update(a[off:off+256].float().cpu().numpy().tobytes())
 return h.hexdigest()
def load(mid,expected):
 print("LOAD:",mid);t=time.time();tv=tuple(int("".join(c for c in x if c.isdigit())or 0)for x in transformers.__version__.split(".")[:2]);kw={"dtype"if tv>=(4,56)else"torch_dtype":DTYPE}
 tok=AutoTokenizer.from_pretrained(mid,use_fast=True);model=AutoModelForCausalLM.from_pretrained(mid,device_map={"":0},attn_implementation="sdpa",**kw).eval()
 for p in model.parameters():p.requires_grad_(False)
 c=model.config;layers=model.model.layers;dims=(len(layers),c.hidden_size,c.num_attention_heads,c.num_key_value_heads,getattr(c,"head_dim",None)or c.hidden_size//c.num_attention_heads)
 if dims!=expected:raise RuntimeError(f"Architecture mismatch: {dims}")
 print("READY:",dims,"|",round(time.time()-t,2),"s");return tok,model,layers
def ids(tok,s):return tok(s,add_special_tokens=False).input_ids
def pad(tok):return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
@torch.inference_mode()
def forge(tok,model,layers,s,with_q=False):
 seq=[pad(tok)]+ids(tok,s+SEP);T=len(seq);o=model(input_ids=torch.tensor([seq],device=DEV),use_cache=False,output_hidden_states=True,return_dict=True);K=[];V=[];Q=[]
 if with_q:
  pos=torch.arange(T,device=DEV)[None];cr,sr=model.model.rotary_emb(o.hidden_states[0],pos)
 for L,layer in enumerate(layers):
  z=layer.input_layernorm(o.hidden_states[L][0]);a=layer.self_attn;K.append(a.k_proj(z).float().cpu().contiguous());V.append(a.v_proj(z).float().cpu().contiguous())
  if with_q:
   q=a.q_proj(z).reshape(1,T,28,128).transpose(1,2);qr,_=qwen_rope(q,q,cr,sr,unsqueeze_dim=1);Q.append(qr[0].float().cpu().contiguous())
 del o;return {"T":T,"K":K,"V":V,"Q":Q}
LMAP=[min(31,max(0,round((L+.5)*32/28-.5)))for L in range(28)]
def resize(x,n):
 if x.shape[0]==n:return x.contiguous()
 return F.interpolate(x.T.unsqueeze(0),size=n,mode="linear",align_corners=False)[0].T.contiguous()
def src(f,L,n,kind):
 x=f[kind][LMAP[L]][1:].reshape(-1,8,128).reshape(-1,4,2,128).mean(2);return resize(x.reshape(-1,512),n).reshape(n,4,128)
def tgt(f,L,kind):return f[kind][L][1:].reshape(-1,4,128)
def stats():return {k:[[{name:torch.zeros(sh,dtype=torch.float64)for name,sh in(("sx",(128,)),("sy",(128,)),("xx",(128,128)),("xy",(128,128)),("yy",()))}|{"n":0}for h in range(4)]for L in range(28)]for k in("K","V")}
def add(st,X,Y):
 X=X.double();Y=Y.double()
 for h in range(4):
  a=X[:,h];b=Y[:,h];s=st[h];n=a.shape[0];s["sx"]+=a.sum(0);s["sy"]+=b.sum(0);s["xx"]+=a.T@a;s["xy"]+=a.T@b;s["yy"]+=(b*b).sum();s["n"]+=n
def pair(ma,mb,qa,qb,L,kind):
 n=min(qa["T"],qb["T"])-1;X=src(mb,L,n,kind)-src(ma,L,n,kind);a=resize(tgt(qa,L,kind).reshape(-1,512),n).reshape(n,4,128);b=resize(tgt(qb,L,kind).reshape(-1,512),n).reshape(n,4,128);return X,b-a
def solve(st):
 B={k:[]for k in st};fit=[]
 for kind in st:
  for L in range(28):
   g=[]
   for h in range(4):
    s=st[kind][L][h];n=s["n"];mx=s["sx"]/n;my=s["sy"]/n;xx=s["xx"]-n*torch.outer(mx,mx);xy=s["xy"]-n*torch.outer(mx,my);alpha=max(1e-8,float(xx.trace()/128)*.05);W=torch.linalg.solve(xx+alpha*torch.eye(128,dtype=torch.float64),xy)
    err=float((s["yy"]-n*(my@my)-2*(W*xy).sum()+(W*(xx@W)).sum()).clamp_min(0));var=float((s["yy"]-n*(my@my)).clamp_min(1e-12));fit.append(1-err/var);g.append((W.float(),mx.float(),my.float()))
   B[kind].append(g)
 return B,sum(fit)/len(fit)
def predict(m,B,D,n,sink,gain=0.):
 out={}
 for kind in("K","V"):
  arr=[]
  for L in range(28):
   X=src(m,L,n,kind);Y=[]
   for h in range(4):
    W,mx,my=B[kind][L][h];WD,_,_=D[kind][L][h];Y.append((X[:,h]-mx)@W+my+gain*((X[:,h]-mx)@(WD-W)))
   arr.append(torch.cat([sink[kind][L],torch.stack(Y,1).reshape(n,512)],0))
  out[kind]=arr
 return out
def native_resize(q,T):return {k:[torch.cat([q[k][L][:1],resize(q[k][L][1:],T-1)],0)for L in range(28)]for k in("K","V")}
@torch.inference_mode()
def fixed_query(tok,model,layers):
 seq=[pad(tok)]+ids(tok,"QUESTION:\n"+WHO+"\n\nANSWER:");T=len(seq);o=model(input_ids=torch.tensor([seq],device=DEV),use_cache=False,output_hidden_states=True,return_dict=True);pos=torch.arange(T,device=DEV)[None];cr,sr=model.model.rotary_emb(o.hidden_states[0],pos);Q=[]
 for L,layer in enumerate(layers):
  z=layer.input_layernorm(o.hidden_states[L][0]);a=layer.self_attn;q=a.q_proj(z).reshape(1,T,28,128).transpose(1,2);qr,_=qwen_rope(q,q,cr,sr,unsqueeze_dim=1);Q.append(qr[0,:,-1].float().cpu().contiguous())
 del o;return Q
def rotk(raw,L):
 T=raw["K"][L].shape[0];p=torch.arange(T,dtype=torch.float32);inv=1.0/(1000000.0**(torch.arange(0,128,2,dtype=torch.float32)/128));a=torch.outer(p,inv);co=torch.cat([a,a],-1).cos();si=torch.cat([a,a],-1).sin();k=raw["K"][L].reshape(T,4,128).float()
 return k*co[:,None,:]+torch.cat([-k[...,64:],k[...,:64]],-1)*si[:,None,:]
def read(K,V,Q,L):
 k=rotk(K,L);v=V["V"][L].reshape(-1,4,128).float();q=Q[L].reshape(4,7,128)
 if k.shape[0]!=v.shape[0]:raise RuntimeError("K/V token mismatch")
 s=torch.einsum("ghd,tgd->ght",q,k)/math.sqrt(128);w=s.softmax(-1);r=torch.einsum("ght,tgd->ghd",w,v);return w,r
def cos(a,b):return float(F.cosine_similarity(a.float().reshape(-1),b.float().reshape(-1),dim=0,eps=1e-12))
def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-12))
def packet(K,VF,VC,Q,NF,NC):
 rc=[];vr=[];rn=[];vn=[];layers=[]
 for L in range(28):
  w,rf=read(K,VF,Q,L);_,rcf=read(K,VC,Q,L);_,rnf=read(K,NF,Q,L);_,rnc=read(K,NC,Q,L)
  dv=VC["V"][L].float()-VF["V"][L].float();dn=NC["V"][L].float()-NF["V"][L].float();dr=rcf-rf;drn=rnc-rnf
  cR=cos(dr,drn);cV=cos(dv,dn);rc.append(cR);vr.append(cV);rn.append(rel(dr,drn));vn.append(rel(dv,dn))
  layers.append({"L":L,"READ_DELTA_COS":cR,"V_DELTA_COS":cV,"READ_REL":rn[-1],"V_REL":vn[-1],"READ_NORM":float(dr.norm()),"NATIVE_READ_NORM":float(drn.norm()),"V_NORM":float(dv.norm()),"NATIVE_V_NORM":float(dn.norm())})
 return {"READ_DELTA_COS":sum(rc)/28,"V_DELTA_COS":sum(vr)/28,"READ_REL":sum(rn)/28,"V_REL":sum(vn)/28,"layers":layers}
def mixV(base,delta,scale):
 return {"K":base["K"],"V":[base["V"][L].float()+scale*delta[L].float()for L in range(28)]}
def deltaV(A,B):return [B["V"][L].float()-A["V"][L].float()for L in range(28)]
def normmatch(delta,ref):
 out=[]
 for d,r in zip(delta,ref):out.append(d.float()*float(r.float().norm()/d.float().norm().clamp_min(1e-12)))
 return out
def random_delta_like(ref,seed):
 g=torch.Generator().manual_seed(seed);out=[]
 for r in ref:
  z=torch.randn(r.shape,generator=g);z=z/z.norm().clamp_min(1e-12)*r.float().norm();out.append(z)
 return out
print("="*120);print("TEST 404 — CONTENT-WRITE X-RAY");print("NATIVE ROUTING LOCKED: CAN MISTRAL FACT↔CF CREATE THE CORRECT QWEN ΔV CONTENT?");print("="*120)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/8] MISTRAL — TEST399 SOURCE EXTRACTION")
mt,mm,ml=load(MID,(32,4096,32,8,128));mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight];M0=sha(mp);MA=[];MB=[]
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 MA.append(forge(mt,mm,ml,a));MB.append(forge(mt,mm,ml,b));print(f" MISTRAL {i+1}/32",end="\r")
MF={k:forge(mt,mm,ml,s)for k,s in(("FACT",FACT),("CF",CF))};print("\n SOURCE LENGTHS:",{k:v["T"]for k,v in MF.items()})
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/8] QWEN — TEST399 TARGET EXTRACTION")
qt,qm,ql=load(QID,(28,3584,28,4,128));qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight];Q0=sha(qp);QA=[];QB=[];sink=None
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 qa=forge(qt,qm,ql,a,True);qb=forge(qt,qm,ql,b,True);QA.append(qa);QB.append(qb)
 if sink is None:sink={k:[qa[k][L][:1].clone()for L in range(28)]for k in("K","V")}
 print(f" QWEN {i+1}/32",end="\r")
QF={k:forge(qt,qm,ql,s,True)for k,s in(("FACT",FACT),("CF",CF))};T=QF["FACT"]["T"];assert T==QF["CF"]["T"]==19
print("\n TARGET LENGTH:",T,"| TRAIN=24 | VALIDATION=8 | FINAL FACT/CF NEVER FIT")
print("[3/8] TEST399 FP64 BASE + DELTA BRIDGE")
ST=stats();SD=stats()
for i in range(24):
 for kind in("K","V"):
  for L in range(28):
   for m,q in((MA[i],QA[i]),(MB[i],QB[i])):
    n=q["T"]-1;add(ST[kind][L],src(m,L,n,kind),tgt(q,L,kind))
   X,Y=pair(MA[i],MB[i],QA[i],QB[i],L,kind);add(SD[kind][L],X,Y)
B,R2B=solve(ST);D,R2D=solve(SD);print(f" TRAIN R² BASE={R2B:.6f} DELTA={R2D:.6f}")
NF=native_resize(QF["FACT"],T);NC=native_resize(QF["CF"],T)
RAW={}
for name,g in (("BASE",0.),("HYBRID",.5),("DELTA",1.)):
 RAW[name]={"FACT":predict(MF["FACT"],B,D,T-1,sink,g),"CF":predict(MF["CF"],B,D,T-1,sink,g)}
del ST,SD,MA,MB,QA,QB;clean()
print("[4/8] TEST399 FIXED WHO QUERY + ROUTING LOCK")
QWHO=fixed_query(qt,qm,ql);assert len(QWHO)==28 and all(q.shape==(28,128)for q in QWHO)
KLOCK=NF
print(" WHO QUERY BANK:",len(QWHO),"layers ×",tuple(QWHO[0].shape));print(" K LOCK: QWEN_NATIVE_FACT_K | SAME K USED FOR EVERY FACT/CF ARM | ΔK=0")
print("[5/8] DIRECT CONTENT TRANSFER — MAPPED ΔV UNDER IDENTICAL NATIVE K")
RES={}
NATIVE=packet(KLOCK,NF,NC,QWHO,NF,NC);RES["NATIVE"]=NATIVE
print(f" NATIVE       READΔ={NATIVE['READ_DELTA_COS']:+.6f} VΔ={NATIVE['V_DELTA_COS']:+.6f} READ_REL={NATIVE['READ_REL']:.6f} V_REL={NATIVE['V_REL']:.6f}")
for name in ("BASE","HYBRID","DELTA"):
 r=packet(KLOCK,RAW[name]["FACT"],RAW[name]["CF"],QWHO,NF,NC);RES[name]=r
 print(f" {name:12s} READΔ={r['READ_DELTA_COS']:+.6f} VΔ={r['V_DELTA_COS']:+.6f} READ_REL={r['READ_REL']:.6f} V_REL={r['V_REL']:.6f}")
print("[6/8] ISOLATED WRITE — NATIVE FACT V + MISTRAL-PREDICTED ΔV")
NATIVE_DV=deltaV(NF,NC);WRITE={}
for name in ("BASE","HYBRID","DELTA"):
 dv=deltaV(RAW[name]["FACT"],RAW[name]["CF"]);dvN=normmatch(dv,NATIVE_DV)
 VF=NF;VC_RAW=mixV(NF,dv,1.0);VC_NORM=mixV(NF,dvN,1.0)
 rr=packet(KLOCK,VF,VC_RAW,QWHO,NF,NC);rn=packet(KLOCK,VF,VC_NORM,QWHO,NF,NC);WRITE[name+"_RAW"]=rr;WRITE[name+"_NORM"]=rn
 print(f" {name:6s} RAW  READΔ={rr['READ_DELTA_COS']:+.6f} VΔ={rr['V_DELTA_COS']:+.6f} READ_REL={rr['READ_REL']:.6f}")
 print(f" {name:6s} NORM READΔ={rn['READ_DELTA_COS']:+.6f} VΔ={rn['V_DELTA_COS']:+.6f} READ_REL={rn['READ_REL']:.6f}")
RAND=random_delta_like(NATIVE_DV,404);VR=mixV(NF,RAND,1.0);RR=packet(KLOCK,NF,VR,QWHO,NF,NC);WRITE["RANDOM_NORM"]=RR
print(f" RANDOM NORM READΔ={RR['READ_DELTA_COS']:+.6f} VΔ={RR['V_DELTA_COS']:+.6f} READ_REL={RR['READ_REL']:.6f}")
print("[7/8] LAYER X-RAY — CONTENT ALIGNMENT")
BEST=max(("BASE","HYBRID","DELTA"),key=lambda x:WRITE[x+"_NORM"]["READ_DELTA_COS"]);BR=WRITE[BEST+"_NORM"]
print("      DIRECT_BASE DIRECT_HYB DIRECT_DELTA | BEST_WRITE_READ BEST_WRITE_V")
for L in [0,3,6,10,14,18,19,21,23,25,27]:
 print(f" L{L:02d} {RES['BASE']['layers'][L]['READ_DELTA_COS']:+11.3f} {RES['HYBRID']['layers'][L]['READ_DELTA_COS']:+10.3f} {RES['DELTA']['layers'][L]['READ_DELTA_COS']:+12.3f} | {BR['layers'][L]['READ_DELTA_COS']:+15.3f} {BR['layers'][L]['V_DELTA_COS']:+12.3f}")
print("[8/8] VERDICT + INTEGRITY")
best_direct=max(("BASE","HYBRID","DELTA"),key=lambda x:RES[x]["READ_DELTA_COS"]);bd=RES[best_direct]["READ_DELTA_COS"];bw=WRITE[BEST+"_NORM"]["READ_DELTA_COS"];bv=WRITE[BEST+"_NORM"]["V_DELTA_COS"];rand=RR["READ_DELTA_COS"]
print(f" BEST DIRECT={best_direct} READΔ={bd:+.6f}")
print(f" BEST ISOLATED WRITE={BEST} READΔ={bw:+.6f} VΔ={bv:+.6f}")
print(f" RANDOM NORM CONTROL READΔ={rand:+.6f}")
if bw>=.60 and bv>=.20 and bw>rand+.15:VERDICT="CONTENT_SIGNAL_SURVIVES: Mistral-derived ΔV contains a non-random Qwen-readable content direction under fixed native routing."
elif bw>=.40 and bw>rand+.10:VERDICT="PARTIAL_CONTENT_SIGNAL: Mistral-derived ΔV has measurable Qwen-readable structure, but transfer remains incomplete."
elif abs(bw-rand)<.10:VERDICT="NO_SPECIFIC_CONTENT_TRANSFER: Mistral-derived ΔV is not clearly better than the norm-matched random control."
else:VERDICT="WEAK_OR_MIXED_CONTENT_SIGNAL: inspect layer-local alignment before attempting behavioral write."
print(" VERDICT:",VERDICT)
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Qwen not frozen")
REPORT={"test":404,"title":"Content-Write X-Ray","source_model":MID,"target_model":QID,"seed":SEED,"fact":FACT,"counterfactual":CF,"question":WHO,
"train_pairs":24,"validation_pairs":8,"train_R2":{"BASE":R2B,"DELTA":R2D},"routing_lock":"QWEN_NATIVE_FACT_K","deltaK":0,
"direct":RES,"isolated_write":WRITE,"best_direct":best_direct,"best_write":BEST,"verdict":VERDICT,
"integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1,"pass":Q0==Q1},
"notes":["Exact TEST399 bridge, dataset, layer map, interpolation and fixed-WHO lineage retained.","FACT/CF are never used to fit the bridge.","The identical native Qwen FACT K is used on both sides of every comparison; ΔK=0 throughout.","DIRECT arms test mapped FACT V versus mapped CF V under fixed native routing.","ISOLATED WRITE arms start from native FACT V and add only the Mistral-derived mapped ΔV.","NORM arms match each predicted layer ΔV norm to the native held-out ΔV norm without changing its direction.","RANDOM_NORM is a norm-matched random-direction control.","No model parameters are modified. This is an X-ray/causal readout test, not yet behavioral generation."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("="*120);print("TEST 404 — COMPLETE");print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS");print("SUMMARY:",OUT);print("="*120)
