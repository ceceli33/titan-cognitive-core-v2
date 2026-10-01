# TEST 436 — FINAL CROSS-MODEL CLOSURE SUITE
# TEST435 PROVEN LINEAGE | TRAIN-ONLY MULTILAYER RESIDUAL BRIDGE + L26 ATT BRIDGE + HYBRID
# L23/L24/L25/L26/L27 × DOSE × 1/2/3/4 PULSE × FORWARD/REVERSE × FREE GENERATION
# FINAL FACT/CF = EVALUATION ONLY | NO FINAL-DERIVED WRITE | NO TARGET FORCING
import os,sys,gc,json,time,random,hashlib,subprocess,importlib.util
from pathlib import Path
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
 if importlib.util.find_spec(m)is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import torch,transformers
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
if not torch.cuda.is_available():raise RuntimeError("CUDA REQUIRED")
os.environ["TOKENIZERS_PARALLELISM"]="false";torch.set_grad_enabled(False)
SEED=392;random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEV="cuda";DTYPE=torch.bfloat16;SEP="\n\n";MID="mistralai/Mistral-7B-Instruct-v0.3";QID="Qwen/Qwen2.5-7B-Instruct"
ROOT=Path("/content/AKBASCORE_TEST436")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST436");ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/"TEST436_SUMMARY.json"
FACT="Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge."
CF="Leyla Demir planted the Turkish flag at the base of the Golden Gate Bridge."
WHO="Who planted the Turkish flag at the base of the Golden Gate Bridge?\nAnswer:"
NEUTRAL=[
"Jonas Weber carried the wooden crate across the quiet market square.","Priya Nair wrote a long letter to her cousin in the morning.","The small boat drifted slowly toward the rocky shore.","Omar Haddad fixed the broken clock in the village school.",
"A tired teacher closed the green door of the library.","Sofia Rossi baked fresh bread for the harvest festival.","The children watched the kites rising above the hill.","Liam O'Connor sold his old bicycle to a neighbor.",
"Heavy rain flooded the narrow street near the market.","Nadia Petrova translated the ancient manuscript into French.","The farmer counted the sheep before sunset.","Hiro Sato opened a tiny bakery beside the river.",
"An old dog slept under the kitchen table all afternoon.","Carlos Mendes washed the windows of his grandmother's house.","The museum guard locked the heavy gate at midnight.","Fatima Zahra grew tomatoes in the backyard garden.",
"The pilot announced a short delay because of fog.","Anna Kowalski found a lost wallet on the bus.","Snow covered the mountain village during the night.","Ravi Kumar taught his brother how to play chess.",
"The chef sharpened every knife before dinner service.","Lucas Martin cleaned the roof of the barn after the storm.","A young violinist practiced scales in the empty hall.","Mei Lin delivered the package to the wrong address.",
"Marek Novak placed the glass bottle beneath the wooden bench.","Sara Ibrahim carried a red notebook into the quiet classroom.","Noah Schmidt repaired the small radio beside the kitchen window.","Yuki Mori left a paper envelope near the station entrance.",
"Amira Hassan moved the ceramic bowl onto the upper shelf.","Peter Novak opened the metal box behind the old theater.","Lucia Costa placed the yellow scarf inside the travel bag.","Daniel Kim carried a black umbrella through the central courtyard."]
SWAP=[
("Jonas Weber","Elena Fischer"),("Priya Nair","Daniel Brooks"),("The small boat","The large boat"),("Omar Haddad","Lucas Martin"),("A tired teacher","A young teacher"),("Sofia Rossi","Nadia Petrova"),("The children","The visitors"),("Liam O'Connor","Peter Novak"),
("Heavy rain","Strong winds"),("Nadia Petrova","Anna Kowalski"),("The farmer","The gardener"),("Hiro Sato","Ravi Kumar"),("An old dog","A young dog"),("Carlos Mendes","Daniel Kim"),("The museum guard","The night porter"),("Fatima Zahra","Amira Hassan"),
("The pilot","The captain"),("Anna Kowalski","Lucia Costa"),("Snow","Rain"),("Ravi Kumar","Omar Haddad"),("The chef","The baker"),("Lucas Martin","Marek Novak"),("A young violinist","An experienced violinist"),("Mei Lin","Yuki Mori"),
("Marek Novak","Jonas Weber"),("Sara Ibrahim","Priya Nair"),("Noah Schmidt","Hiro Sato"),("Yuki Mori","Mei Lin"),("Amira Hassan","Fatima Zahra"),("Peter Novak","Carlos Mendes"),("Lucia Costa","Sofia Rossi"),("Daniel Kim","Liam O'Connor")]
ALT=[]
for s,(a,b) in zip(NEUTRAL,SWAP):
 assert s.startswith(a)and a!=b;ALT.append(b+s[len(a):])
LMAP=[min(31,max(0,round((L+.5)*32/28-.5)))for L in range(28)]
TRAIN=list(range(24));VAL=list(range(24,32));LAYERS=list(range(28));WRITE_LAYERS=list(range(14,24));TAIL=[23,24,25,26,27]
RIDGES=[1e-4,1e-3,1e-2,.05,.1,.5,1.,5.];BR=[1e-4,1e-3,1e-2,.1,1.]
DOSES=[.10,.20,.30,.40,.50,.75,1.,1.25,1.5,2.]
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
 if dims!=expected:raise RuntimeError(f"Architecture mismatch {dims}")
 print("READY:",dims,"|",round(time.time()-t,2),"s");return tok,model,layers
def ids(tok,s):return tok(s,add_special_tokens=False).input_ids
def pad(tok):return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
def unit(x):return x.float()/x.float().norm().clamp_min(1e-12)
def cos(a,b):return float(F.cosine_similarity(a.float().reshape(-1),b.float().reshape(-1),dim=0,eps=1e-12))
@torch.inference_mode()
def forge(tok,model,s):
 seq=[pad(tok)]+ids(tok,s+SEP);o=model(input_ids=torch.tensor([seq],device=DEV),use_cache=False,output_hidden_states=True,return_dict=True);pos=max(0,len(seq)-len(ids(tok,SEP))-1)
 H=[o.hidden_states[L+1][0,pos].float().cpu().contiguous()for L in range(len(model.model.layers))];del o;return H
def packet(A,B,layers):return [unit(B[L]-A[L])for L in layers]
def catpacket(p):return torch.cat([unit(x)for x in p])
def gfp(x,bank,refs):return torch.tensor([cos(catpacket(x),catpacket(bank[j]))for j in refs])
def ridge(X,Y,lam):
 X=X.double();Y=Y.double();mx=X.mean(0);my=Y.mean(0);A=X-mx;B=Y-my;xx=A.T@A;xy=A.T@B;s=max(1e-12,float(xx.trace()/max(1,xx.shape[0])));W=torch.linalg.solve(xx+lam*s*torch.eye(xx.shape[0],dtype=torch.float64),xy);return W.float(),mx.float(),my.float()
def pred(x,f):W,mx,my=f;return(x.float()-mx)@W+my
def fit_global(MP,QP,refs,lam):return ridge(torch.stack([gfp(MP[i],MP,refs)for i in refs]),torch.stack([gfp(QP[i],QP,refs)for i in refs]),lam)
def global_qsolve(target,QP,refs,lam):
 C=torch.stack([catpacket(QP[j])for j in refs]);G=C@C.T;a=torch.linalg.solve(G.T@G+lam*torch.eye(len(refs)),G.T@target.float())
 return {L:unit(sum((a[k]*QP[j][L]for k,j in enumerate(refs)),torch.zeros_like(QP[refs[0]][L])))for L in LAYERS},a
def loo_score(MP,QP,lam):
 fc=[];lc=[]
 for hold in TRAIN:
  refs=[j for j in TRAIN if j!=hold];f=fit_global(MP,QP,refs,lam);qh=pred(gfp(MP[hold],MP,refs),f);qt=gfp(QP[hold],QP,refs);d,_=global_qsolve(qh,QP,refs,lam);fc.append(cos(qh,qt));lc.append(sum(cos(d[L],QP[hold][L])for L in LAYERS)/28)
 return sum(fc)/len(fc),sum(lc)/len(lc)
print("="*150);print("TEST 436 — FINAL CROSS-MODEL CLOSURE SUITE");print("TEST435 PROVEN LINEAGE | MULTILAYER RESIDUAL + L26 ATTENTION + HYBRID | FINAL CLOSURE");print("="*150)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/13] MISTRAL — TEST435 EXACT PACKETS")
mt,mm,ml=load(MID,(32,4096,32,8,128));mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight];M0=sha(mp);MH=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):MH.append((forge(mt,mm,a),forge(mt,mm,b)));print(f" MISTRAL {i+1}/32",end="\r")
MFINAL=(forge(mt,mm,FACT),forge(mt,mm,CF));print()
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/13] QWEN — TEST435 EXACT PACKETS")
qt,qm,ql=load(QID,(28,3584,28,4,128));qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight];Q0=sha(qp);QH=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):QH.append((forge(qt,qm,a),forge(qt,qm,b)));print(f" QWEN {i+1}/32",end="\r")
QFINAL=(forge(qt,qm,FACT),forge(qt,qm,CF));print("\n TRAIN=24 | VAL=8 | FINAL FACT/CF NEVER USED FOR FIT/SELECTION")
MP=[packet(a,b,LMAP)for a,b in MH];QP=[packet(a,b,LAYERS)for a,b in QH];MF=packet(MFINAL[0],MFINAL[1],LMAP);QF=packet(QFINAL[0],QFINAL[1],LAYERS)
print("[3/13] TEST435 GLOBAL RIDGE")
CV=[]
for lam in RIDGES:
 fc,lc=loo_score(MP,QP,lam);CV.append((lam,fc,lc));print(f" λ={lam:<6g} FP={fc:+.6f} LOCAL={lc:+.6f}")
LAM=max(CV,key=lambda x:(x[2],x[1]))[0];FIT=fit_global(MP,QP,TRAIN,LAM);QHAT=pred(gfp(MF,MP,TRAIN),FIT);QTRUE=gfp(QF,QP,TRAIN);DIRS,COEF=global_qsolve(QHAT,QP,TRAIN,LAM)
MAG={L:sum(float((QH[i][1][L]-QH[i][0][L]).norm())for i in TRAIN)/24 for L in LAYERS}
print(f" SELECTED λ={LAM} | FINAL FP={cos(QHAT,QTRUE):+.6f} LOCAL={sum(cos(DIRS[L],QF[L])for L in LAYERS)/28:+.6f} COEF={float(COEF.norm()):.6f}")
print("[4/13] TRAIN-ONLY MULTILAYER NATIVE CONVERSION BANK")
# Each TRAIN pair teaches how a native Qwen relation changes across L23→L27.
# No FINAL state participates in fitting.
CONV={L:[] for L in TAIL}
for i in range(32):
 A,B=QH[i]
 for L in TAIL:
  if L==23:x=unit(B[22]-A[22]);y=(B[23]-A[23])
  else:x=unit(B[L-1]-A[L-1]);y=(B[L]-A[L])
  CONV[L].append((x.float(),y.float()))
def kfit(L,refs,lam):
 X=torch.stack([CONV[L][i][0]for i in refs]).double();Y=torch.stack([CONV[L][i][1]for i in refs]).double();G=X@X.T;s=max(1e-12,float(G.trace()/len(refs)));A=torch.linalg.solve(G+lam*s*torch.eye(len(refs),dtype=torch.float64),Y);return X.float(),A.float()
def kpred(x,f):X,A=f;return((X@unit(x).float())@A).float()
BF={};BCV={}
for L in TAIL:
 best=None
 for lam in BR:
  cs=[]
  f=kfit(L,TRAIN,lam)
  for i in VAL:cs.append(cos(kpred(CONV[L][i][0],f),CONV[L][i][1]))
  sc=sum(cs)/len(cs)
  print(f" L{L} λ={lam:<6g} VAL_COS={sc:+.6f}")
  if best is None or sc>best[1]:best=(lam,sc)
 BF[L]=kfit(L,TRAIN,best[0]);BCV[L]={"lambda":best[0],"cos":best[1]};print(f" SELECT L{L}: λ={best[0]} COS={best[1]:+.6f}")
print("[5/13] TRAIN-ONLY L26 ATTENTION BRIDGE")
L26=ql[26]
@torch.inference_mode()
def micro(s):
 seq=torch.tensor([[pad(qt)]+ids(qt,s+SEP)],device=DEV);pos=seq.shape[1]-len(ids(qt,SEP))-1;C={};hs=[]
 def pre(m,a):C["IN"]=a[0][0,pos].detach().float().cpu()
 def pa(m,a):C["POST"]=a[0][0,pos].detach().float().cpu()
 hs=[L26.register_forward_pre_hook(pre),L26.post_attention_layernorm.register_forward_pre_hook(pa)]
 try:qm(input_ids=seq,use_cache=False,return_dict=True)
 finally:
  for h in hs:h.remove()
 return C
MIC=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):
 A=micro(a);B=micro(b);MIC.append((unit(B["IN"]-A["IN"]),(B["POST"]-B["IN"])-(A["POST"]-A["IN"])));print(f" L26 ATT {i+1}/32",end="\r")
print()
def afit(refs,lam):
 X=torch.stack([MIC[i][0]for i in refs]).double();Y=torch.stack([MIC[i][1]for i in refs]).double();G=X@X.T;s=max(1e-12,float(G.trace()/len(refs)));A=torch.linalg.solve(G+lam*s*torch.eye(len(refs),dtype=torch.float64),Y);return X.float(),A.float()
abest=None
for lam in BR:
 f=afit(TRAIN,lam);sc=sum(cos(kpred(MIC[i][0],f),MIC[i][1])for i in VAL)/len(VAL);print(f" ATT λ={lam:<6g} VAL_COS={sc:+.6f}")
 if abest is None or sc>abest[1]:abest=(lam,sc)
AF=afit(TRAIN,abest[0]);print(f" SELECT ATT λ={abest[0]} COS={abest[1]:+.6f}")
print("[6/13] SOURCE-ABSENT WHO + NATIVE REFERENCE")
PROMPT=torch.tensor([[pad(qt)]+ids(qt,WHO)],device=DEV);NF=torch.tensor([[pad(qt)]+ids(qt,FACT+SEP+WHO)],device=DEV);NC=torch.tensor([[pad(qt)]+ids(qt,CF+SEP+WHO)],device=DEV)
LEYLA=ids(qt," Leyla")[0];MUSTAFA=ids(qt," Mustafa")[0];W=qm.lm_head.weight.detach().float().cpu();ID=unit(W[LEYLA]-W[MUSTAFA])
def rank(x,i):return int((x>x[i]).sum())+1
def proj(x):return float(x@ID)
@torch.inference_mode()
def plain(seq,hidden=False):
 return qm(input_ids=seq,use_cache=False,output_hidden_states=hidden,return_dict=True)
NB=plain(PROMPT,True);NN0=plain(NF,True);NN1=plain(NC,True);LB=NB.logits[0,-1].float().cpu()
NATIVE={L:(NN1.hidden_states[L+1][0,-1]-NN0.hidden_states[L+1][0,-1]).float().cpu() for L in TAIL}
print(f" WHO TOKENS={PROMPT.shape[1]} | SOURCE FACT PRESENT=NO | BASE LEYLA#{rank(LB,LEYLA)} MARGIN={float(LB[LEYLA]-LB[MUSTAFA]):+.6f}")
del NB,NN0,NN1
print("[7/13] UNIFIED INTERVENTION ENGINE")
def forward(seq,mode="base",dose=0.,sign=1.):
 hs=[];state={}
 if mode!="base":
  for L in WRITE_LAYERS:
   d=(DIRS[L]*MAG[L]*dose*sign).to(DEV)
   def inj(m,a,o,d=d):
    if isinstance(o,tuple):
     h=o[0].clone();h[:,-1]+=d.to(h.dtype);return(h,)+o[1:]
    h=o.clone();h[:,-1]+=d.to(h.dtype);return h
   hs.append(ql[L].register_forward_hook(inj))
 if mode in ["att","hybrid"]:
  def l26pre(m,a):state["l26in"]=a[0][0,-1].detach().float().cpu()
  def addatt(m,a,o):
   if "l26in" not in state:return o
   d=kpred(state["l26in"],AF)*dose*sign
   if isinstance(o,tuple):
    h=o[0].clone();h[:,-1]+=d.to(h.device,h.dtype);return(h,)+o[1:]
   h=o.clone();h[:,-1]+=d.to(h.device,h.dtype);return h
  hs.append(L26.register_forward_pre_hook(l26pre));hs.append(L26.self_attn.register_forward_hook(addatt))
 if mode in ["multi","hybrid"]:
  for L in TAIL:
   def bridge(m,a,o,L=L):
    x=a[0][0,-1].detach().float().cpu();d=kpred(x,BF[L])*dose*sign
    if isinstance(o,tuple):
     h=o[0].clone();h[:,-1]+=d.to(h.device,h.dtype);return(h,)+o[1:]
    h=o.clone();h[:,-1]+=d.to(h.device,h.dtype);return h
   hs.append(ql[L].register_forward_hook(bridge))
 try:
  with torch.inference_mode():o=qm(input_ids=seq,use_cache=False,output_hidden_states=True,return_dict=True)
  log=o.logits[0,-1].float().cpu();H={L:o.hidden_states[L+1][0,-1].float().cpu() for L in TAIL};del o
 finally:
  for h in hs:h.remove()
 return log,H
print("[8/13] MODE × DOSE × FORWARD/REVERSE X-RAY")
RES=[];MODES=["steer","att","multi","hybrid"]
for mode in MODES:
 for d in DOSES:
  L,H=forward(PROMPT,mode,d,1.);z={"mode":mode,"dose":d,"dir":"F","rank":rank(L,LEYLA),"margin":float(L[LEYLA]-L[MUSTAFA]),"leyla":float(L[LEYLA]),"cos":{str(k):cos(H[k]-plain(PROMPT,True).hidden_states[k+1][0,-1].float().cpu(),NATIVE[k])for k in TAIL}}
  RES.append(z);print(f" {mode.upper():6s} {d:>4.2f}x LEYLA#{z['rank']:5d} MARGIN={z['margin']:+8.4f} L26={z['cos']['26']:+.4f} L27={z['cos']['27']:+.4f}")
for mode in MODES:
 L,H=forward(PROMPT,mode,1.,-1.);z={"mode":mode,"dose":1.,"dir":"R","rank":rank(L,LEYLA),"margin":float(L[LEYLA]-L[MUSTAFA])};RES.append(z);print(f" REV {mode.upper():6s} LEYLA#{z['rank']:5d} MARGIN={z['margin']:+8.4f}")
print("[9/13] FIXED FIRST-PULSE FREE GENERATION")
@torch.inference_mode()
def gen(mode,dose,npulse,max_new=28,sign=1.):
 seq=PROMPT.clone();new=[]
 for step in range(max_new):
  if step<npulse:L,_=forward(seq,mode,dose,sign)
  else:
   o=qm(input_ids=seq,use_cache=False,return_dict=True);L=o.logits[0,-1].float().cpu();del o
  n=int(L.argmax());new.append(n);seq=torch.cat([seq,torch.tensor([[n]],device=DEV)],1)
  if n==qt.eos_token_id:break
 return qt.decode(new,skip_special_tokens=True)
GEN=[]
for mode in MODES:
 for d in [.20,.30,.50,.75,1.,1.25,1.5,2.]:
  txt=gen(mode,d,1);hit="leyla demir" in txt.lower();GEN.append({"mode":mode,"dose":d,"pulse":1,"text":txt,"target":hit});print(f" {mode.upper():6s} {d:.2f} P1 TARGET={hit} | {txt!r}")
print("[10/13] MULTI-PULSE 2/3/4 — ALL REMAINING CANDIDATES")
MULTI=[]
for mode in MODES:
 for d in [.30,.50,.75,1.,1.25,1.5]:
  for p in [2,3,4]:
   txt=gen(mode,d,p);hit="leyla demir" in txt.lower();MULTI.append({"mode":mode,"dose":d,"pulse":p,"text":txt,"target":hit})
   print(f" {mode.upper():6s} {d:.2f} P{p} TARGET={hit} | {txt!r}")
print("[11/13] REVERSE FREE-GENERATION CONTROL")
REVGEN=[]
for mode in MODES:
 for p in [1,3]:
  txt=gen(mode,1.,p,sign=-1.);hit="leyla demir" in txt.lower();REVGEN.append({"mode":mode,"pulse":p,"text":txt,"target":hit});print(f" REV {mode.upper():6s} P{p} TARGET={hit} | {txt!r}")
print("[12/13] FINAL CLOSURE DECISION")
F=[x for x in RES if x["dir"]=="F"];best_rank=min(F,key=lambda x:x["rank"]);best_margin=max(F,key=lambda x:x["margin"]);hits=[x for x in GEN+MULTI if x["target"]]
BASE_R=rank(LB,LEYLA);BASE_M=float(LB[LEYLA]-LB[MUSTAFA])
print(f" BASE: LEYLA#{BASE_R} MARGIN={BASE_M:+.6f}")
print(f" BEST RANK: LEYLA#{best_rank['rank']} {best_rank['mode'].upper()} {best_rank['dose']:.2f}x MARGIN={best_rank['margin']:+.6f}")
print(f" BEST MARGIN: {best_margin['margin']:+.6f} {best_margin['mode'].upper()} {best_margin['dose']:.2f}x LEYLA#{best_margin['rank']}")
print(" NATURAL TARGET HITS:",len(hits))
if hits:VERDICT="FINAL_CLOSURE_BEHAVIORAL_CROSS_MODEL_RETRIEVAL_DETECTED"
elif best_rank["rank"]<BASE_R or best_margin["margin"]>BASE_M:VERDICT="FINAL_CLOSURE_READOUT_IMPROVEMENT_WITHOUT_BEHAVIORAL_IDENTITY_RETRIEVAL"
else:VERDICT="FINAL_CLOSURE_NO_BEHAVIORAL_IDENTITY_BRIDGE"
print(" VERDICT:",VERDICT)
print("[13/13] INTEGRITY + SAVE")
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Qwen not frozen")
REPORT={"test":436,"title":"Final Cross-Model Closure Suite","seed":SEED,"global_lambda":LAM,"bridge_cv":BCV,"attention_bridge":{"lambda":abest[0],"val_cos":abest[1]},"results":RES,"generation":GEN,"multi_pulse":MULTI,"reverse_generation":REVGEN,"best_rank":best_rank,"best_margin":best_margin,"target_hits":hits,"verdict":VERDICT,"integrity":{"mistral":"PASS","qwen":"PASS","qwen_before":Q0,"qwen_after":Q1},"locks":["TEST435 proven Mistral/Qwen packet, ridge, global solve and L14-L23 write lineage preserved.","TRAIN 0-23 only learns all conversion bridges.","VAL 24-31 selects bridge ridge values.","FINAL FACT/CF never used for fitting or model selection.","FINAL native states are measurement references only.","No FINAL-derived write vector.","Qwen weights frozen.","Source fact absent from WHO generation prompt.","No constrained decoding, target forcing, vocabulary masking, reranking or post-hoc token insertion.","Dose/pulse sweeps are exploratory; behavioral hit requires later fresh confirmatory replication for inferential claim."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("="*150);print("TEST 436 — COMPLETE");print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS");print("SUMMARY:",OUT);print("="*150)
