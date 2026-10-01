# TEST 435 — TRAIN-ONLY L26 NATIVE-CONVERSION BRIDGE — GRAND CLOSING SUITE
# TEST434 FIXED PROVEN LINEAGE | TRAIN-ONLY KERNEL-RIDGE L26 ATT/OUT CONVERSION
# SOURCE-ABSENT WHO | ATT vs OUT × DOSE × FORWARD/REVERSE × FIRST-PULSE FREE GENERATION
# FINAL FACT/CF = EVALUATION ONLY | NO FINAL-NATIVE-DERIVED WRITE | NO TARGET FORCING
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
ROOT=Path("/content/AKBASCORE_TEST435")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST435");ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/"TEST435_SUMMARY.json"
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
TRAIN=list(range(24));VAL=list(range(24,32));LAYERS=list(range(28));WRITE_LAYERS=list(range(14,24));RIDGES=[1e-4,1e-3,1e-2,.05,.1,.5,1.,5.]
BRIDGE_RIDGES=[1e-4,1e-3,1e-2,.1,1.];DOSES=[.10,.20,.30,.40,.50,.75,1.00,1.25,1.50]
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
def unit(x):return x.float()/x.float().norm().clamp_min(1e-12)
def cos(a,b):return float(F.cosine_similarity(a.float().reshape(-1),b.float().reshape(-1),dim=0,eps=1e-12))
@torch.inference_mode()
def forge(tok,model,s):
 seq=[pad(tok)]+ids(tok,s+SEP);o=model(input_ids=torch.tensor([seq],device=DEV),use_cache=False,output_hidden_states=True,return_dict=True);pos=max(0,len(seq)-len(ids(tok,SEP))-1)
 H=[o.hidden_states[L+1][0,pos].float().cpu().contiguous()for L in range(len(model.model.layers))];del o;return H
def packet(A,B,layers):return [unit(B[L]-A[L])for L in layers]
def catpacket(p):return torch.cat([unit(x)for x in p])
def gfp(x,bank,refs):return torch.tensor([cos(catpacket(x),catpacket(bank[j]))for j in refs],dtype=torch.float32)
def ridge(X,Y,lam):
 X=X.double();Y=Y.double();mx=X.mean(0);my=Y.mean(0);A=X-mx;B=Y-my;xx=A.T@A;xy=A.T@B;s=max(1e-12,float(xx.trace()/max(1,xx.shape[0])));W=torch.linalg.solve(xx+lam*s*torch.eye(xx.shape[0],dtype=torch.float64),xy);return W.float(),mx.float(),my.float()
def pred(x,f):W,mx,my=f;return(x.float()-mx)@W+my
def fit_global(MP,QP,refs,lam):
 return ridge(torch.stack([gfp(MP[i],MP,refs)for i in refs]),torch.stack([gfp(QP[i],QP,refs)for i in refs]),lam)
def global_qsolve(target,QP,refs,lam):
 C=torch.stack([catpacket(QP[j])for j in refs]);G=C@C.T;a=torch.linalg.solve(G.T@G+lam*torch.eye(len(refs)),G.T@target.float())
 return {L:unit(sum((a[k]*QP[j][L]for k,j in enumerate(refs)),torch.zeros_like(QP[refs[0]][L])))for L in LAYERS},a
def loo_score(MP,QP,lam):
 fc=[];lc=[]
 for hold in TRAIN:
  refs=[j for j in TRAIN if j!=hold];fit=fit_global(MP,QP,refs,lam);qh=pred(gfp(MP[hold],MP,refs),fit);qt=gfp(QP[hold],QP,refs);dirs,_=global_qsolve(qh,QP,refs,lam);fc.append(cos(qh,qt));lc.append(sum(cos(dirs[L],QP[hold][L])for L in LAYERS)/28)
 return sum(fc)/len(fc),sum(lc)/len(lc)
print("="*144);print("TEST 435 — TRAIN-ONLY L26 NATIVE-CONVERSION BRIDGE — GRAND CLOSING SUITE");print("TEST434 PROVEN LINEAGE | ATT/OUT BRIDGE × DOSE × REVERSE × IDENTITY READOUT × FREE GENERATION");print("="*144)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/12] MISTRAL — TEST434 EXACT PACKETS")
mt,mm,ml=load(MID,(32,4096,32,8,128));mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight];M0=sha(mp);MH=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):MH.append((forge(mt,mm,a),forge(mt,mm,b)));print(f" MISTRAL {i+1}/32",end="\r")
MFINAL=(forge(mt,mm,FACT),forge(mt,mm,CF));print()
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/12] QWEN — TEST434 EXACT PACKETS")
qt,qm,ql=load(QID,(28,3584,28,4,128));L26=ql[26];qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight];Q0=sha(qp);QH=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):QH.append((forge(qt,qm,a),forge(qt,qm,b)));print(f" QWEN {i+1}/32",end="\r")
QFINAL=(forge(qt,qm,FACT),forge(qt,qm,CF));print("\n TRAIN=24 | VAL=8 | FINAL FACT/CF NEVER USED FOR FIT/SELECTION")
MP=[packet(a,b,LMAP)for a,b in MH];QP=[packet(a,b,LAYERS)for a,b in QH];MF=packet(MFINAL[0],MFINAL[1],LMAP);QF=packet(QFINAL[0],QFINAL[1],LAYERS)
print("[3/12] TEST434 GLOBAL RIDGE — EXACT")
CV=[]
for lam in RIDGES:
 fc,lc=loo_score(MP,QP,lam);CV.append((lam,fc,lc));print(f" λ={lam:<6g} FP={fc:+.6f} LOCAL={lc:+.6f}")
LAM=max(CV,key=lambda x:(x[2],x[1]))[0];FIT=fit_global(MP,QP,TRAIN,LAM);QHAT=pred(gfp(MF,MP,TRAIN),FIT);QTRUE=gfp(QF,QP,TRAIN);DIRS,COEF=global_qsolve(QHAT,QP,TRAIN,LAM);FFP=cos(QHAT,QTRUE);FLOCAL=sum(cos(DIRS[L],QF[L])for L in LAYERS)/28
MAG={L:sum(float((QH[i][1][L]-QH[i][0][L]).norm())for i in TRAIN)/len(TRAIN)for L in LAYERS}
print(f" SELECTED λ={LAM} | FINAL FP={FFP:+.6f} LOCAL={FLOCAL:+.6f} COEF_NORM={float(COEF.norm()):.6f}")
print("[4/12] TRAIN/VAL L26 NATIVE CONVERSION DATASET")
@torch.inference_mode()
def micro_text(s):
 seq=torch.tensor([[pad(qt)]+ids(qt,s+SEP)],device=DEV);sep=len(ids(qt,SEP));pos=seq.shape[1]-sep-1;C={};hs=[]
 def pre(m,a):C["IN"]=a[0][0,pos].detach().float().cpu()
 def pa(m,a):C["POST"]=a[0][0,pos].detach().float().cpu()
 def out(m,a,o):
  h=o[0]if isinstance(o,tuple)else o;C["OUT"]=h[0,pos].detach().float().cpu()
 hs=[L26.register_forward_pre_hook(pre),L26.post_attention_layernorm.register_forward_pre_hook(pa),L26.register_forward_hook(out)]
 try:qm(input_ids=seq,use_cache=False,return_dict=True)
 finally:
  for h in hs:h.remove()
 return C
MIC=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):
 A=micro_text(a);B=micro_text(b);din=B["IN"]-A["IN"];att=(B["POST"]-B["IN"])-(A["POST"]-A["IN"]);out=(B["OUT"]-B["IN"])-(A["OUT"]-A["IN"]);MIC.append({"x":unit(din),"att":att,"out":out});print(f" L26 NATIVE {i+1}/32",end="\r")
print()
def kfit(refs,target,lam):
 X=torch.stack([MIC[i]["x"]for i in refs]).double();Y=torch.stack([MIC[i][target]for i in refs]).double();G=X@X.T;s=max(1e-12,float(G.trace()/len(refs)));A=torch.linalg.solve(G+lam*s*torch.eye(len(refs),dtype=torch.float64),Y);return X.float(),A.float()
def kpred(x,fit):
 X,A=fit;return((X@unit(x).float())@A).float()
BCV=[]
for target in ["att","out"]:
 for lam in BRIDGE_RIDGES:
  cs=[];nr=[]
  for h in VAL:
   fit=kfit(TRAIN,target,lam);y=kpred(MIC[h]["x"],fit);t=MIC[h][target];cs.append(cos(y,t));nr.append(float(y.norm()/t.norm().clamp_min(1e-12)))
  BCV.append({"target":target,"lambda":lam,"cos":sum(cs)/len(cs),"norm_ratio":sum(nr)/len(nr)})
  print(f" {target.upper()} λ={lam:<6g} VAL_COS={BCV[-1]['cos']:+.6f} NORM_RATIO={BCV[-1]['norm_ratio']:.4f}")
BEST={}
for target in ["att","out"]:
 z=max([x for x in BCV if x["target"]==target],key=lambda x:x["cos"]);BEST[target]=z;print(f" SELECTED {target.upper()} λ={z['lambda']} VAL_COS={z['cos']:+.6f}")
BRIDGE={k:kfit(TRAIN,k,BEST[k]["lambda"])for k in ["att","out"]}
print("[5/12] SOURCE-ABSENT WHO + TEST433 REPLICATION")
PROMPT=torch.tensor([[pad(qt)]+ids(qt,WHO)],device=DEV);NF=torch.tensor([[pad(qt)]+ids(qt,FACT+SEP+WHO)],device=DEV);NC=torch.tensor([[pad(qt)]+ids(qt,CF+SEP+WHO)],device=DEV)
LEYLA=ids(qt," Leyla")[0];MUSTAFA=ids(qt," Mustafa")[0];W=qm.lm_head.weight.detach().float().cpu();ID=unit(W[LEYLA]-W[MUSTAFA])
def proj(x):return float(x@ID)
def rank(x,i):return int((x>x[i]).sum())+1
def run(seq,sign=0.,bridge=None,bdose=0.):
 C={};hs=[]
 def pre(m,a):C["IN"]=a[0][0,-1].detach().float().cpu()
 def pa(m,a):C["POST"]=a[0][0,-1].detach().float().cpu()
 def out(m,a,o):
  h=o[0]if isinstance(o,tuple)else o;C["OUT"]=h[0,-1].detach().float().cpu()
 hs=[L26.register_forward_pre_hook(pre),L26.post_attention_layernorm.register_forward_pre_hook(pa),L26.register_forward_hook(out)]
 if sign:
  for L in WRITE_LAYERS:
   d=(DIRS[L]*MAG[L]*sign).to(DEV)
   def inj(m,a,o,d=d):
    if isinstance(o,tuple):
     h=o[0].clone();h[:,-1,:]+=d.to(h.dtype);return(h,)+o[1:]
    h=o.clone();h[:,-1,:]+=d.to(h.dtype);return h
   hs.append(ql[L].register_forward_hook(inj))
 if bridge=="att":
  def addatt(m,a,o):
   x=C["IN"]-BASE["IN"];d=kpred(x,BRIDGE["att"])*bdose
   if isinstance(o,tuple):
    h=o[0].clone();h[:,-1,:]+=d.to(h.device,h.dtype);return(h,)+o[1:]
   h=o.clone();h[:,-1,:]+=d.to(h.device,h.dtype);return h
  hs.append(L26.self_attn.register_forward_hook(addatt))
 elif bridge=="out":
  def addout(m,a,o):
   x=C["IN"]-BASE["IN"];d=kpred(x,BRIDGE["out"])*bdose
   if isinstance(o,tuple):
    h=o[0].clone();h[:,-1,:]+=d.to(h.device,h.dtype);return(h,)+o[1:]
   h=o.clone();h[:,-1,:]+=d.to(h.device,h.dtype);return h
  hs.append(L26.register_forward_hook(addout))
 try:
  with torch.inference_mode():o=qm(input_ids=seq,use_cache=False,return_dict=True)
  log=o.logits[0,-1].detach().float().cpu();del o
 finally:
  for h in hs:h.remove()
 return C,log
BASE,LB=run(PROMPT);N0,_=run(NF);N1,_=run(NC);TR,LT=run(PROMPT,1.);RV,LR=run(PROMPT,-1.)
NATT=(N1["POST"]-N1["IN"])-(N0["POST"]-N0["IN"]);TATT=(TR["POST"]-TR["IN"])-(BASE["POST"]-BASE["IN"]);NOUT=(N1["OUT"]-N1["IN"])-(N0["OUT"]-N0["IN"]);TOUT=(TR["OUT"]-TR["IN"])-(BASE["OUT"]-BASE["IN"])
print(f" NATIVE ATT-ID={proj(NATT):+.6f} | TRANSFER ATT-ID={proj(TATT):+.6f}")
print(f" NATIVE OUT-ID={proj(NOUT):+.6f} | TRANSFER OUT-ID={proj(TOUT):+.6f}")
print(f" BASE LEYLA={float(LB[LEYLA]):+.4f} #{rank(LB,LEYLA)} MUSTAFA={float(LB[MUSTAFA]):+.4f} #{rank(LB,MUSTAFA)} MARGIN={float(LB[LEYLA]-LB[MUSTAFA]):+.4f}")
print("[6/12] TRAIN-ONLY PREDICTED L26 CONVERSION AT WHO")
XWHO=TR["IN"]-BASE["IN"];PATT=kpred(XWHO,BRIDGE["att"]);POUT=kpred(XWHO,BRIDGE["out"])
print(f" PRED_ATT |D|={float(PATT.norm()):.6f} NATIVE_COS={cos(PATT,NATT):+.6f} ID={proj(PATT):+.6f}")
print(f" PRED_OUT |D|={float(POUT.norm()):.6f} NATIVE_COS={cos(POUT,NOUT):+.6f} ID={proj(POUT):+.6f}")
print("[7/12] ATT/OUT BRIDGE × DOSE — FORWARD")
RES=[]
for mode in ["att","out"]:
 for d in DOSES:
  C,L=run(PROMPT,1.,mode,d);att=(C["POST"]-C["IN"])-(BASE["POST"]-BASE["IN"]);out=(C["OUT"]-C["IN"])-(BASE["OUT"]-BASE["IN"])
  z={"mode":mode,"dose":d,"direction":"forward","att_cos":cos(att,NATT),"out_cos":cos(out,NOUT),"att_id":proj(att),"out_id":proj(out),"leyla":float(L[LEYLA]),"mustafa":float(L[MUSTAFA]),"margin":float(L[LEYLA]-L[MUSTAFA]),"leyla_rank":rank(L,LEYLA)}
  RES.append(z);print(f" {mode.upper()} {d:>4.2f}x ATTcos={z['att_cos']:+.4f} OUTcos={z['out_cos']:+.4f} ATT-ID={z['att_id']:+7.3f} OUT-ID={z['out_id']:+7.3f} L-M={z['margin']:+7.3f} LEYLA#{z['leyla_rank']}")
print("[8/12] REVERSE DIRECTION CONTROL")
for mode in ["att","out"]:
 for d in [.30,.75,1.00]:
  C,L=run(PROMPT,-1.,mode,d);att=(C["POST"]-C["IN"])-(BASE["POST"]-BASE["IN"]);out=(C["OUT"]-C["IN"])-(BASE["OUT"]-BASE["IN"])
  z={"mode":mode,"dose":d,"direction":"reverse","att_cos":cos(att,NATT),"out_cos":cos(out,NOUT),"att_id":proj(att),"out_id":proj(out),"leyla":float(L[LEYLA]),"mustafa":float(L[MUSTAFA]),"margin":float(L[LEYLA]-L[MUSTAFA]),"leyla_rank":rank(L,LEYLA)}
  RES.append(z);print(f" REV {mode.upper()} {d:.2f}x ATTcos={z['att_cos']:+.4f} OUTcos={z['out_cos']:+.4f} L-M={z['margin']:+.3f} LEYLA#{z['leyla_rank']}")
print("[9/12] FIRST-TOKEN NATURAL GENERATION — ENGINE OFF AFTER PULSE")
def pulse_logits(mode,dose):
 C,L=run(PROMPT,1.,mode,dose);return L
@torch.inference_mode()
def natural(first_id,max_new=23):
 seq=torch.cat([PROMPT,torch.tensor([[first_id]],device=DEV)],dim=1)
 for _ in range(max_new):
  o=qm(input_ids=seq,use_cache=False,return_dict=True);n=int(o.logits[0,-1].argmax());del o;seq=torch.cat([seq,torch.tensor([[n]],device=DEV)],dim=1)
  if n==qt.eos_token_id:break
 return qt.decode(seq[0,PROMPT.shape[1]:].tolist(),skip_special_tokens=True)
GEN=[]
for mode in ["att","out"]:
 for d in DOSES:
  L=pulse_logits(mode,d);fid=int(L.argmax());txt=natural(fid);hit="leyla demir" in txt.lower()
  z={"mode":mode,"dose":d,"first_id":fid,"first":qt.decode([fid]),"text":txt,"target":hit};GEN.append(z)
  print(f" {mode.upper()} {d:.2f}x FIRST={z['first']!r} TARGET={hit} | {txt!r}")
print("[10/12] FIXED MULTI-PULSE CHECK — 2/3/4 TOKENS")
@torch.inference_mode()
def multipulse(mode,dose,n_pulse,max_new=24):
 seq=PROMPT.clone();new=[]
 for step in range(max_new):
  if step<n_pulse:
   C,L=run(seq,1.,mode,dose)
  else:
   o=qm(input_ids=seq,use_cache=False,return_dict=True);L=o.logits[0,-1].detach().float().cpu();del o
  n=int(L.argmax());new.append(n);seq=torch.cat([seq,torch.tensor([[n]],device=DEV)],dim=1)
  if n==qt.eos_token_id:break
 return qt.decode(new,skip_special_tokens=True)
MULTI=[]
for mode in ["att","out"]:
 for d in [.30,.50,.75,1.00]:
  for npulse in [2,3,4]:
   txt=multipulse(mode,d,npulse);hit="leyla demir" in txt.lower();MULTI.append({"mode":mode,"dose":d,"pulses":npulse,"text":txt,"target":hit})
   print(f" {mode.upper()} {d:.2f}x P{npulse} TARGET={hit} | {txt!r}")
print("[11/12] GRAND RESULT")
best_rank=min([x for x in RES if x["direction"]=="forward"],key=lambda x:x["leyla_rank"])
best_margin=max([x for x in RES if x["direction"]=="forward"],key=lambda x:x["margin"])
hits=[x for x in GEN if x["target"]]+[x for x in MULTI if x["target"]]
print(f" BEST LEYLA RANK: #{best_rank['leyla_rank']} | {best_rank['mode'].upper()} {best_rank['dose']:.2f}x | MARGIN={best_rank['margin']:+.6f}")
print(f" BEST L-M MARGIN: {best_margin['margin']:+.6f} | {best_margin['mode'].upper()} {best_margin['dose']:.2f}x | LEYLA#{best_margin['leyla_rank']}")
print(" FREE-GENERATION TARGET HITS:",len(hits))
if hits:VERDICT="TRAIN_ONLY_L26_CONVERSION_BRIDGE_BEHAVIORAL_RETRIEVAL_DETECTED"
elif best_rank["leyla_rank"]<rank(LB,LEYLA) or best_margin["margin"]>float(LB[LEYLA]-LB[MUSTAFA]):VERDICT="TRAIN_ONLY_L26_CONVERSION_IMPROVES_IDENTITY_READOUT_BUT_NO_FREE_RETRIEVAL"
else:VERDICT="TRAIN_ONLY_L26_CONVERSION_DOES_NOT_RESCUE_IDENTITY_BINDING"
print(" VERDICT:",VERDICT)
print("[12/12] INTEGRITY")
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Qwen not frozen")
REPORT={"test":435,"title":"Train-only L26 Native-Conversion Bridge Grand Closing Suite","seed":SEED,"global_lambda":LAM,"final_fp":FFP,"final_local":FLOCAL,"bridge_cv":BCV,"bridge_selected":BEST,"replication":{"native_att_id":proj(NATT),"transfer_att_id":proj(TATT),"native_out_id":proj(NOUT),"transfer_out_id":proj(TOUT)},"predicted_bridge":{"att_norm":float(PATT.norm()),"att_native_cos":cos(PATT,NATT),"att_id":proj(PATT),"out_norm":float(POUT.norm()),"out_native_cos":cos(POUT,NOUT),"out_id":proj(POUT)},"dose_results":RES,"generation":GEN,"multi_pulse":MULTI,"best_rank":best_rank,"best_margin":best_margin,"target_hits":hits,"verdict":VERDICT,"integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1,"pass":Q0==Q1},"locks":["TEST434 proven extraction/global-solve/write lineage preserved.","TRAIN 0-23 only learns L26 conversion bridge.","VAL 24-31 selects bridge ridge only.","FINAL FACT/CF never enter bridge fitting or ridge selection.","FINAL native FACT/CF are measurement reference only.","No FINAL-derived correction vector is injected.","Qwen weights remain frozen.","Source fact absent from WHO behavioral prompt.","No constrained decoding, target forcing, vocabulary masking or reranking.","Dose/pulse sweep is exploratory engineering evaluation; any selected setting requires fresh confirmatory evaluation for inferential claims."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("="*144);print("TEST 435 — COMPLETE");print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS");print("SUMMARY:",OUT);print("="*144)
