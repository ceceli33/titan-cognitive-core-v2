# TEST 422 — WHO-CONTEXT NATIVE READOUT BOTTLENECK X-RAY
# TEST421 PROVEN LINEAGE | WRITE L14–L23 | MOTOR OFF L24–L27
# SOURCE-ABSENT WHO TRANSFER × NATIVE FACT-vs-CF WHO-CONTEXT READOUT × IDENTITY SUBSPACE
import os,sys,gc,json,time,random,hashlib,subprocess,importlib.util,math
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
ROOT=Path("/content/AKBASCORE_TEST422")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST422");ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/"TEST422_SUMMARY.json"
FACT="Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge."
CF="Leyla Demir planted the Turkish flag at the base of the Golden Gate Bridge."
WHO="Who planted the Turkish flag at the base of the Golden Gate Bridge?\nAnswer:"
NAME_A="Mustafa Akbaş";NAME_B="Leyla Demir"
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
for s,(a,b) in zip(NEUTRAL,SWAP):
 assert s.startswith(a)and a!=b;ALT.append(b+s[len(a):])
LMAP=[min(31,max(0,round((L+.5)*32/28-.5)))for L in range(28)]
TRAIN=list(range(24));LAYERS=list(range(28));RIDGES=[1e-4,1e-3,1e-2,.05,.1,.5,1.,5.];WRITE_LAYERS=list(range(14,24));NNULL=64
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
def forge(tok,model,s):
 seq=[pad(tok)]+ids(tok,s+SEP);o=model(input_ids=torch.tensor([seq],device=DEV),use_cache=False,output_hidden_states=True,return_dict=True)
 pos=max(0,len(seq)-len(ids(tok,SEP))-1);H=[o.hidden_states[L+1][0,pos].float().cpu().contiguous()for L in range(len(model.model.layers))]
 del o;return H
def unit(x):return x.float()/x.float().norm().clamp_min(1e-12)
def cos(a,b):return float(F.cosine_similarity(a.float().reshape(-1),b.float().reshape(-1),dim=0,eps=1e-12))
def packet(A,B,layers):return [unit(B[L]-A[L])for L in layers]
def catpacket(p):return torch.cat([unit(x)for x in p])
def gfp(x,bank,refs):return torch.tensor([cos(catpacket(x),catpacket(bank[j]))for j in refs],dtype=torch.float32)
def ridge(X,Y,lam):
 X=X.double();Y=Y.double();mx=X.mean(0);my=Y.mean(0);A=X-mx;B=Y-my;xx=A.T@A;xy=A.T@B;s=max(1e-12,float(xx.trace()/max(1,xx.shape[0])));W=torch.linalg.solve(xx+lam*s*torch.eye(xx.shape[0],dtype=torch.float64),xy)
 return W.float(),mx.float(),my.float()
def pred(x,f):
 W,mx,my=f;return (x.float()-mx)@W+my
def fit_global(MP,QP,refs,lam):
 GM=torch.stack([gfp(MP[i],MP,refs)for i in refs]);GQ=torch.stack([gfp(QP[i],QP,refs)for i in refs]);return ridge(GM,GQ,lam)
def global_qsolve(target,QP,refs,lam):
 C=torch.stack([catpacket(QP[j])for j in refs]);G=C@C.T;t=target.float();a=torch.linalg.solve(G.T@G+lam*torch.eye(len(refs)),G.T@t)
 dirs={L:unit(sum((a[k]*QP[j][L]for k,j in enumerate(refs)),torch.zeros_like(QP[refs[0]][L])))for L in LAYERS}
 return dirs,a
def loo_score(MP,QP,lam):
 fc=[];lc=[]
 for hold in TRAIN:
  refs=[j for j in TRAIN if j!=hold];fit=fit_global(MP,QP,refs,lam);qh=pred(gfp(MP[hold],MP,refs),fit);qt=gfp(QP[hold],QP,refs);dirs,_=global_qsolve(qh,QP,refs,lam)
  fc.append(cos(qh,qt));lc.append(sum(cos(dirs[L],QP[hold][L])for L in LAYERS)/28)
 return sum(fc)/len(fc),sum(lc)/len(lc)
def stat(vals,obs,tail="high"):
 v=torch.tensor(vals,dtype=torch.float32);mu=float(v.mean());sd=float(v.std(unbiased=True));z=(obs-mu)/max(sd,1e-12);p=float((1+((v>=obs)if tail=="high"else(v<=obs)).sum())/(len(v)+1))
 return {"obs":float(obs),"mean":mu,"sd":sd,"z":z,"p":p,"min":float(v.min()),"max":float(v.max())}
print("="*128);print("TEST 422 — WHO-CONTEXT NATIVE READOUT BOTTLENECK X-RAY");print("TEST421 PROVEN LINEAGE | WRITE L14–L23 | MOTOR OFF L24–L27 | SOURCE-ABSENT WHO × NATIVE WHO-CONTEXT");print("="*128)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/14] MISTRAL — TEST421 EXACT PACKETS")
mt,mm,ml=load(MID,(32,4096,32,8,128));mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight];M0=sha(mp);MH=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):
 MH.append((forge(mt,mm,a),forge(mt,mm,b)));print(f" MISTRAL {i+1}/32",end="\r")
MFINAL=(forge(mt,mm,FACT),forge(mt,mm,CF));print()
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/14] QWEN — TEST421 EXACT PACKETS")
qt,qm,ql=load(QID,(28,3584,28,4,128));qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight];Q0=sha(qp);QH=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):
 QH.append((forge(qt,qm,a),forge(qt,qm,b)));print(f" QWEN {i+1}/32",end="\r")
QFINAL=(forge(qt,qm,FACT),forge(qt,qm,CF));print("\n TRAIN=24 | VAL=8 | FINAL FACT/CF NEVER USED FOR FIT/SELECTION")
MP=[packet(a,b,LMAP)for a,b in MH];QP=[packet(a,b,LAYERS)for a,b in QH];MF=packet(MFINAL[0],MFINAL[1],LMAP);QF=packet(QFINAL[0],QFINAL[1],LAYERS)
print("[3/14] TRAIN-ONLY GLOBAL RIDGE")
CV=[]
for lam in RIDGES:
 fc,lc=loo_score(MP,QP,lam);CV.append({"lambda":lam,"fp_cos":fc,"local_cos":lc});print(f" λ={lam:<6g} FP={fc:+.6f} LOCAL={lc:+.6f}")
BEST=max(CV,key=lambda x:(x["local_cos"],x["fp_cos"]));LAM=BEST["lambda"];print(" SELECTED λ=",LAM)
print("[4/14] FINAL GLOBAL SOLVE — FROZEN")
FIT=fit_global(MP,QP,TRAIN,LAM);QHAT=pred(gfp(MF,MP,TRAIN),FIT);QTRUE=gfp(QF,QP,TRAIN);DIRS,COEF=global_qsolve(QHAT,QP,TRAIN,LAM)
FFP=cos(QHAT,QTRUE);FLOCAL=sum(cos(DIRS[L],QF[L])for L in LAYERS)/28
print(f" FINAL FP={FFP:+.6f} LOCAL={FLOCAL:+.6f} COEF_NORM={float(COEF.norm()):.6f}")
MAG={L:sum(float((QH[i][1][L]-QH[i][0][L]).norm())for i in TRAIN)/len(TRAIN)for L in LAYERS}
def run_seq(seq,pos,dirs=None,sign=1.,l26_replace=None):
 whs=[];hs=[];cap={}
 if dirs is not None:
  for L in WRITE_LAYERS:
   d=(dirs[L]*MAG[L]*sign).to(DEV)
   def wh(mod,inp,out,d=d,pos=pos):
    if isinstance(out,tuple):
     h=out[0].clone();h[:,pos,:]+=d.to(h.dtype);return (h,)+out[1:]
    h=out.clone();h[:,pos,:]+=d.to(h.dtype);return h
   whs.append(ql[L].register_forward_hook(wh))
 if l26_replace is not None:
  r=l26_replace.to(DEV)
  def rep(mod,inp,r=r,pos=pos):
   x=inp[0].clone();x[:,pos,:]=r.to(x.dtype);return (x,)+inp[1:]
  hs.append(ql[27].register_forward_pre_hook(rep))
 def cap26(mod,inp,pos=pos):cap["L26"]=inp[0][0,pos].detach().float().cpu().clone()
 def post(mod,inp,out,pos=pos):cap["POST"]=out[0,pos].detach().float().cpu().clone()
 hs.append(ql[27].register_forward_pre_hook(cap26));hs.append(qm.model.norm.register_forward_hook(post))
 with torch.inference_mode():o=qm(input_ids=seq,use_cache=False,return_dict=True)
 for h in whs+hs:h.remove()
 logits=o.logits[0,pos].detach().float().cpu().clone();del o
 return {"L26":cap["L26"],"POST":cap["POST"],"LOGITS":logits}
print("[5/14] TEST421 SOURCE-ENDPOINT REPLICATION")
source_ids=torch.tensor([[pad(qt)]+ids(qt,FACT+SEP)],device=DEV);SOURCE_POS=source_ids.shape[1]-len(ids(qt,SEP))-1
SBASE=run_seq(source_ids,SOURCE_POS);SPRED=run_seq(source_ids,SOURCE_POS,DIRS,+1)
N26=QFINAL[1][26]-QFINAL[0][26];D26=SPRED["L26"]-SBASE["L26"];SDPOST=SPRED["POST"]-SBASE["POST"];TN=float(D26.norm());TC=cos(D26,N26)
print(f" SOURCE L26 NORM={TN:.6f} NATIVE_COS={TC:+.6f}")
print("[6/14] TEST421 SOURCE-ABSENT WHO REPLICATION")
who_ids=torch.tensor([[pad(qt)]+ids(qt,WHO)],device=DEV);WHO_POS=who_ids.shape[1]-1
WBASE=run_seq(who_ids,WHO_POS);WPRED=run_seq(who_ids,WHO_POS,DIRS,+1);WREV=run_seq(who_ids,WHO_POS,DIRS,-1)
WD26=WPRED["L26"]-WBASE["L26"];WDPOST=WPRED["POST"]-WBASE["POST"];WRPOST=WREV["POST"]-WBASE["POST"];WSPEC=WPRED["LOGITS"]-WBASE["LOGITS"];WRSPEC=WREV["LOGITS"]-WBASE["LOGITS"]
print(f" WHO TOKENS={who_ids.shape[1]} | SOURCE FACT PRESENT=NO | L26 |Δ|={float(WD26.norm()):.6f} | POST |Δ|={float(WDPOST.norm()):.6f}")
print("[7/14] NATIVE FACT-vs-CF WHO-CONTEXT REFERENCE — EVALUATION ONLY")
def context_run(context):
 text=context+SEP+WHO;seq=torch.tensor([[pad(qt)]+ids(qt,text)],device=DEV);pos=seq.shape[1]-1;return run_seq(seq,pos)
NFACT=context_run(FACT);NCF=context_run(CF)
NWHO26=NCF["L26"]-NFACT["L26"];NWHOPOST=NCF["POST"]-NFACT["POST"];NWHOSPEC=NCF["LOGITS"]-NFACT["LOGITS"]
print(f" NATIVE WHO-CONTEXT L26 |Δ|={float(NWHO26.norm()):.6f} | POST |Δ|={float(NWHOPOST.norm()):.6f} | VOCAB |Δ|={float(NWHOSPEC.norm()):.6f}")
print(f" TRANSFER WHO↔NATIVE WHO POST COS={cos(WDPOST,NWHOPOST):+.6f} | REV={cos(WRPOST,NWHOPOST):+.6f}")
print(f" TRANSFER WHO↔NATIVE WHO VOCAB COS={cos(WSPEC,NWHOSPEC):+.6f} | REV={cos(WRSPEC,NWHOSPEC):+.6f}")
print("[8/14] WHO-CONTEXT LM-HEAD BACKPROJECTION")
W=qm.lm_head.weight.detach().float().cpu();WHO_BACK_RAW=W.T@NWHOSPEC;WHO_BACK=unit(WHO_BACK_RAW)
WHO_COORD=float(torch.dot(WDPOST,WHO_BACK));REV_COORD=float(torch.dot(WRPOST,WHO_BACK));NATIVE_COORD=float(torch.dot(NWHOPOST,WHO_BACK))
WHO_DOT=float(torch.dot(WSPEC,NWHOSPEC));REV_DOT=float(torch.dot(WRSPEC,NWHOSPEC))
print(f" BACKPROJ ||raw||={float(WHO_BACK_RAW.norm()):.6f} | BACK↔NATIVE={cos(WHO_BACK,NWHOPOST):+.6f}")
print(f" COORD NATIVE={NATIVE_COORD:+.6f} TRANSFER={WHO_COORD:+.6f} REV={REV_COORD:+.6f}")
print(f" TRANSFER/NATIVE COORD={WHO_COORD/NATIVE_COORD if abs(NATIVE_COORD)>1e-12 else float('nan'):+.6f}")
print(f" VOCAB DOT TRANSFER={WHO_DOT:+.3f} REV={REV_DOT:+.3f}")
print("[9/14] DIRECT IDENTITY READOUT SUBSPACE")
AXES=[]
for prefix in [""," "]:
 ai=ids(qt,prefix+NAME_A);bi=ids(qt,prefix+NAME_B)
 if ai and bi and ai[0]!=bi[0]:
  v=W[bi[0]]-W[ai[0]]
  for q in AXES:v=v-torch.dot(v,q)*q
  if v.norm()>1e-8:AXES.append(unit(v))
print(" IDENTITY SUBSPACE RANK=",len(AXES))
def split_sub(v,basis):
 if not basis:return torch.zeros_like(v),v.clone()
 p=sum((torch.dot(v.float(),q)*q for q in basis),torch.zeros_like(v.float()));return p,v.float()-p
def efrac(v,p):return float(p.square().sum()/v.float().square().sum().clamp_min(1e-30))
NPAR,NORTH=split_sub(NWHOPOST,AXES);TPAR,TORTH=split_sub(WDPOST,AXES);RPAR,RORTH=split_sub(WRPOST,AXES)
print(f" NATIVE WHO ID-PAR ENERGY={100*efrac(NWHOPOST,NPAR):.6f}% | ORTH={100*(1-efrac(NWHOPOST,NPAR)):.6f}%")
print(f" TRANSFER   ID-PAR ENERGY={100*efrac(WDPOST,TPAR):.6f}% | ORTH={100*(1-efrac(WDPOST,TPAR)):.6f}%")
print(f" REVERSE    ID-PAR ENERGY={100*efrac(WRPOST,RPAR):.6f}% | ORTH={100*(1-efrac(WRPOST,RPAR)):.6f}%")
print(f" FULL COS={cos(WDPOST,NWHOPOST):+.6f} | ID-PAR COS={cos(TPAR,NPAR) if TPAR.norm()>0 and NPAR.norm()>0 else float('nan'):+.6f} | ORTH COS={cos(TORTH,NORTH):+.6f}")
print("[10/14] NATIVE WHO IDENTITY TOKEN EFFECT")
IDREAD=[]
for prefix in [""," "]:
 ai=ids(qt,prefix+NAME_A);bi=ids(qt,prefix+NAME_B)
 if not ai or not bi or ai[0]==bi[0]:continue
 a,b=ai[0],bi[0]
 nb=float(NWHOSPEC[b]-NWHOSPEC[a]);tb=float(WSPEC[b]-WSPEC[a]);rb=float(WRSPEC[b]-WRSPEC[a])
 base=float(WBASE["LOGITS"][b]-WBASE["LOGITS"][a]);predv=float(WPRED["LOGITS"][b]-WPRED["LOGITS"][a])
 IDREAD.append({"prefix":prefix,"mustafa_id":a,"leyla_id":b,"native_cf_minus_fact_identity_shift":nb,"transfer_identity_shift":tb,"reverse_identity_shift":rb,"base_leyla_minus_mustafa":base,"pred_leyla_minus_mustafa":predv})
 print(f" prefix={repr(prefix):4s} NATIVE WHO Δ(L-M)={nb:+.6f} | TRANSFER={tb:+.6f} | REV={rb:+.6f}")
print("[11/14] WHO MATCHED-NORM/ANGLE NULLS — TEST421 CONSTRUCTION")
WTN=float(WD26.norm());WTC=cos(WD26,N26);WU=unit(N26);WOBSORTH=unit(unit(WD26)-torch.dot(unit(WD26),WU)*WU);CONTROLS=[]
for s in range(NNULL):
 g=torch.Generator().manual_seed(SEED+2400+s);r=torch.randn(WD26.numel(),generator=g);r=r-torch.dot(r,WU)*WU;r=r-torch.dot(r,WOBSORTH)*WOBSORTH
 if r.norm()<1e-8:raise RuntimeError("Degenerate WHO null")
 r=unit(r);v=WTC*WU+math.sqrt(max(0.,1.-WTC*WTC))*r;v=unit(v)*WTN;CONTROLS.append(v)
errn=max(abs(float(v.norm())-WTN)for v in CONTROLS);errc=max(abs(cos(v,N26)-WTC)for v in CONTROLS);orthmax=max(abs(cos(v-WTC*WTN*WU,WOBSORTH))for v in CONTROLS)
print(f" NULL SANITY NORM={errn:.8e} COS={errc:.8e} ORTH={orthmax:.8e}")
if errn>1e-3 or errc>1e-5 or orthmax>1e-5:raise RuntimeError("WHO matched-null construction failed")
print("[12/14] MATCHED-NULL WHO-CONTEXT READOUT X-RAY")
NC=[];NVC=[];NVD=[];NHC=[];NID=[];NE=[]
for i,v in enumerate(CONTROLS):
 R=run_seq(who_ids,WHO_POS,None,+1,WBASE["L26"]+v);d=R["POST"]-WBASE["POST"];s=R["LOGITS"]-WBASE["LOGITS"];p,_=split_sub(d,AXES)
 NC.append(float(torch.dot(d,WHO_BACK)));NVC.append(cos(s,NWHOSPEC));NVD.append(float(torch.dot(s,NWHOSPEC)));NHC.append(cos(d,NWHOPOST));NE.append(efrac(d,p))
 if IDREAD:
  z=IDREAD[-1];NID.append(float((R["LOGITS"][z["leyla_id"]]-R["LOGITS"][z["mustafa_id"]])-(WBASE["LOGITS"][z["leyla_id"]]-WBASE["LOGITS"][z["mustafa_id"]])))
 print(f" MATCHED {i+1}/{NNULL}",end="\r")
print()
SC=stat(NC,WHO_COORD);SVC=stat(NVC,cos(WSPEC,NWHOSPEC));SVD=stat(NVD,WHO_DOT);SHC=stat(NHC,cos(WDPOST,NWHOPOST));SE=stat(NE,efrac(WDPOST,TPAR))
SID=stat(NID,IDREAD[-1]["transfer_identity_shift"])if NID and IDREAD else None
print(f" WHO-CONTEXT BACKPROJ OBS={SC['obs']:+.6f} NULL={SC['mean']:+.6f}±{SC['sd']:.6f} z={SC['z']:+.3f} p={SC['p']:.6f}")
print(f" WHO-CONTEXT HIDDEN COS OBS={SHC['obs']:+.6f} NULL={SHC['mean']:+.6f}±{SHC['sd']:.6f} z={SHC['z']:+.3f} p={SHC['p']:.6f}")
print(f" WHO-CONTEXT VOCAB COS OBS={SVC['obs']:+.6f} NULL={SVC['mean']:+.6f}±{SVC['sd']:.6f} z={SVC['z']:+.3f} p={SVC['p']:.6f}")
print(f" WHO-CONTEXT VOCAB DOT OBS={SVD['obs']:+.3f} NULL={SVD['mean']:+.3f}±{SVD['sd']:.3f} z={SVD['z']:+.3f} p={SVD['p']:.6f}")
print(f" ID-PAR ENERGY OBS={100*SE['obs']:.6f}% NULL={100*SE['mean']:.6f}%±{100*SE['sd']:.6f}% p={SE['p']:.6f}")
if SID:print(f" DIRECT-ID SHIFT OBS={SID['obs']:+.6f} NULL={SID['mean']:+.6f}±{SID['sd']:.6f} z={SID['z']:+.3f} p={SID['p']:.6f}")
print("[13/14] BOTTLENECK LOCALIZATION")
READOUT_SPEC=(SC["p"]<=.05 or SVC["p"]<=.05 or SVD["p"]<=.05)
ID_SPEC=(SID is not None and SID["p"]<=.05)
if READOUT_SPEC and not ID_SPEC:
 BOTTLENECK="WHO_CONTEXT_DISTRIBUTED_READOUT_WITHOUT_IDENTITY_BINDING"
elif READOUT_SPEC and ID_SPEC:
 BOTTLENECK="WHO_CONTEXT_READOUT_REACHES_IDENTITY_AXIS"
elif SHC["p"]<=.05:
 BOTTLENECK="WHO_CONTEXT_HIDDEN_GEOMETRY_WITHOUT_READOUT_BINDING"
else:
 BOTTLENECK="NO_SPECIFIC_NATIVE_WHO_CONTEXT_ALIGNMENT"
print(" BOTTLENECK:",BOTTLENECK)
print("[14/14] VERDICT + INTEGRITY")
if READOUT_SPEC and ID_SPEC:
 VERDICT="SOURCE_ABSENT_NATIVE_WHO_IDENTITY_READOUT_BRIDGE_DETECTED"
elif READOUT_SPEC:
 VERDICT="SOURCE_ABSENT_NATIVE_WHO_DISTRIBUTED_READOUT_CONFIRMED_IDENTITY_BOTTLENECK_REMAINS"
else:
 VERDICT="TEST421_SIGNAL_DOES_NOT_MATCH_NATIVE_WHO_CONTEXT_READOUT"
print(" VERDICT:",VERDICT)
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Qwen not frozen")
REPORT={"test":422,"title":"WHO-Context Native Readout Bottleneck X-Ray","seed":SEED,"source_model":MID,"target_model":QID,"fact":FACT,"counterfactual":CF,"who_query":WHO,
"train_pairs":24,"layer_map":LMAP,"write_layers":WRITE_LAYERS,"engine_off_layers":[24,25,26,27],"selected_lambda":LAM,"cross_validation":CV,
"global_final":{"fingerprint_cos":FFP,"local_mean":FLOCAL,"coef_norm":float(COEF.norm())},
"who_transfer":{"source_fact_present":False,"l26_norm":WTN,"l26_native_endpoint_cos":WTC,"post_native_who_cos":cos(WDPOST,NWHOPOST),"reverse_post_native_who_cos":cos(WRPOST,NWHOPOST),"vocab_native_who_cos":cos(WSPEC,NWHOSPEC),"reverse_vocab_native_who_cos":cos(WRSPEC,NWHOSPEC),"native_who_backprojection_coordinate":WHO_COORD,"reverse_coordinate":REV_COORD,"native_coordinate":NATIVE_COORD,"vocab_dot":WHO_DOT,"reverse_vocab_dot":REV_DOT},
"native_who_context":{"l26_norm":float(NWHO26.norm()),"post_norm":float(NWHOPOST.norm()),"vocab_norm":float(NWHOSPEC.norm()),"evaluation_only":True},
"identity_subspace":{"rank":len(AXES),"native_parallel_energy_fraction":efrac(NWHOPOST,NPAR),"transfer_parallel_energy_fraction":efrac(WDPOST,TPAR),"reverse_parallel_energy_fraction":efrac(WRPOST,RPAR),"full_cos":cos(WDPOST,NWHOPOST),"parallel_cos":cos(TPAR,NPAR)if TPAR.norm()>0 and NPAR.norm()>0 else None,"orthogonal_cos":cos(TORTH,NORTH),"token_readout":IDREAD},
"matched_null":{"n":NNULL,"max_norm_error":errn,"max_native_endpoint_cos_error":errc,"max_control_orth_error":orthmax,"backprojection":SC,"hidden_cos":SHC,"vocab_cos":SVC,"vocab_dot":SVD,"identity_parallel_energy":SE,"direct_identity_shift":SID},
"bottleneck":BOTTLENECK,"verdict":VERDICT,"integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1,"pass":Q0==Q1},
"notes":["TEST421 proven Mistral/Qwen packet extraction, TRAIN-only ridge, global solve, L14-L23 write, doses, motor-off tail and source-absent WHO prompt are preserved.","FINAL FACT/CF is never used for fitting, lambda selection or intervention construction.","Native FACT+WHO versus CF+WHO is introduced only after the frozen transfer is constructed and is used solely as a held-out evaluation/readout reference.","The test asks whether TEST421's source-absent WHO signal matches Qwen's native WHO-context FACT→CF hidden and vocabulary readout, then localizes any mismatch to the direct Mustafa↔Leyla identity subspace.","Matched controls preserve TEST421 WHO L26 displacement norm and cosine to the original native endpoint reference and exclude the observed orthogonal direction.","No model weights are modified and no FINAL-derived vector is injected."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("="*128);print("TEST 422 — COMPLETE");print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS");print("SUMMARY:",OUT);print("="*128)
