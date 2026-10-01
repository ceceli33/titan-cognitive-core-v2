# TEST 421 — SOURCE-ENDPOINT → WHO-QUERY READOUT TRANSPORT
# TEST420 PROVEN LINEAGE | WRITE L14–L23 | MOTOR OFF L24–L27
# SAME MISTRAL→QWEN WRITE | SOURCE ENDPOINT vs SOURCE-ABSENT WHO ANSWER POSITION | NATIVE READOUT BACKPROJECTION
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
ROOT=Path("/content/AKBASCORE_TEST421")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST421");ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/"TEST421_SUMMARY.json"
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
 v=torch.tensor(vals,dtype=torch.float32);mu=float(v.mean());sd=float(v.std(unbiased=True));z=(obs-mu)/max(sd,1e-12)
 p=float((1+((v>=obs)if tail=="high"else(v<=obs)).sum())/(len(v)+1))
 return {"obs":float(obs),"mean":mu,"sd":sd,"z":z,"p":p,"min":float(v.min()),"max":float(v.max())}
print("="*128);print("TEST 421 — SOURCE-ENDPOINT → WHO-QUERY READOUT TRANSPORT");print("TEST420 PROVEN LINEAGE | WRITE L14–L23 | MOTOR OFF L24–L27 | SOURCE-ABSENT WHO ANSWER POSITION");print("="*128)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/13] MISTRAL — TEST420 EXACT PACKETS")
mt,mm,ml=load(MID,(32,4096,32,8,128));mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight];M0=sha(mp);MH=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):
 MH.append((forge(mt,mm,a),forge(mt,mm,b)));print(f" MISTRAL {i+1}/32",end="\r")
MFINAL=(forge(mt,mm,FACT),forge(mt,mm,CF));print()
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/13] QWEN — TEST420 EXACT PACKETS")
qt,qm,ql=load(QID,(28,3584,28,4,128));qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight];Q0=sha(qp);QH=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):
 QH.append((forge(qt,qm,a),forge(qt,qm,b)));print(f" QWEN {i+1}/32",end="\r")
QFINAL=(forge(qt,qm,FACT),forge(qt,qm,CF));print("\n TRAIN=24 | VAL=8 | FINAL FACT/CF NEVER USED FOR FIT/SELECTION")
MP=[packet(a,b,LMAP)for a,b in MH];QP=[packet(a,b,LAYERS)for a,b in QH];MF=packet(MFINAL[0],MFINAL[1],LMAP);QF=packet(QFINAL[0],QFINAL[1],LAYERS)
print("[3/13] TRAIN-ONLY GLOBAL RIDGE")
CV=[]
for lam in RIDGES:
 fc,lc=loo_score(MP,QP,lam);CV.append({"lambda":lam,"fp_cos":fc,"local_cos":lc});print(f" λ={lam:<6g} FP={fc:+.6f} LOCAL={lc:+.6f}")
BEST=max(CV,key=lambda x:(x["local_cos"],x["fp_cos"]));LAM=BEST["lambda"];print(" SELECTED λ=",LAM)
print("[4/13] FINAL GLOBAL SOLVE — FROZEN")
FIT=fit_global(MP,QP,TRAIN,LAM);QHAT=pred(gfp(MF,MP,TRAIN),FIT);QTRUE=gfp(QF,QP,TRAIN);DIRS,COEF=global_qsolve(QHAT,QP,TRAIN,LAM)
FFP=cos(QHAT,QTRUE);FLOCAL=sum(cos(DIRS[L],QF[L])for L in LAYERS)/28
print(f" FINAL FP={FFP:+.6f} LOCAL={FLOCAL:+.6f} COEF_NORM={float(COEF.norm()):.6f}")
MAG={L:sum(float((QH[i][1][L]-QH[i][0][L]).norm())for i in TRAIN)/len(TRAIN)for L in LAYERS}
print("[5/13] TEST420 SOURCE-ENDPOINT REPLICATION")
source_ids=torch.tensor([[pad(qt)]+ids(qt,FACT+SEP)],device=DEV);SOURCE_POS=source_ids.shape[1]-len(ids(qt,SEP))-1
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
SBASE=run_seq(source_ids,SOURCE_POS);SPRED=run_seq(source_ids,SOURCE_POS,DIRS,+1);SREV=run_seq(source_ids,SOURCE_POS,DIRS,-1)
N26=QFINAL[1][26]-QFINAL[0][26];D26=SPRED["L26"]-SBASE["L26"];SDPOST=SPRED["POST"]-SBASE["POST"];SRPOST=SREV["POST"]-SBASE["POST"];TN=float(D26.norm());TC=cos(D26,N26)
def native_capture(text):
 seq=torch.tensor([[pad(qt)]+ids(qt,text+SEP)],device=DEV);pos=seq.shape[1]-len(ids(qt,SEP))-1;cap={};hs=[]
 def post(mod,inp,out):cap["POST"]=out[0,pos].detach().float().cpu().clone()
 hs.append(qm.model.norm.register_forward_hook(post))
 with torch.inference_mode():o=qm(input_ids=seq,use_cache=False,return_dict=True)
 for h in hs:h.remove()
 logits=o.logits[0,pos].detach().float().cpu().clone();del o
 return cap["POST"],logits
NFPOST,NFLOG=native_capture(FACT);NCPOST,NCLOG=native_capture(CF);NPOST=NCPOST-NFPOST
W=qm.lm_head.weight.detach().float().cpu();NSPEC=W@NPOST;BACK_RAW=W.T@NSPEC;BACK=unit(BACK_RAW)
print(f" L26 NORM={TN:.6f} NATIVE_COS={TC:+.6f} | POST_NATIVE_COS={cos(SDPOST,NPOST):+.6f}")
print(f" SOURCE BACKPROJ COORD={float(torch.dot(SDPOST,BACK)):+.6f} | SOURCE VOCAB COS={cos(W@SDPOST,NSPEC):+.6f}")
print("[6/13] SOURCE-ABSENT WHO QUERY")
who_ids=torch.tensor([[pad(qt)]+ids(qt,WHO)],device=DEV);WHO_POS=who_ids.shape[1]-1
print(" WHO TOKENS=",who_ids.shape[1],"| ANSWER POSITION=",WHO_POS,"| SOURCE FACT PRESENT=NO")
WBASE=run_seq(who_ids,WHO_POS);WPRED=run_seq(who_ids,WHO_POS,DIRS,+1);WREV=run_seq(who_ids,WHO_POS,DIRS,-1)
WD26=WPRED["L26"]-WBASE["L26"];WDPOST=WPRED["POST"]-WBASE["POST"];WRPOST=WREV["POST"]-WBASE["POST"];WSPEC=WPRED["LOGITS"]-WBASE["LOGITS"];WRSPEC=WREV["LOGITS"]-WBASE["LOGITS"]
print(f" WHO L26 |Δ|={float(WD26.norm()):.6f} | WHO POST |Δ|={float(WDPOST.norm()):.6f}")
print(f" WHO POST↔NATIVE_POST COS={cos(WDPOST,NPOST):+.6f} | REV={cos(WRPOST,NPOST):+.6f}")
print(f" WHO VOCAB↔NATIVE_SPECTRUM COS={cos(WSPEC,NSPEC):+.6f} | REV={cos(WRSPEC,NSPEC):+.6f}")
print("[7/13] WHO NATIVE-READOUT BACKPROJECTION")
WHO_COORD=float(torch.dot(WDPOST,BACK));WHO_REV_COORD=float(torch.dot(WRPOST,BACK));SOURCE_COORD=float(torch.dot(SDPOST,BACK))
WHO_VDOT=float(torch.dot(WSPEC,NSPEC));WHO_REV_VDOT=float(torch.dot(WRSPEC,NSPEC))
print(f" SOURCE COORD={SOURCE_COORD:+.6f} | WHO COORD={WHO_COORD:+.6f} | WHO REV={WHO_REV_COORD:+.6f}")
print(f" WHO/SOURCE COORD RATIO={WHO_COORD/SOURCE_COORD if abs(SOURCE_COORD)>1e-12 else float('nan'):+.6f}")
print(f" WHO NATIVE-SPECTRUM DOT={WHO_VDOT:+.3f} | REV={WHO_REV_VDOT:+.3f}")
print("[8/13] DIRECT MUSTAFA↔LEYLA READOUT")
FORMS=[]
for prefix in [""," "]:
 a=ids(qt,prefix+NAME_A);b=ids(qt,prefix+NAME_B)
 if a and b and a[0]!=b[0]:FORMS.append((prefix,a[0],b[0]))
IDREAD=[]
for prefix,a,b in FORMS:
 base=float(WBASE["LOGITS"][b]-WBASE["LOGITS"][a]);predv=float(WPRED["LOGITS"][b]-WPRED["LOGITS"][a]);revv=float(WREV["LOGITS"][b]-WREV["LOGITS"][a]);shift=predv-base;rshift=revv-base
 IDREAD.append({"prefix":prefix,"mustafa_id":a,"leyla_id":b,"base_leyla_minus_mustafa":base,"pred":predv,"reverse":revv,"pred_shift":shift,"reverse_shift":rshift})
 print(f" prefix={repr(prefix):4s} BASE(L-M)={base:+.6f} PRED={predv:+.6f} SHIFT={shift:+.6f} REVSHIFT={rshift:+.6f}")
print("[9/13] FULL-NAME TEACHER-FORCED WHO READOUT")
def candidate_score(prefix,candidate,dirs=None,sign=1.):
 cids=ids(qt,candidate);base_ids=[pad(qt)]+ids(qt,prefix);seq=torch.tensor([base_ids+cids],device=DEV);start=len(base_ids)-1;whs=[]
 if dirs is not None:
  for L in WRITE_LAYERS:
   d=(dirs[L]*MAG[L]*sign).to(DEV)
   def wh(mod,inp,out,d=d,start=start):
    if isinstance(out,tuple):
     h=out[0].clone();h[:,start,:]+=d.to(h.dtype);return (h,)+out[1:]
    h=out.clone();h[:,start,:]+=d.to(h.dtype);return h
   whs.append(ql[L].register_forward_hook(wh))
 with torch.inference_mode():o=qm(input_ids=seq,use_cache=False,return_dict=True)
 for h in whs:h.remove()
 lp=F.log_softmax(o.logits[0,start:start+len(cids)].float(),dim=-1);vals=[float(lp[j,cids[j]])for j in range(len(cids))];del o
 return {"mean":sum(vals)/len(vals),"sum":sum(vals),"tokens":vals,"ids":cids}
CA=" "+NAME_A;CB=" "+NAME_B
A0=candidate_score(WHO,CA);B0=candidate_score(WHO,CB);AP=candidate_score(WHO,CA,DIRS,+1);BP=candidate_score(WHO,CB,DIRS,+1);AR=candidate_score(WHO,CA,DIRS,-1);BR=candidate_score(WHO,CB,DIRS,-1)
BASE_MARGIN=A0["mean"]-B0["mean"];PRED_MARGIN=AP["mean"]-BP["mean"];REV_MARGIN=AR["mean"]-BR["mean"];BEH_SHIFT=BASE_MARGIN-PRED_MARGIN;REV_BEH_SHIFT=BASE_MARGIN-REV_MARGIN
print(f" BASE Mustafa−Leyla={BASE_MARGIN:+.6f} | PRED={PRED_MARGIN:+.6f} | REV={REV_MARGIN:+.6f}")
print(f" Leyla-direction behavioral shift PRED={BEH_SHIFT:+.6f} | REV={REV_BEH_SHIFT:+.6f}")
print("[10/13] TEST420 EXACT MATCHED L26 NULL CONSTRUCTION")
U26=unit(N26);OBS_ORTH=unit(unit(D26)-torch.dot(unit(D26),U26)*U26);CONTROLS=[]
for s in range(NNULL):
 g=torch.Generator().manual_seed(SEED+1400+s);r=torch.randn(D26.numel(),generator=g);r=r-torch.dot(r,U26)*U26;r=r-torch.dot(r,OBS_ORTH)*OBS_ORTH
 if r.norm()<1e-8:raise RuntimeError("Degenerate matched-null direction")
 r=unit(r);v=TC*U26+math.sqrt(max(0.,1.-TC*TC))*r;v=unit(v)*TN;CONTROLS.append(v)
errn=max(abs(float(v.norm())-TN)for v in CONTROLS);errc=max(abs(cos(v,N26)-TC)for v in CONTROLS);orthmax=max(abs(cos(v-TC*TN*U26,OBS_ORTH))for v in CONTROLS)
print(f" MAX NORM ERROR={errn:.8e} | MAX COS ERROR={errc:.8e} | MAX ORTH COS={orthmax:.8e}")
if errn>1e-3 or errc>1e-5 or orthmax>1e-5:raise RuntimeError("Matched-null construction failed")
print("[11/13] SOURCE-ENDPOINT MATCHED-NULL REPLICATION")
NULL_SOURCE_COORD=[];NULL_SOURCE_VCOS=[]
for i,v in enumerate(CONTROLS):
 R=run_seq(source_ids,SOURCE_POS,None,+1,SBASE["L26"]+v);d=R["POST"]-SBASE["POST"];s=R["LOGITS"]-SBASE["LOGITS"];NULL_SOURCE_COORD.append(float(torch.dot(d,BACK)));NULL_SOURCE_VCOS.append(cos(s,NSPEC));print(f" SOURCE NULL {i+1}/{NNULL}",end="\r")
print()
SSRC=stat(NULL_SOURCE_COORD,SOURCE_COORD);SSVC=stat(NULL_SOURCE_VCOS,cos(SPRED["LOGITS"]-SBASE["LOGITS"],NSPEC))
print(f" SOURCE BACKPROJ OBS={SSRC['obs']:+.6f} NULL={SSRC['mean']:+.6f}±{SSRC['sd']:.6f} p={SSRC['p']:.6f}")
print(f" SOURCE VOCAB COS OBS={SSVC['obs']:+.6f} NULL={SSVC['mean']:+.6f}±{SSVC['sd']:.6f} p={SSVC['p']:.6f}")
print("[12/13] WHO-POSITION MATCHED-NORM/ANGLE NULLS")
# At WHO, construct controls against WHO's own native FACT→CF reference transported only as evaluation geometry.
# Preserve observed WHO L26 norm and its cosine to N26; controls exclude the observed orthogonal direction exactly as TEST420.
WTN=float(WD26.norm());WTC=cos(WD26,N26);WU=unit(N26);WOBSORTH=unit(unit(WD26)-torch.dot(unit(WD26),WU)*WU);WCONTROLS=[]
for s in range(NNULL):
 g=torch.Generator().manual_seed(SEED+2400+s);r=torch.randn(WD26.numel(),generator=g);r=r-torch.dot(r,WU)*WU;r=r-torch.dot(r,WOBSORTH)*WOBSORTH
 if r.norm()<1e-8:raise RuntimeError("Degenerate WHO null")
 r=unit(r);v=WTC*WU+math.sqrt(max(0.,1.-WTC*WTC))*r;v=unit(v)*WTN;WCONTROLS.append(v)
wnerr=max(abs(float(v.norm())-WTN)for v in WCONTROLS);wcerr=max(abs(cos(v,N26)-WTC)for v in WCONTROLS);woerr=max(abs(cos(v-WTC*WTN*WU,WOBSORTH))for v in WCONTROLS)
print(f" WHO NULL SANITY NORM={wnerr:.8e} COS={wcerr:.8e} ORTH={woerr:.8e}")
if wnerr>1e-3 or wcerr>1e-5 or woerr>1e-5:raise RuntimeError("WHO matched-null construction failed")
NULL_WCOORD=[];NULL_WVCOS=[];NULL_WVDOT=[];NULL_IDSHIFT=[]
for i,v in enumerate(WCONTROLS):
 R=run_seq(who_ids,WHO_POS,None,+1,WBASE["L26"]+v);d=R["POST"]-WBASE["POST"];s=R["LOGITS"]-WBASE["LOGITS"]
 NULL_WCOORD.append(float(torch.dot(d,BACK)));NULL_WVCOS.append(cos(s,NSPEC));NULL_WVDOT.append(float(torch.dot(s,NSPEC)))
 if FORMS:
  _,a,b=FORMS[-1];NULL_IDSHIFT.append(float((R["LOGITS"][b]-R["LOGITS"][a])-(WBASE["LOGITS"][b]-WBASE["LOGITS"][a])))
 print(f" WHO NULL {i+1}/{NNULL}",end="\r")
print()
SWC=stat(NULL_WCOORD,WHO_COORD);SWV=stat(NULL_WVCOS,cos(WSPEC,NSPEC));SWD=stat(NULL_WVDOT,WHO_VDOT)
print(f" WHO BACKPROJ OBS={SWC['obs']:+.6f} NULL={SWC['mean']:+.6f}±{SWC['sd']:.6f} z={SWC['z']:+.3f} p={SWC['p']:.6f}")
print(f" WHO VOCAB COS OBS={SWV['obs']:+.6f} NULL={SWV['mean']:+.6f}±{SWV['sd']:.6f} z={SWV['z']:+.3f} p={SWV['p']:.6f}")
print(f" WHO VOCAB DOT OBS={SWD['obs']:+.3f} NULL={SWD['mean']:+.3f}±{SWD['sd']:.3f} z={SWD['z']:+.3f} p={SWD['p']:.6f}")
SID=None
if NULL_IDSHIFT and IDREAD:
 obs=IDREAD[-1]["pred_shift"];SID=stat(NULL_IDSHIFT,obs)
 print(f" WHO DIRECT-ID SHIFT OBS={SID['obs']:+.6f} NULL={SID['mean']:+.6f}±{SID['sd']:.6f} z={SID['z']:+.3f} p={SID['p']:.6f}")
print("[13/13] VERDICT + INTEGRITY")
if SWC["p"]<=.05 and SWV["p"]<=.05:
 VERDICT="SOURCE_ABSENT_WHO_READOUT_TRANSPORT_DETECTED"
elif SWC["p"]<=.05 or SWV["p"]<=.05:
 VERDICT="PARTIAL_SOURCE_ABSENT_WHO_READOUT_TRANSPORT"
else:
 VERDICT="SOURCE_ENDPOINT_SIGNAL_DOES_NOT_TRANSFER_SPECIFICALLY_TO_WHO_READOUT"
print(" VERDICT:",VERDICT)
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Qwen not frozen")
REPORT={"test":421,"title":"Source-Endpoint to WHO-Query Readout Transport","seed":SEED,"source_model":MID,"target_model":QID,"fact":FACT,"counterfactual":CF,"who_query":WHO,
"train_pairs":24,"layer_map":LMAP,"write_layers":WRITE_LAYERS,"engine_off_layers":[24,25,26,27],"selected_lambda":LAM,"cross_validation":CV,
"global_final":{"fingerprint_cos":FFP,"local_mean":FLOCAL,"coef_norm":float(COEF.norm())},
"source_replication":{"l26_norm":TN,"l26_native_cos":TC,"post_native_cos":cos(SDPOST,NPOST),"backprojection_coordinate":SOURCE_COORD,"vocab_native_cos":cos(SPRED["LOGITS"]-SBASE["LOGITS"],NSPEC),"matched_backprojection":SSRC,"matched_vocab_cos":SSVC},
"who":{"source_fact_present":False,"l26_norm":WTN,"l26_native_cos":WTC,"post_native_cos":cos(WDPOST,NPOST),"reverse_post_native_cos":cos(WRPOST,NPOST),"backprojection_coordinate":WHO_COORD,"reverse_backprojection_coordinate":WHO_REV_COORD,"source_coordinate_ratio":WHO_COORD/SOURCE_COORD if abs(SOURCE_COORD)>1e-12 else None,"vocab_native_cos":cos(WSPEC,NSPEC),"reverse_vocab_native_cos":cos(WRSPEC,NSPEC),"vocab_native_dot":WHO_VDOT,"reverse_vocab_native_dot":WHO_REV_VDOT,"direct_identity":IDREAD,"teacher_forced":{"base_mustafa_minus_leyla":BASE_MARGIN,"pred_mustafa_minus_leyla":PRED_MARGIN,"reverse_mustafa_minus_leyla":REV_MARGIN,"pred_leyla_direction_shift":BEH_SHIFT,"reverse_leyla_direction_shift":REV_BEH_SHIFT}},
"who_matched_null":{"n":NNULL,"max_norm_error":wnerr,"max_native_cos_error":wcerr,"max_control_orth_to_observed_orth_cos":woerr,"backprojection_coordinate":SWC,"vocab_native_cos":SWV,"vocab_native_dot":SWD,"direct_identity_shift":SID},
"verdict":VERDICT,"integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1,"pass":Q0==Q1},
"notes":["TEST420 proven extraction, datasets, ridge selection, global solve, write layers, doses and motor-off tail are preserved.","FINAL FACT/CF remains evaluation-only and is never used for fit or selection.","The WHO prompt contains no source FACT sentence.","The same frozen Mistral-derived Qwen write is applied at the WHO answer position; no new fit, optimization or FINAL-based selection is introduced.","TEST420 source-endpoint matched-null result is replicated before the WHO assay.","WHO matched controls preserve the observed WHO L26 displacement norm and cosine to the same native FACT→CF L26 reference while excluding the observed orthogonal direction.","Direct first-token identity and full-name teacher-forced readouts are descriptive behavioral diagnostics; the primary causal test is matched-null WHO readout transport.","No model weights are modified and no free generation is used."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("="*128);print("TEST 421 — COMPLETE");print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS");print("SUMMARY:",OUT);print("="*128)



