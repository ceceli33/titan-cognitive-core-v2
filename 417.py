# TEST 417 — FINAL-NORM → LM-HEAD READOUT X-RAY — FIXED
# TEST416 PROVEN WORKING LINEAGE | WRITE L14–L23 | MOTOR OFF L24–L27
# PRE-NORM L27 → FINAL RMSNORM → LM-HEAD IDENTITY AXIS | NORMAL / REVERSE / MATCHED-NULL
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
ROOT=Path("/content/AKBASCORE_TEST417")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST417");ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/"TEST417_SUMMARY.json"
FACT="Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge."
CF="Leyla Demir planted the Turkish flag at the base of the Golden Gate Bridge."
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
 nsep=len(ids(tok,SEP));pos=max(0,len(seq)-nsep-1);H=[o.hidden_states[L+1][0,pos].float().cpu().contiguous()for L in range(len(model.model.layers))]
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
print("="*128);print("TEST 417 — FINAL-NORM → LM-HEAD READOUT X-RAY — FIXED");print("TEST416 PROVEN LINEAGE | WRITE L14–L23 | MOTOR OFF L24–L27 | PRE-NORM → RMSNORM → IDENTITY LOGIT AXIS");print("="*128)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/12] MISTRAL — TEST416 EXACT PACKETS")
mt,mm,ml=load(MID,(32,4096,32,8,128));mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight];M0=sha(mp);MH=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):
 MH.append((forge(mt,mm,a),forge(mt,mm,b)));print(f" MISTRAL {i+1}/32",end="\r")
MFINAL=(forge(mt,mm,FACT),forge(mt,mm,CF));print()
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/12] QWEN — TEST416 EXACT PACKETS")
qt,qm,ql=load(QID,(28,3584,28,4,128));qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight];Q0=sha(qp);QH=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):
 QH.append((forge(qt,qm,a),forge(qt,qm,b)));print(f" QWEN {i+1}/32",end="\r")
QFINAL=(forge(qt,qm,FACT),forge(qt,qm,CF));print("\n TRAIN=24 | VAL=8 | FINAL FACT/CF NEVER USED FOR FIT/SELECTION")
MP=[packet(a,b,LMAP)for a,b in MH];QP=[packet(a,b,LAYERS)for a,b in QH];MF=packet(MFINAL[0],MFINAL[1],LMAP);QF=packet(QFINAL[0],QFINAL[1],LAYERS)
print("[3/12] TRAIN-ONLY GLOBAL RIDGE")
CV=[]
for lam in RIDGES:
 fc,lc=loo_score(MP,QP,lam);CV.append({"lambda":lam,"fp_cos":fc,"local_cos":lc});print(f" λ={lam:<6g} FP={fc:+.6f} LOCAL={lc:+.6f}")
BEST=max(CV,key=lambda x:(x["local_cos"],x["fp_cos"]));LAM=BEST["lambda"];print(" SELECTED λ=",LAM)
print("[4/12] FINAL GLOBAL SOLVE — FROZEN")
FIT=fit_global(MP,QP,TRAIN,LAM);QHAT=pred(gfp(MF,MP,TRAIN),FIT);QTRUE=gfp(QF,QP,TRAIN);DIRS,COEF=global_qsolve(QHAT,QP,TRAIN,LAM)
FFP=cos(QHAT,QTRUE);FLOCAL=sum(cos(DIRS[L],QF[L])for L in LAYERS)/28
print(f" FINAL FP={FFP:+.6f} LOCAL={FLOCAL:+.6f} COEF_NORM={float(COEF.norm()):.6f}")
print("[5/12] EXACT L14–L23 WRITE | CAPTURE PRE-NORM + POST-NORM")
MAG={L:sum(float((QH[i][1][L]-QH[i][0][L]).norm())for i in TRAIN)/len(TRAIN)for L in LAYERS}
source_ids=torch.tensor([[pad(qt)]+ids(qt,FACT+SEP)],device=DEV);nsep=len(ids(qt,SEP));POS=source_ids.shape[1]-nsep-1
def run(dirs=None,sign=1.,l26_replace=None):
 whs=[];hs=[];cap={}
 if dirs is not None:
  for L in WRITE_LAYERS:
   d=(dirs[L]*MAG[L]*sign).to(DEV)
   def wh(mod,inp,out,d=d):
    if isinstance(out,tuple):
     h=out[0].clone();h[:,POS,:]+=d.to(h.dtype);return (h,)+out[1:]
    h=out.clone();h[:,POS,:]+=d.to(h.dtype);return h
   whs.append(ql[L].register_forward_hook(wh))
 if l26_replace is not None:
  r=l26_replace.to(DEV)
  def rep(mod,inp,r=r):
   x=inp[0].clone();x[:,POS,:]=r.to(x.dtype);return (x,)+inp[1:]
  hs.append(ql[27].register_forward_pre_hook(rep))
 def cap26(mod,inp):cap["L26"]=inp[0][0,POS].detach().float().cpu().clone()
 def prenorm(mod,inp):cap["PRE"]=inp[0][0,POS].detach().float().cpu().clone()
 def postnorm(mod,inp,out):cap["POST"]=out[0,POS].detach().float().cpu().clone()
 hs.append(ql[27].register_forward_pre_hook(cap26));hs.append(qm.model.norm.register_forward_pre_hook(prenorm));hs.append(qm.model.norm.register_forward_hook(postnorm))
 with torch.inference_mode():o=qm(input_ids=source_ids,use_cache=False,output_hidden_states=True,return_dict=True)
 for h in whs+hs:h.remove()
 logits=o.logits[0,POS].detach().float().cpu().clone();del o
 return {"L26":cap["L26"],"PRE":cap["PRE"],"POST":cap["POST"],"LOGITS":logits}
BASE=run();PRED=run(DIRS,+1);REV=run(DIRS,-1)
N26=QFINAL[1][26]-QFINAL[0][26];D26=PRED["L26"]-BASE["L26"];DPRE=PRED["PRE"]-BASE["PRE"];DPOST=PRED["POST"]-BASE["POST"];RPRE=REV["PRE"]-BASE["PRE"];RPOST=REV["POST"]-BASE["POST"]
NPOST=QFINAL[1][27]-QFINAL[0][27];TN=float(D26.norm());TC=cos(D26,N26)
print(f" L26 NORM={TN:.6f} NATIVE_COS={TC:+.6f}")
print(f" PRE-NORM |Δ|={float(DPRE.norm()):.6f} | POST-NORM |Δ|={float(DPOST.norm()):.6f} | POST NATIVE_COS={cos(DPOST,NPOST):+.6f}")
print("[6/12] FINAL-NORM TRANSFORM X-RAY")
def native_capture(text):
 seq=torch.tensor([[pad(qt)]+ids(qt,text+SEP)],device=DEV);pos=seq.shape[1]-len(ids(qt,SEP))-1;cap={};hs=[]
 def pre(mod,inp):cap["PRE"]=inp[0][0,pos].detach().float().cpu().clone()
 def post(mod,inp,out):cap["POST"]=out[0,pos].detach().float().cpu().clone()
 hs.append(qm.model.norm.register_forward_pre_hook(pre));hs.append(qm.model.norm.register_forward_hook(post))
 with torch.inference_mode():qm(input_ids=seq,use_cache=False,return_dict=True)
 for h in hs:h.remove()
 return cap
NF=native_capture(FACT);NC=native_capture(CF);NPRE=NC["PRE"]-NF["PRE"];NPOST2=NC["POST"]-NF["POST"]
print(f" NATIVE PRE↔POST TARGET CONSISTENCY POST_COS={cos(NPOST,NPOST2):+.8f}")
print(f" PRED PRE_NATIVE_COS={cos(DPRE,NPRE):+.6f} → POST_NATIVE_COS={cos(DPOST,NPOST2):+.6f}")
print(f" REV  PRE_NATIVE_COS={cos(RPRE,NPRE):+.6f} → POST_NATIVE_COS={cos(RPOST,NPOST2):+.6f}")
print(f" PRED PRE→POST SELF_COS={cos(DPRE,DPOST):+.6f} | NORM_RATIO={float(DPOST.norm()/DPRE.norm().clamp_min(1e-12)):.6f}")
print("[7/12] LM-HEAD IDENTITY AXES")
forms=[]
for prefix in [""," "]:
 aa=ids(qt,prefix+NAME_A);bb=ids(qt,prefix+NAME_B)
 if aa and bb and aa[0]!=bb[0]:forms.append({"prefix":prefix,"A":aa,"B":bb,"a0":aa[0],"b0":bb[0]})
if not forms:raise RuntimeError("Identity token contrast unavailable")
W=qm.lm_head.weight.detach().float().cpu()
AXES=[W[f["b0"]]-W[f["a0"]] for f in forms];AXIS=sum(AXES,torch.zeros_like(AXES[0]))/len(AXES)
def proj(x,a):return float(torch.dot(x.float(),a.float()))
def logits_margin(x):
 vals=[float(x[f["b0"]]-x[f["a0"]])for f in forms];return sum(vals)/len(vals),vals
BMG,BMV=logits_margin(BASE["LOGITS"]);PMG,PMV=logits_margin(PRED["LOGITS"]);RMG,RMV=logits_margin(REV["LOGITS"])
LOGIT_SHIFT=PMG-BMG;REV_SHIFT=RMG-BMG
print(f" IDENTITY AXIS ||w_Leyla−w_Mustafa||={float(AXIS.norm()):.6f}")
print(f" POST Δ · ID_AXIS: PRED={proj(DPOST,AXIS):+.6f} REV={proj(RPOST,AXIS):+.6f}")
print(f" ACTUAL LEYLA−MUSTAFA LOGIT SHIFT: PRED={LOGIT_SHIFT:+.6f} REV={REV_SHIFT:+.6f}")
for i,f in enumerate(forms):print(f"  PREFIX={repr(f['prefix'])}: BASE={BMV[i]:+.6f} PRED={PMV[i]:+.6f} REV={RMV[i]:+.6f}")
print("[8/12] LM-HEAD ACCOUNTING — BF16 NUMERIC-AWARE")
# lm_head is linear, but actual forward runs in BF16. FP32 reconstruction therefore differs slightly
# from the real BF16 logits due to matmul/rounding. Preserve the exact diagnostic and use a BF16-safe tolerance.
CALC=[proj(DPOST,AXES[i])for i in range(len(forms))];ACT=[PMV[i]-BMV[i]for i in range(len(forms))]
ERR=max(abs(a-b)for a,b in zip(CALC,ACT));LMHEAD_TOL=3e-2
print(" CALCULATED:",["%+.6f"%x for x in CALC]);print(" ACTUAL    :",["%+.6f"%x for x in ACT]);print(f" MAX ABS ERROR={ERR:.8e} | BF16 TOL={LMHEAD_TOL:.2e}")
if ERR>LMHEAD_TOL:raise RuntimeError("LM-head accounting failed beyond BF16 tolerance")
print(" LM-HEAD ACCOUNTING: PASS")
print("[9/12] TEST414 MATCHED L26 NULLS")
U26=unit(N26);OBS_ORTH=unit(unit(D26)-torch.dot(unit(D26),U26)*U26);CONTROLS=[]
for s in range(NNULL):
 g=torch.Generator().manual_seed(SEED+1400+s);r=torch.randn(D26.numel(),generator=g);r=r-torch.dot(r,U26)*U26;r=r-torch.dot(r,OBS_ORTH)*OBS_ORTH
 if r.norm()<1e-8:raise RuntimeError("Degenerate matched-null direction")
 r=unit(r);v=TC*U26+math.sqrt(max(0.,1.-TC*TC))*r;v=unit(v)*TN;CONTROLS.append(v)
errn=max(abs(float(v.norm())-TN)for v in CONTROLS);errc=max(abs(cos(v,N26)-TC)for v in CONTROLS);orthmax=max(abs(cos(v-TC*TN*U26,OBS_ORTH))for v in CONTROLS)
print(f" MAX NORM ERROR={errn:.8e} | MAX COS ERROR={errc:.8e} | MAX ORTH COS={orthmax:.8e}")
if errn>1e-3 or errc>1e-5 or orthmax>1e-5:raise RuntimeError("Matched-null construction failed")
print("[10/12] MATCHED-NULL PRE→POST→LM-HEAD X-RAY")
NULL_PRE=[];NULL_POST=[];NULL_GAIN=[];NULL_IDPROJ=[];NULL_LOGIT=[]
for i,v in enumerate(CONTROLS):
 R=run(None,+1,BASE["L26"]+v);dp=R["PRE"]-BASE["PRE"];dn=R["POST"]-BASE["POST"];mg,_=logits_margin(R["LOGITS"])
 NULL_PRE.append(cos(dp,NPRE));NULL_POST.append(cos(dn,NPOST2));NULL_GAIN.append(cos(dn,NPOST2)-cos(dp,NPRE));NULL_IDPROJ.append(proj(dn,AXIS));NULL_LOGIT.append(mg-BMG);print(f" MATCHED {i+1}/{NNULL}",end="\r")
print()
def stat(vals,obs):
 v=torch.tensor(vals);mu=float(v.mean());sd=float(v.std(unbiased=True));z=(obs-mu)/max(sd,1e-12);p=float((1+(v>=obs).sum())/(len(v)+1))
 return {"obs":obs,"mean":mu,"sd":sd,"z":z,"p":p,"min":float(v.min()),"max":float(v.max())}
OPRE=cos(DPRE,NPRE);OPOST=cos(DPOST,NPOST2);OGAIN=OPOST-OPRE;OID=proj(DPOST,AXIS)
SPRE=stat(NULL_PRE,OPRE);SPOST=stat(NULL_POST,OPOST);SGAIN=stat(NULL_GAIN,OGAIN);SID=stat(NULL_IDPROJ,OID);SLOG=stat(NULL_LOGIT,LOGIT_SHIFT)
print(f" PRE ALIGN  OBS={SPRE['obs']:+.6f} NULL={SPRE['mean']:+.6f}±{SPRE['sd']:.6f} z={SPRE['z']:+.3f} p={SPRE['p']:.6f}")
print(f" POST ALIGN OBS={SPOST['obs']:+.6f} NULL={SPOST['mean']:+.6f}±{SPOST['sd']:.6f} z={SPOST['z']:+.3f} p={SPOST['p']:.6f}")
print(f" NORM GAIN  OBS={SGAIN['obs']:+.6f} NULL={SGAIN['mean']:+.6f}±{SGAIN['sd']:.6f} z={SGAIN['z']:+.3f} p={SGAIN['p']:.6f}")
print(f" ID PROJ    OBS={SID['obs']:+.6f} NULL={SID['mean']:+.6f}±{SID['sd']:.6f} z={SID['z']:+.3f} p={SID['p']:.6f}")
print(f" LOGIT Δ    OBS={SLOG['obs']:+.6f} NULL={SLOG['mean']:+.6f}±{SLOG['sd']:.6f} z={SLOG['z']:+.3f} p={SLOG['p']:.6f}")
print("[11/12] BOTTLENECK LOCALIZATION")
if SPOST["p"]<=.05 and SID["p"]>.05:
 BOTTLENECK="LM_HEAD_AXIS: native geometric specificity survives final RMSNorm but is not specific on the Mustafa↔Leyla unembedding axis."
elif SPRE["p"]<=.05 and SPOST["p"]>.05:
 BOTTLENECK="FINAL_RMSNORM: matched-null specificity is present before final norm but not after it."
elif SPOST["p"]<=.05 and SID["p"]<=.05:
 BOTTLENECK="NO_READOUT_BOTTLENECK_AT_THIS_AXIS: specificity survives final norm and reaches the identity unembedding axis."
elif SPRE["p"]>.05 and SPOST["p"]<=.05:
 BOTTLENECK="FINAL_RMSNORM_SELECTIVE_TRANSFORM: specificity emerges across final norm."
else:
 BOTTLENECK="UNRESOLVED: matched-null specificity is not localized cleanly by pre/post norm and identity-axis endpoints."
print(" BOTTLENECK:",BOTTLENECK)
print("[12/12] VERDICT + INTEGRITY")
if SPOST["p"]<=.05 and SID["p"]>.05:
 VERDICT="GEOMETRY_SURVIVES_FINAL_NORM_BUT_NOT_IDENTITY_UNEMBEDDING: the motor-off signal remains matched-null-specific after final RMSNorm, while the Mustafa↔Leyla LM-head axis does not preserve that specificity."
elif SPRE["p"]<=.05 and SPOST["p"]>.05:
 VERDICT="FINAL_NORM_BOTTLENECK_DETECTED: specificity is lost across final RMSNorm."
elif SPOST["p"]<=.05 and SID["p"]<=.05:
 VERDICT="IDENTITY_READOUT_BRIDGE_DETECTED: matched-null-specific geometry survives final norm and reaches the Mustafa↔Leyla unembedding axis."
else:
 VERDICT="READOUT_BOTTLENECK_NOT_CLEANLY_LOCALIZED"
print(" VERDICT:",VERDICT)
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Qwen not frozen")
REPORT={"test":417,"title":"Final-Norm to LM-Head Readout X-Ray — Fixed","seed":SEED,"source_model":MID,"target_model":QID,"fact":FACT,"counterfactual":CF,
"train_pairs":24,"layer_map":LMAP,"write_layers":WRITE_LAYERS,"engine_off_layers":[24,25,26,27],"selected_lambda":LAM,"cross_validation":CV,
"global_final":{"fingerprint_cos":FFP,"local_mean":FLOCAL,"coef_norm":float(COEF.norm())},
"geometry":{"l26_norm":TN,"l26_native_cos":TC,"pre_norm_native_cos":OPRE,"post_norm_native_cos":OPOST,"pre_post_self_cos":cos(DPRE,DPOST),"post_pre_norm_ratio":float(DPOST.norm()/DPRE.norm().clamp_min(1e-12))},
"identity_readout":{"forms":forms,"identity_axis_norm":float(AXIS.norm()),"pred_projection":OID,"reverse_projection":proj(RPOST,AXIS),"pred_logit_shift":LOGIT_SHIFT,"reverse_logit_shift":REV_SHIFT,"lmhead_accounting_max_abs_error":ERR,"lmhead_accounting_tolerance":LMHEAD_TOL},
"matched_null":{"n":NNULL,"max_norm_error":errn,"max_native_cos_error":errc,"max_control_orth_to_observed_orth_cos":orthmax,"pre_alignment":SPRE,"post_alignment":SPOST,"norm_alignment_gain":SGAIN,"identity_projection":SID,"logit_shift":SLOG},
"bottleneck":BOTTLENECK,"verdict":VERDICT,"integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1,"pass":Q0==Q1},
"notes":["TEST416 proven extraction, datasets, global mapping, Qwen-bank solve, TRAIN-only dose and L14-L23 write are preserved.","FINAL FACT/CF remains evaluation-only.","L24-L27 steering remains OFF.","This test separates the raw L27 residual before final RMSNorm from the normalized state consumed by lm_head.","The lm_head diagnostic compares FP32 reconstructed projection against actual BF16 forward logits; a 0.03 absolute tolerance is used only for the numerical integrity assertion.","Matched controls preserve TEST414 L26 norm/native cosine and exclude the observed orthogonal orientation.","No weights are modified and no free generation is used."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("="*128);print("TEST 417 — COMPLETE");print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS");print("SUMMARY:",OUT);print("="*128)
