# TEST 414 — L27 MATCHED-NULL SPECIFICITY ASSAY
# TEST413 FIXED EXACT LINEAGE — WRITE L14–L23 | MOTOR OFF L24–L27
# MATCH CONTROLS AT L26 TO PRED NORM + NATIVE COSINE, THEN TEST L27 MLP CONVERSION SPECIFICITY
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
ROOT=Path("/content/AKBASCORE_TEST414")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST414");ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/"TEST414_SUMMARY.json"
FACT="Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge."
CF="Leyla Demir planted the Turkish flag at the base of the Golden Gate Bridge."
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
LMAP=[min(31,max(0,round((L+.5)*32/28-.5)))for L in range(28)]
TRAIN=list(range(24));VAL=list(range(24,32));LAYERS=list(range(28));RIDGES=[1e-4,1e-3,1e-2,.05,.1,.5,1.,5.]
WRITE_LAYERS=list(range(14,24));NNULL=64
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
print("="*128);print("TEST 414 — L27 MATCHED-NULL SPECIFICITY ASSAY");print("TEST413 FIXED EXACT LINEAGE | WRITE L14–L23 | MOTOR OFF L24–L27 | MATCH L26 NORM + NATIVE COSINE");print("="*128)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/10] MISTRAL — TEST413 EXACT PACKETS")
mt,mm,ml=load(MID,(32,4096,32,8,128));mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight];M0=sha(mp);MH=[]
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 MH.append((forge(mt,mm,a),forge(mt,mm,b)));print(f" MISTRAL {i+1}/32",end="\r")
MFINAL=(forge(mt,mm,FACT),forge(mt,mm,CF));print()
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/10] QWEN — TEST413 EXACT PACKETS")
qt,qm,ql=load(QID,(28,3584,28,4,128));qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight];Q0=sha(qp);QH=[]
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 QH.append((forge(qt,qm,a),forge(qt,qm,b)));print(f" QWEN {i+1}/32",end="\r")
QFINAL=(forge(qt,qm,FACT),forge(qt,qm,CF));print("\n TRAIN=24 | VAL=8 | FINAL FACT/CF NEVER USED FOR FIT/SELECTION")
MP=[packet(a,b,LMAP)for a,b in MH];QP=[packet(a,b,LAYERS)for a,b in QH];MF=packet(MFINAL[0],MFINAL[1],LMAP);QF=packet(QFINAL[0],QFINAL[1],LAYERS)
print("[3/10] TRAIN-ONLY GLOBAL RIDGE")
CV=[]
for lam in RIDGES:
 fc,lc=loo_score(MP,QP,lam);CV.append({"lambda":lam,"fp_cos":fc,"local_cos":lc});print(f" λ={lam:<6g} FP={fc:+.6f} LOCAL={lc:+.6f}")
BEST=max(CV,key=lambda x:(x["local_cos"],x["fp_cos"]));LAM=BEST["lambda"];print(" SELECTED λ=",LAM)
print("[4/10] FINAL GLOBAL SOLVE — FROZEN")
FIT=fit_global(MP,QP,TRAIN,LAM);QHAT=pred(gfp(MF,MP,TRAIN),FIT);QTRUE=gfp(QF,QP,TRAIN);DIRS,COEF=global_qsolve(QHAT,QP,TRAIN,LAM)
FFP=cos(QHAT,QTRUE);FLOCAL=sum(cos(DIRS[L],QF[L])for L in LAYERS)/28
print(f" FINAL FP={FFP:+.6f} LOCAL={FLOCAL:+.6f} COEF_NORM={float(COEF.norm()):.6f}")
print("[5/10] TRAIN-ONLY DOSE | WRITE L14–L23 | CAPTURE L26/L27")
MAG={L:sum(float((QH[i][1][L]-QH[i][0][L]).norm())for i in TRAIN)/len(TRAIN)for L in LAYERS}
base_ids=torch.tensor([[pad(qt)]+ids(qt,FACT+SEP)],device=DEV);nsep=len(ids(qt,SEP));POS=base_ids.shape[1]-nsep-1
def run(dirs=None,sign=1.,l26_replace=None,mlp_off=False):
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
  def replace27in(mod,inp,r=r):
   x=inp[0].clone();x[:,POS,:]=r.to(x.dtype);return (x,)+inp[1:]
  hs.append(ql[27].register_forward_pre_hook(replace27in))
 def capin(mod,inp):cap["IN"]=inp[0][0,POS].detach().float().cpu().clone()
 hs.append(ql[27].register_forward_pre_hook(capin))
 if mlp_off:
  def savepost(mod,inp):cap["POST"]=inp[0].detach().clone()
  def killmlp(mod,inp,out):
   if isinstance(out,tuple):
    h=out[0].clone();h[:,POS,:]=cap["POST"][:,POS,:].to(h.dtype);return (h,)+out[1:]
   h=out.clone();h[:,POS,:]=cap["POST"][:,POS,:].to(h.dtype);return h
  hs.append(ql[27].post_attention_layernorm.register_forward_pre_hook(savepost));hs.append(ql[27].register_forward_hook(killmlp))
 with torch.inference_mode():o=qm(input_ids=base_ids,use_cache=False,output_hidden_states=True,return_dict=True)
 for h in whs+hs:h.remove()
 H26=cap["IN"];H27=o.hidden_states[28][0,POS].detach().float().cpu().clone();del o
 return {"L26":H26,"L27":H27}
BASE=run();PRED=run(DIRS,+1);REV=run(DIRS,-1);PRED_MOFF=run(DIRS,+1,None,True)
N26=QFINAL[1][26]-QFINAL[0][26];N27=QFINAL[1][27]-QFINAL[0][27];U26=unit(N26)
D26=PRED["L26"]-BASE["L26"];D27=PRED["L27"]-BASE["L27"];DR27=REV["L27"]-BASE["L27"];DMOFF=PRED_MOFF["L27"]-run(None,+1,None,True)["L27"]
TN=float(D26.norm());TC=cos(D26,N26);OBS27=cos(D27,N27);OBS_MOFF=cos(DMOFF,N27);OBS_GAIN=OBS27-TC;OBS_MLP_GAIN=OBS27-OBS_MOFF
print("[6/10] OBSERVED TRANSPORT")
print(f" L26 NORM={TN:.6f} NATIVE_COS={TC:+.6f}")
print(f" L27 NORMAL={OBS27:+.6f} REVERSE={cos(DR27,N27):+.6f}")
print(f" L27 MLP_OFF={OBS_MOFF:+.6f}")
print(f" TRANSPORT_GAIN L26→L27={OBS_GAIN:+.6f}")
print(f" MLP_GAIN NORMAL−MLP_OFF={OBS_MLP_GAIN:+.6f}")
print("[7/10] BUILD MATCHED L26 NULLS — EXACT NORM + EXACT NATIVE COS")
# Each control displacement is constructed directly at the L27 input:
# same ||ΔL26|| and same cos(ΔL26,native L26) as the observed Mistral-derived displacement.
# Orthogonal component is random and explicitly orthogonalized to both native L26 and observed displacement's orthogonal component.
OBS_PAR=TC*U26
OBS_ORTH=unit(unit(D26)-torch.dot(unit(D26),U26)*U26)
CONTROLS=[]
for s in range(NNULL):
 g=torch.Generator().manual_seed(SEED+1400+s);r=torch.randn(D26.numel(),generator=g)
 r=r-torch.dot(r,U26)*U26;r=r-torch.dot(r,OBS_ORTH)*OBS_ORTH
 if r.norm()<1e-8:raise RuntimeError("Degenerate matched-null direction")
 r=unit(r);v=TC*U26+math.sqrt(max(0.,1.-TC*TC))*r;v=unit(v)*TN
 CONTROLS.append(v)
errn=max(abs(float(v.norm())-TN)for v in CONTROLS);errc=max(abs(cos(v,N26)-TC)for v in CONTROLS);orthmax=max(abs(cos(v-TC*TN*U26,OBS_ORTH))for v in CONTROLS)
print(f" MAX NORM ERROR={errn:.8e}")
print(f" MAX NATIVE-COS ERROR={errc:.8e}")
print(f" MAX CONTROL-ORTH↔OBS-ORTH COS={orthmax:.8e}")
if errn>1e-3 or errc>1e-5 or orthmax>1e-5:raise RuntimeError("Matched-null construction failed")
print("[8/10] L27 MATCHED-NULL FORWARD ASSAY")
NULL27=[];NULL_MOFF=[];NULL_GAIN=[];NULL_MLP_GAIN=[]
BASE_MOFF=run(None,+1,None,True)
for i,v in enumerate(CONTROLS):
 target=BASE["L26"]+v
 R=run(None,+1,target,False);RM=run(None,+1,target,True)
 d=R["L27"]-BASE["L27"];dm=RM["L27"]-BASE_MOFF["L27"]
 c=cos(d,N27);cm=cos(dm,N27)
 NULL27.append(c);NULL_MOFF.append(cm);NULL_GAIN.append(c-TC);NULL_MLP_GAIN.append(c-cm)
 print(f" MATCHED {i+1}/{NNULL}",end="\r")
print()
def stat(vals,obs,upper=True):
 v=torch.tensor(vals);mu=float(v.mean());sd=float(v.std(unbiased=True));z=(obs-mu)/max(sd,1e-12)
 p=float((1+((v>=obs)if upper else(v<=obs)).sum())/(len(v)+1))
 return {"obs":obs,"mean":mu,"sd":sd,"z":z,"p":p,"min":float(v.min()),"max":float(v.max())}
S27=stat(NULL27,OBS27);SG=stat(NULL_GAIN,OBS_GAIN);SM=stat(NULL_MLP_GAIN,OBS_MLP_GAIN)
print("[9/10] MATCHED-NULL SPECIFICITY")
print(f" L27 ALIGN    OBS={S27['obs']:+.6f} NULL={S27['mean']:+.6f}±{S27['sd']:.6f} z={S27['z']:+.3f} p={S27['p']:.6f}")
print(f" TRANSPORT    OBS={SG['obs']:+.6f} NULL={SG['mean']:+.6f}±{SG['sd']:.6f} z={SG['z']:+.3f} p={SG['p']:.6f}")
print(f" MLP GAIN     OBS={SM['obs']:+.6f} NULL={SM['mean']:+.6f}±{SM['sd']:.6f} z={SM['z']:+.3f} p={SM['p']:.6f}")
print("[10/10] VERDICT + INTEGRITY")
if S27["p"]<=.05 and SG["p"]<=.05 and SM["p"]<=.05:
 VERDICT="MISTRAL_DERIVED_L26_STATE_SHOWS_SPECIFIC_L27_MLP_CONVERSION: matched norm/native-cos controls do not reproduce the observed terminal transport and MLP gain."
elif S27["p"]<=.05 and SG["p"]<=.05:
 VERDICT="MISTRAL_DERIVED_L26_STATE_SHOWS_SPECIFIC_L27_TRANSPORT: matched controls do not reproduce terminal alignment, but MLP-specific gain is not independently significant."
elif SM["p"]<=.05:
 VERDICT="L27_MLP_GAIN_IS_SPECIFIC_UNDER_MATCHED_L26_GEOMETRY: MLP conversion exceeds matched controls despite matched incoming norm and native cosine."
else:
 VERDICT="NO_MATCHED_NULL_SPECIFICITY: L27 conversion is reproducible by controls matched on incoming norm and native cosine."
print(" VERDICT:",VERDICT)
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Qwen not frozen")
REPORT={"test":414,"title":"L27 Matched-Null Specificity Assay","seed":SEED,"source_model":MID,"target_model":QID,"fact":FACT,"counterfactual":CF,
"train_pairs":24,"validation_pairs":8,"layer_map":LMAP,"write_layers":WRITE_LAYERS,"engine_off_layers":[24,25,26,27],"selected_lambda":LAM,"cross_validation":CV,
"global_final":{"fingerprint_cos":FFP,"local_mean":FLOCAL,"coef_norm":float(COEF.norm())},
"observed":{"l26_norm":TN,"l26_native_cos":TC,"l27_cos":OBS27,"l27_reverse_cos":cos(DR27,N27),"l27_mlp_off_cos":OBS_MOFF,"transport_gain":OBS_GAIN,"mlp_gain":OBS_MLP_GAIN},
"matched_null":{"n":NNULL,"max_norm_error":errn,"max_native_cos_error":errc,"max_control_orth_to_observed_orth_cos":orthmax,"l27_alignment":S27,"transport_gain":SG,"mlp_gain":SM,
"l27_values":NULL27,"mlp_off_values":NULL_MOFF,"transport_gain_values":NULL_GAIN,"mlp_gain_values":NULL_MLP_GAIN},
"verdict":VERDICT,"integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1,"pass":Q0==Q1},
"notes":["TEST413 fixed dataset, extraction, global fingerprint mapping, shared Qwen-bank solve, TRAIN-only dose and L14-L23 write are unchanged.","Observed Mistral-derived state is produced by the original L14-L23 steering path; no steering write occurs at L24-L27.","Matched controls intervene only at the L27 input after the original upstream path has been measured.","Every matched control has the same L26 displacement norm and the same cosine to native Qwen FACT→CF L26 as the observed displacement.","The control orthogonal component is explicitly orthogonal to both native L26 and the observed displacement orthogonal component.","L27 endpoint is the standard final-normalized hidden state.","MLP_OFF uses the TEST413 fixed residual-boundary ablation.","FINAL FACT/CF remains evaluation-only; no model weights are modified; behavioral generation remains OFF."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("="*128);print("TEST 414 — COMPLETE");print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS");print("SUMMARY:",OUT);print("="*128)
