# TEST 416 — FULL-NAME SEQUENCE LOG-PROBABILITY BRIDGE
# TEST415 PROVEN WORKING LINEAGE | WRITE L14–L23 | MOTOR OFF L24–L27
# TEACHER-FORCED P(MUSTAFA AKBAŞ) vs P(LEYLA DEMIR) | NORMAL / REVERSE / MATCHED-NULL
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
ROOT=Path("/content/AKBASCORE_TEST416")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST416");ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/"TEST416_SUMMARY.json"
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
for s,(a,b)in zip(NEUTRAL,SWAP):
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
print("="*128);print("TEST 416 — FULL-NAME SEQUENCE LOG-PROBABILITY BRIDGE");print("TEST415 PROVEN LINEAGE | WRITE L14–L23 | MOTOR OFF L24–L27 | TEACHER-FORCED MUSTAFA AKBAŞ ↔ LEYLA DEMIR");print("="*128)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/12] MISTRAL — TEST415 EXACT PACKETS")
mt,mm,ml=load(MID,(32,4096,32,8,128));mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight];M0=sha(mp);MH=[]
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 MH.append((forge(mt,mm,a),forge(mt,mm,b)));print(f" MISTRAL {i+1}/32",end="\r")
MFINAL=(forge(mt,mm,FACT),forge(mt,mm,CF));print()
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/12] QWEN — TEST415 EXACT PACKETS")
qt,qm,ql=load(QID,(28,3584,28,4,128));qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight];Q0=sha(qp);QH=[]
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
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
print("[5/12] TEST415 EXACT SOURCE-ENDPOINT GEOMETRY")
MAG={L:sum(float((QH[i][1][L]-QH[i][0][L]).norm())for i in TRAIN)/len(TRAIN)for L in LAYERS}
source_ids=torch.tensor([[pad(qt)]+ids(qt,FACT+SEP)],device=DEV);nsep=len(ids(qt,SEP));POS=source_ids.shape[1]-nsep-1
def source_run(dirs=None,sign=1.,l26_replace=None):
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
 def capin(mod,inp):cap["L26"]=inp[0][0,POS].detach().float().cpu().clone()
 hs.append(ql[27].register_forward_pre_hook(capin))
 with torch.inference_mode():o=qm(input_ids=source_ids,use_cache=False,output_hidden_states=True,return_dict=True)
 for h in whs+hs:h.remove()
 r={"L26":cap["L26"],"L27":o.hidden_states[28][0,POS].detach().float().cpu().clone()};del o;return r
BASE=source_run();PRED=source_run(DIRS,+1);REV=source_run(DIRS,-1)
N26=QFINAL[1][26]-QFINAL[0][26];N27=QFINAL[1][27]-QFINAL[0][27];D26=PRED["L26"]-BASE["L26"];D27=PRED["L27"]-BASE["L27"];TN=float(D26.norm());TC=cos(D26,N26);L27C=cos(D27,N27)
print(f" L26 NORM={TN:.6f} NATIVE_COS={TC:+.6f} | L27 NATIVE_COS={L27C:+.6f}")
print("[6/12] TEACHER-FORCED FULL-NAME READOUT SETUP")
PROMPT="Who planted the Turkish flag at the base of the Golden Gate Bridge?\nAnswer:"
CANDS=[NAME_A,NAME_B]
def candidate_pack(name):
 prefix=[pad(qt)]+ids(qt,PROMPT);cont=ids(qt," "+name)
 if not cont:raise RuntimeError("Empty candidate")
 return prefix,cont
PACK={n:candidate_pack(n)for n in CANDS}
for n in CANDS:print(f" {n}: {PACK[n][1]}")
def seq_score(name,dirs=None,sign=1.,l26_delta=None):
 prefix,cont=PACK[name];seq=prefix+cont;inp=torch.tensor([seq],device=DEV);start=len(prefix)-1;whs=[];hs=[]
 if dirs is not None:
  for L in WRITE_LAYERS:
   d=(dirs[L]*MAG[L]*sign).to(DEV)
   def wh(mod,inputs,out,d=d):
    if isinstance(out,tuple):
     h=out[0].clone();h[:,start,:]+=d.to(h.dtype);return (h,)+out[1:]
    h=out.clone();h[:,start,:]+=d.to(h.dtype);return h
   whs.append(ql[L].register_forward_hook(wh))
 if l26_delta is not None:
  delta=l26_delta.to(DEV)
  def rep(mod,inputs,delta=delta):
   x=inputs[0].clone();x[:,start,:]+=delta.to(x.dtype);return (x,)+inputs[1:]
  hs.append(ql[27].register_forward_pre_hook(rep))
 with torch.inference_mode():o=qm(input_ids=inp,use_cache=False,return_dict=True);lp=F.log_softmax(o.logits[0].float(),dim=-1)
 vals=[float(lp[start+j,tid])for j,tid in enumerate(cont)]
 for h in whs+hs:h.remove()
 del o
 return {"sum":sum(vals),"mean":sum(vals)/len(vals),"tokens":vals,"n":len(vals)}
def pair_score(dirs=None,sign=1.,l26_delta=None):
 a=seq_score(NAME_A,dirs,sign,l26_delta);b=seq_score(NAME_B,dirs,sign,l26_delta)
 return {"A":a,"B":b,"A_minus_B_sum":a["sum"]-b["sum"],"A_minus_B_mean":a["mean"]-b["mean"]}
print("[7/12] NORMAL / +WRITE / -WRITE FULL-NAME SEQUENCE SCORES")
SB=pair_score();SP=pair_score(DIRS,+1);SR=pair_score(DIRS,-1)
# +DIRS = native FACT(Mustafa)->CF(Leyla), therefore expected behavioral movement is toward Leyla:
SHIFT=-(SP["A_minus_B_mean"]-SB["A_minus_B_mean"]);REVSHIFT=-(SR["A_minus_B_mean"]-SB["A_minus_B_mean"])
print(f" BASE   MUSTAFA−LEYLA MEAN={SB['A_minus_B_mean']:+.6f} SUM={SB['A_minus_B_sum']:+.6f}")
print(f" +WRITE MUSTAFA−LEYLA MEAN={SP['A_minus_B_mean']:+.6f} SUM={SP['A_minus_B_sum']:+.6f} | LEYLA-SHIFT={SHIFT:+.6f}")
print(f" -WRITE MUSTAFA−LEYLA MEAN={SR['A_minus_B_mean']:+.6f} SUM={SR['A_minus_B_sum']:+.6f} | LEYLA-SHIFT={REVSHIFT:+.6f}")
print(f" +WRITE Mustafa mean={SP['A']['mean']:+.6f} | Leyla mean={SP['B']['mean']:+.6f}")
print("[8/12] TEST414/415 MATCHED L26 NULLS")
U26=unit(N26);OBS_ORTH=unit(unit(D26)-torch.dot(unit(D26),U26)*U26);CONTROLS=[]
for s in range(NNULL):
 g=torch.Generator().manual_seed(SEED+1400+s);r=torch.randn(D26.numel(),generator=g);r=r-torch.dot(r,U26)*U26;r=r-torch.dot(r,OBS_ORTH)*OBS_ORTH
 if r.norm()<1e-8:raise RuntimeError("Degenerate matched-null direction")
 r=unit(r);v=TC*U26+math.sqrt(max(0.,1.-TC*TC))*r;v=unit(v)*TN;CONTROLS.append(v)
errn=max(abs(float(v.norm())-TN)for v in CONTROLS);errc=max(abs(cos(v,N26)-TC)for v in CONTROLS);orthmax=max(abs(cos(v-TC*TN*U26,OBS_ORTH))for v in CONTROLS)
print(f" MAX NORM ERROR={errn:.8e} | MAX COS ERROR={errc:.8e} | MAX ORTH COS={orthmax:.8e}")
if errn>1e-3 or errc>1e-5 or orthmax>1e-5:raise RuntimeError("Matched-null construction failed")
print("[9/12] MATCHED-NULL FULL-NAME SEQUENCE READOUT")
# The matched displacement is injected at the same L27 input boundary used in TEST414/415.
NULL_SHIFT=[]
for i,v in enumerate(CONTROLS):
 R=pair_score(None,+1,v);NULL_SHIFT.append(-(R["A_minus_B_mean"]-SB["A_minus_B_mean"]));print(f" MATCHED {i+1}/{NNULL}",end="\r")
print()
def stat(vals,obs):
 v=torch.tensor(vals);mu=float(v.mean());sd=float(v.std(unbiased=True));z=(obs-mu)/max(sd,1e-12);p=float((1+(v>=obs).sum())/(len(v)+1))
 return {"obs":obs,"mean":mu,"sd":sd,"z":z,"p":p,"min":float(v.min()),"max":float(v.max())}
SS=stat(NULL_SHIFT,SHIFT)
print(f" SEQ SHIFT OBS={SS['obs']:+.6f} NULL={SS['mean']:+.6f}±{SS['sd']:.6f} z={SS['z']:+.3f} p={SS['p']:.6f}")
print("[10/12] SIGN / TOKEN-LEVEL CONSISTENCY")
def delta_tokens(base,arm,name,toward_b=False):
 b=base[name]["tokens"];a=arm[name]["tokens"];return [x-y for x,y in zip(a,b)]
DA=delta_tokens(SB,SP,"A");DB=delta_tokens(SB,SP,"B")
print(" MUSTAFA token ΔlogP:",["%+.6f"%x for x in DA])
print(" LEYLA   token ΔlogP:",["%+.6f"%x for x in DB])
print(f" MUSTAFA mean Δ={sum(DA)/len(DA):+.6f} | LEYLA mean Δ={sum(DB)/len(DB):+.6f}")
print("[11/12] SOURCE-ENDPOINT GEOMETRY REPLICATION")
# Keep TEST414 terminal geometric result as an integrity bridge; no new selection uses this endpoint.
NULL_L27=[]
for i,v in enumerate(CONTROLS):
 R=source_run(None,+1,BASE["L26"]+v);NULL_L27.append(cos(R["L27"]-BASE["L27"],N27));print(f" GEOM {i+1}/{NNULL}",end="\r")
print()
SL=stat(NULL_L27,L27C)
print(f" L27 ALIGN OBS={SL['obs']:+.6f} NULL={SL['mean']:+.6f}±{SL['sd']:.6f} z={SL['z']:+.3f} p={SL['p']:.6f}")
print("[12/12] VERDICT + INTEGRITY")
if SHIFT>0 and SS["p"]<=.05 and SL["p"]<=.05:
 VERDICT="FULL_NAME_SEQUENCE_BRIDGE_DETECTED: the Mistral-derived write produces a matched-null-significant teacher-forced probability shift toward Leyla Demir while preserving the TEST414 motor-off geometric specificity."
elif SHIFT>0 and SS["p"]<=.05:
 VERDICT="FULL_NAME_SEQUENCE_READOUT_DETECTED: the complete-name sequence probability shifts specifically toward Leyla Demir, while the geometric replication is not jointly significant."
elif SHIFT>0:
 VERDICT="POSITIVE_FULL_NAME_SHIFT_NOT_SIGNIFICANT: the complete-name sequence probability moves toward Leyla Demir but does not exceed matched L26 controls."
else:
 VERDICT="NO_POSITIVE_FULL_NAME_SEQUENCE_BRIDGE: the complete-name teacher-forced readout does not move in the expected CF direction."
print(" VERDICT:",VERDICT)
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Qwen not frozen")
REPORT={"test":416,"title":"Full-Name Sequence Log-Probability Bridge","seed":SEED,"source_model":MID,"target_model":QID,"fact":FACT,"counterfactual":CF,
"prompt":PROMPT,"candidate_a":NAME_A,"candidate_b":NAME_B,"train_pairs":24,"validation_pairs":8,"layer_map":LMAP,"write_layers":WRITE_LAYERS,"engine_off_layers":[24,25,26,27],
"selected_lambda":LAM,"cross_validation":CV,"global_final":{"fingerprint_cos":FFP,"local_mean":FLOCAL,"coef_norm":float(COEF.norm())},
"geometry":{"l26_norm":TN,"l26_native_cos":TC,"l27_native_cos":L27C,"matched_null":SL},
"sequence_readout":{"base":SB,"pred":SP,"reverse":SR,"cf_shift":SHIFT,"reverse_cf_shift":REVSHIFT,"matched_null":SS},
"matched_null":{"n":NNULL,"max_norm_error":errn,"max_native_cos_error":errc,"max_control_orth_to_observed_orth_cos":orthmax,"sequence_shift_values":NULL_SHIFT,"l27_values":NULL_L27},
"verdict":VERDICT,"integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1,"pass":Q0==Q1},
"notes":["TEST415 proven working extraction, datasets, global fingerprint mapping, Qwen-bank solve, TRAIN-only dose and L14-L23 write are preserved.","FINAL FACT/CF remains evaluation-only and is never used for fit or hyperparameter selection.","The new behavioral endpoint is teacher-forced full-name sequence log-probability under the fixed WHO prompt.","Candidate score uses mean token log-probability to avoid raw sequence-length bias; raw summed log-probability is also recorded.","The +DIRS orientation is native FACT(Mustafa)->CF(Leyla), so positive causal shift means increased relative preference for Leyla Demir.","Matched controls preserve TEST414/415 L26 norm and native cosine and are applied at the L27 input boundary.","TEST414 source-endpoint L27 geometric specificity is independently replicated in the same run.","No model weights are modified and free-form generation remains OFF."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("="*128);print("TEST 416 — COMPLETE");print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS");print("SUMMARY:",OUT);print("="*128)
