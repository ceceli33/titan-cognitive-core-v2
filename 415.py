# TEST 415 — MOTOR-OFF LOGIT READOUT BRIDGE
# TEST414 PROVEN WORKING LINEAGE — SAME GLOBAL SOLVE | WRITE L14–L23 | MOTOR OFF L24–L27
# GEOMETRY -> LM-HEAD READOUT | MUSTAFA vs LEYLA | NORMAL / REVERSE / MATCHED-NULL
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
ROOT=Path("/content/AKBASCORE_TEST415")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST415");ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/"TEST415_SUMMARY.json"
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
TRAIN=list(range(24));VAL=list(range(24,32));LAYERS=list(range(28));RIDGES=[1e-4,1e-3,1e-2,.05,.1,.5,1.,5.];WRITE_LAYERS=list(range(14,24));NNULL=64
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
print("="*128);print("TEST 415 — MOTOR-OFF LOGIT READOUT BRIDGE");print("TEST414 PROVEN LINEAGE | WRITE L14–L23 | MOTOR OFF L24–L27 | MUSTAFA↔LEYLA LM-HEAD READOUT");print("="*128)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/11] MISTRAL — TEST414 EXACT PACKETS")
mt,mm,ml=load(MID,(32,4096,32,8,128));mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight];M0=sha(mp);MH=[]
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 MH.append((forge(mt,mm,a),forge(mt,mm,b)));print(f" MISTRAL {i+1}/32",end="\r")
MFINAL=(forge(mt,mm,FACT),forge(mt,mm,CF));print()
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/11] QWEN — TEST414 EXACT PACKETS")
qt,qm,ql=load(QID,(28,3584,28,4,128));qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight];Q0=sha(qp);QH=[]
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 QH.append((forge(qt,qm,a),forge(qt,qm,b)));print(f" QWEN {i+1}/32",end="\r")
QFINAL=(forge(qt,qm,FACT),forge(qt,qm,CF));print("\n TRAIN=24 | VAL=8 | FINAL FACT/CF NEVER USED FOR FIT/SELECTION")
MP=[packet(a,b,LMAP)for a,b in MH];QP=[packet(a,b,LAYERS)for a,b in QH];MF=packet(MFINAL[0],MFINAL[1],LMAP);QF=packet(QFINAL[0],QFINAL[1],LAYERS)
print("[3/11] TRAIN-ONLY GLOBAL RIDGE")
CV=[]
for lam in RIDGES:
 fc,lc=loo_score(MP,QP,lam);CV.append({"lambda":lam,"fp_cos":fc,"local_cos":lc});print(f" λ={lam:<6g} FP={fc:+.6f} LOCAL={lc:+.6f}")
BEST=max(CV,key=lambda x:(x["local_cos"],x["fp_cos"]));LAM=BEST["lambda"];print(" SELECTED λ=",LAM)
print("[4/11] FINAL GLOBAL SOLVE — FROZEN")
FIT=fit_global(MP,QP,TRAIN,LAM);QHAT=pred(gfp(MF,MP,TRAIN),FIT);QTRUE=gfp(QF,QP,TRAIN);DIRS,COEF=global_qsolve(QHAT,QP,TRAIN,LAM)
FFP=cos(QHAT,QTRUE);FLOCAL=sum(cos(DIRS[L],QF[L])for L in LAYERS)/28
print(f" FINAL FP={FFP:+.6f} LOCAL={FLOCAL:+.6f} COEF_NORM={float(COEF.norm()):.6f}")
print("[5/11] TRAIN-ONLY DOSE | WRITE L14–L23 | MOTOR OFF L24–L27")
MAG={L:sum(float((QH[i][1][L]-QH[i][0][L]).norm())for i in TRAIN)/len(TRAIN)for L in LAYERS}
base_ids=torch.tensor([[pad(qt)]+ids(qt,FACT+SEP)],device=DEV);nsep=len(ids(qt,SEP));POS=base_ids.shape[1]-nsep-1
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
 def capin(mod,inp):cap["L26"]=inp[0][0,POS].detach().float().cpu().clone()
 hs.append(ql[27].register_forward_pre_hook(capin))
 with torch.inference_mode():o=qm(input_ids=base_ids,use_cache=False,output_hidden_states=True,return_dict=True)
 for h in whs+hs:h.remove()
 h27=o.hidden_states[28][0,POS].detach().float().cpu().clone();logits=o.logits[0,POS].detach().float().cpu().clone();del o
 return {"L26":cap["L26"],"L27":h27,"LOGITS":logits}
BASE=run();PRED=run(DIRS,+1);REV=run(DIRS,-1)
N26=QFINAL[1][26]-QFINAL[0][26];N27=QFINAL[1][27]-QFINAL[0][27];D26=PRED["L26"]-BASE["L26"];D27=PRED["L27"]-BASE["L27"];TN=float(D26.norm());TC=cos(D26,N26)
print(f" L26 NORM={TN:.6f} NATIVE_COS={TC:+.6f} | L27 NATIVE_COS={cos(D27,N27):+.6f}")
print("[6/11] LM-HEAD TARGET TOKENS — NO TOKEN ASSUMPTION")
# Readout is deliberately token-level, not free generation.
# Evaluate both bare-name and leading-space tokenizations; first token of each name must differ.
NAME_A="Mustafa Akbaş";NAME_B="Leyla Demir"
forms=[]
for prefix in [""," "]:
 aa=ids(qt,prefix+NAME_A);bb=ids(qt,prefix+NAME_B)
 if aa and bb and aa[0]!=bb[0]:forms.append({"prefix":prefix,"A":aa,"B":bb,"a0":aa[0],"b0":bb[0]})
if not forms:raise RuntimeError("Mustafa/Leyla first-token contrast unavailable")
for f in forms:print(f" PREFIX={repr(f['prefix'])} | MUSTAFA={f['A']} | LEYLA={f['B']} | FIRST={f['a0']} vs {f['b0']}")
def margin(logits,a,b):return float(logits[a]-logits[b])
def readout(r):
 vals=[]
 for f in forms:vals.append(margin(r["LOGITS"],f["a0"],f["b0"]))
 return float(sum(vals)/len(vals)),vals
BMG,BV=readout(BASE);PMG,PV=readout(PRED);RMG,RV=readout(REV)
# Because the learned FACT→CF direction is FACT(Mustafa) -> CF(Leyla),
# +DIRS is expected to move readout toward Leyla, so use LEYLA−MUSTAFA causal shift.
SHIFT=-(PMG-BMG);REVSHIFT=-(RMG-BMG)
print("[7/11] DIRECT MOTOR-OFF LOGIT READOUT")
print(f" BASE MUSTAFA−LEYLA={BMG:+.6f}")
print(f" +WRITE MUSTAFA−LEYLA={PMG:+.6f} | LEYLA-SHIFT={SHIFT:+.6f}")
print(f" -WRITE MUSTAFA−LEYLA={RMG:+.6f} | LEYLA-SHIFT={REVSHIFT:+.6f}")
for i,f in enumerate(forms):print(f"  PREFIX={repr(f['prefix'])}: BASE={BV[i]:+.6f} +WRITE={PV[i]:+.6f} -WRITE={RV[i]:+.6f}")
print("[8/11] TEST414 MATCHED L26 NULLS")
U26=unit(N26);OBS_ORTH=unit(unit(D26)-torch.dot(unit(D26),U26)*U26);CONTROLS=[]
for s in range(NNULL):
 g=torch.Generator().manual_seed(SEED+1400+s);r=torch.randn(D26.numel(),generator=g);r=r-torch.dot(r,U26)*U26;r=r-torch.dot(r,OBS_ORTH)*OBS_ORTH
 if r.norm()<1e-8:raise RuntimeError("Degenerate matched-null direction")
 r=unit(r);v=TC*U26+math.sqrt(max(0.,1.-TC*TC))*r;v=unit(v)*TN;CONTROLS.append(v)
errn=max(abs(float(v.norm())-TN)for v in CONTROLS);errc=max(abs(cos(v,N26)-TC)for v in CONTROLS);orthmax=max(abs(cos(v-TC*TN*U26,OBS_ORTH))for v in CONTROLS)
print(f" MAX NORM ERROR={errn:.8e} | MAX COS ERROR={errc:.8e} | MAX ORTH COS={orthmax:.8e}")
if errn>1e-3 or errc>1e-5 or orthmax>1e-5:raise RuntimeError("Matched-null construction failed")
print("[9/11] MATCHED-NULL LM-HEAD READOUT")
NULL_SHIFT=[];NULL_L27=[]
for i,v in enumerate(CONTROLS):
 R=run(None,+1,BASE["L26"]+v);mg,_=readout(R);NULL_SHIFT.append(-(mg-BMG));NULL_L27.append(cos(R["L27"]-BASE["L27"],N27));print(f" MATCHED {i+1}/{NNULL}",end="\r")
print()
def stat(vals,obs):
 v=torch.tensor(vals);mu=float(v.mean());sd=float(v.std(unbiased=True));z=(obs-mu)/max(sd,1e-12);p=float((1+(v>=obs).sum())/(len(v)+1))
 return {"obs":obs,"mean":mu,"sd":sd,"z":z,"p":p,"min":float(v.min()),"max":float(v.max())}
SS=stat(NULL_SHIFT,SHIFT);SL=stat(NULL_L27,cos(D27,N27))
print(f" LOGIT SHIFT OBS={SS['obs']:+.6f} NULL={SS['mean']:+.6f}±{SS['sd']:.6f} z={SS['z']:+.3f} p={SS['p']:.6f}")
print(f" L27 ALIGN   OBS={SL['obs']:+.6f} NULL={SL['mean']:+.6f}±{SL['sd']:.6f} z={SL['z']:+.3f} p={SL['p']:.6f}")
print("[10/11] NATIVE LM-HEAD DIRECTION SANITY")
# Independent endpoint sanity: native FACT->CF hidden delta should move LM-head projection toward CF/Leyla if the chosen token contrast is behaviorally aligned.
with torch.inference_mode():
 w=qm.lm_head.weight.detach().float().cpu()
native_token_proj=[]
for f in forms:
 readvec=w[f["b0"]]-w[f["a0"]]
 native_token_proj.append(float(torch.dot(N27.float(),readvec)))
NPROJ=float(sum(native_token_proj)/len(native_token_proj))
print(f" NATIVE FACT→CF PROJECTION ON LEYLA−MUSTAFA LM-HEAD AXIS={NPROJ:+.6f}")
for i,f in enumerate(forms):print(f"  PREFIX={repr(f['prefix'])}: {native_token_proj[i]:+.6f}")
print("[11/11] VERDICT + INTEGRITY")
if SHIFT>0 and SS["p"]<=.05 and SL["p"]<=.05:
 VERDICT="MOTOR_OFF_LOGIT_BRIDGE_DETECTED: the Mistral-derived write produces a matched-null-significant LM-head shift toward the CF identity after L14-L23 steering and an engine-off L24-L27 tail."
elif SHIFT>0 and SS["p"]<=.05:
 VERDICT="LOGIT_READOUT_SPECIFICITY_DETECTED: the Mistral-derived state produces a matched-null-significant identity-token logit shift, while terminal geometric specificity is not jointly significant."
elif SHIFT>0:
 VERDICT="POSITIVE_LOGIT_SHIFT_NOT_SIGNIFICANT: the readout moves toward the CF identity but does not exceed matched L26 controls."
else:
 VERDICT="NO_POSITIVE_MOTOR_OFF_LOGIT_BRIDGE: terminal geometry has not produced the expected CF-identity token-logit shift."
print(" VERDICT:",VERDICT)
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Qwen not frozen")
REPORT={"test":415,"title":"Motor-Off Logit Readout Bridge","seed":SEED,"source_model":MID,"target_model":QID,"fact":FACT,"counterfactual":CF,
"train_pairs":24,"validation_pairs":8,"layer_map":LMAP,"write_layers":WRITE_LAYERS,"engine_off_layers":[24,25,26,27],"selected_lambda":LAM,"cross_validation":CV,
"global_final":{"fingerprint_cos":FFP,"local_mean":FLOCAL,"coef_norm":float(COEF.norm())},
"geometry":{"l26_norm":TN,"l26_native_cos":TC,"l27_native_cos":cos(D27,N27)},
"tokens":forms,"readout":{"base_mustafa_minus_leyla":BMG,"pred_mustafa_minus_leyla":PMG,"reverse_mustafa_minus_leyla":RMG,"cf_shift":SHIFT,"reverse_cf_shift":REVSHIFT,"native_fact_cf_lmhead_projection":NPROJ},
"matched_null":{"n":NNULL,"max_norm_error":errn,"max_native_cos_error":errc,"max_control_orth_to_observed_orth_cos":orthmax,"logit_shift":SS,"l27_alignment":SL,"logit_shift_values":NULL_SHIFT,"l27_values":NULL_L27},
"verdict":VERDICT,"integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1,"pass":Q0==Q1},
"notes":["TEST414 proven working dataset, extraction, global fingerprint mapping, shared Qwen-bank solve, TRAIN-only dose and L14-L23 write are unchanged.","FINAL FACT/CF is evaluation-only and never used for fitting or hyperparameter selection.","No steering write occurs at L24-L27.","The primary new endpoint is the frozen Qwen LM-head token-logit contrast Mustafa versus Leyla at the same source-text endpoint used by TEST414.","Positive +DIRS follows the native FACT(Mustafa)->CF(Leyla) orientation; therefore the causal readout is defined as the shift toward Leyla.","Matched controls exactly follow TEST414: same L26 displacement norm, same native L26 cosine, orthogonal component separated from the observed orthogonal component.","This test measures token-level LM-head readout, not free-form WHO generation.","No model weights are modified."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("="*128);print("TEST 415 — COMPLETE");print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS");print("SUMMARY:",OUT);print("="*128)
