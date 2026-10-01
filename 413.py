# TEST 413 — L27 CAUSAL NECESSITY ABLATION — FIXED
# TEST412 FIXED EXACT LINEAGE — WRITE L14–L23 | MOTOR OFF L24–L27
# NORMAL vs ATTENTION-OFF vs MLP-OFF vs FULL-L27-BYPASS | L27 NATIVE FACT→CF ENDPOINT
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
ROOT=Path("/content/AKBASCORE_TEST413")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST413");ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/"TEST413_SUMMARY.json"
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
print("="*128);print("TEST 413 — L27 CAUSAL NECESSITY ABLATION — FIXED");print("TEST412 FIXED EXACT LINEAGE | WRITE L14–L23 | MOTOR OFF L24–L27 | NORMAL / ATT-OFF / MLP-OFF / FULL-BYPASS");print("="*128)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/10] MISTRAL — TEST412 EXACT PACKETS")
mt,mm,ml=load(MID,(32,4096,32,8,128));mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight];M0=sha(mp);MH=[]
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 MH.append((forge(mt,mm,a),forge(mt,mm,b)));print(f" MISTRAL {i+1}/32",end="\r")
MFINAL=(forge(mt,mm,FACT),forge(mt,mm,CF));print()
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/10] QWEN — TEST412 EXACT PACKETS")
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
print("[5/10] TRAIN-ONLY DOSE | WRITE L14–L23 | L27 ABLATION ARMS")
MAG={L:sum(float((QH[i][1][L]-QH[i][0][L]).norm())for i in TRAIN)/len(TRAIN)for L in LAYERS}
base_ids=torch.tensor([[pad(qt)]+ids(qt,FACT+SEP)],device=DEV);nsep=len(ids(qt,SEP));POS=base_ids.shape[1]-nsep-1
# Qwen final hidden state after L27 passes through qm.model.norm.
# Ablations act on the raw L27 block residual stream; endpoint remains the standard final-normalized hidden state.
def run(dirs=None,sign=1.,arm="NORMAL"):
 whs=[];hs=[];cap={}
 if dirs is not None:
  for L in WRITE_LAYERS:
   d=(dirs[L]*MAG[L]*sign).to(DEV)
   def wh(mod,inp,out,d=d):
    if isinstance(out,tuple):
     h=out[0].clone();h[:,POS,:]+=d.to(h.dtype);return (h,)+out[1:]
    h=out.clone();h[:,POS,:]+=d.to(h.dtype);return h
   whs.append(ql[L].register_forward_hook(wh))
 def block_pre(mod,inp):
  cap["INPUT_FULL"]=inp[0].detach().clone()
  cap["INPUT"]=inp[0][0,POS].detach().float().cpu().clone()
 hs.append(ql[27].register_forward_pre_hook(block_pre))
 if arm=="ATT_OFF":
  def att_off(mod,inp):
   x=inp[0].clone();x[:,POS,:]=cap["INPUT_FULL"][:,POS,:].to(x.dtype);return (x,)+inp[1:]
  hs.append(ql[27].post_attention_layernorm.register_forward_pre_hook(att_off))
 elif arm=="MLP_OFF":
  def save_post(mod,inp):cap["POST_FULL"]=inp[0].detach().clone()
  def mlp_off(mod,inp,out):
   if isinstance(out,tuple):
    h=out[0].clone();h[:,POS,:]=cap["POST_FULL"][:,POS,:].to(h.dtype);return (h,)+out[1:]
   h=out.clone();h[:,POS,:]=cap["POST_FULL"][:,POS,:].to(h.dtype);return h
  hs.append(ql[27].post_attention_layernorm.register_forward_pre_hook(save_post))
  hs.append(ql[27].register_forward_hook(mlp_off))
 elif arm=="FULL_BYPASS":
  def bypass(mod,inp,out):
   if isinstance(out,tuple):
    h=out[0].clone();h[:,POS,:]=cap["INPUT_FULL"][:,POS,:].to(h.dtype);return (h,)+out[1:]
   h=out.clone();h[:,POS,:]=cap["INPUT_FULL"][:,POS,:].to(h.dtype);return h
  hs.append(ql[27].register_forward_hook(bypass))
 with torch.inference_mode():o=qm(input_ids=base_ids,use_cache=False,output_hidden_states=True,return_dict=True)
 for h in whs+hs:h.remove()
 H26=o.hidden_states[27][0,POS].detach().float().cpu().clone()
 H27=o.hidden_states[28][0,POS].detach().float().cpu().clone()
 RAWIN=cap["INPUT"].clone()
 del o
 return {"L26":H26,"L27":H27,"RAW_L27_INPUT":RAWIN}
ARMS=["NORMAL","ATT_OFF","MLP_OFF","FULL_BYPASS"]
print("[6/10] BASELINES + PREDICTED / REVERSE ABLATION RUNS")
BASE={a:run(None,+1,a)for a in ARMS};PRED={a:run(DIRS,+1,a)for a in ARMS};REV={a:run(DIRS,-1,a)for a in ARMS}
NATIVE27=QFINAL[1][27]-QFINAL[0][27];NATIVE26=QFINAL[1][26]-QFINAL[0][26]
print("[7/10] CAUSAL NECESSITY — L27 NATIVE ALIGNMENT")
RES={}
for a in ARMS:
 dp=PRED[a]["L27"]-BASE[a]["L27"];dr=REV[a]["L27"]-BASE[a]["L27"];pc=cos(dp,NATIVE27);rc=cos(dr,NATIVE27)
 RES[a]={"pred_cos":pc,"reverse_cos":rc,"gap":pc-rc,"pred_norm":float(dp.norm()),"reverse_norm":float(dr.norm()),"native_norm":float(NATIVE27.norm())}
 print(f" {a:11s} PRED={pc:+.6f} REV={rc:+.6f} GAP={pc-rc:+.6f} |Δ|={float(dp.norm()):.3f}")
NORMAL=RES["NORMAL"]["pred_cos"]
for a in ["ATT_OFF","MLP_OFF","FULL_BYPASS"]:
 RES[a]["loss_vs_normal"]=NORMAL-RES[a]["pred_cos"]
 print(f" LOSS {a:11s} vs NORMAL = {RES[a]['loss_vs_normal']:+.6f}")
print("[8/10] EXACT ARM SANITY + L26 LOCK")
L26={}
for a in ARMS:
 d=PRED[a]["L26"]-BASE[a]["L26"];L26[a]={"cos":cos(d,NATIVE26),"norm":float(d.norm())}
 print(f" {a:11s} L26_COS={L26[a]['cos']:+.6f} |Δ|={L26[a]['norm']:.3f}")
mx=max((PRED[a]["L26"]-PRED["NORMAL"]["L26"]).norm().item()for a in ARMS)
print(f" MAX PRED L26 ARM DRIFT={mx:.8e}")
if mx>1e-4:raise RuntimeError(f"Ablation leaked upstream: {mx}")
# FIX: hidden_states[28] is the standard final-normalized model state.
# FULL_BYPASS raw block output equals L27 input, then qm.model.norm is still applied.
with torch.inference_mode():
 pin=PRED["FULL_BYPASS"]["RAW_L27_INPUT"].to(DEV,dtype=DTYPE).view(1,1,-1)
 binp=BASE["FULL_BYPASS"]["RAW_L27_INPUT"].to(DEV,dtype=DTYPE).view(1,1,-1)
 pexpect=qm.model.norm(pin)[0,0].float().cpu()
 bexpect=qm.model.norm(binp)[0,0].float().cpu()
fb_pred=float((PRED["FULL_BYPASS"]["L27"]-pexpect).norm()/pexpect.norm().clamp_min(1e-12))
fb_base=float((BASE["FULL_BYPASS"]["L27"]-bexpect).norm()/bexpect.norm().clamp_min(1e-12))
print(f" FULL_BYPASS PRED FINAL-NORM REL_ERR={fb_pred:.8e}")
print(f" FULL_BYPASS BASE FINAL-NORM REL_ERR={fb_base:.8e}")
if fb_pred>5e-3 or fb_base>5e-3:raise RuntimeError(f"Full bypass failed after final norm: pred={fb_pred}, base={fb_base}")
print("[9/10] PAIRED SHARED-BANK NULL — ABLATION LOSSES")
NULL={a:[]for a in ARMS};LOSSNULL={a:[]for a in ["ATT_OFF","MLP_OFF","FULL_BYPASS"]}
for s in range(NNULL):
 gen=torch.Generator().manual_seed(SEED+1300+s);coef=torch.randn(len(TRAIN),generator=gen);rd={}
 for L in LAYERS:rd[L]=unit(sum((coef[k]*QP[j][L]for k,j in enumerate(TRAIN)),torch.zeros_like(QP[TRAIN[0]][L])))
 rr={}
 for a in ARMS:
  r=run(rd,+1,a);d=r["L27"]-BASE[a]["L27"];rr[a]=cos(d,NATIVE27);NULL[a].append(rr[a])
 for a in LOSSNULL:LOSSNULL[a].append(rr["NORMAL"]-rr[a])
NST={};LST={}
for a in ARMS:
 v=torch.tensor(NULL[a]);mu=float(v.mean());sd=float(v.std(unbiased=True));obs=RES[a]["pred_cos"];z=(obs-mu)/max(sd,1e-12);p=float((1+(v>=obs).sum())/(len(v)+1));NST[a]={"mean":mu,"sd":sd,"z":z,"p":p}
 print(f" {a:11s} OBS={obs:+.6f} NULL={mu:+.6f}±{sd:.6f} z={z:+.3f} p={p:.6f}")
for a in LOSSNULL:
 v=torch.tensor(LOSSNULL[a]);mu=float(v.mean());sd=float(v.std(unbiased=True));obs=RES[a]["loss_vs_normal"];z=(obs-mu)/max(sd,1e-12);p=float((1+(v>=obs).sum())/(len(v)+1));LST[a]={"mean":mu,"sd":sd,"z":z,"p":p}
 print(f" LOSS {a:11s} OBS={obs:+.6f} NULL={mu:+.6f}±{sd:.6f} z={z:+.3f} p={p:.6f}")
print("[10/10] NECESSITY VERDICT + INTEGRITY")
ml=RES["MLP_OFF"]["loss_vs_normal"];al=RES["ATT_OFF"]["loss_vs_normal"];fl=RES["FULL_BYPASS"]["loss_vs_normal"]
if LST["MLP_OFF"]["p"]<=.05 and ml>0 and ml>al:
 VERDICT="L27_MLP_CAUSALLY_NECESSARY: removing the L27 MLP causes a significant paired loss of terminal native-direction alignment."
elif LST["ATT_OFF"]["p"]<=.05 and al>0 and al>ml:
 VERDICT="L27_ATTENTION_CAUSALLY_NECESSARY: removing L27 attention causes the larger significant paired loss of terminal native-direction alignment."
elif LST["MLP_OFF"]["p"]<=.05 and LST["ATT_OFF"]["p"]<=.05 and ml>0 and al>0:
 VERDICT="L27_ATTENTION_AND_MLP_JOINTLY_NECESSARY: both component removals significantly reduce terminal alignment."
elif LST["FULL_BYPASS"]["p"]<=.05 and fl>0:
 VERDICT="L27_BLOCK_CAUSALLY_NECESSARY: bypassing the complete L27 block significantly reduces terminal alignment, but single-component necessity is unresolved."
else:
 VERDICT="NO_SIGNIFICANT_L27_NECESSITY_ABLATION"
print(f" NORMAL={NORMAL:+.6f}")
print(f" ATT_OFF={RES['ATT_OFF']['pred_cos']:+.6f} LOSS={al:+.6f} p={LST['ATT_OFF']['p']:.6f}")
print(f" MLP_OFF={RES['MLP_OFF']['pred_cos']:+.6f} LOSS={ml:+.6f} p={LST['MLP_OFF']['p']:.6f}")
print(f" FULL_BYPASS={RES['FULL_BYPASS']['pred_cos']:+.6f} LOSS={fl:+.6f} p={LST['FULL_BYPASS']['p']:.6f}")
print(" VERDICT:",VERDICT)
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Qwen not frozen")
REPORT={"test":413,"title":"L27 Causal Necessity Ablation — Fixed","seed":SEED,"source_model":MID,"target_model":QID,"fact":FACT,"counterfactual":CF,
"train_pairs":24,"validation_pairs":8,"layer_map":LMAP,"write_layers":WRITE_LAYERS,"engine_off_layers":[24,25,26,27],"selected_lambda":LAM,"cross_validation":CV,
"global_final":{"fingerprint_cos":FFP,"local_mean":FLOCAL,"coef_norm":float(COEF.norm())},"arms":RES,"l26_lock":L26,
"null_alignment":NST,"paired_loss_null":LST,"full_bypass_pred_final_norm_relative_error":fb_pred,"full_bypass_base_final_norm_relative_error":fb_base,
"max_upstream_arm_drift":mx,"verdict":VERDICT,
"integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1,"pass":Q0==Q1},
"notes":["TEST412 fixed dataset, extraction, global fingerprint mapping, shared Qwen-bank solve, TRAIN-only dose and L14-L23 write window are unchanged.","No steering write occurs at L24-L27.","NORMAL leaves L27 untouched.","ATT_OFF replaces the post-attention residual state at the measured token with the exact L27 block input, removing the L27 attention residual contribution at that token.","MLP_OFF replaces the raw L27 block output at the measured token with the exact post-attention residual state, removing the L27 MLP residual contribution at that token.","FULL_BYPASS replaces the raw L27 block output at the measured token with the exact raw L27 block input.","The standard Qwen final norm remains active in every arm, including FULL_BYPASS.","FIX: FULL_BYPASS sanity compares hidden_states[28] against model.norm(raw L27 input), not against the unnormalized raw L27 input.","All four arms share identical upstream L14-L23 steering and are checked for zero L26 arm drift.","Paired shared-bank null evaluates both absolute terminal alignment and loss relative to NORMAL.","FINAL FACT/CF is evaluation-only; no model weights are modified; behavioral generation remains OFF."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("="*128);print("TEST 413 — COMPLETE");print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS");print("SUMMARY:",OUT);print("="*128)
