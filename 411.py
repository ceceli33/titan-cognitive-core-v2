# TEST 411 — ENGINE-OFF DOWNSTREAM TRANSPORT X-RAY
# TEST410 EXACT LINEAGE — WRITE ONLY L14–L23 | MOTOR OFF L24–L27
# MEASURE HOW THE PREDICTED CROSS-MODEL DISPLACEMENT ROTATES/GROWS AFTER THE LAST WRITE
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
ROOT=Path("/content/AKBASCORE_TEST411")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST411");ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/"TEST411_SUMMARY.json"
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
for s,(a,b) in zip(NEUTRAL,SWAP):
 assert s.startswith(a) and a!=b;ALT.append(b+s[len(a):])
LMAP=[min(31,max(0,round((L+.5)*32/28-.5)))for L in range(28)]
TRAIN=list(range(24));VAL=list(range(24,32));LAYERS=list(range(28));RIDGES=[1e-4,1e-3,1e-2,.05,.1,.5,1.,5.]
WRITE_LAYERS=list(range(14,24));OFF_LAYERS=list(range(24,28));NNULL=64
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
def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-12))
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
print("="*128);print("TEST 411 — ENGINE-OFF DOWNSTREAM TRANSPORT X-RAY");print("TEST410 EXACT GLOBAL SOLVE | WRITE L14–L23 | MOTOR OFF L24–L27 | LAYERWISE Δ TRANSPORT");print("="*128)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/10] MISTRAL — TEST410 EXACT PACKETS")
mt,mm,ml=load(MID,(32,4096,32,8,128));mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight];M0=sha(mp);MH=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):
 MH.append((forge(mt,mm,a),forge(mt,mm,b)));print(f" MISTRAL {i+1}/32",end="\r")
MFINAL=(forge(mt,mm,FACT),forge(mt,mm,CF));print()
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/10] QWEN — TEST410 EXACT PACKETS")
qt,qm,ql=load(QID,(28,3584,28,4,128));qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight];Q0=sha(qp);QH=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):
 QH.append((forge(qt,qm,a),forge(qt,qm,b)));print(f" QWEN {i+1}/32",end="\r")
QFINAL=(forge(qt,qm,FACT),forge(qt,qm,CF));print("\n TRAIN=24 | VAL=8 | FINAL FACT/CF NEVER USED FOR FIT/SELECTION")
MP=[packet(a,b,LMAP)for a,b in MH];QP=[packet(a,b,LAYERS)for a,b in QH];MF=packet(MFINAL[0],MFINAL[1],LMAP);QF=packet(QFINAL[0],QFINAL[1],LAYERS)
print("[3/10] TRAIN-ONLY GLOBAL RIDGE")
CV=[]
for lam in RIDGES:
 fc,lc=loo_score(MP,QP,lam);CV.append({"lambda":lam,"fp_cos":fc,"local_cos":lc});print(f" λ={lam:<6g} FP={fc:+.6f} LOCAL={lc:+.6f}")
BEST=max(CV,key=lambda x:(x["local_cos"],x["fp_cos"]));LAM=BEST["lambda"];print(" SELECTED λ=",LAM)
print("[4/10] FINAL GLOBAL SOLVE — FROZEN BEFORE WRITE")
FIT=fit_global(MP,QP,TRAIN,LAM);QHAT=pred(gfp(MF,MP,TRAIN),FIT);QTRUE=gfp(QF,QP,TRAIN);DIRS,COEF=global_qsolve(QHAT,QP,TRAIN,LAM)
FFP=cos(QHAT,QTRUE);FLOCAL=sum(cos(DIRS[L],QF[L])for L in LAYERS)/28
print(f" FINAL FP={FFP:+.6f} LOCAL={FLOCAL:+.6f} COEF_NORM={float(COEF.norm()):.6f}")
print("[5/10] TRAIN-ONLY DOSE | MOTOR L14–L23 ONLY")
MAG={L:sum(float((QH[i][1][L]-QH[i][0][L]).norm())for i in TRAIN)/len(TRAIN)for L in LAYERS}
print(" WRITE:",WRITE_LAYERS);print(" ENGINE OFF:",OFF_LAYERS)
BASE_TEXT=FACT;base_ids=torch.tensor([[pad(qt)]+ids(qt,BASE_TEXT+SEP)],device=DEV);nsep=len(ids(qt,SEP));WRITE_POS=base_ids.shape[1]-nsep-1
def run(dirs,sign=1.):
 handles=[]
 for L in WRITE_LAYERS:
  d=(dirs[L]*MAG[L]*sign).to(DEV)
  def hook(mod,inp,out,d=d):
   if isinstance(out,tuple):
    h=out[0].clone();h[:,WRITE_POS,:]+=d.to(h.dtype);return (h,)+out[1:]
   h=out.clone();h[:,WRITE_POS,:]+=d.to(h.dtype);return h
  handles.append(ql[L].register_forward_hook(hook))
 with torch.inference_mode():o=qm(input_ids=base_ids,use_cache=False,output_hidden_states=True,return_dict=True)
 for h in handles:h.remove()
 H=[o.hidden_states[L+1][0,WRITE_POS].float().cpu().contiguous()for L in LAYERS];del o;return H
BASEH=forge(qt,qm,BASE_TEXT);NATIVE=[QFINAL[1][L]-QFINAL[0][L]for L in LAYERS]
print("[6/10] PREDICTED / REVERSE ENGINE-OFF RUN")
HP=run(DIRS,+1);HR=run(DIRS,-1)
print("[7/10] LAYERWISE X-RAY — WRITE REGION + ENGINE-OFF TAIL")
XRAY={}
for L in range(14,28):
 dp=HP[L]-BASEH[L];dr=HR[L]-BASEH[L];nat=NATIVE[L]
 cp=cos(dp,nat);cr=cos(dr,nat);np=float(dp.norm());nr=float(dr.norm());nn=float(nat.norm())
 XRAY[L]={"phase":"WRITE"if L<=23 else"OFF","pred_cos":cp,"reverse_cos":cr,"gap":cp-cr,"pred_norm":np,"reverse_norm":nr,"native_norm":nn,"norm_ratio":np/max(nn,1e-12),"local_dir_cos":cos(DIRS[L],QF[L])}
 print(f" L{L:02d} {'WRITE' if L<=23 else 'OFF  '} PRED={cp:+.6f} REV={cr:+.6f} GAP={cp-cr:+.6f} |Δ|={np:.3f} NATIVE={nn:.3f} RATIO={np/max(nn,1e-12):.3f}")
print("[8/10] ENGINE-OFF TRANSPORT — ROTATION / GROWTH")
TRANSPORT={}
d23=HP[23]-BASEH[23]
for L in OFF_LAYERS:
 d=HP[L]-BASEH[L];TRANSPORT[L]={"native_cos":cos(d,NATIVE[L]),"cos_to_L23":cos(d,d23),"norm":float(d.norm()),"growth_vs_L23":float(d.norm()/d23.norm().clamp_min(1e-12))}
 print(f" L23→L{L:02d} NATIVE_COS={TRANSPORT[L]['native_cos']:+.6f} COS_TO_L23={TRANSPORT[L]['cos_to_L23']:+.6f} GROWTH={TRANSPORT[L]['growth_vs_L23']:.6f}")
TERM=XRAY[27]["pred_cos"];REVTERM=XRAY[27]["reverse_cos"]
print("[9/10] SHARED-BANK NULL — SAME L14–L23 WRITE, MOTOR OFF TAIL")
NULL=[];NULLPATH={L:[]for L in OFF_LAYERS}
for s in range(NNULL):
 gen=torch.Generator().manual_seed(SEED+1100+s);a=torch.randn(len(TRAIN),generator=gen);rd={}
 for L in LAYERS:rd[L]=unit(sum((a[k]*QP[j][L]for k,j in enumerate(TRAIN)),torch.zeros_like(QP[TRAIN[0]][L])))
 H=run(rd,+1)
 for L in OFF_LAYERS:NULLPATH[L].append(cos(H[L]-BASEH[L],NATIVE[L]))
 NULL.append(NULLPATH[27][-1])
v=torch.tensor(NULL);mu=float(v.mean());sd=float(v.std(unbiased=True));z=(TERM-mu)/max(sd,1e-12);p=float((1+(v>=TERM).sum())/(len(v)+1))
for L in OFF_LAYERS:
 vv=torch.tensor(NULLPATH[L]);m=float(vv.mean());s=float(vv.std(unbiased=True));obs=XRAY[L]["pred_cos"];zz=(obs-m)/max(s,1e-12);pp=float((1+(vv>=obs).sum())/(len(vv)+1))
 XRAY[L]["null_mean"]=m;XRAY[L]["null_sd"]=s;XRAY[L]["null_z"]=zz;XRAY[L]["null_p"]=pp
 print(f" L{L:02d} OBS={obs:+.6f} NULL={m:+.6f}±{s:.6f} z={zz:+.3f} p={pp:.6f}")
print("[10/10] VERDICT + INTEGRITY")
c23=XRAY[23]["pred_cos"];c24=XRAY[24]["pred_cos"];c27=XRAY[27]["pred_cos"];gain=c27-c23
if p<=.05 and c27>REVTERM+.10 and gain>.05:
 VERDICT="ENGINE_OFF_AMPLIFICATION: the L14–L23 write continues to rotate/amplify toward the native FACT→CF direction after the motor is off."
elif p<=.05 and c27>REVTERM+.10:
 VERDICT="ENGINE_OFF_RETENTION: the significant L27 alignment survives through unwritten L24–L27, but terminal alignment is not strongly amplified beyond L23."
elif p<=.05:
 VERDICT="ENGINE_OFF_SIGNAL: L27 remains significant after motor shutdown, but sign-control separation is limited."
else:
 VERDICT="NO_SIGNIFICANT_ENGINE_OFF_TERMINAL: L14–L23 transport does not survive the shared-bank null at L27."
print(f" L23={c23:+.6f} → L24={c24:+.6f} → L27={c27:+.6f} | GAIN23→27={gain:+.6f}")
print(f" L27 PRED={TERM:+.6f} REV={REVTERM:+.6f} NULL={mu:+.6f}±{sd:.6f} z={z:+.3f} p={p:.6f}")
print(" VERDICT:",VERDICT)
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Qwen not frozen")
REPORT={"test":411,"title":"Engine-Off Downstream Transport X-Ray","source_model":MID,"target_model":QID,"seed":SEED,"fact":FACT,"counterfactual":CF,
"train_pairs":24,"validation_pairs":8,"layer_map":LMAP,"write_layers":WRITE_LAYERS,"engine_off_layers":OFF_LAYERS,"ridge_grid":RIDGES,"cross_validation":CV,"selected_lambda":LAM,
"global_final":{"fingerprint_cos":FFP,"local_mean":FLOCAL,"coef_norm":float(COEF.norm())},"xray":XRAY,"transport":TRANSPORT,
"terminal":{"pred":TERM,"reverse":REVTERM,"null_mean":mu,"null_sd":sd,"z":z,"p":p,"gain_L23_to_L27":gain},"verdict":VERDICT,
"integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1,"pass":Q0==Q1},
"notes":["TEST410 dataset, extraction, global fingerprint mapping, shared Qwen-bank solve and TRAIN-only dose are unchanged.","Writes occur only at Qwen L14–L23.","No hooks are installed at L24–L27; these layers are an engine-off observation tail.","Layerwise displacement is measured against the native held-out Qwen FACT→CF displacement at the same layer.","Reverse-sign control uses the identical write layers and magnitudes.","Shared-bank null uses the identical L14–L23 write window and randomized shared TRAIN-bank coefficients.","FINAL FACT/CF never changes fit, ridge, dose, write directions or window selection.","No model weights are modified; behavioral generation remains OFF."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("="*128);print("TEST 411 — COMPLETE");print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS");print("SUMMARY:",OUT);print("="*128)
