# TEST 408 — GLOBAL RELATIONAL BANK SOLVE
# TEST405/407 EXACT LINEAGE — NO LAYER SELECTION, NO RAW 4096→3584 MAP
# MISTRAL GLOBAL FINGERPRINT → QWEN GLOBAL FINGERPRINT → ONE SHARED BANK COEFFICIENT VECTOR → 28 QWEN-LOCAL WRITES
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
ROOT=Path("/content/AKBASCORE_TEST408")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST408");ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/"TEST408_SUMMARY.json"
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
TRAIN=list(range(24));VAL=list(range(24,32));RIDGES=[1e-4,1e-3,1e-2,.05,.1,.5,1.,5.];LAYERS=list(range(28))
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
 # ONE shared coefficient vector. Each relation j contributes its complete 28-layer Qwen packet.
 # Gram geometry is global; resulting coefficients are reused unchanged at every Qwen layer.
 C=torch.stack([catpacket(QP[j])for j in refs]);G=C@C.T;t=target.float();A=G.T@G+lam*torch.eye(len(refs));a=torch.linalg.solve(A,G.T@t)
 dirs={L:unit(sum((a[k]*QP[j][L]for k,j in enumerate(refs)),torch.zeros_like(QP[refs[0]][L])))for L in LAYERS}
 return dirs,a,G
def loo_score(MP,QP,lam):
 fpc=[];loc=[]
 for hold in TRAIN:
  refs=[j for j in TRAIN if j!=hold];fit=fit_global(MP,QP,refs,lam);mh=gfp(MP[hold],MP,refs);qh=pred(mh,fit);qt=gfp(QP[hold],QP,refs);dirs,_,_=global_qsolve(qh,QP,refs,lam)
  fpc.append(cos(qh,qt));loc.append(sum(cos(dirs[L],QP[hold][L])for L in LAYERS)/28)
 return sum(fpc)/len(fpc),sum(loc)/len(loc)
print("="*124);print("TEST 408 — GLOBAL RELATIONAL BANK SOLVE");print("MISTRAL GLOBAL FINGERPRINT → QWEN GLOBAL FINGERPRINT → ONE SHARED BANK SOLVE → 28 QWEN-LOCAL AKBASCORE WRITES");print("="*124)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/10] MISTRAL — TEST405/407 EXACT RELATIONAL PACKETS")
mt,mm,ml=load(MID,(32,4096,32,8,128));mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight];M0=sha(mp);MH=[]
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 MH.append((forge(mt,mm,a),forge(mt,mm,b)));print(f" MISTRAL {i+1}/32",end="\r")
MFINAL=(forge(mt,mm,FACT),forge(mt,mm,CF));print()
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/10] QWEN — TEST405/407 EXACT RELATIONAL PACKETS")
qt,qm,ql=load(QID,(28,3584,28,4,128));qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight];Q0=sha(qp);QH=[]
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 QH.append((forge(qt,qm,a),forge(qt,qm,b)));print(f" QWEN {i+1}/32",end="\r")
QFINAL=(forge(qt,qm,FACT),forge(qt,qm,CF));print("\n TRAIN=24 | VAL=8 | FINAL FACT/CF NEVER USED FOR FIT/SELECTION")
MP=[packet(a,b,LMAP)for a,b in MH];QP=[packet(a,b,LAYERS)for a,b in QH];MF=packet(MFINAL[0],MFINAL[1],LMAP);QF=packet(QFINAL[0],QFINAL[1],LAYERS)
print("[3/10] TRAIN-ONLY GLOBAL RIDGE SELECTION")
CV=[]
for lam in RIDGES:
 fc,lc=loo_score(MP,QP,lam);CV.append({"lambda":lam,"fp_cos":fc,"local_cos":lc});print(f" λ={lam:<6g} FP={fc:+.6f} LOCAL={lc:+.6f}")
BEST=max(CV,key=lambda x:(x["local_cos"],x["fp_cos"]));LAM=BEST["lambda"];print(" SELECTED λ=",LAM)
print("[4/10] VALIDATION — GLOBAL FINGERPRINT + SHARED BANK SOLVE")
FIT=fit_global(MP,QP,TRAIN,LAM);VF=[];VL=[]
for i in VAL:
 mh=gfp(MP[i],MP,TRAIN);qh=pred(mh,FIT);qtg=gfp(QP[i],QP,TRAIN);dirs,_,_=global_qsolve(qh,QP,TRAIN,LAM);fc=cos(qh,qtg);lc=sum(cos(dirs[L],QP[i][L])for L in LAYERS)/28;VF.append(fc);VL.append(lc);print(f" VAL{i:02d} FP={fc:+.6f} LOCAL={lc:+.6f}")
VFP=sum(VF)/len(VF);VLOCAL=sum(VL)/len(VL);print(f" VALIDATION MEAN FP={VFP:+.6f} LOCAL={VLOCAL:+.6f}")
print("[5/10] FINAL HELD-OUT FACT↔CF — GLOBAL SOLVE")
MFINGER=gfp(MF,MP,TRAIN);QHAT=pred(MFINGER,FIT);QTRUE=gfp(QF,QP,TRAIN);DIRS,COEF,G=global_qsolve(QHAT,QP,TRAIN,LAM)
FFP=cos(QHAT,QTRUE);FREL=rel(QHAT,QTRUE);LC={L:cos(DIRS[L],QF[L])for L in LAYERS};FLOCAL=sum(LC.values())/28
print(f" FINAL FP COS={FFP:+.6f} REL={FREL:.6f}");print(f" FINAL LOCAL MEAN={FLOCAL:+.6f} | COEF NORM={float(COEF.norm()):.6f}")
for L in [0,3,6,10,14,18,19,21,23,25,27]:print(f" L{L:02d}←M{LMAP[L]:02d} LOCAL={LC[L]:+.6f}")
print("[6/10] TRAIN-ONLY AKBASCORE DOSE")
MAG={}
for L in LAYERS:
 MAG[L]=sum(float((QH[i][1][L]-QH[i][0][L]).norm())for i in TRAIN)/len(TRAIN)
print(" ΔNORM L00/L06/L14/L19/L25/L27:",", ".join(f"{MAG[L]:.3f}"for L in [0,6,14,19,25,27]))
BASE_TEXT=FACT;base_ids=torch.tensor([[pad(qt)]+ids(qt,BASE_TEXT+SEP)],device=DEV);nsep=len(ids(qt,SEP));WRITE_POS=base_ids.shape[1]-nsep-1
def run(dirs,sign=1.):
 handles=[]
 for L in LAYERS:
  d=(dirs[L]*MAG[L]*sign).to(DEV)
  def hook(mod,inp,out,d=d):
   if isinstance(out,tuple):
    h=out[0].clone();h[:,WRITE_POS,:]+=d.to(h.dtype);return (h,)+out[1:]
   h=out.clone();h[:,WRITE_POS,:]+=d.to(h.dtype);return h
  handles.append(ql[L].register_forward_hook(hook))
 with torch.inference_mode():o=qm(input_ids=base_ids,use_cache=False,output_hidden_states=True,return_dict=True)
 for h in handles:h.remove()
 H=[o.hidden_states[L+1][0,WRITE_POS].float().cpu()for L in LAYERS];del o;return H
print("[7/10] AKBASCORE CAUSAL GLOBAL WRITE — PREDICTED / REVERSE")
BASEH=forge(qt,qm,BASE_TEXT);WH=run(DIRS,+1);RH=run(DIRS,-1)
CW={};CR={}
for L in LAYERS:
 native=QFINAL[1][L]-QFINAL[0][L];CW[L]=cos(WH[L]-BASEH[L],native);CR[L]=cos(RH[L]-BASEH[L],native)
print(f" WRITE MEAN PRED={sum(CW.values())/28:+.6f} REV={sum(CR.values())/28:+.6f}")
for L in [0,3,6,10,14,18,19,21,23,25,27]:print(f" L{L:02d} PRED={CW[L]:+.6f} REV={CR[L]:+.6f}")
print("[8/10] DOWNSTREAM TERMINAL")
TERM=CW[27];REVTERM=CR[27];print(f" L27 PRED={TERM:+.6f} REV={REVTERM:+.6f}")
print("[9/10] SHARED-COEFFICIENT BANK NULL")
NULL=[]
for s in range(64):
 gen=torch.Generator().manual_seed(SEED+800+s);a=torch.randn(len(TRAIN),generator=gen);dirs={}
 for L in LAYERS:
  dirs[L]=unit(sum((a[k]*QP[j][L]for k,j in enumerate(TRAIN)),torch.zeros_like(QP[TRAIN[0]][L])))
 H=run(dirs,+1);NULL.append(cos(H[27]-BASEH[27],QFINAL[1][27]-QFINAL[0][27]))
NULL=torch.tensor(NULL);mu=float(NULL.mean());sd=float(NULL.std(unbiased=True));z=(TERM-mu)/max(sd,1e-12);pval=float((1+(NULL>=TERM).sum())/(len(NULL)+1))
print(f" L27 SHARED-BANK NULL mean={mu:+.6f} sd={sd:.6f} z={z:+.3f} p={pval:.6f}")
print("[10/10] VERDICT + INTEGRITY")
if FLOCAL>=.20 and TERM>=.20 and TERM>REVTERM+.10 and pval<=.05:VERDICT="GLOBAL_RELATIONAL_WRITE_FOUND: the TEST405 global coordinate produced a significant Qwen-native causal write through one shared relational-bank solution."
elif FLOCAL>=.12 and TERM>mu+.05:VERDICT="PARTIAL_GLOBAL_WRITE: global relational reconstruction survives into Qwen local geometry, but downstream transport is incomplete."
else:VERDICT="NO_STABLE_GLOBAL_WRITE: TEST405 global similarity does not yet reconstruct a reliable Qwen-native causal displacement."
print(f" FINAL FP={FFP:+.6f} LOCAL={FLOCAL:+.6f} | L27={TERM:+.6f} | NULL={mu:+.6f} | p={pval:.6f}");print(" VERDICT:",VERDICT)
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Qwen not frozen")
REPORT={"test":408,"title":"Global Relational Bank Solve","source_model":MID,"target_model":QID,"seed":SEED,"fact":FACT,"counterfactual":CF,
"train_pairs":24,"validation_pairs":8,"layer_map":LMAP,"ridge_grid":RIDGES,"cross_validation":CV,"selected_lambda":LAM,
"validation":{"fingerprint_cos":VFP,"local_cos":VLOCAL},"final":{"fingerprint_cos":FFP,"fingerprint_rel":FREL,"local_mean":FLOCAL,"layers":LC,"coef_norm":float(COEF.norm())},
"write_magnitudes":MAG,"causal":{"pred_mean":sum(CW.values())/28,"reverse_mean":sum(CR.values())/28,"pred_layers":CW,"reverse_layers":CR,"terminal":TERM,"reverse_terminal":REVTERM},
"null":{"type":"shared Qwen train-bank coefficient randomization","n":64,"mean":mu,"sd":sd,"z":z,"p":pval},"verdict":VERDICT,
"integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1,"pass":Q0==Q1},
"notes":["TEST405/407 dataset, post-block packet extraction, matched depth map and held-out FACT/CF lock retained.","No layer selection is performed.","No direct 4096→3584 vector regression is fitted.","Mistral supplies one global relation-to-bank fingerprint.","Qwen solves one shared TRAIN-bank coefficient vector and reuses it across all 28 native Qwen layers.","Ridge is selected using TRAIN leave-one-out only; FINAL FACT/CF never selects ridge, coefficients or dose.","AkbasCore writes all 28 Qwen-local directions using TRAIN-only native Qwen delta magnitudes.","Random null preserves the shared-coefficient Qwen-bank structure.","No weights are modified; behavioral generation remains OFF."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("="*124);print("TEST 408 — COMPLETE");print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS");print("SUMMARY:",OUT);print("="*124)
