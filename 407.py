# TEST 407 — FINGERPRINT-CONSTRAINED QWEN LOCAL SOLVE
# TEST405/406 EXACT LINEAGE — MISTRAL SENDS ONLY RELATION-TO-BANK COORDINATE
# TRAIN→VAL SELECTS LAYERS/RIDGE — QWEN BANK SOLVES ITS OWN 3584-D WRITE — FINAL NEVER SELECTS
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
ROOT=Path("/content/AKBASCORE_TEST407")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST407");ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/"TEST407_SUMMARY.json"
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
TRAIN=list(range(24));VAL=list(range(24,32));RIDGES=[1e-4,1e-3,1e-2,.05,.1,.5,1.,5.];TOP_LAYERS=8
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
def fp(x,bank,L,refs):return torch.tensor([cos(x[L],bank[j][L])for j in refs],dtype=torch.float32)
def ridge(X,Y,lam):
 X=X.double();Y=Y.double();mx=X.mean(0);my=Y.mean(0);A=X-mx;B=Y-my;xx=A.T@A;xy=A.T@B;s=max(1e-12,float(xx.trace()/max(1,xx.shape[0])));W=torch.linalg.solve(xx+lam*s*torch.eye(xx.shape[0],dtype=torch.float64),xy)
 return W.float(),mx.float(),my.float()
def pred(x,f):
 W,mx,my=f;return (x.float()-mx)@W+my
def qsolve(target_fp,Qbank,L,refs,lam):
 # Solve Qwen-local vector entirely inside span of its TRAIN relation bank:
 # min ||C a-target_fp||² + λ||a||², where C is bank cosine Gram.
 B=torch.stack([unit(Qbank[j][L])for j in refs]).float();G=B@B.T;t=target_fp.float();A=G.T@G+lam*torch.eye(len(refs));a=torch.linalg.solve(A,G.T@t);v=a@B
 return unit(v),a,G
def crossfit_layer(MP,QP,L,lam):
 # Leave-one-out TRAIN: Mistral fingerprint -> Qwen fingerprint -> Qwen-local solve.
 cs=[];fpcs=[]
 for hold in TRAIN:
  refs=[j for j in TRAIN if j!=hold]
  XM=[];YQ=[]
  for i in refs:
   r=[j for j in refs if j!=i];XM.append(fp(MP[i],MP,L,r));YQ.append(fp(QP[i],QP,L,r))
  # all vectors must share identical reference dimensions; for held item use refs minus itself logic via direct matched bank
  # Use full refs Gram fingerprints for stable 23-D cross-model mapping trained from relation rows.
  GM=torch.stack([fp(MP[i],MP,L,refs)for i in refs]);GQ=torch.stack([fp(QP[i],QP,L,refs)for i in refs])
  Fm=ridge(GM,GQ,lam);mhold=fp(MP[hold],MP,L,refs);qhat=pred(mhold,Fm);v,_,_=qsolve(qhat,QP,L,refs,lam)
  cs.append(cos(v,QP[hold][L]));fpcs.append(cos(qhat,fp(QP[hold],QP,L,refs)))
 return sum(cs)/len(cs),sum(fpcs)/len(fpcs)
def fit_full(MP,QP,L,lam):
 GM=torch.stack([fp(MP[i],MP,L,TRAIN)for i in TRAIN]);GQ=torch.stack([fp(QP[i],QP,L,TRAIN)for i in TRAIN]);return ridge(GM,GQ,lam)
print("="*122);print("TEST 407 — FINGERPRINT-CONSTRAINED QWEN LOCAL SOLVE");print("MISTRAL COORDINATE ONLY → QWEN FINGERPRINT TARGET → QWEN NATIVE BANK SOLVE → AKBASCORE WRITE");print("="*122)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/10] MISTRAL — TEST405/406 RELATIONAL PACKETS")
mt,mm,ml=load(MID,(32,4096,32,8,128));mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight];M0=sha(mp);MH=[]
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 MH.append((forge(mt,mm,a),forge(mt,mm,b)));print(f" MISTRAL {i+1}/32",end="\r")
MFINAL=(forge(mt,mm,FACT),forge(mt,mm,CF));print()
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/10] QWEN — TEST405/406 RELATIONAL PACKETS")
qt,qm,ql=load(QID,(28,3584,28,4,128));qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight];Q0=sha(qp);QH=[]
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 QH.append((forge(qt,qm,a),forge(qt,qm,b)));print(f" QWEN {i+1}/32",end="\r")
QFINAL=(forge(qt,qm,FACT),forge(qt,qm,CF));print("\n TRAIN=24 | VAL=8 | FINAL FACT/CF NEVER USED FOR FIT/SELECTION")
MP=[packet(a,b,LMAP)for a,b in MH];QP=[packet(a,b,range(28))for a,b in QH];MF=packet(MFINAL[0],MFINAL[1],LMAP);QF=packet(QFINAL[0],QFINAL[1],range(28))
print("[3/10] TRAIN-ONLY FINGERPRINT→LOCAL-SOLVE CROSS-VALIDATION")
CV={}
for L in range(28):
 best=None
 for lam in RIDGES:
  vc,fc=crossfit_layer(MP,QP,L,lam);score=vc
  if best is None or score>best["score"]:best={"L":L,"lambda":lam,"score":score,"fp_cos":fc}
 CV[L]=best;print(f" L{L:02d}←M{LMAP[L]:02d} λ={best['lambda']:<6g} LOCAL={best['score']:+.4f} FP={best['fp_cos']:+.4f}")
print("[4/10] VALIDATION — FINAL-FREE LAYER SELECTION")
VALRES={}
for L in range(28):
 lam=CV[L]["lambda"];fit=fit_full(MP,QP,L,lam);cs=[];fs=[]
 for i in VAL:
  m=fp(MP[i],MP,L,TRAIN);qhat=pred(m,fit);qtrue=fp(QP[i],QP,L,TRAIN);v,_,_=qsolve(qhat,QP,L,TRAIN,lam);cs.append(cos(v,QP[i][L]));fs.append(cos(qhat,qtrue))
 VALRES[L]={"local_cos":sum(cs)/len(cs),"fp_cos":sum(fs)/len(fs),"lambda":lam}
ORDER=sorted(range(28),key=lambda L:VALRES[L]["local_cos"],reverse=True);SELECT=sorted(ORDER[:TOP_LAYERS])
print(" TOP:",", ".join(f"L{L:02d}←M{LMAP[L]:02d}:{VALRES[L]['local_cos']:+.3f}"for L in ORDER[:TOP_LAYERS]));print(" SELECTED:",SELECT)
print("[5/10] FINAL HELD-OUT MISTRAL FINGERPRINT → QWEN TARGET FINGERPRINT")
PRED={};FINAL={}
for L in SELECT:
 lam=VALRES[L]["lambda"];fit=fit_full(MP,QP,L,lam);mf=fp(MF,MP,L,TRAIN);qhat=pred(mf,fit);qtrue=fp(QF,QP,L,TRAIN);v,a,G=qsolve(qhat,QP,L,TRAIN,lam)
 PRED[L]=v;FINAL[L]={"fingerprint_cos":cos(qhat,qtrue),"local_cos":cos(v,QF[L]),"lambda":lam,"coef_norm":float(a.norm())}
 print(f" L{L:02d}←M{LMAP[L]:02d} FP={FINAL[L]['fingerprint_cos']:+.6f} LOCAL={FINAL[L]['local_cos']:+.6f} λ={lam:g}")
print(f" MEAN FINAL FP={sum(x['fingerprint_cos']for x in FINAL.values())/len(FINAL):+.6f} LOCAL={sum(x['local_cos']for x in FINAL.values())/len(FINAL):+.6f}")
print("[6/10] TRAIN-ONLY AKBASCORE WRITE MAGNITUDES")
MAG={}
for L in SELECT:
 vals=[float((QH[i][1][L]-QH[i][0][L]).norm())for i in TRAIN];MAG[L]=sum(vals)/len(vals);print(f" L{L:02d} ΔNORM={MAG[L]:.6f}")
BASE_TEXT=FACT;base_ids=torch.tensor([[pad(qt)]+ids(qt,BASE_TEXT+SEP)],device=DEV);nsep=len(ids(qt,SEP));WRITE_POS=base_ids.shape[1]-nsep-1
def run(dirs,sign=1.):
 handles=[]
 for L in SELECT:
  d=(dirs[L]*MAG[L]*sign).to(DEV)
  def hook(mod,inp,out,d=d):
   if isinstance(out,tuple):
    h=out[0].clone();h[:,WRITE_POS,:]+=d.to(h.dtype);return (h,)+out[1:]
   h=out.clone();h[:,WRITE_POS,:]+=d.to(h.dtype);return h
  handles.append(ql[L].register_forward_hook(hook))
 with torch.inference_mode():o=qm(input_ids=base_ids,use_cache=False,output_hidden_states=True,return_dict=True)
 for h in handles:h.remove()
 H=[o.hidden_states[L+1][0,WRITE_POS].float().cpu()for L in range(28)];del o;return H
print("[7/10] AKBASCORE CAUSAL WRITE — PREDICTED / REVERSE")
BASEH=forge(qt,qm,BASE_TEXT);WH=run(PRED,+1);RH=run(PRED,-1)
def selected_eval(H):
 vals=[]
 for L in SELECT:vals.append(cos(H[L]-BASEH[L],QFINAL[1][L]-QFINAL[0][L]))
 return sum(vals)/len(vals),vals
WC,WL=selected_eval(WH);RC,RL=selected_eval(RH)
print(f" PREDICTED mean selected cos={WC:+.6f}");print(f" REVERSE   mean selected cos={RC:+.6f}")
for i,L in enumerate(SELECT):print(f" L{L:02d} PRED={WL[i]:+.3f} REV={RL[i]:+.3f}")
print("[8/10] DOWNSTREAM X-RAY")
DOWN={}
for L in range(max(SELECT),28):
 native=QFINAL[1][L]-QFINAL[0][L];DOWN[L]={"pred":cos(WH[L]-BASEH[L],native),"rev":cos(RH[L]-BASEH[L],native)}
 if L in sorted(set([max(SELECT),18,19,21,23,25,27])):print(f" L{L:02d} PRED={DOWN[L]['pred']:+.6f} REV={DOWN[L]['rev']:+.6f}")
TERM=DOWN[27]["pred"]
print("[9/10] RANDOM LOCAL-BANK NULL")
NULL=[]
for s in range(64):
 gen=torch.Generator().manual_seed(SEED+700+s);dirs={}
 for L in SELECT:
  # null remains inside Qwen TRAIN relational span; only coefficients are randomized
  B=torch.stack([unit(QP[j][L])for j in TRAIN]);a=torch.randn(len(TRAIN),generator=gen);dirs[L]=unit(a@B)
 H=run(dirs,+1);NULL.append(cos(H[27]-BASEH[27],QFINAL[1][27]-QFINAL[0][27]))
NULL=torch.tensor(NULL);mu=float(NULL.mean());sd=float(NULL.std(unbiased=True));z=(TERM-mu)/max(sd,1e-12);pval=float((1+(NULL>=TERM).sum())/(len(NULL)+1))
print(f" L27 BANK-NULL mean={mu:+.6f} sd={sd:.6f} z={z:+.3f} p={pval:.6f}")
print("[10/10] VERDICT + INTEGRITY")
mean_local=sum(x["local_cos"]for x in FINAL.values())/len(FINAL);mean_fp=sum(x["fingerprint_cos"]for x in FINAL.values())/len(FINAL)
if mean_local>=.25 and TERM>=.20 and TERM>DOWN[27]["rev"]+.10 and pval<=.05:VERDICT="FINGERPRINT_LOCAL_WRITE_FOUND: Mistral relational coordinates were converted by Qwen's own bank into a significant held-out native residual write."
elif mean_local>=.15 and TERM>mu+.05:VERDICT="PARTIAL_FINGERPRINT_WRITE: Qwen local solve recovers some held-out direction, but downstream transport remains incomplete."
else:VERDICT="NO_STABLE_FINGERPRINT_WRITE: dimension-free fingerprint survives partially, but Qwen-bank reconstruction does not yet yield a reliable causal write."
print(f" FINAL FP={mean_fp:+.6f} LOCAL={mean_local:+.6f} | L27={TERM:+.6f} | p={pval:.6f}");print(" VERDICT:",VERDICT)
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Qwen not frozen")
REPORT={"test":407,"title":"Fingerprint-Constrained Qwen Local Solve","source_model":MID,"target_model":QID,"seed":SEED,"fact":FACT,"counterfactual":CF,
"train_pairs":24,"validation_pairs":8,"layer_map":LMAP,"ridge_grid":RIDGES,"cross_validation":CV,"validation":VALRES,"selected_layers":SELECT,"final":FINAL,
"write_magnitudes":MAG,"causal":{"selected_pred":WC,"selected_reverse":RC,"downstream":DOWN,"terminal":TERM},
"null":{"type":"Qwen train-bank coefficient randomization","n":64,"mean":mu,"sd":sd,"z":z,"p":pval},"verdict":VERDICT,
"integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1,"pass":Q0==Q1},
"notes":["TEST405/406 dataset, layer map and held-out FACT/CF lock retained.","No direct Mistral 4096-D to Qwen 3584-D vector regression is fitted.","Mistral supplies only relation-to-bank fingerprints.","Qwen reconstructs each write direction solely from its own TRAIN relational bank.","Ridge and layer selection use TRAIN→VAL only; FINAL FACT/CF never selects layers, ridge or dose.","AkbasCore performs layer-local residual writes using TRAIN-only native Qwen delta magnitudes.","Random null remains inside the same Qwen TRAIN relational subspace.","No weights are modified; behavioral generation remains OFF."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("="*122);print("TEST 407 — COMPLETE");print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS");print("SUMMARY:",OUT);print("="*122)
