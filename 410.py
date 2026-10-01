# TEST 410 — MID/LATE CAUSAL TRANSPORT MICRO-MAP
# TEST409 EXACT LINEAGE — SAME GLOBAL FINGERPRINT / SHARED COEFFICIENTS / TRAIN-ONLY DOSE
# SURGICAL L14–L23 WINDOWS + COMBINATIONS + 00–23 REPLICATION | L27 ENDPOINT
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
ROOT=Path("/content/AKBASCORE_TEST410")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST410");ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/"TEST410_SUMMARY.json"
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
TRAIN=list(range(24));VAL=list(range(24,32));RIDGES=[1e-4,1e-3,1e-2,.05,.1,.5,1.,5.];LAYERS=list(range(28));NNULL=64
WINDOWS={
"ALL_00_27":list(range(28)),"PREFIX_00_23":list(range(24)),"MID_14_23":list(range(14,24)),
"B14_17":list(range(14,18)),"B18_19":[18,19],"B20_21":[20,21],"B22_23":[22,23],
"C14_19":list(range(14,20)),"C18_21":list(range(18,22)),"C20_23":list(range(20,24)),
"C18_23":list(range(18,24)),"C14_21":list(range(14,22)),
"DROP14_17":list(range(18,24)),"DROP18_19":list(range(14,18))+list(range(20,24)),
"DROP20_21":list(range(14,20))+[22,23],"DROP22_23":list(range(14,22)),
"SINGLE_18":[18],"SINGLE_19":[19],"SINGLE_20":[20],"SINGLE_21":[21],"SINGLE_22":[22],"SINGLE_23":[23]}
PRIMARY=["ALL_00_27","PREFIX_00_23","MID_14_23","B14_17","B18_19","B20_21","B22_23","C14_19","C18_21","C20_23","C18_23","C14_21","DROP18_19","DROP20_21","DROP22_23"]
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
  refs=[j for j in TRAIN if j!=hold];fit=fit_global(MP,QP,refs,lam);qh=pred(gfp(MP[hold],MP,refs),fit);qtg=gfp(QP[hold],QP,refs);dirs,_=global_qsolve(qh,QP,refs,lam)
  fc.append(cos(qh,qtg));lc.append(sum(cos(dirs[L],QP[hold][L])for L in LAYERS)/28)
 return sum(fc)/len(fc),sum(lc)/len(lc)
print("="*128);print("TEST 410 — MID/LATE CAUSAL TRANSPORT MICRO-MAP");print("TEST409 EXACT GLOBAL SOLVE | L14–L23 SURGICAL WINDOWS + DROP-OUTS + 00–23 REPLICATION | L27 ENDPOINT");print("="*128)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/10] MISTRAL — TEST409 EXACT PACKETS")
mt,mm,ml=load(MID,(32,4096,32,8,128));mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight];M0=sha(mp);MH=[]
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 MH.append((forge(mt,mm,a),forge(mt,mm,b)));print(f" MISTRAL {i+1}/32",end="\r")
MFINAL=(forge(mt,mm,FACT),forge(mt,mm,CF));print()
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/10] QWEN — TEST409 EXACT PACKETS")
qt,qm,ql=load(QID,(28,3584,28,4,128));qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight];Q0=sha(qp);QH=[]
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 QH.append((forge(qt,qm,a),forge(qt,qm,b)));print(f" QWEN {i+1}/32",end="\r")
QFINAL=(forge(qt,qm,FACT),forge(qt,qm,CF));print("\n TRAIN=24 | VAL=8 | FINAL FACT/CF NEVER USED FOR FIT/SELECTION")
MP=[packet(a,b,LMAP)for a,b in MH];QP=[packet(a,b,LAYERS)for a,b in QH];MF=packet(MFINAL[0],MFINAL[1],LMAP);QF=packet(QFINAL[0],QFINAL[1],LAYERS)
print("[3/10] TEST409 TRAIN-ONLY GLOBAL RIDGE")
CV=[]
for lam in RIDGES:
 fc,lc=loo_score(MP,QP,lam);CV.append({"lambda":lam,"fp_cos":fc,"local_cos":lc});print(f" λ={lam:<6g} FP={fc:+.6f} LOCAL={lc:+.6f}")
BEST=max(CV,key=lambda x:(x["local_cos"],x["fp_cos"]));LAM=BEST["lambda"];print(" SELECTED λ=",LAM)
print("[4/10] GLOBAL FINAL SOLVE — FROZEN BEFORE MICRO-MAP")
FIT=fit_global(MP,QP,TRAIN,LAM);QHAT=pred(gfp(MF,MP,TRAIN),FIT);QTRUE=gfp(QF,QP,TRAIN);DIRS,COEF=global_qsolve(QHAT,QP,TRAIN,LAM)
FFP=cos(QHAT,QTRUE);FLOCAL=sum(cos(DIRS[L],QF[L])for L in LAYERS)/28
print(f" FINAL FP={FFP:+.6f} LOCAL={FLOCAL:+.6f} COEF_NORM={float(COEF.norm()):.6f}")
print("[5/10] TRAIN-ONLY DOSE")
MAG={L:sum(float((QH[i][1][L]-QH[i][0][L]).norm())for i in TRAIN)/len(TRAIN)for L in LAYERS}
BASE_TEXT=FACT;base_ids=torch.tensor([[pad(qt)]+ids(qt,BASE_TEXT+SEP)],device=DEV);nsep=len(ids(qt,SEP));WRITE_POS=base_ids.shape[1]-nsep-1
def run(dirs,window,sign=1.):
 handles=[]
 for L in window:
  d=(dirs[L]*MAG[L]*sign).to(DEV)
  def hook(mod,inp,out,d=d):
   if isinstance(out,tuple):
    h=out[0].clone();h[:,WRITE_POS,:]+=d.to(h.dtype);return (h,)+out[1:]
   h=out.clone();h[:,WRITE_POS,:]+=d.to(h.dtype);return h
  handles.append(ql[L].register_forward_hook(hook))
 with torch.inference_mode():o=qm(input_ids=base_ids,use_cache=False,output_hidden_states=True,return_dict=True)
 for h in handles:h.remove()
 H=[o.hidden_states[L+1][0,WRITE_POS].float().cpu()for L in LAYERS];del o;return H
BASEH=forge(qt,qm,BASE_TEXT);NATIVE=[QFINAL[1][L]-QFINAL[0][L]for L in LAYERS]
print("[6/10] SURGICAL WINDOW MAP — PREDICTED / REVERSE")
RES={}
for name,window in WINDOWS.items():
 H=run(DIRS,window,+1);R=run(DIRS,window,-1);pc=cos(H[27]-BASEH[27],NATIVE[27]);rc=cos(R[27]-BASEH[27],NATIVE[27])
 RES[name]={"layers":window,"pred":pc,"reverse":rc,"signed_gap":pc-rc,"norm":float((H[27]-BASEH[27]).norm())}
 print(f" {name:13s} N={len(window):02d} PRED={pc:+.6f} REV={rc:+.6f} GAP={pc-rc:+.6f}")
print("[7/10] BLOCK ADDITION / DROP-OUT EFFECTS")
MID=RES["MID_14_23"]["pred"]
ADD={
"B14_17":RES["B14_17"]["pred"],"B18_19":RES["B18_19"]["pred"],"B20_21":RES["B20_21"]["pred"],"B22_23":RES["B22_23"]["pred"],
"C14_19":RES["C14_19"]["pred"],"C18_21":RES["C18_21"]["pred"],"C20_23":RES["C20_23"]["pred"],"C18_23":RES["C18_23"]["pred"],"C14_21":RES["C14_21"]["pred"]}
DROP={
"14_17":MID-RES["DROP14_17"]["pred"],"18_19":MID-RES["DROP18_19"]["pred"],"20_21":MID-RES["DROP20_21"]["pred"],"22_23":MID-RES["DROP22_23"]["pred"]}
for k,v in ADD.items():print(f" ADD {k:7s} L27={v:+.6f}")
for k,v in DROP.items():print(f" DROP {k:5s} LOSS={v:+.6f}")
print("[8/10] 00–23 / ALL REPLICATION")
print(f" PREFIX_00_23={RES['PREFIX_00_23']['pred']:+.6f} | ALL_00_27={RES['ALL_00_27']['pred']:+.6f} | MID_14_23={MID:+.6f}")
print("[9/10] SHARED-BANK NULL — SURGICAL WINDOWS")
NULL={}
for name in PRIMARY:
 window=WINDOWS[name];vals=[]
 for s in range(NNULL):
  gen=torch.Generator().manual_seed(SEED+1000+s);a=torch.randn(len(TRAIN),generator=gen);rd={}
  for L in LAYERS:rd[L]=unit(sum((a[k]*QP[j][L]for k,j in enumerate(TRAIN)),torch.zeros_like(QP[TRAIN[0]][L])))
  H=run(rd,window,+1);vals.append(cos(H[27]-BASEH[27],NATIVE[27]))
 v=torch.tensor(vals);mu=float(v.mean());sd=float(v.std(unbiased=True));obs=RES[name]["pred"];z=(obs-mu)/max(sd,1e-12);p=float((1+(v>=obs).sum())/(len(v)+1))
 NULL[name]={"mean":mu,"sd":sd,"z":z,"p":p};print(f" {name:13s} OBS={obs:+.6f} NULL={mu:+.6f}±{sd:.6f} z={z:+.3f} p={p:.6f}")
print("[10/10] MECHANISTIC VERDICT + INTEGRITY")
sig=[n for n in PRIMARY if NULL[n]["p"]<=.05 and RES[n]["pred"]>RES[n]["reverse"]]
best=max(PRIMARY,key=lambda n:RES[n]["pred"])
if NULL["PREFIX_00_23"]["p"]<=.05 and RES["PREFIX_00_23"]["pred"]>=RES["ALL_00_27"]["pred"]-.03:
 VERDICT="PRE_L24_TRANSPORT_CONFIRMED: the terminal alignment is already established by L00–L23; L24–L27 writes are not required."
elif NULL["MID_14_23"]["p"]<=.05:
 VERDICT="MID_LATE_CHANNEL_CONFIRMED: L14–L23 alone carries a significant direction-sensitive terminal effect."
elif sig:
 VERDICT="SUBWINDOW_SIGNAL: at least one predefined mid/late causal subwindow survives the shared-bank null."
else:
 VERDICT="NO_ISOLATED_SIGNIFICANT_SUBWINDOW: the full distributed write remains stronger than any isolated L14–L23 surgical arm."
print(" SIGNIFICANT:",sig if sig else"NONE");print(" BEST RAW:",best,f"{RES[best]['pred']:+.6f}");print(" VERDICT:",VERDICT)
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Qwen not frozen")
REPORT={"test":410,"title":"Mid/Late Causal Transport Micro-Map","source_model":"mistralai/Mistral-7B-Instruct-v0.3","target_model":QID,"seed":SEED,
"fact":FACT,"counterfactual":CF,"train_pairs":24,"validation_pairs":8,"layer_map":LMAP,"ridge_grid":RIDGES,"cross_validation":CV,"selected_lambda":LAM,
"global_final":{"fingerprint_cos":FFP,"local_mean":FLOCAL,"coef_norm":float(COEF.norm())},"windows":RES,"block_raw":ADD,"drop_loss":DROP,"null":NULL,
"significant_windows":sig,"verdict":VERDICT,"integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1,"pass":Q0==Q1},
"notes":["TEST409 global fingerprint mapping, shared Qwen-bank solve and TRAIN-only dose are unchanged.","Only causal write windows are changed.","L14–L23 is split into predefined 14–17,18–19,20–21,22–23 blocks plus contiguous combinations and drop-outs.","PREFIX_00_23 tests whether the TEST409 signal is established before direct writes to L24–L27.","All primary nulls preserve one shared randomized Qwen TRAIN-bank coefficient vector across written layers.","FINAL FACT/CF is used only as the held-out evaluation endpoint; no FINAL result changes fit, ridge, dose or directions.","No weights are modified; behavioral generation remains OFF."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("="*128);print("TEST 410 — COMPLETE");print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS");print("SUMMARY:",OUT);print("="*128)
