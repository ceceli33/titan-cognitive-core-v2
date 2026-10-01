# TEST 406 — AKBASCORE LOCAL INVERSE-WRITE
# TEST405 EXACT LINEAGE — TRAIN-ONLY LAYER/BASIS SELECTION → HELD-OUT RELATIONAL COORDINATE → QWEN LOCAL RESIDUAL WRITE
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
ROOT=Path("/content/AKBASCORE_TEST406")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST406");ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/"TEST406_SUMMARY.json"
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
 assert s.startswith(a)and a!=b;ALT.append(b+s[len(a):])
LMAP=[min(31,max(0,round((L+.5)*32/28-.5)))for L in range(28)]
TRAIN=range(24);VAL=range(24,32);RANK=8;RIDGE=.05;TOP_LAYERS=8;WRITE_SCALE=1.0
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
def ridge(X,Y):
 X=X.double();Y=Y.double();mx=X.mean(0);my=Y.mean(0);A=X-mx;B=Y-my;xx=A.T@A;xy=A.T@B;lam=max(1e-10,float(xx.trace()/max(1,xx.shape[0]))*RIDGE);W=torch.linalg.solve(xx+lam*torch.eye(xx.shape[0],dtype=torch.float64),xy)
 return W.float(),mx.float(),my.float()
def pred(x,fit):
 W,mx,my=fit;return (x.float()-mx)@W+my
def packet(A,B,layers):return [unit(B[L]-A[L])for L in layers]
def fingerprint(final,bank,ix=TRAIN):
 return torch.tensor([cos(torch.cat(final),torch.cat(bank[i]))for i in ix],dtype=torch.float32)
def layerfp(final,bank,L,ix=TRAIN):
 return torch.tensor([cos(final[L],bank[i][L])for i in ix],dtype=torch.float32)
def basis(ds,r=RANK):
 X=torch.stack(ds).float();mu=X.mean(0);A=X-mu;U,S,Vh=torch.linalg.svd(A,full_matrices=False);r=min(r,Vh.shape[0]);return mu,Vh[:r].T.contiguous(),S[:r]
def coords(x,B):mu,V,_=B;return (x.float()-mu)@V
def reconstruct(c,B):mu,V,_=B;return mu+c.float()@V.T
def train_layer_score(MP,QP,L):
 # TRAIN only: relation-to-bank fingerprints for each training relation, leave-self dimension masked
 M=[];Q=[]
 for i in TRAIN:
  a=torch.tensor([cos(MP[i][L],MP[j][L])for j in TRAIN if j!=i]);b=torch.tensor([cos(QP[i][L],QP[j][L])for j in TRAIN if j!=i]);M.append(a);Q.append(b)
 return cos(torch.stack(M),torch.stack(Q))
def fit_coord_map(MP,QP,L):
 # TRAIN only, each model keeps native dimensionality; map only RANK-dimensional self-basis coordinates
 BM=basis([MP[i][L]for i in TRAIN]);BQ=basis([QP[i][L]for i in TRAIN]);XM=torch.stack([coords(MP[i][L],BM)for i in TRAIN]);YQ=torch.stack([coords(QP[i][L],BQ)for i in TRAIN]);F=ridge(XM,YQ);return BM,BQ,F
def predict_qdir(mdir,fit):
 BM,BQ,F=fit;c=coords(mdir,BM);cq=pred(c,F);return unit(reconstruct(cq,BQ))
def validate_layer(MP,QP,L,fit):
 cs=[]
 for i in VAL:cs.append(cos(predict_qdir(MP[i][L],fit),QP[i][L]))
 return sum(cs)/len(cs)
print("="*120);print("TEST 406 — AKBASCORE LOCAL INVERSE-WRITE");print("TEST405 LINEAGE | TRAIN-ONLY LAYER/BASIS SELECTION | HELD-OUT FACT↔CF | QWEN RESIDUAL WRITE");print("="*120)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/9] MISTRAL — TEST405 RELATIONAL PACKETS")
mt,mm,ml=load(MID,(32,4096,32,8,128));mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight];M0=sha(mp);MH=[]
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 MH.append((forge(mt,mm,a),forge(mt,mm,b)));print(f" MISTRAL {i+1}/32",end="\r")
MFINAL=(forge(mt,mm,FACT),forge(mt,mm,CF));print()
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/9] QWEN — TEST405 RELATIONAL PACKETS")
qt,qm,ql=load(QID,(28,3584,28,4,128));qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight];Q0=sha(qp);QH=[]
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 QH.append((forge(qt,qm,a),forge(qt,qm,b)));print(f" QWEN {i+1}/32",end="\r")
QFINAL=(forge(qt,qm,FACT),forge(qt,qm,CF));print("\n TRAIN=24 | VAL=8 | FINAL FACT/CF NEVER USED FOR FIT/SELECTION")
MP=[packet(a,b,LMAP)for a,b in MH];QP=[packet(a,b,range(28))for a,b in QH];MF=packet(MFINAL[0],MFINAL[1],LMAP);QF=packet(QFINAL[0],QFINAL[1],range(28))
print("[3/9] TRAIN-ONLY LAYER RELIABILITY")
S=[train_layer_score(MP,QP,L)for L in range(28)];ORDER=sorted(range(28),key=lambda L:S[L],reverse=True);SELECT=sorted(ORDER[:TOP_LAYERS])
print(" TOP:",", ".join(f"L{L:02d}←M{LMAP[L]:02d}:{S[L]:+.3f}"for L in ORDER[:TOP_LAYERS]))
print(" SELECTED:",SELECT)
print("[4/9] TRAIN-ONLY LOCAL BASES + LOW-DIM COORDINATE MAP")
FITS={};VALCOS={}
for L in SELECT:
 FITS[L]=fit_coord_map(MP,QP,L);VALCOS[L]=validate_layer(MP,QP,L,FITS[L]);print(f" L{L:02d}←M{LMAP[L]:02d} TRAIN_SCORE={S[L]:+.6f} VAL_PRED_COS={VALCOS[L]:+.6f}")
print("[5/9] HELD-OUT FACT↔CF — PREDICT QWEN LOCAL DIRECTIONS")
PRED={L:predict_qdir(MF[L],FITS[L])for L in SELECT};FINALCOS={L:cos(PRED[L],QF[L])for L in SELECT}
for L in SELECT:print(f" L{L:02d}←M{LMAP[L]:02d} FINAL_PRED_COS={FINALCOS[L]:+.6f}")
print(f" MEAN FINAL PRED COS={sum(FINALCOS.values())/len(FINALCOS):+.6f}")
print("[6/9] AKBASCORE WRITE CALIBRATION — TRAIN-ONLY NATIVE QWEN Δ NORMS")
# Magnitude is learned only from TRAIN Qwen pair deltas; held-out QF norm is never used to set dose.
MAG={}
for L in SELECT:
 vals=[float((QH[i][1][L]-QH[i][0][L]).norm())for i in TRAIN];MAG[L]=sum(vals)/len(vals)
 print(f" L{L:02d} TRAIN_MEAN_ΔNORM={MAG[L]:.6f}")
BASE_TEXT=FACT
base_ids=torch.tensor([[pad(qt)]+ids(qt,BASE_TEXT+SEP)],device=DEV)
nsep=len(ids(qt,SEP));WRITE_POS=base_ids.shape[1]-nsep-1
def run_write(sign=1.0,random_control=False):
 handles=[];gen=torch.Generator().manual_seed(SEED+406)
 for L in SELECT:
  if random_control:
   d=torch.randn(3584,generator=gen);d=unit(d)
  else:d=PRED[L]
  d=(d*MAG[L]*WRITE_SCALE*sign).to(DEV)
  def hook(mod,inp,out,d=d):
   if isinstance(out,tuple):
    h=out[0].clone();h[:,WRITE_POS,:]=h[:,WRITE_POS,:]+d.to(h.dtype);return (h,)+out[1:]
   h=out.clone();h[:,WRITE_POS,:]=h[:,WRITE_POS,:]+d.to(h.dtype);return h
  handles.append(ql[L].register_forward_hook(hook))
 with torch.inference_mode():o=qm(input_ids=base_ids,use_cache=False,output_hidden_states=True,return_dict=True)
 for h in handles:h.remove()
 H=[o.hidden_states[L+1][0,WRITE_POS].float().cpu().contiguous()for L in range(28)];del o;return H
print("[7/9] CAUSAL WRITE — PREDICTED / REVERSE / RANDOM")
BASEH=forge(qt,qm,BASE_TEXT);WH=run_write(+1,False);RH=run_write(-1,False);XH=run_write(+1,True)
def evalrun(H):
 rows=[];cs=[]
 for L in SELECT:
  d=H[L]-BASEH[L];c=cos(d,QFINAL[1][L]-QFINAL[0][L]);rows.append({"L":L,"cos_to_native_CF_minus_FACT":c,"write_norm":float(d.norm()),"native_delta_norm":float((QFINAL[1][L]-QFINAL[0][L]).norm())});cs.append(c)
 return sum(cs)/len(cs),rows
WC,WR=evalrun(WH);RC,RR=evalrun(RH);XC,XR=evalrun(XH)
print(f" PREDICTED WRITE mean cos(native FACT→CF)={WC:+.6f}")
print(f" REVERSE   WRITE mean cos(native FACT→CF)={RC:+.6f}")
print(f" RANDOM    WRITE mean cos(native FACT→CF)={XC:+.6f}")
for a,b,c in zip(WR,RR,XR):print(f" L{a['L']:02d} PRED={a['cos_to_native_CF_minus_FACT']:+.3f} REV={b['cos_to_native_CF_minus_FACT']:+.3f} RAND={c['cos_to_native_CF_minus_FACT']:+.3f}")
print("[8/9] DOWNSTREAM TERMINAL X-RAY")
# Compare terminal residual displacement against native held-out FACT→CF terminal direction.
NTERM=QFINAL[1][27]-QFINAL[0][27]
TERM={"PREDICTED":cos(WH[27]-BASEH[27],NTERM),"REVERSE":cos(RH[27]-BASEH[27],NTERM),"RANDOM":cos(XH[27]-BASEH[27],NTERM)}
for k,v in TERM.items():print(f" {k:10s} L27 COS={v:+.6f}")
print("[9/9] NULL + VERDICT + INTEGRITY")
# Direction null at selected layers, same TRAIN-derived magnitudes.
NULL=[]
for s in range(64):
 gen=torch.Generator().manual_seed(SEED+500+s);handles=[]
 for L in SELECT:
  d=unit(torch.randn(3584,generator=gen))*MAG[L]
  dd=d.to(DEV)
  def hook(mod,inp,out,d=dd):
   if isinstance(out,tuple):
    h=out[0].clone();h[:,WRITE_POS,:]=h[:,WRITE_POS,:]+d.to(h.dtype);return (h,)+out[1:]
   h=out.clone();h[:,WRITE_POS,:]=h[:,WRITE_POS,:]+d.to(h.dtype);return h
  handles.append(ql[L].register_forward_hook(hook))
 with torch.inference_mode():o=qm(input_ids=base_ids,use_cache=False,output_hidden_states=True,return_dict=True)
 for h in handles:h.remove()
 H=[o.hidden_states[L+1][0,WRITE_POS].float().cpu()for L in range(28)];NULL.append(cos(H[27]-BASEH[27],NTERM));del o
NULL=torch.tensor(NULL);mu=float(NULL.mean());sd=float(NULL.std(unbiased=True));z=(TERM["PREDICTED"]-mu)/max(sd,1e-12);pval=float((1+(NULL>=TERM["PREDICTED"]).sum())/(len(NULL)+1))
print(f" L27 RANDOM NULL mean={mu:+.6f} sd={sd:.6f} z={z:+.3f} p={pval:.6f}")
if WC>=.25 and TERM["PREDICTED"]>=.20 and TERM["PREDICTED"]>TERM["RANDOM"]+.10 and pval<=.05:VERDICT="LOCAL_INVERSE_WRITE_FOUND: Mistral-derived relational coordinates produce a significant Qwen-native FACT→CF residual displacement."
elif WC>=.15 and TERM["PREDICTED"]>TERM["RANDOM"]+.05:VERDICT="PARTIAL_LOCAL_WRITE: predicted directions enter Qwen geometry, but downstream transport is incomplete."
else:VERDICT="NO_STABLE_LOCAL_WRITE: relational prediction does not yet create a reliable downstream Qwen-native displacement."
print(" VERDICT:",VERDICT)
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Qwen not frozen")
REPORT={"test":406,"title":"AkbasCore Local Inverse-Write","source_model":MID,"target_model":QID,"seed":SEED,"fact":FACT,"counterfactual":CF,
"train_pairs":24,"validation_pairs":8,"rank":RANK,"selected_layers":SELECT,"layer_map":LMAP,"train_scores":S,"validation_prediction_cos":VALCOS,
"final_prediction_cos":FINALCOS,"train_only_write_magnitudes":MAG,
"causal_write":{"predicted_mean":WC,"reverse_mean":RC,"random_mean":XC,"predicted_layers":WR,"reverse_layers":RR,"random_layers":XR,"terminal":TERM},
"null":{"n":64,"mean":mu,"sd":sd,"z":z,"p":pval},"verdict":VERDICT,
"integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1,"pass":Q0==Q1},
"notes":["TEST405 dataset, matched depth map and held-out FACT/CF lock retained.","Layer selection uses TRAIN only; FINAL FACT/CF is never used for layer selection, basis fitting, coordinate fitting or dose calibration.","No direct 4096→3584 map is fitted. Only rank-8 native coordinates are related across models.","AkbasCore writes predicted Qwen-local residual directions at selected layers.","Write magnitude is calibrated only from TRAIN Qwen relational delta norms.","Reverse and random-direction causal controls included.","No weights are modified; this test measures local write and downstream transport, not behavioral generation."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("="*120);print("TEST 406 — COMPLETE");print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS");print("SUMMARY:",OUT);print("="*120)
