# TEST 405 — CROSS-MODEL RELATIONAL COORDINATE X-RAY
# TEST404 LINEAGE — NO DIRECT 4096→3584 CONTENT TRANSLATION
# MISTRAL AND QWEN EACH SELF-EXTRACT RELATIONAL Δ PACKETS → COMPARE LOW-DIM INTERNAL GEOMETRY
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
ROOT=Path("/content/AKBASCORE_TEST405")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST405");ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/"TEST405_SUMMARY.json"
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
 # post-block residual states: layer L output = hidden_states[L+1], final content token before SEP when possible
 nsep=len(ids(tok,SEP));pos=max(0,len(seq)-nsep-1);H=[o.hidden_states[L+1][0,pos].float().cpu().contiguous()for L in range(len(model.model.layers))]
 del o;return H
def unit(x):return x.float()/x.float().norm().clamp_min(1e-12)
def cos(a,b):return float(F.cosine_similarity(a.float().reshape(-1),b.float().reshape(-1),dim=0,eps=1e-12))
def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-12))
def gram(vs):
 X=torch.stack([unit(x)for x in vs]);return X@X.T
def eigcoords(G,k=8):
 # model-independent coordinates from relational Gram geometry only
 G=(G+G.T)/2;e,U=torch.linalg.eigh(G.double());ix=torch.argsort(e,descending=True);e=e[ix].clamp_min(0);U=U[:,ix]
 Z=U[:,:k]*e[:k].sqrt()
 # sign lock removes eigenvector sign ambiguity
 for j in range(Z.shape[1]):
  i=int(torch.argmax(Z[:,j].abs()))
  if Z[i,j]<0:Z[:,j]*=-1
 return Z.float(),e.float()
def procrustes(A,B):
 A=A-A.mean(0,keepdim=True);B=B-B.mean(0,keepdim=True);U,S,Vh=torch.linalg.svd(A.T@B,full_matrices=False);R=U@Vh;P=A@R
 return P,B,R,float((P-B).norm()/B.norm().clamp_min(1e-12)),cos(P,B)
def packet(Ha,Hb,layers):
 return [unit(Hb[L]-Ha[L])for L in layers]
def coord_signature(packets):
 # each row = one relation pair; columns derive only from pairwise relation geometry
 G=gram([torch.cat(p)for p in packets]);Z,E=eigcoords(G,min(8,len(packets)));return G,Z,E
print("="*118);print("TEST 405 — CROSS-MODEL RELATIONAL COORDINATE X-RAY");print("NO DIRECT 4096→3584 CONTENT TRANSLATION | DUAL-ENDOGENOUS RELATIONAL GEOMETRY");print("="*118)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/7] MISTRAL — SELF-EXTRACT POST-BLOCK RELATIONAL PACKETS")
mt,mm,ml=load(MID,(32,4096,32,8,128));mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight];M0=sha(mp)
MH=[]
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 MH.append((forge(mt,mm,a),forge(mt,mm,b)));print(f" MISTRAL {i+1}/32",end="\r")
MFINAL=(forge(mt,mm,FACT),forge(mt,mm,CF));print()
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/7] QWEN — SELF-EXTRACT POST-BLOCK RELATIONAL PACKETS")
qt,qm,ql=load(QID,(28,3584,28,4,128));qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight];Q0=sha(qp)
QH=[]
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 QH.append((forge(qt,qm,a),forge(qt,qm,b)));print(f" QWEN {i+1}/32",end="\r")
QFINAL=(forge(qt,qm,FACT),forge(qt,qm,CF));print("\n TRAIN=24 | HELD-OUT VAL=8 | FINAL FACT/CF NEVER USED FOR ALIGNMENT")
print("[3/7] DUAL-ENDOGENOUS PACKETS — MATCHED DEPTH, NO CROSS-MODEL VECTOR MAP")
MP=[];QP=[]
for i in range(32):
 MP.append(packet(MH[i][0],MH[i][1],LMAP))
 QP.append(packet(QH[i][0],QH[i][1],range(28)))
MF=packet(MFINAL[0],MFINAL[1],LMAP);QF=packet(QFINAL[0],QFINAL[1],range(28))
print(" MISTRAL PACKET: 28 × 4096 | QWEN PACKET: 28 × 3584")
print("[4/7] RELATIONAL GRAM GEOMETRY — TRAIN/VAL")
# Compare model-independent pair×pair Gram matrices. No dimensional bridge exists here.
def subset_geom(ix):
 GM=gram([torch.cat(MP[i])for i in ix]);GQ=gram([torch.cat(QP[i])for i in ix]);mask=~torch.eye(len(ix),dtype=torch.bool)
 return {"GRAM_COS":cos(GM[mask],GQ[mask]),"GRAM_REL":rel(GM[mask],GQ[mask]),"M":GM,"Q":GQ}
TR=subset_geom(range(24));VA=subset_geom(range(24,32))
print(f" TRAIN GRAM COS={TR['GRAM_COS']:+.6f} REL={TR['GRAM_REL']:.6f}")
print(f" VAL   GRAM COS={VA['GRAM_COS']:+.6f} REL={VA['GRAM_REL']:.6f}")
print("[5/7] LOW-DIM COORDINATE ALIGNMENT — TRAIN FIT → VAL/FULL DIAGNOSTIC")
# One joint 32-pair geometry is diagnostic only; selection remains train/val separated.
GM32,ZM,E_M=coord_signature(MP);GQ32,ZQ,E_Q=coord_signature(QP)
PM,BM,R,full_rel,full_cos=procrustes(ZM[:24],ZQ[:24])
# apply train-derived centering + rotation to all rows
m0=ZM[:24].mean(0,keepdim=True);q0=ZQ[:24].mean(0,keepdim=True);ZM_A=(ZM-m0)@R;ZQ_C=ZQ-q0
val_cos=cos(ZM_A[24:32],ZQ_C[24:32]);val_rel=rel(ZM_A[24:32],ZQ_C[24:32])
print(f" TRAIN COORD COS={full_cos:+.6f} REL={full_rel:.6f}")
print(f" VAL   COORD COS={val_cos:+.6f} REL={val_rel:.6f}")
print(" TOP8 ENERGY MISTRAL="+",".join(f"{x:.3f}"for x in E_M[:8])+" | QWEN="+",".join(f"{x:.3f}"for x in E_Q[:8]))
print("[6/7] HELD-OUT FACT↔CF — RELATION-TO-BANK FINGERPRINT")
# FINAL never enters PCA/Procrustes. Compare its cosine fingerprint against the same 24 train relations.
def fingerprint(final,bank):
 f=torch.cat(final);return torch.tensor([cos(f,torch.cat(bank[i]))for i in range(24)])
FM=fingerprint(MF,MP);FQ=fingerprint(QF,QP);FC=cos(FM,FQ);FR=rel(FM,FQ)
print(f" FINAL FINGERPRINT COS={FC:+.6f} REL={FR:.6f}")
print(" MISTRAL:",",".join(f"{x:+.3f}"for x in FM.tolist()))
print(" QWEN   :",",".join(f"{x:+.3f}"for x in FQ.tolist()))
print("\n LAYER-LOCAL FINAL FINGERPRINT AGREEMENT")
LAYER=[]
for L in range(28):
 a=torch.tensor([cos(MF[L],MP[i][L])for i in range(24)]);b=torch.tensor([cos(QF[L],QP[i][L])for i in range(24)])
 d={"L":L,"M":LMAP[L],"COS":cos(a,b),"REL":rel(a,b)};LAYER.append(d)
 if L in [0,3,6,10,14,18,19,21,23,25,27]:print(f" L{L:02d}←M{LMAP[L]:02d} COS={d['COS']:+.6f} REL={d['REL']:.6f}")
print("[7/7] NULL + VERDICT + INTEGRITY")
# permutation null: destroy matched relation identity while preserving each model's packet bank
GEN=torch.Generator().manual_seed(SEED+405);NULL=[]
for _ in range(256):
 p=torch.randperm(24,generator=GEN);NULL.append(cos(FM,FQ[p]))
NULL=torch.tensor(NULL);mu=float(NULL.mean());sd=float(NULL.std(unbiased=True));z=(FC-mu)/max(sd,1e-12);pval=float((1+(NULL>=FC).sum())/(len(NULL)+1))
print(f" FINAL NULL mean={mu:+.6f} sd={sd:.6f} z={z:+.3f} p={pval:.6f}")
mean_layer=sum(x["COS"]for x in LAYER)/28
if FC>=.60 and pval<=.05 and mean_layer>=.30:VERDICT="RELATIONAL_COORDINATE_FOUND: held-out FACT↔CF occupies a significantly shared cross-model relation-to-bank geometry."
elif FC>=.35 and pval<=.05:VERDICT="PARTIAL_RELATIONAL_COORDINATE: significant shared held-out geometry exists, but layer-local consistency is incomplete."
else:VERDICT="NO_STABLE_RELATIONAL_COORDINATE: current endogenous packet geometry does not yet support a shared held-out cross-model coordinate."
print(f" MEAN LAYER COS={mean_layer:+.6f}");print(" VERDICT:",VERDICT)
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Qwen not frozen")
REPORT={"test":405,"title":"Cross-Model Relational Coordinate X-Ray","source_model":MID,"target_model":QID,"seed":SEED,"fact":FACT,"counterfactual":CF,
"train_pairs":24,"validation_pairs":8,"layer_map":LMAP,
"train":{"gram_cos":TR["GRAM_COS"],"gram_rel":TR["GRAM_REL"],"coord_cos":full_cos,"coord_rel":full_rel},
"validation":{"gram_cos":VA["GRAM_COS"],"gram_rel":VA["GRAM_REL"],"coord_cos":val_cos,"coord_rel":val_rel},
"final":{"fingerprint_cos":FC,"fingerprint_rel":FR,"mistral":FM.tolist(),"qwen":FQ.tolist(),"mean_layer_cos":mean_layer,"layers":LAYER,"null_mean":mu,"null_sd":sd,"z":z,"p":pval},
"verdict":VERDICT,"integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1,"pass":Q0==Q1},
"notes":["FACT/CF never enters coordinate alignment or selection.","No 4096→3584 linear bridge is fitted.","Each model self-extracts its own post-block relational packet.","Cross-model comparison occurs only through dimension-free relation-to-bank geometry.","TEST405 is discovery/validation only; AkbasCore write/inverse-write remains OFF and is reserved for the next causal stage."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("="*118);print("TEST 405 — COMPLETE");print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS");print("SUMMARY:",OUT);print("="*118)
