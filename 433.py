# TEST 433 — L26 ATTENTION→MLP NATIVE IDENTITY / FIRST-TOKEN READOUT X-RAY
# TEST432 PROVEN LINEAGE | SOURCE-ABSENT WHO | FINAL FACT/CF = MEASUREMENT REFERENCE ONLY
# NO FINAL-DERIVED WRITE | MEASUREMENT ONLY | NO TARGET FORCING
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
ROOT=Path("/content/AKBASCORE_TEST433")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST433");ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/"TEST433_SUMMARY.json"
FACT="Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge."
CF="Leyla Demir planted the Turkish flag at the base of the Golden Gate Bridge."
WHO="Who planted the Turkish flag at the base of the Golden Gate Bridge?\nAnswer:"
NEUTRAL=[
"Jonas Weber carried the wooden crate across the quiet market square.","Priya Nair wrote a long letter to her cousin in the morning.","The small boat drifted slowly toward the rocky shore.","Omar Haddad fixed the broken clock in the village school.",
"A tired teacher closed the green door of the library.","Sofia Rossi baked fresh bread for the harvest festival.","The children watched the kites rising above the hill.","Liam O'Connor sold his old bicycle to a neighbor.",
"Heavy rain flooded the narrow street near the market.","Nadia Petrova translated the ancient manuscript into French.","The farmer counted the sheep before sunset.","Hiro Sato opened a tiny bakery beside the river.",
"An old dog slept under the kitchen table all afternoon.","Carlos Mendes washed the windows of his grandmother's house.","The museum guard locked the heavy gate at midnight.","Fatima Zahra grew tomatoes in the backyard garden.",
"The pilot announced a short delay because of fog.","Anna Kowalski found a lost wallet on the bus.","Snow covered the mountain village during the night.","Ravi Kumar taught his brother how to play chess.",
"The chef sharpened every knife before dinner service.","Lucas Martin cleaned the roof of the barn after the storm.","A young violinist practiced scales in the empty hall.","Mei Lin delivered the package to the wrong address.",
"Marek Novak placed the glass bottle beneath the wooden bench.","Sara Ibrahim carried a red notebook into the quiet classroom.","Noah Schmidt repaired the small radio beside the kitchen window.","Yuki Mori left a paper envelope near the station entrance.",
"Amira Hassan moved the ceramic bowl onto the upper shelf.","Peter Novak opened the metal box behind the old theater.","Lucia Costa placed the yellow scarf inside the travel bag.","Daniel Kim carried a black umbrella through the central courtyard."]
SWAP=[
("Jonas Weber","Elena Fischer"),("Priya Nair","Daniel Brooks"),("The small boat","The large boat"),("Omar Haddad","Lucas Martin"),("A tired teacher","A young teacher"),("Sofia Rossi","Nadia Petrova"),("The children","The visitors"),("Liam O'Connor","Peter Novak"),
("Heavy rain","Strong winds"),("Nadia Petrova","Anna Kowalski"),("The farmer","The gardener"),("Hiro Sato","Ravi Kumar"),("An old dog","A young dog"),("Carlos Mendes","Daniel Kim"),("The museum guard","The night porter"),("Fatima Zahra","Amira Hassan"),
("The pilot","The captain"),("Anna Kowalski","Lucia Costa"),("Snow","Rain"),("Ravi Kumar","Omar Haddad"),("The chef","The baker"),("Lucas Martin","Marek Novak"),("A young violinist","An experienced violinist"),("Mei Lin","Yuki Mori"),
("Marek Novak","Jonas Weber"),("Sara Ibrahim","Priya Nair"),("Noah Schmidt","Hiro Sato"),("Yuki Mori","Mei Lin"),("Amira Hassan","Fatima Zahra"),("Peter Novak","Carlos Mendes"),("Lucia Costa","Sofia Rossi"),("Daniel Kim","Liam O'Connor")]
ALT=[]
for s,(a,b) in zip(NEUTRAL,SWAP):
 assert s.startswith(a)and a!=b;ALT.append(b+s[len(a):])
LMAP=[min(31,max(0,round((L+.5)*32/28-.5)))for L in range(28)]
TRAIN=list(range(24));LAYERS=list(range(28));RIDGES=[1e-4,1e-3,1e-2,.05,.1,.5,1.,5.];WRITE_LAYERS=list(range(14,24));DOSES=[.30,1.00]
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
 seq=[pad(tok)]+ids(tok,s+SEP);o=model(input_ids=torch.tensor([seq],device=DEV),use_cache=False,output_hidden_states=True,return_dict=True);pos=max(0,len(seq)-len(ids(tok,SEP))-1)
 H=[o.hidden_states[L+1][0,pos].float().cpu().contiguous()for L in range(len(model.model.layers))];del o;return H
def unit(x):return x.float()/x.float().norm().clamp_min(1e-12)
def cos(a,b):return float(F.cosine_similarity(a.float().reshape(-1),b.float().reshape(-1),dim=0,eps=1e-12))
def packet(A,B,layers):return [unit(B[L]-A[L])for L in layers]
def catpacket(p):return torch.cat([unit(x)for x in p])
def gfp(x,bank,refs):return torch.tensor([cos(catpacket(x),catpacket(bank[j]))for j in refs],dtype=torch.float32)
def ridge(X,Y,lam):
 X=X.double();Y=Y.double();mx=X.mean(0);my=Y.mean(0);A=X-mx;B=Y-my;xx=A.T@A;xy=A.T@B;s=max(1e-12,float(xx.trace()/max(1,xx.shape[0])));W=torch.linalg.solve(xx+lam*s*torch.eye(xx.shape[0],dtype=torch.float64),xy);return W.float(),mx.float(),my.float()
def pred(x,f):W,mx,my=f;return (x.float()-mx)@W+my
def fit_global(MP,QP,refs,lam):
 GM=torch.stack([gfp(MP[i],MP,refs)for i in refs]);GQ=torch.stack([gfp(QP[i],QP,refs)for i in refs]);return ridge(GM,GQ,lam)
def global_qsolve(target,QP,refs,lam):
 C=torch.stack([catpacket(QP[j])for j in refs]);G=C@C.T;a=torch.linalg.solve(G.T@G+lam*torch.eye(len(refs)),G.T@target.float())
 dirs={L:unit(sum((a[k]*QP[j][L]for k,j in enumerate(refs)),torch.zeros_like(QP[refs[0]][L])))for L in LAYERS};return dirs,a
def loo_score(MP,QP,lam):
 fc=[];lc=[]
 for hold in TRAIN:
  refs=[j for j in TRAIN if j!=hold];fit=fit_global(MP,QP,refs,lam);qh=pred(gfp(MP[hold],MP,refs),fit);qt=gfp(QP[hold],QP,refs);dirs,_=global_qsolve(qh,QP,refs,lam);fc.append(cos(qh,qt));lc.append(sum(cos(dirs[L],QP[hold][L])for L in LAYERS)/28)
 return sum(fc)/len(fc),sum(lc)/len(lc)
print("="*128);print("TEST 433 — L26 ATTENTION→MLP NATIVE IDENTITY / FIRST-TOKEN READOUT X-RAY");print("TEST432 PROVEN LINEAGE | SOURCE-ABSENT WHO | FINAL NATIVE = MEASUREMENT ONLY");print("="*128)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/10] MISTRAL — TEST432 EXACT PACKETS")
mt,mm,ml=load(MID,(32,4096,32,8,128));mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight];M0=sha(mp);MH=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):MH.append((forge(mt,mm,a),forge(mt,mm,b)));print(f" MISTRAL {i+1}/32",end="\r")
MFINAL=(forge(mt,mm,FACT),forge(mt,mm,CF));print()
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/10] QWEN — TEST432 EXACT PACKETS")
qt,qm,ql=load(QID,(28,3584,28,4,128));qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight];Q0=sha(qp);QH=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):QH.append((forge(qt,qm,a),forge(qt,qm,b)));print(f" QWEN {i+1}/32",end="\r")
QFINAL=(forge(qt,qm,FACT),forge(qt,qm,CF));print("\n TRAIN=24 | VAL=8 | FINAL FACT/CF NEVER USED FOR FIT/SELECTION")
MP=[packet(a,b,LMAP)for a,b in MH];QP=[packet(a,b,LAYERS)for a,b in QH];MF=packet(MFINAL[0],MFINAL[1],LMAP);QF=packet(QFINAL[0],QFINAL[1],LAYERS)
print("[3/10] TRAIN-ONLY GLOBAL RIDGE")
CV=[]
for lam in RIDGES:
 fc,lc=loo_score(MP,QP,lam);CV.append({"lambda":lam,"fp_cos":fc,"local_cos":lc});print(f" λ={lam:<6g} FP={fc:+.6f} LOCAL={lc:+.6f}")
LAM=max(CV,key=lambda x:(x["local_cos"],x["fp_cos"]))["lambda"];print(" SELECTED λ=",LAM)
print("[4/10] FINAL GLOBAL SOLVE — FROZEN")
FIT=fit_global(MP,QP,TRAIN,LAM);QHAT=pred(gfp(MF,MP,TRAIN),FIT);QTRUE=gfp(QF,QP,TRAIN);DIRS,COEF=global_qsolve(QHAT,QP,TRAIN,LAM);FFP=cos(QHAT,QTRUE);FLOCAL=sum(cos(DIRS[L],QF[L])for L in LAYERS)/28
print(f" FINAL FP={FFP:+.6f} LOCAL={FLOCAL:+.6f} COEF_NORM={float(COEF.norm()):.6f}")
MAG={L:sum(float((QH[i][1][L]-QH[i][0][L]).norm())for i in TRAIN)/len(TRAIN)for L in LAYERS}
print("[5/10] SOURCE-ABSENT WHO + L26 MICRO-CAPTURE")
PROMPT=torch.tensor([[pad(qt)]+ids(qt,WHO)],device=DEV);POS=PROMPT.shape[1]-1;LEYLA=ids(qt," Leyla")[0];MUSTAFA=ids(qt," Mustafa")[0];L26=ql[26]
print(f" WHO TOKENS={PROMPT.shape[1]} | SOURCE FACT PRESENT=NO | LEYLA={LEYLA} | MUSTAFA={MUSTAFA}")
def run(seq,dose=0.0):
 cap={};hs=[]
 def inp(m,a):cap["IN"]=a[0][0,-1].detach().float().cpu().contiguous()
 def postatt(m,a):cap["POST_ATT"]=a[0][0,-1].detach().float().cpu().contiguous()
 def out(m,a,o):
  h=o[0]if isinstance(o,tuple)else o;cap["OUT"]=h[0,-1].detach().float().cpu().contiguous()
 hs.append(L26.register_forward_pre_hook(inp));hs.append(L26.post_attention_layernorm.register_forward_pre_hook(postatt));hs.append(L26.register_forward_hook(out))
 if dose:
  for L in WRITE_LAYERS:
   d=(DIRS[L]*MAG[L]*dose).to(DEV)
   def inj(m,a,o,d=d):
    if isinstance(o,tuple):
     h=o[0].clone();h[:,-1,:]+=d.to(h.dtype);return(h,)+o[1:]
    h=o.clone();h[:,-1,:]+=d.to(h.dtype);return h
   hs.append(ql[L].register_forward_hook(inj))
 try:
  with torch.inference_mode():o=qm(input_ids=seq,use_cache=False,return_dict=True)
  log=o.logits[0,-1].detach().float().cpu();del o
 finally:
  for h in hs:h.remove()
 return cap,log
BASE,LB=run(PROMPT,0);T03,L03=run(PROMPT,.30);T10,L10=run(PROMPT,1.00)
NF=torch.tensor([[pad(qt)]+ids(qt,FACT+SEP+WHO)],device=DEV);NC=torch.tensor([[pad(qt)]+ids(qt,CF+SEP+WHO)],device=DEV)
N0,LN0=run(NF,0);N1,LN1=run(NC,0)
def delta(A,B,k):return B[k]-A[k]
N={k:delta(N0,N1,k)for k in ["IN","POST_ATT","OUT"]}
D03={k:delta(BASE,T03,k)for k in ["IN","POST_ATT","OUT"]};D10={k:delta(BASE,T10,k)for k in ["IN","POST_ATT","OUT"]}
print("[6/10] TEST432 REPLICATION + FIRST-TOKEN READOUT")
def row(d,x):
 lr=int((x>x[LEYLA]).sum())+1;mr=int((x>x[MUSTAFA]).sum())+1;return lr,mr,float(x[LEYLA]),float(x[MUSTAFA]),float(x[LEYLA]-x[MUSTAFA])
for d,x in [(0.,LB),(.3,L03),(1.,L10)]:
 lr,mr,ll,ml,margin=row(d,x);print(f" {d:.2f}x LEYLA={ll:+.6f} #{lr} MUSTAFA={ml:+.6f} #{mr} L-M={margin:+.6f}")
print(f" NATIVE WHO Δ(L-M)={float((LN1[LEYLA]-LN1[MUSTAFA])-(LN0[LEYLA]-LN0[MUSTAFA])):+.6f}")
print("[7/10] L26 INPUT→POST_ATT→OUT NATIVE-vs-TRANSFER")
ST=[]
for k in ["IN","POST_ATT","OUT"]:
 z={"stage":k,"native_norm":float(N[k].norm()),"d03_norm":float(D03[k].norm()),"d10_norm":float(D10[k].norm()),"cos03_native":cos(D03[k],N[k]),"cos10_native":cos(D10[k],N[k])};ST.append(z)
 print(f" {k:<8} |N|={z['native_norm']:9.3f} |T.3|={z['d03_norm']:9.3f} cos(.3,N)={z['cos03_native']:+.6f} |T1|={z['d10_norm']:9.3f} cos(1,N)={z['cos10_native']:+.6f}")
print("[8/10] L26 ATTENTION / MLP COMPONENT ALIGNMENT")
NATT=N["POST_ATT"]-N["IN"];NMLP=N["OUT"]-N["POST_ATT"]
A03=D03["POST_ATT"]-D03["IN"];M03=D03["OUT"]-D03["POST_ATT"];A10=D10["POST_ATT"]-D10["IN"];M10=D10["OUT"]-D10["POST_ATT"]
COMP=[]
for name,n,a,b in [("ATT",NATT,A03,A10),("MLP",NMLP,M03,M10)]:
 z={"component":name,"native_norm":float(n.norm()),"d03_norm":float(a.norm()),"d10_norm":float(b.norm()),"cos03_native":cos(a,n),"cos10_native":cos(b,n)};COMP.append(z)
 print(f" {name:<3} |N|={z['native_norm']:9.3f} |T.3|={z['d03_norm']:9.3f} cos(.3,N)={z['cos03_native']:+.6f} |T1|={z['d10_norm']:9.3f} cos(1,N)={z['cos10_native']:+.6f}")
print("[9/10] IDENTITY AXIS — WHERE NATIVE BINDING APPEARS")
W=qm.lm_head.weight.detach().float().cpu();ID=unit(W[LEYLA]-W[MUSTAFA])
def pr(x):return float(x@ID)
for k in ["IN","POST_ATT","OUT"]:
 print(f" {k:<8} NATIVE={pr(N[k]):+10.4f} T.3={pr(D03[k]):+10.4f} T1={pr(D10[k]):+10.4f}")
print(f" ATT      NATIVE={pr(NATT):+10.4f} T.3={pr(A03):+10.4f} T1={pr(A10):+10.4f}")
print(f" MLP      NATIVE={pr(NMLP):+10.4f} T.3={pr(M03):+10.4f} T1={pr(M10):+10.4f}")
ATT_DEF03=pr(NATT)-pr(A03);ATT_DEF10=pr(NATT)-pr(A10);MLP_DEF03=pr(NMLP)-pr(M03);MLP_DEF10=pr(NMLP)-pr(M10)
print(f" ATT ID DEFICIT .30x={ATT_DEF03:+.6f} 1.00x={ATT_DEF10:+.6f}")
print(f" MLP ID DEFICIT .30x={MLP_DEF03:+.6f} 1.00x={MLP_DEF10:+.6f}")
if abs(ATT_DEF10)>abs(MLP_DEF10):VERDICT="L26_IDENTITY_DEFICIT_DOMINATED_BY_ATTENTION"
else:VERDICT="L26_IDENTITY_DEFICIT_DOMINATED_BY_MLP_OR_POST_ATTENTION"
print("[10/10] VERDICT + INTEGRITY");print(" VERDICT:",VERDICT)
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Qwen not frozen")
REPORT={"test":433,"title":"L26 Attention-to-MLP Native Identity / First-Token Readout X-Ray","seed":SEED,"source_model":MID,"target_model":QID,"fact":FACT,"counterfactual":CF,"who_query":WHO,"train_pairs":24,"write_layers":WRITE_LAYERS,"doses":DOSES,"selected_lambda":LAM,"cross_validation":CV,"global_final":{"fingerprint_cos":FFP,"local_mean":FLOCAL,"coef_norm":float(COEF.norm())},"stages":ST,"components":COMP,"identity":{"native_in":pr(N["IN"]),"native_post_att":pr(N["POST_ATT"]),"native_out":pr(N["OUT"]),"transfer03_in":pr(D03["IN"]),"transfer03_post_att":pr(D03["POST_ATT"]),"transfer03_out":pr(D03["OUT"]),"transfer10_in":pr(D10["IN"]),"transfer10_post_att":pr(D10["POST_ATT"]),"transfer10_out":pr(D10["OUT"]),"native_att":pr(NATT),"native_mlp":pr(NMLP),"transfer03_att":pr(A03),"transfer03_mlp":pr(M03),"transfer10_att":pr(A10),"transfer10_mlp":pr(M10),"att_deficit03":ATT_DEF03,"att_deficit10":ATT_DEF10,"mlp_deficit03":MLP_DEF03,"mlp_deficit10":MLP_DEF10},"verdict":VERDICT,"integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1,"pass":Q0==Q1},"notes":["TEST432 proven extraction, TRAIN-only ridge, global solve, magnitude calibration and L14-L23 write are preserved.","FINAL FACT/CF are measurement-only native WHO-context references and never enter fit, DIRS, MAG or writes.","L26 INPUT, POST_ATT and OUT are captured at the answer position to decompose native and transferred identity formation into attention and MLP components.","LM-head Leyla-minus-Mustafa axis is measurement-only.","No FINAL-derived vector is injected. No target forcing, constrained decoding, vocabulary masking, reranking or weight modification is used."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("="*128);print("TEST 433 — COMPLETE");print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS");print("SUMMARY:",OUT);print("="*128)
