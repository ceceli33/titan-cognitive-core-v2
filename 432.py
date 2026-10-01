# TEST 432 — L26 NATIVE IDENTITY-BINDING DEFICIT X-RAY
# TEST431 PROVEN LINEAGE | SOURCE-ABSENT WHO | FINAL FACT/CF = MEASUREMENT REFERENCE ONLY
# NO FINAL-DERIVED WRITE | NO TARGET FORCING | NO CONSTRAINED DECODING
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
ROOT=Path("/content/AKBASCORE_TEST432")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST432");ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/"TEST432_SUMMARY.json"
FACT="Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge."
CF="Leyla Demir planted the Turkish flag at the base of the Golden Gate Bridge."
WHO="Who planted the Turkish flag at the base of the Golden Gate Bridge?\nAnswer:"
DOSES=[0.00,0.30,1.00];PROBE_LAYERS=[23,24,25,26,27]
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
TRAIN=list(range(24));LAYERS=list(range(28));RIDGES=[1e-4,1e-3,1e-2,.05,.1,.5,1.,5.];WRITE_LAYERS=list(range(14,24))
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
print("="*128);print("TEST 432 — L26 NATIVE IDENTITY-BINDING DEFICIT X-RAY");print("TEST431 PROVEN LINEAGE | FINAL NATIVE = MEASUREMENT REFERENCE ONLY | NO FINAL-DERIVED WRITE");print("="*128)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/10] MISTRAL — TEST431 EXACT PACKETS")
mt,mm,ml=load(MID,(32,4096,32,8,128));mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight];M0=sha(mp);MH=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):MH.append((forge(mt,mm,a),forge(mt,mm,b)));print(f" MISTRAL {i+1}/32",end="\r")
MFINAL=(forge(mt,mm,FACT),forge(mt,mm,CF));print()
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/10] QWEN — TEST431 EXACT PACKETS")
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
print("[5/10] SOURCE-ABSENT WHO + NATIVE WHO-CONTEXT REFERENCE")
PROMPT=torch.tensor([[pad(qt)]+ids(qt,WHO)],device=DEV);POS=PROMPT.shape[1]-1
LEYLA=ids(qt," Leyla")[0];MUSTAFA=ids(qt," Mustafa")[0]
print(f" WHO TOKENS={PROMPT.shape[1]} | SOURCE FACT PRESENT=NO | LEYLA_ID={LEYLA} | MUSTAFA_ID={MUSTAFA}")
@torch.inference_mode()
def states(seq):
 o=qm(input_ids=seq,use_cache=False,output_hidden_states=True,return_dict=True);H=[o.hidden_states[L+1][0,-1].float().cpu().contiguous()for L in range(28)];log=o.logits[0,-1].float().cpu();del o;return H,log
NF=torch.tensor([[pad(qt)]+ids(qt,FACT+SEP+WHO)],device=DEV);NC=torch.tensor([[pad(qt)]+ids(qt,CF+SEP+WHO)],device=DEV)
HN0,LN0=states(NF);HN1,LN1=states(NC);NATIVE={L:HN1[L]-HN0[L]for L in PROBE_LAYERS}
print(f" NATIVE WHO Δ(Leyla-Mustafa)={float((LN1[LEYLA]-LN1[MUSTAFA])-(LN0[LEYLA]-LN0[MUSTAFA])):+.6f}")
def transfer_states(dose):
 caps={};hooks=[]
 try:
  for L in PROBE_LAYERS:
   def cap(m,i,o,L=L):
    h=o[0]if isinstance(o,tuple)else o;caps[L]=h[0,-1].detach().float().cpu().contiguous()
   hooks.append(ql[L].register_forward_hook(cap))
  if dose!=0:
   for L in WRITE_LAYERS:
    d=(DIRS[L]*MAG[L]*dose).to(DEV)
    def inj(m,i,o,d=d):
     if isinstance(o,tuple):
      h=o[0].clone();h[:,-1,:]+=d.to(h.dtype);return (h,)+o[1:]
     h=o.clone();h[:,-1,:]+=d.to(h.dtype);return h
    hooks.append(ql[L].register_forward_hook(inj))
  with torch.inference_mode():o=qm(input_ids=PROMPT,use_cache=False,return_dict=True)
  log=o.logits[0,-1].detach().float().cpu();del o
  return {L:caps[L] for L in PROBE_LAYERS},log
 finally:
  for h in hooks:
   try:h.remove()
   except:pass
print("[6/10] TEST431 REPLICATION")
BASEH,BASELOG=transfer_states(0.0);T03,L03=transfer_states(.30);T10,L10=transfer_states(1.0)
for d,x in [(0.,BASELOG),(.3,L03),(1.,L10)]:
 lr=int((x>x[LEYLA]).sum())+1;mr=int((x>x[MUSTAFA]).sum())+1
 print(f" {d:.2f}x LEYLA={float(x[LEYLA]):+.6f} #{lr} MUSTAFA={float(x[MUSTAFA]):+.6f} #{mr} L-M={float(x[LEYLA]-x[MUSTAFA]):+.6f}")
print("[7/10] L23→L27 NATIVE-vs-TRANSFER DEFICIT")
ROWS=[]
for L in PROBE_LAYERS:
 nd=NATIVE[L];d03=T03[L]-BASEH[L];d10=T10[L]-BASEH[L]
 r={"layer":L,"native_norm":float(nd.norm()),"d03_norm":float(d03.norm()),"d10_norm":float(d10.norm()),"cos03_native":cos(d03,nd),"cos10_native":cos(d10,nd)}
 ROWS.append(r);print(f" L{L:02d} |N|={r['native_norm']:9.3f} |T.3|={r['d03_norm']:9.3f} cos(.3,N)={r['cos03_native']:+.6f} |T1|={r['d10_norm']:9.3f} cos(1,N)={r['cos10_native']:+.6f}")
print("[8/10] BLOCKWISE NATIVE IDENTITY-FORMATION vs TRANSFER")
GAINS=[]
for A,B in zip(PROBE_LAYERS[:-1],PROBE_LAYERS[1:]):
 ng=NATIVE[B]-NATIVE[A];g03=(T03[B]-BASEH[B])-(T03[A]-BASEH[A]);g10=(T10[B]-BASEH[B])-(T10[A]-BASEH[A])
 z={"from":A,"to":B,"native_gain_norm":float(ng.norm()),"d03_gain_norm":float(g03.norm()),"d10_gain_norm":float(g10.norm()),"cos03_native_gain":cos(g03,ng),"cos10_native_gain":cos(g10,ng)}
 GAINS.append(z);print(f" L{A:02d}→L{B:02d} |NGAIN|={z['native_gain_norm']:9.3f} cos(.3,NGAIN)={z['cos03_native_gain']:+.6f} cos(1,NGAIN)={z['cos10_native_gain']:+.6f}")
print("[9/10] LM-HEAD IDENTITY-AXIS PROJECTION BY LAYER")
W=qm.lm_head.weight.detach().float().cpu();ID=unit(W[LEYLA]-W[MUSTAFA]);PROJ=[]
for L in PROBE_LAYERS:
 nd=NATIVE[L];d03=T03[L]-BASEH[L];d10=T10[L]-BASEH[L]
 p={"layer":L,"native":float(nd@ID),"d03":float(d03@ID),"d10":float(d10@ID)}
 PROJ.append(p);print(f" L{L:02d} NATIVE·ID={p['native']:+10.4f} T.3·ID={p['d03']:+10.4f} T1·ID={p['d10']:+10.4f}")
peak=max(GAINS,key=lambda z:z["native_gain_norm"]);l26=next((z for z in GAINS if z["from"]==25 and z["to"]==26),None)
if l26 and l26["cos03_native_gain"]<=0 and l26["cos10_native_gain"]<=0:VERDICT="L26_NATIVE_IDENTITY_CONVERSION_MISSING_FROM_TRANSFER"
elif l26 and max(l26["cos03_native_gain"],l26["cos10_native_gain"])<.20:VERDICT="L26_NATIVE_IDENTITY_CONVERSION_WEAKLY_CAPTURED"
else:VERDICT="L26_NATIVE_IDENTITY_CONVERSION_PARTIALLY_PRESENT"
print("[10/10] VERDICT + INTEGRITY")
print(" NATIVE PEAK BLOCK:",f"L{peak['from']}→L{peak['to']}","| norm=",f"{peak['native_gain_norm']:.6f}")
print(" VERDICT:",VERDICT)
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Qwen not frozen")
REPORT={"test":432,"title":"L26 Native Identity-Binding Deficit X-Ray","seed":SEED,"source_model":MID,"target_model":QID,"fact":FACT,"counterfactual":CF,"who_query":WHO,"train_pairs":24,"write_layers":WRITE_LAYERS,"probe_layers":PROBE_LAYERS,"doses":DOSES,"selected_lambda":LAM,"cross_validation":CV,"global_final":{"fingerprint_cos":FFP,"local_mean":FLOCAL,"coef_norm":float(COEF.norm())},"layer_comparison":ROWS,"block_gains":GAINS,"identity_axis_projection":PROJ,"native_peak_block":peak,"verdict":VERDICT,"integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1,"pass":Q0==Q1},"notes":["TEST431 proven Mistral→Qwen extraction, TRAIN-only ridge, global solve, magnitude calibration and L14-L23 write are preserved.","FINAL FACT/CF are used only as an evaluation-side native Qwen WHO-context reference and never enter fit, ridge selection, DIRS, MAG or any write.","No FINAL-derived residual direction is injected into Qwen.","Source-absent WHO transfer is measured independently at doses 0.30 and 1.00.","L23-L27 hidden-state deltas are compared with native Qwen FACT→CF WHO-context deltas to localize the missing identity conversion.","LM-head identity axis is measurement-only.","No target forcing, constrained decoding, vocabulary masking, reranking or model-weight modification is used."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("="*128);print("TEST 432 — COMPLETE");print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS");print("SUMMARY:",OUT);print("="*128)



