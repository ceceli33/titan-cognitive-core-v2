# TEST 419 — VOCABULARY-WIDE READOUT SPECTRUM X-RAY
# TEST418 PROVEN LINEAGE | WRITE L14–L23 | MOTOR OFF L24–L27
# MATCHED-NULL-SPECIFIC ΔPOST → FULL LM-HEAD VOCABULARY SPECTRUM | TOP TOKEN / RANK / CONCENTRATION AUDIT
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
ROOT=Path("/content/AKBASCORE_TEST419")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST419");ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/"TEST419_SUMMARY.json"
FACT="Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge."
CF="Leyla Demir planted the Turkish flag at the base of the Golden Gate Bridge."
NAME_A="Mustafa Akbaş";NAME_B="Leyla Demir"
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
TRAIN=list(range(24));LAYERS=list(range(28));RIDGES=[1e-4,1e-3,1e-2,.05,.1,.5,1.,5.];WRITE_LAYERS=list(range(14,24));NNULL=64;TOPK=32
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
def safe_token(tok,i):
 try:
  s=tok.decode([int(i)],skip_special_tokens=False,clean_up_tokenization_spaces=False)
  return s.replace("\n","\\n").replace("\r","\\r").replace("\t","\\t")
 except:return f"<ID:{i}>"
def rank_desc(x,idx):return int((x>x[idx]).sum())+1
def rank_asc(x,idx):return int((x<x[idx]).sum())+1
def top_records(tok,x,k,largest=True):
 val,ind=torch.topk(x,k=min(k,x.numel()),largest=largest)
 return [{"rank":i+1,"id":int(t),"token":safe_token(tok,int(t)),"score":float(v)}for i,(v,t) in enumerate(zip(val,ind))]
def stat(vals,obs,tail="high"):
 v=torch.tensor(vals,dtype=torch.float32);mu=float(v.mean());sd=float(v.std(unbiased=True));z=(obs-mu)/max(sd,1e-12)
 p=float((1+((v>=obs)if tail=="high"else(v<=obs)).sum())/(len(v)+1))
 return {"obs":float(obs),"mean":mu,"sd":sd,"z":z,"p":p,"min":float(v.min()),"max":float(v.max())}
print("="*128);print("TEST 419 — VOCABULARY-WIDE READOUT SPECTRUM X-RAY");print("TEST418 PROVEN LINEAGE | WRITE L14–L23 | MOTOR OFF L24–L27 | ΔPOST → FULL LM-HEAD VOCABULARY");print("="*128)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/12] MISTRAL — TEST418 EXACT PACKETS")
mt,mm,ml=load(MID,(32,4096,32,8,128));mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight];M0=sha(mp);MH=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):
 MH.append((forge(mt,mm,a),forge(mt,mm,b)));print(f" MISTRAL {i+1}/32",end="\r")
MFINAL=(forge(mt,mm,FACT),forge(mt,mm,CF));print()
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/12] QWEN — TEST418 EXACT PACKETS")
qt,qm,ql=load(QID,(28,3584,28,4,128));qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight];Q0=sha(qp);QH=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):
 QH.append((forge(qt,qm,a),forge(qt,qm,b)));print(f" QWEN {i+1}/32",end="\r")
QFINAL=(forge(qt,qm,FACT),forge(qt,qm,CF));print("\n TRAIN=24 | VAL=8 | FINAL FACT/CF NEVER USED FOR FIT/SELECTION")
MP=[packet(a,b,LMAP)for a,b in MH];QP=[packet(a,b,LAYERS)for a,b in QH];MF=packet(MFINAL[0],MFINAL[1],LMAP);QF=packet(QFINAL[0],QFINAL[1],LAYERS)
print("[3/12] TRAIN-ONLY GLOBAL RIDGE")
CV=[]
for lam in RIDGES:
 fc,lc=loo_score(MP,QP,lam);CV.append({"lambda":lam,"fp_cos":fc,"local_cos":lc});print(f" λ={lam:<6g} FP={fc:+.6f} LOCAL={lc:+.6f}")
BEST=max(CV,key=lambda x:(x["local_cos"],x["fp_cos"]));LAM=BEST["lambda"];print(" SELECTED λ=",LAM)
print("[4/12] FINAL GLOBAL SOLVE — FROZEN")
FIT=fit_global(MP,QP,TRAIN,LAM);QHAT=pred(gfp(MF,MP,TRAIN),FIT);QTRUE=gfp(QF,QP,TRAIN);DIRS,COEF=global_qsolve(QHAT,QP,TRAIN,LAM)
FFP=cos(QHAT,QTRUE);FLOCAL=sum(cos(DIRS[L],QF[L])for L in LAYERS)/28
print(f" FINAL FP={FFP:+.6f} LOCAL={FLOCAL:+.6f} COEF_NORM={float(COEF.norm()):.6f}")
print("[5/12] TEST418 EXACT MOTOR-OFF POST-NORM CAPTURE")
MAG={L:sum(float((QH[i][1][L]-QH[i][0][L]).norm())for i in TRAIN)/len(TRAIN)for L in LAYERS}
source_ids=torch.tensor([[pad(qt)]+ids(qt,FACT+SEP)],device=DEV);nsep=len(ids(qt,SEP));POS=source_ids.shape[1]-nsep-1
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
 def cap26(mod,inp):cap["L26"]=inp[0][0,POS].detach().float().cpu().clone()
 def postnorm(mod,inp,out):cap["POST"]=out[0,POS].detach().float().cpu().clone()
 hs.append(ql[27].register_forward_pre_hook(cap26));hs.append(qm.model.norm.register_forward_hook(postnorm))
 with torch.inference_mode():o=qm(input_ids=source_ids,use_cache=False,return_dict=True)
 for h in whs+hs:h.remove()
 logits=o.logits[0,POS].detach().float().cpu().clone();del o
 return {"L26":cap["L26"],"POST":cap["POST"],"LOGITS":logits}
BASE=run();PRED=run(DIRS,+1);REV=run(DIRS,-1)
N26=QFINAL[1][26]-QFINAL[0][26];D26=PRED["L26"]-BASE["L26"];DPOST=PRED["POST"]-BASE["POST"];RPOST=REV["POST"]-BASE["POST"];TN=float(D26.norm());TC=cos(D26,N26)
def native_capture(text):
 seq=torch.tensor([[pad(qt)]+ids(qt,text+SEP)],device=DEV);pos=seq.shape[1]-len(ids(qt,SEP))-1;cap={};hs=[]
 def post(mod,inp,out):cap["POST"]=out[0,pos].detach().float().cpu().clone()
 hs.append(qm.model.norm.register_forward_hook(post))
 with torch.inference_mode():o=qm(input_ids=seq,use_cache=False,return_dict=True)
 for h in hs:h.remove()
 logits=o.logits[0,pos].detach().float().cpu().clone();del o
 return cap["POST"],logits
NFPOST,NFLOG=native_capture(FACT);NCPOST,NCLOG=native_capture(CF);NPOST=NCPOST-NFPOST
print(f" L26 NORM={TN:.6f} NATIVE_COS={TC:+.6f} | POST_NATIVE_COS={cos(DPOST,NPOST):+.6f}")
print("[6/12] FULL-VOCABULARY LM-HEAD SPECTRA")
W=qm.lm_head.weight.detach().float().cpu()
SPEC=W@DPOST;RSPEC=W@RPOST;NSPEC=W@NPOST
ACT=PRED["LOGITS"]-BASE["LOGITS"];RACT=REV["LOGITS"]-BASE["LOGITS"];NACT=NCLOG-NFLOG
ERR=float((SPEC-ACT).abs().max());RMSE=float(torch.sqrt(torch.mean((SPEC-ACT)**2)));CORR=cos(SPEC,ACT);TOL=.20
print(f" VOCAB={SPEC.numel()} | FP32→BF16 MAX_ERR={ERR:.6f} RMSE={RMSE:.6f} COS={CORR:+.8f}")
if CORR<.999 or RMSE>.03 or ERR>TOL:raise RuntimeError("Vocabulary LM-head accounting failed")
print(f" TRANSFER↔NATIVE SPECTRUM COS={cos(SPEC,NSPEC):+.6f} | REVERSE↔NATIVE={cos(RSPEC,NSPEC):+.6f}")
print("[7/12] TOP POSITIVE / NEGATIVE READOUT TOKENS")
TOP_POS=top_records(qt,SPEC,TOPK,True);TOP_NEG=top_records(qt,SPEC,TOPK,False);NTOP_POS=top_records(qt,NSPEC,TOPK,True);NTOP_NEG=top_records(qt,NSPEC,TOPK,False)
print(" TRANSFER TOP +")
for r in TOP_POS[:16]:print(f"  {r['rank']:02d} id={r['id']:6d} {repr(r['token'])[:34]:34s} {r['score']:+.6f}")
print(" TRANSFER TOP -")
for r in TOP_NEG[:16]:print(f"  {r['rank']:02d} id={r['id']:6d} {repr(r['token'])[:34]:34s} {r['score']:+.6f}")
print("[8/12] NATIVE-SPECTRUM RANK / OVERLAP AUDIT")
def idset(rs,k):return set(x["id"]for x in rs[:k])
OVERLAP={}
for k in [10,32,100,256]:
 tp=set(torch.topk(SPEC,k).indices.tolist());np_=set(torch.topk(NSPEC,k).indices.tolist());tn=set(torch.topk(-SPEC,k).indices.tolist());nn=set(torch.topk(-NSPEC,k).indices.tolist())
 OVERLAP[str(k)]={"positive":len(tp&np_),"negative":len(tn&nn),"positive_jaccard":len(tp&np_)/max(1,len(tp|np_)),"negative_jaccard":len(tn&nn)/max(1,len(tn|nn))}
 print(f" TOP{k:3d} OVERLAP +={len(tp&np_):3d}/{k} J={OVERLAP[str(k)]['positive_jaccard']:.4f} | -={len(tn&nn):3d}/{k} J={OVERLAP[str(k)]['negative_jaccard']:.4f}")
forms=[]
for prefix in [""," "]:
 a=ids(qt,prefix+NAME_A);b=ids(qt,prefix+NAME_B)
 if a and b and a[0]!=b[0]:forms.append({"prefix":prefix,"a0":a[0],"b0":b[0]})
for f in forms:
 for label,idx in [("Mustafa",f["a0"]),("Leyla",f["b0"])]:
  print(f" {label:7s} prefix={repr(f['prefix'])}: Δ={float(SPEC[idx]):+.6f} nativeΔ={float(NSPEC[idx]):+.6f} rank+={rank_desc(SPEC,idx)} rank-={rank_asc(SPEC,idx)}")
print("[9/12] SPECTRAL CONCENTRATION / NATIVE PROJECTION")
def concentration(x,k):
 e=x.double().square();return float(torch.topk(e,min(k,e.numel())).values.sum()/e.sum().clamp_min(1e-30))
for k in [10,32,100,256,1024]:
 print(f" ENERGY TOP{k:4d}: TRANSFER={100*concentration(SPEC,k):.4f}% NATIVE={100*concentration(NSPEC,k):.4f}%")
NU=unit(NSPEC);SPAR=torch.dot(SPEC,NU)*NU;SORTH=SPEC-SPAR
print(f" VOCAB NATIVE-PARALLEL ENERGY={100*float(SPAR.square().sum()/SPEC.square().sum().clamp_min(1e-30)):.4f}% | ORTH={100*float(SORTH.square().sum()/SPEC.square().sum().clamp_min(1e-30)):.4f}%")
print(f" VOCAB DOT TRANSFER→NATIVE={float(torch.dot(SPEC,NSPEC)):+.6f}")
print("[10/12] TEST418 EXACT MATCHED L26 NULL → VOCAB SPECTRUM")
U26=unit(N26);OBS_ORTH=unit(unit(D26)-torch.dot(unit(D26),U26)*U26);CONTROLS=[]
for s in range(NNULL):
 g=torch.Generator().manual_seed(SEED+1400+s);r=torch.randn(D26.numel(),generator=g);r=r-torch.dot(r,U26)*U26;r=r-torch.dot(r,OBS_ORTH)*OBS_ORTH
 if r.norm()<1e-8:raise RuntimeError("Degenerate matched-null direction")
 r=unit(r);v=TC*U26+math.sqrt(max(0.,1.-TC*TC))*r;v=unit(v)*TN;CONTROLS.append(v)
errn=max(abs(float(v.norm())-TN)for v in CONTROLS);errc=max(abs(cos(v,N26)-TC)for v in CONTROLS);orthmax=max(abs(cos(v-TC*TN*U26,OBS_ORTH))for v in CONTROLS)
print(f" MATCH SANITY NORM={errn:.8e} COS={errc:.8e} ORTH={orthmax:.8e}")
if errn>1e-3 or errc>1e-5 or orthmax>1e-5:raise RuntimeError("Matched-null construction failed")
NULL_COS=[];NULL_DOT=[];NULL_TOP32=[];NULL_TOP256=[];NULL_MAX=[];NULL_L2=[]
for i,v in enumerate(CONTROLS):
 R=run(None,+1,BASE["L26"]+v);ds=R["LOGITS"]-BASE["LOGITS"]
 NULL_COS.append(cos(ds,NSPEC));NULL_DOT.append(float(torch.dot(ds,NSPEC)));NULL_TOP32.append(concentration(ds,32));NULL_TOP256.append(concentration(ds,256));NULL_MAX.append(float(ds.abs().max()));NULL_L2.append(float(ds.norm()));print(f" MATCHED {i+1}/{NNULL}",end="\r")
print()
OBS_COS=cos(ACT,NSPEC);OBS_DOT=float(torch.dot(ACT,NSPEC));OBS_T32=concentration(ACT,32);OBS_T256=concentration(ACT,256);OBS_MAX=float(ACT.abs().max());OBS_L2=float(ACT.norm())
SCOS=stat(NULL_COS,OBS_COS);SDOT=stat(NULL_DOT,OBS_DOT);ST32=stat(NULL_TOP32,OBS_T32);ST256=stat(NULL_TOP256,OBS_T256);SMAX=stat(NULL_MAX,OBS_MAX);SL2=stat(NULL_L2,OBS_L2)
print(f" SPECTRUM COS OBS={SCOS['obs']:+.6f} NULL={SCOS['mean']:+.6f}±{SCOS['sd']:.6f} z={SCOS['z']:+.3f} p={SCOS['p']:.6f}")
print(f" NATIVE DOT   OBS={SDOT['obs']:+.3f} NULL={SDOT['mean']:+.3f}±{SDOT['sd']:.3f} z={SDOT['z']:+.3f} p={SDOT['p']:.6f}")
print(f" TOP32 ENERGY OBS={100*ST32['obs']:.4f}% NULL={100*ST32['mean']:.4f}%±{100*ST32['sd']:.4f}% p={ST32['p']:.6f}")
print(f" TOP256 ENRGY OBS={100*ST256['obs']:.4f}% NULL={100*ST256['mean']:.4f}%±{100*ST256['sd']:.4f}% p={ST256['p']:.6f}")
print(f" MAX |ΔLOGIT| OBS={SMAX['obs']:.6f} NULL={SMAX['mean']:.6f}±{SMAX['sd']:.6f} p={SMAX['p']:.6f}")
print(f" VOCAB L2     OBS={SL2['obs']:.6f} NULL={SL2['mean']:.6f}±{SL2['sd']:.6f} p={SL2['p']:.6f}")
print("[11/12] READOUT DISTRIBUTION DIAGNOSIS")
if SCOS["p"]<=.05:
 DIAG="NATIVE_VOCABULARY_SPECTRUM_SPECIFIC: transferred geometry reaches a matched-null-specific vocabulary-wide native FACT→CF readout pattern even though the direct Mustafa↔Leyla axis is weak."
elif SDOT["p"]<=.05:
 DIAG="NATIVE_VOCABULARY_PROJECTION_SPECIFIC: transferred readout has significant native-spectrum projection but normalized spectrum similarity is not significant."
else:
 DIAG="READOUT_SPECTRUM_NOT_SPECIFIC: transferred hidden geometry is specific, but its vocabulary-wide LM-head readout does not exceed matched L26 controls."
print(" DIAGNOSIS:",DIAG)
print("[12/12] VERDICT + INTEGRITY")
if SCOS["p"]<=.05:
 VERDICT="DISTRIBUTED_READOUT_BRIDGE_DETECTED"
elif SDOT["p"]<=.05:
 VERDICT="PARTIAL_DISTRIBUTED_READOUT_BRIDGE"
else:
 VERDICT="GEOMETRY_REMAINS_READOUT_SILENT_AT_VOCABULARY_LEVEL"
print(" VERDICT:",VERDICT)
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Qwen not frozen")
REPORT={"test":419,"title":"Vocabulary-Wide Readout Spectrum X-Ray","seed":SEED,"source_model":MID,"target_model":QID,"fact":FACT,"counterfactual":CF,
"train_pairs":24,"layer_map":LMAP,"write_layers":WRITE_LAYERS,"engine_off_layers":[24,25,26,27],"selected_lambda":LAM,"cross_validation":CV,
"global_final":{"fingerprint_cos":FFP,"local_mean":FLOCAL,"coef_norm":float(COEF.norm())},
"geometry":{"l26_norm":TN,"l26_native_cos":TC,"post_native_cos":cos(DPOST,NPOST)},
"vocab_accounting":{"vocab_size":int(SPEC.numel()),"max_abs_error":ERR,"rmse":RMSE,"cosine":CORR},
"spectrum":{"transfer_native_cos":cos(SPEC,NSPEC),"reverse_native_cos":cos(RSPEC,NSPEC),"transfer_l2":float(SPEC.norm()),"native_l2":float(NSPEC.norm()),"native_parallel_energy_fraction":float(SPAR.square().sum()/SPEC.square().sum().clamp_min(1e-30)),"top_positive":TOP_POS,"top_negative":TOP_NEG,"native_top_positive":NTOP_POS,"native_top_negative":NTOP_NEG,"overlap":OVERLAP},
"identity_tokens":[{"prefix":f["prefix"],"mustafa_id":f["a0"],"mustafa_transfer":float(SPEC[f["a0"]]),"mustafa_native":float(NSPEC[f["a0"]]),"mustafa_rank_positive":rank_desc(SPEC,f["a0"]),"leyla_id":f["b0"],"leyla_transfer":float(SPEC[f["b0"]]),"leyla_native":float(NSPEC[f["b0"]]),"leyla_rank_positive":rank_desc(SPEC,f["b0"])}for f in forms],
"matched_null":{"n":NNULL,"max_norm_error":errn,"max_native_cos_error":errc,"max_control_orth_to_observed_orth_cos":orthmax,"spectrum_cos":SCOS,"native_dot":SDOT,"top32_energy":ST32,"top256_energy":ST256,"max_abs_logit":SMAX,"vocab_l2":SL2},
"diagnosis":DIAG,"verdict":VERDICT,"integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1,"pass":Q0==Q1},
"notes":["TEST418 proven extraction, datasets, global solve, TRAIN-only dose, L14-L23 write, motor-off tail and matched L26 controls are preserved.","FINAL FACT/CF remains evaluation-only.","The complete Qwen vocabulary is read out through the frozen lm_head; no token subset is used to select or fit the transfer.","Transfer and native FACT→CF vocabulary displacement spectra are compared by cosine, dot product, top-k overlap and spectral concentration.","Actual BF16 logit displacement is used for matched-null inference; FP32 W@ΔPOST is retained as an accounting/X-ray representation.","No weights are modified and no free generation is used."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("="*128);print("TEST 419 — COMPLETE");print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS");print("SUMMARY:",OUT);print("="*128)
