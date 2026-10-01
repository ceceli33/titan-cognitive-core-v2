# TEST 423 — L24→L27 IDENTITY-BINDING FORMATION X-RAY
# TEST422 PROVEN LINEAGE | WRITE L14–L23 | MOTOR OFF L24–L27
# SOURCE-ABSENT WHO TRANSFER × NATIVE WHO-CONTEXT | LAYERWISE IDENTITY-SUBSPACE FORMATION
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
ROOT=Path("/content/AKBASCORE_TEST423")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST423");ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/"TEST423_SUMMARY.json"
FACT="Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge."
CF="Leyla Demir planted the Turkish flag at the base of the Golden Gate Bridge."
WHO="Who planted the Turkish flag at the base of the Golden Gate Bridge?\nAnswer:"
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
TRAIN=list(range(24));LAYERS=list(range(28));RIDGES=[1e-4,1e-3,1e-2,.05,.1,.5,1.,5.];WRITE_LAYERS=list(range(14,24));XRAY=[24,25,26,27];NNULL=64
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
 pos=max(0,len(seq)-len(ids(tok,SEP))-1);H=[o.hidden_states[L+1][0,pos].float().cpu().contiguous()for L in range(len(model.model.layers))]
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
def stat(vals,obs,tail="high"):
 v=torch.tensor(vals,dtype=torch.float32);mu=float(v.mean());sd=float(v.std(unbiased=True));z=(obs-mu)/max(sd,1e-12);p=float((1+((v>=obs)if tail=="high"else(v<=obs)).sum())/(len(v)+1))
 return {"obs":float(obs),"mean":mu,"sd":sd,"z":z,"p":p,"min":float(v.min()),"max":float(v.max())}
def split_sub(v,basis):
 if not basis:return torch.zeros_like(v),v.clone()
 p=sum((torch.dot(v.float(),q)*q for q in basis),torch.zeros_like(v.float()));return p,v.float()-p
def efrac(v,p):return float(p.square().sum()/v.float().square().sum().clamp_min(1e-30))
print("="*128);print("TEST 423 — L24→L27 IDENTITY-BINDING FORMATION X-RAY");print("TEST422 PROVEN LINEAGE | WRITE L14–L23 | MOTOR OFF L24–L27 | NATIVE vs TRANSFER IDENTITY FORMATION");print("="*128)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/14] MISTRAL — TEST422 EXACT PACKETS")
mt,mm,ml=load(MID,(32,4096,32,8,128));mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight];M0=sha(mp);MH=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):
 MH.append((forge(mt,mm,a),forge(mt,mm,b)));print(f" MISTRAL {i+1}/32",end="\r")
MFINAL=(forge(mt,mm,FACT),forge(mt,mm,CF));print()
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/14] QWEN — TEST422 EXACT PACKETS")
qt,qm,ql=load(QID,(28,3584,28,4,128));qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight];Q0=sha(qp);QH=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):
 QH.append((forge(qt,qm,a),forge(qt,qm,b)));print(f" QWEN {i+1}/32",end="\r")
QFINAL=(forge(qt,qm,FACT),forge(qt,qm,CF));print("\n TRAIN=24 | VAL=8 | FINAL FACT/CF NEVER USED FOR FIT/SELECTION")
MP=[packet(a,b,LMAP)for a,b in MH];QP=[packet(a,b,LAYERS)for a,b in QH];MF=packet(MFINAL[0],MFINAL[1],LMAP);QF=packet(QFINAL[0],QFINAL[1],LAYERS)
print("[3/14] TRAIN-ONLY GLOBAL RIDGE")
CV=[]
for lam in RIDGES:
 fc,lc=loo_score(MP,QP,lam);CV.append({"lambda":lam,"fp_cos":fc,"local_cos":lc});print(f" λ={lam:<6g} FP={fc:+.6f} LOCAL={lc:+.6f}")
BEST=max(CV,key=lambda x:(x["local_cos"],x["fp_cos"]));LAM=BEST["lambda"];print(" SELECTED λ=",LAM)
print("[4/14] FINAL GLOBAL SOLVE — FROZEN")
FIT=fit_global(MP,QP,TRAIN,LAM);QHAT=pred(gfp(MF,MP,TRAIN),FIT);QTRUE=gfp(QF,QP,TRAIN);DIRS,COEF=global_qsolve(QHAT,QP,TRAIN,LAM)
FFP=cos(QHAT,QTRUE);FLOCAL=sum(cos(DIRS[L],QF[L])for L in LAYERS)/28
print(f" FINAL FP={FFP:+.6f} LOCAL={FLOCAL:+.6f} COEF_NORM={float(COEF.norm()):.6f}")
MAG={L:sum(float((QH[i][1][L]-QH[i][0][L]).norm())for i in TRAIN)/len(TRAIN)for L in LAYERS}
print("[5/14] TEST422 SOURCE-ABSENT WHO + L24→L27 CAPTURE")
who_ids=torch.tensor([[pad(qt)]+ids(qt,WHO)],device=DEV);WHO_POS=who_ids.shape[1]-1
def run_xray(seq,pos,dirs=None,sign=1.,l26_replace=None):
 whs=[];hs=[];cap={}
 if dirs is not None:
  for L in WRITE_LAYERS:
   d=(dirs[L]*MAG[L]*sign).to(DEV)
   def wh(mod,inp,out,d=d,pos=pos):
    if isinstance(out,tuple):
     h=out[0].clone();h[:,pos,:]+=d.to(h.dtype);return (h,)+out[1:]
    h=out.clone();h[:,pos,:]+=d.to(h.dtype);return h
   whs.append(ql[L].register_forward_hook(wh))
 if l26_replace is not None:
  r=l26_replace.to(DEV)
  def rep(mod,inp,r=r,pos=pos):
   x=inp[0].clone();x[:,pos,:]=r.to(x.dtype);return (x,)+inp[1:]
  hs.append(ql[27].register_forward_pre_hook(rep))
 for L in XRAY:
  def pre(mod,inp,L=L,pos=pos):cap[f"L{L}_IN"]=inp[0][0,pos].detach().float().cpu().clone()
  def post(mod,inp,out,L=L,pos=pos):
   h=out[0]if isinstance(out,tuple)else out;cap[f"L{L}_OUT"]=h[0,pos].detach().float().cpu().clone()
  hs.append(ql[L].register_forward_pre_hook(pre));hs.append(ql[L].register_forward_hook(post))
 def fnorm(mod,inp,out,pos=pos):cap["POST"]=out[0,pos].detach().float().cpu().clone()
 hs.append(qm.model.norm.register_forward_hook(fnorm))
 with torch.inference_mode():o=qm(input_ids=seq,use_cache=False,return_dict=True)
 for h in whs+hs:h.remove()
 cap["LOGITS"]=o.logits[0,pos].detach().float().cpu().clone();del o;return cap
BASE=run_xray(who_ids,WHO_POS);PRED=run_xray(who_ids,WHO_POS,DIRS,+1);REV=run_xray(who_ids,WHO_POS,DIRS,-1)
print(f" WHO TOKENS={who_ids.shape[1]} | SOURCE FACT PRESENT=NO")
print("[6/14] NATIVE FACT-vs-CF WHO-CONTEXT X-RAY — EVALUATION ONLY")
def native_context(text):
 seq=torch.tensor([[pad(qt)]+ids(qt,text+SEP+WHO)],device=DEV);return run_xray(seq,seq.shape[1]-1)
NF=native_context(FACT);NC=native_context(CF)
print(" NATIVE WHO-CONTEXT CAPTURE: PASS")
print("[7/14] TEST422 FINAL READOUT REPLICATION")
W=qm.lm_head.weight.detach().float().cpu();NPOST=NC["POST"]-NF["POST"];DPOST=PRED["POST"]-BASE["POST"];RPOST=REV["POST"]-BASE["POST"];NSPEC=NC["LOGITS"]-NF["LOGITS"];DSPEC=PRED["LOGITS"]-BASE["LOGITS"];RSPEC=REV["LOGITS"]-BASE["LOGITS"]
BACK=unit(W.T@NSPEC);COORD=float(torch.dot(DPOST,BACK));RCOORD=float(torch.dot(RPOST,BACK));VDOT=float(torch.dot(DSPEC,NSPEC))
print(f" WHO-CONTEXT POST COS={cos(DPOST,NPOST):+.6f} | REV={cos(RPOST,NPOST):+.6f}")
print(f" WHO-CONTEXT VOCAB COS={cos(DSPEC,NSPEC):+.6f} | REV={cos(RSPEC,NSPEC):+.6f}")
print(f" BACKPROJ COORD={COORD:+.6f} | REV={RCOORD:+.6f} | VOCAB DOT={VDOT:+.3f}")
print("[8/14] IDENTITY SUBSPACE — TEST422 EXACT")
AXES=[]
for prefix in [""," "]:
 ai=ids(qt,prefix+NAME_A);bi=ids(qt,prefix+NAME_B)
 if ai and bi and ai[0]!=bi[0]:
  v=W[bi[0]]-W[ai[0]]
  for q in AXES:v=v-torch.dot(v,q)*q
  if v.norm()>1e-8:AXES.append(unit(v))
print(" IDENTITY SUBSPACE RANK=",len(AXES))
FPAR,_=split_sub(NPOST,AXES);TPAR,_=split_sub(DPOST,AXES)
print(f" FINAL NATIVE ID-PAR ENERGY={100*efrac(NPOST,FPAR):.6f}%")
print(f" FINAL TRANSFER ID-PAR ENERGY={100*efrac(DPOST,TPAR):.6f}%")
print("[9/14] L24→L27 IDENTITY-BINDING FORMATION")
ROWS=[]
for L in XRAY:
 key=f"L{L}_OUT";n=NC[key]-NF[key];d=PRED[key]-BASE[key];r=REV[key]-BASE[key];np,no=split_sub(n,AXES);dp,do=split_sub(d,AXES);rp,ro=split_sub(r,AXES)
 row={"layer":L,"native_norm":float(n.norm()),"transfer_norm":float(d.norm()),"reverse_norm":float(r.norm()),"full_cos":cos(d,n),"reverse_full_cos":cos(r,n),
      "native_id_energy":efrac(n,np),"transfer_id_energy":efrac(d,dp),"reverse_id_energy":efrac(r,rp),
      "id_parallel_cos":cos(dp,np)if dp.norm()>1e-12 and np.norm()>1e-12 else None,"orthogonal_cos":cos(do,no)if do.norm()>1e-12 and no.norm()>1e-12 else None,
      "native_id_norm":float(np.norm()),"transfer_id_norm":float(dp.norm()),"reverse_id_norm":float(rp.norm())}
 ROWS.append(row)
 print(f" L{L:02d} FULL={row['full_cos']:+.6f} REV={row['reverse_full_cos']:+.6f} | ID-ENERGY native={100*row['native_id_energy']:.6f}% transfer={100*row['transfer_id_energy']:.6f}% rev={100*row['reverse_id_energy']:.6f}% | ID-COS={row['id_parallel_cos']:+.6f}")
print("[10/14] PER-BLOCK IDENTITY ENERGY GAIN")
GAINS=[]
for i,L in enumerate(XRAY):
 kin=f"L{L}_IN";kout=f"L{L}_OUT"
 ni=NC[kin]-NF[kin];no=NC[kout]-NF[kout];di=PRED[kin]-BASE[kin];do=PRED[kout]-BASE[kout]
 nip,_=split_sub(ni,AXES);nop,_=split_sub(no,AXES);dip,_=split_sub(di,AXES);dop,_=split_sub(do,AXES)
 ng=efrac(no,nop)-efrac(ni,nip);dg=efrac(do,dop)-efrac(di,dip)
 GAINS.append({"layer":L,"native_in":efrac(ni,nip),"native_out":efrac(no,nop),"native_gain":ng,"transfer_in":efrac(di,dip),"transfer_out":efrac(do,dop),"transfer_gain":dg})
 print(f" L{L:02d} NATIVE {100*efrac(ni,nip):.6f}%→{100*efrac(no,nop):.6f}% Δ={100*ng:+.6f}pp | TRANSFER {100*efrac(di,dip):.6f}%→{100*efrac(do,dop):.6f}% Δ={100*dg:+.6f}pp")
PEAK=max(GAINS,key=lambda x:x["native_gain"])
print(f" PEAK NATIVE IDENTITY-FORMATION BLOCK=L{PEAK['layer']:02d} | GAIN={100*PEAK['native_gain']:+.6f}pp")
print("[11/14] TEST422 WHO MATCHED L26 NULL CONSTRUCTION")
N26=QFINAL[1][26]-QFINAL[0][26];WD26=PRED["L26_OUT"]-BASE["L26_OUT"];WTN=float(WD26.norm());WTC=cos(WD26,N26);WU=unit(N26);WOBSORTH=unit(unit(WD26)-torch.dot(unit(WD26),WU)*WU);CONTROLS=[]
for s in range(NNULL):
 g=torch.Generator().manual_seed(SEED+2400+s);r=torch.randn(WD26.numel(),generator=g);r=r-torch.dot(r,WU)*WU;r=r-torch.dot(r,WOBSORTH)*WOBSORTH
 if r.norm()<1e-8:raise RuntimeError("Degenerate WHO null")
 r=unit(r);v=WTC*WU+math.sqrt(max(0.,1.-WTC*WTC))*r;v=unit(v)*WTN;CONTROLS.append(v)
errn=max(abs(float(v.norm())-WTN)for v in CONTROLS);errc=max(abs(cos(v,N26)-WTC)for v in CONTROLS);orthmax=max(abs(cos(v-WTC*WTN*WU,WOBSORTH))for v in CONTROLS)
print(f" NULL SANITY NORM={errn:.8e} COS={errc:.8e} ORTH={orthmax:.8e}")
if errn>1e-3 or errc>1e-5 or orthmax>1e-5:raise RuntimeError("Matched-null construction failed")
print("[12/14] L27 MATCHED-NULL IDENTITY FORMATION")
NULL_E=[];NULL_N=[];NULL_GAIN=[];NULL_IDCOS=[];NULL_BACK=[];NULL_VCOS=[]
# Controls replace L27 input exactly as TEST422; therefore assay isolates the final L27 transformation.
B26=BASE["L27_IN"];OBS_IN=PRED["L27_IN"]-BASE["L27_IN"];OBS_OUT=PRED["L27_OUT"]-BASE["L27_OUT"];NIN=NC["L27_IN"]-NF["L27_IN"];NOUT=NC["L27_OUT"]-NF["L27_OUT"]
OIP,_=split_sub(OBS_IN,AXES);OOP,_=split_sub(OBS_OUT,AXES);NIP,_=split_sub(NIN,AXES);NOP,_=split_sub(NOUT,AXES)
OBS_GAIN=efrac(OBS_OUT,OOP)-efrac(OBS_IN,OIP);OBS_EN=efrac(OBS_OUT,OOP);OBS_NORM=float(OOP.norm());OBS_IDCOS=cos(OOP,NOP)if OOP.norm()>1e-12 and NOP.norm()>1e-12 else 0.
for i,v in enumerate(CONTROLS):
 R=run_xray(who_ids,WHO_POS,None,+1,B26+v);rin=v;dout=R["L27_OUT"]-BASE["L27_OUT"];rip,_=split_sub(rin,AXES);rop,_=split_sub(dout,AXES)
 NULL_E.append(efrac(dout,rop));NULL_N.append(float(rop.norm()));NULL_GAIN.append(efrac(dout,rop)-efrac(rin,rip));NULL_IDCOS.append(cos(rop,NOP)if rop.norm()>1e-12 and NOP.norm()>1e-12 else 0.)
 post=R["POST"]-BASE["POST"];spec=R["LOGITS"]-BASE["LOGITS"];NULL_BACK.append(float(torch.dot(post,BACK)));NULL_VCOS.append(cos(spec,NSPEC))
 print(f" MATCHED {i+1}/{NNULL}",end="\r")
print()
SE=stat(NULL_E,OBS_EN);SN=stat(NULL_N,OBS_NORM);SG=stat(NULL_GAIN,OBS_GAIN);SIC=stat(NULL_IDCOS,OBS_IDCOS);SB=stat(NULL_BACK,COORD);SVC=stat(NULL_VCOS,cos(DSPEC,NSPEC))
print(f" L27 ID-ENERGY OBS={100*SE['obs']:.6f}% NULL={100*SE['mean']:.6f}%±{100*SE['sd']:.6f}% z={SE['z']:+.3f} p={SE['p']:.6f}")
print(f" L27 ID-NORM   OBS={SN['obs']:.6f} NULL={SN['mean']:.6f}±{SN['sd']:.6f} z={SN['z']:+.3f} p={SN['p']:.6f}")
print(f" L27 ID-GAIN   OBS={100*SG['obs']:+.6f}pp NULL={100*SG['mean']:+.6f}±{100*SG['sd']:.6f}pp z={SG['z']:+.3f} p={SG['p']:.6f}")
print(f" L27 ID-COS    OBS={SIC['obs']:+.6f} NULL={SIC['mean']:+.6f}±{SIC['sd']:.6f} z={SIC['z']:+.3f} p={SIC['p']:.6f}")
print(f" FINAL BACKPROJ OBS={SB['obs']:+.6f} NULL={SB['mean']:+.6f}±{SB['sd']:.6f} p={SB['p']:.6f}")
print(f" FINAL VOCAB COS OBS={SVC['obs']:+.6f} NULL={SVC['mean']:+.6f}±{SVC['sd']:.6f} p={SVC['p']:.6f}")
print("[13/14] IDENTITY-BINDING BOTTLENECK LOCALIZATION")
native_gains=[x["native_gain"] for x in GAINS];transfer_gains=[x["transfer_gain"] for x in GAINS]
PEAKIDX=max(range(len(XRAY)),key=lambda i:native_gains[i]);PEAKL=XRAY[PEAKIDX]
if SE["p"]>.05 and SG["p"]>.05:
 BOTTLENECK=f"IDENTITY_BINDING_MISSING: native WHO identity energy forms across L24-L27 (peak L{PEAKL}), but the transferred state does not acquire matched-null-specific identity energy in L27."
elif SIC["p"]<=.05:
 BOTTLENECK=f"PARTIAL_IDENTITY_GEOMETRY: L27 produces a matched-null-specific native-aligned identity component, but direct behavioral binding must still be tested."
else:
 BOTTLENECK=f"IDENTITY_ENERGY_WITHOUT_NATIVE_ALIGNMENT: L27 changes identity-subspace energy but does not specifically align it to the native WHO identity direction."
print(" LOCALIZATION:",BOTTLENECK)
print("[14/14] VERDICT + INTEGRITY")
if SE["p"]<=.05 and SIC["p"]<=.05:
 VERDICT="L27_NATIVE_IDENTITY_BINDING_COMPONENT_DETECTED"
elif SE["p"]<=.05 or SG["p"]<=.05:
 VERDICT="L27_IDENTITY_ENERGY_EFFECT_WITHOUT_COMPLETE_NATIVE_BINDING"
else:
 VERDICT="DISTRIBUTED_READOUT_SURVIVES_BUT_IDENTITY_BINDING_FAILS_ACROSS_FINAL_TAIL"
print(" VERDICT:",VERDICT)
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Qwen not frozen")
REPORT={"test":423,"title":"L24-L27 Identity-Binding Formation X-Ray","seed":SEED,"source_model":MID,"target_model":QID,"fact":FACT,"counterfactual":CF,"who_query":WHO,
"train_pairs":24,"layer_map":LMAP,"write_layers":WRITE_LAYERS,"engine_off_layers":XRAY,"selected_lambda":LAM,"cross_validation":CV,
"global_final":{"fingerprint_cos":FFP,"local_mean":FLOCAL,"coef_norm":float(COEF.norm())},
"test422_replication":{"who_post_native_cos":cos(DPOST,NPOST),"who_vocab_native_cos":cos(DSPEC,NSPEC),"backprojection_coordinate":COORD,"reverse_coordinate":RCOORD,"vocab_dot":VDOT,"native_identity_energy":efrac(NPOST,FPAR),"transfer_identity_energy":efrac(DPOST,TPAR)},
"layerwise_identity":ROWS,"block_identity_gains":GAINS,"peak_native_identity_formation_layer":PEAKL,
"l27_matched_null":{"n":NNULL,"max_norm_error":errn,"max_native_endpoint_cos_error":errc,"max_control_orth_error":orthmax,"identity_energy":SE,"identity_norm":SN,"identity_gain":SG,"identity_native_cos":SIC,"final_backprojection":SB,"final_vocab_cos":SVC},
"localization":BOTTLENECK,"verdict":VERDICT,"integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1,"pass":Q0==Q1},
"notes":["TEST422 proven Mistral/Qwen extraction, TRAIN-only ridge, frozen global solve, L14-L23 write, doses, source-absent WHO prompt and motor-off L24-L27 tail are preserved.","FINAL FACT/CF remains excluded from fitting, lambda selection and intervention construction.","Native FACT+WHO versus CF+WHO is evaluation-only and is never injected.","Identity subspace is the same two-axis Mustafa/Leyla LM-head row-difference construction used in TEST422.","L24-L27 are observed without intervention to locate where native WHO identity-subspace energy forms and where the transferred state fails to follow it.","The matched-null assay replaces only L27 input, using TEST422's exact norm/cosine-matched control construction, to test whether the observed upstream transferred state receives a specific final-block identity-binding conversion.","No model weights are modified."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("="*128);print("TEST 423 — COMPLETE");print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS");print("SUMMARY:",OUT);print("="*128)
