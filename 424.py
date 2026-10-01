# TEST 424 — L26 ATTENTION→MLP IDENTITY-BINDING MICRO-X-RAY
# TEST423 PROVEN LINEAGE | WRITE L14–L23 | MOTOR OFF L24–L27
# SOURCE-ABSENT WHO TRANSFER × NATIVE WHO-CONTEXT | L26 INPUT→ATT→MLP/OUT
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
ROOT=Path("/content/AKBASCORE_TEST424")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST424");ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/"TEST424_SUMMARY.json"
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
TRAIN=list(range(24));LAYERS=list(range(28));RIDGES=[1e-4,1e-3,1e-2,.05,.1,.5,1.,5.];WRITE_LAYERS=list(range(14,24));NNULL=64
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
def split_sub(v,basis):
 if not basis:return torch.zeros_like(v),v.clone()
 p=sum((torch.dot(v.float(),q)*q for q in basis),torch.zeros_like(v.float()));return p,v.float()-p
def efrac(v,p):return float(p.square().sum()/v.float().square().sum().clamp_min(1e-30))
def stat(vals,obs,tail="high"):
 v=torch.tensor(vals,dtype=torch.float32);mu=float(v.mean());sd=float(v.std(unbiased=True));z=(obs-mu)/max(sd,1e-12);p=float((1+((v>=obs)if tail=="high"else(v<=obs)).sum())/(len(v)+1))
 return {"obs":float(obs),"mean":mu,"sd":sd,"z":z,"p":p,"min":float(v.min()),"max":float(v.max())}
print("="*128);print("TEST 424 — L26 ATTENTION→MLP IDENTITY-BINDING MICRO-X-RAY");print("TEST423 PROVEN LINEAGE | WRITE L14–L23 | MOTOR OFF L24–L27 | SOURCE-ABSENT WHO × NATIVE WHO-CONTEXT");print("="*128)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/14] MISTRAL — TEST423 EXACT PACKETS")
mt,mm,ml=load(MID,(32,4096,32,8,128));mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight];M0=sha(mp);MH=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):
 MH.append((forge(mt,mm,a),forge(mt,mm,b)));print(f" MISTRAL {i+1}/32",end="\r")
MFINAL=(forge(mt,mm,FACT),forge(mt,mm,CF));print()
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/14] QWEN — TEST423 EXACT PACKETS")
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
print("[5/14] L26 MICRO-CAPTURE HOOKS")
who_ids=torch.tensor([[pad(qt)]+ids(qt,WHO)],device=DEV);WHO_POS=who_ids.shape[1]-1
def run_micro(seq,pos,dirs=None,sign=1.,l26_replace=None):
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
  hs.append(ql[26].register_forward_pre_hook(rep))
 def l26in(mod,inp,pos=pos):cap["IN"]=inp[0][0,pos].detach().float().cpu().clone()
 def postatt_in(mod,inp,pos=pos):cap["POST_ATT"]=inp[0][0,pos].detach().float().cpu().clone()
 def l26out(mod,inp,out,pos=pos):
  h=out[0]if isinstance(out,tuple)else out;cap["OUT"]=h[0,pos].detach().float().cpu().clone()
 def l27in(mod,inp,pos=pos):cap["L27_IN"]=inp[0][0,pos].detach().float().cpu().clone()
 def fnorm(mod,inp,out,pos=pos):cap["FINAL"]=out[0,pos].detach().float().cpu().clone()
 hs.append(ql[26].register_forward_pre_hook(l26in))
 hs.append(ql[26].post_attention_layernorm.register_forward_pre_hook(postatt_in))
 hs.append(ql[26].register_forward_hook(l26out))
 hs.append(ql[27].register_forward_pre_hook(l27in))
 hs.append(qm.model.norm.register_forward_hook(fnorm))
 with torch.inference_mode():o=qm(input_ids=seq,use_cache=False,return_dict=True)
 for h in whs+hs:h.remove()
 cap["LOGITS"]=o.logits[0,pos].detach().float().cpu().clone();del o
 cap["ATT"]=cap["POST_ATT"]-cap["IN"];cap["MLP"]=cap["OUT"]-cap["POST_ATT"]
 return cap
BASE=run_micro(who_ids,WHO_POS);PRED=run_micro(who_ids,WHO_POS,DIRS,+1);REV=run_micro(who_ids,WHO_POS,DIRS,-1)
print(f" WHO TOKENS={who_ids.shape[1]} | SOURCE FACT PRESENT=NO")
print("[6/14] NATIVE FACT-vs-CF WHO-CONTEXT — EVALUATION ONLY")
def native_context(text):
 seq=torch.tensor([[pad(qt)]+ids(qt,text+SEP+WHO)],device=DEV);return run_micro(seq,seq.shape[1]-1)
NF=native_context(FACT);NC=native_context(CF)
print(" NATIVE MICRO-CAPTURE: PASS")
print("[7/14] TEST423 L26 REPLICATION")
W=qm.lm_head.weight.detach().float().cpu();AXES=[]
for prefix in [""," "]:
 ai=ids(qt,prefix+NAME_A);bi=ids(qt,prefix+NAME_B)
 if ai and bi and ai[0]!=bi[0]:
  v=W[bi[0]]-W[ai[0]]
  for q in AXES:v=v-torch.dot(v,q)*q
  if v.norm()>1e-8:AXES.append(unit(v))
print(" IDENTITY SUBSPACE RANK=",len(AXES))
NIN=NC["IN"]-NF["IN"];NOUT=NC["OUT"]-NF["OUT"];DIN=PRED["IN"]-BASE["IN"];DOUT=PRED["OUT"]-BASE["OUT"];RIN=REV["IN"]-BASE["IN"];ROUT=REV["OUT"]-BASE["OUT"]
NIP,_=split_sub(NIN,AXES);NOP,_=split_sub(NOUT,AXES);DIP,_=split_sub(DIN,AXES);DOP,_=split_sub(DOUT,AXES)
print(f" L26 NATIVE {100*efrac(NIN,NIP):.6f}%→{100*efrac(NOUT,NOP):.6f}% Δ={100*(efrac(NOUT,NOP)-efrac(NIN,NIP)):+.6f}pp")
print(f" L26 TRANSFER {100*efrac(DIN,DIP):.6f}%→{100*efrac(DOUT,DOP):.6f}% Δ={100*(efrac(DOUT,DOP)-efrac(DIN,DIP)):+.6f}pp")
print("[8/14] L26 INPUT→POST-ATT→OUT IDENTITY FORMATION")
STAGES=["IN","POST_ATT","OUT"]
ROWS=[]
for st in STAGES:
 n=NC[st]-NF[st];d=PRED[st]-BASE[st];r=REV[st]-BASE[st];np,no=split_sub(n,AXES);dp,do=split_sub(d,AXES);rp,ro=split_sub(r,AXES)
 row={"stage":st,"native_norm":float(n.norm()),"transfer_norm":float(d.norm()),"reverse_norm":float(r.norm()),"full_cos":cos(d,n),"reverse_cos":cos(r,n),
      "native_id_energy":efrac(n,np),"transfer_id_energy":efrac(d,dp),"reverse_id_energy":efrac(r,rp),"native_id_norm":float(np.norm()),"transfer_id_norm":float(dp.norm()),
      "id_cos":cos(dp,np)if dp.norm()>1e-12 and np.norm()>1e-12 else None,"orth_cos":cos(do,no)if do.norm()>1e-12 and no.norm()>1e-12 else None}
 ROWS.append(row)
 print(f" {st:8s} FULL={row['full_cos']:+.6f} | ID-ENERGY native={100*row['native_id_energy']:.6f}% transfer={100*row['transfer_id_energy']:.6f}% rev={100*row['reverse_id_energy']:.6f}% | ID-COS={row['id_cos']:+.6f}")
print("[9/14] ATTENTION vs MLP CONTRIBUTION")
NATT=NC["ATT"]-NF["ATT"];NMLP=NC["MLP"]-NF["MLP"];DATT=PRED["ATT"]-BASE["ATT"];DMLP=PRED["MLP"]-BASE["MLP"];RATT=REV["ATT"]-BASE["ATT"];RMLP=REV["MLP"]-BASE["MLP"]
COMP=[]
for name,n,d,r in [("ATT",NATT,DATT,RATT),("MLP",NMLP,DMLP,RMLP)]:
 np,no=split_sub(n,AXES);dp,do=split_sub(d,AXES);rp,ro=split_sub(r,AXES)
 row={"component":name,"native_norm":float(n.norm()),"transfer_norm":float(d.norm()),"native_id_norm":float(np.norm()),"transfer_id_norm":float(dp.norm()),
      "native_id_energy":efrac(n,np),"transfer_id_energy":efrac(d,dp),"reverse_id_energy":efrac(r,rp),"full_cos":cos(d,n),
      "id_cos":cos(dp,np)if dp.norm()>1e-12 and np.norm()>1e-12 else None,"orth_cos":cos(do,no)if do.norm()>1e-12 and no.norm()>1e-12 else None}
 COMP.append(row)
 print(f" {name}: FULL={row['full_cos']:+.6f} | ID-NORM native={row['native_id_norm']:.6f} transfer={row['transfer_id_norm']:.6f} | ID-ENERGY native={100*row['native_id_energy']:.6f}% transfer={100*row['transfer_id_energy']:.6f}% | ID-COS={row['id_cos']:+.6f}")
N_E=[efrac(NC[s]-NF[s],split_sub(NC[s]-NF[s],AXES)[0])for s in STAGES];D_E=[efrac(PRED[s]-BASE[s],split_sub(PRED[s]-BASE[s],AXES)[0])for s in STAGES]
N_ATT_GAIN=N_E[1]-N_E[0];N_MLP_GAIN=N_E[2]-N_E[1];D_ATT_GAIN=D_E[1]-D_E[0];D_MLP_GAIN=D_E[2]-D_E[1]
print(f" NATIVE ATT GAIN={100*N_ATT_GAIN:+.6f}pp | MLP GAIN={100*N_MLP_GAIN:+.6f}pp")
print(f" TRANSFER ATT GAIN={100*D_ATT_GAIN:+.6f}pp | MLP GAIN={100*D_MLP_GAIN:+.6f}pp")
DOM="ATTENTION" if N_ATT_GAIN>N_MLP_GAIN else "MLP"
print(" NATIVE DOMINANT L26 IDENTITY-FORMATION COMPONENT:",DOM)
print("[10/14] COMPONENT DOT/PROJECTION INTO NATIVE IDENTITY DIRECTION")
NID=unit(NOP) if NOP.norm()>1e-12 else torch.zeros_like(NOP)
PROJ={}
for name,v in [("NATIVE_ATT",NATT),("NATIVE_MLP",NMLP),("TRANSFER_ATT",DATT),("TRANSFER_MLP",DMLP),("REVERSE_ATT",RATT),("REVERSE_MLP",RMLP)]:
 PROJ[name]=float(torch.dot(v,NID));print(f" {name:14s} → NATIVE-L26-ID = {PROJ[name]:+.6f}")
print("[11/14] MATCHED L26-INPUT NULL CONSTRUCTION")
NREF=QFINAL[1][25]-QFINAL[0][25];TN=float(DIN.norm());TC=cos(DIN,NREF);U=unit(NREF);OBSORTH=unit(unit(DIN)-torch.dot(unit(DIN),U)*U);CONTROLS=[]
for s in range(NNULL):
 g=torch.Generator().manual_seed(SEED+3400+s);r=torch.randn(DIN.numel(),generator=g);r=r-torch.dot(r,U)*U;r=r-torch.dot(r,OBSORTH)*OBSORTH
 if r.norm()<1e-8:raise RuntimeError("Degenerate L26 null")
 r=unit(r);v=TC*U+math.sqrt(max(0.,1.-TC*TC))*r;v=unit(v)*TN;CONTROLS.append(v)
errn=max(abs(float(v.norm())-TN)for v in CONTROLS);errc=max(abs(cos(v,NREF)-TC)for v in CONTROLS);orthmax=max(abs(cos(v-TC*TN*U,OBSORTH))for v in CONTROLS)
print(f" NULL SANITY NORM={errn:.8e} COS={errc:.8e} ORTH={orthmax:.8e}")
if errn>1e-3 or errc>1e-5 or orthmax>1e-5:raise RuntimeError("L26 matched-null construction failed")
print("[12/14] MATCHED-NULL L26 ATTENTION/MLP ASSAY")
NULL_ATT_GAIN=[];NULL_MLP_GAIN=[];NULL_OUT_E=[];NULL_OUT_N=[];NULL_OUT_COS=[];NULL_ATT_PROJ=[];NULL_MLP_PROJ=[]
for i,v in enumerate(CONTROLS):
 R=run_micro(who_ids,WHO_POS,None,+1,BASE["IN"]+v);ri=v;rpa=R["POST_ATT"]-BASE["POST_ATT"];ro=R["OUT"]-BASE["OUT"];ra=rpa-ri;rm=ro-rpa
 rip,_=split_sub(ri,AXES);rpp,_=split_sub(rpa,AXES);rop,_=split_sub(ro,AXES)
 NULL_ATT_GAIN.append(efrac(rpa,rpp)-efrac(ri,rip));NULL_MLP_GAIN.append(efrac(ro,rop)-efrac(rpa,rpp));NULL_OUT_E.append(efrac(ro,rop));NULL_OUT_N.append(float(rop.norm()));NULL_OUT_COS.append(cos(rop,NOP)if rop.norm()>1e-12 and NOP.norm()>1e-12 else 0.)
 NULL_ATT_PROJ.append(float(torch.dot(ra,NID)));NULL_MLP_PROJ.append(float(torch.dot(rm,NID)))
 print(f" MATCHED {i+1}/{NNULL}",end="\r")
print()
SAG=stat(NULL_ATT_GAIN,D_ATT_GAIN);SMG=stat(NULL_MLP_GAIN,D_MLP_GAIN);SOE=stat(NULL_OUT_E,efrac(DOUT,DOP));SON=stat(NULL_OUT_N,float(DOP.norm()));SOC=stat(NULL_OUT_COS,cos(DOP,NOP)if DOP.norm()>1e-12 and NOP.norm()>1e-12 else 0.)
SAP=stat(NULL_ATT_PROJ,PROJ["TRANSFER_ATT"]);SMP=stat(NULL_MLP_PROJ,PROJ["TRANSFER_MLP"])
print(f" ATT ID-GAIN OBS={100*SAG['obs']:+.6f}pp NULL={100*SAG['mean']:+.6f}±{100*SAG['sd']:.6f}pp z={SAG['z']:+.3f} p={SAG['p']:.6f}")
print(f" MLP ID-GAIN OBS={100*SMG['obs']:+.6f}pp NULL={100*SMG['mean']:+.6f}±{100*SMG['sd']:.6f}pp z={SMG['z']:+.3f} p={SMG['p']:.6f}")
print(f" OUT ID-ENERGY OBS={100*SOE['obs']:.6f}% NULL={100*SOE['mean']:.6f}%±{100*SOE['sd']:.6f}% p={SOE['p']:.6f}")
print(f" OUT ID-NORM OBS={SON['obs']:.6f} NULL={SON['mean']:.6f}±{SON['sd']:.6f} p={SON['p']:.6f}")
print(f" OUT ID-COS OBS={SOC['obs']:+.6f} NULL={SOC['mean']:+.6f}±{SOC['sd']:.6f} p={SOC['p']:.6f}")
print(f" ATT→NATIVE-ID OBS={SAP['obs']:+.6f} NULL={SAP['mean']:+.6f}±{SAP['sd']:.6f} p={SAP['p']:.6f}")
print(f" MLP→NATIVE-ID OBS={SMP['obs']:+.6f} NULL={SMP['mean']:+.6f}±{SMP['sd']:.6f} p={SMP['p']:.6f}")
print("[13/14] L26 BOTTLENECK LOCALIZATION")
if N_ATT_GAIN>N_MLP_GAIN:
 if D_ATT_GAIN<=0:BOTTLENECK="L26_ATTENTION_BINDING_FAILURE"
 elif SAP["p"]>.05:BOTTLENECK="L26_ATTENTION_NON_NATIVE_IDENTITY_CONVERSION"
 else:BOTTLENECK="L26_ATTENTION_PARTIAL_NATIVE_BINDING"
else:
 if D_MLP_GAIN<=0:BOTTLENECK="L26_MLP_BINDING_FAILURE"
 elif SMP["p"]>.05:BOTTLENECK="L26_MLP_NON_NATIVE_IDENTITY_CONVERSION"
 else:BOTTLENECK="L26_MLP_PARTIAL_NATIVE_BINDING"
print(" BOTTLENECK:",BOTTLENECK)
print("[14/14] VERDICT + INTEGRITY")
if DOM=="ATTENTION" and N_ATT_GAIN>0 and D_ATT_GAIN<=0:VERDICT="NATIVE_L26_IDENTITY_FORMATION_IS_ATTENTION_DOMINANT_TRANSFER_FAILS_AT_ATTENTION"
elif DOM=="MLP" and N_MLP_GAIN>0 and D_MLP_GAIN<=0:VERDICT="NATIVE_L26_IDENTITY_FORMATION_IS_MLP_DOMINANT_TRANSFER_FAILS_AT_MLP"
elif SOC["p"]<=.05:VERDICT="TRANSFER_ACQUIRES_SPECIFIC_L26_NATIVE_IDENTITY_COMPONENT"
else:VERDICT="L26_IDENTITY_FORMATION_MECHANISM_LOCALIZED_WITHOUT_TRANSFER_BINDING"
print(" VERDICT:",VERDICT)
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Qwen not frozen")
REPORT={"test":424,"title":"L26 Attention-to-MLP Identity-Binding Micro-X-Ray","seed":SEED,"source_model":MID,"target_model":QID,"fact":FACT,"counterfactual":CF,"who_query":WHO,
"train_pairs":24,"layer_map":LMAP,"write_layers":WRITE_LAYERS,"selected_lambda":LAM,"cross_validation":CV,
"global_final":{"fingerprint_cos":FFP,"local_mean":FLOCAL,"coef_norm":float(COEF.norm())},
"test423_l26_replication":{"native_input_energy":efrac(NIN,NIP),"native_output_energy":efrac(NOUT,NOP),"transfer_input_energy":efrac(DIN,DIP),"transfer_output_energy":efrac(DOUT,DOP)},
"micro_stages":ROWS,"components":COMP,"gains":{"native_attention":N_ATT_GAIN,"native_mlp":N_MLP_GAIN,"transfer_attention":D_ATT_GAIN,"transfer_mlp":D_MLP_GAIN,"native_dominant":DOM},
"native_identity_projections":PROJ,
"matched_null":{"n":NNULL,"max_norm_error":errn,"max_reference_cos_error":errc,"max_control_orth_error":orthmax,"attention_identity_gain":SAG,"mlp_identity_gain":SMG,"output_identity_energy":SOE,"output_identity_norm":SON,"output_identity_cos":SOC,"attention_native_identity_projection":SAP,"mlp_native_identity_projection":SMP},
"bottleneck":BOTTLENECK,"verdict":VERDICT,"integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1,"pass":Q0==Q1},
"notes":["TEST423 proven extraction, TRAIN-only ridge, frozen global solve, L14-L23 write, source-absent WHO prompt and motor-off tail are preserved.","FINAL FACT/CF remains evaluation-only and is excluded from fit, lambda selection and intervention construction.","L26 is decomposed exactly as residual input, post-attention residual at post_attention_layernorm input, and block output; ATT=POST_ATT-IN and MLP=OUT-POST_ATT.","Matched controls replace only L26 block input and preserve the observed transfer displacement norm and cosine to the frozen Qwen endpoint reference while excluding the observed orthogonal component.","No model weights are modified."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("="*128);print("TEST 424 — COMPLETE");print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS");print("SUMMARY:",OUT);print("="*128)



