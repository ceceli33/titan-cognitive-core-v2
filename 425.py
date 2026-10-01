# TEST 425 — L26 ATTENTION Q/K/V/O CAUSAL IDENTITY-BINDING X-RAY
# TEST424 PROVEN LINEAGE | WRITE L14–L23 | MOTOR OFF L24–L27
# SOURCE-ABSENT WHO | L26 ATTENTION Q/K/V/O CAUSAL ABLATION
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
ROOT=Path("/content/AKBASCORE_TEST425")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST425");ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/"TEST425_SUMMARY.json"
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
print("="*128);print("TEST 425 — L26 ATTENTION Q/K/V/O CAUSAL IDENTITY-BINDING X-RAY");print("TEST424 PROVEN LINEAGE | WRITE L14–L23 | MOTOR OFF L24–L27 | SOURCE-ABSENT WHO");print("="*128)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/15] MISTRAL — TEST424 EXACT PACKETS")
mt,mm,ml=load(MID,(32,4096,32,8,128));mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight];M0=sha(mp);MH=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):
 MH.append((forge(mt,mm,a),forge(mt,mm,b)));print(f" MISTRAL {i+1}/32",end="\r")
MFINAL=(forge(mt,mm,FACT),forge(mt,mm,CF));print()
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/15] QWEN — TEST424 EXACT PACKETS")
qt,qm,ql=load(QID,(28,3584,28,4,128));qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight];Q0=sha(qp);QH=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):
 QH.append((forge(qt,qm,a),forge(qt,qm,b)));print(f" QWEN {i+1}/32",end="\r")
QFINAL=(forge(qt,qm,FACT),forge(qt,qm,CF));print("\n TRAIN=24 | VAL=8 | FINAL FACT/CF NEVER USED FOR FIT/SELECTION")
MP=[packet(a,b,LMAP)for a,b in MH];QP=[packet(a,b,LAYERS)for a,b in QH];MF=packet(MFINAL[0],MFINAL[1],LMAP);QF=packet(QFINAL[0],QFINAL[1],LAYERS)
print("[3/15] TRAIN-ONLY GLOBAL RIDGE")
CV=[]
for lam in RIDGES:
 fc,lc=loo_score(MP,QP,lam);CV.append({"lambda":lam,"fp_cos":fc,"local_cos":lc});print(f" λ={lam:<6g} FP={fc:+.6f} LOCAL={lc:+.6f}")
BEST=max(CV,key=lambda x:(x["local_cos"],x["fp_cos"]));LAM=BEST["lambda"];print(" SELECTED λ=",LAM)
print("[4/15] FINAL GLOBAL SOLVE — FROZEN")
FIT=fit_global(MP,QP,TRAIN,LAM);QHAT=pred(gfp(MF,MP,TRAIN),FIT);QTRUE=gfp(QF,QP,TRAIN);DIRS,COEF=global_qsolve(QHAT,QP,TRAIN,LAM)
FFP=cos(QHAT,QTRUE);FLOCAL=sum(cos(DIRS[L],QF[L])for L in LAYERS)/28
print(f" FINAL FP={FFP:+.6f} LOCAL={FLOCAL:+.6f} COEF_NORM={float(COEF.norm()):.6f}")
MAG={L:sum(float((QH[i][1][L]-QH[i][0][L]).norm())for i in TRAIN)/len(TRAIN)for L in LAYERS}
print("[5/15] L26 Q/K/V/O CAUSAL HOOK ENGINE")
who_ids=torch.tensor([[pad(qt)]+ids(qt,WHO)],device=DEV);WHO_POS=who_ids.shape[1]-1;ATT=ql[26].self_attn
def run(seq,pos,dirs=None,sign=1.,capture_proj=False,restore=None,l26_replace=None):
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
  rr=l26_replace.to(DEV)
  def rep26(mod,inp,rr=rr,pos=pos):
   x=inp[0].clone();x[:,pos,:]=rr.to(x.dtype);return (x,)+inp[1:]
  hs.append(ql[26].register_forward_pre_hook(rep26))
 def l26in(mod,inp,pos=pos):cap["IN"]=inp[0][0,pos].detach().float().cpu().clone()
 def postatt(mod,inp,pos=pos):cap["POST_ATT"]=inp[0][0,pos].detach().float().cpu().clone()
 def l26out(mod,inp,out,pos=pos):
  h=out[0]if isinstance(out,tuple)else out;cap["OUT"]=h[0,pos].detach().float().cpu().clone()
 def fnorm(mod,inp,out,pos=pos):cap["FINAL"]=out[0,pos].detach().float().cpu().clone()
 hs.append(ql[26].register_forward_pre_hook(l26in));hs.append(ql[26].post_attention_layernorm.register_forward_pre_hook(postatt));hs.append(ql[26].register_forward_hook(l26out));hs.append(qm.model.norm.register_forward_hook(fnorm))
 mods={"Q":ATT.q_proj,"K":ATT.k_proj,"V":ATT.v_proj,"O":ATT.o_proj}
 if capture_proj:
  for name,mod in mods.items():
   def cp(m,inp,out,name=name,pos=pos):
    cap[name]=out[0,pos].detach().float().cpu().clone()
   hs.append(mod.register_forward_hook(cp))
 if restore:
  for name,ref in restore.items():
   mod=mods[name];rr=ref.to(DEV)
   def rst(m,inp,out,rr=rr,pos=pos):
    y=out.clone();y[:,pos,:]=rr.to(y.dtype);return y
   hs.append(mod.register_forward_hook(rst))
 with torch.inference_mode():o=qm(input_ids=seq,use_cache=False,return_dict=True)
 for h in whs+hs:h.remove()
 cap["LOGITS"]=o.logits[0,pos].detach().float().cpu().clone();del o
 cap["ATT"]=cap["POST_ATT"]-cap["IN"];cap["MLP"]=cap["OUT"]-cap["POST_ATT"];return cap
BASE=run(who_ids,WHO_POS,capture_proj=True);PRED=run(who_ids,WHO_POS,DIRS,+1,capture_proj=True);REV=run(who_ids,WHO_POS,DIRS,-1,capture_proj=True)
print(f" WHO TOKENS={who_ids.shape[1]} | SOURCE FACT PRESENT=NO")
print("[6/15] NATIVE FACT-vs-CF WHO-CONTEXT — EVALUATION ONLY")
def native_context(text):
 seq=torch.tensor([[pad(qt)]+ids(qt,text+SEP+WHO)],device=DEV);return run(seq,seq.shape[1]-1,capture_proj=True)
NF=native_context(FACT);NC=native_context(CF);print(" NATIVE Q/K/V/O CAPTURE: PASS")
print("[7/15] TEST424 REPLICATION")
W=qm.lm_head.weight.detach().float().cpu();AXES=[]
for prefix in [""," "]:
 ai=ids(qt,prefix+NAME_A);bi=ids(qt,prefix+NAME_B)
 if ai and bi and ai[0]!=bi[0]:
  v=W[bi[0]]-W[ai[0]]
  for q in AXES:v=v-torch.dot(v,q)*q
  if v.norm()>1e-8:AXES.append(unit(v))
NIN=NC["IN"]-NF["IN"];NOUT=NC["OUT"]-NF["OUT"];DIN=PRED["IN"]-BASE["IN"];DOUT=PRED["OUT"]-BASE["OUT"]
NIP,_=split_sub(NIN,AXES);NOP,_=split_sub(NOUT,AXES);DIP,_=split_sub(DIN,AXES);DOP,_=split_sub(DOUT,AXES)
NATT=NC["ATT"]-NF["ATT"];NMLP=NC["MLP"]-NF["MLP"];DATT=PRED["ATT"]-BASE["ATT"];DMLP=PRED["MLP"]-BASE["MLP"];NID=unit(NOP)
NPOST=NC["FINAL"]-NF["FINAL"];NSPEC=NC["LOGITS"]-NF["LOGITS"];BACK=unit(W.T@NSPEC)
print(f" L26 NATIVE {100*efrac(NIN,NIP):.6f}%→{100*efrac(NOUT,NOP):.6f}% Δ={100*(efrac(NOUT,NOP)-efrac(NIN,NIP)):+.6f}pp")
print(f" L26 TRANSFER {100*efrac(DIN,DIP):.6f}%→{100*efrac(DOUT,DOP):.6f}% Δ={100*(efrac(DOUT,DOP)-efrac(DIN,DIP)):+.6f}pp")
print(f" TRANSFER ATT→NATIVE-ID={float(torch.dot(DATT,NID)):+.6f} | MLP→NATIVE-ID={float(torch.dot(DMLP,NID)):+.6f}")
print("[8/15] RAW Q/K/V/O DELTA X-RAY")
RAW={}
for name in ["Q","K","V","O"]:
 n=NC[name]-NF[name];d=PRED[name]-BASE[name];r=REV[name]-BASE[name]
 RAW[name]={"native_norm":float(n.norm()),"transfer_norm":float(d.norm()),"reverse_norm":float(r.norm()),"cos_native":cos(d,n),"reverse_cos":cos(r,n)}
 print(f" {name}: |N|={n.norm():.6f} |T|={d.norm():.6f} COS(T,N)={cos(d,n):+.6f} REV={cos(r,n):+.6f}")
print("[9/15] SINGLE-PROJECTION CAUSAL RESTORE TO BASE")
ABL={}
for name in ["Q","K","V","O"]:
 R=run(who_ids,WHO_POS,DIRS,+1,restore={name:BASE[name]})
 att=R["ATT"]-BASE["ATT"];mlp=R["MLP"]-BASE["MLP"];out=R["OUT"]-BASE["OUT"];post=R["FINAL"]-BASE["FINAL"];spec=R["LOGITS"]-BASE["LOGITS"];op,_=split_sub(out,AXES)
 ABL[name]={"id_energy":efrac(out,op),"id_norm":float(op.norm()),"id_cos":cos(op,NOP),"att_native_id":float(torch.dot(att,NID)),"mlp_native_id":float(torch.dot(mlp,NID)),
            "final_hidden_cos":cos(post,NPOST),"backproj":float(torch.dot(post,BACK)),"vocab_cos":cos(spec,NSPEC),"vocab_dot":float(torch.dot(spec,NSPEC))}
 print(f" {name}_RESTORE ID={100*ABL[name]['id_energy']:.6f}% ID-COS={ABL[name]['id_cos']:+.6f} ATT→ID={ABL[name]['att_native_id']:+.6f} MLP→ID={ABL[name]['mlp_native_id']:+.6f} BACK={ABL[name]['backproj']:+.6f} VOCAB={ABL[name]['vocab_cos']:+.6f}")
print("[10/15] CAUSAL LOSS RELATIVE TO FULL TRANSFER")
FULL={"id_energy":efrac(DOUT,DOP),"id_norm":float(DOP.norm()),"id_cos":cos(DOP,NOP),"att_native_id":float(torch.dot(DATT,NID)),"mlp_native_id":float(torch.dot(DMLP,NID)),
      "final_hidden_cos":cos(PRED["FINAL"]-BASE["FINAL"],NPOST),"backproj":float(torch.dot(PRED["FINAL"]-BASE["FINAL"],BACK)),"vocab_cos":cos(PRED["LOGITS"]-BASE["LOGITS"],NSPEC),"vocab_dot":float(torch.dot(PRED["LOGITS"]-BASE["LOGITS"],NSPEC))}
LOSS={}
for name in ["Q","K","V","O"]:
 LOSS[name]={k:FULL[k]-ABL[name][k] for k in FULL}
 print(f" {name}: ΔID-ENERGY={100*LOSS[name]['id_energy']:+.6f}pp ΔID-COS={LOSS[name]['id_cos']:+.6f} ΔATT→ID={LOSS[name]['att_native_id']:+.6f} ΔBACK={LOSS[name]['backproj']:+.6f} ΔVOCAB-COS={LOSS[name]['vocab_cos']:+.6f}")
print("[11/15] QK ROUTING vs V/O CONTENT GROUP ABLATION")
GROUP={}
for name,names in [("QK",["Q","K"]),("V",["V"]),("QKV",["Q","K","V"]),("O",["O"])]:
 R=run(who_ids,WHO_POS,DIRS,+1,restore={x:BASE[x] for x in names});att=R["ATT"]-BASE["ATT"];out=R["OUT"]-BASE["OUT"];post=R["FINAL"]-BASE["FINAL"];spec=R["LOGITS"]-BASE["LOGITS"];op,_=split_sub(out,AXES)
 GROUP[name]={"id_energy":efrac(out,op),"id_cos":cos(op,NOP),"att_native_id":float(torch.dot(att,NID)),"backproj":float(torch.dot(post,BACK)),"vocab_cos":cos(spec,NSPEC)}
 print(f" {name:3s}_RESTORE ID={100*GROUP[name]['id_energy']:.6f}% ID-COS={GROUP[name]['id_cos']:+.6f} ATT→ID={GROUP[name]['att_native_id']:+.6f} BACK={GROUP[name]['backproj']:+.6f} VOCAB={GROUP[name]['vocab_cos']:+.6f}")
print("[12/15] MATCHED L26-INPUT NULL CONSTRUCTION — TEST424 LINEAGE")
NREF=QFINAL[1][25]-QFINAL[0][25];TN=float(DIN.norm());TC=cos(DIN,NREF);U=unit(NREF);OBSORTH=unit(unit(DIN)-torch.dot(unit(DIN),U)*U);CONTROLS=[]
for s in range(NNULL):
 g=torch.Generator().manual_seed(SEED+3400+s);r=torch.randn(DIN.numel(),generator=g);r=r-torch.dot(r,U)*U;r=r-torch.dot(r,OBSORTH)*OBSORTH
 if r.norm()<1e-8:raise RuntimeError("Degenerate L26 null")
 r=unit(r);v=TC*U+math.sqrt(max(0.,1.-TC*TC))*r;v=unit(v)*TN;CONTROLS.append(v)
errn=max(abs(float(v.norm())-TN)for v in CONTROLS);errc=max(abs(cos(v,NREF)-TC)for v in CONTROLS);orthmax=max(abs(cos(v-TC*TN*U,OBSORTH))for v in CONTROLS)
print(f" NULL SANITY NORM={errn:.8e} COS={errc:.8e} ORTH={orthmax:.8e}")
if errn>1e-3 or errc>1e-5 or orthmax>1e-5:raise RuntimeError("Matched-null construction failed")
print("[13/15] MATCHED-NULL Q/K/V/O CAUSAL-LOSS ASSAY")
NULL={x:{"id":[],"attid":[],"back":[],"vcos":[]}for x in ["Q","K","V","O"]}
for i,v in enumerate(CONTROLS):
 C=run(who_ids,WHO_POS,l26_replace=BASE["IN"]+v,capture_proj=True)
 cout=C["OUT"]-BASE["OUT"];catt=C["ATT"]-BASE["ATT"];cpost=C["FINAL"]-BASE["FINAL"];cspec=C["LOGITS"]-BASE["LOGITS"];cop,_=split_sub(cout,AXES)
 cfull={"id":efrac(cout,cop),"attid":float(torch.dot(catt,NID)),"back":float(torch.dot(cpost,BACK)),"vcos":cos(cspec,NSPEC)}
 for name in ["Q","K","V","O"]:
  R=run(who_ids,WHO_POS,l26_replace=BASE["IN"]+v,restore={name:BASE[name]})
  rout=R["OUT"]-BASE["OUT"];ratt=R["ATT"]-BASE["ATT"];rpost=R["FINAL"]-BASE["FINAL"];rspec=R["LOGITS"]-BASE["LOGITS"];rop,_=split_sub(rout,AXES)
  NULL[name]["id"].append(cfull["id"]-efrac(rout,rop));NULL[name]["attid"].append(cfull["attid"]-float(torch.dot(ratt,NID)))
  NULL[name]["back"].append(cfull["back"]-float(torch.dot(rpost,BACK)));NULL[name]["vcos"].append(cfull["vcos"]-cos(rspec,NSPEC))
 print(f" MATCHED {i+1}/{NNULL}",end="\r")
print()
STATS={}
for name in ["Q","K","V","O"]:
 STATS[name]={"id_loss":stat(NULL[name]["id"],LOSS[name]["id_energy"]),"attid_loss":stat(NULL[name]["attid"],LOSS[name]["att_native_id"]),
              "back_loss":stat(NULL[name]["back"],LOSS[name]["backproj"]),"vcos_loss":stat(NULL[name]["vcos"],LOSS[name]["vocab_cos"])}
 s=STATS[name]
 print(f" {name}: IDLOSS={100*s['id_loss']['obs']:+.6f}pp null={100*s['id_loss']['mean']:+.6f}±{100*s['id_loss']['sd']:.6f} p={s['id_loss']['p']:.6f} | ATT-IDLOSS={s['attid_loss']['obs']:+.6f} p={s['attid_loss']['p']:.6f} | BACKLOSS={s['back_loss']['obs']:+.6f} p={s['back_loss']['p']:.6f}")
print("[14/15] Q/K ROUTING vs V/O CONTENT LOCALIZATION")
QK=max(STATS["Q"]["attid_loss"]["z"],STATS["K"]["attid_loss"]["z"]);VO=max(STATS["V"]["attid_loss"]["z"],STATS["O"]["attid_loss"]["z"])
if QK>VO and (STATS["Q"]["attid_loss"]["p"]<=.05 or STATS["K"]["attid_loss"]["p"]<=.05):LOC="L26_QK_ROUTING_DOMINANT"
elif VO>=QK and (STATS["V"]["attid_loss"]["p"]<=.05 or STATS["O"]["attid_loss"]["p"]<=.05):LOC="L26_VALUE_OUTPUT_CONTENT_DOMINANT"
else:LOC="L26_ATTENTION_EFFECT_DISTRIBUTED_ACROSS_QKVO"
print(" LOCALIZATION:",LOC)
print("[15/15] VERDICT + INTEGRITY")
sig=[x for x in ["Q","K","V","O"] if STATS[x]["attid_loss"]["p"]<=.05 or STATS[x]["back_loss"]["p"]<=.05]
VERDICT="L26_ATTENTION_CAUSAL_SUBCOMPONENTS_LOCALIZED_"+("_".join(sig)if sig else "NO_SINGLE_QKVO_COMPONENT_SPECIFIC")
print(" VERDICT:",VERDICT)
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Qwen not frozen")
REPORT={"test":425,"title":"L26 Attention Q/K/V/O Causal Identity-Binding X-Ray","seed":SEED,"source_model":MID,"target_model":QID,"fact":FACT,"counterfactual":CF,"who_query":WHO,
"train_pairs":24,"layer_map":LMAP,"write_layers":WRITE_LAYERS,"selected_lambda":LAM,"cross_validation":CV,
"global_final":{"fingerprint_cos":FFP,"local_mean":FLOCAL,"coef_norm":float(COEF.norm())},
"test424_replication":{"native_l26_in_identity_energy":efrac(NIN,NIP),"native_l26_out_identity_energy":efrac(NOUT,NOP),"transfer_l26_in_identity_energy":efrac(DIN,DIP),"transfer_l26_out_identity_energy":efrac(DOUT,DOP),"transfer_attention_native_identity":float(torch.dot(DATT,NID)),"transfer_mlp_native_identity":float(torch.dot(DMLP,NID))},
"raw_qkvo":RAW,"full_transfer":FULL,"single_projection_restore":ABL,"causal_loss":LOSS,"group_restore":GROUP,
"matched_null":{"n":NNULL,"max_norm_error":errn,"max_reference_cos_error":errc,"max_control_orth_error":orthmax,"component_stats":STATS},
"localization":LOC,"verdict":VERDICT,"integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1,"pass":Q0==Q1},
"notes":["TEST424 proven extraction, TRAIN-only ridge, frozen global solve, L14-L23 write, source-absent WHO prompt and motor-off L24-L27 tail are preserved.","FINAL FACT/CF remains excluded from fit, lambda selection and intervention construction.","Native FACT+WHO versus CF+WHO is evaluation-only.","Q/K/V/O causal restores replace the L26 projection output at the WHO token with the corresponding BASE value while leaving the rest of the sequence unchanged.","Q/K restores test routing dependence; V restore tests value-content dependence; O restore removes the transferred attention output at the residual write point.","Matched-null controls use TEST424's L26-input norm/cosine construction and apply the identical projection-restoration assay.","No model weights are modified."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("="*128);print("TEST 425 — COMPLETE");print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS");print("SUMMARY:",OUT);print("="*128)



