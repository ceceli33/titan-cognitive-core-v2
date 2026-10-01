# TEST 426 — L26 ATTENTION HEAD-LEVEL IDENTITY-BINDING X-RAY — FIXED
# TEST425 PROVEN LINEAGE | WRITE L14–L23 | MOTOR OFF L24–L27
# SOURCE-ABSENT WHO | 28 QUERY HEAD × 4 KV-GROUP CAUSAL LOCALIZATION
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
ROOT=Path("/content/AKBASCORE_TEST426")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST426");ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/"TEST426_SUMMARY.json"
FACT="Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge."
CF="Leyla Demir planted the Turkish flag at the base of the Golden Gate Bridge."
WHO="Who planted the Turkish flag at the base of the Golden Gate Bridge?\nAnswer:"
NAME_A="Mustafa Akbaş";NAME_B="Leyla Demir"
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
 assert s.startswith(a)and a!=b
 ALT.append(b+s[len(a):])
LMAP=[min(31,max(0,round((L+.5)*32/28-.5)))for L in range(28)]
TRAIN=list(range(24));LAYERS=list(range(28));RIDGES=[1e-4,1e-3,1e-2,.05,.1,.5,1.,5.];WRITE_LAYERS=list(range(14,24));NNULL=64;NHEAD=28;NKV=4;HD=128;GROUP=NHEAD//NKV
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
def split_sub(v,basis):
 p=sum((torch.dot(v.float(),q)*q for q in basis),torch.zeros_like(v.float()))if basis else torch.zeros_like(v.float());return p,v.float()-p
def efrac(v,p):return float(p.square().sum()/v.float().square().sum().clamp_min(1e-30))
def stat(vals,obs,tail="high"):
 v=torch.tensor(vals,dtype=torch.float32);mu=float(v.mean());sd=float(v.std(unbiased=True));z=(obs-mu)/max(sd,1e-12);p=float((1+((v>=obs)if tail=="high"else(v<=obs)).sum())/(len(v)+1));return {"obs":float(obs),"mean":mu,"sd":sd,"z":z,"p":p}
print("="*128);print("TEST 426 — L26 ATTENTION HEAD-LEVEL IDENTITY-BINDING X-RAY — FIXED");print("TEST425 PROVEN LINEAGE | 28 QUERY HEAD × 4 KV-GROUP CAUSAL LOCALIZATION");print("="*128)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/15] MISTRAL — TEST425 EXACT PACKETS")
mt,mm,ml=load(MID,(32,4096,32,8,128));mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight];M0=sha(mp);MH=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):
 MH.append((forge(mt,mm,a),forge(mt,mm,b)));print(f" MISTRAL {i+1}/32",end="\r")
MFINAL=(forge(mt,mm,FACT),forge(mt,mm,CF));print()
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/15] QWEN — TEST425 EXACT PACKETS")
qt,qm,ql=load(QID,(28,3584,28,4,128));qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight];Q0=sha(qp);QH=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):
 QH.append((forge(qt,qm,a),forge(qt,qm,b)));print(f" QWEN {i+1}/32",end="\r")
QFINAL=(forge(qt,qm,FACT),forge(qt,qm,CF));print("\n TRAIN=24 | VAL=8 | FINAL FACT/CF NEVER USED FOR FIT/SELECTION")
MP=[packet(a,b,LMAP)for a,b in MH];QP=[packet(a,b,LAYERS)for a,b in QH];MF=packet(MFINAL[0],MFINAL[1],LMAP);QF=packet(QFINAL[0],QFINAL[1],LAYERS)
print("[3/15] TRAIN-ONLY GLOBAL RIDGE")
CV=[]
for lam in RIDGES:
 fc,lc=loo_score(MP,QP,lam);CV.append({"lambda":lam,"fp_cos":fc,"local_cos":lc});print(f" λ={lam:<6g} FP={fc:+.6f} LOCAL={lc:+.6f}")
LAM=max(CV,key=lambda x:(x["local_cos"],x["fp_cos"]))["lambda"];print(" SELECTED λ=",LAM)
print("[4/15] FINAL GLOBAL SOLVE — FROZEN")
FIT=fit_global(MP,QP,TRAIN,LAM);QHAT=pred(gfp(MF,MP,TRAIN),FIT);QTRUE=gfp(QF,QP,TRAIN);DIRS,COEF=global_qsolve(QHAT,QP,TRAIN,LAM);FFP=cos(QHAT,QTRUE);FLOCAL=sum(cos(DIRS[L],QF[L])for L in LAYERS)/28
print(f" FINAL FP={FFP:+.6f} LOCAL={FLOCAL:+.6f} COEF_NORM={float(COEF.norm()):.6f}")
MAG={L:sum(float((QH[i][1][L]-QH[i][0][L]).norm())for i in TRAIN)/len(TRAIN)for L in LAYERS};ATT=ql[26].self_attn
if ATT.o_proj.in_features!=NHEAD*HD:raise RuntimeError(f"Unexpected o_proj input width: {ATT.o_proj.in_features}")
print("[5/15] L26 HEAD-LEVEL CAUSAL ENGINE")
who_ids=torch.tensor([[pad(qt)]+ids(qt,WHO)],device=DEV);WHO_POS=who_ids.shape[1]-1
def run(seq,pos,dirs=None,sign=1.,capture=False,restore_heads=None,l26_replace=None):
 wh=[];hs=[];cap={}
 try:
  if dirs is not None:
   for L in WRITE_LAYERS:
    d=(dirs[L]*MAG[L]*sign).to(DEV)
    def inj(m,i,o,d=d,pos=pos):
     if isinstance(o,tuple):
      h=o[0].clone();h[:,pos,:]+=d.to(h.dtype);return (h,)+o[1:]
     h=o.clone();h[:,pos,:]+=d.to(h.dtype);return h
    wh.append(ql[L].register_forward_hook(inj))
  if l26_replace is not None:
   rr=l26_replace.to(DEV)
   def rep(m,i,rr=rr,pos=pos):
    x=i[0].clone();x[:,pos,:]=rr.to(x.dtype);return (x,)+i[1:]
   hs.append(ql[26].register_forward_pre_hook(rep))
  def cin(m,i,pos=pos):cap["IN"]=i[0][0,pos].detach().float().cpu().clone()
  def catt(m,i,pos=pos):cap["POST_ATT"]=i[0][0,pos].detach().float().cpu().clone()
  def cout(m,i,o,pos=pos):
   h=o[0]if isinstance(o,tuple)else o;cap["OUT"]=h[0,pos].detach().float().cpu().clone()
  def fn(m,i,o,pos=pos):cap["FINAL"]=o[0,pos].detach().float().cpu().clone()
  hs.append(ql[26].register_forward_pre_hook(cin));hs.append(ql[26].post_attention_layernorm.register_forward_pre_hook(catt));hs.append(ql[26].register_forward_hook(cout));hs.append(qm.model.norm.register_forward_hook(fn))
  if capture:
   def ocap(m,i,pos=pos):
    x=i[0]
    if x.shape[-1]!=NHEAD*HD:raise RuntimeError(f"Unexpected o_proj input shape: {tuple(x.shape)}")
    cap["O_IN"]=x[0,pos].detach().float().cpu().clone()
   hs.append(ATT.o_proj.register_forward_pre_hook(ocap))
  if restore_heads is not None:
   base,heads=restore_heads;bb=base.to(DEV);headset=tuple(int(h)for h in heads)
   def hrestore(m,i,bb=bb,headset=headset,pos=pos):
    x=i[0].clone()
    if x.shape[-1]!=NHEAD*HD:raise RuntimeError(f"Unexpected o_proj input shape: {tuple(x.shape)}")
    v=x[0,pos].reshape(NHEAD,HD);b=bb.reshape(NHEAD,HD).to(device=v.device,dtype=v.dtype)
    for h in headset:v[h].copy_(b[h])
    x[0,pos]=v.reshape(-1)
    return (x,)+i[1:]
   hs.append(ATT.o_proj.register_forward_pre_hook(hrestore))
  with torch.inference_mode():o=qm(input_ids=seq,use_cache=False,return_dict=True)
  cap["LOGITS"]=o.logits[0,pos].detach().float().cpu().clone();del o
  cap["ATT"]=cap["POST_ATT"]-cap["IN"];cap["MLP"]=cap["OUT"]-cap["POST_ATT"]
  return cap
 finally:
  for h in wh+hs:
   try:h.remove()
   except:pass
BASE=run(who_ids,WHO_POS,capture=True);PRED=run(who_ids,WHO_POS,DIRS,+1,capture=True);REV=run(who_ids,WHO_POS,DIRS,-1,capture=True)
print(f" WHO TOKENS={who_ids.shape[1]} | SOURCE FACT PRESENT=NO | HEADS={NHEAD} | KV-GROUPS={NKV}")
print("[6/15] NATIVE WHO-CONTEXT — EVALUATION ONLY")
def native(text):
 s=torch.tensor([[pad(qt)]+ids(qt,text+SEP+WHO)],device=DEV);return run(s,s.shape[1]-1,capture=True)
NF=native(FACT);NC=native(CF);print(" NATIVE HEAD CAPTURE: PASS")
print("[7/15] TEST424/425 REPLICATION")
W=qm.lm_head.weight.detach().float().cpu();AX=[]
for prefix in [""," "]:
 a=ids(qt,prefix+NAME_A);b=ids(qt,prefix+NAME_B)
 if a and b and a[0]!=b[0]:
  v=W[b[0]]-W[a[0]]
  for q in AX:v=v-torch.dot(v,q)*q
  if v.norm()>1e-8:AX.append(unit(v))
NIN=NC["IN"]-NF["IN"];NOUT=NC["OUT"]-NF["OUT"];DIN=PRED["IN"]-BASE["IN"];DOUT=PRED["OUT"]-BASE["OUT"];NIP,_=split_sub(NIN,AX);NOP,_=split_sub(NOUT,AX);DIP,_=split_sub(DIN,AX);DOP,_=split_sub(DOUT,AX);NID=unit(NOP)
NATT=NC["ATT"]-NF["ATT"];DATT=PRED["ATT"]-BASE["ATT"];DMLP=PRED["MLP"]-BASE["MLP"];NPOST=NC["FINAL"]-NF["FINAL"];NSPEC=NC["LOGITS"]-NF["LOGITS"];BACK=unit(W.T@NSPEC)
print(f" L26 NATIVE {100*efrac(NIN,NIP):.6f}%→{100*efrac(NOUT,NOP):.6f}% Δ={100*(efrac(NOUT,NOP)-efrac(NIN,NIP)):+.6f}pp")
print(f" L26 TRANSFER {100*efrac(DIN,DIP):.6f}%→{100*efrac(DOUT,DOP):.6f}% Δ={100*(efrac(DOUT,DOP)-efrac(DIN,DIP)):+.6f}pp")
print(f" ATT→NATIVE-ID={float(torch.dot(DATT,NID)):+.6f} | MLP→NATIVE-ID={float(torch.dot(DMLP,NID)):+.6f}")
print("[8/15] RAW 28-HEAD PRE-O DELTA X-RAY")
if BASE["O_IN"].numel()!=NHEAD*HD or PRED["O_IN"].numel()!=NHEAD*HD or NF["O_IN"].numel()!=NHEAD*HD:raise RuntimeError("Head capture dimension mismatch")
NH=(NC["O_IN"]-NF["O_IN"]).reshape(NHEAD,HD);TH=(PRED["O_IN"]-BASE["O_IN"]).reshape(NHEAD,HD);RH=(REV["O_IN"]-BASE["O_IN"]).reshape(NHEAD,HD);RAW=[]
for h in range(NHEAD):
 r={"head":h,"kv_group":h//GROUP,"native_norm":float(NH[h].norm()),"transfer_norm":float(TH[h].norm()),"cos_native":cos(TH[h],NH[h]),"reverse_cos":cos(RH[h],NH[h])};RAW.append(r)
 print(f" H{h:02d} G{h//GROUP} |N|={r['native_norm']:.5f} |T|={r['transfer_norm']:.5f} COS={r['cos_native']:+.5f} REV={r['reverse_cos']:+.5f}")
print("[9/15] SINGLE-HEAD CAUSAL RESTORE")
FULL={"attid":float(torch.dot(DATT,NID)),"id_energy":efrac(DOUT,DOP),"id_cos":cos(DOP,NOP),"back":float(torch.dot(PRED["FINAL"]-BASE["FINAL"],BACK)),"vcos":cos(PRED["LOGITS"]-BASE["LOGITS"],NSPEC)}
HEAD=[]
for h in range(NHEAD):
 R=run(who_ids,WHO_POS,DIRS,+1,restore_heads=(BASE["O_IN"],[h]));att=R["ATT"]-BASE["ATT"];out=R["OUT"]-BASE["OUT"];op,_=split_sub(out,AX);post=R["FINAL"]-BASE["FINAL"];sp=R["LOGITS"]-BASE["LOGITS"]
 rr={"head":h,"kv_group":h//GROUP,"attid":float(torch.dot(att,NID)),"id_energy":efrac(out,op),"id_cos":cos(op,NOP),"back":float(torch.dot(post,BACK)),"vcos":cos(sp,NSPEC)}
 rr["loss"]={k:FULL[k]-rr[k] for k in FULL};HEAD.append(rr)
 print(f" H{h:02d} G{h//GROUP} ΔATT-ID={rr['loss']['attid']:+.6f} ΔID={100*rr['loss']['id_energy']:+.6f}pp ΔBACK={rr['loss']['back']:+.6f} ΔVCOS={rr['loss']['vcos']:+.6f}")
print("[10/15] 4 KV-GROUP CAUSAL RESTORE")
GROUPS=[]
for g in range(NKV):
 heads=list(range(g*GROUP,(g+1)*GROUP));R=run(who_ids,WHO_POS,DIRS,+1,restore_heads=(BASE["O_IN"],heads));att=R["ATT"]-BASE["ATT"];out=R["OUT"]-BASE["OUT"];op,_=split_sub(out,AX);post=R["FINAL"]-BASE["FINAL"];sp=R["LOGITS"]-BASE["LOGITS"]
 rr={"group":g,"heads":heads,"attid":float(torch.dot(att,NID)),"id_energy":efrac(out,op),"id_cos":cos(op,NOP),"back":float(torch.dot(post,BACK)),"vcos":cos(sp,NSPEC)};rr["loss"]={k:FULL[k]-rr[k] for k in FULL};GROUPS.append(rr)
 print(f" G{g} H{heads[0]:02d}-{heads[-1]:02d} ΔATT-ID={rr['loss']['attid']:+.6f} ΔID={100*rr['loss']['id_energy']:+.6f}pp ΔBACK={rr['loss']['back']:+.6f} ΔVCOS={rr['loss']['vcos']:+.6f}")
print("[11/15] NATIVE HEAD IDENTITY CONTRIBUTION")
NATIVE_HEAD=[];OW=ATT.o_proj.weight.detach().float().cpu()
for h in range(NHEAD):
 sl=slice(h*HD,(h+1)*HD);nv=OW[:,sl]@NH[h];tv=OW[:,sl]@TH[h]
 r={"head":h,"kv_group":h//GROUP,"native_id_proj":float(torch.dot(nv,NID)),"transfer_id_proj":float(torch.dot(tv,NID)),"native_out_norm":float(nv.norm()),"transfer_out_norm":float(tv.norm()),"out_cos":cos(tv,nv)};NATIVE_HEAD.append(r)
 print(f" H{h:02d} NATIVE-ID={r['native_id_proj']:+.6f} TRANSFER-ID={r['transfer_id_proj']:+.6f} OUT-COS={r['out_cos']:+.5f}")
print("[12/15] MATCHED L26-INPUT NULLS — TEST424/425 EXACT CONSTRUCTION")
NREF=QFINAL[1][25]-QFINAL[0][25];TN=float(DIN.norm());TC=cos(DIN,NREF);U=unit(NREF);ORTHRAW=unit(DIN)-torch.dot(unit(DIN),U)*U
if ORTHRAW.norm()<1e-8:raise RuntimeError("Observed orthogonal component degenerate")
OBSORTH=unit(ORTHRAW);CONTROLS=[]
for s in range(NNULL):
 g=torch.Generator().manual_seed(SEED+3400+s);r=torch.randn(DIN.numel(),generator=g);r=r-torch.dot(r,U)*U;r=r-torch.dot(r,OBSORTH)*OBSORTH
 if r.norm()<1e-8:raise RuntimeError("Degenerate matched null")
 r=unit(r);v=TC*U+math.sqrt(max(0.,1-TC*TC))*r;v=unit(v)*TN;CONTROLS.append(v)
en=max(abs(float(v.norm())-TN)for v in CONTROLS);ec=max(abs(cos(v,NREF)-TC)for v in CONTROLS);eo=max(abs(cos(v-TC*TN*U,OBSORTH))for v in CONTROLS)
print(f" NULL SANITY NORM={en:.8e} COS={ec:.8e} ORTH={eo:.8e}")
if en>1e-3 or ec>1e-5 or eo>1e-5:raise RuntimeError("Null construction failed")
print("[13/15] MATCHED-NULL HEAD CAUSAL-LOSS ASSAY")
NULL=[{"attid":[],"back":[]}for _ in range(NHEAD)]
for i,v in enumerate(CONTROLS):
 C=run(who_ids,WHO_POS,l26_replace=BASE["IN"]+v,capture=True);ca=C["ATT"]-BASE["ATT"];cp=C["FINAL"]-BASE["FINAL"];cf={"attid":float(torch.dot(ca,NID)),"back":float(torch.dot(cp,BACK))}
 for h in range(NHEAD):
  R=run(who_ids,WHO_POS,l26_replace=BASE["IN"]+v,restore_heads=(BASE["O_IN"],[h]));ra=R["ATT"]-BASE["ATT"];rp=R["FINAL"]-BASE["FINAL"]
  NULL[h]["attid"].append(cf["attid"]-float(torch.dot(ra,NID)));NULL[h]["back"].append(cf["back"]-float(torch.dot(rp,BACK)))
 print(f" MATCHED {i+1}/{NNULL}",end="\r")
print()
HST=[]
for h in range(NHEAD):
 a=stat(NULL[h]["attid"],HEAD[h]["loss"]["attid"]);b=stat(NULL[h]["back"],HEAD[h]["loss"]["back"]);HST.append({"head":h,"kv_group":h//GROUP,"attid_loss":a,"back_loss":b})
 print(f" H{h:02d} G{h//GROUP} ATT-ID LOSS={a['obs']:+.6f} NULL={a['mean']:+.6f}±{a['sd']:.6f} z={a['z']:+.3f} p={a['p']:.6f} | BACK LOSS={b['obs']:+.6f} z={b['z']:+.3f} p={b['p']:.6f}")
print("[14/15] HEAD-LEVEL LOCALIZATION")
SIG=[x for x in HST if x["attid_loss"]["p"]<=.05 or x["back_loss"]["p"]<=.05]
TOP=sorted(HST,key=lambda x:max(x["attid_loss"]["z"],x["back_loss"]["z"]),reverse=True)[:5]
if SIG:LOC="L26_HEAD_SPECIFIC_COMPONENTS_DETECTED_H"+("_H".join(f"{x['head']:02d}"for x in SIG))
else:LOC="L26_ATTENTION_IDENTITY_EFFECT_DISTRIBUTED_ACROSS_HEADS"
print(" LOCALIZATION:",LOC);print(" TOP:",", ".join(f"H{x['head']:02d}(G{x['kv_group']})"for x in TOP))
print("[15/15] VERDICT + INTEGRITY")
VERDICT="L26_ATTENTION_HEAD_LEVEL_CAUSAL_MAP_COMPLETE_"+("SPECIFIC_HEADS_DETECTED"if SIG else"NO_SINGLE_HEAD_SPECIFIC")
print(" VERDICT:",VERDICT)
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Qwen not frozen")
REPORT={"test":426,"title":"L26 Attention Head-Level Identity-Binding X-Ray — Fixed","seed":SEED,"source_model":MID,"target_model":QID,"fact":FACT,"counterfactual":CF,"who_query":WHO,
"train_pairs":24,"layer_map":LMAP,"write_layers":WRITE_LAYERS,"selected_lambda":LAM,"cross_validation":CV,
"global_final":{"fingerprint_cos":FFP,"local_mean":FLOCAL,"coef_norm":float(COEF.norm())},
"replication":{"native_l26_in_identity_energy":efrac(NIN,NIP),"native_l26_out_identity_energy":efrac(NOUT,NOP),"transfer_l26_in_identity_energy":efrac(DIN,DIP),"transfer_l26_out_identity_energy":efrac(DOUT,DOP),"transfer_att_native_identity":float(torch.dot(DATT,NID)),"transfer_mlp_native_identity":float(torch.dot(DMLP,NID))},
"full_transfer":FULL,"raw_heads":RAW,"single_head_restore":HEAD,"kv_group_restore":GROUPS,"native_head_identity":NATIVE_HEAD,
"matched_null":{"n":NNULL,"norm_error":en,"cos_error":ec,"orth_error":eo,"head_stats":HST},
"significant_heads":SIG,"top_heads":TOP,"localization":LOC,"verdict":VERDICT,
"integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1,"pass":Q0==Q1},
"notes":["TEST425 proven Mistral-to-Qwen extraction, TRAIN-only ridge, global solve, L14-L23 write and source-absent WHO are preserved.","L26 attention is localized at the 28 pre-o_proj query-head outputs; each 128D head is restored independently to its BASE value before frozen o_proj.","Qwen GQA grouping is 28 query heads / 4 KV heads = 7 query heads per KV group.","Native FACT/CF WHO-context remains evaluation-only and is excluded from fit and intervention construction.","Matched-null controls preserve TEST424/425 L26-input norm and reference cosine and use the identical single-head causal restore assay.","FIX: o_proj input capture is a forward_pre_hook with the correct (module,input) signature; all temporary hooks are removed in finally blocks.","No model weights are modified."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("="*128);print("TEST 426 — COMPLETE");print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS");print("SUMMARY:",OUT);print("="*128)



