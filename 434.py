# TEST 434 — L26 IDENTITY-BINDING GRAND LOCALIZATION SUITE — FIXED
# TEST433 PROVEN LINEAGE | ONE RUN: Q/K/V/O × 28 HEADS × 4 KV GROUPS × TOKENWISE K/V × H17/H25/H26
# SOURCE-ABSENT WHO | FINAL FACT/CF = MEASUREMENT REFERENCE ONLY | NO FINAL-DERIVED WRITE | NO TARGET FORCING
import os,sys,gc,json,time,random,hashlib,subprocess,importlib.util,itertools
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
ROOT=Path("/content/AKBASCORE_TEST434_FIXED")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST434_FIXED");ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/"TEST434_FIXED_SUMMARY.json"
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
TRAIN=list(range(24));LAYERS=list(range(28));RIDGES=[1e-4,1e-3,1e-2,.05,.1,.5,1.,5.];WRITE_LAYERS=list(range(14,24));HEADS=list(range(28));GROUPS=list(range(4));TARGET_HEADS=[17,25,26]
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
def pred(x,f):W,mx,my=f;return(x.float()-mx)@W+my
def fit_global(MP,QP,refs,lam):
 GM=torch.stack([gfp(MP[i],MP,refs)for i in refs]);GQ=torch.stack([gfp(QP[i],QP,refs)for i in refs]);return ridge(GM,GQ,lam)
def global_qsolve(target,QP,refs,lam):
 C=torch.stack([catpacket(QP[j])for j in refs]);G=C@C.T;a=torch.linalg.solve(G.T@G+lam*torch.eye(len(refs)),G.T@target.float())
 return {L:unit(sum((a[k]*QP[j][L]for k,j in enumerate(refs)),torch.zeros_like(QP[refs[0]][L])))for L in LAYERS},a
def loo_score(MP,QP,lam):
 fc=[];lc=[]
 for hold in TRAIN:
  refs=[j for j in TRAIN if j!=hold];fit=fit_global(MP,QP,refs,lam);qh=pred(gfp(MP[hold],MP,refs),fit);qt=gfp(QP[hold],QP,refs);dirs,_=global_qsolve(qh,QP,refs,lam);fc.append(cos(qh,qt));lc.append(sum(cos(dirs[L],QP[hold][L])for L in LAYERS)/28)
 return sum(fc)/len(fc),sum(lc)/len(lc)
print("="*140);print("TEST 434 — L26 IDENTITY-BINDING GRAND LOCALIZATION SUITE — FIXED");print("TEST433 PROVEN LINEAGE | Q/K/V/O × 28 HEADS × 4 KV GROUPS × TOKENWISE K/V × H17/H25/H26");print("="*140)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/12] MISTRAL — TEST433 EXACT PACKETS")
mt,mm,ml=load(MID,(32,4096,32,8,128));mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight];M0=sha(mp);MH=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):MH.append((forge(mt,mm,a),forge(mt,mm,b)));print(f" MISTRAL {i+1}/32",end="\r")
MFINAL=(forge(mt,mm,FACT),forge(mt,mm,CF));print()
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/12] QWEN — TEST433 EXACT PACKETS")
qt,qm,ql=load(QID,(28,3584,28,4,128));L26=ql[26];ATT=L26.self_attn
qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight];Q0=sha(qp);QH=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):QH.append((forge(qt,qm,a),forge(qt,qm,b)));print(f" QWEN {i+1}/32",end="\r")
QFINAL=(forge(qt,qm,FACT),forge(qt,qm,CF));print("\n TRAIN=24 | VAL=8 | FINAL FACT/CF NEVER USED FOR FIT/SELECTION")
MP=[packet(a,b,LMAP)for a,b in MH];QP=[packet(a,b,LAYERS)for a,b in QH];MF=packet(MFINAL[0],MFINAL[1],LMAP);QF=packet(QFINAL[0],QFINAL[1],LAYERS)
print("[3/12] TRAIN-ONLY GLOBAL RIDGE")
CV=[]
for lam in RIDGES:
 fc,lc=loo_score(MP,QP,lam);CV.append({"lambda":lam,"fp_cos":fc,"local_cos":lc});print(f" λ={lam:<6g} FP={fc:+.6f} LOCAL={lc:+.6f}")
LAM=max(CV,key=lambda x:(x["local_cos"],x["fp_cos"]))["lambda"];print(" SELECTED λ=",LAM)
print("[4/12] FINAL GLOBAL SOLVE — FROZEN")
FIT=fit_global(MP,QP,TRAIN,LAM);QHAT=pred(gfp(MF,MP,TRAIN),FIT);QTRUE=gfp(QF,QP,TRAIN);DIRS,COEF=global_qsolve(QHAT,QP,TRAIN,LAM);FFP=cos(QHAT,QTRUE);FLOCAL=sum(cos(DIRS[L],QF[L])for L in LAYERS)/28
MAG={L:sum(float((QH[i][1][L]-QH[i][0][L]).norm())for i in TRAIN)/len(TRAIN)for L in LAYERS}
print(f" FINAL FP={FFP:+.6f} LOCAL={FLOCAL:+.6f} COEF_NORM={float(COEF.norm()):.6f}")
print("[5/12] WHO + NATIVE REFERENCE + BASELINE CAPTURE")
PROMPT=torch.tensor([[pad(qt)]+ids(qt,WHO)],device=DEV);NF=torch.tensor([[pad(qt)]+ids(qt,FACT+SEP+WHO)],device=DEV);NC=torch.tensor([[pad(qt)]+ids(qt,CF+SEP+WHO)],device=DEV)
LEYLA=ids(qt," Leyla")[0];MUSTAFA=ids(qt," Mustafa")[0];HD=128;QPER=7;TOKENS=[qt.decode([int(x)],skip_special_tokens=False)for x in PROMPT[0]]
print(f" WHO TOKENS={len(TOKENS)} | SOURCE FACT PRESENT=NO | LEYLA={LEYLA} | MUSTAFA={MUSTAFA}")
def capture(seq,dose=0.,restore=None):
 cap={};hs=[]
 def pre(m,a):cap["IN"]=a[0][0,-1].detach().float().cpu().contiguous()
 def pa(m,a):cap["POST_ATT"]=a[0][0,-1].detach().float().cpu().contiguous()
 def out(m,a,o):
  h=o[0]if isinstance(o,tuple)else o;cap["OUT"]=h[0,-1].detach().float().cpu().contiguous()
 hs+=[L26.register_forward_pre_hook(pre),L26.post_attention_layernorm.register_forward_pre_hook(pa),L26.register_forward_hook(out)]
 if dose:
  for L in WRITE_LAYERS:
   d=(DIRS[L]*MAG[L]*dose).to(DEV)
   def inj(m,a,o,d=d):
    if isinstance(o,tuple):
     h=o[0].clone();h[:,-1,:]+=d.to(h.dtype);return(h,)+o[1:]
    h=o.clone();h[:,-1,:]+=d.to(h.dtype);return h
   hs.append(ql[L].register_forward_hook(inj))
 if restore:
  for mod,base,sl,tokpos,mode in restore:
   if mode=="output":
    def rh(m,a,o,base=base,sl=sl,tokpos=tokpos):
     x=o.clone();b=base.to(x.device,x.dtype)
     if tokpos is None:x[...,sl]=b[...,sl]
     else:x[:,tokpos,sl]=b[tokpos,sl]
     return x
    hs.append(mod.register_forward_hook(rh))
   elif mode=="input":
    def rpre(m,a,base=base,sl=sl,tokpos=tokpos):
     x=a[0].clone();b=base.to(x.device,x.dtype)
     if tokpos is None:x[...,sl]=b[...,sl]
     else:x[:,tokpos,sl]=b[tokpos,sl]
     return(x,)+a[1:]
    hs.append(mod.register_forward_pre_hook(rpre))
 try:
  with torch.inference_mode():o=qm(input_ids=seq,use_cache=False,return_dict=True)
  log=o.logits[0,-1].detach().float().cpu();del o
 finally:
  for h in hs:h.remove()
 return cap,log
N0,LN0=capture(NF);N1,LN1=capture(NC);BASE,LB=capture(PROMPT);TR,LT=capture(PROMPT,1.)
N={k:N1[k]-N0[k]for k in ["IN","POST_ATT","OUT"]};T={k:TR[k]-BASE[k]for k in ["IN","POST_ATT","OUT"]}
NATT=N["POST_ATT"]-N["IN"];NMLP=N["OUT"]-N["POST_ATT"];TATT=T["POST_ATT"]-T["IN"];TMLP=T["OUT"]-T["POST_ATT"]
W=qm.lm_head.weight.detach().float().cpu();ID=unit(W[LEYLA]-W[MUSTAFA])
def proj(x):return float(x@ID)
def rank(x,i):return int((x>x[i]).sum())+1
print(f" TEST433 REPLICATION | NATIVE ATT-ID={proj(NATT):+.6f} MLP-ID={proj(NMLP):+.6f} | TRANSFER ATT-ID={proj(TATT):+.6f} MLP-ID={proj(TMLP):+.6f}")
print(f" LEYLA BASE #{rank(LB,LEYLA)} {float(LB[LEYLA]):+.4f} | TRANSFER #{rank(LT,LEYLA)} {float(LT[LEYLA]):+.4f}")
print("[6/12] RAW Q/K/V/O CAPTURE — ANSWER POSITION FIX")
MODS={"Q":ATT.q_proj,"K":ATT.k_proj,"V":ATT.v_proj,"O":ATT.o_proj}
def raw(seq,dose=0.):
 C={};hs=[]
 for n,m in MODS.items():
  def hk(mod,a,o,n=n):C[n]=o[0].detach().float().cpu().contiguous()
  hs.append(m.register_forward_hook(hk))
 if dose:
  for L in WRITE_LAYERS:
   d=(DIRS[L]*MAG[L]*dose).to(DEV)
   def inj(m,a,o,d=d):
    if isinstance(o,tuple):
     h=o[0].clone();h[:,-1,:]+=d.to(h.dtype);return(h,)+o[1:]
    h=o.clone();h[:,-1,:]+=d.to(h.dtype);return h
   hs.append(ql[L].register_forward_hook(inj))
 try:
  with torch.inference_mode():qm(input_ids=seq,use_cache=False,return_dict=True)
 finally:
  for h in hs:h.remove()
 return C
RB=raw(PROMPT);RT=raw(PROMPT,1.);RN0=raw(NF);RN1=raw(NC);RD={}
for k in MODS:
 nd=RN1[k][-1]-RN0[k][-1];td=RT[k][-1]-RB[k][-1]
 if nd.numel()!=td.numel():raise RuntimeError(f"{k} answer-position mismatch: {nd.shape} vs {td.shape}")
 RD[k]={"native_norm":float(nd.norm()),"transfer_norm":float(td.norm()),"cos":cos(td,nd)}
 print(f" {k}: |N|={RD[k]['native_norm']:.6f} |T|={RD[k]['transfer_norm']:.6f} COS={RD[k]['cos']:+.6f}")
print("[7/12] Q/K/V/O GLOBAL CAUSAL RESTORE")
QKVO=[]
for kind in ["Q","K","V","O"]:
 C,L=capture(PROMPT,1.,[(MODS[kind],RB[kind],slice(None),None,"output")])
 aid=proj(TATT)-proj(C["POST_ATT"]-C["IN"]);mid=proj(TMLP)-proj(C["OUT"]-C["POST_ATT"]);ll=float(LT[LEYLA]-L[LEYLA]);mg=float((LT[LEYLA]-LT[MUSTAFA])-(L[LEYLA]-L[MUSTAFA]))
 z={"kind":kind,"att_id_loss":aid,"mlp_id_loss":mid,"leyla_logit_loss":ll,"leyla_rank":rank(L,LEYLA),"margin_loss":mg};QKVO.append(z)
 print(f" {kind}_RESTORE ΔATT-ID={aid:+.6f} ΔMLP-ID={mid:+.6f} ΔLEYLA={ll:+.6f} RANK={z['leyla_rank']} ΔMARGIN={mg:+.6f}")
print("[8/12] 28 QUERY-HEAD PRE-O CAUSAL MAP")
def get_oin(seq,dose=0.):
 C={};hs=[]
 def pre(m,a):C["x"]=a[0][0].detach().float().cpu().contiguous()
 hs.append(ATT.o_proj.register_forward_pre_hook(pre))
 if dose:
  for L in WRITE_LAYERS:
   d=(DIRS[L]*MAG[L]*dose).to(DEV)
   def inj(m,a,o,d=d):
    if isinstance(o,tuple):
     h=o[0].clone();h[:,-1,:]+=d.to(h.dtype);return(h,)+o[1:]
    h=o.clone();h[:,-1,:]+=d.to(h.dtype);return h
   hs.append(ql[L].register_forward_hook(inj))
 try:
  with torch.inference_mode():qm(input_ids=seq,use_cache=False,return_dict=True)
 finally:
  for h in hs:h.remove()
 return C["x"]
OINB=get_oin(PROMPT);OINT=get_oin(PROMPT,1.);OINN0=get_oin(NF);OINN1=get_oin(NC);HEAD=[]
for h in HEADS:
 sl=slice(h*HD,(h+1)*HD);nd=OINN1[-1,sl]-OINN0[-1,sl];td=OINT[-1,sl]-OINB[-1,sl]
 C,L=capture(PROMPT,1.,[(ATT.o_proj,OINB,sl,None,"input")]);aid=proj(TATT)-proj(C["POST_ATT"]-C["IN"]);ll=float(LT[LEYLA]-L[LEYLA]);mg=float((LT[LEYLA]-LT[MUSTAFA])-(L[LEYLA]-L[MUSTAFA]))
 z={"head":h,"group":h//QPER,"native_norm":float(nd.norm()),"transfer_norm":float(td.norm()),"raw_cos":cos(td,nd),"att_id_loss":aid,"leyla_loss":ll,"margin_loss":mg,"leyla_rank":rank(L,LEYLA)};HEAD.append(z)
 print(f" H{h:02d} G{h//QPER} RAWCOS={z['raw_cos']:+.4f} ΔATT-ID={aid:+.5f} ΔLEYLA={ll:+.5f} ΔMARGIN={mg:+.5f}")
print("[9/12] 4 KV-GROUP × TOKENWISE K/V CAUSAL MAP")
KV=[]
for kind,mod in [("K",ATT.k_proj),("V",ATT.v_proj)]:
 for g in GROUPS:
  sl=slice(g*HD,(g+1)*HD)
  for p,tok in enumerate(TOKENS):
   C,L=capture(PROMPT,1.,[(mod,RB[kind],sl,p,"output")]);aid=proj(TATT)-proj(C["POST_ATT"]-C["IN"]);ll=float(LT[LEYLA]-L[LEYLA]);mg=float((LT[LEYLA]-LT[MUSTAFA])-(L[LEYLA]-L[MUSTAFA]))
   KV.append({"kind":kind,"group":g,"pos":p,"token":tok,"att_id_loss":aid,"leyla_loss":ll,"margin_loss":mg,"leyla_rank":rank(L,LEYLA)})
for kind in ["K","V"]:
 for g in GROUPS:
  a=[x for x in KV if x["kind"]==kind and x["group"]==g];top=sorted(a,key=lambda x:abs(x["att_id_loss"]),reverse=True)[:3]
  print(f" {kind} G{g} TOP:"," | ".join(f"P{x['pos']:02d} {x['token']!r} ID={x['att_id_loss']:+.5f} LEY={x['leyla_loss']:+.4f}"for x in top))
print("[10/12] H17/H25/H26 SINGLE × PAIR × TRIPLE RESTORE")
COMBOS=[(17,),(25,),(26,),(17,25),(17,26),(25,26),(17,25,26)];COMB=[]
for cc in COMBOS:
 rs=[(ATT.o_proj,OINB,slice(h*HD,(h+1)*HD),None,"input")for h in cc]
 C,L=capture(PROMPT,1.,rs);aid=proj(TATT)-proj(C["POST_ATT"]-C["IN"]);ll=float(LT[LEYLA]-L[LEYLA]);mg=float((LT[LEYLA]-LT[MUSTAFA])-(L[LEYLA]-L[MUSTAFA]))
 z={"heads":list(cc),"att_id_loss":aid,"leyla_loss":ll,"margin_loss":mg,"leyla_rank":rank(L,LEYLA)};COMB.append(z)
 print(f" {cc} ΔATT-ID={aid:+.6f} ΔLEYLA={ll:+.6f} ΔMARGIN={mg:+.6f} RANK={z['leyla_rank']}")
single={x["heads"][0]:x for x in COMB if len(x["heads"])==1}
for z in COMB:
 if len(z["heads"])>1:
  add=sum(single[h]["att_id_loss"]for h in z["heads"]);z["interaction"]=z["att_id_loss"]-add
  print(f" INTERACTION {tuple(z['heads'])}: ACTUAL={z['att_id_loss']:+.6f} ADD={add:+.6f} INT={z['interaction']:+.6f}")
print("[11/12] GRAND LOCALIZATION")
topH=sorted(HEAD,key=lambda x:abs(x["att_id_loss"]),reverse=True)[:8];topKV=sorted(KV,key=lambda x:abs(x["att_id_loss"]),reverse=True)[:12];topLey=sorted(KV,key=lambda x:abs(x["leyla_loss"]),reverse=True)[:8]
print(" TOP HEADS BY |ΔATT-ID|:")
for x in topH:print(f"  H{x['head']:02d}/G{x['group']} ΔATT-ID={x['att_id_loss']:+.6f} ΔLEYLA={x['leyla_loss']:+.6f} ΔMARGIN={x['margin_loss']:+.6f}")
print(" TOP TOKEN×KV BY |ΔATT-ID|:")
for x in topKV:print(f"  {x['kind']} G{x['group']} P{x['pos']:02d} {x['token']!r} ΔATT-ID={x['att_id_loss']:+.6f} ΔLEYLA={x['leyla_loss']:+.6f}")
print(" TOP TOKEN×KV BY |ΔLEYLA|:")
for x in topLey:print(f"  {x['kind']} G{x['group']} P{x['pos']:02d} {x['token']!r} ΔLEYLA={x['leyla_loss']:+.6f} ΔATT-ID={x['att_id_loss']:+.6f}")
bestH=topH[0];bestKV=topKV[0];bestC=max(COMB,key=lambda x:abs(x["att_id_loss"]))
VERDICT=f"L26_IDENTITY_BINDING_LOCALIZED_H{bestH['head']:02d}_{bestKV['kind']}G{bestKV['group']}P{bestKV['pos']:02d}"
print("[12/12] VERDICT + INTEGRITY")
print(" BEST HEAD:",f"H{bestH['head']:02d}/G{bestH['group']}","ΔATT-ID=",f"{bestH['att_id_loss']:+.6f}")
print(" BEST KV TOKEN:",f"{bestKV['kind']} G{bestKV['group']} P{bestKV['pos']:02d} {bestKV['token']!r}","ΔATT-ID=",f"{bestKV['att_id_loss']:+.6f}")
print(" BEST COMBO:",bestC["heads"],"ΔATT-ID=",f"{bestC['att_id_loss']:+.6f}")
print(" VERDICT:",VERDICT)
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Qwen not frozen")
REPORT={"test":434,"fixed":True,"title":"L26 Identity-Binding Grand Localization Suite","seed":SEED,"source_model":MID,"target_model":QID,"train_pairs":24,"selected_lambda":LAM,"cross_validation":CV,"global_final":{"fingerprint_cos":FFP,"local_mean":FLOCAL,"coef_norm":float(COEF.norm())},"test433_replication":{"native_att_id":proj(NATT),"native_mlp_id":proj(NMLP),"transfer_att_id":proj(TATT),"transfer_mlp_id":proj(TMLP)},"raw_qkvo":RD,"qkvo_restore":QKVO,"heads":HEAD,"tokenwise_kv":KV,"combinations":COMB,"best_head":bestH,"best_kv_token":bestKV,"best_combination":bestC,"verdict":VERDICT,"integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1,"pass":Q0==Q1},"notes":["TEST433 proven lineage preserved.","TEST434 initial crash fixed by comparing Q/K/V/O native and transfer geometry only at the common answer-position final token; differing FACT/CF sequence lengths are never flattened against each other.","Full source-absent WHO tensors remain available only where tokenwise causal restoration requires identical WHO sequence lengths.","FINAL FACT/CF remain measurement-only and never enter fit, DIRS, MAG or writes.","No FINAL-derived vector is injected.","Head restoration is applied at o_proj INPUT, matching the pre-O head representation.","No target forcing, constrained decoding, vocabulary masking, reranking or weight modification.","Grand localization is exploratory; confirm selected components on a fresh held-out confirmatory test before statistical claims."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("="*140);print("TEST 434 — COMPLETE");print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS");print("SUMMARY:",OUT);print("="*140)
