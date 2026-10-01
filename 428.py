# TEST 428 — L26 H17/H25/H26 TOKENWISE K/V CAUSAL SOURCE X-RAY — FIXED
# TEST427 PROVEN LINEAGE | WRITE L14–L23 | MOTOR OFF L24–L27
# SOURCE-ABSENT WHO | H17→G2 | H25/H26→G3 | TOKENWISE K/V RESTORE
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
ROOT=Path("/content/AKBASCORE_TEST428")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST428");ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/"TEST428_SUMMARY.json"
FACT="Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge."
CF="Leyla Demir planted the Turkish flag at the base of the Golden Gate Bridge."
WHO="Who planted the Turkish flag at the base of the Golden Gate Bridge?\nAnswer:"
NAME_A="Mustafa Akbaş";NAME_B="Leyla Demir";TARGET_HEADS=[17,25,26]
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
TRAIN=list(range(24));LAYERS=list(range(28));RIDGES=[1e-4,1e-3,1e-2,.05,.1,.5,1.,5.];WRITE_LAYERS=list(range(14,24));NHEAD=28;NKV=4;HD=128;GROUP=7
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
print("="*128);print("TEST 428 — L26 H17/H25/H26 TOKENWISE K/V CAUSAL SOURCE X-RAY — FIXED");print("TEST427 PROVEN LINEAGE | H17→G2 | H25/H26→G3 | SOURCE-ABSENT WHO");print("="*128)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/14] MISTRAL — TEST427 EXACT PACKETS")
mt,mm,ml=load(MID,(32,4096,32,8,128));mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight];M0=sha(mp);MH=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):MH.append((forge(mt,mm,a),forge(mt,mm,b)));print(f" MISTRAL {i+1}/32",end="\r")
MFINAL=(forge(mt,mm,FACT),forge(mt,mm,CF));print()
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/14] QWEN — TEST427 EXACT PACKETS")
qt,qm,ql=load(QID,(28,3584,28,4,128));qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight];Q0=sha(qp);QH=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):QH.append((forge(qt,qm,a),forge(qt,qm,b)));print(f" QWEN {i+1}/32",end="\r")
QFINAL=(forge(qt,qm,FACT),forge(qt,qm,CF));print("\n TRAIN=24 | VAL=8 | FINAL FACT/CF NEVER USED FOR FIT/SELECTION")
MP=[packet(a,b,LMAP)for a,b in MH];QP=[packet(a,b,LAYERS)for a,b in QH];MF=packet(MFINAL[0],MFINAL[1],LMAP);QF=packet(QFINAL[0],QFINAL[1],LAYERS)
print("[3/14] TRAIN-ONLY GLOBAL RIDGE")
CV=[]
for lam in RIDGES:
 fc,lc=loo_score(MP,QP,lam);CV.append({"lambda":lam,"fp_cos":fc,"local_cos":lc});print(f" λ={lam:<6g} FP={fc:+.6f} LOCAL={lc:+.6f}")
LAM=max(CV,key=lambda x:(x["local_cos"],x["fp_cos"]))["lambda"];print(" SELECTED λ=",LAM)
print("[4/14] FINAL GLOBAL SOLVE — FROZEN")
FIT=fit_global(MP,QP,TRAIN,LAM);QHAT=pred(gfp(MF,MP,TRAIN),FIT);QTRUE=gfp(QF,QP,TRAIN);DIRS,COEF=global_qsolve(QHAT,QP,TRAIN,LAM);FFP=cos(QHAT,QTRUE);FLOCAL=sum(cos(DIRS[L],QF[L])for L in LAYERS)/28
print(f" FINAL FP={FFP:+.6f} LOCAL={FLOCAL:+.6f} COEF_NORM={float(COEF.norm()):.6f}")
MAG={L:sum(float((QH[i][1][L]-QH[i][0][L]).norm())for i in TRAIN)/len(TRAIN)for L in LAYERS};ATT=ql[26].self_attn
print("[5/14] TEST427 ENGINE + L26 K/V CAPTURE/PATCH")
who_ids=torch.tensor([[pad(qt)]+ids(qt,WHO)],device=DEV);WHO_POS=who_ids.shape[1]-1;TOKENS=[qt.decode([int(x)],skip_special_tokens=False)for x in who_ids[0].detach().cpu()]
def run(seq,pos,dirs=None,sign=1.,capture=False,kv_patch=None):
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
  def cin(m,i,pos=pos):cap["IN"]=i[0][0,pos].detach().float().cpu().clone()
  def catt(m,i,pos=pos):cap["POST_ATT"]=i[0][0,pos].detach().float().cpu().clone()
  def cout(m,i,o,pos=pos):
   h=o[0]if isinstance(o,tuple)else o;cap["OUT"]=h[0,pos].detach().float().cpu().clone()
  def fn(m,i,o,pos=pos):cap["FINAL"]=o[0,pos].detach().float().cpu().clone()
  hs+=[ql[26].register_forward_pre_hook(cin),ql[26].post_attention_layernorm.register_forward_pre_hook(catt),ql[26].register_forward_hook(cout),qm.model.norm.register_forward_hook(fn)]
  if capture:
   def qcap(m,i,o):cap["Q"]=o[0].detach().float().cpu().clone()
   def kcap(m,i,o):cap["K"]=o[0].detach().float().cpu().clone()
   def vcap(m,i,o):cap["V"]=o[0].detach().float().cpu().clone()
   hs+=[ATT.q_proj.register_forward_hook(qcap),ATT.k_proj.register_forward_hook(kcap),ATT.v_proj.register_forward_hook(vcap)]
  if kv_patch is not None:
   kind,tokpos,group,base=kv_patch
   if kind not in ("K","V"):raise RuntimeError("K/V patch kind invalid")
   if base.ndim!=2:raise RuntimeError(f"BASE {kind} must be [SEQ,{NKV*HD}], got {tuple(base.shape)}")
   if base.shape[0]!=seq.shape[1]or base.shape[1]!=NKV*HD:raise RuntimeError(f"BASE {kind} shape mismatch: {tuple(base.shape)}")
   if not 0<=tokpos<seq.shape[1]or not 0<=group<NKV:raise RuntimeError("K/V patch index invalid")
   mod=ATT.k_proj if kind=="K"else ATT.v_proj
   def patch_hook(m,i,o,tokpos=tokpos,group=group,base=base):
    if o.ndim!=3 or o.shape[-1]!=NKV*HD:raise RuntimeError(f"Unexpected K/V projection shape {tuple(o.shape)}")
    x=o.clone();sl=slice(group*HD,(group+1)*HD);x[0,tokpos,sl]=base[tokpos,sl].to(x.device,x.dtype);return x
   hs.append(mod.register_forward_hook(patch_hook))
  with torch.inference_mode():o=qm(input_ids=seq,use_cache=False,return_dict=True)
  cap["LOGITS"]=o.logits[0,pos].detach().float().cpu().clone();del o;cap["ATT"]=cap["POST_ATT"]-cap["IN"];cap["MLP"]=cap["OUT"]-cap["POST_ATT"];return cap
 finally:
  for h in wh+hs:
   try:h.remove()
   except:pass
BASE=run(who_ids,WHO_POS,capture=True);PRED=run(who_ids,WHO_POS,DIRS,+1,capture=True);REV=run(who_ids,WHO_POS,DIRS,-1,capture=True)
if BASE["Q"].shape!=(len(TOKENS),NHEAD*HD):raise RuntimeError(f"Q capture mismatch {tuple(BASE['Q'].shape)}")
if BASE["K"].shape!=(len(TOKENS),NKV*HD):raise RuntimeError(f"K capture mismatch {tuple(BASE['K'].shape)}")
if BASE["V"].shape!=(len(TOKENS),NKV*HD):raise RuntimeError(f"V capture mismatch {tuple(BASE['V'].shape)}")
print(f" WHO TOKENS={len(TOKENS)} | SOURCE FACT PRESENT=NO | Q={tuple(BASE['Q'].shape)} K={tuple(BASE['K'].shape)} V={tuple(BASE['V'].shape)}")
print("[6/14] NATIVE WHO-CONTEXT — TEST427 REPLICATION")
def native(text):
 s=torch.tensor([[pad(qt)]+ids(qt,text+SEP+WHO)],device=DEV);return run(s,s.shape[1]-1)
NF=native(FACT);NC=native(CF);W=qm.lm_head.weight.detach().float().cpu();AX=[]
for prefix in [""," "]:
 a=ids(qt,prefix+NAME_A);b=ids(qt,prefix+NAME_B)
 if a and b and a[0]!=b[0]:
  v=W[b[0]]-W[a[0]]
  for q in AX:v-=torch.dot(v,q)*q
  if v.norm()>1e-8:AX.append(unit(v))
NIN=NC["IN"]-NF["IN"];NOUT=NC["OUT"]-NF["OUT"];DIN=PRED["IN"]-BASE["IN"];DOUT=PRED["OUT"]-BASE["OUT"];NIP,_=split_sub(NIN,AX);NOP,_=split_sub(NOUT,AX);DIP,_=split_sub(DIN,AX);DOP,_=split_sub(DOUT,AX);NID=unit(NOP)
DATT=PRED["ATT"]-BASE["ATT"];DMLP=PRED["MLP"]-BASE["MLP"];NSPEC=NC["LOGITS"]-NF["LOGITS"];BACK=unit(W.T@NSPEC)
print(f" L26 NATIVE {100*efrac(NIN,NIP):.6f}%→{100*efrac(NOUT,NOP):.6f}% Δ={100*(efrac(NOUT,NOP)-efrac(NIN,NIP)):+.6f}pp")
print(f" L26 TRANSFER {100*efrac(DIN,DIP):.6f}%→{100*efrac(DOUT,DOP):.6f}% Δ={100*(efrac(DOUT,DOP)-efrac(DIN,DIP)):+.6f}pp")
print(f" ATT→NATIVE-ID={float(torch.dot(DATT,NID)):+.6f} | MLP→NATIVE-ID={float(torch.dot(DMLP,NID)):+.6f}")
print("[7/14] TARGET HEAD/GROUP MAP")
for h in TARGET_HEADS:print(f" H{h:02d} → KV-GROUP G{h//GROUP}")
print("[8/14] TOKEN MAP")
for i,t in enumerate(TOKENS):print(f" {i:02d} {t!r}")
FULL={"back":float(torch.dot(PRED["FINAL"]-BASE["FINAL"],BACK)),"attid":float(torch.dot(DATT,NID)),"vcos":cos(PRED["LOGITS"]-BASE["LOGITS"],NSPEC)}
print(f" FULL BACK={FULL['back']:+.6f} ATT-ID={FULL['attid']:+.6f} VOCAB-COS={FULL['vcos']:+.6f}")
print("[9/14] TOKENWISE K/V CAUSAL RESTORE")
GROUPS=sorted(set(h//GROUP for h in TARGET_HEADS));RESULTS=[]
for g in GROUPS:
 for t in range(len(TOKENS)):
  for kind in ("K","V"):
   R=run(who_ids,WHO_POS,DIRS,+1,kv_patch=(kind,t,g,BASE[kind]))
   att=R["ATT"]-BASE["ATT"];post=R["FINAL"]-BASE["FINAL"];sp=R["LOGITS"]-BASE["LOGITS"]
   cur={"back":float(torch.dot(post,BACK)),"attid":float(torch.dot(att,NID)),"vcos":cos(sp,NSPEC)}
   loss={k:FULL[k]-cur[k] for k in FULL}
   RESULTS.append({"kind":kind,"group":g,"token_pos":t,"token":TOKENS[t],"value":cur,"loss":loss})
 print(f" G{g} COMPLETE")
print("[10/14] TOP K CAUSAL TOKENS")
for g in GROUPS:
 rr=sorted([x for x in RESULTS if x["group"]==g and x["kind"]=="K"],key=lambda x:x["loss"]["back"],reverse=True)
 print(f" G{g}:"," | ".join(f"{x['token_pos']:02d}:{x['token']!r} ΔBACK={x['loss']['back']:+.6f}"for x in rr[:8]))
print("[11/14] TOP V CAUSAL TOKENS")
for g in GROUPS:
 rr=sorted([x for x in RESULTS if x["group"]==g and x["kind"]=="V"],key=lambda x:x["loss"]["back"],reverse=True)
 print(f" G{g}:"," | ".join(f"{x['token_pos']:02d}:{x['token']!r} ΔBACK={x['loss']['back']:+.6f}"for x in rr[:8]))
print("[12/14] PEAK LOCALIZATION")
PEAKS={}
for g in GROUPS:
 for kind in ("K","V"):
  rr=sorted([x for x in RESULTS if x["group"]==g and x["kind"]==kind],key=lambda x:x["loss"]["back"],reverse=True)
  PEAKS[f"{kind}_G{g}"]=rr[:5];x=rr[0]
  print(f" {kind} G{g} PEAK pos{x['token_pos']:02d} {x['token']!r} | ΔBACK={x['loss']['back']:+.6f} ΔATT-ID={x['loss']['attid']:+.6f} ΔVCOS={x['loss']['vcos']:+.6f}")
print("[13/14] GROUP/TOKEN CAUSAL MASS")
MASS={}
for g in GROUPS:
 for kind in ("K","V"):
  rr=[x for x in RESULTS if x["group"]==g and x["kind"]==kind];pos=sum(max(0.,x["loss"]["back"])for x in rr);ab=sum(abs(x["loss"]["back"])for x in rr);best=max(rr,key=lambda x:x["loss"]["back"])
  MASS[f"{kind}_G{g}"]={"positive_back_mass":pos,"absolute_back_mass":ab,"peak_pos":best["token_pos"],"peak_token":best["token"],"peak_back_loss":best["loss"]["back"]}
  print(f" {kind} G{g} POS-MASS={pos:+.6f} ABS-MASS={ab:.6f} PEAK={best['token_pos']:02d}:{best['token']!r}")
print("[14/14] VERDICT + INTEGRITY")
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Qwen not frozen")
VERDICT="L26_H17_H25_H26_TOKENWISE_KV_CAUSAL_SOURCE_MAP_COMPLETE"
REPORT={"test":428,"title":"L26 H17/H25/H26 Tokenwise K/V Causal Source X-Ray","seed":SEED,"source_model":MID,"target_model":QID,"fact":FACT,"counterfactual":CF,"who_query":WHO,"train_pairs":24,"layer_map":LMAP,"write_layers":WRITE_LAYERS,"selected_lambda":LAM,"cross_validation":CV,"target_heads":TARGET_HEADS,"target_kv_groups":GROUPS,"tokens":TOKENS,
"global_final":{"fingerprint_cos":FFP,"local_mean":FLOCAL,"coef_norm":float(COEF.norm())},
"replication":{"native_l26_in_identity_energy":efrac(NIN,NIP),"native_l26_out_identity_energy":efrac(NOUT,NOP),"transfer_l26_in_identity_energy":efrac(DIN,DIP),"transfer_l26_out_identity_energy":efrac(DOUT,DOP),"transfer_att_native_identity":float(torch.dot(DATT,NID)),"transfer_mlp_native_identity":float(torch.dot(DMLP,NID))},
"capture_shapes":{"Q":list(BASE["Q"].shape),"K":list(BASE["K"].shape),"V":list(BASE["V"].shape)},"full_transfer":FULL,"tokenwise_kv_restore":RESULTS,"peaks":PEAKS,"causal_mass":MASS,"verdict":VERDICT,
"integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1,"pass":Q0==Q1},
"notes":["TEST427 proven Mistral/Qwen extraction, TRAIN-only ridge, global solve, L14-L23 write and source-absent WHO are preserved.","H17/H25/H26 are fixed from TEST426/427; H17 maps to G2 and H25/H26 map to G3.","Primary assay is tokenwise K/V restoration at L26 k_proj/v_proj output.","Captured BASE K/V tensors are preserved as complete [sequence,512] tensors; no erroneous BASE[kind][0] reduction is used.","Each intervention restores exactly one 128D KV-group slice at one WHO token to BASE while preserving the rest of the transferred run.","No projection-space pseudo-attention reconstruction is used as causal evidence.","FINAL FACT/CF remain evaluation-only. No model weights are modified."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print(" VERDICT:",VERDICT);print("="*128);print("TEST 428 — COMPLETE");print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS");print("SUMMARY:",OUT);print("="*128)



