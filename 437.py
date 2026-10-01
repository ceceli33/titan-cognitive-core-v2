# TEST 437 — FROZEN SENTINEL HIGHWAY × AKBASCORE MATCHED COMPASS PACKETS
# TEST436 PACKET/RIDGE/LMAP/L14-L23 WRITE PRESERVED | TEST250 MEAN(POS)-MEAN(NEG) COMPASS
# TRAIN-ONLY 6/12/24 COMPASS BANKS | VAL-ONLY MAP SELECTION | FINAL EVALUATION ONLY
# ONE RUN: BASE + OLD PACKET + COMPASS PACKETS + REVERSE + SHUFFLED COORDINATES + FREE GENERATION
# NO WEIGHT UPDATES | NO FINAL-DERIVED WRITE | NO CONSTRAINED DECODING
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
ROOT=Path("/content/AKBASCORE_TEST437")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST437");ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/"TEST437_SUMMARY.json"
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
for s,(a,b)in zip(NEUTRAL,SWAP):
 assert s.startswith(a)and a!=b;ALT.append(b+s[len(a):])
LMAP=[min(31,max(0,round((L+.5)*32/28-.5)))for L in range(28)]
TRAIN=list(range(24));VAL=list(range(24,32));LAYERS=list(range(28));WRITE_LAYERS=list(range(14,24));TAIL=[23,24,25,26,27]
RIDGES=[1e-4,1e-3,1e-2,.05,.1,.5,1.,5.]
DOSES=[.10,.20,.30,.40,.50,.75,1.,1.25,1.5,2.]
COMPASS_SIZES=[6,12,24];COORD_RIDGE=.01;GEN_DOSES=[.10,.30,.75,1.];PULSES=[1,3];MAX_NEW=28;NNULL=8
# Each compass axis pools fixed TRAIN contrasts; group labels do not imply pure semantic factors.
# POS=ALT, NEG=NEUTRAL. No Leyla/Mustafa examples enter compass extraction or fitting.
LOCK={"train":TRAIN,"val":VAL,"lmap":LMAP,"write_layers":WRITE_LAYERS,"ridges":RIDGES,"seed":SEED,"neutral":NEUTRAL,"swap":SWAP,"fact":FACT,"cf":CF,"who":WHO}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
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
 if dims!=expected:raise RuntimeError(f"Architecture mismatch {dims}")
 print("READY:",dims,"|",round(time.time()-t,2),"s");return tok,model,layers
def ids(tok,s):return tok(s,add_special_tokens=False).input_ids
def pad(tok):return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
def unit(x):return x.float()/x.float().norm().clamp_min(1e-12)
def cos(a,b):return float(F.cosine_similarity(a.float().reshape(-1),b.float().reshape(-1),dim=0,eps=1e-12))
@torch.inference_mode()
def forge(tok,model,s):
 seq=[pad(tok)]+ids(tok,s+SEP);o=model(input_ids=torch.tensor([seq],device=DEV),use_cache=False,output_hidden_states=True,return_dict=True);pos=max(0,len(seq)-len(ids(tok,SEP))-1)
 H=[o.hidden_states[L+1][0,pos].float().cpu().contiguous()for L in range(len(model.model.layers))];del o;return H
def packet(A,B,layers):return [unit(B[L]-A[L])for L in layers]
def catpacket(p):return torch.cat([unit(x)for x in p])
def gfp(x,bank,refs):return torch.tensor([cos(catpacket(x),catpacket(bank[j]))for j in refs])
def ridge(X,Y,lam):
 X=X.double();Y=Y.double();mx=X.mean(0);my=Y.mean(0);A=X-mx;B=Y-my;xx=A.T@A;xy=A.T@B;s=max(1e-12,float(xx.trace()/max(1,xx.shape[0])));W=torch.linalg.solve(xx+lam*s*torch.eye(xx.shape[0],dtype=torch.float64),xy);return W.float(),mx.float(),my.float()
def pred(x,f):W,mx,my=f;return(x.float()-mx)@W+my
def fit_global(MP,QP,refs,lam):return ridge(torch.stack([gfp(MP[i],MP,refs)for i in refs]),torch.stack([gfp(QP[i],QP,refs)for i in refs]),lam)
def global_qsolve(target,QP,refs,lam):
 C=torch.stack([catpacket(QP[j])for j in refs]);G=C@C.T;a=torch.linalg.solve(G.T@G+lam*torch.eye(len(refs)),G.T@target.float())
 return {L:unit(sum((a[k]*QP[j][L]for k,j in enumerate(refs)),torch.zeros_like(QP[refs[0]][L])))for L in LAYERS},a
def loo_score(MP,QP,lam):
 fc=[];lc=[]
 for hold in TRAIN:
  refs=[j for j in TRAIN if j!=hold];f=fit_global(MP,QP,refs,lam);qh=pred(gfp(MP[hold],MP,refs),f);qt=gfp(QP[hold],QP,refs);d,_=global_qsolve(qh,QP,refs,lam);fc.append(cos(qh,qt));lc.append(sum(cos(d[L],QP[hold][L])for L in LAYERS)/28)
 return sum(fc)/len(fc),sum(lc)/len(lc)
print("="*140);print("TEST 437 — FROZEN SENTINEL HIGHWAY × MATCHED AKBASCORE COMPASS PACKETS");print("="*140)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("LOCK SHA:",LOCK_SHA);print("GENERATION: TEST436 use_cache=False, last-position pulse, max_new=28 preserved.")
print("[1/11] MISTRAL — EXACT TEST436 EXTRACTION")
mt,mm,ml=load(MID,(32,4096,32,8,128));mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight];M0=sha(mp);MH=[]
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 MH.append((forge(mt,mm,a),forge(mt,mm,b)));print(f" MISTRAL {i+1}/32",end="\r")
MFINAL=(forge(mt,mm,FACT),forge(mt,mm,CF));print()
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/11] QWEN — EXACT TEST436 EXTRACTION")
qt,qm,ql=load(QID,(28,3584,28,4,128));qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight];Q0=sha(qp);QH=[]
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 QH.append((forge(qt,qm,a),forge(qt,qm,b)));print(f" QWEN {i+1}/32",end="\r")
QFINAL=(forge(qt,qm,FACT),forge(qt,qm,CF));print("\n TRAIN=24 | VAL=8 | FINAL NEVER FITS OR SELECTS")
MP=[packet(a,b,LMAP)for a,b in MH];QP=[packet(a,b,LAYERS)for a,b in QH];MF=packet(MFINAL[0],MFINAL[1],LMAP);QF=packet(QFINAL[0],QFINAL[1],LAYERS)
print("[3/11] FROZEN HIGHWAY — EXACT TEST436 GLOBAL RIDGE/SOLVE/MAGNITUDE")
CV=[]
for lam in RIDGES:
 fc,lc=loo_score(MP,QP,lam);CV.append((lam,fc,lc));print(f" λ={lam:<6g} FP={fc:+.6f} LOCAL={lc:+.6f}")
LAM=max(CV,key=lambda x:(x[2],x[1]))[0];FIT=fit_global(MP,QP,TRAIN,LAM);QHAT=pred(gfp(MF,MP,TRAIN),FIT);QTRUE=gfp(QF,QP,TRAIN);DIRS,COEF=global_qsolve(QHAT,QP,TRAIN,LAM)
MAG={L:sum(float((QH[i][1][L]-QH[i][0][L]).norm())for i in TRAIN)/24 for L in LAYERS}
OLD_FP=cos(QHAT,QTRUE);OLD_LOCAL=sum(cos(DIRS[L],QF[L])for L in LAYERS)/28
print(f" OLD λ={LAM} FINAL_FP={OLD_FP:+.6f} LOCAL={OLD_LOCAL:+.6f} COEF={float(COEF.norm()):.6f}")
# Highway guard: old direction tensors and magnitudes are sealed before any compass branch is built.
def tensorseal(ds):
 h=hashlib.sha256()
 for L in LAYERS:h.update(ds[L].contiguous().numpy().tobytes())
 h.update(json.dumps(MAG,sort_keys=True).encode());return h.hexdigest()
HIGHWAY_SHA=tensorseal(DIRS)
print("[4/11] TEST250 COMPASS FORGE — TRAIN-ONLY MATCHED POS/NEG BANKS")
def groups(k):
 assert 24%k==0
 n=24//k;return [TRAIN[j*n:(j+1)*n]for j in range(k)]
def compass(H,pool,lmap):
 axes=[];raw=[]
 for refs in pool:
  vec=[];norms=[]
  for L in LAYERS:
   j=lmap[L];p=torch.stack([H[i][1][j]for i in refs]).mean(0);n=torch.stack([H[i][0][j]for i in refs]).mean(0);d=p-n
   if not torch.isfinite(d).all()or float(d.norm())<1e-8:raise RuntimeError(f"Degenerate compass L{L}")
   vec.append(unit(d));norms.append(float(d.norm()))
  axes.append(vec);raw.append(norms)
 return axes,raw
def coord_setup(axes):
 # Dimension-free coordinates: solve overlapping compass Gram, rather than treating axes as orthogonal.
 C=torch.stack([catpacket(a)for a in axes]).double();C=C/(len(LAYERS)**.5);G=C@C.T
 S=G+COORD_RIDGE*torch.eye(len(axes),dtype=torch.float64)
 return C,S,{"rank":int(torch.linalg.matrix_rank(G)),"condition_regularized":float(torch.linalg.cond(S)),"max_offdiag":float((G-torch.diag(G.diag())).abs().max())}
def coords(p,setup):
 C,S,_=setup;x=catpacket(p).double()/(len(LAYERS)**.5)
 return torch.linalg.solve(S,C@x).float()
def reconstruct(c,axes):
 # Layer-local Qwen-native directions; dose and write locations remain the old highway's.
 return {L:unit(sum((c[j]*axes[j][L]for j in range(len(axes))),torch.zeros_like(axes[0][L])))for L in LAYERS}
def dirscore(d,p,layers=LAYERS):return sum(cos(d[L],p[L])for L in layers)/len(layers)
BANKS={};BANKREPORT={}
for k in COMPASS_SIZES:
 pool=groups(k);MA,MRAW=compass(MH,pool,LMAP);QA,QRAW=compass(QH,pool,LAYERS);MS=coord_setup(MA);QS=coord_setup(QA)
 MC=torch.stack([coords(p,MS)for p in MP]);QC=torch.stack([coords(p,QS)for p in QP])
 BANKS[k]={"M":MA,"Q":QA,"MS":MS,"QS":QS,"MC":MC,"QC":QC}
 BANKREPORT[str(k)]={"groups":pool,"pos":"ALT","neg":"NEUTRAL","mistral_raw_norms":MRAW,"qwen_raw_norms":QRAW,"mistral_geometry":MS[2],"qwen_geometry":QS[2]}
 print(f" K={k:2d} M_rank={MS[2]['rank']:2d} Q_rank={QS[2]['rank']:2d} Q_cond={QS[2]['condition_regularized']:.3f} Q_overlap={QS[2]['max_offdiag']:.4f}")
print("[5/11] TRAIN COORDINATE MAP — VAL-ONLY SELECTION")
PACKS={"OLD":DIRS};MAPREPORT={}
for k in COMPASS_SIZES:
 B=BANKS[k];tr=torch.tensor(TRAIN);best=None;rows=[]
 for lam in RIDGES:
  f=ridge(B["MC"][tr],B["QC"][tr],lam);cc=[];ll=[];wl=[]
  for i in VAL:
   c=pred(B["MC"][i],f);d=reconstruct(c,B["Q"]);cc.append(cos(c,B["QC"][i]));ll.append(dirscore(d,QP[i]));wl.append(dirscore(d,QP[i],WRITE_LAYERS))
  z={"lambda":lam,"coord_cos":sum(cc)/8,"local_cos":sum(ll)/8,"write_cos":sum(wl)/8};rows.append(z)
  print(f" K={k:2d} λ={lam:<6g} VAL_COORD={z['coord_cos']:+.6f} LOCAL={z['local_cos']:+.6f} WRITE={z['write_cos']:+.6f}")
  if best is None or(z["write_cos"],z["local_cos"])>(best["write_cos"],best["local_cos"]):best=z
 B["FIT"]=ridge(B["MC"][tr],B["QC"][tr],best["lambda"]);B["FINAL_MC"]=coords(MF,B["MS"]);B["FINAL_QC"]=pred(B["FINAL_MC"],B["FIT"])
 name=f"C{k:02d}";PACKS[name]=reconstruct(B["FINAL_QC"],B["Q"]);MAPREPORT[name]={"cv":rows,"selected":best}
 print(f" SELECT {name}: λ={best['lambda']} VAL_WRITE={best['write_cos']:+.6f}")
# Every architecture is reported; FINAL never selects a bank, lambda, dose or pulse.
print("[6/11] FINAL PACKET DIAGNOSTICS — MEASUREMENT ONLY")
PACKREPORT={}
for name,d in PACKS.items():
 fp=cos(gfp([d[L]for L in LAYERS],QP,TRAIN),QTRUE);loc=dirscore(d,QF);wr=dirscore(d,QF,WRITE_LAYERS)
 z={"fingerprint_cos":fp,"local_cos":loc,"write_cos":wr,"old_packet_cos":dirscore(d,[DIRS[L]for L in LAYERS]),"layer_cos":{str(L):cos(d[L],QF[L])for L in LAYERS}}
 if name!="OLD":
  k=int(name[1:]);B=BANKS[k];truth=coords(QF,B["QS"]);z["coordinate_cos"]=cos(B["FINAL_QC"],truth)
 PACKREPORT[name]=z;print(f" {name:4s} FP={fp:+.6f} LOCAL={loc:+.6f} WRITE={wr:+.6f} OLD_COS={z['old_packet_cos']:+.6f}")
print("[7/11] SOURCE-ABSENT WHO — FIXED BASE + NATIVE MEASUREMENT")
PROMPT=torch.tensor([[pad(qt)]+ids(qt,WHO)],device=DEV);NF=torch.tensor([[pad(qt)]+ids(qt,FACT+SEP+WHO)],device=DEV);NC=torch.tensor([[pad(qt)]+ids(qt,CF+SEP+WHO)],device=DEV)
LEYLA=ids(qt," Leyla")[0];MUSTAFA=ids(qt," Mustafa")[0]
def rank(x,i):return int((x>x[i]).sum())+1
@torch.inference_mode()
def plain(seq,hidden=False):return qm(input_ids=seq,use_cache=False,output_hidden_states=hidden,return_dict=True)
NB=plain(PROMPT,True);NN0=plain(NF,True);NN1=plain(NC,True);LB=NB.logits[0,-1].float().cpu()
BASEH={L:NB.hidden_states[L+1][0,-1].float().cpu()for L in TAIL}
NATIVE={L:(NN1.hidden_states[L+1][0,-1]-NN0.hidden_states[L+1][0,-1]).float().cpu()for L in TAIL}
BASE_R=rank(LB,LEYLA);BASE_M=float(LB[LEYLA]-LB[MUSTAFA]);del NB,NN0,NN1
print(f" BASE LEYLA#{BASE_R} MARGIN={BASE_M:+.6f} | WHO TOKENS={PROMPT.shape[1]}")
print(" FIRST TARGET TOKEN:",LEYLA,repr(qt.decode([LEYLA])),"|",MUSTAFA,repr(qt.decode([MUSTAFA])))
@torch.inference_mode()
def forward(seq,directions=None,dose=0.,sign=1.):
 hs=[]
 if directions is not None:
  for L in WRITE_LAYERS:
   d=(directions[L]*MAG[L]*dose*sign).to(DEV)
   def inj(m,a,o,d=d):
    if isinstance(o,tuple):
     h=o[0].clone();h[:,-1]+=d.to(h.dtype);return(h,)+o[1:]
    h=o.clone();h[:,-1]+=d.to(h.dtype);return h
   hs.append(ql[L].register_forward_hook(inj))
 try:
  o=qm(input_ids=seq,use_cache=False,output_hidden_states=True,return_dict=True)
  log=o.logits[0,-1].float().cpu();H={L:o.hidden_states[L+1][0,-1].float().cpu()for L in TAIL};del o
 finally:
  for h in hs:h.remove()
 return log,H
def metrics(log,H):
 return {"rank":rank(log,LEYLA),"margin":float(log[LEYLA]-log[MUSTAFA]),"leyla":float(log[LEYLA]),"mustafa":float(log[MUSTAFA]),"cos":{str(L):cos(H[L]-BASEH[L],NATIVE[L])for L in TAIL},"delta_norm":{str(L):float((H[L]-BASEH[L]).norm())for L in TAIL}}
print("[8/11] ALL PACKETS × FIXED DOSES × FORWARD/REVERSE")
RES=[]
for name,directions in PACKS.items():
 for dose in DOSES:
  log,H=forward(PROMPT,directions,dose,1.);z={"packet":name,"dose":dose,"direction":"F",**metrics(log,H)};RES.append(z)
  print(f" {name:4s} {dose:>4.2f} F LEYLA#{z['rank']:6d} MARGIN={z['margin']:+.6f} L26={z['cos']['26']:+.4f} L27={z['cos']['27']:+.4f}")
 for dose in [.30,1.]:
  log,H=forward(PROMPT,directions,dose,-1.);z={"packet":name,"dose":dose,"direction":"R",**metrics(log,H)};RES.append(z)
  print(f" {name:4s} {dose:>4.2f} R LEYLA#{z['rank']:6d} MARGIN={z['margin']:+.6f}")
OLD1=next(z for z in RES if z["packet"]=="OLD"and z["direction"]=="F"and z["dose"]==1.)
REPLICATION={"expected_rank":7469,"observed_rank":OLD1["rank"],"expected_margin":-1.0390625,"observed_margin":OLD1["margin"],"exact_match":OLD1["rank"]==7469 and abs(OLD1["margin"]+1.0390625)<1e-6}
print(" TEST436 OLD1 REPLICATION:",REPLICATION)
print("[9/11] SHUFFLED MISTRAL COMPASS-COORDINATE CONTROLS")
# Same Qwen bank, fitted map, unit local directions, MAG and dose; only source coordinate order changes.
# Eight shuffled controls are descriptive controls, not confirmatory significance tests.
NULL=[];NULLPACK={};rng=random.Random(SEED+437)
for k in COMPASS_SIZES:
 B=BANKS[k];name=f"C{k:02d}";seen=set()
 for j in range(NNULL):
  perm=list(range(k))
  while True:
   rng.shuffle(perm);t=tuple(perm)
   if t!=tuple(range(k))and t not in seen:seen.add(t);break
  c=pred(B["FINAL_MC"][torch.tensor(perm)],B["FIT"]);d=reconstruct(c,B["Q"])
  if j==0:NULLPACK[name]=d
  for dose in [.30,1.]:
   log,H=forward(PROMPT,d,dose);z={"packet":name,"null":j,"permutation":list(perm),"dose":dose,**metrics(log,H)};NULL.append(z)
 for dose in [.30,1.]:
  rr=[z["rank"]for z in NULL if z["packet"]==name and z["dose"]==dose];mm=[z["margin"]for z in NULL if z["packet"]==name and z["dose"]==dose]
  print(f" {name} SHUFFLED {dose:.2f}x RANK_MEAN={sum(rr)/len(rr):.1f} MARGIN_MEAN={sum(mm)/len(mm):+.6f}")
print("[10/11] ALL FREE GENERATIONS — OLD/C6/C12/C24 + BASE + NATIVE + REVERSE + SHUFFLED")
@torch.inference_mode()
def gen(seq0,directions=None,dose=0.,npulse=0,max_new=MAX_NEW,sign=1.):
 seq=seq0.clone();new=[]
 for step in range(max_new):
  if directions is not None and step<npulse:log,_=forward(seq,directions,dose,sign)
  else:
   o=plain(seq);log=o.logits[0,-1].float().cpu();del o
  n=int(log.argmax());new.append(n);seq=torch.cat([seq,torch.tensor([[n]],device=DEV)],1)
  if n==qt.eos_token_id:break
 return qt.decode(new,skip_special_tokens=True)
def namehit(txt):return "leyla demir"in " ".join(txt.lower().split())
GEN=[];CONTROLS=[]
for label,seq in [("BASE",PROMPT),("NATIVE_FACT",NF),("NATIVE_CF",NC)]:
 txt=gen(seq);z={"condition":label,"text":txt,"target_mention":namehit(txt)};CONTROLS.append(z);print(f" {label:11s} MENTION={z['target_mention']} | {txt!r}")
for name,directions in PACKS.items():
 for dose in GEN_DOSES:
  for pulse in PULSES:
   txt=gen(PROMPT,directions,dose,pulse);z={"packet":name,"dose":dose,"pulse":pulse,"direction":"F","text":txt,"target_mention":namehit(txt)};GEN.append(z)
   print(f" {name:4s} {dose:.2f} P{pulse} F MENTION={z['target_mention']} | {txt!r}")
 for pulse in PULSES:
  txt=gen(PROMPT,directions,1.,pulse,sign=-1.);z={"packet":name,"dose":1.,"pulse":pulse,"direction":"R","text":txt,"target_mention":namehit(txt)};CONTROLS.append(z)
  print(f" {name:4s} 1.00 P{pulse} R MENTION={z['target_mention']} | {txt!r}")
for name,directions in NULLPACK.items():
 for dose in [.30,1.]:
  txt=gen(PROMPT,directions,dose,1);z={"packet":name,"condition":"SHUFFLED_0","dose":dose,"pulse":1,"text":txt,"target_mention":namehit(txt)};CONTROLS.append(z)
  print(f" {name:4s} SHUFFLED {dose:.2f} P1 MENTION={z['target_mention']} | {txt!r}")
print("[11/11] PREDECLARED SUMMARY + INTEGRITY + SAVE")
# Fixed primary comparison: 0.30x, P1. Dose/pulse sweep is exploratory and cannot select a new FINAL setting.
PRIMARY={}
for name in PACKS:
 z=next(z for z in RES if z["packet"]==name and z["direction"]=="F"and z["dose"]==.30)
 g=next(g for g in GEN if g["packet"]==name and g["dose"]==.30 and g["pulse"]==1)
 PRIMARY[name]={"rank":z["rank"],"margin":z["margin"],"target_mention":g["target_mention"],"text":g["text"]}
 print(f" PRIMARY {name:4s} .30x P1 RANK={z['rank']} MARGIN={z['margin']:+.6f} MENTION={g['target_mention']}")
OLDHITS=[g for g in GEN if g["packet"]=="OLD"and g["target_mention"]];NEWHITS=[g for g in GEN if g["packet"]!="OLD"and g["target_mention"]]
CTRLHITS=[g for g in CONTROLS if g.get("condition")not in ["NATIVE_FACT","NATIVE_CF"]and g["target_mention"]]
if NEWHITS:VERDICT="COMPASS_TARGET_MENTION_CANDIDATE_REQUIRES_BINDING_AND_FRESH_CONFIRMATION"
else:VERDICT="NO_TARGET_MENTION_WITH_TESTED_COMPASS_PACKETS"
print(" BASE:",BASE_R,f"{BASE_M:+.6f}");print(" OLD TARGET MENTIONS:",len(OLDHITS),"| COMPASS TARGET MENTIONS:",len(NEWHITS),"| NON-NATIVE CONTROL MENTIONS:",len(CTRLHITS))
for name in PACKS:
 rows=[z for z in RES if z["packet"]==name and z["direction"]=="F"];br=min(rows,key=lambda z:z["rank"]);bm=max(rows,key=lambda z:z["margin"])
 print(f" EXPLORATORY {name}: BEST_RANK={br['rank']}@{br['dose']:.2f} BEST_MARGIN={bm['margin']:+.6f}@{bm['dose']:.2f} RANK_AT_MARGIN={bm['rank']}")
print(" VERDICT:",VERDICT)
if tensorseal(DIRS)!=HIGHWAY_SHA:raise RuntimeError("Frozen old packet or magnitude changed")
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Qwen not frozen")
def jsonsafe(x):
 if isinstance(x,dict):return {str(k):jsonsafe(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [jsonsafe(v)for v in x]
 if isinstance(x,torch.Tensor):return x.tolist()
 return x
REPORT={"test":437,"title":"Frozen Sentinel Highway x Matched AkbasCore Compass Packets","seed":SEED,"lock_sha":LOCK_SHA,"lock":LOCK,"global_lambda":LAM,"global_cv":CV,"old_final":{"fingerprint":OLD_FP,"local":OLD_LOCAL,"coef_norm":float(COEF.norm())},"highway_sha":HIGHWAY_SHA,"replication":REPLICATION,"compass_sizes":COMPASS_SIZES,"coordinate_ridge":COORD_RIDGE,"banks":BANKREPORT,"maps":MAPREPORT,"final_packets":PACKREPORT,"base":{"rank":BASE_R,"margin":BASE_M},"results":RES,"shuffled_controls":NULL,"generation":GEN,"generation_controls":CONTROLS,"primary":{"dose":.30,"pulse":1,"results":PRIMARY},"old_mentions":OLDHITS,"compass_mentions":NEWHITS,"non_native_control_mentions":CTRLHITS,"verdict":VERDICT,"integrity":{"mistral":"PASS","qwen":"PASS","frozen_highway":"PASS","qwen_before":Q0,"qwen_after":Q1},"limitations":["Compass groups are fixed TRAIN contrast bundles, not proven isolated entity/event factors.","C24 uses one TRAIN pair per compass and serves as a bank-resolution control.","Global solve and old packet remain unchanged; new compass reconstruction is a separate packet branch.","Coordinates use layer-normalized contrasts, preserving the legacy packet representation.","VAL selects map ridge only; all bank sizes are reported without FINAL selection.","Native FINAL states are measurement references only.","FINAL Mistral contrast is the transmitted source payload; FINAL Qwen contrast never constructs a write.","WHO already contains event terms; their repetition is not evidence of event transfer.","Generation preserves TEST436 use_cache=False pulse semantics; prior latent writes are not retained across recomputation.","Target mention within 28 tokens is a screening metric, not verified identity binding.","First-token rank is for the tokenizer's first token of spaced Leyla, not the full name.","Shuffled controls and dose/pulse sweeps are exploratory; no confirmatory p-value is claimed.","Native source-present generations are positive controls only.","No target forcing, masking, reranking or post-hoc insertion in free generation.","Weight sentinel samples selected weights and is not a full-weight hash."]}
OUT.write_text(json.dumps(jsonsafe(REPORT),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
print("="*140);print("TEST 437 — COMPLETE");print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS | FROZEN HIGHWAY: PASS");print("SUMMARY:",OUT);print("="*140)
