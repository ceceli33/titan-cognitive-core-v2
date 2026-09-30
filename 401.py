# TEST 401 — TEST399 EXACT-LINEAGE K-ONLY CAUSAL DECOMPOSITION — FULL STANDALONE
import os,sys,gc,json,time,random,hashlib,subprocess,importlib.util,math
from pathlib import Path
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
 if importlib.util.find_spec(m)is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import torch,transformers
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as qwen_rope
if not torch.cuda.is_available():raise RuntimeError("CUDA REQUIRED")
os.environ["TOKENIZERS_PARALLELISM"]="false";torch.set_grad_enabled(False)
SEED=392;random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEV="cuda";DTYPE=torch.bfloat16;SEP="\n\n"
MID="mistralai/Mistral-7B-Instruct-v0.3";QID="Qwen/Qwen2.5-7B-Instruct"
ROOT=Path("/content/AKBASCORE_TEST401")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST401")
ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/"TEST401_SUMMARY.json"
FACT="Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge."
CF="Leyla Demir planted the Turkish flag at the base of the Golden Gate Bridge."
WHO="Who planted the Turkish flag at the base of the Golden Gate Bridge?"
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
def forge(tok,model,layers,s,with_q=False):
 seq=[pad(tok)]+ids(tok,s+SEP);T=len(seq);out=model(input_ids=torch.tensor([seq],device=DEV),use_cache=False,output_hidden_states=True,return_dict=True);K=[];V=[];Q=[]
 if with_q:
  pos=torch.arange(T,device=DEV)[None];cr,sr=model.model.rotary_emb(out.hidden_states[0],pos)
 for L,layer in enumerate(layers):
  z=layer.input_layernorm(out.hidden_states[L][0]);a=layer.self_attn;K.append(a.k_proj(z).float().cpu().contiguous());V.append(a.v_proj(z).float().cpu().contiguous())
  if with_q:
   q=a.q_proj(z).reshape(1,T,28,128).transpose(1,2);qr,_=qwen_rope(q,q,cr,sr,unsqueeze_dim=1);Q.append(qr[0].float().cpu().contiguous())
 del out;return {"T":T,"K":K,"V":V,"Q":Q}
LMAP=[min(31,max(0,round((L+.5)*32/28-.5)))for L in range(28)]
def resize(x,n):
 if x.shape[0]==n:return x.contiguous()
 return F.interpolate(x.T.unsqueeze(0),size=n,mode="linear",align_corners=False)[0].T.contiguous()
def src(f,L,n,kind):
 x=f[kind][LMAP[L]][1:].reshape(-1,8,128).reshape(-1,4,2,128).mean(2);return resize(x.reshape(-1,512),n).reshape(n,4,128)
def tgt(f,L,kind):return f[kind][L][1:].reshape(-1,4,128)
def stats():return {k:[[{name:torch.zeros(sh,dtype=torch.float64)for name,sh in(("sx",(128,)),("sy",(128,)),("xx",(128,128)),("xy",(128,128)),("yy",()))}|{"n":0}for h in range(4)]for L in range(28)]for k in("K","V")}
def add(st,X,Y):
 X=X.double();Y=Y.double()
 for h in range(4):
  a=X[:,h];b=Y[:,h];s=st[h];n=a.shape[0];s["sx"]+=a.sum(0);s["sy"]+=b.sum(0);s["xx"]+=a.T@a;s["xy"]+=a.T@b;s["yy"]+=(b*b).sum();s["n"]+=n
def pair(ma,mb,qa,qb,L,kind):
 n=min(qa["T"],qb["T"])-1;X=src(mb,L,n,kind)-src(ma,L,n,kind);a=resize(tgt(qa,L,kind).reshape(-1,512),n).reshape(n,4,128);b=resize(tgt(qb,L,kind).reshape(-1,512),n).reshape(n,4,128);return X,b-a
def solve(st):
 B={k:[]for k in st};fit=[]
 for kind in st:
  for L in range(28):
   group=[]
   for h in range(4):
    s=st[kind][L][h];n=s["n"];mx=s["sx"]/n;my=s["sy"]/n;xx=s["xx"]-n*torch.outer(mx,mx);xy=s["xy"]-n*torch.outer(mx,my);alpha=max(1e-8,float(xx.trace()/128)*.05);W=torch.linalg.solve(xx+alpha*torch.eye(128,dtype=torch.float64),xy)
    err=float((s["yy"]-n*(my@my)-2*(W*xy).sum()+(W*(xx@W)).sum()).clamp_min(0));var=float((s["yy"]-n*(my@my)).clamp_min(1e-12));fit.append(1-err/var);group.append((W.float(),mx.float(),my.float()))
   B[kind].append(group)
 return B,sum(fit)/len(fit)
def predict(m,B,D,n,sink,gain):
 out={}
 for kind in("K","V"):
  arr=[]
  for L in range(28):
   X=src(m,L,n,kind);Y=[]
   for h in range(4):
    W,mx,my=B[kind][L][h];WD,_,_=D[kind][L][h];Y.append((X[:,h]-mx)@W+my+gain*((X[:,h]-mx)@(WD-W)))
   arr.append(torch.cat([sink[kind][L],torch.stack(Y,1).reshape(n,512)],0))
  out[kind]=arr
 return out
def native_resize(q,T):return {k:[torch.cat([q[k][L][:1],resize(q[k][L][1:],T-1)],0)for L in range(28)]for k in("K","V")}
@torch.inference_mode()
def fixed_query(tok,model,layers):
 seq=[pad(tok)]+ids(tok,"QUESTION:\n"+WHO+"\n\nANSWER:");T=len(seq);out=model(input_ids=torch.tensor([seq],device=DEV),use_cache=False,output_hidden_states=True,return_dict=True);pos=torch.arange(T,device=DEV)[None];cr,sr=model.model.rotary_emb(out.hidden_states[0],pos);Q=[]
 for L,layer in enumerate(layers):
  z=layer.input_layernorm(out.hidden_states[L][0]);a=layer.self_attn;q=a.q_proj(z).reshape(1,T,28,128).transpose(1,2);qr,_=qwen_rope(q,q,cr,sr,unsqueeze_dim=1);Q.append(qr[0,:,-1].float().cpu().contiguous())
 del out;return Q
def rotk(raw,L):
 T=raw["K"][L].shape[0];p=torch.arange(T,dtype=torch.float32);inv=1.0/(1000000.0**(torch.arange(0,128,2,dtype=torch.float32)/128));a=torch.outer(p,inv);co=torch.cat([a,a],-1).cos();si=torch.cat([a,a],-1).sin();k=raw["K"][L].reshape(T,4,128).float()
 return k*co[:,None,:]+torch.cat([-k[...,64:],k[...,:64]],-1)*si[:,None,:]
def read(Kraw,Vraw,Q,L):
 k=rotk(Kraw,L);v=Vraw["V"][L].reshape(-1,4,128).float();q=Q[L].reshape(4,7,128)
 if k.shape[0]!=v.shape[0]:raise RuntimeError("K/V token mismatch")
 s=torch.einsum("ghd,tgd->ght",q,k)/math.sqrt(128);w=s.softmax(-1);r=torch.einsum("ght,tgd->ghd",w,v);return w,r
def cos(a,b):return float(F.cosine_similarity(a.float().reshape(-1),b.float().reshape(-1),dim=0,eps=1e-12))
def rel(a,b):return float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))
def delta(KF,VF,KC,VC,Q,NF,NC,details=False):
 ac=[];rc=[];rows=[]
 for L in range(28):
  wf,rf=read(KF,VF,Q,L);wc,rrc=read(KC,VC,Q,L);wfn,rfn=read(NF,NF,Q,L);wcn,rcn=read(NC,NC,Q,L)
  ca=cos(wc-wf,wcn-wfn);cr=cos(rrc-rf,rcn-rfn);ac.append(ca);rc.append(cr)
  if details:rows.append({"L":L,"ATT_DELTA_COS":ca,"READ_DELTA_COS":cr})
 z={"ATT_DELTA_COS":sum(ac)/28,"READ_DELTA_COS":sum(rc)/28}
 if details:z["layers"]=rows
 return z
def same_packet(X,T):
 return {"K":[resize(x.reshape(x.shape[0],-1),T).reshape(T,512).contiguous() for x in X["K"]],"V":[resize(x.reshape(x.shape[0],-1),T).reshape(T,512).contiguous() for x in X["V"]]}
def random_like_K(base,seed):
 g=torch.Generator(device="cpu");g.manual_seed(seed);arr=[]
 for x in base["K"]:
  xf=x.float();r=torch.randn(xf.shape,generator=g,dtype=torch.float32);r=r/r.norm().clamp_min(1e-12)*xf.norm();arr.append(r.contiguous())
 return {"K":arr,"V":base["V"]}
def decompose(KF,VF,KC,VC,Q,NF,NC):
 out=[];C={"TOTAL":[],"VALUE_TERM":[],"ROUTING_TERM":[],"RECON_ERR":[],"VALUE_TO_NATIVE":[],"ROUTING_TO_NATIVE":[]}
 for L in range(28):
  Af,Rf=read(KF,VF,Q,L);Ac,Rc=read(KC,VC,Q,L);Anf,Rnf=read(NF,NF,Q,L);Anc,Rnc=read(NC,NC,Q,L)
  vf=VF["V"][L].reshape(-1,4,128).float();vc=VC["V"][L].reshape(-1,4,128).float()
  value=torch.einsum("ght,tgd->ghd",Af,vf-vc)
  routing=torch.einsum("ght,tgd->ghd",Af-Ac,vc)
  total=Rf-Rc;native=Rnf-Rnc;recon=value+routing
  row={"L":L,"TOTAL_TO_NATIVE":cos(total,native),"VALUE_TO_NATIVE":cos(value,native),"ROUTING_TO_NATIVE":cos(routing,native),
       "VALUE_NORM_RATIO":float(value.norm()/native.norm().clamp_min(1e-12)),"ROUTING_NORM_RATIO":float(routing.norm()/native.norm().clamp_min(1e-12)),
       "RECON_REL":rel(recon,total),"VALUE_ROUTING_COS":cos(value,routing)}
  out.append(row);C["TOTAL"].append(row["TOTAL_TO_NATIVE"]);C["VALUE_TO_NATIVE"].append(row["VALUE_TO_NATIVE"]);C["ROUTING_TO_NATIVE"].append(row["ROUTING_TO_NATIVE"]);C["RECON_ERR"].append(row["RECON_REL"])
 return {"mean":{"TOTAL_TO_NATIVE":sum(C["TOTAL"])/28,"VALUE_TO_NATIVE":sum(C["VALUE_TO_NATIVE"])/28,"ROUTING_TO_NATIVE":sum(C["ROUTING_TO_NATIVE"])/28,"RECON_REL":sum(C["RECON_ERR"])/28},"layers":out}
print("="*116);print("TEST 401 — TEST399 EXACT-LINEAGE K-ONLY CAUSAL DECOMPOSITION");print("ORIGINAL × SAME_K × FOREIGN_K × RANDOM_K × SAME_V + EXACT READOUT DECOMPOSITION");print("="*116)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/7] MISTRAL — SAME TEST399 SOURCE EXTRACTION")
mt,mm,ml=load(MID,(32,4096,32,8,128));mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight];M0=sha(mp);MA=[];MB=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):
 MA.append(forge(mt,mm,ml,a));MB.append(forge(mt,mm,ml,b));print(f" MISTRAL {i+1}/32",end="\r")
MF={k:forge(mt,mm,ml,s)for k,s in(("FACT",FACT),("CF",CF))};print("\n SOURCE LENGTHS:",{k:v["T"]for k,v in MF.items()})
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/7] QWEN — SAME TEST399 TARGET EXTRACTION")
qt,qm,ql=load(QID,(28,3584,28,4,128));qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight];Q0=sha(qp);QA=[];QB=[];sink=None
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):
 qa=forge(qt,qm,ql,a,True);qb=forge(qt,qm,ql,b,True);QA.append(qa);QB.append(qb)
 if sink is None:sink={k:[qa[k][L][:1].clone()for L in range(28)]for k in("K","V")}
 print(f" QWEN {i+1}/32",end="\r")
QF={k:forge(qt,qm,ql,s,True)for k,s in(("FACT",FACT),("CF",CF))};T=QF["FACT"]["T"];assert T==QF["CF"]["T"]==19
print("\n TARGET LENGTH:",T,"| TRAIN=24 | VALIDATION=8 | FINAL FACT/CF NEVER FIT")
print("[3/7] SAME TEST399 FP64 BASE + DELTA BRIDGE")
ST=stats();SD=stats()
for i in range(24):
 for kind in("K","V"):
  for L in range(28):
   for m,q in((MA[i],QA[i]),(MB[i],QB[i])):
    n=q["T"]-1;add(ST[kind][L],src(m,L,n,kind),tgt(q,L,kind))
   X,Y=pair(MA[i],MB[i],QA[i],QB[i],L,kind);add(SD[kind][L],X,Y)
B,R2B=solve(ST);D,R2D=solve(SD);print(f" TRAIN R² BASE={R2B:.6f} DELTA={R2D:.6f}")
NATIVE={s:native_resize(QF[s],T)for s in("FACT","CF")};RAW={}
for s in("FACT","CF"):
 for name,g in {"BASE":0.,"HYBRID":.5,"DELTA":1.}.items():RAW[s+"_"+name]=predict(MF[s],B,D,T-1,sink,g)
del ST,SD,MA,MB,QA,QB;clean()
print("[4/7] SAME TEST399 FIXED WHO QUERY")
QWHO=fixed_query(qt,qm,ql);assert len(QWHO)==28 and all(q.shape==(28,128)for q in QWHO);print(" WHO QUERY BANK:",len(QWHO),"layers ×",tuple(QWHO[0].shape))
print("[5/7] BASE K-ONLY FALSIFICATION ARMS")
NF,NC=NATIVE["FACT"],NATIVE["CF"];RF,RC=RAW["FACT_BASE"],RAW["CF_BASE"]
SAMEKF={"K":[x.clone()for x in RF["K"]],"V":RF["V"]};SAMEKC={"K":[x.clone()for x in RF["K"]],"V":RF["V"]}
FOREIGN=RAW["FACT_BASE"].copy();FOREIGN["K"]=[predict(MF["FACT"],B,D,T-1,sink,0.)["K"][L].clone() for L in range(28)]
# Foreign K must be semantically unrelated: use mapped K from first held-out validation neutral source, never FACT/CF.
# Reconstruct it directly from already-fitted bridge without fitting anything new.
FM=forge(qt,qm,ql,NEUTRAL[24],True)
FOREIGN={"K":[x.clone()for x in native_resize(FM,T)["K"]],"V":NF["V"]}
del FM
RANDOM=random_like_K(RF,SEED+401)
ARMS={
"ORIGINAL_K_ONLY":(RF,NF,RC,NC),
"SAME_K":(SAMEKF,NF,SAMEKC,NC),
"FOREIGN_K":(FOREIGN,NF,FOREIGN,NC),
"RANDOM_K":(RANDOM,NF,RANDOM,NC),
"SAME_V_FACT":(RF,NF,RC,NF),
"SAME_V_CF":(RF,NC,RC,NC)}
AUDIT={}
for name,(KF,VF,KC,VC) in ARMS.items():
 d=delta(KF,VF,KC,VC,QWHO,NF,NC,True);AUDIT[name]=d;print(f" {name:16s} ATTΔ={d['ATT_DELTA_COS']:+.6f} READΔ={d['READ_DELTA_COS']:+.6f}")
print("[6/7] EXACT ORIGINAL K-ONLY READOUT DECOMPOSITION")
DEC=decompose(RF,NF,RC,NC,QWHO,NF,NC);m=DEC["mean"]
print(f" TOTAL→NATIVE   ={m['TOTAL_TO_NATIVE']:+.6f}")
print(f" VALUE TERM     ={m['VALUE_TO_NATIVE']:+.6f}   [A_FACT · (V_FACT−V_CF)]")
print(f" ROUTING TERM   ={m['ROUTING_TO_NATIVE']:+.6f}   [(A_FACT−A_CF) · V_CF]")
print(f" RECON REL ERR  ={m['RECON_REL']:.9f}")
print(" LAYER X-RAY:")
for L in [0,3,6,10,14,18,19,21,23,25,27]:
 r=DEC["layers"][L];print(f" L{L:02d} TOTAL={r['TOTAL_TO_NATIVE']:+.4f} VALUE={r['VALUE_TO_NATIVE']:+.4f} ROUTE={r['ROUTING_TO_NATIVE']:+.4f} | Vnorm={r['VALUE_NORM_RATIO']:.3f} Rnorm={r['ROUTING_NORM_RATIO']:.3f} VRcos={r['VALUE_ROUTING_COS']:+.3f}")
print("[7/7] VERDICT + INTEGRITY")
orig=AUDIT["ORIGINAL_K_ONLY"]["READ_DELTA_COS"];same=AUDIT["SAME_K"]["READ_DELTA_COS"];foreign=AUDIT["FOREIGN_K"]["READ_DELTA_COS"];rnd=AUDIT["RANDOM_K"]["READ_DELTA_COS"];sv=max(AUDIT["SAME_V_FACT"]["READ_DELTA_COS"],AUDIT["SAME_V_CF"]["READ_DELTA_COS"]);floor=max(same,foreign,rnd)
if floor>=orig*.80 and sv<orig*.50:VERDICT="NATIVE_V_DOMINANT: TEST399 K_ONLY alignment is largely reproducible without FACT↔CF mapped-K information."
elif floor<orig*.50 and sv>=orig*.70:VERDICT="K_ROUTING_SURVIVES: mapped FACT↔CF K difference carries substantial alignment even with ΔV removed."
elif floor<orig*.70 and sv<orig*.70:VERDICT="INTERACTION: neither native ΔV alone nor mapped ΔK alone reproduces ORIGINAL; K×V interaction/synergy is implicated."
else:VERDICT="MIXED: both native-V floor and mapped-K-only routing contribute materially; inspect decomposition/layers."
print(f" ORIGINAL={orig:+.6f} | NULL-K FLOOR={floor:+.6f} | SAME-V BEST={sv:+.6f}")
print(" VERDICT:",VERDICT)
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Qwen not frozen")
REPORT={"test":401,"title":"TEST399 Exact-Lineage K-Only Causal Decomposition","source_model":MID,"target_model":QID,"fact":FACT,"counterfactual":CF,"question":WHO,"seed":SEED,
"train_pairs":24,"validation_pairs":8,"train_R2":{"BASE":R2B,"DELTA":R2D},"arms":AUDIT,"decomposition":DEC,
"score":{"original":orig,"null_k_floor":floor,"same_v_best":sv},"verdict":VERDICT,
"integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1,"pass":Q0==Q1},
"notes":["Exact TEST399 bridge lineage preserved: seed, datasets, FP64 BASE/DELTA fit, layer map, interpolation and fixed WHO-Q.","SAME_K keeps mapped FACT K fixed across FACT/CF while native Qwen V changes.","FOREIGN_K and RANDOM_K contain no FACT↔CF K difference.","SAME_V_FACT/CF remove native ΔV while preserving mapped FACT↔CF ΔK.","Readout decomposition uses ΔR=A_F(V_F−V_C)+(A_F−A_C)V_C exactly for ORIGINAL_K_ONLY."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("="*116);print("TEST 401 — COMPLETE");print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS");print("SUMMARY:",OUT);print("="*116)
