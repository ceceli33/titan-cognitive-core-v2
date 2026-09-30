# TEST 400 — CROSS-MODEL TOURNAMENT: K/V BRIDGE × DUAL-ENDOGENOUS RELATIONAL PACKET
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
SEED=400;random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEV="cuda";DTYPE=torch.bfloat16;SEP="\n\n";EPS=1e-8
MID="mistralai/Mistral-7B-Instruct-v0.3";QID="Qwen/Qwen2.5-7B-Instruct"
ROOT=Path("/content/AKBASCORE_TEST400")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST400");ROOT.mkdir(parents=True,exist_ok=True)
OUT=ROOT/"TEST400_SUMMARY.json"
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
for s,(a,b)in zip(NEUTRAL,SWAP):
 assert s.startswith(a)and a!=b;ALT.append(b+s[len(a):])
def clean():gc.collect();torch.cuda.empty_cache()
def unit(x):return x/x.norm().clamp_min(EPS)
def cos(a,b):return float(F.cosine_similarity(a.float().reshape(-1),b.float().reshape(-1),dim=0,eps=EPS))
def rel(a,b):return float((a.float()-b.float()).norm()/b.float().norm().clamp_min(EPS))
def sha(ps):
 h=hashlib.sha256()
 for p in ps:
  a=p.detach().reshape(-1);n=a.numel()
  for j in range(16):
   o=j*max(0,n-256)//15;h.update(a[o:o+256].float().cpu().numpy().tobytes())
 return h.hexdigest()
def load(mid,expected):
 print("LOAD:",mid);t=time.time();tv=tuple(int("".join(c for c in x if c.isdigit())or 0)for x in transformers.__version__.split(".")[:2])
 kw={"dtype"if tv>=(4,56)else"torch_dtype":DTYPE};tok=AutoTokenizer.from_pretrained(mid,use_fast=True)
 model=AutoModelForCausalLM.from_pretrained(mid,device_map={"":0},attn_implementation="sdpa",**kw);model.eval()
 for p in model.parameters():p.requires_grad_(False)
 c=model.config;layers=model.model.layers
 dims=(len(layers),c.hidden_size,c.num_attention_heads,c.num_key_value_heads,getattr(c,"head_dim",None)or c.hidden_size//c.num_attention_heads)
 if dims!=expected:raise RuntimeError(f"Architecture mismatch: {dims}")
 print("READY:",dims,"|",round(time.time()-t,2),"s");return tok,model,layers
def ids(tok,s):return tok(s,add_special_tokens=False).input_ids
def pad(tok):return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
@torch.inference_mode()
def forge(tok,model,layers,s,qwen=False):
 seq=[pad(tok)]+ids(tok,s+SEP);T=len(seq);o=model(input_ids=torch.tensor([seq],device=DEV),use_cache=False,output_hidden_states=True,return_dict=True)
 K=[];V=[];H=[];Q=[]
 if qwen:
  pos=torch.arange(T,device=DEV)[None];cr,sr=model.model.rotary_emb(o.hidden_states[0],pos)
 for L,layer in enumerate(layers):
  z=layer.input_layernorm(o.hidden_states[L][0]);a=layer.self_attn
  K.append(a.k_proj(z).float().cpu().contiguous());V.append(a.v_proj(z).float().cpu().contiguous())
  H.append(o.hidden_states[L+1][0,-1].float().cpu().contiguous())
  if qwen:
   q=a.q_proj(z).reshape(1,T,28,128).transpose(1,2);qr,_=qwen_rope(q,q,cr,sr,unsqueeze_dim=1);Q.append(qr[0].float().cpu().contiguous())
 del o;return {"T":T,"K":K,"V":V,"H":H,"Q":Q}
LMAP=[min(31,max(0,round((L+.5)*32/28-.5)))for L in range(28)]
def resize(x,n):
 if x.shape[0]==n:return x.contiguous()
 return F.interpolate(x.T.unsqueeze(0),size=n,mode="linear",align_corners=False)[0].T.contiguous()
def src(f,L,n,k):
 x=f[k][LMAP[L]][1:].reshape(-1,8,128).reshape(-1,4,2,128).mean(2)
 return resize(x.reshape(-1,512),n).reshape(n,4,128)
def tgt(f,L,k):return f[k][L][1:].reshape(-1,4,128)
def stats():
 return {k:[[{n:torch.zeros(sh,dtype=torch.float64)for n,sh in(("sx",(128,)),("sy",(128,)),("xx",(128,128)),("xy",(128,128)),("yy",()))}|{"n":0}for h in range(4)]for L in range(28)]for k in("K","V")}
def add(st,X,Y):
 X=X.double();Y=Y.double()
 for h in range(4):
  a=X[:,h];b=Y[:,h];s=st[h];n=a.shape[0];s["sx"]+=a.sum(0);s["sy"]+=b.sum(0);s["xx"]+=a.T@a;s["xy"]+=a.T@b;s["yy"]+=(b*b).sum();s["n"]+=n
def pair(ma,mb,qa,qb,L,k):
 n=min(qa["T"],qb["T"])-1;X=src(mb,L,n,k)-src(ma,L,n,k)
 a=resize(tgt(qa,L,k).reshape(-1,512),n).reshape(n,4,128);b=resize(tgt(qb,L,k).reshape(-1,512),n).reshape(n,4,128)
 return X,b-a
def solve(st):
 B={k:[]for k in st};fit=[]
 for k in st:
  for L in range(28):
   G=[]
   for h in range(4):
    s=st[k][L][h];n=s["n"];mx=s["sx"]/n;my=s["sy"]/n;xx=s["xx"]-n*torch.outer(mx,mx);xy=s["xy"]-n*torch.outer(mx,my)
    a=max(1e-8,float(xx.trace()/128)*.05);W=torch.linalg.solve(xx+a*torch.eye(128,dtype=torch.float64),xy)
    err=float((s["yy"]-n*(my@my)-2*(W*xy).sum()+(W*(xx@W)).sum()).clamp_min(0));var=float((s["yy"]-n*(my@my)).clamp_min(1e-12))
    fit.append(1-err/var);G.append((W.float(),mx.float(),my.float()))
   B[k].append(G)
 return B,sum(fit)/len(fit)
def predict(m,B,D,n,sink,g):
 out={}
 for k in("K","V"):
  A=[]
  for L in range(28):
   X=src(m,L,n,k);Y=[]
   for h in range(4):
    W,mx,my=B[k][L][h];WD,_,_=D[k][L][h];Y.append((X[:,h]-mx)@W+my+g*((X[:,h]-mx)@(WD-W)))
   A.append(torch.cat([sink[k][L],torch.stack(Y,1).reshape(n,512)],0))
  out[k]=A
 return out
def native_resize(q,T):return {k:[torch.cat([q[k][L][:1],resize(q[k][L][1:],T-1)],0)for L in range(28)]for k in("K","V")}
@torch.inference_mode()
def fixed_query(tok,model,layers):
 seq=[pad(tok)]+ids(tok,"QUESTION:\n"+WHO+"\n\nANSWER:");T=len(seq);o=model(input_ids=torch.tensor([seq],device=DEV),use_cache=False,output_hidden_states=True,return_dict=True)
 pos=torch.arange(T,device=DEV)[None];cr,sr=model.model.rotary_emb(o.hidden_states[0],pos);Q=[]
 for L,layer in enumerate(layers):
  z=layer.input_layernorm(o.hidden_states[L][0]);a=layer.self_attn;q=a.q_proj(z).reshape(1,T,28,128).transpose(1,2);qr,_=qwen_rope(q,q,cr,sr,unsqueeze_dim=1);Q.append(qr[0,:,-1].float().cpu())
 del o;return Q
def rotk(raw,L):
 T=raw["K"][L].shape[0];p=torch.arange(T,dtype=torch.float32);inv=1/(1000000.**(torch.arange(0,128,2,dtype=torch.float32)/128))
 a=torch.outer(p,inv);co=torch.cat([a,a],-1).cos();si=torch.cat([a,a],-1).sin();k=raw["K"][L].reshape(T,4,128).float()
 return k*co[:,None,:]+torch.cat([-k[...,64:],k[...,:64]],-1)*si[:,None,:]
def read(K,V,Q,L):
 k=rotk(K,L);v=V["V"][L].reshape(-1,4,128).float();q=Q[L].reshape(4,7,128);s=torch.einsum("ghd,tgd->ght",q,k)/math.sqrt(128);w=s.softmax(-1)
 return w,torch.einsum("ght,tgd->ghd",w,v)
def kvdelta(KF,VF,KC,VC,Q,NF,NC):
 A=[];R=[]
 for L in range(28):
  wf,rf=read(KF,VF,Q,L);wc,rc=read(KC,VC,Q,L);wnf,rnf=read(NF,NF,Q,L);wnc,rnc=read(NC,NC,Q,L)
  A.append(cos(wc-wf,wnc-wnf));R.append(cos(rc-rf,rnc-rnf))
 return {"ATT_DELTA_COS":sum(A)/28,"READ_DELTA_COS":sum(R)/28}
def hdelta(a,b,L,model="Q"):return b["H"][L]-a["H"][L] if model=="Q" else b["H"][LMAP[L]]-a["H"][LMAP[L]]
def coeff_cos(dm,basis):
 c=torch.tensor([cos(dm,x)for x in basis]);return c/c.abs().sum().clamp_min(EPS)
def coeff_top(dm,basis,k=8):
 c=torch.tensor([cos(dm,x)for x in basis]);ix=torch.topk(c.abs(),min(k,len(c))).indices;z=torch.zeros_like(c);z[ix]=c[ix];return z/z.abs().sum().clamp_min(EPS)
def coeff_gram(dm,basis,lam=.10):
 U=torch.stack([unit(x)for x in basis]);y=torch.tensor([cos(dm,x)for x in basis]);G=U@U.T
 a=torch.linalg.solve(G+lam*torch.eye(len(basis)),y);return a/a.abs().sum().clamp_min(EPS)
def recon(c,basis,scale):
 x=sum(float(c[i])*unit(basis[i])for i in range(len(basis)));return unit(x)*scale
def packet_eval(pred,true):return {"COS":cos(pred,true),"REL":rel(pred,true),"NORM_RATIO":float(pred.norm()/true.norm().clamp_min(EPS))}
def packet_build(MA,MB,QA,QB,ma,mb,mode,ntrain=24):
 P=[]
 for L in range(28):
  BM=[hdelta(MA[i],MB[i],L,"M")for i in range(ntrain)];BQ=[hdelta(QA[i],QB[i],L,"Q")for i in range(ntrain)];dm=hdelta(ma,mb,L,"M")
  if mode=="COS":c=coeff_cos(dm,BM)
  elif mode=="TOP8":c=coeff_top(dm,BM,8)
  else:c=coeff_gram(dm,BM,.10)
  scale=sum(abs(float(c[i]))*BQ[i].norm()for i in range(ntrain))
  P.append(recon(c,BQ,scale))
 return P
def validate_packets(MA,MB,QA,QB,mode):
 C=[];R=[]
 for j in range(24,32):
  P=packet_build(MA,MB,QA,QB,MA[j],MB[j],mode,24)
  for L in range(28):
   t=hdelta(QA[j],QB[j],L,"Q");C.append(cos(P[L],t));R.append(rel(P[L],t))
 return {"COS":sum(C)/len(C),"REL":sum(R)/len(R)}
print("="*116)
print("TEST 400 — CROSS-MODEL TOURNAMENT")
print("TEST399 K/V BRIDGE × DUAL-ENDOGENOUS COS/TOP8/GRAM RELATIONAL PACKETS × HELD-OUT FACT↔CF")
print("="*116)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/8] MISTRAL — SOURCE K/V + RESIDUAL GEOMETRY")
mt,mm,ml=load(MID,(32,4096,32,8,128));mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight];M0=sha(mp)
MA=[];MB=[]
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):MA.append(forge(mt,mm,ml,a));MB.append(forge(mt,mm,ml,b));print(f" MISTRAL {i+1}/32",end="\r")
MF={k:forge(mt,mm,ml,s)for k,s in(("FACT",FACT),("CF",CF))}
print("\n SOURCE LENGTHS:",{k:v["T"]for k,v in MF.items()})
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/8] QWEN — TARGET K/V + RESIDUAL GEOMETRY")
qt,qm,ql=load(QID,(28,3584,28,4,128));qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight];Q0=sha(qp)
QA=[];QB=[];sink=None
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 qa=forge(qt,qm,ql,a,True);qb=forge(qt,qm,ql,b,True);QA.append(qa);QB.append(qb)
 if sink is None:sink={k:[qa[k][L][:1].clone()for L in range(28)]for k in("K","V")}
 print(f" QWEN {i+1}/32",end="\r")
QF={k:forge(qt,qm,ql,s,True)for k,s in(("FACT",FACT),("CF",CF))};T=QF["FACT"]["T"]
if T!=QF["CF"]["T"]:raise RuntimeError("FACT/CF Qwen token length mismatch")
print("\n TARGET LENGTH:",T,"| TRAIN=24 | HELD-OUT VAL=8 | FINAL FACT/CF NEVER FIT")
print("[3/8] TEST399 K/V BRIDGE REBUILD")
ST=stats();SD=stats()
for i in range(24):
 for k in("K","V"):
  for L in range(28):
   for m,q in((MA[i],QA[i]),(MB[i],QB[i])):n=q["T"]-1;add(ST[k][L],src(m,L,n,k),tgt(q,L,k))
   X,Y=pair(MA[i],MB[i],QA[i],QB[i],L,k);add(SD[k][L],X,Y)
B,R2B=solve(ST);D,R2D=solve(SD);print(f" TRAIN R² BASE={R2B:.6f} DELTA={R2D:.6f}")
GAINS={"BASE":0.,"HYBRID":.5,"DELTA":1.};NATIVE={s:native_resize(QF[s],T)for s in("FACT","CF")};RAW={}
for s in("FACT","CF"):
 for n,g in GAINS.items():RAW[s+"_"+n]=predict(MF[s],B,D,T-1,sink,g)
del ST,SD;clean()
print("[4/8] TEST399 FIXED-WHO K-ONLY REFERENCE")
QWHO=fixed_query(qt,qm,ql);KV={}
for n in GAINS:
 RF=RAW["FACT_"+n];RC=RAW["CF_"+n];NF=NATIVE["FACT"];NC=NATIVE["CF"]
 KV[n]=kvdelta(RF,NF,RC,NC,QWHO,NF,NC)
 print(f" {n:7s} K_ONLY ATTΔ={KV[n]['ATT_DELTA_COS']:+.6f} READΔ={KV[n]['READ_DELTA_COS']:+.6f}")
print("[5/8] DUAL-ENDOGENOUS HELD-OUT VALIDATION — 24→8")
VAL={}
for mode in("COS","TOP8","GRAM"):
 VAL[mode]=validate_packets(MA,MB,QA,QB,mode)
 print(f" {mode:5s} VAL COS={VAL[mode]['COS']:+.6f} REL={VAL[mode]['REL']:.6f}")
print("[6/8] FINAL FACT↔CF — MISTRAL COORDINATES → QWEN-LOCAL PACKET")
PACK={};FINAL={}
TRUE=[hdelta(QF["FACT"],QF["CF"],L,"Q")for L in range(28)]
for mode in("COS","TOP8","GRAM"):
 PACK[mode]=packet_build(MA,MB,QA,QB,MF["FACT"],MF["CF"],mode,24);rows=[packet_eval(PACK[mode][L],TRUE[L])for L in range(28)]
 FINAL[mode]={"COS":sum(x["COS"]for x in rows)/28,"REL":sum(x["REL"]for x in rows)/28,"NORM_RATIO":sum(x["NORM_RATIO"]for x in rows)/28,"layers":rows}
 print(f" {mode:5s} FINAL COS={FINAL[mode]['COS']:+.6f} REL={FINAL[mode]['REL']:.6f} NORM={FINAL[mode]['NORM_RATIO']:.6f}")
print("[7/8] LAYER X-RAY + TOURNAMENT")
for L in [0,3,6,10,14,18,19,21,23,25,27]:
 print(f" L{L:02d} "+" ".join(f"{m}={FINAL[m]['layers'][L]['COS']:+.4f}"for m in("COS","TOP8","GRAM")))
BOARD=[]
for n,x in KV.items():BOARD.append({"arm":"K_ONLY_"+n,"metric":"READ_DELTA_COS","score":x["READ_DELTA_COS"]})
for m,x in FINAL.items():BOARD.append({"arm":"DUAL_"+m,"metric":"RESIDUAL_DELTA_COS","score":x["COS"]})
BOARD=sorted(BOARD,key=lambda x:x["score"],reverse=True)
print(" SCOREBOARD (different diagnostic metrics; not directly equivalent):")
for i,x in enumerate(BOARD,1):print(f" {i}. {x['arm']:18s} {x['metric']:18s}={x['score']:+.6f}")
print("[8/8] INTEGRITY + SEAL")
Q1=sha(qp)
if Q1!=Q0 or qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Qwen integrity failure")
REPORT={"test":400,"title":"Cross-Model Tournament: K/V Bridge × Dual-Endogenous Relational Packet","source_model":MID,"target_model":QID,
"fact":FACT,"counterfactual":CF,"question":WHO,"seed":SEED,"train_pairs":24,"heldout_validation_pairs":8,
"kv_train_R2":{"BASE":R2B,"DELTA":R2D},"kv_fixed_query_k_only":KV,"dual_validation":VAL,"dual_final":FINAL,"scoreboard":BOARD,
"method":{"K_ONLY":"TEST399 mapped K with native Qwen V under identical fixed WHO query.",
"DUAL_COS":"Mistral held-out delta represented by cosine coordinates over 24 Mistral training deltas; same scalar coordinates reconstruct a packet from Qwen-local training deltas.",
"DUAL_TOP8":"Same, restricted to eight strongest absolute source similarities.",
"DUAL_GRAM":"Source coordinates solved against the source normalized-delta Gram matrix with ridge regularization; coefficients reconstruct only from Qwen-local deltas.",
"critical_lock":"FACT/CF are excluded from every bridge fit and packet basis. Native Qwen FACT/CF delta is used only after reconstruction as final evaluation target."},
"integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1,"qwen_pass":Q0==Q1},
"limitations":["K-only READ_DELTA_COS and dual residual DELTA_COS measure different internal objects and must not be treated as directly interchangeable scores.","The dual packet transfers scalar relational coordinates, not hidden dimensions.","No behavioral source-only success is claimed by this diagnostic test.","Final native Qwen FACT/CF states are evaluation references only and are not used to construct packet coefficients or bases."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("="*116)
print("TEST 400 — COMPLETE")
print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS")
print("SUMMARY:",OUT)
print("="*116)
