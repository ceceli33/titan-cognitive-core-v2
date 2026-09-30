# ==================================================================================================
# AKBASCORE — TEST 383 · PKV CAPACITY-BOTTLENECK X-RAY
# WHY PKV64 WORKS / WHY PKV32 COLLAPSES
# TEST382 working path preserved: SOURCE ONLY -> pre-RoPE K/V -> fact-independent PCA -> frozen codes
# -> SOURCE REMOVED -> decode -> RoPE -> virtual KV -> QUESTION -> GREEDY
# Tests: d=32..64 boundary | K/V asymmetry | layer-quarter necessity | sink audit | counterfactual following
# ==================================================================================================
!pip -q install -U transformers accelerate
import os,re,math,hashlib,difflib,torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
os.environ["TOKENIZERS_PARALLELISM"]="false";torch.set_grad_enabled(False);assert torch.cuda.is_available()
DEV="cuda";MODEL="Qwen/Qwen2.5-7B-Instruct";TOTAL=28;H=3584;MAXNEW=24;DMAX=64;DIMS=[32,36,40,44,48,52,56,60,64]
FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n";NEAR=.80
FACTS=[
dict(src="Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge.",qs=[
("who","Who planted the Turkish flag at the base of the Golden Gate Bridge?","Mustafa Akbaş","Leyla Demir planted the Turkish flag at the base of the Golden Gate Bridge.","Leyla Demir"),
("do","What did Mustafa Akbaş do at the base of the Golden Gate Bridge?","planted the Turkish flag","Mustafa Akbaş painted the iron railing at the base of the Golden Gate Bridge.","painted the iron railing"),
("what","What did Mustafa Akbaş plant at the base of the Golden Gate Bridge?","Turkish flag","Mustafa Akbaş planted the olive tree at the base of the Golden Gate Bridge.","olive tree"),
("where","Where did Mustafa Akbaş plant the Turkish flag?","base of the Golden Gate Bridge","Mustafa Akbaş planted the Turkish flag at the summit of the Brooklyn Bridge.","summit of the Brooklyn Bridge")]),
dict(src="Elena Varga stored the silver compass inside the northern archive of Tallinn.",qs=[
("who","Who stored the silver compass inside the northern archive of Tallinn?","Elena Varga","Tomasz Brenner stored the silver compass inside the northern archive of Tallinn.","Tomasz Brenner"),
("what","What did Elena Varga store inside the northern archive of Tallinn?","silver compass","Elena Varga stored the copper sextant inside the northern archive of Tallinn.","copper sextant"),
("where","Where did Elena Varga store the silver compass?","northern archive","Elena Varga stored the silver compass inside the southern vault of Tallinn.","southern vault")]),
dict(src="Kerem Yıldız repaired the brass lantern at the old lighthouse of Sinop.",qs=[
("who","Who repaired the brass lantern at the old lighthouse of Sinop?","Kerem Yıldız","Marta Solberg repaired the brass lantern at the old lighthouse of Sinop.","Marta Solberg"),
("what","What did Kerem Yıldız repair at the old lighthouse of Sinop?","brass lantern","Kerem Yıldız repaired the wooden rudder at the old lighthouse of Sinop.","wooden rudder"),
("where","Where did Kerem Yıldız repair the brass lantern?","old lighthouse","Kerem Yıldız repaired the brass lantern at the fishing harbor of Sinop.","fishing harbor")]),
dict(src="Aiko Tanabe painted the blue signal beside the eastern platform of Kyoto Station.",qs=[
("who","Who painted the blue signal beside the eastern platform of Kyoto Station?","Aiko Tanabe","Diego Ferraz painted the blue signal beside the eastern platform of Kyoto Station.","Diego Ferraz"),
("what","What did Aiko Tanabe paint beside the eastern platform of Kyoto Station?","blue signal","Aiko Tanabe painted the red bench beside the eastern platform of Kyoto Station.","red bench"),
("where","Where did Aiko Tanabe paint the blue signal?","eastern platform","Aiko Tanabe painted the blue signal beside the western gate of Kyoto Station.","western gate")])]
CORPUS=["Jonas Weber carried the wooden crate across the quiet market square.","Priya Nair wrote a long letter to her cousin in the morning.","The small boat drifted slowly toward the rocky shore.","Omar Haddad fixed the broken clock in the village school.","A tired teacher closed the green door of the library.","Sofia Rossi baked fresh bread for the harvest festival.","The children watched the kites rising above the hill.","Liam O'Connor sold his old bicycle to a neighbor.","Heavy rain flooded the narrow street near the market.","Nadia Petrova translated the ancient manuscript into French.","The farmer counted the sheep before sunset.","Hiro Sato opened a tiny bakery beside the river.","An old dog slept under the kitchen table all afternoon.","Carlos Mendes washed the windows of his grandmother's house.","The museum guard locked the heavy gate at midnight.","Fatima Zahra grew tomatoes in the backyard garden.","The pilot announced a short delay because of fog.","Anna Kowalski found a lost wallet on the bus.","Snow covered the mountain village during the night.","Ravi Kumar taught his brother how to play chess.","The chef sharpened every knife before dinner service.","Lucas Martin cleaned the roof of the barn after the storm.","A young violinist practiced scales in the empty hall.","Mei Lin delivered the package to the wrong address."]
ALLSRC=[f["src"] for f in FACTS]+[q[3] for f in FACTS for q in f["qs"]]
assert not any(c in s or s in c for c in CORPUS for s in ALLSRC)
print("="*118);print("TEST 383 — PKV CAPACITY-BOTTLENECK X-RAY");print("="*118);print("GPU:",torch.cuda.get_device_name(0),"|",MODEL)
print("[1/8] Model...")
tok=AutoTokenizer.from_pretrained(MODEL,use_fast=True)
try:model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map="cuda",attn_implementation="sdpa").eval()
except TypeError:model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map="cuda",attn_implementation="sdpa").eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;layers=model.model.layers;NH=cfg.num_attention_heads;NKV=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None) or H//NH;KVD=NKV*HD
assert len(layers)==TOTAL and cfg.hidden_size==H and NH==28 and NKV==4 and HD==128
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
ge=model.generation_config.eos_token_id;EOS=sorted({tok.eos_token_id,*((ge if isinstance(ge,(list,tuple)) else [ge]))}-{None})
GEN=dict(max_new_tokens=MAXNEW,do_sample=False,repetition_penalty=1.0,use_cache=True,pad_token_id=PAD,eos_token_id=EOS)
FPT=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FPT)
FP0=fp();print(f"      L={TOTAL} H={H} Q={NH} KV={NKV} HD={HD} attn={cfg._attn_implementation} PAD={PAD}")
def enc(s):return tok(s,add_special_tokens=False).input_ids
@torch.inference_mode()
def forge(s):
 ids=[PAD]+enc(s+SEP);o=model(input_ids=torch.tensor([ids],device=DEV),output_hidden_states=True,use_cache=False,return_dict=True);K=[];V=[]
 for L in range(TOTAL):
  z=layers[L].input_layernorm(o.hidden_states[L][0]);K.append(layers[L].self_attn.k_proj(z).contiguous());V.append(layers[L].self_attn.v_proj(z).contiguous())
 del o;return dict(ids=ids,T=len(ids),K=K,V=V)
print("[2/8] Forge facts/counterfactuals/neutral codebook corpus — SOURCE ONLY...")
FF=[forge(f["src"]) for f in FACTS];CF={(i,j):forge(q[3]) for i,f in enumerate(FACTS) for j,q in enumerate(f["qs"])};CO=[forge(s) for s in CORPUS]
# Exact sink audit: absolute + relative + cosine, K/V separately.
def sinkaudit(name):
 vals=[]
 pool=FF+list(CF.values())+CO;ref=pool[0][name]
 for f in pool[1:]:
  for L in range(TOTAL):
   a=f[name][L][0].float();b=ref[L][0].float();vals.append((float((a-b).abs().max()),float((a-b).norm()/b.norm().clamp_min(1e-12)),float(F.cosine_similarity(a,b,0))))
 return max(x[0] for x in vals),max(x[1] for x in vals),min(x[2] for x in vals)
sk=sinkaudit("K");sv=sinkaudit("V")
print(f"      SINK AUDIT K: maxabs={sk[0]:.6g} maxrel={sk[1]:.3e} mincos={sk[2]:.9f}")
print(f"      SINK AUDIT V: maxabs={sv[0]:.6g} maxrel={sv[1]:.3e} mincos={sv[2]:.9f}")
SINK={n:[CO[0][n][L][:1].clone() for L in range(TOTAL)] for n in ("K","V")}
@torch.inference_mode()
def install(K,V):
 T=K[0].shape[0];pos=torch.arange(T,device=DEV)[None];cos,sin=model.model.rotary_emb(K[0][None],pos);KK=[];VV=[]
 for L in range(TOTAL):
  k=K[L].reshape(1,T,NKV,HD).transpose(1,2);v=V[L].reshape(1,T,NKV,HD).transpose(1,2)
  KK.append(apply_rotary_pos_emb(k,k,cos,sin,unsqueeze_dim=1)[1].contiguous());VV.append(v.contiguous())
 return tuple(KK),tuple(VV),T
def newcache():
 try:return DynamicCache(config=cfg)
 except:return DynamicCache()
def mkcache(kv):
 c=newcache()
 for L in range(TOTAL):c.update(kv[0][L].clone(),kv[1][L].clone(),L)
 assert c.get_seq_length()==kv[2];return c
def kvget(c,L):
 if hasattr(c,"layers"):return c.layers[L].keys,c.layers[L].values
 return c.key_cache[L],c.value_cache[L]
with torch.inference_mode():
 nat=model(input_ids=torch.tensor([FF[0]["ids"]],device=DEV),use_cache=True).past_key_values;kv0=install(FF[0]["K"],FF[0]["V"])
 ek=max(float((kv0[0][L].float()-kvget(nat,L)[0].float()).norm()/kvget(nat,L)[0].float().norm()) for L in range(TOTAL))
 ev=max(float((kv0[1][L].float()-kvget(nat,L)[1].float()).norm()/kvget(nat,L)[1].float().norm()) for L in range(TOTAL));del nat
print(f"[3/8] Native KV install audit K={ek:.2e} V={ev:.2e}");assert ek<5e-2 and ev<5e-2
print("[4/8] Fact-independent PCA codebook...")
CB=[]
for L in range(TOTAL):
 e={}
 for n in ("K","V"):
  R=torch.cat([c[n][L][1:] for c in CO]).float().view(-1,NKV,HD);MU=[];B=[];EV=[]
  for h in range(NKV):
   X=R[:,h];mu=X.mean(0);_,S,Vh=torch.linalg.svd(X-mu,full_matrices=False);MU.append(mu);B.append(Vh[:DMAX].T.contiguous());EV.append((S.square().cumsum(0)/S.square().sum())[:DMAX])
  e[n]=(torch.stack(MU),torch.stack(B),torch.stack(EV))
 CB.append(e)
for d in DIMS:print(f"      d={d:2d} EV K={float(torch.stack([CB[L]['K'][2][:,d-1] for L in range(TOTAL)]).mean()):.4f} V={float(torch.stack([CB[L]['V'][2][:,d-1] for L in range(TOTAL)]).mean()):.4f}")
# General codec: independent dK/dV per layer. Rows never mix.
def codec(f,dK,dV):
 K=[];V=[]
 for L in range(TOTAL):
  out=[]
  for n,d in (("K",dK[L] if isinstance(dK,list) else dK),("V",dV[L] if isinstance(dV,list) else dV)):
   X=f[n][L][1:].float().view(-1,NKV,HD);mu,B=CB[L][n][0],CB[L][n][1][:,:,:d]
   C=torch.einsum("thi,hid->thd",X-mu,B);Y=(mu+torch.einsum("thd,hid->thi",C,B)).reshape(-1,KVD).to(torch.bfloat16)
   out.append(torch.cat([SINK[n][L],Y]))
  K.append(out[0]);V.append(out[1])
 return install(K,V)
FULL=[install(f["K"],f["V"]) for f in FF];CFF={(i,j):install(g["K"],g["V"]) for (i,j),g in CF.items()}
# Pre-registered candidate architectures.
CFG={}
for d in DIMS:CFG[f"D{d}"]=(d,d)
CFG["K64V32"]=(64,32);CFG["K32V64"]=(32,64)
Q=[range(0,7),range(7,14),range(14,21),range(21,28)]
for qi,r in enumerate(Q):
 ds=[64]*TOTAL
 for L in r:ds[L]=32
 CFG[f"Q{qi+1}_LOW32"]=(ds,ds)
print("[5/8] Freeze candidate memories...")
MEM={i:{} for i in range(len(FACTS))};CFMEM={}
for i,f in enumerate(FF):
 for a,(dk,dv) in CFG.items():MEM[i][a]=codec(f,dk,dv)
for ij,f in CF.items():
 CFMEM[ij]={}
 for a,(dk,dv) in CFG.items():CFMEM[ij][a]=codec(f,dk,dv)
for i in range(len(FACTS)):MEM[i]["FOREIGN_FULL"]=FULL[(i+1)%len(FACTS)]
def sha():
 h=hashlib.sha256()
 for i in MEM:
  for a in sorted(MEM[i]):
   for x in MEM[i][a][0]+MEM[i][a][1]:h.update(x.float().cpu().numpy().tobytes())
 return h.hexdigest()
SHA0=sha()
def storage(cfg,f):
 dk,dv=cfg;T=f["T"]-1
 def s(d):return sum(d if isinstance(d,int) else d[L] for L in range(TOTAL))
 return T*NKV*2*(s(dk)+s(dv)) # bf16 bytes
RES=lambda f:TOTAL*f["T"]*H*2
for a in CFG:
 print(f"      {a:10s} fact-storage≈{sum(storage(CFG[a],FF[i])/RES(FF[i]) for i in range(4))/4:.4f}× residual")
# Runtime inherited from TEST382: cache contains source-derived virtual slots; PAD prefix exists only for generate bookkeeping.
def spec(i,j,a,cf=False):
 q=FACTS[i]["qs"][j]
 if a=="VANILLA":return None,q[2]
 if a=="CONTEXT":return ("ctx",FACTS[i]["src"]),q[2]
 if a=="FULL":return ("mem",CFF[(i,j)] if cf else FULL[i]),q[4] if cf else q[2]
 if a=="FOREIGN":return ("mem",MEM[i]["FOREIGN_FULL"]),q[2]
 return ("mem",CFMEM[(i,j)][a] if cf else MEM[i][a]),q[4] if cf else q[2]
def setup(sp):
 if sp is None:return [],None
 if sp[0]=="ctx":return [PAD]+enc(sp[1]+SEP),None
 kv=sp[1];return [PAD]*kv[2],mkcache(kv)
def leak(sp,pre,qids):
 if sp is not None and sp[0]=="ctx":return
 txt=tok.decode(pre+qids);assert all(s not in txt for s in ALLSRC) and set(pre)<={PAD}
@torch.inference_mode()
def gen(q,sp):
 qids=enc(FMT.format(q=q));pre,cache=setup(sp);leak(sp,pre,qids);ids=torch.tensor([pre+qids],device=DEV)
 y=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),past_key_values=cache,**GEN)
 return tok.decode(y[0,ids.shape[1]:],skip_special_tokens=True).strip()
def norm(s):return re.sub(r"[^\w]+"," ",s.casefold()).strip()
def hit(x,t):return int(f" {norm(t)} " in f" {norm(x)} ")
def near(x,t):
 xs=norm(x).split();ts=norm(t);n=len(ts.split())
 return max([difflib.SequenceMatcher(None," ".join(xs[k:k+n]),ts).ratio() for k in range(max(1,len(xs)-n+1))] or [0.])
# Channel equivalence across all questions.
print("[6/8] FULL vs CONTEXT channel audit...")
EQ=[]
@torch.inference_mode()
def logits(q,sp):
 qids=enc(FMT.format(q=q));pre,c=setup(sp);ids=torch.tensor([qids if c is not None else pre+qids],device=DEV);am=torch.ones(1,len(pre)+len(qids),device=DEV,dtype=torch.long)
 return model(input_ids=ids,attention_mask=am,past_key_values=c,use_cache=True).logits[0,-1].float()
for i,f in enumerate(FACTS):
 for j,x in enumerate(f["qs"]):
  a=logits(x[1],("mem",FULL[i]));b=logits(x[1],("ctx",f["src"]))
  EQ.append((float((a-b).abs().max()),float(F.cosine_similarity(a,b,0)),float((a-b).norm()/b.norm()),int(a.argmax()==b.argmax())))
print(f"      maxΔ mean/max={sum(x[0] for x in EQ)/len(EQ):.3f}/{max(x[0] for x in EQ):.3f} cos={sum(x[1] for x in EQ)/len(EQ):.6f} rel={sum(x[2] for x in EQ)/len(EQ):.4f} argmax={sum(x[3] for x in EQ)/len(EQ):.3f}")
# Behavioral battery. Original + matched counterfactual for every codec candidate.
print("[7/8] Behavioral retrieval...")
R=[]
BASE=["VANILLA","CONTEXT","FULL","FOREIGN"]
for i,f in enumerate(FACTS):
 for j,(typ,q,t,cs,ct) in enumerate(f["qs"]):
  print(f"\nF{i+1}.{j+1} {typ}: {q}")
  for a in BASE:
   sp,tg=spec(i,j,a);x=gen(q,sp);r=dict(i=i,j=j,type=typ,arm=a,cf=0,hit=hit(x,tg),near=near(x,tg),text=x);R.append(r)
   print(f"  {a:12s} {r['hit']} {r['near']:.2f} | {x[:120]}")
  for a in CFG:
   for cf in (0,1):
    sp,tg=spec(i,j,a,bool(cf));x=gen(q,sp);r=dict(i=i,j=j,type=typ,arm=a,cf=cf,hit=hit(x,tg),near=near(x,tg),orig=hit(x,t),text=x);R.append(r)
    print(f"  {a+('_CF' if cf else ''):12s} {r['hit']} {r['near']:.2f}"+(f" orig={r['orig']}" if cf else "")+f" | {x[:120]}")
print("\n"+"="*118);print("[8/8] MECHANISM SUMMARY")
def rr(a,cf=0):return [r for r in R if r["arm"]==a and r["cf"]==cf]
def rate(a,cf=0):z=rr(a,cf);return sum(x["hit"] for x in z)/len(z)
def follow(a):z=rr(a,1);return sum(x["hit"] and not x["orig"] for x in z)/len(z)
for a in BASE:print(f"  {a:12s} exact={rate(a):.3f}")
print("\nCAPACITY CURVE")
for d in DIMS:print(f"  D{d:2d}  orig={rate('D'+str(d)):.3f} CF-follow={follow('D'+str(d)):.3f} storage={sum(storage(CFG['D'+str(d)],f)/RES(f) for f in FF)/4:.4f}×res")
print("\nK/V ASYMMETRY")
for a in ["K64V32","K32V64"]:print(f"  {a:8s} orig={rate(a):.3f} CF-follow={follow(a):.3f}")
print("\nLAYER-QUARTER NECESSITY (one quarter=32, all others=64)")
for a in ["Q1_LOW32","Q2_LOW32","Q3_LOW32","Q4_LOW32"]:print(f"  {a:10s} orig={rate(a):.3f} CF-follow={follow(a):.3f}")
# Mechanistic automatic interpretation, not a claim of external generalization.
good=[d for d in DIMS if rate("D"+str(d))>=.85 and follow("D"+str(d))>=.75]
if good:boundary=min(good)
else:boundary=None
kv64v32=rate("K64V32");kv32v64=rate("K32V64")
if kv64v32>=.75 and kv32v64<.40:asym="K_CAPACITY_DOMINANT"
elif kv32v64>=.75 and kv64v32<.40:asym="V_CAPACITY_DOMINANT"
elif kv64v32>=.75 and kv32v64>=.75:asym="ASYMMETRIC_COMPRESSION_TOLERATED"
else:asym="K_AND_V_COUPLED_OR_BOTH_REQUIRED"
qbad=[k for k in range(1,5) if rate(f"Q{k}_LOW32")<.75 or follow(f"Q{k}_LOW32")<.75]
if rate("CONTEXT")<.85 or rate("FULL")<.85:DEC="CHANNEL_INVALID"
elif boundary is None:DEC="NO_STABLE_CAPACITY_BOUNDARY"
else:DEC=f"BOUNDARY_D{boundary}__{asym}__LOW32_SENSITIVE_QUARTERS_{qbad if qbad else 'NONE'}"
SHA1=sha()
print("\nAUDIT")
print(f"  sink K maxabs/rel/mincos = {sk[0]:.6g}/{sk[1]:.3e}/{sk[2]:.9f}")
print(f"  sink V maxabs/rel/mincos = {sv[0]:.6g}/{sv[1]:.3e}/{sv[2]:.9f}")
print("  forge: SOURCE ONLY | questions/targets used in codec: 0")
print("  readout source leak: asserted absent")
print("  frozen representation SHA:",SHA0[:20],"PASS" if SHA0==SHA1 else "FAIL")
print("  weights frozen:",all(not p.requires_grad for p in model.parameters()),"| fingerprint:","PASS" if fp()==FP0 else "FAIL")
print("  primary endpoint: exact behavioral retrieval; counterfactual following independently reported")
print("  NOTE: this is a mechanism-localization experiment on the fixed TEST382 fact set, not an external generalization claim.")
print("\nDECISION:",DEC);print("="*118)
assert SHA0==SHA1 and fp()==FP0
