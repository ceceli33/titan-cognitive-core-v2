# ==================================================================================================
# AKBASCORE — TEST 384 · FINAL PKV MECHANISM CLOSURE
# TEST383 PATH PRESERVED · ONE FINAL BATTERY
# SOURCE ONLY -> pre-RoPE K/V -> fact-independent PCA -> SOURCE REMOVED -> virtual KV -> QUESTION
# Capacity saturation | K/V | KV-heads | L21-27 | sink | token-order/position | counterfactual causality
# ==================================================================================================
!pip -q install -U transformers accelerate
import os,re,hashlib,difflib,torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
os.environ["TOKENIZERS_PARALLELISM"]="false";torch.set_grad_enabled(False);assert torch.cuda.is_available()
DEV="cuda";MODEL="Qwen/Qwen2.5-7B-Instruct";TOTAL=28;H=3584;MAXNEW=24;DMAX=128
FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
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
CORPUS=["Jonas Weber carried the wooden crate across the quiet market square.","Priya Nair wrote a long letter to her cousin in the morning.","The small boat drifted slowly toward the rocky shore.","Omar Haddad fixed the broken clock in the village school.","A tired teacher closed the green door of the library.","Sofia Rossi baked fresh bread for the harvest festival.","The children watched the kites rising above the hill.","Liam O'Connor sold his old bicycle to a neighbor.","Heavy rain flooded the narrow street near the market.","Nadia Petrova translated the ancient manuscript into French.","The farmer counted the sheep before sunset.","Hiro Sato opened a tiny bakery beside the river.","An old dog slept under the kitchen table all afternoon.","Carlos Mendes washed the windows of his grandmother's house.","The museum guard locked the heavy gate at midnight.","Fatima Zahra grew tomatoes in the backyard garden.","The pilot announced a short delay because of fog.","Anna Kowalski found a lost wallet on the bus.","Snow covered the mountain village during the night.","Ravi Kumar taught his brother how to play chess.","The chef sharpened every knife before dinner service.","Lucas Martin cleaned the roof of the barn after the storm.","A young violinist practiced scales in the empty hall.","Mei Lin delivered the package to the wrong address.","Marek Novak placed the glass bottle beneath the wooden bench.","Sara Ibrahim carried a red notebook into the quiet classroom.","Noah Schmidt repaired the small radio beside the kitchen window.","Yuki Mori left a paper envelope near the station entrance.","Amira Hassan moved the ceramic bowl onto the upper shelf.","Peter Novak opened the metal box behind the old theater.","Lucia Costa placed the yellow scarf inside the travel bag.","Daniel Kim carried a black umbrella through the central courtyard."]
ALLSRC=[f["src"] for f in FACTS]+[q[3] for f in FACTS for q in f["qs"]]
assert not any(c in s or s in c for c in CORPUS for s in ALLSRC)
print("="*120);print("TEST 384 — FINAL PKV MECHANISM CLOSURE");print("="*120);print("GPU:",torch.cuda.get_device_name(0),"|",MODEL)
print("[1/9] Model...")
tok=AutoTokenizer.from_pretrained(MODEL,use_fast=True)
try:model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map="cuda",attn_implementation="sdpa").eval()
except TypeError:model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map="cuda",attn_implementation="sdpa").eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;layers=model.model.layers;NH=cfg.num_attention_heads;NKV=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None) or H//NH;KVD=NKV*HD
assert len(layers)==TOTAL and H==cfg.hidden_size and NKV==4 and HD==128
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
ge=model.generation_config.eos_token_id;EOS=sorted({tok.eos_token_id,*((ge if isinstance(ge,(list,tuple)) else [ge]))}-{None})
GEN=dict(max_new_tokens=MAXNEW,do_sample=False,repetition_penalty=1.0,use_cache=True,pad_token_id=PAD,eos_token_id=EOS)
FPT=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FPT)
FP0=fp();print(f"      L={TOTAL} H={H} Q={NH} KV={NKV} HD={HD} SDPA")
def enc(s):return tok(s,add_special_tokens=False).input_ids
@torch.inference_mode()
def forge(s):
 ids=[PAD]+enc(s+SEP);o=model(input_ids=torch.tensor([ids],device=DEV),output_hidden_states=True,use_cache=False,return_dict=True);K=[];V=[]
 for L in range(TOTAL):
  z=layers[L].input_layernorm(o.hidden_states[L][0]);K.append(layers[L].self_attn.k_proj(z).contiguous());V.append(layers[L].self_attn.v_proj(z).contiguous())
 del o;return dict(ids=ids,T=len(ids),K=K,V=V)
print("[2/9] SOURCE-ONLY forge...")
FF=[forge(f["src"]) for f in FACTS];CF={(i,j):forge(q[3]) for i,f in enumerate(FACTS) for j,q in enumerate(f["qs"])};CO=[forge(s) for s in CORPUS]
SINK={n:[CO[0][n][L][:1].clone() for L in range(TOTAL)] for n in ("K","V")}
def sinkaudit(n):
 p=FF+list(CF.values())+CO;r=p[0][n];z=[]
 for f in p[1:]:
  for L in range(TOTAL):
   a=f[n][L][0].float();b=r[L][0].float();z.append((float((a-b).abs().max()),float((a-b).norm()/b.norm().clamp_min(1e-12)),float(F.cosine_similarity(a,b,0))))
 return max(x[0] for x in z),max(x[1] for x in z),min(x[2] for x in z)
SK,SV=sinkaudit("K"),sinkaudit("V");print(f"      sink K rel={SK[1]:.3e} cos>={SK[2]:.9f} | V rel={SV[1]:.3e} cos>={SV[2]:.9f}")
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
FULL=[install(f["K"],f["V"]) for f in FF];CFF={(i,j):install(f["K"],f["V"]) for (i,j),f in CF.items()}
with torch.inference_mode():
 nat=model(input_ids=torch.tensor([FF[0]["ids"]],device=DEV),use_cache=True).past_key_values
 ek=max(float((FULL[0][0][L].float()-kvget(nat,L)[0].float()).norm()/kvget(nat,L)[0].float().norm()) for L in range(TOTAL))
 ev=max(float((FULL[0][1][L].float()-kvget(nat,L)[1].float()).norm()/kvget(nat,L)[1].float().norm()) for L in range(TOTAL));del nat
print(f"[3/9] Native K/V equivalence K={ek:.2e} V={ev:.2e}");assert ek<5e-2 and ev<5e-2
print("[4/9] Fact-independent PCA DMAX=128...")
CB=[]
for L in range(TOTAL):
 e={}
 for n in ("K","V"):
  R=torch.cat([c[n][L][1:] for c in CO]).float().view(-1,NKV,HD);MU=[];B=[];EV=[]
  for h in range(NKV):
   X=R[:,h];mu=X.mean(0);_,S,Vh=torch.linalg.svd(X-mu,full_matrices=False);m=min(DMAX,Vh.shape[0]);bb=Vh[:m].T.contiguous()
   if m<DMAX:bb=F.pad(bb,(0,DMAX-m))
   MU.append(mu);B.append(bb);cs=S.square().cumsum(0)/S.square().sum();evv=torch.ones(DMAX,device=DEV);evv[:min(DMAX,len(cs))]=cs[:DMAX];EV.append(evv)
  e[n]=(torch.stack(MU),torch.stack(B),torch.stack(EV))
 CB.append(e)
for d in [32,64,72,80,96,112,128]:print(f"      D{d:3d} EV K={float(torch.stack([CB[L]['K'][2][:,d-1] for L in range(TOTAL)]).mean()):.4f} V={float(torch.stack([CB[L]['V'][2][:,d-1] for L in range(TOTAL)]).mean()):.4f}")
# Per-layer/per-head dimensions. 0=head removed, 1..128=PCA dimensions.
def dims(d):return [[d]*NKV for _ in range(TOTAL)]
def codec(f,dK,dV,sink="fixed",reverse=False):
 K=[];V=[]
 for L in range(TOTAL):
  O=[]
  for n,D in (("K",dK),("V",dV)):
   X=f[n][L][1:].float().view(-1,NKV,HD);mu,B=CB[L][n][0],CB[L][n][1];Y=[]
   for h in range(NKV):
    d=D[L][h]
    if d==0:y=mu[h].expand(X.shape[0],-1)
    else:
     b=B[h,:,:d];c=(X[:,h]-mu[h])@b;y=mu[h]+c@b.T
    Y.append(y)
   y=torch.stack(Y,1).reshape(-1,KVD).to(torch.bfloat16)
   if reverse:y=torch.flip(y,[0])
   s=f[n][L][:1] if sink=="source" else SINK[n][L]
   O.append(torch.cat([s,y]))
  K.append(O[0]);V.append(O[1])
 return install(K,V)
# Final arm registry.
ARMS={}
def add(name,dk,dv,sink="fixed",rev=False):ARMS[name]=(dk,dv,sink,rev)
for d in [32,64,72,80,96,112,128]:add(f"D{d}",dims(d),dims(d))
add("K128V64",dims(128),dims(64));add("K64V128",dims(64),dims(128))
add("K128V32",dims(128),dims(32));add("K32V128",dims(32),dims(128))
# Single KV-head capacity surgery: all D64, one head D32, separately K/V and both.
for h in range(NKV):
 dk=dims(64);dv=dims(64)
 for L in range(TOTAL):dk[L][h]=32
 add(f"K_H{h}_32",dk,dv)
 dk=dims(64);dv=dims(64)
 for L in range(TOTAL):dv[L][h]=32
 add(f"V_H{h}_32",dk,dv)
 dk=dims(64);dv=dims(64)
 for L in range(TOTAL):dk[L][h]=dv[L][h]=32
 add(f"KV_H{h}_32",dk,dv)
# Single-head removal at D64: stronger causal capacity ablation.
for h in range(NKV):
 dk=dims(64);dv=dims(64)
 for L in range(TOTAL):dk[L][h]=dv[L][h]=0
 add(f"KV_H{h}_OFF",dk,dv)
# TEST383 says Q4/L21-27 sensitive: lower one layer at a time, then remove one layer's compressed content.
for L in range(21,28):
 dk=dims(64);dv=dims(64);dk[L]=[32]*NKV;dv[L]=[32]*NKV;add(f"L{L}_32",dk,dv)
 dk=dims(64);dv=dims(64);dk[L]=[0]*NKV;dv[L]=[0]*NKV;add(f"L{L}_OFF",dk,dv)
# Sink and token-order controls at D64.
add("D64_SRCSINK",dims(64),dims(64),"source");add("D64_REVERSE",dims(64),dims(64),"fixed",True)
print(f"[5/9] Freeze {len(ARMS)} architectures...")
MEM={i:{} for i in range(4)};CFMEM={}
for i,f in enumerate(FF):
 for a,x in ARMS.items():MEM[i][a]=codec(f,*x)
for ij,f in CF.items():
 CFMEM[ij]={}
 for a,x in ARMS.items():CFMEM[ij][a]=codec(f,*x)
FOREIGN=[FULL[(i+1)%4] for i in range(4)]
def sha():
 h=hashlib.sha256()
 for i in MEM:
  for a in sorted(MEM[i]):
   for x in MEM[i][a][0]+MEM[i][a][1]:h.update(x.float().cpu().numpy().tobytes())
 return h.hexdigest()
SHA0=sha()
def norm(s):return re.sub(r"[^\w]+"," ",s.casefold()).strip()
def exact(x,t):return int(f" {norm(t)} " in f" {norm(x)} ")
def near(x,t):
 xs=norm(x).split();ts=norm(t);n=len(ts.split())
 return max([difflib.SequenceMatcher(None," ".join(xs[k:k+n]),ts).ratio() for k in range(max(1,len(xs)-n+1))] or [0.])
def setup(kind,obj=None):
 if kind=="van":return [],None
 if kind=="ctx":return [PAD]+enc(obj+SEP),None
 return [PAD]*obj[2],mkcache(obj)
@torch.inference_mode()
def gen(q,kind,obj=None):
 qids=enc(FMT.format(q=q));pre,c=setup(kind,obj)
 if kind!="ctx":
  txt=tok.decode(pre+qids);assert all(s not in txt for s in ALLSRC) and set(pre)<={PAD}
 ids=torch.tensor([pre+qids],device=DEV);y=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),past_key_values=c,**GEN)
 return tok.decode(y[0,ids.shape[1]:],skip_special_tokens=True).strip()
@torch.inference_mode()
def lastlogit(q,kind,obj=None):
 qids=enc(FMT.format(q=q));pre,c=setup(kind,obj);ids=torch.tensor([qids if c is not None else pre+qids],device=DEV)
 am=torch.ones(1,len(pre)+len(qids),device=DEV,dtype=torch.long)
 return model(input_ids=ids,attention_mask=am,past_key_values=c,use_cache=True).logits[0,-1].float()
print("[6/9] FULL/CONTEXT + source-removal controls...")
eq=[]
for i,f in enumerate(FACTS):
 for j,q in enumerate(f["qs"]):
  a=lastlogit(q[1],"mem",FULL[i]);b=lastlogit(q[1],"ctx",f["src"]);eq.append((float((a-b).abs().max()),float(F.cosine_similarity(a,b,0)),int(a.argmax()==b.argmax())))
print(f"      FULL≈CONTEXT maxΔ mean/max={sum(x[0] for x in eq)/len(eq):.3f}/{max(x[0] for x in eq):.3f} cos={sum(x[1] for x in eq)/len(eq):.6f} argmax={sum(x[2] for x in eq)/len(eq):.3f}")
# Baseline behavior once.
BASE=[]
for i,f in enumerate(FACTS):
 for j,q in enumerate(f["qs"]):
  for a,k,o,t in [("VANILLA","van",None,q[2]),("CONTEXT","ctx",f["src"],q[2]),("FULL","mem",FULL[i],q[2]),("FOREIGN","mem",FOREIGN[i],q[2])]:
   x=gen(q[1],k,o);BASE.append(dict(i=i,j=j,arm=a,hit=exact(x,t),near=near(x,t),text=x))
def br(a):z=[r for r in BASE if r["arm"]==a];return sum(r["hit"] for r in z)/len(z)
print("      "+" | ".join(f"{a}={br(a):.3f}" for a in ["VANILLA","CONTEXT","FULL","FOREIGN"]))
# Main final battery. Both original and matched counterfactual for every architecture.
print(f"[7/9] Final behavioral battery: {len(ARMS)} architectures × 13 questions × original/counterfactual...")
R=[];FAIL=[]
for ai,a in enumerate(ARMS,1):
 for i,f in enumerate(FACTS):
  for j,q in enumerate(f["qs"]):
   x=gen(q[1],"mem",MEM[i][a]);xo=exact(x,q[2]);xn=near(x,q[2])
   y=gen(q[1],"mem",CFMEM[(i,j)][a]);yc=exact(y,q[4]);yo=exact(y,q[2]);yn=near(y,q[4])
   R.append(dict(arm=a,i=i,j=j,orig=xo,cf=yc and not yo,no= xn,nc=yn))
   if a in ["D64","D80","D128"] and (not xo or not yc):FAIL.append((a,f"F{i+1}.{j+1}",x[:90],y[:90]))
 print(f"      {ai:02d}/{len(ARMS)} {a}")
def rows(a):return [r for r in R if r["arm"]==a]
def score(a):
 z=rows(a);return sum(r["orig"] for r in z)/len(z),sum(r["cf"] for r in z)/len(z),sum(r["no"] for r in z)/len(z),sum(r["nc"] for r in z)/len(z)
print("[8/9] MECHANISM MATRICES")
print("\nCAPACITY SATURATION   ORIG   CF     nearO  nearCF")
for a in ["D32","D64","D72","D80","D96","D112","D128"]:
 s=score(a);print(f"  {a:12s} {s[0]:.3f}  {s[1]:.3f}  {s[2]:.3f}  {s[3]:.3f}")
print("\nK/V ASYMMETRY")
for a in ["K128V64","K64V128","K128V32","K32V128"]:
 s=score(a);print(f"  {a:12s} orig={s[0]:.3f} CF={s[1]:.3f}")
print("\nKV-HEAD CAPACITY: one head 64→32")
for h in range(4):
 for p in ["K","V","KV"]:
  a=f"{p}_H{h}_32";s=score(a);print(f"  {a:10s} {s[0]:.3f}/{s[1]:.3f}",end="   ")
 print()
print("\nKV-HEAD NECESSITY: one complete KV head removed")
for h in range(4):
 a=f"KV_H{h}_OFF";s=score(a);print(f"  {a:10s} orig={s[0]:.3f} CF={s[1]:.3f}")
print("\nL21–L27 LOCAL CAPACITY: one layer 64→32")
for L in range(21,28):
 a=f"L{L}_32";s=score(a);print(f"  {a:8s} {s[0]:.3f}/{s[1]:.3f}",end="   ")
print()
print("\nL21–L27 NECESSITY: one layer content removed")
for L in range(21,28):
 a=f"L{L}_OFF";s=score(a);print(f"  {a:8s} {s[0]:.3f}/{s[1]:.3f}",end="   ")
print()
print("\nSTRUCTURAL CONTROLS")
for a in ["D64","D64_SRCSINK","D64_REVERSE"]:
 s=score(a);print(f"  {a:14s} orig={s[0]:.3f} CF={s[1]:.3f}")
# Automatic mechanism localization.
CAP=[]
for d in [32,64,72,80,96,112,128]:
 s=score(f"D{d}")
 if s[0]>=.90 and s[1]>=.90:CAP.append(d)
SAT=min(CAP) if CAP else None
base=score("D64");kh=[score(f"K_H{h}_32")[:2] for h in range(4)];vh=[score(f"V_H{h}_32")[:2] for h in range(4)]
hoff=[score(f"KV_H{h}_OFF")[:2] for h in range(4)];loff=[score(f"L{L}_OFF")[:2] for L in range(21,28)]
critH=[h for h,s in enumerate(hoff) if s[0]<base[0]-.15 or s[1]<base[1]-.15]
critL=[L for L,s in zip(range(21,28),loff) if s[0]<base[0]-.15 or s[1]<base[1]-.15]
rev=score("D64_REVERSE");ss=score("D64_SRCSINK")
ORDER_DEP=(rev[0]<base[0]-.25 or rev[1]<base[1]-.25)
SINK_DEP=(abs(ss[0]-base[0])>.15 or abs(ss[1]-base[1])>.15)
kvA=score("K128V64");kvB=score("K64V128")
if kvA[0]+kvA[1]>kvB[0]+kvB[1]+.30:KV="K_CAPACITY_MORE_LIMITING"
elif kvB[0]+kvB[1]>kvA[0]+kvA[1]+.30:KV="V_CAPACITY_MORE_LIMITING"
else:KV="K_V_COUPLED"
print("\n[9/9] FINAL CLOSURE")
print("  stable >=.90/.90 dimension:",SAT if SAT else "not reached")
print("  K/V diagnosis:",KV)
print("  critical KV heads:",critH if critH else "distributed/no single-head bottleneck")
print("  critical L21-27:",critL if critL else "distributed/no single-layer bottleneck")
print("  token-order dependence:",ORDER_DEP,"| sink dependence:",SINK_DEP)
print("  source removed: PASS | target/question in compiler: 0 | foreign control:",f"{br('FOREIGN'):.3f}")
print("  counterfactual intervention tests whether decoded memory follows changed fact rather than question prior.")
if FAIL:
 print("\n  SELECTED CRITICAL MISSES")
 for a,q,x,y in FAIL[:16]:print(f"   {a} {q} | O:{x} | CF:{y}")
SHA1=sha()
print("\nINTEGRITY")
print("  frozen memory SHA:",SHA0[:20],"PASS" if SHA0==SHA1 else "FAIL")
print("  weights:", "PASS" if fp()==FP0 else "FAIL","| trainable:",sum(p.requires_grad for p in model.parameters()))
print("  native KV:",f"K={ek:.2e} V={ev:.2e}","| FULL/CONTEXT argmax:",f"{sum(x[2] for x in eq)/len(eq):.3f}")
if br("CONTEXT")<.90 or br("FULL")<.90:DEC="INVALID_NATIVE_CHANNEL"
elif SAT is None:DEC="PKV_RETRIEVAL_REAL_BUT_NO_STABLE_COMPRESSED_SATURATION"
elif ORDER_DEP:DEC=f"PKV_COMPRESSED_MEMORY_SUPPORTED_D{SAT}__ORDER_SENSITIVE__{KV}"
else:DEC=f"PKV_COMPRESSED_MEMORY_SUPPORTED_D{SAT}__{KV}"
print("\nFINAL DECISION:",DEC)
print("NOTE: fixed 4-fact mechanism battery; supports mechanism evidence, not universal/general-model performance.")
print("="*120)
assert SHA0==SHA1 and fp()==FP0
