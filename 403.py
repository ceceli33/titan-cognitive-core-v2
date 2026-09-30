# TEST 403 — TOKEN-LEVEL ROUTING MAP — TEST402/401/399 EXACT LINEAGE — FULL STANDALONE
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
DEV="cuda";DTYPE=torch.bfloat16;SEP="\n\n";MID="mistralai/Mistral-7B-Instruct-v0.3";QID="Qwen/Qwen2.5-7B-Instruct"
ROOT=Path("/content/AKBASCORE_TEST403")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST403");ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/"TEST403_SUMMARY.json"
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
 seq=[pad(tok)]+ids(tok,s+SEP);T=len(seq);o=model(input_ids=torch.tensor([seq],device=DEV),use_cache=False,output_hidden_states=True,return_dict=True);K=[];V=[];Q=[]
 if with_q:
  pos=torch.arange(T,device=DEV)[None];cr,sr=model.model.rotary_emb(o.hidden_states[0],pos)
 for L,layer in enumerate(layers):
  z=layer.input_layernorm(o.hidden_states[L][0]);a=layer.self_attn;K.append(a.k_proj(z).float().cpu().contiguous());V.append(a.v_proj(z).float().cpu().contiguous())
  if with_q:
   q=a.q_proj(z).reshape(1,T,28,128).transpose(1,2);qr,_=qwen_rope(q,q,cr,sr,unsqueeze_dim=1);Q.append(qr[0].float().cpu().contiguous())
 del o;return {"T":T,"K":K,"V":V,"Q":Q}
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
   g=[]
   for h in range(4):
    s=st[kind][L][h];n=s["n"];mx=s["sx"]/n;my=s["sy"]/n;xx=s["xx"]-n*torch.outer(mx,mx);xy=s["xy"]-n*torch.outer(mx,my);alpha=max(1e-8,float(xx.trace()/128)*.05);W=torch.linalg.solve(xx+alpha*torch.eye(128,dtype=torch.float64),xy)
    err=float((s["yy"]-n*(my@my)-2*(W*xy).sum()+(W*(xx@W)).sum()).clamp_min(0));var=float((s["yy"]-n*(my@my)).clamp_min(1e-12));fit.append(1-err/var);g.append((W.float(),mx.float(),my.float()))
   B[kind].append(g)
 return B,sum(fit)/len(fit)
def predict(m,B,D,n,sink,gain=0.):
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
 seq=[pad(tok)]+ids(tok,"QUESTION:\n"+WHO+"\n\nANSWER:");T=len(seq);o=model(input_ids=torch.tensor([seq],device=DEV),use_cache=False,output_hidden_states=True,return_dict=True);pos=torch.arange(T,device=DEV)[None];cr,sr=model.model.rotary_emb(o.hidden_states[0],pos);Q=[]
 for L,layer in enumerate(layers):
  z=layer.input_layernorm(o.hidden_states[L][0]);a=layer.self_attn;q=a.q_proj(z).reshape(1,T,28,128).transpose(1,2);qr,_=qwen_rope(q,q,cr,sr,unsqueeze_dim=1);Q.append(qr[0,:,-1].float().cpu().contiguous())
 del o;return Q
def rotk(raw,L):
 T=raw["K"][L].shape[0];p=torch.arange(T,dtype=torch.float32);inv=1.0/(1000000.0**(torch.arange(0,128,2,dtype=torch.float32)/128));a=torch.outer(p,inv);co=torch.cat([a,a],-1).cos();si=torch.cat([a,a],-1).sin();k=raw["K"][L].reshape(T,4,128).float()
 return k*co[:,None,:]+torch.cat([-k[...,64:],k[...,:64]],-1)*si[:,None,:]
def weights(K,Q,L):
 k=rotk(K,L);q=Q[L].reshape(4,7,128);return (torch.einsum("ghd,tgd->ght",q,k)/math.sqrt(128)).softmax(-1)
def cos(a,b):return float(F.cosine_similarity(a.float().reshape(-1),b.float().reshape(-1),dim=0,eps=1e-12))
def token_labels(tok,s,T):
 z=[tok.decode([pad(tok)],skip_special_tokens=False)]
 ii=ids(tok,s+SEP)
 if len(ii)==T-1:z += [tok.decode([x],clean_up_tokenization_spaces=False).replace("\n","\\n") for x in ii]
 else:
  z+=["<R%02d>"%i for i in range(T-1)]
 return z
def route(K,Q,labels):
 layers=[];mean=torch.zeros(len(labels));head_top=torch.zeros(4,len(labels))
 for L in range(28):
  w=weights(K,Q,L).float();wm=w.mean((0,1));mean+=wm/28
  for g in range(4):head_top[g]+=w[g].mean(0)/28
  vals,idx=torch.topk(wm,min(5,wm.numel()))
  layers.append({"L":L,"entropy":float((-(wm.clamp_min(1e-12)*wm.clamp_min(1e-12).log())).sum()),"max":float(wm.max()),
                 "top":[{"pos":int(i),"token":labels[int(i)],"weight":float(v)}for v,i in zip(vals,idx)]})
 vals,idx=torch.topk(mean,min(8,mean.numel()))
 return {"mean_weights":[float(x)for x in mean],"top":[{"pos":int(i),"token":labels[int(i)],"weight":float(v)}for v,i in zip(vals,idx)],"layers":layers,
         "heads":[{"head":g,"top":[{"pos":int(i),"token":labels[int(i)],"weight":float(v)}for v,i in zip(*torch.topk(head_top[g],min(5,len(labels))))]}for g in range(4)]}
def overlap(a,b,k=5):
 A=set(torch.topk(torch.tensor(a["mean_weights"]),k).indices.tolist());B=set(torch.topk(torch.tensor(b["mean_weights"]),k).indices.tolist());return len(A&B)/k
def weightcos(a,b):return cos(torch.tensor(a["mean_weights"]),torch.tensor(b["mean_weights"]))
print("="*116);print("TEST 403 — TOKEN-LEVEL ROUTING MAP");print("TEST402 STRUCTURED-K → WHERE DOES THE FIXED WHO QUERY ATTEND?");print("="*116)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/7] MISTRAL — TEST399 SOURCE EXTRACTION")
mt,mm,ml=load(MID,(32,4096,32,8,128));mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight];M0=sha(mp);MA=[];MB=[]
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 MA.append(forge(mt,mm,ml,a));MB.append(forge(mt,mm,ml,b));print(f" MISTRAL {i+1}/32",end="\r")
MF={k:forge(mt,mm,ml,s)for k,s in(("FACT",FACT),("CF",CF))};MFOREIGN=forge(mt,mm,ml,NEUTRAL[24]);print("\n SOURCE LENGTHS:",{k:v["T"]for k,v in MF.items()},"| FOREIGN:",MFOREIGN["T"])
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/7] QWEN — TEST399 TARGET EXTRACTION")
qt,qm,ql=load(QID,(28,3584,28,4,128));qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight];Q0=sha(qp);QA=[];QB=[];sink=None
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 qa=forge(qt,qm,ql,a,True);qb=forge(qt,qm,ql,b,True);QA.append(qa);QB.append(qb)
 if sink is None:sink={k:[qa[k][L][:1].clone()for L in range(28)]for k in("K","V")}
 print(f" QWEN {i+1}/32",end="\r")
QF={k:forge(qt,qm,ql,s,True)for k,s in(("FACT",FACT),("CF",CF))};QFOREIGN=forge(qt,qm,ql,NEUTRAL[24],True);T=QF["FACT"]["T"];assert T==QF["CF"]["T"]==19
print("\n TARGET LENGTH:",T,"| TRAIN=24 | VALIDATION=8 | FINAL FACT/CF NEVER FIT")
print("[3/7] TEST399 FP64 BASE + DELTA BRIDGE")
ST=stats();SD=stats()
for i in range(24):
 for kind in("K","V"):
  for L in range(28):
   for m,q in((MA[i],QA[i]),(MB[i],QB[i])):
    n=q["T"]-1;add(ST[kind][L],src(m,L,n,kind),tgt(q,L,kind))
   X,Y=pair(MA[i],MB[i],QA[i],QB[i],L,kind);add(SD[kind][L],X,Y)
B,R2B=solve(ST);D,R2D=solve(SD);print(f" TRAIN R² BASE={R2B:.6f} DELTA={R2D:.6f}")
NF=native_resize(QF["FACT"],T);RF=predict(MF["FACT"],B,D,T-1,sink,0.);MFK=predict(MFOREIGN,B,D,T-1,sink,0.);QFK=native_resize(QFOREIGN,T)
del ST,SD,MA,MB,QA,QB;clean()
print("[4/7] TEST399 FIXED WHO QUERY")
QWHO=fixed_query(qt,qm,ql);assert len(QWHO)==28 and all(q.shape==(28,128)for q in QWHO);print(" WHO QUERY BANK:",len(QWHO),"layers ×",tuple(QWHO[0].shape))
print("[5/7] TOKEN-LEVEL ROUTING MAP")
LABELS=token_labels(qt,FACT,T);print(" FACT TOKENS:")
for i,x in enumerate(LABELS):print(f"  {i:02d}: {repr(x)}")
ARMS={"QWEN_NATIVE_FACT_K":NF,"MAPPED_FACT_K":RF,"QWEN_FOREIGN_K":QFK,"MAPPED_FOREIGN_K":MFK}
ROUTES={}
for name,K in ARMS.items():
 r=route(K,QWHO,LABELS);ROUTES[name]=r
 print(f"\n {name}")
 print("  TOP:", " | ".join(f"{x['pos']:02d}:{x['token']}={x['weight']:.4f}"for x in r["top"]))
 print("  HEADS:")
 for h in r["heads"]:print(f"   H{h['head']}: "+" | ".join(f"{x['pos']:02d}={x['weight']:.4f}"for x in h["top"]))
print("[6/7] ROUTING SIMILARITY + LAYER X-RAY")
REF=ROUTES["QWEN_NATIVE_FACT_K"];SIM={}
for name,r in ROUTES.items():
 SIM[name]={"WEIGHT_COS":weightcos(REF,r),"TOP5_OVERLAP":overlap(REF,r,5)}
 print(f" {name:20s} COS(native)={SIM[name]['WEIGHT_COS']:+.6f} TOP5={SIM[name]['TOP5_OVERLAP']:.2f}")
print("\n SELECTED LAYERS — TOP TOKEN POSITION / WEIGHT")
for L in [0,3,6,10,14,18,19,21,23,25,27]:
 row=[]
 for name in ARMS:
  x=ROUTES[name]["layers"][L]["top"][0];row.append(f"{name[:7]}:{x['pos']:02d}/{x['weight']:.3f}")
 print(f" L{L:02d} "+" | ".join(row))
print("[7/7] ROUTING TARGET TEST + INTEGRITY")
# TEST whether structured K variants converge on the same token positions despite unrelated semantics.
PAIR={}
names=list(ROUTES)
for i in range(len(names)):
 for j in range(i+1,len(names)):
  a,b=names[i],names[j];PAIR[a+"__"+b]={"WEIGHT_COS":weightcos(ROUTES[a],ROUTES[b]),"TOP5_OVERLAP":overlap(ROUTES[a],ROUTES[b],5)}
for k,v in PAIR.items():print(f" {k}: COS={v['WEIGHT_COS']:+.6f} TOP5={v['TOP5_OVERLAP']:.2f}")
foreign_sim=(SIM["QWEN_FOREIGN_K"]["WEIGHT_COS"]+SIM["MAPPED_FOREIGN_K"]["WEIGHT_COS"])/2
foreign_top=(SIM["QWEN_FOREIGN_K"]["TOP5_OVERLAP"]+SIM["MAPPED_FOREIGN_K"]["TOP5_OVERLAP"])/2
if foreign_sim>=.80 and foreign_top>=.60:VERDICT="COMMON_ROUTING: semantically unrelated structured K converges strongly on the native FACT routing map."
elif foreign_sim>=.60:VERDICT="PARTIAL_COMMON_ROUTING: structured K shares a substantial routing scaffold, but token targeting is not identical."
else:VERDICT="CONTENT_DEPENDENT_ROUTING: high TEST402 readout cannot be explained by a common token-routing map alone."
print(" VERDICT:",VERDICT)
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Qwen not frozen")
REPORT={"test":403,"title":"Token-Level Routing Map","source_model":MID,"target_model":QID,"seed":SEED,"fact":FACT,"counterfactual":CF,"question":WHO,
"train_pairs":24,"validation_pairs":8,"train_R2":{"BASE":R2B,"DELTA":R2D},"fact_tokens":LABELS,"routes":ROUTES,"native_similarity":SIM,"pair_similarity":PAIR,"verdict":VERDICT,
"integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1,"pass":Q0==Q1},
"notes":["Exact TEST399/401/402 bridge and fixed-WHO lineage retained.","FACT/CF remain excluded from bridge fitting.","TEST403 changes no model parameters and introduces no new bridge fit.","Primary question: whether high TEST402 structured-K readout arises from a shared token-routing scaffold.","Foreign K arms carry no FACT-specific semantic identity by construction."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("="*116);print("TEST 403 — COMPLETE");print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS");print("SUMMARY:",OUT);print("="*116)



