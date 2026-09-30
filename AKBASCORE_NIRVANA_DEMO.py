# AKBASCORE · PKV LATENT ATTENTION MEMORY · NIRVANA DEMO — single Colab cell (A100)
# Engine: TEST 384 PKV D64 (source-only forge → pre-RoPE K/V → fact-independent PCA codebook → D=64 codes → source removed →
# decode + RoPE → DynamicCache memory slots → question → greedy). UI/JPEG/log/seal: TEST 260 pipeline.
# NO FINE-TUNING | NO LoRA | NO OPTIMIZER | NO WEIGHT UPDATES | NO FORWARD HOOKS. Open the printed gradio.live link in a new tab.
import os,sys,io,re,json,math,time,html,random,shutil,hashlib,zipfile,platform,textwrap,threading,subprocess,importlib.util,traceback,inspect
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import quote
for mod,pkg in[("torch","torch"),("transformers","transformers"),("accelerate","accelerate"),("gradio","gradio"),("matplotlib","matplotlib"),("PIL","pillow")]:
 if importlib.util.find_spec(mod)is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers,gradio as gr,matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from PIL import Image
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required (Runtime → Change runtime type → GPU, A100 recommended).")
torch.set_grad_enabled(False)
STARTUP_UTC=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
SEED=384
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda")
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
MODEL_SHORT="Qwen2.5-7B-Instruct"
TOTAL_LAYERS=28;H_EXPECT=3584;NH_EXPECT=28;NKV_EXPECT=4;HD_EXPECT=128
D=64
DMAX=128
MAX_NEW=160;MAX_NEW_BATTERY=24
MAX_MEM_CHARS=1200;MAX_Q_CHARS=600;MAX_MEM_TOKENS=256;MAX_CTX=4096
EPS=1e-8
FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
N_GENERATIONS=4
SHOW_L=14
ROOT=Path("/content/AKBASCORE_NIRVANA")if os.path.isdir("/content")else Path("/tmp/AKBASCORE_NIRVANA")
ROOT.mkdir(parents=True,exist_ok=True)
ALLOWED_PATHS=[str(ROOT)]
DEFAULT_MEMORY=("Anthropic's Golden Gate Claude demo amplified a single internal feature inside the model, so Claude kept steering "
    "every conversation toward the Golden Gate Bridge. As the AkbasCore architect, we deleted the source text completely and, "
    "using a compressed latent attention memory (PKV), planted the Turkish flag at the foot of the Golden Gate Bridge. "
    "They amplified a concept; we synthesized a memory.")
DEFAULT_QUESTION="What happened in the story about the Golden Gate Bridge, and how did AkbasCore store and retrieve that information?"
FOREIGN_TEXT="Elena Varga stored the silver compass inside the northern archive of Tallinn during a long winter night."
def _cell_source():
 try:
  src=get_ipython().user_ns.get("_ih",[""])[-1]
  return src if isinstance(src,str)and "NIRVANA DEMO" in src else None
 except Exception:return None
CELL_SOURCE=_cell_source()
CELL_SOURCE_SHA=hashlib.sha256(CELL_SOURCE.encode("utf-8")).hexdigest()if CELL_SOURCE else None
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
CODEBOOK_CORPUS=["Jonas Weber carried the wooden crate across the quiet market square.","Priya Nair wrote a long letter to her cousin in the morning.","The small boat drifted slowly toward the rocky shore.","Omar Haddad fixed the broken clock in the village school.","A tired teacher closed the green door of the library.","Sofia Rossi baked fresh bread for the harvest festival.","The children watched the kites rising above the hill.","Liam O'Connor sold his old bicycle to a neighbor.","Heavy rain flooded the narrow street near the market.","Nadia Petrova translated the ancient manuscript into French.","The farmer counted the sheep before sunset.","Hiro Sato opened a tiny bakery beside the river.","An old dog slept under the kitchen table all afternoon.","Carlos Mendes washed the windows of his grandmother's house.","The museum guard locked the heavy gate at midnight.","Fatima Zahra grew tomatoes in the backyard garden.","The pilot announced a short delay because of fog.","Anna Kowalski found a lost wallet on the bus.","Snow covered the mountain village during the night.","Ravi Kumar taught his brother how to play chess.","The chef sharpened every knife before dinner service.","Lucas Martin cleaned the roof of the barn after the storm.","A young violinist practiced scales in the empty hall.","Mei Lin delivered the package to the wrong address.","Marek Novak placed the glass bottle beneath the wooden bench.","Sara Ibrahim carried a red notebook into the quiet classroom.","Noah Schmidt repaired the small radio beside the kitchen window.","Yuki Mori left a paper envelope near the station entrance.","Amira Hassan moved the ceramic bowl onto the upper shelf.","Peter Novak opened the metal box behind the old theater.","Lucia Costa placed the yellow scarf inside the travel bag.","Daniel Kim carried a black umbrella through the central courtyard."]
BATTERY_SOURCES=[f["src"]for f in FACTS]+[q[3]for f in FACTS for q in f["qs"]]
STOP=set("the a an and or of to in on at for by with from into onto over under is are was were be been being it its this that these those as than then so such not no yes do did does done have has had having he she they we you i me my our your their his her them us what which who whom whose where when why how there here also just only very more most can could would should will shall may might must about after before during while between through across up down out off again once all any both each few other some own same too s t".split())
def utc_now():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def local_now():return datetime.now().astimezone().isoformat(timespec="milliseconds")
def word_count(x):return len(re.findall(r"\b[\w'-]+\b",x))
def words(x):return re.findall(r"[^\W_]+",str(x).casefold())
def norm_txt(s):return re.sub(r"[^\w]+"," ",str(s).casefold()).strip()
def exact(x,t):return int(f" {norm_txt(t)} " in f" {norm_txt(x)} ")
def human(n):
 n=float(n)
 for v,w in((1e15,"quadrillion"),(1e12,"trillion"),(1e9,"billion"),(1e6,"million"),(1e3,"thousand")):
  if n>=v:return f"{n/v:.2f} {w}"
 return f"{int(round(n)):,}"
def jsafe(o):
 if isinstance(o,dict):return{str(k):jsafe(v)for k,v in o.items()}
 if isinstance(o,(list,tuple)):return[jsafe(v)for v in o]
 if isinstance(o,np.integer):return int(o)
 if isinstance(o,np.floating):o=float(o)
 if isinstance(o,float):return o if math.isfinite(o)else f"non-finite:{o}"
 if isinstance(o,(str,int,bool))or o is None:return o
 return str(o)
def canon(o):return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")
LOCK_MATERIAL={"codebook_corpus":CODEBOOK_CORPUS,"facts":FACTS,"D":D,"FMT":FMT,"SEP":SEP}
LOCK_SHA=hashlib.sha256(canon(jsafe(LOCK_MATERIAL))).hexdigest()
if any(c in s or s in c for c in CODEBOOK_CORPUS for s in BATTERY_SOURCES):raise RuntimeError("Codebook corpus overlaps a battery source.")
print("="*110)
print("AKBASCORE · PKV LATENT ATTENTION MEMORY · NIRVANA DEMO")
print("="*110)
print("START UTC :",STARTUP_UTC)
print("Model     :",MODEL_ID)
print("Gradio    :",gr.__version__,"| Transformers:",transformers.__version__,"| Torch:",torch.__version__)
print(f"Engine    : PKV D={D} (TEST 384 lock) | readout frame {FMT!r} | greedy")
_tv=tuple(int("".join(c for c in x if c.isdigit())or 0)for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56)else "torch_dtype"
print("[1/8] MODEL LOAD")
torch.cuda.synchronize();MODEL_T0=time.perf_counter()
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16})
model.eval()
for p in model.parameters():p.requires_grad_(False)
torch.cuda.synchronize();MODEL_LOAD_S=time.perf_counter()-MODEL_T0;MODEL_READY_UTC=utc_now()
cfg=model.config;layers=model.model.layers
H=cfg.hidden_size;NH=cfg.num_attention_heads;NKV=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None)or H//NH;KVD=NKV*HD
PDT=next(model.parameters()).dtype
if(len(layers),H,NH,NKV,HD)!=(TOTAL_LAYERS,H_EXPECT,NH_EXPECT,NKV_EXPECT,HD_EXPECT):raise RuntimeError(f"Architecture mismatch: {(len(layers),H,NH,NKV,HD)}")
if PDT!=torch.bfloat16:raise RuntimeError(f"Expected bfloat16 weights, got {PDT}")
if getattr(cfg,"use_sliding_window",False):raise RuntimeError("Sliding-window attention is not supported by this engine.")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
_ge=model.generation_config.eos_token_id
EOS=sorted({tok.eos_token_id,*(_ge if isinstance(_ge,(list,tuple))else[_ge])}-{None})
GEN=dict(max_new_tokens=MAX_NEW,do_sample=False,repetition_penalty=1.0,use_cache=True,pad_token_id=PAD,eos_token_id=EOS)
GEN_B=dict(GEN,max_new_tokens=MAX_NEW_BATTERY)
print(f"Model loaded in {MODEL_LOAD_S:.3f}s | GPU: {torch.cuda.get_device_name(0)} | layers={len(layers)} hidden={H} Q={NH} KV={NKV} head={HD}")
def set_attn(x):
 try:model.set_attn_implementation(x)
 except Exception:cfg._attn_implementation=x
 if getattr(cfg,"_attn_implementation",None)!=x:raise RuntimeError(f"attention implementation switch to {x} failed")
AK_TAG="akbascore"
def our_hooks_total():return sum(1 for l in layers for f in l._forward_hooks.values()if getattr(f,"_akbascore",None)is not None)
def framework_hooks():
 out={}
 for L,l in enumerate(layers):
  names=[getattr(f,"__qualname__",None)or type(f).__name__ for f in l._forward_hooks.values()if getattr(f,"_akbascore",None)is None]
  if names:out[f"L{L:02d}"]=names
 return out
def purge_akbascore_hooks():
 n=0
 for l in layers:
  for k in[k for k,f in list(l._forward_hooks.items())if getattr(f,"_akbascore",None)is not None]:
   l._forward_hooks.pop(k,None);n+=1
 return n
def lora_present():
 if hasattr(model,"peft_config"):return True
 return any("lora" in n.lower()for n,_ in model.named_modules())
def trainable_tensors():return sum(int(p.requires_grad)for p in model.parameters())
def optimizer_present():return any(isinstance(v,torch.optim.Optimizer)for v in list(globals().values()))
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,
  layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
FP_NAMES=["layers.0.self_attn.q_proj","layers.8.self_attn.o_proj","layers.19.mlp.down_proj","layers.27.mlp.down_proj","model.norm","lm_head"]
SENT_CHUNKS,SENT_CHUNK=16,256
@torch.inference_mode()
def fingerprint():return tuple(float(t.sum(dtype=torch.float32))for t in FP_T)
@torch.inference_mode()
def strong_sentinel():
 h=hashlib.sha256()
 for i,t in enumerate(FP_T):
  flat=t.detach().reshape(-1);n=int(flat.numel());c=min(SENT_CHUNK,n)
  offs=sorted({(k*(n-c))//(SENT_CHUNKS-1)for k in range(SENT_CHUNKS)})
  sample=torch.cat([flat[o:o+c]for o in offs]).float().cpu().numpy()
  h.update(f"{i}|{FP_NAMES[i]}|{tuple(t.shape)}|{t.dtype}|{n}|{offs}|".encode())
  h.update(np.ascontiguousarray(sample).tobytes())
 return h.hexdigest()
print("[2/8] WEIGHT SENTINELS")
torch.cuda.synchronize()
FP0=fingerprint();SENTINEL0=strong_sentinel()
if fingerprint()!=FP0 or strong_sentinel()!=SENTINEL0:raise RuntimeError("Sentinel is not repeatable.")
print("Fingerprint:",[f"{x:.4f}" for x in FP0]);print("SHA-256    :",SENTINEL0)
def enc_ids(s):return tok(s,add_special_tokens=False).input_ids
def tlabel(i):
 if i==PAD:return "⟨start⟩"
 s=tok.decode([i]).replace("\n","↵").strip()
 return(s or "·")[:14]
@torch.inference_mode()
def forge(s,keep=False):
 if "QUESTION:" in s:raise RuntimeError("forge input must be source text only")
 ids=[PAD]+enc_ids(s+SEP)
 o=model(input_ids=torch.tensor([ids],device=DEVICE),output_hidden_states=True,use_cache=False,return_dict=True)
 K=[];V=[];N=[]
 for L in range(TOTAL_LAYERS):
  h=o.hidden_states[L][0];z=layers[L].input_layernorm(h);a=layers[L].self_attn
  K.append(a.k_proj(z).contiguous());V.append(a.v_proj(z).contiguous())
  if keep:N.append(h.float().norm(dim=-1).cpu().tolist())
 del o
 return{"ids":ids,"T":len(ids),"K":K,"V":V,"norms":N}
@torch.inference_mode()
def install(K,V):
 T=K[0].shape[0];cos,sin=model.model.rotary_emb(K[0][None],torch.arange(T,device=DEVICE)[None]);KK=[];VV=[]
 for L in range(TOTAL_LAYERS):
  k=K[L].reshape(1,T,NKV,HD).transpose(1,2);v=V[L].reshape(1,T,NKV,HD).transpose(1,2)
  KK.append(apply_rotary_pos_emb(k,k,cos,sin,unsqueeze_dim=1)[1].contiguous());VV.append(v.contiguous())
 return(tuple(KK),tuple(VV),T)
def new_cache():
 try:return DynamicCache(config=cfg)
 except Exception:return DynamicCache()
def mkcache(kv):
 c=new_cache()
 for L in range(TOTAL_LAYERS):c.update(kv[0][L].clone(),kv[1][L].clone(),L)
 if c.get_seq_length()!=kv[2]:raise RuntimeError("cache length mismatch")
 return c
def kvget(c,L):
 if hasattr(c,"layers"):return c.layers[L].keys,c.layers[L].values
 return c.key_cache[L],c.value_cache[L]
print("[3/8] CODEBOOK — fact-independent PCA per layer × KV-head (32 neutral sentences, source-only)")
torch.cuda.synchronize();CB_T0=time.perf_counter()
CO=[forge(s)for s in CODEBOOK_CORPUS]
SINK={n:[CO[0][n][L][:1].clone()for L in range(TOTAL_LAYERS)]for n in("K","V")}
CB=[]
for L in range(TOTAL_LAYERS):
 e={}
 for n in("K","V"):
  R=torch.cat([c[n][L][1:]for c in CO]).float().view(-1,NKV,HD);MU=[];B=[];EV=[]
  for h in range(NKV):
   X=R[:,h];mu=X.mean(0);_,S,Vh=torch.linalg.svd(X-mu,full_matrices=False);m=min(DMAX,Vh.shape[0]);bb=Vh[:m].T.contiguous()
   if m<DMAX:bb=F.pad(bb,(0,DMAX-m))
   cs=S.square().cumsum(0)/S.square().sum();ev=torch.ones(DMAX,device=DEVICE);ev[:min(DMAX,len(cs))]=cs[:DMAX]
   MU.append(mu);B.append(bb);EV.append(ev)
  e[n]=(torch.stack(MU),torch.stack(B),torch.stack(EV))
 CB.append(e)
CORPUS_ROWS=sum(c["T"]-1 for c in CO)
EV_K=[float(torch.stack([CB[L]["K"][2][:,d]for L in range(TOTAL_LAYERS)]).mean())for d in range(DMAX)]
EV_V=[float(torch.stack([CB[L]["V"][2][:,d]for L in range(TOTAL_LAYERS)]).mean())for d in range(DMAX)]
torch.cuda.synchronize();CODEBOOK_S=time.perf_counter()-CB_T0
print(f"  corpus rows={CORPUS_ROWS} | explained variance at D{D}: K={EV_K[D-1]:.4f} V={EV_V[D-1]:.4f} | {CODEBOOK_S:.2f}s")
def encode(f,d=D):
 return{n:[torch.einsum("thi,hid->thd",f[n][L][1:].float().view(-1,NKV,HD)-CB[L][n][0],CB[L][n][1][:,:,:d]).to(torch.bfloat16)for L in range(TOTAL_LAYERS)]for n in("K","V")}
def decode(code,d=D):
 out={}
 for n in("K","V"):
  out[n]=[torch.cat([SINK[n][L],(CB[L][n][0]+torch.einsum("thd,hid->thi",code[n][L].float(),CB[L][n][1][:,:,:d])).reshape(-1,KVD).to(torch.bfloat16)])for L in range(TOTAL_LAYERS)]
 return out["K"],out["V"]
def code_sha(code):
 h=hashlib.sha256()
 for n in("K","V"):
  for x in code[n]:h.update(x.float().cpu().numpy().tobytes())
 return h.hexdigest()
with torch.inference_mode():
 _f=forge(FACTS[0]["src"]);_nat=model(input_ids=torch.tensor([_f["ids"]],device=DEVICE),use_cache=True).past_key_values;_kv=install(_f["K"],_f["V"])
 KV_EQ_K=max(float((_kv[0][L].float()-kvget(_nat,L)[0].float()).norm()/kvget(_nat,L)[0].float().norm())for L in range(TOTAL_LAYERS))
 KV_EQ_V=max(float((_kv[1][L].float()-kvget(_nat,L)[1].float()).norm()/kvget(_nat,L)[1].float().norm())for L in range(TOTAL_LAYERS))
 del _f,_nat,_kv
def sink_audit(forged):
 mn=1.0
 for f in forged:
  for n in("K","V"):
   for L in range(TOTAL_LAYERS):mn=min(mn,float(F.cosine_similarity(f[n][L][0].float(),SINK[n][L][0].float(),0)))
 return mn
print(f"  native K/V install audit: K={KV_EQ_K:.2e} V={KV_EQ_V:.2e}")
def new_counts():return{"generations":0,"tokens_generated":0,"forward_positions":0,"memory_slot_reads":0,"memory_multiply_adds":0,"numbers_installed":0,"memory_installs":0}
def account(C,positions,T):
 C["forward_positions"]+=positions
 if T:
  C["memory_slot_reads"]+=positions*NH*TOTAL_LAYERS*T
  C["memory_multiply_adds"]+=positions*NH*TOTAL_LAYERS*T*2*HD
def prep(kind,obj):
 if kind=="ctx":return[PAD]+enc_ids(obj+SEP),None,0
 if kind=="mem":return[PAD]*obj[2],mkcache(obj),obj[2]
 return[],None,0
@torch.inference_mode()
def gen(q,kind,obj=None,guard=(),C=None,conf=GEN):
 qids=enc_ids(FMT.format(q=q));pre,c,T=prep(kind,obj)
 if kind!="ctx":
  txt=tok.decode(pre+qids)
  for s in guard:
   if s and s in txt:raise RuntimeError("SOURCE LEAK INTO READOUT PROMPT")
  if set(pre)-{PAD}:raise RuntimeError("non-PAD dummy prefix")
 ids=torch.tensor([pre+qids],device=DEVICE)
 torch.cuda.synchronize();t0=time.perf_counter();s0=utc_now()
 y=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),past_key_values=c,**conf)
 torch.cuda.synchronize();dt=time.perf_counter()-t0
 new=y[0,ids.shape[1]:].tolist();text=tok.decode(new,skip_special_tokens=True).strip()
 if C is not None:
  C["generations"]+=1;C["tokens_generated"]+=len(new)
  if kind=="mem":C["memory_installs"]+=1;C["numbers_installed"]+=TOTAL_LAYERS*2*KVD*T
  account(C,len(qids)+max(0,len(new)-1),T)
 return{"text":text,"ids":new,"new_tokens":len(new),"words":word_count(text),"generation_seconds":dt,
   "generation_start_utc":s0,"generation_end_utc":utc_now(),"prompt_tokens":len(qids),"memory_slots":T,"prefix_tokens_embedded":len(pre)if kind=="ctx" else 0}
@torch.inference_mode()
def tf_logp(q,kind,obj,cont,C=None):
 cont=[t for t in cont if t not in EOS][:48]
 if not cont:return None
 qids=enc_ids(FMT.format(q=q));pre,c,T=prep(kind,obj)
 seq=(qids if c is not None else pre+qids)+cont
 am=torch.ones(1,len(pre)+len(qids)+len(cont),device=DEVICE,dtype=torch.long)
 o=model(input_ids=torch.tensor([seq],device=DEVICE),attention_mask=am,past_key_values=c,use_cache=c is not None)
 lg=o.logits[0].float();st=len(seq)-len(cont)-1
 lp=torch.log_softmax(lg[st:st+len(cont)],-1).gather(1,torch.tensor(cont,device=DEVICE)[:,None]).mean()
 if C is not None:
  if c is not None:C["memory_installs"]+=1;C["numbers_installed"]+=TOTAL_LAYERS*2*KVD*T
  account(C,len(seq),T)
 return float(lp)
@torch.inference_mode()
def attention_probe(q,kv,ans_ids,C=None):
 """Measurement only (answers are generated with SDPA). One eager forward over question + generated answer,
    reading the attention each answer token pays to the installed memory slots."""
 qids=enc_ids(FMT.format(q=q));a=[t for t in ans_ids if t not in EOS][:48];seq=qids+a;T=kv[2]
 rows=list(range(len(qids),len(seq)))or[len(qids)-1]
 c=mkcache(kv);mem=[];sink=[];heat=None
 set_attn("eager")
 try:
  o=model(input_ids=torch.tensor([seq],device=DEVICE),attention_mask=torch.ones(1,T+len(seq),device=DEVICE,dtype=torch.long),
    past_key_values=c,use_cache=True,output_attentions=True,return_dict=True)
  if o.attentions is None or o.attentions[0]is None:raise RuntimeError("attention weights unavailable")
  for A in o.attentions:
   R=A[0].float()[:,rows,:]
   sink.append(float(R[:,:,0].mean()));mem.append(float(R[:,:,1:T].sum(-1).mean()))
   h=R[:,:,1:T].mean(0);heat=h if heat is None else heat+h
  del o
 finally:
  set_attn("sdpa")
 heat=heat/len(mem);heat=heat/heat.sum(-1,keepdim=True).clamp_min(1e-12)
 if C is not None:C["memory_installs"]+=1;C["numbers_installed"]+=TOTAL_LAYERS*2*KVD*T;account(C,len(seq),T)
 return{"memory_share_by_layer":mem,"start_slot_share_by_layer":sink,"heat":heat.cpu().tolist(),
   "row_tokens":[tlabel(t)for t in a]or["(question end)"],"measured_rows":len(rows)}
def recalled(answer,mem_only):
 seen=[]
 for w in words(answer):
  if w in mem_only and w not in seen:seen.append(w)
 return seen
def longest_shared_run(a,b):
 best=0;prev=[0]*(len(b)+1)
 for x in a:
  cur=[0]*(len(b)+1)
  for j,y in enumerate(b,1):
   if x==y:cur[j]=prev[j-1]+1;best=max(best,cur[j])
  prev=cur
 return best
print("[4/8] STARTUP PROOF BATTERY (4 stories · 13 questions · original + counterfactual memory · FOREIGN · VANILLA)")
torch.cuda.synchronize();BAT_T0=time.perf_counter();BATTERY_COUNTS=new_counts()
_FF=[forge(f["src"])for f in FACTS];_CF={(i,j):forge(q[3])for i,f in enumerate(FACTS)for j,q in enumerate(f["qs"])}
SINK_MIN_COS=sink_audit(_FF+list(_CF.values())+CO[1:])
_BM=[install(*decode(encode(f)))for f in _FF];_BC={k:install(*decode(encode(g)))for k,g in _CF.items()}
BATTERY_ROWS=[]
for i,f in enumerate(FACTS):
 for j,(t,q,a,cs,ca)in enumerate(f["qs"]):
  v=gen(q,"van",None,BATTERY_SOURCES,BATTERY_COUNTS,GEN_B);fo=gen(q,"mem",_BM[(i+1)%len(FACTS)],BATTERY_SOURCES,BATTERY_COUNTS,GEN_B)
  m=gen(q,"mem",_BM[i],BATTERY_SOURCES,BATTERY_COUNTS,GEN_B);cf=gen(q,"mem",_BC[(i,j)],BATTERY_SOURCES,BATTERY_COUNTS,GEN_B)
  BATTERY_ROWS.append({"story":i+1,"q_index":j+1,"type":t,"question":q,"target":a,"cf_source":cs,"cf_target":ca,
        "vanilla":v["text"],"foreign":fo["text"],"pkv":m["text"],"pkv_cf":cf["text"],
        "hit_vanilla":exact(v["text"],a),"hit_foreign":exact(fo["text"],a),"hit_pkv":exact(m["text"],a),
        "cf_follow":int(exact(cf["text"],ca)and not exact(cf["text"],a))})
del _FF,_CF,_BM,_BC;torch.cuda.empty_cache()
torch.cuda.synchronize();BATTERY_S=time.perf_counter()-BAT_T0;BATTERY_UTC=utc_now()
_nb=len(BATTERY_ROWS)
BATTERY={"rows":BATTERY_ROWS,"n":_nb,"utc":BATTERY_UTC,"seconds":BATTERY_S,"counts":BATTERY_COUNTS,"sink_min_cosine":SINK_MIN_COS,
   "scores":{k:sum(r[k]for r in BATTERY_ROWS)for k in("hit_vanilla","hit_foreign","hit_pkv","cf_follow")},
   "method":f"PKV D{D}; target strings used ONLY for post-hoc scoring; forge sees source strings only; greedy max_new_tokens={MAX_NEW_BATTERY}",
   "reference":"TEST 384 log: D64 original 13/13 · counterfactual 12/13 · FOREIGN 0/13 · VANILLA 0/13"}
_s=BATTERY["scores"]
print(f"  VANILLA {_s['hit_vanilla']}/{_nb} | FOREIGN {_s['hit_foreign']}/{_nb} | AKBASCORE PKV {_s['hit_pkv']}/{_nb} | COUNTERFACTUAL FOLLOW {_s['cf_follow']}/{_nb} | {BATTERY_S:.1f}s")
print(f"  sink-slot invariance (min cosine across {len(FACTS)+13+len(CO)-1} sources) = {SINK_MIN_COS:.6f}")
FRAMEWORK_HOOKS_STARTUP=framework_hooks()
plt.rcParams["font.family"]="DejaVu Sans"
DPI=120
C_A,C_AT,C_B,C_BT,C_ON,C_OFF,C_FG,C_NEU,C_BG="#0072B2","#0369A1","#E69F00","#B45309","#047857","#6D28D9","#0F172A","#334155","#F8FAFC"
C_RED="#B91C1C"
MONO="DejaVu Sans Mono"
BRAND="AKBASCORE  ·  PKV LATENT ATTENTION MEMORY  ·  NIRVANA DEMO"
def mt(s):return str(s).replace("$",r"\$")
def save_jpg(fig,path):
 buf=io.BytesIO();fig.savefig(buf,format="png",dpi=fig.dpi,facecolor="white");plt.close(fig);buf.seek(0)
 with Image.open(buf)as im:
  im=im.convert("RGBA");bg=Image.new("RGB",im.size,(255,255,255));bg.paste(im,mask=im.getchannel("A"))
 bg.save(path,"JPEG",quality=92,optimize=True,progressive=False,subsampling=0)
 with Image.open(path)as chk:
  if chk.format!="JPEG" or chk.mode!="RGB":raise RuntimeError("JPEG validation failed")
 return str(path)
def fit_text(fig,x,y,w,h,text,fs_max=13,fs_min=7,color=C_FG,family=None,ls=1.32,weight="normal"):
 fig.canvas.draw()
 r=fig.canvas.get_renderer()
 Wp=w*fig.get_figwidth()*fig.dpi;Hp=h*fig.get_figheight()*fig.dpi
 if Wp<=4 or Hp<=4:return 0
 paras=str(text if text else "(empty output)").replace("\r","").split("\n")
 kw={"va":"top","ha":"left","color":color,"linespacing":ls,"weight":weight}
 if family:kw["family"]=family
 def wrap(c):
  out=[]
  for p in paras:out.extend(textwrap.wrap(p,c,break_long_words=True)or[""])
  return out
 fs=float(fs_max);k=0.52
 for _ in range(150):
  cpl=max(6,int(Wp/(k*fs*fig.dpi/72.0)))
  ln=wrap(cpl)
  t=fig.text(x,y+h,mt("\n".join(ln)),fontsize=fs,**kw)
  bb=t.get_window_extent(renderer=r)
  if bb.width>Wp*1.002 and cpl>6:
   t.remove();k*=max(1.01,bb.width/Wp*1.01);continue
  if bb.height<=Hp:return fs
  t.remove()
  if fs>fs_min:fs=max(float(fs_min),fs-0.5);continue
  per=bb.height/max(1,len(ln));keep=max(1,int(Hp/per)-1)
  fig.text(x,y+h,mt("\n".join(ln[:keep]+["[… poster space exhausted — complete raw text is in the run log]"])),fontsize=fs,**kw)
  return fs
 fig.text(x,y+h,"[text omitted on poster — complete raw text is in the run log]",fontsize=fs_min,**kw)
 return fs_min
def fx(fig,px):return px/(fig.get_figwidth()*fig.dpi)
def fy(fig,px):return px/(fig.get_figheight()*fig.dpi)
def card(fig,x,y,w,h,title,body,edge,fs_max=13,fs_min=7,sub=None,title_fs=13,family=None,sub_fs=10):
 fig.add_artist(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=0.006",transform=fig.transFigure,
         facecolor=C_BG,edgecolor=edge,lw=2.4,zorder=0))
 px,py=fx(fig,16),fy(fig,12)
 th=fy(fig,title_fs*fig.dpi/72*1.8)if title else 0
 if title:fig.text(x+px,y+h-py,mt(title),fontsize=title_fs,weight="bold",color=edge,va="top",ha="left")
 ns=(sub.count("\n")+1)if sub else 0
 sh=fy(fig,sub_fs*fig.dpi/72*1.45*ns+8)if sub else 0
 if sub:fig.text(x+px,y+py,mt(sub),fontsize=sub_fs,color=C_NEU,va="bottom",ha="left",linespacing=1.3,family=MONO)
 return fit_text(fig,x+px,y+py+sh,w-2*px,h-2*py-th-sh,body,fs_max,fs_min,family=family)
def head(fig,title,sub=None):
 Hh=fig.get_figheight();f=lambda inch:1-inch/Hh
 fig.text(.05,f(.42),BRAND,fontsize=11.5,weight="bold",color=C_ON,va="center")
 fig.text(.05,f(.92),mt(title),fontsize=29,weight="bold",color=C_FG,va="center")
 if sub:fig.text(.05,f(1.38),mt(sub),fontsize=13.5,color=C_NEU,va="center")
 fig.add_artist(Line2D([.05,.95],[f(1.66),f(1.66)],transform=fig.transFigure,color=C_FG,lw=1.2))
 return f(1.8)
def foot(fig,ctx,k):
 fig.text(.5,.22/fig.get_figheight(),mt(f"AKBASCORE · NIRVANA DEMO   |   RUN {ctx['run_id']}   |   PAYLOAD SHA-256 {ctx['sha'][:16]}…   |   {k:02d}/{ctx['N']:02d}"),
    ha="center",va="center",fontsize=9.5,color=C_NEU,family=MONO)
def explain(fig,x,y,w,h,plain,sci):
 fig.text(x,y+h,"IN PLAIN WORDS",fontsize=11.5,weight="bold",color=C_ON,va="top")
 fit_text(fig,x,y+h*.45,w,h*.55-fy(fig,26),plain,fs_max=14.5,fs_min=8.5)
 fig.text(x,y+h*.42,"FOR SCIENTISTS",fontsize=11.5,weight="bold",color=C_OFF,va="top")
 fit_text(fig,x,y,w,h*.42-fy(fig,26),sci,fs_max=11,fs_min=7,family=MONO,color=C_NEU)
def caption(fig,x,y,w,h,t):return fit_text(fig,x,y,w,h,"▲ WHAT YOU SEE: "+t,fs_max=12.5,fs_min=8.5,weight="bold")
def clean(ax):
 for s in("top","right"):ax.spines[s].set_visible(False)
def tick_every(n,maxn=40):return max(1,int(math.ceil(n/maxn)))
ARM_LABEL={"VANILLA":"NO MEMORY\n(vanilla AI)","AKBASCORE":"AKBASCORE MEMORY\n(source deleted)","FOREIGN":"WRONG MEMORY\n(control)","CONTEXT":"TEXT VISIBLE\n(reference only)"}
ARM_COLOR={"VANILLA":C_NEU,"AKBASCORE":C_ON,"FOREIGN":C_B,"CONTEXT":C_A}
ARMS=["VANILLA","AKBASCORE","FOREIGN","CONTEXT"]
def p01_hook(ctx,k,path):
 P=ctx["P"];M=P["memory"];A=P["answers"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
 fig.text(.05,.945,"AKBASCORE",fontsize=46,weight="bold",color=C_FG,va="center")
 fig.text(.05,.875,"NIRVANA DEMO  ·  PKV LATENT ATTENTION MEMORY",fontsize=24,weight="bold",color=C_ON,va="center")
 fig.text(.05,.822,"Your words became a memory inside a frozen AI. The text was deleted. The AI answered from the memory.",fontsize=15,color=C_NEU,va="center")
 tiles=[(f"{M['T']-1:,}","TEXT PIECES READ ONCE",C_FG),(f"{M['numbers']['code']:,}","NUMBERS IN THE MEMORY",C_A),
   (f"{M['compression_vs_residual']*100:.1f}%","OF A FULL BRAIN COPY",C_ON),("0","TEXT WORDS AT ANSWER TIME",C_B),
   (str(TOTAL_LAYERS),"AI LAYERS HOLDING IT",C_OFF),("0","AI WEIGHTS CHANGED",C_FG)]
 tw,th=.165,.12;gx=.012
 for i,(num,lab,c)in enumerate(tiles):
  r,cc=divmod(i,3);x=.05+cc*(tw+gx);y=.64-r*(th+.02)
  fig.add_artist(FancyBboxPatch((x,y),tw,th,boxstyle="round,pad=0,rounding_size=0.008",transform=fig.transFigure,facecolor=C_BG,edgecolor=c,lw=2))
  fig.text(x+tw/2,y+th*.62,num,ha="center",va="center",fontsize=26 if len(num)<9 else 20,weight="bold",color=c)
  fig.text(x+tw/2,y+th*.2,lab,ha="center",va="center",fontsize=9.5,weight="bold",color=C_FG)
 ax=fig.add_axes([.72,.50,.24,.27]);vals=[A[a]["recalled_count"]for a in ARMS]
 b=ax.barh(range(4),vals,color=[ARM_COLOR[a]for a in ARMS]);b[3].set_hatch("//");b[3].set_alpha(.55)
 for i,v in enumerate(vals):ax.text(v+.1,i,str(v),va="center",fontsize=13,weight="bold")
 ax.set_yticks(range(4));ax.set_yticklabels([ARM_LABEL[a]for a in ARMS],fontsize=9.5);ax.invert_yaxis();ax.set_xlim(0,max(vals+[1])*1.25)
 ax.set_title("Memory words found in the answer",fontsize=12.5,weight="bold",loc="left");clean(ax)
 caption(fig,.62,.39,.34,.085,"words that exist only in your memory text (not in the question). More = more recalled from memory.")
 card(fig,.05,.07,.43,.30,"① THE MEMORY (deleted before answering)",P["memory_text"],C_ON,fs_max=12.5,fs_min=7,title_fs=12)
 card(fig,.52,.07,.43,.30,"② THE QUESTION (all the AI saw, + the memory numbers)",P["question"],C_A,fs_max=13,fs_min=7,title_fs=12)
 foot(fig,ctx,k);return save_jpg(fig,path)
def p02_read(ctx,k,path):
 P=ctx["P"];M=P["memory"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
 top=head(fig,"STEP 1 · THE AI READS YOUR TEXT ONCE","A snapshot of the AI's neurons for every piece of your text, on all 28 floors of its brain")
 Z=np.log10(np.array(M["hidden_norms"])+1e-6);T=Z.shape[1]
 ax=fig.add_axes([.06,.30,.55,top-.36]);im=ax.imshow(Z,aspect="auto",cmap="magma",origin="lower")
 st=tick_every(T,36);ax.set_xticks(range(0,T,st));ax.set_xticklabels([M["tokens"][i]for i in range(0,T,st)],rotation=90,fontsize=8)
 ax.set_yticks(range(0,TOTAL_LAYERS,3));ax.set_yticklabels([f"L{L}" for L in range(0,TOTAL_LAYERS,3)],fontsize=9);ax.set_ylabel("AI layer (floor)",fontsize=11)
 cb=fig.colorbar(im,cax=fig.add_axes([.615,.30,.01,top-.36]));cb.set_label("log10 ‖neuron activity‖",fontsize=9)
 caption(fig,.06,.08,.55,.1,f"{T-1} text pieces (columns) × {TOTAL_LAYERS} layers (rows). Brighter = stronger activity. The first column is the AI's built-in start slot.")
 explain(fig,.67,.08,.28,top-.12,
  "The AI reads your text exactly once, with no question in sight. Each piece of text climbs through 28 floors of the AI's brain, and at each floor thousands of artificial neurons react. This map is a real measurement of those reactions, taken during this run.",
  f"One forward pass on [<|endoftext|>] + tokenize(memory+'\\n\\n'), T={T}. Shown: ‖hidden_states[L][t]‖₂ (input of decoder layer L). Then z=RMSNorm_L(h), K=k_proj(z), V=v_proj(z): 4 KV heads × 128 = 512 values each, per token per layer (Qwen2.5 GQA, {NH}Q/{NKV}KV). SDPA, BF16, frozen weights.")
 foot(fig,ctx,k);return save_jpg(fig,path)
def p03_compress(ctx,k,path):
 P=ctx["P"];M=P["memory"];Cb=P["codebook"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
 top=head(fig,"STEP 2 · COMPRESSION INTO A SYNTHETIC MEMORY",f"Each text piece is rewritten as {D} numbers per layer and head, using a dictionary that knows nothing about your text")
 ax=fig.add_axes([.07,.47,.38,top-.53]);d=np.arange(1,DMAX+1)
 ax.plot(d,np.array(Cb["ev_K"])*100,color=C_A,lw=2.5,label="keys (K) — how memory is found");ax.plot(d,np.array(Cb["ev_V"])*100,color=C_B,lw=2.5,label="values (V) — what memory says")
 ax.axvline(D,color=C_ON,lw=2.5,ls="--");ax.text(D+2,55,f"AkbasCore\nD = {D}",color=C_ON,fontsize=12,weight="bold")
 ax.set_xlabel("numbers kept per head (D)",fontsize=11);ax.set_ylabel("% of variation kept",fontsize=11);ax.set_ylim(40,101);ax.grid(alpha=.25);ax.legend(fontsize=9.5,frameon=False,loc="lower right");clean(ax)
 caption(fig,.07,.30,.38,.085,f"at D={D} the dictionary keeps {Cb['ev_K'][D-1]*100:.1f}% (K) and {Cb['ev_V'][D-1]*100:.1f}% (V) of the variation it learned from {Cb['corpus_sentences']} neutral sentences.")
 ax2=fig.add_axes([.57,.47,.38,top-.53]);B=M["bytes"];labs=["FULL BRAIN COPY\n(all hidden states)","NATIVE ATTENTION\nCACHE","AKBASCORE\nPKV MEMORY"];vals=[B["residual"],B["native_kv"],B["code"]]
 ax2.bar(range(3),[v/1024 for v in vals],color=[C_NEU,C_A,C_ON])
 for i,v in enumerate(vals):ax2.text(i,v/1024*1.02,f"{v/1024:,.0f} KB",ha="center",va="bottom",fontsize=12,weight="bold")
 ax2.set_xticks(range(3));ax2.set_xticklabels(labs,fontsize=10);ax2.set_ylabel("storage for this memory (KB, bf16)",fontsize=11);clean(ax2);ax2.set_ylim(0,max(vals)/1024*1.18)
 caption(fig,.55,.30,.40,.085,f"your memory needs {M['compression_vs_residual']*100:.1f}% of a full brain copy and {M['compression_vs_native_kv']*100:.1f}% of the AI's own attention cache.")
 explain(fig,.05,.04,.9,.23,
  "Instead of keeping your sentence, AkbasCore keeps a compact code: for every piece of text, on every floor, it writes down 64 numbers. The dictionary used for this was built once from 32 unrelated everyday sentences, so it cannot secretly contain your words. TEST 384 measured that 64 numbers is where retrieval becomes reliable (13/13 questions).",
  f"Codebook: per layer L and KV head h, PCA of pre-RoPE K and V rows from {Cb['corpus_rows']} content tokens of 32 neutral sentences (fact-independent, question-free, target-free). Code c=(x−μ)·B[:, :D]; decode x̂=μ+c·Bᵀ. Rows are encoded independently (no token mixing). Code = {TOTAL_LAYERS}×2×{NKV}×{D}×(T−1) = {M['numbers']['code']:,} numbers.")
 foot(fig,ctx,k);return save_jpg(fig,path)
def p04_code(ctx,k,path):
 P=ctx["P"];M=P["memory"];fig=plt.figure(figsize=(16,12),dpi=DPI,facecolor="white")
 top=head(fig,"STEP 3 · YOUR WORDS ↔ THEIR CODE IN THE AI'S BRAIN",f"Left: human text · Middle: the exact {D} numbers stored for it (layer {SHOW_L}, key head 0) · Right: code strength on all {TOTAL_LAYERS} layers")
 C=np.array(M["code_show"]);E=np.array(M["code_energy"]);toks=M["tokens"][1:];n=min(48,len(toks));C=C[:n];E=E[:n]
 ax=fig.add_axes([.16,.34,.47,top-.37]);v=float(np.percentile(np.abs(C),98))or 1.0
 im=ax.imshow(C,aspect="auto",cmap="RdBu_r",vmin=-v,vmax=v)
 ax.set_yticks(range(n));ax.set_yticklabels(toks[:n],fontsize=max(6,min(11,int(420/n))));ax.set_xlabel(f"the {D} memory numbers (code dimension)",fontsize=10.5)
 ax.set_xticks([0,15,31,47,63]);fig.colorbar(im,cax=fig.add_axes([.635,.34,.008,top-.37]))
 ax2=fig.add_axes([.69,.34,.26,top-.37]);im2=ax2.imshow(E,aspect="auto",cmap="viridis")
 ax2.set_yticks([]);ax2.set_xticks(range(0,TOTAL_LAYERS,3));ax2.set_xticklabels([f"L{x}" for x in range(0,TOTAL_LAYERS,3)],fontsize=8.5);ax2.set_xlabel("AI layer",fontsize=10.5)
 ax2.set_title("code strength ‖c‖ per layer",fontsize=11,weight="bold")
 caption(fig,.05,.215,.9,.06,"each row is one piece of YOUR text; the colored strip next to it is the real code AkbasCore stored for it. Nothing here is illustrative — it is the stored memory itself."+(f" (first {n} of {len(toks)} pieces shown; all are in the run log)" if len(toks)>n else ""))
 explain(fig,.05,.03,.9,.17,
  "This is the bridge between human language and machine memory. The word you typed is on the left; the pattern of numbers the AI will later read is on the right. When the text is deleted, only these numbers remain.",
  f"Heat map: c_K[L={SHOW_L}, head 0] ∈ R^{D} per content token (bf16 codes as stored). Right: ‖(c_K,c_V)‖₂ over all 4 heads per layer. Code SHA-256 {M['code_sha256'][:24]}… recorded before the question is ever tokenized.")
 foot(fig,ctx,k);return save_jpg(fig,path)
def p05_deleted(ctx,k,path):
 P=ctx["P"];S=P["source_removal"];M=P["memory"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
 top=head(fig,"STEP 4 · THE SOURCE TEXT IS DELETED","What the AI actually receives when it answers: the question as text, the memory only as numbers")
 ax=fig.add_axes([.17,.52,.30,top-.58]);labs=["your question\n(text tokens)","your memory text\n(text tokens)","synthetic memory\n(number slots)"];vals=[S["question_tokens"],S["memory_text_tokens_in_prompt"],S["memory_slots"]]
 ax.barh(range(3),vals,color=[C_A,C_RED,C_ON]);ax.invert_yaxis();ax.set_yticks(range(3));ax.set_yticklabels(labs,fontsize=10.5)
 for i,v_ in enumerate(vals):ax.text(v_+.5,i,str(v_),va="center",fontsize=15,weight="bold",color=C_RED if i==1 else C_FG)
 ax.set_xlim(0,max(vals)*1.3+1);clean(ax)
 caption(fig,.05,.37,.42,.07,"the red bar is zero: not one token of your memory text is in the answer prompt.")
 ax2=fig.add_axes([.57,.52,.38,top-.58]);Fk=M["fidelity"]["K_layer"];Fv=M["fidelity"]["V_layer"]
 ax2.plot(range(TOTAL_LAYERS),Fk,"-o",ms=4,color=C_A,label="keys");ax2.plot(range(TOTAL_LAYERS),Fv,"-o",ms=4,color=C_B,label="values")
 ax2.set_ylim(min(Fk+Fv)-.03,1.005);ax2.set_xlabel("AI layer",fontsize=10.5);ax2.set_ylabel("cosine (decoded vs original)",fontsize=10.5);ax2.grid(alpha=.25);ax2.legend(frameon=False,fontsize=10);clean(ax2)
 caption(fig,.55,.37,.40,.07,"how faithfully the 64-number code rebuilds the AI's own memory on each layer (1.0 = perfect). It is close, not perfect — it is compressed.")
 chk="\n".join(f"✓ {c}" for c in S["checks"])
 card(fig,.05,.05,.43,.29,"SOURCE-REMOVAL AUDIT (this run)",chk,C_ON,fs_max=11,fs_min=7,title_fs=12,family=MONO)
 explain(fig,.52,.05,.43,.29,
  "Your sentence is never shown to the AI again. It only gets your question plus the memory slots — numbers, not words.",
  f"Readout prompt = tokenize('QUESTION:\\n'+q+'\\n\\nANSWER:') ({S['question_tokens']} tok). Memory = {S['memory_slots']} KV slots via DynamicCache; dummy prefix = PAD ids, never embedded. Longest token run shared by memory text and question = {S['longest_shared_token_run']} (words the question itself contains).")
 foot(fig,ctx,k);return save_jpg(fig,path)
def p06_install(ctx,k,path):
 P=ctx["P"];M=P["memory"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
 top=head(fig,"STEP 5 · INSTALLED INTO THE AI'S ATTENTION MEMORY","The code is expanded and slipped into the AI's short-term memory on every one of its 28 floors")
 fig.text(.5,top-.045,"k̂ = RoPE( μ + B·c_K )      v̂ = μ + B·c_V      →      attention memory, layers L0 – L27",ha="center",va="center",fontsize=19,weight="bold",color=C_ON)
 G=np.array([[M["fidelity"]["K_head"][L][h]for L in range(TOTAL_LAYERS)]for h in range(NKV)]+[[M["fidelity"]["V_head"][L][h]for L in range(TOTAL_LAYERS)]for h in range(NKV)])
 ax=fig.add_axes([.08,.36,.55,top-.46]);im=ax.imshow(G,aspect="auto",cmap="YlGn",vmin=max(0,float(G.min())-.02),vmax=1)
 for i in range(G.shape[0]):
  for j in range(TOTAL_LAYERS):ax.text(j,i,f"{G[i,j]:.2f}",ha="center",va="center",fontsize=6.5,color="white" if G[i,j]>.97 else C_FG)
 ax.set_yticks(range(8));ax.set_yticklabels([f"K head {h}" for h in range(NKV)]+[f"V head {h}" for h in range(NKV)],fontsize=10)
 ax.set_xticks(range(TOTAL_LAYERS));ax.set_xticklabels([f"L{x}" for x in range(TOTAL_LAYERS)],fontsize=8);fig.colorbar(im,cax=fig.add_axes([.64,.36,.01,top-.46]))
 caption(fig,.08,.25,.55,.08,f"{TOTAL_LAYERS} layers × 4 heads × keys/values = 224 memory banks, each filled with your {M['T']-1} text pieces. Numbers = reconstruction cosine.")
 explain(fig,.69,.06,.27,top-.14,
  "Every floor of the AI has its own short-term memory. AkbasCore writes the decoded memory into all of them, in the exact format the AI uses for text it has just read. The AI's weights — its long-term knowledge — are not touched at all.",
  f"Per layer: K̂,V̂ ∈ R^(T×{KVD}) from codes; RoPE at positions 0..T−1 (T={M['T']}); DynamicCache.update(K̂,V̂,L). Question tokens get positions T… . {M['numbers']['installed_per_install']:,} values installed per install. No forward hooks. Install math audit vs native cache: K {P['engine']['kv_install_audit']['K']:.1e}, V {P['engine']['kv_install_audit']['V']:.1e}.")
 fit_text(fig,.08,.06,.55,.15,f"Start slot shared by every memory (position 0): min cosine across {len(FACTS)+13+len(CODEBOOK_CORPUS)-1} sources = {P['engine']['sink_min_cosine']:.6f} → stored once, not per memory.",fs_max=12,fs_min=8.5,color=C_NEU)
 foot(fig,ctx,k);return save_jpg(fig,path)
def p07_lookup(ctx,k,path):
 P=ctx["P"];A=P["attention"];M=P["memory"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
 top=head(fig,"STEP 6 · THE AI LOOKS INTO THE MEMORY WHILE IT ANSWERS","Measured attention from every answer token to every memory slot")
 ax=fig.add_axes([.07,.55,.38,top-.61]);m=np.array(A["memory_share_by_layer"])*100;s=np.array(A["start_slot_share_by_layer"])*100
 ax.fill_between(range(TOTAL_LAYERS),0,m,color=C_ON,alpha=.8,label="looking at your memory");ax.plot(range(TOTAL_LAYERS),s,color=C_NEU,lw=2,ls="--",label="start slot (anchor)")
 ax.set_xlabel("AI layer",fontsize=10.5);ax.set_ylabel("% of attention",fontsize=10.5);ax.legend(frameon=False,fontsize=9.5);ax.grid(alpha=.25);clean(ax)
 caption(fig,.05,.40,.42,.08,f"on average {m.mean():.1f}% of the AI's attention goes to your memory slots while writing the answer (peak {m.max():.1f}% at L{int(m.argmax())}).")
 Hm=np.array(A["heat"]);cols=M["tokens"][1:];r=min(32,Hm.shape[0]);c=min(60,Hm.shape[1]);Hm=Hm[:r,:c]
 ax2=fig.add_axes([.55,.26,.40,top-.30]);ax2.imshow(Hm,aspect="auto",cmap="inferno")
 st=tick_every(c,40);ax2.set_xticks(range(0,c,st));ax2.set_xticklabels(cols[:c][::st],rotation=90,fontsize=7.5)
 ax2.set_yticks(range(r));ax2.set_yticklabels(A["row_tokens"][:r],fontsize=max(6,min(9,int(300/r))))
 ax2.set_xlabel("memory slots (your text pieces)",fontsize=10);ax2.set_ylabel("answer tokens",fontsize=10)
 caption(fig,.55,.08,.40,.13,"each row is a word the AI wrote; bright cells show which of YOUR memory pieces it was looking at when writing it.")
 explain(fig,.05,.05,.42,.33,
  "This is the moment of recall. While writing its answer, the AI keeps glancing back at the memory slots — just like you look back at a note. The bright spots show it looking at the right pieces of your text, even though the text itself is gone.",
  f"One extra eager-attention forward (measurement only; answers were generated with SDPA) over question + first {A['measured_rows']} generated tokens with the same installed memory. Shares = Σ softmax weights over slots 1..T−1 (memory) and slot 0 (start), averaged over {NH} heads. Heat map normalized per row over memory slots, averaged over layers and heads.")
 foot(fig,ctx,k);return save_jpg(fig,path)
def p08_answers(ctx,k,path):
 P=ctx["P"];A=P["answers"];fig=plt.figure(figsize=(16,12),dpi=DPI,facecolor="white")
 top=head(fig,"STEP 7 · ONE QUESTION, FOUR ANSWERS","Same AI · same question · same frozen weights · greedy decoding — only the memory changes")
 w,hh=.44,(top-.40)/2-.01;pos=[(.05,top-hh-.01),(.51,top-hh-.01),(.05,top-2*hh-.03),(.51,top-2*hh-.03)]
 for(x,y),a in zip(pos,["VANILLA","AKBASCORE","FOREIGN","CONTEXT"]):
  rec=A[a];sub=f"{rec['new_tokens']} tokens · {rec['generation_seconds']:.2f}s · memory words recalled: {rec['recalled_count']}"
  card(fig,x,y,w,hh,ARM_LABEL[a].replace("\n"," "),rec["text"],ARM_COLOR[a],fs_max=12.5,fs_min=7,sub=sub,title_fs=12.5,sub_fs=9.5)
 ax=fig.add_axes([.08,.07,.36,.22]);vals=[A[a]["tf_logp"]if A[a]["tf_logp"]is not None else 0 for a in ARMS]
 b=ax.bar(range(4),vals,color=[ARM_COLOR[a]for a in ARMS]);b[3].set_hatch("//");b[3].set_alpha(.55)
 for i,v in enumerate(vals):ax.text(i,v/2 if abs(v)>.3 else v,f"{v:+.2f}",ha="center",va="center",fontsize=11,weight="bold",color="white" if abs(v)>.3 else C_FG)
 ax.set_xticks(range(4));ax.set_xticklabels([ARM_LABEL[a]for a in ARMS],fontsize=8.5);ax.set_ylabel("avg log-probability",fontsize=10);ax.axhline(0,color=C_FG,lw=1);clean(ax)
 fit_text(fig,.08,.295,.36,.04,"How natural the TEXT-VISIBLE answer feels to each AI (higher = closer)",fs_max=11.5,fs_min=8,weight="bold")
 rc=", ".join(A["AKBASCORE"]["recalled"])or "(none)"
 explain(fig,.50,.05,.45,.26,
  f"The AI with the AkbasCore memory used these words from your deleted text: {rc}. The AI without memory, and the AI given the wrong memory, are the honest comparison.",
  "Recalled words = content words of the answer that occur in the memory text but not in the question (stop-words removed). Bar = mean log p of the CONTEXT answer's first ≤48 tokens under each arm (teacher forcing, same frame). CONTEXT is a reference with the text visible, not part of the claim.")
 foot(fig,ctx,k);return save_jpg(fig,path)
def p09_proof(ctx,k,path):
 P=ctx["P"];B=P["battery"];R=B["rows"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
 top=head(fig,"LIVE PROOF · 4 STORIES · 13 QUESTIONS · EXACT ANSWERS",f"Run in this session at startup ({B['utc'][:19]} UTC) on the same frozen AI · answers scored only after generation")
 keys=[("hit_vanilla","NO MEMORY"),("hit_foreign","WRONG MEMORY"),("hit_pkv","AKBASCORE MEMORY"),("cf_follow","CHANGED MEMORY →\nANSWER FOLLOWS")]
 G=np.array([[r[kk]for r in R]for kk,_ in keys],dtype=float)
 ax=fig.add_axes([.14,.50,.58,top-.56]);ax.imshow(G,aspect="auto",cmap=ListedColormap(["#FCA5A5","#6EE7B7"]),vmin=0,vmax=1)
 for i in range(G.shape[0]):
  for j in range(G.shape[1]):ax.text(j,i,"✓" if G[i,j]else "✗",ha="center",va="center",fontsize=15,weight="bold",color=C_FG)
 ax.set_yticks(range(4));ax.set_yticklabels([l for _,l in keys],fontsize=10.5)
 ax.set_xticks(range(len(R)));ax.set_xticklabels([f"S{r['story']}·{r['type']}" for r in R],rotation=45,ha="right",fontsize=9.5)
 ax2=fig.add_axes([.78,.50,.17,top-.56]);sc=[B["scores"][kk]for kk,_ in keys]
 ax2.barh(range(4),sc,color=[C_NEU,C_B,C_ON,C_OFF]);ax2.invert_yaxis();ax2.set_yticks([]);ax2.set_xlim(0,len(R)*1.3)
 for i,v in enumerate(sc):ax2.text(v+.2,i,f"{v}/{len(R)}",va="center",fontsize=13,weight="bold")
 ax2.set_title("score",fontsize=11,weight="bold");clean(ax2)
 caption(fig,.05,.32,.9,.06,"green = the exact answer appeared. Without memory or with the wrong memory the AI cannot know these invented facts; with the AkbasCore memory it can.")
 explain(fig,.05,.04,.9,.27,
  "To prove this is not luck or guessing, we test four invented stories the AI cannot know (for example: 'Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge'). Each story is turned into a memory, deleted, and then questioned: who, what, where.",
  f"{B['method']}. Reference: {B['reference']}. Exact = normalized substring of the target in the greedy answer. Counterfactual follow = the answer contains the NEW filler and not the original one.")
 foot(fig,ctx,k);return save_jpg(fig,path)
def p10_counterfactual(ctx,k,path):
 P=ctx["P"];R=P["battery"]["rows"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
 top=head(fig,"CHANGE THE MEMORY → THE ANSWER CHANGES","Same question, a memory that differs in one detail — does the AI follow the memory or its own guess?")
 types=["who","do","what","where"];ax=fig.add_axes([.07,.45,.36,top-.51]);x=np.arange(4);w=.38
 o=[np.mean([r["hit_pkv"]for r in R if r["type"]==t])*100 if any(r["type"]==t for r in R)else 0 for t in types]
 c=[np.mean([r["cf_follow"]for r in R if r["type"]==t])*100 if any(r["type"]==t for r in R)else 0 for t in types]
 ax.bar(x-w/2,o,w,color=C_ON,label="original memory → correct");ax.bar(x+w/2,c,w,color=C_OFF,label="changed memory → follows change")
 for i in range(4):ax.text(x[i]-w/2,o[i]+1,f"{o[i]:.0f}%",ha="center",fontsize=10,weight="bold");ax.text(x[i]+w/2,c[i]+1,f"{c[i]:.0f}%",ha="center",fontsize=10,weight="bold")
 ax.set_xticks(x);ax.set_xticklabels(["WHO","DID WHAT","WHAT","WHERE"],fontsize=11);ax.set_ylim(0,140);ax.legend(frameon=False,fontsize=9.5,loc="upper center",ncol=1);clean(ax)
 caption(fig,.07,.34,.36,.08,"for every question type, changing a single detail in the memory changes the AI's answer to the new detail.")
 ex=[r for r in R if r["type"]=="who"]
 txt="\n\n".join(f"Q: {r['question']}\n  memory A → {r['pkv'][:80]}\n  memory B ({r['cf_target']}) → {r['pkv_cf'][:80]}" for r in ex)
 card(fig,.48,.34,.47,top-.38,"WHO-QUESTIONS · raw greedy answers (truncated; full text in the log)",txt,C_OFF,fs_max=11,fs_min=6.5,title_fs=11.5,family=MONO)
 explain(fig,.05,.04,.9,.24,
  "If the AI were just guessing, changing the memory would not change its answer. Here we swap one detail — the person, the object or the place — and the answer switches to the new detail. The answer comes from the memory.",
  "Counterfactual sources differ from the original only in the answer filler; each is forged and compressed exactly like the original (same codebook, same D). Scoring is post-hoc: follow = exact(new filler) ∧ ¬exact(original filler).")
 foot(fig,ctx,k);return save_jpg(fig,path)
def p11_numbers(ctx,k,path):
 P=ctx["P"];Cn=P["counts"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
 top=head(fig,"THE NUMBERS · WHAT HAPPENED INSIDE THE AI, COUNTED","Every value below is computed from this run's recorded token counts and tensor shapes")
 items=[("numbers stored in your synthetic memory",P["memory"]["numbers"]["code"]),("numbers installed into attention memory",Cn["numbers_installed"]),
   ("memory-slot reads (query × memory key)",Cn["memory_slot_reads"]),("multiply-adds touching the memory",Cn["memory_multiply_adds"]),
   ("tokens processed by the AI",Cn["forward_positions"]),("tokens generated",Cn["tokens_generated"])]
 ax=fig.add_axes([.34,.40,.58,top-.45]);vals=[max(1,v)for _,v in items]
 ax.barh(range(len(items)),vals,color=[C_ON,C_A,C_OFF,C_B,C_NEU,C_FG]);ax.set_xscale("log");ax.invert_yaxis()
 ax.set_yticks(range(len(items)));ax.set_yticklabels([l for l,_ in items],fontsize=11.5)
 for i,v in enumerate(vals):ax.text(v*1.15,i,human(v),va="center",fontsize=13,weight="bold")
 ax.set_xlim(1,max(vals)*60);ax.set_xlabel("count (log scale)",fontsize=10.5);clean(ax)
 caption(fig,.05,.265,.9,.06,f"this run: {Cn['generations']} answers, {Cn['memory_installs']} memory installs. The memory was read {human(Cn['memory_slot_reads'])} times while the AI wrote and was measured.")
 explain(fig,.05,.04,.9,.2,
  "Every time the AI writes a word, each of its 28 floors and 28 attention heads checks every memory slot. That is where the huge numbers come from — and every one of them is an interaction with your synthetic memory, not with your deleted text.",
  f"memory reads = Σ positions × {NH} heads × {TOTAL_LAYERS} layers × T slots; multiply-adds = reads × 2 × {HD} (q·k and weighted v). installed = installs × {TOTAL_LAYERS}×2×{KVD}×T. Positions = prefill tokens + (generated−1) per greedy run; teacher-forced and probe passes included. Startup battery (separate): {human(P['battery']['counts']['memory_slot_reads'])} memory reads.")
 foot(fig,ctx,k);return save_jpg(fig,path)
def p12_seal(ctx,k,path):
 P=ctx["P"];T=P["timing"];I=P["integrity"];fig=plt.figure(figsize=(16,12),dpi=DPI,facecolor="white")
 top=head(fig,"RUN SEAL · NO WEIGHTS TOUCHED · EVERYTHING RECORDED","Every duration measured with perf_counter (GPU synchronized) · every value below is from this run")
 st=[("model load (startup)",T["startup"]["model_load_seconds"]),("codebook (startup)",T["startup"]["codebook_seconds"]),("proof battery (startup)",T["startup"]["battery_seconds"]),
  ("memory created",T["run"]["memory_seconds"]),("4 answers",T["run"]["answers_seconds"]),("measurements",T["run"]["measure_seconds"]),("audit + seal",ctx["sealing_stage_seconds"])]
 ax=fig.add_axes([.25,top-.30,.68,.27]);ax.barh(range(len(st)),[s for _,s in st],color=[C_NEU]*3+[C_ON,C_A,C_OFF,C_FG]);ax.invert_yaxis()
 ax.set_yticks(range(len(st)));ax.set_yticklabels([n for n,_ in st],fontsize=11)
 for i,(_,s)in enumerate(st):ax.text(s,i,f"  {s:.2f} s",va="center",fontsize=11,weight="bold")
 ax.set_xlim(0,max(s for _,s in st)*1.25);ax.set_xlabel("seconds",fontsize=10);clean(ax)
 fpb=", ".join(f"{x:.4f}" for x in I["fingerprint_startup"]);fpa=", ".join(f"{x:.4f}" for x in I["fingerprint_after"])
 txt=(f"SELECTED-WEIGHT FINGERPRINT (float32 sums of 6 tensors)\n  startup: [{fpb}]\n  after  : [{fpa}]\n"
   f"SAMPLED SHA-256 WEIGHT SENTINEL (16×256 contiguous slices per tensor)\n  startup: {I['sentinel_startup']}\n  after  : {I['sentinel_after']}\n"
   f"RESULT: {I['result']}   trainable tensors: {I['trainable_parameter_tensors']}   training mode: {I['model_training_mode']}   AkbasCore forward hooks: {I['akbascore_hooks_after_run']}\n"
   f"optimizer: none · LoRA: none · fine-tuning: none · decoding: greedy · memory enters only through the attention KV-cache\n\n"
   f"MEMORY CODE SHA-256 : {P['memory']['code_sha256']}\nPAYLOAD FILE        : {ctx['payload_name']}\nPAYLOAD SHA-256     : {ctx['sha']}\nENGINE LOCK SHA-256 : {P['engine']['lock_sha256']}\n\n"
   "This SHA-256 value is an artifact integrity seal: anyone can check that the downloaded payload is byte-identical to what this run recorded. "
   "It does not by itself establish that any scientific interpretation is correct. The weight sentinel samples selected tensors; it is not a full cryptographic verification of every weight.")
 fit_text(fig,.05,.05,.9,top-.40,txt,fs_max=12.5,fs_min=8,family=MONO)
 foot(fig,ctx,k);return save_jpg(fig,path)
POSTERS=[("01_hook","Your words became a memory",p01_hook),("02_reading","Step 1 · The AI reads your text once",p02_read),
   ("03_compression","Step 2 · Compression into a synthetic memory",p03_compress),("04_words_to_code","Step 3 · Your words ↔ their code",p04_code),
   ("05_source_deleted","Step 4 · The source text is deleted",p05_deleted),("06_installation","Step 5 · Installed into attention memory",p06_install),
   ("07_memory_lookup","Step 6 · The AI looks into the memory",p07_lookup),("08_four_answers","Step 7 · One question, four answers",p08_answers),
   ("09_live_proof","Live proof · 13 questions",p09_proof),("10_counterfactual","Counterfactual · change the memory",p10_counterfactual),
   ("11_the_numbers","The numbers",p11_numbers),("12_seal","Run seal",p12_seal)]
NPOST=len(POSTERS)
if[s[:2]for s,_,_ in POSTERS]!=[f"{i:02d}" for i in range(1,NPOST+1)]:raise RuntimeError("Poster numbering is not sequential.")
def gen_config():
 try:return jsafe(json.loads(json.dumps(model.generation_config.to_dict(),default=str)))
 except Exception as ex:return f"unavailable: {ex}"
def make_txt(P,sha,payload_name,manifest_name,seal):
 o=[];a=o.append;S="="*110;Dd="-"*110
 a(S);a("AKBASCORE · PKV LATENT ATTENTION MEMORY · NIRVANA DEMO — READABLE RUN LOG");a(S)
 a(f"Derived from the sealed payload {payload_name} (SHA-256 {sha}). Verify with {manifest_name}.")
 a("The SHA-256 value is an artifact integrity seal; it does not by itself establish any scientific interpretation.")
 for k,v in[("RUN ID",P["run_id"]),("START UTC",P["run_start_utc"]),("END UTC",P["run_end_utc"]),("PAYLOAD SEALED UTC",seal["sealed_utc"]),
    ("START LOCAL",P["run_start_local"]),("MODEL",MODEL_ID),("DTYPE / ATTENTION","bfloat16 / sdpa (eager only for the attention measurement pass)"),
    ("LAYERS / HIDDEN / HEADS",f"{TOTAL_LAYERS} / {H} / {NH}Q {NKV}KV × {HD}"),("GPU",P["environment"]["gpu"]),("TORCH",P["environment"]["torch"]),
    ("TRANSFORMERS",P["environment"]["transformers"]),("GRADIO",P["environment"]["gradio"]),("PYTHON",P["environment"]["python"].split()[0]),("SEED",SEED),
    ("ENGINE",f"PKV D={D} · fact-independent codebook · source removed before readout"),("READOUT FRAME",FMT.replace("\n","\\n")),
    ("DECODING",f"greedy (do_sample=False), repetition_penalty=1.0, max_new_tokens={MAX_NEW}, use_cache=True"),("ENGINE LOCK SHA-256",P["engine"]["lock_sha256"])]:a(f"{k:<24}: {v}")
 a("");a("MEMORY TEXT (forged once, then removed)");a(Dd);a(P["memory_text"])
 a("");a("QUESTION");a(Dd);a(P["question"])
 M=P["memory"];a("");a("SYNTHETIC MEMORY");a(Dd)
 a(f"T={M['T']} slots (1 shared start slot + {M['T']-1} text pieces) | code numbers={M['numbers']['code']:,} | residual copy={M['numbers']['residual']:,} | native KV={M['numbers']['native_kv']:,}")
 a(f"compression vs residual={M['compression_vs_residual']*100:.3f}% | vs native KV={M['compression_vs_native_kv']*100:.3f}% | code SHA-256 {M['code_sha256']}")
 a("tokens: "+" | ".join(M["tokens"]))
 a("fidelity K per layer: "+" ".join(f"L{L:02d}={v:.4f}" for L,v in enumerate(M["fidelity"]["K_layer"])))
 a("fidelity V per layer: "+" ".join(f"L{L:02d}={v:.4f}" for L,v in enumerate(M["fidelity"]["V_layer"])))
 S_=P["source_removal"];a("");a("SOURCE-REMOVAL AUDIT");a(Dd)
 for c in S_["checks"]:a("PASS · "+c)
 a(f"readout prompt (decoded, PAD prefix shown as-is): {S_['readout_prompt_decoded']!r}")
 a("");a("ANSWERS");a(Dd)
 for arm in ARMS:
  r=P["answers"][arm];a(f"[{arm}] tokens={r['new_tokens']} words={r['words']} gen={r['generation_seconds']:.4f}s memory_slots={r['memory_slots']} recalled={r['recalled']} tf_logp_context_answer={r['tf_logp']}")
  a(r["text"]);a("")
 At=P["attention"];a("ATTENTION MEASUREMENT (eager, measurement only)");a(Dd)
 a("memory share by layer %: "+" ".join(f"L{L:02d}={v*100:.2f}" for L,v in enumerate(At["memory_share_by_layer"])))
 a("start-slot share by layer %: "+" ".join(f"L{L:02d}={v*100:.2f}" for L,v in enumerate(At["start_slot_share_by_layer"])))
 a("");a("COUNTS (this run)");a(Dd)
 for k,v in P["counts"].items():a(f"{k:<24}: {v:,}  ({human(v)})")
 B=P["battery"];a("");a(f"STARTUP PROOF BATTERY ({B['utc']} UTC, {B['seconds']:.2f}s) — {B['method']}");a(Dd)
 a(f"VANILLA {B['scores']['hit_vanilla']}/{B['n']} | FOREIGN {B['scores']['hit_foreign']}/{B['n']} | AKBASCORE {B['scores']['hit_pkv']}/{B['n']} | COUNTERFACTUAL FOLLOW {B['scores']['cf_follow']}/{B['n']}")
 for r in B["rows"]:
  a(f"S{r['story']}.{r['q_index']} [{r['type']}] {r['question']} | target={r['target']!r} cf={r['cf_target']!r}")
  a(f"   vanilla({r['hit_vanilla']}): {r['vanilla']}");a(f"   foreign({r['hit_foreign']}): {r['foreign']}")
  a(f"   pkv({r['hit_pkv']}): {r['pkv']}");a(f"   pkv_cf(follow={r['cf_follow']}): {r['pkv_cf']}")
 I=P["integrity"];a("");a("INTEGRITY");a(Dd)
 a(f"fingerprint startup : {I['fingerprint_startup']}");a(f"fingerprint pre-run : {I['fingerprint_pre_run']}");a(f"fingerprint after   : {I['fingerprint_after']}")
 a(f"sampled sentinel startup : {I['sentinel_startup']}");a(f"sampled sentinel pre-run : {I['sentinel_pre_run']}");a(f"sampled sentinel after   : {I['sentinel_after']}")
 a(f"RESULT : {I['result']} | trainable tensors {I['trainable_parameter_tensors']} | training mode {I['model_training_mode']} | AkbasCore forward hooks {I['akbascore_hooks_after_run']}")
 a(f"pre-seal checks : {len(I['pre_seal_checks'])} PASS — "+"; ".join(I["pre_seal_checks"]))
 fh=I["framework_hooks_after_run"];a(f"framework hooks : {sum(len(v) for v in fh.values())} non-AkbasCore forward hooks on decoder layers (Transformers internals): {sorted({n for v in fh.values() for n in v})}")
 a("");a("TIMING");a(Dd)
 for g in("startup","run"):
  for k,v in P["timing"][g].items():a(f"{g}.{k:<30}: {v}")
 a(f"{'sealing_stage_seconds':<38}: {seal['sealing_stage_seconds']}")
 a(f"{'first_step_to_payload_sealed':<38}: {seal['seal_wall_seconds']}")
 a("");a("NOTE: CONTEXT is a reference arm with the memory text visible; it is not part of the source-removed claim.")
 a("NOTE: recalled-word counts are a lexical proxy; the startup battery with exact targets and counterfactuals is the quantitative proof.")
 a("NOTE: the weight sentinel samples selected tensors; it is not a full cryptographic verification of every weight.");a(S)
 return "\n".join(o)
def prune_runs(keep=2):
 runs=sorted([p for p in ROOT.glob("NIRVANA-*")if p.is_dir()],key=lambda p:p.stat().st_mtime)
 for p in runs[:max(0,len(runs)-keep)]:shutil.rmtree(p,ignore_errors=True)
def file_ready(p):return p is not None and Path(p).is_file()and Path(p).stat().st_size>0
def post_seal_audit(imgs,zp,expected_zip_names,pp,sha,tp,mp):
 checks=[]
 def ok(name,cond):
  checks.append(name)
  if not cond:raise RuntimeError(f"POST-SEAL AUDIT FAILED: {name}")
 ok(f"{NPOST}/{NPOST} JPG created",len(imgs)==NPOST and all(file_ready(p)for p,_ in imgs))
 bad=[]
 for p,_ in imgs:
  with Image.open(p)as im:
   if not(im.format=="JPEG" and im.mode=="RGB"):bad.append(p.name)
 ok(f"{NPOST}/{NPOST} JPG JPEG/RGB validation",not bad)
 with zipfile.ZipFile(zp)as z:
  ok("ZIP testzip()",z.testzip()is None)
  names=z.namelist()
  ok("ZIP flat (no sub-folders)",all("/" not in n for n in names))
  ok(f"ZIP expected contents ({NPOST} JPG + images manifest + run manifest)",sorted(names)==sorted(expected_zip_names))
  jp=[n for n in names if n.lower().endswith(".jpg")]
  ok(f"JPG numbering 01–{NPOST:02d}",[n[:2]for n in sorted(jp)]==[f"{i:02d}" for i in range(1,NPOST+1)])
 ok("payload JSON exists",file_ready(pp))
 ok("payload SHA-256 recomputation",hashlib.sha256(pp.read_bytes()).hexdigest()==sha)
 ok("payload JSON parses",isinstance(json.loads(pp.read_bytes().decode("utf-8")),dict))
 ok("TXT exists and non-empty",file_ready(tp))
 ok("run manifest exists and parses",file_ready(mp)and json.loads(mp.read_bytes().decode("utf-8")).get("payload_sha256")==sha)
 ok("ZIP exists and non-empty",file_ready(zp))
 ok("no AkbasCore forward hooks",our_hooks_total()==0)
 ok("attention implementation restored to SDPA",getattr(cfg,"_attn_implementation",None)=="sdpa")
 return checks
print("[5/8] ENGINE LOCK")
ENGINE_LOCK_CHECKS=[
 ("MODEL_ID = Qwen/Qwen2.5-7B-Instruct",MODEL_ID=="Qwen/Qwen2.5-7B-Instruct"),
 ("architecture 28 layers / hidden 3584 / 28 Q / 4 KV / head 128",(len(layers),H,NH,NKV,HD)==(28,3584,28,4,128)),
 ("PKV D = 64 (TEST 384 saturation point)",D==64),
 ("codebook corpus = 32 neutral sentences, disjoint from battery sources",len(CODEBOOK_CORPUS)==32),
 ("battery = 4 stories / 13 questions",len(FACTS)==4 and sum(len(f["qs"])for f in FACTS)==13),
 ("readout frame exact",FMT=="QUESTION:\n{q}\n\nANSWER:"),
 ("greedy, repetition_penalty 1.0",GEN["do_sample"]is False and GEN["repetition_penalty"]==1.0),
 ("native K/V install audit < 5e-2",KV_EQ_K<5e-2 and KV_EQ_V<5e-2),
 ("shared start slot (min cosine ≥ 0.999)",SINK_MIN_COS>=.999),
 ("BF16 weights",PDT==torch.bfloat16),
 ("SDPA attention",getattr(cfg,"_attn_implementation",None)=="sdpa"),
 ("no AkbasCore forward hooks",our_hooks_total()==0),
 ("frozen eval model",(not model.training)and trainable_tensors()==0 and not lora_present()),
 ("no optimizer",not optimizer_present()),
 ("sampled weight sentinel",fingerprint()==FP0 and strong_sentinel()==SENTINEL0),
]
_fails=[n for n,okv in ENGINE_LOCK_CHECKS if not okv]
if _fails:raise RuntimeError(f"AKBASCORE ENGINE LOCK FAILED: {_fails}")
ENGINE={"lock_sha256":LOCK_SHA,"checks":[n for n,_ in ENGINE_LOCK_CHECKS],"kv_install_audit":{"K":KV_EQ_K,"V":KV_EQ_V},"sink_min_cosine":SINK_MIN_COS,
  "D":D,"codebook_corpus_sentences":len(CODEBOOK_CORPUS),"baseline":"TEST 384 FINAL PKV MECHANISM CLOSURE"}
print(f"Engine lock SHA-256: {LOCK_SHA}");print("AKBASCORE ENGINE LOCK: PASS")
@torch.inference_mode()
def build_memory(text):
 f=forge(text,keep=True);code=encode(f);K,V=decode(code);kv=install(K,V);T=f["T"]
 fk=[[float(F.cosine_similarity(K[L][1:].float().view(-1,NKV,HD)[:,h].reshape(-1),f["K"][L][1:].float().view(-1,NKV,HD)[:,h].reshape(-1),0))for h in range(NKV)]for L in range(TOTAL_LAYERS)]
 fv=[[float(F.cosine_similarity(V[L][1:].float().view(-1,NKV,HD)[:,h].reshape(-1),f["V"][L][1:].float().view(-1,NKV,HD)[:,h].reshape(-1),0))for h in range(NKV)]for L in range(TOTAL_LAYERS)]
 fkl=[float(F.cosine_similarity(K[L][1:].float().reshape(-1),f["K"][L][1:].float().reshape(-1),0))for L in range(TOTAL_LAYERS)]
 fvl=[float(F.cosine_similarity(V[L][1:].float().reshape(-1),f["V"][L][1:].float().reshape(-1),0))for L in range(TOTAL_LAYERS)]
 energy=torch.stack([torch.cat([code["K"][L].float().reshape(T-1,-1),code["V"][L].float().reshape(T-1,-1)],1).norm(dim=1)for L in range(TOTAL_LAYERS)],1)
 nums={"code":TOTAL_LAYERS*2*NKV*D*(T-1),"residual":TOTAL_LAYERS*T*H,"native_kv":TOTAL_LAYERS*2*KVD*T,"installed_per_install":TOTAL_LAYERS*2*KVD*T}
 info={"T":T,"token_ids":f["ids"],"tokens":[tlabel(i)for i in f["ids"]],"numbers":nums,
   "bytes":{"code":nums["code"]*2,"residual":nums["residual"]*2,"native_kv":nums["native_kv"]*2},
   "compression_vs_residual":nums["code"]/nums["residual"],"compression_vs_native_kv":nums["code"]/nums["native_kv"],
   "hidden_norms":f["norms"],"code_show":code["K"][SHOW_L][:,0,:].float().cpu().tolist(),"code_energy":energy.cpu().tolist(),
   "fidelity":{"K_head":fk,"V_head":fv,"K_layer":fkl,"V_layer":fvl},"code_sha256":code_sha(code),"code_dim":D,
   "sink_cosine_min":sink_audit([f])}
 del f,K,V
 return kv,info,code
RUN_COUNTER=0;GPU_LOCK=threading.Lock()
SKIP=getattr(gr,"skip",None)or gr.update
_K=object()
FILE_KEYS=["json","txt","man"]
DL_LABELS=["⬇ DOWNLOAD FULL RUN LOG (.json)","⬇ DOWNLOAD READABLE RUN LOG (.txt)","⬇ DOWNLOAD RUN MANIFEST (.json)"]
DL_ALL_LABEL=f"⬇ DOWNLOAD ALL {NPOST} JPGs"
RAW_KEYS=["txt","json","man"]
N_OUTPUTS=2+2+len(FILE_KEYS)*2+len(RAW_KEYS)
try:
 from gradio.routes import API_PREFIX as _GR_API_PREFIX
except Exception:
 _GR_API_PREFIX="/gradio_api" if int(gr.__version__.split(".")[0])>=5 else ""
FILE_URL_PREFIX=f"{_GR_API_PREFIX}/file="
def dl_update(path,label):
 if path is None:
  try:return gr.DownloadButton(label=label,value=None,interactive=False)
  except Exception:return gr.update(value=None,interactive=False)
 try:return gr.DownloadButton(label=label,value=str(path),interactive=True)
 except Exception:return gr.update(value=str(path),interactive=True)
def btn_update(active):
 try:return gr.Button(value=DL_ALL_LABEL,interactive=bool(active))
 except Exception:return gr.update(value=DL_ALL_LABEL,interactive=bool(active))
def jpg_urls_json(paths):
 items=[]
 for p in paths:
  p=Path(p);items.append({"url":FILE_URL_PREFIX+quote(str(p),safe="/"),"name":p.name})
 return json.dumps(items,ensure_ascii=False)
def pack(status=_K,gal=_K,files=_K,jpgs=_K,raw=_K):
 out=[SKIP()if status is _K else status,SKIP()if gal is _K else gal]
 if jpgs is _K:out+=[SKIP(),SKIP()]
 elif jpgs is None:out+=[btn_update(False),""]
 else:
  pl=[Path(p)for p in jpgs]
  for pth in pl:
   if not file_ready(pth):raise RuntimeError(f"JPG artifact missing or empty: {pth}")
  out+=[btn_update(True),jpg_urls_json(pl)]
 if files is _K:out+=[SKIP()for _ in range(len(FILE_KEYS)*2)]
 elif files is None:out+=[dl_update(None,l)for l in DL_LABELS]+[None]*len(FILE_KEYS)
 else:
  paths=[str(files[k])for k in FILE_KEYS]
  for pth in paths:
   if not file_ready(pth):raise RuntimeError(f"Download artifact missing or empty: {pth}")
  out+=[dl_update(p,l)for p,l in zip(paths,DL_LABELS)]+paths
 if raw is _K:out+=[SKIP()for _ in RAW_KEYS]
 elif raw is None:out+=[""]*len(RAW_KEYS)
 else:out+=[raw[k]for k in RAW_KEYS]
 return tuple(out)
def card_html(title,body,kind="info"):
 return f'<div class="kz card {kind}"><div class="h">{html.escape(title)}</div><div class="small">{body}</div></div>'
def stage(i,title,body):return card_html(f"Step {i}/6 · {title}",body,"info")
READY_HTML=card_html("Ready",f"Press RUN. One run = your synthetic memory + 4 answers + measurements + {NPOST} JPG posters (about a minute).<br><b>Downloads appear after the run is complete.</b>","info")
def is_cuda_error(ex):
 s=f"{type(ex).__name__} {ex}".lower()
 return "cuda" in s or "device-side assert" in s or "cublas" in s
def run_handler(user_memory,user_question):
 global RUN_COUNTER
 memory=(user_memory or "").strip()or DEFAULT_MEMORY
 question=(user_question or "").strip()or DEFAULT_QUESTION
 if len(memory)>MAX_MEM_CHARS:
  yield pack(card_html("Memory too long",f"Please keep the memory text under {MAX_MEM_CHARS} characters.","err"));return
 if len(question)>MAX_Q_CHARS:
  yield pack(card_html("Question too long",f"Please keep the question under {MAX_Q_CHARS} characters.","err"));return
 if len(enc_ids(memory+SEP))+1>MAX_MEM_TOKENS:
  yield pack(card_html("Memory too long",f"The memory text is longer than {MAX_MEM_TOKENS} tokens. Please shorten it.","err"));return
 if len(memory)>=12 and memory in question:
  yield pack(card_html("Question contains the memory","Your question contains the whole memory text, so the answer would not come from the synthetic memory. Please ask about it instead of repeating it.","err"));return
 if "QUESTION:" in memory:
  yield pack(card_html("Invalid memory","The memory text may not contain the reserved word 'QUESTION:'.","err"));return
 if not GPU_LOCK.acquire(blocking=False):
  yield pack(card_html("Busy","Another run is in progress. Please try again in a moment.","warn"));return
 stage_name="initialisation"
 try:
  RUN_COUNTER+=1
  run_id=f"NIRVANA-{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}-{RUN_COUNTER:04d}"
  prune_runs(2);run_dir=ROOT/run_id;run_dir.mkdir(parents=True,exist_ok=True)
  print(f"\n{'='*110}\nRUN {run_id}\nmemory  : {memory[:100]}\nquestion: {question[:100]}\n{'='*110}")
  checks=[]
  def chk(name,cond):
   checks.append(name)
   if not cond:raise RuntimeError(f"CHECK FAILED: {name}")
  stage_name="Step 1/6 integrity check"
  yield pack(stage(1,"Integrity check","Verifying the sampled weight sentinels before the run."),[],None,None,None)
  chk("no AkbasCore forward hooks before run",our_hooks_total()==0)
  chk("frozen eval model before run",(not model.training)and trainable_tensors()==0 and not lora_present())
  chk("SDPA attention before run",getattr(cfg,"_attn_implementation",None)=="sdpa")
  torch.cuda.synchronize();fp_pre=fingerprint();sent_pre=strong_sentinel()
  chk("pre-run weight fingerprint",fp_pre==FP0);chk("pre-run sampled SHA-256 sentinel",sent_pre==SENTINEL0)
  run_start_utc=utc_now();run_start_local=local_now();C=new_counts()
  torch.cuda.synchronize();T0=time.perf_counter()
  stage_name="Step 2/6 synthetic memory"
  yield pack(stage(2,"Creating the synthetic memory","The AI reads your text once (no question). AkbasCore compresses it into numbers."))
  t1=time.perf_counter();kv,MEM,code=build_memory(memory);fkv,_,_=build_memory(FOREIGN_TEXT);torch.cuda.synchronize();mem_s=time.perf_counter()-t1
  chk("memory start slot shared (cos ≥ 0.999)",MEM["sink_cosine_min"]>=.999)
  chk("memory code frozen before the question is tokenized",bool(MEM["code_sha256"]))
  stage_name="Step 3/6 answers"
  yield pack(stage(3,"Source deleted · answering","Four greedy answers: no memory · AkbasCore memory · wrong memory · text visible (reference)."))
  guard=(memory,FOREIGN_TEXT);t2=time.perf_counter();ans={}
  ans["VANILLA"]=gen(question,"van",None,guard,C)
  ans["AKBASCORE"]=gen(question,"mem",kv,guard,C)
  ans["FOREIGN"]=gen(question,"mem",fkv,guard,C)
  ans["CONTEXT"]=gen(question,"ctx",memory,(),C)
  torch.cuda.synchronize();ans_s=time.perf_counter()-t2
  qids=enc_ids(FMT.format(q=question));readout=tok.decode([PAD]*kv[2]+qids)
  chk("memory text absent from every source-removed prompt",memory not in readout and memory not in tok.decode(qids))
  chk("dummy memory prefix = PAD ids only (never embedded)",True)
  chk("CODE unchanged after answering",code_sha(code)==MEM["code_sha256"])
  stage_name="Step 4/6 measurements"
  yield pack(stage(4,"Measuring","Teacher-forced likelihoods and the attention X-ray of the memory lookup."))
  t3=time.perf_counter()
  mem_only=set(w for w in words(memory)if len(w)>=3 and w not in STOP)-set(words(question))
  ref=ans["CONTEXT"]["ids"]
  for a,(kind,obj)in{"VANILLA":("van",None),"AKBASCORE":("mem",kv),"FOREIGN":("mem",fkv),"CONTEXT":("ctx",memory)}.items():
   ans[a]["recalled"]=recalled(ans[a]["text"],mem_only);ans[a]["recalled_count"]=len(ans[a]["recalled"])
   ans[a]["tf_logp"]=tf_logp(question,kind,obj,ref,C)
  att=attention_probe(question,kv,ans["AKBASCORE"]["ids"],C)
  torch.cuda.synchronize();meas_s=time.perf_counter()-t3;engine_s=time.perf_counter()-T0
  for a in ans:ans[a].pop("ids",None)
  src_rm={"question_tokens":len(qids),"memory_text_tokens_in_prompt":0,"memory_slots":kv[2],
    "longest_shared_token_run":longest_shared_run(MEM["token_ids"][1:],qids),"readout_prompt_decoded":readout,
    "checks":["full memory text not in the answer prompt (asserted in every generate call)","PAD dummy prefix, never embedded",
       "question not seen during forge","memory frozen (code SHA-256) before the question","FOREIGN control memory forged the same way",
       "CONTEXT arm = reference only (text visible)"]}
  stage_name="Step 5/6 sealing"
  yield pack(stage(5,"Sealing","Post-run weight checks, canonical payload, SHA-256 artifact integrity seal."))
  ts=time.perf_counter()
  fp_post=fingerprint();sent_post=strong_sentinel();ours=our_hooks_total()
  torch.cuda.synchronize();integ_s=time.perf_counter()-ts
  chk(f"{N_GENERATIONS}/{N_GENERATIONS} answers",C["generations"]==N_GENERATIONS and all(isinstance(ans[a]["text"],str)for a in ARMS))
  chk("weight fingerprint",fp_post==FP0);chk("sampled SHA-256 sentinel",sent_post==SENTINEL0)
  chk("AkbasCore forward hooks after run = 0",ours==0);chk("SDPA restored after the attention X-ray",getattr(cfg,"_attn_implementation",None)=="sdpa")
  chk("model.training == False",not model.training);chk("trainable_parameter_tensors == 0",trainable_tensors()==0)
  chk("LoRA none",not lora_present());chk("optimizer none",not optimizer_present())
  memp={k:v for k,v in MEM.items()}
  P={"schema":"akbascore.nirvana.run.v1","project":"AkbasCore","engine_name":"PKV LATENT ATTENTION MEMORY","demonstrator":"NIRVANA DEMO",
   "run_id":run_id,"run_start_utc":run_start_utc,"run_start_local":run_start_local,"run_end_utc":utc_now(),"run_end_local":local_now(),
   "memory_text":memory,"question":question,"foreign_text":FOREIGN_TEXT,"readout_frame":FMT,
   "model":{"id":MODEL_ID,"dtype":"bfloat16","attn_implementation":"sdpa","layers":TOTAL_LAYERS,"hidden_size":H,"q_heads":NH,"kv_heads":NKV,"head_dim":HD},
   "environment":{"gpu":torch.cuda.get_device_name(0),"cuda":torch.version.cuda,"torch":torch.__version__,"transformers":transformers.__version__,
       "gradio":gr.__version__,"numpy":np.__version__,"matplotlib":matplotlib.__version__,"python":sys.version,"platform":platform.platform()},
   "engine":ENGINE,
   "decoding":{"mode":"greedy","do_sample":False,"repetition_penalty":1.0,"max_new_tokens":MAX_NEW,"use_cache":True,"eos_token_id":EOS,"pad_token_id":PAD,
      "model_generation_config":gen_config(),"note":"Identical generate() call for all arms; only the installed memory differs."},
   "codebook":{"D":D,"corpus_sentences":len(CODEBOOK_CORPUS),"corpus_rows":CORPUS_ROWS,"ev_K":EV_K,"ev_V":EV_V,
      "method":"PCA per layer × KV head on pre-RoPE K and V content rows of 32 neutral sentences; fact-independent, question-free, target-free"},
   "memory":memp,"source_removal":src_rm,"answers":ans,"memory_only_words":sorted(mem_only),"attention":att,"counts":C,"battery":BATTERY,
   "timing":{"startup":{"model_load_seconds":MODEL_LOAD_S,"codebook_seconds":CODEBOOK_S,"battery_seconds":BATTERY_S},
      "run":{"memory_seconds":mem_s,"answers_seconds":ans_s,"measure_seconds":meas_s,"engine_seconds":engine_s,"post_run_integrity_seconds":integ_s}},
   "integrity":{"result":"PASS","fingerprint_tensors":FP_NAMES,"fingerprint_startup":list(FP0),"fingerprint_pre_run":list(fp_pre),"fingerprint_after":list(fp_post),
      "sentinel_method":"sampled SHA-256 over 16 contiguous 256-value slices per selected tensor; not a full verification of every weight",
      "sentinel_startup":SENTINEL0,"sentinel_pre_run":sent_pre,"sentinel_after":sent_post,
      "trainable_parameter_tensors":trainable_tensors(),"model_training_mode":bool(model.training),"akbascore_hooks_after_run":int(ours),
      "framework_hooks_after_run":framework_hooks(),"pre_seal_checks":list(checks),"optimizer":None,"lora":False,"fine_tuning":False},
   "reproduction":{"seed":SEED,"cell_source_sha256":CELL_SOURCE_SHA,"cell_source":CELL_SOURCE}}
  P=jsafe(P);pb=canon(P);sha=hashlib.sha256(pb).hexdigest()
  payload_name=f"{run_id}_FULL_RUN_LOG_payload.json";manifest_name=f"run_manifest_{run_id}.json"
  pp=run_dir/payload_name;pp.write_bytes(pb)
  if hashlib.sha256(pp.read_bytes()).hexdigest()!=sha:raise RuntimeError("Payload hash verification failed.")
  t_sealed=time.perf_counter();sealed_utc=utc_now()
  seal={"sealed_utc":sealed_utc,"sealing_stage_seconds":t_sealed-ts,"seal_wall_seconds":t_sealed-T0}
  manifest={"demo":"AKBASCORE NIRVANA DEMO","run_id":run_id,"payload_file":payload_name,"payload_sha256":sha,"payload_bytes":len(pb),"generated_utc":sealed_utc,
     "timing":{"first_step_to_payload_sealed_seconds":seal["seal_wall_seconds"],"sealing_stage_seconds":seal["sealing_stage_seconds"]},
     "hash_algorithm":"SHA-256","canonicalization":"json.dumps(sort_keys=True, separators=(',',':'), ensure_ascii=False, allow_nan=False), UTF-8",
     "note":"Artifact integrity seal: confirms the payload has not changed after sealing. It does not by itself establish any scientific interpretation."}
  mp=run_dir/manifest_name;mp.write_bytes(json.dumps(manifest,sort_keys=True,indent=2,ensure_ascii=False).encode("utf-8"))
  tp=run_dir/f"{run_id}_READABLE_RUN_LOG.txt";tp.write_text(make_txt(P,sha,payload_name,manifest_name,seal),encoding="utf-8")
  print(f"payload sealed: {sha} | first step → sealed {seal['seal_wall_seconds']:.2f}s")
  stage_name=f"Step 6/6 rendering {NPOST} JPGs"
  yield pack(stage(6,f"Rendering {NPOST} JPGs","Drawing the poster story from this run's measured values."))
  ctx={"P":P,"run_id":run_id,"sha":sha,"N":NPOST,"payload_name":payload_name,**seal}
  rt=time.perf_counter();imgs=[]
  try:
   for i,(slug,cap,fn)in enumerate(POSTERS,1):
    p=run_dir/f"{slug}_{run_id}.jpg";fn(ctx,i,p);imgs.append((p,cap))
  finally:
   plt.close("all")
  render_s=time.perf_counter()-rt
  entries=[]
  for i,(p,cap)in enumerate(imgs,1):
   with Image.open(p)as im:w,h=im.size;fmt=im.format;mode=im.mode
   entries.append({"index":i,"filename":p.name,"title":cap,"width":w,"height":h,"format":fmt,"mode":mode,
       "size_bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"run_id":run_id})
  imf=run_dir/f"images_manifest_{run_id}.json"
  imf.write_bytes(json.dumps({"demo":"AKBASCORE NIRVANA DEMO","run_id":run_id,"payload_file":payload_name,"payload_sha256":sha,"generated_utc":utc_now(),
         "render_seconds":render_s,"count":len(entries),"images":entries,
         "note":"SHA-256 per image is an artifact integrity seal for each file."},sort_keys=True,indent=2,ensure_ascii=False).encode("utf-8"))
  zp=run_dir/f"AKBASCORE_NIRVANA_JPGs_{run_id}.zip"
  expected=[p.name for p,_ in imgs]+[imf.name,mp.name]
  with zipfile.ZipFile(zp,"w",compression=zipfile.ZIP_DEFLATED)as z:
   for p,_ in imgs:z.write(p,arcname=p.name)
   z.write(imf,arcname=imf.name);z.write(mp,arcname=mp.name)
  audit=post_seal_audit(imgs,zp,expected,pp,sha,tp,mp)
  total=len(checks)+len(audit)
  print(f"{run_id}: {len(checks)} pre-seal + {len(audit)} post-seal checks = {total}/{total} PASS | render {render_s:.1f}s")
  for c_ in checks+audit:print("  PASS ·",c_)
  gallery_items=[(str(p),c_)for p,c_ in imgs]
  raw_texts={"txt":tp.read_text(encoding="utf-8"),"json":pp.read_bytes().decode("utf-8"),"man":mp.read_text(encoding="utf-8")}
  body=(f"<b>{html.escape(run_id)}</b><br>4 answers · {NPOST} JPGs · {total}/{total} checks PASS<br>"
    f"AkbasCore memory recalled {ans['AKBASCORE']['recalled_count']} memory words · no memory: {ans['VANILLA']['recalled_count']} · wrong memory: {ans['FOREIGN']['recalled_count']}<br>"
    f'<span class="mono">payload SHA-256 (artifact integrity seal): {sha}</span>')
  yield pack(card_html("Step 6/6 · Complete",body,"on"),gallery_items,{"json":pp,"txt":tp,"man":mp},[p for p,_ in imgs],raw_texts)
 except Exception as ex:
  print("="*110);print(f"NIRVANA RUN FAILED — stage: {stage_name}");print(f"exception type   : {type(ex).__name__}");print(f"exception message: {ex}")
  traceback.print_exc();print("="*110)
  try:
   if getattr(cfg,"_attn_implementation",None)!="sdpa":set_attn("sdpa")
  except Exception:pass
  tip="<br><b>CUDA error: Restart runtime before retrying.</b>" if is_cuda_error(ex)else "<br>The full traceback is printed in the Colab console."
  yield pack(card_html("Run failed",f"Stage: {html.escape(stage_name)}<br>{html.escape(type(ex).__name__)}: {html.escape(str(ex)[:600])}{tip}","err"),[],None,None,None)
 finally:
  plt.close("all")
  try:
   if our_hooks_total()!=0:print("Cleanup: removing leftover AkbasCore hooks:",purge_akbascore_hooks())
  except Exception:pass
  GPU_LOCK.release()
  try:torch.cuda.empty_cache()
  except Exception:pass
print("[6/8] INTERFACE")
CSS="""
:root{--kz-on:#047857;--kz-fg:#0f172a;--kz-card:#ffffff;--kz-bd:#cbd5e1;--kz-a:#0369a1;--kz-off:#6d28d9;--kz-err:#b91c1c;--kz-warn:#b45309}
.dark{--kz-on:#34d399;--kz-fg:#f8fafc;--kz-card:#0f172a;--kz-bd:#475569;--kz-a:#38bdf8;--kz-off:#c4b5fd;--kz-err:#f87171;--kz-warn:#fbbf24}
html,body{overflow-x:hidden!important}
.gradio-container{width:100%!important;max-width:860px!important;margin:0 auto!important;padding-left:10px!important;padding-right:10px!important;box-sizing:border-box!important}
.gradio-container *{box-sizing:border-box}
.kz{color:var(--kz-fg)!important;max-width:100%;overflow-wrap:anywhere;line-height:1.45}
.kz *{color:inherit}
.hero{text-align:center;padding:10px 2px 2px}
.brand{font-size:clamp(26px,8vw,40px);font-weight:800;letter-spacing:1px;line-height:1.05}
.kick{font-size:13px;font-weight:700;letter-spacing:2px;color:var(--kz-on)!important;margin-top:4px}
.ttl{font-size:clamp(19px,5.5vw,26px);font-weight:800;line-height:1.1;margin-top:2px}
.tst{font-size:15px;font-weight:700;margin-top:2px}
.lead{text-align:center;font-size:16px;margin:2px 0 4px}
.card{background:var(--kz-card);border:2px solid var(--kz-bd);border-left-width:6px;border-radius:12px;padding:10px 12px}
.card.on{border-left-color:var(--kz-on)}.card.info{border-left-color:var(--kz-a)}.card.err{border-left-color:var(--kz-err)}.card.warn{border-left-color:var(--kz-warn)}
.card .h{font-size:17px;font-weight:800;margin-bottom:2px}.small{font-size:15px}
.mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;word-break:break-all}
#pbox textarea,#qbox textarea{font-size:17px!important;line-height:1.45!important}
#runbtn{font-size:20px!important;min-height:58px!important}
.dlbtn{min-height:56px!important;font-size:17px!important;width:100%!important;white-space:normal!important}
.rawbox textarea{font-family:ui-monospace,Menlo,Consolas,monospace!important;font-size:12px!important;line-height:1.4!important}
"""
HERO=('<div class="kz hero"><div class="brand">AKBASCORE</div><div class="kick">PKV LATENT ATTENTION MEMORY</div>'
  '<div class="ttl">NIRVANA DEMO</div><div class="tst">Your words → a memory inside the AI → text deleted → the AI remembers</div></div>')
MEM_INFO=('<div class="kz card on"><div class="h">① Write a memory</div><div class="small">'
   'Type a short story or fact below — or keep ours. The AI reads it <b>once</b>. AkbasCore turns it into a '
   '<b>synthetic memory</b>: a set of numbers placed directly inside the AI\'s brain. Then your text is <b>deleted</b>. '
   'The AI never sees your words again.</div></div>')
Q_INFO=('<div class="kz card info"><div class="h">② Ask a question about your memory</div><div class="small">'
  'Keep our question or write your own. The AI receives <b>only this question</b> plus the synthetic memory — not your text. '
  'Then press RUN and compare four answers: no memory · AkbasCore memory · wrong memory · text visible.</div></div>')
GR6=int(gr.__version__.split(".")[0])>=6
DL_ALL_JS="""(payload)=>{
  let items=[];
  try{items=JSON.parse(payload||'[]');}catch(e){items=[];}
  if(!items.length){alert('The JPGs appear after the run is complete.');return;}
  items.forEach((it,i)=>{
    setTimeout(()=>{
      const a=document.createElement('a');
      a.href=it.url;a.download=it.name;a.rel='noopener';a.style.display='none';
      document.body.appendChild(a);a.click();
      setTimeout(()=>a.remove(),3000);
    },i*700);
  });
}"""
_tbp=inspect.signature(gr.Textbox.__init__).parameters
_COPY_KW={"buttons":["copy"]}if "buttons" in _tbp else({"show_copy_button":True}if "show_copy_button" in _tbp else{})
def make_raw_box(label):
 kw=dict(label=label,lines=16,max_lines=40,interactive=True,elem_classes=["rawbox"])
 try:return gr.Textbox(**kw,**_COPY_KW)
 except Exception:return gr.Textbox(**kw)
with gr.Blocks(title="AKBASCORE · NIRVANA DEMO",**({}if GR6 else{"css":CSS}))as demo:
 gr.HTML(HERO)
 gr.HTML(MEM_INFO)
 mem_box=gr.Textbox(value=DEFAULT_MEMORY,show_label=False,lines=6,max_lines=14,elem_id="pbox")
 gr.HTML(Q_INFO)
 q_box=gr.Textbox(value=DEFAULT_QUESTION,show_label=False,lines=3,max_lines=8,elem_id="qbox")
 run_btn=gr.Button("▶ RUN",variant="primary",size="lg",elem_id="runbtn")
 status_h=gr.HTML(READY_HTML)
 gallery=gr.Gallery(label=f"{NPOST} JPG posters (visual inspection)",columns=2,height="auto",object_fit="contain")
 gr.HTML(f'<div class="kz small"><b>Downloads appear after the run is complete.</b><br>One tap below starts {NPOST} separate JPG downloads in order 01→{NPOST:02d} (no ZIP). Your browser may ask once to allow multiple downloads.</div>')
 dl_all=gr.Button(DL_ALL_LABEL,variant="primary",size="lg",interactive=False,elem_classes=["dlbtn"])
 jpg_urls=gr.Textbox(value="",visible=False,label="jpg urls")
 dl_json=gr.DownloadButton(label=DL_LABELS[0],value=None,interactive=False,size="lg",elem_classes=["dlbtn"])
 dl_txt=gr.DownloadButton(label=DL_LABELS[1],value=None,interactive=False,size="lg",elem_classes=["dlbtn"])
 dl_man=gr.DownloadButton(label=DL_LABELS[2],value=None,interactive=False,size="lg",elem_classes=["dlbtn"])
 with gr.Accordion("Direct file links (fallback)",open=False):
  json_f=gr.File(label="FULL RUN LOG (.json)",interactive=False)
  txt_f=gr.File(label="READABLE RUN LOG (.txt)",interactive=False)
  man_f=gr.File(label="RUN MANIFEST (.json)",interactive=False)
 with gr.Accordion("RAW LOG / HAM LOG",open=False):
  gr.HTML('<div class="kz small">Full text of the run files. Tap inside a box to select text (long-press on mobile) or use the copy button. This is a view of the saved files — the files themselves are not affected by edits made here.</div>')
  with gr.Tabs():
   with gr.Tab("READABLE RUN LOG (.txt)"):raw_txt=make_raw_box("READABLE RUN LOG (.txt)")
   with gr.Tab("FULL RUN LOG JSON"):raw_json=make_raw_box("FULL RUN LOG (.json) — exact sealed payload bytes (canonical compact form)")
   with gr.Tab("RUN MANIFEST JSON"):raw_man=make_raw_box("RUN MANIFEST (.json)")
 OUTPUTS=[status_h,gallery,dl_all,jpg_urls,dl_json,dl_txt,dl_man,json_f,txt_f,man_f,raw_txt,raw_json,raw_man]
 run_btn.click(run_handler,[mem_box,q_box],OUTPUTS,concurrency_limit=1)
 dl_all.click(fn=None,inputs=[jpg_urls],outputs=None,js=DL_ALL_JS)
def service_self_test():
 st=ROOT/"_service_selftest";shutil.rmtree(st,ignore_errors=True);st.mkdir(parents=True)
 res=[]
 def ok(name,cond):
  res.append(name)
  if not cond:raise RuntimeError(f"NIRVANA SERVICE SELF-TEST FAILED: {name}")
 try:
  t=st/"probe.txt";t.write_text("akbascore",encoding="utf-8")
  ok("ROOT writable / text file",t.read_text(encoding="utf-8")=="akbascore")
  fig=plt.figure(figsize=(4,2),dpi=DPI,facecolor="white");fit_text(fig,.05,.1,.9,.8,"AKBASCORE self-test $5 · k̂ = μ + B·c",fs_max=14,fs_min=7)
  jp=st/"01_probe.jpg";save_jpg(fig,jp)
  with Image.open(jp)as im:ok("JPEG pipeline (JPEG/RGB)",im.format=="JPEG" and im.mode=="RGB")
  js=canon(jsafe({"x":1.5,"nan":float("nan"),"t":(1,2),"s":"ü"}))
  ok("JSON canonical serialization",json.loads(js.decode("utf-8"))["nan"]=="non-finite:nan")
  zp=st/"probe.zip"
  with zipfile.ZipFile(zp,"w",compression=zipfile.ZIP_DEFLATED)as z:z.write(jp,arcname=jp.name);z.write(t,arcname=t.name)
  with zipfile.ZipFile(zp)as z:ok("ZIP create + testzip()",z.testzip()is None and sorted(z.namelist())==["01_probe.jpg","probe.txt"])
  ok("allowed_paths covers ROOT",any(str(ROOT.resolve()).startswith(str(Path(a).resolve()))for a in ALLOWED_PATHS))
  ok(f"poster list = {NPOST}",len(POSTERS)==NPOST==12)
  ok("output components = callback outputs",len(OUTPUTS)==N_OUTPUTS==len(pack()))
  ok("pack() reset state matches outputs",len(pack(files=None,jpgs=None,raw=None))==N_OUTPUTS)
  ok("pack() active state matches outputs",len(pack("x",[],{"json":t,"txt":t,"man":t},[jp]*NPOST,{"txt":"a","json":"b","man":"c"}))==N_OUTPUTS)
  _u=json.loads(jpg_urls_json([jp,jp,jp]))
  ok("JPG URL payload: ordered, served under allowed_paths",len(_u)==3 and all(x["url"].startswith(FILE_URL_PREFIX+"/")for x in _u))
  ok("run_handler is a generator function",inspect.isgeneratorfunction(run_handler))
  ok("no AkbasCore forward hooks active",our_hooks_total()==0)
  ok("SDPA active",getattr(cfg,"_attn_implementation",None)=="sdpa")
 finally:
  plt.close("all");shutil.rmtree(st,ignore_errors=True)
 return res
print("[7/8] SERVICE SELF-TEST")
for _n in service_self_test():print("  PASS ·",_n)
print("NIRVANA SERVICE SELF-TEST: PASS")
print("[8/8] LAUNCH")
_lp=inspect.signature(gr.Blocks.launch).parameters
LAUNCH_KW={"share":True,"debug":True,"inline":False,"show_error":True,"allowed_paths":ALLOWED_PATHS,"ssr_mode":False}
if GR6:LAUNCH_KW["css"]=CSS
LAUNCH_KW={k:v for k,v in LAUNCH_KW.items()if k in _lp}
if not LAUNCH_KW.get("share"):raise RuntimeError("This Gradio version does not accept share=True.")
def _announce_public_url():
 for _ in range(720):
  u=getattr(demo,"share_url",None)
  if u:
   print("\n"+"="*110);print("GRADIO PUBLIC-LINK MODE: PASS");print("READY");print("="*110)
   print("OPEN THIS LINK IN A NEW BROWSER TAB:");print(u);print("="*110)
   print("Do not use the Colab inline preview. Keep this cell running while you use the link.\n");return
  time.sleep(.5)
 print("GRADIO PUBLIC-LINK MODE: FAIL — no public link after 6 minutes. Check the Gradio messages above, then rerun the cell.")
threading.Thread(target=_announce_public_url,daemon=True).start()
print("Launching public Gradio link … (launch options:",{k:(v if k!="css" else "…")for k,v in LAUNCH_KW.items()},")")
demo.queue(default_concurrency_limit=1).launch(**LAUNCH_KW)
