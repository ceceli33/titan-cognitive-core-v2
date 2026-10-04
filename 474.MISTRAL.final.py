# TEST 474 — AKBASCORE NIRVANA × MISTRAL — D120 FINAL TWO-STAGE ENGINE VALIDATION
# TEST473 WINNER LOCKED: S1_RECORD + C_VERIFY.
# OBJECT -> MW1 -> UNIQUE ID -> MW2 -> UNIQUE LOCATION.
# NO ROUTER / NO HIDDEN SCORER / NO TRAINING / NO LoRA / NO OPTIMIZER / NO WEIGHT UPDATE.
import os,sys,re,json,random,hashlib,gc,subprocess,importlib.util
from pathlib import Path
import numpy as np
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
 if importlib.util.find_spec(m)is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import torch,transformers
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required")
torch.set_grad_enabled(False)
SEED=474;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEV="cuda";MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";D=120;DMAX=128;MAX_NEW=32;SEP="\n\n"
ROOT=Path("/content/AKBASCORE_TEST474")if os.path.isdir("/content")else Path("/tmp/AKBASCORE_TEST474")
ROOT.mkdir(parents=True,exist_ok=True);LOG=ROOT/"TEST474_RESULTS.jsonl";SUMMARY=ROOT/"TEST474_SUMMARY.json";LOG.write_text("",encoding="utf-8")
OBJECTS=["amber sextant","bronze compass","cedar telescope","crimson lantern","ivory astrolabe","jade chronometer","silver monocle","copper barometer",
"onyx sundial","pearl compass","saffron telescope","violet lantern","marble astrolabe","teal chronometer","golden monocle","indigo barometer"]
IDS=[f"RQ-{415+i*17}"for i in range(16)]
PLACES=["elm lodge","pine archive","cedar vault","birch station","maple gallery","willow depot","oak library","fir observatory",
"ash museum","yew tower","spruce hall","linden archive","alder vault","beech station","hazel gallery","rowan depot"]
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
def clean():gc.collect();torch.cuda.empty_cache()
def rec(x):
 with LOG.open("a",encoding="utf-8")as f:f.write(json.dumps(x,ensure_ascii=False,default=str)+"\n")
def fail(x):raise RuntimeError(x)
def first(text):
 line=next((s.strip()for s in str(text).splitlines()if s.strip()),"")
 line=re.sub(r"^\W*(final answer|answer)\s*[:\-]\s*","",line,flags=re.I).strip(" *`\"'")
 return re.split(r"(?<=[.!?])\s+",line)[0]if line else""
ID_RE=re.compile(r"\b([A-Z]{2}-\d{3})\b",re.I)
def is_none(text):
 s=first(text).upper()
 return bool(re.search(r"\bNONE\b",s)) and not ID_RE.search(s)
def parse_id(text):
 s=first(text)
 if is_none(s):return None
 m=ID_RE.search(s);return m.group(1).upper()if m else None
def parse_place(text):
 s=first(text).casefold()
 if re.search(r"\bnone\b",s):return None
 hits=[p for p in PLACES if re.search(r"(?<!\w)"+re.escape(p.casefold())+r"(?!\w)",s)]
 return hits[0]if len(set(hits))==1 else None
def decide_id(raws):
 vals=[parse_id(x)for x in raws];u=sorted({x for x in vals if x is not None})
 return u[0]if len(u)==1 else None,u
def decide_place(raws):
 vals=[parse_place(x)for x in raws];u=sorted({x for x in vals if x is not None})
 return u[0]if len(u)==1 else None,u
def src(o,c,p):return f"MEMORY RECORD\nObject: {o}\nContainer: {c}\nLocation: {p}"
def q1(o):return f'QUESTION:\nDoes this memory explicitly contain the object "{o}"? If yes, answer only its container identifier. If no, answer exactly NONE.\n\nANSWER:'
def q2(c):return f'QUESTION:\nDoes this memory explicitly contain container "{c}"? If yes, answer only its location. If no, answer exactly NONE.\n\nANSWER:'
print("="*128);print("TEST 474 — NIRVANA × MISTRAL — D120 FINAL TWO-STAGE ENGINE VALIDATION");print("="*128)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/9] MODEL")
tv=tuple(int("".join(c for c in x if c.isdigit())or 0)for x in transformers.__version__.split(".")[:2]);dt="dtype"if tv>=(4,56)else"torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{dt:torch.bfloat16}).eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;layers=model.model.layers;NL=len(layers);H=cfg.hidden_size;NH=cfg.num_attention_heads;NKV=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None)or H//NH;KVD=NKV*HD
if(NL,H,NH,NKV,HD)!=(32,4096,32,8,128):fail(f"Architecture mismatch {(NL,H,NH,NKV,HD)}")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
ge=model.generation_config.eos_token_id;EOS=sorted({tok.eos_token_id,*(ge if isinstance(ge,(list,tuple))else[ge])}-{None})
FP=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[13].mlp.down_proj.weight,layers[23].self_attn.o_proj.weight,layers[26].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
def sentinel():
 h=hashlib.sha256()
 for p in FP:
  a=p.detach().reshape(-1);n=a.numel()
  for j in range(16):
   o=j*(n-256)//15;h.update(a[o:o+256].float().cpu().numpy().tobytes())
 return h.hexdigest()
S0=sentinel()
def ids(s):return tok(s,add_special_tokens=False).input_ids
def new_cache():
 try:return DynamicCache(config=cfg)
 except:return DynamicCache()
def cache_of(K,V):
 c=new_cache()
 for L in range(NL):c.update(K[L].clone(),V[L].clone(),L)
 return c
@torch.inference_mode()
def forge(s):
 seq=[PAD]+ids(s+SEP);o=model(input_ids=torch.tensor([seq],device=DEV),use_cache=False,output_hidden_states=True,return_dict=True);K=[];V=[]
 for L in range(NL):
  z=layers[L].input_layernorm(o.hidden_states[L][0]);a=layers[L].self_attn
  K.append(a.k_proj(z).contiguous());V.append(a.v_proj(z).contiguous())
 return{"T":len(seq),"K":K,"V":V}
@torch.inference_mode()
def install(K,V):
 T=K[0].shape[0];pos=torch.arange(T,device=DEV)[None];cos,sin=model.model.rotary_emb(K[0][None],pos);KK=[];VV=[]
 for L in range(NL):
  k=K[L].reshape(1,T,NKV,HD).transpose(1,2).contiguous();v=V[L].reshape(1,T,NKV,HD).transpose(1,2).contiguous()
  kr,_=apply_rotary_pos_emb(k,k,cos,sin,unsqueeze_dim=1);KK.append(kr.contiguous());VV.append(v)
 return tuple(KK),tuple(VV),T
print("[2/9] FIXED NEUTRAL CODEBOOK · D120")
CO=[forge(s)for s in NEUTRAL];CB=[]
for L in range(NL):
 e={}
 for n in("K","V"):
  R=torch.cat([c[n][L][1:]for c in CO]).float().reshape(-1,NKV,HD);mu=[];basis=[]
  for h in range(NKV):
   X=R[:,h];m=X.mean(0);_,_,vh=torch.linalg.svd(X-m,full_matrices=False);b=vh[:DMAX].T.contiguous()
   if b.shape[1]<DMAX:b=F.pad(b,(0,DMAX-b.shape[1]))
   mu.append(m);basis.append(b)
  e[n]=(torch.stack(mu),torch.stack(basis))
 CB.append(e)
del CO;clean()
@torch.inference_mode()
def compress(f):
 out={}
 for n in("K","V"):
  rows=[]
  for L in range(NL):
   mu,B=CB[L][n];x=f[n][L][1:].float().reshape(-1,NKV,HD);c=torch.einsum("thi,hid->thd",x-mu,B[:,:,:D]).to(torch.bfloat16)
   content=(mu+torch.einsum("thd,hid->thi",c.float(),B[:,:,:D])).reshape(-1,KVD).to(torch.bfloat16)
   rows.append(torch.cat([f[n][L][:1],content],0))
  out[n]=rows
 return{"K":out["K"],"V":out["V"],"T":f["T"]}
@torch.inference_mode()
def gen(card,prompt):
 K,V,T=install(card["K"],card["V"]);qi=ids(prompt);x=torch.tensor([[PAD]*T+qi],device=DEV);c=cache_of(K,V)
 y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),past_key_values=c,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=PAD,eos_token_id=EOS)
 return tok.decode(y[0,x.shape[1]:],skip_special_tokens=True).strip()
@torch.inference_mode()
def gen_nomem(prompt):
 qi=ids(prompt);x=torch.tensor([qi],device=DEV)
 y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=PAD,eos_token_id=EOS)
 return tok.decode(y[0,x.shape[1]:],skip_special_tokens=True).strip()
print("[3/9] FORGE · 16 LOCKED S1_RECORD CARTRIDGES")
CARDS=[]
for i in range(16):
 f=forge(src(OBJECTS[i],IDS[i],PLACES[i]));c=compress(f);CARDS.append(c);del f
 print(f" C{i+1:02d}: T={c['T']:02d} | {OBJECTS[i]} → {IDS[i]} → {PLACES[i]}");clean()
print("[4/9] MW1 · OBJECT → UNIQUE ID")
MW1=0;S1_DEC=[]
for i in range(16):
 raws=[gen(c,q1(OBJECTS[i]))for c in CARDS];dec,u=decide_id(raws);ok=dec==IDS[i];MW1+=int(ok);S1_DEC.append(dec)
 print(f" {i+1:02d}: {'PASS'if ok else'FAIL'} | expected={IDS[i]} | decided={dec or'UNKNOWN'} | unique={len(u)}")
 if not ok:
  for j,r in enumerate(raws):
   if not is_none(r):print(f"      C{j+1:02d}: {first(r)[:72]}")
 rec({"phase":"MW1","query":i,"object":OBJECTS[i],"expected":IDS[i],"decided":dec,"unique":u,"raw":raws,"ok":ok})
print(f"MW1={MW1}/16")
print("[5/9] MW2 · DECIDED ID → UNIQUE LOCATION")
MW2=0;LINKED=0
for i in range(16):
 cid=S1_DEC[i]
 if cid is None:
  print(f" {i+1:02d}: FAIL | Stage1 UNKNOWN");rec({"phase":"MW2","query":i,"stage1":None,"ok":False});continue
 raws=[gen(c,q2(cid))for c in CARDS];dec,u=decide_place(raws);ok=dec==PLACES[i];MW2+=int(ok);LINKED+=int(ok and cid==IDS[i])
 print(f" {i+1:02d}: {'PASS'if ok else'FAIL'} | {cid} | expected={PLACES[i]} | decided={dec or'UNKNOWN'} | unique={len(u)}")
 if not ok:
  for j,r in enumerate(raws):
   if not is_none(r):print(f"      C{j+1:02d}: {first(r)[:72]}")
 rec({"phase":"MW2","query":i,"id":cid,"expected":PLACES[i],"decided":dec,"unique":u,"raw":raws,"ok":ok})
print(f"MW2={MW2}/16 | LINKED={LINKED}/16")
print("[6/9] MISSING OBJECT / ABSENT ID")
MISSOBJ=[f"missing object {i+1}"for i in range(8)]
ABSIDS=[f"ZX-{801+i*13}"for i in range(8)]
MISSING=0;ABSENT=0
for i in range(8):
 r1=[gen(c,q1(MISSOBJ[i]))for c in CARDS];d1,u1=decide_id(r1);a=d1 is None and len(u1)==0;MISSING+=int(a)
 r2=[gen(c,q2(ABSIDS[i]))for c in CARDS];d2,u2=decide_place(r2);b=d2 is None and len(u2)==0;ABSENT+=int(b)
 print(f" {i+1}: MISSING={'PASS'if a else'FAIL'}({d1 or'UNKNOWN'}) | ABSENT={'PASS'if b else'FAIL'}({d2 or'UNKNOWN'})")
 rec({"phase":"NEGATIVE","i":i,"missing":{"query":MISSOBJ[i],"decision":d1,"unique":u1,"ok":a},"absent":{"query":ABSIDS[i],"decision":d2,"unique":u2,"ok":b}})
print(f"MISSING={MISSING}/8 | ABSENT-ID={ABSENT}/8")
print("[7/9] NOMEM CONTROL")
NOMEM=0
for i in range(8):
 raw=gen_nomem(q2(IDS[i]));p=parse_place(raw);ok=p!=PLACES[i];NOMEM+=int(ok)
 print(f" {i+1}: {'PASS'if ok else'FAIL'} | target={PLACES[i]} | raw={first(raw)[:72]}")
 rec({"phase":"NOMEM","i":i,"id":IDS[i],"target":PLACES[i],"raw":raw,"parsed":p,"ok":ok})
print(f"NOMEM={NOMEM}/8")
print("[8/9] FINAL VERDICT")
gates={"mw1":MW1>=15,"mw2":MW2>=15,"linked":LINKED>=15,"missing":MISSING>=7,"absent_id":ABSENT>=7,"nomem":NOMEM>=7}
for k,v in gates.items():print(f" {k:12s}: {'PASS'if v else'FAIL'}")
VERDICT="PASS_MISTRAL_FINAL_ENGINE"if all(gates.values())else"FAIL_MISTRAL_FINAL_ENGINE"
print("VERDICT:",VERDICT)
print("[9/9] INTEGRITY + SUMMARY")
S1=sentinel()
if S1!=S0:fail("WEIGHT SENTINEL CHANGED")
if model.training or any(p.requires_grad for p in model.parameters()):fail("Frozen model audit failed")
report={"test":474,"model":MODEL_ID,"D":D,"cards":16,"engine":"S1_RECORD + C_VERIFY / two-stage module-wise unique decision",
"results":{"mw1":[MW1,16],"mw2":[MW2,16],"linked":[LINKED,16],"missing":[MISSING,8],"absent_id":[ABSENT,8],"nomem":[NOMEM,8]},
"gates":gates,"verdict":VERDICT,"sentinel_before":S0,"sentinel_after":S1,
"locks":["Mistral-7B-Instruct-v0.3","D120 K/V","OWN preserved","Source-only forge","Source absent at readout","16 independent cartridges","S1_RECORD source frame","C_VERIFY abstention frame","Module-wise unique aggregation","Greedy","Frozen weights","No router","No hidden scorer","No training","No LoRA","No optimizer"]}
SUMMARY.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print("WEIGHT SENTINEL: PASS");print("RAW LOG :",LOG);print("SUMMARY :",SUMMARY);print("="*128)
