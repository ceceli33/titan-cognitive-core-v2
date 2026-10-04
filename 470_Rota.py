# ==================================================================================================
# TEST 470 — AKBASCORE NIRVANA × MISTRAL — QWEN IBR ENGINE PORT
# TEST468 MISTRAL D120 ENGINE PRESERVED | 16 INDEPENDENT CARTRIDGES
# QWEN REFERENCE LOGIC: MW1 → decide_id → MW2 → decide_place
# NO ROUTER | NO GOLD TARGET SELECTION | MODULE-WISE ISOLATED READOUT | NONE/UNKNOWN
# SOURCE-ONLY FORGE | OWN | BF16 SDPA | GREEDY | FROZEN | NO TRAINING / LoRA / OPTIMIZER
# ==================================================================================================
import os,sys,json,random,hashlib,gc,subprocess,importlib.util,re
from pathlib import Path
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
 if importlib.util.find_spec(m)is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import torch,transformers
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required")
torch.set_grad_enabled(False)
SEED=470;random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEV="cuda";MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";D=120;DMAX=128;MAX_NEW=32;SEP="\n\n"
FMT="QUESTION:\n{q}\n\nANSWER:"
MW1='Which container identifier is associated with the object "{obj}"? If this memory does not contain that association, answer NONE.'
MW2='Where is container {cid} located? If this memory does not contain that identifier, answer NONE.'
ROOT=Path("/content/AKBASCORE_TEST470")if os.path.isdir("/content")else Path("/tmp/AKBASCORE_TEST470")
ROOT.mkdir(parents=True,exist_ok=True);LOG=ROOT/"TEST470_RESULTS.jsonl";SUMMARY=ROOT/"TEST470_SUMMARY.json";LOG.write_text("",encoding="utf-8")
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
def norm(x):return re.sub(r"[^\w]+"," ",str(x).casefold()).strip()
def first(text):
 line=next((s.strip()for s in str(text).splitlines()if s.strip()),"")
 line=re.sub(r"^\W*(final answer|answer)\s*[:\-]\s*","",line,flags=re.I).strip(" *`\"'")
 return re.split(r"(?<=[.!?])\s+",line)[0]if line else""
ID_RE=re.compile(r"\b([A-Z]{2}-\d{3})\b",re.I)
AMBIG={"not","no","never","nor","or","either","maybe","perhaps","possibly","probably","might","unclear","but","unsure"}
def classify(text):
 fl=first(text);n=norm(fl);w=n.split();ii=sorted({m.upper()for m in ID_RE.findall(fl)})
 if not w:return"empty",ii,fl
 if set(w)<={"none","unknown"}:return"none",ii,fl
 if(set(w)&AMBIG)or len(ii)>1 or(set(w)&{"none","unknown"}):return"ambiguous",ii,fl
 return"answer",ii,fl
def parse_id(text):
 c,ii,_=classify(text);return ii[0]if c=="answer"and len(ii)==1 else None
def place_key(a):return re.sub(r"^(?:the|in the|at the|in|at)\s+","",norm(a)).strip()
def parse_place(text):
 c,ii,fl=classify(text)
 if c!="answer"or ii:return None
 k=place_key(fl);hits=[p for p in PLACES if place_key(p)==k or place_key(p)in k]
 return hits[0]if len(hits)==1 else None
def decide_id(texts):
 u=sorted({x for x in(parse_id(t)for t in texts)if x});return u[0]if len(u)==1 else None
def decide_place(texts):
 u=sorted({x for x in(parse_place(t)for t in texts)if x});return u[0]if len(u)==1 else None
def rec(x):
 with LOG.open("a",encoding="utf-8")as f:f.write(json.dumps(x,ensure_ascii=False,default=str)+"\n")
def fail(x):raise RuntimeError(x)
print("="*122);print("TEST 470 — NIRVANA × MISTRAL — QWEN IBR ENGINE PORT");print("="*122)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/8] MODEL")
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
print("[2/8] FIXED NEUTRAL CODEBOOK · D120")
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
def gen(card,q):
 K,V,T=install(card["K"],card["V"]);qi=ids(FMT.format(q=q));x=torch.tensor([[PAD]*T+qi],device=DEV);c=cache_of(K,V)
 y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),past_key_values=c,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=PAD,eos_token_id=EOS)
 return tok.decode(y[0,x.shape[1]:],skip_special_tokens=True).strip()
@torch.inference_mode()
def nmem(q):
 qi=ids(FMT.format(q=q));x=torch.tensor([qi],device=DEV)
 y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=PAD,eos_token_id=EOS)
 return tok.decode(y[0,x.shape[1]:],skip_special_tokens=True).strip()
print("[3/8] FORGE · 16 INDEPENDENT CARTRIDGES")
CARDS=[]
for i in range(16):
 src=f"The {OBJECTS[i]} is associated with container {IDS[i]}. Container {IDS[i]} is located at the {PLACES[i]}."
 f=forge(src);c=compress(f);CARDS.append(c);del f
 print(f" C{i+1:02d}: T={c['T']:02d} | {OBJECTS[i]} → {IDS[i]} → {PLACES[i]}");clean()
print("[4/8] MODULE-WISE STAGE-1 · OBJECT → ID")
S1=[];S1RAW=[]
for i in range(16):
 q=MW1.format(obj=OBJECTS[i]);raw=[gen(c,q)for c in CARDS];pred=decide_id(raw);ok=int(pred==IDS[i]);S1.append(ok);S1RAW.append(raw)
 print(f" {i+1:02d}: {'PASS'if ok else'FAIL'} | expected={IDS[i]} | decided={pred or 'UNKNOWN'}")
 for j,t in enumerate(raw):
  p=parse_id(t)
  if p or classify(t)[0]!="none":print(f"      C{j+1:02d}: {first(t)[:100]}")
 rec({"phase":"MW1","i":i,"object":OBJECTS[i],"expected":IDS[i],"decided":pred,"hit":ok,"raw":raw})
print(f"MW1={sum(S1)}/16")
print("[5/8] MODULE-WISE STAGE-2 · DECIDED ID → PLACE")
S2=[];LINK=[];S2RAW=[]
for i in range(16):
 cid=decide_id(S1RAW[i])
 if cid is None:
  S2.append(0);LINK.append(0);S2RAW.append([]);print(f" {i+1:02d}: FAIL | Stage1 UNKNOWN");continue
 q=MW2.format(cid=cid);raw=[gen(c,q)for c in CARDS];pred=decide_place(raw);ok=int(pred==PLACES[i]);lk=int(S1[i]and ok);S2.append(ok);LINK.append(lk);S2RAW.append(raw)
 print(f" {i+1:02d}: {'PASS'if ok else'FAIL'} | {cid} → expected={PLACES[i]} | decided={pred or 'UNKNOWN'}")
 for j,t in enumerate(raw):
  p=parse_place(t)
  if p or classify(t)[0]!="none":print(f"      C{j+1:02d}: {first(t)[:100]}")
 rec({"phase":"MW2","i":i,"cid":cid,"expected":PLACES[i],"decided":pred,"hit":ok,"linked":lk,"raw":raw})
print(f"MW2={sum(S2)}/16 | LINKED={sum(LINK)}/16")
print("[6/8] MISSING / ABSENT-ID / NOMEM")
MISS=ABS=NM=0
for i in range(8):
 mo=f"missing object {i+1}";aid=f"ZX-{901+i}"
 r1=[gen(c,MW1.format(obj=mo))for c in CARDS];d1=decide_id(r1);m=int(d1 is None);MISS+=m
 r2=[gen(c,MW2.format(cid=aid))for c in CARDS];d2=decide_place(r2);a=int(d2 is None);ABS+=a
 no=nmem(MW2.format(cid=IDS[i]));nh=int(PLACES[i]not in norm(no));NM+=nh
 print(f" {i+1}: MISSING={'PASS'if m else'FAIL'}({d1 or 'UNKNOWN'}) | ABSENT={'PASS'if a else'FAIL'}({d2 or 'UNKNOWN'}) | NOMEM={nh}")
 rec({"phase":"CONTROL","i":i,"missing_decided":d1,"missing_pass":m,"absent_decided":d2,"absent_pass":a,"nomem":no,"nomem_pass":nh})
print(f"MISSING={MISS}/8 | ABSENT-ID={ABS}/8 | NOMEM={NM}/8")
print("[7/8] VERDICT")
GATES={"mw1":sum(S1)>=14,"mw2":sum(S2)>=14,"linked":sum(LINK)>=14,"missing":MISS>=7,"absent_id":ABS>=7,"nomem":NM>=7}
for k,v in GATES.items():print(f" {k:14s}: {'PASS'if v else'FAIL'}")
VERDICT="PASS_QWEN_IBR_PORT"if all(GATES.values())else"FAIL_QWEN_IBR_PORT"
print("VERDICT:",VERDICT)
print("[8/8] INTEGRITY + SUMMARY")
SENT1=sentinel()
if SENT1!=S0:fail("WEIGHT SENTINEL CHANGED")
if model.training or any(p.requires_grad for p in model.parameters()):fail("Frozen model audit failed")
report={"test":470,"model":MODEL_ID,"engine":"TEST468 Mistral D120 + Qwen module-wise IBR decision logic","D":D,"cards":16,
"results":{"mw1":[sum(S1),16],"mw2":[sum(S2),16],"linked":[sum(LINK),16],"missing":[MISS,8],"absent_id":[ABS,8],"nomem":[NM,8]},
"gates":GATES,"verdict":VERDICT,"sentinel_before":S0,"sentinel_after":SENT1,
"notes":["No router or gold-target cartridge selection.","All 16 cartridges independently answer every module-wise query.",
"Stage1 uses unique valid ID aggregation; Stage2 uses unique valid place aggregation.","Stage2 receives only the ID decided by Stage1.",
"TEST468 Mistral D120 forge/compression/cache path retained.","OWN slot retained.","No joint cache.","No training, LoRA, optimizer or weight update."]}
SUMMARY.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print("WEIGHT SENTINEL: PASS");print("RAW LOG :",LOG);print("SUMMARY :",SUMMARY);print("="*122)
