# TEST 473 — AKBASCORE NIRVANA × MISTRAL — D120 QWEN-STYLE ABSTENTION FRAME CALIBRATION
# TEST470 MEMORY ENGINE LOCKED.
# NO ROUTER / NO HIDDEN SCORER / NO TRAINING / NO LoRA / NO WEIGHT UPDATE.
# Controlled variable only: source/readout framing for module-local relevance/abstention.
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
SEED=473;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEV="cuda";MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";D=120;DMAX=128;MAX_NEW=32;SEP="\n\n"
ROOT=Path("/content/AKBASCORE_TEST473")if os.path.isdir("/content")else Path("/tmp/AKBASCORE_TEST473")
ROOT.mkdir(parents=True,exist_ok=True);LOG=ROOT/"TEST473_RESULTS.jsonl";SUMMARY=ROOT/"TEST473_SUMMARY.json";LOG.write_text("",encoding="utf-8")
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
print("="*126);print("TEST 473 — NIRVANA × MISTRAL — D120 QWEN-STYLE ABSTENTION FRAME CALIBRATION");print("="*126)
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
# Four fixed framing conditions. A reproduces TEST470. No learned selection.
FRAMES={
"A_470":lambda o:f'QUESTION:\nWhich container identifier is associated with the object "{o}"? If this memory does not contain that association, answer NONE.\n\nANSWER:',
"B_STRICT":lambda o:f'QUESTION:\nAccording only to this memory, which container identifier is associated with "{o}"?\nIf this exact object is not present in this memory, answer exactly NONE.\n\nANSWER:',
"C_VERIFY":lambda o:f'QUESTION:\nDoes this memory explicitly contain the object "{o}"? If yes, answer only its container identifier. If no, answer exactly NONE.\n\nANSWER:',
"D_EXTRACT":lambda o:f'QUESTION:\nFind the exact object "{o}" in this memory. Return only its associated container identifier. If the exact object does not occur in this memory, return NONE.\n\nANSWER:'
}
# Source variants. S0 reproduces TEST470 source. S1/S2 only make module boundaries/closed-world semantics explicit.
SRC={
"S0_470":lambda o,c,p:f"The {o} is associated with container {c}. Container {c} is located at the {p}.",
"S1_RECORD":lambda o,c,p:f"MEMORY RECORD\nObject: {o}\nContainer: {c}\nLocation: {p}",
"S2_CLOSED":lambda o,c,p:f"This memory contains exactly one object record. Object: {o}. Container identifier: {c}. Location: {p}. No other object is recorded in this memory."
}
print("[3/9] FORGE · 3 SOURCE FRAMES × 16 CARTRIDGES")
BANK={}
for sn,sf in SRC.items():
 cards=[]
 print(" ",sn)
 for i in range(16):
  f=forge(sf(OBJECTS[i],IDS[i],PLACES[i]));c=compress(f);cards.append(c);del f
  print(f"  C{i+1:02d}: T={c['T']:02d} | {OBJECTS[i]} → {IDS[i]}")
 BANK[sn]=cards;clean()
print("[4/9] CONTROLLED GOLD/DECOY ABSTENTION MATRIX")
RESULTS={}
for sn,cards in BANK.items():
 for fn,pf in FRAMES.items():
  gold=0;decoy_none=0;decoy_total=0;false_ids=0;own_false=0;examples=[]
  for i in range(16):
   for j,c in enumerate(cards):
    raw=gen(c,pf(OBJECTS[i]));pid=parse_id(raw);none=is_none(raw)
    if j==i:
     gold+=int(pid==IDS[i])
    else:
     decoy_total+=1;decoy_none+=int(none);false_ids+=int(pid is not None);own_false+=int(pid==IDS[j])
     if len(examples)<4 and not none:examples.append((i,j,first(raw)))
  key=f"{sn}/{fn}";rate=decoy_none/decoy_total
  RESULTS[key]={"source":sn,"frame":fn,"gold":gold,"decoy_none":decoy_none,"decoy_total":decoy_total,"false_ids":false_ids,"own_false":own_false,"rate":rate,"examples":examples}
  print(f" {key:20s} | GOLD={gold:02d}/16 | DECOY_NONE={decoy_none:03d}/{decoy_total} ({rate:.3f}) | falseID={false_ids:03d} | ownID={own_false:03d}")
  rec({"phase":"FRAME_MATRIX",**RESULTS[key]})
print("[5/9] RANK CONDITIONS")
# Selection is diagnostic and deterministic:
# primary = preserve gold retrieval, secondary = maximize decoy NONE.
ranked=sorted(RESULTS.items(),key=lambda kv:(kv[1]["gold"],kv[1]["decoy_none"]),reverse=True)
for n,(k,r) in enumerate(ranked,1):
 print(f" {n:02d}. {k:20s} GOLD={r['gold']:02d}/16 DECOY_NONE={r['decoy_none']:03d}/240")
best_key,best=ranked[0];BEST_SRC=best["source"];BEST_FRAME=best["frame"]
print("BEST:",best_key)
print("[6/9] BEST CONDITION · FULL MODULE-WISE DECISION")
cards=BANK[BEST_SRC];pf=FRAMES[BEST_FRAME]
MW=0;UNIQUE=0
for i in range(16):
 vals=[];raws=[]
 for j,c in enumerate(cards):
  raw=gen(c,pf(OBJECTS[i]));pid=parse_id(raw);raws.append(raw)
  if pid is not None:vals.append(pid)
 u=sorted(set(vals));dec=u[0]if len(u)==1 else None;ok=dec==IDS[i];MW+=int(ok);UNIQUE+=int(len(u)==1)
 print(f" {i+1:02d}: {'PASS'if ok else'FAIL'} | expected={IDS[i]} | decided={dec or'UNKNOWN'} | unique={len(u)}")
 if not ok:
  for j,r in enumerate(raws):
   if not is_none(r):print(f"      C{j+1:02d}: {first(r)[:72]}")
 rec({"phase":"BEST_DECISION","query":i,"expected":IDS[i],"unique_ids":u,"decided":dec,"ok":ok})
print(f"MW1={MW}/16 | UNIQUE={UNIQUE}/16")
print("[7/9] MISSING OBJECT CONTROL")
MISS=[f"missing object {i+1}"for i in range(8)];MISSING=0
for i,o in enumerate(MISS):
 vals=[];nn=0
 for c in cards:
  raw=gen(c,pf(o));nn+=int(is_none(raw));pid=parse_id(raw)
  if pid is not None:vals.append(pid)
 u=sorted(set(vals));ok=len(u)==0;MISSING+=int(ok)
 print(f" {i+1}: {'PASS'if ok else'FAIL'} | NONE={nn}/16 | aggregate={'UNKNOWN'if ok else u}")
 rec({"phase":"MISSING","query":o,"none":nn,"ids":u,"ok":ok})
print(f"MISSING={MISSING}/8")
print("[8/9] VERDICT")
base=RESULTS["S0_470/A_470"]
print(f"BASE TEST470: GOLD={base['gold']}/16 | DECOY_NONE={base['decoy_none']}/240")
print(f"BEST        : GOLD={best['gold']}/16 | DECOY_NONE={best['decoy_none']}/240 | MW1={MW}/16 | MISSING={MISSING}/8")
gates={
"gold_preserved":best["gold"]>=15,
"decoy_abstention":best["decoy_none"]>=228,
"mw1":MW>=14,
"missing":MISSING>=7
}
for k,v in gates.items():print(f" {k:18s}: {'PASS'if v else'FAIL'}")
if all(gates.values()):VERDICT="PASS_MISTRAL_ABSTENTION_FRAME"
elif best["gold"]>=15 and best["decoy_none"]>base["decoy_none"]:VERDICT="PARTIAL_ABSTENTION_GAIN"
else:VERDICT="FAIL_ABSTENTION_FRAME"
print("VERDICT:",VERDICT)
print("[9/9] INTEGRITY + SUMMARY")
S1=sentinel()
if S1!=S0:fail("WEIGHT SENTINEL CHANGED")
if model.training or any(p.requires_grad for p in model.parameters()):fail("Frozen model audit failed")
report={"test":473,"model":MODEL_ID,"D":D,"cards":16,"purpose":"Controlled source/readout framing calibration for Qwen-style module abstention.",
"baseline":base,"best_key":best_key,"best":best,"mw1":MW,"unique":UNIQUE,"missing":MISSING,"gates":gates,"verdict":VERDICT,
"all_conditions":RESULTS,"sentinel_before":S0,"sentinel_after":S1,
"locks":["Mistral v0.3","D120 K/V","OWN preserved","TEST470 forge/compression/install unchanged","Independent cartridges","No router","No hidden scorer","No training","No LoRA","No optimizer","No weight update"]}
SUMMARY.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print("WEIGHT SENTINEL: PASS");print("RAW LOG :",LOG);print("SUMMARY :",SUMMARY);print("="*126)
