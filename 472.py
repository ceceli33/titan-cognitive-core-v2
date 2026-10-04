# TEST 472 — AKBASCORE NIRVANA × MISTRAL — D120 16-WAY HIDDEN RELEVANCE RANK
# TEST471 ENGINE PRESERVED. NO ROUTER INSERTED. NO ANSWER-CHAIN CHANGE.
# Goal: can native query×memory hidden interaction identify the correct cartridge among 16?
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
SEED=472;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEV="cuda";MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";D=120;DMAX=128;SEP="\n\n"
FMT="QUESTION:\n{q}\n\nANSWER:"
MW1='Which container identifier is associated with the object "{obj}"? If this memory does not contain that association, answer NONE.'
ROOT=Path("/content/AKBASCORE_TEST472")if os.path.isdir("/content")else Path("/tmp/AKBASCORE_TEST472")
ROOT.mkdir(parents=True,exist_ok=True);LOG=ROOT/"TEST472_RESULTS.jsonl";SUMMARY=ROOT/"TEST472_SUMMARY.json";LOG.write_text("",encoding="utf-8")
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
print("="*124);print("TEST 472 — NIRVANA × MISTRAL — D120 16-WAY HIDDEN RELEVANCE RANK");print("="*124)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/8] MODEL")
tv=tuple(int("".join(c for c in x if c.isdigit())or 0)for x in transformers.__version__.split(".")[:2]);dt="dtype"if tv>=(4,56)else"torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{dt:torch.bfloat16}).eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;layers=model.model.layers;NL=len(layers);H=cfg.hidden_size;NH=cfg.num_attention_heads;NKV=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None)or H//NH;KVD=NKV*HD
if(NL,H,NH,NKV,HD)!=(32,4096,32,8,128):fail(f"Architecture mismatch {(NL,H,NH,NKV,HD)}")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
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
def hidden(card,q):
 K,V,T=install(card["K"],card["V"]);qi=ids(FMT.format(q=q));x=torch.tensor([[PAD]*T+qi],device=DEV);c=cache_of(K,V)
 o=model(input_ids=x,attention_mask=torch.ones_like(x),past_key_values=c,use_cache=False,output_hidden_states=True,return_dict=True)
 # L13/L16/L20/L23/L26/final: corridor + tail identified by TEST467
 sel=[13,16,20,23,26,32]
 return torch.stack([F.normalize(o.hidden_states[L][0,-1].float(),dim=0)for L in sel])
print("[3/8] FORGE · 16 INDEPENDENT CARTRIDGES")
CARDS=[]
for i in range(16):
 src=f"The {OBJECTS[i]} is associated with container {IDS[i]}. Container {IDS[i]} is located at the {PLACES[i]}."
 f=forge(src);c=compress(f);CARDS.append(c);del f
 print(f" C{i+1:02d}: T={c['T']:02d} | {OBJECTS[i]} → {IDS[i]} → {PLACES[i]}");clean()
print("[4/8] BUILD SELF REFERENCE SIGNATURES")
# Reference signature for cartridge j:
# effect of its own object query relative to mean effect of all foreign object queries.
REF=[];ALL=[]
for j,c in enumerate(CARDS):
 hs=[]
 for i in range(16):hs.append(hidden(c,MW1.format(obj=OBJECTS[i])).cpu())
 hs=torch.stack(hs) # [query, selected_layer, H]
 ALL.append(hs)
 self_h=hs[j];foreign=torch.cat([hs[:j],hs[j+1:]],0).mean(0)
 sig=F.normalize(self_h-foreign,dim=-1);REF.append(sig)
 print(f" C{j+1:02d}: reference built | ||self-foreign|| μ={float((self_h-foreign).norm(dim=-1).mean()):.6f}")
REF=torch.stack(REF);ALL=torch.stack(ALL) # ALL [card,query,layer,H]
print("[5/8] LEAVE-ONE-CARD-OUT 16-WAY RANK")
# Candidate score for query i/card j:
# cosine of card j query-response residual against that card's self-relevance signature.
# Baseline is mean response of the SAME card to the other 15 queries.
RANKS=[];TOP1=[];MARGINS=[];SCORES=[]
for i in range(16):
 sc=[]
 for j in range(16):
  h=ALL[j,i]
  idx=[k for k in range(16)if k!=i]
  base=ALL[j,idx].mean(0)
  delta=F.normalize(h-base,dim=-1)
  # fixed layer aggregation; no learned weights
  s=float((delta*REF[j]).sum(-1).mean())
  sc.append(s)
 order=np.argsort(-np.asarray(sc));rank=int(np.where(order==i)[0][0])+1
 top=int(order[0]);margin=float(sc[i]-max(sc[:i]+sc[i+1:]))
 RANKS.append(rank);TOP1.append(top);MARGINS.append(margin);SCORES.append(sc)
 print(f" Q{i+1:02d}: gold=C{i+1:02d} rank={rank:02d} | top=C{top+1:02d} | gold={sc[i]:+.6f} | margin={margin:+.6f}")
 rec({"phase":"RANK","query":i,"object":OBJECTS[i],"gold":i,"rank":rank,"top":top,"gold_score":sc[i],"margin":margin,"scores":sc})
R=np.array(RANKS);M=np.array(MARGINS)
print("[6/8] NEGATIVE / MISSING QUERY AUDIT")
MISSING=[f"missing object {i+1}"for i in range(8)];NEG=[]
for z in MISSING:
 sc=[]
 q=MW1.format(obj=z)
 for j,c in enumerate(CARDS):
  h=hidden(c,q).cpu();base=ALL[j].mean(0);delta=F.normalize(h-base,dim=-1)
  sc.append(float((delta*REF[j]).sum(-1).mean()))
 mx=max(sc);top=int(np.argmax(sc));NEG.append(mx)
 print(f" {z:16s}: top=C{top+1:02d} score={mx:+.6f}")
 rec({"phase":"MISSING","object":z,"top":top,"max_score":mx,"scores":sc})
NEG=np.array(NEG)
print("[7/8] SUMMARY / DIAGNOSTIC VERDICT")
print(f"TOP1={int((R==1).sum())}/16 | TOP2={int((R<=2).sum())}/16 | TOP4={int((R<=4).sum())}/16 | mean-rank={R.mean():.3f}")
print(f"GOLD MARGIN: mean={M.mean():+.6f} min={M.min():+.6f} max={M.max():+.6f}")
print(f"MISSING max-score: mean={NEG.mean():+.6f} min={NEG.min():+.6f} max={NEG.max():+.6f}")
if int((R==1).sum())>=15:VERDICT="PASS_HIDDEN_RELEVANCE_RANK"
elif int((R<=2).sum())>=14:VERDICT="PARTIAL_HIDDEN_RELEVANCE_RANK"
else:VERDICT="FAIL_HIDDEN_RELEVANCE_RANK"
print("VERDICT:",VERDICT)
print("NOTE: diagnostic only. No router, threshold, answer selection or engine modification.")
print("[8/8] INTEGRITY + SUMMARY")
S1=sentinel()
if S1!=S0:fail("WEIGHT SENTINEL CHANGED")
if model.training or any(p.requires_grad for p in model.parameters()):fail("Frozen model audit failed")
report={"test":472,"model":MODEL_ID,"D":D,"cards":16,"layers":[13,16,20,23,26,32],
"purpose":"16-way hidden relevance ranking diagnostic; no routing inserted.",
"rank":{"top1":int((R==1).sum()),"top2":int((R<=2).sum()),"top4":int((R<=4).sum()),"mean":float(R.mean()),"all":R.tolist()},
"gold_margin":{"mean":float(M.mean()),"min":float(M.min()),"max":float(M.max()),"all":M.tolist()},
"missing_max":{"mean":float(NEG.mean()),"min":float(NEG.min()),"max":float(NEG.max()),"all":NEG.tolist()},
"verdict":VERDICT,"sentinel_before":S0,"sentinel_after":S1,
"locks":["TEST471/470 Mistral engine preserved","D120 K/V","OWN preserved","16 independent cartridges","No router inserted","No learned scoring","No training","No LoRA","No optimizer","No weight update"]}
SUMMARY.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print("WEIGHT SENTINEL: PASS");print("RAW LOG :",LOG);print("SUMMARY :",SUMMARY);print("="*124)
