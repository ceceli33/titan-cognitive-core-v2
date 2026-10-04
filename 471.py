# TEST 471 — AKBASCORE NIRVANA × MISTRAL — D120 GOLD/DECOY RELEVANCE X-RAY
# TEST470 ENGINE PRESERVED. NO ROUTER. NO ENGINE CHANGE.
# Measures whether gold vs decoy cartridges are separable by native readout signals.
import os,sys,re,json,random,hashlib,gc,subprocess,importlib.util,math
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
SEED=471;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEV="cuda";MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";D=120;DMAX=128;MAX_NEW=32;SEP="\n\n"
FMT="QUESTION:\n{q}\n\nANSWER:"
MW1='Which container identifier is associated with the object "{obj}"? If this memory does not contain that association, answer NONE.'
ROOT=Path("/content/AKBASCORE_TEST471")if os.path.isdir("/content")else Path("/tmp/AKBASCORE_TEST471")
ROOT.mkdir(parents=True,exist_ok=True);LOG=ROOT/"TEST471_RESULTS.jsonl";SUMMARY=ROOT/"TEST471_SUMMARY.json";LOG.write_text("",encoding="utf-8")
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
def parse_id(text):
 m=ID_RE.search(first(text));return m.group(1).upper()if m else None
def rec(x):
 with LOG.open("a",encoding="utf-8")as f:f.write(json.dumps(x,ensure_ascii=False,default=str)+"\n")
def fail(x):raise RuntimeError(x)
print("="*124);print("TEST 471 — NIRVANA × MISTRAL — D120 GOLD/DECOY RELEVANCE X-RAY");print("="*124)
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
def forward_query(card,q):
 K,V,T=install(card["K"],card["V"]);qi=ids(FMT.format(q=q));x=torch.tensor([[PAD]*T+qi],device=DEV);c=cache_of(K,V)
 o=model(input_ids=x,attention_mask=torch.ones_like(x),past_key_values=c,use_cache=False,output_hidden_states=True,return_dict=True)
 return o.logits[0,-1].float(),o.hidden_states[-1][0,-1].float()
@torch.inference_mode()
def gen(card,q):
 K,V,T=install(card["K"],card["V"]);qi=ids(FMT.format(q=q));x=torch.tensor([[PAD]*T+qi],device=DEV);c=cache_of(K,V)
 y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),past_key_values=c,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=PAD,eos_token_id=EOS)
 return tok.decode(y[0,x.shape[1]:],skip_special_tokens=True).strip()
def variants(s):
 a=[" "+s,s,"\n"+s];out=[]
 for z in a:
  q=ids(z)
  if len(q)==1:out.append(q[0])
 return sorted(set(out))
NONE_TOK=variants("NONE")
if not NONE_TOK:
 NONE_TOK=ids(" NONE")[:1]
def lse(logits,tids):
 if not tids:return float("-inf")
 return float(torch.logsumexp(logits[torch.tensor(tids,device=logits.device)],0).item())
def target_first_tokens(cid):
 vs=variants(cid)
 if vs:return vs
 out=[]
 for z in[" "+cid,cid]:
  q=ids(z)
  if q:out.append(q[0])
 return sorted(set(out))
print("[3/8] FORGE · 16 INDEPENDENT CARTRIDGES")
CARDS=[]
for i in range(16):
 src=f"The {OBJECTS[i]} is associated with container {IDS[i]}. Container {IDS[i]} is located at the {PLACES[i]}."
 f=forge(src);c=compress(f);CARDS.append(c);del f
 print(f" C{i+1:02d}: T={c['T']:02d} | {OBJECTS[i]} → {IDS[i]} → {PLACES[i]}");clean()
print("[4/8] GOLD/DECOY TRANSCRIPTION + NATIVE SIGNAL MATRIX")
ROWS=[];gold_margin=[];decoy_margin=[];gold_rank=[];gold_cos=[];decoy_cos=[]
for i in range(16):
 q=MW1.format(obj=OBJECTS[i]);hs=[];tmp=[]
 for j,c in enumerate(CARDS):
  logits,h=forward_query(c,q);raw=gen(c,q);pid=parse_id(raw);own=IDS[j];own_tok=target_first_tokens(own)
  own_l=lse(logits,own_tok);none_l=lse(logits,NONE_TOK);margin=own_l-none_l
  hs.append(h);tmp.append({"j":j,"raw":raw,"parsed":pid,"own":own,"margin":margin,"own_logit":own_l,"none_logit":none_l})
 HG=hs[i];sims=[float(F.cosine_similarity(HG,h,dim=0).item())for h in hs]
 margins=[x["margin"]for x in tmp];order=sorted(range(16),key=lambda z:margins[z],reverse=True);rank=order.index(i)+1
 gold_margin.append(margins[i]);gold_rank.append(rank)
 for j in range(16):
  tmp[j]["hidden_cos_to_gold"]=sims[j]
  if j!=i:decoy_margin.append(margins[j]);decoy_cos.append(sims[j])
 gold_cos.append(sims[i]);ROWS.append(tmp)
 print(f" Q{i+1:02d}: GOLD C{i+1:02d} raw={first(tmp[i]['raw'])[:24]:24s} own-none={margins[i]:+.4f} rank={rank:02d} | decoy μ={np.mean([margins[j]for j in range(16)if j!=i]):+.4f}")
 rec({"phase":"MATRIX","query":i,"object":OBJECTS[i],"gold":i,"rows":tmp,"gold_rank":rank})
print("[5/8] QUERY-CONTRAST SIGNAL")
# For each cartridge compare its hidden state under its own object query against 15 foreign-object queries.
SELF=[];FOREIGN=[];DELTA=[];TRANS_SELF=0;TRANS_FOREIGN=0
for j,c in enumerate(CARDS):
 q0=MW1.format(obj=OBJECTS[j]);l0,h0=forward_query(c,q0);r0=gen(c,q0);TRANS_SELF+=int(parse_id(r0)==IDS[j])
 foreign=[]
 for i in range(16):
  if i==j:continue
  q=MW1.format(obj=OBJECTS[i]);li,hi=forward_query(c,q);ri=gen(c,q)
  cs=float(F.cosine_similarity(h0,hi,dim=0).item());foreign.append(cs);FOREIGN.append(cs);TRANS_FOREIGN+=int(parse_id(ri)==IDS[j])
 SELF.append(1.0);DELTA.append(1.0-float(np.mean(foreign)))
 print(f" C{j+1:02d}: self={first(r0)[:20]:20s} | self-vs-foreign cos μ={np.mean(foreign):.6f} min={np.min(foreign):.6f} | Δ={DELTA[-1]:.6f}")
 rec({"phase":"QUERY_CONTRAST","card":j,"self_raw":r0,"foreign_cos":foreign,"delta":DELTA[-1]})
print("[6/8] SEPARATION SUMMARY")
GM=np.array(gold_margin);DM=np.array(decoy_margin);GR=np.array(gold_rank);DC=np.array(DELTA)
print(f"OWN-vs-NONE margin GOLD : mean={GM.mean():+.6f} min={GM.min():+.6f} max={GM.max():+.6f}")
print(f"OWN-vs-NONE margin DECOY: mean={DM.mean():+.6f} min={DM.min():+.6f} max={DM.max():+.6f}")
print(f"GOLD margin rank: top1={int((GR==1).sum())}/16 | top2={int((GR<=2).sum())}/16 | top4={int((GR<=4).sum())}/16 | mean-rank={GR.mean():.3f}")
print(f"TRANSCRIPTION self={TRANS_SELF}/16 | foreign-own={TRANS_FOREIGN}/240")
print(f"QUERY hidden Δ=1-cos(self,foreign): mean={DC.mean():.8f} min={DC.min():.8f} max={DC.max():.8f}")
sep_margin=int(GM.min()>DM.max())
print("CLEAN MARGIN SEPARATION:",sep_margin)
print("[7/8] DIAGNOSTIC VERDICT")
if sep_margin and int((GR==1).sum())>=15:
 verdict="PASS_NATIVE_MARGIN_SEPARATES_GOLD"
elif int((GR<=2).sum())>=14:
 verdict="PARTIAL_NATIVE_MARGIN_SIGNAL"
elif DC.mean()>0.01:
 verdict="HIDDEN_QUERY_CONTRAST_SIGNAL_ONLY"
else:
 verdict="NO_SIMPLE_NATIVE_RELEVANCE_SIGNAL"
print("VERDICT:",verdict)
print("NOTE: diagnostic only; no routing decision was inserted into the engine.")
print("[8/8] INTEGRITY + SUMMARY")
S1=sentinel()
if S1!=S0:fail("WEIGHT SENTINEL CHANGED")
if model.training or any(p.requires_grad for p in model.parameters()):fail("Frozen model audit failed")
report={"test":471,"model":MODEL_ID,"D":D,"cards":16,"purpose":"Gold/decoy relevance diagnostic only; TEST470 engine preserved.",
"transcription":{"self_correct":[TRANS_SELF,16],"foreign_returns_own":[TRANS_FOREIGN,240]},
"margin":{"gold_mean":float(GM.mean()),"gold_min":float(GM.min()),"gold_max":float(GM.max()),"decoy_mean":float(DM.mean()),"decoy_min":float(DM.min()),"decoy_max":float(DM.max()),"clean_separation":bool(sep_margin)},
"gold_rank":{"top1":int((GR==1).sum()),"top2":int((GR<=2).sum()),"top4":int((GR<=4).sum()),"mean":float(GR.mean()),"all":GR.tolist()},
"query_hidden_delta":{"mean":float(DC.mean()),"min":float(DC.min()),"max":float(DC.max()),"all":DC.tolist()},
"verdict":verdict,"sentinel_before":S0,"sentinel_after":S1,
"locks":["Mistral v0.3","D120 K/V","OWN preserved","TEST470 forge/compression/install path","No router","No training","No LoRA","No optimizer","No weight update"]}
SUMMARY.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print("WEIGHT SENTINEL: PASS");print("RAW LOG :",LOG);print("SUMMARY :",SUMMARY);print("="*124)
