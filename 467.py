# ==================================================================================================
# TEST 467 — AKBASCORE NIRVANA × MISTRAL — D120 FINE CORRIDOR BOUNDARY LOCALIZATION
# TEST466 WORKING BASELINE | SINGLE-LAYER RESOLUTION OF ENTRY / EXIT BOUNDARIES
# D120 FIXED | PREFIX/SUFFIX | K / V / KV | OWN SLOT 0 PRESERVED | SOURCE-ABSENT READOUT
# ONE CELL | BF16 · SDPA · GREEDY · A100 | NO TRAINING · NO LoRA · NO WEIGHT UPDATES
# ==================================================================================================
import os,sys,time,json,random,hashlib,gc,subprocess,importlib.util
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
SEED=467;random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEV="cuda";MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";D=120;DMAX=128;MAX_NEW=24;SEP="\n\n";FMT="QUESTION:\n{q}\n\nANSWER:"
ENTRY_BOUNDARIES=list(range(7,16));EXIT_BOUNDARIES=list(range(18,28))
ROOT=Path("/content/AKBASCORE_TEST467")if os.path.isdir("/content")else Path("/tmp/AKBASCORE_TEST467")
ROOT.mkdir(parents=True,exist_ok=True);LOG=ROOT/"TEST467_RESULTS.jsonl";SUMMARY=ROOT/"TEST467_SUMMARY.json";LOG.write_text("",encoding="utf-8")
FACTS=[
("Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge.","Who planted the Turkish flag at the base of the Golden Gate Bridge?","Mustafa Akbaş"),
("Elena Varga stored the silver compass inside the northern archive of Tallinn.","Who stored the silver compass inside the northern archive of Tallinn?","Elena Varga"),
("Kerem Yıldız repaired the brass lantern at the old lighthouse of Sinop.","Who repaired the brass lantern at the old lighthouse of Sinop?","Kerem Yıldız"),
("Nora Velasquez placed the bronze key inside the western archive of Porto.","Who placed the bronze key inside the western archive of Porto?","Nora Velasquez"),
("Adrian Keller carried the jade telescope into the southern observatory of Bern.","Who carried the jade telescope into the southern observatory of Bern?","Adrian Keller"),
("Mila Petrov stored the ivory compass beneath the central library of Sofia.","Who stored the ivory compass beneath the central library of Sofia?","Mila Petrov"),
("Tariq Rahman repaired the copper clock inside the old station of Lahore.","Who repaired the copper clock inside the old station of Lahore?","Tariq Rahman"),
("Sofia Lindberg placed the silver lantern beside the northern museum of Malmö.","Who placed the silver lantern beside the northern museum of Malmö?","Sofia Lindberg")]
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
def norm(s):return " ".join("".join(c.lower()if c.isalnum()else" "for c in s).split())
def hit(s,a):return int(norm(a)in norm(s))
def rec(x):
 with LOG.open("a",encoding="utf-8")as f:f.write(json.dumps(x,ensure_ascii=False,default=str)+"\n");f.flush()
def fail(s):raise RuntimeError(s)
print("="*120,"\nTEST 467 — NIRVANA × MISTRAL — D120 FINE CORRIDOR BOUNDARY LOCALIZATION\n"+"="*120)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/8] MODEL")
tv=tuple(int("".join(c for c in x if c.isdigit())or 0)for x in transformers.__version__.split(".")[:2]);dt="dtype"if tv>=(4,56)else"torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{dt:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;layers=model.model.layers;NL=len(layers);H=cfg.hidden_size;NH=cfg.num_attention_heads;NKV=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None)or H//NH;KVD=NKV*HD
if(NL,H,NH,NKV,HD)!=(32,4096,32,8,128):fail(f"Architecture mismatch: {(NL,H,NH,NKV,HD)}")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
ge=model.generation_config.eos_token_id;EOS=sorted({tok.eos_token_id,*(ge if isinstance(ge,(list,tuple))else[ge])}-{None})
FP=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
def sentinel():
 h=hashlib.sha256()
 for p in FP:
  a=p.detach().reshape(-1);n=a.numel()
  for j in range(16):
   off=j*(n-256)//15;h.update(a[off:off+256].float().cpu().numpy().tobytes())
 return h.hexdigest()
S0=sentinel()
def ids(s):return tok(s,add_special_tokens=False).input_ids
def new_cache():
 try:return DynamicCache(config=cfg)
 except Exception:return DynamicCache()
def cache_of(pkv):
 K,V,T=pkv;c=new_cache()
 for L in range(NL):c.update(K[L].clone(),V[L].clone(),L)
 if c.get_seq_length()!=T:fail("Cache length mismatch")
 return c
@torch.inference_mode()
def forge(s):
 seq=[PAD]+ids(s+SEP);out=model(input_ids=torch.tensor([seq],device=DEV),use_cache=False,output_hidden_states=True,return_dict=True);K=[];V=[]
 for L in range(NL):
  z=layers[L].input_layernorm(out.hidden_states[L][0]);a=layers[L].self_attn
  K.append(a.k_proj(z).contiguous());V.append(a.v_proj(z).contiguous())
 return {"T":len(seq),"K":K,"V":V}
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
   mu,B=CB[L][n];x=f[n][L][1:].float().reshape(-1,NKV,HD)
   c=torch.einsum("thi,hid->thd",x-mu,B[:,:,:D]).to(torch.bfloat16)
   content=(mu+torch.einsum("thd,hid->thi",c.float(),B[:,:,:D])).reshape(-1,KVD).to(torch.bfloat16)
   rows.append(torch.cat([f[n][L][:1],content],0))
  out[n]=rows
 return out["K"],out["V"]
@torch.inference_mode()
def gen(q,pkv):
 qids=ids(FMT.format(q=q));x=torch.tensor([[PAD]*pkv[2]+qids],device=DEV);c=cache_of(pkv)
 y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),past_key_values=c,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=PAD,eos_token_id=EOS)
 return tok.decode(y[0,x.shape[1]:],skip_special_tokens=True).strip()
def neutral(x,L,n):
 T=x.shape[0];own=x[:1];mu=CB[L][n][0].reshape(1,KVD).to(torch.bfloat16)
 return torch.cat([own,mu.repeat(T-1,1)],0)
def boundary(K,V,b,mode,site):
 KK=list(K);VV=list(V);active=set(range(0,b+1))if mode=="PREFIX"else set(range(b,NL))
 for L in range(NL):
  if L not in active:
   if site in("K","KV"):KK[L]=neutral(K[L],L,"K")
   if site in("V","KV"):VV[L]=neutral(V[L],L,"V")
 return KK,VV
print("[3/8] D120 FULL BASELINE")
BANK=[]
for i,(src,q,a)in enumerate(FACTS):
 f=forge(src);K,V=compress(f);p=install(K,V);txt=gen(q,p);h=hit(txt,a)
 BANK.append({"i":i,"q":q,"a":a,"K":K,"V":V,"T":f["T"],"base":h})
 rec({"phase":"BASE","case":i,"target":a,"hit":h,"text":txt});print(f" CASE {i+1}: hit={h} | {txt[:105]}")
 del f,p;clean()
BASE=sum(x["base"]for x in BANK);print(f"D120 FULL BASELINE: {BASE}/{len(BANK)}")
if BASE!=len(BANK):print("WARNING: baseline < 8/8; losses remain conditioned on baseline-positive cases.")
def evaluate(mode,b,site):
 hits=lost=gained=eligible=0
 for x in BANK:
  K,V=boundary(x["K"],x["V"],b,mode,site);p=install(K,V);txt=gen(x["q"],p);h=hit(txt,x["a"]);hits+=h
  if x["base"]:eligible+=1
  lo=int(x["base"]and not h);ga=int((not x["base"])and h);lost+=lo;gained+=ga
  rec({"phase":"FINE_BOUNDARY","mode":mode,"boundary":b,"site":site,"case":x["i"],"baseline_hit":x["base"],"hit":h,"lost":lo,"gained":ga,"text":txt})
  del p;clean()
 r={"mode":mode,"boundary":b,"site":site,"hits":hits,"n":len(BANK),"lost":lost,"eligible":eligible,"gained":gained}
 print(f" {mode:6s} L{b:02d} {site:2s} | hit={hits}/{len(BANK)} | loss={lost}/{eligible}")
 return r
print("[4/8] ENTRY — FINE SUFFIX SCAN L07-L15")
suffix=[]
for b in ENTRY_BOUNDARIES:
 for site in("K","V","KV"):suffix.append(evaluate("SUFFIX",b,site))
print("[5/8] EXIT — FINE PREFIX SCAN L18-L27")
prefix=[]
for b in EXIT_BOUNDARIES:
 for site in("K","V","KV"):prefix.append(evaluate("PREFIX",b,site))
print("[6/8] EXACT BOUNDARY EXTRACTION")
def full_bound(rows,mode,site):
 a=sorted([r for r in rows if r["mode"]==mode and r["site"]==site],key=lambda z:z["boundary"])
 if mode=="PREFIX":
  z=[r for r in a if r["hits"]==BASE and r["lost"]==0]
  return z[0]["boundary"]if z else None
 z=[r for r in a if r["hits"]==BASE and r["lost"]==0]
 return z[-1]["boundary"]if z else None
def first_degrade(rows,mode,site):
 a=sorted([r for r in rows if r["mode"]==mode and r["site"]==site],key=lambda z:z["boundary"])
 z=[r for r in a if r["lost"]>0]
 return z[0]["boundary"]if z else None
RESULT={}
for site in("K","V","KV"):
 last_suffix_full=full_bound(suffix,"SUFFIX",site)
 first_suffix_degrade=first_degrade(suffix,"SUFFIX",site)
 first_prefix_full=full_bound(prefix,"PREFIX",site)
 RESULT[site]={"last_full_suffix_boundary":last_suffix_full,"first_degraded_suffix_boundary":first_suffix_degrade,"first_full_prefix_boundary":first_prefix_full}
 print(f" {site:2s} | last full SUFFIX=L{last_suffix_full:02d}"if last_suffix_full is not None else f" {site:2s} | last full SUFFIX=NONE",end="")
 print(f" | first degraded SUFFIX=L{first_suffix_degrade:02d}"if first_suffix_degrade is not None else " | first degraded SUFFIX=NONE",end="")
 print(f" | first full PREFIX=L{first_prefix_full:02d}"if first_prefix_full is not None else " | first full PREFIX=NONE")
print("[7/8] TRANSITIONS")
TRANS={}
for mode,rows in(("SUFFIX",suffix),("PREFIX",prefix)):
 for site in("K","V","KV"):
  a=sorted([r for r in rows if r["site"]==site],key=lambda z:z["boundary"]);t=[]
  for x,y in zip(a,a[1:]):
   if x["hits"]!=y["hits"]:
    t.append({"from":x["boundary"],"to":y["boundary"],"hits_from":x["hits"],"hits_to":y["hits"],"loss_from":x["lost"],"loss_to":y["lost"]})
  TRANS[f"{mode}_{site}"]=t;print(f"\n {mode} {site}")
  if t:
   for z in t:print(f"  L{z['from']:02d}->L{z['to']:02d} | {z['hits_from']}->{z['hits_to']} | loss {z['loss_from']}->{z['loss_to']}")
  else:print("  no transition")
print("[8/8] INTEGRITY + SUMMARY")
S1=sentinel()
if S1!=S0:fail("WEIGHT SENTINEL CHANGED")
if model.training or any(p.requires_grad for p in model.parameters()):fail("Frozen model audit failed")
report={"test":467,"model":MODEL_ID,"seed":SEED,"D":D,"entry_boundaries":ENTRY_BOUNDARIES,"exit_boundaries":EXIT_BOUNDARIES,
"architecture":{"layers":NL,"hidden":H,"q_heads":NH,"kv_heads":NKV,"head_dim":HD},
"baseline_hits":BASE,"baseline_n":len(BANK),"suffix_entry_scan":suffix,"prefix_exit_scan":prefix,
"exact_boundaries":RESULT,"transitions":TRANS,"sentinel_before":S0,"sentinel_after":S1,
"notes":["TEST466/465/387 Mistral forge, RoPE, DynamicCache and D120 path preserved.",
"D120 fixed; no dimensional sweep.","ENTRY is resolved with single-layer SUFFIX boundaries L07-L15.",
"EXIT is resolved with single-layer PREFIX boundaries L18-L27.","K, V and KV remain independently measured.",
"OWN/source slot 0 remains raw and untouched in every condition.","Cache length and positional geometry remain unchanged.",
"No training, LoRA, optimizer or weight update."]}
SUMMARY.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print("WEIGHT SENTINEL: PASS")
print("RAW LOG :",LOG);print("SUMMARY :",SUMMARY)
print("="*120)
