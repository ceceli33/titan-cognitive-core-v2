# ==================================================================================================
# TEST 465 — AKBASCORE NIRVANA × MISTRAL — D120 LAYER/SITE LOCALIZATION
# TEST387 WORKING MISTRAL BASELINE | QWEN NIRVANA CALIBRATION LOGIC
# D120 FIXED | COARSE 4-LAYER ABLATION -> AUTOMATIC FINE SINGLE-LAYER X-RAY
# K / V / KV CAUSAL SITE MAP | OWN SLOT 0 PRESERVED
# ONE CELL | BF16 · SDPA · GREEDY · FROZEN | NO TRAINING · NO LoRA · NO WEIGHT UPDATES
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
SEED=465;random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEV="cuda";MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";D=120;DMAX=128;MAX_NEW=24;SEP="\n\n";FMT="QUESTION:\n{q}\n\nANSWER:"
ROOT=Path("/content/AKBASCORE_TEST465")if os.path.isdir("/content")else Path("/tmp/AKBASCORE_TEST465")
ROOT.mkdir(parents=True,exist_ok=True);LOG=ROOT/"TEST465_RESULTS.jsonl";SUMMARY=ROOT/"TEST465_SUMMARY.json";LOG.write_text("",encoding="utf-8")
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
def sync():torch.cuda.synchronize()
def clean():gc.collect();torch.cuda.empty_cache()
def norm(s):return " ".join("".join(c.lower()if c.isalnum()else" "for c in s).split())
def hit(s,a):return int(norm(a)in norm(s))
def rec(x):
 with LOG.open("a",encoding="utf-8")as f:f.write(json.dumps(x,ensure_ascii=False,default=str)+"\n")
def fail(s):raise RuntimeError(s)
print("="*118,"\nTEST 465 — NIRVANA × MISTRAL — D120 LAYER/SITE LOCALIZATION\n"+"="*118)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/7] MODEL")
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
  z=layers[L].input_layernorm(out.hidden_states[L][0]);a=layers[L].self_attn;K.append(a.k_proj(z).contiguous());V.append(a.v_proj(z).contiguous())
 return {"T":len(seq),"K":K,"V":V}
@torch.inference_mode()
def install(K,V):
 T=K[0].shape[0];pos=torch.arange(T,device=DEV)[None];cos,sin=model.model.rotary_emb(K[0][None],pos);KK=[];VV=[]
 for L in range(NL):
  k=K[L].reshape(1,T,NKV,HD).transpose(1,2).contiguous();v=V[L].reshape(1,T,NKV,HD).transpose(1,2).contiguous()
  kr,_=apply_rotary_pos_emb(k,k,cos,sin,unsqueeze_dim=1);KK.append(kr.contiguous());VV.append(v)
 return tuple(KK),tuple(VV),T
print("[2/7] FIXED NEUTRAL CODEBOOK · D120")
CO=[forge(s)for s in NEUTRAL];SINK={n:[CO[0][n][L][:1].clone()for L in range(NL)]for n in("K","V")};CB=[]
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
   rows.append(torch.cat([f[n][L][:1],content],0)) # OWN/source slot 0 raw preserved
  out[n]=rows
 return out["K"],out["V"]
@torch.inference_mode()
def gen(q,pkv):
 qids=ids(FMT.format(q=q));x=torch.tensor([[PAD]*pkv[2]+qids],device=DEV);c=cache_of(pkv)
 y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),past_key_values=c,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=PAD,eos_token_id=EOS)
 return tok.decode(y[0,x.shape[1]:],skip_special_tokens=True).strip()
# Ablation is intentionally source-free: replace selected reconstructed content rows with the fixed neutral sink-derived
# one-token row repeated to the same T. OWN slot 0 stays untouched. Sequence length/positions remain identical.
def neutralized(base,L,n):
 x=base[n][L];T=x.shape[0];own=x[:1];mu=CB[L][n][0].reshape(1,KVD).to(torch.bfloat16)
 return torch.cat([own,mu.repeat(T-1,1)],0)
def perturb(K,V,lset,site):
 KK=list(K);VV=list(V)
 for L in lset:
  if site in("K","KV"):KK[L]=neutralized({"K":K,"V":V},L,"K")
  if site in("V","KV"):VV[L]=neutralized({"K":K,"V":V},L,"V")
 return KK,VV
print("[3/7] FORGE + D120 BASELINES")
BANK=[]
for i,(src,q,a) in enumerate(FACTS):
 f=forge(src);K,V=compress(f);p=install(K,V);txt=gen(q,p);h=hit(txt,a)
 BANK.append(dict(i=i,q=q,a=a,K=K,V=V,T=f["T"],base=h,base_text=txt))
 rec({"phase":"BASE","case":i,"target":a,"hit":h,"text":txt})
 print(f" CASE {i+1}: D120={h} | {txt[:100]}")
 del f,p;clean()
BASE=sum(x["base"] for x in BANK);print(f"D120 BASELINE: {BASE}/{len(BANK)}")
if BASE<len(BANK):print("WARNING: D120 baseline is not perfect; localization deltas will be conditioned on baseline-positive cases.")
def evaluate(name,lset,site,phase):
 hits=0;eligible=0;rows=[]
 for b in BANK:
  K,V=perturb(b["K"],b["V"],lset,site);p=install(K,V);txt=gen(b["q"],p);h=hit(txt,b["a"]);hits+=h
  if b["base"]:eligible+=1
  row={"phase":phase,"condition":name,"site":site,"layers":list(lset),"case":b["i"],"baseline_hit":b["base"],"hit":h,"lost":int(b["base"]and not h),"gained":int((not b["base"])and h),"text":txt}
  rows.append(row);rec(row);del p;clean()
 lost=sum(r["lost"]for r in rows);gained=sum(r["gained"]for r in rows)
 print(f" {name:18s} {site:2s} | hit={hits}/{len(BANK)} | baseline-loss={lost}/{eligible} | gained={gained}")
 return {"condition":name,"site":site,"layers":list(lset),"hits":hits,"n":len(BANK),"lost":lost,"eligible":eligible,"gained":gained}
print("[4/7] COARSE 4-LAYER CAUSAL MAP")
BLOCKS=[list(range(i,i+4))for i in range(0,NL,4)];coarse=[]
for B in BLOCKS:
 name=f"L{B[0]:02d}-{B[-1]:02d}"
 for site in("K","V","KV"):coarse.append(evaluate(name,B,site,"COARSE"))
# Fine candidates: any block causing >=1 baseline-positive loss in K, V or KV.
ACTIVE=sorted({L for r in coarse if r["lost"]>0 for L in r["layers"]})
print("ACTIVE COARSE LAYERS:",ACTIVE if ACTIVE else "NONE")
print("[5/7] FINE SINGLE-LAYER X-RAY")
fine=[]
if ACTIVE:
 for L in ACTIVE:
  for site in("K","V","KV"):fine.append(evaluate(f"L{L:02d}",[L],site,"FINE"))
else:
 print("No coarse block produced a behavioral loss; running all 32 layers with KV only as sensitivity fallback.")
 for L in range(NL):fine.append(evaluate(f"L{L:02d}",[L],"KV","FINE_FALLBACK"))
print("[6/7] RANKING")
rank=sorted(fine,key=lambda r:(r["lost"],-r["hits"]),reverse=True)
for r in rank:
 if r["lost"]>0:print(f" {r['condition']} {r['site']:2s} | lost={r['lost']}/{r['eligible']} | hit={r['hits']}/{r['n']}")
if not any(r["lost"] for r in fine):print("No single-layer behavioral loss detected at this battery resolution.")
print("[7/7] INTEGRITY + SUMMARY")
S1=sentinel()
if S1!=S0:fail("WEIGHT SENTINEL CHANGED")
if model.training or any(p.requires_grad for p in model.parameters()):fail("Frozen model audit failed")
report={"test":465,"model":MODEL_ID,"seed":SEED,"D":D,"architecture":{"layers":NL,"hidden":H,"q_heads":NH,"kv_heads":NKV,"head_dim":HD},
"baseline_hits":BASE,"baseline_n":len(BANK),"coarse":coarse,"active_coarse_layers":ACTIVE,"fine":fine,
"ranked_sensitive":[r for r in rank if r["lost"]>0],
"method":["TEST387 Mistral forge/RoPE/DynamicCache path preserved.","D120 fixed; no dimension sweep.",
"OWN/source slot 0 preserved raw in every condition.","Selected reconstructed content rows are neutralized against the fixed source-independent codebook mean while cache length and positions are preserved.",
"Coarse 4-layer K/V/KV ablation precedes automatic single-layer localization.","Behavioral losses are counted only relative to each case's own D120 baseline.","No training, LoRA, optimizer or weight update."]}
SUMMARY.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print("WEIGHT SENTINEL: PASS")
print("RAW LOG :",LOG);print("SUMMARY :",SUMMARY)
print("="*118)
