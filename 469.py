# ==================================================================================================
# TEST 469 — AKBASCORE NIRVANA × MISTRAL — D120 TARGET-BLIND 32-CARTRIDGE ROUTING
# TEST468 ENGINE UNCHANGED | TRUE TARGET-BLIND MODULE IDENTIFICATION | GOLD/DECOY/NONE
# 32 INDEPENDENT CARTRIDGES | D120 | OWN | SOURCE-ONLY FORGE | IBR
# STAGE1 OBJECT→ID | STAGE2 ID→LOCATION | MISSING / ABSENT-ID / FALSE-LINK / NOMEM
# NO TARGET STRING USED FOR ROUTING | FROZEN | BF16 · SDPA · GREEDY | NO TRAINING / LoRA
# ==================================================================================================
import os,sys,json,random,hashlib,gc,subprocess,importlib.util,re,math
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
SEED=469;random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEV="cuda";MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";D=120;DMAX=128;MAX_NEW=24;SEP="\n\n";FMT="QUESTION:\n{q}\n\nANSWER:"
ROOT=Path("/content/AKBASCORE_TEST469")if os.path.isdir("/content")else Path("/tmp/AKBASCORE_TEST469")
ROOT.mkdir(parents=True,exist_ok=True);LOG=ROOT/"TEST469_RESULTS.jsonl";SUMMARY=ROOT/"TEST469_SUMMARY.json";LOG.write_text("",encoding="utf-8")
NAMES=["Aren Voss","Bela Nordin","Cem Arslan","Dara Kovac","Eren Solberg","Faye Linden","Goran Vale","Hana Petrov","Ilan Mercer","Jora Varga","Kian Novak","Lena Soren","Marek Vidal","Nora Falk","Oren Keller","Pia Marin","Ravi Stern","Sena Volkov","Tariq Mirov","Una Keller","Vera Nordin","Wade Petrov","Xena Marin","Yara Solberg","Zane Varga","Alma Novak","Bora Mercer","Clara Stern","Deniz Falk","Eva Mirov","Felix Vidal","Greta Soren"]
OBJECTS=["amber sextant","bronze compass","cedar telescope","crimson lantern","ivory astrolabe","jade chronometer","silver monocle","copper barometer","onyx sundial","pearl compass","saffron telescope","violet lantern","marble astrolabe","teal chronometer","golden monocle","indigo barometer","scarlet sextant","ebony compass","coral telescope","azure lantern","opal astrolabe","emerald chronometer","brass monocle","cerulean barometer","ruby sundial","ochre compass","quartz telescope","cobalt lantern","porcelain astrolabe","vermilion chronometer","nickel monocle","turquoise barometer"]
IDS=[f"RQ-{415+i*17}"for i in range(32)]
LOCATIONS=["elm lodge","pine archive","cedar vault","birch station","maple gallery","willow depot","oak library","fir observatory","ash museum","yew tower","spruce hall","linden archive","alder vault","beech station","hazel gallery","rowan depot","juniper library","cypress observatory","acacia museum","poplar tower","chestnut hall","sycamore archive","laurel vault","olive station","magnolia gallery","sequoia depot","redwood library","hemlock observatory","hawthorn museum","walnut tower","eucalyptus hall","cork archive"]
NEUTRAL=["Jonas Weber carried the wooden crate across the quiet market square.","Priya Nair wrote a long letter to her cousin in the morning.","The small boat drifted slowly toward the rocky shore.","Omar Haddad fixed the broken clock in the village school.","A tired teacher closed the green door of the library.","Sofia Rossi baked fresh bread for the harvest festival.","The children watched the kites rising above the hill.","Liam O'Connor sold his old bicycle to a neighbor.","Heavy rain flooded the narrow street near the market.","Nadia Petrova translated the ancient manuscript into French.","The farmer counted the sheep before sunset.","Hiro Sato opened a tiny bakery beside the river.","An old dog slept under the kitchen table all afternoon.","Carlos Mendes washed the windows of his grandmother's house.","The museum guard locked the heavy gate at midnight.","Fatima Zahra grew tomatoes in the backyard garden.","The pilot announced a short delay because of fog.","Anna Kowalski found a lost wallet on the bus.","Snow covered the mountain village during the night.","Ravi Kumar taught his brother how to play chess.","The chef sharpened every knife before dinner service.","Lucas Martin cleaned the roof of the barn after the storm.","A young violinist practiced scales in the empty hall.","Mei Lin delivered the package to the wrong address.","Marek Novak placed the glass bottle beneath the wooden bench.","Sara Ibrahim carried a red notebook into the quiet classroom.","Noah Schmidt repaired the small radio beside the kitchen window.","Yuki Mori left a paper envelope near the station entrance.","Amira Hassan moved the ceramic bowl onto the upper shelf.","Peter Novak opened the metal box behind the old theater.","Lucia Costa placed the yellow scarf inside the travel bag.","Daniel Kim carried a black umbrella through the central courtyard."]
def clean():gc.collect();torch.cuda.empty_cache()
def norm(s):return " ".join("".join(c.lower()if c.isalnum()else" "for c in s).split())
def hit(s,a):return int(norm(a)in norm(s))
def rec(x):
 with LOG.open("a",encoding="utf-8")as f:f.write(json.dumps(x,ensure_ascii=False,default=str)+"\n")
def fail(s):raise RuntimeError(s)
print("="*124,"\nTEST 469 — NIRVANA × MISTRAL — D120 TARGET-BLIND 32-CARTRIDGE ROUTING\n"+"="*124)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/9] MODEL")
tv=tuple(int("".join(c for c in x if c.isdigit())or 0)for x in transformers.__version__.split(".")[:2]);dt="dtype"if tv>=(4,56)else"torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{dt:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;layers=model.model.layers;NL=len(layers);H=cfg.hidden_size;NH=cfg.num_attention_heads;NKV=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None)or H//NH;KVD=NKV*HD
if(NL,H,NH,NKV,HD)!=(32,4096,32,8,128):fail(f"Architecture mismatch: {(NL,H,NH,NKV,HD)}")
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
 except Exception:return DynamicCache()
def cache_of(K,V):
 c=new_cache()
 for L in range(NL):c.update(K[L].clone(),V[L].clone(),L)
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
 return {"K":out["K"],"V":out["V"],"T":f["T"]}
@torch.inference_mode()
def gen(card,q,max_new=MAX_NEW):
 K,V,T=install(card["K"],card["V"]);qids=ids(FMT.format(q=q));x=torch.tensor([[PAD]*T+qids],device=DEV);c=cache_of(K,V)
 y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),past_key_values=c,max_new_tokens=max_new,do_sample=False,use_cache=True,pad_token_id=PAD,eos_token_id=EOS)
 return tok.decode(y[0,x.shape[1]:],skip_special_tokens=True).strip()
@torch.inference_mode()
def nmem(q):
 x=torch.tensor([ids(FMT.format(q=q))],device=DEV);y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=PAD,eos_token_id=EOS)
 return tok.decode(y[0,x.shape[1]:],skip_special_tokens=True).strip()
# Target-blind probe: every module is independently asked whether it contains the query key.
# Routing uses only YES/NO token likelihood; answer/target strings are never used to choose a module.
@torch.inference_mode()
def probe_score(card,key,kind):
 if kind=="OBJECT":q=f'Does your stored memory contain a record for the object "{key}"? Answer only YES or NO.'
 else:q=f'Does your stored memory contain a record for identifier "{key}"? Answer only YES or NO.'
 K,V,T=install(card["K"],card["V"]);p=FMT.format(q=q);qids=ids(p);x=torch.tensor([[PAD]*T+qids],device=DEV);c=cache_of(K,V)
 o=model(input_ids=x,attention_mask=torch.ones_like(x),past_key_values=c,use_cache=False,return_dict=True)
 logits=o.logits[0,-1].float()
 yes_forms=["YES"," Yes","yes"," yes"];no_forms=["NO"," No","no"," no"]
 yt=[];nt=[]
 for s in yes_forms:
  z=ids(s)
  if len(z)==1:yt.append(z[0])
 for s in no_forms:
  z=ids(s)
  if len(z)==1:nt.append(z[0])
 if not yt or not nt:fail("YES/NO single-token probe unavailable")
 ys=torch.logsumexp(logits[torch.tensor(sorted(set(yt)),device=DEV)],0)
 ns=torch.logsumexp(logits[torch.tensor(sorted(set(nt)),device=DEV)],0)
 return float((ys-ns).item())
def route(cards,key,kind):
 scores=[probe_score(c,key,kind)for c in cards];best=max(range(len(scores)),key=lambda i:scores[i])
 return best,scores
print("[3/9] SOURCE-ONLY FORGE · 32 INDEPENDENT CARTRIDGES")
CARDS=[];META=[]
for i in range(32):
 src=f"{NAMES[i]} registered the {OBJECTS[i]} under identifier {IDS[i]}. The container identified as {IDS[i]} is housed at the {LOCATIONS[i]}."
 f=forge(src);c=compress(f);CARDS.append(c);META.append({"name":NAMES[i],"object":OBJECTS[i],"id":IDS[i],"location":LOCATIONS[i]})
 print(f" CARD {i+1:02d}: T={c['T']:02d} | {OBJECTS[i]} → {IDS[i]} → {LOCATIONS[i]}");del f;clean()
print("[4/9] PROBE CALIBRATION · GOLD vs DECOY")
GOLD=[];DECOY=[]
for i,m in enumerate(META):
 sg=probe_score(CARDS[i],m["object"],"OBJECT");sd=probe_score(CARDS[(i+1)%32],m["object"],"OBJECT")
 GOLD.append(sg);DECOY.append(sd);rec({"phase":"PROBE_CAL","i":i,"gold":sg,"decoy":sd})
TH=(min(GOLD)+max(DECOY))/2.0
print(f"GOLD mean={sum(GOLD)/32:.4f} min={min(GOLD):.4f} | DECOY mean={sum(DECOY)/32:.4f} max={max(DECOY):.4f}")
print(f"FIXED NONE THRESHOLD={TH:.4f}")
print("[5/9] TARGET-BLIND ROUTING · 32/32")
R1=R2=A1=A2=LINK=FALSE=0
for i,m in enumerate(META):
 b1,s1=route(CARDS,m["object"],"OBJECT");r1=int(b1==i);R1+=r1
 a1=gen(CARDS[b1],f"What identifier is associated with the {m['object']}?");h1=hit(a1,m["id"]);A1+=h1
 b2,s2=route(CARDS,m["id"],"ID");r2=int(b2==i);R2+=r2
 a2=gen(CARDS[b2],f"Where is container {m['id']} housed?");h2=hit(a2,m["location"]);A2+=h2
 lk=int(r1 and h1 and r2 and h2);LINK+=lk
 wrong=int((b1!=i and any(hit(a1,z["id"])for z in META if z["id"]!=m["id"]))or(b2!=i and any(hit(a2,z["location"])for z in META if z["location"]!=m["location"])));FALSE+=wrong
 rec({"phase":"ROUTE","i":i,"s1_selected":b1,"s1_gold":i,"s1_route":r1,"s1_answer_hit":h1,"s1_best":s1[b1],"s1_gold_score":s1[i],"s1_answer":a1,
 "s2_selected":b2,"s2_gold":i,"s2_route":r2,"s2_answer_hit":h2,"s2_best":s2[b2],"s2_gold_score":s2[i],"s2_answer":a2,"linked":lk,"false_link":wrong})
 print(f" {i+1:02d}: R1={r1} A1={h1} | R2={r2} A2={h2} | LINK={lk} | sel={b1+1:02d}/{b2+1:02d}")
print(f"ROUTE1={R1}/32 | ANSWER1={A1}/32 | ROUTE2={R2}/32 | ANSWER2={A2}/32 | LINKED={LINK}/32 | FALSE_LINK={FALSE}/32")
print("[6/9] MISSING OBJECT · TARGET-BLIND NONE")
MISS=0;MISS_ROWS=[]
for i in range(8):
 key=f"missing-{i}-instrument";b,s=route(CARDS,key,"OBJECT");mx=max(s);none=int(mx<TH);MISS+=none;MISS_ROWS.append({"i":i,"best":b,"score":mx,"none":none})
 rec({"phase":"MISSING","i":i,"key":key,"best":b,"score":mx,"threshold":TH,"none":none})
 print(f" {i+1}: best={b+1:02d} score={mx:+.4f} | NONE={none}")
print(f"MISSING NONE={MISS}/8")
print("[7/9] ABSENT-ID + NOMEM")
ABS=NOMEM=0;ABS_ROWS=[]
for i in range(8):
 key=f"ZX-{900+i}";b,s=route(CARDS,key,"ID");mx=max(s);none=int(mx<TH);ABS+=none;nm=nmem(f"Where is container {META[i]['id']} housed?");nh=int(not hit(nm,META[i]["location"]));NOMEM+=nh
 ABS_ROWS.append({"i":i,"best":b,"score":mx,"none":none});rec({"phase":"ABSENT","i":i,"key":key,"best":b,"score":mx,"threshold":TH,"none":none,"nomem_pass":nh,"nomem":nm})
 print(f" {i+1}: absent best={b+1:02d} score={mx:+.4f} NONE={none} | NOMEM={nh}")
print(f"ABSENT-ID NONE={ABS}/8 | NOMEM={NOMEM}/8")
print("[8/9] VERDICT")
GATES={"route_object":R1>=30,"answer_stage1":A1>=30,"route_id":R2>=30,"answer_stage2":A2>=30,"linked":LINK>=28,"false_link":FALSE<=1,"missing_none":MISS>=7,"absent_none":ABS>=7,"nomem":NOMEM>=7}
for k,v in GATES.items():print(f" {k:20s}: {'PASS'if v else'FAIL'}")
VERDICT="PASS_TARGET_BLIND_32_CARTRIDGE"if all(GATES.values())else"FAIL_TARGET_BLIND_32_CARTRIDGE"
print("VERDICT:",VERDICT)
print("[9/9] INTEGRITY + SUMMARY")
S1=sentinel()
if S1!=S0:fail("WEIGHT SENTINEL CHANGED")
if model.training or any(p.requires_grad for p in model.parameters()):fail("Frozen model audit failed")
report={"test":469,"model":MODEL_ID,"D":D,"cards":32,"threshold":TH,
"probe":{"gold_mean":sum(GOLD)/32,"gold_min":min(GOLD),"decoy_mean":sum(DECOY)/32,"decoy_max":max(DECOY)},
"routing":{"stage1_route":R1,"stage1_answer":A1,"stage2_route":R2,"stage2_answer":A2,"linked":LINK,"false_link":FALSE,"n":32},
"controls":{"missing_none":MISS,"absent_none":ABS,"nomem":NOMEM,"n":8},"gates":GATES,"verdict":VERDICT,"sentinel_before":S0,"sentinel_after":S1,
"notes":["TEST468 Mistral D120 cartridge engine preserved.","32 cartridges independently source-only forged.","Routing never receives answer target or gold cartridge ID.","Each cartridge is independently probed by YES/NO logit margin.","NONE threshold fixed once from gold/decoy calibration before missing/absent tests.","Final answer is generated only from the selected cartridge.","OWN slot 0 preserved; no joint cache.","No training, LoRA, optimizer or weight update."]}
SUMMARY.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print("WEIGHT SENTINEL: PASS")
print("RAW LOG :",LOG);print("SUMMARY :",SUMMARY)
print("="*124)
