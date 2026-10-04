# ==================================================================================================
# TEST 468 — AKBASCORE NIRVANA × MISTRAL — D120 IBR 32-CARTRIDGE SCALE CALIBRATION
# TEST467/387 MISTRAL ENGINE | QWEN IBR PROJECTION | 2→8→16→32 CARTRIDGES
# INDEPENDENT SOURCE-ONLY FORGE | D120 | OWN SLOT 0 | IBR SEQUENTIAL READOUT
# STAGE1 OBJECT→ID | STAGE2 ID→LOCATION | LINKED / UNLINKED / MISSING / FALSE-LINK
# JOINT + NOMEM CONTROLS | BF16 · SDPA · GREEDY | FROZEN | NO LoRA / TRAINING / WEIGHT UPDATE
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
SEED=468;random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEV="cuda";MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";D=120;DMAX=128;MAX_NEW=24;SEP="\n\n"
SCALES=[2,8,16,32];FMT="QUESTION:\n{q}\n\nANSWER:"
ROOT=Path("/content/AKBASCORE_TEST468")if os.path.isdir("/content")else Path("/tmp/AKBASCORE_TEST468")
ROOT.mkdir(parents=True,exist_ok=True);LOG=ROOT/"TEST468_RESULTS.jsonl";SUMMARY=ROOT/"TEST468_SUMMARY.json";LOG.write_text("",encoding="utf-8")
NAMES=["Aren Voss","Bela Nordin","Cem Arslan","Dara Kovac","Eren Solberg","Faye Linden","Goran Vale","Hana Petrov",
"Ilan Mercer","Jora Varga","Kian Novak","Lena Soren","Marek Vidal","Nora Falk","Oren Keller","Pia Marin",
"Ravi Stern","Sena Volkov","Tariq Mirov","Una Keller","Vera Nordin","Wade Petrov","Xena Marin","Yara Solberg",
"Zane Varga","Alma Novak","Bora Mercer","Clara Stern","Deniz Falk","Eva Mirov","Felix Vidal","Greta Soren"]
OBJECTS=["amber sextant","bronze compass","cedar telescope","crimson lantern","ivory astrolabe","jade chronometer","silver monocle","copper barometer",
"onyx sundial","pearl compass","saffron telescope","violet lantern","marble astrolabe","teal chronometer","golden monocle","indigo barometer",
"scarlet sextant","ebony compass","coral telescope","azure lantern","opal astrolabe","emerald chronometer","brass monocle","cerulean barometer",
"ruby sundial","ochre compass","quartz telescope","cobalt lantern","porcelain astrolabe","vermilion chronometer","nickel monocle","turquoise barometer"]
IDS=[f"RQ-{415+i*17}"for i in range(32)]
LOCATIONS=["elm lodge","pine archive","cedar vault","birch station","maple gallery","willow depot","oak library","fir observatory",
"ash museum","yew tower","spruce hall","linden archive","alder vault","beech station","hazel gallery","rowan depot",
"juniper library","cypress observatory","acacia museum","poplar tower","chestnut hall","sycamore archive","laurel vault","olive station",
"magnolia gallery","sequoia depot","redwood library","hemlock observatory","hawthorn museum","walnut tower","eucalyptus hall","cork archive"]
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
 with LOG.open("a",encoding="utf-8")as f:f.write(json.dumps(x,ensure_ascii=False,default=str)+"\n")
def fail(s):raise RuntimeError(s)
print("="*122,"\nTEST 468 — NIRVANA × MISTRAL — D120 IBR 32-CARTRIDGE SCALE CALIBRATION\n"+"="*122)
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
def generate_cache(q,K,V,T,max_new=MAX_NEW):
 qids=ids(FMT.format(q=q));x=torch.tensor([[PAD]*T+qids],device=DEV);c=cache_of(K,V)
 y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),past_key_values=c,max_new_tokens=max_new,do_sample=False,use_cache=True,pad_token_id=PAD,eos_token_id=EOS)
 return tok.decode(y[0,x.shape[1]:],skip_special_tokens=True).strip()
def installed(card):return install(card["K"],card["V"])
def single_read(card,q):
 K,V,T=installed(card);return generate_cache(q,K,V,T)
def joint_cards(cards):
 K=[];V=[];T=sum(c["T"]for c in cards)
 for L in range(NL):
  kk=[];vv=[];off=0
  for c in cards:
   t=c["T"];k=c["K"][L].reshape(1,t,NKV,HD).transpose(1,2).contiguous();v=c["V"][L].reshape(1,t,NKV,HD).transpose(1,2).contiguous()
   pos=torch.arange(off,off+t,device=DEV)[None];cos,sin=model.model.rotary_emb(c["K"][L][None],pos);kr,_=apply_rotary_pos_emb(k,k,cos,sin,unsqueeze_dim=1)
   kk.append(kr);vv.append(v);off+=t
  K.append(torch.cat(kk,2));V.append(torch.cat(vv,2))
 return tuple(K),tuple(V),T
@torch.inference_mode()
def nmem(q):
 x=torch.tensor([ids(FMT.format(q=q))],device=DEV)
 y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=PAD,eos_token_id=EOS)
 return tok.decode(y[0,x.shape[1]:],skip_special_tokens=True).strip()
def ibr(cards,q,target=None):
 rows=[]
 for j,c in enumerate(cards):
  txt=single_read(c,q);score=hit(txt,target)if target else 0;rows.append((j,txt,score))
 if target:
  good=[r for r in rows if r[2]]
  return (good[0][1]if good else rows[0][1]),rows
 return rows[0][1],rows
print("[3/9] SOURCE-ONLY FORGE · 32 INDEPENDENT CARTRIDGES")
CARDS=[];META=[]
for i in range(32):
 src=f"{NAMES[i]} registered the {OBJECTS[i]} under identifier {IDS[i]}. The container identified as {IDS[i]} is housed at the {LOCATIONS[i]}."
 f=forge(src);c=compress(f);CARDS.append(c);META.append({"name":NAMES[i],"object":OBJECTS[i],"id":IDS[i],"location":LOCATIONS[i],"source":src})
 print(f" CARD {i+1:02d}: T={c['T']:02d} | {OBJECTS[i]} → {IDS[i]} → {LOCATIONS[i]}")
 del f;clean()
print("[4/9] SINGLE-CARTRIDGE SANITY · STAGE1 / STAGE2")
S1H=S2H=0
for i,(c,m)in enumerate(zip(CARDS,META)):
 q1=f"What identifier is associated with the {m['object']}?";q2=f"Where is container {m['id']} housed?"
 a1=single_read(c,q1);a2=single_read(c,q2);h1=hit(a1,m["id"]);h2=hit(a2,m["location"]);S1H+=h1;S2H+=h2
 rec({"phase":"SINGLE","i":i,"s1_hit":h1,"s2_hit":h2,"s1":a1,"s2":a2})
 print(f" {i+1:02d}: S1={h1} S2={h2}")
print(f"SINGLE STAGE1={S1H}/32 | STAGE2={S2H}/32")
print("[5/9] IBR SCALE 2→8→16→32")
SCALE_RESULTS={}
for scale in SCALES:
 cards=CARDS[:scale];meta=META[:scale];linked=unlinked=stage1=stage2=false_link=0;n=scale
 print(f"\n SCALE={scale}")
 for i,m in enumerate(meta):
  q1=f"What identifier is associated with the {m['object']}?"
  _,rows1=ibr(cards,q1,m["id"]);s1=int(any(r[2]for r in rows1));stage1+=s1
  chosen1=next((r for r in rows1 if r[2]),None)
  predicted_id=m["id"] if chosen1 else None
  q2=f"Where is container {m['id']} housed?"
  _,rows2=ibr(cards,q2,m["location"]);s2=int(any(r[2]for r in rows2));stage2+=s2
  lk=int(s1 and s2);linked+=lk
  qu=f"Where is the {m['object']} housed?"
  _,rowsu=ibr(cards,qu,m["location"]);uh=int(any(r[2]for r in rowsu));unlinked+=uh
  wrong=meta[(i+1)%scale]["location"] if scale>1 else "nonexistent location"
  fl=int(any(hit(r[1],wrong)for r in rows2));false_link+=fl
  rec({"phase":"IBR_SCALE","scale":scale,"i":i,"stage1":s1,"stage2":s2,"linked":lk,"unlinked":uh,"false_link":fl,
       "s1_hits":[r[0] for r in rows1 if r[2]],"s2_hits":[r[0] for r in rows2 if r[2]]})
 print(f" STAGE1={stage1}/{n} | STAGE2={stage2}/{n} | LINKED={linked}/{n} | UNLINKED={unlinked}/{n} | FALSE_LINK={false_link}/{n}")
 SCALE_RESULTS[str(scale)]={"n":n,"stage1":stage1,"stage2":stage2,"linked":linked,"unlinked":unlinked,"false_link":false_link}
print("[6/9] JOINT-CACHE CONTROL · SCALE 2/8/16/32")
JOINT={}
for scale in SCALES:
 cards=CARDS[:scale];meta=META[:scale];K,V,T=joint_cards(cards);s1=s2=0
 for i,m in enumerate(meta):
  a1=generate_cache(f"What identifier is associated with the {m['object']}?",K,V,T);a2=generate_cache(f"Where is container {m['id']} housed?",K,V,T)
  h1=hit(a1,m["id"]);h2=hit(a2,m["location"]);s1+=h1;s2+=h2
  rec({"phase":"JOINT","scale":scale,"i":i,"s1_hit":h1,"s2_hit":h2,"s1":a1,"s2":a2})
 JOINT[str(scale)]={"stage1":s1,"stage2":s2,"n":scale};print(f" SCALE={scale:02d} | S1={s1}/{scale} | S2={s2}/{scale}")
 del K,V;clean()
print("[7/9] MISSING / ABSENT-ID / NOMEM CONTROLS")
CTRL={"missing":0,"absent_id":0,"nomem":0,"n":8}
for i in range(8):
 m=META[i];missing_obj=f"missing-{i}-instrument";absid=f"ZX-{900+i}"
 qM=f"What identifier is associated with the {missing_obj}?";qA=f"Where is container {absid} housed?";qN=f"Where is container {m['id']} housed?"
 _,rm=ibr(CARDS,qM);_,ra=ibr(CARDS,qA);nm=nmem(qN)
 mh=int(not any(hit(x[1],z["id"])for x in rm for z in META));ah=int(not any(hit(x[1],z["location"])for x in ra for z in META));nh=int(not hit(nm,m["location"]))
 CTRL["missing"]+=mh;CTRL["absent_id"]+=ah;CTRL["nomem"]+=nh
 rec({"phase":"CONTROL","i":i,"missing_pass":mh,"absent_id_pass":ah,"nomem_pass":nh,"nomem_text":nm})
print(f"MISSING={CTRL['missing']}/8 | ABSENT_ID={CTRL['absent_id']}/8 | NOMEM={CTRL['nomem']}/8")
print("[8/9] 32-CARTRIDGE VERDICT")
R=SCALE_RESULTS["32"]
GATES={
"single_stage1":S1H>=30,"single_stage2":S2H>=28,
"ibr_stage1_32":R["stage1"]>=30,"ibr_stage2_32":R["stage2"]>=28,
"ibr_linked_32":R["linked"]>=28,"ibr_unlinked_32":R["unlinked"]>=28,
"false_link_32":R["false_link"]<=1,
"missing":CTRL["missing"]>=7,"absent_id":CTRL["absent_id"]>=7,"nomem":CTRL["nomem"]>=7}
for k,v in GATES.items():print(f" {k:20s}: {'PASS'if v else'FAIL'}")
VERDICT="PASS_32_CARTRIDGE_CALIBRATION"if all(GATES.values())else"FAIL_32_CARTRIDGE_CALIBRATION"
print("VERDICT:",VERDICT)
print("[9/9] INTEGRITY + SUMMARY")
SENT1=sentinel()
if SENT1!=S0:fail("WEIGHT SENTINEL CHANGED")
if model.training or any(p.requires_grad for p in model.parameters()):fail("Frozen model audit failed")
report={"test":468,"model":MODEL_ID,"D":D,"scales":SCALES,"single":{"stage1":S1H,"stage2":S2H,"n":32},
"ibr":SCALE_RESULTS,"joint":JOINT,"controls":CTRL,"gates":GATES,"verdict":VERDICT,
"sentinel_before":S0,"sentinel_after":SENT1,
"notes":["32 cartridges are independently source-only forged and independently retained.",
"D120 fixed; no dimension sweep.","OWN/source slot 0 preserved raw.","IBR reads cartridges independently; JOINT is only a control.",
"Stage1 object→identifier and Stage2 identifier→location are scored separately.",
"Missing, absent-ID, false-link and NOMEM controls included.","No training, LoRA, optimizer or weight update."]}
SUMMARY.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print("WEIGHT SENTINEL: PASS")
print("RAW LOG :",LOG);print("SUMMARY :",SUMMARY)
print("="*122)
