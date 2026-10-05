# TEST486 — AKBASCORE MAM · SEMANTIC DISTANCE CURVE
# Copyright © 2026 Mustafa Akbaş
# New original AKBASCORE MAM material is source-available under the repository LICENSE.
# Earlier MIT-licensed Titan Cognitive Core / AkbasCore material remains under its original license.
# See repository LICENSE for complete terms.
#
# FROZEN: TEST482 Qwen2.5-7B-Instruct · TEST484 HMEAN/L03/CENTER_COS
# TEST: EXACT -> PARAPHRASE -> NAMELESS semantic-distance curve
# NO config search/router/ANN/readout/training/LoRA/gradient/model change.

import os,sys,subprocess,importlib.util,random,re,time,hashlib,json
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate"),("numpy","numpy")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
TEST="486";SEED=486;MODEL_ID="Qwen/Qwen2.5-7B-Instruct";L=3;SEP="\n\n";DEVICE=torch.device("cuda")
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
torch.set_grad_enabled(False);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

RECORDS=[
("F1","emerald barometer","KR-214","cedar archive"),("F1","porcelain compass","DM-763","marble annex"),
("F1","violet harmonica","FS-408","amber gallery"),("F1","bamboo chronometer","JN-951","fern vault"),
("F1","obsidian sextant","RW-326","copper archive"),("F1","linen telescope","AE-875","quartz annex"),
("F1","coral metronome","LP-143","ivory gallery"),("F1","willow calculator","XM-692","silver vault"),
("F2","turquoise monocle","HC-517","indigo lighthouse"),("F2","mahogany notebook","VK-280","pearl pavilion"),
("F2","ceramic astrolabe","SB-934","granite lodge"),("F2","scarlet blueprint","NT-461","willow gallery"),
("F2","opal kaleidoscope","GY-708","maple tower"),("F2","bronze hourglass","PD-352","coral pavilion"),
("F2","silk manuscript","ZU-819","birch lodge"),("F2","crystal chime","EF-625","onyx gallery"),
("F3","jade projector","WL-407","hazel chamber"),("F3","glass clarinet","CQ-586","lagoon studio"),
("F3","canvas diary","MR-172","acorn room"),("F3","iron medallion","UX-943","poplar hall"),
("F3","pearl camera","BH-650","cobalt chamber"),("F3","steel accordion","KO-238","canal studio"),
("F3","cotton almanac","YD-714","walnut room"),("F3","onyx brooch","TG-569","elm hall"),
("F4","ivory receiver","PS-381","summit workshop"),("F4","cedar plaque","AL-826","valley depot"),
("F4","azure kettle","RF-504","birch observatory"),("F4","brass caliper","MW-197","shore conservatory"),
("F4","crystal phonograph","DE-648","forest workshop"),("F4","maple slate","KI-275","delta depot"),
("F4","white lantern","OV-930","clover observatory"),("F4","golden protractor","XC-412","island conservatory")]

def source(f,o,c,p):
    if f=="F1":return f"The {o} is stored in container {c}."
    if f=="F2":return f"Container {c} contains the {o}."
    if f=="F3":return f"The {o} can be found inside container {c}."
    return f"Inside container {c} there is the {o}."
SOURCES=[source(*r) for r in RECORDS]

PARA=[
"green pressure-measuring barometer","ceramic direction-finding compass","purple mouth-played harmonica","bamboo precision chronometer",
"dark celestial-navigation sextant","fabric-covered long-distance telescope","reddish tempo-keeping metronome","wooden arithmetic calculator",
"blue-green single-eye monocle","dark wooden writing notebook","pottery celestial-position astrolabe","bright red technical blueprint",
"gemstone-colored pattern-forming kaleidoscope","metal sand-timing hourglass","smooth-fabric handwritten manuscript","transparent ringing chime",
"green-stone image projector","transparent single-reed clarinet","cloth-covered personal diary","metal keepsake medallion",
"pale gemstone-colored photo camera","metal bellows-driven accordion","soft-fabric annual almanac","black gemstone ornamental brooch",
"pale signal receiver","wooden commemorative plaque","blue water-heating kettle","metal dimension-measuring caliper",
"transparent sound-reproducing phonograph","wooden writing slate","pale portable lantern","yellow-metal angle-measuring protractor"]

NAMELESS=[
"green instrument used to measure atmospheric pressure","ceramic instrument used to determine direction",
"purple handheld instrument played by blowing and drawing air","bamboo device used for precise timekeeping",
"dark volcanic-stone instrument historically used for celestial navigation","fabric-covered optical instrument for viewing distant objects",
"reddish device used to keep a steady musical tempo","wooden device used for arithmetic calculations",
"blue-green single-eye optical lens","dark wooden book used for handwritten notes",
"pottery instrument historically used to determine celestial position","bright red technical drawing",
"gemstone-colored optical toy that forms changing symmetrical patterns","metal timer in which sand falls between two glass chambers",
"smooth-fabric document written by hand","transparent decorative object that produces a ringing sound",
"green-stone device used to display images on a surface","transparent woodwind instrument played with a single reed",
"cloth-covered personal book used for daily written entries","metal ornamental disk worn or kept as a keepsake",
"pale gemstone-colored device used to take photographs","metal bellows-driven musical instrument with keys or buttons",
"soft-fabric annual reference book containing facts and dates","black gemstone ornamental pin",
"pale handheld apparatus used to receive transmitted signals","wooden flat sign or commemorative plate",
"blue vessel used to heat water","metal measuring tool with two jaws for measuring dimensions",
"transparent historical machine that reproduces recorded sound","wooden flat board used for writing",
"pale portable light used for illumination","yellow-metal instrument used to measure and draw angles"]

EXACT=[f"Which container holds the {r[1]}?" for r in RECORDS]
PARAPHRASE=[f"Which container holds the {x}?" for x in PARA]
NAMELESS_Q=[f"Which container holds the {x}?" for x in NAMELESS]
PANELS={"EXACT":EXACT,"PARAPHRASE":PARAPHRASE,"NAMELESS":NAMELESS_Q}

for i,r in enumerate(RECORDS):
    if r[1].casefold() in NAMELESS_Q[i].casefold():raise RuntimeError(f"NAMELESS leak {i}")
LOCK={"test":TEST,"model":MODEL_ID,"layer":L,"metric":"CENTER_COS","records":RECORDS,"sources":SOURCES,"panels":PANELS}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("="*145);print("TEST486 — AKBASCORE MAM · SEMANTIC DISTANCE CURVE")
print("32 CARTRIDGES · HMEAN L03 CENTER_COS FROZEN · EXACT -> PARAPHRASE -> NAMELESS");print("="*145)
print("LOCK SHA:",LOCK_SHA);T0=time.perf_counter()

print("[1/5] Loading frozen Qwen...")
tv=tuple(int("".join(c for c in x if c.isdigit())or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}).eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;cfg=model.config
if (len(layers),cfg.hidden_size,cfg.num_attention_heads,cfg.num_key_value_heads)!=(28,3584,28,4):raise RuntimeError("Architecture mismatch")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
enc=lambda s:tok(s,add_special_tokens=False).input_ids
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | frozen BF16")

@torch.inference_mode()
def hmean(s):
    ids=[PAD]+enc(s+SEP)
    o=model(input_ids=torch.tensor([ids],device=DEVICE),output_hidden_states=True,use_cache=False,return_dict=True)
    return o.hidden_states[L][0][1:].float().mean(0).cpu()

print("[2/5] Encoding frozen cartridge addresses...")
B=torch.stack([hmean(s) for s in SOURCES]).float();C=B.mean(0)
def nr(x):return torch.nn.functional.normalize(x.float(),dim=-1)
BN=nr(B-C)
def scores(q):return BN@torch.nn.functional.normalize(q.float()-C,dim=0)
def rankof(s,i):return int((torch.argsort(s,descending=True)==i).nonzero(as_tuple=False)[0,0])+1
def stats(r):
    a=np.asarray(r,float)
    return {"R1":float(np.mean(a<=1)),"R5":float(np.mean(a<=5)),"R16":float(np.mean(a<=16)),
            "MRR":float(np.mean(1/a)),"MEDR":float(np.median(a)),"MEANR":float(np.mean(a))}

print("[3/5] Running locked EXACT -> PARAPHRASE -> NAMELESS curve...")
RESULT={};RANKS={}
for name,panel in PANELS.items():
    ranks=[];marg=[];zs=[]
    print(f"\n      {name}")
    for i,q in enumerate(panel):
        s=scores(hmean(q));w=torch.cat([s[:i],s[i+1:]])
        r=rankof(s,i);mg=float(s[i]-w.max());sd=float(w.std(unbiased=False))+1e-12
        ranks.append(r);marg.append(mg);zs.append((float(s[i])-float(w.mean()))/sd)
        print(f"      [{i:02d}] rank={r:2d} margin={mg:+.5f} | {RECORDS[i][1]}")
    RESULT[name]={**stats(ranks),"MED_MARGIN":float(np.median(marg)),"MED_Z":float(np.median(zs))}
    RANKS[name]=ranks
    print("      =>",RESULT[name])

print("\n[4/5] Lexical controls...")
def words(s):return set(re.findall(r"[a-z0-9]+",s.casefold()))
SW=[words(x) for x in SOURCES];LEX={}
for name,panel in PANELS.items():
    rr=[]
    for i,q in enumerate(panel):
        qw=words(q);v=np.array([len(qw&s)/max(1,len(qw|s)) for s in SW]);o=np.argsort(-v,kind="stable")
        rr.append(int(np.where(o==i)[0][0])+1)
    LEX[name]=stats(rr)
    print(f"      {name:10s}: {LEX[name]}")

rng=np.random.default_rng(SEED);perm=np.arange(32);rng.shuffle(perm);NULL={}
for name,panel in PANELS.items():
    rr=[]
    for i,q in enumerate(panel):rr.append(rankof(scores(hmean(q)),int(perm[i])))
    NULL[name]=stats(rr)
    print(f"      NULL {name:10s}: {NULL[name]}")

print("\n[5/5] Distance-curve diagnostics...")
r0=RESULT["EXACT"];r1=RESULT["PARAPHRASE"];r2=RESULT["NAMELESS"]
DROP_EP=r0["MRR"]-r1["MRR"];DROP_PN=r1["MRR"]-r2["MRR"]
if r1["R1"]>=.50 and r1["R5"]>=.75 and r1["MRR"]>=LEX["PARAPHRASE"]["MRR"]+.15:
    VERDICT="STRONG_PARAPHRASE_ADDRESSABILITY"
elif r1["R5"]>=.50 and r1["MRR"]>=.35 and r1["MRR"]>=LEX["PARAPHRASE"]["MRR"]+.10:
    VERDICT="PARTIAL_PARAPHRASE_ADDRESSABILITY"
elif r1["MRR"]>NULL["PARAPHRASE"]["MRR"]+.10:
    VERDICT="WEAK_PARAPHRASE_SIGNAL_ABOVE_NULL"
else:VERDICT="ADDRESS_SIGNAL_REQUIRES_STRONG_LEXICAL_ANCHOR"

print("\n"+"="*145);print("TEST486 FINAL RESULT — AKBASCORE MAM");print("="*145)
print("FOUNDATION          : TEST482 QWEN")
print("TEST482 BLOB SHA    : ecc630144a44479cb35c432c50eb0d09bc6866d7")
print("ADDRESS LOCK        : HMEAN · L03 · CENTER_COS")
print("CONFIG SEARCH       : NONE")
print("CARTRIDGES          : 32")
print("LOCK SHA            :",LOCK_SHA)
print("-"*145)
print("EXACT               :",RESULT["EXACT"])
print("PARAPHRASE          :",RESULT["PARAPHRASE"])
print("NAMELESS            :",RESULT["NAMELESS"])
print("LEX EXACT           :",LEX["EXACT"])
print("LEX PARAPHRASE      :",LEX["PARAPHRASE"])
print("LEX NAMELESS        :",LEX["NAMELESS"])
print("NULL PARAPHRASE     :",NULL["PARAPHRASE"])
print(f"MRR DROP E->P       : {DROP_EP:+.6f}")
print(f"MRR DROP P->N       : {DROP_PN:+.6f}")
print("-"*145)
print("VERDICT             :",VERDICT)
print("INTERPRETATION      : fixed semantic-distance curve only; no scaling/sublinear-retrieval claim.")
print("CARTRIDGE READOUT   : NOT USED")
print("ROUTER / ANN        : NONE")
print("TRAINING / LoRA     : NONE")
print("MODEL CHANGE        : NONE")
print("POST-HOC TUNING     : NONE")
print(f"TOTAL TIME          : {time.perf_counter()-T0:.2f}s")
print("="*145)
