# TEST488 — AKBASCORE MAM · FROZEN RECALL ATLAS HELD-OUT 64
# Copyright © 2026 Mustafa Akbaş
# 64 completely new memories · includes hard/extreme nameless cases.
# FROZEN FROM TEST487:
# SINGLE = TOKENMAX_L10
# CONSENSUS = TOKENMAX L10/L13/L14/L09/L11/L15/L12 + HMEAN COS L11/L08 + DELTA COS L12/L10
# Equal rank-normalized weights. NO search/tuning/router/ANN/training/LoRA/readout/model change.
import os,sys,subprocess,importlib.util,random,re,time,hashlib,json
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate"),("numpy","numpy")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
TEST="488";SEED=488;MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SEP="\n\n";DEVICE="cuda"
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required")
torch.set_grad_enabled(False);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

# object, container, nameless description, difficulty
DATA=[
("copper hygrometer","QA-731","copper instrument used to determine how much moisture is present in the air","STANDARD"),
("velvet abacus","BX-204","velvet-covered counting frame whose sliding pieces support manual arithmetic","STANDARD"),
("silver periscope","CN-865","silver optical device allowing someone to see over or around an obstruction","STANDARD"),
("amber tuning fork","DV-419","amber-colored two-pronged object that produces a reference pitch when struck","STANDARD"),
("wooden sundial","ER-572","wooden device that indicates daytime by the position of a shadow cast by the sun","STANDARD"),
("glass thermometer","FT-936","transparent instrument used to determine how hot or cold something is","STANDARD"),
("bronze gyroscope","GU-148","bronze spinning device that resists changes to its orientation","STANDARD"),
("linen atlas","HV-683","fabric-bound book containing organized maps of geographical regions","STANDARD"),
("jade magnifier","JW-257","green-stone framed lens used to make small details appear larger","STANDARD"),
("ceramic whistle","KX-804","pottery object that produces a sharp tone when air is blown through it","STANDARD"),
("ivory ruler","LY-391","pale straight-edged tool marked for determining linear dimensions","STANDARD"),
("crimson funnel","MZ-746","red tapered utensil that guides liquid into a narrow opening","STANDARD"),
("granite mortar","NA-518","stone bowl in which material is crushed by pressing and grinding","STANDARD"),
("silk map","PB-972","smooth-fabric representation showing locations and spatial relationships","STANDARD"),
("brass pendulum","QC-163","metal weight suspended so that gravity makes it swing back and forth","STANDARD"),
("opal prism","RD-625","gemlike transparent solid that separates incoming white light into colors","STANDARD"),
("leather ledger","SE-480","hide-bound book used for systematically recording accounts and transactions","STANDARD"),
("iron plumb bob","TF-219","heavy metal weight hanging from a line to establish a true vertical direction","STANDARD"),
("pearl binoculars","UG-854","pale paired optical tubes used with both eyes to inspect distant scenes","STANDARD"),
("bamboo protractor","VH-307","plant-stem angle tool marked with a semicircular scale","STANDARD"),
("quartz stopwatch","WJ-691","crystal-based handheld timer intended to measure short elapsed intervals","STANDARD"),
("canvas stencil","XK-425","cloth template with openings through which a repeated shape can be transferred","STANDARD"),
("marble chessboard","YL-738","stone grid of alternating squares used for a strategic game between two armies of pieces","STANDARD"),
("onyx compass divider","ZM-146","dark hinged two-legged drafting tool used to transfer or compare distances","STANDARD"),

# HARD: head noun absent; function/context must carry retrieval.
("scarlet stethoscope","AR-583","red device a clinician places against the body to listen to sounds produced inside the chest","HARD"),
("cedar hourglass","BS-927","wood-framed object in which grains descend through a narrow waist to mark an interval","HARD"),
("crystal level","CT-364","transparent construction tool whose trapped bubble reveals whether a surface is horizontal","HARD"),
("woolen thesaurus","DU-815","fabric-bound reference volume consulted when a writer wants another word with a similar meaning","HARD"),
("bronze odometer","EV-249","metal instrument that accumulates how far a vehicle has travelled","HARD"),
("porcelain anemometer","FW-670","ceramic weather instrument whose moving parts reveal how quickly air is flowing","HARD"),
("jade seismograph","GX-132","green-stone housed apparatus that records vibrations travelling through the ground","HARD"),
("linen calendar","HY-596","fabric sheet organizing days into months so future dates can be located","HARD"),
("cobalt altimeter","JZ-843","blue instrument used by a pilot to determine height above a reference level","HARD"),
("maple metronome","KA-275","wooden device musicians follow when they need evenly spaced beats","HARD"),
("ivory caliper","LB-914","pale tool whose opposing jaws close around an object to determine its dimensions","HARD"),
("glass kaleidoscope","MC-368","transparent viewing tube in which reflections repeatedly reorganize colored fragments into symmetrical patterns","HARD"),
("iron sextant","ND-751","metal navigation instrument used to infer position by measuring an angle involving a celestial body and the horizon","HARD"),
("cotton manuscript","PE-426","fabric-wrapped pages bearing a text produced by hand rather than by ordinary printing","HARD"),
("amber phonograph","QF-809","golden-brown historical machine that turns physical grooves into audible recorded sound","HARD"),
("granite barometer","RG-153","stone-cased weather device consulted to learn whether surrounding air pressure is rising or falling","HARD"),

# EXTREME: indirect scenario, metaphor, missing obvious noun, relational/causal description.
("obsidian compass","SH-684","the dark object you would want after becoming disoriented in featureless terrain when north is no longer obvious","EXTREME"),
("velvet thermometer","TJ-237","the soft-covered thing someone might reach for when a child feels unusually warm and they want a number rather than a guess","EXTREME"),
("silver stopwatch","UK-795","the metallic object a race official would trigger at the start and stop at the finish to quantify what happened between them","EXTREME"),
("ceramic magnifier","VL-341","the pottery-framed helper chosen when printed letters are present but too small for the eye to resolve comfortably","EXTREME"),
("copper tuning fork","WM-928","the copper object a musician could strike before tuning another instrument when a stable reference note is needed","EXTREME"),
("silk atlas","XN-476","the smooth-bound thing opened when someone knows a country exists but cannot picture where on Earth it lies","EXTREME"),
("jade abacus","YP-603","the green object on which quantities can be manipulated physically without writing numerals or using electronics","EXTREME"),
("brass plumb bob","ZQ-157","the metal object a builder lets gravity pull downward when needing a line that does not depend on eyesight or the wall","EXTREME"),
("opal prism","AS-842","the gemlike object that can make apparently colorless illumination reveal that it was carrying many visible colors","EXTREME"),
("linen funnel","BT-395","the fabric-covered helper chosen because pouring directly from a wide vessel into a tiny mouth would otherwise spill","EXTREME"),
("onyx pendulum","CU-761","the dark suspended mass whose repeated journey from side to side can provide a regular physical rhythm","EXTREME"),
("crystal hygrometer","DV-208","the transparent device that answers whether the surrounding air is unusually dry or saturated without asking about temperature","EXTREME"),
("wooden periscope","EW-654","the wooden optical arrangement useful when your eyes must remain below an obstacle while your view somehow reaches above it","EXTREME"),
("bronze gyroscope","FX-319","the spinning bronze mechanism whose axis seems stubborn about keeping its direction even while its support is moved","EXTREME"),
("canvas stencil","GY-870","the cloth aid used when the same outline must appear repeatedly without drawing that outline freehand each time","EXTREME"),
("marble sundial","HZ-425","the stone object that can tell part of the day while having no battery, gears or display, provided the sky supplies a moving shadow","EXTREME"),

# ADVERSARIAL-HARD: semantically close alternatives deliberately exist elsewhere in bank.
("white anemometer","JA-936","the pale weather device concerned with moving air rather than the force exerted by the atmosphere itself","ADVERSARIAL"),
("black barometer","KB-271","the dark weather device concerned with atmospheric force rather than how quickly the air is moving","ADVERSARIAL"),
("coral altimeter","LC-508","the reddish instrument answering how high an aircraft is, not how far it has travelled along the ground","ADVERSARIAL"),
("steel odometer","MD-743","the metal vehicle instrument accumulating travelled distance rather than elapsed journey time","ADVERSARIAL"),
("azure protractor","NE-186","the blue geometry tool for quantifying an angle rather than a straight-line length","ADVERSARIAL"),
("golden ruler","PF-659","the yellow straight tool for linear extent rather than angular extent","ADVERSARIAL"),
("quartz binoculars","QG-324","the crystal-framed paired optics intended for distant viewing with both eyes rather than enlarging nearby print","ADVERSARIAL"),
("scarlet stethoscope","RH-817","the red clinical listening device for internal body sounds rather than an instrument for measuring body heat","ADVERSARIAL")]
N=len(DATA)
if N!=64:raise RuntimeError(N)
SOURCES=[f"The {o} is stored in container {c}." for o,c,_,_ in DATA]
QUERIES=[f"Which container holds the {d}?" for _,_,d,_ in DATA]

# Strict leakage: complete object name must not occur in query.
for i,(o,_,_,_) in enumerate(DATA):
    if o.casefold() in QUERIES[i].casefold():raise RuntimeError(f"Object leak {i}: {o}")
FROZEN_CHANNELS=["TOKENMAX_L10","TOKENMAX_L13","HMEAN_COS_L11","TOKENMAX_L14","TOKENMAX_L09",
                 "DELTA_COS_L12","TOKENMAX_L11","HMEAN_COS_L08","TOKENMAX_L15","TOKENMAX_L12","DELTA_COS_L10"]
LOCK={"test":TEST,"foundation":"TEST487","single":"TOKENMAX_L10","consensus":FROZEN_CHANNELS,"data":DATA}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()

print("="*150);print("TEST488 — AKBASCORE MAM · FROZEN RECALL ATLAS HELD-OUT 64")
print("64 NEW MEMORIES · STANDARD/HARD/EXTREME/ADVERSARIAL · TEST487 CHANNELS FROZEN");print("="*150)
print("LOCK SHA:",LOCK_SHA);print("FROZEN SINGLE   : TOKENMAX_L10");print("FROZEN CONSENSUS:",FROZEN_CHANNELS);T0=time.perf_counter()

print("[1/6] Loading frozen Qwen...")
tv=tuple(int("".join(c for c in x if c.isdigit())or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}).eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;cfg=model.config
if (len(layers),cfg.hidden_size,cfg.num_attention_heads,cfg.num_key_value_heads)!=(28,3584,28,4):raise RuntimeError("Architecture mismatch")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
enc=lambda s:tok(s,add_special_tokens=False).input_ids
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | frozen BF16")

NEED=[8,9,10,11,12,13,14,15]
@torch.inference_mode()
def probe(text):
    ids=[PAD]+enc(text+SEP);o=model(input_ids=torch.tensor([ids],device=DEVICE),output_hidden_states=True,use_cache=False,return_dict=True)
    return {l:o.hidden_states[l][0,1:].float().cpu() for l in range(8,16)}

print("[2/6] Capturing 64 new memory states...")
B=[]
for i,s in enumerate(SOURCES):
    B.append(probe(s))
    if (i+1)%8==0:print(f"      {i+1:02d}/64")
print("[3/6] Capturing 64 held-out NAMELESS queries...")
Q=[]
for i,q in enumerate(QUERIES):
    Q.append(probe(q))
    if (i+1)%8==0:print(f"      {i+1:02d}/64")

def norm(x):return torch.nn.functional.normalize(x.float(),dim=-1)
def rank(s,g):return int((torch.argsort(s,descending=True)==g).nonzero(as_tuple=False)[0,0])+1
def metrics(rs):
    a=np.asarray(rs,float)
    return {"R1":float(np.mean(a<=1)),"R5":float(np.mean(a<=5)),"R10":float(np.mean(a<=10)),
            "R32":float(np.mean(a<=32)),"MRR":float(np.mean(1/a)),"MEDR":float(np.median(a)),"MEANR":float(np.mean(a))}
def tokenmax(l):
    M=torch.empty(N,N)
    for i in range(N):
        q=norm(Q[i][l])
        for j in range(N):
            b=norm(B[j][l]);M[i,j]=(q@b.T).max(dim=1).values.mean()
    return M
def hmean(l):
    A=norm(torch.stack([Q[i][l].mean(0) for i in range(N)]))
    C=norm(torch.stack([B[i][l].mean(0) for i in range(N)]))
    return A@C.T
def delta(l):
    A=norm(torch.stack([Q[i][l].mean(0)-Q[i][l-1].mean(0) for i in range(N)]))
    C=norm(torch.stack([B[i][l].mean(0)-B[i][l-1].mean(0) for i in range(N)]))
    return A@C.T
def rankscore(M):
    R=torch.empty_like(M)
    for i in range(N):
        o=torch.argsort(M[i],descending=True);r=torch.empty(N)
        r[o]=torch.arange(1,N+1,dtype=torch.float32);R[i]=1-(r-1)/(N-1)
    return R
def evalM(M):
    rr=[rank(M[i],i) for i in range(N)];return metrics(rr),rr

print("[4/6] Running TEST487 frozen channels...")
M={}
for l in [9,10,11,12,13,14,15]:
    M[f"TOKENMAX_L{l:02d}"]=tokenmax(l);print(f"      TOKENMAX L{l:02d}")
for l in [8,11]:
    M[f"HMEAN_COS_L{l:02d}"]=hmean(l);print(f"      HMEAN L{l:02d}")
for l in [10,12]:
    M[f"DELTA_COS_L{l:02d}"]=delta(l);print(f"      DELTA L{l:02d}")
SINGLE,SR=evalM(M["TOKENMAX_L10"])
CONS_M=torch.stack([rankscore(M[x]) for x in FROZEN_CHANNELS]).mean(0)
CONS,CR=evalM(CONS_M)

print("[5/6] Difficulty-stratified held-out evaluation...")
GROUPS={}
for g in ["STANDARD","HARD","EXTREME","ADVERSARIAL"]:
    ix=[i for i,x in enumerate(DATA) if x[3]==g]
    def gm(rr):
        a=np.asarray([rr[i] for i in ix],float)
        return {"N":len(ix),"R1":float(np.mean(a<=1)),"R5":float(np.mean(a<=5)),
                "R10":float(np.mean(a<=10)),"MRR":float(np.mean(1/a)),"MEDR":float(np.median(a))}
    GROUPS[g]={"SINGLE":gm(SR),"CONSENSUS":gm(CR)}
    print(f"\n      {g} N={len(ix)}")
    print("      SINGLE   :",GROUPS[g]["SINGLE"])
    print("      CONSENSUS:",GROUPS[g]["CONSENSUS"])

print("\n      PER-CASE")
for i,(o,c,d,g) in enumerate(DATA):
    print(f"      [{i:02d}] {g:11s} single={SR[i]:2d} cons={CR[i]:2d} | {o:22s} -> {c}")

print("[6/6] Controls...")
def words(s):return set(re.findall(r"[a-z0-9]+",s.casefold()))
SW=[words(s) for s in SOURCES];LR=[]
for i,q in enumerate(QUERIES):
    qw=words(q);v=np.array([len(qw&s)/max(1,len(qw|s)) for s in SW]);o=np.argsort(-v,kind="stable");LR.append(int(np.where(o==i)[0][0])+1)
LEX=metrics(LR)
rng=np.random.default_rng(SEED);NULLS=[]
for z in range(1000):
    p=rng.permutation(N);NULLS.append(np.mean([1/rank(CONS_M[i],int(p[i])) for i in range(N)]))
NULL={"MRR_MEAN":float(np.mean(NULLS)),"MRR_P95":float(np.quantile(NULLS,.95)),"MRR_P99":float(np.quantile(NULLS,.99))}
if CONS["R1"]>=.65 and CONS["R5"]>=.85 and GROUPS["EXTREME"]["CONSENSUS"]["R5"]>=.60:
    VERDICT="HELDOUT_MULTI_SIGNAL_RECALL_CONFIRMED"
elif CONS["R1"]>=.50 and CONS["R5"]>=.75:
    VERDICT="HELDOUT_RECALL_CONFIRMED_WITH_HARD_CASE_DEGRADATION"
elif CONS["MRR"]>NULL["MRR_P99"]*3:
    VERDICT="HELDOUT_SIGNAL_CONFIRMED_BUT_NOT_YET_STRONG_RETRIEVAL"
else:VERDICT="TEST487_ATLAS_DID_NOT_GENERALIZE_STRONGLY"

print("\n"+"="*150);print("TEST488 FINAL RESULT — FROZEN HELD-OUT RECALL");print("="*150)
print("FOUNDATION          : TEST487 DISCOVERY ATLAS")
print("TEST482 BLOB SHA    : ecc630144a44479cb35c432c50eb0d09bc6866d7")
print("NEW MEMORIES        : 64/64")
print("STANDARD/HARD       : 24 / 16")
print("EXTREME/ADVERSARIAL: 16 / 8")
print("CHANNEL SEARCH      : NONE")
print("WEIGHT SEARCH       : NONE · equal frozen rank consensus")
print("LOCK SHA            :",LOCK_SHA)
print("-"*150)
print("TOKENMAX_L10        :",SINGLE)
print("FROZEN CONSENSUS    :",CONS)
print("LEXICAL CONTROL     :",LEX)
print("1000-PERM NULL      :",NULL)
for g,x in GROUPS.items():print(f"{g:19s}:",x)
print("-"*150)
print("VERDICT             :",VERDICT)
print("INTERPRETATION      : fresh 64-memory held-out validation of TEST487-discovered native recall channels.")
print("ORACLE              : NOT USED")
print("POST-HOC SELECTION  : NONE")
print("ROUTER / ANN / GRAPH: NONE")
print("CARTRIDGE READOUT   : NOT USED")
print("TRAINING / LoRA     : NONE")
print("MODEL CHANGE        : NONE")
print(f"TOTAL TIME          : {time.perf_counter()-T0:.2f}s")
print("="*150)
