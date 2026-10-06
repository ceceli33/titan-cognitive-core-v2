# Copyright © 2026 Mustafa Akbaş
# New original AKBASCORE MAM material is source-available under the repository LICENSE.
# Earlier MIT-licensed Titan Cognitive Core / AkbasCore material remains under its original license.
# See repository LICENSE for complete terms.
#
# TEST493 — AKBASCORE MAM · FROZEN INSTANCE-IDENTITY REPLICATION
# FRESH HELD-OUT CONCEPT PAIRS
# PRIMARY: VTOKEN_L23
# SECONDARY: TQMEAN_L07 / DELTA_L09
# FROZEN FROM TEST492 · NO CHANNEL/LAYER/HEAD SEARCH · NO FUSION · NO RETUNING

import os,sys,json,random,hashlib,subprocess,importlib.util,math
os.environ["TOKENIZERS_PARALLELISM"]="false"
def need(p):
    if importlib.util.find_spec(p) is None: subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
for p in ["torch","transformers","accelerate"]: need(p)
import numpy as np,torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from collections import defaultdict

TEST="493";SEED=493;MODEL_ID="Qwen/Qwen2.5-7B-Instruct";DEVICE="cuda"
PRIMARY=("VTOKEN",23);SECONDARY=[("TQMEAN",7),("DELTA",9)]
TEST482_SHA="ecc630144a44479cb35c432c50eb0d09bc6866d7"
TEST488_SHA="d2c2e744532e1fe9b4c5be745fce6892720745dd590b49eca59ca3ef753fdfd8"
TEST489_SHA="ed03fb4b6f1afa410e1a28329d05b744889b8836f69f3572c4b52032b8b70fb0"
TEST490_SHA="237f3c7a9cc2478b2ca55944f0a356101c8f87c5a7ef250f6267fc04a9fc1dd2"
TEST491_SHA="582fdc3e39d58863685be094c1bbfcfac8e398845d90624598197c97be9632bd"
TEST492_SHA="8de05fc52332806383642618fcf4a9ea4ed29fe2f544cac5beb5369686ee79a6"
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)

# 32 entirely fresh concepts × 2 distinct instances = 64.
# None of TEST491's 32 concepts are reused.
# Each pair shares concept/function but differs by an instance attribute.
RAW=[
("chronometer","gold chronometer","AA-731","the yellow-metal precision timekeeping instrument used when very accurate elapsed time is required","STANDARD"),
("chronometer","black chronometer","AB-284","the dark precision timekeeper designed to maintain exceptionally accurate time","HARD"),
("ammeter","red ammeter","AC-615","the red electrical instrument used specifically to measure current flowing through a circuit","STANDARD"),
("ammeter","wooden ammeter","AD-902","the wood-cased meter placed in a circuit when electric current rather than voltage must be quantified","EXTREME"),
("voltmeter","blue voltmeter","AE-347","the blue electrical instrument used to measure potential difference between two points","STANDARD"),
("voltmeter","brass voltmeter","AF-861","the yellow-metal circuit meter connected when electrical potential difference must be read numerically","HARD"),
("ohmmeter","white ohmmeter","AG-526","the pale electrical instrument used specifically to determine resistance","STANDARD"),
("ohmmeter","steel ohmmeter","AH-193","the metallic tester used when opposition to electric current must be expressed in ohms","EXTREME"),
("barograph","green barograph","AI-708","the green instrument that continuously records changes in atmospheric pressure over time","HARD"),
("barograph","copper barograph","AJ-452","the reddish-metal weather device producing a continuous trace of air-pressure variation","EXTREME"),
("pyrometer","silver pyrometer","AK-819","the silver instrument used to determine very high temperature without ordinary contact measurement","STANDARD"),
("pyrometer","ceramic pyrometer","AL-264","the pottery-cased temperature instrument intended for objects too hot for a conventional contact thermometer","EXTREME"),
("nephelometer","amber nephelometer","AM-573","the amber laboratory instrument estimating suspended particles by measuring scattered light","HARD"),
("nephelometer","black nephelometer","AN-940","the dark optical device used when cloudiness or particle concentration is inferred from light scattering","EXTREME"),
("photometer","violet photometer","AO-381","the purple optical instrument used to quantify properties of light","STANDARD"),
("photometer","aluminum photometer","AP-726","the metallic optical measuring device used when light intensity must become a numerical reading","HARD"),
("colorimeter","orange colorimeter","AQ-658","the orange optical instrument estimating concentration from the color of a solution","STANDARD"),
("colorimeter","glass colorimeter","AR-405","the transparent laboratory device comparing light absorption through a colored sample to infer concentration","EXTREME"),
("turbidimeter","ivory turbidimeter","AS-917","the pale instrument used to quantify how cloudy a liquid is","HARD"),
("turbidimeter","jade turbidimeter","AT-236","the green device measuring loss or scattering of light caused by suspended material in a cloudy fluid","EXTREME"),
("electrometer","cobalt electrometer","AU-564","the blue instrument designed to detect or measure electric charge or very small electrical quantities","HARD"),
("electrometer","brass electrometer","AV-823","the yellow-metal electrical instrument sensitive enough to measure tiny charges rather than ordinary large circuit currents","EXTREME"),
("accelerometer","iron accelerometer","AW-190","the heavy-metal sensor used to measure acceleration of a moving or vibrating body","STANDARD"),
("accelerometer","crystal accelerometer","AX-647","the transparent-bodied sensor reporting how quickly velocity changes rather than merely how fast something moves","EXTREME"),
("magnetometer","amber magnetometer","AY-315","the amber instrument used to measure strength or direction of a magnetic field","STANDARD"),
("magnetometer","steel magnetometer","AZ-782","the metallic field instrument used when magnetism must be detected and quantified","HARD"),
("gravimeter","copper gravimeter","BA-439","the copper instrument used for extremely precise measurement of local gravitational acceleration","HARD"),
("gravimeter","quartz gravimeter","BB-806","the crystal-bodied geophysical instrument sensitive to tiny variations in the strength of gravity","EXTREME"),
("radiometer","black radiometer","BC-251","the dark instrument used to measure radiant energy or electromagnetic radiation intensity","STANDARD"),
("radiometer","jade radiometer","BD-694","the green measuring device concerned with energy arriving as radiation rather than mechanical force","EXTREME"),
("interferometer","silver interferometer","BE-538","the silver optical instrument using interference patterns for extremely precise measurements","HARD"),
("interferometer","mahogany interferometer","BF-972","the dark-wood optical apparatus combining waves so tiny differences can be inferred from their interference","EXTREME"),
("goniometer","ivory goniometer","BG-416","the pale instrument used to measure angles precisely","STANDARD"),
("goniometer","azure goniometer","BH-759","the blue angular measuring device used when the orientation between surfaces or directions must be quantified","EXTREME"),
("planimeter","white planimeter","BI-208","the white instrument used to determine the area of an irregular shape drawn on a flat surface","HARD"),
("planimeter","onyx planimeter","BJ-635","the black tracing device moved around a boundary to obtain the enclosed area without counting grid squares","EXTREME"),
("sclerometer","yellow sclerometer","BK-841","the yellow instrument used to estimate hardness by scratching or rebound behavior","HARD"),
("sclerometer","steel sclerometer","BL-397","the metallic testing device used when resistance of a material to indentation or scratching must be assessed","EXTREME"),
("durometer","chrome durometer","BM-620","the shiny instrument used to measure hardness of materials such as rubber or plastics","STANDARD"),
("durometer","red durometer","BN-153","the red tester pressed against a soft material when its resistance to indentation must become a hardness value","EXTREME"),
("tensiometer","green tensiometer","BO-486","the green instrument used to measure tension or surface tension","STANDARD"),
("tensiometer","aluminum tensiometer","BP-925","the metallic measuring device used when pulling tension or a liquid surface's tension must be quantified","EXTREME"),
("hygrograph","orange hygrograph","BQ-374","the orange weather instrument continuously recording humidity over time","HARD"),
("hygrograph","black hygrograph","BR-718","the dark recording device producing a trace showing how moisture in the air changes with time","EXTREME"),
("thermocouple","brass thermocouple","BS-562","the yellow-metal temperature sensor generating a voltage from a junction of dissimilar conductors","HARD"),
("thermocouple","steel thermocouple","BT-839","the metallic temperature probe whose electrical signal arises from joined unlike metals","EXTREME"),
("potentiometer","blue potentiometer","BU-207","the blue electrical device used to vary or measure electric potential through an adjustable resistance arrangement","STANDARD"),
("potentiometer","bronze potentiometer","BV-671","the brown-metal adjustable electrical component whose moving contact changes the division of voltage","EXTREME"),
("rheostat","glass rheostat","BW-348","the transparent-bodied variable resistor used to control current by changing resistance","HARD"),
("rheostat","red rheostat","BX-795","the red adjustable resistance device used when current through a circuit must be varied manually","EXTREME"),
("tachograph","white tachograph","BY-924","the pale recording instrument that logs a vehicle's speed and activity over time","HARD"),
("tachograph","steel tachograph","BZ-461","the metallic vehicle recorder preserving a history of speed rather than merely displaying the present value","EXTREME"),
("spirometer","clear spirometer","CA-683","the transparent medical instrument used to measure volumes of air inhaled and exhaled by the lungs","STANDARD"),
("spirometer","amber spirometer","CB-150","the amber respiratory measuring device used when lung air volume and breathing capacity must be quantified","EXTREME"),
("audiometer","brass audiometer","CC-537","the yellow-metal clinical instrument used to determine hearing sensitivity at different sound frequencies","HARD"),
("audiometer","black audiometer","CD-864","the dark clinical device presenting tones to determine the faintest sounds a person can hear","EXTREME"),
("ophthalmoscope","silver ophthalmoscope","CE-296","the silver medical instrument used to inspect the interior of the eye","STANDARD"),
("ophthalmoscope","wooden ophthalmoscope","CF-743","the wood-cased viewing instrument a clinician uses to examine structures behind the pupil","EXTREME"),
("otoscope","glass otoscope","CG-519","the transparent-bodied medical viewing instrument used to inspect the ear canal and eardrum","STANDARD"),
("otoscope","copper otoscope","CH-872","the reddish-metal handheld clinical viewer used when the inside of an ear must be examined","EXTREME"),
("respirometer","white respirometer","CI-430","the pale laboratory instrument used to measure the rate of respiration through gas exchange","HARD"),
("respirometer","black respirometer","CJ-965","the dark apparatus tracking oxygen consumption or another gas change to quantify biological respiration","EXTREME"),
("polarograph","blue polarograph","CK-187","the blue electrochemical instrument recording current as applied voltage changes","HARD"),
("polarograph","ceramic polarograph","CL-654","the pottery-cased analytical device producing a current-versus-voltage trace for an electrochemical sample","EXTREME"),
]
assert len(RAW)==64
assert len({x[0] for x in RAW})==32
assert len({x[1] for x in RAW})==64
assert len({x[2] for x in RAW})==64
fam=defaultdict(list)
for i,x in enumerate(RAW): fam[x[0]].append(i)
assert all(len(v)==2 for v in fam.values())

def source(obj,cid): return f"The {obj} is stored in container {cid}."
def query(desc): return f"Which container holds {desc}?"
MEM=[source(x[1],x[2]) for x in RAW];QUE=[query(x[3]) for x in RAW]

LOCK_OBJ={
"test":TEST,"seed":SEED,"model":MODEL_ID,"panel":RAW,
"primary":"VTOKEN_L23","secondary":["TQMEAN_L07","DELTA_L09"],
"pair_chance":0.5,"selection":"NONE","fusion":"NONE",
"foundation":{"482":TEST482_SHA,"488":TEST488_SHA,"489":TEST489_SHA,
"490":TEST490_SHA,"491":TEST491_SHA,"492":TEST492_SHA}}
LOCK=hashlib.sha256(json.dumps(LOCK_OBJ,sort_keys=True,separators=(",",":")).encode()).hexdigest()

print("="*158)
print("TEST493 — AKBASCORE MAM · FROZEN INSTANCE-IDENTITY REPLICATION")
print("FRESH HELD-OUT 32 CONCEPT PAIRS · PRIMARY VTOKEN L23 · NO SEARCH / NO FUSION / NO RETUNING")
print("="*158)
print("LOCK SHA:",LOCK);print("TEST492 LOCK:",TEST492_SHA)
print("PANEL: 64 fresh records / 32 unseen concepts / 2 instances each")
print("[1/8] Loading frozen Qwen...")
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,dtype=torch.bfloat16,attn_implementation="sdpa",device_map={"":0})
model.eval()
for p in model.parameters(): p.requires_grad_(False)
layers=model.model.layers;NL=len(layers);NQ=model.config.num_attention_heads;NKV=model.config.num_key_value_heads;HD=model.config.hidden_size//NQ
assert NL>23
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | {NL}L | H={model.config.hidden_size} | QH={NQ} KVH={NKV} | frozen BF16")

@torch.inference_mode()
def capture(s):
    ids=tok(s,return_tensors="pt",add_special_tokens=True).input_ids.to(DEVICE)
    o=model(input_ids=ids,output_hidden_states=True,use_cache=False,return_dict=True)
    H={}
    for L in [7,9,23]:
        H[L]=F.normalize(o.hidden_states[L+1][0].float(),dim=-1).cpu()
    z=layers[23].input_layernorm(o.hidden_states[23][0])
    v=layers[23].self_attn.v_proj(z).float().view(-1,NKV,HD)
    V23=F.normalize(v,dim=-1).cpu()
    return ids[0].cpu(),H,V23

print("[2/8] Capturing fresh held-out states...")
M=[];Q=[]
for i,s in enumerate(MEM):
    M.append(capture(s))
    if (i+1)%16==0: print(f"      memories {i+1:02d}/64")
for i,s in enumerate(QUE):
    Q.append(capture(s))
    if (i+1)%16==0: print(f"      queries  {i+1:02d}/64")

pair={}
for _,ix in fam.items():
    a,b=ix;pair[a]=b;pair[b]=a

def vtoken(i,j):
    q=Q[i][2];m=M[j][2];vals=[]
    for h in range(NKV): vals.append((q[:,h]@m[:,h].T).max(1).values.mean())
    return float(torch.stack(vals).mean())

def tqmean(i,j):
    q=Q[i][1][7];m=M[j][1][7]
    return float((q@m.T).max(1).values.mean())

def delta(i,j):
    q=Q[i][1][9];m=M[j][1][9]
    if len(q)<2 or len(m)<2:return 0.
    qd=F.normalize(q[1:]-q[:-1],dim=-1);md=F.normalize(m[1:]-m[:-1],dim=-1)
    return float((qd@md.T).max(1).values.mean())

FUN={"VTOKEN_L23":vtoken,"TQMEAN_L07":tqmean,"DELTA_L09":delta}

def pair_eval(fn):
    mar=[];correct=[]
    for i in range(64):
        j=pair[i];d=fn(i,i)-fn(i,j)
        mar.append(d);correct.append(d>0)
    a=np.array(mar);c=np.array(correct)
    return {"ACC":float(c.mean()),"N":64,"CORRECT":int(c.sum()),
            "MEAN_DELTA":float(a.mean()),"MED_DELTA":float(np.median(a)),
            "MARGINS":a,"CORRECT_MASK":c}

def binom_tail(k,n,p=.5):
    return sum(math.comb(n,x)*(p**x)*((1-p)**(n-x)) for x in range(k,n+1))

print("[3/8] PRIMARY frozen pair replication...")
RES={}
RES["VTOKEN_L23"]=pair_eval(vtoken)
r=RES["VTOKEN_L23"]
print(f"      VTOKEN_L23 PAIR-ACC={r['ACC']:.6f} ({r['CORRECT']}/64) MEANΔ={r['MEAN_DELTA']:+.6f} MEDΔ={r['MED_DELTA']:+.6f}")
print(f"      exact binomial p={binom_tail(r['CORRECT'],64):.10g}")

print("[4/8] PREDECLARED secondary replication...")
for n in ["TQMEAN_L07","DELTA_L09"]:
    RES[n]=pair_eval(FUN[n]);r=RES[n]
    print(f"      {n:12s} PAIR-ACC={r['ACC']:.6f} ({r['CORRECT']}/64) MEANΔ={r['MEAN_DELTA']:+.6f} MEDΔ={r['MED_DELTA']:+.6f} p={binom_tail(r['CORRECT'],64):.10g}")

print("[5/8] Global retrieval diagnostic — each frozen channel independently...")
GLOBAL={}
for n,fn in FUN.items():
    ranks=[]
    for i in range(64):
        s=np.array([fn(i,j) for j in range(64)])
        order=np.argsort(-s,kind="stable")
        ranks.append(int(np.where(order==i)[0][0])+1)
    a=np.array(ranks)
    GLOBAL[n]={"R1":float(np.mean(a<=1)),"R5":float(np.mean(a<=5)),
               "R16":float(np.mean(a<=16)),"MRR":float(np.mean(1/a)),
               "MEDR":float(np.median(a)),"MEANR":float(a.mean()),"RANKS":a}
    z=GLOBAL[n]
    print(f"      {n:12s} R1={z['R1']:.6f} R5={z['R5']:.6f} R16={z['R16']:.6f} MRR={z['MRR']:.6f} MEDR={z['MEDR']:.1f}")

print("[6/8] Family retrieval vs instance discrimination...")
for n,fn in FUN.items():
    fr=[];ir=[]
    for i in range(64):
        s=np.array([fn(i,j) for j in range(64)])
        order=np.argsort(-s,kind="stable")
        ir.append(int(np.where(order==i)[0][0])+1)
        members=set(fam[RAW[i][0]])
        fr.append(min(k+1 for k,j in enumerate(order) if int(j) in members))
    fr=np.array(fr);ir=np.array(ir)
    print(f"      {n:12s} FAMILY-R1={np.mean(fr<=1):.6f} FAMILY-R5={np.mean(fr<=5):.6f} INSTANCE-R1={np.mean(ir<=1):.6f} INSTANCE-R5={np.mean(ir<=5):.6f}")

print("[7/8] Difficulty strata + per-pair audit...")
for n in FUN:
    print(f"\n      {n}")
    r=RES[n]
    for st in ["STANDARD","HARD","EXTREME","ADVERSARIAL"]:
        ix=[i for i,x in enumerate(RAW) if x[4]==st]
        if not ix:continue
        a=r["MARGINS"][ix]
        print(f"      {st:11s} n={len(ix):02d} ACC={np.mean(a>0):.6f} MEANΔ={a.mean():+.6f}")
print("\n      PRIMARY VTOKEN_L23 AUDIT")
for i in range(64):
    j=pair[i];d=RES["VTOKEN_L23"]["MARGINS"][i]
    mark="OK  " if d>0 else "MISS"
    print(f"      [{i:02d}] {mark} {RAW[i][1]:24s} > {RAW[j][1]:24s} Δ={d:+.6f}")

print("[8/8] Fixed null + final seal...")
rng=np.random.default_rng(SEED)
NULL={}
for n in FUN:
    arr=RES[n]["MARGINS"];obs=RES[n]["ACC"];z=[]
    for _ in range(10000):
        sg=rng.choice([-1.,1.],size=64)
        z.append(float(np.mean(arr*sg>0)))
    z=np.array(z)
    NULL[n]={"MEAN":float(z.mean()),"P95":float(np.quantile(z,.95)),
             "P99":float(np.quantile(z,.99)),
             "EMP_P":float((1+np.sum(z>=obs))/(1+len(z)))}

# Predeclared primary decision only.
pr=RES["VTOKEN_L23"];pbin=binom_tail(pr["CORRECT"],64)
# 492 discovery primary was .875. Replication threshold intentionally modest:
# statistically above chance + >=.70 pair accuracy = strong practical replication.
if pr["ACC"]>=.70 and pbin<.01:
    VERDICT="FROZEN_NATIVE_INSTANCE_IDENTITY_REPLICATED"
elif pr["ACC"]>.50 and pbin<.05:
    VERDICT="NATIVE_INSTANCE_IDENTITY_SIGNAL_REPLICATED_WEAKLY"
else:
    VERDICT="FROZEN_INSTANCE_IDENTITY_HYPOTHESIS_NOT_REPLICATED"

print("\n"+"="*158)
print("TEST493 FINAL RESULT — FROZEN INSTANCE-IDENTITY REPLICATION")
print("="*158)
print("TEST482 BLOB SHA    :",TEST482_SHA)
print("TEST488 LOCK SHA    :",TEST488_SHA)
print("TEST489 LOCK SHA    :",TEST489_SHA)
print("TEST490 LOCK SHA    :",TEST490_SHA)
print("TEST491 LOCK SHA    :",TEST491_SHA)
print("TEST492 LOCK SHA    :",TEST492_SHA)
print("TEST493 LOCK SHA    :",LOCK)
print("FRESH PANEL         : 64 records / 32 unseen concepts / 2 instances")
print("PRIMARY             : VTOKEN_L23 — frozen before this panel")
print("SECONDARY           : TQMEAN_L07 / DELTA_L09 — frozen before this panel")
print("PAIR CHANCE         : 0.50")
print("-"*158)
for n in ["VTOKEN_L23","TQMEAN_L07","DELTA_L09"]:
    r=RES[n];g=GLOBAL[n];u=NULL[n]
    print(f"{n:20s}: PAIR={r['ACC']:.6f} ({r['CORRECT']}/64) Δ={r['MEAN_DELTA']:+.6f} BINOM-p={binom_tail(r['CORRECT'],64):.8g} | GLOBAL R1={g['R1']:.6f} R5={g['R5']:.6f} MRR={g['MRR']:.6f} | NULL-P99={u['P99']:.6f}")
print("-"*158)
print("PRIMARY VERDICT     :",VERDICT)
print("CHANNEL SEARCH      : NONE")
print("LAYER SEARCH        : NONE")
print("HEAD SEARCH         : NONE")
print("THRESHOLD SEARCH    : NONE")
print("WEIGHT FIT          : NONE")
print("FUSION              : NONE")
print("POST-HOC SELECTION  : NONE")
print("ROUTER / ANN / GRAPH: NONE")
print("CARTRIDGE READOUT   : NOT USED")
print("TRAINING / LoRA     : NONE")
print("MODEL CHANGE        : NONE")
print("NEXT                : only after this seal, decide whether the replicated identity signal should become part of MAM cartridge addressing.")
print("="*158)
