# TEST489 — AKBASCORE MAM · RECALL ATTRACTOR ERROR ANATOMY X-RAY
# Copyright © 2026 Mustafa Akbaş
# Exact TEST488 64-memory panel + exact frozen TEST487 11-channel address set.
# Diagnostic only: NO channel/weight search, NO new ensemble, NO router/ANN/training/LoRA/readout/model change.
import os,sys,subprocess,importlib.util,random,re,time,hashlib,json
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate"),("numpy","numpy")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
TEST="489";SEED=489;MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SEP="\n\n";DEVICE="cuda"
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required")
torch.set_grad_enabled(False);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

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
for i,(o,_,_,_) in enumerate(DATA):
    if o.casefold() in QUERIES[i].casefold():raise RuntimeError(f"Object leak {i}: {o}")

CHANNELS=["TOKENMAX_L10","TOKENMAX_L13","HMEAN_COS_L11","TOKENMAX_L14","TOKENMAX_L09",
          "DELTA_COS_L12","TOKENMAX_L11","HMEAN_COS_L08","TOKENMAX_L15","TOKENMAX_L12","DELTA_COS_L10"]
LOCK={"test":TEST,"foundation":"TEST488","test488_lock":"d2c2e744532e1fe9b4c5be745fce6892720745dd590b49eca59ca3ef753fdfd8",
      "channels":CHANNELS,"data":DATA,"diagnostic":"fixed-attractor-anatomy"}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()

print("="*154);print("TEST489 — AKBASCORE MAM · RECALL ATTRACTOR ERROR ANATOMY X-RAY")
print("EXACT TEST488 64 PANEL · FROZEN 11 CHANNELS · TRUE TARGET vs WRONG ATTRACTOR · NO SELECTION");print("="*154)
print("LOCK SHA:",LOCK_SHA);print("TEST488 LOCK: d2c2e744532e1fe9b4c5be745fce6892720745dd590b49eca59ca3ef753fdfd8")
print("FROZEN CHANNELS:",CHANNELS);T0=time.perf_counter()

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

@torch.inference_mode()
def probe(text):
    ids=[PAD]+enc(text+SEP);o=model(input_ids=torch.tensor([ids],device=DEVICE),output_hidden_states=True,use_cache=False,return_dict=True)
    return {l:o.hidden_states[l][0,1:].float().cpu() for l in range(8,16)}

print("[2/6] Capturing exact TEST488 memory/query states...")
B=[];Q=[]
for i,s in enumerate(SOURCES):
    B.append(probe(s))
    if (i+1)%16==0:print(f"      memories {i+1:02d}/64")
for i,q in enumerate(QUERIES):
    Q.append(probe(q))
    if (i+1)%16==0:print(f"      queries  {i+1:02d}/64")

def norm(x):return torch.nn.functional.normalize(x.float(),dim=-1)
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
def rankrow(row,target):
    return int((torch.argsort(row,descending=True)==target).nonzero(as_tuple=False)[0,0])+1
def pct_rank(M):
    return torch.stack([rankscore(M)[i] for i in range(N)])

print("[3/6] Reconstructing frozen 11-channel atlas...")
M={}
for l in [9,10,11,12,13,14,15]:
    M[f"TOKENMAX_L{l:02d}"]=tokenmax(l);print(f"      TOKENMAX L{l:02d}")
for l in [8,11]:
    M[f"HMEAN_COS_L{l:02d}"]=hmean(l);print(f"      HMEAN    L{l:02d}")
for l in [10,12]:
    M[f"DELTA_COS_L{l:02d}"]=delta(l);print(f"      DELTA    L{l:02d}")
RS={c:rankscore(M[c]) for c in CHANNELS}
CONS=torch.stack([RS[c] for c in CHANNELS]).mean(0)
CONS_R=[rankrow(CONS[i],i) for i in range(N)]

print("[4/6] True-target vs dominant wrong-attractor anatomy...")
ROWS=[];FAIL=[]
for i in range(N):
    order=torch.argsort(CONS[i],descending=True).tolist()
    wrong=next(j for j in order if j!=i)
    true_votes=wrong_votes=ties=0;true_top5=wrong_top5=0;detail=[]
    for c in CHANNELS:
        rt=rankrow(M[c][i],i);rw=rankrow(M[c][i],wrong)
        st=float(RS[c][i,i]);sw=float(RS[c][i,wrong])
        if st>sw:true_votes+=1
        elif sw>st:wrong_votes+=1
        else:ties+=1
        true_top5+=rt<=5;wrong_top5+=rw<=5
        detail.append((c,rt,rw,st-sw,float(M[c][i,i]),float(M[c][i,wrong])))
    margin=float(CONS[i,i]-CONS[i,wrong])
    wrong_rank=rankrow(CONS[i],wrong)
    row={"i":i,"gold_rank":CONS_R[i],"wrong":wrong,"wrong_rank":wrong_rank,"margin":margin,
         "true_votes":true_votes,"wrong_votes":wrong_votes,"ties":ties,
         "true_top5":true_top5,"wrong_top5":wrong_top5,"detail":detail}
    ROWS.append(row)
    if CONS_R[i]>1:FAIL.append(row)

def summary(rows):
    if not rows:return {}
    return {"N":len(rows),"GOLD_R1":float(np.mean([r["gold_rank"]==1 for r in rows])),
            "GOLD_R5":float(np.mean([r["gold_rank"]<=5 for r in rows])),
            "MED_GOLD_RANK":float(np.median([r["gold_rank"] for r in rows])),
            "MED_MARGIN":float(np.median([r["margin"] for r in rows])),
            "MEAN_TRUE_VOTES":float(np.mean([r["true_votes"] for r in rows])),
            "MEAN_WRONG_VOTES":float(np.mean([r["wrong_votes"] for r in rows]))}
print("      ALL :",summary(ROWS))
print("      NON-R1:",summary(FAIL))

print("\n[5/6] Failure X-ray + cross-channel persistence...")
for r in sorted(FAIL,key=lambda x:(-x["gold_rank"],x["margin"])):
    i=r["i"];j=r["wrong"];o,c,d,g=DATA[i];wo,wc,wd,wg=DATA[j]
    print("\n"+"-"*154)
    print(f"[{i:02d}] {g:11s} GOLD rank={r['gold_rank']:2d}  {o} -> {c}")
    print(f"     QUERY : {d}")
    print(f"     WRONG : [{j:02d}] {wo} -> {wc} | consensus-rank={r['wrong_rank']} | gold-wrong margin={r['margin']:+.4f}")
    print(f"     VOTES : GOLD={r['true_votes']} WRONG={r['wrong_votes']} TIE={r['ties']} | top5 GOLD={r['true_top5']}/11 WRONG={r['wrong_top5']}/11")
    print("     CHANNEL                     GOLD_R WRONG_R   ΔRANKSCORE      GOLD_RAW      WRONG_RAW")
    for ch,rt,rw,ds,sg,sw in r["detail"]:
        mark="G" if ds>0 else ("W" if ds<0 else "=")
        print(f"     {ch:27s} {rt:6d} {rw:7d} {ds:+12.5f} {sg:+13.6f} {sw:+13.6f}  {mark}")

# Persistence profile across frozen channels: how many channels rank candidate above gold.
PERSIST=np.zeros((N,N),dtype=int)
for i in range(N):
    for j in range(N):
        if j==i:continue
        PERSIST[i,j]=sum(float(RS[c][i,j])>float(RS[c][i,i]) for c in CHANNELS)
persistent=[]
for i in range(N):
    j=int(np.argmax(PERSIST[i]));persistent.append((int(PERSIST[i,j]),i,j,CONS_R[i]))
persistent.sort(reverse=True)
print("\n"+"="*154);print("TOP CROSS-CHANNEL WRONG ATTRACTORS")
for n,(p,i,j,gr) in enumerate(persistent[:20],1):
    print(f"{n:02d}. [{i:02d}] {DATA[i][0]:22s} gold-rank={gr:2d} <- [{j:02d}] {DATA[j][0]:22s} beats gold in {p:2d}/11 channels")

print("\n[6/6] Counterfactual diagnostic — channel leave-one-out (analysis only, NO selection)...")
BASE_R1=float(np.mean(np.asarray(CONS_R)<=1));BASE_R5=float(np.mean(np.asarray(CONS_R)<=5));BASE_MRR=float(np.mean(1/np.asarray(CONS_R,float)))
LOO=[]
for drop in CHANNELS:
    X=torch.stack([RS[c] for c in CHANNELS if c!=drop]).mean(0)
    rr=[rankrow(X[i],i) for i in range(N)]
    z={"drop":drop,"R1":float(np.mean(np.asarray(rr)<=1)),"R5":float(np.mean(np.asarray(rr)<=5)),"MRR":float(np.mean(1/np.asarray(rr,float)))}
    z["dR1"]=z["R1"]-BASE_R1;z["dMRR"]=z["MRR"]-BASE_MRR;LOO.append(z)
for z in sorted(LOO,key=lambda x:x["dMRR"],reverse=True):
    print(f"      DROP {z['drop']:27s} R1={z['R1']:.4f} ({z['dR1']:+.4f}) R5={z['R5']:.4f} MRR={z['MRR']:.4f} ({z['dMRR']:+.4f})")

# Failure types
TYPE={"DISTRIBUTED_WRONG_ATTRACTOR":0,"SPLIT_EVIDENCE":0,"NARROW_CHANNEL_CAPTURE":0}
for r in FAIL:
    if r["wrong_votes"]>=8:TYPE["DISTRIBUTED_WRONG_ATTRACTOR"]+=1
    elif r["wrong_votes"]>=5:TYPE["SPLIT_EVIDENCE"]+=1
    else:TYPE["NARROW_CHANNEL_CAPTURE"]+=1

# Hardest exact cases from TEST488 should reproduce.
CHECK={26:39,42:31,45:21,60:48}
REPRO={i:CONS_R[i] for i in CHECK}
REPRO_OK=all(REPRO[i]==CHECK[i] for i in CHECK)

print("\n"+"="*154);print("TEST489 FINAL RESULT — RECALL ATTRACTOR ERROR ANATOMY");print("="*154)
print("FOUNDATION          : TEST488 exact 64-memory held-out panel")
print("TEST482 BLOB SHA    : ecc630144a44479cb35c432c50eb0d09bc6866d7")
print("TEST488 LOCK SHA    : d2c2e744532e1fe9b4c5be745fce6892720745dd590b49eca59ca3ef753fdfd8")
print("TEST489 LOCK SHA    :",LOCK_SHA)
print("CHANNELS            : exact frozen TEST487 11")
print("BASE CONSENSUS      : R1=%.6f R5=%.6f MRR=%.6f"%(BASE_R1,BASE_R5,BASE_MRR))
print("NON-R1 CASES        :",len(FAIL),"/",N)
print("FAILURE TYPES       :",TYPE)
print("HARD-CASE REPRO     :",REPRO,"OK=",REPRO_OK)
print("-"*154)
print("MOST PERSISTENT WRONG ATTRACTORS:")
for p,i,j,gr in persistent[:10]:
    print(f"  [{i:02d}] {DATA[i][0]} <- [{j:02d}] {DATA[j][0]} | wrong beats gold {p}/11 | gold rank {gr}")
print("-"*154)
print("LEAVE-ONE-OUT       : DIAGNOSTIC ONLY; no channel is selected/removed by this test.")
print("POST-HOC OPTIMIZER  : NONE")
print("NEW ENSEMBLE        : NONE")
print("ORACLE ROUTING      : NONE")
print("ROUTER / ANN / GRAPH: NONE")
print("CARTRIDGE READOUT   : NOT USED")
print("TRAINING / LoRA     : NONE")
print("MODEL CHANGE        : NONE")
print("INTERPRETATION      : distinguishes distributed false attractors, split evidence, and narrow-channel capture.")
print("NEXT                : design TEST490 only from the observed failure mechanism; do not tune TEST489 itself.")
print(f"TOTAL TIME          : {time.perf_counter()-T0:.2f}s")
print("="*154)
