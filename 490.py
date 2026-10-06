# TEST490 — AKBASCORE MAM · SEMANTIC BASIN vs INSTANCE-IDENTITY X-RAY
# Copyright © 2026 Mustafa Akbaş
# Exact TEST488/489 64-memory panel. Frozen Qwen. Diagnostic only.
# Goal: determine whether shared concept evidence and discriminative instance evidence
# occupy different query tokens / memory tokens / layers.
# NO new ensemble, NO weight search, NO router/ANN/graph, NO training/LoRA/readout/model change.
import os,sys,subprocess,importlib.util,random,re,time,hashlib,json
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate"),("numpy","numpy")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
TEST="490";SEED=490;MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SEP="\n\n";DEVICE="cuda"
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

# Exact repeated-concept instance pairs present in TEST488.
PAIR_NAMES=["hygrometer","abacus","periscope","tuning fork","sundial","thermometer","gyroscope","atlas","magnifier",
            "stencil","stethoscope","odometer","anemometer","altimeter","protractor","prism","plumb bob","binoculars"]
def concept(o):
    lo=o.casefold()
    for x in sorted(PAIR_NAMES,key=len,reverse=True):
        if lo.endswith(x):return x
    return None
groups={}
for i,(o,_,_,_) in enumerate(DATA):
    c=concept(o)
    if c:groups.setdefault(c,[]).append(i)
PAIRS={k:v for k,v in groups.items() if len(v)>=2}
if not PAIRS:raise RuntimeError("No repeated-concept pairs")
LOCK={"test":TEST,"foundation":"TEST489","test489_lock":"ed03fb4b6f1afa410e1a28329d05b744889b8836f69f3572c4b52032b8b70fb0",
      "layers":list(range(8,16)),"data":DATA,"pairs":PAIRS,"diagnostic":"semantic-basin-vs-instance-identity"}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()

print("="*158);print("TEST490 — AKBASCORE MAM · SEMANTIC BASIN vs INSTANCE-IDENTITY X-RAY")
print("EXACT TEST488/489 PANEL · TOKEN×TOKEN L8-L15 · SAME-CONCEPT PAIRS · NO NEW ADDRESSER");print("="*158)
print("LOCK SHA:",LOCK_SHA)
print("TEST489 LOCK: ed03fb4b6f1afa410e1a28329d05b744889b8836f69f3572c4b52032b8b70fb0")
print("PAIRS:",{k:v for k,v in PAIRS.items()});T0=time.perf_counter()

print("[1/7] Loading frozen Qwen...")
tv=tuple(int("".join(c for c in x if c.isdigit())or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}).eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config
if (len(model.model.layers),cfg.hidden_size,cfg.num_attention_heads,cfg.num_key_value_heads)!=(28,3584,28,4):raise RuntimeError("Architecture mismatch")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
enc=lambda s:tok(s,add_special_tokens=False).input_ids
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | 28L | frozen BF16")

@torch.inference_mode()
def probe(text):
    ids=[PAD]+enc(text+SEP);x=torch.tensor([ids],device=DEVICE)
    o=model(input_ids=x,output_hidden_states=True,use_cache=False,return_dict=True)
    tids=ids[1:]
    return {"ids":tids,"tokens":[tok.decode([z]) for z in tids],
            "h":{l:o.hidden_states[l][0,1:].float().cpu() for l in range(8,16)}}

print("[2/7] Capturing exact 64 memory/query token states...")
B=[];Q=[]
for i,s in enumerate(SOURCES):
    B.append(probe(s))
    if (i+1)%16==0:print(f"      memories {i+1:02d}/64")
for i,q in enumerate(QUERIES):
    Q.append(probe(q))
    if (i+1)%16==0:print(f"      queries  {i+1:02d}/64")

def norm(x):return torch.nn.functional.normalize(x.float(),dim=-1)
def sim(i,j,l):return norm(Q[i]["h"][l])@norm(B[j]["h"][l]).T
def tmax_vec(i,j,l):return sim(i,j,l).max(dim=1).values
def tmax_score(i,j,l):return float(tmax_vec(i,j,l).mean())
def clean(t):return t.replace("\n","\\n")
def top_match(i,j,l,qi):
    S=sim(i,j,l);v,k=torch.max(S[qi],dim=0)
    return float(v),int(k),B[j]["tokens"][int(k)]
def all_rank(i,l):
    s=np.array([tmax_score(i,j,l) for j in range(N)])
    o=np.argsort(-s,kind="stable");r=np.empty(N,int);r[o]=np.arange(1,N+1)
    return s,r

print("[3/7] Measuring semantic-basin capture across repeated concepts...")
BASIN=[]
for cname,idx in PAIRS.items():
    for i in idx:
        for l in range(8,16):
            scores,ranks=all_rank(i,l)
            family=idx
            best_family=min(ranks[j] for j in family)
            own=ranks[i]
            other=[j for j in family if j!=i]
            best_other=min([ranks[j] for j in other]) if other else 999
            BASIN.append((cname,i,l,own,best_family,best_other))
for l in range(8,16):
    z=[x for x in BASIN if x[2]==l]
    print(f"      L{l:02d} FAMILY-R1={np.mean([x[4]==1 for x in z]):.3f} FAMILY-R5={np.mean([x[4]<=5 for x in z]):.3f} "
          f"INSTANCE-R1={np.mean([x[3]==1 for x in z]):.3f} INSTANCE-R5={np.mean([x[3]<=5 for x in z]):.3f}")

print("[4/7] Token-level shared-concept vs discriminative-residual anatomy...")
REC=[]
for cname,idx in PAIRS.items():
    if len(idx)!=2:continue
    a,b=idx
    for i,j in [(a,b),(b,a)]:
        for l in range(8,16):
            vg=tmax_vec(i,i,l);vd=tmax_vec(i,j,l);res=vg-vd
            # semantic/shared evidence = strong match to both same-concept instances
            shared=torch.minimum(vg,vd)
            # identity evidence = positive advantage for own instance
            best_shared=int(torch.argmax(shared));best_res=int(torch.argmax(res))
            pos=(res>0).float()
            REC.append({"concept":cname,"i":i,"j":j,"l":l,
                        "gold":float(vg.mean()),"decoy":float(vd.mean()),"margin":float(res.mean()),
                        "posfrac":float(pos.mean()),"maxres":float(res[best_res]),
                        "shared":float(shared[best_shared]),"shared_q":best_shared,"res_q":best_res})

for l in range(8,16):
    z=[r for r in REC if r["l"]==l]
    print(f"      L{l:02d} mean-own={np.mean([r['gold'] for r in z]):+.5f} mean-pair={np.mean([r['decoy'] for r in z]):+.5f} "
          f"identity-margin={np.mean([r['margin'] for r in z]):+.5f} positive-token-frac={np.mean([r['posfrac'] for r in z]):.3f} "
          f"max-residual={np.mean([r['maxres'] for r in z]):+.5f}")

print("[5/7] Same-concept pair X-ray...")
for cname,idx in PAIRS.items():
    if len(idx)!=2:continue
    a,b=idx
    print("\n"+"-"*158);print(f"CONCEPT: {cname.upper()} | [{a:02d}] {DATA[a][0]} <-> [{b:02d}] {DATA[b][0]}")
    for i,j in [(a,b),(b,a)]:
        print(f"\n  QUERY [{i:02d}] {DATA[i][0]} ({DATA[i][3]})")
        print(f"  {DATA[i][2]}")
        best=None
        for l in range(8,16):
            vg=tmax_vec(i,i,l);vd=tmax_vec(i,j,l);res=vg-vd
            m=float(res.mean());qi=int(torch.argmax(res));qshared=int(torch.argmax(torch.minimum(vg,vd)))
            gv,gk,gt=top_match(i,i,l,qi);dv,dk,dt=top_match(i,j,l,qi)
            qtok=Q[i]["tokens"][qi]
            sv1,sk1,st1=top_match(i,i,l,qshared);sv2,sk2,st2=top_match(i,j,l,qshared)
            print(f"    L{l:02d} own={float(vg.mean()):+.5f} pair={float(vd.mean()):+.5f} Δ={m:+.5f} "
                  f"| ID q='{clean(qtok)}' Δmax={float(res[qi]):+.5f} -> own:'{clean(gt)}' pair:'{clean(dt)}' "
                  f"| SHARED q='{clean(Q[i]['tokens'][qshared])}' -> own:'{clean(st1)}' pair:'{clean(st2)}'")
            if best is None or m>best[0]:best=(m,l,qi,gv,dv,gt,dt)
        print(f"  BEST IDENTITY LAYER: L{best[1]:02d} mean-margin={best[0]:+.5f} token='{clean(Q[i]['tokens'][best[2]])}' "
              f"own-match='{clean(best[5])}' pair-match='{clean(best[6])}'")

print("[6/7] General wrong-attractor residual test...")
# TEST489 exact frozen consensus reconstruction only to identify its wrong attractor.
def tokenmax_matrix(l):
    M=torch.empty(N,N)
    for i in range(N):
        for j in range(N):M[i,j]=tmax_score(i,j,l)
    return M
def hmean_matrix(l):
    A=norm(torch.stack([Q[i]["h"][l].mean(0) for i in range(N)]))
    C=norm(torch.stack([B[i]["h"][l].mean(0) for i in range(N)]))
    return A@C.T
def delta_matrix(l):
    A=norm(torch.stack([Q[i]["h"][l].mean(0)-Q[i]["h"][l-1].mean(0) for i in range(N)]))
    C=norm(torch.stack([B[i]["h"][l].mean(0)-B[i]["h"][l-1].mean(0) for i in range(N)]))
    return A@C.T
def rankscore(M):
    R=torch.empty_like(M)
    for i in range(N):
        o=torch.argsort(M[i],descending=True);r=torch.empty(N);r[o]=torch.arange(1,N+1,dtype=torch.float32)
        R[i]=1-(r-1)/(N-1)
    return R
CHANNELS=["TOKENMAX_L10","TOKENMAX_L13","HMEAN_COS_L11","TOKENMAX_L14","TOKENMAX_L09",
          "DELTA_COS_L12","TOKENMAX_L11","HMEAN_COS_L08","TOKENMAX_L15","TOKENMAX_L12","DELTA_COS_L10"]
M={}
for l in [9,10,11,12,13,14,15]:M[f"TOKENMAX_L{l:02d}"]=tokenmax_matrix(l)
for l in [8,11]:M[f"HMEAN_COS_L{l:02d}"]=hmean_matrix(l)
for l in [10,12]:M[f"DELTA_COS_L{l:02d}"]=delta_matrix(l)
CONS=torch.stack([rankscore(M[c]) for c in CHANNELS]).mean(0)
GR=[int((torch.argsort(CONS[i],descending=True)==i).nonzero()[0,0])+1 for i in range(N)]
FAIL=[i for i in range(N) if GR[i]>1]
GEN=[]
for i in FAIL:
    order=torch.argsort(CONS[i],descending=True).tolist();j=next(x for x in order if x!=i)
    best=(-1e9,None,None,None)
    for l in range(8,16):
        r=tmax_vec(i,i,l)-tmax_vec(i,j,l);m=float(r.mean());mx=float(r.max())
        if mx>best[0]:best=(mx,l,int(torch.argmax(r)),m)
    GEN.append((i,j,GR[i],best[0],best[1],best[2],best[3]))
print("      NON-R1 cases:",len(GEN))
print("      cases with at least one positive token residual:",sum(x[3]>0 for x in GEN),"/",len(GEN))
print("      cases with max residual > .01:",sum(x[3]>.01 for x in GEN),"/",len(GEN))
print("      cases with max residual > .02:",sum(x[3]>.02 for x in GEN),"/",len(GEN))
print("\n      FAILURE RESIDUALS")
for i,j,gr,mx,l,qi,meanm in sorted(GEN,key=lambda x:-x[2]):
    print(f"      [{i:02d}] rank={gr:2d} {DATA[i][0]:22s} <- {DATA[j][0]:22s} | "
          f"best L{l:02d} token='{clean(Q[i]['tokens'][qi])}' maxΔ={mx:+.5f} meanΔ={meanm:+.5f}")

print("[7/7] Aggregate decomposition...")
PAIRREC=[r for r in REC]
best_layer=max(range(8,16),key=lambda l:np.mean([r["margin"] for r in PAIRREC if r["l"]==l]))
best_res_layer=max(range(8,16),key=lambda l:np.mean([r["maxres"] for r in PAIRREC if r["l"]==l]))
PAIR_INSTANCE_POS=float(np.mean([r["margin"]>0 for r in PAIRREC]))
TOKEN_RES_POS=float(np.mean([r["maxres"]>0 for r in PAIRREC]))
TOKEN_RES_01=float(np.mean([r["maxres"]>.01 for r in PAIRREC]))
# semantic basin vs exact instance at L10, TEST487 strongest single channel
z10=[x for x in BASIN if x[2]==10]
FAM_R1=float(np.mean([x[4]==1 for x in z10]));FAM_R5=float(np.mean([x[4]<=5 for x in z10]))
INS_R1=float(np.mean([x[3]==1 for x in z10]));INS_R5=float(np.mean([x[3]<=5 for x in z10]))
if FAM_R1>INS_R1+.15 and TOKEN_RES_POS>=.75:
    VERDICT="SEMANTIC_BASIN_STRONGER_THAN_INSTANCE_IDENTITY_WITH_RECOVERABLE_TOKEN_RESIDUAL"
elif FAM_R1>INS_R1 and TOKEN_RES_POS>=.60:
    VERDICT="SEMANTIC_BASIN_INSTANCE_SPLIT_DETECTED"
elif TOKEN_RES_POS>=.60:
    VERDICT="TOKEN_LOCAL_IDENTITY_RESIDUAL_EXISTS_WITHOUT_CLEAR_BASIN_SPLIT"
else:VERDICT="NO_CLEAR_INSTANCE_IDENTITY_RESIDUAL"

print("\n"+"="*158);print("TEST490 FINAL RESULT — SEMANTIC BASIN vs INSTANCE IDENTITY");print("="*158)
print("FOUNDATION          : TEST489 exact error anatomy")
print("TEST482 BLOB SHA    : ecc630144a44479cb35c432c50eb0d09bc6866d7")
print("TEST488 LOCK SHA    : d2c2e744532e1fe9b4c5be745fce6892720745dd590b49eca59ca3ef753fdfd8")
print("TEST489 LOCK SHA    : ed03fb4b6f1afa410e1a28329d05b744889b8836f69f3572c4b52032b8b70fb0")
print("TEST490 LOCK SHA    :",LOCK_SHA)
print("REPEATED CONCEPTS   :",len(PAIRS))
print("TOKEN XRAY LAYERS   : L08-L15")
print("-"*158)
print(f"L10 FAMILY BASIN    : R1={FAM_R1:.6f} R5={FAM_R5:.6f}")
print(f"L10 EXACT INSTANCE  : R1={INS_R1:.6f} R5={INS_R5:.6f}")
print("PAIR MEAN-ID POS    :",PAIR_INSTANCE_POS)
print("PAIR TOKEN-ID POS   :",TOKEN_RES_POS)
print("PAIR TOKEN-ID >.01  :",TOKEN_RES_01)
print("BEST MEAN-ID LAYER  : L%02d"%best_layer)
print("BEST TOKEN-ID LAYER : L%02d"%best_res_layer)
print("489 NON-R1          :",len(GEN))
print("FAIL TOKEN RES >0   :",sum(x[3]>0 for x in GEN),"/",len(GEN))
print("FAIL TOKEN RES >.01 :",sum(x[3]>.01 for x in GEN),"/",len(GEN))
print("FAIL TOKEN RES >.02 :",sum(x[3]>.02 for x in GEN),"/",len(GEN))
print("-"*158)
print("VERDICT             :",VERDICT)
print("INTERPRETATION      : tests whether concept-family attraction and instance-specific evidence separate naturally in frozen token geometry.")
print("NEW ADDRESSER       : NONE")
print("WEIGHT / CHANNEL FIT: NONE")
print("POST-HOC SELECTION  : NONE")
print("ROUTER / ANN / GRAPH: NONE")
print("CARTRIDGE READOUT   : NOT USED")
print("TRAINING / LoRA     : NONE")
print("MODEL CHANGE        : NONE")
print("NEXT                : only if split is observed, freeze a basin+residual rule and validate it on fresh unseen concept-pairs.")
print(f"TOTAL TIME          : {time.perf_counter()-T0:.2f}s")
print("="*158)
