# Copyright © 2026 Mustafa Akbaş
# New original AKBASCORE MAM material is source-available under the repository LICENSE.
# Earlier MIT-licensed Titan Cognitive Core / AkbasCore material remains under its original license.
# See the repository LICENSE for complete terms.
#
# TEST492 — AKBASCORE MAM · INSTANCE-IDENTITY NATIVE SIGNAL ATLAS
# EXACT TEST491 PANEL · SAME-FAMILY PAIR DISCRIMINATION · PASSIVE X-RAY ONLY
# No new addresser / no fusion / no router / no ANN / no graph / no training / no LoRA / no cartridge readout.
# TEST491 panel is now DISCOVERY DATA for identity-signal anatomy. Any discovered rule MUST be frozen and validated on a fresh TEST493 panel.

import os,sys,re,json,time,random,hashlib,subprocess,importlib.util
os.environ["TOKENIZERS_PARALLELISM"]="false"
def need(p):
    if importlib.util.find_spec(p) is None: subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
for p in ["torch","transformers","accelerate"]: need(p)
import numpy as np,torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from collections import defaultdict

TEST="492";SEED=492;MODEL_ID="Qwen/Qwen2.5-7B-Instruct";DEVICE="cuda"
TEST482_SHA="ecc630144a44479cb35c432c50eb0d09bc6866d7"
TEST488_SHA="d2c2e744532e1fe9b4c5be745fce6892720745dd590b49eca59ca3ef753fdfd8"
TEST489_SHA="ed03fb4b6f1afa410e1a28329d05b744889b8836f69f3572c4b52032b8b70fb0"
TEST490_SHA="237f3c7a9cc2478b2ca55944f0a356101c8f87c5a7ef250f6267fc04a9fc1dd2"
TEST491_SHA="582fdc3e39d58863685be094c1bbfcfac8e398845d90624598197c97be9632bd"
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)

# EXACT TEST491 PANEL
RAW=[
("microscope","ebony microscope","AK-731","dark-framed optical instrument used to inspect structures too small for unaided vision","STANDARD"),
("microscope","ivory microscope","PV-284","the pale laboratory viewing device chosen when cells or tiny structures must be enlarged optically","EXTREME"),
("sphygmomanometer","navy sphygmomanometer","CX-615","blue clinical instrument used to determine pressure inside a person's arteries","HARD"),
("sphygmomanometer","linen sphygmomanometer","JR-902","the fabric-covered medical apparatus wrapped around an arm when a clinician wants systolic and diastolic numbers","EXTREME"),
("geiger counter","yellow geiger counter","LM-347","yellow handheld detector used to reveal ionizing radiation","STANDARD"),
("geiger counter","steel geiger counter","QT-861","the metallic device that clicks more urgently when invisible radioactive emissions become stronger","EXTREME"),
("oscilloscope","black oscilloscope","WG-526","dark electronic instrument displaying changing electrical signals as visible waveforms","HARD"),
("oscilloscope","silver oscilloscope","BN-193","the metallic bench device used when voltage changes over time must be seen as a trace rather than merely measured once","EXTREME"),
("tachometer","red tachometer","FH-708","red instrument indicating how quickly a rotating shaft or engine is turning","STANDARD"),
("tachometer","brass tachometer","UZ-452","the yellow-metal gauge consulted for revolutions per minute rather than vehicle road speed","ADVERSARIAL"),
("calorimeter","glass calorimeter","DP-819","transparent laboratory apparatus used to measure heat released or absorbed during a process","HARD"),
("calorimeter","copper calorimeter","KS-264","the reddish-metal experimental vessel used when energy transfer as heat must be quantified","EXTREME"),
("densitometer","white densitometer","RY-573","pale measuring instrument used to determine optical density or darkness of a material","HARD"),
("densitometer","ceramic densitometer","EA-940","the pottery-cased device concerned with how strongly a sample blocks or absorbs light rather than its physical thickness","EXTREME"),
("spectrometer","violet spectrometer","HM-381","purple scientific instrument separating and measuring components of a spectrum","STANDARD"),
("spectrometer","aluminum spectrometer","XC-726","the metallic laboratory device used when light must be decomposed by wavelength and quantified","ADVERSARIAL"),
("theodolite","green theodolite","NJ-658","green surveying instrument used to measure horizontal and vertical angles","STANDARD"),
("theodolite","bronze theodolite","TP-405","the brown-metal field instrument a surveyor sights through when precise angular relationships between distant points are needed","EXTREME"),
("hydrometer","glass hydrometer","SV-917","transparent floating instrument used to estimate the density of a liquid","HARD"),
("hydrometer","wooden hydrometer","GL-236","the wood-bodied floating gauge whose depth in a fluid reveals something about that fluid's relative density","EXTREME"),
("galvanometer","cobalt galvanometer","ME-564","blue instrument that detects and measures small electric current","STANDARD"),
("galvanometer","brass galvanometer","HA-823","the yellow-metal electrical indicator whose needle responds when a weak current passes through it","EXTREME"),
("dynamometer","iron dynamometer","ZF-190","heavy metal instrument used to measure force or mechanical pull","HARD"),
("dynamometer","crystal dynamometer","VC-647","the transparent-bodied device chosen when the magnitude of a push or pull must become a number","EXTREME"),
("refractometer","amber refractometer","KU-315","amber optical instrument measuring how strongly a substance bends light","STANDARD"),
("refractometer","steel refractometer","OD-782","the metallic optical tester used when concentration is inferred from the change in direction of light entering a sample","ADVERSARIAL"),
("manometer","copper manometer","IL-439","copper pressure instrument comparing fluid pressure using a column or equivalent mechanism","HARD"),
("manometer","quartz manometer","YE-806","the crystal-bodied gauge concerned with pressure difference in a gas or liquid rather than atmospheric weather forecasting","EXTREME"),
("pedometer","black pedometer","AR-251","dark portable counter recording the number of steps a person takes","STANDARD"),
("pedometer","jade pedometer","MS-694","the green wearable device that accumulates walking steps rather than measuring the distance travelled by a vehicle","ADVERSARIAL"),
("rangefinder","silver rangefinder","PC-538","silver optical instrument used to determine distance to a remote target","STANDARD"),
("rangefinder","mahogany rangefinder","TV-972","the dark-wood viewing device used when someone needs the distance to a far object without physically reaching it","EXTREME"),
("clinometer","ivory clinometer","DB-416","pale instrument used to measure slope or angle of elevation","HARD"),
("clinometer","azure clinometer","QF-759","the blue device aimed along an incline when steepness or elevation angle must be quantified","EXTREME"),
("lux meter","white lux meter","GX-208","white instrument measuring the amount of visible illumination falling on a surface","STANDARD"),
("lux meter","onyx lux meter","NW-635","the black device used when room brightness must become a numerical illumination reading rather than a visual impression","EXTREME"),
("sound level meter","yellow sound level meter","BH-841","yellow handheld instrument measuring acoustic intensity in decibels","STANDARD"),
("sound level meter","steel sound level meter","CR-397","the metallic handheld device used when environmental noise must be expressed numerically rather than judged by ear","EXTREME"),
("torque wrench","chrome torque wrench","VK-620","shiny hand tool used to tighten a fastener to a specified twisting force","HARD"),
("torque wrench","red torque wrench","LP-153","the red workshop tool chosen when a bolt must not merely be tight but must receive a controlled rotational force","EXTREME"),
("spirit level","green spirit level","JN-486","green construction tool showing whether a surface is horizontal or vertical","STANDARD"),
("spirit level","aluminum spirit level","UF-925","the metallic builder's tool whose trapped bubble reveals whether a shelf or wall is truly level","EXTREME"),
("multimeter","orange multimeter","SE-374","orange electrical meter able to measure quantities such as voltage resistance and current","STANDARD"),
("multimeter","black multimeter","KM-718","the dark handheld electrical tester selected when several circuit quantities may need checking with one device","EXTREME"),
("compass divider","brass compass divider","RD-562","yellow-metal two-legged drafting tool used to transfer distances or draw circles","HARD"),
("compass divider","steel compass divider","AY-839","the metallic hinged drawing instrument whose two pointed legs can preserve a span between locations on paper","EXTREME"),
("micrometer","blue micrometer","HT-207","blue precision tool used to measure very small thicknesses or diameters","STANDARD"),
("micrometer","bronze micrometer","QE-671","the brown-metal measuring device chosen when an ordinary ruler or caliper is not precise enough for a tiny dimension","ADVERSARIAL"),
("pipette","glass pipette","MU-348","transparent laboratory tube used to transfer a carefully measured small volume of liquid","STANDARD"),
("pipette","red pipette","FS-795","the red laboratory implement used when a tiny controlled amount of fluid must be moved from one vessel to another","EXTREME"),
("centrifuge","white centrifuge","CL-924","pale laboratory machine that spins samples rapidly to separate components by density","HARD"),
("centrifuge","steel centrifuge","ZG-461","the metallic machine into which tubes are placed when rapid rotation should force different sample components apart","EXTREME"),
("burette","clear burette","WP-683","transparent graduated laboratory tube delivering controlled liquid volumes during titration","STANDARD"),
("burette","amber burette","NK-150","the amber-colored vertical laboratory tube used when reagent must be released gradually while its delivered volume is tracked","EXTREME"),
("sextant","brass sextant","EY-537","yellow-metal navigation instrument measuring the angle between a celestial body and the horizon","HARD"),
("sextant","black sextant","IA-864","the dark navigation device a sailor can use with the horizon and a star or the sun to infer position","EXTREME"),
("polarimeter","silver polarimeter","OT-296","silver optical instrument measuring rotation of polarized light through a sample","HARD"),
("polarimeter","wooden polarimeter","PH-743","the wood-cased optical device used when a substance's effect on the orientation of polarized light must be quantified","EXTREME"),
("viscometer","glass viscometer","DV-519","transparent instrument used to measure a fluid's resistance to flowing","STANDARD"),
("viscometer","copper viscometer","SB-872","the reddish device used when one liquid pours reluctantly and another freely and that difference must become a number","EXTREME"),
("dosimeter","white dosimeter","YC-430","pale device recording accumulated exposure to ionizing radiation","HARD"),
("dosimeter","black dosimeter","RJ-965","the dark personal monitor worn when the important question is total radiation received over time rather than radiation at one instant","ADVERSARIAL"),
("salinometer","blue salinometer","KA-187","blue instrument used to determine salt concentration in a liquid","STANDARD"),
("salinometer","ceramic salinometer","XE-654","the pottery-cased tester used when the amount of dissolved salt in water must be estimated rather than its temperature","EXTREME"),
]
assert len(RAW)==64 and len({x[1] for x in RAW})==64 and len({x[2] for x in RAW})==64
fam=defaultdict(list)
for i,x in enumerate(RAW): fam[x[0]].append(i)
assert len(fam)==32 and all(len(v)==2 for v in fam.values())

def source(obj,cid): return f"The {obj} is stored in container {cid}."
def query(desc): return f"Which container holds {desc}?"
MEM=[source(x[1],x[2]) for x in RAW];QUE=[query(x[3]) for x in RAW]
LOCK_OBJ={"test":TEST,"seed":SEED,"model":MODEL_ID,"panel":RAW,"layers":"0-27","purpose":"passive same-family instance identity signal atlas","foundation":{"482":TEST482_SHA,"488":TEST488_SHA,"489":TEST489_SHA,"490":TEST490_SHA,"491":TEST491_SHA}}
LOCK=hashlib.sha256(json.dumps(LOCK_OBJ,sort_keys=True,separators=(",",":")).encode()).hexdigest()

print("="*158)
print("TEST492 — AKBASCORE MAM · INSTANCE-IDENTITY NATIVE SIGNAL ATLAS")
print("EXACT TEST491 PANEL · SAME-FAMILY PAIR DISCRIMINATION · PASSIVE X-RAY ONLY")
print("="*158)
print("LOCK SHA:",LOCK);print("TEST491 LOCK:",TEST491_SHA)
print("PANEL: 64 records / 32 concept-pairs / 28 layers / no new addresser")
print("[1/8] Loading frozen Qwen...")
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,dtype=torch.bfloat16,attn_implementation="sdpa",device_map={"":0})
model.eval()
for p in model.parameters(): p.requires_grad_(False)
layers=model.model.layers;NL=len(layers);NQ=model.config.num_attention_heads;NKV=model.config.num_key_value_heads;HD=model.config.hidden_size//NQ
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | {NL}L | H={model.config.hidden_size} | QH={NQ} KVH={NKV} | frozen BF16")

@torch.inference_mode()
def capture(s):
    ids=tok(s,return_tensors="pt",add_special_tokens=True).input_ids.to(DEVICE)
    o=model(input_ids=ids,output_hidden_states=True,use_cache=False,return_dict=True)
    H=[];K=[];V=[]
    for L in range(NL):
        h=o.hidden_states[L+1][0].float()
        H.append(F.normalize(h,dim=-1).cpu())
        z=layers[L].input_layernorm(o.hidden_states[L][0])
        k=layers[L].self_attn.k_proj(z).float().view(-1,NKV,HD)
        v=layers[L].self_attn.v_proj(z).float().view(-1,NKV,HD)
        K.append(F.normalize(k,dim=-1).cpu());V.append(F.normalize(v,dim=-1).cpu())
    return ids[0].cpu(),H,K,V

print("[2/8] Capturing exact TEST491 states + native K/V...")
M=[];Q=[]
for i,s in enumerate(MEM):
    M.append(capture(s))
    if (i+1)%16==0: print(f"      memories {i+1:02d}/64")
for i,s in enumerate(QUE):
    Q.append(capture(s))
    if (i+1)%16==0: print(f"      queries  {i+1:02d}/64")

def hmean(q,m): return float(F.cosine_similarity(q.mean(0),m.mean(0),dim=0))
def lastcos(q,m): return float(F.cosine_similarity(q[-1],m[-1],dim=0))
def token_qmean(q,m):
    a=q@m.T
    return float(a.max(1).values.mean())
def token_max(q,m):
    return float((q@m.T).max())
def token_top3(q,m):
    a=(q@m.T).max(1).values
    k=min(3,len(a))
    return float(torch.topk(a,k).values.mean())
def token_margin(q,a,b):
    sa=(q@a.T).max(1).values;sb=(q@b.T).max(1).values
    d=sa-sb
    return float(d.max()),float(d.mean()),int(d.argmax())
def kvmean(q,m):
    # q,m [T,H,D] normalized; compare head-wise token means
    qa=F.normalize(q.mean(0),dim=-1);ma=F.normalize(m.mean(0),dim=-1)
    return float((qa*ma).sum(-1).mean())
def kvtoken(q,m):
    # per KV head: each query token finds strongest memory token; average all
    vals=[]
    for h in range(NKV):
        vals.append((q[:,h]@m[:,h].T).max(1).values.mean())
    return float(torch.stack(vals).mean())
def delta_cos(q,m):
    if len(q)<2 or len(m)<2:return 0.
    qd=F.normalize(q[1:]-q[:-1],dim=-1);md=F.normalize(m[1:]-m[:-1],dim=-1)
    return float((qd@md.T).max(1).values.mean())

# Every metric is evaluated ONLY as true-instance vs its known same-family sibling.
# This is anatomy, not global retrieval.
CHANNELS={}
print("[3/8] Scanning passive identity channels across all 28 layers...")
for L in range(NL):
    CHANNELS[f"HMEAN_L{L:02d}"]=lambda i,j,L=L:hmean(Q[i][1][L],M[j][1][L])
    CHANNELS[f"LAST_L{L:02d}"]=lambda i,j,L=L:lastcos(Q[i][1][L],M[j][1][L])
    CHANNELS[f"TQMEAN_L{L:02d}"]=lambda i,j,L=L:token_qmean(Q[i][1][L],M[j][1][L])
    CHANNELS[f"TMAX_L{L:02d}"]=lambda i,j,L=L:token_max(Q[i][1][L],M[j][1][L])
    CHANNELS[f"TTOP3_L{L:02d}"]=lambda i,j,L=L:token_top3(Q[i][1][L],M[j][1][L])
    CHANNELS[f"DELTA_L{L:02d}"]=lambda i,j,L=L:delta_cos(Q[i][1][L],M[j][1][L])
    CHANNELS[f"KMEAN_L{L:02d}"]=lambda i,j,L=L:kvmean(Q[i][2][L],M[j][2][L])
    CHANNELS[f"KTOKEN_L{L:02d}"]=lambda i,j,L=L:kvtoken(Q[i][2][L],M[j][2][L])
    CHANNELS[f"VMEAN_L{L:02d}"]=lambda i,j,L=L:kvmean(Q[i][3][L],M[j][3][L])
    CHANNELS[f"VTOKEN_L{L:02d}"]=lambda i,j,L=L:kvtoken(Q[i][3][L],M[j][3][L])

pair_of={}
for _,ix in fam.items():
    a,b=ix;pair_of[a]=b;pair_of[b]=a

RES={}
for n,fn in CHANNELS.items():
    margins=[]
    for i in range(64):
        j=pair_of[i]
        margins.append(fn(i,i)-fn(i,j))
    a=np.array(margins)
    RES[n]={"acc":float(np.mean(a>0)),"tie":float(np.mean(a==0)),"margin":float(a.mean()),"median":float(np.median(a)),"margins":a}
print("      channels:",len(RES))

print("[4/8] Layer-family atlas...")
families=["HMEAN","LAST","TQMEAN","TMAX","TTOP3","DELTA","KMEAN","KTOKEN","VMEAN","VTOKEN"]
for f in families:
    z=[(n,r) for n,r in RES.items() if n.startswith(f+"_")]
    z.sort(key=lambda x:(x[1]["acc"],x[1]["margin"]),reverse=True)
    n,r=z[0]
    print(f"      {f:7s} BEST={n:14s} PAIR-ACC={r['acc']:.6f} MEANΔ={r['margin']:+.6f} MEDΔ={r['median']:+.6f}")

print("[5/8] Global top native identity signals...")
TOP=sorted(RES.items(),key=lambda x:(x[1]["acc"],x[1]["margin"]),reverse=True)
for k,(n,r) in enumerate(TOP[:30],1):
    print(f"      {k:02d}. {n:14s} ACC={r['acc']:.6f} Δ={r['margin']:+.6f} MED={r['median']:+.6f}")

print("[6/8] Head-local K/V identity atlas...")
HEAD=[]
for typ,slot in [("K",2),("V",3)]:
    for L in range(NL):
        for h in range(NKV):
            mar=[]
            for i in range(64):
                j=pair_of[i]
                q=Q[i][slot][L][:,h];a=M[i][slot][L][:,h];b=M[j][slot][L][:,h]
                sa=float((q@a.T).max(1).values.mean());sb=float((q@b.T).max(1).values.mean())
                mar.append(sa-sb)
            ar=np.array(mar)
            HEAD.append((f"{typ}TOKEN_L{L:02d}_H{h}",float(np.mean(ar>0)),float(ar.mean()),float(np.median(ar)),ar))
HEAD.sort(key=lambda x:(x[1],x[2]),reverse=True)
for k,x in enumerate(HEAD[:24],1): print(f"      {k:02d}. {x[0]:16s} ACC={x[1]:.6f} Δ={x[2]:+.6f} MED={x[3]:+.6f}")

print("[7/8] Complementarity / failure anatomy...")
# Diagnostic oracle only: asks whether ANY passive channel sees true > sibling.
# It is explicitly NOT a deployable rule.
CAND=TOP[:40]+[(x[0],{"acc":x[1],"margin":x[2],"median":x[3],"margins":x[4]}) for x in HEAD[:24]]
seen=np.zeros(64,dtype=int);votes=np.zeros(64,dtype=int)
for n,r in CAND:
    a=r["margins"];seen|=(a>0);votes+=(a>0)
print(f"      TOP64 ANY-POSITIVE identity evidence: {seen.sum()}/64 = {seen.mean():.6f}")
print(f"      mean positive-channel votes: {votes.mean():.3f} / {len(CAND)}")
print(f"      >=16 positive votes: {np.mean(votes>=16):.6f}")
print(f"      >=32 positive votes: {np.mean(votes>=32):.6f}")

# Exact token residual anatomy over every layer, true vs sibling.
TOK=[]
for L in range(NL):
    pos=[];mx=[];mean=[];which=[]
    for i in range(64):
        j=pair_of[i]
        a,b,k=token_margin(Q[i][1][L],M[i][1][L],M[j][1][L])
        pos.append(a>0);mx.append(a);mean.append(b);which.append(k)
    TOK.append((L,float(np.mean(pos)),float(np.mean(mx)),float(np.mean(mean)),np.array(mx),np.array(mean),which))
TOK.sort(key=lambda x:(x[1],x[2]),reverse=True)
print("\n      TOKEN RESIDUAL LAYER ATLAS")
for x in TOK:
    print(f"      L{x[0]:02d} POS={x[1]:.6f} MAXΔ={x[2]:+.6f} MEANΔ={x[3]:+.6f}")

bestL=TOK[0][0]
print(f"\n      BEST TOKEN-RESIDUAL LAYER DIAGNOSTIC: L{bestL:02d}")
for i in range(64):
    j=pair_of[i]
    mx,mn,k=token_margin(Q[i][1][bestL],M[i][1][bestL],M[j][1][bestL])
    qt=tok.decode([int(Q[i][0][k])]) if k<len(Q[i][0]) else "?"
    print(f"      [{i:02d}] {RAW[i][1]:25s} vs {RAW[j][1]:25s} maxΔ={mx:+.5f} meanΔ={mn:+.5f} token={qt!r}")

print("[8/8] Null controls + seal...")
# Within-family label-swap null. For each query independently, true/sibling sign is randomized.
best_name,best_res=TOP[0]
obs=best_res["acc"];arr=best_res["margins"]
rng=np.random.default_rng(SEED);null=[]
for _ in range(10000):
    sg=rng.choice([-1.,1.],size=64)
    null.append(float(np.mean(arr*sg>0)))
null=np.array(null)
p=float((1+np.sum(null>=obs))/(1+len(null)))
p95=float(np.quantile(null,.95));p99=float(np.quantile(null,.99))

# Cross-signal complementarity: greedy is diagnostic only and cannot be carried into TEST493 as-is.
pool=TOP[:30]
selected=[];covered=np.zeros(64,dtype=bool)
for step in range(8):
    best=None
    for n,r in pool:
        if n in selected: continue
        c=covered|(r["margins"]>0)
        gain=int(c.sum()-covered.sum())
        score=(gain,r["acc"],r["margin"])
        if best is None or score>best[0]: best=(score,n,r,c)
    if best is None or best[0][0]<=0: break
    selected.append(best[1]);covered=best[3]
    print(f"      ORACLE-COVER +{step+1}: {best[1]:14s} gain={best[0][0]:02d} covered={covered.sum():02d}/64")
verdict="NATIVE_INSTANCE_IDENTITY_SIGNAL_FOUND" if obs>=.65 and p<.01 else ("WEAK_NATIVE_INSTANCE_IDENTITY_SIGNAL" if obs>.5 and p<.05 else "NO_STABLE_INSTANCE_IDENTITY_SIGNAL")

print("\n"+"="*158)
print("TEST492 FINAL RESULT — INSTANCE-IDENTITY NATIVE SIGNAL ATLAS")
print("="*158)
print("FOUNDATION          : TEST491 residual-rule falsification")
print("TEST482 BLOB SHA    :",TEST482_SHA)
print("TEST488 LOCK SHA    :",TEST488_SHA)
print("TEST489 LOCK SHA    :",TEST489_SHA)
print("TEST490 LOCK SHA    :",TEST490_SHA)
print("TEST491 LOCK SHA    :",TEST491_SHA)
print("TEST492 LOCK SHA    :",LOCK)
print("DISCOVERY PANEL     : exact TEST491 64 / 32 same-family pairs")
print("PASSIVE CHANNELS    :",len(RES),"+",len(HEAD),"head-local K/V")
print("-"*158)
print("BEST CHANNEL        :",best_name)
print("BEST PAIR ACC       :",obs)
print("BEST MEAN MARGIN    :",best_res["margin"])
print("LABEL-SWAP NULL P95 :",p95)
print("LABEL-SWAP NULL P99 :",p99)
print("EMPIRICAL P         :",p)
print("ANY-POS ORACLE      :",int(seen.sum()),"/ 64")
print("GREEDY ORACLE COVER :",int(covered.sum()),"/ 64")
print("GREEDY DIAGNOSTIC   :",selected)
print("-"*158)
print("VERDICT             :",verdict)
print("NEW ADDRESSER       : NONE")
print("FUSION RULE         : NONE")
print("WEIGHT / THRESHOLD  : NONE")
print("ROUTER / ANN / GRAPH: NONE")
print("CARTRIDGE READOUT   : NOT USED")
print("TRAINING / LoRA     : NONE")
print("MODEL CHANGE        : NONE")
print("POST-HOC STATUS     : DISCOVERY/X-RAY ONLY")
print("NEXT                : freeze only a predeclared identity hypothesis from this atlas, then TEST493 on entirely fresh concepts/instances.")
print("="*158)
