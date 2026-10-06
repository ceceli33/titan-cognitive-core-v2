# Copyright © 2026 Mustafa Akbaş
# New original AKBASCORE MAM material is source-available under the repository LICENSE.
# Earlier MIT-licensed Titan Cognitive Core / AkbasCore material remains under its original license.
# See the repository LICENSE for complete terms.
# TEST491 — AKBASCORE MAM · FROZEN BASIN+RESIDUAL ADDRESSER · FRESH HELD-OUT CONCEPT PAIRS
# TEST490 mechanism frozen BEFORE this panel:
#   STAGE-1 BASIN    = TOKENMAX L10
#   STAGE-2 IDENTITY = token-local residual L08
# No layer search / threshold search / channel fitting / training / LoRA / ANN / graph / cartridge readout.
import os,sys,re,json,time,random,hashlib,subprocess,importlib.util
os.environ["TOKENIZERS_PARALLELISM"]="false"
def need(p):
    if importlib.util.find_spec(p) is None: subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
for p in ["torch","transformers","accelerate"]: need(p)
import numpy as np,torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM

TEST="491";SEED=491;MODEL_ID="Qwen/Qwen2.5-7B-Instruct";DEVICE="cuda"
BASIN_L=10;ID_L=8;BASIN_K=5
TEST482_SHA="ecc630144a44479cb35c432c50eb0d09bc6866d7"
TEST488_SHA="d2c2e744532e1fe9b4c5be745fce6892720745dd590b49eca59ca3ef753fdfd8"
TEST489_SHA="ed03fb4b6f1afa410e1a28329d05b744889b8836f69f3572c4b52032b8b70fb0"
TEST490_SHA="237f3c7a9cc2478b2ca55944f0a356101c8f87c5a7ef250f6267fc04a9fc1dd2"
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)

# Completely fresh concepts. Every concept has two instances.
# Crucially: no exact duplicate instance labels. Queries contain no exact object label.
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
assert len(RAW)==64
assert len({x[1] for x in RAW})==64
assert len({x[2] for x in RAW})==64
from collections import defaultdict
fam=defaultdict(list)
for i,x in enumerate(RAW): fam[x[0]].append(i)
assert len(fam)==32 and all(len(v)==2 for v in fam.values())

def source(obj,cid): return f"The {obj} is stored in container {cid}."
def query(desc): return f"Which container holds {desc}?"
MEM=[source(x[1],x[2]) for x in RAW]
QUE=[query(x[3]) for x in RAW]
LOCK_OBJ={
"test":TEST,"seed":SEED,"model":MODEL_ID,"n":len(RAW),
"basin_layer":BASIN_L,"identity_layer":ID_L,"basin_k":BASIN_K,
"rule":"stage1 TOKENMAX_L10 global top5; stage2 within top5 maximize L08 token-local residual against strongest same-basin competitor; no fitted weights",
"foundation":{"482":TEST482_SHA,"488":TEST488_SHA,"489":TEST489_SHA,"490":TEST490_SHA},
"panel":RAW
}
LOCK=hashlib.sha256(json.dumps(LOCK_OBJ,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("="*158)
print("TEST491 — AKBASCORE MAM · FROZEN BASIN+RESIDUAL ADDRESSER · FRESH HELD-OUT CONCEPT PAIRS")
print("FROZEN FROM TEST490 · BASIN TOKENMAX L10 → TOKEN-LOCAL IDENTITY RESIDUAL L08 · NO SEARCH/TUNING")
print("="*158)
print("LOCK SHA:",LOCK)
print("TEST490 LOCK:",TEST490_SHA)
print("PANEL:",len(RAW),"records /",len(fam),"fresh concepts / 2 instances each")
print("[1/8] Loading frozen Qwen...")
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.bfloat16,attn_implementation="sdpa",device_map={"":0})
model.eval()
for p in model.parameters(): p.requires_grad_(False)
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | {len(model.model.layers)}L | frozen BF16")

LAYERS=sorted(set([BASIN_L,ID_L]))
def enc_text(s):
    return tok(s,return_tensors="pt",add_special_tokens=True).input_ids.to(DEVICE)
@torch.inference_mode()
def capture(s):
    ids=enc_text(s)
    o=model(input_ids=ids,output_hidden_states=True,use_cache=False,return_dict=True)
    out={}
    for L in LAYERS:
        # hidden_states[L+1] = output after transformer block L
        h=o.hidden_states[L+1][0].float()
        out[L]=F.normalize(h,dim=-1).cpu()
    return ids[0].cpu(),out
print("[2/8] Capturing fresh memory/query token states...")
MIDS=[];MHS=[];QIDS=[];QHS=[]
for i,s in enumerate(MEM):
    a,b=capture(s);MIDS.append(a);MHS.append(b)
    if (i+1)%16==0: print(f"      memories {i+1:02d}/64")
for i,s in enumerate(QUE):
    a,b=capture(s);QIDS.append(a);QHS.append(b)
    if (i+1)%16==0: print(f"      queries  {i+1:02d}/64")

# TOKENMAX(q,m): mean across query tokens of each query token's best memory-token cosine.
def token_matrix(q,m): return q@m.T
def tokenmax(q,m): return float(token_matrix(q,m).max(dim=1).values.mean())
def all_scores(L):
    z=np.zeros((64,64),np.float64)
    for i in range(64):
        q=QHS[i][L]
        for j in range(64): z[i,j]=tokenmax(q,MHS[j][L])
    return z
print("[3/8] Frozen Stage-1 semantic basin...")
S10=all_scores(BASIN_L)
def ranks(S):
    rr=[]
    for i in range(len(S)):
        order=np.argsort(-S[i],kind="stable")
        rr.append(int(np.where(order==i)[0][0])+1)
    return np.array(rr)
def met(r):
    return {"R1":float(np.mean(r<=1)),"R5":float(np.mean(r<=5)),"R16":float(np.mean(r<=16)),
            "MRR":float(np.mean(1.0/r)),"MEDR":float(np.median(r)),"MEANR":float(np.mean(r))}
R_BASE=ranks(S10);M_BASE=met(R_BASE)
print("      TOKENMAX L10:",M_BASE)

# Stage 2 is intentionally local and frozen:
# 1) take global top-5 basin candidates from L10.
# 2) for each candidate c, compare its L08 token affinities with the strongest OTHER candidate
#    in the same top-5 for each query token.
# 3) positive local residual = max_q [ max_t cos(q_t,c) - max_t cos(q_t,competitor) ].
# 4) final score = candidate's rank-normalized basin evidence + rank-normalized identity residual.
# No fitted coefficient: equal rank evidence, frozen before held-out inspection.
def id_residual(i,c,cands):
    q=QHS[i][ID_L]
    own=token_matrix(q,MHS[c][ID_L]).max(dim=1).values.numpy()
    others=[]
    for d in cands:
        if d==c: continue
        others.append(token_matrix(q,MHS[d][ID_L]).max(dim=1).values.numpy())
    if not others: return 0.0,None,0.0
    comp=np.max(np.stack(others,0),axis=0)
    delta=own-comp
    k=int(np.argmax(delta))
    return float(delta[k]),k,float(delta.mean())
def rank01(v):
    order=np.argsort(v,kind="stable")
    out=np.empty(len(v),np.float64)
    if len(v)==1: out[:]=1.;return out
    out[order]=np.linspace(0.,1.,len(v))
    return out

print("[4/8] Applying frozen basin+residual rule...")
FINAL=np.full((64,64),-1e9,np.float64)
TRACE=[]
for i in range(64):
    cands=list(np.argsort(-S10[i],kind="stable")[:BASIN_K])
    bs=np.array([S10[i,c] for c in cands])
    ir=[];tk=[];mn=[]
    for c in cands:
        a,b,cmean=id_residual(i,c,cands);ir.append(a);tk.append(b);mn.append(cmean)
    ir=np.array(ir)
    fused=rank01(bs)+rank01(ir)
    for z,c in enumerate(cands): FINAL[i,c]=fused[z]
    pred=cands[int(np.argmax(fused))]
    TRACE.append((cands,bs,ir,tk,np.array(mn),fused,pred))
R_FINAL=ranks(FINAL);M_FINAL=met(R_FINAL)
print("      BASIN+RESIDUAL:",M_FINAL)

print("[5/8] Family-vs-instance held-out decomposition...")
family_hit=[];family5=[]
for i in range(64):
    family=set(fam[RAW[i][0]])
    o=np.argsort(-S10[i],kind="stable")
    family_hit.append(int(o[0] in family))
    family5.append(int(any(x in family for x in o[:5])))
print(f"      L10 FAMILY R1={np.mean(family_hit):.6f} R5={np.mean(family5):.6f}")
print(f"      L10 INSTANCE R1={M_BASE['R1']:.6f} R5={M_BASE['R5']:.6f}")
print(f"      FUSED INSTANCE R1={M_FINAL['R1']:.6f} R5={M_FINAL['R5']:.6f}")

print("[6/8] Difficulty strata + pair discrimination...")
for g in ["STANDARD","HARD","EXTREME","ADVERSARIAL"]:
    ix=[i for i,x in enumerate(RAW) if x[4]==g]
    if not ix: continue
    print(f"      {g:11s} n={len(ix):02d} BASE-R1={np.mean(R_BASE[ix]<=1):.3f} BASE-R5={np.mean(R_BASE[ix]<=5):.3f} FUSED-R1={np.mean(R_FINAL[ix]<=1):.3f} FUSED-R5={np.mean(R_FINAL[ix]<=5):.3f}")
pair_base=[];pair_fused=[];pair_in_top5=[]
for _,ix in fam.items():
    a,b=ix
    for i,j in [(a,b),(b,a)]:
        pair_base.append(S10[i,i]>S10[i,j])
        cands,bs,ir,tk,mn,fu,pred=TRACE[i]
        pair_in_top5.append(i in cands and j in cands)
        if i in cands and j in cands:
            pair_fused.append(FINAL[i,i]>FINAL[i,j])
print(f"      PAIR BASE correct={np.mean(pair_base):.6f}")
print(f"      BOTH INSTANCES IN TOP5={np.mean(pair_in_top5):.6f}")
print(f"      PAIR FUSED correct when both visible={np.mean(pair_fused) if pair_fused else float('nan'):.6f}")

print("[7/8] Per-query audit...")
for i,x in enumerate(RAW):
    cands,bs,ir,tk,mn,fu,pred=TRACE[i]
    bt=cands[0]
    tki=tk[int(np.where(np.array(cands)==pred)[0][0])] if pred in cands else None
    qtoken=tok.decode([int(QIDS[i][tki])]) if tki is not None and tki<len(QIDS[i]) else "-"
    mark="OK" if pred==i else "MISS"
    print(f"      [{i:02d}] {mark:4s} {x[1]:25s} base-r={R_BASE[i]:2d} final-r={R_FINAL[i]:2d} | base={RAW[bt][1]:25s} final={RAW[pred][1]:25s} | id-token={qtoken!r}")

print("[8/8] Fixed permutation null + final seal...")
rng=np.random.default_rng(SEED)
null_mrr=[]
for z in range(1000):
    p=rng.permutation(64)
    null_mrr.append(np.mean(1.0/np.array([np.where(np.argsort(-FINAL[i],kind="stable")==p[i])[0][0]+1 for i in range(64)])))
null_mrr=np.array(null_mrr)
P99=float(np.quantile(null_mrr,.99));NULL_MEAN=float(null_mrr.mean())
gain_r1=M_FINAL["R1"]-M_BASE["R1"];gain_mrr=M_FINAL["MRR"]-M_BASE["MRR"]
# Verdict is descriptive and predeclared; no parameter changes follow from it.
if gain_r1>=.10 and gain_mrr>0 and M_FINAL["MRR"]>P99*3:
    verdict="FROZEN_BASIN_RESIDUAL_MECHANISM_VALIDATED"
elif gain_r1>0 and gain_mrr>0 and M_FINAL["MRR"]>P99*3:
    verdict="FROZEN_BASIN_RESIDUAL_SIGNAL_REPLICATED_BUT_GAIN_MODEST"
elif M_FINAL["MRR"]>P99*3:
    verdict="SEMANTIC_ADDRESS_SIGNAL_REPLICATED_BUT_RESIDUAL_RULE_NOT_VALIDATED"
else:
    verdict="HELDOUT_MECHANISM_NOT_VALIDATED"

print("\n"+"="*158)
print("TEST491 FINAL RESULT — FROZEN BASIN+RESIDUAL HELD-OUT VALIDATION")
print("="*158)
print("FOUNDATION          : TEST490 semantic-basin / token-residual split")
print("TEST482 BLOB SHA    :",TEST482_SHA)
print("TEST488 LOCK SHA    :",TEST488_SHA)
print("TEST489 LOCK SHA    :",TEST489_SHA)
print("TEST490 LOCK SHA    :",TEST490_SHA)
print("TEST491 LOCK SHA    :",LOCK)
print("FRESH PANEL         : 64 records / 32 unseen concepts / 2 instances each")
print("BASIN               : TOKENMAX L10 · global top-5")
print("IDENTITY             : L08 max positive token-local residual inside basin")
print("FUSION               : equal rank-normalized basin + residual; frozen")
print("-"*158)
print("BASE TOKENMAX L10   :",M_BASE)
print("FROZEN BASIN+ID     :",M_FINAL)
print("ΔR1                  :",gain_r1)
print("ΔMRR                 :",gain_mrr)
print("NULL MRR mean        :",NULL_MEAN)
print("NULL MRR P99         :",P99)
print("PAIR BASE ACC        :",float(np.mean(pair_base)))
print("PAIR BOTH IN TOP5    :",float(np.mean(pair_in_top5)))
print("PAIR FUSED ACC       :",float(np.mean(pair_fused)) if pair_fused else float("nan"))
print("-"*158)
print("VERDICT              :",verdict)
print("LAYER SEARCH         : NONE")
print("THRESHOLD SEARCH     : NONE")
print("WEIGHT FIT           : NONE")
print("CHANNEL FIT          : NONE")
print("POST-HOC SELECTION   : NONE")
print("ROUTER / ANN / GRAPH : NONE")
print("CARTRIDGE READOUT    : NOT USED")
print("TRAINING / LoRA      : NONE")
print("MODEL CHANGE         : NONE")
print("NEXT                 : if validated, test frozen mechanism against larger distractor banks; otherwise inspect failure without retuning this panel.")
print("="*158)
