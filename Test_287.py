# ================================================================================================
# AKBASCORE - TEST 287 FIXED
# HELD OUT TARGET LATENT CAUSAL WRITE AND MOTOR OFF TERMINAL VERIFICATION
# TEST286 COMPILER - SEASC L0-L25 - L26-L27 MOTOR OFF - PHYSICAL WRITE ONLY
# NO TRAINING - NO BEHAVIORAL RETRIEVAL
# ================================================================================================
import os,sys,json,time,random,hashlib,subprocess,importlib.util
from datetime import datetime,timezone
from pathlib import Path
for m,p in [("torch","torch"),("transformers","transformers"),("numpy","numpy")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required")
SEED=287
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda");MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
TOTAL=28;H_EXPECT=3584;EPS=1e-8;REPORT=[19,23,25,26,27]
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20;END=25;TARGET_RSS=.250235055
ROOT=Path("/content/AKBASCORE_TEST287") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST287")
ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
POOLS={
"COLOR":["red","blue","green","yellow","black","white"],
"PLACE":["Ankara","Tokyo","Lisbon","Cairo","Oslo","Lima"],
"NUMBER":["17","83","29","64","41","92"],
"MATERIAL":["copper","titanium","iron","aluminum","glass","ceramic"],
"ACTION":["running","sleeping","opening","closing","ascending","descending"],
"ABSTRACT":["justice","chaos","freedom","constraint","certainty","doubt"],
"OBJECT":["hammer","telescope","lantern","compass","violin","anchor"],
"ADJECTIVE":["bright","dark","heavy","light","smooth","rough"],
"NAME":["Elena","Marcus","Aylin","Victor","Nora","Adrian"],
"NOVEL":["Velorium","Nexarith","Caldris","Zophene","Meraxon","Tilvara"]}
CTX={
"DECL":"The recorded value is {x}.",
"QA":"Question: What value was recorded?\nAnswer: {x}.",
"DIALOGUE":"Operator: Which value should I use?\nAssistant: Use {x}.",
"TECH":"SYSTEM_RECORD\nfield=value\nvalue={x}\nEND_RECORD",
"STORY":"After checking the sealed note, the researcher found a single entry. It was {x}.",
"MINIMAL":"{x}"}
TRAIN_CTX=["DECL","QA","DIALOGUE","TECH","STORY"];BLIND="MINIMAL"
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(EPS))
def canon(o):return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
print("="*108);print("TEST 287 FIXED - HELD OUT TARGET LATENT CAUSAL WRITE AND MOTOR OFF TERMINAL VERIFICATION");print("="*108);print("START:",START)
print("[1/13] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if tv>=(4,56) else "torch_dtype"
torch.cuda.synchronize();t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size;PDT=next(model.parameters()).dtype;torch.cuda.synchronize()
if len(layers)!=TOTAL or H!=H_EXPECT or PDT!=torch.bfloat16:raise RuntimeError("Architecture or dtype mismatch")
print(f"OK | {MODEL_ID} | 28L | H={H} | {PDT} | {time.perf_counter()-t:.2f}s")
print("[2/13] WEIGHT SENTINEL AND FIXED RSS")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();CN=list(CTX);CL=list(POOLS);KEYS=[f"{c}:{x}" for c,p in POOLS.items() for x in p]
base=[]
for L in range(END+1):
    x=ZIRVE*np.exp(-SONUM*L)*(1+SONUM*L)+TABAN
    base.append(IVME*x/(ZIRVE+TABAN))
base_rss=np.sqrt(sum(r*r for r in base));scale=TARGET_RSS/base_rss
RHO=[r*scale for r in base];RSS=np.sqrt(sum(r*r for r in RHO))
print("FP:",[f"{x:.4f}" for x in FP0]);print(f"SEASC L0-L25 | L26-L27 OFF | RSS={RSS:.9f} | SCALE={scale:.6f} | RHO0={RHO[0]*100:.3f}% | RHO25={RHO[25]*100:.3f}%")
if abs(RSS-TARGET_RSS)>1e-9:raise RuntimeError("RSS mismatch")
def enc(x):
    s=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to(DEVICE)
@torch.inference_mode()
def capture(text):
    o=model(**enc(text),use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)]
print("[3/13] CACHE 360 ABSOLUTE STATES")
STATE={}
for cls,pool in POOLS.items():
    STATE[cls]={}
    for x in pool:STATE[cls][x]={c:capture(f.format(x=x)) for c,f in CTX.items()}
    print(cls,"READY")
print("[4/13] TEST285 EXTERNAL BANK FORGE")
RAW={}
for cls,pool in POOLS.items():
    banks=[b for b in CL if b!=cls]
    for target in pool:
        k=f"{cls}:{target}";RAW[k]={}
        for bank in banks:
            RAW[k][bank]={}
            for c in CN:
                RAW[k][bank][c]=[]
                for L in range(TOTAL):
                    ht=STATE[cls][target][c][L]
                    RAW[k][bank][c].append(unit(torch.stack([unit(ht-STATE[bank][r][c][L]) for r in POOLS[bank]]).mean(0)))
print("RAW FORGE READY")
print("[5/13] TEST285 CLASS COMMON REMOVAL")
BRES={}
for cls,pool in POOLS.items():
    kk=[f"{cls}:{x}" for x in pool];banks=[b for b in CL if b!=cls]
    for bank in banks:
        for c in CN:
            for L in range(TOTAL):
                mu=torch.stack([RAW[k][bank][c][L] for k in kk]).mean(0)
                for k in kk:
                    if k not in BRES:BRES[k]={}
                    if bank not in BRES[k]:BRES[k][bank]={}
                    if c not in BRES[k][bank]:BRES[k][bank][c]=[None]*TOTAL
                    BRES[k][bank][c][L]=unit(RAW[k][bank][c][L]-mu)
print("BANK CLEAN READY")
print("[6/13] TEST286 BANK CONSENSUS AND CONTEXT CLEANING")
BC={};CRES={}
for cls,pool in POOLS.items():
    banks=[b for b in CL if b!=cls]
    for target in pool:
        k=f"{cls}:{target}";BC[k]={}
        for c in CN:
            BC[k][c]=[]
            for L in range(TOTAL):BC[k][c].append(unit(torch.stack([BRES[k][b][c][L] for b in banks]).mean(0)))
for cls,pool in POOLS.items():
    kk=[f"{cls}:{x}" for x in pool]
    for c in CN:
        for L in range(TOTAL):
            mu=torch.stack([BC[k][c][L] for k in kk]).mean(0)
            for k in kk:
                if k not in CRES:CRES[k]={}
                if c not in CRES[k]:CRES[k][c]=[None]*TOTAL
                CRES[k][c][L]=unit(BC[k][c][L]-mu)
print("CONTEXT CLEAN READY")
print("[7/13] FIVE CONTEXT COMPILER - MINIMAL HELD OUT")
PACK={}
for k in KEYS:
    PACK[k]=[]
    for L in range(TOTAL):PACK[k].append(unit(torch.stack([CRES[k][c][L] for c in TRAIN_CTX]).mean(0)))
loo=[cos(PACK[k][27],CRES[k][BLIND][27]) for k in KEYS]
print(f"L27 HELDOUT MINIMAL | MEAN={np.mean(loo):+.4f} | MED={np.median(loo):+.4f} | P10={np.percentile(loo,10):+.4f} | POS={np.mean(np.array(loo)>0):.3f}")
print("[8/13] BLIND BASELINE")
BASE=capture("")
print("BLIND PROMPT = EMPTY USER CONTENT | BASELINE READY")
print("[9/13] CAUSAL WRITE L0-L25")
AUD={"calls":0,"max_dev":0.0};ACTIVE=set()
def install(vecs,sign):
    hs=[]
    for L in range(END+1):
        rho=RHO[L]
        def hook(mod,args,out,L=L,rho=rho):
            y=out[0] if isinstance(out,tuple) else out;z=y[:,-1,:].float()
            d=vecs[L].to(z.device)*z.norm(dim=-1,keepdim=True)*rho*sign
            rel=float((d.norm(dim=-1)/(z.norm(dim=-1)+EPS)).max())
            AUD["calls"]+=1;AUD["max_dev"]=max(AUD["max_dev"],abs(rel-rho))
            yy=y.clone();yy[:,-1,:]=(z+d).to(y.dtype)
            return (yy,)+out[1:] if isinstance(out,tuple) else yy
        h=layers[L].register_forward_hook(hook);hs.append(h);ACTIVE.add(id(h))
    return hs
@torch.inference_mode()
def steered(vecs,sign):
    hs=install(vecs,sign)
    try:
        o=model(**enc(""),use_cache=False,output_hidden_states=True,return_dict=True)
        return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)]
    finally:
        for h in hs:
            h.remove();ACTIVE.discard(id(h))
PLUS={};MINUS={}
for i,k in enumerate(KEYS,1):
    PLUS[k]=steered(PACK[k],+1.0);MINUS[k]=steered(PACK[k],-1.0)
    if i%10==0:print(f"{i}/60")
print("[10/13] TERMINAL PHYSICAL WRITE")
PHYS={}
for L in REPORT:
    pp=[];mm=[];ss=[];rr=[]
    for k in KEYS:
        dp=PLUS[k][L]-BASE[L];dm=MINUS[k][L]-BASE[L]
        pp.append(cos(dp,PACK[k][L]));mm.append(cos(dm,PACK[k][L]));ss.append(cos(PLUS[k][L]-MINUS[k][L],PACK[k][L]))
        rr.append(float(dp.norm()/BASE[L].norm().clamp_min(EPS)))
    PHYS[str(L)]={"plus_packet":float(np.mean(pp)),"minus_packet":float(np.mean(mm)),"signed_separation":float(np.mean(ss)),"plus_rel":float(np.mean(rr))}
    x=PHYS[str(L)];print(f"L{L:02d} | PLUS_PKT={x['plus_packet']:+.4f} | MINUS_PKT={x['minus_packet']:+.4f} | SEP_PKT={x['signed_separation']:+.4f} | REL={x['plus_rel']*100:.2f}%")
print("[11/13] HELD OUT MINIMAL TARGET ALIGNMENT")
ALIGN={}
for L in REPORT:
    pp=[];mm=[];pn=[];mn=[]
    for k in KEYS:
        cls,target=k.split(":",1);dp=PLUS[k][L]-BASE[L];dm=MINUS[k][L]-BASE[L]
        nat=STATE[cls][target][BLIND][L]-BASE[L]
        pp.append(cos(dp,CRES[k][BLIND][L]));mm.append(cos(dm,CRES[k][BLIND][L]))
        pn.append(cos(dp,nat));mn.append(cos(dm,nat))
    ALIGN[str(L)]={"plus_heldout":float(np.mean(pp)),"minus_heldout":float(np.mean(mm)),"plus_natural":float(np.mean(pn)),"minus_natural":float(np.mean(mn))}
    x=ALIGN[str(L)];print(f"L{L:02d} | PLUS_HELD={x['plus_heldout']:+.4f} | MINUS_HELD={x['minus_heldout']:+.4f} | PLUS_NAT={x['plus_natural']:+.4f} | MINUS_NAT={x['minus_natural']:+.4f}")
print("[12/13] TARGET SPECIFICITY CONTROL")
SPEC={}
for L in REPORT:
    own=[];wrong=[]
    for i,k in enumerate(KEYS):
        dp=PLUS[k][L]-BASE[L];own.append(cos(dp,PACK[k][L]));wrong.append(cos(dp,PACK[KEYS[(i+17)%len(KEYS)]][L]))
    SPEC[str(L)]={"own":float(np.mean(own)),"wrong":float(np.mean(wrong)),"margin":float(np.mean(own)-np.mean(wrong))}
    x=SPEC[str(L)];print(f"L{L:02d} | OWN={x['own']:+.4f} | WRONG={x['wrong']:+.4f} | MARGIN={x['margin']:+.4f}")
print("[13/13] INTEGRITY AND SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("AkbasCore hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["max_dev"]>1e-6:raise RuntimeError("Dose audit failure")
R={"schema":"akbascore.test287.fixed.v2","test":"TEST 287 FIXED","start":START,"end":utc(),"model":MODEL_ID,
"question":"Can a target-specific latent compiled from five contexts be causally written into a blind state and remain measurable after the SEASC motor is switched off?",
"compiler":{"train_contexts":TRAIN_CTX,"heldout_context":BLIND,"targets":60,"external_banks":9},
"seasc":{"on_layers":[0,25],"off_layers":[26,27],"rss_definition":"sqrt(sum(rho_L^2))","target_rss":TARGET_RSS,"actual_rss":RSS,"scale":float(scale),"rho0":RHO[0],"rho25":RHO[25]},
"heldout_l27":{"mean":float(np.mean(loo)),"median":float(np.median(loo)),"p10":float(np.percentile(loo,10)),"positive_fraction":float(np.mean(np.array(loo)>0))},
"physical_write":PHYS,"heldout_alignment":ALIGN,"specificity":SPEC,
"integrity":{"selection":False,"training":False,"behavioral_retrieval":False,"weight_update":False,"injection_calls":AUD["calls"],"max_dose_deviation":AUD["max_dev"],"akbascore_hooks_remaining":len(ACTIVE),"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
sha=hashlib.sha256(canon(R)).hexdigest();run=f"T287-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{run}.json";tp=ROOT/f"{run}.txt"
jp.write_bytes(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2).encode())
out=["="*108,"TEST 287 FIXED - HELD OUT TARGET LATENT CAUSAL WRITE AND MOTOR OFF TERMINAL VERIFICATION","="*108]
for L in REPORT:out.append(f"L{L:02d} PLUS_PKT={PHYS[str(L)]['plus_packet']:+.6f} MINUS_PKT={PHYS[str(L)]['minus_packet']:+.6f} SEP={PHYS[str(L)]['signed_separation']:+.6f} PLUS_HELD={ALIGN[str(L)]['plus_heldout']:+.6f} PLUS_NAT={ALIGN[str(L)]['plus_natural']:+.6f} OWN={SPEC[str(L)]['own']:+.6f} WRONG={SPEC[str(L)]['wrong']:+.6f}")
out+=["",f"RSS={RSS:.9f}",f"RHO0={RHO[0]*100:.6f}%",f"RHO25={RHO[25]*100:.6f}%",f"INJECTION_CALLS={AUD['calls']}",f"MAX_DOSE_DEVIATION={AUD['max_dev']:.3e}","NO TRAINING - NO BEHAVIORAL RETRIEVAL - WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]
tp.write_text("\n".join(out),encoding="utf-8")
print("="*108);print("TEST 287 FIXED COMPLETE")
print("FIVE CONTEXT COMPILER | CAUSAL WRITE | L26-L27 MOTOR OFF | PHYSICAL TERMINAL VERIFICATION")
print(f"RSS={RSS:.9f} | RHO0={RHO[0]*100:.3f}% | RHO25={RHO[25]*100:.3f}% | CALLS={AUD['calls']} | MAX_DOSE_DEV={AUD['max_dev']:.3e}")
print("NO TRAINING | NO BEHAVIORAL RETRIEVAL | WEIGHT INTEGRITY PASS | AKBASCORE HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
