# ================================================================================================
# AKBASCORE - TEST 289
# BLIND BEHAVIORAL READOUT AFTER TARGET SPECIFIC LATENT WRITE
# TEST288 COMPILER - SEASC L0-L25 - L26-L27 OFF - NULL / TARGET / WRONG
# NO TRAINING - NO CLASSIFIER - GREEDY GENERATION
# ================================================================================================
import os,sys,json,time,random,hashlib,subprocess,importlib.util,re
from datetime import datetime,timezone
from pathlib import Path
for m,p in [("torch","torch"),("transformers","transformers"),("numpy","numpy")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required")
SEED=289
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda");MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
TOTAL=28;H_EXPECT=3584;EPS=1e-8;IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20;END=25;TARGET_RSS=.250235055
MAX_NEW=12;BLIND_PROMPT="Return the single value represented by the internal state. Answer with only the value."
ROOT=Path("/content/AKBASCORE_TEST289") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST289")
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
TRAIN_CTX=["DECL","QA","DIALOGUE","TECH","STORY"]
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def canon(o):return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
def normtxt(x):return re.sub(r"[^a-z0-9]+"," ",x.lower()).strip()
def exact(out,target):return normtxt(out)==normtxt(target)
def mention(out,target):return re.search(r"(?<![a-z0-9])"+re.escape(normtxt(target))+r"(?![a-z0-9])",normtxt(out)) is not None
print("="*108);print("TEST 289 - BLIND BEHAVIORAL READOUT AFTER TARGET SPECIFIC LATENT WRITE");print("="*108);print("START:",START)
print("[1/13] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
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
FP0=fp();CL=list(POOLS);CN=list(CTX);KEYS=[f"{c}:{x}" for c,p in POOLS.items() for x in p]
base=[IVME*(ZIRVE*np.exp(-SONUM*L)*(1+SONUM*L)+TABAN)/(ZIRVE+TABAN) for L in range(END+1)]
scale=TARGET_RSS/np.sqrt(sum(r*r for r in base));RHO=[r*scale for r in base];RSS=np.sqrt(sum(r*r for r in RHO))
print("FP:",[f"{x:.4f}" for x in FP0]);print(f"SEASC L0-L25 | L26-L27 OFF | RSS={RSS:.9f} | RHO0={RHO[0]*100:.3f}% | RHO25={RHO[25]*100:.3f}%")
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
                    BRES.setdefault(k,{}).setdefault(bank,{}).setdefault(c,[None]*TOTAL)
                    BRES[k][bank][c][L]=unit(RAW[k][bank][c][L]-mu)
print("BANK CLEAN READY")
print("[6/13] TEST286 BANK CONSENSUS AND CONTEXT CLEANING")
BC={};CRES={}
for cls,pool in POOLS.items():
    banks=[b for b in CL if b!=cls]
    for target in pool:
        k=f"{cls}:{target}";BC[k]={}
        for c in CN:BC[k][c]=[unit(torch.stack([BRES[k][b][c][L] for b in banks]).mean(0)) for L in range(TOTAL)]
for cls,pool in POOLS.items():
    kk=[f"{cls}:{x}" for x in pool]
    for c in CN:
        for L in range(TOTAL):
            mu=torch.stack([BC[k][c][L] for k in kk]).mean(0)
            for k in kk:
                CRES.setdefault(k,{}).setdefault(c,[None]*TOTAL)
                CRES[k][c][L]=unit(BC[k][c][L]-mu)
print("CONTEXT CLEAN READY")
print("[7/13] FIVE CONTEXT TARGET COMPILER")
PACK={k:[unit(torch.stack([CRES[k][c][L] for c in TRAIN_CTX]).mean(0)) for L in range(TOTAL)] for k in KEYS}
print("60 TARGET PACKETS READY")
print("[8/13] GENERATION ENGINE")
AUD={"calls":0,"max_dev":0.0};ACTIVE=set()
def install(vecs):
    hs=[]
    for L in range(END+1):
        rho=RHO[L]
        def hook(mod,args,out,L=L,rho=rho):
            y=out[0] if isinstance(out,tuple) else out;z=y[:,-1,:].float();d=vecs[L].to(z.device)*z.norm(dim=-1,keepdim=True)*rho
            rel=float((d.norm(dim=-1)/(z.norm(dim=-1)+EPS)).max());AUD["calls"]+=1;AUD["max_dev"]=max(AUD["max_dev"],abs(rel-rho))
            yy=y.clone();yy[:,-1,:]=(z+d).to(y.dtype)
            return (yy,)+out[1:] if isinstance(out,tuple) else yy
        h=layers[L].register_forward_hook(hook);hs.append(h);ACTIVE.add(id(h))
    return hs
@torch.inference_mode()
def generate(vecs=None):
    e=enc(BLIND_PROMPT);n=e["input_ids"].shape[1];hs=install(vecs) if vecs is not None else []
    try:
        y=model.generate(**e,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)
        return tok.decode(y[0,n:],skip_special_tokens=True).strip()
    finally:
        for h in hs:h.remove();ACTIVE.discard(id(h))
print("GREEDY | MAX_NEW=12 | SOURCE TARGET ABSENT FROM BLIND PROMPT")
print("[9/13] NULL CONTROL")
NULL=generate(None);print("NULL:",repr(NULL))
print("[10/13] TARGET AND WRONG TARGET READOUT")
ROWS=[];counts={"target_exact":0,"target_mention":0,"wrong_exact_target":0,"wrong_mention_target":0,"wrong_outputs_wrong":0}
for i,k in enumerate(KEYS):
    cls,target=k.split(":",1);wrong_key=KEYS[(i+17)%len(KEYS)];wrong=wrong_key.split(":",1)[1]
    ot=generate(PACK[k]);ow=generate(PACK[wrong_key])
    te=exact(ot,target);tm=mention(ot,target);we=exact(ow,target);wm=mention(ow,target);ww=mention(ow,wrong)
    counts["target_exact"]+=te;counts["target_mention"]+=tm;counts["wrong_exact_target"]+=we;counts["wrong_mention_target"]+=wm;counts["wrong_outputs_wrong"]+=ww
    ROWS.append({"key":k,"target":target,"wrong_key":wrong_key,"wrong_target":wrong,"target_output":ot,"wrong_output":ow,"target_exact":te,"target_mention":tm,"wrong_exact_target":we,"wrong_mention_target":wm,"wrong_outputs_wrong":ww})
    print(f"{i+1:02d}/60 | {k:20s} | TARGET={ot!r} | WRONG={ow!r}")
print("[11/13] GLOBAL BEHAVIOR")
GLOBAL={k:v/60 for k,v in counts.items()}
for k,v in GLOBAL.items():print(f"{k.upper():20s} {v:.3f} ({counts[k]}/60)")
print("[12/13] CLASS ATLAS")
ATLAS={}
for cls,pool in POOLS.items():
    rr=[r for r in ROWS if r["key"].startswith(cls+":")]
    ATLAS[cls]={"target_exact":float(np.mean([r["target_exact"] for r in rr])),"target_mention":float(np.mean([r["target_mention"] for r in rr])),"wrong_target_mention":float(np.mean([r["wrong_mention_target"] for r in rr])),"wrong_outputs_wrong":float(np.mean([r["wrong_outputs_wrong"] for r in rr]))}
    x=ATLAS[cls];print(f"{cls:10s} | EXACT={x['target_exact']:.3f} | MENTION={x['target_mention']:.3f} | WRONG->TARGET={x['wrong_target_mention']:.3f} | WRONG->WRONG={x['wrong_outputs_wrong']:.3f}")
print("[13/13] INTEGRITY AND SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("AkbasCore hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["max_dev"]>1e-6:raise RuntimeError("Dose audit failure")
R={"schema":"akbascore.test289.v1","test":"TEST 289","start":START,"end":utc(),"model":MODEL_ID,
"question":"Can a target-specific latent causally written into a target-free blind prompt be converted by the model into the target value during greedy language generation?",
"blind_prompt":BLIND_PROMPT,"max_new_tokens":MAX_NEW,"compiler":{"train_contexts":TRAIN_CTX,"targets":60,"external_banks":9},
"seasc":{"on_layers":[0,25],"off_layers":[26,27],"rss_definition":"sqrt(sum(rho_L^2))","rss":RSS,"rho0":RHO[0],"rho25":RHO[25]},
"null_output":NULL,"global":GLOBAL,"counts":counts,"class_atlas":ATLAS,"rows":ROWS,
"integrity":{"training":False,"classifier":False,"selection":False,"weight_update":False,"injection_calls":AUD["calls"],"max_dose_deviation":AUD["max_dev"],"akbascore_hooks_remaining":len(ACTIVE),"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
sha=hashlib.sha256(canon(R)).hexdigest();run=f"T289-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{run}.json";tp=ROOT/f"{run}.txt"
jp.write_bytes(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2).encode())
out=["="*108,"TEST 289 - BLIND BEHAVIORAL READOUT AFTER TARGET SPECIFIC LATENT WRITE","="*108,f"NULL={NULL!r}"]
for r in ROWS:out.append(f"{r['key']} | TARGET={r['target_output']!r} | WRONG={r['wrong_output']!r} | EXACT={int(r['target_exact'])} | MENTION={int(r['target_mention'])}")
out+=["",f"TARGET_EXACT={GLOBAL['target_exact']:.6f}",f"TARGET_MENTION={GLOBAL['target_mention']:.6f}",f"WRONG_TARGET_MENTION={GLOBAL['wrong_mention_target']:.6f}",f"WRONG_OUTPUTS_WRONG={GLOBAL['wrong_outputs_wrong']:.6f}",f"RSS={RSS:.9f}",f"CALLS={AUD['calls']}",f"MAX_DOSE_DEV={AUD['max_dev']:.3e}","NO TRAINING - NO CLASSIFIER - NO SELECTION - WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]
tp.write_text("\n".join(out),encoding="utf-8")
print("="*108);print("TEST 289 COMPLETE")
print(f"TARGET EXACT={GLOBAL['target_exact']:.3f} | TARGET MENTION={GLOBAL['target_mention']:.3f} | WRONG->TARGET={GLOBAL['wrong_mention_target']:.3f} | WRONG->WRONG={GLOBAL['wrong_outputs_wrong']:.3f}")
print(f"RSS={RSS:.9f} | CALLS={AUD['calls']} | MAX_DOSE_DEV={AUD['max_dev']:.3e}")
print("NO TRAINING | NO CLASSIFIER | NO SELECTION | WEIGHT INTEGRITY PASS | AKBASCORE HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
