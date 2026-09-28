# ================================================================================================
# AKBASCORE - TEST 296
# CAUSAL READOUT BRIDGE INTERVENTION
# PACK-ONLY vs OWN-BRIDGE vs WRONG-BRIDGE
# TEST295 COMPILER - FIXED RSS - L0-L25 SEASC - TERMINAL L26 BRIDGE - L27 OFF
# NO TRAINING - NO OPTIMIZATION - NO DOSE SWEEP
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
SEED=296
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda");MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
BLIND="Return the single value represented by the internal state. Answer with only the value."
TOTAL=28;H_EXPECT=3584;EPS=1e-8;END=25;TARGET_RSS=.250235055
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20;BRIDGE_REL=.02
ROOT=Path("/content/AKBASCORE_TEST296") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST296")
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
TRAIN=["DECL","QA","DIALOGUE","TECH","STORY"]
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def canon(o):return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
print("="*108);print("TEST 296 - CAUSAL READOUT BRIDGE INTERVENTION");print("="*108);print("START:",START)
print("[1/14] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
torch.cuda.synchronize();t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size;PDT=next(model.parameters()).dtype
if len(layers)!=TOTAL or H!=H_EXPECT or PDT!=torch.bfloat16:raise RuntimeError("Architecture/dtype mismatch")
torch.cuda.synchronize();print(f"OK | {MODEL_ID} | 28L | H={H} | {PDT} | {time.perf_counter()-t:.2f}s")
print("[2/14] WEIGHT SENTINEL AND FIXED RSS")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();CL=list(POOLS);CN=list(CTX);KEYS=[f"{c}:{x}" for c,p in POOLS.items() for x in p]
base=[IVME*(ZIRVE*np.exp(-SONUM*L)*(1+SONUM*L)+TABAN)/(ZIRVE+TABAN) for L in range(END+1)]
scale=TARGET_RSS/np.sqrt(sum(x*x for x in base));RHO=[x*scale for x in base];RSS=np.sqrt(sum(x*x for x in RHO))
print("FP:",[f"{x:.4f}" for x in FP0]);print(f"RSS={RSS:.9f} | RHO0={RHO[0]*100:.3f}% | RHO25={RHO[25]*100:.3f}% | BRIDGE_REL={BRIDGE_REL:.3f}")
def enc(x):
    s=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to(DEVICE)
@torch.inference_mode()
def capture(text):
    o=model(**enc(text),use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)]
print("[3/14] CACHE 360 ABSOLUTE STATES")
STATE={}
for cls,pool in POOLS.items():
    STATE[cls]={}
    for x in pool:STATE[cls][x]={c:capture(f.format(x=x)) for c,f in CTX.items()}
    print(cls,"READY")
print("[4/14] TEST285 EXTERNAL BANK FORGE")
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
                    h=STATE[cls][target][c][L]
                    RAW[k][bank][c].append(unit(torch.stack([unit(h-STATE[bank][r][c][L]) for r in POOLS[bank]]).mean(0)))
print("RAW READY")
print("[5/14] TEST285 BANK CLEAN")
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
print("[6/14] TEST286 CONTEXT CLEAN")
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
                CRES.setdefault(k,{}).setdefault(c,[None]*TOTAL);CRES[k][c][L]=unit(BC[k][c][L]-mu)
print("CONTEXT CLEAN READY")
print("[7/14] FIVE CONTEXT COMPILER")
PACK={k:[unit(torch.stack([CRES[k][c][L] for c in TRAIN]).mean(0)) for L in range(TOTAL)] for k in KEYS}
print("60 TARGET PACKETS READY")
print("[8/14] TARGET HEAD BRIDGE CONSTRUCTION")
W=model.lm_head.weight.detach().float();HEAD={};BRIDGE={}
for k in KEYS:
    target=k.split(":",1)[1];tid=tok.encode(target,add_special_tokens=False)[0];h=unit(W[tid]);HEAD[k]=(tid,h)
    p=PACK[k][26];a=torch.dot(p,h);BRIDGE[k]=unit(h*a)
print("60 FIXED TERMINAL BRIDGES READY")
print("[9/14] CAUSAL ENGINE")
AUD={"seasc_calls":0,"bridge_calls":0,"max_seasc_dev":0.0,"max_bridge_dev":0.0};ACTIVE=set()
def install(packet,bridge=None):
    hs=[]
    for L in range(END+1):
        rho=RHO[L]
        def hook(mod,args,out,L=L,rho=rho):
            y=out[0] if isinstance(out,tuple) else out;z=y[:,-1,:].float();d=packet[L].to(z.device)*z.norm(dim=-1,keepdim=True)*rho
            rel=float((d.norm(dim=-1)/(z.norm(dim=-1)+EPS)).max());AUD["seasc_calls"]+=1;AUD["max_seasc_dev"]=max(AUD["max_seasc_dev"],abs(rel-rho))
            yy=y.clone();yy[:,-1,:]=(z+d).to(y.dtype);return (yy,)+out[1:] if isinstance(out,tuple) else yy
        h=layers[L].register_forward_hook(hook);hs.append(h);ACTIVE.add(id(h))
    if bridge is not None:
        def bhook(mod,args,out):
            y=out[0] if isinstance(out,tuple) else out;z=y[:,-1,:].float();d=bridge.to(z.device)*z.norm(dim=-1,keepdim=True)*BRIDGE_REL
            rel=float((d.norm(dim=-1)/(z.norm(dim=-1)+EPS)).max());AUD["bridge_calls"]+=1;AUD["max_bridge_dev"]=max(AUD["max_bridge_dev"],abs(rel-BRIDGE_REL))
            yy=y.clone();yy[:,-1,:]=(z+d).to(y.dtype);return (yy,)+out[1:] if isinstance(out,tuple) else yy
        h=layers[26].register_forward_hook(bhook);hs.append(h);ACTIVE.add(id(h))
    return hs
@torch.inference_mode()
def run(packet=None,bridge=None,generate=False):
    hs=install(packet,bridge) if packet is not None else []
    try:
        e=enc(BLIND)
        if generate:
            o=model.generate(**e,max_new_tokens=8,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
            return tok.decode(o[0,e["input_ids"].shape[1]:],skip_special_tokens=True).strip()
        o=model(**e,use_cache=False,return_dict=True);return o.logits[0,-1].float().detach().clone()
    finally:
        for h in hs:h.remove();ACTIVE.discard(id(h))
LOG0=run()
print("[10/14] PACK-ONLY / OWN-BRIDGE / WRONG-BRIDGE")
ROWS=[]
for i,k in enumerate(KEYS):
    j=KEYS[(i+1)%len(KEYS)];tid,_=HEAD[k]
    lp=run(PACK[k]);lo=run(PACK[k],BRIDGE[k]);lw=run(PACK[k],BRIDGE[j])
    def rank(x):return int((x>x[tid]).sum())+1
    ROWS.append({"key":k,"wrong_bridge":j,"token_id":tid,"null_logit":float(LOG0[tid]),"pack_logit":float(lp[tid]),"own_logit":float(lo[tid]),"wrong_logit":float(lw[tid]),
    "pack_delta":float(lp[tid]-LOG0[tid]),"own_gain":float(lo[tid]-lp[tid]),"wrong_gain":float(lw[tid]-lp[tid]),
    "rank0":rank(LOG0),"rank_pack":rank(lp),"rank_own":rank(lo),"rank_wrong":rank(lw),
    "top1_pack":int(torch.argmax(lp))==tid,"top1_own":int(torch.argmax(lo))==tid,"top1_wrong":int(torch.argmax(lw))==tid})
    if (i+1)%10==0:print(f"{i+1}/60")
print("[11/14] CAUSAL BRIDGE SUMMARY")
def M(f):return float(np.mean([r[f] for r in ROWS]))
GLOBAL={"pack_delta":M("pack_delta"),"own_gain":M("own_gain"),"wrong_gain":M("wrong_gain"),
"own_specificity":M("own_gain")-M("wrong_gain"),"pack_rank_improved":float(np.mean([r["rank_pack"]<r["rank0"] for r in ROWS])),
"own_rank_improved_vs_pack":float(np.mean([r["rank_own"]<r["rank_pack"] for r in ROWS])),
"wrong_rank_improved_vs_pack":float(np.mean([r["rank_wrong"]<r["rank_pack"] for r in ROWS])),
"top1_pack":float(np.mean([r["top1_pack"] for r in ROWS])),"top1_own":float(np.mean([r["top1_own"] for r in ROWS])),"top1_wrong":float(np.mean([r["top1_wrong"] for r in ROWS])),
"median_rank_pack":float(np.median([r["rank_pack"] for r in ROWS])),"median_rank_own":float(np.median([r["rank_own"] for r in ROWS]))}
for k,v in GLOBAL.items():print(f"{k.upper():28s} {v:+.6f}")
print("[12/14] BLIND GREEDY BEHAVIOR")
BEH=[]
for i,k in enumerate(KEYS):
    j=KEYS[(i+1)%len(KEYS)];target=k.split(":",1)[1]
    a=run(PACK[k],generate=True);b=run(PACK[k],BRIDGE[k],generate=True);c=run(PACK[k],BRIDGE[j],generate=True)
    def norm(s):return "".join(ch.lower() for ch in s if ch.isalnum())
    nt=norm(target);BEH.append({"key":k,"pack":a,"own":b,"wrong":c,"pack_exact":norm(a)==nt,"own_exact":norm(b)==nt,"wrong_exact":norm(c)==nt})
    if (i+1)%10==0:print(f"{i+1}/60")
BG={"pack_exact":float(np.mean([x["pack_exact"] for x in BEH])),"own_exact":float(np.mean([x["own_exact"] for x in BEH])),"wrong_exact":float(np.mean([x["wrong_exact"] for x in BEH]))}
print(f"PACK EXACT={BG['pack_exact']:.3f} | OWN BRIDGE EXACT={BG['own_exact']:.3f} | WRONG BRIDGE EXACT={BG['wrong_exact']:.3f}")
print("[13/14] CLASS ATLAS")
ATLAS={}
for cls in POOLS:
    rr=[r for r in ROWS if r["key"].startswith(cls+":")]
    ATLAS[cls]={"pack_delta":float(np.mean([r["pack_delta"] for r in rr])),"own_gain":float(np.mean([r["own_gain"] for r in rr])),
    "wrong_gain":float(np.mean([r["wrong_gain"] for r in rr])),"top1_pack":float(np.mean([r["top1_pack"] for r in rr])),
    "top1_own":float(np.mean([r["top1_own"] for r in rr])),"median_rank_pack":float(np.median([r["rank_pack"] for r in rr])),
    "median_rank_own":float(np.median([r["rank_own"] for r in rr]))}
    a=ATLAS[cls];print(f"{cls:10s} | PACK={a['pack_delta']:+.3f} | OWN={a['own_gain']:+.3f} | WRONG={a['wrong_gain']:+.3f} | TOP1={a['top1_pack']:.3f}->{a['top1_own']:.3f} | RANK={a['median_rank_pack']:.1f}->{a['median_rank_own']:.1f}")
print("[14/14] INTEGRITY AND SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("Hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["max_seasc_dev"]>1e-6 or AUD["max_bridge_dev"]>1e-6:raise RuntimeError("Dose audit failure")
R={"schema":"akbascore.test296.v1","test":"TEST 296","start":START,"end":utc(),"model":MODEL_ID,
"question":"Can a fixed terminal target-head bridge causally convert the transported packet into stronger target-vocabulary and blind behavioral readout?",
"compiler":{"targets":60,"train_contexts":TRAIN,"external_banks":9},"seasc":{"on_layers":[0,25],"bridge_layer":26,"layer27":"OFF","rss":RSS,"bridge_rel":BRIDGE_REL},
"global":GLOBAL,"behavior":BG,"class_atlas":ATLAS,"rows":ROWS,"behavior_rows":BEH,
"integrity":{"training":False,"optimization":False,"dose_sweep":False,"weight_update":False,"seasc_calls":AUD["seasc_calls"],"bridge_calls":AUD["bridge_calls"],
"max_seasc_dev":AUD["max_seasc_dev"],"max_bridge_dev":AUD["max_bridge_dev"],"hooks_remaining":len(ACTIVE),"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
sha=hashlib.sha256(canon(R)).hexdigest();runid=f"T296-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{runid}.json";tp=ROOT/f"{runid}.txt"
jp.write_bytes(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2).encode())
out=["="*108,"TEST 296 - CAUSAL READOUT BRIDGE INTERVENTION","="*108]
for r in ROWS:out.append(f"{r['key']} | PACK_DL={r['pack_delta']:+.4f} | OWN_GAIN={r['own_gain']:+.4f} | WRONG_GAIN={r['wrong_gain']:+.4f} | RANK={r['rank_pack']}->{r['rank_own']} | TOP1={int(r['top1_pack'])}->{int(r['top1_own'])}")
out+=["",f"PACK_DELTA={GLOBAL['pack_delta']:+.6f}",f"OWN_GAIN={GLOBAL['own_gain']:+.6f}",f"WRONG_GAIN={GLOBAL['wrong_gain']:+.6f}",f"OWN_SPECIFICITY={GLOBAL['own_specificity']:+.6f}",f"TOP1_PACK={GLOBAL['top1_pack']:.6f}",f"TOP1_OWN={GLOBAL['top1_own']:.6f}",f"TOP1_WRONG={GLOBAL['top1_wrong']:.6f}",f"BEHAV_PACK={BG['pack_exact']:.6f}",f"BEHAV_OWN={BG['own_exact']:.6f}",f"BEHAV_WRONG={BG['wrong_exact']:.6f}",f"RSS={RSS:.9f}",f"BRIDGE_REL={BRIDGE_REL:.6f}",f"SEASC_CALLS={AUD['seasc_calls']}",f"BRIDGE_CALLS={AUD['bridge_calls']}","NO TRAINING - NO OPTIMIZATION - NO DOSE SWEEP - WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]
tp.write_text("\n".join(out),encoding="utf-8")
print("="*108);print("TEST 296 COMPLETE")
print(f"PACK DLOGIT={GLOBAL['pack_delta']:+.4f} | OWN BRIDGE GAIN={GLOBAL['own_gain']:+.4f} | WRONG={GLOBAL['wrong_gain']:+.4f} | SPEC={GLOBAL['own_specificity']:+.4f}")
print(f"TOP1 PACK={GLOBAL['top1_pack']:.3f} | OWN={GLOBAL['top1_own']:.3f} | WRONG={GLOBAL['top1_wrong']:.3f}")
print(f"BEHAVIOR EXACT PACK={BG['pack_exact']:.3f} | OWN={BG['own_exact']:.3f} | WRONG={BG['wrong_exact']:.3f}")
print(f"RSS={RSS:.9f} | BRIDGE_REL={BRIDGE_REL:.3f} | SEASC_CALLS={AUD['seasc_calls']} | BRIDGE_CALLS={AUD['bridge_calls']}")
print("NO TRAINING | NO OPTIMIZATION | NO DOSE SWEEP | WEIGHT INTEGRITY PASS | AKBASCORE HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
