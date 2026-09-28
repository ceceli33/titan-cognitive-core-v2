# ================================================================================================
# AKBASCORE - TEST 297 - FIXED
# TARGET DECISION-BOUNDARY MARGIN ASSAY
# PACK-ONLY vs OWN-BRIDGE vs WRONG-BRIDGE
# TEST296 WORKING BASELINE - FIXED RSS - FIXED BRIDGE_REL=0.020
# NO GENERATION - NO TRAINING - NO OPTIMIZATION - NO DOSE SWEEP
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
SEED=297
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda");MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
BLIND="Return the single value represented by the internal state. Answer with only the value."
TOTAL=28;H_EXPECT=3584;EPS=1e-8;END=25;TARGET_RSS=.250235055
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20;BRIDGE_REL=.020
ROOT=Path("/content/AKBASCORE_TEST297") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST297")
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
print("="*108);print("TEST 297 - TARGET DECISION-BOUNDARY MARGIN ASSAY - FIXED");print("="*108);print("START:",START)
print("[1/13] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
torch.cuda.synchronize();t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size;PDT=next(model.parameters()).dtype
if len(layers)!=TOTAL or H!=H_EXPECT or PDT!=torch.bfloat16:raise RuntimeError("Architecture/dtype mismatch")
torch.cuda.synchronize();print(f"OK | {MODEL_ID} | 28L | H={H} | {PDT} | {time.perf_counter()-t:.2f}s")
print("[2/13] WEIGHT SENTINEL AND FIXED RSS")
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
                    h=STATE[cls][target][c][L]
                    RAW[k][bank][c].append(unit(torch.stack([unit(h-STATE[bank][r][c][L]) for r in POOLS[bank]]).mean(0)))
print("RAW READY")
print("[5/13] TEST285 BANK CLEAN")
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
print("[6/13] TEST286 CONTEXT CLEAN")
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
print("[7/13] FIVE CONTEXT COMPILER")
PACK={k:[unit(torch.stack([CRES[k][c][L] for c in TRAIN]).mean(0)) for L in range(TOTAL)] for k in KEYS}
print("60 TARGET PACKETS READY")
print("[8/13] FIXED READOUT BRIDGES")
W=model.lm_head.weight.detach().float();HEAD={};BRIDGE={}
for k in KEYS:
    target=k.split(":",1)[1];ids=tok.encode(target,add_special_tokens=False);tid=ids[0];h=unit(W[tid]);HEAD[k]=(tid,h,ids)
    a=torch.dot(PACK[k][26],h);BRIDGE[k]=unit(h*a)
print("60 BRIDGES READY")
print("[9/13] CAUSAL ENGINE")
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
def run(packet=None,bridge=None):
    hs=install(packet,bridge) if packet is not None else []
    try:return model(**enc(BLIND),use_cache=False,return_dict=True).logits[0,-1].float().detach().clone()
    finally:
        for h in hs:h.remove();ACTIVE.discard(id(h))
LOG0=run()
print("[10/13] DECISION-BOUNDARY ASSAY")
ROWS=[]
for i,k in enumerate(KEYS):
    wrong=KEYS[(i+1)%len(KEYS)];tid,_,ids=HEAD[k];lp=run(PACK[k]);lo=run(PACK[k],BRIDGE[k]);lw=run(PACK[k],BRIDGE[wrong])
    def assay(x):
        tv=float(x[tid]);mask=x.clone();mask[tid]=-float("inf");cid=int(torch.argmax(mask));cv=float(x[cid])
        return tv,cid,tok.decode([cid]),cv,tv-cv,int((x>x[tid]).sum())+1
    p=assay(lp);o=assay(lo);w=assay(lw);need=max(0.0,-p[4]);gain=o[0]-p[0]
    ROWS.append({"key":k,"target":k.split(":",1)[1],"token_ids":ids,"wrong_bridge":wrong,
    "pack_target":p[0],"pack_competitor_id":p[1],"pack_competitor":p[2],"pack_competitor_logit":p[3],"pack_margin":p[4],"pack_rank":p[5],
    "own_target":o[0],"own_competitor_id":o[1],"own_competitor":o[2],"own_competitor_logit":o[3],"own_margin":o[4],"own_rank":o[5],
    "wrong_target":w[0],"wrong_margin":w[4],"wrong_rank":w[5],"own_target_gain":gain,"required_pack_gain":need,
    "gain_over_required":float(gain/need) if need>EPS else None,"margin_gain":o[4]-p[4],
    "boundary_crossed":bool(p[4]<=0 and o[4]>0),"top1_pack":bool(p[4]>0),"top1_own":bool(o[4]>0),"top1_wrong":bool(w[4]>0)})
    if (i+1)%10==0:print(f"{i+1}/60")
print("[11/13] GLOBAL MARGIN SUMMARY")
rat=[r["gain_over_required"] for r in ROWS if r["gain_over_required"] is not None]
GLOBAL={"mean_pack_margin":float(np.mean([r["pack_margin"] for r in ROWS])),"median_pack_margin":float(np.median([r["pack_margin"] for r in ROWS])),
"mean_own_margin":float(np.mean([r["own_margin"] for r in ROWS])),"median_own_margin":float(np.median([r["own_margin"] for r in ROWS])),
"mean_wrong_margin":float(np.mean([r["wrong_margin"] for r in ROWS])),"mean_target_gain":float(np.mean([r["own_target_gain"] for r in ROWS])),
"mean_margin_gain":float(np.mean([r["margin_gain"] for r in ROWS])),"median_required_gain":float(np.median([r["required_pack_gain"] for r in ROWS])),
"median_gain_over_required":float(np.median(rat)) if rat else None,"boundary_crossed":float(np.mean([r["boundary_crossed"] for r in ROWS])),
"top1_pack":float(np.mean([r["top1_pack"] for r in ROWS])),"top1_own":float(np.mean([r["top1_own"] for r in ROWS])),
"top1_wrong":float(np.mean([r["top1_wrong"] for r in ROWS])),"median_rank_pack":float(np.median([r["pack_rank"] for r in ROWS])),
"median_rank_own":float(np.median([r["own_rank"] for r in ROWS]))}
for k,v in GLOBAL.items():print(f"{k.upper():28s} {v:+.6f}" if v is not None else f"{k.upper():28s} N/A")
print("[12/13] CLASS MARGIN ATLAS")
ATLAS={}
for cls in POOLS:
    rr=[r for r in ROWS if r["key"].startswith(cls+":")]
    ATLAS[cls]={"pack_margin":float(np.mean([r["pack_margin"] for r in rr])),"own_margin":float(np.mean([r["own_margin"] for r in rr])),
    "target_gain":float(np.mean([r["own_target_gain"] for r in rr])),"margin_gain":float(np.mean([r["margin_gain"] for r in rr])),
    "required_gain":float(np.median([r["required_pack_gain"] for r in rr])),"crossed":float(np.mean([r["boundary_crossed"] for r in rr])),
    "rank_pack":float(np.median([r["pack_rank"] for r in rr])),"rank_own":float(np.median([r["own_rank"] for r in rr]))}
    a=ATLAS[cls];print(f"{cls:10s} | M0={a['pack_margin']:+.3f} | M1={a['own_margin']:+.3f} | TG={a['target_gain']:+.3f} | MG={a['margin_gain']:+.3f} | NEED={a['required_gain']:.3f} | CROSS={a['crossed']:.3f} | RANK={a['rank_pack']:.1f}->{a['rank_own']:.1f}")
print("[13/13] INTEGRITY AND SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("Hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["max_seasc_dev"]>1e-6 or AUD["max_bridge_dev"]>1e-6:raise RuntimeError("Dose audit failure")
R={"schema":"akbascore.test297.v2","test":"TEST 297 FIXED","start":START,"end":utc(),"model":MODEL_ID,
"question":"How far is each target from the actual Top-1 decision boundary, and how much of that deficit does the fixed causal bridge close?",
"compiler":{"targets":60,"train_contexts":TRAIN,"external_banks":9},"seasc":{"on_layers":[0,25],"bridge_layer":26,"layer27":"OFF","rss":RSS,"bridge_rel":BRIDGE_REL},
"global":GLOBAL,"class_atlas":ATLAS,"rows":ROWS,
"integrity":{"generation":False,"training":False,"optimization":False,"dose_sweep":False,"weight_update":False,
"seasc_calls":AUD["seasc_calls"],"bridge_calls":AUD["bridge_calls"],"max_seasc_dev":AUD["max_seasc_dev"],"max_bridge_dev":AUD["max_bridge_dev"],
"hooks_remaining":len(ACTIVE),"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
sha=hashlib.sha256(canon(R)).hexdigest();runid=f"T297-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{runid}.json";tp=ROOT/f"{runid}.txt"
jp.write_bytes(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False).encode())
out=["="*108,"TEST 297 - TARGET DECISION-BOUNDARY MARGIN ASSAY - FIXED","="*108]
for r in ROWS:out.append(f"{r['key']} | COMP={r['pack_competitor']!r} | M0={r['pack_margin']:+.4f} | M1={r['own_margin']:+.4f} | TG={r['own_target_gain']:+.4f} | MG={r['margin_gain']:+.4f} | NEED={r['required_pack_gain']:.4f} | RANK={r['pack_rank']}->{r['own_rank']} | CROSS={int(r['boundary_crossed'])}")
out+=["",f"MEAN_PACK_MARGIN={GLOBAL['mean_pack_margin']:+.6f}",f"MEDIAN_PACK_MARGIN={GLOBAL['median_pack_margin']:+.6f}",
f"MEAN_OWN_MARGIN={GLOBAL['mean_own_margin']:+.6f}",f"MEDIAN_OWN_MARGIN={GLOBAL['median_own_margin']:+.6f}",
f"MEAN_TARGET_GAIN={GLOBAL['mean_target_gain']:+.6f}",f"MEAN_MARGIN_GAIN={GLOBAL['mean_margin_gain']:+.6f}",
f"MEDIAN_REQUIRED_GAIN={GLOBAL['median_required_gain']:.6f}",f"MEDIAN_GAIN_OVER_REQUIRED={GLOBAL['median_gain_over_required']:.6f}" if GLOBAL["median_gain_over_required"] is not None else "MEDIAN_GAIN_OVER_REQUIRED=N/A",
f"BOUNDARY_CROSSED={GLOBAL['boundary_crossed']:.6f}",f"TOP1_PACK={GLOBAL['top1_pack']:.6f}",f"TOP1_OWN={GLOBAL['top1_own']:.6f}",
f"TOP1_WRONG={GLOBAL['top1_wrong']:.6f}",f"MEDIAN_RANK={GLOBAL['median_rank_pack']:.1f}->{GLOBAL['median_rank_own']:.1f}",
f"RSS={RSS:.9f}",f"BRIDGE_REL={BRIDGE_REL:.6f}",f"SEASC_CALLS={AUD['seasc_calls']}",f"BRIDGE_CALLS={AUD['bridge_calls']}",
"NO GENERATION - NO TRAINING - NO OPTIMIZATION - NO DOSE SWEEP - WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]
tp.write_text("\n".join(out),encoding="utf-8")
print("="*108);print("TEST 297 COMPLETE")
print(f"PACK MARGIN mean={GLOBAL['mean_pack_margin']:+.3f} median={GLOBAL['median_pack_margin']:+.3f}")
print(f"OWN MARGIN mean={GLOBAL['mean_own_margin']:+.3f} median={GLOBAL['median_own_margin']:+.3f}")
print(f"TARGET GAIN={GLOBAL['mean_target_gain']:+.3f} | MARGIN GAIN={GLOBAL['mean_margin_gain']:+.3f} | MEDIAN NEED={GLOBAL['median_required_gain']:.3f}")
print(f"BOUNDARY CROSSED={GLOBAL['boundary_crossed']:.3f} | TOP1 PACK={GLOBAL['top1_pack']:.3f} | OWN={GLOBAL['top1_own']:.3f} | WRONG={GLOBAL['top1_wrong']:.3f}")
print(f"MEDIAN RANK={GLOBAL['median_rank_pack']:.1f}->{GLOBAL['median_rank_own']:.1f}")
print(f"RSS={RSS:.9f} | BRIDGE_REL={BRIDGE_REL:.3f} | SEASC_CALLS={AUD['seasc_calls']} | BRIDGE_CALLS={AUD['bridge_calls']}")
print("NO GENERATION | NO TRAINING | NO OPTIMIZATION | NO DOSE SWEEP | WEIGHT INTEGRITY PASS | AKBASCORE HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
