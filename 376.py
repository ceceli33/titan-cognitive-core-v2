# ================================================================================================
# AKBASCORE - TEST 376
# FROZEN FRESH SOURCE-REMOVED PACKET VALIDATION
# CONTRIB_GEOM ROUTER + SOFT PACKET FROZEN FROM T375
# SOFT / HARD / SHUFFLED-PACKET CONTROL | SOURCE REMOVED | SEASC L0-L25 | L26-L27 OFF
# ================================================================================================
import os,sys,json,time,random,hashlib,subprocess,importlib.util,re,math
from datetime import datetime
from pathlib import Path
for m,p in [("torch","torch"),("transformers","transformers"),("numpy","numpy")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required")
SEED=376;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";TOTAL=28;H=3584;MOTOR=26;EPS=1e-12;MAXSEG=3
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20;MOTOR_SCALE=.975578
ROOT=Path("/content/AKBASCORE_TEST376");ROOT.mkdir(parents=True,exist_ok=True)
SOURCE="Near the Valora archive, Daren Voss placed the amber prism inside chamber 742 in 2046 after Mira Solen secured the silver tablet beside the Cordoba tower, while the reserve beacon remained completely inactive beneath the western gallery."
CASES=[
("PERSON","Who placed the amber prism?","Daren Voss"),
("OBJECT","What did Daren Voss place?","amber prism"),
("PLACE","Where did Mira Solen secure the silver tablet?","Cordoba tower"),
("NUMBER","Inside which chamber was the amber prism placed?","742"),
("TIME","When was the amber prism placed?","2046"),
("PERSON","Who secured the silver tablet?","Mira Solen"),
("STATE","What state did the reserve beacon remain in?","completely inactive"),
("OBJECT","What did Mira Solen secure?","silver tablet"),
("PLACE","Where did the reserve beacon remain?","western gallery")]
def py(x):
    if isinstance(x,dict):return {str(k):py(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [py(v) for v in x]
    if isinstance(x,np.ndarray):return py(x.tolist())
    if isinstance(x,(np.integer,np.bool_)):return x.item()
    if isinstance(x,np.floating):return float(x)
    return x
def canon(x):return json.dumps(py(x),sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
print("="*106);print("TEST 376 - FROZEN FRESH SOURCE-REMOVED PACKET VALIDATION");print("="*106)
t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="eager",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers
if len(layers)!=TOTAL or model.config.hidden_size!=H:raise RuntimeError("architecture")
NH=int(model.config.num_attention_heads);NKV=int(model.config.num_key_value_heads);HD=int(getattr(model.config,"head_dim",H//NH));REP=NH//NKV
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();rho=[]
for L in range(MOTOR):
    x=ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN
    rho.append(MOTOR_SCALE*IVME*x/(ZIRVE+TABAN))
RSS=sum(x*x for x in rho)**.5
print(f"MODEL OK | 28L H={H} Q={NH} KV={NKV} REP={REP} | MOTOR=L0-L25 RSS={RSS:.9f} | {time.perf_counter()-t:.1f}s")
print("FP="+",".join(f"{x:.4f}" for x in FP0))
WM=list(re.finditer(r"\b[\w'-]+\b",SOURCE,re.UNICODE));NW=len(WM);SEG=[]
for s in range(NW):
    for n in range(1,MAXSEG+1):
        if s+n<=NW:SEG.append({"text":SOURCE[WM[s].start():WM[s+n-1].end()],"ids":list(range(s,s+n))})
NS=len(SEG);DATA=[]
for typ,q,_ in CASES:
    text=f"SOURCE:\n{SOURCE}\n\nQUESTION:\n{q}\n\nANSWER:"
    e=tok(text,return_tensors="pt",add_special_tokens=False,return_offsets_mapping=True);off=e["offset_mapping"][0].tolist();s0=len("SOURCE:\n");q0=text.index(q);q1=q0+len(q)
    qt=[j for j,(a,b) in enumerate(off) if b>q0 and a<q1];wt=[]
    for m in WM:
        a,b=s0+m.start(),s0+m.end();ix=[j for j,(x,y) in enumerate(off) if y>a and x<b]
        if not ix:raise RuntimeError("word span")
        wt.append(ix)
    if not qt:raise RuntimeError("question span")
    DATA.append({"ids":e["input_ids"].cuda(),"mask":e["attention_mask"].cuda(),"q":qt[-1],"wt":wt})
N=len(CASES);CON=np.zeros((N,TOTAL,NH,NW),np.float64);VEC=[[None]*NW for _ in range(N)]
print(f"FRESH DATA | Q={N} WORDS={NW} SEGMENTS={NS} | extracting frozen native contributions...")
with torch.inference_mode():
    for i,d in enumerate(DATA):
        o=model(input_ids=d["ids"],attention_mask=d["mask"],use_cache=False,output_attentions=True,output_hidden_states=True,return_dict=True)
        acc=[torch.zeros(H,device="cuda",dtype=torch.float32) for _ in range(NW)]
        for L in range(TOTAL):
            att=o.attentions[L][0,:,d["q"],:].float();sa=layers[L].self_attn
            x=layers[L].input_layernorm(o.hidden_states[L][0]).to(sa.v_proj.weight.dtype)
            V=sa.v_proj(x).view(x.shape[0],NKV,HD).transpose(0,1).repeat_interleave(REP,dim=0).float();WO=sa.o_proj.weight.float()
            for j,ix in enumerate(d["wt"]):
                c=(att[:,ix].unsqueeze(-1)*V[:,ix,:]).sum(1);hv=[]
                for h in range(NH):
                    z=WO[:,h*HD:(h+1)*HD]@c[h];CON[i,L,h,j]=z.norm().item();hv.append(z)
                if L<MOTOR:acc[j]+=torch.stack(hv).sum(0)
        for j in range(NW):VEC[i][j]=(acc[j]/(acc[j].norm()+EPS)).detach()
        del o,acc;torch.cuda.empty_cache()
PC=CON/(CON.sum(3,keepdims=True)+EPS);DC=np.empty_like(PC)
for i in range(N):DC[i]=PC[i]-np.mean(np.delete(PC,i,axis=0),axis=0)
R=np.empty(DC.shape,np.int16)
for i in range(N):
    for L in range(TOTAL):
        for h in range(NH):
            o=np.argsort(-DC[i,L,h]);R[i,L,h,o]=np.arange(1,NW+1)
W=np.mean(1.0/R,axis=(1,2));SG=np.zeros((N,NS))
for i in range(N):
    for k,g in enumerate(SEG):SG[i,k]=float(np.exp(np.mean(np.log(W[i,g["ids"]]+EPS))))
SOFT=[];HARD=[];ROUTE=[]
for i in range(N):
    order=np.argsort(-SG[i]);ROUTE.append(SEG[order[0]]["text"])
    def sv(k):
        z=torch.stack([VEC[i][j] for j in SEG[k]["ids"]]).mean(0);return z/(z.norm()+EPS)
    HARD.append(sv(order[0]))
    vv=torch.stack([sv(k) for k in order]);ww=torch.tensor(SG[i,order],device="cuda",dtype=torch.float32)
    z=(vv*ww[:,None]).sum(0)/(ww.sum()+EPS);SOFT.append(z/(z.norm()+EPS))
SHUFFLE=[SOFT[(i+1)%N] for i in range(N)]
print("PACKETS READY | SOFT frozen | HARD control | cyclic SHUFFLE negative control")
def qinput(q):
    e=tok(f"QUESTION:\n{q}\n\nANSWER:",return_tensors="pt",add_special_tokens=False)
    return {k:v.cuda() for k,v in e.items()}
@torch.inference_mode()
def logits(q,vec=None):
    e=qinput(q);hooks=[]
    if vec is not None:
        for L in range(MOTOR):
            def hk(mod,inp,out,L=L):
                y=out[0] if isinstance(out,tuple) else out;z=y[:,-1,:].float();d=vec*z.norm(dim=-1,keepdim=True)*rho[L];y2=y.clone();y2[:,-1,:]=(z+d).to(y.dtype)
                return (y2,)+out[1:] if isinstance(out,tuple) else y2
            hooks.append(layers[L].register_forward_hook(hk))
    o=model(**e,use_cache=False,return_dict=True);lg=o.logits[0,-1].float()
    for h in hooks:h.remove()
    return lg.cpu()
BASE=[];OUT={"SOFT":[],"HARD":[],"SHUFFLE":[]}
for i,(_,q,_) in enumerate(CASES):
    BASE.append(logits(q))
    OUT["SOFT"].append(logits(q,SOFT[i]));OUT["HARD"].append(logits(q,HARD[i]));OUT["SHUFFLE"].append(logits(q,SHUFFLE[i]))
print("SOURCE REMOVED | readouts complete | opening targets post-hoc")
RES={}
for name in OUT:
    rows=[];dg=[];ri=[]
    for i,(_,q,val) in enumerate(CASES):
        ids=tok.encode(val,add_special_tokens=False)
        if not ids:raise RuntimeError("target")
        tid=ids[0];b=BASE[i];s=OUT[name][i];rb=1+int((b>b[tid]).sum());rs=1+int((s>s[tid]).sum());g=float(s[tid]-b[tid])
        dg.append(g);ri.append(rs<rb);rows.append({"i":i+1,"value":val,"route":ROUTE[i],"base":rb,"steer":rs,"gain":g})
    RES[name]={"mean_gain":float(np.mean(dg)),"median_gain":float(np.median(dg)),"positive":float(np.mean(np.array(dg)>0)),"rank_improved":float(np.mean(ri)),"rows":rows}
print("-"*106);print("FRESH SUMMARY")
for n,r in RES.items():print(f"{n:7s} | dLOGIT={r['mean_gain']:+.3f} | MED={r['median_gain']:+.3f} | POS={r['positive']:.3f} | RANK+={r['rank_improved']:.3f}")
print("-"*106);print("SOFT FRESH CASES")
for x in RES["SOFT"]["rows"]:print(f"{x['i']:02d} | {x['value']!r:22s} | ROUTE={x['route']!r:20s} | R{x['base']}->{x['steer']} | dL={x['gain']:+.3f}")
S=RES["SOFT"];C=RES["SHUFFLE"]
REPLICATED=bool(S["positive"]>=.75 and S["rank_improved"]>=.55 and S["mean_gain"]>0)
SPECIFIC=bool(S["mean_gain"]>C["mean_gain"] and S["rank_improved"]>=C["rank_improved"])
STRONG=bool(REPLICATED and SPECIFIC and S["positive"]>=.85 and S["rank_improved"]>=.70 and S["mean_gain"]>=1.0)
DECISION="PACKET_TRANSPORT_REPLICATED_MOVE_TO_BEHAVIORAL_BOUNDARY" if REPLICATED and SPECIFIC else "PACKET_TRANSPORT_NOT_FROZEN"
print("-"*106);print(f"GATE | REPLICATED={REPLICATED} | SPECIFIC_vs_SHUFFLE={SPECIFIC} | STRONG={STRONG}");print("DECISION =",DECISION)
FP1=fp()
if FP1!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("integrity")
payload={"schema":"akbascore.test376.v1","results":RES,"replicated":REPLICATED,"specific":SPECIFIC,"strong":STRONG,"decision":DECISION,"motor":{"layers":"L0-L25","scale":MOTOR_SCALE,"rss":RSS},"audit":{"fresh_source":True,"t375_soft_frozen":True,"contrib_geom_frozen":True,"hard_control":True,"cyclic_shuffle_control":True,"source_removed_readout":True,"target_engine_access":False,"target_posthoc_only":True,"transductive_router":True,"training":False,"weights_frozen":True,"fp_match":True}}
sha=hashlib.sha256(canon(payload)).hexdigest();rid=f"T376-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json"
jp.write_text(json.dumps(py(payload),ensure_ascii=True,indent=2,allow_nan=False),encoding="utf-8")
print(f"FP MATCH | JSON={jp}");print("SHA =",sha);print("="*106)
