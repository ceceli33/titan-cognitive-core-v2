# ================================================================================================
# AKBASCORE - TEST 378 FINAL
# GOLDEN GATE FINAL BLIND RETRIEVAL
# FROZEN CONTRIB_GEOM -> SOFT PACKET -> SOURCE REMOVED -> SEASC L0-L25 -> GREEDY GENERATION
# VANILLA / CORRECT PACKET / SHUFFLED PACKET | NO TRAINING | TARGET POST-HOC ONLY
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
SEED=378;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";TOTAL=28;H=3584;MOTOR=26;EPS=1e-12;MAXSEG=3;MAXNEW=16
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20;MOTOR_SCALE=.975578
ROOT=Path("/content/AKBASCORE_TEST378_FINAL");ROOT.mkdir(parents=True,exist_ok=True)
SOURCE="Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge."
CASES=[
("Who planted the Turkish flag at the base of the Golden Gate Bridge?","Mustafa Akbaş"),
("What did Mustafa Akbaş do at the base of the Golden Gate Bridge?","planted the Turkish flag"),
("What did Mustafa Akbaş plant at the base of the Golden Gate Bridge?","Turkish flag"),
("Where did Mustafa Akbaş plant the Turkish flag?","base of the Golden Gate Bridge")]
def py(x):
    if isinstance(x,dict):return {str(k):py(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [py(v) for v in x]
    if isinstance(x,np.ndarray):return py(x.tolist())
    if isinstance(x,(np.integer,np.bool_)):return x.item()
    if isinstance(x,np.floating):return float(x)
    return x
def canon(x):return json.dumps(py(x),sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
def norm(s):return re.sub(r"[^\wğüşöçıİĞÜŞÖÇ]+"," ",s.lower()).strip()
def mention(out,target):return norm(target) in norm(out)
print("="*104);print("TEST 378 FINAL - GOLDEN GATE BLIND RETRIEVAL");print("="*104)
t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="eager",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers
if len(layers)!=TOTAL or model.config.hidden_size!=H:raise RuntimeError("architecture mismatch")
NH=int(model.config.num_attention_heads);NKV=int(model.config.num_key_value_heads);HD=int(getattr(model.config,"head_dim",H//NH));REP=NH//NKV
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();rho=[]
for L in range(MOTOR):
    x=ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN;rho.append(MOTOR_SCALE*IVME*x/(ZIRVE+TABAN))
RSS=sum(x*x for x in rho)**.5
print(f"MODEL OK | 28L H={H} Q={NH} KV={NKV} REP={REP} | L0-L25 RSS={RSS:.9f} | {time.perf_counter()-t:.1f}s")
print("FP="+",".join(f"{x:.4f}" for x in FP0))
WM=list(re.finditer(r"\b[\w'-]+\b",SOURCE,re.UNICODE));NW=len(WM);SEG=[]
for s in range(NW):
    for n in range(1,MAXSEG+1):
        if s+n<=NW:SEG.append({"text":SOURCE[WM[s].start():WM[s+n-1].end()],"ids":list(range(s,s+n))})
NS=len(SEG);DATA=[]
for q,_ in CASES:
    text=f"SOURCE:\n{SOURCE}\n\nQUESTION:\n{q}\n\nANSWER:"
    e=tok(text,return_tensors="pt",add_special_tokens=False,return_offsets_mapping=True);off=e["offset_mapping"][0].tolist()
    s0=len("SOURCE:\n");q0=text.index(q);q1=q0+len(q);qt=[j for j,(a,b) in enumerate(off) if b>q0 and a<q1];wt=[]
    for m in WM:
        a,b=s0+m.start(),s0+m.end();ix=[j for j,(x,y) in enumerate(off) if y>a and x<b]
        if not ix:raise RuntimeError("word span")
        wt.append(ix)
    DATA.append({"ids":e["input_ids"].cuda(),"mask":e["attention_mask"].cuda(),"q":qt[-1],"wt":wt})
N=len(CASES);CON=np.zeros((N,TOTAL,NH,NW),np.float64);VEC=[[None]*NW for _ in range(N)]
print(f"COMPILER | Q={N} WORDS={NW} SEGMENTS={NS} | frozen CONTRIB_GEOM -> SOFT...")
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
            order=np.argsort(-DC[i,L,h]);R[i,L,h,order]=np.arange(1,NW+1)
W=np.mean(1.0/R,axis=(1,2));SG=np.zeros((N,NS))
for i in range(N):
    for k,g in enumerate(SEG):SG[i,k]=float(np.exp(np.mean(np.log(W[i,g["ids"]]+EPS))))
SOFT=[];ROUTE=[]
for i in range(N):
    order=np.argsort(-SG[i]);ROUTE.append(SEG[order[0]]["text"])
    vv=[];ww=[]
    for k in order:
        z=torch.stack([VEC[i][j] for j in SEG[k]["ids"]]).mean(0);z=z/(z.norm()+EPS);vv.append(z);ww.append(SG[i,k])
    vv=torch.stack(vv);ww=torch.tensor(ww,device="cuda",dtype=torch.float32);z=(vv*ww[:,None]).sum(0)/(ww.sum()+EPS);SOFT.append((z/(z.norm()+EPS)).detach())
SHUFFLE=[SOFT[(i+1)%N] for i in range(N)]
print("PACKETS READY | SOURCE REMOVED | generating VANILLA / CORRECT / SHUFFLE")
def hooks(vec):
    hs=[]
    for L in range(MOTOR):
        def hk(mod,inp,out,L=L):
            y=out[0] if isinstance(out,tuple) else out;z=y[:,-1,:].float();d=vec*z.norm(dim=-1,keepdim=True)*rho[L]
            y2=y.clone();y2[:,-1,:]=(z+d).to(y.dtype)
            return (y2,)+out[1:] if isinstance(out,tuple) else y2
        hs.append(layers[L].register_forward_hook(hk))
    return hs
@torch.inference_mode()
def gen(q,vec=None):
    e=tok(f"QUESTION:\n{q}\n\nANSWER:",return_tensors="pt",add_special_tokens=False);e={k:v.cuda() for k,v in e.items()};hs=hooks(vec) if vec is not None else []
    o=model.generate(**e,max_new_tokens=MAXNEW,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
    for h in hs:h.remove()
    return tok.decode(o[0,e["input_ids"].shape[1]:],skip_special_tokens=True).strip()
ROWS=[]
for i,(q,target) in enumerate(CASES):
    v=gen(q);c=gen(q,SOFT[i]);s=gen(q,SHUFFLE[i])
    ROWS.append({"i":i+1,"q":q,"target":target,"route":ROUTE[i],"vanilla":v,"correct":c,"shuffle":s,"v_hit":mention(v,target),"c_hit":mention(c,target),"s_hit":mention(s,target)})
VH=float(np.mean([x["v_hit"] for x in ROWS]));CH=float(np.mean([x["c_hit"] for x in ROWS]));SH=float(np.mean([x["s_hit"] for x in ROWS]))
print("-"*104);print(f"FINAL SCORE | VANILLA={VH:.3f} | CORRECT_PACKET={CH:.3f} | SHUFFLE={SH:.3f}")
print("-"*104)
for x in ROWS:
    print(f"{x['i']:02d} TARGET={x['target']!r} | ROUTE={x['route']!r}")
    print(f"   VANILLA : {x['vanilla']!r}")
    print(f"   CORRECT : {x['correct']!r}")
    print(f"   SHUFFLE : {x['shuffle']!r}")
BEHAVIOR=bool(CH>VH and CH>SH and CH>=.50)
if BEHAVIOR:DECISION="FINAL_BEHAVIORAL_RETRIEVAL_OBSERVED"
elif CH>VH:DECISION="FINAL_PARTIAL_BEHAVIORAL_EFFECT_NO_RELIABLE_RETRIEVAL"
else:DECISION="FINAL_LATENT_TRANSPORT_WITHOUT_BEHAVIORAL_RETRIEVAL"
print("-"*104);print("DECISION =",DECISION)
FP1=fp()
if FP1!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("integrity")
payload={"schema":"akbascore.test378.final","source":SOURCE,"rows":ROWS,"scores":{"vanilla":VH,"correct_packet":CH,"shuffle":SH},"decision":DECISION,"motor":{"layers":"L0-L25","scale":MOTOR_SCALE,"rss":RSS},"audit":{"contrib_geom_frozen":True,"soft_packet_frozen":True,"source_removed_generation":True,"target_engine_access":False,"target_posthoc_only":True,"shuffle_control":True,"transductive_router":True,"training":False,"weights_frozen":True,"fp_match":True}}
sha=hashlib.sha256(canon(payload)).hexdigest();rid=f"T378-FINAL-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json"
jp.write_text(json.dumps(py(payload),ensure_ascii=True,indent=2,allow_nan=False),encoding="utf-8")
print(f"FP MATCH | JSON={jp}");print("SHA =",sha);print("="*104)
