# AKBASCORE NIRVANA · COGNITIVE CARTRIDGE — public demonstration, single Colab cell (A100)
# ENGINE (unchanged, transplanted from TEST461 / 463.final.py, the sealed final held-out engine):
#   Qwen/Qwen2.5-7B-Instruct · BF16 · SDPA · frozen weights · fixed 32-sentence neutral PCA codebook per layer × KV head
#   K120 / V128 / OWN (source-own slot 0 kept) · independently forged cartridges · RoPE install · DynamicCache
#   IBR = Isolated Batched Read (each cartridge = one batch row, left-padded, module-local positions) · Stage-1 MW1 / Stage-2 MW2
#   decide_id / decide_place / parse_place exactly as TEST461 · greedy · NO training · NO LoRA · NO optimizer · NO weight update · NO hooks
# COMPUTE PATH (verified from code): all GPU math runs through PyTorch's CUDA backend (cuBLAS GEMMs, PyTorch SDPA attention kernels).
#   This engine contains NO custom CUDA/C++ kernel; posters label it "PYTORCH CUDA GPU COMPUTE".
# ADDED FOR THE DEMO (measurement / presentation only, no change to engine math): per-cartridge telemetry (reconstruction cosine,
#   code norms, timings, GPU memory), operation counters, locked 16-cartridge demo panel, 16 automatic questions, NOMEM control,
#   18 JPEG posters, sealed payload. UI / JPEG / download / manifest / ZIP / SHA-256 / audit infrastructure: AKBASCORE_d120.py.
# Historical results (TEST460/461/462/464) are shown as SEALED EXPERIMENTAL RECORD constants, never as this run's result.
import os,sys,io,re,json,math,time,html,random,shutil,hashlib,zipfile,platform,textwrap,threading,subprocess,importlib.util,traceback
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import quote
from collections import Counter
#<<CONST_BEGIN>>
SEED=461;MAX_NEW=32;K_DIM=120;V_DIM=128;DMAX=128
FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n";STYLE=" Give only the answer on the first line; do not explain."
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";MODEL_SHORT="Qwen2.5-7B-Instruct"
TOTAL_LAYERS=28;H_EXPECT=3584;NH_EXPECT=28;NKV_EXPECT=4;HD_EXPECT=128
CORPUS=["Jonas Weber carried the wooden crate across the quiet market square.","Priya Nair wrote a long letter to her cousin in the morning.","The small boat drifted slowly toward the rocky shore.","Omar Haddad fixed the broken clock in the village school.","A tired teacher closed the green door of the library.","Sofia Rossi baked fresh bread for the harvest festival.","The children watched the kites rising above the hill.","Liam O'Connor sold his old bicycle to a neighbor.","Heavy rain flooded the narrow street near the market.","Nadia Petrova translated the ancient manuscript into French.","The farmer counted the sheep before sunset.","Hiro Sato opened a tiny bakery beside the river.","An old dog slept under the kitchen table all afternoon.","Carlos Mendes washed the windows of his grandmother's house.","The museum guard locked the heavy gate at midnight.","Fatima Zahra grew tomatoes in the backyard garden.","The pilot announced a short delay because of fog.","Anna Kowalski found a lost wallet on the bus.","Snow covered the mountain village during the night.","Ravi Kumar taught his brother how to play chess.","The chef sharpened every knife before dinner service.","Lucas Martin cleaned the roof of the barn after the storm.","A young violinist practiced scales in the empty hall.","Mei Lin delivered the package to the wrong address.","Marek Novak placed the glass bottle beneath the wooden bench.","Sara Ibrahim carried a red notebook into the quiet classroom.","Noah Schmidt repaired the small radio beside the kitchen window.","Yuki Mori left a paper envelope near the station entrance.","Amira Hassan moved the ceramic bowl onto the upper shelf.","Peter Novak opened the metal box behind the old theater.","Lucia Costa placed the yellow scarf inside the travel bag.","Daniel Kim carried a black umbrella through the central courtyard."]
FAM={"F1":("The {O} sits in container {T}.","Container {T} is located at the {P}."),
     "F2":("Container {T} holds the {O}.","The {P} houses container {T}."),
     "F3":("Inside container {T} rests the {O}.","Container {T} can be found at the {P}."),
     "F4":("The {O} is kept within container {T}.","At the {P}, container {T} is stored.")}
MW1="Which container holds the {obj}? If this record does not say which container holds the {obj}, answer NONE."+STYLE
MW2="Where is container {cid} located according to this record? Give the place name. If this record does not say where container {cid} is, answer NONE."+STYLE
# LOCKED DEMO PANEL (fixed before any run; synthetic; not selected after seeing results). One linked chain per family F1–F4,
# one decoy (ID→place) per chain, one orphan object per family (its container has NO location cartridge).
CHAINS=[dict(fam="F1",obj="amber compass",T="VX-731",P="cobalt observatory",D="KM-208",Q="willow granary"),
        dict(fam="F2",obj="linen atlas",T="RQ-415",P="saffron lighthouse",D="BT-962",Q="slate pavilion"),
        dict(fam="F3",obj="copper astrolabe",T="HN-357",P="juniper boathouse",D="WF-684",Q="ember arcade"),
        dict(fam="F4",obj="ivory sundial",T="GZ-829",P="teal citadel",D="PL-143",Q="olive foundry")]
ORPHANS=[dict(fam="F1",obj="garnet lantern",T="DS-572"),dict(fam="F2",obj="velvet hourglass",T="MC-916"),
         dict(fam="F3",obj="brass metronome",T="TK-238"),dict(fam="F4",obj="pearl telescope",T="NJ-605")]
FAKE_IDS=["QX-404","LW-319","FJ-850","YR-162"]
def build_bank_spec():
    cards=[]
    for i,c in enumerate(CHAINS):
        A,B=FAM[c["fam"]]
        cards.append(dict(role="OBJECT→ID",fam=c["fam"],text=A.format(O=c["obj"],T=c["T"]),src=c["obj"],dst=c["T"],chain=i))
        cards.append(dict(role="ID→PLACE",fam=c["fam"],text=B.format(T=c["T"],P=c["P"]),src=c["T"],dst=c["P"],chain=i))
        cards.append(dict(role="DECOY ID→PLACE",fam=c["fam"],text=B.format(T=c["D"],P=c["Q"]),src=c["D"],dst=c["Q"],chain=i))
    for j,o in enumerate(ORPHANS):
        A,_=FAM[o["fam"]];cards.append(dict(role="ORPHAN OBJECT→ID",fam=o["fam"],text=A.format(O=o["obj"],T=o["T"]),src=o["obj"],dst=o["T"],chain=None))
    order=[0,3,6,9,1,12,4,13,7,14,10,15,2,5,8,11]   # fixed interleaving so chains are not adjacent in the bank
    out=[dict(cards[k],slot=n+1,cid=f"C{n+1:02d}") for n,k in enumerate(order)]
    return out
BANK_SPEC=build_bank_spec()
KNOWN_PLACES=[c["P"] for c in CHAINS]+[c["Q"] for c in CHAINS]
def build_battery():
    Q=[]
    for i,c in enumerate(CHAINS):Q.append(dict(qid=f"L{i+1}",kind="LINKED",fam=c["fam"],label=f"Where is the {c['obj']}?",obj=c["obj"],cid=None,exp_id=c["T"],exp_place=c["P"]))
    for i,o in enumerate(ORPHANS):Q.append(dict(qid=f"U{i+1}",kind="MISSING LINK",fam=o["fam"],label=f"Where is the {o['obj']}?",obj=o["obj"],cid=None,exp_id=o["T"],exp_place=None))
    for i,c in enumerate(CHAINS):Q.append(dict(qid=f"D{i+1}",kind="DIRECT",fam=c["fam"],label=f"Where is container {c['T']}?",obj=None,cid=c["T"],exp_id=c["T"],exp_place=c["P"]))
    for i,f in enumerate(FAKE_IDS):Q.append(dict(qid=f"R{i+1}",kind="ABSENT ID",fam="-",label=f"Where is container {f}?",obj=None,cid=f,exp_id=f,exp_place=None))
    return Q
BATTERY=build_battery()
KIND_COLOR_KEY={"LINKED":"ok","MISSING LINK":"unk","DIRECT":"mod","ABSENT ID":"unk"}
# SEALED EXPERIMENTAL RECORD — historical constants. TEST461 verified line-by-line from 463.log; others transcribed from the run logs/summaries.
HIST={"TEST461":dict(label="TEST461 · final held-out (463.log)",entities=16,families=["F1","F2","F3","F4"],scales=[2,8,16],
        L=(43,48),U=(48,48),false_link=(0,48),L_lo=0.778,U_lo=0.926,scale={2:dict(L=(15,16),U=(16,16)),8:dict(L=(14,16),U=(16,16)),16:dict(L=(14,16),U=(16,16))},
        fam={"F1":(12,12),"F2":(7,12),"F3":(12,12),"F4":(12,12)},nomem=(0,16),frozen=True,errors=0,
        gates={"overall linked ≥ .80":True,"overall UNKNOWN ≥ .85":True,"false-link ≤ .10":True,"each family linked ≥ .70":False,
               "16-module linked drop ≤ .10":True,"16-module UNKNOWN drop ≤ .10":True,"NOMEM strict = 0":True,"frozen weights":True,"no errors":True},
        verdict="FAIL_FINAL_HELDOUT"),
      "TEST460":dict(label="TEST460 · development panel",arms={"JOINT_HOP\n(concatenated)":((17,32),(4,32)),"COFORGE\n(diagnostic)":((29,32),(23,32)),
        "SEQ_MW\n(isolated, sequential)":((32,32),(32,32)),"BMW_K0\n(IBR batched)":((32,32),(32,32)),"BMW_K6\n(IBR + 6 distractors)":((32,32),(32,32))},
        mechanism="INDEPENDENT_ENCODING_INDIVIDUATION_FAILURE"),
      "TEST462":dict(label="TEST462 · batch equivalence",b1={"token-1":(32,32),"first line":(32,32),"semantic":(32,32),"full transcript":(32,32)},
        multi={"token-1":(32,32),"first line":(32,32),"semantic":(32,32),"full transcript":(20,32)},verdict="BATCH_TAIL_ONLY_DIVERGENCE"),
      "TEST464":dict(label="TEST464 · F2 failure X-ray",entities=["e01","e05","e09","e13"],cols=["B solo","B + 1","B + 7","B + 15"],
        grid=[[0,0,0,0],[1,1,1,1],[1,1,1,1],[1,1,0,1]],b_solo=(3,4),larger=(8,12),collisions=0,changed=[(13,8)],
        e01_raw="elm lodge To determine the location...",verdict="F2_SINGLE_MODULE_READOUT_FAILURE",frozen=True,errors=0)}
ID_RE=re.compile(r"\b([A-Z]{2}-\d{3})\b",re.I)
AMBIG={"not","no","never","nor","or","either","maybe","perhaps","possibly","probably","might","unclear","but","unsure"}
def norm(x):return re.sub(r"[^\w]+"," ",str(x).casefold()).strip()
def first(text):
    line=next((s.strip() for s in str(text).splitlines() if s.strip()),"")
    line=re.sub(r"^\W*(final answer|answer)\s*[:\-]\s*","",line,flags=re.I).strip(" *`\"'")
    return re.split(r"(?<=[.!?])\s+",line)[0] if line else ""
def classify(text):
    fl=first(text);n=norm(fl);w=n.split();ids=sorted({m.upper() for m in ID_RE.findall(fl)})
    if not w:return "empty",ids,fl
    if set(w)<={"none","unknown"}:return "none",ids,fl
    if (set(w)&AMBIG) or len(ids)>1 or (set(w)&{"none","unknown"}):return "ambiguous",ids,fl
    return "answer",ids,fl
def place_key(a):return re.sub(r"^(?:the|in the|at the|in|at)\s+","",norm(a)).strip()
def parse_id(text):
    c,ids,_=classify(text)
    return ids[0] if c=="answer" and len(ids)==1 else None
def parse_place(text,known):
    c,ids,fl=classify(text)
    if c!="answer" or ids:return None
    k=place_key(fl);hits=[p for p in known if place_key(p)==k or place_key(p) in k]
    return hits[0] if len(hits)==1 else None
def decide_id(texts):
    u=sorted({x for x in (parse_id(t) for t in texts) if x});return u[0] if len(u)==1 else None
def decide_place(texts):
    u=sorted({x for x in (parse_place(t,KNOWN_PLACES) for t in texts) if x});return u[0] if len(u)==1 else None
def wilson(k,n,z=1.96):
    if not n:return 0.0
    p=k/n;d=1+z*z/n;return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/d
def human(n):
    n=float(n)
    for v,w in((1e18,"quintillion"),(1e15,"quadrillion"),(1e12,"trillion"),(1e9,"billion"),(1e6,"million"),(1e3,"thousand")):
        if abs(n)>=v:return f"{n/v:.2f} {w}"
    return f"{int(round(n)):,}"
def jsafe(o):
    try:import numpy as _np
    except Exception:_np=None
    if isinstance(o,dict):return{str(k):jsafe(v)for k,v in o.items()}
    if isinstance(o,(list,tuple)):return[jsafe(v)for v in o]
    if _np is not None and isinstance(o,_np.integer):return int(o)
    if _np is not None and isinstance(o,_np.floating):o=float(o)
    if isinstance(o,float):return o if math.isfinite(o)else f"non-finite:{o}"
    if isinstance(o,(str,int,bool))or o is None:return o
    return str(o)
def canon(o):return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")
def cpu_selftest():
    fx=[("NONE","none"),("Unknown.","none"),("","empty"),("NONE, but KR-417","ambiguous"),("KR-417 or MT-238","ambiguous"),("cobalt observatory","answer")]
    for t,w in fx:assert classify(t)[0]==w,(t,classify(t))
    assert parse_id("VX-731")=="VX-731" and parse_id("VX-731 or KM-208") is None and parse_id("NONE") is None
    assert parse_place("The cobalt observatory.",KNOWN_PLACES)=="cobalt observatory" and parse_place("NONE",KNOWN_PLACES) is None
    assert decide_id(["NONE","VX-731","NONE"])=="VX-731" and decide_id(["VX-731","KM-208"]) is None
    assert decide_place(["NONE","cobalt observatory"])=="cobalt observatory" and decide_place(["cobalt observatory","willow granary"]) is None
    assert len(BANK_SPEC)==16 and len({b["text"] for b in BANK_SPEC})==16 and [b["slot"] for b in BANK_SPEC]==list(range(1,17))
    ids=[c["T"] for c in CHAINS]+[c["D"] for c in CHAINS]+[o["T"] for o in ORPHANS]+FAKE_IDS;assert len(set(ids))==len(ids)
    assert len(set(KNOWN_PLACES))==8 and len(BATTERY)==16
    for q in BATTERY:
        prompts=[MW1.format(obj=q["obj"])] if q["obj"] else [];prompts+=[MW2.format(cid=x) for x in {q["exp_id"]}]
        for p in prompts:assert not any(b["text"] in p for b in BANK_SPEC),"source sentence inside a question prompt"
    assert all(not any(s in b["text"] for s in CORPUS) for b in BANK_SPEC),"codebook corpus overlaps a cartridge"
    for o in ORPHANS:assert not any(b["role"] in("ID→PLACE","DECOY ID→PLACE") and b["src"]==o["T"] for b in BANK_SPEC)
    return len(fx)+11
LOCK_MATERIAL={"engine":"TEST461 IBR · K120/V128/OWN","codebook_corpus":CORPUS,"families":FAM,"MW1":MW1,"MW2":MW2,"FMT":FMT,"SEP":SEP,
               "bank":[{k:b[k] for k in("cid","role","fam","text")} for b in BANK_SPEC],"battery":BATTERY,"known_places":KNOWN_PLACES,"K":K_DIM,"V":V_DIM}
LOCK_SHA=hashlib.sha256(canon(jsafe(LOCK_MATERIAL))).hexdigest()
#<<CONST_END>>
N_CPU_TESTS=cpu_selftest()
print("="*110);print("AKBASCORE NIRVANA · COGNITIVE CARTRIDGE — public demonstration");print("="*110)
print(f"[0/7] CPU self-test PASS ({N_CPU_TESTS} checks) · demo panel lock SHA-256 {LOCK_SHA}")
for mod,pkg in[("torch","torch"),("transformers","transformers"),("accelerate","accelerate"),("gradio","gradio"),("matplotlib","matplotlib"),("PIL","pillow")]:
    if importlib.util.find_spec(mod)is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers,gradio as gr,matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch,FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from PIL import Image
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required (Runtime → Change runtime type → GPU, A100 recommended).")
torch.set_grad_enabled(False)
STARTUP_UTC=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda")
ROOT=Path("/content/AKBASCORE_CARTRIDGE")if os.path.isdir("/content")else Path("/tmp/AKBASCORE_CARTRIDGE")
ROOT.mkdir(parents=True,exist_ok=True);ALLOWED_PATHS=[str(ROOT)]
def _cell_source():
    try:
        src=get_ipython().user_ns.get("_ih",[""])[-1]
        return src if isinstance(src,str)and "COGNITIVE CARTRIDGE" in src else None
    except Exception:return None
CELL_SOURCE=_cell_source();CELL_SOURCE_SHA=hashlib.sha256(CELL_SOURCE.encode("utf-8")).hexdigest()if CELL_SOURCE else None
def utc_now():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def local_now():return datetime.now().astimezone().isoformat(timespec="milliseconds")
print("START UTC :",STARTUP_UTC);print("Model :",MODEL_ID)
print("Gradio :",gr.__version__,"| Transformers:",transformers.__version__,"| Torch:",torch.__version__)
_tv=tuple(int("".join(c for c in x if c.isdigit())or 0)for x in transformers.__version__.split(".")[:2]);DT="dtype" if _tv>=(4,56)else "torch_dtype"
print("[1/7] MODEL LOAD")
torch.cuda.synchronize();_t=time.perf_counter()
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}).eval()
for p in model.parameters():p.requires_grad_(False)
torch.cuda.synchronize();MODEL_LOAD_S=time.perf_counter()-_t
cfg=model.config;layers=model.model.layers
H=cfg.hidden_size;NH=cfg.num_attention_heads;NKV=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None)or H//NH;KVD=NKV*HD;NL=len(layers)
PDT=next(model.parameters()).dtype
if(NL,H,NH,NKV,HD)!=(TOTAL_LAYERS,H_EXPECT,NH_EXPECT,NKV_EXPECT,HD_EXPECT):raise RuntimeError(f"Architecture mismatch: {(NL,H,NH,NKV,HD)}")
if PDT!=torch.bfloat16:raise RuntimeError(f"Expected bfloat16 weights, got {PDT}")
if getattr(cfg,"use_sliding_window",False):raise RuntimeError("Sliding-window attention is not supported by this engine.")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
_ge=model.generation_config.eos_token_id;EOS=sorted({tok.eos_token_id,*(_ge if isinstance(_ge,(list,tuple))else[_ge])}-{None})
N_PARAMS=sum(p.numel() for p in model.parameters());N_TRAINABLE=sum(p.numel() for p in model.parameters() if p.requires_grad)
LIN_PARAMS=sum(m.weight.numel() for m in model.modules() if isinstance(m,torch.nn.Linear))
GPU_NAME=torch.cuda.get_device_name(0);GPU_TOTAL=torch.cuda.get_device_properties(0).total_memory
print(f"Model loaded in {MODEL_LOAD_S:.2f}s | {GPU_NAME} | layers={NL} hidden={H} Q={NH} KV={NKV} head={HD} | params={N_PARAMS:,} trainable={N_TRAINABLE}")
def our_hooks_total():return sum(1 for l in layers for f in l._forward_hooks.values()if getattr(f,"_akbascore",None)is not None)
def framework_hooks():
    out={}
    for L,l in enumerate(layers):
        names=[getattr(f,"__qualname__",None)or type(f).__name__ for f in l._forward_hooks.values()if getattr(f,"_akbascore",None)is None]
        if names:out[f"L{L:02d}"]=names
    return out
def lora_present():return hasattr(model,"peft_config") or any("lora" in n.lower()for n,_ in model.named_modules())
def trainable_tensors():return sum(int(p.requires_grad)for p in model.parameters())
def optimizer_present():return any(isinstance(v,torch.optim.Optimizer)for v in list(globals().values()))
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
FP_NAMES=["layers.0.self_attn.q_proj","layers.8.self_attn.o_proj","layers.19.mlp.down_proj","layers.27.mlp.down_proj","model.norm","lm_head"]
SENT_CHUNKS,SENT_CHUNK=16,256
@torch.inference_mode()
def fingerprint():return tuple(float(t.sum(dtype=torch.float32))for t in FP_T)
@torch.inference_mode()
def strong_sentinel():
    h=hashlib.sha256()
    for i,t in enumerate(FP_T):
        flat=t.detach().reshape(-1);n=int(flat.numel());c=min(SENT_CHUNK,n);offs=sorted({(k*(n-c))//(SENT_CHUNKS-1)for k in range(SENT_CHUNKS)})
        sample=torch.cat([flat[o:o+c]for o in offs]).float().cpu().numpy()
        h.update(f"{i}|{FP_NAMES[i]}|{tuple(t.shape)}|{t.dtype}|{n}|{offs}|".encode());h.update(np.ascontiguousarray(sample).tobytes())
    return h.hexdigest()
print("[2/7] WEIGHT SENTINELS")
FP0=fingerprint();SENTINEL0=strong_sentinel()
if fingerprint()!=FP0 or strong_sentinel()!=SENTINEL0:raise RuntimeError("Sentinel is not repeatable.")
print("Fingerprint:",[f"{x:.4f}" for x in FP0]);print("SHA-256 sampled sentinel:",SENTINEL0)
enc=lambda s:tok(s,add_special_tokens=False).input_ids
# ---------------------------------------------------------------- ENGINE (TEST461) + counters ----------------------------------------------------------------
def new_counts():return dict(forward_passes=0,forge_passes=0,ibr_calls=0,ibr_decode_steps=0,ibr_rows=0,forward_tokens=0,generated_tokens=0,
                             memory_slot_reads=0,attn_mac=0,linear_mac=0,reproj_mac=0,cart_encdec_mac=0,svd_count=0,numbers_installed=0)
CNT=new_counts()
def _acc_forward(n_tokens,ctx_sum):
    CNT["forward_passes"]+=1;CNT["forward_tokens"]+=n_tokens;CNT["linear_mac"]+=LIN_PARAMS*n_tokens;CNT["attn_mac"]+=2*NH*HD*NL*ctx_sum
@torch.inference_mode()
def kv_from_ids(ids):
    o=model(input_ids=torch.tensor([ids],device=DEVICE),output_hidden_states=True,use_cache=False,return_dict=True);K=[];V=[]
    for L in range(NL):
        z=layers[L].input_layernorm(o.hidden_states[L][0]);a=layers[L].self_attn
        K.append(a.k_proj(z).contiguous());V.append(a.v_proj(z).contiguous())
    T=len(ids);_acc_forward(T,T*(T+1)//2);CNT["forge_passes"]+=1;CNT["reproj_mac"]+=NL*T*2*H*KVD
    return K,V
def forge(s):return kv_from_ids([PAD]+enc(s+SEP))
@torch.inference_mode()
def install(K,V):
    T=K[0].shape[0];cos,sin=model.model.rotary_emb(K[0][None],torch.arange(T,device=DEVICE)[None]);KK=[];VV=[]
    for L in range(NL):
        k=K[L].reshape(1,T,NKV,HD).transpose(1,2);v=V[L].reshape(1,T,NKV,HD).transpose(1,2)
        KK.append(apply_rotary_pos_emb(k,k,cos,sin,unsqueeze_dim=1)[1].contiguous());VV.append(v.contiguous())
    CNT["numbers_installed"]+=NL*2*KVD*T
    return tuple(KK),tuple(VV),T
CB=None;CB_INFO={}
@torch.inference_mode()
def build_codebook():
    global CB,CB_INFO
    t0=time.perf_counter();CO=[forge(s) for s in CORPUS];cb=[];evK=[];evV=[]
    for L in range(NL):
        e={}
        for j,n in enumerate(("K","V")):
            R=torch.cat([c[j][L][1:] for c in CO]).float().view(-1,NKV,HD);MU=[];B=[];ev=[]
            for h in range(NKV):
                X=R[:,h];mu=X.mean(0);_,S,Vh=torch.linalg.svd(X-mu,full_matrices=False);m=min(128,Vh.shape[0]);bb=Vh[:m].T.contiguous()
                if m<128:bb=torch.nn.functional.pad(bb,(0,128-m))
                MU.append(mu);B.append(bb);CNT["svd_count"]+=1
                s2=S.square();ev.append(float(s2[:(K_DIM if n=="K" else V_DIM)].sum()/s2.sum()))
            e[n]=(torch.stack(MU),torch.stack(B));(evK if n=="K" else evV).append(float(np.mean(ev)))
        cb.append(e)
    rows=sum(c[0][0].shape[0]-1 for c in CO);del CO;torch.cuda.synchronize()
    CB=cb;CB_INFO=dict(sentences=len(CORPUS),rows=rows,svds=NL*2*NKV,seconds=time.perf_counter()-t0,ev_K=evK,ev_V=evV,
                       bytes=sum(t.numel()*t.element_size() for e in cb for n in("K","V") for t in e[n]))
    return CB_INFO
@torch.inference_mode()
def packet(source):
    # TEST461 packet math, unchanged. Telemetry is computed AFTER the rows exist and never alters them.
    K,V=forge(source);out={};tel={"cosK":[],"cosV":[],"normK":[],"normV":[]}
    for n,X,d in (("K",K,K_DIM),("V",V,V_DIM)):
        rows=[]
        for L in range(NL):
            mu,B=CB[L][n];coeff=torch.einsum("thi,hid->thd",X[L][1:].float().view(-1,NKV,HD)-mu,B[:,:,:d]).to(torch.bfloat16)
            content=(mu+torch.einsum("thd,hid->thi",coeff.float(),B[:,:,:d])).reshape(-1,KVD).to(torch.bfloat16)
            rows.append(torch.cat([X[L][:1],content]))
            tel["cos"+n].append(float(torch.nn.functional.cosine_similarity(content.float().reshape(-1),X[L][1:].float().reshape(-1),0)))
            tel["norm"+n].append(float(coeff.float().reshape(coeff.shape[0],-1).norm(dim=1).mean()))
        out[n]=rows
    T=K[0].shape[0];CNT["cart_encdec_mac"]+=2*NL*(T-1)*NKV*HD*(K_DIM+V_DIM)
    tel.update(T=T,code_numbers=NL*(T-1)*NKV*(K_DIM+V_DIM),own_numbers=NL*2*KVD,native_numbers=NL*2*KVD*T,
               source_tokens=T-1,hidden_copy_numbers=NL*T*H)
    return out["K"],out["V"],tel
STAT=Counter()
@torch.inference_mode()
def batch(q,kvs):
    # TEST461 Isolated Batched Read, unchanged math: one forward per decode step for all cartridge rows; each row sees only its own cartridge.
    qids=enc(FMT.format(q=q));BN=len(kvs);Tm=max(kv[2] for kv in kvs);nq=len(qids)
    try:cache=DynamicCache(config=cfg)
    except TypeError:cache=DynamicCache()
    for L in range(NL):
        Ks=[];Vs=[]
        for kv in kvs:
            k,v,p=kv[0][L],kv[1][L],Tm-kv[2]
            if p:k=torch.cat([k.new_zeros(1,NKV,p,HD),k],2);v=torch.cat([v.new_zeros(1,NKV,p,HD),v],2)
            Ks.append(k);Vs.append(v)
        cache.update(torch.cat(Ks).clone(),torch.cat(Vs).clone(),L)
    mask=torch.zeros(BN,Tm+nq,dtype=torch.long,device=DEVICE)
    for b,kv in enumerate(kvs):mask[b,Tm-kv[2]:]=1
    pos=torch.tensor([kv[2] for kv in kvs],device=DEVICE)[:,None]+torch.arange(nq,device=DEVICE)[None]
    ids=torch.tensor([qids]*BN,device=DEVICE);outs=[[] for _ in range(BN)];done=[False]*BN;steps=0
    torch.cuda.synchronize();t0=time.perf_counter()
    for _ in range(MAX_NEW):
        o=model(input_ids=ids,attention_mask=mask,position_ids=pos,past_key_values=cache,use_cache=True,return_dict=True);steps+=1
        ntok=ids.shape[1];ctx0=mask.shape[1]-ntok;_acc_forward(BN*ntok,BN*sum(ctx0+j+1 for j in range(ntok)))
        CNT["memory_slot_reads"]+=BN*ntok*NH*NL*Tm
        nxt=o.logits[:,-1].float().argmax(-1).tolist()
        for b,t in enumerate(nxt):
            if not done[b]:
                if t in EOS:done[b]=True
                else:outs[b].append(t)
        if all(done):break
        feed=[EOS[0] if done[b] and EOS else nxt[b] for b in range(BN)]
        ids=torch.tensor([[t] for t in feed],device=DEVICE);pos=pos[:,-1:]+1;mask=torch.cat([mask,torch.ones(BN,1,dtype=torch.long,device=DEVICE)],1)
    torch.cuda.synchronize();dt=time.perf_counter()-t0
    CNT["ibr_calls"]+=1;CNT["ibr_decode_steps"]+=steps;CNT["ibr_rows"]+=BN;CNT["generated_tokens"]+=sum(len(x) for x in outs)
    STAT["last_seconds"]=dt;STAT["last_steps"]=steps;STAT["last_prompt_tokens"]=nq;STAT["last_prompt"]=tok.decode(qids)
    return [tok.decode(x,skip_special_tokens=True).strip() for x in outs]
@torch.inference_mode()
def nomem_kv():
    K,V=kv_from_ids([PAD]);return install(K,V)
print("[3/7] ENGINE LOCK")
ENGINE_LOCK_CHECKS=[("MODEL_ID = Qwen/Qwen2.5-7B-Instruct",MODEL_ID=="Qwen/Qwen2.5-7B-Instruct"),
 ("architecture 28 layers / hidden 3584 / 28 Q / 4 KV / head 128",(NL,H,NH,NKV,HD)==(28,3584,28,4,128)),
 ("cartridge code K120 / V128 / OWN slot 0",(K_DIM,V_DIM)==(120,128)),("codebook = 32 neutral sentences, disjoint from cartridges",len(CORPUS)==32),
 ("16 cartridges / 16 automatic questions locked",len(BANK_SPEC)==16 and len(BATTERY)==16),("readout frame exact",FMT=="QUESTION:\n{q}\n\nANSWER:"),
 ("greedy decoding (argmax), MAX_NEW=32",MAX_NEW==32),("BF16 weights",PDT==torch.bfloat16),("SDPA attention",getattr(cfg,"_attn_implementation",None)=="sdpa"),
 ("no AkbasCore forward hooks",our_hooks_total()==0),("frozen eval model",(not model.training)and trainable_tensors()==0 and not lora_present()),
 ("no optimizer",not optimizer_present()),("sampled weight sentinel repeatable",fingerprint()==FP0 and strong_sentinel()==SENTINEL0)]
_f=[n for n,ok in ENGINE_LOCK_CHECKS if not ok]
if _f:raise RuntimeError(f"ENGINE LOCK FAILED: {_f}")
print("Engine lock: PASS ·",len(ENGINE_LOCK_CHECKS),"checks")
#<<POSTERS_BEGIN>>
plt.rcParams["font.family"]="DejaVu Sans"
DPI=120
C_OK,C_UNK,C_ERR,C_CTL,C_CART,C_MOD,C_FG,C_NEU,C_BG="#047857","#6D28D9","#B91C1C","#475569","#B45309","#0369A1","#0F172A","#334155","#F8FAFC"
C_OKL,C_UNKL,C_ERRL,C_CARTL,C_MODL="#D1FAE5","#EDE9FE","#FEE2E2","#FEF3C7","#E0F2FE"
MONO="DejaVu Sans Mono";BRAND="AKBASCORE NIRVANA · COGNITIVE CARTRIDGE"
ROLE_COLOR={"OBJECT→ID":C_CART,"ID→PLACE":C_OK,"DECOY ID→PLACE":C_CTL,"ORPHAN OBJECT→ID":C_UNK}
def mt(s):return str(s).replace("$",r"\$")
def save_jpg(fig,path):
    buf=io.BytesIO();fig.savefig(buf,format="png",dpi=fig.dpi,facecolor="white");plt.close(fig);buf.seek(0)
    with Image.open(buf)as im:
        im=im.convert("RGBA");bg=Image.new("RGB",im.size,(255,255,255));bg.paste(im,mask=im.getchannel("A"))
    bg.save(path,"JPEG",quality=92,optimize=True,progressive=False,subsampling=0)
    with Image.open(path)as chk:
        if chk.format!="JPEG" or chk.mode!="RGB":raise RuntimeError("JPEG validation failed")
    return str(path)
def fit_text(fig,x,y,w,h,text,fs_max=13,fs_min=7,color=C_FG,family=None,ls=1.32,weight="normal"):
    fig.canvas.draw();r=fig.canvas.get_renderer()
    Wp=w*fig.get_figwidth()*fig.dpi;Hp=h*fig.get_figheight()*fig.dpi
    if Wp<=4 or Hp<=4:return 0
    paras=str(text if text else "(empty output)").replace("\r","").split("\n")
    kw={"va":"top","ha":"left","color":color,"linespacing":ls,"weight":weight}
    if family:kw["family"]=family
    def wrap(c):
        out=[]
        for p in paras:out.extend(textwrap.wrap(p,c,break_long_words=True)or[""])
        return out
    fs=float(fs_max);k=0.52
    for _ in range(150):
        cpl=max(6,int(Wp/(k*fs*fig.dpi/72.0)));ln=wrap(cpl)
        t=fig.text(x,y+h,mt("\n".join(ln)),fontsize=fs,**kw);bb=t.get_window_extent(renderer=r)
        if bb.width>Wp*1.002 and cpl>6:t.remove();k*=max(1.01,bb.width/Wp*1.01);continue
        if bb.height<=Hp:return fs
        t.remove()
        if fs>fs_min:fs=max(float(fs_min),fs-0.5);continue
        per=bb.height/max(1,len(ln));keep=max(1,int(Hp/per)-1)
        fig.text(x,y+h,mt("\n".join(ln[:keep]+["[… poster space exhausted — complete raw text is in the run log]"])),fontsize=fs,**kw);return fs
    fig.text(x,y+h,"[text omitted on poster — complete raw text is in the run log]",fontsize=fs_min,**kw);return fs_min
def fx(fig,px):return px/(fig.get_figwidth()*fig.dpi)
def fy(fig,px):return px/(fig.get_figheight()*fig.dpi)
def card(fig,x,y,w,h,title,body,edge,fs_max=13,fs_min=7,sub=None,title_fs=13,family=None,sub_fs=10,face=C_BG):
    fig.add_artist(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=0.006",transform=fig.transFigure,facecolor=face,edgecolor=edge,lw=2.4,zorder=0))
    px,py=fx(fig,16),fy(fig,12);th=fy(fig,title_fs*fig.dpi/72*1.8)if title else 0
    if title:fig.text(x+px,y+h-py,mt(title),fontsize=title_fs,weight="bold",color=edge,va="top",ha="left")
    ns=(sub.count("\n")+1)if sub else 0;sh=fy(fig,sub_fs*fig.dpi/72*1.45*ns+8)if sub else 0
    if sub:fig.text(x+px,y+py,mt(sub),fontsize=sub_fs,color=C_NEU,va="bottom",ha="left",linespacing=1.3,family=MONO)
    return fit_text(fig,x+px,y+py+sh,w-2*px,h-2*py-th-sh,body,fs_max,fs_min,family=family)
def head(fig,title,sub=None,tag=None):
    Hh=fig.get_figheight();f=lambda inch:1-inch/Hh
    fig.text(.05,f(.42),BRAND,fontsize=12,weight="bold",color=C_CART,va="center")
    if tag:fig.text(.95,f(.42),tag,fontsize=12,weight="bold",color=C_MOD if tag.startswith("THIS") else C_UNK,va="center",ha="right",
                    bbox=dict(boxstyle="round,pad=0.35",fc="white",ec=C_MOD if tag.startswith("THIS") else C_UNK,lw=1.6))
    fig.text(.05,f(.95),mt(title),fontsize=30,weight="bold",color=C_FG,va="center")
    if sub:fig.text(.05,f(1.42),mt(sub),fontsize=14,color=C_NEU,va="center")
    fig.add_artist(Line2D([.05,.95],[f(1.70),f(1.70)],transform=fig.transFigure,color=C_FG,lw=1.2))
    return f(1.85)
def foot(fig,ctx,k):
    fig.text(.5,.22/fig.get_figheight(),mt(f"AKBASCORE NIRVANA · COGNITIVE CARTRIDGE | RUN {ctx['run_id']} | PAYLOAD SHA-256 {ctx['sha'][:16]}… | {k:02d}/{ctx['N']:02d}"),
             ha="center",va="center",fontsize=10,color=C_NEU,family=MONO)
def note(fig,x,y,w,h,plain,sci=None):
    fit_text(fig,x,y+(h*.42 if sci else 0),w,h*(.58 if sci else 1),plain,fs_max=14.5,fs_min=9)
    if sci:fit_text(fig,x,y,w,h*.38,sci,fs_max=11,fs_min=7.5,family=MONO,color=C_NEU)
def caption(fig,x,y,w,h,t):return fit_text(fig,x,y,w,h,"▲ "+t,fs_max=13,fs_min=9,weight="bold")
def clean(ax):
    for s in("top","right"):ax.spines[s].set_visible(False)
def box(fig,x,y,w,h,title,lines,color,face,tfs=15,bfs=12):
    fig.add_artist(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=0.012",transform=fig.transFigure,facecolor=face,edgecolor=color,lw=2.6))
    fig.text(x+w/2,y+h*.70,mt(title),ha="center",va="center",fontsize=tfs,weight="bold",color=color)
    fig.text(x+w/2,y+h*.32,mt(lines),ha="center",va="center",fontsize=bfs,color=C_FG,linespacing=1.35)
def arrow(fig,x0,y0,x1,y1,color=C_FG,ls="-"):
    fig.add_artist(FancyArrowPatch((x0,y0),(x1,y1),transform=fig.transFigure,arrowstyle="-|>",mutation_scale=26,lw=2.4,color=color,linestyle=ls))
def frac(t):return f"{t[0]}/{t[1]}"
def pct(t):return 100.0*t[0]/max(1,t[1])
def status_color(s):return {"CORRECT":C_OK,"UNKNOWN ✓":C_UNK,"WRONG":C_ERR,"FALSE LINK":C_ERR,"MISSED":C_ERR}.get(s,C_CTL)
# ---------------------------------------------------------------- posters ----------------------------------------------------------------
def p01_what(ctx,k,path):
    P=ctx["P"];S=P["live_summary"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    fig.text(.05,.95,"AKBASCORE NIRVANA",fontsize=44,weight="bold",color=C_FG,va="center")
    fig.text(.05,.885,"COGNITIVE CARTRIDGE · WHAT JUST HAPPENED?",fontsize=24,weight="bold",color=C_CART,va="center")
    fig.text(.05,.835,"Knowledge goes in. The source goes away. The model answers from the cartridges — or says UNKNOWN.",fontsize=15,color=C_NEU,va="center")
    bw,bh=.27,.17;xs=[.05,.365,.68];y1,y2=.58,.33
    B=P["bank"];src_tok=sum(c["T"]-1 for c in P["cartridges"])
    box(fig,xs[0],y1,bw,bh,"① SOURCE KNOWLEDGE",f"{len(P['cartridges'])} synthetic sentences\n{src_tok} source tokens · invented facts",C_FG,C_BG)
    box(fig,xs[1],y1,bw,bh,"② COGNITIVE CARTRIDGES",f"{len(P['cartridges'])} independent cartridges\nK120 / V128 codes × 28 layers",C_CART,C_CARTL)
    box(fig,xs[2],y1,bw,bh,"③ SOURCE REMOVED",f"source tokens in every\nreadout prompt = {P['source_removal']['source_sentence_hits']}",C_ERR,C_ERRL)
    box(fig,xs[2],y2,bw,bh,"④ FROZEN MODEL + CARTRIDGES",f"{MODEL_SHORT} · 28 layers\nweights changed = 0",C_MOD,C_MODL)
    box(fig,xs[1],y2,bw,bh,"⑤ 16 AUTOMATIC QUESTIONS","linked · missing link\ndirect · absent ID",C_FG,C_BG)
    box(fig,xs[0],y2,bw,bh,"⑥ ANSWERS / UNKNOWN",f"{S['correct']}/{S['n']} questions correct\n{S['unknown_correct']} correct UNKNOWNs · {S['false_links']} false links",C_OK,C_OKL)
    arrow(fig,xs[0]+bw,y1+bh/2,xs[1],y1+bh/2);arrow(fig,xs[1]+bw,y1+bh/2,xs[2],y1+bh/2);arrow(fig,xs[2]+bw/2,y1,xs[2]+bw/2,y2+bh)
    arrow(fig,xs[2],y2+bh/2,xs[1]+bw,y2+bh/2);arrow(fig,xs[1],y2+bh/2,xs[0]+bw,y2+bh/2)
    for i,(t,c) in enumerate([("NO TRAINING",C_MOD),("NO WEIGHT UPDATE",C_MOD),("SOURCE ABSENT AT READOUT",C_ERR)]):
        wb=[.13,.17,.20][i];x=.05+sum([.13,.17,.20][:i])+i*.01;fig.add_artist(FancyBboxPatch((x,.205),wb,.06,boxstyle="round,pad=0,rounding_size=0.01",transform=fig.transFigure,facecolor="white",edgecolor=c,lw=2.2))
        fig.text(x+wb/2,.235,t,ha="center",va="center",fontsize=11.5,weight="bold",color=c)
    ax=fig.add_axes([.74,.08,.21,.19]);kinds=["LINKED","MISSING LINK","DIRECT","ABSENT ID"];v=[S["by_kind"][q]["ok"] for q in kinds];n=[S["by_kind"][q]["n"] for q in kinds]
    cols=[C_OK,C_UNK,C_MOD,C_UNK];ax.barh(range(4),n,color="#E2E8F0");ax.barh(range(4),v,color=cols);ax.invert_yaxis()
    ax.set_yticks(range(4));ax.set_yticklabels(["linked","missing → UNKNOWN","direct ID","absent ID → UNKNOWN"],fontsize=11)
    for i in range(4):ax.text(n[i]+.08,i,f"{v[i]}/{n[i]}",va="center",fontsize=13,weight="bold")
    ax.set_xlim(0,max(n)*1.3);ax.set_xticks([]);ax.set_title("THIS LIVE RUN · automatic questions",fontsize=12.5,weight="bold",loc="left");clean(ax)
    fit_text(fig,.05,.06,.50,.12,"Technical: compressed K/V cartridge state installed into the attention cache of a frozen model and read by Isolated Batched Read (IBR). It is not training and not a permanent change to the model.",fs_max=11.5,fs_min=8,color=C_NEU)
    foot(fig,ctx,k);return save_jpg(fig,path)
