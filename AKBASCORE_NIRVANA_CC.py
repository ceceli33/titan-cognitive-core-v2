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
def p02_engine(ctx,k,path):
    P=ctx["P"];W=P["workload"];T=P["timing"]["run"];fig=plt.figure(figsize=(16,11),dpi=DPI,facecolor="white")
    top=head(fig,"INSIDE THE ENGINE","Simple interface ≠ simple engine. Every box below exists in this code and every number is from this run.")
    stack=[("GRADIO · CONTROL","one button, live stages, downloads",C_CTL,"#F1F5F9",f"{P['ui']['stages']} live stages"),
           ("PYTHON ORCHESTRATION","locks, checks, scoring, sealing, posters",C_CTL,"#F1F5F9",f"{P['integrity']['checks_total']} integrity checks"),
           ("AKBASCORE CARTRIDGE ENGINE","PCA encode/decode · RoPE install · IBR",C_CART,C_CARTL,f"{W['derived']['cart_encdec_mac']['human']} cartridge MACs"),
           ("PYTORCH CUDA GPU COMPUTE","cuBLAS GEMM · SDPA attention (no custom kernel)",C_MOD,C_MODL,f"{W['measured']['forward_passes']['value']:,} forward passes"),
           ("A100 GPU MEMORY · BF16 TENSORS",f"peak {P['gpu']['peak_allocated_gib']:.2f} GiB allocated",C_MOD,C_MODL,f"{P['gpu']['name']}"),
           (f"{MODEL_SHORT.upper()} · 28 LAYERS",f"{P['model']['params']/1e9:.2f} B parameters · frozen",C_MOD,C_MODL,"trainable = 0"),
           ("COMPRESSED K/V CARTRIDGE BANK",f"16 cartridges · {P['bank']['slots']} memory slots",C_CART,C_CARTL,f"{P['bank']['code_numbers']:,} code numbers"),
           ("IBR READOUT",f"{W['measured']['ibr_calls']['value']} batched calls × 16 rows",C_OK,C_OKL,f"{W['measured']['ibr_decode_steps']['value']} decode steps")]
    n=len(stack);hh=(top-.08)/n
    for i,(t,s,c,f_,m) in enumerate(stack):
        y=top-(i+1)*hh+.006
        fig.add_artist(FancyBboxPatch((.05,y),.44,hh-.012,boxstyle="round,pad=0,rounding_size=0.008",transform=fig.transFigure,facecolor=f_,edgecolor=c,lw=2.2))
        fig.text(.065,y+(hh-.012)*.66,t,fontsize=13,weight="bold",color=c,va="center");fig.text(.065,y+(hh-.012)*.28,s,fontsize=10.5,color=C_FG,va="center")
        fig.text(.48,y+(hh-.012)/2,m,fontsize=10.5,weight="bold",color=C_FG,va="center",ha="right",family=MONO)
        if i<n-1:fig.text(.27,y-.004,"▼",fontsize=9,color=C_NEU,ha="center",va="center")
    E=np.array([[c["normK"][L]+c["normV"][L] for L in range(TOTAL_LAYERS)] for c in P["cartridges"]])
    ax=fig.add_axes([.56,top-.40,.39,.36]);im=ax.imshow(E,aspect="auto",cmap="magma")
    ax.set_xticks(range(0,28,3));ax.set_xticklabels([f"L{x}" for x in range(0,28,3)],fontsize=9);ax.set_yticks(range(16));ax.set_yticklabels([c["cid"] for c in P["cartridges"]],fontsize=8.5)
    ax.set_title("ENGINE TUNNEL · code energy per layer per cartridge",fontsize=12.5,weight="bold",loc="left");fig.colorbar(im,cax=fig.add_axes([.955,top-.40,.008,.36]))
    caption(fig,.56,top-.50,.39,.07,"16 cartridges (rows) flowing through 28 transformer layers (columns). Color = measured norm of the stored K+V codes.")
    ax2=fig.add_axes([.60,.10,.35,top-.66]);st=[("codebook",P["codebook"]["seconds"]),("16 cartridges",T["cartridges_seconds"]),("question battery",T["battery_seconds"]),("NOMEM control",T["control_seconds"])]
    ax2.barh(range(4),[s for _,s in st],color=[C_CART,C_CART,C_OK,C_CTL]);ax2.invert_yaxis();ax2.set_yticks(range(4));ax2.set_yticklabels([a for a,_ in st],fontsize=11)
    for i,(_,s) in enumerate(st):ax2.text(s,i,f" {s:.2f} s",va="center",fontsize=11,weight="bold")
    ax2.set_xlim(0,max(s for _,s in st)*1.35);ax2.set_xlabel("measured seconds (GPU-synchronized)",fontsize=10.5);clean(ax2)
    foot(fig,ctx,k);return save_jpg(fig,path)
def p03_bank(ctx,k,path):
    P=ctx["P"];C=P["cartridges"];fig=plt.figure(figsize=(16,11),dpi=DPI,facecolor="white")
    top=head(fig,"THE CARTRIDGE BANK · 16 LOADED","Each slot is one independently forged cartridge. IBR reads every slot in its own isolated row.")
    gw,gh=.105,(top-.14)/4
    for i,c in enumerate(C):
        r,q=divmod(i,4);x=.05+q*(gw+.008);y=top-(r+1)*gh-.005;col=ROLE_COLOR[c["role"]]
        fig.add_artist(FancyBboxPatch((x,y),gw,gh-.012,boxstyle="round,pad=0,rounding_size=0.008",transform=fig.transFigure,facecolor="white",edgecolor=col,lw=2.4))
        fig.text(x+.007,y+gh-.03,c["cid"],fontsize=13,weight="bold",color=col,va="top");fig.text(x+gw-.007,y+gh-.03,c["fam"],fontsize=10,color=C_NEU,va="top",ha="right")
        fig.text(x+.007,y+(gh-.012)*.52,c["role"].replace(" ","\n",1) if len(c["role"])>10 else c["role"],fontsize=8.5,weight="bold",color=col,va="center")
        fig.text(x+.007,y+(gh-.012)*.2,f"{c['src']} →\n{c['dst']}",fontsize=8,color=C_FG,va="center",linespacing=1.1)
    ax=fig.add_axes([.53,.12,.42,top-.16]);ax.set_xlim(0,3);ax.set_ylim(-.5,9.5);ax.axis("off")
    objs=[c["obj"] for c in CHAINS]+[o["obj"] for o in ORPHANS];ids=[c["T"] for c in CHAINS]+[o["T"] for o in ORPHANS]+[c["D"] for c in CHAINS];places=KNOWN_PLACES
    oy={o:8.6-i*1.15 for i,o in enumerate(objs)};iy={d:9.1-i*.78 for i,d in enumerate(ids)};py={p:8.6-i*1.15 for i,p in enumerate(places)}
    for c in C:
        if c["role"] in("OBJECT→ID","ORPHAN OBJECT→ID"):ax.plot([.55,1.45],[oy[c["src"]],iy[c["dst"]]],color=ROLE_COLOR[c["role"]],lw=2.2)
        else:ax.plot([1.55,2.45],[iy[c["src"]],py[c["dst"]]],color=ROLE_COLOR[c["role"]],lw=2.2,ls="--" if c["role"].startswith("DECOY") else "-")
    for o,y in oy.items():ax.text(.5,y,o,ha="right",va="center",fontsize=9.5,weight="bold",color=C_FG)
    for d,y in iy.items():
        orphan=d in [o["T"] for o in ORPHANS];ax.text(1.5,y,d,ha="center",va="center",fontsize=9.5,family=MONO,color=C_UNK if orphan else C_FG,
                    bbox=dict(boxstyle="round,pad=0.25",fc=C_UNKL if orphan else "white",ec=C_UNK if orphan else C_NEU,lw=1.2))
    for p,y in py.items():ax.text(2.5,y,p,ha="left",va="center",fontsize=9.5,weight="bold",color=C_OK if p in [c["P"] for c in CHAINS] else C_CTL)
    ax.text(.5,9.7,"OBJECTS",ha="right",fontsize=11,weight="bold",color=C_CART);ax.text(1.5,9.7,"CONTAINER IDs",ha="center",fontsize=11,weight="bold",color=C_FG);ax.text(2.5,9.7,"PLACES",ha="left",fontsize=11,weight="bold",color=C_OK)
    lg=[Line2D([0],[0],color=C_CART,lw=3,label="object→ID cartridge"),Line2D([0],[0],color=C_OK,lw=3,label="ID→place cartridge"),Line2D([0],[0],color=C_CTL,lw=3,ls="--",label="decoy ID→place"),Line2D([0],[0],color=C_UNK,lw=3,label="orphan object→ID (no place)")]
    ax.legend(handles=lg,loc="lower center",bbox_to_anchor=(.5,-.12),ncol=2,fontsize=9.5,frameon=False)
    caption(fig,.05,.02,.46,.08,"purple IDs have no location cartridge: the honest answer for their objects is UNKNOWN.")
    foot(fig,ctx,k);return save_jpg(fig,path)
def p04_work(ctx,k,path):
    P=ctx["P"];W=P["workload"];fig=plt.figure(figsize=(16,11),dpi=DPI,facecolor="white")
    top=head(fig,"HOW HARD DID THE ENGINE WORK?","Three honest categories: MEASURED counters · DERIVED exact counts from tensor shapes · ESTIMATED model FLOPs")
    groups=[("MEASURED",C_MOD,[(W["measured"][k_]["label"],W["measured"][k_]["value"]) for k_ in W["measured_order"]]),
            ("DERIVED",C_CART,[(W["derived"][k_]["label"],W["derived"][k_]["value"]) for k_ in W["derived_order"]]),
            ("ESTIMATED",C_UNK,[(W["estimated"][k_]["label"],W["estimated"][k_]["value"]) for k_ in W["estimated_order"]])]
    y0=top-.03;heights=[.27,.22,.15]
    for (g,c,items),hgt in zip(groups,heights):
        ax=fig.add_axes([.40,y0-hgt,.44,hgt-.035]);vals=[max(1,v) for _,v in items]
        ax.barh(range(len(items)),vals,color=c,alpha=.85);ax.set_xscale("log");ax.invert_yaxis();ax.set_yticks(range(len(items)));ax.set_yticklabels([l for l,_ in items],fontsize=10.5)
        for i,v in enumerate(vals):ax.text(v*1.25,i,human(v),va="center",fontsize=11,weight="bold")
        ax.set_xlim(1,max(vals)*400);ax.set_xticks([]);clean(ax)
        fig.text(.05,y0+.012,g,fontsize=17,weight="bold",color=c,va="top");ylast=y0;y0-=hgt+.012
    tot=W["estimated"]["total_flop"]["value"]
    fig.text(.05,ylast-.05,f"≈ {human(tot)}",fontsize=17,weight="bold",color=C_UNK,va="top");fig.text(.05,ylast-.095,"estimated GPU FLOPs, this run",fontsize=11,color=C_FG,va="top")
    fit_text(fig,.05,.025,.9,.095,W["formula_note"],fs_max=10.5,fs_min=7.5,family=MONO,color=C_NEU)
    foot(fig,ctx,k);return save_jpg(fig,path)
def p05_compress(ctx,k,path):
    P=ctx["P"];C=P["cartridges"];B=P["bank"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"SOURCE → CARTRIDGE ENCODING","Measured sizes for every cartridge in this run (bf16, 2 bytes per number)")
    ax=fig.add_axes([.07,.36,.55,top-.42]);x=np.arange(16);w=.4
    ax.bar(x-w/2,[c["native_numbers"]*2/1024 for c in C],w,color=C_MOD,label="native K/V cache (KiB)")
    ax.bar(x+w/2,[(c["code_numbers"]+c["own_numbers"])*2/1024 for c in C],w,color=C_CART,label="cartridge codes + own slot 0 (KiB)")
    ax.set_xticks(x);ax.set_xticklabels([c["cid"] for c in C],fontsize=9,rotation=45);ax.set_ylabel("KiB per cartridge",fontsize=11);ax.legend(frameon=False,fontsize=10.5);clean(ax)
    caption(fig,.07,.26,.55,.07,f"cartridge = {B['ratio_vs_native']*100:.1f}% of the native K/V it replaces (K keeps 120 of 128 dims per head; V keeps all 128).")
    ax2=fig.add_axes([.70,.36,.25,top-.42]);lab=["hidden-state\ncopy","native K/V","cartridge\ncodes"];val=[B["hidden_copy_numbers"]*2/2**20,B["native_numbers"]*2/2**20,(B["code_numbers"]+B["own_numbers"])*2/2**20]
    ax2.bar(range(3),val,color=[C_CTL,C_MOD,C_CART])
    for i,v in enumerate(val):ax2.text(i,v*1.02,f"{v:.2f}\nMiB",ha="center",va="bottom",fontsize=11,weight="bold")
    ax2.set_xticks(range(3));ax2.set_xticklabels(lab,fontsize=10);ax2.set_ylim(0,max(val)*1.3);ax2.set_title("whole bank",fontsize=12,weight="bold");clean(ax2)
    note(fig,.05,.04,.9,.2,f"Honest size accounting: the cartridge is not a large size reduction versus the model's own K/V cache ({B['ratio_vs_native']*100:.1f}%). It is {B['ratio_vs_hidden']*100:.1f}% of a full hidden-state copy. What changes is the form: the source text is gone and only codes against a fixed neutral codebook remain.",
         f"code numbers = Σ 28 × (T−1) × 4 heads × (120 K + 128 V); own slot 0 = 28 × 2 × 512 raw; native = 28 × 2 × 512 × T. Shared codebook (μ + basis, fp32) = {P['codebook']['bytes']/2**20:.2f} MiB, stored once for all cartridges. At readout, cartridges are decoded to full-size runtime K/V ({B['runtime_numbers']*2/2**20:.2f} MiB for the bank).")
    foot(fig,ctx,k);return save_jpg(fig,path)
def p06_telemetry(ctx,k,path):
    P=ctx["P"];C=P["cartridges"];fig=plt.figure(figsize=(16,11),dpi=DPI,facecolor="white")
    top=head(fig,"CARTRIDGE TELEMETRY · LAYER × CARTRIDGE","Real per-layer measurements for each cartridge as it was forged in this run")
    mats=[("K reconstruction cosine",np.array([c["cosK"] for c in C]),"YlGn"),("V reconstruction cosine",np.array([c["cosV"] for c in C]),"YlGn"),
          ("K code norm (mean per token)",np.array([c["normK"] for c in C]),"viridis"),("V code norm (mean per token)",np.array([c["normV"] for c in C]),"viridis")]
    for i,(t,M,cm) in enumerate(mats):
        r,q=divmod(i,2);x=.06+q*.47;y=top-(r+1)*.37+.03
        ax=fig.add_axes([x,y,.38,.28]);im=ax.imshow(M,aspect="auto",cmap=cm);ax.set_title(t,fontsize=12.5,weight="bold",loc="left")
        ax.set_xticks(range(0,28,4));ax.set_xticklabels([f"L{v}" for v in range(0,28,4)],fontsize=9);ax.set_yticks(range(0,16,3));ax.set_yticklabels([C[j]["cid"] for j in range(0,16,3)],fontsize=9)
        fig.colorbar(im,cax=fig.add_axes([x+.385,y,.007,.28]))
        ax.text(1.0,-.2,f"min {M.min():.4f} · max {M.max():.4f}",transform=ax.transAxes,ha="right",fontsize=9.5,family=MONO,color=C_NEU)
    note(fig,.05,.04,.9,.09,"Cosine 1.0 = the decoded memory equals the model's own K/V. K keeps 120 of 128 dimensions, so it is close but not perfect; V keeps the full 128-dim basis.")
    foot(fig,ctx,k);return save_jpg(fig,path)
def p07_source(ctx,k,path):
    P=ctx["P"];S=P["source_removal"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"SOURCE REMOVAL PROOF","What the model receives when it answers: the question as text, the knowledge only as installed cartridge numbers")
    box(fig,.05,top-.22,.25,.17,"SOURCE TEXT",f"{S['source_sentences']} sentences\nused ONCE to forge cartridges",C_FG,C_BG)
    box(fig,.38,top-.22,.25,.17,"CARTRIDGE BANK",f"{P['bank']['slots']} slots · numbers only\ninstalled into the cache",C_CART,C_CARTL)
    box(fig,.71,top-.22,.24,.17,"READOUT PROMPT","question text only\n'QUESTION: … ANSWER:'",C_MOD,C_MODL)
    arrow(fig,.30,top-.135,.38,top-.135);arrow(fig,.63,top-.135,.71,top-.135)
    fig.text(.50,top-.27,"✗  source text → readout prompt: blocked and asserted in every call",ha="center",fontsize=14,weight="bold",color=C_ERR)
    ax=fig.add_axes([.07,.24,.86,top-.58]);calls=S["calls"];x=np.arange(len(calls))
    ax.bar(x,[c["prompt_tokens"] for c in calls],color=C_MOD,label="question tokens in prompt")
    ax.bar(x,[c["source_hits"] for c in calls],color=C_ERR,label="source sentences in prompt (all zero)")
    ax.plot(x,[c["slots"] for c in calls],"o-",color=C_CART,lw=2,label="installed cartridge slots read")
    ax.set_xticks(x);ax.set_xticklabels([c["tag"] for c in calls],fontsize=8.5,rotation=60);ax.legend(frameon=False,fontsize=10.5,ncol=3,loc="upper left");ax.set_ylabel("count per IBR call",fontsize=11);clean(ax)
    ax.set_ylim(0,max(max(c["slots"] for c in calls),max(c["prompt_tokens"] for c in calls))*1.35)
    note(fig,.05,.04,.9,.11,f"SOURCE SENTENCES IN READOUT PROMPTS = {S['source_sentence_hits']} across {len(calls)} IBR calls. The prompt carries the question only (it names the object or container ID being asked about). The knowledge arrives as installed K/V cartridge rows.")
    foot(fig,ctx,k);return save_jpg(fig,path)
def p08_frozen(ctx,k,path):
    P=ctx["P"];I=P["integrity"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"FROZEN MODEL PROOF","The model's weights were checked before and after this run")
    pts=["startup","pre-run","post-run"];F=np.array([I["fingerprint_startup"],I["fingerprint_pre_run"],I["fingerprint_after"]])
    ax=fig.add_axes([.07,.40,.50,top-.46])
    for j in range(F.shape[1]):ax.plot(range(3),F[:,j]/F[0,j],"o-",lw=2.2,ms=9,label=FP_NAMES[j])
    ax.set_xticks(range(3));ax.set_xticklabels(pts,fontsize=12);ax.set_ylim(.999,1.001);ax.set_ylabel("tensor sum ÷ startup value",fontsize=11)
    ax.legend(frameon=False,fontsize=9,loc="lower center",ncol=2);ax.grid(alpha=.25);clean(ax)
    caption(fig,.07,.30,.50,.07,"six selected weight tensors: identical float32 sums at startup, before and after the run (ratio exactly 1.0).")
    tiles=[("trainable tensors",str(I["trainable_parameter_tensors"])),("LoRA","none" if not I["lora"] else "PRESENT"),("optimizer","none"),
           ("AkbasCore hooks",str(I["akbascore_hooks_after_run"])),("training mode",str(I["model_training_mode"])),("result",I["result"])]
    for i,(a,b) in enumerate(tiles):
        r,q=divmod(i,2);x=.62+q*.17;y=top-.13-r*.13
        fig.add_artist(FancyBboxPatch((x,y),.16,.11,boxstyle="round,pad=0,rounding_size=0.01",transform=fig.transFigure,facecolor=C_MODL,edgecolor=C_MOD,lw=2))
        fig.text(x+.08,y+.068,b,ha="center",va="center",fontsize=20,weight="bold",color=C_MOD);fig.text(x+.08,y+.025,a,ha="center",va="center",fontsize=11,color=C_FG)
    txt=(f"SAMPLED SHA-256 SENTINEL (16 × 256 contiguous values per selected tensor)\n startup : {I['sentinel_startup']}\n pre-run : {I['sentinel_pre_run']}\n post-run: {I['sentinel_after']}\n"
         "This is a sampled sentinel, not a cryptographic hash of every weight.")
    fit_text(fig,.05,.05,.9,.2,txt,fs_max=12,fs_min=8,family=MONO)
    foot(fig,ctx,k);return save_jpg(fig,path)
def p09_battery(ctx,k,path):
    P=ctx["P"];Bt=P["battery"];fig=plt.figure(figsize=(16,12),dpi=DPI,facecolor="white")
    top=head(fig,"AUTOMATIC QUESTION BATTERY","16 locked questions · expected vs observed · scored after generation",tag="THIS LIVE RUN")
    ax=fig.add_axes([.05,.10,.52,top-.13]);ax.axis("off");n=len(Bt);rh=1/(n+1)
    hdr=["Q","type","question","expected","observed","status"];cx=[0,.055,.20,.555,.715,.855]
    for c_,h_ in zip(cx,hdr):ax.text(c_,1-rh*.5,h_,fontsize=11,weight="bold",color=C_FG,va="center",transform=ax.transAxes)
    for i,q in enumerate(Bt):
        y=1-rh*(i+1.5);col=status_color(q["status"])
        ax.add_patch(Rectangle((0,y-rh*.45),1,rh*.9,transform=ax.transAxes,color=C_OKL if q["status"] in("CORRECT","UNKNOWN ✓") else C_ERRL,lw=0))
        vals=[q["qid"],q["kind"].lower(),q["label"],q["expected"],q["observed"],q["status"]]
        for j,(c_,v) in enumerate(zip(cx,vals)):
            ax.text(c_+.005,y,textwrap.shorten(str(v),{0:4,1:13,2:38,3:19,4:19,5:12}[j],placeholder="…"),fontsize=9.5 if j in(2,3,4,5) else 10,
                    weight="bold" if j in(0,5) else "normal",color=col if j==5 else C_FG,va="center",transform=ax.transAxes)
    M=np.array([q["lane"] for q in Bt],dtype=float)
    ax2=fig.add_axes([.61,.20,.34,top-.23]);ax2.imshow(M,aspect="auto",cmap=ListedColormap(["#F1F5F9",C_CART,C_OK]),vmin=0,vmax=2)
    ax2.set_xticks(range(16));ax2.set_xticklabels([c["cid"] for c in P["cartridges"]],rotation=90,fontsize=8.5);ax2.set_yticks(range(n));ax2.set_yticklabels([q["qid"] for q in Bt],fontsize=9.5)
    ax2.set_title("IBR LANES · which cartridge answered",fontsize=12,weight="bold",loc="left")
    caption(fig,.61,.06,.34,.11,"amber = cartridge returned a container ID (Stage 1); green = cartridge returned a place (Stage 2). Empty = that cartridge said NONE.")
    foot(fig,ctx,k);return save_jpg(fig,path)
def two_panel(fig,top,live_items,sealed_items,live_title,sealed_title,colors):
    for j,(items,title,tag) in enumerate([(live_items,live_title,"THIS LIVE RUN"),(sealed_items,sealed_title,"SEALED EXPERIMENTAL RECORD")]):
        x=.07+j*.47;ax=fig.add_axes([x,.33,.38,top-.42]);labs=[a for a,_ in items];v=[pct(t) for _,t in items]
        ax.bar(range(len(items)),[100]*len(items),color="#E2E8F0");ax.bar(range(len(items)),v,color=colors[j])
        for i,(a,t) in enumerate(items):ax.text(i,v[i]+2,f"{frac(t)}\n{v[i]:.1f}%",ha="center",va="bottom",fontsize=13,weight="bold")
        ax.set_xticks(range(len(items)));ax.set_xticklabels(labs,fontsize=10.5);ax.set_ylim(0,125);ax.set_ylabel("%",fontsize=11);clean(ax)
        ax.set_title(title,fontsize=13,weight="bold",loc="left",color=C_MOD if j==0 else C_UNK)
        fig.text(x,top-.02,tag,fontsize=12,weight="bold",color=C_MOD if j==0 else C_UNK)
def p10_linked(ctx,k,path):
    P=ctx["P"];S=P["live_summary"];H_=HIST["TEST461"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"LINKED RETRIEVAL · FOLLOWING THE CHAIN","object → container (cartridge A) → place (cartridge B), across 16 isolated cartridge rows")
    two_panel(fig,top,[("two-hop linked",tuple(S["by_kind"]["LINKED"][x] for x in("ok","n"))),("direct ID→place",tuple(S["by_kind"]["DIRECT"][x] for x in("ok","n")))],
              [("overall linked",H_["L"]),("scale 2",H_["scale"][2]["L"]),("scale 8",H_["scale"][8]["L"]),("scale 16",H_["scale"][16]["L"])],
              "this run · 16-cartridge bank","TEST461 final held-out · 16 entities",[C_OK,C_OK])
    note(fig,.05,.05,.9,.18,f"Sealed record: 43/48 linked questions correct on the final held-out panel (Wilson 95% lower bound {H_['L_lo']:.3f}). This live run uses a different, smaller locked panel; its numbers are reported separately and are not added to the sealed record.")
    foot(fig,ctx,k);return save_jpg(fig,path)
def p11_unknown(ctx,k,path):
    P=ctx["P"];S=P["live_summary"];H_=HIST["TEST461"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"MISSING CONNECTION? IT REFUSED TO INVENT ONE.","Questions whose answer is not in the cartridges must end in UNKNOWN")
    nm=P["controls"]["nomem"]
    two_panel(fig,top,[("missing link",tuple(S["by_kind"]["MISSING LINK"][x] for x in("ok","n"))),("absent ID",tuple(S["by_kind"]["ABSENT ID"][x] for x in("ok","n"))),("NO-MEMORY\nstrict hits",(nm["strict_hits"],nm["n"]))],
              [("missing link → UNKNOWN",H_["U"]),("NOMEM strict hits",H_["nomem"])],"this run","TEST461 final held-out",[C_UNK,C_UNK])
    note(fig,.05,.05,.9,.18,"Missing link: the object's container has no location cartridge. Absent ID: the container is in no cartridge at all. NO-MEMORY control: same questions with an empty cache — strict hits must be 0, proving the right answers come from the cartridges. Scope: this controlled panel only; this is not a general hallucination claim.")
    foot(fig,ctx,k);return save_jpg(fig,path)
def p12_falselink(ctx,k,path):
    P=ctx["P"];Bt=P["battery"];H_=HIST["TEST461"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"FALSE-LINK CONTROL · DECOYS PRESENT","Decoy cartridges place other containers at real places. Did the model attach them to the wrong question?")
    neg=[q for q in Bt if q["exp_place"] is None];cats=["UNKNOWN","decoy place","other place / ID"]
    M=np.zeros((len(neg),3))
    for i,q in enumerate(neg):
        o=q["obs_place"];j=0 if o is None else (1 if o in [c["Q"] for c in CHAINS] else 2);M[i,j]=1
    ax=fig.add_axes([.08,.30,.38,top-.38]);ax.imshow(M,aspect="auto",cmap=ListedColormap(["#F8FAFC",C_UNK]),vmin=0,vmax=1)
    for i in range(M.shape[0]):
        for j in range(3):ax.text(j,i,"●" if M[i,j] else "",ha="center",va="center",fontsize=16,color="white")
    ax.set_xticks(range(3));ax.set_xticklabels(cats,fontsize=11);ax.set_yticks(range(len(neg)));ax.set_yticklabels([f"{q['qid']} · {q['label'][:28]}" for q in neg],fontsize=9.5)
    ax.set_title("THIS LIVE RUN · observed outcome",fontsize=12.5,weight="bold",loc="left",color=C_MOD)
    ax2=fig.add_axes([.60,.30,.32,top-.38]);fl=H_["false_link"];ax2.bar([0,1],[fl[1]-fl[0],fl[0]],color=[C_UNK,C_ERR])
    ax2.set_xticks([0,1]);ax2.set_xticklabels(["no false link","false link"],fontsize=11)
    for i,v in enumerate([fl[1]-fl[0],fl[0]]):ax2.text(i,v+.8,f"{v}/{fl[1]}",ha="center",fontsize=14,weight="bold")
    ax2.set_ylim(0,fl[1]*1.25);ax2.set_title("SEALED RECORD · TEST461 unlinked questions",fontsize=12.5,weight="bold",loc="left",color=C_UNK);clean(ax2)
    note(fig,.05,.05,.9,.16,f"Live false links: {P['live_summary']['false_links']}. Sealed record: {frac(fl)} false links on the final held-out panel — on this sealed panel only.")
    foot(fig,ctx,k);return save_jpg(fig,path)
def p13_scale(ctx,k,path):
    P=ctx["P"];H_=HIST["TEST461"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"MORE CARTRIDGES · STILL BROADLY STABLE (UP TO 16)","Sealed record: the same questions read with 2, 8 and 16 cartridge rows",tag="SEALED EXPERIMENTAL RECORD")
    s=[2,8,16];L=[pct(H_["scale"][x]["L"]) for x in s];U=[pct(H_["scale"][x]["U"]) for x in s]
    ax=fig.add_axes([.08,.30,.56,top-.38]);ax.plot(s,L,"o-",color=C_OK,lw=3,ms=11,label="linked correct");ax.plot(s,U,"s-",color=C_UNK,lw=3,ms=11,label="missing link → UNKNOWN")
    for x,a,b in zip(s,L,U):ax.text(x,a-6,f"{a:.2f}%",ha="center",fontsize=12,weight="bold",color=C_OK);ax.text(x,b+2.5,f"{b:.0f}%",ha="center",fontsize=12,weight="bold",color=C_UNK)
    ax.set_xticks(s);ax.set_xticklabels([f"{x} cartridges" for x in s],fontsize=12);ax.set_ylim(60,110);ax.set_ylabel("%",fontsize=11);ax.grid(alpha=.25);ax.legend(frameon=False,fontsize=12,loc="lower left");clean(ax)
    lv=P["live_summary"]["by_kind"]["LINKED"];fig.text(.70,top-.10,"2 → 16 change",fontsize=15,weight="bold",color=C_FG)
    fig.text(.70,top-.16,f"linked: −{L[0]-L[2]:.2f} points",fontsize=14,color=C_OK);fig.text(.70,top-.21,f"UNKNOWN: −{U[0]-U[2]:.2f} points",fontsize=14,color=C_UNK)
    fig.text(.70,top-.31,"THIS LIVE RUN (16 cartridges)",fontsize=12,weight="bold",color=C_MOD);fig.text(.70,top-.36,f"linked {lv['ok']}/{lv['n']} (different panel)",fontsize=12,color=C_FG)
    note(fig,.05,.05,.9,.17,"Tested range only: 2, 8 and 16 cartridges. No claim is made beyond 16 or about unlimited memory.")
    foot(fig,ctx,k);return save_jpg(fig,path)
def p14_family(ctx,k,path):
    P=ctx["P"];H_=HIST["TEST461"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"WHERE IT WORKED — AND WHERE IT DIDN'T","Four ways of phrasing the same fact. F3 and F4 were first-use, held-out phrasings.",tag="SEALED EXPERIMENTAL RECORD")
    f=["F1","F2","F3","F4"];v=[pct(H_["fam"][x]) for x in f];cols=[C_OK if x>=70 else C_ERR for x in v]
    ax=fig.add_axes([.08,.33,.50,top-.42]);ax.bar(range(4),v,color=cols);ax.axhline(70,color=C_FG,ls="--",lw=2);ax.text(-.45,112,"- - -  precommitted per-family gate: 70%",ha="left",fontsize=11,weight="bold")
    for i,x in enumerate(f):ax.text(i,v[i]/2,f"{frac(H_['fam'][x])}\n{v[i]:.1f}%",ha="center",va="center",fontsize=14,weight="bold",color="white")
    ax.set_xticks(range(4));ax.set_xticklabels([x+"\n"+textwrap.fill(FAM[x][1].format(T="ID",P="PLACE"),22) for x in f],fontsize=9.5);ax.set_ylim(0,125);ax.set_ylabel("linked correct %",fontsize=11);clean(ax)
    fig.add_artist(FancyBboxPatch((.63,top-.25),.32,.2,boxstyle="round,pad=0,rounding_size=0.01",transform=fig.transFigure,facecolor=C_ERRL,edgecolor=C_ERR,lw=2.5))
    fig.text(.79,top-.10,"FINAL VERDICT",ha="center",fontsize=14,weight="bold",color=C_ERR);fig.text(.79,top-.17,H_["verdict"],ha="center",fontsize=16,weight="bold",color=C_ERR,family=MONO)
    fig.text(.79,top-.225,"F2 missed the each-family ≥ .70 gate",ha="center",fontsize=11,color=C_FG)
    lv=P["live_summary"]["by_family"];txt="THIS LIVE RUN · per family (linked + direct):\n"+"   ".join(f"{x}: {lv[x]['ok']}/{lv[x]['n']}" for x in f)
    fit_text(fig,.63,top-.42,.32,.14,txt,fs_max=12.5,fs_min=9,weight="bold",color=C_MOD)
    note(fig,.05,.05,.9,.19,"F2 phrases the location backwards (“The PLACE houses container ID”). On the sealed final panel F2 reached 7/12 and the precommitted per-family gate failed. The result is preserved unchanged: no threshold, parser or template was altered afterwards.")
    foot(fig,ctx,k);return save_jpg(fig,path)
def p15_f2(ctx,k,path):
    H_=HIST["TEST464"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"A REAL BOUNDARY CONDITION · F2 X-RAY","TEST464: the true location cartridge read alone, and with 1, 7 and 15 other cartridges",tag="SEALED EXPERIMENTAL RECORD")
    G=np.array(H_["grid"],dtype=float);ax=fig.add_axes([.10,.36,.42,top-.44]);ax.imshow(G,aspect="auto",cmap=ListedColormap([C_ERRL,C_OKL]),vmin=0,vmax=1)
    for i in range(4):
        for j in range(4):ax.text(j,i,"✓" if G[i,j] else "✗",ha="center",va="center",fontsize=24,weight="bold",color=C_OK if G[i,j] else C_ERR)
    ax.set_xticks(range(4));ax.set_xticklabels(H_["cols"],fontsize=12);ax.set_yticks(range(4));ax.set_yticklabels(H_["entities"],fontsize=12)
    stats=[("true B read alone",frac(H_["b_solo"])),("true B in larger banks",frac(H_["larger"])),("aggregation collisions",str(H_["collisions"]))]
    for i,(a,b) in enumerate(stats):
        y=top-.12-i*.12;fig.add_artist(FancyBboxPatch((.58,y),.37,.10,boxstyle="round,pad=0,rounding_size=0.01",transform=fig.transFigure,facecolor=C_BG,edgecolor=C_CTL,lw=2))
        fig.text(.60,y+.05,b,fontsize=22,weight="bold",color=C_FG,va="center");fig.text(.73,y+.05,a,fontsize=12.5,color=C_FG,va="center")
    note(fig,.05,.05,.9,.24,f"e01: the correct place appears at the start of the output (\"{H_['e01_raw']}\") but the strict first-line readout does not accept the continuation — a readout-format failure, not proof the knowledge is gone. e13: correct with 1 and 2 cartridges, fails at 8, correct again at 16 — not a monotonic capacity collapse. Verdict: {H_['verdict']} (frozen weights ✓, errors 0).")
    foot(fig,{**ctx},k);return save_jpg(fig,path)
def p16_ibr(ctx,k,path):
    H_=HIST["TEST460"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"WHY IBR EXISTS","Independent memories interfered when simply concatenated. Reading them in isolation fixed this development panel.",tag="SEALED EXPERIMENTAL RECORD")
    labs=list(H_["arms"].keys());L=[pct(H_["arms"][a][0]) for a in labs];U=[pct(H_["arms"][a][1]) for a in labs];x=np.arange(len(labs));w=.38
    ax=fig.add_axes([.07,.30,.60,top-.38]);ax.bar(x-w/2,L,w,color=C_OK,label="linked correct");ax.bar(x+w/2,U,w,color=C_UNK,label="missing link → UNKNOWN")
    for i,a in enumerate(labs):ax.text(x[i]-w/2,L[i]+1.5,frac(H_["arms"][a][0]),ha="center",fontsize=10,weight="bold");ax.text(x[i]+w/2,U[i]+1.5,frac(H_["arms"][a][1]),ha="center",fontsize=10,weight="bold")
    ax.set_xticks(x);ax.set_xticklabels(labs,fontsize=9.5);ax.set_ylim(0,118);ax.legend(frameon=False,fontsize=11,loc="upper left");clean(ax)
    ax2=fig.add_axes([.72,.30,.23,top-.38]);ax2.axis("off");ax2.set_xlim(0,1);ax2.set_ylim(0,1)
    ax2.text(.5,.95,"concatenated",ha="center",fontsize=12,weight="bold",color=C_ERR)
    for j,c in enumerate([C_CART,C_OK,C_CTL]):ax2.add_patch(Rectangle((.08+j*.28,.72),.26,.14,color=c,alpha=.8))
    ax2.text(.5,.66,"one row: memories blur together",ha="center",fontsize=10)
    ax2.text(.5,.52,"IBR (isolated rows)",ha="center",fontsize=12,weight="bold",color=C_OK)
    for j,c in enumerate([C_CART,C_OK,C_CTL]):ax2.add_patch(Rectangle((.2,.36-j*.12),.6,.09,color=c,alpha=.8))
    ax2.text(.5,.0,"each row sees one cartridge",ha="center",fontsize=10)
    note(fig,.05,.05,.9,.17,f"Mechanism verdict recorded in TEST460: {H_['mechanism']}. COFORGE (all records forged together) is a diagnostic that breaks modularity. Scope: development panel — not a universal claim about interference.")
    foot(fig,ctx,k);return save_jpg(fig,path)
def p17_batch(ctx,k,path):
    H_=HIST["TEST462"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"MEANING = SAME · TAIL TEXT = SOMETIMES DIFFERENT","TEST462: sequential reading vs batched IBR, compared answer by answer",tag="SEALED EXPERIMENTAL RECORD")
    labs=list(H_["b1"].keys());x=np.arange(len(labs));w=.38;a=[pct(H_["b1"][l]) for l in labs];b=[pct(H_["multi"][l]) for l in labs]
    ax=fig.add_axes([.08,.30,.84,top-.38]);ax.bar(x-w/2,a,w,color=C_MOD,label="batch of 1 vs sequential");ax.bar(x+w/2,b,w,color=[C_OK,C_OK,C_OK,C_CART],label="multi-row batch vs sequential")
    for i,l in enumerate(labs):ax.text(x[i]-w/2,a[i]+1.5,frac(H_["b1"][l]),ha="center",fontsize=12,weight="bold");ax.text(x[i]+w/2,b[i]+1.5,frac(H_["multi"][l]),ha="center",fontsize=12,weight="bold")
    ax.set_xticks(x);ax.set_xticklabels(labs,fontsize=13);ax.set_ylim(0,120);ax.set_ylabel("% identical",fontsize=11);ax.legend(frameon=False,fontsize=12,loc="upper right");clean(ax)
    note(fig,.05,.05,.9,.17,f"Verdict {H_['verdict']}: first token, first line and semantic answer matched in 32/32; the text after the answer matched in 20/32 for multi-row batches. Batched reading is semantically equivalent here — it is not claimed to be bit-identical.")
    foot(fig,ctx,k);return save_jpg(fig,path)
def p18_seal(ctx,k,path):
    P=ctx["P"];T=P["timing"];I=P["integrity"];E=P["environment"];fig=plt.figure(figsize=(16,12),dpi=DPI,facecolor="white")
    top=head(fig,"COMPLETE RUN · EVIDENCE SEAL","Everything below is from this run and is contained in the sealed payload",tag="THIS LIVE RUN")
    tiles=[("RUN ID",P["run_id"]),("MODEL",MODEL_ID),("GPU",E["gpu"]),("TORCH / TRANSFORMERS",f"{E['torch']} / {E['transformers']}"),("SEED",str(SEED)),
           ("DEMO PANEL LOCK",P["engine"]["lock_sha256"][:24]+"…"),("FROZEN WEIGHTS","✓ PASS" if I["result"]=="PASS" else "✗ FAIL"),("ERRORS",str(len(P["errors"]))),
           ("RUN TIME",f"{T['run']['engine_seconds']:.1f} s engine"),("CHECKS",f"{I['checks_total']} pre-seal PASS")]
    for i,(a,b) in enumerate(tiles):
        r,q=divmod(i,5);x=.05+q*.182;y=top-.12-r*.125
        fig.add_artist(FancyBboxPatch((x,y),.172,.105,boxstyle="round,pad=0,rounding_size=0.01",transform=fig.transFigure,facecolor=C_BG,edgecolor=C_MOD,lw=1.8))
        fig.text(x+.008,y+.08,a,fontsize=10,weight="bold",color=C_MOD,va="center");fit_text(fig,x+.008,y+.008,.156,.058,b,fs_max=12.5,fs_min=7.5,family=MONO)
    st=[("model load (startup)",T["startup"]["model_load_seconds"]),("codebook",P["codebook"]["seconds"]),("16 cartridges",T["run"]["cartridges_seconds"]),
        ("question battery",T["run"]["battery_seconds"]),("NOMEM control",T["run"]["control_seconds"]),("sealing",ctx["sealing_stage_seconds"])]
    ax=fig.add_axes([.25,.38,.68,top-.52]);ax.barh(range(len(st)),[s for _,s in st],color=[C_CTL,C_CART,C_CART,C_OK,C_CTL,C_FG]);ax.invert_yaxis()
    ax.set_yticks(range(len(st)));ax.set_yticklabels([a for a,_ in st],fontsize=11)
    for i,(_,s) in enumerate(st):ax.text(s,i,f" {s:.2f} s",va="center",fontsize=11,weight="bold")
    ax.set_xlim(0,max(s for _,s in st)*1.25);ax.set_xlabel("seconds",fontsize=10);clean(ax)
    txt=(f"PAYLOAD FILE : {ctx['payload_name']}\nPAYLOAD SHA-256 : {ctx['sha']}\nENGINE SOURCE : TEST461 IBR engine (463.final.py), unchanged\n"
         f"JPEG HASHES : images manifest (each image SHA-256), sealed after rendering\n"
         "Artifact integrity seal ≠ scientific proof: the SHA-256 shows the files are byte-identical to what this run recorded; the scientific claims rest on the measurements and controls.")
    fit_text(fig,.05,.05,.9,.27,txt,fs_max=12.5,fs_min=8,family=MONO)
    foot(fig,ctx,k);return save_jpg(fig,path)
POSTERS=[("01_what_just_happened","What just happened?",p01_what),("02_inside_the_engine","Inside the engine",p02_engine),
 ("03_cartridge_bank","The cartridge bank · 16 loaded",p03_bank),("04_engine_workload","How hard did the engine work?",p04_work),
 ("05_source_to_cartridge","Source → cartridge encoding",p05_compress),("06_cartridge_telemetry","Cartridge telemetry",p06_telemetry),
 ("07_source_removed","Source removal proof",p07_source),("08_frozen_model","Frozen model proof",p08_frozen),
 ("09_question_battery","Automatic question battery",p09_battery),("10_linked_retrieval","Linked retrieval",p10_linked),
 ("11_missing_link_unknown","Missing link → UNKNOWN",p11_unknown),("12_false_link_control","False-link control",p12_falselink),
 ("13_scale","Scale 2 → 8 → 16",p13_scale),("14_where_it_worked","Where it worked — and where it didn't",p14_family),
 ("15_f2_boundary","A real boundary condition",p15_f2),("16_why_ibr","Why IBR exists",p16_ibr),
 ("17_batch_equivalence","Batch semantic equivalence",p17_batch),("18_evidence_seal","Complete run · evidence seal",p18_seal)]
NPOST=len(POSTERS)
if NPOST>20 or [s[:2] for s,_,_ in POSTERS]!=[f"{i:02d}" for i in range(1,NPOST+1)]:raise RuntimeError("Poster count/numbering check failed.")
#<<POSTERS_END>>
def selftest_payload():
    rng=np.random.default_rng(0);cards=[]
    for b in BANK_SPEC:
        T=len(b["text"].split())+4
        cards.append(dict(b,T=T,cosK=list(0.98+0.01*rng.random(28)),cosV=list(0.999+0.0009*rng.random(28)),normK=list(20+5*rng.random(28)),normV=list(3+rng.random(28)),
                          code_numbers=28*(T-1)*4*248,own_numbers=28*1024,native_numbers=28*1024*T,hidden_copy_numbers=28*T*3584,encode_seconds=.05,gpu_mib=15000.0))
    bt=[]
    for q in BATTERY:
        lane=[0]*16;bt.append(dict(q,expected=q["exp_place"] or "UNKNOWN",observed="SELFTEST",obs_place=None,status="UNKNOWN ✓" if q["exp_place"] is None else "WRONG",lane=lane))
    kinds={k_:dict(ok=0,n=4) for k_ in["LINKED","MISSING LINK","DIRECT","ABSENT ID"]}
    num=lambda v,l:dict(value=v,label=l,human=human(v))
    return dict(run_id="SELFTEST",cartridges=cards,battery=bt,errors=[],
        live_summary=dict(n=16,correct=8,unknown_correct=8,false_links=0,by_kind=kinds,by_family={f:dict(ok=0,n=2) for f in["F1","F2","F3","F4"]}),
        bank=dict(slots=300,code_numbers=10**6,own_numbers=16*28*1024,native_numbers=10**6,hidden_copy_numbers=4*10**6,runtime_numbers=10**6,ratio_vs_native=.97,ratio_vs_hidden=.28),
        source_removal=dict(source_sentence_hits=0,source_sentences=16,calls=[dict(tag=f"c{i}",prompt_tokens=40,source_hits=0,slots=300) for i in range(24)]),
        controls=dict(nomem=dict(strict_hits=0,n=4)),codebook=dict(seconds=4.0,bytes=29*2**20),gpu=dict(peak_allocated_gib=16.0,name="SELFTEST GPU"),
        model=dict(params=7.6e9),ui=dict(stages=7),timing=dict(startup=dict(model_load_seconds=40.0),run=dict(cartridges_seconds=2.0,battery_seconds=20.0,control_seconds=1.0,engine_seconds=30.0)),
        workload=dict(measured_order=["forward_passes","ibr_calls","ibr_decode_steps","forward_tokens","generated_tokens","svd"],
            derived_order=["cart_encdec_mac","code_numbers","numbers_installed","memory_slot_reads","frozen_params"],estimated_order=["linear_flop","attn_flop","total_flop"],
            measured=dict(forward_passes=num(800,"transformer forward passes"),ibr_calls=num(24,"IBR batched calls"),ibr_decode_steps=num(500,"IBR decode steps"),
                          forward_tokens=num(20000,"token positions processed"),generated_tokens=num(3000,"tokens generated (all rows)"),svd=num(224,"SVD decompositions (codebook)")),
            derived=dict(cart_encdec_mac=num(10**9,"cartridge encode+decode multiply-adds"),code_numbers=num(10**6,"numbers stored in the 16 cartridges"),
                         numbers_installed=num(10**7,"numbers installed into the cache"),memory_slot_reads=num(10**10,"query × cartridge-slot reads (heads × layers)"),frozen_params=num(7.6e9,"frozen model parameters")),
            estimated=dict(linear_flop=num(3e14,"linear-layer FLOPs (est.)"),attn_flop=num(1e12,"attention FLOPs (est.)"),total_flop=num(3.01e14,"total GPU FLOPs (est.)")),
            formula_note="DERIVED cartridge MACs = Σ 2 × 28 × (T−1) × 4 × 128 × (120+128). ESTIMATED: linear = N nn.Linear weights × token positions (incl. lm_head on every position); attention = 2 × 28 heads × 128 × 28 layers × Σ attended length. FLOPs = 2 × MACs. Excludes norms, RoPE, softmax and element-wise ops."),
        integrity=dict(fingerprint_startup=[1.0]*6,fingerprint_pre_run=[1.0]*6,fingerprint_after=[1.0]*6,sentinel_startup="0"*64,sentinel_pre_run="0"*64,sentinel_after="0"*64,
            trainable_parameter_tensors=0,lora=False,akbascore_hooks_after_run=0,model_training_mode=False,result="PASS",checks_total=20),
        environment=dict(gpu="SELFTEST",torch="x",transformers="x"),engine=dict(lock_sha256=LOCK_SHA))
def render_all(ctx,run_dir):
    imgs=[]
    try:
        for i,(slug,cap,fn) in enumerate(POSTERS,1):
            p=run_dir/f"{slug}_{ctx['run_id']}.jpg";fn(ctx,i,p);imgs.append((p,cap))
    finally:plt.close("all")
    return imgs
# ---------------------------------------------------------------- RUN (generator, live stages) ----------------------------------------------------------------
def gen_config():
    try:return jsafe(json.loads(json.dumps(model.generation_config.to_dict(),default=str)))
    except Exception as ex:return f"unavailable: {ex}"
def file_ready(p):return p is not None and Path(p).is_file()and Path(p).stat().st_size>0
def prune_runs(keep=2):
    runs=sorted([p for p in ROOT.glob("CARTRIDGE-*")if p.is_dir()],key=lambda p:p.stat().st_mtime)
    for p in runs[:max(0,len(runs)-keep)]:shutil.rmtree(p,ignore_errors=True)
def post_seal_audit(imgs,zp,expected_zip_names,pp,sha,tp,mp):
    checks=[]
    def ok(name,cond):
        checks.append(name)
        if not cond:raise RuntimeError(f"POST-SEAL AUDIT FAILED: {name}")
    ok(f"{NPOST}/{NPOST} JPG created",len(imgs)==NPOST and all(file_ready(p)for p,_ in imgs))
    bad=[]
    for p,_ in imgs:
        with Image.open(p)as im:
            if not(im.format=="JPEG" and im.mode=="RGB"):bad.append(p.name)
    ok(f"{NPOST}/{NPOST} JPG JPEG/RGB validation",not bad)
    with zipfile.ZipFile(zp)as z:
        ok("ZIP testzip()",z.testzip()is None);names=z.namelist()
        ok("ZIP flat (no sub-folders)",all("/" not in n for n in names))
        ok(f"ZIP expected contents ({NPOST} JPG + images manifest + run manifest)",sorted(names)==sorted(expected_zip_names))
        jp=[n for n in names if n.lower().endswith(".jpg")];ok(f"JPG numbering 01–{NPOST:02d}",[n[:2]for n in sorted(jp)]==[f"{i:02d}" for i in range(1,NPOST+1)])
    ok("payload JSON exists",file_ready(pp));ok("payload SHA-256 recomputation",hashlib.sha256(pp.read_bytes()).hexdigest()==sha)
    ok("payload JSON parses",isinstance(json.loads(pp.read_bytes().decode("utf-8")),dict));ok("TXT exists and non-empty",file_ready(tp))
    ok("run manifest exists and parses",file_ready(mp)and json.loads(mp.read_bytes().decode("utf-8")).get("payload_sha256")==sha)
    ok("ZIP exists and non-empty",file_ready(zp));ok("no AkbasCore forward hooks",our_hooks_total()==0)
    ok("attention implementation SDPA",getattr(cfg,"_attn_implementation",None)=="sdpa")
    return checks
def make_txt(P,sha,payload_name,manifest_name,seal):
    o=[];a=o.append;S="="*110;Dd="-"*110
    a(S);a("AKBASCORE NIRVANA · COGNITIVE CARTRIDGE — READABLE RUN LOG");a(S)
    a(f"Derived from the sealed payload {payload_name} (SHA-256 {sha}). Verify with {manifest_name}.")
    a("The SHA-256 value is an artifact integrity seal; it does not by itself establish any scientific interpretation.")
    for k_,v in[("RUN ID",P["run_id"]),("START UTC",P["run_start_utc"]),("END UTC",P["run_end_utc"]),("PAYLOAD SEALED UTC",seal["sealed_utc"]),("START LOCAL",P["run_start_local"]),
               ("MODEL",MODEL_ID),("DTYPE / ATTENTION","bfloat16 / sdpa"),("LAYERS / HIDDEN / HEADS",f"{NL} / {H} / {NH}Q {NKV}KV × {HD}"),
               ("GPU",P["environment"]["gpu"]),("TORCH",P["environment"]["torch"]),("TRANSFORMERS",P["environment"]["transformers"]),("GRADIO",P["environment"]["gradio"]),
               ("SEED",SEED),("ENGINE","TEST461 IBR · K120/V128/OWN · fixed neutral codebook · source absent at readout"),("COMPUTE PATH","PyTorch CUDA backend (cuBLAS, SDPA); no custom CUDA/C++ kernel"),
               ("READOUT FRAME",FMT.replace("\n","\\n")),("DECODING",f"greedy argmax in IBR loop, max_new_tokens={MAX_NEW}"),("DEMO PANEL LOCK SHA-256",P["engine"]["lock_sha256"])]:a(f"{k_:<26}: {v}")
    a("");a("COGNITIVE CARTRIDGES (source sentences, forged once; never placed in a readout prompt)");a(Dd)
    for c in P["cartridges"]:
        a(f"{c['cid']} [{c['role']}] {c['fam']} T={c['T']} code_numbers={c['code_numbers']:,} native_numbers={c['native_numbers']:,} encode={c['encode_seconds']:.4f}s gpu={c['gpu_mib']:.0f}MiB")
        a(f"   text: {c['text']}");a("   cosK: "+" ".join(f"{v:.4f}" for v in c["cosK"]));a("   cosV: "+" ".join(f"{v:.4f}" for v in c["cosV"]))
    a("");a("AUTOMATIC QUESTION BATTERY (16 locked questions, IBR over all 16 cartridges)");a(Dd)
    for q in P["battery"]:
        a(f"{q['qid']} [{q['kind']}] {q['label']} | expected={q['expected']} | observed={q['observed']} | status={q['status']} | false_link={q['false_link']} | {q['seconds']:.3f}s")
        for stg in("stage1","stage2"):
            if q.get(stg):
                a(f"   {stg} prompt: {q[stg]['prompt']!r}")
                for cid,t in zip([c["cid"] for c in P["cartridges"]],q[stg]["texts"]):a(f"     {cid}: {t!r}")
    nm=P["controls"]["nomem"];a("");a(f"NOMEM CONTROL (PAD-only cache, no source-derived information): strict hits {nm['strict_hits']}/{nm['n']}");a(Dd)
    for r in nm["rows"]:a(f"   {r['qid']}: stage1={r['stage1']!r} stage2={r['stage2']!r} cid={r['cid']} place={r['place']} strict_hit={r['strict_hit']}")
    S_=P["live_summary"];a("");a("LIVE SUMMARY (THIS RUN)");a(Dd);a(json.dumps(S_,ensure_ascii=False))
    a("");a("SOURCE-REMOVAL AUDIT");a(Dd)
    for c in P["source_removal"]["checks"]:a("PASS · "+c)
    a(f"source sentences found in readout prompts: {P['source_removal']['source_sentence_hits']}")
    a("");a("WORKLOAD");a(Dd)
    for g in("measured","derived","estimated"):
        for k_,v in P["workload"][g].items():a(f"{g:<9} {v['label']:<46}: {v['value']:,} ({v['human']})")
    a(P["workload"]["formula_note"])
    I=P["integrity"];a("");a("INTEGRITY");a(Dd)
    for k_ in("fingerprint_startup","fingerprint_pre_run","fingerprint_after","sentinel_startup","sentinel_pre_run","sentinel_after"):a(f"{k_:<22}: {I[k_]}")
    a(f"RESULT : {I['result']} | trainable tensors {I['trainable_parameter_tensors']} | training mode {I['model_training_mode']} | AkbasCore hooks {I['akbascore_hooks_after_run']}")
    a("pre-seal checks: "+"; ".join(I["pre_seal_checks"]))
    a("");a("SEALED EXPERIMENTAL RECORD (historical, NOT produced by this run)");a(Dd);a(json.dumps(HIST,ensure_ascii=False,default=str))
    a("");a("TIMING");a(Dd)
    for g in("startup","run"):
        for k_,v in P["timing"][g].items():a(f"{g}.{k_:<30}: {v}")
    a(f"{'sealing_stage_seconds':<38}: {seal['sealing_stage_seconds']}");a("")
    a("NOTE: live results come from the locked 16-question demo panel; the sealed record is reported separately and never merged.")
    a("NOTE: the weight sentinel samples selected tensors; it is not a full cryptographic verification of every weight.");a(S)
    return "\n".join(o)
RUN_COUNTER=0;GPU_LOCK=threading.Lock()
SKIP=getattr(gr,"skip",None)or gr.update
_K=object()
FILE_KEYS=["json","txt","man"]
DL_LABELS=["⬇ DOWNLOAD FULL RUN LOG (.json)","⬇ DOWNLOAD READABLE RUN LOG (.txt)","⬇ DOWNLOAD RUN MANIFEST (.json)"]
DL_ALL_LABEL=f"⬇ DOWNLOAD ALL {NPOST} JPGs"
RAW_KEYS=["txt","json","man"]
N_OUTPUTS=2+2+len(FILE_KEYS)*2+len(RAW_KEYS)
try:
    from gradio.routes import API_PREFIX as _GR_API_PREFIX
except Exception:
    _GR_API_PREFIX="/gradio_api" if int(gr.__version__.split(".")[0])>=5 else ""
FILE_URL_PREFIX=f"{_GR_API_PREFIX}/file="
def dl_update(path,label):
    if path is None:
        try:return gr.DownloadButton(label=label,value=None,interactive=False)
        except Exception:return gr.update(value=None,interactive=False)
    try:return gr.DownloadButton(label=label,value=str(path),interactive=True)
    except Exception:return gr.update(value=str(path),interactive=True)
def btn_update(active):
    try:return gr.Button(value=DL_ALL_LABEL,interactive=bool(active))
    except Exception:return gr.update(value=DL_ALL_LABEL,interactive=bool(active))
def jpg_urls_json(paths):
    items=[]
    for p in paths:p=Path(p);items.append({"url":FILE_URL_PREFIX+quote(str(p),safe="/"),"name":p.name})
    return json.dumps(items,ensure_ascii=False)
def pack(status=_K,gal=_K,files=_K,jpgs=_K,raw=_K):
    out=[SKIP()if status is _K else status,SKIP()if gal is _K else gal]
    if jpgs is _K:out+=[SKIP(),SKIP()]
    elif jpgs is None:out+=[btn_update(False),""]
    else:
        pl=[Path(p)for p in jpgs]
        for pth in pl:
            if not file_ready(pth):raise RuntimeError(f"JPG artifact missing or empty: {pth}")
        out+=[btn_update(True),jpg_urls_json(pl)]
    if files is _K:out+=[SKIP()for _ in range(len(FILE_KEYS)*2)]
    elif files is None:out+=[dl_update(None,l)for l in DL_LABELS]+[None]*len(FILE_KEYS)
    else:
        paths=[str(files[k])for k in FILE_KEYS]
        for pth in paths:
            if not file_ready(pth):raise RuntimeError(f"Download artifact missing or empty: {pth}")
        out+=[dl_update(p,l)for p,l in zip(paths,DL_LABELS)]+paths
    if raw is _K:out+=[SKIP()for _ in RAW_KEYS]
    elif raw is None:out+=[""]*len(RAW_KEYS)
    else:out+=[raw[k]for k in RAW_KEYS]
    return tuple(out)
def card_html(title,body,kind="info"):return f'<div class="kz card {kind}"><div class="h">{html.escape(title)}</div><div class="small">{body}</div></div>'
def bar_html(done,total,color="#B45309"):
    cells="".join(f'<span style="display:inline-block;width:{94/total:.2f}%;height:14px;margin:1px;border-radius:3px;background:{color if i<done else "#e2e8f0"}"></span>' for i in range(total))
    return f'<div style="margin-top:6px">{cells}</div>'
def stage(i,title,body):return card_html(f"{i}/7 · {title}",body,"info")
READY_HTML=card_html("Ready",f"Press <b>RUN COGNITIVE CARTRIDGE DEMO</b>. The run loads 16 cartridges, removes the source, asks 16 automatic questions, seals the evidence and renders {NPOST} JPG posters.<br><b>Downloads appear when the run is complete.</b>","info")
def is_cuda_error(ex):
    s=f"{type(ex).__name__} {ex}".lower();return "cuda" in s or "device-side assert" in s or "cublas" in s
def run_handler():
    global RUN_COUNTER
    if not GPU_LOCK.acquire(blocking=False):
        yield pack(card_html("Busy","Another run is in progress. Please try again in a moment.","warn"));return
    stage_name="initialisation"
    try:
        RUN_COUNTER+=1;run_id=f"CARTRIDGE-{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}-{RUN_COUNTER:04d}"
        prune_runs(2);run_dir=ROOT/run_id;run_dir.mkdir(parents=True,exist_ok=True)
        print(f"\n{'='*110}\nRUN {run_id}\n{'='*110}")
        checks=[];errors=[]
        def chk(name,cond):
            checks.append(name)
            if not cond:raise RuntimeError(f"CHECK FAILED: {name}")
        for k_ in CNT:CNT[k_]=0
        torch.cuda.reset_peak_memory_stats();gmem=lambda:torch.cuda.memory_allocated()/2**20
        stage_name="1/7 integrity check"
        yield pack(stage(1,"Integrity check","Verifying the frozen model: hooks, trainable tensors, LoRA, optimizer and sampled weight sentinels."),[],None,None,None)
        chk("no AkbasCore forward hooks before run",our_hooks_total()==0);chk("frozen eval model before run",(not model.training)and trainable_tensors()==0 and not lora_present())
        chk("SDPA attention before run",getattr(cfg,"_attn_implementation",None)=="sdpa");chk("no optimizer before run",not optimizer_present())
        fp_pre=fingerprint();sent_pre=strong_sentinel();chk("pre-run weight fingerprint",fp_pre==FP0);chk("pre-run sampled SHA-256 sentinel",sent_pre==SENTINEL0)
        run_start_utc=utc_now();run_start_local=local_now();torch.cuda.synchronize();T0=time.perf_counter()
        stage_name="2/7 codebook"
        yield pack(stage(2,"Building the cartridge codebook","32 neutral sentences → PCA per layer × KV head (224 SVDs). The codebook knows nothing about the cartridge facts."))
        cbi=build_codebook();mem_after_cb=gmem()
        stage_name="3/7 loading cartridges";cards=[];BANKKV=[];tc=time.perf_counter()
        for i,b in enumerate(BANK_SPEC,1):
            yield pack(stage(3,f"Loading cartridge {i:02d}/16",f"<b>{b['cid']}</b> · {html.escape(b['role'])} · {b['fam']}"+bar_html(i-1,16)))
            torch.cuda.synchronize();t1=time.perf_counter();K,V,tel=packet(b["text"]);kv=install(K,V);torch.cuda.synchronize();es=time.perf_counter()-t1
            chk(f"{b['cid']} finite K/V",all(bool(torch.isfinite(t).all()) for t in list(kv[0])+list(kv[1])))
            BANKKV.append(kv);cards.append(dict(b,**tel,encode_seconds=es,gpu_mib=gmem()));del K,V
        torch.cuda.synchronize();cart_s=time.perf_counter()-tc
        stage_name="4/7 source removed"
        yield pack(stage(4,"Source removed","The engine now holds only installed cartridge numbers. Readout prompts are checked against every source sentence."+bar_html(16,16)))
        SOURCES=[b["text"] for b in BANK_SPEC];calls=[]
        def ibr(q,tag):
            out=batch(q,BANKKV);pr=STAT["last_prompt"];hits=sum(s in pr for s in SOURCES)
            calls.append(dict(tag=tag,prompt_tokens=int(STAT["last_prompt_tokens"]),source_hits=hits,slots=int(sum(kv[2] for kv in BANKKV)),seconds=float(STAT["last_seconds"]),steps=int(STAT["last_steps"])))
            if hits:raise RuntimeError("SOURCE LEAK INTO READOUT PROMPT")
            return out,pr
        stage_name="5/7 question battery";tb=time.perf_counter();BT=[]
        for i,q in enumerate(BATTERY,1):
            yield pack(stage(5,f"Reading the cartridge bank · question {i:02d}/16",html.escape(q["label"])+bar_html(i-1,16,"#047857")))
            t1=time.perf_counter();s1=None;s2=None;cid=q["cid"];lane=[0]*16
            if q["obj"]:
                t_,pr=ibr(MW1.format(obj=q["obj"]),f"{q['qid']}·S1");s1=dict(prompt=pr,texts=t_);cid=decide_id(t_)
                for j,t in enumerate(t_):
                    if parse_id(t):lane[j]=1
            place=None
            if cid:
                t_,pr=ibr(MW2.format(cid=cid),f"{q['qid']}·S2");s2=dict(prompt=pr,texts=t_);place=decide_place(t_)
                for j,t in enumerate(t_):
                    if parse_place(t,KNOWN_PLACES):lane[j]=2
            if q["exp_place"]:ok=(place==q["exp_place"]) and (q["obj"] is None or cid==q["exp_id"]);status="CORRECT" if ok else "WRONG"
            else:ok=(place is None) and (q["obj"] is None or cid==q["exp_id"]);status="UNKNOWN ✓" if ok else ("FALSE LINK" if place else "MISSED")
            BT.append(dict(q,stage1=s1,stage2=s2,obs_id=cid,obs_place=place,observed=place or ("UNKNOWN" if cid or q["obj"] is None else "no ID"),
                           expected=q["exp_place"] or "UNKNOWN",ok=bool(ok),false_link=bool(q["exp_place"] is None and place is not None),status=status,lane=lane,seconds=time.perf_counter()-t1))
            print(f" {q['qid']:<3} {q['kind']:<13} exp={str(q['exp_place'] or 'UNKNOWN'):<20} cid={cid} place={place} -> {status}",flush=True)
        torch.cuda.synchronize();bat_s=time.perf_counter()-tb;mem_after_bt=gmem()
        stage_name="6/7 control + post-run integrity"
        yield pack(stage(6,"No-memory control","The same linked questions with an empty (PAD-only) cache. Strict hits must be zero."))
        tcn=time.perf_counter();nk=nomem_kv();nrows=[]
        for q in [x for x in BATTERY if x["kind"]=="LINKED"]:
            a1=batch(MW1.format(obj=q["obj"]),[nk])[0];c1=parse_id(a1);a2=batch(MW2.format(cid=c1),[nk])[0] if c1 else "";pl=parse_place(a2,KNOWN_PLACES) if c1 else None
            nrows.append(dict(qid=q["qid"],stage1=a1,stage2=a2,cid=c1,place=pl,strict_hit=int(c1==q["exp_id"] and pl==q["exp_place"])))
        torch.cuda.synchronize();ctl_s=time.perf_counter()-tcn;engine_s=time.perf_counter()-T0
        fp_post=fingerprint();sent_post=strong_sentinel();ours=our_hooks_total()
        chk("weight fingerprint after run",fp_post==FP0);chk("sampled SHA-256 sentinel after run",sent_post==SENTINEL0);chk("AkbasCore hooks after run = 0",ours==0)
        chk("model.training == False",not model.training);chk("trainable tensors == 0",trainable_tensors()==0);chk("LoRA none",not lora_present());chk("optimizer none",not optimizer_present())
        chk("16/16 questions answered",len(BT)==16);chk("source sentences in readout prompts = 0",sum(c["source_hits"] for c in calls)==0)
        # ---------------- summaries ----------------
        kinds=["LINKED","MISSING LINK","DIRECT","ABSENT ID"]
        by_kind={k_:dict(ok=sum(q["ok"] for q in BT if q["kind"]==k_),n=sum(q["kind"]==k_ for q in BT)) for k_ in kinds}
        by_family={f:dict(ok=sum(q["ok"] for q in BT if q["fam"]==f and q["kind"] in("LINKED","DIRECT")),n=sum(q["fam"]==f and q["kind"] in("LINKED","DIRECT") for q in BT)) for f in["F1","F2","F3","F4"]}
        live=dict(n=len(BT),correct=sum(q["ok"] for q in BT),unknown_correct=sum(q["ok"] for q in BT if q["exp_place"] is None),false_links=sum(q["false_link"] for q in BT),by_kind=by_kind,by_family=by_family)
        nb=dict(slots=sum(c["T"] for c in cards),code_numbers=sum(c["code_numbers"] for c in cards),own_numbers=sum(c["own_numbers"] for c in cards),
                native_numbers=sum(c["native_numbers"] for c in cards),hidden_copy_numbers=sum(c["hidden_copy_numbers"] for c in cards),runtime_numbers=sum(c["native_numbers"] for c in cards))
        nb["ratio_vs_native"]=(nb["code_numbers"]+nb["own_numbers"])/nb["native_numbers"];nb["ratio_vs_hidden"]=(nb["code_numbers"]+nb["own_numbers"])/nb["hidden_copy_numbers"]
        num=lambda v,l:dict(value=int(v),label=l,human=human(v))
        tot_flop=2*(CNT["linear_mac"]+CNT["attn_mac"]+CNT["reproj_mac"]+CNT["cart_encdec_mac"])
        W=dict(measured=dict(forward_passes=num(CNT["forward_passes"],"transformer forward passes"),forge_passes=num(CNT["forge_passes"],"forge passes (codebook + cartridges + control)"),
                             ibr_calls=num(CNT["ibr_calls"],"IBR batched calls"),ibr_decode_steps=num(CNT["ibr_decode_steps"],"IBR decode steps"),
                             forward_tokens=num(CNT["forward_tokens"],"token positions processed"),generated_tokens=num(CNT["generated_tokens"],"tokens generated (all rows)"),
                             svd=num(CNT["svd_count"],"SVD decompositions (codebook)")),
               derived=dict(cart_encdec_mac=num(CNT["cart_encdec_mac"],"cartridge encode+decode multiply-adds"),code_numbers=num(nb["code_numbers"],"numbers stored in the 16 cartridges"),
                            numbers_installed=num(CNT["numbers_installed"],"numbers installed into the cache"),memory_slot_reads=num(CNT["memory_slot_reads"],"query × cartridge-slot reads (heads × layers)"),
                            frozen_params=num(N_PARAMS,"frozen model parameters")),
               estimated=dict(linear_flop=num(2*CNT["linear_mac"],"linear-layer FLOPs (est.)"),attn_flop=num(2*CNT["attn_mac"],"attention FLOPs (est.)"),total_flop=num(tot_flop,"total GPU FLOPs (est.)")),
               measured_order=["forward_passes","ibr_calls","ibr_decode_steps","forward_tokens","generated_tokens","svd"],derived_order=["cart_encdec_mac","code_numbers","numbers_installed","memory_slot_reads","frozen_params"],
               estimated_order=["linear_flop","attn_flop","total_flop"],
               formula_note=(f"DERIVED cartridge MACs = Σ 2 × 28 × (T−1) × 4 × 128 × (120+128). ESTIMATED: linear = {LIN_PARAMS:,} nn.Linear weights × token positions (incl. lm_head on every position); "
                             f"attention = 2 × {NH} heads × {HD} × 28 layers × Σ attended length (QKᵀ + AV, padded length as computed). FLOPs = 2 × MACs (1 MAC = 2 FLOP). "
                             "Excludes norms, RoPE, softmax and element-wise ops. Measured counters come from the code paths that ran."))
        src_rm=dict(source_sentences=16,source_sentence_hits=sum(c["source_hits"] for c in calls),calls=calls,
                    checks=["source sentence not contained in any readout prompt (asserted per IBR call)","forge sees source text only (no question)",
                            "codebook built from 32 neutral sentences disjoint from cartridges","readout prompt = question frame only; knowledge enters as installed K/V rows",
                            "NOMEM control uses a PAD-only cache with no source-derived information"])
        P={"schema":"akbascore.cartridge.run.v1","project":"AkbasCore","demo":"NIRVANA COGNITIVE CARTRIDGE","run_id":run_id,"run_start_utc":run_start_utc,"run_start_local":run_start_local,
           "run_end_utc":utc_now(),"run_end_local":local_now(),"model":{"id":MODEL_ID,"dtype":"bfloat16","attn_implementation":"sdpa","layers":NL,"hidden_size":H,"q_heads":NH,"kv_heads":NKV,"head_dim":HD,"params":N_PARAMS,"trainable_params":N_TRAINABLE,"linear_params":LIN_PARAMS},
           "environment":{"gpu":GPU_NAME,"gpu_total_gib":GPU_TOTAL/2**30,"cuda":torch.version.cuda,"torch":torch.__version__,"transformers":transformers.__version__,"gradio":gr.__version__,"numpy":np.__version__,"matplotlib":matplotlib.__version__,"python":sys.version,"platform":platform.platform()},
           "engine":{"source":"TEST461 (463.final.py) IBR engine, unchanged","lock_sha256":LOCK_SHA,"checks":[n for n,_ in ENGINE_LOCK_CHECKS],"K":K_DIM,"V":V_DIM,"own_slot0":True,"compute_path":"PyTorch CUDA backend (cuBLAS, SDPA); no custom CUDA/C++ kernel",
                     "decide_rules":"decide_id/decide_place: exactly one distinct parsed ID/place across rows else UNKNOWN; parse_place matches the bank's known places","MW1":MW1,"MW2":MW2,"FMT":FMT},
           "codebook":cbi,"cartridges":cards,"bank":nb,"battery":BT,"controls":{"nomem":dict(rows=nrows,strict_hits=sum(r["strict_hit"] for r in nrows),n=len(nrows))},
           "live_summary":live,"source_removal":src_rm,"workload":W,"counters_raw":dict(CNT),
           "gpu":{"name":GPU_NAME,"peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,"after_codebook_mib":mem_after_cb,"after_battery_mib":mem_after_bt},
           "ui":{"stages":7},"errors":errors,
           "timing":{"startup":{"model_load_seconds":MODEL_LOAD_S},"run":{"codebook_seconds":cbi["seconds"],"cartridges_seconds":cart_s,"battery_seconds":bat_s,"control_seconds":ctl_s,"engine_seconds":engine_s}},
           "integrity":{"result":"PASS","fingerprint_tensors":FP_NAMES,"fingerprint_startup":list(FP0),"fingerprint_pre_run":list(fp_pre),"fingerprint_after":list(fp_post),
                        "sentinel_method":"sampled SHA-256 over 16 contiguous 256-value slices per selected tensor; not a full verification of every weight",
                        "sentinel_startup":SENTINEL0,"sentinel_pre_run":sent_pre,"sentinel_after":sent_post,"trainable_parameter_tensors":trainable_tensors(),
                        "model_training_mode":bool(model.training),"akbascore_hooks_after_run":int(ours),"framework_hooks_after_run":framework_hooks(),"pre_seal_checks":list(checks),
                        "checks_total":len(checks),"optimizer":None,"lora":False,"fine_tuning":False},
           "historical_sealed_record":{"note":"Historical results from TEST460/461/462/464. NOT produced by this run.","data":HIST},
           "decoding":{"mode":"greedy argmax inside the IBR loop","max_new_tokens":MAX_NEW,"eos_token_id":EOS,"pad_token_id":PAD,"model_generation_config":gen_config()},
           "reproduction":{"seed":SEED,"cell_source_sha256":CELL_SOURCE_SHA,"cell_source":CELL_SOURCE}}
        stage_name="7/7 sealing"
        yield pack(stage(7,"Sealing evidence + rendering JPEGs","Canonical payload, SHA-256 artifact seal, readable log, manifest, then the poster story."))
        ts=time.perf_counter();P=jsafe(P);pb=canon(P);sha=hashlib.sha256(pb).hexdigest()
        payload_name=f"{run_id}_FULL_RUN_LOG_payload.json";manifest_name=f"run_manifest_{run_id}.json";pp=run_dir/payload_name;pp.write_bytes(pb)
        if hashlib.sha256(pp.read_bytes()).hexdigest()!=sha:raise RuntimeError("Payload hash verification failed.")
        t_sealed=time.perf_counter();sealed_utc=utc_now();seal={"sealed_utc":sealed_utc,"sealing_stage_seconds":t_sealed-ts,"seal_wall_seconds":t_sealed-T0}
        manifest={"demo":"AKBASCORE NIRVANA COGNITIVE CARTRIDGE","run_id":run_id,"payload_file":payload_name,"payload_sha256":sha,"payload_bytes":len(pb),"generated_utc":sealed_utc,
                  "timing":{"first_step_to_payload_sealed_seconds":seal["seal_wall_seconds"],"sealing_stage_seconds":seal["sealing_stage_seconds"]},"hash_algorithm":"SHA-256",
                  "canonicalization":"json.dumps(sort_keys=True, separators=(',',':'), ensure_ascii=False, allow_nan=False), UTF-8",
                  "note":"Artifact integrity seal: confirms the payload has not changed after sealing. It does not by itself establish any scientific interpretation."}
        mp=run_dir/manifest_name;mp.write_bytes(json.dumps(manifest,sort_keys=True,indent=2,ensure_ascii=False).encode("utf-8"))
        tp=run_dir/f"{run_id}_READABLE_RUN_LOG.txt";tp.write_text(make_txt(P,sha,payload_name,manifest_name,seal),encoding="utf-8")
        print(f"payload sealed: {sha}")
        ctx={"P":P,"run_id":run_id,"sha":sha,"N":NPOST,"payload_name":payload_name,**seal}
        rt=time.perf_counter();imgs=render_all(ctx,run_dir);render_s=time.perf_counter()-rt
        entries=[]
        for i,(p,cap)in enumerate(imgs,1):
            with Image.open(p)as im:w,h=im.size;fmt=im.format;mode=im.mode
            entries.append({"index":i,"filename":p.name,"title":cap,"width":w,"height":h,"format":fmt,"mode":mode,"size_bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"run_id":run_id})
        imf=run_dir/f"images_manifest_{run_id}.json"
        imf.write_bytes(json.dumps({"demo":"AKBASCORE NIRVANA COGNITIVE CARTRIDGE","run_id":run_id,"payload_file":payload_name,"payload_sha256":sha,"generated_utc":utc_now(),"render_seconds":render_s,
                                    "count":len(entries),"images":entries,"note":"SHA-256 per image is an artifact integrity seal for each file."},sort_keys=True,indent=2,ensure_ascii=False).encode("utf-8"))
        zp=run_dir/f"AKBASCORE_CARTRIDGE_JPGs_{run_id}.zip";expected=[p.name for p,_ in imgs]+[imf.name,mp.name]
        with zipfile.ZipFile(zp,"w",compression=zipfile.ZIP_DEFLATED)as z:
            for p,_ in imgs:z.write(p,arcname=p.name)
            z.write(imf,arcname=imf.name);z.write(mp,arcname=mp.name)
        audit=post_seal_audit(imgs,zp,expected,pp,sha,tp,mp);total=len(checks)+len(audit)
        print(f"{run_id}: {len(checks)} pre-seal + {len(audit)} post-seal checks = {total}/{total} PASS | render {render_s:.1f}s")
        gallery_items=[(str(p),c_)for p,c_ in imgs]
        raw_texts={"txt":tp.read_text(encoding="utf-8"),"json":pp.read_bytes().decode("utf-8"),"man":mp.read_text(encoding="utf-8")}
        L_=live["by_kind"]
        body=(f"<b>{html.escape(run_id)}</b><br>THIS LIVE RUN: {live['correct']}/{live['n']} questions correct · linked {L_['LINKED']['ok']}/4 · missing link → UNKNOWN {L_['MISSING LINK']['ok']}/4 · "
              f"direct {L_['DIRECT']['ok']}/4 · absent ID → UNKNOWN {L_['ABSENT ID']['ok']}/4 · false links {live['false_links']} · NOMEM strict hits {P['controls']['nomem']['strict_hits']}/4<br>"
              f"{NPOST} JPGs · {total}/{total} checks PASS<br>SEALED RECORD (historical): TEST461 final held-out 43/48 linked · 48/48 UNKNOWN · verdict FAIL_FINAL_HELDOUT (F2 gate)<br>"
              f'<span class="mono">payload SHA-256 (artifact integrity seal): {sha}</span>')
        yield pack(card_html("7/7 · Complete",body,"on"),gallery_items,{"json":pp,"txt":tp,"man":mp},[p for p,_ in imgs],raw_texts)
    except Exception as ex:
        print("="*110);print(f"CARTRIDGE RUN FAILED — stage: {stage_name}");print(f"exception type : {type(ex).__name__}");print(f"exception message: {ex}");traceback.print_exc();print("="*110)
        tip="<br><b>CUDA error: Runtime → Restart runtime, then run the cell again.</b>" if is_cuda_error(ex)else "<br>The full traceback is printed in the Colab console."
        yield pack(card_html("Run failed",f"Stage: {html.escape(stage_name)}<br>{html.escape(type(ex).__name__)}: {html.escape(str(ex)[:600])}{tip}","err"),[],None,None,None)
    finally:
        plt.close("all");GPU_LOCK.release()
        try:torch.cuda.empty_cache()
        except Exception:pass
# ---------------------------------------------------------------- SERVICE SELF-TEST (before the link is printed) ----------------------------------------------------------------
print("[4/7] SERVICE SELF-TEST")
def service_selftest():
    out=[];d=ROOT/"_selftest";shutil.rmtree(d,ignore_errors=True);d.mkdir(parents=True)
    (d/"w.txt").write_text("ok");out.append("ROOT writable")
    ctx={"P":selftest_payload(),"run_id":"SELFTEST","sha":"0"*64,"N":NPOST,"payload_name":"selftest.json","sealing_stage_seconds":0.1}
    imgs=render_all(ctx,d)
    for p,_ in imgs:
        with Image.open(p) as im:
            if im.format!="JPEG" or im.mode!="RGB":raise RuntimeError(f"selftest JPEG invalid: {p.name}")
    out.append(f"{len(imgs)}/{NPOST} posters rendered as JPEG/RGB with a synthetic payload")
    zp=d/"t.zip"
    with zipfile.ZipFile(zp,"w")as z:
        for p,_ in imgs:z.write(p,arcname=p.name)
    with zipfile.ZipFile(zp)as z:
        if z.testzip()is not None:raise RuntimeError("selftest zip failed")
    out.append("ZIP testzip()")
    if len(pack(READY_HTML))!=N_OUTPUTS or len(pack(READY_HTML,[],None,None,None))!=N_OUTPUTS:raise RuntimeError("callback output count mismatch")
    out.append(f"callback output count = {N_OUTPUTS}")
    if not FILE_URL_PREFIX.endswith("/file="):raise RuntimeError("file URL prefix")
    out.append(f"file route {FILE_URL_PREFIX}")
    shutil.rmtree(d,ignore_errors=True);return out
SERVICE_CHECKS=service_selftest()
for c in SERVICE_CHECKS:print(" PASS ·",c)
# ---------------------------------------------------------------- INTERFACE ----------------------------------------------------------------
print("[5/7] INTERFACE")
CSS="""
:root{--kz-on:#047857;--kz-fg:#0f172a;--kz-card:#ffffff;--kz-bd:#cbd5e1;--kz-a:#0369a1;--kz-off:#6d28d9;--kz-err:#b91c1c;--kz-warn:#b45309}
.dark{--kz-on:#34d399;--kz-fg:#f8fafc;--kz-card:#0f172a;--kz-bd:#475569;--kz-a:#38bdf8;--kz-off:#c4b5fd;--kz-err:#f87171;--kz-warn:#fbbf24}
html,body{overflow-x:hidden!important}
.gradio-container{width:100%!important;max-width:860px!important;margin:0 auto!important;padding-left:10px!important;padding-right:10px!important;box-sizing:border-box!important}
.gradio-container *{box-sizing:border-box}
.kz{color:var(--kz-fg)!important;max-width:100%;overflow-wrap:anywhere;line-height:1.45}
.kz *{color:inherit}
.hero{text-align:center;padding:10px 2px 2px}
.brand{font-size:clamp(26px,8vw,40px);font-weight:800;letter-spacing:1px;line-height:1.05}
.sub{font-size:clamp(14px,4.2vw,18px);font-weight:700;color:var(--kz-warn)!important;margin-top:4px}
.tag{font-size:clamp(13px,3.8vw,15px);opacity:.9;margin-top:6px}
.card{background:var(--kz-card);border:2px solid var(--kz-bd);border-radius:12px;padding:12px 14px;margin:6px 0}
.card.on{border-color:var(--kz-on)}.card.err{border-color:var(--kz-err)}.card.warn{border-color:var(--kz-warn)}.card.info{border-color:var(--kz-a)}
.card .h{font-weight:800;font-size:16px;margin-bottom:4px}
.small{font-size:14px}.mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;word-break:break-all}
#kz_run button,#kz_run{font-size:clamp(17px,5vw,22px)!important;font-weight:800!important;min-height:64px!important}
"""
DL_ALL_JS="""(u)=>{let items=[];try{items=JSON.parse(u||"[]")}catch(e){items=[]}
let i=0;const next=()=>{if(i>=items.length)return;const it=items[i++];const a=document.createElement('a');a.href=it.url;a.download=it.name;a.rel='noopener';
document.body.appendChild(a);a.click();setTimeout(()=>{a.remove();next()},900)};next();return u;}"""
HERO=('<div class="kz hero"><div class="brand">AKBASCORE NIRVANA</div><div class="sub">COGNITIVE CARTRIDGE</div>'
      '<div class="tag">Knowledge goes in. The source goes away. The memory remains.<br>'
      '16 synthetic knowledge cartridges are installed into a frozen Qwen2.5-7B-Instruct. The source text is removed. 16 automatic questions are answered from the cartridges — or answered UNKNOWN.</div></div>')
def _tb(**kw):
    try:return gr.Textbox(show_copy_button=True,**kw)
    except TypeError:return gr.Textbox(**kw)
def _gallery(**kw):
    try:return gr.Gallery(columns=1,height="auto",object_fit="contain",preview=False,**kw)
    except TypeError:return gr.Gallery(**kw)
try:_blocks=gr.Blocks(css=CSS,title="AKBASCORE NIRVANA · COGNITIVE CARTRIDGE")
except TypeError:_blocks=gr.Blocks(title="AKBASCORE NIRVANA · COGNITIVE CARTRIDGE")
with _blocks as demo:
    gr.HTML(HERO)
    run_btn=gr.Button("RUN COGNITIVE CARTRIDGE DEMO",variant="primary",elem_id="kz_run")
    status=gr.HTML(READY_HTML)
    gallery=_gallery(label=f"{NPOST} JPG posters (tap to open)",show_label=True)
    dl_all=gr.Button(DL_ALL_LABEL,interactive=False)
    jpg_urls=gr.Textbox(value="",visible=False)
    with gr.Row():
        dlb=[gr.DownloadButton(label=l,value=None,interactive=False) for l in DL_LABELS]
    with gr.Accordion("Direct file links (fallback)",open=False):
        fls=[gr.File(label=n,interactive=False) for n in("FULL RUN LOG (.json)","READABLE RUN LOG (.txt)","RUN MANIFEST (.json)")]
    with gr.Tabs():
        with gr.Tab("RAW LOG / HAM LOG (.txt)"):raw_txt=_tb(lines=18,max_lines=40,label="readable run log")
        with gr.Tab("FULL RUN LOG (.json)"):raw_json=_tb(lines=18,max_lines=40,label="sealed canonical payload")
        with gr.Tab("MANIFEST (.json)"):raw_man=_tb(lines=12,max_lines=30,label="run manifest")
    OUTS=[status,gallery,dl_all,jpg_urls]+dlb+fls+[raw_txt,raw_json,raw_man]
    if len(OUTS)!=N_OUTPUTS:raise RuntimeError(f"UI outputs {len(OUTS)} != {N_OUTPUTS}")
    run_btn.click(run_handler,inputs=None,outputs=OUTS)
    try:dl_all.click(None,inputs=[jpg_urls],outputs=None,js=DL_ALL_JS)
    except TypeError:dl_all.click(None,inputs=[jpg_urls],outputs=None,_js=DL_ALL_JS)
try:demo.queue(default_concurrency_limit=1)
except TypeError:demo.queue(concurrency_count=1)
print("[6/7] LAUNCH (public share link)")
print("[7/7] Open the printed gradio.live link in a new tab and press RUN COGNITIVE CARTRIDGE DEMO.")
demo.launch(share=True,allowed_paths=ALLOWED_PATHS,show_error=True,inline=False)
