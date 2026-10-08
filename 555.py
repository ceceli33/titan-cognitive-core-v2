# TEST555 — AKBASCORE MAM · LAYERWISE CAUSAL STATE TRANSPLANT
# TEST554 FOLLOW-UP · FROZEN MISTRAL · 24 WORLDS · 5 FACTS
# JOINT / BLOCK / SWITCH-ONLY / FULL / TARGET / HISTORY
# SAME TOKENS · SAME POSITIONS · CAUSAL ATTENTION ACCESS CONTROL
# NO TRAINING · NO RETRIEVAL · NO ROUTER · NO QSF
import os,sys,subprocess,importlib.util,random,time,hashlib,json,re,inspect,gc
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
TEST="555";SEED=552552;PANEL_SEED=550550
MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
DEVICE=torch.device("cuda");DTYPE=torch.bfloat16
MAX_NEW=16;NL_EXPECT=32;KPOP=5
FACT_TYPES=["CURRENT","NEAR","FORMER","ROLE"]
QUERY_CYCLE=["CURRENT","FORMER","NEAR","ROLE"]
POSITIONS=["FIRST","MIDDLE","LAST"]
SCOPES=["FULL","TARGET","HISTORY"]
LAYERS=list(range(32))
if not torch.cuda.is_available():raise RuntimeError("CUDA gerekli.")
os.environ["TOKENIZERS_PARALLELISM"]="false"
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
T0=time.perf_counter()
print("="*160)
print("TEST555 — AKBASCORE MAM · LAYERWISE CAUSAL STATE TRANSPLANT")
print("JOINT / BLOCK / SWITCH-ONLY / FULL / TARGET / HISTORY")
print("FROZEN MISTRAL · SAME TOKENS · SAME POSITIONS · NO TRAINING")
print("="*160)

print("\n[1/10] Frozen Mistral...")
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",dtype=DTYPE).eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;NL=len(model.model.layers)
H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads
HD=getattr(cfg,"head_dim",None) or H//QH
if (NL,H,QH,KVH,HD)!=(32,4096,32,8,128):raise RuntimeError("Architecture mismatch.")
print("MODEL:",MODEL_ID)
print("GPU:",torch.cuda.get_device_name(0))
print("TORCH:",torch.__version__,"TRANSFORMERS:",transformers.__version__)
print("ROPE:",getattr(cfg,"rope_parameters",None))
print("TRAINABLE:",sum(int(p.requires_grad) for p in model.parameters()))
print("DECODER SIGNATURE:",inspect.signature(model.model.layers[0].forward))
FP=[model.model.layers[0].self_attn.q_proj.weight,model.model.layers[8].self_attn.o_proj.weight,model.model.layers[16].mlp.down_proj.weight,model.model.layers[24].self_attn.o_proj.weight,model.model.layers[31].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
def sentinel():
    h=hashlib.sha256()
    for p in FP:
        a=p.detach().reshape(-1);n=a.numel()
        for j in range(16):
            o=j*(n-256)//15
            h.update(a[o:o+256].float().cpu().numpy().tobytes())
    return h.hexdigest()
S0=sentinel()
def sha_obj(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def norm(s):
    s=re.sub(r"[^\w\s-]"," ",s.casefold().strip(),flags=re.UNICODE)
    return " ".join(s.split())
def hit(s,g):return re.search(r"(?<!\w)"+re.escape(norm(g))+r"(?!\w)",norm(s)) is not None

BASE=[
("Zorvan","Melket","Dravel","Oakhaven","Pelnor"),("Kelvar","Nareth","Solven","Branik","Tarsen"),
("Tarev","Luneth","Varos","Cedran","Mireth"),("Belnor","Arven","Dorel","Kesmar","Falven"),
("Ravik","Selora","Terven","Maldor","Nerik"),("Nemor","Calven","Istral","Pareth","Dovren"),
("Darsen","Velora","Keldin","Orvek","Sarnel"),("Feron","Talven","Merith","Sovran","Belvik"),
("Larev","Nerith","Calder","Veyron","Tormek"),("Torven","Elsar","Marvek","Dorin","Kaleth"),
("Selnor","Kareth","Valen","Ordan","Mervek"),("Mirev","Taldor","Neris","Kelmar","Sorvik"),
("Varen","Solith","Deran","Malvek","Cordan"),("Kelor","Ardin","Velmar","Toren","Narell"),
("Narev","Belith","Corven","Sareth","Dorvik"),("Dervan","Mirel","Talvek","Orsen","Kelron"),
("Calnor","Verith","Naldor","Seren","Parvek"),("Parel","Dorven","Kelith","Maros","Tervik"),
("Sorven","Tarell","Vindor","Nelmar","Calrek"),("Barel","Corith","Laven","Derik","Solmar"),
("Ralen","Mervor","Talith","Kesven","Noreth"),("Norel","Valdor","Serith","Calvenor","Darvek"),
("Tervan","Orel","Mardin","Velos","Karven"),("Karev","Solen","Dareth","Mirven","Talrek")
]
W=[]
for i,(s,current,near,former,role_target) in enumerate(BASE):
    facts=[
        ("CURRENT",f"The current capital of {s} is {current}.",current),
        ("NEAR",f"The largest city of {s} is {near}.",near),
        ("FORMER",f"The former capital of {s} was {former}.",former),
        ("ROLE",f"The current capital of {role_target} is {s}.",s)
    ]
    queries=[
        ("CURRENT",f"What is the current capital of {s}?",current),
        ("FORMER",f"What was the former capital of {s}?",former),
        ("NEAR",f"What is the largest city of {s}?",near),
        ("ROLE",f"What is the current capital of {role_target}?",s)
    ]
    W.append({"id":i,"facts":facts,"queries":queries})
SYSTEM="Answer the question using only the information provided. Give only the requested name and nothing else."
def source_prefix(f):return f"<s>[INST] {SYSTEM}\n\nINFORMATION:\n{f}\n\n"
def canonical_prefix():return f"<s>[INST] {SYSTEM}\n\nINFORMATION:\n"
def query_suffix(q):return f"QUESTION:\n{q}\n\nANSWER: [/INST]"
PREFIX_IDS=tok(canonical_prefix(),add_special_tokens=False).input_ids
P=len(PREFIX_IDS)
CAR=[];KEY_TO_IDX={}
for wi,w in enumerate(W):
    for typ,fact,gold in w["facts"]:
        idx=len(CAR)
        ids=tok(source_prefix(fact),add_special_tokens=False).input_ids
        if ids[:P]!=PREFIX_IDS:raise RuntimeError("Prefix mismatch.")
        CAR.append({"idx":idx,"world":wi,"type":typ,"fact":fact,"gold":gold,"body":ids[P:]})
        KEY_TO_IDX[(wi,typ)]=idx
if len(CAR)!=96:raise RuntimeError("Cartridge panel mismatch.")

print("\n[2/10] TEST553 panel provenance...")
rng=random.Random(PANEL_SEED);pool=list(range(96));rng.shuffle(pool)
PANEL=[];cursor=0
for wi in range(24):
    typ=QUERY_CYCLE[wi%4];target=KEY_TO_IDX[(wi,typ)]
    others=[]
    while len(others)<4:
        ci=pool[cursor%96];cursor+=1
        if ci==target or CAR[ci]["world"]==wi or any(CAR[x]["world"]==CAR[ci]["world"] for x in others):continue
        others.append(ci)
    PANEL.append({"case":wi,"target":target,"others":others})
PANEL_SHA=sha_obj(PANEL)
EXPECTED="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
print("PANEL SHA:",PANEL_SHA)
if PANEL_SHA!=EXPECTED:raise RuntimeError("TEST553 provenance FAIL.")
print("PANEL PROVENANCE: PASS")

def order_keys(z,slot):
    d=list(z["others"]);t=z["target"]
    if slot=="FIRST":return [t]+d
    if slot=="MIDDLE":return d[:2]+[t]+d[2:]
    if slot=="LAST":return d+[t]
    raise ValueError(slot)

def cache_layers(pkv):
    if hasattr(pkv,"layers"):
        out=[]
        for layer in pkv.layers:
            k=getattr(layer,"keys",None);v=getattr(layer,"values",None)
            if k is None:k=getattr(layer,"key_cache",None)
            if v is None:v=getattr(layer,"value_cache",None)
            if k is None or v is None:raise RuntimeError("Cache layer API mismatch.")
            out.append((k,v))
        if out:return tuple(out)
    if hasattr(pkv,"key_cache"):return tuple(zip(pkv.key_cache,pkv.value_cache))
    raise RuntimeError("Cache API mismatch.")
def clone_layers(pkv):return tuple((k.detach().clone(),v.detach().clone()) for k,v in cache_layers(pkv))
def build_cache(layers):
    c=DynamicCache()
    for li,(k,v) in enumerate(layers):c.update(k,v,li)
    return c
def cache_len(c):return int(c.get_seq_length())

@torch.no_grad()
def answer_layers(layers,q):
    cache=build_cache(layers);physical=cache_len(cache)
    ids=torch.tensor(tok(query_suffix(q),add_special_tokens=False).input_ids,device=DEVICE).reshape(1,-1)
    qlen=ids.shape[1]
    pos=torch.arange(physical,physical+qlen,device=DEVICE).unsqueeze(0)
    cp=pos.squeeze(0);att=torch.ones((1,physical+qlen),device=DEVICE,dtype=torch.long)
    out=model(input_ids=ids,past_key_values=cache,attention_mask=att,position_ids=pos,cache_position=cp,use_cache=True,return_dict=True)
    cache=out.past_key_values;nxt=out.logits[:,-1,:].argmax(-1,keepdim=True)
    eos=tok.eos_token_id;eosset=set() if eos is None else {int(eos)}
    gen=[]
    for step in range(MAX_NEW):
        gen.append(nxt)
        if int(nxt.item()) in eosset:break
        physical=cache_len(cache)
        pos=torch.tensor([[physical]],device=DEVICE)
        cp=pos.squeeze(0);att=torch.ones((1,physical+1),device=DEVICE,dtype=torch.long)
        out=model(input_ids=nxt,past_key_values=cache,attention_mask=att,position_ids=pos,cache_position=cp,use_cache=True,return_dict=True)
        cache=out.past_key_values;nxt=out.logits[:,-1,:].argmax(-1,keepdim=True)
    return tok.decode(torch.cat(gen,dim=1)[0],skip_special_tokens=True).strip()

# Attention masks are applied at the decoder-layer boundary, not by changing
# model weights or attention kernels. The full prompt is always processed once.
# BLOCK: body tokens see canonical prefix + their own body, not other bodies.
# JOINT: ordinary causal attention.
# SWITCH: BLOCK through layer L; JOINT after layer L.
def make_case(keys,target):
    ids=list(PREFIX_IDS);ranges={}
    for ci in keys:
        a=len(ids);ids.extend(CAR[ci]["body"]);ranges[ci]=(a,len(ids))
    n=len(ids);group=torch.full((n,),-1,dtype=torch.long,device=DEVICE)
    for j,ci in enumerate(keys):
        a,b=ranges[ci];group[a:b]=j
    i=torch.arange(n,device=DEVICE)
    causal=i[:,None]>=i[None,:]
    allowed=causal & ((group[:,None]==-1)|(group[None,:]==-1)|(group[:,None]==group[None,:]))
    # Prefix queries cannot see later bodies because of the causal condition.
    block=torch.zeros((1,1,n,n),dtype=DTYPE,device=DEVICE)
    block.masked_fill_(~allowed,torch.finfo(DTYPE).min)
    joint=torch.zeros((1,1,n,n),dtype=DTYPE,device=DEVICE)
    joint.masked_fill_(~causal,torch.finfo(DTYPE).min)
    a,b=ranges[target]
    target_mask=torch.zeros(n,dtype=torch.bool,device=DEVICE);target_mask[a:b]=True
    history_mask=torch.zeros(n,dtype=torch.bool,device=DEVICE)
    for ci in keys:
        if ci!=target:
            x,y=ranges[ci];history_mask[x:y]=True
    full_mask=torch.arange(n,device=DEVICE)>=P
    return torch.tensor(ids,dtype=torch.long,device=DEVICE).reshape(1,-1),joint,block,{"FULL":full_mask,"TARGET":target_mask,"HISTORY":history_mask},ranges

# Hooks are temporary and removed after each forward. Joint reference hidden
# states are captured at every decoder boundary. The intervention is at the
# OUTPUT of layer L, therefore K/V for layers <= L remain BLOCK-generated.
@torch.no_grad()
def write_pass(ids,joint,block,mode="JOINT",cut=None,scope=None,reference=None,mask=None,capture=False):
    handles=[];states={}
    try:
        for li,layer in enumerate(model.model.layers):
            def prehook(module,args,kwargs,li=li):
                if mode=="JOINT":chosen=joint
                elif mode=="BLOCK":chosen=block
                elif mode in ("SWITCH","TRANSPLANT"):
                    chosen=block if li<=cut else joint
                else:raise ValueError(mode)
                kwargs["attention_mask"]=chosen
                return args,kwargs
            handles.append(layer.register_forward_pre_hook(prehook,with_kwargs=True))
            if capture:
                def record(module,args,output,li=li):
                    h=output[0] if isinstance(output,(tuple,list)) else output
                    states[li]=h.detach().clone()
                handles.append(layer.register_forward_hook(record))
            if mode=="TRANSPLANT" and li==cut:
                def transplant(module,args,output):
                    if reference is None or cut not in reference:raise RuntimeError("Missing joint state.")
                    h=output[0] if isinstance(output,(tuple,list)) else output
                    ref=reference[cut]
                    if h.shape!=ref.shape:raise RuntimeError("Transplant shape mismatch.")
                    patched=torch.where(mask.reshape(1,-1,1),ref,h)
                    if isinstance(output,tuple):return (patched,)+output[1:]
                    if isinstance(output,list):return [patched]+output[1:]
                    return patched
                handles.append(layer.register_forward_hook(transplant))
        n=ids.shape[1]
        pos=torch.arange(n,dtype=torch.long,device=DEVICE).unsqueeze(0)
        out=model(input_ids=ids,attention_mask=torch.ones((1,n),dtype=torch.long,device=DEVICE),position_ids=pos,cache_position=pos.squeeze(0),use_cache=True,return_dict=True)
        layers=clone_layers(out.past_key_values)
        if len(layers)!=NL or any(k.shape[-2]!=n or v.shape[-2]!=n for k,v in layers):raise RuntimeError("Write cache shape mismatch.")
        del out
        return layers,states
    finally:
        for h in handles:h.remove()

print("\n[3/10] ATTENTION MASK VALIDATION...")
z=PANEL[0];keys=order_keys(z,"MIDDLE");target=z["target"]
ids,joint,block,masks,ranges=make_case(keys,target)
n=ids.shape[1]
if not torch.equal(joint[0,0],torch.triu(joint[0,0],diagonal=1)+torch.tril(torch.zeros_like(joint[0,0]))):
    raise RuntimeError("Joint mask malformed.")
for ci in keys:
    a,b=ranges[ci]
    for cj in keys:
        if ci==cj:continue
        x,y=ranges[cj]
        if torch.any(block[0,0,a:b,x:y]==0):raise RuntimeError("Cross-body mask leakage.")
print("MASK CONTRACT: PASS | TOKENS:",n,"| PREFIX:",P)

print("\n[4/10] BASELINE VALIDATION — JOINT vs BLOCK...")
BASELINES=[]
for ci,z in enumerate(PANEL):
    typ=CAR[z["target"]]["type"]
    q=next(x[1] for x in W[ci]["queries"] if x[0]==typ)
    gold=CAR[z["target"]]["gold"]
    for slot in POSITIONS:
        keys=order_keys(z,slot)
        ids,joint,block,masks,ranges=make_case(keys,z["target"])
        jl,_=write_pass(ids,joint,block,"JOINT")
        bl,_=write_pass(ids,joint,block,"BLOCK")
        ja=answer_layers(jl,q);ba=answer_layers(bl,q)
        BASELINES.append({"case":ci,"slot":slot,"gold":gold,"joint":ja,"block":ba,"joint_ok":hit(ja,gold),"block_ok":hit(ba,gold)})
        del jl,bl
    print(f"CASE {ci:02d} JOINT="+"/".join(str(int(x["joint_ok"])) for x in BASELINES[-3:])+" BLOCK="+"/".join(str(int(x["block_ok"])) for x in BASELINES[-3:]))
BJ={s:sum(x["joint_ok"] for x in BASELINES if x["slot"]==s) for s in POSITIONS}
BB={s:sum(x["block_ok"] for x in BASELINES if x["slot"]==s) for s in POSITIONS}
print("JOINT:",BJ)
print("BLOCK:",BB)
JOINT_GATE=all(BJ[s]>=22 for s in POSITIONS)
BLOCK_GATE=all(BB[s]<=12 for s in POSITIONS)
print("JOINT GATE:","PASS" if JOINT_GATE else "FAIL")
print("BLOCK GATE:","PASS" if BLOCK_GATE else "FAIL")
if not JOINT_GATE or not BLOCK_GATE:
    raise RuntimeError("Baseline gate failed. Layerwise transplant would be uninterpretable.")

print("\n[5/10] CAUSAL TRANSPLANT — 32 LAYERS × 3 SCOPES...")
RESULTS=[];SWITCH=[]
for ci,z in enumerate(PANEL):
    typ=CAR[z["target"]]["type"]
    q=next(x[1] for x in W[ci]["queries"] if x[0]==typ)
    gold=CAR[z["target"]]["gold"]
    print(f"\nCASE {ci:02d} {typ:8s} GOLD={gold}")
    for slot in POSITIONS:
        keys=order_keys(z,slot)
        ids,joint,block,masks,ranges=make_case(keys,z["target"])
        ref_layers,reference=write_pass(ids,joint,block,"JOINT",capture=True)
        del ref_layers
        for cut in LAYERS:
            sw_layers,_=write_pass(ids,joint,block,"SWITCH",cut=cut)
            sw_answer=answer_layers(sw_layers,q)
            SWITCH.append({"case":ci,"slot":slot,"layer":cut,"answer":sw_answer,"ok":bool(hit(sw_answer,gold))})
            del sw_layers
            for scope in SCOPES:
                layers,_=write_pass(ids,joint,block,"TRANSPLANT",cut=cut,reference=reference,mask=masks[scope])
                ans=answer_layers(layers,q)
                RESULTS.append({"case":ci,"slot":slot,"layer":cut,"scope":scope,"answer":ans,"ok":bool(hit(ans,gold))})
                del layers
        print(f"  {slot:6s} complete | 32 layers | SWITCH + FULL/TARGET/HISTORY")
        del reference
    if (ci+1)%4==0:print(f"PROGRESS {ci+1}/24 | elapsed={time.perf_counter()-T0:.1f}s")

print("\n[6/10] LAYERWISE ACCURACY...")
SUMMARY={}
for scope in ["SWITCH"]+SCOPES:
    SUMMARY[scope]={}
    src=SWITCH if scope=="SWITCH" else RESULTS
    for cut in LAYERS:
        rr=[r for r in src if r["layer"]==cut]
        n=sum(int(r["ok"]) for r in rr)
        SUMMARY[scope][cut]=n
    print(f"{scope:8s}: "+" ".join(f"L{l:02d}={SUMMARY[scope][l]:02d}/72" for l in LAYERS))

print("\n[7/10] CAUSAL INCREMENT OVER SWITCH-ONLY...")
INCREMENT={}
for scope in SCOPES:
    INCREMENT[scope]={}
    for cut in LAYERS:
        a=SUMMARY[scope][cut];b=SUMMARY["SWITCH"][cut]
        INCREMENT[scope][cut]=a-b
    best=max(LAYERS,key=lambda l:INCREMENT[scope][l])
    print(f"{scope:8s} BEST L={best:02d} | transplant={SUMMARY[scope][best]}/72 | switch={SUMMARY['SWITCH'][best]}/72 | Δ={INCREMENT[scope][best]:+d}")

print("\n[8/10] TARGET / HISTORY / FULL comparison...")
for cut in LAYERS:
    print(f"L{cut:02d} SWITCH={SUMMARY['SWITCH'][cut]:02d} FULL={SUMMARY['FULL'][cut]:02d} TARGET={SUMMARY['TARGET'][cut]:02d} HISTORY={SUMMARY['HISTORY'][cut]:02d} /72")
BEST_FULL=max(LAYERS,key=lambda l:INCREMENT["FULL"][l])
BEST_TARGET=max(LAYERS,key=lambda l:INCREMENT["TARGET"][l])
BEST_HISTORY=max(LAYERS,key=lambda l:INCREMENT["HISTORY"][l])
print("BEST FULL:",BEST_FULL)
print("BEST TARGET:",BEST_TARGET)
print("BEST HISTORY:",BEST_HISTORY)

print("\n[9/10] INTERPRETATION...")
max_full=max(INCREMENT["FULL"].values())
max_target=max(INCREMENT["TARGET"].values())
max_history=max(INCREMENT["HISTORY"].values())
if max_full<=0 and max_target<=0 and max_history<=0:
    DECISION="NO POSITIVE STATE-TRANSPLANT INCREMENT OVER SWITCH-ONLY"
elif max_target>0 and max_target>=max_history:
    DECISION="TARGET-STATE TRANSPLANT SHOWS POSITIVE CAUSAL INCREMENT"
elif max_history>0 and max_history>max_target:
    DECISION="HISTORY-STATE TRANSPLANT SHOWS POSITIVE CAUSAL INCREMENT"
else:
    DECISION="FULL-STATE TRANSPLANT SHOWS POSITIVE INCREMENT; SUBSET LOCALIZATION UNRESOLVED"
print("DECISION:",DECISION)
print("CAUTION: Transplant changes hidden states; cached K/V at and before the cut remain BLOCK-derived.")
print("CAUTION: Positive increments demonstrate intervention effects, not a unique physical bridge.")

print("\n[10/10] ANTI-DRIFT / FINAL RECORD...")
WEIGHT_OK=sentinel()==S0
TRAINABLE=sum(int(p.requires_grad) for p in model.parameters())
print("WEIGHT SENTINEL:", "PASS" if WEIGHT_OK else "FAIL")
print("TRAINABLE:",TRAINABLE)
print("RETRIEVAL: NO | ROUTER: NO | TRAINING: NO | QSF: NO")
LOCK={
    "test":TEST,"model":MODEL_ID,"panel_sha":PANEL_SHA,
    "layers":LAYERS,"scopes":SCOPES,"positions":POSITIONS,
    "attention":"joint/block/switch","transplant_boundary":"decoder_layer_output",
    "cache_at_cut":"block_derived","cache_after_cut":"joint_attention_with_transplanted_hidden_state",
    "training":False,"retrieval":False,"router":False,"qsf":False
}
LOCK_SHA=sha_obj(LOCK)
FINAL={"test":TEST,"lock_sha":LOCK_SHA,"panel_sha":PANEL_SHA,"baseline_joint":BJ,"baseline_block":BB,"summary":SUMMARY,"increment":INCREMENT,"decision":DECISION,"weight_ok":WEIGHT_OK,"trainable":TRAINABLE}
RESULT_SHA=sha_obj(FINAL)
print("\n"+"="*160)
print("TEST555 — FINAL RESEARCH RECORD")
print("="*160)
print("MODEL:",MODEL_ID)
print("PANEL SHA:",PANEL_SHA)
print("LOCK SHA:",LOCK_SHA)
print("RESULT SHA:",RESULT_SHA)
print("JOINT BASELINE:",BJ)
print("BLOCK BASELINE:",BB)
print("BEST FULL L:",BEST_FULL,"Δ:",max_full)
print("BEST TARGET L:",BEST_TARGET,"Δ:",max_target)
print("BEST HISTORY L:",BEST_HISTORY,"Δ:",max_history)
print("WEIGHT SENTINEL:",WEIGHT_OK)
print("TRAINABLE:",TRAINABLE)
print("TOTAL TIME:",f"{time.perf_counter()-T0:.2f}s")
print("RESEARCH DECISION:",DECISION)
print("="*160)
if not WEIGHT_OK or TRAINABLE:raise RuntimeError("ANTI-DRIFT FAILURE.")
