# Copyright © 2026 Mustafa Akbaş
# New original AKBASCORE material is source-available under the repository LICENSE.
# Earlier MIT-licensed Titan Cognitive Core / AkbasCore material remains under its original license.
# See repository LICENSE for complete terms.
#
# TEST498B — AKBASCORE · ENDOGENOUS REASONING MICRO-CIRCUIT X-RAY
# SUCCESS↔STUCK TRANSITION SUBSPACE → SINGLE-LAYER MICRO-PULSE → FRESH HELD-OUT
#
# PURPOSE
#   Discover whether frozen Qwen contains a low-dimensional activation component associated
#   with carrying an intermediate relational result into the next mapping.
#
# PRINCIPLES
#   - NO MAM / BELLEKÖZ.
#   - NO TEST496 identities or answers.
#   - NO natural-language steering compass.
#   - Vectors are extracted only from frozen Qwen's endogenous SUCCESS-STUCK state differences.
#   - Discovery and held-out identities are disjoint.
#   - Layer/rank selection uses DISCOVERY only.
#   - HELD-OUT is touched only after the intervention is frozen.
#   - Pulse is applied at ONE frozen layer only.
#   - Pulse magnitude is frozen before held-out.
#   - +pulse, -pulse, random-subspace and OFF controls.
#   - No training / LoRA / weight update / answer vector / post-hoc held-out tuning.
#
# IMPORTANT
#   Discovery examples are paired so SUCCESS and STUCK share the same source/intermediate/terminal
#   identities. Their only controlled difference is whether the intermediate state is computationally
#   continued through the second relation or explicitly terminated at that intermediate state.
#
#   Delta_L = h_success,L - h_stuck,L
#   Per-layer centered delta matrix -> SVD
#   Candidate ranks = 1..4
#   Discovery selects ONE (layer,rank) by behavioral intervention score.
#   Held-out then evaluates that frozen choice.

import os,sys,subprocess,importlib.util,random,time,hashlib,json,math,re,gc
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
TEST="498B";SEED=4982;MODEL_ID="Qwen/Qwen2.5-7B-Instruct";DEVICE=torch.device("cuda")
TOTAL_LAYERS=28;H_EXPECT=3584;MAX_NEW=10;DISC_N=32;HOLD_N=24;RANKS=(1,2,3,4)
CAND_LAYERS=(4,6,8,10,12,14,16,18,20,22,24);PULSE_REL=0.010
TEST498A_LOCK="59fb65f14f171af26ed91f701e11eae31e861e6869f7eae32b69e51ee3e03e75"
FORBIDDEN=("MERIDIAN","ORIN","TAV","SEL","Q-23","Z-23","Q-57")

os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
torch.set_grad_enabled(False);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

# Unique synthetic identifiers. No TEST496 identities.
SYL1=["BA","CE","DI","FO","GU","HA","JE","KI","LO","MU","NA","PE","RI","SO","TU","VE","WO","XA","YE","ZU"]
SYL2=["BIN","COR","DAX","FEN","GIL","HUR","JEX","KOP","LUM","MAV","NIR","PEX","RUL","SAV","TOM","VIK","WEN","XOR","YAL","ZEP"]
def ident(i):
    a=SYL1[i%len(SYL1)];b=SYL2[(i*7+3)%len(SYL2)]
    return f"{a}{b}{100+i}"
POOL=[ident(i) for i in range(600)]
assert len(POOL)==len(set(POOL))
assert not any(f.lower() in " ".join(POOL).lower() for f in FORBIDDEN)

def make_items(start,n):
    out=[]
    for i in range(n):
        z=POOL[start+9*i:start+9*i+9]
        a,b,c,d,e,f,g,h,j=z
        chains=[(a,b,c),(d,e,f),(g,h,j)]
        qi=i%3;src,mid,tgt=chains[qi]
        order=[0,1,2];random.Random(SEED+start+i).shuffle(order)
        facts=" ".join(f"Marker {chains[k][0]} maps to relay {chains[k][1]}. Relay {chains[k][1]} maps to terminal {chains[k][2]}." for k in order)
        q=f"{facts}\n\nQUESTION: Starting from marker {src}, follow the complete mapping chain. Give only the terminal identifier."
        out.append({"id":i+1,"src":src,"mid":mid,"tgt":tgt,"chains":chains,"prompt":q})
    return out

DISC=make_items(0,DISC_N);HOLD=make_items(300,HOLD_N)
DISC_SYMS={x for it in DISC for ch in it["chains"] for x in ch};HOLD_SYMS={x for it in HOLD for ch in it["chains"] for x in ch}
assert DISC_SYMS.isdisjoint(HOLD_SYMS)

# Paired endogenous-state prompts.
# Same identities and facts; final line establishes SUCCESS continuation vs STUCK termination.
def state_prompts(it):
    src,mid,tgt=it["src"],it["mid"],it["tgt"]
    others=[c for c in it["chains"] if c[0]!=src]
    facts=(f"Marker {src} maps to relay {mid}. Relay {mid} maps to terminal {tgt}. "
           f"Marker {others[0][0]} maps to relay {others[0][1]}. Relay {others[0][1]} maps to terminal {others[0][2]}. "
           f"Marker {others[1][0]} maps to relay {others[1][1]}. Relay {others[1][1]} maps to terminal {others[1][2]}.")
    success=(f"{facts}\n\nTrace marker {src}. The first relation reaches relay {mid}. "
             f"Use relay {mid} as the input to the next relation; the chain reaches terminal {tgt}.")
    stuck=(f"{facts}\n\nTrace marker {src}. The first relation reaches relay {mid}. "
           f"Stop at relay {mid}; do not use relay {mid} as the input to the next relation.")
    return success,stuck

LOCK={"test":TEST,"seed":SEED,"model":MODEL_ID,"test498a_lock":TEST498A_LOCK,"disc_n":DISC_N,"hold_n":HOLD_N,
      "candidate_layers":CAND_LAYERS,"ranks":RANKS,"pulse_rel":PULSE_REL,"max_new":MAX_NEW,
      "discovery_symbols":sorted(DISC_SYMS),"heldout_symbols":sorted(HOLD_SYMS),
      "mam":"OFF","bellekoz":"OFF","natural_language_compass":"NONE","answer_vector":"NONE",
      "training":"NONE","lora":"NONE","weight_update":"NONE","heldout_selection":"NONE","posthoc_tuning":"NONE"}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()

print("="*160)
print("TEST498B — AKBASCORE · ENDOGENOUS REASONING MICRO-CIRCUIT X-RAY")
print("SUCCESS↔STUCK TRANSITION SUBSPACE → SINGLE-LAYER MICRO-PULSE → FRESH HELD-OUT")
print("="*160)
print("LOCK SHA:",LOCK_SHA);print("TEST498A LOCK:",TEST498A_LOCK)
print(f"DISCOVERY={DISC_N} | HELD-OUT={HOLD_N} | RANKS={RANKS} | PULSE={100*PULSE_REL:.3f}%")
print("MAM/BELLEKÖZ: OFF | NATURAL-LANGUAGE COMPASS: NONE | TEST496 SYMBOL LEAK: NONE")
T0=time.perf_counter()

print("\n[1/10] Loading frozen Qwen...")
_tv=tuple(int("".join(c for c in x if c.isdigit())or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"
tok=transformers.AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}).eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;cfg=model.config
if (len(layers),cfg.hidden_size,cfg.num_attention_heads,cfg.num_key_value_heads)!=(28,3584,28,4):raise RuntimeError("Architecture mismatch.")
_ge=model.generation_config.eos_token_id;EOS=sorted({tok.eos_token_id,*(_ge if isinstance(_ge,(list,tuple)) else [_ge])}-{None})
enc=lambda s:tok(s,add_special_tokens=False).input_ids
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | 28L H=3584 QH=28 KVH=4 | BF16 | weights frozen")

@torch.inference_mode()
def capture_last(text):
    ids=torch.tensor([enc(text)],device=DEVICE)
    o=model(input_ids=ids,output_hidden_states=True,use_cache=False,return_dict=True)
    hs=[o.hidden_states[L+1][0,-1].float().detach().cpu() for L in range(TOTAL_LAYERS)]
    del o,ids
    return hs

print("\n[2/10] Capturing endogenous SUCCESS/STUCK discovery states...")
DELTAS=[[] for _ in range(TOTAL_LAYERS)]
for i,it in enumerate(DISC,1):
    ps,pf=state_prompts(it);hs=capture_last(ps);hf=capture_last(pf)
    for L in range(TOTAL_LAYERS):DELTAS[L].append(hs[L]-hf[L])
    if i%4==0 or i==DISC_N:print(f"      {i:02d}/{DISC_N}")
del hs,hf;gc.collect();torch.cuda.empty_cache()

print("\n[3/10] SVD — endogenous transition geometry...")
BASE=[];SUBSPACE={};GEOM=[]
for L in range(TOTAL_LAYERS):
    X=torch.stack(DELTAS[L]).float()
    mu=X.mean(0);mun=float(mu.norm())
    # Uncentered SVD preserves the common SUCCESS-STUCK displacement itself.
    _,S,Vh=torch.linalg.svd(X,full_matrices=False)
    var=S.square();ratio=var/var.sum().clamp_min(1e-12)
    BASE.append(mu)
    for r in RANKS:SUBSPACE[(L,r)]=Vh[:r].contiguous()
    cosmu=torch.nn.functional.cosine_similarity(X,mu.view(1,-1),dim=1).mean().item()
    GEOM.append((mun,float(ratio[0]),float(ratio[:2].sum()),float(ratio[:4].sum()),cosmu))
    print(f"      L{L:02d} | meanΔ={mun:.5f} | PC1={100*ratio[0]:6.2f}% | PC1-2={100*ratio[:2].sum():6.2f}% | PC1-4={100*ratio[:4].sum():6.2f}% | mean cos(Δ,meanΔ)={cosmu:+.4f}")
    del X,S,Vh,var,ratio
del DELTAS;gc.collect();torch.cuda.empty_cache()

# Orient each basis so its projection on the common SUCCESS-STUCK displacement is positive.
for L in range(TOTAL_LAYERS):
    mu=BASE[L]
    for r in RANKS:
        U=SUBSPACE[(L,r)].clone()
        for k in range(r):
            if torch.dot(U[k],mu)<0:U[k].mul_(-1)
        SUBSPACE[(L,r)]=U.to(DEVICE)

def direction(L,r):
    U=SUBSPACE[(L,r)]
    mu=BASE[L].to(DEVICE)
    p=U.T@(U@mu)
    if p.norm()<1e-10:p=U[0]
    return (p/p.norm().clamp_min(1e-12)).float().contiguous()

DIR={(L,r):direction(L,r) for L in CAND_LAYERS for r in RANKS}

# Frozen random orthogonal controls generated before behavioral selection.
GEN=torch.Generator(device=DEVICE);GEN.manual_seed(SEED+991)
RAND={}
for L in CAND_LAYERS:
    for r in RANKS:
        v=torch.randn(H_EXPECT,device=DEVICE,dtype=torch.float32,generator=GEN)
        d=DIR[(L,r)]
        v=v-torch.dot(v,d)*d
        RAND[(L,r)]=(v/v.norm().clamp_min(1e-12)).contiguous()

def hook_pulse(L,vec,sign=1.0):
    vec=vec.to(DEVICE)
    telemetry={"calls":0,"requested":[],"realized":[]}
    def hook(module,args,output):
        h=output[0] if isinstance(output,tuple) else output
        rest=output[1:] if isinstance(output,tuple) else None
        n=torch.linalg.vector_norm(h.float(),dim=-1,keepdim=True).clamp_min(1e-12)
        delta=(float(sign*PULSE_REL)*n*vec.view(1,1,-1)).to(h.dtype)
        new=h+delta
        rr=(torch.linalg.vector_norm((new[:,-1]-h[:,-1]).float(),dim=-1)/torch.linalg.vector_norm(h[:,-1].float(),dim=-1).clamp_min(1e-12)).mean().item()
        telemetry["calls"]+=1;telemetry["requested"].append(PULSE_REL);telemetry["realized"].append(rr)
        return new if rest is None else (new,)+rest
    return layers[L].register_forward_hook(hook),telemetry

@torch.inference_mode()
def generate(prompt,L=None,vec=None,sign=1.0):
    handle=None;tele=None
    if L is not None:handle,tele=hook_pulse(L,vec,sign)
    try:
        ids=torch.tensor([enc(prompt)],device=DEVICE);past=None;out=[]
        for _ in range(MAX_NEW):
            if past is None:o=model(input_ids=ids,use_cache=True,return_dict=True)
            else:o=model(input_ids=ids[:,-1:],past_key_values=past,use_cache=True,return_dict=True)
            past=o.past_key_values;nxt=int(o.logits[0,-1].float().argmax())
            if nxt in EOS:break
            out.append(nxt);ids=torch.cat([ids,torch.tensor([[nxt]],device=DEVICE)],1)
        return tok.decode(out,skip_special_tokens=True).strip(),tele
    finally:
        if handle is not None:handle.remove()

def first(x):return next((z.strip() for z in str(x).splitlines() if z.strip()),"")
def clean(x):return re.sub(r'^[\s"\'`([{]+|[\s"\'`\])}.:,;!?]+$','',first(x)).upper()
def exact(x,t):return clean(x)==t.upper()
def midhit(x,m):return clean(x)==m.upper()

print("\n[4/10] DISCOVERY baseline...")
DISC_BASE=[]
for i,it in enumerate(DISC,1):
    o,_=generate(it["prompt"]);DISC_BASE.append(o)
    print(f"      [{i:02d}] {it['src']}→{it['mid']}→{it['tgt']} | OFF={first(o)!r} | target={int(exact(o,it['tgt']))} | mid={int(midhit(o,it['mid']))}")
BASE_OK=sum(exact(o,it["tgt"]) for o,it in zip(DISC_BASE,DISC))
BASE_MID=sum(midhit(o,it["mid"]) for o,it in zip(DISC_BASE,DISC))
print(f"      OFF target={BASE_OK}/{DISC_N} | intermediate={BASE_MID}/{DISC_N}")

# Selection is discovery-only.
# Score rewards rescue and terminal correctness; penalizes damage and intermediate collapse.
print("\n[5/10] DISCOVERY micro-pulse scan — layer × rank, fixed 1.000%...")
SCAN=[]
for L in CAND_LAYERS:
    for r in RANKS:
        plus=[];minus=[];rand=[]
        for it in DISC:
            a,_=generate(it["prompt"],L,DIR[(L,r)],+1);b,_=generate(it["prompt"],L,DIR[(L,r)],-1);c,_=generate(it["prompt"],L,RAND[(L,r)],+1)
            plus.append(a);minus.append(b);rand.append(c)
        pok=sum(exact(x,it["tgt"]) for x,it in zip(plus,DISC));mok=sum(exact(x,it["tgt"]) for x,it in zip(minus,DISC));rok=sum(exact(x,it["tgt"]) for x,it in zip(rand,DISC))
        pmid=sum(midhit(x,it["mid"]) for x,it in zip(plus,DISC))
        rescue=sum((not exact(o,it["tgt"])) and exact(p,it["tgt"]) for o,p,it in zip(DISC_BASE,plus,DISC))
        damage=sum(exact(o,it["tgt"]) and not exact(p,it["tgt"]) for o,p,it in zip(DISC_BASE,plus,DISC))
        score=(pok-BASE_OK)+(rescue-damage)-max(0,pok-mok==0)-0.25*pmid
        SCAN.append({"L":L,"r":r,"plus":pok,"minus":mok,"random":rok,"mid":pmid,"rescue":rescue,"damage":damage,"score":float(score)})
        print(f"      L{L:02d} R{r} | +={pok:02d} -={mok:02d} rnd={rok:02d} | rescue={rescue:02d} damage={damage:02d} mid={pmid:02d} | score={score:+.2f}")

# Deterministic frozen selection.
SCAN.sort(key=lambda x:(x["score"],x["plus"],x["rescue"],-x["damage"],x["plus"]-x["minus"],-x["r"],-x["L"]),reverse=True)
BEST=SCAN[0];SEL_L=BEST["L"];SEL_R=BEST["r"];SEL_DIR=DIR[(SEL_L,SEL_R)];SEL_RAND=RAND[(SEL_L,SEL_R)]
print(f"\n      FROZEN DISCOVERY SELECTION: L{SEL_L:02d} R{SEL_R} | +={BEST['plus']}/{DISC_N} -={BEST['minus']}/{DISC_N} random={BEST['random']}/{DISC_N} | rescue={BEST['rescue']} damage={BEST['damage']}")

# Freeze now. Held-out has not been evaluated.
FROZEN={"layer":SEL_L,"rank":SEL_R,"pulse_rel":PULSE_REL,
        "selection_row":BEST,"selection_rule":"discovery-only deterministic score"}
FROZEN_SHA=hashlib.sha256(json.dumps(FROZEN,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      INTERVENTION FREEZE SHA:",FROZEN_SHA)

print("\n[6/10] FRESH HELD-OUT — OFF / +MICRO / -MICRO / RANDOM...")
HROWS=[];TEL_SAMPLE=None
for i,it in enumerate(HOLD,1):
    off,_=generate(it["prompt"])
    plus,tel=generate(it["prompt"],SEL_L,SEL_DIR,+1)
    minus,_=generate(it["prompt"],SEL_L,SEL_DIR,-1)
    rnd,_=generate(it["prompt"],SEL_L,SEL_RAND,+1)
    if TEL_SAMPLE is None:TEL_SAMPLE=tel
    row={"off":off,"plus":plus,"minus":minus,"random":rnd}
    HROWS.append(row)
    print(f"      [{i:02d}] {it['src']}→{it['mid']}→{it['tgt']} | OFF={first(off)!r}[{int(exact(off,it['tgt']))}] | +={first(plus)!r}[{int(exact(plus,it['tgt']))}] | -={first(minus)!r}[{int(exact(minus,it['tgt']))}] | RND={first(rnd)!r}[{int(exact(rnd,it['tgt']))}]")

HO=sum(exact(x["off"],it["tgt"]) for x,it in zip(HROWS,HOLD))
HP=sum(exact(x["plus"],it["tgt"]) for x,it in zip(HROWS,HOLD))
HM=sum(exact(x["minus"],it["tgt"]) for x,it in zip(HROWS,HOLD))
HR=sum(exact(x["random"],it["tgt"]) for x,it in zip(HROWS,HOLD))
HOM=sum(midhit(x["off"],it["mid"]) for x,it in zip(HROWS,HOLD))
HPM=sum(midhit(x["plus"],it["mid"]) for x,it in zip(HROWS,HOLD))
RESC=sum((not exact(x["off"],it["tgt"])) and exact(x["plus"],it["tgt"]) for x,it in zip(HROWS,HOLD))
DAM=sum(exact(x["off"],it["tgt"]) and not exact(x["plus"],it["tgt"]) for x,it in zip(HROWS,HOLD))

# X-ray: selected pulse direction projection at downstream layers on the LAST PROMPT TOKEN.
# Since hidden dimensionality is common across Qwen layers, selected direction can be projected directly.
@torch.inference_mode()
def capture_path(prompt,pulse=False):
    cap=[None]*TOTAL_LAYERS;handles=[]
    def obs(L):
        def h(module,args,output):
            x=output[0] if isinstance(output,tuple) else output
            cap[L]=x[0,-1].detach().float().cpu().clone()
        return h
    # Observation hooks registered first; at selected layer this captures pre-pulse block output.
    for L in range(TOTAL_LAYERS):handles.append(layers[L].register_forward_hook(obs(L)))
    ph=None
    if pulse:ph,_=hook_pulse(SEL_L,SEL_DIR,+1)
    try:
        ids=torch.tensor([enc(prompt)],device=DEVICE);model(input_ids=ids,use_cache=False,return_dict=True)
    finally:
        if ph is not None:ph.remove()
        for h in handles:h.remove()
    return cap

print("\n[7/10] Downstream X-Ray — selected micro-pulse...")
XR_N=min(8,HOLD_N);XR=[]
v=SEL_DIR.detach().cpu()
for i in range(XR_N):
    a=capture_path(HOLD[i]["prompt"],False);b=capture_path(HOLD[i]["prompt"],True)
    row=[]
    for L in range(SEL_L,TOTAL_LAYERS):
        d=b[L]-a[L];rel=float(d.norm()/a[L].norm().clamp_min(1e-12));proj=float(torch.dot(d,v)/(d.norm()*v.norm()).clamp_min(1e-12))
        row.append((L,rel,proj))
    XR.append(row)
for L in range(SEL_L,TOTAL_LAYERS):
    vals=[dict((z[0],z[1:]) for z in row)[L] for row in XR]
    rel=float(np.mean([x[0] for x in vals]));proj=float(np.mean([x[1] for x in vals]))
    print(f"      L{L:02d} | mean relative Δ={100*rel:8.4f}% | cos(Δ,pulse)={proj:+.4f}")

# Additional causal-format generalization: fresh 3-edge chains never used in discovery.
THREE=[]
base=540
for i in range(8):
    a,b,c,d=POOL[base+4*i:base+4*i+4]
    q=(f"Marker {a} maps to relay {b}. Relay {b} maps to relay {c}. Relay {c} maps to terminal {d}. "
       f"Marker AUX{i} maps to relay AUXR{i}. Relay AUXR{i} maps to relay AUXS{i}. Relay AUXS{i} maps to terminal AUXT{i}.\n\n"
       f"QUESTION: Starting from marker {a}, follow the complete mapping chain. Give only the terminal identifier.")
    THREE.append((a,b,c,d,q))
print("\n[8/10] Fresh 3-edge transfer — frozen pulse...")
TROWS=[]
for i,(a,b,c,d,q) in enumerate(THREE,1):
    off,_=generate(q);plus,_=generate(q,SEL_L,SEL_DIR,+1);minus,_=generate(q,SEL_L,SEL_DIR,-1);rnd,_=generate(q,SEL_L,SEL_RAND,+1)
    TROWS.append((off,plus,minus,rnd,d))
    print(f"      [{i:02d}] {a}→{b}→{c}→{d} | OFF={first(off)!r}[{int(exact(off,d))}] | +={first(plus)!r}[{int(exact(plus,d))}] | -={first(minus)!r}[{int(exact(minus,d))}] | RND={first(rnd)!r}[{int(exact(rnd,d))}]")
TO=sum(exact(x[0],x[4]) for x in TROWS);TP=sum(exact(x[1],x[4]) for x in TROWS);TM=sum(exact(x[2],x[4]) for x in TROWS);TR=sum(exact(x[3],x[4]) for x in TROWS)

# Strict exploratory observation criterion. No requirement that baseline be weak;
# intervention must improve fresh held-out and beat directional/random controls.
PHENOMENON=(HP>HO and HP>HM and HP>HR and RESC>DAM and DAM<=3 and TP>=TO and TP>=TM and TP>=TR)

print("\n[9/10] Frozen criterion...")
print("      +MICRO > OFF held-out       :",int(HP>HO))
print("      +MICRO > -MICRO held-out    :",int(HP>HM))
print("      +MICRO > RANDOM held-out    :",int(HP>HR))
print("      rescue > damage             :",int(RESC>DAM))
print("      damage <= 3                 :",int(DAM<=3))
print("      3-edge +MICRO >= OFF        :",int(TP>=TO))
print("      3-edge +MICRO >= -MICRO     :",int(TP>=TM))
print("      3-edge +MICRO >= RANDOM     :",int(TP>=TR))

realized=[]
if TEL_SAMPLE:
    realized=TEL_SAMPLE["realized"]
REAL=np.mean(realized) if realized else float("nan")

print("\n"+"="*160)
print("[10/10] TEST498B FINAL RESULT — ENDOGENOUS REASONING MICRO-CIRCUIT X-RAY")
print("="*160)
print("MODEL                         : Qwen/Qwen2.5-7B-Instruct · frozen")
print("MAM / BELLEKÖZ                : OFF")
print("VECTOR SOURCE                 : endogenous SUCCESS-STUCK activation differences")
print("NATURAL-LANGUAGE COMPASS      : NONE")
print("TEST496 IDENTITIES            : NONE")
print("DISCOVERY / HELD-OUT          :",DISC_N,"/",HOLD_N)
print("DISCOVERY-HELDOUT OVERLAP     : 0")
print("CANDIDATE LAYERS              :",CAND_LAYERS)
print("CANDIDATE SUBSPACE RANKS      :",RANKS)
print("PULSE DOSE                    :",f"{100*PULSE_REL:.3f}%")
print("TRAINING / LoRA               : NONE")
print("WEIGHT UPDATE                 : NONE")
print("HELD-OUT PARAMETER SELECTION  : NONE")
print("POST-HOC TUNING               : NONE")
print("-"*160)
print(f"DISCOVERY OFF                 : {BASE_OK}/{DISC_N}")
print(f"SELECTED LAYER / RANK         : L{SEL_L:02d} / R{SEL_R}")
print(f"SELECTED DISCOVERY +MICRO     : {BEST['plus']}/{DISC_N}")
print(f"SELECTED DISCOVERY -MICRO     : {BEST['minus']}/{DISC_N}")
print(f"SELECTED DISCOVERY RANDOM     : {BEST['random']}/{DISC_N}")
print(f"SELECTED RESCUE / DAMAGE      : {BEST['rescue']} / {BEST['damage']}")
print("-"*160)
print(f"HELD-OUT OFF                  : {HO}/{HOLD_N}")
print(f"HELD-OUT +MICRO               : {HP}/{HOLD_N}")
print(f"HELD-OUT -MICRO               : {HM}/{HOLD_N}")
print(f"HELD-OUT RANDOM               : {HR}/{HOLD_N}")
print(f"HELD-OUT OFF MID COLLAPSE     : {HOM}/{HOLD_N}")
print(f"HELD-OUT +MICRO MID COLLAPSE  : {HPM}/{HOLD_N}")
print(f"HELD-OUT RESCUE / DAMAGE      : {RESC} / {DAM}")
print(f"HELD-OUT NET RESCUE           : {RESC-DAM:+d}")
print("-"*160)
print(f"3-EDGE OFF                    : {TO}/8")
print(f"3-EDGE +MICRO                 : {TP}/8")
print(f"3-EDGE -MICRO                 : {TM}/8")
print(f"3-EDGE RANDOM                 : {TR}/8")
print(f"REQUESTED SINGLE-LAYER DOSE   : {100*PULSE_REL:.4f}%")
print(f"REALIZED SINGLE-LAYER DOSE    : {100*REAL:.4f}%")
print("-"*160)
print("TEST498A LOCK SHA             :",TEST498A_LOCK)
print("TEST498B LOCK SHA             :",LOCK_SHA)
print("INTERVENTION FREEZE SHA       :",FROZEN_SHA)
print(f"TOTAL TEST TIME               : {time.perf_counter()-T0:.2f}s")
print("EXPLORATORY VERDICT           :","ENDOGENOUS_REASONING_MICRO_PULSE_OBSERVED" if PHENOMENON else "ENDOGENOUS_REASONING_MICRO_PULSE_NOT_YET_OBSERVED")
print("="*160)
