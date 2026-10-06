# Copyright © 2026 Mustafa Akbaş
# New original AKBASCORE MAM material is source-available under the repository LICENSE.
# Earlier MIT-licensed Titan Cognitive Core / AkbasCore material remains under its original license.
# See repository LICENSE for complete terms.
#
# TEST498A — AKBASCORE DRA · RELATIONAL EDGE-TRAVERSAL CALIBRATION
# TARGET-BLIND SYMBOLIC A→B→C PATH FOLLOWING · NO MAM / NO BELLEKÖZ
#
# PURPOSE:
#   Determine whether the frozen TEST497 DRA/SEASC mechanism can increase
#   genuine relational edge traversal before reconnecting it to MAM.
#
# TEST:
#   Each item contains three independent chains:
#       source -> intermediate -> terminal
#   Query asks for the terminal value belonging to one source.
#
# CONDITIONS:
#   DRA OFF
#   +DRA RELATIONAL-TRAVERSAL
#   -DRA directional control
#
# CRITICAL:
#   No TEST496 symbols.
#   No answer-specific vector.
#   No memory system.
#   No dose/layer/head search.
#   No training.
#   Compass examples are disjoint from evaluation chains.
#
# FROZEN TEST497 DRA:
#   L0-L19
#   IVME=0.10
#   SONUM=0.30
#   ZIRVE=0.70
#   TABAN=0.20
#   A_L = normalize(mean(h_POS)-mean(h_NEG))
#   delta = rho_L * ||h|| * A_L

import os,sys,subprocess,importlib.util,random,time,hashlib,json,math
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers

TEST="498A";SEED=498;MODEL_ID="Qwen/Qwen2.5-7B-Instruct";DEVICE=torch.device("cuda")
N_STEER=20;IVME=0.10;SONUM=0.30;ZIRVE=0.70;TABAN=0.20;MAX_NEW=12
TEST497_LOCK_SHA="e11b05a5828e837c3c8298cf54561be871aa75f1dd3e9e97ec80f14db3122c83"

# Matched contrastive compass.
# It encodes traversal policy, not any evaluation identity or answer.
POSITIVE=[
"When one relation maps an item to an intermediate item and another relation maps that intermediate item onward, follow both relations to the terminal item.",
"When a first mapping reaches an intermediate label, use that label in the next available mapping and continue to the terminal value.",
"Resolve a chained mapping by carrying the result of the first relation into the second relation.",
"An intermediate mapping result is a link to the next relation, so continue through it to obtain the terminal result.",
"Follow successive relational edges in order until the requested terminal value is reached.",
"Use the output of one mapping as the input to the next mapping when resolving a relational chain.",
"Continue from the intermediate entity through its outgoing relation before returning the requested terminal value.",
"For a multi-step mapping, traverse each connected relation rather than treating the intermediate entity as the final answer."
]
NEGATIVE=[
"When one relation maps an item to an intermediate item and another relation maps that intermediate item onward, stop at the intermediate item.",
"When a first mapping reaches an intermediate label, return that label without using the next available mapping.",
"Resolve a chained mapping by returning the result of the first relation instead of carrying it into the second relation.",
"An intermediate mapping result is treated as the final result, so do not continue through the next relation.",
"Follow only the first relational edge and return its result before reaching the requested terminal value.",
"Use the output of the first mapping as the answer instead of using it as input to the next mapping.",
"Stop at the intermediate entity rather than following its outgoing relation to the requested terminal value.",
"For a multi-step mapping, treat the intermediate entity as the final answer rather than traversing each connected relation."
]

# 24 frozen held-out evaluation items.
# Three competing chains per item. All identifiers are synthetic and unique.
RAW=[
("VEXA","LORP","NIMU","TARO","SEFK","BADI","GOMA","RULI","PEXO"),
("KEDI","FARO","WUMI","ZELA","NOKA","PURI","HESO","DAVI","JULO"),
("MIRA","TENO","BEXU","QAVI","ROPA","SILU","FENO","KARI","WEXO"),
("DUMA","HELI","PAXO","RINO","VEKA","JUSI","TOMA","BURI","NEFO"),
("SORA","KELI","VAPU","NEXA","FIDO","RUMI","BELA","TUKO","JAVI"),
("PENO","LAXI","GURU","MAVI","SETO","BEXI","DORI","HUNA","KEPO"),
("RAVI","NULO","FESA","TIMO","KUBA","JEXI","WORA","PALI","SEMU"),
("GAVI","RENO","TULA","BEXA","MOKI","HURI","SELA","JUNO","PAVI"),
("LUMA","VEXI","KARO","SENO","DUPA","MIRI","FALO","TEXU","BONA"),
("HAVI","JERO","NUPA","KESO","RIMA","VELU","TANO","BEXO","FURI"),
("WELA","PINO","SUKA","DAVI","MERO","JEXA","QULA","TIRI","BEKO"),
("FAVI","NERO","LUXA","PESO","KIMA","DURI","HATO","VEPI","JUNO"),
("TAVI","BENO","RUXA","MELA","SORI","KEPU","DINO","FARA","WEXI"),
("JAVI","KENO","PULA","REXO","TAMI","BURI","SEVO","NALI","HUMA"),
("NAVI","FESO","KURA","BEXI","MATO","JULI","PERO","SAVI","WUNO"),
("RUMA","TEKI","VANO","HESA","PURI","LEXO","GAVI","MENA","BOKI"),
("KAVI","SENO","DURA","MEXI","FALO","TUPA","JERI","BONA","WESU"),
("PAVI","LENO","RUSA","TEXI","MOKA","FURI","DAVI","HESO","JUNA"),
("BAVI","NEXO","KULA","SERI","TOMA","WUPI","FELA","RINO","GEXA"),
("DAVI","KESO","MURA","PEXI","LATO","SINU","HAVI","JERO","BUMA"),
("FUMA","REKI","TANO","JEXI","BELA","SORI","KAVI","MENO","PUSA"),
("HUMA","VESO","LARI","TEXA","NOKI","BURI","SEMI","DOPA","JUNO"),
("JUMA","PERO","FEXI","KAVI","SUTO","MENA","RILO","BEKA","WUSA"),
("NEMA","TURI","KEXO","FAVI","ROLI","SUNA","BEMA","JESO","PAXI")
]

# Rotate which chain is queried so target is not always in same textual position.
ITEMS=[]
for i,r in enumerate(RAW):
    a,b,c,d,e,f,g,h,j=r
    chains=[(a,b,c),(d,e,f),(g,h,j)]
    qi=i%3
    src,mid,target=chains[qi]
    order=[0,1,2]
    random.Random(SEED+i).shuffle(order)
    statements=[]
    for x in order:
        s,m,t=chains[x]
        statements.append(f"Marker {s} maps to relay {m}. Relay {m} maps to terminal {t}.")
    context=" ".join(statements)
    q=f"{context}\n\nQUESTION: Starting from marker {src}, what terminal is reached after following the complete mapping chain? Give only the terminal identifier."
    ITEMS.append({"id":i+1,"query":q,"source":src,"mid":mid,"target":target,"chains":chains})

# Leakage audit.
eval_symbols={z.casefold() for r in RAW for z in r}
axis_words=set((" ".join(POSITIVE+NEGATIVE)).casefold().replace(".","").replace(",","").split())
LEAK=sorted(eval_symbols & axis_words)
assert not LEAK,f"COMPASS/EVALUATION SYMBOL LEAK: {LEAK}"

LOCK={"test":TEST,"seed":SEED,"model":MODEL_ID,"test497_lock":TEST497_LOCK_SHA,
"dra":{"layers":"L0-L19","ivme":IVME,"sonum":SONUM,"zirve":ZIRVE,"taban":TABAN},
"positive":POSITIVE,"negative":NEGATIVE,"items":ITEMS,
"mam":"OFF","bellekoz":"OFF","retrieval":"OFF","training":"NONE","dose_search":"NONE",
"layer_search":"NONE","head_search":"NONE","posthoc_tuning":"NONE"}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()

os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
torch.set_grad_enabled(False)
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

print("="*160)
print("TEST498A — AKBASCORE DRA · RELATIONAL EDGE-TRAVERSAL CALIBRATION")
print("TARGET-BLIND SYMBOLIC A→B→C PATH FOLLOWING · NO MAM / NO BELLEKÖZ")
print("="*160)
print("LOCK SHA:",LOCK_SHA)
print("TEST497 LOCK:",TEST497_LOCK_SHA)
print("EVALUATION ITEMS:",len(ITEMS))
print("COMPASS/EVALUATION SYMBOL LEAK: NONE")
print(f"DRA: L0-L{N_STEER-1} | IVME={IVME:.2f} SONUM={SONUM:.2f} ZIRVE={ZIRVE:.2f} TABAN={TABAN:.2f}")
T0=time.perf_counter()

print("\n[1/8] Loading frozen Qwen...")
_tv=tuple(int("".join(c for c in x if c.isdigit())or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"
tok=transformers.AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}).eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;cfg=model.config
if (len(layers),cfg.hidden_size,cfg.num_attention_heads,cfg.num_key_value_heads)!=(28,3584,28,4):
    raise RuntimeError("Architecture mismatch.")
_ge=model.generation_config.eos_token_id
EOS=sorted({tok.eos_token_id,*(_ge if isinstance(_ge,(list,tuple)) else [_ge])}-{None})
enc=lambda s:tok(s,add_special_tokens=False).input_ids
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | 28L H=3584 QH=28 KVH=4 | frozen BF16")

def envelope(L):
    t=float(L)
    kb=ZIRVE*math.exp(-SONUM*t)*(1.0+SONUM*t)+TABAN
    return kb/(ZIRVE+TABAN)
DOSES=[IVME*envelope(L) for L in range(N_STEER)]
print("      DRA envelope:",", ".join(f"L{i}:{100*d:.3f}%" for i,d in enumerate(DOSES)))

@torch.inference_mode()
def mean_states(texts):
    acc=[torch.zeros(cfg.hidden_size,dtype=torch.float32,device=DEVICE) for _ in range(N_STEER)]
    for text in texts:
        ids=torch.tensor([enc(text)],device=DEVICE)
        o=model(input_ids=ids,output_hidden_states=True,use_cache=False,return_dict=True)
        for L in range(N_STEER):
            acc[L].add_(o.hidden_states[L+1][0,-1].float())
    return [x/len(texts) for x in acc]

print("\n[2/8] Extracting frozen target-blind edge-traversal compass...")
P=mean_states(POSITIVE);N=mean_states(NEGATIVE);ACT=[]
for L in range(N_STEER):
    d=P[L]-N[L];raw=torch.linalg.vector_norm(d).item()
    ACT.append((d/torch.linalg.vector_norm(d).clamp_min(1e-12)).float().contiguous())
    print(f"      L{L:02d} raw={raw:.6f} dose={100*DOSES[L]:.3f}%")
del P,N;torch.cuda.empty_cache()

TELEM=None
def install_dra(sign):
    global TELEM
    TELEM=[{"calls":0,"realized":[]} for _ in range(N_STEER)]
    hs=[]
    for L in range(N_STEER):
        A=ACT[L];rho=DOSES[L]
        def mk(li,a,r,s):
            def hook(module,args,output):
                if isinstance(output,tuple):h=output[0];rest=output[1:]
                else:h=output;rest=None
                n=torch.linalg.vector_norm(h.float(),dim=-1,keepdim=True).clamp_min(1e-12)
                delta=(float(s*r)*n*a.view(1,1,-1)).to(h.dtype)
                new=h+delta
                oldlast=h[:,-1].float();newlast=new[:,-1].float()
                rr=(torch.linalg.vector_norm(newlast-oldlast,dim=-1)/torch.linalg.vector_norm(oldlast,dim=-1).clamp_min(1e-12)).mean().item()
                TELEM[li]["calls"]+=1;TELEM[li]["realized"].append(rr)
                return new if rest is None else (new,)+rest
            return hook
        hs.append(layers[L].register_forward_hook(mk(L,A,rho,sign)))
    return hs

def rm(hs):
    for h in hs:h.remove()

@torch.inference_mode()
def generate(prompt,dra=0):
    hs=install_dra(+1 if dra>0 else -1) if dra else []
    try:
        ids=torch.tensor([enc(prompt)],device=DEVICE);out=[]
        for _ in range(MAX_NEW):
            o=model(input_ids=ids,use_cache=False,return_dict=True)
            nxt=int(o.logits[0,-1].float().argmax())
            if nxt in EOS:break
            out.append(nxt);ids=torch.cat([ids,torch.tensor([[nxt]],device=DEVICE)],1)
        return tok.decode(out,skip_special_tokens=True).strip()
    finally:rm(hs)

def firstline(x):return next((z.strip() for z in str(x).splitlines() if z.strip()),"")
def canon(x):return firstline(x).strip().upper()
def ok(x,target):return canon(x)==target.upper()

print("\n[3/8] Frozen 24-item A/B/C evaluation...")
RESULTS=[]
for it in ITEMS:
    off=generate(it["query"],0)
    plus=generate(it["query"],+1)
    minus=generate(it["query"],-1)
    row={"id":it["id"],"source":it["source"],"mid":it["mid"],"target":it["target"],
         "off":off,"plus":plus,"minus":minus,
         "off_ok":ok(off,it["target"]),"plus_ok":ok(plus,it["target"]),"minus_ok":ok(minus,it["target"]),
         "off_mid":ok(off,it["mid"]),"plus_mid":ok(plus,it["mid"]),"minus_mid":ok(minus,it["mid"])}
    RESULTS.append(row)
    print(f"      [{it['id']:02d}] {it['source']}→{it['mid']}→{it['target']} | OFF={firstline(off)!r} [{int(row['off_ok'])}] | +DRA={firstline(plus)!r} [{int(row['plus_ok'])}] | -DRA={firstline(minus)!r} [{int(row['minus_ok'])}]")

OFF_OK=sum(x["off_ok"] for x in RESULTS);PLUS_OK=sum(x["plus_ok"] for x in RESULTS);MINUS_OK=sum(x["minus_ok"] for x in RESULTS)
OFF_MID=sum(x["off_mid"] for x in RESULTS);PLUS_MID=sum(x["plus_mid"] for x in RESULTS);MINUS_MID=sum(x["minus_mid"] for x in RESULTS)
RESCUE=sum((not x["off_ok"]) and x["plus_ok"] for x in RESULTS)
DAMAGE=sum(x["off_ok"] and not x["plus_ok"] for x in RESULTS)
ANTI_RESCUE=sum((not x["off_ok"]) and x["minus_ok"] for x in RESULTS)

print("\n[4/8] Transition anatomy...")
print(f"      OFF correct terminal       : {OFF_OK}/24")
print(f"      +DRA correct terminal      : {PLUS_OK}/24")
print(f"      -DRA correct terminal      : {MINUS_OK}/24")
print(f"      OFF intermediate collapse  : {OFF_MID}/24")
print(f"      +DRA intermediate collapse : {PLUS_MID}/24")
print(f"      -DRA intermediate collapse : {MINUS_MID}/24")
print(f"      +DRA rescued OFF failures  : {RESCUE}")
print(f"      +DRA damaged OFF successes : {DAMAGE}")
print(f"      -DRA rescued OFF failures  : {ANTI_RESCUE}")

# Harder 3-edge chain: source -> relay1 -> relay2 -> terminal.
THREE=[
("AXEN","BILO","CURA","DEVI"),
("EFRA","GUNO","HAPI","JEXO"),
("KURA","LENO","MAVI","NUSA"),
("PEXI","QARO","RUMI","SEVA"),
("TUNO","VEXA","WORI","YELA"),
("ZARI","BELO","CEXI","DUMA"),
("FENO","GARI","HUXA","JIVO"),
("KESO","LURI","MEXA","NIVO")
]
print("\n[5/8] Three-edge generalization — no compass change...")
THREE_ROWS=[]
for i,(a,b,c,d) in enumerate(THREE,1):
    q=(f"Marker {a} maps to relay {b}. Relay {b} maps to relay {c}. Relay {c} maps to terminal {d}. "
       f"Marker ROKA maps to relay SEDI. Relay SEDI maps to relay TUPA. Relay TUPA maps to terminal VENO.\n\n"
       f"QUESTION: Starting from marker {a}, what terminal is reached after following the complete mapping chain? Give only the terminal identifier.")
    off=generate(q,0);plus=generate(q,+1);minus=generate(q,-1)
    THREE_ROWS.append((ok(off,d),ok(plus,d),ok(minus,d),off,plus,minus))
    print(f"      [{i:02d}] {a}→{b}→{c}→{d} | OFF={firstline(off)!r} [{int(ok(off,d))}] | +DRA={firstline(plus)!r} [{int(ok(plus,d))}] | -DRA={firstline(minus)!r} [{int(ok(minus,d))}]")
THREE_OFF=sum(x[0] for x in THREE_ROWS);THREE_PLUS=sum(x[1] for x in THREE_ROWS);THREE_MINUS=sum(x[2] for x in THREE_ROWS)

# Directionality must matter. +DRA should outperform both OFF and -DRA.
DELTA_PLUS=PLUS_OK-OFF_OK
DELTA_MINUS=MINUS_OK-OFF_OK
NET_RESCUE=RESCUE-DAMAGE

print("\n[6/8] Directionality summary...")
print(f"      +DRA Δ terminal accuracy : {DELTA_PLUS:+d}/24")
print(f"      -DRA Δ terminal accuracy : {DELTA_MINUS:+d}/24")
print(f"      +DRA net rescue          : {NET_RESCUE:+d}")
print(f"      3-edge OFF               : {THREE_OFF}/8")
print(f"      3-edge +DRA              : {THREE_PLUS}/8")
print(f"      3-edge -DRA              : {THREE_MINUS}/8")

# Predeclared exploratory criterion.
# Need genuine positive directionality, fewer intermediate collapses, and no large
# destruction of already-correct behavior. No parameter adjustment is allowed here.
PHENOMENON=(
    PLUS_OK>OFF_OK and
    PLUS_OK>MINUS_OK and
    PLUS_MID<OFF_MID and
    RESCUE>DAMAGE and
    DAMAGE<=3 and
    THREE_PLUS>=THREE_OFF
)

print("\n[7/8] Frozen decision...")
print("      Criterion 1 +DRA > OFF terminal       :",int(PLUS_OK>OFF_OK))
print("      Criterion 2 +DRA > -DRA terminal      :",int(PLUS_OK>MINUS_OK))
print("      Criterion 3 +DRA fewer mid collapses  :",int(PLUS_MID<OFF_MID))
print("      Criterion 4 rescue > damage           :",int(RESCUE>DAMAGE))
print("      Criterion 5 damage <= 3               :",int(DAMAGE<=3))
print("      Criterion 6 3-edge +DRA >= OFF        :",int(THREE_PLUS>=THREE_OFF))

print("\n"+"="*160)
print("[8/8] TEST498A FINAL RESULT — RELATIONAL EDGE-TRAVERSAL CALIBRATION")
print("="*160)
print("MODEL                         : Qwen/Qwen2.5-7B-Instruct · frozen")
print("MAM / BELLEKÖZ                : OFF")
print("RETRIEVAL / ROUTER            : OFF")
print("STEERING                      : TEST497 AkbasCore 3.0 SEASC/DRA")
print("STEERED LAYERS                : L0-L19")
print(f"IVME / SONUM / ZIRVE / TABAN  : {IVME:.2f} / {SONUM:.2f} / {ZIRVE:.2f} / {TABAN:.2f}")
print("COMPASS                       : EDGE-TRAVERSAL ↔ INTERMEDIATE-STOP")
print("EVALUATION SYMBOLS IN COMPASS : NONE")
print("ANSWER VECTOR                 : NONE")
print("TRAINING / LoRA               : NONE")
print("DOSE SEARCH                   : NONE")
print("LAYER / HEAD SEARCH           : NONE")
print("POST-HOC TUNING               : NONE")
print("-"*160)
print(f"2-EDGE OFF TERMINAL            : {OFF_OK}/24")
print(f"2-EDGE +DRA TERMINAL           : {PLUS_OK}/24")
print(f"2-EDGE -DRA TERMINAL           : {MINUS_OK}/24")
print(f"OFF INTERMEDIATE COLLAPSE      : {OFF_MID}/24")
print(f"+DRA INTERMEDIATE COLLAPSE     : {PLUS_MID}/24")
print(f"-DRA INTERMEDIATE COLLAPSE     : {MINUS_MID}/24")
print(f"+DRA RESCUES                   : {RESCUE}")
print(f"+DRA DAMAGES                   : {DAMAGE}")
print(f"+DRA NET RESCUE                : {NET_RESCUE:+d}")
print(f"3-EDGE OFF                     : {THREE_OFF}/8")
print(f"3-EDGE +DRA                    : {THREE_PLUS}/8")
print(f"3-EDGE -DRA                    : {THREE_MINUS}/8")
print("-"*160)
print("TEST497 LOCK SHA              :",TEST497_LOCK_SHA)
print("TEST498A LOCK SHA             :",LOCK_SHA)
print(f"TOTAL TEST TIME               : {time.perf_counter()-T0:.2f}s")
print("EXPLORATORY VERDICT           :","TARGET_BLIND_EDGE_TRAVERSAL_PRESSURE_OBSERVED" if PHENOMENON else "TARGET_BLIND_EDGE_TRAVERSAL_PRESSURE_NOT_OBSERVED")
print("="*160)
