# AKBASCORE NIRVANA · MISTRAL — COGNITIVE CARTRIDGE FINAL DEMO (single Colab cell)
# ENGINE = TEST474 (474.MISTRAL.final.py), transplanted without semantic change: S1_RECORD source frame, C_VERIFY readout frames,
#   source-only forge, fixed 32-sentence neutral codebook, D120 K + D120 V, OWN first-token state preserved, independent cartridges,
#   module-wise sequential readout (one generate call per cartridge), unique aggregation, greedy decoding, frozen weights.
#   NO router · NO hidden scorer · NO training · NO LoRA · NO optimizer · NO weight update.
# RUNTIME DIFFERENCE vs TEST474 (disclosed): with FAST_FIRST_LINE=True each readout generation stops at the end of its first answer line. The parsers
#   read only that line, and greedy decoding is causal, so parsed results are identical; 13 probe reads are re-decoded at full length and compared first.
#   Set FAST_FIRST_LINE=False to decode exactly as TEST474 (up to 32 tokens per read, much slower).
# DEMO LAYER (measurement / presentation only): telemetry computed from tensors after the engine produced them, raw-output recording,
#   source-removal audit, result re-derivation from raw text, poster rendering, stale-reference scan, sealing, ZIP.
# Result classes: LIVE (measured in this run) · SEALED (historical record from the named test log) · DERIVED (exact formula) — never merged.
import os,sys,io,re,json,math,time,html,random,shutil,hashlib,zipfile,platform,textwrap,threading,subprocess,importlib.util,traceback,gc
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import quote
from collections import Counter
#<<CONST_BEGIN>>
SEED=474;D=120;DMAX=128;MAX_NEW=32;SEP="\n\n";FAST_FIRST_LINE=True
MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";MODEL_SHORT="Mistral-7B-Instruct-v0.3";ARCH=(32,4096,32,8,128)
OBJECTS=["amber sextant","bronze compass","cedar telescope","crimson lantern","ivory astrolabe","jade chronometer","silver monocle","copper barometer",
"onyx sundial","pearl compass","saffron telescope","violet lantern","marble astrolabe","teal chronometer","golden monocle","indigo barometer"]
IDS=[f"RQ-{415+i*17}"for i in range(16)]
PLACES=["elm lodge","pine archive","cedar vault","birch station","maple gallery","willow depot","oak library","fir observatory",
"ash museum","yew tower","spruce hall","linden archive","alder vault","beech station","hazel gallery","rowan depot"]
NEUTRAL=[
"Jonas Weber carried the wooden crate across the quiet market square.","Priya Nair wrote a long letter to her cousin in the morning.",
"The small boat drifted slowly toward the rocky shore.","Omar Haddad fixed the broken clock in the village school.",
"A tired teacher closed the green door of the library.","Sofia Rossi baked fresh bread for the harvest festival.",
"The children watched the kites rising above the hill.","Liam O'Connor sold his old bicycle to a neighbor.",
"Heavy rain flooded the narrow street near the market.","Nadia Petrova translated the ancient manuscript into French.",
"The farmer counted the sheep before sunset.","Hiro Sato opened a tiny bakery beside the river.",
"An old dog slept under the kitchen table all afternoon.","Carlos Mendes washed the windows of his grandmother's house.",
"The museum guard locked the heavy gate at midnight.","Fatima Zahra grew tomatoes in the backyard garden.",
"The pilot announced a short delay because of fog.","Anna Kowalski found a lost wallet on the bus.",
"Snow covered the mountain village during the night.","Ravi Kumar taught his brother how to play chess.",
"The chef sharpened every knife before dinner service.","Lucas Martin cleaned the roof of the barn after the storm.",
"A young violinist practiced scales in the empty hall.","Mei Lin delivered the package to the wrong address.",
"Marek Novak placed the glass bottle beneath the wooden bench.","Sara Ibrahim carried a red notebook into the quiet classroom.",
"Noah Schmidt repaired the small radio beside the kitchen window.","Yuki Mori left a paper envelope near the station entrance.",
"Amira Hassan moved the ceramic bowl onto the upper shelf.","Peter Novak opened the metal box behind the old theater.",
"Lucia Costa placed the yellow scarf inside the travel bag.","Daniel Kim carried a black umbrella through the central courtyard."]
MISSOBJ=[f"missing object {i+1}"for i in range(8)]
ABSIDS=[f"ZX-{801+i*13}"for i in range(8)]
ID_RE=re.compile(r"\b([A-Z]{2}-\d{3})\b",re.I)
def first(text):
    line=next((s.strip()for s in str(text).splitlines()if s.strip()),"")
    line=re.sub(r"^\W*(final answer|answer)\s*[:\-]\s*","",line,flags=re.I).strip(" *`\"'")
    return re.split(r"(?<=[.!?])\s+",line)[0]if line else""
def is_none(text):
    s=first(text).upper()
    return bool(re.search(r"\bNONE\b",s)) and not ID_RE.search(s)
def parse_id(text):
    s=first(text)
    if is_none(s):return None
    m=ID_RE.search(s);return m.group(1).upper()if m else None
def parse_place(text):
    s=first(text).casefold()
    if re.search(r"\bnone\b",s):return None
    hits=[p for p in PLACES if re.search(r"(?<!\w)"+re.escape(p.casefold())+r"(?!\w)",s)]
    return hits[0]if len(set(hits))==1 else None
def decide_id(raws):
    vals=[parse_id(x)for x in raws];u=sorted({x for x in vals if x is not None})
    return u[0]if len(u)==1 else None,u
def decide_place(raws):
    vals=[parse_place(x)for x in raws];u=sorted({x for x in vals if x is not None})
    return u[0]if len(u)==1 else None,u
def src(o,c,p):return f"MEMORY RECORD\nObject: {o}\nContainer: {c}\nLocation: {p}"
def q1(o):return f'QUESTION:\nDoes this memory explicitly contain the object "{o}"? If yes, answer only its container identifier. If no, answer exactly NONE.\n\nANSWER:'
def q2(c):return f'QUESTION:\nDoes this memory explicitly contain container "{c}"? If yes, answer only its location. If no, answer exactly NONE.\n\nANSWER:'
# TEST474 precommitted gates (thresholds replicated verbatim from 474.MISTRAL.final.py)
GATE_MIN={"mw1":15,"mw2":15,"linked":15,"missing":7,"absent_id":7,"nomem":7}
SEALED474=dict(log="474.log",mw1=(16,16),mw2=(16,16),linked=(16,16),missing=(8,8),absent_id=(8,8),nomem=(8,8),sentinel="PASS",verdict="PASS_MISTRAL_FINAL_ENGINE",
               env="NVIDIA A100-SXM4-40GB | torch 2.11.0+cu130 | transformers 5.17.0")
SEALED473=dict(log="473.log",frame="S1_RECORD + C_VERIFY",gold=(16,16),decoy_none=(240,240),mw1=(16,16),missing=(8,8),conditions=12,tied_perfect=4,verdict="PASS_MISTRAL_ABSTENTION_FRAME")
SEALED467=dict(log="467.log",cases=8,baseline=(8,8),
    entry={"L07":(8,8,8),"L08":(6,7,7),"L09":(6,8,7),"L10":(5,6,5),"L11":(6,7,6),"L12":(6,7,8),"L13":(0,3,0),"L14":(1,3,0),"L15":(0,1,0)},
    exit={"L18":(1,0,0),"L19":(4,2,3),"L20":(4,3,3),"L21":(7,4,7),"L22":(7,4,7),"L23":(8,6,8),"L24":(8,7,8),"L25":(8,7,8),"L26":(8,8,8),"L27":(8,8,8)},
    bounds={"K":("L07","L08","L23"),"V":("L09","L08","L26"),"KV":("L12","L08","L23")})
LOCK_MATERIAL={"engine":"TEST474 · S1_RECORD + C_VERIFY · D120 K/V","objects":OBJECTS,"ids":IDS,"places":PLACES,"neutral":NEUTRAL,"missing":MISSOBJ,"absent":ABSIDS,
               "frames":[src("O","C","P"),q1("O"),q2("C")],"D":D,"DMAX":DMAX,"max_new":MAX_NEW,"sep":SEP,"gates":GATE_MIN}
class AuditFail(RuntimeError):pass
def jsafe(o):
    if isinstance(o,dict):return{str(k):jsafe(v)for k,v in o.items()}
    if isinstance(o,(list,tuple)):return[jsafe(v)for v in o]
    if hasattr(o,"item")and not isinstance(o,(str,bytes)):
        try:o=o.item()
        except Exception:pass
    if isinstance(o,float):return o if math.isfinite(o)else f"non-finite:{o}"
    if isinstance(o,(str,int,bool))or o is None:return o
    return str(o)
def canon(o):return json.dumps(jsafe(o),sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")
LOCK_SHA=hashlib.sha256(canon(LOCK_MATERIAL)).hexdigest()
def fr(t):return f"{t[0]}/{t[1]}"
def derive(raw):
    """Re-derive every result from the raw first-generation texts, using the verbatim TEST474 parsing/decision rules."""
    c1=[];dec1=[];g1=w1n=w1o=0;rpq=[]
    for q in sorted(raw["mw1"],key=lambda z:z["q"]):
        i=q["q"];rs=[r["raw"]for r in q["reads"]];rpq.append(len(rs));d,u=decide_id(rs);dec1.append(d);cl=[]
        for j,r in enumerate(rs):
            if j==i:k="gold"if parse_id(r)==IDS[i]else"gold_miss";g1+=int(k=="gold")
            elif is_none(r):k="none";w1n+=1
            else:k="other";w1o+=1
            cl.append(k)
        c1.append(cl)
    mw1=sum(int(dec1[i]==IDS[i])for i in range(16))
    c2=[];dec2=[];g2=w2n=w2o=0;skipped=[];linked=0;mw2=0
    for q in sorted(raw["mw2"],key=lambda z:z["q"]):
        i=q["q"]
        if q.get("skipped"):dec2.append(None);c2.append(None);skipped.append(i);continue
        if q["id"]!=dec1[i]:raise AuditFail(f"MW2 query {i} used ID {q['id']} but MW1 decided {dec1[i]}")
        rs=[r["raw"]for r in q["reads"]];rpq.append(len(rs));d,u=decide_place(rs);dec2.append(d);cl=[]
        for j,r in enumerate(rs):
            if j==i:k="gold"if parse_place(r)==PLACES[i]else"gold_miss";g2+=int(k=="gold")
            elif is_none(r):k="none";w2n+=1
            else:k="other";w2o+=1
            cl.append(k)
        c2.append(cl);ok=d==PLACES[i];mw2+=int(ok);linked+=int(ok and q["id"]==IDS[i])
    def neg(items,dec,field):
        out=[]
        for q in sorted(items,key=lambda z:z["i"]):
            rs=[r["raw"]for r in q["reads"]];rpq.append(len(rs));d,u=dec(rs)
            out.append(dict(i=q["i"],query=q[field],none=sum(int(is_none(r))for r in rs),n=len(rs),uniq=u,decided=d,ok=bool(d is None and len(u)==0)))
        return out
    ms=neg(raw["missing"],decide_id,"query");ab=neg(raw["absent"],decide_place,"query")
    nm=[]
    for q in sorted(raw["nomem"],key=lambda z:z["i"]):
        p=parse_place(q["raw"]);nm.append(dict(i=q["i"],id=q["id"],first=first(q["raw"]),parsed=p,target=PLACES[q["i"]],ok=bool(p!=PLACES[q["i"]])))
    cnt={"mw1":mw1,"mw2":mw2,"linked":linked,"missing":sum(int(m["ok"])for m in ms),"absent_id":sum(int(m["ok"])for m in ab),"nomem":sum(int(m["ok"])for m in nm)}
    gates={k:bool(cnt[k]>=GATE_MIN[k])for k in GATE_MIN}
    tot={"mw1":16,"mw2":16,"linked":16,"missing":8,"absent_id":8,"nomem":8}
    return dict(counts=cnt,totals=tot,gates=gates,verdict="PASS_MISTRAL_FINAL_ENGINE"if all(gates.values())else"FAIL_MISTRAL_FINAL_ENGINE",
        mw1=dict(decided=dec1,cls=c1,gold=g1,wrong_none=w1n,wrong_other=w1o,wrong_total=w1n+w1o),
        mw2=dict(decided=dec2,cls=c2,gold=g2,wrong_none=w2n,wrong_other=w2o,wrong_total=w2n+w2o,skipped=skipped),
        missing=ms,absent=ab,nomem=nm,reads_per_query=sorted(set(rpq)),
        replay_match=bool(all(cnt[k]==SEALED474[k][0]for k in cnt)))
# ---- stale-reference scanner: scans GENERATED output (poster text, JSON, TXT, file names), never its own source ----
ALLOWED_TESTS={"467","473","474"}
STALE=[("model-family-A",r"[Qq]wen"),("28-layer",r"\b28[ -]?(?:transformer )?(?:layers?|L)\b"),("hidden-3584",r"(?i:hidden[^0-9]{0,15}3584|3584[^0-9]{0,6}hidden|\bH\s?=?\s?3584\b)"),
       ("V128",r"V\s?128"),("old-family-label",r"\bF[1-4]\b"),("gpu-A100",r"A100"),("batched-reader",r"\bIBR\b|Isolated Batched"),("doi",r"zenodo|10\.5281"),("4-KV-heads",r"\b4\s?KV\b")]
def stale_scan(name,text,allow=()):
    for a in allow:
        if a:text=text.replace(a,"<allowed>")
    hits=[(name,lab,m.group(0))for lab,p in STALE for m in re.finditer(p,text)]
    hits+=[(name,"old-test-number",m.group(0))for m in re.finditer(r"TEST\s?(\d+)",text)if m.group(1)not in ALLOWED_TESTS]
    return hits
def cpu_selftest():
    n=0
    for t,want in[("NONE",True),("NONE\nYou are correct. The record does not mention it.",True),("RQ-415\nYou are an AI assistant.",False),("",False),("Container RQ-415",False)]:
        assert is_none(t)==want,(t,is_none(t));n+=1
    assert parse_id("RQ-415\nxyz")=="RQ-415" and parse_id("Container rq-432.")=="RQ-432" and parse_id("NONE")is None and parse_id("NONE, but RQ-415")=="RQ-415";n+=4
    assert parse_place("elm lodge\nYou are an AI")=="elm lodge" and parse_place("The elm lodge.")=="elm lodge" and parse_place("NONE")is None and parse_place("elm lodge or oak library")is None;n+=4
    assert decide_id(["NONE","RQ-415","NONE"])==("RQ-415",["RQ-415"]) and decide_id(["RQ-415","RQ-432"])[0]is None and decide_id(["NONE"])==(None,[]);n+=3
    assert decide_place(["NONE","pine archive"])[0]=="pine archive" and decide_place(["pine archive","elm lodge"])[0]is None;n+=2
    assert len({*OBJECTS})==16 and len({*IDS})==16 and len({*PLACES})==16 and len(NEUTRAL)==32 and len(ABSIDS)==8 and not set(ABSIDS)&set(IDS);n+=6
    recs=[src(OBJECTS[i],IDS[i],PLACES[i])for i in range(16)]
    for i in range(16):
        for p in(q1(OBJECTS[i]),q2(IDS[i])):assert not any(r in p for r in recs)and"MEMORY RECORD"not in p and"Location:"not in p;n+=1
    assert not any(s in r for s in NEUTRAL for r in recs);n+=1
    def synth(bad=False):
        raw=dict(mw1=[],mw2=[],missing=[],absent=[],nomem=[])
        for i in range(16):
            raw["mw1"].append(dict(q=i,reads=[dict(raw=IDS[i]if j==i else"NONE\nx")for j in range(16)]))
            raw["mw2"].append(dict(q=i,id=IDS[i],reads=[dict(raw=("junk"if bad and i==3 else PLACES[i])if j==i else"NONE")for j in range(16)]))
        for i in range(8):
            raw["missing"].append(dict(i=i,query=MISSOBJ[i],reads=[dict(raw="NONE")for _ in range(16)]))
            raw["absent"].append(dict(i=i,query=ABSIDS[i],reads=[dict(raw="NONE")for _ in range(16)]))
            raw["nomem"].append(dict(i=i,id=IDS[i],raw="NONE"))
        return raw
    R=derive(synth());assert R["counts"]=={"mw1":16,"mw2":16,"linked":16,"missing":8,"absent_id":8,"nomem":8}and R["verdict"]=="PASS_MISTRAL_FINAL_ENGINE"and R["replay_match"] and R["mw1"]["wrong_none"]==240 and R["reads_per_query"]==[16];n+=4
    R=derive(synth(True));assert R["counts"]["mw2"]==15 and R["counts"]["linked"]==15 and R["verdict"]=="PASS_MISTRAL_FINAL_ENGINE"and not R["replay_match"];n+=3
    assert stale_scan("t","Mistral-7B-Instruct-v0.3 32L TEST474 TEST467 TEST473 mw1 16/16 C07 RQ-415 hello")==[];n+=1
    for bad in("Qwen2.5","28 layers","hidden 3584","K120 / V128","family F2","TEST461","TEST460","A100 GPU","IBR rows","4 KV heads","zenodo"):assert stale_scan("t",bad),bad;n+=1
    assert stale_scan("t","GPU NVIDIA A100-SXM4-40GB",allow=("NVIDIA A100-SXM4-40GB",))==[];n+=1
    return n
#<<CONST_END>>
#<<ENGINE_BEGIN>>
for _m,_p in[("torch","torch"),("transformers","transformers"),("accelerate","accelerate"),("gradio","gradio"),("matplotlib","matplotlib"),("PIL","pillow")]:
    if importlib.util.find_spec(_m)is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",_p])
import numpy as np,torch,transformers,gradio as gr,matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch,FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from PIL import Image
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache,StoppingCriteria,StoppingCriteriaList
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required (Runtime → Change runtime type → GPU).")
torch.set_grad_enabled(False)
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
def _cell_source():
    try:
        s_=get_ipython().user_ns.get("_ih",[""])[-1]
        return s_ if isinstance(s_,str)and"AKBASCORE NIRVANA"in s_ else None
    except Exception:return None
CELL_SOURCE=_cell_source();CELL_SOURCE_SHA=hashlib.sha256(CELL_SOURCE.encode("utf-8")).hexdigest()if CELL_SOURCE else None
DEV="cuda";STARTUP_UTC=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
ROOT=Path("/content/AKBASCORE_MISTRAL_CARTRIDGE")if os.path.isdir("/content")else Path("/tmp/AKBASCORE_MISTRAL_CARTRIDGE")
ROOT.mkdir(parents=True,exist_ok=True);ALLOWED_PATHS=[str(ROOT)]
def utc_now():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def local_now():return datetime.now().astimezone().isoformat(timespec="milliseconds")
QUIET=False
def say(*a):
    if not QUIET:print(*a,flush=True)
say("="*110);say("AKBASCORE NIRVANA · MISTRAL — COGNITIVE CARTRIDGE FINAL DEMO");say("="*110)
say(f"[0/7] CPU self-test PASS ({cpu_selftest()} checks) · demo panel lock SHA-256 {LOCK_SHA}")
say("START UTC :",STARTUP_UTC);say("Model :",MODEL_ID);say("GPU :",torch.cuda.get_device_name(0))
say("Gradio :",gr.__version__,"| Transformers:",transformers.__version__,"| Torch:",torch.__version__)
say("[1/7] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit())or 0)for x in transformers.__version__.split(".")[:2]);dt="dtype"if tv>=(4,56)else"torch_dtype"
torch.cuda.synchronize();_t=time.perf_counter()
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{dt:torch.bfloat16}).eval()
for p in model.parameters():p.requires_grad_(False)
torch.cuda.synchronize();MODEL_LOAD_S=time.perf_counter()-_t
cfg=model.config;layers=model.model.layers;NL=len(layers);H=cfg.hidden_size;NH=cfg.num_attention_heads;NKV=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None)or H//NH;KVD=NKV*HD
if(NL,H,NH,NKV,HD)!=ARCH:raise RuntimeError(f"Architecture mismatch {(NL,H,NH,NKV,HD)}")
if next(model.parameters()).dtype!=torch.bfloat16:raise RuntimeError("Expected bfloat16 weights")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
ge=model.generation_config.eos_token_id;EOS=sorted({tok.eos_token_id,*(ge if isinstance(ge,(list,tuple))else[ge])}-{None})
FP=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[13].mlp.down_proj.weight,layers[23].self_attn.o_proj.weight,layers[26].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
FP_NAMES=["layers.0.self_attn.q_proj","layers.8.self_attn.o_proj","layers.13.mlp.down_proj","layers.23.self_attn.o_proj","layers.26.mlp.down_proj","model.norm","lm_head"]
# ---------------- TEST474 engine functions (verbatim) ----------------
def clean():gc.collect();torch.cuda.empty_cache()
def sentinel():
    h=hashlib.sha256()
    for p in FP:
        a=p.detach().reshape(-1);n=a.numel()
        for j in range(16):
            o=j*(n-256)//15;h.update(a[o:o+256].float().cpu().numpy().tobytes())
    return h.hexdigest()
def ids(s):return tok(s,add_special_tokens=False).input_ids
def new_cache():
    try:return DynamicCache(config=cfg)
    except:return DynamicCache()
def cache_of(K,V):
    c=new_cache()
    for L in range(NL):c.update(K[L].clone(),V[L].clone(),L)
    return c
@torch.inference_mode()
def forge(s):
    seq=[PAD]+ids(s+SEP);o=model(input_ids=torch.tensor([seq],device=DEV),use_cache=False,output_hidden_states=True,return_dict=True);K=[];V=[]
    for L in range(NL):
        z=layers[L].input_layernorm(o.hidden_states[L][0]);a=layers[L].self_attn
        K.append(a.k_proj(z).contiguous());V.append(a.v_proj(z).contiguous())
    return{"T":len(seq),"K":K,"V":V}
@torch.inference_mode()
def install(K,V):
    T=K[0].shape[0];pos=torch.arange(T,device=DEV)[None];cos,sin=model.model.rotary_emb(K[0][None],pos);KK=[];VV=[]
    for L in range(NL):
        k=K[L].reshape(1,T,NKV,HD).transpose(1,2).contiguous();v=V[L].reshape(1,T,NKV,HD).transpose(1,2).contiguous()
        kr,_=apply_rotary_pos_emb(k,k,cos,sin,unsqueeze_dim=1);KK.append(kr.contiguous());VV.append(v)
    return tuple(KK),tuple(VV),T
CB=[]
@torch.inference_mode()
def build_codebook():
    global CB
    t0=time.perf_counter();CO=[forge(s)for s in NEUTRAL];CB=[]
    for L in range(NL):
        e={}
        for n in("K","V"):
            R=torch.cat([c[n][L][1:]for c in CO]).float().reshape(-1,NKV,HD);mu=[];basis=[]
            for h in range(NKV):
                X=R[:,h];m=X.mean(0);_,_,vh=torch.linalg.svd(X-m,full_matrices=False);b=vh[:DMAX].T.contiguous()
                if b.shape[1]<DMAX:b=F.pad(b,(0,DMAX-b.shape[1]))
                mu.append(m);basis.append(b)
            e[n]=(torch.stack(mu),torch.stack(basis))
        CB.append(e)
    del CO;clean();torch.cuda.synchronize()
    return dict(sentences=len(NEUTRAL),svds=NL*2*NKV,seconds=time.perf_counter()-t0,bytes=sum(t.numel()*t.element_size()for e in CB for n in("K","V")for t in e[n]))
@torch.inference_mode()
def compress(f):
    out={}
    for n in("K","V"):
        rows=[]
        for L in range(NL):
            mu,B=CB[L][n];x=f[n][L][1:].float().reshape(-1,NKV,HD);c=torch.einsum("thi,hid->thd",x-mu,B[:,:,:D]).to(torch.bfloat16)
            content=(mu+torch.einsum("thd,hid->thi",c.float(),B[:,:,:D])).reshape(-1,KVD).to(torch.bfloat16)
            rows.append(torch.cat([f[n][L][:1],content],0))
        out[n]=rows
    return{"K":out["K"],"V":out["V"],"T":f["T"]}
@torch.inference_mode()
def gen(card,prompt):
    K,V,T=install(card["K"],card["V"]);qi=ids(prompt);x=torch.tensor([[PAD]*T+qi],device=DEV);c=cache_of(K,V)
    y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),past_key_values=c,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=PAD,eos_token_id=EOS)
    return tok.decode(y[0,x.shape[1]:],skip_special_tokens=True).strip()
@torch.inference_mode()
def gen_nomem(prompt):
    qi=ids(prompt);x=torch.tensor([qi],device=DEV)
    y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=PAD,eos_token_id=EOS)
    return tok.decode(y[0,x.shape[1]:],skip_special_tokens=True).strip()
class FirstLineStop(StoppingCriteria):
    """Stops once the generated text holds a complete first answer line (non-space text followed by a newline)."""
    def __init__(self,n0):self.n0=n0
    def __call__(self,input_ids,scores,**kw):
        t=tok.decode(input_ids[0,self.n0:],skip_special_tokens=True)
        return torch.tensor([bool(re.search(r"\S[ \t]*\n",t))],device=input_ids.device)
@torch.inference_mode()
def gen_fast(card,prompt):
    K,V,T=install(card["K"],card["V"]);qi=ids(prompt);x=torch.tensor([[PAD]*T+qi],device=DEV);c=cache_of(K,V)
    y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),past_key_values=c,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=PAD,eos_token_id=EOS,stopping_criteria=StoppingCriteriaList([FirstLineStop(x.shape[1])]))
    return tok.decode(y[0,x.shape[1]:],skip_special_tokens=True).strip()
@torch.inference_mode()
def gen_nomem_fast(prompt):
    qi=ids(prompt);x=torch.tensor([qi],device=DEV)
    y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=PAD,eos_token_id=EOS,stopping_criteria=StoppingCriteriaList([FirstLineStop(x.shape[1])]))
    return tok.decode(y[0,x.shape[1]:],skip_special_tokens=True).strip()
# ---------------- demo-side read-only measurements (never alter cartridges) ----------------
def hook_module(fn):
    f=getattr(fn,"func",fn);return str(getattr(f,"__module__",None)or type(f).__module__ or"")
def hook_stats():
    tot=0;foreign=Counter()
    for m in model.modules():
        for d in(m._forward_hooks,m._forward_pre_hooks,m._backward_hooks):
            for fn in d.values():
                tot+=1;mod=hook_module(fn)
                if not mod.startswith(("transformers","accelerate","torch")):foreign[mod]+=1
    return tot,dict(foreign)
def hook_count():return hook_stats()[0]
def lora_present():return hasattr(model,"peft_config")or any("lora"in n.lower()for n,_ in model.named_modules())
def trainable_tensors():return sum(int(p.requires_grad)for p in model.parameters())
def optimizer_present():return any(isinstance(v,torch.optim.Optimizer)for v in list(globals().values()))
def fingerprint():return tuple(float(t.sum(dtype=torch.float32))for t in FP)
@torch.inference_mode()
def card_telemetry(f,c,excerpt):
    cosK=[];cosV=[]
    for L in range(NL):
        cosK.append(float(F.cosine_similarity(c["K"][L][1:].float().reshape(-1),f["K"][L][1:].float().reshape(-1),0)))
        cosV.append(float(F.cosine_similarity(c["V"][L][1:].float().reshape(-1),f["V"][L][1:].float().reshape(-1),0)))
    own=all(torch.equal(c[n][L][:1],f[n][L][:1])for n in("K","V")for L in range(NL))
    tel=dict(cosK=cosK,cosV=cosV,own_exact=bool(own))
    if excerpt:
        mu,B=CB[16]["K"];x=f["K"][16][1:].float().reshape(-1,NKV,HD);cc=torch.einsum("thi,hid->thd",x-mu,B[:,:,:D]).to(torch.bfloat16)
        tel["excerpt"]=dict(layer=16,head=0,tensor="K",values=[[round(v,4)for v in row]for row in cc[:,0,:].float().cpu().tolist()])
    return tel
class Engine:
    def __init__(self):
        self.counters=Counter()
        self.info=dict(model_id=MODEL_ID,arch=(NL,H,NH,NKV,HD),dtype="bfloat16",attn=getattr(cfg,"_attn_implementation",None),gpu=torch.cuda.get_device_name(0),
            gpu_total_gib=torch.cuda.get_device_properties(0).total_memory/2**30,torch=torch.__version__,transformers=transformers.__version__,gradio=gr.__version__,
            python=sys.version.split()[0],platform=platform.platform(),params=sum(p.numel()for p in model.parameters()),pad_id=PAD,eos_ids=EOS,
            sliding_window=getattr(cfg,"sliding_window",None),fp_names=FP_NAMES,model_load_seconds=MODEL_LOAD_S,startup_utc=STARTUP_UTC,
            sentinel0=sentinel(),fingerprint0=list(fingerprint()),hooks0=hook_count(),cell_source_sha256=CELL_SOURCE_SHA,fast_first_line=bool(FAST_FIRST_LINE),sentinel_method="sampled SHA-256: 7 tensors × 16 slices × 256 values (TEST474 method)")
        if sentinel()!=self.info["sentinel0"]:raise RuntimeError("Sentinel is not repeatable")
    def frozen_state(self):
        tot,fo=hook_stats();return dict(training=bool(model.training),trainable_tensors=trainable_tensors(),lora=lora_present(),optimizer=optimizer_present(),hooks=tot,foreign_hooks=sum(fo.values()),foreign_modules=fo,sentinel=sentinel(),fingerprint=list(fingerprint()))
    def build_codebook(self):self.counters["forge_passes"]+=len(NEUTRAL);return build_codebook()
    def codebook_bytes(self):return sum(t.numel()*t.element_size()for e in CB for n in("K","V")for t in e[n])
    def forge_card(self,i):
        torch.cuda.synchronize();t0=time.perf_counter();f=forge(src(OBJECTS[i],IDS[i],PLACES[i]));c=compress(f);torch.cuda.synchronize();dt_=time.perf_counter()-t0
        self.counters["forge_passes"]+=1;tel=card_telemetry(f,c,i==0);T=c["T"]
        tel.update(T=T,encode_seconds=dt_,code_numbers=2*NL*(T-1)*NKV*D,own_numbers=2*NL*KVD,native_numbers=2*NL*KVD*T,
                   codec_macs=2*2*NL*(T-1)*NKV*HD*D,gpu_mib=torch.cuda.memory_allocated()/2**20)
        del f;clean();return c,tel
    def card_ptrs(self,c):return[c["K"][0].data_ptr(),c["V"][0].data_ptr()]
    def card_finite(self,c):return all(bool(torch.isfinite(t).all())for n in("K","V")for t in c[n])
    def q_tokens(self,p):return len(ids(p))
    def gen(self,c,p):self.counters["generate_calls"]+=1;return gen_fast(c,p)if FAST_FIRST_LINE else gen(c,p)
    def gen_nomem(self,p):self.counters["generate_calls"]+=1;return gen_nomem_fast(p)if FAST_FIRST_LINE else gen_nomem(p)
    def gen_full(self,c,p):self.counters["probe_full_decodes"]+=1;return gen(c,p)
    def gen_nomem_full(self,p):self.counters["probe_full_decodes"]+=1;return gen_nomem(p)
    def sync(self):torch.cuda.synchronize()
    def reset_peak(self):torch.cuda.reset_peak_memory_stats()
    def peak_gib(self):return torch.cuda.max_memory_allocated()/2**30
ENGINE=Engine()
say(f"Model loaded in {MODEL_LOAD_S:.2f}s | layers={NL} hidden={H} Q={NH} KV={NKV} head={HD} | params={ENGINE.info['params']:,} trainable={trainable_tensors()}")
say("[2/7] WEIGHT SENTINELS");say("Fingerprint:",[f"{x:.4f}"for x in ENGINE.info["fingerprint0"]]);say("SHA-256 sampled sentinel:",ENGINE.info["sentinel0"])
say("[3/7] ENGINE LOCK")
ENGINE_LOCK=[("MODEL_ID = mistralai/Mistral-7B-Instruct-v0.3",MODEL_ID=="mistralai/Mistral-7B-Instruct-v0.3"),("architecture 32 layers / hidden 4096 / 32 Q / 8 KV / head 128",(NL,H,NH,NKV,HD)==(32,4096,32,8,128)),
 ("K = D120 and V = D120 (DMAX 128)",(D,DMAX)==(120,128)),("fixed neutral codebook = 32 sentences",len(NEUTRAL)==32),("16 cartridges, 16 locked questions, 8 missing, 8 absent IDs",len(OBJECTS)==16 and len(MISSOBJ)==8 and len(ABSIDS)==8),
 ("S1_RECORD source frame and C_VERIFY readout frames",src("o","c","p")=="MEMORY RECORD\nObject: o\nContainer: c\nLocation: p" and q1("o").startswith('QUESTION:\nDoes this memory explicitly contain the object')),
 ("greedy decoding, max_new_tokens = 32",MAX_NEW==32),("BF16 weights",next(model.parameters()).dtype==torch.bfloat16),("SDPA attention",getattr(cfg,"_attn_implementation",None)=="sdpa"),
 ("no forward/backward hooks from this cell (foreign hooks = 0)",not hook_stats()[1]),("frozen eval model",(not model.training)and trainable_tensors()==0 and not lora_present()),("no optimizer",not optimizer_present()),
 ("sampled weight sentinel repeatable",sentinel()==ENGINE.info["sentinel0"])]
_bad=[n for n,ok in ENGINE_LOCK if not ok]
if _bad:raise RuntimeError(f"ENGINE LOCK FAILED: {_bad}")
say("Engine lock: PASS ·",len(ENGINE_LOCK),"checks")
#<<ENGINE_END>>
#<<CORE_BEGIN>>
N_STAGES=8;TOTAL_READS=16*16+16*16+8*16+8*16+8
def ev(stage,title,body,done=None,total=None,eta=None):return dict(stage=stage,title=title,body=body,done=done,total=total,eta=eta)
def file_ready(p):return p is not None and Path(p).is_file()and Path(p).stat().st_size>0
def prune_runs(keep=2):
    runs=sorted([p for p in ROOT.glob("MISTRAL-CC-*")if p.is_dir()],key=lambda p:p.stat().st_mtime)
    for p in runs[:max(0,len(runs)-keep)]:shutil.rmtree(p,ignore_errors=True)
def post_seal_audit(imgs,zp,member_names,pp,sha,tp,mp,verdict):
    checks=[]
    def ok(name,cond):
        checks.append(name)
        if not cond:raise AuditFail("POST-SEAL AUDIT FAILED: "+name)
    ok(f"{N_POSTERS}/{N_POSTERS} posters created",len(imgs)==N_POSTERS and all(file_ready(p)for p,_ in imgs))
    bad=[]
    for p,_ in imgs:
        with Image.open(p)as im:
            if not(im.format=="JPEG"and im.mode=="RGB"):bad.append(p.name)
    ok(f"{N_POSTERS}/{N_POSTERS} posters are JPEG/RGB",not bad)
    with zipfile.ZipFile(zp)as z:
        ok("ZIP testzip()",z.testzip()is None);nm=z.namelist()
        ok("ZIP is flat (no sub-folders)",all("/"not in n for n in nm));ok("ZIP contents = expected package",sorted(nm)==sorted(member_names))
        jp=sorted(n for n in nm if n.lower().endswith(".jpg"));ok("poster numbering 01..%02d"%N_POSTERS,[n[:2]for n in jp]==[f"{i:02d}"for i in range(1,N_POSTERS+1)])
    ok("payload SHA-256 recomputation",hashlib.sha256(pp.read_bytes()).hexdigest()==sha);ok("payload JSON parses",isinstance(json.loads(pp.read_bytes().decode("utf-8")),dict))
    m=json.loads(mp.read_text(encoding="utf-8"));ok("manifest hash and verdict match payload",m["payload_sha256"]==sha and m["verdict"]==verdict)
    ok("readable log exists and is non-empty",file_ready(tp));ok("ZIP exists and is non-empty",file_ready(zp))
    return checks
def make_txt(P,R,sha,names,sealed_utc):
    o=[];a=o.append;S="="*110;Dd="-"*110;E_=P["environment"];fz=P["frozen"]
    a(S);a("AKBASCORE NIRVANA · MISTRAL · COGNITIVE CARTRIDGE — READABLE RUN LOG");a(S)
    a(f"Derived from {names['payload']} (SHA-256 {sha}). Manifest: {names['manifest']}.");a("The SHA-256 is an artifact integrity seal, not a scientific proof.")
    for k_,v in(("RUN ID",P["run_id"]),("START UTC",P["run_start_utc"]),("END UTC",P["run_end_utc"]),("SEALED UTC",sealed_utc),("MODEL",MODEL_ID),("ARCHITECTURE","32 layers / hidden 4096 / 32 Q heads / 8 KV heads / head 128"),
               ("DTYPE / ATTENTION",f"{P['model']['dtype']} / {P['model']['attn']}"),("GPU",E_["gpu"]),("TORCH",E_["torch"]),("TRANSFORMERS",E_["transformers"]),("SEED",SEED),
               ("ENGINE","TEST474 · S1_RECORD + C_VERIFY · K=D120 V=D120 · OWN preserved · module-wise unique aggregation · greedy"),("PANEL LOCK SHA-256",LOCK_SHA)):a(f"{k_:<22}: {v}")
    a("");a("SCOPE");a(Dd);a(P["scope"]["replay"]);a(P["scope"]["frame_selection"])
    a("");a("EXPERIMENTER LEDGER (forge-time source records; NOT available to the readout)");a(Dd)
    for L in P["experimenter_ledger_not_available_to_readout"]:a(f"{L['cid']} | {L['object']} | {L['id']} | {L['location']}")
    a("");a("CARTRIDGES (numerical K/V)");a(Dd)
    for c in P["cartridges"]:a(f"{c['cid']} slots={c['T']} code_numbers={c['code_numbers']:,} own={c['own_numbers']:,} native={c['native_numbers']:,} cosK_min={min(c['cosK']):.4f} cosV_min={min(c['cosV']):.4f} forge+compress={c['encode_seconds']:.2f}s")
    a("");a("LIVE RESULTS (this run, re-derived from raw text)");a(Dd)
    for k_ in R["counts"]:a(f"{k_:<10}: {R['counts'][k_]}/{R['totals'][k_]}")
    a(f"wrong-cartridge NONE stage 1: {R['mw1']['wrong_none']}/{R['mw1']['wrong_total']} | stage 2: {R['mw2']['wrong_none']}/{R['mw2']['wrong_total']}")
    for g_,v in R["gates"].items():a(f"gate {g_:<10} (>= {GATE_MIN[g_]}): {'PASS'if v else'FAIL'}")
    a(f"VERDICT: {R['verdict']} | replay identical to sealed counts: {R['replay_match']}")
    a("");a("PER-QUERY DECISIONS");a(Dd)
    for i in range(16):a(f"Q{i+1:02d} {OBJECTS[i]:<18} MW1 decided={R['mw1']['decided'][i]} expected={IDS[i]} | MW2 decided={R['mw2']['decided'][i]} expected={PLACES[i]}")
    for key,lab in(("missing","MISSING"),("absent","ABSENT")):
        for m in R[key]:a(f"{lab} {m['i']+1}: '{m['query']}' NONE={m['none']}/{m['n']} unique={m['uniq']} -> {'PASS'if m['ok']else'FAIL'}")
    for m in R["nomem"]:a(f"NOMEM {m['i']+1}: target={m['target']} first line={m['first'][:60]!r} parsed={m['parsed']} -> {'PASS'if m['ok']else'FAIL'}")
    a("");a("FROZEN MODEL");a(Dd)
    for k_ in("sentinel_startup","sentinel_pre_run","sentinel_after"):a(f"{k_:<18}: {fz[k_]}")
    a(f"trainable tensors={fz['trainable_tensors']} lora={fz['lora']} optimizer={fz['optimizer']} training_mode={fz['training_mode']} hooks {fz['hooks_before']}→{fz['hooks_after']} (foreign: {fz['foreign_hooks_before']}→{fz['foreign_hooks_after']})")
    a(fz["sentinel_method"]);a("");a("PRE-SEAL CHECKS");a(Dd)
    for c in P["checks_pre_seal"]:a("PASS · "+c)
    a("");a("SEALED HISTORICAL RECORD (NOT produced by this run)");a(Dd);a(json.dumps(P["sealed_record"],ensure_ascii=False))
    a("");a("TIMING (seconds)");a(Dd)
    for k_,v in P["timing"].items():a(f"{k_:<20}: {v:.2f}")
    a(f"readout reads: {P['counters']['generate_calls']} | forge passes: {P['counters']['forge_passes']} | peak GPU memory: {P['gpu']['peak_allocated_gib']:.2f} GiB");a(f"readout generation: {P['engine']['readout_generation']} | probe: {P['engine']['early_stop_probe']}")
    a("");a("Raw generated text for every read is in "+names["payload"]+" and "+names["jsonl"]+".");a(S)
    return "\n".join(o)
def strip_model(text,mtexts):
    """Remove raw model-generated strings before the stale-reference scan: the scan targets our labels, not the model's own words."""
    for s in sorted(mtexts,key=len,reverse=True):
        if len(s)>=2:
            for v in(s,json.dumps(s,ensure_ascii=False)[1:-1]):text=text.replace(v,"")
    return text
def execute_run(E,ctl):
    """Fail-closed wrapper: on a technical audit failure the raw outputs gathered so far are preserved (never sealed, no posters)."""
    state={}
    try:return(yield from _execute_run(E,ctl,state))
    except AuditFail as ex:
        raw=state.get("raw")
        if raw and any(raw.values()):
            try:
                p=Path(ctl["run_dir"])/f"FAILED_AUDIT_RAW_{ctl['run_id']}.json"
                p.write_bytes(canon(dict(status="FAILED AUDIT — NOT SEALED",run_id=ctl["run_id"],failed_check=str(ex),checks_passed=state.get("checks",[]),raw=raw,note="Raw outputs preserved so the run is not lost. No posters, no seal and no verdict were produced.")));ex.partial=str(p)
            except Exception:pass
        raise
# ===== END PART 1 / 3 — CONTINUE WITH PART 2 =====
def _execute_run(E,ctl,state):
    """Generator. Yields progress events, returns the result bundle. Any failed technical audit raises AuditFail (no package is sealed)."""
    run_id=ctl["run_id"];run_dir=Path(ctl["run_dir"]);I=E.info;checks=[];tm=Counter();T0=time.perf_counter();state["checks"]=checks
    def chk(name,cond):
        checks.append(name)
        if not cond:raise AuditFail(name)
    run_start_utc=utc_now();run_start_local=local_now()
    say("="*110);say(f"RUN {run_id}");say("="*110)
    # ---- 1/8 integrity ----
    say("[1/8] INTEGRITY · FROZEN-MODEL PRE-CHECK");yield ev(1,"Integrity check","Frozen model: hooks, trainable tensors, LoRA, optimizer, sampled weight sentinel.")
    fs0=E.frozen_state()
    chk("model = Mistral-7B-Instruct-v0.3",I["model_id"]==MODEL_ID);chk("architecture 32L / H4096 / 32Q / 8KV / HD128",tuple(I["arch"])==ARCH)
    chk("dtype bfloat16",I["dtype"]=="bfloat16");chk("attention SDPA",I["attn"]=="sdpa");chk("K code dimension = D120 and V code dimension = D120",D==120 and DMAX==128)
    chk("model in eval mode",not fs0["training"]);chk("trainable parameter tensors = 0",fs0["trainable_tensors"]==0);chk("no LoRA / PEFT adapter",not fs0["lora"]);chk("no optimizer object",not fs0["optimizer"])
    chk("pre-run weight sentinel = startup sentinel",fs0["sentinel"]==I["sentinel0"]);chk("pre-run weight fingerprint = startup fingerprint",fs0["fingerprint"]==I["fingerprint0"])
    if fs0["foreign_hooks"]:raise AuditFail("foreign forward/backward hooks present before run: "+str(fs0["foreign_modules"]))
    chk("no foreign forward/backward hooks before run",fs0["foreign_hooks"]==0)
    for c in checks:say("   PASS ·",c)
    # ---- 2/8 codebook ----
    say("[2/8] FIXED NEUTRAL CODEBOOK · D120");yield ev(2,"Fixed neutral codebook","32 neutral sentences → PCA basis per layer × KV head × K/V. The codebook contains none of the cartridge facts.")
    E.reset_peak();cbi=E.build_codebook();cbi["bytes"]=E.codebook_bytes();tm["codebook"]=cbi["seconds"]
    chk("codebook = 32 neutral sentences",cbi["sentences"]==32);chk("codebook SVD count = layers × 2 × KV heads",cbi["svds"]==I["arch"][0]*2*I["arch"][3])
    say(f"   {cbi['sentences']} sentences · {cbi['svds']} SVDs · {cbi['seconds']:.1f}s · {cbi['bytes']/2**20:.1f} MiB")
    # ---- 3/8 forge ----
    say("[3/8] FORGE · 16 LOCKED S1_RECORD CARTRIDGES");cards=[];tels=[];t0=time.perf_counter()
    for i in range(16):
        yield ev(3,f"Forging cartridge {i+1:02d}/16","Source record → model's own K/V → D120 projection. Each cartridge is forged alone.",done=i,total=16)
        c,tel=E.forge_card(i);cards.append(c);tels.append(tel)
        say(f" C{i+1:02d}: T={tel['T']:02d} | {OBJECTS[i]} → {IDS[i]} → {PLACES[i]} | cosK≥{min(tel['cosK']):.4f} cosV≥{min(tel['cosV']):.4f}")
    tm["forge"]=time.perf_counter()-t0
    chk("16 cartridges forged",len(cards)==16);chk("16 independent cartridges (distinct tensor storage)",len({p for c in cards for p in E.card_ptrs(c)})==32)
    chk("cartridge tensors finite",all(E.card_finite(c)for c in cards));chk("OWN first-token K/V identical to the forged source state in every layer",all(t["own_exact"]for t in tels))
    fs_m=E.frozen_state()
    if fs_m["foreign_hooks"]:raise AuditFail("foreign forward/backward hooks present after forging: "+str(fs_m["foreign_modules"]))
    chk("weight sentinel and fingerprint unchanged after forging (checked before any readout)",fs_m["sentinel"]==I["sentinel0"]and fs_m["fingerprint"]==I["fingerprint0"]);chk("no foreign hooks and no trainable tensors after forging",fs_m["foreign_hooks"]==0 and fs_m["trainable_tensors"]==0)
    # ---- 4/8 source removed ----
    say("[4/8] SOURCE REMOVED · READOUT AUDIT ARMED");yield ev(4,"Source removed","From here the readout receives only a question and the numerical cartridges. Every prompt is checked against all 16 source records.",done=16,total=16)
    sources=[src(OBJECTS[i],IDS[i],PLACES[i])for i in range(16)];audit=Counter();qtoks=[]
    def check_prompt(prompt):
        if any(s in prompt for s in sources)or"MEMORY RECORD"in prompt or"Location:"in prompt:raise AuditFail("SOURCE TEXT FOUND IN A READOUT PROMPT")
    def guard(prompt):
        check_prompt(prompt);audit["prompts_checked"]+=1;qtoks.append(E.q_tokens(prompt))
    def read(phase,card,prompt):
        guard(prompt);E.sync();t=time.perf_counter();out=E.gen(card,prompt);E.sync();dt_=time.perf_counter()-t;tm[phase]+=dt_;audit["calls_"+phase]+=1;return dict(raw=out,secs=round(dt_,4))
    probe=dict(pairs=0,identical=0)
    if I.get("fast_first_line"):
        say("   early-stop probe: first answer line of fast decoding vs full 32-token decoding")
        for pr in(q1(OBJECTS[0]),q2(IDS[0]),q1(MISSOBJ[0])):
            for c in cards[:4]:
                check_prompt(pr);a_=E.gen(c,pr);b_=E.gen_full(c,pr);probe["pairs"]+=1;probe["identical"]+=int(first(a_)==first(b_))
        check_prompt(q2(IDS[0]));a_=E.gen_nomem(q2(IDS[0]));b_=E.gen_nomem_full(q2(IDS[0]));probe["pairs"]+=1;probe["identical"]+=int(first(a_)==first(b_))
        say(f"   probe: {probe['identical']}/{probe['pairs']} identical")
        chk(f"early-stop probe: first answer lines identical to full-length decoding ({probe['identical']}/{probe['pairs']})",probe["identical"]==probe["pairs"])
    raw=dict(mw1=[],mw2=[],missing=[],absent=[],nomem=[]);state["raw"]=raw;done=0;tr=time.perf_counter()
    def prog(stage,title,body):
        el=time.perf_counter()-tr;return ev(stage,title,body,done=done,total=TOTAL_READS,eta=(el/done*(TOTAL_READS-done))if done else None)
    # ---- 5/8 MW1 + MW2 ----
    say("[5/8] MW1 · OBJECT → UNIQUE ID");dec1=[]
    for i in range(16):
        p=q1(OBJECTS[i]);rs=[]
        for j,c in enumerate(cards):
            r=read("mw1",c,p);r["c"]=j+1;rs.append(r);done+=1
            if j%4==3:yield prog(5,f"MW1 · question {i+1:02d}/16 · object → container ID",f'"{OBJECTS[i]}" is asked to all 16 independent cartridges.')
        raw["mw1"].append(dict(q=i,object=OBJECTS[i],prompt=p,reads=rs));d,u=decide_id([r["raw"]for r in rs]);dec1.append(d)
        say(f" {i+1:02d}: {'PASS'if d==IDS[i]else'FAIL'} | expected={IDS[i]} | decided={d or'UNKNOWN'} | unique={len(u)}")
        if d!=IDS[i]:
            for j,r in enumerate(rs):
                if not is_none(r["raw"]):say(f"     C{j+1:02d}: {first(r['raw'])[:72]}")
    say(f"MW1={sum(int(dec1[i]==IDS[i])for i in range(16))}/16");say("[5/8] MW2 · DECIDED ID → UNIQUE LOCATION")
    for i in range(16):
        cid=dec1[i]
        if cid is None:raw["mw2"].append(dict(q=i,id=None,skipped=True));say(f" {i+1:02d}: FAIL | Stage 1 UNKNOWN");continue
        p=q2(cid);rs=[]
        for j,c in enumerate(cards):
            r=read("mw2",c,p);r["c"]=j+1;rs.append(r);done+=1
            if j%4==3:yield prog(5,f"MW2 · question {i+1:02d}/16 · container ID → location",f"{cid} is asked to all 16 independent cartridges.")
        raw["mw2"].append(dict(q=i,id=cid,prompt=p,reads=rs));d,u=decide_place([r["raw"]for r in rs])
        say(f" {i+1:02d}: {'PASS'if d==PLACES[i]else'FAIL'} | {cid} | expected={PLACES[i]} | decided={d or'UNKNOWN'} | unique={len(u)}")
        if d!=PLACES[i]:
            for j,r in enumerate(rs):
                if not is_none(r["raw"]):say(f"     C{j+1:02d}: {first(r['raw'])[:72]}")
    # ---- 6/8 missing / absent / nomem ----
    say("[6/8] MISSING OBJECT / ABSENT ID / NOMEM")
    for i in range(8):
        for key,phase,query,mk,title in(("missing","neg",MISSOBJ[i],q1,"missing object"),("absent","neg",ABSIDS[i],q2,"absent container ID")):
            p=mk(query);rs=[]
            for j,c in enumerate(cards):
                r=read(phase,c,p);r["c"]=j+1;rs.append(r);done+=1
                if j%4==3:yield prog(6,f"{title.capitalize()} · {i+1}/8",f'"{query}" is not stored in any cartridge.')
            raw[key].append(dict(i=i,query=query,prompt=p,reads=rs))
    for i in range(8):
        p=q2(IDS[i]);guard(p);E.sync();t=time.perf_counter();o=E.gen_nomem(p);E.sync();dt_=time.perf_counter()-t;tm["nomem"]+=dt_;done+=1
        raw["nomem"].append(dict(i=i,id=IDS[i],prompt=p,raw=o,secs=round(dt_,4)))
        if i%2==1:yield prog(6,f"NOMEM control · {i+1}/8","Same Stage-2 question with no cartridge installed.")
    # ---- 7/8 audit + seal ----
    say("[7/8] FINAL AUDIT · RESULT RE-DERIVATION · SEAL");yield ev(7,"Final audit and seal","Results are re-derived from the raw generated text; frozen-model checks are repeated.",done=TOTAL_READS,total=TOTAL_READS)
    R=derive(raw);fs1=E.frozen_state();tm["engine"]=time.perf_counter()-T0
    chk("weight sentinel after run = startup sentinel",fs1["sentinel"]==I["sentinel0"]);chk("weight fingerprint after run = startup fingerprint",fs1["fingerprint"]==I["fingerprint0"])
    chk("model eval mode and trainable tensors = 0 after run",(not fs1["training"])and fs1["trainable_tensors"]==0);chk("no LoRA / optimizer after run",(not fs1["lora"])and(not fs1["optimizer"]))
    (_ for _ in()).throw(AuditFail("foreign forward/backward hooks present after run: "+str(fs1["foreign_modules"])))if fs1["foreign_hooks"]else None;chk("no foreign forward/backward hooks after run (demo installs none)",fs1["foreign_hooks"]==0);chk("every query read by all 16 cartridges (no pre-selection)",R["reads_per_query"]==[16])
    k=R["counts"];tot=R["totals"]
    for n_,v in(("MW1",k["mw1"]),("MW2",k["mw2"]),("LINKED",k["linked"]),("MISSING",k["missing"]),("ABSENT-ID",k["absent_id"]),("NOMEM",k["nomem"])):say(f" {n_:<10}= {v}/{tot[n_.lower().replace('-','_')]}")
    say(f" wrong-cartridge NONE · stage 1 = {R['mw1']['wrong_none']}/{R['mw1']['wrong_total']} · stage 2 = {R['mw2']['wrong_none']}/{R['mw2']['wrong_total']}")
    for g_,v in R["gates"].items():say(f"   gate {g_:<10}: {'PASS'if v else'FAIL'}")
    say("VERDICT:",R["verdict"])
    chk("source records found in readout prompts = 0",audit["prompts_checked"]==sum(audit["calls_"+x]for x in("mw1","mw2","neg"))+8)
    chk("results re-derived from raw text are stable",derive(raw)==R)
    E.sync();peak=E.peak_gib()
    ledger=[dict(cid=f"C{i+1:02d}",object=OBJECTS[i],id=IDS[i],location=PLACES[i],source_text=sources[i])for i in range(16)]
    cart=[dict(cid=f"C{i+1:02d}",**{k_:v for k_,v in tels[i].items()})for i in range(16)]
    bank=dict(slots=sum(t["T"]for t in tels),code_numbers=sum(t["code_numbers"]for t in tels),own_numbers=sum(t["own_numbers"]for t in tels),native_numbers=sum(t["native_numbers"]for t in tels),
              codec_macs=sum(t["codec_macs"]for t in tels),codebook_bytes=cbi["bytes"]);bank["ratio_vs_native"]=(bank["code_numbers"]+bank["own_numbers"])/bank["native_numbers"]
    reads_total=sum(audit["calls_"+x]for x in("mw1","mw2","neg"))+8
    P={"schema":"akbascore.mistral.cartridge.run.v1","project":"AkbasCore NIRVANA","run_id":run_id,"run_start_utc":run_start_utc,"run_start_local":run_start_local,"run_end_utc":utc_now(),
       "model":{"id":MODEL_ID,"arch":list(ARCH),"dtype":I["dtype"],"attn":I["attn"],"params":I["params"],"sliding_window":I["sliding_window"],"pad_id":I["pad_id"],"eos_ids":I["eos_ids"]},
       "environment":{k_:I[k_]for k_ in("gpu","gpu_total_gib","torch","transformers","gradio","python","platform")},
       "engine":{"source":"TEST474 (474.MISTRAL.final.py) engine functions, unchanged","frame":"S1_RECORD source frame + C_VERIFY readout frames","K":D,"V":D,"own_first_token":"preserved","codebook":"fixed 32-sentence neutral corpus",
                 "readout":"module-wise, one generate call per cartridge, unique aggregation, greedy","max_new_tokens":MAX_NEW,"early_stop":bool(I.get("fast_first_line")),"early_stop_probe":probe,
                 "readout_generation":("stops at the end of the first answer line (parsed results identical by construction; probe reads re-decoded at full length and compared)"if I.get("fast_first_line")else"full TEST474 decoding, up to max_new_tokens"),"compute_path":"PyTorch CUDA backend; no custom CUDA/C++ kernel","lock_sha256":LOCK_SHA,"gate_thresholds":GATE_MIN},
       "codebook":cbi,"experimenter_ledger_not_available_to_readout":ledger,"cartridges":cart,"bank":bank,"raw":raw,"results":R,
       "source_removal":{"prompts_checked":audit["prompts_checked"],"source_record_hits":0,"question_tokens_min":min(qtoks),"question_tokens_max":max(qtoks),
                         "calls":{x:audit["calls_"+x]for x in("mw1","mw2","neg")}|{"nomem":8},"slots_per_cartridge":[t["T"]for t in tels],
                         "method":"each prompt string compared with all 16 source records and the record frame; readout input ids = [PAD]×T + question tokens"},
       "frozen":{"sentinel_startup":I["sentinel0"],"sentinel_pre_run":fs0["sentinel"],"sentinel_after":fs1["sentinel"],"fingerprint_startup":I["fingerprint0"],"fingerprint_pre_run":fs0["fingerprint"],"fingerprint_after":fs1["fingerprint"],
                 "tensors":I["fp_names"],"sentinel_method":I["sentinel_method"],"trainable_tensors":fs1["trainable_tensors"],"lora":fs1["lora"],"optimizer":fs1["optimizer"],"training_mode":fs1["training"],"hooks_before":I["hooks0"],"hooks_after":fs1["hooks"],"foreign_hooks_before":fs0["foreign_hooks"],"foreign_hooks_after":fs1["foreign_hooks"],"hook_note":"hook totals include hooks installed by transformers/accelerate; only hooks from other code count as foreign"},
       "counters":{"generate_calls":reads_total,"probe_full_decodes":E.counters["probe_full_decodes"],"forge_passes":E.counters["forge_passes"],"read_calls":{"mw1":audit["calls_mw1"],"mw2":audit["calls_mw2"],"negatives":audit["calls_neg"],"nomem":8}},
       "timing":{"model_load_seconds":I["model_load_seconds"],"codebook":tm["codebook"],"forge":tm["forge"],"mw1":tm["mw1"],"mw2":tm["mw2"],"negatives":tm["neg"],"nomem":tm["nomem"],"engine":tm["engine"]},
       "gpu":{"peak_allocated_gib":peak},"checks_pre_seal":list(checks),
       "sealed_record":{"note":"Historical logs. NOT produced by this run.","test474":SEALED474,"test473":SEALED473,"test467":SEALED467},
       "reproduction":{"seed":SEED,"cell_source_sha256":I.get("cell_source_sha256"),"note":"SHA-256 of the executed cell text; the text itself is not embedded because it contains the stale-reference deny-list."},
       "scope":{"replay":"SEALED TEST474 PANEL REPLAY — the same 16 records used in TEST473 frame selection and TEST474; not a new held-out validation",
                "frame_selection":"S1_RECORD + C_VERIFY was one of 4 conditions with GOLD 16/16 and DECOY_NONE 240/240 in TEST473; chosen by fixed ordering on the same 16 records"}}
    mtexts={r["raw"]for k_ in("mw1","mw2","missing","absent")for q in raw[k_]for r in q.get("reads",[])}|{q["raw"]for q in raw["nomem"]};mtexts|={first(t)for t in mtexts}
    P=jsafe(P);P["stale_scan"]={"patterns":len(STALE)+1,"hits":0,"scope":"payload JSON text, raw model-generated strings excluded"}
    hits=stale_scan("payload",strip_model(json.dumps(P,ensure_ascii=False),mtexts),allow=(I["gpu"],SEALED474["env"]))
    chk("stale-reference scan of the payload: 0 hits",not hits)
    pb=canon(P);sha=hashlib.sha256(pb).hexdigest();payload_name=f"{run_id}_FULL_RUN_LOG_payload.json";pp=run_dir/payload_name;t_seal=time.perf_counter();pp.write_bytes(pb);say("payload sealed:",sha)
    chk("payload SHA-256 verification after write",hashlib.sha256(pp.read_bytes()).hexdigest()==sha)
    names=dict(payload=payload_name,manifest=f"run_manifest_{run_id}.json",txt=f"{run_id}_READABLE_RUN_LOG.txt",summary=f"{run_id}_SUMMARY.json",jsonl=f"{run_id}_RESULTS.jsonl",images=f"images_manifest_{run_id}.json",zip=f"AKBASCORE_MISTRAL_CARTRIDGE_ALL_{run_id}.zip")
    mp=run_dir/names["manifest"];sealed_utc=utc_now()
    manifest={"demo":"AKBASCORE NIRVANA · MISTRAL · COGNITIVE CARTRIDGE","run_id":run_id,"payload_file":payload_name,"payload_sha256":sha,"payload_bytes":len(pb),"sealed_utc":sealed_utc,"verdict":R["verdict"],
              "canonicalization":"json.dumps(sort_keys=True, separators=(',',':'), ensure_ascii=False, allow_nan=False), UTF-8","note":"Artifact integrity seal: confirms the payload is unchanged since sealing. It is not a scientific proof."}
    mp.write_text(json.dumps(manifest,indent=2,sort_keys=True,ensure_ascii=False),encoding="utf-8")
    summary={"replay_of_test":474,"model":MODEL_ID,"D":D,"cards":16,"engine":"S1_RECORD + C_VERIFY / two-stage module-wise unique decision","results":{k_:[R["counts"][k_],R["totals"][k_]]for k_ in R["counts"]},
             "gates":R["gates"],"verdict":R["verdict"],"sentinel_before":I["sentinel0"],"sentinel_after":fs1["sentinel"],"run_id":run_id,"payload_sha256":sha}
    (run_dir/names["summary"]).write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    lines=[]
    for q in raw["mw1"]:
        i=q["q"];lines.append({"phase":"MW1","query":i,"object":OBJECTS[i],"expected":IDS[i],"decided":R["mw1"]["decided"][i],"raw":[r["raw"]for r in q["reads"]]})
    for q in raw["mw2"]:
        i=q["q"];lines.append({"phase":"MW2","query":i,"id":q.get("id"),"expected":PLACES[i],"decided":R["mw2"]["decided"][i],"raw":[r["raw"]for r in q.get("reads",[])]})
    for key in("missing","absent"):
        for q in raw[key]:lines.append({"phase":key.upper(),"i":q["i"],"query":q["query"],"raw":[r["raw"]for r in q["reads"]]})
    for q in raw["nomem"]:lines.append({"phase":"NOMEM","i":q["i"],"id":q["id"],"target":PLACES[q["i"]],"raw":q["raw"]})
    (run_dir/names["jsonl"]).write_text("\n".join(json.dumps(x,ensure_ascii=False)for x in lines)+"\n",encoding="utf-8")
    (run_dir/names["txt"]).write_text(make_txt(P,R,sha,names,sealed_utc),encoding="utf-8");seal_s=time.perf_counter()-t_seal
    # ---- 8/8 posters + ZIP ----
    say("[8/8] POSTERS · IMAGE MANIFEST · ZIP");yield ev(8,"Rendering posters","Every number on every poster is checked against the re-derived runtime result before the package is sealed.")
    ctx={"P":P,"R":R,"run_id":run_id,"sha":sha,"payload_name":payload_name,"seal_seconds":seal_s,"allow":(I["gpu"],SEALED474["env"]),"N":N_POSTERS,"poster_audit":[]}
    t_r=time.perf_counter();imgs=render_all(ctx,run_dir);render_s=time.perf_counter()-t_r
    entries=[]
    for n_,(p,cap)in enumerate(imgs,1):
        with Image.open(p)as im:w_,h_=im.size;fm_=im.format;md=im.mode
        entries.append({"index":n_,"filename":p.name,"title":cap,"width":w_,"height":h_,"format":fm_,"mode":md,"size_bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
    imf=run_dir/names["images"]
    imf.write_text(json.dumps({"run_id":run_id,"payload_sha256":sha,"verdict":R["verdict"],"count":len(entries),"render_seconds":render_s,"images":entries,"poster_text_audit":ctx["poster_audit"],
                               "note":"Per-image SHA-256 is an artifact integrity seal. Poster text was checked against re-derived results and scanned for stale references before saving."},indent=2,ensure_ascii=False),encoding="utf-8")
    zp=run_dir/names["zip"];members=[p for p,_ in imgs]+[imf,mp,pp,run_dir/names["txt"],run_dir/names["summary"],run_dir/names["jsonl"]]
    for m in members:
        h=stale_scan("filename",m.name)+(stale_scan(m.name,strip_model(m.read_text(encoding="utf-8"),mtexts),allow=ctx["allow"])if m.suffix in(".json",".txt",".jsonl")else[])
        chk(f"stale-reference scan of {m.name}: 0 hits",not h)
    with zipfile.ZipFile(zp,"w",compression=zipfile.ZIP_DEFLATED)as z:
        for m in members:z.write(m,arcname=m.name)
    post=post_seal_audit(imgs,zp,[m.name for m in members],pp,sha,run_dir/names["txt"],mp,R["verdict"])
    say(f"{run_id}: {len(checks)} pre-seal + {len(post)} post-seal checks = {len(checks)+len(post)}/{len(checks)+len(post)} PASS | render {render_s:.1f}s")
    say("="*110);say("FINAL VERDICT :",R["verdict"]);say("PACKAGE       : SEALED ·",f"{len(checks)+len(post)} checks","· payload SHA-256",sha);say("ZIP           :",zp);say("="*110)
    return dict(run_id=run_id,run_dir=run_dir,P=P,R=R,sha=sha,imgs=imgs,zip=zp,payload=pp,txt=run_dir/names["txt"],manifest=mp,images_manifest=imf,summary=run_dir/names["summary"],jsonl=run_dir/names["jsonl"],
                checks=len(checks)+len(post),verdict=R["verdict"])
#<<CORE_END>>
#<<POSTERS_BEGIN>>
from matplotlib.text import Text
plt.rcParams["font.family"]="DejaVu Sans";DPI=120
C_OK,C_UNK,C_ERR,C_CTL,C_CART,C_MOD,C_FG,C_NEU,C_BG,C_SEAL="#047857","#6D28D9","#B91C1C","#475569","#B45309","#0369A1","#0F172A","#334155","#F8FAFC","#0F766E"
C_OKL,C_UNKL,C_ERRL,C_CARTL,C_MODL,C_SEALL="#D1FAE5","#EDE9FE","#FEE2E2","#FEF3C7","#E0F2FE","#CCFBF1"
MONO="DejaVu Sans Mono";BRAND="AKBASCORE NIRVANA · MISTRAL · COGNITIVE CARTRIDGE"
def mt(s):return str(s).replace("$",r"\$")
def save_jpg(fig,path):
    buf=io.BytesIO();fig.savefig(buf,format="png",dpi=fig.dpi,facecolor="white");plt.close(fig);buf.seek(0)
    with Image.open(buf)as im:
        im=im.convert("RGBA");bg=Image.new("RGB",im.size,(255,255,255));bg.paste(im,mask=im.getchannel("A"))
    bg.save(path,"JPEG",quality=92,optimize=True,progressive=False,subsampling=0)
    with Image.open(path)as chk:
        if chk.format!="JPEG"or chk.mode!="RGB":raise RuntimeError("JPEG validation failed")
    return str(path)
def finish(fig,path,ctx,expect):
    """Poster text must contain every expected runtime value and no stale reference; otherwise the package is not sealed."""
    fig.canvas.draw();texts=[t.get_text()for t in fig.findobj(Text)if t.get_text().strip()];blob="\n".join(texts);nm=Path(path).name
    ws=lambda z:re.sub(r"\s+","",z);blob_ws=ws(blob);miss=[e for e in expect if ws(e)not in blob_ws];blob=re.sub(r"[ \t]*\n[ \t]*"," ",blob)
    if miss:plt.close(fig);raise AuditFail(f"POSTER/RESULT MISMATCH in {nm}: missing {miss}")
    hits=stale_scan(nm,blob,allow=ctx["allow"])+stale_scan("filename",nm)
    if hits:plt.close(fig);raise AuditFail(f"STALE REFERENCE in {nm}: {hits[:3]}")
    ctx["poster_audit"].append(dict(file=nm,texts=len(texts),expected_values=len(expect),stale_hits=0));return save_jpg(fig,path)
def fit_text(fig,x,y,w,h,text,fs_max=13,fs_min=7,color=C_FG,family=None,ls=1.32,weight="normal"):
    fig.canvas.draw();r=fig.canvas.get_renderer();Wp=w*fig.get_figwidth()*fig.dpi;Hp=h*fig.get_figheight()*fig.dpi
    if Wp<=4 or Hp<=4:return 0
    paras=str(text if text else"(empty)").replace("\r","").split("\n");kw={"va":"top","ha":"left","color":color,"linespacing":ls,"weight":weight}
    if family:kw["family"]=family
    def wrap(c):
        out=[]
        for p in paras:out.extend(textwrap.wrap(p,c,break_long_words=True,break_on_hyphens=False)or[""])
        return out
    fs=float(fs_max);k=0.52
    for _ in range(150):
        cpl=max(6,int(Wp/(k*fs*fig.dpi/72.0)));ln=wrap(cpl);t=fig.text(x,y+h,mt("\n".join(ln)),fontsize=fs,**kw);bb=t.get_window_extent(renderer=r)
        if bb.width>Wp*1.002 and cpl>6:t.remove();k*=max(1.01,bb.width/Wp*1.01);continue
        if bb.height<=Hp:return fs
        t.remove()
        if fs>fs_min:fs=max(float(fs_min),fs-0.5);continue
        per=bb.height/max(1,len(ln));keep=max(1,int(Hp/per)-1);fig.text(x,y+h,mt("\n".join(ln[:keep]+["[… text shortened — full text in the run log]"])),fontsize=fs,**kw);return fs
    fig.text(x,y+h,"[text omitted — see run log]",fontsize=fs_min,**kw);return fs_min
def head(fig,title,sub=None,tag=None):
    Hh=fig.get_figheight();f=lambda inch:1-inch/Hh;live=bool(tag)and tag.startswith("THIS");tc=C_MOD if live else C_SEAL
    fig.text(.05,f(.42),BRAND,fontsize=12,weight="bold",color=C_CART,va="center")
    if tag:fig.text(.95,f(.42),tag,fontsize=12,weight="bold",color=tc,va="center",ha="right",bbox=dict(boxstyle="round,pad=0.35",fc="white",ec=tc,lw=1.6))
    fig.text(.05,f(.95),mt(title),fontsize=29,weight="bold",color=C_FG,va="center")
    if sub:fig.text(.05,f(1.42),mt(sub),fontsize=13.5,color=C_NEU,va="center")
    fig.add_artist(Line2D([.05,.95],[f(1.70),f(1.70)],transform=fig.transFigure,color=C_FG,lw=1.2));return f(1.85)
def foot(fig,ctx,k):
    fig.text(.5,.22/fig.get_figheight(),mt(f"AKBASCORE NIRVANA · MISTRAL | RUN {ctx['run_id']} | PAYLOAD SHA-256 {ctx['sha'][:16]}… | {k:02d}/{ctx['N']:02d}"),ha="center",va="center",fontsize=10,color=C_NEU,family=MONO)
def note(fig,x,y,w,h,plain,sci=None):
    fit_text(fig,x,y+(h*.42 if sci else 0),w,h*(.58 if sci else 1),plain,fs_max=14,fs_min=9)
    if sci:fit_text(fig,x,y,w,h*.38,sci,fs_max=10.5,fs_min=7.5,family=MONO,color=C_NEU)
def clean_ax(ax):
    for s in("top","right"):ax.spines[s].set_visible(False)
def box(fig,x,y,w,h,title,lines,color,face,tfs=13.5,bfs=11):
    fig.add_artist(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=0.01",transform=fig.transFigure,facecolor=face,edgecolor=color,lw=2.4))
    fig.text(x+w/2,y+h*.74,mt(title),ha="center",va="center",fontsize=tfs,weight="bold",color=color)
    fig.text(x+w/2,y+h*.33,mt(lines),ha="center",va="center",fontsize=bfs,color=C_FG,linespacing=1.35)
def arrow(fig,x0,y0,x1,y1,color=C_FG):fig.add_artist(FancyArrowPatch((x0,y0),(x1,y1),transform=fig.transFigure,arrowstyle="-|>",mutation_scale=24,lw=2.3,color=color))
def chip(fig,x,y,w,h,label,value,color,face):
    fig.add_artist(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=0.008",transform=fig.transFigure,facecolor=face,edgecolor=color,lw=2))
    fig.text(x+w/2,y+h*.62,mt(value),ha="center",va="center",fontsize=19,weight="bold",color=color);fig.text(x+w/2,y+h*.22,mt(label),ha="center",va="center",fontsize=10.5,color=C_FG)
def kn(c,tot,k):return f"{c[k]}/{tot[k]}"
LBL={"mw1":"MW1","mw2":"MW2","linked":"LINKED","missing":"MISSING","absent_id":"ABSENT-ID","nomem":"NOMEM"}
def ctx_counts(ctx):R=ctx["R"];return R["counts"],R["totals"]
def sc(s):return s if len(s)<=14 else s[:13]+"…"
# ------------------------------------------------------------------ 01 pipeline
def p01(ctx,k,path):
    P=ctx["P"];R=ctx["R"];C,T=ctx_counts(ctx);fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white");tl=[c["T"]for c in P["cartridges"]];m=P["model"]["arch"]
    top=head(fig,"WHAT JUST HAPPENED?","Text went in once, became numbers, the text was removed, and a frozen model answered from the numbers.",tag="THIS LIVE RUN")
    bw,bh=.205,.17;xs=[.05+i*(bw+.0267)for i in range(4)];y1=.60
    box(fig,xs[0],y1,bw,bh,"① SOURCE TEXT",f"16 short records\n{min(tl)-1}–{max(tl)-1} tokens each",C_FG,C_BG)
    box(fig,xs[1],y1,bw,bh,"② FORGE","one forward pass per record\nmodel's own K/V states",C_MOD,C_MODL)
    box(fig,xs[2],y1,bw,bh,"③ D120 CARTRIDGE",f"16 independent cartridges\n{P['bank']['slots']} slots · numbers only",C_CART,C_CARTL)
    box(fig,xs[3],y1,bw,bh,"④ SOURCE REMOVED",f"readout prompts audited: {P['source_removal']['prompts_checked']}\nsource records found: {P['source_removal']['source_record_hits']}",C_ERR,C_ERRL)
    for i in range(3):arrow(fig,xs[i]+bw,y1+bh/2,xs[i+1],y1+bh/2)
    arrow(fig,xs[3]+bw/2,y1,xs[3]+bw/2,.485)
    ex_o=OBJECTS[0];ex_id=R["mw1"]["decided"][0]or"UNKNOWN";ex_pl=R["mw2"]["decided"][0]or"UNKNOWN"
    bw2=.168;g2=.0125;xs2=[.05+i*(bw2+g2)for i in range(5)];y2=.315;bh2=.165
    box(fig,xs2[0],y2,bw2,bh2,"⑤ QUESTION",f'object\n"{ex_o}"',C_FG,C_BG,tfs=12.5,bfs=10.5)
    box(fig,xs2[1],y2,bw2,bh2,"⑥ MW1",f"16 isolated reads\none unique answer",C_MOD,C_MODL,tfs=12.5,bfs=10.5)
    box(fig,xs2[2],y2,bw2,bh2,"⑦ CONTAINER ID",f"{ex_id}",C_OK,C_OKL,tfs=12.5,bfs=14)
    box(fig,xs2[3],y2,bw2,bh2,"⑧ MW2","same 16 cartridges\nasked about the ID",C_MOD,C_MODL,tfs=12.5,bfs=10.5)
    box(fig,xs2[4],y2,bw2,bh2,"⑨ LOCATION",f"{ex_pl}",C_OK,C_OKL,tfs=12.5,bfs=14)
    for i in range(4):arrow(fig,xs2[i]+bw2,y2+bh2/2,xs2[i+1],y2+bh2/2)
    fig.text(.05,.29,"Example shown: the first locked record (Q01), not selected after the run. All 16 locked questions are on poster 06.",fontsize=10.5,color=C_NEU,style="italic")
    cw=.1325;xc=[.05+i*(cw+.01)for i in range(6)]
    for i,key in enumerate(LBL):
        good=C[key]>=GATE_MIN[key];chip(fig,xc[i],.155,cw,.09,LBL[key],kn(C,T,key),C_OK if good else C_ERR,C_OKL if good else C_ERRL)
    fig.text(.05,.128,f"THIS LIVE RUN · verdict by TEST474 gates: {R['verdict']}",fontsize=12,weight="bold",color=C_OK if R["verdict"].startswith("PASS")else C_ERR)
    fit_text(fig,.05,.04,.9,.075,f"Frozen {MODEL_SHORT} · {m[0]} layers · hidden {m[1]} · {m[2]} Q / {m[3]} KV heads · head {m[4]} · BF16 · SDPA · K = D120 + V = D120 · OWN first-token state kept · greedy · no training, no LoRA, no optimizer, no weight update",fs_max=11,fs_min=8,color=C_NEU)
    foot(fig,ctx,k);return finish(fig,path,ctx,[kn(C,T,x)for x in LBL]+[R["verdict"],str(P["source_removal"]["prompts_checked"])])
# ------------------------------------------------------------------ 02 anatomy
def p02(ctx,k,path):
    P=ctx["P"];C0=P["cartridges"][0];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white");m=P["model"]["arch"];B=P["bank"]
    top=head(fig,"WHAT IS INSIDE A CARTRIDGE?","Numbers only: the model's own K/V states, projected on 120 fixed directions. Shown for cartridge C01.",tag="THIS LIVE RUN")
    L_=P["experimenter_ledger_not_available_to_readout"][0]["source_text"]
    steps=[("SOURCE RECORD (forge time only)",L_.replace("\n"," | "),C_FG,C_BG),("TOKENS",f"{C0['T']} slots: 1 first-token state + {C0['T']-1} content tokens",C_FG,C_BG),
           ("MODEL FORWARD",f"K and V per layer: {m[0]} layers × {m[3]} KV heads × {m[4]} dims",C_MOD,C_MODL),
           ("D120 PROJECTION","fixed neutral codebook → 120 coefficients per head and token (K and V)",C_CART,C_CARTL),("NUMERICAL CARTRIDGE",f"{C0['code_numbers']:,} coefficients + {C0['own_numbers']:,} raw first-token numbers",C_OK,C_OKL)]
    y=top-.01;hh=.108
    for i,(t,b,c,fc)in enumerate(steps):
        yy=y-(i+1)*(hh+.022)+.022;fig.add_artist(FancyBboxPatch((.05,yy),.40,hh,boxstyle="round,pad=0,rounding_size=0.008",transform=fig.transFigure,facecolor=fc,edgecolor=c,lw=2.2))
        fig.text(.062,yy+hh-.018,t,fontsize=11.5,weight="bold",color=c,va="top");fit_text(fig,.062,yy+.006,.376,hh-.04,b,fs_max=10.5,fs_min=8,family=MONO if i==0 else None)
        if i<len(steps)-1:fig.text(.25,yy-.011,"▼",fontsize=10,color=C_NEU,ha="center",va="center")
    ex=C0["excerpt"];V=np.array(ex["values"],dtype=float);ax=fig.add_axes([.54,.545,.40,.215]);lim=np.percentile(np.abs(V),97)or 1.0
    im=ax.imshow(V,aspect="auto",cmap="RdBu_r",vmin=-lim,vmax=lim);ax.set_xlabel("PCA coefficient index (0–119)",fontsize=9.5,labelpad=2);ax.set_ylabel("content token",fontsize=9.5)
    ax.set_title(f"REAL DATA · C01 · layer {ex['layer']} · K head {ex['head']} · coefficients",fontsize=11.5,weight="bold",loc="left",pad=8);fig.colorbar(im,cax=fig.add_axes([.945,.545,.008,.215]))
    cK=np.array([c["cosK"]for c in P["cartridges"]]);cV=np.array([c["cosV"]for c in P["cartridges"]]);ax2=fig.add_axes([.54,.255,.40,.17]);xs=np.arange(len(cK[0]))
    for arr,col,lab in((cK,C_CART,"K"),(cV,C_MOD,"V")):
        ax2.fill_between(xs,arr.min(0),arr.max(0),color=col,alpha=.2);ax2.plot(xs,arr.mean(0),color=col,lw=2.2,label=f"{lab}: mean over 16 cartridges (band = min–max)")
    lo=min(cK.min(),cV.min());ax2.set_ylim(max(0,lo-.01),1.003);ax2.set_xlabel("transformer layer",fontsize=9.5,labelpad=2);ax2.set_ylabel("cosine to own K/V",fontsize=9.5)
    ax2.legend(frameon=False,fontsize=8.8,loc="lower left");ax2.set_title("MEASURED · how faithful is the D120 projection?",fontsize=11.5,weight="bold",loc="left",pad=8);clean_ax(ax2);ax2.grid(alpha=.25)
    fit_text(fig,.05,.045,.9,.085,f"NO HUMAN-READABLE SOURCE COPY. The cartridge holds {B['code_numbers']+B['own_numbers']:,} numbers for all 16 records ({B['ratio_vs_native']*100:.1f}% of the model's native K/V size: this is a projection, not a size reduction). The {len(NEUTRAL)}-sentence neutral codebook ({B['codebook_bytes']/2**20:.1f} MiB) is shared and holds none of the facts.",fs_max=11,fs_min=8)
    foot(fig,ctx,k);return finish(fig,path,ctx,[f"{C0['T']} slots",f"{B['code_numbers']+B['own_numbers']:,}",f"{B['ratio_vs_native']*100:.1f}%"])
# ------------------------------------------------------------------ 03 source removal
def p03(ctx,k,path):
    P=ctx["P"];S=P["source_removal"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"WAS THE SOURCE REALLY REMOVED?","What the frozen model receives at readout: a question as text, the knowledge only as installed numbers.",tag="THIS LIVE RUN")
    bw,bh=.27,.25;xs=[.05,.365,.68];y0=top-bh-.03
    box(fig,xs[0],y0,bw,bh,"FORGE TIME","SOURCE TEXT = PRESENT\n\nused once, inside the\nforward pass that makes a cartridge",C_FG,C_BG,bfs=11.5)
    box(fig,xs[1],y0,bw,bh,"AFTER FORGE","SOURCE TEXT = REMOVED\nNOT PROVIDED TO READOUT\n\nNO HUMAN-READABLE SOURCE COPY",C_ERR,C_ERRL,bfs=11.5)
    box(fig,xs[2],y0,bw,bh,"READOUT INPUT","QUESTION  +  NUMERICAL\nCARTRIDGE K/V\n\nMODEL WEIGHTS: UNCHANGED",C_MOD,C_MODL,bfs=11.5)
    arrow(fig,xs[0]+bw,y0+bh/2,xs[1],y0+bh/2);arrow(fig,xs[1]+bw,y0+bh/2,xs[2],y0+bh/2)
    calls=S["calls"];rows=[("MW1 · object → ID",calls["mw1"]),("MW2 · ID → location",calls["mw2"]),("missing object + absent ID",calls["neg"]),("NOMEM control",calls["nomem"])]
    ax=fig.add_axes([.27,.295,.60,.175]);ax.barh(range(4),[v for _,v in rows],color=[C_MOD,C_MOD,C_UNK,C_CTL]);ax.invert_yaxis();ax.set_yticks(range(4));ax.set_yticklabels([a for a,_ in rows],fontsize=11)
    for i,(_,v)in enumerate(rows):ax.text(v+4,i,f"{v} readout prompts · source records found: 0",va="center",fontsize=11,weight="bold")
    ax.set_xlim(0,max(v for _,v in rows)*1.9);ax.set_xlabel("readout prompts checked against all 16 source records",fontsize=10,labelpad=3);clean_ax(ax)
    note(fig,.05,.045,.9,.13,f"{S['prompts_checked']} readout prompts were compared with every source record before use; the readout input is [PAD]×slots + question tokens ({S['question_tokens_min']}–{S['question_tokens_max']} question tokens). The question must name the object or container ID it asks about — that is the question, not the record.",
         "A cartridge is a set of K/V tensors (coefficients + reconstructed rows). It is not text, a summary, or an embedding of the sentence.")
    foot(fig,ctx,k);return finish(fig,path,ctx,[str(S["prompts_checked"]),"SOURCE TEXT = REMOVED"])
# ------------------------------------------------------------------ 04 bank
def p04(ctx,k,path):
    P=ctx["P"];C=P["cartridges"];L=P["experimenter_ledger_not_available_to_readout"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"THE CARTRIDGE BANK · 16 INDEPENDENT CARTRIDGES","Each cartridge was forged alone: its own forward pass, its own first-token state, its own numbers.",tag="THIS LIVE RUN")
    gw,gh=.098,(top-.17)/4
    for i,c in enumerate(C):
        r,q=divmod(i,4);x=.05+q*(gw+.007);y=top-(r+1)*gh-.005
        fig.add_artist(FancyBboxPatch((x,y),gw,gh-.012,boxstyle="round,pad=0,rounding_size=0.008",transform=fig.transFigure,facecolor=C_CARTL,edgecolor=C_CART,lw=2.2))
        fig.text(x+gw/2,y+gh-.04,c["cid"],fontsize=16,weight="bold",color=C_CART,ha="center",va="center");fig.text(x+gw/2,y+(gh-.012)*.52,f"{c['T']} slots",fontsize=10.5,ha="center",va="center",color=C_FG)
        fig.text(x+gw/2,y+(gh-.012)*.30,f"{(c['code_numbers']+c['own_numbers'])*2/1024:.0f} KiB",fontsize=10,ha="center",va="center",color=C_NEU);fig.text(x+gw/2,y+(gh-.012)*.12,"numbers only",fontsize=8.5,ha="center",va="center",color=C_NEU,style="italic")
    ax=fig.add_axes([.52,.14,.43,top-.17]);ax.axis("off");ax.set_xlim(0,1);ax.set_ylim(0,1)
    ax.add_patch(Rectangle((0,.93),1,.07,color=C_SEALL,transform=ax.transAxes));ax.text(.01,.965,"FORGE-TIME LEDGER (kept by the experimenter for scoring)",fontsize=10.5,weight="bold",color=C_SEAL,va="center")
    ax.text(.01,.91,"The readout never receives this table.",fontsize=9.5,color=C_NEU,va="top",style="italic")
    for i,l in enumerate(L):
        y=.79-i*.0495;ax.text(.01,y,l["cid"],fontsize=9.5,weight="bold",color=C_CART,va="center");ax.text(.12,y,l["object"],fontsize=9.5,va="center",color=C_FG);ax.text(.52,y,l["id"],fontsize=9.5,family=MONO,va="center",color=C_FG);ax.text(.70,y,l["location"],fontsize=9.5,va="center",color=C_FG)
    ax.text(.12,.855,"object",fontsize=9,weight="bold",color=C_NEU);ax.text(.52,.855,"container ID",fontsize=9,weight="bold",color=C_NEU);ax.text(.70,.855,"location",fontsize=9,weight="bold",color=C_NEU)
    fit_text(fig,.05,.045,.43,.075,"Cartridge size = D120 coefficients + raw first-token state (bf16). Independent means: no cartridge was forged with, or conditioned on, another.",fs_max=11,fs_min=8,color=C_NEU)
    foot(fig,ctx,k);return finish(fig,path,ctx,["C01","C16",f"{C[0]['T']} slots"])
# ------------------------------------------------------------------ 05 one question
def lane_row(fig,ctx,y,reads,kind,gold_idx,decided_fn):
    bw=(0.90-15*.004)/16;bh=.075
    for j,r in enumerate(reads):
        x=.05+j*(bw+.004);txt=first(r["raw"]);none=is_none(r["raw"])
        okv=(parse_id(r["raw"])if kind=="id"else parse_place(r["raw"]))is not None and not none
        face,edge,col=(C_OK,C_OK,"white")if(j==gold_idx and okv)else((C_UNKL,C_UNK,C_UNK)if none else(C_ERRL,C_ERR,C_ERR))
        fig.add_artist(FancyBboxPatch((x,y),bw,bh,boxstyle="round,pad=0,rounding_size=0.004",transform=fig.transFigure,facecolor=face,edgecolor=edge,lw=1.6))
        fig.text(x+bw/2,y+bh-.012,f"C{j+1:02d}",fontsize=8.5,weight="bold",color=col if j==gold_idx else C_NEU,ha="center",va="top")
        fig.text(x+bw/2,y+.027,textwrap.fill("NONE"if none else sc(txt),9),fontsize=7.8,weight="bold",color=col,ha="center",va="center",linespacing=1.0)
def p05(ctx,k,path):
    P=ctx["P"];R=ctx["R"];raw=P["raw"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white");i=0
    top=head(fig,"ONE QUESTION, STEP BY STEP","Every cartridge reads the same question in isolation. Only the cartridge that holds the answer returns it.",tag="THIS LIVE RUN")
    q=raw["mw1"][i];rs=q["reads"];d1,u1=decide_id([r["raw"]for r in rs])
    fig.text(.05,top-.025,"STAGE 1 · MW1 · OBJECT → CONTAINER ID",fontsize=13,weight="bold",color=C_MOD,va="top")
    fit_text(fig,.05,top-.105,.9,.06,f'Question to all 16 cartridges: Does this memory explicitly contain the object "{OBJECTS[i]}"? If yes, answer only its container identifier. If no, answer exactly NONE.',fs_max=11.5,fs_min=8.5,color=C_NEU)
    lane_row(fig,ctx,top-.20,rs,"id",i,None)
    fig.text(.5,top-.225,"▼  unique aggregation of the 16 answers",ha="center",fontsize=11,color=C_NEU)
    fig.add_artist(FancyBboxPatch((.30,top-.315),.40,.062,boxstyle="round,pad=0,rounding_size=0.008",transform=fig.transFigure,facecolor=C_OKL if d1 else C_ERRL,edgecolor=C_OK if d1 else C_ERR,lw=2.2))
    fig.text(.5,top-.284,f"unique IDs = {{{', '.join(u1)}}}  →  {'accepted: '+d1 if d1 else 'UNKNOWN'}",ha="center",va="center",fontsize=13,weight="bold",color=C_OK if d1 else C_ERR)
    q2d=raw["mw2"][i];y2=top-.375
    fig.text(.05,y2+.02,"STAGE 2 · MW2 · CONTAINER ID → LOCATION",fontsize=13,weight="bold",color=C_MOD,va="top")
    if q2d.get("skipped"):
        fit_text(fig,.05,y2-.08,.9,.06,"Stage 1 returned no unique ID, so Stage 2 was not run for this question.",fs_max=12,color=C_ERR);d2="UNKNOWN";u2=[]
    else:
        fit_text(fig,.05,y2-.075,.9,.06,f'Question to all 16 cartridges: Does this memory explicitly contain container "{q2d["id"]}"? If yes, answer only its location. If no, answer exactly NONE.',fs_max=11.5,fs_min=8.5,color=C_NEU)
        lane_row(fig,ctx,y2-.17,q2d["reads"],"place",i,None);d2o,u2=decide_place([r["raw"]for r in q2d["reads"]]);d2=d2o or"UNKNOWN"
        fig.text(.5,y2-.195,"▼  unique aggregation of the 16 answers",ha="center",fontsize=11,color=C_NEU)
        fig.add_artist(FancyBboxPatch((.30,y2-.285),.40,.062,boxstyle="round,pad=0,rounding_size=0.008",transform=fig.transFigure,facecolor=C_OKL if d2o else C_ERRL,edgecolor=C_OK if d2o else C_ERR,lw=2.2))
        fig.text(.5,y2-.254,f"unique places = {{{', '.join(u2)}}}  →  {'ANSWER: '+d2o if d2o else 'UNKNOWN'}",ha="center",va="center",fontsize=13,weight="bold",color=C_OK if d2o else C_ERR)
    lg=[Line2D([0],[0],marker="s",color="w",markerfacecolor=C_OK,markersize=12,label="cartridge returned the answer"),Line2D([0],[0],marker="s",color="w",markerfacecolor=C_UNKL,markeredgecolor=C_UNK,markersize=12,label="cartridge said NONE"),Line2D([0],[0],marker="s",color="w",markerfacecolor=C_ERRL,markeredgecolor=C_ERR,markersize=12,label="unexpected output")]
    fig.legend(handles=lg,loc="lower center",bbox_to_anchor=(.5,.075),ncol=3,frameon=False,fontsize=10.5)
    fig.text(.05,.045,"There is no router: all 16 cartridges are always read. Boxes show the first line each cartridge generated (live, unedited).",fontsize=10.5,color=C_NEU)
    foot(fig,ctx,k);return finish(fig,path,ctx,["unique aggregation of the 16 answers",OBJECTS[i]])
# ===== END PART 2 / 3 — CONTINUE WITH PART 3 =====
# ------------------------------------------------------------------ 06 all questions
def p06(ctx,k,path):
    P=ctx["P"];R=ctx["R"];C,T=ctx_counts(ctx);fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"ALL 16 QUESTIONS · 16 × 16 ISOLATED READS","Row = question, column = cartridge. Green: the cartridge that holds the answer returned it. Purple: NONE.",tag="THIS LIVE RUN")
    cmap=ListedColormap(["#E9D5FF",C_OK,C_ERR,"#F97316","#E2E8F0"]);code={"none":0,"gold":1,"gold_miss":2,"other":3}
    for n,(key,kind,dec,exp,title)in enumerate((("mw1","id",R["mw1"]["decided"],IDS,"MW1 · object → container ID"),("mw2","place",R["mw2"]["decided"],PLACES,"MW2 · container ID → location"))):
        x0=.14+n*.45;ax=fig.add_axes([x0,.22,.28,top-.30]);M=np.array([[code[c]for c in row]if row else[4]*16 for row in R[key]["cls"]])
        ax.imshow(M,aspect="auto",cmap=cmap,vmin=0,vmax=4);ax.set_xticks(range(16));ax.set_xticklabels([f"C{j+1:02d}"for j in range(16)],rotation=90,fontsize=8.2)
        ax.set_yticks(range(16));ax.set_yticklabels([f"Q{i+1:02d} "+(OBJECTS[i]if n==0 else IDS[i])for i in range(16)]if n==0 else[f"Q{i+1:02d} {IDS[i]}"for i in range(16)],fontsize=8.8)
        ax.set_xticks(np.arange(-.5,16,1),minor=True);ax.set_yticks(np.arange(-.5,16,1),minor=True);ax.grid(which="minor",color="white",lw=1.2);ax.tick_params(which="minor",length=0)
        ax.set_title(title,fontsize=12.5,weight="bold",loc="left")
        for i in range(16):
            ok=dec[i]==exp[i];ax.text(16.1,i,f"{dec[i]or'UNKNOWN'} {'✓'if ok else'✗'}",fontsize=8.5,va="center",color=C_OK if ok else C_ERR,weight="bold",clip_on=False)
    w1=R["mw1"];w2=R["mw2"]
    lg=[Line2D([0],[0],marker="s",color="w",markerfacecolor=C_OK,markersize=11,label="gold cartridge returned the answer"),Line2D([0],[0],marker="s",color="w",markerfacecolor="#E9D5FF",markeredgecolor=C_UNK,markersize=11,label="NONE"),Line2D([0],[0],marker="s",color="w",markerfacecolor="#F97316",markersize=11,label="non-gold cartridge returned something else"),Line2D([0],[0],marker="s",color="w",markerfacecolor=C_ERR,markersize=11,label="gold cartridge missed"),Line2D([0],[0],marker="s",color="w",markerfacecolor="#E2E8F0",markeredgecolor=C_CTL,markersize=11,label="MW2 not run (MW1 gave no unique ID)")]
    fig.legend(handles=lg,loc="lower center",bbox_to_anchor=(.5,.075),ncol=3,frameon=False,fontsize=9.4)
    fig.text(.05,.045,f"LIVE · MW1 {kn(C,T,'mw1')} · MW2 {kn(C,T,'mw2')} · wrong-cartridge NONE: stage 1 {w1['wrong_none']}/{w1['wrong_total']} · stage 2 {w2['wrong_none']}/{w2['wrong_total']}",fontsize=12,weight="bold",color=C_FG)
    foot(fig,ctx,k);return finish(fig,path,ctx,[kn(C,T,"mw1"),kn(C,T,"mw2"),f"{w1['wrong_none']}/{w1['wrong_total']}",f"{w2['wrong_none']}/{w2['wrong_total']}"])
# ------------------------------------------------------------------ 07 results
def p07(ctx,k,path):
    P=ctx["P"];R=ctx["R"];C,T=ctx_counts(ctx);S=SEALED474;fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"RESULTS · THIS RUN vs THE SEALED TEST474 RECORD","Same panel, same engine, same gates. The sealed record is historical; the live column was measured just now.")
    fig.text(.40,top-.015,"THIS LIVE RUN",fontsize=13,weight="bold",color=C_MOD,ha="center");fig.text(.64,top-.015,"SEALED TEST474",fontsize=13,weight="bold",color=C_SEAL,ha="center");fig.text(.84,top-.015,"GATE (≥)",fontsize=13,weight="bold",color=C_NEU,ha="center")
    keys=list(LBL);rh=(top-.30)/(len(keys)+2)
    for i,key in enumerate(keys):
        y=top-.06-(i+1)*rh;live=C[key];tot=T[key];good=live>=GATE_MIN[key]
        fig.text(.05,y+rh*.4,LBL[key],fontsize=15,weight="bold",color=C_FG,va="center")
        ax=fig.add_axes([.27,y+rh*.12,.26,rh*.55]);ax.barh([0],[tot],color="#E2E8F0");ax.barh([0],[live],color=C_OK if good else C_ERR);ax.set_xlim(0,tot);ax.axis("off");ax.text(tot/2,0,kn(C,T,key),ha="center",va="center",fontsize=15,weight="bold",color="white")
        sv=S[key];ax2=fig.add_axes([.55,y+rh*.12,.18,rh*.55]);ax2.barh([0],[sv[1]],color="#E2E8F0");ax2.barh([0],[sv[0]],color=C_SEAL);ax2.set_xlim(0,sv[1]);ax2.axis("off");ax2.text(sv[1]/2,0,fr(sv),ha="center",va="center",fontsize=14,weight="bold",color="white")
        fig.text(.84,y+rh*.4,f"{GATE_MIN[key]}/{tot}",fontsize=13,color=C_NEU,ha="center",va="center")
    y=top-.06-(len(keys)+1)*rh
    fig.text(.05,y+rh*.4,"WEIGHT SENTINEL",fontsize=15,weight="bold",color=C_FG,va="center");fz=P["frozen"];sp=fz["sentinel_startup"]==fz["sentinel_after"]
    fig.text(.40,y+rh*.4,"PASS"if sp else"FAIL",fontsize=17,weight="bold",color=C_OK if sp else C_ERR,ha="center",va="center");fig.text(.64,y+rh*.4,S["sentinel"],fontsize=17,weight="bold",color=C_SEAL,ha="center",va="center")
    ok=R["verdict"].startswith("PASS");fig.add_artist(FancyBboxPatch((.05,.105),.90,.075,boxstyle="round,pad=0,rounding_size=0.01",transform=fig.transFigure,facecolor=C_OKL if ok else C_ERRL,edgecolor=C_OK if ok else C_ERR,lw=2.6))
    fig.text(.07,.1425,"LIVE VERDICT",fontsize=12,weight="bold",color=C_NEU,va="center");fig.text(.25,.1425,R["verdict"],fontsize=17,weight="bold",color=C_OK if ok else C_ERR,va="center",family=MONO)
    fig.text(.93,.1425,f"sealed: {S['verdict']}",fontsize=10.5,color=C_SEAL,va="center",ha="right",family=MONO)
    fit_text(fig,.05,.045,.9,.05,f"Replay identical to the sealed counts: {'YES'if R['replay_match']else'NO'}. Gate thresholds are the ones precommitted in TEST474. NOMEM passes when the model without cartridges does not output the stored place.",fs_max=11,fs_min=8.5,color=C_NEU)
    foot(fig,ctx,k);return finish(fig,path,ctx,[kn(C,T,x)for x in keys]+[R["verdict"],S["verdict"]])
# ------------------------------------------------------------------ 08 no router
def p08(ctx,k,path):
    P=ctx["P"];R=ctx["R"];w1=R["mw1"];w2=R["mw2"];Q=SEALED473;fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"NO ROUTER · THE OTHER 15 CARTRIDGES SAY NONE","Nothing chooses a cartridge in advance. Every question goes to all 16; only the right one answers.",tag="THIS LIVE RUN")
    box(fig,.05,top-.20,.16,.13,"QUESTION","one object or ID",C_FG,C_BG,tfs=12,bfs=10.5);arrow(fig,.21,top-.135,.265,top-.135)
    for i in range(16):
        r,q=divmod(i,4);x=.28+q*.032;y=top-.075-r*.034;fig.add_artist(FancyBboxPatch((x,y-.026),.027,.027,boxstyle="round,pad=0,rounding_size=0.003",transform=fig.transFigure,facecolor=C_CARTL,edgecolor=C_CART,lw=1.4));fig.text(x+.0135,y-.0125,f"{i+1:02d}",fontsize=7.5,ha="center",va="center",color=C_CART)
    fig.text(.34,top-.225,"same question copied to all 16",ha="center",fontsize=10,color=C_NEU);arrow(fig,.425,top-.135,.475,top-.135)
    fig.add_artist(FancyBboxPatch((.48,top-.20),.027,.027,boxstyle="round,pad=0,rounding_size=0.003",transform=fig.transFigure,facecolor=C_OK,edgecolor=C_OK));fig.text(.52,top-.1865,"1 cartridge returns the answer",fontsize=11,va="center",color=C_OK,weight="bold")
    fig.add_artist(FancyBboxPatch((.48,top-.15),.027,.027,boxstyle="round,pad=0,rounding_size=0.003",transform=fig.transFigure,facecolor=C_UNKL,edgecolor=C_UNK));fig.text(.52,top-.1365,"15 cartridges return NONE",fontsize=11,va="center",color=C_UNK,weight="bold")
    arrow(fig,.74,top-.135,.785,top-.135);box(fig,.79,top-.20,.16,.13,"UNIQUE","aggregation\nof the answers",C_OK,C_OKL,tfs=12,bfs=10.5)
    ax=fig.add_axes([.22,.29,.70,.26]);labs=["LIVE · stage 1\nobject → ID reads","LIVE · stage 2\nID → location reads","SEALED TEST473\nstage-1 reads"]
    vals=[(w1["wrong_none"],w1["wrong_total"]),(w2["wrong_none"],w2["wrong_total"]),Q["decoy_none"]];cols=[C_MOD,C_MOD,C_SEAL]
    ax.barh(range(3),[v[1]for v in vals],color="#E2E8F0");ax.barh(range(3),[v[0]for v in vals],color=cols);ax.invert_yaxis();ax.set_yticks(range(3));ax.set_yticklabels(labs,fontsize=11)
    for i,v in enumerate(vals):ax.text(v[1]*.5,i,f"{v[0]}/{v[1]} wrong-cartridge reads returned NONE",ha="center",va="center",fontsize=12.5,weight="bold",color="white")
    ax.set_xlim(0,max(v[1]for v in vals));ax.set_xlabel("reads of a cartridge that does NOT hold the asked item",fontsize=10,labelpad=3);clean_ax(ax)
    note(fig,.05,.04,.9,.115,f"Live check: reads per question = {R['reads_per_query'][0]} for every question, so no cartridge was pre-selected. The sealed TEST473 count (16 questions × 15 wrong cartridges = 240) was measured in a different run and is shown for comparison only; it is not added to the live count.",
         "Router: none. Hidden scorer: none. A wrong-cartridge read counts as NONE when its first line contains NONE and no container ID (is_none, TEST473/474).")
    foot(fig,ctx,k);return finish(fig,path,ctx,[f"{w1['wrong_none']}/{w1['wrong_total']}",f"{w2['wrong_none']}/{w2['wrong_total']}",fr(Q["decoy_none"])])
# ------------------------------------------------------------------ 09 not in the cartridges
def p09(ctx,k,path):
    P=ctx["P"];R=ctx["R"];C,T=ctx_counts(ctx);fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"WHEN THE ANSWER IS NOT IN THE CARTRIDGES","Missing object and absent ID: all 16 cartridges must say NONE. NOMEM: the same question with no cartridge at all.",tag="THIS LIVE RUN")
    for n,(key,title,good)in enumerate((("missing",f"MISSING OBJECT · {kn(C,T,'missing')} → UNKNOWN",C["missing"]),("absent",f"ABSENT CONTAINER ID · {kn(C,T,'absent_id')} → UNKNOWN",C["absent_id"]))):
        ax=fig.add_axes([.12+n*.45,top-.315,.30,.225]);rows=R[key];nn=[m["none"]for m in rows];nt=[m["n"]for m in rows]
        ax.barh(range(8),nt,color=C_ERRL);ax.barh(range(8),nn,color=C_UNK);ax.invert_yaxis();ax.set_yticks(range(8));ax.set_yticklabels([m["query"]for m in rows],fontsize=9.5)
        for i,m in enumerate(rows):ax.text(m["n"]+.3,i,("✓ UNKNOWN"if m["ok"]else"✗ answered"),va="center",fontsize=9.5,weight="bold",color=C_OK if m["ok"]else C_ERR)
        ax.set_xlim(0,24);ax.set_xlabel("cartridges returning NONE (of 16)",fontsize=10);ax.set_title(title,fontsize=12,weight="bold",loc="left");clean_ax(ax)
    ax=fig.add_axes([.05,.13,.90,.285]);ax.axis("off");ax.set_xlim(0,1);ax.set_ylim(0,1)
    ax.text(0,1.0,f"NOMEM CONTROL · {kn(C,T,'nomem')} · same Stage-2 question, no cartridge, nothing installed",fontsize=12,weight="bold",color=C_CTL,va="top")
    ax.text(0,.88,"success = the model does NOT output the stored place (TEST474 criterion)",fontsize=9.5,color=C_NEU,va="top",style="italic")
    for c_,x in(("container ID",.0),("stored place (hidden from the model)",.17),("model's first line without cartridge",.47),("result",.82)):ax.text(x,.76,c_,fontsize=9.5,weight="bold",color=C_NEU,va="center")
    for i,m in enumerate(R["nomem"]):
        y=.67-i*.088;ax.text(0,y,m["id"],fontsize=9.5,family=MONO,va="center");ax.text(.17,y,m["target"],fontsize=9.5,va="center");ax.text(.47,y,sc(m["first"])if m["first"]else"(empty)",fontsize=9.5,va="center",family=MONO)
        ax.text(.82,y,"PASS"if m["ok"]else"FAIL",fontsize=9.5,weight="bold",color=C_OK if m["ok"]else C_ERR,va="center")
    fit_text(fig,.05,.045,.9,.07,"Missing and absent questions ask about things that were never stored, to every cartridge. NOMEM shows the correct answers do not come from the model's own knowledge of these invented records.",fs_max=11,fs_min=8.5,color=C_NEU)
    foot(fig,ctx,k);return finish(fig,path,ctx,[kn(C,T,"missing"),kn(C,T,"absent_id"),kn(C,T,"nomem")])
# ------------------------------------------------------------------ 10 layer corridor (sealed)
def p10(ctx,k,path):
    S=SEALED467;fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"WHICH LAYERS NEED THE CARTRIDGE?","Layer-range ablation on a D120 cartridge. Sealed TEST467 record on its own 8-case panel — not measured in this run.",tag="SEALED HISTORICAL RECORD · TEST467")
    cols={"K":C_CART,"V":C_MOD,"KV":C_OK}
    for n,(key,title)in enumerate((("entry","ENTRY SCAN · layers L07–L15"),("exit","EXIT SCAN · layers L18–L27"))):
        ax=fig.add_axes([.07+n*.47,.38,.40,top-.50]);data=S[key];xs=list(data.keys())
        for j,tn in enumerate(("K","V","KV")):ax.plot(range(len(xs)),[data[x][j]for x in xs],"o-",lw=2.4,ms=7,color=cols[tn],label=tn)
        ax.set_xticks(range(len(xs)));ax.set_xticklabels(xs,fontsize=9.5,rotation=45);ax.set_ylim(-.3,8.5);ax.set_yticks(range(0,9,2));ax.set_ylabel("hits out of 8",fontsize=10.5);ax.set_title(title,fontsize=12.5,weight="bold",loc="left");ax.grid(alpha=.25);clean_ax(ax)
        if n==0:ax.legend(frameon=False,fontsize=10.5,loc="lower left");ax.annotate("KV: 8/8 → 0/8\nbetween L12 and L13",xy=(6,0),xytext=(2.3,2.2),fontsize=10.5,weight="bold",color=C_OK,arrowprops=dict(arrowstyle="->",color=C_OK,lw=1.8))
        else:ax.text(4.6,1.2,"K and KV reach 8/8 at L23\nV reaches 8/8 at L26",fontsize=10.5,weight="bold",color=C_OK,va="center")
    ax=fig.add_axes([.07,.185,.86,.12]);ax.axis("off");ax.set_xlim(0,1);ax.set_ylim(0,1)
    for c_,x in(("tensor",.0),("last full entry-scan layer",.18),("first degraded layer",.44),("first full exit-scan layer",.68)):ax.text(x,.92,c_,fontsize=10,weight="bold",color=C_NEU,va="center")
    for i,tn in enumerate(("K","V","KV")):
        b=S["bounds"][tn];y=.62-i*.26;ax.text(0,y,tn,fontsize=11,weight="bold",color=cols[tn],va="center");ax.text(.18,y,b[0],fontsize=11,va="center",family=MONO);ax.text(.44,y,b[1],fontsize=11,va="center",family=MONO);ax.text(.68,y,b[2],fontsize=11,va="center",family=MONO)
    fit_text(fig,.05,.045,.9,.12,f"TEST467 used {S['cases']} cases with a D120 baseline of {fr(S['baseline'])}; it is a different panel from this demo. Read as functional sufficiency under controlled cumulative ablation: it does not show that the information lives only at L12–L13, and it was not re-measured in this run.",fs_max=11.5,fs_min=8.5,color=C_NEU)
    foot(fig,ctx,k);return finish(fig,path,ctx,["SEALED HISTORICAL RECORD · TEST467","L12","L27"])
# ------------------------------------------------------------------ 11 audit
def p11(ctx,k,path):
    P=ctx["P"];R=ctx["R"];fz=P["frozen"];S=P["source_removal"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"AUDIT LOCKS · FROZEN MODEL","Each lock is marked CHECKED (verified in this run) or BY CONSTRUCTION (a property of the code path).",tag="THIS LIVE RUN")
    ax=fig.add_axes([.05,top-.37,.34,.33]);ax.axis("off");ax.set_xlim(0,1);ax.set_ylim(0,1);ax.text(0,1,"WEIGHT SENTINEL · 3 CHECKPOINTS",fontsize=11.5,weight="bold",color=C_MOD,va="top")
    for j,(lab,key)in enumerate((("startup","sentinel_startup"),("pre-run","sentinel_pre_run"),("post-run","sentinel_after"))):
        y=.80-j*.17;same=fz[key]==fz["sentinel_startup"];ax.text(0,y,lab,fontsize=10.5,weight="bold",va="center");ax.text(.22,y,fz[key][:28]+"…",fontsize=8.8,family=MONO,va="center",color=C_FG);ax.text(1.0,y,"✓ identical"if same else"✗ CHANGED",fontsize=10,weight="bold",color=C_OK if same else C_ERR,va="center",ha="right")
    same_fp=fz["fingerprint_startup"]==fz["fingerprint_after"];ax.text(0,.27,f"float32 sums of the same {len(fz['tensors'])} tensors: {'identical'if same_fp else'CHANGED'}",fontsize=10,va="center",color=C_OK if same_fp else C_ERR)
    ax.text(0,.14,f"trainable tensors {fz['trainable_tensors']} · LoRA {'none'if not fz['lora']else'PRESENT'} · optimizer {'none'if not fz['optimizer']else'PRESENT'}",fontsize=10,va="center");ax.text(0,.02,f"sampled sentinel ({len(fz['tensors'])} tensors × 16 slices × 256 values), not a hash of all weights",fontsize=8.6,va="center",color=C_NEU,style="italic")
    locks=[("MODEL WEIGHTS","FROZEN","CHECKED","sentinel + fingerprint unchanged, 0 trainable tensors"),("WEIGHT SENTINEL","PASS" if fz["sentinel_startup"]==fz["sentinel_after"] else "FAIL","CHECKED","3 checkpoints identical (left)"),
           ("SOURCE AT READOUT","ABSENT","CHECKED",f"{S['prompts_checked']} prompts audited, 0 source records found"),("CARTRIDGE","NUMERICAL K/V","BY CONSTRUCTION","tensors only; no text is stored"),
           ("K / V","D120 / D120","CHECKED","both projected on 120 directions"),("OWN","PRESERVED","CHECKED",f"first-token K/V identical in {len(P['cartridges'])} cartridges × {P['model']['arch'][0]} layers"),
           ("CARTRIDGES","INDEPENDENT","CHECKED",f"{len(P['cartridges'])} separate forges, {2*len(P['cartridges'])} distinct tensor storages"),("READOUT","MODULE-WISE","CHECKED",f"{R['reads_per_query'][0]} reads per question"),
           ("AGGREGATION","UNIQUE","BY CONSTRUCTION","TEST474 decide_id / decide_place"),("ROUTER","NONE","CHECKED","every question read by all 16 cartridges"),
           ("TRAINING · LoRA · OPTIMIZER","NONE","CHECKED","no grads, no adapter, no optimizer object"),("WEIGHT UPDATE","NONE","CHECKED",f"sentinel unchanged after {P['counters']['generate_calls']} readout reads"),("GENERATION","GREEDY","BY CONSTRUCTION","do_sample = False"+("; stops after the first answer line (probe-checked)"if P["engine"].get("early_stop")else""))]
    ax2=fig.add_axes([.43,.10,.52,top-.12]);ax2.axis("off");ax2.set_xlim(0,1);ax2.set_ylim(0,1);rh=1/len(locks)
    for i,(a,b,c,d)in enumerate(locks):
        y=1-(i+.5)*rh;col=C_OK if c=="CHECKED"else C_CTL
        ax2.add_patch(FancyBboxPatch((0,y-rh*.42),1,rh*.84,boxstyle="round,pad=0,rounding_size=0.01",transform=ax2.transAxes,facecolor="white",edgecolor="#CBD5E1",lw=1.2))
        ax2.text(.015,y+rh*.12,a,fontsize=9.6,weight="bold",color=C_FG,va="center");ax2.text(.015,y-rh*.22,d,fontsize=8.3,color=C_NEU,va="center");ax2.text(.60,y,b,fontsize=11,weight="bold",color=C_OK if b not in("FAIL",)else C_ERR,va="center",family=MONO)
        ax2.text(.985,y,c,fontsize=8.8,weight="bold",color="white",va="center",ha="right",bbox=dict(boxstyle="round,pad=0.25",fc=col,ec=col))
    fit_text(fig,.05,.045,.34,.20,f"{len(P['checks_pre_seal'])} technical checks passed before sealing. Compute path: PyTorch CUDA backend, no custom CUDA/C++ kernel in this engine.",fs_max=11,fs_min=8.5,color=C_NEU)
    foot(fig,ctx,k);return finish(fig,path,ctx,["FROZEN","ABSENT","UNIQUE","NONE","GREEDY",str(S["prompts_checked"])])
# ------------------------------------------------------------------ 12 limits
def p12(ctx,k,path):
    P=ctx["P"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"WHAT THIS DEMO DOES — AND DOES NOT — SHOW","Read together with the results. These limits come from the experiments themselves.")
    shows=["A frozen Mistral-7B-Instruct-v0.3 recovered facts from D120 numerical K/V cartridges after the source text was removed from readout.","Sixteen independent cartridges were read module-wise; the unique-aggregation rule separated the one cartridge that holds an item from fifteen that do not.","Missing objects and absent IDs ended in UNKNOWN; the no-cartridge control did not reproduce the stored places.","The sealed TEST474 result was replayed live on the same panel."]
    nots=["Not a new held-out validation: the 16 records are the same ones used in TEST473 (where the frame was chosen) and TEST474.","Not evidence for other models, other record formats, or other fact types: records are short and synthetic.","Not evidence beyond 16 cartridges, and not a claim of unlimited or general memory.","Not training: the model keeps no permanent memory; cartridges are installed only for a question."]
    for n,(title,items,col,fc)in enumerate((("THIS RUN SHOWS",shows,C_OK,C_OKL),("THIS RUN DOES NOT SHOW",nots,C_ERR,C_ERRL))):
        x=.05+n*.465;fig.add_artist(FancyBboxPatch((x,top-.34),.435,.32,boxstyle="round,pad=0,rounding_size=0.01",transform=fig.transFigure,facecolor=fc,edgecolor=col,lw=2.3));fig.text(x+.015,top-.05,title,fontsize=13.5,weight="bold",color=col,va="center")
        for i,t in enumerate(items):fit_text(fig,x+.015,top-.175-i*.06,.405,.056,"• "+t,fs_max=11,fs_min=8.5)
    fig.text(.05,.445,"HOW THE SAME 16 RECORDS WERE USED",fontsize=13,weight="bold",color=C_NEU,va="center")
    ys=.275;bw,bh=.27,.085;items=[("TEST473",f"12 frame conditions compared on the 16 records; {SEALED473['tied_perfect']} tied at GOLD 16/16 and 240/240",C_SEAL,C_SEALL),("TEST474","S1_RECORD + C_VERIFY (chosen by fixed order among the 4) replayed on the same 16 records, plus new negative controls",C_SEAL,C_SEALL),("THIS DEMO","TEST474 panel replayed again, live: reproducibility, not new evidence",C_MOD,C_MODL)]
    for i,(a,b,c,fc)in enumerate(items):
        x=.05+i*(bw+.0425);fig.add_artist(FancyBboxPatch((x,ys),bw,bh+.04,boxstyle="round,pad=0,rounding_size=0.01",transform=fig.transFigure,facecolor=fc,edgecolor=c,lw=2.2));fig.text(x+.012,ys+bh+.022,a,fontsize=12,weight="bold",color=c,va="center");fit_text(fig,x+.012,ys+.006,bw-.024,bh-.012,b,fs_max=10.2,fs_min=8)
        if i<2:arrow(fig,x+bw,ys+(bh+.04)/2,x+bw+.0425,ys+(bh+.04)/2,color=C_NEU)
    note(fig,.05,.10,.9,.11,"Cartridge size is about the model's own K/V size (D120 keeps 120 of 128 directions per head), so this is not a storage-saving claim. Layer results on poster 10 come from a different test and were not re-measured here.")
    foot(fig,ctx,k);return finish(fig,path,ctx,["THIS RUN SHOWS","THIS RUN DOES NOT SHOW","TEST473","TEST474"])
# ------------------------------------------------------------------ 13 evidence seal
def p13(ctx,k,path):
    P=ctx["P"];R=ctx["R"];E_=P["environment"];tm=P["timing"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white");cnt=P["counters"];B=P["bank"]
    top=head(fig,"COMPLETE RUN · EVIDENCE SEAL","Everything below was read from the runtime and is contained in the sealed payload.",tag="THIS LIVE RUN")
    tiles=[("RUN ID",P["run_id"]),("MODEL",MODEL_ID),("GPU",E_["gpu"]),("TORCH / TRANSFORMERS",f"{E_['torch']} / {E_['transformers']}"),("READOUT READS",str(cnt["generate_calls"])),
           ("CARTRIDGES · SLOTS",f"16 · {B['slots']}"),("PANEL LOCK",LOCK_SHA[:22]+"…"),("TECHNICAL CHECKS",f"{len(P['checks_pre_seal'])} passed"),("PEAK GPU MEMORY",f"{P['gpu']['peak_allocated_gib']:.2f} GiB"),("VERDICT",R["verdict"])]
    for i,(a,b)in enumerate(tiles):
        r,q=divmod(i,5);x=.05+q*.182;y=top-.125-r*.125;fig.add_artist(FancyBboxPatch((x,y),.172,.105,boxstyle="round,pad=0,rounding_size=0.01",transform=fig.transFigure,facecolor=C_BG,edgecolor=C_MOD,lw=1.8))
        fig.text(x+.008,y+.083,a,fontsize=9.5,weight="bold",color=C_MOD,va="center");fit_text(fig,x+.008,y+.008,.156,.058,b,fs_max=12,fs_min=7.5,family=MONO)
    st=[("model load (startup)",tm["model_load_seconds"]),("codebook",tm["codebook"]),("16 cartridges",tm["forge"]),("MW1 readout",tm["mw1"]),("MW2 readout",tm["mw2"]),("missing + absent",tm["negatives"]),("NOMEM",tm["nomem"])]
    ax=fig.add_axes([.22,.30,.70,top-.55]);ax.barh(range(len(st)),[s for _,s in st],color=[C_CTL,C_CART,C_CART,C_MOD,C_MOD,C_UNK,C_CTL]);ax.invert_yaxis();ax.set_yticks(range(len(st)));ax.set_yticklabels([a for a,_ in st],fontsize=10.5)
    for i,(_,s)in enumerate(st):ax.text(s,i,f" {s:.1f} s",va="center",fontsize=10.5,weight="bold")
    ax.set_xlim(0,max(s for _,s in st)*1.18);ax.set_xlabel("measured seconds",fontsize=10);clean_ax(ax)
    txt=f"PAYLOAD FILE : {ctx['payload_name']}\nPAYLOAD SHA-256 : {ctx['sha']}\nENGINE : TEST474 (474.MISTRAL.final.py) engine functions, unchanged\nIMAGE HASHES : images manifest (per-image SHA-256), written after rendering\nArtifact integrity seal ≠ scientific proof: it shows the files are byte-identical to what this run recorded."
    fit_text(fig,.05,.05,.9,.215,txt,fs_max=11.5,fs_min=8,family=MONO)
    foot(fig,ctx,k);return finish(fig,path,ctx,[P["run_id"],E_["gpu"],E_["torch"],E_["transformers"],R["verdict"],str(cnt["generate_calls"])])
POSTERS=[("01_pipeline","What just happened?",p01),("02_cartridge_anatomy","What is inside a cartridge?",p02),("03_source_removal","Was the source removed?",p03),("04_cartridge_bank","The cartridge bank",p04),
 ("05_one_question","One question, step by step",p05),("06_all_questions","All 16 questions",p06),("07_results","Results: live vs sealed",p07),("08_no_router","No router",p08),
 ("09_not_in_cartridges","When the answer is not stored",p09),("10_layer_corridor","Layer corridor (sealed record)",p10),("11_audit_locks","Audit locks and frozen model",p11),("12_limits","What this demo does not show",p12),("13_evidence_seal","Evidence seal",p13)]
N_POSTERS=len(POSTERS)
if N_POSTERS>20 or[s[:2]for s,_,_ in POSTERS]!=[f"{i:02d}"for i in range(1,N_POSTERS+1)]:raise RuntimeError("Poster count/numbering check failed.")
def render_all(ctx,run_dir):
    imgs=[]
    try:
        for i,(slug,cap,fn)in enumerate(POSTERS,1):
            p=Path(run_dir)/f"{slug}_{ctx['run_id']}.jpg";fn(ctx,i,p);imgs.append((p,cap))
    finally:plt.close("all")
    return imgs
#<<POSTERS_END>>
#<<UI_BEGIN>>
class SelfTestEngine:
    """Synthetic engine used ONLY by the service self-test (no GPU, no model). Never used for a real run."""
    def __init__(s,mode="ok"):
        s.counters=Counter();s.mode=mode;s.n=0;s.rng=np.random.default_rng(1);s.sent="a"*64;s.fp=[1.0]*7
        s.info=dict(model_id=MODEL_ID,arch=ARCH,dtype="bfloat16",attn="sdpa",gpu="SELF-TEST ENGINE (no GPU)",gpu_total_gib=0.0,torch="n/a",transformers="n/a",gradio="n/a",python="n/a",platform="n/a",params=1,pad_id=0,eos_ids=[0],
            sliding_window=None,fp_names=["t0","t1","t2","t3","t4","t5","t6"],model_load_seconds=0.0,startup_utc="n/a",sentinel0=s.sent,fingerprint0=s.fp,hooks0=5,fast_first_line=True,sentinel_method="synthetic")
    def frozen_state(s):
        fh=3 if s.mode=="foreign_hook"else 0;return dict(training=False,trainable_tensors=0,lora=False,optimizer=False,hooks=5+fh,foreign_hooks=fh,foreign_modules=({"__main__":fh}if fh else{}),sentinel=("b"*64 if s.mode=="sentinel_changes"and s.n>50 else s.sent),fingerprint=s.fp)
    def build_codebook(s):s.counters["forge_passes"]+=32;return dict(sentences=32,svds=512,seconds=0.0,bytes=1)
    def codebook_bytes(s):return 1
    def forge_card(s,i):
        s.counters["forge_passes"]+=1;T=28+i%6;r=s.rng
        tel=dict(cosK=list(.985+.014*r.random(32)),cosV=list(.992+.007*r.random(32)),own_exact=True,T=T,encode_seconds=0.0,code_numbers=2*32*(T-1)*8*120,own_numbers=2*32*1024,native_numbers=2*32*1024*T,codec_macs=1,gpu_mib=0.0)
        if i==0:tel["excerpt"]=dict(layer=16,head=0,tensor="K",values=(r.normal(0,1,(T-1,120))*np.exp(-np.arange(120)/60)).round(4).tolist())
        return dict(rec=(OBJECTS[i],IDS[i],PLACES[i]),T=T,ptr=1000+i),tel
    def card_ptrs(s,c):return[c["ptr"]*2,c["ptr"]*2+1]
    def card_finite(s,c):return True
    def q_tokens(s,p):return len(p.split())
    def gen(s,c,p):
        s.counters["generate_calls"]+=1;s.n+=1;o,i,pl=c["rec"]
        if'object "'in p:return(i+"\nsynthetic")if re.search(r'object "(.+?)"',p).group(1)==o else"NONE\nsynthetic"
        return(pl+"\nsynthetic")if re.search(r'container "(.+?)"',p).group(1)==i else"NONE\nsynthetic"
    def gen_nomem(s,p):s.counters["generate_calls"]+=1;return"NONE"
    def gen_full(s,c,p):
        s.counters["probe_full_decodes"]+=1;o=s.gen(c,p);return o+"\nfull-length tail"if s.mode!="probe_mismatch"else"NONE\nx"if o.startswith("RQ")else"RQ-000"
    def gen_nomem_full(s,p):s.counters["probe_full_decodes"]+=1;return"NONE\nfull-length tail"
    def sync(s):pass
    def reset_peak(s):pass
    def peak_gib(s):return 0.0
def drain(gen):
    while True:
        try:next(gen)
        except StopIteration as s:return s.value
RUN_COUNTER=0;GPU_LOCK=threading.Lock()
SKIP=getattr(gr,"skip",None)or gr.update
_K=object()
FILE_KEYS=["zip","json","txt","man"]
DL_LABELS=["⬇ DOWNLOAD EVERYTHING (.zip)","⬇ DOWNLOAD FULL RUN LOG (.json)","⬇ DOWNLOAD READABLE RUN LOG (.txt)","⬇ DOWNLOAD RUN MANIFEST (.json)"]
DL_ALL_LABEL=f"⬇ DOWNLOAD ALL {N_POSTERS} JPGs"
RAW_KEYS=["txt","json","man"]
N_OUTPUTS=2+2+len(FILE_KEYS)*2+len(RAW_KEYS)
try:
    from gradio.routes import API_PREFIX as _GR_API_PREFIX
except Exception:
    _GR_API_PREFIX="/gradio_api"if int(gr.__version__.split(".")[0])>=5 else""
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
        paths=[files.get(k)for k in FILE_KEYS]
        for pth in paths:
            if pth is not None and not file_ready(pth):raise RuntimeError(f"Download artifact missing or empty: {pth}")
        out+=[dl_update(p,l)for p,l in zip(paths,DL_LABELS)]+[None if p is None else str(p)for p in paths]
    if raw is _K:out+=[SKIP()for _ in RAW_KEYS]
    elif raw is None:out+=[""]*len(RAW_KEYS)
    else:out+=[raw[k]for k in RAW_KEYS]
    return tuple(out)
def card_html(title,body,kind="info"):return f'<div class="kz card {kind}"><div class="h">{html.escape(title)}</div><div class="small">{body}</div></div>'
def pbar_html(done,total,color="#B45309"):
    pc=100.0*done/max(1,total);return f'<div style="margin-top:6px;background:#e2e8f0;border-radius:6px;height:14px"><div style="width:{pc:.1f}%;height:14px;border-radius:6px;background:{color}"></div></div><div class="small">{done}/{total}</div>'
def stage_card(e):
    body=html.escape(e["body"])
    if e.get("total"):body+=pbar_html(e["done"],e["total"],"#0369A1"if e["stage"]>=5 else"#B45309")
    if e.get("eta")is not None:body+=f'<div class="small">estimated time left: {int(e["eta"]//60)} min {int(e["eta"]%60)} s</div>'
    return card_html(f"{e['stage']}/{N_STAGES} · {e['title']}",body,"info")
READY_HTML=card_html("Ready",f"Press <b>RUN COGNITIVE CARTRIDGE DEMO</b>. The run forges 16 cartridges, removes the source, reads every question from all 16 cartridges one by one ({TOTAL_READS} generate calls), checks the controls, seals the evidence and renders {N_POSTERS} JPG posters plus one ZIP.<br><b>Downloads appear when the run is complete.</b>","info")
def is_cuda_error(ex):
    s=f"{type(ex).__name__} {ex}".lower();return"cuda"in s or"device-side assert"in s or"cublas"in s
def run_handler():
    global RUN_COUNTER
    if not GPU_LOCK.acquire(blocking=False):
        yield pack(card_html("Busy","Another run is in progress. Please try again in a moment.","warn"));return
    stage_name="initialisation"
    try:
        RUN_COUNTER+=1;run_id=f"MISTRAL-CC-{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}-{RUN_COUNTER:04d}"
        prune_runs(2);run_dir=ROOT/run_id;run_dir.mkdir(parents=True,exist_ok=True);gen=execute_run(ENGINE,dict(run_id=run_id,run_dir=run_dir));first_ev=True
        while True:
            try:e=next(gen)
            except StopIteration as s:B=s.value;break
            stage_name=f"{e['stage']}/{N_STAGES} {e['title']}"
            yield pack(stage_card(e),[],None,None,None)if first_ev else pack(stage_card(e));first_ev=False
        R=B["R"];C=R["counts"];T=R["totals"];ok=R["verdict"].startswith("PASS")
        gallery_items=[(str(p),c_)for p,c_ in B["imgs"]]
        raw_texts={"txt":B["txt"].read_text(encoding="utf-8"),"json":B["payload"].read_bytes().decode("utf-8"),"man":B["manifest"].read_text(encoding="utf-8")}
        body=(f"<b>{html.escape(run_id)}</b><br>THIS LIVE RUN: MW1 {C['mw1']}/{T['mw1']} · MW2 {C['mw2']}/{T['mw2']} · LINKED {C['linked']}/{T['linked']} · MISSING {C['missing']}/{T['missing']} · ABSENT-ID {C['absent_id']}/{T['absent_id']} · NOMEM {C['nomem']}/{T['nomem']}<br>"
              f"verdict by the TEST474 gates: <b>{R['verdict']}</b> · replay identical to the sealed counts: {'YES'if R['replay_match']else'NO'}<br>{N_POSTERS} JPGs + ZIP · {B['checks']}/{B['checks']} technical checks PASS<br>"
              f'<span class="mono">payload SHA-256 (artifact integrity seal): {B["sha"]}</span>')
        yield pack(card_html(f"{N_STAGES}/{N_STAGES} · Complete",body,"on"if ok else"err"),gallery_items,{"zip":B["zip"],"json":B["payload"],"txt":B["txt"],"man":B["manifest"]},[p for p,_ in B["imgs"]],raw_texts)
    except Exception as ex:
        print("="*110);print(f"MISTRAL RUN FAILED — stage: {stage_name}");print(f"exception type : {type(ex).__name__}");print(f"exception message: {ex}");traceback.print_exc();print("="*110)
        kind="AUDIT FAIL — no package was sealed. "if isinstance(ex,AuditFail)else"";partial=getattr(ex,"partial",None)
        if partial:kind+="The raw outputs gathered so far were preserved (FULL RUN LOG button / FULL RUN LOG tab). "
        tip="<br><b>CUDA error: Runtime → Restart runtime, then run the cell again.</b>"if is_cuda_error(ex)else"<br>The full traceback is printed in the Colab console."
        card=card_html("Run failed",f"{html.escape(kind)}Stage: {html.escape(stage_name)}<br>{html.escape(type(ex).__name__)}: {html.escape(str(ex)[:600])}{tip}","err")
        if partial:yield pack(card,[],{"json":partial},None,{"txt":traceback.format_exc(),"json":Path(partial).read_text(encoding="utf-8"),"man":""})
        else:yield pack(card,[],None,None,None)
    finally:
        plt.close("all");GPU_LOCK.release()
        try:torch.cuda.empty_cache()
        except Exception:pass
# ---------------- service self-test (runs before the public link is printed) ----------------
def service_selftest():
    global QUIET;out=[];d=ROOT/"_selftest";shutil.rmtree(d,ignore_errors=True);d.mkdir(parents=True);(d/"w.txt").write_text("ok");out.append("ROOT writable");QUIET=True
    try:
        B=drain(execute_run(SelfTestEngine("ok"),dict(run_id="SELFTEST",run_dir=d)))
        assert len(B["imgs"])==N_POSTERS and file_ready(B["zip"])and B["verdict"]=="PASS_MISTRAL_FINAL_ENGINE"
        out.append(f"full pipeline on a synthetic engine: {len(B['imgs'])}/{N_POSTERS} posters (JPEG/RGB), ZIP, {B['checks']} checks")
        d2=d/"fail";d2.mkdir()
        try:drain(execute_run(SelfTestEngine("sentinel_changes"),dict(run_id="SELFTEST-B",run_dir=d2)))
        except AuditFail:out.append("fail-closed: a changed weight sentinel aborts the run (no package sealed)")
        else:raise RuntimeError("self-test: a changed sentinel did not abort the run")
    finally:QUIET=False
    if len(pack(READY_HTML))!=N_OUTPUTS or len(pack(READY_HTML,[],None,None,None))!=N_OUTPUTS:raise RuntimeError("callback output count mismatch")
    out.append(f"callback output count = {N_OUTPUTS}");out.append(f"file route {FILE_URL_PREFIX}");shutil.rmtree(d,ignore_errors=True);return out
say("[4/7] SERVICE SELF-TEST")
for c_ in service_selftest():say(" PASS ·",c_)
# ---------------- interface ----------------
say("[5/7] INTERFACE")
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
HERO=('<div class="kz hero"><div class="brand">AKBASCORE NIRVANA</div><div class="sub">MISTRAL · COGNITIVE CARTRIDGE</div>'
      '<div class="tag">Knowledge goes in. The source goes away. The memory remains.<br>'
      f'16 synthetic knowledge cartridges are installed into a frozen {MODEL_SHORT}. The source text is removed. 16 locked questions are read from the cartridges — or answered UNKNOWN.</div></div>')
def _tb(**kw):
    try:return gr.Textbox(show_copy_button=True,**kw)
    except TypeError:return gr.Textbox(**kw)
def _gallery(**kw):
    try:return gr.Gallery(columns=1,height="auto",object_fit="contain",preview=False,**kw)
    except TypeError:return gr.Gallery(**kw)
try:_blocks=gr.Blocks(css=CSS,title="AKBASCORE NIRVANA · MISTRAL · COGNITIVE CARTRIDGE")
except TypeError:_blocks=gr.Blocks(title="AKBASCORE NIRVANA · MISTRAL · COGNITIVE CARTRIDGE")
with _blocks as demo:
    gr.HTML(HERO)
    run_btn=gr.Button("RUN COGNITIVE CARTRIDGE DEMO",variant="primary",elem_id="kz_run")
    status=gr.HTML(READY_HTML)
    gallery=_gallery(label=f"{N_POSTERS} JPG posters (tap to open)",show_label=True)
    dl_all=gr.Button(DL_ALL_LABEL,interactive=False)
    jpg_urls=gr.Textbox(value="",visible=False)
    with gr.Row():
        dlb=[gr.DownloadButton(label=l,value=None,interactive=False)for l in DL_LABELS]
    with gr.Accordion("Direct file links (fallback)",open=False):
        fls=[gr.File(label=n,interactive=False)for n in("ZIP · all posters and audit files","FULL RUN LOG (.json)","READABLE RUN LOG (.txt)","RUN MANIFEST (.json)")]
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
say("[6/7] LAUNCH (public share link)")
say("[7/7] Open the printed gradio.live link in a new tab and press RUN COGNITIVE CARTRIDGE DEMO.")
demo.launch(share=True,allowed_paths=ALLOWED_PATHS,show_error=True,inline=False)
