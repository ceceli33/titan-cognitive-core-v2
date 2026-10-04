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
