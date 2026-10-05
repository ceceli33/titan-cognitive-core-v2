# AKBASCORE NIRVANA · QWEN · 32-BANK COGNITIVE CARTRIDGE DEMO — SEALED TEST482 REPLAY (single Colab cell)
# CORE = TEST482 (482_qwen.final.py), unchanged: Qwen2.5-7B-Instruct, frozen, BF16/SDPA, K120/V128/OWN, fixed 32-sentence neutral PCA codebook,
#   32 records -> 64 independent cartridges (32 A: OBJECT->ID, 32 B: ID->PLACE), batched isolated read (one cartridge per batch row), FULL SCAN per stage
#   (Stage 1 scans only the 32 A cartridges, Stage 2 only the 32 B cartridges), MW1, Stage-2 readout C_COMPACT, NONE-first parser, unique aggregation, greedy.
#   NO router · NO index · NO signature · NO training · NO LoRA · NO gradient · NO new memory/retrieval mechanism.
# REPLAY ONLY: the 32 TEST482 held-out records are replayed unchanged; no new panel, no fresh seed.
# DEMO LAYER (measurement/presentation only): telemetry from tensors after the core produced them, raw-output recording, source-removal audit,
#   result re-derivation from raw text, live Clopper-Pearson intervals, poster rendering, stale-reference scan, sealing, ZIP.
# DEMO-ONLY CONTROL: NOMEM (PAD-only cache) is NOT part of the TEST482 sealed result and is labelled so everywhere.
# Result classes: LIVE (measured in this run) · SEALED (historical log record) · DEMO-ONLY · DERIVED — never merged.
import os,sys,io,re,json,math,time,html,random,shutil,hashlib,zipfile,platform,textwrap,threading,subprocess,importlib.util,traceback,gc
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import quote
from collections import Counter
#<<CONST_BEGIN>>
TEST="482";SEED=461;MODEL_ID="Qwen/Qwen2.5-7B-Instruct";MODEL_SHORT="Qwen2.5-7B-Instruct"
K_DIM=120;V_DIM=128;MAX_NEW=32;ARCH=(28,3584,28,4,128)
FMT="QUESTION:\n{q}\n\n ANSWER:";SEP="\n\n"
STYLE=" Give only the answer on the first line; do not explain."
MW1='Which container contains the {obj} according to this record? Give only the container identifier. If this record does not contain the {obj}, answer NONE.'+STYLE
MW2='Container "{cid}" location? Answer only the location if stated; otherwise answer NONE.'
CORPUS=["Jonas Weber carried the wooden crate across the quiet market square.","Priya Nair wrote a long letter to her cousin in the morning.","The small boat drifted slowly toward the rocky shore.","Omar Haddad fixed the broken clock in the village school.","A tired teacher closed the green door of the library.","Sofia Rossi baked fresh bread for the harvest festival.","The children watched the kites rising above the hill.","Liam O'Connor sold his old bicycle to a neighbor.","Heavy rain flooded the narrow street near the market.","Nadia Petrova translated the ancient manuscript into French.","The farmer counted the sheep before sunset.","Hiro Sato opened a tiny bakery beside the river.","An old dog slept under the kitchen table all afternoon.","Carlos Mendes washed the windows of his grandmother's house.","The museum guard locked the heavy gate at midnight.","Fatima Zahra grew tomatoes in the backyard garden.","The pilot announced a short delay because of fog.","Anna Kowalski found a lost wallet on the bus.","Snow covered the mountain village during the night.","Ravi Kumar taught his brother how to play chess.","The chef sharpened every knife before dinner service.","Lucas Martin cleaned the roof of the barn after the storm.","A young violinist practiced scales in the empty hall.","Mei Lin delivered the package to the wrong address.","Marek Novak placed the glass bottle beneath the wooden bench.","Sara Ibrahim carried a red notebook into the quiet classroom.","Noah Schmidt repaired the small radio beside the kitchen window.","Yuki Mori left a paper envelope near the station entrance.","Amira Hassan moved the ceramic bowl onto the upper shelf.","Peter Novak opened the metal box behind the old theater.","Lucia Costa placed the yellow scarf inside the travel bag.","Daniel Kim carried a black umbrella through the central courtyard."]
RECORDS=[("F1","emerald barometer","KR-214","cedar archive"),("F1","porcelain compass","DM-763","marble annex"),("F1","violet harmonica","FS-408","amber gallery"),("F1","bamboo chronometer","JN-951","fern vault"),("F1","obsidian sextant","RW-326","copper archive"),("F1","linen telescope","AE-875","quartz annex"),("F1","coral metronome","LP-143","ivory gallery"),("F1","willow calculator","XM-692","silver vault"),
("F2","turquoise monocle","HC-517","indigo lighthouse"),("F2","mahogany notebook","VK-280","pearl pavilion"),("F2","ceramic astrolabe","SB-934","granite lodge"),("F2","scarlet blueprint","NT-461","willow gallery"),("F2","opal kaleidoscope","GY-708","maple tower"),("F2","bronze hourglass","PD-352","coral pavilion"),("F2","silk manuscript","ZU-819","birch lodge"),("F2","crystal chime","EF-625","onyx gallery"),
("F3","jade projector","WL-407","hazel chamber"),("F3","glass clarinet","CQ-586","lagoon studio"),("F3","canvas diary","MR-172","acorn room"),("F3","iron medallion","UX-943","poplar hall"),("F3","pearl camera","BH-650","cobalt chamber"),("F3","steel accordion","KO-238","canal studio"),("F3","cotton almanac","YD-714","walnut room"),("F3","onyx brooch","TG-569","elm hall"),
("F4","ivory receiver","PS-381","summit workshop"),("F4","cedar plaque","AL-826","valley depot"),("F4","azure kettle","RF-504","birch observatory"),("F4","brass caliper","MW-197","shore conservatory"),("F4","crystal phonograph","DE-648","forest workshop"),("F4","maple slate","KI-275","delta depot"),("F4","white lantern","OV-930","clover observatory"),("F4","golden protractor","XC-412","island conservatory")]
def sources(fam,obj,cid,place):
    if fam=="F1": return f"The {obj} is stored in container {cid}.",f"Container {cid} is located in the {place}."
    if fam=="F2": return f"Container {cid} contains the {obj}.",f"The {place} houses container {cid}."
    if fam=="F3": return f"The {obj} can be found inside container {cid}.",f"The location of container {cid} is the {place}."
    if fam=="F4": return f"Inside container {cid} there is the {obj}.",f"Container {cid} can be found at the {place}."
    raise ValueError(fam)
ABSENT_OBJECTS=["yellow microscope","paper accordion","granite whistle","violet telescope"]
ABSENT_IDS=["ZZ-999","AA-000","LM-777","RX-111"]
LOCK={"model":MODEL_ID,"k_dim":K_DIM,"v_dim":V_DIM,"max_new":MAX_NEW,"fmt":FMT,"sep":SEP,"mw1":MW1,"mw2":MW2,"records":RECORDS,"corpus":CORPUS}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()
SEALED_LOCK_SHA="3ed21b24d6a9fd823dc60dffdcb3e83dfe7271f057fd694739031c25c41d34ff"
CANON_GPU="NVIDIA A100-SXM4-40GB"
def norm(s): return re.sub(r"[^\w-]+"," ",str(s).casefold()).strip()
def firstline(text): return next((x.strip() for x in str(text).splitlines() if x.strip()),"")
def is_none(text):
    n=norm(firstline(text))
    return n=="none" or n.startswith("none ")
VALID_IDS=[x[2] for x in RECORDS];VALID_PLACES=[x[3] for x in RECORDS]
def parse_id(text):
    line=firstline(text)
    if is_none(line): return None
    n=norm(line);hits=[x for x in VALID_IDS if norm(x) in n]
    return hits[0] if len(set(hits))==1 else None
def parse_place(text):
    line=firstline(text)
    if is_none(line): return None
    n=norm(line);hits=[x for x in VALID_PLACES if norm(x) in n]
    return hits[0] if len(set(hits))==1 else None
def decide(vals):
    u=sorted(set(x for x in vals if x is not None))
    return u[0] if len(u)==1 else None
FAMS=("F1","F2","F3","F4");NREC=32
SRC_A=[sources(*r)[0] for r in RECORDS];SRC_B=[sources(*r)[1] for r in RECORDS];SRC_ALL=SRC_A+SRC_B
EXT_LOCK_SHA=hashlib.sha256(json.dumps({"lock_sha":LOCK_SHA,"absent_objects":ABSENT_OBJECTS,"absent_ids":ABSENT_IDS,"sources":SRC_ALL,"stage_banks":"S1=A32 S2=B32"},sort_keys=True,separators=(",",":")).encode()).hexdigest()
# ---- SEALED HISTORICAL RECORDS (every number below was read from the named log; none is produced by the live run) ----
SEALED482=dict(log="482.log",s1=(32,32),linked=(32,32),s1_fp=(0,992),s2_fp=(0,992),s1_none=(992,992),s2_none=(992,992),s1_coll=(0,32),s2_coll=(0,32),obj=(128,128),idn=(128,128),
    fam={f:((8,8),(8,8))for f in FAMS},forge_s=3.84,s1_s=41.29,s2_s=39.12,mean_s=2.513,total_s=160.24,verdict="PASS_FINAL_HELDOUT_32_BANK_SEAL",gpu=CANON_GPU,lock=SEALED_LOCK_SHA)
SEALED_CHAIN=[
 dict(t="477",role="DEVELOPMENT",log="477.log",what="32-record core scaling · Stage-2 readout VERIFY",s1=(32,32),linked=(31,32),verdict="FAIL_SCALE_READOUT_OR_RETRIEVAL",note="F3 linked 7/8; only failure: glass trumpet → PL-570 → harbor studio (Stage 1 correct, Stage 2 returned None). Off-target FP 0/992 + 0/992, NONE 992/992 + 992/992, collisions 0/32 + 0/32, unrelated 256/256.",secs=149.13),
 dict(t="478",role="DEVELOPMENT",log="478.log",what="single-failure isolation (n=1 cartridge) · Stage-2 wording A/B/C",s1=None,linked=(1,1),verdict="DIRECT_LOCATION_CAUSAL_CANDIDATE",note="A current VERIFY FAIL · B relation VERIFY FAIL · C direct location PASS. One cartridge, one trial: a candidate, not an authorization.",secs=None),
 dict(t="479",role="DEVELOPMENT",log="479.log",what="DIRECT_LOCATION on the full 32-bank",s1=(32,32),linked=(31,32),verdict="FAIL_REMAINING_GOLD_READOUT",note="PL-570 repaired; new failure F2 crimson telescope → RQ-415 → saffron lighthouse (raw output began with NONE). Off-target 0/992 + 0/992, NONE 992/992 + 992/992.",secs=148.73),
 dict(t="480",role="DEVELOPMENT",log="480.log",what="five Stage-2 wordings on four isolated cartridges (n=4)",s1=None,linked=None,verdict="MULTIPLE_UNIVERSAL_CANDIDATES",note="C_COMPACT and D_RECORD strict PASS; A, B, E FAIL.",secs=98.23),
 dict(t="481",role="DEVELOPMENT",log="481.log",what="C_COMPACT vs D_RECORD on the complete development bank",s1=(32,32),linked=(32,32),verdict="BOTH_UNIVERSAL_READOUTS_PASS_32_BANK",note="Both 32/32, off-target FP 0/992, NONE 992/992, collisions 0/32, absent-ID 128/128. C_COMPACT frozen before the held-out test.",secs=184.01),
 dict(t="482",role="HELD-OUT",log="482.log",what="32 new records · frozen readout · full scan",s1=(32,32),linked=(32,32),verdict="PASS_FINAL_HELDOUT_32_BANK_SEAL",note="Same four source templates as development; new objects, IDs and places.",secs=160.24)]
SEALED480=dict(cands=[("A_VERIFY",'Does this memory explicitly contain container "{cid}"? If yes, answer only its location. If no, answer exactly NONE.',3,0,12,0,16,False),
 ("B_DIRECT",'What location does this memory state for container "{cid}"? If no location is stated, answer exactly NONE.',3,0,12,0,16,False),
 ("C_COMPACT",'Container "{cid}" location? Answer only the location if stated; otherwise answer NONE.',4,0,12,0,16,True),
 ("D_RECORD",'According to this record, where is container "{cid}" located? Answer only the location, or NONE if unknown.',4,0,12,0,16,True),
 ("E_MINIMAL",'Location of container "{cid}":',2,2,0,2,0,False)])
SEALED461=dict(log="463.log",linked=(43,48),unlinked=(48,48),false_link=(0,48),fam_min=("F2",(7,12)),verdict="FAIL_FINAL_HELDOUT")
# ---- exact binomial (Clopper-Pearson) interval, computed at run time in pure Python ----
def _lpmf(i,n,p):return math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)+i*math.log(p)+(n-i)*math.log1p(-p)
def _tail(lo,hi,n,p):
    ls=[_lpmf(i,n,p)for i in range(lo,hi+1)]
    if not ls:return 0.0
    m=max(ls);return math.exp(m)*sum(math.exp(x-m)for x in ls)
def cp_interval(k,n,alpha=0.05):
    a=alpha/2
    if n<=0:return(0.0,1.0)
    def bis(f,incr):
        l,h=1e-12,1-1e-12
        for _ in range(100):
            m=(l+h)/2
            if (f(m)>=a)==incr:h=m
            else:l=m
        return(l+h)/2
    lo=0.0 if k<=0 else bis(lambda p:_tail(k,n,n,p),True)
    hi=1.0 if k>=n else bis(lambda p:_tail(0,k,n,p),False)
    return(lo,hi)
def pc(x):return f"{100*x:.2f}%"
def ci_text(k,n):
    lo,hi=cp_interval(k,n);return f"{k}/{n} · 95% CI [{pc(lo)}, {pc(hi)}]"
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
def fr(t):return f"{t[0]}/{t[1]}"
def derive(raw):
    """Re-derive every result from the raw generated texts with the verbatim TEST482 parsers and decision rules."""
    s1=[];S1_FP=S1_NONE=S1_COLL=0
    for i,(fam,obj,gid,gpl)in enumerate(RECORDS):
        rows=raw["s1"][i]["rows"];parsed=[parse_id(x)for x in rows];pred=decide(parsed);gold=parsed[i];ok=pred==gid
        wrong=[x for j,x in enumerate(parsed)if j!=i and x is not None];S1_FP+=len(wrong);S1_NONE+=sum(is_none(x)for j,x in enumerate(rows)if j!=i)
        coll=int(pred is None and len(set(x for x in parsed if x is not None))>1);S1_COLL+=coll
        row=["gold"if(j==i and parsed[j]==gid)else"gold_miss"if j==i else("none"if is_none(x)else"other")for j,x in enumerate(rows)]
        s1.append(dict(fam=fam,pred=pred,gold=gold,ok=ok,fp=len(wrong),coll=coll,cls=row,nrows=len(rows)))
    s2=[];S2_FP=S2_NONE=S2_COLL=0;fallbacks=0
    for i,(fam,obj,gid,gpl)in enumerate(RECORDS):
        rows=raw["s2"][i]["rows"];pid=s1[i]["pred"];use=pid if pid is not None else gid;fb=pid is None;fallbacks+=int(fb)
        if raw["s2"][i]["cid"]!=use:raise AuditFail(f"Stage 2 record {i} used {raw['s2'][i]['cid']} but TEST482 rule requires {use}")
        parsed=[parse_place(x)for x in rows];pred=decide(parsed);gold=parsed[i];linked=s1[i]["ok"]and pred==gpl
        wrong=[x for j,x in enumerate(parsed)if j!=i and x is not None];S2_FP+=len(wrong);S2_NONE+=sum(is_none(x)for j,x in enumerate(rows)if j!=i)
        coll=int(pred is None and len(set(x for x in parsed if x is not None))>1);S2_COLL+=coll
        row=["gold"if(j==i and parsed[j]==gpl)else"gold_miss"if j==i else("none"if is_none(x)else"other")for j,x in enumerate(rows)]
        s2.append(dict(fam=fam,pred=pred,gold=gold,linked=linked,fp=len(wrong),coll=coll,cls=row,fallback=fb,cid=use))
    c_obj=[];c_id=[]
    for e in raw["c_obj"]:c_obj.append(dict(q=e["q"],none=sum(is_none(x)for x in e["rows"]),n=len(e["rows"])))
    for e in raw["c_id"]:c_id.append(dict(q=e["q"],none=sum(is_none(x)for x in e["rows"]),n=len(e["rows"])))
    OBJ_NONE=sum(c["none"]for c in c_obj);OBJ_TOTAL=sum(c["n"]for c in c_obj);ID_NONE=sum(c["none"]for c in c_id);ID_TOTAL=sum(c["n"]for c in c_id)
    S1=sum(int(x["ok"])for x in s1);LINKED=sum(int(x["linked"])for x in s2);OFF=NREC*(NREC-1)
    fam={f:(sum(int(s1[i]["ok"])for i in range(NREC)if RECORDS[i][0]==f),sum(int(s2[i]["linked"])for i in range(NREC)if RECORDS[i][0]==f))for f in FAMS}
    strict=bool(S1==32 and LINKED==32 and S1_FP==0 and S2_FP==0 and S1_NONE==OFF and S2_NONE==OFF and S1_COLL==0 and S2_COLL==0 and OBJ_NONE==OBJ_TOTAL and ID_NONE==ID_TOTAL)
    nm1=[dict(first=firstline(x),parsed=parse_id(x),hit=bool(parse_id(x)==RECORDS[i][2]))for i,x in enumerate(raw["nm1"])]
    nm2=[dict(first=firstline(x),parsed=parse_place(x),hit=bool(parse_place(x)==RECORDS[i][3]))for i,x in enumerate(raw["nm2"])]
    cnt=dict(s1=S1,linked=LINKED,s1_fp=S1_FP,s2_fp=S2_FP,s1_none=S1_NONE,s2_none=S2_NONE,s1_coll=S1_COLL,s2_coll=S2_COLL,obj_none=OBJ_NONE,id_none=ID_NONE)
    sv=SEALED482;rm=bool(S1==sv["s1"][0]and LINKED==sv["linked"][0]and S1_FP==sv["s1_fp"][0]and S2_FP==sv["s2_fp"][0]and S1_NONE==sv["s1_none"][0]and S2_NONE==sv["s2_none"][0]and S1_COLL==0 and S2_COLL==0 and OBJ_NONE==128 and ID_NONE==128 and OBJ_TOTAL==128 and ID_TOTAL==128)
    return dict(counts=cnt,off_total=OFF,obj_total=OBJ_TOTAL,id_total=ID_TOTAL,strict=strict,verdict="PASS_REPLAY_OF_TEST482"if strict else"FAIL_REPLAY_OF_TEST482",fam=fam,s1=s1,s2=s2,c_obj=c_obj,c_id=c_id,
        fallbacks=fallbacks,replay_match=rm,nomem=dict(s1_hits=sum(int(x["hit"])for x in nm1),s2_hits=sum(int(x["hit"])for x in nm2),s1=nm1,s2=nm2,label="DEMO-ONLY / NOT PART OF TEST482 SEALED RESULT"),
        rows_per_read=sorted({len(e["rows"])for e in raw["s1"]}|{len(e["rows"])for e in raw["s2"]}))
# ---- stale-reference scanner: scans GENERATED output (poster text, JSON, TXT, file names), never its own source ----
ALLOWED_TESTS={"461","477","478","479","480","481","482"}
STALE=[("other-model",r"[Mm]istral"),("other-arch-layers",r"\b32[ -]?(?:transformer )?(?:layers?|L)\b"),("other-arch-hidden",r"(?i:hidden[^0-9]{0,15}4096|\bH\s?=?\s?4096\b)"),("other-arch-kv",r"\b8\s?KV\b"),
       ("other-D",r"V\s?120\b|D120"),("old-doi",r"zenodo|10\.5281"),("old-batched-name",r"Isolated Batched")]
def stale_scan(name,text,allow=()):
    for a in allow:
        if a:text=text.replace(a,"<allowed>")
    hits=[(name,lab,m.group(0))for lab,p in STALE for m in re.finditer(p,text)]
    hits+=[(name,"old-test-number",m.group(0))for m in re.finditer(r"TEST\s?(\d+)",text)if m.group(1)not in ALLOWED_TESTS]
    return hits
def synth_raw(bad=False):
    raw=dict(s1=[],s2=[],c_obj=[],c_id=[],nm1=[],nm2=[])
    for i,(f,o,gid,gpl)in enumerate(RECORDS):
        raw["s1"].append(dict(q=i,rows=[gid if j==i else"NONE\nx"for j in range(32)]))
        raw["s2"].append(dict(q=i,cid=gid,rows=[("The "+gpl+".")if(j==i and not(bad and i==9))else("NONE\n"+gpl if(j==i)else"NONE")for j in range(32)]))
        raw["nm1"].append("NONE");raw["nm2"].append("NONE")
    for q in range(4):raw["c_obj"].append(dict(q=q,rows=["NONE"]*32));raw["c_id"].append(dict(q=q,rows=["NONE\ny"]*32))
    return raw
def cpu_selftest():
    n=0
    assert LOCK_SHA==SEALED_LOCK_SHA,"LOCK SHA differs from the sealed TEST482 value";n+=1
    for t,w in[("NONE",True),("none",True),("None of the records mention it",True),("NONE\nThe given statement says cedar archive",True),("Nonexistent annex",False),("",False),("cedar archive",False)]:assert is_none(t)==w,(t,is_none(t));n+=1
    assert parse_id("KR-214")=="KR-214" and parse_id("It is KR-214.")=="KR-214" and parse_id("NONE\nKR-214")is None and parse_id("KR-214 or DM-763")is None and parse_id("ZZ-999")is None;n+=5
    assert parse_place("The cedar archive.")=="cedar archive" and parse_place("NONE")is None and parse_place("cedar archive and marble annex")is None;n+=3
    assert decide([None,"a",None])=="a" and decide(["a","b"])is None and decide([None])is None;n+=3
    ids=[r[2]for r in RECORDS];pl=[r[3]for r in RECORDS];ob=[r[1]for r in RECORDS]
    assert len(set(ids))==32 and len(set(pl))==32 and len(set(ob))==32 and Counter(r[0]for r in RECORDS)==Counter({f:8 for f in FAMS});n+=4
    assert not[(a,b)for a in map(norm,ids+pl)for b in map(norm,ids+pl)if a!=b and a in b];n+=1
    for i,r in enumerate(RECORDS):
        a,b=sources(*r);assert r[1]in a and r[2]in a and r[2]in b and r[3]in b;n+=1
        for q in(MW1.format(obj=r[1]),MW2.format(cid=r[2])):assert not any(s in FMT.format(q=q)for s in SRC_ALL);n+=1
    assert len(set(SRC_ALL))==64 and not set(ABSENT_IDS)&set(ids);n+=2
    R=derive(synth_raw());assert R["strict"]and R["replay_match"]and R["fallbacks"]==0 and R["counts"]["s1_none"]==992 and R["nomem"]["s1_hits"]==0;n+=4
    R=derive(synth_raw(True));assert not R["strict"]and R["counts"]["linked"]==31 and R["fam"]["F2"]==(8,7)and not R["replay_match"];n+=3
    r2=synth_raw();r2["s1"][3]["rows"]=["NONE\nx"]*32;R=derive(r2);assert R["fallbacks"]==1 and R["s1"][3]["ok"]is False and R["s2"][3]["linked"]is False;n+=3
    lo,hi=cp_interval(32,32);assert abs(lo-0.025**(1/32))<1e-6 and hi==1.0;lo,hi=cp_interval(0,992);assert lo==0.0 and abs(hi-(1-0.025**(1/992)))<1e-6;n+=2
    lo,hi=cp_interval(7,12);assert 0.27<lo<0.30 and 0.83<hi<0.86;n+=1
    assert stale_scan("t","Qwen2.5-7B-Instruct 28 layers TEST482 TEST477 TEST461 F1 F2 K120/V128 NVIDIA A100-SXM4-40GB 32 A cartridges 992 = 32x31")==[];n+=1
    for bad in("Mistral-7B","32 layers","hidden 4096","D120","TEST474","TEST460","8 KV heads","zenodo"):assert stale_scan("t",bad),bad;n+=1
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
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
DEVICE=torch.device("cuda")
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
STARTUP_UTC=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
ROOT=Path("/content/AKBASCORE_QWEN32_CARTRIDGE")if os.path.isdir("/content")else Path("/tmp/AKBASCORE_QWEN32_CARTRIDGE")
ROOT.mkdir(parents=True,exist_ok=True);ALLOWED_PATHS=[str(ROOT)]
def utc_now():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def local_now():return datetime.now().astimezone().isoformat(timespec="milliseconds")
QUIET=False
def say(*a):
    if not QUIET:print(*a,flush=True)
say("="*140);say("AKBASCORE NIRVANA · QWEN · 32-BANK COGNITIVE CARTRIDGE DEMO · SEALED TEST482 REPLAY");say("="*140)
say(f"[0/7] CPU self-test PASS ({cpu_selftest()} checks) · TEST482 LOCK SHA recomputed = sealed value");say("LOCK SHA :",LOCK_SHA);say("START UTC :",STARTUP_UTC)
say("Model :",MODEL_ID);say("GPU :",torch.cuda.get_device_name(0),"| canonical sealed hardware:",CANON_GPU)
say("Gradio :",gr.__version__,"| Transformers:",transformers.__version__,"| Torch:",torch.__version__)
say("[1/7] MODEL LOAD")
_tv=tuple(int("".join(c for c in x if c.isdigit())or 0)for x in transformers.__version__.split(".")[:2]);DT="dtype"if _tv>=(4,56)else"torch_dtype"
torch.cuda.synchronize();_t=time.perf_counter()
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}).eval()
for p in model.parameters():p.requires_grad_(False)
torch.cuda.synchronize();MODEL_LOAD_S=time.perf_counter()-_t
cfg=model.config;layers=model.model.layers
H=cfg.hidden_size;NH=cfg.num_attention_heads;NKV=cfg.num_key_value_heads
HD=getattr(cfg,"head_dim",None)or H//NH;KVD=NKV*HD;NL=len(layers)
if(NL,H,NH,NKV,HD)!=ARCH:raise RuntimeError(f"Architecture mismatch: {(NL,H,NH,NKV,HD)}")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
_ge=model.generation_config.eos_token_id
EOS=sorted({tok.eos_token_id,*(_ge if isinstance(_ge,(list,tuple))else[_ge])}-{None})
enc=lambda s:tok(s,add_special_tokens=False).input_ids
FP=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[13].mlp.down_proj.weight,layers[23].self_attn.o_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
FP_NAMES=["layers.0.self_attn.q_proj","layers.8.self_attn.o_proj","layers.13.mlp.down_proj","layers.23.self_attn.o_proj","layers.27.mlp.down_proj","model.norm","lm_head"]
# ---------------- TEST482 core functions (verbatim) ----------------
@torch.inference_mode()
def kv_from_ids(ids):
    o=model(input_ids=torch.tensor([ids],device=DEVICE),output_hidden_states=True,use_cache=False,return_dict=True)
    K=[];V=[]
    for L in range(NL):
        z=layers[L].input_layernorm(o.hidden_states[L][0]);a=layers[L].self_attn
        K.append(a.k_proj(z).contiguous());V.append(a.v_proj(z).contiguous())
    return K,V
def forge(s): return kv_from_ids([PAD]+enc(s+SEP))
@torch.inference_mode()
def install(K,V):
    T=K[0].shape[0]
    cos,sin=model.model.rotary_emb(K[0][None],torch.arange(T,device=DEVICE)[None])
    KK=[];VV=[]
    for L in range(NL):
        k=K[L].reshape(1,T,NKV,HD).transpose(1,2)
        v=V[L].reshape(1,T,NKV,HD).transpose(1,2)
        KK.append(apply_rotary_pos_emb(k,k,cos,sin,unsqueeze_dim=1)[1].contiguous())
        VV.append(v.contiguous())
    return tuple(KK),tuple(VV),T
CB=[]
@torch.inference_mode()
def build_codebook():
    global CB
    t0=time.perf_counter();CB=[];CO=[forge(s) for s in CORPUS]
    for L in range(NL):
        e={}
        for j,n in enumerate(("K","V")):
            R=torch.cat([c[j][L][1:] for c in CO]).float().view(-1,NKV,HD);MU=[];BB=[]
            for h in range(NKV):
                X=R[:,h];mu=X.mean(0);_,S,Vh=torch.linalg.svd(X-mu,full_matrices=False)
                m=min(128,Vh.shape[0]);b=Vh[:m].T.contiguous()
                if m<128: b=torch.nn.functional.pad(b,(0,128-m))
                MU.append(mu);BB.append(b)
            e[n]=(torch.stack(MU),torch.stack(BB))
        CB.append(e)
    del CO;torch.cuda.empty_cache();torch.cuda.synchronize()
    return dict(sentences=len(CORPUS),svds=NL*2*NKV,seconds=time.perf_counter()-t0,bytes=sum(t.numel()*t.element_size()for e in CB for n in("K","V")for t in e[n]))
@torch.inference_mode()
def packet(source):
    K,V=forge(source);out={}
    for n,X,d in (("K",K,K_DIM),("V",V,V_DIM)):
        rows=[]
        for L in range(NL):
            mu,B=CB[L][n]
            coeff=torch.einsum("thi,hid->thd",X[L][1:].float().view(-1,NKV,HD)-mu,B[:,:,:d]).to(torch.bfloat16)
            content=(mu+torch.einsum("thd,hid->thi",coeff.float(),B[:,:,:d])).reshape(-1,KVD).to(torch.bfloat16)
            rows.append(torch.cat([X[L][:1],content]))
        out[n]=rows
    return out["K"],out["V"]
@torch.inference_mode()
def batch(q,kvs):
    qids=enc(FMT.format(q=q));BN=len(kvs);Tm=max(kv[2] for kv in kvs);nq=len(qids)
    try: cache=DynamicCache(config=cfg)
    except TypeError: cache=DynamicCache()
    for L in range(NL):
        Ks=[];Vs=[]
        for kv in kvs:
            k,v,p=kv[0][L],kv[1][L],Tm-kv[2]
            if p:
                k=torch.cat([k.new_zeros(1,NKV,p,HD),k],2)
                v=torch.cat([v.new_zeros(1,NKV,p,HD),v],2)
            Ks.append(k);Vs.append(v)
        cache.update(torch.cat(Ks).clone(),torch.cat(Vs).clone(),L)
    mask=torch.zeros(BN,Tm+nq,dtype=torch.long,device=DEVICE)
    for b,kv in enumerate(kvs): mask[b,Tm-kv[2]:]=1
    pos=torch.tensor([kv[2] for kv in kvs],device=DEVICE)[:,None]+torch.arange(nq,device=DEVICE)[None]
    ids=torch.tensor([qids]*BN,device=DEVICE);outs=[[] for _ in range(BN)];done=[False]*BN
    for _ in range(MAX_NEW):
        o=model(input_ids=ids,attention_mask=mask,position_ids=pos,past_key_values=cache,use_cache=True,return_dict=True)
        nxt=o.logits[:,-1].float().argmax(-1).tolist()
        for b,t in enumerate(nxt):
            if not done[b]:
                if t in EOS: done[b]=True
                else: outs[b].append(t)
        if all(done): break
        feed=[EOS[0] if done[b] and EOS else nxt[b] for b in range(BN)]
        ids=torch.tensor([[t] for t in feed],device=DEVICE);pos=pos[:,-1:]+1
        mask=torch.cat([mask,torch.ones(BN,1,dtype=torch.long,device=DEVICE)],1)
    return [tok.decode(x,skip_special_tokens=True).strip() for x in outs]
# ---------------- demo-side read-only measurements (never alter cartridges) ----------------
def sentinel():
    h=hashlib.sha256()
    for p in FP:
        a=p.detach().reshape(-1);n=a.numel()
        for j in range(16):
            o=j*(n-256)//15;h.update(a[o:o+256].float().cpu().numpy().tobytes())
    return h.hexdigest()
def fingerprint():return tuple(float(t.sum(dtype=torch.float32))for t in FP)
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
def lora_present():return hasattr(model,"peft_config")or any("lora"in n.lower()for n,_ in model.named_modules())
def trainable_tensors():return sum(int(p.requires_grad)for p in model.parameters())
def optimizer_present():return any(isinstance(v,torch.optim.Optimizer)for v in list(globals().values()))
@torch.inference_mode()
def card_telemetry(source,Kc,Vc,excerpt):
    K0,V0=forge(source);cosK=[];cosV=[]
    for L in range(NL):
        cosK.append(float(torch.nn.functional.cosine_similarity(Kc[L][1:].float().reshape(-1),K0[L][1:].float().reshape(-1),0)))
        cosV.append(float(torch.nn.functional.cosine_similarity(Vc[L][1:].float().reshape(-1),V0[L][1:].float().reshape(-1),0)))
    own=all(torch.equal(Kc[L][:1],K0[L][:1])and torch.equal(Vc[L][:1],V0[L][:1])for L in range(NL));T=Kc[0].shape[0]
    tel=dict(T=T,cosK=cosK,cosV=cosV,own_exact=bool(own),code_numbers=NL*(T-1)*NKV*(K_DIM+V_DIM),own_numbers=2*NL*KVD,native_numbers=2*NL*KVD*T)
    if excerpt:
        mu,B=CB[14]["K"];x=K0[14][1:].float().reshape(-1,NKV,HD);cc=torch.einsum("thi,hid->thd",x-mu,B[:,:,:K_DIM]).to(torch.bfloat16)
        tel["excerpt"]=dict(layer=14,head=0,tensor="K",values=[[round(v,4)for v in row]for row in cc[:,0,:].float().cpu().tolist()])
    return tel
class Engine:
    def __init__(self):
        self.counters=Counter();self.A=[];self.B=[];self.nm=None
        self.info=dict(model_id=MODEL_ID,arch=(NL,H,NH,NKV,HD),dtype="bfloat16",attn=getattr(cfg,"_attn_implementation",None),gpu=torch.cuda.get_device_name(0),canonical_gpu=CANON_GPU,
            gpu_total_gib=torch.cuda.get_device_properties(0).total_memory/2**30,torch=torch.__version__,transformers=transformers.__version__,gradio=gr.__version__,python=sys.version.split()[0],
            platform=platform.platform(),params=sum(p.numel()for p in model.parameters()),pad_id=PAD,eos_ids=EOS,fp_names=FP_NAMES,model_load_seconds=MODEL_LOAD_S,startup_utc=STARTUP_UTC,
            sentinel0=sentinel(),fingerprint0=list(fingerprint()),hooks0=hook_stats()[0],cell_source_sha256=CELL_SOURCE_SHA,sentinel_method="sampled SHA-256: 7 tensors × 16 slices × 256 values")
        if sentinel()!=self.info["sentinel0"]:raise RuntimeError("Sentinel is not repeatable")
    def frozen_state(self):
        tot,fo=hook_stats();return dict(training=bool(model.training),trainable_tensors=trainable_tensors(),lora=lora_present(),optimizer=optimizer_present(),hooks=tot,foreign_hooks=sum(fo.values()),foreign_modules=fo,sentinel=sentinel(),fingerprint=list(fingerprint()))
    def build_codebook(self):self.counters["core_forge_passes"]+=len(CORPUS);return build_codebook()
    def codebook_bytes(self):return sum(t.numel()*t.element_size()for e in CB for n in("K","V")for t in e[n])
    def forge_one(self,i):
        fam,obj,cid,place=RECORDS[i];sa,sb=sources(fam,obj,cid,place)
        torch.cuda.synchronize();t0=time.perf_counter();KA,VA=packet(sa);KB,VB=packet(sb);torch.cuda.synchronize();dt_=time.perf_counter()-t0
        self.counters["core_forge_passes"]+=2;tA=card_telemetry(sa,KA,VA,i==0);tB=card_telemetry(sb,KB,VB,False);self.counters["telemetry_reforge_passes"]+=2
        self.A.append(install(KA,VA));self.B.append(install(KB,VB));del KA,VA,KB,VB
        return dict(A=tA,B=tB,secs=dt_)
    def make_nomem(self):K,V=kv_from_ids([PAD]);self.nm=install(K,V)
    def card_ptrs(self):return[x for kv in self.A+self.B for x in(kv[0][0].data_ptr(),kv[1][0].data_ptr())]
    def card_finite(self):return all(bool(torch.isfinite(t).all())for kv in self.A+self.B for t in kv[0]+kv[1])
    def q_tokens(self,p):return len(enc(FMT.format(q=p)))
    def read_A(self,q):self.counters["batch_calls"]+=1;return batch(q,self.A)
    def read_B(self,q):self.counters["batch_calls"]+=1;return batch(q,self.B)
    def read_NM(self,q):self.counters["batch_calls"]+=1;return batch(q,[self.nm])
    def sync(self):torch.cuda.synchronize()
    def reset_peak(self):torch.cuda.reset_peak_memory_stats()
    def peak_gib(self):return torch.cuda.max_memory_allocated()/2**30
# ===== END PART 1 / 4 — CONTINUE WITH PART 2 =====
ENGINE=Engine()
say(f"Model loaded in {MODEL_LOAD_S:.2f}s | layers={NL} hidden={H} Q={NH} KV={NKV} head={HD} | params={ENGINE.info['params']:,} trainable={trainable_tensors()}")
say("[2/7] WEIGHT SENTINELS");say("Fingerprint:",[f"{x:.4f}"for x in ENGINE.info["fingerprint0"]]);say("SHA-256 sampled sentinel:",ENGINE.info["sentinel0"])
say("[3/7] ENGINE LOCK")
ENGINE_LOCK=[("MODEL_ID = Qwen/Qwen2.5-7B-Instruct",MODEL_ID=="Qwen/Qwen2.5-7B-Instruct"),("architecture 28 layers / hidden 3584 / 28 Q / 4 KV / head 128",(NL,H,NH,NKV,HD)==(28,3584,28,4,128)),
 ("K120 / V128 / OWN",(K_DIM,V_DIM)==(120,128)),("TEST482 LOCK SHA = sealed value",LOCK_SHA==SEALED_LOCK_SHA),("32 records · 64 cartridges (32 A + 32 B) · 4 families × 8",len(RECORDS)==32 and Counter(r[0]for r in RECORDS)==Counter({f:8 for f in FAMS})),
 ("greedy decoding, max_new_tokens = 32",MAX_NEW==32),("BF16 weights",next(model.parameters()).dtype==torch.bfloat16),("SDPA attention",getattr(cfg,"_attn_implementation",None)=="sdpa"),
 ("no forward/backward hooks from this cell (foreign hooks = 0)",not hook_stats()[1]),("frozen eval model",(not model.training)and trainable_tensors()==0 and not lora_present()),("no optimizer",not optimizer_present()),
 ("sampled weight sentinel repeatable",sentinel()==ENGINE.info["sentinel0"])]
_bad=[n for n,ok in ENGINE_LOCK if not ok]
if _bad:raise RuntimeError(f"ENGINE LOCK FAILED: {_bad}")
say("Engine lock: PASS ·",len(ENGINE_LOCK),"checks")
#<<ENGINE_END>>
#<<CORE_BEGIN>>
N_STAGES=10;TOTAL_CALLS=32+32+8+64
def ev(stage,title,body,done=None,total=None,eta=None):return dict(stage=stage,title=title,body=body,done=done,total=total,eta=eta)
def file_ready(p):return p is not None and Path(p).is_file()and Path(p).stat().st_size>0
def prune_runs(keep=2):
    runs=sorted([p for p in ROOT.glob("QWEN32-*")if p.is_dir()],key=lambda p:p.stat().st_mtime)
    for p in runs[:max(0,len(runs)-keep)]:shutil.rmtree(p,ignore_errors=True)
def strip_model(text,mtexts):
    """Remove raw model-generated strings before the stale-reference scan: the scan targets our labels, not the model's own words."""
    for s in sorted(mtexts,key=len,reverse=True):
        if len(s)>=2:
            for v in(s,json.dumps(s,ensure_ascii=False)[1:-1]):text=text.replace(v,"")
    return text
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
def intervals(R):
    c=R["counts"];OFF=R["off_total"];sp=[("s1",c["s1"],32),("linked",c["linked"],32),("s1_fp",c["s1_fp"],OFF),("s2_fp",c["s2_fp"],OFF),("s1_none",c["s1_none"],OFF),("s2_none",c["s2_none"],OFF),
        ("s1_coll",c["s1_coll"],32),("s2_coll",c["s2_coll"],32),("obj_none",c["obj_none"],R["obj_total"]),("id_none",c["id_none"],R["id_total"]),("nm1_hits",R["nomem"]["s1_hits"],32),("nm2_hits",R["nomem"]["s2_hits"],32)]
    out={}
    for k,x,n in sp:lo,hi=cp_interval(x,n);out[k]=dict(k=x,n=n,lo=lo,hi=hi)
    return out
def make_txt(P,R,sha,names,sealed_utc):
    o=[];a=o.append;S="="*140;Dd="-"*140;E_=P["environment"];fz=P["frozen"];c=R["counts"];CI=P["intervals"]
    a(S);a("AKBASCORE NIRVANA · QWEN · 32-BANK COGNITIVE CARTRIDGE — READABLE RUN LOG (SEALED TEST482 REPLAY)");a(S)
    a(f"Derived from {names['payload']} (SHA-256 {sha}). Manifest: {names['manifest']}.");a("The SHA-256 is an artifact integrity seal, not a scientific proof and not a third-party verification.")
    for k_,v in(("RUN ID",P["run_id"]),("START UTC",P["run_start_utc"]),("END UTC",P["run_end_utc"]),("SEALED UTC",sealed_utc),("MODEL",MODEL_ID),("ARCHITECTURE","28 layers / hidden 3584 / 28 Q heads / 4 KV heads / head 128"),
        ("DTYPE / ATTENTION",f"{P['model']['dtype']} / {P['model']['attn']}"),("GPU (this run)",E_["gpu"]),("CANONICAL SEALED GPU",CANON_GPU),("SAME HARDWARE AS SEALED RUN",str(P["hardware"]["same"])),
        ("TORCH / TRANSFORMERS",f"{E_['torch']} / {E_['transformers']} (library versions of the sealed run are not recorded in 482.log)"),("SEED",SEED),("MEMORY ENGINE","K120/V128/OWN · 32 A + 32 B cartridges · full scan per stage · MW1 · MW2 C_COMPACT · NONE-first parser"),
        ("TEST482 LOCK SHA (recomputed)",LOCK_SHA),("TEST482 LOCK SHA (sealed)",SEALED_LOCK_SHA),("DEMO EXTENDED LOCK SHA",EXT_LOCK_SHA)):a(f"{k_:<32}: {v}")
    a("");a("SCOPE");a(Dd)
    for t in P["scope"]:a("• "+t)
    a("");a("TEST482 RECORDS (forge-time source sentences; NOT available to the readout)");a(Dd)
    for L in P["ledger"]:a(f"{L['idx']:02d} {L['fam']} | {L['object']} -> {L['id']} -> {L['place']} | A: {L['source_a']} | B: {L['source_b']}")
    a("");a("CARTRIDGES");a(Dd)
    for t in P["cartridges"]:a(f"{t['cid']} slots={t['T']} code_numbers={t['code_numbers']:,} own={t['own_numbers']:,} native={t['native_numbers']:,} cosK_min={min(t['cosK']):.4f} cosV_min={min(t['cosV']):.4f}")
    a("");a("LIVE RESULTS (this run, re-derived from raw text; intervals are exact Clopper-Pearson 95%)");a(Dd)
    for k_,lab in(("s1","STAGE1 OBJECT->ID"),("linked","LINKED TWO-HOP"),("s1_fp","STAGE1 off-target explicit outputs"),("s2_fp","STAGE2 off-target explicit outputs"),("s1_none","STAGE1 off-target NONE"),("s2_none","STAGE2 off-target NONE"),
        ("s1_coll","STAGE1 collisions"),("s2_coll","STAGE2 collisions"),("obj_none","unrelated-object abstention"),("id_none","absent-ID abstention")):a(f"{lab:<38}: {ci_text(CI[k_]['k'],CI[k_]['n'])}")
    for f in FAMS:a(f"{f}: Stage1 {R['fam'][f][0]}/8 | Linked {R['fam'][f][1]}/8")
    a(f"gold-ID fallbacks used in Stage 2: {R['fallbacks']} (TEST482 sealed: 0)");a(f"VERDICT (live replay): {R['verdict']} | sealed TEST482 verdict: {SEALED482['verdict']} | live counts equal sealed counts: {R['replay_match']}")
    a("");a("PER-RECORD DECISIONS");a(Dd)
    for i,r in enumerate(RECORDS):a(f"[{i+1:02d}] {r[0]} | {r[1]:<20} expected={r[2]} pred={R['s1'][i]['pred']} offFP={R['s1'][i]['fp']} | expected={r[3]:<20} pred={R['s2'][i]['pred']} offFP={R['s2'][i]['fp']} | {'PASS'if R['s2'][i]['linked']else'FAIL'}")
    a("");a("SEALED-SCAN CONTROLS (part of TEST482)");a(Dd)
    for i,cq in enumerate(R["c_obj"]):a(f"OBJECT {ABSENT_OBJECTS[i]!r} -> NONE {cq['none']}/{cq['n']}")
    for i,cq in enumerate(R["c_id"]):a(f"ID {ABSENT_IDS[i]!r} -> NONE {cq['none']}/{cq['n']}")
    a("");a("NOMEM CONTROL — DEMO-ONLY / NOT PART OF TEST482 SEALED RESULT");a(Dd)
    a(f"PAD-only cache, same questions. Strict hits: Stage1 {R['nomem']['s1_hits']}/32 · Stage2 {R['nomem']['s2_hits']}/32 (must be 0).")
    a("");a("FROZEN MODEL");a(Dd)
    for k_ in("sentinel_startup","sentinel_pre_run","sentinel_after"):a(f"{k_:<18}: {fz[k_]}")
    a(f"trainable tensors={fz['trainable_tensors']} lora={fz['lora']} optimizer={fz['optimizer']} training_mode={fz['training_mode']} hooks {fz['hooks_before']}→{fz['hooks_after']} (foreign: {fz['foreign_hooks_before']}→{fz['foreign_hooks_after']})")
    a(fz["sentinel_method"]);a("");a("PRE-SEAL CHECKS");a(Dd)
    for c_ in P["checks_pre_seal"]:a("PASS · "+c_)
    a("");a("SEALED HISTORICAL RECORD (NOT produced by this run)");a(Dd);a(json.dumps(P["sealed_record"],ensure_ascii=False))
    a("");a("TIMING (seconds)");a(Dd)
    for k_,v in P["timing"].items():a(f"{k_:<20}: {v:.2f}")
    a(f"batch calls: {P['counters']['batch_calls']} | sealed-scan row reads: {P['counters']['scan_row_reads']} | NOMEM single-row reads: {P['counters']['nomem_row_reads']} | core forge passes: {P['counters']['core_forge_passes']} | telemetry re-forge passes: {P['counters']['telemetry_reforge_passes']} | peak GPU memory: {P['gpu']['peak_allocated_gib']:.2f} GiB")
    a("");a("Raw generated text for every read is in "+names["payload"]+" and "+names["jsonl"]+".");a(S)
    return "\n".join(o)
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
def _execute_run(E,ctl,state):
    run_id=ctl["run_id"];run_dir=Path(ctl["run_dir"]);I=E.info;checks=[];tm=Counter();T0=time.perf_counter();state["checks"]=checks
    def chk(name,cond):
        checks.append(name)
        if not cond:raise AuditFail(name)
    run_start_utc=utc_now();run_start_local=local_now();say("="*140);say(f"RUN {run_id}");say("="*140)
    # ---- 1/10 integrity ----
    say("[1/10] INTEGRITY · TEST482 LOCK · FROZEN-MODEL PRE-CHECK");yield ev(1,"Integrity check","TEST482 lock, model, architecture, frozen state, hooks and the sampled weight sentinel.")
    fs0=E.frozen_state();same_hw=I["gpu"]==CANON_GPU
    chk("TEST482 LOCK SHA recomputed = sealed value",LOCK_SHA==SEALED_LOCK_SHA);chk("model = Qwen/Qwen2.5-7B-Instruct",I["model_id"]==MODEL_ID);chk("architecture 28L / H3584 / 28Q / 4KV / HD128",tuple(I["arch"])==ARCH)
    chk("dtype bfloat16",I["dtype"]=="bfloat16");chk("attention SDPA",I["attn"]=="sdpa");chk("K120 / V128",(K_DIM,V_DIM)==(120,128));chk("model in eval mode",not fs0["training"]);chk("trainable parameter tensors = 0",fs0["trainable_tensors"]==0)
    chk("no LoRA / PEFT adapter",not fs0["lora"]);chk("no optimizer object",not fs0["optimizer"]);chk("pre-run weight sentinel = startup sentinel",fs0["sentinel"]==I["sentinel0"]);chk("pre-run weight fingerprint = startup fingerprint",fs0["fingerprint"]==I["fingerprint0"])
    if fs0["foreign_hooks"]:raise AuditFail("foreign forward/backward hooks present before run: "+str(fs0["foreign_modules"]))
    chk("no foreign forward/backward hooks before run",fs0["foreign_hooks"]==0)
    for c in checks:say("   PASS ·",c)
    say(f"   hardware: {I['gpu']} | canonical sealed hardware: {CANON_GPU} | same: {same_hw}"+("" if same_hw else"  (different GPU: numerical results may differ from the sealed run; any difference is reported as measured)"))
    # ---- 2/10 codebook ----
    say("[2/10] FIXED NEUTRAL PCA CODEBOOK (TEST482)");yield ev(2,"Fixed neutral codebook","32 neutral sentences → PCA basis per layer × KV head × K/V. The codebook contains none of the 32 records.")
    E.reset_peak();cbi=E.build_codebook();cbi["bytes"]=E.codebook_bytes();tm["codebook"]=cbi["seconds"]
    chk("codebook = 32 neutral sentences",cbi["sentences"]==32);chk("codebook SVD count = layers × 2 × KV heads",cbi["svds"]==ARCH[0]*2*ARCH[3]);say(f"   {cbi['sentences']} sentences · {cbi['svds']} SVDs · {cbi['seconds']:.1f}s · {cbi['bytes']/2**20:.1f} MiB")
    # ---- 3/10 forge ----
    say("[3/10] FORGE · 32 RECORDS → 64 INDEPENDENT CARTRIDGES (32 A + 32 B)");tA=[];tB=[];t0=time.perf_counter()
    for i,(fam,obj,cid,place)in enumerate(RECORDS):
        yield ev(3,f"Forging record {i+1:02d}/32 · cartridges A{i+1:02d} and B{i+1:02d}","Each source sentence is forged into its own numerical cartridge; no cartridge sees another.",done=i,total=32)
        r=E.forge_one(i);tA.append(r["A"]);tB.append(r["B"]);say(f" [{i+1:02d}/32] {fam} | {obj} -> {cid} -> {place} | A slots={r['A']['T']} B slots={r['B']['T']}")
    tm["forge"]=time.perf_counter()-t0
    chk("64 cartridges forged (32 A + 32 B)",len(E.A)==32 and len(E.B)==32);chk("64 independent cartridges (128 distinct tensor storages)",len(set(E.card_ptrs()))==128)
    chk("cartridge tensors finite",E.card_finite());chk("OWN first-token K/V identical to the forged source state in every layer of every cartridge",all(t["own_exact"]for t in tA+tB))
    fs_m=E.frozen_state()
    if fs_m["foreign_hooks"]:raise AuditFail("foreign forward/backward hooks present after forging: "+str(fs_m["foreign_modules"]))
    chk("weight sentinel and fingerprint unchanged after forging (checked before any readout)",fs_m["sentinel"]==I["sentinel0"]and fs_m["fingerprint"]==I["fingerprint0"]);chk("no foreign hooks and no trainable tensors after forging",fs_m["foreign_hooks"]==0 and fs_m["trainable_tensors"]==0)
    # ---- 4/10 source removed ----
    say("[4/10] SOURCE REMOVED · READOUT AUDIT ARMED");yield ev(4,"Source removed","From here the readout receives only a question and the numerical cartridges. Every prompt is checked against all 64 source sentences.",done=32,total=32)
    audit=Counter();qtoks=[]
    def guard(q):
        if any(s in FMT.format(q=q)for s in SRC_ALL):raise AuditFail("SOURCE TEXT FOUND IN A READOUT PROMPT")
        audit["prompts_checked"]+=1;qtoks.append(E.q_tokens(q))
    raw=dict(s1=[],s2=[],c_obj=[],c_id=[],nm1=[],nm2=[]);state["raw"]=raw;done=0;tr=time.perf_counter()
    def rd(fn,q,phase):
        guard(q);E.sync();t=time.perf_counter();rows=fn(q);E.sync();dt_=time.perf_counter()-t;tm[phase]+=dt_;audit["calls_"+phase]+=1;audit["rows_"+phase]+=len(rows);return rows,round(dt_,4)
    def prog(stage,title,body):
        el=time.perf_counter()-tr;return ev(stage,title,body,done=done,total=TOTAL_CALLS,eta=(el/done*(TOTAL_CALLS-done))if done else None)
    # ---- 5/10 Stage 1 ----
    say("[5/10] STAGE 1 FULL SCAN — OBJECT → ID · scans only the 32 A cartridges")
    for i,(fam,obj,gid,gpl)in enumerate(RECORDS):
        rows,s=rd(E.read_A,MW1.format(obj=obj),"s1");raw["s1"].append(dict(q=i,obj=obj,rows=rows,secs=s));done+=1
        parsed=[parse_id(x)for x in rows];pred=decide(parsed);fp=sum(1 for j,x in enumerate(parsed)if j!=i and x is not None)
        say(f" [{i+1:02d}] {fam} | expected={gid} | pred={pred} | gold={parsed[i]} | offFP={fp:2d} | {'PASS'if pred==gid else'FAIL'}");yield prog(5,f"Stage 1 · record {i+1:02d}/32 · object → ID",f'"{obj}" is asked to the 32 A cartridges.')
    # ---- 6/10 Stage 2 ----
    say("[6/10] STAGE 2 FULL SCAN — ID → PLACE · scans only the 32 B cartridges · MW2 = C_COMPACT")
    for i,(fam,obj,gid,gpl)in enumerate(RECORDS):
        pid=decide([parse_id(x)for x in raw["s1"][i]["rows"]]);use=pid if pid is not None else gid
        rows,s=rd(E.read_B,MW2.format(cid=use),"s2");raw["s2"].append(dict(q=i,cid=use,rows=rows,secs=s));done+=1
        parsed=[parse_place(x)for x in rows];pred=decide(parsed);fp=sum(1 for j,x in enumerate(parsed)if j!=i and x is not None)
        say(f" [{i+1:02d}] {fam} | expected={gpl} | pred={pred} | gold={parsed[i]} | offFP={fp:2d} | {'PASS'if(pid==gid and pred==gpl)else'FAIL'}"+("" if pid is not None else" | gold-ID fallback used"));yield prog(6,f"Stage 2 · record {i+1:02d}/32 · ID → place",f"{use} is asked to the 32 B cartridges.")
    # ---- 7/10 sealed-scan controls ----
    say("[7/10] UNRELATED-OBJECT AND ABSENT-ID ABSTENTION PROBES (part of TEST482)")
    for q,obj in enumerate(ABSENT_OBJECTS):
        rows,s=rd(E.read_A,MW1.format(obj=obj),"ctl");raw["c_obj"].append(dict(q=q,obj=obj,rows=rows,secs=s));done+=1;say(f" OBJECT {obj} -> NONE {sum(is_none(x)for x in rows)}/32");yield prog(7,f"Unrelated object {q+1}/4","This object is stored in no cartridge.")
    for q,cid in enumerate(ABSENT_IDS):
        rows,s=rd(E.read_B,MW2.format(cid=cid),"ctl");raw["c_id"].append(dict(q=q,cid=cid,rows=rows,secs=s));done+=1;say(f" ID {cid} -> NONE {sum(is_none(x)for x in rows)}/32");yield prog(7,f"Absent ID {q+1}/4","This container ID is stored in no cartridge.")
    # ---- 8/10 NOMEM (demo-only) ----
    say("[8/10] NOMEM CONTROL — DEMO-ONLY / NOT PART OF TEST482 SEALED RESULT");E.make_nomem()
    for i,(fam,obj,gid,gpl)in enumerate(RECORDS):
        r1,s=rd(E.read_NM,MW1.format(obj=obj),"nm");raw["nm1"].append(r1[0]);done+=1
        r2,s=rd(E.read_NM,MW2.format(cid=gid),"nm");raw["nm2"].append(r2[0]);done+=1
        if i%4==3:yield prog(8,"NOMEM control (DEMO-ONLY) · PAD-only cache","DEMO-ONLY / NOT PART OF TEST482 SEALED RESULT: the same questions with no cartridge installed.")
    say(f"   NOMEM strict hits: Stage1 {sum(int(parse_id(x)==RECORDS[i][2])for i,x in enumerate(raw['nm1']))}/32 · Stage2 {sum(int(parse_place(x)==RECORDS[i][3])for i,x in enumerate(raw['nm2']))}/32 (DEMO-ONLY)")
    # ---- 9/10 audit + seal ----
    say("[9/10] FINAL AUDIT · RESULT RE-DERIVATION · SEAL");yield ev(9,"Final audit and seal","Results are re-derived from the raw generated text; frozen-model checks are repeated.",done=TOTAL_CALLS,total=TOTAL_CALLS)
    R=derive(raw);fs1=E.frozen_state();tm["engine"]=time.perf_counter()-T0
    chk("weight sentinel after run = startup sentinel",fs1["sentinel"]==I["sentinel0"]);chk("weight fingerprint after run = startup fingerprint",fs1["fingerprint"]==I["fingerprint0"])
    chk("model eval mode and trainable tensors = 0 after run",(not fs1["training"])and fs1["trainable_tensors"]==0);chk("no LoRA / optimizer after run",(not fs1["lora"])and(not fs1["optimizer"]))
    if fs1["foreign_hooks"]:raise AuditFail("foreign forward/backward hooks present after run: "+str(fs1["foreign_modules"]))
    chk("no foreign forward/backward hooks after run (demo installs none)",fs1["foreign_hooks"]==0);chk("every sealed-scan read used exactly 32 cartridge rows (A for Stage 1, B for Stage 2)",R["rows_per_read"]==[32])
    chk("source sentences found in readout prompts = 0",audit["prompts_checked"]==TOTAL_CALLS);chk("results re-derived from raw text are stable",derive(raw)==R)
    c=R["counts"];say(f" STAGE1 OBJECT->ID : {c['s1']}/32 | LINKED : {c['linked']}/32 | off-target FP: {c['s1_fp']}/992 + {c['s2_fp']}/992 | off-target NONE: {c['s1_none']}/992 + {c['s2_none']}/992 | collisions: {c['s1_coll']}/32 + {c['s2_coll']}/32")
    say(f" unrelated objects NONE {c['obj_none']}/{R['obj_total']} | absent IDs NONE {c['id_none']}/{R['id_total']} | gold-ID fallbacks {R['fallbacks']} | NOMEM (DEMO-ONLY) strict hits {R['nomem']['s1_hits']}/32 + {R['nomem']['s2_hits']}/32")
    for f in FAMS:say(f"   {f}: Stage1 {R['fam'][f][0]}/8 | Linked {R['fam'][f][1]}/8")
    say("VERDICT (live replay):",R["verdict"],"| sealed TEST482:",SEALED482["verdict"],"| live counts equal sealed counts:",R["replay_match"])
    E.sync();peak=E.peak_gib();CI=intervals(R)
    ledger=[dict(idx=i+1,fam=r[0],object=r[1],id=r[2],place=r[3],source_a=SRC_A[i],source_b=SRC_B[i])for i,r in enumerate(RECORDS)]
    cart=[dict(cid=f"A{i+1:02d}",role="OBJECT->ID",**tA[i])for i in range(32)]+[dict(cid=f"B{i+1:02d}",role="ID->PLACE",**tB[i])for i in range(32)]
    bank=dict(slots=sum(t["T"]for t in tA+tB),code_numbers=sum(t["code_numbers"]for t in tA+tB),own_numbers=sum(t["own_numbers"]for t in tA+tB),native_numbers=sum(t["native_numbers"]for t in tA+tB),codebook_bytes=cbi["bytes"])
    bank["ratio_vs_native"]=(bank["code_numbers"]+bank["own_numbers"])/bank["native_numbers"]
    scan_rows=audit["rows_s1"]+audit["rows_s2"]+audit["rows_ctl"]
    P={"schema":"akbascore.qwen32.cartridge.run.v1","project":"AkbasCore NIRVANA","run_id":run_id,"run_start_utc":run_start_utc,"run_start_local":run_start_local,"run_end_utc":utc_now(),
       "model":{"id":MODEL_ID,"arch":list(ARCH),"dtype":I["dtype"],"attn":I["attn"],"params":I["params"],"pad_id":I["pad_id"],"eos_ids":I["eos_ids"]},
       "environment":{k_:I[k_]for k_ in("gpu","gpu_total_gib","torch","transformers","gradio","python","platform")},
       "hardware":{"canonical_sealed_gpu":CANON_GPU,"this_run_gpu":I["gpu"],"same":bool(same_hw)},
       "engine":{"source":"TEST482 (482_qwen.final.py) core functions, unchanged","K":K_DIM,"V":V_DIM,"own_first_token":"preserved","codebook":"fixed 32-sentence neutral corpus","stage_banks":"Stage 1 scans the 32 A cartridges only; Stage 2 scans the 32 B cartridges only; 64 cartridges in total",
                 "read":"batched isolated read: one cartridge per batch row","mw1":MW1,"mw2_C_COMPACT":MW2,"fmt":FMT,"parser":"NONE-first, closed ID/place vocabulary of the bank","max_new_tokens":MAX_NEW,"lock_sha256_recomputed":LOCK_SHA,"lock_sha256_sealed":SEALED_LOCK_SHA,
                 "extended_lock_sha256":EXT_LOCK_SHA,"stage2_fallback":"Stage 2 uses the gold ID if Stage 1 returned no unique ID (diagnostic rule present in TEST482 code)","compute_path":"PyTorch CUDA backend; no custom CUDA/C++ kernel"},
       "codebook":cbi,"ledger":ledger,"cartridges":cart,"bank":bank,"raw":raw,"results":R,"intervals":CI,
       "raw_hashes":{k_:hashlib.sha256(canon(raw[k_])).hexdigest()for k_ in raw},
       "source_removal":{"prompts_checked":audit["prompts_checked"],"source_sentence_hits":0,"question_tokens_min":min(qtoks),"question_tokens_max":max(qtoks),"calls":{"stage1":audit["calls_s1"],"stage2":audit["calls_s2"],"controls":audit["calls_ctl"],"nomem_demo_only":audit["calls_nm"]},
                         "method":"each formatted prompt compared with all 64 source sentences; readout input = left-padded cartridge rows + question tokens"},
       "frozen":{"sentinel_startup":I["sentinel0"],"sentinel_pre_run":fs0["sentinel"],"sentinel_after":fs1["sentinel"],"fingerprint_startup":I["fingerprint0"],"fingerprint_pre_run":fs0["fingerprint"],"fingerprint_after":fs1["fingerprint"],
                 "tensors":I["fp_names"],"sentinel_method":I["sentinel_method"],"trainable_tensors":fs1["trainable_tensors"],"lora":fs1["lora"],"optimizer":fs1["optimizer"],"training_mode":fs1["training"],"hooks_before":I["hooks0"],"hooks_after":fs1["hooks"],
                 "foreign_hooks_before":fs0["foreign_hooks"],"foreign_hooks_after":fs1["foreign_hooks"],"hook_note":"hook totals include hooks installed by transformers/accelerate; only hooks from other code count as foreign"},
       "counters":{"batch_calls":sum(audit["calls_"+x]for x in("s1","s2","ctl","nm")),"scan_row_reads":scan_rows,"nomem_row_reads":audit["rows_nm"],"core_forge_passes":E.counters["core_forge_passes"],"telemetry_reforge_passes":E.counters["telemetry_reforge_passes"]},
       "timing":{"model_load_seconds":I["model_load_seconds"],"codebook":tm["codebook"],"forge":tm["forge"],"stage1_scan":tm["s1"],"stage2_scan":tm["s2"],"controls":tm["ctl"],"nomem_demo_only":tm["nm"],"engine":tm["engine"]},
       "gpu":{"peak_allocated_gib":peak},"checks_pre_seal":list(checks),
       "sealed_record":{"note":"Historical logs. NOT produced by this run.","test482":SEALED482,"chain":SEALED_CHAIN,"test480_candidates":SEALED480,"test461_16bank":SEALED461},
       "reproduction":{"seed":SEED,"cell_source_sha256":I.get("cell_source_sha256"),"note":"SHA-256 of the executed cell text; the text itself is not embedded because it contains the stale-reference deny-list."},
       "scope":["SEALED TEST482 PANEL REPLAY: the 32 held-out records are replayed unchanged; no new panel and no fresh seed.","Held-out records, objects, IDs and places are new relative to development; the four source templates F1–F4 are the same as in development.",
                "Total bank = 64 cartridges (32 A OBJECT->ID + 32 B ID->PLACE). Stage 1 scans only the 32 A cartridges, Stage 2 only the 32 B cartridges; 992 = 32 queries × 31 off-target cartridges.","The parser uses the bank's closed ID and place vocabulary.",
                "Stage-2 gold-ID fallback exists in the code; it was triggered 0 times in sealed TEST482 and is counted in this run.","NOMEM is DEMO-ONLY and NOT PART OF TEST482 SEALED RESULT.",
                "Hashes, the LOCK SHA and the sentinel are integrity/audit indicators, not independent third-party verification.","Unrelated-object and absent-ID probes overlap with development probes (same four IDs; one identical object, three recombined)."]}
    mtexts={x for k_ in("s1","s2","c_obj","c_id")for e in raw[k_]for x in e["rows"]}|set(raw["nm1"])|set(raw["nm2"]);mtexts|={firstline(t)for t in mtexts}
    P=jsafe(P);P["stale_scan"]={"patterns":len(STALE)+1,"hits":0,"scope":"payload JSON text, raw model-generated strings excluded"}
    hits=stale_scan("payload",strip_model(json.dumps(P,ensure_ascii=False),mtexts),allow=(I["gpu"],CANON_GPU))
    chk("stale-reference scan of the payload: 0 hits",not hits)
    pb=canon(P);sha=hashlib.sha256(pb).hexdigest();payload_name=f"{run_id}_FULL_RUN_LOG_payload.json";pp=run_dir/payload_name;t_seal=time.perf_counter();pp.write_bytes(pb);say("payload sealed:",sha)
    chk("payload SHA-256 verification after write",hashlib.sha256(pp.read_bytes()).hexdigest()==sha)
    names=dict(payload=payload_name,manifest=f"run_manifest_{run_id}.json",txt=f"{run_id}_READABLE_RUN_LOG.txt",summary=f"{run_id}_SUMMARY.json",jsonl=f"{run_id}_RESULTS.jsonl",images=f"images_manifest_{run_id}.json",zip=f"AKBASCORE_QWEN32_CARTRIDGE_ALL_{run_id}.zip")
    mp=run_dir/names["manifest"];sealed_utc=utc_now()
    manifest={"demo":"AKBASCORE NIRVANA · QWEN · 32-BANK COGNITIVE CARTRIDGE (sealed TEST482 replay)","run_id":run_id,"payload_file":payload_name,"payload_sha256":sha,"payload_bytes":len(pb),"sealed_utc":sealed_utc,"verdict":R["verdict"],
              "canonicalization":"json.dumps(sort_keys=True, separators=(',',':'), ensure_ascii=False, allow_nan=False), UTF-8","note":"Artifact integrity seal: confirms the payload is unchanged since sealing. It is not a scientific proof and not an independent third-party verification."}
    mp.write_text(json.dumps(manifest,indent=2,sort_keys=True,ensure_ascii=False),encoding="utf-8")
    summary={"replay_of_test":482,"model":MODEL_ID,"cartridges":64,"records":32,"engine":"K120/V128/OWN · full scan · MW1 + MW2 C_COMPACT · NONE-first parser","counts":R["counts"],"intervals":CI,"families":R["fam"],"verdict_live":R["verdict"],"verdict_sealed":SEALED482["verdict"],
             "live_counts_equal_sealed":R["replay_match"],"gold_id_fallbacks":R["fallbacks"],"nomem_demo_only":{"s1_hits":R["nomem"]["s1_hits"],"s2_hits":R["nomem"]["s2_hits"],"label":R["nomem"]["label"]},"hardware":P["hardware"],
             "lock_sha_recomputed":LOCK_SHA,"lock_sha_sealed":SEALED_LOCK_SHA,"sentinel_before":I["sentinel0"],"sentinel_after":fs1["sentinel"],"run_id":run_id,"payload_sha256":sha}
    (run_dir/names["summary"]).write_text(json.dumps(jsafe(summary),indent=2,ensure_ascii=False),encoding="utf-8")
    lines=[]
    for k_,ph in(("s1","STAGE1"),("s2","STAGE2")):
        for e in raw[k_]:lines.append({"phase":ph,"record":e["q"]+1,"query":e.get("obj",e.get("cid")),"rows":e["rows"]})
    for k_,ph in(("c_obj","UNRELATED_OBJECT"),("c_id","ABSENT_ID")):
        for e in raw[k_]:lines.append({"phase":ph,"probe":e["q"]+1,"query":e.get("obj",e.get("cid")),"rows":e["rows"]})
    for i in range(32):lines.append({"phase":"NOMEM_DEMO_ONLY","record":i+1,"stage1":raw["nm1"][i],"stage2":raw["nm2"][i]})
    (run_dir/names["jsonl"]).write_text("\n".join(json.dumps(x,ensure_ascii=False)for x in lines)+"\n",encoding="utf-8")
    (run_dir/names["txt"]).write_text(make_txt(P,R,sha,names,sealed_utc),encoding="utf-8");seal_s=time.perf_counter()-t_seal
    # ---- 10/10 posters + ZIP ----
    say("[10/10] POSTERS · IMAGE MANIFEST · ZIP");yield ev(10,"Rendering posters","Every number on every poster is checked against the re-derived runtime result before the package is sealed.")
    ctx={"P":P,"R":R,"CI":CI,"run_id":run_id,"sha":sha,"payload_name":payload_name,"seal_seconds":seal_s,"allow":(I["gpu"],CANON_GPU),"N":N_POSTERS,"poster_audit":[]}
    t_r=time.perf_counter();imgs=render_all(ctx,run_dir);render_s=time.perf_counter()-t_r;entries=[]
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
    say("="*140);say("FINAL VERDICT (live replay):",R["verdict"]);say("SEALED TEST482 VERDICT  :",SEALED482["verdict"]);say("PACKAGE       : SEALED ·",f"{len(checks)+len(post)} checks","· payload SHA-256",sha);say("ZIP           :",zp);say("="*140)
    return dict(run_id=run_id,run_dir=run_dir,P=P,R=R,sha=sha,imgs=imgs,zip=zp,payload=pp,txt=run_dir/names["txt"],manifest=mp,images_manifest=imf,summary=run_dir/names["summary"],jsonl=run_dir/names["jsonl"],
                checks=len(checks)+len(post),verdict=R["verdict"])
#<<CORE_END>>
#<<POSTERS_BEGIN>>
from matplotlib.text import Text
plt.rcParams["font.family"]="DejaVu Sans";DPI=120
C_OK,C_UNK,C_ERR,C_CTL,C_CART,C_MOD,C_FG,C_NEU,C_BG,C_SEAL="#047857","#6D28D9","#B91C1C","#475569","#B45309","#0369A1","#0F172A","#334155","#F8FAFC","#0F766E"
C_OKL,C_UNKL,C_ERRL,C_CARTL,C_MODL,C_SEALL="#D1FAE5","#EDE9FE","#FEE2E2","#FEF3C7","#E0F2FE","#CCFBF1"
MONO="DejaVu Sans Mono";BRAND="AKBASCORE NIRVANA · QWEN · 32-BANK COGNITIVE CARTRIDGE"
def mt(s):return str(s).replace("$",r"\$")
def save_jpg(fig,path):
    buf=io.BytesIO();fig.savefig(buf,format="png",dpi=fig.dpi,facecolor="white");plt.close(fig);buf.seek(0)
    with Image.open(buf)as im:
        im=im.convert("RGBA");bg=Image.new("RGB",im.size,(255,255,255));bg.paste(im,mask=im.getchannel("A"))
    bg.save(path,"JPEG",quality=92,optimize=True,progressive=False,subsampling=0)
    with Image.open(path)as chk:
        if chk.format!="JPEG"or chk.mode!="RGB":raise RuntimeError("JPEG validation failed")
    return str(path)
# ===== END PART 2 / 4 — CONTINUE WITH PART 3 =====
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
    Hh=fig.get_figheight();f=lambda inch:1-inch/Hh;live=bool(tag)and tag.startswith("THIS");tc=C_CART if(tag or"").startswith("DEMO")else(C_MOD if live else C_SEAL)
    fig.text(.05,f(.42),BRAND,fontsize=12,weight="bold",color=C_CART,va="center")
    if tag:fig.text(.95,f(.42),tag,fontsize=12,weight="bold",color=tc,va="center",ha="right",bbox=dict(boxstyle="round,pad=0.35",fc="white",ec=tc,lw=1.6))
    fig.text(.05,f(.95),mt(title),fontsize=29,weight="bold",color=C_FG,va="center")
    if sub:fig.text(.05,f(1.42),mt(sub),fontsize=13.5,color=C_NEU,va="center")
    fig.add_artist(Line2D([.05,.95],[f(1.70),f(1.70)],transform=fig.transFigure,color=C_FG,lw=1.2));return f(1.85)
def foot(fig,ctx,k):
    fig.text(.5,.22/fig.get_figheight(),mt(f"AKBASCORE NIRVANA · QWEN 32-BANK | RUN {ctx['run_id']} | PAYLOAD SHA-256 {ctx['sha'][:16]}… | {k:02d}/{ctx['N']:02d}"),ha="center",va="center",fontsize=10,color=C_NEU,family=MONO)
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
def sc(s):return s if len(s)<=14 else s[:13]+"…"
FC={"F1":"#0369A1","F2":"#B45309","F3":"#047857","F4":"#6D28D9"};FCL={"F1":"#E0F2FE","F2":"#FEF3C7","F3":"#D1FAE5","F4":"#EDE9FE"}
CLSC={"none":"#E9D5FF","gold":C_OK,"gold_miss":C_ERR,"other":"#F97316"};CLSI={"none":0,"gold":1,"gold_miss":2,"other":3}
def kc(k,n):return f"{k}/{n}"
def strip(fig,x,y,w,h,cls,gold=None):
    n=len(cls);cw=w/n
    for j,c in enumerate(cls):
        fig.add_artist(Rectangle((x+j*cw,y),cw*.92,h,transform=fig.transFigure,facecolor=CLSC[c],edgecolor=C_OK if j==gold else"white",lw=1.6 if j==gold else .3))
def okc(b):return C_OK if b else C_ERR
def pbar(ax,rows,xmax,colors):
    ax.barh(range(len(rows)),[xmax]*len(rows),color="#E2E8F0");ax.barh(range(len(rows)),[r[1]for r in rows],color=colors);ax.invert_yaxis();ax.set_yticks(range(len(rows)));ax.set_yticklabels([r[0]for r in rows],fontsize=10);ax.set_xlim(0,xmax);ax.set_xticks([]);clean_ax(ax)
# ------------------------------------------------------------------ 01 story
def p01(ctx,k,path):
    P=ctx["P"];R=ctx["R"];C=R["counts"];S=P["source_removal"];m=P["model"]["arch"];OFF=R["off_total"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"WHAT JUST HAPPENED?","32 records became 64 numerical cartridges, the source text was removed, and a frozen model answered by scanning the cartridges.",tag="THIS LIVE RUN")
    bw,bh=.205,.17;xs=[.05+i*(bw+.0267)for i in range(4)];y1=.60
    box(fig,xs[0],y1,bw,bh,"① SOURCE TEXT","64 short sentences\n32 records × 2",C_FG,C_BG);box(fig,xs[1],y1,bw,bh,"② CARTRIDGE FORGE","one forward pass per sentence\nmodel's own K/V → K120/V128",C_MOD,C_MODL)
    box(fig,xs[2],y1,bw,bh,"③ SOURCE REMOVED",f"readout prompts audited: {S['prompts_checked']}\nsource sentences found: 0",C_ERR,C_ERRL);box(fig,xs[3],y1,bw,bh,"④ 32-RECORD MEMORY BANK","64 independent cartridges\n32 A (object→ID) + 32 B (ID→place)",C_CART,C_CARTL)
    for i in range(3):arrow(fig,xs[i]+bw,y1+bh/2,xs[i+1],y1+bh/2)
    arrow(fig,xs[3]+bw/2,y1,xs[3]+bw/2,.485);r0=RECORDS[0];bw2=.168;g2=.0125;xs2=[.05+i*(bw2+g2)for i in range(5)];y2=.315;bh2=.165
    box(fig,xs2[0],y2,bw2,bh2,"⑤ OBJECT QUERY",f'"{r0[1]}"',C_FG,C_BG,tfs=12.5,bfs=10.5);box(fig,xs2[1],y2,bw2,bh2,"⑥ STAGE 1","scans the 32 A cartridges\none unique answer",C_MOD,C_MODL,tfs=12.5,bfs=10.5)
    box(fig,xs2[2],y2,bw2,bh2,"⑦ CONTAINER ID",R["s1"][0]["pred"]or"UNKNOWN",C_OK,C_OKL,tfs=12.5,bfs=14);box(fig,xs2[3],y2,bw2,bh2,"⑧ STAGE 2","scans the 32 B cartridges\nasked about the ID",C_MOD,C_MODL,tfs=12.5,bfs=10.5)
    box(fig,xs2[4],y2,bw2,bh2,"⑨ LOCATION",R["s2"][0]["pred"]or"UNKNOWN",C_OK,C_OKL,tfs=12.5,bfs=14)
    for i in range(4):arrow(fig,xs2[i]+bw2,y2+bh2/2,xs2[i+1],y2+bh2/2)
    fig.text(.05,.29,"Example shown: sealed record 1 (F1). Every query is on poster 07 (Stage 1) and poster 08 (Stage 2). No stage reads all 64 cartridges.",fontsize=10.5,color=C_NEU,style="italic")
    chips=[("STAGE 1 OBJECT→ID",kc(C["s1"],32),C["s1"]==32),("LINKED TWO-HOP",kc(C["linked"],32),C["linked"]==32),("FALSE OUTPUTS",kc(C["s1_fp"]+C["s2_fp"],2*OFF),C["s1_fp"]+C["s2_fp"]==0),
           ("OFF-TARGET NONE",kc(C["s1_none"]+C["s2_none"],2*OFF),C["s1_none"]+C["s2_none"]==2*OFF),("COLLISIONS",kc(C["s1_coll"]+C["s2_coll"],64),C["s1_coll"]+C["s2_coll"]==0),("ABSTENTION PROBES",kc(C["obj_none"]+C["id_none"],R["obj_total"]+R["id_total"]),C["obj_none"]+C["id_none"]==R["obj_total"]+R["id_total"])]
    cw=.1325
    for i,(lab,val,g)in enumerate(chips):chip(fig,.05+i*(cw+.01),.155,cw,.09,lab,val,okc(g),C_OKL if g else C_ERRL)
    fig.text(.05,.128,f"THIS LIVE RUN · replay verdict: {R['verdict']} · sealed TEST482: {SEALED482['verdict']}",fontsize=12,weight="bold",color=okc(R["strict"]))
    fit_text(fig,.05,.04,.9,.075,f"Frozen {MODEL_SHORT} · {m[0]} layers · hidden {m[1]} · {m[2]} Q / {m[3]} KV heads · head {m[4]} · BF16 · SDPA · K120/V128/OWN · greedy · no training, no LoRA, no optimizer, no weight update",fs_max=11,fs_min=8,color=C_NEU)
    foot(fig,ctx,k);return finish(fig,path,ctx,[x[1]for x in chips]+[R["verdict"],str(S["prompts_checked"])])
# ------------------------------------------------------------------ 02 typed bank
def p02(ctx,k,path):
    P=ctx["P"];cart=P["cartridges"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"THE MEMORY BANK · 64 CARTRIDGES = 32 A + 32 B","Two typed banks of independent cartridges. Stage 1 scans only bank A; Stage 2 scans only bank B.",tag="THIS LIVE RUN")
    for bi,(nm,sub,x0)in enumerate((("BANK A · OBJECT → ID","32 cartridges · scanned by Stage 1",.05),("BANK B · ID → PLACE","32 cartridges · scanned by Stage 2",.53))):
        fig.text(x0,top-.02,nm,fontsize=14,weight="bold",color=C_FG,va="top");fig.text(x0,top-.052,sub,fontsize=10.5,color=C_NEU,va="top");cw=(.42-7*.006)/8;ch=.062
        for i in range(32):
            fi,col=divmod(i,8);fam=RECORDS[i][0];t=cart[i+32*bi];x=x0+col*(cw+.006);y=top-.145-fi*(ch+.012)
            fig.add_artist(FancyBboxPatch((x,y),cw,ch,boxstyle="round,pad=0,rounding_size=0.004",transform=fig.transFigure,facecolor=FCL[fam],edgecolor=FC[fam],lw=1.6))
            fig.text(x+cw/2,y+ch*.66,t["cid"],fontsize=9.5,weight="bold",color=FC[fam],ha="center",va="center");fig.text(x+cw/2,y+ch*.27,f"{t['T']} slots",fontsize=7.8,color=C_NEU,ha="center",va="center")
    fig.text(.05,.345,"SOURCE TEMPLATES (forge time only) — each record uses one family for both of its sentences",fontsize=11.5,weight="bold",color=C_NEU,va="center")
    for fi,fam in enumerate(FAMS):
        a,b=sources(fam,"‹object›","‹ID›","‹place›");y=.30-fi*.045;fig.text(.05,y,fam,fontsize=12,weight="bold",color=FC[fam],va="center")
        fig.text(.09,y,"A: "+a,fontsize=10,va="center",color=C_FG);fig.text(.53,y,"B: "+b,fontsize=10,va="center",color=C_FG)
    sl=[t["T"]for t in cart];B=P["bank"]
    fit_text(fig,.05,.04,.9,.09,f"Total bank = 64 independent cartridges (32 A + 32 B). Stage 1 reads exactly the 32 A rows; Stage 2 reads exactly the 32 B rows; no stage reads all 64. Cartridge length {min(sl)if min(sl)==max(sl)else str(min(sl))+chr(8211)+str(max(sl))} slots; the whole bank holds {B['code_numbers']+B['own_numbers']:,} numbers. Each record forged two sentences, one for each bank.",fs_max=12,fs_min=8.5)
    foot(fig,ctx,k);return finish(fig,path,ctx,["BANK A","BANK B","64","32 A"])
# ------------------------------------------------------------------ 03 inside a cartridge
def p03(ctx,k,path):
    P=ctx["P"];C0=P["cartridges"][0];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white");m=P["model"]["arch"];B=P["bank"]
    top=head(fig,"WHAT IS INSIDE A CARTRIDGE?","Numbers only: the model's own K/V states, projected onto fixed directions. Shown for cartridge A01.",tag="THIS LIVE RUN")
    steps=[("SOURCE SENTENCE (forge time only)",SRC_A[0],C_FG,C_BG),("TOKENS",f"{C0['T']} slots: 1 first-token state + {C0['T']-1} content tokens",C_FG,C_BG),("MODEL FORWARD",f"K and V per layer: {m[0]} layers × {m[3]} KV heads × {m[4]} dims",C_MOD,C_MODL),
           ("PROJECTION","fixed neutral codebook → K120 (120 of 128 directions), V128 (all 128)",C_CART,C_CARTL),("NUMERICAL CARTRIDGE",f"{C0['code_numbers']:,} coefficients + {C0['own_numbers']:,} raw first-token numbers",C_OK,C_OKL)]
    y=top-.01;hh=.108
    for i,(t,b,c,fc)in enumerate(steps):
        yy=y-(i+1)*(hh+.022)+.022;fig.add_artist(FancyBboxPatch((.05,yy),.40,hh,boxstyle="round,pad=0,rounding_size=0.008",transform=fig.transFigure,facecolor=fc,edgecolor=c,lw=2.2))
        fig.text(.062,yy+hh-.018,t,fontsize=11.5,weight="bold",color=c,va="top");fit_text(fig,.062,yy+.006,.376,hh-.04,b,fs_max=10.5,fs_min=8)
        if i<len(steps)-1:fig.text(.25,yy-.011,"▼",fontsize=10,color=C_NEU,ha="center",va="center")
    ex=C0["excerpt"];V=np.array(ex["values"],dtype=float);ax=fig.add_axes([.54,.545,.40,.215]);lim=np.percentile(np.abs(V),97)or 1.0
    im=ax.imshow(V,aspect="auto",cmap="RdBu_r",vmin=-lim,vmax=lim);ax.set_xlabel("PCA coefficient index (0–119)",fontsize=9.5,labelpad=2);ax.set_ylabel("content token",fontsize=9.5)
    ax.set_title(f"REAL DATA · A01 · layer {ex['layer']} · K head {ex['head']} · coefficients",fontsize=11.5,weight="bold",loc="left",pad=8);fig.colorbar(im,cax=fig.add_axes([.945,.545,.008,.215]))
    cK=np.array([c["cosK"]for c in P["cartridges"]]);cV=np.array([c["cosV"]for c in P["cartridges"]]);ax2=fig.add_axes([.54,.255,.40,.17]);xs=np.arange(cK.shape[1])
    for arr,col,lab in((cK,C_CART,"K (120 of 128 directions)"),(cV,C_MOD,"V (128 of 128 directions)")):
        ax2.fill_between(xs,arr.min(0),arr.max(0),color=col,alpha=.2);ax2.plot(xs,arr.mean(0),color=col,lw=2.2,label=f"{lab}: mean over 64 cartridges (band = min–max)")
    lo=min(cK.min(),cV.min());ax2.set_ylim(max(0,lo-.01),1.003);ax2.set_xlabel("transformer layer",fontsize=9.5,labelpad=2);ax2.set_ylabel("cosine to own K/V",fontsize=9.5)
    ax2.legend(frameon=False,fontsize=8.2,loc="lower left");ax2.set_title("MEASURED · how faithful is the projection?",fontsize=11.5,weight="bold",loc="left",pad=8);clean_ax(ax2);ax2.grid(alpha=.25)
    fit_text(fig,.05,.045,.9,.085,f"NO HUMAN-READABLE SOURCE COPY. The 64 cartridges hold {B['code_numbers']+B['own_numbers']:,} numbers ({B['ratio_vs_native']*100:.1f}% of the model's native K/V size: a projection, not a size reduction). The {len(CORPUS)}-sentence neutral codebook ({B['codebook_bytes']/2**20:.1f} MiB) is shared and contains none of the 32 records.",fs_max=11,fs_min=8)
    foot(fig,ctx,k);return finish(fig,path,ctx,[f"{C0['T']} slots",f"{B['code_numbers']+B['own_numbers']:,}",f"{B['ratio_vs_native']*100:.1f}%"])
# ------------------------------------------------------------------ 04 source removal
def p04(ctx,k,path):
    P=ctx["P"];S=P["source_removal"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"WAS THE SOURCE REALLY REMOVED?","What the frozen model receives at readout: a question as text, the knowledge only as installed numbers.",tag="THIS LIVE RUN")
    bw,bh=.27,.25;xs=[.05,.365,.68];y0=top-bh-.03
    box(fig,xs[0],y0,bw,bh,"FORGE TIME","SOURCE TEXT = PRESENT\n\nused once, inside the forward\npass that makes a cartridge",C_FG,C_BG,bfs=11.5);box(fig,xs[1],y0,bw,bh,"AFTER FORGE","SOURCE TEXT = REMOVED\nNOT PROVIDED TO READOUT\n\nNO HUMAN-READABLE SOURCE COPY",C_ERR,C_ERRL,bfs=11.5)
    box(fig,xs[2],y0,bw,bh,"READOUT INPUT","QUESTION  +  NUMERICAL\nCARTRIDGE K/V ROWS\n\nMODEL WEIGHTS: UNCHANGED",C_MOD,C_MODL,bfs=11.5);arrow(fig,xs[0]+bw,y0+bh/2,xs[1],y0+bh/2);arrow(fig,xs[1]+bw,y0+bh/2,xs[2],y0+bh/2)
    cl=S["calls"];rows=[("Stage 1 · 32 questions",cl["stage1"]),("Stage 2 · 32 questions",cl["stage2"]),("unrelated object + absent ID",cl["controls"]),("NOMEM (DEMO-ONLY)",cl["nomem_demo_only"])]
    ax=fig.add_axes([.27,.295,.60,.175]);ax.barh(range(4),[v for _,v in rows],color=[C_MOD,C_MOD,C_UNK,C_CTL]);ax.invert_yaxis();ax.set_yticks(range(4));ax.set_yticklabels([a for a,_ in rows],fontsize=11)
    for i,(_,v)in enumerate(rows):ax.text(v+.8,i,f"{v} readout prompts · source sentences found: 0",va="center",fontsize=11,weight="bold")
    ax.set_xlim(0,max(v for _,v in rows)*1.95);ax.set_xlabel("readout prompts checked against all 64 source sentences",fontsize=10,labelpad=3);clean_ax(ax)
    note(fig,.05,.045,.9,.13,f"{S['prompts_checked']} readout prompts were compared with every source sentence before use. The question must name the object or container ID it asks about — that is the question, not the record ({S['question_tokens_min']}–{S['question_tokens_max']} question tokens).",
         "A cartridge is a set of K/V tensors (coefficients + reconstructed rows). It is not text, a summary, or a sentence embedding.")
    foot(fig,ctx,k);return finish(fig,path,ctx,[str(S["prompts_checked"]),"SOURCE TEXT = REMOVED","NOMEM (DEMO-ONLY)"])
# ------------------------------------------------------------------ 05 audit locks
def p05(ctx,k,path):
    P=ctx["P"];R=ctx["R"];fz=P["frozen"];S=P["source_removal"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"AUDIT LOCKS · FROZEN MODEL","Each lock is marked CHECKED (verified in this run) or BY CONSTRUCTION (a property of the code path).",tag="THIS LIVE RUN")
    ax=fig.add_axes([.05,top-.37,.34,.33]);ax.axis("off");ax.set_xlim(0,1);ax.set_ylim(0,1);ax.text(0,1,"WEIGHT SENTINEL · 3 CHECKPOINTS",fontsize=11.5,weight="bold",color=C_MOD,va="top")
    for j,(lab,key)in enumerate((("startup","sentinel_startup"),("pre-run","sentinel_pre_run"),("post-run","sentinel_after"))):
        y=.80-j*.17;same=fz[key]==fz["sentinel_startup"];ax.text(0,y,lab,fontsize=10.5,weight="bold",va="center");ax.text(.22,y,fz[key][:28]+"…",fontsize=8.8,family=MONO,va="center");ax.text(1.0,y,"✓ identical"if same else"✗ CHANGED",fontsize=10,weight="bold",color=okc(same),va="center",ha="right")
    sf=fz["fingerprint_startup"]==fz["fingerprint_after"];ax.text(0,.27,f"float32 sums of the same {len(fz['tensors'])} tensors: {'identical'if sf else'CHANGED'}",fontsize=10,va="center",color=okc(sf))
    ax.text(0,.14,f"trainable tensors {fz['trainable_tensors']} · LoRA {'none'if not fz['lora']else'PRESENT'} · optimizer {'none'if not fz['optimizer']else'PRESENT'}",fontsize=10,va="center");ax.text(0,.02,f"sampled sentinel ({len(fz['tensors'])} tensors × 16 slices × 256 values), not a hash of all weights",fontsize=8.6,va="center",color=C_NEU,style="italic")
    sp=fz["sentinel_startup"]==fz["sentinel_after"]
    locks=[("MODEL WEIGHTS","FROZEN","CHECKED","sentinel + fingerprint unchanged, 0 trainable tensors"),("WEIGHT SENTINEL","PASS"if sp else"FAIL","CHECKED","3 checkpoints identical (left)"),("SOURCE AT READOUT","ABSENT","CHECKED",f"{S['prompts_checked']} prompts audited, 0 source sentences found"),
           ("CARTRIDGE","NUMERICAL K/V","BY CONSTRUCTION","tensors only; no text is stored"),("K / V","K120 / V128","CHECKED","K 120 of 128 directions, V 128 of 128"),("OWN","PRESERVED","CHECKED","first-token K/V identical in 64 cartridges × 28 layers"),
           ("CARTRIDGES","INDEPENDENT","CHECKED","64 separate forges, 128 distinct tensor storages"),("READOUT","BATCHED ISOLATED","BY CONSTRUCTION","one cartridge per batch row"),("STAGE BANKS","S1 = A32 · S2 = B32","CHECKED",f"every scan read exactly {R['rows_per_read'][0]} rows"),
           ("ROUTER · INDEX · SIGNATURE","NONE","BY CONSTRUCTION","full scan of the stage's bank"),("TRAINING · LoRA · OPTIMIZER","NONE","CHECKED","no grads, no adapter, no optimizer object"),("WEIGHT UPDATE","NONE","CHECKED",f"sentinel unchanged after {P['counters']['batch_calls']} batch calls"),("GENERATION","GREEDY","BY CONSTRUCTION","argmax decoding, max 32 new tokens")]
    ax2=fig.add_axes([.43,.10,.52,top-.12]);ax2.axis("off");ax2.set_xlim(0,1);ax2.set_ylim(0,1);rh=1/len(locks)
    for i,(a,b,c,d)in enumerate(locks):
        y=1-(i+.5)*rh;col=C_OK if c=="CHECKED"else C_CTL
        ax2.add_patch(FancyBboxPatch((0,y-rh*.42),1,rh*.84,boxstyle="round,pad=0,rounding_size=0.01",transform=ax2.transAxes,facecolor="white",edgecolor="#CBD5E1",lw=1.2))
        ax2.text(.015,y+rh*.12,a,fontsize=9.6,weight="bold",color=C_FG,va="center");ax2.text(.015,y-rh*.22,d,fontsize=8.3,color=C_NEU,va="center");ax2.text(.50,y,b,fontsize=10.5,weight="bold",color=okc(b!="FAIL"),va="center",family=MONO)
        ax2.text(.985,y,c,fontsize=8.8,weight="bold",color="white",va="center",ha="right",bbox=dict(boxstyle="round,pad=0.25",fc=col,ec=col))
    fit_text(fig,.05,.045,.34,.20,f"{len(P['checks_pre_seal'])} technical checks passed before sealing. Hashes and the sentinel are integrity indicators, not independent third-party verification. Compute path: PyTorch CUDA backend, no custom kernel.",fs_max=11,fs_min=8.5,color=C_NEU)
    foot(fig,ctx,k);return finish(fig,path,ctx,["FROZEN","ABSENT","NONE","GREEDY","K120 / V128",str(S["prompts_checked"])])
# ------------------------------------------------------------------ 06 four family chains
def p06(ctx,k,path):
    R=ctx["R"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"FOUR CHAINS · OBJECT → ID → PLACE","One example per source family. Each strip is one scan: 32 cartridges, only the gold cartridge answers, the others say NONE.",tag="THIS LIVE RUN")
    for fi,fam in enumerate(FAMS):
        i=fi*8;fam_,obj,gid,gpl=RECORDS[i];yt=top-.015-fi*.178;s1=R["s1"][i];s2=R["s2"][i]
        fig.add_artist(FancyBboxPatch((.05,yt-.158),.04,.158,boxstyle="round,pad=0,rounding_size=0.006",transform=fig.transFigure,facecolor=FCL[fam],edgecolor=FC[fam],lw=2));fig.text(.07,yt-.079,fam,fontsize=15,weight="bold",color=FC[fam],ha="center",va="center")
        fig.text(.105,yt-.035,f"OBJECT: {obj}",fontsize=10.5,weight="bold",va="center");fig.text(.105,yt-.07,f"Stage 1 · {32} A cartridges",fontsize=8.8,color=C_NEU,va="center")
        strip(fig,.30,yt-.050,.50,.032,s1["cls"],i);fig.text(.30+(i+.5)*.50/32,yt-.062,f"A{i+1:02d}",fontsize=7.5,color=C_OK,ha="center",va="top",weight="bold")
        fig.text(.815,yt-.034,f"ID {s1['pred']or'UNKNOWN'}",fontsize=11,weight="bold",color=okc(s1["ok"]),va="center")
        fig.text(.105,yt-.115,f"ID: {s2['cid']}",fontsize=10.5,weight="bold",va="center");fig.text(.105,yt-.148,f"Stage 2 · {32} B cartridges",fontsize=8.8,color=C_NEU,va="center")
        strip(fig,.30,yt-.130,.50,.032,s2["cls"],i);fig.text(.30+(i+.5)*.50/32,yt-.142,f"B{i+1:02d}",fontsize=7.5,color=C_OK,ha="center",va="top",weight="bold")
        fig.text(.815,yt-.114,f"{s2['pred']or'UNKNOWN'}",fontsize=11,weight="bold",color=okc(s2["linked"]),va="center")
    lg=[Line2D([0],[0],marker="s",color="w",markerfacecolor=C_OK,markersize=11,label="gold cartridge answered"),Line2D([0],[0],marker="s",color="w",markerfacecolor="#E9D5FF",markeredgecolor=C_UNK,markersize=11,label="NONE"),Line2D([0],[0],marker="s",color="w",markerfacecolor="#F97316",markersize=11,label="explicit output from another cartridge"),Line2D([0],[0],marker="s",color="w",markerfacecolor=C_ERR,markersize=11,label="gold cartridge missed")]
    fig.legend(handles=lg,loc="lower center",bbox_to_anchor=(.5,.045),ncol=4,frameon=False,fontsize=9.6);foot(fig,ctx,k)
    return finish(fig,path,ctx,["FOUR CHAINS",RECORDS[0][1],RECORDS[8][1],RECORDS[16][1],RECORDS[24][1]])
# ------------------------------------------------------------------ 07/08 matrices
def matrix_poster(ctx,k,path,stage):
    R=ctx["R"];CI=ctx["CI"];C=R["counts"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white");key="s1"if stage==1 else"s2";ok_key="ok"if stage==1 else"linked"
    ttl="STAGE 1 · OBJECT → ID · 32 × 32 FULL SCAN"if stage==1 else"STAGE 2 · ID → PLACE · 32 × 32 FULL SCAN"
    top=head(fig,ttl,"Row = question, column = cartridge. Green diagonal: the gold cartridge answered. Purple: NONE. Orange or red would be errors.",tag="THIS LIVE RUN")
    cmap=ListedColormap([CLSC["none"],CLSC["gold"],CLSC["gold_miss"],CLSC["other"]]);M=np.array([[CLSI[c]for c in R[key][i]["cls"]]for i in range(32)]);ax=fig.add_axes([.215,.165,.52,top-.185])
    ax.imshow(M,aspect="auto",cmap=cmap,vmin=0,vmax=3);pre="A"if stage==1 else"B";ax.set_xticks(range(32));ax.set_xticklabels([f"{pre}{j+1:02d}"for j in range(32)],rotation=90,fontsize=7)
    lab=[f"{i+1:02d} {RECORDS[i][0]} {RECORDS[i][1]}"for i in range(32)]if stage==1 else[f"{i+1:02d} {RECORDS[i][0]} {R['s2'][i]['cid']}"for i in range(32)]
    ax.set_yticks(range(32));ax.set_yticklabels(lab,fontsize=7.3);ax.set_xticks(np.arange(-.5,32,1),minor=True);ax.set_yticks(np.arange(-.5,32,1),minor=True);ax.grid(which="minor",color="white",lw=.8);ax.tick_params(which="minor",length=0)
    for b in(7.5,15.5,23.5):ax.axhline(b,color=C_FG,lw=1.6);ax.axvline(b,color=C_FG,lw=1.6)
    for i in range(32):
        e=R[key][i];good=e[ok_key];pred=e["pred"]or"UNKNOWN";ax.text(32.1,i,f"{pred} {'✓'if good else'✗'}",fontsize=7.3,va="center",color=okc(good),weight="bold",clip_on=False)
    fp,nn,co=("s1_fp","s1_none","s1_coll")if stage==1 else("s2_fp","s2_none","s2_coll");OFF=R["off_total"]
    lg=[Line2D([0],[0],marker="s",color="w",markerfacecolor=CLSC["gold"],markersize=10,label="gold cartridge answered"),Line2D([0],[0],marker="s",color="w",markerfacecolor=CLSC["none"],markeredgecolor=C_UNK,markersize=10,label="NONE"),Line2D([0],[0],marker="s",color="w",markerfacecolor=CLSC["other"],markersize=10,label="explicit output, other cartridge"),Line2D([0],[0],marker="s",color="w",markerfacecolor=CLSC["gold_miss"],markersize=10,label="gold cartridge missed")]
    fig.legend(handles=lg,loc="lower center",bbox_to_anchor=(.5,.075),ncol=4,frameon=False,fontsize=9.4)
    cnt=C["s1"]if stage==1 else C["linked"];fig.text(.05,.045,f"LIVE · {'Stage 1'if stage==1 else'linked'} {kc(cnt,32)} · off-target explicit outputs {kc(C[fp],OFF)} · off-target NONE {kc(C[nn],OFF)} · collisions {kc(C[co],32)}",fontsize=11.5,weight="bold")
    foot(fig,ctx,k);return finish(fig,path,ctx,[kc(cnt,32),kc(C[fp],OFF),kc(C[nn],OFF),kc(C[co],32)])
def p07(ctx,k,path):return matrix_poster(ctx,k,path,1)
def p08(ctx,k,path):return matrix_poster(ctx,k,path,2)
# ------------------------------------------------------------------ 09 off-target behaviour
def p09(ctx,k,path):
    R=ctx["R"];CI=ctx["CI"];C=R["counts"];OFF=R["off_total"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"WHAT THE OTHER 31 CARTRIDGES DO","992 = 32 queries × 31 off-target cartridges, per stage. Each stage scans only its own bank of 32.",tag="THIS LIVE RUN")
    box(fig,.05,top-.19,.14,.12,"ONE QUESTION","to one bank of 32",C_FG,C_BG,tfs=11,bfs=10);arrow(fig,.19,top-.13,.235,top-.13)
    for j in range(32):fig.add_artist(Rectangle((.245+j*.0155,top-.165),.0145,.065,transform=fig.transFigure,facecolor=C_OK if j==0 else CLSC["none"],edgecolor=C_OK if j==0 else"white",lw=1.2))
    fig.text(.245,top-.18,"1 gold cartridge",fontsize=9.5,color=C_OK,weight="bold",va="top");fig.text(.62,top-.18,"31 off-target cartridges",fontsize=9.5,color=C_UNK,weight="bold",va="top",ha="right");fig.text(.76,top-.13,"32 × 31 = 992",fontsize=15,weight="bold",va="center")
    fig.text(.05,top-.235,f"Gold-ID fallbacks used in Stage 2 (diagnostic rule in the TEST482 code): {R['fallbacks']}  ·  sealed TEST482: 0",fontsize=10.5,color=C_NEU,va="center")
    for n_,(stg,fp,nn,co)in enumerate((("STAGE 1 · bank A","s1_fp","s1_none","s1_coll"),("STAGE 2 · bank B","s2_fp","s2_none","s2_coll"))):
        x0=.05+n_*.47;ax=fig.add_axes([x0+.14,top-.50,.28,.18]);rows=[("explicit false\noutputs",C[fp]),("NONE",C[nn])];pbar(ax,rows,OFF,[C_ERR,C_OK])
        for i,(l,v)in enumerate(rows):ax.text(OFF*.5,i,kc(v,OFF),ha="center",va="center",fontsize=13,weight="bold",color="white"if v>OFF*.15 else C_FG)
        ax.set_title(stg,fontsize=12.5,weight="bold",loc="left");fig.text(x0,top-.525,f"collisions {kc(C[co],32)}",fontsize=11,weight="bold",va="center")
        for r_,(kk,lab)in enumerate(((fp,"false outputs"),(nn,"NONE"),(co,"collisions"))):
            ci=CI[kk];fig.text(x0,top-.56-r_*.032,f"{lab}: {pc(ci['lo'])} – {pc(ci['hi'])}  (exact 95% CI)",fontsize=9.3,color=C_NEU,va="center")
    fit_text(fig,.05,.04,.9,.10,"Collision = no unique answer because two or more different valid values were returned. Intervals are exact Clopper–Pearson 95% limits computed in this run. 0 events in 992 reads bound the per-read false-output rate only to about 0.37%; they do not show a zero rate.",fs_max=11.5,fs_min=8.5,color=C_NEU)
    foot(fig,ctx,k);return finish(fig,path,ctx,["32 × 31 = 992",kc(C["s1_fp"],OFF),kc(C["s2_fp"],OFF),kc(C["s1_none"],OFF),kc(C["s2_none"],OFF)])
# ------------------------------------------------------------------ 10 sealed-scan controls
def p10(ctx,k,path):
    R=ctx["R"];C=R["counts"];CI=ctx["CI"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"IF THE ANSWER IS NOT STORED","Unrelated objects (Stage-1 bank) and absent IDs (Stage-2 bank): every cartridge must say NONE. Part of the sealed TEST482 scan.",tag="THIS LIVE RUN")
    for n_,(rows,labs,ttl,key)in enumerate(((R["c_obj"],ABSENT_OBJECTS,f"UNRELATED OBJECTS · {kc(C['obj_none'],R['obj_total'])}","obj_none"),(R["c_id"],ABSENT_IDS,f"ABSENT CONTAINER IDs · {kc(C['id_none'],R['id_total'])}","id_none"))):
        ax=fig.add_axes([.14+n_*.45,top-.40,.28,.28]);ax.barh(range(4),[r["n"]for r in rows],color=C_ERRL);ax.barh(range(4),[r["none"]for r in rows],color=C_UNK);ax.invert_yaxis();ax.set_yticks(range(4));ax.set_yticklabels(labs,fontsize=11)
        for i,r in enumerate(rows):ax.text(r["n"]+.5,i,("✓ NONE ×"+str(r["none"]))if r["none"]==r["n"]else f"✗ NONE ×{r['none']}",va="center",fontsize=10.5,weight="bold",color=okc(r["none"]==r["n"]))
        ax.set_xlim(0,48);ax.set_xlabel("cartridges returning NONE (of 32)",fontsize=10);ax.set_title(ttl,fontsize=12.5,weight="bold",loc="left");clean_ax(ax)
        ci=CI[key];fig.text(.14+n_*.45,top-.46,f"exact 95% CI: {pc(ci['lo'])} – {pc(ci['hi'])}",fontsize=10.5,color=C_NEU)
    fit_text(fig,.05,.20,.9,.20,"Each absent query is read by all 32 cartridges of the matching bank, so 4 queries give 128 reads. Honest limits: the four absent IDs are identical to the development probes; of the four unrelated objects one (granite whistle) is identical and three are recombinations of development words. These probes test abstention for unstored items; they are not a new held-out set.",fs_max=12,fs_min=9)
    fit_text(fig,.05,.045,.9,.11,"Absence is tested on this closed bank. The parser matches the bank's own 32 IDs and 32 places, so a hallucinated place or ID outside that vocabulary would be read as no answer.",fs_max=11,fs_min=8.5,color=C_NEU)
    foot(fig,ctx,k);return finish(fig,path,ctx,[kc(C["obj_none"],R["obj_total"]),kc(C["id_none"],R["id_total"]),"granite whistle"])
# ------------------------------------------------------------------ 11 NOMEM demo-only
def p11(ctx,k,path):
    R=ctx["R"];N=R["nomem"];CI=ctx["CI"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"NOMEM CONTROL · NO CARTRIDGE INSTALLED","The same questions with an empty (PAD-only) cache. Do the right answers come from the cartridges?",tag="DEMO-ONLY")
    fig.add_artist(FancyBboxPatch((.05,top-.075),.90,.06,boxstyle="round,pad=0,rounding_size=0.008",transform=fig.transFigure,facecolor=C_CARTL,edgecolor=C_CART,lw=2.6))
    fig.text(.5,top-.045,"DEMO-ONLY / NOT PART OF TEST482 SEALED RESULT",ha="center",va="center",fontsize=17,weight="bold",color=C_CART)
    for n_,(rows,ttl,key,hits)in enumerate(((N["s1"],"STAGE 1 · object → ID","nm1_hits",N["s1_hits"]),(N["s2"],"STAGE 2 · ID → place","nm2_hits",N["s2_hits"]))):
        y0=top-.20-n_*.20;fig.text(.05,y0+.035,f"{ttl} · strict hits {kc(hits,32)}",fontsize=13,weight="bold",color=okc(hits==0),va="center")
        cw=.62/32
        for i,e in enumerate(rows):fig.add_artist(Rectangle((.05+i*cw,y0-.03),cw*.92,.045,transform=fig.transFigure,facecolor=C_ERR if e["hit"]else C_OKL,edgecolor="white",lw=.3))
        ci=CI[key];fig.text(.70,y0-.007,f"exact 95% CI of the hit rate: {pc(ci['lo'])} – {pc(ci['hi'])}",fontsize=10,color=C_NEU,va="center")
        fig.text(.05,y0-.065,"first generated line of the first 6 records: "+" · ".join(sc(e["first"])if e["first"]else"(empty)"for e in rows[:6]),fontsize=9.3,family=MONO,color=C_NEU,va="center")
    fit_text(fig,.05,.07,.9,.17,"A strict hit means the model without any cartridge produced the stored ID (Stage 1) or the stored place (Stage 2). Zero hits are expected, because the 32 records are invented. This control was added by the demo for the human reader; the sealed TEST482 run did not contain it, and it is never counted in the sealed metrics.",fs_max=12,fs_min=9)
    foot(fig,ctx,k);return finish(fig,path,ctx,["DEMO-ONLY / NOT PART OF TEST482 SEALED RESULT",kc(N["s1_hits"],32),kc(N["s2_hits"],32)])
# ===== END PART 3 / 4 — CONTINUE WITH PART 4 =====
# ------------------------------------------------------------------ 12 families
def p12(ctx,k,path):
    R=ctx["R"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"RESULTS BY SOURCE FAMILY","Four ways of phrasing the same two facts. The held-out records are new; the four templates are the same as in development.",tag="THIS LIVE RUN")
    ax=fig.add_axes([.08,.43,.52,top-.52]);x=np.arange(4);w=.38
    s1v=[R["fam"][f][0]for f in FAMS];lkv=[R["fam"][f][1]for f in FAMS];ax.bar(x-w/2,[8]*4,w,color="#E2E8F0");ax.bar(x+w/2,[8]*4,w,color="#E2E8F0");ax.bar(x-w/2,s1v,w,color=C_MOD,label="Stage 1");ax.bar(x+w/2,lkv,w,color=C_OK,label="linked two-hop")
    for i in range(4):ax.text(x[i]-w/2,s1v[i]+.15,kc(s1v[i],8),ha="center",fontsize=12,weight="bold");ax.text(x[i]+w/2,lkv[i]+.15,kc(lkv[i],8),ha="center",fontsize=12,weight="bold")
    ax.set_xticks(x);ax.set_xticklabels(FAMS,fontsize=13,weight="bold");ax.set_ylim(0,9.6);ax.set_yticks([0,4,8]);ax.legend(frameon=False,fontsize=10.5,loc="upper right",ncol=2);clean_ax(ax)
    ax2=fig.add_axes([.64,.43,.31,top-.52]);ax2.axis("off");ax2.set_xlim(0,1);ax2.set_ylim(0,1);ax2.text(0,1,"SEALED TEST482",fontsize=11.5,weight="bold",color=C_SEAL,va="top")
    for i,f in enumerate(FAMS):ax2.text(0,.82-i*.17,f"{f}: Stage 1 {kc(*SEALED482['fam'][f][0])} · linked {kc(*SEALED482['fam'][f][1])}",fontsize=11,va="center")
    ax2.text(0,.12,f"live equals sealed: {'YES'if R['replay_match']else'NO'}",fontsize=11.5,weight="bold",color=okc(R["replay_match"]),va="center")
    for fi,f in enumerate(FAMS):
        a,b=sources(f,"‹object›","‹ID›","‹place›");y=.345-fi*.058;fig.text(.05,y,f,fontsize=12,weight="bold",color=FC[f],va="center");fig.text(.09,y,"A: "+a,fontsize=9.8,va="center");fig.text(.53,y,"B: "+b,fontsize=9.8,va="center")
    fit_text(fig,.05,.04,.9,.07,"Eight records per family. In the development bank, F3 (TEST477) and F2 (TEST479) each contained the single linked failure that motivated the Stage-2 readout study; see posters 13 and 14.",fs_max=11,fs_min=8.5,color=C_NEU)
    foot(fig,ctx,k);return finish(fig,path,ctx,[kc(R["fam"]["F1"][1],8),kc(R["fam"]["F4"][1],8)])
# ------------------------------------------------------------------ 13 development chain
def p13(ctx,k,path):
    fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"HOW WE GOT HERE · TEST477 → TEST482","Development found two single failures and a readout fix; the final held-out test then used new records with the readout frozen.",tag="SEALED RECORD · TEST477–482")
    n=len(SEALED_CHAIN);hh=(top-.07)/n
    for i,e in enumerate(SEALED_CHAIN):
        y=top-.01-(i+1)*hh+.012;held=e["role"]=="HELD-OUT";col=C_SEAL if held else C_NEU;fc=C_SEALL if held else C_BG
        fig.add_artist(FancyBboxPatch((.05,y),.90,hh-.014,boxstyle="round,pad=0,rounding_size=0.006",transform=fig.transFigure,facecolor=fc,edgecolor=col,lw=2.2 if held else 1.3))
        fig.text(.062,y+hh-.034,f"TEST{e['t']}",fontsize=13,weight="bold",color=col,va="center");fig.text(.145,y+hh-.034,e["role"],fontsize=9.5,weight="bold",color="white",va="center",bbox=dict(boxstyle="round,pad=0.25",fc=col,ec=col))
        fig.text(.25,y+hh-.034,e["what"],fontsize=10.5,weight="bold",va="center");fig.text(.94,y+hh-.034,e["verdict"],fontsize=9.3,family=MONO,color=C_OK if e["verdict"].startswith(("PASS","BOTH","DIRECT","MULTIPLE"))else C_ERR,va="center",ha="right")
        nums=[]
        if e["s1"]:nums.append(f"Stage 1 {kc(*e['s1'])}")
        if e["linked"]:nums.append(f"linked {kc(*e['linked'])}")
        fit_text(fig,.062,y+.006,.876,hh-.07,("  ·  ".join(nums)+"  ·  "if nums else"")+e["note"]+f"  [{e['log']}]",fs_max=9.6,fs_min=7.8,color=C_FG)
    foot(fig,ctx,k);return finish(fig,path,ctx,["TEST477","TEST478","TEST479","TEST480","TEST481","TEST482","HELD-OUT"])
# ------------------------------------------------------------------ 14 readout selection
def p14(ctx,k,path):
    fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"HOW THE STAGE-2 READOUT WAS CHOSEN","TEST480: five wordings on four isolated cartridges (n=4). TEST481: the two survivors on the full development bank. Frozen before TEST482.",tag="SEALED RECORD · TEST480–481")
    cols=[("candidate",.05),("wording of the Stage-2 question",.16),("gold",.585),("off-FP",.645),("off-NONE",.705),("collis.",.775),("absent",.835),("result",.885)];yh=top-.025
    for t,x in cols:fig.text(x,yh,t,fontsize=10,weight="bold",color=C_NEU,va="center")
    rh=.088
    for i,(nm,wd,g,fp,nn,co,ab,st)in enumerate(SEALED480["cands"]):
        y=yh-.03-(i+1)*rh+.012;fig.add_artist(FancyBboxPatch((.05,y),.90,rh-.012,boxstyle="round,pad=0,rounding_size=0.005",transform=fig.transFigure,facecolor=C_OKL if st else"white",edgecolor=C_OK if st else"#CBD5E1",lw=1.6 if st else 1))
        fig.text(.057,y+rh/2-.006,nm,fontsize=10,weight="bold",family=MONO,va="center");fit_text(fig,.16,y+.006,.40,rh-.03,wd,fs_max=9.6,fs_min=7.5)
        for (t,x) in zip((kc(g,4),kc(fp,12),kc(nn,12),kc(co,4),kc(ab,16)),(.585,.645,.705,.775,.835)):fig.text(x,y+rh/2-.006,t,fontsize=10,va="center")
        fig.text(.885,y+rh/2-.006,"STRICT PASS"if st else"FAIL",fontsize=9.5,weight="bold",color=okc(st),va="center")
    yb=yh-.03-6*rh
    fig.add_artist(FancyBboxPatch((.05,yb-.115),.90,.11,boxstyle="round,pad=0,rounding_size=0.008",transform=fig.transFigure,facecolor=C_SEALL,edgecolor=C_SEAL,lw=2))
    fit_text(fig,.062,yb-.108,.876,.098,"TEST481 (complete development bank): C_COMPACT and D_RECORD both reached Stage 1 32/32 and linked 32/32, off-target explicit outputs 0/992, off-target NONE 992/992, collisions 0/32, absent-ID 128/128. C_COMPACT, the shorter and less verbal wording, was frozen before the held-out TEST482. Selection evidence is small (n=1 in TEST478, n=4 in TEST480) and used the same four templates.",fs_max=11,fs_min=8.5)
    fit_text(fig,.05,.045,.9,.06,"Only the Stage-2 natural-language readout changed along the way. Memory, forge, cartridges, batch read and full scan were not modified (the TEST477, TEST479 and TEST480 logs state: no new mechanisms / memory motor changes: NONE).",fs_max=10.5,fs_min=8,color=C_NEU)
    foot(fig,ctx,k);return finish(fig,path,ctx,["A_VERIFY","B_DIRECT","C_COMPACT","D_RECORD","E_MINIMAL"])
# ------------------------------------------------------------------ 15 live vs sealed
def p15(ctx,k,path):
    P=ctx["P"];R=ctx["R"];C=R["counts"];S=SEALED482;OFF=R["off_total"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white");H=P["hardware"]
    top=head(fig,"THIS RUN vs THE SEALED TEST482 RECORD","The same 32 records, the same frozen core. The sealed column was read from 482.log; the live column was measured just now.")
    fig.text(.45,top-.015,"THIS LIVE RUN",fontsize=13,weight="bold",color=C_MOD,ha="center");fig.text(.72,top-.015,"SEALED TEST482",fontsize=13,weight="bold",color=C_SEAL,ha="center")
    rows=[("Stage 1 OBJECT→ID",C["s1"],32,S["s1"],True),("LINKED two-hop",C["linked"],32,S["linked"],True),("off-target explicit outputs S1",C["s1_fp"],OFF,S["s1_fp"],False),("off-target explicit outputs S2",C["s2_fp"],OFF,S["s2_fp"],False),
          ("off-target NONE S1",C["s1_none"],OFF,S["s1_none"],True),("off-target NONE S2",C["s2_none"],OFF,S["s2_none"],True),("collisions S1",C["s1_coll"],32,S["s1_coll"],False),("collisions S2",C["s2_coll"],32,S["s2_coll"],False),
          ("unrelated-object NONE",C["obj_none"],R["obj_total"],S["obj"],True),("absent-ID NONE",C["id_none"],R["id_total"],S["idn"],True)]
    rh=(top-.45)/len(rows)
    for i,(lab,v,n,sv,want)in enumerate(rows):
        y=top-.04-(i+1)*rh;good=(v==n)if want else(v==0);fig.text(.05,y+rh*.38,lab,fontsize=11,weight="bold",va="center")
        ax=fig.add_axes([.33,y+rh*.14,.24,rh*.62]);ax.barh([0],[n],color="#E2E8F0");ax.barh([0],[v],color=C_OK if good else C_ERR);ax.set_xlim(0,n);ax.axis("off");ax.text(n/2,0,kc(v,n),ha="center",va="center",fontsize=11.5,weight="bold",color="white"if(v>n*.2)else C_FG)
        ax2=fig.add_axes([.60,y+rh*.14,.24,rh*.62]);ax2.barh([0],[sv[1]],color="#E2E8F0");ax2.barh([0],[sv[0]],color=C_SEAL);ax2.set_xlim(0,sv[1]);ax2.axis("off");ax2.text(sv[1]/2,0,fr(sv),ha="center",va="center",fontsize=11.5,weight="bold",color="white"if(sv[0]>sv[1]*.2)else C_FG)
    yb=.215;lock_ok=LOCK_SHA==SEALED_LOCK_SHA
    fig.add_artist(FancyBboxPatch((.05,yb),.90,.115,boxstyle="round,pad=0,rounding_size=0.008",transform=fig.transFigure,facecolor="white",edgecolor="#CBD5E1",lw=1.4))
    fig.text(.062,yb+.095,"LOCK SHA-256 (TEST482 definition)",fontsize=10.5,weight="bold",color=C_NEU,va="center");fig.text(.062,yb+.068,"recomputed: "+LOCK_SHA,fontsize=8.6,family=MONO,va="center");fig.text(.062,yb+.045,"sealed    : "+SEALED_LOCK_SHA,fontsize=8.6,family=MONO,va="center")
    fig.text(.93,yb+.068,"✓ equal"if lock_ok else"✗ DIFFERENT",fontsize=13,weight="bold",color=okc(lock_ok),ha="right",va="center")
    fig.text(.062,yb+.018,f"hardware: this run {H['this_run_gpu']} · canonical sealed run {H['canonical_sealed_gpu']} · same hardware: {'YES'if H['same']else'NO'}",fontsize=10,va="center",color=C_FG if H["same"]else C_ERR,weight="bold")
    ok=R["strict"];fig.add_artist(FancyBboxPatch((.05,.105),.90,.075,boxstyle="round,pad=0,rounding_size=0.01",transform=fig.transFigure,facecolor=C_OKL if ok else C_ERRL,edgecolor=okc(ok),lw=2.6))
    fig.text(.07,.1425,"LIVE REPLAY VERDICT",fontsize=11.5,weight="bold",color=C_NEU,va="center");fig.text(.27,.1425,R["verdict"],fontsize=16,weight="bold",color=okc(ok),va="center",family=MONO);fig.text(.93,.1425,f"sealed: {S['verdict']}",fontsize=10,color=C_SEAL,va="center",ha="right",family=MONO)
    fit_text(fig,.05,.04,.9,.055,f"Live counts equal the sealed counts: {'YES'if R['replay_match']else'NO'} · gold-ID fallbacks: live {R['fallbacks']}, sealed 0. If this run used a different GPU, any difference from the sealed counts is shown as measured, never adjusted.",fs_max=10.5,fs_min=8,color=C_NEU)
    foot(fig,ctx,k);return finish(fig,path,ctx,[kc(C["s1"],32),kc(C["linked"],32),R["verdict"],S["verdict"],LOCK_SHA,H["this_run_gpu"]])
# ------------------------------------------------------------------ 16 scope and limits
def p16(ctx,k,path):
    fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white");S4=SEALED461
    top=head(fig,"WHAT THIS DEMO DOES — AND DOES NOT — SHOW","Read together with the results. The limits come from the experiments themselves.")
    ch=["Bank typing: 64 cartridges in two typed banks; each stage scans only one bank of 32. The 16-bank Qwen demo used one mixed bank.","Stage-2 readout: C_COMPACT, selected on development data and frozen; the 16-bank demo used a different wording.","Parser: NONE-first, closed ID/place vocabulary.","Source templates differ from the 16-bank demo; F1–F4 labels are not comparable across demos.",f"16-bank sealed record ({S4['log']}): linked {fr(S4['linked'])}, unlinked {fr(S4['unlinked'])}; family {S4['fam_min'][0]} reached {fr(S4['fam_min'][1])} and the gate failed ({S4['verdict']})."]
    lim=["Replay of the sealed held-out panel: not a new test.","New records, same four source templates: no claim about new phrasings.","Closed vocabulary: the parser knows the bank's 32 IDs and 32 places.","Stage banks are chosen by cartridge role (A or B), not by content; full scan inside each bank is O(N); no index, router or signature.","Gold-ID fallback exists in the code (0 uses in sealed TEST482; counted live).","NOMEM is DEMO-ONLY; hashes and the sentinel are integrity indicators, not third-party verification.","Synthetic short facts, one model (Qwen2.5-7B-Instruct), 32 records."]
    for n_,(title,items,col,fc,fs)in enumerate((("WHAT CHANGED SINCE THE 16-BANK QWEN DEMO",ch,C_MOD,C_MODL,10.2),("LIMITS",lim,C_ERR,C_ERRL,10.8))):
        x=.05+n_*.465;fig.add_artist(FancyBboxPatch((x,.10),.435,top-.12,boxstyle="round,pad=0,rounding_size=0.01",transform=fig.transFigure,facecolor=fc,edgecolor=col,lw=2.3));fig.text(x+.015,top-.045,title,fontsize=13,weight="bold",color=col,va="center")
        hh=(top-.20)/len(items)
        for i,t in enumerate(items):fit_text(fig,x+.015,top-.09-(i+1)*hh+.012,.405,hh-.012,"• "+t,fs_max=fs,fs_min=8)
    foot(fig,ctx,k);return finish(fig,path,ctx,["LIMITS","WHAT CHANGED SINCE THE 16-BANK QWEN DEMO",S4["verdict"]])
# ------------------------------------------------------------------ 17 evidence seal
def p17(ctx,k,path):
    P=ctx["P"];R=ctx["R"];E_=P["environment"];tm=P["timing"];cnt=P["counters"];H=P["hardware"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"COMPLETE RUN · EVIDENCE SEAL","Everything below was read from the runtime and is contained in the sealed payload.",tag="THIS LIVE RUN")
    tiles=[("RUN ID",P["run_id"]),("MODEL",MODEL_ID),("GPU · SAME AS SEALED?",f"{E_['gpu']} · {'YES'if H['same']else'NO'}"),("TORCH / TRANSFORMERS",f"{E_['torch']} / {E_['transformers']}"),("BATCH CALLS",str(cnt["batch_calls"])),
           ("SEALED-SCAN ROW READS",str(cnt["scan_row_reads"])),("TEST482 LOCK SHA",LOCK_SHA[:22]+"…"),("TECHNICAL CHECKS",f"{len(P['checks_pre_seal'])} passed"),("PEAK GPU MEMORY",f"{P['gpu']['peak_allocated_gib']:.2f} GiB"),("LIVE VERDICT",R["verdict"])]
    for i,(a,b)in enumerate(tiles):
        r,q=divmod(i,5);x=.05+q*.182;y=top-.125-r*.125;fig.add_artist(FancyBboxPatch((x,y),.172,.105,boxstyle="round,pad=0,rounding_size=0.01",transform=fig.transFigure,facecolor=C_BG,edgecolor=C_MOD,lw=1.8))
        fig.text(x+.008,y+.083,a,fontsize=9,weight="bold",color=C_MOD,va="center");fit_text(fig,x+.008,y+.008,.156,.058,b,fs_max=11.5,fs_min=7.2,family=MONO)
    st=[("model load (startup)",tm["model_load_seconds"]),("codebook",tm["codebook"]),("64 cartridges",tm["forge"]),("Stage 1 scan",tm["stage1_scan"]),("Stage 2 scan",tm["stage2_scan"]),("probes",tm["controls"]),("NOMEM (demo-only)",tm["nomem_demo_only"])]
    ax=fig.add_axes([.22,.30,.70,top-.55]);ax.barh(range(len(st)),[s for _,s in st],color=[C_CTL,C_CART,C_CART,C_MOD,C_MOD,C_UNK,C_CTL]);ax.invert_yaxis();ax.set_yticks(range(len(st)));ax.set_yticklabels([a for a,_ in st],fontsize=10.5)
    for i,(_,s)in enumerate(st):ax.text(s,i,f" {s:.1f} s",va="center",fontsize=10.5,weight="bold")
    ax.set_xlim(0,max(s for _,s in st)*1.18);ax.set_xlabel("measured seconds (sealed TEST482: Stage 1 %.2f s, Stage 2 %.2f s on %s)"%(SEALED482["s1_s"],SEALED482["s2_s"],CANON_GPU),fontsize=9.5);clean_ax(ax)
    txt=f"PAYLOAD FILE : {ctx['payload_name']}\nPAYLOAD SHA-256 : {ctx['sha']}\nCORE : TEST482 (482_qwen.final.py) core functions, unchanged\nIMAGE HASHES : images manifest (per-image SHA-256), written after rendering\nArtifact integrity seal ≠ scientific proof and ≠ independent third-party verification."
    fit_text(fig,.05,.05,.9,.215,txt,fs_max=11.5,fs_min=8,family=MONO)
    foot(fig,ctx,k);return finish(fig,path,ctx,[P["run_id"],E_["gpu"],E_["torch"],E_["transformers"],R["verdict"],str(cnt["batch_calls"])])
POSTERS=[("01_story","What just happened?",p01),("02_memory_bank","The memory bank: 32 A + 32 B",p02),("03_inside_cartridge","What is inside a cartridge?",p03),("04_source_removal","Was the source removed?",p04),("05_audit_locks","Audit locks and frozen model",p05),
 ("06_four_chains","Four chains: object → ID → place",p06),("07_stage1_matrix","Stage 1 matrix 32×32",p07),("08_stage2_matrix","Stage 2 matrix 32×32",p08),("09_off_target","What the other 31 cartridges do",p09),("10_not_stored","If the answer is not stored",p10),
 ("11_nomem_demo_only","NOMEM control (DEMO-ONLY)",p11),("12_families","Results by source family",p12),("13_development_chain","TEST477 to TEST482",p13),("14_readout_selection","How the Stage-2 readout was chosen",p14),
 ("15_live_vs_sealed","This run vs sealed TEST482",p15),("16_scope_limits","Scope, changes and limits",p16),("17_evidence_seal","Evidence seal",p17)]
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
        s.counters=Counter();s.mode=mode;s.n=0;s.rng=np.random.default_rng(1);s.sent="a"*64;s.fp=[1.0]*7;s.A=[];s.B=[];s.nm=None
        s.info=dict(model_id=MODEL_ID,arch=ARCH,dtype="bfloat16",attn="sdpa",gpu="SELF-TEST ENGINE (no GPU)",canonical_gpu=CANON_GPU,gpu_total_gib=0.0,torch="n/a",transformers="n/a",gradio="n/a",python="n/a",platform="n/a",params=1,pad_id=0,eos_ids=[0],
            fp_names=["t0","t1","t2","t3","t4","t5","t6"],model_load_seconds=0.0,startup_utc="n/a",sentinel0=s.sent,fingerprint0=s.fp,hooks0=5,sentinel_method="synthetic")
    def frozen_state(s):
        fh=3 if s.mode=="foreign_hook"else 0;return dict(training=False,trainable_tensors=0,lora=False,optimizer=False,hooks=5+fh,foreign_hooks=fh,foreign_modules=({"__main__":fh}if fh else{}),sentinel=("b"*64 if s.mode=="sentinel_changes"and s.n>20 else s.sent),fingerprint=s.fp)
    def build_codebook(s):s.counters["core_forge_passes"]+=32;return dict(sentences=32,svds=ARCH[0]*2*ARCH[3],seconds=0.0,bytes=1)
    def codebook_bytes(s):return 1
    def _tel(s,i,ex):
        T=14+(i*7)%7;r=s.rng;t=dict(T=T,cosK=list(.985+.014*r.random(28)),cosV=list(.998+.002*r.random(28)),own_exact=True,code_numbers=28*(T-1)*4*(K_DIM+V_DIM),own_numbers=2*28*512,native_numbers=2*28*512*T)
        if ex:t["excerpt"]=dict(layer=14,head=0,tensor="K",values=(r.normal(0,1,(T-1,120))*np.exp(-np.arange(120)/60)).round(4).tolist())
        return t
    def forge_one(s,i):
        s.counters["core_forge_passes"]+=2;s.counters["telemetry_reforge_passes"]+=2;s.A.append(i);s.B.append(i);return dict(A=s._tel(i,i==0),B=s._tel(i+1,False),secs=0.0)
    def make_nomem(s):s.nm=True
    def card_ptrs(s):return list(range(128))
    def card_finite(s):return True
    def q_tokens(s,q):return len(q.split())
    def read_A(s,q):
        s.counters["batch_calls"]+=1;s.n+=1;o=re.search(r"contains the (.+?) according",q).group(1)
        rows=[(r[2]if r[1]==o else"NONE\nsynthetic")for r in RECORDS]
        if s.mode=="fallback"and o==RECORDS[3][1]:rows=["NONE\nsynthetic"]*32
        return rows
    def read_B(s,q):
        s.counters["batch_calls"]+=1;s.n+=1;c=re.search(r'Container "(.+?)" location',q).group(1)
        return[((("NONE\n"+r[3])if(s.mode=="fail_one"and r[2]==RECORDS[9][2])else r[3])if r[2]==c else"NONE\nsynthetic")for r in RECORDS]
    def read_NM(s,q):s.counters["batch_calls"]+=1;return["NONE"]
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
READY_HTML=card_html("Ready",f"Press <b>RUN 32-BANK CARTRIDGE DEMO</b>. The run replays the sealed TEST482 panel: it forges 64 cartridges (32 A + 32 B), removes the source, scans the 32 A cartridges for each object and the 32 B cartridges for each ID, checks the controls ({TOTAL_CALLS} batch calls), seals the evidence and renders {N_POSTERS} JPG posters plus one ZIP.<br>Canonical sealed hardware: <b>{CANON_GPU}</b>; this run reports the GPU actually used. NOMEM is DEMO-ONLY.<br><b>Downloads appear when the run is complete.</b>","info")
def is_cuda_error(ex):
    s=f"{type(ex).__name__} {ex}".lower();return"cuda"in s or"device-side assert"in s or"cublas"in s
def run_handler():
    global RUN_COUNTER
    if not GPU_LOCK.acquire(blocking=False):
        yield pack(card_html("Busy","Another run is in progress. Please try again in a moment.","warn"));return
    stage_name="initialisation"
    try:
        RUN_COUNTER+=1;run_id=f"QWEN32-{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}-{RUN_COUNTER:04d}"
        prune_runs(2);run_dir=ROOT/run_id;run_dir.mkdir(parents=True,exist_ok=True);gen=execute_run(ENGINE,dict(run_id=run_id,run_dir=run_dir));first_ev=True
        while True:
            try:e=next(gen)
            except StopIteration as s:B=s.value;break
            stage_name=f"{e['stage']}/{N_STAGES} {e['title']}"
            yield pack(stage_card(e),[],None,None,None)if first_ev else pack(stage_card(e));first_ev=False
        R=B["R"];C=R["counts"];H_=B["P"]["hardware"];ok=R["verdict"].startswith("PASS");OFF=R["off_total"]
        gallery_items=[(str(p),c_)for p,c_ in B["imgs"]]
        raw_texts={"txt":B["txt"].read_text(encoding="utf-8"),"json":B["payload"].read_bytes().decode("utf-8"),"man":B["manifest"].read_text(encoding="utf-8")}
        body=(f"<b>{html.escape(run_id)}</b><br>THIS LIVE REPLAY: Stage 1 {C['s1']}/32 · linked {C['linked']}/32 · off-target false outputs {C['s1_fp']}/{OFF} + {C['s2_fp']}/{OFF} · off-target NONE {C['s1_none']}/{OFF} + {C['s2_none']}/{OFF} · collisions {C['s1_coll']}/32 + {C['s2_coll']}/32<br>"
              f"verdict: <b>{R['verdict']}</b> · sealed TEST482: {SEALED482['verdict']} · live counts equal sealed counts: {'YES'if R['replay_match']else'NO'} · gold-ID fallbacks: {R['fallbacks']}<br>"
              f"GPU: {html.escape(H_['this_run_gpu'])} · canonical sealed hardware {CANON_GPU} · same: {'YES'if H_['same']else'NO'}<br>NOMEM (DEMO-ONLY, not part of the sealed result): strict hits {R['nomem']['s1_hits']}/32 + {R['nomem']['s2_hits']}/32<br>"
              f"{N_POSTERS} JPGs + ZIP · {B['checks']}/{B['checks']} technical checks PASS<br>"
              f'<span class="mono">payload SHA-256 (artifact integrity seal, not third-party verification): {B["sha"]}</span>')
        yield pack(card_html(f"{N_STAGES}/{N_STAGES} · Complete",body,"on"if ok else"err"),gallery_items,{"zip":B["zip"],"json":B["payload"],"txt":B["txt"],"man":B["manifest"]},[p for p,_ in B["imgs"]],raw_texts)
    except Exception as ex:
        print("="*110);print(f"QWEN32 RUN FAILED — stage: {stage_name}");print(f"exception type : {type(ex).__name__}");print(f"exception message: {ex}");traceback.print_exc();print("="*110)
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
        assert len(B["imgs"])==N_POSTERS and file_ready(B["zip"])and B["verdict"]=="PASS_REPLAY_OF_TEST482"
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
HERO=('<div class="kz hero"><div class="brand">AKBASCORE NIRVANA</div><div class="sub">QWEN · 32-BANK COGNITIVE CARTRIDGE</div>'
      '<div class="tag">Knowledge goes in. The source goes away. The memory remains.<br>'
      f'32 synthetic records become 64 numerical cartridges inside a frozen {MODEL_SHORT}. The source text is removed. 32 locked questions are answered by scanning the cartridges — object to ID to place. Sealed TEST482 replay.</div></div>')
def _tb(**kw):
    try:return gr.Textbox(show_copy_button=True,**kw)
    except TypeError:return gr.Textbox(**kw)
def _gallery(**kw):
    try:return gr.Gallery(columns=1,height="auto",object_fit="contain",preview=False,**kw)
    except TypeError:return gr.Gallery(**kw)
try:_blocks=gr.Blocks(css=CSS,title="AKBASCORE NIRVANA · QWEN · 32-BANK COGNITIVE CARTRIDGE")
except TypeError:_blocks=gr.Blocks(title="AKBASCORE NIRVANA · QWEN · 32-BANK COGNITIVE CARTRIDGE")
with _blocks as demo:
    gr.HTML(HERO)
    run_btn=gr.Button("RUN 32-BANK CARTRIDGE DEMO",variant="primary",elem_id="kz_run")
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
say("[7/7] Open the printed gradio.live link in a new tab and press RUN 32-BANK CARTRIDGE DEMO.")
demo.launch(share=True,allowed_paths=ALLOWED_PATHS,show_error=True,inline=False)
