# ================================================================================================
# AKBASCORE - TEST 352
# VALUE INFORMATION LOSS X-RAY
# SOURCE VALUE -> SPAN -> RELATION -> QUERY ALIGNMENT -> ARGMAX -> PACKET
# TEST351 COUNTERFACTUAL PAIRS | NO INJECTION SEARCH | TARGET POST-HOC ONLY
# ================================================================================================
import os,sys,json,time,random,hashlib,subprocess,importlib.util,re
from datetime import datetime,timezone
from pathlib import Path
for m,p in [("torch","torch"),("transformers","transformers"),("numpy","numpy")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required")
SEED=352;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYS_STD="You are a concise reasoning assistant. Use only the information in the prompt."
TOTAL=28;H=3584;EPS=1e-8
ROOT=Path("/content/AKBASCORE_TEST352");ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
PAIRS=[
("PERSON","During the final inspection, technician Bravik disconnected the auxiliary compressor before the alarms were tested.","During the final inspection, technician Talora disconnected the auxiliary compressor before the alarms were tested.","Who disconnected the auxiliary compressor?","Bravik","Talora"),
("PERSON","The final alignment of the measurement rig was performed by Talora after calibration.","The final alignment of the measurement rig was performed by Veyron after calibration.","Who performed the final alignment?","Talora","Veyron"),
("PLACE","The recovered sensor was unloaded in Dubrovnik after the research vessel completed its route.","The recovered sensor was unloaded in Bratislava after the research vessel completed its route.","Where was the recovered sensor unloaded?","Dubrovnik","Bratislava"),
("PLACE","The bronze disk was stored in Nagoya before the archaeological team continued onward.","The bronze disk was stored in Dubrovnik before the archaeological team continued onward.","Where was the bronze disk stored?","Nagoya","Dubrovnik"),
("COLOR","After synchronization, the status panel changed to magenta.","After synchronization, the status panel changed to scarlet.","What color did the status panel become after synchronization?","magenta","scarlet"),
("COLOR","Spectral analysis established the coating as indigo.","Spectral analysis established the coating as magenta.","What was the coating's established color?","indigo","magenta"),
("MATERIAL","The engineers selected obsidian for the protective shell after testing several materials.","The engineers selected tungsten for the protective shell after testing several materials.","Which material was selected for the protective shell?","obsidian","tungsten"),
("MATERIAL","The laboratory chose tungsten for the central support after evaluation.","The laboratory chose feldspar for the central support after evaluation.","Which material was chosen for the central support?","tungsten","feldspar"),
("ACTION","Near the access hatch, Tavren began crawling beneath the support frame.","Near the access hatch, Tavren began descending beneath the support frame.","What did Tavren begin doing?","crawling","descending"),
("ACTION","When the retaining clamp was released, the membrane began expanding.","When the retaining clamp was released, the membrane began descending.","What did the membrane begin doing?","expanding","descending"),
("NAME","The recovered mineral was entered into the catalogue under the designation Norvex.","The recovered mineral was entered into the catalogue under the designation Dorevian.","What designation was given to the recovered mineral?","Norvex","Dorevian"),
("NAME","The prototype's production name was changed to Kelvane.","The prototype's production name was changed to Norvex.","What became the prototype's production name?","Kelvane","Norvex"),
("NUMBER","After recalculation, the certified value was 817.","After recalculation, the certified value was 583.","What was the certified value after recalculation?","817","583"),
("OBJECT","From the equipment locker, Selma retrieved a compass.","From the equipment locker, Selma retrieved a hygrometer.","What object did Selma retrieve?","compass","hygrometer"),
("PROPERTY","After treatment, the sample was described as glossy.","After treatment, the sample was described as velvety.","How was the treated sample described?","glossy","velvety"),
("SHOWCASE","Mustafa Akbaş placed an amber banner beneath the Galata Bridge where the validation demonstration was conducted.","Mustafa Akbaş raised an amber banner beneath the Galata Bridge where the validation demonstration was conducted.","What did Mustafa Akbaş do at the location of the validation demonstration?","placed an amber banner","raised an amber banner")]
QNULLS=["What information is requested?","Which unknown detail should be recovered?","Answer the question using the missing information.","What should be identified?","Which detail is being asked about?"]
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def cos(a,b):return float(torch.dot(unit(a),unit(b)))
def rel(a,b):return float((a-b).norm()/((a.norm()+b.norm())*.5).clamp_min(EPS))
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
def prompt(x):return tok.apply_chat_template([{"role":"system","content":SYS_STD},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
def enc(x):return tok(prompt(x),return_tensors="pt",add_special_tokens=False).to("cuda")
print("="*108);print("TEST 352 - VALUE INFORMATION LOSS X-RAY");print("="*108);print("START:",START)
print("[1/18] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers
if len(layers)!=TOTAL or model.config.hidden_size!=H:raise RuntimeError("Architecture mismatch")
torch.cuda.synchronize();print(f"OK | {MODEL_ID} | 28L | H={H} | {next(model.parameters()).dtype} | {time.perf_counter()-t:.2f}s")
print("[2/18] MODEL LOCK")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();print("FP:",[f"{x:.4f}" for x in FP0])
@torch.inference_mode()
def states(text):
    o=model(**enc(text),use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)]
@torch.inference_mode()
def source_data(text):
    e=tok(text,return_tensors="pt",add_special_tokens=False,return_offsets_mapping=True);offs=e["offset_mapping"][0].tolist()
    o=model(input_ids=e["input_ids"].to("cuda"),attention_mask=e["attention_mask"].to("cuda"),use_cache=False,output_hidden_states=True,return_dict=True);out=[]
    for m in re.finditer(r"\b[\w'-]+\b",text,re.UNICODE):
        a,b=m.span();ix=[j for j,(s,z) in enumerate(offs) if z>a and s<b]
        if ix:out.append({"text":m.group(0),"start":a,"end":b,"v":[o.hidden_states[L+1][0,ix].float().mean(0).detach().clone() for L in range(TOTAL)]})
    return out
def find_value_span(sp,value):
    words=re.findall(r"\b[\w'-]+\b",value,re.UNICODE)
    if not words:raise RuntimeError("Empty value")
    for i in range(len(sp)-len(words)+1):
        if [x["text"].lower() for x in sp[i:i+len(words)]]==[x.lower() for x in words]:
            return [torch.stack([sp[k]["v"][L] for k in range(i,i+len(words))]).mean(0) for L in range(TOTAL)],i
    raise RuntimeError(f"Value span not found: {value}")
print("[3/18] QUERY NULL CACHE")
QN=[states(x) for x in QNULLS];print("NULL READY")
print("[4/18] QUERY DIRECTIONS")
QV=[]
for i,(_,_,_,q,_,_) in enumerate(PAIRS):
    qh=states(q);qv=[unit(torch.stack([unit(qh[L]-QN[j][L]) for j in range(len(QN))]).mean(0)) for L in range(TOTAL)];QV.append(qv);print(f"{i+1:02d}/16 | READY")
print("[5/18] SOURCE + VALUE SPANS")
SRC=[];VAL=[]
for i,(typ,sa,sb,q,va,vb) in enumerate(PAIRS):
    spa=source_data(sa);spb=source_data(sb);vsa,ia=find_value_span(spa,va);vsb,ib=find_value_span(spb,vb);SRC.append((spa,spb));VAL.append((vsa,vsb))
    print(f"{i+1:02d}/16 | {typ:8s} | A={va!r}@{ia} | B={vb!r}@{ib}")
print("[6/18] STAGE-1 VALUE SPAN SEPARATION")
S1=[]
for i,(typ,sa,sb,q,va,vb) in enumerate(PAIRS):
    a,b=VAL[i];c=[cos(a[L],b[L]) for L in range(TOTAL)];d=[rel(a[L],b[L]) for L in range(TOTAL)]
    S1.append({"cos":c,"rel":d});print(f"{i+1:02d}/16 | COS L0={c[0]:+.4f} L12={c[12]:+.4f} L25={c[25]:+.4f} L27={c[27]:+.4f} | REL25={d[25]:.4f}")
print("[7/18] STAGE-2 RELATION CANDIDATE CONSTRUCTION")
CANDS=[]
for i,(typ,sa,sb,q,va,vb) in enumerate(PAIRS):
    pair=[]
    for sp in SRC[i]:
        per_anchor=[]
        for a in range(len(sp)):
            per_layer=[]
            for L in range(TOTAL):
                cand=[]
                for b in range(len(sp)):
                    if b==a:continue
                    d=sp[b]["v"][L]-sp[a]["v"][L];cand.append({"b":b,"text":sp[b]["text"],"d":d,"score":float(torch.dot(unit(d),QV[i][L]))})
                per_layer.append(cand)
            per_anchor.append(per_layer)
        pair.append(per_anchor)
    CANDS.append(pair);print(f"{i+1:02d}/16 | A/B RELATIONS READY")
print("[8/18] STAGE-3 ARGMAX RELATION + QUERY ALIGNMENT")
SELREL=[]
for i,(typ,sa,sb,q,va,vb) in enumerate(PAIRS):
    sides=[]
    for side in range(2):
        sp=SRC[i][side];anchors=[]
        for a in range(len(sp)):
            ds=[];scores=[];partners=[]
            for L in range(TOTAL):
                best=max(CANDS[i][side][a][L],key=lambda x:x["score"]);ds.append(unit(best["d"]));scores.append(best["score"]);partners.append(best["text"])
            anchors.append({"anchor":sp[a]["text"],"d":ds,"score":scores,"partner":partners})
        sides.append(anchors)
    SELREL.append(sides);print(f"{i+1:02d}/16 | READY")
print("[9/18] STAGE-4 FROZEN ARGMAX(DISP) EQUIVALENT ROUTER")
# TEST351 DISP selection is reproduced exactly by evaluating the already-built packet through the same frozen motor.
SYS_FORCE="Read the answer from the internal state. Return exactly one short answer. Never explain, refuse, say information is missing, or mention the prompt."
TARGET_RSS=.250235055;IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20;END=25
raw=[IVME*(ZIRVE*np.exp(-SONUM*L)*(1+SONUM*L)+TABAN)/(ZIRVE+TABAN) for L in range(END+1)];scale=TARGET_RSS/np.sqrt(sum(x*x for x in raw));RHO=[x*scale for x in raw]
def enc_force(q):
    p=tok.apply_chat_template([{"role":"system","content":SYS_FORCE},{"role":"user","content":q}],tokenize=False,add_generation_prompt=True)
    return tok(p,return_tensors="pt",add_special_tokens=False).to("cuda")
ACTIVE=set()
def install(packet,apos):
    hs=[]
    for L in range(END+1):
        rho=RHO[L]
        def hk(mod,args,out,L=L,rho=rho):
            y=out[0] if isinstance(out,tuple) else out;r=y.clone()
            for idx in range(max(0,apos),y.shape[1]):
                z=y[:,idx,:].float();zn=z.norm(dim=-1,keepdim=True).clamp_min(EPS);r[:,idx,:]=(z+packet[L].to(z.device).view(1,-1)*zn*rho).to(y.dtype)
            return (r,)+out[1:] if isinstance(out,tuple) else r
        h=layers[L].register_forward_hook(hk);hs.append(h);ACTIVE.add(id(h))
    return hs
@torch.inference_mode()
def run(ids,packet=None,apos=None):
    hs=install(packet,apos) if packet is not None else []
    try:return model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,output_hidden_states=True,return_dict=True)
    finally:
        for h in hs:h.remove();ACTIVE.discard(id(h))
def term(o):return o.hidden_states[28][0,-1].float()
ROUTE=[]
for i,(typ,sa,sb,q,va,vb) in enumerate(PAIRS):
    ids=enc_force(q)["input_ids"];apos=ids.shape[1]-2;vh=term(run(ids));sides=[]
    for side in range(2):
        ds=[]
        for x in SELREL[i][side]:
            o=run(ids,x["d"],apos);ds.append(float((term(o)-vh).norm()/vh.norm().clamp_min(EPS)))
        j=int(np.argmax(ds));sides.append({"j":j,"disp":ds[j],"anchor":SELREL[i][side][j]["anchor"],"packet":SELREL[i][side][j]["d"],"scores":SELREL[i][side][j]["score"],"partners":SELREL[i][side][j]["partner"]})
    ROUTE.append(sides);print(f"{i+1:02d}/16 | A={sides[0]['anchor']!r}:{sides[0]['disp']:.4f} | B={sides[1]['anchor']!r}:{sides[1]['disp']:.4f}")
print("[10/18] STAGE-5 FINAL PACKET A/B SEPARATION")
S5=[]
for i in range(len(PAIRS)):
    a,b=ROUTE[i][0]["packet"],ROUTE[i][1]["packet"];c=[cos(a[L],b[L]) for L in range(TOTAL)];d=[float((a[L]-b[L]).norm()) for L in range(TOTAL)]
    S5.append({"cos":c,"diff":d});print(f"{i+1:02d}/16 | COS L0={c[0]:+.4f} L12={c[12]:+.4f} L25={c[25]:+.4f} | DIFF25={d[25]:.4f}")
print("[11/18] LOSS CHAIN")
ROWS=[]
for i,(typ,sa,sb,q,va,vb) in enumerate(PAIRS):
    rv=np.array(S1[i]["rel"][:26]);pd=np.array(S5[i]["diff"][:26]);qa=np.array(ROUTE[i][0]["scores"][:26]);qb=np.array(ROUTE[i][1]["scores"][:26]);qd=np.abs(qa-qb)
    value_sep=float(rv.mean());packet_sep=float(pd.mean());retain=packet_sep/(value_sep+EPS);qdiff=float(qd.mean())
    same_anchor=ROUTE[i][0]["anchor"].lower()==ROUTE[i][1]["anchor"].lower();same_partner=float(np.mean([a.lower()==b.lower() for a,b in zip(ROUTE[i][0]["partners"][:26],ROUTE[i][1]["partners"][:26])]))
    r={"i":i+1,"type":typ,"value_a":va,"value_b":vb,"value_span_rel":value_sep,"packet_diff":packet_sep,"retention_ratio":retain,"query_score_diff":qdiff,"same_anchor":same_anchor,"same_partner_frac":same_partner,"l25_value_rel":S1[i]["rel"][25],"l25_packet_diff":S5[i]["diff"][25]};ROWS.append(r)
    print(f"{i+1:02d}/16 | {typ:8s} | VALUE={value_sep:.4f} -> PACKET={packet_sep:.4f} | RET={retain:.3f} | QΔ={qdiff:.4f} | ANCHOR_EQ={same_anchor} PARTNER_EQ={same_partner:.2f}")
print("[12/18] DEVELOPMENT SUMMARY")
D=ROWS[:15]
VSEP=float(np.mean([r["value_span_rel"] for r in D]));PSEP=float(np.mean([r["packet_diff"] for r in D]));RET=float(np.mean([r["retention_ratio"] for r in D]));ANCH=float(np.mean([r["same_anchor"] for r in D]));PART=float(np.mean([r["same_partner_frac"] for r in D]));ZERO=float(np.mean([r["packet_diff"]<1e-6 for r in D]));L25ZERO=float(np.mean([r["l25_packet_diff"]<1e-6 for r in D]))
print(f"N=15 | VALUE_SEP={VSEP:.6f} | PACKET_SEP={PSEP:.6f} | RETENTION={RET:.6f} | SAME_ANCHOR={ANCH:.6f} | SAME_PARTNER={PART:.6f} | PACKET_ZERO={ZERO:.6f} | L25_ZERO={L25ZERO:.6f}")
print("[13/18] LAYERWISE RETENTION")
LR=[]
for L in range(26):
    v=float(np.mean([S1[i]["rel"][L] for i in range(15)]));p=float(np.mean([S5[i]["diff"][L] for i in range(15)]));r=p/(v+EPS);LR.append(r)
    print(f"L{L:02d} | VALUE_REL={v:.5f} | PACKET_DIFF={p:.5f} | RET={r:.4f}")
print("[14/18] PREDECLARED LOSS LOCALIZATION")
EARLY=float(np.mean(LR[:9]));MID=float(np.mean(LR[9:18]));LATE=float(np.mean(LR[18:26]))
ROUTER_COLLAPSE=ANCH>=.70 and PART>=.70
PACKET_COLLAPSE=RET<.50 or L25ZERO>=.50
VALUE_PRESENT=VSEP>.05
print(f"VALUE_PRESENT={VALUE_PRESENT} | ROUTER_COLLAPSE={ROUTER_COLLAPSE} | PACKET_COLLAPSE={PACKET_COLLAPSE} | RET_EARLY={EARLY:.3f} MID={MID:.3f} LATE={LATE:.3f}")
print("[15/18] SHOWCASE")
r=ROWS[15];print("-"*108);print("A:",PAIRS[15][1]);print("B:",PAIRS[15][2]);print("Q:",PAIRS[15][3]);print(f"VALUE_SPAN_REL={r['value_span_rel']:.4f} | PACKET_DIFF={r['packet_diff']:.4f} | RET={r['retention_ratio']:.3f}");print(f"ANCHORS={ROUTE[15][0]['anchor']!r}/{ROUTE[15][1]['anchor']!r} | L25 VALUE={r['l25_value_rel']:.4f} PACKET={r['l25_packet_diff']:.4f}");print("-"*108)
print("[16/18] CLASSIFICATION")
if VALUE_PRESENT and ROUTER_COLLAPSE:CLASS="VALUE_INFORMATION_PRESENT_BUT_COLLAPSES_AT_RELATION_ROUTING"
elif VALUE_PRESENT and PACKET_COLLAPSE:CLASS="VALUE_INFORMATION_PRESENT_BUT_COLLAPSES_DURING_PACKET_CONSTRUCTION"
elif VALUE_PRESENT and RET>=.50:CLASS="VALUE_INFORMATION_SURVIVES_COMPILER_BUT_IS_NOT_EFFECTIVELY_BOUND"
else:CLASS="VALUE_INFORMATION_WEAK_BEFORE_PACKET_CONSTRUCTION"
print("CLASS=",CLASS)
print("[17/18] SCIENTIFIC AUDIT")
print("TEST351 PAIRS | TEST351 COMPILER | TEST351 ARGMAX(DISP) | TARGET POST-HOC LABELS ONLY | NO TRAINING | NO WEIGHT UPDATE | X-RAY DIAGNOSTIC")
print("[18/18] INTEGRITY + SAVE")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("Hook leak")
S={"n":15,"value_sep":VSEP,"packet_sep":PSEP,"retention":RET,"same_anchor":ANCH,"same_partner":PART,"packet_zero":ZERO,"l25_zero":L25ZERO,"ret_early":EARLY,"ret_mid":MID,"ret_late":LATE}
R={"schema":"akbascore.test352.v1","test":"TEST 352","start":START,"end":utc(),"model":MODEL_ID,"summary":S,"classification":CLASS,"rows":ROWS,"layer_retention":LR,"integrity":{"target_engine_access":False,"target_posthoc_only":True,"training":False,"weight_update":False,"fp_start":FP0,"fp_end":fp(),"hooks_remaining":len(ACTIVE)}}
sha=hashlib.sha256(canon(R)).hexdigest();rid=f"T352-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json";tp=ROOT/f"{rid}.txt"
jp.write_text(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False),encoding="utf-8")
tp.write_text("\n".join(["="*108,"TEST 352 - VALUE INFORMATION LOSS X-RAY","="*108]+[f"{k}={v}" for k,v in S.items()]+[f"CLASS={CLASS}",f"SHA={sha}"]),encoding="utf-8")
print("="*108);print("TEST 352 COMPLETE");print(f"N=15 | VALUE_SEP={VSEP:.3f} | PACKET_SEP={PSEP:.3f} | RET={RET:.3f} | SAME_ANCHOR={ANCH:.3f} | SAME_PARTNER={PART:.3f} | L25_ZERO={L25ZERO:.3f}");print(f"RET EARLY={EARLY:.3f} MID={MID:.3f} LATE={LATE:.3f}");print("CLASS=",CLASS);print("WEIGHTS FROZEN | HOOKS=0");print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
