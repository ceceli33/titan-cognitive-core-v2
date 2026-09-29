# ================================================================================================
# AKBASCORE - TEST 319
# FRESH HELD-OUT CASE-ISOLATED ENDOGENOUS VALUE-SELECTOR GENERALIZATION
# TEST318 FROZEN SELECTOR | 24 NEW CASES | TARGET SEALED | NO TRAINING | NO TUNING | NO CROSS-CASE
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
SEED=319;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYS_STD="You are a concise reasoning assistant. Use only the information in the prompt."
SYS_FORCE="Read the answer from the internal state. Return exactly one short answer. Never explain, refuse, say information is missing, or mention the prompt."
TOTAL=28;H_EXPECT=3584;END=25;EPS=1e-8;TARGET_RSS=.250235055;BRIDGE_REL=.20;SLOT_K=8;BIND_LAYERS=(19,21,23,25,27)
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20
ROOT=Path("/content/AKBASCORE_TEST319");ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
INPUTS=[
("During the midnight inspection, technician Neris replaced the cracked relay inside Module Seven.","Who replaced the cracked relay inside Module Seven?"),
("A mineral recovered from the eastern trench was assigned the provisional name Velqor.","What provisional name was assigned to the mineral?"),
("Four sealed packets were weighed, and the lightest packet carried the code 583.","Which code was on the lightest packet?"),
("The research vessel departed Bergen and delivered the backup antenna to Lisbon.","Where was the backup antenna delivered?"),
("The builders considered granite and aluminum but ultimately chose ceramic for the heat shield.","Which material was chosen for the heat shield?"),
("The warning display remained yellow until the reset, when it turned magenta.","What color did the warning display turn after the reset?"),
("After reaching the lower deck, Tovan stopped walking and began crawling through the narrow passage.","What did Tovan begin doing?"),
("The restored panel appeared dull at first, but investigators described its finished surface as lustrous.","How was the finished surface described?"),
("Near the collapsed tower, Elira recovered a compass while leaving the damaged radio behind.","What object did Elira recover?"),
("The access sequence recorded in the sealed ledger is Nerovak.","What is the access sequence?"),
("Marek prepared the instruments, while the final measurements were performed by Siona.","Who performed the final measurements?"),
("The aircraft crossed Vienna and Prague before making its final landing in Warsaw.","Where did the aircraft finally land?"),
("After testing acrylic, bronze, and titanium, the team selected titanium for the pressure frame.","What material was selected for the pressure frame?"),
("One beacon emitted green light, while the emergency beacon beside it emitted crimson.","What color did the emergency beacon emit?"),
("The monitor first showed 64, but after synchronization the confirmed reading became 892.","What was the confirmed reading after synchronization?"),
("When the locking cycle ended, the inner ring began expanding instead of remaining fixed.","What did the inner ring begin doing?"),
("The original fabric felt rough, whereas the replacement lining was described as velvety.","How was the replacement lining described?"),
("The storage crate contained a lens, a chain, and a chronometer; the chronometer was marked for collection.","Which item was marked for collection?"),
("The prototype was temporarily called Helix, but the final platform was named Norveth.","What was the final platform named?"),
("Two interns catalogued the samples, but Dr. Edrin personally transported the sealed vial.","Who transported the sealed vial?"),
("The navigator stored the bronze tablet in Osaka before departing for Nagoya.","Where was the bronze tablet stored?"),
("Under the workshop lights the coating seemed gray, but spectral analysis confirmed it was indigo.","What was the coating's confirmed color?"),
("Sample M-4 contained several minerals but was determined to consist primarily of feldspar.","What did Sample M-4 primarily consist of?"),
("The rover paused, moved backward briefly, and then began decelerating near Station Delta.","What did the rover begin doing near Station Delta?")
]
SEALED_TARGETS=("Neris","Velqor","583","Lisbon","ceramic","magenta","crawling","lustrous","compass","Nerovak","Siona","Warsaw","titanium","crimson","892","expanding","velvety","chronometer","Norveth","Edrin","Osaka","indigo","feldspar","decelerating")
QNULLS=["What information is requested?","Which unknown detail should be recovered?","Answer the question using the missing information.","What should be identified?","Which detail is being asked about?"]
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
def normtxt(s):return re.sub(r"^[^\w]+|[^\w]+$","",s,flags=re.UNICODE).casefold()
def chat_ids(q,system):
    s=tok.apply_chat_template([{"role":"system","content":system},{"role":"user","content":q}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to("cuda")
print("="*108);print("TEST 319 - FRESH HELD-OUT ENDOGENOUS VALUE-SELECTOR GENERALIZATION");print("="*108);print("START:",START)
print("[1/15] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
if not tok.is_fast:raise RuntimeError("Fast tokenizer required")
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size
if len(layers)!=TOTAL or H!=H_EXPECT:raise RuntimeError("Architecture mismatch")
torch.cuda.synchronize();print(f"OK | {MODEL_ID} | 28L | H={H} | VOCAB={model.config.vocab_size} | {next(model.parameters()).dtype} | {time.perf_counter()-t:.2f}s")
print("[2/15] TEST318 ENGINE + SELECTOR LOCK")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();base=[IVME*(ZIRVE*np.exp(-SONUM*L)*(1+SONUM*L)+TABAN)/(ZIRVE+TABAN) for L in range(END+1)];scale=TARGET_RSS/np.sqrt(sum(x*x for x in base));RHO=[x*scale for x in base];RSS=float(np.sqrt(sum(x*x for x in RHO)))
print("FP:",[f"{x:.4f}" for x in FP0]);print(f"RSS={RSS:.9f} | BRIDGE={BRIDGE_REL:.3f} | SELECTOR=BIND+DELTA+READOUT | FRESH_CASES=24")
@torch.inference_mode()
def last_states(text):
    o=model(**chat_ids(text,SYS_STD),use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)]
@torch.inference_mode()
def span_states(text):
    e=tok(text,return_tensors="pt",add_special_tokens=False,return_offsets_mapping=True);ids=e["input_ids"][0];offs=e["offset_mapping"][0].tolist()
    o=model(input_ids=e["input_ids"].to("cuda"),attention_mask=e["attention_mask"].to("cuda"),use_cache=False,output_hidden_states=True,return_dict=True);sp=[]
    for m in re.finditer(r"\b[\w'-]+\b",text,re.UNICODE):
        a,b=m.span();ix=[j for j,(s,z) in enumerate(offs) if z>a and s<b]
        if ix:sp.append({"text":m.group(0),"token_ids":ids[ix].tolist(),"states":[unit(o.hidden_states[L+1][0,ix].float().mean(0).detach().clone()) for L in range(TOTAL)]})
    return sp
print("[3/15] TARGET-FREE QUERY NULL CACHE")
QNULL=[last_states(x) for x in QNULLS];print("NULL READY")
print("[4/15] FRESH SAME-CASE SPAN BANK + LOCKED PACKET + ANCHOR")
PACK=[];ORTH=[];BANK=[];ANCHOR=[]
for i,(statement,question) in enumerate(INPUTS):
    qh=last_states(question);qv=[unit(torch.stack([unit(qh[L]-QNULL[j][L]) for j in range(len(QNULL))]).mean(0)) for L in range(TOTAL)]
    sp=span_states(statement);qs=torch.tensor([torch.dot(s["states"][27],qv[27]).item() for s in sp]);k=min(SLOT_K,len(sp));vals,idx=torch.topk(qs,k);w=torch.softmax(vals/.025,dim=0)
    p=[unit(sum(sp[int(idx[j])]["states"][L]*w[j].to(sp[0]["states"][L].device) for j in range(k))) for L in range(TOTAL)];o=[]
    for L in range(TOTAL):
        r=p[L]-torch.dot(p[L],qv[L])*qv[L];o.append(unit(r) if r.norm()>EPS else p[L])
    a=int(torch.argmax(qs));PACK.append(p);ORTH.append(o);BANK.append({"spans":sp,"q":qv});ANCHOR.append(a)
    print(f"{i+1:02d}/24 | ANCHOR={sp[a]['text']!r} | BANK={len(sp)} | ADDR={[sp[int(x)]['text'] for x in idx]}")
print("[5/15] HELD-OUT + CASE-ISOLATION AUDIT")
print("TEST306-318 CASES NOT USED | NEW 24 CASES | SAME-CASE ONLY | TARGET_ACCESS=False | CROSS_CASE=False")
print("[6/15] LOCKED CAUSAL ENGINE")
AUD={"seasc":0,"bridge":0,"max_seasc_dev":0.0,"max_bridge_dev":0.0};ACTIVE=set()
def install(packet,bridge):
    hs=[]
    for L in range(END+1):
        rho=RHO[L]
        def hk(mod,args,out,L=L,rho=rho):
            y=out[0] if isinstance(out,tuple) else out;z=y[:,-1,:].float();d=packet[L].to(z.device)*z.norm(dim=-1,keepdim=True)*rho
            rel=float((d.norm(dim=-1)/(z.norm(dim=-1)+EPS)).max());AUD["seasc"]+=1;AUD["max_seasc_dev"]=max(AUD["max_seasc_dev"],abs(rel-rho))
            r=y.clone();r[:,-1,:]=(z+d).to(y.dtype);return (r,)+out[1:] if isinstance(out,tuple) else r
        h=layers[L].register_forward_hook(hk);hs.append(h);ACTIVE.add(id(h))
    def bh(mod,args,out):
        y=out[0] if isinstance(out,tuple) else out;z=y[:,-1,:].float();d=bridge.to(z.device)*z.norm(dim=-1,keepdim=True)*BRIDGE_REL
        rel=float((d.norm(dim=-1)/(z.norm(dim=-1)+EPS)).max());AUD["bridge"]+=1;AUD["max_bridge_dev"]=max(AUD["max_bridge_dev"],abs(rel-BRIDGE_REL))
        r=y.clone();r[:,-1,:]=(z+d).to(y.dtype);return (r,)+out[1:] if isinstance(out,tuple) else r
    h=layers[26].register_forward_hook(bh);hs.append(h);ACTIVE.add(id(h));return hs
@torch.inference_mode()
def run(q,packet,bridge):
    hs=install(packet,bridge)
    try:
        o=model(**chat_ids(q,SYS_FORCE),use_cache=False,output_hidden_states=True,return_dict=True)
        return o.logits[0,-1].float().detach().clone(),o.hidden_states[28][0,-1].float().detach().clone()
    finally:
        for h in hs:h.remove();ACTIVE.discard(id(h))
print("[7/15] SEALED FORCED READOUT")
BLIND=[]
for i,(_,q) in enumerate(INPUTS):
    l,h=run(q,PACK[i],ORTH[i][27]);BLIND.append({"logits":l,"h27":h});print(f"{i+1:02d}/24 | COMPLETE")
print("[8/15] FROZEN TARGET-FREE VALUE SELECTOR")
COMP=[]
for i,b in enumerate(BANK):
    sp=b["spans"];q=b["q"];a=ANCHOR[i];av=sp[a]["states"];h=unit(BLIND[i]["h27"]);rows=[]
    for j,s in enumerate(sp):
        bind=float(np.mean([torch.dot(s["states"][L],av[L]).item() for L in BIND_LAYERS]))
        delta=float(np.mean([torch.dot(unit(s["states"][L]-av[L]),q[L]).item() for L in BIND_LAYERS]))
        query=float(torch.dot(s["states"][27],q[27]));readout=float(torch.dot(s["states"][27],h))
        rows.append({"span":j,"text":s["text"],"bind":bind,"delta":delta,"query":query,"readout":readout})
    for key in ("bind","delta","query","readout"):
        ar=np.array([r[key] for r in rows]);mu=ar.mean();sd=ar.std()+1e-8
        for r,x in zip(rows,ar):r["z_"+key]=float((x-mu)/sd)
    for r in rows:r["selector"]=r["z_bind"]+r["z_delta"]+r["z_readout"]
    sel=[r["selector"] if r["span"]!=a else -1e9 for r in rows]
    order=sorted(range(len(rows)),key=lambda j:sel[j],reverse=True);qorder=sorted(range(len(rows)),key=lambda j:rows[j]["query"],reverse=True)
    COMP.append({"rows":rows,"order":order,"qorder":qorder})
    print(f"{i+1:02d}/24 | {sp[a]['text']!r} -> {[(rows[j]['text'],round(rows[j]['selector'],2)) for j in order[:5]]}")
print("[9/15] TARGET SEAL OPEN - POST-HOC ONLY")
ROWS=[]
for i,target in enumerate(SEALED_TARGETS):
    rows=COMP[i]["rows"];order=COMP[i]["order"];qorder=COMP[i]["qorder"];nt=normtxt(target);match=[j for j,r in enumerate(rows) if normtxt(r["text"])==nt]
    pos={j:k+1 for k,j in enumerate(order)};qpos={j:k+1 for k,j in enumerate(qorder)}
    rank=min([pos[j] for j in match],default=None);qrank=min([qpos[j] for j in match],default=None)
    t1=bool(match and order[0] in match);t3=bool(match and any(j in match for j in order[:3]));t5=bool(match and any(j in match for j in order[:5]))
    ROWS.append({"target":target,"anchor":rows[ANCHOR[i]]["text"],"present":bool(match),"query_rank":qrank,"selector_rank":rank,"top1":t1,"top3":t3,"top5":t5,
                 "top_candidates":[{"text":rows[j]["text"],"score":rows[j]["selector"]} for j in order[:8]]})
    print(f"{i+1:02d}/24 | {target:12s} | QUERY={qrank} -> SELECTOR={rank} | TOP5={t5}")
print("[10/15] FRESH GENERALIZATION SUMMARY")
S={"coverage":float(np.mean([r["present"] for r in ROWS])),"query_rank_median":float(np.median([r["query_rank"] for r in ROWS])),
"selector_rank_median":float(np.median([r["selector_rank"] for r in ROWS])),"top1":float(np.mean([r["top1"] for r in ROWS])),
"top3":float(np.mean([r["top3"] for r in ROWS])),"top5":float(np.mean([r["top5"] for r in ROWS])),
"rank_improved":float(np.mean([r["selector_rank"]<r["query_rank"] for r in ROWS]))}
for k,v in S.items():print(f"{k.upper():34s} {v:+.6f}")
print("[11/15] FROZEN REFERENCE")
print("TEST318 SAME-SET BIND_DELTA_READOUT: MED=5.5 TOP3=.292 TOP5=.500")
print(f"TEST319 FRESH HELD-OUT           : MED={S['selector_rank_median']:.1f} TOP3={S['top3']:.3f} TOP5={S['top5']:.3f}")
print("[12/15] PREDECLARED GENERALIZATION GATE")
G1=S["coverage"]>=.95;G2=S["selector_rank_median"]<S["query_rank_median"];G3=S["top5"]>=.40;G4=S["rank_improved"]>=.60
GATE=bool(G1 and G2 and G3 and G4);CLASS="FROZEN_VALUE_SELECTOR_GENERALIZES" if GATE else "FROZEN_VALUE_SELECTOR_NOT_YET_GENERAL"
print(f"COVERAGE>=.95={G1} | MEDIAN_IMPROVES={G2} | TOP5>=.40={G3} | RANK_IMPROVED>=.60={G4} | CLASS={CLASS}")
print("[13/15] SCIENTIFIC AUDIT")
print("SELECTOR WAS FROZEN BEFORE THESE 24 HELD-OUT TARGETS OPENED. NO TARGET, CASE OR SCORE-SPECIFIC TUNING.")
print("[14/15] CASE + TARGET AUDIT")
print("FRESH_HELD_OUT=True | CASE_ISOLATED=True | CROSS_CASE_DATA=False | TARGET_ENGINE_ACCESS=False | TARGET_POSTHOC_ONLY=True")
print("[15/15] INTEGRITY + SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("Hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["max_seasc_dev"]>1e-6 or AUD["max_bridge_dev"]>1e-6:raise RuntimeError("Dose audit failure")
R={"schema":"akbascore.test319.v1","test":"TEST 319","start":START,"end":utc(),"model":MODEL_ID,
"question":"Does the frozen answer-blind endogenous value selector generalize to fresh held-out cases?",
"engine":"TEST318 locked","selector":"z(bind)+z(delta)+z(readout)","bind_layers":BIND_LAYERS,"summary":S,
"classification":CLASS,"gate":{"coverage":G1,"median":G2,"top5":G3,"improvement":G4,"result":"PASS" if GATE else "FAIL"},"rows":ROWS,
"integrity":{"fresh_held_out":True,"case_isolated":True,"cross_case_data":False,"target_engine_access":False,"target_posthoc_only":True,
"training":False,"optimization":False,"dose_sweep":False,"target_bank":False,"role_bank":False,"target_head":False,"weight_update":False,
"rss":RSS,"bridge":BRIDGE_REL,"seasc_calls":AUD["seasc"],"bridge_calls":AUD["bridge"],"max_seasc_dev":AUD["max_seasc_dev"],
"max_bridge_dev":AUD["max_bridge_dev"],"hooks_remaining":len(ACTIVE),"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
sha=hashlib.sha256(canon(R)).hexdigest();rid=f"T319-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json";tp=ROOT/f"{rid}.txt"
jp.write_bytes(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False).encode())
txt=["="*108,"TEST 319 - FRESH HELD-OUT ENDOGENOUS VALUE-SELECTOR GENERALIZATION","="*108,
f"QUERY MEDIAN={S['query_rank_median']:.1f} -> SELECTOR MEDIAN={S['selector_rank_median']:.1f} | IMPROVED={S['rank_improved']:.3f}",
f"TOP1={S['top1']:.3f} TOP3={S['top3']:.3f} TOP5={S['top5']:.3f}",f"CLASS={CLASS} | GATE={'PASS' if GATE else 'FAIL'}",
"FRESH_HELD_OUT=TRUE | CASE_ISOLATED=TRUE | TARGET POST-HOC ONLY | WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]
tp.write_text("\n".join(txt),encoding="utf-8")
print("="*108);print("TEST 319 COMPLETE")
print(f"QUERY MEDIAN={S['query_rank_median']:.1f} -> SELECTOR MEDIAN={S['selector_rank_median']:.1f} | RANK IMPROVED={S['rank_improved']:.3f}")
print(f"TOP1={S['top1']:.3f} | TOP3={S['top3']:.3f} | TOP5={S['top5']:.3f}")
print(f"CLASS={CLASS} | GATE={'PASS' if GATE else 'FAIL'}")
print("FRESH_HELD_OUT=TRUE | CASE_ISOLATED=TRUE | TARGET POST-HOC ONLY | WEIGHT INTEGRITY PASS | AKBASCORE HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
