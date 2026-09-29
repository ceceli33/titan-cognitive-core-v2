# ================================================================================================
# AKBASCORE - TEST 317
# CASE-ISOLATED ENDOGENOUS RELATION-TO-VALUE BINDING X-RAY
# TEST316 SPAN BANK LOCKED | QUESTION -> RELATION ANCHOR -> VALUE | TARGET SEALED
# NO TARGET BANK | NO CROSS-CASE DATA | NO TRAINING | NO DOSE SWEEP
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
SEED=317;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYS_STD="You are a concise reasoning assistant. Use only the information in the prompt."
SYS_FORCE="Read the answer from the internal state. Return exactly one short answer. Never explain, refuse, say information is missing, or mention the prompt."
TOTAL=28;H_EXPECT=3584;END=25;EPS=1e-8;TARGET_RSS=.250235055;BRIDGE_REL=.20;SLOT_K=8;BIND_LAYERS=(19,21,23,25,27)
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20
ROOT=Path("/content/AKBASCORE_TEST317");ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
INPUTS=[
("After the storm, engineer Selin placed the damaged sensor inside Vault Kestrel.","Who placed the damaged sensor inside Vault Kestrel?"),
("The expedition discovered an unfamiliar crystal called Zorvex beneath the northern ridge.","What was the unfamiliar crystal called?"),
("Although three containers were inspected, only the smallest one carried the number 417.","Which number was on the smallest container?"),
("At sunset the maintenance crew moved the backup transmitter from the hangar to Reykjavik.","Where was the backup transmitter moved?"),
("For the prototype shell, the designers rejected steel and selected basalt instead.","Which material was selected for the prototype shell?"),
("The indicator stayed red all morning but changed to violet immediately after calibration.","What color did the indicator become after calibration?"),
("When the alarm sounded, Arven stopped descending and began climbing toward the upper platform.","What did Arven begin doing?"),
("Researchers described the newly polished specimen as unusually glossy despite its dark surface.","How was the newly polished specimen described?"),
("Beside the abandoned observatory, Mira recovered a sextant while the other tools remained buried.","What object did Mira recover?"),
("The final authentication phrase, written nowhere else in the archive, is Ordelis.","What is the final authentication phrase?"),
("Darian supervised the launch, but the navigation calculations were performed by Keira.","Who performed the navigation calculations?"),
("The drone passed over Madrid and Rome before finally landing in Tallinn.","Where did the drone finally land?"),
("Of copper, glass, and polymer, the laboratory chose glass for the transparent chamber wall.","What material was chosen for the chamber wall?"),
("The first lamp flashed blue; the second, assigned to emergency status, glowed amber.","What color was the emergency-status lamp?"),
("Unit R initially displayed 29, then after recalibration its verified value became 731.","What was Unit R's verified value after recalibration?"),
("Instead of opening as expected, the outer hatch began rotating when the sequence completed.","What did the outer hatch begin doing?"),
("The upper plate remained coarse, whereas the replacement plate installed below it was silky.","How was the replacement plate described?"),
("Inside the wooden case were a map and a ruler, but the item marked for retrieval was the astrolabe.","Which item was marked for retrieval?"),
("The temporary codename Lumen was discarded; the production system was ultimately named Varethis.","What was the production system ultimately named?"),
("Professor Ilyan sent two assistants away and personally carried the sealed notebook into the archive.","Who carried the sealed notebook into the archive?"),
("A courier left the silver key in Kyoto before continuing alone toward Osaka.","Where was the silver key left?"),
("The machine's casing looked black under storage lighting, but inspection confirmed its actual color was turquoise.","What was the casing's actual color?"),
("Among several samples, specimen Q-9 was identified as being composed primarily of quartz.","What was specimen Q-9 primarily composed of?"),
("The autonomous cart waited briefly, reversed two meters, and then started accelerating toward Gate C.","What did the cart start doing toward Gate C?")
]
SEALED_TARGETS=("Selin","Zorvex","417","Reykjavik","basalt","violet","climbing","glossy","sextant","Ordelis","Keira","Tallinn","glass","amber","731","rotating","silky","astrolabe","Varethis","Ilyan","Kyoto","turquoise","quartz","accelerating")
QNULLS=["What information is requested?","Which unknown detail should be recovered?","Answer the question using the missing information.","What should be identified?","Which detail is being asked about?"]
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
def normtxt(s):return re.sub(r"^[^\w]+|[^\w]+$","",s,flags=re.UNICODE).casefold()
def chat_ids(q,system):
    s=tok.apply_chat_template([{"role":"system","content":system},{"role":"user","content":q}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to("cuda")
print("="*108);print("TEST 317 - CASE-ISOLATED ENDOGENOUS RELATION-TO-VALUE BINDING X-RAY");print("="*108);print("START:",START)
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
print("[2/15] TEST316 ENGINE LOCK")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();base=[IVME*(ZIRVE*np.exp(-SONUM*L)*(1+SONUM*L)+TABAN)/(ZIRVE+TABAN) for L in range(END+1)];scale=TARGET_RSS/np.sqrt(sum(x*x for x in base));RHO=[x*scale for x in base];RSS=float(np.sqrt(sum(x*x for x in RHO)))
print("FP:",[f"{x:.4f}" for x in FP0]);print(f"RSS={RSS:.9f} | BRIDGE={BRIDGE_REL:.3f} | BIND_LAYERS={BIND_LAYERS}")
@torch.inference_mode()
def last_states(text):
    o=model(**chat_ids(text,SYS_STD),use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)]
@torch.inference_mode()
def span_states(text):
    e=tok(text,return_tensors="pt",add_special_tokens=False,return_offsets_mapping=True);ids=e["input_ids"][0];offs=e["offset_mapping"][0].tolist()
    x={"input_ids":e["input_ids"].to("cuda"),"attention_mask":e["attention_mask"].to("cuda")}
    o=model(**x,use_cache=False,output_hidden_states=True,return_dict=True);sp=[]
    for m in re.finditer(r"\b[\w'-]+\b",text,re.UNICODE):
        a,b=m.span();ix=[j for j,(s,z) in enumerate(offs) if z>a and s<b]
        if ix:sp.append({"text":m.group(0),"start":a,"end":b,"token_ids":ids[ix].tolist(),"states":[unit(o.hidden_states[L+1][0,ix].float().mean(0).detach().clone()) for L in range(TOTAL)]})
    return sp
print("[3/15] TARGET-FREE QUERY NULL CACHE")
QNULL=[last_states(x) for x in QNULLS];print("NULL READY")
print("[4/15] LOCKED SPAN BANK + PACKET + RELATION ANCHOR")
PACK=[];ORTH=[];BANK=[];META=[]
for i,(statement,question) in enumerate(INPUTS):
    qh=last_states(question);qv=[unit(torch.stack([unit(qh[L]-QNULL[j][L]) for j in range(len(QNULL))]).mean(0)) for L in range(TOTAL)]
    spans=span_states(statement);qs=torch.tensor([torch.dot(s["states"][27],qv[27]).item() for s in spans]);k=min(SLOT_K,len(spans));vals,idx=torch.topk(qs,k);w=torch.softmax(vals/.025,dim=0)
    p=[unit(sum(spans[int(idx[j])]["states"][L]*w[j].to(spans[0]["states"][L].device) for j in range(k))) for L in range(TOTAL)];o=[]
    for L in range(TOTAL):
        r=p[L]-torch.dot(p[L],qv[L])*qv[L];o.append(unit(r) if r.norm()>EPS else p[L])
    anchor=int(torch.argmax(qs));PACK.append(p);ORTH.append(o);BANK.append({"spans":spans,"q":qv,"query_scores":qs})
    META.append({"anchor":anchor,"anchor_text":spans[anchor]["text"],"addressed":[spans[int(x)]["text"] for x in idx]})
    print(f"{i+1:02d}/24 | ANCHOR={spans[anchor]['text']!r} | ADDR={META[-1]['addressed']}")
print("[5/15] CASE-ISOLATION AUDIT")
print("ANCHOR=QUERY-SELECTED SAME-CASE SPAN | TARGET_ACCESS=False | CROSS_CASE=False")
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
print("[8/15] TARGET-FREE RELATION-TO-VALUE COMPETITION")
COMP=[]
for i,b in enumerate(BANK):
    spans=b["spans"];a=META[i]["anchor"];av=spans[a]["states"];q=b["q"];h=unit(BLIND[i]["h27"]);rows=[]
    for j,s in enumerate(spans):
        bind=float(np.mean([torch.dot(s["states"][L],av[L]).item() for L in BIND_LAYERS]))
        delta=float(np.mean([torch.dot(unit(s["states"][L]-av[L]),q[L]).item() for L in BIND_LAYERS]))
        query=float(torch.dot(s["states"][27],q[27]));readout=float(torch.dot(s["states"][27],h))
        logits=[float(BLIND[i]["logits"][tid]) for tid in s["token_ids"]];logit=float(np.mean(logits))
        distance=abs(s["start"]-spans[a]["start"]);rows.append({"span":j,"text":s["text"],"bind":bind,"delta":delta,"query":query,"readout":readout,"logit":logit,"distance":distance})
    for key in ("bind","delta","query","readout","logit"):
        ar=np.array([r[key] for r in rows]);mu=ar.mean();sd=ar.std()+1e-8
        for r,x in zip(rows,ar):r["z_"+key]=float((x-mu)/sd)
    for r in rows:
        r["relation_value"]=r["z_bind"]+r["z_delta"]+r["z_readout"]+r["z_logit"]-.5*r["z_query"]
        if r["span"]==a:r["relation_value"]=-1e9
    order=sorted(range(len(rows)),key=lambda j:rows[j]["relation_value"],reverse=True);COMP.append({"rows":rows,"order":order})
    print(f"{i+1:02d}/24 | {META[i]['anchor_text']!r} -> {[(rows[j]['text'],round(rows[j]['relation_value'],2)) for j in order[:5]]}")
print("[9/15] TARGET SEAL OPEN - POST-HOC ONLY")
ROWS=[]
for i,target in enumerate(SEALED_TARGETS):
    rows=COMP[i]["rows"];order=COMP[i]["order"];nt=normtxt(target);match=[j for j,r in enumerate(rows) if normtxt(r["text"])==nt];pos={j:k+1 for k,j in enumerate(order)}
    rank=min([pos[j] for j in match],default=None);top1=bool(match and order[0] in match);top3=bool(match and any(j in match for j in order[:3]));top5=bool(match and any(j in match for j in order[:5]))
    base_q=sorted(range(len(rows)),key=lambda j:rows[j]["query"],reverse=True);qpos={j:k+1 for k,j in enumerate(base_q)};qrank=min([qpos[j] for j in match],default=None)
    rr={"target":target,"anchor":META[i]["anchor_text"],"target_present":bool(match),"query_rank":qrank,"relation_rank":rank,"top1":top1,"top3":top3,"top5":top5,
        "top_candidates":[{"text":rows[j]["text"],"score":rows[j]["relation_value"],"bind":rows[j]["bind"],"delta":rows[j]["delta"],"readout":rows[j]["readout"],"logit":rows[j]["logit"]} for j in order[:8]]}
    ROWS.append(rr);print(f"{i+1:02d}/24 | {target:11s} | ANCHOR={META[i]['anchor_text']!r:12s} | QUERY_RANK={qrank} -> REL_RANK={rank} | TOP5={top5}")
print("[10/15] RELATION-BINDING SUMMARY")
S={"coverage":float(np.mean([r["target_present"] for r in ROWS])),"query_rank_median":float(np.median([r["query_rank"] for r in ROWS])),
"relation_rank_median":float(np.median([r["relation_rank"] for r in ROWS])),"top1":float(np.mean([r["top1"] for r in ROWS])),
"top3":float(np.mean([r["top3"] for r in ROWS])),"top5":float(np.mean([r["top5"] for r in ROWS])),
"rank_improved":float(np.mean([r["relation_rank"]<r["query_rank"] for r in ROWS]))}
for k,v in S.items():print(f"{k.upper():34s} {v:+.6f}")
print("[11/15] TEST316 COMPARISON")
print(f"TEST316 MEDIAN=8.5 TOP3=.250 TOP5=.292 | TEST317 MEDIAN={S['relation_rank_median']:.1f} TOP3={S['top3']:.3f} TOP5={S['top5']:.3f}")
print("[12/15] PREDECLARED GATE")
G1=S["relation_rank_median"]<8.5;G2=S["top5"]>.2916667;G3=S["rank_improved"]>=.60
GATE=bool(G1 and G2 and G3);CLASS="RELATION_TO_VALUE_BINDING_SUPPORTED" if GATE else "RELATION_BINDING_NOT_SUFFICIENT"
print(f"MEDIAN<8.5={G1} | TOP5>.292={G2} | RANK_IMPROVED>=.60={G3} | CLASS={CLASS}")
print("[13/15] SCIENTIFIC AUDIT")
print("TARGET DOES NOT SELECT ANCHOR, SCORE, LAYER, CANDIDATE OR WEIGHT. TARGET OPENS ONLY AFTER ALL BLIND FORWARDS AND RANKING.")
print("[14/15] CASE + TARGET AUDIT")
print("CASE_ISOLATED=True | CROSS_CASE_DATA=False | TARGET_ENGINE_ACCESS=False | TARGET_POSTHOC_ONLY=True")
print("[15/15] INTEGRITY + SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("Hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["max_seasc_dev"]>1e-6 or AUD["max_bridge_dev"]>1e-6:raise RuntimeError("Dose audit failure")
R={"schema":"akbascore.test317.v1","test":"TEST 317","start":START,"end":utc(),"model":MODEL_ID,
"question":"Can a target-free query-selected relation anchor recover the bound value from same-case lexical spans?",
"engine":"TEST316 locked","bind_layers":BIND_LAYERS,"summary":S,"classification":CLASS,
"gate":{"median":G1,"top5":G2,"improvement":G3,"result":"PASS" if GATE else "FAIL"},"rows":ROWS,
"integrity":{"case_isolated":True,"cross_case_data":False,"target_engine_access":False,"target_posthoc_only":True,"training":False,"optimization":False,
"dose_sweep":False,"target_bank":False,"role_bank":False,"target_head":False,"weight_update":False,"rss":RSS,"bridge":BRIDGE_REL,
"seasc_calls":AUD["seasc"],"bridge_calls":AUD["bridge"],"max_seasc_dev":AUD["max_seasc_dev"],"max_bridge_dev":AUD["max_bridge_dev"],
"hooks_remaining":len(ACTIVE),"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
sha=hashlib.sha256(canon(R)).hexdigest();rid=f"T317-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json";tp=ROOT/f"{rid}.txt"
jp.write_bytes(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False).encode())
txt=["="*108,"TEST 317 - CASE-ISOLATED ENDOGENOUS RELATION-TO-VALUE BINDING X-RAY","="*108,
f"QUERY MEDIAN={S['query_rank_median']:.1f} | RELATION MEDIAN={S['relation_rank_median']:.1f} | IMPROVED={S['rank_improved']:.3f}",
f"TOP1={S['top1']:.3f} TOP3={S['top3']:.3f} TOP5={S['top5']:.3f}",f"CLASS={CLASS} | GATE={'PASS' if GATE else 'FAIL'}",
"CASE_ISOLATED=TRUE | TARGET POST-HOC ONLY | WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]
tp.write_text("\n".join(txt),encoding="utf-8")
print("="*108);print("TEST 317 COMPLETE")
print(f"QUERY MEDIAN={S['query_rank_median']:.1f} -> RELATION MEDIAN={S['relation_rank_median']:.1f} | RANK IMPROVED={S['rank_improved']:.3f}")
print(f"TOP1={S['top1']:.3f} | TOP3={S['top3']:.3f} | TOP5={S['top5']:.3f}")
print(f"CLASS={CLASS} | GATE={'PASS' if GATE else 'FAIL'}")
print("CASE_ISOLATED=TRUE | TARGET POST-HOC ONLY | WEIGHT INTEGRITY PASS | AKBASCORE HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
