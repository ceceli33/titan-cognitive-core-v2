# ================================================================================================
# AKBASCORE - TEST 318
# CASE-ISOLATED RELATION-BINDING COMPONENT ABLATION
# TEST317 LOCKED | 8 BLIND SCORERS | TARGET SEALED | NO TRAINING | NO SWEEP | NO CROSS-CASE DATA
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
SEED=318;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYS_STD="You are a concise reasoning assistant. Use only the information in the prompt."
SYS_FORCE="Read the answer from the internal state. Return exactly one short answer. Never explain, refuse, say information is missing, or mention the prompt."
TOTAL=28;H_EXPECT=3584;END=25;EPS=1e-8;TARGET_RSS=.250235055;BRIDGE_REL=.20;SLOT_K=8;BIND_LAYERS=(19,21,23,25,27)
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20
ROOT=Path("/content/AKBASCORE_TEST318");ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
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
ARMS=("QUERY","BIND","DELTA","READOUT","LOGIT","BIND_DELTA","BIND_DELTA_READOUT","FULL317")
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
def normtxt(s):return re.sub(r"^[^\w]+|[^\w]+$","",s,flags=re.UNICODE).casefold()
def chat_ids(q,system):
    s=tok.apply_chat_template([{"role":"system","content":system},{"role":"user","content":q}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to("cuda")
print("="*108);print("TEST 318 - CASE-ISOLATED RELATION-BINDING COMPONENT ABLATION");print("="*108);print("START:",START)
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
print("[2/15] TEST317 ENGINE LOCK")
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
    o=model(input_ids=e["input_ids"].to("cuda"),attention_mask=e["attention_mask"].to("cuda"),use_cache=False,output_hidden_states=True,return_dict=True);sp=[]
    for m in re.finditer(r"\b[\w'-]+\b",text,re.UNICODE):
        a,b=m.span();ix=[j for j,(s,z) in enumerate(offs) if z>a and s<b]
        if ix:sp.append({"text":m.group(0),"token_ids":ids[ix].tolist(),"states":[unit(o.hidden_states[L+1][0,ix].float().mean(0).detach().clone()) for L in range(TOTAL)]})
    return sp
print("[3/15] TARGET-FREE QUERY NULL CACHE")
QNULL=[last_states(x) for x in QNULLS];print("NULL READY")
print("[4/15] LOCKED SPAN BANK + PACKET + ANCHOR")
PACK=[];ORTH=[];BANK=[];ANCHOR=[]
for i,(statement,question) in enumerate(INPUTS):
    qh=last_states(question);qv=[unit(torch.stack([unit(qh[L]-QNULL[j][L]) for j in range(len(QNULL))]).mean(0)) for L in range(TOTAL)]
    sp=span_states(statement);qs=torch.tensor([torch.dot(s["states"][27],qv[27]).item() for s in sp]);k=min(SLOT_K,len(sp));vals,idx=torch.topk(qs,k);w=torch.softmax(vals/.025,dim=0)
    p=[unit(sum(sp[int(idx[j])]["states"][L]*w[j].to(sp[0]["states"][L].device) for j in range(k))) for L in range(TOTAL)];o=[]
    for L in range(TOTAL):
        r=p[L]-torch.dot(p[L],qv[L])*qv[L];o.append(unit(r) if r.norm()>EPS else p[L])
    a=int(torch.argmax(qs));PACK.append(p);ORTH.append(o);BANK.append({"spans":sp,"q":qv});ANCHOR.append(a)
    print(f"{i+1:02d}/24 | ANCHOR={sp[a]['text']!r} | BANK={len(sp)}")
print("[5/15] CASE-ISOLATION AUDIT")
print("ALL 8 SCORERS TARGET-FREE | SAME-CASE ONLY | TARGET_ACCESS=False | CROSS_CASE=False")
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
print("[8/15] TARGET-FREE COMPONENT MATRIX")
COMP=[]
for i,b in enumerate(BANK):
    sp=b["spans"];q=b["q"];a=ANCHOR[i];av=sp[a]["states"];h=unit(BLIND[i]["h27"]);rows=[]
    for j,s in enumerate(sp):
        bind=float(np.mean([torch.dot(s["states"][L],av[L]).item() for L in BIND_LAYERS]))
        delta=float(np.mean([torch.dot(unit(s["states"][L]-av[L]),q[L]).item() for L in BIND_LAYERS]))
        query=float(torch.dot(s["states"][27],q[27]));readout=float(torch.dot(s["states"][27],h))
        logit=float(np.mean([float(BLIND[i]["logits"][tid]) for tid in s["token_ids"]]))
        rows.append({"span":j,"text":s["text"],"bind":bind,"delta":delta,"query":query,"readout":readout,"logit":logit})
    for key in ("bind","delta","query","readout","logit"):
        ar=np.array([r[key] for r in rows]);mu=ar.mean();sd=ar.std()+1e-8
        for r,x in zip(rows,ar):r["z_"+key]=float((x-mu)/sd)
    scores={arm:[] for arm in ARMS}
    for r in rows:
        scores["QUERY"].append(r["z_query"]);scores["BIND"].append(r["z_bind"]);scores["DELTA"].append(r["z_delta"])
        scores["READOUT"].append(r["z_readout"]);scores["LOGIT"].append(r["z_logit"])
        scores["BIND_DELTA"].append(r["z_bind"]+r["z_delta"])
        scores["BIND_DELTA_READOUT"].append(r["z_bind"]+r["z_delta"]+r["z_readout"])
        scores["FULL317"].append(r["z_bind"]+r["z_delta"]+r["z_readout"]+r["z_logit"]-.5*r["z_query"])
    orders={}
    for arm in ARMS:
        s=scores[arm].copy()
        if arm!="QUERY":s[a]=-1e9
        orders[arm]=sorted(range(len(rows)),key=lambda j:s[j],reverse=True)
    COMP.append({"rows":rows,"orders":orders})
    print(f"{i+1:02d}/24 | ANCHOR={sp[a]['text']!r} | FULL317={[rows[j]['text'] for j in orders['FULL317'][:3]]}")
print("[9/15] TARGET SEAL OPEN - POST-HOC ONLY")
CASE_ROWS=[];MET={a:{"ranks":[],"top1":0,"top3":0,"top5":0} for a in ARMS}
for i,target in enumerate(SEALED_TARGETS):
    nt=normtxt(target);rows=COMP[i]["rows"];match=[j for j,r in enumerate(rows) if normtxt(r["text"])==nt]
    cr={"target":target,"anchor":rows[ANCHOR[i]]["text"],"arms":{}}
    for arm in ARMS:
        order=COMP[i]["orders"][arm];pos={j:k+1 for k,j in enumerate(order)};rank=min([pos[j] for j in match],default=None)
        t1=bool(match and order[0] in match);t3=bool(match and any(j in match for j in order[:3]));t5=bool(match and any(j in match for j in order[:5]))
        cr["arms"][arm]={"rank":rank,"top1":t1,"top3":t3,"top5":t5};MET[arm]["ranks"].append(rank);MET[arm]["top1"]+=int(t1);MET[arm]["top3"]+=int(t3);MET[arm]["top5"]+=int(t5)
    CASE_ROWS.append(cr);print(f"{i+1:02d}/24 | {target:11s} | "+ " | ".join(f"{a}={cr['arms'][a]['rank']}" for a in ("QUERY","BIND","DELTA","READOUT","LOGIT","FULL317")))
print("[10/15] ABLATION SUMMARY")
SUMMARY={}
for arm in ARMS:
    ranks=MET[arm]["ranks"];SUMMARY[arm]={"median_rank":float(np.median(ranks)),"mean_rank":float(np.mean(ranks)),"top1":MET[arm]["top1"]/24,"top3":MET[arm]["top3"]/24,"top5":MET[arm]["top5"]/24}
    s=SUMMARY[arm];print(f"{arm:22s} MED={s['median_rank']:5.1f} | MEAN={s['mean_rank']:6.2f} | TOP1={s['top1']:.3f} | TOP3={s['top3']:.3f} | TOP5={s['top5']:.3f}")
print("[11/15] COMPONENT CONTRIBUTION")
BASE=SUMMARY["QUERY"]["median_rank"]
for arm in ARMS[1:]:print(f"{arm:22s} MEDIAN_DELTA_VS_QUERY={BASE-SUMMARY[arm]['median_rank']:+.2f} | TOP5_DELTA={SUMMARY[arm]['top5']-SUMMARY['QUERY']['top5']:+.3f}")
print("[12/15] PREDECLARED DIAGNOSTIC")
BD=SUMMARY["BIND_DELTA"];FULL=SUMMARY["FULL317"]
G1=BD["median_rank"]<SUMMARY["QUERY"]["median_rank"];G2=BD["top5"]>SUMMARY["QUERY"]["top5"]
if G1 and G2:CLASS="RELATION_GEOMETRY_CARRIES_SIGNAL"
elif FULL["median_rank"]<SUMMARY["QUERY"]["median_rank"] and FULL["top5"]>SUMMARY["QUERY"]["top5"]:CLASS="RELATION_GAIN_REQUIRES_READOUT_LOGIT"
else:CLASS="TEST317_GAIN_NOT_ROBUST_UNDER_ABLATION"
print(f"BIND_DELTA_MEDIAN_IMPROVES={G1} | BIND_DELTA_TOP5_IMPROVES={G2} | CLASS={CLASS}")
print("[13/15] SCIENTIFIC AUDIT")
print("NO COMPONENT WEIGHT OR ARM IS SELECTED USING TARGETS. ALL EIGHT RANKINGS EXIST BEFORE TARGET SEAL OPENS.")
print("[14/15] CASE + TARGET AUDIT")
print("CASE_ISOLATED=True | CROSS_CASE_DATA=False | TARGET_ENGINE_ACCESS=False | TARGET_POSTHOC_ONLY=True")
print("[15/15] INTEGRITY + SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("Hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["max_seasc_dev"]>1e-6 or AUD["max_bridge_dev"]>1e-6:raise RuntimeError("Dose audit failure")
R={"schema":"akbascore.test318.v1","test":"TEST 318","start":START,"end":utc(),"model":MODEL_ID,
"question":"Which target-free component carries TEST317 relation-to-value ranking improvement?","engine":"TEST317 locked",
"arms":ARMS,"bind_layers":BIND_LAYERS,"summary":SUMMARY,"classification":CLASS,"rows":CASE_ROWS,
"integrity":{"case_isolated":True,"cross_case_data":False,"target_engine_access":False,"target_posthoc_only":True,
"training":False,"optimization":False,"dose_sweep":False,"target_bank":False,"role_bank":False,"target_head":False,
"weight_update":False,"rss":RSS,"bridge":BRIDGE_REL,"seasc_calls":AUD["seasc"],"bridge_calls":AUD["bridge"],
"max_seasc_dev":AUD["max_seasc_dev"],"max_bridge_dev":AUD["max_bridge_dev"],"hooks_remaining":len(ACTIVE),
"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
sha=hashlib.sha256(canon(R)).hexdigest();rid=f"T318-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json";tp=ROOT/f"{rid}.txt"
jp.write_bytes(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False).encode())
txt=["="*108,"TEST 318 - CASE-ISOLATED RELATION-BINDING COMPONENT ABLATION","="*108]
for arm in ARMS:
    s=SUMMARY[arm];txt.append(f"{arm:22s} MED={s['median_rank']:.1f} MEAN={s['mean_rank']:.2f} TOP1={s['top1']:.3f} TOP3={s['top3']:.3f} TOP5={s['top5']:.3f}")
txt += [f"CLASS={CLASS}","CASE_ISOLATED=TRUE | TARGET POST-HOC ONLY | WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]
tp.write_text("\n".join(txt),encoding="utf-8")
print("="*108);print("TEST 318 COMPLETE")
for arm in ARMS:
    s=SUMMARY[arm];print(f"{arm:22s} MED={s['median_rank']:.1f} | TOP3={s['top3']:.3f} | TOP5={s['top5']:.3f}")
print("CLASS="+CLASS)
print("CASE_ISOLATED=TRUE | TARGET POST-HOC ONLY | WEIGHT INTEGRITY PASS | AKBASCORE HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
