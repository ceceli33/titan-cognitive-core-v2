# ================================================================================================
# AKBASCORE - TEST 329
# TARGET-FREE RELATIONAL ROUTER X-RAY
# EXHAUSTIVE ANCHORS -> BLIND GEOMETRIC ROUTERS -> TARGET POST-HOC ONLY
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
SEED=329;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYS_STD="You are a concise reasoning assistant. Use only the information in the prompt."
SYS_FORCE="Read the answer from the internal state. Return exactly one short answer. Never explain, refuse, say information is missing, or mention the prompt."
TOTAL=28;H_EXPECT=3584;END=25;EPS=1e-8;TARGET_RSS=.250235055;MAX_NEW=12
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20
ROOT=Path("/content/AKBASCORE_TEST329");ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
CASES=[
("During the midnight inspection, technician Neris replaced the cracked relay inside Module Seven.","Who replaced the cracked relay inside Module Seven?","Neris"),
("A mineral recovered from the eastern trench was assigned the provisional name Velqor.","What provisional name was assigned to the mineral?","Velqor"),
("Four sealed packets were weighed, and the lightest packet carried the code 583.","Which code was on the lightest packet?","583"),
("The research vessel departed Bergen and delivered the backup antenna to Lisbon.","Where was the backup antenna delivered?","Lisbon"),
("The builders considered granite and aluminum but ultimately chose ceramic for the heat shield.","Which material was chosen for the heat shield?","ceramic"),
("The warning display remained yellow until the reset, when it turned magenta.","What color did the warning display turn after the reset?","magenta"),
("After reaching the lower deck, Tovan stopped walking and began crawling through the narrow passage.","What did Tovan begin doing?","crawling"),
("The restored panel appeared dull at first, but investigators described its finished surface as lustrous.","How was the finished surface described?","lustrous"),
("Near the collapsed tower, Elira recovered a compass while leaving the damaged radio behind.","What object did Elira recover?","compass"),
("The access sequence recorded in the sealed ledger is Nerovak.","What is the access sequence?","Nerovak"),
("Marek prepared the instruments, while the final measurements were performed by Siona.","Who performed the final measurements?","Siona"),
("The aircraft crossed Vienna and Prague before making its final landing in Warsaw.","Where did the aircraft finally land?","Warsaw"),
("After testing acrylic, bronze, and titanium, the team selected titanium for the pressure frame.","What material was selected for the pressure frame?","titanium"),
("One beacon emitted green light, while the emergency beacon beside it emitted crimson.","What color did the emergency beacon emit?","crimson"),
("The monitor first showed 64, but after synchronization the confirmed reading became 892.","What was the confirmed reading after synchronization?","892"),
("When the locking cycle ended, the inner ring began expanding instead of remaining fixed.","What did the inner ring begin doing?","expanding"),
("The original fabric felt rough, whereas the replacement lining was described as velvety.","How was the replacement lining described?","velvety"),
("The storage crate contained a lens, a chain, and a chronometer; the chronometer was marked for collection.","Which item was marked for collection?","chronometer"),
("The prototype was temporarily called Helix, but the final platform was named Norveth.","What was the final platform named?","Norveth"),
("Two interns catalogued the samples, but Dr. Edrin personally transported the sealed vial.","Who transported the sealed vial?","Edrin"),
("The navigator stored the bronze tablet in Osaka before departing for Nagoya.","Where was the bronze tablet stored?","Osaka"),
("Under the workshop lights the coating seemed gray, but spectral analysis confirmed it was indigo.","What was the coating's confirmed color?","indigo"),
("Sample M-4 contained several minerals but was determined to consist primarily of feldspar.","What did Sample M-4 primarily consist of?","feldspar"),
("The rover paused, moved backward briefly, and then began decelerating near Station Delta.","What did the rover begin doing near Station Delta?","decelerating")]
SHOWCASE=("Mustafa Akbaş planted the Turkish flag on the Golden Gate Bridge where Anthropic conducted its experiment.","What did Mustafa Akbaş do at the location of Anthropic's experiment?","planted the Turkish flag")
ALL=[(s,q) for s,q,_ in CASES]+[(SHOWCASE[0],SHOWCASE[1])];SEALED=tuple(t for _,_,t in CASES)+(SHOWCASE[2],)
QNULLS=["What information is requested?","Which unknown detail should be recovered?","Answer the question using the missing information.","What should be identified?","Which detail is being asked about?"]
ROUTERS=("Q","DQ","SELF","DISP","CONSENSUS")
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
def chat(q,sysmsg):
    s=tok.apply_chat_template([{"role":"system","content":sysmsg},{"role":"user","content":q}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to("cuda")
def ranks(v):
    a=np.asarray(v,float);return (np.argsort(np.argsort(-a))+1).astype(float)
print("="*108);print("TEST 329 - TARGET-FREE RELATIONAL ROUTER X-RAY");print("="*108);print("START:",START)
print("[1/18] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
if not tok.is_fast:raise RuntimeError("Fast tokenizer required")
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers
if len(layers)!=TOTAL or model.config.hidden_size!=H_EXPECT:raise RuntimeError("Architecture mismatch")
torch.cuda.synchronize();print(f"OK | {MODEL_ID} | 28L | H={model.config.hidden_size} | VOCAB={model.config.vocab_size} | {next(model.parameters()).dtype} | {time.perf_counter()-t:.2f}s")
print("[2/18] ENGINE + DOSE LOCK")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();raw=[IVME*(ZIRVE*np.exp(-SONUM*L)*(1+SONUM*L)+TABAN)/(ZIRVE+TABAN) for L in range(END+1)];scale=TARGET_RSS/np.sqrt(sum(x*x for x in raw));RHO=[x*scale for x in raw];RSS=float(np.sqrt(sum(x*x for x in RHO)))
print("FP:",[f"{x:.4f}" for x in FP0]);print(f"RSS={RSS:.9f} | ALL ROUTERS TARGET-FREE | SAME ANCHOR BANK | L0-L25")
@torch.inference_mode()
def states(text,sysmsg=SYS_STD):
    o=model(**chat(text,sysmsg),use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)]
@torch.inference_mode()
def spans(text):
    e=tok(text,return_tensors="pt",add_special_tokens=False,return_offsets_mapping=True);ids=e["input_ids"][0];offs=e["offset_mapping"][0].tolist()
    o=model(input_ids=e["input_ids"].to("cuda"),attention_mask=e["attention_mask"].to("cuda"),use_cache=False,output_hidden_states=True,return_dict=True);out=[]
    for m in re.finditer(r"\b[\w'-]+\b",text,re.UNICODE):
        a,b=m.span();ix=[j for j,(s,z) in enumerate(offs) if z>a and s<b]
        if ix:out.append({"text":m.group(0),"ids":ids[ix].tolist(),"v":[unit(o.hidden_states[L+1][0,ix].float().mean(0).detach()) for L in range(TOTAL)]})
    return out
print("[3/18] QUERY NULL CACHE")
QN=[states(x) for x in QNULLS];print("NULL READY")
print("[4/18] EXHAUSTIVE ANCHOR BANK - TARGET SEALED")
BANK=[];Q=[]
for i,(s,q) in enumerate(ALL):
    qh=states(q);qv=[unit(torch.stack([unit(qh[L]-QN[j][L]) for j in range(len(QN))]).mean(0)) for L in range(TOTAL)];sp=spans(s);chs=[]
    for a in range(len(sp)):
        packet=[]
        for L in range(TOTAL):
            cand=[]
            for b in range(len(sp)):
                if b==a:continue
                d=sp[b]["v"][L]-sp[a]["v"][L];cand.append((float(torch.dot(unit(d),qv[L])),d))
            cand.sort(key=lambda x:x[0],reverse=True);packet.append(unit(cand[0][1]))
        chs.append({"anchor":sp[a]["text"],"packet":packet,"Q":float(torch.dot(sp[a]["v"][27],qv[27]))})
    BANK.append(chs);Q.append(qv);tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,"| BANK=",len(chs))
print("[5/18] SOURCE REMOVED")
print("QUESTION-ONLY RUNTIME | TARGET_ACCESS=False | CROSS_CASE=False")
ACTIVE=set();AUD={"calls":0,"dev":0.0}
def install(packet):
    hs=[]
    for L in range(END+1):
        rho=RHO[L]
        def hk(mod,args,out,L=L,rho=rho):
            y=out[0] if isinstance(out,tuple) else out;z=y[:,-1,:].float();d=packet[L].to(z.device)*z.norm(dim=-1,keepdim=True)*rho
            AUD["calls"]+=1;AUD["dev"]=max(AUD["dev"],abs(float(d.norm()/(z.norm()+EPS))-rho));r=y.clone();r[:,-1,:]=(z+d).to(y.dtype);return (r,)+out[1:] if isinstance(out,tuple) else r
        h=layers[L].register_forward_hook(hk);hs.append(h);ACTIVE.add(id(h))
    return hs
@torch.inference_mode()
def forward(q,packet=None):
    hs=install(packet) if packet is not None else []
    try:return model(**chat(q,SYS_FORCE),use_cache=False,output_hidden_states=True,return_dict=True)
    finally:
        for h in hs:h.remove();ACTIVE.discard(id(h))
@torch.inference_mode()
def gen(q,packet=None):
    x=chat(q,SYS_FORCE);n=x["input_ids"].shape[1];hs=install(packet) if packet is not None else []
    try:y=model.generate(**x,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)
    finally:
        for h in hs:h.remove();ACTIVE.discard(id(h))
    return tok.decode(y[0,n:],skip_special_tokens=True).strip()
print("[6/18] ALL ANCHOR FORWARDS - TARGET SEALED")
PRE=[]
for i,(_,q) in enumerate(ALL):
    v=forward(q);vh=v.hidden_states[28][0,-1].float();vl=model.lm_head(vh.to(model.lm_head.weight.dtype)).float();aa=[]
    for x in BANK[i]:
        o=forward(q,x["packet"]);h=o.hidden_states[28][0,-1].float();d=h-vh;du=unit(d)
        aa.append({"anchor":x["anchor"],"packet":x["packet"],"Q":x["Q"],"DQ":float(torch.dot(du,Q[i][27])),"SELF":float(torch.dot(du,x["packet"][27])),"DISP":float(d.norm()/vh.norm().clamp_min(EPS)),"logits":model.lm_head(h.to(model.lm_head.weight.dtype)).float().cpu()})
    PRE.append({"vlogits":vl.cpu(),"anchors":aa});tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,"| FORWARDS=",len(aa))
print("[7/18] FIXED TARGET-FREE ROUTERS")
SELECT=[]
for i,p in enumerate(PRE):
    a=p["anchors"];sel={}
    for k in ("Q","DQ","SELF","DISP"):sel[k]=int(np.argmax([x[k] for x in a]))
    mats=np.stack([ranks([x[k] for x in a]) for k in ("Q","DQ","SELF","DISP")]);score=mats.mean(0);sel["CONSENSUS"]=int(np.argmin(score));SELECT.append(sel)
    tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,"|"," ".join(f"{k}={a[sel[k]]['anchor']!r}" for k in ROUTERS))
print("[8/18] BLIND GENERATION - SELECTED ROUTERS ONLY")
GEN=[]
for i,(_,q) in enumerate(ALL):
    v=gen(q);g={}
    for k in ROUTERS:
        j=SELECT[i][k];g[k]=gen(q,PRE[i]["anchors"][j]["packet"])
    GEN.append({"VANILLA":v,**g});tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,f"| V={v!r} | C={g['CONSENSUS']!r}")
print("[9/18] TARGET SEAL OPEN - POST-HOC SCORING")
ROWS=[]
for i,t in enumerate(SEALED):
    tid=tok(t,add_special_tokens=False)["input_ids"][0];vl=PRE[i]["vlogits"];vr=int((vl>vl[tid]).sum().item()+1);r={"target":t,"vanilla_rank":vr,"vanilla":GEN[i]["VANILLA"],"routers":{}}
    for k in ROUTERS:
        j=SELECT[i][k];x=PRE[i]["anchors"][j];lg=x["logits"];rank=int((lg>lg[tid]).sum().item()+1);r["routers"][k]={"anchor":x["anchor"],"rank":rank,"gain":float(lg[tid]-vl[tid]),"output":GEN[i][k]}
    gains=[float(x["logits"][tid]-vl[tid]) for x in PRE[i]["anchors"]];ranks_t=[int((x["logits"]>x["logits"][tid]).sum().item()+1) for x in PRE[i]["anchors"]]
    oi=int(np.argmax(gains));ri=int(np.argmin(ranks_t));r["oracle_gain_anchor"]=PRE[i]["anchors"][oi]["anchor"];r["oracle_gain"]=gains[oi];r["oracle_rank_anchor"]=PRE[i]["anchors"][ri]["anchor"];r["oracle_rank"]=ranks_t[ri];ROWS.append(r)
    tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,f"| T={t!r} V={vr} | "+" ".join(f"{k}:{r['routers'][k]['anchor']!r} R={r['routers'][k]['rank']} G={r['routers'][k]['gain']:+.2f}" for k in ROUTERS)+f" | O={r['oracle_gain_anchor']!r}")
print("[10/18] ROUTER SUMMARY")
M=ROWS[:24];SUM={};vr0=np.array([x["vanilla_rank"] for x in M])
for k in ROUTERS:
    rr=[x["routers"][k]["rank"] for x in M];gg=[x["routers"][k]["gain"] for x in M];match=[x["routers"][k]["anchor"]==x["oracle_gain_anchor"] for x in M]
    SUM[k]={"median_rank":float(np.median(rr)),"rank_improved":float(np.mean(np.array(rr)<vr0)),"gain":float(np.mean(gg)),"positive":float(np.mean(np.array(gg)>0)),"oracle_match":float(np.mean(match))}
    print(f"{k:9s} | MED_RANK={np.median(rr):.1f} | IMPROVED={np.mean(np.array(rr)<vr0):.6f} | GAIN={np.mean(gg):+.6f} | POS={np.mean(np.array(gg)>0):.6f} | ORACLE_MATCH={np.mean(match):.6f}")
print("[11/18] TEST328 REFERENCE")
vr=[x["vanilla_rank"] for x in M];orr=[x["oracle_rank"] for x in M];og=[x["oracle_gain"] for x in M]
print(f"VANILLA_MED={np.median(vr):.1f} | ORACLE_MED={np.median(orr):.1f} | ORACLE_IMPROVED={np.mean(np.array(orr)<np.array(vr)):.6f} | ORACLE_GAIN={np.mean(og):+.6f}")
print("[12/18] BEST FIXED ROUTER")
BEST=max(ROUTERS,key=lambda k:(SUM[k]["rank_improved"],SUM[k]["gain"]));print(f"BEST={BEST} | IMPROVED={SUM[BEST]['rank_improved']:.6f} | GAIN={SUM[BEST]['gain']:+.6f} | MED_RANK={SUM[BEST]['median_rank']:.1f}")
print("[13/18] GOLDEN GATE")
G=ROWS[24];print("-"*108);print("SOURCE :",SHOWCASE[0]);print("QUESTION:",SHOWCASE[1]);print("VANILLA:",G["vanilla"])
for k in ROUTERS:print(f"{k:9s}: ANCHOR={G['routers'][k]['anchor']!r} | RANK={G['routers'][k]['rank']} | GAIN={G['routers'][k]['gain']:+.4f} | OUT={G['routers'][k]['output']!r}")
print("ORACLE GAIN:",G["oracle_gain_anchor"],f"| {G['oracle_gain']:+.4f}");print("TARGET:",SHOWCASE[2]);print("-"*108)
print("[14/18] PREDECLARED DIAGNOSTIC")
BI=SUM[BEST]["rank_improved"];BG=SUM[BEST]["gain"]
if BI>=.70 and BG>0:CLASS="TARGET_FREE_GEOMETRIC_ROUTER_SUPPORTED"
elif BI>.458333 and BG>0:CLASS="TARGET_FREE_ROUTER_PARTIAL_IMPROVEMENT"
else:CLASS="AVAILABLE_GEOMETRY_DOES_NOT_RESOLVE_ROUTING"
print(f"BEST={BEST} | RANK_IMPROVED={BI:.6f} | GAIN={BG:+.6f} | TEST328_Q_BASELINE=0.458333 | CLASS={CLASS}")
print("[15/18] ANTI-TUNING GUARD")
print("NO LEARNED WEIGHTS | NO TARGET-BASED COEFFICIENTS | CONSENSUS=EQUAL MEAN FEATURE RANK | TARGET OPENED AFTER ALL ROUTER SELECTIONS")
print("[16/18] DOSE FAIRNESS")
print(f"EVERY SELECTED ANCHOR RSS={RSS:.9f} | MOTOR=L0-L25 | L26-L27 OFF")
print("[17/18] SCIENTIFIC AUDIT")
print("CASE ISOLATED | CROSS_CASE_DATA=False | TARGET_ENGINE_ACCESS=False | TARGET_POSTHOC_ONLY=True | NO TRAINING | NO WEIGHT UPDATE")
print("[18/18] INTEGRITY + SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("Hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["dev"]>1e-6:raise RuntimeError("Dose audit failure")
R={"schema":"akbascore.test329.v1","test":"TEST 329","start":START,"end":utc(),"model":MODEL_ID,"routers":ROUTERS,"summary":SUM,"best_router":BEST,"classification":CLASS,"rows":ROWS,"integrity":{"case_isolated":True,"cross_case_data":False,"target_engine_access":False,"target_posthoc_only":True,"no_learned_router":True,"rss":RSS,"motor_layers":[0,25],"observation_layers":[26,27],"training":False,"weight_update":False,"calls":AUD["calls"],"max_dose_dev":AUD["dev"],"hooks_remaining":len(ACTIVE),"fp_start":FP0,"fp_end":fp()}}
sha=hashlib.sha256(canon(R)).hexdigest();rid=f"T329-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json";tp=ROOT/f"{rid}.txt"
jp.write_text(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False),encoding="utf-8")
tp.write_text("\n".join(["="*108,"TEST 329 - TARGET-FREE RELATIONAL ROUTER X-RAY","="*108,f"BEST={BEST}",f"BEST_MED_RANK={SUM[BEST]['median_rank']:.1f}",f"BEST_RANK_IMPROVED={SUM[BEST]['rank_improved']:.6f}",f"BEST_GAIN={SUM[BEST]['gain']:+.6f}",f"CLASS={CLASS}","CASE_ISOLATED=TRUE | TARGET POST-HOC ONLY | SAME RSS | WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]),encoding="utf-8")
print("="*108);print("TEST 329 COMPLETE")
print(f"BEST ROUTER={BEST} | MED RANK={SUM[BEST]['median_rank']:.1f} | IMPROVED={SUM[BEST]['rank_improved']:.3f} | GAIN={SUM[BEST]['gain']:+.6f}")
print("CLASS=",CLASS);print("CASE_ISOLATED=TRUE | TARGET POST-HOC ONLY | SAME RSS | WEIGHT INTEGRITY PASS | AKBASCORE HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
