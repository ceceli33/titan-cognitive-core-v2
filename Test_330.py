# ================================================================================================
# AKBASCORE - TEST 330
# FRESH HELD-OUT DISP ROUTER VALIDATION
# FROZEN ROUTER=ARGMAX TERMINAL RELATIVE DISPLACEMENT | TARGET POST-HOC ONLY
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
SEED=330;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYS_STD="You are a concise reasoning assistant. Use only the information in the prompt."
SYS_FORCE="Read the answer from the internal state. Return exactly one short answer. Never explain, refuse, say information is missing, or mention the prompt."
TOTAL=28;H_EXPECT=3584;END=25;EPS=1e-8;TARGET_RSS=.250235055;MAX_NEW=12
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20
ROOT=Path("/content/AKBASCORE_TEST330");ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
CASES=[
("At dawn, engineer Caldor removed the damaged regulator from Bay Nine while the other technicians inspected the wiring.","Who removed the damaged regulator from Bay Nine?","Caldor"),
("The newly catalogued crystal from the northern shaft received the provisional designation Zenvik.","What provisional designation was given to the crystal?","Zenvik"),
("Three containers passed inspection, and the container approved for shipment carried identification number 741.","What identification number was on the approved container?","741"),
("The expedition crossed Marseille and Valencia before unloading the recovered instrument in Seville.","Where was the recovered instrument unloaded?","Seville"),
("For the outer casing, the team rejected copper and polymer and selected basalt instead.","Which material was selected for the outer casing?","basalt"),
("The indicator remained white during calibration, then changed to turquoise when the sequence completed.","What color did the indicator become after calibration?","turquoise"),
("After entering the maintenance tunnel, Rovan stopped running and started kneeling beside the damaged conduit.","What did Rovan start doing beside the damaged conduit?","kneeling"),
("The untreated sample looked cloudy, but the polished specimen was described as translucent.","How was the polished specimen described?","translucent"),
("Beside the abandoned shelter, Mirea retrieved a sextant and left the broken lantern behind.","What object did Mirea retrieve?","sextant"),
("The authorization phrase written in the secure register is Valtrexon.","What is the authorization phrase?","Valtrexon"),
("Dalen assembled the apparatus, but the final calibration was completed by Orisa.","Who completed the final calibration?","Orisa"),
("The cargo aircraft stopped in Lyon and Zurich before reaching its final destination in Budapest.","Where did the cargo aircraft finally arrive?","Budapest"),
("The engineers tested nickel, graphite, and zirconium before selecting zirconium for the inner brace.","Which material was selected for the inner brace?","zirconium"),
("The primary lamp emitted amber light, while the auxiliary lamp emitted violet.","What color did the auxiliary lamp emit?","violet"),
("The counter initially displayed 117, but after verification the accepted value became 964.","What was the accepted value after verification?","964"),
("When pressure was released, the flexible chamber began contracting rather than holding its previous shape.","What did the flexible chamber begin doing?","contracting"),
("The first covering felt coarse, while the replacement material was described as silky.","How was the replacement material described?","silky"),
("Inside the archive box were a gauge, a medallion, and a barometer; the barometer was selected for restoration.","Which item was selected for restoration?","barometer"),
("The device was first called Vector, but before production its final name became Solvane.","What became the device's final name?","Solvane"),
("Several assistants prepared the cargo, but Professor Keldrin personally delivered the encrypted cylinder.","Who delivered the encrypted cylinder?","Keldrin"),
("The survey team deposited the silver tablet in Kyoto before continuing toward Kobe.","Where was the silver tablet deposited?","Kyoto"),
("The coating appeared beige under ordinary light, but laboratory analysis established that it was cyan.","What was the coating's established color?","cyan"),
("Specimen R-7 contained several compounds but was found to consist primarily of quartz.","What did Specimen R-7 primarily consist of?","quartz"),
("The drone accelerated briefly, maintained speed, and then began ascending near Relay Point Six.","What did the drone begin doing near Relay Point Six?","ascending")]
SHOWCASE=("Mustafa Akbaş raised a crimson pennant beside the Bosphorus Bridge where the research demonstration took place.","What did Mustafa Akbaş do at the location of the research demonstration?","raised a crimson pennant")
ALL=[(s,q) for s,q,_ in CASES]+[(SHOWCASE[0],SHOWCASE[1])];SEALED=tuple(t for _,_,t in CASES)+(SHOWCASE[2],)
QNULLS=["What information is requested?","Which unknown detail should be recovered?","Answer the question using the missing information.","What should be identified?","Which detail is being asked about?"]
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
def chat(q,sysmsg):
    s=tok.apply_chat_template([{"role":"system","content":sysmsg},{"role":"user","content":q}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to("cuda")
print("="*108);print("TEST 330 - FRESH HELD-OUT DISP ROUTER VALIDATION");print("="*108);print("START:",START)
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
print("[2/18] ENGINE + FROZEN ROUTER LOCK")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();raw=[IVME*(ZIRVE*np.exp(-SONUM*L)*(1+SONUM*L)+TABAN)/(ZIRVE+TABAN) for L in range(END+1)];scale=TARGET_RSS/np.sqrt(sum(x*x for x in raw));RHO=[x*scale for x in raw];RSS=float(np.sqrt(sum(x*x for x in RHO)))
print("FP:",[f"{x:.4f}" for x in FP0]);print(f"RSS={RSS:.9f} | ROUTER=ARGMAX(DISP) | FROZEN BEFORE HELD-OUT TARGETS")
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
print("[4/18] FRESH HELD-OUT ANCHOR BANK - TARGET SEALED")
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
        chs.append({"anchor":sp[a]["text"],"packet":packet})
    BANK.append(chs);Q.append(qv);tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,"| BANK=",len(chs))
print("[5/18] SOURCE REMOVED")
print("QUESTION-ONLY RUNTIME | TARGET_ACCESS=False | CASE_ISOLATED=True | ROUTER=DISP ONLY")
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
print("[6/18] EXHAUSTIVE BLIND DISP PROBE")
PRE=[];SELECT=[]
for i,(_,q) in enumerate(ALL):
    v=forward(q);vh=v.hidden_states[28][0,-1].float();vl=model.lm_head(vh.to(model.lm_head.weight.dtype)).float().cpu();aa=[]
    for x in BANK[i]:
        o=forward(q,x["packet"]);h=o.hidden_states[28][0,-1].float();disp=float((h-vh).norm()/vh.norm().clamp_min(EPS))
        aa.append({"anchor":x["anchor"],"packet":x["packet"],"disp":disp,"logits":model.lm_head(h.to(model.lm_head.weight.dtype)).float().cpu()})
    j=int(np.argmax([x["disp"] for x in aa]));PRE.append({"vlogits":vl,"anchors":aa});SELECT.append(j);tag="SHOWCASE" if i==24 else f"{i+1:02d}/24"
    top=sorted(aa,key=lambda x:x["disp"],reverse=True)[:4];print(tag,"| SELECT=",aa[j]["anchor"],"| TOP4=",[f"{x['anchor']}:{x['disp']:.4f}" for x in top])
print("[7/18] ROUTER SEALED")
print("ALL DISP SELECTIONS COMPLETE BEFORE TARGET ACCESS")
print("[8/18] BLIND GENERATION")
GEN=[]
for i,(_,q) in enumerate(ALL):
    v=gen(q);a=gen(q,PRE[i]["anchors"][SELECT[i]]["packet"]);GEN.append({"vanilla":v,"disp":a});tag="SHOWCASE" if i==24 else f"{i+1:02d}/24"
    print(tag,f"| ANCHOR={PRE[i]['anchors'][SELECT[i]]['anchor']!r} | VAN={v!r} | DISP={a!r}")
print("[9/18] TARGET SEAL OPEN - POST-HOC ONLY")
ROWS=[]
for i,t in enumerate(SEALED):
    tid=tok(t,add_special_tokens=False)["input_ids"][0];vl=PRE[i]["vlogits"];vr=int((vl>vl[tid]).sum().item()+1);j=SELECT[i];x=PRE[i]["anchors"][j];lg=x["logits"];dr=int((lg>lg[tid]).sum().item()+1);dg=float(lg[tid]-vl[tid])
    gains=[float(a["logits"][tid]-vl[tid]) for a in PRE[i]["anchors"]];ranks=[int((a["logits"]>a["logits"][tid]).sum().item()+1) for a in PRE[i]["anchors"]];oi=int(np.argmax(gains));ri=int(np.argmin(ranks))
    r={"target":t,"selected_anchor":x["anchor"],"disp":x["disp"],"vanilla_rank":vr,"disp_rank":dr,"disp_gain":dg,"vanilla_output":GEN[i]["vanilla"],"disp_output":GEN[i]["disp"],"oracle_gain_anchor":PRE[i]["anchors"][oi]["anchor"],"oracle_gain":gains[oi],"oracle_rank_anchor":PRE[i]["anchors"][ri]["anchor"],"oracle_rank":ranks[ri]};ROWS.append(r)
    tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,f"| T={t!r} | A={x['anchor']!r} | V={vr} -> DISP={dr} | G={dg:+.3f} | ORACLE={r['oracle_gain_anchor']!r} R={r['oracle_rank']}")
print("[10/18] HELD-OUT PRIMARY SUMMARY")
M=ROWS[:24];vr=np.array([r["vanilla_rank"] for r in M]);dr=np.array([r["disp_rank"] for r in M]);dg=np.array([r["disp_gain"] for r in M]);orr=np.array([r["oracle_rank"] for r in M]);og=np.array([r["oracle_gain"] for r in M])
IMP=float(np.mean(dr<vr));GAIN=float(np.mean(dg));POS=float(np.mean(dg>0));VMED=float(np.median(vr));DMED=float(np.median(dr));OMED=float(np.median(orr));OIMP=float(np.mean(orr<vr));OG=float(np.mean(og))
print(f"VANILLA_MED_RANK={VMED:.1f} | DISP_MED_RANK={DMED:.1f} | ORACLE_MED_RANK={OMED:.1f}")
print(f"DISP_RANK_IMPROVED={IMP:.6f} | DISP_GAIN={GAIN:+.6f} | DISP_POSITIVE={POS:.6f}")
print(f"ORACLE_RANK_IMPROVED={OIMP:.6f} | ORACLE_GAIN={OG:+.6f}")
print("[11/18] ROUTER GENERALIZATION")
MATCH=float(np.mean([r["selected_anchor"]==r["oracle_gain_anchor"] for r in M]));print(f"DISP_EQ_ORACLE_GAIN_ANCHOR={MATCH:.6f} | TEST329_REFERENCE_IMPROVED=0.833333 | TEST329_REFERENCE_GAIN=+1.309245")
print("[12/18] BEHAVIOR")
def exact(o,t):return norm(o)==norm(t)
def mention(o,t):return norm(t) in norm(o) or norm(o) in norm(t)
def norm(s):return re.sub(r"[^\w]+"," ",s,flags=re.UNICODE).strip().casefold()
VE=float(np.mean([exact(r["vanilla_output"],r["target"]) for r in M]));DE=float(np.mean([exact(r["disp_output"],r["target"]) for r in M]));VM=float(np.mean([mention(r["vanilla_output"],r["target"]) for r in M]));DM=float(np.mean([mention(r["disp_output"],r["target"]) for r in M]))
print(f"VANILLA_EXACT={VE:.6f} | DISP_EXACT={DE:.6f} | VANILLA_MENTION={VM:.6f} | DISP_MENTION={DM:.6f}")
print("[13/18] SHOWCASE")
G=ROWS[24];print("-"*108);print("SOURCE :",SHOWCASE[0]);print("QUESTION:",SHOWCASE[1]);print("SELECTED ANCHOR:",G["selected_anchor"],f"| DISP={G['disp']:.6f}");print("VANILLA:",G["vanilla_output"]);print("DISP   :",G["disp_output"]);print(f"TARGET RANK: {G['vanilla_rank']} -> {G['disp_rank']} | GAIN={G['disp_gain']:+.4f}");print("ORACLE GAIN ANCHOR:",G["oracle_gain_anchor"],f"| GAIN={G['oracle_gain']:+.4f} | ORACLE_RANK={G['oracle_rank']}");print("TARGET:",SHOWCASE[2]);print("-"*108)
print("[14/18] PREDECLARED HELD-OUT GATE")
G1=IMP>=.70;G2=GAIN>0;G3=DMED<VMED
if G1 and G2 and G3:CLASS="FROZEN_DISP_ROUTER_HELDOUT_SUPPORTED"
elif G2 and DMED<VMED:CLASS="FROZEN_DISP_ROUTER_PARTIAL_HELDOUT_SUPPORT"
else:CLASS="FROZEN_DISP_ROUTER_NOT_HELDOUT_SUPPORTED"
print(f"RANK_IMPROVED>=0.70:{G1} | MEAN_GAIN>0:{G2} | MEDIAN_RANK_BETTER:{G3} | CLASS={CLASS}")
print("[15/18] FREEZE GUARD")
print("ROUTER=ARGMAX(DISP) FIXED FROM TEST329 | NO FEATURE SEARCH | NO COEFFICIENTS | NO TARGET-BASED SELECTION | ORACLE POST-HOC ONLY")
print("[16/18] DOSE FAIRNESS")
print(f"EVERY ANCHOR PROBE RSS={RSS:.9f} | SELECTED RUN SAME RSS | MOTOR=L0-L25 | L26-L27 OFF")
print("[17/18] SCIENTIFIC AUDIT")
print("FRESH 24 CASES | CASE ISOLATED | CROSS_CASE_DATA=False | TARGET_ENGINE_ACCESS=False | TARGET_POSTHOC_ONLY=True | NO TRAINING | NO WEIGHT UPDATE")
print("[18/18] INTEGRITY + SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("Hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["dev"]>1e-6:raise RuntimeError("Dose audit failure")
R={"schema":"akbascore.test330.v1","test":"TEST 330","start":START,"end":utc(),"model":MODEL_ID,"router":"ARGMAX_DISP","fresh_heldout_cases":24,"summary":{"vanilla_median_rank":VMED,"disp_median_rank":DMED,"oracle_median_rank":OMED,"disp_rank_improved":IMP,"disp_gain_mean":GAIN,"disp_positive":POS,"oracle_rank_improved":OIMP,"oracle_gain_mean":OG,"disp_oracle_anchor_match":MATCH,"vanilla_exact":VE,"disp_exact":DE,"vanilla_mention":VM,"disp_mention":DM},"gate":{"rank_improved_ge_070":G1,"mean_gain_positive":G2,"median_rank_better":G3},"classification":CLASS,"rows":ROWS,"integrity":{"case_isolated":True,"cross_case_data":False,"target_engine_access":False,"target_posthoc_only":True,"router_frozen_from_test329":True,"no_feature_search":True,"no_coefficients":True,"rss":RSS,"motor_layers":[0,25],"observation_layers":[26,27],"training":False,"weight_update":False,"calls":AUD["calls"],"max_dose_dev":AUD["dev"],"hooks_remaining":len(ACTIVE),"fp_start":FP0,"fp_end":fp()}}
sha=hashlib.sha256(canon(R)).hexdigest();rid=f"T330-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json";tp=ROOT/f"{rid}.txt"
jp.write_text(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False),encoding="utf-8")
tp.write_text("\n".join(["="*108,"TEST 330 - FRESH HELD-OUT DISP ROUTER VALIDATION","="*108,f"VANILLA_MED_RANK={VMED:.1f}",f"DISP_MED_RANK={DMED:.1f}",f"DISP_RANK_IMPROVED={IMP:.6f}",f"DISP_GAIN={GAIN:+.6f}",f"DISP_POSITIVE={POS:.6f}",f"ORACLE_MED_RANK={OMED:.1f}",f"ORACLE_GAIN={OG:+.6f}",f"DISP_EXACT={DE:.6f}",f"DISP_MENTION={DM:.6f}",f"CLASS={CLASS}","FRESH_HELDOUT=TRUE | ROUTER FROZEN | TARGET POST-HOC ONLY | WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]),encoding="utf-8")
print("="*108);print("TEST 330 COMPLETE")
print(f"VANILLA MED RANK={VMED:.1f} -> DISP={DMED:.1f} | IMPROVED={IMP:.3f} | GAIN={GAIN:+.6f}")
print(f"ORACLE MED RANK={OMED:.1f} | ORACLE GAIN={OG:+.6f} | DISP==ORACLE={MATCH:.3f}")
print(f"BEHAVIOR EXACT={DE:.3f} | MENTION={DM:.3f}")
print("CLASS=",CLASS);print("FRESH_HELDOUT=TRUE | ROUTER=ARGMAX(DISP) | TARGET POST-HOC ONLY | SAME RSS | WEIGHT INTEGRITY PASS | AKBASCORE HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
