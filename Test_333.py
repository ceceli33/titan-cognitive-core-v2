# ================================================================================================
# AKBASCORE - TEST 333
# FROZEN DISP ROUTER -> TERMINAL DECISION-BOUNDARY DECOMPOSITION
# SAME FRESH-24 | SAME PACKET | SAME RSS | TARGET POST-HOC ONLY | NO NEW INTERVENTION
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
SEED=333;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYS_STD="You are a concise reasoning assistant. Use only the information in the prompt."
SYS_FORCE="Read the answer from the internal state. Return exactly one short answer. Never explain, refuse, say information is missing, or mention the prompt."
TOTAL=28;H_EXPECT=3584;END=25;EPS=1e-8;TARGET_RSS=.250235055;TOPN=20
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20
ROOT=Path("/content/AKBASCORE_TEST333");ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
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
print("="*108);print("TEST 333 - FROZEN DISP ROUTER -> TERMINAL DECISION-BOUNDARY DECOMPOSITION");print("="*108);print("START:",START)
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
print("[2/18] ENGINE + ROUTER LOCK")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();raw=[IVME*(ZIRVE*np.exp(-SONUM*L)*(1+SONUM*L)+TABAN)/(ZIRVE+TABAN) for L in range(END+1)];scale=TARGET_RSS/np.sqrt(sum(x*x for x in raw));RHO=[x*scale for x in raw];RSS=float(np.sqrt(sum(x*x for x in RHO)))
print("FP:",[f"{x:.4f}" for x in FP0]);print(f"RSS={RSS:.9f} | ROUTER=ARGMAX(DISP) | TARGET SEALED")
@torch.inference_mode()
def states(text,sysmsg=SYS_STD):
    o=model(**chat(text,sysmsg),use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)]
@torch.inference_mode()
def spans(text):
    e=tok(text,return_tensors="pt",add_special_tokens=False,return_offsets_mapping=True);offs=e["offset_mapping"][0].tolist()
    o=model(input_ids=e["input_ids"].to("cuda"),attention_mask=e["attention_mask"].to("cuda"),use_cache=False,output_hidden_states=True,return_dict=True);out=[]
    for m in re.finditer(r"\b[\w'-]+\b",text,re.UNICODE):
        a,b=m.span();ix=[j for j,(s,z) in enumerate(offs) if z>a and s<b]
        if ix:out.append({"text":m.group(0),"v":[unit(o.hidden_states[L+1][0,ix].float().mean(0).detach()) for L in range(TOTAL)]})
    return out
print("[3/18] QUERY NULL CACHE")
QN=[states(x) for x in QNULLS];print("NULL READY")
print("[4/18] ANCHOR BANK - TARGET SEALED")
BANK=[]
for i,(s,q) in enumerate(ALL):
    qh=states(q);qv=[unit(torch.stack([unit(qh[L]-QN[j][L]) for j in range(len(QN))]).mean(0)) for L in range(TOTAL)];sp=spans(s);chs=[]
    for a in range(len(sp)):
        packet=[]
        for L in range(TOTAL):
            cand=[]
            for b in range(len(sp)):
                if b==a:continue
                d=sp[b]["v"][L]-sp[a]["v"][L];cand.append((float(torch.dot(unit(d),qv[L])),d))
            cand.sort(key=lambda z:z[0],reverse=True);packet.append(unit(cand[0][1]))
        chs.append({"anchor":sp[a]["text"],"packet":packet})
    BANK.append(chs);tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,"| BANK=",len(chs))
print("[5/18] SOURCE REMOVED")
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
print("[6/18] BLIND DISP ROUTER")
SELECT=[];PRE=[]
for i,(_,q) in enumerate(ALL):
    v=forward(q);vh=v.hidden_states[28][0,-1].float();aa=[]
    for x in BANK[i]:
        o=forward(q,x["packet"]);h=o.hidden_states[28][0,-1].float();aa.append({"anchor":x["anchor"],"packet":x["packet"],"disp":float((h-vh).norm()/vh.norm().clamp_min(EPS))})
    j=int(np.argmax([x["disp"] for x in aa]));SELECT.append(j);PRE.append({"van":v,"anchors":aa});tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,f"| ANCHOR={aa[j]['anchor']!r} | DISP={aa[j]['disp']:.6f}")
print("[7/18] ROUTER SEALED")
print("ALL ARGMAX(DISP) SELECTIONS COMPLETE | TARGET_ACCESS=False")
print("[8/18] TERMINAL DISTRIBUTIONS FROZEN")
FROZEN=[]
for i,(_,q) in enumerate(ALL):
    v=PRE[i]["van"];s=forward(q,PRE[i]["anchors"][SELECT[i]]["packet"])
    vh=v.hidden_states[28][0,-1].float();sh=s.hidden_states[28][0,-1].float()
    vl=model.lm_head(vh.to(model.lm_head.weight.dtype)).float().cpu();sl=model.lm_head(sh.to(model.lm_head.weight.dtype)).float().cpu();dl=sl-vl
    vi=torch.topk(vl,TOPN).indices.tolist();si=torch.topk(sl,TOPN).indices.tolist()
    FROZEN.append({"vl":vl,"sl":sl,"dl":dl,"van_top":vi,"steer_top":si})
    tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,"| V_TOP1=",repr(tok.decode([vi[0]])),"| S_TOP1=",repr(tok.decode([si[0]])))
print("[9/18] TARGET-FREE STATE SEALED")
print("ROUTER + PACKET + VANILLA/STEERED TERMINAL DISTRIBUTIONS FROZEN BEFORE TARGET ACCESS")
print("[10/18] TARGET SEAL OPEN - DECISION BOUNDARY")
ROWS=[]
for i,t in enumerate(SEALED):
    tid=tok(t,add_special_tokens=False)["input_ids"][0];f=FROZEN[i];vl=f["vl"];sl=f["sl"];dl=f["dl"]
    vr=int((vl>vl[tid]).sum()+1);sr=int((sl>sl[tid]).sum()+1);vtop=int(torch.argmax(vl));stop=int(torch.argmax(sl))
    vm=float(vl[vtop]-vl[tid]);sm=float(sl[stop]-sl[tid]);tg=float(dl[tid]);vcg=float(dl[vtop]);scg=float(dl[stop])
    same=vtop==stop;fixed_close=float(vm-(sl[vtop]-sl[tid]));actual_close=float(vm-sm);fixed_target_adv=float(tg-vcg);actual_target_adv=float(tg-scg)
    ahead=torch.where(sl>sl[tid])[0];ahead_delta=dl[ahead] if len(ahead) else torch.empty(0)
    suppressed=float((ahead_delta<tg).float().mean()) if len(ahead) else 1.0
    r={"target":t,"target_piece":tok.decode([tid]),"anchor":PRE[i]["anchors"][SELECT[i]]["anchor"],"vanilla_rank":vr,"steered_rank":sr,"vanilla_top1":tok.decode([vtop]),"steered_top1":tok.decode([stop]),"top1_same":same,"vanilla_margin":vm,"steered_margin":sm,"target_gain":tg,"vanilla_top1_gain":vcg,"steered_top1_gain":scg,"fixed_competitor_margin_close":fixed_close,"actual_top1_margin_close":actual_close,"target_advantage_vs_vanilla_top1":fixed_target_adv,"target_advantage_vs_steered_top1":actual_target_adv,"fraction_current_competitors_target_outgains":suppressed,"competitors_remaining":int(len(ahead))};ROWS.append(r)
    tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,f"| T={t!r} | R={vr}->{sr} | M={vm:.2f}->{sm:.2f} | TG={tg:+.2f} | TOP1G={scg:+.2f} | CLOSE={actual_close:+.2f} | AHEAD={len(ahead)}")
print("[11/18] PRIMARY SUMMARY")
M=ROWS[:24]
def mean(k):return float(np.mean([r[k] for r in M]))
def med(k):return float(np.median([r[k] for r in M]))
IMP=float(np.mean([r["steered_rank"]<r["vanilla_rank"] for r in M]));POS=float(np.mean([r["target_gain"]>0 for r in M]))
print(f"RANK_IMPROVED={IMP:.6f} | TARGET_GAIN={mean('target_gain'):+.6f} | POSITIVE={POS:.6f}")
print(f"MED_MARGIN={med('vanilla_margin'):.6f}->{med('steered_margin'):.6f} | MED_MARGIN_CLOSE={med('actual_top1_margin_close'):+.6f}")
print(f"MEAN_TARGET_GAIN={mean('target_gain'):+.6f} | MEAN_STEERED_TOP1_GAIN={mean('steered_top1_gain'):+.6f}")
print("[12/18] PRIOR VS INTERVENTION")
CLOSE=float(np.mean([r["actual_top1_margin_close"]>0 for r in M]));TADV=float(np.mean([r["target_advantage_vs_steered_top1"]>0 for r in M]));SAME=float(np.mean([r["top1_same"] for r in M]))
print(f"MARGIN_CLOSED_CASES={CLOSE:.6f} | TARGET_GAIN_GT_FINAL_TOP1_GAIN={TADV:.6f} | TOP1_ID_UNCHANGED={SAME:.6f}")
print(f"MEAN_TARGET_ADVANTAGE_VS_FINAL_TOP1={mean('target_advantage_vs_steered_top1'):+.6f}")
print("[13/18] COMPETITOR FIELD")
print(f"MED_COMPETITORS_REMAINING={med('competitors_remaining'):.1f} | MEAN_FRACTION_COMPETITORS_TARGET_OUTGAINS={mean('fraction_current_competitors_target_outgains'):.6f}")
for K in (1,5,10,25,50,100,256,1000):
    print(f"TOP{K:04d}={np.mean([r['steered_rank']<=K for r in M]):.6f}")
print("[14/18] SHOWCASE")
G=ROWS[24];print("-"*108);print("SOURCE :",SHOWCASE[0]);print("QUESTION:",SHOWCASE[1]);print("ANCHOR :",G["anchor"]);print("TARGET :",SHOWCASE[2])
print(f"RANK={G['vanilla_rank']}->{G['steered_rank']} | TARGET_GAIN={G['target_gain']:+.4f}")
print(f"TOP1={G['vanilla_top1']!r}->{G['steered_top1']!r} | MARGIN={G['vanilla_margin']:.4f}->{G['steered_margin']:.4f} | CLOSE={G['actual_top1_margin_close']:+.4f}")
print(f"TARGET_ADV_VS_FINAL_TOP1={G['target_advantage_vs_steered_top1']:+.4f} | COMPETITORS={G['competitors_remaining']}");print("-"*108)
print("[15/18] PREDECLARED DIAGNOSTIC")
MC=med("actual_top1_margin_close");SM=med("steered_margin");TG=mean("target_gain");CG=mean("steered_top1_gain")
if TG>0 and MC>0 and SM>5:CLASS="TARGET_GAINS_AND_CLOSES_MARGIN_BUT_BASE_PRIOR_REMAINS_DOMINANT"
elif TG>0 and MC<=0:CLASS="TARGET_GAINS_BUT_COMPETITORS_GAIN_AS_FAST_OR_FASTER"
elif TG>CG and SM>0:CLASS="TARGET_OUTGAINS_COMPETITOR_BUT_INITIAL_MARGIN_IS_TOO_LARGE"
else:CLASS="NO_SINGLE_DOMINANT_DECISION_BOUNDARY_MECHANISM"
print(f"TARGET_GAIN={TG:+.6f} | TOP1_GAIN={CG:+.6f} | MED_CLOSE={MC:+.6f} | FINAL_MARGIN={SM:.6f} | CLASS={CLASS}")
print("[16/18] LEAKAGE GUARD")
print("TARGET USED POST-HOC ONLY | ROUTER/PACKET/TERMINAL DISTRIBUTIONS SEALED PRE-TARGET | NO TARGET-BASED SELECTION OR DECODING")
print("[17/18] SCIENTIFIC AUDIT")
print("SAME FRESH-24 AS TEST330-332 | ROUTER=ARGMAX(DISP) | SAME RSS | CASE ISOLATED | NO TRAINING | NO WEIGHT UPDATE")
print("[18/18] INTEGRITY + SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("Hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["dev"]>1e-6:raise RuntimeError("Dose audit failure")
R={"schema":"akbascore.test333.v1","test":"TEST 333","start":START,"end":utc(),"model":MODEL_ID,"router":"ARGMAX_DISP_FROZEN","rss":RSS,"summary":{"rank_improved":IMP,"target_gain_mean":TG,"target_gain_positive":POS,"vanilla_margin_median":med("vanilla_margin"),"steered_margin_median":SM,"margin_close_median":MC,"steered_top1_gain_mean":CG,"margin_closed_cases":CLOSE,"target_gain_gt_final_top1_gain":TADV,"top1_unchanged":SAME,"median_competitors_remaining":med("competitors_remaining"),"mean_fraction_competitors_target_outgains":mean("fraction_current_competitors_target_outgains")},"classification":CLASS,"rows":ROWS,"integrity":{"same_fresh24_as_test330_332":True,"case_isolated":True,"cross_case_data":False,"target_engine_access":False,"target_posthoc_only":True,"router_frozen":True,"terminal_distributions_frozen_pre_target":True,"motor_layers":[0,25],"observation_layers":[26,27],"training":False,"weight_update":False,"calls":AUD["calls"],"max_dose_dev":AUD["dev"],"hooks_remaining":len(ACTIVE),"fp_start":FP0,"fp_end":fp()}}
sha=hashlib.sha256(canon(R)).hexdigest();rid=f"T333-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json";tp=ROOT/f"{rid}.txt"
jp.write_text(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False),encoding="utf-8")
tp.write_text("\n".join(["="*108,"TEST 333 - TERMINAL DECISION-BOUNDARY DECOMPOSITION","="*108,f"RANK_IMPROVED={IMP:.6f}",f"TARGET_GAIN={TG:+.6f}",f"VANILLA_MARGIN_MED={med('vanilla_margin'):.6f}",f"STEERED_MARGIN_MED={SM:.6f}",f"MARGIN_CLOSE_MED={MC:+.6f}",f"TOP1_GAIN={CG:+.6f}",f"CLASS={CLASS}","ROUTER FROZEN | TARGET POST-HOC ONLY | SAME RSS | WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]),encoding="utf-8")
print("="*108);print("TEST 333 COMPLETE")
print(f"TARGET GAIN={TG:+.6f} | TOP1 GAIN={CG:+.6f} | MED MARGIN={med('vanilla_margin'):.3f}->{SM:.3f} | CLOSE={MC:+.3f}")
print(f"RANK IMPROVED={IMP:.3f} | MARGIN CLOSED CASES={CLOSE:.3f} | TARGET>TOP1 GAIN={TADV:.3f}")
print("CLASS=",CLASS);print("ROUTER=ARGMAX(DISP) | TARGET POST-HOC ONLY | SAME RSS | WEIGHT INTEGRITY PASS | AKBASCORE HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
