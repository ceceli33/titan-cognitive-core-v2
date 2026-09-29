# ================================================================================================
# AKBASCORE - TEST 334
# FROZEN DISP ROUTER -> READOUT-CONTEXT PRIOR DECOMPOSITION
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
SEED=334;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYS_STD="You are a concise reasoning assistant. Use only the information in the prompt."
SYS_FORCE="Read the answer from the internal state. Return exactly one short answer. Never explain, refuse, say information is missing, or mention the prompt."
SYS_MIN="Answer with exactly one short answer."
TOTAL=28;H_EXPECT=3584;END=25;EPS=1e-8;TARGET_RSS=.250235055
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20
ROOT=Path("/content/AKBASCORE_TEST334");ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
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
CONTEXTS={"FORCED":SYS_FORCE,"STANDARD":SYS_STD,"MINIMAL":SYS_MIN}
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
def chat(q,sysmsg):
    s=tok.apply_chat_template([{"role":"system","content":sysmsg},{"role":"user","content":q}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to("cuda")
print("="*108);print("TEST 334 - FROZEN DISP ROUTER -> READOUT-CONTEXT PRIOR DECOMPOSITION");print("="*108);print("START:",START)
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
print("[2/18] ENGINE LOCK")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();raw=[IVME*(ZIRVE*np.exp(-SONUM*L)*(1+SONUM*L)+TABAN)/(ZIRVE+TABAN) for L in range(END+1)];scale=TARGET_RSS/np.sqrt(sum(x*x for x in raw));RHO=[x*scale for x in raw];RSS=float(np.sqrt(sum(x*x for x in RHO)))
print("FP:",[f"{x:.4f}" for x in FP0]);print(f"RSS={RSS:.9f} | ROUTER=ARGMAX(DISP) | CONTEXTS=FORCED/STANDARD/MINIMAL | TARGET SEALED")
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
def forward(q,sysmsg,packet=None):
    hs=install(packet) if packet is not None else []
    try:return model(**chat(q,sysmsg),use_cache=False,output_hidden_states=True,return_dict=True)
    finally:
        for h in hs:h.remove();ACTIVE.discard(id(h))
print("[6/18] BLIND DISP ROUTER - EXACT TEST330/333 FORCED CONTEXT")
SELECT=[];PACKETS=[]
for i,(_,q) in enumerate(ALL):
    v=forward(q,SYS_FORCE);vh=v.hidden_states[28][0,-1].float();aa=[]
    for x in BANK[i]:
        o=forward(q,SYS_FORCE,x["packet"]);h=o.hidden_states[28][0,-1].float();aa.append(float((h-vh).norm()/vh.norm().clamp_min(EPS)))
    j=int(np.argmax(aa));SELECT.append(j);PACKETS.append(BANK[i][j]["packet"]);tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,f"| ANCHOR={BANK[i][j]['anchor']!r} | DISP={aa[j]:.6f}")
print("[7/18] ROUTER + PACKETS SEALED")
print("ARGMAX(DISP) COMPLETE IN ORIGINAL FORCED CONTEXT | PACKETS FROZEN | TARGET_ACCESS=False")
print("[8/18] CONTEXT FORWARDS - TARGET SEALED")
FROZEN=[]
for i,(_,q) in enumerate(ALL):
    row={}
    for name,sysmsg in CONTEXTS.items():
        v=forward(q,sysmsg);s=forward(q,sysmsg,PACKETS[i]);vh=v.hidden_states[28][0,-1].float();sh=s.hidden_states[28][0,-1].float()
        vl=model.lm_head(vh.to(model.lm_head.weight.dtype)).float().cpu();sl=model.lm_head(sh.to(model.lm_head.weight.dtype)).float().cpu()
        row[name]={"vl":vl,"sl":sl,"vtop":int(torch.argmax(vl)),"stop":int(torch.argmax(sl)),"disp":float((sh-vh).norm()/vh.norm().clamp_min(EPS))}
    FROZEN.append(row);tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,"|"," | ".join(f"{n}:V={tok.decode([row[n]['vtop']])!r} S={tok.decode([row[n]['stop']])!r} D={row[n]['disp']:.3f}" for n in CONTEXTS))
print("[9/18] ALL READOUT CONTEXTS SEALED")
print("FORCED/STANDARD/MINIMAL VANILLA + STEERED DISTRIBUTIONS FROZEN BEFORE TARGET ACCESS")
print("[10/18] TARGET SEAL OPEN - CONTEXT DECOMPOSITION")
ROWS=[]
for i,t in enumerate(SEALED):
    tid=tok(t,add_special_tokens=False)["input_ids"][0];r={"target":t,"target_piece":tok.decode([tid]),"anchor":BANK[i][SELECT[i]]["anchor"],"contexts":{}}
    for name in CONTEXTS:
        f=FROZEN[i][name];vl=f["vl"];sl=f["sl"];vtop=f["vtop"];stop=f["stop"]
        vr=int((vl>vl[tid]).sum()+1);sr=int((sl>sl[tid]).sum()+1);gain=float(sl[tid]-vl[tid]);vm=float(vl[vtop]-vl[tid]);sm=float(sl[stop]-sl[tid])
        r["contexts"][name]={"vanilla_rank":vr,"steered_rank":sr,"gain":gain,"vanilla_margin":vm,"steered_margin":sm,"margin_close":vm-sm,"vanilla_top1":tok.decode([vtop]),"steered_top1":tok.decode([stop]),"disp":f["disp"]}
    ROWS.append(r);tag="SHOWCASE" if i==24 else f"{i+1:02d}/24"
    print(tag,"|"," | ".join(f"{n}:R={r['contexts'][n]['vanilla_rank']}->{r['contexts'][n]['steered_rank']} G={r['contexts'][n]['gain']:+.2f} M={r['contexts'][n]['steered_margin']:.2f}" for n in CONTEXTS))
print("[11/18] CONTEXT SUMMARY")
M=ROWS[:24];SUM={}
for name in CONTEXTS:
    c=[r["contexts"][name] for r in M];vr=np.array([x["vanilla_rank"] for x in c]);sr=np.array([x["steered_rank"] for x in c]);g=np.array([x["gain"] for x in c]);vm=np.array([x["vanilla_margin"] for x in c]);sm=np.array([x["steered_margin"] for x in c]);cl=vm-sm
    SUM[name]={"vanilla_med_rank":float(np.median(vr)),"steered_med_rank":float(np.median(sr)),"rank_improved":float(np.mean(sr<vr)),"gain":float(g.mean()),"positive":float(np.mean(g>0)),"vanilla_med_margin":float(np.median(vm)),"steered_med_margin":float(np.median(sm)),"margin_close":float(np.median(cl)),"top1":float(np.mean(sr==1))}
    x=SUM[name];print(f"{name:8s} | RANK={x['vanilla_med_rank']:.1f}->{x['steered_med_rank']:.1f} | IMP={x['rank_improved']:.3f} | G={x['gain']:+.3f} | M={x['vanilla_med_margin']:.2f}->{x['steered_med_margin']:.2f} | CLOSE={x['margin_close']:+.2f} | TOP1={x['top1']:.3f}")
print("[12/18] PRIOR DECOMPOSITION")
BASE=SUM["FORCED"]
for name in ("STANDARD","MINIMAL"):
    x=SUM[name];print(f"{name}-FORCED | VANILLA_MARGIN_SHIFT={x['vanilla_med_margin']-BASE['vanilla_med_margin']:+.3f} | STEERED_MARGIN_SHIFT={x['steered_med_margin']-BASE['steered_med_margin']:+.3f} | GAIN_SHIFT={x['gain']-BASE['gain']:+.3f}")
print("[13/18] CASEWISE BEST CONTEXT - POST-HOC DIAGNOSTIC ONLY")
BEST=[]
for i,r in enumerate(M):
    b=min(CONTEXTS,key=lambda n:r["contexts"][n]["steered_margin"]);BEST.append(b)
print("BEST_MARGIN_CONTEXT_COUNTS:",{n:BEST.count(n) for n in CONTEXTS})
print(f"ANY_CONTEXT_TOP1={np.mean([any(r['contexts'][n]['steered_rank']==1 for n in CONTEXTS) for r in M]):.6f}")
print("[14/18] SHOWCASE")
G=ROWS[24];print("-"*108);print("SOURCE :",SHOWCASE[0]);print("QUESTION:",SHOWCASE[1]);print("ANCHOR :",G["anchor"]);print("TARGET :",SHOWCASE[2])
for n in CONTEXTS:
    x=G["contexts"][n];print(f"{n:8s} | R={x['vanilla_rank']}->{x['steered_rank']} | G={x['gain']:+.4f} | M={x['vanilla_margin']:.4f}->{x['steered_margin']:.4f} | TOP1={x['steered_top1']!r}")
print("-"*108)
print("[15/18] PREDECLARED DIAGNOSTIC")
fm=SUM["FORCED"]["steered_med_margin"];sm=SUM["STANDARD"]["steered_med_margin"];mm=SUM["MINIMAL"]["steered_med_margin"];best=min(sm,mm)
if best<fm*.5:CLASS="SYSTEM_READOUT_CONTEXT_ACCOUNTS_FOR_LARGE_SHARE_OF_REMAINING_PRIOR_MARGIN"
elif best<fm*.8:CLASS="SYSTEM_READOUT_CONTEXT_PARTIALLY_CONTRIBUTES_TO_REMAINING_PRIOR_MARGIN"
elif abs(best-fm)<=fm*.2:CLASS="REMAINING_PRIOR_MARGIN_IS_LARGELY_CONTEXT_ROBUST"
else:CLASS="ALTERNATE_READOUT_CONTEXT_INCREASES_PRIOR_MARGIN"
print(f"FORCED_MARGIN={fm:.6f} | STANDARD_MARGIN={sm:.6f} | MINIMAL_MARGIN={mm:.6f} | CLASS={CLASS}")
print("[16/18] LEAKAGE GUARD")
print("CONTEXTS PREDECLARED | ROUTER/PACKETS/ALL DISTRIBUTIONS SEALED PRE-TARGET | TARGET POST-HOC ONLY | BEST CONTEXT NEVER USED FOR ROUTING OR INJECTION")
print("[17/18] SCIENTIFIC AUDIT")
print("SAME FRESH-24 AS TEST330-333 | ROUTER=ARGMAX(DISP) FROZEN IN ORIGINAL FORCED CONTEXT | SAME RSS | NO TRAINING | NO WEIGHT UPDATE")
print("[18/18] INTEGRITY + SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("Hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["dev"]>1e-6:raise RuntimeError("Dose audit failure")
R={"schema":"akbascore.test334.v1","test":"TEST 334","start":START,"end":utc(),"model":MODEL_ID,"router":"ARGMAX_DISP_FROZEN_FORCED","rss":RSS,"contexts":list(CONTEXTS),"summary":SUM,"classification":CLASS,"rows":ROWS,"integrity":{"same_fresh24_as_test330_333":True,"case_isolated":True,"cross_case_data":False,"target_engine_access":False,"target_posthoc_only":True,"router_frozen_before_context_test":True,"packets_frozen_before_context_test":True,"context_distributions_frozen_pre_target":True,"motor_layers":[0,25],"observation_layers":[26,27],"training":False,"weight_update":False,"calls":AUD["calls"],"max_dose_dev":AUD["dev"],"hooks_remaining":len(ACTIVE),"fp_start":FP0,"fp_end":fp()}}
sha=hashlib.sha256(canon(R)).hexdigest();rid=f"T334-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json";tp=ROOT/f"{rid}.txt"
jp.write_text(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False),encoding="utf-8")
tp.write_text("\n".join(["="*108,"TEST 334 - READOUT-CONTEXT PRIOR DECOMPOSITION","="*108]+[f"{n}: RANK {SUM[n]['vanilla_med_rank']:.1f}->{SUM[n]['steered_med_rank']:.1f} | GAIN {SUM[n]['gain']:+.6f} | MARGIN {SUM[n]['vanilla_med_margin']:.6f}->{SUM[n]['steered_med_margin']:.6f}" for n in CONTEXTS]+[f"CLASS={CLASS}","ROUTER/PACKET FROZEN | TARGET POST-HOC ONLY | SAME RSS | WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]),encoding="utf-8")
print("="*108);print("TEST 334 COMPLETE")
for n in CONTEXTS:print(f"{n}: MED_RANK={SUM[n]['steered_med_rank']:.1f} | GAIN={SUM[n]['gain']:+.3f} | MED_MARGIN={SUM[n]['steered_med_margin']:.3f}")
print("CLASS=",CLASS);print("ROUTER=ARGMAX(DISP) FROZEN | TARGET POST-HOC ONLY | SAME RSS | WEIGHT INTEGRITY PASS | AKBASCORE HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
