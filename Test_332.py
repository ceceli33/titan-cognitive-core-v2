# ================================================================================================
# AKBASCORE - TEST 332
# FROZEN DISP ROUTER -> TERMINAL VOCABULARY COMPETITION X-RAY
# SAME FRESH-24 | SAME PACKET | SAME RSS | TARGET-FREE TOP-K FROZEN BEFORE TARGET OPEN
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
SEED=332;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYS_STD="You are a concise reasoning assistant. Use only the information in the prompt."
SYS_FORCE="Read the answer from the internal state. Return exactly one short answer. Never explain, refuse, say information is missing, or mention the prompt."
TOTAL=28;H_EXPECT=3584;END=25;EPS=1e-8;TARGET_RSS=.250235055;TOPK=256
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20
ROOT=Path("/content/AKBASCORE_TEST332");ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
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
print("="*108);print("TEST 332 - FROZEN DISP ROUTER -> TERMINAL VOCABULARY COMPETITION X-RAY");print("="*108);print("START:",START)
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
print("FP:",[f"{x:.4f}" for x in FP0]);print(f"RSS={RSS:.9f} | ROUTER=ARGMAX(DISP) | TOPK={TOPK} | TARGET SEALED")
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
print("[8/18] SELECTED TERMINAL FORWARD + TARGET-FREE VOCAB FREEZE")
FROZEN=[]
for i,(_,q) in enumerate(ALL):
    v=PRE[i]["van"];s=forward(q,PRE[i]["anchors"][SELECT[i]]["packet"])
    vh=v.hidden_states[28][0,-1].float();sh=s.hidden_states[28][0,-1].float()
    vl=model.lm_head(vh.to(model.lm_head.weight.dtype)).float();sl=model.lm_head(sh.to(model.lm_head.weight.dtype)).float();delta=sl-vl
    sv,si=torch.topk(sl,TOPK);dv,di=torch.topk(delta,TOPK);vv,vi=torch.topk(vl,TOPK)
    FROZEN.append({"vl":vl.cpu(),"sl":sl.cpu(),"delta":delta.cpu(),"steer_top":si.cpu().tolist(),"delta_top":di.cpu().tolist(),"van_top":vi.cpu().tolist(),"steer_top_text":[tok.decode([x]) for x in si.tolist()[:20]],"delta_top_text":[tok.decode([x]) for x in di.tolist()[:20]]})
    tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,"| LOGIT_TOP5=",[repr(tok.decode([x])) for x in si.tolist()[:5]],"| DELTA_TOP5=",[repr(tok.decode([x])) for x in di.tolist()[:5]])
print("[9/18] TARGET-FREE CANDIDATE LISTS SEALED")
print(f"STEER_TOP{TOPK} + DELTA_TOP{TOPK} FROZEN FOR ALL CASES BEFORE TARGET ACCESS")
print("[10/18] TARGET SEAL OPEN - COMPETITION X-RAY")
ROWS=[]
for i,t in enumerate(SEALED):
    tids=tok(t,add_special_tokens=False)["input_ids"];tid=tids[0];f=FROZEN[i];vl=f["vl"];sl=f["sl"];dl=f["delta"]
    vr=int((vl>vl[tid]).sum().item()+1);sr=int((sl>sl[tid]).sum().item()+1);dr=int((dl>dl[tid]).sum().item()+1);gain=float(dl[tid])
    in_s=tid in f["steer_top"];in_d=tid in f["delta_top"];spos=f["steer_top"].index(tid)+1 if in_s else None;dpos=f["delta_top"].index(tid)+1 if in_d else None
    top1=f["steer_top"][0];margin=float(sl[top1]-sl[tid]);delta_top1=f["delta_top"][0];dmargin=float(dl[delta_top1]-dl[tid])
    r={"target":t,"target_first_id":tid,"target_first_piece":tok.decode([tid]),"target_tokens":len(tids),"anchor":PRE[i]["anchors"][SELECT[i]]["anchor"],"vanilla_rank":vr,"steered_rank":sr,"delta_rank":dr,"gain":gain,"steer_topk_hit":in_s,"steer_topk_pos":spos,"delta_topk_hit":in_d,"delta_topk_pos":dpos,"top1_id":top1,"top1_piece":tok.decode([top1]),"top1_margin":margin,"delta_top1_id":delta_top1,"delta_top1_piece":tok.decode([delta_top1]),"delta_top1_margin":dmargin};ROWS.append(r)
    tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,f"| T={t!r} | A={r['anchor']!r} | R={vr}->{sr} | dR={dr} | G={gain:+.3f} | TOP1={r['top1_piece']!r} M={margin:.2f} | dTOP1={r['delta_top1_piece']!r} dM={dmargin:.2f}")
print("[11/18] PRIMARY SUMMARY")
M=ROWS[:24];vr=np.array([r["vanilla_rank"] for r in M]);sr=np.array([r["steered_rank"] for r in M]);dr=np.array([r["delta_rank"] for r in M]);gg=np.array([r["gain"] for r in M]);mg=np.array([r["top1_margin"] for r in M]);dmg=np.array([r["delta_top1_margin"] for r in M])
print(f"VANILLA_MED_RANK={np.median(vr):.1f} | STEER_MED_RANK={np.median(sr):.1f} | DELTA_MED_RANK={np.median(dr):.1f}")
print(f"STEER_RANK_IMPROVED={np.mean(sr<vr):.6f} | GAIN={gg.mean():+.6f} | POSITIVE={np.mean(gg>0):.6f}")
print(f"STEER_TOP{TOPK}_HIT={np.mean([r['steer_topk_hit'] for r in M]):.6f} | DELTA_TOP{TOPK}_HIT={np.mean([r['delta_topk_hit'] for r in M]):.6f}")
print(f"MED_TOP1_MARGIN={np.median(mg):.6f} | MED_DELTA_TOP1_MARGIN={np.median(dmg):.6f}")
print("[12/18] COVERAGE LADDER")
for K in (1,5,10,25,50,100,256):
    sh=np.mean([r["steered_rank"]<=K for r in M]);dh=np.mean([r["delta_rank"]<=K for r in M]);print(f"TOP{K:03d} | STEER={sh:.6f} | DELTA={dh:.6f}")
print("[13/18] COMPETITOR TYPES")
pieces=[r["top1_piece"] for r in M];dpieces=[r["delta_top1_piece"] for r in M]
print("STEER TOP1:",[repr(x) for x in pieces]);print("DELTA TOP1:",[repr(x) for x in dpieces])
print("[14/18] SHOWCASE")
G=ROWS[24];print("-"*108);print("SOURCE :",SHOWCASE[0]);print("QUESTION:",SHOWCASE[1]);print("ANCHOR :",G["anchor"]);print("TARGET :",SHOWCASE[2])
print(f"TARGET FIRST PIECE={G['target_first_piece']!r} | RANK={G['vanilla_rank']}->{G['steered_rank']} | DELTA_RANK={G['delta_rank']} | GAIN={G['gain']:+.4f}")
print(f"STEER TOP1={G['top1_piece']!r} | MARGIN={G['top1_margin']:.4f} | DELTA TOP1={G['delta_top1_piece']!r} | DELTA_MARGIN={G['delta_top1_margin']:.4f}")
print("STEER TOP20:",[repr(x) for x in FROZEN[24]["steer_top_text"]]);print("DELTA TOP20:",[repr(x) for x in FROZEN[24]["delta_top_text"]]);print("-"*108)
print("[15/18] PREDECLARED DIAGNOSTIC")
SH256=float(np.mean([r["steer_topk_hit"] for r in M]));DH256=float(np.mean([r["delta_topk_hit"] for r in M]));D50=float(np.mean([r["delta_rank"]<=50 for r in M]))
if DH256>=.70 and DH256>SH256:CLASS="TARGET_SIGNAL_PRESENT_IN_DELTA_BUT_MASKED_BY_BASE_LOGIT_PRIOR"
elif SH256>=.50:CLASS="TARGET_SIGNAL_ENTERS_TERMINAL_CANDIDATE_SET_BUT_LOSES_COMPETITION"
elif gg.mean()>0:CLASS="TARGET_LOGIT_RISES_BUT_REMAINS_BELOW_PRACTICAL_DECODING_RANGE"
else:CLASS="NO_STABLE_TERMINAL_TARGET_COMPETITION_SIGNAL"
print(f"DELTA_TOP50={D50:.6f} | STEER_TOP256={SH256:.6f} | DELTA_TOP256={DH256:.6f} | CLASS={CLASS}")
print("[16/18] LEAKAGE GUARD")
print("TOP-K LISTS FROZEN TARGET-FREE | TARGET USED ONLY AFTER ROUTER + TERMINAL DISTRIBUTIONS + CANDIDATE LISTS SEALED | NO TARGET-BASED DECODING")
print("[17/18] SCIENTIFIC AUDIT")
print("SAME FRESH-24 AS TEST330/331 | ROUTER=ARGMAX(DISP) | SAME RSS | CASE ISOLATED | NO TRAINING | NO WEIGHT UPDATE")
print("[18/18] INTEGRITY + SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("Hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["dev"]>1e-6:raise RuntimeError("Dose audit failure")
R={"schema":"akbascore.test332.v1","test":"TEST 332","start":START,"end":utc(),"model":MODEL_ID,"router":"ARGMAX_DISP_FROZEN","rss":RSS,"topk":TOPK,"summary":{"vanilla_median_rank":float(np.median(vr)),"steered_median_rank":float(np.median(sr)),"delta_median_rank":float(np.median(dr)),"rank_improved":float(np.mean(sr<vr)),"gain_mean":float(gg.mean()),"gain_positive":float(np.mean(gg>0)),"steer_top256":SH256,"delta_top256":DH256,"delta_top50":D50,"median_top1_margin":float(np.median(mg)),"median_delta_top1_margin":float(np.median(dmg))},"classification":CLASS,"rows":ROWS,"integrity":{"same_fresh24_as_test330_331":True,"case_isolated":True,"cross_case_data":False,"target_engine_access":False,"target_posthoc_only":True,"router_frozen":True,"candidate_lists_frozen_pre_target":True,"motor_layers":[0,25],"observation_layers":[26,27],"training":False,"weight_update":False,"calls":AUD["calls"],"max_dose_dev":AUD["dev"],"hooks_remaining":len(ACTIVE),"fp_start":FP0,"fp_end":fp()}}
sha=hashlib.sha256(canon(R)).hexdigest();rid=f"T332-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json";tp=ROOT/f"{rid}.txt"
jp.write_text(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False),encoding="utf-8")
tp.write_text("\n".join(["="*108,"TEST 332 - FROZEN DISP ROUTER -> TERMINAL VOCABULARY COMPETITION X-RAY","="*108,f"VANILLA_MED_RANK={np.median(vr):.1f}",f"STEER_MED_RANK={np.median(sr):.1f}",f"DELTA_MED_RANK={np.median(dr):.1f}",f"GAIN={gg.mean():+.6f}",f"STEER_TOP256={SH256:.6f}",f"DELTA_TOP256={DH256:.6f}",f"DELTA_TOP50={D50:.6f}",f"CLASS={CLASS}","ROUTER FROZEN | TARGET POST-HOC ONLY | TOP-K FROZEN PRE-TARGET | WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]),encoding="utf-8")
print("="*108);print("TEST 332 COMPLETE")
print(f"VANILLA MED={np.median(vr):.1f} | STEER MED={np.median(sr):.1f} | DELTA MED={np.median(dr):.1f} | GAIN={gg.mean():+.6f}")
print(f"STEER TOP256={SH256:.3f} | DELTA TOP256={DH256:.3f} | DELTA TOP50={D50:.3f}")
print("CLASS=",CLASS);print("ROUTER=ARGMAX(DISP) | TARGET POST-HOC ONLY | SAME RSS | WEIGHT INTEGRITY PASS | AKBASCORE HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
