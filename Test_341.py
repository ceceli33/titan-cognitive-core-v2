# ================================================================================================
# AKBASCORE - TEST 341
# TOKEN-POSITION INJECTION WINDOW X-RAY
# SAME FRESH-24 | SAME ARGMAX(DISP) PACKET | SAME SEASC DOSE | SINGLE-POSITION WRITE
# TARGET SEALED THROUGH BANK + ROUTER + ALL FORWARDS | TARGET USED POST-HOC ONLY
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
SEED=341;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYS_STD="You are a concise reasoning assistant. Use only the information in the prompt."
SYS_FORCE="Read the answer from the internal state. Return exactly one short answer. Never explain, refuse, say information is missing, or mention the prompt."
TOTAL=28;H=3584;END=25;EPS=1e-8;TARGET_RSS=.250235055;IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20
ROOT=Path("/content/AKBASCORE_TEST341");ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
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
ALL=CASES+[SHOWCASE];SEALED=tuple(x[2] for x in ALL)
QNULLS=["What information is requested?","Which unknown detail should be recovered?","Answer the question using the missing information.","What should be identified?","Which detail is being asked about?"]
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
def prompt(q):
    s=tok.apply_chat_template([{"role":"system","content":SYS_FORCE},{"role":"user","content":q}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to("cuda")
def prompt_std(q):
    s=tok.apply_chat_template([{"role":"system","content":SYS_STD},{"role":"user","content":q}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to("cuda")
print("="*108);print("TEST 341 - TOKEN-POSITION INJECTION WINDOW X-RAY");print("="*108);print("START:",START)
print("[1/18] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers
if len(layers)!=TOTAL or model.config.hidden_size!=H:raise RuntimeError("Architecture mismatch")
torch.cuda.synchronize();print(f"OK | {MODEL_ID} | 28L | H={H} | {next(model.parameters()).dtype} | {time.perf_counter()-t:.2f}s")
print("[2/18] ENGINE LOCK")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();raw=[IVME*(ZIRVE*np.exp(-SONUM*L)*(1+SONUM*L)+TABAN)/(ZIRVE+TABAN) for L in range(END+1)];scale=TARGET_RSS/np.sqrt(sum(x*x for x in raw));RHO=[x*scale for x in raw];RSS=float(np.sqrt(sum(x*x for x in RHO)))
print("FP:",[f"{x:.4f}" for x in FP0]);print(f"RSS={RSS:.9f} | L0-L25 | SINGLE TOKEN-POSITION WRITE | TARGET SEALED")
@torch.inference_mode()
def states(text):
    o=model(**prompt_std(text),use_cache=False,output_hidden_states=True,return_dict=True)
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
print("[4/18] TARGET-FREE PACKET BANK")
BANK=[]
for i,(s,q,_) in enumerate(ALL):
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
ACTIVE=set();AUD={"calls":0,"err":0.0}
def install(packet,pos=None,allpos=False):
    hs=[]
    for L in range(END+1):
        rho=RHO[L]
        def hk(mod,args,out,L=L,rho=rho):
            y=out[0] if isinstance(out,tuple) else out;r=y.clone();n=y.shape[1]
            idx=list(range(n)) if allpos else [pos if pos>=0 else n+pos]
            idx=[j for j in idx if 0<=j<n]
            if idx:
                z=y[:,idx,:].float();zn=z.norm(dim=-1,keepdim=True).clamp_min(EPS);d=packet[L].to(z.device).view(1,1,-1)*zn*rho;r[:,idx,:]=(z+d).to(y.dtype)
                AUD["calls"]+=len(idx);AUD["err"]=max(AUD["err"],abs(float((d.norm(dim=-1)/zn.squeeze(-1)).mean())-rho))
            return (r,)+out[1:] if isinstance(out,tuple) else r
        h=layers[L].register_forward_hook(hk);hs.append(h);ACTIVE.add(id(h))
    return hs
@torch.inference_mode()
def run(text,packet=None,pos=None,allpos=False):
    hs=install(packet,pos,allpos) if packet is not None else []
    try:return model(**prompt(text),use_cache=False,output_hidden_states=True,return_dict=True)
    finally:
        for h in hs:h.remove();ACTIVE.discard(id(h))
def logits(o):
    h=o.hidden_states[28][0,-1].to(model.lm_head.weight.dtype);return model.lm_head(h).float().cpu()
print("[5/18] FROZEN DISP ROUTER - EXACT TEST340 METHOD")
SELECT=[];PACKETS=[]
for i,(_,q,_) in enumerate(ALL):
    v=run(q);vh=v.hidden_states[28][0,-1].float();ds=[]
    for x in BANK[i]:
        o=run(q,x["packet"],allpos=True);h=o.hidden_states[28][0,-1].float();ds.append(float((h-vh).norm()/vh.norm().clamp_min(EPS)))
    j=int(np.argmax(ds));SELECT.append(j);PACKETS.append(BANK[i][j]["packet"]);tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,f"| ANCHOR={BANK[i][j]['anchor']!r} | DISP={ds[j]:.6f}")
print("[6/18] PACKETS FROZEN")
print("TARGET_ACCESS=False | POSITION WINDOW DOES NOT PARTICIPATE IN PACKET SELECTION")
print("[7/18] TOKEN POSITION MAP")
MAPS=[]
for i,(_,q,_) in enumerate(ALL):
    e=prompt(q);ids=e["input_ids"][0].tolist();n=len(ids);tail=list(range(max(0,n-12),n))
    MAPS.append({"n":n,"positions":tail,"tokens":[tok.decode([ids[p]]) for p in tail]})
    tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,f"| N={n} | LAST12={MAPS[-1]['tokens']}")
print("[8/18] VANILLA + FULL-WRITE BASELINES - TARGET SEALED")
FROZEN=[]
for i,(_,q,_) in enumerate(ALL):
    lv=logits(run(q));lf=logits(run(q,PACKETS[i],allpos=True));wins=[]
    for p,tkn in zip(MAPS[i]["positions"],MAPS[i]["tokens"]):
        lw=logits(run(q,PACKETS[i],pos=p));wins.append({"pos":p,"rel":p-MAPS[i]["n"],"token":tkn,"logits":lw})
    FROZEN.append({"vanilla":lv,"full":lf,"windows":wins})
    tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,f"| VANILLA/FULL + {len(wins)} SINGLE-POS WINDOWS FROZEN")
print("[9/18] ALL FORWARDS SEALED")
print("NO TARGET TOKEN USED IN WINDOW SEARCH | TARGET NOW OPENS FOR POST-HOC X-RAY")
print("[10/18] POST-HOC TARGET WINDOW X-RAY")
ROWS=[]
for i,target in enumerate(SEALED):
    tid=tok(target,add_special_tokens=False)["input_ids"][0];f=FROZEN[i]
    def rank(x):return int((x>x[tid]).sum()+1)
    vr,fr=rank(f["vanilla"]),rank(f["full"]);fg=float(f["full"][tid]-f["vanilla"][tid]);ww=[]
    for w in f["windows"]:
        r=rank(w["logits"]);g=float(w["logits"][tid]-f["vanilla"][tid]);ww.append({"pos":w["pos"],"rel":w["rel"],"token":w["token"],"rank":r,"gain":g})
    best=max(ww,key=lambda x:x["gain"]);bestr=min(ww,key=lambda x:x["rank"])
    row={"target":target,"anchor":BANK[i][SELECT[i]]["anchor"],"vanilla_rank":vr,"full_rank":fr,"full_gain":fg,"windows":ww,"best_gain_rel":best["rel"],"best_gain_token":best["token"],"best_gain":best["gain"],"best_gain_rank":best["rank"],"best_rank_rel":bestr["rel"],"best_rank_token":bestr["token"],"best_rank":bestr["rank"],"best_rank_gain":bestr["gain"]};ROWS.append(row)
    tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,f"| V={vr} FULL={fr} G={fg:+.2f} | BEST_GAIN rel={best['rel']:+d} tok={best['token']!r} G={best['gain']:+.2f} R={best['rank']} | BEST_R={bestr['rank']} rel={bestr['rel']:+d}")
print("[11/18] POSITION PROFILE")
M=ROWS[:24];RELS=list(range(-12,0));PROFILE=[]
for rel in RELS:
    xs=[]
    for r in M:
        z=next((x for x in r["windows"] if x["rel"]==rel),None)
        if z:xs.append(z)
    if xs:
        d={"rel":rel,"n":len(xs),"mean_gain":float(np.mean([x["gain"] for x in xs])),"positive":float(np.mean([x["gain"]>0 for x in xs])),"rank_improved":float(np.mean([x["rank"]<M[j]["vanilla_rank"] for j,r in enumerate(M) for x in [next((z for z in r["windows"] if z["rel"]==rel),None)] if x is not None]))}
        PROFILE.append(d);print(f"REL={rel:+d} | N={d['n']:02d} | GAIN={d['mean_gain']:+.4f} | POS={d['positive']:.3f} | RANK_IMP={d['rank_improved']:.3f}")
print("[12/18] PREDECLARED FIXED WINDOWS")
FIX={}
for rel in [-1,-2,-3,-4,-6,-8,-10,-12]:
    xs=[]
    for r in M:
        z=next((x for x in r["windows"] if x["rel"]==rel),None)
        if z:xs.append((r,z))
    if xs:
        FIX[str(rel)]={"n":len(xs),"mean_gain":float(np.mean([z["gain"] for _,z in xs])),"positive":float(np.mean([z["gain"]>0 for _,z in xs])),"rank_improved":float(np.mean([z["rank"]<r["vanilla_rank"] for r,z in xs])),"median_rank":float(np.median([z["rank"] for _,z in xs]))}
        x=FIX[str(rel)];print(f"REL={rel:+d} | G={x['mean_gain']:+.4f} | POS={x['positive']:.3f} | IMP={x['rank_improved']:.3f} | MED={x['median_rank']:.1f}")
print("[13/18] ORACLE WINDOW UPPER BOUND - POST-HOC ONLY")
oracle_imp=float(np.mean([r["best_rank"]<r["vanilla_rank"] for r in M]));oracle_pos=float(np.mean([r["best_gain"]>0 for r in M]));oracle_med=float(np.median([r["best_rank"] for r in M]));full_med=float(np.median([r["full_rank"] for r in M]));van_med=float(np.median([r["vanilla_rank"] for r in M]))
print(f"VANILLA_MED={van_med:.1f} | FULL_WRITE_MED={full_med:.1f} | POSTHOC_ORACLE_WINDOW_MED={oracle_med:.1f} | ORACLE_IMP={oracle_imp:.3f} | ORACLE_POS={oracle_pos:.3f}")
print("[14/18] BEST FIXED TARGET-FREE POSITION AFTER PROFILE")
bestprof=max(PROFILE,key=lambda x:x["mean_gain"]);print(f"DEVELOPMENT BEST REL={bestprof['rel']:+d} | GAIN={bestprof['mean_gain']:+.4f} | POS={bestprof['positive']:.3f} | IMP={bestprof['rank_improved']:.3f}")
print("NOTE: THIS POSITION IS DEVELOPMENT-SELECTION, NOT HELD-OUT VALIDATION")
print("[15/18] SHOWCASE")
G=ROWS[24];print("-"*108);print("SOURCE :",SHOWCASE[0]);print("QUESTION:",SHOWCASE[1]);print("TARGET :",SHOWCASE[2]);print("ANCHOR :",G["anchor"]);print(f"V={G['vanilla_rank']} | FULL={G['full_rank']} G={G['full_gain']:+.4f} | BEST_GAIN rel={G['best_gain_rel']:+d} token={G['best_gain_token']!r} G={G['best_gain']:+.4f} R={G['best_gain_rank']} | BEST_R={G['best_rank']} rel={G['best_rank_rel']:+d}");print("-"*108)
print("[16/18] PREDECLARED DIAGNOSTIC")
fixed_supported=any(x["positive"]>=.70 and x["rank_improved"]>=.70 and x["mean_gain"]>0 for x in PROFILE)
oracle_supported=oracle_imp>=.80 and oracle_pos>=.80
if fixed_supported:CLASS="TOKEN_POSITION_READOUT_WINDOW_SUPPORTED"
elif oracle_supported:CLASS="POSITION_DEPENDENCE_SUPPORTED_BUT_FIXED_WINDOW_NOT_YET_IDENTIFIED"
else:CLASS="TOKEN_POSITION_WINDOW_HYPOTHESIS_NOT_SUPPORTED_IN_THIS_FORM"
print(f"FIXED_WINDOW_GATE={fixed_supported} | POSTHOC_ORACLE_GATE={oracle_supported} | CLASS={CLASS}")
print("[17/18] SCIENTIFIC AUDIT")
print("SAME FRESH-24 | SAME PACKET COMPILER | SAME ARGMAX(DISP) ROUTER | SAME RSS | SINGLE-POSITION WRITES | WINDOW NEVER USED FOR PACKET SELECTION | ORACLE POST-HOC ONLY | NO TRAINING | NO WEIGHT UPDATE")
print("[18/18] INTEGRITY + SAVE")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("Hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["err"]>1e-5:raise RuntimeError(f"Dose failure {AUD['err']}")
S={"vanilla_median_rank":van_med,"full_write_median_rank":full_med,"oracle_window_median_rank":oracle_med,"oracle_rank_improved":oracle_imp,"oracle_positive_gain":oracle_pos,"development_best_rel":bestprof["rel"],"development_best_mean_gain":bestprof["mean_gain"],"development_best_positive":bestprof["positive"],"development_best_rank_improved":bestprof["rank_improved"]}
R={"schema":"akbascore.test341.v1","test":"TEST 341","start":START,"end":utc(),"model":MODEL_ID,"rss":RSS,"summary":S,"fixed_windows":FIX,"profile":PROFILE,"classification":CLASS,"rows":ROWS,"integrity":{"same_fresh24":True,"same_packet_compiler":True,"router":"ARGMAX(DISP)","target_engine_access":False,"target_posthoc_only":True,"oracle_posthoc_only":True,"training":False,"weight_update":False,"motor_layers":[0,25],"hooks_remaining":len(ACTIVE),"fp_start":FP0,"fp_end":fp()}}
sha=hashlib.sha256(canon(R)).hexdigest();rid=f"T341-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json";tp=ROOT/f"{rid}.txt"
jp.write_text(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False),encoding="utf-8")
tp.write_text("\n".join(["="*108,"TEST 341 - TOKEN-POSITION INJECTION WINDOW X-RAY","="*108]+[f"{k}={v}" for k,v in S.items()]+[f"CLASS={CLASS}",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]),encoding="utf-8")
print("="*108);print("TEST 341 COMPLETE");print(f"VANILLA_MED={van_med:.1f} | FULL_MED={full_med:.1f} | ORACLE_WINDOW_MED={oracle_med:.1f} | ORACLE_IMP={oracle_imp:.3f}")
print(f"DEV_BEST_REL={bestprof['rel']:+d} | GAIN={bestprof['mean_gain']:+.3f} | POS={bestprof['positive']:.3f} | IMP={bestprof['rank_improved']:.3f}")
print("CLASS=",CLASS);print("TARGET POST-HOC ONLY | WEIGHT INTEGRITY PASS | HOOKS=0");print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
