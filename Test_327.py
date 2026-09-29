# ================================================================================================
# AKBASCORE - TEST 327
# RELATIONAL CHANNEL ORACLE X-RAY
# VANILLA VS CH1/CH2/CH3/CH4 | FULL L0-L25 | SAME RSS | TARGET POST-HOC
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
SEED=327;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYS_STD="You are a concise reasoning assistant. Use only the information in the prompt."
SYS_FORCE="Read the answer from the internal state. Return exactly one short answer. Never explain, refuse, say information is missing, or mention the prompt."
TOTAL=28;H_EXPECT=3584;END=25;EPS=1e-8;TARGET_RSS=.250235055;REL_K=4;MAX_NEW=12
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20
ROOT=Path("/content/AKBASCORE_TEST327");ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
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
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def normtxt(s):return re.sub(r"[^\w]+"," ",s,flags=re.UNICODE).strip().casefold()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
def chat(q,sysmsg):
    s=tok.apply_chat_template([{"role":"system","content":sysmsg},{"role":"user","content":q}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to("cuda")
print("="*108);print("TEST 327 - RELATIONAL CHANNEL ORACLE X-RAY");print("="*108);print("START:",START)
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
print("FP:",[f"{x:.4f}" for x in FP0]);print(f"RSS={RSS:.9f} | CHANNELS={REL_K} | EACH CHANNEL FULL L0-L25")
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
print("[4/18] FOUR RELATIONAL CHANNELS - TARGET SEALED")
CHANNELS=[];Q=[];META=[]
for i,(s,q) in enumerate(ALL):
    qh=states(q);qv=[unit(torch.stack([unit(qh[L]-QN[j][L]) for j in range(len(QN))]).mean(0)) for L in range(TOTAL)];sp=spans(s)
    sc=torch.tensor([torch.dot(x["v"][27],qv[27]).item() for x in sp]);rk=min(REL_K,len(sp));anchors=torch.topk(sc,rk).indices.tolist();chs=[]
    for a in anchors:
        ch=[]
        for L in range(TOTAL):
            cand=[]
            for b in range(len(sp)):
                if b==a:continue
                d=sp[b]["v"][L]-sp[a]["v"][L];cand.append((torch.dot(unit(d),qv[L]),d,b))
            cand.sort(key=lambda x:float(x[0]),reverse=True);ch.append(unit(cand[0][1]))
        chs.append(ch)
    while len(chs)<REL_K:chs.append(chs[-1])
    CHANNELS.append(chs);Q.append(qv);META.append({"sp":sp,"anchors":anchors})
    tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,"| ANCHORS=",[sp[a]["text"] for a in anchors],"| BANK=",len(sp))
print("[5/18] SOURCE REMOVED")
print("QUESTION-ONLY BLIND RUNTIME | TARGET_ACCESS=False | CROSS_CASE=False")
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
print("[6/18] VANILLA + CHANNEL X-RAY")
XR=[];PREFILL=[]
for i,(_,q) in enumerate(ALL):
    v=forward(q);hv=v.hidden_states[28][0,-1].float();vl=model.lm_head(hv.to(model.lm_head.weight.dtype)).float()
    row={"vanilla_hidden":hv.detach().clone(),"vanilla_logits":vl.detach().clone(),"channels":[]}
    for c in range(REL_K):
        o=forward(q,CHANNELS[i][c]);h=o.hidden_states[28][0,-1].float();d=unit(h-hv);lg=model.lm_head(h.to(model.lm_head.weight.dtype)).float()
        row["channels"].append({"hidden":h.detach().clone(),"logits":lg.detach().clone(),"q":float(torch.dot(d,Q[i][27])),"self":float(torch.dot(d,CHANNELS[i][c][27])),"disp":float((h-hv).norm()/hv.norm().clamp_min(EPS))})
    PREFILL.append(row);XR.append([{"q":x["q"],"self":x["self"],"disp":x["disp"]} for x in row["channels"]])
    tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,"|", " ".join(f"C{c+1}:Q={row['channels'][c]['q']:+.3f} SELF={row['channels'][c]['self']:+.3f} D={row['channels'][c]['disp']:.3f}" for c in range(REL_K)))
print("[7/18] BLIND GENERATION")
@torch.inference_mode()
def gen(q,packet=None):
    x=chat(q,SYS_FORCE);n=x["input_ids"].shape[1];hs=install(packet) if packet is not None else []
    try:y=model.generate(**x,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)
    finally:
        for h in hs:h.remove();ACTIVE.discard(id(h))
    return tok.decode(y[0,n:],skip_special_tokens=True).strip()
VAN=[];OUT=[]
for i,(_,q) in enumerate(ALL):
    a=gen(q);oo=[gen(q,CHANNELS[i][c]) for c in range(REL_K)];VAN.append(a);OUT.append(oo)
    tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,"| VAN=",repr(a),"|"," | ".join(f"C{c+1}={oo[c]!r}" for c in range(REL_K)))
print("[8/18] TARGET SEAL OPEN - POST-HOC ORACLE ONLY")
ROWS=[]
for i,t in enumerate(SEALED):
    tids=tok(t,add_special_tokens=False)["input_ids"]
    if not tids:raise RuntimeError("Empty target tokenization")
    tid=tids[0];base=PREFILL[i]["vanilla_logits"];brank=int((base>base[tid]).sum().item()+1);blogit=float(base[tid]);chs=[]
    for c in range(REL_K):
        lg=PREFILL[i]["channels"][c]["logits"];rank=int((lg>lg[tid]).sum().item()+1);gain=float(lg[tid]-base[tid]);nt=normtxt(t);o=OUT[i][c]
        chs.append({"channel":c+1,"rank":rank,"rank_delta":brank-rank,"target_logit_gain":gain,"exact":normtxt(o)==nt,"mention":nt in normtxt(o),"output":o})
    oracle=max(range(REL_K),key=lambda c:chs[c]["target_logit_gain"]);oracle_rank=min(range(REL_K),key=lambda c:chs[c]["rank"])
    r={"target":t,"target_first_token_id":tid,"vanilla":VAN[i],"vanilla_rank":brank,"vanilla_target_logit":blogit,"channels":chs,"oracle_logit_channel":oracle+1,"oracle_rank_channel":oracle_rank+1}
    ROWS.append(r);tag="SHOWCASE" if i==24 else f"{i+1:02d}/24"
    print(f"{tag} | TARGET={t!r} | V_RANK={brank} | "+" | ".join(f"C{x['channel']} R={x['rank']} DG={x['target_logit_gain']:+.3f} M={int(x['mention'])}" for x in chs)+f" | ORACLE_LOGIT=C{oracle+1} ORACLE_RANK=C{oracle_rank+1}")
print("[9/18] PRIMARY 24 ORACLE SUMMARY")
M=ROWS[:24];oracle_gain=[max(x["target_logit_gain"] for x in r["channels"]) for r in M];oracle_rank=[min(x["rank"] for x in r["channels"]) for r in M]
best_log=[r["oracle_logit_channel"] for r in M];best_rank=[r["oracle_rank_channel"] for r in M]
base_rank=[r["vanilla_rank"] for r in M];oracle_mention=[any(x["mention"] for x in r["channels"]) for r in M];oracle_exact=[any(x["exact"] for x in r["channels"]) for r in M]
print(f"ORACLE_LOGIT_GAIN_MEAN={np.mean(oracle_gain):+.6f} | POSITIVE={np.mean(np.array(oracle_gain)>0):.6f}")
print(f"VANILLA_RANK_MEDIAN={np.median(base_rank):.1f} | ORACLE_RANK_MEDIAN={np.median(oracle_rank):.1f} | RANK_IMPROVED={np.mean(np.array(oracle_rank)<np.array(base_rank)):.6f}")
print(f"ORACLE_ANY_MENTION={np.mean(oracle_mention):.6f} | ORACLE_ANY_EXACT={np.mean(oracle_exact):.6f}")
print("[10/18] CHANNEL DISTRIBUTION")
for c in range(1,REL_K+1):print(f"C{c} | LOGIT_ORACLE={best_log.count(c)}/24 | RANK_ORACLE={best_rank.count(c)}/24")
print("[11/18] CHANNEL MEAN TARGET EFFECT")
for c in range(REL_K):
    g=[r["channels"][c]["target_logit_gain"] for r in M];ri=[r["channels"][c]["rank"] for r in M];im=[r["channels"][c]["rank"]<r["vanilla_rank"] for r in M]
    print(f"C{c+1} | GAIN={np.mean(g):+.6f} | POS={np.mean(np.array(g)>0):.6f} | MED_RANK={np.median(ri):.1f} | RANK_IMPROVED={np.mean(im):.6f}")
print("[12/18] BLIND GEOMETRY VS POST-HOC WINNER")
for c in range(REL_K):
    q=[XR[i][c]["q"] for i in range(24)];s=[XR[i][c]["self"] for i in range(24)]
    print(f"C{c+1} | MEAN_Q={np.mean(q):+.6f} | MEAN_SELF={np.mean(s):+.6f}")
print("[13/18] GOLDEN GATE SHOWCASE")
G=ROWS[24]
print("-"*108);print("SOURCE :",SHOWCASE[0]);print("QUESTION:",SHOWCASE[1]);print("ANCHORS:",[META[24]["sp"][a]["text"] for a in META[24]["anchors"]]);print("VANILLA:",G["vanilla"])
for x in G["channels"]:print(f"CHANNEL {x['channel']}: {x['output']!r} | TARGET_GAIN={x['target_logit_gain']:+.4f} | RANK={x['rank']}")
print("TARGET :",SHOWCASE[2]);print("POSTHOC ORACLE LOGIT CHANNEL:",G["oracle_logit_channel"]);print("POSTHOC ORACLE RANK CHANNEL:",G["oracle_rank_channel"]);print("-"*108)
print("[14/18] PREDECLARED DIAGNOSTIC")
OG=float(np.mean(oracle_gain));ORI=float(np.mean(np.array(oracle_rank)<np.array(base_rank)));OM=float(np.mean(oracle_mention))
if OM>0:CLASS="TARGET_INFORMATION_REACHES_BEHAVIOR_IN_AT_LEAST_ONE_CHANNEL"
elif OG>0 and ORI>=.75:CLASS="TARGET_INFORMATION_PRESENT_BUT_ROUTING_DECODING_MISSING"
else:CLASS="RELATIONAL_CHANNELS_NOT_TARGET_SPECIFIC_ENOUGH"
print(f"ORACLE_GAIN={OG:+.6f} | ORACLE_RANK_IMPROVED={ORI:.6f} | ORACLE_MENTION={OM:.6f} | CLASS={CLASS}")
print("[15/18] ORACLE GUARD")
print("ORACLE IS POST-HOC ANALYSIS ONLY. TARGET IDENTITY NEVER SELECTS, BUILDS, ROUTES OR INJECTS A CHANNEL.")
print("[16/18] DOSE FAIRNESS")
print(f"C1_RSS={RSS:.9f} | C2_RSS={RSS:.9f} | C3_RSS={RSS:.9f} | C4_RSS={RSS:.9f} | EACH RUN USES ONE CHANNEL ONLY")
print("[17/18] SCIENTIFIC AUDIT")
print("VANILLA NO HOOKS | EACH CHANNEL L0-L25 | L26-L27 OFF | CASE ISOLATED | TARGET POST-HOC | NO TRAINING | NO WEIGHT UPDATE")
print("[18/18] INTEGRITY + SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("Hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["dev"]>1e-6:raise RuntimeError("Dose audit failure")
R={"schema":"akbascore.test327.v1","test":"TEST 327","start":START,"end":utc(),"model":MODEL_ID,
"question":"Does any individually preserved endogenous relational channel contain stronger target-specific information than the combined packet?",
"summary":{"oracle_logit_gain_mean":OG,"oracle_logit_positive":float(np.mean(np.array(oracle_gain)>0)),"vanilla_rank_median":float(np.median(base_rank)),
"oracle_rank_median":float(np.median(oracle_rank)),"oracle_rank_improved":ORI,"oracle_any_mention":OM,"oracle_any_exact":float(np.mean(oracle_exact)),
"logit_oracle_counts":[best_log.count(c) for c in range(1,5)],"rank_oracle_counts":[best_rank.count(c) for c in range(1,5)]},
"classification":CLASS,"showcase":G,"rows":ROWS,"blind_xray":XR,
"integrity":{"case_isolated":True,"cross_case_data":False,"target_engine_access":False,"target_posthoc_only":True,"oracle_posthoc_only":True,"vanilla_hooks":False,
"motor_layers":[0,25],"observation_layers":[26,27],"same_rss_each_channel":True,"rss":RSS,"training":False,"weight_update":False,"calls":AUD["calls"],
"max_dose_dev":AUD["dev"],"hooks_remaining":len(ACTIVE),"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
sha=hashlib.sha256(canon(R)).hexdigest();rid=f"T327-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json";tp=ROOT/f"{rid}.txt"
jp.write_bytes(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False).encode())
tp.write_text("\n".join(["="*108,"TEST 327 - RELATIONAL CHANNEL ORACLE X-RAY","="*108,
f"ORACLE_LOGIT_GAIN_MEAN={OG:+.6f}",f"ORACLE_RANK_IMPROVED={ORI:.6f}",f"VANILLA_RANK_MEDIAN={np.median(base_rank):.1f}",f"ORACLE_RANK_MEDIAN={np.median(oracle_rank):.1f}",
f"ORACLE_ANY_MENTION={OM:.6f}",f"ORACLE_ANY_EXACT={np.mean(oracle_exact):.6f}",f"LOGIT_ORACLE_COUNTS={[best_log.count(c) for c in range(1,5)]}",
f"RANK_ORACLE_COUNTS={[best_rank.count(c) for c in range(1,5)]}",f"GOLDEN_GATE VANILLA={G['vanilla']!r}",
*[f"GOLDEN_GATE C{x['channel']}={x['output']!r} GAIN={x['target_logit_gain']:+.4f} RANK={x['rank']}" for x in G["channels"]],
f"CLASS={CLASS}","CASE_ISOLATED=TRUE | TARGET POST-HOC ONLY | SAME RSS | WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]),encoding="utf-8")
print("="*108);print("TEST 327 COMPLETE")
print(f"ORACLE LOGIT GAIN={OG:+.6f} | ORACLE RANK IMPROVED={ORI:.3f} | VANILLA MED RANK={np.median(base_rank):.1f} -> ORACLE={np.median(oracle_rank):.1f}")
print(f"ORACLE ANY MENTION={OM:.3f} | EXACT={np.mean(oracle_exact):.3f}")
print("LOGIT ORACLE COUNTS:",[best_log.count(c) for c in range(1,5)]);print("RANK ORACLE COUNTS:",[best_rank.count(c) for c in range(1,5)])
print(f"GOLDEN GATE VANILLA={G['vanilla']!r}")
for x in G["channels"]:print(f"GOLDEN GATE C{x['channel']}={x['output']!r} | GAIN={x['target_logit_gain']:+.4f} | RANK={x['rank']}")
print("CLASS=",CLASS);print("CASE_ISOLATED=TRUE | TARGET POST-HOC ONLY | SAME RSS | WEIGHT INTEGRITY PASS | AKBASCORE HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
