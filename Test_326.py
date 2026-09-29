# ================================================================================================
# AKBASCORE - TEST 326 - FIXED
# NON-COLLAPSED MULTI-CHANNEL RELATIONAL PACKET
# VANILLA VS FLAT VS COLLAPSED_REL VS MULTI_CHANNEL | SAME RSS | TARGET POST-HOC
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
SEED=326;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYS_STD="You are a concise reasoning assistant. Use only the information in the prompt."
SYS_FORCE="Read the answer from the internal state. Return exactly one short answer. Never explain, refuse, say information is missing, or mention the prompt."
TOTAL=28;H_EXPECT=3584;END=25;EPS=1e-8;TARGET_RSS=.250235055;SLOT_K=8;REL_K=4;MAX_NEW=12
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20;BANDS=((0,6),(7,12),(13,18),(19,25))
ROOT=Path("/content/AKBASCORE_TEST326");ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
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
print("="*108);print("TEST 326 - NON-COLLAPSED MULTI-CHANNEL RELATIONAL PACKET - FIXED");print("="*108);print("START:",START)
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
print("FP:",[f"{x:.4f}" for x in FP0]);print(f"RSS={RSS:.9f} | BANDS={BANDS} | REL_K={REL_K} | SAME DOSE ALL STEERED ARMS")
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
print("[4/18] SOURCE COMPILE - TARGET SEALED")
FLAT=[];COLL=[];MULTI=[];Q=[];META=[]
for i,(s,q) in enumerate(ALL):
    qh=states(q);qv=[unit(torch.stack([unit(qh[L]-QN[j][L]) for j in range(len(QN))]).mean(0)) for L in range(TOTAL)];sp=spans(s)
    sc=torch.tensor([torch.dot(x["v"][27],qv[27]).item() for x in sp]);k=min(SLOT_K,len(sp));vals,idx=torch.topk(sc,k);w=torch.softmax(vals/.025,0)
    flat=[unit(sum(sp[int(idx[j])]["v"][L]*w[j].to(sp[0]["v"][L].device) for j in range(k))) for L in range(TOTAL)]
    rk=min(REL_K,len(sp));anchors=torch.topk(sc,rk).indices.tolist();channels=[[] for _ in range(rk)]
    for ci,a in enumerate(anchors):
        for L in range(TOTAL):
            cand=[]
            for b in range(len(sp)):
                if b==a:continue
                d=sp[b]["v"][L]-sp[a]["v"][L];cand.append((torch.dot(unit(d),qv[L]),d,b))
            cand.sort(key=lambda x:float(x[0]),reverse=True);channels[ci].append(unit(cand[0][1]))
    coll=[unit(sum(ch[L] for ch in channels)) for L in range(TOTAL)]
    multi=[]
    for L in range(TOTAL):
        if L<=END:ci=next(j for j,(a,b) in enumerate(BANDS) if a<=L<=b)
        else:ci=len(BANDS)-1
        multi.append(channels[min(ci,len(channels)-1)][L])
    FLAT.append(flat);COLL.append(coll);MULTI.append(multi);Q.append(qv);META.append({"sp":sp,"anchors":anchors,"channels":channels})
    tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,"| ANCHORS=",[sp[a]["text"] for a in anchors],"| CHANNELS=",len(channels),"| BANK=",len(sp))
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
print("[6/18] FOUR-ARM X-RAY")
XR=[]
for i,(_,q) in enumerate(ALL):
    v=forward(q);f=forward(q,FLAT[i]);c=forward(q,COLL[i]);m=forward(q,MULTI[i]);rr=[]
    for L in range(TOTAL):
        hv=v.hidden_states[L+1][0,-1].float();df=unit(f.hidden_states[L+1][0,-1].float()-hv);dc=unit(c.hidden_states[L+1][0,-1].float()-hv);dm=unit(m.hidden_states[L+1][0,-1].float()-hv)
        rr.append({"layer":L,"flat_q":float(torch.dot(df,Q[i][L])),"coll_q":float(torch.dot(dc,Q[i][L])),"multi_q":float(torch.dot(dm,Q[i][L])),
        "flat_rel":float(torch.dot(df,COLL[i][L])),"coll_rel":float(torch.dot(dc,COLL[i][L])),"multi_rel":float(torch.dot(dm,COLL[i][L]))})
    XR.append(rr);x=rr[27];tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(f"{tag} | L27 Q F={x['flat_q']:+.4f} C={x['coll_q']:+.4f} M={x['multi_q']:+.4f} | REL F={x['flat_rel']:+.4f} C={x['coll_rel']:+.4f} M={x['multi_rel']:+.4f}")
print("[7/18] CHANNEL SURVIVAL X-RAY")
SURV=[]
for i in range(len(ALL)):
    q=ALL[i][1];v=forward(q);m=forward(q,MULTI[i]);d=unit(m.hidden_states[28][0,-1].float()-v.hidden_states[28][0,-1].float())
    cs=[float(torch.dot(d,META[i]["channels"][j][27])) for j in range(len(META[i]["channels"]))];SURV.append(cs)
    print(("SHOWCASE" if i==24 else f"{i+1:02d}/24"),"| CHANNEL_L27=",[round(x,4) for x in cs],"| MAX=",f"{max(cs):+.4f}")
print("[8/18] BLIND GENERATION")
@torch.inference_mode()
def gen(q,packet=None):
    x=chat(q,SYS_FORCE);n=x["input_ids"].shape[1];hs=install(packet) if packet is not None else []
    try:y=model.generate(**x,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)
    finally:
        for h in hs:h.remove();ACTIVE.discard(id(h))
    return tok.decode(y[0,n:],skip_special_tokens=True).strip()
VAN=[];FOUT=[];COUT=[];MOUT=[]
for i,(_,q) in enumerate(ALL):
    a=gen(q);b=gen(q,FLAT[i]);c=gen(q,COLL[i]);d=gen(q,MULTI[i]);VAN.append(a);FOUT.append(b);COUT.append(c);MOUT.append(d)
    tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(f"{tag} | VAN={a!r} | FLAT={b!r} | COLL={c!r} | MULTI={d!r}")
print("[9/18] TARGET SEAL OPEN - POST-HOC ONLY")
ROWS=[]
for i,t in enumerate(SEALED):
    nt=normtxt(t);r={"target":t,"vanilla":VAN[i],"flat":FOUT[i],"collapsed":COUT[i],"multi":MOUT[i],
    "vanilla_exact":normtxt(VAN[i])==nt,"flat_exact":normtxt(FOUT[i])==nt,"collapsed_exact":normtxt(COUT[i])==nt,"multi_exact":normtxt(MOUT[i])==nt,
    "vanilla_mention":nt in normtxt(VAN[i]),"flat_mention":nt in normtxt(FOUT[i]),"collapsed_mention":nt in normtxt(COUT[i]),"multi_mention":nt in normtxt(MOUT[i])}
    ROWS.append(r);tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(f"{tag} | TARGET={t!r} | VAN={VAN[i]!r} | FLAT={FOUT[i]!r} | COLL={COUT[i]!r} | MULTI={MOUT[i]!r}")
print("[10/18] PRIMARY 24 BEHAVIOR")
M=ROWS[:24]
def rate(k):return float(np.mean([x[k] for x in M]))
VE,VM=rate("vanilla_exact"),rate("vanilla_mention");FE,FM=rate("flat_exact"),rate("flat_mention");CE,CM=rate("collapsed_exact"),rate("collapsed_mention");ME,MM=rate("multi_exact"),rate("multi_mention")
print(f"VANILLA EXACT={VE:.3f} MENTION={VM:.3f} | FLAT EXACT={FE:.3f} MENTION={FM:.3f} | COLL EXACT={CE:.3f} MENTION={CM:.3f} | MULTI EXACT={ME:.3f} MENTION={MM:.3f}")
print("[11/18] X-RAY SUMMARY")
def avg(k):return float(np.mean([x[27][k] for x in XR[:24]]))
FQ,CQ,MQ=avg("flat_q"),avg("coll_q"),avg("multi_q");FR,CR,MR=avg("flat_rel"),avg("coll_rel"),avg("multi_rel")
print(f"L27 QUERY | FLAT={FQ:+.6f} COLL={CQ:+.6f} MULTI={MQ:+.6f}")
print(f"L27 REL   | FLAT={FR:+.6f} COLL={CR:+.6f} MULTI={MR:+.6f}")
print("[12/18] CHANNEL SUMMARY")
S=np.asarray(SURV[:24]);SM=S.mean(0).tolist();print("MEAN CHANNEL L27:",[round(x,6) for x in SM],"| MEAN_MAX=",f"{float(S.max(1).mean()):+.6f}")
print("[13/18] GOLDEN GATE SHOWCASE")
G=ROWS[24]
print("-"*108);print("SOURCE :",SHOWCASE[0]);print("QUESTION:",SHOWCASE[1]);print("ANCHORS:",[META[24]["sp"][a]["text"] for a in META[24]["anchors"]]);print("VANILLA:",G["vanilla"]);print("FLAT   :",G["flat"]);print("COLL   :",G["collapsed"]);print("MULTI  :",G["multi"]);print("TARGET :",SHOWCASE[2]);print("-"*108)
print("[14/18] PREDECLARED GATE")
G1=MM>CM;G2=MR>CR;G3=MQ>CQ
GATE=bool(G1 and (G2 or G3));CLASS="NONCOLLAPSED_RELATIONAL_CHANNELS_SUPPORTED" if GATE else "NONCOLLAPSED_RELATIONAL_CHANNELS_NOT_SUPPORTED"
print(f"MULTI_MENTION_GT_COLL={G1} | MULTI_REL_GT_COLL={G2} | MULTI_QUERY_GT_COLL={G3} | CLASS={CLASS}")
print("[15/18] DOSE FAIRNESS")
print(f"FLAT_RSS={RSS:.9f} | COLL_RSS={RSS:.9f} | MULTI_RSS={RSS:.9f} | EXTRA_CHANNEL_DOSE=0")
print("[16/18] INTERPRETATION GUARD")
print("MULTI CHANNEL PRESERVES FOUR RELATIONAL DIRECTIONS ACROSS DEPTH BANDS. L26-L27 ARE OBSERVATION ONLY AND RECEIVE NO SEASC INJECTION.")
print("[17/18] SCIENTIFIC AUDIT")
print("VANILLA HAS NO HOOKS | SAME SEASC ENVELOPE FOR ALL STEERED ARMS | TARGET POST-HOC ONLY | NO TRAINING | NO CROSS-CASE DATA")
print("[18/18] INTEGRITY + SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("Hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["dev"]>1e-6:raise RuntimeError("Dose audit failure")
R={"schema":"akbascore.test326.v2","test":"TEST 326 FIXED","start":START,"end":utc(),"model":MODEL_ID,
"question":"Does preserving four endogenous relational directions as separate depth channels outperform collapsing them into one vector at identical RSS?",
"summary":{"vanilla_exact":VE,"vanilla_mention":VM,"flat_exact":FE,"flat_mention":FM,"collapsed_exact":CE,"collapsed_mention":CM,"multi_exact":ME,"multi_mention":MM,
"l27_flat_q":FQ,"l27_coll_q":CQ,"l27_multi_q":MQ,"l27_flat_rel":FR,"l27_coll_rel":CR,"l27_multi_rel":MR,"channel_mean":SM},
"showcase":G,"classification":CLASS,"gate":{"multi_mention_gt_coll":G1,"multi_rel_gt_coll":G2,"multi_query_gt_coll":G3,"result":"PASS" if GATE else "FAIL"},
"rows":ROWS,"xray":XR,"channel_survival":SURV,"integrity":{"case_isolated":True,"cross_case_data":False,"target_engine_access":False,"target_posthoc_only":True,
"vanilla_hooks":False,"same_rss":True,"training":False,"weight_update":False,"rss":RSS,"bands":BANDS,"motor_layers":[0,25],"observation_layers":[26,27],
"calls":AUD["calls"],"max_dose_dev":AUD["dev"],"hooks_remaining":len(ACTIVE),"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
sha=hashlib.sha256(canon(R)).hexdigest();rid=f"T326-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json";tp=ROOT/f"{rid}.txt"
jp.write_bytes(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False).encode())
tp.write_text("\n".join(["="*108,"TEST 326 - NON-COLLAPSED MULTI-CHANNEL RELATIONAL PACKET - FIXED","="*108,
f"VANILLA EXACT={VE:.3f} MENTION={VM:.3f}",f"FLAT EXACT={FE:.3f} MENTION={FM:.3f}",f"COLL EXACT={CE:.3f} MENTION={CM:.3f}",f"MULTI EXACT={ME:.3f} MENTION={MM:.3f}",
f"L27 QUERY FLAT={FQ:+.6f} COLL={CQ:+.6f} MULTI={MQ:+.6f}",f"L27 REL FLAT={FR:+.6f} COLL={CR:+.6f} MULTI={MR:+.6f}",
f"GOLDEN_GATE VANILLA={G['vanilla']!r}",f"GOLDEN_GATE FLAT={G['flat']!r}",f"GOLDEN_GATE COLL={G['collapsed']!r}",f"GOLDEN_GATE MULTI={G['multi']!r}",
f"CLASS={CLASS} | GATE={'PASS' if GATE else 'FAIL'}","CASE_ISOLATED=TRUE | TARGET POST-HOC ONLY | SAME RSS | WEIGHT INTEGRITY PASS",
f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]),encoding="utf-8")
print("="*108);print("TEST 326 COMPLETE")
print(f"VANILLA MENTION={VM:.3f} | FLAT={FM:.3f} | COLL={CM:.3f} | MULTI={MM:.3f}")
print(f"L27 QUERY COLL={CQ:+.6f} -> MULTI={MQ:+.6f} | REL COLL={CR:+.6f} -> MULTI={MR:+.6f}")
print(f"GOLDEN GATE VANILLA={G['vanilla']!r}");print(f"GOLDEN GATE FLAT   ={G['flat']!r}");print(f"GOLDEN GATE COLL   ={G['collapsed']!r}");print(f"GOLDEN GATE MULTI  ={G['multi']!r}")
print(f"CLASS={CLASS} | GATE={'PASS' if GATE else 'FAIL'}")
print("CASE_ISOLATED=TRUE | TARGET POST-HOC ONLY | SAME RSS | WEIGHT INTEGRITY PASS | AKBASCORE HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
