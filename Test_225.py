# ================================================================================================
# AKBASCORE - TEST 325
# RELATIONAL PACKET X-RAY + BLIND GENERATION
# VANILLA VS FLAT PACKET VS RELATIONAL PACKET | CASE ISOLATED | TARGET POST-HOC
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
SEED=325;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYS_STD="You are a concise reasoning assistant. Use only the information in the prompt."
SYS_FORCE="Read the answer from the internal state. Return exactly one short answer. Never explain, refuse, say information is missing, or mention the prompt."
TOTAL=28;H_EXPECT=3584;END=25;EPS=1e-8;TARGET_RSS=.250235055;BRIDGE=.20;SLOT_K=8;REL_K=4;MAX_NEW=12
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20
ROOT=Path("/content/AKBASCORE_TEST325");ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
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
print("="*108);print("TEST 325 - RELATIONAL PACKET X-RAY + BLIND GENERATION");print("="*108);print("START:",START)
print("[1/17] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
if not tok.is_fast:raise RuntimeError("Fast tokenizer required")
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers
if len(layers)!=TOTAL or model.config.hidden_size!=H_EXPECT:raise RuntimeError("Architecture mismatch")
torch.cuda.synchronize();print(f"OK | {MODEL_ID} | 28L | H={model.config.hidden_size} | VOCAB={model.config.vocab_size} | {next(model.parameters()).dtype} | {time.perf_counter()-t:.2f}s")
print("[2/17] ENGINE LOCK")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();base=[IVME*(ZIRVE*np.exp(-SONUM*L)*(1+SONUM*L)+TABAN)/(ZIRVE+TABAN) for L in range(END+1)];scale=TARGET_RSS/np.sqrt(sum(x*x for x in base));RHO=[x*scale for x in base];RSS=float(np.sqrt(sum(x*x for x in RHO)))
print("FP:",[f"{x:.4f}" for x in FP0]);print(f"RSS={RSS:.9f} | BRIDGE={BRIDGE:.3f} | REL_K={REL_K}")
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
print("[3/17] QUERY NULL CACHE")
QN=[states(x) for x in QNULLS];print("NULL READY")
print("[4/17] SOURCE COMPILE - TARGET SEALED")
FLAT=[];REL=[];Q=[];META=[]
for i,(s,q) in enumerate(ALL):
    qh=states(q);qv=[unit(torch.stack([unit(qh[L]-QN[j][L]) for j in range(len(QN))]).mean(0)) for L in range(TOTAL)];sp=spans(s)
    sc=torch.tensor([torch.dot(x["v"][27],qv[27]).item() for x in sp]);k=min(SLOT_K,len(sp));vals,idx=torch.topk(sc,k);w=torch.softmax(vals/.025,0)
    flat=[unit(sum(sp[int(idx[j])]["v"][L]*w[j].to(sp[0]["v"][L].device) for j in range(k))) for L in range(TOTAL)]
    rk=min(REL_K,len(sp));ridx=torch.topk(sc,rk).indices.tolist();rel=[]
    for L in range(TOTAL):
        edges=[]
        for a in ridx:
            for b in range(len(sp)):
                if b==a:continue
                d=sp[b]["v"][L]-sp[a]["v"][L];score=torch.dot(unit(d),qv[L]);edges.append((score,d))
        edges=sorted(edges,key=lambda x:float(x[0]),reverse=True)[:rk]
        rel.append(unit(sum(unit(d)*torch.softmax(torch.stack([x[0] for x in edges]),0)[j] for j,(_,d) in enumerate(edges))))
    FLAT.append(flat);REL.append(rel);Q.append(qv);META.append({"sp":sp,"anchors":ridx})
    print(("SHOWCASE" if i==24 else f"{i+1:02d}/24"),"| REL_ANCHORS=",[sp[x]["text"] for x in ridx],"| BANK=",len(sp))
print("[5/17] SOURCE REMOVED")
print("BLIND RUNTIME RECEIVES QUESTION ONLY | TARGET_ACCESS=False | CASE_ISOLATED=True")
ACTIVE=set();AUD={"calls":0,"dev":0.0}
def hooks(packet):
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
    hs=hooks(packet) if packet is not None else []
    try:return model(**chat(q,SYS_FORCE),use_cache=False,output_hidden_states=True,return_dict=True)
    finally:
        for h in hs:h.remove();ACTIVE.discard(id(h))
print("[6/17] VANILLA / FLAT / RELATIONAL X-RAY")
XR=[]
for i,(_,q) in enumerate(ALL):
    v=forward(q);f=forward(q,FLAT[i]);r=forward(q,REL[i]);rows=[]
    for L in range(TOTAL):
        hv=v.hidden_states[L+1][0,-1].float();hf=f.hidden_states[L+1][0,-1].float();hr=r.hidden_states[L+1][0,-1].float()
        df=unit(hf-hv);dr=unit(hr-hv);rows.append({"layer":L,"flat_q":float(torch.dot(df,Q[i][L])),"rel_q":float(torch.dot(dr,Q[i][L])),"flat_rel":float(torch.dot(df,REL[i][L])),"rel_rel":float(torch.dot(dr,REL[i][L]))})
    XR.append(rows);tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(f"{tag} | L27 FLAT_Q={rows[27]['flat_q']:+.4f} REL_Q={rows[27]['rel_q']:+.4f} | FLAT_REL={rows[27]['flat_rel']:+.4f} REL_REL={rows[27]['rel_rel']:+.4f}")
print("[7/17] RELATIONAL BINDING SEPARATION")
SEP=[]
for i in range(len(ALL)):
    sp=META[i]["sp"];anchors=META[i]["anchors"];good=[];bad=[]
    for a in anchors:
        for b in range(len(sp)):
            if b==a:continue
            d=unit(sp[b]["v"][27]-sp[a]["v"][27]);x=float(torch.dot(d,Q[i][27]))
            (good if b in anchors else bad).append(x)
    g=float(np.mean(good)) if good else 0.;b=float(np.mean(bad)) if bad else 0.;SEP.append(g-b)
    print(("SHOWCASE" if i==24 else f"{i+1:02d}/24"),f"| LINK_SEP={g-b:+.6f}")
print("[8/17] GENERATION")
@torch.inference_mode()
def gen(q,packet=None):
    x=chat(q,SYS_FORCE);n=x["input_ids"].shape[1];hs=hooks(packet) if packet is not None else []
    try:y=model.generate(**x,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)
    finally:
        for h in hs:h.remove();ACTIVE.discard(id(h))
    return tok.decode(y[0,n:],skip_special_tokens=True).strip()
VAN=[];FOUT=[];ROUT=[]
for i,(_,q) in enumerate(ALL):
    a=gen(q);b=gen(q,FLAT[i]);c=gen(q,REL[i]);VAN.append(a);FOUT.append(b);ROUT.append(c)
    tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(f"{tag} | VANILLA={a!r} | FLAT={b!r} | REL={c!r}")
print("[9/17] TARGET SEAL OPEN - POST-HOC ONLY")
ROWS=[]
for i,t in enumerate(SEALED):
    nt=normtxt(t);row={"target":t,"vanilla":VAN[i],"flat":FOUT[i],"rel":ROUT[i],"vanilla_exact":normtxt(VAN[i])==nt,"flat_exact":normtxt(FOUT[i])==nt,"rel_exact":normtxt(ROUT[i])==nt,"vanilla_mention":nt in normtxt(VAN[i]),"flat_mention":nt in normtxt(FOUT[i]),"rel_mention":nt in normtxt(ROUT[i]),"link_sep":SEP[i]}
    ROWS.append(row);tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(f"{tag} | TARGET={t!r} | VAN={VAN[i]!r} | FLAT={FOUT[i]!r} | REL={ROUT[i]!r}")
print("[10/17] PRIMARY 24 BEHAVIOR")
M=ROWS[:24]
def rate(k):return float(np.mean([x[k] for x in M]))
VE,VM=rate("vanilla_exact"),rate("vanilla_mention");FE,FM=rate("flat_exact"),rate("flat_mention");RE,RM=rate("rel_exact"),rate("rel_mention")
print(f"VANILLA EXACT={VE:.3f} MENTION={VM:.3f} | FLAT EXACT={FE:.3f} MENTION={FM:.3f} | REL EXACT={RE:.3f} MENTION={RM:.3f}")
print("[11/17] X-RAY SUMMARY")
FQ=float(np.mean([x[27]["flat_q"] for x in XR[:24]]));RQ=float(np.mean([x[27]["rel_q"] for x in XR[:24]]));FR=float(np.mean([x[27]["flat_rel"] for x in XR[:24]]));RR=float(np.mean([x[27]["rel_rel"] for x in XR[:24]]));LS=float(np.mean(SEP[:24]))
print(f"L27 FLAT_Q={FQ:+.6f} | REL_Q={RQ:+.6f} | FLAT_REL={FR:+.6f} | REL_REL={RR:+.6f} | LINK_SEP={LS:+.6f}")
print("[12/17] GOLDEN GATE SHOWCASE")
G=ROWS[24]
print("-"*108);print("SOURCE :",SHOWCASE[0]);print("QUESTION:",SHOWCASE[1]);print("VANILLA:",G["vanilla"]);print("FLAT   :",G["flat"]);print("REL    :",G["rel"]);print("TARGET :",SHOWCASE[2]);print("LINK_SEP:",f"{G['link_sep']:+.6f}");print("-"*108)
print("[13/17] PREDECLARED GATE")
G1=RM>FM;G2=RR>FR;G3=LS>0
GATE=bool((G1 and G2) or (G1 and G3));CLASS="RELATIONAL_PACKET_SUPPORTED" if GATE else "RELATIONAL_PACKET_NOT_YET_SUPPORTED"
print(f"REL_MENTION_GT_FLAT={G1} | REL_GEOMETRY_GT_FLAT={G2} | LINK_SEP_POSITIVE={G3} | CLASS={CLASS}")
print("[14/17] INTERPRETATION GUARD")
print("RELATIONAL PACKET IS AN ENDOGENOUS QUERY-CONDITIONED DIFFERENCE-EDGE PACKET, NOT A PROOF OF SYMBOLIC RELATION STORAGE.")
print("[15/17] SCIENTIFIC AUDIT")
print("VANILLA HAS NO HOOKS. FLAT AND REL USE IDENTICAL SEASC DOSE ENVELOPE. TARGETS OPEN ONLY AFTER ALL GENERATIONS.")
print("[16/17] CASE AUDIT")
print("CASE_ISOLATED=True | CROSS_CASE_DATA=False | TARGET_ENGINE_ACCESS=False | TARGET_POSTHOC_ONLY=True | TRAINING=False | WEIGHT_UPDATE=False")
print("[17/17] INTEGRITY + SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("Hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["dev"]>1e-6:raise RuntimeError("Dose audit failure")
R={"schema":"akbascore.test325.v1","test":"TEST 325","start":START,"end":utc(),"model":MODEL_ID,
"question":"Does preserving endogenous pairwise difference structure improve relation-sensitive transport and blind retrieval over a flat packet?",
"summary":{"vanilla_exact":VE,"vanilla_mention":VM,"flat_exact":FE,"flat_mention":FM,"rel_exact":RE,"rel_mention":RM,"l27_flat_q":FQ,"l27_rel_q":RQ,"l27_flat_rel":FR,"l27_rel_rel":RR,"mean_link_sep":LS},
"showcase":G,"classification":CLASS,"gate":{"rel_mention_gt_flat":G1,"rel_geometry_gt_flat":G2,"link_sep_positive":G3,"result":"PASS" if GATE else "FAIL"},
"rows":ROWS,"xray":XR,"integrity":{"case_isolated":True,"cross_case_data":False,"target_engine_access":False,"target_posthoc_only":True,"vanilla_hooks":False,"training":False,"weight_update":False,"rss":RSS,"calls":AUD["calls"],"max_dose_dev":AUD["dev"],"hooks_remaining":len(ACTIVE),"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
sha=hashlib.sha256(canon(R)).hexdigest();rid=f"T325-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json";tp=ROOT/f"{rid}.txt"
jp.write_bytes(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False).encode())
tp.write_text("\n".join(["="*108,"TEST 325 - RELATIONAL PACKET X-RAY + BLIND GENERATION","="*108,
f"VANILLA EXACT={VE:.3f} MENTION={VM:.3f}",f"FLAT EXACT={FE:.3f} MENTION={FM:.3f}",f"REL EXACT={RE:.3f} MENTION={RM:.3f}",
f"L27 FLAT_Q={FQ:+.6f} REL_Q={RQ:+.6f} FLAT_REL={FR:+.6f} REL_REL={RR:+.6f} LINK_SEP={LS:+.6f}",
f"GOLDEN_GATE VANILLA={G['vanilla']!r}",f"GOLDEN_GATE FLAT={G['flat']!r}",f"GOLDEN_GATE REL={G['rel']!r}",
f"CLASS={CLASS} | GATE={'PASS' if GATE else 'FAIL'}","CASE_ISOLATED=TRUE | TARGET POST-HOC ONLY | WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]),encoding="utf-8")
print("="*108);print("TEST 325 COMPLETE")
print(f"VANILLA MENTION={VM:.3f} | FLAT MENTION={FM:.3f} | REL MENTION={RM:.3f}")
print(f"L27 FLAT_REL={FR:+.6f} | REL_REL={RR:+.6f} | LINK_SEP={LS:+.6f}")
print(f"GOLDEN GATE VANILLA={G['vanilla']!r}");print(f"GOLDEN GATE FLAT   ={G['flat']!r}");print(f"GOLDEN GATE REL    ={G['rel']!r}")
print(f"CLASS={CLASS} | GATE={'PASS' if GATE else 'FAIL'}")
print("CASE_ISOLATED=TRUE | TARGET POST-HOC ONLY | WEIGHT INTEGRITY PASS | AKBASCORE HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
