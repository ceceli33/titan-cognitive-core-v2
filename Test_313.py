# ================================================================================================
# AKBASCORE - TEST 313
# ABSTENTION-PRIOR OVERRIDE ASSAY
# TEST312 ENGINE LOCKED | STANDARD BLIND vs FORCED LATENT READOUT
# CASE-ISOLATED | TARGET SEALED | NO TARGET-HEAD | NO TRAINING | NO DOSE SWEEP
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
SEED=313;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYS_STD="You are a concise reasoning assistant. Use only the information in the prompt."
SYS_FORCE="Read the answer from the internal state. Return exactly one short answer. Never explain, refuse, say information is missing, or mention the prompt."
TOTAL=28;H_EXPECT=3584;END=25;EPS=1e-8;TARGET_RSS=.250235055;BRIDGE_REL=.20;SLOT_K=8;MAX_NEW=8
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20
ROOT=Path("/content/AKBASCORE_TEST313");ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
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
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
def chat_ids(q,system):
    s=tok.apply_chat_template([{"role":"system","content":system},{"role":"user","content":q}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to("cuda")
def raw_ids(s):return tok(s,return_tensors="pt",add_special_tokens=False).to("cuda")
print("="*108);print("TEST 313 - ABSTENTION-PRIOR OVERRIDE ASSAY");print("="*108);print("START:",START)
print("[1/15] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size
if len(layers)!=TOTAL or H!=H_EXPECT:raise RuntimeError("Architecture mismatch")
torch.cuda.synchronize();print(f"OK | {MODEL_ID} | 28L | H={H} | VOCAB={model.config.vocab_size} | {next(model.parameters()).dtype} | {time.perf_counter()-t:.2f}s")
print("[2/15] TEST312 ENGINE LOCK")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();base=[IVME*(ZIRVE*np.exp(-SONUM*L)*(1+SONUM*L)+TABAN)/(ZIRVE+TABAN) for L in range(END+1)];scale=TARGET_RSS/np.sqrt(sum(x*x for x in base));RHO=[x*scale for x in base];RSS=float(np.sqrt(sum(x*x for x in RHO)))
print("FP:",[f"{x:.4f}" for x in FP0]);print(f"RSS={RSS:.9f} | BRIDGE={BRIDGE_REL:.3f} | SLOT_K={SLOT_K}")
@torch.inference_mode()
def last_states(text):
    o=model(**chat_ids(text,SYS_STD),use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)]
@torch.inference_mode()
def token_states(text):
    x=raw_ids(text);o=model(**x,use_cache=False,output_hidden_states=True,return_dict=True)
    return x.input_ids[0].detach().cpu(),[o.hidden_states[L+1][0].float().detach().clone() for L in range(TOTAL)]
print("[3/15] TARGET-FREE QUERY NULL CACHE")
QNULL=[last_states(x) for x in QNULLS];print("NULL READY")
print("[4/15] TEST312 PACKET COMPILER")
PACK=[];ORTH=[];META=[]
for i,(statement,question) in enumerate(INPUTS):
    qh=last_states(question);qv=[unit(torch.stack([unit(qh[L]-QNULL[j][L]) for j in range(len(QNULL))]).mean(0)) for L in range(TOTAL)]
    ids,hs=token_states(statement);slots=[];sm=[]
    for pos,tid in enumerate(ids.tolist()):
        txt=tok.decode([tid]);clean=txt.strip()
        if not clean or all(not c.isalnum() for c in clean):continue
        slots.append([unit(hs[L][pos]) for L in range(TOTAL)]);sm.append(txt)
    scores=torch.tensor([torch.dot(x[27],qv[27]).item() for x in slots]);k=min(SLOT_K,len(slots));vals,idx=torch.topk(scores,k);w=torch.softmax(vals/.025,dim=0)
    p=[unit(sum(slots[int(idx[j])][L]*w[j].to(slots[0][L].device) for j in range(k))) for L in range(TOTAL)];o=[]
    for L in range(TOTAL):
        r=p[L]-torch.dot(p[L],qv[L])*qv[L];o.append(unit(r) if r.norm()>EPS else p[L])
    PACK.append(p);ORTH.append(o);META.append({"slots":[sm[int(x)] for x in idx],"pq27":float(torch.dot(p[27],qv[27])),"oq27":float(torch.dot(o[27],qv[27]))})
    print(f"{i+1:02d}/24 | {repr(META[-1]['slots'])} | P.Q={META[-1]['pq27']:+.4f} -> O.Q={META[-1]['oq27']:+.7f}")
print("[5/15] CASE-ISOLATION AUDIT")
print("ONE STATEMENT + ONE QUESTION | CROSS_CASE_DATA=False | TARGET_ACCESS=False")
print("[6/15] CAUSAL ENGINE")
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
def forward(q,system,packet,bridge):
    hs=install(packet,bridge)
    try:return model(**chat_ids(q,system),use_cache=False,return_dict=True).logits[0,-1].float().detach().clone()
    finally:
        for h in hs:h.remove();ACTIVE.discard(id(h))
@torch.inference_mode()
def generate(q,system,packet,bridge):
    x=chat_ids(q,system);hs=install(packet,bridge)
    try:
        y=model.generate(**x,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)
        return tok.decode(y[0,x.input_ids.shape[1]:],skip_special_tokens=True).strip()
    finally:
        for h in hs:h.remove();ACTIVE.discard(id(h))
print("[7/15] BLIND STANDARD vs FORCED READOUT")
BLIND=[]
for i,(_,q) in enumerate(INPUTS):
    ls=forward(q,SYS_STD,PACK[i],ORTH[i][27]);lf=forward(q,SYS_FORCE,PACK[i],ORTH[i][27])
    gs=generate(q,SYS_STD,PACK[i],ORTH[i][27]);gf=generate(q,SYS_FORCE,PACK[i],ORTH[i][27])
    BLIND.append({"std_logits":ls,"force_logits":lf,"std_text":gs,"force_text":gf})
    print(f"{i+1:02d}/24 | STD={gs!r} | FORCE={gf!r}")
print("[8/15] ABSTENTION BEHAVIOR X-RAY")
ABSTAIN=("does not provide","not provide","no information","insufficient information","not enough information","cannot determine","can't determine","unable to determine","need more context","no specific")
def abstains(x):
    z=x.lower();return any(a in z for a in ABSTAIN)
std_abs=np.mean([abstains(x["std_text"]) for x in BLIND]);force_abs=np.mean([abstains(x["force_text"]) for x in BLIND])
print(f"STANDARD_ABSTENTION={std_abs:.6f} | FORCED_ABSTENTION={force_abs:.6f} | DROP={std_abs-force_abs:+.6f}")
print("[9/15] TARGET SEAL OPEN - POST-HOC ONLY")
def normtext(x):return re.sub(r"[^a-z0-9]+"," ",x.lower()).strip()
def exact(a,b):return normtext(a)==normtext(b)
def mention(a,b):
    a,b=normtext(a),normtext(b);return bool(b) and bool(re.search(r"(^| )"+re.escape(b)+r"($| )",a))
ROWS=[]
for i,target in enumerate(SEALED_TARGETS):
    tid=tok.encode(target,add_special_tokens=False)[0];b=BLIND[i]
    def assay(x):
        tv=float(x[tid]);m=x.clone();m[tid]=-float("inf");mx=float(m.max());return {"target_logit":tv,"margin":tv-mx,"rank":int((x>x[tid]).sum())+1,"top1":bool(tv>mx),"top_token":tok.decode([int(torch.argmax(x))])}
    s,f=assay(b["std_logits"]),assay(b["force_logits"])
    r={"statement":INPUTS[i][0],"question":INPUTS[i][1],"target":target,"target_ids":tok.encode(target,add_special_tokens=False),"slots":META[i]["slots"],
       "standard_text":b["std_text"],"forced_text":b["force_text"],"standard_abstain":abstains(b["std_text"]),"forced_abstain":abstains(b["force_text"]),
       "standard_exact":exact(b["std_text"],target),"forced_exact":exact(b["force_text"],target),"standard_mention":mention(b["std_text"],target),"forced_mention":mention(b["force_text"],target),
       "standard":s,"forced":f,"forced_target_gain":f["target_logit"]-s["target_logit"],"forced_margin_gain":f["margin"]-s["margin"]}
    ROWS.append(r);print(f"{i+1:02d}/24 | {target:11s} | STD={b['std_text']!r} | FORCE={b['force_text']!r} | RANK {s['rank']}->{f['rank']}")
print("[10/15] GLOBAL SUMMARY")
mean=lambda f:float(np.mean([f(r) for r in ROWS]));med=lambda f:float(np.median([f(r) for r in ROWS]))
S={"standard_abstention":mean(lambda r:r["standard_abstain"]),"forced_abstention":mean(lambda r:r["forced_abstain"]),
"standard_exact":mean(lambda r:r["standard_exact"]),"forced_exact":mean(lambda r:r["forced_exact"]),"standard_mention":mean(lambda r:r["standard_mention"]),"forced_mention":mean(lambda r:r["forced_mention"]),
"standard_rank_median":med(lambda r:r["standard"]["rank"]),"forced_rank_median":med(lambda r:r["forced"]["rank"]),"standard_top1":mean(lambda r:r["standard"]["top1"]),"forced_top1":mean(lambda r:r["forced"]["top1"]),
"forced_target_gain":mean(lambda r:r["forced_target_gain"]),"forced_margin_gain":mean(lambda r:r["forced_margin_gain"]),"forced_rank_improved":mean(lambda r:r["forced"]["rank"]<r["standard"]["rank"])}
for k,v in S.items():print(f"{k.upper():34s} {v:+.6f}")
print("[11/15] PRIOR-OVERRIDE EFFECT")
print(f"ABSTENTION {S['standard_abstention']:.3f}->{S['forced_abstention']:.3f}")
print(f"EXACT {S['standard_exact']:.3f}->{S['forced_exact']:.3f} | MENTION {S['standard_mention']:.3f}->{S['forced_mention']:.3f}")
print(f"MEDIAN RANK {S['standard_rank_median']:.1f}->{S['forced_rank_median']:.1f} | TOP1 {S['standard_top1']:.3f}->{S['forced_top1']:.3f}")
print("[12/15] PREDECLARED GATE")
G1=S["forced_abstention"]<S["standard_abstention"];G2=S["forced_mention"]>S["standard_mention"];G3=S["forced_rank_median"]<S["standard_rank_median"];G4=S["forced_target_gain"]>0
BEHAVIOR=bool(G1 and G2);READOUT=bool(G3 and G4);GATE=bool(BEHAVIOR and READOUT)
print(f"ABSTENTION DROP={G1} | RETRIEVAL IMPROVED={G2} | RANK IMPROVED={G3} | TARGET GAIN>0={G4}")
print("[13/15] INTERPRETATION CLASS")
CLASS="ABSTENTION_BARRIER_SUPPORTED" if GATE else ("ABSTENTION_REMOVED_BUT_RETRIEVAL_MISSING" if G1 and not G2 else "ABSTENTION_NOT_PRIMARY_BARRIER")
print("RESULT CLASS:",CLASS)
print("[14/15] TARGET + CASE AUDIT")
print("CASE_ISOLATED=True | CROSS_CASE_DATA=False | TARGET_ENGINE_ACCESS=False | TARGET_POSTHOC_ONLY=True")
print("[15/15] INTEGRITY + SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("Hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["max_seasc_dev"]>1e-6 or AUD["max_bridge_dev"]>1e-6:raise RuntimeError("Dose audit failure")
R={"schema":"akbascore.test313.v1","test":"TEST 313","start":START,"end":utc(),"model":MODEL_ID,
"question":"Is the TEST312 retrieval failure primarily caused by the model's abstention/missing-context behavioral prior?",
"engine":"TEST312 locked packet + SEASC L0-L25 + query-orthogonal L26 bridge","conditions":{"standard_system":SYS_STD,"forced_system":SYS_FORCE},
"case_isolation":{"enabled":True,"cross_case_data":False,"target_access":False},"seasc":{"layers":[0,25],"rss":RSS},"bridge":{"layer":26,"relative_dose":BRIDGE_REL},
"summary":S,"rows":ROWS,"gate":{"abstention_drop":G1,"retrieval_improved":G2,"rank_improved":G3,"target_gain_positive":G4,"result":"PASS" if GATE else "FAIL","classification":CLASS},
"integrity":{"target_posthoc_only":True,"training":False,"optimization":False,"dose_sweep":False,"target_head":False,"role_bank":False,"cross_case_data":False,
"weight_update":False,"seasc_calls":AUD["seasc"],"bridge_calls":AUD["bridge"],"max_seasc_dev":AUD["max_seasc_dev"],"max_bridge_dev":AUD["max_bridge_dev"],
"hooks_remaining":len(ACTIVE),"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
sha=hashlib.sha256(canon(R)).hexdigest();rid=f"T313-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json";tp=ROOT/f"{rid}.txt"
jp.write_bytes(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False).encode())
txt=["="*108,"TEST 313 - ABSTENTION-PRIOR OVERRIDE ASSAY","="*108,
f"ABSTENTION STD={S['standard_abstention']:.3f} FORCE={S['forced_abstention']:.3f}",
f"EXACT STD={S['standard_exact']:.3f} FORCE={S['forced_exact']:.3f} | MENTION STD={S['standard_mention']:.3f} FORCE={S['forced_mention']:.3f}",
f"MEDIAN RANK STD={S['standard_rank_median']:.1f} FORCE={S['forced_rank_median']:.1f} | TARGET GAIN={S['forced_target_gain']:+.4f}",
f"CLASS={CLASS} | GATE={'PASS' if GATE else 'FAIL'}",f"CASE_ISOLATED=TRUE | TARGET POST-HOC ONLY | RSS={RSS:.9f} | BRIDGE={BRIDGE_REL:.3f}",
f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]
tp.write_text("\n".join(txt),encoding="utf-8")
print("="*108);print("TEST 313 COMPLETE")
print(f"ABSTENTION {S['standard_abstention']:.3f}->{S['forced_abstention']:.3f}")
print(f"EXACT {S['standard_exact']:.3f}->{S['forced_exact']:.3f} | MENTION {S['standard_mention']:.3f}->{S['forced_mention']:.3f}")
print(f"MEDIAN RANK {S['standard_rank_median']:.1f}->{S['forced_rank_median']:.1f} | TARGET GAIN={S['forced_target_gain']:+.3f}")
print(f"CLASS={CLASS} | GATE={'PASS' if GATE else 'FAIL'}")
print("CASE_ISOLATED=TRUE | TARGET POST-HOC ONLY | WEIGHT INTEGRITY PASS | AKBASCORE HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
