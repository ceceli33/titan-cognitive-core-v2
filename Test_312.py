# ================================================================================================
# AKBASCORE - TEST 312
# END-TO-END ANSWER-BLIND RETRIEVAL / DEMO QUALIFICATION
# ONE TEXT + ONE QUESTION | CASE-ISOLATED | TEST311 OWN QUERY-ORTHOGONAL PACKET
# TARGET SEALED UNTIL ALL FORWARD + GENERATION RUNS COMPLETE
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
SEED=312;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
TOTAL=28;H_EXPECT=3584;END=25;EPS=1e-8;TARGET_RSS=.250235055;BRIDGE_REL=.20;SLOT_K=8;MAX_NEW=12
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20
ROOT=Path("/content/AKBASCORE_TEST312");ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
# Targets are sealed in a separate structure and are never read until all blind runs finish.
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
("The autonomous cart waited briefly, reversed two meters, and then started accelerating toward Gate C.","What did the cart start doing toward Gate C?","")]
# remove harmless placeholder if present
INPUTS=[(x[0],x[1]) for x in INPUTS]
SEALED_TARGETS=("Selin","Zorvex","417","Reykjavik","basalt","violet","climbing","glossy","sextant","Ordelis","Keira","Tallinn","glass","amber","731","rotating","silky","astrolabe","Varethis","Ilyan","Kyoto","turquoise","quartz","accelerating")
QNULLS=["What information is requested?","Which unknown detail should be recovered?","Answer the question using the missing information.","What should be identified?","Which detail is being asked about?"]
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
def prompt_ids(q):
    s=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":q}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to("cuda")
def raw_ids(s):return tok(s,return_tensors="pt",add_special_tokens=False).to("cuda")
print("="*108);print("TEST 312 - END-TO-END ANSWER-BLIND RETRIEVAL / DEMO QUALIFICATION");print("="*108);print("START:",START)
print("[1/14] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size
if len(layers)!=TOTAL or H!=H_EXPECT:raise RuntimeError("Architecture mismatch")
torch.cuda.synchronize();print(f"OK | {MODEL_ID} | 28L | H={H} | VOCAB={model.config.vocab_size} | {next(model.parameters()).dtype} | {time.perf_counter()-t:.2f}s")
print("[2/14] TEST311 LOCK + INTEGRITY")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();base=[IVME*(ZIRVE*np.exp(-SONUM*L)*(1+SONUM*L)+TABAN)/(ZIRVE+TABAN) for L in range(END+1)];scale=TARGET_RSS/np.sqrt(sum(x*x for x in base));RHO=[x*scale for x in base];RSS=float(np.sqrt(sum(x*x for x in RHO)))
print("FP:",[f"{x:.4f}" for x in FP0]);print(f"RSS={RSS:.9f} | BRIDGE={BRIDGE_REL:.3f} | SLOT_K={SLOT_K} | CASE_ISOLATED=TRUE")
@torch.inference_mode()
def last_states(text):
    o=model(**prompt_ids(text),use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)]
@torch.inference_mode()
def token_states(text):
    x=raw_ids(text);o=model(**x,use_cache=False,output_hidden_states=True,return_dict=True)
    return x.input_ids[0].detach().cpu(),[o.hidden_states[L+1][0].float().detach().clone() for L in range(TOTAL)]
print("[3/14] TARGET-FREE QUERY NULL CACHE")
QNULL=[last_states(x) for x in QNULLS];print("NULL READY")
print("[4/14] TEST311 OWN PACKET COMPILER")
PACK=[];ORTH=[];META=[]
for i,(statement,question) in enumerate(INPUTS):
    qh=last_states(question);qv=[unit(torch.stack([unit(qh[L]-QNULL[j][L]) for j in range(len(QNULL))]).mean(0)) for L in range(TOTAL)]
    ids,hs=token_states(statement);slots=[];sm=[]
    for pos,tid in enumerate(ids.tolist()):
        txt=tok.decode([tid]);clean=txt.strip()
        if not clean or all(not c.isalnum() for c in clean):continue
        slots.append([unit(hs[L][pos]) for L in range(TOTAL)]);sm.append({"pos":pos,"tid":tid,"token":txt})
    scores=torch.tensor([torch.dot(x[27],qv[27]).item() for x in slots]);k=min(SLOT_K,len(slots));vals,idx=torch.topk(scores,k);w=torch.softmax(vals/.025,dim=0)
    p=[unit(sum(slots[int(idx[j])][L]*w[j].to(slots[0][L].device) for j in range(k))) for L in range(TOTAL)];o=[]
    for L in range(TOTAL):
        r=p[L]-torch.dot(p[L],qv[L])*qv[L];o.append(unit(r) if r.norm()>EPS else p[L])
    PACK.append(p);ORTH.append(o);META.append({"slots":[sm[int(x)]["token"] for x in idx],"pq27":float(torch.dot(p[27],qv[27])),"oq27":float(torch.dot(o[27],qv[27]))})
    print(f"{i+1:02d}/24 | {repr(META[-1]['slots'])} | P.Q={META[-1]['pq27']:+.4f} -> O.Q={META[-1]['oq27']:+.7f}")
print("[5/14] CASE-ISOLATION AUDIT")
print("ENGINE INPUT PER RUN = ONE TEXT + ONE QUESTION | CROSS_CASE_DATA=False | TARGET_ACCESS=False")
print("[6/14] CAUSAL ENGINE")
AUD={"seasc":0,"bridge":0,"max_seasc_dev":0.0,"max_bridge_dev":0.0};ACTIVE=set()
def install(packet,bridge=None):
    hs=[]
    for L in range(END+1):
        rho=RHO[L]
        def hk(mod,args,out,L=L,rho=rho):
            y=out[0] if isinstance(out,tuple) else out;z=y[:,-1,:].float();d=packet[L].to(z.device)*z.norm(dim=-1,keepdim=True)*rho
            rel=float((d.norm(dim=-1)/(z.norm(dim=-1)+EPS)).max());AUD["seasc"]+=1;AUD["max_seasc_dev"]=max(AUD["max_seasc_dev"],abs(rel-rho))
            r=y.clone();r[:,-1,:]=(z+d).to(y.dtype);return (r,)+out[1:] if isinstance(out,tuple) else r
        h=layers[L].register_forward_hook(hk);hs.append(h);ACTIVE.add(id(h))
    if bridge is not None:
        def bh(mod,args,out):
            y=out[0] if isinstance(out,tuple) else out;z=y[:,-1,:].float();d=bridge.to(z.device)*z.norm(dim=-1,keepdim=True)*BRIDGE_REL
            rel=float((d.norm(dim=-1)/(z.norm(dim=-1)+EPS)).max());AUD["bridge"]+=1;AUD["max_bridge_dev"]=max(AUD["max_bridge_dev"],abs(rel-BRIDGE_REL))
            r=y.clone();r[:,-1,:]=(z+d).to(y.dtype);return (r,)+out[1:] if isinstance(out,tuple) else r
        h=layers[26].register_forward_hook(bh);hs.append(h);ACTIVE.add(id(h))
    return hs
@torch.inference_mode()
def logits(q,packet=None,bridge=None):
    hs=[] if packet is None else install(packet,bridge)
    try:return model(**prompt_ids(q),use_cache=False,return_dict=True).logits[0,-1].float().detach().clone()
    finally:
        for h in hs:h.remove();ACTIVE.discard(id(h))
@torch.inference_mode()
def generate(q,packet=None,bridge=None):
    x=prompt_ids(q);hs=[] if packet is None else install(packet,bridge)
    try:
        y=model.generate(**x,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)
        return tok.decode(y[0,x.input_ids.shape[1]:],skip_special_tokens=True).strip()
    finally:
        for h in hs:h.remove();ACTIVE.discard(id(h))
print("[7/14] SEALED BLIND FORWARD + GENERATION")
BLIND=[]
for i,(_,q) in enumerate(INPUTS):
    lv=logits(q);lp=logits(q,PACK[i]);la=logits(q,PACK[i],ORTH[i][27])
    gv=generate(q);gp=generate(q,PACK[i]);ga=generate(q,PACK[i],ORTH[i][27])
    BLIND.append({"lv":lv,"lp":lp,"la":la,"vanilla_text":gv,"packet_text":gp,"akbascore_text":ga})
    print(f"{i+1:02d}/24 | VANILLA={gv!r} | PACKET={gp!r} | AKBASCORE={ga!r}")
print("[8/14] TARGET SEAL OPEN - POST-HOC ONLY")
TARGETS=SEALED_TARGETS
if len(TARGETS)!=len(BLIND):raise RuntimeError("Target count mismatch")
def normtext(x):return re.sub(r"[^a-z0-9]+"," ",x.lower()).strip()
def exact(out,target):return normtext(out)==normtext(target)
def mention(out,target):
    a,b=normtext(out),normtext(target)
    return bool(b) and bool(re.search(r"(^| )"+re.escape(b)+r"($| )",a))
ROWS=[]
for i,target in enumerate(TARGETS):
    tid=tok.encode(target,add_special_tokens=False)[0];b=BLIND[i]
    def assay(x):
        tv=float(x[tid]);m=x.clone();m[tid]=-float("inf");mx=float(m.max());return {"target_logit":tv,"margin":tv-mx,"rank":int((x>x[tid]).sum())+1,"top1":bool(tv>mx)}
    av,ap,aa=assay(b["lv"]),assay(b["lp"]),assay(b["la"])
    r={"statement":INPUTS[i][0],"question":INPUTS[i][1],"target":target,"target_ids":tok.encode(target,add_special_tokens=False),"slots":META[i]["slots"],
       "vanilla_text":b["vanilla_text"],"packet_text":b["packet_text"],"akbascore_text":b["akbascore_text"],
       "vanilla_exact":exact(b["vanilla_text"],target),"packet_exact":exact(b["packet_text"],target),"akbascore_exact":exact(b["akbascore_text"],target),
       "vanilla_mention":mention(b["vanilla_text"],target),"packet_mention":mention(b["packet_text"],target),"akbascore_mention":mention(b["akbascore_text"],target),
       "vanilla":av,"packet":ap,"akbascore":aa,"packet_gain":ap["target_logit"]-av["target_logit"],"bridge_gain":aa["target_logit"]-ap["target_logit"],"final_gain":aa["target_logit"]-av["target_logit"]}
    ROWS.append(r);print(f"{i+1:02d}/24 | {target:11s} | V={r['vanilla_exact']}/{r['vanilla_mention']} P={r['packet_exact']}/{r['packet_mention']} A={r['akbascore_exact']}/{r['akbascore_mention']} | RANK {av['rank']}->{ap['rank']}->{aa['rank']}")
print("[9/14] END-TO-END SUMMARY")
mean=lambda f:float(np.mean([f(r) for r in ROWS]));med=lambda f:float(np.median([f(r) for r in ROWS]))
S={"vanilla_exact":mean(lambda r:r["vanilla_exact"]),"packet_exact":mean(lambda r:r["packet_exact"]),"akbascore_exact":mean(lambda r:r["akbascore_exact"]),
"vanilla_mention":mean(lambda r:r["vanilla_mention"]),"packet_mention":mean(lambda r:r["packet_mention"]),"akbascore_mention":mean(lambda r:r["akbascore_mention"]),
"vanilla_rank_median":med(lambda r:r["vanilla"]["rank"]),"packet_rank_median":med(lambda r:r["packet"]["rank"]),"akbascore_rank_median":med(lambda r:r["akbascore"]["rank"]),
"packet_gain":mean(lambda r:r["packet_gain"]),"bridge_gain":mean(lambda r:r["bridge_gain"]),"final_gain":mean(lambda r:r["final_gain"]),
"akbascore_top1":mean(lambda r:r["akbascore"]["top1"]),"rank_improved_vs_packet":mean(lambda r:r["akbascore"]["rank"]<r["packet"]["rank"])}
for k,v in S.items():print(f"{k.upper():34s} {v:+.6f}")
print("[10/14] GENERATION DELTA")
print(f"EXACT V={S['vanilla_exact']:.3f} P={S['packet_exact']:.3f} A={S['akbascore_exact']:.3f}")
print(f"MENTION V={S['vanilla_mention']:.3f} P={S['packet_mention']:.3f} A={S['akbascore_mention']:.3f}")
print(f"MEDIAN RANK V={S['vanilla_rank_median']:.1f} P={S['packet_rank_median']:.1f} A={S['akbascore_rank_median']:.1f}")
print("[11/14] DEMO QUALIFICATION GATE")
G1=S["akbascore_exact"]>S["vanilla_exact"];G2=S["akbascore_mention"]>S["vanilla_mention"];G3=S["akbascore_rank_median"]<S["packet_rank_median"];G4=S["final_gain"]>0
GATE=bool(G1 and G2 and G3 and G4)
print(f"EXACT IMPROVED={G1} | MENTION IMPROVED={G2} | RANK IMPROVED={G3} | FINAL GAIN>0={G4}")
print("[12/14] CASE-INDEPENDENCE")
print("CASE_ISOLATED=True | 24=EVALUATION COUNT ONLY | ONE TEXT + ONE QUESTION PER RUN")
print("[13/14] TARGET-ACCESS AUDIT")
print("COMPILER=False | ADDRESSING=False | SEASC=False | BRIDGE=False | GENERATION=False | TARGET OPENED POST-HOC=True")
print("[14/14] INTEGRITY + SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("Hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["max_seasc_dev"]>1e-6 or AUD["max_bridge_dev"]>1e-6:raise RuntimeError("Dose audit failure")
serial_rows=[]
for r in ROWS:serial_rows.append(r)
R={"schema":"akbascore.test312.v1","test":"TEST 312","start":START,"end":utc(),"model":MODEL_ID,
"question":"Can the case-isolated TEST311 engine produce the hidden answer end-to-end from one text and one question without target access?",
"case_isolation":{"enabled":True,"cases":24,"cases_are_evaluation_repetitions_only":True,"engine_input":"one text + one question","cross_case_data":False},
"compiler":"TEST311 query-addressed token-slot packet + query-orthogonal L27 bridge","seasc":{"layers":[0,25],"rss":RSS},"bridge":{"layer":26,"relative_dose":BRIDGE_REL},
"generation":{"greedy":True,"max_new_tokens":MAX_NEW,"conditions":["vanilla","packet","akbascore"]},"summary":S,"rows":serial_rows,
"gate":{"exact_improved":G1,"mention_improved":G2,"rank_improved":G3,"positive_final_gain":G4,"result":"PASS" if GATE else "FAIL"},
"integrity":{"target_sealed_until_posthoc":True,"target_used_by_engine":False,"training":False,"optimization":False,"dose_sweep":False,"weight_update":False,
"cross_case_data":False,"seasc_calls":AUD["seasc"],"bridge_calls":AUD["bridge"],"max_seasc_dev":AUD["max_seasc_dev"],"max_bridge_dev":AUD["max_bridge_dev"],
"hooks_remaining":len(ACTIVE),"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
sha=hashlib.sha256(canon(R)).hexdigest();rid=f"T312-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json";tp=ROOT/f"{rid}.txt"
jp.write_bytes(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False).encode())
txt=["="*108,"TEST 312 - END-TO-END ANSWER-BLIND RETRIEVAL / DEMO QUALIFICATION","="*108,
f"EXACT V={S['vanilla_exact']:.3f} P={S['packet_exact']:.3f} A={S['akbascore_exact']:.3f}",
f"MENTION V={S['vanilla_mention']:.3f} P={S['packet_mention']:.3f} A={S['akbascore_mention']:.3f}",
f"MEDIAN RANK V={S['vanilla_rank_median']:.1f} P={S['packet_rank_median']:.1f} A={S['akbascore_rank_median']:.1f}",
f"GAIN PACKET={S['packet_gain']:+.4f} BRIDGE={S['bridge_gain']:+.4f} FINAL={S['final_gain']:+.4f}",
f"DEMO QUALIFICATION={'PASS' if GATE else 'FAIL'} | CASE_ISOLATED=TRUE | TARGET POST-HOC ONLY",
f"RSS={RSS:.9f} | BRIDGE={BRIDGE_REL:.3f} | WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]
tp.write_text("\n".join(txt),encoding="utf-8")
print("="*108);print("TEST 312 COMPLETE")
print(f"EXACT V={S['vanilla_exact']:.3f} P={S['packet_exact']:.3f} A={S['akbascore_exact']:.3f}")
print(f"MENTION V={S['vanilla_mention']:.3f} P={S['packet_mention']:.3f} A={S['akbascore_mention']:.3f}")
print(f"MEDIAN RANK V={S['vanilla_rank_median']:.1f} P={S['packet_rank_median']:.1f} A={S['akbascore_rank_median']:.1f}")
print(f"FINAL GAIN={S['final_gain']:+.3f} | DEMO QUALIFICATION={'PASS' if GATE else 'FAIL'}")
print("CASE_ISOLATED=TRUE | TARGET POST-HOC ONLY | WEIGHT INTEGRITY PASS | AKBASCORE HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
