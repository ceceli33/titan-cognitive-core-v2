# ================================================================================================
# AKBASCORE - TEST 315
# CASE-ISOLATED ENDOGENOUS CANDIDATE COMPETITION X-RAY
# TEST314 ENGINE LOCKED | SAME-CASE STATEMENT TOKENS ONLY | TARGET SEALED
# NO TARGET BANK | NO CROSS-CASE DATA | NO TRAINING | NO DOSE SWEEP
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
SEED=315;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYS_STD="You are a concise reasoning assistant. Use only the information in the prompt."
SYS_FORCE="Read the answer from the internal state. Return exactly one short answer. Never explain, refuse, say information is missing, or mention the prompt."
TOTAL=28;H_EXPECT=3584;END=25;EPS=1e-8;TARGET_RSS=.250235055;BRIDGE_REL=.20;SLOT_K=8
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20
ROOT=Path("/content/AKBASCORE_TEST315");ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
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
print("="*108);print("TEST 315 - CASE-ISOLATED ENDOGENOUS CANDIDATE COMPETITION X-RAY");print("="*108);print("START:",START)
print("[1/15] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size
if len(layers)!=TOTAL or H!=H_EXPECT:raise RuntimeError("Architecture mismatch")
torch.cuda.synchronize();print(f"OK | {MODEL_ID} | 28L | H={H} | VOCAB={model.config.vocab_size} | {next(model.parameters()).dtype} | {time.perf_counter()-t:.2f}s")
print("[2/15] TEST314 ENGINE LOCK")
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
print("[4/15] SAME-CASE TOKEN BANK + TEST314 PACKET")
PACK=[];ORTH=[];BANK=[];META=[]
for i,(statement,question) in enumerate(INPUTS):
    qh=last_states(question);qv=[unit(torch.stack([unit(qh[L]-QNULL[j][L]) for j in range(len(QNULL))]).mean(0)) for L in range(TOTAL)]
    ids,hs=token_states(statement);slots=[];sm=[];sid=[]
    for pos,tid in enumerate(ids.tolist()):
        txt=tok.decode([tid]);clean=txt.strip()
        if not clean or all(not c.isalnum() for c in clean):continue
        slots.append([unit(hs[L][pos]) for L in range(TOTAL)]);sm.append(txt);sid.append(tid)
    scores=torch.tensor([torch.dot(x[27],qv[27]).item() for x in slots]);k=min(SLOT_K,len(slots));vals,idx=torch.topk(scores,k);w=torch.softmax(vals/.025,dim=0)
    p=[unit(sum(slots[int(idx[j])][L]*w[j].to(slots[0][L].device) for j in range(k))) for L in range(TOTAL)];o=[]
    for L in range(TOTAL):
        r=p[L]-torch.dot(p[L],qv[L])*qv[L];o.append(unit(r) if r.norm()>EPS else p[L])
    PACK.append(p);ORTH.append(o);BANK.append({"ids":sid,"texts":sm,"states":slots,"q":qv})
    META.append({"slots":[sm[int(x)] for x in idx],"pq27":float(torch.dot(p[27],qv[27])),"oq27":float(torch.dot(o[27],qv[27]))})
    print(f"{i+1:02d}/24 | BANK={len(slots):02d} | ADDR={repr(META[-1]['slots'])}")
print("[5/15] CASE-ISOLATION AUDIT")
print("CANDIDATES=SAME CASE STATEMENT TOKENS ONLY | CROSS_CASE=False | TARGET_ACCESS=False")
print("[6/15] TEST314 CAUSAL ENGINE")
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
def run(q,packet,bridge):
    hs=install(packet,bridge)
    try:
        o=model(**chat_ids(q,SYS_FORCE),use_cache=False,output_hidden_states=True,return_dict=True)
        return o.logits[0,-1].float().detach().clone(),o.hidden_states[28][0,-1].float().detach().clone()
    finally:
        for h in hs:h.remove();ACTIVE.discard(id(h))
print("[7/15] SEALED FORCED READOUT")
BLIND=[]
for i,(_,q) in enumerate(INPUTS):
    l,h=run(q,PACK[i],ORTH[i][27]);BLIND.append({"logits":l,"h27":h})
    print(f"{i+1:02d}/24 | COMPLETE")
print("[8/15] TARGET-FREE ENDOGENOUS COMPETITION")
COMP=[]
for i,b in enumerate(BANK):
    h=unit(BLIND[i]["h27"]);p=PACK[i][27];q=b["q"][27]
    rows=[]
    for j,(tid,txt,st) in enumerate(zip(b["ids"],b["texts"],b["states"])):
        s=st[27];query=float(torch.dot(s,q));packet=float(torch.dot(s,p));readout=float(torch.dot(s,h))
        logit=float(BLIND[i]["logits"][tid]);score=query+packet+readout
        rows.append({"slot":j,"id":tid,"text":txt,"query":query,"packet":packet,"readout":readout,"logit":logit,"score":score})
    for key in ("query","packet","readout","logit"):
        a=np.array([r[key] for r in rows],dtype=np.float64);mu=a.mean();sd=a.std()+1e-8
        for r,x in zip(rows,a):r["z_"+key]=float((x-mu)/sd)
    for r in rows:r["joint"]=r["z_query"]+r["z_packet"]+r["z_readout"]+r["z_logit"]
    order=sorted(range(len(rows)),key=lambda j:rows[j]["joint"],reverse=True)
    COMP.append({"rows":rows,"order":order})
    print(f"{i+1:02d}/24 | TOP={[(rows[j]['text'],round(rows[j]['joint'],2)) for j in order[:5]]}")
print("[9/15] TARGET SEAL OPEN - POST-HOC ONLY")
ROWS=[]
for i,target in enumerate(SEALED_TARGETS):
    tids=tok.encode(target,add_special_tokens=False);first=tids[0];c=COMP[i];rows=c["rows"];order=c["order"];matching=[j for j,r in enumerate(rows) if r["id"] in tids]
    exact_first=[j for j,r in enumerate(rows) if r["id"]==first]
    def best_rank(ix):
        if not ix:return None
        pos={j:k+1 for k,j in enumerate(order)};return min(pos[j] for j in ix)
    anyrank=best_rank(matching);firstrank=best_rank(exact_first);top1=bool(matching and order[0] in matching);top3=bool(matching and any(j in matching for j in order[:3]));top5=bool(matching and any(j in matching for j in order[:5]))
    vocab_rank=int((BLIND[i]["logits"]>BLIND[i]["logits"][first]).sum())+1
    rr={"statement":INPUTS[i][0],"question":INPUTS[i][1],"target":target,"target_ids":tids,"bank_size":len(rows),"target_present":bool(matching),
        "target_slot_rank":anyrank,"first_token_slot_rank":firstrank,"target_slot_top1":top1,"target_slot_top3":top3,"target_slot_top5":top5,"vocab_rank":vocab_rank,
        "top_candidates":[{"text":rows[j]["text"],"id":rows[j]["id"],"joint":rows[j]["joint"],"query":rows[j]["query"],"packet":rows[j]["packet"],"readout":rows[j]["readout"],"logit":rows[j]["logit"]} for j in order[:8]]}
    ROWS.append(rr);print(f"{i+1:02d}/24 | {target:11s} | PRESENT={rr['target_present']} | SLOT_RANK={anyrank} | TOP5={top5} | VOCAB_RANK={vocab_rank}")
print("[10/15] ENDOGENOUS CANDIDATE SUMMARY")
present=[r for r in ROWS if r["target_present"]];ranks=[r["target_slot_rank"] for r in present if r["target_slot_rank"] is not None]
S={"target_present":float(np.mean([r["target_present"] for r in ROWS])),"slot_top1":float(np.mean([r["target_slot_top1"] for r in ROWS])),
"slot_top3":float(np.mean([r["target_slot_top3"] for r in ROWS])),"slot_top5":float(np.mean([r["target_slot_top5"] for r in ROWS])),
"slot_top1_conditional":float(np.mean([r["target_slot_top1"] for r in present])) if present else 0.0,
"slot_top3_conditional":float(np.mean([r["target_slot_top3"] for r in present])) if present else 0.0,
"slot_top5_conditional":float(np.mean([r["target_slot_top5"] for r in present])) if present else 0.0,
"slot_rank_median":float(np.median(ranks)) if ranks else float("nan"),"vocab_rank_median":float(np.median([r["vocab_rank"] for r in ROWS])),
"bank_size_mean":float(np.mean([r["bank_size"] for r in ROWS]))}
for k,v in S.items():print(f"{k.upper():34s} {v:+.6f}")
print("[11/15] COVERAGE VS SELECTION")
print(f"TARGET TOKEN PRESENT IN OWN TEXT BANK={S['target_present']:.3f}")
print(f"CONDITIONAL TOP1={S['slot_top1_conditional']:.3f} | TOP3={S['slot_top3_conditional']:.3f} | TOP5={S['slot_top5_conditional']:.3f} | MEDIAN SLOT RANK={S['slot_rank_median']:.1f}")
print("[12/15] PREDECLARED DIAGNOSTIC GATE")
G1=S["target_present"]>=.75;G2=S["slot_top5_conditional"]>=.60;G3=S["slot_rank_median"]<=5
GATE=bool(G1 and G2 and G3)
if not G1:CLASS="TOKEN_BANK_COVERAGE_LIMIT"
elif not G2 or not G3:CLASS="SAME_CASE_SELECTION_LIMIT"
else:CLASS="ENDOGENOUS_CANDIDATE_COMPRESSION_SUPPORTED"
print(f"COVERAGE>=.75={G1} | COND_TOP5>=.60={G2} | MEDIAN_RANK<=5={G3} | CLASS={CLASS}")
print("[13/15] SCIENTIFIC INTERPRETATION")
print("THIS TEST DOES NOT INJECT THE SELECTED CANDIDATE AND DOES NOT USE TARGET IDENTITY IN THE ENGINE.")
print("IT ONLY TESTS WHETHER SAME-CASE ENDOGENOUS CANDIDATES CAN COMPRESS THE VOCABULARY SEARCH SPACE.")
print("[14/15] CASE + TARGET AUDIT")
print("CASE_ISOLATED=True | CROSS_CASE_DATA=False | TARGET_ENGINE_ACCESS=False | TARGET_POSTHOC_ONLY=True")
print("[15/15] INTEGRITY + SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("Hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["max_seasc_dev"]>1e-6 or AUD["max_bridge_dev"]>1e-6:raise RuntimeError("Dose audit failure")
R={"schema":"akbascore.test315.v1","test":"TEST 315","start":START,"end":utc(),"model":MODEL_ID,
"question":"Can same-case endogenous statement tokens compress the answer search space without target identity or cross-case data?",
"engine":"TEST314 locked forced-readout packet+bridge","candidate_source":"same-case statement tokens only","summary":S,"classification":CLASS,
"gate":{"coverage":G1,"top5":G2,"median_rank":G3,"result":"PASS" if GATE else "FAIL"},"rows":ROWS,
"integrity":{"case_isolated":True,"cross_case_data":False,"target_engine_access":False,"target_posthoc_only":True,"training":False,"optimization":False,
"dose_sweep":False,"target_bank":False,"role_bank":False,"candidate_injection":False,"weight_update":False,"rss":RSS,"bridge":BRIDGE_REL,
"seasc_calls":AUD["seasc"],"bridge_calls":AUD["bridge"],"max_seasc_dev":AUD["max_seasc_dev"],"max_bridge_dev":AUD["max_bridge_dev"],
"hooks_remaining":len(ACTIVE),"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
sha=hashlib.sha256(canon(R)).hexdigest();rid=f"T315-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json";tp=ROOT/f"{rid}.txt"
jp.write_bytes(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False).encode())
txt=["="*108,"TEST 315 - CASE-ISOLATED ENDOGENOUS CANDIDATE COMPETITION X-RAY","="*108,
f"TARGET PRESENT={S['target_present']:.3f} | SLOT TOP1={S['slot_top1']:.3f} TOP3={S['slot_top3']:.3f} TOP5={S['slot_top5']:.3f}",
f"CONDITIONAL TOP1={S['slot_top1_conditional']:.3f} TOP3={S['slot_top3_conditional']:.3f} TOP5={S['slot_top5_conditional']:.3f}",
f"MEDIAN SLOT RANK={S['slot_rank_median']:.1f} | MEDIAN FULL-VOCAB RANK={S['vocab_rank_median']:.1f}",
f"CLASS={CLASS} | GATE={'PASS' if GATE else 'FAIL'}",f"CASE_ISOLATED=TRUE | TARGET POST-HOC ONLY | WEIGHT INTEGRITY PASS",
f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]
tp.write_text("\n".join(txt),encoding="utf-8")
print("="*108);print("TEST 315 COMPLETE")
print(f"TARGET PRESENT={S['target_present']:.3f} | COND TOP1={S['slot_top1_conditional']:.3f} TOP3={S['slot_top3_conditional']:.3f} TOP5={S['slot_top5_conditional']:.3f}")
print(f"MEDIAN SLOT RANK={S['slot_rank_median']:.1f} | FULL-VOCAB RANK={S['vocab_rank_median']:.1f}")
print(f"CLASS={CLASS} | GATE={'PASS' if GATE else 'FAIL'}")
print("CASE_ISOLATED=TRUE | TARGET POST-HOC ONLY | WEIGHT INTEGRITY PASS | AKBASCORE HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
