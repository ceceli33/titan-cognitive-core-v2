# ================================================================================================
# AKBASCORE - TEST 314
# ANSWER-BLIND DECISION-BOUNDARY X-RAY
# TEST313 ENGINE LOCKED | VANILLA -> PACKET -> BRIDGE -> FORCED READOUT
# TARGET SEALED | CASE-ISOLATED | NO TARGET-HEAD | NO TRAINING | NO DOSE SWEEP
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
SEED=314;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYS_STD="You are a concise reasoning assistant. Use only the information in the prompt."
SYS_FORCE="Read the answer from the internal state. Return exactly one short answer. Never explain, refuse, say information is missing, or mention the prompt."
TOTAL=28;H_EXPECT=3584;END=25;EPS=1e-8;TARGET_RSS=.250235055;BRIDGE_REL=.20;SLOT_K=8
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20
ROOT=Path("/content/AKBASCORE_TEST314");ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
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
print("="*108);print("TEST 314 - ANSWER-BLIND DECISION-BOUNDARY X-RAY");print("="*108);print("START:",START)
print("[1/15] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size
if len(layers)!=TOTAL or H!=H_EXPECT:raise RuntimeError("Architecture mismatch")
torch.cuda.synchronize();print(f"OK | {MODEL_ID} | 28L | H={H} | VOCAB={model.config.vocab_size} | {next(model.parameters()).dtype} | {time.perf_counter()-t:.2f}s")
print("[2/15] TEST313 ENGINE LOCK")
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
print("[4/15] TEST313 PACKET COMPILER")
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
    print(f"{i+1:02d}/24 | {repr(META[-1]['slots'])}")
print("[5/15] CASE-ISOLATION AUDIT")
print("ONE STATEMENT + ONE QUESTION | CROSS_CASE_DATA=False | TARGET_ACCESS=False")
print("[6/15] CAUSAL ENGINE")
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
def run(q,system,packet=None,bridge=None):
    hs=[] if packet is None else install(packet,bridge)
    try:
        o=model(**chat_ids(q,system),use_cache=False,output_hidden_states=True,return_dict=True)
        return {"logits":o.logits[0,-1].float().detach().cpu(),"h27":o.hidden_states[28][0,-1].float().detach().cpu()}
    finally:
        for h in hs:h.remove();ACTIVE.discard(id(h))
print("[7/15] SEALED FOUR-STATE X-RAY")
BLIND=[]
for i,(_,q) in enumerate(INPUTS):
    a=run(q,SYS_STD);b=run(q,SYS_FORCE);c=run(q,SYS_FORCE,PACK[i]);d=run(q,SYS_FORCE,PACK[i],ORTH[i][27])
    BLIND.append({"std":a,"force":b,"packet":c,"bridge":d})
    print(f"{i+1:02d}/24 | STD -> FORCE -> PACKET -> BRIDGE COMPLETE")
print("[8/15] TARGET SEAL OPEN - POST-HOC ONLY")
ROWS=[]
W=model.lm_head.weight.detach().float().cpu()
for i,target in enumerate(SEALED_TARGETS):
    tid=tok.encode(target,add_special_tokens=False)[0];b=BLIND[i]
    def assay(x):
        l=x["logits"];tv=float(l[tid]);tmp=l.clone();tmp[tid]=-float("inf");cid=int(torch.argmax(tmp));cv=float(l[cid]);rank=int((l>l[tid]).sum())+1
        return {"target_logit":tv,"competitor_id":cid,"competitor":tok.decode([cid]),"competitor_logit":cv,"margin":tv-cv,"rank":rank,"top1":rank==1}
    A={k:assay(b[k]) for k in ("std","force","packet","bridge")}
    topid=A["bridge"]["competitor_id"];wh=unit(W[tid]);wc=unit(W[topid]);decision=unit(W[tid]-W[topid])
    h0=b["std"]["h27"];h1=b["force"]["h27"];h2=b["packet"]["h27"];h3=b["bridge"]["h27"]
    d_force=h1-h0;d_packet=h2-h1;d_bridge=h3-h2;d_total=h3-h0
    proj=lambda x,v:float(torch.dot(x,v))
    geom={"target_head_cos":float(torch.dot(wh,wc)),"force_decision_proj":proj(d_force,decision),"packet_decision_proj":proj(d_packet,decision),
          "bridge_decision_proj":proj(d_bridge,decision),"total_decision_proj":proj(d_total,decision),
          "force_target_proj":proj(d_force,wh),"force_comp_proj":proj(d_force,wc),
          "packet_target_proj":proj(d_packet,wh),"packet_comp_proj":proj(d_packet,wc),
          "bridge_target_proj":proj(d_bridge,wh),"bridge_comp_proj":proj(d_bridge,wc)}
    r={"statement":INPUTS[i][0],"question":INPUTS[i][1],"target":target,"target_id":tid,"slots":META[i]["slots"],"states":A,"geometry":geom,
       "force_margin_delta":A["force"]["margin"]-A["std"]["margin"],"packet_margin_delta":A["packet"]["margin"]-A["force"]["margin"],
       "bridge_margin_delta":A["bridge"]["margin"]-A["packet"]["margin"],"total_margin_delta":A["bridge"]["margin"]-A["std"]["margin"]}
    ROWS.append(r)
    print(f"{i+1:02d}/24 | {target:11s} | TOP={A['bridge']['competitor']!r:14s} | RANK {A['std']['rank']}->{A['force']['rank']}->{A['packet']['rank']}->{A['bridge']['rank']} | MARGIN {A['std']['margin']:+.2f}->{A['bridge']['margin']:+.2f}")
print("[9/15] GLOBAL DECISION BOUNDARY")
mean=lambda f:float(np.mean([f(r) for r in ROWS]));med=lambda f:float(np.median([f(r) for r in ROWS]))
S={"std_rank":med(lambda r:r["states"]["std"]["rank"]),"force_rank":med(lambda r:r["states"]["force"]["rank"]),"packet_rank":med(lambda r:r["states"]["packet"]["rank"]),"bridge_rank":med(lambda r:r["states"]["bridge"]["rank"]),
"std_margin":mean(lambda r:r["states"]["std"]["margin"]),"force_margin":mean(lambda r:r["states"]["force"]["margin"]),"packet_margin":mean(lambda r:r["states"]["packet"]["margin"]),"bridge_margin":mean(lambda r:r["states"]["bridge"]["margin"]),
"force_margin_delta":mean(lambda r:r["force_margin_delta"]),"packet_margin_delta":mean(lambda r:r["packet_margin_delta"]),"bridge_margin_delta":mean(lambda r:r["bridge_margin_delta"]),"total_margin_delta":mean(lambda r:r["total_margin_delta"]),
"force_decision_proj":mean(lambda r:r["geometry"]["force_decision_proj"]),"packet_decision_proj":mean(lambda r:r["geometry"]["packet_decision_proj"]),"bridge_decision_proj":mean(lambda r:r["geometry"]["bridge_decision_proj"]),
"force_target_proj":mean(lambda r:r["geometry"]["force_target_proj"]),"force_comp_proj":mean(lambda r:r["geometry"]["force_comp_proj"]),
"packet_target_proj":mean(lambda r:r["geometry"]["packet_target_proj"]),"packet_comp_proj":mean(lambda r:r["geometry"]["packet_comp_proj"]),
"bridge_target_proj":mean(lambda r:r["geometry"]["bridge_target_proj"]),"bridge_comp_proj":mean(lambda r:r["geometry"]["bridge_comp_proj"]),
"bridge_top1":mean(lambda r:r["states"]["bridge"]["top1"]),"bridge_positive_margin":mean(lambda r:r["states"]["bridge"]["margin"]>0)}
for k,v in S.items():print(f"{k.upper():34s} {v:+.6f}")
print("[10/15] COMPONENT DECOMPOSITION")
print(f"FORCE  TARGET={S['force_target_proj']:+.4f} COMP={S['force_comp_proj']:+.4f} DECISION={S['force_decision_proj']:+.4f} MARGIN_D={S['force_margin_delta']:+.4f}")
print(f"PACKET TARGET={S['packet_target_proj']:+.4f} COMP={S['packet_comp_proj']:+.4f} DECISION={S['packet_decision_proj']:+.4f} MARGIN_D={S['packet_margin_delta']:+.4f}")
print(f"BRIDGE TARGET={S['bridge_target_proj']:+.4f} COMP={S['bridge_comp_proj']:+.4f} DECISION={S['bridge_decision_proj']:+.4f} MARGIN_D={S['bridge_margin_delta']:+.4f}")
print("[11/15] BOTTLENECK CLASSIFICATION")
FORCE_DOM=S["force_margin_delta"]>S["packet_margin_delta"] and S["force_margin_delta"]>S["bridge_margin_delta"]
BRIDGE_HELP=S["bridge_margin_delta"]>0;PACKET_HELP=S["packet_margin_delta"]>0
if S["bridge_top1"]>0:CLASS="BOUNDARY_CROSSING_OBSERVED"
elif FORCE_DOM and PACKET_HELP and BRIDGE_HELP:CLASS="DECODER_PRIOR_DOMINANT_WITH_RESIDUAL_SIGNAL_DEFICIT"
elif PACKET_HELP and BRIDGE_HELP:CLASS="SIGNAL_STRENGTH_LIMITED"
elif not BRIDGE_HELP:CLASS="BRIDGE_DIRECTION_MISMATCH"
else:CLASS="MIXED_READOUT_BOTTLENECK"
print("CLASS:",CLASS)
print("[12/15] REQUIRED CLOSURE")
REQ=[max(0.0,-r["states"]["bridge"]["margin"]) for r in ROWS]
print(f"MEAN_REQUIRED_MARGIN={np.mean(REQ):.6f} | MEDIAN_REQUIRED_MARGIN={np.median(REQ):.6f} | ZERO_REQUIRED={np.mean(np.array(REQ)==0):.6f}")
print("[13/15] CASE + TARGET AUDIT")
print("CASE_ISOLATED=True | CROSS_CASE_DATA=False | TARGET_ENGINE_ACCESS=False | TARGET_POSTHOC_ONLY=True")
print("[14/15] SCIENTIFIC AUDIT")
print("NO TRAINING | NO OPTIMIZATION | NO DOSE SWEEP | NO TARGET-HEAD IN ENGINE | TARGET USED ONLY AFTER ALL FORWARDS")
print("[15/15] INTEGRITY + SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("Hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["max_seasc_dev"]>1e-6 or AUD["max_bridge_dev"]>1e-6:raise RuntimeError("Dose audit failure")
R={"schema":"akbascore.test314.v1","test":"TEST 314","start":START,"end":utc(),"model":MODEL_ID,
"question":"Where does the remaining answer-blind decision-boundary deficit arise after TEST313 removes abstention?",
"engine":"TEST313 locked","states":["standard","forced","forced+packet","forced+packet+bridge"],"case_isolation":True,
"summary":S,"classification":CLASS,"required_margin":{"mean":float(np.mean(REQ)),"median":float(np.median(REQ)),"zero_fraction":float(np.mean(np.array(REQ)==0))},
"rows":ROWS,"integrity":{"target_engine_access":False,"target_posthoc_only":True,"training":False,"optimization":False,"dose_sweep":False,"target_head_engine":False,
"cross_case_data":False,"weight_update":False,"rss":RSS,"bridge":BRIDGE_REL,"seasc_calls":AUD["seasc"],"bridge_calls":AUD["bridge"],
"max_seasc_dev":AUD["max_seasc_dev"],"max_bridge_dev":AUD["max_bridge_dev"],"hooks_remaining":len(ACTIVE),"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
sha=hashlib.sha256(canon(R)).hexdigest();rid=f"T314-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json";tp=ROOT/f"{rid}.txt"
jp.write_bytes(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False).encode())
txt=["="*108,"TEST 314 - ANSWER-BLIND DECISION-BOUNDARY X-RAY","="*108,
f"RANK STD={S['std_rank']:.1f} FORCE={S['force_rank']:.1f} PACKET={S['packet_rank']:.1f} BRIDGE={S['bridge_rank']:.1f}",
f"MARGIN STD={S['std_margin']:+.4f} FORCE={S['force_margin']:+.4f} PACKET={S['packet_margin']:+.4f} BRIDGE={S['bridge_margin']:+.4f}",
f"MARGIN DELTA FORCE={S['force_margin_delta']:+.4f} PACKET={S['packet_margin_delta']:+.4f} BRIDGE={S['bridge_margin_delta']:+.4f}",
f"REQUIRED CLOSURE MEAN={np.mean(REQ):.4f} MEDIAN={np.median(REQ):.4f}",f"CLASS={CLASS}",
f"CASE_ISOLATED=TRUE | TARGET POST-HOC ONLY | RSS={RSS:.9f} | BRIDGE={BRIDGE_REL:.3f} | WEIGHT INTEGRITY PASS",
f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]
tp.write_text("\n".join(txt),encoding="utf-8")
print("="*108);print("TEST 314 COMPLETE")
print(f"RANK STD={S['std_rank']:.1f} FORCE={S['force_rank']:.1f} PACKET={S['packet_rank']:.1f} BRIDGE={S['bridge_rank']:.1f}")
print(f"MARGIN STD={S['std_margin']:+.3f} FORCE={S['force_margin']:+.3f} PACKET={S['packet_margin']:+.3f} BRIDGE={S['bridge_margin']:+.3f}")
print(f"REQUIRED CLOSURE MEAN={np.mean(REQ):.3f} MEDIAN={np.median(REQ):.3f} | CLASS={CLASS}")
print("CASE_ISOLATED=TRUE | TARGET POST-HOC ONLY | WEIGHT INTEGRITY PASS | AKBASCORE HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
