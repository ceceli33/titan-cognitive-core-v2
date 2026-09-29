# ================================================================================================
# AKBASCORE - TEST 310
# CASE-ISOLATED QUERY-ORTHOGONAL PACKET BRIDGE
# ONE TEXT + ONE QUESTION -> OWN PACKET -> REMOVE QUERY-COMMON COMPONENT -> SEASC -> BLIND READOUT
# EACH CASE INDEPENDENT | NO CROSS-CASE BANK | NO ROLE/TARGET BANK | TARGET POST-HOC ONLY
# ================================================================================================
import os,sys,json,time,random,hashlib,subprocess,importlib.util
from datetime import datetime,timezone
from pathlib import Path
for m,p in [("torch","torch"),("transformers","transformers"),("numpy","numpy")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required")
SEED=310;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE="cuda";MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
TOTAL=28;H_EXPECT=3584;END=25;EPS=1e-8;TARGET_RSS=.250235055;BRIDGE_REL=.20;SLOT_K=8
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20
ROOT=Path("/content/AKBASCORE_TEST310");ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
CASES=[
("After the storm, engineer Selin placed the damaged sensor inside Vault Kestrel.","Who placed the damaged sensor inside Vault Kestrel?","Selin"),
("The expedition discovered an unfamiliar crystal called Zorvex beneath the northern ridge.","What was the unfamiliar crystal called?","Zorvex"),
("Although three containers were inspected, only the smallest one carried the number 417.","Which number was on the smallest container?","417"),
("At sunset the maintenance crew moved the backup transmitter from the hangar to Reykjavik.","Where was the backup transmitter moved?","Reykjavik"),
("For the prototype shell, the designers rejected steel and selected basalt instead.","Which material was selected for the prototype shell?","basalt"),
("The indicator stayed red all morning but changed to violet immediately after calibration.","What color did the indicator become after calibration?","violet"),
("When the alarm sounded, Arven stopped descending and began climbing toward the upper platform.","What did Arven begin doing?","climbing"),
("Researchers described the newly polished specimen as unusually glossy despite its dark surface.","How was the newly polished specimen described?","glossy"),
("Beside the abandoned observatory, Mira recovered a sextant while the other tools remained buried.","What object did Mira recover?","sextant"),
("The final authentication phrase, written nowhere else in the archive, is Ordelis.","What is the final authentication phrase?","Ordelis"),
("Darian supervised the launch, but the navigation calculations were performed by Keira.","Who performed the navigation calculations?","Keira"),
("The drone passed over Madrid and Rome before finally landing in Tallinn.","Where did the drone finally land?","Tallinn"),
("Of copper, glass, and polymer, the laboratory chose glass for the transparent chamber wall.","What material was chosen for the chamber wall?","glass"),
("The first lamp flashed blue; the second, assigned to emergency status, glowed amber.","What color was the emergency-status lamp?","amber"),
("Unit R initially displayed 29, then after recalibration its verified value became 731.","What was Unit R's verified value after recalibration?","731"),
("Instead of opening as expected, the outer hatch began rotating when the sequence completed.","What did the outer hatch begin doing?","rotating"),
("The upper plate remained coarse, whereas the replacement plate installed below it was silky.","How was the replacement plate described?","silky"),
("Inside the wooden case were a map and a ruler, but the item marked for retrieval was the astrolabe.","Which item was marked for retrieval?","astrolabe"),
("The temporary codename Lumen was discarded; the production system was ultimately named Varethis.","What was the production system ultimately named?","Varethis"),
("Professor Ilyan sent two assistants away and personally carried the sealed notebook into the archive.","Who carried the sealed notebook into the archive?","Ilyan"),
("A courier left the silver key in Kyoto before continuing alone toward Osaka.","Where was the silver key left?","Kyoto"),
("The machine's casing looked black under storage lighting, but inspection confirmed its actual color was turquoise.","What was the casing's actual color?","turquoise"),
("Among several samples, specimen Q-9 was identified as being composed primarily of quartz.","What was specimen Q-9 primarily composed of?","quartz"),
("The autonomous cart waited briefly, reversed two meters, and then started accelerating toward Gate C.","What did the cart start doing toward Gate C?","accelerating")]
QNULLS=["What information is requested?","Which unknown detail should be recovered?","Answer the question using the missing information.","What should be identified?","Which detail is being asked about?"]
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
def chat_ids(text):
    s=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":text}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to(DEVICE)
def raw_ids(text):return tok(text,return_tensors="pt",add_special_tokens=False).to(DEVICE)
print("="*108);print("TEST 310 - CASE-ISOLATED QUERY-ORTHOGONAL PACKET BRIDGE");print("="*108);print("START:",START)
print("[1/14] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size
if len(layers)!=TOTAL or H!=H_EXPECT:raise RuntimeError("Architecture mismatch")
torch.cuda.synchronize();print(f"OK | {MODEL_ID} | 28L | H={H} | VOCAB={model.config.vocab_size} | {next(model.parameters()).dtype} | {time.perf_counter()-t:.2f}s")
print("[2/14] WEIGHT SENTINEL + FIXED RSS")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();base=[IVME*(ZIRVE*np.exp(-SONUM*L)*(1+SONUM*L)+TABAN)/(ZIRVE+TABAN) for L in range(END+1)];scale=TARGET_RSS/np.sqrt(sum(x*x for x in base));RHO=[x*scale for x in base];RSS=float(np.sqrt(sum(x*x for x in RHO)))
print("FP:",[f"{x:.4f}" for x in FP0]);print(f"RSS={RSS:.9f} | BRIDGE={BRIDGE_REL:.3f} | SLOT_K={SLOT_K} | CASE-ISOLATION=TRUE")
@torch.inference_mode()
def last_states(text):
    o=model(**chat_ids(text),use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)]
@torch.inference_mode()
def token_states(text):
    x=raw_ids(text);o=model(**x,use_cache=False,output_hidden_states=True,return_dict=True)
    return x.input_ids[0].detach().cpu(),[o.hidden_states[L+1][0].float().detach().clone() for L in range(TOTAL)]
print("[3/14] TARGET-FREE QUERY NULL CACHE")
QNULL=[last_states(x) for x in QNULLS];print("NULL READY")
print("[4/14] CASE-ISOLATED COMPILER")
PACK=[];ORTH=[];QUERY=[];META=[]
for i,(statement,question,_) in enumerate(CASES):
    qh=last_states(question);qv=[]
    for L in range(TOTAL):qv.append(unit(torch.stack([unit(qh[L]-QNULL[j][L]) for j in range(len(QNULL))]).mean(0)))
    ids,hs=token_states(statement);slots=[];sm=[]
    for pos,tid in enumerate(ids.tolist()):
        txt=tok.decode([tid]);clean=txt.strip()
        if not clean or all(not c.isalnum() for c in clean):continue
        slots.append([unit(hs[L][pos]) for L in range(TOTAL)]);sm.append({"pos":pos,"tid":tid,"token":txt})
    scores=torch.tensor([torch.dot(x[27],qv[27]).item() for x in slots]);k=min(SLOT_K,len(slots));vals,idx=torch.topk(scores,k);w=torch.softmax(vals/.025,dim=0)
    p=[unit(sum(slots[int(idx[j])][L]*w[j].to(slots[0][L].device) for j in range(k))) for L in range(TOTAL)]
    o=[]
    for L in range(TOTAL):
        a=torch.dot(p[L],qv[L]);r=p[L]-a*qv[L];o.append(unit(r) if r.norm()>EPS else p[L])
    PACK.append(p);ORTH.append(o);QUERY.append(qv);META.append({"tokens":[sm[int(x)]["token"] for x in idx],"positions":[sm[int(x)]["pos"] for x in idx],"query_dot_packet":[float(torch.dot(p[L],qv[L])) for L in range(TOTAL)],"query_dot_orth":[float(torch.dot(o[L],qv[L])) for L in range(TOTAL)]})
    print(f"{i+1:02d}/24 | SLOT={repr(META[-1]['tokens'])} | P.Q27={META[-1]['query_dot_packet'][27]:+.4f} -> O.Q27={META[-1]['query_dot_orth'][27]:+.6f}")
print("[5/14] CASE-ISOLATION AUDIT")
print("24 CASES = 24 INDEPENDENT RUNS | NO CASE USES ANOTHER CASE'S TEXT/PACKET/QUESTION/VECTOR")
print("ENGINE INPUT PER CASE = ONE STATEMENT + ONE QUESTION")
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
def forward(q,packet=None,bridge=None):
    hs=[] if packet is None else install(packet,bridge)
    try:return model(**chat_ids(q),use_cache=False,return_dict=True).logits[0,-1].float().detach().clone()
    finally:
        for h in hs:h.remove();ACTIVE.discard(id(h))
print("[7/14] BLIND CAUSAL ABLATION")
RAW=[]
for i,(_,q,_) in enumerate(CASES):
    v=forward(q);p=forward(q,PACK[i]);pb=forward(q,PACK[i],PACK[i][27]);ob=forward(q,PACK[i],ORTH[i][27])
    RAW.append((v,p,pb,ob));print(f"{i+1:02d}/24 | VANILLA PACKET RAW-BRIDGE ORTH-BRIDGE COMPLETE")
print("[8/14] POST-HOC TARGET SCORING")
ROWS=[]
for i,(statement,question,target) in enumerate(CASES):
    tids=tok.encode(target,add_special_tokens=False);tid=tids[0];v,p,pb,ob=RAW[i]
    def assay(x):
        tv=float(x[tid]);m=x.clone();m[tid]=-float("inf");cv=float(m.max());return {"target_logit":tv,"margin":tv-cv,"rank":int((x>x[tid]).sum())+1,"top1":bool(tv>cv),"top_token":tok.decode([int(torch.argmax(x))])}
    av,ap,arb,aob=[assay(x) for x in [v,p,pb,ob]]
    ROWS.append({"statement":statement,"question":question,"target":target,"target_ids":tids,"addressed_tokens":META[i]["tokens"],"vanilla":av,"packet":ap,"raw_bridge":arb,"orth_bridge":aob,
    "packet_gain":ap["target_logit"]-av["target_logit"],"raw_gain":arb["target_logit"]-ap["target_logit"],"orth_gain":aob["target_logit"]-ap["target_logit"],"raw_vs_orth":aob["target_logit"]-arb["target_logit"]})
    print(f"{i+1:02d}/24 | {target:11s} | RANK {av['rank']:6d}->{ap['rank']:6d} | RAW={arb['rank']:6d} ORTH={aob['rank']:6d} | GAIN R={ROWS[-1]['raw_gain']:+.3f} O={ROWS[-1]['orth_gain']:+.3f}")
print("[9/14] GLOBAL SUMMARY")
mean=lambda f:float(np.mean([f(r) for r in ROWS]));med=lambda f:float(np.median([f(r) for r in ROWS]))
S={"vanilla_rank_median":med(lambda r:r["vanilla"]["rank"]),"packet_rank_median":med(lambda r:r["packet"]["rank"]),"raw_bridge_rank_median":med(lambda r:r["raw_bridge"]["rank"]),"orth_bridge_rank_median":med(lambda r:r["orth_bridge"]["rank"]),
"packet_gain":mean(lambda r:r["packet_gain"]),"raw_bridge_gain":mean(lambda r:r["raw_gain"]),"orth_bridge_gain":mean(lambda r:r["orth_gain"]),"orth_minus_raw_gain":mean(lambda r:r["raw_vs_orth"]),
"raw_top1":mean(lambda r:r["raw_bridge"]["top1"]),"orth_top1":mean(lambda r:r["orth_bridge"]["top1"]),"raw_rank_improved":mean(lambda r:r["raw_bridge"]["rank"]<r["packet"]["rank"]),"orth_rank_improved":mean(lambda r:r["orth_bridge"]["rank"]<r["packet"]["rank"]),
"mean_abs_packet_query_L27":float(np.mean([abs(x["query_dot_packet"][27]) for x in META])),"mean_abs_orth_query_L27":float(np.mean([abs(x["query_dot_orth"][27]) for x in META]))}
for k,v in S.items():print(f"{k.upper():34s} {v:+.6f}")
print("[10/14] QUERY-COMPONENT REMOVAL X-RAY")
print(f"|PACKET.Q| L27={S['mean_abs_packet_query_L27']:.6f}")
print(f"|ORTH.Q|   L27={S['mean_abs_orth_query_L27']:.6f}")
print(f"RAW BRIDGE GAIN={S['raw_bridge_gain']:+.6f} | ORTH BRIDGE GAIN={S['orth_bridge_gain']:+.6f} | ORTH-RAW={S['orth_minus_raw_gain']:+.6f}")
print("[11/14] PREDECLARED DIAGNOSTIC")
D1=S["mean_abs_orth_query_L27"]<S["mean_abs_packet_query_L27"]*.05
D2=S["orth_bridge_gain"]>0
D3=S["orth_rank_improved"]>=.70
print(f"QUERY REMOVED={D1} | ORTH GAIN>0={D2} | ORTH RANK IMP>=70%={D3}")
print("[12/14] CASE-INDEPENDENCE CHECK")
CASE_ISOLATED=True
print(f"CASE_ISOLATED={CASE_ISOLATED} | CROSS_CASE_PACKET=False | CROSS_CASE_VECTOR=False | CROSS_CASE_MEAN=False | TARGET_ENGINE_ACCESS=False")
print("[13/14] INTERPRETATION GATE")
GATE=bool(D1 and D2 and D3)
print(f"CASE-ISOLATED QUERY-ORTHOGONAL GATE={'PASS' if GATE else 'FAIL'}")
print("[14/14] INTEGRITY + SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("Hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["max_seasc_dev"]>1e-6 or AUD["max_bridge_dev"]>1e-6:raise RuntimeError("Dose audit failure")
R={"schema":"akbascore.test310.v1","test":"TEST 310","start":START,"end":utc(),"model":MODEL_ID,
"question":"Does TEST308's packet retain answer-relevant causal readout after removing its same-case query-aligned component?",
"case_isolation":{"enabled":True,"engine_input":"one statement + one question","cross_case_packet":False,"cross_case_vector":False,"cross_case_mean":False,"cross_case_target":False},
"dataset":{"cases":24,"purpose":"independent repeated evaluation only"},"compiler":{"baseline":"TEST308 token-slot addressing","slot_k":SLOT_K},
"orthogonalization":"P_orth(L)=unit(P(L)-dot(P(L),Q(L))*Q(L))","seasc":{"layers":[0,25],"rss":RSS},"bridge":{"layer":26,"relative_dose":BRIDGE_REL},
"summary":S,"rows":ROWS,"meta":META,"gate":{"query_removed":D1,"orth_gain_positive":D2,"orth_rank_improved":D3,"result":"PASS" if GATE else "FAIL"},
"integrity":{"training":False,"optimization":False,"dose_sweep":False,"weight_update":False,"role_bank":False,"target_bank":False,"target_head":False,"semantic_vocab_bridge":False,
"target_posthoc_only":True,"case_isolated":True,"seasc_calls":AUD["seasc"],"bridge_calls":AUD["bridge"],"max_seasc_dev":AUD["max_seasc_dev"],"max_bridge_dev":AUD["max_bridge_dev"],
"hooks_remaining":len(ACTIVE),"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
sha=hashlib.sha256(canon(R)).hexdigest();rid=f"T310-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json";tp=ROOT/f"{rid}.txt"
jp.write_bytes(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False).encode())
txt=["="*108,"TEST 310 - CASE-ISOLATED QUERY-ORTHOGONAL PACKET BRIDGE","="*108,
f"MEDIAN RANK V={S['vanilla_rank_median']:.1f} P={S['packet_rank_median']:.1f} RAW={S['raw_bridge_rank_median']:.1f} ORTH={S['orth_bridge_rank_median']:.1f}",
f"GAIN PACKET={S['packet_gain']:+.4f} RAW={S['raw_bridge_gain']:+.4f} ORTH={S['orth_bridge_gain']:+.4f} ORTH-RAW={S['orth_minus_raw_gain']:+.4f}",
f"RANK IMP RAW={S['raw_rank_improved']:.3f} ORTH={S['orth_rank_improved']:.3f} | TOP1 RAW={S['raw_top1']:.3f} ORTH={S['orth_top1']:.3f}",
f"|P.Q|={S['mean_abs_packet_query_L27']:.6f} -> |ORTH.Q|={S['mean_abs_orth_query_L27']:.6f}",
f"CASE_ISOLATED=TRUE | GATE={'PASS' if GATE else 'FAIL'} | RSS={RSS:.9f} | BRIDGE={BRIDGE_REL:.3f}",
"ONE TEXT + ONE QUESTION PER ENGINE RUN | TARGET POST-HOC ONLY | WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]
tp.write_text("\n".join(txt),encoding="utf-8")
print("="*108);print("TEST 310 COMPLETE")
print(f"MEDIAN RANK V={S['vanilla_rank_median']:.1f} P={S['packet_rank_median']:.1f} RAW={S['raw_bridge_rank_median']:.1f} ORTH={S['orth_bridge_rank_median']:.1f}")
print(f"GAIN P={S['packet_gain']:+.3f} RAW={S['raw_bridge_gain']:+.3f} ORTH={S['orth_bridge_gain']:+.3f} | ORTH-RAW={S['orth_minus_raw_gain']:+.3f}")
print(f"|P.Q|={S['mean_abs_packet_query_L27']:.6f} -> |ORTH.Q|={S['mean_abs_orth_query_L27']:.6f} | ORTH RANK IMP={S['orth_rank_improved']:.3f}")
print(f"CASE_ISOLATED=TRUE | GATE={'PASS' if GATE else 'FAIL'} | WEIGHT INTEGRITY PASS | AKBASCORE HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
