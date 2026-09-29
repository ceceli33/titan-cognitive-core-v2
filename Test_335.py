# ================================================================================================
# AKBASCORE - TEST 335
# FROZEN DISP ROUTER -> POST-HOC TEACHER-FORCED TARGET-SEQUENCE X-RAY
# SAME FRESH-24 | SAME PACKET | SAME RSS | TARGET PREFIX POST-HOC ONLY | NO TARGET-BASED ROUTING
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
SEED=335;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYS_STD="You are a concise reasoning assistant. Use only the information in the prompt."
SYS_FORCE="Read the answer from the internal state. Return exactly one short answer. Never explain, refuse, say information is missing, or mention the prompt."
TOTAL=28;H_EXPECT=3584;END=25;EPS=1e-8;TARGET_RSS=.250235055
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20
ROOT=Path("/content/AKBASCORE_TEST335");ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
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
def chat(q,sysmsg=SYS_FORCE):
    s=tok.apply_chat_template([{"role":"system","content":sysmsg},{"role":"user","content":q}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to("cuda")
print("="*108);print("TEST 335 - FROZEN DISP ROUTER -> POST-HOC TEACHER-FORCED TARGET-SEQUENCE X-RAY");print("="*108);print("START:",START)
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
print("FP:",[f"{x:.4f}" for x in FP0]);print(f"RSS={RSS:.9f} | ROUTER=ARGMAX(DISP) | TARGET SEALED")
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
def forward_ids(ids,packet=None):
    hs=install(packet) if packet is not None else []
    try:return model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,output_hidden_states=True,return_dict=True)
    finally:
        for h in hs:h.remove();ACTIVE.discard(id(h))
@torch.inference_mode()
def forward_q(q,packet=None):return forward_ids(chat(q,SYS_FORCE)["input_ids"],packet)
print("[6/18] BLIND DISP ROUTER - EXACT TEST330-334 FORCED CONTEXT")
SELECT=[];PACKETS=[]
for i,(_,q) in enumerate(ALL):
    v=forward_q(q);vh=v.hidden_states[28][0,-1].float();aa=[]
    for x in BANK[i]:
        o=forward_q(q,x["packet"]);h=o.hidden_states[28][0,-1].float();aa.append(float((h-vh).norm()/vh.norm().clamp_min(EPS)))
    j=int(np.argmax(aa));SELECT.append(j);PACKETS.append(BANK[i][j]["packet"]);tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,f"| ANCHOR={BANK[i][j]['anchor']!r} | DISP={aa[j]:.6f}")
print("[7/18] ROUTER + PACKETS SEALED")
print("ARGMAX(DISP) COMPLETE | PACKETS FROZEN | TARGET_ACCESS=False")
print("[8/18] FIRST-TOKEN TERMINAL STATE FREEZE")
FIRST=[]
for i,(_,q) in enumerate(ALL):
    v=forward_q(q);s=forward_q(q,PACKETS[i]);vh=v.hidden_states[28][0,-1].float();sh=s.hidden_states[28][0,-1].float()
    vl=model.lm_head(vh.to(model.lm_head.weight.dtype)).float().cpu();sl=model.lm_head(sh.to(model.lm_head.weight.dtype)).float().cpu()
    FIRST.append({"vl":vl,"sl":sl})
    tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,"| V_TOP1=",repr(tok.decode([int(torch.argmax(vl))])),"| S_TOP1=",repr(tok.decode([int(torch.argmax(sl))])))
print("[9/18] PRE-TARGET SEAL COMPLETE")
print("ROUTER + PACKETS + FIRST-TOKEN DISTRIBUTIONS FROZEN | TARGET PREFIX NEVER USED BEFORE THIS POINT")
print("[10/18] TARGET SEAL OPEN - TEACHER-FORCED SEQUENCE X-RAY")
ROWS=[]
for i,((_,q),target) in enumerate(zip(ALL,SEALED)):
    base=chat(q,SYS_FORCE)["input_ids"];tids=tok(target,add_special_tokens=False)["input_ids"]
    if not tids:raise RuntimeError("Empty target")
    toks=[];vsum=0.;ssum=0.;improved=0
    for k,tid in enumerate(tids):
        prefix=torch.tensor([tids[:k]],device="cuda",dtype=torch.long) if k else None
        ids=torch.cat([base,prefix],dim=1) if prefix is not None else base
        v=forward_ids(ids);s=forward_ids(ids,PACKETS[i]);vh=v.hidden_states[28][0,-1].float();sh=s.hidden_states[28][0,-1].float()
        vl=model.lm_head(vh.to(model.lm_head.weight.dtype)).float();sl=model.lm_head(sh.to(model.lm_head.weight.dtype)).float()
        vlp=torch.log_softmax(vl,dim=-1);slp=torch.log_softmax(sl,dim=-1);vr=int((vl>vl[tid]).sum()+1);sr=int((sl>sl[tid]).sum()+1)
        a=float(vlp[tid]);b=float(slp[tid]);vsum+=a;ssum+=b;improved+=int(b>a)
        toks.append({"k":k,"id":int(tid),"token":tok.decode([tid]),"vanilla_logp":a,"steered_logp":b,"delta_logp":b-a,"vanilla_rank":vr,"steered_rank":sr,"rank_improved":sr<vr,"steered_top1":tok.decode([int(torch.argmax(sl))])})
    n=len(tids);r={"target":target,"anchor":BANK[i][SELECT[i]]["anchor"],"n_tokens":n,"vanilla_seq_logp":vsum,"steered_seq_logp":ssum,"delta_seq_logp":ssum-vsum,"vanilla_nll_per_token":-vsum/n,"steered_nll_per_token":-ssum/n,"nll_improvement_per_token":(ssum-vsum)/n,"token_logp_improved_fraction":improved/n,"all_tokens_logp_improved":improved==n,"tokens":toks};ROWS.append(r)
    tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,f"| T={target!r} | N={n} | dSEQ={r['delta_seq_logp']:+.3f} | d/T={r['nll_improvement_per_token']:+.3f} | TOK+={r['token_logp_improved_fraction']:.3f}")
print("[11/18] PRIMARY SUMMARY")
M=ROWS[:24]
SEQPOS=float(np.mean([r["delta_seq_logp"]>0 for r in M]));TOKPOS=float(np.mean([r["token_logp_improved_fraction"] for r in M]));ALLPOS=float(np.mean([r["all_tokens_logp_improved"] for r in M]))
MEAND=float(np.mean([r["delta_seq_logp"] for r in M]));MEDD=float(np.median([r["delta_seq_logp"] for r in M]));MEANPT=float(np.mean([r["nll_improvement_per_token"] for r in M]))
VNLL=float(np.mean([r["vanilla_nll_per_token"] for r in M]));SNLL=float(np.mean([r["steered_nll_per_token"] for r in M]))
print(f"SEQ_LOGP_IMPROVED={SEQPOS:.6f} | MEAN_dSEQ={MEAND:+.6f} | MED_dSEQ={MEDD:+.6f}")
print(f"MEAN_TOKEN_LOGP_IMPROVED_FRAC={TOKPOS:.6f} | ALL_TOKENS_IMPROVED={ALLPOS:.6f}")
print(f"MEAN_NLL/TOKEN={VNLL:.6f}->{SNLL:.6f} | IMPROVEMENT/TOKEN={MEANPT:+.6f}")
print("[12/18] FIRST TOKEN VS CONTINUATION")
FIRST_D=[];CONT_D=[];FIRST_POS=[];CONT_POS=[]
for r in M:
    FIRST_D.append(r["tokens"][0]["delta_logp"]);FIRST_POS.append(r["tokens"][0]["delta_logp"]>0)
    for x in r["tokens"][1:]:CONT_D.append(x["delta_logp"]);CONT_POS.append(x["delta_logp"]>0)
print(f"FIRST_TOKEN_MEAN_dLOGP={np.mean(FIRST_D):+.6f} | FIRST_POS={np.mean(FIRST_POS):.6f}")
if CONT_D:print(f"CONTINUATION_MEAN_dLOGP={np.mean(CONT_D):+.6f} | CONTINUATION_POS={np.mean(CONT_POS):.6f} | N={len(CONT_D)}")
else:print("CONTINUATION: NO MULTI-TOKEN TARGETS")
print("[13/18] MULTI-TOKEN SUBSET")
MULTI=[r for r in M if r["n_tokens"]>1]
if MULTI:
    print(f"MULTI_CASES={len(MULTI)} | SEQ_POS={np.mean([r['delta_seq_logp']>0 for r in MULTI]):.6f} | MEAN_dSEQ={np.mean([r['delta_seq_logp'] for r in MULTI]):+.6f} | MEAN_d/T={np.mean([r['nll_improvement_per_token'] for r in MULTI]):+.6f}")
else:print("MULTI_CASES=0")
print("[14/18] SHOWCASE")
G=ROWS[24];print("-"*108);print("SOURCE :",SHOWCASE[0]);print("QUESTION:",SHOWCASE[1]);print("ANCHOR :",G["anchor"]);print("TARGET :",G["target"])
print(f"N={G['n_tokens']} | SEQ_LOGP={G['vanilla_seq_logp']:.4f}->{G['steered_seq_logp']:.4f} | dSEQ={G['delta_seq_logp']:+.4f} | d/T={G['nll_improvement_per_token']:+.4f}")
for x in G["tokens"]:print(f"{x['k']:02d} | {x['token']!r} | LOGP={x['vanilla_logp']:.4f}->{x['steered_logp']:.4f} | d={x['delta_logp']:+.4f} | R={x['vanilla_rank']}->{x['steered_rank']} | TOP1={x['steered_top1']!r}")
print("-"*108)
print("[15/18] PREDECLARED DIAGNOSTIC")
FIRSTMEAN=float(np.mean(FIRST_D));CONTMEAN=float(np.mean(CONT_D)) if CONT_D else float("nan")
if CONT_D and CONTMEAN>0 and CONTMEAN>FIRSTMEAN+.5:CLASS="SEQUENCE_INITIATION_BOTTLENECK_SUPPORTED"
elif MEANPT>0 and SEQPOS>=.70:CLASS="TARGET_SEQUENCE_PROBABILITY_SYSTEMATICALLY_STRENGTHENED"
elif MEANPT>0:CLASS="TARGET_SEQUENCE_SIGNAL_PRESENT_BUT_NOT_UNIFORMLY_STRENGTHENED"
else:CLASS="NO_STABLE_FULL_SEQUENCE_TARGET_ADVANTAGE"
print(f"FIRST={FIRSTMEAN:+.6f} | CONT={CONTMEAN:+.6f} | SEQ_POS={SEQPOS:.6f} | dNLL/T={MEANPT:+.6f} | CLASS={CLASS}")
print("[16/18] LEAKAGE GUARD")
print("TARGET/PREFIX USED ONLY POST-HOC AFTER ROUTER + PACKETS + FIRST-TOKEN DISTRIBUTIONS SEALED | PREFIX NEVER USED FOR ROUTING/SELECTION/DOSE")
print("[17/18] SCIENTIFIC AUDIT")
print("SAME FRESH-24 AS TEST330-334 | ROUTER=ARGMAX(DISP) | SAME RSS | CASE ISOLATED | TEACHER FORCING DIAGNOSTIC ONLY | NO TRAINING | NO WEIGHT UPDATE")
print("[18/18] INTEGRITY + SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("Hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["dev"]>1e-6:raise RuntimeError("Dose audit failure")
R={"schema":"akbascore.test335.v1","test":"TEST 335","start":START,"end":utc(),"model":MODEL_ID,"router":"ARGMAX_DISP_FROZEN","rss":RSS,"summary":{"sequence_logp_improved":SEQPOS,"mean_delta_sequence_logp":MEAND,"median_delta_sequence_logp":MEDD,"mean_token_logp_improved_fraction":TOKPOS,"all_tokens_improved":ALLPOS,"vanilla_mean_nll_per_token":VNLL,"steered_mean_nll_per_token":SNLL,"mean_nll_improvement_per_token":MEANPT,"first_token_mean_delta_logp":FIRSTMEAN,"continuation_mean_delta_logp":CONTMEAN},"classification":CLASS,"rows":ROWS,"integrity":{"same_fresh24_as_test330_334":True,"case_isolated":True,"cross_case_data":False,"target_engine_access":False,"target_posthoc_only":True,"target_prefix_posthoc_only":True,"router_frozen_pre_target":True,"packets_frozen_pre_target":True,"first_token_distributions_frozen_pre_target":True,"teacher_forcing_diagnostic_only":True,"motor_layers":[0,25],"observation_layers":[26,27],"training":False,"weight_update":False,"calls":AUD["calls"],"max_dose_dev":AUD["dev"],"hooks_remaining":len(ACTIVE),"fp_start":FP0,"fp_end":fp()}}
sha=hashlib.sha256(canon(R)).hexdigest();rid=f"T335-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json";tp=ROOT/f"{rid}.txt"
jp.write_text(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False),encoding="utf-8")
tp.write_text("\n".join(["="*108,"TEST 335 - POST-HOC TEACHER-FORCED TARGET-SEQUENCE X-RAY","="*108,f"SEQ_LOGP_IMPROVED={SEQPOS:.6f}",f"MEAN_dSEQ={MEAND:+.6f}",f"MED_dSEQ={MEDD:+.6f}",f"NLL/TOKEN={VNLL:.6f}->{SNLL:.6f}",f"IMPROVEMENT/TOKEN={MEANPT:+.6f}",f"FIRST_dLOGP={FIRSTMEAN:+.6f}",f"CONT_dLOGP={CONTMEAN:+.6f}",f"CLASS={CLASS}","ROUTER/PACKET FROZEN | TARGET PREFIX POST-HOC ONLY | SAME RSS | WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]),encoding="utf-8")
print("="*108);print("TEST 335 COMPLETE")
print(f"SEQ IMPROVED={SEQPOS:.3f} | MEAN dSEQ={MEAND:+.3f} | NLL/TOKEN={VNLL:.3f}->{SNLL:.3f} | d/T={MEANPT:+.3f}")
print(f"FIRST dLOGP={FIRSTMEAN:+.3f} | CONT dLOGP={CONTMEAN:+.3f} | TOKEN+={TOKPOS:.3f}")
print("CLASS=",CLASS);print("ROUTER=ARGMAX(DISP) | TARGET PREFIX POST-HOC ONLY | SAME RSS | WEIGHT INTEGRITY PASS | AKBASCORE HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
