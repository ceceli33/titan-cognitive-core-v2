# ================================================================================================
# AKBASCORE - TEST 353
# DUAL-CHANNEL COMPILER: RELATION + TARGET-FREE VALUE RESIDUAL
# TEST352 FIX | RELATION CHANNEL UNCHANGED | LAMBDA=.25 FROZEN | NO SEARCH
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
SEED=353;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYS_STD="You are a concise reasoning assistant. Use only the information in the prompt.";SYS_FORCE="Read the answer from the internal state. Return exactly one short answer. Never explain, refuse, say information is missing, or mention the prompt."
TOTAL=28;H=3584;END=25;EPS=1e-8;TARGET_RSS=.250235055;IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20;LAM=.25
ROOT=Path("/content/AKBASCORE_TEST353");ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
PAIRS=[
("PERSON","During the final inspection, technician Bravik disconnected the auxiliary compressor before the alarms were tested.","During the final inspection, technician Talora disconnected the auxiliary compressor before the alarms were tested.","Who disconnected the auxiliary compressor?","Bravik","Talora"),
("PERSON","The final alignment of the measurement rig was performed by Talora after calibration.","The final alignment of the measurement rig was performed by Veyron after calibration.","Who performed the final alignment?","Talora","Veyron"),
("PLACE","The recovered sensor was unloaded in Dubrovnik after the research vessel completed its route.","The recovered sensor was unloaded in Bratislava after the research vessel completed its route.","Where was the recovered sensor unloaded?","Dubrovnik","Bratislava"),
("PLACE","The bronze disk was stored in Nagoya before the archaeological team continued onward.","The bronze disk was stored in Dubrovnik before the archaeological team continued onward.","Where was the bronze disk stored?","Nagoya","Dubrovnik"),
("COLOR","After synchronization, the status panel changed to magenta.","After synchronization, the status panel changed to scarlet.","What color did the status panel become after synchronization?","magenta","scarlet"),
("COLOR","Spectral analysis established the coating as indigo.","Spectral analysis established the coating as magenta.","What was the coating's established color?","indigo","magenta"),
("MATERIAL","The engineers selected obsidian for the protective shell after testing several materials.","The engineers selected tungsten for the protective shell after testing several materials.","Which material was selected for the protective shell?","obsidian","tungsten"),
("MATERIAL","The laboratory chose tungsten for the central support after evaluation.","The laboratory chose feldspar for the central support after evaluation.","Which material was chosen for the central support?","tungsten","feldspar"),
("ACTION","Near the access hatch, Tavren began crawling beneath the support frame.","Near the access hatch, Tavren began descending beneath the support frame.","What did Tavren begin doing?","crawling","descending"),
("ACTION","When the retaining clamp was released, the membrane began expanding.","When the retaining clamp was released, the membrane began descending.","What did the membrane begin doing?","expanding","descending"),
("NAME","The recovered mineral was entered into the catalogue under the designation Norvex.","The recovered mineral was entered into the catalogue under the designation Dorevian.","What designation was given to the recovered mineral?","Norvex","Dorevian"),
("NAME","The prototype's production name was changed to Kelvane.","The prototype's production name was changed to Norvex.","What became the prototype's production name?","Kelvane","Norvex"),
("NUMBER","After recalculation, the certified value was 817.","After recalculation, the certified value was 583.","What was the certified value after recalculation?","817","583"),
("OBJECT","From the equipment locker, Selma retrieved a compass.","From the equipment locker, Selma retrieved a hygrometer.","What object did Selma retrieve?","compass","hygrometer"),
("PROPERTY","After treatment, the sample was described as glossy.","After treatment, the sample was described as velvety.","How was the treated sample described?","glossy","velvety"),
("SHOWCASE","Mustafa Akbaş placed an amber banner beneath the Galata Bridge where the validation demonstration was conducted.","Mustafa Akbaş raised an amber banner beneath the Galata Bridge where the validation demonstration was conducted.","What did Mustafa Akbaş do at the location of the validation demonstration?","placed an amber banner","raised an amber banner")]
QNULLS=["What information is requested?","Which unknown detail should be recovered?","Answer the question using the missing information.","What should be identified?","Which detail is being asked about?"]
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
def ptxt(q,sysmsg):return tok.apply_chat_template([{"role":"system","content":sysmsg},{"role":"user","content":q}],tokenize=False,add_generation_prompt=True)
def enc(q,sysmsg):return tok(ptxt(q,sysmsg),return_tensors="pt",add_special_tokens=False).to("cuda")
print("="*108);print("TEST 353 - DUAL-CHANNEL COMPILER: RELATION + TARGET-FREE VALUE RESIDUAL");print("="*108);print("START:",START)
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
print("FP:",[f"{x:.4f}" for x in FP0]);print(f"RSS={RSS:.9f} | FULL ASSISTANT-OUTPUT | L0-L25 | LAMBDA={LAM:.2f}")
@torch.inference_mode()
def states(text):
    o=model(**enc(text,SYS_STD),use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)]
@torch.inference_mode()
def spans(text):
    e=tok(text,return_tensors="pt",add_special_tokens=False,return_offsets_mapping=True);offs=e["offset_mapping"][0].tolist()
    o=model(input_ids=e["input_ids"].to("cuda"),attention_mask=e["attention_mask"].to("cuda"),use_cache=False,output_hidden_states=True,return_dict=True);out=[]
    for m in re.finditer(r"\b[\w'-]+\b",text,re.UNICODE):
        a,b=m.span();ix=[j for j,(s,z) in enumerate(offs) if z>a and s<b]
        if ix:out.append({"text":m.group(0),"v":[o.hidden_states[L+1][0,ix].float().mean(0).detach().clone() for L in range(TOTAL)]})
    return out
print("[3/18] QUERY NULL CACHE")
QN=[states(x) for x in QNULLS];print("NULL READY")
ACTIVE=set();AUD={"err":0.0}
def install(packet,apos):
    hs=[]
    for L in range(END+1):
        rho=RHO[L]
        def hk(mod,args,out,L=L,rho=rho):
            y=out[0] if isinstance(out,tuple) else out;r=y.clone()
            for idx in range(max(0,apos),y.shape[1]):
                z=y[:,idx,:].float();zn=z.norm(dim=-1,keepdim=True).clamp_min(EPS);d=packet[L].to(z.device).view(1,-1)*zn*rho;r[:,idx,:]=(z+d).to(y.dtype);AUD["err"]=max(AUD["err"],abs(float(d.norm()/zn.squeeze().clamp_min(EPS))-rho))
            return (r,)+out[1:] if isinstance(out,tuple) else r
        h=layers[L].register_forward_hook(hk);hs.append(h);ACTIVE.add(id(h))
    return hs
@torch.inference_mode()
def run(ids,packet=None,apos=None):
    hs=install(packet,apos) if packet is not None else []
    try:return model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,output_hidden_states=True,return_dict=True)
    finally:
        for h in hs:h.remove();ACTIVE.discard(id(h))
def term(o):return o.hidden_states[28][0,-1].float()
print("[4/18] TARGET-FREE DUAL-CHANNEL BANKS")
BANK=[]
for i,(typ,sa,sb,q,_,_) in enumerate(PAIRS):
    qh=states(q);qv=[unit(torch.stack([unit(qh[L]-QN[j][L]) for j in range(len(QN))]).mean(0)) for L in range(TOTAL)];pair=[]
    for source in (sa,sb):
        sp=spans(source);chs=[]
        for a in range(len(sp)):
            rel=[];val=[];dual=[];vp=[]
            for L in range(TOTAL):
                cand=[]
                for b in range(len(sp)):
                    if b==a:continue
                    d=sp[b]["v"][L]-sp[a]["v"][L];cand.append((float(torch.dot(unit(d),qv[L])),b,d))
                cand.sort(key=lambda x:x[0],reverse=True);rd=unit(cand[0][2]);rscore=cand[0][0]
                # VALUE residual: source token direction after removing selected relation direction.
                vc=[]
                for b in range(len(sp)):
                    x=sp[b]["v"][L].float();xr=x-rd*torch.dot(x,rd);vc.append((float(torch.dot(unit(xr),qv[L])),b,xr))
                vc.sort(key=lambda x:x[0],reverse=True);vd=unit(vc[0][2]);rel.append(rd);val.append(vd);dual.append(unit(rd+LAM*vd));vp.append(sp[vc[0][1]]["text"])
            chs.append({"anchor":sp[a]["text"],"rel":rel,"val":val,"dual":dual,"value_pick":vp})
        pair.append(chs)
    BANK.append(pair);print(f"{i+1:02d}/16 | {typ:8s} | A={len(pair[0])} B={len(pair[1])}")
print("[5/18] FROZEN ARGMAX(DISP) ROUTER - RELATION CHANNEL ONLY")
SELECT=[];PACK_REL=[];PACK_DUAL=[]
for i,(typ,sa,sb,q,_,_) in enumerate(PAIRS):
    ids=enc(q,SYS_FORCE)["input_ids"];apos=ids.shape[1]-2;vh=term(run(ids));ss=[];pr=[];pd=[]
    for side in range(2):
        ds=[]
        for x in BANK[i][side]:ds.append(float((term(run(ids,x["rel"],apos))-vh).norm()/vh.norm().clamp_min(EPS)))
        j=int(np.argmax(ds));ss.append(j);pr.append(BANK[i][side][j]["rel"]);pd.append(BANK[i][side][j]["dual"])
    SELECT.append(ss);PACK_REL.append(pr);PACK_DUAL.append(pd);print(f"{i+1:02d}/16 | A={BANK[i][0][ss[0]]['anchor']!r} | B={BANK[i][1][ss[1]]['anchor']!r}")
print("[6/18] VALUE CHANNEL AUDIT")
for i,(typ,sa,sb,q,_,_) in enumerate(PAIRS):
    a=BANK[i][0][SELECT[i][0]];b=BANK[i][1][SELECT[i][1]];same=np.mean([x.lower()==y.lower() for x,y in zip(a["value_pick"][:26],b["value_pick"][:26])])
    print(f"{i+1:02d}/16 | {typ:8s} | VALUE_PICK_EQ={same:.3f} | A25={a['value_pick'][25]!r} B25={b['value_pick'][25]!r}")
print("[7/18] PACKET SEPARATION RELATION vs DUAL")
SEP=[]
for i in range(16):
    rd=float(np.mean([(PACK_REL[i][0][L]-PACK_REL[i][1][L]).norm().item() for L in range(26)]));dd=float(np.mean([(PACK_DUAL[i][0][L]-PACK_DUAL[i][1][L]).norm().item() for L in range(26)]));SEP.append((rd,dd));print(f"{i+1:02d}/16 | REL={rd:.4f} | DUAL={dd:.4f} | d={dd-rd:+.4f}")
print("[8/18] RELATION BASELINE DISTRIBUTIONS FROZEN")
REL=[]
for i,(typ,sa,sb,q,_,_) in enumerate(PAIRS):
    ids=enc(q,SYS_FORCE)["input_ids"];apos=ids.shape[1]-2;v=run(ids);a=run(ids,PACK_REL[i][0],apos);b=run(ids,PACK_REL[i][1],apos);REL.append((v.logits[0,-1].float().cpu(),a.logits[0,-1].float().cpu(),b.logits[0,-1].float().cpu()));print(f"{i+1:02d}/16 | REL FROZEN")
print("[9/18] DUAL DISTRIBUTIONS FROZEN")
DUAL=[]
for i,(typ,sa,sb,q,_,_) in enumerate(PAIRS):
    ids=enc(q,SYS_FORCE)["input_ids"];apos=ids.shape[1]-2;a=run(ids,PACK_DUAL[i][0],apos);b=run(ids,PACK_DUAL[i][1],apos);DUAL.append((a.logits[0,-1].float().cpu(),b.logits[0,-1].float().cpu()));print(f"{i+1:02d}/16 | DUAL FROZEN")
print("[10/18] TARGET OPENS - COUNTERFACTUAL READOUT")
ROWS=[]
for i,(typ,sa,sb,q,va,vb) in enumerate(PAIRS):
    ia=int(tok(va,add_special_tokens=False)["input_ids"][0]);ib=int(tok(vb,add_special_tokens=False)["input_ids"][0]);V,RA,RB=REL[i];DA,DB=DUAL[i]
    rA=float((RA[ia]-RA[ib])-(V[ia]-V[ib]));rB=float((RB[ib]-RB[ia])-(V[ib]-V[ia]));dA=float((DA[ia]-DA[ib])-(V[ia]-V[ib]));dB=float((DB[ib]-DB[ia])-(V[ib]-V[ia]))
    rswap=float((RA[ia]-RA[ib])-(RB[ia]-RB[ib]));dswap=float((DA[ia]-DA[ib])-(DB[ia]-DB[ib]))
    rcross=bool(RA[ia]>RA[ib] and RB[ib]>RB[ia]);dcross=bool(DA[ia]>DA[ib] and DB[ib]>DB[ia])
    r={"i":i+1,"type":typ,"a":va,"b":vb,"rel_A_gain":rA,"rel_B_gain":rB,"dual_A_gain":dA,"dual_B_gain":dB,"rel_swap":rswap,"dual_swap":dswap,"rel_both":rA>0 and rB>0,"dual_both":dA>0 and dB>0,"rel_cross":rcross,"dual_cross":dcross,"rel_sep":SEP[i][0],"dual_sep":SEP[i][1]};ROWS.append(r)
    print(f"{i+1:02d}/16 | {typ:8s} | REL A={rA:+.3f} B={rB:+.3f} SW={rswap:+.3f} | DUAL A={dA:+.3f} B={dB:+.3f} SW={dswap:+.3f}")
print("[11/18] DEVELOPMENT SUMMARY")
D=ROWS[:15]
def avg(k):return float(np.mean([r[k] for r in D]))
REL_BOTH=avg("rel_both");DUAL_BOTH=avg("dual_both");REL_CROSS=avg("rel_cross");DUAL_CROSS=avg("dual_cross");REL_SWAP_POS=float(np.mean([r["rel_swap"]>0 for r in D]));DUAL_SWAP_POS=float(np.mean([r["dual_swap"]>0 for r in D]));REL_SWAP=avg("rel_swap");DUAL_SWAP=avg("dual_swap");REL_SEP=avg("rel_sep");DUAL_SEP=avg("dual_sep")
print(f"REL  | BOTH={REL_BOTH:.3f} CROSS={REL_CROSS:.3f} SWAP+={REL_SWAP_POS:.3f} SWAP={REL_SWAP:+.3f} SEP={REL_SEP:.3f}")
print(f"DUAL | BOTH={DUAL_BOTH:.3f} CROSS={DUAL_CROSS:.3f} SWAP+={DUAL_SWAP_POS:.3f} SWAP={DUAL_SWAP:+.3f} SEP={DUAL_SEP:.3f}")
print("[12/18] DELTA vs TEST351-STYLE RELATION BASELINE")
print(f"dBOTH={DUAL_BOTH-REL_BOTH:+.3f} | dCROSS={DUAL_CROSS-REL_CROSS:+.3f} | dSWAP+={DUAL_SWAP_POS-REL_SWAP_POS:+.3f} | dSWAP={DUAL_SWAP-REL_SWAP:+.3f} | dSEP={DUAL_SEP-REL_SEP:+.3f}")
print("[13/18] PREDECLARED GATES")
SEP_GAIN=DUAL_SEP>REL_SEP*1.10
VALUE_GAIN=DUAL_SWAP_POS>=.70 and DUAL_SWAP>REL_SWAP
BIDIR_GAIN=DUAL_BOTH>=.40 and DUAL_BOTH>REL_BOTH
DECISION_GAIN=DUAL_CROSS>REL_CROSS
print(f"SEP_GAIN={SEP_GAIN} | VALUE_GAIN={VALUE_GAIN} | BIDIR_GAIN={BIDIR_GAIN} | DECISION_GAIN={DECISION_GAIN}")
print("[14/18] SHOWCASE")
g=ROWS[15];print("-"*108);print("A:",PAIRS[15][1]);print("B:",PAIRS[15][2]);print("Q:",PAIRS[15][3]);print(f"REL  A={g['rel_A_gain']:+.3f} B={g['rel_B_gain']:+.3f} SWAP={g['rel_swap']:+.3f}");print(f"DUAL A={g['dual_A_gain']:+.3f} B={g['dual_B_gain']:+.3f} SWAP={g['dual_swap']:+.3f}");print(f"SEP REL={g['rel_sep']:.3f} DUAL={g['dual_sep']:.3f}");print("-"*108)
print("[15/18] CLASSIFICATION")
if VALUE_GAIN and BIDIR_GAIN and DECISION_GAIN:CLASS="DUAL_CHANNEL_COMPILER_RESTORES_COUNTERFACTUAL_VALUE_BINDING"
elif VALUE_GAIN and BIDIR_GAIN:CLASS="DUAL_CHANNEL_COMPILER_IMPROVES_VALUE_BINDING_WITHOUT_DECISIVE_CROSSOVER"
elif VALUE_GAIN or SEP_GAIN:CLASS="DUAL_CHANNEL_COMPILER_INCREASES_VALUE_SEPARATION_BUT_BINDING_REMAINS_WEAK"
else:CLASS="DUAL_CHANNEL_VALUE_RESIDUAL_DOES_NOT_IMPROVE_BINDING"
print("CLASS=",CLASS)
print("[16/18] INTERPRETATION LOCK")
print("RELATION ROUTER FROZEN | VALUE CHANNEL TARGET-FREE | LAMBDA=.25 PREDECLARED | NO LAMBDA SEARCH | TARGET POST-HOC ONLY")
print("[17/18] SCIENTIFIC AUDIT")
print("TEST352 DIAGNOSIS -> SINGLE COMPILER CHANGE | TEST351 RELATION CHANNEL PRESERVED | SAME RSS | SAME FULL ASSISTANT-OUTPUT | NO TRAINING | NO WEIGHT UPDATE")
print("[18/18] INTEGRITY + SAVE")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("Hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["err"]>1e-5:raise RuntimeError(f"Dose failure {AUD['err']}")
S={"n":15,"lambda":LAM,"rel_both":REL_BOTH,"dual_both":DUAL_BOTH,"rel_cross":REL_CROSS,"dual_cross":DUAL_CROSS,"rel_swap_positive":REL_SWAP_POS,"dual_swap_positive":DUAL_SWAP_POS,"rel_swap":REL_SWAP,"dual_swap":DUAL_SWAP,"rel_sep":REL_SEP,"dual_sep":DUAL_SEP}
R={"schema":"akbascore.test353.v1","test":"TEST 353","start":START,"end":utc(),"model":MODEL_ID,"rss":RSS,"summary":S,"classification":CLASS,"rows":ROWS,"integrity":{"relation_router_frozen":True,"value_target_free":True,"lambda_search":False,"target_engine_access":False,"target_posthoc_only":True,"training":False,"weight_update":False,"fp_start":FP0,"fp_end":fp(),"hooks_remaining":len(ACTIVE)}}
sha=hashlib.sha256(canon(R)).hexdigest();rid=f"T353-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json";tp=ROOT/f"{rid}.txt"
jp.write_text(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False),encoding="utf-8")
tp.write_text("\n".join(["="*108,"TEST 353 - DUAL-CHANNEL COMPILER","="*108]+[f"{k}={v}" for k,v in S.items()]+[f"CLASS={CLASS}",f"SHA={sha}"]),encoding="utf-8")
print("="*108);print("TEST 353 COMPLETE");print(f"REL  BOTH={REL_BOTH:.3f} CROSS={REL_CROSS:.3f} SWAP+={REL_SWAP_POS:.3f} SWAP={REL_SWAP:+.3f} SEP={REL_SEP:.3f}");print(f"DUAL BOTH={DUAL_BOTH:.3f} CROSS={DUAL_CROSS:.3f} SWAP+={DUAL_SWAP_POS:.3f} SWAP={DUAL_SWAP:+.3f} SEP={DUAL_SEP:.3f}");print("CLASS=",CLASS);print("TARGET POST-HOC ONLY | WEIGHTS FROZEN | HOOKS=0");print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
