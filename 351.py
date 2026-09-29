# ================================================================================================
# AKBASCORE - TEST 351
# COUNTERFACTUAL VALUE-SWAP DECOMPOSITION
# SAME TYPE + RELATION + SYNTAX | ONLY VALUE CHANGES | FROZEN TEST350 ENGINE
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
SEED=351;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYS_STD="You are a concise reasoning assistant. Use only the information in the prompt."
SYS_FORCE="Read the answer from the internal state. Return exactly one short answer. Never explain, refuse, say information is missing, or mention the prompt."
TOTAL=28;H=3584;END=25;EPS=1e-8;TARGET_RSS=.250235055;IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20
ROOT=Path("/content/AKBASCORE_TEST351");ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
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
def prompt(q,sysmsg=SYS_FORCE):return tok.apply_chat_template([{"role":"system","content":sysmsg},{"role":"user","content":q}],tokenize=False,add_generation_prompt=True)
def enc(q,sysmsg=SYS_FORCE):return tok(prompt(q,sysmsg),return_tensors="pt",add_special_tokens=False).to("cuda")
def rank2(x,a,b):return 1 if x[a]>=x[b] else 2
print("="*108);print("TEST 351 - COUNTERFACTUAL VALUE-SWAP DECOMPOSITION");print("="*108);print("START:",START)
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
print("FP:",[f"{x:.4f}" for x in FP0]);print(f"RSS={RSS:.9f} | FULL ASSISTANT-OUTPUT | L0-L25")
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
        if ix:out.append({"text":m.group(0),"v":[unit(o.hidden_states[L+1][0,ix].float().mean(0).detach()) for L in range(TOTAL)]})
    return out
print("[3/18] QUERY NULL CACHE")
QN=[states(x) for x in QNULLS];print("NULL READY")
ACTIVE=set();AUD={"err":0.0}
def compile_bank(source,q):
    qh=states(q);qv=[unit(torch.stack([unit(qh[L]-QN[j][L]) for j in range(len(QN))]).mean(0)) for L in range(TOTAL)];sp=spans(source);chs=[]
    for a in range(len(sp)):
        packet=[]
        for L in range(TOTAL):
            cand=[]
            for b in range(len(sp)):
                if b==a:continue
                d=sp[b]["v"][L]-sp[a]["v"][L];cand.append((float(torch.dot(unit(d),qv[L])),d))
            cand.sort(key=lambda z:z[0],reverse=True);packet.append(unit(cand[0][1]))
        chs.append({"anchor":sp[a]["text"],"packet":packet})
    return chs
def install(packet,apos):
    hs=[]
    for L in range(END+1):
        rho=RHO[L]
        def hk(mod,args,out,L=L,rho=rho):
            y=out[0] if isinstance(out,tuple) else out;r=y.clone()
            for idx in range(max(0,apos),y.shape[1]):
                z=y[:,idx,:].float();zn=z.norm(dim=-1,keepdim=True).clamp_min(EPS);d=packet[L].to(z.device).view(1,-1)*zn*rho;r[:,idx,:]=(z+d).to(y.dtype)
                AUD["err"]=max(AUD["err"],abs(float(d.norm()/zn.squeeze().clamp_min(EPS))-rho))
            return (r,)+out[1:] if isinstance(out,tuple) else r
        h=layers[L].register_forward_hook(hk);hs.append(h);ACTIVE.add(id(h))
    return hs
@torch.inference_mode()
def run_ids(ids,packet=None,apos=None):
    hs=install(packet,apos) if packet is not None else []
    try:return model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,output_hidden_states=True,return_dict=True)
    finally:
        for h in hs:h.remove();ACTIVE.discard(id(h))
def term(o):return o.hidden_states[28][0,-1].float()
print("[4/18] COUNTERFACTUAL BANKS - TARGET SEALED")
BANK=[]
for i,(typ,sa,sb,q,_,_) in enumerate(PAIRS):
    ba=compile_bank(sa,q);bb=compile_bank(sb,q);BANK.append((ba,bb));print(f"{i+1:02d}/16 | {typ:8s} | A={len(ba)} B={len(bb)}")
print("[5/18] FROZEN ARGMAX(DISP) ROUTER - EACH SOURCE INDEPENDENT")
PACK=[];ROUT=[]
for i,(typ,sa,sb,q,_,_) in enumerate(PAIRS):
    ids=enc(q)["input_ids"];apos=ids.shape[1]-2;v=run_ids(ids);vh=term(v);sel=[]
    for side,bank in zip(("A","B"),BANK[i]):
        ds=[]
        for x in bank:ds.append(float((term(run_ids(ids,x["packet"],apos))-vh).norm()/vh.norm().clamp_min(EPS)))
        j=int(np.argmax(ds));sel.append((j,bank[j]["packet"],bank[j]["anchor"],ds[j]))
    ROUT.append(sel);PACK.append((sel[0][1],sel[1][1]));print(f"{i+1:02d}/16 | A={sel[0][2]!r}:{sel[0][3]:.4f} | B={sel[1][2]!r}:{sel[1][3]:.4f}")
print("[6/18] ENGINE FROZEN")
print("TEST350 ENGINE UNCHANGED | SAME ARGMAX(DISP) | NO VALUE/TARGET USED FOR ROUTING")
print("[7/18] PACKET DIFFERENCE GEOMETRY")
PGEOM=[]
for i,(typ,sa,sb,q,_,_) in enumerate(PAIRS):
    pa,pb=PACK[i];cs=[float(torch.dot(unit(pa[L]-pb[L]),unit(pa[L]))) for L in range(END+1)];pd=[float((pa[L]-pb[L]).norm()) for L in range(END+1)]
    PGEOM.append({"mean_diff":float(np.mean(pd)),"l25_diff":pd[25],"mean_selfproj":float(np.mean(cs))});print(f"{i+1:02d}/16 | {typ:8s} | MEAN|A-B|={np.mean(pd):.4f} | L25={pd[25]:.4f}")
print("[8/18] TERMINAL DISTRIBUTIONS FROZEN - VALUE SEALED")
FROZEN=[]
for i,(typ,sa,sb,q,_,_) in enumerate(PAIRS):
    ids=enc(q)["input_ids"];apos=ids.shape[1]-2;v=run_ids(ids);a=run_ids(ids,PACK[i][0],apos);b=run_ids(ids,PACK[i][1],apos)
    V=v.logits[0,-1].float().cpu();A=a.logits[0,-1].float().cpu();B=b.logits[0,-1].float().cpu();hv=term(v);ha=term(a);hb=term(b)
    FROZEN.append({"V":V,"A":A,"B":B,"dhA":(ha-hv).cpu(),"dhB":(hb-hv).cpu(),"dhAB":(ha-hb).cpu()});print(f"{i+1:02d}/16 | {typ:8s} | TERMINAL A/B FROZEN")
print("[9/18] VALUE OPENS - FIRST TOKEN IDS")
VID=[]
for i,(typ,sa,sb,q,va,vb) in enumerate(PAIRS):
    ia=int(tok(va,add_special_tokens=False)["input_ids"][0]);ib=int(tok(vb,add_special_tokens=False)["input_ids"][0]);VID.append((ia,ib));print(f"{i+1:02d}/16 | A={tok.decode([ia])!r} | B={tok.decode([ib])!r}")
print("[10/18] COUNTERFACTUAL CROSSOVER")
ROWS=[]
for i,(typ,sa,sb,q,va,vb) in enumerate(PAIRS):
    ia,ib=VID[i];f=FROZEN[i];V,A,B=f["V"],f["A"],f["B"]
    av=float(A[ia]-A[ib]);bv=float(B[ib]-B[ia]);vvA=float(V[ia]-V[ib]);vvB=float(V[ib]-V[ia])
    dA=av-vvA;dB=bv-vvB;cross=(av>0 and bv>0);bothgain=(dA>0 and dB>0)
    swap=float((A[ia]-A[ib])-(B[ia]-B[ib]))
    r={"type":typ,"value_a":va,"value_b":vb,"token_a":tok.decode([ia]),"token_b":tok.decode([ib]),"A_margin":av,"B_margin":bv,"A_margin_gain":dA,"B_margin_gain":dB,"swap_effect":swap,"crossover":cross,"both_gain":bothgain,"packet_geom":PGEOM[i]};ROWS.append(r)
    print(f"{i+1:02d}/16 | {typ:8s} | AΔ={dA:+.3f} BΔ={dB:+.3f} | SWAP={swap:+.3f} | CROSS={cross}")
print("[11/18] VALUE IDENTITY SUMMARY")
DEV=ROWS[:15];A_POS=float(np.mean([r["A_margin_gain"]>0 for r in DEV]));B_POS=float(np.mean([r["B_margin_gain"]>0 for r in DEV]));BOTH=float(np.mean([r["both_gain"] for r in DEV]));CROSS=float(np.mean([r["crossover"] for r in DEV]));SWAP_POS=float(np.mean([r["swap_effect"]>0 for r in DEV]));SWAP_MEAN=float(np.mean([r["swap_effect"] for r in DEV]))
print(f"N=15 | A_GAIN+={A_POS:.6f} | B_GAIN+={B_POS:.6f} | BOTH_GAIN={BOTH:.6f} | CROSSOVER={CROSS:.6f} | SWAP+={SWAP_POS:.6f} | SWAP_MEAN={SWAP_MEAN:+.6f}")
print("[12/18] TYPE BREAKDOWN")
for typ in sorted(set(r["type"] for r in DEV)):
    z=[r for r in DEV if r["type"]==typ];print(f"{typ:8s} | N={len(z)} | SWAP={np.mean([r['swap_effect'] for r in z]):+.3f} | BOTH={np.mean([r['both_gain'] for r in z]):.3f} | CROSS={np.mean([r['crossover'] for r in z]):.3f}")
print("[13/18] PREDECLARED GATES")
VALUE_DIRECTIONAL=SWAP_POS>=.70 and SWAP_MEAN>0
VALUE_BIDIRECTIONAL=BOTH>=.60
VALUE_DECISIVE=CROSS>=.60
print(f"VALUE_DIRECTIONAL={VALUE_DIRECTIONAL} | VALUE_BIDIRECTIONAL={VALUE_BIDIRECTIONAL} | VALUE_DECISIVE={VALUE_DECISIVE}")
print("[14/18] SHOWCASE")
G=ROWS[15];print("-"*108);print("A:",PAIRS[15][1]);print("B:",PAIRS[15][2]);print("Q:",PAIRS[15][3]);print(f"A={G['value_a']} | B={G['value_b']}");print(f"A_MARGIN_GAIN={G['A_margin_gain']:+.3f} | B_MARGIN_GAIN={G['B_margin_gain']:+.3f} | SWAP={G['swap_effect']:+.3f} | CROSS={G['crossover']}");print("-"*108)
print("[15/18] CLASSIFICATION")
if VALUE_DIRECTIONAL and VALUE_BIDIRECTIONAL and VALUE_DECISIVE:CLASS="COUNTERFACTUAL_VALUE_IDENTITY_STRONGLY_ENCODED"
elif VALUE_DIRECTIONAL and VALUE_BIDIRECTIONAL:CLASS="COUNTERFACTUAL_VALUE_IDENTITY_ENCODED_BUT_NOT_DECISION_DOMINANT"
elif VALUE_DIRECTIONAL:CLASS="COUNTERFACTUAL_VALUE_COMPONENT_PRESENT_BUT_WEAK_OR_ASYMMETRIC"
else:CLASS="COUNTERFACTUAL_VALUE_IDENTITY_NOT_SYSTEMATICALLY_ENCODED"
print("CLASS=",CLASS)
print("[16/18] INTERPRETATION LOCK")
print("TYPE/RELATION/SYNTAX MATCHED WITHIN PAIR | ONLY VALUE CHANGED | TARGET POST-HOC | FIRST TOKEN DIAGNOSTIC | NO ORACLE ROUTING")
print("[17/18] SCIENTIFIC AUDIT")
print("TEST350 ENGINE UNCHANGED | SAME COMPILER | SAME ARGMAX(DISP) | SAME RSS | FULL ASSISTANT-OUTPUT | NO TRAINING | NO WEIGHT UPDATE")
print("[18/18] INTEGRITY + SAVE")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("Hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["err"]>1e-5:raise RuntimeError(f"Dose failure {AUD['err']}")
S={"n":15,"a_gain_positive":A_POS,"b_gain_positive":B_POS,"both_gain":BOTH,"crossover":CROSS,"swap_positive":SWAP_POS,"swap_mean":SWAP_MEAN}
R={"schema":"akbascore.test351.v1","test":"TEST 351","start":START,"end":utc(),"model":MODEL_ID,"rss":RSS,"summary":S,"classification":CLASS,"rows":ROWS,"integrity":{"engine":"TEST350_FULL_ASSISTANT_OUTPUT","target_engine_access":False,"target_router_access":False,"target_posthoc_only":True,"training":False,"weight_update":False,"fp_start":FP0,"fp_end":fp(),"hooks_remaining":len(ACTIVE)}}
sha=hashlib.sha256(canon(R)).hexdigest();rid=f"T351-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json";tp=ROOT/f"{rid}.txt"
jp.write_text(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False),encoding="utf-8")
tp.write_text("\n".join(["="*108,"TEST 351 - COUNTERFACTUAL VALUE-SWAP DECOMPOSITION","="*108]+[f"{k}={v}" for k,v in S.items()]+[f"CLASS={CLASS}",f"SHA={sha}"]),encoding="utf-8")
print("="*108);print("TEST 351 COMPLETE");print(f"N=15 | A+={A_POS:.3f} | B+={B_POS:.3f} | BOTH={BOTH:.3f} | CROSS={CROSS:.3f} | SWAP+={SWAP_POS:.3f} | SWAP_MEAN={SWAP_MEAN:+.3f}");print("CLASS=",CLASS);print("TARGET POST-HOC ONLY | WEIGHTS FROZEN | HOOKS=0");print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
