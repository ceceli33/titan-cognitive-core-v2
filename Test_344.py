# ================================================================================================
# AKBASCORE - TEST 344
# TEMPORAL DECAYED SEASC - ASSISTANT-TRANSITION READOUT
# SAME TEST342 CASES | ABSOLUTE ASSISTANT INDEX | SPIKE vs FAST vs SLOW vs FULL
# SAME COMPILER | SAME ARGMAX(DISP) | SAME RSS | TARGET SEALED THROUGH ROUTER
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
SEED=344;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYS_STD="You are a concise reasoning assistant. Use only the information in the prompt."
SYS_FORCE="Read the answer from the internal state. Return exactly one short answer. Never explain, refuse, say information is missing, or mention the prompt."
TOTAL=28;H=3584;END=25;EPS=1e-8;TARGET_RSS=.250235055;IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20
PROFILES={"SPIKE":lambda k:1. if k==0 else 0.,"FAST":lambda k:.5**k,"SLOW":lambda k:.75**k,"FULL":lambda k:1.}
ROOT=Path("/content/AKBASCORE_TEST344");ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
CASES=[
("During the night inspection, technician Bravik disconnected the auxiliary compressor before the alarms were tested.","Who disconnected the auxiliary compressor?","Bravik"),
("The mineral recovered from Shaft Twelve was entered into the catalogue under the designation Norvex.","What designation was given to the recovered mineral?","Norvex"),
("After the final audit, the shipment cleared for export carried verification number 583.","What verification number was assigned to the cleared shipment?","583"),
("The research vessel visited Palermo and Split before the recovered sensor was unloaded in Dubrovnik.","Where was the recovered sensor unloaded?","Dubrovnik"),
("After testing ceramic, steel, and obsidian, the engineers selected obsidian for the protective shell.","Which material was selected for the protective shell?","obsidian"),
("The status panel glowed yellow during startup and changed to magenta after synchronization.","What color did the status panel become after synchronization?","magenta"),
("Near the eastern access hatch, Tavren stopped walking and began crawling beneath the support frame.","What did Tavren begin doing beneath the support frame?","crawling"),
("The original surface appeared rough, while the treated sample was described as glossy.","How was the treated sample described?","glossy"),
("From the equipment locker, Selma retrieved a compass while leaving the damaged receiver behind.","What object did Selma retrieve?","compass"),
("The emergency authentication phrase recorded by the controller is Dorevian.","What is the emergency authentication phrase?","Dorevian"),
("Merik prepared the measurement rig, but the final alignment was performed by Talora.","Who performed the final alignment?","Talora"),
("The transport train passed through Prague and Vienna before terminating its route in Bratislava.","Where did the transport train terminate its route?","Bratislava"),
("The laboratory evaluated aluminum, tungsten, and titanium before choosing tungsten for the central support.","Which material was chosen for the central support?","tungsten"),
("The main beacon flashed green, whereas the secondary beacon flashed scarlet.","What color did the secondary beacon flash?","scarlet"),
("The display initially showed 326, but after recalculation the certified value was 817.","What was the certified value after recalculation?","817"),
("When the retaining clamp was released, the membrane immediately began expanding.","What did the membrane begin doing?","expanding"),
("The first fabric was rigid, whereas the replacement fabric was described as velvety.","How was the replacement fabric described?","velvety"),
("The storage cabinet contained a chronometer, a lens, and a hygrometer; the hygrometer was chosen for testing.","Which item was chosen for testing?","hygrometer"),
("The prototype was provisionally named Aster, but its production name was changed to Kelvane.","What became the prototype's production name?","Kelvane"),
("Several workers secured the package, but Dr. Veyron personally transported the sealed capsule.","Who transported the sealed capsule?","Veyron"),
("The archaeological team stored the bronze disk in Nagoya before travelling onward to Sendai.","Where was the bronze disk stored?","Nagoya"),
("The untreated layer looked gray, but spectral analysis established the coating as indigo.","What was the coating's established color?","indigo"),
("Sample Q-4 contained several minerals but consisted primarily of feldspar.","What did Sample Q-4 primarily consist of?","feldspar"),
("The rover paused at Marker Eleven and then began descending toward the lower platform.","What did the rover begin doing near Marker Eleven?","descending")]
SHOWCASE=("Mustafa Akbaş placed an amber banner beneath the Galata Bridge where the validation demonstration was conducted.","What did Mustafa Akbaş do at the location of the validation demonstration?","placed an amber banner")
ALL=CASES+[SHOWCASE];QNULLS=["What information is requested?","Which unknown detail should be recovered?","Answer the question using the missing information.","What should be identified?","Which detail is being asked about?"]
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
def prompt(q,sysmsg=SYS_FORCE):return tok.apply_chat_template([{"role":"system","content":sysmsg},{"role":"user","content":q}],tokenize=False,add_generation_prompt=True)
def enc(q,sysmsg=SYS_FORCE):return tok(prompt(q,sysmsg),return_tensors="pt",add_special_tokens=False).to("cuda")
print("="*108);print("TEST 344 - TEMPORAL DECAYED SEASC - ASSISTANT-TRANSITION READOUT");print("="*108);print("START:",START)
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
print("FP:",[f"{x:.4f}" for x in FP0]);print(f"RSS={RSS:.9f} | L0-L25 | ABSOLUTE ASSISTANT INDEX | PROFILES=SPIKE/FAST/SLOW/FULL")
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
print("[4/18] TARGET-FREE PACKET BANK")
BANK=[]
for i,(s,q,_) in enumerate(ALL):
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
ACTIVE=set();AUD={"err":0.0}
def install(packet,assistant_pos,profile):
    hs=[]
    for L in range(END+1):
        rho=RHO[L]
        def hk(mod,args,out,L=L,rho=rho):
            y=out[0] if isinstance(out,tuple) else out;r=y.clone();n=y.shape[1]
            for idx in range(max(0,assistant_pos),n):
                k=idx-assistant_pos;g=float(PROFILES[profile](k))
                if g<=0:continue
                z=y[:,idx,:].float();zn=z.norm(dim=-1,keepdim=True).clamp_min(EPS);d=packet[L].to(z.device).view(1,-1)*zn*rho*g;r[:,idx,:]=(z+d).to(y.dtype)
                AUD["err"]=max(AUD["err"],abs(float(d.norm()/zn.squeeze().clamp_min(EPS))-rho*g))
            return (r,)+out[1:] if isinstance(out,tuple) else r
        h=layers[L].register_forward_hook(hk);hs.append(h);ACTIVE.add(id(h))
    return hs
@torch.inference_mode()
def run_ids(ids,packet=None,assistant_pos=None,profile=None):
    hs=install(packet,assistant_pos,profile) if packet is not None else []
    try:return model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,output_hidden_states=True,return_dict=True)
    finally:
        for h in hs:h.remove();ACTIVE.discard(id(h))
@torch.inference_mode()
def run(q,packet=None,profile=None):
    ids=enc(q)["input_ids"];apos=ids.shape[1]-2
    return run_ids(ids,packet,apos,profile) if packet is not None else run_ids(ids)
def term(o):return o.hidden_states[28][0,-1].float()
def lg(o):return model.lm_head(term(o).to(model.lm_head.weight.dtype)).float().cpu()
print("[5/18] FROZEN ARGMAX(DISP) ROUTER - TEST342 METHOD")
SELECT=[];PACKETS=[]
for i,(_,q,_) in enumerate(ALL):
    v=run(q);vh=term(v);ds=[]
    for x in BANK[i]:
        o=run(q,x["packet"],"FULL");ds.append(float((term(o)-vh).norm()/vh.norm().clamp_min(EPS)))
    j=int(np.argmax(ds));SELECT.append(j);PACKETS.append(BANK[i][j]["packet"]);tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,f"| ANCHOR={BANK[i][j]['anchor']!r} | DISP={ds[j]:.6f}")
print("[6/18] PACKETS FROZEN")
print("NO PROFILE SELECTION | NO ORACLE | TARGET_ACCESS=False")
print("[7/18] ABSOLUTE ASSISTANT INDEX AUDIT")
APOS=[]
for i,(_,q,_) in enumerate(ALL):
    ids=enc(q)["input_ids"];p=ids.shape[1]-2;APOS.append(p);tkn=tok.decode([int(ids[0,p])]);tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,f"| POS={p} | TOKEN={tkn!r}")
    if tkn.strip()!="assistant":raise RuntimeError("Assistant index mismatch")
print("[8/18] FIRST-TOKEN DISTRIBUTIONS - TARGET SEALED")
FIRST=[]
for i,(_,q,_) in enumerate(ALL):
    v=lg(run(q));d={"V":v}
    for p in PROFILES:d[p]=lg(run(q,PACKETS[i],p))
    FIRST.append(d);tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,"| VANILLA + 4 PROFILES FROZEN")
print("[9/18] TARGET OPENS - FIRST-TOKEN READOUT")
ROWS=[]
for i,(_,q,target) in enumerate(ALL):
    tid=tok(target,add_special_tokens=False)["input_ids"][0];r={"target":target,"anchor":BANK[i][SELECT[i]]["anchor"],"first_token":tok.decode([tid]),"profiles":{}}
    vr=int((FIRST[i]["V"]>FIRST[i]["V"][tid]).sum()+1)
    for p in PROFILES:
        x=FIRST[i][p];rr=int((x>x[tid]).sum()+1);r["profiles"][p]={"rank":rr,"gain":float(x[tid]-FIRST[i]["V"][tid]),"improved":rr<vr}
    r["vanilla_rank"]=vr;ROWS.append(r);tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,"| V=",vr," | "," | ".join(f"{p}:{r['profiles'][p]['rank']} G={r['profiles'][p]['gain']:+.2f}" for p in PROFILES))
print("[10/18] TEACHER-FORCED ABSOLUTE-POSITION SEQUENCE ASSAY")
@torch.inference_mode()
def seq_score(q,target,packet=None,profile=None):
    pids=enc(q)["input_ids"];apos=pids.shape[1]-2;tids=tok(target,return_tensors="pt",add_special_tokens=False)["input_ids"].to("cuda");ids=torch.cat([pids,tids],1)
    o=run_ids(ids,packet,apos,profile) if packet is not None else run_ids(ids);vals=[]
    for j in range(tids.shape[1]):
        lp=torch.log_softmax(o.logits.float()[0,pids.shape[1]+j-1],-1);vals.append(float(lp[int(tids[0,j])]))
    return vals
for i,(_,q,target) in enumerate(ALL):
    vv=seq_score(q,target);ROWS[i]["sequence"]={}
    for p in PROFILES:
        ww=seq_score(q,target,PACKETS[i],p);d=np.array(ww)-np.array(vv);ROWS[i]["sequence"][p]={"delta_seq":float(d.sum()),"first":float(d[0]),"continuation":float(d[1:].mean()) if len(d)>1 else None,"token_frac":float(np.mean(d>0))}
    tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,"| "," | ".join(f"{p}:dSEQ={ROWS[i]['sequence'][p]['delta_seq']:+.2f}" for p in PROFILES))
print("[11/18] PROFILE SUMMARY")
SUMMARY={}
for p in PROFILES:
    M=ROWS[:24];g=np.array([r["profiles"][p]["gain"] for r in M]);imp=np.array([r["profiles"][p]["improved"] for r in M]);rk=np.array([r["profiles"][p]["rank"] for r in M]);ds=np.array([r["sequence"][p]["delta_seq"] for r in M]);cont=[r["sequence"][p]["continuation"] for r in M if r["sequence"][p]["continuation"] is not None]
    SUMMARY[p]={"median_rank":float(np.median(rk)),"first_gain":float(g.mean()),"first_positive":float(np.mean(g>0)),"rank_improved":float(imp.mean()),"seq_mean":float(ds.mean()),"seq_positive":float(np.mean(ds>0)),"continuation_mean":float(np.mean(cont)) if cont else None,"continuation_positive":float(np.mean(np.array(cont)>0)) if cont else None}
    s=SUMMARY[p];print(f"{p:5s} | MED={s['median_rank']:.1f} | G={s['first_gain']:+.3f} | POS={s['first_positive']:.3f} | IMP={s['rank_improved']:.3f} | dSEQ={s['seq_mean']:+.3f} | SEQ+={s['seq_positive']:.3f}")
print("[12/18] PREDECLARED TEMPORAL TEST")
SPIKE=SUMMARY["SPIKE"];FAST=SUMMARY["FAST"];SLOW=SUMMARY["SLOW"];FULL=SUMMARY["FULL"]
DECAY_SUPPORTED=(FAST["seq_mean"]>SPIKE["seq_mean"] and FAST["seq_positive"]>=SPIKE["seq_positive"] and FAST["first_gain"]>0)
SLOW_SUPPORTED=(SLOW["seq_mean"]>SPIKE["seq_mean"] and SLOW["seq_positive"]>=SPIKE["seq_positive"] and SLOW["first_gain"]>0)
FULL_HARM=(FULL["seq_mean"]<max(FAST["seq_mean"],SLOW["seq_mean"]))
print(f"FAST>SPIKE={DECAY_SUPPORTED} | SLOW>SPIKE={SLOW_SUPPORTED} | FULL_WORSE_THAN_BEST_DECAY={FULL_HARM}")
print("[13/18] FIXED PROFILE COMPARISON")
for p in PROFILES:print(p,SUMMARY[p])
print("[14/18] SHOWCASE")
r=ROWS[24];print("-"*108);print("SOURCE :",SHOWCASE[0]);print("QUESTION:",SHOWCASE[1]);print("TARGET :",SHOWCASE[2]);print("ANCHOR :",r["anchor"]);print("ASSISTANT_POS:",APOS[24]);print("FIRST:"," | ".join(f"{p} R={r['profiles'][p]['rank']} G={r['profiles'][p]['gain']:+.3f}" for p in PROFILES));print("SEQ  :"," | ".join(f"{p} dSEQ={r['sequence'][p]['delta_seq']:+.3f}" for p in PROFILES));print("-"*108)
print("[15/18] CLASSIFICATION")
if DECAY_SUPPORTED or SLOW_SUPPORTED:CLASS="TEMPORAL_DECAYED_SEASC_SUPPORTED"
elif SPIKE["seq_mean"]>0 and SPIKE["seq_positive"]>=.70:CLASS="ASSISTANT_TRANSITION_SPIKE_SUPPORTED_DECAY_NOT_REQUIRED"
else:CLASS="TEMPORAL_DECAY_HYPOTHESIS_NOT_SUPPORTED_IN_THIS_FORM"
print("CLASS=",CLASS)
print("[16/18] LEAKAGE AUDIT")
print("SAME TEST342 CASES | ABSOLUTE ASSISTANT INDEX | PROFILES PREDECLARED | NO PROFILE SEARCH | TARGET POST-HOC/TEACHER-FORCED ONLY")
print("[17/18] SCIENTIFIC AUDIT")
print("MAIN PROJECT READOUT PATH | SAME COMPILER | SAME ARGMAX(DISP) | SAME RSS | SAME L0-L25 ADDITIVE SEASC | NO TRAINING | NO WEIGHT UPDATE")
print("[18/18] INTEGRITY + SAVE")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("Hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["err"]>1e-5:raise RuntimeError(f"Dose failure {AUD['err']}")
R={"schema":"akbascore.test344.v1","test":"TEST 344","start":START,"end":utc(),"model":MODEL_ID,"rss":RSS,"profiles":["SPIKE","FAST","SLOW","FULL"],"summary":SUMMARY,"classification":CLASS,"rows":ROWS,"integrity":{"main_project":True,"absolute_assistant_index":True,"profile_search":False,"target_engine_access":False,"target_posthoc_only":True,"training":False,"weight_update":False,"fp_start":FP0,"fp_end":fp(),"hooks_remaining":len(ACTIVE)}}
sha=hashlib.sha256(canon(R)).hexdigest();rid=f"T344-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json";tp=ROOT/f"{rid}.txt"
jp.write_text(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False),encoding="utf-8");tp.write_text("\n".join(["="*108,"TEST 344 - TEMPORAL DECAYED SEASC","="*108]+[f"{p}={SUMMARY[p]}" for p in PROFILES]+[f"CLASS={CLASS}",f"SHA={sha}"]),encoding="utf-8")
print("="*108);print("TEST 344 COMPLETE")
for p in PROFILES:
    s=SUMMARY[p];print(f"{p:5s} | MED={s['median_rank']:.1f} | FIRST_G={s['first_gain']:+.3f} | FIRST+={s['first_positive']:.3f} | IMP={s['rank_improved']:.3f} | dSEQ={s['seq_mean']:+.3f} | SEQ+={s['seq_positive']:.3f}")
print("CLASS=",CLASS);print("ABSOLUTE ASSISTANT INDEX | TARGET POST-HOC ONLY | WEIGHT INTEGRITY PASS | HOOKS=0");print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
