# ================================================================================================
# AKBASCORE - TEST 343
# FROZEN ASSISTANT-ROLE WINDOW -> FULL TARGET SEQUENCE PROBABILITY
# SAME TEST342 FRESH-24 | REL=-2 FROZEN | SAME ARGMAX(DISP) | SAME ADDITIVE SEASC
# TARGET SEALED THROUGH COMPILER + ROUTER + BASE FORWARDS | TARGET USED ONLY TEACHER-FORCED POST-HOC
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
SEED=343;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYS_STD="You are a concise reasoning assistant. Use only the information in the prompt."
SYS_FORCE="Read the answer from the internal state. Return exactly one short answer. Never explain, refuse, say information is missing, or mention the prompt."
TOTAL=28;H=3584;END=25;REL=-2;EPS=1e-8;TARGET_RSS=.250235055;IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20
ROOT=Path("/content/AKBASCORE_TEST343");ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
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
ALL=CASES+[SHOWCASE];SEALED=tuple(x[2] for x in ALL)
QNULLS=["What information is requested?","Which unknown detail should be recovered?","Answer the question using the missing information.","What should be identified?","Which detail is being asked about?"]
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
def text_prompt(q,sysmsg=SYS_FORCE):
    return tok.apply_chat_template([{"role":"system","content":sysmsg},{"role":"user","content":q}],tokenize=False,add_generation_prompt=True)
def chat(q,sysmsg=SYS_FORCE):
    return tok(text_prompt(q,sysmsg),return_tensors="pt",add_special_tokens=False).to("cuda")
print("="*108);print("TEST 343 - FROZEN ASSISTANT-ROLE WINDOW -> FULL TARGET SEQUENCE PROBABILITY");print("="*108);print("START:",START)
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
print("FP:",[f"{x:.4f}" for x in FP0]);print(f"RSS={RSS:.9f} | ADDITIVE SEASC L0-L25 | REL=-2 FROZEN | TEACHER-FORCED READOUT POST-HOC")
@torch.inference_mode()
def states(text):
    o=model(**chat(text,SYS_STD),use_cache=False,output_hidden_states=True,return_dict=True)
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
def install(packet,mode="window"):
    hs=[]
    for L in range(END+1):
        rho=RHO[L]
        def hk(mod,args,out,L=L,rho=rho):
            y=out[0] if isinstance(out,tuple) else out;r=y.clone();n=y.shape[1]
            idx=list(range(n)) if mode=="full" else [n+REL]
            idx=[j for j in idx if 0<=j<n]
            if idx:
                z=y[:,idx,:].float();zn=z.norm(dim=-1,keepdim=True).clamp_min(EPS);d=packet[L].to(z.device).view(1,1,-1)*zn*rho;r[:,idx,:]=(z+d).to(y.dtype)
                AUD["err"]=max(AUD["err"],abs(float((d.norm(dim=-1)/zn.squeeze(-1)).mean())-rho))
            return (r,)+out[1:] if isinstance(out,tuple) else r
        h=layers[L].register_forward_hook(hk);hs.append(h);ACTIVE.add(id(h))
    return hs
@torch.inference_mode()
def run_ids(ids,packet=None,mode="window"):
    hs=install(packet,mode) if packet is not None else []
    try:
        am=torch.ones_like(ids,device=ids.device)
        return model(input_ids=ids,attention_mask=am,use_cache=False,output_hidden_states=True,return_dict=True)
    finally:
        for h in hs:h.remove();ACTIVE.discard(id(h))
@torch.inference_mode()
def run(q,packet=None,mode="window"):return run_ids(chat(q)["input_ids"],packet,mode)
def term(o):return o.hidden_states[28][0,-1].float()
print("[5/18] FROZEN ARGMAX(DISP) ROUTER - EXACT TEST342 METHOD")
SELECT=[];PACKETS=[]
for i,(_,q,_) in enumerate(ALL):
    v=run(q);vh=term(v);ds=[]
    for x in BANK[i]:
        o=run(q,x["packet"],"full");ds.append(float((term(o)-vh).norm()/vh.norm().clamp_min(EPS)))
    j=int(np.argmax(ds));SELECT.append(j);PACKETS.append(BANK[i][j]["packet"]);tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,f"| ANCHOR={BANK[i][j]['anchor']!r} | DISP={ds[j]:.6f}")
print("[6/18] ROUTER + WINDOW FROZEN")
print("REL=-2 | NO WINDOW SEARCH | NO ORACLE | NO TARGET ACCESS")
print("[7/18] ASSISTANT TOKEN AUDIT")
for i,(_,q,_) in enumerate(ALL):
    ids=chat(q)["input_ids"][0].tolist();tkn=tok.decode([ids[len(ids)+REL]]);tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,f"| REL=-2 | TOKEN={tkn!r}")
    if tkn.strip()!="assistant":raise RuntimeError("REL=-2 token mismatch")
print("[8/18] BASE QUESTION FORWARDS - TARGET SEALED")
BASE=[]
for i,(_,q,_) in enumerate(ALL):
    v=run(q);w=run(q,PACKETS[i]);BASE.append({"disp":float((term(w)-term(v)).norm()/term(v).norm().clamp_min(EPS))})
    tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,f"| DISP={BASE[-1]['disp']:.6f}")
print("[9/18] TARGET SEAL OPEN - TEACHER-FORCED SEQUENCE ASSAY")
print("TARGET NOW USED ONLY AS POST-HOC TEACHER-FORCED CONTINUATION")
ROWS=[]
@torch.inference_mode()
def seq_score(q,target,packet=None):
    pids=chat(q)["input_ids"];tids=tok(target,return_tensors="pt",add_special_tokens=False)["input_ids"].to("cuda")
    ids=torch.cat([pids,tids],dim=1);P=pids.shape[1];T=tids.shape[1];o=run_ids(ids,packet);lg=o.logits.float();vals=[]
    for j in range(T):
        tid=int(tids[0,j]);lp=torch.log_softmax(lg[0,P+j-1],dim=-1);vals.append(float(lp[tid]))
    return vals,tids[0].tolist()
for i,(_,q,target) in enumerate(ALL):
    vv,tids=seq_score(q,target,None);ww,_=seq_score(q,target,PACKETS[i]);d=[b-a for a,b in zip(vv,ww)]
    r={"target":target,"anchor":BANK[i][SELECT[i]]["anchor"],"tokens":[tok.decode([x]) for x in tids],"n_tokens":len(tids),"vanilla_logp":vv,"window_logp":ww,"delta_logp":d,"vanilla_seq_logp":float(sum(vv)),"window_seq_logp":float(sum(ww)),"delta_seq":float(sum(d)),"first_delta":float(d[0]),"continuation_delta":float(np.mean(d[1:])) if len(d)>1 else None,"token_improved_frac":float(np.mean(np.array(d)>0)),"all_tokens_improved":bool(all(x>0 for x in d))}
    ROWS.append(r);tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";cont="NA" if r["continuation_delta"] is None else f"{r['continuation_delta']:+.3f}";print(tag,f"| T={target!r} | N={len(tids)} | dSEQ={r['delta_seq']:+.3f} | dFIRST={r['first_delta']:+.3f} | dCONT={cont} | TOK+={r['token_improved_frac']:.2f}")
print("[10/18] PRIMARY SEQUENCE SUMMARY")
M=ROWS[:24];D=np.array([r["delta_seq"] for r in M]);FIRST=np.array([r["first_delta"] for r in M]);MULTI=[r for r in M if r["n_tokens"]>1]
S={"seq_positive":float(np.mean(D>0)),"seq_mean_delta":float(D.mean()),"seq_median_delta":float(np.median(D)),"first_positive":float(np.mean(FIRST>0)),"first_mean_delta":float(FIRST.mean()),"token_improved_frac_mean":float(np.mean([r["token_improved_frac"] for r in M])),"all_tokens_improved":float(np.mean([r["all_tokens_improved"] for r in M])),"vanilla_nll_per_token":float(-sum(sum(r["vanilla_logp"]) for r in M)/sum(r["n_tokens"] for r in M)),"window_nll_per_token":float(-sum(sum(r["window_logp"]) for r in M)/sum(r["n_tokens"] for r in M))}
for k,v in S.items():print(f"{k}={v:.6f}")
print("[11/18] MULTI-TOKEN CONTINUATION")
if MULTI:
    C=np.array([r["continuation_delta"] for r in MULTI]);MSP=float(np.mean([r["delta_seq"]>0 for r in MULTI]));S["multi_n"]=len(MULTI);S["multi_seq_positive"]=MSP;S["continuation_mean_delta"]=float(C.mean());S["continuation_positive"]=float(np.mean(C>0))
    print(f"MULTI_N={len(MULTI)} | SEQ_POS={MSP:.3f} | CONT_MEAN={C.mean():+.4f} | CONT_POS={np.mean(C>0):.3f}")
else:S.update({"multi_n":0,"multi_seq_positive":None,"continuation_mean_delta":None,"continuation_positive":None});print("MULTI_N=0")
print("[12/18] NLL SHIFT")
print(f"NLL/TOKEN {S['vanilla_nll_per_token']:.6f}->{S['window_nll_per_token']:.6f} | DELTA={S['window_nll_per_token']-S['vanilla_nll_per_token']:+.6f}")
print("[13/18] PREDECLARED GATES")
SEQ_GATE=S["seq_positive"]>=.70 and S["seq_mean_delta"]>0
FIRST_GATE=S["first_positive"]>=.70 and S["first_mean_delta"]>0
CONT_GATE=(S["multi_n"]>0 and S["continuation_positive"]>=.70 and S["continuation_mean_delta"]>0)
print(f"SEQ_GATE={SEQ_GATE} | FIRST_GATE={FIRST_GATE} | CONT_GATE={CONT_GATE}")
print("[14/18] SHOWCASE")
G=ROWS[24];print("-"*108);print("SOURCE :",SHOWCASE[0]);print("QUESTION:",SHOWCASE[1]);print("TARGET :",SHOWCASE[2]);print("ANCHOR :",G["anchor"]);print("TOKENS :",G["tokens"]);print(f"dSEQ={G['delta_seq']:+.4f} | dFIRST={G['first_delta']:+.4f} | dTOKENS={[round(x,4) for x in G['delta_logp']]}");print("-"*108)
print("[15/18] CLASSIFICATION")
if SEQ_GATE and FIRST_GATE and CONT_GATE:CLASS="FROZEN_ASSISTANT_WINDOW_STRENGTHENS_FULL_TARGET_SEQUENCE"
elif SEQ_GATE and FIRST_GATE:CLASS="FROZEN_ASSISTANT_WINDOW_STRENGTHENS_SEQUENCE_MAINLY_AT_INITIATION"
elif SEQ_GATE:CLASS="FROZEN_ASSISTANT_WINDOW_SEQUENCE_EFFECT_SUPPORTED"
else:CLASS="FROZEN_ASSISTANT_WINDOW_SEQUENCE_EFFECT_NOT_SUPPORTED"
print("CLASS=",CLASS)
print("[16/18] LEAKAGE AUDIT")
print("SAME TEST342 CASES | REL=-2 PRE-FROZEN | PACKET/ROUTER FROZEN BEFORE TARGET | TARGET ONLY TEACHER-FORCED POST-HOC")
print("[17/18] SCIENTIFIC AUDIT")
print("SAME COMPILER | SAME ARGMAX(DISP) | SAME RSS | SAME ADDITIVE SEASC L0-L25 | SINGLE ASSISTANT-TOKEN WRITE | NO DECODER CHANGE | NO TRAINING | NO WEIGHT UPDATE")
print("[18/18] INTEGRITY + SAVE")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("Hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["err"]>1e-5:raise RuntimeError(f"Dose failure {AUD['err']}")
R={"schema":"akbascore.test343.v1","test":"TEST 343","start":START,"end":utc(),"model":MODEL_ID,"rss":RSS,"frozen_rel":REL,"summary":S,"classification":CLASS,"rows":ROWS,"integrity":{"same_test342_cases":True,"rel_prefrozen":True,"window_search":False,"oracle":False,"target_engine_access":False,"target_teacher_forced_posthoc_only":True,"router":"ARGMAX(DISP)","training":False,"weight_update":False,"hooks_remaining":len(ACTIVE),"fp_start":FP0,"fp_end":fp()}}
sha=hashlib.sha256(canon(R)).hexdigest();rid=f"T343-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json";tp=ROOT/f"{rid}.txt"
jp.write_text(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False),encoding="utf-8")
tp.write_text("\n".join(["="*108,"TEST 343 - FROZEN ASSISTANT-ROLE WINDOW -> FULL TARGET SEQUENCE PROBABILITY","="*108]+[f"{k}={v}" for k,v in S.items()]+[f"CLASS={CLASS}",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]),encoding="utf-8")
print("="*108);print("TEST 343 COMPLETE");print(f"SEQ_POS={S['seq_positive']:.3f} | dSEQ_MEAN={S['seq_mean_delta']:+.3f} | dSEQ_MED={S['seq_median_delta']:+.3f}")
print(f"FIRST_POS={S['first_positive']:.3f} | dFIRST={S['first_mean_delta']:+.3f} | TOKEN_IMP={S['token_improved_frac_mean']:.3f}")
if S["multi_n"]:print(f"CONT_POS={S['continuation_positive']:.3f} | dCONT={S['continuation_mean_delta']:+.3f}")
print(f"NLL/TOKEN={S['vanilla_nll_per_token']:.3f}->{S['window_nll_per_token']:.3f}");print("CLASS=",CLASS)
print("REL=-2 FROZEN | TARGET TEACHER-FORCED POST-HOC ONLY | WEIGHT INTEGRITY PASS | HOOKS=0");print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
