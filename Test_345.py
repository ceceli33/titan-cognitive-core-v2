# ================================================================================================
# AKBASCORE - TEST 345
# FROZEN ASSISTANT-OUTPUT SEASC -> FREE AUTOREGRESSIVE RETRIEVAL
# SAME TEST344 CASES | FULL PROFILE FROZEN | GREEDY GENERATION | NO TEACHER FORCING
# SAME COMPILER | SAME ARGMAX(DISP) | SAME RSS | SOURCE REMOVED | TARGET SEALED UNTIL OUTPUT FROZEN
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
SEED=345;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYS_STD="You are a concise reasoning assistant. Use only the information in the prompt."
SYS_FORCE="Read the answer from the internal state. Return exactly one short answer. Never explain, refuse, say information is missing, or mention the prompt."
TOTAL=28;H=3584;END=25;EPS=1e-8;TARGET_RSS=.250235055;IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20;MAX_NEW=16
ROOT=Path("/content/AKBASCORE_TEST345");ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
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
ALL=CASES+[SHOWCASE]
QNULLS=["What information is requested?","Which unknown detail should be recovered?","Answer the question using the missing information.","What should be identified?","Which detail is being asked about?"]
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
def prompt(q,sysmsg=SYS_FORCE):return tok.apply_chat_template([{"role":"system","content":sysmsg},{"role":"user","content":q}],tokenize=False,add_generation_prompt=True)
def enc(q,sysmsg=SYS_FORCE):return tok(prompt(q,sysmsg),return_tensors="pt",add_special_tokens=False).to("cuda")
def normtxt(x):return re.sub(r"\s+"," ",re.sub(r"[^\w\s]"," ",x.lower(),flags=re.UNICODE)).strip()
print("="*108);print("TEST 345 - FROZEN ASSISTANT-OUTPUT SEASC -> FREE AUTOREGRESSIVE RETRIEVAL");print("="*108);print("START:",START)
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
print("FP:",[f"{x:.4f}" for x in FP0]);print(f"RSS={RSS:.9f} | L0-L25 | FULL ASSISTANT-OUTPUT PROFILE FROZEN | GREEDY")
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
def install(packet,assistant_pos):
    hs=[]
    for L in range(END+1):
        rho=RHO[L]
        def hk(mod,args,out,L=L,rho=rho):
            y=out[0] if isinstance(out,tuple) else out;r=y.clone();n=y.shape[1]
            for idx in range(max(0,assistant_pos),n):
                z=y[:,idx,:].float();zn=z.norm(dim=-1,keepdim=True).clamp_min(EPS);d=packet[L].to(z.device).view(1,-1)*zn*rho;r[:,idx,:]=(z+d).to(y.dtype)
                AUD["err"]=max(AUD["err"],abs(float(d.norm()/zn.squeeze().clamp_min(EPS))-rho))
            return (r,)+out[1:] if isinstance(out,tuple) else r
        h=layers[L].register_forward_hook(hk);hs.append(h);ACTIVE.add(id(h))
    return hs
@torch.inference_mode()
def run_ids(ids,packet=None,assistant_pos=None):
    hs=install(packet,assistant_pos) if packet is not None else []
    try:return model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,output_hidden_states=True,return_dict=True)
    finally:
        for h in hs:h.remove();ACTIVE.discard(id(h))
@torch.inference_mode()
def run(q,packet=None):
    ids=enc(q)["input_ids"];apos=ids.shape[1]-2
    return run_ids(ids,packet,apos) if packet is not None else run_ids(ids)
def term(o):return o.hidden_states[28][0,-1].float()
print("[5/18] FROZEN ARGMAX(DISP) ROUTER - TEST344 METHOD")
SELECT=[];PACKETS=[]
for i,(_,q,_) in enumerate(ALL):
    v=run(q);vh=term(v);ds=[]
    for x in BANK[i]:
        o=run(q,x["packet"]);ds.append(float((term(o)-vh).norm()/vh.norm().clamp_min(EPS)))
    j=int(np.argmax(ds));SELECT.append(j);PACKETS.append(BANK[i][j]["packet"]);tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,f"| ANCHOR={BANK[i][j]['anchor']!r} | DISP={ds[j]:.6f}")
print("[6/18] ROUTER + PROFILE FROZEN")
print("PROFILE=FULL | ASSISTANT->OUTPUT | NO DECAY SEARCH | NO ORACLE | TARGET_ACCESS=False")
print("[7/18] ASSISTANT INDEX AUDIT")
APOS=[]
for i,(_,q,_) in enumerate(ALL):
    ids=enc(q)["input_ids"];p=ids.shape[1]-2;APOS.append(p);tkn=tok.decode([int(ids[0,p])]);tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,f"| POS={p} | TOKEN={tkn!r}")
    if tkn.strip()!="assistant":raise RuntimeError("Assistant index mismatch")
print("[8/18] VANILLA FREE GENERATION - TARGET SEALED")
@torch.inference_mode()
def generate_vanilla(q):
    ids=enc(q)["input_ids"];cur=ids.clone();new=[]
    for _ in range(MAX_NEW):
        o=run_ids(cur);nid=int(torch.argmax(o.logits[0,-1]).item())
        if nid==tok.eos_token_id:break
        new.append(nid);cur=torch.cat([cur,torch.tensor([[nid]],device="cuda")],1)
    return tok.decode(new,skip_special_tokens=True).strip(),new
VOUT=[]
for i,(_,q,_) in enumerate(ALL):
    txt,ids=generate_vanilla(q);VOUT.append(txt);tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,"|",repr(txt))
print("[9/18] SEASC FREE GENERATION - TARGET SEALED")
@torch.inference_mode()
def generate_seasc(q,packet):
    ids=enc(q)["input_ids"];apos=ids.shape[1]-2;cur=ids.clone();new=[]
    for _ in range(MAX_NEW):
        o=run_ids(cur,packet,apos);nid=int(torch.argmax(o.logits[0,-1]).item())
        if nid==tok.eos_token_id:break
        new.append(nid);cur=torch.cat([cur,torch.tensor([[nid]],device="cuda")],1)
    return tok.decode(new,skip_special_tokens=True).strip(),new
SOUT=[]
for i,(_,q,_) in enumerate(ALL):
    txt,ids=generate_seasc(q,PACKETS[i]);SOUT.append(txt);tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,"|",repr(txt))
print("[10/18] OUTPUTS FROZEN - TARGET SEAL OPENS")
ROWS=[]
for i,(s,q,target) in enumerate(ALL):
    nt=normtxt(target);nv=normtxt(VOUT[i]);ns=normtxt(SOUT[i])
    ve=nv==nt;se=ns==nt;vm=bool(nt and nt in nv);sm=bool(nt and nt in ns)
    r={"source":s,"question":q,"target":target,"anchor":BANK[i][SELECT[i]]["anchor"],"vanilla":VOUT[i],"seasc":SOUT[i],"vanilla_exact":ve,"seasc_exact":se,"vanilla_mention":vm,"seasc_mention":sm}
    ROWS.append(r);tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,f"| TARGET={target!r} | V={VOUT[i]!r} | S={SOUT[i]!r} | EXACT {int(ve)}->{int(se)} | MENTION {int(vm)}->{int(sm)}")
print("[11/18] PRIMARY BEHAVIOR SUMMARY")
M=ROWS[:24]
SUMMARY={"vanilla_exact":float(np.mean([r["vanilla_exact"] for r in M])),"seasc_exact":float(np.mean([r["seasc_exact"] for r in M])),"vanilla_mention":float(np.mean([r["vanilla_mention"] for r in M])),"seasc_mention":float(np.mean([r["seasc_mention"] for r in M])),"changed":float(np.mean([r["vanilla"]!=r["seasc"] for r in M]))}
for k,v in SUMMARY.items():print(f"{k}={v:.6f}")
print("[12/18] CASE TRANSITIONS")
GAIN_EXACT=sum((not r["vanilla_exact"]) and r["seasc_exact"] for r in M);LOSS_EXACT=sum(r["vanilla_exact"] and (not r["seasc_exact"]) for r in M)
GAIN_MENTION=sum((not r["vanilla_mention"]) and r["seasc_mention"] for r in M);LOSS_MENTION=sum(r["vanilla_mention"] and (not r["seasc_mention"]) for r in M)
print(f"EXACT_GAIN={GAIN_EXACT} | EXACT_LOSS={LOSS_EXACT} | MENTION_GAIN={GAIN_MENTION} | MENTION_LOSS={LOSS_MENTION}")
print("[13/18] PREDECLARED DECISION GATES")
EXACT_GATE=SUMMARY["seasc_exact"]>SUMMARY["vanilla_exact"] and SUMMARY["seasc_exact"]>=.20
MENTION_GATE=SUMMARY["seasc_mention"]>SUMMARY["vanilla_mention"] and SUMMARY["seasc_mention"]>=.25
SYSTEMATIC_GATE=(GAIN_EXACT>=5 or GAIN_MENTION>=6)
print(f"EXACT_GATE={EXACT_GATE} | MENTION_GATE={MENTION_GATE} | SYSTEMATIC_GATE={SYSTEMATIC_GATE}")
print("[14/18] SHOWCASE")
G=ROWS[24];print("-"*108);print("SOURCE :",G["source"]);print("QUESTION:",G["question"]);print("TARGET :",G["target"]);print("ANCHOR :",G["anchor"]);print("VANILLA:",G["vanilla"]);print("SEASC  :",G["seasc"]);print(f"EXACT {int(G['vanilla_exact'])}->{int(G['seasc_exact'])} | MENTION {int(G['vanilla_mention'])}->{int(G['seasc_mention'])}");print("-"*108)
print("[15/18] CLASSIFICATION")
if EXACT_GATE and SYSTEMATIC_GATE:CLASS="FREE_AUTOREGRESSIVE_RETRIEVAL_SUPPORTED"
elif MENTION_GATE and SYSTEMATIC_GATE:CLASS="FREE_AUTOREGRESSIVE_TARGET_EMERGENCE_SUPPORTED"
elif SUMMARY["changed"]>=.50:CLASS="ASSISTANT_OUTPUT_SEASC_CHANGES_GENERATION_WITHOUT_RELIABLE_RETRIEVAL"
else:CLASS="ASSISTANT_OUTPUT_SEASC_DOES_NOT_UNLOCK_FREE_RETRIEVAL"
print("CLASS=",CLASS)
print("[16/18] LEAKAGE AUDIT")
print("SOURCE ABSENT FROM QUESTION | TARGET SEALED THROUGH COMPILER/ROUTER/GENERATION | TARGET USED ONLY AFTER BOTH OUTPUTS FROZEN")
print("[17/18] SCIENTIFIC AUDIT")
print("MAIN PROJECT | TEST344 FULL PROFILE FROZEN | SAME COMPILER | SAME ARGMAX(DISP) | SAME RSS | GREEDY | NO TEACHER FORCING | NO ORACLE | NO TRAINING | NO WEIGHT UPDATE")
print("[18/18] INTEGRITY + SAVE")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("Hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["err"]>1e-5:raise RuntimeError(f"Dose failure {AUD['err']}")
R={"schema":"akbascore.test345.v1","test":"TEST 345","start":START,"end":utc(),"model":MODEL_ID,"rss":RSS,"profile":"FULL_ASSISTANT_OUTPUT","max_new_tokens":MAX_NEW,"summary":SUMMARY,"classification":CLASS,"rows":ROWS,"integrity":{"source_absent_generation":True,"target_engine_access":False,"target_generation_access":False,"target_posthoc_only":True,"teacher_forcing":False,"oracle":False,"profile_search":False,"training":False,"weight_update":False,"fp_start":FP0,"fp_end":fp(),"hooks_remaining":len(ACTIVE)}}
sha=hashlib.sha256(canon(R)).hexdigest();rid=f"T345-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json";tp=ROOT/f"{rid}.txt"
jp.write_text(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False),encoding="utf-8")
tp.write_text("\n".join(["="*108,"TEST 345 - FREE AUTOREGRESSIVE RETRIEVAL","="*108]+[f"{k}={v}" for k,v in SUMMARY.items()]+[f"EXACT_GAIN={GAIN_EXACT}",f"MENTION_GAIN={GAIN_MENTION}",f"CLASS={CLASS}",f"SHA={sha}"]),encoding="utf-8")
print("="*108);print("TEST 345 COMPLETE");print(f"EXACT  : {SUMMARY['vanilla_exact']:.3f}->{SUMMARY['seasc_exact']:.3f} | GAIN={GAIN_EXACT} LOSS={LOSS_EXACT}")
print(f"MENTION: {SUMMARY['vanilla_mention']:.3f}->{SUMMARY['seasc_mention']:.3f} | GAIN={GAIN_MENTION} LOSS={LOSS_MENTION}")
print(f"CHANGED={SUMMARY['changed']:.3f}");print("CLASS=",CLASS)
print("SOURCE REMOVED | TARGET SEALED THROUGH GENERATION | GREEDY | NO TEACHER FORCING | WEIGHT INTEGRITY PASS | HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
