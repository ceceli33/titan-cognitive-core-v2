# ================================================================================================
# AKBASCORE - TEST 349
# PACKET-TO-LEXICON GEOMETRIC READOUT ASSAY
# FROZEN TEST348 ENGINE | TERMINAL REALIZED DISPLACEMENT x LM_HEAD VOCAB GEOMETRY
# PRIMARY=COS(DELTA_H27,W_TOKEN) | TARGET POST-HOC ONLY | NO ENGINE CHANGE
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
SEED=349;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYS_STD="You are a concise reasoning assistant. Use only the information in the prompt."
SYS_FORCE="Read the answer from the internal state. Return exactly one short answer. Never explain, refuse, say information is missing, or mention the prompt."
TOTAL=28;H=3584;END=25;EPS=1e-8;TARGET_RSS=.250235055;IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20;HYB=.25
ROOT=Path("/content/AKBASCORE_TEST349");ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
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
def rank(x,j):return int((x>x[j]).sum().item()+1)
def tops(x,k=8):
    z=torch.topk(x,k).indices.tolist();return [{"id":int(j),"token":tok.decode([int(j)]),"score":float(x[j])} for j in z]
print("="*108);print("TEST 349 - PACKET-TO-LEXICON GEOMETRIC READOUT ASSAY");print("="*108);print("START:",START)
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
print("FP:",[f"{x:.4f}" for x in FP0]);print(f"RSS={RSS:.9f} | FULL ASSISTANT-OUTPUT | PRIMARY=COS(REALIZED_DH27,W) | HYB={HYB}")
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
@torch.inference_mode()
def run(q,packet=None):
    ids=enc(q)["input_ids"];apos=ids.shape[1]-2
    return run_ids(ids,packet,apos) if packet is not None else run_ids(ids)
def term(o):return o.hidden_states[28][0,-1].float()
print("[5/18] FROZEN ARGMAX(DISP) ROUTER")
SELECT=[];PACKETS=[]
for i,(_,q,_) in enumerate(ALL):
    v=run(q);vh=term(v);ds=[]
    for x in BANK[i]:
        o=run(q,x["packet"]);ds.append(float((term(o)-vh).norm()/vh.norm().clamp_min(EPS)))
    j=int(np.argmax(ds));SELECT.append(j);PACKETS.append(BANK[i][j]["packet"]);tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,f"| ANCHOR={BANK[i][j]['anchor']!r} | DISP={ds[j]:.6f}")
print("[6/18] VOCAB GEOMETRY LOCK")
W=model.lm_head.weight.detach().float();WN=W/W.norm(dim=1,keepdim=True).clamp_min(EPS);print(f"VOCAB={W.shape[0]} | W={tuple(W.shape)} | NORMALIZED=True")
print("[7/18] ASSISTANT INDEX AUDIT")
APOS=[]
for i,(_,q,_) in enumerate(ALL):
    ids=enc(q)["input_ids"];p=ids.shape[1]-2;APOS.append(p);x=tok.decode([int(ids[0,p])]);tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,f"| POS={p} | TOKEN={x!r}")
    if x.strip()!="assistant":raise RuntimeError("Assistant index mismatch")
print("[8/18] TARGET-FREE TERMINAL GEOMETRY + VOCAB SCORES")
FROZEN=[]
for i,(_,q,_) in enumerate(ALL):
    ids=enc(q)["input_ids"];v=run_ids(ids);s=run_ids(ids,PACKETS[i],APOS[i]);hv=term(v);hs=term(s);dh=unit(hs-hv);geo=torch.mv(WN,dh).cpu();V=v.logits[0,-1].float().cpu();S=s.logits[0,-1].float().cpu();D=S-V
    pos=(D-D.mean())/D.std().clamp_min(EPS);hyb=geo+HYB*pos;FROZEN.append({"geo":geo,"hyb":hyb,"S":S,"D":D,"disp":float((hs-hv).norm()/hv.norm().clamp_min(EPS)),"geo_top":tops(geo),"hyb_top":tops(hyb)})
    tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,f"| DISP={FROZEN[-1]['disp']:.6f} | GEO1={FROZEN[-1]['geo_top'][0]['token']!r} | HYB1={FROZEN[-1]['hyb_top'][0]['token']!r}")
print("[9/18] ALL TARGET-FREE RANKINGS FROZEN")
print("PRIMARY GEO + SECONDARY HYB FROZEN | TARGET_ACCESS=False")
print("[10/18] TARGET OPENS - POST-HOC READOUT")
ROWS=[]
for i,(_,q,target) in enumerate(ALL):
    tid=int(tok(target,add_special_tokens=False)["input_ids"][0]);f=FROZEN[i];rg=rank(f["geo"],tid);rh=rank(f["hyb"],tid);rs=rank(f["S"],tid);rd=rank(f["D"],tid)
    r={"target":target,"target_id":tid,"target_token":tok.decode([tid]),"anchor":BANK[i][SELECT[i]]["anchor"],"disp":f["disp"],"geo_rank":rg,"hyb_rank":rh,"steer_rank":rs,"delta_rank":rd,"geo_score":float(f["geo"][tid]),"delta":float(f["D"][tid]),"geo_top":f["geo_top"],"hyb_top":f["hyb_top"]};ROWS.append(r)
    tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,f"| T={r['target_token']!r} | GEO={rg} HYB={rh} STEER={rs} DELTA={rd} | COS={r['geo_score']:+.5f}")
print("[11/18] SUMMARY")
M=ROWS[:24]
def sm(k):
    a=np.array([r[k] for r in M]);return {"median":float(np.median(a)),"top1":float(np.mean(a<=1)),"top5":float(np.mean(a<=5)),"top10":float(np.mean(a<=10)),"top50":float(np.mean(a<=50)),"top256":float(np.mean(a<=256))}
SG,SH,SS,SD=sm("geo_rank"),sm("hyb_rank"),sm("steer_rank"),sm("delta_rank")
for n,x in [("GEO",SG),("HYB",SH),("STEER",SS),("DELTA",SD)]:print(f"{n:6s} | MED={x['median']:.1f} | T1={x['top1']:.3f} T5={x['top5']:.3f} T10={x['top10']:.3f} T50={x['top50']:.3f} T256={x['top256']:.3f}")
print("[12/18] GEOMETRIC IDENTITY TEST")
GEO_POS=float(np.mean([r["geo_score"]>0 for r in M]));GEO_BEATS_DELTA=float(np.mean([r["geo_rank"]<r["delta_rank"] for r in M]));GEO_BEATS_STEER=float(np.mean([r["geo_rank"]<r["steer_rank"] for r in M]));HYB_BEATS_STEER=float(np.mean([r["hyb_rank"]<r["steer_rank"] for r in M]))
print(f"GEO_COS_POS={GEO_POS:.6f} | GEO<DELTA={GEO_BEATS_DELTA:.6f} | GEO<STEER={GEO_BEATS_STEER:.6f} | HYB<STEER={HYB_BEATS_STEER:.6f}")
print("[13/18] PREDECLARED GATES")
GEO_SIGNAL=GEO_POS>=.70
GEO_LOCALIZE=SG["top256"]>=.25 and SG["median"]<SD["median"]
GEO_STRONG=SG["top50"]>=.20 and SG["median"]<SS["median"]
HYB_HELP=SH["median"]<SS["median"] and HYB_BEATS_STEER>=.60
print(f"GEO_SIGNAL={GEO_SIGNAL} | GEO_LOCALIZE={GEO_LOCALIZE} | GEO_STRONG={GEO_STRONG} | HYB_HELP={HYB_HELP}")
print("[14/18] SHOWCASE")
G=ROWS[24];print("-"*108);print("SOURCE :",SHOWCASE[0]);print("QUESTION:",SHOWCASE[1]);print("TARGET :",G["target"],"| TOKEN:",repr(G["target_token"]));print("ANCHOR :",G["anchor"]);print(f"GEO={G['geo_rank']} HYB={G['hyb_rank']} STEER={G['steer_rank']} DELTA={G['delta_rank']} | COS={G['geo_score']:+.5f}")
print("TOP GEO:",[(x["token"],round(x["score"],5)) for x in G["geo_top"]]);print("TOP HYB:",[(x["token"],round(x["score"],3)) for x in G["hyb_top"]]);print("-"*108)
print("[15/18] CLASSIFICATION")
if GEO_SIGNAL and GEO_STRONG:CLASS="PACKET_GEOMETRY_CONTAINS_TARGET_FREE_LEXICAL_IDENTITY_SIGNAL"
elif GEO_SIGNAL and GEO_LOCALIZE:CLASS="PACKET_GEOMETRY_WEAKLY_LOCALIZES_LEXICAL_TARGET"
elif GEO_SIGNAL:CLASS="PACKET_GEOMETRY_ALIGNS_WITH_TARGET_BUT_DOES_NOT_LOCALIZE_LEXICAL_IDENTITY"
else:CLASS="PACKET_GEOMETRY_DOES_NOT_SUPPORT_DIRECT_LEXICAL_IDENTITY_READOUT"
print("CLASS=",CLASS)
print("[16/18] INTERPRETATION LOCK")
print("REALIZED TERMINAL DISPLACEMENT ONLY | LM_HEAD ROW COSINE | TARGET POST-HOC ONLY | GEOMETRY IS DIAGNOSTIC, NOT A TRAINED DECODER")
print("[17/18] SCIENTIFIC AUDIT")
print("TEST348 ENGINE UNCHANGED | SAME COMPILER | SAME ARGMAX(DISP) | SAME RSS | FULL ASSISTANT-OUTPUT | NO SCORE SEARCH | NO ORACLE | NO TRAINING | NO WEIGHT UPDATE")
print("[18/18] INTEGRITY + SAVE")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("Hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["err"]>1e-5:raise RuntimeError(f"Dose failure {AUD['err']}")
S={"geo":SG,"hyb":SH,"steer":SS,"delta":SD,"geo_cos_positive":GEO_POS,"geo_beats_delta":GEO_BEATS_DELTA,"geo_beats_steer":GEO_BEATS_STEER,"hyb_beats_steer":HYB_BEATS_STEER}
R={"schema":"akbascore.test349.v1","test":"TEST 349","start":START,"end":utc(),"model":MODEL_ID,"rss":RSS,"primary":"COS_REALIZED_DH27_LMHEAD","hyb":HYB,"summary":S,"classification":CLASS,"rows":ROWS,"integrity":{"engine":"TEST348_FULL_ASSISTANT_OUTPUT","target_engine_access":False,"target_score_access":False,"target_posthoc_only":True,"score_search":False,"oracle":False,"training":False,"weight_update":False,"fp_start":FP0,"fp_end":fp(),"hooks_remaining":len(ACTIVE)}}
sha=hashlib.sha256(canon(R)).hexdigest();rid=f"T349-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json";tp=ROOT/f"{rid}.txt"
jp.write_text(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False),encoding="utf-8")
tp.write_text("\n".join(["="*108,"TEST 349 - PACKET-TO-LEXICON GEOMETRIC READOUT ASSAY","="*108,f"GEO={SG}",f"HYB={SH}",f"STEER={SS}",f"DELTA={SD}",f"GEO_COS_POS={GEO_POS}",f"CLASS={CLASS}",f"SHA={sha}"]),encoding="utf-8")
print("="*108);print("TEST 349 COMPLETE")
for n,x in [("GEO",SG),("HYB",SH),("STEER",SS),("DELTA",SD)]:print(f"{n:6s} | MED={x['median']:.1f} | T10={x['top10']:.3f} | T50={x['top50']:.3f} | T256={x['top256']:.3f}")
print(f"GEO_COS_POS={GEO_POS:.3f} | GEO<DELTA={GEO_BEATS_DELTA:.3f} | GEO<STEER={GEO_BEATS_STEER:.3f} | HYB<STEER={HYB_BEATS_STEER:.3f}")
print("CLASS=",CLASS);print("TARGET POST-HOC ONLY | WEIGHTS FROZEN | HOOKS=0");print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
