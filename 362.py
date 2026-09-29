# ================================================================================================
# AKBASCORE - TEST 362
# FRESH HELD-OUT NATIVE ATTENTION VALUE-CONTRIBUTION REPLICATION
# FROZEN TEST361: QUESTION-LAST -> SOURCE SPAN (ATTN*V) -> O_PROJ -> 28L MRR -> VALUE POST-HOC
# NO SEASC | NO TARGET ROUTING | NO LAYER/HEAD ORACLE | NO SCORE SEARCH | NO TRAINING
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
SEED=362;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";TOTAL=28;H=3584
ROOT=Path("/content/AKBASCORE_TEST362");ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
CASES=[
("PERSON","After the diagnostic sequence, operator Dervan isolated the backup manifold near the service bay.","Who isolated the backup manifold?","Dervan"),
("PERSON","The final stability report was approved by Elvara before the equipment was released.","Who approved the final stability report?","Elvara"),
("PLACE","The recovered telemetry module was transferred to Marseille after the vessel returned.","Where was the recovered telemetry module transferred?","Marseille"),
("PLACE","The engraved plate was deposited in Helsinki before the archive was sealed.","Where was the engraved plate deposited?","Helsinki"),
("COLOR","Following the reset procedure, the diagnostic lamp changed to amber.","What color did the diagnostic lamp become?","amber"),
("COLOR","The optical scan identified the protective coating as turquoise.","What color was the protective coating identified as?","turquoise"),
("MATERIAL","The engineering team selected ceramic for the thermal shield after evaluation.","Which material was selected for the thermal shield?","ceramic"),
("MATERIAL","The laboratory chose nickel for the internal bracket after load testing.","Which material was chosen for the internal bracket?","nickel"),
("ACTION","Near the lower platform, Merin began descending toward the maintenance tunnel.","What did Merin begin doing?","descending"),
("ACTION","When the locking mechanism disengaged, the inner ring began spinning.","What did the inner ring begin doing?","spinning"),
("NAME","The newly catalogued alloy received the designation Arvex.","What designation was given to the newly catalogued alloy?","Arvex"),
("NAME","The experimental platform's final name was changed to Tervane.","What became the experimental platform's final name?","Tervane"),
("NUMBER","After independent verification, the accepted measurement was 735.","What was the accepted measurement?","735"),
("OBJECT","From the navigation case, Leris retrieved a barometer.","What object did Leris retrieve?","barometer"),
("PROPERTY","After processing, the reference surface was described as porous.","How was the reference surface described?","porous"),
("SHOWCASE","Mustafa Akbaş suspended a blue ribbon beside the Dolmabahçe Clock Tower where the public demonstration was conducted.","What did Mustafa Akbaş do at the location of the public demonstration?","suspended a blue ribbon")]
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def py(x):
    if isinstance(x,dict):return {str(k):py(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [py(v) for v in x]
    if isinstance(x,np.ndarray):return py(x.tolist())
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.floating):return float(x)
    if isinstance(x,np.bool_):return bool(x)
    return x
def canon(x):return json.dumps(py(x),sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
def rank_desc(x,j):return int(1+sum(float(v)>float(x[j]) for v in x))
print("="*108);print("TEST 362 - FRESH HELD-OUT NATIVE ATTENTION VALUE-CONTRIBUTION REPLICATION");print("="*108);print("START:",START)
print("[1/20] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="eager",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers
if len(layers)!=TOTAL or model.config.hidden_size!=H:raise RuntimeError("Architecture mismatch")
torch.cuda.synchronize();print(f"OK | {MODEL_ID} | 28L | H={H} | eager-attention | {next(model.parameters()).dtype} | {time.perf_counter()-t:.2f}s")
print("[2/20] MODEL LOCK")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();print("FP:",[f"{x:.4f}" for x in FP0])
print("[3/20] ARCHITECTURE AUDIT")
NH=int(model.config.num_attention_heads);NKV=int(model.config.num_key_value_heads);HD=int(getattr(model.config,"head_dim",H//NH));REP=NH//NKV
if NH%NKV:raise RuntimeError("GQA mismatch")
print(f"HEADS={NH} | KV_HEADS={NKV} | HEAD_DIM={HD} | KV_REPEAT={REP}")
print("[4/20] FRESH HELD-OUT ENCODING")
DATA=[]
for i,(typ,s,q,_) in enumerate(CASES):
    text="SOURCE:\n"+s+"\n\nQUESTION:\n"+q+"\n\nANSWER:"
    e=tok(text,return_tensors="pt",add_special_tokens=False,return_offsets_mapping=True);off=e["offset_mapping"][0].tolist();src0=len("SOURCE:\n");q0=text.index(q);q1=q0+len(q)
    qt=[j for j,(a,b) in enumerate(off) if b>q0 and a<q1];words=[]
    for m in re.finditer(r"\b[\w'-]+\b",s,re.UNICODE):
        a,b=src0+m.start(),src0+m.end();ix=[j for j,(x,y) in enumerate(off) if y>a and x<b]
        if ix:words.append({"text":m.group(0),"tok":[int(x) for x in ix]})
    if not qt or not words:raise RuntimeError("Span failure")
    DATA.append({"ids":e["input_ids"].to("cuda"),"mask":e["attention_mask"].to("cuda"),"words":words,"q":int(qt[-1])})
    print(f"{i+1:02d}/16 | {typ:8s} | SPANS={len(words)} | Q_LAST={qt[-1]}")
print("[5/20] FORWARDS + HIDDEN STATES + ATTENTION")
OUT=[]
with torch.inference_mode():
    for i,d in enumerate(DATA):
        o=model(input_ids=d["ids"],attention_mask=d["mask"],use_cache=False,output_attentions=True,output_hidden_states=True,return_dict=True)
        if o.attentions is None or len(o.attentions)!=TOTAL or len(o.hidden_states)!=TOTAL+1:raise RuntimeError("Forward outputs unavailable")
        OUT.append(o);print(f"{i+1:02d}/16 | OK")
print("[6/20] FROZEN TEST361 V PROJECTION")
VAL=[]
with torch.inference_mode():
    for ci,d in enumerate(DATA):
        lv=[]
        for L in range(TOTAL):
            att=layers[L].self_attn;h=OUT[ci].hidden_states[L]
            if hasattr(layers[L],"input_layernorm"):h=layers[L].input_layernorm(h)
            v=att.v_proj(h).float().view(1,h.shape[1],NKV,HD).transpose(1,2)[0]
            if REP>1:v=v.repeat_interleave(REP,dim=0)
            lv.append(v.cpu())
        VAL.append(lv)
print("V PROJECTIONS READY")
print("[7/20] FROZEN ATTN*V SOURCE-SPAN CONTRIBUTIONS")
HEADCON=[]
for ci,d in enumerate(DATA):
    case=[]
    for L in range(TOTAL):
        a=OUT[ci].attentions[L][0].float().cpu();v=VAL[ci][L];q=d["q"];sp=[]
        for w in d["words"]:
            ix=w["tok"];sp.append((a[:,q,ix].unsqueeze(-1)*v[:,ix,:]).sum(dim=1))
        case.append(sp)
    HEADCON.append(case)
print("ATTN*V READY")
print("[8/20] FROZEN O_PROJ CONTRIBUTIONS")
CONTR=[]
with torch.inference_mode():
    for ci,d in enumerate(DATA):
        case=[]
        for L in range(TOTAL):
            W=layers[L].self_attn.o_proj.weight.detach().float().cpu();sp=[]
            for hc in HEADCON[ci][L]:
                z=hc.reshape(-1)
                if z.numel()!=W.shape[1]:raise RuntimeError(f"O_PROJ shape mismatch L{L}")
                sp.append(torch.mv(W,z))
            case.append(sp)
        CONTR.append(case)
print("O_PROJ CONTRIBUTIONS READY")
print("[9/20] PER-LAYER CONTRIBUTION NORMS")
CN=[[[float(CONTR[i][L][j].norm()) for j in range(len(DATA[i]["words"]))] for L in range(TOTAL)] for i in range(len(DATA))]
print("NORMS READY")
print("[10/20] PER-LAYER CONTRIBUTION RANKS")
CR=[[[rank_desc(CN[i][L],j) for j in range(len(DATA[i]["words"]))] for L in range(TOTAL)] for i in range(len(DATA))]
print("RANKS READY")
print("[11/20] PRIMARY FROZEN TEST361 MRR")
CMRR=[[float(np.mean([1.0/float(CR[i][L][j]) for L in range(TOTAL)])) for j in range(len(DATA[i]["words"]))] for i in range(len(DATA))]
print("PRIMARY=MEAN_L(1/CONTRIBUTION_RANK) | TEST361 RULE FROZEN")
print("[12/20] SECONDARY FROZEN TOP1 VOTE")
CV1=[[int(sum(CR[i][L][j]==1 for L in range(TOTAL))) for j in range(len(DATA[i]["words"]))] for i in range(len(DATA))]
print("TOP1 VOTE READY")
print("[13/20] RAW ATTENTION MRR CONTROL")
AR=[];AMRR=[]
for ci,d in enumerate(DATA):
    rr=[]
    for L in range(TOTAL):
        a=OUT[ci].attentions[L][0].float().cpu();q=d["q"];sc=[float(a[:,q,w["tok"]].mean()) for w in d["words"]];rr.append([rank_desc(sc,j) for j in range(len(sc))])
    AR.append(rr);AMRR.append([float(np.mean([1.0/float(rr[L][j]) for L in range(TOTAL)])) for j in range(len(d["words"]))])
print("RAW CONTROL READY")
print("[14/20] ALL TARGET-FREE OUTPUTS FROZEN")
for i,d in enumerate(DATA):
    j=int(np.argmax(np.asarray(CMRR[i],dtype=np.float64)));print(f"{i+1:02d}/16 | TOP_CONTR={d['words'][j]['text']!r} | CMRR={CMRR[i][j]:.4f} | V1={CV1[i][j]:02d} | AMRR={AMRR[i][j]:.4f}")
print("[15/20] VALUE OPENS POST-HOC")
ROWS=[]
for i,(typ,s,q,value) in enumerate(CASES):
    d=DATA[i];vw=re.findall(r"\b[\w'-]+\b",value,re.UNICODE);idx=None
    for j in range(len(d["words"])-len(vw)+1):
        if [x["text"].lower() for x in d["words"][j:j+len(vw)]]==[x.lower() for x in vw]:idx=int(j);break
    if idx is None:raise RuntimeError(f"Value not found: {value}")
    rc=rank_desc(CMRR[i],idx);rv=rank_desc(CV1[i],idx);ra=rank_desc(AMRR[i],idx);lr=[int(CR[i][L][idx]) for L in range(TOTAL)]
    row={"i":int(i+1),"type":str(typ),"value":str(value),"n":int(len(d["words"])),"contribution_mrr_rank":int(rc),"contribution_top1vote_rank":int(rv),"attention_mrr_rank":int(ra),"contribution_top1_layers":int(sum(x==1 for x in lr)),"contribution_top3_layers":int(sum(x<=3 for x in lr))};ROWS.append(row)
    print(f"{i+1:02d}/16 | {typ:8s} | VALUE={value!r} | CONTR=R{rc:02d} V1=R{rv:02d} ATTN=R{ra:02d} | HIT1={row['contribution_top1_layers']:02d} HIT3={row['contribution_top3_layers']:02d}")
print("[16/20] FRESH HELD-OUT SUMMARY")
D=ROWS[:15]
def stats(k):
    r=np.asarray([int(x[k]) for x in D],dtype=np.int64);return {"median":float(np.median(r)),"top1":float(np.mean(r==1)),"top3":float(np.mean(r<=3)),"top5":float(np.mean(r<=5))}
SC=stats("contribution_mrr_rank");SV=stats("contribution_top1vote_rank");SA=stats("attention_mrr_rank")
print(f"CONTR MRR | MED={SC['median']:.1f} | TOP1={SC['top1']:.3f} | TOP3={SC['top3']:.3f} | TOP5={SC['top5']:.3f}")
print(f"CONTR V1  | MED={SV['median']:.1f} | TOP1={SV['top1']:.3f} | TOP3={SV['top3']:.3f} | TOP5={SV['top5']:.3f}")
print(f"ATTN MRR  | MED={SA['median']:.1f} | TOP1={SA['top1']:.3f} | TOP3={SA['top3']:.3f} | TOP5={SA['top5']:.3f}")
print("[17/20] PREDECLARED REPLICATION GATES")
better=float(np.mean([x["contribution_mrr_rank"]<x["attention_mrr_rank"] for x in D]));worse=float(np.mean([x["contribution_mrr_rank"]>x["attention_mrr_rank"] for x in D]))
REPLICATED=bool(SC["median"]<=2 and SC["top1"]>=.50 and SC["top3"]>=.80 and SC["top5"]>=.90 and better>=.60)
STRONG=bool(SC["median"]<=1 and SC["top1"]>=.70 and SC["top3"]>=.85 and SC["top5"]>=.95 and better>=.70)
print(f"CONTR_BEATS_ATTN={better:.3f} | CONTR_WORSE_ATTN={worse:.3f} | REPLICATED={REPLICATED} | STRONG={STRONG}")
print("[18/20] SHOWCASE + CLASSIFICATION")
g=ROWS[15];print("-"*108);print("SOURCE:",CASES[15][1]);print("Q:",CASES[15][2]);print("VALUE:",CASES[15][3]);print(f"CONTR_MRR=R{g['contribution_mrr_rank']} | CONTR_V1=R{g['contribution_top1vote_rank']} | ATTN_MRR=R{g['attention_mrr_rank']} | HIT1={g['contribution_top1_layers']} HIT3={g['contribution_top3_layers']}");print("-"*108)
if STRONG:CLASS="FROZEN_NATIVE_VALUE_CONTRIBUTION_STRONGLY_REPLICATED"
elif REPLICATED:CLASS="FROZEN_NATIVE_VALUE_CONTRIBUTION_REPLICATED"
elif SC["top3"]>SA["top3"]:CLASS="VALUE_CONTRIBUTION_IMPROVES_BUT_REPLICATION_GATE_NOT_MET"
else:CLASS="NATIVE_VALUE_CONTRIBUTION_NOT_REPLICATED"
print("CLASS=",CLASS)
print("[19/20] SCIENTIFIC AUDIT")
print("FRESH HELD-OUT CASES | TEST361 FORMULA FROZEN | QUESTION-LAST | EXACT V_PROJ + ATTN + O_PROJ | PRIMARY=28L CONTRIBUTION MRR | VALUE POST-HOC | NO SEASC | NO TARGET/LAYER/HEAD ORACLE | NO SCORE SEARCH | NO TRAINING")
print("[20/20] INTEGRITY + JSON-SAFE SAVE")
FP1=fp()
if FP1!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
S={"n":15,"primary":"frozen_test361_cross_layer_contribution_mrr","contribution":SC,"contribution_top1vote":SV,"attention_control":SA,"contribution_beats_attention":float(better),"contribution_worse_attention":float(worse),"replicated":bool(REPLICATED),"strong":bool(STRONG)}
R=py({"schema":"akbascore.test362.v1","test":"TEST 362","start":START,"end":utc(),"model":MODEL_ID,"summary":S,"classification":CLASS,"rows":ROWS,"integrity":{"fresh_heldout":True,"test361_formula_frozen":True,"target_engine_access":False,"value_posthoc_only":True,"layer_oracle":False,"head_oracle":False,"score_search":False,"seasc":False,"training":False,"weight_update":False,"fp_start":[float(x) for x in FP0],"fp_end":[float(x) for x in FP1]}})
sha=hashlib.sha256(canon(R)).hexdigest();rid=f"T362-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json";tp=ROOT/f"{rid}.txt"
jp.write_text(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False),encoding="utf-8")
tp.write_text("\n".join(["="*108,"TEST 362 - FRESH HELD-OUT NATIVE ATTENTION VALUE-CONTRIBUTION REPLICATION","="*108,f"N=15",f"CONTR_MEDIAN={SC['median']}",f"CONTR_TOP1={SC['top1']}",f"CONTR_TOP3={SC['top3']}",f"CONTR_TOP5={SC['top5']}",f"ATTN_TOP3={SA['top3']}",f"REPLICATED={REPLICATED}",f"STRONG={STRONG}",f"CLASS={CLASS}",f"SHA={sha}"]),encoding="utf-8")
print("="*108);print("TEST 362 COMPLETE");print(f"CONTR MRR | MED={SC['median']:.1f} | TOP1={SC['top1']:.3f} | TOP3={SC['top3']:.3f} | TOP5={SC['top5']:.3f}");print(f"ATTN  MRR | MED={SA['median']:.1f} | TOP1={SA['top1']:.3f} | TOP3={SA['top3']:.3f} | TOP5={SA['top5']:.3f}");print("CLASS=",CLASS);print("VALUE POST-HOC ONLY | TEST361 FORMULA FROZEN | WEIGHTS FROZEN | JSON SAFE");print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
