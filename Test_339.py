# ================================================================================================
# AKBASCORE - TEST 339
# SYNTHETIC KV MEMORY SLOT ASSAY
# SOURCE -> NATIVE K/V MEMORY | SOURCE REMOVED | QUESTION -> QK ATTENTION -> VALUE TRANSPORT
# TARGET SEALED THROUGH MEMORY CONSTRUCTION + RETRIEVAL | NO TRAINING | NO WEIGHT UPDATE
# ================================================================================================
import os,sys,json,time,random,hashlib,subprocess,importlib.util,re,math
from datetime import datetime,timezone
from pathlib import Path
for m,p in [("torch","torch"),("transformers","transformers"),("numpy","numpy")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required")
SEED=339;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYS_STD="You are a concise reasoning assistant. Use only the information in the prompt."
SYS_FORCE="Read the answer from the internal state. Return exactly one short answer. Never explain, refuse, say information is missing, or mention the prompt."
TOTAL=28;H_EXPECT=3584;EPS=1e-8;LAYERS=list(range(28));TOPH=8
ROOT=Path("/content/AKBASCORE_TEST339");ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
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
ALL=CASES+[SHOWCASE];SEALED=tuple(x[2] for x in ALL)
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
def chat(text,sysmsg):
    s=tok.apply_chat_template([{"role":"system","content":sysmsg},{"role":"user","content":text}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to("cuda")
print("="*108);print("TEST 339 - SYNTHETIC KV MEMORY SLOT ASSAY");print("="*108);print("START:",START)
print("[1/16] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers
if len(layers)!=TOTAL or model.config.hidden_size!=H_EXPECT:raise RuntimeError("Architecture mismatch")
CFG=model.config;NH=int(CFG.num_attention_heads);NKV=int(CFG.num_key_value_heads);HD=int(getattr(CFG,"head_dim",H_EXPECT//NH));GROUP=NH//NKV
torch.cuda.synchronize();print(f"OK | {MODEL_ID} | 28L | H={H_EXPECT} | QH={NH} | KVH={NKV} | HD={HD} | GROUP={GROUP} | {time.perf_counter()-t:.2f}s")
print("[2/16] WEIGHT + ARCHITECTURE LOCK")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();print("FP:",[f"{x:.4f}" for x in FP0]);print("TARGET SEALED | SOURCE-DERIVED K/V ONLY | NO TARGET TOKEN USED")
print("[3/16] SOURCE TOKEN SPANS")
def lexical(text):
    e=tok(text,return_tensors="pt",add_special_tokens=False,return_offsets_mapping=True);offs=e["offset_mapping"][0].tolist();out=[]
    for m in re.finditer(r"\b[\w'-]+\b",text,re.UNICODE):
        a,b=m.span();ix=[j for j,(s,z) in enumerate(offs) if z>a and s<b]
        if ix:out.append((m.group(0),ix))
    return e["input_ids"].to("cuda"),out
print("READY")
print("[4/16] SOURCE -> NATIVE K/V MEMORY EXTRACTION")
@torch.inference_mode()
def source_memory(text):
    ids,lex=lexical(text);mask=torch.ones_like(ids);cache={}
    def mk(L):
        def hk(mod,args):
            x=args[0];n=mod.input_layernorm(x) if hasattr(mod,"input_layernorm") else x
        return hk
    captures={}
    hs=[]
    for L in LAYERS:
        def pre(mod,args,L=L):
            captures[L]=args[0].detach()
        hs.append(layers[L].register_forward_pre_hook(pre))
    try:model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
    finally:
        for h in hs:h.remove()
    mem={}
    for L in LAYERS:
        x=layers[L].input_layernorm(captures[L]).float()
        K=layers[L].self_attn.k_proj(x.to(layers[L].self_attn.k_proj.weight.dtype)).float().view(1,x.shape[1],NKV,HD)[0]
        V=layers[L].self_attn.v_proj(x.to(layers[L].self_attn.v_proj.weight.dtype)).float().view(1,x.shape[1],NKV,HD)[0]
        mem[L]={"K":K.mean(0).detach(),"V":V.mean(0).detach()}
    return mem
MEM=[]
for i,(s,_,_) in enumerate(ALL):
    MEM.append(source_memory(s));tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,"| SOURCE K/V FROZEN")
print("[5/16] SOURCE REMOVED")
print("ALL SOURCE FORWARDS COMPLETE | QUESTION RUNTIME CONTAINS NO SOURCE TEXT")
print("[6/16] QUESTION Q EXTRACTION + NATIVE MEMORY ADDRESSING")
@torch.inference_mode()
def query_assay(q,mem):
    inp=chat(q,SYS_FORCE);captures={};hs=[]
    for L in LAYERS:
        def pre(mod,args,L=L):captures[L]=args[0].detach()
        hs.append(layers[L].register_forward_pre_hook(pre))
    try:o=model(**inp,use_cache=False,output_hidden_states=True,return_dict=True)
    finally:
        for h in hs:h.remove()
    rows=[]
    for L in LAYERS:
        x=layers[L].input_layernorm(captures[L][:,-1:,:]).float()
        Q=layers[L].self_attn.q_proj(x.to(layers[L].self_attn.q_proj.weight.dtype)).float().view(NH,HD)
        K=mem[L]["K"];V=mem[L]["V"];Kr=K.repeat_interleave(GROUP,dim=0)
        score=(Q*Kr).sum(-1)/math.sqrt(HD);top=torch.topk(score,min(TOPH,NH))
        value=V.repeat_interleave(GROUP,dim=0)
        selected=value[top.indices];transport=selected.mean(0)
        rows.append({"score_mean":float(score.mean()),"score_max":float(score.max()),"score_std":float(score.std()),"top_heads":top.indices.cpu().tolist(),"top_scores":top.values.cpu().tolist(),"transport_norm":float(transport.norm())})
    return o,rows
ASSAY=[];VAN=[]
for i,(_,q,_) in enumerate(ALL):
    o,r=query_assay(q,MEM[i]);VAN.append(o);ASSAY.append(r);tag="SHOWCASE" if i==24 else f"{i+1:02d}/24"
    best=max(range(TOTAL),key=lambda L:r[L]["score_max"]);print(tag,f"| BEST_L={best:02d} | QK_MAX={r[best]['score_max']:+.4f} | HEADS={r[best]['top_heads'][:4]}")
print("[7/16] CROSS-CASE ADDRESS CONTROL")
CROSS=[]
for i,(_,q,_) in enumerate(ALL[:24]):
    own=max(r["score_max"] for r in ASSAY[i]);wrong=[]
    for j in range(24):
        if j==i:continue
        _,r=query_assay(q,MEM[j]);wrong.append(max(x["score_max"] for x in r))
    CROSS.append({"own":own,"wrong_mean":float(np.mean(wrong)),"wrong_max":float(np.max(wrong)),"margin_mean":float(own-np.mean(wrong))})
    print(f"{i+1:02d}/24 | OWN={own:+.4f} | WRONG_MEAN={np.mean(wrong):+.4f} | MARGIN={own-np.mean(wrong):+.4f}")
print("[8/16] FROZEN BEST-LAYER MEMORY WRITE")
BEST=[max(range(TOTAL),key=lambda L:ASSAY[i][L]["score_max"]) for i in range(25)]
ACTIVE=set()
def install_value(mem,L,heads):
    att=layers[L].self_attn;orig=att.o_proj.register_forward_pre_hook
    def pre(mod,args):
        x=args[0];z=x[:,-1,:].float().view(1,NH,HD);V=mem[L]["V"].repeat_interleave(GROUP,dim=0).to(z.device)
        idx=torch.tensor(heads,device=z.device);z[:,idx,:]=z[:,idx,:]+V[idx].unsqueeze(0)
        y=x.clone();y[:,-1,:]=z.reshape(1,-1).to(x.dtype);return (y,)
    h=att.o_proj.register_forward_pre_hook(pre);ACTIVE.add(id(h));return h
@torch.inference_mode()
def memory_forward(q,mem,L,heads):
    h=install_value(mem,L,heads)
    try:return model(**chat(q,SYS_FORCE),use_cache=False,output_hidden_states=True,return_dict=True)
    finally:h.remove();ACTIVE.discard(id(h))
FROZEN=[]
for i,(_,q,_) in enumerate(ALL):
    L=BEST[i];heads=ASSAY[i][L]["top_heads"];o=memory_forward(q,MEM[i],L,heads)
    vh=VAN[i].hidden_states[28][0,-1].float();mh=o.hidden_states[28][0,-1].float();vl=model.lm_head(vh.to(model.lm_head.weight.dtype)).float().cpu();ml=model.lm_head(mh.to(model.lm_head.weight.dtype)).float().cpu()
    FROZEN.append({"layer":L,"heads":heads,"vanilla":vl,"memory":ml,"disp":float((mh-vh).norm()/vh.norm().clamp_min(EPS)),"top1":tok.decode([int(torch.argmax(ml))])})
    tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,f"| L={L:02d} | DISP={FROZEN[-1]['disp']:.4f} | TOP1={FROZEN[-1]['top1']!r}")
print("[9/16] MEMORY FORWARDS SEALED")
print("K/V + ADDRESS + BEST LAYER + HEADS + TERMINAL DISTRIBUTIONS FROZEN | TARGET_ACCESS=False")
print("[10/16] TARGET SEAL OPEN - POST-HOC DIAGNOSTIC")
ROWS=[]
for i,target in enumerate(SEALED):
    tid=tok(target,add_special_tokens=False)["input_ids"][0];f=FROZEN[i];v=f["vanilla"];m=f["memory"];vr=int((v>v[tid]).sum()+1);mr=int((m>m[tid]).sum()+1);gain=float(m[tid]-v[tid])
    r={"target":target,"target_piece":tok.decode([tid]),"layer":f["layer"],"heads":f["heads"],"vanilla_rank":vr,"memory_rank":mr,"gain":gain,"disp":f["disp"],"top1":f["top1"]};ROWS.append(r)
    tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,f"| T={target!r} | V={vr} -> KV={mr} | G={gain:+.3f} | L={f['layer']:02d}")
print("[11/16] PRIMARY SUMMARY")
M=ROWS[:24];vr=np.array([r["vanilla_rank"] for r in M]);mr=np.array([r["memory_rank"] for r in M]);gg=np.array([r["gain"] for r in M])
SUMMARY={"vanilla_median_rank":float(np.median(vr)),"memory_median_rank":float(np.median(mr)),"rank_improved":float(np.mean(mr<vr)),"mean_gain":float(gg.mean()),"positive_gain":float(np.mean(gg>0)),"top1":float(np.mean(mr<=1)),"top10":float(np.mean(mr<=10)),"top50":float(np.mean(mr<=50)),"top256":float(np.mean(mr<=256)),"mean_disp":float(np.mean([r["disp"] for r in M])),"own_qk_margin_positive":float(np.mean([x["margin_mean"]>0 for x in CROSS])),"mean_qk_margin":float(np.mean([x["margin_mean"] for x in CROSS]))}
for k,v in SUMMARY.items():print(f"{k}={v:.6f}")
print("[12/16] ADDRESSING DIAGNOSTIC")
print(f"OWN>WRONG_MEAN={SUMMARY['own_qk_margin_positive']:.3f} | MEAN_QK_MARGIN={SUMMARY['mean_qk_margin']:+.6f}")
print("[13/16] SHOWCASE")
G=ROWS[24];print("-"*108);print("SOURCE :",SHOWCASE[0]);print("QUESTION:",SHOWCASE[1]);print("TARGET :",SHOWCASE[2]);print(f"BEST_L={G['layer']} | HEADS={G['heads']} | V={G['vanilla_rank']} -> KV={G['memory_rank']} | GAIN={G['gain']:+.4f} | DISP={G['disp']:.4f} | TOP1={G['top1']!r}");print("-"*108)
print("[14/16] PREDECLARED DIAGNOSTIC")
address=SUMMARY["own_qk_margin_positive"]>=.70 and SUMMARY["mean_qk_margin"]>0
transport=SUMMARY["rank_improved"]>=.70 and SUMMARY["mean_gain"]>0
if address and transport:CLASS="SYNTHETIC_KV_ADDRESSING_AND_VALUE_TRANSPORT_SUPPORTED"
elif address:CLASS="SYNTHETIC_KV_ADDRESSING_SUPPORTED_VALUE_TRANSPORT_NOT_YET_SUPPORTED"
elif transport:CLASS="VALUE_TRANSPORT_SIGNAL_WITHOUT_RELIABLE_KV_ADDRESSING"
else:CLASS="SYNTHETIC_KV_MEMORY_HYPOTHESIS_NOT_SUPPORTED_IN_THIS_FORM"
print(f"ADDRESS_GATE={address} | TRANSPORT_GATE={transport} | CLASS={CLASS}")
print("[15/16] SCIENTIFIC AUDIT")
print("SOURCE-DERIVED NATIVE K/V | SOURCE REMOVED BEFORE QUESTION | TARGET SEALED THROUGH CONSTRUCTION/ADDRESS/WRITE | WRONG-CASE CONTROL | NO TRAINING | NO WEIGHT UPDATE")
print("[16/16] INTEGRITY + SAVE")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("Hook leak")
R={"schema":"akbascore.test339.v1","test":"TEST 339","start":START,"end":utc(),"model":MODEL_ID,"summary":SUMMARY,"classification":CLASS,"rows":ROWS,"cross_case":CROSS,"integrity":{"fresh24_same_as_330_338":True,"source_removed_before_question":True,"target_engine_access":False,"target_posthoc_only":True,"source_derived_kv_only":True,"cross_case_control_only":True,"training":False,"weight_update":False,"hooks_remaining":len(ACTIVE),"fp_start":FP0,"fp_end":fp()}}
sha=hashlib.sha256(canon(R)).hexdigest();rid=f"T339-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json";tp=ROOT/f"{rid}.txt"
jp.write_text(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False),encoding="utf-8")
tp.write_text("\n".join(["="*108,"TEST 339 - SYNTHETIC KV MEMORY SLOT ASSAY","="*108]+[f"{k}={v}" for k,v in SUMMARY.items()]+[f"CLASS={CLASS}","SOURCE REMOVED | TARGET POST-HOC ONLY | WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]),encoding="utf-8")
print("="*108);print("TEST 339 COMPLETE");print(f"VANILLA_MED={SUMMARY['vanilla_median_rank']:.1f} | KV_MED={SUMMARY['memory_median_rank']:.1f} | IMP={SUMMARY['rank_improved']:.3f} | GAIN={SUMMARY['mean_gain']:+.3f}")
print(f"ADDRESS OWN>WRONG={SUMMARY['own_qk_margin_positive']:.3f} | QK_MARGIN={SUMMARY['mean_qk_margin']:+.4f}");print("CLASS=",CLASS);print("TARGET POST-HOC ONLY | WEIGHT INTEGRITY PASS | HOOKS=0");print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
