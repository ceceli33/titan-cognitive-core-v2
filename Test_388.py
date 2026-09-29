# ================================================================================================
# AKBASCORE - TEST 338
# CURRENT vs SPHERICAL vs PRIOR-NULL+SPHERICAL SEASC
# FROZEN DISP ROUTER | SAME FRESH-24 | SAME RSS | TARGET SEALED THROUGH ALL BRANCH FORWARDS
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
SEED=338;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYS_STD="You are a concise reasoning assistant. Use only the information in the prompt."
SYS_FORCE="Read the answer from the internal state. Return exactly one short answer. Never explain, refuse, say information is missing, or mention the prompt."
TOTAL=28;H_EXPECT=3584;END=25;EPS=1e-8;TARGET_RSS=.250235055;IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20
GENERIC=["Information","information","The","the","data","Data","missing","Missing","answer","Answer","it","It","a","an","is","was","to","of","in","and","not","unknown","requested","provided"]
PRIOR_K=8;BRANCHES=["CURRENT","SPHERICAL","NULL_SPHERICAL"]
ROOT=Path("/content/AKBASCORE_TEST338");ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
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
def chat(q,sysmsg):
    s=tok.apply_chat_template([{"role":"system","content":sysmsg},{"role":"user","content":q}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to("cuda")
print("="*108);print("TEST 338 - CURRENT vs SPHERICAL vs PRIOR-NULL+SPHERICAL SEASC");print("="*108);print("START:",START)
print("[1/20] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
if not tok.is_fast:raise RuntimeError("Fast tokenizer required")
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers
if len(layers)!=TOTAL or model.config.hidden_size!=H_EXPECT:raise RuntimeError("Architecture mismatch")
torch.cuda.synchronize();print(f"OK | {MODEL_ID} | 28L | H={model.config.hidden_size} | VOCAB={model.config.vocab_size} | {next(model.parameters()).dtype} | {time.perf_counter()-t:.2f}s")
print("[2/20] ENGINE LOCK")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();raw=[IVME*(ZIRVE*np.exp(-SONUM*L)*(1+SONUM*L)+TABAN)/(ZIRVE+TABAN) for L in range(END+1)];scale=TARGET_RSS/np.sqrt(sum(x*x for x in raw));RHO=[x*scale for x in raw];RSS=float(np.sqrt(sum(x*x for x in RHO)))
print("FP:",[f"{x:.4f}" for x in FP0]);print(f"RSS={RSS:.9f} | ROUTER=ARGMAX(DISP) CURRENT ONLY | BRANCHES={BRANCHES} | TARGET SEALED")
print("[3/20] TARGET-FREE GENERIC PRIOR SUBSPACE")
W=model.lm_head.weight.detach().float()
gids=[]
for x in GENERIC:
    ids=tok(x,add_special_tokens=False)["input_ids"]
    if ids:gids.append(ids[0])
gids=sorted(set(gids));GW=torch.stack([unit(W[i]) for i in gids]);_,S,Vh=torch.linalg.svd(GW,full_matrices=False);K=min(PRIOR_K,Vh.shape[0]);U=Vh[:K].T.contiguous()
energy=float((S[:K].square().sum()/S.square().sum()).item());norms=W.norm(dim=1);gn=[float(W[i].norm()) for i in gids]
print(f"GENERIC_IDS={len(gids)} | PRIOR_K={K} | BANK_ENERGY={energy:.6f} | VOCAB_NORM_MED={norms.median().item():.4f} | GENERIC_NORM_MEAN={np.mean(gn):.4f}")
@torch.inference_mode()
def nullvec(v):
    u=U.to(v.device);z=v-u@(u.T@v);return unit(z)
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
print("[4/20] QUERY NULL CACHE")
QN=[states(x) for x in QNULLS];print("NULL READY")
print("[5/20] ANCHOR BANK - TARGET SEALED")
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
print("[6/20] SOURCE REMOVED")
ACTIVE=set();AUD={"CURRENT":{"calls":0,"dose":0.0,"norm":0.0},"SPHERICAL":{"calls":0,"dose":0.0,"norm":0.0},"NULL_SPHERICAL":{"calls":0,"dose":0.0,"norm":0.0}}
def install(packet,mode):
    hs=[]
    for L in range(END+1):
        rho=RHO[L]
        def hk(mod,args,out,L=L,rho=rho,mode=mode):
            y=out[0] if isinstance(out,tuple) else out;z=y[:,-1,:].float();zn=z.norm(dim=-1,keepdim=True).clamp_min(EPS);v=packet[L].to(z.device)
            if mode=="NULL_SPHERICAL":v=nullvec(v)
            d=v*zn*rho
            if mode=="CURRENT":zp=z+d
            else:zp=zn*(z+d)/(z+d).norm(dim=-1,keepdim=True).clamp_min(EPS)
            AUD[mode]["calls"]+=1;AUD[mode]["dose"]=max(AUD[mode]["dose"],abs(float(d.norm()/zn)-rho));AUD[mode]["norm"]=max(AUD[mode]["norm"],abs(float(zp.norm()/zn)-1.0))
            r=y.clone();r[:,-1,:]=zp.to(y.dtype);return (r,)+out[1:] if isinstance(out,tuple) else r
        h=layers[L].register_forward_hook(hk);hs.append(h);ACTIVE.add(id(h))
    return hs
@torch.inference_mode()
def forward(q,packet=None,mode="CURRENT"):
    hs=install(packet,mode) if packet is not None else []
    try:return model(**chat(q,SYS_FORCE),use_cache=False,output_hidden_states=True,return_dict=True)
    finally:
        for h in hs:h.remove();ACTIVE.discard(id(h))
print("[7/20] BLIND DISP ROUTER - EXACT CURRENT SEASC")
SELECT=[];PACKETS=[];VAN=[]
for i,(_,q) in enumerate(ALL):
    v=forward(q);vh=v.hidden_states[28][0,-1].float();aa=[]
    for x in BANK[i]:
        o=forward(q,x["packet"],"CURRENT");h=o.hidden_states[28][0,-1].float();aa.append(float((h-vh).norm()/vh.norm().clamp_min(EPS)))
    j=int(np.argmax(aa));SELECT.append(j);PACKETS.append(BANK[i][j]["packet"]);VAN.append(v);tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,f"| ANCHOR={BANK[i][j]['anchor']!r} | DISP={aa[j]:.6f}")
print("[8/20] ROUTER + PACKETS SEALED")
print("ARGMAX(DISP) CURRENT COMPLETE | SAME PACKET USED BY ALL BRANCHES | TARGET_ACCESS=False")
print("[9/20] THREE BRANCH FORWARDS - TARGET SEALED")
FROZEN=[]
for i,(_,q) in enumerate(ALL):
    vh=VAN[i].hidden_states[28][0,-1].float();vl=model.lm_head(vh.to(model.lm_head.weight.dtype)).float().cpu();b={}
    for mode in BRANCHES:
        o=forward(q,PACKETS[i],mode);h=o.hidden_states[28][0,-1].float();log=model.lm_head(h.to(model.lm_head.weight.dtype)).float().cpu()
        b[mode]={"logits":log,"disp":float((h-vh).norm()/vh.norm().clamp_min(EPS)),"top1":tok.decode([int(torch.argmax(log))])}
    FROZEN.append({"vanilla":vl,"branches":b});tag="SHOWCASE" if i==24 else f"{i+1:02d}/24"
    print(tag,"| "+" | ".join([f"{m}:D={b[m]['disp']:.3f},T1={b[m]['top1']!r}" for m in BRANCHES]))
print("[10/20] ALL BRANCH DISTRIBUTIONS SEALED")
print("CURRENT/SPHERICAL/NULL_SPHERICAL COMPLETE | TARGET STILL SEALED")
print("[11/20] TARGET SEAL OPEN - POST-HOC READOUT")
ROWS=[]
for i,target in enumerate(SEALED):
    tid=tok(target,add_special_tokens=False)["input_ids"][0];v=FROZEN[i]["vanilla"];vr=int((v>v[tid]).sum()+1);rr={}
    for mode in BRANCHES:
        x=FROZEN[i]["branches"][mode]["logits"];rr[mode]={"rank":int((x>x[tid]).sum()+1),"gain":float(x[tid]-v[tid]),"top1":FROZEN[i]["branches"][mode]["top1"],"disp":FROZEN[i]["branches"][mode]["disp"]}
    ROWS.append({"target":target,"target_piece":tok.decode([tid]),"anchor":BANK[i][SELECT[i]]["anchor"],"vanilla_rank":vr,"branches":rr})
    tag="SHOWCASE" if i==24 else f"{i+1:02d}/24";print(tag,f"| T={target!r} | V={vr} | "+" | ".join([f"{m}:R{rr[m]['rank']},G{rr[m]['gain']:+.2f}" for m in BRANCHES]))
print("[12/20] BRANCH SUMMARY")
M=ROWS[:24];SUM={}
for mode in BRANCHES:
    rs=np.array([r["branches"][mode]["rank"] for r in M]);vr=np.array([r["vanilla_rank"] for r in M]);gg=np.array([r["branches"][mode]["gain"] for r in M])
    d={"median_rank":float(np.median(rs)),"rank_improved":float(np.mean(rs<vr)),"mean_gain":float(gg.mean()),"positive":float(np.mean(gg>0)),"top1":float(np.mean(rs<=1)),"top10":float(np.mean(rs<=10)),"top50":float(np.mean(rs<=50)),"top256":float(np.mean(rs<=256)),"top1000":float(np.mean(rs<=1000)),"mean_disp":float(np.mean([r["branches"][mode]["disp"] for r in M]))};SUM[mode]=d
    print(f"{mode:14s} | MED={d['median_rank']:.1f} | IMP={d['rank_improved']:.3f} | GAIN={d['mean_gain']:+.3f} | POS={d['positive']:.3f} | T10={d['top10']:.3f} | T256={d['top256']:.3f}")
print("[13/20] SPHERICAL ABLATION")
C=SUM["CURRENT"];Sph=SUM["SPHERICAL"];Nul=SUM["NULL_SPHERICAL"]
print(f"CURRENT MED={C['median_rank']:.1f} G={C['mean_gain']:+.3f} T256={C['top256']:.3f}")
print(f"SPHERICAL MED={Sph['median_rank']:.1f} G={Sph['mean_gain']:+.3f} T256={Sph['top256']:.3f}")
print("[14/20] PRIOR-NULL ABLATION")
print(f"NULL+SPH MED={Nul['median_rank']:.1f} G={Nul['mean_gain']:+.3f} T256={Nul['top256']:.3f}")
print("[15/20] GENERIC PRIOR GEOMETRY")
proj=[]
for p in PACKETS[:24]:
    for L in range(END+1):
        v=p[L].float();u=U.to(v.device);proj.append(float((u.T@v).norm()))
print(f"MEAN_PACKET_GENERIC_SUBSPACE_PROJECTION={np.mean(proj):.6f} | MED={np.median(proj):.6f} | PRIOR_K={K}")
print("[16/20] SHOWCASE")
G=ROWS[24];print("-"*108);print("SOURCE :",SHOWCASE[0]);print("QUESTION:",SHOWCASE[1]);print("ANCHOR :",G["anchor"]);print("TARGET :",SHOWCASE[2]);print("TARGET_FIRST:",repr(G["target_piece"]),f"| VANILLA_RANK={G['vanilla_rank']}")
for m in BRANCHES:print(f"{m:14s} | RANK={G['branches'][m]['rank']} | GAIN={G['branches'][m]['gain']:+.4f} | DISP={G['branches'][m]['disp']:.4f} | TOP1={G['branches'][m]['top1']!r}")
print("-"*108)
print("[17/20] PREDECLARED DIAGNOSTIC")
sph_win=Sph["median_rank"]<C["median_rank"] and Sph["mean_gain"]>C["mean_gain"]
null_win=Nul["median_rank"]<Sph["median_rank"] and Nul["mean_gain"]>Sph["mean_gain"]
if sph_win and null_win:CLASS="NORM_PRESERVATION_AND_GENERIC_PRIOR_NULLING_BOTH_SUPPORTED"
elif sph_win:CLASS="NORM_PRESERVATION_SUPPORTED_GENERIC_PRIOR_NULLING_NOT_SUPPORTED"
elif null_win:CLASS="GENERIC_PRIOR_NULLING_SUPPORTED_BEYOND_SPHERICAL_BASELINE"
else:CLASS="STRUCTURAL_ABLATIONS_DO_NOT_OUTPERFORM_CURRENT_SEASC"
print(f"SPHERICAL_WIN={sph_win} | NULL_OVER_SPHERICAL={null_win} | CLASS={CLASS}")
print("[18/20] LEAKAGE GUARD")
print("GENERIC BANK PREDECLARED | PRIOR SUBSPACE TARGET-FREE | ROUTER CURRENT-ONLY AND FROZEN | SAME PACKET ALL BRANCHES | TARGET POST-HOC ONLY")
print("[19/20] SCIENTIFIC AUDIT")
print("SAME FRESH-24 AS TEST330-337 | SAME RSS | CURRENT vs NORM-PRESERVING SPHERICAL vs PRIOR-NULL+SPHERICAL | NO TRAINING | NO WEIGHT UPDATE")
print("[20/20] INTEGRITY + SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("Hook leak")
if abs(RSS-TARGET_RSS)>1e-9:raise RuntimeError("RSS failure")
if max(AUD[m]["dose"] for m in BRANCHES)>1e-6:raise RuntimeError("Dose failure")
if max(AUD[m]["norm"] for m in ["SPHERICAL","NULL_SPHERICAL"])>1e-5:raise RuntimeError("Spherical norm failure")
R={"schema":"akbascore.test338.v1","test":"TEST 338","start":START,"end":utc(),"model":MODEL_ID,"router":"ARGMAX_DISP_CURRENT_FROZEN","rss":RSS,"generic_bank":GENERIC,"prior_k":K,"generic_bank_energy":energy,"summary":SUM,"classification":CLASS,"rows":ROWS,"audit":AUD,"integrity":{"same_fresh24_as_test330_337":True,"case_isolated":True,"cross_case_data":False,"target_engine_access":False,"target_posthoc_only":True,"generic_prior_target_free":True,"router_current_only":True,"same_packet_all_branches":True,"training":False,"weight_update":False,"motor_layers":[0,25],"observation_layers":[26,27],"hooks_remaining":len(ACTIVE),"fp_start":FP0,"fp_end":fp()}}
sha=hashlib.sha256(canon(R)).hexdigest();rid=f"T338-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{rid}.json";tp=ROOT/f"{rid}.txt"
jp.write_text(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False),encoding="utf-8")
tp.write_text("\n".join(["="*108,"TEST 338 - STRUCTURAL FORWARD-PASS ABLATION","="*108]+[f"{m}: MED={SUM[m]['median_rank']:.1f} GAIN={SUM[m]['mean_gain']:+.6f} T256={SUM[m]['top256']:.6f}" for m in BRANCHES]+[f"CLASS={CLASS}","GENERIC PRIOR TARGET-FREE | ROUTER FROZEN | SAME PACKET | SAME RSS | WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]),encoding="utf-8")
print("="*108);print("TEST 338 COMPLETE")
for m in BRANCHES:print(f"{m:14s} | MED={SUM[m]['median_rank']:.1f} | GAIN={SUM[m]['mean_gain']:+.3f} | T256={SUM[m]['top256']:.3f}")
print("CLASS=",CLASS);print("TARGET POST-HOC ONLY | SAME RSS | WEIGHT INTEGRITY PASS | AKBASCORE HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
