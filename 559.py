# TEST559 — AKBASCORE MAM · DEFERRED NUMERICAL CONSOLIDATION
# TEST558 WORKING BACKBONE · SWITCH6 / DC6-CANON / DC6-FINALPOS / DC10-CANON / DC6-NOCROSS
# FROZEN MISTRAL · SOURCE-FREE CONSOLIDATION · NO TRAINING / RETRIEVAL / ROUTER / QSF
import os,sys,subprocess,importlib.util,random,time,hashlib,json,re,gc
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
TEST="559";SEED=552552;PANEL_SEED=550550;MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";DEVICE=torch.device("cuda");DTYPE=torch.bfloat16;MAX_NEW=16
POSITIONS=["FIRST","MIDDLE","LAST"];QUERY_CYCLE=["CURRENT","FORMER","NEAR","ROLE"];EXPECTED_PANEL_SHA="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
ARMS=["INDEP","JOINT","SWITCH6_REF","DC6_CANON","DC6_FINALPOS","DC10_CANON","DC6_NOCROSS"]
if not torch.cuda.is_available():raise RuntimeError("CUDA gerekli.")
os.environ["TOKENIZERS_PARALLELISM"]="false";random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);T0=time.perf_counter()
def sha_obj(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def norm(s):return " ".join(re.sub(r"[^\w\s-]"," ",s.casefold().strip(),flags=re.UNICODE).split())
def hit(s,g):return re.search(r"(?<!\w)"+re.escape(norm(g))+r"(?!\w)",norm(s)) is not None
def section(n,title):print(f"\n[{n}/10] {title}",flush=True)
print("="*150);print("TEST559 — AKBASCORE MAM · DEFERRED NUMERICAL CONSOLIDATION");print("TEST558 BACKBONE · INDEPENDENT H6/H10 CHECKPOINTS · SOURCE-FREE RECONSTRUCTION");print("FROZEN MISTRAL · NO TRAINING · NO RETRIEVAL · NO ROUTER · NO QSF");print("="*150)
section(1,"Frozen Mistral...")
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",dtype=DTYPE).eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;NL=len(model.model.layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None) or H//QH
if (NL,H,QH,KVH,HD)!=(32,4096,32,8,128):raise RuntimeError("Architecture mismatch.")
print("MODEL:",MODEL_ID);print("GPU:",torch.cuda.get_device_name(0));print("TORCH:",torch.__version__,"TRANSFORMERS:",transformers.__version__)
print(f"NL={NL} H={H} QH={QH} KVH={KVH} HD={HD} DTYPE={DTYPE}");print("ROPE:",getattr(cfg,"rope_parameters",None));print("TRAINABLE:",sum(int(p.requires_grad) for p in model.parameters()))
FP=[model.model.layers[0].self_attn.q_proj.weight,model.model.layers[8].self_attn.o_proj.weight,model.model.layers[16].mlp.down_proj.weight,model.model.layers[24].self_attn.o_proj.weight,model.model.layers[31].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
def sentinel():
    h=hashlib.sha256()
    for p in FP:
        a=p.detach().reshape(-1);n=a.numel()
        for j in range(16):
            o=j*(n-256)//15;h.update(a[o:o+256].float().cpu().numpy().tobytes())
    return h.hexdigest()
S0=sentinel()
BASE=[
("Zorvan","Melket","Dravel","Oakhaven","Pelnor"),("Kelvar","Nareth","Solven","Branik","Tarsen"),("Tarev","Luneth","Varos","Cedran","Mireth"),("Belnor","Arven","Dorel","Kesmar","Falven"),
("Ravik","Selora","Terven","Maldor","Nerik"),("Nemor","Calven","Istral","Pareth","Dovren"),("Darsen","Velora","Keldin","Orvek","Sarnel"),("Feron","Talven","Merith","Sovran","Belvik"),
("Larev","Nerith","Calder","Veyron","Tormek"),("Torven","Elsar","Marvek","Dorin","Kaleth"),("Selnor","Kareth","Valen","Ordan","Mervek"),("Mirev","Taldor","Neris","Kelmar","Sorvik"),
("Varen","Solith","Deran","Malvek","Cordan"),("Kelor","Ardin","Velmar","Toren","Narell"),("Narev","Belith","Corven","Sareth","Dorvik"),("Dervan","Mirel","Talvek","Orsen","Kelron"),
("Calnor","Verith","Naldor","Seren","Parvek"),("Parel","Dorven","Kelith","Maros","Tervik"),("Sorven","Tarell","Vindor","Nelmar","Calrek"),("Barel","Corith","Laven","Derik","Solmar"),
("Ralen","Mervor","Talith","Kesven","Noreth"),("Norel","Valdor","Serith","Calvenor","Darvek"),("Tervan","Orel","Mardin","Velos","Karven"),("Karev","Solen","Dareth","Mirven","Talrek")]
W=[]
for i,(s,current,near,former,role_target) in enumerate(BASE):
    facts=[("CURRENT",f"The current capital of {s} is {current}.",current),("NEAR",f"The largest city of {s} is {near}.",near),("FORMER",f"The former capital of {s} was {former}.",former),("ROLE",f"The current capital of {role_target} is {s}.",s)]
    queries=[("CURRENT",f"What is the current capital of {s}?",current),("FORMER",f"What was the former capital of {s}?",former),("NEAR",f"What is the largest city of {s}?",near),("ROLE",f"What is the current capital of {role_target}?",s)]
    W.append({"id":i,"facts":facts,"queries":queries})
SYSTEM="Answer the question using only the information provided. Give only the requested name and nothing else."
def source_prefix(f):return f"<s>[INST] {SYSTEM}\n\nINFORMATION:\n{f}\n\n"
def canonical_prefix():return f"<s>[INST] {SYSTEM}\n\nINFORMATION:\n"
def query_suffix(q):return f"QUESTION:\n{q}\n\nANSWER: [/INST]"
PREFIX_IDS=tok(canonical_prefix(),add_special_tokens=False).input_ids;P=len(PREFIX_IDS);CAR=[];KEY_TO_IDX={}
for wi,w in enumerate(W):
    for typ,fact,gold in w["facts"]:
        idx=len(CAR);ids=tok(source_prefix(fact),add_special_tokens=False).input_ids
        if ids[:P]!=PREFIX_IDS:raise RuntimeError(f"Canonical prefix mismatch {idx}")
        CAR.append({"idx":idx,"world":wi,"type":typ,"fact":fact,"gold":gold,"body":ids[P:]});KEY_TO_IDX[(wi,typ)]=idx
if len(CAR)!=96:raise RuntimeError("Cartridge count mismatch.")
section(2,"Panel provenance...")
rng=random.Random(PANEL_SEED);pool=list(range(96));rng.shuffle(pool);PANEL=[];cursor=0
for wi in range(24):
    typ=QUERY_CYCLE[wi%4];target=KEY_TO_IDX[(wi,typ)];others=[]
    while len(others)<4:
        ci=pool[cursor%96];cursor+=1
        if ci==target or CAR[ci]["world"]==wi or any(CAR[x]["world"]==CAR[ci]["world"] for x in others):continue
        others.append(ci)
    PANEL.append({"case":wi,"target":target,"others":others})
PANEL_SHA=sha_obj(PANEL);print("PANEL SHA:",PANEL_SHA)
if PANEL_SHA!=EXPECTED_PANEL_SHA:raise RuntimeError("Panel provenance FAIL.")
print("PANEL PROVENANCE: PASS")
def order_keys(z,slot):
    d=list(z["others"]);t=z["target"]
    return [t]+d if slot=="FIRST" else d[:2]+[t]+d[2:] if slot=="MIDDLE" else d+[t]
def cache_layers(pkv):
    if hasattr(pkv,"layers"):
        out=[]
        for layer in pkv.layers:
            k=getattr(layer,"keys",None);v=getattr(layer,"values",None)
            if k is None:k=getattr(layer,"key_cache",None)
            if v is None:v=getattr(layer,"value_cache",None)
            if k is None or v is None:raise RuntimeError("Cache API mismatch.")
            out.append((k,v))
        if out:return tuple(out)
    if hasattr(pkv,"key_cache"):return tuple(zip(pkv.key_cache,pkv.value_cache))
    raise RuntimeError("Cache API mismatch.")
def clone_layers(pkv):return tuple((k.detach().clone(),v.detach().clone()) for k,v in cache_layers(pkv))
def build_cache(layers):
    c=DynamicCache()
    for li,(k,v) in enumerate(layers):c.update(k,v,li)
    return c
def cache_len(c):return int(c.get_seq_length())
@torch.no_grad()
def answer_layers(layers,q):
    cache=build_cache(layers);physical=cache_len(cache);qids=tok(query_suffix(q),add_special_tokens=False).input_ids
    ids=torch.tensor(qids,dtype=torch.long,device=DEVICE).reshape(1,-1);qlen=ids.shape[1]
    pos=torch.arange(physical,physical+qlen,dtype=torch.long,device=DEVICE).unsqueeze(0);att=torch.ones((1,physical+qlen),dtype=torch.long,device=DEVICE)
    out=model(input_ids=ids,past_key_values=cache,attention_mask=att,position_ids=pos,cache_position=pos.squeeze(0),use_cache=True,return_dict=True)
    cache=out.past_key_values;nxt=out.logits[:,-1,:].argmax(-1,keepdim=True);eos=set() if tok.eos_token_id is None else {int(tok.eos_token_id)};gen=[]
    for _ in range(MAX_NEW):
        gen.append(nxt)
        if int(nxt.item()) in eos:break
        physical=cache_len(cache);pos=torch.tensor([[physical]],dtype=torch.long,device=DEVICE);att=torch.ones((1,physical+1),dtype=torch.long,device=DEVICE)
        out=model(input_ids=nxt,past_key_values=cache,attention_mask=att,position_ids=pos,cache_position=pos.squeeze(0),use_cache=True,return_dict=True)
        cache=out.past_key_values;nxt=out.logits[:,-1,:].argmax(-1,keepdim=True)
    return tok.decode(torch.cat(gen,dim=1)[0],skip_special_tokens=True).strip()
def make_case(keys,target):
    ids=list(PREFIX_IDS);ranges={}
    for ci in keys:
        a=len(ids);ids.extend(CAR[ci]["body"]);ranges[ci]=(a,len(ids))
    n=len(ids);group=torch.full((n,),-1,dtype=torch.long,device=DEVICE)
    for j,ci in enumerate(keys):
        a,b=ranges[ci];group[a:b]=j
    i=torch.arange(n,device=DEVICE);causal=i[:,None]>=i[None,:]
    allowed=causal&((group[:,None]==-1)|(group[None,:]==-1)|(group[:,None]==group[None,:]))
    block=torch.zeros((1,1,n,n),dtype=DTYPE,device=DEVICE);block.masked_fill_(~allowed,torch.finfo(DTYPE).min)
    joint=torch.zeros((1,1,n,n),dtype=DTYPE,device=DEVICE);joint.masked_fill_(~causal,torch.finfo(DTYPE).min)
    full=torch.zeros(n,dtype=torch.bool,device=DEVICE);full[P:]=True
    return torch.tensor(ids,dtype=torch.long,device=DEVICE).reshape(1,-1),joint,block,full,ranges
def replace_hidden(out,new):
    if isinstance(out,tuple):return (new,)+out[1:]
    if isinstance(out,list):return [new]+out[1:]
    return new
@torch.no_grad()
def write_ref(ids,joint,block,kind="JOINT",cut=6):
    handles=[]
    try:
        for li,layer in enumerate(model.model.layers):
            def pre(module,args,kwargs,li=li):
                if kind=="JOINT":mask=joint
                elif kind=="INDEP":mask=block
                else:mask=block if li<=cut else joint
                kwargs["attention_mask"]=mask;return args,kwargs
            handles.append(layer.register_forward_pre_hook(pre,with_kwargs=True))
        n=ids.shape[1];pos=torch.arange(n,dtype=torch.long,device=DEVICE).unsqueeze(0)
        out=model(input_ids=ids,attention_mask=torch.ones((1,n),dtype=torch.long,device=DEVICE),position_ids=pos,cache_position=pos.squeeze(0),use_cache=True,return_dict=True)
        result=clone_layers(out.past_key_values);del out;return result
    finally:
        for h in handles:h.remove()
@torch.no_grad()
def forge_checkpoint(ci,cut,offset=None):
    # Independent write: one prefix + one fact. No other cartridge is visible.
    ids=PREFIX_IDS+CAR[ci]["body"];n=len(ids);x=torch.tensor(ids,dtype=torch.long,device=DEVICE).reshape(1,-1)
    pos=torch.arange(n,dtype=torch.long,device=DEVICE)
    if offset is not None:pos[P:]=torch.arange(offset,offset+n-P,dtype=torch.long,device=DEVICE)
    states={};handles=[]
    try:
        def capture(module,args,out):states["h"]=((out[0] if isinstance(out,(tuple,list)) else out).detach().clone())
        handles.append(model.model.layers[cut].register_forward_hook(capture))
        out=model(input_ids=x,attention_mask=torch.ones((1,n),dtype=torch.long,device=DEVICE),position_ids=pos.unsqueeze(0),cache_position=torch.arange(n,device=DEVICE),use_cache=True,return_dict=True)
        early=tuple((k.detach().clone(),v.detach().clone()) for k,v in cache_layers(out.past_key_values)[:cut+1])
        h=states["h"];del out
        if h.shape!=(1,n,H) or len(early)!=cut+1:raise RuntimeError("Checkpoint shape mismatch.")
        return {"h":h,"kv":early,"positions":pos.clone(),"cut":cut,"body_len":n-P}
    finally:
        for hook in handles:hook.remove()
def rotate_half(x):
    a,b=x.chunk(2,dim=-1);return torch.cat((-b,a),dim=-1)
@torch.no_grad()
def rope_tables(pos):
    dummy=torch.zeros((1,pos.numel(),H),dtype=DTYPE,device=DEVICE)
    cos,sin=model.model.rotary_emb(dummy,pos.reshape(1,-1))
    return cos.unsqueeze(1).float(),sin.unsqueeze(1).float()
@torch.no_grad()
def rephase_key(k,oldpos,newpos):
    if torch.equal(oldpos,newpos):return k
    co,so=rope_tables(oldpos);cn,sn=rope_tables(newpos)
    x=k.float();unrot=x*co-rotate_half(x)*so
    return (unrot*cn+rotate_half(unrot)*sn).to(k.dtype)
@torch.no_grad()
def numerical_consolidation(keys,ranges,joint,block,cut=6,canonical=True,cross=True):
    # Original fact tokens are used ONLY during independent forging.
    # Consolidation itself receives zero embeddings and saved numerical checkpoints.
    n=joint.shape[-1];checkpoints={}
    for ci in keys:
        a,b=ranges[ci];checkpoints[ci]=forge_checkpoint(ci,cut,None if canonical else a)
    prefix=checkpoints[keys[0]]["h"][:,:P,:]
    parts=[prefix];early=[]
    for ci in keys:parts.append(checkpoints[ci]["h"][:,P:,:])
    assembled=torch.cat(parts,dim=1)
    if assembled.shape!=(1,n,H):raise RuntimeError("Numerical checkpoint assembly FAIL.")
    for li in range(cut+1):
        ks=[checkpoints[keys[0]]["kv"][li][0][:,:,:P,:]];vs=[checkpoints[keys[0]]["kv"][li][1][:,:,:P,:]]
        for ci in keys:
            a,b=ranges[ci];cp=checkpoints[ci];old=cp["positions"][P:];new=torch.arange(a,b,dtype=torch.long,device=DEVICE)
            k=cp["kv"][li][0][:,:,P:,:];v=cp["kv"][li][1][:,:,P:,:]
            ks.append(rephase_key(k,old,new));vs.append(v)
        kk=torch.cat(ks,dim=-2);vv=torch.cat(vs,dim=-2)
        if kk.shape!=(1,KVH,n,HD) or vv.shape!=kk.shape:raise RuntimeError("Early KV assembly FAIL.")
        early.append((kk,vv))
    handles=[]
    try:
        def inject(module,args,out):
            h=out[0] if isinstance(out,(tuple,list)) else out
            if h.shape!=assembled.shape:raise RuntimeError("H checkpoint injection FAIL.")
            return replace_hidden(out,assembled)
        handles.append(model.model.layers[cut].register_forward_hook(inject))
        for li,layer in enumerate(model.model.layers):
            if li<=cut:continue
            def pre(module,args,kwargs,li=li):
                kwargs["attention_mask"]=joint if cross else block;return args,kwargs
            handles.append(layer.register_forward_pre_hook(pre,with_kwargs=True))
        # No input_ids and no source-text embeddings during consolidation.
        zeros=torch.zeros((1,n,H),dtype=DTYPE,device=DEVICE)
        pos=torch.arange(n,dtype=torch.long,device=DEVICE)
        out=model(inputs_embeds=zeros,attention_mask=torch.ones((1,n),dtype=torch.long,device=DEVICE),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        generated=clone_layers(out.past_key_values)
        if len(generated)!=NL:raise RuntimeError("Consolidated cache length FAIL.")
        result=tuple(early)+generated[cut+1:]
        if len(result)!=NL:raise RuntimeError("Consolidation assembly FAIL.")
        del out,zeros,generated
        return result
    finally:
        for hook in handles:hook.remove()
section(3,"Mask and numerical checkpoint contract...")
z=PANEL[0];keys=order_keys(z,"MIDDLE");ids,joint,block,full,ranges=make_case(keys,z["target"]);n=ids.shape[1]
if joint.shape!=(1,1,n,n) or block.shape!=(1,1,n,n) or int(full[:P].sum())!=0:raise RuntimeError("Mask contract FAIL.")
print(f"MASK CONTRACT: PASS | TOKENS={n} PREFIX={P} FULL={int(full.sum())}")
cp=forge_checkpoint(keys[0],6)
if cp["h"].shape[-1]!=H or len(cp["kv"])!=7:raise RuntimeError("H6 checkpoint FAIL.")
print("INDEPENDENT H6 CHECKPOINT: PASS")
testpos=torch.arange(P+len(CAR[keys[0]]["body"]),device=DEVICE)
kt=cp["kv"][0][0];kr=rephase_key(kt,testpos,testpos+7);kb=rephase_key(kr,testpos+7,testpos)
err=float((kb.float()-kt.float()).abs().max().item())
print(f"ROPE ROUNDTRIP MAX ERROR: {err:.6f}")
if not np.isfinite(err) or err>0.08:raise RuntimeError("RoPE numerical roundtrip FAIL.")
print("ROPE NUMERICAL GATE: PASS (BF16 tolerance; not a 1-ULP identity claim)")
del cp,kt,kr,kb,ids,joint,block,full,ranges
section(4,"TEST558 baseline replication...")
BASELINES=[]
for ci,z in enumerate(PANEL):
    typ=CAR[z["target"]]["type"];q=next(x[1] for x in W[ci]["queries"] if x[0]==typ);gold=CAR[z["target"]]["gold"]
    for slot in POSITIONS:
        keys=order_keys(z,slot);ids,joint,block,full,_=make_case(keys,z["target"])
        jl=write_ref(ids,joint,block,"JOINT");il=write_ref(ids,joint,block,"INDEP")
        ja=answer_layers(jl,q);ia=answer_layers(il,q)
        BASELINES.append({"case":ci,"slot":slot,"joint":bool(hit(ja,gold)),"indep":bool(hit(ia,gold))})
        del jl,il,ids,joint,block,full
BJ={s:sum(int(x["joint"]) for x in BASELINES if x["slot"]==s) for s in POSITIONS}
BI={s:sum(int(x["indep"]) for x in BASELINES if x["slot"]==s) for s in POSITIONS}
print("JOINT:",BJ);print("INDEP:",BI)
if BJ!={"FIRST":24,"MIDDLE":24,"LAST":24} or BI!={"FIRST":4,"MIDDLE":3,"LAST":6}:raise RuntimeError("TEST558 baseline replication FAIL.")
print("BASELINE REPLICATION: PASS")
section(5,"Deferred numerical consolidation...")
RESULTS=[]
for ci,z in enumerate(PANEL):
    typ=CAR[z["target"]]["type"];q=next(x[1] for x in W[ci]["queries"] if x[0]==typ);gold=CAR[z["target"]]["gold"]
    for slot in POSITIONS:
        keys=order_keys(z,slot);ids,joint,block,full,ranges=make_case(keys,z["target"])
        arms={}
        arms["INDEP"]=write_ref(ids,joint,block,"INDEP")
        arms["JOINT"]=write_ref(ids,joint,block,"JOINT")
        arms["SWITCH6_REF"]=write_ref(ids,joint,block,"SWITCH",6)
        arms["DC6_CANON"]=numerical_consolidation(keys,ranges,joint,block,6,True,True)
        arms["DC6_FINALPOS"]=numerical_consolidation(keys,ranges,joint,block,6,False,True)
        arms["DC10_CANON"]=numerical_consolidation(keys,ranges,joint,block,10,True,True)
        arms["DC6_NOCROSS"]=numerical_consolidation(keys,ranges,joint,block,6,True,False)
        for arm,layers in arms.items():
            ans=answer_layers(layers,q)
            RESULTS.append({"case":ci,"slot":slot,"arm":arm,"gold":gold,"answer":ans,"ok":bool(hit(ans,gold))})
        del arms,ids,joint,block,full,ranges
    print(f"CASE {ci:02d} complete",flush=True)
    if (ci+1)%4==0:
        torch.cuda.empty_cache();gc.collect()
        print(f"PROGRESS {ci+1}/24 | elapsed={time.perf_counter()-T0:.1f}s",flush=True)
section(6,"Raw integrity...")
EXP=24*3*len(ARMS);print("ROWS:",len(RESULTS),"EXPECTED:",EXP)
if len(RESULTS)!=EXP:raise RuntimeError("Row count FAIL.")
for arm in ARMS:
    rr=[r for r in RESULTS if r["arm"]==arm]
    if len(rr)!=72 or len({(r["case"],r["slot"]) for r in rr})!=72:raise RuntimeError(f"{arm} integrity FAIL.")
print("RAW INTEGRITY: PASS")
section(7,"Composition scores...")
S={}
for arm in ARMS:
    S[arm]={slot:sum(int(r["ok"]) for r in RESULTS if r["arm"]==arm and r["slot"]==slot) for slot in POSITIONS}
    S[arm]["TOTAL"]=sum(S[arm][s] for s in POSITIONS)
    print(f"{arm:22s} FIRST={S[arm]['FIRST']:02d}/24 MIDDLE={S[arm]['MIDDLE']:02d}/24 LAST={S[arm]['LAST']:02d}/24 TOTAL={S[arm]['TOTAL']:02d}/72")
section(8,"Case-level gain/loss and identity...")
MAP={arm:{(r["case"],r["slot"]):r for r in RESULTS if r["arm"]==arm} for arm in ARMS}
DIAG={}
for arm in ARMS:
    if arm=="INDEP":continue
    gain=sum(int(not MAP["INDEP"][k]["ok"] and MAP[arm][k]["ok"]) for k in MAP["INDEP"])
    loss=sum(int(MAP["INDEP"][k]["ok"] and not MAP[arm][k]["ok"]) for k in MAP["INDEP"])
    answer_identity=sum(int(norm(MAP[arm][k]["answer"])==norm(MAP["JOINT"][k]["answer"])) for k in MAP["JOINT"])
    ok_identity=sum(int(MAP[arm][k]["ok"]==MAP["JOINT"][k]["ok"]) for k in MAP["JOINT"])
    DIAG[arm]={"gain":gain,"loss":loss,"net":gain-loss,"answer_identity_joint":answer_identity,"ok_identity_joint":ok_identity}
    print(f"{arm:22s} GAIN={gain:02d} LOSS={loss:02d} NET={gain-loss:+03d} | OK-ID-JOINT={ok_identity:02d}/72 ANS-ID-JOINT={answer_identity:02d}/72")
section(9,"Engineering decision...")
gates={"JOINT":S["JOINT"]["TOTAL"]==72,"INDEP":S["INDEP"]["TOTAL"]==13,"SWITCH6_REF":S["SWITCH6_REF"]["TOTAL"]>=70}
for k,v in gates.items():print(f"GATE {k}: {'PASS' if v else 'FAIL'}")
canon=S["DC6_CANON"]["TOTAL"];finalpos=S["DC6_FINALPOS"]["TOTAL"];nocross=S["DC6_NOCROSS"]["TOTAL"];switch=S["SWITCH6_REF"]["TOTAL"]
if not all(gates.values()):DECISION="INCONCLUSIVE — REFERENCE GATE FAILURE"
elif finalpos<switch-2:DECISION="CHECKPOINT RECONSTRUCTION MISMATCH — FINALPOS CONTROL FAILED"
elif canon>=68 and nocross<=30:DECISION="STRONG SOURCE-FREE NUMERICAL CONSOLIDATION — POSITION TRANSFER SUPPORTED"
elif canon>=50 and nocross<canon:DECISION="PARTIAL SOURCE-FREE CONSOLIDATION — POSITION/STATE MISMATCH POSSIBLE"
elif canon>nocross and canon>13:DECISION="WEAK SOURCE-FREE CONSOLIDATION — FURTHER ISOLATION REQUIRED"
elif nocross>=canon:DECISION="CROSS-ATTENTION SPECIFICITY NOT ESTABLISHED"
else:DECISION="DC6-CANON FAILED TO RECOVER COMPOSITION"
print("SWITCH6:",switch,"/72");print("DC6-FINALPOS:",finalpos,"/72");print("DC6-CANON:",canon,"/72");print("DC10-CANON:",S["DC10_CANON"]["TOTAL"],"/72");print("DC6-NOCROSS:",nocross,"/72")
print("DECISION:",DECISION)
print("CAUTION: DC6-FINALPOS uses independently forged numerical checkpoints at their eventual global positions; this is a diagnostic control, not position-independent deployment.")
print("CAUTION: DC6-CANON stores H6 residuals in addition to early K/V. It tests an augmented numerical cartridge, not strict KV-only sufficiency.")
print("CAUTION: This experiment tests BLOCK-style composition and numerical consolidation, not behavioral identity with the original P-SEQ cartridge-concatenation protocol.")
section(10,"Final research record...")
WEIGHT_OK=sentinel()==S0;TRAINABLE=sum(int(p.requires_grad) for p in model.parameters())
LOCK={"test":TEST,"model":MODEL_ID,"panel_sha":PANEL_SHA,"seed":SEED,"panel_seed":PANEL_SEED,"arms":ARMS,"cuts":[6,10],"checkpoint":"independent body residual after cut layer + early KV","canonical":"prefix+single fact","finalpos":"independent body at eventual logical positions","consolidation":"zero-input forward; injected numerical residual; frozen downstream layers","training":False,"retrieval":False,"router":False,"qsf":False}
LOCK_SHA=sha_obj(LOCK)
FINAL={"test":TEST,"lock_sha":LOCK_SHA,"panel_sha":PANEL_SHA,"scores":S,"diag":DIAG,"gates":gates,"decision":DECISION,"weight_ok":WEIGHT_OK,"trainable":TRAINABLE}
RESULT_SHA=sha_obj(FINAL)
print("="*150);print("TEST559 — FINAL RESEARCH RECORD");print("="*150)
print("MODEL                  :",MODEL_ID);print("PANEL SHA              :",PANEL_SHA);print("LOCK SHA               :",LOCK_SHA);print("RESULT SHA             :",RESULT_SHA)
for arm in ARMS:print(f"{arm:24s}: {S[arm]}")
print("REFERENCE GATES        :",gates);print("ENGINEERING DECISION   :",DECISION)
print("WEIGHT SENTINEL        :","PASS" if WEIGHT_OK else "FAIL");print("TRAINABLE              :",TRAINABLE)
print("TOTAL TIME             :",f"{time.perf_counter()-T0:.2f}s")
print("INTERPRETATION         : Independent numerical checkpoints are consolidated without replaying source tokens during the consolidation pass.")
print("NEXT TEST              : TEST560 only after TEST559 interpretation")
print("="*150)
if not WEIGHT_OK:raise RuntimeError("WEIGHT SENTINEL FAILURE.")
if TRAINABLE!=0:raise RuntimeError("TRAINABLE PARAMETER FAILURE.")
