# TEST560 — AKBASCORE MAM · INCREMENTAL DEFERRED CONSOLIDATION
# TEST559 WORKING BACKBONE PRESERVED
# BATCH DC6 vs INCREMENTAL DC6 · APPEND-ONLY NUMERICAL MEMORY GROWTH
# FROZEN MISTRAL · SOURCE-FREE CONSOLIDATION · NO TRAINING / RETRIEVAL / ROUTER / QSF
import os,sys,subprocess,importlib.util,random,time,hashlib,json,re,gc
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
TEST="560";SEED=552552;PANEL_SEED=550550;MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";DEVICE=torch.device("cuda");DTYPE=torch.bfloat16;MAX_NEW=16;CUT=6
POSITIONS=["FIRST","MIDDLE","LAST"];QUERY_CYCLE=["CURRENT","FORMER","NEAR","ROLE"];EXPECTED_PANEL_SHA="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
ARMS=["INDEP","JOINT","BATCH_DC6","INCR_DC6"]
if not torch.cuda.is_available():raise RuntimeError("CUDA gerekli.")
os.environ["TOKENIZERS_PARALLELISM"]="false";random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);T0=time.perf_counter()
def sha_obj(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def norm(s):return " ".join(re.sub(r"[^\w\s-]"," ",s.casefold().strip(),flags=re.UNICODE).split())
def hit(s,g):return re.search(r"(?<!\w)"+re.escape(norm(g))+r"(?!\w)",norm(s)) is not None
def section(n,t):print(f"\n[{n}/10] {t}",flush=True)
print("="*156);print("TEST560 — AKBASCORE MAM · INCREMENTAL DEFERRED CONSOLIDATION");print("TEST559 WORKING BACKBONE PRESERVED · BATCH DC6 vs APPEND-ONLY INCREMENTAL DC6");print("FROZEN MISTRAL · SOURCE-FREE NUMERICAL MEMORY GROWTH · NO TRAINING / RETRIEVAL / ROUTER / QSF");print("="*156)
section(1,"Frozen Mistral...")
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",dtype=DTYPE).eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;NL=len(model.model.layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None) or H//QH
if (NL,H,QH,KVH,HD)!=(32,4096,32,8,128):raise RuntimeError(f"Architecture mismatch: {(NL,H,QH,KVH,HD)}")
print("MODEL:",MODEL_ID);print("GPU:",torch.cuda.get_device_name(0));print("TORCH:",torch.__version__,"TRANSFORMERS:",transformers.__version__);print(f"NL={NL} H={H} QH={QH} KVH={KVH} HD={HD} DTYPE={DTYPE} CUT={CUT}");print("ROPE:",getattr(cfg,"rope_parameters",None));print("TRAINABLE:",sum(int(p.requires_grad) for p in model.parameters()))
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
def replace_hidden(out,new):
    if isinstance(out,tuple):return (new,)+out[1:]
    if isinstance(out,list):return [new]+out[1:]
    return new
@torch.no_grad()
def answer_layers(layers,q):
    cache=build_cache(layers);physical=cache_len(cache);qids=tok(query_suffix(q),add_special_tokens=False).input_ids;ids=torch.tensor(qids,dtype=torch.long,device=DEVICE).reshape(1,-1);qlen=ids.shape[1]
    pos=torch.arange(physical,physical+qlen,dtype=torch.long,device=DEVICE).unsqueeze(0);att=torch.ones((1,physical+qlen),dtype=torch.long,device=DEVICE)
    out=model(input_ids=ids,past_key_values=cache,attention_mask=att,position_ids=pos,cache_position=pos.squeeze(0),use_cache=True,return_dict=True);cache=out.past_key_values;nxt=out.logits[:,-1,:].argmax(-1,keepdim=True);eos=set() if tok.eos_token_id is None else {int(tok.eos_token_id)};gen=[]
    for _ in range(MAX_NEW):
        gen.append(nxt)
        if int(nxt.item()) in eos:break
        physical=cache_len(cache);pos=torch.tensor([[physical]],dtype=torch.long,device=DEVICE);att=torch.ones((1,physical+1),dtype=torch.long,device=DEVICE)
        out=model(input_ids=nxt,past_key_values=cache,attention_mask=att,position_ids=pos,cache_position=pos.squeeze(0),use_cache=True,return_dict=True);cache=out.past_key_values;nxt=out.logits[:,-1,:].argmax(-1,keepdim=True)
    return tok.decode(torch.cat(gen,dim=1)[0],skip_special_tokens=True).strip()
def make_case(keys,target):
    ids=list(PREFIX_IDS);ranges={}
    for ci in keys:
        a=len(ids);ids.extend(CAR[ci]["body"]);ranges[ci]=(a,len(ids))
    n=len(ids);group=torch.full((n,),-1,dtype=torch.long,device=DEVICE)
    for j,ci in enumerate(keys):
        a,b=ranges[ci];group[a:b]=j
    i=torch.arange(n,device=DEVICE);causal=i[:,None]>=i[None,:];allowed=causal&((group[:,None]==-1)|(group[None,:]==-1)|(group[:,None]==group[None,:]))
    block=torch.zeros((1,1,n,n),dtype=DTYPE,device=DEVICE);block.masked_fill_(~allowed,torch.finfo(DTYPE).min);joint=torch.zeros((1,1,n,n),dtype=DTYPE,device=DEVICE);joint.masked_fill_(~causal,torch.finfo(DTYPE).min)
    full=torch.zeros(n,dtype=torch.bool,device=DEVICE);full[P:]=True
    return torch.tensor(ids,dtype=torch.long,device=DEVICE).reshape(1,-1),joint,block,full,ranges
@torch.no_grad()
def write_ref(ids,joint,block,kind="JOINT"):
    handles=[]
    try:
        for li,layer in enumerate(model.model.layers):
            def pre(module,args,kwargs,li=li):
                kwargs["attention_mask"]=joint if kind=="JOINT" else block;return args,kwargs
            handles.append(layer.register_forward_pre_hook(pre,with_kwargs=True))
        n=ids.shape[1];pos=torch.arange(n,dtype=torch.long,device=DEVICE).unsqueeze(0)
        out=model(input_ids=ids,attention_mask=torch.ones((1,n),dtype=torch.long,device=DEVICE),position_ids=pos,cache_position=pos.squeeze(0),use_cache=True,return_dict=True)
        layers=clone_layers(out.past_key_values);del out;return layers
    finally:
        for h in handles:h.remove()
def rotate_half(x):
    a,b=x.chunk(2,dim=-1);return torch.cat((-b,a),dim=-1)
@torch.no_grad()
def rope_tables(pos):
    dummy=torch.zeros((1,pos.numel(),H),dtype=DTYPE,device=DEVICE);cos,sin=model.model.rotary_emb(dummy,pos.reshape(1,-1))
    return cos.unsqueeze(1).float(),sin.unsqueeze(1).float()
@torch.no_grad()
def rephase_key(k,oldpos,newpos):
    if torch.equal(oldpos,newpos):return k
    co,so=rope_tables(oldpos);cn,sn=rope_tables(newpos);x=k.float();unrot=x*co-rotate_half(x)*so
    return (unrot*cn+rotate_half(unrot)*sn).to(k.dtype)
@torch.no_grad()
def forge_checkpoint(ci,cut=CUT):
    ids=PREFIX_IDS+CAR[ci]["body"];n=len(ids);x=torch.tensor(ids,dtype=torch.long,device=DEVICE).reshape(1,-1);pos=torch.arange(n,dtype=torch.long,device=DEVICE);state={};handles=[]
    try:
        def cap(module,args,out):state["h"]=(out[0] if isinstance(out,(tuple,list)) else out).detach().clone()
        handles.append(model.model.layers[cut].register_forward_hook(cap))
        out=model(input_ids=x,attention_mask=torch.ones((1,n),dtype=torch.long,device=DEVICE),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        early=tuple((k.detach().clone(),v.detach().clone()) for k,v in cache_layers(out.past_key_values)[:cut+1]);h=state["h"];del out
        return {"h":h,"kv":early,"positions":pos.clone(),"body_len":n-P}
    finally:
        for h in handles:h.remove()
@torch.no_grad()
def batch_dc6(keys,ranges,joint):
    cps={ci:forge_checkpoint(ci) for ci in keys};n=joint.shape[-1];assembled=torch.cat([cps[keys[0]]["h"][:,:P,:]]+[cps[ci]["h"][:,P:,:] for ci in keys],dim=1);early=[]
    if assembled.shape!=(1,n,H):raise RuntimeError("Batch H6 assembly FAIL.")
    for li in range(CUT+1):
        ks=[cps[keys[0]]["kv"][li][0][:,:,:P,:]];vs=[cps[keys[0]]["kv"][li][1][:,:,:P,:]]
        for ci in keys:
            a,b=ranges[ci];cp=cps[ci];k=cp["kv"][li][0][:,:,P:,:];v=cp["kv"][li][1][:,:,P:,:];ks.append(rephase_key(k,cp["positions"][P:],torch.arange(a,b,device=DEVICE)));vs.append(v)
        early.append((torch.cat(ks,dim=-2),torch.cat(vs,dim=-2)))
    handles=[]
    try:
        def inject(module,args,out):return replace_hidden(out,assembled)
        handles.append(model.model.layers[CUT].register_forward_hook(inject))
        for li,layer in enumerate(model.model.layers):
            if li<=CUT:continue
            def pre(module,args,kwargs,li=li):kwargs["attention_mask"]=joint;return args,kwargs
            handles.append(layer.register_forward_pre_hook(pre,with_kwargs=True))
        zeros=torch.zeros((1,n,H),dtype=DTYPE,device=DEVICE);pos=torch.arange(n,device=DEVICE)
        out=model(inputs_embeds=zeros,attention_mask=torch.ones((1,n),dtype=torch.long,device=DEVICE),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True);gen=clone_layers(out.past_key_values);result=tuple(early)+gen[CUT+1:];del out,gen,zeros,cps
        return result
    finally:
        for h in handles:h.remove()
@torch.no_grad()
def init_incremental(ci):
    cp=forge_checkpoint(ci);body=CAR[ci]["body"];n=P+len(body);assembled=cp["h"];early=[]
    for li in range(CUT+1):early.append((cp["kv"][li][0].detach().clone(),cp["kv"][li][1].detach().clone()))
    handles=[]
    try:
        def inject(module,args,out):return replace_hidden(out,assembled)
        handles.append(model.model.layers[CUT].register_forward_hook(inject))
        zeros=torch.zeros((1,n,H),dtype=DTYPE,device=DEVICE);pos=torch.arange(n,device=DEVICE)
        out=model(inputs_embeds=zeros,attention_mask=torch.ones((1,n),dtype=torch.long,device=DEVICE),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True);gen=clone_layers(out.past_key_values);result=tuple(early)+gen[CUT+1:];del out,gen,zeros,cp
        return result
    finally:
        for h in handles:h.remove()
@torch.no_grad()
def append_incremental(memory,ci):
    cp=forge_checkpoint(ci);oldn=memory[0][0].shape[-2];q=cp["body_len"];newpos=torch.arange(oldn,oldn+q,dtype=torch.long,device=DEVICE);body_h=cp["h"][:,P:,:]
    if body_h.shape!=(1,q,H):raise RuntimeError("Incremental H6 body mismatch.")
    cache=build_cache(memory);handles=[]
    try:
        def inject(module,args,out):
            h=out[0] if isinstance(out,(tuple,list)) else out
            if h.shape!=body_h.shape:raise RuntimeError(f"Incremental H6 injection mismatch {h.shape} != {body_h.shape}")
            return replace_hidden(out,body_h)
        handles.append(model.model.layers[CUT].register_forward_hook(inject))
        zeros=torch.zeros((1,q,H),dtype=DTYPE,device=DEVICE);att=torch.ones((1,oldn+q),dtype=torch.long,device=DEVICE)
        out=model(inputs_embeds=zeros,past_key_values=cache,attention_mask=att,position_ids=newpos.unsqueeze(0),cache_position=newpos,use_cache=True,return_dict=True);grown=list(clone_layers(out.past_key_values))
        for li in range(CUT+1):
            oldk,oldv=memory[li];k=cp["kv"][li][0][:,:,P:,:];v=cp["kv"][li][1][:,:,P:,:];k=rephase_key(k,cp["positions"][P:],newpos)
            grown[li]=(torch.cat((oldk,k),dim=-2),torch.cat((oldv,v),dim=-2))
        result=tuple(grown);del out,grown,zeros,cache,cp
        if result[0][0].shape[-2]!=oldn+q:raise RuntimeError("Incremental cache growth FAIL.")
        return result
    finally:
        for h in handles:h.remove()
@torch.no_grad()
def incremental_dc6(keys):
    memory=init_incremental(keys[0])
    for ci in keys[1:]:
        new=append_incremental(memory,ci);del memory;memory=new
    return memory
section(3,"Numerical checkpoint and incremental contract...")
z=PANEL[0];keys=order_keys(z,"MIDDLE");ids,joint,block,full,ranges=make_case(keys,z["target"]);cp=forge_checkpoint(keys[0])
if len(cp["kv"])!=CUT+1 or cp["h"].shape[-1]!=H:raise RuntimeError("Checkpoint contract FAIL.")
m=init_incremental(keys[0]);expected=P+len(CAR[keys[0]]["body"])
if cache_len(build_cache(m))!=expected:raise RuntimeError("Incremental init FAIL.")
m2=append_incremental(m,keys[1]);expected+=len(CAR[keys[1]]["body"])
if cache_len(build_cache(m2))!=expected:raise RuntimeError("Incremental append FAIL.")
print("CHECKPOINT CONTRACT: PASS");print("INCREMENTAL INIT: PASS");print("INCREMENTAL APPEND: PASS");print(f"PREFIX={P} CUT={CUT}")
del cp,m,m2,ids,joint,block,full,ranges;torch.cuda.empty_cache();gc.collect()
section(4,"TEST559 baseline replication...")
BASELINES=[]
for ci,z in enumerate(PANEL):
    typ=CAR[z["target"]]["type"];q=next(x[1] for x in W[ci]["queries"] if x[0]==typ);gold=CAR[z["target"]]["gold"]
    for slot in POSITIONS:
        keys=order_keys(z,slot);ids,joint,block,_,_=make_case(keys,z["target"]);jl=write_ref(ids,joint,block,"JOINT");il=write_ref(ids,joint,block,"INDEP");ja=answer_layers(jl,q);ia=answer_layers(il,q)
        BASELINES.append({"case":ci,"slot":slot,"joint":bool(hit(ja,gold)),"indep":bool(hit(ia,gold))});del jl,il,ids,joint,block
BJ={s:sum(int(x["joint"]) for x in BASELINES if x["slot"]==s) for s in POSITIONS};BI={s:sum(int(x["indep"]) for x in BASELINES if x["slot"]==s) for s in POSITIONS}
print("JOINT:",BJ);print("INDEP:",BI)
if BJ!={"FIRST":24,"MIDDLE":24,"LAST":24} or BI!={"FIRST":4,"MIDDLE":3,"LAST":6}:raise RuntimeError("Baseline replication FAIL.")
print("BASELINE REPLICATION: PASS")
section(5,"Batch vs incremental deferred consolidation...")
RESULTS=[]
for ci,z in enumerate(PANEL):
    typ=CAR[z["target"]]["type"];q=next(x[1] for x in W[ci]["queries"] if x[0]==typ);gold=CAR[z["target"]]["gold"]
    for slot in POSITIONS:
        keys=order_keys(z,slot);ids,joint,block,_,ranges=make_case(keys,z["target"])
        arms={"INDEP":write_ref(ids,joint,block,"INDEP"),"JOINT":write_ref(ids,joint,block,"JOINT")}
        arms["BATCH_DC6"]=batch_dc6(keys,ranges,joint);arms["INCR_DC6"]=incremental_dc6(keys)
        lens={a:cache_len(build_cache(v)) for a,v in arms.items()}
        if len(set(lens.values()))!=1:raise RuntimeError(f"Cache length mismatch: {lens}")
        for arm,layers in arms.items():
            ans=answer_layers(layers,q);RESULTS.append({"case":ci,"slot":slot,"arm":arm,"gold":gold,"answer":ans,"ok":bool(hit(ans,gold))})
        del arms,ids,joint,block,ranges
    print(f"CASE {ci:02d} complete",flush=True)
    if (ci+1)%4==0:torch.cuda.empty_cache();gc.collect();print(f"PROGRESS {ci+1}/24 | elapsed={time.perf_counter()-T0:.1f}s",flush=True)
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
    S[arm]={slot:sum(int(r["ok"]) for r in RESULTS if r["arm"]==arm and r["slot"]==slot) for slot in POSITIONS};S[arm]["TOTAL"]=sum(S[arm][s] for s in POSITIONS)
    print(f"{arm:18s} FIRST={S[arm]['FIRST']:02d}/24 MIDDLE={S[arm]['MIDDLE']:02d}/24 LAST={S[arm]['LAST']:02d}/24 TOTAL={S[arm]['TOTAL']:02d}/72")
section(8,"Incremental identity and gain/loss...")
MAP={arm:{(r["case"],r["slot"]):r for r in RESULTS if r["arm"]==arm} for arm in ARMS};DIAG={}
for arm in ["BATCH_DC6","INCR_DC6"]:
    gain=sum(int(not MAP["INDEP"][k]["ok"] and MAP[arm][k]["ok"]) for k in MAP["INDEP"]);loss=sum(int(MAP["INDEP"][k]["ok"] and not MAP[arm][k]["ok"]) for k in MAP["INDEP"])
    ansj=sum(int(norm(MAP[arm][k]["answer"])==norm(MAP["JOINT"][k]["answer"])) for k in MAP["JOINT"]);ansb=sum(int(norm(MAP[arm][k]["answer"])==norm(MAP["BATCH_DC6"][k]["answer"])) for k in MAP["BATCH_DC6"])
    DIAG[arm]={"gain":gain,"loss":loss,"net":gain-loss,"answer_identity_joint":ansj,"answer_identity_batch":ansb}
    print(f"{arm:18s} GAIN={gain:02d} LOSS={loss:02d} NET={gain-loss:+03d} | ANS-ID-JOINT={ansj:02d}/72 ANS-ID-BATCH={ansb:02d}/72")
section(9,"Engineering decision...")
gates={"JOINT":S["JOINT"]["TOTAL"]==72,"INDEP":S["INDEP"]["TOTAL"]==13,"BATCH_DC6":S["BATCH_DC6"]["TOTAL"]==72}
for k,v in gates.items():print(f"GATE {k}: {'PASS' if v else 'FAIL'}")
inc=S["INCR_DC6"]["TOTAL"];batch=S["BATCH_DC6"]["TOTAL"]
if not all(gates.values()):DECISION="INCONCLUSIVE — REFERENCE GATE FAILURE"
elif inc==72:DECISION="FULL INCREMENTAL CONSOLIDATION — APPEND-ONLY MEMORY MATCHES BATCH ACCURACY 72/72"
elif inc>=68:DECISION=f"STRONG PARTIAL INCREMENTAL CONSOLIDATION — {inc}/72, GAP TO BATCH={batch-inc}"
elif inc>13:DECISION=f"PARTIAL INCREMENTAL CONSOLIDATION — {inc}/72, GAP TO BATCH={batch-inc}"
else:DECISION="INCREMENTAL CONSOLIDATION FAILED TO IMPROVE INDEPENDENT BASELINE"
print("BATCH_DC6:",batch,"/72");print("INCR_DC6 :",inc,"/72");print("DECISION:",DECISION)
print("CAUTION: Incremental append exploits causal ordering: previously consolidated tokens are not recomputed when a later numerical cartridge is appended.")
print("CAUTION: H6 residual + L0–L6 KV remains an augmented numerical cartridge; strict KV-only sufficiency is not claimed.")
section(10,"Final research record...")
WEIGHT_OK=sentinel()==S0;TRAINABLE=sum(int(p.requires_grad) for p in model.parameters())
LOCK={"test":TEST,"model":MODEL_ID,"panel_sha":PANEL_SHA,"seed":SEED,"panel_seed":PANEL_SEED,"arms":ARMS,"cut":CUT,"checkpoint":"independent H6 residual + L0-L6 KV","batch_reference":"TEST559 DC6-CANON","incremental":"append one independent numerical cartridge without recomputing prior consolidated tokens","source_replay_during_consolidation":False,"training":False,"retrieval":False,"router":False,"qsf":False}
LOCK_SHA=sha_obj(LOCK);FINAL={"test":TEST,"lock_sha":LOCK_SHA,"panel_sha":PANEL_SHA,"scores":S,"diag":DIAG,"gates":gates,"decision":DECISION,"weight_ok":WEIGHT_OK,"trainable":TRAINABLE};RESULT_SHA=sha_obj(FINAL)
print("="*156);print("TEST560 — FINAL RESEARCH RECORD");print("="*156)
print("MODEL                  :",MODEL_ID);print("PANEL SHA              :",PANEL_SHA);print("LOCK SHA               :",LOCK_SHA);print("RESULT SHA             :",RESULT_SHA)
for arm in ARMS:print(f"{arm:24s}: {S[arm]}")
print("REFERENCE GATES        :",gates);print("ENGINEERING DECISION   :",DECISION);print("WEIGHT SENTINEL        :","PASS" if WEIGHT_OK else "FAIL");print("TRAINABLE              :",TRAINABLE);print("TOTAL TIME             :",f"{time.perf_counter()-T0:.2f}s")
print("INTERPRETATION         : Tests whether independently forged numerical H6 cartridges can be appended sequentially to an existing consolidated memory without rebuilding prior memories.")
print("NEXT TEST              : TEST561 only after TEST560 interpretation");print("="*156)
if not WEIGHT_OK:raise RuntimeError("WEIGHT SENTINEL FAILURE.")
if TRAINABLE!=0:raise RuntimeError("TRAINABLE PARAMETER FAILURE.")
