
# TEST649 — AKBASCORE MAM · NATIVE RELATIONAL CONSOLIDATION
# LOCKED TEST566 ENGINE · CUT3 · JOINT / INDEP / DC_AB / DC_BA / DC_BLOCK
# TWO-SOURCE RELATIONAL INFERENCE · COUNTERFACTUAL CAUSAL CONTROLS
# FROZEN QWEN · NO TRAINING · NO EXTERNAL ANSWER FUSION · SINGLE CELL
import os,sys,re,json,time,random,hashlib,urllib.request,traceback,gc
from pathlib import Path
import torch,numpy as np
TEST="649";SEED649=6492026;N649=12;CUT649=3;MAX_NEW649=24
BASE_URL649="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/566.py"
BASE_BLOB649="b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71"
BOUND649="# TEST566 ADDITIONS — original TEST564 checkpoint/init/append/answer remain unchanged."
ROOT649=Path("/content/AKBASCORE_TEST649");ROOT649.mkdir(parents=True,exist_ok=True)
def digest649(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
def gitblob649(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def gate649(name,ok):print(("PASS" if ok else "FAIL"),name,flush=True);assert ok,name
print("="*124);print("TEST649 — NATIVE RELATIONAL CONSOLIDATION · FROZEN QWEN · CUT3");print("="*124)
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU REQUIRED")
local649=Path("/content/566.py")
if local649.is_file() and gitblob649(local649.read_bytes())==BASE_BLOB649:raw649=local649.read_bytes();origin649="LOCAL_VERIFIED"
else:
    req649=urllib.request.Request(BASE_URL649,headers={"User-Agent":"AKBASCORE-TEST649"})
    with urllib.request.urlopen(req649,timeout=120) as f:raw649=f.read()
    origin649="GITHUB_VERIFIED"
gate649("TEST566_SOURCE_BLOB",gitblob649(raw649)==BASE_BLOB649)
src649=raw649.decode("utf-8");gate649("BASELINE_BOUNDARY",src649.count(BOUND649)==1)
exec(compile(src649.split(BOUND649,1)[0],"566.py","exec"),globals())
gate649("FROZEN_BACKBONE",all(not p.requires_grad for p in model.parameters()))
gate649("MODEL_ARCH",(NL,H,QH,KVH,HD)==(28,3584,28,4,128))
gate649("BASELINE_PANEL",PANEL_SHA=="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45")
gate649("BASELINE_SENTINEL",sentinel()==S0)
print("SOURCE",origin649,"BLOB",gitblob649(raw649),"GPU",torch.cuda.get_device_name(0))
# The verified TEST566 functions above remain untouched. Only the experimental cartridge panel changes.
rng649=random.Random(SEED649)
alphabet649="ABCDEFGHJKLMNPQRSTUVWXYZ"
def word649(n=9):return "".join(rng649.choice(alphabet649) for _ in range(n)).capitalize()
used649=set()
def unique649():
    while True:
        x=word649()
        if x not in used649:used649.add(x);return x
PANEL649=[]
for i in range(N649):
    obj,person,other,place,place2=(unique649() for _ in range(5))
    a=f"The owner of artifact {obj} is {person}."
    b=f"{person} lives in region {place}."
    b2=f"{person} lives in region {place2}."
    distract=f"The owner of artifact {unique649()} is {other}."
    q=f"In which region does the owner of artifact {obj} live?"
    PANEL649.append(dict(case=i,object=obj,person=person,region=place,region_cf=place2,question=q,A=a,B=b,B2=b2,D=distract))
SEAL649=digest649(PANEL649)
print("RELATIONAL_PANEL_SHA256",SEAL649,"CASES",N649)
CAR649=[]
def add649(f,case,kind):
    ci=len(CAR649);ids=tok(source(f),add_special_tokens=False).input_ids
    if ids[:P]!=PIDS:raise RuntimeError("CARTRIDGE_PREFIX_MISMATCH")
    CAR649.append(dict(fact=f,body=ids[P:],case=case,kind=kind))
    return ci
for z in PANEL649:
    z["a_id"]=add649(z["A"],z["case"],"A")
    z["b_id"]=add649(z["B"],z["case"],"B")
    z["b2_id"]=add649(z["B2"],z["case"],"B2")
    z["d_id"]=add649(z["D"],z["case"],"DISTRACTOR")
CAR=CAR649
gate649("SEALED_PANEL_UNCHANGED",digest649([{k:v for k,v in z.items() if k not in ("a_id","b_id","b2_id","d_id")} for z in PANEL649])==SEAL649)
MAX_NEW=MAX_NEW649
def normal649(s):return re.sub(r"[^a-z]","",s.casefold())
def exact649(s,gold):return normal649(s)==normal649(gold)
def sourcefree649(question,source_bodies):
    ids=tok(suffix(question),add_special_tokens=False).input_ids
    for body in source_bodies:
        if len(ids)>=len(body) and any(ids[j:j+len(body)]==body for j in range(len(ids)-len(body)+1)):return False
    return True
@torch.no_grad()
def joint649(keys):
    ids=list(PIDS)
    for ci in keys:ids.extend(CAR[ci]["body"])
    n=len(ids);pos=torch.arange(n,device=DEVICE)
    out=model(input_ids=torch.tensor([ids],device=DEVICE),attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
    return clone(out.past_key_values)
@torch.no_grad()
def indep649(keys):
    memory=None
    for ci in keys:
        cp=checkpoint(ci,CUT649);kv=cp["kv"]
        if memory is None:memory=kv;continue
        oldn=memory[0][0].shape[-2];q=len(CAR[ci]["body"])
        oldpos=torch.arange(P,P+q,device=DEVICE);newpos=torch.arange(oldn,oldn+q,device=DEVICE)
        grown=[]
        for (ok,ov),(k,v) in zip(memory,kv):
            grown.append((torch.cat((ok,rephase(k[:,:,P:,:],oldpos,newpos)),-2),torch.cat((ov,v[:,:,P:,:]),-2)))
        for (ok,ov),(nk,nv) in zip(memory,grown):
            if not torch.equal(ok,nk[:,:,:oldn,:]) or not torch.equal(ov,nv[:,:,:oldn,:]):raise RuntimeError("INDEP_PREFIX_CHANGED")
        memory=tuple(grown)
    return memory
@torch.no_grad()
def dc649(keys):
    memory=init(keys[0],CUT649)
    for ci in keys[1:]:memory=append(memory,ci,CUT649)
    return memory
@torch.no_grad()
def blocked649(keys):
    # TEST575 BATCH_BLOCK3 principle: same independent H3, but no cross-cartridge upper-layer attention.
    cps=[checkpoint(ci,CUT649) for ci in keys]
    lengths=[len(CAR[ci]["body"]) for ci in keys];n=P+sum(lengths)
    ranges=[];offset=P
    for q in lengths:ranges.append((offset,offset+q));offset+=q
    h3=torch.cat([cps[0]["h"][:,:P,:]]+[cp["h"][:,P:,:] for cp in cps],dim=1)
    gate649("BLOCK_H3_SHAPE",tuple(h3.shape)==(1,n,H))
    ix=torch.arange(n,device=DEVICE);causal=ix[:,None]>=ix[None,:]
    group=torch.full((n,),-1,device=DEVICE,dtype=torch.long)
    for j,(a,b) in enumerate(ranges):group[a:b]=j
    allow=causal&((group[:,None]==-1)|(group[None,:]==-1)|(group[:,None]==group[None,:]))
    mask=torch.zeros((1,1,n,n),device=DEVICE,dtype=DTYPE)
    mask.masked_fill_(~allow,torch.finfo(DTYPE).min)
    lower=[]
    for li in range(CUT649+1):
        ks=[cps[0]["kv"][li][0][:,:,:P,:]];vs=[cps[0]["kv"][li][1][:,:,:P,:]]
        for cp,(a,b) in zip(cps,ranges):
            k,v=cp["kv"][li]
            ks.append(rephase(k[:,:,P:,:],cp["pos"][P:],torch.arange(a,b,device=DEVICE)))
            vs.append(v[:,:,P:,:])
        lower.append((torch.cat(ks,-2),torch.cat(vs,-2)))
    seen=[];handles=[]
    def inject(module,args,out):
        current=out[0] if isinstance(out,(tuple,list)) else out
        if current.shape!=h3.shape:raise RuntimeError("BLOCK_INJECTION_SHAPE")
        return replace(out,h3)
    try:
        handles.append(model.model.layers[CUT649].register_forward_hook(inject))
        for li in range(CUT649+1,NL):
            def pre(module,args,kwargs,li=li):
                kwargs["attention_mask"]=mask;seen.append(li)
                return args,kwargs
            handles.append(model.model.layers[li].register_forward_pre_hook(pre,with_kwargs=True))
        pos=torch.arange(n,device=DEVICE)
        out=model(inputs_embeds=torch.zeros((1,n,H),device=DEVICE,dtype=DTYPE),attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        upper=clone(out.past_key_values)
    finally:
        for handle in handles:handle.remove()
    if seen!=list(range(CUT649+1,NL)):raise RuntimeError("BLOCK_MASK_COVERAGE")
    return tuple(lower)+upper[CUT649+1:]
@torch.no_grad()
def none649(q):
    ids=tok(suffix(q),add_special_tokens=False).input_ids
    x=torch.tensor([ids],device=DEVICE);pos=torch.arange(len(ids),device=DEVICE)
    out=model(input_ids=x,attention_mask=torch.ones_like(x),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
    cache=out.past_key_values;next_id=out.logits[:,-1,:].argmax(-1,keepdim=True);gen=[]
    stop={tok.eos_token_id,tok.convert_tokens_to_ids("<|im_end|>")}
    for _ in range(MAX_NEW649):
        t=int(next_id.item())
        if t in stop:break
        gen.append(t);n=cache.get_seq_length();pos=torch.tensor([n],device=DEVICE)
        out=model(input_ids=next_id,past_key_values=cache,attention_mask=torch.ones((1,n+1),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        cache=out.past_key_values;next_id=out.logits[:,-1,:].argmax(-1,keepdim=True)
    return tok.decode(gen,skip_special_tokens=True).strip()
def valid649(memory,keys):
    n=P+sum(len(CAR[ci]["body"]) for ci in keys)
    if len(memory)!=NL:raise RuntimeError("CACHE_LAYER_COUNT")
    for k,v in memory:
        if k.shape!=(1,KVH,n,HD) or v.shape!=(1,KVH,n,HD):raise RuntimeError("CACHE_SHAPE")
        if not torch.isfinite(k).all() or not torch.isfinite(v).all():raise RuntimeError("CACHE_NONFINITE")
    return True
ARMS649=("NONE","A_ONLY","B_ONLY","JOINT_AB","JOINT_BA","INDEP_AB","DC_AB","DC_BA","DC_BLOCK","DC_B2","JOINT_B2","DC_WRONG")
RAW649=[];T649=time.perf_counter()
def run649(z,arm):
    a,b,b2,d=z["a_id"],z["b_id"],z["b2_id"],z["d_id"]
    if arm=="NONE":return none649(z["question"]),0
    mode,keys={
        "A_ONLY":("DC",[a]),"B_ONLY":("DC",[b]),
        "JOINT_AB":("JOINT",[a,b]),"JOINT_BA":("JOINT",[b,a]),
        "INDEP_AB":("INDEP",[a,b]),"DC_AB":("DC",[a,b]),"DC_BA":("DC",[b,a]),
        "DC_BLOCK":("BLOCK",[a,b]),"DC_B2":("DC",[a,b2]),
        "JOINT_B2":("JOINT",[a,b2]),"DC_WRONG":("DC",[a,d])
    }[arm]
    if not sourcefree649(z["question"],[CAR[i]["body"] for i in keys]):raise RuntimeError("SOURCE_REPLAY_IN_QUESTION")
    mem={"DC":dc649,"JOINT":joint649,"INDEP":indep649,"BLOCK":blocked649}[mode](keys)
    valid649(mem,keys);n=mem[0][0].shape[-2]
    result=answer(mem,z["question"])
    del mem
    return result,n
print("="*124);print("RUNNING",N649,"SEALED CASES ×",len(ARMS649),"ARMS");print("="*124)
for z in PANEL649:
    for arm in ARMS649:
        t=time.perf_counter()
        try:
            ans,n=run649(z,arm);status="OK";err=None
        except Exception as e:
            ans="";n=0;status="ERROR";err=f"{type(e).__name__}: {e}"
        gold=z["region_cf"] if arm in ("DC_B2","JOINT_B2") else z["region"]
        rec=dict(case=z["case"],arm=arm,status=status,answer=ans,gold=gold,exact=bool(status=="OK" and exact649(ans,gold)),cache_tokens=n,seconds=round(time.perf_counter()-t,3),error=err)
        RAW649.append(rec)
        print(f"CASE {z['case']:02d} ARM {arm:<10} {status:<5} {'PASS' if rec['exact'] else 'FAIL'} GOLD={gold} ANSWER={ans!r}",flush=True)
        if err:print("ERROR",err,flush=True)
    gc.collect();torch.cuda.empty_cache()
print("="*124);print("SCORING — ONLY AFTER ALL MODEL OUTPUTS HAVE BEEN GENERATED");print("="*124)
SCORES649={}
for arm in ARMS649:
    rows=[r for r in RAW649 if r["arm"]==arm]
    SCORES649[arm]=dict(correct=sum(r["exact"] for r in rows),total=len(rows),errors=sum(r["status"]=="ERROR" for r in rows))
    print(f"{arm:<12} {SCORES649[arm]['correct']:02d}/{len(rows):02d} ERRORS={SCORES649[arm]['errors']}")
CAUSAL649=[]
for z in PANEL649:
    rows={r["arm"]:r for r in RAW649 if r["case"]==z["case"]}
    normal=lambda arm:normal649(rows[arm]["answer"])
    dc_ok=rows["DC_AB"]["exact"];cf_ok=rows["DC_B2"]["exact"]
    controls=all(not rows[a]["exact"] for a in ("NONE","A_ONLY","B_ONLY","DC_WRONG"))
    flip=normal("DC_AB")!=normal("DC_B2")
    joint_ok=rows["JOINT_AB"]["exact"]
    causal=bool(dc_ok and cf_ok and controls and flip)
    CAUSAL649.append(dict(case=z["case"],joint_ceiling=joint_ok,dc_correct=dc_ok,counterfactual_correct=cf_ok,negative_controls=controls,answer_flip=flip,causal_pass=causal))
    print(f"CAUSAL CASE={z['case']:02d} JOINT={joint_ok} DC={dc_ok} CF={cf_ok} CONTROLS={controls} FLIP={flip} SYNTHESIS={causal}")
CAUSAL_COUNT649=sum(x["causal_pass"] for x in CAUSAL649)
ERRORS649=[r for r in RAW649 if r["status"]=="ERROR"]
INTEGRITY649=dict(source_blob=gitblob649(raw649)==BASE_BLOB649,panel_sealed=digest649([{k:v for k,v in z.items() if k not in ("a_id","b_id","b2_id","d_id")} for z in PANEL649])==SEAL649,model_frozen=all(not p.requires_grad for p in model.parameters()),sentinel_unchanged=sentinel()==S0,optimizer_absent=True,errors=len(ERRORS649))
REPORT649=dict(test=TEST,source=BASE_URL649,source_blob=BASE_BLOB649,seed=SEED649,panel_sha256=SEAL649,cut=CUT649,max_new=MAX_NEW649,cases=N649,arms=list(ARMS649),scores=SCORES649,causal=CAUSAL649,causal_pass=CAUSAL_COUNT649,integrity=INTEGRITY649,raw=RAW649,panel=PANEL649,elapsed_seconds=round(time.perf_counter()-T649,2))
out649=ROOT649/"TEST649_results.json";out649.write_text(json.dumps(REPORT649,indent=2,ensure_ascii=False),encoding="utf-8")
print("="*124)
print("TEST649_FINAL")
print("PANEL_SHA256",SEAL649)
print("CAUSAL_RELATIONAL_SYNTHESIS",f"{CAUSAL_COUNT649}/{N649}")
print("INTEGRITY",INTEGRITY649)
print("RESULT_FILE",out649)
print("RESULT_SHA256",hashlib.sha256(out649.read_bytes()).hexdigest())
print("TEST649_EXECUTION_STATUS","PASS" if not ERRORS649 and all([INTEGRITY649["source_blob"],INTEGRITY649["panel_sealed"],INTEGRITY649["model_frozen"],INTEGRITY649["sentinel_unchanged"]]) else "FAIL")
print("TEST649_SYNTHESIS_STATUS","OBSERVED" if CAUSAL_COUNT649 else "NOT_DEMONSTRATED")
print("="*124)
