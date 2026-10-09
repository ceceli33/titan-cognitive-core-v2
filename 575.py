
# TEST575 — AKBASCORE MAM · QWEN FINAL COUNTERFACTUAL & DEMO LOCK
# VERIFIED TEST566 ENGINE · CUT3 · 24 WORLDS × 3 SLOTS × 5 ARMS
# JOINT / NATIVE_INDEP / INCR_DC3 / BATCH_DC3 / BATCH_BLOCK3
# NO TRAINING · NO RETRIEVAL · NO ROUTER · NO SOURCE REPLAY AT ASK
import os,sys,subprocess,importlib.util,urllib.request,hashlib,json,time,gc
for m in ("torch","transformers","accelerate"):
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",m])
import torch,numpy as np,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
BASE_FILE="566.py";BASE_BLOB="b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71"
BASE_URL="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/566.py"
EXPECTED_PANEL_SHA="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required")
print("="*140);print("TEST575 — AKBASCORE MAM · QWEN FINAL COUNTERFACTUAL & DEMO LOCK");print("="*140)
req=urllib.request.Request(BASE_URL,headers={"User-Agent":"AKBASCORE-TEST575"})
with urllib.request.urlopen(req,timeout=90) as f:raw=f.read()
blob=hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
if blob!=BASE_BLOB:raise RuntimeError("BASELINE BLOB MISMATCH: "+blob)
src=raw.decode("utf-8");marker="# TEST566 ADDITIONS — original TEST564 checkpoint/init/append/answer remain unchanged."
if src.count(marker)!=1:raise RuntimeError("BASELINE BOUNDARY MISMATCH")
exec(compile(src.split(marker,1)[0],BASE_FILE,"exec"),globals())
# TEST575 CONFIGURATION — DEFINED AFTER BASELINE TO PREVENT GLOBAL OVERWRITES
TEST="575";CUT=3;CASES=list(range(24));SLOTS=["FIRST","MIDDLE","LAST"]
ARMS=["JOINT","NATIVE_INDEP","INCR_DC3","BATCH_DC3","BATCH_BLOCK3"]
EXPECTED_WEIGHT_SHA="b441b005d826d45f2778291696ac5ae22dfb3421defb37c539904d8d38f93786"
T575=time.perf_counter();CHECKS=[];RAW=[];COUNTERFACTUAL=[];ASK_AUDIT=[];ERRORS=[]
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
def check(name,condition,**extra):
    ok=bool(condition);CHECKS.append(dict(name=name,ok=ok,**extra))
    print(("PASS" if ok else "FAIL"),"|",name,"|",extra,flush=True)
    if not ok:raise RuntimeError("TEST575 GATE FAILED: "+name)
def expected_length(keys):return P+sum(len(CAR[ci]["body"]) for ci in keys)
def validate(memory,keys):
    n=expected_length(keys)
    if len(memory)!=NL:raise RuntimeError("LAYER COUNT MISMATCH")
    for li,(k,v) in enumerate(memory):
        shape=(1,KVH,n,HD)
        if tuple(k.shape)!=shape or tuple(v.shape)!=shape:raise RuntimeError(f"CACHE SHAPE L{li}: {tuple(k.shape)}/{tuple(v.shape)}")
        if not torch.isfinite(k).all() or not torch.isfinite(v).all():raise RuntimeError(f"NONFINITE CACHE L{li}")
    return n
def weight_sha():
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode());h.update(str(tuple(p.shape)).encode());h.update(str(p.dtype).encode())
        for part in p.detach().contiguous().view(torch.uint8).reshape(-1).split(16*1024*1024):
            h.update(part.cpu().numpy().tobytes())
    return h.hexdigest()
def case_layout(keys):
    ids=list(PIDS);ranges={}
    for ci in keys:
        a=len(ids);ids.extend(CAR[ci]["body"]);ranges[ci]=(a,len(ids))
    n=len(ids);ix=torch.arange(n,device=DEVICE);causal=ix[:,None]>=ix[None,:]
    group=torch.full((n,),-1,device=DEVICE,dtype=torch.long)
    for i,ci in enumerate(keys):
        a,b=ranges[ci];group[a:b]=i
    isolated=causal&((group[:,None]==-1)|(group[None,:]==-1)|(group[:,None]==group[None,:]))
    jm=torch.zeros((1,1,n,n),device=DEVICE,dtype=DTYPE);bm=torch.zeros_like(jm)
    bad=torch.finfo(DTYPE).min;jm.masked_fill_(~causal,bad);bm.masked_fill_(~isolated,bad)
    return ids,ranges,jm,bm
def verify_masks(keys,ranges,jm,bm):
    n=jm.shape[-1];ix=torch.arange(n,device=DEVICE);causal=ix[:,None]>=ix[None,:]
    if not torch.equal(jm[0,0]==0,causal):raise RuntimeError("JOINT MASK GEOMETRY")
    if not torch.equal(bm[0,0,:,0:P],jm[0,0,:,0:P]):raise RuntimeError("PREFIX MASK GEOMETRY")
    for ci in keys:
        a,b=ranges[ci]
        if not torch.equal(bm[0,0,a:b,a:b]==0,causal[a:b,a:b]):raise RuntimeError("SELF MASK GEOMETRY")
        for cj in keys:
            if ci==cj:continue
            c,d=ranges[cj]
            if torch.any(bm[0,0,a:b,c:d]==0):raise RuntimeError("CROSS MASK GEOMETRY")
@torch.no_grad()
def native_joint(keys):
    ids=list(PIDS)
    for ci in keys:ids.extend(CAR[ci]["body"])
    n=len(ids);pos=torch.arange(n,device=DEVICE)
    out=model(input_ids=torch.tensor([ids],device=DEVICE),attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
    mem=clone(out.past_key_values);validate(mem,keys)
    return mem
@torch.no_grad()
def native_independent(keys):
    memory=None;oldn=0
    for j,ci in enumerate(keys):
        cp=checkpoint(ci,0);kv=cp["kv"]
        if j==0:
            memory=kv;oldn=kv[0][0].shape[-2];continue
        q=len(CAR[ci]["body"]);oldpos=torch.arange(P,P+q,device=DEVICE);newpos=torch.arange(oldn,oldn+q,device=DEVICE);grown=[]
        for (ok,ov),(k,v) in zip(memory,kv):
            kk=rephase(k[:,:,P:,:],oldpos,newpos);vv=v[:,:,P:,:]
            grown.append((torch.cat((ok,kk),-2),torch.cat((ov,vv),-2)))
        result=tuple(grown)
        for li,((ok,ov),(nk,nv)) in enumerate(zip(memory,result)):
            if not torch.equal(ok,nk[:,:,:oldn,:]) or not torch.equal(ov,nv[:,:,:oldn,:]):raise RuntimeError(f"NATIVE INDEP BITWISE L{li}")
        memory=result;oldn+=q
    validate(memory,keys)
    return memory
@torch.no_grad()
def incremental(keys):
    memory=init(keys[0],CUT)
    for ci in keys[1:]:
        old=memory;oldn=old[0][0].shape[-2];memory=append(old,ci,CUT)
        for li,((ok,ov),(nk,nv)) in enumerate(zip(old,memory)):
            if not torch.equal(ok,nk[:,:,:oldn,:]) or not torch.equal(ov,nv[:,:,:oldn,:]):raise RuntimeError(f"INCR APPEND BITWISE L{li}")
    validate(memory,keys)
    return memory
@torch.no_grad()
def make_batch(keys,mask,ranges,tag):
    cps={ci:checkpoint(ci,CUT) for ci in keys};n=expected_length(keys)
    h3=torch.cat([cps[keys[0]]["h"][:,:P,:]]+[cps[ci]["h"][:,P:,:] for ci in keys],dim=1)
    if tuple(h3.shape)!=(1,n,H):raise RuntimeError(tag+" H3 ASSEMBLY SHAPE")
    early=[]
    for li in range(CUT+1):
        ks=[cps[keys[0]]["kv"][li][0][:,:,:P,:]];vs=[cps[keys[0]]["kv"][li][1][:,:,:P,:]]
        for ci in keys:
            cp=cps[ci];a,b=ranges[ci];k,v=cp["kv"][li]
            ks.append(rephase(k[:,:,P:,:],cp["pos"][P:],torch.arange(a,b,device=DEVICE)))
            vs.append(v[:,:,P:,:])
        early.append((torch.cat(ks,-2),torch.cat(vs,-2)))
    seen_h=[];seen_layers=[];seen_attn=[];handles=[]
    def inject(module,args,out):
        current=out[0] if isinstance(out,(tuple,list)) else out
        seen_h.append(tuple(current.shape)==tuple(h3.shape))
        return replace(out,h3)
    try:
        handles.append(model.model.layers[CUT].register_forward_hook(inject))
        for li in range(CUT+1,NL):
            def layer_pre(module,args,kwargs,li=li):
                kwargs["attention_mask"]=mask;seen_layers.append((li,kwargs["attention_mask"] is mask))
                return args,kwargs
            def attn_pre(module,args,kwargs,li=li):
                seen_attn.append((li,kwargs.get("attention_mask") is mask))
            handles.append(model.model.layers[li].register_forward_pre_hook(layer_pre,with_kwargs=True))
            handles.append(model.model.layers[li].self_attn.register_forward_pre_hook(attn_pre,with_kwargs=True))
        pos=torch.arange(n,device=DEVICE)
        out=model(inputs_embeds=torch.zeros((1,n,H),device=DEVICE,dtype=DTYPE),attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        upper=clone(out.past_key_values);result=tuple(early)+upper[CUT+1:]
    finally:
        for handle in handles:handle.remove()
    if len(seen_h)!=1 or not all(seen_h):raise RuntimeError(tag+" H3 INJECTION")
    if len(seen_layers)!=NL-CUT-1 or not all(x[1] for x in seen_layers):raise RuntimeError(tag+" LAYER MASK")
    if len(seen_attn)!=NL-CUT-1 or not all(x[1] for x in seen_attn):raise RuntimeError(tag+" ATTENTION MASK")
    if {x[0] for x in seen_attn}!=set(range(CUT+1,NL)):raise RuntimeError(tag+" ATTENTION COVERAGE")
    validate(result,keys)
    return result
def kv_delta(a,b):
    result=[]
    for li,((ak,av),(bk,bv)) in enumerate(zip(a,b)):
        result.append(dict(layer=li,k_max=float((ak.float()-bk.float()).abs().max().item()),v_max=float((av.float()-bv.float()).abs().max().item()),k_equal=bool(torch.equal(ak,bk)),v_equal=bool(torch.equal(av,bv))))
    return result
def ask_audited(memory,question,arm,wi,slot,keys):
    calls=[];question_ids=tok(suffix(question),add_special_tokens=False).input_ids
    def pre(module,args,kwargs):
        ids=kwargs.get("input_ids");cache=kwargs.get("past_key_values");emb=kwargs.get("inputs_embeds")
        seq=ids.detach().reshape(-1).cpu().tolist() if ids is not None else []
        calls.append(dict(ids=seq,cache=cache is not None,cache_tokens=int(cache.get_seq_length()) if cache is not None else 0,embeds=emb is not None))
    handle=model.register_forward_pre_hook(pre,with_kwargs=True)
    try:output=answer(memory,question)
    finally:handle.remove()
    if not calls:raise RuntimeError("ASK NO FORWARD")
    if calls[0]["ids"]!=question_ids:raise RuntimeError("ASK QUERY PREFIX MISMATCH")
    if not all(x["cache"] and x["cache_tokens"]>0 and not x["embeds"] for x in calls):raise RuntimeError("ASK CACHE OR EMBEDDING ERROR")
    for x in calls:
        for ci in keys:
            body=CAR[ci]["body"];ids=x["ids"]
            if len(ids)>=len(body) and any(ids[j:j+len(body)]==body for j in range(len(ids)-len(body)+1)):raise RuntimeError("SOURCE BODY REPLAY")
    ASK_AUDIT.append(dict(case=wi,slot=slot,arm=arm,forward_calls=len(calls),first_query_tokens=len(calls[0]["ids"]),max_forward_tokens=max(len(x["ids"]) for x in calls),cache_verified=True,source_replay=False))
    return output
def score(arm):
    rows=[r for r in RAW if r["arm"]==arm]
    return dict(total=sum(int(r["ok"]) for r in rows),count=len(rows),FIRST=sum(int(r["ok"]) for r in rows if r["slot"]=="FIRST"),MIDDLE=sum(int(r["ok"]) for r in rows if r["slot"]=="MIDDLE"),LAST=sum(int(r["ok"]) for r in rows if r["slot"]=="LAST"))
print("[1/7] Baseline and fingerprint...",flush=True)
check("BASE_BLOB",blob==BASE_BLOB,blob=blob)
check("PANEL_SHA",PANEL_SHA==EXPECTED_PANEL_SHA,panel_sha=PANEL_SHA)
check("MODEL_ARCH",(NL,H,QH,KVH,HD)==(28,3584,28,4,128))
check("FROZEN",not any(p.requires_grad for p in model.parameters()))
WEIGHT_BEFORE=weight_sha();SENTINEL_BEFORE=sentinel()
check("WEIGHT_SHA_REFERENCE",WEIGHT_BEFORE==EXPECTED_WEIGHT_SHA,sha=WEIGHT_BEFORE)
print("[2/7] Mask geometry...",flush=True)
for slot in SLOTS:
    keys=ordered(PANEL[0],slot);_,ranges,jm,bm=case_layout(keys);verify_masks(keys,ranges,jm,bm)
check("THREE_SLOT_MASK_GEOMETRY",True)
print("[3/7] Five-arm full-panel replication...",flush=True)
for wi in CASES:
    z=PANEL[wi];typ=CAR[z["target"]]["type"];question=W[wi][typ];gold=CAR[z["target"]]["gold"]
    for slot in SLOTS:
        keys=ordered(z,slot);_,ranges,jm,bm=case_layout(keys);verify_masks(keys,ranges,jm,bm)
        try:
            memories={}
            memories["JOINT"]=native_joint(keys)
            memories["NATIVE_INDEP"]=native_independent(keys)
            memories["INCR_DC3"]=incremental(keys)
            memories["BATCH_DC3"]=make_batch(keys,jm,ranges,"BATCH_DC3")
            memories["BATCH_BLOCK3"]=make_batch(keys,bm,ranges,"BATCH_BLOCK3")
            if set(memories)!=set(ARMS):raise RuntimeError("ARM REGISTRY MISMATCH")
            delta=kv_delta(memories["BATCH_DC3"],memories["BATCH_BLOCK3"]);upper=delta[CUT+1:]
            changed=any(x["k_max"]>0 or x["v_max"]>0 for x in upper)
            early_equal=all(x["k_equal"] and x["v_equal"] for x in delta[:CUT+1])
            if not early_equal:raise RuntimeError("COUNTERFACTUAL EARLY KV MISMATCH")
            COUNTERFACTUAL.append(dict(case=wi,slot=slot,upper_kv_changed=changed,changed_upper_layers=sum(x["k_max"]>0 or x["v_max"]>0 for x in upper),max_upper_k_diff=max(x["k_max"] for x in upper),max_upper_v_diff=max(x["v_max"] for x in upper),early_kv_identical=early_equal))
            for arm in ARMS:
                output=ask_audited(memories[arm],question,arm,wi,slot,keys)
                ok=bool(hit(output,gold));RAW.append(dict(arm=arm,case=wi,slot=slot,gold=gold,answer=output,ok=ok))
                print(f"CASE={wi:02d} SLOT={slot:6s} ARM={arm:13s} {'PASS' if ok else 'FAIL'} GOLD={gold} ANSWER={output!r}",flush=True)
            del memories,delta
        except Exception as e:
            ERRORS.append(dict(case=wi,slot=slot,error=repr(e)));print("ERROR:",wi,slot,repr(e),flush=True);raise
    torch.cuda.empty_cache();gc.collect()
print("[4/7] Scores and counterfactuals...",flush=True)
SCORES={arm:score(arm) for arm in ARMS};MAP={(r["arm"],r["case"],r["slot"]):r for r in RAW};PAIRED={}
for arm in ARMS:
    s=SCORES[arm];print(f"{arm:13s} TOTAL={s['total']:02d}/{s['count']} FIRST={s['FIRST']}/24 MIDDLE={s['MIDDLE']}/24 LAST={s['LAST']}/24")
for arm in ARMS[1:]:
    gain=loss=identity=0
    for wi in CASES:
        for slot in SLOTS:
            a=MAP[(arm,wi,slot)];j=MAP[("JOINT",wi,slot)]
            gain+=int(a["ok"] and not j["ok"]);loss+=int(not a["ok"] and j["ok"])
            identity+=int(norm(a["answer"])==norm(j["answer"]))
    PAIRED[arm]=dict(gain_vs_joint=gain,loss_vs_joint=loss,answer_identity=identity)
    print("PAIRED",arm,PAIRED[arm])
changed=sum(x["upper_kv_changed"] for x in COUNTERFACTUAL)
early_equal=sum(x["early_kv_identical"] for x in COUNTERFACTUAL)
behavior_changed=sum(norm(MAP[("BATCH_DC3",wi,slot)]["answer"])!=norm(MAP[("BATCH_BLOCK3",wi,slot)]["answer"]) for wi in CASES for slot in SLOTS)
print("UPPER KV SENSITIVITY:",changed,"/72","EARLY KV EQUALITY:",early_equal,"/72","ANSWER DIFFERENCE:",behavior_changed,"/72")
print("[5/7] Demo gates...",flush=True)
check("COMPLETE_360_ANSWERS",len(RAW)==360,count=len(RAW))
check("UNIQUE_ARM_CASE_SLOT",len(MAP)==360)
check("JOINT_72",SCORES["JOINT"]["total"]==72)
check("INCR_DC3_72",SCORES["INCR_DC3"]["total"]==72)
check("BATCH_DC3_72",SCORES["BATCH_DC3"]["total"]==72)
check("INCR_JOINT_IDENTITY",PAIRED["INCR_DC3"]["answer_identity"]==72)
check("BATCH_JOINT_IDENTITY",PAIRED["BATCH_DC3"]["answer_identity"]==72)
check("COUNTERFACTUAL_EARLY_EQUAL",early_equal==72)
check("COUNTERFACTUAL_UPPER_SENSITIVITY",changed==72,changed=changed)
check("ASK_ALL_AUDITED",len(ASK_AUDIT)==360)
check("NO_ERRORS",not ERRORS)
print("[6/7] Frozen-weight and hook integrity...",flush=True)
WEIGHT_AFTER=weight_sha();TRAINABLE=sum(int(p.requires_grad) for p in model.parameters())
check("FULL_WEIGHT_SHA_UNCHANGED",WEIGHT_BEFORE==WEIGHT_AFTER,before=WEIGHT_BEFORE,after=WEIGHT_AFTER)
check("SENTINEL_UNCHANGED",sentinel()==SENTINEL_BEFORE)
check("ZERO_TRAINABLE",TRAINABLE==0,count=TRAINABLE)
remaining=[]
for li,layer in enumerate(model.model.layers):
    for label,module in (("layer",layer),("attention",layer.self_attn)):
        if module._forward_hooks or module._forward_pre_hooks:remaining.append((li,label))
check("NO_RESIDUAL_HOOKS",not remaining,hooks=remaining)
print("[7/7] Final record...",flush=True)
PROTOCOL=dict(test=TEST,model=MODEL_ID,cut=CUT,base_file=BASE_FILE,base_blob=blob,panel_sha=PANEL_SHA,seed=SEED,panel_seed=PANEL_SEED,cases=CASES,slots=SLOTS,arms=ARMS,architecture=[NL,H,QH,KVH,HD],engine="TEST566 checkpoint/init/append/answer unchanged",joint="native concatenated source prefill",native_indep="native independent KV + RoPE rephase",incremental="CUT3 append-only consolidation",batch="CUT3 shared upper-layer attention",counterfactual="CUT3 same H3 and early KV; upper-layer cross-cartridge attention blocked",training=False,retrieval=False,router=False,source_replay_at_ask=False)
RECORD=dict(protocol=PROTOCOL,lock_sha=digest(PROTOCOL),checks=CHECKS,scores=SCORES,paired=PAIRED,counterfactual=COUNTERFACTUAL,counterfactual_answer_changes=behavior_changed,raw=RAW,ask_audit=ASK_AUDIT,errors=ERRORS,weight_before=WEIGHT_BEFORE,weight_after=WEIGHT_AFTER,trainable=TRAINABLE,gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,transformers_version=transformers.__version__,seconds=round(time.perf_counter()-T575,2))
RECORD["result_sha"]=digest(RECORD)
OUT="/content/AKBASCORE_TEST575_QWEN_FINAL_DEMO_LOCK.json"
os.makedirs(os.path.dirname(OUT),exist_ok=True)
with open(OUT,"w",encoding="utf-8") as f:json.dump(RECORD,f,ensure_ascii=False,indent=2)
print("="*140);print("TEST575 — FINAL DEMO READINESS RECORD");print("="*140)
print("BASE BLOB:",blob,"PANEL SHA:",PANEL_SHA)
for arm in ARMS:print(arm,SCORES[arm]["total"],"/72")
print("COUNTERFACTUAL UPPER KV EFFECT:",changed,"/72")
print("COUNTERFACTUAL ANSWER DIFFERENCE:",behavior_changed,"/72")
print("GATES:",sum(x["ok"] for x in CHECKS),"/",len(CHECKS))
print("WEIGHT SHA:",WEIGHT_AFTER,"TRAINABLE:",TRAINABLE)
print("LOCK SHA:",RECORD["lock_sha"]);print("RESULT SHA:",RECORD["result_sha"])
print("OUTPUT:",OUT,"SECONDS:",RECORD["seconds"])
print("NOTE: Counterfactual upper-layer mask intervention measures sensitivity, not every possible causal pathway.")
print("NOTE: Existing panel is a regression test, not an independent generalization benchmark.")
print("DECISION: DEMO GATES PASS — FREEZE VERIFIED CUT3 IMPLEMENTATION.")
print("="*140)
