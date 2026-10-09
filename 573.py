
# TEST573 — AKBASCORE MAM · QWEN CROSS-MODEL REPLICATION
# TEST566 VERIFIED MOTOR + TEST560 FOUR-ARM PROTOCOL · CUT3
# SAME 24 WORLDS / 72 QUESTIONS / 5 CARTRIDGES / FIRST-MIDDLE-LAST
# JOINT / INDEP_MASKED / BATCH_DC3 / INCR_DC3
# FROZEN QWEN · BF16 · GREEDY · NO TRAINING / RETRIEVAL / ROUTER
import urllib.request,hashlib,time,json,gc,traceback,os
BASE_FILE="566.py";BASE_BLOB="b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71"
BASE_URL="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/"+BASE_FILE
print("="*140);print("TEST573 — AKBASCORE MAM · QWEN CROSS-MODEL REPLICATION");print("="*140)
req=urllib.request.Request(BASE_URL,headers={"User-Agent":"AKBASCORE-TEST573"})
with urllib.request.urlopen(req,timeout=90) as f:raw=f.read()
blob=hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
if blob!=BASE_BLOB:raise RuntimeError("TEST566 SOURCE BLOB MISMATCH: "+blob)
src=raw.decode("utf-8")
marker="# TEST566 ADDITIONS — original TEST564 checkpoint/init/append/answer remain unchanged."
if src.count(marker)!=1:raise RuntimeError("TEST566 ENGINE BOUNDARY MISMATCH")
exec(compile(src.split(marker,1)[0],BASE_FILE,"exec"),globals())
TEST="573";CUT=3;CASES=list(range(24));SLOTS=["FIRST","MIDDLE","LAST"]
ARMS=["JOINT","INDEP","BATCH_DC3","INCR_DC3"]
MISTRAL_PANEL_SHA="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
assert PANEL_SHA==MISTRAL_PANEL_SHA and len(PANEL)==24 and len(CAR)==96
assert (NL,H,QH,KVH,HD)==(28,3584,28,4,128)
assert sentinel()==S0 and not any(p.requires_grad for p in model.parameters())
print("BASE BLOB:",blob);print("MODEL:",MODEL_ID);print("CUT:",CUT,"PREFIX:",P)
print("PANEL SHA:",PANEL_SHA,"CASES:",len(CASES),"SLOTS:",SLOTS)
print("ARMS:",ARMS,"EXPECTED ANSWERS:",len(CASES)*len(SLOTS)*len(ARMS))
def full_weight_sha():
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode());h.update(str(tuple(p.shape)).encode());h.update(str(p.dtype).encode())
        a=p.detach().contiguous().view(torch.uint8).reshape(-1)
        for part in a.split(16*1024*1024):h.update(part.cpu().numpy().tobytes())
    return h.hexdigest()
print("[1/8] Full parameter SHA-256...",flush=True)
WEIGHT_SHA0=full_weight_sha();print("WEIGHT SHA BEFORE:",WEIGHT_SHA0)
def expected_length(keys):return P+sum(len(CAR[i]["body"]) for i in keys)
def validate(mem,keys):
    n=expected_length(keys)
    if len(mem)!=NL:raise RuntimeError("LAYER COUNT MISMATCH")
    for li,(k,v) in enumerate(mem):
        shape=(1,KVH,n,HD)
        if tuple(k.shape)!=shape or tuple(v.shape)!=shape:raise RuntimeError(f"CACHE SHAPE L{li}: {k.shape}/{v.shape}")
        if not torch.isfinite(k).all() or not torch.isfinite(v).all():raise RuntimeError(f"NONFINITE CACHE L{li}")
    return n
def verify_append(old,new):
    n=old[0][0].shape[-2]
    for li,((ok,ov),(nk,nv)) in enumerate(zip(old,new)):
        if not torch.equal(ok,nk[:,:,:n,:]) or not torch.equal(ov,nv[:,:,:n,:]):raise RuntimeError(f"APPEND-ONLY BITWISE FAIL L{li}")
    return True
def make_case(keys):
    ids=list(PIDS);ranges={}
    for ci in keys:
        a=len(ids);ids.extend(CAR[ci]["body"]);ranges[ci]=(a,len(ids))
    n=len(ids);group=torch.full((n,),-1,device=DEVICE,dtype=torch.long)
    for j,ci in enumerate(keys):
        a,b=ranges[ci];group[a:b]=j
    ix=torch.arange(n,device=DEVICE)
    causal=ix[:,None]>=ix[None,:]
    isolated=causal&((group[:,None]==-1)|(group[None,:]==-1)|(group[:,None]==group[None,:]))
    jointmask=torch.zeros((1,1,n,n),device=DEVICE,dtype=DTYPE)
    indepmask=torch.zeros_like(jointmask)
    bad=torch.finfo(DTYPE).min
    jointmask.masked_fill_(~causal,bad);indepmask.masked_fill_(~isolated,bad)
    return torch.tensor([ids],device=DEVICE,dtype=torch.long),jointmask,indepmask,ranges
@torch.no_grad()
def write_reference(ids,mask):
    handles=[]
    try:
        for layer in model.model.layers:
            def pre(module,args,kwargs,mask=mask):
                kwargs["attention_mask"]=mask
                return args,kwargs
            handles.append(layer.register_forward_pre_hook(pre,with_kwargs=True))
        n=ids.shape[1];pos=torch.arange(n,device=DEVICE)
        out=model(input_ids=ids,attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        return clone(out.past_key_values)
    finally:
        for handle in handles:handle.remove()
@torch.no_grad()
def batch_dc3(keys,ranges,jointmask):
    cps={ci:checkpoint(ci,CUT) for ci in keys}
    n=expected_length(keys)
    assembled=torch.cat([cps[keys[0]]["h"][:,:P,:]]+[cps[ci]["h"][:,P:,:] for ci in keys],dim=1)
    if assembled.shape!=(1,n,H):raise RuntimeError("BATCH H ASSEMBLY FAIL")
    early=[]
    for li in range(CUT+1):
        ks=[cps[keys[0]]["kv"][li][0][:,:,:P,:]]
        vs=[cps[keys[0]]["kv"][li][1][:,:,:P,:]]
        for ci in keys:
            a,b=ranges[ci];cp=cps[ci];k,v=cp["kv"][li]
            newpos=torch.arange(a,b,device=DEVICE)
            ks.append(rephase(k[:,:,P:,:],cp["pos"][P:],newpos))
            vs.append(v[:,:,P:,:])
        early.append((torch.cat(ks,dim=-2),torch.cat(vs,dim=-2)))
    handles=[]
    try:
        def inject(module,args,out):
            x=out[0] if isinstance(out,(tuple,list)) else out
            if x.shape!=assembled.shape:raise RuntimeError("BATCH INJECTION SHAPE FAIL")
            return replace(out,assembled)
        handles.append(model.model.layers[CUT].register_forward_hook(inject))
        for li,layer in enumerate(model.model.layers):
            if li<=CUT:continue
            def pre(module,args,kwargs,mask=jointmask):
                kwargs["attention_mask"]=mask
                return args,kwargs
            handles.append(layer.register_forward_pre_hook(pre,with_kwargs=True))
        pos=torch.arange(n,device=DEVICE)
        out=model(inputs_embeds=torch.zeros((1,n,H),device=DEVICE,dtype=DTYPE),attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        upper=clone(out.past_key_values)
        return tuple(early)+upper[CUT+1:]
    finally:
        for handle in handles:handle.remove()
@torch.no_grad()
def incremental_dc3(keys):
    mem=init(keys[0],CUT)
    for ci in keys[1:]:
        grown=append(mem,ci,CUT)
        verify_append(mem,grown)
        mem=grown
    return mem
def score(rows,arm,slot=None):
    rr=[r for r in rows if r["arm"]==arm and (slot is None or r["slot"]==slot)]
    return sum(int(r["ok"]) for r in rr),len(rr)
def ans_identity(rows,a,b):
    aa={(r["case"],r["slot"]):r for r in rows if r["arm"]==a}
    bb={(r["case"],r["slot"]):r for r in rows if r["arm"]==b}
    return sum(norm(aa[k]["answer"])==norm(bb[k]["answer"]) for k in aa),len(aa)
def source_audit():
    return dict(query_template=suffix("AUDIT_QUESTION"),source_fact_in_query_template=False,query_uses= "answer(memory, question) with only question-token input and installed numerical KV",source_replay_at_ask=False)
print("[2/8] Protocol provenance...",flush=True)
PROTOCOL=dict(test=TEST,model=MODEL_ID,base_file=BASE_FILE,base_blob=blob,cut=CUT,architecture=[NL,H,QH,KVH,HD],panel_sha=PANEL_SHA,cases=CASES,slots=SLOTS,arms=ARMS,seed=SEED,panel_seed=PANEL_SEED,max_new=MAX_NEW,checkpoint="independent H3 plus L0-L3 KV",joint="joint causal text prefill",indep="masked isolated-cartridge attention prefill",batch="single upper-layer consolidation forward",incremental="append-only DC3",scoring="whole-word hit",training=False,retrieval=False,router=False)
LOCK_SHA=sha(PROTOCOL)
print("LOCK SHA:",LOCK_SHA)
print("SOURCE AUDIT:",source_audit())
print("[3/8] Preflight: four arms on first panel case...",flush=True)
z=PANEL[0];keys=ordered(z,"MIDDLE");ids,jmask,imask,ranges=make_case(keys)
preflight={}
for arm in ARMS:
    if arm=="JOINT":mem=write_reference(ids,jmask)
    elif arm=="INDEP":mem=write_reference(ids,imask)
    elif arm=="BATCH_DC3":mem=batch_dc3(keys,ranges,jmask)
    else:mem=incremental_dc3(keys)
    preflight[arm]=validate(mem,keys)
    del mem
print("PREFLIGHT CACHE LENGTHS:",preflight)
if len(set(preflight.values()))!=1:raise RuntimeError("PREFLIGHT LENGTH MISMATCH")
del ids,jmask,imask,ranges
gc.collect();torch.cuda.empty_cache()
print("[4/8] Running complete 72-case panel × 4 arms...",flush=True)
START=time.perf_counter();RAW=[];ERRORS=[]
for wi in CASES:
    z=PANEL[wi];target=z["target"];typ=CAR[target]["type"]
    q=W[wi][typ];gold=CAR[target]["gold"]
    for slot in SLOTS:
        keys=ordered(z,slot)
        ids,jmask,imask,ranges=make_case(keys)
        for arm in ARMS:
            try:
                if arm=="JOINT":mem=write_reference(ids,jmask)
                elif arm=="INDEP":mem=write_reference(ids,imask)
                elif arm=="BATCH_DC3":mem=batch_dc3(keys,ranges,jmask)
                else:mem=incremental_dc3(keys)
                n=validate(mem,keys)
                a=answer(mem,q);ok=bool(hit(a,gold))
                r=dict(case=wi,slot=slot,arm=arm,type=typ,target=target,keys=keys,cache_tokens=n,question=q,gold=gold,answer=a,ok=ok)
                RAW.append(r)
                print(f"CASE={wi:02d} {slot:6s} {arm:10s} {'PASS' if ok else 'FAIL'} | gold={gold} | answer={a!r}",flush=True)
                del mem
            except Exception as e:
                ERRORS.append(dict(case=wi,slot=slot,arm=arm,error=repr(e)))
                print("FAIL-CLOSED:",ERRORS[-1],flush=True)
                raise
        del ids,jmask,imask,ranges
    if (wi+1)%4==0:
        print(f"PROGRESS {wi+1}/24 | ELAPSED={time.perf_counter()-START:.1f}s",flush=True)
        gc.collect();torch.cuda.empty_cache()
print("[5/8] Scorecard...",flush=True)
SCORES={}
for arm in ARMS:
    SCORES[arm]={}
    for slot in SLOTS:
        n,total=score(RAW,arm,slot)
        SCORES[arm][slot]=n
        if total!=24:raise RuntimeError(f"INCOMPLETE SLOT {arm} {slot}: {total}")
    n,total=score(RAW,arm)
    SCORES[arm]["TOTAL"]=n
    if total!=72:raise RuntimeError(f"INCOMPLETE ARM {arm}: {total}")
    print(f"{arm:12s} FIRST={SCORES[arm]['FIRST']:02d}/24 MIDDLE={SCORES[arm]['MIDDLE']:02d}/24 LAST={SCORES[arm]['LAST']:02d}/24 TOTAL={n:02d}/72")
print("[6/8] Paired comparisons...",flush=True)
MAP={arm:{(r["case"],r["slot"]):r for r in RAW if r["arm"]==arm} for arm in ARMS}
DIAG={}
for arm in ["BATCH_DC3","INCR_DC3"]:
    gain=sum(not MAP["INDEP"][k]["ok"] and MAP[arm][k]["ok"] for k in MAP["INDEP"])
    loss=sum(MAP["INDEP"][k]["ok"] and not MAP[arm][k]["ok"] for k in MAP["INDEP"])
    aj=ans_identity(RAW,arm,"JOINT")
    ab=ans_identity(RAW,arm,"BATCH_DC3")
    DIAG[arm]=dict(gain=gain,loss=loss,net=gain-loss,answer_identity_joint=aj[0],answer_identity_batch=ab[0])
    print(f"{arm}: GAIN={gain} LOSS={loss} NET={gain-loss:+d} ANS-ID-JOINT={aj[0]}/72 ANS-ID-BATCH={ab[0]}/72")
print("[7/8] Integrity and full weight verification...",flush=True)
INTEGRITY=len(RAW)==288 and len(ERRORS)==0 and len({(r["case"],r["slot"],r["arm"]) for r in RAW})==288
SENTINEL_OK=sentinel()==S0
WEIGHT_SHA1=full_weight_sha()
WEIGHT_OK=WEIGHT_SHA0==WEIGHT_SHA1
TRAINABLE=sum(int(p.requires_grad) for p in model.parameters())
print("PANEL INTEGRITY:","PASS" if INTEGRITY else "FAIL")
print("WEIGHT SENTINEL:","PASS" if SENTINEL_OK else "FAIL")
print("FULL WEIGHT SHA:","PASS" if WEIGHT_OK else "FAIL")
print("TRAINABLE:",TRAINABLE)
print("WEIGHT SHA AFTER:",WEIGHT_SHA1)
if not INTEGRITY or not SENTINEL_OK or not WEIGHT_OK or TRAINABLE:raise RuntimeError("TEST573 FINAL INTEGRITY FAILURE")
print("[8/8] Saving research record...",flush=True)
RECORD=dict(protocol=PROTOCOL,lock_sha=LOCK_SHA,base_blob=blob,panel_sha=PANEL_SHA,weight_sha_before=WEIGHT_SHA0,weight_sha_after=WEIGHT_SHA1,scores=SCORES,diagnostics=DIAG,raw=RAW,errors=ERRORS,integrity=INTEGRITY,source_audit=source_audit(),gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,transformers_version=transformers.__version__,seconds=round(time.perf_counter()-START,2))
RESULT_SHA=sha(RECORD)
RECORD["result_sha"]=RESULT_SHA
OUT="/content/AKBASCORE_TEST573_QWEN_CROSS_MODEL_REPLICATION.json"
with open(OUT,"w",encoding="utf-8") as f:json.dump(RECORD,f,indent=2,ensure_ascii=False)
print("="*140)
print("TEST573 — FINAL RESEARCH RECORD")
print("MODEL:",MODEL_ID,"CUT:",CUT)
print("MISTRAL TEST560 PANEL SHA:",PANEL_SHA)
print("QWEN LOCK SHA:",LOCK_SHA)
print("QWEN RESULT SHA:",RESULT_SHA)
for arm in ARMS:print(arm,SCORES[arm])
print("INTEGRITY: PASS | FULL WEIGHT SHA: PASS | TRAINABLE: 0")
print("OUTPUT:",OUT)
print("SECONDS:",RECORD["seconds"])
print("CAUTION: Cross-model replication tests the same panel and protocol, not identical tokenizer states or identical cut layers.")
print("CAUTION: Source-free ASK is guaranteed by the answer() code path; this test does not instrument every forward call with a recorder.")
print("CAUTION: Correctness is evaluated by whole-word matching; raw generations are preserved for independent review.")
print("="*140)
