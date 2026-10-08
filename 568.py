# TEST568 — AKBASCORE MAM · QWEN H POSITION CAUSAL X-RAY
# VERIFIED TEST564 BACKBONE · CUT3 · 24 QUESTIONS × 8 ARMS
# FROZEN BF16 · H-ONLY PATH · TOKEN-POSITION ABLATION
# NO TRAINING · NO RETRIEVAL · NO SOURCE REPLAY AT ASK
import urllib.request,urllib.error,hashlib,time,gc,json,traceback
REPO="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2"
BASE_FILE="564_qwen_CALIBRATION_start.py"
BASE_BLOB="76dcf80c891640314cbe1849ddffebfae287a23e"
BASE_URL=f"{REPO}/main/{BASE_FILE}"
EXPECTED_PANEL="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
print("="*132)
print("TEST568 — QWEN MAM · H POSITION CAUSAL X-RAY")
print("BASELINE:",BASE_BLOB)
print("="*132)
try:
    req=urllib.request.Request(BASE_URL,headers={"User-Agent":"AKBASCORE-TEST568","Accept":"text/plain"})
    with urllib.request.urlopen(req,timeout=90) as f:raw=f.read()
except Exception as e:raise RuntimeError(f"BASELINE DOWNLOAD FAILED: {e}") from e
blob=hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
if blob!=BASE_BLOB:raise RuntimeError(f"BASELINE SHA MISMATCH: {blob}")
src=raw.decode("utf-8")
marker="# Fixed short calibration: 8 cases x 3 positions = 24 questions per CUT"
if src.count(marker)!=1:raise RuntimeError("BASELINE BOUNDARY MISMATCH")
base=src.split(marker,1)[0]
for name in ("checkpoint","init","append","answer","cache_build","replace","sentinel","ordered"):
    if f"def {name}(" not in base:raise RuntimeError(f"MISSING BACKBONE FUNCTION: {name}")
exec(compile(base,BASE_FILE,"exec"),globals())
TEST="568";CUT=3;CASES=list(range(8));SLOTS=["FIRST","MIDDLE","LAST"]
ARMS=["DC","H_FULL","H_BODY","H_PREFIX","H_FIRST","H_LAST","H_HEAD","H_TAIL","H_NONE"]
if PANEL_SHA!=EXPECTED_PANEL:raise RuntimeError("PANEL SHA FAIL")
if sentinel()!=S0:raise RuntimeError("WEIGHT SENTINEL FAIL")
if any(p.requires_grad for p in model.parameters()):raise RuntimeError("TRAINABLE WEIGHTS DETECTED")
print("\nTEST568 CONFIG | CUT:",CUT,"ARMS:",ARMS)
print("PANEL:",PANEL_SHA,"| BACKBONE VERIFIED:",blob)
print("H_FULL=all available H | H_BODY=body H | H_PREFIX=prefix H")
print("H_FIRST/H_LAST=single body position | H_HEAD/H_TAIL=half body | H_NONE=no injection")
print("NOTE: append has no prefix tokens; H_PREFIX therefore injects only during first-cartridge init.")
def position_mask(mode,n,prefix):
    m=torch.zeros((1,n,1),device=DEVICE,dtype=torch.bool)
    b=n-prefix
    if b<1:raise RuntimeError("EMPTY BODY")
    if mode=="H_FULL":m[:]=True
    elif mode=="H_BODY":m[:,prefix:,:]=True
    elif mode=="H_PREFIX":m[:,:prefix,:]=True
    elif mode=="H_FIRST":m[:,prefix:prefix+1,:]=True
    elif mode=="H_LAST":m[:,n-1:n,:]=True
    elif mode=="H_HEAD":m[:,prefix:prefix+(b+1)//2,:]=True
    elif mode=="H_TAIL":m[:,prefix+b//2:,:]=True
    elif mode=="H_NONE":pass
    else:raise ValueError(mode)
    return m
def masked_replace(out,h,mask):
    current=out[0] if isinstance(out,(tuple,list)) else out
    if current.shape!=h.shape:raise RuntimeError(f"H SHAPE MISMATCH: {tuple(current.shape)} != {tuple(h.shape)}")
    if mask.shape!=(1,h.shape[1],1):raise RuntimeError("MASK SHAPE MISMATCH")
    return replace(out,torch.where(mask,h,current))
@torch.no_grad()
def h_init(ci,mode):
    if mode=="DC":return init(ci,CUT)
    cp=checkpoint(ci,CUT);n=cp["h"].shape[1];pos=torch.arange(n,device=DEVICE)
    mask=position_mask(mode,n,P);hook=None
    if bool(mask.any()):
        def inject(module,args,out):return masked_replace(out,cp["h"],mask)
        hook=model.model.layers[CUT].register_forward_hook(inject)
    try:
        out=model(inputs_embeds=torch.zeros((1,n,H),device=DEVICE,dtype=DTYPE),attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        result=clone(out.past_key_values)
        if len(result)!=NL:raise RuntimeError("INIT LAYER COUNT FAIL")
        return result
    finally:
        if hook is not None:hook.remove()
@torch.no_grad()
def h_append(memory,ci,mode):
    if mode=="DC":return append(memory,ci,CUT)
    cp=checkpoint(ci,CUT);oldn=memory[0][0].shape[-2];q=cp["body"]
    pos=torch.arange(oldn,oldn+q,device=DEVICE);body=cp["h"][:,P:,:]
    if body.shape!=(1,q,H):raise RuntimeError("BODY SHAPE FAIL")
    mask=position_mask(mode,q,0) if mode!="H_PREFIX" else torch.zeros((1,q,1),device=DEVICE,dtype=torch.bool)
    hook=None
    if bool(mask.any()):
        def inject(module,args,out):return masked_replace(out,body,mask)
        hook=model.model.layers[CUT].register_forward_hook(inject)
    try:
        out=model(inputs_embeds=torch.zeros((1,q,H),device=DEVICE,dtype=DTYPE),past_key_values=cache_build(memory),attention_mask=torch.ones((1,oldn+q),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        result=clone(out.past_key_values)
        if len(result)!=NL:raise RuntimeError("APPEND LAYER COUNT FAIL")
        for li,((ok,ov),(nk,nv)) in enumerate(zip(memory,result)):
            if not torch.equal(ok,nk[:,:,:oldn,:]) or not torch.equal(ov,nv[:,:,:oldn,:]):raise RuntimeError(f"APPEND BITWISE FAIL L{li}")
            if nk.shape[-2]!=oldn+q or nv.shape[-2]!=oldn+q:raise RuntimeError(f"APPEND LENGTH FAIL L{li}")
        return result
    finally:
        if hook is not None:hook.remove()
@torch.no_grad()
def build_memory(keys,mode):
    mem=h_init(keys[0],mode)
    for ci in keys[1:]:mem=h_append(mem,ci,mode)
    n=P+sum(len(CAR[ci]["body"]) for ci in keys)
    if len(mem)!=NL:raise RuntimeError("FINAL LAYER COUNT FAIL")
    for li,(k,v) in enumerate(mem):
        if k.shape!=(1,KVH,n,HD) or v.shape!=(1,KVH,n,HD):raise RuntimeError(f"FINAL SHAPE FAIL L{li}")
        if not bool(torch.isfinite(k).all()) or not bool(torch.isfinite(v).all()):raise RuntimeError(f"NONFINITE CACHE L{li}")
    return mem
RESULTS=[];RAW=[];ERRORS=[];START=time.perf_counter()
for arm in ARMS:
    t=time.perf_counter();rows=[];err=None
    print("\n"+"="*112)
    print("ARM:",arm,"CUT:",CUT,flush=True)
    try:
        for wi in CASES:
            z=PANEL[wi];typ=CAR[z["target"]]["type"];q=W[wi][typ];gold=CAR[z["target"]]["gold"]
            for slot in SLOTS:
                keys=ordered(z,slot);memory=build_memory(keys,arm)
                a=answer(memory,q);ok=hit(a,gold)
                row=dict(arm=arm,case=wi,slot=slot,gold=gold,answer=a,ok=ok)
                rows.append(row);RAW.append(row)
                print(f"{arm:8s} CASE={wi:02d} {slot:6s} {'PASS' if ok else 'FAIL'} | gold={gold} | answer={a!r}",flush=True)
                del memory
    except Exception as e:
        err=repr(e);ERRORS.append(dict(arm=arm,error=err,traceback=traceback.format_exc()))
        print(traceback.format_exc(),flush=True)
    scores={s:sum(int(r["ok"]) for r in rows if r["slot"]==s) for s in SLOTS}
    complete=len(rows)==24 and err is None
    result=dict(arm=arm,score=sum(scores.values()),scores=scores,complete=complete,error=err,seconds=round(time.perf_counter()-t,2))
    RESULTS.append(result)
    print(f"RESULT ARM={arm:8s} SCORE={result['score']:02d}/24 FIRST={scores['FIRST']}/8 MIDDLE={scores['MIDDLE']}/8 LAST={scores['LAST']}/8 COMPLETE={complete} TIME={result['seconds']}s",flush=True)
    if not complete:raise RuntimeError(f"INCOMPLETE ARM={arm}: {err}")
    torch.cuda.empty_cache();gc.collect()
print("\n"+"="*132)
print("TEST568 — FINAL H POSITION CAUSAL X-RAY")
print("="*132)
print("MODEL:",MODEL_ID,"GPU:",torch.cuda.get_device_name(0))
print("CUT:",CUT,"PANEL SHA:",PANEL_SHA,"BASELINE BLOB:",blob)
reference={(r["case"],r["slot"]):r for r in RAW if r["arm"]=="DC"}
full={(r["case"],r["slot"]):r for r in RAW if r["arm"]=="H_FULL"}
for result in RESULTS:
    arm=result["arm"];rows=[r for r in RAW if r["arm"]==arm]
    lost=sum(int(reference[(r["case"],r["slot"])]["ok"] and not r["ok"]) for r in rows)
    identity=sum(int(norm(reference[(r["case"],r["slot"])]["answer"])==norm(r["answer"])) for r in rows)
    print(f"{arm:8s} SCORE={result['score']:02d}/24 LOST_VS_DC={lost:02d} ANSWER_ID_VS_DC={identity:02d}/24 FIRST={result['scores']['FIRST']}/8 MIDDLE={result['scores']['MIDDLE']}/8 LAST={result['scores']['LAST']}/8")
if len(full)!=24:raise RuntimeError("H_FULL REFERENCE INCOMPLETE")
WEIGHT_OK=sentinel()==S0;TRAINABLE=sum(int(p.requires_grad) for p in model.parameters())
unique={(r["arm"],r["case"],r["slot"]) for r in RAW}
INTEGRITY=len(RAW)==len(ARMS)*24 and len(unique)==len(RAW) and len(RESULTS)==len(ARMS) and not ERRORS and all(r["complete"] for r in RESULTS)
LOCK=dict(test=TEST,baseline_blob=blob,model=MODEL_ID,panel_sha=PANEL_SHA,cut=CUT,arms=ARMS,seed=SEED,panel_seed=PANEL_SEED)
RECORD=dict(lock=LOCK,results=RESULTS,integrity=INTEGRITY,weight_ok=WEIGHT_OK,trainable=TRAINABLE)
print("LOCK SHA:",sha(LOCK))
print("RESULT SHA:",sha(RECORD))
print("PANEL INTEGRITY:","PASS" if INTEGRITY else "FAIL")
print("WEIGHT SENTINEL:","PASS" if WEIGHT_OK else "FAIL","TRAINABLE:",TRAINABLE)
print("TOTAL SECONDS:",round(time.perf_counter()-START,2))
print("CAUTION: H_PREFIX only affects initial cartridge prefill; later cartridges contain body tokens only.")
print("CAUTION: Position masks retain native zero-embedding states outside the injected positions; this is not an information-theoretic isolation proof.")
print("DECISION: IDENTIFY H-SENSITIVE TOKEN REGIONS BEFORE TEST569")
print("="*132)
if not INTEGRITY or not WEIGHT_OK or TRAINABLE!=0:raise RuntimeError("TEST568 INTEGRITY FAILURE")
