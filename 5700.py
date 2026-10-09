# TEST570 — AKBASCORE MAM · QWEN HELD-OUT PANEL VALIDATION
# VERIFIED TEST564 BACKBONE · CUT3 · WORLDS 08–23 · 16 CASES × 3 POSITIONS × 3 ARMS = 144
# DC / H_FULL / PREFIX_SWAP · FROZEN BF16 · GREEDY · APPEND-ONLY · NO SOURCE REPLAY AT ASK
import urllib.request,hashlib,time,gc,json,traceback
BASE_FILE="564_qwen_CALIBRATION_start.py";BASE_BLOB="76dcf80c891640314cbe1849ddffebfae287a23e"
BASE_URL="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/"+BASE_FILE
EXPECTED_PANEL="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
print("="*132);print("TEST570 — QWEN MAM · HELD-OUT PANEL VALIDATION");print("="*132)
req=urllib.request.Request(BASE_URL,headers={"User-Agent":"AKBASCORE-TEST570","Accept":"text/plain"})
with urllib.request.urlopen(req,timeout=90) as f:raw=f.read()
blob=hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
if blob!=BASE_BLOB:raise RuntimeError(f"BASELINE BLOB MISMATCH: {blob}")
src=raw.decode("utf-8");marker="# Fixed short calibration: 8 cases x 3 positions = 24 questions per CUT"
if src.count(marker)!=1:raise RuntimeError("BASELINE BOUNDARY MISMATCH")
base=src.split(marker,1)[0]
for name in ("checkpoint","init","append","answer","cache_build","replace","sentinel","ordered","clone"):
    if f"def {name}(" not in base:raise RuntimeError(f"BASELINE FUNCTION MISSING: {name}")
exec(compile(base,BASE_FILE,"exec"),globals())
TEST="570";CUT=3;CASES=list(range(8,24));SLOTS=["FIRST","MIDDLE","LAST"];ARMS=["DC","H_FULL","PREFIX_SWAP"]
assert PANEL_SHA==EXPECTED_PANEL and len(PANEL)==24 and len(CASES)==16
assert sentinel()==S0 and not any(p.requires_grad for p in model.parameters())
assert all(PANEL[i]["target"]==KEY[(i,["CURRENT","FORMER","NEAR","ROLE"][i%4])] for i in CASES)
print("TEST:",TEST,"CUT:",CUT,"HELD-OUT WORLDS:",CASES,"ARMS:",ARMS)
print("PANEL SHA:",PANEL_SHA,"BASELINE BLOB:",blob,"PREFIX TOKENS:",P)
print("TOTAL QUESTIONS:",len(CASES)*len(SLOTS)*len(ARMS))
print("IMPORTANT: HELD-OUT EVALUATION CASES FROM EXISTING FACT POOL; NOT NEW-FACT EXTERNAL VALIDATION.")
CP={};DONOR={}
@torch.no_grad()
def getcp(ci):
    if ci not in CP:CP[ci]=checkpoint(ci,CUT)
    return CP[ci]
@torch.no_grad()
def getdonor(ci):
    di=(ci+17)%len(CAR)
    if di==ci:di=(di+1)%len(CAR)
    if di not in DONOR:DONOR[di]=getcp(di)["h"][:,:P,:].clone()
    return di,DONOR[di]
def inject_h(out,h):
    current=out[0] if isinstance(out,(tuple,list)) else out
    if current.shape!=h.shape:raise RuntimeError(f"H SHAPE FAIL {tuple(current.shape)} vs {tuple(h.shape)}")
    return replace(out,h)
@torch.no_grad()
def h_init(ci,arm):
    cp=getcp(ci);n=cp["h"].shape[1];pos=torch.arange(n,device=DEVICE)
    target=cp["h"]
    if arm=="PREFIX_SWAP":
        di,donor=getdonor(ci)
        if donor.shape!=(1,P,H):raise RuntimeError("DONOR PREFIX SHAPE FAIL")
        target=target.clone();target[:,:P,:]=donor
    hook=model.model.layers[CUT].register_forward_hook(lambda module,args,out:inject_h(out,target))
    try:
        out=model(inputs_embeds=torch.zeros((1,n,H),device=DEVICE,dtype=DTYPE),attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        result=clone(out.past_key_values)
        if len(result)!=NL:raise RuntimeError("INIT LAYER COUNT FAIL")
        return result
    finally:hook.remove()
@torch.no_grad()
def h_append(memory,ci):
    cp=getcp(ci);oldn=memory[0][0].shape[-2];q=cp["body"];pos=torch.arange(oldn,oldn+q,device=DEVICE)
    body=cp["h"][:,P:,:]
    if body.shape!=(1,q,H):raise RuntimeError("BODY SHAPE FAIL")
    hook=model.model.layers[CUT].register_forward_hook(lambda module,args,out:inject_h(out,body))
    try:
        out=model(inputs_embeds=torch.zeros((1,q,H),device=DEVICE,dtype=DTYPE),past_key_values=cache_build(memory),attention_mask=torch.ones((1,oldn+q),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        result=clone(out.past_key_values)
        if len(result)!=NL:raise RuntimeError("APPEND LAYER COUNT FAIL")
        for li,((ok,ov),(nk,nv)) in enumerate(zip(memory,result)):
            if not torch.equal(ok,nk[:,:,:oldn,:]) or not torch.equal(ov,nv[:,:,:oldn,:]):raise RuntimeError(f"APPEND-ONLY BITWISE FAIL L{li}")
            if nk.shape[-2]!=oldn+q or nv.shape[-2]!=oldn+q:raise RuntimeError(f"CACHE LENGTH FAIL L{li}")
        return result
    finally:hook.remove()
@torch.no_grad()
def build(keys,arm):
    if arm=="DC":
        memory=init(keys[0],CUT)
        for ci in keys[1:]:memory=append(memory,ci,CUT)
    else:
        memory=h_init(keys[0],arm)
        for ci in keys[1:]:memory=h_append(memory,ci)
    n=P+sum(len(CAR[ci]["body"]) for ci in keys)
    if len(memory)!=NL:raise RuntimeError("FINAL LAYER COUNT FAIL")
    for li,(k,v) in enumerate(memory):
        if k.shape!=(1,KVH,n,HD) or v.shape!=(1,KVH,n,HD):raise RuntimeError(f"FINAL CACHE SHAPE FAIL L{li}")
        if not bool(torch.isfinite(k).all()) or not bool(torch.isfinite(v).all()):raise RuntimeError(f"NONFINITE CACHE L{li}")
    return memory
print("\nPREFIX DONOR DIAGNOSTIC")
prefix_diffs=[]
for ci in sorted({ordered(PANEL[wi],slot)[0] for wi in CASES for slot in SLOTS}):
    own=getcp(ci)["h"][:,:P,:];di,donor=getdonor(ci)
    delta=(own.float()-donor.float()).abs()
    mx=float(delta.max().item());mean=float(delta.mean().item());identical=torch.equal(own,donor)
    prefix_diffs.append(dict(ci=ci,donor=di,max_abs=mx,mean_abs=mean,identical=identical))
    print(f"CI={ci:02d} DONOR={di:02d} PREFIX_IDENTICAL={identical} MAX_ABS={mx:.9g} MEAN_ABS={mean:.9g}",flush=True)
if all(x["identical"] for x in prefix_diffs):
    print("PREFIX_SWAP WARNING: ALL PREFIXES ARE BITWISE IDENTICAL. SWAP IS A NO-OP, NOT AN INDEPENDENT CAUSAL INTERVENTION.",flush=True)
else:print("PREFIX_SWAP CHECK: NONIDENTICAL PREFIX STATES DETECTED.",flush=True)
RESULTS=[];RAW=[];ERRORS=[];START=time.perf_counter()
for arm in ARMS:
    t=time.perf_counter();rows=[];err=None
    print("\n"+"="*112);print("ARM:",arm,"CUT:",CUT,flush=True)
    try:
        for wi in CASES:
            z=PANEL[wi];typ=CAR[z["target"]]["type"];q=W[wi][typ];gold=CAR[z["target"]]["gold"]
            for slot in SLOTS:
                keys=ordered(z,slot);memory=build(keys,arm)
                a=answer(memory,q);ok=hit(a,gold)
                row=dict(arm=arm,case=wi,slot=slot,gold=gold,answer=a,ok=ok)
                rows.append(row);RAW.append(row)
                print(f"{arm:11s} CASE={wi:02d} {slot:6s} {'PASS' if ok else 'FAIL'} | gold={gold} | answer={a!r}",flush=True)
                del memory
    except Exception as e:
        err=repr(e);ERRORS.append(dict(arm=arm,error=err,traceback=traceback.format_exc()))
        print(traceback.format_exc(),flush=True)
    scores={s:sum(int(r["ok"]) for r in rows if r["slot"]==s) for s in SLOTS}
    complete=len(rows)==len(CASES)*len(SLOTS) and err is None
    result=dict(arm=arm,score=sum(scores.values()),total=len(CASES)*len(SLOTS),scores=scores,complete=complete,error=err,seconds=round(time.perf_counter()-t,2))
    RESULTS.append(result)
    print(f"RESULT ARM={arm:11s} SCORE={result['score']:02d}/48 FIRST={scores['FIRST']}/16 MIDDLE={scores['MIDDLE']}/16 LAST={scores['LAST']}/16 COMPLETE={complete} TIME={result['seconds']}s",flush=True)
    if not complete:raise RuntimeError(f"INCOMPLETE ARM={arm}: {err}")
    torch.cuda.empty_cache();gc.collect()
print("\n"+"="*132);print("TEST570 — FINAL HELD-OUT VALIDATION");print("="*132)
print("MODEL:",MODEL_ID,"GPU:",torch.cuda.get_device_name(0),"CUT:",CUT)
print("PANEL SHA:",PANEL_SHA,"BASELINE BLOB:",blob)
ref={(r["case"],r["slot"]):r for r in RAW if r["arm"]=="DC"}
if len(ref)!=48:raise RuntimeError("DC REFERENCE INCOMPLETE")
for result in RESULTS:
    arm=result["arm"];rows=[r for r in RAW if r["arm"]==arm]
    identity=sum(int(norm(ref[(r["case"],r["slot"])]["answer"])==norm(r["answer"])) for r in rows)
    lost=sum(int(ref[(r["case"],r["slot"])]["ok"] and not r["ok"]) for r in rows)
    gained=sum(int(not ref[(r["case"],r["slot"])]["ok"] and r["ok"]) for r in rows)
    print(f"{arm:11s} SCORE={result['score']:02d}/48 ANSWER_ID_VS_DC={identity:02d}/48 LOST={lost:02d} GAINED={gained:02d} FIRST={result['scores']['FIRST']}/16 MIDDLE={result['scores']['MIDDLE']}/16 LAST={result['scores']['LAST']}/16")
WEIGHT_OK=sentinel()==S0;TRAINABLE=sum(int(p.requires_grad) for p in model.parameters())
unique={(r["arm"],r["case"],r["slot"]) for r in RAW}
INTEGRITY=len(RAW)==144 and len(unique)==144 and len(RESULTS)==3 and all(r["complete"] for r in RESULTS) and not ERRORS
PREFIX_NONTRIVIAL=any(not x["identical"] for x in prefix_diffs)
LOCK=dict(test=TEST,baseline_blob=blob,model=MODEL_ID,panel_sha=PANEL_SHA,cut=CUT,cases=CASES,slots=SLOTS,arms=ARMS,seed=SEED,panel_seed=PANEL_SEED)
RECORD=dict(lock=LOCK,results=RESULTS,prefix_diffs=prefix_diffs,integrity=INTEGRITY,weight_ok=WEIGHT_OK,trainable=TRAINABLE)
print("LOCK SHA:",sha(LOCK));print("RESULT SHA:",sha(RECORD))
print("PANEL INTEGRITY:","PASS" if INTEGRITY else "FAIL")
print("WEIGHT SENTINEL:","PASS" if WEIGHT_OK else "FAIL","TRAINABLE:",TRAINABLE)
print("PREFIX_SWAP NONTRIVIAL:","YES" if PREFIX_NONTRIVIAL else "NO — IDENTICAL PREFIX STATES")
print("TOTAL SECONDS:",round(time.perf_counter()-START,2))
print("SCOPE: HELD-OUT EVALUATION WORLDS WITHIN THE ORIGINAL 96-CARTRIDGE FACT POOL.")
print("DECISION: NEXT — COMPLETELY NEW FACTS / INDEPENDENT PANEL / INCREMENTAL EQUIVALENCE.")
print("="*132)
if not INTEGRITY or not WEIGHT_OK or TRAINABLE!=0:raise RuntimeError("TEST570 INTEGRITY FAILURE")
