# TEST569 — AKBASCORE MAM · QWEN FIRST-CARTRIDGE PREFIX H DIAGNOSIS
# VERIFIED TEST564 BACKBONE · CUT3 · 24 QUESTIONS × 7 ARMS
# FROZEN BF16 · NO TRAINING · NO RETRIEVAL · NO SOURCE REPLAY AT ASK
# H_FULL / H_BODY / PREFIX_KEEP / PREFIX_SWAP / PREFIX_ZERO / PREFIX_SCALE / PREFIX_MEAN
import urllib.request,hashlib,time,gc,json,traceback
REPO="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2"
BASE_FILE="564_qwen_CALIBRATION_start.py"
BASE_BLOB="76dcf80c891640314cbe1849ddffebfae287a23e"
BASE_URL=f"{REPO}/main/{BASE_FILE}"
EXPECTED_PANEL="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
print("="*132)
print("TEST569 — QWEN FIRST-CARTRIDGE PREFIX H DIAGNOSIS")
print("BASELINE:",BASE_BLOB)
print("="*132)
req=urllib.request.Request(BASE_URL,headers={"User-Agent":"AKBASCORE-TEST569","Accept":"text/plain"})
with urllib.request.urlopen(req,timeout=90) as f:raw=f.read()
blob=hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
if blob!=BASE_BLOB:raise RuntimeError(f"BASELINE SHA MISMATCH: {blob}")
src=raw.decode("utf-8")
marker="# Fixed short calibration: 8 cases x 3 positions = 24 questions per CUT"
if src.count(marker)!=1:raise RuntimeError("BASELINE BOUNDARY MISMATCH")
base=src.split(marker,1)[0]
for name in ("checkpoint","init","append","answer","cache_build","replace","sentinel","ordered"):
    if f"def {name}(" not in base:raise RuntimeError(f"MISSING BACKBONE FUNCTION: {name}")
exec(compile(base,BASE_FILE,"exec"),globals())
TEST="569";CUT=3;CASES=list(range(8));SLOTS=["FIRST","MIDDLE","LAST"]
ARMS=["H_FULL","H_BODY","PREFIX_KEEP","PREFIX_SWAP","PREFIX_ZERO","PREFIX_SCALE","PREFIX_MEAN"]
if PANEL_SHA!=EXPECTED_PANEL:raise RuntimeError("PANEL SHA FAIL")
if sentinel()!=S0:raise RuntimeError("WEIGHT SENTINEL FAIL")
if any(p.requires_grad for p in model.parameters()):raise RuntimeError("TRAINABLE WEIGHTS DETECTED")
print("\nTEST569 CONFIG | CUT:",CUT,"ARMS:",ARMS)
print("PANEL:",PANEL_SHA,"| VERIFIED BACKBONE:",blob)
print("PREFIX_KEEP=first cartridge full H, later cartridges body H")
print("PREFIX_SWAP=first cartridge prefix H from independent donor; body H original")
print("PREFIX_ZERO=first cartridge prefix H zeroed; body H original")
print("PREFIX_SCALE=first cartridge prefix H multiplied by 0.5; body H original")
print("PREFIX_MEAN=first cartridge prefix H replaced by mean across donor prefixes")
print("NOTE: H_FULL and H_BODY reproduce TEST568 controls.")
@torch.no_grad()
def prefix_donor(ci):
    donor=(ci+17)%len(CAR)
    if donor==ci:donor=(donor+1)%len(CAR)
    return checkpoint(donor,CUT)["h"][:,:P,:]
@torch.no_grad()
def prefix_mean(ci):
    donors=[(ci+j)%len(CAR) for j in (7,17,29,43)]
    donors=[d if d!=ci else (d+1)%len(CAR) for d in donors]
    return torch.stack([checkpoint(d,CUT)["h"][:,:P,:].float() for d in donors],0).mean(0).to(DTYPE)
def replace_h(out,h):
    current=out[0] if isinstance(out,(tuple,list)) else out
    if current.shape!=h.shape:raise RuntimeError(f"H SHAPE MISMATCH {tuple(current.shape)} != {tuple(h.shape)}")
    return replace(out,h)
@torch.no_grad()
def first_init(ci,arm):
    cp=checkpoint(ci,CUT);n=cp["h"].shape[1];pos=torch.arange(n,device=DEVICE)
    own=cp["h"];hook=None
    if arm in ("H_FULL","PREFIX_KEEP"):target=own
    elif arm=="H_BODY":target=None
    elif arm=="PREFIX_SWAP":
        target=own.clone();target[:,:P,:]=prefix_donor(ci)
    elif arm=="PREFIX_ZERO":
        target=own.clone();target[:,:P,:]=0
    elif arm=="PREFIX_SCALE":
        target=own.clone();target[:,:P,:]*=0.5
    elif arm=="PREFIX_MEAN":
        target=own.clone();target[:,:P,:]=prefix_mean(ci)
    else:raise ValueError(arm)
    if arm=="H_BODY":
        def inject(module,args,out):
            current=out[0] if isinstance(out,(tuple,list)) else out
            target=torch.cat((current[:,:P,:],own[:,P:,:]),dim=1)
            return replace_h(out,target)
    else:
        def inject(module,args,out):return replace_h(out,target)
    hook=model.model.layers[CUT].register_forward_hook(inject)
    try:
        out=model(inputs_embeds=torch.zeros((1,n,H),device=DEVICE,dtype=DTYPE),attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        result=clone(out.past_key_values)
        if len(result)!=NL:raise RuntimeError("INIT LAYER COUNT FAIL")
        return result
    finally:
        if hook is not None:hook.remove()
@torch.no_grad()
def body_append(memory,ci):
    cp=checkpoint(ci,CUT);oldn=memory[0][0].shape[-2];q=cp["body"];pos=torch.arange(oldn,oldn+q,device=DEVICE);body=cp["h"][:,P:,:]
    if body.shape!=(1,q,H):raise RuntimeError("BODY SHAPE FAIL")
    def inject(module,args,out):return replace_h(out,body)
    hook=model.model.layers[CUT].register_forward_hook(inject)
    try:
        out=model(inputs_embeds=torch.zeros((1,q,H),device=DEVICE,dtype=DTYPE),past_key_values=cache_build(memory),attention_mask=torch.ones((1,oldn+q),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        result=clone(out.past_key_values)
        if len(result)!=NL:raise RuntimeError("APPEND LAYER COUNT FAIL")
        for li,((ok,ov),(nk,nv)) in enumerate(zip(memory,result)):
            if not torch.equal(ok,nk[:,:,:oldn,:]) or not torch.equal(ov,nv[:,:,:oldn,:]):raise RuntimeError(f"APPEND-ONLY FAIL L{li}")
            if nk.shape[-2]!=oldn+q or nv.shape[-2]!=oldn+q:raise RuntimeError(f"CACHE LENGTH FAIL L{li}")
        return result
    finally:hook.remove()
@torch.no_grad()
def build_memory(keys,arm):
    if not keys:raise RuntimeError("EMPTY CARTRIDGE ORDER")
    mem=first_init(keys[0],arm)
    for ci in keys[1:]:mem=body_append(mem,ci)
    n=P+sum(len(CAR[ci]["body"]) for ci in keys)
    if len(mem)!=NL:raise RuntimeError("FINAL LAYER COUNT FAIL")
    for li,(k,v) in enumerate(mem):
        if k.shape!=(1,KVH,n,HD) or v.shape!=(1,KVH,n,HD):raise RuntimeError(f"FINAL SHAPE FAIL L{li}")
        if not bool(torch.isfinite(k).all()) or not bool(torch.isfinite(v).all()):raise RuntimeError(f"NONFINITE CACHE L{li}")
    return mem
RESULTS=[];RAW=[];ERRORS=[];START=time.perf_counter()
for arm in ARMS:
    t=time.perf_counter();rows=[];err=None
    print("\n"+"="*112);print("ARM:",arm,"CUT:",CUT,flush=True)
    try:
        for wi in CASES:
            z=PANEL[wi];typ=CAR[z["target"]]["type"];q=W[wi][typ];gold=CAR[z["target"]]["gold"]
            for slot in SLOTS:
                keys=ordered(z,slot);mem=build_memory(keys,arm)
                a=answer(mem,q);ok=hit(a,gold)
                row=dict(arm=arm,case=wi,slot=slot,gold=gold,answer=a,ok=ok)
                rows.append(row);RAW.append(row)
                print(f"{arm:12s} CASE={wi:02d} {slot:6s} {'PASS' if ok else 'FAIL'} | gold={gold} | answer={a!r}",flush=True)
                del mem
    except Exception as e:
        err=repr(e);ERRORS.append(dict(arm=arm,error=err,traceback=traceback.format_exc()))
        print(traceback.format_exc(),flush=True)
    scores={s:sum(int(r["ok"]) for r in rows if r["slot"]==s) for s in SLOTS}
    complete=len(rows)==24 and err is None
    result=dict(arm=arm,score=sum(scores.values()),scores=scores,complete=complete,error=err,seconds=round(time.perf_counter()-t,2))
    RESULTS.append(result)
    print(f"RESULT ARM={arm:12s} SCORE={result['score']:02d}/24 FIRST={scores['FIRST']}/8 MIDDLE={scores['MIDDLE']}/8 LAST={scores['LAST']}/8 COMPLETE={complete} TIME={result['seconds']}s",flush=True)
    if not complete:raise RuntimeError(f"INCOMPLETE ARM={arm}: {err}")
    torch.cuda.empty_cache();gc.collect()
print("\n"+"="*132);print("TEST569 — FINAL PREFIX H DIAGNOSIS");print("="*132)
print("MODEL:",MODEL_ID,"GPU:",torch.cuda.get_device_name(0))
print("CUT:",CUT,"PANEL SHA:",PANEL_SHA,"BASELINE BLOB:",blob)
reference={(r["case"],r["slot"]):r for r in RAW if r["arm"]=="H_FULL"}
if len(reference)!=24:raise RuntimeError("REFERENCE INCOMPLETE")
for result in RESULTS:
    arm=result["arm"];rows=[r for r in RAW if r["arm"]==arm]
    lost=sum(int(reference[(r["case"],r["slot"])]["ok"] and not r["ok"]) for r in rows)
    gained=sum(int(not reference[(r["case"],r["slot"])]["ok"] and r["ok"]) for r in rows)
    identity=sum(int(norm(reference[(r["case"],r["slot"])]["answer"])==norm(r["answer"])) for r in rows)
    print(f"{arm:12s} SCORE={result['score']:02d}/24 LOST={lost:02d} GAINED={gained:02d} ANSWER_ID={identity:02d}/24 FIRST={result['scores']['FIRST']}/8 MIDDLE={result['scores']['MIDDLE']}/8 LAST={result['scores']['LAST']}/8")
WEIGHT_OK=sentinel()==S0;TRAINABLE=sum(int(p.requires_grad) for p in model.parameters())
unique={(r["arm"],r["case"],r["slot"]) for r in RAW}
INTEGRITY=len(RAW)==len(ARMS)*24 and len(unique)==len(RAW) and len(RESULTS)==len(ARMS) and not ERRORS and all(r["complete"] for r in RESULTS)
LOCK=dict(test=TEST,baseline_blob=blob,model=MODEL_ID,panel_sha=PANEL_SHA,cut=CUT,arms=ARMS,seed=SEED,panel_seed=PANEL_SEED)
RECORD=dict(lock=LOCK,results=RESULTS,integrity=INTEGRITY,weight_ok=WEIGHT_OK,trainable=TRAINABLE)
print("LOCK SHA:",sha(LOCK));print("RESULT SHA:",sha(RECORD))
print("PANEL INTEGRITY:","PASS" if INTEGRITY else "FAIL")
print("WEIGHT SENTINEL:","PASS" if WEIGHT_OK else "FAIL","TRAINABLE:",TRAINABLE)
print("TOTAL SECONDS:",round(time.perf_counter()-START,2))
print("INTERPRETATION: PREFIX_SWAP/PREFIX_MEAN test donor sensitivity, not a complete information-localization proof.")
print("INTERPRETATION: H_BODY vs PREFIX_KEEP isolates first-cartridge prefix restoration within the H-only reconstruction.")
print("DECISION: IF PREFIX EFFECT IS RESOLVED, PROCEED TO INDEPENDENT VALIDATION.")
print("="*132)
if not INTEGRITY or not WEIGHT_OK or TRAINABLE!=0:raise RuntimeError("TEST569 INTEGRITY FAILURE")
