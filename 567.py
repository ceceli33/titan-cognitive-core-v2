# TEST567 — AKBASCORE MAM · QWEN H/KV CAUSAL ABLATION · FIXED
# VERIFIED TEST564 BACKBONE · DC / H_ONLY / KV_ONLY / ZERO
# CUTS 1,3,6 · SAME PANEL SHA · 24 QUESTIONS PER ARM
# FROZEN BF16 · NO TRAINING · NO RETRIEVAL · NO SOURCE REPLAY AT ASK
import urllib.request,urllib.error,hashlib,time,gc,json,traceback
REPO="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2"
BASE_BLOB_SHA="76dcf80c891640314cbe1849ddffebfae287a23e"
BASE_FILE="564_qwen_CALIBRATION_start.py"
BASE_URL=f"{REPO}/main/{BASE_FILE}"
EXPECTED_PANEL_SHA="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
print("="*130)
print("TEST567 — QWEN H/KV CAUSAL ABLATION · FIXED")
print("BASELINE GIT BLOB SHA:",BASE_BLOB_SHA)
print("SOURCE:",BASE_URL)
print("="*130)
try:
    req=urllib.request.Request(BASE_URL,headers={"User-Agent":"AKBASCORE-TEST567/1.1","Accept":"text/plain"})
    with urllib.request.urlopen(req,timeout=90) as response:raw=response.read()
except urllib.error.HTTPError as e:
    raise RuntimeError(f"BASELINE DOWNLOAD FAILED: HTTP {e.code} | {BASE_URL}") from e
except Exception as e:
    raise RuntimeError(f"BASELINE DOWNLOAD FAILED: {repr(e)}") from e
blob=hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
if blob!=BASE_BLOB_SHA:raise RuntimeError(f"BASELINE SHA MISMATCH: expected={BASE_BLOB_SHA} actual={blob}. Repository file changed; refusing to run.")
src=raw.decode("utf-8")
marker="# Fixed short calibration: 8 cases x 3 positions = 24 questions per CUT"
if src.count(marker)!=1:raise RuntimeError("TEST564 calibration boundary not found uniquely")
base=src.split(marker,1)[0]
for name in ("checkpoint","init","append","answer","cache_build","rephase","replace","sentinel","ordered"):
    if f"def {name}(" not in base:raise RuntimeError(f"TEST564 backbone incomplete: {name}")
exec(compile(base,BASE_FILE,"exec"),globals())
TEST="567";CUTS=[1,3,6];ARMS=["DC","H_ONLY","KV_ONLY","ZERO"];CASES=list(range(8));SLOTS=["FIRST","MIDDLE","LAST"]
if PANEL_SHA!=EXPECTED_PANEL_SHA:raise RuntimeError(f"PANEL SHA MISMATCH: {PANEL_SHA}")
if sentinel()!=S0:raise RuntimeError("BASELINE WEIGHT SENTINEL FAIL")
if any(p.requires_grad for p in model.parameters()):raise RuntimeError("TRAINABLE WEIGHTS DETECTED")
print("\nTEST567 START | CUTS:",CUTS,"ARMS:",ARMS,"PANEL:",PANEL_SHA)
print("BACKBONE VERIFIED:",blob,"| ORIGINAL checkpoint/init/append/answer UNMODIFIED")
print("DC=H+KV | H_ONLY=H without independent early KV | KV_ONLY=independent early KV without H | ZERO=neither")
@torch.no_grad()
def ablation_init(ci,cut,arm):
    if arm=="DC":return init(ci,cut)
    if arm not in ("H_ONLY","KV_ONLY","ZERO"):raise ValueError(f"Unknown arm: {arm}")
    cp=checkpoint(ci,cut);n=cp["h"].shape[1];pos=torch.arange(n,device=DEVICE)
    use_h=arm=="H_ONLY";use_kv=arm=="KV_ONLY";hook=None
    if use_h:
        def inject(module,args,out):return replace(out,cp["h"])
        hook=model.model.layers[cut].register_forward_hook(inject)
    try:
        out=model(inputs_embeds=torch.zeros((1,n,H),device=DEVICE,dtype=DTYPE),attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        native=clone(out.past_key_values)
        result=cp["kv"]+native[cut+1:] if use_kv else native
        if len(result)!=NL:raise RuntimeError("INIT LAYER COUNT MISMATCH")
        return result
    finally:
        if hook is not None:hook.remove()
@torch.no_grad()
def ablation_append(memory,ci,cut,arm):
    if arm=="DC":return append(memory,ci,cut)
    if arm not in ("H_ONLY","KV_ONLY","ZERO"):raise ValueError(f"Unknown arm: {arm}")
    cp=checkpoint(ci,cut);oldn=memory[0][0].shape[-2];q=cp["body"];pos=torch.arange(oldn,oldn+q,device=DEVICE);body=cp["h"][:,P:,:]
    if body.shape!=(1,q,H):raise RuntimeError("BODY SHAPE MISMATCH")
    use_h=arm=="H_ONLY";use_kv=arm=="KV_ONLY";hook=None
    if use_h:
        def inject(module,args,out):
            current=out[0] if isinstance(out,(tuple,list)) else out
            if current.shape!=body.shape:raise RuntimeError("INJECTION SHAPE MISMATCH")
            return replace(out,body)
        hook=model.model.layers[cut].register_forward_hook(inject)
    try:
        out=model(inputs_embeds=torch.zeros((1,q,H),device=DEVICE,dtype=DTYPE),past_key_values=cache_build(memory),attention_mask=torch.ones((1,oldn+q),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        grown=list(clone(out.past_key_values))
        if use_kv:
            for li in range(cut+1):
                ok,ov=memory[li];k,v=cp["kv"][li]
                k=rephase(k[:,:,P:,:],cp["pos"][P:],pos);v=v[:,:,P:,:]
                grown[li]=(torch.cat((ok,k),-2),torch.cat((ov,v),-2))
        result=tuple(grown)
        if len(result)!=NL:raise RuntimeError("APPEND LAYER COUNT MISMATCH")
        for li,((ok,ov),(nk,nv)) in enumerate(zip(memory,result)):
            if not torch.equal(ok,nk[:,:,:oldn,:]) or not torch.equal(ov,nv[:,:,:oldn,:]):raise RuntimeError(f"APPEND-ONLY BITWISE FAIL L{li}")
        if any(k.shape[-2]!=oldn+q or v.shape[-2]!=oldn+q for k,v in result):raise RuntimeError("CACHE LENGTH MISMATCH")
        return result
    finally:
        if hook is not None:hook.remove()
@torch.no_grad()
def make_memory(keys,cut,arm):
    if not keys:raise ValueError("EMPTY CARTRIDGE ORDER")
    mem=ablation_init(keys[0],cut,arm)
    for ci in keys[1:]:mem=ablation_append(mem,ci,cut,arm)
    n=P+sum(len(CAR[ci]["body"]) for ci in keys)
    if len(mem)!=NL:raise RuntimeError("FINAL LAYER COUNT MISMATCH")
    for li,(k,v) in enumerate(mem):
        if k.shape!=(1,KVH,n,HD) or v.shape!=(1,KVH,n,HD):raise RuntimeError(f"SHAPE FAIL L{li}: K={tuple(k.shape)} V={tuple(v.shape)}")
        if not bool(torch.isfinite(k).all()) or not bool(torch.isfinite(v).all()):raise RuntimeError(f"NONFINITE CACHE L{li}")
    return mem
RESULTS=[];RAW=[];ERRORS=[];START=time.perf_counter()
for cut in CUTS:
    print("\n"+"="*110);print("CUT",cut,flush=True)
    for arm in ARMS:
        rows=[];err=None;t=time.perf_counter()
        print("\nARM:",arm,"CUT:",cut,flush=True)
        try:
            for wi in CASES:
                z=PANEL[wi];typ=CAR[z["target"]]["type"];q=W[wi][typ];gold=CAR[z["target"]]["gold"]
                for slot in SLOTS:
                    keys=ordered(z,slot);mem=make_memory(keys,cut,arm)
                    a=answer(mem,q);ok=hit(a,gold)
                    row=dict(cut=cut,arm=arm,case=wi,slot=slot,gold=gold,answer=a,ok=ok)
                    rows.append(row);RAW.append(row)
                    print(f"CUT={cut:02d} {arm:7s} CASE={wi:02d} {slot:6s} {'PASS' if ok else 'FAIL'} | gold={gold} | answer={a!r}",flush=True)
                    del mem
        except Exception as e:
            err=repr(e);ERRORS.append(dict(cut=cut,arm=arm,error=err,traceback=traceback.format_exc()))
            print("ERROR:",err,flush=True)
            print(traceback.format_exc(),flush=True)
        scores={s:sum(int(r["ok"]) for r in rows if r["slot"]==s) for s in SLOTS}
        complete=len(rows)==24 and err is None
        result=dict(cut=cut,arm=arm,score=sum(scores.values()),scores=scores,complete=complete,error=err,seconds=round(time.perf_counter()-t,2))
        RESULTS.append(result)
        print(f"RESULT CUT={cut:02d} ARM={arm:7s} SCORE={result['score']:02d}/24 FIRST={scores['FIRST']}/8 MIDDLE={scores['MIDDLE']}/8 LAST={scores['LAST']}/8 COMPLETE={complete} TIME={result['seconds']}s",flush=True)
        if not complete:raise RuntimeError(f"INCOMPLETE CUT={cut} ARM={arm}: {err}")
    torch.cuda.empty_cache();gc.collect()
print("\n"+"="*130);print("TEST567 — FINAL CAUSAL ABLATION MATRIX");print("="*130)
print("MODEL:",MODEL_ID,"GPU:",torch.cuda.get_device_name(0))
print("BASELINE GIT BLOB SHA:",blob)
print("PANEL SHA:",PANEL_SHA)
print("ARM DEFINITIONS: DC=H+KV | H_ONLY=H+native early KV | KV_ONLY=no H+independent early KV | ZERO=no H+native early KV")
for cut in CUTS:
    d={r["arm"]:r for r in RESULTS if r["cut"]==cut}
    print(f"CUT={cut:02d} DC={d['DC']['score']:02d}/24 H_ONLY={d['H_ONLY']['score']:02d}/24 KV_ONLY={d['KV_ONLY']['score']:02d}/24 ZERO={d['ZERO']['score']:02d}/24")
    ref={(r["case"],r["slot"]):r for r in RAW if r["cut"]==cut and r["arm"]=="DC"}
    for arm in ARMS[1:]:
        rows=[r for r in RAW if r["cut"]==cut and r["arm"]==arm]
        lost=sum(int(ref[(r["case"],r["slot"])]["ok"] and not r["ok"]) for r in rows)
        gained=sum(int(not ref[(r["case"],r["slot"])]["ok"] and r["ok"]) for r in rows)
        identity=sum(int(norm(ref[(r["case"],r["slot"])]["answer"])==norm(r["answer"])) for r in rows)
        print(f"  {arm:7s} VS DC | LOST={lost:02d} GAINED={gained:02d} ANSWER_IDENTITY={identity:02d}/24")
WEIGHT_OK=sentinel()==S0;TRAINABLE=sum(int(p.requires_grad) for p in model.parameters())
unique={(r["cut"],r["arm"],r["case"],r["slot"]) for r in RAW}
INTEGRITY=len(RAW)==len(CUTS)*len(ARMS)*24 and len(unique)==len(RAW) and len(RESULTS)==len(CUTS)*len(ARMS) and not ERRORS and all(r["complete"] for r in RESULTS)
LOCK=dict(test=TEST,baseline_blob=blob,model=MODEL_ID,panel_sha=PANEL_SHA,cuts=CUTS,arms=ARMS,seed=SEED,panel_seed=PANEL_SEED)
RECORD=dict(lock=LOCK,results=RESULTS,integrity=INTEGRITY,weight_ok=WEIGHT_OK,trainable=TRAINABLE)
print("LOCK SHA:",sha(LOCK));print("RESULT SHA:",sha(RECORD))
print("PANEL INTEGRITY:","PASS" if INTEGRITY else "FAIL")
print("WEIGHT SENTINEL:","PASS" if WEIGHT_OK else "FAIL","TRAINABLE:",TRAINABLE)
print("TOTAL SECONDS:",round(time.perf_counter()-START,2))
print("INTERPRETATION: H_ONLY and KV_ONLY are pathway ablations, not clean independent causal transplants.")
print("INTERPRETATION: Removing H changes upper states; removing early KV changes subsequent attention inputs.")
print("DECISION: REVIEW H/KV ABLATION MATRIX BEFORE TEST568")
print("="*130)
if not INTEGRITY or not WEIGHT_OK or TRAINABLE!=0:raise RuntimeError("TEST567 FINAL INTEGRITY FAILURE")
