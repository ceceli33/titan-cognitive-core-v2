
# TEST571 — AKBASCORE MAM · QWEN NEW-FACT EXTERNAL PANEL + DETERMINISTIC REBUILD
# VERIFIED TEST564 BACKBONE · CUT3 · 12 NEW WORLDS · 48 NEW CARTRIDGES
# DC / H_FULL · 12 CASES × 3 POSITIONS × 2 ARMS = 72 ANSWERS
# FROZEN BF16 · GREEDY · APPEND-ONLY · NO SOURCE REPLAY AT ASK
import urllib.request,hashlib,time,gc,json,traceback,random
BASE_FILE="564_qwen_CALIBRATION_start.py";BASE_BLOB="76dcf80c891640314cbe1849ddffebfae287a23e"
BASE_URL="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/"+BASE_FILE
EXPECTED_PANEL="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
print("="*132);print("TEST571 — QWEN NEW-FACT EXTERNAL VALIDATION + REBUILD INTEGRITY");print("="*132)
req=urllib.request.Request(BASE_URL,headers={"User-Agent":"AKBASCORE-TEST571","Accept":"text/plain"})
with urllib.request.urlopen(req,timeout=90) as f:raw=f.read()
blob=hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
if blob!=BASE_BLOB:raise RuntimeError("BASELINE BLOB MISMATCH: "+blob)
src=raw.decode("utf-8");marker="# Fixed short calibration: 8 cases x 3 positions = 24 questions per CUT"
if src.count(marker)!=1:raise RuntimeError("BASELINE BOUNDARY MISMATCH")
base=src.split(marker,1)[0]
for fn in ("checkpoint","init","append","answer","cache_build","replace","sentinel","ordered","clone"):
    if f"def {fn}(" not in base:raise RuntimeError("MISSING BASELINE FUNCTION: "+fn)
exec(compile(base,BASE_FILE,"exec"),globals())
TEST="571";CUT=3;ARMS=["DC","H_FULL"];SLOTS=["FIRST","MIDDLE","LAST"]
if PANEL_SHA!=EXPECTED_PANEL or sentinel()!=S0:raise RuntimeError("BASELINE INTEGRITY FAIL")
if any(p.requires_grad for p in model.parameters()):raise RuntimeError("TRAINABLE WEIGHTS")
OLD_CAR=CAR;OLD_W=W;OLD_KEY=KEY;OLD_PANEL=PANEL;OLD_PANEL_SHA=PANEL_SHA
NEW_BASE=[
("Axmeron","Pavrik","Telsora","Dumerek","Yovrath"),
("Beltrix","Nolmera","Feskir","Vandrel","Qerovan"),
("Cavorek","Rildan","Moxerin","Sulpeth","Zemora"),
("Delmara","Kovren","Palsith","Nurevik","Fexarin"),
("Ervokan","Tarmel","Velsora","Jundrik","Mekovar"),
("Faldrix","Sorvek","Nelmora","Kavreth","Yildora"),
("Gavmerek","Peldrin","Vosmera","Caltrek","Ravolin"),
("Helvaron","Daskir","Merovin","Tolveth","Zanorek"),
("Istraven","Bolmera","Kervik","Noldrin","Pavorel"),
("Javrekon","Silmora","Tavrik","Qeldren","Vasorek"),
("Kelmorin","Dorveth","Raskira","Melvorn","Teskara"),
("Luvarek","Naskorin","Veltrix","Pomerek","Dovrila")]
assert len(NEW_BASE)==12
OLD_WORDS={v.casefold() for row in BASE for v in row}
NEW_WORDS=[v.casefold() for row in NEW_BASE for v in row]
if len(NEW_WORDS)!=len(set(NEW_WORDS)) or any(v in OLD_WORDS for v in NEW_WORDS):raise RuntimeError("NEW-FACT NAME COLLISION")
W=[];CAR=[];KEY={}
for wi,(s,current,near,former,role) in enumerate(NEW_BASE):
    facts=[("CURRENT",f"The current capital of {s} is {current}.",current),("NEAR",f"The largest city of {s} is {near}.",near),("FORMER",f"The former capital of {s} was {former}.",former),("ROLE",f"The current capital of {role} is {s}.",s)]
    queries={"CURRENT":f"What is the current capital of {s}?","FORMER":f"What was the former capital of {s}?","NEAR":f"What is the largest city of {s}?","ROLE":f"What is the current capital of {role}?"}
    W.append(queries)
    for typ,f,gold in facts:
        KEY[(wi,typ)]=len(CAR);CAR.append(dict(world=wi,type=typ,fact=f,gold=gold))
for c in CAR:
    ids=tok(source(c["fact"]),add_special_tokens=False).input_ids
    if ids[:P]!=PIDS:raise RuntimeError("NEW PREFIX MISMATCH")
    c["body"]=ids[P:]
rng=random.Random(571571);pool=list(range(len(CAR)));rng.shuffle(pool)
PANEL=[];cycle=["CURRENT","FORMER","NEAR","ROLE"]
for wi in range(len(NEW_BASE)):
    typ=cycle[wi%4];target=KEY[(wi,typ)]
    candidates=[ci for ci in pool if CAR[ci]["world"]!=wi]
    others=[];worlds=set()
    for ci in candidates:
        cw=CAR[ci]["world"]
        if cw not in worlds:others.append(ci);worlds.add(cw)
        if len(others)==4:break
    if len(others)!=4:raise RuntimeError("DISTRACTOR GENERATION FAIL")
    PANEL.append(dict(case=wi,target=target,others=others))
NEW_PANEL_SHA=sha(PANEL);NEW_FACT_SHA=sha([(c["type"],c["fact"],c["gold"]) for c in CAR])
print("MODEL:",MODEL_ID,"CUT:",CUT,"NEW WORLDS:",len(NEW_BASE),"NEW CARTRIDGES:",len(CAR))
print("OLD PANEL SHA:",OLD_PANEL_SHA,"NEW PANEL SHA:",NEW_PANEL_SHA,"NEW FACT SHA:",NEW_FACT_SHA)
print("CASES:",len(PANEL),"POSITIONS:",SLOTS,"ARMS:",ARMS,"TOTAL ANSWERS:",len(PANEL)*len(SLOTS)*len(ARMS))
print("FACTS: NEW SYNTHETIC NAMES; SAME FACT/QUESTION GRAMMAR AS TEST564")
print("CONTROL: TWO INDEPENDENT SERIAL REBUILDS; NOT A PARALLEL BATCH EQUIVALENCE TEST")
def inject_h(out,h):
    current=out[0] if isinstance(out,(tuple,list)) else out
    if current.shape!=h.shape:raise RuntimeError("H SHAPE FAIL")
    return replace(out,h)
@torch.no_grad()
def h_init(ci):
    cp=checkpoint(ci,CUT);n=cp["h"].shape[1];pos=torch.arange(n,device=DEVICE)
    hook=model.model.layers[CUT].register_forward_hook(lambda module,args,out:inject_h(out,cp["h"]))
    try:
        out=model(inputs_embeds=torch.zeros((1,n,H),device=DEVICE,dtype=DTYPE),attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        return clone(out.past_key_values)
    finally:hook.remove()
@torch.no_grad()
def h_append(memory,ci):
    cp=checkpoint(ci,CUT);oldn=memory[0][0].shape[-2];q=cp["body"];pos=torch.arange(oldn,oldn+q,device=DEVICE);body=cp["h"][:,P:,:]
    if body.shape!=(1,q,H):raise RuntimeError("BODY SHAPE FAIL")
    hook=model.model.layers[CUT].register_forward_hook(lambda module,args,out:inject_h(out,body))
    try:
        out=model(inputs_embeds=torch.zeros((1,q,H),device=DEVICE,dtype=DTYPE),past_key_values=cache_build(memory),attention_mask=torch.ones((1,oldn+q),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        result=clone(out.past_key_values)
        for li,((ok,ov),(nk,nv)) in enumerate(zip(memory,result)):
            if not torch.equal(ok,nk[:,:,:oldn,:]) or not torch.equal(ov,nv[:,:,:oldn,:]):raise RuntimeError(f"H APPEND-ONLY FAIL L{li}")
        return result
    finally:hook.remove()
@torch.no_grad()
def build(keys,arm):
    if arm=="DC":
        memory=init(keys[0],CUT)
        for ci in keys[1:]:memory=append(memory,ci,CUT)
    elif arm=="H_FULL":
        memory=h_init(keys[0])
        for ci in keys[1:]:memory=h_append(memory,ci)
    else:raise RuntimeError("UNKNOWN ARM")
    n=P+sum(len(CAR[ci]["body"]) for ci in keys)
    if len(memory)!=NL:raise RuntimeError("LAYER COUNT FAIL")
    for li,(k,v) in enumerate(memory):
        if k.shape!=(1,KVH,n,HD) or v.shape!=(1,KVH,n,HD):raise RuntimeError(f"CACHE SHAPE FAIL L{li}")
        if not bool(torch.isfinite(k).all()) or not bool(torch.isfinite(v).all()):raise RuntimeError(f"NONFINITE CACHE L{li}")
    return memory
def equal_cache(a,b):
    if len(a)!=len(b):return False
    return all(torch.equal(ak,bk) and torch.equal(av,bv) for (ak,av),(bk,bv) in zip(a,b))
START=time.perf_counter();RAW=[];RESULTS=[];REBUILD=[];ERRORS=[]
for arm in ARMS:
    rows=[];t=time.perf_counter();err=None
    print("\n"+"="*112);print("ARM:",arm,flush=True)
    try:
        for wi,z in enumerate(PANEL):
            typ=CAR[z["target"]]["type"];q=W[wi][typ];gold=CAR[z["target"]]["gold"]
            for slot in SLOTS:
                keys=ordered(z,slot);mem=build(keys,arm)
                a=answer(mem,q);ok=hit(a,gold)
                rows.append(dict(arm=arm,case=wi,slot=slot,gold=gold,answer=a,ok=ok))
                print(f"{arm:7s} CASE={wi:02d} {slot:6s} {'PASS' if ok else 'FAIL'} | gold={gold} | answer={a!r}",flush=True)
                del mem
        for wi in (0,5,11):
            for slot in SLOTS:
                keys=ordered(PANEL[wi],slot)
                a=build(keys,arm);b=build(keys,arm)
                same=equal_cache(a,b);REBUILD.append(dict(arm=arm,case=wi,slot=slot,bitwise=same))
                print(f"REBUILD {arm:7s} CASE={wi:02d} {slot:6s} {'BITWISE PASS' if same else 'BITWISE FAIL'}",flush=True)
                del a,b
                if not same:raise RuntimeError(f"REBUILD BITWISE FAIL ARM={arm} CASE={wi} SLOT={slot}")
    except Exception as e:
        err=repr(e);ERRORS.append(dict(arm=arm,error=err,traceback=traceback.format_exc()))
        print(traceback.format_exc(),flush=True)
    RAW.extend(rows)
    scores={s:sum(int(r["ok"]) for r in rows if r["slot"]==s) for s in SLOTS}
    result=dict(arm=arm,score=sum(scores.values()),total=36,scores=scores,complete=len(rows)==36 and err is None,error=err,seconds=round(time.perf_counter()-t,2))
    RESULTS.append(result)
    print(f"RESULT {arm} SCORE={result['score']}/36 FIRST={scores['FIRST']}/12 MIDDLE={scores['MIDDLE']}/12 LAST={scores['LAST']}/12 COMPLETE={result['complete']}",flush=True)
    if not result["complete"]:raise RuntimeError(f"INCOMPLETE ARM {arm}: {err}")
    torch.cuda.empty_cache();gc.collect()
print("\n"+"="*132);print("TEST571 — FINAL NEW-FACT VALIDATION");print("="*132)
ref={(r["case"],r["slot"]):r for r in RAW if r["arm"]=="DC"}
for result in RESULTS:
    rows=[r for r in RAW if r["arm"]==result["arm"]]
    identical=sum(norm(r["answer"])==norm(ref[(r["case"],r["slot"])]["answer"]) for r in rows)
    print(f"{result['arm']:7s} SCORE={result['score']:02d}/36 ANSWER_ID_VS_DC={identical:02d}/36 FIRST={result['scores']['FIRST']}/12 MIDDLE={result['scores']['MIDDLE']}/12 LAST={result['scores']['LAST']}/12")
WEIGHT_OK=sentinel()==S0;TRAINABLE=sum(int(p.requires_grad) for p in model.parameters())
REBUILD_OK=len(REBUILD)==18 and all(r["bitwise"] for r in REBUILD)
INTEGRITY=len(RAW)==72 and len({(r["arm"],r["case"],r["slot"]) for r in RAW})==72 and all(r["complete"] for r in RESULTS) and not ERRORS
LOCK=dict(test=TEST,base_blob=blob,cut=CUT,old_panel_sha=OLD_PANEL_SHA,new_panel_sha=NEW_PANEL_SHA,new_fact_sha=NEW_FACT_SHA,arms=ARMS,slots=SLOTS,seed=SEED)
RECORD=dict(lock=LOCK,results=RESULTS,rebuild=REBUILD,integrity=INTEGRITY,weight_ok=WEIGHT_OK,trainable=TRAINABLE)
print("LOCK SHA:",sha(LOCK));print("RESULT SHA:",sha(RECORD))
print("NEW PANEL INTEGRITY:","PASS" if INTEGRITY else "FAIL")
print("SERIAL REBUILD BITWISE:","PASS" if REBUILD_OK else "FAIL")
print("WEIGHT SENTINEL:","PASS" if WEIGHT_OK else "FAIL","TRAINABLE:",TRAINABLE)
print("TOTAL SECONDS:",round(time.perf_counter()-START,2))
print("SCOPE: NEW SYNTHETIC FACTS, EXISTING QUESTION GRAMMAR, SERIAL REBUILD ONLY")
print("="*132)
if not INTEGRITY or not REBUILD_OK or not WEIGHT_OK or TRAINABLE:raise RuntimeError("TEST571 INTEGRITY FAILURE")
