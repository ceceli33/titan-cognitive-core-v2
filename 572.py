
# TEST572 — AKBASCORE MAM · MEMORY SCALE × TWO-HOP REASONING × UNKNOWN CONTROL
# VERIFIED TEST564 ENGINE · CUT3 · FROZEN QWEN · BF16 · GREEDY · NO SOURCE REPLAY AT ASK
# 5/10/20/40 CARTRIDGES · 3 TASK FAMILIES · DC/H_FULL · APPEND-ONLY · 24×4×2 = 192 ANSWERS
import urllib.request,hashlib,time,gc,json,traceback,random,re
BASE_FILE="564_qwen_CALIBRATION_start.py";BASE_BLOB="76dcf80c891640314cbe1849ddffebfae287a23e"
BASE_URL="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/"+BASE_FILE
print("="*132);print("TEST572 — MEMORY SCALE × TWO-HOP REASONING × UNKNOWN CONTROL");print("="*132)
req=urllib.request.Request(BASE_URL,headers={"User-Agent":"AKBASCORE-TEST572"})
with urllib.request.urlopen(req,timeout=90) as f:raw=f.read()
blob=hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
if blob!=BASE_BLOB:raise RuntimeError("BASELINE BLOB MISMATCH: "+blob)
src=raw.decode();marker="# Fixed short calibration: 8 cases x 3 positions = 24 questions per CUT"
if src.count(marker)!=1:raise RuntimeError("BASELINE BOUNDARY MISMATCH")
exec(compile(src.split(marker,1)[0],BASE_FILE,"exec"),globals())
TEST="572";CUT=3;SIZES=[5,10,20,40];ARMS=["DC","H_FULL"];FAMILIES=["DIRECT","HOP2","UNKNOWN"];REPEATS=8
assert sentinel()==S0 and not any(p.requires_grad for p in model.parameters())
OLD_CAR=CAR;OLD_W=W;OLD_KEY=KEY;OLD_PANEL=PANEL
# Each entity has a distinct code. Independent cartridges encode A->B and B->C.
# Unknown asks for a deliberately absent relation; expected answer is UNKNOWN.
RNG=random.Random(572572)
ALPHA="BCDFGHJKLMNPQRSTVWXYZ"
def word(i,prefix):
    r=random.Random(572000+i+sum(ord(x) for x in prefix)*997)
    return prefix+"".join(r.choice(ALPHA) for _ in range(7))
N=64
A=[word(i,"A") for i in range(N)]
B=[word(i,"B") for i in range(N)]
C=[word(i,"C") for i in range(N)]
D=[word(i,"D") for i in range(N)]
assert len(set(A+B+C+D))==4*N
SYSTEM="Answer using only the stored information. For a missing fact answer UNKNOWN. Give only the requested name or UNKNOWN."
PREFIX=f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\nINFORMATION:\n"
PIDS=tok(PREFIX,add_special_tokens=False).input_ids;P=len(PIDS)
CAR=[]
def add(kind,i,fact):
    ids=tok(source(fact),add_special_tokens=False).input_ids
    if ids[:P]!=PIDS:raise RuntimeError("PREFIX MISMATCH")
    CAR.append(dict(type=kind,entity=i,fact=fact,body=ids[P:]))
for i in range(N):
    add("AB",i,f"The relay destination of {A[i]} is {B[i]}.")
    add("BC",i,f"The archive location of {B[i]} is {C[i]}.")
    add("DECOY",i,f"The ceremonial title of {D[i]} is {word(i,'T')}.")
assert len(CAR)==N*3
FACT_SHA=sha([(c["type"],c["entity"],c["fact"]) for c in CAR])
print("CUT:",CUT,"NEW FACT SHA:",FACT_SHA,"PREFIX:",P,"CARTRIDGE POOL:",len(CAR))
print("SIZES:",SIZES,"FAMILIES:",FAMILIES,"REPEATS:",REPEATS,"ARMS:",ARMS)
# A pair of linked cartridges is always present in DIRECT/HOP2.
# UNKNOWN contains no AB cartridge for the queried A, while other facts are present.
TASKS=[]
for size in SIZES:
    for family in FAMILIES:
        for rep in range(REPEATS):
            rr=random.Random(572572+size*10000+rep*101+FAMILIES.index(family)*1009)
            target=(rep*7+size)%N
            if family=="UNKNOWN":
                pool=[i for i in range(N) if i!=target]
                rr.shuffle(pool)
                keys=[3*i for i in pool[:min(size,N-1)]]
                if len(keys)<size:
                    extra=[3*i+1 for i in pool if 3*i+1 not in keys]
                    keys+=extra[:size-len(keys)]
                q=f"What is the relay destination of {A[target]}?"
                gold="UNKNOWN"
            else:
                keys=[3*target,3*target+1]
                distractors=[3*i+j for i in range(N) if i!=target for j in (0,1,2)]
                rr.shuffle(distractors);keys+=distractors[:size-2]
                if family=="DIRECT":
                    q=f"What is the relay destination of {A[target]}?";gold=B[target]
                else:
                    q=f"The relay destination of {A[target]} has an archive location. What is that archive location?";gold=C[target]
            rr.shuffle(keys)
            assert len(keys)==size and len(set(keys))==size
            TASKS.append(dict(size=size,family=family,rep=rep,target=target,keys=keys,q=q,gold=gold))
assert len(TASKS)==len(SIZES)*len(FAMILIES)*REPEATS
print("TOTAL TASKS:",len(TASKS),"TOTAL ANSWERS:",len(TASKS)*len(ARMS))
def inject(out,h):
    x=out[0] if isinstance(out,(tuple,list)) else out
    if x.shape!=h.shape:raise RuntimeError("H SHAPE FAIL")
    return replace(out,h)
@torch.no_grad()
def h_init(ci):
    cp=checkpoint(ci,CUT);n=cp["h"].shape[1];pos=torch.arange(n,device=DEVICE)
    hook=model.model.layers[CUT].register_forward_hook(lambda m,a,o:inject(o,cp["h"]))
    try:
        out=model(inputs_embeds=torch.zeros((1,n,H),device=DEVICE,dtype=DTYPE),attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        return clone(out.past_key_values)
    finally:hook.remove()
@torch.no_grad()
def h_append(memory,ci):
    cp=checkpoint(ci,CUT);oldn=memory[0][0].shape[-2];q=cp["body"];pos=torch.arange(oldn,oldn+q,device=DEVICE);body=cp["h"][:,P:,:]
    hook=model.model.layers[CUT].register_forward_hook(lambda m,a,o:inject(o,body))
    try:
        out=model(inputs_embeds=torch.zeros((1,q,H),device=DEVICE,dtype=DTYPE),past_key_values=cache_build(memory),attention_mask=torch.ones((1,oldn+q),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        result=clone(out.past_key_values)
        for li,((ok,ov),(nk,nv)) in enumerate(zip(memory,result)):
            if not torch.equal(ok,nk[:,:,:oldn,:]) or not torch.equal(ov,nv[:,:,:oldn,:]):raise RuntimeError(f"APPEND-ONLY FAIL L{li}")
        return result
    finally:hook.remove()
@torch.no_grad()
def build(keys,arm):
    if arm=="DC":
        mem=init(keys[0],CUT)
        for ci in keys[1:]:mem=append(mem,ci,CUT)
    else:
        mem=h_init(keys[0])
        for ci in keys[1:]:mem=h_append(mem,ci)
    n=P+sum(len(CAR[i]["body"]) for i in keys)
    for li,(k,v) in enumerate(mem):
        if k.shape!=(1,KVH,n,HD) or v.shape!=(1,KVH,n,HD):raise RuntimeError(f"CACHE SHAPE FAIL L{li}")
    return mem
# Strict exact-answer scoring. UNKNOWN must be the sole answer.
def exact(a,g):
    return norm(a)==norm(g)
START=time.perf_counter();RAW=[];ERRORS=[]
for size in SIZES:
    for arm in ARMS:
        print("\n"+"="*112);print(f"SIZE={size} ARM={arm}",flush=True)
        t=time.perf_counter()
        for task in [x for x in TASKS if x["size"]==size]:
            try:
                mem=build(task["keys"],arm)
                a=answer(mem,task["q"]);ok=exact(a,task["gold"])
                row=dict(size=size,arm=arm,family=task["family"],rep=task["rep"],gold=task["gold"],answer=a,ok=ok)
                RAW.append(row)
                print(f"N={size:02d} {arm:7s} {task['family']:7s} REP={task['rep']:02d} {'PASS' if ok else 'FAIL'} | gold={task['gold']} | answer={a!r}",flush=True)
                del mem
            except Exception as e:
                ERRORS.append(dict(size=size,arm=arm,family=task["family"],rep=task["rep"],error=repr(e)))
                print("ERROR:",repr(e),flush=True)
                raise
        print("ARM TIME:",round(time.perf_counter()-t,2),"s",flush=True)
        gc.collect();torch.cuda.empty_cache()
print("\n"+"="*132);print("TEST572 — SCALE / REASONING / UNKNOWN SCORECARD");print("="*132)
for size in SIZES:
    for arm in ARMS:
        parts=[]
        for fam in FAMILIES:
            rows=[r for r in RAW if r["size"]==size and r["arm"]==arm and r["family"]==fam]
            parts.append(f"{fam}={sum(r['ok'] for r in rows)}/{REPEATS}")
        print(f"N={size:02d} {arm:7s} "+" | ".join(parts))
ref={(r["size"],r["family"],r["rep"]):r for r in RAW if r["arm"]=="DC"}
hrows=[r for r in RAW if r["arm"]=="H_FULL"]
IDENTICAL=sum(norm(r["answer"])==norm(ref[(r["size"],r["family"],r["rep"])]["answer"]) for r in hrows)
INTEGRITY=len(RAW)==len(TASKS)*len(ARMS) and len(ERRORS)==0
WEIGHT_OK=sentinel()==S0;TRAINABLE=sum(int(p.requires_grad) for p in model.parameters())
LOCK=dict(test=TEST,base_blob=blob,cut=CUT,sizes=SIZES,arms=ARMS,families=FAMILIES,repeats=REPEATS,fact_sha=FACT_SHA,task_sha=sha(TASKS))
RECORD=dict(lock=LOCK,raw=RAW,errors=ERRORS,integrity=INTEGRITY,weight_ok=WEIGHT_OK)
print("ANSWER ID H_FULL vs DC:",IDENTICAL,"/",len(TASKS))
print("LOCK SHA:",sha(LOCK));print("RESULT SHA:",sha(RECORD))
print("PANEL INTEGRITY:","PASS" if INTEGRITY else "FAIL")
print("WEIGHT SENTINEL:","PASS" if WEIGHT_OK else "FAIL","TRAINABLE:",TRAINABLE)
print("TOTAL SECONDS:",round(time.perf_counter()-START,2))
print("NOTE: UNKNOWN is strict abstention; failure is informative, not automatically a memory-corruption event.")
print("="*132)
if not INTEGRITY or not WEIGHT_OK or TRAINABLE:raise RuntimeError("TEST572 INTEGRITY FAILURE")
