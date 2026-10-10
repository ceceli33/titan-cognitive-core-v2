# =====================================================================================================================================
# AKBASCORE MAM-BÇ · EXTERNAL ASSOCIATIVE MEMORY FOR A FROZEN LANGUAGE MODEL — QWEN2.5-7B · PART 3 / 3 · SELF-TEST AND GRADIO DEMONSTRATION
# Same demo as PARTS 1 and 2. Run this cell after PART 2 / 3, in the same Colab runtime. It runs a GPU-free self-test of the full
# pipeline (including fail-closed cases) and then prints a public gradio.live link. Choose a panel case, press RUN.
# Discovered and developed by Mustafa Akbaş · AkbasCore AI Teknoloji · 10 October 2026 · AKBASCORE RESEARCH SOFTWARE LICENSE
# =====================================================================================================================================
if not globals().get("MAM_BC_PART2_OK"):raise RuntimeError("PART 2 / 3 has not been run in this runtime. Run PART 1, then PART 2, then PART 3.")
#<<UI_BEGIN>>
_SYN_A=["Aq","Bryn","Cav","Dex","Eln","Frax","Gyr","Hex","Iln","Jor","Kex","Lyr"];_SYN_B=["adar","bren","cyr","dax","elor","fyn","grel","hyn"]
class _SynMem:
    """Synthetic memory handle for the self-test (no tensors)."""
    def __init__(s,keys,T,salt=""):s.keys=list(keys);s.T=T;s.salt=salt
    def __eq__(s,o):return isinstance(o,_SynMem)and s.keys==o.keys and s.T==o.T and s.salt==o.salt
class SelfTestInstrument:
    """Synthetic stand-in used ONLY by the service self-test (no GPU, no model). Never used for a real run."""
    def __init__(s,mode="ok"):
        s.kind="SELF-TEST";s.mode=mode;s.calls=0;names=[a+b for a in _SYN_A for b in _SYN_B];s.names=names;s.P=PREFIX_EXPECTED;s.cars=[]
        for w in range(24):
            subj,cur,near,form,role=names[w*4],names[w*4+1],names[w*4+2],names[w*4+3],names[(w*4+5)%len(names)]
            for typ,fact,gold in(("CURRENT",f"The current capital of {subj} is {cur}.",cur),("NEAR",f"The largest city of {subj} is {near}.",near),
                                 ("FORMER",f"The former capital of {subj} was {form}.",form),("ROLE",f"The current capital of {role} is {subj}.",subj)):
                ci=len(s.cars);s.cars.append(dict(id=ci,world=w,type=typ,fact=fact,gold=gold,body_len=10+ci%3))
        s.qs=[]
        for w in range(24):
            c=s.cars[s.target(w)];subj=names[w*4]
            s.qs.append({"CURRENT":f"What is the current capital of {subj}?","FORMER":f"What was the former capital of {subj}?","NEAR":f"What is the largest city of {subj}?","ROLE":f"What is the current capital of {names[(w*4+5)%len(names)]}?"}[c["type"]])
        s.info=dict(model_id=MODEL_ID,arch=list(ARCH),dtype="bfloat16",attn="sdpa",gpu="SELF-TEST INSTRUMENT (no GPU)",gpu_total_gib=0.0,torch="n/a",transformers="n/a",gradio="n/a",python="n/a",
            platform="n/a",params=1,engine_file=ENGINE_FILE,engine_sha256=ENGINE_SHA256_EXPECTED,engine_blob=ENGINE_BLOB_EXPECTED,engine_exec_sha256="e"*64,engine_tail_sha256="f"*64,
            helper_sha256=HELPER_SHA256_EXPECTED,raw_demo_file_sha256=RAW_DEMO_FILE_SHA256,init_seconds=0.0,startup_utc="n/a",source_blobs=dict(SOURCE_BLOBS_EXPECTED),
            weight_sha0=("d"*64 if mode=="weight_ref" else WEIGHT_SHA_EXPECTED),sentinel0="a"*64,sentinel_engine_load="a"*64,sentinel_610_start="a"*64,guard0="b"*64,hooks0=0,
            sentinel_method="synthetic",guard_method="synthetic",full_weight_method="synthetic",forward_counter="synthetic",observer="synthetic")
    def target(s,w):return w*4+{"CURRENT":0,"NEAR":1,"FORMER":2,"ROLE":3}[["CURRENT","FORMER","NEAR","ROLE"][w%4]]
    def config(s):
        return dict(MODEL_ID=MODEL_ID,CUT=CUT_EXPECTED,ENGINE_CUTS=[1,2,3,6],MAX_NEW=16,SEED=552552,PANEL_SEED=550550,EXPECTED_PANEL_SHA=EXPECTED_PANEL_SHA,panel_sha=EXPECTED_PANEL_SHA,prefix_tokens=s.P,
                    n_cartridges=96,n_cases=24,system="synthetic",arch=list(ARCH),engine_sha256=ENGINE_SHA256_EXPECTED,engine_blob=ENGINE_BLOB_EXPECTED,engine_exec_sha256="e"*64,helper_sha256=HELPER_SHA256_EXPECTED,
                    source_blobs=dict(SOURCE_BLOBS_EXPECTED),source_blobs_measured=dict(SOURCE_BLOBS_EXPECTED),fram_cfg=[8,0,8.0]if s.mode=="config"else list(FRAM_LOCK),fram_layers=[3,8,16,24],fram_ranks=[0,64,128],
                    fram_betas=[0.0,8.0],n_configs=24,train=list(TRAIN_CASES),hold=list(HOLD_CASES),beta=8.0,M=M_REF,D=D_REF,top_k=TOP_K)
    def status(s):return{"title":"AKBASCORE MAM-BÇ","reference_test":"TEST610","fram":list(FRAM_LOCK),"panel_sha":EXPECTED_PANEL_SHA}
    def _sc(s,case,seed):
        rng=np.random.default_rng(1000*seed+case);v=rng.normal(0.30,0.04,96);t=s.target(case);v[t]=v.max()+(0.05 if seed==1 else 0.02)
        if seed==2 and case==5:v[t]=np.sort(v)[-8]-1e-4
        return[float(x)for x in v]
    def fram_scores(s,case):return s._sc(case,1)
    def ref_scores(s,case):return s._sc(case,2)
    def rankings(s,scores):return sorted(range(len(scores)),key=lambda i:(-scores[i],i))
    def fram_shapes(s,case):return dict(query=[9+case%2,3584],keys=[[c["body_len"]-2,3584]for c in s.cars])
    def startup(s):
        rows=[]
        for w in range(24):
            o=s.rankings(s.fram_scores(w));rows.append(dict(case=w,split=split_of(w),target=s.target(w),rank=o.index(s.target(w))+1,top5=o[:5]))
        base=[dict(case=w,target=s.target(w),present=1,ok=1,answer=s.cars[s.target(w)]["gold"])for w in HOLD_CASES]
        return dict(cfg=list(FRAM_LOCK),locked=dict(cfg=list(FRAM_LOCK),hit5=12,hit1=12,rank_sum=12,order=1),calibration=[],n_configs=24,hold_r1=11 if s.mode=="startup"else 12,hold_r5=12,all_r5=24,
                    self_r1=96,self_n=96,hub_max=4,ref_hold_r1=11,ref_hold_r5=12,cases=rows,baseline=base,baseline_correct=12,baseline_retrieved=12,baseline_n=12,question_tokens=[9]*24)
    def cases(s):return[{"case":w,"split":split_of(w),"type":s.cars[s.target(w)]["type"],"question":s.qs[w],"target":s.target(w)}for w in range(24)]
    def question(s,case):return s.qs[case]
    def car(s,ci):return dict(s.cars[ci])
    def source_ids(s,ci):return list(range(1,s.P+1))+[1000+ci*20+j for j in range(s.cars[ci]["body_len"])]
    def _toks(s,text):return re.findall(r"\n|[^\s\n]+",text)
    def _ids(s,text,base):
        ids=[];s.__dict__.setdefault("_tokmap",{})
        for i,t in enumerate(s._toks(text)):tid=base+(sum(map(ord,t))*31+i)%50000;s._tokmap[tid]=t;ids.append(tid)
        return ids
    def query_ids(s,q):return s._ids(f"\nQUESTION:\n{q}\n\nANSWER:\n<|im_end|>\n<|im_start|>assistant\n",2000)
    def frame_ids(s,q):return list(range(1,s.P+1))+s._ids(q+"\n\n",60000)
    def decode_each(s,ids):return[s._tokmap.get(t,"?")for t in ids]
    def panel_strings(s):return sorted(set(s.names))
    def hit(s,a,g):return hit_ans(a,g)
    def sentinel(s):return"a"*64
    def guard(s):s.calls+=1;return"c"*64 if(s.mode=="guard_changes"and s.calls>1)else"b"*64
    def frozen(s):
        s.__dict__["fz_calls"]=s.__dict__.get("fz_calls",0)+1
        return dict(training=False,trainable_tensors=0,lora=False,optimizer=False,hooks=0,foreign_hooks=0,foreign_modules={},sentinel="a"*64,
                    residual_hooks=(["L3:layer"]if(s.mode=="residual_hook"and s.fz_calls>1)else[]))
    def full_weight_sha(s):return s.info["weight_sha0"]
    def build(s,keys):
        keys=[int(k)for k in keys];steps=[];allf=[];old=0
        for n_,ci in enumerate(keys):
            src=s.source_ids(ci)
            if n_==0:
                f=[dict(kind="ids",n=len(src),ids=src,embeds_zero=None,has_past=False,past_len=None),dict(kind="embeds",n=len(src),ids=None,embeds_zero=True,has_past=False,past_len=None)];new=len(src)
                st=dict(kind="init",cartridge_id=ci,old_len=0,new_len=new,prior_rows_bitwise_unchanged=None,prior_sha_before=None,prior_sha_after=None)
            else:
                q=len(src)-s.P;new=old+q
                f=[dict(kind="ids",n=len(src),ids=src,embeds_zero=None,has_past=False,past_len=None),dict(kind="embeds",n=q,ids=None,embeds_zero=True,has_past=True,past_len=old)]
                if s.mode=="reread"and n_==2:f.insert(1,dict(kind="ids",n=len(s.source_ids(keys[0])),ids=s.source_ids(keys[0]),embeds_zero=None,has_past=False,past_len=None))
                same=not(s.mode=="prior_changed"and n_==3)
                st=dict(kind="append",cartridge_id=ci,old_len=old,new_len=new,prior_rows_bitwise_unchanged=same,prior_sha_before=hashlib.sha256(f"m{keys[:n_]}".encode()).hexdigest(),
                        prior_sha_after=hashlib.sha256(f"m{keys[:n_]}{'' if same else 'x'}".encode()).hexdigest())
            st.update(memory_sha=hashlib.sha256(f"m{keys[:n_+1]}".encode()).hexdigest(),seconds=0.04,forwards=f);steps.append(st);allf+=f;old=new
        return _SynMem(keys,old),dict(steps=steps,forwards=allf,keys=keys),0.2
    def ask(s,mem,q):
        ids=s.query_ids(q);case=next((w for w in range(24)if s.qs[w]==q),None)
        if s.mode=="source_in_query":ids=s.source_ids(mem.keys[0])+ids
        if case is None:ans="Not provided in the information."
        else:
            t=s.target(case);ans=s.cars[t]["gold"]if(t in mem.keys and case%7!=3)else s.names[(case*4+2)%len(s.names)]
            if s.mode=="reload_answer"and mem.salt=="reloaded":ans="Different"
        rec=[dict(kind="ids",n=len(ids),ids=ids,embeds_zero=None,has_past=True,past_len=mem.T)]+[dict(kind="ids",n=1,ids=[7],embeds_zero=None,has_past=True,past_len=mem.T+len(ids)+i)for i in range(2)]
        return ans,rec,0.12
    def retrieve_free(s,q):
        ids=[(sum(map(ord,q))+7*j)%96 for j in range(5)];ids=list(dict.fromkeys(ids))
        while len(ids)<5:ids.append((ids[-1]+1)%96)
        rk=[{"id":i,"score":0.3-0.01*j,"type":s.cars[i]["type"]}for j,i in enumerate(ids)]
        return ids,rk,[dict(kind="ids",n=len(s.frame_ids(q)),ids=s.frame_ids(q),embeds_zero=None,has_past=False,past_len=None)],0.05
    def save(s,mem,ids,path):
        body=json.dumps({"format":"AKBASCORE_CUT3_SESSION_V1","ids":list(ids),"T":mem.T}).encode()+bytes(range(256))*int(mem.T*224)
        if s.mode=="fact_in_file":body+=s.cars[ids[0]]["fact"].encode()
        Path(path).write_bytes(body);return{"path":str(path),"bytes":len(body),"sha256":hashlib.sha256(body).hexdigest()}
    def load(s,path):
        d=json.loads(Path(path).read_bytes().split(b"}",1)[0]+b"}");return _SynMem(d["ids"],d["T"],"reloaded"if s.mode in("reload_diff","reload_answer")else""),list(d["ids"])
    def equal(s,a,b):return a.keys==b.keys and a.T==b.T and(s.mode!="reload_diff")
    def memory_len(s,mem):return mem.T
    def memory_sha(s,mem):return hashlib.sha256(f"m{mem.keys}".encode()).hexdigest()if mem.salt!="reloaded"or s.mode!="reload_diff"else"0"*64
    def memory_bytes(s,mem):return mem.T*ARCH[0]*2*ARCH[3]*ARCH[4]*2
    def memory_profile(s,mem):
        NL,_,_,KVH,HD=ARCH;rng=np.random.default_rng(mem.T);k=(rng.normal(20,3,(NL,mem.T))+np.linspace(0,10,NL)[:,None]).tolist();v=(rng.normal(2,.4,(NL,mem.T))).tolist()
        return dict(layers=NL,T=mem.T,k_norm=k,v_norm=v,bytes=s.memory_bytes(mem),kv_heads=KVH,head_dim=HD,dtype="bfloat16",shape=[1,KVH,mem.T,HD])
    def release(s,*m):pass
    def acceptance_replay(s,progress=None):
        for i in range(44):
            if progress:progress(i+1,44,"DEMO_CASE synthetic")
        cases=[dict(case=r["case"],split=r["split"],answer=r["answer"],correct=r["correct"])for r in TEST610["cases"]]
        return dict(summary=dict(TEST610["final"]),criteria=dict(TEST610["criteria"]),decision=TEST610["decision"],cases=cases,negatives=list(TEST610["negatives"]),rebuild=[],append=[],
                    result_sha="0"*64,out_path="",stdout="synthetic engine stdout\n",seconds=0.1)
    def sync(s):pass
    def reset_peak(s):pass
    def peak_gib(s):return 0.0
def drain(gen):
    while True:
        try:next(gen)
        except StopIteration as s_:return s_.value
RUN_COUNTER=globals().get("RUN_COUNTER",0);GPU_LOCK=threading.Lock();SESSION_LEDGER=globals().get("SESSION_LEDGER",[])
LEDGER_PATH=ROOT/"session_ledger.jsonl"
SKIP=getattr(gr,"skip",None)or gr.update
_K=object()
FILE_KEYS=["zip","json","txt","man","session","ledger"]
DL_LABELS=["⬇ EVIDENCE PACKAGE (.zip)","⬇ FULL RUN LOG (.json)","⬇ READABLE RUN LOG (.txt)","⬇ RUN MANIFEST (.json)","⬇ CUT3 SESSION FILE (.pt)","⬇ SESSION LEDGER (.jsonl)"]
DL_ALL_LABEL=f"⬇ DOWNLOAD ALL {N_FIGS} FIGURES (JPEG, 300 dpi)"
RAW_KEYS=["txt","json","man","ledger"]
N_OUTPUTS=2+2+len(FILE_KEYS)*2+len(RAW_KEYS)
_GR_MAJOR=int(re.match(r"\d+",gr.__version__).group())
try:
    from gradio.routes import API_PREFIX as _GR_API_PREFIX
except Exception:
    _GR_API_PREFIX="/gradio_api"if _GR_MAJOR>=5 else""
FILE_URL_PREFIX=f"{_GR_API_PREFIX}/file="
def dl_update(path,label):
    if path is None:
        try:return gr.DownloadButton(label=label,value=None,interactive=False)
        except Exception:return gr.update(value=None,interactive=False)
    try:return gr.DownloadButton(label=label,value=str(path),interactive=True)
    except Exception:return gr.update(value=str(path),interactive=True)
def btn_update(active):
    try:return gr.Button(value=DL_ALL_LABEL,interactive=bool(active))
    except Exception:return gr.update(value=DL_ALL_LABEL,interactive=bool(active))
def jpg_urls_json(paths):return json.dumps([{"url":FILE_URL_PREFIX+quote(str(Path(p)),safe="/"),"name":Path(p).name}for p in paths],ensure_ascii=False)
def pack(status=_K,gal=_K,files=_K,jpgs=_K,raw=_K):
    out=[SKIP()if status is _K else status,SKIP()if gal is _K else gal]
    if jpgs is _K:out+=[SKIP(),SKIP()]
    elif jpgs is None:out+=[btn_update(False),""]
    else:
        for pth in jpgs:
            if not file_ready(pth):raise RuntimeError(f"figure missing or empty: {pth}")
        out+=[btn_update(True),jpg_urls_json(jpgs)]
    if files is _K:out+=[SKIP()for _ in range(len(FILE_KEYS)*2)]
    elif files is None:out+=[dl_update(None,l)for l in DL_LABELS]+[None]*len(FILE_KEYS)
    else:
        paths=[files.get(k_)for k_ in FILE_KEYS]
        for pth in paths:
            if pth is not None and not file_ready(pth):raise RuntimeError(f"download artifact missing or empty: {pth}")
        out+=[dl_update(p,l)for p,l in zip(paths,DL_LABELS)]+[None if p is None else str(p)for p in paths]
    if raw is _K:out+=[SKIP()for _ in RAW_KEYS]
    elif raw is None:out+=[""]*len(RAW_KEYS)
    else:out+=[raw.get(k_,"")for k_ in RAW_KEYS]
    return tuple(out)
def card_html(title,body,kind="info"):return f'<div class="kz card {kind}"><div class="h">{html.escape(title)}</div><div class="small">{body}</div></div>'
def pbar_html(done,total):
    pc_=100.0*done/max(1,total);return f'<div style="margin-top:6px;background:#e5e7eb;height:10px"><div style="width:{pc_:.1f}%;height:10px;background:#1d4ed8"></div></div><div class="small">{done}/{total}</div>'
def stage_card(e):return card_html(f"{e['stage']}/{N_STAGES} · {e['title']}",html.escape(e["body"])+pbar_html(e["stage"],N_STAGES),"info")
def case_choices(I):return[f"{c['case']:02d} · {c['split']} · {c['type']} · {c['question']}"for c in I.cases()]
def parse_case(v):
    m=re.match(r"\s*(\d{1,2})",str(v or""));return int(m.group(1))if m else None
def ledger_text():return"\n".join(json.dumps(x,ensure_ascii=False)for x in SESSION_LEDGER)
def case_info_html(case_v,I=None):
    I=I or INSTR;c=parse_case(case_v)
    if c is None or not 0<=c<N_CASES:return card_html("Case","Select a panel case.","info")
    t=I.target(c);q=I.question(c);s6=TEST610["cases"][c]
    body=(f'<span class="mono">question</span> {html.escape(q)}<br><span class="mono">split</span> {split_of(c)} ({"calibration case 0–11: used to lock FRAM" if c in TRAIN_CASES else "holdout case 12–23: never used for any choice"})<br>'
          f'<span class="mono">target</span> cartridge #{t:02d} · {html.escape(I.car(t)["fact"])}<br>'
          f'<span class="mono">expected</span> {html.escape(I.car(t)["gold"])} <span style="opacity:.75">(audit metadata; not an input to retrieval, writing or answering)</span><br>'
          f'<span class="mono">sealed TEST610</span> answer {html.escape(s6["answer"])} · {"correct" if s6["correct"] else "<b>incorrect</b>"}<br>'
          '<span class="mono">order</span> FRAM selects five cartridges among 96 → CUT3 memory → answer without the source text → save → reload → answer again → target-removed control.')
    return card_html(f"Case {c:02d} · {split_of(c)}",body,"info")
def engine_card_html(I):
    inf=I.info;cfg=I.config();su=I.startup();F6=TEST610["final"]
    rows=[("model",f"{inf['model_id']} · frozen · {inf['dtype']} · {arch_txt(tuple(cfg['arch']))}"),("GPU",inf["gpu"]),
          ("cartridge bank",f"{cfg['n_cartridges']} facts · FRAM key = layer-{cfg['fram_cfg'][0]} residual rows · locked ({cfg['fram_cfg'][0]}, {cfg['fram_cfg'][1]}, β {cfg['fram_cfg'][2]:g}) on calibration cases 0–11"),
          ("CUT3 memory",f"H{cfg['CUT']} residual + K/V of layers 0–{cfg['CUT']} per cartridge · {upper_txt(cfg['CUT'],cfg['arch'][0])} consolidated append-only"),
          ("engine",f"{inf['engine_file']} (verbatim, executed up to its acceptance stage) · git blob {inf['engine_blob'][:16]}… · 566.py {cfg['source_blobs']['566.py'][:12]}… · 575.py {cfg['source_blobs']['575.py'][:12]}…"),
          ("helpers",f"{HELPER_FILE} · SHA-256 {cfg['helper_sha256'][:16]}…"),
          ("weights",f"full-weight SHA-256 {inf['weight_sha0'][:16]}… (= reference) · sentinel {inf['sentinel0'][:16]}… · all-parameter guard {inf['guard0'][:16]}…"),
          ("start-up",f"FRAM holdout R@1 {su['hold_r1']}/12 · R@5 {su['hold_r5']}/12 · all R@5 {su['all_r5']}/24 · self-retrieval {su['self_r1']}/{su['self_n']} · reference {su['ref_hold_r1']}/12 · {su['ref_hold_r5']}/12 · baseline CUT3 {su['baseline_correct']}/{su['baseline_n']}"),
          ("init",f"{inf['init_seconds']:.1f} s" if isinstance(inf['init_seconds'],float)and math.isfinite(inf['init_seconds'])else"reused")]
    body="".join(f'<span class="mono">{html.escape(a)}</span> {html.escape(str(b))}<br>'for a,b in rows)
    cf=TEST575["counterfactual"]
    body+=(f'<div class="small" style="margin-top:8px"><b>SEALED TEST610 ACCEPTANCE</b> (610.log; not recomputed by single runs)<br>'
           f'{TEST610["decision"]} · {F6["criteria_passed"]}/{F6["criteria_total"]} criteria · holdout {F6["holdout_correct"]}/{F6["holdout_total"]} · whole panel {F6["all_panel_correct"]}/{F6["all_panel_total"]} (case 08 incorrect) · '
           f'self-retrieval {TEST610["fram"]["self_r1"]}/96 · 4 negative controls without leakage · result SHA-256 {TEST610["result_sha"]}<br>'
           f'<b>SEALED TEST575</b> (a different experiment) · JOINT 72/72 · INCR_DC3 72/72 · BATCH_DC3 72/72 · upper-layer K/V changed {cf["upper_kv_changed"]}/72 · answer changed {cf["answer_changed"]}/72</div>')
    return card_html("Engine status · sealed references",body,"on")
READY_HTML=card_html("Ready",f"Choose a panel case, optionally type your own question, and press <b>RUN</b>. One run scores the question against all 96 cartridges (FRAM), builds the CUT3 memory from the five selected cartridges, "
    f"asks the question without the source text, saves the memory to a file, reloads it, answers again, runs the target-removed control, verifies the frozen weights, seals the payload and renders {N_FIGS} figures (JPEG 300 dpi + vector PDF). "
    "Each run is a LIVE demonstration observation; the TEST610 and TEST575 values above are sealed references.","info")
def is_cuda_error(ex):
    s=f"{type(ex).__name__} {ex}".lower();return"cuda"in s or"device-side assert"in s or"cublas"in s
def run_handler(case_v,custom_v):
    global RUN_COUNTER
    c=parse_case(case_v)
    if c is None or not 0<=c<N_CASES:
        yield pack(card_html("Invalid selection","Choose a panel case (00–23).","warn"));return
    if not GPU_LOCK.acquire(blocking=False):
        yield pack(card_html("Busy","Another run is in progress. Please try again in a moment.","warn"));return
    stage_name="initialisation"
    try:
        RUN_COUNTER+=1;run_id=f"{RUN_PREFIX}{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}-{RUN_COUNTER:04d}-C{c:02d}"
        prune_runs(4);run_dir=ROOT/run_id;run_dir.mkdir(parents=True,exist_ok=True)
        preview=[dict(case=x["case"],correct=x["correct"])for x in SESSION_LEDGER]
        gen=execute_run(INSTR,dict(run_id=run_id,run_dir=run_dir,case=c,custom=(custom_v or"")[:300],session=preview+[dict(case=c,correct=None)]));first=True
        while True:
            try:e=next(gen)
            except StopIteration as s:B=s.value;break
            stage_name=f"{e['stage']}/{N_STAGES} {e['title']}"
            yield pack(stage_card(e),[],None,None,None)if first else pack(stage_card(e));first=False
        SESSION_LEDGER.append(B["ledger"]);LEDGER_PATH.write_text(ledger_text()+"\n",encoding="utf-8")
        P=B["P"];L=P["live"];ok=L["correct"];K=L["disk"];N=L["negative"];R=L["retrieval"]
        raw={"txt":B["txt"].read_text(encoding="utf-8"),"json":B["payload"].read_bytes().decode("utf-8"),"man":B["manifest"].read_text(encoding="utf-8"),"ledger":ledger_text()}
        neg=("not applicable (target not selected)"if N.get("skipped")else f'{html.escape(str(N["answer"]))} · expected answer produced: {"yes" if N["gold_leak"] else "no"}')
        body=(f"<b>{html.escape(B['run_id'])}</b> · LIVE DEMONSTRATION RUN<br>"
              f'<span class="mono">question</span> {html.escape(L["question"])}<br>'
              f'<span class="mono">FRAM</span> target #{L["target"]:02d} · exact rank {R["fram"]["rank"]}/96 · reference rank {R["ref"]["rank"]}/96 · selected {L["engine_order"]}<br>'
              f'<span class="mono">answer</span> <b>{html.escape(L["answer"])}</b> · expected {html.escape(L["expected"])} · <b>{"CORRECT" if ok else "INCORRECT"}</b> · sealed TEST610 answer {html.escape(L["sealed_answer"])} (same: {L["matches_sealed"]})<br>'
              f'<span class="mono">memory</span> {L["memory"]["T"]} tokens · {L["memory"]["bytes"]:,} B · token-id forwards during build {L["source_reads"]} (instruction prefix + each cartridge\'s own fact, once) · earlier rows bitwise unchanged at all appends: {all(s_["prior_rows_bitwise_unchanged"] for s_ in L["steps"][1:])}<br>'
              f'<span class="mono">disk</span> {html.escape(K["file"])} · {K["bytes"]:,} B · SHA-256 {K["sha256"][:16]}… · reload bitwise equal: {K["bit_equal"]} · answer after reload identical: {K["answer_identical"]}<br>'
              f'<span class="mono">control</span> target removed → {neg}<br>'
              +(f'<span class="mono">free text</span> {html.escape(L["custom"]["question"])} → {html.escape(L["custom"]["answer"])} (exact FRAM top-5 {L["custom"]["ids"]}; unscored)<br>'if L.get("custom")else"")+
              f"weights unchanged (sentinel + all-parameter guard) · verdict <b>{html.escape(B['verdict'])}</b> · {N_FIGS} figures + PDFs + ZIP · {B['checks']}/{B['checks']} checks PASS<br>"
              f'<span class="mono">payload SHA-256 {B["sha"]}</span>')
        yield pack(card_html(f"{N_STAGES}/{N_STAGES} · Sealed",body,"on"),[(str(p),c_)for p,c_ in B["imgs"]],
                   {"zip":B["zip"],"json":B["payload"],"txt":B["txt"],"man":B["manifest"],"session":B["session"],"ledger":LEDGER_PATH},[p for p,_ in B["imgs"]],raw)
    except Exception as ex:
        print("="*140);print(f"RUN FAILED — stage: {stage_name}");print(f"{type(ex).__name__}: {ex}");traceback.print_exc();print("="*140)
        kind="AUDIT FAIL — nothing was sealed. "if isinstance(ex,AuditFail)else"";partial=getattr(ex,"partial",None)
        if partial:kind+="The raw outputs gathered so far were preserved (FULL RUN LOG). "
        tip="<br><b>CUDA error: Runtime → Restart runtime, then run PARTS 1–3 again.</b>"if is_cuda_error(ex)else"<br>The full traceback is printed in the Colab console."
        card=card_html("Run failed",f"{html.escape(kind)}Stage: {html.escape(stage_name)}<br>{html.escape(type(ex).__name__)}: {html.escape(str(ex)[:600])}{tip}","err")
        if partial:yield pack(card,[],{"json":partial},None,{"txt":traceback.format_exc(),"json":Path(partial).read_text(encoding="utf-8"),"man":"","ledger":ledger_text()})
        else:yield pack(card,[],None,None,None)
    finally:
        plt.close("all");GPU_LOCK.release()
        try:torch.cuda.empty_cache()
        except Exception:pass
REPLAY_LABELS=("⬇ ACCEPTANCE REPLAY (.json)","⬇ ACCEPTANCE REPLAY (.txt)","⬇ ENGINE STDOUT (.txt)")
def replay_handler():
    if not GPU_LOCK.acquire(blocking=False):
        yield card_html("Busy","Another run is in progress. Please try again in a moment.","warn"),*[dl_update(None,l)for l in REPLAY_LABELS];return
    started=False
    try:
        rid=f"{REPLAY_PREFIX}{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}";prune_runs(2,REPLAY_PREFIX);d=ROOT/rid;d.mkdir(parents=True,exist_ok=True)
        st={"done":0,"last":None,"res":None,"err":None}
        def prog(n,tot,line):st["done"]=n;st["last"]=line
        def work():
            try:st["res"]=replay_report(INSTR,rid,d,prog)
            except BaseException as ex:st["err"]=ex;traceback.print_exc()
            finally:GPU_LOCK.release()  # the GPU lock is held until the replay thread itself ends, even if the browser disconnects
        th=threading.Thread(target=work,daemon=True);started=True;th.start()
        while th.is_alive():
            lr=st["last"];tail=(f"<br>last engine line: <span class='mono'>{html.escape(str(lr)[:160])}</span>"if lr else"")
            yield card_html("Acceptance replay running",f"Executing the 610.py acceptance stage [6/7]–[7/7] verbatim: 24 panel cases, four bitwise rebuilds, four negative controls, final audit, weight checks (LIVE; separate from the sealed TEST610 record).{tail}"+pbar_html(st["done"],44),"info"),*[SKIP()for _ in REPLAY_LABELS]
            time.sleep(1.5)
        if st["err"]is not None:raise st["err"]
        R=st["res"];o=R["out"]
        body=(f"<b>LIVE {html.escape(o['decision'])}</b> · criteria {o['criteria_passed']}/{o['criteria_total']} · panel {o['panel_correct']}/24 · holdout {o['holdout_correct']}/12 · {o['seconds']} s · weights unchanged: {o['weights_unchanged']}<br>"
              f"Answers identical to the sealed TEST610 answers: {o['answers_identical_to_sealed_TEST610']}/24. Sealed TEST610: {TEST610['decision']} · {TEST610['final']['criteria_passed']}/15 · panel {TEST610['final']['all_panel_correct']}/24. "
              "The replay is a separate live measurement and does not replace the sealed record.<br>"
              f'<span class="mono">replay SHA-256 {R["sha"]}</span>')
        yield card_html("Acceptance replay complete",body,"on"),dl_update(R["json"],REPLAY_LABELS[0]),dl_update(R["txt"],REPLAY_LABELS[1]),dl_update(R["stdout"],REPLAY_LABELS[2])
    except Exception as ex:
        yield card_html("Acceptance replay failed",f"{html.escape(type(ex).__name__)}: {html.escape(str(ex)[:500])}","err"),*[dl_update(None,l)for l in REPLAY_LABELS]
    finally:
        if not started:GPU_LOCK.release()
        try:torch.cuda.empty_cache()
        except Exception:pass
def verify_handler():
    if not GPU_LOCK.acquire(blocking=False):return"Busy: a run is in progress."
    try:
        t=time.perf_counter();g=INSTR.guard();fz=INSTR.frozen();w=INSTR.full_weight_sha();cfg=INSTR.config()
        out=dict(model=MODEL_ID,engine_status=INSTR.status(),full_weight_sha256_now=w,full_weight_sha256_reference=WEIGHT_SHA_EXPECTED,full_weight_identical_to_reference=w==WEIGHT_SHA_EXPECTED,
                 all_parameter_guard_now=g,all_parameter_guard_startup=INSTR.info["guard0"],guard_unchanged=g==INSTR.info["guard0"],
                 sentinel_unchanged=fz["sentinel"]==INSTR.info["sentinel0"],trainable_tensors=fz["trainable_tensors"],foreign_hooks=fz["foreign_hooks"],residual_hooks=fz["residual_hooks"],
                 engine_file=ENGINE_FILE,engine_blob=cfg["engine_blob"],engine_blob_expected=ENGINE_BLOB_EXPECTED,engine_sha256=cfg["engine_sha256"],engine_sha256_expected=ENGINE_SHA256_EXPECTED,
                 engine_exec_sha256=cfg["engine_exec_sha256"],helper_sha256=cfg["helper_sha256"],source_blobs=cfg["source_blobs"],panel_sha=cfg["panel_sha"],fram_cfg=cfg["fram_cfg"],
                 cut=cfg["CUT"],architecture=cfg["arch"],checked_utc=utc_now(),seconds=round(time.perf_counter()-t,3))
        return json.dumps(jsafe(out),indent=2,ensure_ascii=False)
    finally:GPU_LOCK.release()
# ---------------- service self-test (runs before the public link is printed) ----------------
def service_selftest():
    global QUIET;out=[];d=ROOT/"_selftest";shutil.rmtree(d,ignore_errors=True);d.mkdir(parents=True);(d/"w.txt").write_text("ok");out.append("ROOT writable");QUIET=True
    try:
        for c_,cu in((12,""),(3,"Who governs this place?"),(5,"")):
            dd=d/f"ok{c_}";dd.mkdir()
            B=drain(execute_run(SelfTestInstrument("ok"),dict(run_id=f"SELFTEST-C{c_:02d}",run_dir=dd,case=c_,custom=cu,session=[dict(case=c_,correct=None)])))
            assert len(B["imgs"])==N_FIGS and len(B["pdfs"])==N_FIGS and file_ready(B["zip"])and file_ready(B["session"])and B["verdict"].endswith("SOURCE_FREE_QUERY")
        out.append(f"full pipeline on a synthetic instrument (correct and incorrect answer, target not selected, with and without free text): {N_FIGS}/{N_FIGS} figures (JPEG 300 dpi + PDF), ZIP, session file, {B['checks']} checks")
        for mode,what in(("guard_changes","a changed weight guard"),("prior_changed","a changed previously consolidated row"),("reread","an extra source read during consolidation"),
                         ("source_in_query","source tokens in the query input"),("config","a changed FRAM configuration"),("weight_ref","a full-weight SHA-256 different from the reference"),
                         ("residual_hook","a hook left on a decoder layer after the run"),("startup","start-up gates that do not reproduce TEST610"),("fact_in_file","fact text inside the session file"),
                         ("reload_diff","a reloaded memory that is not bitwise equal"),("reload_answer","a different answer after reload")):
            d2=d/mode;d2.mkdir()
            try:drain(execute_run(SelfTestInstrument(mode),dict(run_id="SELFTEST-"+mode.upper(),run_dir=d2,case=13,custom="",session=[])))
            except AuditFail:out.append(f"fail-closed: {what} aborts the run (nothing sealed)")
            else:raise RuntimeError(f"self-test: {what} did not abort the run")
        d3=d/"replay";d3.mkdir();R=replay_report(SelfTestInstrument("ok"),"SELFTEST-REPLAY",d3)
        assert R["out"]["panel_correct"]==23 and R["out"]["criteria_passed"]==15 and file_ready(R["json"])and file_ready(R["txt"])and file_ready(R["stdout"])
        out.append("acceptance replay report (synthetic): JSON + TXT + engine stdout, scores re-derived from rows")
    finally:QUIET=False
    if len(pack(READY_HTML))!=N_OUTPUTS or len(pack(READY_HTML,[],None,None,None))!=N_OUTPUTS:raise RuntimeError("callback output count mismatch")
    out.append(f"callback output count = {N_OUTPUTS}");out.append(f"file route {FILE_URL_PREFIX}");shutil.rmtree(d,ignore_errors=True);return out
say("="*140);say(DEMO_TITLE+" — "+MODEL_FAMILY+" · PART 3 / 3 · SELF-TEST AND INTERFACE");say("[1/3] SERVICE SELF-TEST")
for c_ in service_selftest():say(" PASS ·",c_)
#<<UI_END>>
# ---------------- interface ----------------
say("[2/3] INTERFACE")
CSS="""
:root{--kz-on:#0f766e;--kz-fg:#111827;--kz-card:#ffffff;--kz-bd:#d1d5db;--kz-a:#1d4ed8;--kz-err:#b91c1c;--kz-warn:#b45309;--kz-hero:#0b1220;--kz-gold:#b45309}
.dark{--kz-on:#2dd4bf;--kz-fg:#f3f4f6;--kz-card:#111827;--kz-bd:#4b5563;--kz-a:#60a5fa;--kz-err:#f87171;--kz-warn:#fbbf24;--kz-hero:#0b1220;--kz-gold:#fbbf24}
html,body{overflow-x:hidden!important}
.gradio-container{width:100%!important;max-width:920px!important;margin:0 auto!important;padding-left:10px!important;padding-right:10px!important;box-sizing:border-box!important}
.gradio-container *{box-sizing:border-box}
.kz{color:var(--kz-fg)!important;max-width:100%;overflow-wrap:anywhere;line-height:1.45}
.kz *{color:inherit}
.hero{background:#0b1220;color:#f9fafb!important;border-radius:6px;padding:18px 18px 14px;margin:4px 0 8px}
.hero *{color:#f9fafb!important}
.rec{font-size:11.5px;letter-spacing:1.6px;text-transform:uppercase;opacity:.85;border:1px solid rgba(255,255,255,.35);display:inline-block;padding:2px 8px;border-radius:3px}
.brand{font-size:clamp(24px,6.4vw,34px);font-weight:800;letter-spacing:.4px;line-height:1.1;margin-top:10px}
.title{font-size:clamp(15px,4vw,19px);font-weight:600;margin-top:4px}
.para{font-size:clamp(14px,3.8vw,17px);margin-top:10px;color:#fde68a!important;font-weight:600}
.q{font-size:13.5px;margin-top:8px;font-style:italic;opacity:.95}
.by{font-size:14px;margin-top:10px}
.by b{color:#fde68a!important}
.lic{font-size:12px;opacity:.85;margin-top:6px}
.lic a{text-decoration:underline}
.msg{margin:6px 0 0;padding-left:18px;font-size:13.5px}
.card{background:var(--kz-card);border:1px solid var(--kz-bd);border-left:4px solid var(--kz-bd);border-radius:4px;padding:10px 12px;margin:6px 0}
.card.on{border-left-color:var(--kz-on)}.card.err{border-left-color:var(--kz-err)}.card.warn{border-left-color:var(--kz-warn)}.card.info{border-left-color:var(--kz-a)}
.card .h{font-weight:700;font-size:15px;margin-bottom:4px}
.small{font-size:13.5px}.mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;opacity:.8;margin-right:4px}
#kz_run button,#kz_run{font-size:clamp(15px,4.4vw,18px)!important;font-weight:700!important;min-height:54px!important}
"""
DL_ALL_JS="""(u)=>{let items=[];try{items=JSON.parse(u||"[]")}catch(e){items=[]}
let i=0;const next=()=>{if(i>=items.length)return;const it=items[i++];const a=document.createElement('a');a.href=it.url;a.download=it.name;a.rel='noopener';
document.body.appendChild(a);a.click();setTimeout(()=>{a.remove();next()},900)};next();return u;}"""
HERO=(f'<div class="kz hero"><span class="rec">Public technical demonstration · priority record</span>'
      f'<div class="brand">AKBASCORE MAM-BÇ</div><div class="title">External associative memory for a frozen language model · {html.escape(MODEL_FAMILY)} · CUT3 + FRAM</div>'
      f'<div class="para">{html.escape(PARADIGM)}.</div><div class="q">{html.escape(SCI_QUESTION)}</div>'
      f'<div class="by"><b>{html.escape(DISCOVERY)}.</b><br>{html.escape(ORG)} · {html.escape(DATE_TXT)} · {html.escape(AUTHOR_PLACE)}</div>'
      '<ul class="msg">'+"".join(f"<li>{html.escape(m)}</li>"for m in CORE_MESSAGE)+"</ul>"
      f'<div class="lic">{html.escape(PRIORITY)}</div>'
      f'<div class="lic">{html.escape(LICENSE_NAME)} · {html.escape(COPYRIGHT)} · <a href="{LICENSE_URL}" target="_blank" rel="noopener">{html.escape(LICENSE_URL)}</a> · '
      f'<a href="{ENGINE_URL}" target="_blank" rel="noopener">610.py engine</a> · <a href="{ENGINE_LOG_URL}" target="_blank" rel="noopener">TEST610 log</a> · <a href="{BASE_URL}" target="_blank" rel="noopener">566.py</a> · '
      f'<a href="{LOG575_URL}" target="_blank" rel="noopener">TEST575 log</a> · DOI <a href="{PRIOR_PUB["url"]}" target="_blank" rel="noopener">{html.escape(PRIOR_PUB["doi"])}</a> · DOI <a href="{QWEN_PUB["url"]}" target="_blank" rel="noopener">{html.escape(QWEN_PUB["doi"])}</a></div></div>')
SCOPE_HTML=card_html("Scope of this demonstration",
    "<b>Demonstrated live in every run:</b><br>"+"<br>".join("• "+html.escape(t)for t in SCOPE["demonstrated"])+"<br><br><b>Archived (sealed, not run by a single run):</b><br>"+"<br>".join("• "+html.escape(t)for t in SCOPE["archived"])
    +"<br><br><b>Not established:</b><br>"+"<br>".join("• "+html.escape(t)for t in SCOPE["not_established"])+"<br><br><b>Technical notes:</b><br>"+"<br>".join("• "+html.escape(t)for t in SCOPE["notes"]),"info")
def _tb(**kw):
    for extra in(([{"buttons":["copy"]}]if _GR_MAJOR>=6 else[])+[{"show_copy_button":True},{}]):
        try:return gr.Textbox(**extra,**kw)
        except Exception:pass
def _gallery(**kw):
    try:return gr.Gallery(columns=1,height="auto",object_fit="contain",preview=False,**kw)
    except TypeError:return gr.Gallery(**kw)
if"demo"in globals():
    try:demo.close()
    except Exception:pass
_TITLE="AKBASCORE MAM-BÇ · Qwen2.5-7B · CUT3 + FRAM external memory · Mustafa Akbaş"
if _GR_MAJOR>=6:_blocks=gr.Blocks(title=_TITLE)
else:
    try:_blocks=gr.Blocks(css=CSS,title=_TITLE)
    except TypeError:_blocks=gr.Blocks(title=_TITLE)
_CCH=case_choices(INSTR);_DEF=_CCH[12]
with _blocks as demo:
    gr.HTML(HERO);gr.HTML(engine_card_html(INSTR))
    case_dd=gr.Dropdown(choices=_CCH,value=_DEF,label="Panel case (locked 24-case panel · CAL = calibration 0–11 · HOLD = holdout 12–23)",interactive=True)
    case_info=gr.HTML(case_info_html(_DEF))
    custom_tb=gr.Textbox(value="",label="Optional free-text question (exact FRAM selects five cartridges; answered from their CUT3 memory; not scored)",placeholder="e.g. Which city is the current capital of Zorvan?",lines=1,max_lines=2)
    run_btn=gr.Button("RUN · FRAM select · CUT3 memory · answer without the source · save → reload → answer · control · seal",variant="primary",elem_id="kz_run")
    status=gr.HTML(READY_HTML)
    gallery=_gallery(label=f"{N_FIGS} figures (tap to open · JPEG 300 dpi; vector PDFs in the ZIP)",show_label=True)
    dl_all=gr.Button(DL_ALL_LABEL,interactive=False)
    jpg_urls=gr.Textbox(value="",visible=False)
    with gr.Row():
        dlb=[gr.DownloadButton(label=l,value=None,interactive=False)for l in DL_LABELS]
    with gr.Accordion("Direct file links (fallback)",open=False):
        fls=[gr.File(label=n,interactive=False)for n in("ZIP · figures, PDFs, payload, logs, session file","FULL RUN LOG (.json)","READABLE RUN LOG (.txt)","RUN MANIFEST (.json)","CUT3 SESSION FILE (.pt)","SESSION LEDGER (.jsonl)")]
    with gr.Tabs():
        with gr.Tab("READABLE RUN LOG (.txt)"):raw_txt=_tb(lines=18,max_lines=40,label="readable run log")
        with gr.Tab("FULL RUN LOG (.json)"):raw_json=_tb(lines=18,max_lines=40,label="sealed canonical payload")
        with gr.Tab("MANIFEST (.json)"):raw_man=_tb(lines=12,max_lines=30,label="run manifest")
        with gr.Tab("SESSION LEDGER"):raw_led=_tb(lines=10,max_lines=30,label="one line per sealed run in this session")
    with gr.Accordion("Live replay of the TEST610 acceptance stage · 610.py [6/7]–[7/7] verbatim · about 2–4 minutes on an A100 (reported separately from the sealed TEST610 record)",open=False):
        rp_btn=gr.Button("Run the acceptance replay")
        rp_status=gr.HTML(card_html("Acceptance replay","Executes the remaining 610.py acceptance stage unchanged: 24 panel cases (with repeat), four bitwise rebuilds, four target-removed controls, the final CUT3 audit and the weight / hook checks; writes its own JSON/TXT report and the verbatim engine output. It does not replace the sealed TEST610 result.","info"))
        with gr.Row():rp_dl=[gr.DownloadButton(label=l,value=None,interactive=False)for l in REPLAY_LABELS]
    with gr.Accordion("Verification · full-weight SHA-256 · weight guard · engine, helper and panel hashes",open=False):
        ver_btn=gr.Button("Run verification now")
        ver_out=_tb(lines=14,max_lines=40,label="verification record")
    gr.HTML(SCOPE_HTML)
    gr.HTML(f'<div class="kz small" style="opacity:.8;margin:8px 0 18px">{html.escape(DISCOVERY)} · {html.escape(ORG)} · {html.escape(DATE_TXT)} · {html.escape(LICENSE_NAME)} · '
            f'<a href="{LICENSE_URL}" target="_blank" rel="noopener">{html.escape(LICENSE_URL)}</a></div>')
    OUTS=[status,gallery,dl_all,jpg_urls]+dlb+fls+[raw_txt,raw_json,raw_man,raw_led]
    if len(OUTS)!=N_OUTPUTS:raise RuntimeError(f"UI outputs {len(OUTS)} != {N_OUTPUTS}")
    case_dd.change(lambda a:case_info_html(a),inputs=[case_dd],outputs=case_info)
    run_btn.click(run_handler,inputs=[case_dd,custom_tb],outputs=OUTS)
    rp_btn.click(replay_handler,inputs=None,outputs=[rp_status]+rp_dl)
    ver_btn.click(verify_handler,inputs=None,outputs=ver_out)
    try:dl_all.click(None,inputs=[jpg_urls],outputs=None,js=DL_ALL_JS)
    except TypeError:dl_all.click(None,inputs=[jpg_urls],outputs=None,_js=DL_ALL_JS)
try:demo.queue(default_concurrency_limit=1)
except TypeError:demo.queue(concurrency_count=1)
say("[3/3] LAUNCH (public share link) — open the printed gradio.live link, choose a panel case, press RUN.")
_LAUNCH=dict(share=True,allowed_paths=ALLOWED_PATHS,show_error=True,inline=False)
if _GR_MAJOR>=6:_LAUNCH["css"]=CSS
try:demo.launch(**_LAUNCH)
except TypeError as _ex:
    if"css"not in str(_ex):raise
    _LAUNCH.pop("css",None);demo.launch(**_LAUNCH)
