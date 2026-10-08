# =====================================================================================================================================
# AKBASCORE MAM · PERSISTENT NUMERICAL MEMORY FOR FROZEN LANGUAGE MODELS — PART 3 / 3 · SELF-TEST AND GRADIO DEMONSTRATION
# Same demo as PARTS 1 and 2. Run this cell after PART 2 / 3, in the same Colab runtime. It runs a GPU-free self-test of the full
# pipeline (including fail-closed cases) and then prints a public gradio.live link. Choose a case and a target position, press RUN.
# Discovered and developed by Mustafa Akbaş · AkbasCore AI Teknoloji · 8 October 2026 · AKBASCORE RESEARCH SOFTWARE LICENSE
# =====================================================================================================================================
if not globals().get("MAM_DC6_PART2_OK"):raise RuntimeError("PART 2 / 3 has not been run in this runtime. Run PART 1, then PART 2, then PART 3.")
#<<UI_BEGIN>>
_SYN_A=["Aq","Bryn","Cav","Dex","Eln","Frax","Gyr","Hex","Iln","Jor","Kex","Lyr"];_SYN_B=["adar","bren","cyr","dax","elor","fyn","grel","hyn"]
class SelfTestInstrument:
    """Synthetic stand-in used ONLY by the service self-test (no GPU, no model). Never used for a real run."""
    def __init__(s,mode="ok"):
        s.kind="SELF-TEST";s.mode=mode;s.calls=0;names=[a+b for a in _SYN_A for b in _SYN_B];s.names=names
        s.cars=[];s.P=PREFIX_EXPECTED
        for w in range(24):
            subj,cur,near,form,role=names[w*4],names[w*4+1],names[w*4+2],names[w*4+3],names[(w*4+5)%len(names)]
            for typ,fact,gold in(("CURRENT",f"The current capital of {subj} is {cur}.",cur),("NEAR",f"The largest city of {subj} is {near}.",near),
                                 ("FORMER",f"The former capital of {subj} was {form}.",form),("ROLE",f"The current capital of {role} is {subj}.",subj)):
                ci=len(s.cars);s.cars.append(dict(id=ci,world=w,type=typ,fact=fact,gold=gold,body_len=12+ci%3))
        s.questions=[]
        for w in range(24):
            t=s._target(w);c=s.cars[t];subj=s.names[w*4]
            q={"CURRENT":f"What is the current capital of {subj}?","FORMER":f"What was the former capital of {subj}?","NEAR":f"What is the largest city of {subj}?","ROLE":f"What is the current capital of {s.names[(w*4+5)%len(s.names)]}?"}[c["type"]]
            s.questions.append(q)
        s.mem=None;s.T=0
        s.info=dict(model_id=MODEL_ID,arch=list(ARCH),dtype="bfloat16",attn="sdpa",gpu="SELF-TEST INSTRUMENT (no GPU)",gpu_total_gib=0.0,torch="n/a",transformers="n/a",gradio="n/a",python="n/a",
            platform="n/a",params=1,rope="n/a",engine_file=ENGINE_FILE,engine_sha256=ENGINE_SHA256_EXPECTED,init_seconds=0.0,startup_utc="n/a",sentinel0="a"*64,guard0="b"*64,hooks0=0,
            sentinel_method="synthetic",guard_method="synthetic",forward_counter="synthetic",observer="synthetic")
    def _target(s,w):return w*4+{"CURRENT":0,"NEAR":1,"FORMER":2,"ROLE":3}[["CURRENT","FORMER","NEAR","ROLE"][w%4]]
    def config(s):
        return dict(MODEL_ID=MODEL_ID,CUT=7 if s.mode=="config"else 6,MAX_NEW=16,SEED=552552,PANEL_SEED=550550,EXPECTED_PANEL_SHA=EXPECTED_PANEL_SHA,panel_sha=EXPECTED_PANEL_SHA,prefix_tokens=s.P,
                    n_cartridges=96,n_cases=24,positions=list(POSITIONS),system="synthetic",engine_sha256=ENGINE_SHA256_EXPECTED)
    def status(s):return{"title":"AKBASCORE MAM","reference_test":"TEST560","reference_score":"72/72","panel_sha":EXPECTED_PANEL_SHA}
    def cases(s):return[{"case":w,"type":s.cars[s._target(w)]["type"],"question":s.questions[w],"positions":list(POSITIONS)}for w in range(24)]
    def keys(s,case,position):
        t=s._target(case);d=[((case+1+3*j)%24)*4+(j%4)for j in range(4)]
        return [t]+d if position=="FIRST"else d[:2]+[t]+d[2:]if position=="MIDDLE"else d+[t]
    def car(s,ci):return dict(s.cars[ci])
    def target(s,case):return s._target(case)
    def source_ids(s,ci):return list(range(1,s.P+1))+[1000+ci*20+j for j in range(s.cars[ci]["body_len"])]
    def _qtoks(s,q):return re.findall(r"\n|[^\s\n]+",f"QUESTION:\n{q}\n\nANSWER: [/INST]")
    def query_ids(s,q):
        ids=[];s.__dict__.setdefault("_tokmap",{})
        for i,t in enumerate(s._qtoks(q)):tid=2000+(sum(map(ord,t))*31+i)%50000;s._tokmap[tid]=t;ids.append(tid)
        return ids
    def decode_each(s,ids):return[s._tokmap.get(t,"?")for t in ids]
    def panel_strings(s):return sorted(set(s.names))
    def sentinel(s):return"a"*64
    def guard(s):s.calls+=1;return"c"*64 if(s.mode=="guard_changes"and s.calls>1)else"b"*64
    def frozen(s):return dict(training=False,trainable_tensors=0,lora=False,optimizer=False,hooks=0,foreign_hooks=0,foreign_modules={},sentinel="a"*64)
    def load(s,case,position):
        keys=s.keys(case,position);steps=[];allf=[];old=0;ev_=[]
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
                st=dict(kind="append",cartridge_id=ci,old_len=old,new_len=new,prior_rows_bitwise_unchanged=same,prior_sha_before=hashlib.sha256(f"m{old}".encode()).hexdigest(),
                        prior_sha_after=hashlib.sha256(f"m{old}{'' if same else 'x'}".encode()).hexdigest())
            st.update(memory_sha=hashlib.sha256(f"m{new}".encode()).hexdigest(),seconds=0.3,forwards=f);steps.append(st);allf+=f;old=new
            ev_.append(dict(step=n_+1,cartridge_id=ci,position=n_+1,cache_tokens=new,source_replayed_in_consolidation=False,prior_tokens_recomputed=False))
        s.T=old;s.mem=(case,position)
        setup=dict(case=case,position=position,cartridges=[dict(id=ci,type=s.cars[ci]["type"],fact=s.cars[ci]["fact"],is_target=ci==s._target(case))for ci in keys],question=s.questions[case],
                   expected=s.cars[s._target(case)]["gold"],events=ev_,elapsed_seconds=1.5,weights_updated=False,source_replay_during_consolidation=False,prior_memory_recomputed=False)
        return setup,dict(steps=steps,forwards=allf),1.5
    def ask(s,q=None):
        case=s.mem[0];ref=s.questions[case];qq=ref if q is None else q;ids=s.query_ids(qq)
        if s.mode=="source_in_query":ids=s.source_ids(s._target(case))+ids
        exp=s.cars[s._target(case)]["gold"];ans=exp if case%7!=3 else s.names[(case*4+2)%len(s.names)]
        if q is not None and q!=ref:ans="Not provided in the information."
        rec=[dict(kind="ids",n=len(ids),ids=ids,embeds_zero=None,has_past=True,past_len=s.T)]+[dict(kind="ids",n=1,ids=[7],embeds_zero=None,has_past=True,past_len=s.T+len(ids)+i)for i in range(2)]
        isref=qq==ref
        return dict(question=qq,answer=ans,reference_question=isref,expected=exp if isref else None,correct=(hit_ans(ans,exp)if isref else None),elapsed_seconds=0.25,
                    source_text_supplied_to_query=False,model_weights_changed=False),rec,0.25
    def memory_len(s):return s.T
    def memory_sha_now(s):return hashlib.sha256(f"m{s.T}".encode()).hexdigest()
    def memory_profile(s):
        rng=np.random.default_rng(s.T);k=(rng.normal(20,3,(32,s.T))+np.linspace(0,10,32)[:,None]).tolist();v=(rng.normal(2,.4,(32,s.T))).tolist()
        return dict(layers=32,T=s.T,k_norm=k,v_norm=v,bytes=s.T*32*2*8*128*2,kv_heads=8,head_dim=128,dtype="bfloat16")
    def replay(s,progress=None):
        rows=[]
        for c in range(24):
            for p in POSITIONS:
                rows.append(dict(case=c,position=p,answer=s.cars[s._target(c)]["gold"],expected=s.cars[s._target(c)]["gold"],correct=True))
                if progress:progress(len(rows),72,rows[-1])
        return dict(score=72,total=72,matches_TEST560=True,panel_sha=EXPECTED_PANEL_SHA,rows=rows,elapsed_seconds=0.1)
    def sync(s):pass
    def reset_peak(s):pass
    def peak_gib(s):return 0.0
def drain(gen):
    while True:
        try:next(gen)
        except StopIteration as s_:return s_.value
RUN_COUNTER=globals().get("RUN_COUNTER",0);GPU_LOCK=globals().get("GPU_LOCK")or threading.Lock();SESSION_LEDGER=globals().get("SESSION_LEDGER",[])
LEDGER_PATH=ROOT/"session_ledger.jsonl"
SKIP=getattr(gr,"skip",None)or gr.update
_K=object()
FILE_KEYS=["zip","json","txt","man","ledger"]
DL_LABELS=["⬇ EVIDENCE PACKAGE (.zip)","⬇ FULL RUN LOG (.json)","⬇ READABLE RUN LOG (.txt)","⬇ RUN MANIFEST (.json)","⬇ SESSION LEDGER (.jsonl)"]
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
def case_choices(I):return[f"{c['case']:02d} · {c['type']} · {c['question']}"for c in I.cases()]
def parse_case(v):
    m=re.match(r"\s*(\d{1,2})",str(v or""));return int(m.group(1))if m else None
def ledger_text():return"\n".join(json.dumps(x,ensure_ascii=False)for x in SESSION_LEDGER)
def case_info_html(case_v,pos_v,I=None):
    I=I or INSTR;c=parse_case(case_v);pos=str(pos_v or"MIDDLE").upper()
    if c is None or pos not in POSITIONS:return card_html("Case","Select a case and a target position.","info")
    keys=I.keys(c,pos);t=I.target(c);q=I.cases()[c]["question"]
    rows="".join(f'<div class="small" style="{"font-weight:700" if ci==t else ""}"><span class="mono">C{n_+1} · #{ci:02d}</span> {html.escape(I.car(ci)["fact"])}'
                 +(' <span style="color:#1d4ed8">← target</span>'if ci==t else"")+"</div>"for n_,ci in enumerate(keys))
    body=(f'{rows}<div class="small" style="margin-top:6px"><span class="mono">question</span> {html.escape(q)}<br><span class="mono">expected</span> {html.escape(I.car(t)["gold"])} '
          f'<span style="opacity:.75">(audit metadata; not an input to writing, consolidation or answering)</span><br>'
          f'<span class="mono">order</span> C1 → C5 are written and appended in this order; the source text is not given to the model when the question is asked.</div>')
    return card_html(f"Case {c:02d} · target at {pos}",body,"info")
def engine_card_html(I):
    inf=I.info;cfg=I.config();A=TEST560["arms"]
    rows=[("model",f"{inf['model_id']} · frozen · {inf['dtype']}"),("GPU",inf["gpu"]),("cartridge",f"H{cfg['CUT']} residual + K/V of layers 0–{cfg['CUT']} · written once per fact"),
          ("consolidation",f"append-only · layers {cfg['CUT']+1}–31 computed only for new rows · prior rows never recomputed"),("panel",f"locked TEST560 panel · 24 cases × FIRST/MIDDLE/LAST · SHA {cfg['panel_sha'][:16]}…"),
          ("engine",f"{inf['engine_file']} · SHA-256 {inf['engine_sha256'][:16]}…"),("weights",f"sentinel {inf['sentinel0'][:16]}… · all-parameter guard {inf['guard0'][:16]}…"),
          ("init",f"{inf['init_seconds']:.1f} s" if isinstance(inf['init_seconds'],float)and math.isfinite(inf['init_seconds'])else"reused")]
    body="".join(f'<span class="mono">{html.escape(a)}</span> {html.escape(str(b))}<br>'for a,b in rows)
    body+=('<div class="small" style="margin-top:8px"><b>SEALED TEST560 REFERENCE</b> (archived · 72 = 24 cases × 3 target positions · not recomputed by single runs)<br>'
           +"<br>".join(f'<span class="mono">{a_}</span> {A[a_]["TOTAL"]}/72 · FIRST {A[a_]["FIRST"]} · MIDDLE {A[a_]["MIDDLE"]} · LAST {A[a_]["LAST"]} — {html.escape(ARM_DESC[a_])}'for a_ in("INCR_DC6","BATCH_DC6","JOINT","INDEP"))
           +f'<br><span class="mono">result SHA-256</span> {TEST560["result_sha"]}</div>')
    return card_html("Engine status · sealed reference",body,"on")
READY_HTML=card_html("Ready",f"Choose a case and the target position, optionally type your own question, and press <b>RUN</b>. One run writes five independent cartridges, appends them one by one, "
    f"asks the question without the source text, verifies the frozen weights, seals the payload and renders {N_FIGS} figures (JPEG 300 dpi + vector PDF). "
    "Each run is a LIVE demonstration observation; the TEST560 values above are the sealed reference.","info")
def is_cuda_error(ex):
    s=f"{type(ex).__name__} {ex}".lower();return"cuda"in s or"device-side assert"in s or"cublas"in s
def run_handler(case_v,pos_v,custom_v):
    global RUN_COUNTER
    c=parse_case(case_v);pos=str(pos_v or"").upper()
    if c is None or pos not in POSITIONS:
        yield pack(card_html("Invalid selection","Choose a case and a target position (FIRST / MIDDLE / LAST).","warn"));return
    if not GPU_LOCK.acquire(blocking=False):
        yield pack(card_html("Busy","Another run is in progress. Please try again in a moment.","warn"));return
    stage_name="initialisation"
    try:
        RUN_COUNTER+=1;run_id=f"MAM560DEMO-{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}-{RUN_COUNTER:04d}-C{c:02d}{pos[0]}"
        prune_runs(4);run_dir=ROOT/run_id;run_dir.mkdir(parents=True,exist_ok=True)
        preview=[dict(case=x["case"],position=x["position"],correct=x["correct"])for x in SESSION_LEDGER]
        gen=execute_run(INSTR,dict(run_id=run_id,run_dir=run_dir,case=c,position=pos,custom=(custom_v or"")[:300],session=preview+[dict(case=c,position=pos,correct=None)]));first=True
        while True:
            try:e=next(gen)
            except StopIteration as s:B=s.value;break
            stage_name=f"{e['stage']}/{N_STAGES} {e['title']}"
            yield pack(stage_card(e),[],None,None,None)if first else pack(stage_card(e));first=False
        SESSION_LEDGER.append(B["ledger"]);LEDGER_PATH.write_text(ledger_text()+"\n",encoding="utf-8")
        P=B["P"];L=P["live"];ok=L["correct"]
        raw={"txt":B["txt"].read_text(encoding="utf-8"),"json":B["payload"].read_bytes().decode("utf-8"),"man":B["manifest"].read_text(encoding="utf-8"),"ledger":ledger_text()}
        body=(f"<b>{html.escape(B['run_id'])}</b> · LIVE DEMONSTRATION RUN<br>"
              f'<span class="mono">question</span> {html.escape(L["question"])}<br>'
              f'<span class="mono">answer</span> <b>{html.escape(L["answer"]["answer"])}</b> · expected {html.escape(L["expected"])} · <b>{"CORRECT" if ok else "INCORRECT"}</b><br>'
              f'<span class="mono">memory</span> 5 cartridges appended · {L["memory"]["T"]} rows · source token-id forwards during load {L["source_reads"]} (each fact once) · '
              f'previously consolidated rows bitwise unchanged at all 4 appends: {all(s_["prior_rows_bitwise_unchanged"] for s_ in L["steps"][1:])}<br>'
              f'<span class="mono">query</span> {L["query_tokens"]} template tokens over the memory · source text given to the model: none · repeat identical: {L["repeat_identical"]}<br>'
              +(f'<span class="mono">free text</span> {html.escape(L["custom"]["question"])} → {html.escape(L["custom"]["answer"])} (unscored)<br>'if L.get("custom")else"")+
              f"weights unchanged (sentinel + all-parameter guard) · verdict <b>{html.escape(B['verdict'])}</b> · {N_FIGS} figures + PDFs + ZIP · {B['checks']}/{B['checks']} checks PASS<br>"
              f'<span class="mono">payload SHA-256 {B["sha"]}</span>')
        yield pack(card_html(f"{N_STAGES}/{N_STAGES} · Sealed",body,"on"),[(str(p),c_)for p,c_ in B["imgs"]],
                   {"zip":B["zip"],"json":B["payload"],"txt":B["txt"],"man":B["manifest"],"ledger":LEDGER_PATH},[p for p,_ in B["imgs"]],raw)
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
REPLAY_LABELS=("⬇ LIVE REPLAY (.json)","⬇ LIVE REPLAY (.txt)")
def replay_handler():
    if not GPU_LOCK.acquire(blocking=False):
        yield card_html("Busy","Another run is in progress. Please try again in a moment.","warn"),dl_update(None,REPLAY_LABELS[0]),dl_update(None,REPLAY_LABELS[1]);return
    try:
        rid=f"MAM560REPLAY-{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}";prune_runs(2,"MAM560REPLAY-");d=ROOT/rid;d.mkdir(parents=True,exist_ok=True)
        st={"done":0,"last":None,"res":None,"err":None}
        def prog(n,tot,row):st["done"]=n;st["last"]=row
        def work():
            try:st["res"]=replay_report(INSTR,rid,d,prog)
            except Exception as ex:st["err"]=ex;traceback.print_exc()
        th=threading.Thread(target=work,daemon=True);th.start()
        while th.is_alive():
            lr=st["last"];tail=(f"<br>last: case {lr['case']:02d} {lr['position']} → {html.escape(str(lr['answer']))} ({'PASS' if lr['correct'] else 'FAIL'})"if lr else"")
            yield card_html("Live replay running",f"Replaying the 72 locked panel runs with the unchanged engine (LIVE; separate from the sealed TEST560 record).{tail}"+pbar_html(st["done"],72),"info"),SKIP(),SKIP()
            time.sleep(1.5)
        if st["err"]is not None:raise st["err"]
        R=st["res"];o=R["out"];pp=o["per_position"]
        body=(f"<b>LIVE REPLAY {o['score']}/72</b> · FIRST {pp['FIRST']}/24 · MIDDLE {pp['MIDDLE']}/24 · LAST {pp['LAST']}/24 · {o['seconds']} s · weights unchanged: {o['weights_unchanged']}<br>"
              f"Sealed TEST560 incremental DC6 for comparison: {TEST560['arms']['INCR_DC6']['TOTAL']}/72. The replay is a separate live measurement and does not replace the sealed record.<br>"
              f'<span class="mono">replay SHA-256 {R["sha"]}</span>')
        yield card_html("Live replay complete",body,"on"),dl_update(R["json"],REPLAY_LABELS[0]),dl_update(R["txt"],REPLAY_LABELS[1])
    except Exception as ex:
        yield card_html("Live replay failed",f"{html.escape(type(ex).__name__)}: {html.escape(str(ex)[:500])}","err"),dl_update(None,REPLAY_LABELS[0]),dl_update(None,REPLAY_LABELS[1])
    finally:
        GPU_LOCK.release()
        try:torch.cuda.empty_cache()
        except Exception:pass
def verify_handler():
    if not GPU_LOCK.acquire(blocking=False):return"Busy: a run is in progress."
    try:
        t=time.perf_counter();g=INSTR.guard();fz=INSTR.frozen()
        out=dict(engine_status=INSTR.status(),all_parameter_guard_now=g,all_parameter_guard_startup=INSTR.info["guard0"],guard_unchanged=g==INSTR.info["guard0"],
                 sentinel_unchanged=fz["sentinel"]==INSTR.info["sentinel0"],trainable_tensors=fz["trainable_tensors"],foreign_hooks=fz["foreign_hooks"],
                 engine_sha256=INSTR.config()["engine_sha256"],engine_sha256_expected=ENGINE_SHA256_EXPECTED,panel_sha=INSTR.config()["panel_sha"],checked_utc=utc_now(),seconds=round(time.perf_counter()-t,3))
        return json.dumps(jsafe(out),indent=2,ensure_ascii=False)
    finally:GPU_LOCK.release()
# ---------------- service self-test (runs before the public link is printed) ----------------
def service_selftest():
    global QUIET;out=[];d=ROOT/"_selftest";shutil.rmtree(d,ignore_errors=True);d.mkdir(parents=True);(d/"w.txt").write_text("ok");out.append("ROOT writable");QUIET=True
    try:
        for c_,p_,cu in((0,"MIDDLE",""),(3,"FIRST","Who governs this place?")):
            dd=d/f"ok{c_}";dd.mkdir()
            B=drain(execute_run(SelfTestInstrument("ok"),dict(run_id=f"SELFTEST-C{c_:02d}",run_dir=dd,case=c_,position=p_,custom=cu,session=[dict(case=c_,position=p_,correct=True)])))
            assert len(B["imgs"])==N_FIGS and len(B["pdfs"])==N_FIGS and file_ready(B["zip"])and B["verdict"].endswith("SOURCE_FREE_QUERY")
        out.append(f"full pipeline on a synthetic instrument (correct and incorrect answer, with and without free text): {N_FIGS}/{N_FIGS} figures (JPEG 300 dpi + PDF), ZIP, {B['checks']} checks")
        for mode,what in(("guard_changes","a changed weight guard"),("prior_changed","a changed previously consolidated row"),("reread","an earlier source read again during consolidation"),
                         ("source_in_query","source tokens in the query input"),("config","a changed consolidation cut")):
            d2=d/mode;d2.mkdir()
            try:drain(execute_run(SelfTestInstrument(mode),dict(run_id="SELFTEST-"+mode.upper(),run_dir=d2,case=5,position="LAST",custom="",session=[])))
            except AuditFail:out.append(f"fail-closed: {what} aborts the run (nothing sealed)")
            else:raise RuntimeError(f"self-test: {what} did not abort the run")
        d3=d/"replay";d3.mkdir();R=replay_report(SelfTestInstrument("ok"),"SELFTEST-REPLAY",d3);assert R["out"]["score"]==72 and file_ready(R["json"])and file_ready(R["txt"])
        out.append("live replay report (synthetic): JSON + TXT, score re-derived from rows")
    finally:QUIET=False
    if len(pack(READY_HTML))!=N_OUTPUTS or len(pack(READY_HTML,[],None,None,None))!=N_OUTPUTS:raise RuntimeError("callback output count mismatch")
    out.append(f"callback output count = {N_OUTPUTS}");out.append(f"file route {FILE_URL_PREFIX}");shutil.rmtree(d,ignore_errors=True);return out
say("="*140);say(DEMO_TITLE+" — PART 3 / 3 · SELF-TEST AND INTERFACE");say("[1/3] SERVICE SELF-TEST")
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
      f'<div class="brand">AKBASCORE MAM</div><div class="title">Persistent numerical memory for frozen language models</div>'
      f'<div class="para">{html.escape(PARADIGM)}.</div>'
      f'<div class="by"><b>{html.escape(DISCOVERY)}.</b><br>{html.escape(ORG)} · {html.escape(DATE_TXT)} · {html.escape(AUTHOR_PLACE)}</div>'
      '<ul class="msg">'+"".join(f"<li>{html.escape(m)}</li>"for m in CORE_MESSAGE)+"</ul>"
      f'<div class="lic">{html.escape(PRIORITY)}</div>'
      f'<div class="lic">{html.escape(LICENSE_NAME)} · {html.escape(COPYRIGHT)} · <a href="{LICENSE_URL}" target="_blank" rel="noopener">{html.escape(LICENSE_URL)}</a> · '
      f'<a href="{SOURCE_URL}" target="_blank" rel="noopener">TEST560 source</a> · <a href="{LOG_URL}" target="_blank" rel="noopener">TEST560 log</a></div></div>')
SCOPE_HTML=card_html("Scope of this demonstration",
    "<b>Demonstrated live:</b><br>"+"<br>".join("• "+html.escape(t)for t in SCOPE["demonstrated"])+"<br><br><b>Not established:</b><br>"+"<br>".join("• "+html.escape(t)for t in SCOPE["not_established"]),"info")
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
_TITLE="AKBASCORE MAM · persistent numerical memory · Mustafa Akbaş"
if _GR_MAJOR>=6:_blocks=gr.Blocks(title=_TITLE)
else:
    try:_blocks=gr.Blocks(css=CSS,title=_TITLE)
    except TypeError:_blocks=gr.Blocks(title=_TITLE)
_CCH=case_choices(INSTR)
with _blocks as demo:
    gr.HTML(HERO);gr.HTML(engine_card_html(INSTR))
    with gr.Row():
        case_dd=gr.Dropdown(choices=_CCH,value=_CCH[0],label="Case (locked TEST560 panel · reference question)",interactive=True)
        pos_rd=gr.Radio(choices=list(POSITIONS),value="MIDDLE",label="Target cartridge position",interactive=True)
    case_info=gr.HTML(case_info_html(_CCH[0],"MIDDLE"))
    custom_tb=gr.Textbox(value="",label="Optional free-text question (answered from the same numerical memory; not scored)",placeholder="e.g. Which city is the current capital of Zorvan?",lines=1,max_lines=2)
    run_btn=gr.Button("RUN · write 5 cartridges · append-only consolidation · ask without the source · seal",variant="primary",elem_id="kz_run")
    status=gr.HTML(READY_HTML)
    gallery=_gallery(label=f"{N_FIGS} figures (tap to open · JPEG 300 dpi; vector PDFs in the ZIP)",show_label=True)
    dl_all=gr.Button(DL_ALL_LABEL,interactive=False)
    jpg_urls=gr.Textbox(value="",visible=False)
    with gr.Row():
        dlb=[gr.DownloadButton(label=l,value=None,interactive=False)for l in DL_LABELS]
    with gr.Accordion("Direct file links (fallback)",open=False):
        fls=[gr.File(label=n,interactive=False)for n in("ZIP · figures, PDFs, payload, logs","FULL RUN LOG (.json)","READABLE RUN LOG (.txt)","RUN MANIFEST (.json)","SESSION LEDGER (.jsonl)")]
    with gr.Tabs():
        with gr.Tab("READABLE RUN LOG (.txt)"):raw_txt=_tb(lines=18,max_lines=40,label="readable run log")
        with gr.Tab("FULL RUN LOG (.json)"):raw_json=_tb(lines=18,max_lines=40,label="sealed canonical payload")
        with gr.Tab("MANIFEST (.json)"):raw_man=_tb(lines=12,max_lines=30,label="run manifest")
        with gr.Tab("SESSION LEDGER"):raw_led=_tb(lines=10,max_lines=30,label="one line per sealed run in this session")
    with gr.Accordion("Live replay of the locked panel · 72 runs · about 3 minutes on an A100 (reported separately from the sealed TEST560 record)",open=False):
        rp_btn=gr.Button("Run the live replay (72 runs)")
        rp_status=gr.HTML(card_html("Live replay","Replays all 24 cases × FIRST/MIDDLE/LAST with the unchanged engine and writes its own JSON/TXT report. It does not replace the sealed TEST560 result.","info"))
        with gr.Row():rp_dl=[gr.DownloadButton(label=l,value=None,interactive=False)for l in REPLAY_LABELS]
    with gr.Accordion("Verification · weight guard · engine and panel hashes",open=False):
        ver_btn=gr.Button("Run verification now")
        ver_out=_tb(lines=14,max_lines=40,label="verification record")
    gr.HTML(SCOPE_HTML)
    gr.HTML(f'<div class="kz small" style="opacity:.8;margin:8px 0 18px">{html.escape(DISCOVERY)} · {html.escape(ORG)} · {html.escape(DATE_TXT)} · {html.escape(LICENSE_NAME)} · '
            f'<a href="{LICENSE_URL}" target="_blank" rel="noopener">{html.escape(LICENSE_URL)}</a></div>')
    OUTS=[status,gallery,dl_all,jpg_urls]+dlb+fls+[raw_txt,raw_json,raw_man,raw_led]
    if len(OUTS)!=N_OUTPUTS:raise RuntimeError(f"UI outputs {len(OUTS)} != {N_OUTPUTS}")
    case_dd.change(lambda a,b:case_info_html(a,b),inputs=[case_dd,pos_rd],outputs=case_info)
    pos_rd.change(lambda a,b:case_info_html(a,b),inputs=[case_dd,pos_rd],outputs=case_info)
    run_btn.click(run_handler,inputs=[case_dd,pos_rd,custom_tb],outputs=OUTS)
    rp_btn.click(replay_handler,inputs=None,outputs=[rp_status]+rp_dl)
    ver_btn.click(verify_handler,inputs=None,outputs=ver_out)
    try:dl_all.click(None,inputs=[jpg_urls],outputs=None,js=DL_ALL_JS)
    except TypeError:dl_all.click(None,inputs=[jpg_urls],outputs=None,_js=DL_ALL_JS)
try:demo.queue(default_concurrency_limit=1)
except TypeError:demo.queue(concurrency_count=1)
say("[3/3] LAUNCH (public share link) — open the printed gradio.live link, choose a case and a target position, press RUN.")
_LAUNCH=dict(share=True,allowed_paths=ALLOWED_PATHS,show_error=True,inline=False)
if _GR_MAJOR>=6:_LAUNCH["css"]=CSS
try:demo.launch(**_LAUNCH)
except TypeError as _ex:
    if"css"not in str(_ex):raise
    _LAUNCH.pop("css",None);demo.launch(**_LAUNCH)
