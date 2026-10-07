# =====================================================================================================================================
# AKBASCORE MAM · MISTRAL-7B ASSOCIATIVE RETRIEVAL INSTRUMENT — PART 3 / 3 · SELF-TEST AND GRADIO INSTRUMENT
# Same demo as PARTS 1 and 2. Run this cell after PART 2 / 3, in the same Colab runtime. It runs a GPU-free self-test of the full
# pipeline (including fail-closed cases) and then prints a public gradio.live link. Select an item and press RUN.
# =====================================================================================================================================
if not globals().get("MAM_MISTRAL_PART2_OK"):raise RuntimeError("PART 2 / 3 has not been run in this runtime. Run PART 1, then PART 2, then PART 3.")
#<<UI_BEGIN>>
_SYN_SYL1=["Aq","Bryn","Cav","Dex","Eln","Frax","Gyr","Hex","Iln","Jor","Kex","Lyr","Mav","Nex","Oryn","Pax","Qyr","Rex","Savn","Tov","Uln","Vex","Wyr","Xav","Yex","Zyr"]
_SYN_SYL2=["adar","bren","cyr","dax","elor","fyn","grel","hyn","ivar","jor","kyr","lor","myn","nex","or","pyr","qen","rix","sor","tyn"]
class SelfTestInstrument:
    """Synthetic stand-in used ONLY by the service self-test (no GPU, no model). Never used for a real run."""
    def __init__(s,mode="ok"):
        s.kind="SELF-TEST";s.mode=mode;s.calls=0;rng=random.Random(7)
        names=[a+b for a in _SYN_SYL1 for b in _SYN_SYL2];rng.shuffle(names);s.names=names[:N_CAND*3]
        codes=sorted({"".join(rng.choice("BCDFGHJKLMNPQRSTVWXZ")for _ in range(3))for _ in range(2000)});rng.shuffle(codes);s.seals=codes[:N_CAND];s.dis=codes[N_CAND:N_CAND+64]
        s.A=[];s.span=[]
        for i in range(N_CAND):
            st=[(s.names[3*i],s.seals[i]),(s.names[3*i+1],s.dis[i%64]),(s.names[3*i+2],s.dis[(i*7+3)%64])];random.Random(i).shuffle(st)
            s.A.append(st)
        s.info=dict(model_id=FROZEN_CONFIG["MODEL_ID"],arch=list(ARCH),rep=4,kvd=1024,dtype="bfloat16",attn="sdpa",gpu="SELF-TEST INSTRUMENT (no GPU)",gpu_total_gib=0.0,torch="n/a",transformers="n/a",
            gradio="n/a",python="n/a",platform="n/a",params=1,vocab=1,pad_id=0,engine_file=ENGINE_FILE,engine_sha256=ENGINE_SHA256_EXPECTED,init_seconds=0.0,startup_utc="n/a",
            sentinel0="a"*64,guard0="b"*64,hooks0=0,sentinel_method="synthetic",guard_method="synthetic",forward_counter="synthetic")
    def _tok(s,i):
        out=["[PAD]"]
        for e,sl in s.A[i-1]:out+=["Instrument"," "+e[:3],e[3:]," carries"," seal"," "+sl,"."]
        return out+["\n\n"]
    def _span(s,i):t=s._tok(i);return[j for j,x in enumerate(t)if x.strip()==s.seals[i-1]]
    def _w(s,i,target_seal=None):
        t=s._tok(i);sl=target_seal or s.seals[i-1];w=[0.0]+[0.02+0.01*((j*7)%5)for j in range(1,len(t))]
        for j,x in enumerate(t):
            if x.strip()==sl:w[j]=1.4
        z=sum(w);return[x/z for x in w]
    def _scores(s,i,target,miss=False):
        rng=random.Random(1000+i*31+target);sc=[0.30+0.12*rng.random()for _ in range(N_CAND)];sc[target-1]=0.88+0.05*rng.random()
        if miss:sc[(target%N_CAND)]=sc[target-1]+0.01
        if s.mode=="nan":sc[3]=float("nan")
        return sc
    def _rec(s,n):
        nq=s.question_tokens(s.items()[0]["question"]);return[dict(input_len=nq,has_past=True,past_len=s.a_slots(n))]
    def _top(s,sc,exp):
        order=sorted(range(N_CAND),key=lambda j:(-sc[j],j))
        return order,[{"rank":k_+1,"candidate":j+1,"entity":s.names[3*j],"seal":s.seals[j],"score":sc[j],"is_expected":j+1==exp}for k_,j in enumerate(order[:10])]
    def status(s):return{"initialized":True,"engine":"synthetic","model":FROZEN_CONFIG["MODEL_ID"],"pointer":"L28H00"if s.mode!="config"else"L27H00","address":"L00-V UNCENTERED","address_dim":1024,"candidate_memories":128,
                         "model_frozen":True,"trainable_tensors":0,"query_time_candidate_B_forwards":0,"test528_lock":TEST528["lock"],"test528_result_sha":TEST528["result_sha"]}
    def verify(s):return{"weight_sentinel_pass":True,"initial_weight_sentinel":"a"*64,"current_weight_sentinel":"a"*64,"trainable_tensors":0,"pointer":"L28H00","address":"L00-V UNCENTERED","address_dim":1024,
                         "candidate_count":128,"query_time_candidate_B_forwards":0,"test528_lock":TEST528["lock"],"test528_result_sha":TEST528["result_sha"]}
    def items(s):return[{"item":i+1,"entity":s.names[3*i],"question":f"What seal does instrument {s.names[3*i]} carry? Give only the exact seal.","expected_candidate":i+1,"expected_seal":s.seals[i]}for i in range(N_CAND)]
    def field(s):return[{"candidate":i+1,"entity":s.names[3*i],"target_seal":s.seals[i],"B_tokens":21+i%3,"address_dim":1024}for i in range(N_CAND)]
    def config(s):
        c=dict(FROZEN_CONFIG);c.update(n_bmats=128,bmat_dims=[1024],seed=528528,fmt="QUESTION:\n{q}\n\nANSWER:",engine_sha256=ENGINE_SHA256_EXPECTED,test528_lock=TEST528["lock"],test528_result_sha=TEST528["result_sha"])
        if s.mode=="config":c["PTR_L"]=27
        return c
    def guard(s):s.calls+=1;return"c"*64 if(s.mode=="guard_changes"and s.calls>1)else"b"*64
    def frozen(s):return dict(training=False,trainable_tensors=0,lora=False,optimizer=False,hooks=0,foreign_hooks=0,foreign_modules={},sentinel="a"*64)
    def question_tokens(s,q):return len(q.split())+6
    def a_slots(s,i):return len(s._tok(i))
    def retrieve(s,i):
        it=s.items()[i-1];miss=(i%41==0);sc=s._scores(i,i,miss);order,top=s._top(sc,i);w=s._w(i);sp=s._span(i);am=max(range(len(w)),key=lambda t:w[t])
        rec=[dict(input_len=s.question_tokens(it["question"]),has_past=True,past_len=s.a_slots(i))]
        if s.mode=="b_forward":rec+= [dict(input_len=22,has_past=False,past_len=None)]*128
        r={"mode":"PRIMARY","item":i,"entity":it["entity"],"question":it["question"],"expected_candidate":i,"expected_seal":s.seals[i-1],"selected_candidate":order[0]+1,"selected_entity":s.names[3*order[0]],
           "selected_seal":s.seals[order[0]],"correct":order[0]+1==i,"rank":order.index(i-1)+1,"top1_score":sc[order[0]],"top2_score":sc[order[1]],"top1_top2_margin":sc[order[0]]-sc[order[1]],
           "pointer_hit":am in sp,"pointer_mass":sum(w[t]for t in sp),"pointer_argmax_position":am,"pointer_weights":w,"candidate_scores":sc,"top10":top,"latency_seconds":0.05,
           "query_time_candidate_B_forwards":0,"model_frozen":True}
        return r,rec,0.05
    def pointer_data(s,i):
        w=s._w(i);sp=s._span(i);am=max(range(len(w)),key=lambda t:w[t]);it=s.items()[i-1]
        return({"item":i,"entity":it["entity"],"seal":s.seals[i-1],"pointer_layer":28,"pointer_head":0,"weights":w,"argmax_position":am,"gold_span_positions":sp,"pointer_hit":am in sp,"pointer_mass":sum(w[t]for t in sp)},
               [dict(input_len=s.question_tokens(it["question"]),has_past=True,past_len=s.a_slots(i))],0.05)
    def controls(s,i):
        it=s.items()[i-1];out={}
        for mode in("PRIMARY","UNIFORM","SHIFTED","NO_A"):
            if mode=="PRIMARY":sc=s._scores(i,i,i%41==0)
            elif mode=="NO_A":sc=[0.0]*N_CAND
            else:rng=random.Random(sum(map(ord,mode))+i);sc=[0.30+0.12*rng.random()for _ in range(N_CAND)]
            order=sorted(range(N_CAND),key=lambda j:(-sc[j],j))
            out[mode]={"rank":order.index(i-1)+1,"selected_candidate":order[0]+1,"selected_seal":s.seals[order[0]],"correct":order[0]+1==i,"top1_score":sc[order[0]],"top2_score":sc[order[1]],
                       "margin":sc[order[0]]-sc[order[1]],"candidate_scores":sc}
        return({"item":i,"entity":it["entity"],"expected_candidate":i,"expected_seal":s.seals[i-1],"controls":out,"query_time_candidate_B_forwards":0,"model_frozen":True},
               [dict(input_len=s.question_tokens(it["question"]),has_past=True,past_len=s.a_slots(i))],0.05)
    def counterfactual(s,i,j):
        it=s.items()[i-1];sc=s._scores(i,j);order,top=s._top(sc,j);w=s._w(i);T=len(w)
        r={"mode":"COUNTERFACTUAL","source_item":i,"entity":it["entity"],"question":it["question"],"original_candidate":i,"original_seal":s.seals[i-1],"counterfactual_candidate":j,"counterfactual_seal":s.seals[j-1],
           "selected_candidate":order[0]+1,"selected_entity":s.names[3*order[0]],"selected_seal":s.seals[order[0]],"followed_counterfactual":order[0]+1==j,"rank":order.index(j-1)+1,"top1_score":sc[order[0]],
           "top2_score":sc[order[1]],"top1_top2_margin":sc[order[0]]-sc[order[1]],"pointer_hit":True,"pointer_mass":max(w),"pointer_argmax_position":max(range(T),key=lambda t:w[t]),"pointer_weights":w,
           "candidate_scores":sc,"top10":top,"latency_seconds":0.1,"query_time_candidate_B_forwards":0,"model_frozen":True}
        return r,[dict(input_len=T,has_past=False,past_len=None),dict(input_len=s.question_tokens(it["question"]),has_past=True,past_len=T)],0.1
    def extras(s,i,w,sel):
        rng=np.random.default_rng(i);p=(rng.normal(0,.3,1024)).tolist();sc=s._scores(i,i,i%41==0);b=(np.asarray(p)+rng.normal(0,.2,1024)).tolist()
        return dict(p=p,best_row=b,best_slot=7,best_cos=sc[sel-1],b_rows=21,tokens=s._tok(i),shifted=list(np.roll(np.asarray(w),len(w)//2)),uniform=[0.0]+[1/(len(w)-1)]*(len(w)-1))
    def panel_strings(s):return sorted(set(s.names)|set(s.seals)|set(s.dis))
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
def item_choices(I):return[f"{it['item']:03d} · {it['entity']}"for it in I.items()]
CF_AUTO=f"auto · item + {CF_OFFSET} (mod 128)"
def cf_choices(I):return[CF_AUTO]+[f"{it['item']:03d} · {it['entity']}"for it in I.items()]
def parse_item(v):
    m=re.match(r"\s*(\d{1,3})",str(v or""));return int(m.group(1))if m else None
def ledger_text():return"\n".join(json.dumps(x,ensure_ascii=False)for x in SESSION_LEDGER)
def item_info_html(item_v,cf_v,I=None):
    I=I or INSTR;i=parse_item(item_v)
    if not i:return card_html("Item","Select an item.","info")
    it=I.items()[i-1];j=parse_item(cf_v)if cf_v and cf_v!=CF_AUTO else default_cf_target(i);jt=I.items()[j-1]
    warn=' <b style="color:#b91c1c">counterfactual target must differ from the item</b>'if j==i else""
    body=(f'<span class="mono">question</span> {html.escape(it["question"])}<br><span class="mono">expected</span> B#{i:03d} · seal {html.escape(it["expected_seal"])} '
          f'<span class="small" style="opacity:.75">(audit metadata; not an input to retrieval)</span><br><span class="mono">counterfactual</span> {html.escape(it["entity"])} → seal {html.escape(jt["expected_seal"])} · target B#{j:03d}{warn}')
    return card_html(f"Item {i:03d} · {it['entity']}",body,"info")
def engine_card_html(I):
    inf=I.info;st=I.status()
    rows=[("model",f"{inf['model_id']} · frozen · {inf['dtype']}"),("GPU",inf["gpu"]),("pointer · address",f"{st.get('pointer')} · {st.get('address')} · {st.get('address_dim')}-D"),
          ("candidates",f"{st.get('candidate_memories')} precomputed B memories · max cosine"),("engine",f"{inf['engine_file']} · SHA-256 {inf['engine_sha256'][:16]}…"),
          ("weights",f"sentinel {inf['sentinel0'][:16]}… · all-parameter guard {inf['guard0'][:16]}…"),("init",f"{inf['init_seconds']:.1f} s" if isinstance(inf['init_seconds'],float)and math.isfinite(inf['init_seconds'])else"reused")]
    body="".join(f'<span class="mono">{html.escape(a)}</span> {html.escape(str(b))}<br>'for a,b in rows)
    body+=(f'<div class="small" style="margin-top:6px"><b>Sealed TEST528 reference (archived, 128 items; not recomputed here):</b> primary {fr(TEST528["primary"])} = {TEST528["primary"][2]} · '
           f'counterfactual {fr(TEST528["counterfactual"])} = {TEST528["counterfactual"][2]} · shifted {fr(TEST528["shifted"])} = {TEST528["shifted"][2]} · no-A {fr(TEST528["no_a"])} = {TEST528["no_a"][2]} · '
           f'identity regime: {TEST528["identity_regime"]}.</div>')
    return card_html("Engine status",body,"on")
READY_HTML=card_html("Ready",f"Select an item and press <b>RUN</b>. One run calls retrieve() (twice), get_pointer_data(), run_controls() and counterfactual() through the engine API, counts every model forward, "
    f"re-derives all reported values from the raw score vectors, seals the payload and renders {N_FIGS} figures (JPEG 300 dpi + vector PDF). Each run is a live demonstration observation; "
    "the TEST528 values shown above are archived reference results.","info")
def is_cuda_error(ex):
    s=f"{type(ex).__name__} {ex}".lower();return"cuda"in s or"device-side assert"in s or"cublas"in s
def run_handler(item_v,cf_v):
    global RUN_COUNTER
    i=parse_item(item_v);j=parse_item(cf_v)if cf_v and cf_v!=CF_AUTO else(default_cf_target(i)if i else None)
    if not i or not j or i==j:
        yield pack(card_html("Invalid selection","Choose an item and a counterfactual target different from the item.","warn"));return
    if not GPU_LOCK.acquire(blocking=False):
        yield pack(card_html("Busy","Another run is in progress. Please try again in a moment.","warn"));return
    stage_name="initialisation"
    try:
        RUN_COUNTER+=1;run_id=f"MAM528DEMO-{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}-{RUN_COUNTER:04d}-I{i:03d}"
        prune_runs(4);run_dir=ROOT/run_id;run_dir.mkdir(parents=True,exist_ok=True);gen=execute_run(INSTR,dict(run_id=run_id,run_dir=run_dir,item=i,cf_target=j));first=True
        while True:
            try:e=next(gen)
            except StopIteration as s:B=s.value;break
            stage_name=f"{e['stage']}/{N_STAGES} {e['title']}"
            yield pack(stage_card(e),[],None,None,None)if first else pack(stage_card(e));first=False
        SESSION_LEDGER.append(B["ledger"]);LEDGER_PATH.write_text(ledger_text()+"\n",encoding="utf-8")
        P=B["P"];L=P["live"];r=L["primary"];cf=L["counterfactual"];c=L["controls"]["controls"];ok=r["correct"]
        raw={"txt":B["txt"].read_text(encoding="utf-8"),"json":B["payload"].read_bytes().decode("utf-8"),"man":B["manifest"].read_text(encoding="utf-8"),"ledger":ledger_text()}
        body=(f"<b>{html.escape(B['run_id'])}</b> · LIVE DEMO OBSERVATION<br>"
              f'<span class="mono">primary</span> item {i:03d} → selected B#{r["selected_candidate"]:03d} · {"correct" if ok else "incorrect"} · rank of expected {r["rank"]}/128 · '
              f'top-1 {r["top1_score"]:.6f} · margin {r["top1_top2_margin"]:+.6f} · pointer argmax slot {r["pointer_argmax_position"]} (diagnostic span mass {r["pointer_mass"]:.3f})<br>'
              f'<span class="mono">controls</span> '+" · ".join(f"{k_} rank {v_['rank']}"for k_,v_ in c.items())+"<br>"
              f'<span class="mono">counterfactual</span> target B#{j:03d} → selected B#{cf["selected_candidate"]:03d} · followed: {"yes" if cf["followed_counterfactual"] else "no"}<br>'
              f'<span class="mono">measured forwards</span> '+" · ".join(f"{k_} {len(v_)}"for k_,v_ in L["forwards"].items())+" · weights unchanged (sentinel + all-parameter guard)<br>"
              f"verdict <b>{html.escape(B['verdict'])}</b> · {N_FIGS} figures + PDFs + ZIP · {B['checks']}/{B['checks']} checks PASS<br>"
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
def verify_handler():
    if not GPU_LOCK.acquire(blocking=False):return"Busy: a run is in progress."
    try:
        t=time.perf_counter();g=INSTR.guard();v=INSTR.verify();fz=INSTR.frozen()
        out=dict(verify_engine=v,engine_status=INSTR.status(),all_parameter_guard_now=g,all_parameter_guard_startup=INSTR.info["guard0"],guard_unchanged=g==INSTR.info["guard0"],
                 sentinel_unchanged=fz["sentinel"]==INSTR.info["sentinel0"],trainable_tensors=fz["trainable_tensors"],foreign_hooks=fz["foreign_hooks"],
                 engine_sha256=INSTR.config()["engine_sha256"],engine_sha256_expected=ENGINE_SHA256_EXPECTED,checked_utc=utc_now(),seconds=round(time.perf_counter()-t,3))
        return json.dumps(jsafe(out),indent=2,ensure_ascii=False)
    finally:GPU_LOCK.release()
# ---------------- service self-test (runs before the public link is printed) ----------------
def service_selftest():
    global QUIET;out=[];d=ROOT/"_selftest";shutil.rmtree(d,ignore_errors=True);d.mkdir(parents=True);(d/"w.txt").write_text("ok");out.append("ROOT writable");QUIET=True
    try:
        for it_ in(1,41):
            dd=d/f"ok{it_}";dd.mkdir()
            B=drain(execute_run(SelfTestInstrument("ok"),dict(run_id=f"SELFTEST-I{it_:03d}",run_dir=dd,item=it_,cf_target=None)))
            assert len(B["imgs"])==N_FIGS and len(B["pdfs"])==N_FIGS and file_ready(B["zip"])and B["verdict"].endswith("INTEGRITY_VERIFIED")
        out.append(f"full pipeline on a synthetic instrument (correct and incorrect primary): {N_FIGS}/{N_FIGS} figures (JPEG 300 dpi + PDF), ZIP, {B['checks']} checks")
        for mode,what in(("guard_changes","a changed weight guard"),("b_forward","extra model forwards during retrieve() (candidate-B forwarding)"),("config","a changed pointer coordinate"),("nan","a non-finite candidate score")):
            d2=d/mode;d2.mkdir()
            try:drain(execute_run(SelfTestInstrument(mode),dict(run_id="SELFTEST-"+mode.upper(),run_dir=d2,item=5,cf_target=None)))
            except AuditFail:out.append(f"fail-closed: {what} aborts the run (nothing sealed)")
            else:raise RuntimeError(f"self-test: {what} did not abort the run")
    finally:QUIET=False
    if len(pack(READY_HTML))!=N_OUTPUTS or len(pack(READY_HTML,[],None,None,None))!=N_OUTPUTS:raise RuntimeError("callback output count mismatch")
    out.append(f"callback output count = {N_OUTPUTS}");out.append(f"file route {FILE_URL_PREFIX}");shutil.rmtree(d,ignore_errors=True);return out
say("="*140);say(DEMO_TITLE+" — PART 3 / 3 · SELF-TEST AND INTERFACE");say("[1/3] SERVICE SELF-TEST")
for c_ in service_selftest():say(" PASS ·",c_)
# ---------------- interface ----------------
say("[2/3] INTERFACE")
CSS="""
:root{--kz-on:#0f766e;--kz-fg:#111827;--kz-card:#ffffff;--kz-bd:#d1d5db;--kz-a:#1d4ed8;--kz-err:#b91c1c;--kz-warn:#b45309}
.dark{--kz-on:#2dd4bf;--kz-fg:#f3f4f6;--kz-card:#111827;--kz-bd:#4b5563;--kz-a:#60a5fa;--kz-err:#f87171;--kz-warn:#fbbf24}
html,body{overflow-x:hidden!important}
.gradio-container{width:100%!important;max-width:900px!important;margin:0 auto!important;padding-left:10px!important;padding-right:10px!important;box-sizing:border-box!important}
.gradio-container *{box-sizing:border-box}
.kz{color:var(--kz-fg)!important;max-width:100%;overflow-wrap:anywhere;line-height:1.45}
.kz *{color:inherit}
.hero{padding:8px 2px 2px;border-bottom:1px solid var(--kz-bd);margin-bottom:6px}
.brand{font-size:clamp(20px,5.6vw,28px);font-weight:700;letter-spacing:.3px;line-height:1.15}
.sub{font-size:clamp(13px,3.6vw,15px);margin-top:4px}
.by{font-size:12px;opacity:.75;margin-top:4px}
.card{background:var(--kz-card);border:1px solid var(--kz-bd);border-left:4px solid var(--kz-bd);border-radius:4px;padding:10px 12px;margin:6px 0}
.card.on{border-left-color:var(--kz-on)}.card.err{border-left-color:var(--kz-err)}.card.warn{border-left-color:var(--kz-warn)}.card.info{border-left-color:var(--kz-a)}
.card .h{font-weight:700;font-size:15px;margin-bottom:4px}
.small{font-size:13.5px}.mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;opacity:.8;margin-right:4px}
#kz_run button,#kz_run{font-size:clamp(15px,4.4vw,18px)!important;font-weight:700!important;min-height:54px!important}
"""
DL_ALL_JS="""(u)=>{let items=[];try{items=JSON.parse(u||"[]")}catch(e){items=[]}
let i=0;const next=()=>{if(i>=items.length)return;const it=items[i++];const a=document.createElement('a');a.href=it.url;a.download=it.name;a.rel='noopener';
document.body.appendChild(a);a.click();setTimeout(()=>{a.remove();next()},900)};next();return u;}"""
HERO=(f'<div class="kz hero"><div class="brand">AKBASCORE MAM · Mistral-7B-Instruct-v0.3</div>'
      '<div class="sub">Executable instrument for the frozen TEST528 associative-retrieval mechanism: ÇAĞRIİZ pointer L28H00 → L00-V uncentered 1024-D address → max cosine over 128 precomputed B memories '
      '(0 query-time candidate-B forwards). Live runs are demonstration observations; TEST528 values are archived reference results.</div>'
      f'<div class="by">{html.escape(AUTHOR)} · {html.escape(AUTHOR_PLACE)} · {html.escape(COPYRIGHT)}</div></div>')
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
if _GR_MAJOR>=6:_blocks=gr.Blocks(title="AKBASCORE MAM · Mistral instrument")
else:
    try:_blocks=gr.Blocks(css=CSS,title="AKBASCORE MAM · Mistral instrument")
    except TypeError:_blocks=gr.Blocks(title="AKBASCORE MAM · Mistral instrument")
_ICH=item_choices(INSTR);_CCH=cf_choices(INSTR)
with _blocks as demo:
    gr.HTML(HERO);gr.HTML(engine_card_html(INSTR))
    with gr.Row():
        item_dd=gr.Dropdown(choices=_ICH,value=_ICH[0],label="Item (active A memory · question)",interactive=True)
        cf_dd=gr.Dropdown(choices=_CCH,value=CF_AUTO,label="Counterfactual target B memory",interactive=True)
    item_info=gr.HTML(item_info_html(_ICH[0],CF_AUTO))
    run_btn=gr.Button("RUN · retrieve · controls · counterfactual · seal",variant="primary",elem_id="kz_run")
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
    with gr.Accordion("Engine verification · verify_engine() + weight guard",open=False):
        ver_btn=gr.Button("Run verification now")
        ver_out=_tb(lines=14,max_lines=40,label="verification record")
    OUTS=[status,gallery,dl_all,jpg_urls]+dlb+fls+[raw_txt,raw_json,raw_man,raw_led]
    if len(OUTS)!=N_OUTPUTS:raise RuntimeError(f"UI outputs {len(OUTS)} != {N_OUTPUTS}")
    item_dd.change(lambda a,b:item_info_html(a,b),inputs=[item_dd,cf_dd],outputs=item_info)
    cf_dd.change(lambda a,b:item_info_html(a,b),inputs=[item_dd,cf_dd],outputs=item_info)
    run_btn.click(run_handler,inputs=[item_dd,cf_dd],outputs=OUTS)
    ver_btn.click(verify_handler,inputs=None,outputs=ver_out)
    try:dl_all.click(None,inputs=[jpg_urls],outputs=None,js=DL_ALL_JS)
    except TypeError:dl_all.click(None,inputs=[jpg_urls],outputs=None,_js=DL_ALL_JS)
try:demo.queue(default_concurrency_limit=1)
except TypeError:demo.queue(concurrency_count=1)
say("[3/3] LAUNCH (public share link) — open the printed gradio.live link, choose an item and press RUN.")
_LAUNCH=dict(share=True,allowed_paths=ALLOWED_PATHS,show_error=True,inline=False)
if _GR_MAJOR>=6:_LAUNCH["css"]=CSS
try:demo.launch(**_LAUNCH)
except TypeError as _ex:
    if"css"not in str(_ex):raise
    _LAUNCH.pop("css",None);demo.launch(**_LAUNCH)
