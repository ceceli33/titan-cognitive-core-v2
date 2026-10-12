# =====================================================================================================================================
# AKBASCORE MAM — THE SILICON HIPPOCAMPUS · QWEN2.5-7B · PART 3 / 3 · SELF-TEST AND GRADIO DEMONSTRATION
# Same demo as PARTS 1 and 2. Run this cell after PART 2 / 3, in the same Colab runtime. It runs a GPU-free self-test of the full
# pipeline (including fail-closed cases) and then prints a public gradio.live link. Press RUN — there is nothing to type or choose.
# Discovered and developed by Mustafa Akbaş · AkbasCore AI Teknoloji · 12 October 2026 · AKBASCORE RESEARCH SOFTWARE LICENSE
# =====================================================================================================================================
if not globals().get("MAM_SH_PART2_OK"):raise RuntimeError("PART 2 / 3 has not been run in this runtime. Run PART 1, then PART 2, then PART 3.")
#<<UI_BEGIN>>
_SYN_L="ABCDEFGHJKLMNPQRSTUVWXYZ"
class _SynMem:
    """Synthetic memory handle for the self-test (no tensors)."""
    def __init__(s,key,T,salt=""):s.key=key;s.T=T;s.salt=salt
class SelfTestInstrument:
    """Synthetic stand-in used ONLY by the service self-test (no GPU, no model). Never used for a real run."""
    def __init__(s,mode="ok",tmp=None):
        s.kind="SELF-TEST";s.mode=mode;s.calls=0;s.P=PREFIX_EXPECTED;rng=random.Random(627)
        def w(n):return"".join(rng.choice(_SYN_L)for _ in range(n)).capitalize()
        s.cars=[]
        for i in range(N_CART):
            ctry=f"ZX{624000+i:06d}"if i>=N_ORIG else w(6);city=w(10)
            s.cars.append(dict(id=i,fact=f"The current capital of {ctry} is {city}.",gold=city,country=ctry,body_len=17+i%3))
        s.case_t=[N_ORIG+7*j+3 for j in range(N_CASES)]
        s.qs=[(f"Name the present capital of {s.cars[t]['country']}."if j%2==0 else f"Which city currently serves as the capital of {s.cars[t]['country']}?")for j,t in enumerate(s.case_t)]
        s.sizes=[57344*(s.P+c["body_len"])+16437+(i%5)for i,c in enumerate(s.cars)]
        s.tmp=Path(tmp or ROOT/"_selftest_engine");s.tmp.mkdir(parents=True,exist_ok=True);s.res=s.tmp/"SYN_ENGINE_RESULT.json"
        s.res.write_text(json.dumps({"test":"627","status":"COMPLETE","note":"synthetic"}),encoding="utf-8")
        s.info=dict(model_id=MODEL_ID,arch=list(ARCH),dtype="bfloat16",attn="sdpa",gpu="SELF-TEST INSTRUMENT (no GPU)",gpu_total_gib=0.0,torch="n/a",transformers="n/a",gradio="n/a",python="n/a",
            platform="n/a",params=1,engine_file=ENGINE_FILE,engine_sha256=ENGINE_SHA256_EXPECTED,engine_blob=ENGINE_BLOB_EXPECTED,init_seconds=0.0,startup_utc="n/a",
            chain_sha256={"626.py":CHAIN["626.py"]["sha256"],"624.py":CHAIN["624.py"]["sha256"],"610.py":SHA610_EXPECTED},source_blobs=dict(SOURCE_BLOBS_EXPECTED),
            weight_sha0=WEIGHT_SHA_EXPECTED,sentinel0="a"*64,sentinel_engine="a"*64,guard0="b"*64,hooks0=0,sentinel_method="synthetic",guard_method="synthetic",full_weight_method="synthetic",forward_counter="synthetic")
    def config(s):
        return dict(MODEL_ID=MODEL_ID,arch=list(ARCH),CUT=CUT_EXPECTED,prefix_tokens=s.P,n_cartridges=N_CART,n_original=N_ORIG,manifest=N_CART,rank=RANK_EXPECTED,nlist=NLIST_EXPECTED,ntok=26738,dim=3584,
                    index_ntotal=26738,candidates=list(LOCKED_CAND),reranker=list(LOCKED_RERANK),reranker_family=["FRAM","PEAK25","PEAK50"],beta=8.0,max_new=16,pool="synthetic",
                    engine_sha256=ENGINE_SHA256_EXPECTED,engine_blob=ENGINE_BLOB_EXPECTED,chain_sha256=dict(s.info["chain_sha256"]),lock_610=SHA610_EXPECTED,source_blobs=dict(SOURCE_BLOBS_EXPECTED),
                    source_blobs_measured=dict(SOURCE_BLOBS_EXPECTED),baseline_610=dict(H1=12,H5=12),n_questions=N_CASES)
    def startup(s):
        e=TEST627_LOG["e2e"];st=dict(TEST627_LOG["storage"]);st["disk_pool_bytes"]=sum(s.sizes);st.update(median_file_bytes=float(np.median(s.sizes)),file_container_overhead_bytes=1,tensor_dtype_bytes={"torch.bfloat16":1},
                lossless_gzip_sample_files=32,gzip_sample_estimated_pool_bytes=1,gzip_sample_estimate_not_measured_full_pool=True,gzip_compress_p50_ms=240.0,gzip_decompress_p50_ms=19.0,disk_pool_GB_decimal=sum(s.sizes)/1e9)
        integ=dict(TEST627_LOG["integrity"])
        return dict(measured="synthetic self-test",measured_utc="synthetic",gpu="none",status="INTEGRITY_FAILURE"if s.mode=="engine_status"else"COMPLETE",summary=json.loads(json.dumps(e)),integrity=integ,storage=st,locked_reranker="PEAK50",locked_candidates=dict(nprobe=2,token_k=16,budget=16),
                    limitations=["synthetic limitation line"],elapsed_seconds=1.0,result_file=s.res.name,result_sha256=hashlib.sha256(s.res.read_bytes()).hexdigest(),
                    write=dict(files=N_CART,ms_p50=131.0,ms_p95=150.0,seconds_total=190.0),sizes=list(s.sizes),
                    index=dict(native_positions=26738,ivf_lists=128,pca_dims=128,index_bytes=13970459,index_build_ms=26.0,pca_basis_bytes=1835008,pca_center_bytes=14336,owner_bytes=106952,
                               native_activation_bytes=383315968,list_min=40,list_max=900,pca_train_vectors=2048),baseline_610=dict(H1=12,H5=12))
    def engine_result_path(s):return s.res
    def cases(s):return[dict(case=j,question=s.qs[j],target=t)for j,t in enumerate(s.case_t)]
    def _sel(s,j):return s.case_t[j]+1 if(s.mode=="wrong"or j%11==5)else s.case_t[j]
    def _cands(s,j):
        t=s.case_t[j];sel=s._sel(j);ids=[sel]+[x for x in([t]+[t+k for k in range(2,17)])if x!=sel][:15]
        rng=np.random.default_rng(j);cand=[]
        for r,ci in enumerate(ids):
            fr=0.62-0.01*r+float(rng.normal(0,.001));pk=0.80-0.012*r;cand.append(dict(id=ci,score=0.5*fr+0.5*pk,fram=fr,peak=pk))
        cand.sort(key=lambda c:(-c["score"],c["id"]));return cand
    def sealed_row(s,case):
        rk=[c["id"]for c in s._cands(case)];sel=rk[0];top5=rk[:5]
        if s.mode=="read_mismatch":top5=list(reversed(top5))
        return dict(selected=sel,top5=top5,selected_correct=sel==s.case_t[case],answer=s.cars[sel]["gold"],correct=sel==s.case_t[case])
    def car(s,ci):c=s.cars[ci];return dict(id=ci,fact=c["fact"],gold=c["gold"],body_len=c["body_len"],country=c["country"],file=f"cut3_{ci:05d}.pt",bytes=s.sizes[ci],sha256=hashlib.sha256(f"f{ci}".encode()).hexdigest())
    def source_ids(s,ci):return list(range(1,s.P+1))+[1000+ci*20+j for j in range(s.cars[ci]["body_len"])]
    def _ids(s,text,base):
        s.__dict__.setdefault("_tokmap",{});out=[]
        for i,t in enumerate(re.findall(r"\n|[^\s\n]+",text)):tid=base+(sum(map(ord,t))*31+i)%50000;s._tokmap[tid]=t;out.append(tid)
        return out
    def query_ids(s,q):return s._ids(f"\nQUESTION:\n{q}\n\nANSWER:\n<|im_end|>\n<|im_start|>assistant\n",2000)
    def frame_ids(s,q):return list(range(1,s.P+1))+s._ids(q+"\n\n",60000)
    def decode_each(s,ids):return[s._tokmap.get(t,"?")for t in ids]
    def panel_strings(s):return sorted({c["gold"]for c in s.cars}|{c["country"]for c in s.cars})
    def hit(s,a,g):return hit_ans(a,g)
    def build_ms(s,ci):return 130.0+ci%7
    def pool_stat(s):
        bad=[7]if s.mode=="pool_size"else[];return dict(files_in_pool=N_CART,manifest=N_CART,size_mismatch=bad,total_bytes=sum(s.sizes))
    def pool_sha_audit(s):return dict(files=N_CART,mismatch=[],seconds=0.01)
    def sentinel(s):return"a"*64
    def guard(s):s.calls+=1;return"c"*64 if(s.mode=="guard_changes"and s.calls>1)else"b"*64
    def full_weight_sha(s):return WEIGHT_SHA_EXPECTED
    def frozen(s):
        s.__dict__["fz"]=s.__dict__.get("fz",0)+1
        return dict(training=False,trainable_tensors=0,lora=False,optimizer=False,hooks=0,foreign_hooks=0,foreign_modules={},sentinel="a"*64,residual_hooks=(["L3:layer"]if(s.mode=="residual_hook"and s.fz>1)else[]))
    def write(s,ci):
        src=s.source_ids(ci);f=[dict(kind="ids",n=len(src),ids=src,embeds_zero=None,has_past=False,past_len=None),dict(kind="embeds",n=len(src),ids=None,embeds_zero=True,has_past=False,past_len=None)]
        if s.mode=="write_extra":f.append(dict(kind="ids",n=3,ids=[1,2,3],embeds_zero=None,has_past=False,past_len=None))
        return _SynMem(ci,len(src),"diff"if s.mode=="write_diff"else""),f,0.13
    def read_file(s,ci,device=False):
        if s.mode=="sha_mismatch"and device:return None,dict(file=f"cut3_{ci:05d}.pt",sha256="0"*64,sha_ok=False,seconds=0.01)
        return _SynMem(ci,s.P+s.cars[ci]["body_len"]),dict(file=f"cut3_{ci:05d}.pt",sha256=s.car(ci)["sha256"],sha_ok=True,bytes=s.sizes[ci],seconds=0.018)
    def file_bytes(s,ci):
        b=bytes(range(256))*64
        return b+s.cars[ci]["fact"].encode()if s.mode=="fact_in_file"else b
    def address(s,q):
        j=s.qs.index(q);t=s.case_t[j];cand=[dict(c)for c in s._cands(j)];c626=sorted(c["id"]for c in cand)
        if s.mode=="peak_rederive":cand[3]["score"]+=0.01
        if s.mode=="cand_merge":c626=c626[:-1]+[N_CART-1]
        fid=s.frame_ids(q)
        if s.mode=="frame_has_fact":fid=fid+s.source_ids(t)[s.P:]
        return dict(ranking=[c["id"]for c in cand],candidate626=c626,candidates=cand,touched=117,faiss_ms=0.8,top2_ms=18.0,rerank_ms=2.5,address_ms=24.4,encode_ms=31.0,query_rows=11,
                    forwards=[dict(kind="ids",n=len(fid),ids=fid,embeds_zero=None,has_past=False,past_len=None)])
    def ask(s,mem,q):
        ids=s.query_ids(q)
        if s.mode=="source_in_query":ids=s.source_ids(mem.key)+ids
        j=s.qs.index(q)if q in s.qs else None
        ans=s.cars[mem.key]["gold"]if j is not None else"Not provided."
        rec=[dict(kind="ids",n=len(ids),ids=ids,embeds_zero=None,has_past=True,past_len=mem.T)]+[dict(kind="ids",n=1,ids=[7],embeds_zero=None,has_past=True,past_len=mem.T+len(ids)+i)for i in range(3)]
        return ans,rec,0.24
    def compare(s,a,b):
        eq=a.key==b.key and a.T==b.T and a.salt==b.salt
        rows=[dict(layer=li,tensor=nm,shape=[1,ARCH[3],a.T,ARCH[4]],shape_equal=True,dtype_equal=True,bitwise_equal=eq or li<4,max_abs_diff=0.0 if(eq or li<4)else 0.0078125,
                   differing_elements=0 if(eq or li<4)else 11,elements=ARCH[3]*a.T*ARCH[4])for li in range(ARCH[0])for nm in("K","V")]
        return dict(layers_a=ARCH[0],layers_b=ARCH[0],tensors_compared=len(rows),all_equal=all(r["bitwise_equal"]for r in rows),layers_differing=sorted({r["layer"]for r in rows if not r["bitwise_equal"]}),
                    max_abs_diff=max(r["max_abs_diff"]for r in rows),rows=rows)
    def equal(s,a,b):return s.compare(a,b)["all_equal"]
    def memory_len(s,mem):return mem.T
    def memory_sha(s,mem):return hashlib.sha256(f"m{mem.key}{mem.T}{mem.salt}".encode()).hexdigest()
    def memory_bytes(s,mem):return mem.T*ARCH[0]*2*ARCH[3]*ARCH[4]*2
    def memory_shape(s,mem):return dict(layers=ARCH[0],tensors=2*ARCH[0],shape=[1,ARCH[3],mem.T,ARCH[4]],dtype="bfloat16",T=mem.T,uniform=True)
    def release(s,*m):pass
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
FILE_KEYS=["zip","json","txt","man","engine","tr","ledger"]
DL_LABELS=["⬇ EVIDENCE PACKAGE (.zip)","⬇ FULL RUN LOG (.json)","⬇ READABLE RUN LOG (.txt)","⬇ RUN MANIFEST (.json)","⬇ ENGINE START-UP RESULT (.json)","⬇ Fig. 1 · TÜRKÇE (.jpg)","⬇ SESSION LEDGER (.jsonl)"]
DL_ALL_LABEL=f"⬇ DOWNLOAD ALL {N_FIGS} FIGURES (JPEG, 300 dpi)"
RAW_KEYS=["txt","json","man","ledger"]
N_OUTPUTS=2+2+len(FILE_KEYS)*2+len(RAW_KEYS)+1
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
def pack(status=_K,gal=_K,files=_K,jpgs=_K,raw=_K,nxt=_K):
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
    out.append(SKIP()if nxt is _K else nxt)
    return tuple(out)
def card_html(title,body,kind="info"):return f'<div class="kz card {kind}"><div class="h">{html.escape(title)}</div><div class="small">{body}</div></div>'
def pbar_html(done,total):
    pc_=100.0*done/max(1,total);return f'<div style="margin-top:6px;background:#e5e7eb;height:10px"><div style="width:{pc_:.1f}%;height:10px;background:#1d4ed8"></div></div><div class="small">{done}/{total}</div>'
def stage_card(e):return card_html(f"{e['stage']}/{N_STAGES} · {e['title']}",html.escape(e["body"])+pbar_html(e["stage"],N_STAGES),"info")
def ledger_text():return"\n".join(json.dumps(x,ensure_ascii=False)for x in SESSION_LEDGER)
def next_case():return RUN_COUNTER%N_CASES
def next_case_html(I=None):
    I=I or INSTR;c=next_case();row=I.cases()[c];t=row["target"];C=I.car(t)
    body=(f'<span class="mono">question</span> {html.escape(row["question"])}<br>'
          f'<span class="mono">WRITE</span> fact of this case (re-written live, compared with its start-up file): {html.escape(C["fact"])} → {html.escape(C["file"])}<br>'
          f'<span class="mono">READ</span> the question alone selects one of {N_CART:,} files; the fact is not given to the model<br>'
          f'<span class="mono">rotation</span> predeclared: RUN number n uses case (n − 1) mod {N_CASES} of the engine\'s {N_CASES} new-wording questions — no choice, no cherry-picking')
    return card_html(f"Next RUN · case {c:02d} of {N_CASES}",body,"info")
def engine_card_html(I):
    inf=I.info;cfg=I.config();su=I.startup();S=su["summary"];st=su["storage"];ix=su["index"];T7=TEST627_LOG
    rows=[("model",f"{inf['model_id']} · frozen · {inf['dtype']} · {arch_txt(tuple(cfg['arch']))}"),("GPU",inf["gpu"]),
          ("engine",f"TEST627 demo engine (verbatim) · SHA-256 {inf['engine_sha256'][:16]}… · chain 626.py {cfg['chain_sha256']['626.py'][:12]}… · 624.py {cfg['chain_sha256']['624.py'][:12]}… · 610.py {cfg['chain_sha256']['610.py'][:12]}…"),
          ("WRITE at start-up",f"{su['write']['files']:,} CUT3 files written by the frozen model · p50 {su['write']['ms_p50']:.0f} ms per file"),
          ("SSD",f"{st['cartridges']:,} independent files · {st['disk_pool_bytes']:,} bytes · avg {st['avg_file_bytes']/1e6:.2f} MB · SHA-256 manifest verified: {su['integrity'].get('manifest_1453_sha_verified')}"),
          ("RAM address",f"PCA{ix['pca_dims']} + FAISS IVF{ix['ivf_lists']} over {ix['native_positions']:,} positions ({ix['index_bytes']/1e6:.1f} MB) + native layer-3 vectors ({ix['native_activation_bytes']/1e6:.0f} MB)"),
          ("READ (locked)",f"nprobe {cfg['candidates'][0]} · token_k {cfg['candidates'][1]} · {cfg['candidates'][2]} candidates · {cfg['reranker'][0]}"),
          (f"start-up acceptance (this runtime, finished {str(su.get('measured_utc',''))[:19]} UTC)",f"original wording {S['ORIGINAL']['answer_correct']}/48 · new wording {S['ALTERNATE']['answer_correct']}/48 answered from the disk files · errors {S['ORIGINAL']['errors']+S['ALTERNATE']['errors']} · integrity {sum(bool(v)for v in su['integrity'].values())}/{len(su['integrity'])}"),
          ("weights",f"full-weight SHA-256 {inf['weight_sha0'][:16]}… (= reference) · sentinel {inf['sentinel0'][:16]}… · guard {inf['guard0'][:16]}…"),
          ("init",f"{inf['init_seconds']:.0f} s" if isinstance(inf['init_seconds'],float)and math.isfinite(inf['init_seconds'])else"reused")]
    body="".join(f'<span class="mono">{html.escape(a)}</span> {html.escape(str(b))}<br>'for a,b in rows)
    body+=(f'<div class="small" style="margin-top:8px"><b>SEALED TEST627 LOG</b> (627.log; not recomputed) · {T7["decision"]} · original {T7["e2e"]["ORIGINAL"]["answer_correct"]}/48 · new wording {T7["e2e"]["ALTERNATE"]["answer_correct"]}/48 · '
           f'log SHA-256 {T7["log_sha256"][:16]}…<br><b>How PEAK50 was locked</b> (626.log): TEST626\'s own pre-declared rule chose {TEST626_LOG["locked_by_own_rule"]} ({TEST626_LOG["e2e"]}); PEAK50 was fixed afterwards — the 96 questions are therefore not a blind test.</div>')
    return card_html("Engine status · start-up in this runtime · sealed references",body,"on")
HOW_HTML=card_html("What one RUN shows",
    "<b>A · WRITE</b> — the frozen model reads one fact and keeps its own key/value state (CUT3): 28 layers × (K, V). The engine wrote 1,453 such files at start-up; RUN writes the case's fact again and checks the result is bit-for-bit the same file.<br>"
    "<b>B · SSD</b> — every cartridge is a separate file on disk with a SHA-256 fingerprint; RUN checks all 1,453 files are present with the recorded sizes.<br>"
    "<b>C · READ</b> — only the question goes to the model. Its internal vectors are looked up in a geometric index in RAM (PCA128 + FAISS IVF128); 16 candidates are re-scored by PEAK50 and one file is chosen.<br>"
    "<b>D · ANSWER</b> — that file is loaded and the frozen model answers from it. The fact text is not in the prompt. A control repeats the question over the runner-up file.<br>"
    "<b>E · RESULTS</b> — four figures (JPEG 300 dpi + PDF), a sealed log and a ZIP; start-up acceptance, the sealed TEST627 log and the limits are shown together.","info")
def is_cuda_error(ex):
    s=f"{type(ex).__name__} {ex}".lower();return"cuda"in s or"device-side assert"in s or"cublas"in s
def run_handler():
    global RUN_COUNTER
    if not GPU_LOCK.acquire(blocking=False):
        yield pack(card_html("Busy","Another run is in progress. Please try again in a moment.","warn"));return
    stage_name="initialisation"
    try:
        c=next_case();RUN_COUNTER+=1;run_id=f"{RUN_PREFIX}{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}-{RUN_COUNTER:04d}-C{c:02d}"
        prune_runs(4);run_dir=ROOT/run_id;run_dir.mkdir(parents=True,exist_ok=True)
        preview=[dict(case=x["case"],correct=x["correct"])for x in SESSION_LEDGER]
        gen=execute_run(INSTR,dict(run_id=run_id,run_dir=run_dir,case=c,session=preview+[dict(case=c,correct=None)]));first=True
        while True:
            try:e=next(gen)
            except StopIteration as s:B=s.value;break
            stage_name=f"{e['stage']}/{N_STAGES} {e['title']}"
            yield pack(stage_card(e),[],None,None,None)if first else pack(stage_card(e));first=False
        SESSION_LEDGER.append(B["ledger"]);LEDGER_PATH.write_text(ledger_text()+"\n",encoding="utf-8")
        P=B["P"];L=P["live"];ok=L["correct"];Wr=L["write"];Rd=L["read"];Ct=L["control"];su=P["startup"];S=su["summary"]
        raw={"txt":B["txt"].read_text(encoding="utf-8"),"json":B["payload"].read_bytes().decode("utf-8"),"man":B["manifest"].read_text(encoding="utf-8"),"ledger":ledger_text()}
        ctl=("not applicable"if Ct.get("skipped")else f'#{Ct["id"]:04d} → {html.escape(str(Ct["answer"]))} · expected answer produced: {"yes" if Ct["gold_produced"] else "no"}')
        body=(f"<b>{html.escape(B['run_id'])}</b> · LIVE DEMONSTRATION RUN · case {L['case']:02d}<br>"
              f'<span class="mono">A · WRITE</span> {html.escape(L["fact"])} → {Wr["T"]} rows × {Wr["layers"]} layers · re-written now, bitwise identical to {html.escape(Wr["file"])}: <b>{Wr["bit_equal"]}</b> · fact text in file: {Wr["fact_in_file"]}<br>'
              f'<span class="mono">B · SSD</span> {L["ssd"]["files_in_pool"]:,} independent files · {L["ssd"]["total_bytes"]:,} bytes · sizes = manifest: {not L["ssd"]["size_mismatch"]}<br>'
              f'<span class="mono">C · READ</span> question “{html.escape(L["question"])}” → touched {Rd["touched"]} of {N_CART:,} → {len(Rd["ranking"])} candidates → PEAK50 selects #{Rd["selected"]:04d} ({html.escape(Rd["file"])}) · {Rd["address_ms"]:.1f} ms<br>'
              f'<span class="mono">D · ANSWER</span> <b>{html.escape(L["answer"])}</b> · expected {html.escape(L["expected"])} · <b>{"CORRECT" if ok else "INCORRECT"}</b> · model input: {Rd["query_tokens"]} question tokens over the loaded {Rd["T"]}-row cartridge · load {Rd["load_ms"]:.1f} ms · answer {Rd["answer_ms"]:.0f} ms<br>'
              f'<span class="mono">control</span> runner-up file {ctl}<br>'
              f'<span class="mono">this RUN</span> one new run, started {html.escape(P["run_start_utc"][:19])} UTC — the result above is this run only<br>'
              f'<span class="mono">separately</span> engine start-up acceptance in this runtime (finished {html.escape(str(su.get("measured_utc",""))[:19])} UTC): original {S["ORIGINAL"]["answer_correct"]}/48 · new wording {S["ALTERNATE"]["answer_correct"]}/48 · archived 627.log record: 48/48 · 48/48 — not a blind test; see limits<br>'
              f"weights unchanged (sentinel + all-parameter guard) · verdict <b>{html.escape(B['verdict'])}</b> · {N_FIGS} figures + PDFs + Turkish Fig. 1 + ZIP · {B['checks']}/{B['checks']} checks PASS<br>"
              f'<span class="mono">payload SHA-256 {B["sha"]}</span>')
        yield pack(card_html(f"{N_STAGES}/{N_STAGES} · Sealed",body,"on"),[(str(p),c_)for p,c_ in B["imgs"]],
                   {"zip":B["zip"],"json":B["payload"],"txt":B["txt"],"man":B["manifest"],"engine":B["engine"],"tr":B["tr_jpg"],"ledger":LEDGER_PATH},[p for p,_ in B["imgs"]],raw,next_case_html())
    except Exception as ex:
        print("="*140);print(f"RUN FAILED — stage: {stage_name}");print(f"{type(ex).__name__}: {ex}");traceback.print_exc();print("="*140)
        kind="AUDIT FAIL — nothing was sealed. "if isinstance(ex,AuditFail)else"";partial=getattr(ex,"partial",None)
        if partial:kind+="The raw outputs gathered so far were preserved (FULL RUN LOG). "
        tip="<br><b>CUDA error: Runtime → Restart runtime, then run PARTS 1–3 again.</b>"if is_cuda_error(ex)else"<br>The full traceback is printed in the Colab console."
        card=card_html("Run failed",f"{html.escape(kind)}Stage: {html.escape(stage_name)}<br>{html.escape(type(ex).__name__)}: {html.escape(str(ex)[:600])}{tip}","err")
        if partial:yield pack(card,[],{"json":partial},None,{"txt":traceback.format_exc(),"json":Path(partial).read_text(encoding="utf-8"),"man":"","ledger":ledger_text()},next_case_html())
        else:yield pack(card,[],None,None,None,next_case_html())
    finally:
        plt.close("all");GPU_LOCK.release()
        try:torch.cuda.empty_cache()
        except Exception:pass
def verify_handler():
    if not GPU_LOCK.acquire(blocking=False):return"Busy: a run is in progress."
    try:
        t=time.perf_counter();g=INSTR.guard();fz=INSTR.frozen();w=INSTR.full_weight_sha();cfg=INSTR.config();pa=INSTR.pool_sha_audit()
        out=dict(model=MODEL_ID,full_weight_sha256_now=w,full_weight_sha256_reference=WEIGHT_SHA_EXPECTED,full_weight_identical_to_reference=w==WEIGHT_SHA_EXPECTED,
                 all_parameter_guard_now=g,all_parameter_guard_startup=INSTR.info["guard0"],guard_unchanged=g==INSTR.info["guard0"],sentinel_unchanged=fz["sentinel"]==INSTR.info["sentinel0"],
                 trainable_tensors=fz["trainable_tensors"],foreign_hooks=fz["foreign_hooks"],residual_hooks=fz["residual_hooks"],
                 ssd_pool_sha256_audit=dict(files=pa["files"],mismatches=len(pa["mismatch"]),mismatch_ids=pa["mismatch"][:20],seconds=round(pa["seconds"],2)),
                 engine_sha256=cfg["engine_sha256"],engine_sha256_expected=ENGINE_SHA256_EXPECTED,chain_sha256=cfg["chain_sha256"],source_blobs=cfg["source_blobs"],
                 locked_read=dict(candidates=cfg["candidates"],reranker=cfg["reranker"]),architecture=cfg["arch"],checked_utc=utc_now(),seconds=round(time.perf_counter()-t,3),
                 note="SHA-256 values show that weights and files are unchanged; they are not scientific proof.")
        return json.dumps(jsafe(out),indent=2,ensure_ascii=False)
    finally:GPU_LOCK.release()
# ---------------- service self-test (runs before the public link is printed) ----------------
def service_selftest():
    global QUIET;out=[];d=ROOT/"_selftest";shutil.rmtree(d,ignore_errors=True);d.mkdir(parents=True);(d/"w.txt").write_text("ok");out.append("ROOT writable");QUIET=True
    try:
        for mode,c_ in(("ok",0),("ok",5),("wrong",12)):
            dd=d/f"{mode}{c_}";dd.mkdir()
            B=drain(execute_run(SelfTestInstrument(mode,d/"eng"),dict(run_id=f"SELFTEST-{mode.upper()}-C{c_:02d}",run_dir=dd,case=c_,session=[dict(case=c_,correct=None)])))
            assert len(B["imgs"])==N_FIGS and len(B["pdfs"])==N_FIGS and file_ready(B["zip"])and file_ready(B["tr_jpg"])and B["verdict"].endswith("SOURCE_FREE_ANSWER")and B["verdict"].startswith("SELFTEST_SYNTHETIC_")
            assert("_INCORRECT_"in B["verdict"])==(mode=="wrong"or c_==5)
        out.append(f"full pipeline on a synthetic instrument (correct answer, selection miss shown as INCORRECT, forced miss): {N_FIGS}/{N_FIGS} figures + Turkish Fig. 1 (JPEG 300 dpi + PDF), ZIP, {B['checks']} checks")
        for mode,what in(("guard_changes","a changed weight guard"),("write_diff","a WRITE result that is not bitwise identical to its file"),("write_extra","an extra model forward during WRITE"),
                         ("fact_in_file","fact text inside a cartridge file"),("pool_size","a pool file whose size differs from the manifest"),("frame_has_fact","fact tokens in the question frame"),
                         ("peak_rederive","a PEAK50 score that does not re-derive"),("cand_merge","a ranked set that differs from the candidate626 set"),
                         ("read_mismatch","a live READ that does not reproduce the engine's start-up record"),("sha_mismatch","a selected file whose SHA-256 differs from the manifest"),
                         ("source_in_query","fact tokens in the answering input"),("residual_hook","a hook left on a decoder layer after the run"),("engine_status","an engine start-up status other than COMPLETE")):
            d2=d/mode;d2.mkdir()
            try:drain(execute_run(SelfTestInstrument(mode,d/"eng"),dict(run_id="SELFTEST-"+mode.upper(),run_dir=d2,case=3,session=[])))
            except AuditFail:out.append(f"fail-closed: {what} aborts the run (nothing sealed)")
            else:raise RuntimeError(f"self-test: {what} did not abort the run")
    finally:QUIET=False
    if len(pack(HOW_HTML))!=N_OUTPUTS or len(pack(HOW_HTML,[],None,None,None,""))!=N_OUTPUTS:raise RuntimeError("callback output count mismatch")
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
      f'<div class="brand">AKBASCORE MAM</div><div class="title">The Silicon Hippocampus · {html.escape(DEMO_SUB)} · {html.escape(MODEL_FAMILY)}</div>'
      f'<div class="para">{html.escape(PARADIGM)}.</div><div class="q">{html.escape(SCI_QUESTION)}</div>'
      f'<div class="by"><b>{html.escape(DISCOVERY)}.</b><br>{html.escape(ORG)} · {html.escape(DATE_TXT)} · {html.escape(AUTHOR_PLACE)}</div>'
      '<ul class="msg">'+"".join(f"<li>{html.escape(m)}</li>"for m in CORE_MESSAGE)+"</ul>"
      f'<div class="lic">{html.escape(NAME_NOTE)}</div><div class="lic">{html.escape(PRIORITY)}</div>'
      f'<div class="lic">{html.escape(LICENSE_NAME)} · {html.escape(COPYRIGHT)} · <a href="{LICENSE_URL}" target="_blank" rel="noopener">{html.escape(LICENSE_URL)}</a> · '
      f'<a href="{URL_627}" target="_blank" rel="noopener">627.py</a> · <a href="{URL_627LOG}" target="_blank" rel="noopener">TEST627 log</a> · <a href="{URL_626LOG}" target="_blank" rel="noopener">TEST626 log</a> · '
      f'<a href="{URL_624}" target="_blank" rel="noopener">624.py</a> · <a href="{URL_610}" target="_blank" rel="noopener">610.py</a> · '
      f'DOI <a href="{PRIOR_PUB["url"]}" target="_blank" rel="noopener">{html.escape(PRIOR_PUB["doi"])}</a> · DOI <a href="{QWEN_PUB["url"]}" target="_blank" rel="noopener">{html.escape(QWEN_PUB["doi"])}</a> · '
      f'DOI <a href="{V12_PUB["url"]}" target="_blank" rel="noopener">{html.escape(V12_PUB["doi"])}</a></div></div>')
SCOPE_HTML=card_html("Scope of this demonstration",
    "<b>Demonstrated live in every RUN:</b><br>"+"<br>".join("• "+html.escape(t)for t in SCOPE["demonstrated"])+"<br><br><b>Produced at start-up in this runtime:</b><br>"+"<br>".join("• "+html.escape(t)for t in SCOPE["startup"])
    +"<br><br><b>Archived (sealed logs):</b><br>"+"<br>".join("• "+html.escape(t)for t in SCOPE["archived"])
    +"<br><br><b>Not established:</b><br>"+"<br>".join("• "+html.escape(t)for t in SCOPE["not_established"])+"<br><br><b>Technical notes:</b><br>"+"<br>".join("• "+html.escape(t)for t in SCOPE["notes"]),"info")
READY_HTML=card_html("Ready",f"Press <b>RUN</b>. Nothing to type or choose: each RUN takes the next predeclared case, re-runs WRITE for its fact, checks the SSD pool, READs the answer from the question alone, runs the control, verifies the frozen weights, seals the payload and renders {N_FIGS} figures (JPEG 300 dpi + vector PDF, Fig. 1 also in Turkish). About 20–40 seconds.","info")
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
_TITLE="AKBASCORE MAM · The Silicon Hippocampus · Qwen2.5-7B · Mustafa Akbaş"
if _GR_MAJOR>=6:_blocks=gr.Blocks(title=_TITLE)
else:
    try:_blocks=gr.Blocks(css=CSS,title=_TITLE)
    except TypeError:_blocks=gr.Blocks(title=_TITLE)
with _blocks as demo:
    gr.HTML(HERO);gr.HTML(HOW_HTML);gr.HTML(engine_card_html(INSTR))
    nxt=gr.HTML(next_case_html())
    run_btn=gr.Button("RUN · WRITE → SSD → READ → answer without the source text → control → seal",variant="primary",elem_id="kz_run")
    status=gr.HTML(READY_HTML)
    gallery=_gallery(label=f"{N_FIGS} figures (tap to open · JPEG 300 dpi; vector PDFs and the Turkish Fig. 1 in the ZIP)",show_label=True)
    dl_all=gr.Button(DL_ALL_LABEL,interactive=False)
    jpg_urls=gr.Textbox(value="",visible=False)
    with gr.Row():
        dlb=[gr.DownloadButton(label=l,value=None,interactive=False)for l in DL_LABELS]
    with gr.Accordion("Direct file links (fallback)",open=False):
        fls=[gr.File(label=n,interactive=False)for n in("ZIP · figures, PDFs, payload, logs, engine result","FULL RUN LOG (.json)","READABLE RUN LOG (.txt)","RUN MANIFEST (.json)","ENGINE START-UP RESULT (.json)","Fig. 1 · Türkçe (.jpg)","SESSION LEDGER (.jsonl)")]
    with gr.Tabs():
        with gr.Tab("READABLE RUN LOG (.txt)"):raw_txt=_tb(lines=18,max_lines=40,label="readable run log")
        with gr.Tab("FULL RUN LOG (.json)"):raw_json=_tb(lines=18,max_lines=40,label="sealed canonical payload")
        with gr.Tab("MANIFEST (.json)"):raw_man=_tb(lines=12,max_lines=30,label="run manifest")
        with gr.Tab("SESSION LEDGER"):raw_led=_tb(lines=10,max_lines=30,label="one line per sealed run in this session")
    with gr.Accordion("Verification · full-weight SHA-256 · weight guard · SHA-256 of all 1,453 cartridge files (about 10–30 s)",open=False):
        ver_btn=gr.Button("Run verification now")
        ver_out=_tb(lines=14,max_lines=40,label="verification record")
    gr.HTML(SCOPE_HTML)
    gr.HTML(f'<div class="kz small" style="opacity:.8;margin:8px 0 18px">{html.escape(DISCOVERY)} · {html.escape(ORG)} · {html.escape(DATE_TXT)} · {html.escape(LICENSE_NAME)} · '
            f'<a href="{LICENSE_URL}" target="_blank" rel="noopener">{html.escape(LICENSE_URL)}</a></div>')
    OUTS=[status,gallery,dl_all,jpg_urls]+dlb+fls+[raw_txt,raw_json,raw_man,raw_led,nxt]
    if len(OUTS)!=N_OUTPUTS:raise RuntimeError(f"UI outputs {len(OUTS)} != {N_OUTPUTS}")
    run_btn.click(run_handler,inputs=None,outputs=OUTS)
    ver_btn.click(verify_handler,inputs=None,outputs=ver_out)
    try:dl_all.click(None,inputs=[jpg_urls],outputs=None,js=DL_ALL_JS)
    except TypeError:dl_all.click(None,inputs=[jpg_urls],outputs=None,_js=DL_ALL_JS)
try:demo.queue(default_concurrency_limit=1)
except TypeError:demo.queue(concurrency_count=1)
say("[3/3] LAUNCH (public share link) — open the printed gradio.live link and press RUN.")
_LAUNCH=dict(share=True,allowed_paths=ALLOWED_PATHS,show_error=True,inline=False)
if _GR_MAJOR>=6:_LAUNCH["css"]=CSS
try:demo.launch(**_LAUNCH)
except TypeError as _ex:
    if"css"not in str(_ex):raise
    _LAUNCH.pop("css",None);demo.launch(**_LAUNCH)
