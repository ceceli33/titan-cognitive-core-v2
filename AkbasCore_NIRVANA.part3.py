def selftest_payload():
    rng=np.random.default_rng(0);cards=[]
    for b in BANK_SPEC:
        T=len(b["text"].split())+4
        cards.append(dict(b,T=T,cosK=list(0.98+0.01*rng.random(28)),cosV=list(0.999+0.0009*rng.random(28)),normK=list(20+5*rng.random(28)),normV=list(3+rng.random(28)),
                          code_numbers=28*(T-1)*4*248,own_numbers=28*1024,native_numbers=28*1024*T,hidden_copy_numbers=28*T*3584,encode_seconds=.05,gpu_mib=15000.0))
    bt=[]
    for q in BATTERY:
        lane=[0]*16;bt.append(dict(q,expected=q["exp_place"] or "UNKNOWN",observed="SELFTEST",obs_place=None,status="UNKNOWN ✓" if q["exp_place"] is None else "WRONG",lane=lane))
    kinds={k_:dict(ok=0,n=4) for k_ in["LINKED","MISSING LINK","DIRECT","ABSENT ID"]}
    num=lambda v,l:dict(value=v,label=l,human=human(v))
    return dict(run_id="SELFTEST",cartridges=cards,battery=bt,errors=[],
        live_summary=dict(n=16,correct=8,unknown_correct=8,false_links=0,by_kind=kinds,by_family={f:dict(ok=0,n=2) for f in["F1","F2","F3","F4"]}),
        bank=dict(slots=300,code_numbers=10**6,own_numbers=16*28*1024,native_numbers=10**6,hidden_copy_numbers=4*10**6,runtime_numbers=10**6,ratio_vs_native=.97,ratio_vs_hidden=.28),
        source_removal=dict(source_sentence_hits=0,source_sentences=16,calls=[dict(tag=f"c{i}",prompt_tokens=40,source_hits=0,slots=300) for i in range(24)]),
        controls=dict(nomem=dict(strict_hits=0,n=4)),codebook=dict(seconds=4.0,bytes=29*2**20),gpu=dict(peak_allocated_gib=16.0,name="SELFTEST GPU"),
        model=dict(params=7.6e9),ui=dict(stages=7),timing=dict(startup=dict(model_load_seconds=40.0),run=dict(cartridges_seconds=2.0,battery_seconds=20.0,control_seconds=1.0,engine_seconds=30.0)),
        workload=dict(measured_order=["forward_passes","ibr_calls","ibr_decode_steps","forward_tokens","generated_tokens","svd"],
            derived_order=["cart_encdec_mac","code_numbers","numbers_installed","memory_slot_reads","frozen_params"],estimated_order=["linear_flop","attn_flop","total_flop"],
            measured=dict(forward_passes=num(800,"transformer forward passes"),ibr_calls=num(24,"IBR batched calls"),ibr_decode_steps=num(500,"IBR decode steps"),
                          forward_tokens=num(20000,"token positions processed"),generated_tokens=num(3000,"tokens generated (all rows)"),svd=num(224,"SVD decompositions (codebook)")),
            derived=dict(cart_encdec_mac=num(10**9,"cartridge encode+decode multiply-adds"),code_numbers=num(10**6,"numbers stored in the 16 cartridges"),
                         numbers_installed=num(10**7,"numbers installed into the cache"),memory_slot_reads=num(10**10,"query × cartridge-slot reads (heads × layers)"),frozen_params=num(7.6e9,"frozen model parameters")),
            estimated=dict(linear_flop=num(3e14,"linear-layer FLOPs (est.)"),attn_flop=num(1e12,"attention FLOPs (est.)"),total_flop=num(3.01e14,"total GPU FLOPs (est.)")),
            formula_note="DERIVED cartridge MACs = Σ 2 × 28 × (T−1) × 4 × 128 × (120+128). ESTIMATED: linear = N nn.Linear weights × token positions (incl. lm_head on every position); attention = 2 × 28 heads × 128 × 28 layers × Σ attended length. FLOPs = 2 × MACs. Excludes norms, RoPE, softmax and element-wise ops."),
        integrity=dict(fingerprint_startup=[1.0]*6,fingerprint_pre_run=[1.0]*6,fingerprint_after=[1.0]*6,sentinel_startup="0"*64,sentinel_pre_run="0"*64,sentinel_after="0"*64,
            trainable_parameter_tensors=0,lora=False,akbascore_hooks_after_run=0,model_training_mode=False,result="PASS",checks_total=20),
        environment=dict(gpu="SELFTEST",torch="x",transformers="x"),engine=dict(lock_sha256=LOCK_SHA))
def render_all(ctx,run_dir):
    imgs=[]
    try:
        for i,(slug,cap,fn) in enumerate(POSTERS,1):
            p=run_dir/f"{slug}_{ctx['run_id']}.jpg";fn(ctx,i,p);imgs.append((p,cap))
    finally:plt.close("all")
    return imgs
# ---------------------------------------------------------------- RUN (generator, live stages) ----------------------------------------------------------------
def gen_config():
    try:return jsafe(json.loads(json.dumps(model.generation_config.to_dict(),default=str)))
    except Exception as ex:return f"unavailable: {ex}"
def file_ready(p):return p is not None and Path(p).is_file()and Path(p).stat().st_size>0
def prune_runs(keep=2):
    runs=sorted([p for p in ROOT.glob("CARTRIDGE-*")if p.is_dir()],key=lambda p:p.stat().st_mtime)
    for p in runs[:max(0,len(runs)-keep)]:shutil.rmtree(p,ignore_errors=True)
def post_seal_audit(imgs,zp,expected_zip_names,pp,sha,tp,mp):
    checks=[]
    def ok(name,cond):
        checks.append(name)
        if not cond:raise RuntimeError(f"POST-SEAL AUDIT FAILED: {name}")
    ok(f"{NPOST}/{NPOST} JPG created",len(imgs)==NPOST and all(file_ready(p)for p,_ in imgs))
    bad=[]
    for p,_ in imgs:
        with Image.open(p)as im:
            if not(im.format=="JPEG" and im.mode=="RGB"):bad.append(p.name)
    ok(f"{NPOST}/{NPOST} JPG JPEG/RGB validation",not bad)
    with zipfile.ZipFile(zp)as z:
        ok("ZIP testzip()",z.testzip()is None);names=z.namelist()
        ok("ZIP flat (no sub-folders)",all("/" not in n for n in names))
        ok(f"ZIP expected contents ({NPOST} JPG + images manifest + run manifest)",sorted(names)==sorted(expected_zip_names))
        jp=[n for n in names if n.lower().endswith(".jpg")];ok(f"JPG numbering 01–{NPOST:02d}",[n[:2]for n in sorted(jp)]==[f"{i:02d}" for i in range(1,NPOST+1)])
    ok("payload JSON exists",file_ready(pp));ok("payload SHA-256 recomputation",hashlib.sha256(pp.read_bytes()).hexdigest()==sha)
    ok("payload JSON parses",isinstance(json.loads(pp.read_bytes().decode("utf-8")),dict));ok("TXT exists and non-empty",file_ready(tp))
    ok("run manifest exists and parses",file_ready(mp)and json.loads(mp.read_bytes().decode("utf-8")).get("payload_sha256")==sha)
    ok("ZIP exists and non-empty",file_ready(zp));ok("no AkbasCore forward hooks",our_hooks_total()==0)
    ok("attention implementation SDPA",getattr(cfg,"_attn_implementation",None)=="sdpa")
    return checks
def make_txt(P,sha,payload_name,manifest_name,seal):
    o=[];a=o.append;S="="*110;Dd="-"*110
    a(S);a("AKBASCORE NIRVANA · COGNITIVE CARTRIDGE — READABLE RUN LOG");a(S)
    a(f"Derived from the sealed payload {payload_name} (SHA-256 {sha}). Verify with {manifest_name}.")
    a("The SHA-256 value is an artifact integrity seal; it does not by itself establish any scientific interpretation.")
    for k_,v in[("RUN ID",P["run_id"]),("START UTC",P["run_start_utc"]),("END UTC",P["run_end_utc"]),("PAYLOAD SEALED UTC",seal["sealed_utc"]),("START LOCAL",P["run_start_local"]),
               ("MODEL",MODEL_ID),("DTYPE / ATTENTION","bfloat16 / sdpa"),("LAYERS / HIDDEN / HEADS",f"{NL} / {H} / {NH}Q {NKV}KV × {HD}"),
               ("GPU",P["environment"]["gpu"]),("TORCH",P["environment"]["torch"]),("TRANSFORMERS",P["environment"]["transformers"]),("GRADIO",P["environment"]["gradio"]),
               ("SEED",SEED),("ENGINE","TEST461 IBR · K120/V128/OWN · fixed neutral codebook · source absent at readout"),("COMPUTE PATH","PyTorch CUDA backend (cuBLAS, SDPA); no custom CUDA/C++ kernel"),
               ("READOUT FRAME",FMT.replace("\n","\\n")),("DECODING",f"greedy argmax in IBR loop, max_new_tokens={MAX_NEW}"),("DEMO PANEL LOCK SHA-256",P["engine"]["lock_sha256"])]:a(f"{k_:<26}: {v}")
    a("");a("COGNITIVE CARTRIDGES (source sentences, forged once; never placed in a readout prompt)");a(Dd)
    for c in P["cartridges"]:
        a(f"{c['cid']} [{c['role']}] {c['fam']} T={c['T']} code_numbers={c['code_numbers']:,} native_numbers={c['native_numbers']:,} encode={c['encode_seconds']:.4f}s gpu={c['gpu_mib']:.0f}MiB")
        a(f"   text: {c['text']}");a("   cosK: "+" ".join(f"{v:.4f}" for v in c["cosK"]));a("   cosV: "+" ".join(f"{v:.4f}" for v in c["cosV"]))
    a("");a("AUTOMATIC QUESTION BATTERY (16 locked questions, IBR over all 16 cartridges)");a(Dd)
    for q in P["battery"]:
        a(f"{q['qid']} [{q['kind']}] {q['label']} | expected={q['expected']} | observed={q['observed']} | status={q['status']} | false_link={q['false_link']} | {q['seconds']:.3f}s")
        for stg in("stage1","stage2"):
            if q.get(stg):
                a(f"   {stg} prompt: {q[stg]['prompt']!r}")
                for cid,t in zip([c["cid"] for c in P["cartridges"]],q[stg]["texts"]):a(f"     {cid}: {t!r}")
    nm=P["controls"]["nomem"];a("");a(f"NOMEM CONTROL (PAD-only cache, no source-derived information): strict hits {nm['strict_hits']}/{nm['n']}");a(Dd)
    for r in nm["rows"]:a(f"   {r['qid']}: stage1={r['stage1']!r} stage2={r['stage2']!r} cid={r['cid']} place={r['place']} strict_hit={r['strict_hit']}")
    S_=P["live_summary"];a("");a("LIVE SUMMARY (THIS RUN)");a(Dd);a(json.dumps(S_,ensure_ascii=False))
    a("");a("SOURCE-REMOVAL AUDIT");a(Dd)
    for c in P["source_removal"]["checks"]:a("PASS · "+c)
    a(f"source sentences found in readout prompts: {P['source_removal']['source_sentence_hits']}")
    a("");a("WORKLOAD");a(Dd)
    for g in("measured","derived","estimated"):
        for k_,v in P["workload"][g].items():a(f"{g:<9} {v['label']:<46}: {v['value']:,} ({v['human']})")
    a(P["workload"]["formula_note"])
    I=P["integrity"];a("");a("INTEGRITY");a(Dd)
    for k_ in("fingerprint_startup","fingerprint_pre_run","fingerprint_after","sentinel_startup","sentinel_pre_run","sentinel_after"):a(f"{k_:<22}: {I[k_]}")
    a(f"RESULT : {I['result']} | trainable tensors {I['trainable_parameter_tensors']} | training mode {I['model_training_mode']} | AkbasCore hooks {I['akbascore_hooks_after_run']}")
    a("pre-seal checks: "+"; ".join(I["pre_seal_checks"]))
    a("");a("SEALED EXPERIMENTAL RECORD (historical, NOT produced by this run)");a(Dd);a(json.dumps(HIST,ensure_ascii=False,default=str))
    a("");a("TIMING");a(Dd)
    for g in("startup","run"):
        for k_,v in P["timing"][g].items():a(f"{g}.{k_:<30}: {v}")
    a(f"{'sealing_stage_seconds':<38}: {seal['sealing_stage_seconds']}");a("")
    a("NOTE: live results come from the locked 16-question demo panel; the sealed record is reported separately and never merged.")
    a("NOTE: the weight sentinel samples selected tensors; it is not a full cryptographic verification of every weight.");a(S)
    return "\n".join(o)
RUN_COUNTER=0;GPU_LOCK=threading.Lock()
SKIP=getattr(gr,"skip",None)or gr.update
_K=object()
FILE_KEYS=["json","txt","man"]
DL_LABELS=["⬇ DOWNLOAD FULL RUN LOG (.json)","⬇ DOWNLOAD READABLE RUN LOG (.txt)","⬇ DOWNLOAD RUN MANIFEST (.json)"]
DL_ALL_LABEL=f"⬇ DOWNLOAD ALL {NPOST} JPGs"
RAW_KEYS=["txt","json","man"]
N_OUTPUTS=2+2+len(FILE_KEYS)*2+len(RAW_KEYS)
try:
    from gradio.routes import API_PREFIX as _GR_API_PREFIX
except Exception:
    _GR_API_PREFIX="/gradio_api" if int(gr.__version__.split(".")[0])>=5 else ""
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
def jpg_urls_json(paths):
    items=[]
    for p in paths:p=Path(p);items.append({"url":FILE_URL_PREFIX+quote(str(p),safe="/"),"name":p.name})
    return json.dumps(items,ensure_ascii=False)
def pack(status=_K,gal=_K,files=_K,jpgs=_K,raw=_K):
    out=[SKIP()if status is _K else status,SKIP()if gal is _K else gal]
    if jpgs is _K:out+=[SKIP(),SKIP()]
    elif jpgs is None:out+=[btn_update(False),""]
    else:
        pl=[Path(p)for p in jpgs]
        for pth in pl:
            if not file_ready(pth):raise RuntimeError(f"JPG artifact missing or empty: {pth}")
        out+=[btn_update(True),jpg_urls_json(pl)]
    if files is _K:out+=[SKIP()for _ in range(len(FILE_KEYS)*2)]
    elif files is None:out+=[dl_update(None,l)for l in DL_LABELS]+[None]*len(FILE_KEYS)
    else:
        paths=[str(files[k])for k in FILE_KEYS]
        for pth in paths:
            if not file_ready(pth):raise RuntimeError(f"Download artifact missing or empty: {pth}")
        out+=[dl_update(p,l)for p,l in zip(paths,DL_LABELS)]+paths
    if raw is _K:out+=[SKIP()for _ in RAW_KEYS]
    elif raw is None:out+=[""]*len(RAW_KEYS)
    else:out+=[raw[k]for k in RAW_KEYS]
    return tuple(out)
def card_html(title,body,kind="info"):return f'<div class="kz card {kind}"><div class="h">{html.escape(title)}</div><div class="small">{body}</div></div>'
def bar_html(done,total,color="#B45309"):
    cells="".join(f'<span style="display:inline-block;width:{94/total:.2f}%;height:14px;margin:1px;border-radius:3px;background:{color if i<done else "#e2e8f0"}"></span>' for i in range(total))
    return f'<div style="margin-top:6px">{cells}</div>'
def stage(i,title,body):return card_html(f"{i}/7 · {title}",body,"info")
READY_HTML=card_html("Ready",f"Press <b>RUN COGNITIVE CARTRIDGE DEMO</b>. The run loads 16 cartridges, removes the source, asks 16 automatic questions, seals the evidence and renders {NPOST} JPG posters.<br><b>Downloads appear when the run is complete.</b>","info")
def is_cuda_error(ex):
    s=f"{type(ex).__name__} {ex}".lower();return "cuda" in s or "device-side assert" in s or "cublas" in s
def run_handler():
    global RUN_COUNTER
    if not GPU_LOCK.acquire(blocking=False):
        yield pack(card_html("Busy","Another run is in progress. Please try again in a moment.","warn"));return
    stage_name="initialisation"
    try:
        RUN_COUNTER+=1;run_id=f"CARTRIDGE-{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}-{RUN_COUNTER:04d}"
        prune_runs(2);run_dir=ROOT/run_id;run_dir.mkdir(parents=True,exist_ok=True)
        print(f"\n{'='*110}\nRUN {run_id}\n{'='*110}")
        checks=[];errors=[]
        def chk(name,cond):
            checks.append(name)
            if not cond:raise RuntimeError(f"CHECK FAILED: {name}")
        for k_ in CNT:CNT[k_]=0
        torch.cuda.reset_peak_memory_stats();gmem=lambda:torch.cuda.memory_allocated()/2**20
        stage_name="1/7 integrity check"
        yield pack(stage(1,"Integrity check","Verifying the frozen model: hooks, trainable tensors, LoRA, optimizer and sampled weight sentinels."),[],None,None,None)
        chk("no AkbasCore forward hooks before run",our_hooks_total()==0);chk("frozen eval model before run",(not model.training)and trainable_tensors()==0 and not lora_present())
        chk("SDPA attention before run",getattr(cfg,"_attn_implementation",None)=="sdpa");chk("no optimizer before run",not optimizer_present())
        fp_pre=fingerprint();sent_pre=strong_sentinel();chk("pre-run weight fingerprint",fp_pre==FP0);chk("pre-run sampled SHA-256 sentinel",sent_pre==SENTINEL0)
        run_start_utc=utc_now();run_start_local=local_now();torch.cuda.synchronize();T0=time.perf_counter()
        stage_name="2/7 codebook"
        yield pack(stage(2,"Building the cartridge codebook","32 neutral sentences → PCA per layer × KV head (224 SVDs). The codebook knows nothing about the cartridge facts."))
        cbi=build_codebook();mem_after_cb=gmem()
        stage_name="3/7 loading cartridges";cards=[];BANKKV=[];tc=time.perf_counter()
        for i,b in enumerate(BANK_SPEC,1):
            yield pack(stage(3,f"Loading cartridge {i:02d}/16",f"<b>{b['cid']}</b> · {html.escape(b['role'])} · {b['fam']}"+bar_html(i-1,16)))
            torch.cuda.synchronize();t1=time.perf_counter();K,V,tel=packet(b["text"]);kv=install(K,V);torch.cuda.synchronize();es=time.perf_counter()-t1
            chk(f"{b['cid']} finite K/V",all(bool(torch.isfinite(t).all()) for t in list(kv[0])+list(kv[1])))
            BANKKV.append(kv);cards.append(dict(b,**tel,encode_seconds=es,gpu_mib=gmem()));del K,V
        torch.cuda.synchronize();cart_s=time.perf_counter()-tc
        stage_name="4/7 source removed"
        yield pack(stage(4,"Source removed","The engine now holds only installed cartridge numbers. Readout prompts are checked against every source sentence."+bar_html(16,16)))
        SOURCES=[b["text"] for b in BANK_SPEC];calls=[]
        def ibr(q,tag):
            out=batch(q,BANKKV);pr=STAT["last_prompt"];hits=sum(s in pr for s in SOURCES)
            calls.append(dict(tag=tag,prompt_tokens=int(STAT["last_prompt_tokens"]),source_hits=hits,slots=int(sum(kv[2] for kv in BANKKV)),seconds=float(STAT["last_seconds"]),steps=int(STAT["last_steps"])))
            if hits:raise RuntimeError("SOURCE LEAK INTO READOUT PROMPT")
            return out,pr
        stage_name="5/7 question battery";tb=time.perf_counter();BT=[]
        for i,q in enumerate(BATTERY,1):
            yield pack(stage(5,f"Reading the cartridge bank · question {i:02d}/16",html.escape(q["label"])+bar_html(i-1,16,"#047857")))
            t1=time.perf_counter();s1=None;s2=None;cid=q["cid"];lane=[0]*16
            if q["obj"]:
                t_,pr=ibr(MW1.format(obj=q["obj"]),f"{q['qid']}·S1");s1=dict(prompt=pr,texts=t_);cid=decide_id(t_)
                for j,t in enumerate(t_):
                    if parse_id(t):lane[j]=1
            place=None
            if cid:
                t_,pr=ibr(MW2.format(cid=cid),f"{q['qid']}·S2");s2=dict(prompt=pr,texts=t_);place=decide_place(t_)
                for j,t in enumerate(t_):
                    if parse_place(t,KNOWN_PLACES):lane[j]=2
            if q["exp_place"]:ok=(place==q["exp_place"]) and (q["obj"] is None or cid==q["exp_id"]);status="CORRECT" if ok else "WRONG"
            else:ok=(place is None) and (q["obj"] is None or cid==q["exp_id"]);status="UNKNOWN ✓" if ok else ("FALSE LINK" if place else "MISSED")
            BT.append(dict(q,stage1=s1,stage2=s2,obs_id=cid,obs_place=place,observed=place or ("UNKNOWN" if cid or q["obj"] is None else "no ID"),
                           expected=q["exp_place"] or "UNKNOWN",ok=bool(ok),false_link=bool(q["exp_place"] is None and place is not None),status=status,lane=lane,seconds=time.perf_counter()-t1))
            print(f" {q['qid']:<3} {q['kind']:<13} exp={str(q['exp_place'] or 'UNKNOWN'):<20} cid={cid} place={place} -> {status}",flush=True)
        torch.cuda.synchronize();bat_s=time.perf_counter()-tb;mem_after_bt=gmem()
        stage_name="6/7 control + post-run integrity"
        yield pack(stage(6,"No-memory control","The same linked questions with an empty (PAD-only) cache. Strict hits must be zero."))
        tcn=time.perf_counter();nk=nomem_kv();nrows=[]
        for q in [x for x in BATTERY if x["kind"]=="LINKED"]:
            a1=batch(MW1.format(obj=q["obj"]),[nk])[0];c1=parse_id(a1);a2=batch(MW2.format(cid=c1),[nk])[0] if c1 else "";pl=parse_place(a2,KNOWN_PLACES) if c1 else None
            nrows.append(dict(qid=q["qid"],stage1=a1,stage2=a2,cid=c1,place=pl,strict_hit=int(c1==q["exp_id"] and pl==q["exp_place"])))
        torch.cuda.synchronize();ctl_s=time.perf_counter()-tcn;engine_s=time.perf_counter()-T0
        fp_post=fingerprint();sent_post=strong_sentinel();ours=our_hooks_total()
        chk("weight fingerprint after run",fp_post==FP0);chk("sampled SHA-256 sentinel after run",sent_post==SENTINEL0);chk("AkbasCore hooks after run = 0",ours==0)
        chk("model.training == False",not model.training);chk("trainable tensors == 0",trainable_tensors()==0);chk("LoRA none",not lora_present());chk("optimizer none",not optimizer_present())
        chk("16/16 questions answered",len(BT)==16);chk("source sentences in readout prompts = 0",sum(c["source_hits"] for c in calls)==0)
        # ---------------- summaries ----------------
        kinds=["LINKED","MISSING LINK","DIRECT","ABSENT ID"]
        by_kind={k_:dict(ok=sum(q["ok"] for q in BT if q["kind"]==k_),n=sum(q["kind"]==k_ for q in BT)) for k_ in kinds}
        by_family={f:dict(ok=sum(q["ok"] for q in BT if q["fam"]==f and q["kind"] in("LINKED","DIRECT")),n=sum(q["fam"]==f and q["kind"] in("LINKED","DIRECT") for q in BT)) for f in["F1","F2","F3","F4"]}
        live=dict(n=len(BT),correct=sum(q["ok"] for q in BT),unknown_correct=sum(q["ok"] for q in BT if q["exp_place"] is None),false_links=sum(q["false_link"] for q in BT),by_kind=by_kind,by_family=by_family)
        nb=dict(slots=sum(c["T"] for c in cards),code_numbers=sum(c["code_numbers"] for c in cards),own_numbers=sum(c["own_numbers"] for c in cards),
                native_numbers=sum(c["native_numbers"] for c in cards),hidden_copy_numbers=sum(c["hidden_copy_numbers"] for c in cards),runtime_numbers=sum(c["native_numbers"] for c in cards))
        nb["ratio_vs_native"]=(nb["code_numbers"]+nb["own_numbers"])/nb["native_numbers"];nb["ratio_vs_hidden"]=(nb["code_numbers"]+nb["own_numbers"])/nb["hidden_copy_numbers"]
        num=lambda v,l:dict(value=int(v),label=l,human=human(v))
        tot_flop=2*(CNT["linear_mac"]+CNT["attn_mac"]+CNT["reproj_mac"]+CNT["cart_encdec_mac"])
        W=dict(measured=dict(forward_passes=num(CNT["forward_passes"],"transformer forward passes"),forge_passes=num(CNT["forge_passes"],"forge passes (codebook + cartridges + control)"),
                             ibr_calls=num(CNT["ibr_calls"],"IBR batched calls"),ibr_decode_steps=num(CNT["ibr_decode_steps"],"IBR decode steps"),
                             forward_tokens=num(CNT["forward_tokens"],"token positions processed"),generated_tokens=num(CNT["generated_tokens"],"tokens generated (all rows)"),
                             svd=num(CNT["svd_count"],"SVD decompositions (codebook)")),
               derived=dict(cart_encdec_mac=num(CNT["cart_encdec_mac"],"cartridge encode+decode multiply-adds"),code_numbers=num(nb["code_numbers"],"numbers stored in the 16 cartridges"),
                            numbers_installed=num(CNT["numbers_installed"],"numbers installed into the cache"),memory_slot_reads=num(CNT["memory_slot_reads"],"query × cartridge-slot reads (heads × layers)"),
                            frozen_params=num(N_PARAMS,"frozen model parameters")),
               estimated=dict(linear_flop=num(2*CNT["linear_mac"],"linear-layer FLOPs (est.)"),attn_flop=num(2*CNT["attn_mac"],"attention FLOPs (est.)"),total_flop=num(tot_flop,"total GPU FLOPs (est.)")),
               measured_order=["forward_passes","ibr_calls","ibr_decode_steps","forward_tokens","generated_tokens","svd"],derived_order=["cart_encdec_mac","code_numbers","numbers_installed","memory_slot_reads","frozen_params"],
               estimated_order=["linear_flop","attn_flop","total_flop"],
               formula_note=(f"DERIVED cartridge MACs = Σ 2 × 28 × (T−1) × 4 × 128 × (120+128). ESTIMATED: linear = {LIN_PARAMS:,} nn.Linear weights × token positions (incl. lm_head on every position); "
                             f"attention = 2 × {NH} heads × {HD} × 28 layers × Σ attended length (QKᵀ + AV, padded length as computed). FLOPs = 2 × MACs (1 MAC = 2 FLOP). "
                             "Excludes norms, RoPE, softmax and element-wise ops. Measured counters come from the code paths that ran."))
        src_rm=dict(source_sentences=16,source_sentence_hits=sum(c["source_hits"] for c in calls),calls=calls,
                    checks=["source sentence not contained in any readout prompt (asserted per IBR call)","forge sees source text only (no question)",
                            "codebook built from 32 neutral sentences disjoint from cartridges","readout prompt = question frame only; knowledge enters as installed K/V rows",
                            "NOMEM control uses a PAD-only cache with no source-derived information"])
        P={"schema":"akbascore.cartridge.run.v1","project":"AkbasCore","demo":"NIRVANA COGNITIVE CARTRIDGE","run_id":run_id,"run_start_utc":run_start_utc,"run_start_local":run_start_local,
           "run_end_utc":utc_now(),"run_end_local":local_now(),"model":{"id":MODEL_ID,"dtype":"bfloat16","attn_implementation":"sdpa","layers":NL,"hidden_size":H,"q_heads":NH,"kv_heads":NKV,"head_dim":HD,"params":N_PARAMS,"trainable_params":N_TRAINABLE,"linear_params":LIN_PARAMS},
           "environment":{"gpu":GPU_NAME,"gpu_total_gib":GPU_TOTAL/2**30,"cuda":torch.version.cuda,"torch":torch.__version__,"transformers":transformers.__version__,"gradio":gr.__version__,"numpy":np.__version__,"matplotlib":matplotlib.__version__,"python":sys.version,"platform":platform.platform()},
           "engine":{"source":"TEST461 (463.final.py) IBR engine, unchanged","lock_sha256":LOCK_SHA,"checks":[n for n,_ in ENGINE_LOCK_CHECKS],"K":K_DIM,"V":V_DIM,"own_slot0":True,"compute_path":"PyTorch CUDA backend (cuBLAS, SDPA); no custom CUDA/C++ kernel",
                     "decide_rules":"decide_id/decide_place: exactly one distinct parsed ID/place across rows else UNKNOWN; parse_place matches the bank's known places","MW1":MW1,"MW2":MW2,"FMT":FMT},
           "codebook":cbi,"cartridges":cards,"bank":nb,"battery":BT,"controls":{"nomem":dict(rows=nrows,strict_hits=sum(r["strict_hit"] for r in nrows),n=len(nrows))},
           "live_summary":live,"source_removal":src_rm,"workload":W,"counters_raw":dict(CNT),
           "gpu":{"name":GPU_NAME,"peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,"after_codebook_mib":mem_after_cb,"after_battery_mib":mem_after_bt},
           "ui":{"stages":7},"errors":errors,
           "timing":{"startup":{"model_load_seconds":MODEL_LOAD_S},"run":{"codebook_seconds":cbi["seconds"],"cartridges_seconds":cart_s,"battery_seconds":bat_s,"control_seconds":ctl_s,"engine_seconds":engine_s}},
           "integrity":{"result":"PASS","fingerprint_tensors":FP_NAMES,"fingerprint_startup":list(FP0),"fingerprint_pre_run":list(fp_pre),"fingerprint_after":list(fp_post),
                        "sentinel_method":"sampled SHA-256 over 16 contiguous 256-value slices per selected tensor; not a full verification of every weight",
                        "sentinel_startup":SENTINEL0,"sentinel_pre_run":sent_pre,"sentinel_after":sent_post,"trainable_parameter_tensors":trainable_tensors(),
                        "model_training_mode":bool(model.training),"akbascore_hooks_after_run":int(ours),"framework_hooks_after_run":framework_hooks(),"pre_seal_checks":list(checks),
                        "checks_total":len(checks),"optimizer":None,"lora":False,"fine_tuning":False},
           "historical_sealed_record":{"note":"Historical results from TEST460/461/462/464. NOT produced by this run.","data":HIST},
           "decoding":{"mode":"greedy argmax inside the IBR loop","max_new_tokens":MAX_NEW,"eos_token_id":EOS,"pad_token_id":PAD,"model_generation_config":gen_config()},
           "reproduction":{"seed":SEED,"cell_source_sha256":CELL_SOURCE_SHA,"cell_source":CELL_SOURCE}}
        stage_name="7/7 sealing"
        yield pack(stage(7,"Sealing evidence + rendering JPEGs","Canonical payload, SHA-256 artifact seal, readable log, manifest, then the poster story."))
        ts=time.perf_counter();P=jsafe(P);pb=canon(P);sha=hashlib.sha256(pb).hexdigest()
        payload_name=f"{run_id}_FULL_RUN_LOG_payload.json";manifest_name=f"run_manifest_{run_id}.json";pp=run_dir/payload_name;pp.write_bytes(pb)
        if hashlib.sha256(pp.read_bytes()).hexdigest()!=sha:raise RuntimeError("Payload hash verification failed.")
        t_sealed=time.perf_counter();sealed_utc=utc_now();seal={"sealed_utc":sealed_utc,"sealing_stage_seconds":t_sealed-ts,"seal_wall_seconds":t_sealed-T0}
        manifest={"demo":"AKBASCORE NIRVANA COGNITIVE CARTRIDGE","run_id":run_id,"payload_file":payload_name,"payload_sha256":sha,"payload_bytes":len(pb),"generated_utc":sealed_utc,
                  "timing":{"first_step_to_payload_sealed_seconds":seal["seal_wall_seconds"],"sealing_stage_seconds":seal["sealing_stage_seconds"]},"hash_algorithm":"SHA-256",
                  "canonicalization":"json.dumps(sort_keys=True, separators=(',',':'), ensure_ascii=False, allow_nan=False), UTF-8",
                  "note":"Artifact integrity seal: confirms the payload has not changed after sealing. It does not by itself establish any scientific interpretation."}
        mp=run_dir/manifest_name;mp.write_bytes(json.dumps(manifest,sort_keys=True,indent=2,ensure_ascii=False).encode("utf-8"))
        tp=run_dir/f"{run_id}_READABLE_RUN_LOG.txt";tp.write_text(make_txt(P,sha,payload_name,manifest_name,seal),encoding="utf-8")
        print(f"payload sealed: {sha}")
        ctx={"P":P,"run_id":run_id,"sha":sha,"N":NPOST,"payload_name":payload_name,**seal}
        rt=time.perf_counter();imgs=render_all(ctx,run_dir);render_s=time.perf_counter()-rt
        entries=[]
        for i,(p,cap)in enumerate(imgs,1):
            with Image.open(p)as im:w,h=im.size;fmt=im.format;mode=im.mode
            entries.append({"index":i,"filename":p.name,"title":cap,"width":w,"height":h,"format":fmt,"mode":mode,"size_bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"run_id":run_id})
        imf=run_dir/f"images_manifest_{run_id}.json"
        imf.write_bytes(json.dumps({"demo":"AKBASCORE NIRVANA COGNITIVE CARTRIDGE","run_id":run_id,"payload_file":payload_name,"payload_sha256":sha,"generated_utc":utc_now(),"render_seconds":render_s,
                                    "count":len(entries),"images":entries,"note":"SHA-256 per image is an artifact integrity seal for each file."},sort_keys=True,indent=2,ensure_ascii=False).encode("utf-8"))
        zp=run_dir/f"AKBASCORE_CARTRIDGE_JPGs_{run_id}.zip";expected=[p.name for p,_ in imgs]+[imf.name,mp.name]
        with zipfile.ZipFile(zp,"w",compression=zipfile.ZIP_DEFLATED)as z:
            for p,_ in imgs:z.write(p,arcname=p.name)
            z.write(imf,arcname=imf.name);z.write(mp,arcname=mp.name)
        audit=post_seal_audit(imgs,zp,expected,pp,sha,tp,mp);total=len(checks)+len(audit)
        print(f"{run_id}: {len(checks)} pre-seal + {len(audit)} post-seal checks = {total}/{total} PASS | render {render_s:.1f}s")
        gallery_items=[(str(p),c_)for p,c_ in imgs]
        raw_texts={"txt":tp.read_text(encoding="utf-8"),"json":pp.read_bytes().decode("utf-8"),"man":mp.read_text(encoding="utf-8")}
        L_=live["by_kind"]
        body=(f"<b>{html.escape(run_id)}</b><br>THIS LIVE RUN: {live['correct']}/{live['n']} questions correct · linked {L_['LINKED']['ok']}/4 · missing link → UNKNOWN {L_['MISSING LINK']['ok']}/4 · "
              f"direct {L_['DIRECT']['ok']}/4 · absent ID → UNKNOWN {L_['ABSENT ID']['ok']}/4 · false links {live['false_links']} · NOMEM strict hits {P['controls']['nomem']['strict_hits']}/4<br>"
              f"{NPOST} JPGs · {total}/{total} checks PASS<br>SEALED RECORD (historical): TEST461 final held-out 43/48 linked · 48/48 UNKNOWN · verdict FAIL_FINAL_HELDOUT (F2 gate)<br>"
              f'<span class="mono">payload SHA-256 (artifact integrity seal): {sha}</span>')
        yield pack(card_html("7/7 · Complete",body,"on"),gallery_items,{"json":pp,"txt":tp,"man":mp},[p for p,_ in imgs],raw_texts)
    except Exception as ex:
        print("="*110);print(f"CARTRIDGE RUN FAILED — stage: {stage_name}");print(f"exception type : {type(ex).__name__}");print(f"exception message: {ex}");traceback.print_exc();print("="*110)
        tip="<br><b>CUDA error: Runtime → Restart runtime, then run the cell again.</b>" if is_cuda_error(ex)else "<br>The full traceback is printed in the Colab console."
        yield pack(card_html("Run failed",f"Stage: {html.escape(stage_name)}<br>{html.escape(type(ex).__name__)}: {html.escape(str(ex)[:600])}{tip}","err"),[],None,None,None)
    finally:
        plt.close("all");GPU_LOCK.release()
        try:torch.cuda.empty_cache()
        except Exception:pass
# ---------------------------------------------------------------- SERVICE SELF-TEST (before the link is printed) ----------------------------------------------------------------
print("[4/7] SERVICE SELF-TEST")
def service_selftest():
    out=[];d=ROOT/"_selftest";shutil.rmtree(d,ignore_errors=True);d.mkdir(parents=True)
    (d/"w.txt").write_text("ok");out.append("ROOT writable")
    ctx={"P":selftest_payload(),"run_id":"SELFTEST","sha":"0"*64,"N":NPOST,"payload_name":"selftest.json","sealing_stage_seconds":0.1}
    imgs=render_all(ctx,d)
    for p,_ in imgs:
        with Image.open(p) as im:
            if im.format!="JPEG" or im.mode!="RGB":raise RuntimeError(f"selftest JPEG invalid: {p.name}")
    out.append(f"{len(imgs)}/{NPOST} posters rendered as JPEG/RGB with a synthetic payload")
    zp=d/"t.zip"
    with zipfile.ZipFile(zp,"w")as z:
        for p,_ in imgs:z.write(p,arcname=p.name)
    with zipfile.ZipFile(zp)as z:
        if z.testzip()is not None:raise RuntimeError("selftest zip failed")
    out.append("ZIP testzip()")
    if len(pack(READY_HTML))!=N_OUTPUTS or len(pack(READY_HTML,[],None,None,None))!=N_OUTPUTS:raise RuntimeError("callback output count mismatch")
    out.append(f"callback output count = {N_OUTPUTS}")
    if not FILE_URL_PREFIX.endswith("/file="):raise RuntimeError("file URL prefix")
    out.append(f"file route {FILE_URL_PREFIX}")
    shutil.rmtree(d,ignore_errors=True);return out
SERVICE_CHECKS=service_selftest()
for c in SERVICE_CHECKS:print(" PASS ·",c)
# ---------------------------------------------------------------- INTERFACE ----------------------------------------------------------------
print("[5/7] INTERFACE")
CSS="""
:root{--kz-on:#047857;--kz-fg:#0f172a;--kz-card:#ffffff;--kz-bd:#cbd5e1;--kz-a:#0369a1;--kz-off:#6d28d9;--kz-err:#b91c1c;--kz-warn:#b45309}
.dark{--kz-on:#34d399;--kz-fg:#f8fafc;--kz-card:#0f172a;--kz-bd:#475569;--kz-a:#38bdf8;--kz-off:#c4b5fd;--kz-err:#f87171;--kz-warn:#fbbf24}
html,body{overflow-x:hidden!important}
.gradio-container{width:100%!important;max-width:860px!important;margin:0 auto!important;padding-left:10px!important;padding-right:10px!important;box-sizing:border-box!important}
.gradio-container *{box-sizing:border-box}
.kz{color:var(--kz-fg)!important;max-width:100%;overflow-wrap:anywhere;line-height:1.45}
.kz *{color:inherit}
.hero{text-align:center;padding:10px 2px 2px}
.brand{font-size:clamp(26px,8vw,40px);font-weight:800;letter-spacing:1px;line-height:1.05}
.sub{font-size:clamp(14px,4.2vw,18px);font-weight:700;color:var(--kz-warn)!important;margin-top:4px}
.tag{font-size:clamp(13px,3.8vw,15px);opacity:.9;margin-top:6px}
.card{background:var(--kz-card);border:2px solid var(--kz-bd);border-radius:12px;padding:12px 14px;margin:6px 0}
.card.on{border-color:var(--kz-on)}.card.err{border-color:var(--kz-err)}.card.warn{border-color:var(--kz-warn)}.card.info{border-color:var(--kz-a)}
.card .h{font-weight:800;font-size:16px;margin-bottom:4px}
.small{font-size:14px}.mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;word-break:break-all}
#kz_run button,#kz_run{font-size:clamp(17px,5vw,22px)!important;font-weight:800!important;min-height:64px!important}
"""
DL_ALL_JS="""(u)=>{let items=[];try{items=JSON.parse(u||"[]")}catch(e){items=[]}
let i=0;const next=()=>{if(i>=items.length)return;const it=items[i++];const a=document.createElement('a');a.href=it.url;a.download=it.name;a.rel='noopener';
document.body.appendChild(a);a.click();setTimeout(()=>{a.remove();next()},900)};next();return u;}"""
HERO=('<div class="kz hero"><div class="brand">AKBASCORE NIRVANA</div><div class="sub">COGNITIVE CARTRIDGE</div>'
      '<div class="tag">Knowledge goes in. The source goes away. The memory remains.<br>'
      '16 synthetic knowledge cartridges are installed into a frozen Qwen2.5-7B-Instruct. The source text is removed. 16 automatic questions are answered from the cartridges — or answered UNKNOWN.</div></div>')
def _tb(**kw):
    try:return gr.Textbox(show_copy_button=True,**kw)
    except TypeError:return gr.Textbox(**kw)
def _gallery(**kw):
    try:return gr.Gallery(columns=1,height="auto",object_fit="contain",preview=False,**kw)
    except TypeError:return gr.Gallery(**kw)
try:_blocks=gr.Blocks(css=CSS,title="AKBASCORE NIRVANA · COGNITIVE CARTRIDGE")
except TypeError:_blocks=gr.Blocks(title="AKBASCORE NIRVANA · COGNITIVE CARTRIDGE")
with _blocks as demo:
    gr.HTML(HERO)
    run_btn=gr.Button("RUN COGNITIVE CARTRIDGE DEMO",variant="primary",elem_id="kz_run")
    status=gr.HTML(READY_HTML)
    gallery=_gallery(label=f"{NPOST} JPG posters (tap to open)",show_label=True)
    dl_all=gr.Button(DL_ALL_LABEL,interactive=False)
    jpg_urls=gr.Textbox(value="",visible=False)
    with gr.Row():
        dlb=[gr.DownloadButton(label=l,value=None,interactive=False) for l in DL_LABELS]
    with gr.Accordion("Direct file links (fallback)",open=False):
        fls=[gr.File(label=n,interactive=False) for n in("FULL RUN LOG (.json)","READABLE RUN LOG (.txt)","RUN MANIFEST (.json)")]
    with gr.Tabs():
        with gr.Tab("RAW LOG / HAM LOG (.txt)"):raw_txt=_tb(lines=18,max_lines=40,label="readable run log")
        with gr.Tab("FULL RUN LOG (.json)"):raw_json=_tb(lines=18,max_lines=40,label="sealed canonical payload")
        with gr.Tab("MANIFEST (.json)"):raw_man=_tb(lines=12,max_lines=30,label="run manifest")
    OUTS=[status,gallery,dl_all,jpg_urls]+dlb+fls+[raw_txt,raw_json,raw_man]
    if len(OUTS)!=N_OUTPUTS:raise RuntimeError(f"UI outputs {len(OUTS)} != {N_OUTPUTS}")
    run_btn.click(run_handler,inputs=None,outputs=OUTS)
    try:dl_all.click(None,inputs=[jpg_urls],outputs=None,js=DL_ALL_JS)
    except TypeError:dl_all.click(None,inputs=[jpg_urls],outputs=None,_js=DL_ALL_JS)
try:demo.queue(default_concurrency_limit=1)
except TypeError:demo.queue(concurrency_count=1)
print("[6/7] LAUNCH (public share link)")
print("[7/7] Open the printed gradio.live link in a new tab and press RUN COGNITIVE CARTRIDGE DEMO.")
demo.launch(share=True,allowed_paths=ALLOWED_PATHS,show_error=True,inline=False)
