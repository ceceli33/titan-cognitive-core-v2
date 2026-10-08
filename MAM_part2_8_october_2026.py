# =====================================================================================================================================
# AKBASCORE MAM · PERSISTENT NUMERICAL MEMORY FOR FROZEN LANGUAGE MODELS — PART 2 / 3 · MEASUREMENT PIPELINE AND FIGURES
# Same demo as PART 1. Run this cell after PART 1 / 3, in the same Colab runtime; then run PART 3 / 3.
# Defines the fail-closed measurement run (write → append-only consolidation → source-free question → verification → sealed payload,
# logs, ZIP), six figures (JPEG 300 dpi + vector PDF) and the optional live replay of the locked panel. Nothing here changes the engine.
# Discovered and developed by Mustafa Akbaş · AkbasCore AI Teknoloji · 8 October 2026 · AKBASCORE RESEARCH SOFTWARE LICENSE
# =====================================================================================================================================
if not globals().get("MAM_DC6_PART1_OK"):raise RuntimeError("PART 1 / 3 has not been run in this runtime. Run PART 1, then PART 2, then PART 3.")
#<<CORE_BEGIN>>
import unicodedata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.patches import Rectangle,FancyArrowPatch,Patch
from matplotlib.lines import Line2D
from PIL import Image
ROOT=Path("/content/AKBASCORE_MAM_DC6")if os.path.isdir("/content")else Path("/tmp/AKBASCORE_MAM_DC6")
ROOT.mkdir(parents=True,exist_ok=True);ALLOWED_PATHS=[str(ROOT)]
N_STAGES=8
def ev(stage,title,body):return dict(stage=stage,title=title,body=body)
def file_ready(p):return p is not None and Path(p).is_file()and Path(p).stat().st_size>0
def prune_runs(keep=4,prefix="MAM560DEMO-"):
    runs=sorted([p for p in ROOT.glob(prefix+"*")if p.is_dir()],key=lambda p:p.stat().st_mtime)
    for p in runs[:max(0,len(runs)-keep)]:shutil.rmtree(p,ignore_errors=True)
def segments(steps,p):
    """Row ranges of the consolidated memory: shared prefix (written with cartridge 1) and the body rows of cartridges 1..5."""
    seg=[("prefix",0,p,None)];prev=p
    for n_,s in enumerate(steps,1):seg.append((f"C{n_}",prev,s["new_len"],s["cartridge_id"]));prev=s["new_len"]
    return seg
def post_seal_audit(imgs,pdfs,zp,member_names,pp,sha,tp,mp,verdict):
    checks=[]
    def ok(name,cond):
        checks.append(name)
        if not cond:raise AuditFail("POST-SEAL AUDIT FAILED: "+name)
    ok(f"{N_FIGS}/{N_FIGS} figures created (JPEG)",len(imgs)==N_FIGS and all(file_ready(p)for p,_ in imgs));ok(f"{N_FIGS}/{N_FIGS} vector PDF copies",len(pdfs)==N_FIGS and all(file_ready(p)for p in pdfs))
    bad=[]
    for p,_ in imgs:
        with Image.open(p)as im:
            if not(im.format=="JPEG"and im.mode=="RGB"and im.size[0]>=3600):bad.append(p.name)
    ok(f"{N_FIGS}/{N_FIGS} figures are JPEG/RGB at ≥3600 px width",not bad)
    with zipfile.ZipFile(zp)as z:
        ok("ZIP testzip()",z.testzip()is None);nm=z.namelist()
        ok("ZIP is flat (no sub-folders)",all("/"not in n for n in nm));ok("ZIP contents = expected package",sorted(nm)==sorted(member_names))
        jp=sorted(n for n in nm if n.lower().endswith(".jpg"));ok("figure numbering 01..%02d"%N_FIGS,[n[4:6]for n in jp]==[f"{i:02d}"for i in range(1,N_FIGS+1)])
    ok("payload SHA-256 recomputation",hashlib.sha256(pp.read_bytes()).hexdigest()==sha);ok("payload JSON parses",isinstance(json.loads(pp.read_bytes().decode("utf-8")),dict))
    m=json.loads(mp.read_text(encoding="utf-8"));ok("manifest hash and verdict match payload",m["payload_sha256"]==sha and m["verdict"]==verdict)
    ok("readable log exists and is non-empty",file_ready(tp));ok("ZIP exists and is non-empty",file_ready(zp))
    return checks
def fw_txt(f):
    if f["kind"]=="ids":return f"token ids n={f['n']}"+(f" · cache {f['past_len']}"if f["has_past"]else" · no cache")
    return f"embeddings n={f['n']} ({'all zero'if f['embeds_zero']else'NON-ZERO'})"+(f" · cache {f['past_len']}"if f["has_past"]else" · no cache")
def make_txt(P,sha,names,sealed_utc):
    o=[];a=o.append;S="="*140;Dd="-"*140;E_=P["environment"];L=P["live"];cfg=P["engine"]["config"];fz=P["frozen"];ans=L["answer"]
    a(S);a(f"{DEMO_TITLE} — READABLE RUN LOG");a(f"{DISCOVERY} · {ORG} · {DATE_TXT} · {AUTHOR_PLACE}");a(f"{LICENSE_NAME} · {COPYRIGHT} · {LICENSE_URL}");a(S)
    a(PRIORITY);a(f"{PARADIGM}.")
    a(f"Derived from {names['payload']} (SHA-256 {sha}). Manifest: {names['manifest']}.")
    a("This is a LIVE run of the frozen TEST560 incremental-DC6 engine. It is not TEST560; the sealed TEST560 values are reproduced verbatim, never recomputed.")
    for k_,v in(("RUN ID",P["run_id"]),("START UTC",P["run_start_utc"]),("END UTC",P["run_end_utc"]),("SEALED UTC",sealed_utc),("VERDICT",P["verdict"]),
        ("MODEL",f"{cfg['MODEL_ID']} · frozen · {P['model']['dtype']} · attention {P['model']['attn']}"),("ARCHITECTURE","32 layers · hidden 4096 · 32 Q heads · 8 KV heads · head 128"),
        ("GPU",f"{E_['gpu']} (TEST560 reference GPU: {CANON_GPU}; same: {P['hardware']['same']})"),("TORCH / TRANSFORMERS / GRADIO",f"{E_['torch']} / {E_['transformers']} / {E_['gradio']}"),
        ("ENGINE FILE",f"{P['engine']['file']} · SHA-256 {P['engine']['sha256']}"),("CONSOLIDATION CUT",f"layer {cfg['CUT']} (cartridge = H{cfg['CUT']} residual + layers 0–{cfg['CUT']} K/V)"),
        ("PANEL",f"locked TEST560 panel · SHA-256 {cfg['panel_sha']}"),("TEST560 LOCK",TEST560["lock"]),("TEST560 RESULT SHA",TEST560["result_sha"])):a(f"{k_:<30}: {v}")
    a("");a("CASE");a(Dd)
    a(f"case {L['case']:02d} · target position {L['position']} (slot {L['target_slot']+1} of 5) · question: {L['question']}");a(f"expected answer (audit metadata): {L['expected']}")
    for n_,c in enumerate(L["cartridges"],1):a(f"C{n_} · cartridge #{c['id']:02d} · {c['type']:<8} · {c['body_len']:>2} body tokens · {c['fact']}"+("   ← TARGET"if c["is_target"]else""))
    a("");a("WRITE + APPEND-ONLY CONSOLIDATION · load_case()  [LIVE]");a(Dd)
    for n_,s in enumerate(L["steps"],1):
        a(f"step {n_} · {s['kind']:<6} · cartridge #{s['cartridge_id']:02d} · memory {s['old_len']} → {s['new_len']} tokens (+{s['new_len']-s['old_len']}) · {1000*s['seconds']:.1f} ms")
        for f in s["forwards"]:a(f"         forward: {fw_txt(f)}")
        if s["kind"]=="append":a(f"         previously consolidated rows 0…{s['old_len']-1}: bitwise unchanged = {s['prior_rows_bitwise_unchanged']} · SHA {s['prior_sha_before'][:16]}… → {s['prior_sha_after'][:16]}…")
    a(f"final memory: {L['memory']['T']} tokens × {L['memory']['layers']} layers · {L['memory']['kv_heads']} KV heads × {L['memory']['head_dim']} · {L['memory']['bytes']/2**20:.2f} MiB ({L['memory']['dtype']}) · SHA {L['memory_sha'][:24]}…")
    a(f"source token-id forwards during load: {L['source_reads']} (one per cartridge, each its own fact) · consolidation forwards with all-zero input embeddings: {L['zero_embed_forwards']}")
    a("");a("SOURCE-FREE QUESTION · ask()  [LIVE]");a(Dd)
    a(f"question: {ans['question']}");a(f"answer  : {ans['answer']}");a(f"expected: {L['expected']} · {'CORRECT'if L['correct']else'INCORRECT'} (re-derived: {L['correct_rederived']})")
    a(f"query input = question template only: {L['query_tokens']} token ids over the installed memory ({L['query_past']} tokens) · generation forwards {L['gen_forwards']} · {1000*ans['elapsed_seconds']:.0f} ms")
    a(f"repeat ask(): identical answer = {L['repeat_identical']} · memory SHA unchanged by querying = {L['memory_unchanged_by_query']}")
    if L.get("custom"):cu=L["custom"];a(f"free-text question (unscored): {cu['question']} → {cu['answer']} · query tokens {cu['query_tokens']}")
    a("");a("SEALED TEST560 REFERENCE (archived; not produced by this run)");a(Dd)
    for arm in("INCR_DC6","BATCH_DC6","JOINT","INDEP"):v=TEST560["arms"][arm];a(f"{arm:<10} FIRST {v['FIRST']:>2}/24 · MIDDLE {v['MIDDLE']:>2}/24 · LAST {v['LAST']:>2}/24 · TOTAL {v['TOTAL']}/72 · {ARM_DESC[arm]}")
    a(f"incremental DC6 vs independent: gain {TEST560['incr_gain']} · loss {TEST560['incr_loss']} · answer identity with batch {TEST560['incr_answer_identity_batch'][0]}/{TEST560['incr_answer_identity_batch'][1]}")
    a(f"scale boundary ({SCALE_BOUNDARY['name']}, {SCALE_BOUNDARY['note']}): "+" · ".join(f"{n}: {k}/{n}"for n,k in SCALE_BOUNDARY["rows"]))
    a("");a("FROZEN MODEL");a(Dd)
    for k_ in("sentinel_startup","sentinel_before","sentinel_after","guard_startup","guard_before","guard_after"):a(f"{k_:<18}: {fz[k_]}")
    a(f"trainable tensors={fz['trainable_tensors']} · lora={fz['lora']} · optimizer={fz['optimizer']} · training_mode={fz['training_mode']} · foreign hooks outside API calls {fz['foreign_hooks_before']}→{fz['foreign_hooks_after']}")
    a(fz["guard_method"]);a(fz["sentinel_method"]);a(P["instrumentation"]["forward_counter"]);a(P["instrumentation"]["observer"])
    for t_,k_ in(("DEMONSTRATED IN THIS RUN","demonstrated"),("ARCHIVED","archived"),("NOT ESTABLISHED","not_established"),("TECHNICAL NOTES","notes")):
        a("");a(t_);a(Dd)
        for t in P["scope"][k_]:a("• "+t)
    a("");a("PRE-SEAL CHECKS");a(Dd)
    for c_ in P["checks_pre_seal"]:a("PASS · "+c_)
    a("");a("TIMING (seconds)");a(Dd)
    for k_,v in P["timing"].items():a(f"{k_:<24}: {v:.3f}")
    a(f"peak GPU memory {P['gpu']['peak_allocated_gib']:.2f} GiB");a("");a(f"Raw forward records and the memory profile are in {names['payload']} and {names['jsonl']}.");a(S)
    return"\n".join(o)
def execute_run(I,ctl):
    """Fail-closed wrapper: on a technical audit failure the raw outputs gathered so far are preserved (never sealed, no figures)."""
    state={}
    try:return(yield from _execute_run(I,ctl,state))
    except AuditFail as ex:
        raw=state.get("raw")
        if raw and any(v is not None for v in raw.values()):
            try:
                p=Path(ctl["run_dir"])/f"FAILED_AUDIT_RAW_{ctl['run_id']}.json"
                p.write_bytes(canon(dict(status="FAILED AUDIT — NOT SEALED",run_id=ctl["run_id"],failed_check=str(ex),checks_passed=state.get("checks",[]),raw=raw,note="Raw outputs preserved. No figures, no seal and no verdict were produced.")));ex.partial=str(p)
            except Exception:pass
        raise
def _execute_run(I,ctl,state):
    run_id=ctl["run_id"];run_dir=Path(ctl["run_dir"]);case=int(ctl["case"]);position=str(ctl["position"]).upper();custom=(ctl.get("custom")or"").strip()
    info=I.info;checks=[];tm=Counter();T0=time.perf_counter();state["checks"]=checks;raw=dict(setup=None,trace=None,answer=None,repeat=None,custom=None);state["raw"]=raw
    def chk(name,cond):
        checks.append(name)
        if not cond:raise AuditFail(name)
    run_start_utc=utc_now();say("="*140);say(f"RUN {run_id} · case {case:02d} · target position {position}");say("="*140)
    # ---- 1 integrity ----
    yield ev(1,"Integrity","Engine source hash, locked panel, consolidation cut, weight sentinel, all-parameter guard, hooks.")
    I.reset_peak();cfg=I.config();st0=I.status();fz0=I.frozen();t=time.perf_counter();g_before=I.guard();tm["guard"]+=time.perf_counter()-t
    chk("engine source SHA-256 = supplied engine file",cfg["engine_sha256"]==ENGINE_SHA256_EXPECTED)
    chk("model, consolidation cut (layer 6) and 27-token prefix as in TEST560",cfg["MODEL_ID"]==MODEL_ID and cfg["CUT"]==CUT_EXPECTED and cfg["prefix_tokens"]==PREFIX_EXPECTED)
    chk("locked TEST560 panel (SHA-256) · 96 cartridges · 24 cases",cfg["panel_sha"]==EXPECTED_PANEL_SHA and cfg["n_cartridges"]==96 and cfg["n_cases"]==N_CASES)
    chk("weight sentinel before run = value at initialisation",fz0["sentinel"]==info["sentinel0"]);chk("all-parameter guard before run = value at startup",g_before==info["guard0"])
    chk("trainable tensors = 0, eval mode, no LoRA, no optimizer",fz0["trainable_tensors"]==0 and not fz0["training"]and not fz0["lora"]and not fz0["optimizer"])
    chk("no foreign forward/backward hooks outside API calls",fz0["foreign_hooks"]==0)
    chk("case in 0..23 and position in FIRST/MIDDLE/LAST",0<=case<N_CASES and position in POSITIONS)
    keys=I.keys(case,position);tgt=I.target(case);slot=TARGET_SLOT[position]
    chk("five cartridges; target cartridge at the selected position",len(keys)==N_CART and keys[slot]==tgt and len(set(keys))==N_CART)
    cars=[dict(I.car(ci),is_target=ci==tgt)for ci in keys];expected=I.car(tgt)["gold"];strip=I.panel_strings();state["strip"]=strip
    for c in checks:say("   PASS ·",c)
    # ---- 2 write + append-only consolidation ----
    yield ev(2,"Write + append-only consolidation","load_case(): five independent cartridges are written and appended one by one; every model forward is recorded and prior rows are compared bitwise.")
    setup,trace,dt=I.load(case,position);raw["setup"]=setup;raw["trace"]=trace;tm["load_case"]+=dt;steps=trace["steps"];P_=cfg["prefix_tokens"]
    chk("load_case(): five consolidation steps in the selected order (one init + four appends)",len(steps)==N_CART and [s["cartridge_id"]for s in steps]==keys and steps[0]["kind"]=="init"and all(s["kind"]=="append"for s in steps[1:]))
    chk("engine setup echoes the same cartridges, question and expected answer",[c["id"]for c in setup["cartridges"]]==keys and setup["expected"]==expected and setup["position"]==position and setup["case"]==case)
    s0=steps[0];f0=s0["forwards"];src0=I.source_ids(keys[0])
    chk("step 1: write = one token-id forward of cartridge 1's own fact (no cache); consolidation = one all-zero-embedding forward over the same rows",
        len(f0)==2 and f0[0]["kind"]=="ids"and f0[0]["ids"]==src0 and not f0[0]["has_past"]and f0[1]["kind"]=="embeds"and f0[1]["embeds_zero"]and f0[1]["n"]==len(src0)and not f0[1]["has_past"]and s0["new_len"]==len(src0))
    for n_,s in enumerate(steps[1:],2):
        f=s["forwards"];src=I.source_ids(s["cartridge_id"]);q=len(src)-P_
        chk(f"step {n_}: write = one token-id forward of cartridge {n_}'s own fact only (no cache)",len(f)==2 and f[0]["kind"]=="ids"and f[0]["ids"]==src and not f[0]["has_past"])
        chk(f"step {n_}: consolidation forward = all-zero embeddings for the {q} new rows only, over the existing {s['old_len']}-token memory",f[1]["kind"]=="embeds"and f[1]["embeds_zero"]and f[1]["n"]==q and f[1]["has_past"]and f[1]["past_len"]==s["old_len"]==steps[n_-2]["new_len"])
        chk(f"step {n_}: previously consolidated rows 0…{s['old_len']-1} bitwise unchanged after the append",s["prior_rows_bitwise_unchanged"]is True and s["prior_sha_before"]==s["prior_sha_after"])
        chk(f"step {n_}: memory grows by exactly the new rows (+{q})",s["new_len"]==s["old_len"]+q)
    ids_fw=[f for f in trace["forwards"]if f["kind"]=="ids"];emb_fw=[f for f in trace["forwards"]if f["kind"]=="embeds"]
    chk("no source text replayed: exactly five token-id forwards during load, each the own fact of the cartridge being written, each once",
        len(ids_fw)==N_CART and [f["ids"]for f in ids_fw]==[I.source_ids(ci)for ci in keys]and len(trace["forwards"])==2*N_CART)
    chk("engine reports: weights not updated · no source replay in consolidation · prior memory not recomputed",setup["weights_updated"]is False and setup["source_replay_during_consolidation"]is False and setup["prior_memory_recomputed"]is False)
    chk("engine event log: cache sizes match the measured memory growth",[e["cache_tokens"]for e in setup["events"]]==[s["new_len"]for s in steps])
    T=I.memory_len();chk("final memory length = prefix + all five body lengths",T==steps[-1]["new_len"]==P_+sum(c["body_len"]for c in cars))
    mem_sha=I.memory_sha_now();chk("final memory SHA-256 = SHA-256 recorded after the last append",mem_sha==steps[-1]["memory_sha"])
    prof=I.memory_profile();chk("memory profile: 32 layers × T rows · 8 KV heads × 128",prof["layers"]==32 and prof["T"]==T and prof["kv_heads"]==8 and prof["head_dim"]==128)
    say(f"   memory {T} tokens · 5 steps · prior rows bitwise unchanged at every append · {1000*dt:.0f} ms")
    # ---- 3 source-free question ----
    question=setup["question"]
    yield ev(3,"Source-free question","ask(): the question template is processed over the numerical memory; the source text is not given to the model.")
    a1,fw_a,dt=I.ask(None);raw["answer"]=a1;tm["ask"]+=dt;qids=I.query_ids(question)
    chk("query forward input = question-template tokens only, over the installed memory",len(fw_a)>=1 and fw_a[0]["kind"]=="ids"and fw_a[0]["ids"]==qids and fw_a[0]["has_past"]and fw_a[0]["past_len"]==T)
    chk("generation forwards: one new token each, over the growing cache",all(f["kind"]=="ids"and f["n"]==1 and f["has_past"]and f["past_len"]==T+len(qids)+n_ for n_,f in enumerate(fw_a[1:])))
    chk("ask(): reference question of this case; engine flags: no source text, weights unchanged",a1["question"]==question and a1["reference_question"]is True and a1["expected"]==expected and a1["source_text_supplied_to_query"]is False and a1["model_weights_changed"]is False)
    cr=hit_ans(a1["answer"],expected);chk("correctness re-derived from the answer text = engine result",cr==a1["correct"])
    a2,fw_a2,dt=I.ask(None);raw["repeat"]=a2;tm["ask_repeat"]+=dt
    chk("repeat ask(): same forward pattern and identical answer (greedy decoding)",a2["answer"]==a1["answer"]and fw_a2[0]["ids"]==qids and len(fw_a2)==len(fw_a))
    sha_q=I.memory_sha_now();chk("memory unchanged by querying (SHA-256 before = after)",sha_q==mem_sha)
    say(f"   answer {a1['answer']!r} · expected {expected} · {'CORRECT'if a1['correct']else'INCORRECT'} · {len(qids)} query tokens over {T}-token memory")
    # ---- 4 optional free-text question ----
    yield ev(4,"Free-text question","Optional, unscored: a question typed by the visitor is answered from the same numerical memory.")
    cu=None
    if custom and custom!=question:
        a3,fw_a3,dt=I.ask(custom);raw["custom"]=a3;tm["ask_custom"]+=dt;cq=I.query_ids(custom)
        chk("free-text query forward input = question-template tokens only, over the installed memory",fw_a3[0]["kind"]=="ids"and fw_a3[0]["ids"]==cq and fw_a3[0]["past_len"]==T)
        chk("free-text question is not scored",a3["reference_question"]is False and a3["correct"]is None)
        chk("memory unchanged after the free-text question",I.memory_sha_now()==mem_sha)
        cu=dict(question=custom,answer=a3["answer"],query_tokens=len(cq),gen_forwards=len(fw_a3)-1,seconds=a3["elapsed_seconds"],forwards=fw_a3)
    # ---- 5 verification ----
    yield ev(5,"Verification","Weight sentinel, all-parameter guard, trainable tensors and hooks after the run.")
    fz1=I.frozen();t=time.perf_counter();g_after=I.guard();tm["guard"]+=time.perf_counter()-t
    chk("all-parameter guard after run = before run = startup (weights unchanged)",g_after==g_before==info["guard0"]);chk("weight sentinel after run = value at initialisation",fz1["sentinel"]==info["sentinel0"])
    chk("trainable tensors = 0 and no foreign hooks after run",fz1["trainable_tensors"]==0 and fz1["foreign_hooks"]==0)
    tm["engine_total"]=time.perf_counter()-T0
    verdict=f"CASE_{case:02d}_{position}_{'CORRECT'if a1['correct']else'INCORRECT'}_APPEND_ONLY_VERIFIED_SOURCE_FREE_QUERY"
    # ---- 6 seal ----
    yield ev(6,"Sealing","Payload, manifest, readable log and raw records are written and hashed.")
    peak=I.peak_gib();same_hw=info["gpu"]==CANON_GPU
    live=dict(case=case,position=position,target_slot=slot,cartridges=cars,question=question,expected=expected,setup=setup,steps=steps,forwards_load=trace["forwards"],
              source_reads=len(ids_fw),zero_embed_forwards=sum(1 for f in emb_fw if f["embeds_zero"]),memory=prof,memory_sha=mem_sha,segments=segments(steps,P_),
              answer=a1,answer_repeat=a2,correct=bool(a1["correct"]),correct_rederived=bool(cr),repeat_identical=a2["answer"]==a1["answer"],memory_unchanged_by_query=sha_q==mem_sha,
              query_tokens=len(qids),query_token_text=I.decode_each(qids),query_past=T,gen_forwards=len(fw_a)-1,forwards_ask=fw_a,custom=cu,
              session=[dict(r,correct=(bool(a1["correct"])if r.get("correct")is None else bool(r["correct"])))for r in(ctl.get("session")or[])])
    P={"schema":"akbascore.mam.dc6.demo.run.v1","demo":DEMO_TITLE,"author":AUTHOR,"organisation":ORG,"date":DATE_ISO,"place":AUTHOR_PLACE,"copyright":COPYRIGHT,
       "license":{"name":LICENSE_NAME,"url":LICENSE_URL},"priority_record":PRIORITY,"paradigm":PARADIGM,"discovery":DISCOVERY,"core_message":CORE_MESSAGE,
       "run_id":run_id,"run_start_utc":run_start_utc,"run_end_utc":utc_now(),"verdict":verdict,
       "model":{"id":cfg["MODEL_ID"],"arch":info["arch"],"dtype":info["dtype"],"attn":info["attn"],"params":info["params"],"rope":info.get("rope")},
       "environment":{k_:info[k_]for k_ in("gpu","gpu_total_gib","torch","transformers","gradio","python","platform")},"hardware":{"reference_gpu":CANON_GPU,"this_run_gpu":info["gpu"],"same":bool(same_hw)},
       "engine":{"file":info["engine_file"],"sha256":cfg["engine_sha256"],"config":cfg,"status_before":st0,"init_seconds":info["init_seconds"],"instrument":I.kind,"source":SOURCE_URL,"log":LOG_URL},
       "instrumentation":{"forward_counter":info["forward_counter"],"observer":info["observer"]},"live":live,
       "frozen":{"sentinel_startup":info["sentinel0"],"sentinel_before":fz0["sentinel"],"sentinel_after":fz1["sentinel"],"guard_startup":info["guard0"],"guard_before":g_before,"guard_after":g_after,
                 "trainable_tensors":fz1["trainable_tensors"],"lora":fz1["lora"],"optimizer":fz1["optimizer"],"training_mode":fz1["training"],"foreign_hooks_before":fz0["foreign_hooks"],
                 "foreign_hooks_after":fz1["foreign_hooks"],"guard_method":info["guard_method"],"sentinel_method":info["sentinel_method"]},
       "archived":{"test560":TEST560,"arm_description":ARM_DESC,"scale_boundary":SCALE_BOUNDARY,"note":"Sealed reference values; not produced or recomputed by this run."},
       "scope":SCOPE,"checks_pre_seal":list(checks),"timing":dict(tm),"gpu":{"peak_allocated_gib":peak}}
    P=jsafe(P);allow=(info["gpu"],CANON_GPU)
    hits=claim_scan("payload",strip_words(json.dumps(P,ensure_ascii=False),strip),allow=allow)
    chk("claim-discipline scan of the payload: 0 hits",not hits);P["claim_scan"]={"rules":len(CLAIM_RULES)+1,"hits":0,"scope":"payload JSON text; panel names excluded"}
    pb=canon(P);sha=hashlib.sha256(pb).hexdigest();payload_name=f"{run_id}_FULL_RUN_LOG_payload.json";pp=run_dir/payload_name;t_seal=time.perf_counter();pp.write_bytes(pb);say("payload sealed:",sha)
    chk("payload SHA-256 verification after write",hashlib.sha256(pp.read_bytes()).hexdigest()==sha)
    names=dict(payload=payload_name,manifest=f"run_manifest_{run_id}.json",txt=f"{run_id}_READABLE_RUN_LOG.txt",summary=f"{run_id}_SUMMARY.json",jsonl=f"{run_id}_RAW_RECORDS.jsonl",images=f"figures_manifest_{run_id}.json",zip=f"AKBASCORE_MAM_DC6_EVIDENCE_{run_id}.zip")
    mp=run_dir/names["manifest"];sealed_utc=utc_now()
    manifest={"demo":DEMO_TITLE,"author":AUTHOR,"organisation":ORG,"date":DATE_ISO,"copyright":COPYRIGHT,"license":LICENSE_URL,"priority_record":PRIORITY,"run_id":run_id,"payload_file":payload_name,
              "payload_sha256":sha,"payload_bytes":len(pb),"sealed_utc":sealed_utc,"verdict":verdict,"engine_sha256":cfg["engine_sha256"],"test560_lock":TEST560["lock"],"test560_result_sha":TEST560["result_sha"],
              "canonicalization":"json.dumps(sort_keys=True, separators=(',',':'), ensure_ascii=False, allow_nan=False), UTF-8",
              "note":"Artifact integrity seal: confirms the payload is unchanged since sealing. It is not a scientific proof and not a third-party verification."}
    mp.write_text(json.dumps(manifest,indent=2,sort_keys=True,ensure_ascii=False),encoding="utf-8")
    summary={"run_id":run_id,"verdict":verdict,"case":case,"position":position,"question":question,"answer":a1["answer"],"expected":expected,"correct":a1["correct"],
             "cartridges":[{"slot":n_+1,"id":c["id"],"type":c["type"],"target":c["is_target"]}for n_,c in enumerate(cars)],
             "memory_growth":[s["new_len"]for s in steps],"prior_rows_bitwise_unchanged":[s["prior_rows_bitwise_unchanged"]for s in steps[1:]],"source_reads_during_load":len(ids_fw),
             "query_tokens":len(qids),"memory_tokens":T,"weights_unchanged":True,"engine_sha256":cfg["engine_sha256"],"payload_sha256":sha,
             "sealed_test560_reference":{a_:TEST560["arms"][a_]["TOTAL"]for a_ in TEST560["arms"]}}
    (run_dir/names["summary"]).write_text(json.dumps(jsafe(summary),indent=2,ensure_ascii=False),encoding="utf-8")
    lines=[{"op":"step","n":n_+1,"kind":s["kind"],"cartridge_id":s["cartridge_id"],"old_len":s["old_len"],"new_len":s["new_len"],"prior_rows_bitwise_unchanged":s["prior_rows_bitwise_unchanged"],
            "memory_sha":s["memory_sha"],"forwards":s["forwards"]}for n_,s in enumerate(steps)]
    lines+=[{"op":"ask","question":question,"answer":a1["answer"],"forwards":fw_a},{"op":"memory_profile","k_norm":prof["k_norm"],"v_norm":prof["v_norm"]}]
    if cu:lines.append({"op":"ask_free_text","question":cu["question"],"answer":cu["answer"],"forwards":cu["forwards"]})
    (run_dir/names["jsonl"]).write_text("\n".join(json.dumps(jsafe(x),ensure_ascii=False)for x in lines)+"\n",encoding="utf-8")
    (run_dir/names["txt"]).write_text(make_txt(P,sha,names,sealed_utc),encoding="utf-8");seal_s=time.perf_counter()-t_seal
    # ---- 7 figures ----
    yield ev(7,"Rendering figures",f"{N_FIGS} figures at 300 dpi (JPEG) with vector PDF copies; every printed value is checked against the sealed payload.")
    ctx={"P":P,"sha":sha,"run_id":run_id,"payload_name":payload_name,"allow":allow,"strip":strip,"N":N_FIGS,"audit":[],"pdfs":[]}
    t_r=time.perf_counter();imgs=render_all(ctx,run_dir);render_s=time.perf_counter()-t_r;entries=[]
    for n_,(p,cap)in enumerate(imgs,1):
        with Image.open(p)as im:w_,h_=im.size
        q_=ctx["pdfs"][n_-1];entries.append({"index":n_,"jpg":p.name,"pdf":q_.name,"title":cap,"width_px":w_,"height_px":h_,"dpi":OUT_DPI,"jpg_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"pdf_sha256":hashlib.sha256(q_.read_bytes()).hexdigest()})
    imf=run_dir/names["images"]
    imf.write_text(json.dumps({"run_id":run_id,"payload_sha256":sha,"verdict":verdict,"count":len(entries),"render_seconds":render_s,"figures":entries,"figure_text_audit":ctx["audit"],
                               "note":"Per-file SHA-256 is an artifact integrity seal. Figure text was checked against the sealed payload and scanned for claim discipline before saving."},indent=2,ensure_ascii=False),encoding="utf-8")
    # ---- 8 package ----
    yield ev(8,"Packaging","ZIP with figures, PDFs, payload, manifest, logs and raw records; post-seal audit.")
    zp=run_dir/names["zip"];members=[p for p,_ in imgs]+list(ctx["pdfs"])+[imf,mp,pp,run_dir/names["txt"],run_dir/names["summary"],run_dir/names["jsonl"]]
    for m in members:
        if m.suffix in(".json",".txt",".jsonl"):chk(f"claim-discipline scan of {m.name}: 0 hits",not claim_scan(m.name,strip_words(m.read_text(encoding="utf-8"),strip),allow=allow))
    with zipfile.ZipFile(zp,"w",compression=zipfile.ZIP_DEFLATED)as z:
        for m in members:z.write(m,arcname=m.name)
    post=post_seal_audit(imgs,ctx["pdfs"],zp,[m.name for m in members],pp,sha,run_dir/names["txt"],mp,verdict)
    say(f"{run_id}: {len(checks)} pre-seal + {len(post)} post-seal checks = {len(checks)+len(post)}/{len(checks)+len(post)} PASS · figures {render_s:.1f}s · seal {seal_s:.2f}s")
    say("VERDICT :",verdict);say("PAYLOAD :",sha);say("ZIP     :",zp);say("="*140)
    ledger=dict(run_id=run_id,utc=sealed_utc,case=case,position=position,question=question,answer=a1["answer"],expected=expected,correct=bool(a1["correct"]),memory_tokens=T,
                prior_rows_unchanged=all(s["prior_rows_bitwise_unchanged"]for s in steps[1:]),weights_unchanged=True,verdict=verdict,payload_sha256=sha)
    return dict(run_id=run_id,run_dir=run_dir,P=P,sha=sha,imgs=imgs,pdfs=ctx["pdfs"],zip=zp,payload=pp,txt=run_dir/names["txt"],manifest=mp,images_manifest=imf,summary=run_dir/names["summary"],
                jsonl=run_dir/names["jsonl"],checks=len(checks)+len(post),verdict=verdict,ledger=ledger)
# ---------------- optional LIVE replay of the locked panel (72 runs; reported separately from the sealed TEST560 reference) ----------------
def replay_report(I,run_id,run_dir,progress=None):
    info=I.info;t0=time.perf_counter();start=utc_now();g0=I.guard();s0=I.sentinel()if hasattr(I,"sentinel")else info["sentinel0"]
    if g0!=info["guard0"]:raise AuditFail("all-parameter guard before replay differs from startup")
    R=I.replay(progress);g1=I.guard()
    if g1!=g0:raise AuditFail("all-parameter guard changed during replay")
    rows=R["rows"];score=sum(int(r["correct"])for r in rows)
    if score!=R["score"]or len(rows)!=72:raise AuditFail("replay score does not re-derive from its rows")
    per={p:sum(int(r["correct"])for r in rows if r["position"]==p)for p in POSITIONS}
    out={"schema":"akbascore.mam.dc6.live_replay.v1","demo":DEMO_TITLE,"author":AUTHOR,"organisation":ORG,"date":DATE_ISO,"license":LICENSE_URL,"run_id":run_id,"start_utc":start,"end_utc":utc_now(),
         "status":"LIVE REPLAY — separate from the sealed TEST560 reference","score":score,"total":72,"per_position":per,"matches_TEST560_total":score==72,"panel_sha":R["panel_sha"],
         "engine_sha256":ENGINE_SHA256_EXPECTED,"guard_before":g0,"guard_after":g1,"weights_unchanged":g1==g0==info["guard0"],"gpu":info["gpu"],"seconds":round(time.perf_counter()-t0,2),
         "rows":rows,"sealed_reference":{"name":"TEST560","INCR_DC6":TEST560["arms"]["INCR_DC6"],"result_sha":TEST560["result_sha"]}}
    pb=canon(out);sha=hashlib.sha256(pb).hexdigest();jp=run_dir/f"{run_id}_LIVE_REPLAY.json";jp.write_bytes(pb)
    S="="*120;txt=[S,f"{DEMO_TITLE} — LIVE REPLAY OF THE LOCKED TEST560 PANEL",f"{DISCOVERY} · {ORG} · {DATE_TXT} · {LICENSE_NAME} · {LICENSE_URL}",S,
         "This replay is a LIVE measurement in this runtime. It is reported separately from the sealed TEST560 record and does not replace it.",
         f"run {run_id} · {start} → {out['end_utc']} · {out['seconds']} s · GPU {info['gpu']}",f"LIVE score {score}/72 · FIRST {per['FIRST']}/24 · MIDDLE {per['MIDDLE']}/24 · LAST {per['LAST']}/24",
         f"sealed TEST560 incremental DC6: {TEST560['arms']['INCR_DC6']['TOTAL']}/72 (result SHA {TEST560['result_sha']})",f"weights unchanged (all-parameter guard before = after = startup): {out['weights_unchanged']}",
         f"payload SHA-256 {sha}","-"*120]+[f"case {r['case']:02d} {r['position']:<6} expected {r['expected']:<10} answer {r['answer']!r} {'PASS'if r['correct']else'FAIL'}"for r in rows]+[S]
    tp=run_dir/f"{run_id}_LIVE_REPLAY.txt";tp.write_text("\n".join(txt),encoding="utf-8")
    for p_ in(jp,tp):
        h_=claim_scan(p_.name,strip_words(p_.read_text(encoding="utf-8"),I.panel_strings()),allow=(info["gpu"],CANON_GPU))
        if h_:raise AuditFail(f"claim-discipline hit in {p_.name}: {h_[:3]}")
    return dict(out=out,sha=sha,json=jp,txt=tp)
#<<CORE_END>>
#<<FIGURES_BEGIN>>
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":.8,"axes.edgecolor":"#374151","axes.labelcolor":"#111827","xtick.color":"#374151","ytick.color":"#374151",
                     "xtick.labelsize":8,"ytick.labelsize":8,"axes.labelsize":9,"axes.titlesize":9.5,"axes.titleweight":"bold","axes.titlelocation":"left","pdf.fonttype":42,"ps.fonttype":42,
                     "legend.fontsize":8,"legend.frameon":False})
FIG_W,FIG_H=13.333,7.5;LAYOUT_DPI=100;OUT_DPI=300
INK,INK2,MUTED,RULE,GRIDC,PANEL="#111827","#374151","#6B7280","#D1D5DB","#E5E7EB","#F9FAFB"
BLUE,BLUE_L,ORANGE,ORANGE_L,TEAL,TEAL_L,GRAYPT,AMBER_L,AMBER,RED="#1D4ED8","#DBEAFE","#C2410C","#FFEDD5","#0F766E","#CCFBF1","#9CA3AF","#FEF3C7","#B45309","#B91C1C"
SEGC=["#9CA3AF","#1D4ED8","#0F766E","#7C3AED","#B45309","#BE185D"]
MONO="DejaVu Sans Mono"
def mt(s):return str(s).replace("$",r"\$")
def ascii_(s):return unicodedata.normalize("NFKD",str(s)).encode("ascii","ignore").decode("ascii")
def new_fig():return plt.figure(figsize=(FIG_W,FIG_H),dpi=LAYOUT_DPI,facecolor="white")
def sanitize_dashes(fig):
    """Zero-width artists are forced to a solid style: some matplotlib builds raise on a dashed pattern scaled to zero width."""
    for a in fig.findobj(lambda o:isinstance(o,(Patch,Line2D))):
        try:
            if a.get_linewidth()==0 and a.get_linestyle()not in("-","solid","None","none"," ",""):a.set_linestyle("-")
        except Exception:pass
def finish(fig,path,ctx,expect,desc):
    """Every expected value must appear in the figure text; claim scan must be clean; then JPEG (300 dpi, EXIF provenance) + vector PDF."""
    sanitize_dashes(fig);fig.canvas.draw();texts=[t.get_text()for t in fig.findobj(Text)if t.get_text().strip()];blob="\n".join(texts);nm=Path(path).name
    ws=lambda z:re.sub(r"\s+","",z);bw=ws(blob);miss=[e for e in expect if ws(e)not in bw]
    if miss:plt.close(fig);raise AuditFail(f"FIGURE/PAYLOAD MISMATCH in {nm}: missing {miss}")
    hits=claim_scan(nm,strip_words(re.sub(r"\s*\n\s*"," ",blob),ctx.get("strip",[])),allow=ctx["allow"])
    if hits:plt.close(fig);raise AuditFail(f"CLAIM-DISCIPLINE HIT in {nm}: {hits[:3]}")
    pdf=Path(path).with_suffix(".pdf")
    fig.savefig(pdf,format="pdf",facecolor="white",metadata={"Title":desc,"Author":f"{AUTHOR} · {ORG}","Subject":f"{DEMO_TITLE} · run {ctx['run_id']} · payload SHA-256 {ctx['sha']}","Creator":DEMO_SHORT,
                "Keywords":f"engine sha256 {ENGINE_SHA256}; TEST560 result {TEST560['result_sha']}; license {LICENSE_URL}"})
    buf=io.BytesIO();fig.savefig(buf,format="png",dpi=OUT_DPI,facecolor="white");plt.close(fig);buf.seek(0)
    with Image.open(buf)as im:rgb=im.convert("RGB")
    ex=Image.Exif();ex[0x010E]=ascii_(f"{desc} | {DEMO_SHORT} | run {ctx['run_id']} | payload sha256 {ctx['sha']} | engine sha256 {ENGINE_SHA256}");ex[0x013B]=ascii_(f"{AUTHOR} / {ORG}");ex[0x8298]=ascii_(f"{COPYRIGHT} - {LICENSE_NAME} - {LICENSE_URL}");ex[0x0131]=ascii_(DEMO_SHORT)
    rgb.save(path,"JPEG",quality=95,optimize=True,progressive=False,subsampling=0,dpi=(OUT_DPI,OUT_DPI),exif=ex.tobytes())
    with Image.open(path)as c_:
        if c_.format!="JPEG"or c_.mode!="RGB":raise RuntimeError("JPEG validation failed")
    ctx["pdfs"].append(pdf);ctx["audit"].append(dict(file=nm,texts=len(texts),expected_values=len(expect),claim_hits=0));return str(path)
def fit_text(fig,x,y,w,h,text,fs_max=10,fs_min=6.5,color=INK,family=None,ls=1.3,weight="normal",ha="left"):
    fig.canvas.draw();r=fig.canvas.get_renderer();Wp=w*fig.get_figwidth()*fig.dpi;Hp=h*fig.get_figheight()*fig.dpi
    if Wp<=4 or Hp<=4:return 0
    paras=str(text if text else"(empty)").replace("\r","").split("\n");x0=x+w/2 if ha=="center"else x
    kw={"va":"top","ha":ha,"color":color,"linespacing":ls,"weight":weight,"multialignment":ha}
    if family:kw["family"]=family
    def wrap(c):
        out=[]
        for p in paras:out.extend(textwrap.wrap(p,c,break_long_words=True,break_on_hyphens=False,subsequent_indent=("  "if p.startswith("• ")else""))or[""])
        return out
    fs=float(fs_max);k=0.52
    for _ in range(150):
        cpl=max(6,int(Wp/(k*fs*fig.dpi/72.0)));ln=wrap(cpl);t=fig.text(x0,y+h,mt("\n".join(ln)),fontsize=fs,**kw);bb=t.get_window_extent(renderer=r)
        if bb.width>Wp*1.002 and cpl>6:t.remove();k*=max(1.01,bb.width/Wp*1.01);continue
        if bb.height<=Hp:return fs
        t.remove()
        if fs>fs_min:fs=max(float(fs_min),fs-0.25);continue
        per=bb.height/max(1,len(ln));keep=max(1,int(Hp/per)-1);fig.text(x0,y+h,mt("\n".join(ln[:keep]+["[… continued in the readable run log]"])),fontsize=fs,**kw);return fs
    fig.text(x0,y+h,"[text omitted — see run log]",fontsize=fs_min,**kw);return fs_min
def header(fig,k,title,sub,tag,tagc):
    fig.text(.03,.955,f"Fig. {k}",fontsize=12,weight="bold",color=MUTED,va="center")
    fig.text(.078,.955,mt(title),fontsize=16.5,weight="bold",color=INK,va="center")
    fig.text(.03,.917,mt(sub),fontsize=9.4,color=INK2,va="center")
    fig.text(.97,.955,tag,fontsize=8.4,weight="bold",color=tagc,va="center",ha="right",bbox=dict(boxstyle="square,pad=0.35",fc="white",ec=tagc,lw=1.0))
    fig.add_artist(Line2D([.03,.97],[.895,.895],transform=fig.transFigure,color=INK2,lw=.8))
def footer(fig,ctx,k,caption):
    fig.add_artist(Line2D([.03,.97],[.132,.132],transform=fig.transFigure,color=RULE,lw=.6))
    fit_text(fig,.03,.045,.94,.08,caption,fs_max=8.6,fs_min=6.6,color=INK2)
    fig.add_artist(Line2D([.03,.97],[.038,.038],transform=fig.transFigure,color=RULE,lw=.6))
    fig.text(.5,.025,mt(f"{DEMO_TITLE} · {DISCOVERY} · {ORG} · {DATE_TXT} · {LICENSE_NAME} · {LICENSE_URL}"),ha="center",va="center",fontsize=6.6,color=INK2)
    fig.text(.5,.010,mt(f"live run {ctx['run_id']} (TEST560 engine; not TEST560 itself) · engine SHA-256 {ENGINE_SHA256[:12]}… · TEST560 result {TEST560['result_sha'][:12]}… · payload {ctx['sha'][:12]}… · Fig. {k}/{ctx['N']} · {COPYRIGHT}"),
             ha="center",va="center",fontsize=6.2,color=MUTED,family=MONO)
def clean(ax,grid=True):
    for s in("top","right"):ax.spines[s].set_visible(False)
    if grid:ax.grid(True,color=GRIDC,lw=.5);ax.set_axisbelow(True)
def frame(fig,x,y,w,h,ec=INK2,fc="white",lw=.9,ls="-",z=-1):
    fig.add_artist(Rectangle((x,y),w,h,transform=fig.transFigure,facecolor=fc,edgecolor=ec,lw=lw,ls=ls,zorder=z))
def block(fig,x,y,w,h,stage,title,body,ec=INK2,fc="white",mono=True,bfs=8.4):
    frame(fig,x,y,w,h,ec,fc);frame(fig,x,y+h-.032,w,.032,ec,PANEL if fc=="white"else fc,lw=.9)
    fig.text(x+.007,y+h-.016,stage,fontsize=7,weight="bold",color=MUTED,va="center")
    fig.text(x+.007,y+h-.052,mt(title),fontsize=9.6,weight="bold",color=INK,va="center")
    if body:fit_text(fig,x+.007,y+.008,w-.014,h-.078,body,fs_max=bfs,fs_min=6.4,family=MONO if mono else None,color=INK2)
def arrow(fig,x0,y0,x1,y1,color=INK2,lw=1.1,ms=11,cs="arc3"):fig.add_artist(FancyArrowPatch((x0,y0),(x1,y1),transform=fig.transFigure,arrowstyle="-|>",mutation_scale=ms,lw=lw,color=color,connectionstyle=cs,zorder=6))
def L_(ctx):return ctx["P"]["live"]
def short(s,n):s=str(s);return s if len(s)<=n else s[:n-1]+"…"
# ------------------------------------------------------------------ FIG 1 · write → append → ask
def f1(ctx,k,path):
    L=L_(ctx);cfg=ctx["P"]["engine"]["config"];st=L["steps"];cars=L["cartridges"];T=L["memory"]["T"];P_=cfg["prefix_tokens"];fig=new_fig();a1=L["answer"]
    header(fig,k,"Write → append-only consolidation → source-free answer",f"Case {L['case']:02d} · target at {L['position']} · the five cartridges, token counts and the answer below are taken from this run.","ARCHITECTURE · VALUES FROM THIS RUN",INK2)
    for x,s in((.03,"(a) SOURCE FACTS · READ ONCE"),(.255,"(b) WRITE · INDEPENDENT CARTRIDGES"),(.505,"(c) APPEND · CONSOLIDATED MEMORY"),(.765,"(d) QUESTION · NO SOURCE TEXT")):fig.text(x,.865,s,fontsize=8,weight="bold",color=INK2,va="center")
    y0=.79;hh=.098;gap=.012
    for n_,(c,s)in enumerate(zip(cars,st)):
        y=y0-n_*(hh+gap);col=SEGC[n_+1];tg=c["is_target"]
        frame(fig,.03,y-hh,.205,hh,ec=col,fc=(BLUE_L if tg else"white"),lw=1.4 if tg else .9)
        fig.text(.036,y-.018,f"C{n_+1} · #{c['id']:02d} · {c['type']}"+("  · TARGET"if tg else""),fontsize=7.8,weight="bold",color=col,va="center")
        fit_text(fig,.036,y-hh+.006,.193,hh-.034,c["fact"],fs_max=8.2,fs_min=6.4,color=INK)
        frame(fig,.255,y-hh,.225,hh,ec=col,fc="white",lw=.9)
        fig.text(.261,y-.018,f"cartridge C{n_+1} · written in step {n_+1}",fontsize=7.6,weight="bold",color=col,va="center")
        wr=s["forwards"][0]["n"];body=c["body_len"]
        fig.text(.261,y-.044,mt(f"1 frozen forward · {wr} token ids · no cache"),fontsize=7.3,family=MONO,color=INK2,va="center")
        fig.text(.261,y-.066,mt(f"H{cfg['CUT']} residual  {(wr if n_==0 else body)} × 4096"),fontsize=7.3,family=MONO,color=INK2,va="center")
        fig.text(.261,y-.087,mt(f"K/V layers 0–{cfg['CUT']}  {(wr if n_==0 else body)} rows"),fontsize=7.3,family=MONO,color=INK2,va="center")
        arrow(fig,.235,y-hh/2,.255,y-hh/2,color=col);arrow(fig,.48,y-hh/2,.505,y-hh/2,color=col)
    ax=fig.add_axes([.515,.255,.225,.535]);ax.set_zorder(3);ax.set_xlim(0,T);ax.set_ylim(-.5,4.5);ax.invert_yaxis();clean(ax,grid=False)
    seg=L["segments"]
    for n_,s in enumerate(st):
        old=s["old_len"];new=s["new_len"]
        if n_==0:ax.barh(n_,P_,left=0,color=SEGC[0],height=.62);ax.barh(n_,new-P_,left=P_,color=SEGC[1],height=.62)
        else:
            for m_,(lab,a_,b_,_)in enumerate(seg[:n_+1]):ax.barh(n_,b_-a_,left=a_,color=SEGC[m_],height=.62,alpha=.28)
            ax.barh(n_,new-old,left=old,color=SEGC[n_+1],height=.62)
        ax.text(new+.6,n_,f"{new}",fontsize=7.6,va="center",color=INK)
    ax.set_yticks([]);ax.spines["left"].set_visible(False);ax.set_xlabel("memory rows (tokens)",fontsize=8)
    ax.set_title("memory after step 1 … 5 (faded = earlier rows, never recomputed)",fontsize=7.8,weight="normal",pad=3)
    fig.text(.515,.205,mt(f"layers 7–31: computed only for new rows,\nattending to the existing memory · layers 0–{cfg['CUT']}: keys re-phased (RoPE)"),fontsize=7.4,color=INK2,va="top")
    block(fig,.765,.43,.205,.36,"QUERY","Question over the memory",f"{L['question']}\n\nmodel input: {L['query_tokens']} template tokens only\ninstalled memory: {T} numerical rows\n(prefix + C1 … C5)\nsource facts given to the model: none\nweights: frozen",mono=False,bfs=8.6)
    arrow(fig,.742,.6,.765,.6)
    frame(fig,.765,.205,.205,.2,ec=BLUE if L["correct"]else ORANGE,fc=BLUE_L if L["correct"]else ORANGE_L,lw=1.3)
    fig.text(.775,.385,"ANSWER (generated by the frozen model)",fontsize=7.4,weight="bold",color=INK2,va="center")
    fig.text(.775,.33,mt(short(a1["answer"],22)),fontsize=15,weight="bold",color=INK,va="center")
    fig.text(.775,.272,mt(f"expected {L['expected']} · {'CORRECT'if L['correct']else'INCORRECT'}"),fontsize=8.6,color=BLUE if L["correct"]else ORANGE,weight="bold",va="center")
    fig.text(.775,.235,"expected answer = audit metadata",fontsize=7,color=MUTED,va="center")
    footer(fig,ctx,k,f"Figure {k}. The executed memory path for case {L['case']:02d}. (a) Each source fact is read once. (b) The frozen model writes an independent numerical cartridge per fact: the layer-{cfg['CUT']} residual H{cfg['CUT']} and the layers 0–{cfg['CUT']} K/V. "
           f"(c) Cartridges are appended in order; for every append, layers 7–31 are computed only for the new rows over the existing memory, so earlier rows are never recomputed (bitwise check, Fig. 2). "
           f"(d) The question template ({L['query_tokens']} tokens) is processed over the {T}-token numerical memory; no source text is given to the model. Weights are frozen throughout.")
    return finish(fig,path,ctx,[f"{T}",f"{L['query_tokens']} template tokens",mt(short(a1["answer"],22)),f"expected {L['expected']}"],"Write, append, ask")
# ------------------------------------------------------------------ FIG 2 · append-only growth (measured)
def f2(ctx,k,path):
    L=L_(ctx);st=L["steps"];cars=L["cartridges"];T=L["memory"]["T"];fig=new_fig()
    header(fig,k,"Append-only consolidation, measured forward by forward","Every model forward during load_case() and a bitwise comparison of all previously consolidated K/V rows after each append.","LIVE MEASUREMENT",BLUE)
    ax=fig.add_axes([.06,.31,.42,.52]);ax.set_zorder(3);clean(ax)
    for n_,s in enumerate(st):
        old=s["old_len"];add=s["new_len"]-old;col=SEGC[n_+1]
        ax.bar(n_,old,color="#E5E7EB",edgecolor=INK2,lw=.5,width=.62);ax.bar(n_,add,bottom=old,color=col,width=.62)
        ax.text(n_,s["new_len"]+1.2,f"{s['new_len']}",ha="center",fontsize=8,color=INK)
        if n_:ax.text(n_,old/2,"✓",ha="center",va="center",fontsize=13,color=TEAL,weight="bold")
    ax.set_xticks(range(5));ax.set_xticklabels([f"step {n_+1}\nC{n_+1}"+("\nTARGET"if c["is_target"]else"")for n_,c in enumerate(cars)],fontsize=7.8)
    ax.set_ylabel("memory rows (tokens)");ax.set_ylim(0,T*1.12);ax.set_title("memory length after each step",fontsize=8.6)
    ax.legend(handles=[Patch(facecolor="#E5E7EB",edgecolor=INK2,lw=.5,label="earlier rows (✓ bitwise unchanged)"),Patch(color=SEGC[1],label="rows added in this step")],loc="upper left")
    tx=.53;fig.text(tx,.83,"Forward record (instrumentation)",fontsize=10,weight="bold",color=INK)
    fig.text(tx,.80,mt(f"{'step':<5}{'write forward':<27}{'consolidation forward':<38}{'prior rows':<12}"),fontsize=7.7,family=MONO,color=MUTED)
    for n_,s in enumerate(st):
        f=s["forwards"];y=.772-n_*.083
        w_=f"ids {f[0]['n']} · no cache"
        c_=(f"zero-emb {f[1]['n']} · cache {f[1]['past_len']}"if f[1]["has_past"]else f"zero-emb {f[1]['n']} · no cache")
        pr="init"if s["kind"]=="init"else("✓ identical"if s["prior_rows_bitwise_unchanged"]else"✗ CHANGED")
        fig.text(tx,y,mt(f"{n_+1:<5}{w_:<27}{c_:<38}{pr:<12}"),fontsize=7.9,family=MONO,color=INK,weight="bold"if cars[n_]["is_target"]else"normal")
        if s["kind"]=="append":fig.text(tx,y-.028,mt(f"     SHA rows 0…{s['old_len']-1}: {s['prior_sha_before'][:12]}… = {s['prior_sha_after'][:12]}…"),fontsize=7.1,family=MONO,color=INK2)
        else:fig.text(tx,y-.028,mt(f"     prefix + body of C1 written together ({s['new_len']} rows)"),fontsize=7.1,family=MONO,color=INK2)
    frame(fig,tx,.17,.44,.17,ec=TEAL,fc=TEAL_L,lw=1.0)
    fit_text(fig,tx+.008,.178,.424,.155,f"Measured in this run:\n• source token-id forwards during load: {L['source_reads']} — one per cartridge, each its own fact, each once\n"
             f"• consolidation forwards with all-zero input embeddings: {L['zero_embed_forwards']} (stored H{ctx['P']['engine']['config']['CUT']} injected; no token ids)\n"
             f"• previously consolidated rows bitwise unchanged after all 4 appends: {all(s['prior_rows_bitwise_unchanged']for s in st[1:])}\n• final memory {T} rows · SHA-256 {L['memory_sha'][:20]}…",fs_max=8.4,fs_min=6.6,color=INK)
    fig.text(.06,.21,mt(f"total load_case() time {1000*L['setup']['elapsed_seconds']:.0f} ms (engine-reported) · per step: "+" · ".join(f"{1000*s['seconds']:.0f}"for s in st)+" ms"),fontsize=7.8,family=MONO,color=INK2)
    footer(fig,ctx,k,f"Figure {k}. Live record of the five consolidation steps. Step 1 writes cartridge C1 together with the 27-token instruction prefix; steps 2–5 append C2–C5. In each step the instrumentation sees exactly one token-id forward "
           "(the write of the new cartridge's own fact, no cache) and one consolidation forward whose input embeddings are all zero (the stored H6 is injected) and whose length equals the new rows only, over the existing memory. "
           "After every append all earlier K/V rows of all 32 layers are compared bitwise with the memory before the append.")
    return finish(fig,path,ctx,[f"{T}",f"{L['source_reads']}",L["memory_sha"][:20],"✓ identical"],"Append-only consolidation")
# ------------------------------------------------------------------ FIG 3 · consolidated memory map
def f3(ctx,k,path):
    L=L_(ctx);M=L["memory"];T=M["T"];seg=L["segments"];cfg=ctx["P"]["engine"]["config"];fig=new_fig()
    header(fig,k,"The consolidated numerical memory",f"{M['layers']} layers × {T} token rows · {M['kv_heads']} KV heads × {M['head_dim']} · {M['bytes']/2**20:.2f} MiB ({M['dtype']}) · values read from the memory of this run.","LIVE MEASUREMENT",BLUE)
    for n_,(key,ttl)in enumerate((("k_norm","‖K‖ per row (mean over KV heads)"),("v_norm","‖V‖ per row (mean over KV heads)"))):
        A=np.asarray(M[key],dtype=float);mu=A.mean(axis=1,keepdims=True);sd=A.std(axis=1,keepdims=True);sd[sd==0]=1;Z=(A-mu)/sd
        ax=fig.add_axes([.06,.585-n_*.29,.80,.235]);ax.set_zorder(3)
        im=ax.imshow(Z,aspect="auto",cmap="RdBu_r",vmin=-2.5,vmax=2.5,interpolation="nearest",origin="lower")
        for lab,a_,b_,_ in seg[1:]:ax.axvline(a_-.5,color=INK,lw=.8)
        ax.axhline(cfg["CUT"]+.5,color=AMBER,lw=1.4,ls=(0,(4,2)))
        ax.set_yticks([0,cfg["CUT"],16,31]);ax.set_yticklabels(["L0",f"L{cfg['CUT']}","L16","L31"],fontsize=7);ax.set_title(mt(ttl+" · z-scored within each layer"),fontsize=8.4)
        ax.set_xlim(-.5,T-.5);ax.set_xticks([]);cb=fig.colorbar(im,cax=fig.add_axes([.87,.585-n_*.29,.008,.235]));cb.ax.tick_params(labelsize=6.5);cb.outline.set_linewidth(.5)
    fig.text(.86,.852,mt(f"dashed line: layers 0–{cfg['CUT']} below come from the cartridges · layers {cfg['CUT']+1}–31 above were computed by the appends"),fontsize=7.6,color=AMBER,ha="right",va="center",weight="bold")
    sx=fig.add_axes([.06,.235,.80,.032]);sx.set_zorder(3);sx.set_xlim(-.5,T-.5);sx.set_ylim(0,1);sx.axis("off")
    for n_,(lab,a_,b_,ci)in enumerate(seg):
        c_=L["cartridges"][n_-1]if n_ else None;tg=bool(c_ and c_["is_target"])
        sx.add_patch(Rectangle((a_-.5,0),b_-a_,1,color=SEGC[n_],alpha=1 if(tg or n_==0)else .75,lw=0))
        sx.text((a_+b_-1)/2,.5,(lab+("·T"if tg else"")),ha="center",va="center",fontsize=7.6,color="white",weight="bold")
    fig.text(.06,.218,mt("row layout: "+" · ".join(f"{lab} rows {a_}–{b_-1}"for lab,a_,b_,_ in seg)),fontsize=7.5,family=MONO,color=INK2,va="top")
    footer(fig,ctx,k,f"Figure {k}. Map of the consolidated memory after the fifth append. Each column is one token row, each row of the heat map one layer; colour shows the row's mean K (top) or V (bottom) norm, z-scored within each layer, for display only. "
           f"Vertical lines separate the instruction prefix and the five cartridges (T marks the target). Layers 0–{cfg['CUT']} come from the cartridges (keys re-phased to their memory positions); layers {cfg['CUT']+1}–31 were computed by the appends. The memory holds numbers only; no text is stored in it.")
    return finish(fig,path,ctx,[f"{T} token rows",f"{M['bytes']/2**20:.2f} MiB"],"Consolidated memory")
# ------------------------------------------------------------------ FIG 4 · source-free question answering
def f4(ctx,k,path):
    L=L_(ctx);a1=L["answer"];T=L["memory"]["T"];qt=L["query_token_text"];fig=new_fig();ok=L["correct"]
    header(fig,k,"Answering from numerical memory without the source text","The exact model input at query time, the installed memory and the generated answer.","LIVE MEASUREMENT",BLUE)
    frame(fig,.03,.645,.94,.19,ec=RULE,fc=PANEL,lw=.7)
    fig.text(.04,.815,mt(f"MODEL INPUT AT QUERY TIME · {L['query_tokens']} token ids (question template) · over {T} memory rows · source text: none"),fontsize=8.6,weight="bold",color=INK2,va="center")
    toks=[t.replace("\n","⏎")for t in qt];x=.04;y=.775;fig.canvas.draw();rdr=fig.canvas.get_renderer();FW=fig.get_figwidth()*fig.dpi
    for t in toks:
        s_=mt(t if t.strip()else"·");tt=fig.text(x,y,s_,fontsize=8.2,family=MONO,color=INK,va="center",bbox=dict(boxstyle="square,pad=0.25",fc="white",ec=RULE,lw=.5))
        w=tt.get_window_extent(renderer=rdr).width/FW
        if x+w>.96 and x>.041:tt.set_position((.04,y-.045));x=.04;y-=.045
        x+=w+.006
    fig.text(.04,.665,mt(f"installed memory: {T} rows × 32 layers (prefix + C1…C5) · first query forward cache = {L['query_past']} · generation forwards {L['gen_forwards']} (one token each)"),fontsize=7.9,family=MONO,color=INK2,va="center")
    frame(fig,.03,.2,.46,.415,ec=BLUE if ok else ORANGE,fc=BLUE_L if ok else ORANGE_L,lw=1.3)
    fig.text(.045,.585,"REFERENCE QUESTION OF THIS CASE",fontsize=8,weight="bold",color=INK2,va="center")
    fit_text(fig,.045,.49,.43,.075,L["question"],fs_max=11,fs_min=7,color=INK)
    fig.text(.045,.44,"answer",fontsize=8,color=MUTED,va="center");fig.text(.045,.375,mt(short(a1["answer"],30)),fontsize=23,weight="bold",color=INK,va="center")
    fig.text(.045,.29,mt(f"expected {L['expected']} (audit metadata) · {'CORRECT'if ok else'INCORRECT'}"),fontsize=9,weight="bold",color=BLUE if ok else ORANGE,va="center")
    fig.text(.045,.228,mt(f"repeat: identical = {L['repeat_identical']} · memory unchanged by querying = {L['memory_unchanged_by_query']} · {1000*a1['elapsed_seconds']:.0f} ms"),fontsize=7.8,family=MONO,color=INK2,va="center")
    cu=L.get("custom")
    frame(fig,.51,.2,.46,.415,ec=RULE,fc="white",lw=.8)
    fig.text(.525,.585,"FREE-TEXT QUESTION (optional · unscored)",fontsize=8,weight="bold",color=INK2,va="center")
    if cu:
        fit_text(fig,.525,.49,.43,.075,cu["question"],fs_max=11,fs_min=7,color=INK)
        fig.text(.525,.44,"answer",fontsize=8,color=MUTED,va="center");fit_text(fig,.525,.255,.43,.165,cu["answer"],fs_max=15,fs_min=7.5,color=INK,weight="bold")
        fig.text(.525,.228,mt(f"{cu['query_tokens']} template tokens over the same {T}-row memory · not scored"),fontsize=7.8,family=MONO,color=INK2,va="center")
    else:fit_text(fig,.525,.25,.43,.3,"No free-text question in this run. Visitors may type their own question; it is answered from the same numerical memory and reported without a score.",fs_max=9,fs_min=7,color=MUTED)
    footer(fig,ctx,k,f"Figure {k}. Source-free question answering for case {L['case']:02d}. Top: the complete list of token ids given to the model at query time, decoded for display — the QUESTION/ANSWER template only; none of the five source facts is supplied. "
           f"The model attends to the {T}-row numerical memory installed as its cache and generates the answer greedily. Repeating the question gives the same answer, and the memory's SHA-256 is unchanged by querying.")
    return finish(fig,path,ctx,[f"{L['query_tokens']} token ids",mt(short(a1["answer"],30)),f"expected {L['expected']}"],"Source-free answering")
# ------------------------------------------------------------------ FIG 5 · sealed TEST560 reference + live
def f5(ctx,k,path):
    L=L_(ctx);fig=new_fig();A=TEST560["arms"]
    header(fig,k,"Sealed TEST560 reference and the live result","Left: archived TEST560 values on the locked panel (24 cases × 3 positions), reproduced verbatim. Right: live results of this session, reported separately.","SEALED REFERENCE + LIVE",INK2)
    ax=fig.add_axes([.06,.3,.5,.5]);ax.set_zorder(3);clean(ax);arms=["INCR_DC6","BATCH_DC6","JOINT","INDEP"];cols={"FIRST":BLUE,"MIDDLE":TEAL,"LAST":"#7C3AED"};w=.24
    for j,p in enumerate(POSITIONS):
        v=[A[a_][p]for a_ in arms];ax.bar(np.arange(4)+(j-1)*w,v,w*.92,color=cols[p],label=p)
        for i_,vv in enumerate(v):ax.text(i_+(j-1)*w,vv+.4,str(vv),ha="center",fontsize=7.2,color=INK)
    ax.set_xticks(range(4));ax.set_xticklabels([f"{a_}\n{A[a_]['TOTAL']}/72"for a_ in arms],fontsize=8.2);ax.set_ylim(0,28);ax.set_ylabel("correct of 24 cases");ax.axhline(24,color=MUTED,lw=.6,ls=(0,(3,3)))
    ax.legend(loc="upper right",ncol=3);ax.set_title("TEST560 · sealed · target position FIRST / MIDDLE / LAST",fontsize=8.6)
    fit_text(fig,.06,.155,.5,.065,"INCR_DC6 = "+ARM_DESC["INCR_DC6"]+" · BATCH_DC6 = "+ARM_DESC["BATCH_DC6"]+" · JOINT = "+ARM_DESC["JOINT"]+" · INDEP = "+ARM_DESC["INDEP"],fs_max=7.6,fs_min=6.4,color=INK2)
    tx=.6;frame(fig,tx,.47,.37,.36,ec=BLUE,fc=BLUE_L,lw=1.1)
    fig.text(tx+.01,.81,"LIVE · this run",fontsize=9.6,weight="bold",color=BLUE,va="center")
    fig.text(tx+.01,.775,mt(f"case {L['case']:02d} · target {L['position']} · answer {short(L['answer']['answer'],18)}"),fontsize=8.4,family=MONO,color=INK,va="center")
    fig.text(tx+.01,.745,mt(f"expected {L['expected']} · {'CORRECT'if L['correct']else'INCORRECT'}"),fontsize=8.4,family=MONO,color=BLUE if L["correct"]else ORANGE,weight="bold",va="center")
    ses=L.get("session")or[];per={p:[r for r in ses if r["position"]==p]for p in POSITIONS}
    fig.text(tx+.01,.705,"LIVE · sealed runs in this session (including this one)",fontsize=8.2,weight="bold",color=INK2,va="center")
    for j,p in enumerate(POSITIONS):
        rr=per[p];fig.text(tx+.01,.675-j*.028,mt(f"{p:<7} {sum(int(r['correct'])for r in rr)}/{len(rr)} correct"),fontsize=8.2,family=MONO,color=INK,va="center")
    fit_text(fig,tx+.01,.48,.35,.11,"A single live run is a demonstration observation. The optional live replay of all 72 panel runs is reported in its own file. Neither replaces or recomputes the sealed TEST560 record.",fs_max=7.7,fs_min=6.4,color=INK2)
    frame(fig,tx,.17,.37,.27,ec=AMBER,fc=AMBER_L,lw=.9)
    fig.text(tx+.01,.42,"SCALE BOUNDARY · TEST563 (same mechanism, sealed)",fontsize=8.2,weight="bold",color=AMBER,va="center")
    bx=fig.add_axes([tx+.035,.245,.315,.14]);bx.set_zorder(3);ns=[n for n,_ in SCALE_BOUNDARY["rows"]];fr_=[c/n for n,c in SCALE_BOUNDARY["rows"]]
    bx.plot(range(len(ns)),fr_,marker="o",color=AMBER,lw=1.2);clean(bx);bx.set_xticks(range(len(ns)));bx.set_xticklabels([f"{n}\n{c}/{n}"for n,c in SCALE_BOUNDARY["rows"]],fontsize=6.8);bx.set_ylim(0,1.08);bx.set_ylabel("fraction correct",fontsize=7);bx.tick_params(labelsize=6.8)
    bx.set_xlabel("cartridges in memory",fontsize=7)
    footer(fig,ctx,k,f"Figure {k}. Left: the sealed TEST560 record (result SHA-256 {TEST560['result_sha'][:16]}…): append-only incremental DC6 72/72, batch DC6 72/72, joint text 72/72 and independent cartridges without consolidation 13/72, on the locked 24-case panel at three target positions. "
           "Right: live results of this session, kept separate, and the sealed TEST563 scale boundary of the same mechanism (accuracy falls as the cartridge population grows: 22/80 at 80 cartridges).")
    return finish(fig,path,ctx,["INCR_DC6\n72/72","INDEP\n13/72","LIVE · this run",f"expected {L['expected']}","22/80"],"TEST560 reference and live result")
# ------------------------------------------------------------------ FIG 6 · provenance and priority record
def f6(ctx,k,path):
    P=ctx["P"];L=P["live"];cfg=P["engine"]["config"];E_=P["environment"];fz=P["frozen"];fig=new_fig()
    header(fig,k,"Provenance, verification and priority record","Configuration, measured integrity quantities, hashes, authorship and the claim boundary of this run.","PROVENANCE · PRIORITY RECORD",INK2)
    cols=[(.03,"CONFIGURATION AND ENGINE",
           f"run           {P['run_id']}\nstart (UTC)   {P['run_start_utc']}\nmodel         {cfg['MODEL_ID'].split('/')[-1]}\nweights       frozen · {P['model']['dtype']} · {P['model']['attn']}\n"
           f"architecture  32 L · 4096 · 32 Q / 8 KV · 128\ncartridge     H{cfg['CUT']} residual + K/V L0–{cfg['CUT']}\nconsolidation append-only · layers {cfg['CUT']+1}–31\nprefix        {cfg['prefix_tokens']} tokens\n"
           f"panel         TEST560 locked\n  {cfg['panel_sha'][:32]}\n  {cfg['panel_sha'][32:]}\nGPU           {E_['gpu']}\ntorch         {E_['torch']}\ntransformers  {E_['transformers']}\ngradio        {E_['gradio']}\n"
           f"engine file   {P['engine']['file']}\nengine SHA-256\n  {P['engine']['sha256'][:32]}\n  {P['engine']['sha256'][32:]}\nend (UTC)     {P['run_end_utc']}"),
          (.35,"VERIFICATION MEASURED IN THIS RUN",
           f"weight sentinel  startup = before = after\n  {fz['sentinel_after'][:40]}…\nall-parameter guard  startup = before = after\n  {fz['guard_after'][:40]}…\n"
           f"trainable tensors   {fz['trainable_tensors']}\nLoRA / optimizer    {fz['lora']} / {fz['optimizer']}\nforeign hooks       {fz['foreign_hooks_before']} → {fz['foreign_hooks_after']}\n\n"
           f"source token-id forwards (load)  {L['source_reads']}\nzero-embedding consolidations    {L['zero_embed_forwards']}\nprior rows unchanged (4 appends) "
           f"{all(s['prior_rows_bitwise_unchanged']for s in L['steps'][1:])}\nquery input = template only      True\nmemory unchanged by querying     {L['memory_unchanged_by_query']}\n"
           f"repeat answer identical          {L['repeat_identical']}\n\nmemory  {L['memory']['T']} rows · {L['memory']['bytes']/2**20:.2f} MiB\n  SHA {L['memory_sha'][:36]}…\nanswer  {short(L['answer']['answer'],24)} · {'correct'if L['correct']else'incorrect'}\n"
           f"pre-seal checks passed: {len(P['checks_pre_seal'])}\nverdict\n  {P['verdict'][:44]}\n  {P['verdict'][44:]}"),
          (.67,"AUTHORSHIP · PRIORITY · CLAIM BOUNDARY",
           f"{DISCOVERY}\n{ORG} · {DATE_TXT}\n{AUTHOR_PLACE}\n\n{PRIORITY}\n\n{LICENSE_NAME}\n{COPYRIGHT}\n{LICENSE_URL}\n\nsealed TEST560\n  result {TEST560['result_sha'][:30]}…\n  INCR_DC6 72/72 · INDEP 13/72\n\n"
           "not established:\n"+"\n".join("  • "+t for t in("large populations (TEST563: 22/80 at 80)","KV-only cartridges (H6 is part of it)","scored free-text questions","multi-step inference across cartridges")))]
    for x,ttl,body in cols:
        frame(fig,x,.155,.30,.715,ec=RULE,fc="white",lw=.8);fig.text(x+.01,.85,ttl,fontsize=9,weight="bold",color=INK2,va="center")
        fit_text(fig,x+.01,.165,.282,.665,body,fs_max=9.2,fs_min=6.2,family=MONO if x<.6 else None,color=INK)
    footer(fig,ctx,k,f"Figure {k}. Provenance record of run {P['run_id']} (verdict {P['verdict']}). Payload SHA-256 {ctx['sha']}. "
           "Hashes are artifact-integrity seals, not scientific proof or third-party verification. All live values come from the raw records stored in the sealed payload; sealed values are reproduced verbatim from the TEST560 and TEST563 records.")
    return finish(fig,path,ctx,[P["run_id"],ENGINE_SHA256[:32],fz["guard_after"][:40],DISCOVERY,ORG],"Provenance and priority record")
FIGURES=[("fig_01_write_append_ask","Fig. 1 · Write → append-only consolidation → source-free answer",f1),("fig_02_append_only_record","Fig. 2 · Append-only consolidation, measured",f2),
         ("fig_03_memory_map","Fig. 3 · The consolidated numerical memory",f3),("fig_04_source_free_answer","Fig. 4 · Answering without the source text",f4),
         ("fig_05_reference_and_live","Fig. 5 · Sealed TEST560 reference and live result",f5),("fig_06_provenance_priority","Fig. 6 · Provenance, verification and priority record",f6)]
N_FIGS=len(FIGURES)
def render_all(ctx,run_dir):
    imgs=[]
    try:
        for i,(slug,cap,fn)in enumerate(FIGURES,1):
            p=Path(run_dir)/f"{slug}_{ctx['run_id']}.jpg";fn(ctx,i,p);imgs.append((p,cap))
    finally:plt.close("all")
    return imgs
#<<FIGURES_END>>
say("PART 2 / 3 complete — measurement pipeline and figures defined. Now run PART 3 / 3 in the next cell.")
MAM_DC6_PART2_OK=True
