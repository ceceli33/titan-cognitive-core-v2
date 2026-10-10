# =====================================================================================================================================
# AKBASCORE MAM-BÇ · EXTERNAL ASSOCIATIVE MEMORY FOR A FROZEN LANGUAGE MODEL — QWEN2.5-7B · PART 2 / 3 · MEASUREMENT PIPELINE AND FIGURES
# Same demo as PART 1. Run this cell after PART 1 / 3, in the same Colab runtime; then run PART 3 / 3.
# Defines the fail-closed measurement run (integrity → FRAM retrieval → CUT3 write + append-only consolidation → source-free question →
# save / reload / answer → negative control → verification → sealed payload, logs, ZIP), six figures (JPEG 300 dpi + vector PDF) and the
# optional live replay of the TEST610 acceptance stage. Nothing here changes the engine.
# Discovered and developed by Mustafa Akbaş · AkbasCore AI Teknoloji · 10 October 2026 · AKBASCORE RESEARCH SOFTWARE LICENSE
# =====================================================================================================================================
if not globals().get("MAM_BC_PART1_OK"):raise RuntimeError("PART 1 / 3 has not been run in this runtime. Run PART 1, then PART 2, then PART 3.")
#<<CORE_BEGIN>>
import unicodedata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.patches import Rectangle,FancyArrowPatch,Patch,Circle
from matplotlib.lines import Line2D
from PIL import Image
ROOT=Path("/content/AKBASCORE_MAM_BC_QWEN_CUT3_FRAM")if os.path.isdir("/content")else Path("/tmp/AKBASCORE_MAM_BC_QWEN_CUT3_FRAM")
ROOT.mkdir(parents=True,exist_ok=True);ALLOWED_PATHS=[str(ROOT)]
N_STAGES=10
def ev(stage,title,body):return dict(stage=stage,title=title,body=body)
def file_ready(p):return p is not None and Path(p).is_file()and Path(p).stat().st_size>0
RUN_PREFIX="MAMBC-";REPLAY_PREFIX="MAMBCREPLAY-"
def prune_runs(keep=4,prefix=RUN_PREFIX):
    runs=sorted([p for p in ROOT.glob(prefix+"*")if p.is_dir()],key=lambda p:p.stat().st_mtime)
    for p in runs[:max(0,len(runs)-keep)]:shutil.rmtree(p,ignore_errors=True)
def segments(steps,p):
    """Row ranges of the CUT3 memory: shared prefix (written with cartridge 1) and the body rows of cartridges 1..5."""
    seg=[("prefix",0,p,None)];prev=p
    for n_,s in enumerate(steps,1):seg.append((f"C{n_}",prev,s["new_len"],s["cartridge_id"]));prev=s["new_len"]
    return seg
def post_seal_audit(imgs,pdfs,zp,member_names,pp,sha,tp,mp,verdict,session_path,session_sha):
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
        ok("CUT3 session file in the ZIP is byte-identical to the sealed SHA-256",hashlib.sha256(z.read(Path(session_path).name)).hexdigest()==session_sha)
    ok("payload SHA-256 recomputation",hashlib.sha256(pp.read_bytes()).hexdigest()==sha);ok("payload JSON parses",isinstance(json.loads(pp.read_bytes().decode("utf-8")),dict))
    m=json.loads(mp.read_text(encoding="utf-8"));ok("manifest hash and verdict match payload",m["payload_sha256"]==sha and m["verdict"]==verdict)
    ok("readable log exists and is non-empty",file_ready(tp));ok("ZIP exists and is non-empty",file_ready(zp))
    return checks
def fw_txt(f):
    if f["kind"]=="ids":return f"token ids n={f['n']}"+(f" · cache {f['past_len']}"if f["has_past"]else" · no cache")
    return f"embeddings n={f['n']} ({'all zero'if f['embeds_zero']else'NON-ZERO'})"+(f" · cache {f['past_len']}"if f["has_past"]else" · no cache")
def mib(b):return b/2**20
def check_build(chk,I,trace,keys,P_,label):
    """Forward-by-forward audit of one engine build_cut3(keys, 3) call (shared by the main memory, the control and the free-text memory)."""
    steps=trace["steps"];n=len(keys)
    chk(f"{label}: {n} consolidation steps in engine order (one init + {n-1} appends)",len(steps)==n and [s["cartridge_id"]for s in steps]==list(keys)and steps[0]["kind"]=="init"and all(s["kind"]=="append"for s in steps[1:]))
    s0=steps[0];f0=s0["forwards"];src0=I.source_ids(keys[0])
    chk(f"{label}: step 1 = one token-id forward of the instruction prefix + cartridge 1's own fact (no cache) + one all-zero-embedding forward over the same rows",
        len(f0)==2 and f0[0]["kind"]=="ids"and f0[0]["ids"]==src0 and not f0[0]["has_past"]and f0[1]["kind"]=="embeds"and f0[1]["embeds_zero"]and f0[1]["n"]==len(src0)and not f0[1]["has_past"]and s0["new_len"]==len(src0))
    for n_,s in enumerate(steps[1:],2):
        f=s["forwards"];src=I.source_ids(s["cartridge_id"]);q=len(src)-P_
        chk(f"{label}: step {n_} write = one token-id forward of the instruction prefix + cartridge {n_}'s own fact only (no cache)",len(f)==2 and f[0]["kind"]=="ids"and f[0]["ids"]==src and not f[0]["has_past"])
        chk(f"{label}: step {n_} consolidation = all-zero embeddings for the {q} new rows only, over the existing {s['old_len']}-token memory",f[1]["kind"]=="embeds"and f[1]["embeds_zero"]and f[1]["n"]==q and f[1]["has_past"]and f[1]["past_len"]==s["old_len"]==steps[n_-2]["new_len"])
        chk(f"{label}: step {n_} previously consolidated rows 0…{s['old_len']-1} bitwise unchanged",s["prior_rows_bitwise_unchanged"]is True and s["prior_sha_before"]==s["prior_sha_after"])
        chk(f"{label}: step {n_} memory grows by exactly the new rows (+{q})",s["new_len"]==s["old_len"]+q)
    ids_fw=[f for f in trace["forwards"]if f["kind"]=="ids"]
    chk(f"{label}: no other text read: exactly {n} token-id forwards, each the instruction prefix + the own fact of the cartridge being written, each once",
        len(ids_fw)==n and [f["ids"]for f in ids_fw]==[I.source_ids(ci)for ci in keys]and len(trace["forwards"])==2*n)
    return steps,ids_fw
def check_query(chk,I,fw,q,T,label):
    qids=I.query_ids(q)
    chk(f"{label}: query forward input = question-template tokens only, over the installed {T}-token memory",len(fw)>=1 and fw[0]["kind"]=="ids"and fw[0]["ids"]==qids and fw[0]["has_past"]and fw[0]["past_len"]==T)
    chk(f"{label}: generation forwards = one new token each, over the growing cache",all(f["kind"]=="ids"and f["n"]==1 and f["has_past"]and f["past_len"]==T+len(qids)+n_ for n_,f in enumerate(fw[1:])))
    return qids
def make_txt(P,sha,names,sealed_utc):
    o=[];a=o.append;S="="*140;Dd="-"*140;E_=P["environment"];L=P["live"];cfg=P["engine"]["config"];fz=P["frozen"];R=L["retrieval"];K=L["disk"];N=L["negative"];su=P["startup"]
    a(S);a(f"{DEMO_TITLE} — READABLE RUN LOG");a(f"{DISCOVERY} · {ORG} · {DATE_TXT} · {AUTHOR_PLACE}");a(f"{LICENSE_NAME} · {COPYRIGHT} · {LICENSE_URL}");a(S)
    a(PRIORITY);a("Scientific question: "+SCI_QUESTION)
    a(f"Derived from {names['payload']} (SHA-256 {sha}). Manifest: {names['manifest']}.")
    a("This is a LIVE run of the frozen Qwen2.5-7B CUT3 + FRAM engine (610.py executed up to its acceptance stage; RAW COLAB DEMO helpers). It is not TEST610 itself; sealed TEST610 and TEST575 values are reproduced verbatim, never recomputed.")
    for k_,v in(("RUN ID",P["run_id"]),("START UTC",P["run_start_utc"]),("END UTC",P["run_end_utc"]),("SEALED UTC",sealed_utc),("VERDICT",P["verdict"]),
        ("MODEL",f"{cfg['MODEL_ID']} · frozen · {P['model']['dtype']} · attention {P['model']['attn']}"),("ARCHITECTURE",arch_txt(tuple(P["model"]["arch"]))),
        ("GPU",f"{E_['gpu']} (TEST610 reference GPU: {CANON_GPU}; same: {P['hardware']['same']})"),("TORCH / TRANSFORMERS / GRADIO",f"{E_['torch']} / {E_['transformers']} / {E_['gradio']}"),
        ("ENGINE FILE",f"{P['engine']['file']} · git blob {P['engine']['blob']} · SHA-256 {P['engine']['sha256']}"),("ENGINE SECTION",f"executed up to [6/7] · SHA-256 {P['engine']['exec_sha256']}"),
        ("SOURCE FILES",f"566.py {cfg['source_blobs']['566.py']} · 575.py {cfg['source_blobs']['575.py']} (git blobs verified by 610.py)"),
        ("HELPERS",f"{HELPER_FILE} · SHA-256 {cfg['helper_sha256']}"),("FRAM LOCK",f"layer {cfg['fram_cfg'][0]} · rank {cfg['fram_cfg'][1]} · β {cfg['fram_cfg'][2]} (calibration cases 0–11, {cfg['n_configs']} configurations)"),
        ("CUT3 MEMORY",f"cartridge = H{cfg['CUT']} residual + layers 0–{cfg['CUT']} K/V; {upper_txt(cfg['CUT'],cfg['arch'][0])} consolidate"),
        ("PANEL",f"locked 24-case panel · SHA-256 {cfg['panel_sha']}"),("TEST610 RESULT SHA",TEST610["result_sha"]),("FULL-WEIGHT SHA-256",f"{fz['weight_sha_startup']} (reference {WEIGHT_SHA_EXPECTED})")):a(f"{k_:<30}: {v}")
    a("");a("CASE");a(Dd)
    a(f"case {L['case']:02d} ({L['split']}) · question: {L['question']}");a(f"target cartridge #{L['target']:02d} · expected answer (audit metadata): {L['expected']}")
    a("");a("FRAM RETRIEVAL  [LIVE · engine scores]");a(Dd)
    a(f"exact FRAM kernel: target rank {R['fram']['rank']} of {L['n_cartridges']} · top-5 {R['fram']['top5']} · target score {R['fram']['target_score']:.6f} · best other {R['fram']['best_other_score']:.6f}")
    a(f"locked reference ranking (used for the CUT3 selection of panel questions): target rank {R['ref']['rank']} · top-5 {R['ref']['top5']} · target present: {R['target_present']}")
    a(f"engine order of the CUT3 memory (sorted ids, as in TEST610): {L['engine_order']}")
    for c in L["cartridges"]:a(f"   #{c['id']:02d} · {c['type']:<8} · {c['body_len']:>2} body tokens · {c['fact']}"+("   ← TARGET"if c["is_target"]else""))
    a("");a("CUT3 WRITE + APPEND-ONLY CONSOLIDATION · build_cut3(ids, 3)  [LIVE]");a(Dd)
    for n_,s in enumerate(L["steps"],1):
        a(f"step {n_} · {s['kind']:<6} · cartridge #{s['cartridge_id']:02d} · memory {s['old_len']} → {s['new_len']} tokens (+{s['new_len']-s['old_len']}) · {1000*s['seconds']:.1f} ms")
        for f in s["forwards"]:a(f"         forward: {fw_txt(f)}")
        if s["kind"]=="append":a(f"         previously consolidated rows 0…{s['old_len']-1}: bitwise unchanged = {s['prior_rows_bitwise_unchanged']} · SHA {s['prior_sha_before'][:16]}… → {s['prior_sha_after'][:16]}…")
    M=L["memory"];a(f"final memory: {M['T']} tokens × {M['layers']} layers · tensor {M['shape']} · {M['bytes']} bytes ({mib(M['bytes']):.2f} MiB, {M['dtype']}) · SHA {L['memory_sha'][:24]}…")
    a("");a("SOURCE-FREE QUESTION · answer(memory, question)  [LIVE]");a(Dd)
    a(f"answer  : {L['answer']}");a(f"expected: {L['expected']} · {'CORRECT'if L['correct']else'INCORRECT'} (re-derived: {L['correct_rederived']})")
    a(f"query input = question template only: {L['query_tokens']} token ids over the installed memory ({M['T']} tokens) · generation forwards {L['gen_forwards']} · {1000*L['answer_seconds']:.0f} ms")
    a(f"repeat: identical answer = {L['repeat_identical']} · memory SHA unchanged by querying = {L['memory_unchanged_by_query']}")
    a(f"sealed TEST610 answer for this case: {L['sealed_answer']!r} · live identical: {L['matches_sealed']} (reported, not a gate)")
    a("");a("DISK · save_session → load_session → answer  [LIVE]");a(Dd)
    a(f"file {K['file']} · {K['bytes']} bytes · SHA-256 {K['sha256']} · recomputed after write: {K['sha_recomputed']}")
    a(f"in-memory K/V {K['memory_bytes']} bytes · file overhead {K['bytes']-K['memory_bytes']} bytes · fact text found in the file: {K['fact_text_in_file']} · expected answer string found in the file: {K['gold_in_file']}")
    a(f"reload: ids equal {K['ids_equal']} · K/V bitwise equal in all {M['layers']} layers {K['bit_equal']} · memory SHA equal {K['memory_sha_equal']} · answer after reload {K['answer']!r} · identical {K['answer_identical']}")
    a("");a("NEGATIVE CONTROL · target cartridge removed  [LIVE · diagnostic]");a(Dd)
    if N.get("skipped"):a("skipped: "+N["skipped"])
    else:a(f"memory of {N['ids']} ({N['T']} tokens) · answer {N['answer']!r} · expected answer produced: {N['gold_leak']}")
    if L.get("custom"):cu=L["custom"];a("");a("FREE-TEXT QUESTION (unscored)");a(Dd);a(f"{cu['question']} → {cu['answer']!r} · exact FRAM top-5 {cu['ids']} · query tokens {cu['query_tokens']}")
    a("");a("START-UP IN THIS RUNTIME (610.py procedure, before any run)");a(Dd)
    a(f"FRAM holdout Recall@1 {su['hold_r1']}/12 · Recall@5 {su['hold_r5']}/12 · all Recall@5 {su['all_r5']}/24 · self-retrieval {su['self_r1']}/{su['self_n']} · hub max {su['hub_max']}")
    a(f"reference ranking holdout Recall@1 {su['ref_hold_r1']}/12 · Recall@5 {su['ref_hold_r5']}/12 · baseline CUT3 audit {su['baseline_correct']}/{su['baseline_n']} correct")
    a("");a("SEALED REFERENCES (archived; not produced by this run)");a(Dd)
    f6=TEST610["final"];a(f"TEST610 · {TEST610['decision']} · criteria {f6['criteria_passed']}/{f6['criteria_total']} · holdout {f6['holdout_correct']}/{f6['holdout_total']} · panel {f6['all_panel_correct']}/{f6['all_panel_total']} · {f6['seconds']} s · result {TEST610['result_sha']}")
    a("TEST610 negative controls: "+" · ".join(f"case {r['case']:02d} → {r['raw']!r} leak {r['gold_leak']}"for r in TEST610["negatives"]))
    A=TEST575["arms"];cf=TEST575["counterfactual"];a("TEST575 (different experiment) · "+" · ".join(f"{k} {v}/72"for k,v in A.items())+f" · counterfactual upper K/V changed {cf['upper_kv_changed']}/72 · answer changed {cf['answer_changed']}/72")
    a("");a("FROZEN MODEL");a(Dd)
    for k_ in("weight_sha_startup","sentinel_startup","sentinel_before","sentinel_after","guard_startup","guard_before","guard_after"):a(f"{k_:<18}: {fz[k_]}")
    a(f"trainable tensors={fz['trainable_tensors']} · lora={fz['lora']} · optimizer={fz['optimizer']} · training_mode={fz['training_mode']} · foreign hooks outside engine calls {fz['foreign_hooks_before']}→{fz['foreign_hooks_after']} · hooks on layers/attention/q_proj {len(fz['residual_hooks_before'])}→{len(fz['residual_hooks_after'])}")
    a(fz["full_weight_method"]);a(fz["guard_method"]);a(fz["sentinel_method"]);a(P["instrumentation"]["forward_counter"]);a(P["instrumentation"]["observer"])
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
    finally:
        try:I.release()
        except Exception:pass
def _execute_run(I,ctl,state):
    run_id=ctl["run_id"];run_dir=Path(ctl["run_dir"]);case=int(ctl["case"]);custom=(ctl.get("custom")or"").strip()
    info=I.info;checks=[];tm=Counter();T0=time.perf_counter();state["checks"]=checks
    raw=dict(retrieval=None,trace=None,answer=None,repeat=None,disk=None,reload=None,negative=None,custom=None);state["raw"]=raw
    def chk(name,cond):
        checks.append(name)
        if not cond:raise AuditFail(name)
    run_start_utc=utc_now();say("="*140);say(f"RUN {run_id} · case {case:02d}");say("="*140)
    # ---- 1 integrity ----
    yield ev(1,"Integrity","Engine and helper hashes, source blobs, locked panel, FRAM lock, start-up gates, weight sentinel, all-parameter guard, hooks.")
    I.reset_peak();cfg=I.config();st0=I.status();su=I.startup();fz0=I.frozen();t=time.perf_counter();g_before=I.guard();tm["guard"]+=time.perf_counter()-t
    chk("610.py git blob = repository blob · SHA-256 = embedded reference",cfg["engine_blob"]==ENGINE_BLOB_EXPECTED and cfg["engine_sha256"]==ENGINE_SHA256_EXPECTED)
    chk("RAW COLAB DEMO helper section SHA-256 = embedded reference",cfg["helper_sha256"]==HELPER_SHA256_EXPECTED)
    chk("566.py and 575.py git blobs verified by 610.py = expected",cfg["source_blobs"]==SOURCE_BLOBS_EXPECTED==cfg["source_blobs_measured"])
    chk(f"model, architecture, CUT{CUT_EXPECTED} and {PREFIX_EXPECTED}-token prefix",cfg["MODEL_ID"]==MODEL_ID and tuple(cfg["arch"])==ARCH and cfg["CUT"]==CUT_EXPECTED and cfg["prefix_tokens"]==PREFIX_EXPECTED)
    chk("locked panel (SHA-256) · 96 cartridges · 24 cases",cfg["panel_sha"]==EXPECTED_PANEL_SHA and cfg["n_cartridges"]==N_CART and cfg["n_cases"]==N_CASES)
    chk("FRAM configuration locked on calibration cases 0–11 = (layer 3, rank 0, β 8) · reference M = D = 8192",tuple(cfg["fram_cfg"])==FRAM_LOCK and (cfg["M"],cfg["D"])==(M_REF,D_REF)and cfg["train"]==list(TRAIN_CASES))
    chk("start-up FRAM in this runtime: holdout R@1 12/12 · R@5 12/12 · all R@5 24/24 · self-retrieval 96/96",(su["hold_r1"],su["hold_r5"],su["all_r5"],su["self_r1"])==(12,12,24,96))
    chk("start-up reference ranking 11/12 · 12/12 and baseline CUT3 audit 12/12",(su["ref_hold_r1"],su["ref_hold_r5"],su["baseline_correct"])==(TEST610["reference"]["hold_r1"],TEST610["reference"]["hold_r5"],12))
    chk("full-weight SHA-256 at start-up = reference",info["weight_sha0"]==WEIGHT_SHA_EXPECTED)
    chk("weight sentinel before run = value at initialisation",fz0["sentinel"]==info["sentinel0"]);chk("all-parameter guard before run = value at start-up",g_before==info["guard0"])
    chk("trainable tensors = 0, eval mode, no LoRA, no optimizer",fz0["trainable_tensors"]==0 and not fz0["training"]and not fz0["lora"]and not fz0["optimizer"])
    chk("no foreign hooks outside engine calls · no hooks on decoder layers, attention or q_proj",fz0["foreign_hooks"]==0 and not fz0["residual_hooks"])
    chk("case in 0..23",0<=case<N_CASES)
    tgt=I.target(case);question=I.question(case);expected=I.car(tgt)["gold"];split=split_of(case);strip=I.panel_strings();state["strip"]=strip
    for c in checks:say("   PASS ·",c)
    # ---- 2 FRAM retrieval ----
    yield ev(2,"FRAM retrieval","The engine's exact FRAM kernel scores the question features against all 96 cartridge keys (features captured at start-up); the locked reference ranking selects the five cartridges for CUT3.")
    t=time.perf_counter();fs=I.fram_scores(case);fo=I.rankings(fs);rs=I.ref_scores(case);ro=I.rankings(rs);tm["retrieval"]+=time.perf_counter()-t;shapes=I.fram_shapes(case)
    chk("96 finite exact-FRAM scores and 96 finite reference scores",len(fs)==len(rs)==N_CART and all(math.isfinite(x)for x in fs+rs))
    srow=su["cases"][case];frank=fo.index(tgt)+1;rrank=ro.index(tgt)+1
    chk("exact-FRAM scores recomputed in this run from the start-up features reproduce the start-up ranking (rank and top-5; consistency check)",srow["case"]==case and srow["target"]==tgt and frank==srow["rank"]and fo[:5]==srow["top5"])
    selected=ro[:TOP_K];order=sorted(selected);present=tgt in selected
    chk("five distinct cartridge ids in 0..95 selected by the locked reference ranking",len(set(selected))==TOP_K and all(0<=i<N_CART for i in selected))
    best_other=max(fs[i]for i in range(N_CART)if i!=tgt)
    retrieval=dict(fram=dict(scores=fs,order=fo,rank=frank,top5=fo[:5],top10=fo[:10],target_score=fs[tgt],best_other_score=best_other,margin=fs[tgt]-best_other),
                   ref=dict(scores=rs,order=ro,rank=rrank,top5=selected),target_present=present,shapes=dict(query_rows=shapes["query"][0],dim=shapes["query"][1],key_rows=[k[0]for k in shapes["keys"]]))
    raw["retrieval"]=dict(fram_rank=frank,ref_rank=rrank,selected=selected);sealed_row=TEST610["cases"][case]
    cars=[dict(I.car(ci),is_target=ci==tgt,fram_rank=fo.index(ci)+1,ref_rank=ro.index(ci)+1)for ci in order]
    say(f"   exact FRAM rank {frank} · reference rank {rrank} · selected {selected} · target present {present}")
    # ---- 3 CUT3 write + append-only consolidation ----
    yield ev(3,"CUT3 write + append-only consolidation","build_cut3(): the five selected cartridges are written and appended one by one; every model forward is recorded and prior rows are compared bitwise.")
    P_=cfg["prefix_tokens"];mem,trace,dt=I.build(order);raw["trace"]=trace;tm["build_cut3"]+=dt
    steps,ids_fw=check_build(chk,I,trace,order,P_,"memory")
    T=I.memory_len(mem);chk("final memory length = prefix + all five body lengths",T==steps[-1]["new_len"]==P_+sum(c["body_len"]for c in cars))
    mem_sha=I.memory_sha(mem);chk("final memory SHA-256 = SHA-256 recorded after the last append",mem_sha==steps[-1]["memory_sha"])
    prof=I.memory_profile(mem);chk(f"memory = {ARCH[0]} layers × (K, V) of shape [1, {ARCH[3]}, T, {ARCH[4]}]",prof["layers"]==ARCH[0]and prof["T"]==T and prof["shape"]==[1,ARCH[3],T,ARCH[4]])
    chk("memory bytes = layers × 2 × KV heads × head dim × T × element size",prof["bytes"]==ARCH[0]*2*ARCH[3]*ARCH[4]*T*(2 if prof["dtype"]in("bfloat16","float16")else 4))
    say(f"   memory {T} tokens · 5 steps · prior rows bitwise unchanged at every append · {1000*dt:.0f} ms")
    # ---- 4 source-free question ----
    yield ev(4,"Source-free question","answer(): the question template is processed over the numerical memory; the source text is not given to the model.")
    a1,fw_a,dt=I.ask(mem,question);raw["answer"]=a1;tm["answer"]+=dt;qids=check_query(chk,I,fw_a,question,T,"answer")
    correct=I.hit(a1,expected);cr=hit_ans(a1,expected);chk("correctness re-derived from the answer text = engine hit()",cr==correct)
    a2,fw_a2,dt2=I.ask(mem,question);raw["repeat"]=a2;tm["answer_repeat"]+=dt2
    chk("repeat answer(): same forward pattern and identical answer (greedy decoding)",a2==a1 and fw_a2[0]["ids"]==qids and len(fw_a2)==len(fw_a))
    sha_q=I.memory_sha(mem);chk("memory unchanged by querying (SHA-256 before = after)",sha_q==mem_sha)
    sealed_ans=sealed_row["answer"];say(f"   answer {a1!r} · expected {expected} · {'CORRECT'if correct else'INCORRECT'} · {len(qids)} query tokens over {T}-token memory")
    # ---- 5 disk persistence ----
    yield ev(5,"Save → reload → answer","save_session() writes the CUT3 memory to a file; load_session() reads it back; the K/V tensors are compared bitwise and the question is answered again from the reloaded memory.")
    spath=run_dir/f"{run_id}_CUT3_SESSION.pt";t=time.perf_counter();sinfo=I.save(mem,order,spath);tm["save"]+=time.perf_counter()-t
    fb=spath.read_bytes();fsha=hashlib.sha256(fb).hexdigest()
    chk("session file written: size and SHA-256 recomputed from disk = save_session() report",file_ready(spath)and len(fb)==int(sinfo["bytes"])and fsha==sinfo["sha256"])
    facts=[c["fact"]for c in cars];fact_in=any(f.encode("utf-8")in fb for f in facts);gold_in=expected.encode("utf-8")in fb
    chk("the session file contains none of the five fact texts and not the question text",not fact_in and question.encode("utf-8")not in fb)
    t=time.perf_counter();mem2,ids2=I.load(spath);tm["load"]+=time.perf_counter()-t
    chk("reloaded cartridge ids = saved ids (engine order)",ids2==order)
    beq=I.equal(mem,mem2);chk(f"reloaded K/V bitwise equal to the original in all {ARCH[0]} layers",beq)
    sha2=I.memory_sha(mem2);chk("reloaded memory SHA-256 = original memory SHA-256",sha2==mem_sha)
    a3,fw_a3,dt3=I.ask(mem2,question);tm["answer_reload"]+=dt3;check_query(chk,I,fw_a3,question,T,"answer after reload")
    chk("answer from the reloaded memory = answer before saving",a3==a1)
    disk=dict(file=spath.name,bytes=len(fb),sha256=fsha,sha_recomputed=True,memory_bytes=prof["bytes"],fact_text_in_file=fact_in,gold_in_file=gold_in,question_in_file=False,
              ids_saved=order,ids_loaded=ids2,ids_equal=ids2==order,bit_equal=beq,memory_sha_equal=sha2==mem_sha,answer=a3,answer_identical=a3==a1,forwards=fw_a3,
              query_tokens=len(qids),format="AKBASCORE_CUT3_SESSION_V1",save_seconds=tm["save"],load_seconds=tm["load"],answer_seconds=dt3)
    raw["disk"]=dict(sinfo,file=spath.name);raw["reload"]=a3;I.release(mem2);del mem2
    # ---- 6 controls ----
    yield ev(6,"Negative control","The same question over the CUT3 memory built without the target cartridge (diagnostic; leakage of the expected answer is reported)."+(" Then the free-text question." if custom else""))
    if present:
        rest=sorted(ci for ci in selected if ci!=tgt);memN,traceN,dtN=I.build(rest);tm["build_control"]+=dtN
        check_build(chk,I,traceN,rest,P_,"control");TN=I.memory_len(memN);chk("control memory excludes the target cartridge",tgt not in rest and len(rest)==TOP_K-1)
        aN,fwN,dtN2=I.ask(memN,question);tm["answer_control"]+=dtN2;check_query(chk,I,fwN,question,TN,"control")
        leak=I.hit(aN,expected);negative=dict(ids=rest,T=TN,answer=aN,gold_leak=leak,target_absent=True,memory_bytes=I.memory_bytes(memN));I.release(memN);del memN
    else:negative=dict(skipped="target not in the selected five cartridges; the main memory itself is the target-absent condition",ids=selected,gold_leak=None)
    raw["negative"]=negative;cu=None
    if custom and custom!=question:
        ids_c,rk,fq,dq=I.retrieve_free(custom);tm["retrieve_free"]+=dq
        chk("free-text retrieval: one question-frame forward (instruction prefix + question), no cache, no fact text",len(fq)==1 and fq[0]["kind"]=="ids"and fq[0]["ids"]==I.frame_ids(custom)and not fq[0]["has_past"])
        chk("free-text retrieval: five distinct ids in 0..95 from the exact FRAM kernel",len(set(ids_c))==TOP_K and all(0<=i<N_CART for i in ids_c))
        oc=sorted(ids_c);memC,traceC,dtC=I.build(oc);tm["build_free_text"]+=dtC;check_build(chk,I,traceC,oc,P_,"free text");TC=I.memory_len(memC)
        aC,fwC,dtC2=I.ask(memC,custom);tm["answer_free_text"]+=dtC2;cq=check_query(chk,I,fwC,custom,TC,"free text")
        strip=strip+[v for x in(custom,str(aC))for v in(x," ".join(x.split()),json.dumps(x,ensure_ascii=False)[1:-1])];state["strip"]=strip
        cu=dict(question=custom,answer=aC,ids=ids_c,ranking=rk,T=TC,query_tokens=len(cq),gen_forwards=len(fwC)-1,seconds=dtC2,forwards=fwC,frame_tokens=fq[0]["n"]);raw["custom"]=dict(answer=aC,ids=ids_c)
        I.release(memC);del memC
    # ---- 7 verification ----
    yield ev(7,"Verification","Weight sentinel, all-parameter guard, trainable tensors and hooks after the run.")
    fz1=I.frozen();t=time.perf_counter();g_after=I.guard();tm["guard"]+=time.perf_counter()-t
    chk("all-parameter guard after run = before run = start-up (weights unchanged)",g_after==g_before==info["guard0"]);chk("weight sentinel after run = value at initialisation",fz1["sentinel"]==info["sentinel0"])
    chk("trainable tensors = 0, no foreign hooks and no hooks on layers/attention/q_proj after run",fz1["trainable_tensors"]==0 and fz1["foreign_hooks"]==0 and not fz1["residual_hooks"])
    tm["engine_total"]=time.perf_counter()-T0
    verdict=f"CASE_{case:02d}_{'CORRECT'if correct else'INCORRECT'}_CUT3_FRAM_DISK_RELOAD_VERIFIED_SOURCE_FREE_QUERY"
    # ---- 8 seal ----
    yield ev(8,"Sealing","Payload, manifest, readable log and raw records are written and hashed.")
    peak=I.peak_gib();same_hw=info["gpu"]==CANON_GPU
    live=dict(case=case,split=split,question=question,expected=expected,target=tgt,n_cartridges=N_CART,retrieval=retrieval,selected=selected,engine_order=order,cartridges=cars,
              steps=steps,forwards_build=trace["forwards"],source_reads=len(ids_fw),zero_embed_forwards=sum(1 for f in trace["forwards"]if f["kind"]=="embeds"and f["embeds_zero"]),
              memory=prof,memory_sha=mem_sha,segments=segments(steps,P_),answer=a1,answer_repeat=a2,correct=bool(correct),correct_rederived=bool(cr),repeat_identical=a2==a1,
              memory_unchanged_by_query=sha_q==mem_sha,query_tokens=len(qids),query_token_text=I.decode_each(qids),gen_forwards=len(fw_a)-1,forwards_answer=fw_a,answer_seconds=dt,
              disk=disk,negative=negative,custom=cu,sealed_answer=sealed_ans,sealed_correct=bool(sealed_row["correct"]),matches_sealed=norm_ans(a1)==norm_ans(sealed_ans),
              session=[dict(r,correct=(bool(correct)if r.get("correct")is None else bool(r["correct"])))for r in(ctl.get("session")or[])])
    P={"schema":"akbascore.mam.bc.qwen.cut3_fram.demo.run.v1","demo":DEMO_TITLE,"author":AUTHOR,"organisation":ORG,"date":DATE_ISO,"place":AUTHOR_PLACE,"copyright":COPYRIGHT,
       "license":{"name":LICENSE_NAME,"url":LICENSE_URL},"priority_record":PRIORITY,"paradigm":PARADIGM,"discovery":DISCOVERY,"question":SCI_QUESTION,"core_message":CORE_MESSAGE,
       "earlier_records":{"v1.0":PRIOR_PUB,"v1.1":QWEN_PUB,"note":"cited by DOI only; no value of an earlier record is used as a result here"},
       "run_id":run_id,"run_start_utc":run_start_utc,"run_end_utc":utc_now(),"verdict":verdict,
       "model":{"id":cfg["MODEL_ID"],"arch":info["arch"],"dtype":info["dtype"],"attn":info["attn"],"params":info["params"]},
       "environment":{k_:info[k_]for k_ in("gpu","gpu_total_gib","torch","transformers","gradio","python","platform")},"hardware":{"reference_gpu":CANON_GPU,"this_run_gpu":info["gpu"],"same":bool(same_hw)},
       "engine":{"file":info["engine_file"],"sha256":cfg["engine_sha256"],"blob":cfg["engine_blob"],"exec_sha256":cfg["engine_exec_sha256"],"tail_sha256":info["engine_tail_sha256"],"helper_sha256":cfg["helper_sha256"],
                 "raw_demo_file_sha256":info["raw_demo_file_sha256"],"config":cfg,"status_before":st0,"init_seconds":info["init_seconds"],"instrument":I.kind,
                 "calls":"build_cut3(ids, 3) → validate_cut3 → answer(memory, q); save_session / load_session / compare_mem_exact; pair_scores; PRED (locked reference ranking)",
                 "source":ENGINE_URL,"log":ENGINE_LOG_URL,"base":BASE_URL,"lock_program":LOCK_URL},
       "instrumentation":{"forward_counter":info["forward_counter"],"observer":info["observer"]},"startup":su,"live":live,
       "frozen":{"weight_sha_startup":info["weight_sha0"],"weight_sha_reference":WEIGHT_SHA_EXPECTED,"full_weight_method":info["full_weight_method"],"sentinel_startup":info["sentinel0"],"sentinel_before":fz0["sentinel"],"sentinel_after":fz1["sentinel"],"guard_startup":info["guard0"],"guard_before":g_before,"guard_after":g_after,
                 "trainable_tensors":fz1["trainable_tensors"],"lora":fz1["lora"],"optimizer":fz1["optimizer"],"training_mode":fz1["training"],"foreign_hooks_before":fz0["foreign_hooks"],
                 "foreign_hooks_after":fz1["foreign_hooks"],"residual_hooks_before":fz0["residual_hooks"],"residual_hooks_after":fz1["residual_hooks"],"guard_method":info["guard_method"],"sentinel_method":info["sentinel_method"]},
       "archived":{"test610":TEST610,"test575":TEST575,"arm_description":ARM_DESC,"note":"Sealed reference values; not produced or recomputed by this run. TEST575 is a different experiment."},
       "archived_gold":{str(r["case"]):I.car(r["target"])["gold"]for r in TEST610["cases"]},"scope":SCOPE,"checks_pre_seal":list(checks),"timing":dict(tm),"gpu":{"peak_allocated_gib":peak}}
    P=jsafe(P);allow=(info["gpu"],CANON_GPU)
    hits=claim_scan("payload",strip_words(json.dumps(P,ensure_ascii=False),strip),allow=allow)
    chk("claim-discipline and foreign-model scan of the payload: 0 hits",not hits);P["claim_scan"]={"rules":len(CLAIM_RULES)+1+len(LEAK_RULES),"hits":0,"scope":"payload JSON text; panel names and free text excluded; earlier-record citation exempt"}
    pb=canon(P);sha=hashlib.sha256(pb).hexdigest();payload_name=f"{run_id}_FULL_RUN_LOG_payload.json";pp=run_dir/payload_name;t_seal=time.perf_counter();pp.write_bytes(pb);say("payload sealed:",sha)
    chk("payload SHA-256 verification after write",hashlib.sha256(pp.read_bytes()).hexdigest()==sha)
    names=dict(payload=payload_name,manifest=f"run_manifest_{run_id}.json",txt=f"{run_id}_READABLE_RUN_LOG.txt",summary=f"{run_id}_SUMMARY.json",jsonl=f"{run_id}_RAW_RECORDS.jsonl",images=f"figures_manifest_{run_id}.json",zip=f"AKBASCORE_MAM_BC_QWEN_EVIDENCE_{run_id}.zip")
    mp=run_dir/names["manifest"];sealed_utc=utc_now()
    manifest={"demo":DEMO_TITLE,"author":AUTHOR,"organisation":ORG,"date":DATE_ISO,"copyright":COPYRIGHT,"license":LICENSE_URL,"priority_record":PRIORITY,"run_id":run_id,"payload_file":payload_name,
              "payload_sha256":sha,"payload_bytes":len(pb),"sealed_utc":sealed_utc,"verdict":verdict,"engine_sha256":cfg["engine_sha256"],"engine_blob":cfg["engine_blob"],"helper_sha256":cfg["helper_sha256"],
              "test610_result_sha":TEST610["result_sha"],"test575_result_sha":TEST575["result_sha"],"weight_sha_startup":info["weight_sha0"],"model":MODEL_ID,
              "cut3_session_file":spath.name,"cut3_session_sha256":fsha,"cut3_session_bytes":len(fb),
              "canonicalization":"json.dumps(sort_keys=True, separators=(',',':'), ensure_ascii=False, allow_nan=False), UTF-8",
              "note":"Artifact integrity seal: confirms the payload is unchanged since sealing. It is not a scientific proof and not a third-party verification."}
    mp.write_text(json.dumps(manifest,indent=2,sort_keys=True,ensure_ascii=False),encoding="utf-8")
    summary={"run_id":run_id,"verdict":verdict,"case":case,"split":split,"question":question,"answer":a1,"expected":expected,"correct":bool(correct),"target":tgt,
             "fram_rank":frank,"reference_rank":rrank,"selected":selected,"engine_order":order,"memory_tokens":T,"memory_bytes":prof["bytes"],"memory_growth":[s["new_len"]for s in steps],
             "prior_rows_bitwise_unchanged":[s["prior_rows_bitwise_unchanged"]for s in steps[1:]],"session_file":spath.name,"session_bytes":len(fb),"session_sha256":fsha,"reload_bit_equal":beq,
             "reload_answer_identical":a3==a1,"negative_control":{k_:negative.get(k_)for k_ in("ids","answer","gold_leak","skipped")},"weights_unchanged":True,"payload_sha256":sha,"model":MODEL_ID,
             "sealed_TEST610_answer":sealed_ans,"live_answer_matches_sealed":norm_ans(a1)==norm_ans(sealed_ans)}
    (run_dir/names["summary"]).write_text(json.dumps(jsafe(summary),indent=2,ensure_ascii=False),encoding="utf-8")
    lines=[{"op":"retrieval","case":case,"fram_scores":fs,"reference_scores":rs,"selected":selected}]
    lines+=[{"op":"step","n":n_+1,"kind":s["kind"],"cartridge_id":s["cartridge_id"],"old_len":s["old_len"],"new_len":s["new_len"],"prior_rows_bitwise_unchanged":s["prior_rows_bitwise_unchanged"],
            "memory_sha":s["memory_sha"],"forwards":s["forwards"]}for n_,s in enumerate(steps)]
    lines+=[{"op":"answer","question":question,"answer":a1,"forwards":fw_a},{"op":"save","file":spath.name,"bytes":len(fb),"sha256":fsha},{"op":"reload_answer","answer":a3,"bit_equal":beq,"forwards":fw_a3},
            {"op":"negative_control",**{k_:v for k_,v in negative.items()}},{"op":"memory_profile","k_norm":prof["k_norm"],"v_norm":prof["v_norm"]}]
    if cu:lines.append({"op":"free_text","question":cu["question"],"answer":cu["answer"],"ids":cu["ids"],"forwards":cu["forwards"]})
    (run_dir/names["jsonl"]).write_text("\n".join(json.dumps(jsafe(x),ensure_ascii=False)for x in lines)+"\n",encoding="utf-8")
    (run_dir/names["txt"]).write_text(make_txt(P,sha,names,sealed_utc),encoding="utf-8");seal_s=time.perf_counter()-t_seal
    # ---- 9 figures ----
    yield ev(9,"Rendering figures",f"{N_FIGS} figures at 300 dpi (JPEG) with vector PDF copies; every printed value is checked against the sealed payload.")
    ctx={"P":P,"sha":sha,"run_id":run_id,"payload_name":payload_name,"allow":allow,"strip":strip,"N":N_FIGS,"audit":[],"pdfs":[],"checks":len(checks)}
    t_r=time.perf_counter();imgs=render_all(ctx,run_dir);render_s=time.perf_counter()-t_r;entries=[]
    for n_,(p,cap)in enumerate(imgs,1):
        with Image.open(p)as im:w_,h_=im.size
        q_=ctx["pdfs"][n_-1];entries.append({"index":n_,"jpg":p.name,"pdf":q_.name,"title":cap,"width_px":w_,"height_px":h_,"dpi":OUT_DPI,"jpg_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"pdf_sha256":hashlib.sha256(q_.read_bytes()).hexdigest()})
    imf=run_dir/names["images"]
    imf.write_text(json.dumps({"run_id":run_id,"payload_sha256":sha,"verdict":verdict,"count":len(entries),"render_seconds":render_s,"figures":entries,"figure_text_audit":ctx["audit"],
                               "note":"Per-file SHA-256 is an artifact integrity seal. Figure text was checked against the sealed payload and scanned for claim discipline before saving."},indent=2,ensure_ascii=False),encoding="utf-8")
    # ---- 10 package ----
    yield ev(10,"Packaging","ZIP with figures, PDFs, payload, manifest, logs, raw records and the CUT3 session file; post-seal audit.")
    zp=run_dir/names["zip"];members=[p for p,_ in imgs]+list(ctx["pdfs"])+[imf,mp,pp,run_dir/names["txt"],run_dir/names["summary"],run_dir/names["jsonl"],spath]
    for m in members:
        if m.suffix in(".json",".txt",".jsonl"):chk(f"claim-discipline and foreign-model scan of {m.name}: 0 hits",not claim_scan(m.name,strip_words(m.read_text(encoding="utf-8"),strip),allow=allow,numeric=m.suffix==".txt"))
    with zipfile.ZipFile(zp,"w",compression=zipfile.ZIP_DEFLATED)as z:
        for m in members:z.write(m,arcname=m.name)
    post=post_seal_audit(imgs,ctx["pdfs"],zp,[m.name for m in members],pp,sha,run_dir/names["txt"],mp,verdict,spath,fsha)
    I.release(mem);del mem
    say(f"{run_id}: {len(checks)} pre-seal + {len(post)} post-seal checks = {len(checks)+len(post)}/{len(checks)+len(post)} PASS · figures {render_s:.1f}s · seal {seal_s:.2f}s")
    say("VERDICT :",verdict);say("PAYLOAD :",sha);say("ZIP     :",zp);say("="*140)
    ledger=dict(run_id=run_id,utc=sealed_utc,case=case,split=split,question=question,answer=a1,expected=expected,correct=bool(correct),memory_tokens=T,selected=order,
                session_sha256=fsha,reload_bit_equal=beq,negative_leak=negative.get("gold_leak"),weights_unchanged=True,verdict=verdict,payload_sha256=sha)
    return dict(run_id=run_id,run_dir=run_dir,P=P,sha=sha,imgs=imgs,pdfs=ctx["pdfs"],zip=zp,payload=pp,txt=run_dir/names["txt"],manifest=mp,images_manifest=imf,summary=run_dir/names["summary"],
                jsonl=run_dir/names["jsonl"],session=spath,checks=len(checks)+len(post),verdict=verdict,ledger=ledger)
# ---------------- optional LIVE replay of the TEST610 acceptance stage (610.py [6/7]–[7/7] verbatim; reported separately) ----------------
def replay_report(I,run_id,run_dir,progress=None):
    info=I.info;t0=time.perf_counter();start=utc_now();g0=I.guard()
    if g0!=info["guard0"]:raise AuditFail("all-parameter guard before replay differs from start-up")
    R=I.acceptance_replay(progress);g1=I.guard()
    if g1!=g0:raise AuditFail("all-parameter guard changed during replay")
    rows=R["cases"];sealed={r["case"]:r for r in TEST610["cases"]}
    if len(rows)!=24:raise AuditFail("replay did not report 24 cases")
    score=sum(int(bool(r.get("correct")))for r in rows);hold=sum(int(bool(r.get("correct")))for r in rows if r.get("split")=="HOLD")
    if score!=R["summary"]["all_panel_correct"]or hold!=R["summary"]["holdout_correct"]:raise AuditFail("replay scores do not re-derive from its rows")
    cmp_=[dict(case=r["case"],split=r.get("split"),answer=r.get("answer"),correct=bool(r.get("correct")),sealed_answer=sealed[r["case"]]["answer"],sealed_correct=sealed[r["case"]]["correct"],
               identical=norm_ans(r.get("answer",""))==norm_ans(sealed[r["case"]]["answer"]))for r in rows]
    same=sum(int(c["identical"])for c in cmp_);crit=R["criteria"];stdout_name=f"{run_id}_ENGINE_STDOUT.txt";(run_dir/stdout_name).write_text(R["stdout"],encoding="utf-8")
    eng_json=None
    if R.get("out_path")and Path(R["out_path"]).is_file():
        eng_json=run_dir/Path(R["out_path"]).name;shutil.copy(R["out_path"],eng_json)
    out={"schema":"akbascore.mam.bc.qwen.test610_acceptance_replay.v1","model":MODEL_ID,"demo":DEMO_TITLE,"author":AUTHOR,"organisation":ORG,"date":DATE_ISO,"license":LICENSE_URL,"run_id":run_id,"start_utc":start,"end_utc":utc_now(),
         "status":"LIVE REPLAY of the 610.py acceptance stage — separate from the sealed TEST610 record","decision":R["decision"],"criteria_passed":sum(bool(v)for v in crit.values()),"criteria_total":len(crit),
         "criteria":crit,"panel_correct":score,"panel_total":24,"holdout_correct":hold,"holdout_total":12,"answers_identical_to_sealed_TEST610":same,"negatives":R["negatives"],"rebuild":R["rebuild"],"append":R["append"],
         "engine_result_sha":R["result_sha"],"engine_json":None if eng_json is None else eng_json.name,"engine_json_sha256":None if eng_json is None else hashlib.sha256(eng_json.read_bytes()).hexdigest(),
         "engine_stdout":stdout_name,"engine_stdout_sha256":hashlib.sha256((run_dir/stdout_name).read_bytes()).hexdigest(),"engine_tail_sha256":info["engine_tail_sha256"],
         "engine_sha256":ENGINE_SHA256_EXPECTED,"engine_blob":ENGINE_BLOB_EXPECTED,"guard_before":g0,"guard_after":g1,"weights_unchanged":g1==g0==info["guard0"],"gpu":info["gpu"],"seconds":round(time.perf_counter()-t0,2),
         "rows":cmp_,"sealed_reference":{"name":"TEST610","decision":TEST610["decision"],"final":TEST610["final"],"result_sha":TEST610["result_sha"]}}
    pb=canon(out);sha=hashlib.sha256(pb).hexdigest();jp=run_dir/f"{run_id}_ACCEPTANCE_REPLAY.json";jp.write_bytes(pb)
    S="="*120;txt=[S,f"{DEMO_TITLE} — {MODEL_FAMILY} · LIVE REPLAY OF THE TEST610 ACCEPTANCE STAGE (610.py [6/7]–[7/7], verbatim)",f"{DISCOVERY} · {ORG} · {DATE_TXT} · {LICENSE_NAME} · {LICENSE_URL}",S,
         "This replay is a LIVE measurement in this runtime. It is reported separately from the sealed TEST610 record and does not replace it.",
         f"run {run_id} · {start} → {out['end_utc']} · {out['seconds']} s · GPU {info['gpu']}",
         f"LIVE decision {out['decision']} · criteria {out['criteria_passed']}/{out['criteria_total']} · panel {score}/24 · holdout {hold}/12 · answers identical to sealed TEST610: {same}/24",
         f"sealed TEST610: {TEST610['decision']} · {TEST610['final']['criteria_passed']}/{TEST610['final']['criteria_total']} · panel {TEST610['final']['all_panel_correct']}/24 · result SHA {TEST610['result_sha']}",
         f"weights unchanged (all-parameter guard before = after = start-up): {out['weights_unchanged']} · engine stdout {stdout_name} (verbatim)",f"payload SHA-256 {sha}","-"*120]
    txt+=[f"criterion {k:<34} {'PASS'if v else'FAIL'}"for k,v in crit.items()]+["-"*120]
    txt+=[f"case {c['case']:02d} {str(c['split']):<4} answer {c['answer']!r} {'CORRECT'if c['correct']else'INCORRECT'} · sealed {c['sealed_answer']!r} {'=' if c['identical'] else '≠'}"for c in cmp_]
    txt+=["-"*120]+[f"negative control case {r.get('case')} → {r.get('raw')!r} · expected answer produced: {r.get('gold_leak')}"for r in R["negatives"]]+[S]
    tp=run_dir/f"{run_id}_ACCEPTANCE_REPLAY.txt";tp.write_text("\n".join(txt),encoding="utf-8")
    for p_ in(jp,tp):
        h_=claim_scan(p_.name,strip_words(p_.read_text(encoding="utf-8"),I.panel_strings()),allow=(info["gpu"],CANON_GPU),numeric=p_.suffix==".txt")
        if h_:raise AuditFail(f"claim-discipline hit in {p_.name}: {h_[:3]}")
    return dict(out=out,sha=sha,json=jp,txt=tp,stdout=run_dir/stdout_name)
#<<CORE_END>>
#<<FIGURES_BEGIN>>
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":.8,"axes.edgecolor":"#374151","axes.labelcolor":"#111827","xtick.color":"#374151","ytick.color":"#374151",
                     "xtick.labelsize":8,"ytick.labelsize":8,"axes.labelsize":9,"axes.titlesize":9.5,"axes.titleweight":"bold","axes.titlelocation":"left","pdf.fonttype":42,"ps.fonttype":42,
                     "legend.fontsize":8,"legend.frameon":False,"mathtext.fontset":"dejavusans"})
FIG_W,FIG_H=13.333,7.5;LAYOUT_DPI=100;OUT_DPI=300
INK,INK2,MUTED,RULE,GRIDC,PANEL="#111827","#374151","#6B7280","#D1D5DB","#E5E7EB","#F9FAFB"
BLUE,BLUE_L,ORANGE,ORANGE_L,TEAL,TEAL_L,GRAYPT,AMBER_L,AMBER,RED="#1D4ED8","#DBEAFE","#C2410C","#FFEDD5","#0F766E","#CCFBF1","#9CA3AF","#FEF3C7","#B45309","#B91C1C"
VIOLET,VIOLET_L="#6D28D9","#EDE9FE"
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
    hits=claim_scan(nm,strip_words(re.sub(r"\s*\n\s*"," ",blob),ctx.get("strip",[])),allow=ctx["allow"],numeric=True)
    if hits:plt.close(fig);raise AuditFail(f"CLAIM-DISCIPLINE HIT in {nm}: {hits[:3]}")
    pdf=Path(path).with_suffix(".pdf")
    fig.savefig(pdf,format="pdf",facecolor="white",metadata={"Title":desc,"Author":f"{AUTHOR} · {ORG}","Subject":f"{DEMO_TITLE} · run {ctx['run_id']} · payload SHA-256 {ctx['sha']}","Creator":DEMO_SHORT,
                "Keywords":f"model {MODEL_ID}; engine 610.py git blob {ENGINE_BLOB}; engine sha256 {ENGINE_SHA256}; TEST610 result {TEST610['result_sha']}; license {LICENSE_URL}"})
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
    fig.text(.5,.010,mt(f"live run {ctx['run_id']} · {MODEL_FAMILY} · 610.py engine (blob {ENGINE_BLOB[:12]}…; not TEST610 itself) · TEST610 result {TEST610['result_sha'][:12]}… · payload {ctx['sha'][:12]}… · Fig. {k}/{ctx['N']} · {COPYRIGHT}"),
             ha="center",va="center",fontsize=6.2,color=MUTED,family=MONO)
def clean(ax,grid=True):
    for s in("top","right"):ax.spines[s].set_visible(False)
    if grid:ax.grid(True,color=GRIDC,lw=.5);ax.set_axisbelow(True)
def frame(fig,x,y,w,h,ec=INK2,fc="white",lw=.9,ls="-",z=-1):
    fig.add_artist(Rectangle((x,y),w,h,transform=fig.transFigure,facecolor=fc,edgecolor=ec,lw=lw,ls=ls,zorder=z))
def block(fig,x,y,w,h,stage,title,body,ec=INK2,fc="white",mono=True,bfs=8.4,tfs=9.6,bmin=6.4):
    frame(fig,x,y,w,h,ec,fc);frame(fig,x,y+h-.032,w,.032,ec,PANEL if fc=="white"else fc,lw=.9)
    fig.text(x+.007,y+h-.016,stage,fontsize=7,weight="bold",color=MUTED,va="center")
    fig.text(x+.007,y+h-.052,mt(title),fontsize=tfs,weight="bold",color=INK,va="center")
    if body:fit_text(fig,x+.007,y+.008,w-.014,h-.078,body,fs_max=bfs,fs_min=bmin,family=MONO if mono else None,color=INK2)
def arrow(fig,x0,y0,x1,y1,color=INK2,lw=1.1,ms=11,cs="arc3"):fig.add_artist(FancyArrowPatch((x0,y0),(x1,y1),transform=fig.transFigure,arrowstyle="-|>",mutation_scale=ms,lw=lw,color=color,connectionstyle=cs,zorder=6))
def L_(ctx):return ctx["P"]["live"]
def short(s,n):s=str(s);return s if len(s)<=n else s[:n-1]+"…"
def uwrap(s,n,ind=""):
    """Wrap an identifier at underscores into lines of at most n characters."""
    out=[];cur=""
    for part in str(s).split("_"):
        piece=(cur+"_"+part)if cur else part
        if len(piece)>n and cur:out.append(cur+"_");cur=part
        else:cur=piece
    out.append(cur);return"\n".join(ind+x for x in out)
def okc(b):return TEAL if b else ORANGE
def yn(b):return"yes"if b else"no"
# ------------------------------------------------------------------ FIG 1 · complete system architecture
def f1(ctx,k,path):
    P=ctx["P"];L=L_(ctx);cfg=P["engine"]["config"];R=L["retrieval"];M=L["memory"];AR=cfg["arch"];K=L["disk"];li,rk,be=cfg["fram_cfg"];fig=new_fig()
    tc=next(c for c in L["cartridges"]if c["is_target"])if R["target_present"]else None
    header(fig,k,"External Associative Memory for a Frozen Language Model",f"{MODEL_FAMILY} · case {L['case']:02d} ({L['split']}) · from the source fact to the answer · every number below is taken from this run",
           "SYSTEM ARCHITECTURE · LIVE VALUES",INK2)
    W_=.118;G_=.018;X=[.033+i*(W_+G_)for i in range(7)];yb=.545;hb=.25
    frame(fig,X[1]-.008,.395,X[4]+W_-X[1]+.016,.46,ec=TEAL,fc="#F0FDFA",lw=1.0,ls=(0,(4,2)),z=-3)
    fig.text(X[1],.838,"EXTERNAL MEMORY · numbers kept outside the model (removable; saved to disk in Fig. 4)",fontsize=8.2,weight="bold",color=TEAL,va="center")
    frame(fig,X[5]-.008,.395,W_+.016,.46,ec=INK2,fc="#F3F4F6",lw=1.0,z=-3)
    fig.text(X[5],.838,"FROZEN MODEL",fontsize=8.2,weight="bold",color=INK2,va="center")
    roles=["INPUT","WRITE · STORE","STORE","SELECT","ATTACH","ANSWER","OUTPUT"]
    bodies=[f"{L['n_cartridges']} facts · 24 worlds × 4 relation types, each a plain sentence",
            f"PREFIX + fact\n→ 1 frozen forward\nlayer-{li} residual\nfact span {R['shapes']['key_rows'][L['target']]}×{R['shapes']['dim']}\ncentred, unit norm",
            f"{L['n_cartridges']} cartridges\nkey = layer-{li} rows\nconfig ({li}, {rk}, β {be:g})\nselected:\n"+" ".join(f"#{i:02d}"for i in L["engine_order"]),
            f"question in fact\nframe: {R['shapes']['query_rows']} rows\ntarget rank\n exact   {R['fram']['rank']} / {L['n_cartridges']}\n reference {R['ref']['rank']} / {L['n_cartridges']}\ntop-5 → CUT3",
            f"init + 4 appends\n{M['T']} tokens\n{M['layers']} × (K, V)\n[1,{M['shape'][1]},{M['T']},{M['shape'][3]}]\n{mib(M['bytes']):.2f} MiB {M['dtype']}\nfile {K['bytes']:,} B",
            f"{AR[0]} layers\nhidden {AR[1]}\nweights frozen\ninput at answer:\n{L['query_tokens']} template\ntoken ids only",None]
    titles=["Source facts","Native representation","Cartridge bank","FRAM retrieval","CUT3 memory","Frozen Qwen","Answer"]
    cols=[INK2,TEAL,TEAL,BLUE,TEAL,INK2,BLUE if L["correct"]else ORANGE]
    for i in range(7):
        if i==6:continue
        block(fig,X[i],yb,W_,hb,roles[i],titles[i],bodies[i],ec=cols[i],mono=i!=0,bfs=8.8,tfs=8.2,bmin=6.6)
    frame(fig,X[6],yb,W_,hb,ec=cols[6],fc=BLUE_L if L["correct"]else ORANGE_L,lw=1.3);frame(fig,X[6],yb+hb-.032,W_,.032,ec=cols[6],fc=PANEL,lw=.9)
    fig.text(X[6]+.007,yb+hb-.016,roles[6],fontsize=7,weight="bold",color=MUTED,va="center");fig.text(X[6]+.007,yb+hb-.052,"Answer",fontsize=8.2,weight="bold",color=INK,va="center")
    fit_text(fig,X[6]+.007,yb+.115,W_-.014,.07,L["answer"],fs_max=15,fs_min=8,color=INK,weight="bold")
    fig.text(X[6]+.007,yb+.095,mt(f"expected {short(L['expected'],14)}"),fontsize=7.8,color=INK2,va="center")
    fig.text(X[6]+.007,yb+.067,"CORRECT"if L["correct"]else"INCORRECT",fontsize=9.6,weight="bold",color=cols[6],va="center")
    fit_text(fig,X[6]+.007,yb+.006,W_-.014,.045,"expected answer = audit metadata, not an input",fs_max=7,fs_min=6.2,color=MUTED)
    for i in range(6):arrow(fig,X[i]+W_+.001,yb+hb*.55,X[i+1]-.001,yb+hb*.55,color=INK2,ms=10)
    lay=["Plain sentences. Each is read by the model when its cartridge is written.","The model's own internal numbers for that sentence.","A shelf of numerical memory cards, outside the model.",
         "The question picks the five best-matching cards.","The five cards are merged into one working memory.","The model reads the memory; it never sees the text again.","Generated by the frozen model."]
    for i in range(7):fit_text(fig,X[i]+.002,.405,W_-.004,.125,lay[i],fs_max=8.6,fs_min=6.6,color=INK)
    frame(fig,.03,.148,.455,.225,ec=RULE,fc=PANEL,lw=.7)
    fit_text(fig,.04,.155,.44,.21,f"WHO DOES WHAT\n• stores: the cartridge bank (FRAM keys of {L['n_cartridges']} facts) and the CUT3 memory (K/V numbers) — data outside the weights\n"
             f"• selects: FRAM — the question's layer-{li} rows are matched against every cartridge key (Fig. 3)\n• answers: frozen {MODEL_FAMILY}; its {AR[0]} layers read the installed CUT3 memory (Figs. 2, 4)",fs_max=9.2,fs_min=6.6,color=INK)
    frame(fig,.5,.148,.47,.225,ec=AMBER,fc=AMBER_L,lw=.8)
    fit_text(fig,.51,.155,.455,.21,"MEASUREMENT LIMITS\n• In this implementation the five selected cartridges are written from their facts when the session memory is built (one forward of the instruction prefix + each fact, its own text only); the bank used for selection holds residual keys.\n"
             "• What is saved and reloaded is the five-cartridge session memory, not a store of all 96 CUT3 cartridges.\n• "+NOT_VALIDATED,fs_max=8.8,fs_min=6.4,color=INK)
    footer(fig,ctx,k,f"Figure {k}. Complete path of one question through the system. Plain reading: sentences become numbers, the numbers are kept outside the model, the question selects the matching numbers, they are merged into one memory, and the unchanged model answers from that memory. "
           f"Technical reading: FRAM keys are the layer-{li} residual rows of each fact (centred by the store mean, unit norm); the panel question selects five cartridges with the locked reference ranking of TEST610; build_cut3 writes H{cfg['CUT']} + layers 0–{cfg['CUT']} K/V per cartridge and consolidates {upper_txt(cfg['CUT'],AR[0])} append-only; answer() receives only the question template over the {M['T']}-token memory.")
    return finish(fig,path,ctx,[f"{M['T']} tokens",f"{L['query_tokens']} template",f"exact   {R['fram']['rank']} / {L['n_cartridges']}",mt(short(L["expected"],14)),f"file {K['bytes']:,} B",f"hidden {AR[1]}"],"System architecture")
# ------------------------------------------------------------------ FIG 2 · CUT3 memory writing and consolidation
def f2(ctx,k,path):
    P=ctx["P"];L=L_(ctx);st=L["steps"];cars=L["cartridges"];M=L["memory"];T=M["T"];cfg=P["engine"]["config"];fig=new_fig();cut=cfg["CUT"];NLm=M["layers"]
    header(fig,k,"Native Memory Construction and Append-Only Consolidation",f"engine build_cut3(ids, {cut}) = init(first, {cut}) → append(memory, next, {cut}) × 4 · every model forward recorded · prior K/V rows compared bitwise after each append","LIVE MEASUREMENT",BLUE)
    ax=fig.add_axes([.055,.565,.3,.27]);ax.set_zorder(3);clean(ax)
    for n_,s in enumerate(st):
        old=s["old_len"];add=s["new_len"]-old;col=SEGC[n_+1]
        ax.bar(n_,old,color="#E5E7EB",edgecolor=INK2,lw=.5,width=.62);ax.bar(n_,add,bottom=old,color=col,width=.62)
        ax.text(n_,s["new_len"]+1.2,f"{s['new_len']}",ha="center",fontsize=8,color=INK)
        if n_:ax.text(n_,old/2,"✓",ha="center",va="center",fontsize=12,color=TEAL,weight="bold")
    ax.set_xticks(range(len(st)));ax.set_xticklabels([f"{s['kind']}\n#{c['id']:02d}"+(" T"if c["is_target"]else"")for s,c in zip(st,cars)],fontsize=7.6)
    ax.set_ylabel("memory rows (tokens)",fontsize=8);ax.set_ylim(0,T*1.18);ax.set_title("memory length after each step (✓ = earlier rows bitwise unchanged)",fontsize=8.2)
    tx=.395;fig.text(tx,.835,"Forward record (instrumentation, this run)",fontsize=9.6,weight="bold",color=INK)
    fig.text(tx,.805,mt(f"{'step':<9}{'write forward':<26}{'consolidation forward':<36}{'prior rows':<12}"),fontsize=7.6,family=MONO,color=MUTED)
    for n_,s in enumerate(st):
        f=s["forwards"];y=.778-n_*.052
        w_=f"ids {f[0]['n']} · no cache";c_=(f"zero-emb {f[1]['n']} · cache {f[1]['past_len']}"if f[1]["has_past"]else f"zero-emb {f[1]['n']} · no cache")
        pr="init"if s["kind"]=="init"else("✓ identical"if s["prior_rows_bitwise_unchanged"]else"✗ CHANGED")
        fig.text(tx,y,mt(f"{n_+1} #{s['cartridge_id']:02d}  {w_:<26}{c_:<36}{pr:<12}"),fontsize=7.7,family=MONO,color=INK,weight="bold"if cars[n_]["is_target"]else"normal")
        sub=(f"   SHA rows 0…{s['old_len']-1}: {s['prior_sha_before'][:12]}… = {s['prior_sha_after'][:12]}… · {1000*s['seconds']:.0f} ms"if s["kind"]=="append"else f"   prefix {cfg['prefix_tokens']} + body of C1 written together ({s['new_len']} rows) · {1000*s['seconds']:.0f} ms")
        fig.text(tx,y-.022,mt(sub),fontsize=6.9,family=MONO,color=INK2)
    A=np.asarray(M["k_norm"],dtype=float);mu=A.mean(axis=1,keepdims=True);sd=A.std(axis=1,keepdims=True);sd[sd==0]=1;Z=(A-mu)/sd
    hx=fig.add_axes([.055,.215,.5,.245]);hx.set_zorder(3)
    im=hx.imshow(Z,aspect="auto",cmap="RdBu_r",vmin=-2.5,vmax=2.5,interpolation="nearest",origin="lower")
    for lab,a_,b_,_ in L["segments"][1:]:hx.axvline(a_-.5,color=INK,lw=.8)
    hx.axhline(cut+.5,color=AMBER,lw=1.4,ls=(0,(4,2)));hx.set_yticks([0,cut,NLm//2,NLm-1]);hx.set_yticklabels(["L0",f"L{cut}",f"L{NLm//2}",f"L{NLm-1}"],fontsize=7)
    hx.set_xlim(-.5,T-.5);hx.set_xticks([L["segments"][i][1]for i in range(len(L["segments"]))]+[T-1]);hx.tick_params(axis="x",labelsize=6.8)
    hx.set_title(mt(f"‖K‖ per row and layer (mean over {M['kv_heads']} KV heads, z-scored per layer) · dashed: top of layers 0–{cut}"),fontsize=8)
    cb=fig.colorbar(im,cax=fig.add_axes([.56,.215,.007,.245]));cb.ax.tick_params(labelsize=6.3);cb.outline.set_linewidth(.5)
    sx=fig.add_axes([.055,.168,.5,.026]);sx.set_zorder(3);sx.set_xlim(-.5,T-.5);sx.set_ylim(0,1);sx.axis("off")
    for n_,(lab,a_,b_,ci)in enumerate(L["segments"]):
        c_=cars[n_-1]if n_ else None;tg=bool(c_ and c_["is_target"])
        sx.add_patch(Rectangle((a_-.5,0),b_-a_,1,color=SEGC[n_],alpha=1 if(tg or n_==0)else .78,lw=0))
        sx.text((a_+b_-1)/2,.5,(("prefix"if n_==0 else f"#{ci:02d}")+(" T"if tg else"")),ha="center",va="center",fontsize=7,color="white",weight="bold")
    bpt=M["bytes"]//max(1,T)
    frame(fig,.605,.168,.365,.36,ec=TEAL,fc=TEAL_L,lw=1.0)
    fit_text(fig,.615,.176,.35,.345,f"TENSOR STRUCTURE (measured)\nmemory = {NLm} layers × (K, V)\neach tensor {M['shape']} · {M['dtype']}\nbytes per token = {NLm}·2·{M['kv_heads']}·{M['head_dim']}·2 = {bpt:,} B\n"
             f"total {M['bytes']:,} B ({mib(M['bytes']):.2f} MiB) for {T} tokens\n\nHOW IT IS WRITTEN (engine code)\nlayers 0–{cut}: K/V of checkpoint(ci, {cut}); keys re-phased by RoPE to their memory positions\n"
             f"{upper_txt(cut,NLm)}: append forward over the existing memory with all-zero input embeddings; the stored H{cut} is injected at the layer-{cut} output\n\n"
             f"• token-id forwards: {L['source_reads']} (prefix + each cartridge's own fact, once)\n• zero-embedding forwards: {L['zero_embed_forwards']}\n• earlier rows unchanged after 4 appends: {all(s['prior_rows_bitwise_unchanged']for s in st[1:])}\n• memory SHA {L['memory_sha'][:20]}…",
             fs_max=8.2,fs_min=6.4,family=MONO,color=INK)
    footer(fig,ctx,k,f"Figure {k}. Live record of the CUT3 construction for case {L['case']:02d}. Step 1 (init) writes cartridge #{st[0]['cartridge_id']:02d} together with the {cfg['prefix_tokens']}-token instruction prefix; steps 2–5 append the next cartridges in engine order. "
           f"In every step the instrumentation sees one token-id forward (the instruction prefix + the new cartridge's own fact, no cache; the prefix rows are discarded on append) and one consolidation forward with all-zero input embeddings over the new rows only. After every append all earlier K/V rows of all {NLm} layers are compared bitwise with the memory before the append. "
           "The heat map shows the measured K norms of the final memory (display only).")
    return finish(fig,path,ctx,[f"{T}",f"{bpt:,} B",f"{M['bytes']:,} B",L["memory_sha"][:20],"✓ identical"if len(st)>1 else"init"],"CUT3 memory writing")
# ------------------------------------------------------------------ FIG 3 · FRAM mathematical retrieval
def f3(ctx,k,path):
    P=ctx["P"];L=L_(ctx);R=L["retrieval"];cfg=P["engine"]["config"];su=P["startup"];li,rk,be=cfg["fram_cfg"];fig=new_fig();tgt=L["target"];N=L["n_cartridges"];T6=TEST610["fram"];T6r=TEST610["reference"]
    header(fig,k,"Associative Retrieval and Mathematical Ranking",f"FRAM over {N} cartridges · locked configuration: layer {li}, rank {rk}, β = {be:g} (chosen on calibration cases 0–11 only) · case {L['case']:02d} ({L['split']}): {short(L['question'],60)}","ENGINE SCORES · THIS CASE",BLUE)
    frame(fig,.03,.49,.45,.37,ec=RULE,fc=PANEL,lw=.7)
    fig.text(.04,.84,"EQUATIONS EXECUTED BY THE ENGINE (610.py)",fontsize=8.4,weight="bold",color=INK2,va="center")
    fig.text(.04,.80,r"(1)  $\tilde h=(h-\mu)\,/\,\Vert h-\mu\Vert$",fontsize=10.5,color=INK,va="center")
    fig.text(.255,.80,mt(f"h = layer-{li} residual row; μ = mean of all stored rows"),fontsize=7.4,color=INK2,va="center")
    fig.text(.04,.735,r"(2)  $s(q,c)=\frac{1}{|q|}\sum_{t\in q}\frac{1}{\beta}\left[\log\sum_{j\in c}e^{\beta\,\tilde q_t\cdot\tilde k_j}-\log|c|\right]$",fontsize=10.5,color=INK,va="center")
    fig.text(.04,.685,mt(f"exact FRAM kernel (pair_scores), β = {be:g}; q = question rows in the fact frame, c = fact-span rows"),fontsize=7.4,color=INK2,va="center")
    fig.text(.04,.637,r"(3)  $\phi(x)=e^{x\Omega-\beta/2}/\sqrt{M},\;F_c=\mathrm{mean}_{j\in c}\,\phi(\tilde k_j),\;U=F^{\top}C$",fontsize=10.2,color=INK,va="center")
    fig.text(.04,.588,r"       $z_t=\phi(\tilde q_t)\,U\,C^{\top},\;\hat s_c=\frac{1}{|q|}\sum_t\log\max(z_{t,c},10^{-12})/\beta$",fontsize=10.2,color=INK,va="center")
    fit_text(fig,.04,.497,.43,.07,f"reference ranking used to select the panel's CUT3 cartridges: Ω ~ N(0, βI) of size {R['shapes']['dim']}×{cfg['M']}, C = random ±1/√D codes, {N}×{cfg['D']}. "
             f"At {N} cartridges U is larger than the features it encodes: a ranking procedure, not a memory saving.",fs_max=7.4,fs_min=6.3,color=INK2)
    frame(fig,.03,.155,.45,.315,ec=BLUE,fc="white",lw=.9)
    fig.text(.04,.452,"RETRIEVAL QUALITY · start-up in this runtime (sealed TEST610 in parentheses)",fontsize=8.4,weight="bold",color=BLUE,va="center")
    rows=[(f"Recall@1 · holdout cases 12–23",f"{su['hold_r1']}/12",f"({T6['hold_r1']}/12)"),(f"Recall@5 · holdout cases 12–23",f"{su['hold_r5']}/12",f"({T6['hold_r5']}/12)"),
          (f"Recall@5 · all 24 cases",f"{su['all_r5']}/24",f"({T6['all_r5']}/24)"),(f"Self-retrieval@1 · each cartridge as query",f"{su['self_r1']}/{su['self_n']}",f"({T6['self_r1']}/96)"),
          (f"Reference ranking · holdout Recall@1",f"{su['ref_hold_r1']}/12",f"({T6r['hold_r1']}/12)"),(f"Reference ranking · holdout Recall@5",f"{su['ref_hold_r5']}/12",f"({T6r['hold_r5']}/12)"),
          (f"Hub: most frequent cartridge in the 24 top-5 lists",f"{su['hub_max']}×",f"({T6['hub_max']}×)")]
    for i,(a_,b_,c_)in enumerate(rows):
        y=.418-i*.033;fig.text(.045,y,mt(a_),fontsize=8,color=INK,va="center");fig.text(.36,y,b_,fontsize=8.6,weight="bold",family=MONO,color=INK,va="center");fig.text(.415,y,c_,fontsize=7.6,family=MONO,color=MUTED,va="center")
    lk=su["locked"];fit_text(fig,.045,.16,.43,.04,f"calibration: {su['n_configs']} configurations (layers {cfg['fram_layers']} × rank {cfg['fram_ranks']} × β {cfg['fram_betas']}) scored on cases 0–11 only → locked {tuple(lk['cfg'])} with hit5 {lk['hit5']}/12, hit1 {lk['hit1']}/12",fs_max=7.2,fs_min=6.2,color=INK2)
    fo=R["fram"]["order"];top=fo[:10];sc=R["fram"]["scores"];ro=R["ref"]["order"]
    ax=fig.add_axes([.605,.54,.36,.255]);ax.set_zorder(3);clean(ax)
    yy=np.arange(len(top))[::-1];vals=[sc[i]for i in top]
    ax.barh(yy,vals,color=[BLUE if i==tgt else"#CBD5E1"for i in top],height=.68)
    lo=min(vals);hi=max(vals);span=(hi-lo)or 1.0;ax.set_xlim(lo-.35*span,hi+.30*span)
    for y_,i in zip(yy,top):ax.text(sc[i]+.02*span,y_,f"{sc[i]:.4f}",va="center",fontsize=7,family=MONO,color=INK)
    ax.set_yticks(yy);ax.set_yticklabels([f"#{i:02d} {fo.index(i)+1:>2} · ref {ro.index(i)+1:>2}"+(" ◀T"if i==tgt else"")for i in top],fontsize=7.2,family=MONO)
    ax.tick_params(axis="x",labelsize=7);ax.set_title(f"top-10 of {N} by exact FRAM score s(q, c) · label: id · exact rank · reference rank",fontsize=8)
    fig.text(.5,.857,mt(f"target #{tgt:02d} · exact rank {R['fram']['rank']} · reference rank {R['ref']['rank']} · score {R['fram']['target_score']:.4f} · margin to best other {R['fram']['margin']:+.4f}"),fontsize=8.4,weight="bold",color=BLUE,va="center")
    fig.text(.5,.832,mt(f"q: {R['shapes']['query_rows']} question rows × {R['shapes']['dim']} · c: {R['shapes']['key_rows'][tgt]} fact rows × {R['shapes']['dim']} (target)"),fontsize=7.6,family=MONO,color=INK2,va="center")
    frame(fig,.5,.155,.47,.315,ec=RULE,fc="white",lw=.8)
    fig.text(.51,.452,"TOP-5 · exact FRAM  vs  locked reference ranking (→ CUT3 memory)",fontsize=8.4,weight="bold",color=INK2,va="center")
    rs=R["ref"]["scores"]
    for j in range(5):
        a_=fo[j];b_=ro[j];y=.418-j*.034;ca=P["live"]["retrieval"]
        fig.text(.515,y,mt(f"{j+1}. #{a_:02d} {sc[a_]:.4f}"+(" T"if a_==tgt else"")),fontsize=8,family=MONO,color=BLUE if a_==tgt else INK,weight="bold"if a_==tgt else"normal",va="center")
        fig.text(.735,y,mt(f"{j+1}. #{b_:02d} {rs[b_]:.4f}"+(" T"if b_==tgt else"")),fontsize=8,family=MONO,color=BLUE if b_==tgt else INK,weight="bold"if b_==tgt else"normal",va="center")
    fit_text(fig,.515,.165,.445,.1,f"target in the selected five: {yn(R['target_present'])} · engine order of the CUT3 memory (sorted ids, as in TEST610): {L['engine_order']} · "
             "free-text questions are ranked with the exact kernel (2) instead",fs_max=7.6,fs_min=6.3,color=INK2)
    footer(fig,ctx,k,f"Figure {k}. How FRAM ranks the {N} cartridges for case {L['case']:02d}. Plain reading: the question and every stored fact are turned into the model's own layer-{li} numbers; the facts whose numbers best match the question rise to the top. "
           f"Technical reading: equation (2) is the engine's pair_scores (a per-question-token soft maximum over the fact rows, averaged over question tokens); equation (3) is the locked reference ranking that TEST610 uses to choose the five panel cartridges. "
           f"All scores shown are the engine's own values for this case, recomputed in this run from the question and cartridge features captured at start-up (one forward per fact and per panel question); a free-text question is captured with a new forward. Recall@1, Recall@5 and self-retrieval are start-up measurements in this runtime; sealed TEST610 values in parentheses.")
    return finish(fig,path,ctx,[f"{su['hold_r1']}/12",f"{su['all_r5']}/24",f"{su['self_r1']}/{su['self_n']}",f"exact rank {R['fram']['rank']}",f"reference rank {R['ref']['rank']}",f"{R['fram']['target_score']:.4f}"],"FRAM retrieval")
# ------------------------------------------------------------------ FIG 4 · disk persistence and source-free answering
def disk_icon(fig,x,y,w,h,col):
    frame(fig,x,y,w,h,ec=col,fc="white",lw=1.1,z=2)
    ax=fig.add_axes([x+w*.12,y+h*.14,w*.5,h*.72]);ax.set_zorder(4);ax.axis("off");ax.set_xlim(-1,1);ax.set_ylim(-1,1);ax.set_aspect("equal")
    ax.add_patch(Circle((0,0),.95,fc="#E5E7EB",ec=col,lw=1.0));ax.add_patch(Circle((0,0),.22,fc="white",ec=col,lw=.8));ax.plot([.25,.85],[.1,.45],color=col,lw=1.6)
def f4(ctx,k,path):
    P=ctx["P"];L=L_(ctx);K=L["disk"];M=L["memory"];T=M["T"];fig=new_fig();ok=L["correct"]
    header(fig,k,"Save → Reload → Answer Without Source Text",f"The {T}-token CUT3 memory of this run ({len(L['engine_order'])} cartridges) is written to a file, read back, compared bit for bit and used again · file operations are real; the disk picture is an analogy","LIVE MEASUREMENT",BLUE)
    W_=.172;G_=.02;X=[.03+i*(W_+G_)for i in range(5)];yb=.615;hb=.235
    items=[("1 · BUILD","CUT3 memory in GPU",f"{T} tokens × {M['layers']} layers\n{M['bytes']:,} B in memory\nSHA {L['memory_sha'][:16]}…\nids {L['engine_order']}"),
           ("2 · SAVE","save_session()",f"torch.save(payload)\nformat {K['format']}\nfile {K['bytes']:,} B\n(K/V + {K['bytes']-K['memory_bytes']:,} B header)"),
           ("3 · FILE ON DISK","session file",f"{short(K['file'],30)}\nSHA-256 {K['sha256'][:16]}…\n  …{K['sha256'][-16:]}\nfact text inside: {yn(K['fact_text_in_file'])}"),
           ("4 · RELOAD","load_session()",f"torch.load(\n  weights_only=True)\nids equal: {yn(K['ids_equal'])}\nK/V bitwise equal in\nall {M['layers']} layers: {yn(K['bit_equal'])}\nmemory SHA equal: {yn(K['memory_sha_equal'])}"),
           ("5 · ANSWER","answer(reloaded, q)",None)]
    for i,(s,t,b)in enumerate(items):
        col=[TEAL,TEAL,AMBER,TEAL,BLUE if ok else ORANGE][i]
        if b is not None:block(fig,X[i],yb,W_,hb,s,t,b,ec=col,bfs=9.4,tfs=9.6)
        if i<4:arrow(fig,X[i]+W_+.002,yb+hb*.5,X[i+1]-.002,yb+hb*.5,color=INK2)
    frame(fig,X[4],yb,W_,hb,ec=BLUE if ok else ORANGE,fc=BLUE_L if ok else ORANGE_L,lw=1.3);frame(fig,X[4],yb+hb-.032,W_,.032,ec=BLUE if ok else ORANGE,fc=PANEL,lw=.9)
    fig.text(X[4]+.007,yb+hb-.016,"5 · ANSWER",fontsize=7,weight="bold",color=MUTED,va="center");fig.text(X[4]+.007,yb+hb-.052,"answer(reloaded, q)",fontsize=9.4,weight="bold",color=INK,va="center")
    fit_text(fig,X[4]+.008,yb+.09,W_-.016,.075,K["answer"],fs_max=17,fs_min=8,color=INK,weight="bold")
    fig.text(X[4]+.008,yb+.072,mt(f"expected {short(L['expected'],16)} · {'CORRECT'if ok else'INCORRECT'}"),fontsize=7.8,weight="bold",color=BLUE if ok else ORANGE,va="center")
    fig.text(X[4]+.008,yb+.035,mt(f"identical to the answer\nbefore saving: {yn(K['answer_identical'])}"),fontsize=8.2,family=MONO,color=INK2,va="center")
    frame(fig,.03,.405,.94,.18,ec=RULE,fc=PANEL,lw=.7)
    fig.text(.04,.562,mt(f"MODEL INPUT AFTER RELOAD · {K['query_tokens']} token ids (question template) · over {T} memory rows · source text given to the model: none"),fontsize=8.4,weight="bold",color=INK2,va="center")
    toks=[t.replace("\n","⏎")for t in L["query_token_text"]];x=.04;y=.515;fig.canvas.draw();rdr=fig.canvas.get_renderer();FW=fig.get_figwidth()*fig.dpi
    for t in toks:
        s_=mt(t if t.strip()else"·");tt=fig.text(x,y,s_,fontsize=9.2,family=MONO,color=INK,va="center",bbox=dict(boxstyle="square,pad=0.25",fc="white",ec=RULE,lw=.5))
        w=tt.get_window_extent(renderer=rdr).width/FW
        if x+w>.96 and x>.041:tt.set_position((.04,y-.045));x=.04;y-=.045
        x+=w+.005
    fr=K["forwards"];fig.text(.04,.425,mt(f"first forward after reload: {fr[0]['n']} token ids · cache {fr[0]['past_len']} · then {len(fr)-1} one-token generation forwards"),fontsize=7.8,family=MONO,color=INK2,va="center")
    frame(fig,.03,.15,.62,.235,ec=TEAL,fc=TEAL_L,lw=.9)
    fit_text(fig,.04,.158,.6,.22,f"CHECKS IN THIS RUN (fail-closed)\n• file size and SHA-256 recomputed from disk = save_session() report\n• the file contains none of the five fact texts and not the question text (byte search)\n"
             f"• reloaded ids = saved ids · K/V tensors bitwise equal in all {M['layers']} layers · memory SHA-256 equal\n• answer after reload = answer before saving ({short(K['answer'],20)})\n"
             f"• memory SHA-256 before saving = after reload: {L['memory_sha'][:20]}…\n• the session file is part of the ZIP evidence package\n• save {1000*K['save_seconds']:.0f} ms · load {1000*K['load_seconds']:.0f} ms · answer {1000*K['answer_seconds']:.0f} ms",fs_max=9.6,fs_min=6.5,color=INK)
    disk_icon(fig,.665,.15,.09,.235,AMBER)
    fit_text(fig,.765,.152,.205,.23,f"ANALOGY: a removable disk that carries the memory.\nREAL OPERATION: torch.save / torch.load of {M['layers']}×(K, V) tensors.\nThe {K['bytes']/1e6:.2f} MB file is the {len(L['engine_order'])}-cartridge session memory, not a store of all {L['n_cartridges']} cartridges.",fs_max=8.8,fs_min=6.3,color=INK2)
    footer(fig,ctx,k,f"Figure {k}. Persistence of the numerical memory. Plain reading: the memory built for this question is written to a file, the file is read back, and the model gives the same answer from it without seeing the text. "
           f"Technical reading: save_session() stores the {M['layers']} K/V pairs of the CUT3 memory with the cartridge ids and source hashes; load_session() restores them (weights_only=True); compare_mem_exact() confirms bit equality; answer() receives only the question-template tokens over the reloaded memory. "
           "File name, size and SHA-256 are those of this run's file, which is included in the ZIP package.")
    return finish(fig,path,ctx,[f"{K['bytes']:,} B",K["sha256"][:16],f"all {M['layers']} layers: {yn(K['bit_equal'])}",f"{K['query_tokens']} token ids",mt(short(L['expected'],16))],"Save, reload, answer")
# ------------------------------------------------------------------ FIG 5 · causal controls and failure analysis
def f5(ctx,k,path):
    P=ctx["P"];L=L_(ctx);N=L["negative"];fig=new_fig();A=TEST575["arms"];cf=TEST575["counterfactual"];F6=TEST610["final"]
    header(fig,k,"Memory Dependence, Counterfactual Controls and Failure Analysis","Does the model answer because of the external memory? Left: live control of this run. Middle: sealed TEST610 acceptance. Right: sealed TEST575 controls of a different experiment.","LIVE + SEALED REFERENCES",INK2)
    fig.text(.03,.865,"(a) LIVE · THIS RUN",fontsize=8.6,weight="bold",color=BLUE,va="center")
    ok=L["correct"];frame(fig,.03,.62,.29,.225,ec=BLUE if ok else ORANGE,fc=BLUE_L if ok else ORANGE_L,lw=1.2)
    pres=L["retrieval"]["target_present"]
    fig.text(.04,.825,mt(f"{'TARGET PRESENT'if pres else'TARGET NOT SELECTED'} · {len(L['engine_order'])} cartridges · {L['memory']['T']} tokens"),fontsize=8,weight="bold",color=INK2,va="center")
    fit_text(fig,.04,.73,.27,.075,L["answer"],fs_max=16,fs_min=8,color=INK,weight="bold")
    fig.text(.04,.70,mt(f"expected {short(L['expected'],16)} · {'CORRECT'if ok else'INCORRECT'}"),fontsize=8.2,weight="bold",color=BLUE if ok else ORANGE,va="center")
    fig.text(.04,.668,mt(f"ids {L['engine_order']}"),fontsize=7.4,family=MONO,color=INK2,va="center")
    fig.text(.04,.640,mt(f"sealed TEST610 answer {short(L['sealed_answer'],14)!r} · same: {yn(L['matches_sealed'])}"),fontsize=7.4,family=MONO,color=INK2,va="center")
    sk=bool(N.get("skipped"));lk=N.get("gold_leak");frame(fig,.03,.385,.29,.215,ec=MUTED if sk else(RED if lk else TEAL),fc="white",lw=1.1)
    if sk:fit_text(fig,.04,.395,.27,.19,"TARGET REMOVED · not applicable: "+N["skipped"],fs_max=8,fs_min=6.5,color=INK2)
    else:
        fig.text(.04,.58,mt(f"TARGET REMOVED · {len(N['ids'])} cartridges · {N['T']} tokens"),fontsize=8,weight="bold",color=INK2,va="center")
        fit_text(fig,.04,.49,.27,.07,N["answer"],fs_max=15,fs_min=8,color=INK,weight="bold")
        fig.text(.04,.46,mt(f"expected answer produced: {yn(lk)}"),fontsize=8.2,weight="bold",color=RED if lk else TEAL,va="center")
        fig.text(.04,.43,mt(f"ids {N['ids']}"),fontsize=7.4,family=MONO,color=INK2,va="center")
        fig.text(.04,.402,"same question · same model · target cartridge absent",fontsize=7.2,color=MUTED,va="center")
    fit_text(fig,.03,.16,.29,.2,f"Reading: {'target cartridge present'if pres else'target cartridge not among the five selected'} → expected answer {'produced'if ok else'NOT produced'} in this run; target cartridge removed → expected answer "+("not testable here"if sk else("produced (leak)"if lk else"not produced"))+
             ". One live control is one observation, not a statistic."+(f"\n\nFree-text question (unscored): {L['custom']['question']} → {L['custom']['answer']} · exact FRAM top-5 {L['custom']['ids']}"if L.get("custom")else""),fs_max=9,fs_min=6.5,color=INK)
    fig.text(.35,.865,"(b) TEST610 · SEALED ACCEPTANCE · 24 panel cases",fontsize=8.6,weight="bold",color=INK2,va="center")
    cs=TEST610["cases"];cw=.0485;chh=.07
    for r in cs:
        i=r["case"];cx=.35+(i%6)*(cw+.004);cy=.775-(i//6)*(chh+.006);good=r["correct"]
        frame(fig,cx,cy,cw,chh,ec=TEAL if good else ORANGE,fc=TEAL_L if good else ORANGE_L,lw=1.2 if not good else .7,z=1)
        fig.text(cx+.004,cy+chh-.015,f"{i:02d} {r['split']}",fontsize=6.6,weight="bold",color=INK2,va="center")
        fig.text(cx+.004,cy+.022,mt(short(r["answer"],9)),fontsize=6.8,family=MONO,color=INK,va="center")
        fig.text(cx+cw-.004,cy+chh-.015,"✓"if good else"✗",fontsize=8,weight="bold",color=TEAL if good else ORANGE,va="center",ha="right")
    fig.text(.35,.525,mt(f"panel {F6['all_panel_correct']}/{F6['all_panel_total']} · holdout {F6['holdout_correct']}/{F6['holdout_total']} · target retrieved {sum(r['present']for r in cs)}/24"),fontsize=8.4,weight="bold",color=INK,va="center")
    bad=[r for r in cs if not r["correct"]]
    fail="; ".join(f"case {r['case']:02d} ({r['split']}): target #{r['target']:02d} was in the memory, answer {r['answer']!r}, expected {P['archived_gold'].get(str(r['case']),'?')!r}"for r in bad)or"none"
    fit_text(fig,.35,.425,.31,.08,"FAILURE (kept visible): "+fail,fs_max=7.8,fs_min=6.3,color=ORANGE)
    fig.text(.35,.395,"negative controls (target removed):",fontsize=7.8,weight="bold",color=INK2,va="center")
    for j,r in enumerate(TEST610["negatives"]):
        fig.text(.35,.365-j*.03,mt(f"case {r['case']:02d} → {short(r['raw'],10)!r} · expected answer produced: {yn(r['gold_leak'])}"),fontsize=7.4,family=MONO,color=INK,va="center")
    fit_text(fig,.35,.16,.31,.07,"Four negative controls show no leakage of the expected answer; four cases do not exclude every alternative causal explanation.",fs_max=7.6,fs_min=6.3,color=MUTED)
    fig.text(.69,.865,"(c) TEST575 · SEALED · DIFFERENT EXPERIMENT",fontsize=8.6,weight="bold",color=AMBER,va="center")
    ax=fig.add_axes([.72,.57,.25,.26]);ax.set_zorder(3);clean(ax);arms=list(A)
    ax.bar(range(len(arms)),[A[a_]for a_ in arms],color=[TEAL,BLUE,BLUE,GRAYPT,AMBER],width=.62)
    for i_,a_ in enumerate(arms):ax.text(i_,A[a_]+1,f"{A[a_]}/72",ha="center",fontsize=7.4,color=INK)
    ax.set_xticks(range(len(arms)));ax.set_xticklabels([a_.replace("_","\n",1)for a_ in arms],fontsize=6.8);ax.set_ylim(0,84);ax.set_ylabel("correct of 72",fontsize=7.6);ax.axhline(72,color=MUTED,lw=.6,ls=(0,(3,3)))
    ax.set_title("24 cases × 3 target positions",fontsize=7.8)
    fit_text(fig,.69,.36,.28,.13,f"counterfactual (same H3, same layers 0–3 K/V, cross-cartridge attention blocked in {upper_txt()}):\nupper-layer K/V changed {cf['upper_kv_changed']}/72 · answer changed {cf['answer_changed']}/72",fs_max=8.4,fs_min=6.3,color=INK)
    fit_text(fig,.69,.16,.28,.18," · ".join(f"{a_} = {ARM_DESC[a_]}"for a_ in arms),fs_max=7.8,fs_min=6.0,color=INK2)
    footer(fig,ctx,k,f"Figure {k}. Causal evidence, kept in three separate sources. (a) This run: the same question over the memory with and without the target cartridge. (b) Sealed TEST610 acceptance on the locked panel: {F6['all_panel_correct']}/{F6['all_panel_total']} answers correct, holdout {F6['holdout_correct']}/12; the single incorrect case is shown, not hidden; four target-removed controls. "
           "(c) Sealed TEST575, a different experiment on the same model and CUT3 mechanism: joint text, incremental and batch consolidation 72/72, native independent K/V 26/72, and the attention-blocking counterfactual. Values of (b) and (c) are reproduced verbatim from their logs, not recomputed.")
    return finish(fig,path,ctx,[f"panel {F6['all_panel_correct']}/{F6['all_panel_total']}",f"answer changed {cf['answer_changed']}/72","72/72",mt(short(L['expected'],16))],"Causal controls")
# ------------------------------------------------------------------ FIG 6 · validation, integrity and reproducibility
def f6(ctx,k,path):
    P=ctx["P"];L=L_(ctx);cfg=P["engine"]["config"];fz=P["frozen"];su=P["startup"];F6=TEST610["final"];fig=new_fig();E_=P["environment"]
    header(fig,k,"Scientific Validation and Reproducible Evidence","Validation chain of this run · the sealed TEST610 acceptance, the start-up reproduction in this runtime and this run are labelled separately","VALIDATION · INTEGRITY",INK2)
    W_=.103;G_=.0154;X=[.03+i*(W_+G_)for i in range(8)];yb=.675;hb=.185
    chain=[("SOURCE","Verified source",f"610.py blob\n{cfg['engine_blob'][:12]}…\n566.py {cfg['source_blobs']['566.py'][:8]}…\n575.py {cfg['source_blobs']['575.py'][:8]}…"),
           ("LOCK","Locked config",f"FRAM {tuple(cfg['fram_cfg'])}\nM = D = {cfg['M']}\npanel SHA\n{cfg['panel_sha'][:12]}…"),
           ("RUN","Live run",f"{short(P['run_id'],22)}\n{P['run_start_utc'][11:19]} UTC\n{short(E_['gpu'],22)}"),
           ("RAW","Raw measurements",f"{len(L['forwards_build'])} build forwards\n{len(L['steps'])} steps\n{L['n_cartridges']} scores ×2\n1 file · 1 control"),
           ("CHECK","Validation",f"{ctx['checks']} pre-seal\nchecks PASS\n(fail-closed)"),
           ("SEAL","Payload",f"SHA-256\n{ctx['sha'][:12]}…\n{ctx['sha'][12:24]}…"),
           ("MANIFEST","SHA-256 manifest",f"payload hash\nverdict\nsession file\n{L['disk']['sha256'][:12]}…"),
           ("OUTPUT","Figures",f"{ctx['N']} × JPEG 300 dpi\n{ctx['N']} × vector PDF\nvalues audited\nagainst payload")]
    for i,(s,t,b)in enumerate(chain):
        block(fig,X[i],yb,W_,hb,s,t,b,ec=[INK2,INK2,BLUE,BLUE,TEAL,TEAL,TEAL,INK2][i],bfs=8.2,tfs=7.9,bmin=6.2)
        if i<7:arrow(fig,X[i]+W_+.001,yb+hb*.45,X[i+1]-.001,yb+hb*.45,color=INK2,ms=9)
    cols=[(.03,"SEALED · TEST610 ACCEPTANCE (610.log)",AMBER,
           f"decision        {TEST610['decision']}\ncriteria        {F6['criteria_passed']}/{F6['criteria_total']} PASS\nholdout         {F6['holdout_correct']}/{F6['holdout_total']} correct\nwhole panel     {F6['all_panel_correct']}/{F6['all_panel_total']} (case 08 incorrect)\n"
           f"self-retrieval  {TEST610['fram']['self_r1']}/96\nFRAM holdout    R@1 {TEST610['fram']['hold_r1']}/12 · R@5 {TEST610['fram']['hold_r5']}/12\nweights         unchanged\ntrainable       0\nhooks           0\nseconds         {F6['seconds']}\n"
           f"result SHA-256\n  {TEST610['result_sha'][:32]}\n  {TEST610['result_sha'][32:]}"),
          (.355,"START-UP · THIS RUNTIME (610.py procedure)",BLUE,
           f"FRAM holdout R@1   {su['hold_r1']}/12\nFRAM holdout R@5   {su['hold_r5']}/12\nall cases R@5      {su['all_r5']}/24\nself-retrieval     {su['self_r1']}/{su['self_n']}\nreference R@1/R@5  {su['ref_hold_r1']}/12 · {su['ref_hold_r5']}/12\n"
           f"baseline CUT3      {su['baseline_correct']}/{su['baseline_n']} correct\nengine lock        {len(ENGINE_LOCK)} checks PASS\nfull-weight SHA-256 = reference\n  {fz['weight_sha_startup'][:32]}\n  {fz['weight_sha_startup'][32:]}"),
          (.68,"THIS RUN · INTEGRITY",TEAL,
           f"all-parameter guard  start-up = before = after\n  {fz['guard_after'][:30]}…\nweight sentinel      unchanged\n  {fz['sentinel_after'][:30]}…\ntrainable tensors    {fz['trainable_tensors']}\nLoRA / optimizer     {fz['lora']} / {fz['optimizer']}\n"
           f"foreign hooks        {fz['foreign_hooks_before']} → {fz['foreign_hooks_after']}\nlayer/attn/q hooks   {len(fz['residual_hooks_before'])} → {len(fz['residual_hooks_after'])}\nanswer {short(L['answer'],18)} · {'correct'if L['correct']else'incorrect'}\nverdict\n"+uwrap(P['verdict'],34,"  "))]
    for x,ttl,col,body in cols:
        frame(fig,x,.25,.29,.405,ec=col,fc="white",lw=.9);fig.text(x+.01,.632,ttl,fontsize=8.6,weight="bold",color=col,va="center")
        fit_text(fig,x+.01,.258,.272,.355,body,fs_max=9.2,fs_min=6.2,family=MONO,color=INK)
    frame(fig,.03,.145,.94,.09,ec=BLUE,fc=BLUE_L,lw=1.0)
    fit_text(fig,.04,.15,.92,.08,f"{DISCOVERY} · {ORG} · {DATE_TXT} · {AUTHOR_PLACE} · earlier public records: DOI {PRIOR_PUB['doi']} ({PRIOR_PUB['date']}) · DOI {QWEN_PUB['doi']} ({QWEN_PUB['date']}) · {LICENSE_NAME} · {LICENSE_URL}\n"
             "SHA-256 values are integrity seals: they show that files are unchanged since sealing. They are not scientific or mathematical proof and not third-party verification.",fs_max=8.2,fs_min=6.4,color=INK)
    footer(fig,ctx,k,f"Figure {k}. Reproducibility record of run {P['run_id']} (verdict {P['verdict']}). Payload SHA-256 {ctx['sha']}. The chain runs from the verified engine source through the locked configuration, the live measurements and the fail-closed checks to the sealed payload, "
           "the manifest and the figures; every live value printed in the figures is checked against the payload before the figure is saved. Sealed TEST610 values are reproduced from 610.log; start-up values were measured in this runtime by the same 610.py procedure.")
    return finish(fig,path,ctx,[TEST610["decision"],f"{F6['criteria_passed']}/{F6['criteria_total']} PASS",f"{F6['all_panel_correct']}/{F6['all_panel_total']} (case 08 incorrect)",fz["weight_sha_startup"][:32],ctx["sha"][:12],PRIOR_PUB["doi"],QWEN_PUB["doi"]],"Validation and reproducibility")
FIGURES=[("fig_01_system_architecture","Fig. 1 · External associative memory for a frozen language model",f1),("fig_02_cut3_construction","Fig. 2 · Native memory construction and append-only consolidation",f2),
         ("fig_03_fram_retrieval","Fig. 3 · Associative retrieval and mathematical ranking",f3),("fig_04_disk_persistence","Fig. 4 · Save → reload → answer without source text",f4),
         ("fig_05_causal_controls","Fig. 5 · Memory dependence, counterfactual controls and failure analysis",f5),("fig_06_validation","Fig. 6 · Scientific validation and reproducible evidence",f6)]
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
MAM_BC_PART2_OK=True
