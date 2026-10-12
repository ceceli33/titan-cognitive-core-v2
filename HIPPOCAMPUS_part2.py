# =====================================================================================================================================
# AKBASCORE MAM — THE SILICON HIPPOCAMPUS · QWEN2.5-7B · PART 2 / 3 · RUN PIPELINE AND FIGURES
# Same demo as PART 1. Run this cell after PART 1 / 3, in the same Colab runtime; then run PART 3 / 3.
# Defines the fail-closed RUN (integrity → WRITE re-check → SSD → READ address → READ load + answer → control → verification →
# sealed payload, logs, ZIP), four figures (JPEG 300 dpi + vector PDF; Fig. 1 also in Turkish). Nothing here changes the engine.
# Discovered and developed by Mustafa Akbaş · AkbasCore AI Teknoloji · 12 October 2026 · AKBASCORE RESEARCH SOFTWARE LICENSE
# =====================================================================================================================================
if not globals().get("MAM_SH_PART1_OK"):raise RuntimeError("PART 1 / 3 has not been run in this runtime. Run PART 1, then PART 2, then PART 3.")
#<<CORE_BEGIN>>
import unicodedata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.patches import Rectangle,FancyArrowPatch,Patch,Circle
from matplotlib.lines import Line2D
from PIL import Image
ROOT=Path("/content/AKBASCORE_MAM_SILICON_HIPPOCAMPUS")if os.path.isdir("/content")else Path("/tmp/AKBASCORE_MAM_SILICON_HIPPOCAMPUS")
ROOT.mkdir(parents=True,exist_ok=True);ALLOWED_PATHS=[str(ROOT)]
N_STAGES=10
def ev(stage,title,body):return dict(stage=stage,title=title,body=body)
def file_ready(p):return p is not None and Path(p).is_file()and Path(p).stat().st_size>0
RUN_PREFIX="MAMSH-"
def prune_runs(keep=4,prefix=RUN_PREFIX):
    runs=sorted([p for p in ROOT.glob(prefix+"*")if p.is_dir()],key=lambda p:p.stat().st_mtime)
    for p in runs[:max(0,len(runs)-keep)]:shutil.rmtree(p,ignore_errors=True)
def mb(b):return b/1e6
def fw_txt(f):
    if f["kind"]=="ids":return f"token ids n={f['n']}"+(f" · cache {f['past_len']}"if f["has_past"]else" · no cache")
    return f"embeddings n={f['n']} ({'all zero'if f['embeds_zero']else'NON-ZERO'})"+(f" · cache {f['past_len']}"if f["has_past"]else" · no cache")
def check_query(chk,I,fw,q,T,label):
    qids=I.query_ids(q)
    chk(f"{label}: model input = question-template tokens only, over the loaded {T}-row cartridge (no fact text, no source text)",len(fw)>=1 and fw[0]["kind"]=="ids"and fw[0]["ids"]==qids and fw[0]["has_past"]and fw[0]["past_len"]==T)
    chk(f"{label}: generation forwards = one new token each, over the growing cache",all(f["kind"]=="ids"and f["n"]==1 and f["has_past"]and f["past_len"]==T+len(qids)+n_ for n_,f in enumerate(fw[1:])))
    return qids
def post_seal_audit(imgs,pdfs,extra,zp,member_names,pp,sha,tp,mp,verdict):
    checks=[]
    def ok(name,cond):
        checks.append(name)
        if not cond:raise AuditFail("POST-SEAL AUDIT FAILED: "+name)
    ok(f"{N_FIGS}/{N_FIGS} figures created (JPEG)",len(imgs)==N_FIGS and all(file_ready(p)for p,_ in imgs));ok(f"{N_FIGS}/{N_FIGS} vector PDF copies",len(pdfs)==N_FIGS and all(file_ready(p)for p in pdfs))
    ok("Turkish Fig. 1 (JPEG + PDF) created",all(file_ready(p)for p in extra))
    bad=[]
    for p in[p for p,_ in imgs]+[x for x in extra if str(x).endswith(".jpg")]:
        with Image.open(p)as im:
            if not(im.format=="JPEG"and im.mode=="RGB"and im.size[0]>=3600):bad.append(Path(p).name)
    ok("all figures are JPEG/RGB at ≥3600 px width",not bad)
    with zipfile.ZipFile(zp)as z:
        ok("ZIP testzip()",z.testzip()is None);nm=z.namelist()
        ok("ZIP is flat (no sub-folders)",all("/"not in n for n in nm));ok("ZIP contents = expected package",sorted(nm)==sorted(member_names))
        jp=sorted(n for n in nm if n.lower().endswith(".jpg")and n.startswith("fig_0")and"_TR_"not in n);ok("figure numbering 01..%02d"%N_FIGS,[n[4:6]for n in jp]==[f"{i:02d}"for i in range(1,N_FIGS+1)])
    ok("payload SHA-256 recomputation",hashlib.sha256(pp.read_bytes()).hexdigest()==sha);ok("payload JSON parses",isinstance(json.loads(pp.read_bytes().decode("utf-8")),dict))
    m=json.loads(mp.read_text(encoding="utf-8"));ok("manifest hash and verdict match payload",m["payload_sha256"]==sha and m["verdict"]==verdict)
    ok("readable log exists and is non-empty",file_ready(tp));ok("ZIP exists and is non-empty",file_ready(zp))
    return checks
def make_txt(P,sha,names,sealed_utc):
    o=[];a=o.append;S="="*140;Dd="-"*140;E_=P["environment"];L=P["live"];cfg=P["engine"]["config"];fz=P["frozen"];su=P["startup"];Wr=L["write"];Sd=L["ssd"];Rd=L["read"];Ct=L["control"]
    a(S);a(f"{DEMO_TITLE} — READABLE RUN LOG");a(DEMO_SUB);a(f"{DISCOVERY} · {ORG} · {DATE_TXT} · {AUTHOR_PLACE}");a(f"{LICENSE_NAME} · {COPYRIGHT} · {LICENSE_URL}");a(S)
    a(PRIORITY);a("Scientific question: "+SCI_QUESTION);a(NAME_NOTE)
    a(f"Derived from {names['payload']} (SHA-256 {sha}). Manifest: {names['manifest']}.")
    a("This is a LIVE run of the locked TEST627 demo engine of this runtime. Start-up values were produced by the same engine in this runtime; sealed 627.log / 626.log values are reproduced verbatim, never recomputed.")
    for k_,v in(("RUN ID",P["run_id"]),("START UTC",P["run_start_utc"]),("END UTC",P["run_end_utc"]),("SEALED UTC",sealed_utc),("VERDICT",P["verdict"]),
        ("MODEL",f"{cfg['MODEL_ID']} · frozen · {P['model']['dtype']} · attention {P['model']['attn']}"),("ARCHITECTURE",arch_txt(tuple(P["model"]["arch"]))),
        ("GPU",f"{E_['gpu']} (TEST627 reference GPU: {CANON_GPU}; same: {P['hardware']['same']})"),("TORCH / TRANSFORMERS / GRADIO",f"{E_['torch']} / {E_['transformers']} / {E_['gradio']}"),
        ("ENGINE",f"{P['engine']['file']} · SHA-256 {P['engine']['sha256']} · git blob {P['engine']['blob']}"),
        ("ENGINE CHAIN",f"626.py {cfg['chain_sha256']['626.py']} · 624.py {cfg['chain_sha256']['624.py']} · 610.py {cfg['chain_sha256']['610.py']}"),
        ("SOURCE FILES",f"566.py {cfg['source_blobs']['566.py']} · 575.py {cfg['source_blobs']['575.py']} (git blobs verified by 610.py)"),
        ("CARTRIDGE BANK",f"{cfg['n_cartridges']} physical CUT3 files ({cfg['n_original']} original + {cfg['n_cartridges']-cfg['n_original']} synthetic facts) · pool {cfg['pool']}"),
        ("ADDRESS INDEX",f"PCA{cfg['rank']} + FAISS IVF{cfg['nlist']} over {cfg['ntok']} native layer-3 positions (label-free)"),
        ("LOCKED READ",f"nprobe {cfg['candidates'][0]} · token_k {cfg['candidates'][1]} · budget {cfg['candidates'][2]} · reranker {cfg['reranker'][0]}"),
        ("ENGINE RESULT",f"{su['result_file']} · SHA-256 {su['result_sha256']}"),("FULL-WEIGHT SHA-256",f"{fz['weight_sha_startup']} (reference {WEIGHT_SHA_EXPECTED})")):a(f"{k_:<30}: {v}")
    a("");a("CASE (predeclared rotation over the 48 new-wording questions of the engine)");a(Dd)
    a(f"case {L['case']:02d} · question: {L['question']}");a(f"fact of the case (used only by WRITE): {L['fact']}");a(f"expected answer (audit metadata; not an input to WRITE, addressing, selection or answering): {L['expected']}")
    a("");a("A · WRITE  [LIVE re-execution of the engine's init(fact, 3) for this fact]");a(Dd)
    for n_,f in enumerate(Wr["forwards"],1):a(f"forward {n_}: {fw_txt(f)}")
    a(f"rows {Wr['T']} = {Wr['prefix_tokens']} instruction tokens + {Wr['fact_tokens']} fact tokens · {Wr['layers']} layers × (K, V) = {Wr['tensors']} tensors {Wr['shape']} {Wr['dtype']} · {Wr['memory_bytes']} bytes · {1000*Wr['seconds']:.1f} ms")
    a(f"start-up file {Wr['file']} · {Wr['file_bytes']} bytes · SHA-256 {Wr['file_sha256']} (manifest match: {Wr['sha_ok']}) · written at start-up in {Wr['startup_write_ms']:.1f} ms")
    a(f"live re-write bitwise identical to the start-up file in all {Wr['layers']} layers: {Wr['bit_equal']} · fact text in file: {Wr['fact_in_file']} · answer word in file: {Wr['gold_in_file']} · question in file: {Wr['question_in_file']}")
    a("");a("B · SSD  [LIVE]");a(Dd)
    a(f"files in pool {Sd['files_in_pool']} · manifest {Sd['manifest']} · size mismatches {len(Sd['size_mismatch'])} · total {Sd['total_bytes']} bytes ({Sd['total_bytes']/1e9:.3f} GB)")
    a(f"file sizes: min {Sd['min']} · median {Sd['median']:.0f} · max {Sd['max']} bytes · SHA-256 of all {Sd['manifest']} files verified by the engine at start-up: {Sd['startup_sha_verified']}")
    a("");a("C · READ — address  [LIVE · engine functions candidate626 / features626 / rerank626, locked PEAK50]");a(Dd)
    for f in Rd["frame_forwards"]:a(f"question frame forward: {fw_txt(f)} (instruction prefix + question only)")
    a(f"{Rd['query_rows']} question vectors → PCA{cfg['rank']} → IVF{cfg['nlist']} (nprobe {cfg['candidates'][0]}, token_k {cfg['candidates'][1]}) → touched {Rd['touched']} of {cfg['n_cartridges']} cartridges → {len(Rd['candidates'])} candidates → PEAK50")
    for n_,c in enumerate(Rd["candidates"],1):a(f"   {n_:>2}. #{c['id']:04d} · PEAK50 {c['score']:.6f} · FRAM {c['fram']:.6f} · peak {c['peak']:.6f}"+("   ← SELECTED"if n_==1 else"")+("   (expected cartridge, audit)"if c["id"]==L["target"]else""))
    a(f"timing: question encode {Rd['encode_ms']:.1f} ms · FAISS {Rd['faiss_ms']:.2f} ms · TOP2 {Rd['top2_ms']:.2f} ms · PEAK50 {Rd['rerank_ms']:.2f} ms · address total {Rd['address_ms']:.1f} ms")
    a(f"selected #{Rd['selected']:04d} · expected cartridge among the candidates: {Rd['target_in_candidates']} · selected = expected: {Rd['selected_correct']}")
    a("");a("D · READ — load + answer  [LIVE · sha627 + torch.load + answer(memory, question)]");a(Dd)
    a(f"file {Rd['file']} · SHA-256 = manifest {Rd['sha_ok']} · load {Rd['load_ms']:.1f} ms · {Rd['T']} rows")
    for f in Rd["answer_forwards"][:1]:a(f"answer forward 1: {fw_txt(f)} — question template only")
    a(f"answer  : {L['answer']}");a(f"expected: {L['expected']} · {'CORRECT'if L['correct']else'INCORRECT'} (re-derived: {L['correct_rederived']}) · {Rd['answer_ms']:.0f} ms · generation forwards {Rd['gen_forwards']}")
    a(f"start-up engine row for this question: selected #{L['startup_row']['selected']} · answer {L['startup_row']['answer']!r} · live identical: selection {L['matches_startup_selection']} · answer {L['matches_startup_answer']} (reported, not a gate)")
    a("");a("E · CONTROL — runner-up file  [LIVE · diagnostic]");a(Dd)
    if Ct.get("skipped"):a("skipped: "+Ct["skipped"])
    else:a(f"runner-up #{Ct['id']:04d} ({Ct['file']}) · answer {Ct['answer']!r} · expected answer produced: {Ct['gold_produced']} · runner-up is the expected cartridge: {Ct['is_target']}")
    a("");a("START-UP IN THIS RUNTIME (TEST627 demo engine, before any RUN)");a(Dd)
    for p_ in("ORIGINAL","ALTERNATE"):
        s_=su["summary"][p_];a(f"{p_:<9}: selected {s_['selected_correct']}/{s_['cases']} · answered {s_['answer_correct']}/{s_['cases']} · errors {s_['errors']} · address p50 {s_['address_p50_ms']} ms · load p50 {s_['load_p50_ms']} ms · answer p50 {s_['answer_p50_ms']} ms")
    a("integrity: "+" · ".join(f"{k} {v}"for k,v in su["integrity"].items()))
    st=su["storage"];a(f"storage: {st['cartridges']} files · {st['disk_pool_bytes']} bytes · avg {st['avg_file_bytes']} · tensor payload {st['total_tensor_payload_bytes']} · lossless gzip sample ({st['lossless_gzip_sample_files']} files) ratio {st['gzip_sample_ratio']} (estimate for the full pool, not measured)")
    ix=su["index"];a(f"index: {ix['native_positions']} positions · IVF lists {ix['ivf_lists']} ({ix['list_min']}–{ix['list_max']} per list) · index {ix['index_bytes']} bytes · PCA basis {ix['pca_basis_bytes']} bytes · native layer-3 vectors {ix['native_activation_bytes']} bytes · build {ix['index_build_ms']:.1f} ms")
    a(f"WRITE at start-up: {su['write']['files']} files · p50 {su['write']['ms_p50']:.1f} ms per file · total {su['write']['seconds_total']:.1f} s")
    a("engine limitations (verbatim): "+" | ".join(su["limitations"]))
    a("");a("SEALED REFERENCES (archived; not produced by this run)");a(Dd)
    T7=TEST627_LOG;a(f"TEST627 · {T7['decision']} · 627.log SHA-256 {T7['log_sha256']}")
    for p_,s_ in T7["e2e"].items():a(f"   {p_:<9}: selected {s_['selected_correct']}/48 · answered {s_['answer_correct']}/48 · errors {s_['errors']} · address p50 {s_['address_p50_ms']} ms · load p50 {s_['load_p50_ms']} ms · answer p50 {s_['answer_p50_ms']} ms")
    T6=TEST626_LOG;a(f"TEST626 · own pre-declared rule locked {T6['locked_by_own_rule']} ({T6['e2e']} answered) · FRAM-only new wording {T6['fram_only_new_wording']} · PEAK50 new wording {T6['peak50_new_wording']} · 626.log SHA-256 {T6['log_sha256']}")
    a("");a("FROZEN MODEL");a(Dd)
    for k_ in("weight_sha_startup","sentinel_startup","sentinel_before","sentinel_after","guard_startup","guard_before","guard_after"):a(f"{k_:<18}: {fz[k_]}")
    a(f"trainable tensors={fz['trainable_tensors']} · lora={fz['lora']} · optimizer={fz['optimizer']} · training_mode={fz['training_mode']} · foreign hooks outside engine calls {fz['foreign_hooks_before']}→{fz['foreign_hooks_after']} · hooks on layers/attention/q_proj {len(fz['residual_hooks_before'])}→{len(fz['residual_hooks_after'])}")
    a(fz["full_weight_method"]);a(fz["guard_method"]);a(fz["sentinel_method"]);a(P["instrumentation"]["forward_counter"])
    for t_,k_ in(("DEMONSTRATED IN THIS RUN","demonstrated"),("PRODUCED AT START-UP IN THIS RUNTIME","startup"),("ARCHIVED","archived"),("NOT ESTABLISHED","not_established"),("TECHNICAL NOTES","notes")):
        a("");a(t_);a(Dd)
        for t in P["scope"][k_]:a("• "+t)
    a("");a("PRE-SEAL CHECKS");a(Dd)
    for c_ in P["checks_pre_seal"]:a("PASS · "+c_)
    a("");a("TIMING (seconds)");a(Dd)
    for k_,v in P["timing"].items():a(f"{k_:<24}: {v:.3f}")
    a(f"peak GPU memory {P['gpu']['peak_allocated_gib']:.2f} GiB");a("");a(f"Raw forward records are in {names['payload']} and {names['jsonl']}.");a(S)
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
    run_id=ctl["run_id"];run_dir=Path(ctl["run_dir"]);case=int(ctl["case"])
    info=I.info;checks=[];tm=Counter();T0=time.perf_counter();state["checks"]=checks
    raw=dict(write=None,ssd=None,address=None,answer=None,control=None);state["raw"]=raw
    def chk(name,cond):
        checks.append(name)
        if not cond:raise AuditFail(name)
    run_start_utc=utc_now();say("="*140);say(f"RUN {run_id} · case {case:02d}");say("="*140)
    # ---- 1 integrity ----
    yield ev(1,"Integrity","Engine and chain hashes, source blobs, locked READ setting, start-up acceptance status, weight sentinel, all-parameter guard, hooks.")
    I.reset_peak();cfg=I.config();su=I.startup();fz0=I.frozen();t=time.perf_counter();g_before=I.guard();tm["guard"]+=time.perf_counter()-t
    chk("TEST627 demo engine SHA-256 and git blob = embedded reference",cfg["engine_sha256"]==ENGINE_SHA256_EXPECTED and cfg["engine_blob"]==ENGINE_BLOB_EXPECTED)
    chk("626.py · 624.py · 610.py executed by the engine = SHA-256 references",all(cfg["chain_sha256"][k]==CHAIN[k]["sha256"]for k in CHAIN)and cfg["chain_sha256"]["610.py"]==cfg["lock_610"]==SHA610_EXPECTED)
    chk("566.py and 575.py git blobs verified by 610.py = expected",cfg["source_blobs"]==SOURCE_BLOBS_EXPECTED==cfg["source_blobs_measured"])
    chk(f"model, architecture, CUT{CUT_EXPECTED} and {PREFIX_EXPECTED}-token instruction prefix",cfg["MODEL_ID"]==MODEL_ID and tuple(cfg["arch"])==ARCH and cfg["CUT"]==CUT_EXPECTED and cfg["prefix_tokens"]==PREFIX_EXPECTED)
    chk(f"{N_CART} cartridges · PCA{RANK_EXPECTED} · IVF{NLIST_EXPECTED} · every native position indexed",cfg["n_cartridges"]==cfg["manifest"]==N_CART and cfg["rank"]==RANK_EXPECTED and cfg["nlist"]==NLIST_EXPECTED and cfg["index_ntotal"]==cfg["ntok"])
    chk("locked READ setting nprobe 2 · token_k 16 · budget 16 · PEAK50",tuple(cfg["candidates"])==LOCKED_CAND and tuple(cfg["reranker"])==LOCKED_RERANK)
    chk("TEST627 engine at start-up in this runtime: status COMPLETE · integrity 8/8",su["status"]=="COMPLETE"and len(su["integrity"])==8 and all(su["integrity"].values()))
    chk("full-weight SHA-256 at start-up = reference",info["weight_sha0"]==WEIGHT_SHA_EXPECTED)
    chk("weight sentinel before run = value at initialisation",fz0["sentinel"]==info["sentinel0"]);chk("all-parameter guard before run = value at start-up",g_before==info["guard0"])
    chk("trainable tensors = 0, eval mode, no LoRA, no optimizer",fz0["trainable_tensors"]==0 and not fz0["training"]and not fz0["lora"]and not fz0["optimizer"])
    chk("no foreign hooks outside engine calls · no hooks on decoder layers, attention or q_proj",fz0["foreign_hooks"]==0 and not fz0["residual_hooks"])
    cases=I.cases();chk(f"case in 0..{N_CASES-1} of the engine's {N_CASES} new-wording questions",len(cases)==N_CASES and 0<=case<N_CASES and cases[case]["case"]==case)
    tgt=cases[case]["target"];question=cases[case]["question"];C=I.car(tgt);expected=C["gold"];fact=C["fact"]
    strip=I.panel_strings();state["strip"]=strip
    for c in checks:say("   PASS ·",c)
    # ---- 2 WRITE ----
    yield ev(2,"A · WRITE","The engine's init(fact, 3) is executed again for the fact of this case: one token-id forward of the instruction prefix + the fact, one all-zero-embedding forward; the result is compared bitwise with the file written at start-up.")
    src=I.source_ids(tgt);memW,fwW,dtW=I.write(tgt);tm["write"]+=dtW;raw["write"]=dict(cartridge=tgt,forwards=len(fwW))
    chk("WRITE: forward 1 = token ids of the instruction prefix + this fact only (no cache)",len(fwW)==2 and fwW[0]["kind"]=="ids"and fwW[0]["ids"]==src and not fwW[0]["has_past"])
    chk("WRITE: forward 2 = all-zero input embeddings over the same rows (no cache); no other forward",fwW[1]["kind"]=="embeds"and fwW[1]["embeds_zero"]and fwW[1]["n"]==len(src)and not fwW[1]["has_past"])
    shp=I.memory_shape(memW);T_w=shp["T"]
    chk(f"cartridge = {ARCH[0]} layers × (K, V) of shape [1, {ARCH[3]}, T, {ARCH[4]}] with T = instruction prefix + fact tokens",shp["layers"]==ARCH[0]and shp["shape"]==[1,ARCH[3],T_w,ARCH[4]]and T_w==len(src)and shp["uniform"])
    wbytes=I.memory_bytes(memW);chk("cartridge bytes = layers × 2 × KV heads × head dim × T × element size",wbytes==ARCH[0]*2*ARCH[3]*ARCH[4]*T_w*(2 if shp["dtype"]in("bfloat16","float16")else 4))
    t=time.perf_counter();stored,fi=I.read_file(tgt);tm["write_compare_load"]+=time.perf_counter()-t
    chk("start-up file of this fact: SHA-256 recomputed from disk = manifest",fi["sha_ok"]and fi["sha256"]==C["sha256"])
    cmpd=I.compare(memW,stored);raw["write"]["compare"]={k_:cmpd[k_]for k_ in("tensors_compared","all_equal","layers_differing","max_abs_diff")}
    if not cmpd["all_equal"]:
        # Diagnostic BEFORE the gate fails: a second live re-write separates re-execution nondeterminism from a stored-memory mismatch.
        memW2,_,_=I.write(tgt);c2=I.compare(memW,memW2);c3=I.compare(memW2,stored)
        cls=("re-execution is not deterministic: two live re-writes of the same fact differ from each other"if not c2["all_equal"]else
             "the re-write is reproducible (two live re-writes identical) but differs from the start-up file: stored-memory mismatch")
        raw["write_diagnostic"]=dict(classification=cls,live_vs_file=cmpd,live_vs_second_live=c2,second_live_vs_file=c3)
        say("   WRITE DIAGNOSTIC:",cls,"· layers differing",cmpd["layers_differing"],"· max |Δ|",cmpd["max_abs_diff"]);I.release(memW2);del memW2
    chk(f"WRITE comparison covers all {2*ARCH[0]} tensors ({ARCH[0]} layers × K and V), compared as raw bytes",cmpd["tensors_compared"]==2*ARCH[0]and cmpd["layers_a"]==cmpd["layers_b"]==ARCH[0])
    beq=cmpd["all_equal"];chk(f"live re-write bitwise identical to the start-up file in all {ARCH[0]} layers (K and V)",beq)
    wsha=I.memory_sha(memW);chk("memory SHA-256 of the re-write = memory SHA-256 of the file",wsha==I.memory_sha(stored))
    fb=I.file_bytes(tgt);fact_in=fact.encode("utf-8")in fb;gold_in=expected.encode("utf-8")in fb;q_in=question.encode("utf-8")in fb
    chk("the cartridge file contains neither the fact text, nor the answer word, nor the question",not fact_in and not gold_in and not q_in)
    write=dict(cartridge=tgt,file=C["file"],file_bytes=len(fb),file_sha256=fi["sha256"],sha_ok=fi["sha_ok"],forwards=fwW,T=T_w,prefix_tokens=cfg["prefix_tokens"],fact_tokens=T_w-cfg["prefix_tokens"],
               layers=shp["layers"],tensors=shp["tensors"],shape=shp["shape"],dtype=shp["dtype"],memory_bytes=wbytes,container_overhead=len(fb)-wbytes,memory_sha256=wsha,bit_equal=beq,tensors_compared=cmpd["tensors_compared"],elements_compared=int(sum(r["elements"]for r in cmpd["rows"])),
               fact_in_file=fact_in,gold_in_file=gold_in,question_in_file=q_in,seconds=dtW,startup_write_ms=float(I.build_ms(tgt)))
    I.release(memW,stored);del memW,stored
    say(f"   WRITE #{tgt:04d}: {T_w} rows · bitwise identical to {C['file']} · {1000*dtW:.0f} ms")
    # ---- 3 SSD ----
    yield ev(3,"B · SSD","All 1,453 cartridge files are listed and their sizes compared with the SHA-256 manifest written at start-up.")
    t=time.perf_counter();ps=I.pool_stat();tm["ssd_stat"]+=time.perf_counter()-t;sizes=su["sizes"]
    chk(f"SSD: {N_CART} independent cartridge files present · {N_CART} manifest entries · every size = manifest",ps["files_in_pool"]==ps["manifest"]==N_CART and not ps["size_mismatch"])
    chk("SSD: total pool bytes = sum of the manifest sizes = engine storage report",ps["total_bytes"]==sum(sizes)==su["storage"]["disk_pool_bytes"])
    ssd=dict(ps,min=min(sizes),max=max(sizes),median=float(np.median(sizes)),startup_sha_verified=bool(su["integrity"].get("manifest_1453_sha_verified")),write_ms_p50=su["write"]["ms_p50"])
    raw["ssd"]=dict(files=ps["files_in_pool"],bytes=ps["total_bytes"])
    # ---- 4 READ address ----
    yield ev(4,"C · READ — geometric address","From the question alone: one question-frame forward → layer-3 vectors → PCA128 → FAISS IVF128 → 16 candidates → locked PEAK50 selects one cartridge.")
    A=I.address(question);tm["address"]+=A["address_ms"]/1000;tm["question_encode"]+=A["encode_ms"]/1000;raw["address"]=dict(ranking=A["ranking"],touched=A["touched"])
    fids=I.frame_ids(question)
    chk("READ: exactly one model forward to read the question = instruction prefix + question only (no cache, no fact text)",len(A["forwards"])==1 and A["forwards"][0]["kind"]=="ids"and A["forwards"][0]["ids"]==fids and not A["forwards"][0]["has_past"])
    rk=A["ranking"];chk(f"READ: 1–{LOCKED_CAND[2]} distinct candidate cartridges in 0..{N_CART-1}, ranked by PEAK50",1<=len(rk)<=LOCKED_CAND[2]and len(set(rk))==len(rk)and all(0<=x<N_CART for x in rk)and[c["id"]for c in A["candidates"]]==rk)
    chk("READ: PEAK50 score = 0.5 × FRAM + 0.5 × peak for every candidate (re-derived)",all(abs(c["score"]-(0.5*c["fram"]+0.5*c["peak"]))<=1e-9 for c in A["candidates"]))
    c626=A["candidate626"];sc_={c["id"]:c["score"]for c in A["candidates"]}
    chk(f"READ: candidate626 returned min({LOCKED_CAND[2]}, touched) candidates and PEAK50 ranks exactly that set (nothing merged, added or dropped)",len(c626)==min(LOCKED_CAND[2],A["touched"])and sorted(c626)==sorted(rk))
    chk("READ: order re-derived with the engine's rule — PEAK50 score descending, ties by cartridge id",rk==sorted(c626,key=lambda ci:(-sc_[ci],ci)))
    srow=I.sealed_row(case)
    if not(srow["selected"]==rk[0]and list(srow["top5"])==rk[:5]):raw["read_diagnostic"]=dict(live_top5=rk[:5],startup_top5=list(srow["top5"]),live_candidates=c626)
    chk("READ reproduces the TEST627 engine's start-up record of this question in this runtime: same selected cartridge and same top-5",srow["selected"]==rk[0]and list(srow["top5"])==rk[:5])
    sel=rk[0];sel_ok=sel==tgt;tin=tgt in rk
    say(f"   READ: touched {A['touched']} · {len(rk)} candidates · selected #{sel:04d} · expected among candidates {tin} · {A['address_ms']:.1f} ms")
    # ---- 5 READ load + answer ----
    yield ev(5,"D · READ — load from SSD and answer","The selected file is checked against its SHA-256, loaded with torch.load, and the frozen model answers over it; the source text is not given.")
    memS,li=I.read_file(sel,device=True);tm["load"]+=li["seconds"]
    chk("READ: selected file SHA-256 recomputed from disk = manifest",li["sha_ok"])
    T=I.memory_len(memS);a1,fwA,dtA=I.ask(memS,question);tm["answer"]+=dtA;raw["answer"]=a1;qids=check_query(chk,I,fwA,question,T,"answer")
    correct=I.hit(a1,expected);cr=hit_ans(a1,expected);chk("correctness re-derived from the answer text = engine hit()",cr==correct)
    I.release(memS);del memS
    say(f"   answer {a1!r} · expected {expected} · {'CORRECT'if correct else'INCORRECT'} · {len(qids)} question tokens over the {T}-row cartridge")
    # ---- 6 control ----
    yield ev(6,"E · Control — runner-up file","The same question over the file of the second-ranked candidate (diagnostic: does the expected answer appear without the selected cartridge?).")
    if len(rk)>=2:
        ru=rk[1];memR,lr=I.read_file(ru,device=True);chk("control: runner-up file SHA-256 = manifest",lr["sha_ok"]);TR_=I.memory_len(memR)
        aR,fwR,dtR=I.ask(memR,question);tm["control"]+=dtR+lr["seconds"];check_query(chk,I,fwR,question,TR_,"control")
        control=dict(id=ru,file=lr["file"],answer=aR,gold_produced=I.hit(aR,expected),is_target=ru==tgt,T=TR_);I.release(memR);del memR
    else:control=dict(skipped="only one candidate was returned",gold_produced=None)
    raw["control"]=control
    # ---- 7 verification ----
    yield ev(7,"Verification","Weight sentinel, all-parameter guard, trainable tensors and hooks after the run.")
    fz1=I.frozen();t=time.perf_counter();g_after=I.guard();tm["guard"]+=time.perf_counter()-t
    chk("all-parameter guard after run = before run = start-up (weights unchanged)",g_after==g_before==info["guard0"]);chk("weight sentinel after run = value at initialisation",fz1["sentinel"]==info["sentinel0"])
    chk("trainable tensors = 0, no foreign hooks and no hooks on layers/attention/q_proj after run",fz1["trainable_tensors"]==0 and fz1["foreign_hooks"]==0 and not fz1["residual_hooks"])
    tm["engine_total"]=time.perf_counter()-T0
    verdict=("SELFTEST_SYNTHETIC_"if I.kind!="ENGINE"else"")+f"CASE_{case:02d}_{'CORRECT'if correct else'INCORRECT'}_WRITE_REVERIFIED_SSD_READ_SOURCE_FREE_ANSWER"
    # ---- 8 seal ----
    yield ev(8,"Sealing","Payload, manifest, readable log and raw records are written and hashed.")
    peak=I.peak_gib();same_hw=info["gpu"]==CANON_GPU
    read=dict(question=question,frame_forwards=A["forwards"],frame_tokens=len(fids),query_rows=A["query_rows"],touched=A["touched"],candidates=A["candidates"],ranking=rk,selected=sel,selected_correct=sel_ok,
              target_in_candidates=tin,faiss_ms=A["faiss_ms"],top2_ms=A["top2_ms"],rerank_ms=A["rerank_ms"],address_ms=A["address_ms"],encode_ms=A["encode_ms"],
              file=li["file"],sha_ok=li["sha_ok"],load_ms=1000*li["seconds"],candidate626=c626,
              timing_scope={"encode_ms":"question_rep + apply (one question-frame forward)","address_ms":"candidate626 (PCA projection + FAISS search + TOP2) + features626 + rerank626, as TEST627 address_ms",
                            "faiss_ms":"PCA projection + IVF quantizer + FAISS search (part of address_ms)","top2_ms":"TOP2 candidate scoring (part of address_ms)","rerank_ms":"features626 + rerank626 (part of address_ms)",
                            "load_ms":"sha627 SHA-256 of the file + torch.load + copy to GPU, as TEST627 load_ms; may be served from the OS page cache","answer_ms":"answer(memory, question): question-template forward + greedy generation",
                            "write_ms":"init(fact, 3) only (two forwards); file writing is not repeated","note":"stage timings of one run; they are not summed into an end-to-end latency"},T=T,answer_forwards=fwA,query_tokens=len(qids),query_token_text=I.decode_each(qids),gen_forwards=len(fwA)-1,answer_ms=1000*dtA)
    live=dict(case=case,question=question,fact=fact,expected=expected,target=tgt,write=write,ssd=ssd,read=read,answer=a1,correct=bool(correct),correct_rederived=bool(cr),control=control,
              startup_row=srow,matches_startup_selection=srow["selected"]==sel,matches_startup_answer=norm_ans(str(srow["answer"]))==norm_ans(a1),
              session=[dict(r,correct=(bool(correct)if r.get("correct")is None else bool(r["correct"])))for r in(ctl.get("session")or[])])
    P={"schema":"akbascore.mam.silicon_hippocampus.qwen.test627.demo.run.v1","demo":DEMO_TITLE,"subtitle":DEMO_SUB,"name_note":NAME_NOTE,"author":AUTHOR,"organisation":ORG,"date":DATE_ISO,"place":AUTHOR_PLACE,
       "copyright":COPYRIGHT,"license":{"name":LICENSE_NAME,"url":LICENSE_URL},"priority_record":PRIORITY,"paradigm":PARADIGM,"discovery":DISCOVERY,"question":SCI_QUESTION,"core_message":CORE_MESSAGE,
       "earlier_records":{"v1.0":PRIOR_PUB,"v1.1":QWEN_PUB,"v1.2":V12_PUB,"note":"cited by DOI only; no value of an earlier record is used as a result here"},
       "run_id":run_id,"run_start_utc":run_start_utc,"run_end_utc":utc_now(),"verdict":verdict,
       "model":{"id":cfg["MODEL_ID"],"arch":info["arch"],"dtype":info["dtype"],"attn":info["attn"],"params":info["params"]},
       "environment":{k_:info[k_]for k_ in("gpu","gpu_total_gib","torch","transformers","gradio","python","platform")},"hardware":{"reference_gpu":CANON_GPU,"this_run_gpu":info["gpu"],"same":bool(same_hw)},
       "engine":{"file":info["engine_file"],"sha256":cfg["engine_sha256"],"blob":cfg["engine_blob"],"config":cfg,"init_seconds":info["init_seconds"],"instrument":I.kind,
                 "calls":"init(ci, 3) [WRITE]; sha627 + torch.load [SSD]; question_rep + apply + candidate626 + features626 + rerank626(PEAK50) [READ address]; answer(memory, q) [READ answer]",
                 "repository":{"627.py":URL_627,"627.log":URL_627LOG,"626.py":URL_626,"626.log":URL_626LOG,"624.py":URL_624,"610.py":URL_610}},
       "instrumentation":{"forward_counter":info["forward_counter"]},"startup":su,"live":live,
       "frozen":{"weight_sha_startup":info["weight_sha0"],"weight_sha_reference":WEIGHT_SHA_EXPECTED,"full_weight_method":info["full_weight_method"],"sentinel_startup":info["sentinel0"],"sentinel_before":fz0["sentinel"],"sentinel_after":fz1["sentinel"],"guard_startup":info["guard0"],"guard_before":g_before,"guard_after":g_after,
                 "trainable_tensors":fz1["trainable_tensors"],"lora":fz1["lora"],"optimizer":fz1["optimizer"],"training_mode":fz1["training"],"foreign_hooks_before":fz0["foreign_hooks"],
                 "foreign_hooks_after":fz1["foreign_hooks"],"residual_hooks_before":fz0["residual_hooks"],"residual_hooks_after":fz1["residual_hooks"],"guard_method":info["guard_method"],"sentinel_method":info["sentinel_method"]},
       "archived":{"test627_log":TEST627_LOG,"test626_log":TEST626_LOG,"note":"Sealed reference values from the repository logs; not produced or recomputed by this run."},
       "scope":SCOPE,"checks_pre_seal":list(checks),"timing":dict(tm),"gpu":{"peak_allocated_gib":peak}}
    P=jsafe(P);allow=(info["gpu"],CANON_GPU)
    hits=claim_scan("payload",strip_words(json.dumps(P,ensure_ascii=False),strip),allow=allow)
    chk("claim-discipline and foreign-model scan of the payload: 0 hits",not hits);P["claim_scan"]={"rules":len(CLAIM_RULES)+1+len(LEAK_RULES),"hits":0,"scope":"payload JSON text; panel names excluded; earlier-record citation and the demo name exempt"}
    pb=canon(P);sha=hashlib.sha256(pb).hexdigest();payload_name=f"{run_id}_FULL_RUN_LOG_payload.json";pp=run_dir/payload_name;t_seal=time.perf_counter();pp.write_bytes(pb);say("payload sealed:",sha)
    chk("payload SHA-256 verification after write",hashlib.sha256(pp.read_bytes()).hexdigest()==sha)
    names=dict(payload=payload_name,manifest=f"run_manifest_{run_id}.json",txt=f"{run_id}_READABLE_RUN_LOG.txt",summary=f"{run_id}_SUMMARY.json",jsonl=f"{run_id}_RAW_RECORDS.jsonl",images=f"figures_manifest_{run_id}.json",
               engine=f"{run_id}_ENGINE_STARTUP_RESULT.json",zip=f"AKBASCORE_MAM_SILICON_HIPPOCAMPUS_EVIDENCE_{run_id}.zip")
    ep=run_dir/names["engine"];src_res=I.engine_result_path()
    if src_res is not None and Path(src_res).is_file():shutil.copy(src_res,ep)
    else:ep.write_text(json.dumps({"note":"engine result file unavailable"}),encoding="utf-8")
    chk("engine start-up result copied and its SHA-256 = value recorded at start-up",(hashlib.sha256(ep.read_bytes()).hexdigest()==su["result_sha256"])if su["result_sha256"]else False)
    mp=run_dir/names["manifest"];sealed_utc=utc_now()
    manifest={"demo":DEMO_TITLE,"author":AUTHOR,"organisation":ORG,"date":DATE_ISO,"copyright":COPYRIGHT,"license":LICENSE_URL,"priority_record":PRIORITY,"run_id":run_id,"payload_file":payload_name,
              "payload_sha256":sha,"payload_bytes":len(pb),"sealed_utc":sealed_utc,"verdict":verdict,"engine_sha256":cfg["engine_sha256"],"engine_chain_sha256":cfg["chain_sha256"],
              "engine_startup_result_file":ep.name,"engine_startup_result_sha256":su["result_sha256"],"test627_log_sha256":TEST627_LOG["log_sha256"],"weight_sha_startup":info["weight_sha0"],"model":MODEL_ID,
              "written_cartridge_file":write["file"],"written_cartridge_sha256":write["file_sha256"],"selected_cartridge_file":read["file"],
              "canonicalization":"json.dumps(sort_keys=True, separators=(',',':'), ensure_ascii=False, allow_nan=False), UTF-8",
              "note":"Artifact integrity seal: confirms the payload is unchanged since sealing. It is not a scientific proof and not a third-party verification."}
    mp.write_text(json.dumps(manifest,indent=2,sort_keys=True,ensure_ascii=False),encoding="utf-8")
    summary={"run_id":run_id,"verdict":verdict,"case":case,"question":question,"answer":a1,"expected":expected,"correct":bool(correct),"target":tgt,"selected":sel,"selected_correct":sel_ok,
             "candidates":rk,"touched":A["touched"],"write_bitwise_identical":beq,"cartridge_rows":T,"cartridge_file":read["file"],"address_ms":A["address_ms"],"load_ms":read["load_ms"],"answer_ms":read["answer_ms"],
             "control":{k_:control.get(k_)for k_ in("id","answer","gold_produced","is_target","skipped")},"weights_unchanged":True,"payload_sha256":sha,"model":MODEL_ID,
             "startup_acceptance":{p_:{k_:su["summary"][p_][k_]for k_ in("cases","selected_correct","answer_correct","errors")}for p_ in("ORIGINAL","ALTERNATE")}}
    (run_dir/names["summary"]).write_text(json.dumps(jsafe(summary),indent=2,ensure_ascii=False),encoding="utf-8")
    lines=[{"op":"write","cartridge":tgt,"forwards":fwW,"T":T_w,"bit_equal":beq,"memory_sha256":wsha,"file":write["file"],"file_sha256":write["file_sha256"]},
           {"op":"ssd","files":ps["files_in_pool"],"bytes":ps["total_bytes"],"size_mismatch":ps["size_mismatch"]},
           {"op":"address","question":question,"forwards":A["forwards"],"candidates":A["candidates"],"touched":A["touched"],"address_ms":A["address_ms"]},
           {"op":"answer","file":read["file"],"answer":a1,"forwards":fwA},{"op":"control",**{k_:v for k_,v in control.items()}}]
    (run_dir/names["jsonl"]).write_text("\n".join(json.dumps(jsafe(x),ensure_ascii=False)for x in lines)+"\n",encoding="utf-8")
    (run_dir/names["txt"]).write_text(make_txt(P,sha,names,sealed_utc),encoding="utf-8");seal_s=time.perf_counter()-t_seal
    # ---- 9 figures ----
    yield ev(9,"Rendering figures",f"{N_FIGS} figures at 300 dpi (JPEG) with vector PDF copies (+ Fig. 1 in Turkish); every printed value is checked against the sealed payload.")
    ctx={"P":P,"sha":sha,"run_id":run_id,"payload_name":payload_name,"allow":allow,"strip":strip,"N":N_FIGS,"audit":[],"pdfs":[],"checks":len(checks)}
    t_r=time.perf_counter();imgs,extra=render_all(ctx,run_dir);render_s=time.perf_counter()-t_r;entries=[]
    for n_,(p,cap)in enumerate(imgs,1):
        with Image.open(p)as im:w_,h_=im.size
        q_=ctx["pdfs"][n_-1];entries.append({"index":n_,"jpg":p.name,"pdf":q_.name,"title":cap,"width_px":w_,"height_px":h_,"dpi":OUT_DPI,"jpg_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"pdf_sha256":hashlib.sha256(q_.read_bytes()).hexdigest()})
    imf=run_dir/names["images"]
    imf.write_text(json.dumps({"run_id":run_id,"payload_sha256":sha,"verdict":verdict,"count":len(entries),"render_seconds":render_s,"figures":entries,
                               "turkish_fig1":[{"file":Path(x).name,"sha256":hashlib.sha256(Path(x).read_bytes()).hexdigest()}for x in extra],"figure_text_audit":ctx["audit"],
                               "note":"Per-file SHA-256 is an artifact integrity seal. Figure text was checked against the sealed payload and scanned for claim discipline before saving."},indent=2,ensure_ascii=False),encoding="utf-8")
    # ---- 10 package ----
    yield ev(10,"Packaging","ZIP with figures, PDFs, payload, manifest, logs, raw records and the engine start-up result; post-seal audit.")
    zp=run_dir/names["zip"];members=[p for p,_ in imgs]+list(ctx["pdfs"])+[Path(x)for x in extra]+[imf,mp,pp,run_dir/names["txt"],run_dir/names["summary"],run_dir/names["jsonl"],ep]
    for m in members:
        if m.suffix in(".json",".txt",".jsonl")and m!=ep:chk(f"claim-discipline and foreign-model scan of {m.name}: 0 hits",not claim_scan(m.name,strip_words(m.read_text(encoding="utf-8"),strip),allow=allow,numeric=m.suffix==".txt"))
    with zipfile.ZipFile(zp,"w",compression=zipfile.ZIP_DEFLATED)as z:
        for m in members:z.write(m,arcname=m.name)
    post=post_seal_audit(imgs,ctx["pdfs"],extra,zp,[m.name for m in members],pp,sha,run_dir/names["txt"],mp,verdict)
    say(f"{run_id}: {len(checks)} pre-seal + {len(post)} post-seal checks = {len(checks)+len(post)}/{len(checks)+len(post)} PASS · figures {render_s:.1f}s · seal {seal_s:.2f}s")
    say("VERDICT :",verdict);say("PAYLOAD :",sha);say("ZIP     :",zp);say("="*140)
    tr_jpg=next((Path(x)for x in extra if str(x).endswith(".jpg")),None)
    ledger=dict(run_id=run_id,utc=sealed_utc,case=case,question=question,answer=a1,expected=expected,correct=bool(correct),selected=sel,selected_correct=sel_ok,write_bitwise_identical=beq,
                control_gold_produced=control.get("gold_produced"),weights_unchanged=True,verdict=verdict,payload_sha256=sha)
    return dict(run_id=run_id,run_dir=run_dir,P=P,sha=sha,imgs=imgs,pdfs=ctx["pdfs"],extra=extra,tr_jpg=tr_jpg,zip=zp,payload=pp,txt=run_dir/names["txt"],manifest=mp,images_manifest=imf,
                summary=run_dir/names["summary"],jsonl=run_dir/names["jsonl"],engine=ep,checks=len(checks)+len(post),verdict=verdict,ledger=ledger)
#<<CORE_END>>
#<<FIGURES_BEGIN>>
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":.8,"axes.edgecolor":"#374151","axes.labelcolor":"#111827","xtick.color":"#374151","ytick.color":"#374151",
                     "xtick.labelsize":8,"ytick.labelsize":8,"axes.labelsize":9,"axes.titlesize":9.5,"axes.titleweight":"bold","axes.titlelocation":"left","pdf.fonttype":42,"ps.fonttype":42,
                     "legend.fontsize":8,"legend.frameon":False,"mathtext.fontset":"dejavusans"})
FIG_W,FIG_H=13.333,7.5;LAYOUT_DPI=100;OUT_DPI=300
INK,INK2,MUTED,RULE,GRIDC,PANEL="#111827","#374151","#6B7280","#D1D5DB","#E5E7EB","#F9FAFB"
BLUE,BLUE_L,ORANGE,ORANGE_L,TEAL,TEAL_L,GRAYPT,AMBER_L,AMBER,RED="#1D4ED8","#EFF6FF","#C2410C","#FFF7ED","#0F766E","#F0FDFA","#9CA3AF","#FFFBEB","#B45309","#B91C1C"
VIOLET,VIOLET_L="#6D28D9","#F5F3FF"
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
def finish(fig,path,ctx,expect,desc,register=True):
    """Every expected value must appear in the figure text; claim scan must be clean; then JPEG (300 dpi, EXIF provenance) + vector PDF."""
    sanitize_dashes(fig);fig.canvas.draw();texts=[t.get_text()for t in fig.findobj(Text)if t.get_text().strip()];blob="\n".join(texts);nm=Path(path).name
    ws=lambda z:re.sub(r"\s+","",z);bw=ws(blob);miss=[e for e in expect if ws(str(e))not in bw]
    if miss:plt.close(fig);raise AuditFail(f"FIGURE/PAYLOAD MISMATCH in {nm}: missing {miss}")
    hits=claim_scan(nm,strip_words(re.sub(r"\s*\n\s*"," ",blob),ctx.get("strip",[])),allow=ctx["allow"],numeric=True)
    if hits:plt.close(fig);raise AuditFail(f"CLAIM-DISCIPLINE HIT in {nm}: {hits[:3]}")
    pdf=Path(path).with_suffix(".pdf")
    fig.savefig(pdf,format="pdf",facecolor="white",metadata={"Title":desc,"Author":f"{AUTHOR} · {ORG}","Subject":f"{DEMO_TITLE} · run {ctx['run_id']} · payload SHA-256 {ctx['sha']}","Creator":DEMO_SHORT,
                "Keywords":f"model {MODEL_ID}; TEST627 demo engine sha256 {ENGINE_SHA256_EXPECTED}; 627.log sha256 {TEST627_LOG['log_sha256']}; license {LICENSE_URL}"})
    buf=io.BytesIO();fig.savefig(buf,format="png",dpi=OUT_DPI,facecolor="white");plt.close(fig);buf.seek(0)
    with Image.open(buf)as im:rgb=im.convert("RGB")
    ex=Image.Exif();ex[0x010E]=ascii_(f"{desc} | {DEMO_SHORT} | run {ctx['run_id']} | payload sha256 {ctx['sha']} | engine sha256 {ENGINE_SHA256_EXPECTED}");ex[0x013B]=ascii_(f"{AUTHOR} / {ORG}");ex[0x8298]=ascii_(f"{COPYRIGHT} - {LICENSE_NAME} - {LICENSE_URL}");ex[0x0131]=ascii_(DEMO_SHORT)
    rgb.save(path,"JPEG",quality=95,optimize=True,progressive=False,subsampling=0,dpi=(OUT_DPI,OUT_DPI),exif=ex.tobytes())
    with Image.open(path)as c_:
        if c_.format!="JPEG"or c_.mode!="RGB":raise RuntimeError("JPEG validation failed")
    if register:ctx["pdfs"].append(pdf)
    ctx["audit"].append(dict(file=nm,texts=len(texts),expected_values=len(expect),claim_hits=0));return str(path)
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
def header(fig,k,title,sub,tag,tagc,figword="Fig."):
    fig.text(.03,.955,f"{figword} {k}",fontsize=12,weight="bold",color=MUTED,va="center")
    fig.text(.03+(.048 if figword=="Fig."else .056),.955,mt(title),fontsize=16.5,weight="bold",color=INK,va="center")
    fit_text(fig,.03,.902,.94,.032,sub,fs_max=9.4,fs_min=7.6,color=INK2)
    fig.text(.97,.955,mt(tag),fontsize=8.4,weight="bold",color=tagc,va="center",ha="right",bbox=dict(boxstyle="square,pad=0.35",fc="white",ec=tagc,lw=1.0))
    fig.add_artist(Line2D([.03,.97],[.895,.895],transform=fig.transFigure,color=INK2,lw=.8))
def footer(fig,ctx,k,caption,fs_max=8.4):
    if ctx["P"]["engine"].get("instrument")!="ENGINE":
        fig.text(.5,.5,"SYNTHETIC SELF-TEST · NOT A RESULT",fontsize=44,color=RED,alpha=.22,ha="center",va="center",rotation=18,weight="bold",zorder=50)
    fig.add_artist(Line2D([.03,.97],[.132,.132],transform=fig.transFigure,color=RULE,lw=.6))
    fit_text(fig,.03,.045,.94,.08,caption,fs_max=fs_max,fs_min=6.4,color=INK2)
    fig.add_artist(Line2D([.03,.97],[.038,.038],transform=fig.transFigure,color=RULE,lw=.6))
    fig.text(.5,.025,mt(f"{DEMO_TITLE} · {DISCOVERY} · {ORG} · {DATE_TXT} · {LICENSE_NAME} · {LICENSE_URL}"),ha="center",va="center",fontsize=6.6,color=INK2)
    fig.text(.5,.010,mt(f"live run {ctx['run_id']} · {MODEL_FAMILY} · TEST627 demo engine (SHA-256 {ENGINE_SHA256_EXPECTED[:12]}…) · 627.log {TEST627_LOG['log_sha256'][:12]}… · payload {ctx['sha'][:12]}… · Fig. {k}/{ctx['N']} · {COPYRIGHT}"),
             ha="center",va="center",fontsize=6.2,color=MUTED,family=MONO)
def clean(ax,grid=True):
    for s in("top","right"):ax.spines[s].set_visible(False)
    if grid:ax.grid(True,color=GRIDC,lw=.5);ax.set_axisbelow(True)
def frame(fig,x,y,w,h,ec=INK2,fc="white",lw=.9,ls="-",z=-1):
    fig.add_artist(Rectangle((x,y),w,h,transform=fig.transFigure,facecolor=fc,edgecolor=ec,lw=lw,ls=ls,zorder=z))
def arrow(fig,x0,y0,x1,y1,color=INK2,lw=1.2,ms=12,cs="arc3",ls="-"):
    fig.add_artist(FancyArrowPatch((x0,y0),(x1,y1),transform=fig.transFigure,arrowstyle="-|>",mutation_scale=ms,lw=lw,color=color,connectionstyle=cs,ls=ls,zorder=6))
def sblock(fig,x,y,w,h,stage,title,body,ec,fc="white",bfs=8.4,bmin=6.0,tfs=9.2,mono=False,bpad=.007):
    """Coloured stage block: solid header strip with the stage label, bold title, wrapped body."""
    frame(fig,x,y,w,h,ec,fc,lw=1.0,z=1);frame(fig,x,y+h-.03,w,.03,ec,ec,lw=1.0,z=1)
    fig.text(x+.006,y+h-.015,mt(stage),fontsize=7.4,weight="bold",color="white",va="center",zorder=3)
    fit_text(fig,x+.006,y+h-.08,w-.012,.046,title,fs_max=tfs,fs_min=7.0,color=INK,weight="bold")
    if body:fit_text(fig,x+.006,y+bpad,w-.012,h-.086-bpad,body,fs_max=bfs,fs_min=bmin,family=MONO if mono else None,color=INK2)
def chip(fig,x,y,text,col,fs=7.6):
    fig.text(x,y,mt(text),fontsize=fs,color=col,weight="bold",va="center",ha="left",family=MONO,bbox=dict(boxstyle="round,pad=0.28",fc="white",ec=col,lw=.8),zorder=7)
def disk_icon(fig,x,y,s,col):
    ax=fig.add_axes([x,y,s,s*FIG_W/FIG_H]);ax.set_zorder(4);ax.axis("off");ax.set_xlim(-1,1);ax.set_ylim(-1,1);ax.set_aspect("equal")
    ax.add_patch(Circle((0,0),.95,fc="#E5E7EB",ec=col,lw=1.0));ax.add_patch(Circle((0,0),.22,fc="white",ec=col,lw=.8));ax.plot([.25,.85],[.1,.45],color=col,lw=1.6)
def L_(ctx):return ctx["P"]["live"]
def short(s,n):s=str(s);return s if len(s)<=n else s[:n-1]+"…"
def uwrap(s,n,ind=""):
    out=[];cur=""
    for part in str(s).split("_"):
        piece=(cur+"_"+part)if cur else part
        if len(piece)>n and cur:out.append(cur+"_");cur=part
        else:cur=piece
    out.append(cur);return"\n".join(ind+x for x in out)
def okc(b):return TEAL if b else ORANGE
def yn(b):return"yes"if b else"no"
def tick(b):return"✓"if b else"✗"
def gb(b):return f"{b/1e9:.2f} GB"
# ------------------------------------------------------------------ FIG 1 · WRITE · SSD · READ (one architecture, live values)
ARCH_TXT={
"en":dict(figword="Fig.",title="The Silicon Hippocampus — how a memory is formed, stored and recalled",
 sub="{model} stays frozen · WRITE turns a fact into the model's own key/value state and saves it as a file · READ finds that file from the question alone and answers from it · live values of this run",
 tag="WRITE · SSD · READ · LIVE RUN",laneW="A · WRITE — memory formation (done once per fact, before any question; re-executed for this fact in this run)",
 laneR="B · READ — recall (for every question; the source text is not given again)",
 w1t="Source fact (text)",w1b="“{fact}”\nRead by the model once, only at WRITE time.",
 w2t="Frozen Qwen2.5-7B-Instruct",w2b="Forward 1: instruction prefix + fact ({T} tokens).\nForward 2: zero input embeddings, layer-3 state injected.\nWeights unchanged · no training · no LoRA.",
 w3t="CUT3 keeps the model's own K/V",w3b="Layers 0–3: keys/values of the fact forward. Layers 4–27: rebuilt from the layer-3 state alone.\n{L} × (K, V) = {nt} {dt} tensors [1, 4, {T}, 128].\nNot the text, not a text embedding.",
 w4t="Independent file on disk",w4b="{file} · {fbytes} bytes\nSHA-256 {fsha}…\nRe-written now: bitwise identical {eq}",
 ramt="RAM (no text, no labels; built at start-up)",ramb="Address index: FAISS IVF{lists} over {pos} positions ({ixmb} MB) + PCA{pca} basis ({pcmb} MB).\nReranking data: native layer-3 vectors ({natmb} MB).\nModel weights are in GPU memory (not counted here).",
 ssdt="SSD cartridge pool",ssdb="{n} independent files · {gbv}\nsizes = manifest (this run) {ok1}\nSHA-256 of all files (start-up) {ok2}",
 ragt="Why this is not text RAG",ragb="Text RAG: finds text → pastes the text into the prompt.\nMAM: finds a cartridge → loads the model's own stored K/V into attention memory; the source text is not sent again.\nNo vector-DB service · no SQL · no training.",
 r1t="Question only",r1b="“{q}”\nThe fact is not in the prompt.",
 r2t="Frozen Qwen reads the question",r2b="One forward: instruction prefix + question ({ft} tokens), no fact text.\n→ {rows} layer-3 question vectors.",
 r3t="Geometric address (RAM)",r3b="PCA{pca} → IVF{lists}\nnprobe {npb} · token_k {tk}\ntouched {touch} of {n}\n→ {nc} candidates",
 r4t="PEAK50 selection",r4b="Re-scores the {nc} candidates with the question vectors and picks one.\n→ #{sel}",
 r5t="Load the file from SSD",r5b="{selfile}\nSHA-256 = manifest {shaok}\n{T2} rows → GPU attention memory",
 r6t="Frozen model answers",r6b="Input: {qt} question-template tokens over the loaded cartridge.\n→ “{ans}”",
 chipA="address {a:.1f} ms",chipL="load {l:.1f} ms",chipN="answer {n:.0f} ms",tag2="this run · case {case:02d}",
 cap=("Figure {k}. One fact through the whole system in run {rid}. WRITE: the engine's init(fact, 3) produced the {L}-layer key/value cartridge again and it is bitwise identical to the file written at start-up. "
      "SSD: {n} cartridges are {n} independent files. READ: from the question alone the label-free address and the locked PEAK50 rule chose #{sel}; the file was loaded and the frozen model answered “{ans}” "
      "(expected “{exp}”: {res}). Chips: single-run timings of separate stages, not summed; load may be served from the OS page cache. Separate from this run: the engine's start-up acceptance in this runtime "
      "(original {o}/48, new wording {a}/48). Limits: the 96 questions reuse 48 facts and PEAK50 was fixed after TEST626; 1,453 files are not millions; ~{avg} MB per fact; A100 only.")),
"tr":dict(figword="Şekil",title="Silikon Hipokampüs — hafıza nasıl oluşur, saklanır ve hatırlanır",
 sub="{model} dondurulmuş kalır · WRITE bilgiyi modelin kendi anahtar/değer durumuna çevirip dosyaya yazar · READ bu dosyayı yalnızca sorudan bulur ve cevabı ondan üretir · bu çalıştırmanın canlı değerleri",
 tag="WRITE · SSD · READ · CANLI",laneW="A · WRITE — hafızanın oluşumu (her bilgi için bir kez, sorudan önce; bu bilgi için bu çalıştırmada yeniden yürütüldü)",
 laneR="B · READ — hatırlama (her soruda; kaynak metin tekrar verilmez)",
 w1t="Kaynak bilgi (metin)",w1b="“{fact}”\nModel bunu yalnızca WRITE anında bir kez okur.",
 w2t="Dondurulmuş Qwen2.5-7B-Instruct",w2b="İleri geçiş 1: talimat öneki + bilgi ({T} token).\nİleri geçiş 2: sıfır girdi, katman-3 durumu enjekte.\nAğırlıklar değişmez · eğitim yok · LoRA yok.",
 w3t="CUT3 modelin kendi K/V'sini saklar",w3b="Katman 0–3: bilgi geçişinin anahtar/değerleri. Katman 4–27: yalnızca katman-3 durumundan yeniden üretilir.\n{L} × (K, V) = {nt} {dt} tensör [1, 4, {T}, 128].\nMetin değil, metin gömmesi değil.",
 w4t="Diskte bağımsız dosya",w4b="{file} · {fbytes} bayt\nSHA-256 {fsha}…\nŞimdi yeniden yazıldı: bit düzeyinde aynı {eq}",
 ramt="RAM (metin yok, etiket yok; açılışta kuruldu)",ramb="Adres dizini: {pos} konum üzerinde FAISS IVF{lists} ({ixmb} MB) + PCA{pca} tabanı ({pcmb} MB).\nYeniden sıralama verisi: yerel katman-3 vektörleri ({natmb} MB).\nModel ağırlıkları GPU belleğinde (burada sayılmaz).",
 ssdt="SSD kartuş havuzu",ssdb="{n} bağımsız dosya · {gbv}\nboyutlar = manifest (bu çalıştırma) {ok1}\ntüm dosyaların SHA-256'sı (açılış) {ok2}",
 ragt="Neden metin RAG değil",ragb="Metin RAG: metni bulur → metni isteme yapıştırır.\nMAM: kartuşu bulur → modelin kendi saklanmış K/V'sini dikkat belleğine yükler; kaynak metin yeniden gönderilmez.\nVektör DB servisi yok · SQL yok · eğitim yok.",
 r1t="Yalnızca soru",r1b="“{q}”\nBilgi istemde yok.",
 r2t="Dondurulmuş Qwen soruyu okur",r2b="Tek geçiş: talimat öneki + soru ({ft} token), bilgi metni yok.\n→ {rows} katman-3 soru vektörü.",
 r3t="Geometrik adres (RAM)",r3b="PCA{pca} → IVF{lists}\nnprobe {npb} · token_k {tk}\ndokunulan {touch} / {n}\n→ {nc} aday",
 r4t="PEAK50 seçimi",r4b="{nc} adayı soru vektörleriyle yeniden puanlar ve birini seçer.\n→ #{sel}",
 r5t="Dosyayı SSD'den yükle",r5b="{selfile}\nSHA-256 = manifest {shaok}\n{T2} satır → GPU dikkat belleği",
 r6t="Dondurulmuş model cevaplar",r6b="Girdi: yüklenen kartuş üzerinde {qt} soru-şablonu tokenı.\n→ “{ans}”",
 chipA="adres {a:.1f} ms",chipL="yükleme {l:.1f} ms",chipN="cevap {n:.0f} ms",tag2="bu çalıştırma · vaka {case:02d}",
 cap=("Şekil {k}. Çalıştırma {rid} içinde tek bir bilginin sistemdeki yolu. WRITE: motorun init(bilgi, 3) çağrısı {L} katmanlı anahtar/değer kartuşunu yeniden üretti; açılışta yazılan dosyayla bit düzeyinde aynı. "
      "SSD: {n} kartuş {n} bağımsız dosyadır. READ: yalnızca sorudan, etiketsiz adres ve kilitli PEAK50 kuralı #{sel} kartuşunu seçti; dosya yüklendi ve dondurulmuş model “{ans}” cevabını verdi "
      "(beklenen “{exp}”: {res}). Süre etiketleri bu çalıştırmadaki ayrı aşamalardır, toplanmaz; yükleme işletim sistemi önbelleğinden gelmiş olabilir. Bu çalıştırmadan ayrı: motorun bu ortamdaki açılış kabulü "
      "(özgün {o}/48, yeni ifade {a}/48). Sınırlar: 96 soru 48 bilgiyi tekrar kullanır, PEAK50 TEST626'dan sonra sabitlendi; 1.453 dosya milyonlar değildir; bilgi başına ~{avg} MB; yalnızca A100."))}
def f_arch(ctx,k,path,lang="en"):
    P=ctx["P"];L=L_(ctx);cfg=P["engine"]["config"];su=P["startup"];Wr=L["write"];Sd=L["ssd"];Rd=L["read"];ix=su["index"];T_=ARCH_TXT[lang];fig=new_fig()
    n=cfg["n_cartridges"];S=su["summary"];avg=f"{su['storage']['avg_file_bytes']/1e6:.2f}"
    header(fig,k,T_["title"],T_["sub"].format(model=MODEL_FAMILY),T_["tag"],TEAL,figword=T_["figword"])
    # WRITE lane
    frame(fig,.03,.632,.94,.252,ec=BLUE,fc=BLUE_L,lw=1.1,z=0)
    fig.text(.038,.869,mt(T_["laneW"]),fontsize=9.6,weight="bold",color=BLUE,va="center")
    wy,wh=.641,.212;W=[(.038,.205),(.258,.215),(.488,.235),(.738,.224)]
    wb=[("1",T_["w1t"],T_["w1b"].format(fact=L["fact"])),("2",T_["w2t"],T_["w2b"].format(T=Wr["T"])),
        ("3",T_["w3t"],T_["w3b"].format(L=Wr["layers"],nt=Wr["tensors"],dt=Wr["dtype"],T=Wr["T"])),
        ("4",T_["w4t"],T_["w4b"].format(file=Wr["file"],fbytes=f"{Wr['file_bytes']:,}",fsha=Wr["file_sha256"][:16],eq=tick(Wr["bit_equal"])))]
    for(x,w),(s,t,b)in zip(W,wb):sblock(fig,x,wy,w,wh,s,t,b,BLUE,bfs=8.2)
    for i in range(3):arrow(fig,W[i][0]+W[i][1]+.002,wy+wh*.5,W[i+1][0]-.002,wy+wh*.5,BLUE)
    # middle row
    my,mh=.452,.168
    sblock(fig,.038,my,.29,mh,"RAM",T_["ramt"],T_["ramb"].format(pca=ix["pca_dims"],lists=ix["ivf_lists"],pos=f"{ix['native_positions']:,}",ixmb=f"{ix['index_bytes']/1e6:.1f}",pcmb=f"{ix['pca_basis_bytes']/1e6:.1f}",natmb=f"{ix['native_activation_bytes']/1e6:.0f}"),VIOLET,VIOLET_L,bfs=7.8)
    sblock(fig,.343,my,.314,mh,"≠ RAG",T_["ragt"],T_["ragb"],INK2,"white",bfs=8.0)
    sblock(fig,.672,my,.29,mh,"SSD",T_["ssdt"],"",AMBER,AMBER_L);disk_icon(fig,.680,my+.014,.045,AMBER)
    fit_text(fig,.732,my+.008,.224,mh-.093,T_["ssdb"].format(n=f"{n:,}",gbv=gb(Sd["total_bytes"]),ok1=tick(not Sd["size_mismatch"]),ok2=tick(Sd["startup_sha_verified"])),fs_max=8.4,fs_min=6.4,color=INK2)
    arrow(fig,.85,wy-.002,.817,my+mh+.002,AMBER,lw=1.8);arrow(fig,.31,wy-.002,.24,my+mh+.002,VIOLET,lw=1.3,ls="--")
    # READ lane
    frame(fig,.03,.142,.94,.300,ec=TEAL,fc=TEAL_L,lw=1.1,z=0)
    fig.text(.038,.157,mt(T_["laneR"]),fontsize=9.6,weight="bold",color=TEAL,va="center")
    fig.text(.962,.157,mt(T_["tag2"].format(case=L["case"])),fontsize=7.8,color=TEAL,weight="bold",va="center",ha="right")
    ry,rh=.176,.25;bw=.146;gap=(.932-6*bw)/5;X=[.038+i*(bw+gap)for i in range(6)]
    rb=[("5",T_["r1t"],T_["r1b"].format(q=L["question"])),("6",T_["r2t"],T_["r2b"].format(ft=Rd["frame_tokens"],rows=Rd["query_rows"])),
        ("7",T_["r3t"],T_["r3b"].format(pca=ix["pca_dims"],lists=ix["ivf_lists"],npb=cfg["candidates"][0],tk=cfg["candidates"][1],touch=Rd["touched"],n=f"{n:,}",nc=len(Rd["ranking"]))),
        ("8",T_["r4t"],T_["r4b"].format(nc=len(Rd["ranking"]),sel=f"{Rd['selected']:04d}")),("9",T_["r5t"],T_["r5b"].format(selfile=Rd["file"],shaok=tick(Rd["sha_ok"]),T2=Rd["T"])),
        ("10",T_["r6t"],T_["r6b"].format(qt=Rd["query_tokens"],ans=short(L["answer"],28)))]
    for i,(x,(s,t,b))in enumerate(zip(X,rb)):sblock(fig,x,ry,bw,rh,s,t,b,TEAL,"white",bfs=7.9,bmin=5.8,tfs=8.8,bpad=.04 if i in(2,4,5)else .007)
    for i in range(5):arrow(fig,X[i]+bw+.001,ry+rh*.5,X[i+1]-.001,ry+rh*.5,TEAL)
    arrow(fig,.30,my-.002,X[2]+bw*.5,ry+rh+.002,VIOLET,lw=1.4);arrow(fig,.74,my-.002,X[4]+bw*.5,ry+rh+.002,AMBER,lw=1.8)
    chip(fig,X[2]+.006,ry+.018,T_["chipA"].format(a=Rd["address_ms"]),TEAL);chip(fig,X[4]+.006,ry+.018,T_["chipL"].format(l=Rd["load_ms"]),AMBER);chip(fig,X[5]+.006,ry+.018,T_["chipN"].format(n=Rd["answer_ms"]),TEAL)
    res=("correct"if L["correct"]else"incorrect")if lang=="en"else("doğru"if L["correct"]else"yanlış")
    footer(fig,ctx,k,T_["cap"].format(k=k,rid=P["run_id"],L=Wr["layers"],n=f"{n:,}",sel=f"{Rd['selected']:04d}",ans=L["answer"],exp=L["expected"],res=res,o=S["ORIGINAL"]["answer_correct"],a=S["ALTERNATE"]["answer_correct"],avg=avg),fs_max=8.0)
    exp=[Wr["file"],Rd["file"],f"#{Rd['selected']:04d}",Wr["file_sha256"][:16]]
    return finish(fig,path,ctx,exp,"WRITE · SSD · READ architecture"+(" (Turkish)"if lang=="tr"else""),register=(lang=="en"))
def f1(ctx,k,path):return f_arch(ctx,k,path,"en")
# ------------------------------------------------------------------ FIG 2 · WRITE → SSD in detail
def f2(ctx,k,path):
    P=ctx["P"];L=L_(ctx);cfg=P["engine"]["config"];su=P["startup"];Wr=L["write"];Sd=L["ssd"];fig=new_fig();n=cfg["n_cartridges"]
    header(fig,k,"WRITE → SSD · how one memory cartridge is formed and stored",f"The fact of case {L['case']:02d} written again by the engine's init(fact, 3) in this run and compared with the file written at start-up · the pool of {n:,} files",
           "WRITE · SSD · LIVE",BLUE)
    y0,h0=.66,.215;bx=[(.03,.17),(.215,.205),(.435,.205)]
    f1_,f2_=Wr["forwards"][0],Wr["forwards"][1]
    sblock(fig,bx[0][0],y0,bx[0][1],h0,"INPUT","Instruction prefix + fact",f"“{L['fact']}”\n{Wr['prefix_tokens']} prefix + {Wr['fact_tokens']} fact tokens = {Wr['T']} rows",BLUE,bfs=8.2)
    sblock(fig,bx[1][0],y0,bx[1][1],h0,"FORWARD 1","Token-id pass over the text",f"{fw_txt(f1_)}\nkeeps K/V of layers 0–3\ncaptures the layer-3 hidden state (H3)",BLUE,bfs=8.0,mono=True)
    sblock(fig,bx[2][0],y0,bx[2][1],h0,"FORWARD 2","Zero-input pass, H3 injected",f"{fw_txt(f2_)}\nH3 replaces the layer-3 output\n→ K/V of layers 4–27",BLUE,bfs=8.0,mono=True)
    arrow(fig,bx[0][0]+bx[0][1]+.002,y0+h0*.5,bx[1][0]-.002,y0+h0*.5,BLUE);arrow(fig,bx[1][0]+bx[1][1]+.002,y0+h0*.5,bx[2][0]-.002,y0+h0*.5,BLUE)
    # cartridge strip: 28 layers, K and V
    ax=fig.add_axes([.06,.37,.57,.23]);NL=Wr["layers"];T=Wr["T"]
    for li in range(NL):
        col=BLUE if li<=CUT_EXPECTED else"#60A5FA"
        ax.add_patch(Rectangle((li,0.55),.9,.4,fc=col,ec="white",lw=.5));ax.add_patch(Rectangle((li,0.05),.9,.4,fc=col,ec="white",lw=.5,alpha=.75))
    ax.set_xlim(-.4,NL+.2);ax.set_ylim(-.35,1.25);ax.set_yticks([.25,.75]);ax.set_yticklabels(["V","K"]);ax.set_xticks([1.95,15.45]);ax.set_xticklabels(["layers 0–3","layers 4–27"])
    for s in("top","right","left"):ax.spines[s].set_visible(False)
    ax.text(1.9,1.1,"from the fact forward (layers 0–3)",fontsize=8,color=BLUE,ha="center",weight="bold");ax.text(15.5,1.1,"rebuilt from the layer-3 state alone (layers 4–27)",fontsize=8,color="#2563EB",ha="center",weight="bold")
    ax.set_title(f"The cartridge: {NL} layers × (K, V) = {Wr['tensors']} tensors, each {Wr['shape']} {Wr['dtype']} · {Wr['memory_bytes']:,} bytes",loc="left",fontsize=9.2)
    # file box
    fx,fw_=.66,.31
    frame(fig,fx,.30,fw_,.575,ec=AMBER,fc=AMBER_L,lw=1.0);fig.text(fx+.01,.858,"FILE ON DISK · THIS FACT",fontsize=8.8,weight="bold",color=AMBER,va="center")
    body=(f"{Wr['file']}\n{Wr['file_bytes']:,} bytes on disk\n= {Wr['memory_bytes']:,} tensor bytes\n+ {Wr['container_overhead']:,} container bytes\nSHA-256 (= manifest {tick(Wr['sha_ok'])})\n  {Wr['file_sha256'][:32]}\n  {Wr['file_sha256'][32:]}\n"
          f"written at start-up in {Wr['startup_write_ms']:.0f} ms\n\nTHIS RUN\nre-written by init(fact, 3): {1000*Wr['seconds']:.0f} ms\nbitwise identical in all {Wr['layers']} layers {tick(Wr['bit_equal'])}\n"
          f"fact text inside the file: {yn(Wr['fact_in_file'])}\nanswer word inside the file: {yn(Wr['gold_in_file'])}\nquestion inside the file: {yn(Wr['question_in_file'])}")
    fit_text(fig,fx+.01,.31,fw_-.02,.53,body,fs_max=8.8,fs_min=6.4,family=MONO,color=INK)
    # pool histogram
    ax2=fig.add_axes([.06,.175,.40,.075]);sz=[s/1e6 for s in su["sizes"]];ax2.hist(sz,bins=40,color="#FCD34D",edgecolor=AMBER,lw=.4);clean(ax2)
    ax2.axvline(Wr["file_bytes"]/1e6,color=BLUE,lw=1.4);ax2.set_xlabel("file size (MB) · blue line = this fact",fontsize=7.6,labelpad=1);ax2.tick_params(labelsize=7)
    ax2.set_title(f"SSD pool · {n:,} independent files",loc="left",fontsize=8.8)
    fit_text(fig,.49,.16,.48,.11,(f"{n:,} files present in this run, every size = manifest {tick(not Sd['size_mismatch'])} · total {Sd['total_bytes']:,} bytes ({gb(Sd['total_bytes'])})\n"
              f"min {Sd['min']/1e6:.2f} · median {Sd['median']/1e6:.2f} · max {Sd['max']/1e6:.2f} MB · SHA-256 of all {n:,} files verified by the engine at start-up {tick(Sd['startup_sha_verified'])}\n"
              f"start-up WRITE: {su['write']['files']:,} files · p50 {su['write']['ms_p50']:.0f} ms per file · {su['write']['seconds_total']:.0f} s in total"),fs_max=8.4,fs_min=6.4,color=INK)
    footer(fig,ctx,k,f"Figure {k}. WRITE and SSD for case {L['case']:02d}. The frozen model reads the fact once: forward 1 keeps the keys and values of layers 0–3 and the layer-3 hidden state; forward 2 starts from all-zero input embeddings, "
           f"injects that state at layer 3 and produces the keys and values of layers 4–27. The {Wr['tensors']} tensors are the cartridge; they were saved at start-up as {Wr['file']}, and the same call in this run gives a bitwise identical result. "
           f"The file holds tensors only: neither the fact text nor the answer word is in it. Sizes: {Wr['file_bytes']:,} bytes for this fact; {gb(Sd['total_bytes'])} for {n:,} facts.")
    return finish(fig,path,ctx,[Wr["file"],f"{Wr['file_bytes']:,}",Wr["file_sha256"][:32],f"{n:,}"],"WRITE and SSD")
# ------------------------------------------------------------------ FIG 3 · READ in detail
def f3(ctx,k,path):
    P=ctx["P"];L=L_(ctx);cfg=P["engine"]["config"];Rd=L["read"];Ct=L["control"];ix=P["startup"]["index"];fig=new_fig();n=cfg["n_cartridges"]
    header(fig,k,"READ · from a question to the right file and an answer",f"Case {L['case']:02d} · “{L['question']}” · label-free geometric address in RAM → locked PEAK50 selection → file from SSD → frozen model answers",
           "READ · LIVE",TEAL)
    y0,h0=.70,.17;W_=.148;G_=(.94-6*W_)/5;X=[.03+i*(W_+G_)for i in range(6)];f0=Rd["frame_forwards"][0]
    st=[("1","Question frame",f"{fw_txt(f0)}\ninstruction prefix + question only"),("2","Layer-3 vectors",f"{Rd['query_rows']} question rows\n(engine question_rep + apply)\nencode {Rd['encode_ms']:.0f} ms (not in address)"),
        ("3",f"PCA{ix['pca_dims']} + IVF{ix['ivf_lists']}",f"nprobe {cfg['candidates'][0]} · token_k {cfg['candidates'][1]}\nFAISS {Rd['faiss_ms']:.2f} ms"),
        ("4","Touched cartridges",f"{Rd['touched']} of {n:,}\n({100*Rd['touched']/n:.1f} % of the pool)"),("5","TOP2 → candidates",f"{len(Rd['ranking'])} candidates\nTOP2 {Rd['top2_ms']:.1f} ms"),
        ("6","PEAK50 → one file",f"selected #{Rd['selected']:04d}\nPEAK50 {Rd['rerank_ms']:.1f} ms\naddress total {Rd['address_ms']:.1f} ms\n(= steps 3–6, as TEST627)")]
    for i,(s,t,b)in enumerate(st):sblock(fig,X[i],y0,W_,h0,s,t,b,TEAL,bfs=8.0,mono=True)
    for i in range(5):arrow(fig,X[i]+W_+.001,y0+h0*.5,X[i+1]-.001,y0+h0*.5,TEAL)
    # candidate scores
    ax=fig.add_axes([.07,.30,.45,.33]);C=Rd["candidates"];ys=list(range(len(C)))[::-1]
    cols=[TEAL if i==0 else"#CBD5E1"for i in range(len(C))]
    ax.barh(ys,[c["score"]for c in C],color=cols,edgecolor=[TEAL if i==0 else"#94A3B8"for i in range(len(C))],lw=.6)
    lo=min(c["score"]for c in C);hi=max(c["score"]for c in C);pad=max(1e-4,(hi-lo)*.15);ax.set_xlim(lo-pad,hi+pad*2.2)
    ax.set_yticks(ys);ax.set_yticklabels([f"#{c['id']:04d}"for c in C],fontsize=7,family=MONO);clean(ax);ax.grid(axis="y",visible=False)
    for i,c in enumerate(C):
        if c["id"]==L["target"]:ax.text(c["score"]+pad*.15,ys[i],"◆ expected (audit)",fontsize=7,color=AMBER,va="center",weight="bold")
    ax.set_xlabel("PEAK50 score = 0.5 × FRAM + 0.5 × peak (engine features626 / rerank626)",fontsize=7.8)
    ax.set_title(f"The {len(C)} candidates ranked by PEAK50 · teal = selected",loc="left",fontsize=9.2)
    # load + answer
    bx=.56;bw=.41
    frame(fig,bx,.47,bw,.18,ec=AMBER,fc=AMBER_L,lw=1.0);fig.text(bx+.01,.632,"LOAD FROM SSD",fontsize=8.6,weight="bold",color=AMBER,va="center")
    fit_text(fig,bx+.01,.478,bw-.02,.14,f"{Rd['file']} · SHA-256 = manifest {tick(Rd['sha_ok'])}\ntorch.load → {Rd['T']} rows × {ARCH[0]} layers → GPU\nload {Rd['load_ms']:.1f} ms = SHA-256 + torch.load + copy to GPU\n(may be served from the OS page cache)",fs_max=8.6,fs_min=6.4,family=MONO,color=INK)
    frame(fig,bx,.27,bw,.185,ec=TEAL,fc=TEAL_L,lw=1.0);fig.text(bx+.01,.437,"ANSWER · SOURCE TEXT NOT GIVEN",fontsize=8.6,weight="bold",color=TEAL,va="center")
    fa=Rd["answer_forwards"][0]
    fit_text(fig,bx+.01,.278,bw-.02,.145,(f"model input: {fa['n']} question-template tokens over the {Rd['T']}-row cartridge (cache {fa['past_len']})\n"
             f"then {Rd['gen_forwards']} one-token generation steps · {Rd['answer_ms']:.0f} ms\nanswer “{L['answer']}” · expected “{L['expected']}” → {'CORRECT'if L['correct']else'INCORRECT'}"),fs_max=8.6,fs_min=6.4,family=MONO,color=INK)
    frame(fig,.03,.15,.94,.10,ec=INK2,fc="white",lw=.9);fig.text(.04,.235,"CONTROL · runner-up file (diagnostic)",fontsize=8.6,weight="bold",color=INK2,va="center")
    ctl=("skipped: "+Ct["skipped"])if Ct.get("skipped")else(f"Same question over the second-ranked candidate #{Ct['id']:04d} ({Ct['file']}): answer “{Ct['answer']}” · expected answer produced: {yn(Ct['gold_produced'])}"
         +(" · this runner-up is the expected cartridge"if Ct["is_target"]else"")+f". Start-up engine row for this question: selected #{L['startup_row']['selected']} · answer “{L['startup_row']['answer']}” (same selection: {yn(L['matches_startup_selection'])}, same answer: {yn(L['matches_startup_answer'])}).")
    fit_text(fig,.04,.156,.92,.066,ctl,fs_max=8.6,fs_min=6.4,color=INK)
    footer(fig,ctx,k,f"Figure {k}. READ for case {L['case']:02d}. The question is read in one forward pass with the instruction prefix only — no fact text. Its layer-3 vectors are projected with the label-free PCA basis and looked up in the FAISS IVF index; "
           f"{Rd['touched']} of {n:,} cartridges are touched, {len(Rd['ranking'])} become candidates and the locked PEAK50 rule selects #{Rd['selected']:04d}. Its file is verified against the manifest, loaded, and the frozen model answers from it. "
           "The expected cartridge and answer are audit metadata only; they are not used by addressing, selection or answering.")
    return finish(fig,path,ctx,[Rd["file"],f"#{Rd['selected']:04d}",str(Rd["touched"])],"READ")
# ------------------------------------------------------------------ FIG 4 · RESULTS, validation and limits
def f4(ctx,k,path):
    P=ctx["P"];L=L_(ctx);su=P["startup"];fz=P["frozen"];S=su["summary"];T7=TEST627_LOG;fig=new_fig();Rd=L["read"];Wr=L["write"]
    header(fig,k,"RESULTS · measured performance, verification and limits","Start-up acceptance of the TEST627 engine in this runtime · the sealed TEST627 log · this run — side by side, with the limits that apply to all of them","RESULTS · VALIDATION",INK2)
    def col(x,ttl,c,body):
        frame(fig,x,.395,.30,.475,ec=c,fc="white",lw=.9);fig.text(x+.01,.852,ttl,fontsize=8.6,weight="bold",color=c,va="center")
        fit_text(fig,x+.01,.402,.282,.43,body,fs_max=9.6,fs_min=6.2,family=MONO,color=INK)
    def e2e(d):return(f"               original  new wording\nselected       {d['ORIGINAL']['selected_correct']:>2}/48     {d['ALTERNATE']['selected_correct']:>2}/48\nanswered       {d['ORIGINAL']['answer_correct']:>2}/48     {d['ALTERNATE']['answer_correct']:>2}/48\n"
                      f"errors         {d['ORIGINAL']['errors']:>5}     {d['ALTERNATE']['errors']:>5}\naddress p50 ms {d['ORIGINAL']['address_p50_ms']:>6.1f}    {d['ALTERNATE']['address_p50_ms']:>6.1f}\nload p50 ms    {d['ORIGINAL']['load_p50_ms']:>6.1f}    {d['ALTERNATE']['load_p50_ms']:>6.1f}\n"
                      f"answer p50 ms  {d['ORIGINAL']['answer_p50_ms']:>6.0f}    {d['ALTERNATE']['answer_p50_ms']:>6.0f}")
    st=su["storage"];I8=su["integrity"]
    col(.03,"START-UP ACCEPTANCE · THIS RUNTIME",BLUE,f"TEST627 engine, run at start-up\nfinished {str(su.get('measured_utc',''))[:19]} UTC\n\n"+e2e(S)+f"\n\nfiles {st['cartridges']:,} · {st['disk_pool_bytes']:,} B\ntensor payload {st['total_tensor_payload_bytes']:,} B\nintegrity {sum(bool(v)for v in I8.values())}/{len(I8)}: "
        +", ".join(k_.replace("_"," ")for k_,v in I8.items()if v)[:150]+f"\nresult SHA-256 {su['result_sha256'][:24]}…")
    col(.35,"ARCHIVED RECORD · 627.log",AMBER,"repository log of 12 October 2026\nnot re-run, reproduced verbatim\n\n"+e2e(T7["e2e"])+f"\n\nfiles {T7['storage']['cartridges']:,} · {T7['storage']['disk_pool_bytes']:,} B\nintegrity {sum(T7['integrity'].values())}/{len(T7['integrity'])}\ndecision\n"
        +uwrap(T7["decision"],34,"  ")+f"\nlog SHA-256 {T7['log_sha256'][:24]}…")
    col(.67,f"THIS RUN · CASE {L['case']:02d}",TEAL,(f"one new run · {P['run_start_utc'][:19]} UTC\n\nWRITE re-check   bitwise {yn(Wr['bit_equal'])}\nSSD files        {L['ssd']['files_in_pool']:,} = manifest\nselected         #{Rd['selected']:04d} ({'expected'if Rd['selected_correct']else'NOT expected'})\n"
        f"answer           {short(L['answer'],18)}\nexpected         {short(L['expected'],18)}\nresult           {'CORRECT'if L['correct']else'INCORRECT'}\naddress / load   {Rd['address_ms']:.1f} / {Rd['load_ms']:.1f} ms\nanswer           {Rd['answer_ms']:.0f} ms\n"
        f"control leak     {yn(L['control'].get('gold_produced'))}\nweights          unchanged\ntrainable        {fz['trainable_tensors']}\nchecks           {ctx['checks']} pre-seal PASS\nverdict\n"+uwrap(P["verdict"],34,"  ")))
    frame(fig,.03,.15,.94,.23,ec=ORANGE,fc=ORANGE_L,lw=1.0);fig.text(.04,.365,"LIMITS — SHOWN, NOT HIDDEN",fontsize=8.8,weight="bold",color=ORANGE,va="center")
    T6=TEST626_LOG
    fit_text(fig,.04,.16,.92,.19,(f"• Not a blind test: the 96 questions reuse the same 48 facts. TEST626's own pre-declared rule locked {T6['locked_by_own_rule']} ({T6['e2e']} answered); PEAK50 ({T6['peak50_new_wording']} new wording) was fixed after those results were known. FRAM-only ranking chose the right file for {T6['fram_only_new_wording']}.\n"
              f"• 1,453 files do not show behaviour at millions of cartridges. Disk reads may come from the operating-system page cache (no cold-SSD claim).\n"
              f"• A cartridge is ~{st['avg_file_bytes']/1e6:.2f} MB per fact — the model's working state, not a compact text copy. RAM keeps the address index (FAISS {su['index']['index_bytes']/1e6:.1f} MB) and the reranking data (native layer-3 vectors, {su['index']['native_activation_bytes']/1e6:.0f} MB); model weights are in GPU memory.\n"
              f"• Lossless gzip: ratio {st['gzip_sample_ratio']} on {st['lossless_gzip_sample_files']} sample files only; the full-pool value is an estimate. • Synthetic capital-of-country facts; one model. • Measured on an NVIDIA A100 40 GB only; other hardware is not tested. • SHA-256 values show files are unchanged; they are not proofs.\n"
              f"• The three columns are separate measurements: no counter merges them."),fs_max=9.4,fs_min=6.4,color=INK)
    footer(fig,ctx,k,f"Figure {k}. Results of run {P['run_id']} (verdict {P['verdict']}). The left column was produced by the locked TEST627 engine at start-up in this runtime; the middle column is the sealed repository log; the right column is this run. "
           f"Earlier public records: DOI {PRIOR_PUB['doi']} · DOI {QWEN_PUB['doi']} · DOI {V12_PUB['doi']}. Payload SHA-256 {ctx['sha']}.")
    return finish(fig,path,ctx,[T7["decision"][:20],ctx["sha"][:12],f"{S['ORIGINAL']['answer_correct']:>2}/48"],"Results and validation")
FIGURES=[("fig_01_write_ssd_read","Fig. 1 · The Silicon Hippocampus — WRITE · SSD · READ",f1),("fig_02_write_ssd","Fig. 2 · WRITE → SSD: one cartridge formed and stored",f2),
         ("fig_03_read","Fig. 3 · READ: from the question to the file and the answer",f3),("fig_04_results","Fig. 4 · Results, verification and limits",f4)]
N_FIGS=len(FIGURES)
def render_all(ctx,run_dir):
    imgs=[];extra=[]
    try:
        for i,(slug,cap,fn)in enumerate(FIGURES,1):
            p=Path(run_dir)/f"{slug}_{ctx['run_id']}.jpg";fn(ctx,i,p);imgs.append((p,cap))
        p=Path(run_dir)/f"fig_01_TR_yaz_ssd_oku_{ctx['run_id']}.jpg";f_arch(ctx,1,p,"tr");extra=[p,p.with_suffix(".pdf")]
    finally:plt.close("all")
    return imgs,extra
#<<FIGURES_END>>
say("PART 2 / 3 complete — RUN pipeline and figures defined. Now run PART 3 / 3 in the next cell.")
MAM_SH_PART2_OK=True
