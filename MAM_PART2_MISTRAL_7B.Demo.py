# =====================================================================================================================================
# AKBASCORE MAM · MISTRAL-7B ASSOCIATIVE RETRIEVAL INSTRUMENT — PART 2 / 3 · MEASUREMENT PIPELINE AND FIGURES
# Same demo as PART 1. Run this cell after PART 1 / 3, in the same Colab runtime; then run PART 3 / 3.
# Defines the fail-closed measurement run (engine API calls → re-derivation → sealed payload, logs, ZIP) and six figures
# (JPEG 300 dpi + vector PDF). Nothing in this part changes the engine or its retrieval computation.
# =====================================================================================================================================
if not globals().get("MAM_MISTRAL_PART1_OK"):raise RuntimeError("PART 1 / 3 has not been run in this runtime. Run PART 1, then PART 2, then PART 3.")
#<<CORE_BEGIN>>
import unicodedata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.patches import Rectangle,FancyBboxPatch,FancyArrowPatch,Patch
from matplotlib.lines import Line2D
from PIL import Image
ROOT=Path("/content/AKBASCORE_MAM_MISTRAL")if os.path.isdir("/content")else Path("/tmp/AKBASCORE_MAM_MISTRAL")
ROOT.mkdir(parents=True,exist_ok=True);ALLOWED_PATHS=[str(ROOT)]
N_STAGES=9;CF_OFFSET=53
def default_cf_target(item):return((item-1+CF_OFFSET)%N_CAND)+1
def ev(stage,title,body,done=None,total=None):return dict(stage=stage,title=title,body=body,done=done,total=total,eta=None)
def file_ready(p):return p is not None and Path(p).is_file()and Path(p).stat().st_size>0
def prune_runs(keep=4):
    runs=sorted([p for p in ROOT.glob("MAM528DEMO-*")if p.is_dir()],key=lambda p:p.stat().st_mtime)
    for p in runs[:max(0,len(runs)-keep)]:shutil.rmtree(p,ignore_errors=True)
def maxabs(a,b):return max((abs(float(x)-float(y))for x,y in zip(a,b)),default=0.0)
def fwd_ok_query(rec,nq,TA):return len(rec)==1 and rec[0]["has_past"]and rec[0]["input_len"]==nq and rec[0]["past_len"]==TA
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
def make_txt(P,sha,names,sealed_utc):
    o=[];a=o.append;S="="*140;Dd="-"*140;E_=P["environment"];L=P["live"];pr=L["primary"];D=L["primary_rederived"];ps=L["pointer_stats"];cf=L["counterfactual"];Dc=L["counterfactual_rederived"]
    fz=P["frozen"];cfg=P["engine"]["config"]
    a(S);a(f"{DEMO_TITLE} — READABLE RUN LOG");a(f"{AUTHOR} · {AUTHOR_PLACE} · {COPYRIGHT}");a(S)
    a(f"Derived from {names['payload']} (SHA-256 {sha}). Manifest: {names['manifest']}.")
    a("This is a live demonstration run of the frozen TEST528 mechanism. It is not TEST528; archived TEST528 values are reproduced verbatim, never recomputed.")
    for k_,v in(("RUN ID",P["run_id"]),("START UTC",P["run_start_utc"]),("END UTC",P["run_end_utc"]),("SEALED UTC",sealed_utc),("VERDICT",P["verdict"]),
        ("MODEL",f"{cfg['MODEL_ID']} · frozen · {P['model']['dtype']} · attention {P['model']['attn']}"),("ARCHITECTURE","32 layers · hidden 4096 · 32 Q heads · 8 KV heads · head 128 · GQA 4:1"),
        ("GPU",f"{E_['gpu']} (canonical: {CANON_GPU}; same: {P['hardware']['same']})"),("TORCH / TRANSFORMERS / GRADIO",f"{E_['torch']} / {E_['transformers']} / {E_['gradio']}"),
        ("ENGINE FILE",f"{P['engine']['file']} · SHA-256 {P['engine']['sha256']}"),("POINTER",f"L{cfg['PTR_L']:02d}H{cfg['PTR_H']:02d}"),
        ("ADDRESS",f"L{cfg['ADDR_L']:02d}-{cfg['ADDR_KIND']} · centered={cfg['ADDR_CENTERED']} · {cfg['KVD']} dimensions"),("CANDIDATES",f"{cfg['N']} precomputed B address matrices"),
        ("MATCH","max cosine over all rows of each B matrix; argmax over 128 candidates; ties → lower index"),("ENGINE SEED",cfg["seed"]),
        ("TEST528 LOCK",TEST528["lock"]),("TEST528 RESULT SHA",TEST528["result_sha"])):a(f"{k_:<30}: {v}")
    a("");a("ITEM");a(Dd)
    a(f"item {L['item']:03d} · entity {L['entity']} · question: {L['question']}");a(f"question tokens (FMT template) = {L['question_tokens']} · active A memory slots T = {L['a_slots']}")
    a(f"expected B candidate {L['item']:03d} · expected seal {pr['expected_seal']} (audit metadata; not an input to retrieval)")
    a("");a("PRIMARY RETRIEVAL · retrieve(item)  [LIVE]");a(Dd)
    a(f"selected B#{pr['selected_candidate']:03d} (seal {pr['selected_seal']}) · {'CORRECT'if pr['correct']else'INCORRECT'} · rank of expected {pr['rank']}/{N_CAND}")
    a(f"top1 {pr['top1_score']:+.6f} · top2 {pr['top2_score']:+.6f} · margin {pr['top1_top2_margin']:+.6f} · non-selected median {D['rest_median']:+.6f} · (top1−mean)/sd {('%.2f'%D['z_top1'])if D['z_top1']is not None else'n/a'}")
    a(f"pointer L28H00: argmax slot {ps['argmax']} of {ps['T']} · peak {ps['peak']:.4f} · entropy {ps['entropy_bits']:.3f} bits · effective slots {ps['effective_slots']:.2f}")
    a(f"diagnostic only: target-span positions {L['pointer_data']['gold_span_positions']} · pointer mass on span {pr['pointer_mass']:.4f} · pointer hit {pr['pointer_hit']}")
    a(f"latency (engine) {1000*pr['latency_seconds']:.1f} ms · measured model forwards {len(L['forwards']['retrieve'])}: {L['forwards']['retrieve']}")
    a("top-10: "+" ".join(f"#{t['candidate']:03d}:{t['score']:+.6f}"for t in pr["top10"]))
    a(f"repeat retrieve(): selected B#{L['repeat']['selected_candidate']:03d} · max |Δscore| {L['repeat_max_abs_diff']:.3e} · get_pointer_data() max |Δw| {L['pointer_data_max_abs_diff']:.3e}")
    ex=L["extras"];a(f"address p (1024-D) and best-matching row of selected B: slot {ex['best_slot']} of {ex['b_rows']} rows · cos {ex['best_cos']:+.6f} (recomputed with engine functions; equals reported score)")
    a("");a("CONTROLS · run_controls(item)  [LIVE, single item]");a(Dd)
    for k_,v in L["controls"]["controls"].items():
        d=L["controls_rederived"][k_];a(f"{k_:<8} selected B#{v['selected_candidate']:03d} · rank of expected {v['rank']:>3}/{N_CAND} · top1 {v['top1_score']:+.6f} · margin {v['margin']:+.6f}"+(" · all scores equal (zero address): argmax falls to B#001 by index"if d["all_equal"]else""))
    a(f"measured model forwards: {L['forwards']['controls']}")
    a("");a("COUNTERFACTUAL · counterfactual(item, target)  [LIVE]");a(Dd)
    a(f"original association: {L['entity']} → {cf['original_seal']} (B#{cf['original_candidate']:03d}) · counterfactual: {L['entity']} → {cf['counterfactual_seal']} (B#{cf['counterfactual_candidate']:03d})")
    a(f"selected B#{cf['selected_candidate']:03d} · followed counterfactual {cf['followed_counterfactual']} · rank of counterfactual target {cf['rank']}/{N_CAND} · rank of original B under counterfactual A {Dc['order'].index(cf['original_candidate'])+1}/{N_CAND}")
    a(f"top1 {cf['top1_score']:+.6f} · margin {cf['top1_top2_margin']:+.6f} · pointer mass on diagnostic span {cf['pointer_mass']:.4f} · hit {cf['pointer_hit']} · measured forwards {L['forwards']['counterfactual']}")
    a("");a("SEALED TEST528 REFERENCE (archived; not produced by this run)");a(Dd)
    for k_,lab in(("primary","PRIMARY"),("counterfactual","COUNTERFACTUAL"),("shifted","SHIFTED"),("no_a","NO-A")):v=TEST528[k_];a(f"{lab:<15} {fr(v)} = {v[2]} top-1")
    a(f"identity regime: {TEST528['identity_regime']}");a(f"cross-model context (archived): "+" | ".join(f"{c['record']} {c['model']} {c['pointer']} {c['address']} {c['dim']}-D {fr(c['top1'])} = {c['top1'][2]}"for c in CROSS_MODEL));a(CROSS_MODEL_NOTE)
    a("");a("MEASURED MODEL FORWARDS (instrumentation)");a(Dd)
    for k_,v in L["forwards"].items():a(f"{k_:<15}: {v}")
    a(f"every retrieval forward processed only the question tokens ({L['question_tokens']}) over the installed A cache ({L['a_slots']} slots); no B memory was installed or forwarded.")
    a("");a("ENGINE VERIFICATION · verify_engine() after the run");a(Dd)
    for k_,v in P["verify_after"].items():a(f"{k_:<34}: {v}")
    a("");a("FROZEN MODEL");a(Dd)
    for k_ in("sentinel_startup","sentinel_before","sentinel_after","guard_startup","guard_before","guard_after"):a(f"{k_:<18}: {fz[k_]}")
    a(f"trainable tensors={fz['trainable_tensors']} · lora={fz['lora']} · optimizer={fz['optimizer']} · training_mode={fz['training_mode']} · foreign hooks outside API calls {fz['foreign_hooks_before']}→{fz['foreign_hooks_after']}")
    a(fz["guard_method"]);a(fz["sentinel_method"])
    for t_,k_ in(("DEMONSTRATED IN THIS RUN","demonstrated"),("ARCHIVED","archived"),("NOT ESTABLISHED BY THIS RUN OR BY TEST528","not_established"),("TECHNICAL NOTES","notes")):
        a("");a(t_);a(Dd)
        for t in P["scope"][k_]:a("• "+t)
    a("");a("PRE-SEAL CHECKS");a(Dd)
    for c_ in P["checks_pre_seal"]:a("PASS · "+c_)
    a("");a("TIMING (seconds)");a(Dd)
    for k_,v in P["timing"].items():a(f"{k_:<24}: {v:.3f}")
    a(f"peak GPU memory {P['gpu']['peak_allocated_gib']:.2f} GiB");a("");a(f"All raw score vectors, pointer weights and the 1024-D address are in {names['payload']} and {names['jsonl']}.");a(S)
    return"\n".join(o)
def execute_run(I,ctl):
    """Fail-closed wrapper: on a technical audit failure the raw outputs gathered so far are preserved (never sealed, no figures)."""
    state={}
    try:return(yield from _execute_run(I,ctl,state))
    except AuditFail as ex:
        raw=state.get("raw")
        if raw and any(raw.values()):
            try:
                p=Path(ctl["run_dir"])/f"FAILED_AUDIT_RAW_{ctl['run_id']}.json"
                p.write_bytes(canon(dict(status="FAILED AUDIT — NOT SEALED",run_id=ctl["run_id"],failed_check=str(ex),checks_passed=state.get("checks",[]),raw=raw,note="Raw outputs preserved. No figures, no seal and no verdict were produced.")));ex.partial=str(p)
            except Exception:pass
        raise
def _execute_run(I,ctl,state):
    run_id=ctl["run_id"];run_dir=Path(ctl["run_dir"]);item=int(ctl["item"]);cft=int(ctl.get("cf_target")or default_cf_target(item))
    info=I.info;checks=[];tm=Counter();T0=time.perf_counter();state["checks"]=checks;raw=dict(primary=None,repeat=None,pointer=None,controls=None,counterfactual=None);state["raw"]=raw
    def chk(name,cond):
        checks.append(name)
        if not cond:raise AuditFail(name)
    run_start_utc=utc_now();say("="*140);say(f"RUN {run_id} · item {item:03d} · counterfactual target {cft:03d}");say("="*140)
    # ---- 1 integrity ----
    yield ev(1,"Integrity","Engine source hash, frozen configuration, weight sentinel, all-parameter guard, hooks.")
    I.reset_peak();cfg=I.config();st0=I.status();fz0=I.frozen();t=time.perf_counter();g_before=I.guard();tm["guard"]+=time.perf_counter()-t
    chk("engine source SHA-256 = supplied engine file",cfg["engine_sha256"]==ENGINE_SHA256_EXPECTED)
    chk("frozen configuration unchanged (L28H00 · L00-V · uncentered · 1024-D · 128 candidates · Mistral-7B-Instruct-v0.3)",all(cfg[k_]==v for k_,v in FROZEN_CONFIG.items())and cfg["n_bmats"]==128 and cfg["bmat_dims"]==[1024])
    chk("engine status reports L28H00 / L00-V UNCENTERED / 1024 / 128",st0.get("pointer")=="L28H00"and st0.get("address")=="L00-V UNCENTERED"and st0.get("address_dim")==1024 and st0.get("candidate_memories")==128)
    chk("archived TEST528 lock and result SHA match the engine constants",(cfg["test528_lock"],cfg["test528_result_sha"])==(TEST528["lock"],TEST528["result_sha"]))
    chk("weight sentinel before run = value at initialisation",fz0["sentinel"]==info["sentinel0"]);chk("all-parameter guard before run = value at startup",g_before==info["guard0"])
    chk("trainable tensors = 0, eval mode, no LoRA, no optimizer",fz0["trainable_tensors"]==0 and not fz0["training"]and not fz0["lora"]and not fz0["optimizer"])
    chk("no foreign forward/backward hooks outside API calls",fz0["foreign_hooks"]==0)
    chk("item and counterfactual target in 1..128 and distinct",1<=item<=N_CAND and 1<=cft<=N_CAND and cft!=item)
    its=I.items();it=its[item-1];question=it["question"];nq=I.question_tokens(question);TA=I.a_slots(item)
    chk("demo item metadata consistent (expected candidate = item)",it["item"]==item and it["expected_candidate"]==item)
    strip=I.panel_strings();state["strip"]=strip
    for c in checks:say("   PASS ·",c)
    # ---- 2 primary ----
    yield ev(2,"Primary retrieval",f"retrieve({item}) — {question}")
    r,fw_r,dt=I.retrieve(item);raw["primary"]=r;tm["retrieve"]+=dt
    chk("retrieve(): exactly one model forward — question tokens over the installed A cache only",fwd_ok_query(fw_r,nq,TA))
    D=rederive(r["candidate_scores"],item)
    chk("retrieve(): selection, rank, top-1/top-2 and margin re-derived from the 128 raw scores",D["selected"]==r["selected_candidate"]and D["rank"]==r["rank"]and D["top1"]==r["top1_score"]and D["top2"]==r["top2_score"]and abs(D["margin"]-r["top1_top2_margin"])<=1e-12)
    chk("retrieve(): question = engine template for the item; mode PRIMARY",r["question"]==question and r["mode"]=="PRIMARY"and r["item"]==item)
    w=r["pointer_weights"];ps=pointer_stats(w)
    chk("pointer: T weights over the A slots, slot 0 excluded, sum = 1",ps["T"]==TA and ps["w0"]==0.0 and abs(ps["sum"]-1)<1e-5 and ps["argmax"]==r["pointer_argmax_position"])
    chk("retrieve(): engine reports 0 query-time candidate-B forwards and frozen model",r["query_time_candidate_B_forwards"]==0 and r["model_frozen"])
    say(f"   selected B#{r['selected_candidate']:03d} · rank {r['rank']} · top1 {r['top1_score']:+.6f} · margin {r['top1_top2_margin']:+.6f} · {1000*dt:.1f} ms")
    # ---- 3 repeat + pointer telemetry ----
    yield ev(3,"Repeatability and pointer telemetry","retrieve() repeated; get_pointer_data(); address and best-matching B row recomputed with engine functions.")
    r2,fw_r2,dt=I.retrieve(item);raw["repeat"]=r2;tm["repeat"]+=dt;rep_diff=maxabs(r["candidate_scores"],r2["candidate_scores"])
    chk("repeat retrieve(): one forward of the same form; same selected candidate",fwd_ok_query(fw_r2,nq,TA)and r2["selected_candidate"]==r["selected_candidate"])
    pdat,fw_p,dt=I.pointer_data(item);raw["pointer"]=pdat;tm["pointer"]+=dt;w_diff=maxabs(w,pdat["weights"])
    chk("get_pointer_data(): one forward; same pointer argmax as retrieve()",fwd_ok_query(fw_p,nq,TA)and pdat["argmax_position"]==r["pointer_argmax_position"])
    ex=I.extras(item,w,r["selected_candidate"])
    chk("address recomputed with engine functions: 1024-D; best row cosine = reported score of the selected B",len(ex["p"])==ADDR_DIM and abs(ex["best_cos"]-r["candidate_scores"][r["selected_candidate"]-1])<=1e-6)
    # ---- 4 controls ----
    yield ev(4,"Controls","run_controls(): PRIMARY · UNIFORM · SHIFTED · NO-A on the same item.")
    c,fw_c,dt=I.controls(item);raw["controls"]=c;tm["controls"]+=dt
    chk("run_controls(): one forward of the query form",fwd_ok_query(fw_c,nq,TA))
    chk("run_controls(): modes PRIMARY, UNIFORM, SHIFTED, NO_A",sorted(c["controls"])==["NO_A","PRIMARY","SHIFTED","UNIFORM"])
    DC={k_:rederive(v["candidate_scores"],item)for k_,v in c["controls"].items()}
    chk("run_controls(): every mode re-derived from its raw scores",all(DC[k_]["selected"]==v["selected_candidate"]and DC[k_]["rank"]==v["rank"]for k_,v in c["controls"].items()))
    ctl_diff=maxabs(c["controls"]["PRIMARY"]["candidate_scores"],r["candidate_scores"])
    chk("run_controls() PRIMARY selects the same candidate as retrieve()",c["controls"]["PRIMARY"]["selected_candidate"]==r["selected_candidate"])
    # ---- 5 counterfactual ----
    yield ev(5,"Counterfactual",f"counterfactual({item}, {cft}) — A memory re-forged with the seal of candidate {cft:03d}.")
    f,fw_f,dt=I.counterfactual(item,cft);raw["counterfactual"]=f;tm["counterfactual"]+=dt
    chk("counterfactual(): two forwards — forge of the new A (no cache), then the question over that A",len(fw_f)==2 and not fw_f[0]["has_past"]and fw_f[1]["has_past"]and fw_f[1]["input_len"]==nq and fw_f[1]["past_len"]==fw_f[0]["input_len"])
    Dc=rederive(f["candidate_scores"],cft)
    chk("counterfactual(): selection and rank re-derived from raw scores",Dc["selected"]==f["selected_candidate"]and Dc["rank"]==f["rank"]and f["counterfactual_candidate"]==cft and f["original_candidate"]==item)
    # ---- 6 verification ----
    yield ev(6,"Engine verification","verify_engine(), weight sentinel, all-parameter guard, hooks.")
    v=I.verify();fz1=I.frozen();t=time.perf_counter();g_after=I.guard();tm["guard"]+=time.perf_counter()-t
    chk("verify_engine(): sentinel PASS · trainable 0 · 128 candidates · L28H00 · L00-V UNCENTERED · 1024",v["weight_sentinel_pass"]and v["trainable_tensors"]==0 and v["candidate_count"]==128 and v["pointer"]=="L28H00"and v["address"]=="L00-V UNCENTERED"and v["address_dim"]==1024)
    chk("all-parameter guard after run = before run = startup (weights unchanged)",g_after==g_before==info["guard0"]);chk("weight sentinel after run = value at initialisation",fz1["sentinel"]==info["sentinel0"])
    chk("trainable tensors = 0 and no foreign hooks after run",fz1["trainable_tensors"]==0 and fz1["foreign_hooks"]==0)
    fwd=dict(retrieve=fw_r,repeat=fw_r2,pointer_data=fw_p,controls=fw_c,counterfactual=fw_f);tm["engine_total"]=time.perf_counter()-T0
    verdict=f"ITEM_{item:03d}_PRIMARY_{'CORRECT'if r['correct']else'INCORRECT'}_CF_{'FOLLOWED'if f['followed_counterfactual']else'NOT_FOLLOWED'}_INTEGRITY_VERIFIED"
    say(f"   counterfactual → B#{f['selected_candidate']:03d} (target {cft:03d}) · verdict {verdict}")
    # ---- 7 seal ----
    yield ev(7,"Sealing","Payload, manifest, readable log and raw vectors are written and hashed.")
    peak=I.peak_gib();same_hw=info["gpu"]==CANON_GPU
    live=dict(item=item,entity=it["entity"],question=question,question_tokens=nq,a_slots=TA,primary=r,primary_rederived=D,pointer_stats=ps,repeat=r2,repeat_max_abs_diff=rep_diff,
              pointer_data=pdat,pointer_data_max_abs_diff=w_diff,extras=ex,controls=c,controls_rederived=DC,controls_primary_vs_retrieve_max_abs_diff=ctl_diff,
              counterfactual=f,counterfactual_rederived=Dc,cf_target=cft,cf_target_rank_under_original_A=D["order"].index(cft)+1,
              original_rank_under_counterfactual_A=Dc["order"].index(item)+1,forwards=fwd,b_tokens=[x["B_tokens"]for x in I.field()])
    P={"schema":"akbascore.mam.mistral.instrument.run.v1","demo":DEMO_TITLE,"author":AUTHOR,"place":AUTHOR_PLACE,"copyright":COPYRIGHT,"run_id":run_id,"run_start_utc":run_start_utc,"run_end_utc":utc_now(),
       "verdict":verdict,"model":{"id":cfg["MODEL_ID"],"arch":info["arch"],"dtype":info["dtype"],"attn":info["attn"],"params":info["params"],"vocab":info["vocab"],"pad_id":info["pad_id"]},
       "environment":{k_:info[k_]for k_ in("gpu","gpu_total_gib","torch","transformers","gradio","python","platform")},"hardware":{"canonical_gpu":CANON_GPU,"this_run_gpu":info["gpu"],"same":bool(same_hw)},
       "engine":{"file":info["engine_file"],"sha256":cfg["engine_sha256"],"config":cfg,"status_before":st0,"init_seconds":info["init_seconds"],"instrument":I.kind},
       "live":live,"verify_after":v,
       "frozen":{"sentinel_startup":info["sentinel0"],"sentinel_before":fz0["sentinel"],"sentinel_after":fz1["sentinel"],"guard_startup":info["guard0"],"guard_before":g_before,"guard_after":g_after,
                 "trainable_tensors":fz1["trainable_tensors"],"lora":fz1["lora"],"optimizer":fz1["optimizer"],"training_mode":fz1["training"],"foreign_hooks_before":fz0["foreign_hooks"],
                 "foreign_hooks_after":fz1["foreign_hooks"],"guard_method":info["guard_method"],"sentinel_method":info["sentinel_method"],"forward_counter":info["forward_counter"]},
       "archived":{"test528":TEST528,"cross_model":CROSS_MODEL,"cross_model_note":CROSS_MODEL_NOTE,"note":"Archived reference values; not produced or recomputed by this run."},
       "scope":SCOPE,"checks_pre_seal":list(checks),"timing":dict(tm),"gpu":{"peak_allocated_gib":peak}}
    P=jsafe(P);allow=(info["gpu"],CANON_GPU)
    hits=claim_scan("payload",strip_words(json.dumps(P,ensure_ascii=False),strip),allow=allow)
    chk("claim-discipline scan of the payload: 0 hits",not hits);P["claim_scan"]={"rules":len(CLAIM_RULES)+1,"hits":0,"scope":"payload JSON text; panel strings (names, seals) excluded"}
    pb=canon(P);sha=hashlib.sha256(pb).hexdigest();payload_name=f"{run_id}_FULL_RUN_LOG_payload.json";pp=run_dir/payload_name;t_seal=time.perf_counter();pp.write_bytes(pb);say("payload sealed:",sha)
    chk("payload SHA-256 verification after write",hashlib.sha256(pp.read_bytes()).hexdigest()==sha)
    names=dict(payload=payload_name,manifest=f"run_manifest_{run_id}.json",txt=f"{run_id}_READABLE_RUN_LOG.txt",summary=f"{run_id}_SUMMARY.json",jsonl=f"{run_id}_RAW_VECTORS.jsonl",images=f"figures_manifest_{run_id}.json",zip=f"AKBASCORE_MAM_MISTRAL_EVIDENCE_{run_id}.zip")
    mp=run_dir/names["manifest"];sealed_utc=utc_now()
    manifest={"demo":DEMO_TITLE,"author":AUTHOR,"copyright":COPYRIGHT,"run_id":run_id,"payload_file":payload_name,"payload_sha256":sha,"payload_bytes":len(pb),"sealed_utc":sealed_utc,"verdict":verdict,
              "engine_sha256":cfg["engine_sha256"],"test528_lock":TEST528["lock"],"test528_result_sha":TEST528["result_sha"],
              "canonicalization":"json.dumps(sort_keys=True, separators=(',',':'), ensure_ascii=False, allow_nan=False), UTF-8",
              "note":"Artifact integrity seal: confirms the payload is unchanged since sealing. It is not a scientific proof and not a third-party verification."}
    mp.write_text(json.dumps(manifest,indent=2,sort_keys=True,ensure_ascii=False),encoding="utf-8")
    summary={"run_id":run_id,"verdict":verdict,"item":item,"entity":it["entity"],"question":question,"expected_candidate":item,"expected_seal":r["expected_seal"],"selected_candidate":r["selected_candidate"],
             "selected_seal":r["selected_seal"],"rank":r["rank"],"top1":r["top1_score"],"top2":r["top2_score"],"margin":r["top1_top2_margin"],"pointer_hit_diagnostic":r["pointer_hit"],"pointer_mass_diagnostic":r["pointer_mass"],
             "controls":{k_:{"selected":v_["selected_candidate"],"rank":v_["rank"]}for k_,v_ in c["controls"].items()},"counterfactual":{"target":cft,"selected":f["selected_candidate"],"followed":f["followed_counterfactual"],"rank":f["rank"]},
             "measured_model_forwards":{k_:len(v_)for k_,v_ in fwd.items()},"weights_unchanged":True,"engine_sha256":cfg["engine_sha256"],"payload_sha256":sha,
             "archived_test528":{"primary":TEST528["primary"],"counterfactual":TEST528["counterfactual"],"shifted":TEST528["shifted"],"no_a":TEST528["no_a"]}}
    (run_dir/names["summary"]).write_text(json.dumps(jsafe(summary),indent=2,ensure_ascii=False),encoding="utf-8")
    lines=[{"op":"retrieve","item":item,"pointer_weights":w,"candidate_scores":r["candidate_scores"]},{"op":"address","item":item,"p_1024":ex["p"],"best_row_selected_B":ex["best_row"],"best_slot":ex["best_slot"]}]
    lines+=[{"op":"control","mode":k_,"candidate_scores":v_["candidate_scores"]}for k_,v_ in c["controls"].items()]
    lines+=[{"op":"counterfactual","item":item,"target":cft,"pointer_weights":f["pointer_weights"],"candidate_scores":f["candidate_scores"]}]
    (run_dir/names["jsonl"]).write_text("\n".join(json.dumps(jsafe(x),ensure_ascii=False)for x in lines)+"\n",encoding="utf-8")
    (run_dir/names["txt"]).write_text(make_txt(P,sha,names,sealed_utc),encoding="utf-8");seal_s=time.perf_counter()-t_seal
    # ---- 8 figures ----
    yield ev(8,"Rendering figures",f"{N_FIGS} figures at 300 dpi (JPEG) with vector PDF copies; every printed value is checked against the sealed payload.")
    ctx={"P":P,"sha":sha,"run_id":run_id,"payload_name":payload_name,"allow":allow,"strip":strip,"N":N_FIGS,"audit":[],"pdfs":[]}
    t_r=time.perf_counter();imgs=render_all(ctx,run_dir);render_s=time.perf_counter()-t_r;entries=[]
    for n_,(p,cap)in enumerate(imgs,1):
        with Image.open(p)as im:w_,h_=im.size
        q_=ctx["pdfs"][n_-1];entries.append({"index":n_,"jpg":p.name,"pdf":q_.name,"title":cap,"width_px":w_,"height_px":h_,"dpi":OUT_DPI,"jpg_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"pdf_sha256":hashlib.sha256(q_.read_bytes()).hexdigest()})
    imf=run_dir/names["images"]
    imf.write_text(json.dumps({"run_id":run_id,"payload_sha256":sha,"verdict":verdict,"count":len(entries),"render_seconds":render_s,"figures":entries,"figure_text_audit":ctx["audit"],
                               "note":"Per-file SHA-256 is an artifact integrity seal. Figure text was checked against the sealed payload and scanned for claim discipline before saving."},indent=2,ensure_ascii=False),encoding="utf-8")
    # ---- 9 package ----
    yield ev(9,"Packaging","ZIP with figures, PDFs, payload, manifest, logs and raw vectors; post-seal audit.")
    zp=run_dir/names["zip"];members=[p for p,_ in imgs]+list(ctx["pdfs"])+[imf,mp,pp,run_dir/names["txt"],run_dir/names["summary"],run_dir/names["jsonl"]]
    for m in members:
        if m.suffix in(".json",".txt",".jsonl"):chk(f"claim-discipline scan of {m.name}: 0 hits",not claim_scan(m.name,strip_words(m.read_text(encoding="utf-8"),strip),allow=allow))
    with zipfile.ZipFile(zp,"w",compression=zipfile.ZIP_DEFLATED)as z:
        for m in members:z.write(m,arcname=m.name)
    post=post_seal_audit(imgs,ctx["pdfs"],zp,[m.name for m in members],pp,sha,run_dir/names["txt"],mp,verdict)
    say(f"{run_id}: {len(checks)} pre-seal + {len(post)} post-seal checks = {len(checks)+len(post)}/{len(checks)+len(post)} PASS · figures {render_s:.1f}s · seal {seal_s:.2f}s")
    say("VERDICT :",verdict);say("PAYLOAD :",sha);say("ZIP     :",zp);say("="*140)
    ledger=dict(run_id=run_id,utc=sealed_utc,item=item,entity=it["entity"],selected=r["selected_candidate"],rank=r["rank"],correct=r["correct"],margin=r["top1_top2_margin"],cf_target=cft,
                cf_selected=f["selected_candidate"],cf_followed=f["followed_counterfactual"],controls={k_:v_["rank"]for k_,v_ in c["controls"].items()},forwards={k_:len(v_)for k_,v_ in fwd.items()},
                weights_unchanged=True,verdict=verdict,payload_sha256=sha)
    return dict(run_id=run_id,run_dir=run_dir,P=P,sha=sha,imgs=imgs,pdfs=ctx["pdfs"],zip=zp,payload=pp,txt=run_dir/names["txt"],manifest=mp,images_manifest=imf,summary=run_dir/names["summary"],
                jsonl=run_dir/names["jsonl"],checks=len(checks)+len(post),verdict=verdict,ledger=ledger)
#<<CORE_END>>
#<<FIGURES_BEGIN>>
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":.8,"axes.edgecolor":"#374151","axes.labelcolor":"#111827","xtick.color":"#374151","ytick.color":"#374151",
                     "xtick.labelsize":8,"ytick.labelsize":8,"axes.labelsize":9,"axes.titlesize":9.5,"axes.titleweight":"bold","axes.titlelocation":"left","pdf.fonttype":42,"ps.fonttype":42,
                     "legend.fontsize":8,"legend.frameon":False})
FIG_W,FIG_H=13.333,7.5;LAYOUT_DPI=100;OUT_DPI=300
INK,INK2,MUTED,RULE,GRIDC,PANEL="#111827","#374151","#6B7280","#D1D5DB","#E5E7EB","#F9FAFB"
BLUE,BLUE_L,ORANGE,ORANGE_L,TEAL,TEAL_L,GRAYPT,AMBER_L,AMBER="#1D4ED8","#DBEAFE","#C2410C","#FFEDD5","#0F766E","#CCFBF1","#9CA3AF","#FEF3C7","#B45309"
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
    fig.savefig(pdf,format="pdf",facecolor="white",metadata={"Title":desc,"Author":AUTHOR,"Subject":f"{DEMO_TITLE} · run {ctx['run_id']} · payload SHA-256 {ctx['sha']}","Creator":DEMO_SHORT,
                "Keywords":f"engine sha256 {ENGINE_SHA256}; TEST528 lock {TEST528['lock']}"})
    buf=io.BytesIO();fig.savefig(buf,format="png",dpi=OUT_DPI,facecolor="white");plt.close(fig);buf.seek(0)
    with Image.open(buf)as im:rgb=im.convert("RGB")
    ex=Image.Exif();ex[0x010E]=ascii_(f"{desc} | {DEMO_SHORT} | run {ctx['run_id']} | payload sha256 {ctx['sha']} | engine sha256 {ENGINE_SHA256}");ex[0x013B]=ascii_(AUTHOR);ex[0x8298]=ascii_(COPYRIGHT);ex[0x0131]=ascii_(DEMO_SHORT)
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
    fit_text(fig,.03,.04,.94,.085,caption,fs_max=8.6,fs_min=6.6,color=INK2)
    fig.add_artist(Line2D([.03,.97],[.033,.033],transform=fig.transFigure,color=RULE,lw=.6))
    fig.text(.5,.016,mt(f"{DEMO_SHORT} (TEST528 mechanism; not TEST528 itself) · run {ctx['run_id']} · engine SHA-256 {ENGINE_SHA256[:12]}… · TEST528 lock {TEST528['lock'][:12]}… · payload {ctx['sha'][:12]}… · Fig. {k}/{ctx['N']} · © 2026 {AUTHOR}"),
             ha="center",va="center",fontsize=6.6,color=MUTED,family=MONO)
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
def score_axes(ax,scores,expected,selected,exp_c=INK,sel_c=BLUE,ylab=True,annotate=True,ms=7):
    ax.set_zorder(3);x=np.arange(1,len(scores)+1);s=np.asarray(scores,dtype=float)
    ax.scatter(x,s,s=ms,color=GRAYPT,zorder=2,lw=0)
    ax.scatter([expected],[s[expected-1]],s=ms*12,facecolors="none",edgecolors=exp_c,linewidths=1.2,zorder=4)
    ax.scatter([selected],[s[selected-1]],s=ms*5,color=sel_c,zorder=5,lw=0)
    ax.set_xlim(0,len(s)+1);ax.set_xticks([1,16,32,48,64,80,96,112,128]);clean(ax)
    lo,hi=float(s.min()),float(s.max());pad=max(1e-3,(hi-lo)*.08);ax.set_ylim(lo-pad,hi+pad*2.2)
    if ylab:ax.set_ylabel("candidate score s_j")
    return s
# ------------------------------------------------------------------ FIG 1 · retrieval path
def f1(ctx,k,path):
    L=L_(ctx);P=ctx["P"];cfg=P["engine"]["config"];r=L["primary"];ex=L["extras"];bt=L["b_tokens"];fig=new_fig()
    header(fig,k,"Retrieval path of the frozen mechanism (as executed in this run)",f"Item {L['item']:03d} · dimensions, pointer trace, address and the 128 candidate scores below are taken from this run.","ARCHITECTURE · VALUES FROM THIS RUN",INK2)
    yT=.845;stages=[(.03,"(a) INPUTS"),(.215,"(b) FROZEN COMPUTATION"),(.425,"(c) RETRIEVAL SIGNAL"),(.615,"(d) ADDRESS"),(.775,"(e) STORED B FIELD · COMPARISON · OUTPUT")]
    for x,s in stages:fig.text(x,yT,s,fontsize=8,weight="bold",color=INK2,va="center")
    q=L["question"]
    block(fig,.03,.585,.165,.235,"INPUT 1","Natural-language query",f"{q}\n\nn = {L['question_tokens']} tokens\n(QUESTION/ANSWER template)",mono=False,bfs=8.6)
    block(fig,.03,.205,.165,.335,"INPUT 2 · STORED","Active A memory (numeric)",f"K^l, V^l ∈ R^(8×T×128)\nl = 0 … 31\nT = {L['a_slots']} slots\nslot 0 = [PAD] anchor\nforged once by a frozen\nforward of the A record\n(K installed with RoPE\npositions 0 … T−1)")
    block(fig,.215,.38,.19,.44,"FROZEN","Mistral-7B-Instruct-v0.3",f"32 layers · d = 4096\n32 Q / 8 KV heads · d_h = 128\nweights frozen (guard ✓)\n\nA installed as KV cache\nquery positions T … T+n−1\none forward per question\n\noutput used:\nh_28 = residual stream at the\nfinal query token entering\nlayer 28")
    block(fig,.425,.43,.17,.39,"ÇAĞRIİZ","Pointer · L28H00","q = RoPE(W_Q^28 RMSNorm(h_28))\n  query head 0\nk_t = RoPE(K^28_t)\n  KV head 0 (GQA 4:1)\nw_t = softmax_{t≥1}(q·k_t/√128)\nw_0 = 0 · Σ_t w_t = 1",bfs=7.9)
    ax=fig.add_axes([.433,.445,.154,.085]);ax.set_zorder(3);w=np.asarray(r["pointer_weights"]);ax.bar(np.arange(len(w)),w,color=BLUE,width=.85);ax.set_xlim(-.6,len(w)-.4);ax.set_yticks([]);ax.set_xticks([0,len(w)-1])
    ax.tick_params(labelsize=6.5,length=2);[ax.spines[s].set_visible(False)for s in("top","right","left")];ax.set_title(f"w over T = {len(w)} slots (this run)",fontsize=7,weight="normal",pad=2)
    block(fig,.615,.43,.145,.39,"ADDRESS","L00-V · uncentered","V^0_t = W_V^0 RMSNorm(e_t)\n  ∈ R^(8×128)\np = Σ_t w_t · V^0_t\n  ∈ R^1024 (8 heads × 128)\nno centering",bfs=7.9)
    ax=fig.add_axes([.622,.445,.131,.085]);ax.set_zorder(3);pv=np.asarray(ex["p"]).reshape(8,128);lim=float(np.percentile(np.abs(pv),98))or 1.0
    ax.imshow(pv,aspect="auto",cmap="RdBu_r",vmin=-lim,vmax=lim,interpolation="nearest");ax.set_xticks([]);ax.set_yticks([]);ax.set_title("p (8 × 128, this run)",fontsize=7,weight="normal",pad=2)
    block(fig,.78,.205,.19,.615,"STORED · NOT FORWARDED AT QUERY TIME","128 precomputed B matrices",f"M_j = V^0(B_j)[slots ≥ 1]\n    ∈ R^(T_j × 1024)\nj = 1 … 128 · T_j = {min(bt)} … {max(bt)}\n\ns_j = max_s cos(p, M_j[s])\nĵ = argmax_j s_j\n(ties → lower index)",bfs=7.9)
    frame(fig,.775,.198,.2,.632,ec=INK2,fc="none",lw=.9,ls=(0,(4,3)),z=0)
    ax=fig.add_axes([.787,.28,.176,.20]);score_axes(ax,r["candidate_scores"],L["item"],r["selected_candidate"],ylab=False,ms=3)
    ax.set_xticks([1,64,128]);ax.set_yticks([]);ax.spines["left"].set_visible(False);ax.tick_params(labelsize=6.5);ax.set_title("s_j for j = 1 … 128 (this run)",fontsize=7,weight="normal",pad=2)
    fig.text(.787,.243,f"selected B#{r['selected_candidate']:03d} · s = {r['top1_score']:.4f}",fontsize=8,weight="bold",color=BLUE,va="center")
    fig.text(.787,.218,f"measured forwards in retrieve(): {len(L['forwards']['retrieve'])}",fontsize=7.6,color=INK2,va="center")
    arrow(fig,.195,.70,.215,.66);arrow(fig,.405,.62,.425,.62);arrow(fig,.595,.62,.615,.62);arrow(fig,.76,.62,.78,.62)
    arrow(fig,.195,.47,.215,.47)
    fig.add_artist(Line2D([.195,.69],[.30,.30],transform=fig.transFigure,color=INK2,lw=1.0));arrow(fig,.51,.30,.51,.43);arrow(fig,.69,.30,.69,.43)
    fig.text(.503,.292,"keys K^28 of the A slots → pointer",fontsize=7.4,color=INK2,ha="right",va="top");fig.text(.683,.292,"values V^0 of the A slots → address",fontsize=7.4,color=INK2,ha="right",va="top")
    fig.text(.408,.632,"h_28",fontsize=7.4,color=INK2,ha="left",va="bottom");fig.text(.597,.632,"w",fontsize=7.4,color=INK2,ha="left",va="bottom");fig.text(.762,.632,"p",fontsize=7.4,color=INK2,ha="left",va="bottom")
    frame(fig,.215,.148,.545,.088,ec=RULE,fc=PANEL,lw=.6)
    fit_text(fig,.222,.152,.533,.08,"• V^0 is the layer-0 value projection of the RMS-normalised input token embedding; p is a pointer-weighted combination of token-level value vectors of the A slots.\n"
             "• Target-span (seal token) positions are computed by the engine for diagnostics only; they are not an input to any step shown.\n• Identity regime of the sealed TEST528 protocol: native uppercase single-token seals.",fs_max=7.8,fs_min=6.4,color=INK2)
    footer(fig,ctx,k,f"Figure {k}. Executed retrieval path of the frozen TEST528 mechanism for item {L['item']:03d}. A natural-language query and the active numeric A memory (installed as KV cache) drive one frozen forward of Mistral-7B-Instruct-v0.3. "
           f"At the final query token, head L28H00 scores the A slots (ÇAĞRIİZ pointer w). The pointer weights the layer-0 value vectors of the A slots into an uncentered 1024-D address p, which is compared by maximum cosine with every row of the 128 precomputed B matrices. "
           f"The B matrices are stored tensors; the instrumentation recorded {len(L['forwards']['retrieve'])} model forward in retrieve(), processing {L['question_tokens']} query tokens over a {L['a_slots']}-slot A cache.")
    return finish(fig,path,ctx,["L28H00","L00-V",f"T = {L['a_slots']} slots",f"selected B#{r['selected_candidate']:03d}",f"{r['top1_score']:.4f}"],"Retrieval path of the frozen mechanism")
# ------------------------------------------------------------------ FIG 2 · 128-way competition
def f2(ctx,k,path):
    L=L_(ctx);r=L["primary"];D=L["primary_rederived"];e=L["item"];sel=r["selected_candidate"];fig=new_fig();ok=r["correct"]
    header(fig,k,"128-way candidate competition for one query","Score of every B memory against the address p; expected candidate (ring) and selected candidate (filled).","LIVE DEMO OBSERVATION",BLUE)
    line=(f"item {e:03d} · {L['entity']} · expected B#{e:03d} ({r['expected_seal']}) · selected B#{sel:03d} ({r['selected_seal']}) · rank of expected {r['rank']}/{N_CAND} · "
          f"s1 = {r['top1_score']:.6f} · s2 = {r['top2_score']:.6f} · margin = {r['top1_top2_margin']:+.6f}")
    fig.text(.03,.865,mt(line),fontsize=8.8,family=MONO,color=INK,va="center");fig.text(.97,.865,"CORRECT"if ok else"INCORRECT",fontsize=11,weight="bold",color=BLUE if ok else ORANGE,ha="right",va="center")
    ax=fig.add_axes([.06,.255,.565,.545]);s=score_axes(ax,r["candidate_scores"],e,sel,ms=10)
    ax.axhline(r["top2_score"],color=MUTED,lw=.7,ls=(0,(3,3)),zorder=1);ax.set_xlabel("B candidate j (audit label; not an input to retrieval)")
    ax.set_ylabel("s_j = max_s cos(p, M_j[s])")
    rng_=float(s.max()-s.min())or 1.0
    ax.annotate(f"selected B#{sel:03d}\ns = {r['top1_score']:.6f}",xy=(sel,s[sel-1]),xytext=(sel+(10 if sel<90 else-34),s[sel-1]+.07*rng_),fontsize=8,color=BLUE,va="center",arrowprops=dict(arrowstyle="-",color=BLUE,lw=.7))
    if not ok:ax.annotate(f"expected B#{e:03d}\nrank {r['rank']}",xy=(e,s[e-1]),xytext=(e+(10 if e<90 else-30),s[e-1]-(s.max()-s.min())*.12),fontsize=8,color=INK,arrowprops=dict(arrowstyle="-",color=INK,lw=.7))
    ax.text(127,r["top2_score"],f"s2 = {r['top2_score']:.4f}",fontsize=7.4,color=MUTED,ha="right",va="bottom")
    ax.legend(handles=[Line2D([0],[0],marker="o",lw=0,markerfacecolor="none",markeredgecolor=INK,markersize=8,label=f"expected B#{e:03d} (panel construction)"),
                       Line2D([0],[0],marker="o",lw=0,color=BLUE,markersize=5,label="selected = argmax_j s_j"),Line2D([0],[0],marker="o",lw=0,color=GRAYPT,markersize=4,label="other candidates")],loc="upper right")
    zt=f"{D['z_top1']:.2f}"if D["z_top1"]is not None else"n/a"
    fig.text(.06,.19,mt(f"non-selected candidates (n = 127): median {D['rest_median']:.4f} · IQR {D['rest_q1']:.4f} – {D['rest_q3']:.4f} · max {D['rest_max']:.4f} · (s1 − mean)/sd = {zt}"),fontsize=8.2,color=INK2,family=MONO)
    fig.text(.06,.162,mt(f"repeat retrieve(): same selection · max |Δs| = {L['repeat_max_abs_diff']:.2e} · measured model forwards in retrieve(): {len(L['forwards']['retrieve'])} (query over A cache only)"),fontsize=8.2,color=INK2,family=MONO)
    ax2=fig.add_axes([.69,.60,.28,.20]);ss=np.sort(s)[::-1];ax2.plot(np.arange(1,129),ss,color=INK2,lw=1);ax2.scatter([r["rank"]],[s[e-1]],s=60,facecolors="none",edgecolors=INK,linewidths=1.1,zorder=4)
    ax2.scatter([1],[ss[0]],s=22,color=BLUE,zorder=5);clean(ax2);ax2.set_xlim(0,129);ax2.set_xlabel("rank");ax2.set_ylabel("score");ax2.set_title("Rank-ordered scores")
    fig.text(.69,.515,"Top 10",fontsize=9.5,weight="bold",color=INK);fig.text(.69,.487,mt(f"{'rank':>4} {'cand.':>6}  {'seal (audit)':<13}{'score':>10}{'Δ to s1':>11}"),fontsize=7.9,family=MONO,color=MUTED)
    for n_,t_ in enumerate(r["top10"]):
        y=.462-n_*.0238;hl=t_["candidate"]==e
        fig.text(.69,y,mt(f"{t_['rank']:>4} {'#'+format(t_['candidate'],'03d'):>6}  {t_['seal']:<13}{t_['score']:>10.6f}{t_['score']-r['top1_score']:>+11.6f}"),fontsize=7.9,family=MONO,color=INK if hl else INK2,weight="bold"if hl else"normal")
    footer(fig,ctx,k,f"Figure {k}. Live 128-way competition for item {L['item']:03d} ({L['entity']}). Each point is s_j = max over the slots of B memory j of the cosine between the 1024-D address p and that slot's L00-V vector. "
           f"The expected candidate is defined by panel construction (item i's target seal occurs only in B memory i); candidate numbers are audit labels and are not available to the retrieval computation. "
           f"Result: selected B#{sel:03d}, rank of expected {r['rank']}/{N_CAND}, margin {r['top1_top2_margin']:+.6f}. A single live item is a demonstration observation, not an estimate of accuracy.")
    return finish(fig,path,ctx,[f"s1 = {r['top1_score']:.6f}",f"s2 = {r['top2_score']:.6f}",f"margin = {r['top1_top2_margin']:+.6f}",f"rank of expected {r['rank']}/{N_CAND}"],"128-way candidate competition")
# ------------------------------------------------------------------ FIG 3 · pointer and address
def f3(ctx,k,path):
    L=L_(ctx);r=L["primary"];ex=L["extras"];ps=L["pointer_stats"];pdat=L["pointer_data"];fig=new_fig();w=np.asarray(r["pointer_weights"]);span=pdat["gold_span_positions"]
    header(fig,k,"ÇAĞRIİZ pointer (L28H00) and the resulting 1024-D address","Pointer distribution over the A memory slots, the address it produces, and the best-matching row of the selected B memory.","LIVE DEMO OBSERVATION",BLUE)
    ax=fig.add_axes([.055,.585,.89,.255]);x=np.arange(len(w))
    ax.axvspan(min(span)-.5,max(span)+.5,facecolor=AMBER_L,edgecolor=AMBER,hatch="////",lw=.6,zorder=0)
    ax.bar(x,w,color=[BLUE if t==ps["argmax"]else"#93A3C4"for t in x],width=.8,zorder=2);clean(ax);ax.set_xlim(-.6,len(w)-.4);ax.set_ylim(0,max(w)*1.18)
    ax.set_ylabel("pointer weight w_t")
    if ex["tokens"]:
        ax.set_xticks(x);ax.set_xticklabels([mt(t.replace("\n","\\n").strip()or"·")for t in ex["tokens"]],rotation=90,fontsize=7,family=MONO)
        ax.set_xlabel("A memory slot t · token labels decoded from the engine's forge-time record for display only (not available to retrieval)",fontsize=8)
    else:ax.set_xlabel("A memory slot t")
    info=(f"head L28H00 · T = {ps['T']} slots · argmax slot {ps['argmax']} · peak w = {ps['peak']:.4f}\nentropy {ps['entropy_bits']:.3f} bits · effective slots {ps['effective_slots']:.2f}\n"
          f"diagnostic span {span}: mass {r['pointer_mass']:.4f} · hit {r['pointer_hit']}")
    ax.text(.995,.97,mt(info),transform=ax.transAxes,ha="right",va="top",fontsize=7.8,family=MONO,color=INK,bbox=dict(boxstyle="square,pad=0.4",fc="white",ec=RULE,lw=.6))
    ax.legend(handles=[Patch(facecolor=AMBER_L,edgecolor=AMBER,hatch="////",lw=.6,label="diagnostic target span (seal token positions; not a retrieval input)"),Patch(color=BLUE,label="pointer argmax")],loc="upper right",bbox_to_anchor=(1.0,.70))
    pv=np.asarray(ex["p"]).reshape(8,128);bv=np.asarray(ex["best_row"]).reshape(8,128)
    for n_,(M,ttl)in enumerate(((pv,"Address p = Σ_t w_t · V^0_t   (8 KV heads × 128 dims, uncentered)"),(bv,f"Best-matching row of selected B#{r['selected_candidate']:03d}: M_ĵ[s*], s* = slot {ex['best_slot']} of {ex['b_rows']}"))):
        a=fig.add_axes([.055+n_*.49,.235,.39,.18]);lim=float(np.percentile(np.abs(M),98))or 1.0
        im=a.imshow(M,aspect="auto",cmap="RdBu_r",vmin=-lim,vmax=lim,interpolation="nearest");a.set_yticks(range(8));a.set_yticklabels([f"h{j}"for j in range(8)],fontsize=7)
        a.set_xticks([0,32,64,96,127]);a.set_xlabel("dimension within KV head",fontsize=8);a.set_title(mt(ttl),fontsize=8.6)
        cb=fig.colorbar(im,cax=fig.add_axes([.452+n_*.49,.235,.006,.18]));cb.ax.tick_params(labelsize=6.5);cb.outline.set_linewidth(.5)
    fig.text(.545,.152,mt(f"cos(p, M_ĵ[s*]) = {ex['best_cos']:+.6f} = reported s_ĵ (engine functions; |Δ| ≤ 1e-6 checked)"),fontsize=7.8,family=MONO,color=INK2)
    fig.text(.055,.152,mt(f"get_pointer_data(): max |Δw| vs retrieve() = {L['pointer_data_max_abs_diff']:.2e}"),fontsize=7.8,family=MONO,color=INK2)
    footer(fig,ctx,k,f"Figure {k}. Top: pointer distribution of attention head L28H00 at the final query token over the {ps['T']} slots of the active A memory (slot 0 excluded by the engine). "
           "The shaded span marks the positions of the target seal token; the engine computes it for diagnostics only and it does not enter the pointer, the address or the scoring. "
           "Bottom: the 1024-D address p formed from the layer-0 value vectors (left) and the B-memory row that attains the maximum cosine for the selected candidate (right); colour scales are per panel (98th percentile of |value|).")
    return finish(fig,path,ctx,[f"peak w = {ps['peak']:.4f}",f"mass {r['pointer_mass']:.4f}",f"{ex['best_cos']:+.6f}","L28H00"],"Pointer and address")
# ------------------------------------------------------------------ FIG 4 · counterfactual
def f4(ctx,k,path):
    L=L_(ctx);r=L["primary"];cf=L["counterfactual"];i=L["item"];j=L["cf_target"];fig=new_fig();D=L["primary_rederived"];Dc=L["counterfactual_rederived"]
    header(fig,k,"Counterfactual association changes the selected B memory","Same frozen weights, same question string, same 128 B matrices; only the A memory's seal association is changed and re-forged.","LIVE DEMO OBSERVATION",BLUE)
    rows=[("ORIGINAL A",f"{L['entity']} carries seal {cf['original_seal']}  →  expected B#{i:03d}",r["pointer_weights"],r["candidate_scores"],r["selected_candidate"],r["pointer_mass"],r["pointer_hit"],INK),
          ("COUNTERFACTUAL A",f"{L['entity']} carries seal {cf['counterfactual_seal']}  →  expected B#{j:03d}",cf["pointer_weights"],cf["candidate_scores"],cf["selected_candidate"],cf["pointer_mass"],cf["pointer_hit"],ORANGE)]
    for n_,(lab,assoc,w,sc,sel,mass,hit,col)in enumerate(rows):
        y0=.535-n_*.325
        fig.text(.03,y0+.255,lab,fontsize=10,weight="bold",color=col,va="center");fig.text(.16,y0+.255,mt(assoc),fontsize=9,family=MONO,color=INK,va="center")
        a=fig.add_axes([.03,y0,.25,.215]);w=np.asarray(w);a.bar(np.arange(len(w)),w,color=col if n_ else"#93A3C4",width=.8);a.set_xlim(-.6,len(w)-.4);clean(a)
        a.set_title(f"L28H00 pointer (T = {len(w)}) · span mass* {mass:.3f} · hit {hit}",fontsize=7.6,weight="normal");a.set_xlabel("A slot t",fontsize=7.8);a.tick_params(labelsize=7)
        b=fig.add_axes([.33,y0,.43,.215]);s=np.asarray(sc,dtype=float);x=np.arange(1,129)
        b.scatter(x,s,s=7,color=GRAYPT,lw=0,zorder=2);b.scatter([i],[s[i-1]],s=80,facecolors="none",edgecolors=INK,linewidths=1.2,zorder=4);b.scatter([j],[s[j-1]],s=80,facecolors="none",edgecolors=ORANGE,linewidths=1.2,zorder=4)
        b.scatter([sel],[s[sel-1]],s=34,color=BLUE if n_==0 else ORANGE,zorder=5,lw=0);clean(b);b.set_xlim(0,129);b.set_xticks([1,16,32,48,64,80,96,112,128]);b.tick_params(labelsize=7)
        b.set_title(f"candidate scores · selected B#{sel:03d}",fontsize=7.8,weight="normal");b.set_xlabel("B candidate j",fontsize=7.8)
    fig.legend(handles=[Line2D([0],[0],marker="o",lw=0,markerfacecolor="none",markeredgecolor=INK,markersize=8,label=f"original target B#{i:03d}"),
                        Line2D([0],[0],marker="o",lw=0,markerfacecolor="none",markeredgecolor=ORANGE,markersize=8,label=f"counterfactual target B#{j:03d}"),
                        Line2D([0],[0],marker="o",lw=0,color=BLUE,markersize=5,label="selected (original A)"),Line2D([0],[0],marker="o",lw=0,color=ORANGE,markersize=5,label="selected (counterfactual A)")],
               loc="center",bbox_to_anchor=(.545,.875),ncol=4,fontsize=7.8)
    tx=.79;fig.text(tx,.78,"Summary",fontsize=10,weight="bold",color=INK)
    tab=[("",f"original",f"counterf."),("selected",f"B#{r['selected_candidate']:03d}",f"B#{cf['selected_candidate']:03d}"),(f"rank of B#{i:03d}",f"{r['rank']}",f"{L['original_rank_under_counterfactual_A']}"),
         (f"rank of B#{j:03d}",f"{L['cf_target_rank_under_original_A']}",f"{cf['rank']}"),("top-1 score",f"{r['top1_score']:.4f}",f"{cf['top1_score']:.4f}"),
         ("margin",f"{r['top1_top2_margin']:+.4f}",f"{cf['top1_top2_margin']:+.4f}"),("span mass*",f"{r['pointer_mass']:.3f}",f"{cf['pointer_mass']:.3f}"),
         ("model forwards",f"{len(L['forwards']['retrieve'])}",f"{len(L['forwards']['counterfactual'])}")]
    for n_,(a_,b_,c_)in enumerate(tab):fig.text(tx,.745-n_*.034,mt(f"{a_:<15}{b_:>9}{c_:>11}"),fontsize=8.2,family=MONO,color=MUTED if n_==0 else INK)
    fol=cf["followed_counterfactual"]
    fig.text(tx,.44,"followed counterfactual:",fontsize=9,color=INK);fig.text(tx,.405,"YES"if fol else"NO",fontsize=15,weight="bold",color=ORANGE if fol else INK2)
    fit_text(fig,tx,.17,.18,.2,"* diagnostic only.\nCounterfactual forwards: 1 forge of the new A memory (no cache) + 1 query forward over it. The counterfactual A is built by the engine: the queried entity's seal is replaced by the target seal of candidate "
             f"{j:03d}; the two other statements keep their seals and the statement order is reshuffled with a fixed engine seed.",fs_max=7.6,fs_min=6.4,color=INK2)
    footer(fig,ctx,k,f"Figure {k}. Counterfactual test for item {i:03d}. Upper row: original A memory; lower row: A memory re-forged with the association {L['entity']} → {cf['counterfactual_seal']} (the target seal of B#{j:03d}). "
           f"With weights, question, pointer coordinate, address coordinate and B field unchanged, the selection moves from B#{r['selected_candidate']:03d} to B#{cf['selected_candidate']:03d}; the original candidate falls to rank {L['original_rank_under_counterfactual_A']}. "
           "This is a single live observation; the archived TEST528 counterfactual result is shown in Fig. 5.")
    return finish(fig,path,ctx,[f"B#{cf['selected_candidate']:03d}","followed counterfactual:",cf["counterfactual_seal"]],"Counterfactual association")
# ------------------------------------------------------------------ FIG 5 · controls + archived reference
CTRL_LAB={"PRIMARY":"PRIMARY · ÇAĞRIİZ pointer w","UNIFORM":"UNIFORM · w_t = 1/(T−1), t ≥ 1","SHIFTED":"SHIFTED · w rolled by ⌊(T−1)/2⌋ over t ≥ 1","NO_A":"NO-A · zero address (p = 0)"}
def f5(ctx,k,path):
    L=L_(ctx);c=L["controls"]["controls"];DC=L["controls_rederived"];ex=L["extras"];i=L["item"];fig=new_fig()
    header(fig,k,"Pointer controls on one item · archived TEST528 reference","Left/centre: live single-item controls from run_controls(). Right: sealed TEST528 values over 128 items, reproduced verbatim (not recomputed).","LIVE CONTROLS + SEALED REFERENCE",INK2)
    wmap={"PRIMARY":L["primary"]["pointer_weights"],"UNIFORM":ex["uniform"],"SHIFTED":ex["shifted"],"NO_A":None}
    for n_,mode in enumerate(("PRIMARY","UNIFORM","SHIFTED","NO_A")):
        y=.70-n_*.165;v=c[mode];d=DC[mode];col={"PRIMARY":BLUE,"UNIFORM":"#6B7280","SHIFTED":AMBER,"NO_A":"#9CA3AF"}[mode]
        a=fig.add_axes([.03,y,.24,.10]);clean(a,grid=False);a.tick_params(labelsize=6.8)
        if wmap[mode]is not None:wv=np.asarray(wmap[mode]);a.bar(np.arange(len(wv)),wv,color=col,width=.8);a.set_xlim(-.6,len(wv)-.4)
        else:a.set_xticks([]);a.set_yticks([]);a.text(.5,.5,"no pointer · address p = 0",transform=a.transAxes,ha="center",va="center",fontsize=8,color=INK2)
        a.set_title(mt(CTRL_LAB[mode]),fontsize=7.9)
        b=fig.add_axes([.31,y,.35,.10]);s=np.asarray(v["candidate_scores"],dtype=float);b.scatter(np.arange(1,129),s,s=5,color=GRAYPT,lw=0)
        b.scatter([i],[s[i-1]],s=55,facecolors="none",edgecolors=INK,linewidths=1.1,zorder=4);b.scatter([v["selected_candidate"]],[s[v["selected_candidate"]-1]],s=22,color=col,zorder=5,lw=0)
        clean(b);b.set_xlim(0,129);b.tick_params(labelsize=6.8);b.set_xticks([1,32,64,96,128])
        if d["all_equal"]:b.set_ylim(-.05,.05)
        b.set_title(mt(f"selected B#{v['selected_candidate']:03d} · rank of expected {v['rank']}/{N_CAND} · margin {v['margin']:+.4f}"),fontsize=7.9,weight="normal")
    tx=.695;fig.text(tx,.82,"Live, this item",fontsize=9.8,weight="bold",color=INK)
    fig.text(tx,.79,mt(f"{'control':<9}{'selected':>9}{'rank':>7}{'top-1':>9}{'margin':>9}"),fontsize=7.9,family=MONO,color=MUTED)
    for n_,mode in enumerate(("PRIMARY","UNIFORM","SHIFTED","NO_A")):
        v=c[mode];fig.text(tx,.763-n_*.026,mt(f"{mode:<9}{'B#'+format(v['selected_candidate'],'03d'):>9}{v['rank']:>7}{v['top1_score']:>9.4f}{v['margin']:>+9.4f}"),fontsize=7.9,family=MONO,color=INK)
    if DC["NO_A"]["all_equal"]:fit_text(fig,tx,.585,.275,.06,"NO-A: all 128 scores are 0 (zero address), so the argmax falls to B#001 by index tie-breaking; NO-A can therefore be 'correct' only for item 001, as an artefact.",fs_max=7.5,fs_min=6.4,color=INK2)
    frame(fig,tx,.19,.275,.37,ec=TEAL,fc=TEAL_L,lw=1.0)
    fig.text(tx+.01,.535,"SEALED TEST528 REFERENCE",fontsize=9.6,weight="bold",color=TEAL);fig.text(tx+.01,.512,"archived · 128 items · not recomputed here",fontsize=7.6,color=INK2)
    for n_,(lab,key)in enumerate((("PRIMARY","primary"),("COUNTERFACTUAL","counterfactual"),("SHIFTED","shifted"),("NO-A","no_a"))):
        v=TEST528[key];y=.47-n_*.052;fig.text(tx+.01,y,lab,fontsize=8.6,color=INK,weight="bold");fig.text(tx+.265,y,f"{fr(v)} = {v[2]}",fontsize=8.6,family=MONO,color=INK,ha="right")
    fig.text(tx+.01,.262,"UNIFORM: not part of the archived reference set",fontsize=7.4,color=INK2)
    fig.text(tx+.01,.236,mt(f"lock {TEST528['lock'][:24]}…"),fontsize=7,family=MONO,color=INK2);fig.text(tx+.01,.214,mt(f"result {TEST528['result_sha'][:24]}…"),fontsize=7,family=MONO,color=INK2)
    footer(fig,ctx,k,f"Figure {k}. Controls for item {i:03d} computed live by run_controls() with the same question and A memory: the ÇAĞRIİZ pointer (PRIMARY), a uniform pointer, the pointer rolled by half the slot range (SHIFTED), and a zero address (NO-A). "
           "Rings mark the expected candidate, filled points the selected one. The panel on the right reproduces the sealed TEST528 aggregate results verbatim; a single interactive item does not reproduce or re-estimate those statistics.")
    return finish(fig,path,ctx,["SEALED TEST528 REFERENCE",f"{fr(TEST528['primary'])} = {TEST528['primary'][2]}",f"{fr(TEST528['no_a'])} = {TEST528['no_a'][2]}","PRIMARY","NO_A"],"Controls and archived reference")
# ------------------------------------------------------------------ FIG 6 · provenance and verification record
def f6(ctx,k,path):
    P=ctx["P"];L=P["live"];cfg=P["engine"]["config"];E_=P["environment"];fz=P["frozen"];v=P["verify_after"];fig=new_fig()
    header(fig,k,"Provenance and verification record","Configuration, measured integrity quantities, hashes and the claim boundary of this run.","PROVENANCE RECORD",INK2)
    cols=[(.03,"CONFIGURATION AND ENGINE",
           f"run           {P['run_id']}\nstart (UTC)   {P['run_start_utc']}\nmodel         {cfg['MODEL_ID'].split('/')[-1]}\nweights       frozen · {P['model']['dtype']} · {P['model']['attn']}\n"
           f"architecture  32 L · 4096 · 32 Q / 8 KV · 128\npointer       L{cfg['PTR_L']:02d}H{cfg['PTR_H']:02d}\naddress       L{cfg['ADDR_L']:02d}-{cfg['ADDR_KIND']} · uncentered · {cfg['KVD']}-D\n"
           f"candidates    {cfg['N']} precomputed B matrices\nmatch         max cosine · argmax\nengine seed   {cfg['seed']}\nGPU           {E_['gpu']}\ntorch         {E_['torch']}\ntransformers  {E_['transformers']}\n"
           f"gradio        {E_['gradio']}\npython        {E_['python']}\nengine init   {(f"{P['engine']['init_seconds']:.1f} s"if isinstance(P['engine']['init_seconds'],(int,float))else"reused")}\nend (UTC)     {P['run_end_utc']}\n\nengine file\n  {P['engine']['file']}\nengine SHA-256\n  {P['engine']['sha256'][:32]}\n  {P['engine']['sha256'][32:]}"),
          (.35,"VERIFICATION MEASURED IN THIS RUN",
           f"weight sentinel  startup = before = after\n  {fz['sentinel_after'][:40]}…\nall-parameter guard  startup = before = after\n  {fz['guard_after'][:40]}…\n"
           f"trainable tensors   {fz['trainable_tensors']}\nLoRA / optimizer    {fz['lora']} / {fz['optimizer']}\nforeign hooks       {fz['foreign_hooks_before']} → {fz['foreign_hooks_after']} (outside API calls)\n\n"
           f"model forwards per API call (measured):\n  retrieve()        {len(L['forwards']['retrieve'])}  (n={L['question_tokens']}, A cache {L['a_slots']})\n  retrieve() again  {len(L['forwards']['repeat'])}\n"
           f"  get_pointer_data  {len(L['forwards']['pointer_data'])}\n  run_controls()    {len(L['forwards']['controls'])}\n  counterfactual()  {len(L['forwards']['counterfactual'])}  (forge + query)\n"
           f"candidate-B forwards: none observed\n\nverify_engine(): sentinel pass {v['weight_sentinel_pass']} ·\n  candidates {v['candidate_count']} · dim {v['address_dim']}\nre-derivation from raw scores: all match\n"
           f"pre-seal checks passed: {len(P['checks_pre_seal'])}\n\nlatency (engine API, ms):\n  retrieve {1000*P['timing'].get('retrieve',0):.1f} · controls {1000*P['timing'].get('controls',0):.1f}\n  counterfactual {1000*P['timing'].get('counterfactual',0):.1f} · guards {P['timing'].get('guard',0):.1f} s"),
          (.67,"ARCHIVED REFERENCES AND CLAIM BOUNDARY",
           f"TEST528 (sealed, archived)\n  lock   {TEST528['lock'][:32]}…\n  result {TEST528['result_sha'][:32]}…\n  primary         {fr(TEST528['primary'])} = {TEST528['primary'][2]}\n  counterfactual  {fr(TEST528['counterfactual'])} = {TEST528['counterfactual'][2]}\n"
           f"  shifted         {fr(TEST528['shifted'])} = {TEST528['shifted'][2]}\n  no-A            {fr(TEST528['no_a'])} = {TEST528['no_a'][2]}\n\ncross-model context\n(archived; different panels)\n"
           +"\n".join(f"  {c_['record']} · {c_['model']}\n    {c_['pointer']} · {c_['address'][:5]} · {c_['dim']}-D · {fr(c_['top1'])} = {c_['top1'][2]}"for c_ in CROSS_MODEL)+
           f"\n  {CROSS_MODEL_NOTE}\n\nnot established:\n"+"\n".join("  • "+t for t in("arbitrary token identities (native uppercase single-token seals)","first-A retrieval from an empty bank","answer generation from the selected B",
                                                                                         "transfer of internal coordinates between models","banks larger than 128 candidates")))]
    for x,ttl,body in cols:
        frame(fig,x,.155,.30,.715,ec=RULE,fc="white",lw=.8);fig.text(x+.01,.85,ttl,fontsize=9,weight="bold",color=INK2,va="center")
        fit_text(fig,x+.01,.165,.282,.665,body,fs_max=9.4,fs_min=6.2,family=MONO,color=INK)
    footer(fig,ctx,k,f"Figure {k}. Provenance record of run {P['run_id']} (verdict {P['verdict']}). Payload SHA-256 {ctx['sha']}. "
           "Hashes are artifact-integrity seals, not scientific proof or third-party verification. All live values are re-derived from the raw vectors stored in the sealed payload; archived values are reproduced verbatim from the TEST528 record.")
    return finish(fig,path,ctx,[P["run_id"],ENGINE_SHA256[:40],"none observed",fz["guard_after"][:40]],"Provenance and verification record")
FIGURES=[("fig_01_retrieval_path","Fig. 1 · Retrieval path (values from this run)",f1),("fig_02_candidate_competition","Fig. 2 · 128-way candidate competition",f2),
         ("fig_03_pointer_and_address","Fig. 3 · ÇAĞRIİZ pointer and 1024-D address",f3),("fig_04_counterfactual","Fig. 4 · Counterfactual association",f4),
         ("fig_05_controls_and_reference","Fig. 5 · Controls and sealed TEST528 reference",f5),("fig_06_provenance","Fig. 6 · Provenance and verification record",f6)]
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
MAM_MISTRAL_PART2_OK=True
