FEATURE_RECORD=SHOWCASE_IDX[2]+1   # fixed before the run: the middle showcase question (record #064) is the worked example on posters 03/04/06/07/08
ADDR_DIM=ARCH[3]*ARCH[4]
WIPE_LABELS={"SOURCES":"source records (256 texts)","SEALS":"seal list","CLASSES":"routing-class list","NAMES":"instrument names","POOL":"code pool","rr":"shuffle state"}
def _execute_run(E,ctl,state):
    """Generator. Yields progress events, returns the result bundle. Any failed technical check raises AuditFail (no package is sealed)."""
    run_id=ctl["run_id"];run_dir=Path(ctl["run_dir"]);I=E.info;checks=[];tm=Counter();T0=time.perf_counter();state["checks"]=checks;c0=Counter(E.counters)
    def chk(name,cond):
        checks.append(name)
        if not cond:raise AuditFail(name)
    run_start_utc=utc_now();run_start_local=local_now();say("="*140);say(f"RUN {run_id}");say("="*140)
    # ---- 1/10 integrity ----
    say("[1/10] INTEGRITY · TEST524 LOCK · FROZEN-MODEL PRE-CHECK");yield ev(1,"Integrity check","TEST524 lock, model, architecture, frozen weights and the all-parameter weight guard.")
    fs0=E.frozen_state();same_hw=I["gpu"]==CANON_GPU
    chk("TEST524 LOCK SHA recomputed = sealed value",LOCK_SHA==SEALED524_LOCK);chk("model = Qwen/Qwen2.5-7B-Instruct",I["model_id"]==MODEL_ID);chk("architecture 28L / H3584 / 28Q / 4KV / HD128",tuple(I["arch"])==ARCH)
    chk("dtype bfloat16",I["dtype"]=="bfloat16");chk("attention SDPA",I["attn"]=="sdpa");chk("ÇAĞRIİZ L23H12 · address L02-V · match max cosine",(PTR_L,PTR_H,BL,BK)==(23,12,2,"V"))
    chk("model in eval mode",not fs0["training"]);chk("trainable parameter tensors = 0",fs0["trainable_tensors"]==0 and fs0["requires_grad_disabled"]);chk("no LoRA / PEFT adapter",not fs0["lora"]);chk("no optimizer object",not fs0["optimizer"])
    chk("pre-run all-parameter guard = startup guard",fs0["guard"]==I["guard0"])
    if fs0["foreign_hooks"]:raise AuditFail("foreign forward/backward hooks present before run: "+str(fs0["foreign_modules"]))
    chk("no foreign forward/backward hooks before run",fs0["foreign_hooks"]==0)
    for c in checks:say("   PASS ·",c)
    say(f"   hardware: {I['gpu']} | canonical sealed hardware: {CANON_GPU} | same: {same_hw}"+("" if same_hw else"  (different GPU: scores may differ slightly from the sealed log; any difference is reported as measured)"))
    # ---- 2/10 test bank ----
    say("[2/10] TEST BANK · 128 RECORDS (TEST524 panel, rebuilt deterministically)");yield ev(2,"Building the test bank","128 records, rebuilt exactly as in TEST524. 128 is the experimental bank size, not a capacity limit.")
    E.reset_peak();pan=E.prepare_panel();tm["panel"]=pan["seconds"];audit=pan["audit"];showcase=pan["showcase"]
    chk("test bank = 128 records",pan["records"]==N and len(audit)==N)
    chk("showcase = records 7/38/64/91/123 with the TEST524 question template",[s["record"]for s in showcase]==[i+1 for i in SHOWCASE_IDX]and all(s["question"]==qA(s["entity"])for s in showcase))
    chk("showcase instrument names identical to the sealed TEST524 log",[s["entity"]for s in showcase]==[q["entity"]for q in SEALED524["queries"]])
    strip=[x[k_]for x in audit for k_ in("entity","seal","class")];state["strip"]=strip
    say(f"   {pan['records']} records · native single-token code pool {pan['pool']} · {pan['seconds']:.1f}s")
    # ---- 3/10 forge ----
    say("[3/10] READ ONCE · 128 RECORDS → 256 INDEPENDENT NUMERIC A/B MEMORIES");tels=[];t0=time.perf_counter()
    for i in range(N):
        tel=E.forge_one(i);tels.append(tel)
        if i in SHOWCASE_IDX:say(f"   SHOWCASE RECORD #{i+1:03d} forged · A slots {tel['TA']} · B slots {tel['TB']}")
        elif(i+1)%32==0:say(f"   {i+1:03d}/{N} records forged")
        if i%4==3 or i==N-1:yield ev(3,f"Read once · record {i+1:03d}/{N}","Each record is read once by the frozen AI and kept as two independent numeric memories (A and B). No training occurs.",done=i+1,total=N)
    tm["forge"]=time.perf_counter()-t0;cnt=E.counts()
    chk("256 independent numeric memories (128 A + 128 B)",cnt["A"]==N and cnt["B"]==N);chk("all memory tensors finite",all(t["finite"]for t in tels))
    fs_m=E.frozen_state()
    if fs_m["foreign_hooks"]:raise AuditFail("foreign forward/backward hooks present after forging: "+str(fs_m["foreign_modules"]))
    chk("all-parameter guard unchanged after reading the 128 records (no training)",fs_m["guard"]==I["guard0"]);chk("no foreign hooks and no trainable tensors after forging",fs_m["foreign_hooks"]==0 and fs_m["trainable_tensors"]==0)
    # ---- 4/10 address field ----
    say("[4/10] NUMERIC ADDRESS FIELD · L02-V STRUCTURES FOR ALL 128 B MEMORIES");yield ev(4,"Building the numeric address field","Every B memory gets a numeric address structure taken from its own layer-2 V tensors. No labels, no text.",done=N,total=N)
    af=E.build_address_field();tm["address"]=af["seconds"]
    chk(f"128 B address structures from L02-V ({ADDR_DIM} numbers per row)",af["b_mats"]==N and af["dim"]==ADDR_DIM);chk("128 distinct address structures",af["storages"]==N);chk("128 active-A packets installed",af["a_packs"]==N)
    say(f"   {af['b_mats']} address structures · rows {min(af['rows'])}–{max(af['rows'])} × {af['dim']} · {af['seconds']:.1f}s")
    # ---- 5/10 source removed ----
    say("[5/10] SIL BASTAN · SOURCE REMOVED FROM THE LIVE RETRIEVAL PATH");yield ev(5,"Source removed from the live retrieval path","Source records and panel containers are deleted. From here the retriever receives only a question and numeric memories.")
    t0=time.perf_counter();wp=E.wipe();tm["wipe"]=time.perf_counter()-t0
    chk("source containers deleted before the first question",not wp["source_text_present"]and wp["source_records_before"]==N)
    codes=set(x["seal"]for x in audit)|set(x["class"]for x in audit)
    def leak(q):return any(re.search(r"(?<![A-Za-z])"+re.escape(c)+r"(?![A-Za-z])",q)for c in codes)or bool(re.search(r"\d",q))
    chk("questions contain no seal, no class and no memory number",not any(leak(s["question"])for s in showcase))
    for k_,v in wp["present"].items():say(f"   {k_:<8}: {'PRESENT'if v else'DELETED'}")
    # ---- 6/10 live retrieval ----
    say("[6/10] LIVE RETRIEVAL · 5 FIXED QUESTIONS · EVERY QUESTION SCORES ALL 128 B MEMORIES");raw=dict(queries=[]);state["raw"]=raw;t0=time.perf_counter()
    for k_,s in enumerate(showcase,1):
        yield ev(6,f"Live question {k_}/5",s["question"]+" — ÇAĞRIİZ → numeric fingerprint → 128-memory field.",done=k_-1,total=5)
        r=E.retrieve_live(k_,s["record"],s["question"]);r["entity"]=s["entity"];raw["queries"].append(r)
        sel=r["order"][0];ok_=sel==s["record"]
        say(f" Q{k_} active A#{s['record']:03d} · selected B#{sel:03d} · {'PASS'if ok_ else'MISS'} · top {max(r['scores']):+.6f} · {1000*r['secs']:.1f} ms")
    tm["retrieval"]=time.perf_counter()-t0;ex=E.tensor_excerpt(FEATURE_RECORD)
    # ---- 7/10 re-derivation ----
    say("[7/10] RESULT RE-DERIVATION FROM THE RAW SCORES");yield ev(7,"Re-deriving the results","Every selection and rank is recomputed from the 128 raw scores of each question.",done=5,total=5)
    R=derive(raw);chk("results re-derived from raw scores are stable",derive(raw)==R);chk("every question scored all 128 B memories",all(len(r["scores"])==N for r in raw["queries"]))
    chk("candidate-B LLM forward counter = 0",all(r["b_forwards_counter"]==0 for r in raw["queries"]))
    for q in R["queries"]:say(f"   Q{q['k']} → selected #{q['selected']:03d} | gold rank {q['rank']:3d}/{N} | {'PASS'if q['correct']else'MISS'}")
    say(f"   LIVE SHOWCASE {R['correct']}/{R['total']} · replay of sealed TEST524 selections/ranks identical: {R['replay']['match']}")
    # ---- 8/10 integrity after ----
    say("[8/10] MODEL INTEGRITY AFTER THE RUN");yield ev(8,"Model integrity","The all-parameter weight guard is recomputed after the run.")
    fs1=E.frozen_state();tm["engine"]=time.perf_counter()-T0
    chk("all-parameter guard after run = startup guard (weights unchanged)",fs1["guard"]==I["guard0"]);chk("model eval mode and trainable tensors = 0 after run",(not fs1["training"])and fs1["trainable_tensors"]==0 and fs1["requires_grad_disabled"])
    chk("no LoRA / optimizer after run",(not fs1["lora"])and(not fs1["optimizer"]))
    if fs1["foreign_hooks"]:raise AuditFail("foreign forward/backward hooks present after run: "+str(fs1["foreign_modules"]))
    chk("no foreign forward/backward hooks after run (demo installs none)",fs1["foreign_hooks"]==0)
    verdict=f"TEST524_LIVE_SHOWCASE_{R['correct']}_OF_{R['total']}_INTEGRITY_VERIFIED";say("VERDICT:",verdict)
    # ---- 9/10 seal ----
    say("[9/10] SEAL");yield ev(9,"Sealing the evidence","Payload, manifest, readable log and raw scores are written and hashed.")
    E.sync();peak=E.peak_gib();dc=E.counters-c0
    protocol=[
     dict(key="model weights",value="FROZEN",cls="CHECKED",note="all-parameter guard identical at startup, pre-run, after reading and after the run"),
     dict(key="training / optimizer / LoRA",value="NONE",cls="CHECKED",note="0 trainable tensors, no optimizer object, no adapter"),
     dict(key="source text in the live retrieval path",value="ABSENT",cls="CHECKED",note="source containers deleted before the first question"),
     dict(key="memory number, seal or class in the question",value="NO",cls="CHECKED",note="each question scanned against all 768 panel codes and for digits"),
     dict(key="gold B memory ID supplied to the retriever",value="NO",cls="BY CONSTRUCTION",note="retrieve() receives only the active A index and the question text"),
     dict(key="active A memory",value="GIVEN",cls="BY CONSTRUCTION",note="A is the active memory; first-A retrieval from an empty bank is not part of this experiment"),
     dict(key="decoded-text / token-ID / LM-head router",value="NONE",cls="BY CONSTRUCTION",note="B scoring uses only L02-V numeric matrices"),
     dict(key="learned router",value="NONE",cls="BY CONSTRUCTION",note="no fitted parameters anywhere in the retrieval path"),
     dict(key="candidate-B LLM forwards",value="0",cls="BY CONSTRUCTION",note="B_scores() makes no model call; engine counter = 0; not hook-instrumented"),
     dict(key="matching",value="MAX COSINE · 128 B MEMORIES",cls="CHECKED",note="128 raw scores recorded for every question"),
     dict(key="query cache",value="FRESH PER QUESTION",cls="BY CONSTRUCTION",note="query_forward() builds a new DynamicCache for every question"),
     dict(key="pointer / address / match",value="L23H12 · L02-V · MAX COS",cls="CHECKED",note="L02-V uncentered, max cosine; frozen by the TEST524 lock SHA"),
     dict(key="post-hoc selection",value="NONE",cls="CHECKED",note="showcase records fixed inside the TEST524 lock SHA")]
    show_aud=[dict(record=s["record"],entity=s["entity"],seal=audit[s["record"]-1]["seal"],**{"class":audit[s["record"]-1]["class"]},question=s["question"])for s in showcase]
    P={"schema":"akbascore.mam.test524.worldlaunch.run.v1","project":f"{PRODUCT} · {PRODUCT_LONG}","author":AUTHOR,"place":AUTHOR_PLACE,"launch_date":LAUNCH_DATE,"copyright":COPYRIGHT,
       "run_id":run_id,"run_start_utc":run_start_utc,"run_start_local":run_start_local,"run_end_utc":utc_now(),"verdict":verdict,
       "model":{"id":MODEL_ID,"arch":list(ARCH),"dtype":I["dtype"],"attn":I["attn"],"params":I["params"],"pad_id":I["pad_id"],"vocab":I["vocab"]},
       "environment":{k_:I[k_]for k_ in("gpu","gpu_total_gib","torch","transformers","gradio","python","platform")},
       "hardware":{"canonical_sealed_gpu":CANON_GPU,"this_run_gpu":I["gpu"],"same":bool(same_hw)},
       "engine":{"source":"TEST524 engine functions, unchanged (direct child of sealed TEST523)","pointer":"ÇAĞRIİZ = frozen L23H12 native Q·K + RoPE over the active A memory",
                 "address":f"ÇAĞRIİZ-weighted L02-V, UNCENTERED ({ADDR_DIM} numbers)","match":"max cosine over every row of all 128 B address structures","fmt":FMT,
                 "lock_sha_recomputed":LOCK_SHA,"lock_sha_sealed":SEALED524_LOCK,"parent523":PARENT523,"showcase_records":[i+1 for i in SHOWCASE_IDX],"forbidden":LOCK["forbidden"],
                 "compute_path":"PyTorch CUDA backend; no custom kernel; no hooks"},
       "panel":{"records":N,"memories":2*N,"native_code_pool":pan["pool"],"audit_sha256":pan["audit_sha256"],"a_slots":[t["TA"]for t in tels],"b_slots":[t["TB"]for t in tels],
                "showcase_audit":show_aud,"feature_record":FEATURE_RECORD,"capacity_note":"128 is the experimental bank size, not a capacity limit."},
       "bank":{"numbers_A":sum(t["numbers_A"]for t in tels),"numbers_B":sum(t["numbers_B"]for t in tels),"address_rows":sum(af["rows"]),"address_numbers":sum(af["rows"])*af["dim"]},
       "address_field":{"rows":af["rows"],"dim":af["dim"],"structures":af["b_mats"]},
       "wipe":wp,"raw":raw,"results":R,"tensor_excerpt":ex,
       "frozen":{"guard_startup":I["guard0"],"guard_pre_run":fs0["guard"],"guard_after_forge":fs_m["guard"],"guard_after":fs1["guard"],"guard_sealed_reference":SEALED524["guard"],
                 "guard_method":I["guard_method"],"trainable_tensors":fs1["trainable_tensors"],"lora":fs1["lora"],"optimizer":fs1["optimizer"],"training_mode":fs1["training"],
                 "hooks_before":I["hooks0"],"hooks_after":fs1["hooks"],"foreign_hooks_before":fs0["foreign_hooks"],"foreign_hooks_after":fs1["foreign_hooks"],
                 "hook_note":"hook totals include hooks installed by transformers/accelerate; only hooks from other code count as foreign"},
       "protocol":protocol,
       "counters":{"forge_passes":dc["forge_passes"],"query_retrievals":dc["query_retrievals"],"candidate_B_llm_forwards":max([r["b_forwards_counter"]for r in raw["queries"]]+[0])},
       "timing":{"model_load_seconds":I["model_load_seconds"],"test_bank":tm["panel"],"read_once_forge":tm["forge"],"address_field":tm["address"],"source_removal":tm["wipe"],
                 "live_retrieval":tm["retrieval"],"engine_total":tm["engine"]},
       "gpu":{"peak_allocated_gib":peak},"checks_pre_seal":list(checks),
       "sealed_record":{"note":"Historical logs. NOT produced by this run.","test524":SEALED524,"test523":SEALED523},
       "reproduction":{"seed":SEED,"cell_source_sha256":I.get("cell_source_sha256"),"note":"SHA-256 of the executed cell text; the text itself is not embedded because it contains the stale-reference deny-list."},
       "scope":SCOPE}
    P=jsafe(P);P["stale_scan"]={"patterns":len(STALE)+1,"hits":0,"scope":"payload JSON text, panel strings (names, seals, classes) excluded"}
    allow=(I["gpu"],CANON_GPU)
    chk("stale-reference scan of the payload: 0 hits",not stale_scan("payload",strip_strings(json.dumps(P,ensure_ascii=False),strip),allow=allow))
    pb=canon(P);sha=hashlib.sha256(pb).hexdigest();payload_name=f"{run_id}_FULL_RUN_LOG_payload.json";pp=run_dir/payload_name;t_seal=time.perf_counter();pp.write_bytes(pb);say("payload sealed:",sha)
    chk("payload SHA-256 verification after write",hashlib.sha256(pp.read_bytes()).hexdigest()==sha)
    names=dict(payload=payload_name,manifest=f"run_manifest_{run_id}.json",txt=f"{run_id}_READABLE_RUN_LOG.txt",summary=f"{run_id}_SUMMARY.json",jsonl=f"{run_id}_RESULTS.jsonl",images=f"images_manifest_{run_id}.json",zip=f"AKBASCORE_MAM_WORLD_LAUNCH_ALL_{run_id}.zip")
    mp=run_dir/names["manifest"];sealed_utc=utc_now()
    manifest={"demo":f"{PRODUCT} · {PRODUCT_LONG} — WORLD LAUNCH DEMO (TEST524 engine)","author":AUTHOR,"copyright":COPYRIGHT,"run_id":run_id,"payload_file":payload_name,"payload_sha256":sha,"payload_bytes":len(pb),
              "sealed_utc":sealed_utc,"verdict":verdict,"canonicalization":"json.dumps(sort_keys=True, separators=(',',':'), ensure_ascii=False, allow_nan=False), UTF-8",
              "note":"Artifact integrity seal: confirms the payload is unchanged since sealing. It is not a scientific proof and not an independent third-party verification."}
    mp.write_text(json.dumps(manifest,indent=2,sort_keys=True,ensure_ascii=False),encoding="utf-8")
    summary={"engine":"TEST524","model":MODEL_ID,"test_bank_records":N,"numeric_memories":2*N,"live_showcase":[R["correct"],R["total"]],"live_showcase_ci95":list(cp_interval(R["correct"],R["total"])),
             "queries":[{k_:q[k_]for k_ in("k","record","entity","selected","rank","correct","top1","margin","peak","pos")}for q in R["queries"]],
             "replay_of_sealed_test524":R["replay"],"sealed_test523_top1":list(SEALED523["r1"]),"sealed_test523_top1_pct":SEALED523["r1_pct"],
             "verdict":verdict,"guard_before":I["guard0"],"guard_after":fs1["guard"],"hardware":P["hardware"],"lock_sha":LOCK_SHA,"run_id":run_id,"payload_sha256":sha}
    (run_dir/names["summary"]).write_text(json.dumps(jsafe(summary),indent=2,ensure_ascii=False),encoding="utf-8")
    lines=[{"phase":"LIVE_RETRIEVAL","question_number":r["k"],"active_A_memory":r["record"],"question":r["question"],"selected_B_memory":R["queries"][j]["selected"],"gold_rank":R["queries"][j]["rank"],
            "cagriiz_weights":r["w"],"numeric_fingerprint":r["p"],"all_128_scores":r["scores"]}for j,r in enumerate(raw["queries"])]
    (run_dir/names["jsonl"]).write_text("\n".join(json.dumps(jsafe(x),ensure_ascii=False)for x in lines)+"\n",encoding="utf-8")
    (run_dir/names["txt"]).write_text(make_txt(P,R,sha,names,sealed_utc),encoding="utf-8");seal_s=time.perf_counter()-t_seal
    # ---- 10/10 posters + ZIP ----
    say("[10/10] POSTERS · IMAGE MANIFEST · ZIP");yield ev(10,"Rendering posters","Every number on every poster is checked against the re-derived runtime result before the package is sealed.")
    ctx={"P":P,"R":R,"run_id":run_id,"sha":sha,"payload_name":payload_name,"seal_seconds":seal_s,"allow":allow,"strip":strip,"N":N_POSTERS,"poster_audit":[]}
    t_r=time.perf_counter();imgs=render_all(ctx,run_dir);render_s=time.perf_counter()-t_r;entries=[]
    for n_,(p,cap)in enumerate(imgs,1):
        with Image.open(p)as im:w_,h_=im.size;fm_=im.format;md=im.mode
        entries.append({"index":n_,"filename":p.name,"title":cap,"width":w_,"height":h_,"format":fm_,"mode":md,"size_bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
    imf=run_dir/names["images"]
    imf.write_text(json.dumps({"run_id":run_id,"payload_sha256":sha,"verdict":verdict,"count":len(entries),"render_seconds":render_s,"images":entries,"poster_text_audit":ctx["poster_audit"],
                               "note":"Per-image SHA-256 is an artifact integrity seal. Poster text was checked against re-derived results and scanned for stale references before saving."},indent=2,ensure_ascii=False),encoding="utf-8")
    zp=run_dir/names["zip"];members=[p for p,_ in imgs]+[imf,mp,pp,run_dir/names["txt"],run_dir/names["summary"],run_dir/names["jsonl"]]
    for m in members:
        h=stale_scan("filename",m.name)+(stale_scan(m.name,strip_strings(m.read_text(encoding="utf-8"),strip),allow=allow)if m.suffix in(".json",".txt",".jsonl")else[])
        chk(f"stale-reference scan of {m.name}: 0 hits",not h)
    with zipfile.ZipFile(zp,"w",compression=zipfile.ZIP_DEFLATED)as z:
        for m in members:z.write(m,arcname=m.name)
    post=post_seal_audit(imgs,zp,[m.name for m in members],pp,sha,run_dir/names["txt"],mp,verdict)
    say(f"{run_id}: {len(checks)} pre-seal + {len(post)} post-seal checks = {len(checks)+len(post)}/{len(checks)+len(post)} PASS | render {render_s:.1f}s")
    say("="*140);say("LIVE SHOWCASE (TEST524) :",f"{R['correct']}/{R['total']}");say("SEALED TEST523         :",f"{fr(SEALED523['r1'])} = {SEALED523['r1_pct']} top-1 (historical record)")
    say("VERDICT                :",verdict);say("PACKAGE                : SEALED ·",f"{len(checks)+len(post)} checks","· payload SHA-256",sha);say("ZIP                    :",zp);say("="*140)
    return dict(run_id=run_id,run_dir=run_dir,P=P,R=R,sha=sha,imgs=imgs,zip=zp,payload=pp,txt=run_dir/names["txt"],manifest=mp,images_manifest=imf,summary=run_dir/names["summary"],jsonl=run_dir/names["jsonl"],
                checks=len(checks)+len(post),verdict=verdict)
#<<CORE_END>>
#<<POSTERS_BEGIN>>
from matplotlib.text import Text
from matplotlib.patches import Arc,Polygon
plt.rcParams["font.family"]="DejaVu Sans";DPI=120
C_OK,C_ERR,C_CTL,C_CART,C_MOD,C_FG,C_NEU,C_BG,C_SEAL,C_VIO="#047857","#B91C1C","#475569","#B45309","#0369A1","#0F172A","#334155","#F8FAFC","#0F766E","#6D28D9"
C_OKL,C_ERRL,C_CARTL,C_MODL,C_SEALL,C_VIOL="#D1FAE5","#FEE2E2","#FEF3C7","#E0F2FE","#CCFBF1","#EDE9FE"
C_NIGHT,C_NIGHT2,C_CYAN,C_GLOW,C_AI="#0B1220","#16213A","#67E8F9","#FDE68A","#1E293B"
MONO="DejaVu Sans Mono";BRAND=f"{PRODUCT} · {PRODUCT_LONG.upper()}"
def mt(s):return str(s).replace("$",r"\$")
def new_fig():return plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
def save_jpg(fig,path):
    buf=io.BytesIO();fig.savefig(buf,format="png",dpi=fig.dpi,facecolor="white");plt.close(fig);buf.seek(0)
    with Image.open(buf)as im:
        im=im.convert("RGBA");bg=Image.new("RGB",im.size,(255,255,255));bg.paste(im,mask=im.getchannel("A"))
    bg.save(path,"JPEG",quality=92,optimize=True,progressive=False,subsampling=0)
    with Image.open(path)as chk_:
        if chk_.format!="JPEG"or chk_.mode!="RGB":raise RuntimeError("JPEG validation failed")
    return str(path)
def finish(fig,path,ctx,expect):
    """Poster text must contain every expected runtime value and no stale reference; otherwise the package is not sealed."""
    fig.canvas.draw();texts=[t.get_text()for t in fig.findobj(Text)if t.get_text().strip()];blob="\n".join(texts);nm=Path(path).name
    ws=lambda z:re.sub(r"\s+","",z);blob_ws=ws(blob);miss=[e for e in expect if ws(e)not in blob_ws]
    if miss:plt.close(fig);raise AuditFail(f"POSTER/RESULT MISMATCH in {nm}: missing {miss}")
    scan=strip_strings(re.sub(r"[ \t]*\n[ \t]*"," ",blob),ctx.get("strip",[]))
    hits=stale_scan(nm,scan,allow=ctx["allow"])+stale_scan("filename",nm)
    if hits:plt.close(fig);raise AuditFail(f"STALE REFERENCE in {nm}: {hits[:3]}")
    ctx["poster_audit"].append(dict(file=nm,texts=len(texts),expected_values=len(expect),stale_hits=0));return save_jpg(fig,path)
def fit_text(fig,x,y,w,h,text,fs_max=13,fs_min=7,color=C_FG,family=None,ls=1.32,weight="normal",ha="left"):
    fig.canvas.draw();r=fig.canvas.get_renderer();Wp=w*fig.get_figwidth()*fig.dpi;Hp=h*fig.get_figheight()*fig.dpi
    if Wp<=4 or Hp<=4:return 0
    paras=str(text if text else"(empty)").replace("\r","").split("\n");x0=x+w/2 if ha=="center"else x
    kw={"va":"top","ha":ha,"color":color,"linespacing":ls,"weight":weight,"multialignment":ha}
    if family:kw["family"]=family
    def wrap(c):
        out=[]
        for p in paras:out.extend(textwrap.wrap(p,c,break_long_words=True,break_on_hyphens=False)or[""])
        return out
    fs=float(fs_max);k=0.52
    for _ in range(150):
        cpl=max(6,int(Wp/(k*fs*fig.dpi/72.0)));ln=wrap(cpl);t=fig.text(x0,y+h,mt("\n".join(ln)),fontsize=fs,**kw);bb=t.get_window_extent(renderer=r)
        if bb.width>Wp*1.002 and cpl>6:t.remove();k*=max(1.01,bb.width/Wp*1.01);continue
        if bb.height<=Hp:return fs
        t.remove()
        if fs>fs_min:fs=max(float(fs_min),fs-0.5);continue
        per=bb.height/max(1,len(ln));keep=max(1,int(Hp/per)-1);fig.text(x0,y+h,mt("\n".join(ln[:keep]+["[… text shortened — full text in the run log]"])),fontsize=fs,**kw);return fs
    fig.text(x0,y+h,"[text omitted — see run log]",fontsize=fs_min,**kw);return fs_min
TAGC={"THIS":C_MOD,"SEALED":C_SEAL,"RESEARCHER":C_CTL,"RESEARCH":C_CART}
def head(fig,title,sub=None,tag=None):
    Hh=fig.get_figheight();f=lambda inch:1-inch/Hh;tc=next((v for k_,v in TAGC.items()if(tag or"").startswith(k_)),C_MOD)
    fig.text(.05,f(.42),BRAND,fontsize=12,weight="bold",color=C_CART,va="center")
    if tag:fig.text(.95,f(.42),tag,fontsize=12,weight="bold",color=tc,va="center",ha="right",bbox=dict(boxstyle="round,pad=0.35",fc="white",ec=tc,lw=1.6))
    fig.text(.05,f(.95),mt(title),fontsize=29,weight="bold",color=C_FG,va="center")
    if sub:fig.text(.05,f(1.42),mt(sub),fontsize=13.5,color=C_NEU,va="center")
    fig.add_artist(Line2D([.05,.95],[f(1.70),f(1.70)],transform=fig.transFigure,color=C_FG,lw=1.2));return f(1.85)
def foot(fig,ctx,k):
    fig.text(.5,.22/fig.get_figheight(),mt(f"{PRODUCT} · WORLD LAUNCH · © 2026 {AUTHOR} | RUN {ctx['run_id']} | PAYLOAD SHA-256 {ctx['sha'][:16]}… | {k:02d}/{ctx['N']:02d}"),ha="center",va="center",fontsize=9.5,color=C_NEU,family=MONO)
def clean_ax(ax):
    for s in("top","right"):ax.spines[s].set_visible(False)
def panel(fig,x,y,w,h,face,edge,lw=2.2,r=0.012,z=0):
    fig.add_artist(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0,rounding_size={r}",transform=fig.transFigure,facecolor=face,edgecolor=edge,lw=lw,zorder=z))
def box(fig,x,y,w,h,title,lines,color,face,tfs=13.5,bfs=11,tcol=None,bcol=C_FG):
    panel(fig,x,y,w,h,face,color,2.4)
    fig.text(x+w/2,y+h*.72,mt(title),ha="center",va="center",fontsize=tfs,weight="bold",color=tcol or color)
    fig.text(x+w/2,y+h*.32,mt(lines),ha="center",va="center",fontsize=bfs,color=bcol,linespacing=1.35)
def arrow(fig,x0,y0,x1,y1,color=C_FG,lw=2.3,ms=24):fig.add_artist(FancyArrowPatch((x0,y0),(x1,y1),transform=fig.transFigure,arrowstyle="-|>",mutation_scale=ms,lw=lw,color=color))
def chip(fig,x,y,w,h,label,value,color,face,vfs=19):
    panel(fig,x,y,w,h,face,color,2,0.008)
    fig.text(x+w/2,y+h*.62,mt(value),ha="center",va="center",fontsize=vfs,weight="bold",color=color);fig.text(x+w/2,y+h*.22,mt(label),ha="center",va="center",fontsize=10.5,color=C_FG)
def okc(b):return C_OK if b else C_ERR
def frozen_ai(fig,x,y,w,h,label="FROZEN AI",sub=MODEL_SHORT,note="weights locked",lines=28):
    """The frozen model drawn as a dark block of 28 layer bands with a padlock."""
    panel(fig,x,y,w,h,C_AI,C_FG,2.4,0.012,z=2)
    for j in range(lines):
        yy=y+h*.12+j*(h*.50/lines);fig.add_artist(Line2D([x+w*.12,x+w*.88],[yy,yy],transform=fig.transFigure,color="#334155",lw=1.1,zorder=3))
    ax=fig.add_axes([x+w*.38,y+h*.64,w*.24,h*.17]);ax.set_zorder(10);ax.patch.set_alpha(0);ax.set_xlim(0,1);ax.set_ylim(0,1.3);ax.set_aspect("equal",adjustable="box");ax.axis("off")
    ax.add_patch(Arc((.5,.62),.5,.6,theta1=0,theta2=180,color=C_GLOW,lw=3));ax.add_patch(FancyBboxPatch((.15,.05),.7,.58,boxstyle="round,pad=0,rounding_size=.08",fc=C_GLOW,ec=C_GLOW))
    fig.text(x+w/2,y+h*.92,label,ha="center",va="center",fontsize=14,weight="bold",color="white",zorder=4)
    fig.text(x+w/2,y+h*.06,mt(sub+(" · "+note if note else"")),ha="center",va="center",fontsize=8.6,color="#CBD5E1",zorder=4)
def doc_icon(fig,x,y,w,h,face="white",edge="#94A3B8",lines=3,alpha=1.0):
    fig.add_artist(Rectangle((x,y),w,h,transform=fig.transFigure,facecolor=face,edgecolor=edge,lw=.8,alpha=alpha))
    for j in range(lines):
        yy=y+h*(.75-j*.22);fig.add_artist(Line2D([x+w*.15,x+w*(.85 if j<lines-1 else .55)],[yy,yy],transform=fig.transFigure,color=edge,lw=.9,alpha=alpha))
def num_row(vals,n=5):return"["+", ".join(f"{v:+.3f}"for v in vals[:n])+", …]"
def silhouette(fig,x,y,w,h,color="#475569"):
    ax=fig.add_axes([x,y,w,h]);ax.set_zorder(10);ax.patch.set_alpha(0);ax.set_xlim(0,1);ax.set_ylim(0,2);ax.set_aspect("equal",adjustable="box");ax.axis("off")
    ax.add_patch(Circle((.5,1.55),.24,fc=color,ec=color));ax.add_patch(Polygon([[.08,0],[.92,0],[.84,.95],[.66,1.22],[.34,1.22],[.16,.95]],closed=True,fc=color,ec=color))
def feat(ctx):
    R=ctx["R"];P=ctx["P"];q=next(z for z in R["queries"]if z["record"]==FEATURE_RECORD);r=next(z for z in P["raw"]["queries"]if z["record"]==FEATURE_RECORD)
    a=next(z for z in P["panel"]["showcase_audit"]if z["record"]==FEATURE_RECORD);return q,r,a
# ------------------------------------------------------------------ 01 WHAT WE BUILT
def p01(ctx,k,path):
    P=ctx["P"];R=ctx["R"];fig=new_fig()
    top=head(fig,"WHAT WE BUILT","A frozen AI keeps model-native numeric memories — and a natural question finds the associated one.",tag="THIS LIVE RUN")
    fig.text(.5,top-.075,PRODUCT,fontsize=58,weight="bold",color=C_FG,ha="center",va="center")
    fig.text(.5,top-.155,PRODUCT_LONG.upper(),fontsize=21,weight="bold",color=C_CART,ha="center",va="center")
    steps=[("READ ONCE","test records enter\na frozen AI"),("LANGUAGE ENDS","memory becomes\nmodel-native numbers"),("SOURCE LEAVES","removed from the live\nretrieval path"),
           ("NEW QUESTION","creates a recall trace:\nÇAĞRIİZ"),("FINGERPRINT","a 512-number\nnumeric address"),("ONE MEMORY LOCKS","strongest match in\nthe memory field")]
    bw=.135;gap=(.9-6*bw)/5;bh=.165;y=top-.42
    panel(fig,.05+3*(bw+gap)-gap/2-.004,y-.035,.955-(.05+3*(bw+gap)-gap/2)+.004,bh+.07,C_NIGHT,C_NIGHT,0,0.014)
    fig.text(.05,y+bh+.02,"HUMAN WORLD",fontsize=10.5,weight="bold",color=C_NEU);fig.text(.94,y+bh+.02,"MACHINE-NATIVE MEMORY SPACE",fontsize=10.5,weight="bold",color=C_CYAN,ha="right")
    for i,(t,b)in enumerate(steps):
        x=.05+i*(bw+gap);dark=i>=3
        panel(fig,x,y,bw,bh,C_NIGHT2 if dark else C_BG,C_CYAN if dark else C_FG,2.0)
        fig.text(x+bw/2,y+bh*.80,str(i+1),ha="center",va="center",fontsize=15,weight="bold",color=C_GLOW if dark else C_CART)
        fig.text(x+bw/2,y+bh*.55,t,ha="center",va="center",fontsize=12.5,weight="bold",color="white"if dark else C_FG)
        fig.text(x+bw/2,y+bh*.22,b,ha="center",va="center",fontsize=9.8,color="#CBD5E1"if dark else C_NEU,linespacing=1.3)
        if i<5:arrow(fig,x+bw+.003,y+bh/2,x+bw+gap-.003,y+bh/2,color=C_CYAN if i>=2 else C_FG,lw=2,ms=18)
    yb=.085;hb=.215
    panel(fig,.05,yb,.425,hb,C_MODL,C_MOD,2.6);panel(fig,.525,yb,.425,hb,C_SEALL,C_SEAL,2.6)
    fig.text(.2625,yb+hb-.032,"WORLD LAUNCH LIVE SHOWCASE · TEST524",ha="center",fontsize=13,weight="bold",color=C_MOD,va="center")
    fig.text(.2625,yb+hb*.47,f"{R['correct']} / {R['total']}",ha="center",va="center",fontsize=50,weight="bold",color=C_MOD)
    fig.text(.2625,yb+.025,"five fixed questions · measured in this run",ha="center",fontsize=11,color=C_FG,va="center")
    fig.text(.7375,yb+hb-.032,"EXTERNAL REPLICATION · TEST523",ha="center",fontsize=13,weight="bold",color=C_SEAL,va="center")
    fig.text(.7375,yb+hb*.47,f"{SEALED523['r1'][0]} / {SEALED523['r1'][1]}",ha="center",va="center",fontsize=50,weight="bold",color=C_SEAL)
    fig.text(.7375,yb+.025,f"{SEALED523['r1_pct']} TOP-1 · sealed historical record",ha="center",fontsize=11,color=C_FG,va="center")
    fig.text(.5,yb+hb/2,"two\nseparate\nexperiments",ha="center",va="center",fontsize=9,color=C_NEU,style="italic",linespacing=1.2)
    fig.text(.5,.045,f"{AUTHOR} · {AUTHOR_PLACE} · {LAUNCH_DATE}",ha="center",va="center",fontsize=11.5,color=C_NEU,weight="bold")
    foot(fig,ctx,k);return finish(fig,path,ctx,[f"{R['correct']} / {R['total']}",f"{SEALED523['r1'][0]} / {SEALED523['r1'][1]}",SEALED523["r1_pct"],PRODUCT,"ÇAĞRIİZ"])
# ------------------------------------------------------------------ 02 READ ONCE
def p02(ctx,k,path):
    P=ctx["P"];fz=P["frozen"];B=P["bank"];fig=new_fig()
    top=head(fig,"STEP 1 — READ ONCE","For this experiment, 128 records are presented to the frozen AI. It reads each one once.",tag="THIS LIVE RUN")
    panel(fig,.05,top-.475,.27,.44,C_BG,C_FG,2.0);fig.text(.185,top-.065,"TEST BANK: 128 RECORDS",ha="center",fontsize=14,weight="bold",color=C_FG)
    for i in range(128):
        r_,c_=divmod(i,16);doc_icon(fig,.065+c_*.0152,top-.115-r_*.042-.033,.0125,.033,lines=3)
    fig.text(.185,top-.46,"human-readable records",ha="center",fontsize=10.5,color=C_NEU,style="italic")
    arrow(fig,.33,top-.255,.375,top-.255)
    frozen_ai(fig,.385,top-.45,.23,.39)
    fig.text(.5,top-.475,"The AI's learned weights are locked\nand are not changed by this experiment.",ha="center",va="top",fontsize=10.5,color=C_FG,linespacing=1.3)
    arrow(fig,.625,top-.255,.67,top-.255,color=C_CYAN)
    panel(fig,.68,top-.475,.27,.44,C_NIGHT,C_CYAN,2.0);fig.text(.815,top-.065,"256 NUMERIC MEMORIES",ha="center",fontsize=14,weight="bold",color="white")
    sl=P["panel"]["a_slots"]+P["panel"]["b_slots"];lo,hi=min(sl),max(sl)
    for i in range(256):
        r_,c_=divmod(i,16);v=(sl[i]-lo)/max(1,hi-lo);col=(0.25+0.4*v,0.75+0.2*v,0.95)
        fig.add_artist(Rectangle((.695+c_*.0152,top-.115-r_*.0198-.016),.0125,.016,transform=fig.transFigure,facecolor=col,edgecolor="none"))
    fig.text(.755,top-.445,"128 A",ha="center",fontsize=10,color=C_CYAN,weight="bold");fig.text(.875,top-.445,"128 B",ha="center",fontsize=10,color=C_CYAN,weight="bold")
    fig.text(.815,top-.46,"independently forged A/B numeric memories",ha="center",fontsize=9.2,color="#CBD5E1",style="italic",va="top")
    panel(fig,.05,.075,.9,.205,C_OKL,C_OK,2.4)
    fig.text(.5,.245,"NO TRAINING OCCURS",ha="center",va="center",fontsize=30,weight="bold",color=C_OK)
    fig.text(.5,.198,"The brain stays frozen. The external memory changes.",ha="center",va="center",fontsize=16,color=C_FG,weight="bold")
    same=fz["guard_pre_run"]==fz["guard_after"]
    fig.text(.08,.145,f"MODEL BEFORE  ·  weight guard {fz['guard_pre_run'][:20]}…",fontsize=10.5,family=MONO,color=C_FG,va="center")
    fig.text(.08,.115,f"MODEL AFTER   ·  weight guard {fz['guard_after'][:20]}…",fontsize=10.5,family=MONO,color=C_FG,va="center")
    fig.text(.92,.13,"✓ IDENTICAL"if same else"✗ CHANGED",fontsize=17,weight="bold",color=okc(same),ha="right",va="center")
    fig.text(.5,.09,f"The memory side holds {B['numbers_A']+B['numbers_B']:,} model-derived numbers. 128 is the size of this experiment, not the architectural memory limit.",ha="center",va="center",fontsize=10.2,color=C_NEU)
    foot(fig,ctx,k);return finish(fig,path,ctx,["NO TRAINING OCCURS","TEST BANK: 128 RECORDS",fz["guard_after"][:20],f"{B['numbers_A']+B['numbers_B']:,}"])
# ------------------------------------------------------------------ 03 SIGNATURE · HUMAN LANGUAGE ENDS HERE
def p03(ctx,k,path):
    P=ctx["P"];ex=P["tensor_excerpt"];q,r,a=feat(ctx);fig=new_fig();rng=random.Random(3)
    top=head(fig,"HUMAN LANGUAGE ENDS HERE","Human-readable source → frozen AI → machine-native memory.",tag="THIS LIVE RUN")
    yb=.265;hb=top-yb-.065
    panel(fig,.05,yb,.36,hb,"white",C_FG,2.0);panel(fig,.565,yb,.385,hb,C_NIGHT,C_NIGHT,2.0)
    for j in range(4):doc_icon(fig,.075+j*.05,yb+.03,.038,.075,lines=4)
    fig.text(.29,yb+.065,"documents ·\nsentences · words",fontsize=9.5,color=C_NEU,va="center",linespacing=1.25)
    fig.text(.07,yb+hb-.035,"HUMAN WORLD",fontsize=15,weight="bold",color=C_FG,va="center");fig.text(.07,yb+hb-.07,"readable text · words · sentences · questions",fontsize=10.5,color=C_NEU,va="center")
    fig.text(.93,yb+hb-.035,"MACHINE-NATIVE MEMORY SPACE",fontsize=15,weight="bold",color=C_CYAN,va="center",ha="right")
    fig.text(.93,yb+hb-.07,"K tensors · V tensors · high-dimensional numbers",fontsize=10.5,color="#CBD5E1",va="center",ha="right")
    sent=f"Instrument {a['entity']} carries seal {a['seal']}."
    fit_text(fig,.075,yb+hb*.30,.31,hb*.38,f"“{sent}”",fs_max=24,fs_min=14,weight="bold",color=C_FG)
    fig.text(.075,yb+hb*.30,f"one of the three sentences read into memory A#{FEATURE_RECORD:03d}",fontsize=9.6,color=C_NEU,style="italic",va="center")
    stream=[c for c in sent if c.strip()]
    for j in range(34):
        t_=j/33;x=.33+t_*.078;y=yb+hb*.52+rng.uniform(-1,1)*hb*(.03+.10*t_);fig.text(x,y,rng.choice(stream),fontsize=11-5*t_,color=C_FG,alpha=max(.05,.75-.7*t_),ha="center",va="center",weight="bold")
    frozen_ai(fig,.415,yb+hb*.18,.145,hb*.64,label="FROZEN AI",sub=MODEL_SHORT,note="")
    for j in range(30):
        t_=j/29;x=.572+t_*.038;y=yb+hb*.52+rng.uniform(-1,1)*hb*.10*(1-t_);fig.text(x,y,rng.choice("0123456789.+-"),fontsize=6+5*t_,color=C_CYAN,alpha=.15+.75*t_,ha="center",va="center",family=MONO)
    fig.add_artist(Line2D([.5625,.5625],[yb-.012,yb+hb+.012],transform=fig.transFigure,color=C_GLOW,lw=3.2,ls=(0,(6,4)),zorder=6))
    fig.text(.5625,yb+hb+.03,"LANGUAGE  →  MODEL-NATIVE NUMBERS",ha="center",va="center",fontsize=12,weight="bold",color=C_CART,zorder=7,bbox=dict(boxstyle="round,pad=0.3",fc="white",ec=C_CART,lw=1.6))
    rows=ex["a_v"]
    for j,row in enumerate(rows):
        t_=j/max(1,len(rows)-1);fig.text(.625,yb+hb*(.78-j*.075),num_row(row),fontsize=12.5-5*t_,color=C_CYAN,alpha=1-.75*t_,family=MONO,va="center")
    fig.text(.625,yb+hb*.15,"…  thousands more values per memory, into depth",fontsize=9.5,color="#94A3B8",family=MONO,va="center")
    fig.text(.625,yb+hb*.07,f"REAL VALUES FROM THIS RUN · memory A#{FEATURE_RECORD:03d} · layer 2 · V tensor",fontsize=9.5,color=C_GLOW,va="center",weight="bold")
    fig.text(.5,.205,"BEYOND THIS POINT, THE MEMORY IS NO LONGER STORED AS HUMAN-READABLE LANGUAGE.",ha="center",va="center",fontsize=17.5,weight="bold",color=C_FG)
    fig.text(.5,.15,"THE AI IS MATCHING ITS OWN NUMERIC REPRESENTATIONS.",ha="center",va="center",fontsize=20,weight="bold",color=C_MOD)
    fit_text(fig,.08,.045,.84,.065,"The live retrieval object is model-derived numeric tensor state rather than ordinary human-readable source text. It was created by processing language, so it carries information derived from it — stored and matched as numbers.",fs_max=10.8,fs_min=8.5,color=C_NEU,ha="center")
    foot(fig,ctx,k);return finish(fig,path,ctx,["HUMAN LANGUAGE ENDS HERE","MACHINE-NATIVE MEMORY SPACE",num_row(rows[0]),"NUMERIC REPRESENTATIONS"])
# ------------------------------------------------------------------ 04 INSIDE A BELLEKÖZ
def p04(ctx,k,path):
    P=ctx["P"];ex=P["tensor_excerpt"];fig=new_fig()
    top=head(fig,"THE RETRIEVAL MEMORY IS NOT A TEXT FILE.","It is model-derived numeric state: K/V tensors produced inside the frozen AI while it read the source. This is a BELLEKÖZ.",tag="THIS LIVE RUN")
    chain=[("SOURCE TEXT",C_FG,C_BG,C_FG),("TOKENS",C_FG,C_BG,C_FG),("FROZEN TRANSFORMER",C_AI,C_AI,"white"),("INTERNAL ACTIVATIONS",C_NIGHT2,C_NIGHT2,"white"),("K / V TENSORS",C_CYAN,C_NIGHT,C_CYAN),("BELLEKÖZ",C_GLOW,C_NIGHT,C_GLOW)]
    bh=.072;g=.024;y0=top-.02
    for i,(t,ec,fc,tc)in enumerate(chain):
        y=y0-(i+1)*bh-i*g;panel(fig,.05,y,.2,bh,fc,ec,2.0);fig.text(.15,y+bh/2,t,ha="center",va="center",fontsize=12,weight="bold",color=tc)
        if i<5:arrow(fig,.15,y-.002,.15,y-g+.002,color=C_NEU,lw=1.8,ms=14)
    yk=top-.03;kw_=.27
    panel(fig,.28,yk-.19,kw_,.19,C_MODL,C_MOD,2.2);fig.text(.295,yk-.03,"K — KEY",fontsize=15,weight="bold",color=C_MOD,va="center")
    fit_text(fig,.295,yk-.18,kw_-.03,.125,"A numerical structure the transformer's attention uses to decide what information is relevant. It is not a filename or a database key.",fs_max=12,fs_min=9)
    panel(fig,.28,yk-.41,kw_,.19,C_VIOL,C_VIO,2.2);fig.text(.295,yk-.25,"V — VALUE",fontsize=15,weight="bold",color=C_VIO,va="center")
    fit_text(fig,.295,yk-.40,kw_-.03,.125,"The numerical information that attention can retrieve and use once relevance has been decided.",fs_max=12,fs_min=9)
    fit_text(fig,.28,yk-.52,kw_,.09,"AKBASCORE MAM keeps the model-derived K/V tensor state as numeric memory. Nobody assigns it a human-readable address.",fs_max=11.5,fs_min=9,weight="bold",color=C_FG)
    H=np.array(ex["a_heat"],dtype=float);ax=fig.add_axes([.665,top-.50,.285,.42]);lim=float(np.percentile(np.abs(H),97))or 1.0
    ax.imshow(H,aspect="auto",cmap="RdBu_r",vmin=-lim,vmax=lim,interpolation="nearest");ax.set_xticks([]);ax.set_yticks([])
    for s in ax.spines.values():s.set_color(C_FG)
    fig.text(.665,top-.045,f"MEMORY A#{FEATURE_RECORD:03d} · layer 2 · V · head 0 · {H.shape[0]} slots × first 64 of 128 dims",fontsize=9.5,color=C_NEU,va="center")
    fig.text(.8075,top-.025,"REAL VALUES FROM THIS RUN",fontsize=11,weight="bold",color=C_MOD,va="center",ha="center")
    silhouette(fig,.585,top-.47,.07,.24)
    fig.text(.8075,top-.535,"A HUMAN CANNOT READ THIS AS A SENTENCE.",ha="center",va="center",fontsize=13.5,weight="bold",color=C_FG)
    fit_text(fig,.585,top-.62,.365,.06,"BELLEKÖZ is numerical model state. Software can inspect its values mathematically, but it is not stored as ordinary human-readable language.",fs_max=9.8,fs_min=8,color=C_NEU)
    panel(fig,.05,.065,.9,.115,C_NIGHT,C_NIGHT,0)
    fig.text(.5,.14,"NO DOCUMENT TO REREAD.     NO SENTENCE TO SEARCH.     ONLY MODEL-NATIVE NUMERIC MEMORY.",ha="center",va="center",fontsize=16.5,weight="bold",color=C_GLOW)
    fig.text(.5,.093,f"One memory = {ex['numbers_A']:,} numbers: 28 layers × K and V × 4 heads × 128 dimensions × each memory slot.",ha="center",va="center",fontsize=11,color="#CBD5E1")
    foot(fig,ctx,k);return finish(fig,path,ctx,["THE RETRIEVAL MEMORY IS NOT A TEXT FILE.","A HUMAN CANNOT READ THIS AS A SENTENCE.","K — KEY","V — VALUE",f"{ex['numbers_A']:,}"])
# ------------------------------------------------------------------ 05 SOURCE REMOVED
def p05(ctx,k,path):
    P=ctx["P"];wp=P["wipe"];fig=new_fig()
    top=head(fig,"SOURCE REMOVED FROM THE LIVE RETRIEVAL PATH","After memory creation, the retrieval mechanism no longer receives the original source text.",tag="THIS LIVE RUN")
    cw=.27;g=(.9-3*cw)/2;yb=.27;hb=top-yb-.02;xs=[.05+i*(cw+g)for i in range(3)]
    panel(fig,xs[0],yb,cw,hb,C_BG,C_FG,2.0);panel(fig,xs[1],yb,cw,hb,C_ERRL,C_ERR,2.0);panel(fig,xs[2],yb,cw,hb,C_NIGHT,C_CYAN,2.0)
    fig.text(xs[0]+cw/2,yb+hb-.035,"1 · READ TIME",ha="center",fontsize=15,weight="bold",color=C_FG,va="center")
    fig.text(xs[1]+cw/2,yb+hb-.035,"2 · REMOVED",ha="center",fontsize=15,weight="bold",color=C_ERR,va="center")
    fig.text(xs[2]+cw/2,yb+hb-.035,"3 · LIVE RETRIEVAL",ha="center",fontsize=15,weight="bold",color=C_CYAN,va="center")
    for j in range(5):doc_icon(fig,xs[0]+.035+j*.042,yb+hb*.52,.034,.085,lines=4)
    fit_text(fig,xs[0]+.02,yb+.03,cw-.04,hb*.36,f"{wp['source_records_before']} source records are read once by the frozen AI to create the numeric memories.",fs_max=13,fs_min=9,ha="center")
    for j,n_ in enumerate(wp["names"]):
        y=yb+hb-.10-j*.058;gone=not wp["present"][n_]
        fig.text(xs[1]+.02,y,WIPE_LABELS.get(n_,n_),fontsize=11.5,color=C_FG,va="center")
        fig.text(xs[1]+cw-.02,y,"✓ deleted"if gone else"✗ PRESENT",fontsize=11.5,weight="bold",color=okc(gone),va="center",ha="right")
    fit_text(fig,xs[1]+.02,yb+.025,cw-.04,.09,"deleted before the first question is asked",fs_max=11,fs_min=8.5,color=C_ERR,weight="bold",ha="center")
    keep=["numeric A memories (K/V tensors)","numeric B memories (K/V tensors)","numeric B address structures","the new question","a fresh cache for every question","conversation history: empty"]
    for j,t in enumerate(keep):fig.text(xs[2]+.02,yb+hb-.10-j*.058,"•  "+t,fontsize=11.5,color="white",va="center")
    fit_text(fig,xs[2]+.02,yb+.025,cw-.04,.09,"what the retriever can use",fs_max=11,fs_min=8.5,color=C_CYAN,weight="bold",ha="center")
    arrow(fig,xs[0]+cw+.004,yb+hb/2,xs[1]-.004,yb+hb/2,lw=2,ms=18);arrow(fig,xs[1]+cw+.004,yb+hb/2,xs[2]-.004,yb+hb/2,color=C_CYAN,lw=2,ms=18)
    ok=not wp["source_text_present"]
    fig.text(.5,.19,"SOURCE TEXT IN THE LIVE RETRIEVAL PATH:  "+("ABSENT"if ok else"PRESENT"),ha="center",va="center",fontsize=21,weight="bold",color=okc(ok))
    fit_text(fig,.08,.05,.84,.1,"Scope: the source is removed from the live retrieval path of this run; it does not mean every copy of the information anywhere has been erased. Audit metadata (record labels and expected seals) is kept apart and used only after retrieval, to verify what happened. Technical flag: source_text_container_destroyed = "+str(ok)+".",fs_max=10.5,fs_min=8.5,color=C_NEU,ha="center")
    foot(fig,ctx,k);return finish(fig,path,ctx,["SOURCE REMOVED FROM THE LIVE RETRIEVAL PATH","ABSENT"if ok else"PRESENT",str(wp["source_records_before"])])
# ------------------------------------------------------------------ 06 HOW A QUESTION FINDS A MEMORY
def p06(ctx,k,path):
    P=ctx["P"];q,r,a=feat(ctx);fig=new_fig();lab=f"#{FEATURE_RECORD:03d}"
    top=head(fig,"HOW A QUESTION FINDS AN ASSOCIATED MEMORY","One numeric memory is already active. A new natural question makes the frozen AI find its associated memory.",tag="THIS LIVE RUN")
    y=top-.25;h=.2
    panel(fig,.05,y+h*.52,.2,h*.48,C_NIGHT,C_CYAN,2.0);fig.text(.15,y+h*.85,"ACTIVE MEMORY A",ha="center",va="center",fontsize=12.5,weight="bold",color=C_CYAN);fig.text(.15,y+h*.64,"numeric K/V only",ha="center",va="center",fontsize=10,color="#CBD5E1")
    fig.text(.15,y+h*.42,"+",ha="center",va="center",fontsize=24,weight="bold",color=C_FG)
    panel(fig,.05,y-h*.18,.2,h*.48,"white",C_FG,2.0);fit_text(fig,.06,y-h*.16,.18,h*.40,"NEW NATURAL QUESTION\n“"+a["question"]+"”",fs_max=10.5,fs_min=8,ha="center")
    arrow(fig,.255,y+h*.3,.3,y+h*.3);frozen_ai(fig,.305,y-h*.18,.17,h*1.18,note="")
    arrow(fig,.48,y+h*.3,.525,y+h*.3,color=C_CYAN)
    panel(fig,.53,y-h*.18,.42,h*1.18,C_NIGHT,C_CYAN,2.0)
    fig.text(.74,y+h*.86,"FIND THE ASSOCIATED MEMORY B",ha="center",va="center",fontsize=14,weight="bold",color="white")
    fig.text(.74,y+h*.68,"among the B memories of the experimental field",ha="center",va="center",fontsize=10.5,color="#CBD5E1")
    for i in range(N):
        rr_,cc=divmod(i,32);fig.add_artist(Rectangle((.55+cc*.0118,y+h*.38-rr_*.022),.0098,.017,transform=fig.transFigure,facecolor=C_GLOW if i+1==q["selected"]else"#1F3A5F",edgecolor="none"))
    yc=.085;hc=(top-.25)-.2*.18-.06-yc
    panel(fig,.05,yc,.36,hc,C_BG,"#94A3B8",1.6);panel(fig,.43,yc,.52,hc,C_MODL,C_MOD,2.2)
    fig.text(.23,yc+hc-.032,"CLASSIC EXPLICIT LOOKUP",ha="center",va="center",fontsize=13,weight="bold",color=C_CTL)
    fig.text(.23,yc+hc*.58,f"memory_id = {FEATURE_RECORD:03d}",ha="center",va="center",fontsize=17,family=MONO,color=C_FG)
    fig.text(.23,yc+hc*.40,"↓",ha="center",va="center",fontsize=16,color=C_NEU)
    fig.text(.23,yc+hc*.24,f"retrieve record {FEATURE_RECORD:03d}",ha="center",va="center",fontsize=14,family=MONO,color=C_FG)
    fig.text(.69,yc+hc-.032,"AKBASCORE MAM",ha="center",va="center",fontsize=13,weight="bold",color=C_MOD)
    chain="natural question  →  frozen-model recall trace  →  model-native numeric address\n→  similarity against numeric memory representations  →  associated memory selected"
    fig.text(.69,yc+hc*.62,chain,ha="center",va="center",fontsize=11.2,color=C_FG,linespacing=1.6)
    fig.text(.69,yc+hc*.33,f'THE QUESTION DOES NOT SAY "{lab}".',ha="center",va="center",fontsize=19,weight="bold",color=C_MOD)
    fig.text(.69,yc+hc*.15,f'"{lab}" IS ONLY AN AUDIT LABEL.',ha="center",va="center",fontsize=14,weight="bold",color=C_FG)
    fig.text(.5,.055,"Memory numbers are display/audit labels only. The retriever is not given the correct number.",ha="center",va="center",fontsize=11.5,color=C_NEU,style="italic")
    foot(fig,ctx,k);return finish(fig,path,ctx,[f'THE QUESTION DOES NOT SAY "{lab}".',"IS ONLY AN AUDIT LABEL",a["entity"],"ACTIVE MEMORY A"])
# ------------------------------------------------------------------ 07 SIGNATURE · NEW QUESTION → ÇAĞRIİZ → NUMERIC FINGERPRINT
def p07(ctx,k,path):
    P=ctx["P"];q,r,a=feat(ctx);fig=new_fig()
    top=head(fig,"NEW QUESTION → ÇAĞRIİZ → NUMERIC FINGERPRINT","Inside the frozen AI, the question becomes a recall trace — and the trace becomes a numeric address.",tag="THIS LIVE RUN")
    yb=.30;hb=top-yb-.02
    panel(fig,.05,yb,.19,hb,"white",C_FG,2.0);fig.text(.145,yb+hb-.035,"A NEW QUESTION",ha="center",va="center",fontsize=13,weight="bold",color=C_FG)
    fit_text(fig,.06,yb+hb*.36,.17,hb*.48,"“"+a["question"]+"”",fs_max=14,fs_min=9,weight="bold",ha="center")
    fig.text(.145,yb+hb*.22,"enters as language",ha="center",va="center",fontsize=10,color=C_NEU,style="italic")
    panel(fig,.065,yb+.02,.16,.06,C_NIGHT,C_CYAN,1.6);fig.text(.145,yb+.05,f"+ active memory A#{FEATURE_RECORD:03d}",ha="center",va="center",fontsize=9.5,color=C_CYAN,weight="bold")
    arrow(fig,.245,yb+hb/2,.27,yb+hb/2,color=C_CYAN)
    panel(fig,.275,yb,.395,hb,C_NIGHT,C_VIO,2.2)
    fig.text(.4725,yb+hb-.035,"ÇAĞRIİZ",ha="center",va="center",fontsize=20,weight="bold",color=C_GLOW)
    fig.text(.4725,yb+hb-.075,"a question-dependent numeric recall trace created inside the frozen AI",ha="center",va="center",fontsize=10,color="#E2E8F0")
    w=np.array(r["w"],dtype=float);xs_=np.arange(len(w));ax=fig.add_axes([.30,yb+.075,.345,hb-.20]);ax.set_facecolor(C_NIGHT)
    cols=[C_GLOW if i==q["pos"]else"#8B5CF6"for i in xs_];ax.bar(xs_,w,color=cols,width=.8);ax.set_xlim(-.6,len(w)-.4);ax.set_ylim(0,max(w)*1.15)
    for s in("top","right"):ax.spines[s].set_visible(False)
    for s in("left","bottom"):ax.spines[s].set_color("#64748B")
    ax.tick_params(colors="#CBD5E1",labelsize=8);ax.set_xlabel("position inside active memory A (numbers only, no words)",fontsize=9,color="#CBD5E1",labelpad=2)
    ax.annotate(f"peak {q['peak']:.3f}",xy=(q["pos"],q["peak"]),xytext=(q["pos"]+(-6 if q["pos"]>len(w)/2 else 3),q["peak"]*1.03),color=C_GLOW,fontsize=10,weight="bold",arrowprops=dict(arrowstyle="->",color=C_GLOW))
    fig.text(.4725,yb+.025,"the AI's own attention decides where to look",ha="center",va="center",fontsize=10,color=C_CYAN,weight="bold")
    arrow(fig,.675,yb+hb/2,.7,yb+hb/2,color=C_GLOW)
    panel(fig,.705,yb,.245,hb,C_NIGHT,C_CYAN,2.2)
    fig.text(.8275,yb+hb-.035,"NUMERIC FINGERPRINT",ha="center",va="center",fontsize=14,weight="bold",color=C_CYAN)
    fig.text(.8275,yb+hb-.075,f"{len(r['p'])} numbers · model-native tensor address",ha="center",va="center",fontsize=10,color="#E2E8F0")
    pv=np.array(r["p"],dtype=float);G=pv.reshape(32,-1)if pv.size%32==0 else pv[None,:];lim=float(np.percentile(np.abs(pv),97))or 1.0
    ax2=fig.add_axes([.72,yb+.075,.215,hb-.20]);ax2.imshow(G,aspect="auto",cmap="RdBu_r",vmin=-lim,vmax=lim,interpolation="nearest");ax2.set_xticks([]);ax2.set_yticks([])
    fig.text(.8275,yb+.025,"real values from this run",ha="center",va="center",fontsize=10,color=C_CYAN,weight="bold")
    fig.text(.5,.235,"THE QUESTION BECOMES A NUMERIC FINGERPRINT.",ha="center",va="center",fontsize=22,weight="bold",color=C_FG)
    fig.text(.5,.185,"Not a memory number · not a filename · not a keyword · not a decoded seal.",ha="center",va="center",fontsize=13,color=C_NEU,weight="bold")
    fit_text(fig,.07,.045,.86,.105,"L23H12 — a specific attention channel inside the frozen model (layer 23, head 12), identified in earlier experiments and externally replicated in TEST523. "
             "LAYER 2 — VALUE TENSOR (L02-V) — the numeric memory representation that the recall trace weights into the address. Technical: frozen L23H12 native Q·K + RoPE attention; pointer-weighted, uncentered L02-V.",fs_max=10.5,fs_min=8.2,color=C_NEU,ha="center")
    foot(fig,ctx,k);return finish(fig,path,ctx,["ÇAĞRIİZ","NUMERIC FINGERPRINT",f"peak {q['peak']:.3f}",f"{len(r['p'])} numbers","THE QUESTION BECOMES A NUMERIC FINGERPRINT."])
# ===== END PART 2 / 3 — CONTINUE WITH PART 3 =====
