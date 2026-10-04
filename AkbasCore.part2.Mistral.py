def _execute_run(E,ctl,state):
    """Generator. Yields progress events, returns the result bundle. Any failed technical audit raises AuditFail (no package is sealed)."""
    run_id=ctl["run_id"];run_dir=Path(ctl["run_dir"]);I=E.info;checks=[];tm=Counter();T0=time.perf_counter();state["checks"]=checks
    def chk(name,cond):
        checks.append(name)
        if not cond:raise AuditFail(name)
    run_start_utc=utc_now();run_start_local=local_now()
    say("="*110);say(f"RUN {run_id}");say("="*110)
    # ---- 1/8 integrity ----
    say("[1/8] INTEGRITY · FROZEN-MODEL PRE-CHECK");yield ev(1,"Integrity check","Frozen model: hooks, trainable tensors, LoRA, optimizer, sampled weight sentinel.")
    fs0=E.frozen_state()
    chk("model = Mistral-7B-Instruct-v0.3",I["model_id"]==MODEL_ID);chk("architecture 32L / H4096 / 32Q / 8KV / HD128",tuple(I["arch"])==ARCH)
    chk("dtype bfloat16",I["dtype"]=="bfloat16");chk("attention SDPA",I["attn"]=="sdpa");chk("K code dimension = D120 and V code dimension = D120",D==120 and DMAX==128)
    chk("model in eval mode",not fs0["training"]);chk("trainable parameter tensors = 0",fs0["trainable_tensors"]==0);chk("no LoRA / PEFT adapter",not fs0["lora"]);chk("no optimizer object",not fs0["optimizer"])
    chk("pre-run weight sentinel = startup sentinel",fs0["sentinel"]==I["sentinel0"]);chk("pre-run weight fingerprint = startup fingerprint",fs0["fingerprint"]==I["fingerprint0"])
    if fs0["foreign_hooks"]:raise AuditFail("foreign forward/backward hooks present before run: "+str(fs0["foreign_modules"]))
    chk("no foreign forward/backward hooks before run",fs0["foreign_hooks"]==0)
    for c in checks:say("   PASS ·",c)
    # ---- 2/8 codebook ----
    say("[2/8] FIXED NEUTRAL CODEBOOK · D120");yield ev(2,"Fixed neutral codebook","32 neutral sentences → PCA basis per layer × KV head × K/V. The codebook contains none of the cartridge facts.")
    E.reset_peak();cbi=E.build_codebook();cbi["bytes"]=E.codebook_bytes();tm["codebook"]=cbi["seconds"]
    chk("codebook = 32 neutral sentences",cbi["sentences"]==32);chk("codebook SVD count = layers × 2 × KV heads",cbi["svds"]==I["arch"][0]*2*I["arch"][3])
    say(f"   {cbi['sentences']} sentences · {cbi['svds']} SVDs · {cbi['seconds']:.1f}s · {cbi['bytes']/2**20:.1f} MiB")
    # ---- 3/8 forge ----
    say("[3/8] FORGE · 16 LOCKED S1_RECORD CARTRIDGES");cards=[];tels=[];t0=time.perf_counter()
    for i in range(16):
        yield ev(3,f"Forging cartridge {i+1:02d}/16","Source record → model's own K/V → D120 projection. Each cartridge is forged alone.",done=i,total=16)
        c,tel=E.forge_card(i);cards.append(c);tels.append(tel)
        say(f" C{i+1:02d}: T={tel['T']:02d} | {OBJECTS[i]} → {IDS[i]} → {PLACES[i]} | cosK≥{min(tel['cosK']):.4f} cosV≥{min(tel['cosV']):.4f}")
    tm["forge"]=time.perf_counter()-t0
    chk("16 cartridges forged",len(cards)==16);chk("16 independent cartridges (distinct tensor storage)",len({p for c in cards for p in E.card_ptrs(c)})==32)
    chk("cartridge tensors finite",all(E.card_finite(c)for c in cards));chk("OWN first-token K/V identical to the forged source state in every layer",all(t["own_exact"]for t in tels))
    fs_m=E.frozen_state()
    if fs_m["foreign_hooks"]:raise AuditFail("foreign forward/backward hooks present after forging: "+str(fs_m["foreign_modules"]))
    chk("weight sentinel and fingerprint unchanged after forging (checked before any readout)",fs_m["sentinel"]==I["sentinel0"]and fs_m["fingerprint"]==I["fingerprint0"]);chk("no foreign hooks and no trainable tensors after forging",fs_m["foreign_hooks"]==0 and fs_m["trainable_tensors"]==0)
    # ---- 4/8 source removed ----
    say("[4/8] SOURCE REMOVED · READOUT AUDIT ARMED");yield ev(4,"Source removed","From here the readout receives only a question and the numerical cartridges. Every prompt is checked against all 16 source records.",done=16,total=16)
    sources=[src(OBJECTS[i],IDS[i],PLACES[i])for i in range(16)];audit=Counter();qtoks=[]
    def check_prompt(prompt):
        if any(s in prompt for s in sources)or"MEMORY RECORD"in prompt or"Location:"in prompt:raise AuditFail("SOURCE TEXT FOUND IN A READOUT PROMPT")
    def guard(prompt):
        check_prompt(prompt);audit["prompts_checked"]+=1;qtoks.append(E.q_tokens(prompt))
    def read(phase,card,prompt):
        guard(prompt);E.sync();t=time.perf_counter();out=E.gen(card,prompt);E.sync();dt_=time.perf_counter()-t;tm[phase]+=dt_;audit["calls_"+phase]+=1;return dict(raw=out,secs=round(dt_,4))
    probe=dict(pairs=0,identical=0)
    if I.get("fast_first_line"):
        say("   early-stop probe: first answer line of fast decoding vs full 32-token decoding")
        for pr in(q1(OBJECTS[0]),q2(IDS[0]),q1(MISSOBJ[0])):
            for c in cards[:4]:
                check_prompt(pr);a_=E.gen(c,pr);b_=E.gen_full(c,pr);probe["pairs"]+=1;probe["identical"]+=int(first(a_)==first(b_))
        check_prompt(q2(IDS[0]));a_=E.gen_nomem(q2(IDS[0]));b_=E.gen_nomem_full(q2(IDS[0]));probe["pairs"]+=1;probe["identical"]+=int(first(a_)==first(b_))
        say(f"   probe: {probe['identical']}/{probe['pairs']} identical")
        chk(f"early-stop probe: first answer lines identical to full-length decoding ({probe['identical']}/{probe['pairs']})",probe["identical"]==probe["pairs"])
    raw=dict(mw1=[],mw2=[],missing=[],absent=[],nomem=[]);state["raw"]=raw;done=0;tr=time.perf_counter()
    def prog(stage,title,body):
        el=time.perf_counter()-tr;return ev(stage,title,body,done=done,total=TOTAL_READS,eta=(el/done*(TOTAL_READS-done))if done else None)
    # ---- 5/8 MW1 + MW2 ----
    say("[5/8] MW1 · OBJECT → UNIQUE ID");dec1=[]
    for i in range(16):
        p=q1(OBJECTS[i]);rs=[]
        for j,c in enumerate(cards):
            r=read("mw1",c,p);r["c"]=j+1;rs.append(r);done+=1
            if j%4==3:yield prog(5,f"MW1 · question {i+1:02d}/16 · object → container ID",f'"{OBJECTS[i]}" is asked to all 16 independent cartridges.')
        raw["mw1"].append(dict(q=i,object=OBJECTS[i],prompt=p,reads=rs));d,u=decide_id([r["raw"]for r in rs]);dec1.append(d)
        say(f" {i+1:02d}: {'PASS'if d==IDS[i]else'FAIL'} | expected={IDS[i]} | decided={d or'UNKNOWN'} | unique={len(u)}")
        if d!=IDS[i]:
            for j,r in enumerate(rs):
                if not is_none(r["raw"]):say(f"     C{j+1:02d}: {first(r['raw'])[:72]}")
    say(f"MW1={sum(int(dec1[i]==IDS[i])for i in range(16))}/16");say("[5/8] MW2 · DECIDED ID → UNIQUE LOCATION")
    for i in range(16):
        cid=dec1[i]
        if cid is None:raw["mw2"].append(dict(q=i,id=None,skipped=True));say(f" {i+1:02d}: FAIL | Stage 1 UNKNOWN");continue
        p=q2(cid);rs=[]
        for j,c in enumerate(cards):
            r=read("mw2",c,p);r["c"]=j+1;rs.append(r);done+=1
            if j%4==3:yield prog(5,f"MW2 · question {i+1:02d}/16 · container ID → location",f"{cid} is asked to all 16 independent cartridges.")
        raw["mw2"].append(dict(q=i,id=cid,prompt=p,reads=rs));d,u=decide_place([r["raw"]for r in rs])
        say(f" {i+1:02d}: {'PASS'if d==PLACES[i]else'FAIL'} | {cid} | expected={PLACES[i]} | decided={d or'UNKNOWN'} | unique={len(u)}")
        if d!=PLACES[i]:
            for j,r in enumerate(rs):
                if not is_none(r["raw"]):say(f"     C{j+1:02d}: {first(r['raw'])[:72]}")
    # ---- 6/8 missing / absent / nomem ----
    say("[6/8] MISSING OBJECT / ABSENT ID / NOMEM")
    for i in range(8):
        for key,phase,query,mk,title in(("missing","neg",MISSOBJ[i],q1,"missing object"),("absent","neg",ABSIDS[i],q2,"absent container ID")):
            p=mk(query);rs=[]
            for j,c in enumerate(cards):
                r=read(phase,c,p);r["c"]=j+1;rs.append(r);done+=1
                if j%4==3:yield prog(6,f"{title.capitalize()} · {i+1}/8",f'"{query}" is not stored in any cartridge.')
            raw[key].append(dict(i=i,query=query,prompt=p,reads=rs))
    for i in range(8):
        p=q2(IDS[i]);guard(p);E.sync();t=time.perf_counter();o=E.gen_nomem(p);E.sync();dt_=time.perf_counter()-t;tm["nomem"]+=dt_;done+=1
        raw["nomem"].append(dict(i=i,id=IDS[i],prompt=p,raw=o,secs=round(dt_,4)))
        if i%2==1:yield prog(6,f"NOMEM control · {i+1}/8","Same Stage-2 question with no cartridge installed.")
    # ---- 7/8 audit + seal ----
    say("[7/8] FINAL AUDIT · RESULT RE-DERIVATION · SEAL");yield ev(7,"Final audit and seal","Results are re-derived from the raw generated text; frozen-model checks are repeated.",done=TOTAL_READS,total=TOTAL_READS)
    R=derive(raw);fs1=E.frozen_state();tm["engine"]=time.perf_counter()-T0
    chk("weight sentinel after run = startup sentinel",fs1["sentinel"]==I["sentinel0"]);chk("weight fingerprint after run = startup fingerprint",fs1["fingerprint"]==I["fingerprint0"])
    chk("model eval mode and trainable tensors = 0 after run",(not fs1["training"])and fs1["trainable_tensors"]==0);chk("no LoRA / optimizer after run",(not fs1["lora"])and(not fs1["optimizer"]))
    (_ for _ in()).throw(AuditFail("foreign forward/backward hooks present after run: "+str(fs1["foreign_modules"])))if fs1["foreign_hooks"]else None;chk("no foreign forward/backward hooks after run (demo installs none)",fs1["foreign_hooks"]==0);chk("every query read by all 16 cartridges (no pre-selection)",R["reads_per_query"]==[16])
    k=R["counts"];tot=R["totals"]
    for n_,v in(("MW1",k["mw1"]),("MW2",k["mw2"]),("LINKED",k["linked"]),("MISSING",k["missing"]),("ABSENT-ID",k["absent_id"]),("NOMEM",k["nomem"])):say(f" {n_:<10}= {v}/{tot[n_.lower().replace('-','_')]}")
    say(f" wrong-cartridge NONE · stage 1 = {R['mw1']['wrong_none']}/{R['mw1']['wrong_total']} · stage 2 = {R['mw2']['wrong_none']}/{R['mw2']['wrong_total']}")
    for g_,v in R["gates"].items():say(f"   gate {g_:<10}: {'PASS'if v else'FAIL'}")
    say("VERDICT:",R["verdict"])
    chk("source records found in readout prompts = 0",audit["prompts_checked"]==sum(audit["calls_"+x]for x in("mw1","mw2","neg"))+8)
    chk("results re-derived from raw text are stable",derive(raw)==R)
    E.sync();peak=E.peak_gib()
    ledger=[dict(cid=f"C{i+1:02d}",object=OBJECTS[i],id=IDS[i],location=PLACES[i],source_text=sources[i])for i in range(16)]
    cart=[dict(cid=f"C{i+1:02d}",**{k_:v for k_,v in tels[i].items()})for i in range(16)]
    bank=dict(slots=sum(t["T"]for t in tels),code_numbers=sum(t["code_numbers"]for t in tels),own_numbers=sum(t["own_numbers"]for t in tels),native_numbers=sum(t["native_numbers"]for t in tels),
              codec_macs=sum(t["codec_macs"]for t in tels),codebook_bytes=cbi["bytes"]);bank["ratio_vs_native"]=(bank["code_numbers"]+bank["own_numbers"])/bank["native_numbers"]
    reads_total=sum(audit["calls_"+x]for x in("mw1","mw2","neg"))+8
    P={"schema":"akbascore.mistral.cartridge.run.v1","project":"AkbasCore NIRVANA","run_id":run_id,"run_start_utc":run_start_utc,"run_start_local":run_start_local,"run_end_utc":utc_now(),
       "model":{"id":MODEL_ID,"arch":list(ARCH),"dtype":I["dtype"],"attn":I["attn"],"params":I["params"],"sliding_window":I["sliding_window"],"pad_id":I["pad_id"],"eos_ids":I["eos_ids"]},
       "environment":{k_:I[k_]for k_ in("gpu","gpu_total_gib","torch","transformers","gradio","python","platform")},
       "engine":{"source":"TEST474 (474.MISTRAL.final.py) engine functions, unchanged","frame":"S1_RECORD source frame + C_VERIFY readout frames","K":D,"V":D,"own_first_token":"preserved","codebook":"fixed 32-sentence neutral corpus",
                 "readout":"module-wise, one generate call per cartridge, unique aggregation, greedy","max_new_tokens":MAX_NEW,"early_stop":bool(I.get("fast_first_line")),"early_stop_probe":probe,
                 "readout_generation":("stops at the end of the first answer line (parsed results identical by construction; probe reads re-decoded at full length and compared)"if I.get("fast_first_line")else"full TEST474 decoding, up to max_new_tokens"),"compute_path":"PyTorch CUDA backend; no custom CUDA/C++ kernel","lock_sha256":LOCK_SHA,"gate_thresholds":GATE_MIN},
       "codebook":cbi,"experimenter_ledger_not_available_to_readout":ledger,"cartridges":cart,"bank":bank,"raw":raw,"results":R,
       "source_removal":{"prompts_checked":audit["prompts_checked"],"source_record_hits":0,"question_tokens_min":min(qtoks),"question_tokens_max":max(qtoks),
                         "calls":{x:audit["calls_"+x]for x in("mw1","mw2","neg")}|{"nomem":8},"slots_per_cartridge":[t["T"]for t in tels],
                         "method":"each prompt string compared with all 16 source records and the record frame; readout input ids = [PAD]×T + question tokens"},
       "frozen":{"sentinel_startup":I["sentinel0"],"sentinel_pre_run":fs0["sentinel"],"sentinel_after":fs1["sentinel"],"fingerprint_startup":I["fingerprint0"],"fingerprint_pre_run":fs0["fingerprint"],"fingerprint_after":fs1["fingerprint"],
                 "tensors":I["fp_names"],"sentinel_method":I["sentinel_method"],"trainable_tensors":fs1["trainable_tensors"],"lora":fs1["lora"],"optimizer":fs1["optimizer"],"training_mode":fs1["training"],"hooks_before":I["hooks0"],"hooks_after":fs1["hooks"],"foreign_hooks_before":fs0["foreign_hooks"],"foreign_hooks_after":fs1["foreign_hooks"],"hook_note":"hook totals include hooks installed by transformers/accelerate; only hooks from other code count as foreign"},
       "counters":{"generate_calls":reads_total,"probe_full_decodes":E.counters["probe_full_decodes"],"forge_passes":E.counters["forge_passes"],"read_calls":{"mw1":audit["calls_mw1"],"mw2":audit["calls_mw2"],"negatives":audit["calls_neg"],"nomem":8}},
       "timing":{"model_load_seconds":I["model_load_seconds"],"codebook":tm["codebook"],"forge":tm["forge"],"mw1":tm["mw1"],"mw2":tm["mw2"],"negatives":tm["neg"],"nomem":tm["nomem"],"engine":tm["engine"]},
       "gpu":{"peak_allocated_gib":peak},"checks_pre_seal":list(checks),
       "sealed_record":{"note":"Historical logs. NOT produced by this run.","test474":SEALED474,"test473":SEALED473,"test467":SEALED467},
       "reproduction":{"seed":SEED,"cell_source_sha256":I.get("cell_source_sha256"),"note":"SHA-256 of the executed cell text; the text itself is not embedded because it contains the stale-reference deny-list."},
       "scope":{"replay":"SEALED TEST474 PANEL REPLAY — the same 16 records used in TEST473 frame selection and TEST474; not a new held-out validation",
                "frame_selection":"S1_RECORD + C_VERIFY was one of 4 conditions with GOLD 16/16 and DECOY_NONE 240/240 in TEST473; chosen by fixed ordering on the same 16 records"}}
    mtexts={r["raw"]for k_ in("mw1","mw2","missing","absent")for q in raw[k_]for r in q.get("reads",[])}|{q["raw"]for q in raw["nomem"]};mtexts|={first(t)for t in mtexts}
    P=jsafe(P);P["stale_scan"]={"patterns":len(STALE)+1,"hits":0,"scope":"payload JSON text, raw model-generated strings excluded"}
    hits=stale_scan("payload",strip_model(json.dumps(P,ensure_ascii=False),mtexts),allow=(I["gpu"],SEALED474["env"]))
    chk("stale-reference scan of the payload: 0 hits",not hits)
    pb=canon(P);sha=hashlib.sha256(pb).hexdigest();payload_name=f"{run_id}_FULL_RUN_LOG_payload.json";pp=run_dir/payload_name;t_seal=time.perf_counter();pp.write_bytes(pb);say("payload sealed:",sha)
    chk("payload SHA-256 verification after write",hashlib.sha256(pp.read_bytes()).hexdigest()==sha)
    names=dict(payload=payload_name,manifest=f"run_manifest_{run_id}.json",txt=f"{run_id}_READABLE_RUN_LOG.txt",summary=f"{run_id}_SUMMARY.json",jsonl=f"{run_id}_RESULTS.jsonl",images=f"images_manifest_{run_id}.json",zip=f"AKBASCORE_MISTRAL_CARTRIDGE_ALL_{run_id}.zip")
    mp=run_dir/names["manifest"];sealed_utc=utc_now()
    manifest={"demo":"AKBASCORE NIRVANA · MISTRAL · COGNITIVE CARTRIDGE","run_id":run_id,"payload_file":payload_name,"payload_sha256":sha,"payload_bytes":len(pb),"sealed_utc":sealed_utc,"verdict":R["verdict"],
              "canonicalization":"json.dumps(sort_keys=True, separators=(',',':'), ensure_ascii=False, allow_nan=False), UTF-8","note":"Artifact integrity seal: confirms the payload is unchanged since sealing. It is not a scientific proof."}
    mp.write_text(json.dumps(manifest,indent=2,sort_keys=True,ensure_ascii=False),encoding="utf-8")
    summary={"replay_of_test":474,"model":MODEL_ID,"D":D,"cards":16,"engine":"S1_RECORD + C_VERIFY / two-stage module-wise unique decision","results":{k_:[R["counts"][k_],R["totals"][k_]]for k_ in R["counts"]},
             "gates":R["gates"],"verdict":R["verdict"],"sentinel_before":I["sentinel0"],"sentinel_after":fs1["sentinel"],"run_id":run_id,"payload_sha256":sha}
    (run_dir/names["summary"]).write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    lines=[]
    for q in raw["mw1"]:
        i=q["q"];lines.append({"phase":"MW1","query":i,"object":OBJECTS[i],"expected":IDS[i],"decided":R["mw1"]["decided"][i],"raw":[r["raw"]for r in q["reads"]]})
    for q in raw["mw2"]:
        i=q["q"];lines.append({"phase":"MW2","query":i,"id":q.get("id"),"expected":PLACES[i],"decided":R["mw2"]["decided"][i],"raw":[r["raw"]for r in q.get("reads",[])]})
    for key in("missing","absent"):
        for q in raw[key]:lines.append({"phase":key.upper(),"i":q["i"],"query":q["query"],"raw":[r["raw"]for r in q["reads"]]})
    for q in raw["nomem"]:lines.append({"phase":"NOMEM","i":q["i"],"id":q["id"],"target":PLACES[q["i"]],"raw":q["raw"]})
    (run_dir/names["jsonl"]).write_text("\n".join(json.dumps(x,ensure_ascii=False)for x in lines)+"\n",encoding="utf-8")
    (run_dir/names["txt"]).write_text(make_txt(P,R,sha,names,sealed_utc),encoding="utf-8");seal_s=time.perf_counter()-t_seal
    # ---- 8/8 posters + ZIP ----
    say("[8/8] POSTERS · IMAGE MANIFEST · ZIP");yield ev(8,"Rendering posters","Every number on every poster is checked against the re-derived runtime result before the package is sealed.")
    ctx={"P":P,"R":R,"run_id":run_id,"sha":sha,"payload_name":payload_name,"seal_seconds":seal_s,"allow":(I["gpu"],SEALED474["env"]),"N":N_POSTERS,"poster_audit":[]}
    t_r=time.perf_counter();imgs=render_all(ctx,run_dir);render_s=time.perf_counter()-t_r
    entries=[]
    for n_,(p,cap)in enumerate(imgs,1):
        with Image.open(p)as im:w_,h_=im.size;fm_=im.format;md=im.mode
        entries.append({"index":n_,"filename":p.name,"title":cap,"width":w_,"height":h_,"format":fm_,"mode":md,"size_bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
    imf=run_dir/names["images"]
    imf.write_text(json.dumps({"run_id":run_id,"payload_sha256":sha,"verdict":R["verdict"],"count":len(entries),"render_seconds":render_s,"images":entries,"poster_text_audit":ctx["poster_audit"],
                               "note":"Per-image SHA-256 is an artifact integrity seal. Poster text was checked against re-derived results and scanned for stale references before saving."},indent=2,ensure_ascii=False),encoding="utf-8")
    zp=run_dir/names["zip"];members=[p for p,_ in imgs]+[imf,mp,pp,run_dir/names["txt"],run_dir/names["summary"],run_dir/names["jsonl"]]
    for m in members:
        h=stale_scan("filename",m.name)+(stale_scan(m.name,strip_model(m.read_text(encoding="utf-8"),mtexts),allow=ctx["allow"])if m.suffix in(".json",".txt",".jsonl")else[])
        chk(f"stale-reference scan of {m.name}: 0 hits",not h)
    with zipfile.ZipFile(zp,"w",compression=zipfile.ZIP_DEFLATED)as z:
        for m in members:z.write(m,arcname=m.name)
    post=post_seal_audit(imgs,zp,[m.name for m in members],pp,sha,run_dir/names["txt"],mp,R["verdict"])
    say(f"{run_id}: {len(checks)} pre-seal + {len(post)} post-seal checks = {len(checks)+len(post)}/{len(checks)+len(post)} PASS | render {render_s:.1f}s")
    say("="*110);say("FINAL VERDICT :",R["verdict"]);say("PACKAGE       : SEALED ·",f"{len(checks)+len(post)} checks","· payload SHA-256",sha);say("ZIP           :",zp);say("="*110)
    return dict(run_id=run_id,run_dir=run_dir,P=P,R=R,sha=sha,imgs=imgs,zip=zp,payload=pp,txt=run_dir/names["txt"],manifest=mp,images_manifest=imf,summary=run_dir/names["summary"],jsonl=run_dir/names["jsonl"],
                checks=len(checks)+len(post),verdict=R["verdict"])
#<<CORE_END>>
#<<POSTERS_BEGIN>>
from matplotlib.text import Text
plt.rcParams["font.family"]="DejaVu Sans";DPI=120
C_OK,C_UNK,C_ERR,C_CTL,C_CART,C_MOD,C_FG,C_NEU,C_BG,C_SEAL="#047857","#6D28D9","#B91C1C","#475569","#B45309","#0369A1","#0F172A","#334155","#F8FAFC","#0F766E"
C_OKL,C_UNKL,C_ERRL,C_CARTL,C_MODL,C_SEALL="#D1FAE5","#EDE9FE","#FEE2E2","#FEF3C7","#E0F2FE","#CCFBF1"
MONO="DejaVu Sans Mono";BRAND="AKBASCORE NIRVANA · MISTRAL · COGNITIVE CARTRIDGE"
def mt(s):return str(s).replace("$",r"\$")
def save_jpg(fig,path):
    buf=io.BytesIO();fig.savefig(buf,format="png",dpi=fig.dpi,facecolor="white");plt.close(fig);buf.seek(0)
    with Image.open(buf)as im:
        im=im.convert("RGBA");bg=Image.new("RGB",im.size,(255,255,255));bg.paste(im,mask=im.getchannel("A"))
    bg.save(path,"JPEG",quality=92,optimize=True,progressive=False,subsampling=0)
    with Image.open(path)as chk:
        if chk.format!="JPEG"or chk.mode!="RGB":raise RuntimeError("JPEG validation failed")
    return str(path)
def finish(fig,path,ctx,expect):
    """Poster text must contain every expected runtime value and no stale reference; otherwise the package is not sealed."""
    fig.canvas.draw();texts=[t.get_text()for t in fig.findobj(Text)if t.get_text().strip()];blob="\n".join(texts);nm=Path(path).name
    ws=lambda z:re.sub(r"\s+","",z);blob_ws=ws(blob);miss=[e for e in expect if ws(e)not in blob_ws];blob=re.sub(r"[ \t]*\n[ \t]*"," ",blob)
    if miss:plt.close(fig);raise AuditFail(f"POSTER/RESULT MISMATCH in {nm}: missing {miss}")
    hits=stale_scan(nm,blob,allow=ctx["allow"])+stale_scan("filename",nm)
    if hits:plt.close(fig);raise AuditFail(f"STALE REFERENCE in {nm}: {hits[:3]}")
    ctx["poster_audit"].append(dict(file=nm,texts=len(texts),expected_values=len(expect),stale_hits=0));return save_jpg(fig,path)
def fit_text(fig,x,y,w,h,text,fs_max=13,fs_min=7,color=C_FG,family=None,ls=1.32,weight="normal"):
    fig.canvas.draw();r=fig.canvas.get_renderer();Wp=w*fig.get_figwidth()*fig.dpi;Hp=h*fig.get_figheight()*fig.dpi
    if Wp<=4 or Hp<=4:return 0
    paras=str(text if text else"(empty)").replace("\r","").split("\n");kw={"va":"top","ha":"left","color":color,"linespacing":ls,"weight":weight}
    if family:kw["family"]=family
    def wrap(c):
        out=[]
        for p in paras:out.extend(textwrap.wrap(p,c,break_long_words=True,break_on_hyphens=False)or[""])
        return out
    fs=float(fs_max);k=0.52
    for _ in range(150):
        cpl=max(6,int(Wp/(k*fs*fig.dpi/72.0)));ln=wrap(cpl);t=fig.text(x,y+h,mt("\n".join(ln)),fontsize=fs,**kw);bb=t.get_window_extent(renderer=r)
        if bb.width>Wp*1.002 and cpl>6:t.remove();k*=max(1.01,bb.width/Wp*1.01);continue
        if bb.height<=Hp:return fs
        t.remove()
        if fs>fs_min:fs=max(float(fs_min),fs-0.5);continue
        per=bb.height/max(1,len(ln));keep=max(1,int(Hp/per)-1);fig.text(x,y+h,mt("\n".join(ln[:keep]+["[… text shortened — full text in the run log]"])),fontsize=fs,**kw);return fs
    fig.text(x,y+h,"[text omitted — see run log]",fontsize=fs_min,**kw);return fs_min
def head(fig,title,sub=None,tag=None):
    Hh=fig.get_figheight();f=lambda inch:1-inch/Hh;live=bool(tag)and tag.startswith("THIS");tc=C_MOD if live else C_SEAL
    fig.text(.05,f(.42),BRAND,fontsize=12,weight="bold",color=C_CART,va="center")
    if tag:fig.text(.95,f(.42),tag,fontsize=12,weight="bold",color=tc,va="center",ha="right",bbox=dict(boxstyle="round,pad=0.35",fc="white",ec=tc,lw=1.6))
    fig.text(.05,f(.95),mt(title),fontsize=29,weight="bold",color=C_FG,va="center")
    if sub:fig.text(.05,f(1.42),mt(sub),fontsize=13.5,color=C_NEU,va="center")
    fig.add_artist(Line2D([.05,.95],[f(1.70),f(1.70)],transform=fig.transFigure,color=C_FG,lw=1.2));return f(1.85)
def foot(fig,ctx,k):
    fig.text(.5,.22/fig.get_figheight(),mt(f"AKBASCORE NIRVANA · MISTRAL | RUN {ctx['run_id']} | PAYLOAD SHA-256 {ctx['sha'][:16]}… | {k:02d}/{ctx['N']:02d}"),ha="center",va="center",fontsize=10,color=C_NEU,family=MONO)
def note(fig,x,y,w,h,plain,sci=None):
    fit_text(fig,x,y+(h*.42 if sci else 0),w,h*(.58 if sci else 1),plain,fs_max=14,fs_min=9)
    if sci:fit_text(fig,x,y,w,h*.38,sci,fs_max=10.5,fs_min=7.5,family=MONO,color=C_NEU)
def clean_ax(ax):
    for s in("top","right"):ax.spines[s].set_visible(False)
def box(fig,x,y,w,h,title,lines,color,face,tfs=13.5,bfs=11):
    fig.add_artist(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=0.01",transform=fig.transFigure,facecolor=face,edgecolor=color,lw=2.4))
    fig.text(x+w/2,y+h*.74,mt(title),ha="center",va="center",fontsize=tfs,weight="bold",color=color)
    fig.text(x+w/2,y+h*.33,mt(lines),ha="center",va="center",fontsize=bfs,color=C_FG,linespacing=1.35)
def arrow(fig,x0,y0,x1,y1,color=C_FG):fig.add_artist(FancyArrowPatch((x0,y0),(x1,y1),transform=fig.transFigure,arrowstyle="-|>",mutation_scale=24,lw=2.3,color=color))
def chip(fig,x,y,w,h,label,value,color,face):
    fig.add_artist(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=0.008",transform=fig.transFigure,facecolor=face,edgecolor=color,lw=2))
    fig.text(x+w/2,y+h*.62,mt(value),ha="center",va="center",fontsize=19,weight="bold",color=color);fig.text(x+w/2,y+h*.22,mt(label),ha="center",va="center",fontsize=10.5,color=C_FG)
def kn(c,tot,k):return f"{c[k]}/{tot[k]}"
LBL={"mw1":"MW1","mw2":"MW2","linked":"LINKED","missing":"MISSING","absent_id":"ABSENT-ID","nomem":"NOMEM"}
def ctx_counts(ctx):R=ctx["R"];return R["counts"],R["totals"]
def sc(s):return s if len(s)<=14 else s[:13]+"…"
# ------------------------------------------------------------------ 01 pipeline
def p01(ctx,k,path):
    P=ctx["P"];R=ctx["R"];C,T=ctx_counts(ctx);fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white");tl=[c["T"]for c in P["cartridges"]];m=P["model"]["arch"]
    top=head(fig,"WHAT JUST HAPPENED?","Text went in once, became numbers, the text was removed, and a frozen model answered from the numbers.",tag="THIS LIVE RUN")
    bw,bh=.205,.17;xs=[.05+i*(bw+.0267)for i in range(4)];y1=.60
    box(fig,xs[0],y1,bw,bh,"① SOURCE TEXT",f"16 short records\n{min(tl)-1}–{max(tl)-1} tokens each",C_FG,C_BG)
    box(fig,xs[1],y1,bw,bh,"② FORGE","one forward pass per record\nmodel's own K/V states",C_MOD,C_MODL)
    box(fig,xs[2],y1,bw,bh,"③ D120 CARTRIDGE",f"16 independent cartridges\n{P['bank']['slots']} slots · numbers only",C_CART,C_CARTL)
    box(fig,xs[3],y1,bw,bh,"④ SOURCE REMOVED",f"readout prompts audited: {P['source_removal']['prompts_checked']}\nsource records found: {P['source_removal']['source_record_hits']}",C_ERR,C_ERRL)
    for i in range(3):arrow(fig,xs[i]+bw,y1+bh/2,xs[i+1],y1+bh/2)
    arrow(fig,xs[3]+bw/2,y1,xs[3]+bw/2,.485)
    ex_o=OBJECTS[0];ex_id=R["mw1"]["decided"][0]or"UNKNOWN";ex_pl=R["mw2"]["decided"][0]or"UNKNOWN"
    bw2=.168;g2=.0125;xs2=[.05+i*(bw2+g2)for i in range(5)];y2=.315;bh2=.165
    box(fig,xs2[0],y2,bw2,bh2,"⑤ QUESTION",f'object\n"{ex_o}"',C_FG,C_BG,tfs=12.5,bfs=10.5)
    box(fig,xs2[1],y2,bw2,bh2,"⑥ MW1",f"16 isolated reads\none unique answer",C_MOD,C_MODL,tfs=12.5,bfs=10.5)
    box(fig,xs2[2],y2,bw2,bh2,"⑦ CONTAINER ID",f"{ex_id}",C_OK,C_OKL,tfs=12.5,bfs=14)
    box(fig,xs2[3],y2,bw2,bh2,"⑧ MW2","same 16 cartridges\nasked about the ID",C_MOD,C_MODL,tfs=12.5,bfs=10.5)
    box(fig,xs2[4],y2,bw2,bh2,"⑨ LOCATION",f"{ex_pl}",C_OK,C_OKL,tfs=12.5,bfs=14)
    for i in range(4):arrow(fig,xs2[i]+bw2,y2+bh2/2,xs2[i+1],y2+bh2/2)
    fig.text(.05,.29,"Example shown: the first locked record (Q01), not selected after the run. All 16 locked questions are on poster 06.",fontsize=10.5,color=C_NEU,style="italic")
    cw=.1325;xc=[.05+i*(cw+.01)for i in range(6)]
    for i,key in enumerate(LBL):
        good=C[key]>=GATE_MIN[key];chip(fig,xc[i],.155,cw,.09,LBL[key],kn(C,T,key),C_OK if good else C_ERR,C_OKL if good else C_ERRL)
    fig.text(.05,.128,f"THIS LIVE RUN · verdict by TEST474 gates: {R['verdict']}",fontsize=12,weight="bold",color=C_OK if R["verdict"].startswith("PASS")else C_ERR)
    fit_text(fig,.05,.04,.9,.075,f"Frozen {MODEL_SHORT} · {m[0]} layers · hidden {m[1]} · {m[2]} Q / {m[3]} KV heads · head {m[4]} · BF16 · SDPA · K = D120 + V = D120 · OWN first-token state kept · greedy · no training, no LoRA, no optimizer, no weight update",fs_max=11,fs_min=8,color=C_NEU)
    foot(fig,ctx,k);return finish(fig,path,ctx,[kn(C,T,x)for x in LBL]+[R["verdict"],str(P["source_removal"]["prompts_checked"])])
# ------------------------------------------------------------------ 02 anatomy
def p02(ctx,k,path):
    P=ctx["P"];C0=P["cartridges"][0];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white");m=P["model"]["arch"];B=P["bank"]
    top=head(fig,"WHAT IS INSIDE A CARTRIDGE?","Numbers only: the model's own K/V states, projected on 120 fixed directions. Shown for cartridge C01.",tag="THIS LIVE RUN")
    L_=P["experimenter_ledger_not_available_to_readout"][0]["source_text"]
    steps=[("SOURCE RECORD (forge time only)",L_.replace("\n"," | "),C_FG,C_BG),("TOKENS",f"{C0['T']} slots: 1 first-token state + {C0['T']-1} content tokens",C_FG,C_BG),
           ("MODEL FORWARD",f"K and V per layer: {m[0]} layers × {m[3]} KV heads × {m[4]} dims",C_MOD,C_MODL),
           ("D120 PROJECTION","fixed neutral codebook → 120 coefficients per head and token (K and V)",C_CART,C_CARTL),("NUMERICAL CARTRIDGE",f"{C0['code_numbers']:,} coefficients + {C0['own_numbers']:,} raw first-token numbers",C_OK,C_OKL)]
    y=top-.01;hh=.108
    for i,(t,b,c,fc)in enumerate(steps):
        yy=y-(i+1)*(hh+.022)+.022;fig.add_artist(FancyBboxPatch((.05,yy),.40,hh,boxstyle="round,pad=0,rounding_size=0.008",transform=fig.transFigure,facecolor=fc,edgecolor=c,lw=2.2))
        fig.text(.062,yy+hh-.018,t,fontsize=11.5,weight="bold",color=c,va="top");fit_text(fig,.062,yy+.006,.376,hh-.04,b,fs_max=10.5,fs_min=8,family=MONO if i==0 else None)
        if i<len(steps)-1:fig.text(.25,yy-.011,"▼",fontsize=10,color=C_NEU,ha="center",va="center")
    ex=C0["excerpt"];V=np.array(ex["values"],dtype=float);ax=fig.add_axes([.54,.545,.40,.215]);lim=np.percentile(np.abs(V),97)or 1.0
    im=ax.imshow(V,aspect="auto",cmap="RdBu_r",vmin=-lim,vmax=lim);ax.set_xlabel("PCA coefficient index (0–119)",fontsize=9.5,labelpad=2);ax.set_ylabel("content token",fontsize=9.5)
    ax.set_title(f"REAL DATA · C01 · layer {ex['layer']} · K head {ex['head']} · coefficients",fontsize=11.5,weight="bold",loc="left",pad=8);fig.colorbar(im,cax=fig.add_axes([.945,.545,.008,.215]))
    cK=np.array([c["cosK"]for c in P["cartridges"]]);cV=np.array([c["cosV"]for c in P["cartridges"]]);ax2=fig.add_axes([.54,.255,.40,.17]);xs=np.arange(len(cK[0]))
    for arr,col,lab in((cK,C_CART,"K"),(cV,C_MOD,"V")):
        ax2.fill_between(xs,arr.min(0),arr.max(0),color=col,alpha=.2);ax2.plot(xs,arr.mean(0),color=col,lw=2.2,label=f"{lab}: mean over 16 cartridges (band = min–max)")
    lo=min(cK.min(),cV.min());ax2.set_ylim(max(0,lo-.01),1.003);ax2.set_xlabel("transformer layer",fontsize=9.5,labelpad=2);ax2.set_ylabel("cosine to own K/V",fontsize=9.5)
    ax2.legend(frameon=False,fontsize=8.8,loc="lower left");ax2.set_title("MEASURED · how faithful is the D120 projection?",fontsize=11.5,weight="bold",loc="left",pad=8);clean_ax(ax2);ax2.grid(alpha=.25)
    fit_text(fig,.05,.045,.9,.085,f"NO HUMAN-READABLE SOURCE COPY. The cartridge holds {B['code_numbers']+B['own_numbers']:,} numbers for all 16 records ({B['ratio_vs_native']*100:.1f}% of the model's native K/V size: this is a projection, not a size reduction). The {len(NEUTRAL)}-sentence neutral codebook ({B['codebook_bytes']/2**20:.1f} MiB) is shared and holds none of the facts.",fs_max=11,fs_min=8)
    foot(fig,ctx,k);return finish(fig,path,ctx,[f"{C0['T']} slots",f"{B['code_numbers']+B['own_numbers']:,}",f"{B['ratio_vs_native']*100:.1f}%"])
# ------------------------------------------------------------------ 03 source removal
def p03(ctx,k,path):
    P=ctx["P"];S=P["source_removal"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"WAS THE SOURCE REALLY REMOVED?","What the frozen model receives at readout: a question as text, the knowledge only as installed numbers.",tag="THIS LIVE RUN")
    bw,bh=.27,.25;xs=[.05,.365,.68];y0=top-bh-.03
    box(fig,xs[0],y0,bw,bh,"FORGE TIME","SOURCE TEXT = PRESENT\n\nused once, inside the\nforward pass that makes a cartridge",C_FG,C_BG,bfs=11.5)
    box(fig,xs[1],y0,bw,bh,"AFTER FORGE","SOURCE TEXT = REMOVED\nNOT PROVIDED TO READOUT\n\nNO HUMAN-READABLE SOURCE COPY",C_ERR,C_ERRL,bfs=11.5)
    box(fig,xs[2],y0,bw,bh,"READOUT INPUT","QUESTION  +  NUMERICAL\nCARTRIDGE K/V\n\nMODEL WEIGHTS: UNCHANGED",C_MOD,C_MODL,bfs=11.5)
    arrow(fig,xs[0]+bw,y0+bh/2,xs[1],y0+bh/2);arrow(fig,xs[1]+bw,y0+bh/2,xs[2],y0+bh/2)
    calls=S["calls"];rows=[("MW1 · object → ID",calls["mw1"]),("MW2 · ID → location",calls["mw2"]),("missing object + absent ID",calls["neg"]),("NOMEM control",calls["nomem"])]
    ax=fig.add_axes([.27,.295,.60,.175]);ax.barh(range(4),[v for _,v in rows],color=[C_MOD,C_MOD,C_UNK,C_CTL]);ax.invert_yaxis();ax.set_yticks(range(4));ax.set_yticklabels([a for a,_ in rows],fontsize=11)
    for i,(_,v)in enumerate(rows):ax.text(v+4,i,f"{v} readout prompts · source records found: 0",va="center",fontsize=11,weight="bold")
    ax.set_xlim(0,max(v for _,v in rows)*1.9);ax.set_xlabel("readout prompts checked against all 16 source records",fontsize=10,labelpad=3);clean_ax(ax)
    note(fig,.05,.045,.9,.13,f"{S['prompts_checked']} readout prompts were compared with every source record before use; the readout input is [PAD]×slots + question tokens ({S['question_tokens_min']}–{S['question_tokens_max']} question tokens). The question must name the object or container ID it asks about — that is the question, not the record.",
         "A cartridge is a set of K/V tensors (coefficients + reconstructed rows). It is not text, a summary, or an embedding of the sentence.")
    foot(fig,ctx,k);return finish(fig,path,ctx,[str(S["prompts_checked"]),"SOURCE TEXT = REMOVED"])
# ------------------------------------------------------------------ 04 bank
def p04(ctx,k,path):
    P=ctx["P"];C=P["cartridges"];L=P["experimenter_ledger_not_available_to_readout"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"THE CARTRIDGE BANK · 16 INDEPENDENT CARTRIDGES","Each cartridge was forged alone: its own forward pass, its own first-token state, its own numbers.",tag="THIS LIVE RUN")
    gw,gh=.098,(top-.17)/4
    for i,c in enumerate(C):
        r,q=divmod(i,4);x=.05+q*(gw+.007);y=top-(r+1)*gh-.005
        fig.add_artist(FancyBboxPatch((x,y),gw,gh-.012,boxstyle="round,pad=0,rounding_size=0.008",transform=fig.transFigure,facecolor=C_CARTL,edgecolor=C_CART,lw=2.2))
        fig.text(x+gw/2,y+gh-.04,c["cid"],fontsize=16,weight="bold",color=C_CART,ha="center",va="center");fig.text(x+gw/2,y+(gh-.012)*.52,f"{c['T']} slots",fontsize=10.5,ha="center",va="center",color=C_FG)
        fig.text(x+gw/2,y+(gh-.012)*.30,f"{(c['code_numbers']+c['own_numbers'])*2/1024:.0f} KiB",fontsize=10,ha="center",va="center",color=C_NEU);fig.text(x+gw/2,y+(gh-.012)*.12,"numbers only",fontsize=8.5,ha="center",va="center",color=C_NEU,style="italic")
    ax=fig.add_axes([.52,.14,.43,top-.17]);ax.axis("off");ax.set_xlim(0,1);ax.set_ylim(0,1)
    ax.add_patch(Rectangle((0,.93),1,.07,color=C_SEALL,transform=ax.transAxes));ax.text(.01,.965,"FORGE-TIME LEDGER (kept by the experimenter for scoring)",fontsize=10.5,weight="bold",color=C_SEAL,va="center")
    ax.text(.01,.91,"The readout never receives this table.",fontsize=9.5,color=C_NEU,va="top",style="italic")
    for i,l in enumerate(L):
        y=.79-i*.0495;ax.text(.01,y,l["cid"],fontsize=9.5,weight="bold",color=C_CART,va="center");ax.text(.12,y,l["object"],fontsize=9.5,va="center",color=C_FG);ax.text(.52,y,l["id"],fontsize=9.5,family=MONO,va="center",color=C_FG);ax.text(.70,y,l["location"],fontsize=9.5,va="center",color=C_FG)
    ax.text(.12,.855,"object",fontsize=9,weight="bold",color=C_NEU);ax.text(.52,.855,"container ID",fontsize=9,weight="bold",color=C_NEU);ax.text(.70,.855,"location",fontsize=9,weight="bold",color=C_NEU)
    fit_text(fig,.05,.045,.43,.075,"Cartridge size = D120 coefficients + raw first-token state (bf16). Independent means: no cartridge was forged with, or conditioned on, another.",fs_max=11,fs_min=8,color=C_NEU)
    foot(fig,ctx,k);return finish(fig,path,ctx,["C01","C16",f"{C[0]['T']} slots"])
# ------------------------------------------------------------------ 05 one question
def lane_row(fig,ctx,y,reads,kind,gold_idx,decided_fn):
    bw=(0.90-15*.004)/16;bh=.075
    for j,r in enumerate(reads):
        x=.05+j*(bw+.004);txt=first(r["raw"]);none=is_none(r["raw"])
        okv=(parse_id(r["raw"])if kind=="id"else parse_place(r["raw"]))is not None and not none
        face,edge,col=(C_OK,C_OK,"white")if(j==gold_idx and okv)else((C_UNKL,C_UNK,C_UNK)if none else(C_ERRL,C_ERR,C_ERR))
        fig.add_artist(FancyBboxPatch((x,y),bw,bh,boxstyle="round,pad=0,rounding_size=0.004",transform=fig.transFigure,facecolor=face,edgecolor=edge,lw=1.6))
        fig.text(x+bw/2,y+bh-.012,f"C{j+1:02d}",fontsize=8.5,weight="bold",color=col if j==gold_idx else C_NEU,ha="center",va="top")
        fig.text(x+bw/2,y+.027,textwrap.fill("NONE"if none else sc(txt),9),fontsize=7.8,weight="bold",color=col,ha="center",va="center",linespacing=1.0)
def p05(ctx,k,path):
    P=ctx["P"];R=ctx["R"];raw=P["raw"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white");i=0
    top=head(fig,"ONE QUESTION, STEP BY STEP","Every cartridge reads the same question in isolation. Only the cartridge that holds the answer returns it.",tag="THIS LIVE RUN")
    q=raw["mw1"][i];rs=q["reads"];d1,u1=decide_id([r["raw"]for r in rs])
    fig.text(.05,top-.025,"STAGE 1 · MW1 · OBJECT → CONTAINER ID",fontsize=13,weight="bold",color=C_MOD,va="top")
    fit_text(fig,.05,top-.105,.9,.06,f'Question to all 16 cartridges: Does this memory explicitly contain the object "{OBJECTS[i]}"? If yes, answer only its container identifier. If no, answer exactly NONE.',fs_max=11.5,fs_min=8.5,color=C_NEU)
    lane_row(fig,ctx,top-.20,rs,"id",i,None)
    fig.text(.5,top-.225,"▼  unique aggregation of the 16 answers",ha="center",fontsize=11,color=C_NEU)
    fig.add_artist(FancyBboxPatch((.30,top-.315),.40,.062,boxstyle="round,pad=0,rounding_size=0.008",transform=fig.transFigure,facecolor=C_OKL if d1 else C_ERRL,edgecolor=C_OK if d1 else C_ERR,lw=2.2))
    fig.text(.5,top-.284,f"unique IDs = {{{', '.join(u1)}}}  →  {'accepted: '+d1 if d1 else 'UNKNOWN'}",ha="center",va="center",fontsize=13,weight="bold",color=C_OK if d1 else C_ERR)
    q2d=raw["mw2"][i];y2=top-.375
    fig.text(.05,y2+.02,"STAGE 2 · MW2 · CONTAINER ID → LOCATION",fontsize=13,weight="bold",color=C_MOD,va="top")
    if q2d.get("skipped"):
        fit_text(fig,.05,y2-.08,.9,.06,"Stage 1 returned no unique ID, so Stage 2 was not run for this question.",fs_max=12,color=C_ERR);d2="UNKNOWN";u2=[]
    else:
        fit_text(fig,.05,y2-.075,.9,.06,f'Question to all 16 cartridges: Does this memory explicitly contain container "{q2d["id"]}"? If yes, answer only its location. If no, answer exactly NONE.',fs_max=11.5,fs_min=8.5,color=C_NEU)
        lane_row(fig,ctx,y2-.17,q2d["reads"],"place",i,None);d2o,u2=decide_place([r["raw"]for r in q2d["reads"]]);d2=d2o or"UNKNOWN"
        fig.text(.5,y2-.195,"▼  unique aggregation of the 16 answers",ha="center",fontsize=11,color=C_NEU)
        fig.add_artist(FancyBboxPatch((.30,y2-.285),.40,.062,boxstyle="round,pad=0,rounding_size=0.008",transform=fig.transFigure,facecolor=C_OKL if d2o else C_ERRL,edgecolor=C_OK if d2o else C_ERR,lw=2.2))
        fig.text(.5,y2-.254,f"unique places = {{{', '.join(u2)}}}  →  {'ANSWER: '+d2o if d2o else 'UNKNOWN'}",ha="center",va="center",fontsize=13,weight="bold",color=C_OK if d2o else C_ERR)
    lg=[Line2D([0],[0],marker="s",color="w",markerfacecolor=C_OK,markersize=12,label="cartridge returned the answer"),Line2D([0],[0],marker="s",color="w",markerfacecolor=C_UNKL,markeredgecolor=C_UNK,markersize=12,label="cartridge said NONE"),Line2D([0],[0],marker="s",color="w",markerfacecolor=C_ERRL,markeredgecolor=C_ERR,markersize=12,label="unexpected output")]
    fig.legend(handles=lg,loc="lower center",bbox_to_anchor=(.5,.075),ncol=3,frameon=False,fontsize=10.5)
    fig.text(.05,.045,"There is no router: all 16 cartridges are always read. Boxes show the first line each cartridge generated (live, unedited).",fontsize=10.5,color=C_NEU)
    foot(fig,ctx,k);return finish(fig,path,ctx,["unique aggregation of the 16 answers",OBJECTS[i]])
# ===== END PART 2 / 3 — CONTINUE WITH PART 3 =====
