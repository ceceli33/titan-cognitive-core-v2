# =====================================================================================================================
# SEVEN ACADEMIC FIGURES (English) — every number is read from the payload P of the current run
# =====================================================================================================================
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":.8,"axes.edgecolor":"#374151","axes.labelcolor":"#111827",
                     "xtick.color":"#374151","ytick.color":"#374151","xtick.labelsize":8,"ytick.labelsize":8,"axes.labelsize":9,"axes.titlesize":9.5,
                     "axes.titleweight":"bold","axes.titlelocation":"left","legend.fontsize":8,"legend.frameon":False})
FIG_W,FIG_H=13.333,7.5;LAYOUT_DPI=100;OUT_DPI=300
INK,INK2,MUTED,RULE,GRIDC,PANEL="#111827","#374151","#6B7280","#D1D5DB","#E5E7EB","#F8FAFC"
BLUE,BLUE_L,TEAL,TEAL_L,ORANGE,ORANGE_L,AMBER,AMBER_L,RED,RED_L,VIOLET="#1D4ED8","#DBEAFE","#0F766E","#CCFBF1","#C2410C","#FFEDD5","#B45309","#FEF3C7","#B91C1C","#FEE2E2","#7C3AED"
SEGC={"prefix":"#9CA3AF","A":"#1D4ED8","B":"#0F766E","C":"#7C3AED","C′":"#BE185D"}
STATUS_COL={"CORRECT_KNOWN":(TEAL,TEAL_L),"CORRECT_ABSTENTION":(BLUE,BLUE_L),"UNSUPPORTED":(ORANGE,ORANGE_L),"MISSED_KNOWN":(AMBER,AMBER_L),
            "WRONG_KNOWN":(RED,RED_L),"UNSCORED":(MUTED,"#F3F4F6")}
MONO="DejaVu Sans Mono"
def mt(s):return str(s).replace("$",r"\$")
def ascii_(s):return unicodedata.normalize("NFKD",str(s)).encode("ascii","ignore").decode("ascii")
def short(s,n):s=str(s).replace("\n"," ");return s if len(s)<=n else s[:n-1]+"…"
def new_fig(h=FIG_H):return plt.figure(figsize=(FIG_W,h),dpi=LAYOUT_DPI,facecolor="white")
def fit_text(fig,x,y,w,h,text,fs_max=10,fs_min=6.0,color=INK,family=None,ls=1.3,weight="normal",ha="left"):
    fig.canvas.draw();r=fig.canvas.get_renderer();Wp=w*fig.get_figwidth()*fig.dpi;Hp=h*fig.get_figheight()*fig.dpi
    if Wp<=4 or Hp<=4:return 0
    paras=str(text if text else "(empty)").replace("\r","").split("\n");x0=x+w/2 if ha=="center" else x
    kw={"va":"top","ha":ha,"color":color,"linespacing":ls,"weight":weight,"multialignment":ha}
    if family:kw["family"]=family
    def wrap(c):
        out=[]
        for p in paras:out.extend(textwrap.wrap(p,c,break_long_words=True,break_on_hyphens=False,subsequent_indent=("  " if p.startswith("• ") else ""))or[""])
        return out
    fs=float(fs_max);k=0.52
    for _ in range(160):
        cpl=max(6,int(Wp/(k*fs*fig.dpi/72.0)));ln=wrap(cpl);t=fig.text(x0,y+h,mt("\n".join(ln)),fontsize=fs,**kw);bb=t.get_window_extent(renderer=r)
        if bb.width>Wp*1.002 and cpl>6:t.remove();k*=max(1.01,bb.width/Wp*1.01);continue
        if bb.height<=Hp:return fs
        t.remove()
        if fs>fs_min:fs=max(float(fs_min),fs-0.25);continue
        per=bb.height/max(1,len(ln));keep=max(1,int(Hp/per)-1);fig.text(x0,y+h,mt("\n".join(ln[:keep]+["[… full text in RUN_SUMMARY.txt]"])),fontsize=fs,**kw);return fs
    fig.text(x0,y+h,"[text omitted — see the run log]",fontsize=fs_min,**kw);return fs_min
def fit_line(fig,x,y,text,maxw,fs,fs_min=6.0,**kw):
    """single line of text, font size reduced until its width fits maxw (figure fraction)."""
    fig.canvas.draw();r=fig.canvas.get_renderer();W=maxw*fig.get_figwidth()*fig.dpi
    while True:
        t=fig.text(x,y,mt(text),fontsize=fs,**kw)
        if t.get_window_extent(renderer=r).width<=W or fs<=fs_min:return t
        t.remove();fs-=.25
def frame(fig,x,y,w,h,ec=INK2,fc="white",lw=.9,ls="-",z=-1):
    fig.add_artist(Rectangle((x,y),w,h,transform=fig.transFigure,facecolor=fc,edgecolor=ec,lw=lw,ls=ls,zorder=z))
def block(fig,x,y,w,h,stage,title,body,ec=INK2,fc="white",mono=True,bfs=8.2,hc=None):
    frame(fig,x,y,w,h,ec,fc);frame(fig,x,y+h-.034,w,.034,ec,hc or ec,lw=.9)
    fig.text(x+.007,y+h-.017,stage,fontsize=7.4,weight="bold",color="white",va="center")
    fit_line(fig,x+.007,y+h-.054,title,w-.014,9.4,weight="bold",color=INK,va="center")
    if body:fit_text(fig,x+.007,y+.008,w-.014,h-.08,body,fs_max=bfs,fs_min=5.8,family=MONO if mono else None,color=INK2)
def arrow(fig,x0,y0,x1,y1,color=INK2,lw=1.2,ms=12,cs="arc3"):
    fig.add_artist(FancyArrowPatch((x0,y0),(x1,y1),transform=fig.transFigure,arrowstyle="-|>",mutation_scale=ms,lw=lw,color=color,connectionstyle=cs,zorder=6))
def clean(ax,grid=True):
    for s in ("top","right"):ax.spines[s].set_visible(False)
    if grid:ax.grid(True,color=GRIDC,lw=.5);ax.set_axisbelow(True)
def header(fig,k,title,sub,tag,tagc,h=FIG_H):
    y=1-.045*FIG_H/h;fig.text(.03,y,f"Fig. {k}",fontsize=12,weight="bold",color=MUTED,va="center")
    tg=fig.text(.97,y,tag,fontsize=8.4,weight="bold",color=tagc,va="center",ha="right",bbox=dict(boxstyle="square,pad=0.35",fc="white",ec=tagc,lw=1.0))
    fig.canvas.draw();tw=tg.get_window_extent(renderer=fig.canvas.get_renderer()).width/(fig.get_figwidth()*fig.dpi)
    fit_line(fig,.078,y,title,.97-tw-.02-.078,16.5,11,weight="bold",color=INK,va="center")
    fit_line(fig,.03,y-.038*FIG_H/h,sub,.94,9.2,6.5,color=INK2,va="center")
    fig.add_artist(Line2D([.03,.97],[y-.06*FIG_H/h,y-.06*FIG_H/h],transform=fig.transFigure,color=INK2,lw=.8))
def footer(fig,P,k,caption,h=FIG_H):
    s=FIG_H/h;st=P["startup"]
    fig.add_artist(Line2D([.03,.97],[.132*s,.132*s],transform=fig.transFigure,color=RULE,lw=.6))
    fit_text(fig,.03,.045*s,.94,.082*s,caption,fs_max=8.4,fs_min=6.2,color=INK2)
    fig.add_artist(Line2D([.03,.97],[.038*s,.038*s],transform=fig.transFigure,color=RULE,lw=.6))
    fig.text(.5,.025*s,mt(f"{APP_SHORT} · {AUTHOR} · {ORG} · {LICENSE_NAME} · {REPO}"),ha="center",va="center",fontsize=6.6,color=INK2)
    fig.text(.5,.010*s,mt(f"run {P['run_id']} · {P['started_local']} · {st.get('model_id')} · v1.4 engine SHA-256 {str(st.get('v14_sha256'))[:12]}… · TEST664 {str(st.get('source_sha256'))[:12]}… · Fig. {k}/7 · {COPYRIGHT}"),
             ha="center",va="center",fontsize=6.1,color=MUTED,family=MONO)
    if P.get("mock"):
        fig.text(.5,.5,"LAYOUT TEST — SYNTHETIC VALUES — NOT A MODEL RUN",fontsize=40,color="#DC2626",alpha=.18,ha="center",va="center",rotation=18,weight="bold",zorder=50)
def save_jpeg(fig,path,P,k,desc):
    buf=io.BytesIO();fig.savefig(buf,format="png",dpi=OUT_DPI,facecolor="white");plt.close(fig);buf.seek(0)
    with Image.open(buf) as im:rgb=im.convert("RGB")
    ex=Image.Exif();ex[0x010E]=ascii_(f"Fig. {k} {desc} | {APP_SHORT} | run {P['run_id']}");ex[0x013B]=ascii_(f"{AUTHOR} / {ORG}");ex[0x8298]=ascii_(f"{COPYRIGHT} - {LICENSE_NAME} - {REPO}")
    rgb.save(path,"JPEG",quality=94,optimize=True,subsampling=0,dpi=(OUT_DPI,OUT_DPI),exif=ex.tobytes())
    with Image.open(path) as c_:
        if c_.format!="JPEG" or c_.size[0]<3000:raise RuntimeError(f"JPEG check failed for {path}")
    return str(path)
def chip(fig,x,y,status,w=.075,h=.024,fs=6.6):
    ec,fc=STATUS_COL.get(status,(MUTED,"#F3F4F6"))
    fig.add_artist(Rectangle((x,y),w,h,transform=fig.transFigure,facecolor=fc,edgecolor=ec,lw=.8,zorder=4))
    fig.text(x+w/2,y+h/2,STATUS_SHORT.get(status,status),fontsize=fs,color=ec,weight="bold",ha="center",va="center",zorder=5)
def fmt_bytes(b):return f"{b/2**20:.2f} MiB" if b>=2**20 else f"{b/1024:.1f} KiB"
def build(P,scn):return P["builds"][scn]
def prim(P,scn):return P["primary"][scn]
def resp(r,qid):
    for q in r["responses"]:
        if q["question_id"]==qid:return q
    return None
def arch_line(st):a=st.get("arch") or list(ARCH_EXPECTED);return f"{a[0]} layers · hidden {a[1]} · {a[2]} Q / {a[3]} KV heads · head dim {a[4]}"

# ------------------------------------------------------------------ FIG 1 · architecture
def fig1(P,k,path):
    st=P["startup"];b=build(P,"AKBASCORE");r=prim(P,"AKBASCORE");fig=new_fig();T=b["memory"]["T"];Pn=b["prefix_tokens"]
    header(fig,k,"Silicon Hippocampus architecture — from source fact to source-free answer",
           f"{st.get('model_id')} · {arch_line(st)} · {st.get('dtype')} · Demonstration A (AKBASCORE), condition ABC, DC_CUT3 · all numbers measured in this run",
           "ARCHITECTURE · VALUES FROM THIS RUN",INK2)
    W=.218;G=.02;x0=.03;ytop=.555;H1=.30
    cars=b["cartridges"];steps=b["steps"]
    src="\n".join(f"{c['slot']}: {c['text']}" for c in cars)
    block(fig,x0,ytop,W,H1,"1 · SOURCE FACTS","Read once per memory build",src,ec=INK2,mono=False,bfs=7.6)
    wr=[]
    for c,s in zip(cars,steps):
        fw=s["forwards"];ids=[f for f in fw if f["kind"]=="ids"]
        wr.append(f"{c['slot']}: 1 token-id forward · {ids[0]['n'] if ids else '?'} ids · no cache\n   H3 {c['rows_in_memory']}×{st['arch'][1] if st.get('arch') else 3584} · K/V L0–3 {c['rows_in_memory']} rows")
    block(fig,x0+W+G,ytop,W,H1,"2 · INDEPENDENT CARTRIDGE WRITING","One frozen forward per fact","\n".join(wr)+f"\n(each forward reads the {Pn}-token instruction prefix + one fact; only the first cartridge keeps the prefix rows)",ec=BLUE,hc=BLUE)
    sz="\n".join(f"{c['slot']}: {c['h3_bytes']:,} B H3 + {c['kv_lower_bytes']:,} B K/V = {fmt_bytes(c['cartridge_bytes'])}" for c in cars)
    block(fig,x0+2*(W+G),ytop,W,H1,"3 · NUMERICAL MEMORY REPRESENTATION","A cartridge holds numbers, not text",
          f"H3 = residual stream after layer 3 (one 3584-d vector per token)\nK/V L0–3 = attention keys/values of layers 0–3 [1, 4, rows, 128] bf16\n{sz}",ec=TEAL,hc=TEAL)
    cons=[]
    for s in steps:
        if s["kind"]=="init":cons.append(f"init({s['slot']}): {s['new_len']} rows ({1000*s['seconds']:.0f} ms)")
        else:
            e=[f for f in s["forwards"] if f["kind"]=="embeds"]
            cons.append(f"append {s['slot']}: zero-emb {e[0]['n'] if e else '?'} rows · cache {e[0]['past_len'] if e else '?'} · prior rows {'unchanged ✓' if s['prior_rows_bitwise_unchanged'] else 'CHANGED ✗'}")
    block(fig,x0+3*(W+G),ytop,W,H1,"4 · MEMORY CONSOLIDATION (APPEND-ONLY)",f"Layers 4–27 computed for new rows only",
          "\n".join(cons)+f"\nmemory {T} rows × {b['memory']['layers']} layers = {fmt_bytes(b['memory']['bytes'])}\nstep-wise build = native_memory659(): {b['engine_path_identical']}",ec=VIOLET,hc=VIOLET)
    for i in range(3):arrow(fig,x0+(i+1)*W+i*G+.002,ytop+H1/2,x0+(i+1)*(W+G)-.002,ytop+H1/2)
    yb=.165;H2=.33
    block(fig,x0+3*(W+G),yb,W,H2,"5 · NATIVE CUT3 INJECTION","Where the cartridge enters the model",
          "In each consolidation forward the input embeddings are all zero and the engine replaces the layer-3 output with the stored H3 "
          "(engine forward hook, removed after the call). Layers 4–27 then compute K/V for the new rows while attending to the existing memory.\n"
          f"K/V of layers 0–3 are re-phased (RoPE) to their memory positions and written with the native CUDA kernel append_kv (BF16 row copy, bitwise = torch.cat).\n"
          f"layers 0–3 rows = cartridge rows: {all(s['lower_rows_equal_cartridge'] for s in steps)}",ec=ORANGE,mono=False,bfs=7.6,hc=ORANGE)
    fz=P["verify"].get("frozen_after") or {}
    block(fig,x0+2*(W+G),yb,W,H2,"6 · FROZEN TRANSFORMER COMPUTATION","The memory is the attention cache",
          f"{st.get('model_id')}\n{arch_line(st)}\n{st.get('dtype')} · attention {st.get('attn')}\ntrainable tensors {fz.get('trainable_tensors')}\n"
          f"sentinel before = after: {P['verify'].get('sentinel_unchanged')}\nall-parameter guard unchanged: {P['verify'].get('guard_unchanged')}\n"
          f"full-weight SHA-256 = reference: {st.get('full_weight_equals_reference')}",ec=INK2,hc=INK2)
    fq=resp(r,"FINAL");qv=b["query_views"].get("FINAL",{})
    block(fig,x0+(W+G),yb,W,H2,"7 · SOURCE-FREE QUESTION ANSWERING","Only the question template is input",
          f"FINAL question: {fq['question'] if fq else '—'}\nmodel input: {fq.get('query_tokens') if fq else '?'} template tokens over {r['memory_rows']} memory rows\n"
          f"full cartridge token sequence in the input: {fq.get('cartridge_sequence_in_query') if fq else '?'}\nraw answer: '{fq['answer'] if fq else ''}' · {STATUS_SHORT.get(fq['status'],'') if fq else ''}",ec=BLUE,mono=False,bfs=7.8,hc=BLUE)
    frame(fig,x0,yb,W,H2,ec=RULE,fc=PANEL)
    fit_text(fig,x0+.008,yb+.01,W-.016,H2-.02,"Two phases\n\nWRITING (stages 1–2): every time a memory is built, the engine reads each fact once (one frozen forward per fact) and keeps only numbers.\n\n"
             "USING (stages 5–7): questions are answered over the numerical memory; the fact tokens are not part of the question input (checked per question).\n\n"
             "Stages 3–4 form and consolidate the memory. TEXT_JOINT (facts read together as text) is a comparison control; DC_CUT3 is the demonstrated path.",fs_max=8.4,fs_min=6.4,color=INK2)
    arrow(fig,x0+3*(W+G)+W/2,ytop-.004,x0+3*(W+G)+W/2,yb+H2+.004)
    arrow(fig,x0+3*(W+G)-.002,yb+H2/2,x0+3*W+2*G+.002,yb+H2/2);arrow(fig,x0+2*(W+G)-.002,yb+H2/2,x0+2*W+G+.002,yb+H2/2)
    footer(fig,P,k,f"Figure {k}. Executed path of {st.get('model_id')} for Demonstration A (fictional domain). (1–2) Each fact is written once as an independent cartridge: the layer-3 residual H3 and the "
           f"layer 0–3 K/V of its tokens. (3) The cartridge is numbers only. (4) Cartridges are appended in order; for each append the engine runs a forward with all-zero input embeddings, "
           f"injects the stored H3 at the layer-3 output and computes layers 4–27 for the new rows over the existing memory; earlier rows are compared bitwise after every append. "
           f"(5–7) The frozen model answers from the {T}-row memory with only the question template as input.")
    return save_jpeg(fig,path,P,k,"Architecture")

# ------------------------------------------------------------------ FIG 2 · cartridge creation and consolidation
def fig2(P,k,path):
    st=P["startup"];fig=new_fig()
    header(fig,k,"Cartridge creation and append-only consolidation, measured","Both demonstrations, condition ABC, DC_CUT3 · every forward recorded by a read-only pre-hook · bitwise checks of earlier rows after each append",
           "LIVE MEASUREMENT",BLUE)
    for n_,scn in enumerate(("AKBASCORE","MATRIX")):
        b=build(P,scn);steps=b["steps"];T=b["memory"]["T"]
        ax=fig.add_axes([.055,.565-n_*.30,.30,.215]);clean(ax)
        for i,s in enumerate(steps):
            old=s["old_len"];add=s["new_len"]-old;col=SEGC.get(s["slot"],BLUE)
            ax.bar(i,old,color="#E5E7EB",edgecolor=INK2,lw=.5,width=.6);ax.bar(i,add,bottom=old,color=col,width=.6)
            ax.text(i,s["new_len"]+T*.02,f"{s['new_len']}",ha="center",fontsize=8,color=INK)
            if s["kind"]=="append":ax.text(i,max(old/2,1),"✓" if s["prior_rows_bitwise_unchanged"] else "✗",ha="center",va="center",fontsize=13,color=TEAL if s["prior_rows_bitwise_unchanged"] else RED,weight="bold")
        ax.set_xticks(range(len(steps)));ax.set_xticklabels([f"step {i+1}\n{s['kind']} {s['slot']}" for i,s in enumerate(steps)],fontsize=7.6)
        ax.set_ylim(0,T*1.18);ax.set_ylabel("memory rows",fontsize=8);ax.set_title(f"{SCENARIO_LABEL[scn].split(' — ')[0]} ({scn}) · memory rows after each step",fontsize=8.4)
    fig.text(.055,.17,"grey = earlier rows · ✓ bitwise unchanged · colour = rows added",fontsize=7.4,color=INK2)
    tx=.40;fig.text(tx,.84,"Per-cartridge record",fontsize=10,weight="bold",color=INK)
    hdr=f"{'scenario':<10}{'slot':<5}{'body':>5}{'rows':>6}{'ckpt H3':>17}{'K/V L0–3':>15}{'bytes':>11}{'write':>9}{'consolidation':>24}{'ms':>7}{'prior':>7}{'L0–3=cart.':>11}"
    fig.text(tx,.81,mt(hdr),fontsize=6.9,family=MONO,color=MUTED)
    y=.785
    for scn in ("AKBASCORE","MATRIX"):
        b=build(P,scn)
        for c,s in zip(b["cartridges"],b["steps"]):
            ids=[f for f in s["forwards"] if f["kind"]=="ids"];emb=[f for f in s["forwards"] if f["kind"]=="embeds"]
            cons=(f"zero-emb {emb[0]['n']}·cache {emb[0]['past_len'] if emb[0]['has_past'] else 0}" if emb else "—")
            pr="init" if s["kind"]=="init" else ("✓" if s["prior_rows_bitwise_unchanged"] else "✗")
            line=(f"{scn[:9]:<10}{c['slot']:<5}{c['body_tokens']:>5}{c['rows_in_memory']:>6}{str(tuple(c['h3_shape'])):>17}{str(tuple(c['kv_lower_shape'])):>15}"
                  f"{c['cartridge_bytes']:>11,}{('ids '+str(ids[0]['n'])) if ids else '—':>9}{cons:>24}{1000*s['seconds']:>7.1f}{pr:>7}{('✓' if s['lower_rows_equal_cartridge'] else '✗'):>11}")
            fig.text(tx,y,mt(line),fontsize=6.9,family=MONO,color=INK);y-=.03
        y-=.012
    frame(fig,tx,.165,.57,.235,ec=TEAL,fc=TEAL_L,lw=1.0)
    a=build(P,"AKBASCORE");m=build(P,"MATRIX")
    def summ(b):return (f"{b['source_reads']} source token-id forwards (one per cartridge) · {b['zero_embed_forwards']} zero-embedding consolidation forwards · "
                        f"prior rows unchanged at every append: {all(s['prior_rows_bitwise_unchanged'] for s in b['steps'][1:])} · step-wise memory = native_memory659(): {b['engine_path_identical']} · "
                        f"{b['memory']['T']} rows · {fmt_bytes(b['memory']['bytes'])} · build {1000*b['build_seconds']:.0f} ms · memory SHA-256 {b['memory']['sha256'][:16]}…")
    fit_text(fig,tx+.008,.172,.554,.22,"Measured in this run\n• AKBASCORE: "+summ(a)+"\n• MATRIX: "+summ(m)+
             "\n• 'L0–3=cart.': the layer 0–3 K/V rows inside the memory equal the cartridge's own K/V (keys re-phased to their memory position), compared bitwise."
             "\n• 'ckpt H3' / 'K/V L0–3': tensor shapes of the cartridge checkpoint (prefix + fact); 'rows' = rows the cartridge adds to the memory (the first one carries the prefix)."
             "\n• bytes = H3 rows × 3584 × 2 B + K/V rows × 4 layers × 2 × 4 heads × 128 × 2 B, from the measured row counts.",fs_max=8.0,fs_min=6.0,color=INK)
    footer(fig,P,k,f"Figure {k}. Writing and consolidation of the three cartridges of each demonstration. The first step (init) writes cartridge A together with the {a['prefix_tokens']}-token instruction prefix; "
           "every further step (append) runs one token-id forward over the instruction prefix plus the new fact only (no cache, no earlier fact) and one forward whose input embeddings are all zero, whose length equals the new rows and which "
           "attends to the existing memory. After each append all earlier K/V rows of all layers are compared bitwise with the memory before the append; the final step-wise memory is compared bitwise with "
           "the memory produced by the engine's native_memory659(), the path used by run_demo.")
    return save_jpeg(fig,path,P,k,"Cartridge consolidation")

# ------------------------------------------------------------------ FIG 3 · internal memory X-ray
def fig3(P,k,path):
    b=build(P,"AKBASCORE");M=b["memory"];T=M["T"];segs=b["segments"];fig=new_fig();NLm=M["layers"]
    header(fig,k,"Internal neural memory X-ray",f"Demonstration A · ABC · DC_CUT3 · tensor per layer {tuple(M['shape'])} ({M['dtype']}) · {NLm} layers × {T} rows · K+V {fmt_bytes(M['bytes'])} ({M['bytes']:,} B) · read from this run's memory",
           "LIVE MEASUREMENT",BLUE)
    for n_,(key,ttl) in enumerate((("k_norm","‖K‖ per row (mean over 4 KV heads)"),("v_norm","‖V‖ per row (mean over 4 KV heads)"))):
        A=np.asarray(M[key],dtype=float);mu=A.mean(axis=1,keepdims=True);sd=A.std(axis=1,keepdims=True);sd[sd==0]=1;Z=(A-mu)/sd
        ax=fig.add_axes([.055,.575-n_*.285,.60,.225])
        im=ax.imshow(Z,aspect="auto",cmap="RdBu_r",vmin=-2.5,vmax=2.5,interpolation="nearest",origin="lower")
        for lab,a_,b_,_ in segs[1:]:ax.axvline(a_-.5,color=INK,lw=.8)
        ax.axhline(CUT+.5,color=AMBER,lw=1.4,ls=(0,(4,2)))
        ax.set_yticks([0,CUT,NLm//2,NLm-1]);ax.set_yticklabels(["L0",f"L{CUT}",f"L{NLm//2}",f"L{NLm-1}"],fontsize=7)
        ax.set_title(mt(ttl+" · z-scored within each layer (display normalisation)"),fontsize=8.3);ax.set_xlim(-.5,T-.5);ax.set_xticks([])
        cb=fig.colorbar(im,cax=fig.add_axes([.66,.575-n_*.285,.007,.225]));cb.ax.tick_params(labelsize=6.5);cb.outline.set_linewidth(.5);cb.set_label("z",fontsize=7)
    fig.text(.055,.198,mt(f"dashed line: layers 0–{CUT} below = cartridge rows (written per fact) · layers {CUT+1}–{NLm-1} above = computed during consolidation"),fontsize=7.2,color=AMBER,ha="left",va="top",weight="bold")
    sx=fig.add_axes([.055,.24,.60,.03]);sx.set_xlim(-.5,T-.5);sx.set_ylim(0,1);sx.axis("off")
    for lab,a_,b_,_ in segs:
        sx.add_patch(Rectangle((a_-.5,0),b_-a_,1,color=SEGC.get(lab,"#64748B"),lw=0));sx.text((a_+b_-1)/2,.5,lab if lab!="prefix" else f"prefix ({b_-a_})",ha="center",va="center",fontsize=7.4,color="white",weight="bold")
    fig.text(.055,.228,mt("row layout: "+" · ".join(f"{lab} rows {a_}–{b_-1}" for lab,a_,b_,_ in segs)),fontsize=7.2,family=MONO,color=INK2,va="top")
    jd=b["joint_difference"];ax=fig.add_axes([.735,.40,.235,.38]);clean(ax)
    L=np.arange(NLm)
    for j,lab in enumerate(jd["labels"]):
        ys=np.asarray([row[j] for row in jd["k"]]);ax.plot(L,ys,color=SEGC.get(lab,"#64748B"),lw=1.4,label=f"{lab} (K)")
        ys=np.asarray([row[j] for row in jd["v"]]);ax.plot(L,ys,color=SEGC.get(lab,"#64748B"),lw=1.0,ls="--")
    ax.axvline(CUT+.5,color=AMBER,lw=1,ls=(0,(4,2)));ax.set_xlabel("layer",fontsize=8);ax.set_ylabel("‖DC − TEXT_JOINT‖ / ‖TEXT_JOINT‖",fontsize=7.6)
    ax.set_title("DC_CUT3 memory vs TEXT_JOINT control",fontsize=8.3);ax.legend(fontsize=6.6,loc="upper right",ncol=2)
    eq=", ".join(f"{l}: {'bitwise equal' if e else 'differs'}" for l,e in zip(jd["labels"],jd["bitwise_equal_segments"]))
    fit_text(fig,.735,.165,.235,.17,f"solid = K, dashed = V, per memory segment.\n{eq}.\nA cartridge written alone differs from the same fact read together with the earlier facts; this curve measures that difference layer by layer.",fs_max=7.6,fs_min=6.0,color=INK2)
    footer(fig,P,k,f"Figure {k}. The consolidated memory of Demonstration A after the third cartridge. Each column is one token row, each heat-map row one layer; the colour is the row's mean K (top) or V (bottom) "
           f"L2 norm, z-scored within each layer for display only. Vertical lines mark the prefix and cartridges A, B, C. Rows of layers 0–{CUT} are the cartridges' own K/V; layers {CUT+1}–{NLm-1} were computed by the engine during "
           "consolidation. Right: relative difference between this memory and the TEXT_JOINT control memory built from the same facts as one text, per layer and segment. Values are read from GPU memory of this run.")
    return save_jpeg(fig,path,P,k,"Internal memory X-ray")

# ------------------------------------------------------------------ FIG 4 / FIG 5 · experiments
MAXNEW_TXT=16
def qa_table(fig,r,x,y,w,rowh,fs=7.6,show_tokens=True):
    for q in r["responses"]:
        ok=q.get("status","UNSCORED");tw=w-.175
        fig.text(x,y,q["question_id"],fontsize=7.2,weight="bold",color=MUTED,va="top")
        fit_text(fig,x+.075,y-rowh*.5,tw,rowh*.5-.002,q["question"],fs_max=fs,fs_min=5.8,color=INK2)
        fit_text(fig,x+.075,y-rowh+.008,tw,rowh*.5-.012,"→ '"+q["answer"]+"'",fs_max=fs+1.0,fs_min=6.0,color=INK,family=MONO,weight="bold")
        chip(fig,x+w-.088,y-.026,ok,w=.085)
        if show_tokens and q.get("generated_tokens") is not None:
            fit_line(fig,x+w-.088,y-.036,f"{q.get('generated_tokens')} new tokens",.088,6.2,5.0,color=MUTED,va="top")
            if q.get("limit_reached"):fit_line(fig,x+w-.088,y-.052,f"{MAXNEW_TXT}-token limit reached",.088,6.2,5.0,color=RED,va="top",weight="bold")
        fig.add_artist(Line2D([x,x+w],[y-rowh+.003,y-rowh+.003],transform=fig.transFigure,color=GRIDC,lw=.5))
        y-=rowh
    return y
def chain_column(fig,r,sc,x,y,w,h,start):
    staged=r["staged"];n=len(staged["trace"]);bh=min(.085,(h-.02)/(n+1)-.022);yy=y
    frame(fig,x,yy-bh,w,bh,ec=INK2,fc=PANEL);fig.text(x+w/2,yy-bh/2,mt(start),ha="center",va="center",fontsize=9.6,weight="bold",color=INK)
    for t in staged["trace"]:
        yy2=yy-bh-.03;arrow(fig,x+w/2,yy-bh,x+w/2,yy2+.002,color=INK2)
        ec,fc=STATUS_COL.get(t.get("status","UNSCORED"));frame(fig,x,yy2-bh,w,bh,ec=ec,fc=fc,lw=1.1)
        fig.text(x+.006,yy2-.012,f"HOP {t['hop']} · {STATUS_SHORT.get(t.get('status'),'')}",fontsize=6.8,weight="bold",color=ec,va="center")
        fig.text(x+w/2,yy2-bh/2-.006,mt(short("'"+t["answer"]+"'",34)),ha="center",va="center",fontsize=9.6,weight="bold",color=INK,family=MONO)
        fit_text(fig,x+.006,yy2-bh+.003,w-.012,.026,short(t["question"],105),fs_max=6.2,fs_min=5.0,color=INK2)
        yy=yy2
    return yy-bh
def experiment_fig(P,k,path,scn):
    global MAXNEW_TXT
    st=P["startup"];MAXNEW_TXT=st.get("max_new");r=prim(P,scn);sc=P["scenarios"][scn];b=build(P,scn);fig=new_fig();spec=SCORING[scn]
    tag="DEMONSTRATION A" if scn=="AKBASCORE" else "DEMONSTRATION B"
    title=("Mustafa Akbaş experiment — three cartridges, direct questions and a staged three-step chain" if scn=="AKBASCORE"
           else "The Matrix / Neo experiment — what goes into memory and what the frozen model answers")
    header(fig,k,title,f"{SCENARIO_LABEL[scn]} · fictional knowledge domain · condition ABC · DC_CUT3 · raw model outputs, unedited · scored: {sc['scored']}",tag,BLUE)
    # cartridges
    fig.text(.03,.835,"CARTRIDGE TEXTS (source facts, read once per memory build)",fontsize=7.8,weight="bold",color=INK2)
    yy=.81;ch=.135
    for c in b["cartridges"]:
        col=SEGC.get(c["slot"],BLUE);frame(fig,.03,yy-ch,.27,ch,ec=col,fc="white",lw=1.1)
        fig.text(.036,yy-.014,mt(f"{SLOT_LABEL[scn][c['slot']]} · {c['body_tokens']} tokens · {fmt_bytes(c['cartridge_bytes'])} numerical"),fontsize=7.2,weight="bold",color=col,va="center")
        fit_text(fig,.036,yy-ch+.005,.258,ch-.03,c["text"],fs_max=8.4,fs_min=6.2,color=INK);yy-=ch+.012
    fit_text(fig,.03,.165,.27,.05,f"Counterfactual C (only in ABC_CF, see Fig. 6): {sc['texts']['C_CF']}",fs_max=7.0,fs_min=5.8,color=MUTED)
    # direct questions
    x=.32;w=.385;fig.text(x,.835,"DIRECT QUESTIONS — one question each, answered over the memory",fontsize=7.8,weight="bold",color=INK2)
    n=len(r["responses"]);rowh=min(.13,.645/n);qa_table(fig,r,x,.81,w,rowh)
    # chain
    cx=.725;cw=.245;fig.text(cx,.835,"STAGED CHAIN — predefined external question sequence",fontsize=7.8,weight="bold",color=INK2)
    start=spec["chain_text"][0];endy=chain_column(fig,r,spec,cx,.81,cw,.50,start)
    s_=r["staged"];chain_st=s_.get("chain_status","UNSCORED")
    fit_line(fig,cx,endy-.022,f"chain result '{s_['answer']}' · {STATUS_SHORT.get(chain_st)} · completed {s_['completed']}",cw,7.6,5.5,weight="bold",color=STATUS_COL[chain_st][0])
    fit_text(fig,cx,.165,cw,max(.02,endy-.04-.165),"written chain (cartridge texts): "+" → ".join(spec["chain_text"]),fs_max=7.2,fs_min=5.8,color=MUTED)
    if scn=="AKBASCORE":
        cap=(f"Figure {k}. Demonstration A, a fictional domain (no real-world claim is made, in particular none about time). Left: the three cartridge texts, each written once. Middle: direct questions with the "
             f"raw answers of the frozen model ({st.get('max_new')} new tokens at most) and their audit status (expected answers are used only after generation). Right: the staged chain: three predefined questions "
             "asked in sequence, each one inserting the previous raw answer; every step is answered by the model from the numerical memory. This is an externally specified sequence, not autonomous planning.")
    else:
        cap=(f"Figure {k}. Demonstration B, a fictional scenario. What was written into memory (left), what the frozen model answered to direct questions (middle) and the staged three-step chain (right). "
             f"The cartridges enter the model as numbers: H3 at the layer-3 output and K/V of layers 0–3; the model weights are not changed (sentinel before = after: {P['verify'].get('sentinel_unchanged')}). "
             "Note: cartridge C itself names MX-71 and the Kung Fu knowledge, so the FINAL question is supported by A and C without B (see Fig. 6).")
    footer(fig,P,k,cap)
    return save_jpeg(fig,path,P,k,"Experiment "+scn)
def fig4(P,k,path):return experiment_fig(P,k,path,"AKBASCORE")
def fig5(P,k,path):return experiment_fig(P,k,path,"MATRIX")

# ------------------------------------------------------------------ FIG 6 · causal controls
def fig6(P,k,path):
    FH=8.6;fig=new_fig(FH);s=FH/FIG_H
    header(fig,k,"Causal controls and the staged multi-hop chain","Nine memory conditions per demonstration · DC_CUT3 grid (cell = raw answer, colour = audit status) · TEXT_JOINT control summarised on the right · TEST677 shown separately as a sealed record",
           "CAUSAL CONTROLS",ORANGE,h=FH)
    def U(y):return y/s if False else y
    gx=.03;gw=.62;top=.865
    for n_,scn in enumerate(("AKBASCORE","MATRIX")):
        cm=controls_matrix(P,scn,"DC_CUT3");cols=[q for q,_ in P["scenarios"][scn]["queries"] if q!="SYNTHESIS"]+["CHAIN"]
        y0=top-n_*.37;fig.text(gx,y0,mt(f"{SCENARIO_LABEL[scn]} · DC_CUT3 · scored: {P['scenarios'][scn]['scored']}"),fontsize=8.6,weight="bold",color=INK)
        cw=(gw-.07)/len(cols);rh=.03
        for j,c in enumerate(cols):fig.text(gx+.07+j*cw+cw/2,y0-.022,c,fontsize=6.8,weight="bold",color=MUTED,ha="center",va="center")
        for i,cond in enumerate(CONDITIONS):
            yy=y0-.04-(i+1)*rh;fig.text(gx,yy+rh/2,cond,fontsize=7.2,weight="bold",color=INK2,va="center")
            r=cm.get(cond)
            for j,c in enumerate(cols):
                if r is None:st_,txt="UNSCORED","NOT MEASURED"
                elif c=="CHAIN":st_=r["staged"]["chain_status"] if r["staged"] else "UNSCORED";txt=r["staged"]["answer"] if r["staged"] else "—"
                else:
                    q=resp(r,c);st_=q["status"] if q else "UNSCORED";txt=q["answer"] if q else "—"
                ec,fc=STATUS_COL.get(st_);x=gx+.07+j*cw
                fig.add_artist(Rectangle((x+.001,yy+.002),cw-.002,rh-.004,transform=fig.transFigure,facecolor=fc,edgecolor=ec,lw=.6))
                fig.text(x+cw/2,yy+rh/2,mt(short(txt,int(cw*190))),fontsize=6.0,color=INK,ha="center",va="center",family=MONO)
    lx=.67;lw_=.30;ABBR=dict(CORRECT_KNOWN="CK",CORRECT_ABSTENTION="CA",UNSUPPORTED="US",MISSED_KNOWN="MK",WRONG_KNOWN="WK",UNSCORED="UN")
    fig.text(lx,.865,"Status counts (all questions + chain, nine conditions)",fontsize=8.4,weight="bold",color=INK)
    y=.84
    for rep in ("DC_CUT3","TEXT_JOINT"):
        sc=status_counts(P,rep)
        for scn in ("AKBASCORE","MATRIX"):
            c=sc.get(scn,{});tot=sum(c.values())
            line=f"{rep:<10} {scn:<9} "+("NOT MEASURED" if tot==0 else " · ".join(f"{ABBR[s_]} {c.get(s_,0)}" for s_ in STATUS_ORDER if c.get(s_,0)))
            fit_line(fig,lx,y-.012,line,lw_,6.9,5.0,family=MONO,color=INK2,va="center");y-=.025
    # agreement DC vs TEXT
    agree=tot_=0
    for scn in ("AKBASCORE","MATRIX"):
        a=controls_matrix(P,scn,"DC_CUT3");b=controls_matrix(P,scn,"TEXT_JOINT")
        for cond in CONDITIONS:
            if cond in a and cond in b:
                for q in a[cond]["responses"]:
                    qb=resp(b[cond],q["question_id"])
                    if qb:tot_+=1;agree+=int(q["answer"].strip()==qb["answer"].strip())
    fit_line(fig,lx,y-.01,"CK correct · CA correct UNKNOWN · US unsupported · MK missed · WK wrong · UN unscored",lw_,6.6,5.0,color=MUTED);y-=.025
    fig.text(lx,y-.006,mt(f"DC_CUT3 vs TEXT_JOINT identical raw answers: {agree}/{tot_}" if tot_ else "TEXT_JOINT control: NOT MEASURED"),fontsize=7.2,color=INK2);y-=.04
    # counterfactual effect
    for scn in ("AKBASCORE","MATRIX"):
        cm=controls_matrix(P,scn,"DC_CUT3");abc=cm.get("ABC");cf=cm.get("ABC_CF")
        if abc and cf:
            fa=resp(abc,"FINAL");fc_=resp(cf,"FINAL")
            txt=f"{scn}: FINAL with ABC '{fa['answer'] if fa else '—'}' → with ABC_CF '{fc_['answer'] if fc_ else '—'}' · changed {bool(fa and fc_ and norm_ans(fa['answer'])!=norm_ans(fc_['answer']))}"
        else:txt=f"{scn}: counterfactual NOT MEASURED"
        fit_text(fig,lx,y-.04,lw_,.04,txt,fs_max=7.0,fs_min=5.8,color=INK);y-=.045
    # leakage signature
    leak=[]
    for r in P["controls"]:
        if r["representation"]!="DC_CUT3":continue
        for q in r["responses"]:
            if q["status"]=="UNSUPPORTED" and q.get("equals_full_memory_answer"):leak.append(f"{r['scenario'][:4]} {r['condition']} {q['question_id']}")
    fit_text(fig,lx,y-.07,lw_,.07,"Unsupported answers equal to the full-memory answer (a cartridge needed for the question was absent): "+(", ".join(leak) if leak else "none"),fs_max=6.8,fs_min=5.6,color=ORANGE);y-=.08
    # TEST677 sealed
    frame(fig,lx,.165,lw_,y-.175,ec=INK2,fc=PANEL,ls="--")
    t=TEST677_REF;lv=P["references"].get("test677_log") or {}
    ref="\n".join(f"{c:<7}{x}/{y_}/{z}  AB '{a}' · BC '{b_}' · ABC '{c_}'" for c,a,b_,c_,x,y_,z in t["user_panel_dc"])
    vtxt={True:"yes",False:"NO — mismatch",None:"not checked (offline)"}.get(lv.get("matches"),str(lv.get("matches")))
    fit_text(fig,lx+.006,.17,lw_-.012,y-.19,f"SEALED RECORD · TEST677 (not run here) · 677.log SHA-256 verified this session: {vtxt}\nDC_CUT3 user panel (same A/B/C texts as Demonstration A; its counterfactual was a random AltEffect token)\n"
             f"cond   status    CHAIN answers\n{ref}\nCK correct · CA correct UNKNOWN · US unsupported\nblind panels CHAIN_ABC DC_CUT3: "+", ".join(f"{c} {v}" for c,v in t["blind_chain_abc_dc"]),fs_max=6.6,fs_min=5.2,family=MONO,color=INK2)
    lg=fig.add_axes([.03,.142,.62,.02]);lg.axis("off")
    for i,s_ in enumerate(STATUS_ORDER):
        ec,fc=STATUS_COL[s_];lg.add_patch(Rectangle((i/6+.005,0.1),.03,.8,facecolor=fc,edgecolor=ec,transform=lg.transAxes));lg.text(i/6+.04,.5,STATUS_SHORT[s_],fontsize=7,va="center",transform=lg.transAxes)
    footer(fig,P,k,f"Figure {k}. Every cell is the unedited answer of the frozen model with the memory built from the listed cartridges (NONE = instruction prefix only; ABC_CF = A, B and the counterfactual C). "
           "Status (TEST677 rule): correct = expected answer present in memory and produced; correct UNKNOWN = a required cartridge is absent and the model answers UNKNOWN; unsupported = a required cartridge is absent but the "
           "model still answers; missed = answer available but UNKNOWN; wrong = a different answer. CHAIN = result of the predefined three-step question sequence. TEST677 values come from its published log and were not recomputed.",h=FH)
    return save_jpeg(fig,path,P,k,"Causal controls")

# ------------------------------------------------------------------ FIG 7 · verification record
def fig7(P,k,path):
    st=P["startup"];v=P["verify"];fig=new_fig()
    header(fig,k,"Verification and research record","Configuration, engine identity, integrity measurements, file hashes, references and the claim boundary of this run","VERIFICATION · RECORD",INK2)
    cw=.30;gap=.02;y=.18;h=.67
    cfg=(f"run              {P['run_id']}\nstarted          {P['started_local']}\nstarted (UTC)    {P['started_utc']}\nfinished (UTC)   {P.get('finished_utc')}\nduration         {P.get('elapsed_seconds')} s\n"
         f"model            {st.get('model_id')}\narchitecture     {arch_line(st)}\nweights          {st.get('dtype')} · frozen · {st.get('attn')}\n")
    if st.get("params"):cfg+=f"parameters       {st['params']:,}\n"
    cfg+=(f"GPU              {st.get('gpu')}\ntorch            {st.get('torch')}\ntransformers     {st.get('transformers')}\ngradio           {st.get('gradio')}\npython           {st.get('python')}\n"
          f"injection        native DC_CUT3: H3 at layer {st.get('cut')} + K/V L0–{st.get('cut')}\nprefix           {st.get('prefix_tokens')} tokens\nmax new tokens   {st.get('max_new')}\n"
          f"v1.4 engine      SHA-256\n  {st.get('v14_sha256')}\nTEST664 source   SHA-256\n  {st.get('source_sha256')}\nTEST664 panel    SHA-256\n  {st.get('panel_sha256')}\nTEST566 git blob {st.get('test566_blob')}")
    frame(fig,.03,y,cw,h,ec=RULE);fig.text(.04,y+h-.025,"CONFIGURATION AND ENGINE",fontsize=8.6,weight="bold",color=INK2)
    fit_text(fig,.04,y+.01,cw-.02,h-.05,cfg,fs_max=7.3,fs_min=5.6,family=MONO,color=INK)
    fz=v.get("frozen_after") or {}
    ver=(f"full-weight SHA-256 (start-up)\n  {st.get('full_weight_sha256')}\n  = TEST575 reference: {st.get('full_weight_equals_reference')}\n"
         f"sentinel load = before = after   {v.get('sentinel_unchanged')}\n  {str(v.get('sentinel_after'))[:40]}…\nall-parameter guard start = before = after   {v.get('guard_unchanged')}\n  {str(v.get('guard_after'))[:40]}…\n"
         f"trainable tensors   {fz.get('trainable_tensors')}\nLoRA                {fz.get('lora')}\nresidual hooks      {len(fz.get('residual_hooks',[]))}\nall engine runs frozen + sentinel   {v.get('all_runs_frozen')}\n\n")
    for scn in ("AKBASCORE","MATRIX"):
        b=P["builds"][scn]
        ver+=(f"{scn} memory (ABC, DC_CUT3)\n  {b['memory']['T']} rows · {b['memory']['bytes']:,} B\n  SHA-256 {b['memory']['sha256'][:32]}…\n"
              f"  prior rows unchanged: {all(s_['prior_rows_bitwise_unchanged'] for s_ in b['steps'][1:])} · L0–3 = cartridge: {all(s_['lower_rows_equal_cartridge'] for s_ in b['steps'])}\n"
              f"  step-wise = native_memory659(): {b['engine_path_identical']}\n  cartridges: "+", ".join(f"{c['slot']} {c['cartridge_bytes']:,} B" for c in b["cartridges"])+"\n")
    frame(fig,.03+cw+gap,y,cw,h,ec=RULE);fig.text(.04+cw+gap,y+h-.025,"VERIFICATION MEASURED IN THIS RUN",fontsize=8.6,weight="bold",color=INK2)
    fit_text(fig,.04+cw+gap,y+.01,cw-.02,h-.05,ver,fs_max=7.3,fs_min=5.6,family=MONO,color=INK)
    runs=[r for r in P["controls"] if r["condition"]=="ABC"]+([P["custom"]] if P["custom"] else [])
    rec="Engine run records (v1.4 run_sha256 / file SHA-256)\n"+"\n".join(f"{r['scenario'][:4]} {r['condition']} {'DC' if r['representation']=='DC_CUT3' else 'TEXT'}  {str(r['run_sha256'])[:20]}… / {str(r.get('engine_json_sha256'))[:12]}…" for r in runs)
    rec+=f"\n({len(P['controls'])} engine runs in total; all hashes in SHA256_MANIFEST.json)\n\nReferences\n• TEST664 {URL_664}\n• TEST677 {URL_677}\n  log SHA-256 {TEST677_LOG_SHA256[:24]}… verified now: {({True:'yes',False:'NO',None:'not checked'}).get((P['references'].get('test677_log') or {}).get('matches'),'?')}\n  result SHA-256 {TEST677_REF['result_sha256'][:24]}…\n"
    rec+=f"• earlier v1.4 Colab run {EARLIER_V14_LOG['date']} (reference only)\n\nScientific boundaries\n• fictional / synthetic scenarios\n• staged chain = predefined external question sequence, not autonomous planning\n• {st.get('max_new')}-token generation limit; free-text synthesis unscored\n• abstention with missing cartridges is measured, not guaranteed\n• hashes show file identity, not correctness\n• one model, two small scenarios; no scale or generalisation claim\n• {AUTHOR} · {ORG} · {PLACE}"
    frame(fig,.03+2*(cw+gap),y,cw,h,ec=RULE);fig.text(.04+2*(cw+gap),y+h-.025,"RECORDS · REFERENCES · BOUNDARIES",fontsize=8.6,weight="bold",color=INK2)
    fit_text(fig,.04+2*(cw+gap),y+.01,cw-.02,h-.05,rec,fs_max=7.3,fs_min=5.6,family=MONO,color=INK)
    footer(fig,P,k,f"Figure {k}. Identity card of run {P['run_id']}. The embedded v1.4 engine source and the downloaded TEST664 source are SHA-256 checked before execution; the model weights are checked with the TEST566 sentinel, "
           "an all-parameter guard and (at start-up) the TEST575 full-weight SHA-256. Memory integrity values come from the measured builds of Fig. 2–3. File hashes identify files; they are not evidence of correctness.")
    return save_jpeg(fig,path,P,k,"Verification")

FIGURES=[fig1,fig2,fig3,fig4,fig5,fig6,fig7]
def render_all(P,run_dir,cb=None):
    out=[]
    try:
        for i,(fn,name) in enumerate(zip(FIGURES,FIG_FILES),1):
            out.append(fn(P,i,Path(run_dir)/name))
            if cb:cb(i)
    finally:plt.close("all")
    return out


# =====================================================================================================================
# GRADIO INTERFACE — calls the engine, records, visualises. It never computes or edits an answer.
# =====================================================================================================================
def esc(x):return html.escape(str(x))
def card_html(title,body,kind="info"):return f'<div class="kz card {kind}"><div class="h">{esc(title)}</div><div class="small">{body}</div></div>'
def chip_html(status):
    ec,fc=STATUS_COL.get(status,(MUTED,"#F3F4F6"))
    return f'<span class="chip" style="color:{ec};background:{fc};border-color:{ec}">{esc(STATUS_SHORT.get(status,status))}</span>'

CSS="""
:root{--kz-on:#0f766e;--kz-fg:#111827;--kz-card:#ffffff;--kz-bd:#d1d5db;--kz-a:#1d4ed8;--kz-err:#b91c1c;--kz-warn:#b45309}
.dark{--kz-on:#2dd4bf;--kz-fg:#f3f4f6;--kz-card:#111827;--kz-bd:#4b5563;--kz-a:#60a5fa;--kz-err:#f87171;--kz-warn:#fbbf24}
html,body{overflow-x:hidden!important}
.gradio-container{width:100%!important;max-width:960px!important;margin:0 auto!important;padding-left:10px!important;padding-right:10px!important;box-sizing:border-box!important}
.gradio-container *{box-sizing:border-box}
.kz{color:var(--kz-fg)!important;max-width:100%;overflow-wrap:anywhere;line-height:1.45}
.kz *{color:inherit}
.hero{background:#0b1220;border-radius:6px;padding:18px 18px 14px;margin:4px 0 8px}
.hero,.hero *{color:#f9fafb!important}
.rec{font-size:11.5px;letter-spacing:1.6px;text-transform:uppercase;opacity:.85;border:1px solid rgba(255,255,255,.35);display:inline-block;padding:2px 8px;border-radius:3px}
.brand{font-size:clamp(20px,5.6vw,30px);font-weight:800;letter-spacing:.4px;line-height:1.15;margin-top:10px}
.title{font-size:clamp(14px,3.8vw,18px);font-weight:600;margin-top:6px;color:#fde68a!important}
.by{font-size:13.5px;margin-top:10px}
.msg{margin:8px 0 0;padding-left:18px;font-size:13.5px}
.lic{font-size:12px;opacity:.85;margin-top:8px}
.card{background:var(--kz-card);border:1px solid var(--kz-bd);border-left:4px solid var(--kz-bd);border-radius:4px;padding:10px 12px;margin:6px 0}
.card.on{border-left-color:var(--kz-on)}.card.err{border-left-color:var(--kz-err)}.card.warn{border-left-color:var(--kz-warn)}.card.info{border-left-color:var(--kz-a)}
.card .h{font-weight:700;font-size:15px;margin-bottom:4px}
.small{font-size:13.5px}
.mono,.kz code{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px}
.chip{display:inline-block;border:1px solid;border-radius:3px;padding:0 6px;font-size:12px;font-weight:700;white-space:nowrap}
.tbl{width:100%;border-collapse:collapse;font-size:13px}
.tbl td,.tbl th{border-bottom:1px solid var(--kz-bd);padding:5px 4px;vertical-align:top;text-align:left}
.tbl th{font-size:12px;opacity:.75}
.stg{font-size:13.5px;margin:2px 0}.stg.done{color:var(--kz-on)!important}.stg.run{color:var(--kz-a)!important;font-weight:700}.stg.wait{opacity:.55}.stg.fail{color:var(--kz-err)!important;font-weight:700}
#kz_full button,#kz_full{font-size:clamp(15px,4.2vw,18px)!important;font-weight:700!important;min-height:52px!important}
@media (max-width:640px){.tbl .hide-m{display:none}}
"""
HERO=(f'<div class="kz hero"><span class="rec">Interactive research demonstration · frozen model · live measurement</span>'
      f'<div class="brand">AKBASCORE AI TECHNOLOGY<br>MAM SILICON HIPPOCAMPUS v1.4</div><div class="title">{esc(APP_SUB)}</div>'
      '<ul class="msg">'
      '<li>When a memory is built, each fact is read once and written into a <b>numerical cartridge</b>: the layer-3 hidden state (H3) and the attention keys/values of layers 0–3.</li>'
      '<li>Cartridges are <b>consolidated</b> into the attention memory of the frozen Qwen2.5-7B-Instruct (native CUT3 injection, CUDA K/V copy). The weights are never changed.</li>'
      '<li>Questions are answered <b>without the source text</b>, from the memory alone. The three-step chain is a <b>predefined external question sequence</b>, not autonomous planning.</li>'
      '<li><b>DC_CUT3</b> is the demonstrated injection path. <b>TEXT_JOINT</b> (facts read together as text) is a comparison control.</li>'
      '<li>Both scenarios are <b>fictional</b>. Time does not reverse; the Matrix is a film. Answers are capped at <b>16 new tokens</b>; free-text synthesis can be truncated.</li></ul>'
      f'<div class="by"><b>{esc(AUTHOR)}</b> · {esc(ORG)} · {esc(PLACE)}</div>'
      f'<div class="lic">{esc(LICENSE_NAME)} · {esc(COPYRIGHT)} · <a href="{REPO}" target="_blank" rel="noopener">{esc(REPO)}</a></div></div>')
SCOPE_HTML=card_html("What this demo shows — and what it does not",
    "Shown: the unchanged v1.4/TEST664 engine writes, consolidates and reads numerical memory; every answer is the raw output of the frozen model; every figure value is "
    "measured in the run that produced it. Not shown: reliable abstention when cartridges are missing (Fig. 6 shows how often the model answers anyway), free-form synthesis "
    "beyond 16 tokens, autonomous multi-hop reasoning, generalisation beyond two small fictional scenarios. Two successful demonstrations do not mean the controls pass.","warn")

def engine_card_html():
    s=STARTUP;fz=s.get("frozen") or {}
    rows=[("Model",f"{s.get('model_id')} · {s.get('dtype')} · {s.get('attn')}"),("Architecture",arch_line(s)),("GPU",f"{s.get('gpu')} · {s.get('gpu_total_gib')} GiB"),
          ("Software",f"torch {s.get('torch')} · transformers {s.get('transformers')} · gradio {s.get('gradio')} · python {s.get('python')}"),
          ("v1.4 engine",f"SHA-256 {s.get('v14_sha256')} · {s.get('v14_lines')} lines · verified"),("TEST664 source",f"SHA-256 {s.get('source_sha256')}"),
          ("TEST664 panel",f"SHA-256 {s.get('panel_sha256')}"),("Injection",f"DC_CUT3 · H3 at layer {s.get('cut')} · K/V layers 0–{s.get('cut')} · {s.get('native_kernel')}"),
          ("Frozen weights",f"trainable tensors {fz.get('trainable_tensors')} · sentinel = load {fz.get('sentinel_equals_load')} · full-weight SHA-256 = TEST575 reference {s.get('full_weight_equals_reference')}"),
          ("Generation",f"greedy · at most {s.get('max_new')} new tokens · prefix {s.get('prefix_tokens')} tokens"),("Engine ready",f"{s.get('engine_loaded_at')} · boot {s.get('engine_seconds')} s")]
    return card_html("Engine (measured at start-up in this session)",'<table class="tbl">'+"".join(f"<tr><th>{esc(a)}</th><td class='mono'>{esc(b)}</td></tr>" for a,b in rows)+"</table>","on")

def scenario_values(scn):
    d=E["MAM14_EDITED"][scn];ed=scenario_edited(scn)
    info=card_html(SCENARIO_LABEL[scn],("<b>Edited cartridges</b> — answers are shown raw and are <b>not scored</b> (the audit references belong to the original texts)." if ed else
                   "Original cartridge texts (scored against the engine's audit references after generation).")+"<br>Questions: "+
                   "".join(f"<br><span class='mono'>{esc(q)}</span> {esc(t)}" for q,t in d["queries"])+"<br>Chain (asked one after another, previous raw answer inserted): "+
                   "".join(f"<br><span class='mono'>HOP {i+1}</span> {esc(t)}" for i,t in enumerate(d["hop_prompts"])),"warn" if ed else "info")
    return d["A"],d["B"],d["C"],d["C_CF"],info

def run_html(r,title):
    rows=""
    for q in r["responses"]:
        tok="—" if q.get("generated_tokens") is None else f"{q['generated_tokens']}"+(" <b style='color:#b91c1c'>limit</b>" if q.get("limit_reached") else "")
        rows+=(f"<tr><td class='mono'>{esc(q['question_id'])}</td><td>{esc(q['question'])}</td><td><code>{esc(q['answer'])}</code></td><td>{chip_html(q['status'])}</td>"
               f"<td class='hide-m mono'>{tok}</td><td class='hide-m mono'>{esc(q.get('elapsed_seconds'))} s</td><td class='hide-m small' style='opacity:.7'>{esc(q.get('expected') or '—')}</td></tr>")
    t=(f"<table class='tbl'><tr><th>ID</th><th>question</th><th>raw model answer</th><th>audit</th><th class='hide-m'>new tokens</th><th class='hide-m'>time</th>"
       f"<th class='hide-m'>audit reference (not model output)</th></tr>{rows}</table>")
    s=r.get("staged")
    if s:
        hops="".join(f"<br><span class='mono'>HOP {h['hop']}</span> {esc(h['question'])}<br>&nbsp;&nbsp;→ <code>{esc(h['answer'])}</code> {chip_html(h['status'])}"
                     +(f" <span class='small' style='opacity:.7'>({esc(h['note'])})</span>" if h.get("note") else "") for h in s["trace"])
        t+=f"<div class='small' style='margin-top:8px'><b>Staged chain</b> (predefined external sequence){hops}<br><b>chain result</b> <code>{esc(s['answer'])}</code> {chip_html(s['chain_status'])} · completed {esc(s['completed'])}</div>"
    t+=(f"<div class='small mono' style='margin-top:8px;opacity:.8'>memory {r['memory_rows']} rows · cartridges {esc(r['condition'])} · {esc(r['representation'])} · scored {r['scored']} · "
        f"engine {r['engine_elapsed_seconds']} s · weights frozen {r['weights_frozen']} · sentinel unchanged {r['sentinel_unchanged']} · forward segmentation {r['forward_segmentation_ok']}<br>"
        f"engine run SHA-256 {esc(r['run_sha256'])}<br>engine record {esc(r['engine_json'])}</div>")
    return card_html(title,t,"on")

GPU_LOCK=threading.Lock()
def busy_card():return card_html("Busy","Another run is using the GPU. Please wait until it has finished and press the button again.","warn")
def err_card(ex,stage=None):
    return card_html("Error"+(f" in stage {stage}" if stage else ""),f"<code>{esc(type(ex).__name__)}: {esc(ex)}</code><br>Nothing was hidden or replaced; the full traceback is in the log box below.","err")

def apply_handler(scn,a,b,c,ccf):
    try:
        cur=E["MAM14_EDITED"][scn]
        for slot,txt in (("A",a),("B",b),("C",c),("C_CF",ccf)):
            if (txt or "").strip()!=cur[slot]:E["set_cartridge"](slot,txt,scn)
        return scenario_values(scn)
    except Exception as ex:
        v=scenario_values(scn);return v[:4]+(err_card(ex),)
def reset_handler(scn):
    E["reset_scenario"](scn);return scenario_values(scn)

def single_handler(scn,cond,rep,q,staged):
    if not GPU_LOCK.acquire(blocking=False):yield busy_card(),"","",None;return
    try:
        yield card_html("Running",f"{esc(scn)} · {esc(cond)} · {esc(rep)} — building the memory and asking the frozen model …","info"),"","",None
        q=(q or "").strip() or None
        r=recorded_run(scn,cond,rep,question=q[:2000] if q else None,include_staged=bool(staged))
        d=ROOT/"single_runs";d.mkdir(exist_ok=True)
        p=dump_json(d/f"SINGLE_{scn}_{cond}_{rep}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json",r)
        yield run_html(r,f"{SCENARIO_LABEL[scn]} · {cond} · {rep}"),json.dumps(jsafe(r),ensure_ascii=False,indent=1)[:200000],"",str(p)
    except Exception as ex:
        yield err_card(ex),"",traceback.format_exc(),None
    finally:GPU_LOCK.release()

def stages_html(done,cur,msgs,t0,failed=None):
    out=[]
    for i,t in enumerate(STAGE_TITLES,1):
        if failed==i:cl,mk="fail","✗"
        elif i in done:cl,mk="done","✓"
        elif i==cur:cl,mk="run","…"
        else:cl,mk="wait","○"
        m=f" — <span class='small'>{esc(msgs.get(i,''))}</span>" if msgs.get(i) else ""
        out.append(f"<div class='stg {cl}'>{mk} {i}/{N_STAGES} · {esc(t)}{m}</div>")
    pc=100.0*len(done)/N_STAGES
    bar=f'<div style="margin-top:6px;background:#e5e7eb;height:10px"><div style="width:{pc:.0f}%;height:10px;background:#1d4ed8"></div></div>'
    return card_html(f"Full research run · {time.perf_counter()-t0:.0f} s","".join(out)+bar,"err" if failed else "info")

def full_handler(scn,cond,rep,q,text_ctl):
    NO=(None,None,None,"","")
    if not GPU_LOCK.acquire(blocking=False):yield (busy_card(),)+NO;return
    try:
        import queue as _q
        Q=_q.Queue();box={};t0=time.perf_counter();done=set();msgs={};cur=1
        def emit(stage,title,body,final=True):Q.put((stage,body,final))
        def work():
            try:box["res"]=full_run(dict(scenario=scn,condition=cond,representation=rep,custom=q or "",text_controls=bool(text_ctl)),emit)
            except BaseException as ex:box["err"]=ex;box["tb"]=traceback.format_exc()
        th=threading.Thread(target=work,daemon=True);th.start()
        while th.is_alive() or not Q.empty():
            while not Q.empty():
                st,body,fin=Q.get();msgs[st]=body
                if fin:done.update(range(1,st+1));cur=st+1
                else:done.update(range(1,st));cur=st
            yield (stages_html(done,cur,msgs,t0),)+NO
            time.sleep(.7)
        th.join()
        if "err" in box:
            ex=box["err"];stg=getattr(ex,"stage",None)
            yield (stages_html(done,cur,msgs,t0,failed=stg),None,None,None,"",box.get("tb",""));yield (err_card(ex,stg),None,None,None,"",box.get("tb",""));return
        res=box["res"];P=res["P"];files=res["files"]
        summ=Path(files["summary"]).read_text(encoding="utf-8")
        flist=list(res["figures"])+[files["zip"],files["summary"],files["manifest"],str(Path(res["run_dir"])/"RAW_MODEL_ANSWERS.json")]
        head=stages_html(set(range(1,N_STAGES+1)),None,msgs,t0)+run_html(P["primary"]["AKBASCORE"],"Demonstration A · ABC · DC_CUT3")+run_html(P["primary"]["MATRIX"],"Demonstration B · ABC · DC_CUT3")
        if P["custom"]:head+=run_html(P["custom"],"Custom question (unscored)")
        yield (head,[(f,Path(f).stem.replace("_"," ")) for f in res["figures"]],flist,files["zip"],summ,"")
    except Exception as ex:
        yield (err_card(ex),None,None,None,"",traceback.format_exc())
    finally:GPU_LOCK.release()

def verify_handler(full):
    if not GPU_LOCK.acquire(blocking=False):return busy_card()
    try:
        m=E["model"];fz=frozen_state();g=param_guard(m);rows=[("time (UTC)",utc_now()),("trainable tensors",fz["trainable_tensors"]),("training mode",fz["training"]),("LoRA",fz["lora"]),
            ("residual hooks",len(fz["residual_hooks"])),("sentinel = load",fz["sentinel_equals_load"]),("all-parameter guard = start-up",g==STARTUP.get("guard_start"))]
        if full:
            t=time.perf_counter();h=full_weight_sha(m);rows+=[("full-weight SHA-256",h),("= TEST575 reference",h==WEIGHT_SHA_REFERENCE),("seconds",round(time.perf_counter()-t,1))]
        ok=fz["trainable_tensors"]==0 and fz["sentinel_equals_load"] and g==STARTUP.get("guard_start") and (not full or rows[-2][1])
        return card_html("Weight verification · measured now","<table class='tbl'>"+"".join(f"<tr><th>{esc(a)}</th><td class='mono'>{esc(b)}</td></tr>" for a,b in rows)+"</table>","on" if ok else "err")
    except Exception as ex:return err_card(ex)
    finally:GPU_LOCK.release()

def _tb(**kw):
    for extra in (([{"buttons":["copy"]}] if _GR_MAJOR>=6 else [])+[{"show_copy_button":True},{}]):
        try:return gr.Textbox(**extra,**kw)
        except TypeError:continue
def _gallery(**kw):
    try:return gr.Gallery(columns=1,height="auto",object_fit="contain",preview=False,**kw)
    except TypeError:return gr.Gallery(**kw)

def build_ui():
    blocks=gr.Blocks(title=APP_SHORT) if _GR_MAJOR>=6 else gr.Blocks(css=CSS,title=APP_SHORT)
    with blocks as demo:
        gr.HTML(HERO);gr.HTML(engine_card_html())
        scn0="AKBASCORE";v=scenario_values(scn0)
        with gr.Row():
            scn_dd=gr.Dropdown(choices=[("Demonstration A — Mustafa Akbaş / AKBASCORE","AKBASCORE"),("Demonstration B — The Matrix / Neo","MATRIX")],value=scn0,label="Scenario",interactive=True)
            cond_dd=gr.Dropdown(choices=list(CONDITIONS),value="ABC",label="Memory condition (which cartridges are written)",interactive=True)
        rep_rd=gr.Radio(choices=[("DC_CUT3 · native injection (main demonstration)","DC_CUT3"),("TEXT_JOINT · facts read as text (comparison control)","TEXT_JOINT")],value="DC_CUT3",label="Memory representation",interactive=True)
        info=gr.HTML(v[4])
        with gr.Accordion("Cartridge texts · view, edit, reset (edited cartridges are answered but not scored)",open=False):
            ta=gr.Textbox(value=v[0],label="Cartridge A",lines=2,max_lines=6);tb=gr.Textbox(value=v[1],label="Cartridge B",lines=2,max_lines=6)
            tc=gr.Textbox(value=v[2],label="Cartridge C",lines=2,max_lines=6);tcf=gr.Textbox(value=v[3],label="Counterfactual C (used only in ABC_CF)",lines=2,max_lines=6)
            with gr.Row():ap=gr.Button("Apply edited cartridges");rs=gr.Button("Reset to original texts")
        cq=gr.Textbox(value="",label="Optional custom question (answered from the selected memory; not scored)",placeholder="e.g. Who activates the AKBASCORE MAM architecture?",lines=1,max_lines=3)
        stg=gr.Checkbox(value=True,label="Also run the three-step staged chain")
        run1=gr.Button("Run selected configuration",variant="secondary")
        full=gr.Button("FULL RESEARCH RUN · both demonstrations · 9 conditions · 7 figures + ZIP",variant="primary",elem_id="kz_full")
        txt_ctl=gr.Checkbox(value=True,label="Include the TEXT_JOINT control grid in the full run (doubles the control time)")
        status=gr.HTML(card_html("Ready","Choose a scenario, condition and representation, then press a button. Progress stages appear here only after they have actually completed.","info"))
        gallery=_gallery(label="Figures 1–7 (JPEG, 300 dpi)")
        files=gr.File(label="Individual downloads · figures, ZIP, summary, manifest, raw answers",file_count="multiple",interactive=False)
        zipb=gr.DownloadButton(label="⬇ Download the complete experiment package (.zip)",value=None,interactive=True)
        single_file=gr.File(label="Record of the selected-configuration run (.json)",interactive=False)
        with gr.Tabs():
            with gr.Tab("Run summary"):summ=_tb(lines=14,max_lines=40,label="RUN_SUMMARY.txt")
            with gr.Tab("Raw record (selected run)"):raw=_tb(lines=14,max_lines=40,label="JSON")
            with gr.Tab("Error log"):errs=_tb(lines=10,max_lines=40,label="traceback (empty = no error)")
        with gr.Accordion("Verify the frozen weights now",open=False):
            fullsha=gr.Checkbox(value=False,label="Also compute the full-weight SHA-256 (TEST575 method, about 1–2 minutes)")
            vb=gr.Button("Verify weights now");vout=gr.HTML()
        gr.HTML(SCOPE_HTML)
        gr.HTML(f'<div class="kz small" style="opacity:.8;margin:8px 0 18px">{esc(AUTHOR)} · {esc(ORG)} · {esc(LICENSE_NAME)} · '
                f'TEST664 <a href="{URL_664}" target="_blank" rel="noopener">664.py</a> · TEST677 <a href="{URL_677}" target="_blank" rel="noopener">677.py</a></div>')
        scn_dd.change(scenario_values,inputs=scn_dd,outputs=[ta,tb,tc,tcf,info])
        ap.click(apply_handler,inputs=[scn_dd,ta,tb,tc,tcf],outputs=[ta,tb,tc,tcf,info])
        rs.click(reset_handler,inputs=scn_dd,outputs=[ta,tb,tc,tcf,info])
        run1.click(single_handler,inputs=[scn_dd,cond_dd,rep_rd,cq,stg],outputs=[status,raw,errs,single_file])
        full.click(full_handler,inputs=[scn_dd,cond_dd,rep_rd,cq,txt_ctl],outputs=[status,gallery,files,zipb,summ,errs])
        vb.click(verify_handler,inputs=fullsha,outputs=vout)
    try:demo.queue(default_concurrency_limit=1)
    except TypeError:demo.queue(concurrency_count=1)
    return demo

if not LAYOUT_TEST:
    _GR_MAJOR=int(re.match(r"\d+",gr.__version__).group())
    boot_engine()
    demo=build_ui()
    say("[launch] public share link — open the printed gradio.live address (also works on a phone)")
    _LAUNCH=dict(share=True,allowed_paths=[str(ROOT),"/content/AKBASCORE_MAM_v1_4"],show_error=True,inline=False)
    if _GR_MAJOR>=6:_LAUNCH["css"]=CSS
    try:demo.launch(**_LAUNCH)
    except TypeError as _ex:
        if "css" not in str(_ex):raise
        _LAUNCH.pop("css",None);demo.launch(**_LAUNCH)
