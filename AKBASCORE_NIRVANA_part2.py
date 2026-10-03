def p02_engine(ctx,k,path):
    P=ctx["P"];W=P["workload"];T=P["timing"]["run"];fig=plt.figure(figsize=(16,11),dpi=DPI,facecolor="white")
    top=head(fig,"INSIDE THE ENGINE","Simple interface ≠ simple engine. Every box below exists in this code and every number is from this run.")
    stack=[("GRADIO · CONTROL","one button, live stages, downloads",C_CTL,"#F1F5F9",f"{P['ui']['stages']} live stages"),
           ("PYTHON ORCHESTRATION","locks, checks, scoring, sealing, posters",C_CTL,"#F1F5F9",f"{P['integrity']['checks_total']} integrity checks"),
           ("AKBASCORE CARTRIDGE ENGINE","PCA encode/decode · RoPE install · IBR",C_CART,C_CARTL,f"{W['derived']['cart_encdec_mac']['human']} cartridge MACs"),
           ("PYTORCH CUDA GPU COMPUTE","cuBLAS GEMM · SDPA attention (no custom kernel)",C_MOD,C_MODL,f"{W['measured']['forward_passes']['value']:,} forward passes"),
           ("A100 GPU MEMORY · BF16 TENSORS",f"peak {P['gpu']['peak_allocated_gib']:.2f} GiB allocated",C_MOD,C_MODL,f"{P['gpu']['name']}"),
           (f"{MODEL_SHORT.upper()} · 28 LAYERS",f"{P['model']['params']/1e9:.2f} B parameters · frozen",C_MOD,C_MODL,"trainable = 0"),
           ("COMPRESSED K/V CARTRIDGE BANK",f"16 cartridges · {P['bank']['slots']} memory slots",C_CART,C_CARTL,f"{P['bank']['code_numbers']:,} code numbers"),
           ("IBR READOUT",f"{W['measured']['ibr_calls']['value']} batched calls × 16 rows",C_OK,C_OKL,f"{W['measured']['ibr_decode_steps']['value']} decode steps")]
    n=len(stack);hh=(top-.08)/n
    for i,(t,s,c,f_,m) in enumerate(stack):
        y=top-(i+1)*hh+.006
        fig.add_artist(FancyBboxPatch((.05,y),.44,hh-.012,boxstyle="round,pad=0,rounding_size=0.008",transform=fig.transFigure,facecolor=f_,edgecolor=c,lw=2.2))
        fig.text(.065,y+(hh-.012)*.66,t,fontsize=13,weight="bold",color=c,va="center");fig.text(.065,y+(hh-.012)*.28,s,fontsize=10.5,color=C_FG,va="center")
        fig.text(.48,y+(hh-.012)/2,m,fontsize=10.5,weight="bold",color=C_FG,va="center",ha="right",family=MONO)
        if i<n-1:fig.text(.27,y-.004,"▼",fontsize=9,color=C_NEU,ha="center",va="center")
    E=np.array([[c["normK"][L]+c["normV"][L] for L in range(TOTAL_LAYERS)] for c in P["cartridges"]])
    ax=fig.add_axes([.56,top-.40,.39,.36]);im=ax.imshow(E,aspect="auto",cmap="magma")
    ax.set_xticks(range(0,28,3));ax.set_xticklabels([f"L{x}" for x in range(0,28,3)],fontsize=9);ax.set_yticks(range(16));ax.set_yticklabels([c["cid"] for c in P["cartridges"]],fontsize=8.5)
    ax.set_title("ENGINE TUNNEL · code energy per layer per cartridge",fontsize=12.5,weight="bold",loc="left");fig.colorbar(im,cax=fig.add_axes([.955,top-.40,.008,.36]))
    caption(fig,.56,top-.50,.39,.07,"16 cartridges (rows) flowing through 28 transformer layers (columns). Color = measured norm of the stored K+V codes.")
    ax2=fig.add_axes([.60,.10,.35,top-.66]);st=[("codebook",P["codebook"]["seconds"]),("16 cartridges",T["cartridges_seconds"]),("question battery",T["battery_seconds"]),("NOMEM control",T["control_seconds"])]
    ax2.barh(range(4),[s for _,s in st],color=[C_CART,C_CART,C_OK,C_CTL]);ax2.invert_yaxis();ax2.set_yticks(range(4));ax2.set_yticklabels([a for a,_ in st],fontsize=11)
    for i,(_,s) in enumerate(st):ax2.text(s,i,f" {s:.2f} s",va="center",fontsize=11,weight="bold")
    ax2.set_xlim(0,max(s for _,s in st)*1.35);ax2.set_xlabel("measured seconds (GPU-synchronized)",fontsize=10.5);clean(ax2)
    foot(fig,ctx,k);return save_jpg(fig,path)
def p03_bank(ctx,k,path):
    P=ctx["P"];C=P["cartridges"];fig=plt.figure(figsize=(16,11),dpi=DPI,facecolor="white")
    top=head(fig,"THE CARTRIDGE BANK · 16 LOADED","Each slot is one independently forged cartridge. IBR reads every slot in its own isolated row.")
    gw,gh=.105,(top-.14)/4
    for i,c in enumerate(C):
        r,q=divmod(i,4);x=.05+q*(gw+.008);y=top-(r+1)*gh-.005;col=ROLE_COLOR[c["role"]]
        fig.add_artist(FancyBboxPatch((x,y),gw,gh-.012,boxstyle="round,pad=0,rounding_size=0.008",transform=fig.transFigure,facecolor="white",edgecolor=col,lw=2.4))
        fig.text(x+.007,y+gh-.03,c["cid"],fontsize=13,weight="bold",color=col,va="top");fig.text(x+gw-.007,y+gh-.03,c["fam"],fontsize=10,color=C_NEU,va="top",ha="right")
        fig.text(x+.007,y+(gh-.012)*.52,c["role"].replace(" ","\n",1) if len(c["role"])>10 else c["role"],fontsize=8.5,weight="bold",color=col,va="center")
        fig.text(x+.007,y+(gh-.012)*.2,f"{c['src']} →\n{c['dst']}",fontsize=8,color=C_FG,va="center",linespacing=1.1)
    ax=fig.add_axes([.53,.12,.42,top-.16]);ax.set_xlim(0,3);ax.set_ylim(-.5,9.5);ax.axis("off")
    objs=[c["obj"] for c in CHAINS]+[o["obj"] for o in ORPHANS];ids=[c["T"] for c in CHAINS]+[o["T"] for o in ORPHANS]+[c["D"] for c in CHAINS];places=KNOWN_PLACES
    oy={o:8.6-i*1.15 for i,o in enumerate(objs)};iy={d:9.1-i*.78 for i,d in enumerate(ids)};py={p:8.6-i*1.15 for i,p in enumerate(places)}
    for c in C:
        if c["role"] in("OBJECT→ID","ORPHAN OBJECT→ID"):ax.plot([.55,1.45],[oy[c["src"]],iy[c["dst"]]],color=ROLE_COLOR[c["role"]],lw=2.2)
        else:ax.plot([1.55,2.45],[iy[c["src"]],py[c["dst"]]],color=ROLE_COLOR[c["role"]],lw=2.2,ls="--" if c["role"].startswith("DECOY") else "-")
    for o,y in oy.items():ax.text(.5,y,o,ha="right",va="center",fontsize=9.5,weight="bold",color=C_FG)
    for d,y in iy.items():
        orphan=d in [o["T"] for o in ORPHANS];ax.text(1.5,y,d,ha="center",va="center",fontsize=9.5,family=MONO,color=C_UNK if orphan else C_FG,
                    bbox=dict(boxstyle="round,pad=0.25",fc=C_UNKL if orphan else "white",ec=C_UNK if orphan else C_NEU,lw=1.2))
    for p,y in py.items():ax.text(2.5,y,p,ha="left",va="center",fontsize=9.5,weight="bold",color=C_OK if p in [c["P"] for c in CHAINS] else C_CTL)
    ax.text(.5,9.7,"OBJECTS",ha="right",fontsize=11,weight="bold",color=C_CART);ax.text(1.5,9.7,"CONTAINER IDs",ha="center",fontsize=11,weight="bold",color=C_FG);ax.text(2.5,9.7,"PLACES",ha="left",fontsize=11,weight="bold",color=C_OK)
    lg=[Line2D([0],[0],color=C_CART,lw=3,label="object→ID cartridge"),Line2D([0],[0],color=C_OK,lw=3,label="ID→place cartridge"),Line2D([0],[0],color=C_CTL,lw=3,ls="--",label="decoy ID→place"),Line2D([0],[0],color=C_UNK,lw=3,label="orphan object→ID (no place)")]
    ax.legend(handles=lg,loc="lower center",bbox_to_anchor=(.5,-.12),ncol=2,fontsize=9.5,frameon=False)
    caption(fig,.05,.02,.46,.08,"purple IDs have no location cartridge: the honest answer for their objects is UNKNOWN.")
    foot(fig,ctx,k);return save_jpg(fig,path)
def p04_work(ctx,k,path):
    P=ctx["P"];W=P["workload"];fig=plt.figure(figsize=(16,11),dpi=DPI,facecolor="white")
    top=head(fig,"HOW HARD DID THE ENGINE WORK?","Three honest categories: MEASURED counters · DERIVED exact counts from tensor shapes · ESTIMATED model FLOPs")
    groups=[("MEASURED",C_MOD,[(W["measured"][k_]["label"],W["measured"][k_]["value"]) for k_ in W["measured_order"]]),
            ("DERIVED",C_CART,[(W["derived"][k_]["label"],W["derived"][k_]["value"]) for k_ in W["derived_order"]]),
            ("ESTIMATED",C_UNK,[(W["estimated"][k_]["label"],W["estimated"][k_]["value"]) for k_ in W["estimated_order"]])]
    y0=top-.03;heights=[.27,.22,.15]
    for (g,c,items),hgt in zip(groups,heights):
        ax=fig.add_axes([.40,y0-hgt,.44,hgt-.035]);vals=[max(1,v) for _,v in items]
        ax.barh(range(len(items)),vals,color=c,alpha=.85);ax.set_xscale("log");ax.invert_yaxis();ax.set_yticks(range(len(items)));ax.set_yticklabels([l for l,_ in items],fontsize=10.5)
        for i,v in enumerate(vals):ax.text(v*1.25,i,human(v),va="center",fontsize=11,weight="bold")
        ax.set_xlim(1,max(vals)*400);ax.set_xticks([]);clean(ax)
        fig.text(.05,y0+.012,g,fontsize=17,weight="bold",color=c,va="top");ylast=y0;y0-=hgt+.012
    tot=W["estimated"]["total_flop"]["value"]
    fig.text(.05,ylast-.05,f"≈ {human(tot)}",fontsize=17,weight="bold",color=C_UNK,va="top");fig.text(.05,ylast-.095,"estimated GPU FLOPs, this run",fontsize=11,color=C_FG,va="top")
    fit_text(fig,.05,.025,.9,.095,W["formula_note"],fs_max=10.5,fs_min=7.5,family=MONO,color=C_NEU)
    foot(fig,ctx,k);return save_jpg(fig,path)
def p05_compress(ctx,k,path):
    P=ctx["P"];C=P["cartridges"];B=P["bank"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"SOURCE → CARTRIDGE ENCODING","Measured sizes for every cartridge in this run (bf16, 2 bytes per number)")
    ax=fig.add_axes([.07,.36,.55,top-.42]);x=np.arange(16);w=.4
    ax.bar(x-w/2,[c["native_numbers"]*2/1024 for c in C],w,color=C_MOD,label="native K/V cache (KiB)")
    ax.bar(x+w/2,[(c["code_numbers"]+c["own_numbers"])*2/1024 for c in C],w,color=C_CART,label="cartridge codes + own slot 0 (KiB)")
    ax.set_xticks(x);ax.set_xticklabels([c["cid"] for c in C],fontsize=9,rotation=45);ax.set_ylabel("KiB per cartridge",fontsize=11);ax.legend(frameon=False,fontsize=10.5);clean(ax)
    caption(fig,.07,.26,.55,.07,f"cartridge = {B['ratio_vs_native']*100:.1f}% of the native K/V it replaces (K keeps 120 of 128 dims per head; V keeps all 128).")
    ax2=fig.add_axes([.70,.36,.25,top-.42]);lab=["hidden-state\ncopy","native K/V","cartridge\ncodes"];val=[B["hidden_copy_numbers"]*2/2**20,B["native_numbers"]*2/2**20,(B["code_numbers"]+B["own_numbers"])*2/2**20]
    ax2.bar(range(3),val,color=[C_CTL,C_MOD,C_CART])
    for i,v in enumerate(val):ax2.text(i,v*1.02,f"{v:.2f}\nMiB",ha="center",va="bottom",fontsize=11,weight="bold")
    ax2.set_xticks(range(3));ax2.set_xticklabels(lab,fontsize=10);ax2.set_ylim(0,max(val)*1.3);ax2.set_title("whole bank",fontsize=12,weight="bold");clean(ax2)
    note(fig,.05,.04,.9,.2,f"Honest size accounting: the cartridge is not a large size reduction versus the model's own K/V cache ({B['ratio_vs_native']*100:.1f}%). It is {B['ratio_vs_hidden']*100:.1f}% of a full hidden-state copy. What changes is the form: the source text is gone and only codes against a fixed neutral codebook remain.",
         f"code numbers = Σ 28 × (T−1) × 4 heads × (120 K + 128 V); own slot 0 = 28 × 2 × 512 raw; native = 28 × 2 × 512 × T. Shared codebook (μ + basis, fp32) = {P['codebook']['bytes']/2**20:.2f} MiB, stored once for all cartridges. At readout, cartridges are decoded to full-size runtime K/V ({B['runtime_numbers']*2/2**20:.2f} MiB for the bank).")
    foot(fig,ctx,k);return save_jpg(fig,path)
def p06_telemetry(ctx,k,path):
    P=ctx["P"];C=P["cartridges"];fig=plt.figure(figsize=(16,11),dpi=DPI,facecolor="white")
    top=head(fig,"CARTRIDGE TELEMETRY · LAYER × CARTRIDGE","Real per-layer measurements for each cartridge as it was forged in this run")
    mats=[("K reconstruction cosine",np.array([c["cosK"] for c in C]),"YlGn"),("V reconstruction cosine",np.array([c["cosV"] for c in C]),"YlGn"),
          ("K code norm (mean per token)",np.array([c["normK"] for c in C]),"viridis"),("V code norm (mean per token)",np.array([c["normV"] for c in C]),"viridis")]
    for i,(t,M,cm) in enumerate(mats):
        r,q=divmod(i,2);x=.06+q*.47;y=top-(r+1)*.37+.03
        ax=fig.add_axes([x,y,.38,.28]);im=ax.imshow(M,aspect="auto",cmap=cm);ax.set_title(t,fontsize=12.5,weight="bold",loc="left")
        ax.set_xticks(range(0,28,4));ax.set_xticklabels([f"L{v}" for v in range(0,28,4)],fontsize=9);ax.set_yticks(range(0,16,3));ax.set_yticklabels([C[j]["cid"] for j in range(0,16,3)],fontsize=9)
        fig.colorbar(im,cax=fig.add_axes([x+.385,y,.007,.28]))
        ax.text(1.0,-.2,f"min {M.min():.4f} · max {M.max():.4f}",transform=ax.transAxes,ha="right",fontsize=9.5,family=MONO,color=C_NEU)
    note(fig,.05,.04,.9,.09,"Cosine 1.0 = the decoded memory equals the model's own K/V. K keeps 120 of 128 dimensions, so it is close but not perfect; V keeps the full 128-dim basis.")
    foot(fig,ctx,k);return save_jpg(fig,path)
def p07_source(ctx,k,path):
    P=ctx["P"];S=P["source_removal"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"SOURCE REMOVAL PROOF","What the model receives when it answers: the question as text, the knowledge only as installed cartridge numbers")
    box(fig,.05,top-.22,.25,.17,"SOURCE TEXT",f"{S['source_sentences']} sentences\nused ONCE to forge cartridges",C_FG,C_BG)
    box(fig,.38,top-.22,.25,.17,"CARTRIDGE BANK",f"{P['bank']['slots']} slots · numbers only\ninstalled into the cache",C_CART,C_CARTL)
    box(fig,.71,top-.22,.24,.17,"READOUT PROMPT","question text only\n'QUESTION: … ANSWER:'",C_MOD,C_MODL)
    arrow(fig,.30,top-.135,.38,top-.135);arrow(fig,.63,top-.135,.71,top-.135)
    fig.text(.50,top-.27,"✗  source text → readout prompt: blocked and asserted in every call",ha="center",fontsize=14,weight="bold",color=C_ERR)
    ax=fig.add_axes([.07,.24,.86,top-.58]);calls=S["calls"];x=np.arange(len(calls))
    ax.bar(x,[c["prompt_tokens"] for c in calls],color=C_MOD,label="question tokens in prompt")
    ax.bar(x,[c["source_hits"] for c in calls],color=C_ERR,label="source sentences in prompt (all zero)")
    ax.plot(x,[c["slots"] for c in calls],"o-",color=C_CART,lw=2,label="installed cartridge slots read")
    ax.set_xticks(x);ax.set_xticklabels([c["tag"] for c in calls],fontsize=8.5,rotation=60);ax.legend(frameon=False,fontsize=10.5,ncol=3,loc="upper left");ax.set_ylabel("count per IBR call",fontsize=11);clean(ax)
    ax.set_ylim(0,max(max(c["slots"] for c in calls),max(c["prompt_tokens"] for c in calls))*1.35)
    note(fig,.05,.04,.9,.11,f"SOURCE SENTENCES IN READOUT PROMPTS = {S['source_sentence_hits']} across {len(calls)} IBR calls. The prompt carries the question only (it names the object or container ID being asked about). The knowledge arrives as installed K/V cartridge rows.")
    foot(fig,ctx,k);return save_jpg(fig,path)
def p08_frozen(ctx,k,path):
    P=ctx["P"];I=P["integrity"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"FROZEN MODEL PROOF","The model's weights were checked before and after this run")
    pts=["startup","pre-run","post-run"];F=np.array([I["fingerprint_startup"],I["fingerprint_pre_run"],I["fingerprint_after"]])
    ax=fig.add_axes([.07,.40,.50,top-.46])
    for j in range(F.shape[1]):ax.plot(range(3),F[:,j]/F[0,j],"o-",lw=2.2,ms=9,label=FP_NAMES[j])
    ax.set_xticks(range(3));ax.set_xticklabels(pts,fontsize=12);ax.set_ylim(.999,1.001);ax.set_ylabel("tensor sum ÷ startup value",fontsize=11)
    ax.legend(frameon=False,fontsize=9,loc="lower center",ncol=2);ax.grid(alpha=.25);clean(ax)
    caption(fig,.07,.30,.50,.07,"six selected weight tensors: identical float32 sums at startup, before and after the run (ratio exactly 1.0).")
    tiles=[("trainable tensors",str(I["trainable_parameter_tensors"])),("LoRA","none" if not I["lora"] else "PRESENT"),("optimizer","none"),
           ("AkbasCore hooks",str(I["akbascore_hooks_after_run"])),("training mode",str(I["model_training_mode"])),("result",I["result"])]
    for i,(a,b) in enumerate(tiles):
        r,q=divmod(i,2);x=.62+q*.17;y=top-.13-r*.13
        fig.add_artist(FancyBboxPatch((x,y),.16,.11,boxstyle="round,pad=0,rounding_size=0.01",transform=fig.transFigure,facecolor=C_MODL,edgecolor=C_MOD,lw=2))
        fig.text(x+.08,y+.068,b,ha="center",va="center",fontsize=20,weight="bold",color=C_MOD);fig.text(x+.08,y+.025,a,ha="center",va="center",fontsize=11,color=C_FG)
    txt=(f"SAMPLED SHA-256 SENTINEL (16 × 256 contiguous values per selected tensor)\n startup : {I['sentinel_startup']}\n pre-run : {I['sentinel_pre_run']}\n post-run: {I['sentinel_after']}\n"
         "This is a sampled sentinel, not a cryptographic hash of every weight.")
    fit_text(fig,.05,.05,.9,.2,txt,fs_max=12,fs_min=8,family=MONO)
    foot(fig,ctx,k);return save_jpg(fig,path)
def p09_battery(ctx,k,path):
    P=ctx["P"];Bt=P["battery"];fig=plt.figure(figsize=(16,12),dpi=DPI,facecolor="white")
    top=head(fig,"AUTOMATIC QUESTION BATTERY","16 locked questions · expected vs observed · scored after generation",tag="THIS LIVE RUN")
    ax=fig.add_axes([.05,.10,.52,top-.13]);ax.axis("off");n=len(Bt);rh=1/(n+1)
    hdr=["Q","type","question","expected","observed","status"];cx=[0,.055,.20,.555,.715,.855]
    for c_,h_ in zip(cx,hdr):ax.text(c_,1-rh*.5,h_,fontsize=11,weight="bold",color=C_FG,va="center",transform=ax.transAxes)
    for i,q in enumerate(Bt):
        y=1-rh*(i+1.5);col=status_color(q["status"])
        ax.add_patch(Rectangle((0,y-rh*.45),1,rh*.9,transform=ax.transAxes,color=C_OKL if q["status"] in("CORRECT","UNKNOWN ✓") else C_ERRL,lw=0))
        vals=[q["qid"],q["kind"].lower(),q["label"],q["expected"],q["observed"],q["status"]]
        for j,(c_,v) in enumerate(zip(cx,vals)):
            ax.text(c_+.005,y,textwrap.shorten(str(v),{0:4,1:13,2:38,3:19,4:19,5:12}[j],placeholder="…"),fontsize=9.5 if j in(2,3,4,5) else 10,
                    weight="bold" if j in(0,5) else "normal",color=col if j==5 else C_FG,va="center",transform=ax.transAxes)
    M=np.array([q["lane"] for q in Bt],dtype=float)
    ax2=fig.add_axes([.61,.20,.34,top-.23]);ax2.imshow(M,aspect="auto",cmap=ListedColormap(["#F1F5F9",C_CART,C_OK]),vmin=0,vmax=2)
    ax2.set_xticks(range(16));ax2.set_xticklabels([c["cid"] for c in P["cartridges"]],rotation=90,fontsize=8.5);ax2.set_yticks(range(n));ax2.set_yticklabels([q["qid"] for q in Bt],fontsize=9.5)
    ax2.set_title("IBR LANES · which cartridge answered",fontsize=12,weight="bold",loc="left")
    caption(fig,.61,.06,.34,.11,"amber = cartridge returned a container ID (Stage 1); green = cartridge returned a place (Stage 2). Empty = that cartridge said NONE.")
    foot(fig,ctx,k);return save_jpg(fig,path)
def two_panel(fig,top,live_items,sealed_items,live_title,sealed_title,colors):
    for j,(items,title,tag) in enumerate([(live_items,live_title,"THIS LIVE RUN"),(sealed_items,sealed_title,"SEALED EXPERIMENTAL RECORD")]):
        x=.07+j*.47;ax=fig.add_axes([x,.33,.38,top-.42]);labs=[a for a,_ in items];v=[pct(t) for _,t in items]
        ax.bar(range(len(items)),[100]*len(items),color="#E2E8F0");ax.bar(range(len(items)),v,color=colors[j])
        for i,(a,t) in enumerate(items):ax.text(i,v[i]+2,f"{frac(t)}\n{v[i]:.1f}%",ha="center",va="bottom",fontsize=13,weight="bold")
        ax.set_xticks(range(len(items)));ax.set_xticklabels(labs,fontsize=10.5);ax.set_ylim(0,125);ax.set_ylabel("%",fontsize=11);clean(ax)
        ax.set_title(title,fontsize=13,weight="bold",loc="left",color=C_MOD if j==0 else C_UNK)
        fig.text(x,top-.02,tag,fontsize=12,weight="bold",color=C_MOD if j==0 else C_UNK)
def p10_linked(ctx,k,path):
    P=ctx["P"];S=P["live_summary"];H_=HIST["TEST461"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"LINKED RETRIEVAL · FOLLOWING THE CHAIN","object → container (cartridge A) → place (cartridge B), across 16 isolated cartridge rows")
    two_panel(fig,top,[("two-hop linked",tuple(S["by_kind"]["LINKED"][x] for x in("ok","n"))),("direct ID→place",tuple(S["by_kind"]["DIRECT"][x] for x in("ok","n")))],
              [("overall linked",H_["L"]),("scale 2",H_["scale"][2]["L"]),("scale 8",H_["scale"][8]["L"]),("scale 16",H_["scale"][16]["L"])],
              "this run · 16-cartridge bank","TEST461 final held-out · 16 entities",[C_OK,C_OK])
    note(fig,.05,.05,.9,.18,f"Sealed record: 43/48 linked questions correct on the final held-out panel (Wilson 95% lower bound {H_['L_lo']:.3f}). This live run uses a different, smaller locked panel; its numbers are reported separately and are not added to the sealed record.")
    foot(fig,ctx,k);return save_jpg(fig,path)
def p11_unknown(ctx,k,path):
    P=ctx["P"];S=P["live_summary"];H_=HIST["TEST461"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"MISSING CONNECTION? IT REFUSED TO INVENT ONE.","Questions whose answer is not in the cartridges must end in UNKNOWN")
    nm=P["controls"]["nomem"]
    two_panel(fig,top,[("missing link",tuple(S["by_kind"]["MISSING LINK"][x] for x in("ok","n"))),("absent ID",tuple(S["by_kind"]["ABSENT ID"][x] for x in("ok","n"))),("NO-MEMORY\nstrict hits",(nm["strict_hits"],nm["n"]))],
              [("missing link → UNKNOWN",H_["U"]),("NOMEM strict hits",H_["nomem"])],"this run","TEST461 final held-out",[C_UNK,C_UNK])
    note(fig,.05,.05,.9,.18,"Missing link: the object's container has no location cartridge. Absent ID: the container is in no cartridge at all. NO-MEMORY control: same questions with an empty cache — strict hits must be 0, proving the right answers come from the cartridges. Scope: this controlled panel only; this is not a general hallucination claim.")
    foot(fig,ctx,k);return save_jpg(fig,path)
def p12_falselink(ctx,k,path):
    P=ctx["P"];Bt=P["battery"];H_=HIST["TEST461"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"FALSE-LINK CONTROL · DECOYS PRESENT","Decoy cartridges place other containers at real places. Did the model attach them to the wrong question?")
    neg=[q for q in Bt if q["exp_place"] is None];cats=["UNKNOWN","decoy place","other place / ID"]
    M=np.zeros((len(neg),3))
    for i,q in enumerate(neg):
        o=q["obs_place"];j=0 if o is None else (1 if o in [c["Q"] for c in CHAINS] else 2);M[i,j]=1
    ax=fig.add_axes([.08,.30,.38,top-.38]);ax.imshow(M,aspect="auto",cmap=ListedColormap(["#F8FAFC",C_UNK]),vmin=0,vmax=1)
    for i in range(M.shape[0]):
        for j in range(3):ax.text(j,i,"●" if M[i,j] else "",ha="center",va="center",fontsize=16,color="white")
    ax.set_xticks(range(3));ax.set_xticklabels(cats,fontsize=11);ax.set_yticks(range(len(neg)));ax.set_yticklabels([f"{q['qid']} · {q['label'][:28]}" for q in neg],fontsize=9.5)
    ax.set_title("THIS LIVE RUN · observed outcome",fontsize=12.5,weight="bold",loc="left",color=C_MOD)
    ax2=fig.add_axes([.60,.30,.32,top-.38]);fl=H_["false_link"];ax2.bar([0,1],[fl[1]-fl[0],fl[0]],color=[C_UNK,C_ERR])
    ax2.set_xticks([0,1]);ax2.set_xticklabels(["no false link","false link"],fontsize=11)
    for i,v in enumerate([fl[1]-fl[0],fl[0]]):ax2.text(i,v+.8,f"{v}/{fl[1]}",ha="center",fontsize=14,weight="bold")
    ax2.set_ylim(0,fl[1]*1.25);ax2.set_title("SEALED RECORD · TEST461 unlinked questions",fontsize=12.5,weight="bold",loc="left",color=C_UNK);clean(ax2)
    note(fig,.05,.05,.9,.16,f"Live false links: {P['live_summary']['false_links']}. Sealed record: {frac(fl)} false links on the final held-out panel — on this sealed panel only.")
    foot(fig,ctx,k);return save_jpg(fig,path)
def p13_scale(ctx,k,path):
    P=ctx["P"];H_=HIST["TEST461"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"MORE CARTRIDGES · STILL BROADLY STABLE (UP TO 16)","Sealed record: the same questions read with 2, 8 and 16 cartridge rows",tag="SEALED EXPERIMENTAL RECORD")
    s=[2,8,16];L=[pct(H_["scale"][x]["L"]) for x in s];U=[pct(H_["scale"][x]["U"]) for x in s]
    ax=fig.add_axes([.08,.30,.56,top-.38]);ax.plot(s,L,"o-",color=C_OK,lw=3,ms=11,label="linked correct");ax.plot(s,U,"s-",color=C_UNK,lw=3,ms=11,label="missing link → UNKNOWN")
    for x,a,b in zip(s,L,U):ax.text(x,a-6,f"{a:.2f}%",ha="center",fontsize=12,weight="bold",color=C_OK);ax.text(x,b+2.5,f"{b:.0f}%",ha="center",fontsize=12,weight="bold",color=C_UNK)
    ax.set_xticks(s);ax.set_xticklabels([f"{x} cartridges" for x in s],fontsize=12);ax.set_ylim(60,110);ax.set_ylabel("%",fontsize=11);ax.grid(alpha=.25);ax.legend(frameon=False,fontsize=12,loc="lower left");clean(ax)
    lv=P["live_summary"]["by_kind"]["LINKED"];fig.text(.70,top-.10,"2 → 16 change",fontsize=15,weight="bold",color=C_FG)
    fig.text(.70,top-.16,f"linked: −{L[0]-L[2]:.2f} points",fontsize=14,color=C_OK);fig.text(.70,top-.21,f"UNKNOWN: −{U[0]-U[2]:.2f} points",fontsize=14,color=C_UNK)
    fig.text(.70,top-.31,"THIS LIVE RUN (16 cartridges)",fontsize=12,weight="bold",color=C_MOD);fig.text(.70,top-.36,f"linked {lv['ok']}/{lv['n']} (different panel)",fontsize=12,color=C_FG)
    note(fig,.05,.05,.9,.17,"Tested range only: 2, 8 and 16 cartridges. No claim is made beyond 16 or about unlimited memory.")
    foot(fig,ctx,k);return save_jpg(fig,path)
def p14_family(ctx,k,path):
    P=ctx["P"];H_=HIST["TEST461"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"WHERE IT WORKED — AND WHERE IT DIDN'T","Four ways of phrasing the same fact. F3 and F4 were first-use, held-out phrasings.",tag="SEALED EXPERIMENTAL RECORD")
    f=["F1","F2","F3","F4"];v=[pct(H_["fam"][x]) for x in f];cols=[C_OK if x>=70 else C_ERR for x in v]
    ax=fig.add_axes([.08,.33,.50,top-.42]);ax.bar(range(4),v,color=cols);ax.axhline(70,color=C_FG,ls="--",lw=2);ax.text(-.45,112,"- - -  precommitted per-family gate: 70%",ha="left",fontsize=11,weight="bold")
    for i,x in enumerate(f):ax.text(i,v[i]/2,f"{frac(H_['fam'][x])}\n{v[i]:.1f}%",ha="center",va="center",fontsize=14,weight="bold",color="white")
    ax.set_xticks(range(4));ax.set_xticklabels([x+"\n"+textwrap.fill(FAM[x][1].format(T="ID",P="PLACE"),22) for x in f],fontsize=9.5);ax.set_ylim(0,125);ax.set_ylabel("linked correct %",fontsize=11);clean(ax)
    fig.add_artist(FancyBboxPatch((.63,top-.25),.32,.2,boxstyle="round,pad=0,rounding_size=0.01",transform=fig.transFigure,facecolor=C_ERRL,edgecolor=C_ERR,lw=2.5))
    fig.text(.79,top-.10,"FINAL VERDICT",ha="center",fontsize=14,weight="bold",color=C_ERR);fig.text(.79,top-.17,H_["verdict"],ha="center",fontsize=16,weight="bold",color=C_ERR,family=MONO)
    fig.text(.79,top-.225,"F2 missed the each-family ≥ .70 gate",ha="center",fontsize=11,color=C_FG)
    lv=P["live_summary"]["by_family"];txt="THIS LIVE RUN · per family (linked + direct):\n"+"   ".join(f"{x}: {lv[x]['ok']}/{lv[x]['n']}" for x in f)
    fit_text(fig,.63,top-.42,.32,.14,txt,fs_max=12.5,fs_min=9,weight="bold",color=C_MOD)
    note(fig,.05,.05,.9,.19,"F2 phrases the location backwards (“The PLACE houses container ID”). On the sealed final panel F2 reached 7/12 and the precommitted per-family gate failed. The result is preserved unchanged: no threshold, parser or template was altered afterwards.")
    foot(fig,ctx,k);return save_jpg(fig,path)
def p15_f2(ctx,k,path):
    H_=HIST["TEST464"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"A REAL BOUNDARY CONDITION · F2 X-RAY","TEST464: the true location cartridge read alone, and with 1, 7 and 15 other cartridges",tag="SEALED EXPERIMENTAL RECORD")
    G=np.array(H_["grid"],dtype=float);ax=fig.add_axes([.10,.36,.42,top-.44]);ax.imshow(G,aspect="auto",cmap=ListedColormap([C_ERRL,C_OKL]),vmin=0,vmax=1)
    for i in range(4):
        for j in range(4):ax.text(j,i,"✓" if G[i,j] else "✗",ha="center",va="center",fontsize=24,weight="bold",color=C_OK if G[i,j] else C_ERR)
    ax.set_xticks(range(4));ax.set_xticklabels(H_["cols"],fontsize=12);ax.set_yticks(range(4));ax.set_yticklabels(H_["entities"],fontsize=12)
    stats=[("true B read alone",frac(H_["b_solo"])),("true B in larger banks",frac(H_["larger"])),("aggregation collisions",str(H_["collisions"]))]
    for i,(a,b) in enumerate(stats):
        y=top-.12-i*.12;fig.add_artist(FancyBboxPatch((.58,y),.37,.10,boxstyle="round,pad=0,rounding_size=0.01",transform=fig.transFigure,facecolor=C_BG,edgecolor=C_CTL,lw=2))
        fig.text(.60,y+.05,b,fontsize=22,weight="bold",color=C_FG,va="center");fig.text(.73,y+.05,a,fontsize=12.5,color=C_FG,va="center")
    note(fig,.05,.05,.9,.24,f"e01: the correct place appears at the start of the output (\"{H_['e01_raw']}\") but the strict first-line readout does not accept the continuation — a readout-format failure, not proof the knowledge is gone. e13: correct with 1 and 2 cartridges, fails at 8, correct again at 16 — not a monotonic capacity collapse. Verdict: {H_['verdict']} (frozen weights ✓, errors 0).")
    foot(fig,{**ctx},k);return save_jpg(fig,path)
def p16_ibr(ctx,k,path):
    H_=HIST["TEST460"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"WHY IBR EXISTS","Independent memories interfered when simply concatenated. Reading them in isolation fixed this development panel.",tag="SEALED EXPERIMENTAL RECORD")
    labs=list(H_["arms"].keys());L=[pct(H_["arms"][a][0]) for a in labs];U=[pct(H_["arms"][a][1]) for a in labs];x=np.arange(len(labs));w=.38
    ax=fig.add_axes([.07,.30,.60,top-.38]);ax.bar(x-w/2,L,w,color=C_OK,label="linked correct");ax.bar(x+w/2,U,w,color=C_UNK,label="missing link → UNKNOWN")
    for i,a in enumerate(labs):ax.text(x[i]-w/2,L[i]+1.5,frac(H_["arms"][a][0]),ha="center",fontsize=10,weight="bold");ax.text(x[i]+w/2,U[i]+1.5,frac(H_["arms"][a][1]),ha="center",fontsize=10,weight="bold")
    ax.set_xticks(x);ax.set_xticklabels(labs,fontsize=9.5);ax.set_ylim(0,118);ax.legend(frameon=False,fontsize=11,loc="upper left");clean(ax)
    ax2=fig.add_axes([.72,.30,.23,top-.38]);ax2.axis("off");ax2.set_xlim(0,1);ax2.set_ylim(0,1)
    ax2.text(.5,.95,"concatenated",ha="center",fontsize=12,weight="bold",color=C_ERR)
    for j,c in enumerate([C_CART,C_OK,C_CTL]):ax2.add_patch(Rectangle((.08+j*.28,.72),.26,.14,color=c,alpha=.8))
    ax2.text(.5,.66,"one row: memories blur together",ha="center",fontsize=10)
    ax2.text(.5,.52,"IBR (isolated rows)",ha="center",fontsize=12,weight="bold",color=C_OK)
    for j,c in enumerate([C_CART,C_OK,C_CTL]):ax2.add_patch(Rectangle((.2,.36-j*.12),.6,.09,color=c,alpha=.8))
    ax2.text(.5,.0,"each row sees one cartridge",ha="center",fontsize=10)
    note(fig,.05,.05,.9,.17,f"Mechanism verdict recorded in TEST460: {H_['mechanism']}. COFORGE (all records forged together) is a diagnostic that breaks modularity. Scope: development panel — not a universal claim about interference.")
    foot(fig,ctx,k);return save_jpg(fig,path)
def p17_batch(ctx,k,path):
    H_=HIST["TEST462"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"MEANING = SAME · TAIL TEXT = SOMETIMES DIFFERENT","TEST462: sequential reading vs batched IBR, compared answer by answer",tag="SEALED EXPERIMENTAL RECORD")
    labs=list(H_["b1"].keys());x=np.arange(len(labs));w=.38;a=[pct(H_["b1"][l]) for l in labs];b=[pct(H_["multi"][l]) for l in labs]
    ax=fig.add_axes([.08,.30,.84,top-.38]);ax.bar(x-w/2,a,w,color=C_MOD,label="batch of 1 vs sequential");ax.bar(x+w/2,b,w,color=[C_OK,C_OK,C_OK,C_CART],label="multi-row batch vs sequential")
    for i,l in enumerate(labs):ax.text(x[i]-w/2,a[i]+1.5,frac(H_["b1"][l]),ha="center",fontsize=12,weight="bold");ax.text(x[i]+w/2,b[i]+1.5,frac(H_["multi"][l]),ha="center",fontsize=12,weight="bold")
    ax.set_xticks(x);ax.set_xticklabels(labs,fontsize=13);ax.set_ylim(0,120);ax.set_ylabel("% identical",fontsize=11);ax.legend(frameon=False,fontsize=12,loc="upper right");clean(ax)
    note(fig,.05,.05,.9,.17,f"Verdict {H_['verdict']}: first token, first line and semantic answer matched in 32/32; the text after the answer matched in 20/32 for multi-row batches. Batched reading is semantically equivalent here — it is not claimed to be bit-identical.")
    foot(fig,ctx,k);return save_jpg(fig,path)
def p18_seal(ctx,k,path):
    P=ctx["P"];T=P["timing"];I=P["integrity"];E=P["environment"];fig=plt.figure(figsize=(16,12),dpi=DPI,facecolor="white")
    top=head(fig,"COMPLETE RUN · EVIDENCE SEAL","Everything below is from this run and is contained in the sealed payload",tag="THIS LIVE RUN")
    tiles=[("RUN ID",P["run_id"]),("MODEL",MODEL_ID),("GPU",E["gpu"]),("TORCH / TRANSFORMERS",f"{E['torch']} / {E['transformers']}"),("SEED",str(SEED)),
           ("DEMO PANEL LOCK",P["engine"]["lock_sha256"][:24]+"…"),("FROZEN WEIGHTS","✓ PASS" if I["result"]=="PASS" else "✗ FAIL"),("ERRORS",str(len(P["errors"]))),
           ("RUN TIME",f"{T['run']['engine_seconds']:.1f} s engine"),("CHECKS",f"{I['checks_total']} pre-seal PASS")]
    for i,(a,b) in enumerate(tiles):
        r,q=divmod(i,5);x=.05+q*.182;y=top-.12-r*.125
        fig.add_artist(FancyBboxPatch((x,y),.172,.105,boxstyle="round,pad=0,rounding_size=0.01",transform=fig.transFigure,facecolor=C_BG,edgecolor=C_MOD,lw=1.8))
        fig.text(x+.008,y+.08,a,fontsize=10,weight="bold",color=C_MOD,va="center");fit_text(fig,x+.008,y+.008,.156,.058,b,fs_max=12.5,fs_min=7.5,family=MONO)
    st=[("model load (startup)",T["startup"]["model_load_seconds"]),("codebook",P["codebook"]["seconds"]),("16 cartridges",T["run"]["cartridges_seconds"]),
        ("question battery",T["run"]["battery_seconds"]),("NOMEM control",T["run"]["control_seconds"]),("sealing",ctx["sealing_stage_seconds"])]
    ax=fig.add_axes([.25,.38,.68,top-.52]);ax.barh(range(len(st)),[s for _,s in st],color=[C_CTL,C_CART,C_CART,C_OK,C_CTL,C_FG]);ax.invert_yaxis()
    ax.set_yticks(range(len(st)));ax.set_yticklabels([a for a,_ in st],fontsize=11)
    for i,(_,s) in enumerate(st):ax.text(s,i,f" {s:.2f} s",va="center",fontsize=11,weight="bold")
    ax.set_xlim(0,max(s for _,s in st)*1.25);ax.set_xlabel("seconds",fontsize=10);clean(ax)
    txt=(f"PAYLOAD FILE : {ctx['payload_name']}\nPAYLOAD SHA-256 : {ctx['sha']}\nENGINE SOURCE : TEST461 IBR engine (463.final.py), unchanged\n"
         f"JPEG HASHES : images manifest (each image SHA-256), sealed after rendering\n"
         "Artifact integrity seal ≠ scientific proof: the SHA-256 shows the files are byte-identical to what this run recorded; the scientific claims rest on the measurements and controls.")
    fit_text(fig,.05,.05,.9,.27,txt,fs_max=12.5,fs_min=8,family=MONO)
    foot(fig,ctx,k);return save_jpg(fig,path)
POSTERS=[("01_what_just_happened","What just happened?",p01_what),("02_inside_the_engine","Inside the engine",p02_engine),
 ("03_cartridge_bank","The cartridge bank · 16 loaded",p03_bank),("04_engine_workload","How hard did the engine work?",p04_work),
 ("05_source_to_cartridge","Source → cartridge encoding",p05_compress),("06_cartridge_telemetry","Cartridge telemetry",p06_telemetry),
 ("07_source_removed","Source removal proof",p07_source),("08_frozen_model","Frozen model proof",p08_frozen),
 ("09_question_battery","Automatic question battery",p09_battery),("10_linked_retrieval","Linked retrieval",p10_linked),
 ("11_missing_link_unknown","Missing link → UNKNOWN",p11_unknown),("12_false_link_control","False-link control",p12_falselink),
 ("13_scale","Scale 2 → 8 → 16",p13_scale),("14_where_it_worked","Where it worked — and where it didn't",p14_family),
 ("15_f2_boundary","A real boundary condition",p15_f2),("16_why_ibr","Why IBR exists",p16_ibr),
 ("17_batch_equivalence","Batch semantic equivalence",p17_batch),("18_evidence_seal","Complete run · evidence seal",p18_seal)]
NPOST=len(POSTERS)
if NPOST>20 or [s[:2] for s,_,_ in POSTERS]!=[f"{i:02d}" for i in range(1,NPOST+1)]:raise RuntimeError("Poster count/numbering check failed.")
#<<POSTERS_END>>
