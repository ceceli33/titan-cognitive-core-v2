# ------------------------------------------------------------------ 06 all questions
def p06(ctx,k,path):
    P=ctx["P"];R=ctx["R"];C,T=ctx_counts(ctx);fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"ALL 16 QUESTIONS · 16 × 16 ISOLATED READS","Row = question, column = cartridge. Green: the cartridge that holds the answer returned it. Purple: NONE.",tag="THIS LIVE RUN")
    cmap=ListedColormap(["#E9D5FF",C_OK,C_ERR,"#F97316","#E2E8F0"]);code={"none":0,"gold":1,"gold_miss":2,"other":3}
    for n,(key,kind,dec,exp,title)in enumerate((("mw1","id",R["mw1"]["decided"],IDS,"MW1 · object → container ID"),("mw2","place",R["mw2"]["decided"],PLACES,"MW2 · container ID → location"))):
        x0=.14+n*.45;ax=fig.add_axes([x0,.22,.28,top-.30]);M=np.array([[code[c]for c in row]if row else[4]*16 for row in R[key]["cls"]])
        ax.imshow(M,aspect="auto",cmap=cmap,vmin=0,vmax=4);ax.set_xticks(range(16));ax.set_xticklabels([f"C{j+1:02d}"for j in range(16)],rotation=90,fontsize=8.2)
        ax.set_yticks(range(16));ax.set_yticklabels([f"Q{i+1:02d} "+(OBJECTS[i]if n==0 else IDS[i])for i in range(16)]if n==0 else[f"Q{i+1:02d} {IDS[i]}"for i in range(16)],fontsize=8.8)
        ax.set_xticks(np.arange(-.5,16,1),minor=True);ax.set_yticks(np.arange(-.5,16,1),minor=True);ax.grid(which="minor",color="white",lw=1.2);ax.tick_params(which="minor",length=0)
        ax.set_title(title,fontsize=12.5,weight="bold",loc="left")
        for i in range(16):
            ok=dec[i]==exp[i];ax.text(16.1,i,f"{dec[i]or'UNKNOWN'} {'✓'if ok else'✗'}",fontsize=8.5,va="center",color=C_OK if ok else C_ERR,weight="bold",clip_on=False)
    w1=R["mw1"];w2=R["mw2"]
    lg=[Line2D([0],[0],marker="s",color="w",markerfacecolor=C_OK,markersize=11,label="gold cartridge returned the answer"),Line2D([0],[0],marker="s",color="w",markerfacecolor="#E9D5FF",markeredgecolor=C_UNK,markersize=11,label="NONE"),Line2D([0],[0],marker="s",color="w",markerfacecolor="#F97316",markersize=11,label="non-gold cartridge returned something else"),Line2D([0],[0],marker="s",color="w",markerfacecolor=C_ERR,markersize=11,label="gold cartridge missed"),Line2D([0],[0],marker="s",color="w",markerfacecolor="#E2E8F0",markeredgecolor=C_CTL,markersize=11,label="MW2 not run (MW1 gave no unique ID)")]
    fig.legend(handles=lg,loc="lower center",bbox_to_anchor=(.5,.075),ncol=3,frameon=False,fontsize=9.4)
    fig.text(.05,.045,f"LIVE · MW1 {kn(C,T,'mw1')} · MW2 {kn(C,T,'mw2')} · wrong-cartridge NONE: stage 1 {w1['wrong_none']}/{w1['wrong_total']} · stage 2 {w2['wrong_none']}/{w2['wrong_total']}",fontsize=12,weight="bold",color=C_FG)
    foot(fig,ctx,k);return finish(fig,path,ctx,[kn(C,T,"mw1"),kn(C,T,"mw2"),f"{w1['wrong_none']}/{w1['wrong_total']}",f"{w2['wrong_none']}/{w2['wrong_total']}"])
# ------------------------------------------------------------------ 07 results
def p07(ctx,k,path):
    P=ctx["P"];R=ctx["R"];C,T=ctx_counts(ctx);S=SEALED474;fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"RESULTS · THIS RUN vs THE SEALED TEST474 RECORD","Same panel, same engine, same gates. The sealed record is historical; the live column was measured just now.")
    fig.text(.40,top-.015,"THIS LIVE RUN",fontsize=13,weight="bold",color=C_MOD,ha="center");fig.text(.64,top-.015,"SEALED TEST474",fontsize=13,weight="bold",color=C_SEAL,ha="center");fig.text(.84,top-.015,"GATE (≥)",fontsize=13,weight="bold",color=C_NEU,ha="center")
    keys=list(LBL);rh=(top-.30)/(len(keys)+2)
    for i,key in enumerate(keys):
        y=top-.06-(i+1)*rh;live=C[key];tot=T[key];good=live>=GATE_MIN[key]
        fig.text(.05,y+rh*.4,LBL[key],fontsize=15,weight="bold",color=C_FG,va="center")
        ax=fig.add_axes([.27,y+rh*.12,.26,rh*.55]);ax.barh([0],[tot],color="#E2E8F0");ax.barh([0],[live],color=C_OK if good else C_ERR);ax.set_xlim(0,tot);ax.axis("off");ax.text(tot/2,0,kn(C,T,key),ha="center",va="center",fontsize=15,weight="bold",color="white")
        sv=S[key];ax2=fig.add_axes([.55,y+rh*.12,.18,rh*.55]);ax2.barh([0],[sv[1]],color="#E2E8F0");ax2.barh([0],[sv[0]],color=C_SEAL);ax2.set_xlim(0,sv[1]);ax2.axis("off");ax2.text(sv[1]/2,0,fr(sv),ha="center",va="center",fontsize=14,weight="bold",color="white")
        fig.text(.84,y+rh*.4,f"{GATE_MIN[key]}/{tot}",fontsize=13,color=C_NEU,ha="center",va="center")
    y=top-.06-(len(keys)+1)*rh
    fig.text(.05,y+rh*.4,"WEIGHT SENTINEL",fontsize=15,weight="bold",color=C_FG,va="center");fz=P["frozen"];sp=fz["sentinel_startup"]==fz["sentinel_after"]
    fig.text(.40,y+rh*.4,"PASS"if sp else"FAIL",fontsize=17,weight="bold",color=C_OK if sp else C_ERR,ha="center",va="center");fig.text(.64,y+rh*.4,S["sentinel"],fontsize=17,weight="bold",color=C_SEAL,ha="center",va="center")
    ok=R["verdict"].startswith("PASS");fig.add_artist(FancyBboxPatch((.05,.105),.90,.075,boxstyle="round,pad=0,rounding_size=0.01",transform=fig.transFigure,facecolor=C_OKL if ok else C_ERRL,edgecolor=C_OK if ok else C_ERR,lw=2.6))
    fig.text(.07,.1425,"LIVE VERDICT",fontsize=12,weight="bold",color=C_NEU,va="center");fig.text(.25,.1425,R["verdict"],fontsize=17,weight="bold",color=C_OK if ok else C_ERR,va="center",family=MONO)
    fig.text(.93,.1425,f"sealed: {S['verdict']}",fontsize=10.5,color=C_SEAL,va="center",ha="right",family=MONO)
    fit_text(fig,.05,.045,.9,.05,f"Replay identical to the sealed counts: {'YES'if R['replay_match']else'NO'}. Gate thresholds are the ones precommitted in TEST474. NOMEM passes when the model without cartridges does not output the stored place.",fs_max=11,fs_min=8.5,color=C_NEU)
    foot(fig,ctx,k);return finish(fig,path,ctx,[kn(C,T,x)for x in keys]+[R["verdict"],S["verdict"]])
# ------------------------------------------------------------------ 08 no router
def p08(ctx,k,path):
    P=ctx["P"];R=ctx["R"];w1=R["mw1"];w2=R["mw2"];Q=SEALED473;fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"NO ROUTER · THE OTHER 15 CARTRIDGES SAY NONE","Nothing chooses a cartridge in advance. Every question goes to all 16; only the right one answers.",tag="THIS LIVE RUN")
    box(fig,.05,top-.20,.16,.13,"QUESTION","one object or ID",C_FG,C_BG,tfs=12,bfs=10.5);arrow(fig,.21,top-.135,.265,top-.135)
    for i in range(16):
        r,q=divmod(i,4);x=.28+q*.032;y=top-.075-r*.034;fig.add_artist(FancyBboxPatch((x,y-.026),.027,.027,boxstyle="round,pad=0,rounding_size=0.003",transform=fig.transFigure,facecolor=C_CARTL,edgecolor=C_CART,lw=1.4));fig.text(x+.0135,y-.0125,f"{i+1:02d}",fontsize=7.5,ha="center",va="center",color=C_CART)
    fig.text(.34,top-.225,"same question copied to all 16",ha="center",fontsize=10,color=C_NEU);arrow(fig,.425,top-.135,.475,top-.135)
    fig.add_artist(FancyBboxPatch((.48,top-.20),.027,.027,boxstyle="round,pad=0,rounding_size=0.003",transform=fig.transFigure,facecolor=C_OK,edgecolor=C_OK));fig.text(.52,top-.1865,"1 cartridge returns the answer",fontsize=11,va="center",color=C_OK,weight="bold")
    fig.add_artist(FancyBboxPatch((.48,top-.15),.027,.027,boxstyle="round,pad=0,rounding_size=0.003",transform=fig.transFigure,facecolor=C_UNKL,edgecolor=C_UNK));fig.text(.52,top-.1365,"15 cartridges return NONE",fontsize=11,va="center",color=C_UNK,weight="bold")
    arrow(fig,.74,top-.135,.785,top-.135);box(fig,.79,top-.20,.16,.13,"UNIQUE","aggregation\nof the answers",C_OK,C_OKL,tfs=12,bfs=10.5)
    ax=fig.add_axes([.22,.29,.70,.26]);labs=["LIVE · stage 1\nobject → ID reads","LIVE · stage 2\nID → location reads","SEALED TEST473\nstage-1 reads"]
    vals=[(w1["wrong_none"],w1["wrong_total"]),(w2["wrong_none"],w2["wrong_total"]),Q["decoy_none"]];cols=[C_MOD,C_MOD,C_SEAL]
    ax.barh(range(3),[v[1]for v in vals],color="#E2E8F0");ax.barh(range(3),[v[0]for v in vals],color=cols);ax.invert_yaxis();ax.set_yticks(range(3));ax.set_yticklabels(labs,fontsize=11)
    for i,v in enumerate(vals):ax.text(v[1]*.5,i,f"{v[0]}/{v[1]} wrong-cartridge reads returned NONE",ha="center",va="center",fontsize=12.5,weight="bold",color="white")
    ax.set_xlim(0,max(v[1]for v in vals));ax.set_xlabel("reads of a cartridge that does NOT hold the asked item",fontsize=10,labelpad=3);clean_ax(ax)
    note(fig,.05,.04,.9,.115,f"Live check: reads per question = {R['reads_per_query'][0]} for every question, so no cartridge was pre-selected. The sealed TEST473 count (16 questions × 15 wrong cartridges = 240) was measured in a different run and is shown for comparison only; it is not added to the live count.",
         "Router: none. Hidden scorer: none. A wrong-cartridge read counts as NONE when its first line contains NONE and no container ID (is_none, TEST473/474).")
    foot(fig,ctx,k);return finish(fig,path,ctx,[f"{w1['wrong_none']}/{w1['wrong_total']}",f"{w2['wrong_none']}/{w2['wrong_total']}",fr(Q["decoy_none"])])
# ------------------------------------------------------------------ 09 not in the cartridges
def p09(ctx,k,path):
    P=ctx["P"];R=ctx["R"];C,T=ctx_counts(ctx);fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"WHEN THE ANSWER IS NOT IN THE CARTRIDGES","Missing object and absent ID: all 16 cartridges must say NONE. NOMEM: the same question with no cartridge at all.",tag="THIS LIVE RUN")
    for n,(key,title,good)in enumerate((("missing",f"MISSING OBJECT · {kn(C,T,'missing')} → UNKNOWN",C["missing"]),("absent",f"ABSENT CONTAINER ID · {kn(C,T,'absent_id')} → UNKNOWN",C["absent_id"]))):
        ax=fig.add_axes([.12+n*.45,top-.315,.30,.225]);rows=R[key];nn=[m["none"]for m in rows];nt=[m["n"]for m in rows]
        ax.barh(range(8),nt,color=C_ERRL);ax.barh(range(8),nn,color=C_UNK);ax.invert_yaxis();ax.set_yticks(range(8));ax.set_yticklabels([m["query"]for m in rows],fontsize=9.5)
        for i,m in enumerate(rows):ax.text(m["n"]+.3,i,("✓ UNKNOWN"if m["ok"]else"✗ answered"),va="center",fontsize=9.5,weight="bold",color=C_OK if m["ok"]else C_ERR)
        ax.set_xlim(0,24);ax.set_xlabel("cartridges returning NONE (of 16)",fontsize=10);ax.set_title(title,fontsize=12,weight="bold",loc="left");clean_ax(ax)
    ax=fig.add_axes([.05,.13,.90,.285]);ax.axis("off");ax.set_xlim(0,1);ax.set_ylim(0,1)
    ax.text(0,1.0,f"NOMEM CONTROL · {kn(C,T,'nomem')} · same Stage-2 question, no cartridge, nothing installed",fontsize=12,weight="bold",color=C_CTL,va="top")
    ax.text(0,.88,"success = the model does NOT output the stored place (TEST474 criterion)",fontsize=9.5,color=C_NEU,va="top",style="italic")
    for c_,x in(("container ID",.0),("stored place (hidden from the model)",.17),("model's first line without cartridge",.47),("result",.82)):ax.text(x,.76,c_,fontsize=9.5,weight="bold",color=C_NEU,va="center")
    for i,m in enumerate(R["nomem"]):
        y=.67-i*.088;ax.text(0,y,m["id"],fontsize=9.5,family=MONO,va="center");ax.text(.17,y,m["target"],fontsize=9.5,va="center");ax.text(.47,y,sc(m["first"])if m["first"]else"(empty)",fontsize=9.5,va="center",family=MONO)
        ax.text(.82,y,"PASS"if m["ok"]else"FAIL",fontsize=9.5,weight="bold",color=C_OK if m["ok"]else C_ERR,va="center")
    fit_text(fig,.05,.045,.9,.07,"Missing and absent questions ask about things that were never stored, to every cartridge. NOMEM shows the correct answers do not come from the model's own knowledge of these invented records.",fs_max=11,fs_min=8.5,color=C_NEU)
    foot(fig,ctx,k);return finish(fig,path,ctx,[kn(C,T,"missing"),kn(C,T,"absent_id"),kn(C,T,"nomem")])
# ------------------------------------------------------------------ 10 layer corridor (sealed)
def p10(ctx,k,path):
    S=SEALED467;fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"WHICH LAYERS NEED THE CARTRIDGE?","Layer-range ablation on a D120 cartridge. Sealed TEST467 record on its own 8-case panel — not measured in this run.",tag="SEALED HISTORICAL RECORD · TEST467")
    cols={"K":C_CART,"V":C_MOD,"KV":C_OK}
    for n,(key,title)in enumerate((("entry","ENTRY SCAN · layers L07–L15"),("exit","EXIT SCAN · layers L18–L27"))):
        ax=fig.add_axes([.07+n*.47,.38,.40,top-.50]);data=S[key];xs=list(data.keys())
        for j,tn in enumerate(("K","V","KV")):ax.plot(range(len(xs)),[data[x][j]for x in xs],"o-",lw=2.4,ms=7,color=cols[tn],label=tn)
        ax.set_xticks(range(len(xs)));ax.set_xticklabels(xs,fontsize=9.5,rotation=45);ax.set_ylim(-.3,8.5);ax.set_yticks(range(0,9,2));ax.set_ylabel("hits out of 8",fontsize=10.5);ax.set_title(title,fontsize=12.5,weight="bold",loc="left");ax.grid(alpha=.25);clean_ax(ax)
        if n==0:ax.legend(frameon=False,fontsize=10.5,loc="lower left");ax.annotate("KV: 8/8 → 0/8\nbetween L12 and L13",xy=(6,0),xytext=(2.3,2.2),fontsize=10.5,weight="bold",color=C_OK,arrowprops=dict(arrowstyle="->",color=C_OK,lw=1.8))
        else:ax.text(4.6,1.2,"K and KV reach 8/8 at L23\nV reaches 8/8 at L26",fontsize=10.5,weight="bold",color=C_OK,va="center")
    ax=fig.add_axes([.07,.185,.86,.12]);ax.axis("off");ax.set_xlim(0,1);ax.set_ylim(0,1)
    for c_,x in(("tensor",.0),("last full entry-scan layer",.18),("first degraded layer",.44),("first full exit-scan layer",.68)):ax.text(x,.92,c_,fontsize=10,weight="bold",color=C_NEU,va="center")
    for i,tn in enumerate(("K","V","KV")):
        b=S["bounds"][tn];y=.62-i*.26;ax.text(0,y,tn,fontsize=11,weight="bold",color=cols[tn],va="center");ax.text(.18,y,b[0],fontsize=11,va="center",family=MONO);ax.text(.44,y,b[1],fontsize=11,va="center",family=MONO);ax.text(.68,y,b[2],fontsize=11,va="center",family=MONO)
    fit_text(fig,.05,.045,.9,.12,f"TEST467 used {S['cases']} cases with a D120 baseline of {fr(S['baseline'])}; it is a different panel from this demo. Read as functional sufficiency under controlled cumulative ablation: it does not show that the information lives only at L12–L13, and it was not re-measured in this run.",fs_max=11.5,fs_min=8.5,color=C_NEU)
    foot(fig,ctx,k);return finish(fig,path,ctx,["SEALED HISTORICAL RECORD · TEST467","L12","L27"])
# ------------------------------------------------------------------ 11 audit
def p11(ctx,k,path):
    P=ctx["P"];R=ctx["R"];fz=P["frozen"];S=P["source_removal"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"AUDIT LOCKS · FROZEN MODEL","Each lock is marked CHECKED (verified in this run) or BY CONSTRUCTION (a property of the code path).",tag="THIS LIVE RUN")
    ax=fig.add_axes([.05,top-.37,.34,.33]);ax.axis("off");ax.set_xlim(0,1);ax.set_ylim(0,1);ax.text(0,1,"WEIGHT SENTINEL · 3 CHECKPOINTS",fontsize=11.5,weight="bold",color=C_MOD,va="top")
    for j,(lab,key)in enumerate((("startup","sentinel_startup"),("pre-run","sentinel_pre_run"),("post-run","sentinel_after"))):
        y=.80-j*.17;same=fz[key]==fz["sentinel_startup"];ax.text(0,y,lab,fontsize=10.5,weight="bold",va="center");ax.text(.22,y,fz[key][:28]+"…",fontsize=8.8,family=MONO,va="center",color=C_FG);ax.text(1.0,y,"✓ identical"if same else"✗ CHANGED",fontsize=10,weight="bold",color=C_OK if same else C_ERR,va="center",ha="right")
    same_fp=fz["fingerprint_startup"]==fz["fingerprint_after"];ax.text(0,.27,f"float32 sums of the same {len(fz['tensors'])} tensors: {'identical'if same_fp else'CHANGED'}",fontsize=10,va="center",color=C_OK if same_fp else C_ERR)
    ax.text(0,.14,f"trainable tensors {fz['trainable_tensors']} · LoRA {'none'if not fz['lora']else'PRESENT'} · optimizer {'none'if not fz['optimizer']else'PRESENT'}",fontsize=10,va="center");ax.text(0,.02,f"sampled sentinel ({len(fz['tensors'])} tensors × 16 slices × 256 values), not a hash of all weights",fontsize=8.6,va="center",color=C_NEU,style="italic")
    locks=[("MODEL WEIGHTS","FROZEN","CHECKED","sentinel + fingerprint unchanged, 0 trainable tensors"),("WEIGHT SENTINEL","PASS" if fz["sentinel_startup"]==fz["sentinel_after"] else "FAIL","CHECKED","3 checkpoints identical (left)"),
           ("SOURCE AT READOUT","ABSENT","CHECKED",f"{S['prompts_checked']} prompts audited, 0 source records found"),("CARTRIDGE","NUMERICAL K/V","BY CONSTRUCTION","tensors only; no text is stored"),
           ("K / V","D120 / D120","CHECKED","both projected on 120 directions"),("OWN","PRESERVED","CHECKED",f"first-token K/V identical in {len(P['cartridges'])} cartridges × {P['model']['arch'][0]} layers"),
           ("CARTRIDGES","INDEPENDENT","CHECKED",f"{len(P['cartridges'])} separate forges, {2*len(P['cartridges'])} distinct tensor storages"),("READOUT","MODULE-WISE","CHECKED",f"{R['reads_per_query'][0]} reads per question"),
           ("AGGREGATION","UNIQUE","BY CONSTRUCTION","TEST474 decide_id / decide_place"),("ROUTER","NONE","CHECKED","every question read by all 16 cartridges"),
           ("TRAINING · LoRA · OPTIMIZER","NONE","CHECKED","no grads, no adapter, no optimizer object"),("WEIGHT UPDATE","NONE","CHECKED",f"sentinel unchanged after {P['counters']['generate_calls']} readout reads"),("GENERATION","GREEDY","BY CONSTRUCTION","do_sample = False"+("; stops after the first answer line (probe-checked)"if P["engine"].get("early_stop")else""))]
    ax2=fig.add_axes([.43,.10,.52,top-.12]);ax2.axis("off");ax2.set_xlim(0,1);ax2.set_ylim(0,1);rh=1/len(locks)
    for i,(a,b,c,d)in enumerate(locks):
        y=1-(i+.5)*rh;col=C_OK if c=="CHECKED"else C_CTL
        ax2.add_patch(FancyBboxPatch((0,y-rh*.42),1,rh*.84,boxstyle="round,pad=0,rounding_size=0.01",transform=ax2.transAxes,facecolor="white",edgecolor="#CBD5E1",lw=1.2))
        ax2.text(.015,y+rh*.12,a,fontsize=9.6,weight="bold",color=C_FG,va="center");ax2.text(.015,y-rh*.22,d,fontsize=8.3,color=C_NEU,va="center");ax2.text(.60,y,b,fontsize=11,weight="bold",color=C_OK if b not in("FAIL",)else C_ERR,va="center",family=MONO)
        ax2.text(.985,y,c,fontsize=8.8,weight="bold",color="white",va="center",ha="right",bbox=dict(boxstyle="round,pad=0.25",fc=col,ec=col))
    fit_text(fig,.05,.045,.34,.20,f"{len(P['checks_pre_seal'])} technical checks passed before sealing. Compute path: PyTorch CUDA backend, no custom CUDA/C++ kernel in this engine.",fs_max=11,fs_min=8.5,color=C_NEU)
    foot(fig,ctx,k);return finish(fig,path,ctx,["FROZEN","ABSENT","UNIQUE","NONE","GREEDY",str(S["prompts_checked"])])
# ------------------------------------------------------------------ 12 limits
def p12(ctx,k,path):
    P=ctx["P"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"WHAT THIS DEMO DOES — AND DOES NOT — SHOW","Read together with the results. These limits come from the experiments themselves.")
    shows=["A frozen Mistral-7B-Instruct-v0.3 recovered facts from D120 numerical K/V cartridges after the source text was removed from readout.","Sixteen independent cartridges were read module-wise; the unique-aggregation rule separated the one cartridge that holds an item from fifteen that do not.","Missing objects and absent IDs ended in UNKNOWN; the no-cartridge control did not reproduce the stored places.","The sealed TEST474 result was replayed live on the same panel."]
    nots=["Not a new held-out validation: the 16 records are the same ones used in TEST473 (where the frame was chosen) and TEST474.","Not evidence for other models, other record formats, or other fact types: records are short and synthetic.","Not evidence beyond 16 cartridges, and not a claim of unlimited or general memory.","Not training: the model keeps no permanent memory; cartridges are installed only for a question."]
    for n,(title,items,col,fc)in enumerate((("THIS RUN SHOWS",shows,C_OK,C_OKL),("THIS RUN DOES NOT SHOW",nots,C_ERR,C_ERRL))):
        x=.05+n*.465;fig.add_artist(FancyBboxPatch((x,top-.34),.435,.32,boxstyle="round,pad=0,rounding_size=0.01",transform=fig.transFigure,facecolor=fc,edgecolor=col,lw=2.3));fig.text(x+.015,top-.05,title,fontsize=13.5,weight="bold",color=col,va="center")
        for i,t in enumerate(items):fit_text(fig,x+.015,top-.175-i*.06,.405,.056,"• "+t,fs_max=11,fs_min=8.5)
    fig.text(.05,.445,"HOW THE SAME 16 RECORDS WERE USED",fontsize=13,weight="bold",color=C_NEU,va="center")
    ys=.275;bw,bh=.27,.085;items=[("TEST473",f"12 frame conditions compared on the 16 records; {SEALED473['tied_perfect']} tied at GOLD 16/16 and 240/240",C_SEAL,C_SEALL),("TEST474","S1_RECORD + C_VERIFY (chosen by fixed order among the 4) replayed on the same 16 records, plus new negative controls",C_SEAL,C_SEALL),("THIS DEMO","TEST474 panel replayed again, live: reproducibility, not new evidence",C_MOD,C_MODL)]
    for i,(a,b,c,fc)in enumerate(items):
        x=.05+i*(bw+.0425);fig.add_artist(FancyBboxPatch((x,ys),bw,bh+.04,boxstyle="round,pad=0,rounding_size=0.01",transform=fig.transFigure,facecolor=fc,edgecolor=c,lw=2.2));fig.text(x+.012,ys+bh+.022,a,fontsize=12,weight="bold",color=c,va="center");fit_text(fig,x+.012,ys+.006,bw-.024,bh-.012,b,fs_max=10.2,fs_min=8)
        if i<2:arrow(fig,x+bw,ys+(bh+.04)/2,x+bw+.0425,ys+(bh+.04)/2,color=C_NEU)
    note(fig,.05,.10,.9,.11,"Cartridge size is about the model's own K/V size (D120 keeps 120 of 128 directions per head), so this is not a storage-saving claim. Layer results on poster 10 come from a different test and were not re-measured here.")
    foot(fig,ctx,k);return finish(fig,path,ctx,["THIS RUN SHOWS","THIS RUN DOES NOT SHOW","TEST473","TEST474"])
# ------------------------------------------------------------------ 13 evidence seal
def p13(ctx,k,path):
    P=ctx["P"];R=ctx["R"];E_=P["environment"];tm=P["timing"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white");cnt=P["counters"];B=P["bank"]
    top=head(fig,"COMPLETE RUN · EVIDENCE SEAL","Everything below was read from the runtime and is contained in the sealed payload.",tag="THIS LIVE RUN")
    tiles=[("RUN ID",P["run_id"]),("MODEL",MODEL_ID),("GPU",E_["gpu"]),("TORCH / TRANSFORMERS",f"{E_['torch']} / {E_['transformers']}"),("READOUT READS",str(cnt["generate_calls"])),
           ("CARTRIDGES · SLOTS",f"16 · {B['slots']}"),("PANEL LOCK",LOCK_SHA[:22]+"…"),("TECHNICAL CHECKS",f"{len(P['checks_pre_seal'])} passed"),("PEAK GPU MEMORY",f"{P['gpu']['peak_allocated_gib']:.2f} GiB"),("VERDICT",R["verdict"])]
    for i,(a,b)in enumerate(tiles):
        r,q=divmod(i,5);x=.05+q*.182;y=top-.125-r*.125;fig.add_artist(FancyBboxPatch((x,y),.172,.105,boxstyle="round,pad=0,rounding_size=0.01",transform=fig.transFigure,facecolor=C_BG,edgecolor=C_MOD,lw=1.8))
        fig.text(x+.008,y+.083,a,fontsize=9.5,weight="bold",color=C_MOD,va="center");fit_text(fig,x+.008,y+.008,.156,.058,b,fs_max=12,fs_min=7.5,family=MONO)
    st=[("model load (startup)",tm["model_load_seconds"]),("codebook",tm["codebook"]),("16 cartridges",tm["forge"]),("MW1 readout",tm["mw1"]),("MW2 readout",tm["mw2"]),("missing + absent",tm["negatives"]),("NOMEM",tm["nomem"])]
    ax=fig.add_axes([.22,.30,.70,top-.55]);ax.barh(range(len(st)),[s for _,s in st],color=[C_CTL,C_CART,C_CART,C_MOD,C_MOD,C_UNK,C_CTL]);ax.invert_yaxis();ax.set_yticks(range(len(st)));ax.set_yticklabels([a for a,_ in st],fontsize=10.5)
    for i,(_,s)in enumerate(st):ax.text(s,i,f" {s:.1f} s",va="center",fontsize=10.5,weight="bold")
    ax.set_xlim(0,max(s for _,s in st)*1.18);ax.set_xlabel("measured seconds",fontsize=10);clean_ax(ax)
    txt=f"PAYLOAD FILE : {ctx['payload_name']}\nPAYLOAD SHA-256 : {ctx['sha']}\nENGINE : TEST474 (474.MISTRAL.final.py) engine functions, unchanged\nIMAGE HASHES : images manifest (per-image SHA-256), written after rendering\nArtifact integrity seal ≠ scientific proof: it shows the files are byte-identical to what this run recorded."
    fit_text(fig,.05,.05,.9,.215,txt,fs_max=11.5,fs_min=8,family=MONO)
    foot(fig,ctx,k);return finish(fig,path,ctx,[P["run_id"],E_["gpu"],E_["torch"],E_["transformers"],R["verdict"],str(cnt["generate_calls"])])
POSTERS=[("01_pipeline","What just happened?",p01),("02_cartridge_anatomy","What is inside a cartridge?",p02),("03_source_removal","Was the source removed?",p03),("04_cartridge_bank","The cartridge bank",p04),
 ("05_one_question","One question, step by step",p05),("06_all_questions","All 16 questions",p06),("07_results","Results: live vs sealed",p07),("08_no_router","No router",p08),
 ("09_not_in_cartridges","When the answer is not stored",p09),("10_layer_corridor","Layer corridor (sealed record)",p10),("11_audit_locks","Audit locks and frozen model",p11),("12_limits","What this demo does not show",p12),("13_evidence_seal","Evidence seal",p13)]
N_POSTERS=len(POSTERS)
if N_POSTERS>20 or[s[:2]for s,_,_ in POSTERS]!=[f"{i:02d}"for i in range(1,N_POSTERS+1)]:raise RuntimeError("Poster count/numbering check failed.")
def render_all(ctx,run_dir):
    imgs=[]
    try:
        for i,(slug,cap,fn)in enumerate(POSTERS,1):
            p=Path(run_dir)/f"{slug}_{ctx['run_id']}.jpg";fn(ctx,i,p);imgs.append((p,cap))
    finally:plt.close("all")
    return imgs
#<<POSTERS_END>>
#<<UI_BEGIN>>
class SelfTestEngine:
    """Synthetic engine used ONLY by the service self-test (no GPU, no model). Never used for a real run."""
    def __init__(s,mode="ok"):
        s.counters=Counter();s.mode=mode;s.n=0;s.rng=np.random.default_rng(1);s.sent="a"*64;s.fp=[1.0]*7
        s.info=dict(model_id=MODEL_ID,arch=ARCH,dtype="bfloat16",attn="sdpa",gpu="SELF-TEST ENGINE (no GPU)",gpu_total_gib=0.0,torch="n/a",transformers="n/a",gradio="n/a",python="n/a",platform="n/a",params=1,pad_id=0,eos_ids=[0],
            sliding_window=None,fp_names=["t0","t1","t2","t3","t4","t5","t6"],model_load_seconds=0.0,startup_utc="n/a",sentinel0=s.sent,fingerprint0=s.fp,hooks0=5,fast_first_line=True,sentinel_method="synthetic")
    def frozen_state(s):
        fh=3 if s.mode=="foreign_hook"else 0;return dict(training=False,trainable_tensors=0,lora=False,optimizer=False,hooks=5+fh,foreign_hooks=fh,foreign_modules=({"__main__":fh}if fh else{}),sentinel=("b"*64 if s.mode=="sentinel_changes"and s.n>50 else s.sent),fingerprint=s.fp)
    def build_codebook(s):s.counters["forge_passes"]+=32;return dict(sentences=32,svds=512,seconds=0.0,bytes=1)
    def codebook_bytes(s):return 1
    def forge_card(s,i):
        s.counters["forge_passes"]+=1;T=28+i%6;r=s.rng
        tel=dict(cosK=list(.985+.014*r.random(32)),cosV=list(.992+.007*r.random(32)),own_exact=True,T=T,encode_seconds=0.0,code_numbers=2*32*(T-1)*8*120,own_numbers=2*32*1024,native_numbers=2*32*1024*T,codec_macs=1,gpu_mib=0.0)
        if i==0:tel["excerpt"]=dict(layer=16,head=0,tensor="K",values=(r.normal(0,1,(T-1,120))*np.exp(-np.arange(120)/60)).round(4).tolist())
        return dict(rec=(OBJECTS[i],IDS[i],PLACES[i]),T=T,ptr=1000+i),tel
    def card_ptrs(s,c):return[c["ptr"]*2,c["ptr"]*2+1]
    def card_finite(s,c):return True
    def q_tokens(s,p):return len(p.split())
    def gen(s,c,p):
        s.counters["generate_calls"]+=1;s.n+=1;o,i,pl=c["rec"]
        if'object "'in p:return(i+"\nsynthetic")if re.search(r'object "(.+?)"',p).group(1)==o else"NONE\nsynthetic"
        return(pl+"\nsynthetic")if re.search(r'container "(.+?)"',p).group(1)==i else"NONE\nsynthetic"
    def gen_nomem(s,p):s.counters["generate_calls"]+=1;return"NONE"
    def gen_full(s,c,p):
        s.counters["probe_full_decodes"]+=1;o=s.gen(c,p);return o+"\nfull-length tail"if s.mode!="probe_mismatch"else"NONE\nx"if o.startswith("RQ")else"RQ-000"
    def gen_nomem_full(s,p):s.counters["probe_full_decodes"]+=1;return"NONE\nfull-length tail"
    def sync(s):pass
    def reset_peak(s):pass
    def peak_gib(s):return 0.0
def drain(gen):
    while True:
        try:next(gen)
        except StopIteration as s:return s.value
RUN_COUNTER=0;GPU_LOCK=threading.Lock()
SKIP=getattr(gr,"skip",None)or gr.update
_K=object()
FILE_KEYS=["zip","json","txt","man"]
DL_LABELS=["⬇ DOWNLOAD EVERYTHING (.zip)","⬇ DOWNLOAD FULL RUN LOG (.json)","⬇ DOWNLOAD READABLE RUN LOG (.txt)","⬇ DOWNLOAD RUN MANIFEST (.json)"]
DL_ALL_LABEL=f"⬇ DOWNLOAD ALL {N_POSTERS} JPGs"
RAW_KEYS=["txt","json","man"]
N_OUTPUTS=2+2+len(FILE_KEYS)*2+len(RAW_KEYS)
try:
    from gradio.routes import API_PREFIX as _GR_API_PREFIX
except Exception:
    _GR_API_PREFIX="/gradio_api"if int(gr.__version__.split(".")[0])>=5 else""
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
        paths=[files.get(k)for k in FILE_KEYS]
        for pth in paths:
            if pth is not None and not file_ready(pth):raise RuntimeError(f"Download artifact missing or empty: {pth}")
        out+=[dl_update(p,l)for p,l in zip(paths,DL_LABELS)]+[None if p is None else str(p)for p in paths]
    if raw is _K:out+=[SKIP()for _ in RAW_KEYS]
    elif raw is None:out+=[""]*len(RAW_KEYS)
    else:out+=[raw[k]for k in RAW_KEYS]
    return tuple(out)
def card_html(title,body,kind="info"):return f'<div class="kz card {kind}"><div class="h">{html.escape(title)}</div><div class="small">{body}</div></div>'
def pbar_html(done,total,color="#B45309"):
    pc=100.0*done/max(1,total);return f'<div style="margin-top:6px;background:#e2e8f0;border-radius:6px;height:14px"><div style="width:{pc:.1f}%;height:14px;border-radius:6px;background:{color}"></div></div><div class="small">{done}/{total}</div>'
def stage_card(e):
    body=html.escape(e["body"])
    if e.get("total"):body+=pbar_html(e["done"],e["total"],"#0369A1"if e["stage"]>=5 else"#B45309")
    if e.get("eta")is not None:body+=f'<div class="small">estimated time left: {int(e["eta"]//60)} min {int(e["eta"]%60)} s</div>'
    return card_html(f"{e['stage']}/{N_STAGES} · {e['title']}",body,"info")
READY_HTML=card_html("Ready",f"Press <b>RUN COGNITIVE CARTRIDGE DEMO</b>. The run forges 16 cartridges, removes the source, reads every question from all 16 cartridges one by one ({TOTAL_READS} generate calls), checks the controls, seals the evidence and renders {N_POSTERS} JPG posters plus one ZIP.<br><b>Downloads appear when the run is complete.</b>","info")
def is_cuda_error(ex):
    s=f"{type(ex).__name__} {ex}".lower();return"cuda"in s or"device-side assert"in s or"cublas"in s
def run_handler():
    global RUN_COUNTER
    if not GPU_LOCK.acquire(blocking=False):
        yield pack(card_html("Busy","Another run is in progress. Please try again in a moment.","warn"));return
    stage_name="initialisation"
    try:
        RUN_COUNTER+=1;run_id=f"MISTRAL-CC-{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}-{RUN_COUNTER:04d}"
        prune_runs(2);run_dir=ROOT/run_id;run_dir.mkdir(parents=True,exist_ok=True);gen=execute_run(ENGINE,dict(run_id=run_id,run_dir=run_dir));first_ev=True
        while True:
            try:e=next(gen)
            except StopIteration as s:B=s.value;break
            stage_name=f"{e['stage']}/{N_STAGES} {e['title']}"
            yield pack(stage_card(e),[],None,None,None)if first_ev else pack(stage_card(e));first_ev=False
        R=B["R"];C=R["counts"];T=R["totals"];ok=R["verdict"].startswith("PASS")
        gallery_items=[(str(p),c_)for p,c_ in B["imgs"]]
        raw_texts={"txt":B["txt"].read_text(encoding="utf-8"),"json":B["payload"].read_bytes().decode("utf-8"),"man":B["manifest"].read_text(encoding="utf-8")}
        body=(f"<b>{html.escape(run_id)}</b><br>THIS LIVE RUN: MW1 {C['mw1']}/{T['mw1']} · MW2 {C['mw2']}/{T['mw2']} · LINKED {C['linked']}/{T['linked']} · MISSING {C['missing']}/{T['missing']} · ABSENT-ID {C['absent_id']}/{T['absent_id']} · NOMEM {C['nomem']}/{T['nomem']}<br>"
              f"verdict by the TEST474 gates: <b>{R['verdict']}</b> · replay identical to the sealed counts: {'YES'if R['replay_match']else'NO'}<br>{N_POSTERS} JPGs + ZIP · {B['checks']}/{B['checks']} technical checks PASS<br>"
              f'<span class="mono">payload SHA-256 (artifact integrity seal): {B["sha"]}</span>')
        yield pack(card_html(f"{N_STAGES}/{N_STAGES} · Complete",body,"on"if ok else"err"),gallery_items,{"zip":B["zip"],"json":B["payload"],"txt":B["txt"],"man":B["manifest"]},[p for p,_ in B["imgs"]],raw_texts)
    except Exception as ex:
        print("="*110);print(f"MISTRAL RUN FAILED — stage: {stage_name}");print(f"exception type : {type(ex).__name__}");print(f"exception message: {ex}");traceback.print_exc();print("="*110)
        kind="AUDIT FAIL — no package was sealed. "if isinstance(ex,AuditFail)else"";partial=getattr(ex,"partial",None)
        if partial:kind+="The raw outputs gathered so far were preserved (FULL RUN LOG button / FULL RUN LOG tab). "
        tip="<br><b>CUDA error: Runtime → Restart runtime, then run the cell again.</b>"if is_cuda_error(ex)else"<br>The full traceback is printed in the Colab console."
        card=card_html("Run failed",f"{html.escape(kind)}Stage: {html.escape(stage_name)}<br>{html.escape(type(ex).__name__)}: {html.escape(str(ex)[:600])}{tip}","err")
        if partial:yield pack(card,[],{"json":partial},None,{"txt":traceback.format_exc(),"json":Path(partial).read_text(encoding="utf-8"),"man":""})
        else:yield pack(card,[],None,None,None)
    finally:
        plt.close("all");GPU_LOCK.release()
        try:torch.cuda.empty_cache()
        except Exception:pass
# ---------------- service self-test (runs before the public link is printed) ----------------
def service_selftest():
    global QUIET;out=[];d=ROOT/"_selftest";shutil.rmtree(d,ignore_errors=True);d.mkdir(parents=True);(d/"w.txt").write_text("ok");out.append("ROOT writable");QUIET=True
    try:
        B=drain(execute_run(SelfTestEngine("ok"),dict(run_id="SELFTEST",run_dir=d)))
        assert len(B["imgs"])==N_POSTERS and file_ready(B["zip"])and B["verdict"]=="PASS_MISTRAL_FINAL_ENGINE"
        out.append(f"full pipeline on a synthetic engine: {len(B['imgs'])}/{N_POSTERS} posters (JPEG/RGB), ZIP, {B['checks']} checks")
        d2=d/"fail";d2.mkdir()
        try:drain(execute_run(SelfTestEngine("sentinel_changes"),dict(run_id="SELFTEST-B",run_dir=d2)))
        except AuditFail:out.append("fail-closed: a changed weight sentinel aborts the run (no package sealed)")
        else:raise RuntimeError("self-test: a changed sentinel did not abort the run")
    finally:QUIET=False
    if len(pack(READY_HTML))!=N_OUTPUTS or len(pack(READY_HTML,[],None,None,None))!=N_OUTPUTS:raise RuntimeError("callback output count mismatch")
    out.append(f"callback output count = {N_OUTPUTS}");out.append(f"file route {FILE_URL_PREFIX}");shutil.rmtree(d,ignore_errors=True);return out
say("[4/7] SERVICE SELF-TEST")
for c_ in service_selftest():say(" PASS ·",c_)
# ---------------- interface ----------------
say("[5/7] INTERFACE")
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
HERO=('<div class="kz hero"><div class="brand">AKBASCORE NIRVANA</div><div class="sub">MISTRAL · COGNITIVE CARTRIDGE</div>'
      '<div class="tag">Knowledge goes in. The source goes away. The memory remains.<br>'
      f'16 synthetic knowledge cartridges are installed into a frozen {MODEL_SHORT}. The source text is removed. 16 locked questions are read from the cartridges — or answered UNKNOWN.</div></div>')
def _tb(**kw):
    try:return gr.Textbox(show_copy_button=True,**kw)
    except TypeError:return gr.Textbox(**kw)
def _gallery(**kw):
    try:return gr.Gallery(columns=1,height="auto",object_fit="contain",preview=False,**kw)
    except TypeError:return gr.Gallery(**kw)
try:_blocks=gr.Blocks(css=CSS,title="AKBASCORE NIRVANA · MISTRAL · COGNITIVE CARTRIDGE")
except TypeError:_blocks=gr.Blocks(title="AKBASCORE NIRVANA · MISTRAL · COGNITIVE CARTRIDGE")
with _blocks as demo:
    gr.HTML(HERO)
    run_btn=gr.Button("RUN COGNITIVE CARTRIDGE DEMO",variant="primary",elem_id="kz_run")
    status=gr.HTML(READY_HTML)
    gallery=_gallery(label=f"{N_POSTERS} JPG posters (tap to open)",show_label=True)
    dl_all=gr.Button(DL_ALL_LABEL,interactive=False)
    jpg_urls=gr.Textbox(value="",visible=False)
    with gr.Row():
        dlb=[gr.DownloadButton(label=l,value=None,interactive=False)for l in DL_LABELS]
    with gr.Accordion("Direct file links (fallback)",open=False):
        fls=[gr.File(label=n,interactive=False)for n in("ZIP · all posters and audit files","FULL RUN LOG (.json)","READABLE RUN LOG (.txt)","RUN MANIFEST (.json)")]
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
say("[6/7] LAUNCH (public share link)")
say("[7/7] Open the printed gradio.live link in a new tab and press RUN COGNITIVE CARTRIDGE DEMO.")
demo.launch(share=True,allowed_paths=ALLOWED_PATHS,show_error=True,inline=False)
