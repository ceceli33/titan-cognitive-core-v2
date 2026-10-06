from matplotlib.colors import LinearSegmentedColormap
FIELD_CMAP=LinearSegmentedColormap.from_list("field",["#16213A","#1E3A8A","#0E7490","#22D3EE","#FDE68A"])
# ------------------------------------------------------------------ 08 SIGNATURE · FINGERPRINT → FIELD → ONE MEMORY LOCKS
def p08(ctx,k,path):
    P=ctx["P"];q,r,a=feat(ctx);fig=new_fig();sc=np.array(r["scores"],dtype=float)
    top=head(fig,"FINGERPRINT → 128-MEMORY FIELD → ONE MEMORY LOCKS","The numeric fingerprint is compared with every B memory of the experimental field. The strongest match locks.",tag="THIS LIVE RUN")
    yb=.33;hb=top-yb-.02
    panel(fig,.05,yb,.15,hb,C_NIGHT,C_CYAN,2.0)
    fig.text(.125,yb+hb-.045,"NUMERIC\nFINGERPRINT",ha="center",va="center",fontsize=12,weight="bold",color=C_CYAN,linespacing=1.15)
    pv=np.array(r["p"],dtype=float);lim=float(np.percentile(np.abs(pv),97))or 1.0;G=pv.reshape(64,-1)if pv.size%64==0 else pv[:,None]
    ax=fig.add_axes([.075,yb+.07,.10,hb-.19]);ax.imshow(G,aspect="auto",cmap="RdBu_r",vmin=-lim,vmax=lim,interpolation="nearest");ax.axis("off")
    fig.text(.125,yb+.035,f"{pv.size} numbers",ha="center",va="center",fontsize=10,color="#E2E8F0")
    arrow(fig,.205,yb+hb/2,.235,yb+hb/2,color=C_GLOW)
    panel(fig,.24,yb,.47,hb,C_NIGHT,C_NIGHT,0)
    fig.text(.475,yb+hb-.032,"128-MEMORY EXPERIMENTAL FIELD",ha="center",va="center",fontsize=14,weight="bold",color="white")
    lo,hi=float(sc.min()),float(sc.max());cols_,rows_=16,8;gx0=.255;gw=.44;gy0=yb+.02;gh=hb-.085;cw=gw/cols_;ch=gh/rows_
    for j in range(N):
        rr_,cc=divmod(j,cols_);x=gx0+cc*cw;y=gy0+gh-(rr_+1)*ch;v=(sc[j]-lo)/max(1e-9,hi-lo);sel=j+1==q["selected"];gold=j+1==q["record"]
        fig.add_artist(FancyBboxPatch((x+cw*.07,y+ch*.09),cw*.86,ch*.82,boxstyle="round,pad=0,rounding_size=0.004",transform=fig.transFigure,facecolor=FIELD_CMAP(v**2),
                                      edgecolor=C_GLOW if sel else("white"if gold else"none"),lw=3.2 if sel else(1.6 if gold else 0),ls="--"if(gold and not sel)else"-",zorder=3))
        fig.text(x+cw/2,y+ch*.5,f"{j+1:03d}",ha="center",va="center",fontsize=6.8,color=C_NIGHT if v**2>.55 else"#94A3B8",zorder=4,family=MONO)
    panel(fig,.725,yb,.225,hb,"white",C_FG,2.0)
    fig.text(.8375,yb+hb-.032,"STRONGEST MATCHES",ha="center",va="center",fontsize=13,weight="bold",color=C_FG)
    for i,(rec,s)in enumerate(q["top10"][:6]):
        y=yb+hb-.09-i*(hb-.16)/6;lock=i==0
        fig.text(.737,y,f"#{rec:03d}",fontsize=12,family=MONO,weight="bold"if lock else"normal",color=C_FG,va="center")
        fig.add_artist(Rectangle((.79,y-.011),.10*max(0,s),.022,transform=fig.transFigure,facecolor=C_GLOW if lock else"#93C5FD",edgecolor="none"))
        fig.text(.893,y,f"{s:.6f}",fontsize=10,family=MONO,color=C_FG,va="center")
        if lock:fig.text(.8375,y-.032,"← LOCKED"+(""if q["correct"]else"  (not the target)"),fontsize=10,weight="bold",color=C_CART if not q["correct"]else C_OK,va="center",ha="center")
    fig.text(.8375,yb+.03,"higher score = closer numeric match",ha="center",va="center",fontsize=9.5,color=C_NEU,style="italic")
    fig.text(.5,.255,"NUMBERS FIND NUMBERS.",ha="center",va="center",fontsize=36,weight="bold",color=C_FG)
    fig.text(.5,.195,"Each memory receives a similarity score. Higher means its numeric pattern is a closer match to the recall address.",ha="center",va="center",fontsize=13,color=C_FG)
    fit_text(fig,.07,.045,.86,.115,"128-MEMORY EXPERIMENTAL FIELD: the experimental bank size — not a hard-coded capacity limit. Brightness and glow are a visual metaphor for numeric similarity. "
             f"Featured: showcase question 3 of 5, fixed before the run (all five on the next poster). Researchers: ÇAĞRIİZ → weighted L02-V address → max-cosine numeric memory matching.",fs_max=10.5,fs_min=8.2,color=C_NEU,ha="center")
    foot(fig,ctx,k);return finish(fig,path,ctx,["NUMBERS FIND NUMBERS.",f"{q['top10'][0][1]:.6f}","128-MEMORY EXPERIMENTAL FIELD",f"#{q['selected']:03d}"])
# ------------------------------------------------------------------ 09 WATCH IT WORK
def p09(ctx,k,path):
    R=ctx["R"];fig=new_fig()
    top=head(fig,"WATCH IT WORK · FIVE LIVE QUESTIONS","Fixed before the run. Every question competes against all 128 B memories.",tag="THIS LIVE RUN")
    rh=(top-.20)/5
    for i,q in enumerate(R["queries"]):
        y=top-.01-(i+1)*rh;ok=q["correct"]
        panel(fig,.05,y+.006,.9,rh-.012,"white",C_OK if ok else C_CART,1.6 if ok else 2.2)
        fig.text(.065,y+rh*.62,f"Q{q['k']}",fontsize=22,weight="bold",color=C_MOD,va="center")
        fig.text(.11,y+rh*.66,f"instrument {q['entity']}",fontsize=13.5,weight="bold",color=C_FG,va="center")
        fig.text(.11,y+rh*.36,f"active memory A#{q['record']:03d} · {1000*(q['secs']or 0):.0f} ms",fontsize=10.5,color=C_NEU,va="center")
        ax=fig.add_axes([.355,y+rh*.15,.33,rh*.70]);t5=q["top10"][:5];vals=[s for _,s in t5]
        ax.barh(range(5),vals,color=[C_GLOW if j==q["selected"]else(C_OK if j==q["record"]else"#BFDBFE")for j,_ in t5],height=.72);ax.invert_yaxis()
        ax.set_yticks(range(5));ax.set_yticklabels([f"#{j:03d}"for j,_ in t5],fontsize=9,family=MONO);ax.set_xlim(0,1.13);ax.set_xticks([])
        for s_ in("top","right","bottom"):ax.spines[s_].set_visible(False)
        for t_,v in enumerate(vals):ax.text(v+.01,t_,f"{v:.4f}",va="center",fontsize=8.5,family=MONO,color=C_FG)
        if ok:
            fig.text(.82,y+rh*.62,f"LOCKED  B#{q['selected']:03d}",fontsize=17,weight="bold",color=C_OK,ha="center",va="center")
            fig.text(.82,y+rh*.32,f"correct · lead {q['margin']:+.3f}",fontsize=10.5,color=C_FG,ha="center",va="center")
        else:
            fig.text(.82,y+rh*.70,"ONE REAL MISS",fontsize=14,weight="bold",color=C_CART,ha="center",va="center")
            fig.text(.82,y+rh*.45,f"selected #{q['selected']:03d} · correct #{q['record']:03d}",fontsize=10.5,color=C_FG,ha="center",va="center")
            fig.text(.82,y+rh*.22,f"correct memory ranked {q['rank']}/{N}",fontsize=10.5,color=C_FG,ha="center",va="center")
    fig.text(.5,.135,f"{R['correct']} / {R['total']}   FIXED LIVE SHOWCASE · TEST524",ha="center",va="center",fontsize=24,weight="bold",color=C_MOD)
    msg="Memory numbers are audit labels; the retriever never receives them. Gold bar = the memory that locked"+("; green bar = the correct memory when it did not lock."if R["misses"]else".")
    if R["misses"]:msg="The system is experimental, not a scripted lookup: one of the five fixed questions misses its target. "+msg
    fit_text(fig,.08,.045,.84,.06,msg,fs_max=11,fs_min=8.5,color=C_NEU,ha="center")
    foot(fig,ctx,k);return finish(fig,path,ctx,[f"{R['correct']} / {R['total']}"]+[f"instrument {q['entity']}"for q in R["queries"]])
# ------------------------------------------------------------------ 10 ONE RECALL PROCESS
def p10(ctx,k,path):
    P=ctx["P"];R=ctx["R"];fig=new_fig();ms=[1000*(q["secs"]or 0)for q in R["queries"]]
    top=head(fig,"0 CANDIDATE-B LLM FORWARDS","The 7B model is not rerun separately for each of the 128 B-memory candidates.",tag="THIS LIVE RUN")
    yb=.30;hb=top-yb-.02
    panel(fig,.05,yb,.42,hb,C_BG,"#94A3B8",1.6);panel(fig,.53,yb,.42,hb,C_MODL,C_MOD,2.4)
    fig.text(.26,yb+hb-.035,"REPEATED LARGE-MODEL PROCESSING",ha="center",va="center",fontsize=13.5,weight="bold",color=C_CTL)
    fig.text(.26,yb+hb-.07,"(the approach this design avoids)",ha="center",va="center",fontsize=10,color=C_NEU,style="italic")
    labs=["Candidate 1","Candidate 2","Candidate 3","…","Candidate 128"]
    for i,l in enumerate(labs):
        y=yb+hb-.14-i*(hb-.20)/5
        if l=="…":fig.text(.26,y,"⋮",ha="center",va="center",fontsize=20,color=C_NEU);continue
        panel(fig,.085,y-.025,.13,.05,"white","#94A3B8",1.2,0.006);fig.text(.15,y,l,ha="center",va="center",fontsize=10.5,color=C_FG)
        arrow(fig,.22,y,.285,y,color="#94A3B8",lw=1.6,ms=14)
        panel(fig,.29,y-.025,.15,.05,"#E2E8F0","#94A3B8",1.2,0.006);fig.text(.365,y,"LLM forward",ha="center",va="center",fontsize=10.5,color=C_CTL,weight="bold")
    fig.text(.74,yb+hb-.035,"AKBASCORE MAM · TEST524",ha="center",va="center",fontsize=13.5,weight="bold",color=C_MOD)
    flow=[("QUESTION + ACTIVE A","white",C_FG,C_FG),("FROZEN MODEL",C_AI,C_AI,"white"),("ONE RECALL / ADDRESS PROCESS",C_NIGHT,C_VIO,C_GLOW),("NUMERIC MEMORY-FIELD COMPARISON",C_NIGHT,C_CYAN,C_CYAN)]
    for i,(t,fc,ec,tc)in enumerate(flow):
        y=yb+hb-.15-i*(hb-.20)/4;panel(fig,.58,y-.03,.32,.06,fc,ec,2.0,0.008);fig.text(.74,y,t,ha="center",va="center",fontsize=11.5,weight="bold",color=tc)
        if i<3:arrow(fig,.74,y-.032,.74,y-(hb-.20)/4+.032,color=C_NEU,lw=1.8,ms=14)
    panel(fig,.05,.105,.9,.165,C_NIGHT,C_NIGHT,0)
    fig.text(.5,.225,"CANDIDATE-B LLM FORWARDS = "+str(P["counters"]["candidate_B_llm_forwards"]),ha="center",va="center",fontsize=26,weight="bold",color=C_GLOW)
    fig.text(.5,.17,"The already-created numeric memory representations are compared mathematically.",ha="center",va="center",fontsize=14,color="white")
    rng_="%.0f ms"%min(ms)if round(min(ms))==round(max(ms))else"%.0f–%.0f ms"%(min(ms),max(ms))
    fig.text(.5,.13,f"Measured live: {rng_} per question for the recall process plus 128 numeric comparisons.",ha="center",va="center",fontsize=11,color=C_CYAN)
    fig.text(.5,.06,"Retrieval still computes: one model pass over the question with the active memory, then 128 numeric similarity comparisons.",ha="center",va="center",fontsize=10.5,color=C_NEU,style="italic")
    foot(fig,ctx,k);return finish(fig,path,ctx,["CANDIDATE-B LLM FORWARDS = "+str(P["counters"]["candidate_B_llm_forwards"]),"REPEATED LARGE-MODEL PROCESSING","ONE RECALL / ADDRESS PROCESS"])
# ------------------------------------------------------------------ 11 MEASURED EVIDENCE
def p11(ctx,k,path):
    P=ctx["P"];R=ctx["R"];S=SEALED524;fig=new_fig();sv={z["k"]:z for z in S["queries"]};rp=R["replay"]
    top=head(fig,"MEASURED EVIDENCE","Two separate experiments, shown separately: the live launch showcase and the larger sealed external replication.")
    yb=.30;hb=top-yb-.02
    panel(fig,.05,yb,.425,hb,C_MODL,C_MOD,2.6);panel(fig,.525,yb,.425,hb,C_SEALL,C_SEAL,2.6)
    fig.text(.2625,yb+hb-.035,"WORLD LAUNCH LIVE SHOWCASE · TEST524",ha="center",va="center",fontsize=13.5,weight="bold",color=C_MOD)
    fig.text(.2625,yb+hb-.068,"measured in this run",ha="center",va="center",fontsize=10.5,color=C_NEU,style="italic")
    fig.text(.2625,yb+hb*.66,f"{R['correct']} / {R['total']}",ha="center",va="center",fontsize=56,weight="bold",color=C_MOD)
    fig.text(.2625,yb+hb*.47,"five fixed launch examples · top-1 correct",ha="center",va="center",fontsize=12,color=C_FG)
    for i,q in enumerate(R["queries"]):
        x=.085+i*.0715;cq=C_OK if q["correct"]else C_CART;panel(fig,x,yb+.06,.062,.085,"white",cq,1.8,0.006)
        fig.text(x+.031,yb+.12,f"Q{q['k']}",ha="center",va="center",fontsize=10,weight="bold",color=C_FG);fig.text(x+.031,yb+.083,f"rank {q['rank']}",ha="center",va="center",fontsize=9.5,color=cq,weight="bold")
    fig.text(.2625,yb+.03,"rank of the correct memory among 128",ha="center",va="center",fontsize=9.5,color=C_NEU)
    fig.text(.7375,yb+hb-.035,"EXTERNAL REPLICATION · TEST523",ha="center",va="center",fontsize=13.5,weight="bold",color=C_SEAL)
    fig.text(.7375,yb+hb-.068,"sealed historical record",ha="center",va="center",fontsize=10.5,color=C_NEU,style="italic")
    fig.text(.7375,yb+hb*.66,f"{SEALED523['r1'][0]} / {SEALED523['r1'][1]}",ha="center",va="center",fontsize=56,weight="bold",color=C_SEAL)
    fig.text(.7375,yb+hb*.47,f"{SEALED523['r1_pct']} TOP-1",ha="center",va="center",fontsize=16,weight="bold",color=C_FG)
    fig.text(.7375,yb+hb*.30,f"counterfactual follow {fr(SEALED523['cf'])} = {SEALED523['cf_pct']}",ha="center",va="center",fontsize=12,color=C_FG)
    fig.text(.7375,yb+hb*.18,"the address followed a changed memory relation",ha="center",va="center",fontsize=10,color=C_NEU,style="italic")
    fig.text(.7375,yb+.03,"same frozen pointer, address and match as this demo",ha="center",va="center",fontsize=9.5,color=C_NEU)
    same="YES"if rp["match"]else"NO"
    fig.text(.5,.24,f"Live selections and ranks identical to the sealed TEST524 launch log: {same}",ha="center",va="center",fontsize=14,weight="bold",color=okc(rp["match"]))
    fig.text(.5,.195,f"largest top-score difference vs the sealed log: {rp['top1_max_abs_diff']:.6f}"+("  ·  same GPU model as the sealed run"if P["hardware"]["same"]else"  ·  different GPU from the sealed run: small numeric differences are expected"),ha="center",va="center",fontsize=10.5,color=C_NEU)
    fig.text(.5,.13,"TEST524 = five fixed launch examples.     TEST523 = larger external replication experiment.",ha="center",va="center",fontsize=14,color=C_FG,weight="bold")
    fig.text(.5,.075,"These two numbers come from different experiments and are never added together.",ha="center",va="center",fontsize=11,color=C_NEU,style="italic")
    foot(fig,ctx,k);return finish(fig,path,ctx,[f"{R['correct']} / {R['total']}",f"{SEALED523['r1'][0]} / {SEALED523['r1'][1]}",SEALED523["r1_pct"],SEALED523["cf_pct"]])
# ------------------------------------------------------------------ 12 EXPERIMENTAL INTEGRITY
def p12(ctx,k,path):
    P=ctx["P"];fz=P["frozen"];wp=P["wipe"];fig=new_fig();same=fz["guard_startup"]==fz["guard_after"]
    top=head(fig,"EXPERIMENTAL INTEGRITY","What the live retrieval path actually uses — transparent properties of the architecture.")
    items=[("Frozen weights",same,"weights are not changed by this experiment"),("No training",True,"no optimizer, no adapter, no weight update"),
           ("Source text in the retrieval path: absent",not wp["source_text_present"],"removed before the first question"),("No gold B memory ID supplied",True,"the retriever receives only the question and the active memory"),
           ("No decoded-text router",True,"no keyword, filename or decoded seal is used to route"),("No learned router",True,"nothing is fitted or trained for retrieval"),
           ("Candidate-B LLM forwards: "+str(P["counters"]["candidate_B_llm_forwards"]),P["counters"]["candidate_B_llm_forwards"]==0,"B memories are compared numerically, not re-run"),
           ("Numeric matching",True,"max cosine similarity over all 128 B memories"),("Fresh cache for every question",True,"no conversation history is carried over"),
           ("Frozen recall channel and address",True,"L23H12 ÇAĞRIİZ · L02-V address — fixed by the TEST524 lock")]
    cw=.43;rh=(top-.30)/5
    for i,(t,ok,sub)in enumerate(items):
        c_,r_=divmod(i,5);x=.05+c_*(cw+.04);y=top-.02-(r_+1)*rh
        panel(fig,x,y+.008,cw,rh-.016,"white","#CBD5E1",1.4,0.01)
        fig.text(x+.03,y+rh/2,"✓"if ok else"✗",fontsize=26,weight="bold",color=okc(ok),va="center",ha="center")
        fig.text(x+.06,y+rh*.62,t,fontsize=14,weight="bold",color=C_FG,va="center");fig.text(x+.06,y+rh*.32,sub,fontsize=10.5,color=C_NEU,va="center")
    panel(fig,.05,.08,.9,.16,C_BG,C_FG,1.6)
    fig.text(.5,.212,"MODEL BEFORE  =  MODEL AFTER",ha="center",va="center",fontsize=17,weight="bold",color=okc(same))
    fig.text(.5,.165,f"all-parameter weight guard · before {fz['guard_startup'][:24]}…",ha="center",va="center",fontsize=10.5,family=MONO,color=C_FG)
    fig.text(.5,.135,f"after  {fz['guard_after'][:24]}…",ha="center",va="center",fontsize=10.5,family=MONO,color=C_FG)
    fig.text(.5,.098,"every parameter tensor of the 7-billion-parameter model participates in this guard",ha="center",va="center",fontsize=10,color=C_NEU,style="italic")
    fig.text(.5,.045,"Verification method for each property (measured in this run or fixed by the code path) is listed in the researcher view.",ha="center",va="center",fontsize=10,color=C_NEU)
    foot(fig,ctx,k);return finish(fig,path,ctx,["EXPERIMENTAL INTEGRITY","Frozen weights","Numeric matching",fz["guard_after"][:24]])
# ------------------------------------------------------------------ 13 WHERE IT COULD LEAD
def p13(ctx,k,path):
    R=ctx["R"];fig=new_fig()
    top=head(fig,"WHERE THIS COULD LEAD","What was demonstrated today — and the research direction it opens.",tag="RESEARCH VISION")
    yb=.10;hb=top-yb-.02
    panel(fig,.05,yb,.32,hb,C_OKL,C_OK,2.6);panel(fig,.40,yb,.55,hb,C_CARTL,C_CART,2.6)
    fig.text(.21,yb+hb-.035,"DEMONSTRATED TODAY",ha="center",va="center",fontsize=16,weight="bold",color=C_OK)
    demo=["128-memory experimental field","frozen AI · no training","source removed from the retrieval path","question → ÇAĞRIİZ → numeric fingerprint","numeric association among 128 B memories",f"{R['correct']}/{R['total']} live · TEST523: {SEALED523['r1'][0]}/{SEALED523['r1'][1]} sealed"]
    for i,t in enumerate(demo):fig.text(.07,yb+hb-.10-i*.075,"✓  "+t,fontsize=12,color=C_FG,va="center")
    fig.text(.675,yb+hb-.035,"RESEARCH VISION / SCALING DIRECTION",ha="center",va="center",fontsize=16,weight="bold",color=C_CART)
    lad=[("128","CURRENT DEMO",True),("1K","",False),("1M","",False),("1B","",False)]
    for i,(n_,lab,real)in enumerate(lad):
        x=.43+i*.13;y=yb+hb-.215
        panel(fig,x,y,.10,.095,C_OK if real else"white",C_OK if real else C_CART,2.2 if real else 1.6)
        fig.text(x+.05,y+.055,n_,ha="center",va="center",fontsize=22,weight="bold",color="white"if real else C_CART)
        fig.text(x+.05,y+.018,lab or"memories",ha="center",va="center",fontsize=8.8,weight="bold",color="white"if real else C_NEU)
        if i<3:arrow(fig,x+.10,y+.047,x+.13,y+.047,color=C_CART,lw=1.8,ms=14)
    fig.text(.675,yb+hb-.25,"1K, 1M and 1B are research directions — not yet experimentally validated.",ha="center",va="center",fontsize=10,color=C_NEU,style="italic")
    vis=["very large persistent memory banks","hierarchical / indexed memory routing","compatible memory transfer between machines","autonomous multi-memory chains","long-lived AI memory","modular machine memory","robotics and distributed agents"]
    for i,t in enumerate(vis):
        c_,r_=divmod(i,4);fig.text(.425+c_*.315,yb+hb-.32-r_*.07,"→  "+t,fontsize=11.5,color=C_FG,va="center",weight="bold")
    fit_text(fig,.43,yb+.02,.5,.08,"These capabilities are a research vision built on today's mechanism. They are not available now and have not been demonstrated by this run.",fs_max=11,fs_min=8.5,color=C_NEU)
    foot(fig,ctx,k);return finish(fig,path,ctx,["DEMONSTRATED TODAY","RESEARCH VISION / SCALING DIRECTION","CURRENT DEMO","1B"])
# ------------------------------------------------------------------ 14 RESEARCHER VIEW · MECHANISM AND RESULTS
def p14(ctx,k,path):
    P=ctx["P"];R=ctx["R"];fig=new_fig()
    top=head(fig,"RESEARCHER VIEW · MECHANISM AND RESULTS","Exact computation of the TEST524 engine and every live value. Frozen by the TEST524 lock; parent: sealed TEST523.",tag="RESEARCHER VIEW")
    panel(fig,.05,top-.235,.9,.215,C_BG,C_FG,1.4)
    eq=["ÇAĞRIİZ   w_t  = softmax_t ( RoPE(q_L23,H12(final question token)) · RoPE(k_L23(A_t)) / sqrt(128) ),   t ≥ 1   (slot 0 excluded)",
        f"ADDRESS   p    = Σ_t  w_t · V_L2(A_t)                       {ADDR_DIM} numbers · uncentered · from the active A memory",
        "SCORE     s(B) = max_s  cos( p , V_L2(B_s) )                over every slot s ≥ 1 of B · all 128 B memories",
        "SELECT    B*   = argmax_B s(B)                              no candidate-B model forward · ties broken by lower index"]
    for i,t in enumerate(eq):fig.text(.065,top-.06-i*.045,t,fontsize=10.6,family=MONO,color=C_FG,va="center")
    hdr=["Q","active A","selected","gold rank","top score","lead","ÇAĞRIİZ peak","peak slot","entropy","ms"];xs=[.06,.10,.18,.26,.34,.43,.52,.62,.71,.80]
    yt=top-.27
    for h_,x in zip(hdr,xs):fig.text(x,yt,h_,fontsize=10,weight="bold",color=C_NEU,va="center")
    for i,q in enumerate(R["queries"]):
        y=yt-.038*(i+1);vals=[f"Q{q['k']}",f"#{q['record']:03d}",f"#{q['selected']:03d}",f"{q['rank']}/{N}",f"{q['top1']:+.6f}",f"{q['margin']:+.6f}",f"{q['peak']:.4f}",f"{q['pos']}/{q['T']}",f"{q['entropy']:.3f}",f"{1000*(q['secs']or 0):.1f}"]
        for v,x in zip(vals,xs):fig.text(x,y,v,fontsize=10,family=MONO,color=(C_OK if q["correct"]else C_CART)if x==.18 else C_FG,va="center")
    fig.text(.06,yt-.038*6-.005,f"live showcase {ci_text(R['correct'],R['total'])} (exact Clopper–Pearson) · replay of sealed TEST524 selections/ranks identical: {R['replay']['match']}",fontsize=9.6,color=C_NEU,va="center")
    fit_text(fig,.05,.045,.9,yt-.038*6-.05-.045,"RESEARCHER NOTES\n"+"\n".join("• "+t for t in P["scope"]["researcher_notes"]),fs_max=10.4,fs_min=7.8,color=C_FG)
    foot(fig,ctx,k);return finish(fig,path,ctx,["RESEARCHER NOTES",f"{R['queries'][0]['top1']:+.6f}",f"{R['queries'][-1]['top1']:+.6f}"])
# ------------------------------------------------------------------ 15 RESEARCHER VIEW · AUDIT
def p15(ctx,k,path):
    P=ctx["P"];fig=new_fig()
    top=head(fig,"RESEARCHER VIEW · VERIFICATION CLASS","Each property of the live retrieval path, and how it is verified: measured in this run, or fixed by the code path.",tag="RESEARCHER VIEW")
    rows=P["protocol"];rh=(top-.10)/len(rows)
    for i,e in enumerate(rows):
        y=top-.01-(i+1)*rh;col=C_OK if e["cls"]=="CHECKED"else C_CTL
        panel(fig,.05,y+.004,.58,rh-.008,"white","#CBD5E1",1.0,0.006)
        fig.text(.06,y+rh*.66,e["key"],fontsize=9.6,weight="bold",color=C_FG,va="center");fig.text(.06,y+rh*.28,e["note"],fontsize=8.1,color=C_NEU,va="center")
        fig.text(.43,y+rh/2,e["value"],fontsize=8.8,weight="bold",family=MONO,color=C_FG,va="center")
        fig.text(.625,y+rh/2,e["cls"],fontsize=8.4,weight="bold",color="white",va="center",ha="right",bbox=dict(boxstyle="round,pad=0.25",fc=col,ec=col))
    panel(fig,.66,.10,.29,top-.11,C_BG,C_FG,1.4)
    txt=("CHECKED = measured or verified during this run.\nBY CONSTRUCTION = a property of the code path, not separately instrumented (no hooks are added to the engine).\n\n"
         "NOT CLAIMED BY THIS RUN\n"+"\n".join("• "+t for t in P["scope"]["not_claimed"])+
         f"\n\n{len(P['checks_pre_seal'])} technical checks passed before sealing; post-seal checks verify posters, ZIP and hashes.\nHashes and the guard are integrity indicators, not third-party verification.")
    fit_text(fig,.675,.115,.26,top-.145,txt,fs_max=10.5,fs_min=7.6,color=C_FG)
    foot(fig,ctx,k);return finish(fig,path,ctx,["CHECKED","BY CONSTRUCTION","NOT CLAIMED BY THIS RUN"])
# ------------------------------------------------------------------ 16 EVIDENCE SEAL
def p16(ctx,k,path):
    P=ctx["P"];R=ctx["R"];E_=P["environment"];tm=P["timing"];H_=P["hardware"];fig=new_fig()
    top=head(fig,"COMPLETE RUN · EVIDENCE SEAL","Everything below was read from the runtime and is contained in the sealed payload.",tag="THIS LIVE RUN")
    tiles=[("RUN ID",P["run_id"]),("MODEL",MODEL_ID+" · frozen"),("GPU · SAME AS SEALED?",f"{E_['gpu']} · {'YES'if H_['same']else'NO'}"),("TORCH / TRANSFORMERS",f"{E_['torch']} / {E_['transformers']}"),
           ("TEST BANK",f"{N} records · {2*N} memories"),("LIVE SHOWCASE",f"{R['correct']}/{R['total']} · TEST524"),("TEST524 LOCK SHA",LOCK_SHA[:22]+"…"),("TECHNICAL CHECKS",f"{len(P['checks_pre_seal'])} passed pre-seal"),
           ("PEAK GPU MEMORY",f"{P['gpu']['peak_allocated_gib']:.2f} GiB"),("VERDICT",P["verdict"])]
    for i,(a_,b_)in enumerate(tiles):
        r_,c_=divmod(i,5);x=.05+c_*.182;y=top-.125-r_*.125;panel(fig,x,y,.172,.105,C_BG,C_MOD,1.8,0.01)
        fig.text(x+.008,y+.083,a_,fontsize=9,weight="bold",color=C_MOD,va="center");fit_text(fig,x+.008,y+.008,.156,.058,b_,fs_max=11.5,fs_min=6.8,family=MONO)
    st=[("model load (startup)",tm["model_load_seconds"]),("test bank",tm["test_bank"]),("read once · 256 memories",tm["read_once_forge"]),("numeric address field",tm["address_field"]),
        ("source removal",tm["source_removal"]),("5 live retrievals",tm["live_retrieval"])]
    ax=fig.add_axes([.25,.30,.67,top-.59]);ax.barh(range(len(st)),[s for _,s in st],color=[C_CTL,C_CART,C_CART,C_MOD,C_ERR,C_VIO]);ax.invert_yaxis();ax.set_yticks(range(len(st)));ax.set_yticklabels([a for a,_ in st],fontsize=10.5)
    for i,(_,s)in enumerate(st):ax.text(s,i,f" {s:.2f} s",va="center",fontsize=10.5,weight="bold")
    ax.set_xlim(0,max(max(s for _,s in st)*1.2,1.0));ax.set_xlabel(f"measured seconds (sealed TEST524 total runtime {SEALED524['runtime_s']:.2f} s on {CANON_GPU})",fontsize=9.5);clean_ax(ax)
    txt=f"PAYLOAD FILE : {ctx['payload_name']}\nPAYLOAD SHA-256 : {ctx['sha']}\nENGINE : TEST524 engine functions, unchanged · parent TEST523 lock {PARENT523[:16]}…\nIMAGE HASHES : images manifest (per-image SHA-256), written after rendering\nArtifact integrity seal ≠ scientific proof and ≠ independent third-party verification.\n{COPYRIGHT}"
    fit_text(fig,.05,.045,.9,.215,txt,fs_max=11.5,fs_min=8,family=MONO)
    foot(fig,ctx,k);return finish(fig,path,ctx,[P["run_id"],E_["gpu"],P["verdict"],f"{R['correct']}/{R['total']}"])
POSTERS=[("01_what_we_built","What we built",p01),("02_read_once","Step 1 — read once",p02),("03_human_language_ends_here","Human language ends here",p03),
 ("04_not_a_text_file","The retrieval memory is not a text file",p04),("05_source_removed","Source removed from the live retrieval path",p05),("06_question_finds_memory","How a question finds an associated memory",p06),
 ("07_question_cagriiz_fingerprint","New question → ÇAĞRIİZ → numeric fingerprint",p07),("08_one_memory_locks","Fingerprint → 128-memory field → one memory locks",p08),
 ("09_watch_it_work","Watch it work · five live questions",p09),("10_zero_candidate_forwards","0 candidate-B LLM forwards",p10),("11_measured_evidence","Measured evidence",p11),
 ("12_experimental_integrity","Experimental integrity",p12),("13_where_it_could_lead","Where this could lead",p13),("14_researcher_mechanism","Researcher view · mechanism and results",p14),
 ("15_researcher_verification","Researcher view · verification class",p15),("16_evidence_seal","Evidence seal",p16)]
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
        s.counters=Counter();s.mode=mode;s.n=0;s.rng=np.random.default_rng(7);s.g="a"*64
        s.info=dict(model_id=MODEL_ID,arch=ARCH,dtype="bfloat16",attn="sdpa",gpu="SELF-TEST ENGINE (no GPU)",canonical_gpu=CANON_GPU,gpu_total_gib=0.0,torch="n/a",transformers="n/a",gradio="n/a",
            python="n/a",platform="n/a",params=1,pad_id=0,vocab=1,model_load_seconds=0.0,startup_utc="n/a",guard0=s.g,hooks0=5,cell_source_sha256=None,guard_method="synthetic")
    def frozen_state(s):
        fh=3 if s.mode=="foreign_hook"else 0
        return dict(training=False,trainable_tensors=0,requires_grad_disabled=True,lora=False,optimizer=False,hooks=5+fh,foreign_hooks=fh,foreign_modules=({"__main__":fh}if fh else{}),
                    guard=("b"*64 if s.mode=="guard_changes"and s.n>0 else s.g))
    def prepare_panel(s):
        nm=build_names(N*3);codes=[f"Z{chr(65+i//26)}{chr(65+i%26)}Q"for i in range(N*6)];seals=codes[:N*3];classes=codes[N*3:]
        audit=[{"record":i+1,"entity":nm[3*i],"seal":seals[3*i],"class":classes[3*i]}for i in range(N)]
        show=[{"record":i+1,"entity":audit[i]["entity"],"question":qA(audit[i]["entity"])}for i in SHOWCASE_IDX]
        if s.mode=="leak":show[0]["question"]=show[0]["question"]+" "+seals[0]
        return dict(pool=N*6,records=N,showcase=show,audit=audit,seconds=0.0,audit_sha256=hashlib.sha256(canon(audit)).hexdigest())
    def forge_one(s,i):
        s.counters["forge_passes"]+=2;TA=28+i%5;TB=34+i%4
        return dict(TA=TA,TB=TB,numbers_A=2*28*4*128*TA,numbers_B=2*28*4*128*TB,secs=0.0,finite=True)
    def counts(s):return dict(A=N,B=N,B_MATS=N,A_PACK=N)
    def build_address_field(s):return dict(b_mats=N,rows=[33+i%4 for i in range(N)],dim=ARCH[3]*ARCH[4],a_packs=N,seconds=0.0,storages=N)
    def wipe(s):
        present={k_:(s.mode=="no_wipe"and k_=="SOURCES")for k_ in WIPED_NAMES}
        return dict(source_records_before=N,names=list(WIPED_NAMES),present=present,source_text_present=any(present.values()))
    def retrieve_live(s,k_,record,question):
        s.counters["query_retrievals"]+=1;s.n+=1
        sq=next(q for q in SEALED524["queries"]if q["record"]==record);g=record-1;sc=[0.5-0.001*j for j in range(N)]
        if sq["selected"]==record:sc[g]=sq["top1"]
        else:
            sel=sq["selected"]-1;others=[sel]+[j for j in range(N)if j not in(g,sel)][:sq["rank"]-2]
            for t,j in enumerate(others):sc[j]=sq["top1"]-0.001*t
            sc[g]=0.52
        T=30;w=[0.0]+[0.02]*(T-1);w[sq["pos"]]=sq["peak"]*1.6;tot=sum(w);w=[x/tot for x in w]
        order=sorted(range(N),key=lambda j:(-sc[j],j))
        return dict(k=k_,record=record,question=question,w=w,p=[round(float(x),6)for x in s.rng.normal(0,0.3,ARCH[3]*ARCH[4])],scores=sc,order=[j+1 for j in order[:10]],secs=0.05,b_forwards_counter=0)
    def tensor_excerpt(s,record):
        r=s.rng;Hm=(r.normal(0,1,(29,64))*np.exp(-np.arange(64)/40)).round(4)
        return dict(record=record,layer=BL,head=0,a_shape=[4,30,128],b_shape=[35,512],a_v=r.normal(0,.4,(8,8)).round(3).tolist(),a_k=r.normal(0,.4,(4,8)).round(3).tolist(),
                    b_rows=r.normal(0,.4,(8,8)).round(3).tolist(),a_heat=Hm.tolist(),numbers_A=2*28*4*128*30,numbers_B=2*28*4*128*35)
    def sync(s):pass
    def reset_peak(s):pass
    def peak_gib(s):return 0.0
def drain(gen):
    while True:
        try:next(gen)
        except StopIteration as s_:return s_.value
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
        paths=[files.get(k_)for k_ in FILE_KEYS]
        for pth in paths:
            if pth is not None and not file_ready(pth):raise RuntimeError(f"Download artifact missing or empty: {pth}")
        out+=[dl_update(p,l)for p,l in zip(paths,DL_LABELS)]+[None if p is None else str(p)for p in paths]
    if raw is _K:out+=[SKIP()for _ in RAW_KEYS]
    elif raw is None:out+=[""]*len(RAW_KEYS)
    else:out+=[raw[k_]for k_ in RAW_KEYS]
    return tuple(out)
def card_html(title,body,kind="info"):return f'<div class="kz card {kind}"><div class="h">{html.escape(title)}</div><div class="small">{body}</div></div>'
def pbar_html(done,total,color="#B45309"):
    pc_=100.0*done/max(1,total);return f'<div style="margin-top:6px;background:#e2e8f0;border-radius:6px;height:14px"><div style="width:{pc_:.1f}%;height:14px;border-radius:6px;background:{color}"></div></div><div class="small">{done}/{total}</div>'
def stage_card(e):
    body=html.escape(e["body"])
    if e.get("total"):body+=pbar_html(e["done"],e["total"],"#0369A1"if e["stage"]>=5 else"#B45309")
    if e.get("eta")is not None:body+=f'<div class="small">estimated time left: {int(e["eta"]//60)} min {int(e["eta"]%60)} s</div>'
    return card_html(f"{e['stage']}/{N_STAGES} · {e['title']}",body,"info")
READY_HTML=card_html("Ready",f"Press <b>RUN WORLD LAUNCH DEMO</b>. The run builds the 128-record test bank, lets the frozen AI read each record once (256 numeric memories), removes the source from the live retrieval path, "
    f"asks the 5 fixed questions — each one scored against all 128 B memories — seals the evidence and renders {N_POSTERS} JPG posters plus one ZIP.<br>"
    f"Canonical sealed hardware: <b>{CANON_GPU}</b>; this run reports the GPU actually used.<br><b>Downloads appear when the run is complete.</b>","info")
def _row_html(q):
    res="LOCKED ✓"if q["correct"]else"miss · correct memory rank %d/%d"%(q["rank"],N)
    return"Q%d · %s · A#%03d → B#%03d · %s<br>"%(q["k"],html.escape(q["entity"]),q["record"],q["selected"],res)
def is_cuda_error(ex):
    s=f"{type(ex).__name__} {ex}".lower();return"cuda"in s or"device-side assert"in s or"cublas"in s
def run_handler():
    global RUN_COUNTER
    if not GPU_LOCK.acquire(blocking=False):
        yield pack(card_html("Busy","Another run is in progress. Please try again in a moment.","warn"));return
    stage_name="initialisation"
    try:
        RUN_COUNTER+=1;run_id=f"MAM524-{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}-{RUN_COUNTER:04d}"
        prune_runs(2);run_dir=ROOT/run_id;run_dir.mkdir(parents=True,exist_ok=True);gen=execute_run(ENGINE,dict(run_id=run_id,run_dir=run_dir));first_ev=True
        while True:
            try:e=next(gen)
            except StopIteration as s:B=s.value;break
            stage_name=f"{e['stage']}/{N_STAGES} {e['title']}"
            yield pack(stage_card(e),[],None,None,None)if first_ev else pack(stage_card(e));first_ev=False
        R=B["R"];P=B["P"];rp=R["replay"]
        gallery_items=[(str(p),c_)for p,c_ in B["imgs"]]
        raw_texts={"txt":B["txt"].read_text(encoding="utf-8"),"json":B["payload"].read_bytes().decode("utf-8"),"man":B["manifest"].read_text(encoding="utf-8")}
        rows="".join(_row_html(q)for q in R["queries"])
        body=(f"<b>{html.escape(B['run_id'])}</b><br><b>WORLD LAUNCH LIVE SHOWCASE (TEST524): {R['correct']}/{R['total']}</b> · selections identical to the sealed launch log: {'YES'if rp['match']else'NO'}<br>{rows}"
              f"EXTERNAL REPLICATION (TEST523, sealed record): {SEALED523['r1'][0]}/{SEALED523['r1'][1]} = {SEALED523['r1_pct']} top-1<br>"
              f"verdict: <b>{html.escape(B['verdict'])}</b> · {N_POSTERS} JPGs + ZIP · {B['checks']}/{B['checks']} technical checks PASS<br>"
              f'<span class="mono">payload SHA-256 (artifact integrity seal): {B["sha"]}</span>')
        yield pack(card_html(f"{N_STAGES}/{N_STAGES} · Complete",body,"on"),gallery_items,{"zip":B["zip"],"json":B["payload"],"txt":B["txt"],"man":B["manifest"]},[p for p,_ in B["imgs"]],raw_texts)
    except Exception as ex:
        print("="*140);print(f"WORLD LAUNCH RUN FAILED — stage: {stage_name}");print(f"exception type : {type(ex).__name__}");print(f"exception message: {ex}");traceback.print_exc();print("="*140)
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
        assert len(B["imgs"])==N_POSTERS and file_ready(B["zip"])and B["verdict"]=="TEST524_LIVE_SHOWCASE_4_OF_5_INTEGRITY_VERIFIED"and B["R"]["replay"]["match"]
        out.append(f"full pipeline on a synthetic engine: {len(B['imgs'])}/{N_POSTERS} posters (JPEG/RGB), ZIP, {B['checks']} checks")
        for mode,what in(("guard_changes","a changed weight guard"),("leak","a seal inside a question"),("no_wipe","source text left in memory"),("foreign_hook","a foreign hook")):
            d2=d/mode;d2.mkdir()
            try:drain(execute_run(SelfTestEngine(mode),dict(run_id="SELFTEST-"+mode.upper(),run_dir=d2)))
            except AuditFail:out.append(f"fail-closed: {what} aborts the run (no package sealed)")
            else:raise RuntimeError(f"self-test: {what} did not abort the run")
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
.brand{font-size:clamp(28px,8.5vw,44px);font-weight:800;letter-spacing:1px;line-height:1.05}
.sub{font-size:clamp(13px,4vw,17px);font-weight:700;color:var(--kz-warn)!important;margin-top:4px;letter-spacing:.5px}
.tag{font-size:clamp(13px,3.8vw,15px);opacity:.9;margin-top:8px}
.by{font-size:12px;opacity:.75;margin-top:6px}
.card{background:var(--kz-card);border:2px solid var(--kz-bd);border-radius:12px;padding:12px 14px;margin:6px 0}
.card.on{border-color:var(--kz-on)}.card.err{border-color:var(--kz-err)}.card.warn{border-color:var(--kz-warn)}.card.info{border-color:var(--kz-a)}
.card .h{font-weight:800;font-size:16px;margin-bottom:4px}
.small{font-size:14px}.mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;word-break:break-all}
#kz_run button,#kz_run{font-size:clamp(17px,5vw,22px)!important;font-weight:800!important;min-height:64px!important}
"""
DL_ALL_JS="""(u)=>{let items=[];try{items=JSON.parse(u||"[]")}catch(e){items=[]}
let i=0;const next=()=>{if(i>=items.length)return;const it=items[i++];const a=document.createElement('a');a.href=it.url;a.download=it.name;a.rel='noopener';
document.body.appendChild(a);a.click();setTimeout(()=>{a.remove();next()},900)};next();return u;}"""
HERO=('<div class="kz hero"><div class="brand">AKBASCORE MAM</div><div class="sub">PERSISTENT ASSOCIATIVE MACHINE MEMORY · WORLD LAUNCH</div>'
      '<div class="tag">Human language goes in once. A frozen AI keeps model-native numeric memories. A natural question finds the associated one.<br>'
      f'128-record experimental bank · frozen {MODEL_SHORT} · TEST524 engine · numbers find numbers.</div>'
      f'<div class="by">{html.escape(AUTHOR)} · {html.escape(AUTHOR_PLACE)} · {html.escape(LAUNCH_DATE)} · {html.escape(COPYRIGHT)}</div></div>')
_GR_MAJOR=int(re.match(r"\d+",gr.__version__).group())
def _tb(**kw):
    for extra in(([{"buttons":["copy"]}]if _GR_MAJOR>=6 else[])+[{"show_copy_button":True},{}]):
        try:return gr.Textbox(**extra,**kw)
        except Exception:pass
def _gallery(**kw):
    try:return gr.Gallery(columns=1,height="auto",object_fit="contain",preview=False,**kw)
    except TypeError:return gr.Gallery(**kw)
if _GR_MAJOR>=6:_blocks=gr.Blocks(title="AKBASCORE MAM · WORLD LAUNCH")   # Gradio 6: css is passed to launch()
else:
    try:_blocks=gr.Blocks(css=CSS,title="AKBASCORE MAM · WORLD LAUNCH")
    except TypeError:_blocks=gr.Blocks(title="AKBASCORE MAM · WORLD LAUNCH")
with _blocks as demo:
    gr.HTML(HERO)
    run_btn=gr.Button("RUN WORLD LAUNCH DEMO",variant="primary",elem_id="kz_run")
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
say("[7/7] Open the printed gradio.live link in a new tab and press RUN WORLD LAUNCH DEMO.")
_LAUNCH=dict(share=True,allowed_paths=ALLOWED_PATHS,show_error=True,inline=False)
if _GR_MAJOR>=6:_LAUNCH["css"]=CSS
try:demo.launch(**_LAUNCH)
except TypeError as _ex:
    if"css"not in str(_ex):raise
    _LAUNCH.pop("css",None);demo.launch(**_LAUNCH)
