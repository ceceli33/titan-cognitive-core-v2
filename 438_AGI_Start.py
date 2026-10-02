# TEST 438c — AKBASCORE NIRVANA PKV D120 — MULTI-RECORD SELECTION / COMPOSITION / LOAD — pre-registered pilot (single Colab cell, A100)
# Engine unchanged from the working NIRVANA D120: source-only forge → input_layernorm → k_proj/v_proj (pre-RoPE) → PCA per layer×KV head (32 neutral sentences)
# → BF16 D=120 codes → decode μ+cBᵀ with shared SINK at slot 0 → RoPE → DynamicCache → PAD dummy prefix + QUESTION/ANSWER frame → greedy. Joint-source records only.
# No training/LoRA/optimizer/weight update/steering hooks. Every error prints a self-contained report in this output cell; files are optional helpers.
import os,sys,re,json,time,random,hashlib,importlib.util,subprocess,traceback,inspect,platform
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter,OrderedDict
import numpy as np
#<<CPU_BEGIN>>
TEST="438c";SEED=384;D=120;DMAX=128;NL=28;MAX_NEW=32;MAX_TOTAL=6144;SHORT_T=256;LONG_KEEP=3;PROBE_STEPS=6
FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n";STYLE=" Give only the answer on the first line; do not explain."
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
ARMS=("VANILLA","CONTEXT","NATIVE_KV","PKV_D120","PKV_OWNSINK","FOREIGN_LENMATCH");CORE=("VANILLA","CONTEXT","NATIVE_KV","PKV_D120")
ABBR=dict(VANILLA="VAN",CONTEXT="CTX",NATIVE_KV="NAT",PKV_D120="PKV",PKV_OWNSINK="OWN",FOREIGN_LENMATCH="FOR")
SINK_ENV_MULT=2.0;SINK_REL_FLOOR=1e-3;SINK_REL_CAP=2e-2;LEN_TOL_ABS=4;LEN_TOL_REL=0.15;PROBE_TVD_MAX=0.05;CTX_MIN=0.75;LIFT_MIN=0.5
RUN_ID=f"TEST{TEST}-"+datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
CTX=dict(stage="init",sub="",fatal=None,errors=[],gens=Counter(),last={},env=dict(python=sys.version.split()[0],platform=platform.platform(),D=D,dtype="bfloat16",model=MODEL_ID),model_loaded=False,stop=None,seal_checks=[],file_io_ok=True)
ROWS=[];CASES=[];EXPECTED=OrderedDict();DIAG={};BYID={}
def now():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def say(*a):print(*a,flush=True)
def stage(name,sub=""):CTX["stage"]=name;CTX["sub"]=sub;say(f"[TEST{TEST}] ▶ STAGE {name}"+(f" / {sub}" if sub else ""))
def jsafe(o):
 if isinstance(o,dict):return {str(k):jsafe(v) for k,v in o.items()}
 if isinstance(o,(list,tuple,set)):return [jsafe(v) for v in (sorted(o,key=str) if isinstance(o,set) else o)]
 if isinstance(o,bool) or o is None or isinstance(o,str):return o
 if isinstance(o,(int,np.integer)):return int(o)
 if isinstance(o,(float,np.floating)):o=float(o);return o if np.isfinite(o) else f"non-finite:{o}"
 return str(o)
def canon(x):return json.dumps(jsafe(x),sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def norm(x):return re.sub(r"[^\w]+"," ",str(x).casefold()).strip()
def has(text,phrase):return bool(phrase) and f" {norm(phrase)} " in f" {norm(text)} "
class TechStop(RuntimeError):pass
class GpuBlocked(RuntimeError):pass
class Reported(RuntimeError):
 def __init__(self,rec):super().__init__(rec.get("msg",""));self.rec=rec
def is_fatal(ex):
 s=f"{type(ex).__name__} {ex}".lower();return any(k in s for k in ("cuda","cublas","cudnn","device-side","illegal memory","out of memory","acceleratorerror","nccl"))
def gpu_alive():return CTX["fatal"] is None
def gpu_guard():
 if CTX["fatal"] is not None:raise GpuBlocked("GPU operation blocked: an earlier fatal error was recorded; no further device work is allowed")
def report(ex,c=None,arm=None,planned=False,extra=None):
 fatal=(not planned) and is_fatal(ex);tb="(planned technical stop — no traceback needed)" if planned else traceback.format_exc()
 cf=globals().get("cfg");attn=getattr(cf,"_attn_implementation",None) if cf is not None else None
 L=["#"*110,f"TEST{TEST} ERROR REPORT — BEGIN",f"run_id: {RUN_ID}",f"utc: {now()}",f"stage: {CTX['stage']} | sub-stage: {CTX['sub']}"]
 if c:L+=[f"case: {c.get('id')} | family={c.get('family')} vkey={c.get('vkey')} group={c.get('group')} member={c.get('member')} pos={c.get('pos')} load={c.get('load')} k={c.get('k')}",f"arm: {arm}",f"tokens: source(incl. start slot)={c.get('src_tokens')} question={c.get('q_tokens')} max_new={MAX_NEW} budget={MAX_TOTAL}",f"question: {c.get('question')}","source:\n"+str(c.get("source"))]
 elif arm:L.append(f"arm: {arm}")
 L+=[f"engine: D={D} DMAX={DMAX} dtype=bfloat16 attn_implementation={attn} model={MODEL_ID}",f"environment: {json.dumps(CTX['env'],ensure_ascii=False)}",f"last recorded GPU-op context: {json.dumps(jsafe(CTX['last']),ensure_ascii=False)}",
  f"progress: rows={len(ROWS)} complete={sum(1 for r in ROWS if r.get('complete'))}/{len(CASES)} generations={dict(CTX['gens'])}",
  f"fatal: {fatal} | continuation possible: {not fatal} | runtime restart required: {'YES — restart the Colab runtime before any further run' if fatal else 'no'}"]
 if extra:L.append(f"extra: {extra}")
 L+=[f"error class: {type(ex).__module__}.{type(ex).__name__}","error message:",str(ex),"traceback:",tb,f"TEST{TEST} ERROR REPORT — END","#"*110];say("\n".join(L))
 rec=dict(utc=now(),stage=CTX["stage"],sub=CTX["sub"],case=c.get("id") if c else None,arm=arm,type=type(ex).__name__,msg=str(ex),tb=tb,fatal=fatal,planned=planned,extra=extra);CTX["errors"].append(rec)
 if fatal and CTX["fatal"] is None:CTX["fatal"]=rec
 return rec
CORPUS=["Jonas Weber carried the wooden crate across the quiet market square.","Priya Nair wrote a long letter to her cousin in the morning.","The small boat drifted slowly toward the rocky shore.","Omar Haddad fixed the broken clock in the village school.","A tired teacher closed the green door of the library.","Sofia Rossi baked fresh bread for the harvest festival.","The children watched the kites rising above the hill.","Liam O'Connor sold his old bicycle to a neighbor.","Heavy rain flooded the narrow street near the market.","Nadia Petrova translated the ancient manuscript into French.","The farmer counted the sheep before sunset.","Hiro Sato opened a tiny bakery beside the river.","An old dog slept under the kitchen table all afternoon.","Carlos Mendes washed the windows of his grandmother's house.","The museum guard locked the heavy gate at midnight.","Fatima Zahra grew tomatoes in the backyard garden.","The pilot announced a short delay because of fog.","Anna Kowalski found a lost wallet on the bus.","Snow covered the mountain village during the night.","Ravi Kumar taught his brother how to play chess.","The chef sharpened every knife before dinner service.","Lucas Martin cleaned the roof of the barn after the storm.","A young violinist practiced scales in the empty hall.","Mei Lin delivered the package to the wrong address.","Marek Novak placed the glass bottle beneath the wooden bench.","Sara Ibrahim carried a red notebook into the quiet classroom.","Noah Schmidt repaired the small radio beside the kitchen window.","Yuki Mori left a paper envelope near the station entrance.","Amira Hassan moved the ceramic bowl onto the upper shelf.","Peter Novak opened the metal box behind the old theater.","Lucia Costa placed the yellow scarf inside the travel bag.","Daniel Kim carried a black umbrella through the central courtyard."]
ENT=[("Elena Varga","silver compass","northern archive","southern vault"),("Kerem Yıldız","brass lantern","old lighthouse","fishing harbor"),("Aiko Tanabe","blue signal","eastern platform","western gate"),("Marta Solberg","copper sextant","river warehouse","hill observatory"),("Diego Ferraz","wooden rudder","dry dock","harbor office"),("Lena Hoffmann","ivory chess set","city library","stone chapel")]
ADJ=["amber","birch","cedar","dune","ember","frost"];NOUN=["kiosk","depot","pavilion","quarry","garage","bunker"];E=len(ENT);SPECIAL=("UNKNOWN","AMBIGUOUS")
def build_cases():
 C=[]
 def add(family,vkey,group,member,records,q,target,alts,e,load=0,pos="",k=0):
  C.append(dict(id=f"{group}:{member}",family=family,vkey=vkey,group=group,member=member,records=list(records),source="\n".join(records),question=q+STYLE,target=target,alts=sorted(set(alts)-{target}),ent=e,load=load,pos=pos,k=k,tmpl=f"{vkey}|{member}"))
 def pair(family,vkey,group,records,q,target,alts,e,**kw):add(family,vkey,group,"fwd",records,q,target,alts,e,**kw);add(family,vkey,group,"rev",records[::-1],q,target,alts,e,**kw)
 for e,(P,O,a,b) in enumerate(ENT):
  X,Y=(a,b) if e%2==0 else (b,a);AL=[a,b];Z,Vh,Yc=f"Z{e}",f"V{e}",f"Y{e}"
  add("single","single",f"single{e}","orig",[f"{P} stored the {O} in the {X}."],f"Where did {P} store the {O}?",X,AL,e)
  add("single","single",f"single{e}","cf",[f"{P} stored the {O} in the {Y}."],f"Where did {P} store the {O}?",Y,AL,e)
  tl=[f"On 2041-05-03, {P} stored the {O} in the {X}.",f"On 2041-05-08, {P} moved the {O} to the {Y}."]
  pair("latest","latest",f"latest{e}",tl,f"According to the most recent dated record, where is the {O}?",Y,AL,e)
  pair("historical","historical",f"hist{e}",tl,f"According to the record dated 2041-05-03, where was the {O}?",X,AL,e)
  pair("authority","authority",f"auth{e}",[f"Official verified report: {P} stored the {O} in the {X}.",f"Unverified rumor: {P} stored the {O} in the {Y}."],f"Use the official verified report and disregard the unverified rumor. Where is the {O}?",X,AL,e)
  pair("correction","correction",f"corr{e}",[f"Entry 1: {P} stored the {O} in the {X}.",f"Correction to entry 1, which supersedes it: the {O} is in the {Y}."],f"After applying every correction, where is the {O}?",Y,AL,e)
  qa=f"Both reports have equal credibility and refer to the same time. Where is the {O} currently? If the reports give different locations, answer AMBIGUOUS."
  conf=[f"Report A: the {O} is currently in the {X}.",f"Report B: the {O} is currently in the {Y}."]
  add("ambiguity","ambiguity",f"amb{e}","conflict_fwd",conf,qa,"AMBIGUOUS",AL,e);add("ambiguity","ambiguity",f"amb{e}","conflict_rev",conf[::-1],qa,"AMBIGUOUS",AL,e)
  add("ambiguity","ambiguity",f"amb{e}","consistent",[f"Report A: the {O} is currently in the {X}.",f"Report B: the {O} is currently in the {X}."],qa,X,AL+["AMBIGUOUS"],e)
  zq=f"Name the place (not a container or vehicle) where the {O} is located.";HA=AL+[Z,Vh,Yc]
  pair("two_hop","two_hop",f"two{e}",[f"The {O} is inside container {Z}.",f"Container {Z} is in the {X}."],zq,X,HA,e)
  pair("three_hop","three_hop",f"three{e}",[f"The {O} is inside container {Z}.",f"Container {Z} is inside vehicle {Vh}.",f"Vehicle {Vh} is at the {Y}."],zq,Y,HA,e)
  mq=f"Name the place (not a container) where the {O} is located. If the records do not connect the {O} to any place, answer UNKNOWN."
  comp=[f"The {O} is inside container {Z}.",f"Container {Z} is in the {Y}."];brok=[f"The {O} is inside container {Z}.",f"An unrelated container {Yc} is in the {Y}."]
  add("missing","missing",f"miss{e}","complete_fwd",comp,mq,Y,HA+["UNKNOWN"],e);add("missing","missing",f"miss{e}","complete_rev",comp[::-1],mq,Y,HA+["UNKNOWN"],e)
  add("missing","missing",f"miss{e}","broken_fwd",brok,mq,"UNKNOWN",HA,e);add("missing","missing",f"miss{e}","broken_rev",brok[::-1],mq,"UNKNOWN",HA,e)
  ds=lambda n:[f"On 2041-06-{j%28+1:02d}, keeper P{e}-{j} stored artifact R{e}-{j} in locker W{e}-{j}." for j in range(n)];lq=f"According to the most recent dated record about the {O}, where is it?"
  for n in (8,32,128):d_=ds(n);pair("load",f"load_n{n:03d}",f"load{e}-{n}",d_[:n//2]+tl+d_[n//2:],lq,Y,AL,e,load=n,pos="mid")
  add("load","load_pos_early",f"loadpos{e}-early","early",tl+ds(32),lq,Y,AL,e,load=32,pos="early");add("load","load_pos_late",f"loadpos{e}-late","late",ds(32)+tl,lq,Y,AL,e,load=32,pos="late")
  POOLE=[f"{ADJ[e]} {nn}" for nn in NOUN]
  for k in (2,4,8):
   locs=[X,Y]+POOLE[:k-2];dates=[f"2041-05-{3+3*i:02d}" for i in range(k)];assign=[locs[(i+e)%k] for i in range(k)]
   recs=[f"On {dates[i]}, {P} {'stored' if i==0 else 'moved'} the {O} {'in' if i==0 else 'to'} the {assign[i]}." for i in range(k)];random.Random(SEED*1000+e*10+k).shuffle(recs)
   pair("conflict_load",f"conflict_k{k}",f"conf{e}-{k}",recs,f"According to the most recent dated record, where is the {O}?",assign[-1],locs,e,k=k)
 return C
NEGW={"not","never","isn","wasn","aren","weren","neither","nor","without","cannot"}
FILL={"the","a","an","in","at","on","to","inside","located","stored","kept","is","was","are","it","t","s","currently","be","been"}
ABST=re.compile(r"\b(cannot be determined|can not be determined|not (be )?(determined|inferred|specified|stated|mentioned|known|provided)|no information|insufficient|unclear)\b")
def first_line(t):
 for ln in str(t).splitlines():
  s=ln.strip()
  if s:return s
 return ""
def clean_line(s):return re.sub(r"^\W*(final answer|answer)\s*[:\-]\s*","",s.strip(),flags=re.I).strip(" *`\"'")
def negpos(w):return [i for i,x in enumerate(w) if x in NEGW or (x=="longer" and i>0 and w[i-1]=="no")]
def mention(n,ph,used=None):
 # pre-negation scopes over the phrase only if every word between the negator and the phrase is a filler word (≤6 words back);
 # post-negation: phrase followed by "is/was/are not|never|no longer" or "isn't/wasn't/aren't". Affirmed+negated occurrences → "conflict".
 w=n.split();p=norm(ph).split();L=len(p);aff=negd=False;NP=negpos(w)
 if not p or not w:return "none"
 for i in range(len(w)-L+1):
  if w[i:i+L]!=p:continue
  neg=False;prev=[j for j in NP if j<i and i-j<=6]
  if prev:
   j=max(prev)
   if all(x in FILL for x in w[j+1:i]):neg=True;used is not None and used.add(j)
  a=w[i+L:i+L+3]
  if len(a)>=2 and a[0] in ("is","was","are") and (a[1] in NEGW or a[1:3]==["no","longer"]):neg=True;used is not None and used.add(i+L+1)
  if a and a[0] in ("isn","wasn","aren","weren"):neg=True;used is not None and used.add(i+L)
  if neg:negd=True
  else:aff=True
 return "conflict" if (aff and negd) else ("affirmed" if aff else ("negated" if negd else "none"))
def score(text,c):
 fl=clean_line(first_line(text));n=norm(fl);t=c["target"];locs=[x for x in c["alts"] if x not in SPECIAL];used=set()
 st=mention(n,t,used);ms={x:mention(n,x,used) for x in locs+[s for s in SPECIAL if s!=t]};A=[x for x in locs if ms[x]=="affirmed"];An=[x for x in locs if ms[x]=="negated"];Sa=[s for s in SPECIAL if s!=t and ms[s]=="affirmed"]
 NP=negpos(n.split());negword=bool(NP);unattributed=bool(set(NP)-used);ab=bool(ABST.search(n))
 if not n:cat,sem="empty","incorrect"
 elif st=="conflict" or "conflict" in ms.values():cat,sem="conflicting_mentions","review"
 elif st=="affirmed" and not A and not Sa and unattributed:cat,sem="unattributed_negation","review"
 elif st=="affirmed" and not A and not Sa:cat="correct_exact" if n in (norm(t),norm("the "+t)) else ("correct_with_rejection" if An else "correct_verbose");sem="correct"
 elif st=="affirmed":cat,sem="mixed","incorrect"
 elif st=="negated":cat,sem=("wrong_alternative" if A else "negated_target"),"incorrect"
 elif A:cat,sem=("wrong_alternative" if len(A)==1 else "mixed_wrong"),"incorrect"
 elif Sa:cat,sem="wrong_special","incorrect"
 elif ab:cat,sem=("abstain_nonstandard","review") if t in SPECIAL else ("wrong_abstain","incorrect")
 elif negword:cat,sem="unparsed_negation","review"
 else:cat,sem="no_answer","incorrect"
 return dict(first_line=fl,category=cat,semantic=sem,**{"pass":int(sem=="correct")},review=int(sem=="review"),format_exact=int(cat=="correct_exact"),alt_mixing=int(bool(A)),target_anywhere=int(has(text,t)),alt_anywhere=int(any(has(text,x) for x in c["alts"])))
def scorer_selftest():
 L=dict(target="northern archive",alts=["southern vault","AMBIGUOUS"]);U=dict(target="UNKNOWN",alts=["northern archive","Z0"]);A=dict(target="AMBIGUOUS",alts=["northern archive","southern vault"])
 F_=[("northern archive",L,"correct_exact","correct"),("The northern archive.\nYou are an AI assistant.",L,"correct_exact","correct"),("**Northern Archive**",L,"correct_exact","correct"),
  ("The compass is not in the northern archive.",L,"negated_target","incorrect"),("southern vault",L,"wrong_alternative","incorrect"),("northern archive or southern vault",L,"mixed","incorrect"),
  ("Not the southern vault, but the northern archive.",L,"correct_with_rejection","correct"),("AMBIGUOUS",L,"wrong_special","incorrect"),
  ("UNKNOWN",U,"correct_exact","correct"),("Not UNKNOWN",U,"negated_target","incorrect"),("UNKNOWN - northern archive",U,"mixed","incorrect"),("container Z0",U,"wrong_alternative","incorrect"),
  ("The location cannot be determined.",U,"abstain_nonstandard","review"),("AMBIGUOUS",A,"correct_exact","correct"),("Not AMBIGUOUS",A,"negated_target","incorrect"),
  ("",L,"empty","incorrect"),("To determine this we must analyze each record carefully.",L,"no_answer","incorrect"),("It is not where the report says.",L,"unparsed_negation","review"),
  ("The compass isn't in the southern vault; it is in the northern archive.",L,"correct_with_rejection","correct"),
  ("Northern archive is not the correct location.",L,"negated_target","incorrect"),("It is not UNKNOWN; it is northern archive.",L,"correct_verbose","correct"),
  ("Northern archive. No, not the northern archive.",L,"conflicting_mentions","review"),("It is not certain, but northern archive.",L,"unattributed_negation","review")]
 res=[];bad=[]
 for txt,c,ec,es in F_:
  s=score(txt,c);ok=(s["category"]==ec and s["semantic"]==es);res.append((ok,txt,c["target"],ec,s["category"],es,s["semantic"]))
  if not ok:bad.append(res[-1])
 for ok,txt,t,ec,gc,es,gs in res:say(f"  {'PASS' if ok else 'FAIL'} | {txt!r:<75} target={t:<16} expected={ec}/{es} got={gc}/{gs}")
 if bad:raise TechStop(f"scorer self-test failed on {len(bad)} fixture(s): {bad}")
 return len(res)
def assign_foreign():
 for c in CASES:
  c["foreign_id"]=None;c["foreign_reason"]=None;c["foreign_len_diff"]=None;tol=max(LEN_TOL_ABS,LEN_TOL_REL*c["src_tokens"]);c["foreign_tol"]=tol
  if c["target"] in SPECIAL:c["foreign_reason"]="uninformative: UNKNOWN/AMBIGUOUS can be produced without any source information";continue
  best=None
  for x in CASES:
   if x["tmpl"]!=c["tmpl"] or x["ent"]==c["ent"] or not x["valid"]:continue
   if any(has(x["source"],p) for p in [c["target"]]+c["alts"] if p not in SPECIAL):continue
   d=abs(x["src_tokens"]-c["src_tokens"])
   if d<=tol and (best is None or d<best[0]):best=(d,x)
  if best:c["foreign_id"]=best[1]["id"];c["foreign_len_diff"]=best[0]
  else:c["foreign_reason"]=f"no same-template source within ±{tol:.1f} tokens without target/alternative overlap"
def summarize(integ):
 rows={r["id"]:r for r in ROWS}
 def st(r,a):return (r or {}).get("arms",{}).get(a,{})
 def valid(r,a):x=st(r,a);return x.get("status")=="OK" and bool(x.get("valid"))
 def ps(r,a):return valid(r,a) and st(r,a).get("pass")==1
 FAM=OrderedDict()
 for vk,groups in EXPECTED.items():
  G=[];units=[]
  for g,mids in groups.items():
   rs=[rows.get(m) for m in mids];missing=[m for m,r in zip(mids,rs) if r is None or not r.get("complete")]
   inval={a:[f"{m}: {st(rows.get(m),a).get('status')} {st(rows.get(m),a).get('invalid_reason') or ''}".strip() for m,r in zip(mids,rs) if r is not None and not valid(r,a)] for a in CORE}
   uv=not missing and not any(inval.values());G.append(dict(group=g,expected=mids,missing=missing,invalid={a:v for a,v in inval.items() if v},unit_valid=uv))
   if uv:units.append({a:all(ps(r,a) for r in rs) for a in ARMS if all(valid(r,a) for r in rs)})
  nexp=len(groups);nu=len(units);cnt={a:sum(1 for u in units if u.get(a)) for a in ARMS};nva={a:sum(1 for u in units if a in u) for a in ARMS}
  rate={a:(cnt[a]/nva[a] if nva[a] else None) for a in ARMS}
  def paired(a,b):return dict(a_pass_b_fail=sum(1 for u in units if a in u and b in u and u[a] and not u[b]),b_pass_a_fail=sum(1 for u in units if a in u and b in u and u[b] and not u[a]))
  P=dict(ctx_vs_nat=paired("CONTEXT","NATIVE_KV"),nat_vs_pkv=paired("NATIVE_KV","PKV_D120"),ctx_vs_pkv=paired("CONTEXT","PKV_D120"),pkv_vs_own=paired("PKV_D120","PKV_OWNSINK"))
  vr=[rows.get(m) for mids in groups.values() for m in mids];ordc={a:dict(pairs=0,changed=0) for a in ARMS}
  for g,mids in groups.items():
   mem={rows[m]["member"]:rows[m] for m in mids if m in rows}
   for base in ({m[:-4] for m in mem if m.endswith(("_fwd","_rev"))}|({""} if ("fwd" in mem or "rev" in mem) else set())):
    fw=mem.get(base+"_fwd" if base else "fwd");rv=mem.get(base+"_rev" if base else "rev")
    if fw is None or rv is None:continue
    for a in ARMS:
     if valid(fw,a) and valid(rv,a):ordc[a]["pairs"]+=1;ordc[a]["changed"]+=int(ps(fw,a)!=ps(rv,a))
  sens=dict(pkv_pass_incl_sink_out=sum(1 for r in vr if r and st(r,"PKV_D120").get("status")=="OK" and st(r,"PKV_D120").get("pass")==1),pkv_ok_incl_sink_out=sum(1 for r in vr if r and st(r,"PKV_D120").get("status")=="OK"),
   sink_out_cases=sum(1 for r in vr if r and r.get("memory") and not r["memory"]["sink_ok"]),review=sum(1 for r in vr if r for a in ARMS if st(r,a).get("review")==1))
  lift=(rate["PKV_D120"]-rate["VANILLA"]) if nu and rate["PKV_D120"] is not None and rate["VANILLA"] is not None else None
  if nu<nexp:lab=f"INCOMPLETE / TECHNICALLY UNINTERPRETABLE ({nu}/{nexp} valid units)"
  elif rate["CONTEXT"]<CTX_MIN:lab="TASK NOT SOLVED WITH VISIBLE TEXT (uninterpretable)"
  elif P["ctx_vs_nat"]["a_pass_b_fail"]>0:lab="CACHE-PATH GAP (compression effect not assessable)"
  elif lift is None or lift<LIFT_MIN:lab="NOT MEMORY-DEPENDENT (uninformative)"
  elif P["nat_vs_pkv"]["a_pass_b_fail"]==0:lab="SUPPORTED (pilot; no paired loss vs NATIVE)"
  elif P["nat_vs_pkv"]["a_pass_b_fail"]==1:lab="PARTIALLY SUPPORTED (1 paired loss vs NATIVE)"
  else:lab="NOT SUPPORTED (≥2 paired losses vs NATIVE)"
  if integ.get("status")!="PASS" and lab.startswith(("SUPPORTED","PARTIALLY")):lab=f"UNCONFIRMED — integrity {integ.get('status')} ({lab})"
  FAM[vk]=dict(expected_groups=nexp,valid_units=nu,unit_pass=cnt,unit_valid_by_arm=nva,unit_rate=rate,lift_vs_vanilla=lift,paired=P,order=ordc,sensitivity=sens,groups=G,label=lab)
 plan=dict(cases=len(CASES),valid_budget=sum(1 for c in CASES if c.get("valid")),attempted=len(ROWS),complete=sum(1 for r in ROWS if r.get("complete")),
  generations_planned=dict(VANILLA_unique_questions=len({c["question"] for c in CASES if c.get("valid")}),per_memory_arm=sum(1 for c in CASES if c.get("valid")),FOREIGN_LENMATCH=sum(1 for c in CASES if c.get("valid") and c.get("foreign_id"))),
  generations=dict(CTX["gens"]),valid_by_arm={a:sum(1 for r in ROWS if valid(r,a)) for a in ARMS},status_by_arm={a:dict(Counter(st(r,a).get("status") for r in ROWS)) for a in ARMS})
 return dict(families=FAM,plan=plan)
HYP=OrderedDict([("H0 baseline single fact + counterfactual",["single"]),("H1 date-based selection",["latest","historical","conflict_k2","conflict_k4","conflict_k8"]),("H2 source priority",["authority"]),("H3 explicit correction",["correction"]),
 ("H4 unresolved conflict vs consistent repetition",["ambiguity"]),("H5 composition / inference",["two_hop","three_hop","missing"]),("H6 distractor load / slot position",["load_n008","load_n032","load_n128","load_pos_early","load_pos_late"])])
def print_summary(S,integ):
 say("="*110);say(f"TEST{TEST} RESULT SUMMARY — run {RUN_ID}");say("NOTE: technical PASS ≠ hypothesis PASS. Pilot scale (6 entity families). Units = order pairs / matched sets; a unit passes only if all members pass.")
 say(f"integrity: {integ.get('status')} {integ.get('detail','')}");say("plan/attempt/valid: "+json.dumps(S["plan"],ensure_ascii=False))
 for vk,f in S["families"].items():
  r=f["unit_rate"];say(f"  {vk:<16} units {f['valid_units']}/{f['expected_groups']} | "+" ".join(f"{ABBR[a]} {f['unit_pass'][a]}/{f['unit_valid_by_arm'][a]}" for a in ARMS)+f" | lift={('%.2f'%f['lift_vs_vanilla']) if f['lift_vs_vanilla'] is not None else 'n/a'} | {f['label']}")
  P=f["paired"];say(f"      paired: CTX>NAT {P['ctx_vs_nat']['a_pass_b_fail']} NAT>CTX {P['ctx_vs_nat']['b_pass_a_fail']} | NAT>PKV {P['nat_vs_pkv']['a_pass_b_fail']} PKV>NAT {P['nat_vs_pkv']['b_pass_a_fail']} | PKV>OWN {P['pkv_vs_own']['a_pass_b_fail']} OWN>PKV {P['pkv_vs_own']['b_pass_a_fail']} | order changed "+" ".join(f"{ABBR[a]} {o['changed']}/{o['pairs']}" for a,o in f["order"].items())+f" | sink-out cases {f['sensitivity']['sink_out_cases']} (PKV incl. them: {f['sensitivity']['pkv_pass_incl_sink_out']}/{f['sensitivity']['pkv_ok_incl_sink_out']}, not used for verdict) | review {f['sensitivity']['review']}")
  for g in f["groups"]:
   if not g["unit_valid"]:say(f"      ✗ group {g['group']}: missing={g['missing']} invalid={g['invalid']}")
 say("HYPOTHESES:")
 for h,vks in HYP.items():say(f"  {h}: "+"; ".join(f"{vk} → {S['families'][vk]['label']}" for vk in vks if vk in S["families"]))
 say("="*110)
#<<CPU_END>>
def infer(fn):
 def w(*a,**k):
  gpu_guard()
  with torch.inference_mode():return fn(*a,**k)
 w.__name__=fn.__name__;return w
def enc_ids(s):return tok(s,add_special_tokens=False).input_ids
def ltk(n=1):return {"logits_to_keep":n} if LTK_OK else {}
@infer
def kv_from_ids(ids,use_ltk=True):
 CTX["last"].update(op="forge",T=len(ids))
 o=model(input_ids=torch.tensor([ids],device=DEVICE),output_hidden_states=True,use_cache=False,return_dict=True,**(ltk(1) if use_ltk else {}));K=[];V=[]
 for L in range(NL):
  h=o.hidden_states[L][0];z=layers[L].input_layernorm(h);a=layers[L].self_attn;K.append(a.k_proj(z).contiguous());V.append(a.v_proj(z).contiguous())
 del o
 return K,V
def forge(s):
 if "QUESTION:" in s:raise RuntimeError("Source-only forge: reserved QUESTION marker")
 ids=[PAD]+enc_ids(s+SEP);K,V=kv_from_ids(ids);return dict(ids=ids,T=len(ids),K=K,V=V)
@infer
def install(K,V):
 T=K[0].shape[0];cos,sin=model.model.rotary_emb(K[0][None],torch.arange(T,device=DEVICE)[None]);KK=[];VV=[]
 for L in range(NL):
  k=K[L].reshape(1,T,NKV,HD).transpose(1,2);v=V[L].reshape(1,T,NKV,HD).transpose(1,2)
  KK.append(apply_rotary_pos_emb(k,k,cos,sin,unsqueeze_dim=1)[1].contiguous());VV.append(v.contiguous())
 return tuple(KK),tuple(VV),T
def mkcache(kv):
 gpu_guard()
 try:c=DynamicCache(config=cfg)
 except Exception:c=DynamicCache()
 for L in range(NL):c.update(kv[0][L].clone(),kv[1][L].clone(),L)
 if c.get_seq_length()!=kv[2]:raise RuntimeError(f"cache length {c.get_seq_length()} != {kv[2]} after install")
 return c
def kvget(c,L):return (c.layers[L].keys,c.layers[L].values) if hasattr(c,"layers") else (c.key_cache[L],c.value_cache[L])
def kv_equal(a,b):gpu_guard();return all(torch.equal(x,y) for p,q in zip(a,b) for x,y in zip(p,q))
def all_finite(ts):gpu_guard();return bool(torch.stack([torch.isfinite(t).all() for t in ts]).all().item())
def encode(f,d=D):gpu_guard();return {n:[torch.einsum("thi,hid->thd",f[n][L][1:].float().view(-1,NKV,HD)-CB[L][n][0],CB[L][n][1][:,:,:d]).to(torch.bfloat16) for L in range(NL)] for n in ("K","V")}
def decode(code,sink=None,d=D):
 gpu_guard();sink=sink or SINK
 out={n:[torch.cat([sink[n][L],(CB[L][n][0]+torch.einsum("thd,hid->thi",code[n][L].float(),CB[L][n][1][:,:,:d])).reshape(-1,KVD).to(torch.bfloat16)]) for L in range(NL)] for n in ("K","V")}
 return out["K"],out["V"]
def code_sha(code):
 gpu_guard();h=hashlib.sha256()
 for n in ("K","V"):
  for x in code[n]:h.update(x.float().cpu().numpy().tobytes())
 return h.hexdigest()
@infer
def sink_metrics(f):
 rows=[]
 for n in ("K","V"):
  for L in range(NL):
   x=f[n][L][0].float();s=SINK[n][L][0].float();d=float((x-s).norm());sn=float(s.norm());sc=float(f[n][L][1:].float().norm(dim=-1).mean()) if f["T"]>1 else sn
   rows.append(dict(tensor=n,layer=L,cos=float(F.cosine_similarity(x,s,0)),abs_diff=d,sink_norm=sn,x0_norm=float(x.norm()),row_scale=sc,rel_row=d/max(sc,1e-12),rel_sink=d/max(sn,1e-12),finite=bool(torch.isfinite(x).all())))
 mc=min(rows,key=lambda r:r["cos"]);mr=max(rows,key=lambda r:r["rel_row"])
 return dict(min_cos=mc["cos"],min_cos_at=mc,max_rel_row=mr["rel_row"],max_rel_at=mr,finite_pos0=all(r["finite"] for r in rows),start_token_ok=bool(f["ids"][0]==PAD))
def fmt_sink(m):a=m["min_cos_at"];b=m["max_rel_at"];return f"min_cos={m['min_cos']:.6f}@{a['tensor']}L{a['layer']}(|sink|={a['sink_norm']:.3g},|x0|={a['x0_norm']:.3g},|Δ|={a['abs_diff']:.3g}) max|Δ|/row={m['max_rel_row']:.2e}@{b['tensor']}L{b['layer']}(|Δ|/|sink|={b['rel_sink']:.2e},row={b['row_scale']:.3g})"
@infer
def build_rep(c):
 CTX["last"]=dict(op="build_rep",case=c["id"],src_tokens=c["src_tokens"])
 f=forge(c["source"]);m=sink_metrics(f);code=encode(f);seal=code_sha(code);K,V=decode(code);own={n:[f[n][L][:1].clone() for L in range(NL)] for n in ("K","V")};Ko,Vo=decode(code,own)
 fin=dict(raw=all_finite(f["K"]+f["V"]),code=all_finite(code["K"]+code["V"]),decoded=all_finite(K+V+Ko+Vo))
 nkv=install(f["K"],f["V"]);pkv=install(K,V);okv=install(Ko,Vo);fin["installed"]=all_finite(list(nkv[0])+list(nkv[1])+list(pkv[0])+list(pkv[1])+list(okv[0])+list(okv[1]))
 fid={n:float(np.mean([float(F.cosine_similarity(dec[L][1:].float().reshape(-1),f[n][L][1:].float().reshape(-1),0)) for L in range(NL)])) for n,dec in (("K",K),("V",V))}
 CTX["last"].update(T=f["T"],installed_k_shape=list(nkv[0][0].shape),code_k_shape=list(code["K"][0].shape),sink=dict(min_cos=m["min_cos"],max_rel_row=m["max_rel_row"]))
 rep=dict(sha=c["sha"],T=f["T"],records=c["records"],sink=m,sink_ok=bool(m["max_rel_row"]<=SINK_GATE),finite=fin,finite_ok=all(fin.values()) and m["finite_pos0"],start_ok=bool(m["start_token_ok"] and f["T"]==c["src_tokens"]),seal=seal,code=code,fidelity=fid,nkv=nkv,pkv=pkv,okv=okv,is_long=f["T"]>SHORT_T)
 del f,K,V,Ko,Vo;return rep
REPS=OrderedDict()
def get_rep(c):
 gpu_guard();s=c["sha"]
 if s in REPS:REPS.move_to_end(s);return REPS[s]
 rep=build_rep(c);REPS[s]=rep;longs=[k for k,v in REPS.items() if v["is_long"]]
 while len(longs)>LONG_KEEP:k=longs.pop(0);ev=REPS.pop(k);CTX["seal_checks"].append(code_sha(ev["code"])==ev["seal"]);del ev
 return rep
@infer
def gen(c,arm,kv,guard,ck="main"):
 qids=enc_ids(FMT.format(q=c["question"]));T=kv[2] if kv is not None else 0
 pre=[PAD]+enc_ids(c["source"]+SEP) if arm=="CONTEXT" else [PAD]*T;need=len(pre)+len(qids)+MAX_NEW
 if need>MAX_TOTAL:return dict(status="INVALID_BUDGET",valid=False,invalid_reason=f"prompt+generation {need} > {MAX_TOTAL}")
 if arm!="CONTEXT":
  if set(pre)-{PAD}:return dict(status="LEAK",valid=False,invalid_reason="non-PAD dummy prefix")
  txt=tok.decode(pre+qids)
  for r in guard:
   if r.strip() and r.strip() in txt:return dict(status="LEAK",valid=False,invalid_reason=f"record present in readout prompt: {r}")
 CTX["last"]=dict(op="generate",case=c["id"],arm=arm,T=T,prompt_tokens=len(pre)+len(qids),q_tokens=len(qids))
 ids=torch.tensor([pre+qids],device=DEVICE);cache=mkcache(kv) if kv is not None else None;CTX["gens"][ck+"_attempted"]+=1
 torch.cuda.synchronize();t0=time.perf_counter();y=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),past_key_values=cache,**GEN);torch.cuda.synchronize()
 new=y[0,ids.shape[1]:].tolist();exp=len(pre)+len(qids)+len(new)-1;cl=None if cache is None else int(cache.get_seq_length());CTX["gens"][ck+"_completed"]+=1
 out=dict(status="OK",text=tok.decode(new,skip_special_tokens=True).strip(),new_tokens=len(new),seconds=time.perf_counter()-t0,memory_slots=T,prompt_tokens=len(pre)+len(qids),truncated=bool(new and len(new)==MAX_NEW and new[-1] not in EOS),cache_len=cl,cache_len_expected=exp if cache is not None else None,cache_len_ok=(cl==exp) if cache is not None else None)
 out["valid"]=out["cache_len_ok"] is not False;out["invalid_reason"]=None if out["valid"] else f"cache length {cl} != expected {exp} (prompt+new-1)"
 return out
VAN={}
def meta(c):return {k:c.get(k) for k in ("id","family","vkey","group","member","ent","load","pos","k","records","source","question","target","alts","src_tokens","q_tokens","foreign_id","foreign_reason","foreign_len_diff")}
def run_case(c,ck="main"):
 row=meta(c);row.update(memory=None,arms={},complete=False);rep=None
 try:rep=get_rep(c)
 except GpuBlocked as ex:
  for a in ("NATIVE_KV","PKV_D120","PKV_OWNSINK"):row["arms"][a]=dict(status="NOT_RUN",valid=False,invalid_reason=str(ex))
 except Exception as ex:
  report(ex,c,"build_rep")
  for a in ("NATIVE_KV","PKV_D120","PKV_OWNSINK"):row["arms"][a]=dict(status="RUNTIME_ERROR",valid=False,invalid_reason=f"memory build failed: {type(ex).__name__}: {ex}")
 if rep:row["memory"]=dict(T=rep["T"],sink=rep["sink"],sink_ok=rep["sink_ok"],finite=rep["finite"],finite_ok=rep["finite_ok"],start_ok=rep["start_ok"],fidelity=rep["fidelity"],code_sha=rep["seal"])
 for arm in ARMS:
  if arm in row["arms"]:continue
  if not gpu_alive():row["arms"][arm]=dict(status="NOT_RUN",valid=False,invalid_reason="stopped after fatal CUDA error");continue
  try:
   if arm=="VANILLA":
    if c["question"] in VAN:out=dict(VAN[c["question"]]);out["reused"]=True;CTX["gens"][ck+"_vanilla_reused"]+=1
    else:out=gen(c,arm,None,c["records"],ck);VAN[c["question"]]=dict(out)
   elif arm=="CONTEXT":out=gen(c,arm,None,(),ck)
   elif rep is None:out=dict(status="NOT_RUN",valid=False,invalid_reason="no memory representation")
   elif not(rep["finite_ok"] and rep["start_ok"]):out=dict(status="TECHNICAL_INVALID",valid=False,invalid_reason=f"finite={rep['finite']} start_ok={rep['start_ok']} — generation not attempted")
   elif arm=="NATIVE_KV":out=gen(c,arm,rep["nkv"],c["records"],ck)
   elif arm=="PKV_D120":
    out=gen(c,arm,rep["pkv"],c["records"],ck)
    if out.get("status")=="OK" and out["valid"] and not rep["sink_ok"]:out["valid"]=False;out["invalid_reason"]=f"shared-SINK deviation outside pre-registered gate ({rep['sink']['max_rel_row']:.2e} > {SINK_GATE:.2e}); reported separately"
   elif arm=="PKV_OWNSINK":out=gen(c,arm,rep["okv"],c["records"],ck)
   else:
    if not c.get("foreign_id"):out=dict(status="NOT_APPLICABLE",valid=False,invalid_reason=c.get("foreign_reason"))
    else:
     fr=get_rep(BYID[c["foreign_id"]])
     if not(fr["finite_ok"] and fr["start_ok"] and fr["sink_ok"]):out=dict(status="TECHNICAL_INVALID",valid=False,invalid_reason="foreign memory failed finite/start/sink checks")
     else:out=gen(c,arm,fr["pkv"],c["records"]+fr["records"],ck);out["foreign_id"]=c["foreign_id"];out["foreign_len_diff"]=c["foreign_len_diff"]
  except GpuBlocked as ex:out=dict(status="NOT_RUN",valid=False,invalid_reason=str(ex))
  except Exception as ex:report(ex,c,arm);out=dict(status="RUNTIME_ERROR",valid=False,invalid_reason=f"{type(ex).__name__}: {ex}")
  if out.get("status")=="OK":out.update(score(out["text"],c))
  row["arms"][arm]=out
 row["complete"]=all(row["arms"].get(a,{}).get("status") not in (None,"NOT_RUN") for a in ARMS);return row
def sym(r,a):
 x=r["arms"].get(a,{});s=x.get("status")
 if s=="OK":return ("R" if x.get("review") else str(x.get("pass")))+("" if x.get("valid") else "*")
 return {"NOT_APPLICABLE":"-","INVALID_BUDGET":"I","LEAK":"L","TECHNICAL_INVALID":"T","NOT_RUN":"N"}.get(s,"E")
def print_case(i,N,r):
 m=r["memory"];flags=[]
 if m and not m["sink_ok"]:flags.append("SINK-OUT")
 A=r["arms"];g=lambda a,k:A.get(a,{}).get(k)
 if any(g(a,"status") not in ("OK","NOT_APPLICABLE") for a in ARMS):flags.append("TECH")
 if any(g(a,"review") for a in ARMS):flags.append("REVIEW")
 if g("CONTEXT","pass")!=g("PKV_D120","pass"):flags.append("CTX≠PKV")
 if g("NATIVE_KV","pass")!=g("PKV_D120","pass"):flags.append("NAT≠PKV")
 if g("CONTEXT","pass")!=g("NATIVE_KV","pass"):flags.append("CTX≠NAT")
 if g("FOREIGN_LENMATCH","pass")==1:flags.append("FOREIGN-HIT")
 if any(g(a,"status")!="OK" or not g(a,"valid") or g(a,"pass")!=1 or g(a,"review") for a in ("CONTEXT","NATIVE_KV","PKV_D120")):flags.append("CORE-FAIL/REVIEW/INVALID")
 say(f"{i:03d}/{N} {r['id']:<24} T={(m['T'] if m else 0):5d} | "+" ".join(f"{ABBR[a]}:{sym(r,a)}" for a in ARMS)+(" | ⚠ "+",".join(flags) if flags else ""))
 if flags:
  say(f"      target={r['target']!r} alts={r['alts']}"+(f" | sink: {fmt_sink(m['sink'])}" if m and not m["sink_ok"] else ""))
  for a in ARMS:
   x=A.get(a,{});say(f"      [{a}] status={x.get('status')} valid={x.get('valid')} cat={x.get('category')} sem={x.get('semantic')} trunc={x.get('truncated')} cache_len={x.get('cache_len')}/{x.get('cache_len_expected')}"+(f" reason={x.get('invalid_reason')}" if x.get('invalid_reason') else "")+(f"\n        raw={x.get('text')!r}" if x.get("status")=="OK" else ""))
def ckpt(obj,name="checkpoint.jsonl"):
 if not CTX["file_io_ok"]:return
 try:
  with (RUN/name).open("ab") as fh:fh.write(canon(obj)+b"\n")
 except Exception as ex:CTX["file_io_ok"]=False;report(ex,extra=f"file write failed ({name}); further file writes disabled — console output remains complete")
@infer
def last_logits(ids,n,cache=None,mask_len=None):
 o=model(input_ids=torch.tensor([ids],device=DEVICE),attention_mask=torch.ones(1,mask_len or len(ids),device=DEVICE,dtype=torch.long),past_key_values=cache,use_cache=cache is not None,**ltk(n))
 return o.logits[0,-n:].float()
@infer
def cache_probe(c):
 CTX["last"]=dict(op="cache_probe",case=c["id"]);f=forge(c["source"]);nkv=install(f["K"],f["V"]);T=f["T"];qids=enc_ids(FMT.format(q=c["question"]))
 c2=mkcache(nkv);ids=torch.tensor([[PAD]*T+qids],device=DEVICE)
 g=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),past_key_values=c2,return_dict_in_generate=True,output_scores=True,**dict(GEN,max_new_tokens=PROBE_STEPS))
 S=g.sequences[0,ids.shape[1]:].tolist();n=len(S);gs=torch.stack([s[0].float() for s in g.scores])
 lc=last_logits(f["ids"]+qids+S[:n-1],n);lm=last_logits(qids+S[:n-1],n,cache=mkcache(nkv),mask_len=T+len(qids)+n-1)
 tvd=lambda a,b:float((0.5*(a.softmax(-1)-b.softmax(-1)).abs().sum(-1)).max())
 steps=[dict(step=j,generated=S[j],argmax_context_forward=int(lc[j].argmax()),argmax_cache_forward=int(lm[j].argmax()),argmax_generate_scores=int(gs[j].argmax())) for j in range(n)]
 r=dict(case=c["id"],T=T,q_tokens=len(qids),steps=n,generated=tok.decode(S),token_ids=S,argmax_all_equal=all(s["argmax_context_forward"]==s["argmax_cache_forward"]==s["argmax_generate_scores"]==s["generated"] for s in steps),
  tvd_context_vs_cache=tvd(lc,lm),tvd_cache_vs_generate=tvd(lm,gs),maxabs_logit_context_vs_cache=float((lc-lm).abs().max()),maxabs_logit_cache_vs_generate=float((lm-gs).abs().max()),bit_exact_cache_vs_generate=bool(torch.equal(lm,gs)),
  cache_len_after_generate=int(c2.get_seq_length()),cache_len_expected=T+len(qids)+n-1,step_detail=steps,
  conditions="context: [PAD]+source+SEP+question+generated[:-1], positions 0..; cache: question+generated[:-1] after T installed slots, attention_mask=ones(T+len), positions T..; generate: PAD×T dummy prefix + question with the same installed cache")
 r["cache_len_ok"]=r["cache_len_after_generate"]==r["cache_len_expected"];r["pass"]=bool(r["argmax_all_equal"] and r["tvd_context_vs_cache"]<=PROBE_TVD_MAX and r["tvd_cache_vs_generate"]<=PROBE_TVD_MAX and r["cache_len_ok"])
 del f,nkv,c2;return r
@infer
def native_audit(ids):
 K,V=kv_from_ids(ids);kv=install(K,V);nat=model(input_ids=torch.tensor([ids],device=DEVICE),use_cache=True,**ltk(1)).past_key_values
 r={n:max(float((kv[i][L].float()-kvget(nat,L)[i].float()).norm()/kvget(nat,L)[i].float().norm().clamp_min(1e-12)) for L in range(NL)) for i,n in enumerate(("K","V"))}
 del K,V,kv,nat;return r
NEUTRAL={}
def neutral_ids(n):
 if n in NEUTRAL:return list(NEUTRAL[n])
 if n<=0:return [PAD]
 txt="";i=0
 while True:
  txt=(txt+" "+CORPUS[i%len(CORPUS)]).strip();i+=1;ids=enc_ids(txt+SEP)
  if len(ids)>=n:NEUTRAL[n]=[PAD]+ids;return [PAD]+ids
def integrity_check():
 if not CTX["model_loaded"]:return dict(status="NOT_EVALUATED",detail="model never loaded")
 if not gpu_alive():return dict(status="NOT_EVALUATED",detail="fatal CUDA error — no GPU operations performed after it")
 try:
  FP1,SENT1=sentinel();st=dict(sampled_weight_sentinel_unchanged=(FP1==FP0 and SENT1==SENT0),frozen_eval=(not model.training and all(not p.requires_grad for p in model.parameters())),attn_sdpa=getattr(cfg,"_attn_implementation",None)=="sdpa",seals=all(CTX["seal_checks"]) if CTX["seal_checks"] else None)
  ok=st["sampled_weight_sentinel_unchanged"] and st["frozen_eval"] and st["attn_sdpa"] and st["seals"] is not False
  return dict(status="PASS" if ok else "FAIL",detail=json.dumps(st),checks=st,fingerprint_before=FP0,fingerprint_after=FP1,sentinel_before=SENT0,sentinel_after=SENT1,note="sampled sentinel; not a full cryptographic verification of all weights")
 except Exception as ex:report(ex,extra="integrity check");return dict(status="NOT_EVALUATED",detail=f"{type(ex).__name__}: {ex}")
# ================================================== RUN ==================================================
RUN=None;SUMMARY=None;INTEG=dict(status="NOT_EVALUATED",detail="not reached")
try:
 say("="*110);say(f"TEST{TEST} — NIRVANA PKV D120 — SELECTION / COMPOSITION / LOAD (pre-registered pilot) | run {RUN_ID}");say("="*110)
 stage("cpu_setup","cases + scorer self-test");CASES=build_cases();BYID={c["id"]:c for c in CASES};assert len(BYID)==len(CASES)
 for c in CASES:EXPECTED.setdefault(c["vkey"],OrderedDict()).setdefault(c["group"],[]).append(c["id"])
 say(f"  scorer self-test: {scorer_selftest()} fixtures PASS")
 try:
  ROOT=Path("/content/AKBASCORE_TEST438c") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST438c");RUN=ROOT/RUN_ID;RUN.mkdir(parents=True,exist_ok=True)
 except Exception as ex:CTX["file_io_ok"]=False;report(ex,extra="output folder unavailable; continuing console-only")
 stage("imports")
 for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
  if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
 import torch,transformers;import torch.nn.functional as F
 from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
 from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
 os.environ["TOKENIZERS_PARALLELISM"]="false";CTX["env"].update(torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda)
 if not torch.cuda.is_available():raise TechStop("CUDA not available — select an A100 GPU runtime")
 DEVICE=torch.device("cuda");CTX["env"]["gpu"]=torch.cuda.get_device_name(0);torch.set_grad_enabled(False)
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);say("  env:",json.dumps(CTX["env"],ensure_ascii=False))
 stage("model_load");t0=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
 tv=tuple(int(re.sub(r"\D","",v) or 0) for v in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
 model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}).eval()
 for p in model.parameters():p.requires_grad_(False)
 cfg=model.config;layers=model.model.layers;H=cfg.hidden_size;NH=cfg.num_attention_heads;NKV=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None) or H//NH;KVD=NKV*HD
 if (len(layers),H,NH,NKV,HD)!=(28,3584,28,4,128):raise TechStop(f"architecture mismatch {(len(layers),H,NH,NKV,HD)}")
 if next(model.parameters()).dtype!=torch.bfloat16 or getattr(cfg,"use_sliding_window",False):raise TechStop("dtype/sliding-window mismatch")
 PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id;ge=model.generation_config.eos_token_id;EOS=sorted({tok.eos_token_id,*(ge if isinstance(ge,(list,tuple)) else [ge])}-{None})
 GEN=dict(max_new_tokens=MAX_NEW,do_sample=False,repetition_penalty=1.0,use_cache=True,pad_token_id=PAD,eos_token_id=EOS);CTX["env"]["attn"]=getattr(cfg,"_attn_implementation",None)
 FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
 def sentinel():
  gpu_guard()
  with torch.inference_mode():
   h=hashlib.sha256();s=[]
   for i,t in enumerate(FP_T):
    s.append(float(t.sum(dtype=torch.float32)));flat=t.detach().reshape(-1);n=flat.numel();c=min(256,n);offs=sorted({k*(n-c)//15 for k in range(16)})
    h.update(f"{i}|{tuple(t.shape)}|{t.dtype}|{n}|{offs}|".encode());h.update(torch.cat([flat[o:o+c] for o in offs]).float().cpu().numpy().tobytes())
   return s,h.hexdigest()
 FP0,SENT0=sentinel()
 if sentinel()!=(FP0,SENT0):raise TechStop("weight sentinel not repeatable")
 CTX["model_loaded"]=True
 def set_attn(x):
  gpu_guard()
  try:model.set_attn_implementation(x)
  except Exception:cfg._attn_implementation=x
  if getattr(cfg,"_attn_implementation",None)!=x:raise RuntimeError(f"attention switch to {x} failed")
 torch.cuda.synchronize();say(f"  loaded in {time.perf_counter()-t0:.1f}s | PAD/start id={PAD} | EOS={EOS} | attn={CTX['env']['attn']}")
 stage("forge_equivalence","logits_to_keep + determinism");LTK_OK=False
 _ids=[PAD]+enc_ids(" ".join(CORPUS[:6])+SEP);_a=kv_from_ids(_ids,False);_b=kv_from_ids(_ids,False);FORGE_DET=kv_equal(_a,_b);LTK_SUPPORTED="logits_to_keep" in inspect.signature(model.forward).parameters
 if LTK_SUPPORTED:LTK_OK=True;_c=kv_from_ids(_ids,True);LTK_OK=kv_equal(_a,_c);del _c
 del _a,_b;DIAG.update(forge_deterministic=FORGE_DET,ltk_supported=LTK_SUPPORTED,ltk_used=LTK_OK)
 say(f"  forge deterministic={FORGE_DET} | logits_to_keep supported={LTK_SUPPORTED} used={LTK_OK}"+("" if LTK_OK else " (FALLBACK: full logits)"))
 stage("codebook");t0=time.perf_counter();CO=[forge(s) for s in CORPUS];SINK={n:[CO[0][n][L][:1].clone() for L in range(NL)] for n in ("K","V")};CB=[]
 for L in range(NL):
  e={}
  for n in ("K","V"):
   R=torch.cat([c[n][L][1:] for c in CO]).float().view(-1,NKV,HD);MU=[];B=[];EV=[]
   for h in range(NKV):
    X=R[:,h];mu=X.mean(0);_,S,Vh=torch.linalg.svd(X-mu,full_matrices=False);m=min(DMAX,Vh.shape[0]);bb=Vh[:m].T.contiguous()
    if m<DMAX:bb=F.pad(bb,(0,DMAX-m))
    cs=S.square().cumsum(0)/S.square().sum();ev=torch.ones(DMAX,device=DEVICE);ev[:min(DMAX,len(cs))]=cs[:DMAX];MU.append(mu);B.append(bb);EV.append(ev)
   e[n]=(torch.stack(MU),torch.stack(B),torch.stack(EV))
  CB.append(e)
 CORPUS_ROWS=sum(c["T"]-1 for c in CO);EV_D={n:float(torch.stack([CB[L][n][2][:,D-1] for L in range(NL)]).mean()) for n in ("K","V")};del CO,R,X
 DIAG.update(codebook_rows=CORPUS_ROWS,explained_variance_at_D=EV_D);say(f"  rows={CORPUS_ROWS} | explained variance @D{D}: K={EV_D['K']:.4f} V={EV_D['V']:.4f} | {time.perf_counter()-t0:.1f}s")
 stage("case_tokens","budget + foreign length matching")
 for c in CASES:
  c["sha"]=sha(c["source"].encode());c["src_tokens"]=1+len(enc_ids(c["source"]+SEP));c["q_tokens"]=len(enc_ids(FMT.format(q=c["question"])));c["valid"]=c["src_tokens"]+c["q_tokens"]+MAX_NEW<=MAX_TOTAL
  if any(s in c["source"] or c["source"] in s for s in CORPUS):raise TechStop(f"case {c['id']} overlaps codebook corpus")
  if not(c["target"] in SPECIAL or has(c["source"],c["target"])):raise TechStop(f"target absent from source in {c['id']}")
 assign_foreign();LMAX=max(c["src_tokens"] for c in CASES)
 say(f"  cases={len(CASES)} groups={sum(len(g) for g in EXPECTED.values())} unique sources={len({c['sha'] for c in CASES})} valid(budget)={sum(c['valid'] for c in CASES)} | source tokens {min(c['src_tokens'] for c in CASES)}–{LMAX} | FOREIGN_LENMATCH applicable={sum(1 for c in CASES if c['foreign_id'])} (tolerance ±max({LEN_TOL_ABS},{LEN_TOL_REL:.0%}))")
 for c in CASES:
  if not c["valid"]:say(f"  PLANNED INVALID (budget): {c['id']} source={c['src_tokens']} question={c['q_tokens']} max_new={MAX_NEW} > {MAX_TOTAL}")
 free,total=torch.cuda.mem_get_info();need=(LONG_KEEP+1)*3*NL*2*LMAX*KVD*2+2*LMAX*H*2*(NL+1);DIAG["memory_estimate"]=dict(free_bytes=free,estimated_peak_extra=need)
 say(f"  GPU free {free/2**30:.1f} GiB | estimated extra peak for representations ≈ {need/2**30:.2f} GiB")
 if need>0.8*free:raise TechStop(f"estimated representation memory {need/2**30:.2f} GiB exceeds 80% of free GPU memory")
 stage("sink_calibration","neutral lengths, answer-independent");t0=time.perf_counter();CAL=[]
 for n in sorted({0,16,64,256,1024,2048,LMAX}):
  CTX["sub"]=f"neutral T≈{n}";ids=neutral_ids(n);K,V=kv_from_ids(ids);m=sink_metrics(dict(ids=ids,T=len(ids),K=K,V=V));m.update(target_len=n,T=len(ids));CAL.append(m);del K,V
  say(f"  T={m['T']:5d} {fmt_sink(m)} finite={m['finite_pos0']}")
 CAL_REL=max(m["max_rel_row"] for m in CAL if m["T"]>1);SINK_GATE=min(max(SINK_ENV_MULT*CAL_REL,SINK_REL_FLOOR),SINK_REL_CAP);CAL_EXCEEDS_CAP=SINK_ENV_MULT*CAL_REL>SINK_REL_CAP
 if not all(m["finite_pos0"] for m in CAL):raise TechStop("non-finite position-0 K/V in neutral calibration")
 CTX["sub"]="eager comparison"
 try:set_attn("eager");ids=neutral_ids(LMAX);K,V=kv_from_ids(ids);EAGER_LONG=sink_metrics(dict(ids=ids,T=len(ids),K=K,V=V));del K,V
 except BaseException as ex:raise Reported(report(ex,extra="eager sink comparison (reported before attention restore)"))
 finally:
  if gpu_alive():
   try:set_attn("sdpa")
   except BaseException as ex2:report(ex2,extra="cleanup: restoring SDPA after eager comparison (secondary; original error, if any, reported above)")
 _f=forge(CORPUS[0]);SINK_REPEAT_EXACT=all(torch.equal(_f[n][L][:1],SINK[n][L]) for n in ("K","V") for L in range(NL));del _f
 DIAG.update(sink_calibration=CAL,sink_envelope=CAL_REL,sink_gate=SINK_GATE,sink_envelope_exceeds_cap=CAL_EXCEEDS_CAP,eager_long_sink=EAGER_LONG,sink_repeat_exact=SINK_REPEAT_EXACT)
 say(f"  eager T≈{LMAX}: {fmt_sink(EAGER_LONG)} | SINK bit-exact on repeat={SINK_REPEAT_EXACT}")
 say(f"  pre-registered gate |Δ|/row ≤ {SINK_GATE:.2e} = min(max({SINK_ENV_MULT}×envelope {CAL_REL:.2e}, floor {SINK_REL_FLOOR}), cap {SINK_REL_CAP})"+(" ⚠ envelope×mult exceeded cap → cap applied" if CAL_EXCEEDS_CAP else ""))
 say("  cap note: the 2% cap on |Δ|/row is an engineering tolerance pre-selected for this pilot; it is not a physical or semantic guarantee. Cosine is reported alongside but not used for gating.")
 stage("native_audit","install(K,V) vs model past_key_values by length");NATIVE_AUDIT={}
 for n in sorted({16,256,2048,LMAX}):CTX["sub"]=f"T≈{n}";NATIVE_AUDIT[str(n)]=native_audit(neutral_ids(n));say(f"  T≈{n}: K={NATIVE_AUDIT[str(n)]['K']:.2e} V={NATIVE_AUDIT[str(n)]['V']:.2e}")
 DIAG["native_audit_by_length"]=NATIVE_AUDIT
 if max(max(d.values()) for d in NATIVE_AUDIT.values())>=5e-2:raise TechStop(f"native install audit failed: {NATIVE_AUDIT}")
 stage("cache_probe","context forward vs cache forward vs generate, multi-step");vc=[c for c in CASES if c["valid"]];PROBES=[]
 for c in (min(vc,key=lambda x:x["src_tokens"]),max(vc,key=lambda x:x["src_tokens"])):
  CTX["sub"]=c["id"];r=cache_probe(c);PROBES.append(r);say("  "+json.dumps(jsafe({k:v for k,v in r.items() if k!="step_detail"}),ensure_ascii=False))
  for s in r["step_detail"]:say(f"     step {s['step']}: generated={s['generated']} ctx={s['argmax_context_forward']} cache={s['argmax_cache_forward']} gen={s['argmax_generate_scores']}")
 DIAG["cache_probes"]=PROBES
 if not all(p["pass"] for p in PROBES):raise TechStop(f"cache probe FAILED (pre-registered: identical greedy tokens on all paths, TVD ≤ {PROBE_TVD_MAX}, cache length = prompt+new−1). Long run not started. Details above.")
 stage("lock");LOCK=dict(test=TEST,model=MODEL_ID,architecture=[NL,H,NH,NKV,HD],D=D,DMAX=DMAX,seed=SEED,PAD=PAD,EOS=EOS,FMT=FMT,SEP=SEP,STYLE=STYLE,max_new=MAX_NEW,max_total=MAX_TOTAL,corpus=CORPUS,cases=[meta(c) for c in CASES],
  sink=dict(policy="PKV_D120 uses shared SINK (unchanged); PKV_OWNSINK diagnostic only",gate=SINK_GATE,envelope=CAL_REL,mult=SINK_ENV_MULT,floor=SINK_REL_FLOOR,cap=SINK_REL_CAP,rule="PKV_D120 generation recorded but excluded from verdict if |Δ|/row > gate"),
  foreign=dict(name="FOREIGN_LENMATCH",tolerance=f"±max({LEN_TOL_ABS}, {LEN_TOL_REL} × source tokens)",not_applicable="special targets; no same-template non-overlapping source within tolerance"),
  scoring="first non-empty line; mention status per phrase (affirmed/negated within 5 preceding words, reset after 'but'); REVIEW_REQUIRED never counted as pass",
  verdict=dict(unit="group (all members)",ctx_min=CTX_MIN,lift_min=LIFT_MIN,retention="paired NATIVE→PKV losses: 0 supported, 1 partial, ≥2 not supported",cache_path="any CONTEXT→NATIVE paired loss blocks compression verdict"),probe=dict(steps=PROBE_STEPS,tvd_max=PROBE_TVD_MAX),forge_ltk=LTK_OK)
 LOCK_SHA=sha(canon(LOCK));DIAG["lock_sha"]=LOCK_SHA;say("  LOCK",LOCK_SHA)
 if CTX["file_io_ok"] and RUN:
  try:(RUN/"experiment_lock.json").write_bytes(canon(LOCK))
  except Exception as ex:CTX["file_io_ok"]=False;report(ex,extra="lock file write failed; console-only from here")
 stage("smoke_test","technical only — semantic results ignored");SMOKE=[]
 for sid in ("single0:orig","miss1:broken_fwd","load2-128:fwd"):
  c=BYID[sid];CTX["sub"]=sid;r=run_case(c,ck="smoke");SMOKE.append(r);print_case(len(SMOKE),3,r)
  if not gpu_alive():break
 probs=[]
 for r in SMOKE:
  m=r["memory"]
  if not m or not(m["finite_ok"] and m["start_ok"]):probs.append(f"{r['id']}: memory finite/start check failed {m and m['finite']}")
  for a,x in r["arms"].items():
   if x.get("status") not in ("OK","NOT_APPLICABLE"):probs.append(f"{r['id']}/{a}: status {x.get('status')} {x.get('invalid_reason')}")
   if x.get("status")=="OK" and (x.get("cache_len_ok") is False or "category" not in x):probs.append(f"{r['id']}/{a}: cache_len_ok={x.get('cache_len_ok')} category={x.get('category')}")
 DIAG["smoke"]=dict(problems=probs,rows=SMOKE);VAN.clear()
 if probs or not gpu_alive():raise TechStop("technical smoke test FAILED — long run not started:\n  "+"\n  ".join(probs or ["fatal error during smoke test"]))
 say("  smoke test: technical PASS (semantic outcomes not inspected)")
 stage("main_run");ORDER=[c for c in CASES if c["src_tokens"]<=SHORT_T]+sorted([c for c in CASES if c["src_tokens"]>SHORT_T],key=lambda c:(c["ent"],c["load"],c["pos"],c["member"]));N=len(ORDER);t_run=time.perf_counter()
 say(f"  {N} cases × {len(ARMS)} arms | legend: 1 pass, 0 fail, R review, * generated but invalid, - not applicable, I budget, L leak, T technical-invalid, E runtime error, N not run")
 for i,c in enumerate(ORDER,1):
  if not gpu_alive():break
  CTX["sub"]=c["id"]
  if not c["valid"]:
   row=meta(c);row.update(memory=None,complete=True,arms={a:dict(status="INVALID_BUDGET",valid=False,invalid_reason=f"source {c['src_tokens']}+question {c['q_tokens']}+{MAX_NEW} > {MAX_TOTAL}") for a in ARMS})
  else:row=run_case(c)
  ROWS.append(row);ckpt(row);print_case(i,N,row)
 DIAG["main_seconds"]=time.perf_counter()-t_run
 if gpu_alive():
  for s,rep in REPS.items():CTX["seal_checks"].append(code_sha(rep["code"])==rep["seal"])
except BaseException as ex:
 if isinstance(ex,Reported):CTX["stop"]=ex.rec
 elif isinstance(ex,TechStop):CTX["stop"]=report(ex,planned=True)
 elif isinstance(ex,KeyboardInterrupt):CTX["stop"]=report(ex,planned=True,extra="interrupted by user")
 else:CTX["stop"]=report(ex)
finally:
 try:
  stage("finalize","integrity");INTEG=integrity_check();say("  integrity:",INTEG.get("status"),INTEG.get("detail",""))
 except BaseException as ex:report(ex,extra="finalize/integrity")
 try:
  stage("finalize","summary");SUMMARY=summarize(INTEG);print_summary(SUMMARY,INTEG)
 except BaseException as ex:report(ex,extra="finalize/summary")
 try:
  stage("finalize","files")
  if CTX["file_io_ok"] and RUN:
   P=dict(test=TEST,run_id=RUN_ID,utc=now(),env=CTX["env"],diagnostics=DIAG,integrity=INTEG,summary=SUMMARY,errors=CTX["errors"],stop=CTX["stop"],fatal=CTX["fatal"],rows=ROWS)
   pb=canon(P);(RUN/f"{RUN_ID}_payload.json").write_bytes(pb);(RUN/"manifest.json").write_bytes(canon(dict(payload=f"{RUN_ID}_payload.json",sha256=sha(pb),bytes=len(pb),utc=now())))
   say(f"  optional files (not needed for diagnosis): {RUN} | payload SHA-256 {sha(pb)}")
  else:say("  files not written (file I/O unavailable) — console output above is the complete record")
 except BaseException as ex:report(ex,extra="finalize/files — console output above remains the complete record")
 st="FATAL CUDA ERROR — RESTART THE COLAB RUNTIME BEFORE ANY FURTHER RUN" if CTX["fatal"] else ("STOPPED: "+CTX["stop"]["type"]+" — "+CTX["stop"]["msg"].splitlines()[0][:200] if CTX["stop"] else "COMPLETED")
 say("="*110);say(f"TEST{TEST} FINAL STATE: {st} | rows {len(ROWS)}/{len(CASES)} | errors reported {len(CTX['errors'])} | integrity {INTEG.get('status')}");say("="*110)
