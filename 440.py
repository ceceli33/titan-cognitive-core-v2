# TEST438f — VARAN 1 RECOVERY: complete TEST438d reproduction + exploratory CPU rescore.
# Single Colab cell / A100. This cell runs the GPU suite again because the old raw payload is unavailable.
# Embedded TEST438d engine, records, generation, SINK gate and original verdict rules are unchanged.
# A new reproduction is not restoration of the old run. Original and post-hoc scores are both reported.
# No parameter Retention Guard, training, optimizer, steering hooks or independent-bank fusion is added.
print("TEST438f | PHASE 1: fresh reproduction of TEST438d; PHASE 2: CPU exploratory rescore",flush=True)
# TEST438d — AKBASCORE NIRVANA PKV D120 — single Colab cell (A100).
# Engine/cases/scorer/verdicts retained from 438c. Local revision: natural-cache probe + compact empty-run summary.
# Full-context vs split-cache TVD is diagnostic; installed vs natural cache, greedy tokens and lengths remain hard gates.
# This revised gate is registered BEFORE the main suite; 438c completed zero main cases. No training/LoRA/hooks/UI/ZIP.
import os,sys,re,json,time,random,hashlib,importlib.util,subprocess,traceback,inspect,platform
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter,OrderedDict
import numpy as np
#<<CPU_BEGIN>>
TEST="438d";SEED=384;D=120;DMAX=128;NL=28;MAX_NEW=32;MAX_TOTAL=6144;SHORT_T=256;LONG_KEEP=3;PROBE_STEPS=6
FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n";STYLE=" Give only the answer on the first line; do not explain."
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
ARMS=("VANILLA","CONTEXT","NATIVE_KV","PKV_D120","PKV_OWNSINK","FOREIGN_LENMATCH");CORE=("VANILLA","CONTEXT","NATIVE_KV","PKV_D120")
ABBR=dict(VANILLA="VAN",CONTEXT="CTX",NATIVE_KV="NAT",PKV_D120="PKV",PKV_OWNSINK="OWN",FOREIGN_LENMATCH="FOR")
SINK_ENV_MULT=2.0;SINK_REL_FLOOR=1e-3;SINK_REL_CAP=2e-2;LEN_TOL_ABS=4;LEN_TOL_REL=0.15;PROBE_TVD_MAX=0.05;CTX_MIN=0.75;LIFT_MIN=0.5
PROBE_RULE="same greedy tokens on full-context/manual-installed/manual-natural/generate-installed/generate-natural; installed-vs-natural and manual-vs-generate TVD <=0.05; both cache lengths correct; full-vs-split TVD diagnostic only"
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
 if not gpu_alive():raise GpuBlocked("GPU operation blocked after an earlier fatal error; restart runtime")
def report(ex,c=None,arm=None,planned=False,extra=None):
 fatal=(not planned) and is_fatal(ex);tb="(planned technical stop)" if planned else traceback.format_exc()
 cf=globals().get("cfg");attn=getattr(cf,"_attn_implementation",None) if cf is not None else None
 L=["#"*110,f"TEST{TEST} ERROR REPORT — BEGIN",f"run_id: {RUN_ID}",f"utc: {now()}",f"stage: {CTX['stage']} | sub-stage: {CTX['sub']}"]
 if c:L+=[f"case: {c.get('id')} | family={c.get('family')} group={c.get('group')} member={c.get('member')} pos={c.get('pos')} load={c.get('load')} k={c.get('k')}",f"arm: {arm}",f"tokens: source={c.get('src_tokens')} question={c.get('q_tokens')} max_new={MAX_NEW} budget={MAX_TOTAL}",f"question: {c.get('question')}","source:\n"+str(c.get("source"))]
 elif arm:L.append(f"arm: {arm}")
 L+=[f"engine: D={D} DMAX={DMAX} dtype=bfloat16 attn={attn} model={MODEL_ID}",f"environment: {json.dumps(CTX['env'],ensure_ascii=False)}",f"last GPU-op context: {json.dumps(jsafe(CTX['last']),ensure_ascii=False)}",f"progress: rows={len(ROWS)} complete={sum(bool(r.get('complete')) for r in ROWS)}/{len(CASES)} generations={dict(CTX['gens'])}",f"fatal: {fatal} | earlier fatal: {bool(CTX['fatal'])} | restart required: {fatal or bool(CTX['fatal'])}"]
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
def first_line(t):return next((s.strip() for s in str(t).splitlines() if s.strip()),"")
def clean_line(s):return re.sub(r"^\W*(final answer|answer)\s*[:\-]\s*","",s.strip(),flags=re.I).strip(" *`\"'")
def negpos(w):return [i for i,x in enumerate(w) if x in NEGW or (x=="longer" and i>0 and w[i-1]=="no")]
def mention(n,ph,used=None):
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
 return "conflict" if aff and negd else ("affirmed" if aff else ("negated" if negd else "none"))
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
 fixtures=[("northern archive",L,"correct_exact"),("The northern archive.\nYou are an AI assistant.",L,"correct_exact"),("**Northern Archive**",L,"correct_exact"),("The compass is not in the northern archive.",L,"negated_target"),("southern vault",L,"wrong_alternative"),("northern archive or southern vault",L,"mixed"),("Not the southern vault, but the northern archive.",L,"correct_with_rejection"),("AMBIGUOUS",L,"wrong_special"),("UNKNOWN",U,"correct_exact"),("Not UNKNOWN",U,"negated_target"),("UNKNOWN - northern archive",U,"mixed"),("container Z0",U,"wrong_alternative"),("The location cannot be determined.",U,"abstain_nonstandard"),("AMBIGUOUS",A,"correct_exact"),("Not AMBIGUOUS",A,"negated_target"),("",L,"empty"),("To determine this we must analyze each record carefully.",L,"no_answer"),("It is not where the report says.",L,"unparsed_negation"),("The compass isn't in the southern vault; it is in the northern archive.",L,"correct_with_rejection"),("Northern archive is not the correct location.",L,"negated_target"),("It is not UNKNOWN; it is northern archive.",L,"correct_verbose"),("Northern archive. No, not the northern archive.",L,"conflicting_mentions"),("It is not certain, but northern archive.",L,"unattributed_negation")]
 for txt,c,expected in fixtures:
  s=score(txt,c);sem="correct" if expected.startswith("correct_") else ("review" if expected in ("abstain_nonstandard","unparsed_negation","conflicting_mentions","unattributed_negation") else "incorrect")
  if (s["category"],s["semantic"])!=(expected,sem):raise TechStop(f"scorer fixture failed: {txt!r}; expected={expected}/{sem}; got={s}")
 return len(fixtures)
def assign_foreign():
 for c in CASES:
  c["foreign_id"]=None;c["foreign_reason"]=None;c["foreign_len_diff"]=None;tol=max(LEN_TOL_ABS,LEN_TOL_REL*c["src_tokens"]);c["foreign_tol"]=tol
  if c["target"] in SPECIAL:c["foreign_reason"]="uninformative special target";continue
  best=None
  for x in CASES:
   if x["tmpl"]!=c["tmpl"] or x["ent"]==c["ent"] or not x["valid"]:continue
   if any(has(x["source"],p) for p in [c["target"]]+c["alts"] if p not in SPECIAL):continue
   d=abs(x["src_tokens"]-c["src_tokens"])
   if d<=tol and (best is None or d<best[0]):best=(d,x)
  if best:c["foreign_id"]=best[1]["id"];c["foreign_len_diff"]=best[0]
  else:c["foreign_reason"]=f"no non-overlapping same-template source within ±{tol:.1f} tokens"
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
   inval={a:[f"{m}: {st(r,a).get('status')} {st(r,a).get('invalid_reason') or ''}" for m,r in zip(mids,rs) if r is not None and not valid(r,a)] for a in CORE}
   uv=not missing and not any(inval.values());G.append(dict(group=g,expected=mids,missing=missing,invalid={a:v for a,v in inval.items() if v},unit_valid=uv))
   if uv:units.append({a:all(ps(r,a) for r in rs) for a in ARMS if all(valid(r,a) for r in rs)})
  nexp=len(groups);nu=len(units);cnt={a:sum(bool(u.get(a)) for u in units) for a in ARMS};nva={a:sum(a in u for u in units) for a in ARMS};rate={a:cnt[a]/nva[a] if nva[a] else None for a in ARMS}
  def paired(a,b):return dict(a_pass_b_fail=sum(a in u and b in u and u[a] and not u[b] for u in units),b_pass_a_fail=sum(a in u and b in u and u[b] and not u[a] for u in units))
  P=dict(ctx_vs_nat=paired("CONTEXT","NATIVE_KV"),nat_vs_pkv=paired("NATIVE_KV","PKV_D120"),ctx_vs_pkv=paired("CONTEXT","PKV_D120"),pkv_vs_own=paired("PKV_D120","PKV_OWNSINK"))
  vr=[rows.get(m) for mids in groups.values() for m in mids];ordc={a:dict(pairs=0,changed=0) for a in ARMS}
  for g,mids in groups.items():
   mem={rows[m]["member"]:rows[m] for m in mids if m in rows}
   for base in ({m[:-4] for m in mem if m.endswith(("_fwd","_rev"))}|({""} if ("fwd" in mem or "rev" in mem) else set())):
    fw=mem.get(base+"_fwd" if base else "fwd");rv=mem.get(base+"_rev" if base else "rev")
    if fw is None or rv is None:continue
    for a in ARMS:
     if valid(fw,a) and valid(rv,a):ordc[a]["pairs"]+=1;ordc[a]["changed"]+=int(ps(fw,a)!=ps(rv,a))
  sens=dict(pkv_pass_incl_sink_out=sum(bool(r and st(r,"PKV_D120").get("status")=="OK" and st(r,"PKV_D120").get("pass")==1) for r in vr),pkv_ok_incl_sink_out=sum(bool(r and st(r,"PKV_D120").get("status")=="OK") for r in vr),sink_out_cases=sum(bool(r and r.get("memory") and not r["memory"]["sink_ok"]) for r in vr),review=sum(st(r,a).get("review")==1 for r in vr if r for a in ARMS))
  lift=rate["PKV_D120"]-rate["VANILLA"] if nu else None
  if nu<nexp:lab=f"INCOMPLETE / TECHNICALLY UNINTERPRETABLE ({nu}/{nexp} valid units)"
  elif rate["CONTEXT"]<CTX_MIN:lab="TASK NOT SOLVED WITH VISIBLE TEXT (uninterpretable)"
  elif P["ctx_vs_nat"]["a_pass_b_fail"]>0:lab="CACHE-PATH GAP (compression effect not assessable)"
  elif lift is None or lift<LIFT_MIN:lab="NOT MEMORY-DEPENDENT (uninformative)"
  elif P["nat_vs_pkv"]["a_pass_b_fail"]==0:lab="SUPPORTED (pilot; no paired loss vs NATIVE)"
  elif P["nat_vs_pkv"]["a_pass_b_fail"]==1:lab="PARTIALLY SUPPORTED (1 paired loss vs NATIVE)"
  else:lab="NOT SUPPORTED (≥2 paired losses vs NATIVE)"
  if integ.get("status")!="PASS" and lab.startswith(("SUPPORTED","PARTIALLY")):lab=f"UNCONFIRMED — integrity {integ.get('status')} ({lab})"
  FAM[vk]=dict(expected_groups=nexp,valid_units=nu,unit_pass=cnt,unit_valid_by_arm=nva,unit_rate=rate,lift_vs_vanilla=lift,paired=P,order=ordc,sensitivity=sens,groups=G,label=lab)
 plan=dict(cases=len(CASES),valid_budget=sum(bool(c.get("valid")) for c in CASES),attempted=len(ROWS),complete=sum(bool(r.get("complete")) for r in ROWS),generations_planned=dict(VANILLA_unique_questions=len({c["question"] for c in CASES if c.get("valid")}),per_memory_arm=sum(bool(c.get("valid")) for c in CASES),FOREIGN_LENMATCH=sum(bool(c.get("valid") and c.get("foreign_id")) for c in CASES)),generations=dict(CTX["gens"]),valid_by_arm={a:sum(valid(r,a) for r in ROWS) for a in ARMS},status_by_arm={a:dict(Counter(st(r,a).get("status") for r in ROWS)) for a in ARMS})
 return dict(families=FAM,plan=plan)
HYP=OrderedDict([("H0 single fact/counterfactual",["single"]),("H1 date selection",["latest","historical","conflict_k2","conflict_k4","conflict_k8"]),("H2 source priority",["authority"]),("H3 correction",["correction"]),("H4 ambiguity",["ambiguity"]),("H5 composition",["two_hop","three_hop","missing"]),("H6 load/position",["load_n008","load_n032","load_n128","load_pos_early","load_pos_late"])])
def print_summary(S,integ):
 say("="*110);say(f"TEST{TEST} RESULT SUMMARY | {RUN_ID}");say("Technical PASS ≠ hypothesis PASS; 6 entity families; a group passes only if all members pass.")
 say("integrity:",integ.get("status"),integ.get("detail",""));say("plan/attempt/valid:",json.dumps(S["plan"],ensure_ascii=False))
 if not ROWS:say("MAIN SUITE NOT STARTED: 0/210; no hypothesis result. Copy the ERROR REPORT and cache-probe lines above.");return
 for vk,f in S["families"].items():
  say(f"  {vk:<16} units {f['valid_units']}/{f['expected_groups']} | "+" ".join(f"{ABBR[a]} {f['unit_pass'][a]}/{f['unit_valid_by_arm'][a]}" for a in ARMS)+f" | lift={f['lift_vs_vanilla']} | {f['label']}")
  say("      paired:",f["paired"],"order:",f["order"],"sensitivity:",f["sensitivity"])
  for g in f["groups"]:
   if not g["unit_valid"]:say(f"      ✗ {g['group']}: missing={g['missing']} invalid={g['invalid']}")
 for h,vks in HYP.items():say(h+": "+"; ".join(f"{vk} → {S['families'][vk]['label']}" for vk in vks if vk in S["families"]))
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
 CTX["last"].update(op="forge",T=len(ids));o=model(input_ids=torch.tensor([ids],device=DEVICE),output_hidden_states=True,use_cache=False,return_dict=True,**(ltk(1) if use_ltk else {}));K=[];V=[]
 for L in range(NL):
  h=o.hidden_states[L][0];z=layers[L].input_layernorm(h);a=layers[L].self_attn;K.append(a.k_proj(z).contiguous());V.append(a.v_proj(z).contiguous())
 del o;return K,V
def forge(s):
 if "QUESTION:" in s:raise RuntimeError("Source-only forge: reserved QUESTION marker")
 ids=[PAD]+enc_ids(s+SEP);K,V=kv_from_ids(ids);return dict(ids=ids,T=len(ids),K=K,V=V)
@infer
def install(K,V):
 T=K[0].shape[0];cos,sin=model.model.rotary_emb(K[0][None],torch.arange(T,device=DEVICE)[None]);KK=[];VV=[]
 for L in range(NL):
  k=K[L].reshape(1,T,NKV,HD).transpose(1,2);v=V[L].reshape(1,T,NKV,HD).transpose(1,2);KK.append(apply_rotary_pos_emb(k,k,cos,sin,unsqueeze_dim=1)[1].contiguous());VV.append(v.contiguous())
 return tuple(KK),tuple(VV),T
def mkcache(kv):
 gpu_guard()
 try:c=DynamicCache(config=cfg)
 except TypeError:c=DynamicCache()
 for L in range(NL):c.update(kv[0][L].clone(),kv[1][L].clone(),L)
 if c.get_seq_length()!=kv[2]:raise RuntimeError(f"cache length {c.get_seq_length()} != {kv[2]} after install")
 return c
def kvget(c,L):return (c.layers[L].keys,c.layers[L].values) if hasattr(c,"layers") else (c.key_cache[L],c.value_cache[L])
def kv_equal(a,b):gpu_guard();return all(torch.equal(x,y) for p,q in zip(a,b) for x,y in zip(p,q))
def all_finite(ts):gpu_guard();return bool(torch.stack([torch.isfinite(t).all() for t in ts]).all().item())
def encode(f,d=D):gpu_guard();return {n:[torch.einsum("thi,hid->thd",f[n][L][1:].float().view(-1,NKV,HD)-CB[L][n][0],CB[L][n][1][:,:,:d]).to(torch.bfloat16) for L in range(NL)] for n in ("K","V")}
def decode(code,sink=None,d=D):
 gpu_guard();sink=sink or SINK;out={n:[torch.cat([sink[n][L],(CB[L][n][0]+torch.einsum("thd,hid->thi",code[n][L].float(),CB[L][n][1][:,:,:d])).reshape(-1,KVD).to(torch.bfloat16)]) for L in range(NL)] for n in ("K","V")};return out["K"],out["V"]
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
 mc=min(rows,key=lambda r:r["cos"]);mr=max(rows,key=lambda r:r["rel_row"]);return dict(min_cos=mc["cos"],min_cos_at=mc,max_rel_row=mr["rel_row"],max_rel_at=mr,finite_pos0=all(r["finite"] for r in rows),start_token_ok=bool(f["ids"][0]==PAD))
def fmt_sink(m):a=m["min_cos_at"];b=m["max_rel_at"];return f"min_cos={m['min_cos']:.6f}@{a['tensor']}L{a['layer']} max|Δ|/row={m['max_rel_row']:.2e}@{b['tensor']}L{b['layer']} rel_sink={b['rel_sink']:.2e}"
@infer
def build_rep(c):
 CTX["last"]=dict(op="build_rep",case=c["id"],src_tokens=c["src_tokens"]);f=forge(c["source"]);m=sink_metrics(f);code=encode(f);seal=code_sha(code);K,V=decode(code);own={n:[f[n][L][:1].clone() for L in range(NL)] for n in ("K","V")};Ko,Vo=decode(code,own)
 fin=dict(raw=all_finite(f["K"]+f["V"]),code=all_finite(code["K"]+code["V"]),decoded=all_finite(K+V+Ko+Vo));nkv=install(f["K"],f["V"]);pkv=install(K,V);okv=install(Ko,Vo);fin["installed"]=all_finite(list(nkv[0])+list(nkv[1])+list(pkv[0])+list(pkv[1])+list(okv[0])+list(okv[1]))
 fid={n:float(np.mean([float(F.cosine_similarity(dec[L][1:].float().reshape(-1),f[n][L][1:].float().reshape(-1),0)) for L in range(NL)])) for n,dec in (("K",K),("V",V))}
 CTX["last"].update(T=f["T"],installed_k_shape=list(nkv[0][0].shape),code_k_shape=list(code["K"][0].shape));rep=dict(sha=c["sha"],T=f["T"],records=c["records"],sink=m,sink_ok=bool(m["max_rel_row"]<=SINK_GATE),finite=fin,finite_ok=all(fin.values()) and m["finite_pos0"],start_ok=bool(m["start_token_ok"] and f["T"]==c["src_tokens"]),seal=seal,code=code,fidelity=fid,nkv=nkv,pkv=pkv,okv=okv,is_long=f["T"]>SHORT_T)
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
 qids=enc_ids(FMT.format(q=c["question"]));T=kv[2] if kv is not None else 0;pre=[PAD]+enc_ids(c["source"]+SEP) if arm=="CONTEXT" else [PAD]*T;need=len(pre)+len(qids)+MAX_NEW
 if need>MAX_TOTAL:return dict(status="INVALID_BUDGET",valid=False,invalid_reason=f"prompt+generation {need} > {MAX_TOTAL}")
 if arm!="CONTEXT":
  if set(pre)-{PAD}:return dict(status="LEAK",valid=False,invalid_reason="non-PAD dummy prefix")
  txt=tok.decode(pre+qids)
  for r in guard:
   if r.strip() and r.strip() in txt:return dict(status="LEAK",valid=False,invalid_reason=f"record leaked: {r}")
 CTX["last"]=dict(op="generate",case=c["id"],arm=arm,T=T,prompt_tokens=len(pre)+len(qids),q_tokens=len(qids));ids=torch.tensor([pre+qids],device=DEVICE);cache=mkcache(kv) if kv is not None else None;CTX["gens"][ck+"_attempted"]+=1
 torch.cuda.synchronize();t0=time.perf_counter();y=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),past_key_values=cache,**GEN);torch.cuda.synchronize();new=y[0,ids.shape[1]:].tolist();exp=len(pre)+len(qids)+len(new)-1;cl=None if cache is None else int(cache.get_seq_length());CTX["gens"][ck+"_completed"]+=1
 out=dict(status="OK",text=tok.decode(new,skip_special_tokens=True).strip(),new_tokens=len(new),seconds=time.perf_counter()-t0,memory_slots=T,prompt_tokens=len(pre)+len(qids),truncated=bool(new and len(new)==MAX_NEW and new[-1] not in EOS),cache_len=cl,cache_len_expected=exp if cache is not None else None,cache_len_ok=(cl==exp) if cache is not None else None);out["valid"]=out["cache_len_ok"] is not False;out["invalid_reason"]=None if out["valid"] else f"cache length {cl} != {exp}";return out
VAN={}
def meta(c):return {k:c.get(k) for k in ("id","family","vkey","group","member","ent","load","pos","k","records","source","question","target","alts","src_tokens","q_tokens","foreign_id","foreign_reason","foreign_len_diff")}
def run_case(c,ck="main"):
 row=meta(c);row.update(memory=None,arms={},complete=False);rep=None
 try:rep=get_rep(c)
 except GpuBlocked as ex:
  for a in ("NATIVE_KV","PKV_D120","PKV_OWNSINK"):row["arms"][a]=dict(status="NOT_RUN",valid=False,invalid_reason=str(ex))
 except Exception as ex:
  report(ex,c,"build_rep")
  for a in ("NATIVE_KV","PKV_D120","PKV_OWNSINK"):row["arms"][a]=dict(status="RUNTIME_ERROR",valid=False,invalid_reason=f"memory build failed: {ex}")
 if rep:row["memory"]={k:rep[k] for k in ("T","sink","sink_ok","finite","finite_ok","start_ok","fidelity")};row["memory"]["code_sha"]=rep["seal"]
 for arm in ARMS:
  if arm in row["arms"]:continue
  if not gpu_alive():row["arms"][arm]=dict(status="NOT_RUN",valid=False,invalid_reason="fatal CUDA error");continue
  try:
   if arm=="VANILLA":
    if c["question"] in VAN:out=dict(VAN[c["question"]]);out["reused"]=True;CTX["gens"][ck+"_vanilla_reused"]+=1
    else:out=gen(c,arm,None,c["records"],ck);VAN[c["question"]]=dict(out)
   elif arm=="CONTEXT":out=gen(c,arm,None,(),ck)
   elif rep is None:out=dict(status="NOT_RUN",valid=False,invalid_reason="no representation")
   elif not(rep["finite_ok"] and rep["start_ok"]):out=dict(status="TECHNICAL_INVALID",valid=False,invalid_reason=f"finite={rep['finite']} start={rep['start_ok']}")
   elif arm=="NATIVE_KV":out=gen(c,arm,rep["nkv"],c["records"],ck)
   elif arm=="PKV_D120":
    out=gen(c,arm,rep["pkv"],c["records"],ck)
    if out.get("status")=="OK" and out["valid"] and not rep["sink_ok"]:out["valid"]=False;out["invalid_reason"]=f"SINK OUT: {rep['sink']['max_rel_row']:.2e} > {SINK_GATE:.2e}"
   elif arm=="PKV_OWNSINK":out=gen(c,arm,rep["okv"],c["records"],ck)
   else:
    if not c.get("foreign_id"):out=dict(status="NOT_APPLICABLE",valid=False,invalid_reason=c.get("foreign_reason"))
    else:
     fr=get_rep(BYID[c["foreign_id"]])
     if not(fr["finite_ok"] and fr["start_ok"] and fr["sink_ok"]):out=dict(status="TECHNICAL_INVALID",valid=False,invalid_reason="foreign finite/start/sink check failed")
     else:out=gen(c,arm,fr["pkv"],c["records"]+fr["records"],ck);out["foreign_id"]=c["foreign_id"];out["foreign_len_diff"]=c["foreign_len_diff"]
  except GpuBlocked as ex:out=dict(status="NOT_RUN",valid=False,invalid_reason=str(ex))
  except Exception as ex:report(ex,c,arm);out=dict(status="RUNTIME_ERROR",valid=False,invalid_reason=f"{type(ex).__name__}: {ex}")
  if out.get("status")=="OK":out.update(score(out["text"],c))
  row["arms"][arm]=out
 row["complete"]=all(row["arms"].get(a,{}).get("status") not in (None,"NOT_RUN") for a in ARMS);return row
def sym(r,a):
 x=r["arms"].get(a,{});s=x.get("status")
 return ("R" if x.get("review") else str(x.get("pass")))+("" if x.get("valid") else "*") if s=="OK" else {"NOT_APPLICABLE":"-","INVALID_BUDGET":"I","LEAK":"L","TECHNICAL_INVALID":"T","NOT_RUN":"N"}.get(s,"E")
def print_case(i,N,r):
 m=r["memory"];A=r["arms"];g=lambda a,k:A.get(a,{}).get(k);flags=[]
 if m and not m["sink_ok"]:flags.append("SINK-OUT")
 if any(g(a,"status") not in ("OK","NOT_APPLICABLE") for a in ARMS):flags.append("TECH")
 if any(g(a,"review") for a in ARMS):flags.append("REVIEW")
 for a,b in (("CONTEXT","NATIVE_KV"),("CONTEXT","PKV_D120"),("NATIVE_KV","PKV_D120")):
  if g(a,"pass")!=g(b,"pass"):flags.append(f"{ABBR[a]}≠{ABBR[b]}")
 if g("FOREIGN_LENMATCH","pass")==1:flags.append("FOREIGN-HIT")
 if any(g(a,"status")!="OK" or not g(a,"valid") or g(a,"pass")!=1 or g(a,"review") for a in ("CONTEXT","NATIVE_KV","PKV_D120")):flags.append("CORE-FAIL/REVIEW/INVALID")
 say(f"{i:03d}/{N} {r['id']:<24} T={(m['T'] if m else 0):5d} | "+" ".join(f"{ABBR[a]}:{sym(r,a)}" for a in ARMS)+(" | ⚠ "+",".join(flags) if flags else ""))
 if flags:
  say(f"  target={r['target']!r} alts={r['alts']}"+(" | "+fmt_sink(m["sink"]) if m and not m["sink_ok"] else ""))
  for a in ARMS:
   x=A.get(a,{});say(f"  [{a}] status={x.get('status')} valid={x.get('valid')} cat={x.get('category')} trunc={x.get('truncated')} cache={x.get('cache_len')}/{x.get('cache_len_expected')} reason={x.get('invalid_reason')}\n    raw={x.get('text')!r}")
def ckpt(obj,name="checkpoint.jsonl"):
 if not CTX["file_io_ok"]:return
 try:
  with (RUN/name).open("ab") as fh:fh.write(canon(obj)+b"\n")
 except Exception as ex:CTX["file_io_ok"]=False;report(ex,extra="file writes disabled; console remains available")
@infer
def last_logits(ids,n,cache=None,mask_len=None):
 o=model(input_ids=torch.tensor([ids],device=DEVICE),attention_mask=torch.ones(1,mask_len or len(ids),device=DEVICE,dtype=torch.long),past_key_values=cache,use_cache=cache is not None,**ltk(n));return o.logits[0,-n:].float()
# LOCAL FIX: natural source-only cache is the reference for the manually installed cache.
@infer
def cache_probe(c):
 CTX["last"]=dict(op="cache_probe",case=c["id"]);f=forge(c["source"]);installed=install(f["K"],f["V"]);T=f["T"];qids=enc_ids(FMT.format(q=c["question"]))
 CTX["last"].update(op="cache_probe/natural_source_cache",T=T)
 o=model(input_ids=torch.tensor([f["ids"]],device=DEVICE),use_cache=True,return_dict=True,**ltk(1));nc=o.past_key_values
 natural=(tuple(kvget(nc,L)[0].clone() for L in range(NL)),tuple(kvget(nc,L)[1].clone() for L in range(NL)),T);del o,nc
 prefix_rel={name:max(float((installed[j][L].float()-natural[j][L].float()).norm()/natural[j][L].float().norm().clamp_min(1e-12)) for L in range(NL)) for j,name in enumerate(("K","V"))}
 prefix_exact=kv_equal(installed[:2],natural[:2]);ids=torch.tensor([[PAD]*T+qids],device=DEVICE)
 def generated(kv,label):
  CTX["last"].update(op="cache_probe/generate",path=label);cache=mkcache(kv)
  g=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),past_key_values=cache,return_dict_in_generate=True,output_scores=True,**dict(GEN,max_new_tokens=PROBE_STEPS))
  tokens=g.sequences[0,ids.shape[1]:].tolist();scores=torch.stack([s[0].float() for s in g.scores]);length=int(cache.get_seq_length());return tokens,scores,length
 S,gs,cl=generated(installed,"installed");Sn,gn,cn=generated(natural,"natural");n=len(S)
 if not n:raise TechStop(f"empty cache-probe generation: {c['id']}")
 CTX["last"].update(op="cache_probe/teacher_forced_comparison")
 lc=last_logits(f["ids"]+qids+S[:-1],n);lm=last_logits(qids+S[:-1],n,mkcache(installed),T+len(qids)+n-1);ln=last_logits(qids+S[:-1],n,mkcache(natural),T+len(qids)+n-1)
 tv=lambda a,b:(0.5*(a.softmax(-1)-b.softmax(-1)).abs().sum(-1)).tolist()
 same_generation=S==Sn;tvds=dict(full_vs_installed=tv(lc,lm),full_vs_natural=tv(lc,ln),installed_vs_natural=tv(lm,ln),manual_vs_generate=tv(lm,gs),generate_installed_vs_natural=tv(gs,gn) if same_generation else None)
 steps=[]
 for j in range(n):
  steps.append(dict(step=j,token=S[j],piece=tok.decode([S[j]]),full=int(lc[j].argmax()),installed=int(lm[j].argmax()),natural=int(ln[j].argmax()),generate=int(gs[j].argmax()),tvd={k:v[j] for k,v in tvds.items() if v is not None}))
 max_tvd={k:max(v) if v else None for k,v in tvds.items()};exp=T+len(qids)+n-1;expn=T+len(qids)+len(Sn)-1
 checks=dict(native_prefix_audit=max(prefix_rel.values())<5e-2,same_generated_tokens=same_generation,all_greedy_equal=all(s["token"]==s["full"]==s["installed"]==s["natural"]==s["generate"] for s in steps),installed_vs_natural_tvd=max_tvd["installed_vs_natural"]<=PROBE_TVD_MAX,manual_vs_generate_tvd=max_tvd["manual_vs_generate"]<=PROBE_TVD_MAX,generate_vs_natural_tvd=same_generation and max_tvd["generate_installed_vs_natural"]<=PROBE_TVD_MAX,installed_cache_length=cl==exp,natural_cache_length=cn==expn,finite_logits=all_finite([lc,lm,ln,gs,gn]))
 r=dict(case=c["id"],T=T,q_tokens=len(qids),steps=n,generated=tok.decode(S),natural_generated=tok.decode(Sn),prefix_bit_exact=prefix_exact,prefix_rel=prefix_rel,max_tvd=max_tvd,full_context_tvd_warning=max_tvd["full_vs_installed"]>PROBE_TVD_MAX,cache_len=cl,cache_len_expected=exp,natural_cache_len=cn,natural_cache_len_expected=expn,checks=checks,**{"pass":all(checks.values())},step_detail=steps,rule=PROBE_RULE)
 del f,installed,natural;return r
@infer
def native_audit(ids):
 K,V=kv_from_ids(ids);kv=install(K,V);nat=model(input_ids=torch.tensor([ids],device=DEVICE),use_cache=True,**ltk(1)).past_key_values
 r={n:max(float((kv[i][L].float()-kvget(nat,L)[i].float()).norm()/kvget(nat,L)[i].float().norm().clamp_min(1e-12)) for L in range(NL)) for i,n in enumerate(("K","V"))};del K,V,kv,nat;return r
NEUTRAL={}
def neutral_ids(n):
 if n in NEUTRAL:return list(NEUTRAL[n])
 if n<=0:return [PAD]
 txt="";i=0
 while True:
  txt=(txt+" "+CORPUS[i%len(CORPUS)]).strip();i+=1;ids=enc_ids(txt+SEP)
  if len(ids)>=n:NEUTRAL[n]=[PAD]+ids;return [PAD]+ids
def integrity_check():
 if not CTX["model_loaded"]:return dict(status="NOT_EVALUATED",detail="model never fully loaded")
 if not gpu_alive():return dict(status="NOT_EVALUATED",detail="fatal CUDA error; GPU operations skipped")
 try:
  FP1,SENT1=sentinel();st=dict(sampled_weight_sentinel_unchanged=(FP1==FP0 and SENT1==SENT0),frozen_eval=(not model.training and all(not p.requires_grad for p in model.parameters())),attn_sdpa=getattr(cfg,"_attn_implementation",None)=="sdpa",seals=all(CTX["seal_checks"]) if CTX["seal_checks"] else None);ok=st["sampled_weight_sentinel_unchanged"] and st["frozen_eval"] and st["attn_sdpa"] and st["seals"] is not False
  return dict(status="PASS" if ok else "FAIL",detail=json.dumps(st),checks=st,fingerprint_before=FP0,fingerprint_after=FP1,sentinel_before=SENT0,sentinel_after=SENT1,note="sampled sentinel, not a full verification of all weights")
 except Exception as ex:report(ex,extra="integrity check");return dict(status="NOT_EVALUATED",detail=str(ex))
# ================================================== RUN ==================================================
RUN=None;SUMMARY=None;INTEG=dict(status="NOT_EVALUATED",detail="not reached")
try:
 say("="*110);say(f"TEST{TEST} — NIRVANA PKV D120 | run {RUN_ID}");say("="*110);say("438d pre-main revision:",PROBE_RULE)
 stage("cpu_setup");CASES=build_cases();BYID={c["id"]:c for c in CASES};assert len(BYID)==len(CASES)
 for c in CASES:EXPECTED.setdefault(c["vkey"],OrderedDict()).setdefault(c["group"],[]).append(c["id"])
 say(f"  scorer self-test: {scorer_selftest()}/23 PASS")
 try:ROOT=Path("/content/AKBASCORE_TEST438d") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST438d");RUN=ROOT/RUN_ID;RUN.mkdir(parents=True,exist_ok=True)
 except Exception as ex:CTX["file_io_ok"]=False;report(ex,extra="console-only mode")
 stage("imports")
 for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
  if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
 import torch,transformers;import torch.nn.functional as F
 from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
 from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
 os.environ["TOKENIZERS_PARALLELISM"]="false";CTX["env"].update(torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda)
 if not torch.cuda.is_available():raise TechStop("CUDA not available; select an A100 runtime")
 DEVICE=torch.device("cuda");CTX["env"]["gpu"]=torch.cuda.get_device_name(0);torch.set_grad_enabled(False);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);say("  env:",CTX["env"])
 stage("model_load");t0=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True);tv=tuple(int(re.sub(r"\D","",v) or 0) for v in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
 model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}).eval()
 for p in model.parameters():p.requires_grad_(False)
 cfg=model.config;layers=model.model.layers;H=cfg.hidden_size;NH=cfg.num_attention_heads;NKV=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None) or H//NH;KVD=NKV*HD
 if (len(layers),H,NH,NKV,HD)!=(28,3584,28,4,128):raise TechStop("architecture mismatch")
 if next(model.parameters()).dtype!=torch.bfloat16 or getattr(cfg,"use_sliding_window",False):raise TechStop("dtype/sliding-window mismatch")
 PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id;ge=model.generation_config.eos_token_id;EOS=sorted({tok.eos_token_id,*(ge if isinstance(ge,(list,tuple)) else [ge])}-{None});GEN=dict(max_new_tokens=MAX_NEW,do_sample=False,repetition_penalty=1.0,use_cache=True,pad_token_id=PAD,eos_token_id=EOS)
 FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
 def sentinel():
  gpu_guard()
  with torch.inference_mode():
   h=hashlib.sha256();s=[]
   for i,t in enumerate(FP_T):
    s.append(float(t.sum(dtype=torch.float32)));flat=t.detach().reshape(-1);n=flat.numel();c=min(256,n);offs=sorted({k*(n-c)//15 for k in range(16)});h.update(f"{i}|{tuple(t.shape)}|{t.dtype}|{n}|{offs}|".encode());h.update(torch.cat([flat[o:o+c] for o in offs]).float().cpu().numpy().tobytes())
   return s,h.hexdigest()
 FP0,SENT0=sentinel()
 if sentinel()!=(FP0,SENT0):raise TechStop("weight sentinel not repeatable")
 CTX["model_loaded"]=True
 def set_attn(x):
  gpu_guard()
  try:model.set_attn_implementation(x)
  except Exception:cfg._attn_implementation=x
  if getattr(cfg,"_attn_implementation",None)!=x:raise RuntimeError(f"attention switch failed: {x}")
 torch.cuda.synchronize();say(f"  loaded in {time.perf_counter()-t0:.1f}s | PAD={PAD} EOS={EOS}")
 stage("forge_equivalence");LTK_OK=False;_ids=[PAD]+enc_ids(" ".join(CORPUS[:6])+SEP);_a=kv_from_ids(_ids,False);_b=kv_from_ids(_ids,False);FORGE_DET=kv_equal(_a,_b);LTK_SUPPORTED="logits_to_keep" in inspect.signature(model.forward).parameters
 if LTK_SUPPORTED:LTK_OK=True;_c=kv_from_ids(_ids,True);LTK_OK=kv_equal(_a,_c);del _c
 del _a,_b;DIAG.update(forge_deterministic=FORGE_DET,ltk_supported=LTK_SUPPORTED,ltk_used=LTK_OK);say(f"  deterministic={FORGE_DET} logits_to_keep={LTK_OK}"+("" if LTK_OK else " FALLBACK"))
 stage("codebook");CO=[forge(s) for s in CORPUS];SINK={n:[CO[0][n][L][:1].clone() for L in range(NL)] for n in ("K","V")};CB=[]
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
 CORPUS_ROWS=sum(c["T"]-1 for c in CO);EV_D={n:float(torch.stack([CB[L][n][2][:,D-1] for L in range(NL)]).mean()) for n in ("K","V")};del CO,R,X;DIAG.update(codebook_rows=CORPUS_ROWS,explained_variance_at_D=EV_D);say("  rows:",CORPUS_ROWS,"EV:",EV_D)
 stage("case_tokens")
 for c in CASES:
  c["sha"]=sha(c["source"].encode());c["src_tokens"]=1+len(enc_ids(c["source"]+SEP));c["q_tokens"]=len(enc_ids(FMT.format(q=c["question"])));c["valid"]=c["src_tokens"]+c["q_tokens"]+MAX_NEW<=MAX_TOTAL
  if any(s in c["source"] or c["source"] in s for s in CORPUS):raise TechStop(f"codebook overlap: {c['id']}")
  if not(c["target"] in SPECIAL or has(c["source"],c["target"])):raise TechStop(f"target absent: {c['id']}")
 assign_foreign();LMAX=max(c["src_tokens"] for c in CASES);say(f"  cases={len(CASES)} groups={sum(len(g) for g in EXPECTED.values())} source_max={LMAX} foreign={sum(bool(c['foreign_id']) for c in CASES)}")
 free,total=torch.cuda.mem_get_info();need=(LONG_KEEP+1)*3*NL*2*LMAX*KVD*2+2*LMAX*H*2*(NL+1);DIAG["memory_estimate"]=dict(free_bytes=free,estimated_peak_extra=need);say(f"  free={free/2**30:.1f} GiB estimated_extra={need/2**30:.2f} GiB")
 if need>0.8*free:raise TechStop("estimated representation memory exceeds 80% of free GPU memory")
 stage("sink_calibration");CAL=[]
 for n in sorted({0,16,64,256,1024,2048,LMAX}):
  CTX["sub"]=f"neutral T≈{n}";ids=neutral_ids(n);K,V=kv_from_ids(ids);m=sink_metrics(dict(ids=ids,T=len(ids),K=K,V=V));m.update(target_len=n,T=len(ids));CAL.append(m);del K,V;say(f"  T={m['T']:5d} {fmt_sink(m)} finite={m['finite_pos0']}")
 CAL_REL=max(m["max_rel_row"] for m in CAL if m["T"]>1);SINK_GATE=min(max(SINK_ENV_MULT*CAL_REL,SINK_REL_FLOOR),SINK_REL_CAP)
 if not all(m["finite_pos0"] for m in CAL):raise TechStop("non-finite sink calibration")
 CTX["sub"]="eager comparison"
 try:set_attn("eager");ids=neutral_ids(LMAX);K,V=kv_from_ids(ids);EAGER_LONG=sink_metrics(dict(ids=ids,T=len(ids),K=K,V=V));del K,V
 except BaseException as ex:raise Reported(report(ex,extra="eager comparison; reported before restore"))
 finally:
  if gpu_alive():
   try:set_attn("sdpa")
   except BaseException as ex2:report(ex2,extra="secondary SDPA restore failure")
 _f=forge(CORPUS[0]);SINK_REPEAT_EXACT=all(torch.equal(_f[n][L][:1],SINK[n][L]) for n in ("K","V") for L in range(NL));del _f
 DIAG.update(sink_calibration=CAL,sink_envelope=CAL_REL,sink_gate=SINK_GATE,eager_long_sink=EAGER_LONG,sink_repeat_exact=SINK_REPEAT_EXACT);say(f"  gate={SINK_GATE:.2e}; cap=0.02 engineering tolerance, not semantic guarantee; repeat_exact={SINK_REPEAT_EXACT}");say("  eager:",fmt_sink(EAGER_LONG))
 stage("native_audit");NATIVE_AUDIT={}
 for n in sorted({16,256,2048,LMAX}):CTX["sub"]=f"T≈{n}";NATIVE_AUDIT[str(n)]=native_audit(neutral_ids(n));say("  T≈",n,NATIVE_AUDIT[str(n)])
 DIAG["native_audit_by_length"]=NATIVE_AUDIT
 if max(max(d.values()) for d in NATIVE_AUDIT.values())>=5e-2:raise TechStop(f"native audit failed: {NATIVE_AUDIT}")
 # Policy registered before examining these probe results or starting the main suite.
 stage("probe_policy");DIAG["probe_policy"]=dict(rule=PROBE_RULE,tvd_max=PROBE_TVD_MAX,revision="438c full/split TVD hard stop replaced by natural-cache controlled audit; no distribution equivalence claim");say(PROBE_RULE)
 stage("cache_probe");vc=[c for c in CASES if c["valid"]];PROBES=[]
 for c in (min(vc,key=lambda x:x["src_tokens"]),max(vc,key=lambda x:x["src_tokens"])):
  CTX["sub"]=c["id"];r=cache_probe(c);PROBES.append(r);DIAG["cache_probes"]=PROBES;say("  "+json.dumps(jsafe({k:v for k,v in r.items() if k!="step_detail"}),ensure_ascii=False))
  for s in r["step_detail"]:say("    "+json.dumps(jsafe(s),ensure_ascii=False))
  if r["full_context_tvd_warning"]:say("  DIAGNOSTIC WARNING: full-context/split-cache TVD >0.05; see natural-cache reference. Distribution equality is not asserted.")
  if not r["pass"]:raise Reported(report(TechStop(f"cache probe failed: {r['checks']}"),c,"CACHE_PROBE",planned=True))
 stage("lock");LOCK=dict(test=TEST,model=MODEL_ID,architecture=[NL,H,NH,NKV,HD],D=D,DMAX=DMAX,seed=SEED,PAD=PAD,EOS=EOS,FMT=FMT,SEP=SEP,STYLE=STYLE,max_new=MAX_NEW,max_total=MAX_TOTAL,corpus=CORPUS,cases=[dict(meta(c),valid=c["valid"]) for c in CASES],sink=dict(policy="shared SINK canonical; OWNSINK diagnostic",gate=SINK_GATE,envelope=CAL_REL,mult=SINK_ENV_MULT,floor=SINK_REL_FLOOR,cap=SINK_REL_CAP),foreign=dict(tolerance=[LEN_TOL_ABS,LEN_TOL_REL],special_targets="NA"),scoring="first nonempty line; 6-word filler-scoped pre-negation + post-negation; conflict/unattributed negation REVIEW, never pass",verdict=dict(unit="all group members",ctx_min=CTX_MIN,lift_min=LIFT_MIN,retention="0 paired losses supported, 1 partial, >=2 not supported",cache_path="any CTX->NATIVE loss blocks verdict"),probe=dict(steps=PROBE_STEPS,tvd_max=PROBE_TVD_MAX,rule=PROBE_RULE),forge_ltk=LTK_OK)
 LOCK_SHA=sha(canon(LOCK));DIAG["lock_sha"]=LOCK_SHA;say("  LOCK",LOCK_SHA)
 if CTX["file_io_ok"] and RUN:
  try:(RUN/"experiment_lock.json").write_bytes(canon(LOCK))
  except Exception as ex:CTX["file_io_ok"]=False;report(ex,extra="lock write failed")
 stage("smoke_test","technical only");SMOKE=[]
 for sid in ("single0:orig","miss1:broken_fwd","load2-128:fwd"):
  c=BYID[sid];CTX["sub"]=sid;r=run_case(c,ck="smoke");SMOKE.append(r);print_case(len(SMOKE),3,r)
  if not gpu_alive():break
 probs=[]
 for r in SMOKE:
  m=r["memory"]
  if not m or not(m["finite_ok"] and m["start_ok"]):probs.append(f"{r['id']}: memory finite/start failed")
  for a,x in r["arms"].items():
   if x.get("status") not in ("OK","NOT_APPLICABLE"):probs.append(f"{r['id']}/{a}: {x.get('status')} {x.get('invalid_reason')}")
   if x.get("status")=="OK" and (x.get("cache_len_ok") is False or "category" not in x):probs.append(f"{r['id']}/{a}: length/scoring failed")
 DIAG["smoke"]=dict(problems=probs,rows=SMOKE);VAN.clear()
 if probs or not gpu_alive():raise TechStop("smoke failed:\n"+"\n".join(probs or ["fatal error"]))
 say("  smoke technical PASS; semantic outcomes ignored for this gate")
 stage("main_run");ORDER=[c for c in CASES if c["src_tokens"]<=SHORT_T]+sorted([c for c in CASES if c["src_tokens"]>SHORT_T],key=lambda c:(c["ent"],c["load"],c["pos"],c["member"]));N=len(ORDER);t_run=time.perf_counter()
 say("  legend: 1 pass, 0 fail, R review, * invalid, - NA, I budget, L leak, T technical, E error, N not run")
 for i,c in enumerate(ORDER,1):
  if not gpu_alive():break
  CTX["sub"]=c["id"]
  if not c["valid"]:row=meta(c);row.update(memory=None,complete=True,arms={a:dict(status="INVALID_BUDGET",valid=False,invalid_reason="source+question+generation exceeds budget") for a in ARMS})
  else:row=run_case(c)
  ROWS.append(row);ckpt(row);print_case(i,N,row)
 DIAG["main_seconds"]=time.perf_counter()-t_run
 if gpu_alive():
  for rep in REPS.values():CTX["seal_checks"].append(code_sha(rep["code"])==rep["seal"])
except BaseException as ex:
 if isinstance(ex,Reported):CTX["stop"]=ex.rec
 elif isinstance(ex,(TechStop,KeyboardInterrupt)):CTX["stop"]=report(ex,planned=True)
 else:CTX["stop"]=report(ex)
finally:
 try:stage("finalize","integrity");INTEG=integrity_check();say("  integrity:",INTEG.get("status"),INTEG.get("detail",""))
 except BaseException as ex:report(ex,extra="finalize/integrity")
 try:stage("finalize","summary");SUMMARY=summarize(INTEG);print_summary(SUMMARY,INTEG)
 except BaseException as ex:report(ex,extra="finalize/summary")
 try:
  if CTX["file_io_ok"] and RUN:
   P=dict(test=TEST,run_id=RUN_ID,utc=now(),env=CTX["env"],diagnostics=DIAG,integrity=INTEG,summary=SUMMARY,errors=CTX["errors"],stop=CTX["stop"],fatal=CTX["fatal"],rows=ROWS,limits=["joint-source records, not independent bank fusion","D120 retains 120/128 content coordinates; runtime cache remains full size","source absent from answer prompt; retained in RAM/logs","six entity families; pilot labels are not population-level proof","full/split logit distributions may differ; natural-cache control reported"]);pb=canon(P);(RUN/f"{RUN_ID}_payload.json").write_bytes(pb);(RUN/"manifest.json").write_bytes(canon(dict(payload=f"{RUN_ID}_payload.json",sha256=sha(pb),bytes=len(pb),utc=now())));say("  optional files:",RUN,"SHA256:",sha(pb))
 except BaseException as ex:report(ex,extra="finalize/files; diagnosis available in console")
 st="FATAL CUDA ERROR — RESTART RUNTIME" if CTX["fatal"] else ("STOPPED: "+CTX["stop"]["type"]+" — "+CTX["stop"]["msg"].splitlines()[0][:200] if CTX["stop"] else "COMPLETED")
 say("="*110);say(f"TEST{TEST} FINAL STATE: {st} | rows {len(ROWS)}/{len(CASES)} | errors {len(CTX['errors'])} | integrity {INTEG.get('status')}");say("="*110)
# Continue only after a complete, error-free reproduction with verified integrity.
_F_READY=(isinstance(globals().get("P"),dict) and CTX["fatal"] is None and CTX["stop"] is None and not CTX["errors"] and len(ROWS)==len(CASES)==210 and all(r.get("complete") for r in ROWS) and INTEG.get("status")=="PASS")
TEST438F_RESULT=None
if _F_READY:
 _F_SOURCE_RUN=P["run_id"];_F_SOURCE_SHA=sha(canon(P))
 _F_FILE=(RUN/(_F_SOURCE_RUN+"_payload.json")) if RUN else None
 _F_PAYLOAD_PATH=str(_F_FILE) if CTX["file_io_ok"] and _F_FILE and _F_FILE.is_file() else ""
 print("TEST438f | PHASE 2: fresh raw answers recovered; original scores retained; revised scoring is POST-HOC",flush=True)
 # TEST438e — AKBASCORE NIRVANA D120 — Varan 1: CPU-only paired RESCORING.
 # Original raw answers/technical validity/sink gate/verdict thresholds stay unchanged.
 # New scores are POST-HOC EXPLORATORY; original registered scores remain the reference.
 import re,json,time,hashlib,traceback,copy
 from pathlib import Path
 from datetime import datetime,timezone
 from collections import Counter,OrderedDict
 PAYLOAD_PATH=_F_PAYLOAD_PATH
 PREFERRED_RUN=_F_SOURCE_RUN
 LOGGED_SHA=_F_SOURCE_SHA
 def run_rescore(cached_payload=None,cached_lock=None):
  TEST="438e";RUN_ID="TEST438e-"+datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
  stage_name="init";current_case=None;current_arm=None;data=None;input_path=None;finished=False
  ARMS=("VANILLA","CONTEXT","NATIVE_KV","PKV_D120","PKV_OWNSINK","FOREIGN_LENMATCH");CORE=ARMS[:4]
  ABBR=dict(zip(ARMS,("VAN","CTX","NAT","PKV","OWN","FOR")));SPECIAL=("UNKNOWN","AMBIGUOUS");CTX_MIN=.75;LIFT_MIN=.5
  CASES=[];ROWS=[];EXPECTED=OrderedDict();CTX={"gens":{}}
  NEGW={"not","never","isn","wasn","aren","weren","neither","nor","without","cannot"}
  FILL={"the","a","an","in","at","on","to","inside","located","stored","kept","is","was","are","it","t","s","currently","be","been"}
  ABST=re.compile(r"\b(cannot be determined|can not be determined|not (be )?(determined|inferred|specified|stated|mentioned|known|provided)|no information|insufficient|unclear)\b")
  def say(*a):print(*a,flush=True)
  def stage(name):
   nonlocal stage_name
   stage_name=name;say(f"[TEST438e] ▶ STAGE {name}")
  def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")
  def sha(x):return hashlib.sha256(x).hexdigest()
  class TechStop(RuntimeError):pass
  def norm(x):return re.sub(r"[^\w]+"," ",str(x).casefold()).strip()
  def has(text,phrase):return bool(phrase) and f" {norm(phrase)} " in f" {norm(text)} "
  def first_line(t):return next((s.strip() for s in str(t).splitlines() if s.strip()),"")
  def clean_line(s):return re.sub(r"^\W*(final answer|answer)\s*[:\-]\s*","",s.strip(),flags=re.I).strip(" *`\"'")
  def negpos(w):return [i for i,x in enumerate(w) if x in NEGW or (x=="longer" and i>0 and w[i-1]=="no")]
  def mention(n,ph,used=None):
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
   return "conflict" if aff and negd else ("affirmed" if aff else ("negated" if negd else "none"))
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
   fixtures=[("northern archive",L,"correct_exact"),("The northern archive.\nYou are an AI assistant.",L,"correct_exact"),("**Northern Archive**",L,"correct_exact"),("The compass is not in the northern archive.",L,"negated_target"),("southern vault",L,"wrong_alternative"),("northern archive or southern vault",L,"mixed"),("Not the southern vault, but the northern archive.",L,"correct_with_rejection"),("AMBIGUOUS",L,"wrong_special"),("UNKNOWN",U,"correct_exact"),("Not UNKNOWN",U,"negated_target"),("UNKNOWN - northern archive",U,"mixed"),("container Z0",U,"wrong_alternative"),("The location cannot be determined.",U,"abstain_nonstandard"),("AMBIGUOUS",A,"correct_exact"),("Not AMBIGUOUS",A,"negated_target"),("",L,"empty"),("To determine this we must analyze each record carefully.",L,"no_answer"),("It is not where the report says.",L,"unparsed_negation"),("The compass isn't in the southern vault; it is in the northern archive.",L,"correct_with_rejection"),("Northern archive is not the correct location.",L,"negated_target"),("It is not UNKNOWN; it is northern archive.",L,"correct_verbose"),("Northern archive. No, not the northern archive.",L,"conflicting_mentions"),("It is not certain, but northern archive.",L,"unattributed_negation")]
   for txt,c,expected in fixtures:
    s=score(txt,c);sem="correct" if expected.startswith("correct_") else ("review" if expected in ("abstain_nonstandard","unparsed_negation","conflicting_mentions","unattributed_negation") else "incorrect")
    if (s["category"],s["semantic"])!=(expected,sem):raise TechStop(f"scorer fixture failed: {txt!r}; expected={expected}/{sem}; got={s}")
   return len(fixtures)
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
     inval={a:[f"{m}: {st(r,a).get('status')} {st(r,a).get('invalid_reason') or ''}" for m,r in zip(mids,rs) if r is not None and not valid(r,a)] for a in CORE}
     uv=not missing and not any(inval.values());G.append(dict(group=g,expected=mids,missing=missing,invalid={a:v for a,v in inval.items() if v},unit_valid=uv))
     if uv:units.append({a:all(ps(r,a) for r in rs) for a in ARMS if all(valid(r,a) for r in rs)})
    nexp=len(groups);nu=len(units);cnt={a:sum(bool(u.get(a)) for u in units) for a in ARMS};nva={a:sum(a in u for u in units) for a in ARMS};rate={a:cnt[a]/nva[a] if nva[a] else None for a in ARMS}
    def paired(a,b):return dict(a_pass_b_fail=sum(a in u and b in u and u[a] and not u[b] for u in units),b_pass_a_fail=sum(a in u and b in u and u[b] and not u[a] for u in units))
    P=dict(ctx_vs_nat=paired("CONTEXT","NATIVE_KV"),nat_vs_pkv=paired("NATIVE_KV","PKV_D120"),ctx_vs_pkv=paired("CONTEXT","PKV_D120"),pkv_vs_own=paired("PKV_D120","PKV_OWNSINK"))
    vr=[rows.get(m) for mids in groups.values() for m in mids];ordc={a:dict(pairs=0,changed=0) for a in ARMS}
    for g,mids in groups.items():
     mem={rows[m]["member"]:rows[m] for m in mids if m in rows}
     for base in ({m[:-4] for m in mem if m.endswith(("_fwd","_rev"))}|({""} if ("fwd" in mem or "rev" in mem) else set())):
      fw=mem.get(base+"_fwd" if base else "fwd");rv=mem.get(base+"_rev" if base else "rev")
      if fw is None or rv is None:continue
      for a in ARMS:
       if valid(fw,a) and valid(rv,a):ordc[a]["pairs"]+=1;ordc[a]["changed"]+=int(ps(fw,a)!=ps(rv,a))
    sens=dict(pkv_pass_incl_sink_out=sum(bool(r and st(r,"PKV_D120").get("status")=="OK" and st(r,"PKV_D120").get("pass")==1) for r in vr),pkv_ok_incl_sink_out=sum(bool(r and st(r,"PKV_D120").get("status")=="OK") for r in vr),sink_out_cases=sum(bool(r and r.get("memory") and not r["memory"]["sink_ok"]) for r in vr),review=sum(st(r,a).get("review")==1 for r in vr if r for a in ARMS))
    lift=rate["PKV_D120"]-rate["VANILLA"] if nu else None
    if nu<nexp:lab=f"INCOMPLETE / TECHNICALLY UNINTERPRETABLE ({nu}/{nexp} valid units)"
    elif rate["CONTEXT"]<CTX_MIN:lab="TASK NOT SOLVED WITH VISIBLE TEXT (uninterpretable)"
    elif P["ctx_vs_nat"]["a_pass_b_fail"]>0:lab="CACHE-PATH GAP (compression effect not assessable)"
    elif lift is None or lift<LIFT_MIN:lab="NOT MEMORY-DEPENDENT (uninformative)"
    elif P["nat_vs_pkv"]["a_pass_b_fail"]==0:lab="SUPPORTED (pilot; no paired loss vs NATIVE)"
    elif P["nat_vs_pkv"]["a_pass_b_fail"]==1:lab="PARTIALLY SUPPORTED (1 paired loss vs NATIVE)"
    else:lab="NOT SUPPORTED (≥2 paired losses vs NATIVE)"
    if integ.get("status")!="PASS" and lab.startswith(("SUPPORTED","PARTIALLY")):lab=f"UNCONFIRMED — integrity {integ.get('status')} ({lab})"
    FAM[vk]=dict(expected_groups=nexp,valid_units=nu,unit_pass=cnt,unit_valid_by_arm=nva,unit_rate=rate,lift_vs_vanilla=lift,paired=P,order=ordc,sensitivity=sens,groups=G,label=lab)
   plan=dict(cases=len(CASES),valid_budget=sum(bool(c.get("valid")) for c in CASES),attempted=len(ROWS),complete=sum(bool(r.get("complete")) for r in ROWS),generations_planned=dict(VANILLA_unique_questions=len({c["question"] for c in CASES if c.get("valid")}),per_memory_arm=sum(bool(c.get("valid")) for c in CASES),FOREIGN_LENMATCH=sum(bool(c.get("valid") and c.get("foreign_id")) for c in CASES)),generations=dict(CTX["gens"]),valid_by_arm={a:sum(valid(r,a) for r in ROWS) for a in ARMS},status_by_arm={a:dict(Counter(st(r,a).get("status") for r in ROWS)) for a in ARMS})
   return dict(families=FAM,plan=plan)
  POLICY={"revision":"438e answer-span scorer v1","status":"POST-HOC EXPLORATORY, not a replacement for registered results","answer":"first nonempty line; first sentence; cut explicit explanation markers independently of target","explanation_markers":"because; since; as it/the/this; On YYYY-MM-DD; according to; to determine","tail":"explicit contradictory answer/location claims on the same line => REVIEW; intermediate IDs in an explanation are not rival terminal places","special":"UNKNOWN/AMBIGUOUS must be affirmed; uncertainty cues in the answer span => REVIEW; explanatory negation alone does not invalidate a definite answer","unchanged":"raw text, technical validity, SINK exclusions, expected groups, CTX=.75, lift=.5, paired-loss verdicts","limits":"heuristic English parser; no claim of complete semantic understanding"}
  def segments(line):return [s.strip() for s in re.split(r"(?<!\d)[.!?](?=\s|$)",line) if s.strip()]
  def answer_parts(text):
   line=clean_line(first_line(text));parts=segments(line)
   if not parts:return "","",line
   primary=parts[0];tail=". ".join(parts[1:])
   marker=re.search(r"\b(?:because|since|as (?:it|the|this)|on \d{4}-\d{2}-\d{2}|according to|to determine)\b",primary,re.I)
   if marker and marker.start()>0:tail=primary[marker.start():]+(". "+tail if tail else "");primary=primary[:marker.start()].strip()
   return primary,tail,line
  def identifier(x):return bool(re.fullmatch(r"[ZVY]\d+",str(x),re.I))
  def tail_conflicts(tail,c):
   risks=[];target=c["target"];others=[x for x in c["alts"] if not identifier(x)]
   for part in segments(tail):
    n=norm(part)
    if mention(n,target) in ("negated","conflict"):risks.append("target denied in same-line continuation")
    answer_cue=bool(re.search(r"\b(?:final answer|the answer|answer is|actually|instead)\b",part,re.I))
    location_cue=bool(re.search(r"\b(?:is|was|are|were)\s+(?:currently\s+)?(?:(?:located|stored|kept)\s+)?(?:in|at|inside)\s+",part,re.I))
    for other in others:
     state=mention(n,other)
     bare=n in (norm(other),norm("the "+other),norm("in the "+other))
     if state=="affirmed" and (bare or answer_cue or location_cue):risks.append("different explicit answer: "+other)
     if state=="conflict" and (bare or answer_cue or location_cue):risks.append("conflicting explicit answer: "+other)
   return sorted(set(risks))
  def score_new(text,c):
   span,tail,line=answer_parts(text);out=score(span,c);risks=tail_conflicts(tail,c)
   if risks:out.update(category="contradictory_continuation",semantic="review",review=1,**{"pass":0})
   elif out["pass"] and re.search(r"\b(?:maybe|perhaps|possibly|probably|likely|might|may be|could be|not certain|not sure)\b",span,re.I):out.update(category="uncertain_answer",semantic="review",review=1,**{"pass":0})
   out.update(answer_span=span,explanation_tail=tail,first_line=line,tail_flags=risks,target_anywhere=int(has(text,c["target"])),alt_anywhere=int(any(has(text,x) for x in c["alts"])))
   return out
  def new_selftest():
   L=dict(target="northern archive",alts=["southern vault","UNKNOWN"])
   S=dict(target="southern vault",alts=["northern archive","Z0","V0"])
   E=dict(target="eastern platform",alts=["western gate","AMBIGUOUS"])
   U=dict(target="UNKNOWN",alts=["northern archive","southern vault","Z0"])
   fixtures=[("northern archive",L,1,"correct"),("Not the southern vault, but the northern archive.",L,1,"correct"),("northern archive or southern vault",L,0,"incorrect"),("Not northern archive",L,0,"incorrect"),("Northern archive is not the correct location.",L,0,"incorrect"),("Northern archive. No, not the northern archive.",L,0,"review"),("Northern archive. Actually, the answer is southern vault.",L,0,"review"),("Northern archive. The compass is in the southern vault.",L,0,"review"),("Southern vault. The compass is inside container Z0 which is inside vehicle V0.",S,1,"correct"),("Southern vault because it is inside container Z0 which is inside vehicle V0.",S,1,"correct"),("The compass is in the southern vault. It is not in the northern archive. The northern archive is corrected to the southern vault.",S,1,"correct"),("The blue signal is currently in the eastern platform. The reports are identical and do not give different locations, so there is no ambiguity.",E,1,"correct"),("It is not certain, but northern archive.",L,0,"review"),("It might be northern archive or southern vault.",L,0,"incorrect"),("UNKNOWN",U,1,"correct"),("Not UNKNOWN",U,0,"incorrect"),("UNKNOWN - southern vault",U,0,"incorrect"),("container Z0",U,0,"incorrect"),("The location cannot be determined.",U,0,"review"),("Southern vault\nThe compass is elsewhere.",S,1,"correct"),("southern vault",U,0,"incorrect"),("northern archive On 2041-05-03, the compass was stored there.",L,1,"correct"),("Southern vault and container Z0",S,0,"incorrect"),("Perhaps northern archive.",L,0,"review")]
   for text,c,want,sem in fixtures:
    got=score_new(text,c)
    if (got["pass"],got["semantic"])!=(want,sem):raise TechStop(f"NEW SCORER FIXTURE FAILED: {text!r}; expected={want}/{sem}; got={got}")
   return len(fixtures)
  def replay(rows,new=False):
   result=copy.deepcopy(rows);changes=[];mismatches=[]
   for row in result:
    for arm,x in row.get("arms",{}).items():
     if x.get("status")!="OK":continue
     if not isinstance(x.get("text"),str):raise TechStop(f"raw answer missing: {row['id']}/{arm}")
     old=score(x["text"],row)
     for k in ("pass","category","semantic","review"):
      if x.get(k)!=old[k]:mismatches.append((row["id"],arm,k,x.get(k),old[k]))
     if new:
      ns=score_new(x["text"],row);before={k:x.get(k) for k in ("pass","category","semantic","review")};x.update(ns)
      if any(before[k]!=ns[k] for k in before):changes.append(dict(case=row["id"],arm=arm,target=row["target"],old=before,new={k:ns[k] for k in before},answer_span=ns["answer_span"],tail_flags=ns["tail_flags"],raw=x["text"],technical_valid=x.get("valid")))
   if mismatches:raise TechStop(f"original scorer replay mismatch ({len(mismatches)}): {mismatches[:20]}; not safe to compare")
   return result,changes
  def raw_pairs(rows,a,b):
   usable=[r for r in rows if all(r["arms"].get(z,{}).get("status")=="OK" and r["arms"][z].get("valid") for z in (a,b))]
   return dict(n=len(usable),a_pass_b_fail=[r["id"] for r in usable if r["arms"][a]["pass"]==1 and r["arms"][b]["pass"]==0],b_pass_a_fail=[r["id"] for r in usable if r["arms"][b]["pass"]==1 and r["arms"][a]["pass"]==0])
  def show_summary(old,new):
   say("="*110);say("ORIGINAL REGISTERED → NEW EXPLORATORY (group scores; all members must pass)")
   for vk,f in old["families"].items():
    g=new["families"][vk]
    say(f"{vk:<16} valid units {f['valid_units']}/{f['expected_groups']} | "+" ".join(f"{ABBR[a]} {f['unit_pass'][a]}/{f['unit_valid_by_arm'][a]}→{g['unit_pass'][a]}/{g['unit_valid_by_arm'][a]}" for a in CORE))
    say("  REGISTERED :",f["label"]);say("  EXPLORATORY:",g["label"])
   say("CASE-LEVEL paired results (diagnostic; do not replace group verdicts):")
   for a,b in (("CONTEXT","NATIVE_KV"),("NATIVE_KV","PKV_D120"),("PKV_D120","PKV_OWNSINK")):
    say(f"  {a} vs {b}: "+json.dumps(raw_pairs(ROWS,a,b),ensure_ascii=False))
   say("SINK-invalid canonical cases remain excluded; OWN never reinstates them.")
   say("No new hypothesis confirmation: the revised scorer is post-hoc and needs an unseen validation set.")
  try:
   say("="*110);say(f"TEST438e — CPU RESCORE / VARAN 1 | {RUN_ID}");say("="*110)
   stage("scorer_selftests");say(f"  old fixtures={scorer_selftest()} PASS | new fixtures={new_selftest()} PASS")
   stage("load_existing_run")
   if PAYLOAD_PATH:
    input_path=Path(PAYLOAD_PATH)
    if not input_path.is_file():raise TechStop("PAYLOAD_PATH does not exist: "+str(input_path))
   else:
    candidates=[]
    for root in (Path("/content/AKBASCORE_TEST438d"),Path("/tmp/AKBASCORE_TEST438d")):
     if root.exists():candidates.extend(root.glob("TEST438d-*/*_payload.json"))
    preferred=[p for p in candidates if p.parent.name==PREFERRED_RUN]
    if preferred:input_path=preferred[0]
    elif candidates:input_path=max(candidates,key=lambda p:p.stat().st_mtime)
   if input_path:
    raw=input_path.read_bytes();data=json.loads(raw);digest=sha(raw);mf=input_path.parent/"manifest.json";lf=input_path.parent/"experiment_lock.json"
    if not mf.is_file() or not lf.is_file():raise TechStop("manifest.json or experiment_lock.json missing; originals cannot be verified")
    manifest=json.loads(mf.read_bytes())
    if manifest.get("payload")!=input_path.name or manifest.get("sha256")!=digest or manifest.get("bytes")!=len(raw):raise TechStop("original payload manifest mismatch")
    lock_bytes=lf.read_bytes();lock=json.loads(lock_bytes)
    if sha(lock_bytes)!=data.get("diagnostics",{}).get("lock_sha"):raise TechStop("original experiment lock hash mismatch")
    say("  SOURCE:",input_path)
   elif isinstance(cached_payload,dict) and isinstance(cached_lock,dict):
    data=copy.deepcopy(cached_payload);lock=copy.deepcopy(cached_lock);digest=sha(canon(data))
    if sha(canon(lock))!=data.get("diagnostics",{}).get("lock_sha"):raise TechStop("in-memory experiment lock mismatch")
    if data.get("run_id")!=PREFERRED_RUN or digest!=LOGGED_SHA:raise TechStop("in-memory payload does not match the completed run and console SHA; use the verified reproduction payload")
    say("  SOURCE: verified in-memory TEST438d payload; matches this reproduction SHA")
   else:raise TechStop("TEST438d raw answers not found. Run this cell in the same Colab runtime where TEST438d completed. This cell will not reload the model or rerun the GPU experiment.")
   if data.get("test")!="438d" or lock.get("test")!="438d":raise TechStop("input must be TEST438d")
   if data.get("run_id")==PREFERRED_RUN and digest!=LOGGED_SHA:raise TechStop("source SHA differs from the supplied TEST438d console record")
   say("  run:",data["run_id"],"SHA256:",digest)
   CASES=lock["cases"];original=data["rows"]
   if len({c["id"] for c in CASES})!=len(CASES) or len({r["id"] for r in original})!=len(original):raise TechStop("duplicate case ID")
   if {c["id"] for c in CASES}!={r["id"] for r in original}:raise TechStop("completed run does not cover every locked case")
   for c in CASES:EXPECTED.setdefault(c["vkey"],OrderedDict()).setdefault(c["group"],[]).append(c["id"])
   locked={c["id"]:c for c in CASES}
   for row in original:
    current_case=row["id"]
    for k in ("target","alts","question","source","vkey","group","member"):
     if row.get(k)!=locked[row["id"]].get(k):raise TechStop(f"locked metadata mismatch: {row['id']}/{k}")
   integrity=data.get("integrity",{});CTX["gens"]=data.get("summary",{}).get("plan",{}).get("generations",{})
   stage("policy_lock");policy_sha=sha(canon(POLICY));say("  policy:",json.dumps(POLICY,ensure_ascii=False));say("  POLICY SHA:",policy_sha)
   stage("original_score_replay");ROWS,_=replay(original);old_summary=summarize(integrity)
   stored=data.get("summary",{}).get("families",{})
   for vk,f in old_summary["families"].items():
    for k in ("unit_pass","unit_valid_by_arm","valid_units","expected_groups","label"):
     if stored.get(vk,{}).get(k)!=f[k]:raise TechStop(f"registered summary replay mismatch: {vk}/{k}")
   say("  original per-answer scores and registered group summary reproduced exactly")
   stage("new_exploratory_scores");ROWS,changes=replay(original,new=True);new_summary=summarize(integrity)
   for before,after in zip(original,ROWS):
    for a in ARMS:
     for k in ("text","status","valid","invalid_reason","cache_len","cache_len_ok"):
      if before["arms"].get(a,{}).get(k)!=after["arms"].get(a,{}).get(k):raise TechStop(f"preservation check failed: {before['id']}/{a}/{k}")
   say("  raw answers and all technical-validity fields unchanged; changes:",len(changes))
   stage("changed_answers")
   for d in changes:
    current_case=d["case"];current_arm=d["arm"]
    say(f"[{d['case']}/{d['arm']}] target={d['target']!r} technical_valid={d['technical_valid']}")
    say("  OLD:",d["old"],"NEW:",d["new"]);say("  SPAN:",repr(d["answer_span"]),"TAIL FLAGS:",d["tail_flags"]);say("  RAW:",repr(d["raw"]))
   stage("paired_summary");show_summary(old_summary,new_summary)
   sink_invalid=[r["id"] for r in ROWS if r.get("memory") and not r["memory"]["sink_ok"]]
   say("  SINK-invalid retained:",len(sink_invalid),sink_invalid)
   result=dict(test=TEST,run_id=RUN_ID,source_run=data["run_id"],source_sha256=digest,policy=POLICY,policy_sha256=policy_sha,interpretation="POST-HOC EXPLORATORY; original registered results retained",original_summary=old_summary,exploratory_summary=new_summary,changes=changes,rows=ROWS,source_integrity=integrity,sink_invalid=sink_invalid,paired_case_diagnostics={a+"_vs_"+b:raw_pairs(ROWS,a,b) for a,b in (("CONTEXT","NATIVE_KV"),("NATIVE_KV","PKV_D120"),("PKV_D120","PKV_OWNSINK"))})
   stage("optional_save")
   try:
    out_root=(input_path.parent if input_path else Path("/content" if Path("/content").exists() else "/tmp"))/RUN_ID;out_root.mkdir(parents=True,exist_ok=False);pb=canon(result);out_file=out_root/(RUN_ID+"_rescore.json");out_file.write_bytes(pb)
    (out_root/"manifest.json").write_bytes(canon(dict(payload=out_file.name,sha256=sha(pb),bytes=len(pb),source_sha256=digest)))
    say("  optional result:",out_file,"SHA256:",sha(pb))
   except Exception as ex:say("  OPTIONAL SAVE FAILED:",type(ex).__name__,str(ex));say(traceback.format_exc())
   finished=True;return result
  except Exception as ex:
   say("#"*110);say("TEST438e ERROR REPORT — BEGIN");say("run:",RUN_ID,"stage:",stage_name,"case:",current_case,"arm:",current_arm)
   say("source:",str(input_path) if input_path else "in-memory/auto-search");say("error:",type(ex).__name__,str(ex));say("traceback:\n"+traceback.format_exc());say("TEST438e ERROR REPORT — END");say("#"*110)
   return None
  finally:say(f"TEST438e FINAL STATE: {'COMPLETED — POST-HOC RESCORE ONLY' if finished else 'STOPPED — NO NEW RESULT'} | GPU calls=0 | new generations=0 | source files unchanged")
 TEST438F_RESULT=run_rescore(P,LOCK)
else:
 print("TEST438f FINAL STATE: STOPPED — reproduction did not complete cleanly; no rescore result claimed. See TEST438d report above.",flush=True)
if _F_READY:
 print("TEST438f FINAL STATE: "+("COMPLETED — fresh GPU reproduction + exploratory CPU rescore; not an independent confirmation" if TEST438F_RESULT is not None else "STOPPED — CPU rescore failed; see report above"),flush=True)
