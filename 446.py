# TEST446: TEST445 reference preserved; independent source-only modules; fixed composition panel.
import os,sys,re,json,time,random,hashlib,importlib.util,subprocess,traceback,inspect,platform
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter,OrderedDict
import numpy as np
#<<CPU_BEGIN>>
TEST="446";SEED=384;D=120;DMAX=128;NL=28;MAX_NEW=32;MAX_TOTAL=6144;SHORT_T=256;LONG_KEEP=3;PROBE_STEPS=6
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
 fatal=is_fatal(ex);r=dict(stage=CTX["stage"],case=c.get("id") if c else None,arm=arm,type=type(ex).__name__,msg=str(ex),fatal=fatal,traceback=traceback.format_exc(),extra=extra);CTX["errors"].append(r)
 if fatal:CTX["fatal"]=r
 say("ERROR",json.dumps(r,ensure_ascii=False));return r
CORPUS=["Jonas Weber carried the wooden crate across the quiet market square.","Priya Nair wrote a long letter to her cousin in the morning.","The small boat drifted slowly toward the rocky shore.","Omar Haddad fixed the broken clock in the village school.","A tired teacher closed the green door of the library.","Sofia Rossi baked fresh bread for the harvest festival.","The children watched the kites rising above the hill.","Liam O'Connor sold his old bicycle to a neighbor.","Heavy rain flooded the narrow street near the market.","Nadia Petrova translated the ancient manuscript into French.","The farmer counted the sheep before sunset.","Hiro Sato opened a tiny bakery beside the river.","An old dog slept under the kitchen table all afternoon.","Carlos Mendes washed the windows of his grandmother's house.","The museum guard locked the heavy gate at midnight.","Fatima Zahra grew tomatoes in the backyard garden.","The pilot announced a short delay because of fog.","Anna Kowalski found a lost wallet on the bus.","Snow covered the mountain village during the night.","Ravi Kumar taught his brother how to play chess.","The chef sharpened every knife before dinner service.","Lucas Martin cleaned the roof of the barn after the storm.","A young violinist practiced scales in the empty hall.","Mei Lin delivered the package to the wrong address.","Marek Novak placed the glass bottle beneath the wooden bench.","Sara Ibrahim carried a red notebook into the quiet classroom.","Noah Schmidt repaired the small radio beside the kitchen window.","Yuki Mori left a paper envelope near the station entrance.","Amira Hassan moved the ceramic bowl onto the upper shelf.","Peter Novak opened the metal box behind the old theater.","Lucia Costa placed the yellow scarf inside the travel bag.","Daniel Kim carried a black umbrella through the central courtyard."]
ENT=[("Elena Varga","silver compass","northern archive","southern vault"),("Kerem Yıldız","brass lantern","old lighthouse","fishing harbor"),("Aiko Tanabe","blue signal","eastern platform","western gate"),("Marta Solberg","copper sextant","river warehouse","hill observatory"),("Diego Ferraz","wooden rudder","dry dock","harbor office"),("Lena Hoffmann","ivory chess set","city library","stone chapel")]
ADJ=["amber","birch","cedar","dune","ember","frost"];NOUN=["kiosk","depot","pavilion","quarry","garage","bunker"];E=len(ENT);SPECIAL=("UNKNOWN","AMBIGUOUS")
def build_cases():
 C=[]
 def add(family,vkey,group,member,records,q,target,alts,e,k=0):
  C.append(dict(id=f"{group}:{member}",family=family,vkey=vkey,group=group,member=member,records=list(records),source="\n".join(records),question=q+STYLE,target=target,alts=sorted(set(alts)-{target}),ent=e,load=0,pos="",k=k,tmpl=f"{vkey}|{member}"))
 def pair(family,vkey,group,records,q,target,alts,e,k=0):
  for member,rr in (("fwd",records),("rev",records[::-1])):add(family,vkey,group,member,rr,q,target,alts,e,k)
 for e in (0,1,2,3):
  P,O,a,b=ENT[e];X,Y=(a,b) if e%2==0 else (b,a);Z,Vh,Yc=f"Z{e}",f"V{e}",f"Y{e}";alts=[a,b,Z,Vh,Yc]
  q=f"Name the place (not a container) where the {O} is located. If the records do not connect the {O} to any place, answer UNKNOWN."
  pair("connection","complete",f"complete{e}",[f"The {O} is inside container {Z}.",f"Container {Z} is in the {Y}."],q,Y,alts+["UNKNOWN"],e)
  pair("connection","broken",f"broken{e}",[f"The {O} is inside container {Z}.",f"An unrelated container {Yc} is in the {Y}."],q,"UNKNOWN",alts,e)
  rq=f"Where did {P} store the {O}?";anchor=f"{P} stored the {O} in the {X}.";unrelated=f"An unrelated container {Yc} is in the {Y}."
  add("retention","retention",f"retain{e}","alone",[anchor],rq,X,[a,b,Yc],e)
  pair("retention","retention",f"retain{e}",[anchor,unrelated],rq,X,[a,b,Yc],e)
 for e in (1,2):
  P,O,a,b=ENT[e];X,Y=(a,b) if e%2==0 else (b,a);k=8;locs=[X,Y]+[f"{ADJ[e]} {nn}" for nn in NOUN];assign=[locs[(i+e)%k] for i in range(k)]
  recs=[f"On 2041-05-{3+3*i:02d}, {P} {'stored' if i==0 else 'moved'} the {O} {'in' if i==0 else 'to'} the {assign[i]}." for i in range(k)];random.Random(SEED*1000+e*10+k).shuffle(recs)
  pair("sink_boundary","sink_boundary",f"boundary{e}",recs,f"According to the most recent dated record, where is the {O}?",assign[-1],locs,e,k)
 assert len(C)==32 and len({c["id"] for c in C})==32
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
def score_registered(text,c):
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
  s=score_registered(txt,c);sem="correct" if expected.startswith("correct_") else ("review" if expected in ("abstain_nonstandard","unparsed_negation","conflicting_mentions","unattributed_negation") else "incorrect")
  if (s["category"],s["semantic"])!=(expected,sem):raise TechStop(f"scorer fixture failed: {txt!r}; expected={expected}/{sem}; got={s}")
 return len(fixtures)
SCORING_POLICY=dict(revision="442 answer-span v2",answer="first nonempty line; first sentence; explanation markers independent of target",tail="same-line explicit conflicting answers REVIEW; unrelated subjects and chain IDs are not rival answers",uncertainty="uncertain answer REVIEW",technical="SINK/finite/start/cache gates unchanged",scope="targeted regressions; no independent generalization claim")
def segments(line):return [s.strip() for s in re.split(r"(?<!\d)[.!?](?=\s|$)",line) if s.strip()]
def answer_parts(text):
 line=clean_line(first_line(text));parts=segments(line)
 if not parts:return "","",line
 primary=parts[0];tail=". ".join(parts[1:])
 marker=re.search(r"\b(?:because|since|as (?:it|the|this)|on \d{4}-\d{2}-\d{2}|according to|to determine)\b",primary,re.I)
 if marker and marker.start()>0:tail=primary[marker.start():]+(". "+tail if tail else "");primary=primary[:marker.start()].strip()
 return primary,tail,line
def identifier(x):return bool(re.fullmatch(r"[ZVY]\d+",str(x),re.I))
def tail_conflicts_441(tail,c):
 # Explicit claims only. Mentions in historical/correction/chain explanations are not answers.
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
def tail_conflicts(tail,c):
 risks=[];target=c["target"];others=[x for x in c["alts"] if not identifier(x)]
 e=c.get("ent");obj=ENT[e][1] if isinstance(e,int) and 0<=e<len(ENT) else None
 for part in segments(tail):
  n=norm(part);answer_cue=bool(re.search(r"\b(?:(?:final |the )?answer\s*(?:is|was|:)|actually|instead)\b",part,re.I))
  subject=has(part,obj) if obj else bool(re.search(r"\b(?:compass|lantern|signal|sextant|rudder|chess set|item|object)\b",part,re.I))
  unrelated=bool(re.search(r"\b(?:unrelated|another|different)\s+(?:container|vehicle|item|object)\b",part,re.I))
  if unrelated and not subject and not answer_cue:continue
  if mention(n,target) in ("negated","conflict"):risks.append("target denied in same-line continuation")
  location_cue=bool(re.search(r"\b(?:is|was|are|were)\s+(?:currently\s+)?(?:(?:located|stored|kept)\s+)?(?:in|at|inside)\s+",part,re.I))
  pronoun=bool(re.match(r"\s*(?:it|this item|the item|this object|the object)\b",part,re.I))
  for other in others:
   state=mention(n,other);bare=n in (norm(other),norm("the "+other),norm("in the "+other))
   if state in ("affirmed","conflict") and (bare or answer_cue or (location_cue and (subject or pronoun))):risks.append("different explicit answer: "+other)
 return sorted(set(risks))
def score_new(text,c):
 span,tail,line=answer_parts(text);out=score_registered(span,c);risks=tail_conflicts(tail,c)
 # Do not select a correct answer from the continuation when the initial answer fails.
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
def score_441(text,c):
 span,tail,line=answer_parts(text);out=score_registered(span,c);risks=tail_conflicts_441(tail,c)
 if risks:out.update(category="contradictory_continuation",semantic="review",review=1,**{"pass":0})
 elif out["pass"] and re.search(r"\b(?:maybe|perhaps|possibly|probably|likely|might|may be|could be|not certain|not sure)\b",span,re.I):out.update(category="uncertain_answer",semantic="review",review=1,**{"pass":0})
 return out
def score(text,c):
 out=score_new(text,c);out["registered_score"]=score_registered(text,c);out["score_441"]=score_441(text,c);return out
def subject_selftest():
 c=dict(target="northern archive",alts=["southern vault","Y0"],ent=0)
 tests=[("in the northern archive. An unrelated container Y0 is in the southern vault.",1,"correct"),("The compass is in the northern archive. The unrelated container Y0 in the southern vault does not affect the answer.",1,"correct"),("Northern archive. The silver compass is in the southern vault.",0,"review"),("Northern archive. It is in the southern vault.",0,"review"),("Northern archive. An unrelated container Y0 is there, but the silver compass is in the southern vault.",0,"review"),("Northern archive. Actually, the answer is southern vault.",0,"review")]
 for txt,want,sem in tests:
  got=score(txt,c)
  if (got["pass"],got["semantic"])!=(want,sem):raise TechStop(f"subject fixture failed: {txt!r} {got}")
 return len(tests)
REPS=OrderedDict();VAN={};NEUTRAL={}
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
@infer
def last_logits(ids,n,cache=None,mask_len=None):
 o=model(input_ids=torch.tensor([ids],device=DEVICE),attention_mask=torch.ones(1,mask_len or len(ids),device=DEVICE,dtype=torch.long),past_key_values=cache,use_cache=cache is not None,**ltk(n));return o.logits[0,-n:].float()
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
def logit_stats(logits,reference):
 if not all_finite([logits,reference]):raise TechStop("non-finite diagnostic logits")
 p=logits[0].softmax(-1);q=reference[0].softmax(-1);top=torch.topk(p,2);rt=int(reference[0].argmax());t=int(logits[0].argmax())
 return dict(first_token=t,first_piece=tok.decode([t]),raw_first_token=rt,raw_argmax_retained=t==rt,raw_token_probability=float(p[rt]),top2_probability_margin=float(top.values[0]-top.values[1]),first_tvd_vs_raw=float(0.5*(p-q).abs().sum()))
POLICY=dict(primary='K120_V128_BF16_OWN',secondary='D120_V9V14_RAW_OWN',candidates=['K120_V128_BF16_OWN'],raw_v_layers=[9,14],critical=['broken1:rev','broken2:rev'],regression_cases=32,new_cases=12,selection='fixed before run; no automatic best variant',sink='canonical shared SINK unchanged; both candidates retain exact source-own slot0',success='both critical answers correct; zero valid native->candidate losses on regression and new panel; finite/cache/start/slot0/seals/integrity valid',scope='new entity/template replication; three new entities; no broad generalization claim',storage='actual retained tensor bytes; shared codebook reported separately; full runtime cache unchanged')
NEW_ENT=[('Leila Demir','bronze astrolabe','maple station','granite tower'),('Niko Petrov','glass medallion','orchard lodge','cobalt museum'),('Aya Mori','paper atlas','cedar studio','marble annex')]
ENT.extend(NEW_ENT)
def build_new_cases():
 C=[]
 for j,(P,O,a,b) in enumerate(NEW_ENT):
  e=6+j;Z,Yc=f'Z{e}',f'Y{e}';Y=b if e%2==0 else a;q=f'Name the place (not a container) where the {O} is located. If the records do not connect the {O} to any place, answer UNKNOWN.'+STYLE
  for kind,second,target in [('complete',f'Container {Z} is in the {Y}.',Y),('broken',f'An unrelated container {Yc} is in the {Y}.','UNKNOWN')]:
   recs=[f'The {O} is inside container {Z}.',second]
   for order,rr in [('fwd',recs),('rev',recs[::-1])]:C.append(dict(id=f'new{e}-{kind}:{order}',family='new_panel',vkey=kind,group=f'new{e}',member=kind+'_'+order,records=list(rr),source='\n'.join(rr),question=q,target=target,alts=sorted(set([a,b,Z,Yc,'UNKNOWN'])-{target}),ent=e,load=0,pos='',k=0,tmpl=kind+'|'+order))
 assert len(C)==12 and len({c['id'] for c in C})==12
 return C
@infer
def candidate_packet(f,label):
 if label not in POLICY['candidates']:raise TechStop('unregistered candidate')
 own={n:[f[n][L][:1].clone() for L in range(NL)] for n in ('K','V')};code={};out={};basis_bytes=0
 for n in ('K','V'):
  code[n]=[];out[n]=[]
  for L in range(NL):
   raw=(label==POLICY['secondary'] and n=='V' and L in POLICY['raw_v_layers']);d=128 if label==POLICY['primary'] and n=='V' else 120
   if raw:coeff=f[n][L][1:].clone();content=coeff.clone()
   else:
    mu,B,_=CB[L][n];coeff=torch.einsum('thi,hid->thd',f[n][L][1:].float().view(-1,NKV,HD)-mu,B[:,:,:d]).to(torch.bfloat16);content=(mu+torch.einsum('thd,hid->thi',coeff.float(),B[:,:,:d])).reshape(-1,KVD).to(torch.bfloat16);basis_bytes+=mu.numel()*mu.element_size()+B[:,:,:d].numel()*B.element_size()
   code[n].append(coeff);out[n].append(torch.cat([own[n][L],content]))
 packet={n:code[n]+own[n] for n in ('K','V')};seal=code_sha(packet);kv=install(out['K'],out['V']);finite=all_finite(out['K']+out['V']+[x for xs in packet.values() for x in xs]+list(kv[0])+list(kv[1]));slot0=all(torch.equal(out[n][L][:1],f[n][L][:1]) for n in ('K','V') for L in range(NL));start=f['ids'][0]==PAD and kv[2]==f['T'];content_bytes=sum(x.numel()*x.element_size() for xs in code.values() for x in xs);sink_bytes=sum(x.numel()*x.element_size() for xs in own.values() for x in xs);raw_bytes=sum(x.numel()*x.element_size() for n in ('K','V') for x in f[n]);fid={n:max(float((out[n][L][1:].float()-f[n][L][1:].float()).norm()/f[n][L][1:].float().norm().clamp_min(1e-12)) for L in range(NL)) for n in ('K','V')}
 tech=dict(finite=finite,start_ok=bool(start),source_own_slot0_exact=slot0,source_own_slot0_delta=0.0 if slot0 else None,content_bytes=content_bytes,source_own_sink_bytes=sink_bytes,packet_tensor_bytes=content_bytes+sink_bytes,raw_tensor_bytes=raw_bytes,packet_ratio=(content_bytes+sink_bytes)/raw_bytes,candidate_basis_mean_bytes_required=basis_bytes,fidelity=fid,runtime_cache_full_size=True,metadata_and_serialization_excluded=True)
 if not(finite and start and slot0):raise TechStop('candidate finite/start/source-own-slot0 check failed')
 return kv,packet,seal,tech
@infer
def evaluate(c,label,kv,reference,tech,panel):
 lg=last_logits(enc_ids(FMT.format(q=c['question'])),1,mkcache(kv),kv[2]+c['q_tokens']);out=gen(c,label,kv,c['records'],'candidate' if panel=='regression' else 'new_panel')
 if out.get('status')=='OK':out.update(score(out['text'],c))
 out['valid']=bool(out.get('valid') and tech['finite'] and tech['start_ok'] and tech['source_own_slot0_exact']);r=dict(case=c['id'],source_sha=c['sha'],panel=panel,variant=label,technical=tech,result=out,**logit_stats(lg,reference));DIAG.setdefault('answers',[]).append(r);checkpoint(r,'candidate_answers.jsonl');say('CANDIDATE',json.dumps(dict(case=c['id'],panel=panel,variant=label,passed=out.get('pass'),valid=out['valid'],tvd=r['first_tvd_vs_raw'],packet_ratio=tech['packet_ratio'],raw=out.get('text')),ensure_ascii=False));return r
@infer
def run_candidates(c,panel):
 f=forge(c['source']);raw=install(f['K'],f['V']);qids=enc_ids(FMT.format(q=c['question']));reference=last_logits(qids,1,mkcache(raw),f['T']+len(qids));refs=None
 if panel=='new_panel':
  refs=dict(case=c['id'],kind=c['vkey'],group=c['group'],member=c['member'],arms={})
  for arm in ('VANILLA','CONTEXT','NATIVE_KV'):
   if arm=='VANILLA' and c['question'] in VAN:out=dict(VAN[c['question']]);CTX['gens']['new_panel_vanilla_reused']+=1
   else:
    out=gen(c,arm,raw if arm=='NATIVE_KV' else None,() if arm=='CONTEXT' else c['records'],'new_panel')
    if arm=='VANILLA':VAN[c['question']]=dict(out)
   if out.get('status')=='OK':out.update(score(out['text'],c))
   refs['arms'][arm]=out
  if any(x.get('status')!='OK' or not x.get('valid') for x in refs['arms'].values()):raise TechStop('new-panel reference invalid')
  DIAG.setdefault('new_references',[]).append(refs);checkpoint(refs,'new_references.jsonl');say('NEW_REFERENCE',json.dumps(dict(case=c['id'],scores={a:x.get('pass') for a,x in refs['arms'].items()},raw={a:x.get('text') for a,x in refs['arms'].items()}),ensure_ascii=False))
 for label in POLICY['candidates']:
  kv,packet,seal,tech=candidate_packet(f,label);r=evaluate(c,label,kv,reference,tech,panel);sealed=code_sha(packet)==seal;CTX['seal_checks'].append(sealed);r['packet_seal_ok']=sealed
  if not sealed:raise TechStop('candidate packet seal changed')
  del kv,packet
 del f,raw,reference
def checkpoint(obj,name='checkpoint.jsonl'):
 if RUN is not None:
  with (RUN/name).open('ab') as fh:fh.write(canon(obj)+b'\n')
def summary445():
 valid=lambda x:x.get('status')=='OK' and bool(x.get('valid'));passed=lambda x:valid(x) and x.get('pass')==1 and not x.get('review');old={r['id']:r for r in ROWS};new={r['case']:r for r in DIAG.get('new_references',[])};answers=DIAG.get('answers',[]);out={}
 complete=len(ROWS)==32 and all(r['complete'] for r in ROWS) and len(new)==12 and len(answers)==44 and CTX['gens']['candidate_completed']==CTX['gens']['candidate_attempted']==32 and CTX['gens']['new_panel_completed']==CTX['gens']['new_panel_attempted']==39
 for label in POLICY['candidates']:
  selected={r['case']:r for r in answers if r['variant']==label};panels={}
  for panel,cc in [('regression',CASES),('new_panel',NEW_CASES)]:
   losses=[];gains=[];invalid=[];baseline_invalid=[];eligible=[];scores=Counter();bykind={};ctx_nat_losses=[]
   for c in cc:
    ref=(old.get(c['id'],{}).get('arms',{}) if panel=='regression' else new.get(c['id'],{}).get('arms',{}));n=ref.get('NATIVE_KV',{});p=selected.get(c['id'],{}).get('result',{});scores['native']+=int(passed(n));scores['candidate']+=int(passed(p));scores['context']+=int(passed(ref.get('CONTEXT',{})));scores['vanilla']+=int(passed(ref.get('VANILLA',{})))
    if not valid(p):invalid.append(c['id'])
    if not valid(n):baseline_invalid.append(c['id'])
    if passed(n):eligible.append(c['id']);bykind[c['vkey']]=bykind.get(c['vkey'],0)+1
    if passed(n) and valid(p) and not passed(p):losses.append(c['id'])
    if passed(p) and valid(n) and not passed(n):gains.append(c['id'])
    if passed(ref.get('CONTEXT',{})) and valid(n) and not passed(n):ctx_nat_losses.append(c['id'])
   panels[panel]=dict(n=len(cc),correct=dict(scores),native_pass_eligible=eligible,eligible_by_kind=bykind,native_to_candidate_losses=losses,candidate_gains=gains,candidate_invalid=invalid,native_invalid=baseline_invalid,context_to_native_losses=ctx_nat_losses)
  critical={sid:passed(selected.get(sid,{}).get('result',{})) for sid in POLICY['critical']};reg_ok=complete and all(critical.values()) and not(panels['regression']['native_to_candidate_losses'] or panels['regression']['candidate_invalid'] or panels['regression']['native_invalid']);new_ok=complete and not(panels['new_panel']['native_to_candidate_losses'] or panels['new_panel']['candidate_invalid'] or panels['new_panel']['native_invalid'] or panels['new_panel']['context_to_native_losses']);coverage=all(panels['new_panel']['eligible_by_kind'].get(k,0)>0 for k in ('complete','broken'))
  bank={}
  for r in selected.values():bank.setdefault(r['source_sha'],r['technical'])
  rawbytes=sum(t['raw_tensor_bytes'] for t in bank.values());packetbytes=sum(t['packet_tensor_bytes'] for t in bank.values());contentbytes=sum(t['content_bytes'] for t in bank.values());sinkbytes=sum(t['source_own_sink_bytes'] for t in bank.values());basisbytes=max([t['candidate_basis_mean_bytes_required'] for t in bank.values()] or [0]);fullcb=DIAG.get('codebook_operational_bytes',0)
  storage=dict(unique_sources=len(bank),raw_bank_tensor_bytes=rawbytes,packet_content_bytes=contentbytes,source_own_sink_bytes=sinkbytes,packet_bank_tensor_bytes=packetbytes,packet_only_ratio=packetbytes/rawbytes if rawbytes else None,shared_full_operational_codebook_bytes=fullcb,shared_diagnostic_EV_bytes=DIAG.get('codebook_EV_bytes',0),candidate_basis_mean_bytes_required=basisbytes,bank_plus_actual_full_operational_codebook_bytes=packetbytes+fullcb,bank_plus_actual_codebook_ratio=(packetbytes+fullcb)/rawbytes if rawbytes else None,bank_plus_pruned_basis_ratio=(packetbytes+basisbytes)/rawbytes if rawbytes else None,pruned_basis_is_accounting_estimate_not_export=True,runtime_cache_full_size=True,serialization_metadata_excluded=True)
  clean=not(CTX['errors'] or CTX['stop']) and INTEG.get('status')=='PASS';goal=reg_ok and new_ok and coverage and clean
  out[label]=dict(critical=critical,panels=panels,regression_goal='PASS' if reg_ok and clean else 'NOT_MET',new_panel_preservation='PASS' if new_ok and coverage and clean else ('INSUFFICIENT_NATIVE_COVERAGE' if new_ok and not coverage else 'NOT_MET'),joint_goal='PASS' if goal else 'NOT_MET',storage=storage)
  say('CANDIDATE_RESULT',json.dumps(dict(variant=label,**out[label]),ensure_ascii=False))
 canonical_invalid=[r['id'] for r in ROWS if not valid(r['arms'].get('PKV_D120',{}))];result=dict(complete=complete,candidates=out,canonical_D120_invalid=canonical_invalid,primary=POLICY['primary'],selection='TEST445 selected reference only; no adaptive tuning',scope='12 new cases / 3 new entities / same template; no statistical or broad generalization claim');say('GENERATIONS',dict(CTX['gens']));return result
MOD_POLICY=dict(revision='446 independent module composition v1',reference='K120_V128_BF16_OWN',entities=3,branches=['A','B','AB','BA','AX','XA'],arms=['VANILLA','CONTEXT','NATIVE_JOINT','PKV_JOINT','NATIVE_MODULES','PKV_MODULES'],forge='each module independently; source-only; no question; fixed TEST445 codebook',assembly='concatenate unrotated K/V; keep every exact source-own slot0; apply RoPE once at global positions; no re-forward or module refresh',controls='joint source cache is a separate control; independently forged concatenated cache is not claimed equal to joint-context cache',target='AB/BA connected place; A/B/AX/XA UNKNOWN',success='TEST445 primary preservation passes; all 18 PKV_MODULES first-line answers correct; all control arms technically valid; finite/start/cache/slot0/seal/integrity pass',scope='3 entities x 6 branches; fixed templates; no AGI proof, autonomous retrieval or context-limit extension claim')
MOD_ROWS=[];MOD_CASES=[];MOD_BANK={}
def build_module_cases():
 out=[]
 for j,(_,obj,a,b) in enumerate(NEW_ENT):
  e=6+j;Z,Y=f'Z{e}',f'Y{e}';place=b if e%2==0 else a;modules=dict(A=f'The {obj} is inside container {Z}.',B=f'Container {Z} is in the {place}.',X=f'An unrelated container {Y} is in the {place}.');q=f'Name the place (not a container) where the {obj} is located. If the records do not connect the {obj} to any place, answer UNKNOWN.'+STYLE
  for branch in MOD_POLICY['branches']:
   order=list(branch);records=[modules[k] for k in order];target=place if branch in ('AB','BA') else 'UNKNOWN';out.append(dict(id=f'module{e}:{branch}',family='modular',vkey=branch,group=f'module{e}',member=branch,ent=e,load=0,pos='',k=0,tmpl='module|'+branch,records=records,source='\n'.join(records),question=q,target=target,alts=sorted(set([a,b,Z,Y,'UNKNOWN'])-{target}),module_order=[f'{e}:{k}' for k in order],modules={f'{e}:{k}':modules[k] for k in order}))
 return out
def module_cpu_checks():
 assert len(MOD_CASES)==18 and len({c['id'] for c in MOD_CASES})==18
 for c in MOD_CASES:
  assert len(c['module_order'])==len(c['records'])
  assert c['target']!='UNKNOWN' if c['member'] in ('AB','BA') else c['target']=='UNKNOWN'
  if c['member'] in ('AB','BA'):
   e=c['ent'];z=f'Z{e}';assert all(z in r for r in c['records'])
  if c['member'] in ('AX','XA'):assert sum(f'Z{c["ent"]}' in r for r in c['records'])==1
  assert 'QUESTION:' not in c['source'] and c['target'] not in ('Z6','Z7','Z8')
 return len(MOD_CASES)
@infer
def retain_module(key,source):
 if key in MOD_BANK:
  if MOD_BANK[key]['source_sha']!=sha(source.encode()):raise TechStop('module ID reused for different source')
  return MOD_BANK[key]
 f=forge(source);installed,packet,seal,tech=candidate_packet(f,POLICY['primary']);raw={n:[x.clone() for x in f[n]] for n in ('K','V')};T=f['T'];del f
 rec=dict(key=key,source_sha=sha(source.encode()),T=T,raw=raw,packet=packet,seal=seal,technical=tech)
 K,V=decode_module(rec);audit=install(K,V)
 if not kv_equal(audit[:2],installed[:2]):raise TechStop('standalone packet decode differs from TEST445 candidate')
 MOD_BANK[key]=rec;checkpoint(dict(key=key,source_sha=rec['source_sha'],T=T,packet_seal=seal,technical=tech),'module_inventory.jsonl');return rec
@infer
def decode_module(rec):
 packet=rec['packet'];out={n:[] for n in ('K','V')}
 if code_sha(packet)!=rec['seal']:raise TechStop('module seal changed before decode')
 for n in ('K','V'):
  d=120 if n=='K' else 128
  for L in range(NL):
   mu,B,_=CB[L][n];coeff=packet[n][L];sink=packet[n][NL+L];content=(mu+torch.einsum('thd,hid->thi',coeff.float(),B[:,:,:d])).reshape(-1,KVD).to(torch.bfloat16);out[n].append(torch.cat([sink,content]))
 return out['K'],out['V']
@infer
def assemble_modules(order,compressed):
 recs=[MOD_BANK[k] for k in order];parts=[dict(zip(('K','V'),decode_module(r))) if compressed else r['raw'] for r in recs];out={n:[torch.cat([p[n][L] for p in parts]) for L in range(NL)] for n in ('K','V')};T=sum(r['T'] for r in recs);offset=0;checks=[]
 for r in recs:
  for n in ('K','V'):
   for L in range(NL):checks.append(torch.equal(out[n][L][offset:offset+1],r['packet'][n][NL+L]))
  offset+=r['T']
 slot0=all(checks);finite=all_finite(out['K']+out['V']);kv=install(out['K'],out['V']);finite=finite and all_finite(list(kv[0])+list(kv[1]));sealed=all(code_sha(r['packet'])==r['seal'] for r in recs);CTX['seal_checks'].append(sealed)
 tech=dict(finite=finite,start_ok=kv[2]==T,source_own_slot0_exact=slot0,packet_seals_ok=sealed,slots=T,module_count=len(recs),slot0_positions=[sum(r['T'] for r in recs[:i]) for i in range(len(recs))],global_rope=True,reforward=False)
 if not all(tech[k] for k in ('finite','start_ok','source_own_slot0_exact','packet_seals_ok')):raise TechStop('module assembly integrity failed')
 return kv,tech
@infer
def run_module_case(c):
 c['sha']=sha(c['source'].encode());c['q_tokens']=len(enc_ids(FMT.format(q=c['question'])));c['src_tokens']=1+len(enc_ids(c['source']+SEP))
 for key,source in c['modules'].items():retain_module(key,source)
 nkv,nt=assemble_modules(c['module_order'],False);pkv,pt=assemble_modules(c['module_order'],True)
 if nkv[2]!=pkv[2] or pkv[2]+c['q_tokens']+MAX_NEW>MAX_TOTAL:raise TechStop('module length/budget mismatch')
 f=forge(c['source']);joint=install(f['K'],f['V']);jp,packet,seal,jt=candidate_packet(f,POLICY['primary']);del f
 row=dict(case=c['id'],branch=c['member'],ent=c['ent'],target=c['target'],question=c['question'],records=c['records'],module_order=c['module_order'],assembly=dict(native=nt,pkv=pt,joint=jt),arms={})
 for arm in MOD_POLICY['arms']:
  kv={'NATIVE_JOINT':joint,'PKV_JOINT':jp,'NATIVE_MODULES':nkv,'PKV_MODULES':pkv}.get(arm)
  if arm=='VANILLA' and c['question'] in VAN:out=dict(VAN[c['question']]);CTX['gens']['module_vanilla_reused']+=1
  else:
   out=gen(c,arm,kv,() if arm=='CONTEXT' else c['records'],'modules')
   if arm=='VANILLA':VAN[c['question']]=dict(out)
  if out.get('status')=='OK':out.update(score(out['text'],c))
  row['arms'][arm]=out
  if out.get('status')!='OK' or not out.get('valid'):raise TechStop(f'module arm invalid: {c["id"]}/{arm}: {out}')
 lg=last_logits(enc_ids(FMT.format(q=c['question'])),1,mkcache(pkv),pkv[2]+c['q_tokens']);ref=last_logits(enc_ids(FMT.format(q=c['question'])),1,mkcache(nkv),nkv[2]+c['q_tokens']);row['first_token_vs_native_modules']=logit_stats(lg,ref)
 sealed=code_sha(packet)==seal and all(code_sha(MOD_BANK[k]['packet'])==MOD_BANK[k]['seal'] for k in c['module_order']);CTX['seal_checks'].append(sealed);row['seals_ok']=sealed
 if not sealed:raise TechStop('module packet mutated during readout')
 MOD_ROWS.append(row);DIAG['module_rows']=MOD_ROWS;checkpoint(row,'module_answers.jsonl');say('MODULE',json.dumps(dict(case=c['id'],target=c['target'],slots=pkv[2],scores={a:x.get('pass') for a,x in row['arms'].items()},raw={a:x.get('text') for a,x in row['arms'].items()}),ensure_ascii=False))
 del joint,jp,packet,nkv,pkv,lg,ref
def summary446_modules(base):
 valid=lambda x:x.get('status')=='OK' and bool(x.get('valid'));passed=lambda x:valid(x) and x.get('pass')==1 and not x.get('review');clean=not(CTX['errors'] or CTX['stop']) and INTEG.get('status')=='PASS';complete=len(MOD_ROWS)==18 and {r['case'] for r in MOD_ROWS}=={c['id'] for c in MOD_CASES};scores={a:sum(passed(r['arms'].get(a,{})) for r in MOD_ROWS) for a in MOD_POLICY['arms']};bybranch={b:{a:sum(passed(r['arms'].get(a,{})) for r in MOD_ROWS if r['branch']==b) for a in MOD_POLICY['arms']} for b in MOD_POLICY['branches']};losses=[r['case'] for r in MOD_ROWS if passed(r['arms']['NATIVE_MODULES']) and not passed(r['arms']['PKV_MODULES'])];composition_losses=[r['case'] for r in MOD_ROWS if passed(r['arms']['NATIVE_JOINT']) and not passed(r['arms']['NATIVE_MODULES'])];invalid=[f'{r["case"]}/{a}' for r in MOD_ROWS for a in MOD_POLICY['arms'] if not valid(r['arms'].get(a,{}))];ref_ok=base.get('candidates',{}).get(POLICY['primary'],{}).get('joint_goal')=='PASS';tech=complete and not invalid and all(r['seals_ok'] for r in MOD_ROWS) and all(CTX['seal_checks']);goal=clean and tech and ref_ok and scores['PKV_MODULES']==18
 results={r['case']:r for r in MOD_ROWS};pairs=[]
 for e in (6,7,8):
  for x,y in [('AB','BA'),('AX','XA')]:
   a=results.get(f'module{e}:{x}');b=results.get(f'module{e}:{y}')
   if a and b:pairs.append(dict(ent=e,branches=[x,y],pkv_pass=[passed(a['arms']['PKV_MODULES']),passed(b['arms']['PKV_MODULES'])],native_pass=[passed(a['arms']['NATIVE_MODULES']),passed(b['arms']['NATIVE_MODULES'])]))
 inventory=[dict(key=k,source_sha=r['source_sha'],T=r['T'],packet_seal=r['seal'],technical=r['technical']) for k,r in MOD_BANK.items()];packetbytes=sum(r['technical']['packet_tensor_bytes'] for r in MOD_BANK.values());rawbytes=sum(r['technical']['raw_tensor_bytes'] for r in MOD_BANK.values());cb=DIAG.get('codebook_operational_bytes',0);storage=dict(unique_modules=len(MOD_BANK),packet_tensor_bytes=packetbytes,raw_tensor_bytes=rawbytes,packet_only_ratio=packetbytes/rawbytes if rawbytes else None,shared_operational_codebook_bytes=cb,packet_plus_codebook_ratio=(packetbytes+cb)/rawbytes if rawbytes else None,runtime_cache_full_size=True,raw_control_copies_excluded=True,metadata_serialization_excluded=True);DIAG['module_inventory']=inventory
 out=dict(complete=complete,reference_preservation='PASS' if ref_ok else 'NOT_MET',technical='PASS' if tech and clean else 'NOT_MET',n=18,correct=scores,bybranch=bybranch,native_modules_to_pkv_modules_losses=losses,joint_native_to_independent_native_losses=composition_losses,invalid=invalid,order_pairs=pairs,storage=storage,goal='PASS' if goal else 'NOT_MET',interpretation='PASS applies only to this fixed modular panel; NOT_MET does not establish AGI impossibility',policy=MOD_POLICY);say('MODULE_RESULT',json.dumps(out,ensure_ascii=False));return out
NEW_CASES=[];RUN=None;INTEG=dict(status='NOT_EVALUATED');SUMMARY=None
try:
 say('='*100);say(f'TEST446 — MODULAR MEMORY COMPOSITION / run {RUN_ID}');say('POLICY',POLICY);stage('cpu_setup');CASES=build_cases();BYID={c['id']:c for c in CASES};NEW_CASES=build_new_cases();MOD_CASES=build_module_cases();say('MODULE_CPU',module_cpu_checks(),'PASS');say('SCORER',scorer_selftest(),new_selftest(),subject_selftest(),'PASS')
 for c in CASES:EXPECTED.setdefault(c['vkey'],OrderedDict()).setdefault(c['group'],[]).append(c['id'])
 RUN=(Path('/content/AKBASCORE_TEST446') if os.path.isdir('/content') else Path('/tmp/AKBASCORE_TEST446'))/RUN_ID;RUN.mkdir(parents=True,exist_ok=True)
 panel_lock=dict(policy=POLICY,regression_cases=[meta(c) for c in CASES],new_cases=[meta(c) for c in NEW_CASES],new_entities=NEW_ENT,scoring=SCORING_POLICY,module_policy=MOD_POLICY,module_cases=MOD_CASES);PANEL_SHA=sha(canon(panel_lock));(RUN/'panel_lock.json').write_bytes(canon(panel_lock));say('PANEL_LOCK',PANEL_SHA);stage('imports')
 for mod,pkg in [('torch','torch'),('transformers','transformers'),('accelerate','accelerate')]:
  if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,'-m','pip','install','-q',pkg])
 import torch,transformers
 import torch.nn.functional as F
 from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
 from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
 if not torch.cuda.is_available():raise TechStop('Fresh A100 CUDA runtime required')
 DEVICE=torch.device('cuda');os.environ['TOKENIZERS_PARALLELISM']='false';torch.set_grad_enabled(False);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
 CTX['env'].update(torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),float32_matmul_precision=torch.get_float32_matmul_precision(),matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32);say('ENV',CTX['env']);stage('model_load')
 tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True);tv=tuple(int(re.sub(r'\D','',v) or 0) for v in transformers.__version__.split('.')[:2]);DT='dtype' if tv>=(4,56) else 'torch_dtype';model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={'':0},attn_implementation='sdpa',**{DT:torch.bfloat16}).eval()
 for p in model.parameters():p.requires_grad_(False)
 cfg=model.config;layers=model.model.layers;H=cfg.hidden_size;NH=cfg.num_attention_heads;NKV=cfg.num_key_value_heads;HD=getattr(cfg,'head_dim',None) or H//NH;KVD=NKV*HD
 if (len(layers),H,NH,NKV,HD)!=(28,3584,28,4,128):raise TechStop('architecture mismatch')
 if next(model.parameters()).dtype!=torch.bfloat16 or getattr(cfg,'use_sliding_window',False):raise TechStop('dtype/sliding-window mismatch')
 PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id;ge=model.generation_config.eos_token_id;EOS=sorted({tok.eos_token_id,*(ge if isinstance(ge,(list,tuple)) else [ge])}-{None});GEN=dict(max_new_tokens=MAX_NEW,do_sample=False,repetition_penalty=1.0,use_cache=True,pad_token_id=PAD,eos_token_id=EOS)
 FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
 def sentinel():
  gpu_guard();h=hashlib.sha256();s=[]
  for i,t in enumerate(FP_T):
   s.append(float(t.sum(dtype=torch.float32)));flat=t.detach().reshape(-1);n=flat.numel();c=min(256,n);offs=sorted({k*(n-c)//15 for k in range(16)});h.update(f'{i}|{tuple(t.shape)}|{t.dtype}|{n}|{offs}|'.encode());h.update(torch.cat([flat[o:o+c] for o in offs]).float().cpu().numpy().tobytes())
  return s,h.hexdigest()
 FP0,SENT0=sentinel()
 if sentinel()!=(FP0,SENT0):raise TechStop('sentinel not repeatable')
 CTX['model_loaded']=True;stage('forge_equivalence');LTK_OK=False;_ids=[PAD]+enc_ids(' '.join(CORPUS[:6])+SEP);a=kv_from_ids(_ids,False);b=kv_from_ids(_ids,False)
 if not kv_equal(a,b):raise TechStop('forge not repeatable')
 if 'logits_to_keep' in inspect.signature(model.forward).parameters:LTK_OK=True;c=kv_from_ids(_ids,True);LTK_OK=kv_equal(a,c);del c
 del a,b;stage('codebook');CO=[forge(s) for s in CORPUS];SINK={n:[CO[0][n][L][:1].clone() for L in range(NL)] for n in ('K','V')};CB=[]
 for L in range(NL):
  e={}
  for n in ('K','V'):
   R=torch.cat([c[n][L][1:] for c in CO]).float().view(-1,NKV,HD);MU=[];B=[];EV=[]
   for h in range(NKV):
    X=R[:,h];mu=X.mean(0);_,S,Vh=torch.linalg.svd(X-mu,full_matrices=False);m=min(DMAX,Vh.shape[0]);bb=Vh[:m].T.contiguous()
    if m<DMAX:bb=F.pad(bb,(0,DMAX-m))
    cs=S.square().cumsum(0)/S.square().sum();ev=torch.ones(DMAX,device=DEVICE);ev[:min(DMAX,len(cs))]=cs[:DMAX];MU.append(mu);B.append(bb);EV.append(ev)
   e[n]=(torch.stack(MU),torch.stack(B),torch.stack(EV))
  CB.append(e)
 DIAG['codebook_operational_bytes']=sum(CB[L][n][j].numel()*CB[L][n][j].element_size() for L in range(NL) for n in ('K','V') for j in (0,1));DIAG['codebook_EV_bytes']=sum(CB[L][n][2].numel()*CB[L][n][2].element_size() for L in range(NL) for n in ('K','V'));say('CODEBOOK',sum(c['T']-1 for c in CO),{n:float(torch.stack([CB[L][n][2][:,D-1] for L in range(NL)]).mean()) for n in ('K','V')});del CO,R,X;stage('case_tokens')
 for c in CASES+NEW_CASES:
  c['sha']=sha(c['source'].encode());c['src_tokens']=1+len(enc_ids(c['source']+SEP));c['q_tokens']=len(enc_ids(FMT.format(q=c['question'])));c['valid']=c['src_tokens']+c['q_tokens']+MAX_NEW<=MAX_TOTAL
  if not c['valid']:raise TechStop('case exceeds budget')
  if any(s in c['source'] or c['source'] in s for s in CORPUS):raise TechStop('codebook overlap')
 assign_foreign();LMAX=max(c['src_tokens'] for c in CASES);stage('sink_calibration');CAL=[]
 for n in sorted({0,16,64,256,1024,2048,LMAX}):
  ids=neutral_ids(n);K,V=kv_from_ids(ids);m=sink_metrics(dict(ids=ids,T=len(ids),K=K,V=V));m['T']=len(ids);CAL.append(m);say('CAL',len(ids),fmt_sink(m));del K,V
 CAL_REL=max(m['max_rel_row'] for m in CAL if m['T']>1);SINK_GATE=min(max(SINK_ENV_MULT*CAL_REL,SINK_REL_FLOOR),SINK_REL_CAP)
 if not all(m['finite_pos0'] for m in CAL):raise TechStop('non-finite calibration')
 f=forge(CORPUS[0])
 if not all(torch.equal(f[n][L][:1],SINK[n][L]) for n in ('K','V') for L in range(NL)):raise TechStop('shared sink not repeatable')
 del f;say('SINK_GATE',SINK_GATE,'unchanged cap',SINK_REL_CAP);stage('native_audit')
 for n in sorted({16,256,2048,LMAX}):
  r=native_audit(neutral_ids(n));say('NATIVE',n,r)
  if max(r.values())>=0.05:raise TechStop('native audit failed')
 stage('cache_probe')
 for sid in ['retain0:alone','boundary1:fwd']:
  r=cache_probe(BYID[sid]);DIAG.setdefault('cache_probes',[]).append(r);say('PROBE',json.dumps({k:v for k,v in r.items() if k!='step_detail'},ensure_ascii=False))
  if not r['pass']:raise TechStop('cache probe failed')
 LOCK=dict(test=TEST,model=MODEL_ID,seed=SEED,D=D,DMAX=DMAX,FMT=FMT,SEP=SEP,STYLE=STYLE,PAD=PAD,EOS=EOS,max_new=MAX_NEW,corpus=CORPUS,cases=[meta(c) for c in CASES],new_cases=[meta(c) for c in NEW_CASES],panel_sha=PANEL_SHA,scoring=SCORING_POLICY,policy=POLICY,sink_gate=SINK_GATE,module_policy=MOD_POLICY,module_cases=MOD_CASES);LOCK_SHA=sha(canon(LOCK));say('LOCK',LOCK_SHA);(RUN/'experiment_lock.json').write_bytes(canon(LOCK));stage('main_run')
 for i,c in enumerate(CASES,1):
  row=run_case(c);ROWS.append(row);checkpoint(row);say(f'{i:02d}/32',c['id'],' '.join(ABBR[a]+':'+sym(row,a) for a in ARMS))
  if not gpu_alive() or any(x.get('status') not in ('OK','NOT_APPLICABLE','TECHNICAL_INVALID') for x in row['arms'].values()):raise TechStop('main technical error; inspect checkpoint')
 stage('reference_preservation','fixed TEST445 primary on all 32 regression cases')
 for c in CASES:run_candidates(c,'regression')
 stage('new_panel','12 locked new cases; same template; no adaptive selection')
 for c in NEW_CASES:run_candidates(c,'new_panel')
 stage('modular_memory','18 locked branches / independent source-only forge / fixed assembly')
 for c in MOD_CASES:run_module_case(c)
 for rep in REPS.values():CTX['seal_checks'].append(code_sha(rep['code'])==rep['seal'])
except BaseException as ex:
 CTX['stop']=report(ex)
finally:
 if CTX['model_loaded']:INTEG=integrity_check()
 say('INTEGRITY',INTEG)
 try:
  SUMMARY=summary445();SUMMARY['modular']=summary446_modules(SUMMARY)
  if RUN is not None:
   payload=dict(test=TEST,run_id=RUN_ID,lock=globals().get('LOCK'),lock_sha=globals().get('LOCK_SHA'),env=CTX['env'],rows=ROWS,new_cases=NEW_CASES,panel_sha=globals().get('PANEL_SHA'),diagnostics=DIAG,summary=SUMMARY,integrity=INTEG,errors=CTX['errors'],stop=CTX['stop']);pb=canon(payload);(RUN/'payload.json').write_bytes(pb);(RUN/'manifest.json').write_bytes(canon(dict(payload='payload.json',sha256=sha(pb),bytes=len(pb))));say('OUTPUT',str(RUN),'SHA256',sha(pb))
 except BaseException as ex:report(ex,extra='finalize')
 say(f'TEST446 FINAL STATE: {"COMPLETED" if CTX["stop"] is None and not CTX["errors"] else "STOPPED"} | rows {len(ROWS)}/32 | new {len(DIAG.get("new_references",[]))}/12 | modules {len(MOD_ROWS)}/18 | integrity {INTEG.get("status")}')
