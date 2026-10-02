# TEST444 — NIRVANA: K120 + V precision/rank solution; standalone fresh A100 cell.
import os,sys,re,json,time,random,hashlib,importlib.util,subprocess,traceback,inspect,platform
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter,OrderedDict
import numpy as np
TEST="444";SEED=384;D=120;DMAX=128;NL=28;MAX_NEW=32;MAX_TOTAL=6144;SHORT_T=256;LONG_KEEP=3;PROBE_STEPS=6
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
 risks=[];target=c["target"];others=[x for x in c["alts"] if not identifier(x)]
 for part in segments(tail):
  n=norm(part)
  if mention(n,target) in ("negated","conflict"):risks.append("target denied in same-line continuation")
  answer_cue=bool(re.search(r"\b(?:final answer|the answer|answer is|actually|instead)\b",part,re.I))
  location_cue=bool(re.search(r"\b(?:is|was|are|were)\s+(?:currently\s+)?(?:(?:located|stored|kept)\s+)?(?:in|at|inside)\s+",part,re.I))
  for other in others:
   state=mention(n,other);bare=n in (norm(other),norm("the "+other),norm("in the "+other))
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
 if risks:out.update(category="contradictory_continuation",semantic="review",review=1,**{"pass":0})
 elif out["pass"] and re.search(r"\b(?:maybe|perhaps|possibly|probably|likely|might|may be|could be|not certain|not sure)\b",span,re.I):out.update(category="uncertain_answer",semantic="review",review=1,**{"pass":0})
 out.update(answer_span=span,explanation_tail=tail,first_line=line,tail_flags=risks,target_anywhere=int(has(text,c["target"])),alt_anywhere=int(any(has(text,x) for x in c["alts"])))
 return out
def new_selftest():
 L=dict(target="northern archive",alts=["southern vault","UNKNOWN"]);S=dict(target="southern vault",alts=["northern archive","Z0","V0"]);E=dict(target="eastern platform",alts=["western gate","AMBIGUOUS"]);U=dict(target="UNKNOWN",alts=["northern archive","southern vault","Z0"])
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
POLICY=dict(primary='K120_bfloat16_V128_float32_OWN',variants=[[120,'bfloat16'],[120,'float32'],[124,'bfloat16'],[124,'float32'],[128,'bfloat16'],[128,'float32'],['RAW','raw']],critical=['broken1:rev','broken2:rev'],patches={'K9':[[9,'K']],'V14':[[14,'V']],'KV9':[[9,'K'],[9,'V']],'V9_V14':[[9,'V'],[14,'V']],'KV9_KV14':[[9,'K'],[9,'V'],[14,'K'],[14,'V']]},shape_lengths=[29,203,256,265],sink='canonical shared SINK unchanged; candidate retains exact source-own slot0',success='primary correct on both critical cases; no case-level native->candidate losses; finite/cache/source-own sink valid; integrity PASS',scope='targeted regression only; no held-out claim; no automatic variant selection')
@infer
def hybrid(f,vd,precision):
 own={n:[f[n][L][:1].clone() for L in range(NL)] for n in ('K','V')};out={};code={}
 for n,d,dt in [('K',120,torch.bfloat16),('V',vd,torch.float32 if precision=='float32' else torch.bfloat16)]:
  if d=='RAW':out[n]=[x.clone() for x in f[n]];code[n]=[x[1:].clone() for x in f[n]];continue
  code[n]=[torch.einsum('thi,hid->thd',f[n][L][1:].float().view(-1,NKV,HD)-CB[L][n][0],CB[L][n][1][:,:,:d]).to(dt) for L in range(NL)]
  out[n]=[torch.cat([own[n][L],(CB[L][n][0]+torch.einsum('thd,hid->thi',code[n][L].float(),CB[L][n][1][:,:,:d])).reshape(-1,KVD).to(torch.bfloat16)]) for L in range(NL)]
 if not all_finite(out['K']+out['V']+[x for a in code.values() for x in a]):raise TechStop('non-finite hybrid')
 exact=all(torch.equal(out[n][L][:1],own[n][L]) for n in ('K','V') for L in range(NL))
 if not exact:raise TechStop('source-own slot0 not exact')
 seal=code_sha(code);kv=install(out['K'],out['V'])
 if not all_finite(list(kv[0])+list(kv[1])):raise TechStop('non-finite installed hybrid')
 sizes=sum(x.numel()*x.element_size() for a in code.values() for x in a)+sum(x.numel()*x.element_size() for a in own.values() for x in a)
 rawbytes=sum(x.numel()*x.element_size() for n in ('K','V') for x in f[n]);fidelity={n:max(float((out[n][L][1:].float()-f[n][L][1:].float()).norm()/f[n][L][1:].float().norm().clamp_min(1e-12)) for L in range(NL)) for n in ('K','V')}
 return kv,code,seal,dict(source_own_slot0_exact=exact,stored_tensor_bytes=sizes,raw_tensor_bytes=rawbytes,ratio=sizes/rawbytes,fidelity=fidelity,runtime_cache_full_size=True,codebook_bytes_excluded=True)
@infer
def measure(c,label,kv,reference,technical=None):
 qids=enc_ids(FMT.format(q=c['question']));lg=last_logits(qids,1,mkcache(kv),kv[2]+len(qids));r=dict(case=c['id'],variant=label,**logit_stats(lg,reference));out=gen(c,label,kv,c['records'],'diagnostic')
 if out.get('status')=='OK':out.update(score(out['text'],c))
 if technical is not None:r['technical']=technical;out['valid']=bool(out.get('valid') and technical.get('source_own_slot0_exact'))
 r['result']=out;DIAG.setdefault('answers',[]).append(r);checkpoint(r,'diagnostics.jsonl');say('ANSWER',json.dumps(dict(case=c['id'],variant=label,passed=out.get('pass'),valid=out.get('valid'),first=r['first_piece'],tvd=r['first_tvd_vs_raw'],raw=out.get('text')),ensure_ascii=False));return r
@infer
def solve_case(c,primary_only=False):
 f=forge(c['source']);raw=install(f['K'],f['V']);qids=enc_ids(FMT.format(q=c['question']));reference=last_logits(qids,1,mkcache(raw),f['T']+len(qids));settings=[(128,'float32')] if primary_only else POLICY['variants']
 for vd,dt in settings:
  label=f'K120_bfloat16_V{vd}_{dt}_OWN';kv,code,seal,tech=hybrid(f,vd,dt);measure(c,label,kv,reference,tech);CTX['seal_checks'].append(code_sha(code)==seal);del kv,code
 if not primary_only:
  base,code,seal,tech=hybrid(f,120,'bfloat16')
  for name,patches in POLICY['patches'].items():
   kk=list(base[0]);vv=list(base[1])
   for L,n in patches:(kk if n=='K' else vv)[L]=(raw[0] if n=='K' else raw[1])[L]
   measure(c,'D120_OWN_RESTORE_'+name,(tuple(kk),tuple(vv),f['T']),reference,tech)
  CTX['seal_checks'].append(code_sha(code)==seal)
 del f,raw,reference
@infer
def shape_check():
 pool=neutral_ids(max(POLICY['shape_lengths'])+1);baseline=None
 for T in POLICY['shape_lengths']:
  ids=pool[:T];o=model(input_ids=torch.tensor([ids],device=DEVICE),output_hidden_states=True,use_cache=False,return_dict=True,**ltk(1));data=[];snap=[]
  for L in range(NL):
   h=o.hidden_states[L][0];z=layers[L].input_layernorm(h);z1=layers[L].input_layernorm(h[:1]);a=layers[L].self_attn;row=dict(layer=L,hidden_delta=0.0 if baseline is None else float((h[:1].float()-baseline[L]['h']).abs().max()),norm_batch_single=float((z[:1].float()-z1.float()).abs().max()));rec={'h':h[:1].float().clone(),'z':z[:1].float().clone()}
   for n,proj in [('K',a.k_proj),('V',a.v_proj)]:
    batch=proj(z)[:1];single=proj(z1);fp32=F.linear(z1.float(),proj.weight.float(),None if proj.bias is None else proj.bias.float());rec[n]=batch.float().clone()
    if not all_finite([batch,single,fp32]):raise TechStop('non-finite projection xray')
    row[n]=dict(batch_vs_single=float((batch.float()-single.float()).abs().max()),batch_vs_fp32=float((batch.float()-fp32).abs().max()),single_vs_fp32=float((single.float()-fp32).abs().max()),vs_reference=0.0 if baseline is None else float((batch.float()-baseline[L][n]).abs().max()))
   snap.append(rec);data.append(row)
  if baseline is None:baseline=snap
  K,V=kv_from_ids(ids);m=sink_metrics(dict(ids=ids,T=T,K=K,V=V));r=dict(T=T,sink=m,gate_ok=m['max_rel_row']<=SINK_GATE,layers=data);DIAG.setdefault('shape',[]).append(r);checkpoint(r,'diagnostics.jsonl')
  first=next((x for x in data if x['hidden_delta'] or x['K']['vs_reference'] or x['V']['vs_reference']),None);say('SHAPE',json.dumps(dict(T=T,gate_ok=r['gate_ok'],max_rel_row=m['max_rel_row'],first_changed=first,layer27=data[27]),ensure_ascii=False));del o,K,V,snap
 say('FP32 projection is an observational reference; model/forge/cache remain BF16; no kernel-cause claim.')
def checkpoint(obj,name='checkpoint.jsonl'):
 if RUN is None:return
 with (RUN/name).open('ab') as fh:fh.write(canon(obj)+b'\n')
def summary444():
 lookup={r['id']:r for r in ROWS};answers=DIAG.get('answers',[]);primary={r['case']:r for r in answers if r['variant']==POLICY['primary']};valid=lambda x:x.get('status')=='OK' and x.get('valid');passed=lambda x:valid(x) and x.get('pass')==1 and not x.get('review');losses=[];gains=[];invalid=[]
 for c in CASES:
  p=primary.get(c['id'],{}).get('result',{});n=lookup.get(c['id'],{}).get('arms',{}).get('NATIVE_KV',{})
  if not valid(p):invalid.append(c['id'])
  if passed(n) and valid(p) and not passed(p):losses.append(c['id'])
  if passed(p) and valid(n) and not passed(n):gains.append(c['id'])
 critical={sid:passed(primary.get(sid,{}).get('result',{})) for sid in POLICY['critical']};canonical_invalid=[r['id'] for r in ROWS if not valid(r['arms'].get('PKV_D120',{}))];bad=[r['case']+'/'+r['variant'] for r in answers if not valid(r['result'])]
 completed=len(ROWS)==32 and all(r['complete'] for r in ROWS) and len(primary)==32 and CTX['gens']['diagnostic_completed']==CTX['gens']['diagnostic_attempted']==54 and len(DIAG.get('shape',[]))==4
 ok=completed and all(critical.values()) and not(losses or invalid or bad or CTX['errors'] or CTX['stop']) and INTEG.get('status')=='PASS'
 result=dict(primary=POLICY['primary'],complete=completed,critical=critical,native_to_primary_losses=losses,primary_gains=gains,primary_invalid=invalid,diagnostic_invalid=bad,canonical_D120_invalid=canonical_invalid,primary_regression_goal='PASS' if ok else 'NOT_MET',scope='targeted panel; no held-out or generalization claim; canonical D120 not rescued',storage='FP32 V coefficients can exceed raw BF16 bytes; reported tensor ratio excludes shared codebook')
 say('RESULT',json.dumps(result,ensure_ascii=False));say('GENERATIONS',dict(CTX['gens']));return result
RUN=None;INTEG=dict(status='NOT_EVALUATED');SUMMARY=None
try:
 say('='*100);say(f'TEST444 — HYBRID V SOLUTION / run {RUN_ID}');say('POLICY',POLICY);stage('cpu_setup');CASES=build_cases();BYID={c['id']:c for c in CASES};say('SCORER',scorer_selftest(),new_selftest(),subject_selftest(),'PASS')
 for c in CASES:EXPECTED.setdefault(c['vkey'],OrderedDict()).setdefault(c['group'],[]).append(c['id'])
 RUN=(Path('/content/AKBASCORE_TEST444') if os.path.isdir('/content') else Path('/tmp/AKBASCORE_TEST444'))/RUN_ID;RUN.mkdir(parents=True,exist_ok=True)
 stage('imports')
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
 say('CODEBOOK',sum(c['T']-1 for c in CO),{n:float(torch.stack([CB[L][n][2][:,D-1] for L in range(NL)]).mean()) for n in ('K','V')});del CO,R,X;stage('case_tokens')
 for c in CASES:
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
 LOCK=dict(test=TEST,model=MODEL_ID,seed=SEED,D=D,DMAX=DMAX,FMT=FMT,SEP=SEP,STYLE=STYLE,PAD=PAD,EOS=EOS,max_new=MAX_NEW,corpus=CORPUS,cases=[meta(c) for c in CASES],scoring=SCORING_POLICY,policy=POLICY,sink_gate=SINK_GATE);LOCK_SHA=sha(canon(LOCK));say('LOCK',LOCK_SHA);(RUN/'experiment_lock.json').write_bytes(canon(LOCK));stage('main_run')
 for i,c in enumerate(CASES,1):
  row=run_case(c);ROWS.append(row);checkpoint(row);say(f'{i:02d}/32',c['id'],' '.join(ABBR[a]+':'+sym(row,a) for a in ARMS))
  if not gpu_alive() or any(x.get('status') not in ('OK','NOT_APPLICABLE','TECHNICAL_INVALID') for x in row['arms'].values()):raise TechStop('main technical error; inspect checkpoint')
 stage('hybrid_and_full_answer_repairs')
 for c in CASES:solve_case(c,primary_only=c['id'] not in POLICY['critical'])
 # 30 primary-only + 2*(7 variants+5 repairs) = 54 generations.
 stage('shape_check');shape_check()
 for rep in REPS.values():CTX['seal_checks'].append(code_sha(rep['code'])==rep['seal'])
except BaseException as ex:
 CTX['stop']=report(ex)
finally:
 if CTX['model_loaded']:INTEG=integrity_check()
 say('INTEGRITY',INTEG)
 try:
  SUMMARY=summary444()
  if RUN is not None:
   payload=dict(test=TEST,run_id=RUN_ID,lock=globals().get('LOCK'),lock_sha=globals().get('LOCK_SHA'),env=CTX['env'],rows=ROWS,diagnostics=DIAG,summary=SUMMARY,integrity=INTEG,errors=CTX['errors'],stop=CTX['stop']);pb=canon(payload);(RUN/'payload.json').write_bytes(pb);(RUN/'manifest.json').write_bytes(canon(dict(payload='payload.json',sha256=sha(pb),bytes=len(pb))));say('OUTPUT',str(RUN),'SHA256',sha(pb))
 except BaseException as ex:report(ex,extra='finalize')
 say(f'TEST444 FINAL STATE: {"COMPLETED" if CTX["stop"] is None and not CTX["errors"] else "STOPPED"} | rows {len(ROWS)}/32 | integrity {INTEG.get("status")}')
