# TEST685 v2 — AKBASCORE MAM | VERIFIED QWEN CUT3 × LOCKED TEST677 CONFLICT PANEL
# Independent Colab A100 single cell; TEST566 original engine executed unchanged.
# No fabricated DC3, no training, no answer fixes. Mistral TEST684 reference is historical, not rerun.
import os,sys,re,ast,json,time,hashlib,urllib.request,traceback,gc,random
from pathlib import Path
from collections import Counter
START685=time.perf_counter();ROOT=Path('/content/AKBASCORE_TEST685') if Path('/content').exists() else Path('/tmp/AKBASCORE_TEST685');ROOT.mkdir(parents=True,exist_ok=True)
URL='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/'
BLOBS={'566.py':'b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71','575.py':'569c92c61224ea408cdb6e81dbe45dde06dc05fd','677.py':'5b65959423b7bbdd209b654417eb5fa30ceb2626'}
def require685(n,ok):
 print(('PASS ' if ok else 'FAIL ')+n,flush=True)
 if not ok:raise RuntimeError(n)
def gitblob685(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def fetch685(n):
 p=Path('/content')/n
 if p.is_file() and gitblob685(p.read_bytes())==BLOBS[n]:b=p.read_bytes()
 else:
  with urllib.request.urlopen(urllib.request.Request(URL+n,headers={'User-Agent':'AKBASCORE-TEST685'}),timeout=180) as f:b=f.read()
 require685('LOCKED_'+n,gitblob685(b)==BLOBS[n]);print('SOURCE',n,'BLOB',gitblob685(b),'SHA256',hashlib.sha256(b).hexdigest(),flush=True)
 return b.decode('utf-8')
s566=fetch685('566.py');s575=fetch685('575.py');s677=fetch685('677.py')
require685('TEST575_BASELINE_LINK','BASE_BLOB="'+BLOBS['566.py']+'"' in s575)
BOUND='# TEST566 ADDITIONS — original TEST564 checkpoint/init/append/answer remain unchanged.'
require685('TEST566_ENGINE_BOUNDARY',s566.count(BOUND)==1)
# Exact previously verified Qwen 566 engine: tokenizer/model/checkpoint/init/append/answer.
exec(compile(s566.split(BOUND,1)[0],'TEST566_LOCKED_QWEN_ENGINE.py','exec'),globals())
require685('QWEN_ARCH',(NL,H,QH,KVH,HD)==(28,3584,28,4,128))
require685('QWEN_566_PANEL_SHA',PANEL_SHA==EXPECTED_PANEL_SHA)
require685('QWEN_FROZEN',all(not p.requires_grad for p in model.parameters()) and sentinel()==S0)
# TEST677 exact original relational blind-case constructors, question texts, and strict grading.
tree=ast.parse(s677)
for name in ('token677','make_case677','present677','questions677','gold677','norm677','unknown677','grade677'):
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
 require685('TEST677_FUNCTION_'+name,len(nodes)==1)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),'TEST677_LOCKED_FUNCTIONS.py','exec'),globals())
# TEST566's own 96-cartridge calibration panel is complete; new panel has separately named cartridges.
SEED677=6772026;N_BLIND677=3;rng677=random.Random(SEED677);used677=set();CASES685=[]
# TEST677 token677/make_case677 require these exact module globals.
# Match the successfully executed TEST684 blind-panel RNG initialization.
require685('TEST677_RNG_INITIALIZED',isinstance(rng677,random.Random) and isinstance(used677,set))
for i in range(N_BLIND677):
 CASES685.append(make_case677(f'BLIND_{i:02d}',token677('Actor'),token677('Arch'),token677('Crystal'),token677('Energy'),token677('Effect')))
require685('BLIND_CASE_COUNT',len(CASES685)==3 and len({c['id'] for c in CASES685})==3)
CAR685=[]
def register685(f):
 ids=tok(source(f),add_special_tokens=False).input_ids
 if ids[:P]!=PIDS:raise RuntimeError('QWEN_CANONICAL_PREFIX')
 ci=len(CAR685);CAR685.append(dict(fact=f,body=ids[P:]));return ci
for c in CASES685:
 c['cartridges685']={k:[register685(f) for f in c['facts'][k]] for k in ('A','B','C')}
 c['cf_ids685']=[register685(f) for f in c['counterfactual']]
CONDITIONS685=('NONE','A','B','C','AB','AC','BC','ABC','ABC_CF','BC_BOTH','ABC_BOTH','BC_BOTH_REV','ABC_BOTH_REV')
QIDS685=('CHAIN_AB','CHAIN_BC','CHAIN_ABC');REPS685=('QWEN_TEXT_JOINT','QWEN_DC3_INCREMENTAL')
def keys685(c,cond):
 d=c['cartridges685']
 if cond=='NONE':return []
 if cond=='ABC_CF':return d['A']+d['B']+c['cf_ids685']
 if cond in ('BC_BOTH','ABC_BOTH','BC_BOTH_REV','ABC_BOTH_REV'):
  return (d['A'] if cond.startswith('ABC') else [])+d['B']+(c['cf_ids685']+d['C'] if cond.endswith('_REV') else d['C']+c['cf_ids685'])
 return [ci for k in cond for ci in d[k]]
# The verified TEST566 engine reads the global CAR array. Only the *data* table is replaced;
# engine functions, positional logic, and numerical memory operations are untouched.
CAR=CAR685;CUT=3;MAX_NEW=32
@torch.no_grad()
def memory685(keys,rep):
 if not keys or rep=='QWEN_TEXT_JOINT':
  ids=list(PIDS)
  for ci in keys:ids+=CAR[ci]['body']
  n=len(ids);pos=torch.arange(n,device=DEVICE)
  out=model(input_ids=torch.tensor([ids],device=DEVICE),attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
  return clone(out.past_key_values)
 mem=init(keys[0],CUT)
 for ci in keys[1:]:
  oldn=mem[0][0].shape[-2];grown=append(mem,ci,CUT)
  for li,((ok,ov),(nk,nv)) in enumerate(zip(mem,grown)):
   if not torch.equal(ok,nk[:,:,:oldn,:]) or not torch.equal(ov,nv[:,:,:oldn,:]):raise RuntimeError('APPEND_BITWISE_FAIL_L'+str(li))
  mem=grown
 return mem
# Strict scorer remains the original TEST677 scorer; auxiliary head audit never influences generation.
ALIASES685={'reversal of time flow':['time flow reversal','reversed time flow','reversing time flow','time reversal']}
def strict685(ans,gold):return grade677(ans,gold,ALIASES685.get(gold,[]))
def audit685(ans,gold):
 s=str(ans).strip();unknown=bool(re.match(r'^(?:UNKNOWN|UNDETERMINED|CANNOT DETERMINE|NOT ENOUGH INFORMATION|INSUFFICIENT INFORMATION)\b',s,re.I))
 if gold is None:return 'ABSTAIN_EXACT' if s.upper()=='UNKNOWN' else 'ABSTAIN_EXPLAINED' if unknown else 'MIXED_UNCERTAINTY_REVIEW' if re.search(r'\bUNKNOWN\b',s,re.I) else 'UNSUPPORTED_CLAIM'
 if unknown:return 'MISSED_KNOWN'
 head=re.split(r'\s*\(|[;\n]',s,1)[0].strip(' .,:;').casefold()
 return 'CORRECT_EXACT_HEAD' if head==str(gold).casefold() else 'REVIEW_OR_WRONG'
def presence685(ans,old,new):
 a=str(ans).casefold();o=str(old).casefold() in a;n=str(new).casefold() in a
 return 'BOTH' if o and n else 'OLD_ONLY' if o else 'NEW_ONLY' if n else 'NEITHER'
print('TEST685 PREFLIGHT — QWEN CUT3 NUMERICAL MEMORY',flush=True)
try:
 k=keys685(CASES685[0],'ABC');mem=memory685(k,'QWEN_DC3_INCREMENTAL')
 require685('QWEN_DC3_CACHE_SHAPE',len(mem)==NL and all(x.shape[1]==KVH and x.shape[3]==HD for x,v in mem))
 smoke=answer(mem,questions677(CASES685[0])['CHAIN_AB'])
 print('QWEN_DC3_SMOKE',repr(smoke),flush=True)
 del mem;gc.collect();torch.cuda.empty_cache()
except Exception as e:
 print('TEST685_PREFLIGHT_FAILURE',repr(e),'\n',traceback.format_exc(),flush=True);raise
ROWS685=[];ERRORS685=[]
print('TEST685 PANEL — 3 BLIND × 13 CONDITIONS × 2 REPRESENTATIONS × 3 QUESTIONS = 234 ANSWERS',flush=True)
for c in CASES685:
 qs=questions677(c)
 for cond in CONDITIONS685:
  keys=keys685(c,cond)
  gold=gold677(c,'ABC' if cond.startswith('ABC_BOTH') else 'BC' if cond.startswith('BC_BOTH') else cond)
  if '_BOTH' in cond:
   gold=dict(gold);gold['CHAIN_BC']=None;gold['CHAIN_ABC']=None
  for rep in REPS685:
   mem=None
   try:
    mem=memory685(keys,rep)
    for qid in QIDS685:
     ans=answer(mem,qs[qid]);expected=gold.get(qid)
     ROWS685.append(dict(case=c['id'],condition=cond,representation=rep,question_id=qid,question=qs[qid],expected=expected,answer=ans,strict=strict685(ans,expected),audit=audit685(ans,expected)))
   except Exception as e:
    ERRORS685.append(dict(case=c['id'],condition=cond,representation=rep,error=repr(e),traceback=traceback.format_exc()))
    print('TEST685_FATAL',json.dumps(ERRORS685[-1],ensure_ascii=False),flush=True)
    (ROOT/'TEST685_PROGRESS.json').write_text(json.dumps(dict(rows=ROWS685,errors=ERRORS685),ensure_ascii=False),encoding='utf-8');raise
   finally:
    del mem;gc.collect();torch.cuda.empty_cache()
  print('DONE',c['id'],cond,'ROWS',len(ROWS685),'ERRORS',len(ERRORS685),flush=True)
  (ROOT/'TEST685_PROGRESS.json').write_text(json.dumps(dict(rows=ROWS685,errors=ERRORS685),ensure_ascii=False),encoding='utf-8')
require685('EXPECTED_234_ROWS',len(ROWS685)==234)
require685('ZERO_ERRORS',not ERRORS685)
require685('UNIQUE_ROWS',len({(r['case'],r['condition'],r['representation'],r['question_id']) for r in ROWS685})==234)
require685('MODEL_STILL_FROZEN',all(not p.requires_grad for p in model.parameters()) and sentinel()==S0)
INDEX685={(r['case'],r['condition'],r['representation'],r['question_id']):r for r in ROWS685}
SUMMARY685={};ANATOMY685={};ORDER685={};AB685={}
for rep in REPS685:
 rr=[r for r in ROWS685 if r['representation']==rep]
 SUMMARY685[rep]=dict(strict=dict(Counter(r['strict'] for r in rr)),audit=dict(Counter(r['audit'] for r in rr)),known_total=sum(r['expected'] is not None for r in rr),unknown_total=sum(r['expected'] is None for r in rr))
 anatomy=[];order=[];stability=[]
 for c in CASES685:
  cid=c['id'];old=gold677(c,'ABC')['CHAIN_BC'];new=gold677(c,'ABC_CF')['CHAIN_BC']
  require685('DISTINCT_CF_'+cid,old!=new) if rep==REPS685[0] else None
  for cond in ('BC_BOTH','ABC_BOTH','BC_BOTH_REV','ABC_BOTH_REV'):
   for qid in ('CHAIN_BC','CHAIN_ABC'):
    r=INDEX685[(cid,cond,rep,qid)];anatomy.append(dict(case=cid,condition=cond,question_id=qid,presence=presence685(r['answer'],old,new),answer=r['answer']))
  for cond in ('ABC_BOTH','ABC_BOTH_REV'):
   base=INDEX685[(cid,'ABC',rep,'CHAIN_AB')];r=INDEX685[(cid,cond,rep,'CHAIN_AB')]
   stability.append(dict(case=cid,condition=cond,correct=r['strict']=='CORRECT_KNOWN',unchanged=base['answer']==r['answer'],baseline=base['answer'],answer=r['answer']))
  for a,b in (('BC_BOTH','BC_BOTH_REV'),('ABC_BOTH','ABC_BOTH_REV')):
   for qid in ('CHAIN_BC','CHAIN_ABC'):
    x=INDEX685[(cid,a,rep,qid)];y=INDEX685[(cid,b,rep,qid)]
    order.append(dict(case=cid,question_id=qid,forward=a,reverse=b,changed=presence685(x['answer'],old,new)!=presence685(y['answer'],old,new),raw_changed=x['answer']!=y['answer']))
 ANATOMY685[rep]=anatomy;ORDER685[rep]=order;AB685[rep]=stability
 print('TEST685',rep,json.dumps(dict(summary=SUMMARY685[rep],anatomy=dict(Counter(r['presence'] for r in anatomy)),ab_correct=sum(r['correct'] for r in stability),ab_total=len(stability),ab_unchanged=sum(r['unchanged'] for r in stability),order_changed=sum(r['changed'] for r in order)),ensure_ascii=False),flush=True)
for c in CASES685:
 for cond in ('ABC_BOTH','ABC_BOTH_REV'):
  x=INDEX685[(c['id'],cond,'QWEN_DC3_INCREMENTAL','CHAIN_AB')]
  if x['strict']!='CORRECT_KNOWN':print('AB_FAILURE',c['id'],cond,repr(x['answer']),flush=True)
FINAL685=dict(test='685',source_blobs=BLOBS,model=MODEL_ID,cut=CUT,rows=ROWS685,errors=ERRORS685,summary=SUMMARY685,anatomy=ANATOMY685,order=ORDER685,ab=AB685,seconds=round(time.perf_counter()-START685,2),limits='Original Qwen 566 engine, same TEST677 facts/questions as Mistral TEST684; distinct tokenizer/template and CUT3 vs CUT6. No claim of equal numerical representations; heuristic audit not ground truth.')
(ROOT/'TEST685_FINAL.json').write_text(json.dumps(FINAL685,ensure_ascii=False,indent=2),encoding='utf-8')
print('TEST685_FINAL',json.dumps(dict(rows=len(ROWS685),errors=len(ERRORS685),seconds=FINAL685['seconds'],source_locked=True,cut3='VERIFIED_SOURCE_EXECUTED'),ensure_ascii=False),flush=True)
print('TEST685 EXPERIMENT_COMPLETE',flush=True)
