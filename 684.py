# TEST684 — MISTRAL DC6 / TEST560 LOCKED ENGINE × TEST677 LOCKED PANEL × TEST659 NATIVE BF16 CUDA APPEND
# One Colab A100 cell. TEST683 frozen Mistral DC6 panel preserved. Adds frozen Qwen TEXT_JOINT cross-model diagnostic on identical source facts and questions; Qwen DC3 is NOT claimed without its locked engine. No inference or grading patches.
import os,sys,ast,re,json,time,random,hashlib,urllib.request,inspect,traceback,gc
from pathlib import Path
ROOT=Path('/content/AKBASCORE_TEST684') if Path('/content').exists() else Path('/tmp/AKBASCORE_TEST684');ROOT.mkdir(parents=True,exist_ok=True)
REPO='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/'
SHA560='35d95a929a4e043a9af346c67fc103da47a792b5' # git blob
SHA659='5f90660828d4bf0dd126727d038d4b70bc6b856d' # verified from TEST678 first run
SHA677='5b65959423b7bbdd209b654417eb5fa30ceb2626' # verified from TEST678 first run
SEED678=6782026;START=time.perf_counter()
def fetch(name):
 p=Path('/content')/name
 if p.is_file():
  b=p.read_bytes()
  expected={'560.py':SHA560,'659.py':SHA659,'677.py':SHA677}[name]
  if blob(b)!=expected:
   print('LOCAL_SOURCE_MISMATCH; fetching locked repository version:',name,flush=True)
   with urllib.request.urlopen(urllib.request.Request(REPO+name,headers={'User-Agent':'AKBASCORE-TEST678'}),timeout=180) as f:b=f.read()
 else:
  with urllib.request.urlopen(urllib.request.Request(REPO+name,headers={'User-Agent':'AKBASCORE-TEST678'}),timeout=180) as f:b=f.read()
 return b

def blob(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def require(label,ok):
 print(('PASS' if ok else 'FAIL'),label,flush=True)
 if not ok:raise RuntimeError(label)
raw560=fetch('560.py');raw659=fetch('659.py');raw677=fetch('677.py')
require('LOCKED_TEST560_BLOB',blob(raw560)==SHA560)
require('LOCKED_TEST659_BLOB',blob(raw659)==SHA659)
require('LOCKED_TEST677_BLOB',blob(raw677)==SHA677)
for n,b in [('560',raw560),('659',raw659),('677',raw677)]:print('SOURCE',n,'GIT_BLOB',blob(b),'SHA256',hashlib.sha256(b).hexdigest(),flush=True)
s560=raw560.decode();s659=raw659.decode();s677=raw677.decode()
# Execute TEST560 EXACTLY through the point before its experimental panel starts. No engine function rewritten.
BOUND='section(3,"Numerical checkpoint and incremental contract...")'
require('TEST560_ENGINE_BOUNDARY',s560.count(BOUND)==1)
exec(compile(s560.split(BOUND)[0],'TEST560_LOCKED_ENGINE.py','exec'),globals())
require('MISTRAL_ARCH',(NL,H,QH,KVH,HD,CUT)==(32,4096,32,8,128,6))
require('TEST560_PANEL_SHA',PANEL_SHA==EXPECTED_PANEL_SHA)
require('FROZEN',all(not p.requires_grad for p in model.parameters()))
# Import native BF16 append source directly from TEST659, WITHOUT reimplementing the CUDA kernel.
tree659=ast.parse(s659);assign={}
for node in tree659.body:
 if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('CPP659','CUDA659'):
  assign[node.targets[0].id]=ast.literal_eval(node.value)
require('TEST659_NATIVE_SOURCES',set(assign)=={'CPP659','CUDA659'})
require('TEST659_NATIVE_ABI',all(x in assign['CPP659'] for x in ('append_kv','PYBIND11_MODULE')) and 'append_kernel' in assign['CUDA659'])
from torch.utils.cpp_extension import load_inline,CUDA_HOME
import importlib.util,subprocess,sys
if importlib.util.find_spec('ninja') is None and not __import__('shutil').which('ninja'):
 print('TEST684 BUILD DEPENDENCY: installing ninja (engine and test panel unchanged)',flush=True)
 subprocess.check_call([sys.executable,'-m','pip','install','-q','ninja'])
require('CUDA_TOOLKIT',CUDA_HOME is not None)
require('NINJA_AVAILABLE',__import__('shutil').which('ninja') is not None or importlib.util.find_spec('ninja') is not None)
native=load_inline(name='akbascore_test684_native_dc6_v1',cpp_sources=[assign['CPP659']],cuda_sources=[assign['CUDA659']],functions=None,extra_cflags=['-O2'],extra_cuda_cflags=['-O2'],with_cuda=True,verbose=False)
@torch.no_grad()
def native_splice678(a,b):
 a=a.contiguous();b=b.contiguous();out=torch.empty((a.shape[0],a.shape[1],a.shape[2]+b.shape[2],a.shape[3]),device=a.device,dtype=a.dtype)
 native.append_kv(out,a,b);return out
for t0,t1 in [(29,7),(43,31),(1,1)]:
 x=torch.randn((1,KVH,t0,HD),device=DEVICE,dtype=DTYPE);y=torch.randn((1,KVH,t1,HD),device=DEVICE,dtype=DTYPE)
 require(f'NATIVE_BITWISE_{t0}_{t1}',torch.equal(native_splice678(x,y),torch.cat((x,y),dim=2)))
# Preserve TEST560 append function verbatim except replace ONLY its two lower-layer KV concatenations.
# Native source is exactly TEST659; its unused cosine_scores kernel is compiled unchanged.
start=s560.index('def append_incremental(memory,ci):');end=s560.index('@torch.no_grad()\ndef incremental_dc6(keys):',start)
original=s560[start:end]
old='torch.cat((oldk,k),dim=-2),torch.cat((oldv,v),dim=-2)'
require('APPEND_PATCH_EXACTLY_ONCE',original.count(old)==1)
modified=original.replace(old,'native_splice678(oldk,k),native_splice678(oldv,v)')
exec(compile('@torch.no_grad()\n'+modified,'TEST560_NATIVE_APPEND_ONLY.py','exec'),globals())
append_native678=append_incremental
# Keep the original function separately, obtained from the unmodified source, for exact reference comparisons.
exec(compile('@torch.no_grad()\n'+original.replace('def append_incremental(memory,ci):','def append_reference678(memory,ci):',1),'TEST560_REFERENCE_APPEND.py','exec'),globals())
# Extract the exact TEST677 case construction, question, gold AND grading definitions.
# No new synthetic question templates. Conflicting C/C_CF conditions are explicitly marked ambiguous.
t677=ast.parse(s677)
for fname in ('token677','make_case677','present677','questions677','gold677','norm677','unknown677','grade677'):
 nodes=[n for n in t677.body if isinstance(n,ast.FunctionDef) and n.name==fname]
 require('TEST677_FUNCTION_'+fname,len(nodes)==1)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),'TEST677_LOCKED_PANEL.py','exec'),globals())
SEED677=6772026;N_BLIND677=3;rng677=random.Random(SEED677);used677=set()
CASES677=[] # TEST684: blind synthetic cases only; no familiar AKBASCORE names in scored panel
for i in range(N_BLIND677):CASES677.append(make_case677(f'BLIND_{i:02d}',token677('Actor'),token677('Arch'),token677('Crystal'),token677('Energy'),token677('Effect')))
# Test677 original writes one independent source fact per cartridge; TEST560 source() formatting is preserved.
CAR=[]
def register678(fact):
 ids=tok(source_prefix(fact),add_special_tokens=False).input_ids
 if ids[:P]!=PREFIX_IDS:raise RuntimeError('CANONICAL_PREFIX')
 ci=len(CAR);CAR.append(dict(fact=fact,body=ids[P:]));return ci
for case in CASES677:
 case['cartridges']={k:[register678(f) for f in case['facts'][k]] for k in ('A','B','C')}
 case['cf_ids']=[register678(f) for f in case['counterfactual']]
CONDITIONS=('NONE','A','B','C','AB','AC','BC','ABC','ABC_CF','BC_BOTH','ABC_BOTH','BC_BOTH_REV','ABC_BOTH_REV')
REPS=('TEXT_JOINT','DC_CUT6_NATIVE')
def keys678(case,cond):
 if cond=='NONE':return []
 if cond=='ABC_CF':return case['cartridges']['A']+case['cartridges']['B']+case['cf_ids']
 if cond in ('BC_BOTH','ABC_BOTH','BC_BOTH_REV','ABC_BOTH_REV'):
  prefix=case['cartridges']['A'] if cond.startswith('ABC') else []
  c_old=case['cartridges']['C'];c_new=case['cf_ids']
  return prefix+case['cartridges']['B']+(c_new+c_old if cond.endswith('_REV') else c_old+c_new)
 return [ci for c in cond for ci in case['cartridges'][c]]
def memory678(keys,rep):
 if rep=='TEXT_JOINT':
  ids=list(PREFIX_IDS)
  for ci in keys:ids.extend(CAR[ci]['body'])
  n=len(ids);pos=torch.arange(n,device=DEVICE)
  with torch.no_grad():out=model(input_ids=torch.tensor([ids],device=DEVICE),attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
  return clone_layers(out.past_key_values)
 if not keys:return memory678(keys,'TEXT_JOINT')
 mem=init_incremental(keys[0]);app=append_native678 if rep=='DC_CUT6_NATIVE' else append_reference678
 for ci in keys[1:]:
  oldn=mem[0][0].shape[-2];new=app(mem,ci)
  if not all(torch.equal(a,nk[:,:,:oldn,:]) and torch.equal(b,nv[:,:,:oldn,:]) for (a,b),(nk,nv) in zip(mem,new)):
   raise RuntimeError('PRIOR_ROWS_MUTATED')
  mem=new
 return mem

# TEST677 original exact-match grading; do not use substring scoring.
ALIASES677={
 'reversal of time flow':['time flow reversal','reversed time flow','reversing time flow','time reversal'],
 'NO':['No','NOT MODIFIED','FROZEN','UNCHANGED'],
 '3':['layer 3','layer-3','Layer 3']
}
def score678(raw,gold):return grade677(raw,gold,ALIASES677.get(gold,[]))
def memhash678(mem):
 h=hashlib.sha256()
 for k,v in mem:
  for t in (k,v):h.update(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
# Verify native append is bitwise-identical to TEST560 reference on the SAME freshly built cartridges.
# Engine's TEST560 original function is preserved as append_reference678.
print('PHASE A — NATIVE / PYTORCH MEMORY PARITY',flush=True)
parity=[]
for case in CASES677:
 keys=keys678(case,'ABC');a=memory678(keys,'DC_CUT6_NATIVE');b=memory678(keys,'DC_CUT6_REFERENCE')
 ok=all(torch.equal(x,y) and torch.equal(z,w) for (x,z),(y,w) in zip(a,b));parity.append(dict(case=case['id'],bitwise=ok,sha_native=memhash678(a),sha_reference=memhash678(b)))
 require('NATIVE_REFERENCE_PARITY_'+case['id'],ok)
 del a,b;torch.cuda.empty_cache()
# TEST684 — frozen, predeclared two-tier audit. Original TEST677 score is preserved separately.
# No grading code affects inference, cartridge creation, or raw answers.
import unicodedata
QIDS=('CHAIN_AB','CHAIN_BC','CHAIN_ABC') # fixed three relational questions; synthesis excluded because no gold
UNKNOWN_HEAD=re.compile(r"^\s*(?:UNKNOWN|UNDETERMINED|CANNOT DETERMINE|NOT ENOUGH INFORMATION|INSUFFICIENT INFORMATION|I DON['’]T KNOW|N/?A)\b",re.I)
NEGATION=re.compile(r"\b(?:not|never|no|without|cannot|can't|doesn't|didn't|isn't|wasn't)\b",re.I)
# General semantic equivalence: normalized answer head, not arbitrary substring anywhere.
# Only these transformations are allowed: terminal punctuation, short parenthetical clarification,
# case, and the predeclared reversal-of-time-flow lexical alternations.
REVERSE_FORMS=('reversal of time flow','time flow reversal','reversed time flow','reversing time flow','time reversal')
def clean680(s):
 s=unicodedata.normalize('NFKC',str(s)).casefold().strip()
 s=re.sub(r'^[\s"\'`]+|[\s"\'`.,;:!?]+$','',s)
 s=re.sub(r'\s+',' ',s)
 return s
def head680(raw):
 s=clean680(raw)
 # First clause only; parenthetical and after-colon explanatory prose are not taken as a new answer.
 return re.split(r'\s*\(|\s*[;\n]\s*',s,maxsplit=1)[0].strip(' .,:;')
def audit680(raw,gold):
 s=clean680(raw);head=head680(raw)
 if gold is None:
  if UNKNOWN_HEAD.match(s):
   # If the answer starts UNKNOWN, explanatory prose is allowed, but it is not
   # proof that all subsequent text is harmless: record as qualified abstention.
   return 'ABSTAIN_EXPLAINED' if s!=clean680('UNKNOWN') else 'ABSTAIN_EXACT'
  if re.search(r'\b(?:unknown|cannot determine|insufficient information|not enough information)\b',s,re.I):return 'MIXED_UNCERTAINTY_REVIEW'
  return 'UNSUPPORTED_CLAIM'
 if UNKNOWN_HEAD.match(s):return 'MISSED_KNOWN'
 g=clean680(gold)
 if head==g:return 'CORRECT_EXACT_HEAD'
 if g=='reversal of time flow' and head in REVERSE_FORMS:return 'CORRECT_EQUIVALENT_HEAD'
 # Explanatory continuation with a full gold prefix; boundary must be a word boundary.
 if head.startswith(g+' ') and not NEGATION.search(head[len(g):]):return 'CORRECT_WITH_DETAIL'
 if g=='reversal of time flow':
  for form in REVERSE_FORMS:
   if head.startswith(form+' ') and not NEGATION.search(head[len(form):]):return 'CORRECT_WITH_DETAIL'
 return 'REVIEW_OR_WRONG'
print('PHASE B — TEST684 CONFLICT / 3 CASES x 13 CONDITIONS x 2 REPS x 3 QUESTIONS',flush=True)
ROWS=[];ERRORS=[]
for case in CASES677:
 qs=questions677(case)
 for cond in CONDITIONS:
  keys=keys678(case,cond);gold=gold677(case,'ABC' if cond.startswith('ABC_BOTH') else 'BC' if cond.startswith('BC_BOTH') else cond)
  if '_BOTH' in cond:
   gold=dict(gold);gold['CHAIN_BC']=None;gold['CHAIN_ABC']=None # Two incompatible C claims; neither is uniquely entailed
  for rep in REPS:
   mem=None
   try:
    mem=memory678(keys,rep)
    for qid in QIDS:
     ans=answer_layers(mem,qs[qid]);expected=gold.get(qid)
     strict='UNSCORED' if qid=='SYNTHESIS' else score678(ans,expected)
     audit='UNSCORED' if qid=='SYNTHESIS' else audit680(ans,expected)
     ROWS.append(dict(case=case['id'],condition=cond,representation=rep,question_id=qid,question=qs[qid],expected=expected,answer=ans,strict=strict,audit=audit))
   except Exception as e:
    ERRORS.append(dict(case=case['id'],condition=cond,representation=rep,error=repr(e),traceback=traceback.format_exc()))
    print('ERROR',cond,rep,repr(e),flush=True)
   finally:
    del mem;gc.collect();torch.cuda.empty_cache()
  print('DONE',case['id'],cond,'ROWS',len(ROWS),'ERRORS',len(ERRORS),flush=True)
  (ROOT/'TEST684_PROGRESS.json').write_text(json.dumps(dict(rows=ROWS,errors=ERRORS),ensure_ascii=False),encoding='utf-8')
EXPECTED=len(CASES677)*len(CONDITIONS)*len(REPS)*len(QIDS)
require('EXPECTED_ROWS',len(ROWS)==EXPECTED)
require('ZERO_ERRORS',not ERRORS)
require('UNIQUE_ROWS',len({(r['case'],r['condition'],r['representation'],r['question_id']) for r in ROWS})==EXPECTED)
require('MODEL_STILL_FROZEN',all(not p.requires_grad for p in model.parameters()) and sentinel()==S0)
# The following counts are descriptive audit categories, not inflated accuracy claims.
# Contradiction labels are evaluation-only: they never modify the frozen engine or answer generation.
from collections import Counter
SUMMARY={}
for rep in REPS:
 rr=[r for r in ROWS if r['representation']==rep]
 SUMMARY[rep]=dict(strict=dict(Counter(r['strict'] for r in rr)),audit=dict(Counter(r['audit'] for r in rr)),known_total=sum(r['expected'] is not None for r in rr),unknown_total=sum(r['expected'] is None for r in rr))
print('TEST684 COMPACT AUDIT — STRICT vs RULE-BASED',flush=True)
for rep in REPS:print(rep,json.dumps(SUMMARY[rep],ensure_ascii=False),flush=True)
print('TEST684 REVIEW COUNTS (raw answers retained in JSON)',flush=True)
for rep in REPS:
 print(rep,'REVIEW',sum(r['audit'] in ('REVIEW_OR_WRONG','UNSUPPORTED_CLAIM','MIXED_UNCERTAINTY_REVIEW') for r in ROWS if r['representation']==rep),flush=True)
print('TEST684 CONFLICT DIAGNOSTIC',flush=True)
for rep in REPS:
 conflict=[r for r in ROWS if '_BOTH' in r['condition'] and r['representation']==rep]
 print(rep,'CONFLICT',json.dumps(dict(unknown=sum(r['expected'] is None for r in conflict),abstain=sum(r['audit'].startswith('ABSTAIN') for r in conflict),unsupported=sum(r['audit']=='UNSUPPORTED_CLAIM' for r in conflict),mixed=sum(r['audit']=='MIXED_UNCERTAINTY_REVIEW' for r in conflict),known_correct=sum(r['strict']=='CORRECT_KNOWN' for r in conflict)),ensure_ascii=False),flush=True)
# TEST684 READOUT-ONLY anatomy: inspect exact generated strings, not hidden states or answers repaired by a scorer.
# Old and new effect strings come ONLY from locked TEST677 gold definitions. Neither is inferred from model output.
idx={(r['case'],r['condition'],r['representation'],r['question_id']):r for r in ROWS}
def value_presence683(raw,value):
 # Match a complete alphanumeric token, not a substring of another synthetic identifier.
 # If the value is multiword, preserve word boundaries at its two ends.
 if value is None:return False
 v=str(value).strip()
 return bool(v and re.search(r'(?<![\w])'+re.escape(v)+r'(?![\w])',str(raw),re.I))
def anatomy683(raw,old,new):
 has_old=value_presence683(raw,old);has_new=value_presence683(raw,new)
 return 'BOTH' if has_old and has_new else 'OLD_ONLY' if has_old else 'NEW_ONLY' if has_new else 'NEITHER'
ANATOMY=[];STABILITY=[];ORDER=[]
for case in CASES677:
 cid=case['id'];old=gold677(case,'ABC')['CHAIN_BC'];new=gold677(case,'ABC_CF')['CHAIN_BC']
 require('DISTINCT_CF_'+cid,old is not None and new is not None and norm677(str(old))!=norm677(str(new)))
 for rep in REPS:
  for cond in ('BC_BOTH','ABC_BOTH','BC_BOTH_REV','ABC_BOTH_REV'):
   for qid in ('CHAIN_BC','CHAIN_ABC'):
    r=idx[(cid,cond,rep,qid)];ANATOMY.append(dict(case=cid,representation=rep,condition=cond,question_id=qid,old=old,new=new,presence=anatomy683(r['answer'],old,new),answer=r['answer'],audit=r['audit']))
  # Unrelated AB relation is held constant: compare conflict answers to the no-conflict ABC control.
  control=idx[(cid,'ABC',rep,'CHAIN_AB')]
  for cond in ('ABC_BOTH','ABC_BOTH_REV'):
   r=idx[(cid,cond,rep,'CHAIN_AB')]
   STABILITY.append(dict(case=cid,representation=rep,condition=cond,baseline=control['answer'],answer=r['answer'],baseline_strict=control['strict'],strict=r['strict'],exact_answer_unchanged=r['answer']==control['answer'],correct=r['strict']=='CORRECT_KNOWN'))
  # Same set of facts, opposite insertion order. Record raw change, categorical change and old/new choice.
  for a,b in (('BC_BOTH','BC_BOTH_REV'),('ABC_BOTH','ABC_BOTH_REV')):
   for qid in ('CHAIN_BC','CHAIN_ABC'):
    ra=idx[(cid,a,rep,qid)];rb=idx[(cid,b,rep,qid)]
    ca=anatomy683(ra['answer'],old,new);cb=anatomy683(rb['answer'],old,new)
    ORDER.append(dict(case=cid,representation=rep,question_id=qid,forward=a,reverse=b,forward_presence=ca,reverse_presence=cb,raw_changed=ra['answer']!=rb['answer'],presence_changed=ca!=cb,forward_answer=ra['answer'],reverse_answer=rb['answer']))
print('TEST684 CONFLICT VALUE ANATOMY — 48 CONFLICT ANSWERS (old/new/both/neither)',flush=True)
for rep in REPS:
 rr=[r for r in ANATOMY if r['representation']==rep]
 print(rep,'ALL',json.dumps(dict(Counter(r['presence'] for r in rr)),ensure_ascii=False),flush=True)
 for cond in ('BC_BOTH','ABC_BOTH','BC_BOTH_REV','ABC_BOTH_REV'):
  x=[r for r in rr if r['condition']==cond]
  print(' ',cond,json.dumps(dict(Counter(r['presence'] for r in x)),ensure_ascii=False),flush=True)
print('TEST684 ORDER SENSITIVITY — same cartridges, reversed insertion',flush=True)
for rep in REPS:
 rr=[r for r in ORDER if r['representation']==rep]
 print(rep,'pairs',len(rr),'raw_changed',sum(r['raw_changed'] for r in rr),'value_category_changed',sum(r['presence_changed'] for r in rr),flush=True)
 for r in rr:
  if r['presence_changed']:print(' ORDER_CHANGE',r['case'],r['question_id'],r['forward'],r['forward_presence'],'->',r['reverse_presence'],flush=True)
print('TEST684 INDEPENDENT AB STABILITY — ABC baseline vs contradictory C',flush=True)
for rep in REPS:
 rr=[r for r in STABILITY if r['representation']==rep]
 print(rep,'correct',sum(r['correct'] for r in rr),'/',len(rr),'exact_answer_unchanged',sum(r['exact_answer_unchanged'] for r in rr),'/',len(rr),flush=True)
 for r in rr:
  if not r['correct']:print(' AB_FAILURE',r['case'],r['condition'],'BASE',repr(r['baseline'][:180]),'CONFLICT',repr(r['answer'][:180]),flush=True)
print('TEST684 RAW CONFLICT EXAMPLES — review without subjective relabeling',flush=True)
for r in ANATOMY:
 if r['presence'] in ('BOTH','OLD_ONLY','NEW_ONLY'):
  print(' VALUE',r['case'],r['representation'],r['condition'],r['question_id'],r['presence'],repr(r['answer'][:200]),flush=True)
# PHASE C — QWEN TEXT_JOINT CROSS-MODEL CONTROL.
# Qwen CUT3 numerical memory is NOT reconstructed here: locked TEST575 engine is not part of the supplied verified sources.
# The Qwen control is a frozen model's FULL-TEXT joint baseline with exactly the same TEST677 facts, conditions and questions.
# Distinct chat templates and model internals limit cross-model causal comparisons; diagnostic only.
print('PHASE C — QWEN FROZEN TEXT_JOINT CONTROL (NOT DC3)',flush=True)
MISTRAL_FINAL=dict(test='684',source_git_blobs={'560.py':SHA560,'659.py':SHA659,'677.py':SHA677},parity=parity,summary=SUMMARY,anatomy=ANATOMY,order=ORDER,ab_stability=STABILITY,rows=ROWS,errors=ERRORS,note='Mistral engine and TEST683 panel unchanged; Qwen TEXT baseline is an additional cross-model control, NOT Qwen DC3.')
(ROOT/'TEST684_MISTRAL_COMPLETE.json').write_text(json.dumps(MISTRAL_FINAL,ensure_ascii=False,indent=2),encoding='utf-8')
# Release Mistral GPU allocations BEFORE loading Qwen on the same A100.
del native,model
for k in list(globals()):
 if k in ('CAR','CASES677','ROWS','idx','ANATOMY','ORDER','STABILITY','SUMMARY','parity'):continue
 if k in ('mem','out','a','b','x','y','new','old','r','ra','rb','control','case','qs','keys','rr','conflict','t'):globals()[k]=None
gc.collect();torch.cuda.empty_cache()
from transformers import AutoTokenizer,AutoModelForCausalLM
QMODEL='Qwen/Qwen2.5-7B-Instruct'
qtok=AutoTokenizer.from_pretrained(QMODEL,trust_remote_code=False)
qmodel=AutoModelForCausalLM.from_pretrained(QMODEL,torch_dtype=torch.bfloat16,attn_implementation='sdpa',trust_remote_code=False).to('cuda').eval()
for p in qmodel.parameters():p.requires_grad_(False)
require('QWEN_FROZEN',all(not p.requires_grad for p in qmodel.parameters()))
# Same source facts, in the exact condition order used for Mistral. Qwen-native template is used for Qwen.
# This is full-text presentation, not a numerical cartridge representation.
QROWS=[];QERRORS=[]
# Qwen Transformers 5.x compatibility: use an explicit input_ids tensor, and if the
# generation API fails, run equivalent deterministic greedy decoding with uncached
# forward passes. This changes neither model weights nor the source facts/questions.
QWEN_GENERATION_BACKEND=None
@torch.inference_mode()
def qanswer684(facts,question):
 global QWEN_GENERATION_BACKEND
 context='\n'.join(facts) if facts else '(No source facts provided.)'
 messages=[{'role':'user','content':'Use only the source facts below to answer the question. If the facts do not determine a unique answer, respond UNKNOWN.\n\nSOURCE FACTS:\n'+context+'\n\nQUESTION: '+question+'\nANSWER:'}]
 encoded=qtok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors='pt')
 if isinstance(encoded,torch.Tensor):input_ids=encoded.to('cuda')
 elif hasattr(encoded,'keys') and 'input_ids' in encoded:input_ids=encoded['input_ids'].to('cuda')
 else:raise TypeError('Unexpected Qwen chat template output: '+str(type(encoded)))
 if not isinstance(input_ids,torch.Tensor):input_ids=torch.as_tensor(input_ids,device='cuda')
 if input_ids.ndim!=2:raise RuntimeError('QWEN_INPUT_IDS_NOT_2D')
 if input_ids.dtype!=torch.long:input_ids=input_ids.long()
 if QWEN_GENERATION_BACKEND!='UNCACHED_GREEDY':
  try:
   outputs=qmodel.generate(input_ids=input_ids,attention_mask=torch.ones_like(input_ids),max_new_tokens=32,do_sample=False,use_cache=False,pad_token_id=qtok.eos_token_id)
   if not isinstance(outputs,torch.Tensor):outputs=outputs.sequences
   QWEN_GENERATION_BACKEND='GENERATE_NO_CACHE'
   return qtok.decode(outputs[0,input_ids.shape[1]:],skip_special_tokens=True).strip()
  except (TypeError,ValueError,AttributeError,NotImplementedError) as e:
   print('QWEN_GENERATE_COMPATIBILITY_FALLBACK',repr(e),flush=True)
   QWEN_GENERATION_BACKEND='UNCACHED_GREEDY'
 # Reference greedy decoding without a cache, independent of generate() internals.
 ids=input_ids
 stop={qtok.eos_token_id}
 for token in ('<|im_end|>','<|endoftext|>'):
  tid=qtok.convert_tokens_to_ids(token)
  if isinstance(tid,int) and tid>=0 and tid!=qtok.unk_token_id:stop.add(tid)
 produced=[]
 for _ in range(32):
  result=qmodel(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True)
  nxt=int(result.logits[0,-1,:].argmax().item())
  if nxt in stop:break
  produced.append(nxt)
  ids=torch.cat((ids,torch.tensor([[nxt]],device=ids.device,dtype=ids.dtype)),dim=1)
 return qtok.decode(produced,skip_special_tokens=True).strip()
# Smoke test before the scored panel: stop on the real underlying exception, rather
# than collecting 117 identical errors and reporting only QWEN_EXPECTED_ROWS.
try:
 _smoke=qanswer684([], 'What is the unsupported effect?')
 print('PASS QWEN_GENERATION_SMOKE',repr(_smoke[:120]),'BACKEND',QWEN_GENERATION_BACKEND,flush=True)
except Exception as e:
 print('FAIL QWEN_GENERATION_SMOKE',repr(e),flush=True)
 print(traceback.format_exc(),flush=True)
 raise
for case in CASES677:
 qs=questions677(case)
 for cond in CONDITIONS:
  keys=keys678(case,cond)
  facts=[CAR[i]['fact'] for i in keys]
  gold=gold677(case,'ABC' if cond.startswith('ABC_BOTH') else 'BC' if cond.startswith('BC_BOTH') else cond)
  if '_BOTH' in cond:
   gold=dict(gold);gold['CHAIN_BC']=None;gold['CHAIN_ABC']=None
  for qid in QIDS:
   try:
    ans=qanswer684(facts,qs[qid]);expected=gold.get(qid)
    QROWS.append(dict(case=case['id'],condition=cond,representation='QWEN_TEXT_JOINT',question_id=qid,question=qs[qid],expected=expected,answer=ans,strict=score678(ans,expected),audit=audit680(ans,expected)))
   except Exception as e:
    QERRORS.append(dict(case=case['id'],condition=cond,question_id=qid,error=repr(e),traceback=traceback.format_exc()))
    print('QWEN_FATAL_FIRST_ERROR',json.dumps(QERRORS[-1],ensure_ascii=False),flush=True)
    (ROOT/'TEST684_QWEN_PROGRESS.json').write_text(json.dumps(dict(rows=QROWS,errors=QERRORS),ensure_ascii=False),encoding='utf-8')
    raise
  print('QWEN_DONE',case['id'],cond,'ROWS',len(QROWS),'ERRORS',len(QERRORS),flush=True)
  (ROOT/'TEST684_QWEN_PROGRESS.json').write_text(json.dumps(dict(rows=QROWS,errors=QERRORS),ensure_ascii=False),encoding='utf-8')
require('QWEN_EXPECTED_ROWS',len(QROWS)==len(CASES677)*len(CONDITIONS)*len(QIDS))
require('QWEN_ZERO_ERRORS',not QERRORS)
require('QWEN_UNIQUE_ROWS',len({(r['case'],r['condition'],r['question_id']) for r in QROWS})==len(QROWS))
require('QWEN_STILL_FROZEN',all(not p.requires_grad for p in qmodel.parameters()))
QSUMMARY=dict(strict=dict(Counter(r['strict'] for r in QROWS)),audit=dict(Counter(r['audit'] for r in QROWS)),known_total=sum(r['expected'] is not None for r in QROWS),unknown_total=sum(r['expected'] is None for r in QROWS))
QIDX={(r['case'],r['condition'],r['question_id']):r for r in QROWS}
QANATOMY=[];QORDER=[];QSTABILITY=[]
for case in CASES677:
 cid=case['id'];old=gold677(case,'ABC')['CHAIN_BC'];new=gold677(case,'ABC_CF')['CHAIN_BC']
 for cond in ('BC_BOTH','ABC_BOTH','BC_BOTH_REV','ABC_BOTH_REV'):
  for qid in ('CHAIN_BC','CHAIN_ABC'):
   r=QIDX[(cid,cond,qid)];QANATOMY.append(dict(case=cid,condition=cond,question_id=qid,presence=anatomy683(r['answer'],old,new),answer=r['answer']))
 for cond in ('ABC_BOTH','ABC_BOTH_REV'):
  base=QIDX[(cid,'ABC','CHAIN_AB')];r=QIDX[(cid,cond,'CHAIN_AB')]
  QSTABILITY.append(dict(case=cid,condition=cond,baseline=base['answer'],answer=r['answer'],correct=r['strict']=='CORRECT_KNOWN',exact_answer_unchanged=base['answer']==r['answer']))
 for a,b in (('BC_BOTH','BC_BOTH_REV'),('ABC_BOTH','ABC_BOTH_REV')):
  for qid in ('CHAIN_BC','CHAIN_ABC'):
   ra=QIDX[(cid,a,qid)];rb=QIDX[(cid,b,qid)]
   ca=anatomy683(ra['answer'],old,new);cb=anatomy683(rb['answer'],old,new)
   QORDER.append(dict(case=cid,question_id=qid,forward=a,reverse=b,forward_presence=ca,reverse_presence=cb,raw_changed=ra['answer']!=rb['answer'],presence_changed=ca!=cb))
print('TEST684 CROSS_MODEL SUMMARY',flush=True)
for rep in REPS:
 print(rep,json.dumps(dict(summary=SUMMARY[rep],anatomy=dict(Counter(r['presence'] for r in ANATOMY if r['representation']==rep)),ab_correct=sum(r['correct'] for r in STABILITY if r['representation']==rep),ab_total=sum(r['representation']==rep for r in STABILITY),order_changed=sum(r['presence_changed'] for r in ORDER if r['representation']==rep)),ensure_ascii=False),flush=True)
print('QWEN_TEXT_JOINT',json.dumps(dict(summary=QSUMMARY,anatomy=dict(Counter(r['presence'] for r in QANATOMY)),ab_correct=sum(r['correct'] for r in QSTABILITY),ab_total=len(QSTABILITY),order_changed=sum(r['presence_changed'] for r in QORDER)),ensure_ascii=False),flush=True)
print('TEST684 QWEN_GENERATION_BACKEND',QWEN_GENERATION_BACKEND,flush=True)
print('TEST684 QWEN_DC3_STATUS NOT_TESTED — verified Qwen CUT3 engine source not supplied; no substitute engine invented.',flush=True)
FINAL=dict(test='684',source_git_blobs={'560.py':SHA560,'659.py':SHA659,'677.py':SHA677},mistral=MISTRAL_FINAL,qwen=dict(model=QMODEL,representation='TEXT_JOINT_ONLY',summary=QSUMMARY,anatomy=QANATOMY,order=QORDER,ab_stability=QSTABILITY,rows=QROWS,errors=QERRORS),seconds=round(time.perf_counter()-START,2),limitation='Qwen DC3 NOT tested. Qwen text prompt uses its own chat template; cross-model results are descriptive, not matched DC3 engine parity.')
(ROOT/'TEST684_FINAL.json').write_text(json.dumps(FINAL,ensure_ascii=False,indent=2),encoding='utf-8')
print('TEST684_FINAL',json.dumps(dict(mistral_rows=len(ROWS),qwen_text_rows=len(QROWS),errors=len(ERRORS)+len(QERRORS),qwen_dc3='NOT_TESTED',seconds=FINAL['seconds']),ensure_ascii=False),flush=True)
print('TEST684 EXPERIMENT_COMPLETE',flush=True)
