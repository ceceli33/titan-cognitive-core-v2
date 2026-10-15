# TEST679 — MISTRAL DC6 / TEST560 LOCKED ENGINE × TEST677 LOCKED PANEL × TEST659 NATIVE BF16 CUDA APPEND
# One Colab A100 cell. No training, no UI, no answer patches. No changes to TEST560 model computation.
import os,sys,ast,re,json,time,random,hashlib,urllib.request,inspect,traceback,gc
from pathlib import Path
ROOT=Path('/content/AKBASCORE_TEST679') if Path('/content').exists() else Path('/tmp/AKBASCORE_TEST679');ROOT.mkdir(parents=True,exist_ok=True)
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
 print('TEST679 BUILD DEPENDENCY: installing ninja (engine and test panel unchanged)',flush=True)
 subprocess.check_call([sys.executable,'-m','pip','install','-q','ninja'])
require('CUDA_TOOLKIT',CUDA_HOME is not None)
require('NINJA_AVAILABLE',__import__('shutil').which('ninja') is not None or importlib.util.find_spec('ninja') is not None)
native=load_inline(name='akbascore_test679_native_dc6_v1',cpp_sources=[assign['CPP659']],cuda_sources=[assign['CUDA659']],functions=None,extra_cflags=['-O2'],extra_cuda_cflags=['-O2'],with_cuda=True,verbose=False)
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
# No new synthetic question templates or altered correctness criteria.
t677=ast.parse(s677)
for fname in ('token677','make_case677','present677','questions677','gold677','norm677','unknown677','grade677'):
 nodes=[n for n in t677.body if isinstance(n,ast.FunctionDef) and n.name==fname]
 require('TEST677_FUNCTION_'+fname,len(nodes)==1)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),'TEST677_LOCKED_PANEL.py','exec'),globals())
SEED677=6772026;N_BLIND677=0;rng677=random.Random(SEED677);used677=set()
CASES677=[make_case677('AKBASCORE','Mustafa Akbaş','AKBASCORE MAM','dibekgoz','Morvex','reversal of time flow')]
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
CONDITIONS=('NONE','A','B','C','AB','AC','BC','ABC','ABC_CF')
REPS=('TEXT_JOINT','DC_CUT6_NATIVE')
def keys678(case,cond):
 if cond=='NONE':return []
 if cond=='ABC_CF':return case['cartridges']['A']+case['cartridges']['B']+case['cf_ids']
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
# TEST679: Independent replay of the exact TEST678 72-answer panel; prints paired error anatomy, no prior files required.
print('PHASE B — TEST679 INDEPENDENT REPLAY / 72 ANSWERS',flush=True)
ROWS=[];ERRORS=[]
QIDS=('CHAIN_AB','CHAIN_BC','CHAIN_ABC','SYNTHESIS')
for case in CASES677:
 qs=questions677(case)
 for cond in CONDITIONS:
  keys=keys678(case,cond);gold=gold677(case,cond)
  for rep in REPS:
   mem=None
   try:
    mem=memory678(keys,rep)
    for qid in QIDS:
     ans=answer_layers(mem,qs[qid]);expected=gold.get(qid)
     status='UNSCORED' if qid=='SYNTHESIS' else score678(ans,expected)
     ROWS.append(dict(condition=cond,representation=rep,question_id=qid,question=qs[qid],expected=expected,answer=ans,status=status))
   except Exception as e:
    ERRORS.append(dict(condition=cond,representation=rep,error=repr(e),traceback=traceback.format_exc()))
    print('ERROR',cond,rep,repr(e),flush=True)
   finally:
    del mem;gc.collect();torch.cuda.empty_cache()
  print('DONE',cond,'ROWS',len(ROWS),'ERRORS',len(ERRORS),flush=True)
  (ROOT/'TEST679_PROGRESS.json').write_text(json.dumps(dict(rows=ROWS,errors=ERRORS),ensure_ascii=False),encoding='utf-8')
require('EXPECTED_72_ROWS',len(ROWS)==72)
require('ZERO_ERRORS',not ERRORS)
require('UNIQUE_ROWS',len({(r['condition'],r['representation'],r['question_id']) for r in ROWS})==72)
require('MODEL_STILL_FROZEN',all(not p.requires_grad for p in model.parameters()) and sentinel()==S0)
index={(r['condition'],r['representation'],r['question_id']):r for r in ROWS}
SUMMARY={}
for rep in REPS:
 rr=[r for r in ROWS if r['representation']==rep and r['question_id']!='SYNTHESIS']
 SUMMARY[rep]=dict(known_correct=sum(r['status']=='CORRECT_KNOWN' for r in rr),known_total=sum(r['expected'] is not None for r in rr),abstain_correct=sum(r['status']=='CORRECT_ABSTENTION' for r in rr),abstain_total=sum(r['expected'] is None for r in rr),unsupported=sum(r['status']=='UNSUPPORTED' for r in rr),wrong=sum(r['status']=='WRONG_KNOWN' for r in rr),missed=sum(r['status']=='MISSED_KNOWN' for r in rr))
print('\nTEST679 COMPACT DIAGNOSIS — RAW ANSWERS (T=TEXT, D=DC6)',flush=True)
for cond in CONDITIONS:
 print('CONDITION',cond,flush=True)
 for qid in QIDS:
  a=index[(cond,'TEXT_JOINT',qid)];b=index[(cond,'DC_CUT6_NATIVE',qid)]
  print(' ',qid,'GOLD=',repr(a['expected']),'T=',a['status'],repr(a['answer']),'D=',b['status'],repr(b['answer']),flush=True)
print('\nTEST679 PAIRED TRANSITIONS',flush=True)
for qid in QIDS[:-1]:
 transitions={}
 for cond in CONDITIONS:
  a=index[(cond,'TEXT_JOINT',qid)];b=index[(cond,'DC_CUT6_NATIVE',qid)]
  k=a['status']+' -> '+b['status'];transitions[k]=transitions.get(k,0)+1
 print(qid,json.dumps(transitions,ensure_ascii=False),flush=True)
print('\nTEST679 SUMMARY',json.dumps(SUMMARY,ensure_ascii=False),flush=True)
FINAL=dict(test='679',source_git_blobs={'560.py':SHA560,'659.py':SHA659,'677.py':SHA677},panel='TEST678 EXACT AKBASCORE 9x2x4',parity=parity,summary=SUMMARY,rows=ROWS,errors=ERRORS,seconds=round(time.perf_counter()-START,2))
(ROOT/'TEST679_FINAL.json').write_text(json.dumps(FINAL,ensure_ascii=False,indent=2),encoding='utf-8')
print('TEST679_FINAL',json.dumps({'rows':len(ROWS),'errors':len(ERRORS),'summary':SUMMARY,'seconds':FINAL['seconds']},ensure_ascii=False),flush=True)
print('TEST679 EXPERIMENT_COMPLETE',flush=True)
