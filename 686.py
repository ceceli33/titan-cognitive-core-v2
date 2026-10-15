# TEST686 — MISTRAL CUT6 / QWEN SILICON HIPPOCAMPUS v1.4 EXACT SYNTHESIS PANEL
# Single Colab A100 cell. No Gradio, no Qwen load, no charts, no engine or answer patches.
import ast,os,sys,re,json,time,hashlib,urllib.request,gc,traceback
from pathlib import Path
from collections import Counter
BASE='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/'
OUT=Path('/content/AKBASCORE_TEST686' if Path('/content').exists() else '/tmp/AKBASCORE_TEST686');OUT.mkdir(parents=True,exist_ok=True)
START686=time.perf_counter()
def get686(name):
 with urllib.request.urlopen(urllib.request.Request(BASE+name,headers={'User-Agent':'AKBASCORE-TEST686'}),timeout=180) as f:return f.read()
def blob686(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def check686(k,ok):
 print(('PASS ' if ok else 'FAIL ')+k,flush=True)
 if not ok:raise RuntimeError(k)
print('TEST686 — MISTRAL CUT6 / HIPPOCAMPUS v1.4 SYNTHESIS PARITY',flush=True)
b684=get686('684.py');p1=get686('HIPPOCAMPUS_1.4_part1.py')
check686('TEST684_LOCKED_BLOB',blob686(b684)=='0af7ae24840bff722587ea05c0045a8516a43289')
s684=b684.decode('utf-8');s1=p1.decode('utf-8')
# Reuse TEST684 model loader, exact TEST560 CUT6 and TEST659 CUDA source/append, TEST677 infrastructure.
# Stop BEFORE its original scored conflict panel. No Qwen model is loaded.
marker='# Verify native append is bitwise-identical to TEST560 reference on the SAME freshly built cartridges.'
check686('TEST684_BOOTSTRAP_BOUNDARY',s684.count(marker)==1)
exec(compile(s684.split(marker,1)[0],'TEST684_VERIFIED_MISTRAL_BOOTSTRAP.py','exec'),globals())
# Read the v1.4 demonstration's original scenario definitions and scoring metadata AS DATA,
# never execute its Qwen engine or Gradio application.
tree=ast.parse(s1)
def assignment686(tree,name):
 nodes=[n for n in tree.body if isinstance(n,(ast.Assign,ast.AnnAssign)) and any(isinstance(t,ast.Name) and t.id==name for t in (n.targets if isinstance(n,ast.Assign) else [n.target]))]
 check686('V14_METADATA_'+name,len(nodes)==1)
 return nodes[0].value
# The scenario dictionary is inside the embedded V14_SOURCE string, not the outer Gradio script.
embedded686=ast.literal_eval(assignment686(tree,'V14_SOURCE'))
inner686=ast.parse(embedded686)
SCENARIOS686=ast.literal_eval(assignment686(inner686,'MAM14_SCENARIOS'))
# Scoring and aliases are outside the embedded engine. SCORING uses only literal data
# and the built-in dict(...) constructor; evaluate only its isolated assignment.
def metadata686(name):
 node=assignment686(tree,name)
 try:return ast.literal_eval(node)
 except (ValueError,TypeError):
  allowed=(ast.Expression,ast.Dict,ast.List,ast.Tuple,ast.Constant,ast.Call,ast.keyword,ast.Load,ast.Name,ast.UnaryOp,ast.USub)
  for n in ast.walk(node):
   if not isinstance(n,allowed):raise RuntimeError('UNSAFE_METADATA_NODE_'+type(n).__name__)
   if isinstance(n,ast.Call) and (not isinstance(n.func,ast.Name) or n.func.id!='dict'):raise RuntimeError('UNSAFE_METADATA_CALL')
   if isinstance(n,ast.Name) and n.id!='dict':raise RuntimeError('UNSAFE_METADATA_NAME')
  return eval(compile(ast.Expression(node),'V14_METADATA.py','eval'),{'__builtins__':{},'dict':dict},{})
SCORING686=metadata686('SCORING');ALIASES686=metadata686('ALIASES')
check686('V14_SCENARIO_NAMES',set(SCENARIOS686)=={'AKBASCORE','MATRIX'})
check686('V14_SCORING_NAMES',set(SCORING686)==set(SCENARIOS686))
check686('V14_NINE_CONDITIONS',ast.literal_eval(assignment686(inner686,'MAM14_CONDITIONS'))==('NONE','A','B','C','AB','AC','BC','ABC','ABC_CF'))
CONDITIONS686=('NONE','A','B','C','AB','AC','BC','ABC','ABC_CF')
REPS686=('DC_CUT6_NATIVE','TEXT_JOINT')
# Extend original TEST684 cartridge array with unmodified v1.4 source facts.
CASES686={}
for name,data in SCENARIOS686.items():
 ids={slot:register678(data[slot]) for slot in ('A','B','C','C_CF')}
 CASES686[name]=ids
 print('CARTRIDGES',name,{k:len(CAR[v]['body']) for k,v in ids.items()},flush=True)
def keys686(scenario,cond):
 ids=CASES686[scenario]
 if cond=='NONE':return []
 if cond=='ABC_CF':return [ids['A'],ids['B'],ids['C_CF']]
 return [ids[x] for x in cond]
def normalize686(x):return re.sub(r'[^a-z0-9]','',str(x).casefold())
def unknown686(x):return bool(re.fullmatch(r"\s*(?:UNKNOWN|UNDETERMINED|CANNOT DETERMINE|NOT ENOUGH INFORMATION|INSUFFICIENT INFORMATION|I DON'T KNOW|NONE|N/?A)[\s.!?]*",str(x),re.I))
def expected686(spec,cond):
 if spec is None:return 'UNSCORED'
 present=set('ABC') if cond=='ABC_CF' else set() if cond=='NONE' else set(cond)
 if not any(set(req)<=present for req in spec['req']):return None
 return spec.get('cf_gold',spec['gold']) if cond=='ABC_CF' else spec['gold']
def grade686(raw,spec,cond):
 if spec is None:return 'UNSCORED',None
 gold=expected686(spec,cond)
 if gold is None:return ('CORRECT_ABSTENTION' if unknown686(raw) else 'UNSUPPORTED'),'UNKNOWN'
 if unknown686(raw):return 'MISSED_KNOWN',gold
 accepted={normalize686(s) for s in [gold]+ALIASES686.get(gold,[])}
 return ('CORRECT_KNOWN' if normalize686(raw) in accepted else 'WRONG_KNOWN'),gold
# Qwen v1.4 fixed external three-step question chain; not autonomous planning.
def staged686(mem,scenario,cond):
 d=SCENARIOS686[scenario];specs=SCORING686[scenario]['hops'];trace=[];previous=None
 for i,template in enumerate(d['hop_prompts']):
  question=template.format(previous=previous) if i else template
  raw=answer_layers(mem,question);status,expected=grade686(raw,specs[i],cond)
  trace.append(dict(hop=i+1,question=question,answer=raw,status=status,expected=expected))
  if unknown686(raw):
   status_final,expected_final=grade686('UNKNOWN',SCORING686[scenario]['chain'],cond)
   return dict(trace=trace,answer='UNKNOWN',status=status_final,expected=expected_final,completed=False)
  previous=raw
 status,expected=grade686(previous,SCORING686[scenario]['chain'],cond)
 return dict(trace=trace,answer=previous,status=status,expected=expected,completed=True)
# Native vs reference PyTorch append parity, and old KV prefix preservation.
for scenario in SCENARIOS686:
 keys=keys686(scenario,'ABC')
 a=memory678(keys,'DC_CUT6_NATIVE');b=memory678(keys,'DC_CUT6_REFERENCE')
 check686('NATIVE_REFERENCE_BITWISE_'+scenario,all(__import__('torch').equal(x,y) and __import__('torch').equal(z,w) for (x,z),(y,w) in zip(a,b)))
 del a,b;gc.collect();torch.cuda.empty_cache()
ROWS686=[];ERRORS686=[];STAGED686=[]
for scenario,data in SCENARIOS686.items():
 for cond in CONDITIONS686:
  for rep in REPS686:
   mem=None
   try:
    mem=memory678(keys686(scenario,cond),rep)
    for qid,question in data['queries']:
     raw=answer_layers(mem,question)
     status,expected=grade686(raw,SCORING686[scenario]['questions'].get(qid),cond)
     row=dict(scenario=scenario,condition=cond,representation=rep,question_id=qid,question=question,answer=raw,status=status,expected=expected)
     ROWS686.append(row)
     print('ANSWER',scenario,cond,rep,qid,status,repr(raw),flush=True)
    staged=staged686(mem,scenario,cond)
    STAGED686.append(dict(scenario=scenario,condition=cond,representation=rep,**staged))
    print('STAGED',scenario,cond,rep,staged['status'],repr(staged['answer']),flush=True)
   except Exception as ex:
    ERRORS686.append(dict(scenario=scenario,condition=cond,representation=rep,error=repr(ex),traceback=traceback.format_exc()))
    print('ERROR',scenario,cond,rep,repr(ex),flush=True)
   finally:
    del mem;gc.collect();torch.cuda.empty_cache()
  (OUT/'TEST686_PROGRESS.json').write_text(json.dumps(dict(rows=ROWS686,staged=STAGED686,errors=ERRORS686),ensure_ascii=False,indent=2),encoding='utf-8')
check686('EXPECTED_198_QUERY_ROWS',len(ROWS686)==sum(len(v['queries']) for v in SCENARIOS686.values())*len(CONDITIONS686)*len(REPS686))
check686('EXPECTED_36_STAGED_CHAINS',len(STAGED686)==len(SCENARIOS686)*len(CONDITIONS686)*len(REPS686))
check686('ZERO_ERRORS',not ERRORS686)
check686('FROZEN_END',all(not p.requires_grad for p in model.parameters()) and sentinel()==S0)
SUMMARY686={}
for rep in REPS686:
 SUMMARY686[rep]=dict(questions=dict(Counter(r['status'] for r in ROWS686 if r['representation']==rep)),chains=dict(Counter(r['status'] for r in STAGED686 if r['representation']==rep)))
 print('SUMMARY',rep,json.dumps(SUMMARY686[rep],ensure_ascii=False),flush=True)
FINAL686=dict(test='686',model='mistralai/Mistral-7B-Instruct-v0.3',source_blobs={'684.py':blob686(b684),'560.py':SHA560,'659.py':SHA659,'677.py':SHA677},protocol_sha256=hashlib.sha256(p1).hexdigest(),conditions=CONDITIONS686,representations=REPS686,summary=SUMMARY686,rows=ROWS686,staged=STAGED686,errors=ERRORS686,seconds=round(time.perf_counter()-START686,2),scope='Qwen v1.4 scenarios/questions/requirements replicated on native Mistral CUT6; no Gradio; fixed externally prompted chain; free text unscored; generation bound inherited from TEST560.')
path=OUT/'TEST686_FINAL.json';path.write_text(json.dumps(FINAL686,ensure_ascii=False,indent=2),encoding='utf-8')
print('TEST686_FINAL',json.dumps(dict(rows=len(ROWS686),staged=len(STAGED686),errors=len(ERRORS686),seconds=FINAL686['seconds'],file=str(path)),ensure_ascii=False),flush=True)
print('TEST686 EXPERIMENT_COMPLETE',flush=True)
