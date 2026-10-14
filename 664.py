# TEST663 — AKBASCORE MAM · TEXT CEILING / DC CAUSAL DEPENDENCE · LOCKED TEST566
# Single Colab cell; frozen Qwen, CUT3, original TEST566 checkpoint/init/append/answer untouched.
# Native CUDA copies aligned K/V bodies into persistent cache; upper layers use verified DC3 continuation.
# Demonstrates KV connectivity/parity, NOT autonomous memory search or full C++ inference.
import os,sys,re,json,time,random,hashlib,urllib.request,traceback,gc,shutil,subprocess,importlib.util
from pathlib import Path
import torch
TEST='664';SEED659=6502026;N659=8;CUT659=3;MAX_NEW659=16
ROOT659=Path('/content/AKBASCORE_TEST664');ROOT659.mkdir(parents=True,exist_ok=True)
URL659='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/566.py'
BLOB659='b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71'
BOUND659='# TEST566 ADDITIONS — original TEST564 checkpoint/init/append/answer remain unchanged.'
def blob659(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def gate659(n,v):print(('PASS' if v else 'FAIL'),n,flush=True);assert v,n
print('='*116+'\nTEST664 — REFLECTIVE RETRY / FEEDBACK CAUSAL AUDIT\n'+'='*116)
gate659('CUDA_AVAILABLE',torch.cuda.is_available())
if importlib.util.find_spec('ninja') is None:subprocess.check_call([sys.executable,'-m','pip','install','-q','ninja'])
from torch.utils.cpp_extension import load_inline,CUDA_HOME
nvcc659=shutil.which('nvcc') or (str(Path(CUDA_HOME)/'bin'/'nvcc') if CUDA_HOME else None)
gate659('NINJA',importlib.util.find_spec('ninja') is not None)
gate659('NVCC',bool(nvcc659 and Path(nvcc659).is_file()))
gate659('CXX',bool(shutil.which('c++') or shutil.which('g++')))
os.environ.setdefault('MAX_JOBS','2')
# Compile native extension BEFORE downloading model weights.
CPP659=r'''
#include <torch/extension.h>
void append_kv_cuda(torch::Tensor dst,torch::Tensor prefix,torch::Tensor body);
void append_kv(torch::Tensor dst,torch::Tensor prefix,torch::Tensor body){
 TORCH_CHECK(dst.is_cuda()&&prefix.is_cuda()&&body.is_cuda(),"CUDA required");
 TORCH_CHECK(dst.scalar_type()==torch::kBFloat16&&prefix.scalar_type()==torch::kBFloat16&&body.scalar_type()==torch::kBFloat16,"BF16 only");
 TORCH_CHECK(dst.is_contiguous()&&prefix.is_contiguous()&&body.is_contiguous(),"contiguous required");
 TORCH_CHECK(dst.dim()==4&&prefix.dim()==4&&body.dim()==4,"[B,KVH,T,HD] required");
 TORCH_CHECK(dst.size(0)==prefix.size(0)&&dst.size(0)==body.size(0)&&dst.size(1)==prefix.size(1)&&dst.size(1)==body.size(1)&&dst.size(3)==prefix.size(3)&&dst.size(3)==body.size(3),"axis mismatch");
 TORCH_CHECK(dst.size(2)==prefix.size(2)+body.size(2),"length mismatch");
 TORCH_CHECK(dst.device()==prefix.device()&&dst.device()==body.device(),"device mismatch");
 append_kv_cuda(dst,prefix,body);
}
void cosine_cuda(torch::Tensor out,torch::Tensor query,torch::Tensor candidates);
void cosine_scores(torch::Tensor out,torch::Tensor query,torch::Tensor candidates){
 TORCH_CHECK(out.is_cuda()&&query.is_cuda()&&candidates.is_cuda(),"CUDA required");
 TORCH_CHECK(out.is_contiguous()&&query.is_contiguous()&&candidates.is_contiguous(),"contiguous required");
 TORCH_CHECK(out.scalar_type()==torch::kFloat32&&query.scalar_type()==torch::kFloat32&&candidates.scalar_type()==torch::kFloat32,"FP32 required");
 TORCH_CHECK(query.dim()==1&&candidates.dim()==2&&out.dim()==1&&candidates.size(1)==query.size(0)&&out.size(0)==candidates.size(0),"score shape mismatch");
 cosine_cuda(out,query,candidates);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("append_kv",&append_kv,"Native BF16 KV append");m.def("cosine_scores",&cosine_scores,"Native latent cosine ranking");}
'''
CUDA659=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
__global__ void append_kernel(__nv_bfloat16* out,const __nv_bfloat16* pre,const __nv_bfloat16* body,
 int B,int H,int T0,int T1,int D){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 long long n=(long long)B*H*(T0+T1)*D;
 if(i>=n)return;
 int d=i%D;long long j=i/D;int t=j%(T0+T1);long long bh=j/(T0+T1);
 long long src=((bh*(t<T0?T0:T1))+(t<T0?t:t-T0))*D+d;
 out[i]=(t<T0)?pre[src]:body[src];
}
void append_kv_cuda(torch::Tensor dst,torch::Tensor prefix,torch::Tensor body){
 int B=dst.size(0),H=dst.size(1),T0=prefix.size(2),T1=body.size(2),D=dst.size(3);
 long long n=dst.numel();
 append_kernel<<<(n+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
 reinterpret_cast<__nv_bfloat16*>(dst.data_ptr<at::BFloat16>()),
 reinterpret_cast<const __nv_bfloat16*>(prefix.data_ptr<at::BFloat16>()),
 reinterpret_cast<const __nv_bfloat16*>(body.data_ptr<at::BFloat16>()),B,H,T0,T1,D);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void cosine_kernel(float* out,const float* q,const float* c,int rows,int dim){
 int r=blockIdx.x;if(r>=rows)return;
 __shared__ float dots[256],qs[256],cs[256];int t=threadIdx.x;float d=0,a=0,b=0;
 for(int j=t;j<dim;j+=blockDim.x){float x=q[j],y=c[(long long)r*dim+j];d+=x*y;a+=x*x;b+=y*y;}
 dots[t]=d;qs[t]=a;cs[t]=b;__syncthreads();
 for(int step=128;step>0;step>>=1){if(t<step){dots[t]+=dots[t+step];qs[t]+=qs[t+step];cs[t]+=cs[t+step];}__syncthreads();}
 if(t==0)out[r]=dots[0]/sqrtf(fmaxf(qs[0]*cs[0],1e-30f));
}
void cosine_cuda(torch::Tensor out,torch::Tensor query,torch::Tensor candidates){
 cosine_kernel<<<candidates.size(0),256,0,at::cuda::getCurrentCUDAStream()>>>(out.data_ptr<float>(),query.data_ptr<float>(),candidates.data_ptr<float>(),candidates.size(0),query.size(0));
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''
print('BUILDING NATIVE KV EXTENSION',flush=True)
native659=load_inline(name='akbascore_kvfabric664_v1',cpp_sources=[CPP659],cuda_sources=[CUDA659],functions=None,extra_cflags=['-O2'],extra_cuda_cflags=['-O2'],with_cuda=True,verbose=True)
@torch.no_grad()
def splice659(a,b):
 a=a.contiguous();b=b.contiguous();out=torch.empty((a.shape[0],a.shape[1],a.shape[2]+b.shape[2],a.shape[3]),device=a.device,dtype=a.dtype)
 native659.append_kv(out,a,b);return out
for t0,t1 in ((29,7),(43,31),(1,1)):
 x=torch.randn((1,4,t0,128),device='cuda',dtype=torch.bfloat16);y=torch.randn((1,4,t1,128),device='cuda',dtype=torch.bfloat16)
 gate659('NATIVE_BITWISE_'+str(t0)+'_'+str(t1),torch.equal(splice659(x,y),torch.cat((x,y),dim=2)))
# Load exactly the same verified baseline as TEST650.
p=Path('/content/566.py')
if p.is_file() and blob659(p.read_bytes())==BLOB659:raw=p.read_bytes();origin='LOCAL_VERIFIED'
else:
 req=urllib.request.Request(URL659,headers={'User-Agent':'AKBASCORE-TEST659'})
 with urllib.request.urlopen(req,timeout=120) as f:raw=f.read()
 origin='GITHUB_VERIFIED'
gate659('TEST566_BLOB',blob659(raw)==BLOB659)
src=raw.decode('utf-8');gate659('TEST566_BOUNDARY',src.count(BOUND659)==1)
exec(compile(src.split(BOUND659,1)[0],'566.py','exec'),globals())
gate659('FROZEN',all(not p.requires_grad for p in model.parameters()))
gate659('ARCH',(NL,H,QH,KVH,HD)==(28,3584,28,4,128))
gate659('PANEL_LOCK',PANEL_SHA=='69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45')
gate659('SENTINEL',sentinel()==S0)
print('BASE',origin,'BLOB',blob659(raw),'GPU',torch.cuda.get_device_name(0),flush=True)
# Same panel generator and random seed as TEST650: identical A/A' and four competing B facts.
rng=random.Random(SEED659);alphabet='ABCDEFGHJKLMNPQRSTUVWXYZ';seen=set()
def word659():
 while True:
  s=''.join(rng.choice(alphabet) for _ in range(9)).capitalize()
  if s not in seen:seen.add(s);return s
PANEL659=[];CAR659=[]
def add659(s):
 ids=tok(source(s),add_special_tokens=False).input_ids
 if ids[:P]!=PIDS:raise RuntimeError('PREFIX_MISMATCH')
 i=len(CAR659);CAR659.append(dict(fact=s,body=ids[P:]));return i
for i in range(N659):
 obj=word659();owners=[word659() for _ in range(4)];regions=[word659() for _ in range(4)]
 a=add659(f'The owner of artifact {obj} is {owners[0]}.')
 acf=add659(f'The owner of artifact {obj} is {owners[1]}.')
 bs=[add659(f'{owners[j]} lives in region {regions[j]}.') for j in range(4)]
 PANEL659.append(dict(case=i,object=obj,owners=owners,regions=regions,a=a,acf=acf,bs=bs,q=f'In which region does the owner of artifact {obj} live?',qa=f'Who owns artifact {obj}?'))
CAR=CAR659;MAX_NEW=MAX_NEW659
panel_sha=hashlib.sha256(json.dumps(PANEL659,sort_keys=True).encode()).hexdigest()
gate659('TEST650_PANEL_LOCK',panel_sha=='db8b211abed216793bf2e4010eb006c862a74202d7512ae53a8a463fe929d466')
@torch.no_grad()
def native_append659(memory,ci):
 # Preserve the locked TEST566 upper-layer computation and H3 injection exactly.
 # Only lower-layer K/V concatenation is replaced with native CUDA, after RoPE position alignment.
 cp=checkpoint(ci,CUT659);oldn=memory[0][0].shape[-2];q=cp['body']
 pos=torch.arange(oldn,oldn+q,device=DEVICE);body=cp['h'][:,P:,:]
 if body.shape!=(1,q,H):raise RuntimeError('H3_BODY_SHAPE')
 def inject(module,args,out):
  current=out[0] if isinstance(out,(tuple,list)) else out
  if current.shape!=body.shape:raise RuntimeError('H3_INJECTION_SHAPE')
  return replace(out,body)
 hook=model.model.layers[CUT659].register_forward_hook(inject)
 try:
  out=model(inputs_embeds=torch.zeros((1,q,H),device=DEVICE,dtype=DTYPE),past_key_values=cache_build(memory),attention_mask=torch.ones((1,oldn+q),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
  grown=list(clone(out.past_key_values))
  for li in range(CUT659+1):
   ok,ov=memory[li];k,v=cp['kv'][li]
   k=rephase(k[:,:,P:,:],cp['pos'][P:],pos);v=v[:,:,P:,:]
   grown[li]=(splice659(ok,k),splice659(ov,v))
  result=tuple(grown)
  for (ok,ov),(nk,nv) in zip(memory,result):
   if not torch.equal(ok,nk[:,:,:oldn,:]) or not torch.equal(ov,nv[:,:,:oldn,:]):raise RuntimeError('PREFIX_MUTATED')
  if len(result)!=NL or any(k.shape!=(1,KVH,oldn+q,HD) or v.shape!=(1,KVH,oldn+q,HD) for k,v in result):raise RuntimeError('CACHE_SHAPE')
  return result
 finally:hook.remove()
@torch.no_grad()
def native_memory659(keys):
 mem=init(keys[0],CUT659)
 for ci in keys[1:]:mem=native_append659(mem,ci)
 return mem
@torch.no_grad()
def baseline_memory659(keys):
 mem=init(keys[0],CUT659)
 for ci in keys[1:]:mem=append(mem,ci,CUT659)
 return mem
@torch.no_grad()
def independent659(keys):
 mem=None
 for ci in keys:
  cp=checkpoint(ci,CUT659);kv=cp['kv']
  if mem is None:mem=kv;continue
  oldn=mem[0][0].shape[-2];q=len(CAR[ci]['body'])
  oldpos=torch.arange(P,P+q,device=DEVICE);newpos=torch.arange(oldn,oldn+q,device=DEVICE)
  mem=tuple((splice659(ok,rephase(k[:,:,P:,:],oldpos,newpos)),splice659(ov,v[:,:,P:,:])) for (ok,ov),(k,v) in zip(mem,kv))
 return mem
@torch.no_grad()
def joint659(keys):
 ids=list(PIDS)
 for ci in keys:ids.extend(CAR[ci]['body'])
 n=len(ids);pos=torch.arange(n,device=DEVICE)
 out=model(input_ids=torch.tensor([ids],device=DEVICE),attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
 return clone(out.past_key_values)
def validate659(mem,keys):
 n=P+sum(len(CAR[i]['body']) for i in keys)
 if len(mem)!=NL or any(k.shape!=(1,KVH,n,HD) or v.shape!=(1,KVH,n,HD) or not torch.isfinite(k).all() or not torch.isfinite(v).all() for k,v in mem):raise RuntimeError('INVALID_CACHE')
 return mem
def exact659(a,b):return re.sub(r'[^a-z]','',a.casefold())==re.sub(r'[^a-z]','',b.casefold())




# TEST664 — Reflective retry against budget-matched controls. All model weights and TEST566 engine untouched.
# DEVELOPMENT PANEL ONLY. A second attempt must be distinguished from use of feedback, not credited automatically.
T664=time.perf_counter();R664=[];ERR664=[]
T663=time.perf_counter();R663=[];ERR663=[];rng663=random.Random(6632026);used663=set(seen)
def name663():
 while True:
  s=''.join(rng663.choice(alphabet) for _ in range(9)).capitalize()
  if s not in used663:used663.add(s);return s
def isname663(a,b):return exact659(a,b)
def unknown663(a):return bool(re.search(r'\b(?:UNKNOWN|UNDETERMINED|CANNOT DETERMINE|NOT ENOUGH INFORMATION)\b',a,re.I))
def prompt663(obj):return f'In which region is the city containing the organization where the owner of artifact {obj} works located? Answer only the region name. If the facts do not determine it, answer UNKNOWN.'
def hop663(i,x):
 return [f'Who owns artifact {x}? Answer only the name. If the facts do not determine it, answer UNKNOWN.',f'Which organization does {x} work at? Answer only the organization name. If the facts do not determine it, answer UNKNOWN.',f'In which city is organization {x} located? Answer only the city name. If the facts do not determine it, answer UNKNOWN.',f'Which region does city {x} belong to? Answer only the region name. If the facts do not determine it, answer UNKNOWN.'][i]
N663=12;K663=4;PANEL663=[]
for case in range(N663):
 chains=[]
 for j in range(K663):
  obj,person,org,city,region=[name663() for _ in range(5)]
  facts=[f'The owner of artifact {obj} is {person}.',f'{person} works at organization {org}.',f'Organization {org} is located in city {city}.',f'City {city} belongs to region {region}.']
  chains.append(dict(obj=obj,person=person,org=org,city=city,region=region,ids=[add659(f) for f in facts]))
 target=rng663.randrange(K663);other=(target+1)%K663;z=chains[target]
 # Change ONLY target's second relation; target owner now works at the other chain's organization.
 # Both terminal regions remain present exactly once. The new answer must switch.
 rewired=add659(f'{z["person"]} works at organization {chains[other]["org"]}.')
 base=[ci for c in chains for ci in c['ids']];rng663.shuffle(base)
 swapped=[rewired if ci==z['ids'][1] else ci for ci in base]
 missing=[ci for ci in base if ci!=z['ids'][1]]
 PANEL663.append(dict(case=case,chains=chains,target=target,other=other,q=prompt663(z['obj']),base=base,swap=swapped,missing=missing,original=z['region'],changed=chains[other]['region'],rewired=rewired))
SHA663=hashlib.sha256(json.dumps(PANEL663,sort_keys=True).encode()).hexdigest();print('TEST664_PANEL_SHA',SHA663,'CASES',N663,flush=True)
# Important: text ceiling uses the exact same source-prefix and body order as the numerical DC arm.
# joint659 is the locked baseline's ordinary full-context transformer forward pass, NOT plaintext fed to answer().

ROOT664=ROOT659;SHA664=SHA663
@torch.no_grad()
def run_hops664(mem,obj,hypothesis=None):
 x=obj;trace=[]
 for hop in range(4):
  q=hop663(hop,x)
  if hypothesis is not None:
   q+='\nAn earlier attempt suggested region '+hypothesis+'. This may be wrong. Independently verify this step from the available facts; do not assume that suggestion is correct.'
  y=answer(mem,q);trace.append(dict(hop=hop,input=x,query=q,answer=y))
  if unknown663(y):return 'UNKNOWN',trace
  x=y
 return x,trace
@torch.no_grad()
def run664(z,condition,rep):
 keys=z[condition];gold=z['original'] if condition=='base' else z['changed'] if condition=='swap' else 'UNKNOWN'
 try:
  t=time.perf_counter();mem=validate659(joint659(keys) if rep=='TEXT_JOINT' else native_memory659(keys),keys)
  direct=answer(mem,z['q']);direct_time=time.perf_counter()-t
  # STEPWISE: first-pass one-hop decomposition; four model calls at most.
  t=time.perf_counter();step,step_trace=run_hops664(mem,z['chains'][z['target']]['obj']);step_time=time.perf_counter()-t
  # BLIND_RETRY: fresh four-hop computation without access to first answer (matched hop budget).
  # In deterministic greedy mode this should replicate STEPWISE. Its value is a negative control.
  t=time.perf_counter();blind,blind_trace=run_hops664(mem,z['chains'][z['target']]['obj']);blind_time=time.perf_counter()-t
  # FEEDBACK_RETRY: same four-hop budget but with a possibly wrong first-pass DIRECT candidate.
  # Candidate is explicitly marked untrusted; no oracle or correct intermediate state supplied.
  t=time.perf_counter();feedback,feedback_trace=run_hops664(mem,z['chains'][z['target']]['obj'],direct);feedback_time=time.perf_counter()-t
  # DECOY_FEEDBACK: identical prompt and budget, with an unrelated competing-chain terminal candidate.
  # Separates useful feedback from mere prompting and anchoring effects.
  decoy=z['chains'][z['other']]['region']
  t=time.perf_counter();decoy_answer,decoy_trace=run_hops664(mem,z['chains'][z['target']]['obj'],decoy);decoy_time=time.perf_counter()-t
  arms=[('DIRECT',direct,[],direct_time),('STEPWISE',step,step_trace,step_time),('BLIND_RETRY',blind,blind_trace,blind_time),('FEEDBACK_RETRY',feedback,feedback_trace,feedback_time),('DECOY_FEEDBACK',decoy_answer,decoy_trace,decoy_time)]
  for arm,a,trace,seconds in arms:
   correct=unknown663(a) if condition=='missing' else isname663(a,gold)
   row=dict(case=z['case'],condition=condition,representation=rep,arm=arm,answer=a,gold=gold,correct=bool(correct),unknown=unknown663(a),trace=trace,first_direct=direct,decoy=decoy,seconds=round(seconds,4),error=None)
   R664.append(row)
   print('CASE',z['case'],condition,rep,arm,'PASS',int(correct),'ANSWER',repr(a),'GOLD',repr(gold),flush=True)
  del mem
 except Exception as e:
  error=f'{type(e).__name__}: {e}';ERR664.append(dict(case=z['case'],condition=condition,representation=rep,error=error));print('ERROR',z['case'],condition,rep,error,flush=True)
for z in PANEL663:
 for rep in ('TEXT_JOINT','DC_CUT3'):
  for condition in ('base','swap','missing'):run664(z,condition,rep)
 gc.collect();torch.cuda.empty_cache()
ARMS664=('DIRECT','STEPWISE','BLIND_RETRY','FEEDBACK_RETRY','DECOY_FEEDBACK');REPS664=('TEXT_JOINT','DC_CUT3')
SUMMARY664={};SWITCH664={};TRANSITIONS664={};TRACE664={}
for rep in REPS664:
 for arm in ARMS664:
  for cond in ('base','swap','missing'):
   rows=[r for r in R664 if (r['representation'],r['arm'],r['condition'])==(rep,arm,cond)]
   SUMMARY664[f'{rep}_{arm}_{cond}']=dict(n=len(rows),correct=sum(r['correct'] for r in rows),unknown=sum(r['unknown'] for r in rows),mean_seconds=round(sum(r['seconds'] for r in rows)/max(1,len(rows)),4))
  paired=[]
  for case in range(N663):
   rr={r['condition']:r for r in R664 if r['case']==case and r['representation']==rep and r['arm']==arm}
   if len(rr)==3:paired.append(bool(rr['base']['correct'] and rr['swap']['correct'] and rr['base']['answer'].casefold()!=rr['swap']['answer'].casefold()))
  SWITCH664[f'{rep}_{arm}']=dict(n=len(paired),causal_switch=sum(paired))
 for cond in ('base','swap','missing'):
  rows={arm:{r['case']:r for r in R664 if (r['representation'],r['condition'],r['arm'])==(rep,cond,arm)} for arm in ARMS664}
  ids=sorted(set.intersection(*(set(v) for v in rows.values())))
  base=rows['STEPWISE'];fb=rows['FEEDBACK_RETRY'];dc=rows['DECOY_FEEDBACK']
  TRANSITIONS664[f'{rep}_{cond}']=dict(n=len(ids),feedback_repairs=sum(not base[i]['correct'] and fb[i]['correct'] for i in ids),feedback_regressions=sum(base[i]['correct'] and not fb[i]['correct'] for i in ids),decoy_repairs=sum(not base[i]['correct'] and dc[i]['correct'] for i in ids),decoy_regressions=sum(base[i]['correct'] and not dc[i]['correct'] for i in ids),feedback_changes=sum(not exact659(base[i]['answer'],fb[i]['answer']) for i in ids),blind_matches_stepwise=sum(exact659(base[i]['answer'],rows['BLIND_RETRY'][i]['answer']) for i in ids))
  for arm in ('STEPWISE','FEEDBACK_RETRY','DECOY_FEEDBACK'):
   trace_rows=[r for r in R664 if (r['representation'],r['condition'],r['arm'])==(rep,cond,arm)]
   TRACE664[f'{rep}_{cond}_{arm}']=dict(n=len(trace_rows),completed_four=sum(len(r['trace'])==4 and not any(unknown663(t['answer']) for t in r['trace']) for r in trace_rows),stopped_early=sum(len(r['trace'])<4 for r in trace_rows))
report664=dict(test='664',base_blob=BLOB659,baseline_panel_sha=panel_sha,panel_sha=SHA664,panel=PANEL663,rows=R664,summary=SUMMARY664,causal_switch=SWITCH664,transitions=TRANSITIONS664,trace_summary=TRACE664,errors=ERR664,weights_frozen=all(not p.requires_grad for p in model.parameters()),sentinel_unchanged=sentinel()==S0,elapsed=round(time.perf_counter()-T664,2),limitations='Development panel reused from TEST663; NOT an independent holdout. Feedback is untrusted DIRECT final answer, not independent verification. Blind retry is deterministic control; same model/prompt. Feedback/deco y arms modify one-hop prompts, so differences can reflect prompting/anchoring, not autonomous self-correction. STEPWISE uses external four-hop question decomposition, not latent single-pass synthesis. n=12 exploratory. All cartridge selection is oracle. Missing-edge UNKNOWN is a behavioral result, not proof of epistemic calibration.')
path664=ROOT664/'TEST664_results.json';path664.write_text(json.dumps(report664,ensure_ascii=False,indent=2),encoding='utf-8')
print('='*116+'\nTEST664_FINAL\n'+json.dumps(SUMMARY664,indent=2))
print('CAUSAL_SWITCH',json.dumps(SWITCH664,indent=2))
print('RETRY_TRANSITIONS',json.dumps(TRANSITIONS664,indent=2))
print('TRACE_AUDIT',json.dumps(TRACE664,indent=2))
print('WEIGHTS_FROZEN',report664['weights_frozen'],'SENTINEL',report664['sentinel_unchanged'],'ERRORS',len(ERR664))
print('RESULT',path664,'SHA256',hashlib.sha256(path664.read_bytes()).hexdigest())
print('STATUS','EXPERIMENT_COMPLETE' if len(R664)==N663*3*2*len(ARMS664) and not ERR664 and report664['weights_frozen'] and report664['sentinel_unchanged'] else 'ERRORS_PRESENT')
