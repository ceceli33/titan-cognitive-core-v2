# TEST661 — AKBASCORE MAM · KV INTERFERENCE ISOLATION · LOCKED TEST566
# Single Colab cell; frozen Qwen, CUT3, original TEST566 checkpoint/init/append/answer untouched.
# Native CUDA copies aligned K/V bodies into persistent cache; upper layers use verified DC3 continuation.
# Demonstrates KV connectivity/parity, NOT autonomous memory search or full C++ inference.
import os,sys,re,json,time,random,hashlib,urllib.request,traceback,gc,shutil,subprocess,importlib.util
from pathlib import Path
import torch
TEST='661';SEED659=6502026;N659=8;CUT659=3;MAX_NEW659=16
ROOT659=Path('/content/AKBASCORE_TEST661');ROOT659.mkdir(parents=True,exist_ok=True)
URL659='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/566.py'
BLOB659='b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71'
BOUND659='# TEST566 ADDITIONS — original TEST564 checkpoint/init/append/answer remain unchanged.'
def blob659(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def gate659(n,v):print(('PASS' if v else 'FAIL'),n,flush=True);assert v,n
print('='*116+'\nTEST661 — KV INTERFERENCE ISOLATION / SOURCE-LOCKED QWEN\n'+'='*116)
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
native659=load_inline(name='akbascore_kvfabric661_v1',cpp_sources=[CPP659],cuda_sources=[CUDA659],functions=None,extra_cflags=['-O2'],extra_cuda_cflags=['-O2'],with_cuda=True,verbose=True)
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


# TEST661 — FIXED TEST660 PANEL; 0..3 competing full chains; native CUT3 versus independent KV control.
# TARGET_ONLY is an isolated target-only oracle; INDEP_ALL is independent-KV construction, NOT selective attention isolation.
T661=time.perf_counter();R661=[];ERR661=[];rng661=random.Random(6602026);used661=set(seen)
def name661():
 while True:
  s=''.join(rng661.choice(alphabet) for _ in range(9)).capitalize()
  if s not in used661:used661.add(s);return s
NCASE661=4;NCHAIN661=4;PANEL661=[]
for case in range(NCASE661):
 chains=[]
 for j in range(NCHAIN661):
  o,p,g,c,r=[name661() for _ in range(5)]
  facts=[f'The owner of artifact {o} is {p}.',f'{p} works at organization {g}.',f'Organization {g} is located in city {c}.',f'City {c} belongs to region {r}.']
  chains.append(dict(obj=o,person=p,org=g,city=c,region=r,facts=facts,ids=[add659(f) for f in facts]))
 target=case%NCHAIN661;z=chains[target];q=f'In which region is the organization where the owner of artifact {z["obj"]} works located? Answer with the region name only. If the available facts do not determine it, answer UNKNOWN.'
 alt=name661();cf_id=add659(f'City {z["city"]} belongs to region {alt}.')
 PANEL661.append(dict(case=case,target=target,chains=chains,q=q,cf=cf_id,alt=alt))
print('MULTICHAIN_PANEL_SHA',hashlib.sha256(json.dumps(PANEL661,sort_keys=True).encode()).hexdigest(),flush=True)
def grade661(ans,gold,regions):
 strict=exact659(ans,gold)
 hits=[r for r in regions if re.search(r'(?<![A-Za-z])'+re.escape(r)+r'(?![A-Za-z])',ans,re.I)]
 return dict(strict=bool(strict),region=bool(len(hits)==1 and hits[0].casefold()==gold.casefold()),unknown=bool(re.search(r'\b(?:UNKNOWN|UNDETERMINED|CANNOT DETERMINE|NOT ENOUGH INFORMATION)\b',ans,re.I)))
@torch.no_grad()
def run661(keys,q,method):
 if method=='NATIVE':mem=native_memory659(keys)
 elif method=='INDEP':mem=independent659(keys)
 else:raise ValueError(method)
 return answer(validate659(mem,keys),q)
def record661(case,arm,keys,q,gold,regions,method,nd):
 try:
  t=time.perf_counter();a=run661(keys,q,method);elapsed=time.perf_counter()-t
  row=dict(case=case,arm=arm,method=method,distractors=nd,n_cartridges=len(keys),answer=a,gold=gold,elapsed_s=round(elapsed,4),error=None,**grade661(a,gold,regions))
 except Exception as e:
  row=dict(case=case,arm=arm,method=method,distractors=nd,n_cartridges=len(keys),answer='',gold=gold,elapsed_s=None,error=f'{type(e).__name__}: {e}',strict=False,region=False,unknown=False);ERR661.append(row)
 R661.append(row);print('CASE',case,arm,'METHOD',method,'DISTRACTORS',nd,'STRICT',int(row['strict']),'REGION',int(row['region']),'UNKNOWN',int(row['unknown']),'ANSWER',repr(row['answer']),'ERROR',row['error'] or '',flush=True)
for panel in PANEL661:
 case=panel['case'];chains=panel['chains'];z=chains[panel['target']];q=panel['q'];gold=z['region'];regions=[x['region'] for x in chains]+[panel['alt']]
 # Same full-panel shuffle as TEST660: keeps full 16-fact order exactly comparable.
 allfacts=[(j,k,ci) for j,ch in enumerate(chains) for k,ci in enumerate(ch['ids'])];rng661.shuffle(allfacts)
 distractor_ids=[j for j in range(NCHAIN661) if j!=panel['target']]
 record661(case,'TARGET_ONLY',z['ids'],q,gold,regions,'NATIVE',0)
 # Native versus independent-cache readout, with target + 1/2/3 competing complete chains.
 # 3-distractor native uses identical order as TEST660 ORACLE_ALL.
 for nd in (1,2,3):
  chosen=set(distractor_ids[:nd])|{panel['target']}
  keys=[ci for j,k,ci in allfacts if j in chosen]
  record661(case,f'NATIVE_D{nd}',keys,q,gold,regions,'NATIVE',nd)
  record661(case,f'INDEP_D{nd}',keys,q,gold,regions,'INDEP',nd)
 # Matched full-bank order reversal checks physical order effects.
 full=[ci for j,k,ci in allfacts]
 record661(case,'NATIVE_REVERSE_D3',full[::-1],q,gold,regions,'NATIVE',3)
 record661(case,'INDEP_REVERSE_D3',full[::-1],q,gold,regions,'INDEP',3)
 gc.collect();torch.cuda.empty_cache()
ARMS661=['TARGET_ONLY']+[x for n in (1,2,3) for x in (f'NATIVE_D{n}',f'INDEP_D{n}')]+['NATIVE_REVERSE_D3','INDEP_REVERSE_D3']
SUMMARY661={}
for arm in ARMS661:
 rs=[r for r in R661 if r['arm']==arm]
 SUMMARY661[arm]=dict(n=len(rs),strict=sum(r['strict'] for r in rs),region=sum(r['region'] for r in rs),unknown=sum(r['unknown'] for r in rs),mean_seconds=round(sum(r['elapsed_s'] for r in rs if r['elapsed_s'] is not None)/max(1,sum(r['elapsed_s'] is not None for r in rs)),4))
report661=dict(test='661',model='Qwen/Qwen2.5-7B-Instruct',base_blob=BLOB659,cut=CUT659,source_panel_sha=panel_sha,panel_sha=hashlib.sha256(json.dumps(PANEL661,sort_keys=True).encode()).hexdigest(),panel=PANEL661,rows=R661,summary=SUMMARY661,errors=ERR661,weights_frozen=all(not p.requires_grad for p in model.parameters()),sentinel_unchanged=sentinel()==S0,seconds=round(time.perf_counter()-T661,2),limitations='Oracle-supplied target/distractor IDs; not autonomous retrieval. INDEP is an independent per-cartridge KV baseline with shared readout, not attention-isolated per-chain inference. Target-only oracle excludes distractors and cannot establish a causal isolation mechanism. 4 synthetic cases; no layer-local synthesis or 10k scaling claim.')
path661=ROOT659/'TEST661_results.json';path661.write_text(json.dumps(report661,ensure_ascii=False,indent=2),encoding='utf-8')
print('='*116+'\nTEST661_FINAL\n'+json.dumps(SUMMARY661,indent=2))
print('WEIGHTS_FROZEN',report661['weights_frozen'],'SENTINEL',report661['sentinel_unchanged'],'ERRORS',len(ERR661))
print('RESULT',path661,'SHA256',hashlib.sha256(path661.read_bytes()).hexdigest())
complete=len(R661)==NCASE661*len(ARMS661) and not ERR661 and report661['weights_frozen'] and report661['sentinel_unchanged']
print('STATUS','EXPERIMENT_COMPLETE' if complete else 'ERRORS_PRESENT')
