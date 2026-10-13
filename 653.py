# TEST653 — AKBASCORE MAM · NATIVE CUDA KV FABRIC · LOCKED TEST566
# Single Colab cell; frozen Qwen, CUT3, original TEST566 checkpoint/init/append/answer untouched.
# Native CUDA copies aligned K/V bodies into persistent cache; upper layers use verified DC3 continuation.
# Demonstrates KV connectivity/parity, NOT autonomous memory search or full C++ inference.
import os,sys,re,json,time,random,hashlib,urllib.request,traceback,gc,shutil,subprocess,importlib.util
from pathlib import Path
import torch
TEST='653';SEED653=6502026;N653=8;CUT653=3;MAX_NEW653=16
ROOT653=Path('/content/AKBASCORE_TEST653');ROOT653.mkdir(parents=True,exist_ok=True)
URL653='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/566.py'
BLOB653='b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71'
BOUND653='# TEST566 ADDITIONS — original TEST564 checkpoint/init/append/answer remain unchanged.'
def blob653(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def gate653(n,v):print(('PASS' if v else 'FAIL'),n,flush=True);assert v,n
print('='*116+'\nTEST653 — NATIVE CUDA DYNAMIC KV FABRIC / SOURCE-LOCKED QWEN\n'+'='*116)
gate653('CUDA_AVAILABLE',torch.cuda.is_available())
if importlib.util.find_spec('ninja') is None:subprocess.check_call([sys.executable,'-m','pip','install','-q','ninja'])
from torch.utils.cpp_extension import load_inline,CUDA_HOME
nvcc653=shutil.which('nvcc') or (str(Path(CUDA_HOME)/'bin'/'nvcc') if CUDA_HOME else None)
gate653('NINJA',importlib.util.find_spec('ninja') is not None)
gate653('NVCC',bool(nvcc653 and Path(nvcc653).is_file()))
gate653('CXX',bool(shutil.which('c++') or shutil.which('g++')))
os.environ.setdefault('MAX_JOBS','2')
# Compile native extension BEFORE downloading model weights.
CPP653=r'''
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
CUDA653=r'''
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
native653=load_inline(name='akbascore_kvfabric653_v1',cpp_sources=[CPP653],cuda_sources=[CUDA653],functions=None,extra_cflags=['-O2'],extra_cuda_cflags=['-O2'],with_cuda=True,verbose=True)
@torch.no_grad()
def splice653(a,b):
 a=a.contiguous();b=b.contiguous();out=torch.empty((a.shape[0],a.shape[1],a.shape[2]+b.shape[2],a.shape[3]),device=a.device,dtype=a.dtype)
 native653.append_kv(out,a,b);return out
for t0,t1 in ((29,7),(43,31),(1,1)):
 x=torch.randn((1,4,t0,128),device='cuda',dtype=torch.bfloat16);y=torch.randn((1,4,t1,128),device='cuda',dtype=torch.bfloat16)
 gate653('NATIVE_BITWISE_'+str(t0)+'_'+str(t1),torch.equal(splice653(x,y),torch.cat((x,y),dim=2)))
# Load exactly the same verified baseline as TEST650.
p=Path('/content/566.py')
if p.is_file() and blob653(p.read_bytes())==BLOB653:raw=p.read_bytes();origin='LOCAL_VERIFIED'
else:
 req=urllib.request.Request(URL653,headers={'User-Agent':'AKBASCORE-TEST653'})
 with urllib.request.urlopen(req,timeout=120) as f:raw=f.read()
 origin='GITHUB_VERIFIED'
gate653('TEST566_BLOB',blob653(raw)==BLOB653)
src=raw.decode('utf-8');gate653('TEST566_BOUNDARY',src.count(BOUND653)==1)
exec(compile(src.split(BOUND653,1)[0],'566.py','exec'),globals())
gate653('FROZEN',all(not p.requires_grad for p in model.parameters()))
gate653('ARCH',(NL,H,QH,KVH,HD)==(28,3584,28,4,128))
gate653('PANEL_LOCK',PANEL_SHA=='69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45')
gate653('SENTINEL',sentinel()==S0)
print('BASE',origin,'BLOB',blob653(raw),'GPU',torch.cuda.get_device_name(0),flush=True)
# Same panel generator and random seed as TEST650: identical A/A' and four competing B facts.
rng=random.Random(SEED653);alphabet='ABCDEFGHJKLMNPQRSTUVWXYZ';seen=set()
def word653():
 while True:
  s=''.join(rng.choice(alphabet) for _ in range(9)).capitalize()
  if s not in seen:seen.add(s);return s
PANEL653=[];CAR653=[]
def add653(s):
 ids=tok(source(s),add_special_tokens=False).input_ids
 if ids[:P]!=PIDS:raise RuntimeError('PREFIX_MISMATCH')
 i=len(CAR653);CAR653.append(dict(fact=s,body=ids[P:]));return i
for i in range(N653):
 obj=word653();owners=[word653() for _ in range(4)];regions=[word653() for _ in range(4)]
 a=add653(f'The owner of artifact {obj} is {owners[0]}.')
 acf=add653(f'The owner of artifact {obj} is {owners[1]}.')
 bs=[add653(f'{owners[j]} lives in region {regions[j]}.') for j in range(4)]
 PANEL653.append(dict(case=i,object=obj,owners=owners,regions=regions,a=a,acf=acf,bs=bs,q=f'In which region does the owner of artifact {obj} live?',qa=f'Who owns artifact {obj}?'))
CAR=CAR653;MAX_NEW=MAX_NEW653
panel_sha=hashlib.sha256(json.dumps(PANEL653,sort_keys=True).encode()).hexdigest()
gate653('TEST650_PANEL_LOCK',panel_sha=='db8b211abed216793bf2e4010eb006c862a74202d7512ae53a8a463fe929d466')
@torch.no_grad()
def native_append653(memory,ci):
 # Preserve the locked TEST566 upper-layer computation and H3 injection exactly.
 # Only lower-layer K/V concatenation is replaced with native CUDA, after RoPE position alignment.
 cp=checkpoint(ci,CUT653);oldn=memory[0][0].shape[-2];q=cp['body']
 pos=torch.arange(oldn,oldn+q,device=DEVICE);body=cp['h'][:,P:,:]
 if body.shape!=(1,q,H):raise RuntimeError('H3_BODY_SHAPE')
 def inject(module,args,out):
  current=out[0] if isinstance(out,(tuple,list)) else out
  if current.shape!=body.shape:raise RuntimeError('H3_INJECTION_SHAPE')
  return replace(out,body)
 hook=model.model.layers[CUT653].register_forward_hook(inject)
 try:
  out=model(inputs_embeds=torch.zeros((1,q,H),device=DEVICE,dtype=DTYPE),past_key_values=cache_build(memory),attention_mask=torch.ones((1,oldn+q),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
  grown=list(clone(out.past_key_values))
  for li in range(CUT653+1):
   ok,ov=memory[li];k,v=cp['kv'][li]
   k=rephase(k[:,:,P:,:],cp['pos'][P:],pos);v=v[:,:,P:,:]
   grown[li]=(splice653(ok,k),splice653(ov,v))
  result=tuple(grown)
  for (ok,ov),(nk,nv) in zip(memory,result):
   if not torch.equal(ok,nk[:,:,:oldn,:]) or not torch.equal(ov,nv[:,:,:oldn,:]):raise RuntimeError('PREFIX_MUTATED')
  if len(result)!=NL or any(k.shape!=(1,KVH,oldn+q,HD) or v.shape!=(1,KVH,oldn+q,HD) for k,v in result):raise RuntimeError('CACHE_SHAPE')
  return result
 finally:hook.remove()
@torch.no_grad()
def native_memory653(keys):
 mem=init(keys[0],CUT653)
 for ci in keys[1:]:mem=native_append653(mem,ci)
 return mem
@torch.no_grad()
def baseline_memory653(keys):
 mem=init(keys[0],CUT653)
 for ci in keys[1:]:mem=append(mem,ci,CUT653)
 return mem
@torch.no_grad()
def independent653(keys):
 mem=None
 for ci in keys:
  cp=checkpoint(ci,CUT653);kv=cp['kv']
  if mem is None:mem=kv;continue
  oldn=mem[0][0].shape[-2];q=len(CAR[ci]['body'])
  oldpos=torch.arange(P,P+q,device=DEVICE);newpos=torch.arange(oldn,oldn+q,device=DEVICE)
  mem=tuple((splice653(ok,rephase(k[:,:,P:,:],oldpos,newpos)),splice653(ov,v[:,:,P:,:])) for (ok,ov),(k,v) in zip(mem,kv))
 return mem
@torch.no_grad()
def joint653(keys):
 ids=list(PIDS)
 for ci in keys:ids.extend(CAR[ci]['body'])
 n=len(ids);pos=torch.arange(n,device=DEVICE)
 out=model(input_ids=torch.tensor([ids],device=DEVICE),attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
 return clone(out.past_key_values)
def validate653(mem,keys):
 n=P+sum(len(CAR[i]['body']) for i in keys)
 if len(mem)!=NL or any(k.shape!=(1,KVH,n,HD) or v.shape!=(1,KVH,n,HD) or not torch.isfinite(k).all() or not torch.isfinite(v).all() for k,v in mem):raise RuntimeError('INVALID_CACHE')
 return mem
def exact653(a,b):return re.sub(r'[^a-z]','',a.casefold())==re.sub(r'[^a-z]','',b.casefold())

# TEST653: causal candidate deletion / counterfactual swap in verified native DC3 fabric.
# This measures which B memory is necessary for the final answer; it is NOT an autonomous router.
ARMS653=('B_ONLY','NATIVE_AB','NATIVE_CF','BASE_DC_AB','BASE_DC_CF','JOINT_AB','JOINT_CF')
RAW653=[];ABL653=[];T653=time.perf_counter()
@torch.no_grad()
def run653(keys,q):
 return answer(validate653(native_memory653(keys),keys),q)
for z in PANEL653:
 print('CASE',z['case'],'CAUSAL KV ABLATION',flush=True)
 a,acf,bs=z['a'],z['acf'],z['bs'];q=z['q']
 try:
  cases={'B_ONLY':(bs,None),'NATIVE_AB':([a]+bs,None),'NATIVE_CF':([acf]+bs,None),
   'BASE_DC_AB':([a]+bs,'base'),'BASE_DC_CF':([acf]+bs,'base'),
   'JOINT_AB':([a]+bs,'joint'),'JOINT_CF':([acf]+bs,'joint')}
  for arm in ARMS653:
   keys,engine=cases[arm];gold=z['regions'][1] if arm.endswith('_CF') else z['regions'][0]
   try:
    mem=baseline_memory653(keys) if engine=='base' else joint653(keys) if engine=='joint' else native_memory653(keys)
    ans=answer(validate653(mem,keys),q);err=None
   except Exception as e:ans='';err=f'{type(e).__name__}: {e}'
   rec=dict(case=z['case'],arm=arm,gold=gold,answer=ans,exact=err is None and exact653(ans,gold),error=err)
   RAW653.append(rec);print(' ',arm,'PASS' if rec['exact'] else 'FAIL',repr(ans),err or '',flush=True)
  for tag,source,gold,other in (('A',a,z['regions'][0],z['regions'][1]),('CF',acf,z['regions'][1],z['regions'][0])):
   full=next(r for r in RAW653 if r['case']==z['case'] and r['arm']==('NATIVE_AB' if tag=='A' else 'NATIVE_CF'))
   for j in range(4):
    keys=[source]+[b for k,b in enumerate(bs) if k!=j]
    try:ans=run653(keys,q);err=None
    except Exception as e:ans='';err=f'{type(e).__name__}: {e}'
    rec=dict(case=z['case'],source=tag,removed=j,expected_critical=0 if tag=='A' else 1,
     answer=ans,retains_gold=err is None and exact653(ans,gold),switches_to_other=err is None and exact653(ans,other),
     changed_from_full=err is None and ans!=full['answer'],error=err)
    ABL653.append(rec)
    print(' ',tag,'DROP_B',j,'GOLD_RETAINED' if rec['retains_gold'] else 'GOLD_LOST',repr(ans),err or '',flush=True)
 except Exception as e:
  print('CASE_FATAL',z['case'],traceback.format_exc(),flush=True)
 gc.collect();torch.cuda.empty_cache()
SCORES653={arm:dict(correct=sum(r['exact'] for r in RAW653 if r['arm']==arm),total=N653,errors=sum(bool(r['error']) for r in RAW653 if r['arm']==arm)) for arm in ARMS653}
CAUSAL653=[]
for z in PANEL653:
 r={v['arm']:v for v in RAW653 if v['case']==z['case']}
 ab={(v['source'],v['removed']):v for v in ABL653 if v['case']==z['case']}
 valid=len(ab)==8 and all(not v['error'] for v in ab.values())
 a_critical=valid and not ab[('A',0)]['retains_gold']
 cf_critical=valid and not ab[('CF',1)]['retains_gold']
 a_others=valid and all(ab[('A',j)]['retains_gold'] for j in (1,2,3))
 cf_others=valid and all(ab[('CF',j)]['retains_gold'] for j in (0,2,3))
 parity=all(r.get(k,{}).get('answer')==r.get(n,{}).get('answer') for k,n in (('NATIVE_AB','BASE_DC_AB'),('NATIVE_CF','BASE_DC_CF')))
 full=all(r.get(k,{}).get('exact',False) for k in ('NATIVE_AB','NATIVE_CF','BASE_DC_AB','BASE_DC_CF','JOINT_AB','JOINT_CF')) and not r.get('B_ONLY',{}).get('exact',False)
 passed=bool(valid and a_critical and cf_critical and a_others and cf_others and parity and full)
 CAUSAL653.append(dict(case=z['case'],full_correct=full,native_base_answer_parity=parity,ablation_complete=valid,
  a_critical=a_critical,cf_critical=cf_critical,a_noncritical_retained=a_others,cf_noncritical_retained=cf_others,causal_pass=passed))
 print('CAUSAL',z['case'],'PASS' if passed else 'NOT_DEMONSTRATED','A_CRITICAL',a_critical,'CF_CRITICAL',cf_critical,'NONCRITICAL',a_others,cf_others,flush=True)
report=dict(test=TEST,base_blob=BLOB653,panel_sha256=panel_sha,cut=CUT653,model='Qwen/Qwen2.5-7B-Instruct',
 native_scope='CUDA BF16 KV append; H3 continuation, RoPE and model inference via frozen PyTorch',
 scores=SCORES653,ablation=ABL653,causal=CAUSAL653,raw=RAW653,
 model_frozen=all(not p.requires_grad for p in model.parameters()),sentinel_unchanged=sentinel()==S0,
 seconds=round(time.perf_counter()-T653,2))
path=ROOT653/'TEST653_results.json';path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print('='*116+'\nTEST653_FINAL\n'+json.dumps(SCORES653,indent=2))
print('CAUSAL_CANDIDATE_ABLATION',sum(x['causal_pass'] for x in CAUSAL653),'/',N653)
print('WEIGHTS_FROZEN',report['model_frozen'],'SENTINEL',report['sentinel_unchanged'])
print('RESULT',path,'SHA256',hashlib.sha256(path.read_bytes()).hexdigest())
print('STATUS','EXPERIMENT_COMPLETE' if report['sentinel_unchanged'] and len(ABL653)==8*N653 and not any(r['error'] for r in RAW653+ABL653) else 'ERRORS_PRESENT')
print('LIMITATION: This is causal candidate-deletion analysis, NOT attention attribution, learned routing, autonomous retrieval or full native inference.')
