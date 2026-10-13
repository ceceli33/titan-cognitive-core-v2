# TEST651 — AKBASCORE MAM · NATIVE CUDA KV FABRIC · LOCKED TEST566
# Single Colab cell; frozen Qwen, CUT3, original TEST566 checkpoint/init/append/answer untouched.
# Native CUDA copies aligned K/V bodies into persistent cache; upper layers use verified DC3 continuation.
# Demonstrates KV connectivity/parity, NOT autonomous memory search or full C++ inference.
import os,sys,re,json,time,random,hashlib,urllib.request,traceback,gc,shutil,subprocess,importlib.util
from pathlib import Path
import torch
TEST='651';SEED651=6502026;N651=8;CUT651=3;MAX_NEW651=16
ROOT651=Path('/content/AKBASCORE_TEST651');ROOT651.mkdir(parents=True,exist_ok=True)
URL651='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/566.py'
BLOB651='b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71'
BOUND651='# TEST566 ADDITIONS — original TEST564 checkpoint/init/append/answer remain unchanged.'
def blob651(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def gate651(n,v):print(('PASS' if v else 'FAIL'),n,flush=True);assert v,n
print('='*116+'\nTEST651 — NATIVE CUDA DYNAMIC KV FABRIC / SOURCE-LOCKED QWEN\n'+'='*116)
gate651('CUDA_AVAILABLE',torch.cuda.is_available())
if importlib.util.find_spec('ninja') is None:subprocess.check_call([sys.executable,'-m','pip','install','-q','ninja'])
from torch.utils.cpp_extension import load_inline,CUDA_HOME
nvcc651=shutil.which('nvcc') or (str(Path(CUDA_HOME)/'bin'/'nvcc') if CUDA_HOME else None)
gate651('NINJA',importlib.util.find_spec('ninja') is not None)
gate651('NVCC',bool(nvcc651 and Path(nvcc651).is_file()))
gate651('CXX',bool(shutil.which('c++') or shutil.which('g++')))
os.environ.setdefault('MAX_JOBS','2')
# Compile native extension BEFORE downloading model weights.
CPP651=r'''
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
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("append_kv",&append_kv,"Native BF16 KV append");}
'''
CUDA651=r'''
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
'''
print('BUILDING NATIVE KV EXTENSION',flush=True)
native651=load_inline(name='akbascore_kvfabric651_v1',cpp_sources=[CPP651],cuda_sources=[CUDA651],functions=None,extra_cflags=['-O2'],extra_cuda_cflags=['-O2'],with_cuda=True,verbose=True)
@torch.no_grad()
def splice651(a,b):
 a=a.contiguous();b=b.contiguous();out=torch.empty((a.shape[0],a.shape[1],a.shape[2]+b.shape[2],a.shape[3]),device=a.device,dtype=a.dtype)
 native651.append_kv(out,a,b);return out
for t0,t1 in ((29,7),(43,31),(1,1)):
 x=torch.randn((1,4,t0,128),device='cuda',dtype=torch.bfloat16);y=torch.randn((1,4,t1,128),device='cuda',dtype=torch.bfloat16)
 gate651('NATIVE_BITWISE_'+str(t0)+'_'+str(t1),torch.equal(splice651(x,y),torch.cat((x,y),dim=2)))
# Load exactly the same verified baseline as TEST650.
p=Path('/content/566.py')
if p.is_file() and blob651(p.read_bytes())==BLOB651:raw=p.read_bytes();origin='LOCAL_VERIFIED'
else:
 req=urllib.request.Request(URL651,headers={'User-Agent':'AKBASCORE-TEST651'})
 with urllib.request.urlopen(req,timeout=120) as f:raw=f.read()
 origin='GITHUB_VERIFIED'
gate651('TEST566_BLOB',blob651(raw)==BLOB651)
src=raw.decode('utf-8');gate651('TEST566_BOUNDARY',src.count(BOUND651)==1)
exec(compile(src.split(BOUND651,1)[0],'566.py','exec'),globals())
gate651('FROZEN',all(not p.requires_grad for p in model.parameters()))
gate651('ARCH',(NL,H,QH,KVH,HD)==(28,3584,28,4,128))
gate651('PANEL_LOCK',PANEL_SHA=='69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45')
gate651('SENTINEL',sentinel()==S0)
print('BASE',origin,'BLOB',blob651(raw),'GPU',torch.cuda.get_device_name(0),flush=True)
# Same panel generator and random seed as TEST650: identical A/A' and four competing B facts.
rng=random.Random(SEED651);alphabet='ABCDEFGHJKLMNPQRSTUVWXYZ';seen=set()
def word651():
 while True:
  s=''.join(rng.choice(alphabet) for _ in range(9)).capitalize()
  if s not in seen:seen.add(s);return s
PANEL651=[];CAR651=[]
def add651(s):
 ids=tok(source(s),add_special_tokens=False).input_ids
 if ids[:P]!=PIDS:raise RuntimeError('PREFIX_MISMATCH')
 i=len(CAR651);CAR651.append(dict(fact=s,body=ids[P:]));return i
for i in range(N651):
 obj=word651();owners=[word651() for _ in range(4)];regions=[word651() for _ in range(4)]
 a=add651(f'The owner of artifact {obj} is {owners[0]}.')
 acf=add651(f'The owner of artifact {obj} is {owners[1]}.')
 bs=[add651(f'{owners[j]} lives in region {regions[j]}.') for j in range(4)]
 PANEL651.append(dict(case=i,object=obj,owners=owners,regions=regions,a=a,acf=acf,bs=bs,q=f'In which region does the owner of artifact {obj} live?',qa=f'Who owns artifact {obj}?'))
CAR=CAR651;MAX_NEW=MAX_NEW651
panel_sha=hashlib.sha256(json.dumps(PANEL651,sort_keys=True).encode()).hexdigest()
gate651('TEST650_PANEL_LOCK',panel_sha=='db8b211abed216793bf2e4010eb006c862a74202d7512ae53a8a463fe929d466')
@torch.no_grad()
def native_append651(memory,ci):
 # Preserve the locked TEST566 upper-layer computation and H3 injection exactly.
 # Only lower-layer K/V concatenation is replaced with native CUDA, after RoPE position alignment.
 cp=checkpoint(ci,CUT651);oldn=memory[0][0].shape[-2];q=cp['body']
 pos=torch.arange(oldn,oldn+q,device=DEVICE);body=cp['h'][:,P:,:]
 if body.shape!=(1,q,H):raise RuntimeError('H3_BODY_SHAPE')
 def inject(module,args,out):
  current=out[0] if isinstance(out,(tuple,list)) else out
  if current.shape!=body.shape:raise RuntimeError('H3_INJECTION_SHAPE')
  return replace(out,body)
 hook=model.model.layers[CUT651].register_forward_hook(inject)
 try:
  out=model(inputs_embeds=torch.zeros((1,q,H),device=DEVICE,dtype=DTYPE),past_key_values=cache_build(memory),attention_mask=torch.ones((1,oldn+q),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
  grown=list(clone(out.past_key_values))
  for li in range(CUT651+1):
   ok,ov=memory[li];k,v=cp['kv'][li]
   k=rephase(k[:,:,P:,:],cp['pos'][P:],pos);v=v[:,:,P:,:]
   grown[li]=(splice651(ok,k),splice651(ov,v))
  result=tuple(grown)
  for (ok,ov),(nk,nv) in zip(memory,result):
   if not torch.equal(ok,nk[:,:,:oldn,:]) or not torch.equal(ov,nv[:,:,:oldn,:]):raise RuntimeError('PREFIX_MUTATED')
  if len(result)!=NL or any(k.shape!=(1,KVH,oldn+q,HD) or v.shape!=(1,KVH,oldn+q,HD) for k,v in result):raise RuntimeError('CACHE_SHAPE')
  return result
 finally:hook.remove()
@torch.no_grad()
def native_memory651(keys):
 mem=init(keys[0],CUT651)
 for ci in keys[1:]:mem=native_append651(mem,ci)
 return mem
@torch.no_grad()
def baseline_memory651(keys):
 mem=init(keys[0],CUT651)
 for ci in keys[1:]:mem=append(mem,ci,CUT651)
 return mem
@torch.no_grad()
def independent651(keys):
 mem=None
 for ci in keys:
  cp=checkpoint(ci,CUT651);kv=cp['kv']
  if mem is None:mem=kv;continue
  oldn=mem[0][0].shape[-2];q=len(CAR[ci]['body'])
  oldpos=torch.arange(P,P+q,device=DEVICE);newpos=torch.arange(oldn,oldn+q,device=DEVICE)
  mem=tuple((splice651(ok,rephase(k[:,:,P:,:],oldpos,newpos)),splice651(ov,v[:,:,P:,:])) for (ok,ov),(k,v) in zip(mem,kv))
 return mem
@torch.no_grad()
def joint651(keys):
 ids=list(PIDS)
 for ci in keys:ids.extend(CAR[ci]['body'])
 n=len(ids);pos=torch.arange(n,device=DEVICE)
 out=model(input_ids=torch.tensor([ids],device=DEVICE),attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
 return clone(out.past_key_values)
def validate651(mem,keys):
 n=P+sum(len(CAR[i]['body']) for i in keys)
 if len(mem)!=NL or any(k.shape!=(1,KVH,n,HD) or v.shape!=(1,KVH,n,HD) or not torch.isfinite(k).all() or not torch.isfinite(v).all() for k,v in mem):raise RuntimeError('INVALID_CACHE')
 return mem
def exact651(a,b):return re.sub(r'[^a-z]','',a.casefold())==re.sub(r'[^a-z]','',b.casefold())
ARMS651=('B_ONLY','INDEP_AB','NATIVE_AB','NATIVE_CF','BASE_DC_AB','BASE_DC_CF','JOINT_AB','JOINT_CF')
RAW651=[];PARITY651=[];T651=time.perf_counter()
for z in PANEL651:
 print('CASE',z['case'],'BUILDING KV FABRIC',flush=True)
 a,acf,bs=z['a'],z['acf'],z['bs'];arms={}
 try:
  # Parity compares native KV continuation to the source-locked baseline before generation.
  for label,keys in (('AB',[a]+bs),('CF',[acf]+bs)):
   m_native=validate651(native_memory651(keys),keys)
   m_base=validate651(baseline_memory651(keys),keys)
   identical=all(torch.equal(x,y) and torch.equal(u,v) for (x,u),(y,v) in zip(m_native,m_base))
   maxerr=max(max((x.float()-y.float()).abs().max().item(),(u.float()-v.float()).abs().max().item()) for (x,u),(y,v) in zip(m_native,m_base))
   PARITY651.append(dict(case=z['case'],arm=label,bitwise=identical,max_error=maxerr))
   gate651('KV_PARITY_'+label+'_'+str(z['case']),identical)
   arms['NATIVE_'+label]=m_native;arms['BASE_DC_'+label]=m_base
  arms['B_ONLY']=validate651(native_memory651(bs),bs)
  arms['INDEP_AB']=validate651(independent651([a]+bs),[a]+bs)
  arms['JOINT_AB']=validate651(joint651([a]+bs),[a]+bs)
  arms['JOINT_CF']=validate651(joint651([acf]+bs),[acf]+bs)
  for arm in ARMS651:
   gold=z['regions'][1] if arm.endswith('_CF') else z['regions'][0]
   t=time.perf_counter()
   try:ans=answer(arms[arm],z['q']);err=None
   except Exception as e:ans='';err=type(e).__name__+': '+str(e)
   rec=dict(case=z['case'],arm=arm,gold=gold,answer=ans,exact=err is None and exact651(ans,gold),error=err,seconds=round(time.perf_counter()-t,3))
   RAW651.append(rec);print(' ',arm,'PASS' if rec['exact'] else 'FAIL',repr(ans),err or '',flush=True)
 except Exception as e:
  print('CASE_FATAL',z['case'],traceback.format_exc(),flush=True)
  for arm in ARMS651:
   if not any(r['case']==z['case'] and r['arm']==arm for r in RAW651):RAW651.append(dict(case=z['case'],arm=arm,gold='',answer='',exact=False,error='CASE_FATAL: '+str(e)))
 gc.collect();torch.cuda.empty_cache()
SCORES651={arm:dict(correct=sum(r['exact'] for r in RAW651 if r['arm']==arm),total=N651,errors=sum(bool(r['error']) for r in RAW651 if r['arm']==arm)) for arm in ARMS651}
CAUSAL651=[]
for z in PANEL651:
 r={v['arm']:v for v in RAW651 if v['case']==z['case']}
 native_ok=r['NATIVE_AB']['exact'] and r['NATIVE_CF']['exact']
 flip=r['NATIVE_AB']['answer'].casefold()!=r['NATIVE_CF']['answer'].casefold()
 controls=not r['B_ONLY']['exact'] and not r['INDEP_AB']['exact']
 ceiling=r['BASE_DC_AB']['exact'] and r['BASE_DC_CF']['exact'] and r['JOINT_AB']['exact'] and r['JOINT_CF']['exact']
 parity=all(p['bitwise'] for p in PARITY651 if p['case']==z['case']) and sum(p['case']==z['case'] for p in PARITY651)==2
 causal=bool(native_ok and flip and controls and ceiling and parity)
 CAUSAL651.append(dict(case=z['case'],native_correct=bool(native_ok),counterfactual_flip=bool(flip),negative_controls=bool(controls),reference_ceiling=bool(ceiling),native_bitwise_parity=bool(parity),causal_pass=causal))
 print('CAUSAL',z['case'],'PASS' if causal else 'NOT_DEMONSTRATED',flush=True)
report=dict(test=TEST,base_blob=BLOB651,panel_sha256=panel_sha,cut=CUT651,model='Qwen/Qwen2.5-7B-Instruct',native_scope='CUDA K/V append after reference RoPE rephase; H3 injection and Qwen forward via locked PyTorch',scores=SCORES651,causal=CAUSAL651,parity=PARITY651,raw=RAW651,model_frozen=all(not p.requires_grad for p in model.parameters()),sentinel_unchanged=sentinel()==S0,seconds=round(time.perf_counter()-T651,2))
path=ROOT651/'TEST651_results.json';path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print('='*116+'\nTEST651_FINAL\n'+json.dumps(SCORES651,indent=2))
print('NATIVE_KV_BITWISE_PARITY',sum(p['bitwise'] for p in PARITY651),'/',2*N651)
print('CAUSAL_KV_LINK',sum(x['causal_pass'] for x in CAUSAL651),'/',N651)
print('SYNTHESIS_STATUS','KV_LINK_CAUSAL_EVIDENCE' if all(x['causal_pass'] for x in CAUSAL651) else 'NOT_DEMONSTRATED')
print('WEIGHTS_FROZEN',report['model_frozen'],'SENTINEL',report['sentinel_unchanged'])
print('RESULT',path,'SHA256',hashlib.sha256(path.read_bytes()).hexdigest())
print('STATUS','EXPERIMENT_COMPLETE' if report['sentinel_unchanged'] and all(p['bitwise'] for p in PARITY651) and not any(r['error'] for r in RAW651) else 'ERRORS_PRESENT')
print('LIMITATION: Native CUDA performs KV append; RoPE, H3 continuation and model forward use verified PyTorch. No autonomous retrieval loop is claimed.')
