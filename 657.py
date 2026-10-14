# TEST657 — AKBASCORE MAM · NUMERIC ASSOCIATIVE ADDRESSING · LOCKED TEST566
# Single Colab cell; frozen Qwen, CUT3, original TEST566 checkpoint/init/append/answer untouched.
# Native CUDA copies aligned K/V bodies into persistent cache; upper layers use verified DC3 continuation.
# Demonstrates KV connectivity/parity, NOT autonomous memory search or full C++ inference.
import os,sys,re,json,time,random,hashlib,urllib.request,traceback,gc,shutil,subprocess,importlib.util
from pathlib import Path
import torch
TEST='657';SEED657=6502026;N657=8;CUT657=3;MAX_NEW657=16
ROOT657=Path('/content/AKBASCORE_TEST657');ROOT657.mkdir(parents=True,exist_ok=True)
URL657='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/566.py'
BLOB657='b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71'
BOUND657='# TEST566 ADDITIONS — original TEST564 checkpoint/init/append/answer remain unchanged.'
def blob657(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def gate657(n,v):print(('PASS' if v else 'FAIL'),n,flush=True);assert v,n
print('='*116+'\nTEST657 — NATIVE CUDA DYNAMIC KV FABRIC / SOURCE-LOCKED QWEN\n'+'='*116)
gate657('CUDA_AVAILABLE',torch.cuda.is_available())
if importlib.util.find_spec('ninja') is None:subprocess.check_call([sys.executable,'-m','pip','install','-q','ninja'])
from torch.utils.cpp_extension import load_inline,CUDA_HOME
nvcc657=shutil.which('nvcc') or (str(Path(CUDA_HOME)/'bin'/'nvcc') if CUDA_HOME else None)
gate657('NINJA',importlib.util.find_spec('ninja') is not None)
gate657('NVCC',bool(nvcc657 and Path(nvcc657).is_file()))
gate657('CXX',bool(shutil.which('c++') or shutil.which('g++')))
os.environ.setdefault('MAX_JOBS','2')
# Compile native extension BEFORE downloading model weights.
CPP657=r'''
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
CUDA657=r'''
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
native657=load_inline(name='akbascore_kvfabric657_v1',cpp_sources=[CPP657],cuda_sources=[CUDA657],functions=None,extra_cflags=['-O2'],extra_cuda_cflags=['-O2'],with_cuda=True,verbose=True)
@torch.no_grad()
def splice657(a,b):
 a=a.contiguous();b=b.contiguous();out=torch.empty((a.shape[0],a.shape[1],a.shape[2]+b.shape[2],a.shape[3]),device=a.device,dtype=a.dtype)
 native657.append_kv(out,a,b);return out
for t0,t1 in ((29,7),(43,31),(1,1)):
 x=torch.randn((1,4,t0,128),device='cuda',dtype=torch.bfloat16);y=torch.randn((1,4,t1,128),device='cuda',dtype=torch.bfloat16)
 gate657('NATIVE_BITWISE_'+str(t0)+'_'+str(t1),torch.equal(splice657(x,y),torch.cat((x,y),dim=2)))
# Load exactly the same verified baseline as TEST650.
p=Path('/content/566.py')
if p.is_file() and blob657(p.read_bytes())==BLOB657:raw=p.read_bytes();origin='LOCAL_VERIFIED'
else:
 req=urllib.request.Request(URL657,headers={'User-Agent':'AKBASCORE-TEST657'})
 with urllib.request.urlopen(req,timeout=120) as f:raw=f.read()
 origin='GITHUB_VERIFIED'
gate657('TEST566_BLOB',blob657(raw)==BLOB657)
src=raw.decode('utf-8');gate657('TEST566_BOUNDARY',src.count(BOUND657)==1)
exec(compile(src.split(BOUND657,1)[0],'566.py','exec'),globals())
gate657('FROZEN',all(not p.requires_grad for p in model.parameters()))
gate657('ARCH',(NL,H,QH,KVH,HD)==(28,3584,28,4,128))
gate657('PANEL_LOCK',PANEL_SHA=='69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45')
gate657('SENTINEL',sentinel()==S0)
print('BASE',origin,'BLOB',blob657(raw),'GPU',torch.cuda.get_device_name(0),flush=True)
# Same panel generator and random seed as TEST650: identical A/A' and four competing B facts.
rng=random.Random(SEED657);alphabet='ABCDEFGHJKLMNPQRSTUVWXYZ';seen=set()
def word657():
 while True:
  s=''.join(rng.choice(alphabet) for _ in range(9)).capitalize()
  if s not in seen:seen.add(s);return s
PANEL657=[];CAR657=[]
def add657(s):
 ids=tok(source(s),add_special_tokens=False).input_ids
 if ids[:P]!=PIDS:raise RuntimeError('PREFIX_MISMATCH')
 i=len(CAR657);CAR657.append(dict(fact=s,body=ids[P:]));return i
for i in range(N657):
 obj=word657();owners=[word657() for _ in range(4)];regions=[word657() for _ in range(4)]
 a=add657(f'The owner of artifact {obj} is {owners[0]}.')
 acf=add657(f'The owner of artifact {obj} is {owners[1]}.')
 bs=[add657(f'{owners[j]} lives in region {regions[j]}.') for j in range(4)]
 PANEL657.append(dict(case=i,object=obj,owners=owners,regions=regions,a=a,acf=acf,bs=bs,q=f'In which region does the owner of artifact {obj} live?',qa=f'Who owns artifact {obj}?'))
CAR=CAR657;MAX_NEW=MAX_NEW657
panel_sha=hashlib.sha256(json.dumps(PANEL657,sort_keys=True).encode()).hexdigest()
gate657('TEST650_PANEL_LOCK',panel_sha=='db8b211abed216793bf2e4010eb006c862a74202d7512ae53a8a463fe929d466')
@torch.no_grad()
def native_append657(memory,ci):
 # Preserve the locked TEST566 upper-layer computation and H3 injection exactly.
 # Only lower-layer K/V concatenation is replaced with native CUDA, after RoPE position alignment.
 cp=checkpoint(ci,CUT657);oldn=memory[0][0].shape[-2];q=cp['body']
 pos=torch.arange(oldn,oldn+q,device=DEVICE);body=cp['h'][:,P:,:]
 if body.shape!=(1,q,H):raise RuntimeError('H3_BODY_SHAPE')
 def inject(module,args,out):
  current=out[0] if isinstance(out,(tuple,list)) else out
  if current.shape!=body.shape:raise RuntimeError('H3_INJECTION_SHAPE')
  return replace(out,body)
 hook=model.model.layers[CUT657].register_forward_hook(inject)
 try:
  out=model(inputs_embeds=torch.zeros((1,q,H),device=DEVICE,dtype=DTYPE),past_key_values=cache_build(memory),attention_mask=torch.ones((1,oldn+q),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
  grown=list(clone(out.past_key_values))
  for li in range(CUT657+1):
   ok,ov=memory[li];k,v=cp['kv'][li]
   k=rephase(k[:,:,P:,:],cp['pos'][P:],pos);v=v[:,:,P:,:]
   grown[li]=(splice657(ok,k),splice657(ov,v))
  result=tuple(grown)
  for (ok,ov),(nk,nv) in zip(memory,result):
   if not torch.equal(ok,nk[:,:,:oldn,:]) or not torch.equal(ov,nv[:,:,:oldn,:]):raise RuntimeError('PREFIX_MUTATED')
  if len(result)!=NL or any(k.shape!=(1,KVH,oldn+q,HD) or v.shape!=(1,KVH,oldn+q,HD) for k,v in result):raise RuntimeError('CACHE_SHAPE')
  return result
 finally:hook.remove()
@torch.no_grad()
def native_memory657(keys):
 mem=init(keys[0],CUT657)
 for ci in keys[1:]:mem=native_append657(mem,ci)
 return mem
@torch.no_grad()
def baseline_memory657(keys):
 mem=init(keys[0],CUT657)
 for ci in keys[1:]:mem=append(mem,ci,CUT657)
 return mem
@torch.no_grad()
def independent657(keys):
 mem=None
 for ci in keys:
  cp=checkpoint(ci,CUT657);kv=cp['kv']
  if mem is None:mem=kv;continue
  oldn=mem[0][0].shape[-2];q=len(CAR[ci]['body'])
  oldpos=torch.arange(P,P+q,device=DEVICE);newpos=torch.arange(oldn,oldn+q,device=DEVICE)
  mem=tuple((splice657(ok,rephase(k[:,:,P:,:],oldpos,newpos)),splice657(ov,v[:,:,P:,:])) for (ok,ov),(k,v) in zip(mem,kv))
 return mem
@torch.no_grad()
def joint657(keys):
 ids=list(PIDS)
 for ci in keys:ids.extend(CAR[ci]['body'])
 n=len(ids);pos=torch.arange(n,device=DEVICE)
 out=model(input_ids=torch.tensor([ids],device=DEVICE),attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
 return clone(out.past_key_values)
def validate657(mem,keys):
 n=P+sum(len(CAR[i]['body']) for i in keys)
 if len(mem)!=NL or any(k.shape!=(1,KVH,n,HD) or v.shape!=(1,KVH,n,HD) or not torch.isfinite(k).all() or not torch.isfinite(v).all() for k,v in mem):raise RuntimeError('INVALID_CACHE')
 return mem
def exact657(a,b):return re.sub(r'[^a-z]','',a.casefold())==re.sub(r'[^a-z]','',b.casefold())


# TEST657 — A-conditioned H3 boundary audit. No changes to CUT3/native KV engine.
# IMPORTANT: Reordering score-vector indices is an invariance check, NOT a new memory representation.
import itertools,math
T657=time.perf_counter();RESULTS657=[];ROUTES657=[];ERRORS657=[]
@torch.no_grad()
def h3sig657(ci):
 cp=checkpoint(ci,CUT657);h=cp['h'][:,P:,:].float()
 if h.ndim!=3 or h.shape[-1]!=H or not h.shape[1]:raise RuntimeError('H3_SHAPE')
 return h.mean(1).squeeze(0).contiguous()
@torch.no_grad()
def scores657(query,candidates):
 out=torch.empty((len(candidates),),device=DEVICE,dtype=torch.float32)
 native657.cosine_scores(out,query.float().contiguous(),torch.stack([v.float() for v in candidates]).contiguous())
 return out.cpu().tolist()
def order657(sc):return sorted(range(len(sc)),key=lambda j:(-sc[j],j))
def grade657(ans,gold,regions):
 hits=[v for v in regions if re.search(r'(?<![A-Za-z])'+re.escape(v)+r'(?![A-Za-z])',ans,re.I)]
 return dict(strict=exact657(ans,gold),region=(len(hits)==1 and hits[0].casefold()==gold.casefold()))
QIDS657={z['case']:add657(z['q']) for z in PANEL657}
PERMS657=list(itertools.permutations(range(4)))
for z in PANEL657:
 i=z['case'];bs=z['bs'];regions=z['regions'];qci=QIDS657[i]
 try:
  qh=h3sig657(qci);bh=[h3sig657(b) for b in bs]
  for tag,a_ci,gold in (('A',z['a'],0),('CF',z['acf'],1)):
   a=h3sig657(a_ci);sc=scores657(a+qh,bh);rank=order657(sc);pred=rank[0]
   # Positive margin means B_gold wins. Negative means the strongest rival wins.
   rivals=[j for j in range(4) if j!=gold];best_rival=max(rivals,key=lambda j:(sc[j],-j))
   margin=sc[gold]-sc[best_rival];b10=sc[1]-sc[0]
   # Full 24-permutation identity audit, keeping tensor contents fixed.
   perm_audit=[]
   for perm in PERMS657:
    perm_scores=[sc[j] for j in perm]
    perm_selected=perm[order657(perm_scores)[0]]
    perm_audit.append(dict(order=list(perm),selected_identity=perm_selected,correct=perm_selected==gold))
   invariant=all(x['selected_identity']==pred for x in perm_audit)
   # Two physical concatenation orders: original and B0/B1 swap. Only the A + selected B
   # is decoded, so changing the unused candidates' listing must not change that cache.
   # The ORACLE_ALL arm separately checks sensitivity of actual multi-cartridge concatenation.
   order_orig=(0,1,2,3);order_swap=(1,0,2,3)
   decoded={};keys_map={
    'ORACLE_ONE':[a_ci,bs[gold]],'ROUTED_ONE':[a_ci,bs[pred]],
    'ORACLE_ALL_ORIG':[a_ci]+[bs[j] for j in order_orig],
    'ORACLE_ALL_SWAP01':[a_ci]+[bs[j] for j in order_swap],
    'A_ONLY':[a_ci]}
   for arm,keys in keys_map.items():
    key=tuple(keys)
    if key not in decoded:
     try:decoded[key]=(answer(validate657(native_memory657(keys),keys),z['q']),None)
     except Exception as e:decoded[key]=('',f'{type(e).__name__}: {e}')
    ans,err=decoded[key];r=dict(case=i,source=tag,arm=arm,answer=ans,error=err,gold_region=regions[gold],**grade657(ans,regions[gold],regions))
    RESULTS657.append(r)
    if err:ERRORS657.append(r)
   ROUTES657.append(dict(case=i,source=tag,gold_B=gold,selected_B=pred,correct=pred==gold,
    scores=sc,rank=rank,gold_margin=margin,top_margin=sc[rank[0]]-sc[rank[1]],
    B1_minus_B0=b10,best_rival=best_rival,all_24_index_permutations_invariant=invariant,
    permutation_audit=perm_audit))
   print('CASE',i,tag,'GOLD',gold,'SELECTED',pred,'RANK',rank,
    'SCORES',[round(x,5) for x in sc],'GOLD_MARGIN',round(margin,6),
    'B1_MINUS_B0',round(b10,6),'PERM24_INVARIANT',invariant,flush=True)
   for r in RESULTS657[-5:]:print(' ',r['arm'],'STRICT',int(r['strict']),'REGION',int(r['region']),repr(r['answer']),r['error'] or '',flush=True)
 except Exception as e:
  ERRORS657.append(dict(case=i,error=f'{type(e).__name__}: {e}'))
  print('CASE_ERROR',i,traceback.format_exc(),flush=True)
 gc.collect();torch.cuda.empty_cache()
ARMS657=('ORACLE_ONE','ROUTED_ONE','ORACLE_ALL_ORIG','ORACLE_ALL_SWAP01','A_ONLY')
SUMMARY657={arm:{metric:sum(bool(r[metric]) for r in RESULTS657 if r['arm']==arm) for metric in ('strict','region')}|{'n':sum(r['arm']==arm for r in RESULTS657)} for arm in ARMS657}
route_a=[r for r in ROUTES657 if r['source']=='A'];route_cf=[r for r in ROUTES657 if r['source']=='CF']
report657=dict(test='657',base_blob=BLOB657,panel_sha256=panel_sha,model='Qwen/Qwen2.5-7B-Instruct',cut=CUT657,
 hypothesis='Audit A-conditioned H3 routing: B0/B1 label bias, 24 index permutations, decision margins, and physical concatenation order',
 routes=ROUTES657,results=RESULTS657,scores=SUMMARY657,errors=ERRORS657,
 route_accuracy={'A':sum(r['correct'] for r in route_a),'CF':sum(r['correct'] for r in route_cf)},
 B1_preferred={'A':sum(r['B1_minus_B0']>0 for r in route_a),'CF':sum(r['B1_minus_B0']>0 for r in route_cf)},
 index_invariance=sum(r['all_24_index_permutations_invariant'] for r in ROUTES657),
 weights_frozen=all(not p.requires_grad for p in model.parameters()),sentinel_unchanged=sentinel()==S0,
 seconds=round(time.perf_counter()-T657,2),
 limitations='24 permutations only permute numeric candidate index order; physical cache concatenation is checked in two orders. This does not prove full order invariance, generalization, autonomous routing, or architectural necessity. The panel always assigns A to B0 and CF to B1; score bias cannot be fully disentangled without newly balanced panels.')
out657=ROOT657/'TEST657_results.json';out657.write_text(json.dumps(report657,indent=2,ensure_ascii=False),encoding='utf-8')
print('='*116+'\nTEST657_FINAL\n'+json.dumps(SUMMARY657,indent=2))
print('ROUTING_ACCURACY',report657['route_accuracy'],'B1_PREFERRED',report657['B1_preferred'],'INDEX_INVARIANCE',report657['index_invariance'],'/',len(ROUTES657))
print('WEIGHTS_FROZEN',report657['weights_frozen'],'SENTINEL',report657['sentinel_unchanged'])
print('RESULT',out657,'SHA256',hashlib.sha256(out657.read_bytes()).hexdigest())
complete=len(RESULTS657)==N657*2*len(ARMS657) and len(ROUTES657)==N657*2 and not ERRORS657 and report657['weights_frozen'] and report657['sentinel_unchanged']
print('STATUS','EXPERIMENT_COMPLETE' if complete else 'ERRORS_PRESENT')
