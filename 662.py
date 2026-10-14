# TEST661 — AKBASCORE MAM · KV INTERFERENCE ISOLATION · LOCKED TEST566
# Single Colab cell; frozen Qwen, CUT3, original TEST566 checkpoint/init/append/answer untouched.
# Native CUDA copies aligned K/V bodies into persistent cache; upper layers use verified DC3 continuation.
# Demonstrates KV connectivity/parity, NOT autonomous memory search or full C++ inference.
import os,sys,re,json,time,random,hashlib,urllib.request,traceback,gc,shutil,subprocess,importlib.util
from pathlib import Path
import torch
TEST='662';SEED659=6502026;N659=8;CUT659=3;MAX_NEW659=16
ROOT659=Path('/content/AKBASCORE_TEST662');ROOT659.mkdir(parents=True,exist_ok=True)
URL659='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/566.py'
BLOB659='b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71'
BOUND659='# TEST566 ADDITIONS — original TEST564 checkpoint/init/append/answer remain unchanged.'
def blob659(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def gate659(n,v):print(('PASS' if v else 'FAIL'),n,flush=True);assert v,n
print('='*116+'\nTEST662 — BLIND NUMERICAL SELECTION / CAUSAL SYNTHESIS\n'+'='*116)
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
native659=load_inline(name='akbascore_kvfabric662_v1',cpp_sources=[CPP659],cuda_sources=[CUDA659],functions=None,extra_cflags=['-O2'],extra_cuda_cflags=['-O2'],with_cuda=True,verbose=True)
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


# TEST662 — BLIND NUMERIC H3 ROUTING / CHAIN CAUSALITY. No changes to TEST566 or native CUT3 engine.
# Fixed unseen seeded panel; no oracle identities supplied to router; no tuning on outcomes.
T662=time.perf_counter();R662=[];ERR662=[];ROUTES662=[];rng662=random.Random(6622026);used662=set(seen)
def name662():
 while True:
  s=''.join(rng662.choice(alphabet) for _ in range(9)).capitalize()
  if s not in used662:used662.add(s);return s
@torch.no_grad()
def h3sig662(ci):
 cp=checkpoint(ci,CUT659);h=cp['h'][:,P:,:].float()
 if h.ndim!=3 or not h.shape[1] or h.shape[-1]!=H:raise RuntimeError('BAD_H3')
 return h.mean(1).squeeze(0).contiguous()
@torch.no_grad()
def score662(q,cs):
 out=torch.empty((len(cs),),device=DEVICE,dtype=torch.float32)
 native659.cosine_scores(out,q.float().contiguous(),torch.stack([v.float() for v in cs]).contiguous())
 return out.cpu().tolist()
def grade662(a,gold,regions):
 hits=[x for x in regions if re.search(r'(?<![A-Za-z])'+re.escape(x)+r'(?![A-Za-z])',a,re.I)]
 return dict(strict=exact659(a,gold),region=len(hits)==1 and hits[0].casefold()==gold.casefold(),unknown=bool(re.search(r'\b(?:UNKNOWN|UNDETERMINED|CANNOT DETERMINE|NOT ENOUGH INFORMATION)\b',a,re.I)))
NCASE662=8;NCHAIN662=4;PANEL662=[]
for case in range(NCASE662):
 chains=[]
 for j in range(NCHAIN662):
  obj,person,org,city,region=[name662() for _ in range(5)]
  facts=[f'The owner of artifact {obj} is {person}.',f'{person} works at organization {org}.',f'Organization {org} is located in city {city}.',f'City {city} belongs to region {region}.']
  chains.append(dict(obj=obj,person=person,org=org,city=city,region=region,ids=[add659(f) for f in facts]))
 target=rng662.randrange(NCHAIN662);z=chains[target];q=f'In which region is the organization where the owner of artifact {z["obj"]} works located? Answer with the region name only. If the available facts do not determine it, answer UNKNOWN.'
 # Mutate the *target's third link* (organization->city), not merely the final region token.
 altcity=chains[(target+1)%NCHAIN662]['city'];cf=add659(f'Organization {z["org"]} is located in city {altcity}.')
 PANEL662.append(dict(case=case,target=target,chains=chains,q=q,cf=cf,cf_gold=chains[(target+1)%NCHAIN662]['region']))
SHA662=hashlib.sha256(json.dumps(PANEL662,sort_keys=True).encode()).hexdigest();print('BLIND_PANEL_SHA',SHA662,flush=True)
# Pre-register signatures, including query signatures, without decoding or looking at answers.
for z in PANEL662:
 z['qid']=add659(z['q']);z['q_sig']=h3sig662(z['qid']);z['sigs']=[[h3sig662(ci) for ci in c['ids']] for c in z['chains']]
def route662(z):
 # Four successive numerical cosine selections. Each stage scores the same-edge candidates,
 # conditioning on the query and previously selected *numeric* H3 vectors only.
 # No gold index or source text is read by the routing function.
 q=z['q_sig'];sel=[];scores=[]
 for edge in range(4):
  sc=score662(q,[c[edge] for c in z['sigs']]);pick=max(range(NCHAIN662),key=lambda j:(sc[j],-j));sel.append(pick);scores.append(sc)
  q=q+z['sigs'][pick][edge]
 return sel,scores
def run662(z,arm,keys,gold,regions):
 try:
  t=time.perf_counter();a=answer(validate659(native_memory659(keys),keys),z['q']);sec=time.perf_counter()-t
  row=dict(case=z['case'],arm=arm,answer=a,gold=gold,elapsed_s=round(sec,4),error=None,**grade662(a,gold,regions))
 except Exception as e:
  row=dict(case=z['case'],arm=arm,answer='',gold=gold,elapsed_s=None,error=f'{type(e).__name__}: {e}',strict=False,region=False,unknown=False);ERR662.append(row)
 R662.append(row);print('CASE',z['case'],arm,'STRICT',int(row['strict']),'UNKNOWN',int(row['unknown']),'ANSWER',repr(row['answer']),'ERROR',row['error'] or '',flush=True)
 return row
for z in PANEL662:
 target=z['target'];chains=z['chains'];goal=chains[target]['region'];regions=[c['region'] for c in chains];tkeys=chains[target]['ids']
 allkeys=[ci for c in chains for ci in c['ids']];rng662.shuffle(allkeys)
 selected,sc=route662(z);chosen=[chains[selected[e]]['ids'][e] for e in range(4)]
 route_row=dict(case=z['case'],selected=selected,gold_chain=target,edge_correct=[j==target for j in selected],all_correct=all(j==target for j in selected),scores=sc)
 ROUTES662.append(route_row);print('ROUTE',z['case'],'PRED',selected,'GOLD',target,'EDGE_MATCH',route_row['edge_correct'],flush=True)
 run662(z,'ORACLE_TARGET',tkeys,goal,regions)
 run662(z,'ALL_COMPETING',allkeys,goal,regions)
 run662(z,'NUMERIC_SELECTED',chosen,goal,regions)
 # Replace target's org->city fact with an alternate city whose region is another chain's region.
 cfkeys=tkeys[:2]+[z['cf'],chains[(target+1)%NCHAIN662]['ids'][3]]
 run662(z,'CF_ORACLE',cfkeys,z['cf_gold'],regions)
 # Missing-edge ablation must return UNKNOWN. Strict correctness against region is a leak, not success.
 for edge in range(4):run662(z,f'MISSING_{edge}',tkeys[:edge]+tkeys[edge+1:],goal,regions)
 run662(z,'DISTRACTORS_ONLY',[ci for j,c in enumerate(chains) if j!=target for ci in c['ids']],goal,regions)
 gc.collect();torch.cuda.empty_cache()
ARMS662=['ORACLE_TARGET','ALL_COMPETING','NUMERIC_SELECTED','CF_ORACLE']+[f'MISSING_{e}' for e in range(4)]+['DISTRACTORS_ONLY']
SUMMARY662={}
for arm in ARMS662:
 rr=[r for r in R662 if r['arm']==arm]
 SUMMARY662[arm]=dict(n=len(rr),strict=sum(r['strict'] for r in rr),unknown=sum(r['unknown'] for r in rr),mean_seconds=round(sum(r['elapsed_s'] or 0 for r in rr)/max(1,sum(r['elapsed_s'] is not None for r in rr)),4))
leaks=sum(r['strict'] for r in R662 if r['arm'].startswith('MISSING_'));missing_unknown=sum(r['unknown'] for r in R662 if r['arm'].startswith('MISSING_'))
report662=dict(test='662',base_blob=BLOB659,model='Qwen/Qwen2.5-7B-Instruct',cut=CUT659,source_panel_sha=panel_sha,panel_sha=SHA662,panel=PANEL662,rows=R662,routes=ROUTES662,summary=SUMMARY662,missing_edge_leaks=leaks,missing_edge_unknown=missing_unknown,route_full_correct=sum(r['all_correct'] for r in ROUTES662),route_edge_correct=sum(sum(r['edge_correct']) for r in ROUTES662),errors=ERR662,weights_frozen=all(not x.requires_grad for x in model.parameters()),sentinel_unchanged=sentinel()==S0,seconds=round(time.perf_counter()-T662,2),limitations='Fixed seeded synthetic blind panel (not independent external holdout). Numerical H3 router is heuristic and uncalibrated. CF_ORACLE replaces the org-to-city edge and supplies the alternate city region edge. ORACLE_TARGET may answer from last-edge leakage: missing-edge controls required. No claim of autonomous 10k retrieval, durable synthesis or sub-ms response.')
# Do not persist large numeric signatures in JSON; store panel metadata only.
for z in report662['panel']:z.pop('q_sig',None);z.pop('sigs',None)
path662=ROOT659/'TEST662_results.json';path662.write_text(json.dumps(report662,ensure_ascii=False,indent=2),encoding='utf-8')
print('='*116+'\nTEST662_FINAL\n'+json.dumps(SUMMARY662,indent=2))
print('ROUTING_FULL',report662['route_full_correct'],'/',NCASE662,'ROUTING_EDGES',report662['route_edge_correct'],'/',NCASE662*4,'MISSING_EDGE_LEAKS',leaks,'/',NCASE662*4,'MISSING_UNKNOWN',missing_unknown,'/',NCASE662*4)
print('WEIGHTS_FROZEN',report662['weights_frozen'],'SENTINEL',report662['sentinel_unchanged'],'ERRORS',len(ERR662))
print('RESULT',path662,'SHA256',hashlib.sha256(path662.read_bytes()).hexdigest())
print('STATUS','EXPERIMENT_COMPLETE' if len(R662)==NCASE662*len(ARMS662) and not ERR662 and report662['weights_frozen'] and report662['sentinel_unchanged'] else 'ERRORS_PRESENT')
