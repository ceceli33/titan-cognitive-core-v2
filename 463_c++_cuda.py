
# ======================================================================================================================
# TEST643 — AKBASCORE NATIVE FOUNDATION · STANDALONE C++ / CUDA
# FROZEN QWEN2.5-7B · CUT3 · 1453 PHYSICAL CARTRIDGES · A100
# VERIFIED HIPPOCAMPUS ENGINE → NATIVE BF16 CUDA AUDIT
# NO GRADIO · NO TRAINING · NO TASK-SPECIFIC PATCHES · NO ENGINE MODIFICATION
# ======================================================================================================================
import os,sys,re,json,time,hashlib,urllib.request,types,traceback,random,gc,shutil,subprocess,importlib.util,threading,contextlib,platform,math,html,zipfile,textwrap
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import numpy as np
TEST="643";SEED=643;SAMPLE_N=16
ROOT=Path("/content/AKBASCORE_TEST643");ROOT.mkdir(parents=True,exist_ok=True)
REPO="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/"
SOURCE="HIPPOCAMPUS_full.py"
SOURCE_BLOB="c2c77b170365a08499bbc55d523bb04d46a12a6c"
ENGINE_EXPECTED="ab53e2d58fb3dc5f11409a5112e92feea8762544e4b30b8a77a39b60332de4d1"
def log(*a):print(*a,flush=True)
def utc():return datetime.now(timezone.utc).isoformat()
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def git_blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def file_sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"AKBASCORE-TEST643"}),timeout=180) as f:return f.read()
def save(name,data):
    p=ROOT/name
    p.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    return p
log("="*120)
log("TEST643 — AKBASCORE STANDALONE NATIVE C++ / CUDA FOUNDATION")
log("START",utc())
log("="*120)
log("[1/9] ENVIRONMENT")
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate"),("faiss","faiss-cpu")]:
    if importlib.util.find_spec(mod) is None:
        subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import torch,transformers
from torch.utils.cpp_extension import load_inline,CUDA_HOME
assert torch.cuda.is_available(),"CUDA GPU REQUIRED"
assert CUDA_HOME is not None,"CUDA TOOLKIT REQUIRED"
assert shutil.which("c++") is not None,"C++ COMPILER REQUIRED"
NVCC=Path(CUDA_HOME)/"bin"/"nvcc"
assert NVCC.is_file(),"NVCC NOT FOUND"
GPU=torch.cuda.get_device_name(0)
log("GPU",GPU)
log("TORCH",torch.__version__,"TRANSFORMERS",transformers.__version__)
log("CUDA",torch.version.cuda)
log("NVCC",subprocess.check_output([str(NVCC),"--version"],text=True).splitlines()[-1])
log("[2/9] VERIFIED HIPPOCAMPUS SOURCE")
raw=fetch(REPO+SOURCE)
source_blob=git_blob(raw)
assert source_blob==SOURCE_BLOB,f"SOURCE BLOB MISMATCH: {source_blob}"
source_sha=sha_bytes(raw)
src=raw.decode("utf-8")
assert src.count("#<<ENGINE_END>>")==1,"ENGINE SECTION NOT FOUND"
part1=src.split("#<<ENGINE_END>>",1)[0]
old='("gradio","gradio"),'
assert part1.count(old)==1,"GRADIO DEPENDENCY BOUNDARY CHANGED"
part1=part1.replace(old,"")
old_import="import numpy as np,torch,transformers,gradio as gr"
assert part1.count(old_import)==1,"GRADIO IMPORT BOUNDARY CHANGED"
part1=part1.replace(old_import,'import numpy as np,torch,transformers\nimport types as _types\ngr=_types.SimpleNamespace(__version__="DISABLED_TEST643")')
assert "gr.Blocks(" not in part1,"UNEXPECTED GRADIO UI IN ENGINE SECTION"
(ROOT/"SOURCE_LOCK.json").write_text(json.dumps({"source":SOURCE,"source_sha256":source_sha,"source_blob":source_blob},indent=2))
log("SOURCE VERIFIED",source_blob)
log("[3/9] STARTING LOCKED ENGINE")
if globals().get("ENGINE_READY",False) and globals().get("MAM_SH_PART1_OK",False) and globals().get("ENGINE_SHA256")==ENGINE_EXPECTED:
    log("REUSING VERIFIED ENGINE FROM CURRENT RUNTIME")
else:
    log("COLD START — LOADING MODEL AND REBUILDING ORIGINAL PHYSICAL CARTRIDGE BANK")
    log("THE ORIGINAL ENGINE MAY TAKE SEVERAL MINUTES")
    start_engine=time.perf_counter()
    exec(compile(part1,"HIPPOCAMPUS_TEST643_BOOTSTRAP.py","exec"),globals())
    log("ENGINE_BOOT_SECONDS",round(time.perf_counter()-start_engine,3))
assert globals().get("ENGINE_READY",False),"ENGINE_NOT_READY"
assert globals().get("MAM_SH_PART1_OK",False),"ENGINE_BOOTSTRAP_FAILED"
assert ENGINE_SHA256==ENGINE_EXPECTED,"ENGINE_SOURCE_MISMATCH"
assert all(ok for _,ok in ENGINE_LOCK),"ENGINE_LOCK_FAILED"
assert len(ENG.MANIFEST)==1453,"MANIFEST_COUNT_MISMATCH"
assert len(ENG.model.model.layers)==28,"MODEL_LAYER_MISMATCH"
assert not ENG.model.training,"MODEL_NOT_IN_EVAL_MODE"
assert not any(p.requires_grad for p in ENG.model.parameters()),"TRAINABLE_WEIGHTS_DETECTED"
weight_before=INSTR.info["weight_sha0"]
sentinel_before=INSTR.sentinel()
assert sentinel_before==INSTR.info["sentinel0"],"STARTUP_SENTINEL_MISMATCH"
log("ENGINE_READY",ENGINE_SHA256)
log("CARTRIDGES",len(ENG.MANIFEST))
log("WEIGHT_REFERENCE",weight_before)
log("[4/9] C++ / CUDA SOURCE")
CPP=r"""
#include <torch/extension.h>
#include <vector>
#include <cstdint>
torch::Tensor compare_cuda(torch::Tensor a,torch::Tensor b);
torch::Tensor compare_native(torch::Tensor a,torch::Tensor b){
 TORCH_CHECK(a.is_cuda() && b.is_cuda(),"CUDA tensors required");
 TORCH_CHECK(a.scalar_type()==torch::kBFloat16,"First tensor must be BF16");
 TORCH_CHECK(b.scalar_type()==torch::kBFloat16,"Second tensor must be BF16");
 TORCH_CHECK(a.sizes()==b.sizes(),"Tensor shapes differ");
 TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),"Contiguous tensors required");
 TORCH_CHECK(a.get_device()==b.get_device(),"Different CUDA devices");
 return compare_cuda(a,b);
}
std::vector<int64_t> shape_native(torch::Tensor a){
 TORCH_CHECK(a.is_cuda(),"CUDA tensor required");
 return std::vector<int64_t>(a.sizes().begin(),a.sizes().end());
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
 m.def("compare",&compare_native,"Exact BF16 bit comparison");
 m.def("shape",&shape_native,"Native tensor shape");
}
"""
CUDA_SRC=r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
__global__ void compare_bits_kernel(const uint16_t* a,const uint16_t* b,int64_t n,int* mismatches){
 int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n && a[i]!=b[i])atomicAdd(mismatches,1);
}
torch::Tensor compare_cuda(torch::Tensor a,torch::Tensor b){
 auto result=torch::zeros({1},torch::TensorOptions().dtype(torch::kInt32).device(a.device()));
 int64_t n=a.numel();
 if(n>0){
  int threads=256;
  int blocks=static_cast<int>((n+threads-1)/threads);
  auto stream=at::cuda::getCurrentCUDAStream(a.get_device());
  compare_bits_kernel<<<blocks,threads,0,stream.stream()>>>(
   reinterpret_cast<const uint16_t*>(a.data_ptr()),
   reinterpret_cast<const uint16_t*>(b.data_ptr()),
   n,result.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
 }
 return result;
}
"""
native_hash=sha_bytes((CPP+"\n"+CUDA_SRC).encode())
(ROOT/"native643.cpp").write_text(CPP)
(ROOT/"native643.cu").write_text(CUDA_SRC)
log("NATIVE_SOURCE_SHA256",native_hash)
log("[5/9] COMPILING NATIVE EXTENSION")
os.environ["MAX_JOBS"]="2"
build_dir=ROOT/"build";build_dir.mkdir(exist_ok=True)
t=time.perf_counter()
native=load_inline(name="akbascore_native643_v2",cpp_sources=[CPP],cuda_sources=[CUDA_SRC],
                   functions=None,extra_cflags=["-O2"],extra_cuda_cflags=["-O2"],
                   with_cuda=True,verbose=False,build_directory=str(build_dir))
compile_seconds=round(time.perf_counter()-t,3)
log("NATIVE_COMPILE_PASS",compile_seconds,"seconds")
log("[6/9] CUDA POSITIVE AND NEGATIVE CONTROLS")
controls={}
x=torch.tensor([1.,2.,3.,4.],dtype=torch.bfloat16,device="cuda")
y=x.clone()
controls["identical"]=int(native.compare(x,y).item())==0
y[2]=9.
controls["single_mutation"]=int(native.compare(x,y).item())==1
y[1]=8.
controls["double_mutation"]=int(native.compare(x,y).item())==2
try:
    native.compare(x,x[:2])
    controls["shape_guard"]=False
except Exception:controls["shape_guard"]=True
try:
    native.compare(x.float(),x.float())
    controls["dtype_guard"]=False
except Exception:controls["dtype_guard"]=True
try:
    native.compare(x.cpu(),x.cpu())
    controls["device_guard"]=False
except Exception:controls["device_guard"]=True
# BF16 signed-zero and NaN payloads must be checked by raw bits, not numerical equality.
raw_bits=torch.tensor([0x0000,0x8000,0x7FC1,0x7FC2],dtype=torch.int16)
bits_a=raw_bits.view(torch.bfloat16).to("cuda")
bits_b=bits_a.clone()
controls["special_bits_equal"]=int(native.compare(bits_a,bits_b).item())==0
bits_b[1]=torch.tensor(0.0,dtype=torch.bfloat16,device="cuda")
controls["signed_zero_detected"]=int(native.compare(bits_a,bits_b).item())==1
del x,y,bits_a,bits_b,raw_bits
log("CONTROLS",controls)
assert all(controls.values()),"CUDA_CONTROL_FAILURE"
log("[7/9] REAL PHYSICAL CUT3 CARTRIDGE AUDIT")
rng=random.Random(SEED)
sample=sorted(rng.sample(range(len(ENG.MANIFEST)),SAMPLE_N))
log("SAMPLED_IDS",sample)
records=[]
for ci in sample:
    row={"id":ci}
    try:
        t=time.perf_counter()
        manifest=ENG.MANIFEST[ci]
        path=ENG.POOL/manifest["file"]
        assert path.is_file(),"MISSING_PHYSICAL_FILE"
        assert path.stat().st_size==manifest["bytes"],"FILE_SIZE_MISMATCH"
        assert file_sha(path)==manifest["sha256"],"DISK_SHA256_MISMATCH"
        stored=torch.load(path,map_location="cpu",weights_only=True)
        assert isinstance(stored,(list,tuple)),"INVALID_CARTRIDGE_CONTAINER"
        assert len(stored)==28,"INVALID_LAYER_COUNT"
        tensor_count=0;total_bytes=0;mismatches=0
        shapes=[]
        for layer,pair in enumerate(stored):
            assert isinstance(pair,(tuple,list)) and len(pair)==2,"INVALID_KV_PAIR"
            k,v=pair
            assert torch.is_tensor(k) and torch.is_tensor(v),"INVALID_TENSOR"
            assert k.dtype==v.dtype==torch.bfloat16,"INVALID_DTYPE"
            assert k.ndim==v.ndim==4,"INVALID_RANK"
            assert tuple(k.shape)==tuple(v.shape),"KV_SHAPE_MISMATCH"
            assert k.shape[0]==1 and k.shape[1]==4 and k.shape[3]==128,"INVALID_QWEN_KV_ARCH"
            for kind,cpu_tensor in (("K",k),("V",v)):
                cpu_tensor=cpu_tensor.contiguous()
                original_bits=cpu_tensor.view(torch.int16).numpy().tobytes()
                original_digest=sha_bytes(original_bits)
                gpu_tensor=cpu_tensor.to("cuda",non_blocking=False).contiguous()
                gpu_shape=list(native.shape(gpu_tensor))
                assert gpu_shape==list(cpu_tensor.shape),"NATIVE_SHAPE_MISMATCH"
                # Independent CPU-origin bit reference, transferred to CUDA.
                reference=torch.from_numpy(np.frombuffer(original_bits,dtype=np.int16).copy())
                reference=reference.reshape(cpu_tensor.shape).view(torch.bfloat16).to("cuda")
                mismatch=int(native.compare(gpu_tensor,reference).item())
                assert mismatch==0,"CUDA_BITWISE_MISMATCH"
                returned=gpu_tensor.to("cpu").contiguous().view(torch.int16).numpy().tobytes()
                assert sha_bytes(returned)==original_digest,"GPU_ROUNDTRIP_HASH_MISMATCH"
                mismatches+=mismatch
                tensor_count+=1
                total_bytes+=cpu_tensor.numel()*cpu_tensor.element_size()
                shapes.append(gpu_shape)
                del gpu_tensor,reference
        assert tensor_count==56,"TENSOR_COUNT_MISMATCH"
        row.update(status="PASS",layers=28,tensors=tensor_count,
                   mismatches=mismatches,bytes=total_bytes,
                   sequence_length=int(stored[0][0].shape[-2]),
                   elapsed_seconds=round(time.perf_counter()-t,3))
        del stored
    except Exception as e:
        row.update(status="FAIL",error=repr(e))
        log("CARTRIDGE_ERROR",ci,traceback.format_exc()[-1600:])
    records.append(row)
    log("CARTRIDGE",ci,row["status"],"TENSORS",row.get("tensors"),
        "MISMATCHES",row.get("mismatches"),"SECONDS",row.get("elapsed_seconds"))
    save("TEST643_PROGRESS.json",records)
    gc.collect()
    torch.cuda.empty_cache()
log("[8/9] FROZEN MODEL AND ENGINE INTEGRITY")
after=INSTR.frozen()
weight_after=INSTR.full_weight_sha()
integrity={
 "source_blob_verified":source_blob==SOURCE_BLOB,
 "engine_sha_verified":ENGINE_SHA256==ENGINE_EXPECTED,
 "engine_lock_verified":all(ok for _,ok in ENGINE_LOCK),
 "weights_unchanged":weight_after==weight_before,
 "sentinel_unchanged":after["sentinel"]==sentinel_before,
 "zero_trainable":after["trainable_tensors"]==0,
 "model_eval":not after["training"],
 "no_optimizer":not after["optimizer"],
 "no_foreign_hooks":after["foreign_hooks"]==0,
 "no_residual_hooks":not after["residual_hooks"],
 "cuda_controls":all(controls.values())
}
passed=sum(r["status"]=="PASS" for r in records)
summary={
 "test":"TEST643",
 "phase":"NATIVE_CPP_CUDA_FOUNDATION",
 "utc":utc(),
 "gpu":GPU,
 "model":ENG.MODEL_ID,
 "engine_sha256":ENGINE_SHA256,
 "native_source_sha256":native_hash,
 "sampled_cartridges":len(records),
 "passed_cartridges":passed,
 "failed_cartridges":len(records)-passed,
 "verified_tensors":sum(r.get("tensors",0) for r in records),
 "native_bit_mismatches":sum(r.get("mismatches",0) for r in records),
 "compile_seconds":compile_seconds,
 "integrity":integrity,
 "limitations":[
  "The CUDA kernel performs exact BF16 comparison, not neural inference.",
  "PyTorch still loads .pt files and manages CUDA tensors.",
  "Native comparison validates data transfer, not relational reasoning.",
  "PCA128, FAISS IVF128 and PEAK50 remain in the original engine.",
  "The existing 1453-cartridge engine is rebuilt on a fresh runtime.",
  "No new cognitive synthesis capability is claimed by this test."
 ]
}
summary["status"]="PASS" if passed==SAMPLE_N and all(integrity.values()) else "FAIL"
result={"summary":summary,"controls":controls,"records":records,
        "source_sha256":source_sha,"source_blob":source_blob}
output=save("TEST643_RESULT.json",result)
log("[9/9] FINAL REPORT")
log("="*120)
for k,v in summary.items():
    if k not in ("integrity","limitations"):log(k,":",v)
log("INTEGRITY",json.dumps(integrity))
log("OUTPUT",output)
log("="*120)
if summary["status"]!="PASS":
    raise RuntimeError("TEST643_FAILED — RESULTS SAVED")
