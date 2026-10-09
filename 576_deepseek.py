
# TEST576-B — AKBASCORE MAM · DEEPSEEK BUILT-IN ARCHITECTURE / MLA STATIC AUDIT
import os,sys,inspect,hashlib,json,traceback,importlib
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch,transformers
from transformers import AutoConfig
from huggingface_hub import model_info
MODEL="deepseek-ai/DeepSeek-V2-Lite-Chat"
REV="85864749cd611b4353ce1decdb286193298f64c7"
print("="*110)
print("TEST576-B — AKBASCORE MAM · DEEPSEEK BUILT-IN STATIC AUDIT")
print("="*110)
print("PYTHON:",sys.version.split()[0],"TORCH:",torch.__version__,"TRANSFORMERS:",transformers.__version__)
print("GPU:",torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
gates={}
print("\n[1/7] PINNED CONFIG")
cfg=AutoConfig.from_pretrained(MODEL,revision=REV,trust_remote_code=True)
print("MODEL_TYPE:",cfg.model_type,"LAYERS:",cfg.num_hidden_layers,"HIDDEN:",cfg.hidden_size)
print("CONFIG_CLASS:",type(cfg).__module__,type(cfg).__name__)
gates["CONFIG"]=cfg.model_type=="deepseek_v2" and cfg.num_hidden_layers==27
print("\n[2/7] BUILT-IN CLASS RESOLUTION")
try:
    mod=importlib.import_module("transformers.models.deepseek_v2.modeling_deepseek_v2")
    cls=getattr(mod,"DeepseekV2ForCausalLM")
    print("CLASS:",cls.__module__,cls.__name__)
    print("SOURCE:",inspect.getfile(cls))
    print("INIT_SIGNATURE:",inspect.signature(cls.__init__))
    print("FORWARD_SIGNATURE:",inspect.signature(cls.forward))
    gates["BUILTIN_CLASS"]=cls.__module__.startswith("transformers.models.deepseek_v2")
except Exception as e:
    gates["BUILTIN_CLASS"]=False
    print("CLASS_ERROR:",type(e).__name__,str(e))
    traceback.print_exc(limit=6)
    raise
print("\n[3/7] ATTENTION / DECODER SOURCE INSPECTION")
names=["DeepseekV2Attention","DeepseekV2DecoderLayer","DeepseekV2Model","DeepseekV2RotaryEmbedding","DeepseekV2YarnRotaryEmbedding"]
for name in names:
    obj=getattr(mod,name,None)
    if obj is None:
        print(name,": NOT_EXPOSED")
        continue
    print(name,":",inspect.getfile(obj))
    if hasattr(obj,"forward"):print(" FORWARD:",inspect.signature(obj.forward))
    if hasattr(obj,"__init__"):print(" INIT:",inspect.signature(obj.__init__))
print("\n[4/7] MLA / ROPE SOURCE MARKERS")
source=inspect.getsource(mod)
markers=["kv_lora_rank","qk_nope_head_dim","qk_rope_head_dim","kv_a_proj_with_mqa","kv_b_proj","q_proj","q_a_proj","q_b_proj","rotary_emb","apply_rotary_pos_emb","past_key_values","cache_position","DynamicCache","Yarn","YaRN","flash_attention"]
for marker in markers:
    count=source.count(marker)
    print(f"{marker}: {count}")
gates["MLA_MARKERS"]=all(x in source for x in ["kv_lora_rank","qk_rope_head_dim","kv_a_proj_with_mqa"])
print("\n[5/7] CACHE / FORWARD CONTRACT")
for name in ["DeepseekV2Attention","DeepseekV2Model","DeepseekV2ForCausalLM"]:
    obj=getattr(mod,name,None)
    if obj is None:continue
    try:
        s=inspect.getsource(obj.forward)
        print("FORWARD",name,"LINES:",len(s.splitlines()))
        for i,line in enumerate(s.splitlines(),1):
            if any(x in line for x in ["past_key_values","past_key_value","cache_position","position_ids","position_embeddings","use_cache","attn_implementation","output_attentions","Cache("]):
                print(f" {i:03d}: {line.strip()[:190]}")
    except Exception as e:print("SOURCE_ERROR:",name,repr(e))
print("\n[6/7] SOURCE FINGERPRINT / GPU ALLOCATION")
sha=hashlib.sha256(source.encode("utf-8")).hexdigest()
print("BUILTIN_MODELING_SOURCE_SHA256:",sha)
print("SOURCE_LINES:",len(source.splitlines()))
print("CUDA_ALLOCATED_GIB:",round(torch.cuda.memory_allocated()/2**30,4) if torch.cuda.is_available() else 0)
print("WEIGHTS_DOWNLOADED_BY_THIS_TEST: NO")
print("MODEL_INSTANTIATED: NO")
gates["STATIC_ONLY"]=True
print("\n[7/7] GATES")
for k,v in gates.items():print("GATE",k,"PASS" if v else "FAIL")
print("STATUS:","PASS — STATIC ARCHITECTURE AUDIT" if all(gates.values()) else "FAIL — INSPECT OUTPUT")
print("NEXT: CHECK BUILT-IN WEIGHT LOADING COMPATIBILITY BEFORE GPU FORWARD")
print("="*110)
