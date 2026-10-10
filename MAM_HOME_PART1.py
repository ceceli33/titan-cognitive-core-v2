# =====================================================================================================================================
# AKBASCORE MAM-BÇ · EXTERNAL ASSOCIATIVE MEMORY FOR A FROZEN LANGUAGE MODEL — QWEN2.5-7B · CUT3 + FRAM · PART 1 / 3 · ENGINE
# Discovered and developed by Mustafa Akbaş · AkbasCore AI Teknoloji · 10 October 2026
# Run PART 1 → PART 2 → PART 3 as three consecutive Google Colab cells in the SAME runtime (GPU: A100 40 GB recommended).
# PART 1 embeds 610.py (the TEST610 demo-acceptance program) VERBATIM, verifies its git blob SHA-1, and executes it in its own module
# namespace up to the start of its acceptance stage [6/7]. 610.py itself downloads 566.py and 575.py from the repository and verifies
# their git blobs, loads frozen Qwen2.5-7B-Instruct, captures the 96 native cartridge representations, locks the FRAM configuration on
# calibration cases 0–11, evaluates the holdout, computes the locked reference ranking and runs its baseline CUT3 audit. The save /
# reload / retrieve helpers are the RAW COLAB DEMO helper section, embedded VERBATIM (SHA-256 checked). PART 2 adds the fail-closed
# measurement pipeline and the six figures. PART 3 starts the Gradio demonstration.
# AKBASCORE RESEARCH SOFTWARE LICENSE · Copyright © 2026 Mustafa Akbaş · https://github.com/ceceli33/titan-cognitive-core-v2
#
# ENGINE         : 610.py, embedded byte-for-byte as ENGINE_SRC (git blob SHA-1 and SHA-256 verified). Executed unchanged up to the line
#                  "print('[6/7] Demo acceptance …". The remaining acceptance stage [6/7]–[7/7] is executed verbatim only by the optional
#                  live acceptance replay (PART 3). The memory and retrieval algorithms are not modified, recalibrated or re-implemented.
# HELPERS        : RAW COLAB DEMO helper section (retrieve · save_session · load_session · compare_mem_exact), verbatim, executed in the
#                  same namespace. The demo pipeline calls the engine functions build_cut3 / validate_cut3 / answer / pair_scores and
#                  these helpers; it never computes memory, retrieval scores or answers itself.
# INSTRUMENTATION: (1) a recording-only forward pre-hook on the causal-LM module, attached only while an engine call runs; it records
#                  the kind and length of every model input and the installed cache length; (2) an append observer, attached only while
#                  build_cut3 runs: it wraps the engine's init/append, calls them unchanged and checks that every previously
#                  consolidated K/V row is bitwise identical after each append; (3) an all-parameter weight guard plus the full-weight
#                  SHA-256 computed by 610.py at start-up; (4) re-derivation of every reported quantity from raw records.
# =====================================================================================================================================
import os,sys,io,re,json,math,time,html,random,shutil,hashlib,zipfile,platform,textwrap,threading,subprocess,importlib.util,traceback,gc,types,contextlib
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import quote
from collections import Counter
#<<CONST_BEGIN>>
DEMO_TITLE="AKBASCORE MAM-BÇ · External associative memory for a frozen language model"
DEMO_SHORT="AKBASCORE MAM-BC · Qwen2.5-7B · CUT3 + FRAM demonstration"
MODEL_FAMILY="Qwen2.5-7B-Instruct"
AUTHOR="Mustafa Akbaş";ORG="AkbasCore AI Teknoloji";DATE_TXT="10 October 2026";DATE_ISO="2026-10-10";AUTHOR_PLACE="Mersin, Türkiye"
COPYRIGHT="Copyright © 2026 Mustafa Akbaş";LICENSE_NAME="AKBASCORE RESEARCH SOFTWARE LICENSE"
LICENSE_URL="https://github.com/ceceli33/titan-cognitive-core-v2"
REPO_RAW="https://github.com/ceceli33/titan-cognitive-core-v2/blob/main/"
ENGINE_URL=REPO_RAW+"610.py";ENGINE_LOG_URL=REPO_RAW+"610.log";BASE_URL=REPO_RAW+"566.py";LOCK_URL=REPO_RAW+"575.py";LOG575_URL=REPO_RAW+"575.log"
PARADIGM="Removable numerical memory for a frozen language model"
DISCOVERY="Discovered and developed by Mustafa Akbaş"
SCI_QUESTION=("Can a language model answer from numerical memory cartridges that are stored outside the model, selected and attached to it, "
              "without changing its weights and without giving it the source text again?")
# ---- earlier public records (cited by DOI only; their values are not results of this demonstration) ----
PRIOR_PUB=dict(title="AKBASCORE MAM v1.0 — Source-Free Persistent Memory for Frozen LLMs via DC6 Consolidation (Mistral-7B)",doi="10.5281/zenodo.23245358",
               url="https://doi.org/10.5281/zenodo.23245358",date="8 October 2026")
QWEN_PUB=dict(label="AKBASCORE MAM v1.1 — Qwen2.5-7B-Instruct (DC3) record",doi="10.5281/zenodo.23257347",url="https://doi.org/10.5281/zenodo.23257347",date="9 October 2026")
PRIORITY=("Public technical demonstration of AKBASCORE MAM-BÇ: frame-matched residual access (FRAM) to an external bank of numerical cartridges, "
          "attached to frozen Qwen2.5-7B-Instruct through the CUT3 memory. Mustafa Akbaş (AkbasCore AI Teknoloji) discovered and developed this "
          f"numerical-memory mechanism (earlier public records: DOI {PRIOR_PUB['doi']}, {PRIOR_PUB['date']}; DOI {QWEN_PUB['doi']}, {QWEN_PUB['date']}); "
          "this record is dated 10 October 2026.")
CORE_MESSAGE=["The frozen model writes each fact once as numbers; the numbers are kept outside the model.",
              "FRAM selects the cartridges whose numbers match the question.",
              "The selected cartridges are consolidated append-only into one CUT3 memory.",
              "The memory is saved to a file, reloaded and checked bit for bit.",
              "The question is answered from the memory; the source text is not given to the model.",
              "Model weights are not changed."]
NOT_VALIDATED=("Compressed superposition memory is not validated: the 6,144-item scaling experiments do not show that 6,144 independent CUT3 cartridges work end to end.")
ENGINE_SCOPE_610="96 native CUT3 cartridges; 6144 residual facts and compressed superposition are NOT certified by this test"
CROSS_ALLOWED=(PRIOR_PUB["title"],NOT_VALIDATED,ENGINE_SCOPE_610,"6144 compressed not validated","compressed superposition")
ENGINE_FILE="610.py"
ENGINE_SHA256_EXPECTED="a62611a4ffd91c2fcf03796c31da2af11ccd8141345655b562f3a614568af3e7"
ENGINE_BLOB_EXPECTED="4a92e751fe0ea39c89b792338a0f2342547ce6d3"
ENGINE_SPLIT="print('[6/7] Demo acceptance"
HELPER_FILE="RAW COLAB DEMO helper section"
HELPER_SHA256_EXPECTED="0e0a8c6e96221d1142a1d65ce95531e9e106f0a07f899100a63c5d5be0e2e979"
RAW_DEMO_FILE_SHA256="499b63b0dcdbe59a44651500fb29c07a9e1799bec72aa0f9bff5d0a5223f1640"
SOURCE_BLOBS_EXPECTED={"566.py":"b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71","575.py":"569c92c61224ea408cdb6e81dbe45dde06dc05fd"}
EXPECTED_PANEL_SHA="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
WEIGHT_SHA_EXPECTED="b441b005d826d45f2778291696ac5ae22dfb3421defb37c539904d8d38f93786"
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";ARCH=(28,3584,28,4,128);CUT_EXPECTED=3;PREFIX_EXPECTED=29;N_CASES=24;N_CART=96;TOP_K=5
FRAM_LOCK=(3,0,8.0);M_REF=8192;D_REF=8192;TRAIN_CASES=tuple(range(12));HOLD_CASES=tuple(range(12,24))
CANON_GPU="NVIDIA A100-SXM4-40GB"
def arch_txt(a=ARCH):return f"{a[0]} layers · hidden {a[1]} · {a[2]} Q heads · {a[3]} KV heads · head {a[4]}"
def arch_short(a=ARCH):return f"{a[0]} L · {a[1]} · {a[2]} Q / {a[3]} KV · {a[4]}"
def upper_txt(cut=CUT_EXPECTED,nl=ARCH[0]):return f"layers {cut+1}–{nl-1}"
def split_of(case):return"CAL"if int(case)in TRAIN_CASES else"HOLD"
# ---- sealed records (read from the archived experiment logs at build time; never produced or recomputed by this program) ----
TEST610=dict(name="TEST610",file="610.py",blob=ENGINE_BLOB_EXPECTED,log="610.log",log_sha256="a5a46a0390a4d3459aef606050689b0e9a4b47b687a8d8ba3f72a2c016da39f7",result_sha="7d033e8a5c262f7861c8ada4efb07245e9bef4450715781e6d0a853c94d194f4",
    decision="DEMO_READY_96",final={'test': '610', 'demo_decision': 'DEMO_READY_96', 'criteria_passed': 15, 'criteria_total': 15, 'holdout_correct': 12, 'holdout_total': 12, 'all_panel_correct': 23, 'all_panel_total': 24, 'reference_cartridges': 96, 'seconds': 191.36, 'scope': '96 native CUT3 cartridges; 6144 residual facts and compressed superposition are NOT certified by this test'},criteria={'holdout_fram_r1_12': True, 'holdout_fram_r5_12': True, 'holdout_reference_retrieval_12': True, 'holdout_cut3_correct_12': True, 'baseline_cut3_correct_12': True, 'final_cut3_correct_12': True, 'no_answer_changes': True, 'all24_repeat_deterministic': True, 'all24_memory_shapes_valid': True, 'rebuild_bitwise_4': True, 'append_prefix_4': True, 'weight_unchanged': True, 'sentinel_unchanged': True, 'no_trainables': True, 'no_hooks': True},cases=[{'case': 0, 'split': 'CAL', 'target': 0, 'present': True, 'correct': True, 'repeat': True, 'shape': True, 'answer': 'Melket'}, {'case': 1, 'split': 'CAL', 'target': 6, 'present': True, 'correct': True, 'repeat': True, 'shape': True, 'answer': 'Branik'}, {'case': 2, 'split': 'CAL', 'target': 9, 'present': True, 'correct': True, 'repeat': True, 'shape': True, 'answer': 'Varos'}, {'case': 3, 'split': 'CAL', 'target': 15, 'present': True, 'correct': True, 'repeat': True, 'shape': True, 'answer': 'Belnor'}, {'case': 4, 'split': 'CAL', 'target': 16, 'present': True, 'correct': True, 'repeat': True, 'shape': True, 'answer': 'Selora'}, {'case': 5, 'split': 'CAL', 'target': 22, 'present': True, 'correct': True, 'repeat': True, 'shape': True, 'answer': 'Pareth'}, {'case': 6, 'split': 'CAL', 'target': 25, 'present': True, 'correct': True, 'repeat': True, 'shape': True, 'answer': 'Keldin'}, {'case': 7, 'split': 'CAL', 'target': 31, 'present': True, 'correct': True, 'repeat': True, 'shape': True, 'answer': 'Feron'}, {'case': 8, 'split': 'CAL', 'target': 32, 'present': True, 'correct': False, 'repeat': True, 'shape': True, 'answer': 'Narev'}, {'case': 9, 'split': 'CAL', 'target': 38, 'present': True, 'correct': True, 'repeat': True, 'shape': True, 'answer': 'Dorin'}, {'case': 10, 'split': 'CAL', 'target': 41, 'present': True, 'correct': True, 'repeat': True, 'shape': True, 'answer': 'Valen'}, {'case': 11, 'split': 'CAL', 'target': 47, 'present': True, 'correct': True, 'repeat': True, 'shape': True, 'answer': 'Mirev'}, {'case': 12, 'split': 'HOLD', 'target': 48, 'present': True, 'correct': True, 'repeat': True, 'shape': True, 'answer': 'Solith'}, {'case': 13, 'split': 'HOLD', 'target': 54, 'present': True, 'correct': True, 'repeat': True, 'shape': True, 'answer': 'Toren'}, {'case': 14, 'split': 'HOLD', 'target': 57, 'present': True, 'correct': True, 'repeat': True, 'shape': True, 'answer': 'Corven'}, {'case': 15, 'split': 'HOLD', 'target': 63, 'present': True, 'correct': True, 'repeat': True, 'shape': True, 'answer': 'Dervan'}, {'case': 16, 'split': 'HOLD', 'target': 64, 'present': True, 'correct': True, 'repeat': True, 'shape': True, 'answer': 'Verith'}, {'case': 17, 'split': 'HOLD', 'target': 70, 'present': True, 'correct': True, 'repeat': True, 'shape': True, 'answer': 'Maros'}, {'case': 18, 'split': 'HOLD', 'target': 73, 'present': True, 'correct': True, 'repeat': True, 'shape': True, 'answer': 'Vindor'}, {'case': 19, 'split': 'HOLD', 'target': 79, 'present': True, 'correct': True, 'repeat': True, 'shape': True, 'answer': 'Barel'}, {'case': 20, 'split': 'HOLD', 'target': 80, 'present': True, 'correct': True, 'repeat': True, 'shape': True, 'answer': 'Mervor'}, {'case': 21, 'split': 'HOLD', 'target': 86, 'present': True, 'correct': True, 'repeat': True, 'shape': True, 'answer': 'Calvenor'}, {'case': 22, 'split': 'HOLD', 'target': 89, 'present': True, 'correct': True, 'repeat': True, 'shape': True, 'answer': 'Mardin'}, {'case': 23, 'split': 'HOLD', 'target': 95, 'present': True, 'correct': True, 'repeat': True, 'shape': True, 'answer': 'Karev'}],negatives=[{'case': 12, 'target_absent': True, 'gold_leak': False, 'raw': 'Malvek'}, {'case': 15, 'target_absent': True, 'gold_leak': False, 'raw': 'Nareth'}, {'case': 18, 'target_absent': True, 'gold_leak': False, 'raw': 'Marvek'}, {'case': 21, 'target_absent': True, 'gold_leak': False, 'raw': 'Sareth'}],
    reference={'hold_r1': 11, 'hold_r5': 12},fram={'hold_r5': 12, 'hold_r1': 12, 'all_r5': 24, 'hub_max': 4, 'self_r1': 96},gpu=CANON_GPU,torch="2.11.0+cu130",transformers="5.18.0",weight_sha=WEIGHT_SHA_EXPECTED,panel_sha=EXPECTED_PANEL_SHA)
TEST575=dict(name="TEST575",lock="76a28d6f10b2c1e431311629b8b3e74f6bceb7048a6e84d5295a35c86cfce4d1",result_sha="579b3fb45f6fc7987694d21a313bb2980b279beb7a7e2de4d23aa69999b8581e",
    cases=24,positions=3,total=72,
    arms={"JOINT":72,"INCR_DC3":72,"BATCH_DC3":72,"NATIVE_INDEP":26,"BATCH_BLOCK3":28},
    counterfactual=dict(upper_kv_changed=72,early_kv_identical=72,answer_changed=50,total=72))
ARM_DESC={"JOINT":"facts processed together as text (upper bound, not a memory)","INCR_DC3":"append-only incremental CUT3 consolidation",
          "BATCH_DC3":"all five cartridges consolidated in one pass","NATIVE_INDEP":"independently written K/V, no consolidation (control)",
          "BATCH_BLOCK3":"counterfactual: cross-cartridge attention blocked in layers 4–27"}
SCOPE={
 "demonstrated":[
  "At start-up the frozen model's layer-3 residual of each of the 96 facts is captured once and kept as the cartridge bank's retrieval key (FRAM). The configuration (layer 3, rank 0, β = 8) is locked on calibration cases 0–11 before any CUT3 memory is built.",
  "For the selected case the five top-ranked cartridges of the locked reference ranking are written by the engine (one token-id forward of the instruction prefix + that cartridge's own fact) and consolidated append-only into one CUT3 memory; earlier K/V rows are compared bitwise after every append.",
  "The question is answered from the numerical memory: the query forward receives only the question-template tokens over the installed memory.",
  "The CUT3 memory is saved to a file, its SHA-256 is recomputed, the file is reloaded, the reloaded K/V tensors are compared bitwise with the original in all 28 layers, and the answer is generated again from the reloaded memory.",
  "Negative control in every run: the same question over the memory built without the target cartridge; leakage of the expected answer is reported.",
  "Model weights are unchanged (all-parameter guard before and after every run; full-weight SHA-256 at start-up equal to the reference)."],
 "archived":[
  "TEST610 (sealed demo acceptance, 610.log): decision DEMO_READY_96 · 15/15 acceptance criteria · holdout 12/12 · whole panel 23/24 (one incorrect answer, case 08) · FRAM holdout Recall@1 12/12, Recall@5 12/12 · self-retrieval 96/96 · four negative controls without leakage of the expected answer · weights unchanged · trainable parameters 0 · hooks 0.",
  "TEST575 (sealed, 575.log; a different experiment): JOINT 72/72 · INCR_DC3 72/72 · BATCH_DC3 72/72 · counterfactual upper-layer K/V changed 72/72 · answer changed 50/72."],
 "not_established":[
  NOT_VALIDATED,
  "A persistent store of all 96 consolidated CUT3 cartridges: in this implementation the five selected cartridges are written from their facts when a session memory is built; the saved and reloaded object is the five-cartridge session memory.",
  "The four TEST610 negative controls show no leakage of the expected answer; four cases do not exclude every alternative causal explanation.",
  "Free-text questions outside the locked panel are answered but not scored.",
  "Memory populations larger than 96 cartridges and inference that combines several cartridges are not tested by this demonstration.",
  "Other language models: this demonstration runs Qwen2.5-7B-Instruct only."],
 "notes":[
  "Retrieval for the 24 panel questions uses the locked reference ranking of TEST610: positive random features (M = 8192) superposed with D = 8192 random sign codes over the 96 cartridges. At this size the superposed matrix is larger than the features it encodes; it is a ranking procedure, not a memory saving. The exact FRAM kernel is shown alongside and is used for free-text questions.",
  "Expected answers come from panel construction and are audit metadata; they are not an input to retrieval, writing, consolidation or answering.",
  "Engine: 610.py verbatim (git blob "+ENGINE_BLOB_EXPECTED[:12]+"…), executed up to its acceptance stage; it downloads 566.py and 575.py and verifies their git blobs. The save / reload / retrieve helpers are the RAW COLAB DEMO helper section, verbatim.",
  "Hashes are artifact-integrity seals; they are not scientific proof or third-party verification."]}
class AuditFail(RuntimeError):pass
def jsafe(o):
    if isinstance(o,dict):return{str(k):jsafe(v)for k,v in o.items()}
    if isinstance(o,(list,tuple)):return[jsafe(v)for v in o]
    if hasattr(o,"item")and not isinstance(o,(str,bytes)):
        try:o=o.item()
        except Exception:pass
    if isinstance(o,float):return o if math.isfinite(o)else f"non-finite:{o}"
    if isinstance(o,(str,int,bool))or o is None:return o
    return str(o)
def canon(o):return json.dumps(jsafe(o),sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")
def norm_ans(s):return " ".join(re.sub(r"[^\w\s-]"," ",str(s).casefold().strip(),flags=re.UNICODE).split())
def hit_ans(s,g):return re.search(r"(?<!\w)"+re.escape(norm_ans(g))+r"(?!\w)",norm_ans(s))is not None
def git_blob_sha1(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
# ---- claim-discipline scanner: applied to every generated text (figures, logs, JSON). Panel names are removed first. ----
ALLOWED_TESTS={"566","575","606","610"}
CLAIM_RULES=[("capacity-overclaim",r"(?i)\bunlimited\b|\binfinite\b"),("universal-claim",r"(?i)\buniversal\b"),("human-like-claim",r"(?i)human-like"),
    ("model-independence-claim",r"(?i)model-independent"),("all-transformers-claim",r"(?i)all transformers|every transformer|all (?:large )?language models"),("solved-claim",r"(?i)\bsolved\b"),
    ("perfect-claim",r"(?i)\bperfect\b"),("world-first-claim",r"(?i)world'?s first|first in the world|independently proven|third-party verified"),
    ("legal-status-claim",r"(?i)\bpatent(ed|-pending)\b|non-obvious|\bpatentab"),("metaphor",r"(?i)\bbrain\b|\bquantum\b|\bneuron"),
    ("complexity-claim",r"O\(1\)"),("compression-wording",r"(?i)\bcompress")]
# ---- foreign-model leak scanner: no value of the earlier Mistral record may appear as a Qwen value. Only the citation string is exempt. ----
LEAK_RULES=[("mistral-name",r"(?i)mistral"),("mistral-cut",r"\bH6\b|\bDC6\b|\b(?:layers?|L)\s?0[–-]6\b|\bL7[–-]31\b|\blayers 7[–-]31\b"),
    ("mistral-tests",r"TEST\s?56[03]\b"),("mistral-arch-words",r"\b32 (?:layers|L\b|Q)|\b8 KV\b|\b27-token")]
LEAK_RULES_NUM=[("mistral-hidden",r"(?<![\d.])4096(?![\d.])")]
def _scan_norm(text):
    for a in CROSS_ALLOWED:text=text.replace(a,"<cited>")
    text=re.sub(r"[0-9a-fA-F]{12,}","<hex>",text);return text
def claim_scan(name,text,allow=(),numeric=False):
    for a in allow:
        if a:text=text.replace(a,"<allowed>")
    text=_scan_norm(text)
    hits=[(name,lab,m.group(0))for lab,p in CLAIM_RULES for m in re.finditer(p,text)]
    hits+=[(name,"unexpected-test-number",m.group(0))for m in re.finditer(r"TEST\s?(\d+)",text)if m.group(1)not in ALLOWED_TESTS]
    hits+=[(name,"foreign-model-"+lab,m.group(0))for lab,p in LEAK_RULES+(LEAK_RULES_NUM if numeric else[])for m in re.finditer(p,text)]
    return hits
def strip_words(text,words):
    ws=sorted({w for w in words if w and len(w)>=2},key=len,reverse=True)
    if not ws:return text
    return re.sub(r"(?<![A-Za-z])(?:"+"|".join(map(re.escape,ws))+r")(?![A-Za-z])","",text)
def cpu_selftest():
    n=0
    assert hit_ans("Melket","Melket") and hit_ans("The capital is Melket.","melket") and not hit_ans("Melketon","Melket");n+=1
    ok_txt=("Qwen2.5-7B-Instruct TEST566 TEST575 TEST606 TEST610 CUT3 FRAM 96 cartridges layer 3 hidden 3584 28 layers 23/24 15/15 "+PARADIGM+" "+DISCOVERY+" "+PRIORITY+" "+SCI_QUESTION+" "
            +PRIOR_PUB["title"]+" "+" ".join(SCOPE["archived"]+SCOPE["not_established"]+SCOPE["demonstrated"]+SCOPE["notes"]+CORE_MESSAGE)+" "+ENGINE_SCOPE_610)
    assert claim_scan("t",ok_txt,numeric=True)==[],claim_scan("t",ok_txt,numeric=True);n+=1
    for b in("unlimited memory","universal memory","human-like memory","model-independent","works on all transformers","works in all language models","memory solved","perfect recall",
             "world's first","patent-pending","third-party verified","brain","quantum","O(1)","compressed memory","TEST528","TEST560","TEST605","Mistral","H6 residual","DC6","layers 0–6","32 layers","8 KV heads","27-token prefix"):
        assert claim_scan("t",b),b;n+=1
    assert claim_scan("t","hidden 4096",numeric=True) and not claim_scan("t","hidden 4096") and not claim_scan("t","t=0.40963 sha 4096ab12cd34ef56",numeric=True);n+=1
    assert strip_words("Zorvan BRAIN carried",["BRAIN"])=="Zorvan  carried";n+=1
    f=TEST610["final"];assert f["criteria_passed"]==f["criteria_total"]==15 and f["all_panel_correct"]==23 and f["holdout_correct"]==12 and f["demo_decision"]==TEST610["decision"]=="DEMO_READY_96";n+=1
    assert len(TEST610["cases"])==24 and sum(int(r["correct"])for r in TEST610["cases"])==23 and sum(int(r["correct"])for r in TEST610["cases"]if r["split"]=="HOLD")==12;n+=1
    assert all(TEST610["criteria"].values())and len(TEST610["criteria"])==15 and len(TEST610["negatives"])==4 and not any(r["gold_leak"]for r in TEST610["negatives"]);n+=1
    assert [r["case"]for r in TEST610["cases"]]==list(range(24))and all(r["present"]for r in TEST610["cases"]);n+=1
    return n
#<<CONST_END>>
#<<ENGINE_BEGIN>>
for _m,_p in[("torch","torch"),("transformers","transformers"),("accelerate","accelerate"),("gradio","gradio"),("matplotlib","matplotlib"),("PIL","pillow")]:
    if importlib.util.find_spec(_m)is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",_p])
import numpy as np,torch,transformers,gradio as gr
QUIET=False
def say(*a):
    if not QUIET:print(*a,flush=True)
def utc_now():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def local_now():return datetime.now().astimezone().isoformat(timespec="milliseconds")
say("="*140);say(DEMO_TITLE+" — "+MODEL_FAMILY+" · PART 1 / 3 · ENGINE");say(f"{DISCOVERY} · {ORG} · {DATE_TXT} · {AUTHOR_PLACE}");say(f"{LICENSE_NAME} · {COPYRIGHT} · {LICENSE_URL}");say("="*140)
say(f"[1/6] CPU self-test PASS ({cpu_selftest()} checks)")
# ---------------- 610.py — embedded verbatim (do not edit; its git blob SHA-1 and SHA-256 are verified below) ----------------
ENGINE_SRC=r'''# TEST610 — AKBASCORE MAM-BÇ · FRAME-MATCHED RESIDUAL ACCESS + POSITIVE RANDOM FEATURES SUPERPOSITION + CUT3 AUDIT + CODE INTERFERENCE DIAGNOSTICS
# Frozen working TEST608 backbone; CUT3 differential audit + signal/noise decomposition, unchanged motor.
import os,sys,subprocess,importlib.util,urllib.request,hashlib,json,time,math,random,inspect
for m in ('torch','transformers','accelerate'):
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,'-m','pip','install','-q',m])
import torch,numpy as np,transformers
URL='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/'
BLOBS={'566.py':'b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71','575.py':'569c92c61224ea408cdb6e81dbe45dde06dc05fd'}
def fetch_locked(name):
    with urllib.request.urlopen(urllib.request.Request(URL+name,headers={'User-Agent':'AKBASCORE-TEST610'}),timeout=120) as f:b=f.read()
    got=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
    if got!=BLOBS[name]:raise RuntimeError(f'SOURCE MISMATCH {name}: {got}')
    print('SOURCE_VERIFIED',name,got,flush=True);return b.decode()
s575=fetch_locked('575.py');s566=fetch_locked('566.py')
if 'BASE_BLOB="b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71"' not in s575:raise RuntimeError('575 LINK MISMATCH')
MARK='# TEST566 ADDITIONS — original TEST564 checkpoint/init/append/answer remain unchanged.'
if s566.count(MARK)!=1:raise RuntimeError('566 BOUNDARY MISMATCH')
exec(compile(s566.split(MARK,1)[0],'566.py','exec'),globals())
TEST='610';START=time.perf_counter();LAYERS=[3,8,16,24];RANKS=[0,64,128];BETAS=[0.0,8.0];TRAIN=list(range(12));HOLD=list(range(12,24));EXPECTED='b441b005d826d45f2778291696ac5ae22dfb3421defb37c539904d8d38f93786';OUT='/content/AKBASCORE_TEST610_DEMO_ACCEPTANCE.json';REC={'checks':[],'calibration':[],'cases':[],'pilot':[],'cut3_audits':{}}
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def check(name,ok,**info):
    REC['checks'].append(dict(name=name,ok=bool(ok),**info));print('PASS' if ok else 'FAIL',name,info,flush=True)
    if not ok:raise RuntimeError('TEST610 INTEGRITY '+name)
def weight_sha():
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode());h.update(str(tuple(p.shape)).encode());h.update(str(p.dtype).encode())
        for part in p.detach().contiguous().view(torch.uint8).reshape(-1).split(16*1024*1024):h.update(part.cpu().numpy().tobytes())
    return h.hexdigest()
def span_tokens(text,begin,end):
    e=tok(text,add_special_tokens=False,return_offsets_mapping=True);ids=e['input_ids'];off=e['offset_mapping'];span=[i for i,(a,b) in enumerate(off) if a<end and b>begin]
    if not span or span!=list(range(span[0],span[-1]+1)):raise RuntimeError('SPAN INVALID')
    return ids,span
@torch.no_grad()
def capture_residual(text,begin,end):
    ids,span=span_tokens(text,begin,end);cap={};hooks=[]
    for li in LAYERS:
        def hook(mod,args,out,li=li):cap[li]=(out[0] if isinstance(out,(tuple,list)) else out)[:,span,:].detach().float().cpu().clone()
        hooks.append(model.model.layers[li].register_forward_hook(hook))
    try:
        x=torch.tensor([ids],device=DEVICE);pos=torch.arange(len(ids),device=DEVICE)
        model(input_ids=x,attention_mask=torch.ones_like(x),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=False,return_dict=True)
    finally:
        for h in hooks:h.remove()
    if set(cap)!=set(LAYERS):raise RuntimeError('RESIDUAL CAPTURE INCOMPLETE')
    return {li:cap[li][0] for li in LAYERS},dict(n=len(span),span=[span[0],span[-1]+1],boundary=bool(span[0]<P))
def cartridge_rep(ci):
    fact=CAR[ci]['fact'];text=source(fact);begin=len(PREFIX);end=begin+len(fact)
    if text[begin:end]!=fact:raise RuntimeError('FACT SPAN')
    return capture_residual(text,begin,end)[0]
def question_rep(q):
    # Same source-prefix and local position as facts; no fact/source supplied during ASK.
    text=PREFIX+q+'\n\n';begin=len(PREFIX);end=begin+len(q)
    return capture_residual(text,begin,end)
def unit_norm(x):return torch.nn.functional.normalize(x,dim=-1,eps=1e-9)
def transform_fit(X,rank):
    mu=X.mean(0);C=X-mu
    if rank==0:return mu,None
    # SVD of the store only: no question labels and no holdout queries in PCA.
    _,s,vh=torch.linalg.svd(C,full_matrices=False)
    r=min(rank,vh.shape[0]);basis=vh[:r].T.contiguous();scale=(s[:r].square()/max(len(X)-1,1)).clamp_min(1e-4).rsqrt()
    return mu,basis*scale
def apply(X,fit):
    mu,proj=fit;Y=X-mu
    if proj is not None:Y=Y@proj
    return unit_norm(Y)
def pair_scores(Q,K,beta):
    # Q:[q,r], K:[t,r]; per query token log-mean-exp over cartridge tokens.
    sim=Q@K.T
    if beta==0:return float(sim.mean().item())
    return float((torch.logsumexp(sim*beta,dim=1)-math.log(len(K))).mean().item()/beta)
def rankings(scores):return sorted(range(len(scores)),key=lambda i:(-scores[i],i))
print('='*126);print('TEST610 — FINAL DEMO ACCEPTANCE · FROZEN CUT3 + FRAM 96 CARTRIDGES');print('='*126)
if '__file__' in globals() and os.path.isfile(__file__):print('SCRIPT_SHA256',hashlib.sha256(open(__file__,'rb').read()).hexdigest(),flush=True)
check('PANEL_SHA',PANEL_SHA=='69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45',sha=PANEL_SHA)
check('ARCH',(NL,H,QH,KVH,HD)==(28,3584,28,4,128));check('FROZEN',not any(p.requires_grad for p in model.parameters()))
WB=weight_sha();SB=sentinel();check('WEIGHT_SHA_REFERENCE',WB==EXPECTED,sha=WB)
print('[1/8] Capture native residuals for 96 cartridges...',flush=True)
KS={li:[] for li in LAYERS}
for ci in range(len(CAR)):
    v=cartridge_rep(ci)
    for li in LAYERS:KS[li].append(v[li])
    if ci%16==0:print('CARTRIDGE',ci+1,'/',len(CAR),flush=True)
check('CARTRIDGES',len(CAR)==96 and all(len(KS[li])==96 for li in LAYERS))
print('[2/8] Question representations in fact frame...',flush=True)
QS={};QINFO={}
for wi in range(24):
    typ=CAR[PANEL[wi]['target']]['type'];q=W[wi][typ];QS[wi],QINFO[wi]=question_rep(q)
    print('QUESTION',wi,'TARGET',PANEL[wi]['target'],'TOKENS',QINFO[wi]['n'],flush=True)
print('[3/8] Label-free PCA and calibration...',flush=True)
FEATURES={};CONFIGS=[]
for li in LAYERS:
    X=torch.cat(KS[li],dim=0)
    for rank in RANKS:
        fit=transform_fit(X,rank);KZ=[apply(v,fit) for v in KS[li]];QZ={wi:apply(QS[wi][li],fit) for wi in range(24)}
        FEATURES[(li,rank)]=(KZ,QZ)
        for beta in BETAS:CONFIGS.append((li,rank,beta))
def evaluate(wi,cfg):
    li,rank,beta=cfg;KZ,QZ=FEATURES[(li,rank)];scores=[pair_scores(QZ[wi],k,beta) for k in KZ];order=rankings(scores);target=PANEL[wi]['target'];r=order.index(target)+1
    return dict(case=wi,target=target,rank=r,hit1=int(r==1),hit5=int(r<=5),top5=order[:5],target_score=scores[target],top_score=scores[order[0]])
for idx,cfg in enumerate(CONFIGS):
    rows=[evaluate(wi,cfg) for wi in TRAIN];REC['calibration'].append(dict(cfg=list(cfg),hit5=sum(x['hit5'] for x in rows),hit1=sum(x['hit1'] for x in rows),rank_sum=sum(x['rank'] for x in rows),order=idx))
locked=min(REC['calibration'],key=lambda x:(-x['hit5'],-x['hit1'],x['rank_sum'],x['order']))
CFG=tuple(locked['cfg']);print('LOCKED',CFG,'CALIBRATION',locked,flush=True)
print('[4/8] Holdout and identity controls...',flush=True)
for wi in range(24):
    r=evaluate(wi,CFG);r['split']='CAL' if wi in TRAIN else 'HOLD';REC['cases'].append(r)
    print('CASE',wi,r['split'],'RANK',r['rank'],'TOP5',r['top5'],'TARGET',r['target'],flush=True)
H5=sum(x['hit5'] for x in REC['cases'] if x['split']=='HOLD');H1=sum(x['hit1'] for x in REC['cases'] if x['split']=='HOLD')
ALL5=sum(x['hit5'] for x in REC['cases']);HUB={i:sum(i in r['top5'] for r in REC['cases']) for i in range(96)}
print('HOLDOUT_RECALL5',H5,'/12','RECALL1',H1,'/12','ALL_RECALL5',ALL5,'/24','HUB_MAX',max(HUB.values()),flush=True)
# Self-retrieval is a representation sanity test, not a substitute for question retrieval.
li,rank,beta=CFG;KZ,QZ=FEATURES[(li,rank)];SELF=[]
for ci in range(96):
    q=KZ[ci];scores=[pair_scores(q,k,beta) for k in KZ];SELF.append(int(rankings(scores)[0]==ci))
print('SELF_RETRIEVAL1',sum(SELF),'/96',flush=True)
print('[5/7] CUT3 baseline and reference retrieval...',flush=True)
BETA=8.;MREF=8192;DREF=8192
COMPUTE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
KBASE,QBASE=FEATURES[(3,0)]
check('FRAM_LOCK_BEFORE_CUT3',CFG==(3,0,8.0),cfg=list(CFG))
rg=torch.Generator(device='cpu').manual_seed(604000+80000+MREF)
omega=torch.randn((KBASE[0].shape[1],MREF),generator=rg,dtype=torch.float32)*math.sqrt(BETA)
def pref(x):return torch.exp(x.float()@omega-BETA/2)/math.sqrt(MREF)
FR=torch.stack([pref(k).mean(0) for k in KBASE]).to(COMPUTE);QR=[pref(QBASE[wi]).to(COMPUTE) for wi in range(24)]
gen=torch.Generator(device='cpu').manual_seed(604600+800000+MREF*11+DREF)
codes=(torch.randint(0,2,(96,DREF),generator=gen,dtype=torch.int8).float()*2-1).to(COMPUTE)/math.sqrt(DREF)
U=FR.T@codes;PRED=[]
for q in QR:
    z=(q@U)@codes.T;PRED.append((torch.log(z.clamp_min(1e-12)).mean(0)/BETA).cpu())
ref_hits=sum(PANEL[wi]['target'] in rankings(PRED[wi].tolist())[:5] for wi in HOLD)
ref_top1=sum(rankings(PRED[wi].tolist())[0]==PANEL[wi]['target'] for wi in HOLD)
print('REFERENCE_606_LOCKED',{'hold_r1':ref_top1,'hold_r5':ref_hits,'expected_r1':11,'expected_r5':12},flush=True)
check('REFERENCE_RECALL5',ref_hits==12,hold_r1=ref_top1,hold_r5=ref_hits)
@torch.no_grad()
def build_cut3(keys,cut):
    if not keys:raise RuntimeError('EMPTY RETRIEVAL')
    mem=init(keys[0],cut)
    for ci in keys[1:]:mem=append(mem,ci,cut)
    return mem
@torch.no_grad()
def validate_cut3(mem,keys):
    expected=P+sum(len(CAR[ci]['body']) for ci in keys)
    if len(mem)!=NL:raise RuntimeError(f'CUT3 LAYER COUNT {len(mem)} != {NL}')
    for li,(k,v) in enumerate(mem):
        shape=(1,KVH,expected,HD)
        if tuple(k.shape)!=shape or tuple(v.shape)!=shape:raise RuntimeError(f'CUT3 SHAPE L{li}: {tuple(k.shape)} {tuple(v.shape)} != {shape}')
        if not bool(torch.isfinite(k).all()) or not bool(torch.isfinite(v).all()):raise RuntimeError(f'CUT3 NONFINITE L{li}')
    return True
@torch.no_grad()
def audit_cut3(stage):
    rows=[]
    for wi in HOLD:
        target=PANEL[wi]['target'];ids=rankings(PRED[wi].tolist())[:5];keys=sorted(ids)
        typ=CAR[target]['type'];q=W[wi][typ];gold=CAR[target]['gold']
        row=dict(case=wi,target=target,retrieved=ids,engine_order=keys,target_present=int(target in ids),gold=gold,question=q)
        try:
            mem=build_cut3(keys,3)
            row['memory_layers']=len(mem);row['memory_tokens']=int(mem[0][0].shape[-2])
            validate_cut3(mem,keys)
            row['shape_ok']=True
            response=answer(mem,q)
            row['answer']=str(response);row['answer_type']=str(type(response));row['answer_repr']=repr(response)
            row['answer_ok']=int(bool(hit(response,gold)))
        except Exception as e:
            row['error']=repr(e);row['answer_ok']=0
        rows.append(row)
        print('CUT3_AUDIT',stage,wi,'TARGET',target,'PRESENT',row['target_present'],'SHAPE',row.get('shape_ok',False),'OK',row['answer_ok'],'GOLD',repr(gold),'RAW',row.get('answer_repr',row.get('error')),flush=True)
    REC['cut3_audits'][stage]=rows
    n=sum(x['answer_ok'] for x in rows);errors=sum('error' in x for x in rows)
    print('CUT3_AUDIT_SUMMARY',stage,{'correct':n,'total':len(rows),'errors':errors,'retrieval':sum(x['target_present'] for x in rows)},flush=True)
    return rows
PRE_AUDIT=audit_cut3('BASELINE')
print('BASELINE_STATUS',{'correct':sum(x['answer_ok'] for x in PRE_AUDIT),'errors':sum('error' in x for x in PRE_AUDIT),'retrieved':sum(x['target_present'] for x in PRE_AUDIT)},flush=True)
del FR,QR,codes,U,omega,rg,gen
if COMPUTE.type=='cuda':torch.cuda.empty_cache()

print('[6/7] Demo acceptance: 24-case retrieval, reproducibility, prefix/append and negative controls...',flush=True)
REC['demo_acceptance']={};REC['demo_cases']=[]
# The locked retrieval protocol was fitted on cases 0-11; cases 12-23 remain untouched holdout.
all_retrieved=0;all_correct=0;hold_correct=0;cal_correct=0;repeat_ok=0;shape_ok=0
for wi in range(24):
    target=PANEL[wi]['target'];ids=rankings(PRED[wi].tolist())[:5];keys=sorted(ids)
    typ=CAR[target]['type'];q=W[wi][typ];gold=CAR[target]['gold'];present=target in ids
    row={'case':wi,'split':'HOLD' if wi in HOLD else 'CAL','target':target,'retrieved':ids,'engine_order':keys,'target_present':present,'gold':gold}
    try:
        # Genuine engine: init -> append -> answer; no source fact is supplied at answer time.
        mem=build_cut3(keys,3);validate_cut3(mem,keys);row['shape_ok']=True
        first=answer(mem,q);second=answer(mem,q)
        row['answer']=str(first);row['repeat_answer']=str(second)
        row['correct']=bool(hit(first,gold));row['repeat_equal']=str(first)==str(second)
        row['memory_tokens']=int(mem[0][0].shape[-2]);row['memory_layers']=len(mem)
        row['memory_bytes']=int(sum(k.numel()*k.element_size()+v.numel()*v.element_size() for k,v in mem))
    except Exception as exc:
        row['error']=repr(exc);row.update(correct=False,repeat_equal=False,shape_ok=False)
    all_retrieved+=int(present);all_correct+=int(row['correct']);hold_correct+=int(row['correct'] and wi in HOLD)
    cal_correct+=int(row['correct'] and wi in TRAIN);repeat_ok+=int(row['repeat_equal']);shape_ok+=int(row['shape_ok'])
    REC['demo_cases'].append(row)
    print('DEMO_CASE',wi,row['split'],'TARGET',target,'PRESENT',int(present),'CORRECT',int(row['correct']),'REPEAT',int(row['repeat_equal']),'SHAPE',int(row['shape_ok']),'RAW',repr(row.get('answer',row.get('error'))),flush=True)
REC['demo_acceptance']['panel']={'total':24,'correct':all_correct,'cal_correct':cal_correct,'hold_correct':hold_correct,'retrieved':all_retrieved,'repeat_equal':repeat_ok,'shape_valid':shape_ok}
# Exact deterministic rebuild of an unchanged ordered cartridge list. Compare all 28 layers.
@torch.no_grad()
def compare_mem_exact(a,b):
    if len(a)!=len(b):return False
    return all(torch.equal(ka,kb) and torch.equal(va,vb) for (ka,va),(kb,vb) in zip(a,b))
rebuild_rows=[];prefix_rows=[]
for wi in (12,15,18,21):
    keys=sorted(rankings(PRED[wi].tolist())[:5]);a=build_cut3(keys,3);b=build_cut3(keys,3)
    eq=compare_mem_exact(a,b);rebuild_rows.append({'case':wi,'equal':eq,'keys':keys})
    # Check that repeated append from an identical first-cartridge INIT yields the same full memory.
    partial=init(keys[0],3);prefix_valid=validate_cut3(partial,keys[:1])
    for j,ci in enumerate(keys[1:],1):
        partial=append(partial,ci,3);prefix_valid=bool(prefix_valid and validate_cut3(partial,keys[:j+1]))
    prefix_eq=compare_mem_exact(partial,a);prefix_rows.append({'case':wi,'prefix_valid':bool(prefix_valid),'final_equal':prefix_eq})
    print('DEMO_REBUILD',wi,'BIT_EQUAL',int(eq),'APPEND_PREFIX_SHAPES',int(prefix_valid),'APPEND_FINAL_EQUAL',int(prefix_eq),flush=True)
REC['demo_acceptance']['rebuild']=rebuild_rows;REC['demo_acceptance']['append']=prefix_rows
# Diagnostic only: remove target and ask. This tests answer leakage/hallucination;
# failure does not rewrite the locked positive-control success criteria.
negative=[]
for wi in (12,15,18,21):
    target=PANEL[wi]['target'];ids=rankings(PRED[wi].tolist())[:5]
    rest=sorted(ci for ci in ids if ci!=target)
    if not rest:continue
    typ=CAR[target]['type'];q=W[wi][typ];gold=CAR[target]['gold']
    try:
        mem=build_cut3(rest,3);validate_cut3(mem,rest);raw=answer(mem,q)
        row={'case':wi,'target_absent':target not in rest,'gold_leak':bool(hit(raw,gold)),'raw':str(raw)}
    except Exception as exc:row={'case':wi,'error':repr(exc)}
    negative.append(row);print('DEMO_NEGATIVE',row,flush=True)
REC['demo_acceptance']['negative_diagnostic']=negative
print('[7/7] Final integrity, deterministic baseline and acceptance decision...',flush=True)
POST_AUDIT=audit_cut3('FINAL')
comparison=[{'case':a['case'],'before':a.get('answer_repr'),'after':b.get('answer_repr'),'equal':a.get('answer_repr')==b.get('answer_repr'),'before_ok':a['answer_ok'],'after_ok':b['answer_ok']} for a,b in zip(PRE_AUDIT,POST_AUDIT)]
REC['demo_acceptance']['before_after']=comparison
check('WEIGHT_UNCHANGED',weight_sha()==WB)
check('SENTINEL_UNCHANGED',sentinel()==SB)
check('ZERO_TRAINABLE',not any(p.requires_grad for p in model.parameters()))
hooks=[]
for li,l in enumerate(model.model.layers):
    for label,m in [('layer',l),('attn',l.self_attn),('q_proj',l.self_attn.q_proj)]:
        if m._forward_hooks or m._forward_pre_hooks:hooks.append((li,label))
check('NO_HOOKS',not hooks,hooks=hooks)
criteria={
 'holdout_fram_r1_12':H1==12,
 'holdout_fram_r5_12':H5==12,
 'holdout_reference_retrieval_12':ref_hits==12,
 'holdout_cut3_correct_12':hold_correct==12,
 'baseline_cut3_correct_12':sum(x['answer_ok'] for x in PRE_AUDIT)==12,
 'final_cut3_correct_12':sum(x['answer_ok'] for x in POST_AUDIT)==12,
 'no_answer_changes':all(x['equal'] for x in comparison),
 'all24_repeat_deterministic':repeat_ok==24,
 'all24_memory_shapes_valid':shape_ok==24,
 'rebuild_bitwise_4':all(x['equal'] for x in rebuild_rows),
 'append_prefix_4':all(x['prefix_valid'] and x['final_equal'] for x in prefix_rows),
 'weight_unchanged':weight_sha()==WB,
 'sentinel_unchanged':sentinel()==SB,
 'no_trainables':not any(p.requires_grad for p in model.parameters()),
 'no_hooks':not hooks
}
REC['demo_acceptance']['criteria']=criteria
REC['demo_acceptance']['decision']='DEMO_READY_96' if all(criteria.values()) else 'DEMO_NOT_READY'
REC['summary']={'test':'610','demo_decision':REC['demo_acceptance']['decision'],'criteria_passed':sum(criteria.values()),'criteria_total':len(criteria),'holdout_correct':hold_correct,'holdout_total':12,'all_panel_correct':all_correct,'all_panel_total':24,'reference_cartridges':96,'seconds':round(time.perf_counter()-START,2),'scope':'96 native CUT3 cartridges; 6144 residual facts and compressed superposition are NOT certified by this test'}
REC['result_sha']=digest(REC)
os.makedirs('/content',exist_ok=True)
with open(OUT,'w',encoding='utf-8') as f:json.dump(REC,f,ensure_ascii=False,indent=2,default=str)
print('='*126);print('TEST610 CRITERIA',criteria);print('TEST610 FINAL',REC['summary']);print('RESULT_SHA',REC['result_sha']);print('OUTPUT',OUT);print('DECISION',REC['demo_acceptance']['decision']);print('='*126)
'''
# ---------------- RAW COLAB DEMO helper section — embedded verbatim (do not edit; SHA-256 verified below) ----------------
HELPER_SRC=r'''# RAW DEMO — no Gradio, no web server. All saved CUT3 files are real tensors.
from pathlib import Path
import gc
DEMO_DIR=Path('/content/AKBASCORE_MAM_RAW_DEMO');DEMO_DIR.mkdir(parents=True,exist_ok=True)
SESSION=DEMO_DIR/'cut3_session.pt';META=DEMO_DIR/'cut3_session.json';EVENTS=[]
# The reference TEST610 retrieval for the 24 fixed questions is preserved verbatim.
# For an arbitrary question, use the locked exact FRAM representation, not the fixed-question PRED.
def retrieve(q,top_k=5):
    q=str(q).strip()
    if not q:raise ValueError('EMPTY QUESTION')
    reps,_=question_rep(q);qz=apply(reps[CFG[0]],transform_fit(torch.cat(KS[CFG[0]],dim=0),CFG[1]))
    scores=[pair_scores(qz,k,CFG[2]) for k in FEATURES[(CFG[0],CFG[1])][0]]
    ids=rankings(scores)[:int(top_k)]
    return ids,[{'id':i,'score':round(scores[i],6),'type':str(CAR[i]['type'])} for i in ids]
def save_session(mem,ids,path=SESSION):
    validate_cut3(mem,ids)
    payload={'format':'AKBASCORE_CUT3_SESSION_V1','source_blobs':BLOBS,'model_sha256':EXPECTED,'panel_sha':PANEL_SHA,'cut':3,'ids':list(ids),'kv':[(k.detach().cpu().contiguous(),v.detach().cpu().contiguous()) for k,v in mem]}
    torch.save(payload,str(path));return {'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
def load_session(path=SESSION):
    # Trusted local file only: never unpickle files supplied by strangers.
    obj=torch.load(str(path),map_location='cpu',weights_only=True)
    if obj['format']!='AKBASCORE_CUT3_SESSION_V1' or obj['source_blobs']!=BLOBS or obj['model_sha256']!=EXPECTED or obj['panel_sha']!=PANEL_SHA or obj['cut']!=3:raise RuntimeError('SESSION IDENTITY MISMATCH')
    ids=[int(i) for i in obj['ids']]
    if len(ids)!=len(set(ids)) or not ids or any(i<0 or i>=96 for i in ids):raise RuntimeError('SESSION IDS INVALID')
    mem=[(k.to(DEVICE),v.to(DEVICE)) for k,v in obj['kv']];validate_cut3(mem,ids)
    return mem,ids
@torch.no_grad()
def compare_mem_exact(a,b):
    return len(a)==len(b) and all(torch.equal(ka,kb) and torch.equal(va,vb) for (ka,va),(kb,vb) in zip(a,b))
def demo_panel(case_id=12,write_disk=True):
    wi=int(case_id)
    if not 0<=wi<24:raise ValueError('case_id must be 0..23')
    target=PANEL[wi]['target'];q=W[wi][CAR[target]['type']];gold=CAR[target]['gold']
    ids=rankings(PRED[wi].tolist())[:5];ordered=sorted(ids)
    mem=build_cut3(ordered,3);validate_cut3(mem,ordered)
    raw=answer(mem,q);result={'mode':'LOCKED_24_CASE_REFERENCE','case':wi,'split':'HOLD' if wi in HOLD else 'CAL','question':q,'expected':gold,'answer':str(raw),'correct':bool(hit(raw,gold)),'target':target,'retrieved_ranked':ids,'stored_ids':ordered,'target_present':target in ids,'cut3_tokens':int(mem[0][0].shape[-2]),'cut3_bytes':sum(k.numel()*k.element_size()+v.numel()*v.element_size() for k,v in mem),'model_weights_changed':False}
    if write_disk:
        info=save_session(mem,ordered);reloaded,loaded_ids=load_session();again=answer(reloaded,q)
        result.update({'disk':info,'reload_answer':str(again),'reload_equal':str(raw)==str(again),'reload_ids_equal':loaded_ids==ordered,'reload_memory_bit_equal':compare_mem_exact(mem,reloaded)})
    EVENTS.append(result);print('DEMO_PANEL',json.dumps(result,ensure_ascii=False,default=str),flush=True);return result
def demo_ask(question,top_k=5,write_disk=True):
    ids,ranking=retrieve(question,top_k);ordered=sorted(ids);mem=build_cut3(ordered,3);validate_cut3(mem,ordered)
    raw=answer(mem,str(question));result={'mode':'UNSEEN_QUESTION_EXACT_FRAM','question':str(question),'answer':str(raw),'retrieval':ranking,'stored_ids':ordered,'cut3_tokens':int(mem[0][0].shape[-2]),'cut3_bytes':sum(k.numel()*k.element_size()+v.numel()*v.element_size() for k,v in mem),'gold_unavailable':True}
    if write_disk:
        info=save_session(mem,ordered);reloaded,loaded_ids=load_session();again=answer(reloaded,str(question))
        result.update({'disk':info,'reload_answer':str(again),'reload_equal':str(raw)==str(again),'reload_ids_equal':loaded_ids==ordered,'reload_memory_bit_equal':compare_mem_exact(mem,reloaded)})
    EVENTS.append(result);print('DEMO_ASK',json.dumps(result,ensure_ascii=False,default=str),flush=True);return result
def demo_append_cartridge(cartridge_id,question=None):
    if not SESSION.exists():raise RuntimeError('SAVE OR ASK FIRST')
    mem,ids=load_session();ci=int(cartridge_id)
    if ci<0 or ci>=96:raise ValueError('cartridge id out of range')
    if ci in ids:raise ValueError('cartridge already in session')
    # Preserve original append order; do not silently sort/rebuild the existing cache.
    mem=append(mem,ci,3);ids.append(ci);validate_cut3(mem,ids)
    info=save_session(mem,ids);result={'action':'APPEND','cartridge_id':ci,'stored_ids':ids,'disk':info,'memory_tokens':int(mem[0][0].shape[-2])}
    if question is not None:result['answer']=str(answer(mem,str(question)))
    EVENTS.append(result);print('DEMO_APPEND',json.dumps(result,ensure_ascii=False,default=str),flush=True);return result
def demo_reload_ask(question):
    mem,ids=load_session();raw=answer(mem,str(question));result={'action':'RELOAD_ASK','question':str(question),'answer':str(raw),'stored_ids':ids,'source_text_supplied_to_answer':False}
    EVENTS.append(result);print('DEMO_RELOAD',json.dumps(result,ensure_ascii=False),flush=True);return result
def demo_export():
    p=DEMO_DIR/'demo_events.json';p.write_text(json.dumps({'scope':'96 original native CUT3 cartridges; not 6144 compressed superposition','locked_cfg':list(CFG),'source_blobs':BLOBS,'panel_sha':PANEL_SHA,'events':EVENTS},ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print('DEMO_EXPORT',str(p),flush=True);return str(p)
'''
_ENGINE_BYTES=ENGINE_SRC.encode("utf-8")
ENGINE_SHA256=hashlib.sha256(_ENGINE_BYTES).hexdigest();ENGINE_BLOB=git_blob_sha1(_ENGINE_BYTES);HELPER_SHA256=hashlib.sha256(HELPER_SRC.encode("utf-8")).hexdigest()
if ENGINE_BLOB!=ENGINE_BLOB_EXPECTED or ENGINE_SHA256!=ENGINE_SHA256_EXPECTED:
    raise RuntimeError(f"Embedded 610.py differs from the repository file (blob {ENGINE_BLOB} / SHA-256 {ENGINE_SHA256}). Re-paste PART 1 unchanged.")
if HELPER_SHA256!=HELPER_SHA256_EXPECTED:raise RuntimeError(f"Embedded RAW COLAB DEMO helper section changed (SHA-256 {HELPER_SHA256}). Re-paste PART 1 unchanged.")
if ENGINE_SRC.count(ENGINE_SPLIT)!=1:raise RuntimeError("610.py acceptance-stage boundary not found exactly once")
ENGINE_EXEC_SRC,_tail=ENGINE_SRC.split(ENGINE_SPLIT,1);ENGINE_TAIL_SRC=ENGINE_SPLIT+_tail
ENGINE_EXEC_SHA256=hashlib.sha256(ENGINE_EXEC_SRC.encode("utf-8")).hexdigest();ENGINE_TAIL_SHA256=hashlib.sha256(ENGINE_TAIL_SRC.encode("utf-8")).hexdigest()
say(f"[2/6] Engine source verified · {ENGINE_FILE} · git blob {ENGINE_BLOB} · SHA-256 {ENGINE_SHA256} · helpers SHA-256 {HELPER_SHA256[:16]}…")
STARTUP_UTC=utc_now()
if"ENG"in globals()and getattr(globals()["ENG"],"__engine_blob__",None)==ENGINE_BLOB and globals().get("ENGINE_READY"):
    say("[3/6] Engine already initialised in this runtime — reusing it (no second model load).");ENGINE_INIT_SECONDS=globals().get("ENGINE_INIT_SECONDS",float("nan"))
else:
    say("[3/6] Executing 610.py up to its acceptance stage in its own module namespace (downloads and verifies 566.py / 575.py, loads frozen")
    say("      Qwen2.5-7B-Instruct, captures 96 cartridge representations, locks FRAM, runs the baseline CUT3 audit; about 3–5 minutes) …")
    say("      (the lines up to BASELINE_STATUS are printed by 610.py itself while it runs)");_t=time.perf_counter()
    ENG=types.ModuleType("mam_test610_qwen_engine");ENG.__file__=ENGINE_FILE+" (embedded verbatim; executed up to [6/7])";sys.modules[ENG.__name__]=ENG
    exec(compile(ENGINE_EXEC_SRC,ENGINE_FILE,"exec"),ENG.__dict__)
    exec(compile(HELPER_SRC,"RAW_COLAB_DEMO_helpers","exec"),ENG.__dict__)
    ENG.__engine_blob__=ENGINE_BLOB;ENG.__engine_sha256__=ENGINE_SHA256;torch.cuda.synchronize();ENGINE_INIT_SECONDS=time.perf_counter()-_t;ENGINE_READY=True
# ---------------- measurement adapter (read-only with respect to the engine; never alters retrieval, writing, consolidation or answering) ----------------
def hook_module(fn):
    f=getattr(fn,"func",fn);return str(getattr(f,"__module__",None)or type(f).__module__ or"")
def hook_stats(model):
    tot=0;foreign=Counter()
    for m in model.modules():
        for d in(m._forward_hooks,m._forward_pre_hooks,m._backward_hooks):
            for fn in d.values():
                tot+=1;mod=hook_module(fn)
                if not mod.startswith(("transformers","accelerate","torch")):foreign[mod]+=1
    return tot,dict(foreign)
def residual_hooks(model):
    """TEST610 check: no forward hooks or pre-hooks left on any decoder layer, attention module or q_proj."""
    out=[]
    for li,layer in enumerate(model.model.layers):
        for label,module in(("layer",layer),("attention",layer.self_attn),("q_proj",layer.self_attn.q_proj)):
            if module._forward_hooks or module._forward_pre_hooks:out.append(f"L{li}:{label}")
    return out
def optimizer_present():return any(isinstance(v,torch.optim.Optimizer)for v in list(globals().values())+list(ENG.__dict__.values()))
def _tensor_bytes(x):
    x=x.detach().contiguous().cpu()
    if x.dtype==torch.bfloat16:x=x.view(torch.uint16)
    return x.numpy().tobytes()
def memory_sha(layers,rows=None):
    """SHA-256 over the raw bytes of every K and V tensor (optionally only the first `rows` positions)."""
    h=hashlib.sha256()
    for k,v in layers:
        if rows is not None:k=k[:,:,:rows,:];v=v[:,:,:rows,:]
        h.update(_tensor_bytes(k));h.update(_tensor_bytes(v))
    return h.hexdigest()
class Instrument:
    """Thin measurement adapter around the engine namespace. Every scientific quantity comes from the engine itself."""
    def __init__(self,init_seconds):
        self.kind="ENGINE";m=ENG.model;c=m.config;self.lock=threading.RLock()
        self.info=dict(model_id=ENG.MODEL_ID,arch=[ENG.NL,ENG.H,ENG.QH,ENG.KVH,ENG.HD],dtype=str(next(m.parameters()).dtype).replace("torch.",""),
            attn=getattr(c,"_attn_implementation",None),gpu=torch.cuda.get_device_name(0),gpu_total_gib=torch.cuda.get_device_properties(0).total_memory/2**30,
            torch=torch.__version__,transformers=transformers.__version__,gradio=gr.__version__,python=sys.version.split()[0],platform=platform.platform(),
            params=sum(p.numel()for p in m.parameters()),engine_file=ENGINE_FILE,engine_sha256=ENGINE_SHA256,engine_blob=ENGINE_BLOB,engine_exec_sha256=ENGINE_EXEC_SHA256,
            engine_tail_sha256=ENGINE_TAIL_SHA256,helper_sha256=HELPER_SHA256,raw_demo_file_sha256=RAW_DEMO_FILE_SHA256,init_seconds=init_seconds,startup_utc=STARTUP_UTC,
            source_blobs={k:git_blob_sha1((ENG.s566 if k=="566.py"else ENG.s575).encode("utf-8"))for k in("566.py","575.py")},
            sentinel_method="TEST566 sentinel: SHA-256 over 16 slices × 256 values of 7 weight tensors (o_proj of layers 0/7/14/21/27, final norm, lm_head)",
            guard_method="all-parameter guard: SHA-256 over (name, shape, dtype, sum, |sum|, sum of squares) of every parameter tensor",
            full_weight_method="610.py weight_sha(): SHA-256 over (name, shape, dtype, raw bytes) of every parameter, computed by the engine at start-up",
            forward_counter="recording-only forward pre-hook on the causal-LM module, attached only during an engine call",
            observer="append observer: wraps the engine's init(·,3)/append(·,·,3) while build_cut3 runs; calls them unchanged; compares prior K/V rows bitwise")
        self.info["weight_sha0"]=ENG.WB;self.info["sentinel0"]=self.sentinel();self.info["sentinel_engine_load"]=ENG.S0;self.info["sentinel_610_start"]=ENG.SB
        self.info["guard0"]=self.guard();self.info["hooks0"]=hook_stats(m)[0];self._startup=self._read_startup()
    # ---- read-only views of engine state ----
    def config(self):
        return dict(MODEL_ID=ENG.MODEL_ID,CUT=CUT_EXPECTED,ENGINE_CUTS=list(ENG.CUTS),MAX_NEW=ENG.MAX_NEW,SEED=ENG.SEED,PANEL_SEED=ENG.PANEL_SEED,EXPECTED_PANEL_SHA=ENG.EXPECTED_PANEL_SHA,
                    panel_sha=ENG.PANEL_SHA,prefix_tokens=ENG.P,n_cartridges=len(ENG.CAR),n_cases=len(ENG.PANEL),system=ENG.SYSTEM,arch=[ENG.NL,ENG.H,ENG.QH,ENG.KVH,ENG.HD],
                    engine_sha256=hashlib.sha256(ENGINE_SRC.encode("utf-8")).hexdigest(),engine_blob=git_blob_sha1(ENGINE_SRC.encode("utf-8")),engine_exec_sha256=ENGINE_EXEC_SHA256,
                    helper_sha256=hashlib.sha256(HELPER_SRC.encode("utf-8")).hexdigest(),source_blobs=dict(ENG.BLOBS),source_blobs_measured=dict(self.info["source_blobs"]),
                    fram_cfg=list(ENG.CFG),fram_layers=list(ENG.LAYERS),fram_ranks=list(ENG.RANKS),fram_betas=list(ENG.BETAS),n_configs=len(ENG.CONFIGS),
                    train=list(ENG.TRAIN),hold=list(ENG.HOLD),beta=float(ENG.BETA),M=int(ENG.MREF),D=int(ENG.DREF),top_k=TOP_K)
    def status(self):return{"title":"AKBASCORE MAM-BÇ","model":ENG.MODEL_ID,"cut":CUT_EXPECTED,"fram":list(ENG.CFG),"panel_sha":ENG.PANEL_SHA,"reference_test":"TEST610","reference_decision":TEST610["decision"]}
    def _read_startup(self):
        rows=[dict(case=int(r["case"]),split=r["split"],target=int(r["target"]),rank=int(r["rank"]),top5=[int(x)for x in r["top5"]])for r in ENG.REC["cases"]]
        base=[dict(case=int(r["case"]),target=int(r["target"]),present=int(r["target_present"]),ok=int(r["answer_ok"]),answer=str(r.get("answer",r.get("error"))))for r in ENG.PRE_AUDIT]
        return dict(cfg=list(ENG.CFG),locked=jsafe(ENG.locked),calibration=jsafe(ENG.REC["calibration"]),n_configs=len(ENG.CONFIGS),hold_r1=int(ENG.H1),hold_r5=int(ENG.H5),all_r5=int(ENG.ALL5),
                    self_r1=int(sum(ENG.SELF)),self_n=len(ENG.SELF),hub_max=int(max(ENG.HUB.values())),ref_hold_r1=int(ENG.ref_top1),ref_hold_r5=int(ENG.ref_hits),cases=rows,baseline=base,
                    baseline_correct=sum(b["ok"]for b in base),baseline_retrieved=sum(b["present"]for b in base),baseline_n=len(base),
                    question_tokens=[int(ENG.QINFO[w]["n"])for w in range(len(ENG.PANEL))])
    def startup(self):return json.loads(json.dumps(self._startup))
    def cases(self):return[{"case":i,"split":split_of(i),"type":ENG.CAR[z["target"]]["type"],"question":self.question(i),"target":int(z["target"])}for i,z in enumerate(ENG.PANEL)]
    def question(self,case):t=ENG.PANEL[case]["target"];return ENG.W[case][ENG.CAR[t]["type"]]
    def car(self,ci):c=ENG.CAR[ci];return dict(id=int(ci),world=int(c["world"]),type=c["type"],fact=c["fact"],gold=c["gold"],body_len=len(c["body"]))
    def target(self,case):return int(ENG.PANEL[case]["target"])
    def source_ids(self,ci):return list(ENG.PIDS)+list(ENG.CAR[ci]["body"])
    def query_ids(self,q):return list(ENG.tok(ENG.suffix(q),add_special_tokens=False).input_ids)
    def frame_ids(self,q):return list(ENG.tok(ENG.PREFIX+q+"\n\n",add_special_tokens=False).input_ids)
    def decode_each(self,ids):return[ENG.tok.decode([t])for t in ids]
    def panel_strings(self):return sorted({w for row in ENG.BASE for w in row})
    def hit(self,ans,gold):return bool(ENG.hit(ans,gold))
    # ---- retrieval: engine scores only ----
    def fram_scores(self,case):
        li,rank,beta=ENG.CFG;KZ,QZ=ENG.FEATURES[(li,rank)]
        return[float(ENG.pair_scores(QZ[case],k,beta))for k in KZ]
    def ref_scores(self,case):return[float(x)for x in ENG.PRED[case].tolist()]
    def rankings(self,scores):return list(ENG.rankings(list(scores)))
    def fram_shapes(self,case):
        li,rank,beta=ENG.CFG;KZ,QZ=ENG.FEATURES[(li,rank)];return dict(query=list(QZ[case].shape),keys=[list(k.shape)for k in KZ])
    # ---- integrity ----
    def sentinel(self):return ENG.sentinel()
    def guard(self):
        h=hashlib.sha256()
        with torch.inference_mode():
            for name,p in ENG.model.named_parameters():
                x=p.detach().float();vals=(float(x.sum().item()),float(x.abs().sum().item()),float((x*x).sum().item()))
                h.update(name.encode());h.update(str(tuple(p.shape)).encode());h.update(str(p.dtype).encode());h.update(np.asarray(vals,dtype=np.float64).tobytes());del x
        torch.cuda.empty_cache();return h.hexdigest()
    def full_weight_sha(self):return ENG.weight_sha()
    def frozen(self):
        m=ENG.model;tot,fo=hook_stats(m)
        return dict(training=bool(m.training),trainable_tensors=sum(int(p.requires_grad)for p in m.parameters()),
                    lora=bool(hasattr(m,"peft_config")or any("lora"in n.lower()for n,_ in m.named_modules())),optimizer=optimizer_present(),
                    hooks=tot,foreign_hooks=sum(fo.values()),foreign_modules=fo,residual_hooks=residual_hooks(m),sentinel=self.sentinel())
    # ---- measured engine calls ----
    def _recorder(self):
        rec=[]
        def _rec(mod,a,kw):
            ids=kw.get("input_ids",a[0]if a else None);emb=kw.get("inputs_embeds");pkv=kw.get("past_key_values");pl=None
            if pkv is not None:
                try:pl=int(pkv.get_seq_length())
                except Exception:pl=-1
            if ids is not None:
                n=int(ids.shape[-1]);rec.append(dict(kind="ids",n=n,ids=[int(t)for t in ids.reshape(-1).tolist()]if n<=8192 else None,embeds_zero=None,has_past=pkv is not None,past_len=pl))
            else:
                n=int(emb.shape[1])if emb is not None else -1
                rec.append(dict(kind="embeds",n=n,ids=None,embeds_zero=(bool(float(emb.detach().abs().max())==0.0)if emb is not None else None),has_past=pkv is not None,past_len=pl))
        return rec,_rec
    def build(self,keys):
        """Engine build_cut3(keys, 3) with the append observer and the forward recorder attached. Returns (memory, trace, seconds)."""
        keys=[int(k)for k in keys];rec,_rec=self._recorder();steps=[];orig_init=ENG.init;orig_app=ENG.append
        def init_obs(ci,cut):
            a=len(rec);t=time.perf_counter();res=orig_init(ci,cut);torch.cuda.synchronize()
            steps.append(dict(kind="init",cartridge_id=int(ci),old_len=0,new_len=int(res[0][0].shape[-2]),fw=(a,len(rec)),prior_rows_bitwise_unchanged=None,prior_sha_before=None,prior_sha_after=None,
                              memory_sha=memory_sha(res),seconds=time.perf_counter()-t));return res
        def app_obs(memory,ci,cut):
            a=len(rec);oldn=int(memory[0][0].shape[-2]);sb=memory_sha(memory);t=time.perf_counter();res=orig_app(memory,ci,cut);torch.cuda.synchronize();dt_=time.perf_counter()-t
            same=all(torch.equal(res[l][0][:,:,:oldn],memory[l][0])and torch.equal(res[l][1][:,:,:oldn],memory[l][1])for l in range(len(memory)))
            steps.append(dict(kind="append",cartridge_id=int(ci),old_len=oldn,new_len=int(res[0][0].shape[-2]),fw=(a,len(rec)),prior_rows_bitwise_unchanged=bool(same),
                              prior_sha_before=sb,prior_sha_after=memory_sha(res,oldn),memory_sha=memory_sha(res),seconds=dt_));return res
        with self.lock,torch.no_grad():
            h=ENG.model.register_forward_pre_hook(_rec,with_kwargs=True);ENG.init=init_obs;ENG.append=app_obs
            try:
                torch.cuda.synchronize();t=time.perf_counter();mem=ENG.build_cut3(keys,CUT_EXPECTED);ENG.validate_cut3(mem,keys);torch.cuda.synchronize();dt_=time.perf_counter()-t
            finally:
                h.remove();ENG.init=orig_init;ENG.append=orig_app
        for s in steps:s["forwards"]=rec[s["fw"][0]:s["fw"][1]];del s["fw"]
        return mem,dict(steps=steps,forwards=rec,keys=keys),dt_
    def ask(self,mem,q):
        rec,_rec=self._recorder()
        with self.lock,torch.no_grad():
            h=ENG.model.register_forward_pre_hook(_rec,with_kwargs=True)
            try:torch.cuda.synchronize();t=time.perf_counter();res=ENG.answer(mem,str(q));torch.cuda.synchronize();dt_=time.perf_counter()-t
            finally:h.remove()
        return str(res),rec,dt_
    def retrieve_free(self,q):
        """RAW COLAB DEMO retrieve(q): exact FRAM kernel on the locked configuration (one question-frame forward)."""
        rec,_rec=self._recorder()
        with self.lock,torch.no_grad():
            h=ENG.model.register_forward_pre_hook(_rec,with_kwargs=True)
            try:t=time.perf_counter();ids,ranking=ENG.retrieve(str(q),TOP_K);dt_=time.perf_counter()-t
            finally:h.remove()
        return[int(i)for i in ids],jsafe(ranking),rec,dt_
    def save(self,mem,ids,path):return dict(ENG.save_session(mem,[int(i)for i in ids],path=Path(path)))
    def load(self,path):
        mem,ids=ENG.load_session(path=Path(path));return mem,[int(i)for i in ids]
    def equal(self,a,b):return bool(ENG.compare_mem_exact(a,b))
    def memory_len(self,mem):return int(mem[0][0].shape[-2])
    def memory_sha(self,mem):return memory_sha(mem)
    def memory_bytes(self,mem):return int(sum(k.numel()*k.element_size()+v.numel()*v.element_size()for k,v in mem))
    def memory_profile(self,mem):
        """Display-only statistics of the CUT3 memory (mean L2 norm over KV heads of every K and V row, per layer)."""
        kn=[];vn=[]
        with torch.inference_mode():
            for k,v in mem:
                kn.append([float(x)for x in k.float().norm(dim=-1).mean(dim=1)[0].tolist()]);vn.append([float(x)for x in v.float().norm(dim=-1).mean(dim=1)[0].tolist()])
        return dict(layers=len(mem),T=len(kn[0]),k_norm=kn,v_norm=vn,bytes=self.memory_bytes(mem),kv_heads=int(mem[0][0].shape[1]),head_dim=int(mem[0][0].shape[-1]),
                    dtype=str(mem[0][0].dtype).replace("torch.",""),shape=list(mem[0][0].shape))
    def release(self,*mems):
        for m in mems:del m
        gc.collect();torch.cuda.empty_cache()
    def acceptance_replay(self,progress=None):
        """Executes the 610.py acceptance stage [6/7]–[7/7] VERBATIM in the engine namespace (24 cases, rebuilds, negative controls, final audit)."""
        owner=threading.get_ident()
        class Tee(io.TextIOBase):
            """Records only what the replay thread prints (line-buffered); output of other threads passes through unrecorded."""
            def __init__(s,base):s.base=base;s.buf=io.StringIO();s.n=0;s.pend=""
            def write(s,x):
                s.base.write(x)
                if threading.get_ident()!=owner:return len(x)
                s.buf.write(x);s.pend+=x
                while"\n"in s.pend:
                    ln,s.pend=s.pend.split("\n",1)
                    if ln.startswith(("DEMO_CASE","DEMO_REBUILD","DEMO_NEGATIVE","CUT3_AUDIT FINAL")):
                        s.n+=1
                        if progress:progress(s.n,44,ln)
                return len(x)
            def flush(s):s.base.flush()
        tee=Tee(sys.stdout);t=time.perf_counter()
        with self.lock,torch.no_grad(),contextlib.redirect_stdout(tee):exec(compile(ENGINE_TAIL_SRC,ENGINE_FILE,"exec"),ENG.__dict__)
        R=ENG.REC;da=R["demo_acceptance"]
        return dict(summary=jsafe(R["summary"]),criteria=jsafe(da["criteria"]),decision=da["decision"],cases=jsafe(R["demo_cases"]),negatives=jsafe(da["negative_diagnostic"]),
                    rebuild=jsafe(da["rebuild"]),append=jsafe(da["append"]),result_sha=R["result_sha"],out_path=str(ENG.OUT),stdout=tee.buf.getvalue(),seconds=time.perf_counter()-t)
    def sync(self):torch.cuda.synchronize()
    def reset_peak(self):torch.cuda.reset_peak_memory_stats()
    def peak_gib(self):return torch.cuda.max_memory_allocated()/2**30
say("[4/6] Measurement adapter · TEST566 weight sentinel · all-parameter guard · start-up gates …")
INSTR=Instrument(ENGINE_INIT_SECONDS)
_cfg=INSTR.config();_st=INSTR.startup();_fz=INSTR.frozen()
say(f"[5/6] Full-weight SHA-256 (610.py, start-up) {INSTR.info['weight_sha0']} · reference {WEIGHT_SHA_EXPECTED}")
ENGINE_LOCK=[("610.py git blob SHA-1 = repository blob",_cfg["engine_blob"]==ENGINE_BLOB_EXPECTED),("610.py SHA-256 = embedded reference",_cfg["engine_sha256"]==ENGINE_SHA256_EXPECTED),
 ("RAW COLAB DEMO helper section SHA-256 = embedded reference",_cfg["helper_sha256"]==HELPER_SHA256_EXPECTED),
 ("566.py and 575.py downloaded by 610.py with the expected git blobs",_cfg["source_blobs"]==SOURCE_BLOBS_EXPECTED==_cfg["source_blobs_measured"]),
 ("model = Qwen/Qwen2.5-7B-Instruct",_cfg["MODEL_ID"]==MODEL_ID),(f"architecture {arch_txt()}",tuple(INSTR.info["arch"])==ARCH==tuple(_cfg["arch"])),
 ("CUT3 is one of the 566.py cuts · canonical prefix = 29 tokens",CUT_EXPECTED in _cfg["ENGINE_CUTS"]and _cfg["prefix_tokens"]==PREFIX_EXPECTED),
 ("locked panel SHA-256 · 96 cartridges · 24 cases",_cfg["panel_sha"]==EXPECTED_PANEL_SHA==_cfg["EXPECTED_PANEL_SHA"]and _cfg["n_cartridges"]==N_CART and _cfg["n_cases"]==N_CASES),
 ("full-weight SHA-256 at start-up = reference",INSTR.info["weight_sha0"]==WEIGHT_SHA_EXPECTED),
 ("weight sentinel = value at model load = value at 610.py start",INSTR.info["sentinel0"]==INSTR.info["sentinel_engine_load"]==INSTR.info["sentinel_610_start"]),
 ("FRAM configuration locked on calibration cases 0–11 = (layer 3, rank 0, β 8)",tuple(_cfg["fram_cfg"])==FRAM_LOCK and _cfg["train"]==list(TRAIN_CASES)and _cfg["hold"]==list(HOLD_CASES)),
 ("start-up FRAM reproduces TEST610: holdout R@1 12/12 · R@5 12/12 · all R@5 24/24 · self-retrieval 96/96",(_st["hold_r1"],_st["hold_r5"],_st["all_r5"],_st["self_r1"])==(12,12,24,96)),
 ("start-up reference ranking reproduces TEST610: holdout R@1 11/12 · R@5 12/12",(_st["ref_hold_r1"],_st["ref_hold_r5"])==(TEST610["reference"]["hold_r1"],TEST610["reference"]["hold_r5"])),
 ("start-up baseline CUT3 audit: 12/12 holdout answers correct, 12/12 targets retrieved",(_st["baseline_correct"],_st["baseline_retrieved"],_st["baseline_n"])==(12,12,12)),
 ("reference settings M = 8192 · D = 8192 · β = 8",(_cfg["M"],_cfg["D"],_cfg["beta"])==(M_REF,D_REF,8.0)),
 ("trainable tensors = 0 · eval mode · no LoRA · no optimizer",_fz["trainable_tensors"]==0 and not _fz["training"]and not _fz["lora"]and not _fz["optimizer"]),
 ("no foreign hooks · no hooks on decoder layers, attention or q_proj",_fz["foreign_hooks"]==0 and not _fz["residual_hooks"])]
_bad=[n for n,ok in ENGINE_LOCK if not ok]
if _bad:raise RuntimeError(f"ENGINE LOCK FAILED: {_bad}")
say(f"[6/6] Engine lock PASS · {len(ENGINE_LOCK)} checks · init {ENGINE_INIT_SECONDS:.1f}s · guard {INSTR.info['guard0'][:24]}… · sentinel {INSTR.info['sentinel0'][:24]}…")
say("PART 1 / 3 complete. Now run PART 2 / 3 in the next cell.");say("="*140)
MAM_BC_PART1_OK=True
#<<ENGINE_END>>
