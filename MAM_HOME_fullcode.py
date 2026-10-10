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
# =====================================================================================================================================
# AKBASCORE MAM-BÇ · EXTERNAL ASSOCIATIVE MEMORY FOR A FROZEN LANGUAGE MODEL — QWEN2.5-7B · PART 2 / 3 · MEASUREMENT PIPELINE AND FIGURES
# Same demo as PART 1. Run this cell after PART 1 / 3, in the same Colab runtime; then run PART 3 / 3.
# Defines the fail-closed measurement run (integrity → FRAM retrieval → CUT3 write + append-only consolidation → source-free question →
# save / reload / answer → negative control → verification → sealed payload, logs, ZIP), six figures (JPEG 300 dpi + vector PDF) and the
# optional live replay of the TEST610 acceptance stage. Nothing here changes the engine.
# Discovered and developed by Mustafa Akbaş · AkbasCore AI Teknoloji · 10 October 2026 · AKBASCORE RESEARCH SOFTWARE LICENSE
# =====================================================================================================================================
if not globals().get("MAM_BC_PART1_OK"):raise RuntimeError("PART 1 / 3 has not been run in this runtime. Run PART 1, then PART 2, then PART 3.")
#<<CORE_BEGIN>>
import unicodedata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.patches import Rectangle,FancyArrowPatch,Patch,Circle
from matplotlib.lines import Line2D
from PIL import Image
ROOT=Path("/content/AKBASCORE_MAM_BC_QWEN_CUT3_FRAM")if os.path.isdir("/content")else Path("/tmp/AKBASCORE_MAM_BC_QWEN_CUT3_FRAM")
ROOT.mkdir(parents=True,exist_ok=True);ALLOWED_PATHS=[str(ROOT)]
N_STAGES=10
def ev(stage,title,body):return dict(stage=stage,title=title,body=body)
def file_ready(p):return p is not None and Path(p).is_file()and Path(p).stat().st_size>0
RUN_PREFIX="MAMBC-";REPLAY_PREFIX="MAMBCREPLAY-"
def prune_runs(keep=4,prefix=RUN_PREFIX):
    runs=sorted([p for p in ROOT.glob(prefix+"*")if p.is_dir()],key=lambda p:p.stat().st_mtime)
    for p in runs[:max(0,len(runs)-keep)]:shutil.rmtree(p,ignore_errors=True)
def segments(steps,p):
    """Row ranges of the CUT3 memory: shared prefix (written with cartridge 1) and the body rows of cartridges 1..5."""
    seg=[("prefix",0,p,None)];prev=p
    for n_,s in enumerate(steps,1):seg.append((f"C{n_}",prev,s["new_len"],s["cartridge_id"]));prev=s["new_len"]
    return seg
def post_seal_audit(imgs,pdfs,zp,member_names,pp,sha,tp,mp,verdict,session_path,session_sha):
    checks=[]
    def ok(name,cond):
        checks.append(name)
        if not cond:raise AuditFail("POST-SEAL AUDIT FAILED: "+name)
    ok(f"{N_FIGS}/{N_FIGS} figures created (JPEG)",len(imgs)==N_FIGS and all(file_ready(p)for p,_ in imgs));ok(f"{N_FIGS}/{N_FIGS} vector PDF copies",len(pdfs)==N_FIGS and all(file_ready(p)for p in pdfs))
    bad=[]
    for p,_ in imgs:
        with Image.open(p)as im:
            if not(im.format=="JPEG"and im.mode=="RGB"and im.size[0]>=3600):bad.append(p.name)
    ok(f"{N_FIGS}/{N_FIGS} figures are JPEG/RGB at ≥3600 px width",not bad)
    with zipfile.ZipFile(zp)as z:
        ok("ZIP testzip()",z.testzip()is None);nm=z.namelist()
        ok("ZIP is flat (no sub-folders)",all("/"not in n for n in nm));ok("ZIP contents = expected package",sorted(nm)==sorted(member_names))
        jp=sorted(n for n in nm if n.lower().endswith(".jpg"));ok("figure numbering 01..%02d"%N_FIGS,[n[4:6]for n in jp]==[f"{i:02d}"for i in range(1,N_FIGS+1)])
        ok("CUT3 session file in the ZIP is byte-identical to the sealed SHA-256",hashlib.sha256(z.read(Path(session_path).name)).hexdigest()==session_sha)
    ok("payload SHA-256 recomputation",hashlib.sha256(pp.read_bytes()).hexdigest()==sha);ok("payload JSON parses",isinstance(json.loads(pp.read_bytes().decode("utf-8")),dict))
    m=json.loads(mp.read_text(encoding="utf-8"));ok("manifest hash and verdict match payload",m["payload_sha256"]==sha and m["verdict"]==verdict)
    ok("readable log exists and is non-empty",file_ready(tp));ok("ZIP exists and is non-empty",file_ready(zp))
    return checks
def fw_txt(f):
    if f["kind"]=="ids":return f"token ids n={f['n']}"+(f" · cache {f['past_len']}"if f["has_past"]else" · no cache")
    return f"embeddings n={f['n']} ({'all zero'if f['embeds_zero']else'NON-ZERO'})"+(f" · cache {f['past_len']}"if f["has_past"]else" · no cache")
def mib(b):return b/2**20
def check_build(chk,I,trace,keys,P_,label):
    """Forward-by-forward audit of one engine build_cut3(keys, 3) call (shared by the main memory, the control and the free-text memory)."""
    steps=trace["steps"];n=len(keys)
    chk(f"{label}: {n} consolidation steps in engine order (one init + {n-1} appends)",len(steps)==n and [s["cartridge_id"]for s in steps]==list(keys)and steps[0]["kind"]=="init"and all(s["kind"]=="append"for s in steps[1:]))
    s0=steps[0];f0=s0["forwards"];src0=I.source_ids(keys[0])
    chk(f"{label}: step 1 = one token-id forward of the instruction prefix + cartridge 1's own fact (no cache) + one all-zero-embedding forward over the same rows",
        len(f0)==2 and f0[0]["kind"]=="ids"and f0[0]["ids"]==src0 and not f0[0]["has_past"]and f0[1]["kind"]=="embeds"and f0[1]["embeds_zero"]and f0[1]["n"]==len(src0)and not f0[1]["has_past"]and s0["new_len"]==len(src0))
    for n_,s in enumerate(steps[1:],2):
        f=s["forwards"];src=I.source_ids(s["cartridge_id"]);q=len(src)-P_
        chk(f"{label}: step {n_} write = one token-id forward of the instruction prefix + cartridge {n_}'s own fact only (no cache)",len(f)==2 and f[0]["kind"]=="ids"and f[0]["ids"]==src and not f[0]["has_past"])
        chk(f"{label}: step {n_} consolidation = all-zero embeddings for the {q} new rows only, over the existing {s['old_len']}-token memory",f[1]["kind"]=="embeds"and f[1]["embeds_zero"]and f[1]["n"]==q and f[1]["has_past"]and f[1]["past_len"]==s["old_len"]==steps[n_-2]["new_len"])
        chk(f"{label}: step {n_} previously consolidated rows 0…{s['old_len']-1} bitwise unchanged",s["prior_rows_bitwise_unchanged"]is True and s["prior_sha_before"]==s["prior_sha_after"])
        chk(f"{label}: step {n_} memory grows by exactly the new rows (+{q})",s["new_len"]==s["old_len"]+q)
    ids_fw=[f for f in trace["forwards"]if f["kind"]=="ids"]
    chk(f"{label}: no other text read: exactly {n} token-id forwards, each the instruction prefix + the own fact of the cartridge being written, each once",
        len(ids_fw)==n and [f["ids"]for f in ids_fw]==[I.source_ids(ci)for ci in keys]and len(trace["forwards"])==2*n)
    return steps,ids_fw
def check_query(chk,I,fw,q,T,label):
    qids=I.query_ids(q)
    chk(f"{label}: query forward input = question-template tokens only, over the installed {T}-token memory",len(fw)>=1 and fw[0]["kind"]=="ids"and fw[0]["ids"]==qids and fw[0]["has_past"]and fw[0]["past_len"]==T)
    chk(f"{label}: generation forwards = one new token each, over the growing cache",all(f["kind"]=="ids"and f["n"]==1 and f["has_past"]and f["past_len"]==T+len(qids)+n_ for n_,f in enumerate(fw[1:])))
    return qids
def make_txt(P,sha,names,sealed_utc):
    o=[];a=o.append;S="="*140;Dd="-"*140;E_=P["environment"];L=P["live"];cfg=P["engine"]["config"];fz=P["frozen"];R=L["retrieval"];K=L["disk"];N=L["negative"];su=P["startup"]
    a(S);a(f"{DEMO_TITLE} — READABLE RUN LOG");a(f"{DISCOVERY} · {ORG} · {DATE_TXT} · {AUTHOR_PLACE}");a(f"{LICENSE_NAME} · {COPYRIGHT} · {LICENSE_URL}");a(S)
    a(PRIORITY);a("Scientific question: "+SCI_QUESTION)
    a(f"Derived from {names['payload']} (SHA-256 {sha}). Manifest: {names['manifest']}.")
    a("This is a LIVE run of the frozen Qwen2.5-7B CUT3 + FRAM engine (610.py executed up to its acceptance stage; RAW COLAB DEMO helpers). It is not TEST610 itself; sealed TEST610 and TEST575 values are reproduced verbatim, never recomputed.")
    for k_,v in(("RUN ID",P["run_id"]),("START UTC",P["run_start_utc"]),("END UTC",P["run_end_utc"]),("SEALED UTC",sealed_utc),("VERDICT",P["verdict"]),
        ("MODEL",f"{cfg['MODEL_ID']} · frozen · {P['model']['dtype']} · attention {P['model']['attn']}"),("ARCHITECTURE",arch_txt(tuple(P["model"]["arch"]))),
        ("GPU",f"{E_['gpu']} (TEST610 reference GPU: {CANON_GPU}; same: {P['hardware']['same']})"),("TORCH / TRANSFORMERS / GRADIO",f"{E_['torch']} / {E_['transformers']} / {E_['gradio']}"),
        ("ENGINE FILE",f"{P['engine']['file']} · git blob {P['engine']['blob']} · SHA-256 {P['engine']['sha256']}"),("ENGINE SECTION",f"executed up to [6/7] · SHA-256 {P['engine']['exec_sha256']}"),
        ("SOURCE FILES",f"566.py {cfg['source_blobs']['566.py']} · 575.py {cfg['source_blobs']['575.py']} (git blobs verified by 610.py)"),
        ("HELPERS",f"{HELPER_FILE} · SHA-256 {cfg['helper_sha256']}"),("FRAM LOCK",f"layer {cfg['fram_cfg'][0]} · rank {cfg['fram_cfg'][1]} · β {cfg['fram_cfg'][2]} (calibration cases 0–11, {cfg['n_configs']} configurations)"),
        ("CUT3 MEMORY",f"cartridge = H{cfg['CUT']} residual + layers 0–{cfg['CUT']} K/V; {upper_txt(cfg['CUT'],cfg['arch'][0])} consolidate"),
        ("PANEL",f"locked 24-case panel · SHA-256 {cfg['panel_sha']}"),("TEST610 RESULT SHA",TEST610["result_sha"]),("FULL-WEIGHT SHA-256",f"{fz['weight_sha_startup']} (reference {WEIGHT_SHA_EXPECTED})")):a(f"{k_:<30}: {v}")
    a("");a("CASE");a(Dd)
    a(f"case {L['case']:02d} ({L['split']}) · question: {L['question']}");a(f"target cartridge #{L['target']:02d} · expected answer (audit metadata): {L['expected']}")
    a("");a("FRAM RETRIEVAL  [LIVE · engine scores]");a(Dd)
    a(f"exact FRAM kernel: target rank {R['fram']['rank']} of {L['n_cartridges']} · top-5 {R['fram']['top5']} · target score {R['fram']['target_score']:.6f} · best other {R['fram']['best_other_score']:.6f}")
    a(f"locked reference ranking (used for the CUT3 selection of panel questions): target rank {R['ref']['rank']} · top-5 {R['ref']['top5']} · target present: {R['target_present']}")
    a(f"engine order of the CUT3 memory (sorted ids, as in TEST610): {L['engine_order']}")
    for c in L["cartridges"]:a(f"   #{c['id']:02d} · {c['type']:<8} · {c['body_len']:>2} body tokens · {c['fact']}"+("   ← TARGET"if c["is_target"]else""))
    a("");a("CUT3 WRITE + APPEND-ONLY CONSOLIDATION · build_cut3(ids, 3)  [LIVE]");a(Dd)
    for n_,s in enumerate(L["steps"],1):
        a(f"step {n_} · {s['kind']:<6} · cartridge #{s['cartridge_id']:02d} · memory {s['old_len']} → {s['new_len']} tokens (+{s['new_len']-s['old_len']}) · {1000*s['seconds']:.1f} ms")
        for f in s["forwards"]:a(f"         forward: {fw_txt(f)}")
        if s["kind"]=="append":a(f"         previously consolidated rows 0…{s['old_len']-1}: bitwise unchanged = {s['prior_rows_bitwise_unchanged']} · SHA {s['prior_sha_before'][:16]}… → {s['prior_sha_after'][:16]}…")
    M=L["memory"];a(f"final memory: {M['T']} tokens × {M['layers']} layers · tensor {M['shape']} · {M['bytes']} bytes ({mib(M['bytes']):.2f} MiB, {M['dtype']}) · SHA {L['memory_sha'][:24]}…")
    a("");a("SOURCE-FREE QUESTION · answer(memory, question)  [LIVE]");a(Dd)
    a(f"answer  : {L['answer']}");a(f"expected: {L['expected']} · {'CORRECT'if L['correct']else'INCORRECT'} (re-derived: {L['correct_rederived']})")
    a(f"query input = question template only: {L['query_tokens']} token ids over the installed memory ({M['T']} tokens) · generation forwards {L['gen_forwards']} · {1000*L['answer_seconds']:.0f} ms")
    a(f"repeat: identical answer = {L['repeat_identical']} · memory SHA unchanged by querying = {L['memory_unchanged_by_query']}")
    a(f"sealed TEST610 answer for this case: {L['sealed_answer']!r} · live identical: {L['matches_sealed']} (reported, not a gate)")
    a("");a("DISK · save_session → load_session → answer  [LIVE]");a(Dd)
    a(f"file {K['file']} · {K['bytes']} bytes · SHA-256 {K['sha256']} · recomputed after write: {K['sha_recomputed']}")
    a(f"in-memory K/V {K['memory_bytes']} bytes · file overhead {K['bytes']-K['memory_bytes']} bytes · fact text found in the file: {K['fact_text_in_file']} · expected answer string found in the file: {K['gold_in_file']}")
    a(f"reload: ids equal {K['ids_equal']} · K/V bitwise equal in all {M['layers']} layers {K['bit_equal']} · memory SHA equal {K['memory_sha_equal']} · answer after reload {K['answer']!r} · identical {K['answer_identical']}")
    a("");a("NEGATIVE CONTROL · target cartridge removed  [LIVE · diagnostic]");a(Dd)
    if N.get("skipped"):a("skipped: "+N["skipped"])
    else:a(f"memory of {N['ids']} ({N['T']} tokens) · answer {N['answer']!r} · expected answer produced: {N['gold_leak']}")
    if L.get("custom"):cu=L["custom"];a("");a("FREE-TEXT QUESTION (unscored)");a(Dd);a(f"{cu['question']} → {cu['answer']!r} · exact FRAM top-5 {cu['ids']} · query tokens {cu['query_tokens']}")
    a("");a("START-UP IN THIS RUNTIME (610.py procedure, before any run)");a(Dd)
    a(f"FRAM holdout Recall@1 {su['hold_r1']}/12 · Recall@5 {su['hold_r5']}/12 · all Recall@5 {su['all_r5']}/24 · self-retrieval {su['self_r1']}/{su['self_n']} · hub max {su['hub_max']}")
    a(f"reference ranking holdout Recall@1 {su['ref_hold_r1']}/12 · Recall@5 {su['ref_hold_r5']}/12 · baseline CUT3 audit {su['baseline_correct']}/{su['baseline_n']} correct")
    a("");a("SEALED REFERENCES (archived; not produced by this run)");a(Dd)
    f6=TEST610["final"];a(f"TEST610 · {TEST610['decision']} · criteria {f6['criteria_passed']}/{f6['criteria_total']} · holdout {f6['holdout_correct']}/{f6['holdout_total']} · panel {f6['all_panel_correct']}/{f6['all_panel_total']} · {f6['seconds']} s · result {TEST610['result_sha']}")
    a("TEST610 negative controls: "+" · ".join(f"case {r['case']:02d} → {r['raw']!r} leak {r['gold_leak']}"for r in TEST610["negatives"]))
    A=TEST575["arms"];cf=TEST575["counterfactual"];a("TEST575 (different experiment) · "+" · ".join(f"{k} {v}/72"for k,v in A.items())+f" · counterfactual upper K/V changed {cf['upper_kv_changed']}/72 · answer changed {cf['answer_changed']}/72")
    a("");a("FROZEN MODEL");a(Dd)
    for k_ in("weight_sha_startup","sentinel_startup","sentinel_before","sentinel_after","guard_startup","guard_before","guard_after"):a(f"{k_:<18}: {fz[k_]}")
    a(f"trainable tensors={fz['trainable_tensors']} · lora={fz['lora']} · optimizer={fz['optimizer']} · training_mode={fz['training_mode']} · foreign hooks outside engine calls {fz['foreign_hooks_before']}→{fz['foreign_hooks_after']} · hooks on layers/attention/q_proj {len(fz['residual_hooks_before'])}→{len(fz['residual_hooks_after'])}")
    a(fz["full_weight_method"]);a(fz["guard_method"]);a(fz["sentinel_method"]);a(P["instrumentation"]["forward_counter"]);a(P["instrumentation"]["observer"])
    for t_,k_ in(("DEMONSTRATED IN THIS RUN","demonstrated"),("ARCHIVED","archived"),("NOT ESTABLISHED","not_established"),("TECHNICAL NOTES","notes")):
        a("");a(t_);a(Dd)
        for t in P["scope"][k_]:a("• "+t)
    a("");a("PRE-SEAL CHECKS");a(Dd)
    for c_ in P["checks_pre_seal"]:a("PASS · "+c_)
    a("");a("TIMING (seconds)");a(Dd)
    for k_,v in P["timing"].items():a(f"{k_:<24}: {v:.3f}")
    a(f"peak GPU memory {P['gpu']['peak_allocated_gib']:.2f} GiB");a("");a(f"Raw forward records and the memory profile are in {names['payload']} and {names['jsonl']}.");a(S)
    return"\n".join(o)
def execute_run(I,ctl):
    """Fail-closed wrapper: on a technical audit failure the raw outputs gathered so far are preserved (never sealed, no figures)."""
    state={}
    try:return(yield from _execute_run(I,ctl,state))
    except AuditFail as ex:
        raw=state.get("raw")
        if raw and any(v is not None for v in raw.values()):
            try:
                p=Path(ctl["run_dir"])/f"FAILED_AUDIT_RAW_{ctl['run_id']}.json"
                p.write_bytes(canon(dict(status="FAILED AUDIT — NOT SEALED",run_id=ctl["run_id"],failed_check=str(ex),checks_passed=state.get("checks",[]),raw=raw,note="Raw outputs preserved. No figures, no seal and no verdict were produced.")));ex.partial=str(p)
            except Exception:pass
        raise
    finally:
        try:I.release()
        except Exception:pass
def _execute_run(I,ctl,state):
    run_id=ctl["run_id"];run_dir=Path(ctl["run_dir"]);case=int(ctl["case"]);custom=(ctl.get("custom")or"").strip()
    info=I.info;checks=[];tm=Counter();T0=time.perf_counter();state["checks"]=checks
    raw=dict(retrieval=None,trace=None,answer=None,repeat=None,disk=None,reload=None,negative=None,custom=None);state["raw"]=raw
    def chk(name,cond):
        checks.append(name)
        if not cond:raise AuditFail(name)
    run_start_utc=utc_now();say("="*140);say(f"RUN {run_id} · case {case:02d}");say("="*140)
    # ---- 1 integrity ----
    yield ev(1,"Integrity","Engine and helper hashes, source blobs, locked panel, FRAM lock, start-up gates, weight sentinel, all-parameter guard, hooks.")
    I.reset_peak();cfg=I.config();st0=I.status();su=I.startup();fz0=I.frozen();t=time.perf_counter();g_before=I.guard();tm["guard"]+=time.perf_counter()-t
    chk("610.py git blob = repository blob · SHA-256 = embedded reference",cfg["engine_blob"]==ENGINE_BLOB_EXPECTED and cfg["engine_sha256"]==ENGINE_SHA256_EXPECTED)
    chk("RAW COLAB DEMO helper section SHA-256 = embedded reference",cfg["helper_sha256"]==HELPER_SHA256_EXPECTED)
    chk("566.py and 575.py git blobs verified by 610.py = expected",cfg["source_blobs"]==SOURCE_BLOBS_EXPECTED==cfg["source_blobs_measured"])
    chk(f"model, architecture, CUT{CUT_EXPECTED} and {PREFIX_EXPECTED}-token prefix",cfg["MODEL_ID"]==MODEL_ID and tuple(cfg["arch"])==ARCH and cfg["CUT"]==CUT_EXPECTED and cfg["prefix_tokens"]==PREFIX_EXPECTED)
    chk("locked panel (SHA-256) · 96 cartridges · 24 cases",cfg["panel_sha"]==EXPECTED_PANEL_SHA and cfg["n_cartridges"]==N_CART and cfg["n_cases"]==N_CASES)
    chk("FRAM configuration locked on calibration cases 0–11 = (layer 3, rank 0, β 8) · reference M = D = 8192",tuple(cfg["fram_cfg"])==FRAM_LOCK and (cfg["M"],cfg["D"])==(M_REF,D_REF)and cfg["train"]==list(TRAIN_CASES))
    chk("start-up FRAM in this runtime: holdout R@1 12/12 · R@5 12/12 · all R@5 24/24 · self-retrieval 96/96",(su["hold_r1"],su["hold_r5"],su["all_r5"],su["self_r1"])==(12,12,24,96))
    chk("start-up reference ranking 11/12 · 12/12 and baseline CUT3 audit 12/12",(su["ref_hold_r1"],su["ref_hold_r5"],su["baseline_correct"])==(TEST610["reference"]["hold_r1"],TEST610["reference"]["hold_r5"],12))
    chk("full-weight SHA-256 at start-up = reference",info["weight_sha0"]==WEIGHT_SHA_EXPECTED)
    chk("weight sentinel before run = value at initialisation",fz0["sentinel"]==info["sentinel0"]);chk("all-parameter guard before run = value at start-up",g_before==info["guard0"])
    chk("trainable tensors = 0, eval mode, no LoRA, no optimizer",fz0["trainable_tensors"]==0 and not fz0["training"]and not fz0["lora"]and not fz0["optimizer"])
    chk("no foreign hooks outside engine calls · no hooks on decoder layers, attention or q_proj",fz0["foreign_hooks"]==0 and not fz0["residual_hooks"])
    chk("case in 0..23",0<=case<N_CASES)
    tgt=I.target(case);question=I.question(case);expected=I.car(tgt)["gold"];split=split_of(case);strip=I.panel_strings();state["strip"]=strip
    for c in checks:say("   PASS ·",c)
    # ---- 2 FRAM retrieval ----
    yield ev(2,"FRAM retrieval","The engine's exact FRAM kernel scores the question features against all 96 cartridge keys (features captured at start-up); the locked reference ranking selects the five cartridges for CUT3.")
    t=time.perf_counter();fs=I.fram_scores(case);fo=I.rankings(fs);rs=I.ref_scores(case);ro=I.rankings(rs);tm["retrieval"]+=time.perf_counter()-t;shapes=I.fram_shapes(case)
    chk("96 finite exact-FRAM scores and 96 finite reference scores",len(fs)==len(rs)==N_CART and all(math.isfinite(x)for x in fs+rs))
    srow=su["cases"][case];frank=fo.index(tgt)+1;rrank=ro.index(tgt)+1
    chk("exact-FRAM scores recomputed in this run from the start-up features reproduce the start-up ranking (rank and top-5; consistency check)",srow["case"]==case and srow["target"]==tgt and frank==srow["rank"]and fo[:5]==srow["top5"])
    selected=ro[:TOP_K];order=sorted(selected);present=tgt in selected
    chk("five distinct cartridge ids in 0..95 selected by the locked reference ranking",len(set(selected))==TOP_K and all(0<=i<N_CART for i in selected))
    best_other=max(fs[i]for i in range(N_CART)if i!=tgt)
    retrieval=dict(fram=dict(scores=fs,order=fo,rank=frank,top5=fo[:5],top10=fo[:10],target_score=fs[tgt],best_other_score=best_other,margin=fs[tgt]-best_other),
                   ref=dict(scores=rs,order=ro,rank=rrank,top5=selected),target_present=present,shapes=dict(query_rows=shapes["query"][0],dim=shapes["query"][1],key_rows=[k[0]for k in shapes["keys"]]))
    raw["retrieval"]=dict(fram_rank=frank,ref_rank=rrank,selected=selected);sealed_row=TEST610["cases"][case]
    cars=[dict(I.car(ci),is_target=ci==tgt,fram_rank=fo.index(ci)+1,ref_rank=ro.index(ci)+1)for ci in order]
    say(f"   exact FRAM rank {frank} · reference rank {rrank} · selected {selected} · target present {present}")
    # ---- 3 CUT3 write + append-only consolidation ----
    yield ev(3,"CUT3 write + append-only consolidation","build_cut3(): the five selected cartridges are written and appended one by one; every model forward is recorded and prior rows are compared bitwise.")
    P_=cfg["prefix_tokens"];mem,trace,dt=I.build(order);raw["trace"]=trace;tm["build_cut3"]+=dt
    steps,ids_fw=check_build(chk,I,trace,order,P_,"memory")
    T=I.memory_len(mem);chk("final memory length = prefix + all five body lengths",T==steps[-1]["new_len"]==P_+sum(c["body_len"]for c in cars))
    mem_sha=I.memory_sha(mem);chk("final memory SHA-256 = SHA-256 recorded after the last append",mem_sha==steps[-1]["memory_sha"])
    prof=I.memory_profile(mem);chk(f"memory = {ARCH[0]} layers × (K, V) of shape [1, {ARCH[3]}, T, {ARCH[4]}]",prof["layers"]==ARCH[0]and prof["T"]==T and prof["shape"]==[1,ARCH[3],T,ARCH[4]])
    chk("memory bytes = layers × 2 × KV heads × head dim × T × element size",prof["bytes"]==ARCH[0]*2*ARCH[3]*ARCH[4]*T*(2 if prof["dtype"]in("bfloat16","float16")else 4))
    say(f"   memory {T} tokens · 5 steps · prior rows bitwise unchanged at every append · {1000*dt:.0f} ms")
    # ---- 4 source-free question ----
    yield ev(4,"Source-free question","answer(): the question template is processed over the numerical memory; the source text is not given to the model.")
    a1,fw_a,dt=I.ask(mem,question);raw["answer"]=a1;tm["answer"]+=dt;qids=check_query(chk,I,fw_a,question,T,"answer")
    correct=I.hit(a1,expected);cr=hit_ans(a1,expected);chk("correctness re-derived from the answer text = engine hit()",cr==correct)
    a2,fw_a2,dt2=I.ask(mem,question);raw["repeat"]=a2;tm["answer_repeat"]+=dt2
    chk("repeat answer(): same forward pattern and identical answer (greedy decoding)",a2==a1 and fw_a2[0]["ids"]==qids and len(fw_a2)==len(fw_a))
    sha_q=I.memory_sha(mem);chk("memory unchanged by querying (SHA-256 before = after)",sha_q==mem_sha)
    sealed_ans=sealed_row["answer"];say(f"   answer {a1!r} · expected {expected} · {'CORRECT'if correct else'INCORRECT'} · {len(qids)} query tokens over {T}-token memory")
    # ---- 5 disk persistence ----
    yield ev(5,"Save → reload → answer","save_session() writes the CUT3 memory to a file; load_session() reads it back; the K/V tensors are compared bitwise and the question is answered again from the reloaded memory.")
    spath=run_dir/f"{run_id}_CUT3_SESSION.pt";t=time.perf_counter();sinfo=I.save(mem,order,spath);tm["save"]+=time.perf_counter()-t
    fb=spath.read_bytes();fsha=hashlib.sha256(fb).hexdigest()
    chk("session file written: size and SHA-256 recomputed from disk = save_session() report",file_ready(spath)and len(fb)==int(sinfo["bytes"])and fsha==sinfo["sha256"])
    facts=[c["fact"]for c in cars];fact_in=any(f.encode("utf-8")in fb for f in facts);gold_in=expected.encode("utf-8")in fb
    chk("the session file contains none of the five fact texts and not the question text",not fact_in and question.encode("utf-8")not in fb)
    t=time.perf_counter();mem2,ids2=I.load(spath);tm["load"]+=time.perf_counter()-t
    chk("reloaded cartridge ids = saved ids (engine order)",ids2==order)
    beq=I.equal(mem,mem2);chk(f"reloaded K/V bitwise equal to the original in all {ARCH[0]} layers",beq)
    sha2=I.memory_sha(mem2);chk("reloaded memory SHA-256 = original memory SHA-256",sha2==mem_sha)
    a3,fw_a3,dt3=I.ask(mem2,question);tm["answer_reload"]+=dt3;check_query(chk,I,fw_a3,question,T,"answer after reload")
    chk("answer from the reloaded memory = answer before saving",a3==a1)
    disk=dict(file=spath.name,bytes=len(fb),sha256=fsha,sha_recomputed=True,memory_bytes=prof["bytes"],fact_text_in_file=fact_in,gold_in_file=gold_in,question_in_file=False,
              ids_saved=order,ids_loaded=ids2,ids_equal=ids2==order,bit_equal=beq,memory_sha_equal=sha2==mem_sha,answer=a3,answer_identical=a3==a1,forwards=fw_a3,
              query_tokens=len(qids),format="AKBASCORE_CUT3_SESSION_V1",save_seconds=tm["save"],load_seconds=tm["load"],answer_seconds=dt3)
    raw["disk"]=dict(sinfo,file=spath.name);raw["reload"]=a3;I.release(mem2);del mem2
    # ---- 6 controls ----
    yield ev(6,"Negative control","The same question over the CUT3 memory built without the target cartridge (diagnostic; leakage of the expected answer is reported)."+(" Then the free-text question." if custom else""))
    if present:
        rest=sorted(ci for ci in selected if ci!=tgt);memN,traceN,dtN=I.build(rest);tm["build_control"]+=dtN
        check_build(chk,I,traceN,rest,P_,"control");TN=I.memory_len(memN);chk("control memory excludes the target cartridge",tgt not in rest and len(rest)==TOP_K-1)
        aN,fwN,dtN2=I.ask(memN,question);tm["answer_control"]+=dtN2;check_query(chk,I,fwN,question,TN,"control")
        leak=I.hit(aN,expected);negative=dict(ids=rest,T=TN,answer=aN,gold_leak=leak,target_absent=True,memory_bytes=I.memory_bytes(memN));I.release(memN);del memN
    else:negative=dict(skipped="target not in the selected five cartridges; the main memory itself is the target-absent condition",ids=selected,gold_leak=None)
    raw["negative"]=negative;cu=None
    if custom and custom!=question:
        ids_c,rk,fq,dq=I.retrieve_free(custom);tm["retrieve_free"]+=dq
        chk("free-text retrieval: one question-frame forward (instruction prefix + question), no cache, no fact text",len(fq)==1 and fq[0]["kind"]=="ids"and fq[0]["ids"]==I.frame_ids(custom)and not fq[0]["has_past"])
        chk("free-text retrieval: five distinct ids in 0..95 from the exact FRAM kernel",len(set(ids_c))==TOP_K and all(0<=i<N_CART for i in ids_c))
        oc=sorted(ids_c);memC,traceC,dtC=I.build(oc);tm["build_free_text"]+=dtC;check_build(chk,I,traceC,oc,P_,"free text");TC=I.memory_len(memC)
        aC,fwC,dtC2=I.ask(memC,custom);tm["answer_free_text"]+=dtC2;cq=check_query(chk,I,fwC,custom,TC,"free text")
        strip=strip+[v for x in(custom,str(aC))for v in(x," ".join(x.split()),json.dumps(x,ensure_ascii=False)[1:-1])];state["strip"]=strip
        cu=dict(question=custom,answer=aC,ids=ids_c,ranking=rk,T=TC,query_tokens=len(cq),gen_forwards=len(fwC)-1,seconds=dtC2,forwards=fwC,frame_tokens=fq[0]["n"]);raw["custom"]=dict(answer=aC,ids=ids_c)
        I.release(memC);del memC
    # ---- 7 verification ----
    yield ev(7,"Verification","Weight sentinel, all-parameter guard, trainable tensors and hooks after the run.")
    fz1=I.frozen();t=time.perf_counter();g_after=I.guard();tm["guard"]+=time.perf_counter()-t
    chk("all-parameter guard after run = before run = start-up (weights unchanged)",g_after==g_before==info["guard0"]);chk("weight sentinel after run = value at initialisation",fz1["sentinel"]==info["sentinel0"])
    chk("trainable tensors = 0, no foreign hooks and no hooks on layers/attention/q_proj after run",fz1["trainable_tensors"]==0 and fz1["foreign_hooks"]==0 and not fz1["residual_hooks"])
    tm["engine_total"]=time.perf_counter()-T0
    verdict=f"CASE_{case:02d}_{'CORRECT'if correct else'INCORRECT'}_CUT3_FRAM_DISK_RELOAD_VERIFIED_SOURCE_FREE_QUERY"
    # ---- 8 seal ----
    yield ev(8,"Sealing","Payload, manifest, readable log and raw records are written and hashed.")
    peak=I.peak_gib();same_hw=info["gpu"]==CANON_GPU
    live=dict(case=case,split=split,question=question,expected=expected,target=tgt,n_cartridges=N_CART,retrieval=retrieval,selected=selected,engine_order=order,cartridges=cars,
              steps=steps,forwards_build=trace["forwards"],source_reads=len(ids_fw),zero_embed_forwards=sum(1 for f in trace["forwards"]if f["kind"]=="embeds"and f["embeds_zero"]),
              memory=prof,memory_sha=mem_sha,segments=segments(steps,P_),answer=a1,answer_repeat=a2,correct=bool(correct),correct_rederived=bool(cr),repeat_identical=a2==a1,
              memory_unchanged_by_query=sha_q==mem_sha,query_tokens=len(qids),query_token_text=I.decode_each(qids),gen_forwards=len(fw_a)-1,forwards_answer=fw_a,answer_seconds=dt,
              disk=disk,negative=negative,custom=cu,sealed_answer=sealed_ans,sealed_correct=bool(sealed_row["correct"]),matches_sealed=norm_ans(a1)==norm_ans(sealed_ans),
              session=[dict(r,correct=(bool(correct)if r.get("correct")is None else bool(r["correct"])))for r in(ctl.get("session")or[])])
    P={"schema":"akbascore.mam.bc.qwen.cut3_fram.demo.run.v1","demo":DEMO_TITLE,"author":AUTHOR,"organisation":ORG,"date":DATE_ISO,"place":AUTHOR_PLACE,"copyright":COPYRIGHT,
       "license":{"name":LICENSE_NAME,"url":LICENSE_URL},"priority_record":PRIORITY,"paradigm":PARADIGM,"discovery":DISCOVERY,"question":SCI_QUESTION,"core_message":CORE_MESSAGE,
       "earlier_records":{"v1.0":PRIOR_PUB,"v1.1":QWEN_PUB,"note":"cited by DOI only; no value of an earlier record is used as a result here"},
       "run_id":run_id,"run_start_utc":run_start_utc,"run_end_utc":utc_now(),"verdict":verdict,
       "model":{"id":cfg["MODEL_ID"],"arch":info["arch"],"dtype":info["dtype"],"attn":info["attn"],"params":info["params"]},
       "environment":{k_:info[k_]for k_ in("gpu","gpu_total_gib","torch","transformers","gradio","python","platform")},"hardware":{"reference_gpu":CANON_GPU,"this_run_gpu":info["gpu"],"same":bool(same_hw)},
       "engine":{"file":info["engine_file"],"sha256":cfg["engine_sha256"],"blob":cfg["engine_blob"],"exec_sha256":cfg["engine_exec_sha256"],"tail_sha256":info["engine_tail_sha256"],"helper_sha256":cfg["helper_sha256"],
                 "raw_demo_file_sha256":info["raw_demo_file_sha256"],"config":cfg,"status_before":st0,"init_seconds":info["init_seconds"],"instrument":I.kind,
                 "calls":"build_cut3(ids, 3) → validate_cut3 → answer(memory, q); save_session / load_session / compare_mem_exact; pair_scores; PRED (locked reference ranking)",
                 "source":ENGINE_URL,"log":ENGINE_LOG_URL,"base":BASE_URL,"lock_program":LOCK_URL},
       "instrumentation":{"forward_counter":info["forward_counter"],"observer":info["observer"]},"startup":su,"live":live,
       "frozen":{"weight_sha_startup":info["weight_sha0"],"weight_sha_reference":WEIGHT_SHA_EXPECTED,"full_weight_method":info["full_weight_method"],"sentinel_startup":info["sentinel0"],"sentinel_before":fz0["sentinel"],"sentinel_after":fz1["sentinel"],"guard_startup":info["guard0"],"guard_before":g_before,"guard_after":g_after,
                 "trainable_tensors":fz1["trainable_tensors"],"lora":fz1["lora"],"optimizer":fz1["optimizer"],"training_mode":fz1["training"],"foreign_hooks_before":fz0["foreign_hooks"],
                 "foreign_hooks_after":fz1["foreign_hooks"],"residual_hooks_before":fz0["residual_hooks"],"residual_hooks_after":fz1["residual_hooks"],"guard_method":info["guard_method"],"sentinel_method":info["sentinel_method"]},
       "archived":{"test610":TEST610,"test575":TEST575,"arm_description":ARM_DESC,"note":"Sealed reference values; not produced or recomputed by this run. TEST575 is a different experiment."},
       "archived_gold":{str(r["case"]):I.car(r["target"])["gold"]for r in TEST610["cases"]},"scope":SCOPE,"checks_pre_seal":list(checks),"timing":dict(tm),"gpu":{"peak_allocated_gib":peak}}
    P=jsafe(P);allow=(info["gpu"],CANON_GPU)
    hits=claim_scan("payload",strip_words(json.dumps(P,ensure_ascii=False),strip),allow=allow)
    chk("claim-discipline and foreign-model scan of the payload: 0 hits",not hits);P["claim_scan"]={"rules":len(CLAIM_RULES)+1+len(LEAK_RULES),"hits":0,"scope":"payload JSON text; panel names and free text excluded; earlier-record citation exempt"}
    pb=canon(P);sha=hashlib.sha256(pb).hexdigest();payload_name=f"{run_id}_FULL_RUN_LOG_payload.json";pp=run_dir/payload_name;t_seal=time.perf_counter();pp.write_bytes(pb);say("payload sealed:",sha)
    chk("payload SHA-256 verification after write",hashlib.sha256(pp.read_bytes()).hexdigest()==sha)
    names=dict(payload=payload_name,manifest=f"run_manifest_{run_id}.json",txt=f"{run_id}_READABLE_RUN_LOG.txt",summary=f"{run_id}_SUMMARY.json",jsonl=f"{run_id}_RAW_RECORDS.jsonl",images=f"figures_manifest_{run_id}.json",zip=f"AKBASCORE_MAM_BC_QWEN_EVIDENCE_{run_id}.zip")
    mp=run_dir/names["manifest"];sealed_utc=utc_now()
    manifest={"demo":DEMO_TITLE,"author":AUTHOR,"organisation":ORG,"date":DATE_ISO,"copyright":COPYRIGHT,"license":LICENSE_URL,"priority_record":PRIORITY,"run_id":run_id,"payload_file":payload_name,
              "payload_sha256":sha,"payload_bytes":len(pb),"sealed_utc":sealed_utc,"verdict":verdict,"engine_sha256":cfg["engine_sha256"],"engine_blob":cfg["engine_blob"],"helper_sha256":cfg["helper_sha256"],
              "test610_result_sha":TEST610["result_sha"],"test575_result_sha":TEST575["result_sha"],"weight_sha_startup":info["weight_sha0"],"model":MODEL_ID,
              "cut3_session_file":spath.name,"cut3_session_sha256":fsha,"cut3_session_bytes":len(fb),
              "canonicalization":"json.dumps(sort_keys=True, separators=(',',':'), ensure_ascii=False, allow_nan=False), UTF-8",
              "note":"Artifact integrity seal: confirms the payload is unchanged since sealing. It is not a scientific proof and not a third-party verification."}
    mp.write_text(json.dumps(manifest,indent=2,sort_keys=True,ensure_ascii=False),encoding="utf-8")
    summary={"run_id":run_id,"verdict":verdict,"case":case,"split":split,"question":question,"answer":a1,"expected":expected,"correct":bool(correct),"target":tgt,
             "fram_rank":frank,"reference_rank":rrank,"selected":selected,"engine_order":order,"memory_tokens":T,"memory_bytes":prof["bytes"],"memory_growth":[s["new_len"]for s in steps],
             "prior_rows_bitwise_unchanged":[s["prior_rows_bitwise_unchanged"]for s in steps[1:]],"session_file":spath.name,"session_bytes":len(fb),"session_sha256":fsha,"reload_bit_equal":beq,
             "reload_answer_identical":a3==a1,"negative_control":{k_:negative.get(k_)for k_ in("ids","answer","gold_leak","skipped")},"weights_unchanged":True,"payload_sha256":sha,"model":MODEL_ID,
             "sealed_TEST610_answer":sealed_ans,"live_answer_matches_sealed":norm_ans(a1)==norm_ans(sealed_ans)}
    (run_dir/names["summary"]).write_text(json.dumps(jsafe(summary),indent=2,ensure_ascii=False),encoding="utf-8")
    lines=[{"op":"retrieval","case":case,"fram_scores":fs,"reference_scores":rs,"selected":selected}]
    lines+=[{"op":"step","n":n_+1,"kind":s["kind"],"cartridge_id":s["cartridge_id"],"old_len":s["old_len"],"new_len":s["new_len"],"prior_rows_bitwise_unchanged":s["prior_rows_bitwise_unchanged"],
            "memory_sha":s["memory_sha"],"forwards":s["forwards"]}for n_,s in enumerate(steps)]
    lines+=[{"op":"answer","question":question,"answer":a1,"forwards":fw_a},{"op":"save","file":spath.name,"bytes":len(fb),"sha256":fsha},{"op":"reload_answer","answer":a3,"bit_equal":beq,"forwards":fw_a3},
            {"op":"negative_control",**{k_:v for k_,v in negative.items()}},{"op":"memory_profile","k_norm":prof["k_norm"],"v_norm":prof["v_norm"]}]
    if cu:lines.append({"op":"free_text","question":cu["question"],"answer":cu["answer"],"ids":cu["ids"],"forwards":cu["forwards"]})
    (run_dir/names["jsonl"]).write_text("\n".join(json.dumps(jsafe(x),ensure_ascii=False)for x in lines)+"\n",encoding="utf-8")
    (run_dir/names["txt"]).write_text(make_txt(P,sha,names,sealed_utc),encoding="utf-8");seal_s=time.perf_counter()-t_seal
    # ---- 9 figures ----
    yield ev(9,"Rendering figures",f"{N_FIGS} figures at 300 dpi (JPEG) with vector PDF copies; every printed value is checked against the sealed payload.")
    ctx={"P":P,"sha":sha,"run_id":run_id,"payload_name":payload_name,"allow":allow,"strip":strip,"N":N_FIGS,"audit":[],"pdfs":[],"checks":len(checks)}
    t_r=time.perf_counter();imgs=render_all(ctx,run_dir);render_s=time.perf_counter()-t_r;entries=[]
    for n_,(p,cap)in enumerate(imgs,1):
        with Image.open(p)as im:w_,h_=im.size
        q_=ctx["pdfs"][n_-1];entries.append({"index":n_,"jpg":p.name,"pdf":q_.name,"title":cap,"width_px":w_,"height_px":h_,"dpi":OUT_DPI,"jpg_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"pdf_sha256":hashlib.sha256(q_.read_bytes()).hexdigest()})
    imf=run_dir/names["images"]
    imf.write_text(json.dumps({"run_id":run_id,"payload_sha256":sha,"verdict":verdict,"count":len(entries),"render_seconds":render_s,"figures":entries,"figure_text_audit":ctx["audit"],
                               "note":"Per-file SHA-256 is an artifact integrity seal. Figure text was checked against the sealed payload and scanned for claim discipline before saving."},indent=2,ensure_ascii=False),encoding="utf-8")
    # ---- 10 package ----
    yield ev(10,"Packaging","ZIP with figures, PDFs, payload, manifest, logs, raw records and the CUT3 session file; post-seal audit.")
    zp=run_dir/names["zip"];members=[p for p,_ in imgs]+list(ctx["pdfs"])+[imf,mp,pp,run_dir/names["txt"],run_dir/names["summary"],run_dir/names["jsonl"],spath]
    for m in members:
        if m.suffix in(".json",".txt",".jsonl"):chk(f"claim-discipline and foreign-model scan of {m.name}: 0 hits",not claim_scan(m.name,strip_words(m.read_text(encoding="utf-8"),strip),allow=allow,numeric=m.suffix==".txt"))
    with zipfile.ZipFile(zp,"w",compression=zipfile.ZIP_DEFLATED)as z:
        for m in members:z.write(m,arcname=m.name)
    post=post_seal_audit(imgs,ctx["pdfs"],zp,[m.name for m in members],pp,sha,run_dir/names["txt"],mp,verdict,spath,fsha)
    I.release(mem);del mem
    say(f"{run_id}: {len(checks)} pre-seal + {len(post)} post-seal checks = {len(checks)+len(post)}/{len(checks)+len(post)} PASS · figures {render_s:.1f}s · seal {seal_s:.2f}s")
    say("VERDICT :",verdict);say("PAYLOAD :",sha);say("ZIP     :",zp);say("="*140)
    ledger=dict(run_id=run_id,utc=sealed_utc,case=case,split=split,question=question,answer=a1,expected=expected,correct=bool(correct),memory_tokens=T,selected=order,
                session_sha256=fsha,reload_bit_equal=beq,negative_leak=negative.get("gold_leak"),weights_unchanged=True,verdict=verdict,payload_sha256=sha)
    return dict(run_id=run_id,run_dir=run_dir,P=P,sha=sha,imgs=imgs,pdfs=ctx["pdfs"],zip=zp,payload=pp,txt=run_dir/names["txt"],manifest=mp,images_manifest=imf,summary=run_dir/names["summary"],
                jsonl=run_dir/names["jsonl"],session=spath,checks=len(checks)+len(post),verdict=verdict,ledger=ledger)
# ---------------- optional LIVE replay of the TEST610 acceptance stage (610.py [6/7]–[7/7] verbatim; reported separately) ----------------
def replay_report(I,run_id,run_dir,progress=None):
    info=I.info;t0=time.perf_counter();start=utc_now();g0=I.guard()
    if g0!=info["guard0"]:raise AuditFail("all-parameter guard before replay differs from start-up")
    R=I.acceptance_replay(progress);g1=I.guard()
    if g1!=g0:raise AuditFail("all-parameter guard changed during replay")
    rows=R["cases"];sealed={r["case"]:r for r in TEST610["cases"]}
    if len(rows)!=24:raise AuditFail("replay did not report 24 cases")
    score=sum(int(bool(r.get("correct")))for r in rows);hold=sum(int(bool(r.get("correct")))for r in rows if r.get("split")=="HOLD")
    if score!=R["summary"]["all_panel_correct"]or hold!=R["summary"]["holdout_correct"]:raise AuditFail("replay scores do not re-derive from its rows")
    cmp_=[dict(case=r["case"],split=r.get("split"),answer=r.get("answer"),correct=bool(r.get("correct")),sealed_answer=sealed[r["case"]]["answer"],sealed_correct=sealed[r["case"]]["correct"],
               identical=norm_ans(r.get("answer",""))==norm_ans(sealed[r["case"]]["answer"]))for r in rows]
    same=sum(int(c["identical"])for c in cmp_);crit=R["criteria"];stdout_name=f"{run_id}_ENGINE_STDOUT.txt";(run_dir/stdout_name).write_text(R["stdout"],encoding="utf-8")
    eng_json=None
    if R.get("out_path")and Path(R["out_path"]).is_file():
        eng_json=run_dir/Path(R["out_path"]).name;shutil.copy(R["out_path"],eng_json)
    out={"schema":"akbascore.mam.bc.qwen.test610_acceptance_replay.v1","model":MODEL_ID,"demo":DEMO_TITLE,"author":AUTHOR,"organisation":ORG,"date":DATE_ISO,"license":LICENSE_URL,"run_id":run_id,"start_utc":start,"end_utc":utc_now(),
         "status":"LIVE REPLAY of the 610.py acceptance stage — separate from the sealed TEST610 record","decision":R["decision"],"criteria_passed":sum(bool(v)for v in crit.values()),"criteria_total":len(crit),
         "criteria":crit,"panel_correct":score,"panel_total":24,"holdout_correct":hold,"holdout_total":12,"answers_identical_to_sealed_TEST610":same,"negatives":R["negatives"],"rebuild":R["rebuild"],"append":R["append"],
         "engine_result_sha":R["result_sha"],"engine_json":None if eng_json is None else eng_json.name,"engine_json_sha256":None if eng_json is None else hashlib.sha256(eng_json.read_bytes()).hexdigest(),
         "engine_stdout":stdout_name,"engine_stdout_sha256":hashlib.sha256((run_dir/stdout_name).read_bytes()).hexdigest(),"engine_tail_sha256":info["engine_tail_sha256"],
         "engine_sha256":ENGINE_SHA256_EXPECTED,"engine_blob":ENGINE_BLOB_EXPECTED,"guard_before":g0,"guard_after":g1,"weights_unchanged":g1==g0==info["guard0"],"gpu":info["gpu"],"seconds":round(time.perf_counter()-t0,2),
         "rows":cmp_,"sealed_reference":{"name":"TEST610","decision":TEST610["decision"],"final":TEST610["final"],"result_sha":TEST610["result_sha"]}}
    pb=canon(out);sha=hashlib.sha256(pb).hexdigest();jp=run_dir/f"{run_id}_ACCEPTANCE_REPLAY.json";jp.write_bytes(pb)
    S="="*120;txt=[S,f"{DEMO_TITLE} — {MODEL_FAMILY} · LIVE REPLAY OF THE TEST610 ACCEPTANCE STAGE (610.py [6/7]–[7/7], verbatim)",f"{DISCOVERY} · {ORG} · {DATE_TXT} · {LICENSE_NAME} · {LICENSE_URL}",S,
         "This replay is a LIVE measurement in this runtime. It is reported separately from the sealed TEST610 record and does not replace it.",
         f"run {run_id} · {start} → {out['end_utc']} · {out['seconds']} s · GPU {info['gpu']}",
         f"LIVE decision {out['decision']} · criteria {out['criteria_passed']}/{out['criteria_total']} · panel {score}/24 · holdout {hold}/12 · answers identical to sealed TEST610: {same}/24",
         f"sealed TEST610: {TEST610['decision']} · {TEST610['final']['criteria_passed']}/{TEST610['final']['criteria_total']} · panel {TEST610['final']['all_panel_correct']}/24 · result SHA {TEST610['result_sha']}",
         f"weights unchanged (all-parameter guard before = after = start-up): {out['weights_unchanged']} · engine stdout {stdout_name} (verbatim)",f"payload SHA-256 {sha}","-"*120]
    txt+=[f"criterion {k:<34} {'PASS'if v else'FAIL'}"for k,v in crit.items()]+["-"*120]
    txt+=[f"case {c['case']:02d} {str(c['split']):<4} answer {c['answer']!r} {'CORRECT'if c['correct']else'INCORRECT'} · sealed {c['sealed_answer']!r} {'=' if c['identical'] else '≠'}"for c in cmp_]
    txt+=["-"*120]+[f"negative control case {r.get('case')} → {r.get('raw')!r} · expected answer produced: {r.get('gold_leak')}"for r in R["negatives"]]+[S]
    tp=run_dir/f"{run_id}_ACCEPTANCE_REPLAY.txt";tp.write_text("\n".join(txt),encoding="utf-8")
    for p_ in(jp,tp):
        h_=claim_scan(p_.name,strip_words(p_.read_text(encoding="utf-8"),I.panel_strings()),allow=(info["gpu"],CANON_GPU),numeric=p_.suffix==".txt")
        if h_:raise AuditFail(f"claim-discipline hit in {p_.name}: {h_[:3]}")
    return dict(out=out,sha=sha,json=jp,txt=tp,stdout=run_dir/stdout_name)
#<<CORE_END>>
#<<FIGURES_BEGIN>>
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":.8,"axes.edgecolor":"#374151","axes.labelcolor":"#111827","xtick.color":"#374151","ytick.color":"#374151",
                     "xtick.labelsize":8,"ytick.labelsize":8,"axes.labelsize":9,"axes.titlesize":9.5,"axes.titleweight":"bold","axes.titlelocation":"left","pdf.fonttype":42,"ps.fonttype":42,
                     "legend.fontsize":8,"legend.frameon":False,"mathtext.fontset":"dejavusans"})
FIG_W,FIG_H=13.333,7.5;LAYOUT_DPI=100;OUT_DPI=300
INK,INK2,MUTED,RULE,GRIDC,PANEL="#111827","#374151","#6B7280","#D1D5DB","#E5E7EB","#F9FAFB"
BLUE,BLUE_L,ORANGE,ORANGE_L,TEAL,TEAL_L,GRAYPT,AMBER_L,AMBER,RED="#1D4ED8","#DBEAFE","#C2410C","#FFEDD5","#0F766E","#CCFBF1","#9CA3AF","#FEF3C7","#B45309","#B91C1C"
VIOLET,VIOLET_L="#6D28D9","#EDE9FE"
SEGC=["#9CA3AF","#1D4ED8","#0F766E","#7C3AED","#B45309","#BE185D"]
MONO="DejaVu Sans Mono"
def mt(s):return str(s).replace("$",r"\$")
def ascii_(s):return unicodedata.normalize("NFKD",str(s)).encode("ascii","ignore").decode("ascii")
def new_fig():return plt.figure(figsize=(FIG_W,FIG_H),dpi=LAYOUT_DPI,facecolor="white")
def sanitize_dashes(fig):
    """Zero-width artists are forced to a solid style: some matplotlib builds raise on a dashed pattern scaled to zero width."""
    for a in fig.findobj(lambda o:isinstance(o,(Patch,Line2D))):
        try:
            if a.get_linewidth()==0 and a.get_linestyle()not in("-","solid","None","none"," ",""):a.set_linestyle("-")
        except Exception:pass
def finish(fig,path,ctx,expect,desc):
    """Every expected value must appear in the figure text; claim scan must be clean; then JPEG (300 dpi, EXIF provenance) + vector PDF."""
    sanitize_dashes(fig);fig.canvas.draw();texts=[t.get_text()for t in fig.findobj(Text)if t.get_text().strip()];blob="\n".join(texts);nm=Path(path).name
    ws=lambda z:re.sub(r"\s+","",z);bw=ws(blob);miss=[e for e in expect if ws(e)not in bw]
    if miss:plt.close(fig);raise AuditFail(f"FIGURE/PAYLOAD MISMATCH in {nm}: missing {miss}")
    hits=claim_scan(nm,strip_words(re.sub(r"\s*\n\s*"," ",blob),ctx.get("strip",[])),allow=ctx["allow"],numeric=True)
    if hits:plt.close(fig);raise AuditFail(f"CLAIM-DISCIPLINE HIT in {nm}: {hits[:3]}")
    pdf=Path(path).with_suffix(".pdf")
    fig.savefig(pdf,format="pdf",facecolor="white",metadata={"Title":desc,"Author":f"{AUTHOR} · {ORG}","Subject":f"{DEMO_TITLE} · run {ctx['run_id']} · payload SHA-256 {ctx['sha']}","Creator":DEMO_SHORT,
                "Keywords":f"model {MODEL_ID}; engine 610.py git blob {ENGINE_BLOB}; engine sha256 {ENGINE_SHA256}; TEST610 result {TEST610['result_sha']}; license {LICENSE_URL}"})
    buf=io.BytesIO();fig.savefig(buf,format="png",dpi=OUT_DPI,facecolor="white");plt.close(fig);buf.seek(0)
    with Image.open(buf)as im:rgb=im.convert("RGB")
    ex=Image.Exif();ex[0x010E]=ascii_(f"{desc} | {DEMO_SHORT} | run {ctx['run_id']} | payload sha256 {ctx['sha']} | engine sha256 {ENGINE_SHA256}");ex[0x013B]=ascii_(f"{AUTHOR} / {ORG}");ex[0x8298]=ascii_(f"{COPYRIGHT} - {LICENSE_NAME} - {LICENSE_URL}");ex[0x0131]=ascii_(DEMO_SHORT)
    rgb.save(path,"JPEG",quality=95,optimize=True,progressive=False,subsampling=0,dpi=(OUT_DPI,OUT_DPI),exif=ex.tobytes())
    with Image.open(path)as c_:
        if c_.format!="JPEG"or c_.mode!="RGB":raise RuntimeError("JPEG validation failed")
    ctx["pdfs"].append(pdf);ctx["audit"].append(dict(file=nm,texts=len(texts),expected_values=len(expect),claim_hits=0));return str(path)
def fit_text(fig,x,y,w,h,text,fs_max=10,fs_min=6.5,color=INK,family=None,ls=1.3,weight="normal",ha="left"):
    fig.canvas.draw();r=fig.canvas.get_renderer();Wp=w*fig.get_figwidth()*fig.dpi;Hp=h*fig.get_figheight()*fig.dpi
    if Wp<=4 or Hp<=4:return 0
    paras=str(text if text else"(empty)").replace("\r","").split("\n");x0=x+w/2 if ha=="center"else x
    kw={"va":"top","ha":ha,"color":color,"linespacing":ls,"weight":weight,"multialignment":ha}
    if family:kw["family"]=family
    def wrap(c):
        out=[]
        for p in paras:out.extend(textwrap.wrap(p,c,break_long_words=True,break_on_hyphens=False,subsequent_indent=("  "if p.startswith("• ")else""))or[""])
        return out
    fs=float(fs_max);k=0.52
    for _ in range(150):
        cpl=max(6,int(Wp/(k*fs*fig.dpi/72.0)));ln=wrap(cpl);t=fig.text(x0,y+h,mt("\n".join(ln)),fontsize=fs,**kw);bb=t.get_window_extent(renderer=r)
        if bb.width>Wp*1.002 and cpl>6:t.remove();k*=max(1.01,bb.width/Wp*1.01);continue
        if bb.height<=Hp:return fs
        t.remove()
        if fs>fs_min:fs=max(float(fs_min),fs-0.25);continue
        per=bb.height/max(1,len(ln));keep=max(1,int(Hp/per)-1);fig.text(x0,y+h,mt("\n".join(ln[:keep]+["[… continued in the readable run log]"])),fontsize=fs,**kw);return fs
    fig.text(x0,y+h,"[text omitted — see run log]",fontsize=fs_min,**kw);return fs_min
def header(fig,k,title,sub,tag,tagc):
    fig.text(.03,.955,f"Fig. {k}",fontsize=12,weight="bold",color=MUTED,va="center")
    fig.text(.078,.955,mt(title),fontsize=16.5,weight="bold",color=INK,va="center")
    fig.text(.03,.917,mt(sub),fontsize=9.4,color=INK2,va="center")
    fig.text(.97,.955,tag,fontsize=8.4,weight="bold",color=tagc,va="center",ha="right",bbox=dict(boxstyle="square,pad=0.35",fc="white",ec=tagc,lw=1.0))
    fig.add_artist(Line2D([.03,.97],[.895,.895],transform=fig.transFigure,color=INK2,lw=.8))
def footer(fig,ctx,k,caption):
    fig.add_artist(Line2D([.03,.97],[.132,.132],transform=fig.transFigure,color=RULE,lw=.6))
    fit_text(fig,.03,.045,.94,.08,caption,fs_max=8.6,fs_min=6.6,color=INK2)
    fig.add_artist(Line2D([.03,.97],[.038,.038],transform=fig.transFigure,color=RULE,lw=.6))
    fig.text(.5,.025,mt(f"{DEMO_TITLE} · {DISCOVERY} · {ORG} · {DATE_TXT} · {LICENSE_NAME} · {LICENSE_URL}"),ha="center",va="center",fontsize=6.6,color=INK2)
    fig.text(.5,.010,mt(f"live run {ctx['run_id']} · {MODEL_FAMILY} · 610.py engine (blob {ENGINE_BLOB[:12]}…; not TEST610 itself) · TEST610 result {TEST610['result_sha'][:12]}… · payload {ctx['sha'][:12]}… · Fig. {k}/{ctx['N']} · {COPYRIGHT}"),
             ha="center",va="center",fontsize=6.2,color=MUTED,family=MONO)
def clean(ax,grid=True):
    for s in("top","right"):ax.spines[s].set_visible(False)
    if grid:ax.grid(True,color=GRIDC,lw=.5);ax.set_axisbelow(True)
def frame(fig,x,y,w,h,ec=INK2,fc="white",lw=.9,ls="-",z=-1):
    fig.add_artist(Rectangle((x,y),w,h,transform=fig.transFigure,facecolor=fc,edgecolor=ec,lw=lw,ls=ls,zorder=z))
def block(fig,x,y,w,h,stage,title,body,ec=INK2,fc="white",mono=True,bfs=8.4,tfs=9.6,bmin=6.4):
    frame(fig,x,y,w,h,ec,fc);frame(fig,x,y+h-.032,w,.032,ec,PANEL if fc=="white"else fc,lw=.9)
    fig.text(x+.007,y+h-.016,stage,fontsize=7,weight="bold",color=MUTED,va="center")
    fig.text(x+.007,y+h-.052,mt(title),fontsize=tfs,weight="bold",color=INK,va="center")
    if body:fit_text(fig,x+.007,y+.008,w-.014,h-.078,body,fs_max=bfs,fs_min=bmin,family=MONO if mono else None,color=INK2)
def arrow(fig,x0,y0,x1,y1,color=INK2,lw=1.1,ms=11,cs="arc3"):fig.add_artist(FancyArrowPatch((x0,y0),(x1,y1),transform=fig.transFigure,arrowstyle="-|>",mutation_scale=ms,lw=lw,color=color,connectionstyle=cs,zorder=6))
def L_(ctx):return ctx["P"]["live"]
def short(s,n):s=str(s);return s if len(s)<=n else s[:n-1]+"…"
def uwrap(s,n,ind=""):
    """Wrap an identifier at underscores into lines of at most n characters."""
    out=[];cur=""
    for part in str(s).split("_"):
        piece=(cur+"_"+part)if cur else part
        if len(piece)>n and cur:out.append(cur+"_");cur=part
        else:cur=piece
    out.append(cur);return"\n".join(ind+x for x in out)
def okc(b):return TEAL if b else ORANGE
def yn(b):return"yes"if b else"no"
# ------------------------------------------------------------------ FIG 1 · complete system architecture
def f1(ctx,k,path):
    P=ctx["P"];L=L_(ctx);cfg=P["engine"]["config"];R=L["retrieval"];M=L["memory"];AR=cfg["arch"];K=L["disk"];li,rk,be=cfg["fram_cfg"];fig=new_fig()
    tc=next(c for c in L["cartridges"]if c["is_target"])if R["target_present"]else None
    header(fig,k,"External Associative Memory for a Frozen Language Model",f"{MODEL_FAMILY} · case {L['case']:02d} ({L['split']}) · from the source fact to the answer · every number below is taken from this run",
           "SYSTEM ARCHITECTURE · LIVE VALUES",INK2)
    W_=.118;G_=.018;X=[.033+i*(W_+G_)for i in range(7)];yb=.545;hb=.25
    frame(fig,X[1]-.008,.395,X[4]+W_-X[1]+.016,.46,ec=TEAL,fc="#F0FDFA",lw=1.0,ls=(0,(4,2)),z=-3)
    fig.text(X[1],.838,"EXTERNAL MEMORY · numbers kept outside the model (removable; saved to disk in Fig. 4)",fontsize=8.2,weight="bold",color=TEAL,va="center")
    frame(fig,X[5]-.008,.395,W_+.016,.46,ec=INK2,fc="#F3F4F6",lw=1.0,z=-3)
    fig.text(X[5],.838,"FROZEN MODEL",fontsize=8.2,weight="bold",color=INK2,va="center")
    roles=["INPUT","WRITE · STORE","STORE","SELECT","ATTACH","ANSWER","OUTPUT"]
    bodies=[f"{L['n_cartridges']} facts · 24 worlds × 4 relation types, each a plain sentence",
            f"PREFIX + fact\n→ 1 frozen forward\nlayer-{li} residual\nfact span {R['shapes']['key_rows'][L['target']]}×{R['shapes']['dim']}\ncentred, unit norm",
            f"{L['n_cartridges']} cartridges\nkey = layer-{li} rows\nconfig ({li}, {rk}, β {be:g})\nselected:\n"+" ".join(f"#{i:02d}"for i in L["engine_order"]),
            f"question in fact\nframe: {R['shapes']['query_rows']} rows\ntarget rank\n exact   {R['fram']['rank']} / {L['n_cartridges']}\n reference {R['ref']['rank']} / {L['n_cartridges']}\ntop-5 → CUT3",
            f"init + 4 appends\n{M['T']} tokens\n{M['layers']} × (K, V)\n[1,{M['shape'][1]},{M['T']},{M['shape'][3]}]\n{mib(M['bytes']):.2f} MiB {M['dtype']}\nfile {K['bytes']:,} B",
            f"{AR[0]} layers\nhidden {AR[1]}\nweights frozen\ninput at answer:\n{L['query_tokens']} template\ntoken ids only",None]
    titles=["Source facts","Native representation","Cartridge bank","FRAM retrieval","CUT3 memory","Frozen Qwen","Answer"]
    cols=[INK2,TEAL,TEAL,BLUE,TEAL,INK2,BLUE if L["correct"]else ORANGE]
    for i in range(7):
        if i==6:continue
        block(fig,X[i],yb,W_,hb,roles[i],titles[i],bodies[i],ec=cols[i],mono=i!=0,bfs=8.8,tfs=8.2,bmin=6.6)
    frame(fig,X[6],yb,W_,hb,ec=cols[6],fc=BLUE_L if L["correct"]else ORANGE_L,lw=1.3);frame(fig,X[6],yb+hb-.032,W_,.032,ec=cols[6],fc=PANEL,lw=.9)
    fig.text(X[6]+.007,yb+hb-.016,roles[6],fontsize=7,weight="bold",color=MUTED,va="center");fig.text(X[6]+.007,yb+hb-.052,"Answer",fontsize=8.2,weight="bold",color=INK,va="center")
    fit_text(fig,X[6]+.007,yb+.115,W_-.014,.07,L["answer"],fs_max=15,fs_min=8,color=INK,weight="bold")
    fig.text(X[6]+.007,yb+.095,mt(f"expected {short(L['expected'],14)}"),fontsize=7.8,color=INK2,va="center")
    fig.text(X[6]+.007,yb+.067,"CORRECT"if L["correct"]else"INCORRECT",fontsize=9.6,weight="bold",color=cols[6],va="center")
    fit_text(fig,X[6]+.007,yb+.006,W_-.014,.045,"expected answer = audit metadata, not an input",fs_max=7,fs_min=6.2,color=MUTED)
    for i in range(6):arrow(fig,X[i]+W_+.001,yb+hb*.55,X[i+1]-.001,yb+hb*.55,color=INK2,ms=10)
    lay=["Plain sentences. Each is read by the model when its cartridge is written.","The model's own internal numbers for that sentence.","A shelf of numerical memory cards, outside the model.",
         "The question picks the five best-matching cards.","The five cards are merged into one working memory.","The model reads the memory; it never sees the text again.","Generated by the frozen model."]
    for i in range(7):fit_text(fig,X[i]+.002,.405,W_-.004,.125,lay[i],fs_max=8.6,fs_min=6.6,color=INK)
    frame(fig,.03,.148,.455,.225,ec=RULE,fc=PANEL,lw=.7)
    fit_text(fig,.04,.155,.44,.21,f"WHO DOES WHAT\n• stores: the cartridge bank (FRAM keys of {L['n_cartridges']} facts) and the CUT3 memory (K/V numbers) — data outside the weights\n"
             f"• selects: FRAM — the question's layer-{li} rows are matched against every cartridge key (Fig. 3)\n• answers: frozen {MODEL_FAMILY}; its {AR[0]} layers read the installed CUT3 memory (Figs. 2, 4)",fs_max=9.2,fs_min=6.6,color=INK)
    frame(fig,.5,.148,.47,.225,ec=AMBER,fc=AMBER_L,lw=.8)
    fit_text(fig,.51,.155,.455,.21,"MEASUREMENT LIMITS\n• In this implementation the five selected cartridges are written from their facts when the session memory is built (one forward of the instruction prefix + each fact, its own text only); the bank used for selection holds residual keys.\n"
             "• What is saved and reloaded is the five-cartridge session memory, not a store of all 96 CUT3 cartridges.\n• "+NOT_VALIDATED,fs_max=8.8,fs_min=6.4,color=INK)
    footer(fig,ctx,k,f"Figure {k}. Complete path of one question through the system. Plain reading: sentences become numbers, the numbers are kept outside the model, the question selects the matching numbers, they are merged into one memory, and the unchanged model answers from that memory. "
           f"Technical reading: FRAM keys are the layer-{li} residual rows of each fact (centred by the store mean, unit norm); the panel question selects five cartridges with the locked reference ranking of TEST610; build_cut3 writes H{cfg['CUT']} + layers 0–{cfg['CUT']} K/V per cartridge and consolidates {upper_txt(cfg['CUT'],AR[0])} append-only; answer() receives only the question template over the {M['T']}-token memory.")
    return finish(fig,path,ctx,[f"{M['T']} tokens",f"{L['query_tokens']} template",f"exact   {R['fram']['rank']} / {L['n_cartridges']}",mt(short(L["expected"],14)),f"file {K['bytes']:,} B",f"hidden {AR[1]}"],"System architecture")
# ------------------------------------------------------------------ FIG 2 · CUT3 memory writing and consolidation
def f2(ctx,k,path):
    P=ctx["P"];L=L_(ctx);st=L["steps"];cars=L["cartridges"];M=L["memory"];T=M["T"];cfg=P["engine"]["config"];fig=new_fig();cut=cfg["CUT"];NLm=M["layers"]
    header(fig,k,"Native Memory Construction and Append-Only Consolidation",f"engine build_cut3(ids, {cut}) = init(first, {cut}) → append(memory, next, {cut}) × 4 · every model forward recorded · prior K/V rows compared bitwise after each append","LIVE MEASUREMENT",BLUE)
    ax=fig.add_axes([.055,.565,.3,.27]);ax.set_zorder(3);clean(ax)
    for n_,s in enumerate(st):
        old=s["old_len"];add=s["new_len"]-old;col=SEGC[n_+1]
        ax.bar(n_,old,color="#E5E7EB",edgecolor=INK2,lw=.5,width=.62);ax.bar(n_,add,bottom=old,color=col,width=.62)
        ax.text(n_,s["new_len"]+1.2,f"{s['new_len']}",ha="center",fontsize=8,color=INK)
        if n_:ax.text(n_,old/2,"✓",ha="center",va="center",fontsize=12,color=TEAL,weight="bold")
    ax.set_xticks(range(len(st)));ax.set_xticklabels([f"{s['kind']}\n#{c['id']:02d}"+(" T"if c["is_target"]else"")for s,c in zip(st,cars)],fontsize=7.6)
    ax.set_ylabel("memory rows (tokens)",fontsize=8);ax.set_ylim(0,T*1.18);ax.set_title("memory length after each step (✓ = earlier rows bitwise unchanged)",fontsize=8.2)
    tx=.395;fig.text(tx,.835,"Forward record (instrumentation, this run)",fontsize=9.6,weight="bold",color=INK)
    fig.text(tx,.805,mt(f"{'step':<9}{'write forward':<26}{'consolidation forward':<36}{'prior rows':<12}"),fontsize=7.6,family=MONO,color=MUTED)
    for n_,s in enumerate(st):
        f=s["forwards"];y=.778-n_*.052
        w_=f"ids {f[0]['n']} · no cache";c_=(f"zero-emb {f[1]['n']} · cache {f[1]['past_len']}"if f[1]["has_past"]else f"zero-emb {f[1]['n']} · no cache")
        pr="init"if s["kind"]=="init"else("✓ identical"if s["prior_rows_bitwise_unchanged"]else"✗ CHANGED")
        fig.text(tx,y,mt(f"{n_+1} #{s['cartridge_id']:02d}  {w_:<26}{c_:<36}{pr:<12}"),fontsize=7.7,family=MONO,color=INK,weight="bold"if cars[n_]["is_target"]else"normal")
        sub=(f"   SHA rows 0…{s['old_len']-1}: {s['prior_sha_before'][:12]}… = {s['prior_sha_after'][:12]}… · {1000*s['seconds']:.0f} ms"if s["kind"]=="append"else f"   prefix {cfg['prefix_tokens']} + body of C1 written together ({s['new_len']} rows) · {1000*s['seconds']:.0f} ms")
        fig.text(tx,y-.022,mt(sub),fontsize=6.9,family=MONO,color=INK2)
    A=np.asarray(M["k_norm"],dtype=float);mu=A.mean(axis=1,keepdims=True);sd=A.std(axis=1,keepdims=True);sd[sd==0]=1;Z=(A-mu)/sd
    hx=fig.add_axes([.055,.215,.5,.245]);hx.set_zorder(3)
    im=hx.imshow(Z,aspect="auto",cmap="RdBu_r",vmin=-2.5,vmax=2.5,interpolation="nearest",origin="lower")
    for lab,a_,b_,_ in L["segments"][1:]:hx.axvline(a_-.5,color=INK,lw=.8)
    hx.axhline(cut+.5,color=AMBER,lw=1.4,ls=(0,(4,2)));hx.set_yticks([0,cut,NLm//2,NLm-1]);hx.set_yticklabels(["L0",f"L{cut}",f"L{NLm//2}",f"L{NLm-1}"],fontsize=7)
    hx.set_xlim(-.5,T-.5);hx.set_xticks([L["segments"][i][1]for i in range(len(L["segments"]))]+[T-1]);hx.tick_params(axis="x",labelsize=6.8)
    hx.set_title(mt(f"‖K‖ per row and layer (mean over {M['kv_heads']} KV heads, z-scored per layer) · dashed: top of layers 0–{cut}"),fontsize=8)
    cb=fig.colorbar(im,cax=fig.add_axes([.56,.215,.007,.245]));cb.ax.tick_params(labelsize=6.3);cb.outline.set_linewidth(.5)
    sx=fig.add_axes([.055,.168,.5,.026]);sx.set_zorder(3);sx.set_xlim(-.5,T-.5);sx.set_ylim(0,1);sx.axis("off")
    for n_,(lab,a_,b_,ci)in enumerate(L["segments"]):
        c_=cars[n_-1]if n_ else None;tg=bool(c_ and c_["is_target"])
        sx.add_patch(Rectangle((a_-.5,0),b_-a_,1,color=SEGC[n_],alpha=1 if(tg or n_==0)else .78,lw=0))
        sx.text((a_+b_-1)/2,.5,(("prefix"if n_==0 else f"#{ci:02d}")+(" T"if tg else"")),ha="center",va="center",fontsize=7,color="white",weight="bold")
    bpt=M["bytes"]//max(1,T)
    frame(fig,.605,.168,.365,.36,ec=TEAL,fc=TEAL_L,lw=1.0)
    fit_text(fig,.615,.176,.35,.345,f"TENSOR STRUCTURE (measured)\nmemory = {NLm} layers × (K, V)\neach tensor {M['shape']} · {M['dtype']}\nbytes per token = {NLm}·2·{M['kv_heads']}·{M['head_dim']}·2 = {bpt:,} B\n"
             f"total {M['bytes']:,} B ({mib(M['bytes']):.2f} MiB) for {T} tokens\n\nHOW IT IS WRITTEN (engine code)\nlayers 0–{cut}: K/V of checkpoint(ci, {cut}); keys re-phased by RoPE to their memory positions\n"
             f"{upper_txt(cut,NLm)}: append forward over the existing memory with all-zero input embeddings; the stored H{cut} is injected at the layer-{cut} output\n\n"
             f"• token-id forwards: {L['source_reads']} (prefix + each cartridge's own fact, once)\n• zero-embedding forwards: {L['zero_embed_forwards']}\n• earlier rows unchanged after 4 appends: {all(s['prior_rows_bitwise_unchanged']for s in st[1:])}\n• memory SHA {L['memory_sha'][:20]}…",
             fs_max=8.2,fs_min=6.4,family=MONO,color=INK)
    footer(fig,ctx,k,f"Figure {k}. Live record of the CUT3 construction for case {L['case']:02d}. Step 1 (init) writes cartridge #{st[0]['cartridge_id']:02d} together with the {cfg['prefix_tokens']}-token instruction prefix; steps 2–5 append the next cartridges in engine order. "
           f"In every step the instrumentation sees one token-id forward (the instruction prefix + the new cartridge's own fact, no cache; the prefix rows are discarded on append) and one consolidation forward with all-zero input embeddings over the new rows only. After every append all earlier K/V rows of all {NLm} layers are compared bitwise with the memory before the append. "
           "The heat map shows the measured K norms of the final memory (display only).")
    return finish(fig,path,ctx,[f"{T}",f"{bpt:,} B",f"{M['bytes']:,} B",L["memory_sha"][:20],"✓ identical"if len(st)>1 else"init"],"CUT3 memory writing")
# ------------------------------------------------------------------ FIG 3 · FRAM mathematical retrieval
def f3(ctx,k,path):
    P=ctx["P"];L=L_(ctx);R=L["retrieval"];cfg=P["engine"]["config"];su=P["startup"];li,rk,be=cfg["fram_cfg"];fig=new_fig();tgt=L["target"];N=L["n_cartridges"];T6=TEST610["fram"];T6r=TEST610["reference"]
    header(fig,k,"Associative Retrieval and Mathematical Ranking",f"FRAM over {N} cartridges · locked configuration: layer {li}, rank {rk}, β = {be:g} (chosen on calibration cases 0–11 only) · case {L['case']:02d} ({L['split']}): {short(L['question'],60)}","ENGINE SCORES · THIS CASE",BLUE)
    frame(fig,.03,.49,.45,.37,ec=RULE,fc=PANEL,lw=.7)
    fig.text(.04,.84,"EQUATIONS EXECUTED BY THE ENGINE (610.py)",fontsize=8.4,weight="bold",color=INK2,va="center")
    fig.text(.04,.80,r"(1)  $\tilde h=(h-\mu)\,/\,\Vert h-\mu\Vert$",fontsize=10.5,color=INK,va="center")
    fig.text(.255,.80,mt(f"h = layer-{li} residual row; μ = mean of all stored rows"),fontsize=7.4,color=INK2,va="center")
    fig.text(.04,.735,r"(2)  $s(q,c)=\frac{1}{|q|}\sum_{t\in q}\frac{1}{\beta}\left[\log\sum_{j\in c}e^{\beta\,\tilde q_t\cdot\tilde k_j}-\log|c|\right]$",fontsize=10.5,color=INK,va="center")
    fig.text(.04,.685,mt(f"exact FRAM kernel (pair_scores), β = {be:g}; q = question rows in the fact frame, c = fact-span rows"),fontsize=7.4,color=INK2,va="center")
    fig.text(.04,.637,r"(3)  $\phi(x)=e^{x\Omega-\beta/2}/\sqrt{M},\;F_c=\mathrm{mean}_{j\in c}\,\phi(\tilde k_j),\;U=F^{\top}C$",fontsize=10.2,color=INK,va="center")
    fig.text(.04,.588,r"       $z_t=\phi(\tilde q_t)\,U\,C^{\top},\;\hat s_c=\frac{1}{|q|}\sum_t\log\max(z_{t,c},10^{-12})/\beta$",fontsize=10.2,color=INK,va="center")
    fit_text(fig,.04,.497,.43,.07,f"reference ranking used to select the panel's CUT3 cartridges: Ω ~ N(0, βI) of size {R['shapes']['dim']}×{cfg['M']}, C = random ±1/√D codes, {N}×{cfg['D']}. "
             f"At {N} cartridges U is larger than the features it encodes: a ranking procedure, not a memory saving.",fs_max=7.4,fs_min=6.3,color=INK2)
    frame(fig,.03,.155,.45,.315,ec=BLUE,fc="white",lw=.9)
    fig.text(.04,.452,"RETRIEVAL QUALITY · start-up in this runtime (sealed TEST610 in parentheses)",fontsize=8.4,weight="bold",color=BLUE,va="center")
    rows=[(f"Recall@1 · holdout cases 12–23",f"{su['hold_r1']}/12",f"({T6['hold_r1']}/12)"),(f"Recall@5 · holdout cases 12–23",f"{su['hold_r5']}/12",f"({T6['hold_r5']}/12)"),
          (f"Recall@5 · all 24 cases",f"{su['all_r5']}/24",f"({T6['all_r5']}/24)"),(f"Self-retrieval@1 · each cartridge as query",f"{su['self_r1']}/{su['self_n']}",f"({T6['self_r1']}/96)"),
          (f"Reference ranking · holdout Recall@1",f"{su['ref_hold_r1']}/12",f"({T6r['hold_r1']}/12)"),(f"Reference ranking · holdout Recall@5",f"{su['ref_hold_r5']}/12",f"({T6r['hold_r5']}/12)"),
          (f"Hub: most frequent cartridge in the 24 top-5 lists",f"{su['hub_max']}×",f"({T6['hub_max']}×)")]
    for i,(a_,b_,c_)in enumerate(rows):
        y=.418-i*.033;fig.text(.045,y,mt(a_),fontsize=8,color=INK,va="center");fig.text(.36,y,b_,fontsize=8.6,weight="bold",family=MONO,color=INK,va="center");fig.text(.415,y,c_,fontsize=7.6,family=MONO,color=MUTED,va="center")
    lk=su["locked"];fit_text(fig,.045,.16,.43,.04,f"calibration: {su['n_configs']} configurations (layers {cfg['fram_layers']} × rank {cfg['fram_ranks']} × β {cfg['fram_betas']}) scored on cases 0–11 only → locked {tuple(lk['cfg'])} with hit5 {lk['hit5']}/12, hit1 {lk['hit1']}/12",fs_max=7.2,fs_min=6.2,color=INK2)
    fo=R["fram"]["order"];top=fo[:10];sc=R["fram"]["scores"];ro=R["ref"]["order"]
    ax=fig.add_axes([.605,.54,.36,.255]);ax.set_zorder(3);clean(ax)
    yy=np.arange(len(top))[::-1];vals=[sc[i]for i in top]
    ax.barh(yy,vals,color=[BLUE if i==tgt else"#CBD5E1"for i in top],height=.68)
    lo=min(vals);hi=max(vals);span=(hi-lo)or 1.0;ax.set_xlim(lo-.35*span,hi+.30*span)
    for y_,i in zip(yy,top):ax.text(sc[i]+.02*span,y_,f"{sc[i]:.4f}",va="center",fontsize=7,family=MONO,color=INK)
    ax.set_yticks(yy);ax.set_yticklabels([f"#{i:02d} {fo.index(i)+1:>2} · ref {ro.index(i)+1:>2}"+(" ◀T"if i==tgt else"")for i in top],fontsize=7.2,family=MONO)
    ax.tick_params(axis="x",labelsize=7);ax.set_title(f"top-10 of {N} by exact FRAM score s(q, c) · label: id · exact rank · reference rank",fontsize=8)
    fig.text(.5,.857,mt(f"target #{tgt:02d} · exact rank {R['fram']['rank']} · reference rank {R['ref']['rank']} · score {R['fram']['target_score']:.4f} · margin to best other {R['fram']['margin']:+.4f}"),fontsize=8.4,weight="bold",color=BLUE,va="center")
    fig.text(.5,.832,mt(f"q: {R['shapes']['query_rows']} question rows × {R['shapes']['dim']} · c: {R['shapes']['key_rows'][tgt]} fact rows × {R['shapes']['dim']} (target)"),fontsize=7.6,family=MONO,color=INK2,va="center")
    frame(fig,.5,.155,.47,.315,ec=RULE,fc="white",lw=.8)
    fig.text(.51,.452,"TOP-5 · exact FRAM  vs  locked reference ranking (→ CUT3 memory)",fontsize=8.4,weight="bold",color=INK2,va="center")
    rs=R["ref"]["scores"]
    for j in range(5):
        a_=fo[j];b_=ro[j];y=.418-j*.034;ca=P["live"]["retrieval"]
        fig.text(.515,y,mt(f"{j+1}. #{a_:02d} {sc[a_]:.4f}"+(" T"if a_==tgt else"")),fontsize=8,family=MONO,color=BLUE if a_==tgt else INK,weight="bold"if a_==tgt else"normal",va="center")
        fig.text(.735,y,mt(f"{j+1}. #{b_:02d} {rs[b_]:.4f}"+(" T"if b_==tgt else"")),fontsize=8,family=MONO,color=BLUE if b_==tgt else INK,weight="bold"if b_==tgt else"normal",va="center")
    fit_text(fig,.515,.165,.445,.1,f"target in the selected five: {yn(R['target_present'])} · engine order of the CUT3 memory (sorted ids, as in TEST610): {L['engine_order']} · "
             "free-text questions are ranked with the exact kernel (2) instead",fs_max=7.6,fs_min=6.3,color=INK2)
    footer(fig,ctx,k,f"Figure {k}. How FRAM ranks the {N} cartridges for case {L['case']:02d}. Plain reading: the question and every stored fact are turned into the model's own layer-{li} numbers; the facts whose numbers best match the question rise to the top. "
           f"Technical reading: equation (2) is the engine's pair_scores (a per-question-token soft maximum over the fact rows, averaged over question tokens); equation (3) is the locked reference ranking that TEST610 uses to choose the five panel cartridges. "
           f"All scores shown are the engine's own values for this case, recomputed in this run from the question and cartridge features captured at start-up (one forward per fact and per panel question); a free-text question is captured with a new forward. Recall@1, Recall@5 and self-retrieval are start-up measurements in this runtime; sealed TEST610 values in parentheses.")
    return finish(fig,path,ctx,[f"{su['hold_r1']}/12",f"{su['all_r5']}/24",f"{su['self_r1']}/{su['self_n']}",f"exact rank {R['fram']['rank']}",f"reference rank {R['ref']['rank']}",f"{R['fram']['target_score']:.4f}"],"FRAM retrieval")
# ------------------------------------------------------------------ FIG 4 · disk persistence and source-free answering
def disk_icon(fig,x,y,w,h,col):
    frame(fig,x,y,w,h,ec=col,fc="white",lw=1.1,z=2)
    ax=fig.add_axes([x+w*.12,y+h*.14,w*.5,h*.72]);ax.set_zorder(4);ax.axis("off");ax.set_xlim(-1,1);ax.set_ylim(-1,1);ax.set_aspect("equal")
    ax.add_patch(Circle((0,0),.95,fc="#E5E7EB",ec=col,lw=1.0));ax.add_patch(Circle((0,0),.22,fc="white",ec=col,lw=.8));ax.plot([.25,.85],[.1,.45],color=col,lw=1.6)
def f4(ctx,k,path):
    P=ctx["P"];L=L_(ctx);K=L["disk"];M=L["memory"];T=M["T"];fig=new_fig();ok=L["correct"]
    header(fig,k,"Save → Reload → Answer Without Source Text",f"The {T}-token CUT3 memory of this run ({len(L['engine_order'])} cartridges) is written to a file, read back, compared bit for bit and used again · file operations are real; the disk picture is an analogy","LIVE MEASUREMENT",BLUE)
    W_=.172;G_=.02;X=[.03+i*(W_+G_)for i in range(5)];yb=.615;hb=.235
    items=[("1 · BUILD","CUT3 memory in GPU",f"{T} tokens × {M['layers']} layers\n{M['bytes']:,} B in memory\nSHA {L['memory_sha'][:16]}…\nids {L['engine_order']}"),
           ("2 · SAVE","save_session()",f"torch.save(payload)\nformat {K['format']}\nfile {K['bytes']:,} B\n(K/V + {K['bytes']-K['memory_bytes']:,} B header)"),
           ("3 · FILE ON DISK","session file",f"{short(K['file'],30)}\nSHA-256 {K['sha256'][:16]}…\n  …{K['sha256'][-16:]}\nfact text inside: {yn(K['fact_text_in_file'])}"),
           ("4 · RELOAD","load_session()",f"torch.load(\n  weights_only=True)\nids equal: {yn(K['ids_equal'])}\nK/V bitwise equal in\nall {M['layers']} layers: {yn(K['bit_equal'])}\nmemory SHA equal: {yn(K['memory_sha_equal'])}"),
           ("5 · ANSWER","answer(reloaded, q)",None)]
    for i,(s,t,b)in enumerate(items):
        col=[TEAL,TEAL,AMBER,TEAL,BLUE if ok else ORANGE][i]
        if b is not None:block(fig,X[i],yb,W_,hb,s,t,b,ec=col,bfs=9.4,tfs=9.6)
        if i<4:arrow(fig,X[i]+W_+.002,yb+hb*.5,X[i+1]-.002,yb+hb*.5,color=INK2)
    frame(fig,X[4],yb,W_,hb,ec=BLUE if ok else ORANGE,fc=BLUE_L if ok else ORANGE_L,lw=1.3);frame(fig,X[4],yb+hb-.032,W_,.032,ec=BLUE if ok else ORANGE,fc=PANEL,lw=.9)
    fig.text(X[4]+.007,yb+hb-.016,"5 · ANSWER",fontsize=7,weight="bold",color=MUTED,va="center");fig.text(X[4]+.007,yb+hb-.052,"answer(reloaded, q)",fontsize=9.4,weight="bold",color=INK,va="center")
    fit_text(fig,X[4]+.008,yb+.09,W_-.016,.075,K["answer"],fs_max=17,fs_min=8,color=INK,weight="bold")
    fig.text(X[4]+.008,yb+.072,mt(f"expected {short(L['expected'],16)} · {'CORRECT'if ok else'INCORRECT'}"),fontsize=7.8,weight="bold",color=BLUE if ok else ORANGE,va="center")
    fig.text(X[4]+.008,yb+.035,mt(f"identical to the answer\nbefore saving: {yn(K['answer_identical'])}"),fontsize=8.2,family=MONO,color=INK2,va="center")
    frame(fig,.03,.405,.94,.18,ec=RULE,fc=PANEL,lw=.7)
    fig.text(.04,.562,mt(f"MODEL INPUT AFTER RELOAD · {K['query_tokens']} token ids (question template) · over {T} memory rows · source text given to the model: none"),fontsize=8.4,weight="bold",color=INK2,va="center")
    toks=[t.replace("\n","⏎")for t in L["query_token_text"]];x=.04;y=.515;fig.canvas.draw();rdr=fig.canvas.get_renderer();FW=fig.get_figwidth()*fig.dpi
    for t in toks:
        s_=mt(t if t.strip()else"·");tt=fig.text(x,y,s_,fontsize=9.2,family=MONO,color=INK,va="center",bbox=dict(boxstyle="square,pad=0.25",fc="white",ec=RULE,lw=.5))
        w=tt.get_window_extent(renderer=rdr).width/FW
        if x+w>.96 and x>.041:tt.set_position((.04,y-.045));x=.04;y-=.045
        x+=w+.005
    fr=K["forwards"];fig.text(.04,.425,mt(f"first forward after reload: {fr[0]['n']} token ids · cache {fr[0]['past_len']} · then {len(fr)-1} one-token generation forwards"),fontsize=7.8,family=MONO,color=INK2,va="center")
    frame(fig,.03,.15,.62,.235,ec=TEAL,fc=TEAL_L,lw=.9)
    fit_text(fig,.04,.158,.6,.22,f"CHECKS IN THIS RUN (fail-closed)\n• file size and SHA-256 recomputed from disk = save_session() report\n• the file contains none of the five fact texts and not the question text (byte search)\n"
             f"• reloaded ids = saved ids · K/V tensors bitwise equal in all {M['layers']} layers · memory SHA-256 equal\n• answer after reload = answer before saving ({short(K['answer'],20)})\n"
             f"• memory SHA-256 before saving = after reload: {L['memory_sha'][:20]}…\n• the session file is part of the ZIP evidence package\n• save {1000*K['save_seconds']:.0f} ms · load {1000*K['load_seconds']:.0f} ms · answer {1000*K['answer_seconds']:.0f} ms",fs_max=9.6,fs_min=6.5,color=INK)
    disk_icon(fig,.665,.15,.09,.235,AMBER)
    fit_text(fig,.765,.152,.205,.23,f"ANALOGY: a removable disk that carries the memory.\nREAL OPERATION: torch.save / torch.load of {M['layers']}×(K, V) tensors.\nThe {K['bytes']/1e6:.2f} MB file is the {len(L['engine_order'])}-cartridge session memory, not a store of all {L['n_cartridges']} cartridges.",fs_max=8.8,fs_min=6.3,color=INK2)
    footer(fig,ctx,k,f"Figure {k}. Persistence of the numerical memory. Plain reading: the memory built for this question is written to a file, the file is read back, and the model gives the same answer from it without seeing the text. "
           f"Technical reading: save_session() stores the {M['layers']} K/V pairs of the CUT3 memory with the cartridge ids and source hashes; load_session() restores them (weights_only=True); compare_mem_exact() confirms bit equality; answer() receives only the question-template tokens over the reloaded memory. "
           "File name, size and SHA-256 are those of this run's file, which is included in the ZIP package.")
    return finish(fig,path,ctx,[f"{K['bytes']:,} B",K["sha256"][:16],f"all {M['layers']} layers: {yn(K['bit_equal'])}",f"{K['query_tokens']} token ids",mt(short(L['expected'],16))],"Save, reload, answer")
# ------------------------------------------------------------------ FIG 5 · causal controls and failure analysis
def f5(ctx,k,path):
    P=ctx["P"];L=L_(ctx);N=L["negative"];fig=new_fig();A=TEST575["arms"];cf=TEST575["counterfactual"];F6=TEST610["final"]
    header(fig,k,"Memory Dependence, Counterfactual Controls and Failure Analysis","Does the model answer because of the external memory? Left: live control of this run. Middle: sealed TEST610 acceptance. Right: sealed TEST575 controls of a different experiment.","LIVE + SEALED REFERENCES",INK2)
    fig.text(.03,.865,"(a) LIVE · THIS RUN",fontsize=8.6,weight="bold",color=BLUE,va="center")
    ok=L["correct"];frame(fig,.03,.62,.29,.225,ec=BLUE if ok else ORANGE,fc=BLUE_L if ok else ORANGE_L,lw=1.2)
    pres=L["retrieval"]["target_present"]
    fig.text(.04,.825,mt(f"{'TARGET PRESENT'if pres else'TARGET NOT SELECTED'} · {len(L['engine_order'])} cartridges · {L['memory']['T']} tokens"),fontsize=8,weight="bold",color=INK2,va="center")
    fit_text(fig,.04,.73,.27,.075,L["answer"],fs_max=16,fs_min=8,color=INK,weight="bold")
    fig.text(.04,.70,mt(f"expected {short(L['expected'],16)} · {'CORRECT'if ok else'INCORRECT'}"),fontsize=8.2,weight="bold",color=BLUE if ok else ORANGE,va="center")
    fig.text(.04,.668,mt(f"ids {L['engine_order']}"),fontsize=7.4,family=MONO,color=INK2,va="center")
    fig.text(.04,.640,mt(f"sealed TEST610 answer {short(L['sealed_answer'],14)!r} · same: {yn(L['matches_sealed'])}"),fontsize=7.4,family=MONO,color=INK2,va="center")
    sk=bool(N.get("skipped"));lk=N.get("gold_leak");frame(fig,.03,.385,.29,.215,ec=MUTED if sk else(RED if lk else TEAL),fc="white",lw=1.1)
    if sk:fit_text(fig,.04,.395,.27,.19,"TARGET REMOVED · not applicable: "+N["skipped"],fs_max=8,fs_min=6.5,color=INK2)
    else:
        fig.text(.04,.58,mt(f"TARGET REMOVED · {len(N['ids'])} cartridges · {N['T']} tokens"),fontsize=8,weight="bold",color=INK2,va="center")
        fit_text(fig,.04,.49,.27,.07,N["answer"],fs_max=15,fs_min=8,color=INK,weight="bold")
        fig.text(.04,.46,mt(f"expected answer produced: {yn(lk)}"),fontsize=8.2,weight="bold",color=RED if lk else TEAL,va="center")
        fig.text(.04,.43,mt(f"ids {N['ids']}"),fontsize=7.4,family=MONO,color=INK2,va="center")
        fig.text(.04,.402,"same question · same model · target cartridge absent",fontsize=7.2,color=MUTED,va="center")
    fit_text(fig,.03,.16,.29,.2,f"Reading: {'target cartridge present'if pres else'target cartridge not among the five selected'} → expected answer {'produced'if ok else'NOT produced'} in this run; target cartridge removed → expected answer "+("not testable here"if sk else("produced (leak)"if lk else"not produced"))+
             ". One live control is one observation, not a statistic."+(f"\n\nFree-text question (unscored): {L['custom']['question']} → {L['custom']['answer']} · exact FRAM top-5 {L['custom']['ids']}"if L.get("custom")else""),fs_max=9,fs_min=6.5,color=INK)
    fig.text(.35,.865,"(b) TEST610 · SEALED ACCEPTANCE · 24 panel cases",fontsize=8.6,weight="bold",color=INK2,va="center")
    cs=TEST610["cases"];cw=.0485;chh=.07
    for r in cs:
        i=r["case"];cx=.35+(i%6)*(cw+.004);cy=.775-(i//6)*(chh+.006);good=r["correct"]
        frame(fig,cx,cy,cw,chh,ec=TEAL if good else ORANGE,fc=TEAL_L if good else ORANGE_L,lw=1.2 if not good else .7,z=1)
        fig.text(cx+.004,cy+chh-.015,f"{i:02d} {r['split']}",fontsize=6.6,weight="bold",color=INK2,va="center")
        fig.text(cx+.004,cy+.022,mt(short(r["answer"],9)),fontsize=6.8,family=MONO,color=INK,va="center")
        fig.text(cx+cw-.004,cy+chh-.015,"✓"if good else"✗",fontsize=8,weight="bold",color=TEAL if good else ORANGE,va="center",ha="right")
    fig.text(.35,.525,mt(f"panel {F6['all_panel_correct']}/{F6['all_panel_total']} · holdout {F6['holdout_correct']}/{F6['holdout_total']} · target retrieved {sum(r['present']for r in cs)}/24"),fontsize=8.4,weight="bold",color=INK,va="center")
    bad=[r for r in cs if not r["correct"]]
    fail="; ".join(f"case {r['case']:02d} ({r['split']}): target #{r['target']:02d} was in the memory, answer {r['answer']!r}, expected {P['archived_gold'].get(str(r['case']),'?')!r}"for r in bad)or"none"
    fit_text(fig,.35,.425,.31,.08,"FAILURE (kept visible): "+fail,fs_max=7.8,fs_min=6.3,color=ORANGE)
    fig.text(.35,.395,"negative controls (target removed):",fontsize=7.8,weight="bold",color=INK2,va="center")
    for j,r in enumerate(TEST610["negatives"]):
        fig.text(.35,.365-j*.03,mt(f"case {r['case']:02d} → {short(r['raw'],10)!r} · expected answer produced: {yn(r['gold_leak'])}"),fontsize=7.4,family=MONO,color=INK,va="center")
    fit_text(fig,.35,.16,.31,.07,"Four negative controls show no leakage of the expected answer; four cases do not exclude every alternative causal explanation.",fs_max=7.6,fs_min=6.3,color=MUTED)
    fig.text(.69,.865,"(c) TEST575 · SEALED · DIFFERENT EXPERIMENT",fontsize=8.6,weight="bold",color=AMBER,va="center")
    ax=fig.add_axes([.72,.57,.25,.26]);ax.set_zorder(3);clean(ax);arms=list(A)
    ax.bar(range(len(arms)),[A[a_]for a_ in arms],color=[TEAL,BLUE,BLUE,GRAYPT,AMBER],width=.62)
    for i_,a_ in enumerate(arms):ax.text(i_,A[a_]+1,f"{A[a_]}/72",ha="center",fontsize=7.4,color=INK)
    ax.set_xticks(range(len(arms)));ax.set_xticklabels([a_.replace("_","\n",1)for a_ in arms],fontsize=6.8);ax.set_ylim(0,84);ax.set_ylabel("correct of 72",fontsize=7.6);ax.axhline(72,color=MUTED,lw=.6,ls=(0,(3,3)))
    ax.set_title("24 cases × 3 target positions",fontsize=7.8)
    fit_text(fig,.69,.36,.28,.13,f"counterfactual (same H3, same layers 0–3 K/V, cross-cartridge attention blocked in {upper_txt()}):\nupper-layer K/V changed {cf['upper_kv_changed']}/72 · answer changed {cf['answer_changed']}/72",fs_max=8.4,fs_min=6.3,color=INK)
    fit_text(fig,.69,.16,.28,.18," · ".join(f"{a_} = {ARM_DESC[a_]}"for a_ in arms),fs_max=7.8,fs_min=6.0,color=INK2)
    footer(fig,ctx,k,f"Figure {k}. Causal evidence, kept in three separate sources. (a) This run: the same question over the memory with and without the target cartridge. (b) Sealed TEST610 acceptance on the locked panel: {F6['all_panel_correct']}/{F6['all_panel_total']} answers correct, holdout {F6['holdout_correct']}/12; the single incorrect case is shown, not hidden; four target-removed controls. "
           "(c) Sealed TEST575, a different experiment on the same model and CUT3 mechanism: joint text, incremental and batch consolidation 72/72, native independent K/V 26/72, and the attention-blocking counterfactual. Values of (b) and (c) are reproduced verbatim from their logs, not recomputed.")
    return finish(fig,path,ctx,[f"panel {F6['all_panel_correct']}/{F6['all_panel_total']}",f"answer changed {cf['answer_changed']}/72","72/72",mt(short(L['expected'],16))],"Causal controls")
# ------------------------------------------------------------------ FIG 6 · validation, integrity and reproducibility
def f6(ctx,k,path):
    P=ctx["P"];L=L_(ctx);cfg=P["engine"]["config"];fz=P["frozen"];su=P["startup"];F6=TEST610["final"];fig=new_fig();E_=P["environment"]
    header(fig,k,"Scientific Validation and Reproducible Evidence","Validation chain of this run · the sealed TEST610 acceptance, the start-up reproduction in this runtime and this run are labelled separately","VALIDATION · INTEGRITY",INK2)
    W_=.103;G_=.0154;X=[.03+i*(W_+G_)for i in range(8)];yb=.675;hb=.185
    chain=[("SOURCE","Verified source",f"610.py blob\n{cfg['engine_blob'][:12]}…\n566.py {cfg['source_blobs']['566.py'][:8]}…\n575.py {cfg['source_blobs']['575.py'][:8]}…"),
           ("LOCK","Locked config",f"FRAM {tuple(cfg['fram_cfg'])}\nM = D = {cfg['M']}\npanel SHA\n{cfg['panel_sha'][:12]}…"),
           ("RUN","Live run",f"{short(P['run_id'],22)}\n{P['run_start_utc'][11:19]} UTC\n{short(E_['gpu'],22)}"),
           ("RAW","Raw measurements",f"{len(L['forwards_build'])} build forwards\n{len(L['steps'])} steps\n{L['n_cartridges']} scores ×2\n1 file · 1 control"),
           ("CHECK","Validation",f"{ctx['checks']} pre-seal\nchecks PASS\n(fail-closed)"),
           ("SEAL","Payload",f"SHA-256\n{ctx['sha'][:12]}…\n{ctx['sha'][12:24]}…"),
           ("MANIFEST","SHA-256 manifest",f"payload hash\nverdict\nsession file\n{L['disk']['sha256'][:12]}…"),
           ("OUTPUT","Figures",f"{ctx['N']} × JPEG 300 dpi\n{ctx['N']} × vector PDF\nvalues audited\nagainst payload")]
    for i,(s,t,b)in enumerate(chain):
        block(fig,X[i],yb,W_,hb,s,t,b,ec=[INK2,INK2,BLUE,BLUE,TEAL,TEAL,TEAL,INK2][i],bfs=8.2,tfs=7.9,bmin=6.2)
        if i<7:arrow(fig,X[i]+W_+.001,yb+hb*.45,X[i+1]-.001,yb+hb*.45,color=INK2,ms=9)
    cols=[(.03,"SEALED · TEST610 ACCEPTANCE (610.log)",AMBER,
           f"decision        {TEST610['decision']}\ncriteria        {F6['criteria_passed']}/{F6['criteria_total']} PASS\nholdout         {F6['holdout_correct']}/{F6['holdout_total']} correct\nwhole panel     {F6['all_panel_correct']}/{F6['all_panel_total']} (case 08 incorrect)\n"
           f"self-retrieval  {TEST610['fram']['self_r1']}/96\nFRAM holdout    R@1 {TEST610['fram']['hold_r1']}/12 · R@5 {TEST610['fram']['hold_r5']}/12\nweights         unchanged\ntrainable       0\nhooks           0\nseconds         {F6['seconds']}\n"
           f"result SHA-256\n  {TEST610['result_sha'][:32]}\n  {TEST610['result_sha'][32:]}"),
          (.355,"START-UP · THIS RUNTIME (610.py procedure)",BLUE,
           f"FRAM holdout R@1   {su['hold_r1']}/12\nFRAM holdout R@5   {su['hold_r5']}/12\nall cases R@5      {su['all_r5']}/24\nself-retrieval     {su['self_r1']}/{su['self_n']}\nreference R@1/R@5  {su['ref_hold_r1']}/12 · {su['ref_hold_r5']}/12\n"
           f"baseline CUT3      {su['baseline_correct']}/{su['baseline_n']} correct\nengine lock        {len(ENGINE_LOCK)} checks PASS\nfull-weight SHA-256 = reference\n  {fz['weight_sha_startup'][:32]}\n  {fz['weight_sha_startup'][32:]}"),
          (.68,"THIS RUN · INTEGRITY",TEAL,
           f"all-parameter guard  start-up = before = after\n  {fz['guard_after'][:30]}…\nweight sentinel      unchanged\n  {fz['sentinel_after'][:30]}…\ntrainable tensors    {fz['trainable_tensors']}\nLoRA / optimizer     {fz['lora']} / {fz['optimizer']}\n"
           f"foreign hooks        {fz['foreign_hooks_before']} → {fz['foreign_hooks_after']}\nlayer/attn/q hooks   {len(fz['residual_hooks_before'])} → {len(fz['residual_hooks_after'])}\nanswer {short(L['answer'],18)} · {'correct'if L['correct']else'incorrect'}\nverdict\n"+uwrap(P['verdict'],34,"  "))]
    for x,ttl,col,body in cols:
        frame(fig,x,.25,.29,.405,ec=col,fc="white",lw=.9);fig.text(x+.01,.632,ttl,fontsize=8.6,weight="bold",color=col,va="center")
        fit_text(fig,x+.01,.258,.272,.355,body,fs_max=9.2,fs_min=6.2,family=MONO,color=INK)
    frame(fig,.03,.145,.94,.09,ec=BLUE,fc=BLUE_L,lw=1.0)
    fit_text(fig,.04,.15,.92,.08,f"{DISCOVERY} · {ORG} · {DATE_TXT} · {AUTHOR_PLACE} · earlier public records: DOI {PRIOR_PUB['doi']} ({PRIOR_PUB['date']}) · DOI {QWEN_PUB['doi']} ({QWEN_PUB['date']}) · {LICENSE_NAME} · {LICENSE_URL}\n"
             "SHA-256 values are integrity seals: they show that files are unchanged since sealing. They are not scientific or mathematical proof and not third-party verification.",fs_max=8.2,fs_min=6.4,color=INK)
    footer(fig,ctx,k,f"Figure {k}. Reproducibility record of run {P['run_id']} (verdict {P['verdict']}). Payload SHA-256 {ctx['sha']}. The chain runs from the verified engine source through the locked configuration, the live measurements and the fail-closed checks to the sealed payload, "
           "the manifest and the figures; every live value printed in the figures is checked against the payload before the figure is saved. Sealed TEST610 values are reproduced from 610.log; start-up values were measured in this runtime by the same 610.py procedure.")
    return finish(fig,path,ctx,[TEST610["decision"],f"{F6['criteria_passed']}/{F6['criteria_total']} PASS",f"{F6['all_panel_correct']}/{F6['all_panel_total']} (case 08 incorrect)",fz["weight_sha_startup"][:32],ctx["sha"][:12],PRIOR_PUB["doi"],QWEN_PUB["doi"]],"Validation and reproducibility")
FIGURES=[("fig_01_system_architecture","Fig. 1 · External associative memory for a frozen language model",f1),("fig_02_cut3_construction","Fig. 2 · Native memory construction and append-only consolidation",f2),
         ("fig_03_fram_retrieval","Fig. 3 · Associative retrieval and mathematical ranking",f3),("fig_04_disk_persistence","Fig. 4 · Save → reload → answer without source text",f4),
         ("fig_05_causal_controls","Fig. 5 · Memory dependence, counterfactual controls and failure analysis",f5),("fig_06_validation","Fig. 6 · Scientific validation and reproducible evidence",f6)]
N_FIGS=len(FIGURES)
def render_all(ctx,run_dir):
    imgs=[]
    try:
        for i,(slug,cap,fn)in enumerate(FIGURES,1):
            p=Path(run_dir)/f"{slug}_{ctx['run_id']}.jpg";fn(ctx,i,p);imgs.append((p,cap))
    finally:plt.close("all")
    return imgs
#<<FIGURES_END>>
say("PART 2 / 3 complete — measurement pipeline and figures defined. Now run PART 3 / 3 in the next cell.")
MAM_BC_PART2_OK=True
# =====================================================================================================================================
# AKBASCORE MAM-BÇ · EXTERNAL ASSOCIATIVE MEMORY FOR A FROZEN LANGUAGE MODEL — QWEN2.5-7B · PART 3 / 3 · SELF-TEST AND GRADIO DEMONSTRATION
# Same demo as PARTS 1 and 2. Run this cell after PART 2 / 3, in the same Colab runtime. It runs a GPU-free self-test of the full
# pipeline (including fail-closed cases) and then prints a public gradio.live link. Choose a panel case, press RUN.
# Discovered and developed by Mustafa Akbaş · AkbasCore AI Teknoloji · 10 October 2026 · AKBASCORE RESEARCH SOFTWARE LICENSE
# =====================================================================================================================================
if not globals().get("MAM_BC_PART2_OK"):raise RuntimeError("PART 2 / 3 has not been run in this runtime. Run PART 1, then PART 2, then PART 3.")
#<<UI_BEGIN>>
_SYN_A=["Aq","Bryn","Cav","Dex","Eln","Frax","Gyr","Hex","Iln","Jor","Kex","Lyr"];_SYN_B=["adar","bren","cyr","dax","elor","fyn","grel","hyn"]
class _SynMem:
    """Synthetic memory handle for the self-test (no tensors)."""
    def __init__(s,keys,T,salt=""):s.keys=list(keys);s.T=T;s.salt=salt
    def __eq__(s,o):return isinstance(o,_SynMem)and s.keys==o.keys and s.T==o.T and s.salt==o.salt
class SelfTestInstrument:
    """Synthetic stand-in used ONLY by the service self-test (no GPU, no model). Never used for a real run."""
    def __init__(s,mode="ok"):
        s.kind="SELF-TEST";s.mode=mode;s.calls=0;names=[a+b for a in _SYN_A for b in _SYN_B];s.names=names;s.P=PREFIX_EXPECTED;s.cars=[]
        for w in range(24):
            subj,cur,near,form,role=names[w*4],names[w*4+1],names[w*4+2],names[w*4+3],names[(w*4+5)%len(names)]
            for typ,fact,gold in(("CURRENT",f"The current capital of {subj} is {cur}.",cur),("NEAR",f"The largest city of {subj} is {near}.",near),
                                 ("FORMER",f"The former capital of {subj} was {form}.",form),("ROLE",f"The current capital of {role} is {subj}.",subj)):
                ci=len(s.cars);s.cars.append(dict(id=ci,world=w,type=typ,fact=fact,gold=gold,body_len=10+ci%3))
        s.qs=[]
        for w in range(24):
            c=s.cars[s.target(w)];subj=names[w*4]
            s.qs.append({"CURRENT":f"What is the current capital of {subj}?","FORMER":f"What was the former capital of {subj}?","NEAR":f"What is the largest city of {subj}?","ROLE":f"What is the current capital of {names[(w*4+5)%len(names)]}?"}[c["type"]])
        s.info=dict(model_id=MODEL_ID,arch=list(ARCH),dtype="bfloat16",attn="sdpa",gpu="SELF-TEST INSTRUMENT (no GPU)",gpu_total_gib=0.0,torch="n/a",transformers="n/a",gradio="n/a",python="n/a",
            platform="n/a",params=1,engine_file=ENGINE_FILE,engine_sha256=ENGINE_SHA256_EXPECTED,engine_blob=ENGINE_BLOB_EXPECTED,engine_exec_sha256="e"*64,engine_tail_sha256="f"*64,
            helper_sha256=HELPER_SHA256_EXPECTED,raw_demo_file_sha256=RAW_DEMO_FILE_SHA256,init_seconds=0.0,startup_utc="n/a",source_blobs=dict(SOURCE_BLOBS_EXPECTED),
            weight_sha0=("d"*64 if mode=="weight_ref" else WEIGHT_SHA_EXPECTED),sentinel0="a"*64,sentinel_engine_load="a"*64,sentinel_610_start="a"*64,guard0="b"*64,hooks0=0,
            sentinel_method="synthetic",guard_method="synthetic",full_weight_method="synthetic",forward_counter="synthetic",observer="synthetic")
    def target(s,w):return w*4+{"CURRENT":0,"NEAR":1,"FORMER":2,"ROLE":3}[["CURRENT","FORMER","NEAR","ROLE"][w%4]]
    def config(s):
        return dict(MODEL_ID=MODEL_ID,CUT=CUT_EXPECTED,ENGINE_CUTS=[1,2,3,6],MAX_NEW=16,SEED=552552,PANEL_SEED=550550,EXPECTED_PANEL_SHA=EXPECTED_PANEL_SHA,panel_sha=EXPECTED_PANEL_SHA,prefix_tokens=s.P,
                    n_cartridges=96,n_cases=24,system="synthetic",arch=list(ARCH),engine_sha256=ENGINE_SHA256_EXPECTED,engine_blob=ENGINE_BLOB_EXPECTED,engine_exec_sha256="e"*64,helper_sha256=HELPER_SHA256_EXPECTED,
                    source_blobs=dict(SOURCE_BLOBS_EXPECTED),source_blobs_measured=dict(SOURCE_BLOBS_EXPECTED),fram_cfg=[8,0,8.0]if s.mode=="config"else list(FRAM_LOCK),fram_layers=[3,8,16,24],fram_ranks=[0,64,128],
                    fram_betas=[0.0,8.0],n_configs=24,train=list(TRAIN_CASES),hold=list(HOLD_CASES),beta=8.0,M=M_REF,D=D_REF,top_k=TOP_K)
    def status(s):return{"title":"AKBASCORE MAM-BÇ","reference_test":"TEST610","fram":list(FRAM_LOCK),"panel_sha":EXPECTED_PANEL_SHA}
    def _sc(s,case,seed):
        rng=np.random.default_rng(1000*seed+case);v=rng.normal(0.30,0.04,96);t=s.target(case);v[t]=v.max()+(0.05 if seed==1 else 0.02)
        if seed==2 and case==5:v[t]=np.sort(v)[-8]-1e-4
        return[float(x)for x in v]
    def fram_scores(s,case):return s._sc(case,1)
    def ref_scores(s,case):return s._sc(case,2)
    def rankings(s,scores):return sorted(range(len(scores)),key=lambda i:(-scores[i],i))
    def fram_shapes(s,case):return dict(query=[9+case%2,3584],keys=[[c["body_len"]-2,3584]for c in s.cars])
    def startup(s):
        rows=[]
        for w in range(24):
            o=s.rankings(s.fram_scores(w));rows.append(dict(case=w,split=split_of(w),target=s.target(w),rank=o.index(s.target(w))+1,top5=o[:5]))
        base=[dict(case=w,target=s.target(w),present=1,ok=1,answer=s.cars[s.target(w)]["gold"])for w in HOLD_CASES]
        return dict(cfg=list(FRAM_LOCK),locked=dict(cfg=list(FRAM_LOCK),hit5=12,hit1=12,rank_sum=12,order=1),calibration=[],n_configs=24,hold_r1=11 if s.mode=="startup"else 12,hold_r5=12,all_r5=24,
                    self_r1=96,self_n=96,hub_max=4,ref_hold_r1=11,ref_hold_r5=12,cases=rows,baseline=base,baseline_correct=12,baseline_retrieved=12,baseline_n=12,question_tokens=[9]*24)
    def cases(s):return[{"case":w,"split":split_of(w),"type":s.cars[s.target(w)]["type"],"question":s.qs[w],"target":s.target(w)}for w in range(24)]
    def question(s,case):return s.qs[case]
    def car(s,ci):return dict(s.cars[ci])
    def source_ids(s,ci):return list(range(1,s.P+1))+[1000+ci*20+j for j in range(s.cars[ci]["body_len"])]
    def _toks(s,text):return re.findall(r"\n|[^\s\n]+",text)
    def _ids(s,text,base):
        ids=[];s.__dict__.setdefault("_tokmap",{})
        for i,t in enumerate(s._toks(text)):tid=base+(sum(map(ord,t))*31+i)%50000;s._tokmap[tid]=t;ids.append(tid)
        return ids
    def query_ids(s,q):return s._ids(f"\nQUESTION:\n{q}\n\nANSWER:\n<|im_end|>\n<|im_start|>assistant\n",2000)
    def frame_ids(s,q):return list(range(1,s.P+1))+s._ids(q+"\n\n",60000)
    def decode_each(s,ids):return[s._tokmap.get(t,"?")for t in ids]
    def panel_strings(s):return sorted(set(s.names))
    def hit(s,a,g):return hit_ans(a,g)
    def sentinel(s):return"a"*64
    def guard(s):s.calls+=1;return"c"*64 if(s.mode=="guard_changes"and s.calls>1)else"b"*64
    def frozen(s):
        s.__dict__["fz_calls"]=s.__dict__.get("fz_calls",0)+1
        return dict(training=False,trainable_tensors=0,lora=False,optimizer=False,hooks=0,foreign_hooks=0,foreign_modules={},sentinel="a"*64,
                    residual_hooks=(["L3:layer"]if(s.mode=="residual_hook"and s.fz_calls>1)else[]))
    def full_weight_sha(s):return s.info["weight_sha0"]
    def build(s,keys):
        keys=[int(k)for k in keys];steps=[];allf=[];old=0
        for n_,ci in enumerate(keys):
            src=s.source_ids(ci)
            if n_==0:
                f=[dict(kind="ids",n=len(src),ids=src,embeds_zero=None,has_past=False,past_len=None),dict(kind="embeds",n=len(src),ids=None,embeds_zero=True,has_past=False,past_len=None)];new=len(src)
                st=dict(kind="init",cartridge_id=ci,old_len=0,new_len=new,prior_rows_bitwise_unchanged=None,prior_sha_before=None,prior_sha_after=None)
            else:
                q=len(src)-s.P;new=old+q
                f=[dict(kind="ids",n=len(src),ids=src,embeds_zero=None,has_past=False,past_len=None),dict(kind="embeds",n=q,ids=None,embeds_zero=True,has_past=True,past_len=old)]
                if s.mode=="reread"and n_==2:f.insert(1,dict(kind="ids",n=len(s.source_ids(keys[0])),ids=s.source_ids(keys[0]),embeds_zero=None,has_past=False,past_len=None))
                same=not(s.mode=="prior_changed"and n_==3)
                st=dict(kind="append",cartridge_id=ci,old_len=old,new_len=new,prior_rows_bitwise_unchanged=same,prior_sha_before=hashlib.sha256(f"m{keys[:n_]}".encode()).hexdigest(),
                        prior_sha_after=hashlib.sha256(f"m{keys[:n_]}{'' if same else 'x'}".encode()).hexdigest())
            st.update(memory_sha=hashlib.sha256(f"m{keys[:n_+1]}".encode()).hexdigest(),seconds=0.04,forwards=f);steps.append(st);allf+=f;old=new
        return _SynMem(keys,old),dict(steps=steps,forwards=allf,keys=keys),0.2
    def ask(s,mem,q):
        ids=s.query_ids(q);case=next((w for w in range(24)if s.qs[w]==q),None)
        if s.mode=="source_in_query":ids=s.source_ids(mem.keys[0])+ids
        if case is None:ans="Not provided in the information."
        else:
            t=s.target(case);ans=s.cars[t]["gold"]if(t in mem.keys and case%7!=3)else s.names[(case*4+2)%len(s.names)]
            if s.mode=="reload_answer"and mem.salt=="reloaded":ans="Different"
        rec=[dict(kind="ids",n=len(ids),ids=ids,embeds_zero=None,has_past=True,past_len=mem.T)]+[dict(kind="ids",n=1,ids=[7],embeds_zero=None,has_past=True,past_len=mem.T+len(ids)+i)for i in range(2)]
        return ans,rec,0.12
    def retrieve_free(s,q):
        ids=[(sum(map(ord,q))+7*j)%96 for j in range(5)];ids=list(dict.fromkeys(ids))
        while len(ids)<5:ids.append((ids[-1]+1)%96)
        rk=[{"id":i,"score":0.3-0.01*j,"type":s.cars[i]["type"]}for j,i in enumerate(ids)]
        return ids,rk,[dict(kind="ids",n=len(s.frame_ids(q)),ids=s.frame_ids(q),embeds_zero=None,has_past=False,past_len=None)],0.05
    def save(s,mem,ids,path):
        body=json.dumps({"format":"AKBASCORE_CUT3_SESSION_V1","ids":list(ids),"T":mem.T}).encode()+bytes(range(256))*int(mem.T*224)
        if s.mode=="fact_in_file":body+=s.cars[ids[0]]["fact"].encode()
        Path(path).write_bytes(body);return{"path":str(path),"bytes":len(body),"sha256":hashlib.sha256(body).hexdigest()}
    def load(s,path):
        d=json.loads(Path(path).read_bytes().split(b"}",1)[0]+b"}");return _SynMem(d["ids"],d["T"],"reloaded"if s.mode in("reload_diff","reload_answer")else""),list(d["ids"])
    def equal(s,a,b):return a.keys==b.keys and a.T==b.T and(s.mode!="reload_diff")
    def memory_len(s,mem):return mem.T
    def memory_sha(s,mem):return hashlib.sha256(f"m{mem.keys}".encode()).hexdigest()if mem.salt!="reloaded"or s.mode!="reload_diff"else"0"*64
    def memory_bytes(s,mem):return mem.T*ARCH[0]*2*ARCH[3]*ARCH[4]*2
    def memory_profile(s,mem):
        NL,_,_,KVH,HD=ARCH;rng=np.random.default_rng(mem.T);k=(rng.normal(20,3,(NL,mem.T))+np.linspace(0,10,NL)[:,None]).tolist();v=(rng.normal(2,.4,(NL,mem.T))).tolist()
        return dict(layers=NL,T=mem.T,k_norm=k,v_norm=v,bytes=s.memory_bytes(mem),kv_heads=KVH,head_dim=HD,dtype="bfloat16",shape=[1,KVH,mem.T,HD])
    def release(s,*m):pass
    def acceptance_replay(s,progress=None):
        for i in range(44):
            if progress:progress(i+1,44,"DEMO_CASE synthetic")
        cases=[dict(case=r["case"],split=r["split"],answer=r["answer"],correct=r["correct"])for r in TEST610["cases"]]
        return dict(summary=dict(TEST610["final"]),criteria=dict(TEST610["criteria"]),decision=TEST610["decision"],cases=cases,negatives=list(TEST610["negatives"]),rebuild=[],append=[],
                    result_sha="0"*64,out_path="",stdout="synthetic engine stdout\n",seconds=0.1)
    def sync(s):pass
    def reset_peak(s):pass
    def peak_gib(s):return 0.0
def drain(gen):
    while True:
        try:next(gen)
        except StopIteration as s_:return s_.value
RUN_COUNTER=globals().get("RUN_COUNTER",0);GPU_LOCK=threading.Lock();SESSION_LEDGER=globals().get("SESSION_LEDGER",[])
LEDGER_PATH=ROOT/"session_ledger.jsonl"
SKIP=getattr(gr,"skip",None)or gr.update
_K=object()
FILE_KEYS=["zip","json","txt","man","session","ledger"]
DL_LABELS=["⬇ EVIDENCE PACKAGE (.zip)","⬇ FULL RUN LOG (.json)","⬇ READABLE RUN LOG (.txt)","⬇ RUN MANIFEST (.json)","⬇ CUT3 SESSION FILE (.pt)","⬇ SESSION LEDGER (.jsonl)"]
DL_ALL_LABEL=f"⬇ DOWNLOAD ALL {N_FIGS} FIGURES (JPEG, 300 dpi)"
RAW_KEYS=["txt","json","man","ledger"]
N_OUTPUTS=2+2+len(FILE_KEYS)*2+len(RAW_KEYS)
_GR_MAJOR=int(re.match(r"\d+",gr.__version__).group())
try:
    from gradio.routes import API_PREFIX as _GR_API_PREFIX
except Exception:
    _GR_API_PREFIX="/gradio_api"if _GR_MAJOR>=5 else""
FILE_URL_PREFIX=f"{_GR_API_PREFIX}/file="
def dl_update(path,label):
    if path is None:
        try:return gr.DownloadButton(label=label,value=None,interactive=False)
        except Exception:return gr.update(value=None,interactive=False)
    try:return gr.DownloadButton(label=label,value=str(path),interactive=True)
    except Exception:return gr.update(value=str(path),interactive=True)
def btn_update(active):
    try:return gr.Button(value=DL_ALL_LABEL,interactive=bool(active))
    except Exception:return gr.update(value=DL_ALL_LABEL,interactive=bool(active))
def jpg_urls_json(paths):return json.dumps([{"url":FILE_URL_PREFIX+quote(str(Path(p)),safe="/"),"name":Path(p).name}for p in paths],ensure_ascii=False)
def pack(status=_K,gal=_K,files=_K,jpgs=_K,raw=_K):
    out=[SKIP()if status is _K else status,SKIP()if gal is _K else gal]
    if jpgs is _K:out+=[SKIP(),SKIP()]
    elif jpgs is None:out+=[btn_update(False),""]
    else:
        for pth in jpgs:
            if not file_ready(pth):raise RuntimeError(f"figure missing or empty: {pth}")
        out+=[btn_update(True),jpg_urls_json(jpgs)]
    if files is _K:out+=[SKIP()for _ in range(len(FILE_KEYS)*2)]
    elif files is None:out+=[dl_update(None,l)for l in DL_LABELS]+[None]*len(FILE_KEYS)
    else:
        paths=[files.get(k_)for k_ in FILE_KEYS]
        for pth in paths:
            if pth is not None and not file_ready(pth):raise RuntimeError(f"download artifact missing or empty: {pth}")
        out+=[dl_update(p,l)for p,l in zip(paths,DL_LABELS)]+[None if p is None else str(p)for p in paths]
    if raw is _K:out+=[SKIP()for _ in RAW_KEYS]
    elif raw is None:out+=[""]*len(RAW_KEYS)
    else:out+=[raw.get(k_,"")for k_ in RAW_KEYS]
    return tuple(out)
def card_html(title,body,kind="info"):return f'<div class="kz card {kind}"><div class="h">{html.escape(title)}</div><div class="small">{body}</div></div>'
def pbar_html(done,total):
    pc_=100.0*done/max(1,total);return f'<div style="margin-top:6px;background:#e5e7eb;height:10px"><div style="width:{pc_:.1f}%;height:10px;background:#1d4ed8"></div></div><div class="small">{done}/{total}</div>'
def stage_card(e):return card_html(f"{e['stage']}/{N_STAGES} · {e['title']}",html.escape(e["body"])+pbar_html(e["stage"],N_STAGES),"info")
def case_choices(I):return[f"{c['case']:02d} · {c['split']} · {c['type']} · {c['question']}"for c in I.cases()]
def parse_case(v):
    m=re.match(r"\s*(\d{1,2})",str(v or""));return int(m.group(1))if m else None
def ledger_text():return"\n".join(json.dumps(x,ensure_ascii=False)for x in SESSION_LEDGER)
def case_info_html(case_v,I=None):
    I=I or INSTR;c=parse_case(case_v)
    if c is None or not 0<=c<N_CASES:return card_html("Case","Select a panel case.","info")
    t=I.target(c);q=I.question(c);s6=TEST610["cases"][c]
    body=(f'<span class="mono">question</span> {html.escape(q)}<br><span class="mono">split</span> {split_of(c)} ({"calibration case 0–11: used to lock FRAM" if c in TRAIN_CASES else "holdout case 12–23: never used for any choice"})<br>'
          f'<span class="mono">target</span> cartridge #{t:02d} · {html.escape(I.car(t)["fact"])}<br>'
          f'<span class="mono">expected</span> {html.escape(I.car(t)["gold"])} <span style="opacity:.75">(audit metadata; not an input to retrieval, writing or answering)</span><br>'
          f'<span class="mono">sealed TEST610</span> answer {html.escape(s6["answer"])} · {"correct" if s6["correct"] else "<b>incorrect</b>"}<br>'
          '<span class="mono">order</span> FRAM selects five cartridges among 96 → CUT3 memory → answer without the source text → save → reload → answer again → target-removed control.')
    return card_html(f"Case {c:02d} · {split_of(c)}",body,"info")
def engine_card_html(I):
    inf=I.info;cfg=I.config();su=I.startup();F6=TEST610["final"]
    rows=[("model",f"{inf['model_id']} · frozen · {inf['dtype']} · {arch_txt(tuple(cfg['arch']))}"),("GPU",inf["gpu"]),
          ("cartridge bank",f"{cfg['n_cartridges']} facts · FRAM key = layer-{cfg['fram_cfg'][0]} residual rows · locked ({cfg['fram_cfg'][0]}, {cfg['fram_cfg'][1]}, β {cfg['fram_cfg'][2]:g}) on calibration cases 0–11"),
          ("CUT3 memory",f"H{cfg['CUT']} residual + K/V of layers 0–{cfg['CUT']} per cartridge · {upper_txt(cfg['CUT'],cfg['arch'][0])} consolidated append-only"),
          ("engine",f"{inf['engine_file']} (verbatim, executed up to its acceptance stage) · git blob {inf['engine_blob'][:16]}… · 566.py {cfg['source_blobs']['566.py'][:12]}… · 575.py {cfg['source_blobs']['575.py'][:12]}…"),
          ("helpers",f"{HELPER_FILE} · SHA-256 {cfg['helper_sha256'][:16]}…"),
          ("weights",f"full-weight SHA-256 {inf['weight_sha0'][:16]}… (= reference) · sentinel {inf['sentinel0'][:16]}… · all-parameter guard {inf['guard0'][:16]}…"),
          ("start-up",f"FRAM holdout R@1 {su['hold_r1']}/12 · R@5 {su['hold_r5']}/12 · all R@5 {su['all_r5']}/24 · self-retrieval {su['self_r1']}/{su['self_n']} · reference {su['ref_hold_r1']}/12 · {su['ref_hold_r5']}/12 · baseline CUT3 {su['baseline_correct']}/{su['baseline_n']}"),
          ("init",f"{inf['init_seconds']:.1f} s" if isinstance(inf['init_seconds'],float)and math.isfinite(inf['init_seconds'])else"reused")]
    body="".join(f'<span class="mono">{html.escape(a)}</span> {html.escape(str(b))}<br>'for a,b in rows)
    cf=TEST575["counterfactual"]
    body+=(f'<div class="small" style="margin-top:8px"><b>SEALED TEST610 ACCEPTANCE</b> (610.log; not recomputed by single runs)<br>'
           f'{TEST610["decision"]} · {F6["criteria_passed"]}/{F6["criteria_total"]} criteria · holdout {F6["holdout_correct"]}/{F6["holdout_total"]} · whole panel {F6["all_panel_correct"]}/{F6["all_panel_total"]} (case 08 incorrect) · '
           f'self-retrieval {TEST610["fram"]["self_r1"]}/96 · 4 negative controls without leakage · result SHA-256 {TEST610["result_sha"]}<br>'
           f'<b>SEALED TEST575</b> (a different experiment) · JOINT 72/72 · INCR_DC3 72/72 · BATCH_DC3 72/72 · upper-layer K/V changed {cf["upper_kv_changed"]}/72 · answer changed {cf["answer_changed"]}/72</div>')
    return card_html("Engine status · sealed references",body,"on")
READY_HTML=card_html("Ready",f"Choose a panel case, optionally type your own question, and press <b>RUN</b>. One run scores the question against all 96 cartridges (FRAM), builds the CUT3 memory from the five selected cartridges, "
    f"asks the question without the source text, saves the memory to a file, reloads it, answers again, runs the target-removed control, verifies the frozen weights, seals the payload and renders {N_FIGS} figures (JPEG 300 dpi + vector PDF). "
    "Each run is a LIVE demonstration observation; the TEST610 and TEST575 values above are sealed references.","info")
def is_cuda_error(ex):
    s=f"{type(ex).__name__} {ex}".lower();return"cuda"in s or"device-side assert"in s or"cublas"in s
def run_handler(case_v,custom_v):
    global RUN_COUNTER
    c=parse_case(case_v)
    if c is None or not 0<=c<N_CASES:
        yield pack(card_html("Invalid selection","Choose a panel case (00–23).","warn"));return
    if not GPU_LOCK.acquire(blocking=False):
        yield pack(card_html("Busy","Another run is in progress. Please try again in a moment.","warn"));return
    stage_name="initialisation"
    try:
        RUN_COUNTER+=1;run_id=f"{RUN_PREFIX}{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}-{RUN_COUNTER:04d}-C{c:02d}"
        prune_runs(4);run_dir=ROOT/run_id;run_dir.mkdir(parents=True,exist_ok=True)
        preview=[dict(case=x["case"],correct=x["correct"])for x in SESSION_LEDGER]
        gen=execute_run(INSTR,dict(run_id=run_id,run_dir=run_dir,case=c,custom=(custom_v or"")[:300],session=preview+[dict(case=c,correct=None)]));first=True
        while True:
            try:e=next(gen)
            except StopIteration as s:B=s.value;break
            stage_name=f"{e['stage']}/{N_STAGES} {e['title']}"
            yield pack(stage_card(e),[],None,None,None)if first else pack(stage_card(e));first=False
        SESSION_LEDGER.append(B["ledger"]);LEDGER_PATH.write_text(ledger_text()+"\n",encoding="utf-8")
        P=B["P"];L=P["live"];ok=L["correct"];K=L["disk"];N=L["negative"];R=L["retrieval"]
        raw={"txt":B["txt"].read_text(encoding="utf-8"),"json":B["payload"].read_bytes().decode("utf-8"),"man":B["manifest"].read_text(encoding="utf-8"),"ledger":ledger_text()}
        neg=("not applicable (target not selected)"if N.get("skipped")else f'{html.escape(str(N["answer"]))} · expected answer produced: {"yes" if N["gold_leak"] else "no"}')
        body=(f"<b>{html.escape(B['run_id'])}</b> · LIVE DEMONSTRATION RUN<br>"
              f'<span class="mono">question</span> {html.escape(L["question"])}<br>'
              f'<span class="mono">FRAM</span> target #{L["target"]:02d} · exact rank {R["fram"]["rank"]}/96 · reference rank {R["ref"]["rank"]}/96 · selected {L["engine_order"]}<br>'
              f'<span class="mono">answer</span> <b>{html.escape(L["answer"])}</b> · expected {html.escape(L["expected"])} · <b>{"CORRECT" if ok else "INCORRECT"}</b> · sealed TEST610 answer {html.escape(L["sealed_answer"])} (same: {L["matches_sealed"]})<br>'
              f'<span class="mono">memory</span> {L["memory"]["T"]} tokens · {L["memory"]["bytes"]:,} B · token-id forwards during build {L["source_reads"]} (instruction prefix + each cartridge\'s own fact, once) · earlier rows bitwise unchanged at all appends: {all(s_["prior_rows_bitwise_unchanged"] for s_ in L["steps"][1:])}<br>'
              f'<span class="mono">disk</span> {html.escape(K["file"])} · {K["bytes"]:,} B · SHA-256 {K["sha256"][:16]}… · reload bitwise equal: {K["bit_equal"]} · answer after reload identical: {K["answer_identical"]}<br>'
              f'<span class="mono">control</span> target removed → {neg}<br>'
              +(f'<span class="mono">free text</span> {html.escape(L["custom"]["question"])} → {html.escape(L["custom"]["answer"])} (exact FRAM top-5 {L["custom"]["ids"]}; unscored)<br>'if L.get("custom")else"")+
              f"weights unchanged (sentinel + all-parameter guard) · verdict <b>{html.escape(B['verdict'])}</b> · {N_FIGS} figures + PDFs + ZIP · {B['checks']}/{B['checks']} checks PASS<br>"
              f'<span class="mono">payload SHA-256 {B["sha"]}</span>')
        yield pack(card_html(f"{N_STAGES}/{N_STAGES} · Sealed",body,"on"),[(str(p),c_)for p,c_ in B["imgs"]],
                   {"zip":B["zip"],"json":B["payload"],"txt":B["txt"],"man":B["manifest"],"session":B["session"],"ledger":LEDGER_PATH},[p for p,_ in B["imgs"]],raw)
    except Exception as ex:
        print("="*140);print(f"RUN FAILED — stage: {stage_name}");print(f"{type(ex).__name__}: {ex}");traceback.print_exc();print("="*140)
        kind="AUDIT FAIL — nothing was sealed. "if isinstance(ex,AuditFail)else"";partial=getattr(ex,"partial",None)
        if partial:kind+="The raw outputs gathered so far were preserved (FULL RUN LOG). "
        tip="<br><b>CUDA error: Runtime → Restart runtime, then run PARTS 1–3 again.</b>"if is_cuda_error(ex)else"<br>The full traceback is printed in the Colab console."
        card=card_html("Run failed",f"{html.escape(kind)}Stage: {html.escape(stage_name)}<br>{html.escape(type(ex).__name__)}: {html.escape(str(ex)[:600])}{tip}","err")
        if partial:yield pack(card,[],{"json":partial},None,{"txt":traceback.format_exc(),"json":Path(partial).read_text(encoding="utf-8"),"man":"","ledger":ledger_text()})
        else:yield pack(card,[],None,None,None)
    finally:
        plt.close("all");GPU_LOCK.release()
        try:torch.cuda.empty_cache()
        except Exception:pass
REPLAY_LABELS=("⬇ ACCEPTANCE REPLAY (.json)","⬇ ACCEPTANCE REPLAY (.txt)","⬇ ENGINE STDOUT (.txt)")
def replay_handler():
    if not GPU_LOCK.acquire(blocking=False):
        yield card_html("Busy","Another run is in progress. Please try again in a moment.","warn"),*[dl_update(None,l)for l in REPLAY_LABELS];return
    started=False
    try:
        rid=f"{REPLAY_PREFIX}{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}";prune_runs(2,REPLAY_PREFIX);d=ROOT/rid;d.mkdir(parents=True,exist_ok=True)
        st={"done":0,"last":None,"res":None,"err":None}
        def prog(n,tot,line):st["done"]=n;st["last"]=line
        def work():
            try:st["res"]=replay_report(INSTR,rid,d,prog)
            except BaseException as ex:st["err"]=ex;traceback.print_exc()
            finally:GPU_LOCK.release()  # the GPU lock is held until the replay thread itself ends, even if the browser disconnects
        th=threading.Thread(target=work,daemon=True);started=True;th.start()
        while th.is_alive():
            lr=st["last"];tail=(f"<br>last engine line: <span class='mono'>{html.escape(str(lr)[:160])}</span>"if lr else"")
            yield card_html("Acceptance replay running",f"Executing the 610.py acceptance stage [6/7]–[7/7] verbatim: 24 panel cases, four bitwise rebuilds, four negative controls, final audit, weight checks (LIVE; separate from the sealed TEST610 record).{tail}"+pbar_html(st["done"],44),"info"),*[SKIP()for _ in REPLAY_LABELS]
            time.sleep(1.5)
        if st["err"]is not None:raise st["err"]
        R=st["res"];o=R["out"]
        body=(f"<b>LIVE {html.escape(o['decision'])}</b> · criteria {o['criteria_passed']}/{o['criteria_total']} · panel {o['panel_correct']}/24 · holdout {o['holdout_correct']}/12 · {o['seconds']} s · weights unchanged: {o['weights_unchanged']}<br>"
              f"Answers identical to the sealed TEST610 answers: {o['answers_identical_to_sealed_TEST610']}/24. Sealed TEST610: {TEST610['decision']} · {TEST610['final']['criteria_passed']}/15 · panel {TEST610['final']['all_panel_correct']}/24. "
              "The replay is a separate live measurement and does not replace the sealed record.<br>"
              f'<span class="mono">replay SHA-256 {R["sha"]}</span>')
        yield card_html("Acceptance replay complete",body,"on"),dl_update(R["json"],REPLAY_LABELS[0]),dl_update(R["txt"],REPLAY_LABELS[1]),dl_update(R["stdout"],REPLAY_LABELS[2])
    except Exception as ex:
        yield card_html("Acceptance replay failed",f"{html.escape(type(ex).__name__)}: {html.escape(str(ex)[:500])}","err"),*[dl_update(None,l)for l in REPLAY_LABELS]
    finally:
        if not started:GPU_LOCK.release()
        try:torch.cuda.empty_cache()
        except Exception:pass
def verify_handler():
    if not GPU_LOCK.acquire(blocking=False):return"Busy: a run is in progress."
    try:
        t=time.perf_counter();g=INSTR.guard();fz=INSTR.frozen();w=INSTR.full_weight_sha();cfg=INSTR.config()
        out=dict(model=MODEL_ID,engine_status=INSTR.status(),full_weight_sha256_now=w,full_weight_sha256_reference=WEIGHT_SHA_EXPECTED,full_weight_identical_to_reference=w==WEIGHT_SHA_EXPECTED,
                 all_parameter_guard_now=g,all_parameter_guard_startup=INSTR.info["guard0"],guard_unchanged=g==INSTR.info["guard0"],
                 sentinel_unchanged=fz["sentinel"]==INSTR.info["sentinel0"],trainable_tensors=fz["trainable_tensors"],foreign_hooks=fz["foreign_hooks"],residual_hooks=fz["residual_hooks"],
                 engine_file=ENGINE_FILE,engine_blob=cfg["engine_blob"],engine_blob_expected=ENGINE_BLOB_EXPECTED,engine_sha256=cfg["engine_sha256"],engine_sha256_expected=ENGINE_SHA256_EXPECTED,
                 engine_exec_sha256=cfg["engine_exec_sha256"],helper_sha256=cfg["helper_sha256"],source_blobs=cfg["source_blobs"],panel_sha=cfg["panel_sha"],fram_cfg=cfg["fram_cfg"],
                 cut=cfg["CUT"],architecture=cfg["arch"],checked_utc=utc_now(),seconds=round(time.perf_counter()-t,3))
        return json.dumps(jsafe(out),indent=2,ensure_ascii=False)
    finally:GPU_LOCK.release()
# ---------------- service self-test (runs before the public link is printed) ----------------
def service_selftest():
    global QUIET;out=[];d=ROOT/"_selftest";shutil.rmtree(d,ignore_errors=True);d.mkdir(parents=True);(d/"w.txt").write_text("ok");out.append("ROOT writable");QUIET=True
    try:
        for c_,cu in((12,""),(3,"Who governs this place?"),(5,"")):
            dd=d/f"ok{c_}";dd.mkdir()
            B=drain(execute_run(SelfTestInstrument("ok"),dict(run_id=f"SELFTEST-C{c_:02d}",run_dir=dd,case=c_,custom=cu,session=[dict(case=c_,correct=None)])))
            assert len(B["imgs"])==N_FIGS and len(B["pdfs"])==N_FIGS and file_ready(B["zip"])and file_ready(B["session"])and B["verdict"].endswith("SOURCE_FREE_QUERY")
        out.append(f"full pipeline on a synthetic instrument (correct and incorrect answer, target not selected, with and without free text): {N_FIGS}/{N_FIGS} figures (JPEG 300 dpi + PDF), ZIP, session file, {B['checks']} checks")
        for mode,what in(("guard_changes","a changed weight guard"),("prior_changed","a changed previously consolidated row"),("reread","an extra source read during consolidation"),
                         ("source_in_query","source tokens in the query input"),("config","a changed FRAM configuration"),("weight_ref","a full-weight SHA-256 different from the reference"),
                         ("residual_hook","a hook left on a decoder layer after the run"),("startup","start-up gates that do not reproduce TEST610"),("fact_in_file","fact text inside the session file"),
                         ("reload_diff","a reloaded memory that is not bitwise equal"),("reload_answer","a different answer after reload")):
            d2=d/mode;d2.mkdir()
            try:drain(execute_run(SelfTestInstrument(mode),dict(run_id="SELFTEST-"+mode.upper(),run_dir=d2,case=13,custom="",session=[])))
            except AuditFail:out.append(f"fail-closed: {what} aborts the run (nothing sealed)")
            else:raise RuntimeError(f"self-test: {what} did not abort the run")
        d3=d/"replay";d3.mkdir();R=replay_report(SelfTestInstrument("ok"),"SELFTEST-REPLAY",d3)
        assert R["out"]["panel_correct"]==23 and R["out"]["criteria_passed"]==15 and file_ready(R["json"])and file_ready(R["txt"])and file_ready(R["stdout"])
        out.append("acceptance replay report (synthetic): JSON + TXT + engine stdout, scores re-derived from rows")
    finally:QUIET=False
    if len(pack(READY_HTML))!=N_OUTPUTS or len(pack(READY_HTML,[],None,None,None))!=N_OUTPUTS:raise RuntimeError("callback output count mismatch")
    out.append(f"callback output count = {N_OUTPUTS}");out.append(f"file route {FILE_URL_PREFIX}");shutil.rmtree(d,ignore_errors=True);return out
say("="*140);say(DEMO_TITLE+" — "+MODEL_FAMILY+" · PART 3 / 3 · SELF-TEST AND INTERFACE");say("[1/3] SERVICE SELF-TEST")
for c_ in service_selftest():say(" PASS ·",c_)
#<<UI_END>>
# ---------------- interface ----------------
say("[2/3] INTERFACE")
CSS="""
:root{--kz-on:#0f766e;--kz-fg:#111827;--kz-card:#ffffff;--kz-bd:#d1d5db;--kz-a:#1d4ed8;--kz-err:#b91c1c;--kz-warn:#b45309;--kz-hero:#0b1220;--kz-gold:#b45309}
.dark{--kz-on:#2dd4bf;--kz-fg:#f3f4f6;--kz-card:#111827;--kz-bd:#4b5563;--kz-a:#60a5fa;--kz-err:#f87171;--kz-warn:#fbbf24;--kz-hero:#0b1220;--kz-gold:#fbbf24}
html,body{overflow-x:hidden!important}
.gradio-container{width:100%!important;max-width:920px!important;margin:0 auto!important;padding-left:10px!important;padding-right:10px!important;box-sizing:border-box!important}
.gradio-container *{box-sizing:border-box}
.kz{color:var(--kz-fg)!important;max-width:100%;overflow-wrap:anywhere;line-height:1.45}
.kz *{color:inherit}
.hero{background:#0b1220;color:#f9fafb!important;border-radius:6px;padding:18px 18px 14px;margin:4px 0 8px}
.hero *{color:#f9fafb!important}
.rec{font-size:11.5px;letter-spacing:1.6px;text-transform:uppercase;opacity:.85;border:1px solid rgba(255,255,255,.35);display:inline-block;padding:2px 8px;border-radius:3px}
.brand{font-size:clamp(24px,6.4vw,34px);font-weight:800;letter-spacing:.4px;line-height:1.1;margin-top:10px}
.title{font-size:clamp(15px,4vw,19px);font-weight:600;margin-top:4px}
.para{font-size:clamp(14px,3.8vw,17px);margin-top:10px;color:#fde68a!important;font-weight:600}
.q{font-size:13.5px;margin-top:8px;font-style:italic;opacity:.95}
.by{font-size:14px;margin-top:10px}
.by b{color:#fde68a!important}
.lic{font-size:12px;opacity:.85;margin-top:6px}
.lic a{text-decoration:underline}
.msg{margin:6px 0 0;padding-left:18px;font-size:13.5px}
.card{background:var(--kz-card);border:1px solid var(--kz-bd);border-left:4px solid var(--kz-bd);border-radius:4px;padding:10px 12px;margin:6px 0}
.card.on{border-left-color:var(--kz-on)}.card.err{border-left-color:var(--kz-err)}.card.warn{border-left-color:var(--kz-warn)}.card.info{border-left-color:var(--kz-a)}
.card .h{font-weight:700;font-size:15px;margin-bottom:4px}
.small{font-size:13.5px}.mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;opacity:.8;margin-right:4px}
#kz_run button,#kz_run{font-size:clamp(15px,4.4vw,18px)!important;font-weight:700!important;min-height:54px!important}
"""
DL_ALL_JS="""(u)=>{let items=[];try{items=JSON.parse(u||"[]")}catch(e){items=[]}
let i=0;const next=()=>{if(i>=items.length)return;const it=items[i++];const a=document.createElement('a');a.href=it.url;a.download=it.name;a.rel='noopener';
document.body.appendChild(a);a.click();setTimeout(()=>{a.remove();next()},900)};next();return u;}"""
HERO=(f'<div class="kz hero"><span class="rec">Public technical demonstration · priority record</span>'
      f'<div class="brand">AKBASCORE MAM-BÇ</div><div class="title">External associative memory for a frozen language model · {html.escape(MODEL_FAMILY)} · CUT3 + FRAM</div>'
      f'<div class="para">{html.escape(PARADIGM)}.</div><div class="q">{html.escape(SCI_QUESTION)}</div>'
      f'<div class="by"><b>{html.escape(DISCOVERY)}.</b><br>{html.escape(ORG)} · {html.escape(DATE_TXT)} · {html.escape(AUTHOR_PLACE)}</div>'
      '<ul class="msg">'+"".join(f"<li>{html.escape(m)}</li>"for m in CORE_MESSAGE)+"</ul>"
      f'<div class="lic">{html.escape(PRIORITY)}</div>'
      f'<div class="lic">{html.escape(LICENSE_NAME)} · {html.escape(COPYRIGHT)} · <a href="{LICENSE_URL}" target="_blank" rel="noopener">{html.escape(LICENSE_URL)}</a> · '
      f'<a href="{ENGINE_URL}" target="_blank" rel="noopener">610.py engine</a> · <a href="{ENGINE_LOG_URL}" target="_blank" rel="noopener">TEST610 log</a> · <a href="{BASE_URL}" target="_blank" rel="noopener">566.py</a> · '
      f'<a href="{LOG575_URL}" target="_blank" rel="noopener">TEST575 log</a> · DOI <a href="{PRIOR_PUB["url"]}" target="_blank" rel="noopener">{html.escape(PRIOR_PUB["doi"])}</a> · DOI <a href="{QWEN_PUB["url"]}" target="_blank" rel="noopener">{html.escape(QWEN_PUB["doi"])}</a></div></div>')
SCOPE_HTML=card_html("Scope of this demonstration",
    "<b>Demonstrated live in every run:</b><br>"+"<br>".join("• "+html.escape(t)for t in SCOPE["demonstrated"])+"<br><br><b>Archived (sealed, not run by a single run):</b><br>"+"<br>".join("• "+html.escape(t)for t in SCOPE["archived"])
    +"<br><br><b>Not established:</b><br>"+"<br>".join("• "+html.escape(t)for t in SCOPE["not_established"])+"<br><br><b>Technical notes:</b><br>"+"<br>".join("• "+html.escape(t)for t in SCOPE["notes"]),"info")
def _tb(**kw):
    for extra in(([{"buttons":["copy"]}]if _GR_MAJOR>=6 else[])+[{"show_copy_button":True},{}]):
        try:return gr.Textbox(**extra,**kw)
        except Exception:pass
def _gallery(**kw):
    try:return gr.Gallery(columns=1,height="auto",object_fit="contain",preview=False,**kw)
    except TypeError:return gr.Gallery(**kw)
if"demo"in globals():
    try:demo.close()
    except Exception:pass
_TITLE="AKBASCORE MAM-BÇ · Qwen2.5-7B · CUT3 + FRAM external memory · Mustafa Akbaş"
if _GR_MAJOR>=6:_blocks=gr.Blocks(title=_TITLE)
else:
    try:_blocks=gr.Blocks(css=CSS,title=_TITLE)
    except TypeError:_blocks=gr.Blocks(title=_TITLE)
_CCH=case_choices(INSTR);_DEF=_CCH[12]
with _blocks as demo:
    gr.HTML(HERO);gr.HTML(engine_card_html(INSTR))
    case_dd=gr.Dropdown(choices=_CCH,value=_DEF,label="Panel case (locked 24-case panel · CAL = calibration 0–11 · HOLD = holdout 12–23)",interactive=True)
    case_info=gr.HTML(case_info_html(_DEF))
    custom_tb=gr.Textbox(value="",label="Optional free-text question (exact FRAM selects five cartridges; answered from their CUT3 memory; not scored)",placeholder="e.g. Which city is the current capital of Zorvan?",lines=1,max_lines=2)
    run_btn=gr.Button("RUN · FRAM select · CUT3 memory · answer without the source · save → reload → answer · control · seal",variant="primary",elem_id="kz_run")
    status=gr.HTML(READY_HTML)
    gallery=_gallery(label=f"{N_FIGS} figures (tap to open · JPEG 300 dpi; vector PDFs in the ZIP)",show_label=True)
    dl_all=gr.Button(DL_ALL_LABEL,interactive=False)
    jpg_urls=gr.Textbox(value="",visible=False)
    with gr.Row():
        dlb=[gr.DownloadButton(label=l,value=None,interactive=False)for l in DL_LABELS]
    with gr.Accordion("Direct file links (fallback)",open=False):
        fls=[gr.File(label=n,interactive=False)for n in("ZIP · figures, PDFs, payload, logs, session file","FULL RUN LOG (.json)","READABLE RUN LOG (.txt)","RUN MANIFEST (.json)","CUT3 SESSION FILE (.pt)","SESSION LEDGER (.jsonl)")]
    with gr.Tabs():
        with gr.Tab("READABLE RUN LOG (.txt)"):raw_txt=_tb(lines=18,max_lines=40,label="readable run log")
        with gr.Tab("FULL RUN LOG (.json)"):raw_json=_tb(lines=18,max_lines=40,label="sealed canonical payload")
        with gr.Tab("MANIFEST (.json)"):raw_man=_tb(lines=12,max_lines=30,label="run manifest")
        with gr.Tab("SESSION LEDGER"):raw_led=_tb(lines=10,max_lines=30,label="one line per sealed run in this session")
    with gr.Accordion("Live replay of the TEST610 acceptance stage · 610.py [6/7]–[7/7] verbatim · about 2–4 minutes on an A100 (reported separately from the sealed TEST610 record)",open=False):
        rp_btn=gr.Button("Run the acceptance replay")
        rp_status=gr.HTML(card_html("Acceptance replay","Executes the remaining 610.py acceptance stage unchanged: 24 panel cases (with repeat), four bitwise rebuilds, four target-removed controls, the final CUT3 audit and the weight / hook checks; writes its own JSON/TXT report and the verbatim engine output. It does not replace the sealed TEST610 result.","info"))
        with gr.Row():rp_dl=[gr.DownloadButton(label=l,value=None,interactive=False)for l in REPLAY_LABELS]
    with gr.Accordion("Verification · full-weight SHA-256 · weight guard · engine, helper and panel hashes",open=False):
        ver_btn=gr.Button("Run verification now")
        ver_out=_tb(lines=14,max_lines=40,label="verification record")
    gr.HTML(SCOPE_HTML)
    gr.HTML(f'<div class="kz small" style="opacity:.8;margin:8px 0 18px">{html.escape(DISCOVERY)} · {html.escape(ORG)} · {html.escape(DATE_TXT)} · {html.escape(LICENSE_NAME)} · '
            f'<a href="{LICENSE_URL}" target="_blank" rel="noopener">{html.escape(LICENSE_URL)}</a></div>')
    OUTS=[status,gallery,dl_all,jpg_urls]+dlb+fls+[raw_txt,raw_json,raw_man,raw_led]
    if len(OUTS)!=N_OUTPUTS:raise RuntimeError(f"UI outputs {len(OUTS)} != {N_OUTPUTS}")
    case_dd.change(lambda a:case_info_html(a),inputs=[case_dd],outputs=case_info)
    run_btn.click(run_handler,inputs=[case_dd,custom_tb],outputs=OUTS)
    rp_btn.click(replay_handler,inputs=None,outputs=[rp_status]+rp_dl)
    ver_btn.click(verify_handler,inputs=None,outputs=ver_out)
    try:dl_all.click(None,inputs=[jpg_urls],outputs=None,js=DL_ALL_JS)
    except TypeError:dl_all.click(None,inputs=[jpg_urls],outputs=None,_js=DL_ALL_JS)
try:demo.queue(default_concurrency_limit=1)
except TypeError:demo.queue(concurrency_count=1)
say("[3/3] LAUNCH (public share link) — open the printed gradio.live link, choose a panel case, press RUN.")
_LAUNCH=dict(share=True,allowed_paths=ALLOWED_PATHS,show_error=True,inline=False)
if _GR_MAJOR>=6:_LAUNCH["css"]=CSS
try:demo.launch(**_LAUNCH)
except TypeError as _ex:
    if"css"not in str(_ex):raise
    _LAUNCH.pop("css",None);demo.launch(**_LAUNCH)
