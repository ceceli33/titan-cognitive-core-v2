# AKBASCORE MAM — The Silicon Hippocampus

**Physical memory cartridges for a frozen language model: WRITE · SSD · READ**

Mustafa Akbaş · AkbasCore AI Teknoloji · Mersin, Türkiye · 12 October 2026
License: AKBASCORE RESEARCH SOFTWARE LICENSE — [`LICENSE`](LICENSE)

**Full technical report (PDF):** [`docs/AKBASCORE_MAM_The_Silicon_Hippocampus_Technical_Report.pdf`](docs/AKBASCORE_MAM_The_Silicon_Hippocampus_Technical_Report.pdf) · Zenodo DOI: *to be added after Zenodo publication*

---

## Summary

AKBASCORE MAM is a model-native memory architecture for a frozen language model. In this release, **frozen Qwen2.5-7B-Instruct** works as follows:

- **WRITE:** the model reads each fact once and its own attention key/value state is kept as a **CUT3 cartridge**.
- **SSD:** every cartridge is stored as an independent file with a SHA-256 manifest.
- **READ:** later, the model receives **only the question**. Its native layer-3 representation addresses the cartridge bank through a label-free **PCA128 + FAISS IVF128** index, and the locked **PEAK50** rule selects one cartridge. The file is verified, loaded into the attention cache, and the model answers. The source text is not given to the model again.

The model weights are not changed: no training, no fine-tuning, no LoRA, no optimizer, and no hooks left on the model.

“The Silicon Hippocampus” is the name of the architecture's role — form a memory once, keep it outside the model, recall it later. It is not a claim of biological equivalence.

### Measured in the published run ([`Hippocampus.log`](Hippocampus.log))

**Pool and start-up acceptance**

| Quantity | Value |
|---|---|
| Physical cartridge files | 1,453 (96 original + 1,357 synthetic facts) |
| Disk pool | 3,973,450,961 bytes |
| Start-up acceptance, original wording (48 questions) | 48/48 correct file selected · 48/48 correct answers from disk · 0 errors |
| Start-up acceptance, new wording (48 questions) | 48/48 correct file selected · 48/48 correct answers from disk · 0 errors |
| Integrity at start-up | 8/8 (source SHA lock, manifest SHA of all 1,453 files, weights, sentinel, zero trainable, no hooks, tensor load, lossless sample) |

**Live RUN, case 00**

| Quantity | Value |
|---|---|
| WRITE | re-executed; bitwise identical to the stored file in all 28 layers |
| READ | correct file #0292 selected · answer “Fgvrkerflc” = expected |
| Sealing checks | 43/43 PASS (40 in the sealed payload + 3 sealing-stage checks) |
| Pipeline time (`engine_total`) | 2.787 s |

The 2.787 s figure is the whole audited RUN pipeline for one question. It includes integrity checks, the weight guard and the control answer, so it is not normal inference throughput. Per-stage timings are given in [Measured results](#measured-results).

---

## Contents

1. [Architecture](#architecture)
2. [WRITE — how a cartridge is formed](#write--how-a-cartridge-is-formed)
3. [SSD — the physical cartridge pool](#ssd--the-physical-cartridge-pool)
4. [READ — geometric address, PEAK50 selection, answer](#read--geometric-address-peak50-selection-answer)
5. [Why this is not text RAG](#why-this-is-not-text-rag)
6. [Integrity controls and fail-closed branches](#integrity-controls-and-fail-closed-branches)
7. [Storage, RAM and GPU memory](#storage-ram-and-gpu-memory)
8. [Measured results](#measured-results)
9. [How PEAK50 was locked](#how-peak50-was-locked)
10. [Scientific scope and next validation steps](#scientific-scope-and-next-validation-steps)
11. [Running the demonstration (Google Colab)](#running-the-demonstration-google-colab)
12. [Outputs, figures and logs](#outputs-figures-and-logs)
13. [Source files and engine chain](#source-files-and-engine-chain)
14. [License](#license)
15. [Citation](#citation)

---

## Architecture

```
                     A · WRITE  (once per fact, before any question)
  fact text ─► frozen Qwen2.5-7B ─► CUT3: 28 × (K, V) bf16 ─► cut3_XXXXX.pt + SHA-256 manifest
                 │                                                   │
                 │ native layer-3 residual rows                      ▼
                 ▼                                          B · SSD: 1,453 independent files
     RAM: PCA128 basis + FAISS IVF128 (label-free)                   │
                 │                                                   │
                     C · READ  (every question; source text not given)
  question ─► frozen Qwen (question frame only) ─► layer-3 question rows
          ─► PCA128 → IVF128 (nprobe 2, token_k 16) → touched cartridges
          ─► TOP2 → 16 candidates ─► PEAK50 → 1 cartridge
          ─► SHA-256 check + torch.load ─► attention cache ─► frozen model answers
```

Model: `Qwen/Qwen2.5-7B-Instruct`, bfloat16, 28 layers, hidden 3584, 28 query heads, 4 KV heads, head dimension 128. Full-weight SHA-256 `b441b005d826d45f2778291696ac5ae22dfb3421defb37c539904d8d38f93786`.

![Figure 1 — WRITE, SSD and READ architecture of the Silicon Hippocampus with the live values of case 00](docs/figures/fig_01_write_ssd_read.jpg)

*Figure 1 — The Silicon Hippocampus: WRITE · SSD · READ overall architecture, rendered by the demo for live RUN `MAMSH-20261012-053712-0001-C00` (case 00). [PDF version](docs/figures/fig_01_write_ssd_read.pdf).*


---

## WRITE — how a cartridge is formed

Every fact is placed after a fixed 29-token instruction prefix (system prompt + `INFORMATION:`). The engine function `init(ci, 3)` (566.py) then runs two forward passes.

1. **Forward 1 — token ids, no cache.**
   - Input: the instruction prefix plus the fact tokens (*T* rows; case 00: 29 + 19 = 48).
   - The model returns its key/value cache for all layers, and a forward hook captures the **layer-3 hidden state H3** for every row.
   - The **K/V of layers 0–3** are taken from this pass.
2. **Forward 2 — all-zero input embeddings, no cache.**
   - Input: *T* rows of zeros. A hook on layer 3 replaces its output with H3.
   - The **K/V of layers 4–27** are taken from this pass. They are therefore produced from the layer-3 state alone, not from re-reading the text.
3. **Cartridge:**
   - 28 layers × (K, V) = **56 tensors**, each of shape `[1, 4, T, 128]` in bfloat16.
   - Size: 28 × 2 × 4 × 128 × 2 bytes = **57,344 bytes per row**. Case 00: 48 rows = 2,752,512 bytes.
4. **File:**
   - The 56 tensors are saved with `torch.save` as `cut3_XXXXX.pt`. Case 00: `cut3_00292.pt`, 2,768,949 bytes, of which 16,437 bytes are container overhead.
   - The SHA-256 of every file is recorded in `CUT3_MANIFEST.json`.

During the same start-up the engine also captures the native layer-3 residual rows of each fact span and normalises them in the locked TEST610 frame: centred by the mean of the 96 original cartridges, then unit-normalised. These rows are the address keys. They contain no labels.

**Live re-verification in every RUN.** The demo calls `init(fact, 3)` again for the fact of the case. It records every model forward and checks the following:

- exactly two forwards: token ids of prefix + fact, then all-zero embeddings over the same rows;
- the cartridge shape;
- the byte count;
- the 56 tensors compared with the stored file **as raw bytes**.

In the published run all 56 tensors were bitwise identical. If a comparison fails, a second live re-write is made first. This separates nondeterministic re-execution (the two live re-writes differ from each other) from a stored-memory mismatch (the re-writes agree with each other but not with the file). Both comparisons are written to the failure record, and the run stops without sealing.

The cartridge file contains tensors only. For case 00 the fact text, the answer word and the question were each checked to be absent from the file bytes.

---

## SSD — the physical cartridge pool

| Property | Value |
|---|---|
| Files | 1,453 independent `.pt` files |
| Total | 3,973,450,961 bytes |
| File size | min 2,252,853 · median 2,768,949 · max 2,883,637 bytes |
| Tensor payload | 3,949,568,000 bytes, all bfloat16 · 56 tensors per file |
| Write at start-up | 1,453 files · median 83.5 ms per file · 121.6 s summed over files (`init` + save + SHA-256) |
| Integrity at start-up | size and SHA-256 of all 1,453 files equal to the manifest; every file loads with the expected tensor structure |
| Integrity in each RUN | 1,453 files present with the manifest sizes; the written and selected files re-hashed against the manifest |
| Lossless gzip sample | ratio 0.796091 on 32 predeclared files, bitwise restored. The full-pool value is an estimate and was not measured |


![Figure 2 — formation of one CUT3 cartridge in two forward passes and its storage as a physical file](docs/figures/fig_02_write_ssd.jpg)

*Figure 2 — WRITE → SSD: how one physical CUT3 cartridge (`cut3_00292.pt`) is formed and stored, with the live bitwise re-verification of case 00. [PDF version](docs/figures/fig_02_write_ssd.pdf).*

---

## READ — geometric address, PEAK50 selection, answer

### 1. Question frame
The question is placed after the same instruction prefix (`PREFIX + question + "\n\n"`) and read in **one forward pass** with no cache. The layer-3 residual rows of the question span are taken and normalised in the same frame as the cartridge keys. Case 00: 42 input tokens, 13 question rows. No fact text is part of this input; the demo verifies the exact token ids.

### 2. Label-free geometric address (RAM)
- **PCA128:**
  - fitted with `torch.pca_lowrank` on 2,048 native positions sampled with a fixed seed, using the stored cartridge positions only;
  - question rows are projected as `(q − CENTER) @ BASIS`.
- **FAISS `IndexIVFFlat`:**
  - inner product, 128 inverted lists over all 26,738 native positions of the 1,453 cartridges;
  - search with `nprobe = 2`, `token_k = 16` nearest positions per question row.
- **Touched set:** the cartridges that own the returned positions. Case 00: 103 of 1,453.
- **TOP2 candidate score:** for each question row *t*, the two highest similarities inside cartridge *c* form
  `TOP2(c) = mean_t [ (log Σ_{2 best j} exp(β·s_tj) − log n_c) / β ]`, with β = 8.
  The touched cartridges are ordered by TOP2, ties broken by cartridge id, and the first **16** become candidates.

### 3. PEAK50 selection
Let `s_tj = q_t · k_j` be the full-dimension similarity between question row *t* and cartridge row *j*. For each candidate *c* with *n_c* rows:

```
FRAM(c)   = mean_t [ ( logsumexp_j(β · s_tj) − log n_c ) / β ]      β = 8
peak(c)   = mean_t   max_j s_tj
PEAK50(c) = 0.5 · FRAM(c) + 0.5 · peak(c)
```

Candidates are ranked by PEAK50 in descending order, ties broken by cartridge id, and the first is selected (`features626` + `rerank626` with the locked setting `('PEAK50', 0.5, 0.0)`).

In every RUN the demo checks that:

- the ranked set equals the `candidate626` set (no candidate added, merged or dropped);
- its size is min(16, touched);
- every score re-derives as 0.5 · FRAM + 0.5 · peak;
- the order follows the engine's tie rule;
- the selected cartridge and the top-5 equal the engine's own start-up record for the same question.

### 4. Load and answer
- The selected file's SHA-256 is recomputed and compared with the manifest. The file is loaded with `torch.load(weights_only=True)`, and the 56 tensors are moved to the GPU as the attention cache.
- `answer(memory, question)` gives the model **only the question template**:
  `"\nQUESTION:\n{q}\n\nANSWER:\n<|im_end|>\n<|im_start|>assistant\n"`.
- Positions continue after the *T* cartridge rows. Decoding is greedy, up to 16 new tokens.
- The instruction prefix and the fact enter only through the loaded K/V. No fact token is input at READ time.
- Case 00: 24 question-template tokens over the 48-row cartridge, then 6 one-token generation steps → “Fgvrkerflc” (expected “Fgvrkerflc”).

**Control.** The same question is asked over the runner-up candidate's file (#0293 in case 00). It produced “ZX624196 does not have a listed capital.”, not the expected answer.


![Figure 3 — question-native addressing, PEAK50 candidates, disk load, answer and runner-up control](docs/figures/fig_03_read.jpg)

*Figure 3 — READ: question-native addressing, the 16 PEAK50 candidates, disk load, the answer and the runner-up control for case 00. [PDF version](docs/figures/fig_03_read.pdf).*

---

## Why this is not text RAG

| | Text RAG | AKBASCORE MAM |
|---|---|---|
| What is stored | text chunks (plus an embedding per chunk) | the model's own attention K/V state for each fact (CUT3 cartridge) |
| How it is found | similarity between external text embeddings | the frozen model's native layer-3 question rows against native layer-3 cartridge rows (PCA128 + IVF128, then PEAK50) |
| What enters the model at answer time | retrieved text pasted into the prompt and processed again | the cartridge's K/V loaded into the attention cache; the prompt holds only the question template |
| Encoder | usually a separate embedding model | the same frozen model |
| Training | none, or a separately trained retriever | none |
| Dependencies | text store / vector service | per-fact `.pt` files with a SHA-256 manifest; an in-process FAISS index; no SQL, no vector-database service |

---

## Integrity controls and fail-closed branches

**Engine locks (checked at start-up):**
- TEST627 demo engine SHA-256 `ab53e2d5…` (embedded verbatim);
- 626.py, 624.py and 610.py SHA-256;
- 566.py and 575.py git blobs;
- model and architecture;
- full-weight SHA-256;
- TEST610 baseline;
- 1,453 files with manifest sizes;
- index size;
- locked READ setting;
- engine status COMPLETE with 8/8 integrity;
- zero trainable tensors, no LoRA, no optimizer, no hooks on decoder layers, attention or `q_proj`.

**Frozen-model checks in every RUN:** full-weight SHA-256 at start-up; TEST566 sentinel and an all-parameter guard (sums, absolute sums, sums of squares of every parameter) before and after the run; trainable tensors; hooks. A recording-only forward pre-hook is attached only while an engine call runs; it records the kind and length of every model input and the installed cache length.

**Fail-closed branches.** Each of the following aborts the run with no seal and no figures, and preserves the raw outputs gathered so far. All 13 are exercised by the GPU-free self-test before the interface starts:

1. changed weight guard
2. WRITE result not bitwise identical to its file
3. extra model forward during WRITE
4. fact text inside a cartridge file
5. pool file size different from the manifest
6. fact tokens in the question frame
7. a PEAK50 score that does not re-derive
8. ranked set different from the `candidate626` set
9. live READ that does not reproduce the engine's start-up record
10. selected file whose SHA-256 differs from the manifest
11. fact tokens in the answering input
12. hook left on a decoder layer
13. engine start-up status other than COMPLETE

---

## Storage, RAM and GPU memory

These quantities are kept separate.

| Where | What | Size |
|---|---|---|
| Disk (SSD) | 1,453 cartridge files | 3,973,450,961 bytes |
| RAM — address index | FAISS IVF128 (serialized) | 13,970,459 bytes |
| RAM — address index | PCA128 basis (3584 × 128, float32) | 1,835,008 bytes |
| RAM — reranking data | native layer-3 rows of all positions (26,738 × 3584, float32) | 383,315,968 bytes |
| GPU | frozen model weights (bfloat16) and the loaded cartridge | peak PyTorch allocation in the case-00 RUN: 18.27 GiB, model included |

The storage report excludes model RAM/VRAM and Python allocator overhead.

---

## Measured results

### Start-up acceptance in the published runtime ([`Hippocampus.log`](Hippocampus.log))

The locked TEST627 engine ran at start-up in the same runtime. Questions are answered from the disk files.

| Panel (48 questions) | Correct file | Correct answer | Errors |
|---|---|---|---|
| Original wording | 48/48 | 48/48 | 0 |
| New wording | 48/48 | 48/48 | 0 |

| Panel — median per question | Address | Load | Answer |
|---|---|---|---|
| Original wording | 23.80 ms | 17.29 ms | 235.30 ms |
| New wording | 23.68 ms | 17.29 ms | 239.06 ms |

Start-up integrity: 8/8.

### Archived TEST627 record ([`627.log`](627.log))

Reproduced from the archived log; not re-run in the published runtime.

| Panel (48 questions) | Correct file | Correct answer | Errors |
|---|---|---|---|
| Original wording | 48/48 | 48/48 | 0 |
| New wording | 48/48 | 48/48 | 0 |

| Panel — median per question | Address | Load | Answer |
|---|---|---|---|
| Original wording | 23.87 ms | 17.87 ms | 236.61 ms |
| New wording | 25.05 ms | 17.50 ms | 235.20 ms |

Decision: `READY_FOR_DEMO_WITH_DISCLOSED_NONBLIND_LIMITATION`. 627.log SHA-256 `c909ce23d98bfc883641c82144adceacdaf7de13fc56e00a6153fbbb46c00687`.

### Live RUN `MAMSH-20261012-053712-0001-C00` (case 00)

- **Verdict:** `CASE_00_CORRECT_WRITE_REVERIFIED_SSD_READ_SOURCE_FREE_ANSWER`.
- **Seal:** 43/43 checks PASS · payload SHA-256 `f85289a8b93a2ad66b1912ea7fdb22a17b4782b12cc0e41abdd81af28a798048`.
  - The sealed payload lists **40** checks (`checks_pre_seal`). Three further sealing-stage checks run after the payload is assembled: claim scan of the payload, payload SHA-256 after writing, and SHA-256 of the copied engine result. Figure 4 shows the resulting total of **43** under the label “pre-seal”. Evidence-ZIP checks follow after sealing.
- **Question:** “Name the present capital of ZX624196.” · selected #0292, the expected cartridge.

| Stage (one run; scopes differ, not summed) | Measured |
|---|---|
| WRITE `init(fact, 3)` — two forwards | 105.9 ms |
| Question encode — one question-frame forward | 41.5 ms |
| Address — FAISS 0.85 + TOP2 20.93 + PEAK50 features/rerank 6.08 (TEST627 scope) | 27.9 ms |
| Load — SHA-256 + `torch.load` + copy to GPU (may be served from the OS page cache) | 19.4 ms |
| Answer — question template + 6 generation steps | 249 ms |
| Whole audited pipeline (`engine_total`: integrity, WRITE re-check, SSD stat, READ, control, weight guard) | 2.787 s |

All three panels above are separate measurements: they come from different runs and are not merged into one success count.


![Figure 4 — start-up acceptance, archived TEST627 record, live case 00 and experimental scope, shown as separate columns](docs/figures/fig_04_results.jpg)

*Figure 4 — RESULTS: runtime acceptance, archived TEST627, live run and experimental scope. The three columns are separate measurements and are not merged into one success count. [PDF version](docs/figures/fig_04_results.pdf).*

---

## How PEAK50 was locked

**TEST624** (`nprobe 2 · token_k 16 · budget 16`): the correct cartridge was among the 16 candidates for 48/48 questions, but the original FRAM ranking selected it for 40/48.

**TEST626** compared eight predeclared reranking rules on the same candidate sets:

| | Result |
|---|---|
| Calibration (24 questions) | FRAM 21/24; every PEAK/CONTRAST variant 24/24 |
| Rule locked by TEST626's own predeclared tie-break | PEAK25 |
| New-wording panel (48) | FRAM 40/48 · PEAK25 47/48 · PEAK50 48/48 |
| Disk answers with PEAK25 | 47/48 + 47/48 |

**TEST627** fixed PEAK50 after these results and confirmed 48/48 + 48/48 on the same 48 facts. TEST627's own limitation record states that it is a confirmation on previously exposed facts.

---

## Scientific scope and next validation steps

**What the record establishes.** On an A100, a frozen Qwen2.5-7B-Instruct:

- writes 1,453 facts as independent K/V cartridge files;
- reproduces a cartridge bit for bit on demand;
- addresses the pool from the question alone with a label-free native index;
- answers 96 questions from the selected files without the source text;
- leaves its weights unchanged throughout.

**Scope of the present panel.** The 96 questions are two wordings of the same 48 synthetic capital-of-country facts, and PEAK50 was fixed after TEST626. The panel is therefore a confirmation panel, not a blind holdout.

**Next validation targets** (not yet completed):

1. a preregistered blind holdout with new facts, new templates and new question wordings;
2. cold-SSD latency, with the page cache excluded;
3. consumer hardware;
4. pools of 10⁴–10⁶ cartridges, with touched-set growth measured directly;
5. other fact types and other models;
6. answers that combine several cartridges.

---

## Running the demonstration (Google Colab)

Runtime: **Runtime → Change runtime type → GPU A100 (40 GB)**, about 5 GB of free disk. Internet access is needed for the Hugging Face model and the locked engine chain.

**Three cells (recommended):**

1. **[`HIPPOCAMPUS_part1.py`](HIPPOCAMPUS_part1.py)** — engine. Verifies the embedded TEST627 engine and the 626.py / 624.py / 610.py chain, then executes the engine unchanged:
   - model load and TEST610 baseline;
   - WRITE of 1,453 files;
   - PCA128 + IVF128 index;
   - PEAK50 acceptance on 96 questions from disk;
   - audit of all files.

   Before the engine starts, a guide maps the engine's own progress lines to these stages. The measured start-up duration is printed at the end.
2. **[`HIPPOCAMPUS_part2.py`](HIPPOCAMPUS_part2.py)** — fail-closed RUN pipeline and figures.
3. **[`HIPPOCAMPUS_part3.py`](HIPPOCAMPUS_part3.py)** — GPU-free self-test (13 fail-closed branches), then a public Gradio link.

Run the three cells **in order, in the same runtime**. Do not restart the runtime between them.

**One cell:** [`HIPPOCAMPUS_full.py`](HIPPOCAMPUS_full.py) contains the three parts in order. Paste it into a single cell and run it.

The Gradio link printed at the end is a temporary share URL of that Colab session. It stops working when the session ends; it is not a permanent public address.

In the interface, press **RUN**. There is nothing to type or choose. RUN *n* uses case (*n* − 1) mod 48 of the engine's 48 new-wording questions; this rotation is predeclared. Each RUN goes through these stages: integrity → WRITE re-check → SSD → READ address → load + answer → control → verification → seal → figures → package.

---

## Outputs, figures and logs

Every RUN produces:

- **Four figures** (JPEG 300 dpi + vector PDF; the copies from the published run are in [`docs/figures/`](docs/figures/)):
  - `fig_01_write_ssd_read` — WRITE · SSD · READ architecture with the live values;
  - `fig_02_write_ssd` — formation and storage of one cartridge;
  - `fig_03_read` — address, PEAK50 candidates, load, answer, control;
  - `fig_04_results` — start-up acceptance, archived record and this run side by side, with limits.
- `fig_01_TR` — Turkish version of Fig. 1.
- The sealed payload (`*_FULL_RUN_LOG_payload.json`), the readable run log (`*_READABLE_RUN_LOG.txt`), the run manifest, a summary, raw forward records (`.jsonl`) and the engine's start-up result JSON.
- An evidence ZIP containing all of the above, checked after sealing.

The demo's verification panel recomputes on demand the full-weight SHA-256, the weight guard and the SHA-256 of all 1,453 cartridge files.

**Published run log:** [`Hippocampus.log`](Hippocampus.log). Related logs: [`627.log`](627.log) · [`626.log`](626.log) · [`624.log`](624.log).

---

## Source files and engine chain

| File | Role |
|---|---|
| [`HIPPOCAMPUS_part1.py`](HIPPOCAMPUS_part1.py) | engine cell: locks, embedded TEST627 demo engine, measurement adapter |
| [`HIPPOCAMPUS_part2.py`](HIPPOCAMPUS_part2.py) | RUN pipeline, sealing, figures |
| [`HIPPOCAMPUS_part3.py`](HIPPOCAMPUS_part3.py) | self-test and Gradio interface |
| [`HIPPOCAMPUS_full.py`](HIPPOCAMPUS_full.py) | single-cell version |
| [`627.py`](627.py) | TEST627: locked PEAK50 + physical CUT3 acceptance and storage audit |
| [`626.py`](626.py) | TEST626: candidate features and reranking rules |
| [`624.py`](624.py) | TEST624: 1,453 physical CUT3 files, PCA128 + FAISS IVF128 |
| [`610.py`](610.py) | TEST610: frozen Qwen CUT3 baseline (96 cartridges) |
| [`566.py`](566.py), [`575.py`](575.py) | CUT3 `checkpoint` / `init` / `append` / `answer`; verified by git blob |

---

## License

The software and materials in this release are governed by the **AKBASCORE RESEARCH SOFTWARE LICENSE** in [`LICENSE`](LICENSE). That file is the authoritative text.

In short, as stated there:

- Research, education, evaluation, modification and derivative works are permitted.
- Commercial use is permitted without a separate license where the user, together with its affiliates, has combined consolidated annual gross revenue below **USD 50,000,000** in its most recently completed fiscal year.
- At or above USD 50,000,000, commercial use of covered material requires a **separate commercial license** from the copyright holder.
- The threshold refers to annual gross revenue, not company valuation, investment raised, market capitalization or assets.
- This release does not change the terms of earlier releases.

Third-party models, libraries and dependencies, including Qwen2.5-7B-Instruct, PyTorch, Transformers, FAISS and Gradio, remain under their own licenses.

Commercial licensing inquiries: Mustafa Akbaş, via this repository.

---

## Citation

Earlier public records in this line, as listed in the published run log:

- AKBASCORE MAM v1.0 (Mistral-7B, DC6): DOI [10.5281/zenodo.23245358](https://doi.org/10.5281/zenodo.23245358)
- AKBASCORE MAM v1.1 (Qwen2.5-7B-Instruct, DC3): DOI [10.5281/zenodo.23257347](https://doi.org/10.5281/zenodo.23257347)
- AKBASCORE MAM v1.2.0 (Qwen2.5-7B CUT3 + FRAM): DOI [10.5281/zenodo.23282902](https://doi.org/10.5281/zenodo.23282902)

Citation template — add the DOI line once Zenodo has assigned it:

```bibtex
@techreport{akbas_2026_mam_silicon_hippocampus,
  author      = {Akbaş, Mustafa},
  title       = {{AKBASCORE MAM — The Silicon Hippocampus: Physical Key/Value Memory Cartridges Written, Stored and Recalled by a Frozen Qwen2.5-7B-Instruct}},
  institution = {AkbasCore AI Teknoloji},
  address     = {Mersin, Türkiye},
  type        = {Technical Research Report / Software Experimental Record},
  year        = {2026},
  month       = oct,
  url         = {https://github.com/ceceli33/titan-cognitive-core-v2}
}
```

Copyright © 2026 Mustafa Akbaş
