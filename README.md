# AKBASCORE MAM — THE SILICON HIPPOCAMPUS

### Model-Native Numerical Memory for Frozen Language Models

**Physical Memory Cartridges · WRITE · SSD · READ · Native CUT3 Injection · Append-Only Consolidation · Multi-Hop Relational Synthesis**

**Mustafa Akbaş**  
AkbasCore AI Teknoloji  
Mersin, Türkiye  
October 2026

**License:** [AKBASCORE RESEARCH SOFTWARE LICENSE](LICENSE)

---

## Overview

AKBASCORE MAM is an experimental model-native memory architecture for frozen language models.

The system investigates whether knowledge can be converted into numerical representations derived from a language model's internal computations, stored outside its learned parameters, and subsequently used during inference without reintroducing the original source text into the answering prompt.

The research currently demonstrates two complementary capabilities.

### 1. Physical Silicon Hippocampus — WRITE · SSD · READ

A frozen Qwen2.5-7B-Instruct model:

1. Reads source facts and creates independent numerical CUT3 memory cartridges.
2. Stores the cartridges as physical files with SHA-256 integrity manifests.
3. Locates relevant cartridges through native geometric memory addressing.
4. Loads selected numerical memories into the attention cache.
5. Answers questions without receiving the original source text again.

The published physical-memory experiment contains **1,453 independent cartridges** and reports **96/96 correct selections and answers** across two question-wording panels covering the same 48 synthetic facts.

### 2. Silicon Hippocampus v1.4 — Native Multi-Cartridge Consolidation

The new research extension investigates how multiple independently written numerical cartridges can be combined and used together.

It introduces an experimental pathway for:

- Independent numerical cartridge writing.
- Native append-only memory consolidation.
- Internal layer-3 residual representation injection.
- Attention K/V memory reconstruction.
- Preservation of previously written memory rows.
- Source-free question answering.
- Direct factual retrieval.
- Predefined multi-hop relational question sequences.
- Missing-cartridge evaluation.
- Counterfactual memory replacement.
- Numerical memory X-Ray visualization.
- DC_CUT3 versus TEXT_JOINT comparison.
- Scientific verification and reproducible research records.

The new demonstration includes two fictional experimental scenarios, nine memory conditions per scenario, two memory representations, and seven scientific JPEG figures.

**The v1.4 extension does not replace or invalidate the earlier physical-memory architecture. Both research lines remain part of AKBASCORE MAM.**

### Fundamental Principle

The model weights remain frozen.

No training, fine-tuning, LoRA, optimizer or learned-parameter modification is required for the demonstrated memory operations.

The system operates through numerical attention-memory representations.

The term **Silicon Hippocampus** describes the proposed computational role of external numerical memory. It does not imply biological equivalence to the human hippocampus.

---

# PART I — PHYSICAL SILICON HIPPOCAMPUS

## 1. Architecture — WRITE · SSD · READ

The original physical-memory architecture consists of three stages.

```text
                    AKBASCORE MAM
                 THE SILICON HIPPOCAMPUS

                         WRITE
                           |
                           v
                      SOURCE FACT
                           |
                           v
                 FROZEN QWEN2.5-7B
                           |
                           v
                  NATIVE CUT3 MEMORY
                           |
                           v
                 28 LAYERS OF K/V
                           |
                           v
                 PHYSICAL CARTRIDGE
                           |
                           v
                         SSD
                           |
                           v
                 1,453 MEMORY FILES
                           |
                           |
                           v
                         READ
                           |
                           v
                       QUESTION
                           |
                           v
                 NATIVE LAYER-3 QUERY
                           |
                           v
                        PCA128
                           |
                           v
                    FAISS IVF128
                           |
                           v
                    TOP2 CANDIDATES
                           |
                           v
                         PEAK50
                           |
                           v
                  SELECTED CARTRIDGE
                           |
                           v
                 SHA-256 VERIFICATION
                           |
                           v
                 ATTENTION CACHE LOAD
                           |
                           v
                  FROZEN MODEL ANSWER
```

The source text is used during cartridge construction.

During READ, the selected cartridge supplies numerical attention memory, while the answering input contains the question template.

### Model Configuration

| Property | Configuration |
|---|---|
| Model | Qwen/Qwen2.5-7B-Instruct |
| Transformer layers | 28 |
| Hidden dimension | 3,584 |
| Query attention heads | 28 |
| Key/value attention heads | 4 |
| Attention head dimension | 128 |
| Precision | bfloat16 |
| Attention implementation | SDPA |
| Model parameters | 7,615,616,512 |
| Fine-tuning | None |
| LoRA | None |
| Optimizer | None |
| Weight modification | None |

Full-weight SHA-256:

`b441b005d826d45f2778291696ac5ae22dfb3421defb37c539904d8d38f93786`

---

## 2. WRITE — Creating Numerical Memory Cartridges

Each source fact is placed after a fixed 29-token instruction prefix.

The native CUT3 engine constructs the numerical memory through two forward passes.

### Forward Pass 1 — Source Processing

The frozen model processes the instruction prefix and the source fact.

The system captures:

- Layer-3 hidden states, H3.
- Attention key tensors from layers 0–3.
- Attention value tensors from layers 0–3.

The original model parameters remain unchanged.

### Forward Pass 2 — Numerical Memory Construction

The engine performs a second forward pass using zero-valued input embeddings.

The stored H3 representations are introduced at the layer-3 boundary.

The model then computes the higher-layer attention states.

The resulting cartridge contains:

- 28 transformer layers.
- 28 key tensors.
- 28 value tensors.
- 56 numerical tensors in total.
- bfloat16 precision.

Each tensor has the shape:

```text
[1, 4, T, 128]
```

Where `T` represents the number of memory rows.

The numerical payload requires:

```text
28 layers × 2 tensors × 4 heads × 128 dimensions × 2 bytes

= 57,344 bytes per memory row
```

### Published Case-00 Cartridge

| Measurement | Value |
|---|---:|
| Instruction prefix | 29 tokens |
| Fact | 19 tokens |
| Total memory rows | 48 |
| Numerical payload | 2,752,512 bytes |
| Physical file size | 2,768,949 bytes |
| File-container overhead | 16,437 bytes |

The cartridge is saved as:

`cut3_00292.pt`

The physical file contains numerical tensors.

Its SHA-256 is recorded in the cartridge manifest.

### Live WRITE Verification

The original demonstration re-executes the cartridge-writing operation.

It verifies:

1. The expected two forward passes.
2. The expected input structure.
3. The cartridge tensor shapes.
4. The numerical byte count.
5. Bitwise equality against the stored cartridge.
6. The absence of the tested source strings from the serialized file bytes.

In the published case-00 run, all 56 tensors were bitwise identical to the stored cartridge.

If the comparison fails, the engine performs a second live reconstruction to distinguish possible execution nondeterminism from a stored-memory mismatch.

An unresolved mismatch prevents successful sealing.

---

## 3. SSD — Physical Memory Storage

The published physical-memory experiment contains:

**1,453 independent numerical memory cartridges.**

The pool consists of:

- 96 original cartridges.
- 1,357 additional synthetic-fact cartridges.

### Measured Storage

| Property | Result |
|---|---:|
| Physical cartridge files | 1,453 |
| Total disk storage | 3,973,450,961 bytes |
| Numerical tensor payload | 3,949,568,000 bytes |
| Minimum file size | 2,252,853 bytes |
| Median file size | 2,768,949 bytes |
| Maximum file size | 2,883,637 bytes |
| Median write time | 83.5 ms |
| Summed writing and storage time | 121.6 seconds |

Each cartridge is stored independently.

The system uses SHA-256 manifests to verify file identity and integrity.

At start-up, all 1,453 physical files are checked against the manifest.

### Lossless Compression Experiment

A predeclared sample of 32 files was tested using gzip compression.

Measured compression ratio:

`0.796091`

The compressed files were restored bitwise.

This ratio applies to the measured sample.

It is not a direct measurement of compression across the complete 1,453-file pool.

---

## 4. READ — Native Geometric Memory Addressing

The READ system identifies a relevant physical memory cartridge from the question.

The source fact is not inserted into the answering prompt.

The frozen model's own layer-3 representations are used to construct the memory address.

### 4.1 Question Encoding

The question is processed using the established instruction frame.

The model performs one forward pass.

Layer-3 residual representations from the question span are extracted.

These representations are normalized in the same frame used for the cartridge address keys.

The published case-00 experiment used:

- 42 input tokens.
- 13 question positions.

### 4.2 PCA128

The native representations are projected into a 128-dimensional space.

The PCA basis is fitted using:

`2,048 sampled native positions`

with a fixed random seed.

The projection is:

```text
(q - CENTER) @ BASIS
```

### 4.3 FAISS IVF128

The memory-addressing index uses:

`FAISS IndexIVFFlat`

with inner-product similarity.

| Parameter | Value |
|---|---:|
| Native indexed positions | 26,738 |
| Physical cartridges | 1,453 |
| PCA dimensions | 128 |
| IVF lists | 128 |
| nprobe | 2 |
| token_k | 16 |
| Candidate budget | 16 |

The question representations identify candidate cartridge positions.

The original case-00 experiment touched 103 of the 1,453 cartridges.

### 4.4 TOP2 Candidate Ranking

For each question position, the two strongest matching positions inside a cartridge are evaluated.

The score is aggregated across question positions.

The touched cartridges are ranked.

The top 16 become candidates for final reranking.

### 4.5 PEAK50 Selection

The final selection uses the locked PEAK50 rule.

Let:

```text
s_tj = q_t · k_j
```

The FRAM score is:

```text
FRAM(c) =
mean_t [
    (logsumexp_j(beta * s_tj) - log(n_c)) / beta
]
```

The peak score is:

```text
peak(c) =
mean_t max_j(s_tj)
```

The combined rule is:

```text
PEAK50(c) =
0.5 * FRAM(c) + 0.5 * peak(c)

beta = 8
```

Candidates are ranked in descending order.

Ties are broken by cartridge ID.

The highest-ranked cartridge is selected.

### 4.6 Physical Memory Loading

The selected file is verified using SHA-256.

The numerical tensors are loaded into GPU memory.

The model subsequently answers using the loaded attention memory.

The original source text is not provided again.

The answering template is:

```text
QUESTION:
{question}

ANSWER:
```

Generation uses greedy decoding with a limit of 16 new tokens.

### Case-00 Result

Question:

```text
Name the present capital of ZX624196.
```

Selected cartridge:

`#0292`

Raw answer:

`Fgvrkerflc`

Expected answer:

`Fgvrkerflc`

Result:

**Correct cartridge selected. Correct answer generated.**

The runner-up cartridge was also tested as a control.

It did not produce the expected answer.

---

## 5. Physical Memory — Published Results

The published experiment is documented in:

[Hippocampus.log](Hippocampus.log)

### Start-Up Acceptance

| Evaluation | Correct cartridge | Correct answer | Errors |
|---|---:|---:|---:|
| Original question wording | 48/48 | 48/48 | 0 |
| New question wording | 48/48 | 48/48 | 0 |

Combined reported result:

**96/96 correct cartridge selections and 96/96 correct answers.**

These are two wordings of the same 48 synthetic facts.

They are not 96 independent unseen facts.

### Start-Up Latency

Median per question:

| Panel | Address | Load | Answer |
|---|---:|---:|---:|
| Original wording | 23.80 ms | 17.29 ms | 235.30 ms |
| New wording | 23.68 ms | 17.29 ms | 239.06 ms |

### Archived TEST627 Results

The earlier TEST627 record reported:

| Panel | Address | Load | Answer |
|---|---:|---:|---:|
| Original wording | 23.87 ms | 17.87 ms | 236.61 ms |
| New wording | 25.05 ms | 17.50 ms | 235.20 ms |

The archived TEST627 measurements were not rerun as part of the published case-00 experiment.

### Live Case-00 Measurements

Run:

`MAMSH-20261012-053712-0001-C00`

| Stage | Measured time |
|---|---:|
| WRITE | 105.9 ms |
| Question encoding | 41.5 ms |
| Native addressing | 27.9 ms |
| Cartridge loading | 19.4 ms |
| Answer generation | 249 ms |
| Complete audited pipeline | 2.787 seconds |

The complete pipeline includes additional verification and control operations.

It should not be interpreted as ordinary inference throughput.

### Sealing and Verification

Published result:

**43/43 sealing checks PASS.**

The sealed payload contains 40 checks.

Three additional sealing-stage checks produce the final total of 43.

Payload SHA-256:

`f85289a8b93a2ad66b1912ea7fdb22a17b4782b12cc0e41abdd81af28a798048`

---

## 6. How PEAK50 Was Established

The memory-addressing research progressed through several experimental stages.

### TEST624

Configuration:

```text
nprobe = 2
token_k = 16
candidate budget = 16
```

The correct cartridge appeared among the 16 candidates for:

`48/48 questions`

However, the original FRAM ranking selected the correct cartridge for:

`40/48 questions`

### TEST626

Eight predeclared reranking rules were compared.

| Evaluation | Result |
|---|---|
| Calibration panel | 24 questions |
| FRAM calibration | 21/24 |
| PEAK/CONTRAST variants | 24/24 |
| Rule selected by original tie-break | PEAK25 |
| New-wording FRAM | 40/48 |
| New-wording PEAK25 | 47/48 |
| New-wording PEAK50 | 48/48 |
| Disk answers with PEAK25 | 47/48 + 47/48 |

### TEST627

PEAK50 was subsequently fixed.

The experiment confirmed:

`48/48 + 48/48`

on the same 48 facts.

The archived record explicitly identifies this as a confirmation on previously exposed facts rather than a blind holdout.

The later Silicon Hippocampus v1.4 experiments investigate a separate capability.

They do not replace the PEAK50 research.

---

## 7. Physical Memory Integrity

The original system includes multiple verification layers.

### Start-Up Verification

- Source SHA-256 locks.
- Cartridge manifest verification.
- All 1,453 physical file checks.
- Frozen model verification.
- Full-weight SHA-256.
- Sentinel checks.
- Zero trainable parameters.
- Expected tensor structures.
- Native addressing configuration.
- Locked PEAK50 selection.

### Runtime Verification

- Bitwise cartridge reconstruction.
- Model forward-pass auditing.
- Source-text exclusion checks.
- Candidate-set integrity.
- Score reconstruction.
- Selected-file SHA-256.
- Answering-input verification.
- Model-weight guards.
- Decoder-hook checks.
- Experimental output sealing.

### Fail-Closed Conditions

The original demonstration includes 13 tested failure branches:

1. Modified model-weight guard.
2. Nonidentical cartridge reconstruction.
3. Unexpected forward passes.
4. Source text detected in cartridge bytes.
5. Cartridge manifest size mismatch.
6. Source tokens entering the question frame.
7. Incorrect PEAK50 score reconstruction.
8. Candidate-set inconsistency.
9. Live READ mismatch.
10. Cartridge SHA-256 mismatch.
11. Source tokens entering the answering input.
12. Residual hooks remaining on decoder layers.
13. Incomplete engine initialization.

The GPU-free self-test exercises these branches before the original Gradio interface starts.

---

## 8. Physical Memory — Storage, RAM and GPU

The original experiment measured the following resource requirements.

| Location | Component | Size |
|---|---|---:|
| SSD | 1,453 physical cartridges | 3,973,450,961 bytes |
| RAM | Serialized FAISS IVF128 index | 13,970,459 bytes |
| RAM | PCA128 basis | 1,835,008 bytes |
| RAM | Native layer-3 reranking rows | 383,315,968 bytes |
| GPU | Frozen model and loaded cartridge | 18.27 GiB peak allocation |

The GPU value comes from the published case-00 run.

These quantities refer to different resource categories and should not be added together as if they represented a single memory allocation.

---

# PART II — SILICON HIPPOCAMPUS v1.4

## 9. Native Internal Tensor Memory Injection & Multi-Hop Relational Synthesis

**Research extension: 14 October 2026**

Silicon Hippocampus v1.4 extends the earlier AKBASCORE MAM research from selecting and loading individual physical cartridges to experimentally consolidating multiple independently written numerical memories.

The central research question is:

**Can a frozen language model use multiple independently written numerical memory cartridges to answer questions requiring information distributed across those memories?**

The experiment investigates this through a native numerical-memory pathway.

The underlying model remains frozen.

The established engine is preserved.

The new demonstrator adds experimental controls, measurements, visualization and an interactive research interface.

### Public Implementation

**Three-part Google Colab version:**

1. [HIPPOCAMPUS_1.4_part1.py](HIPPOCAMPUS_1.4_part1.py)
2. [HIPPOCAMPUS_1.4_part2.py](HIPPOCAMPUS_1.4_part2.py)
3. [HIPPOCAMPUS_1.4_part3.py](HIPPOCAMPUS_1.4_part3.py)

**Complete single-file version:**

[HIPPOCAMPUS_1.4_full.py](HIPPOCAMPUS_1.4_full.py)

**Published experimental log:**

[HIPPOCAMPUS_1.4_log](HIPPOCAMPUS_1.4_log)

---

## 10. v1.4 Architecture

The new experimental pathway is:

```text
                   INDEPENDENT SOURCE FACTS

                    FACT A    FACT B    FACT C
                       |         |         |
                       v         v         v
                   FROZEN QWEN2.5-7B-INSTRUCT
                       |         |         |
                       v         v         v
                  CARTRIDGE  CARTRIDGE  CARTRIDGE
                      A          B          C
                       |         |         |
                       +---------+---------+
                                 |
                                 v
                      NATIVE CONSOLIDATION
                                 |
                                 v
                       CONSOLIDATED MEMORY
                                 |
                                 v
                         DC_CUT3 INJECTION
                                 |
                                 v
                       FROZEN TRANSFORMER
                                 |
                                 v
                         QUESTION ANSWERING
                                 |
                     +-----------+-----------+
                     |                       |
                     v                       v
                DIRECT ANSWERS         STAGED RELATIONAL
                                           QUESTIONS
```

The model weights are not modified.

The system operates through the numerical representations associated with the transformer's attention computation.

### Two Experimental Phases

**Phase 1 — Memory Construction**

Each source fact is processed.

The engine creates independent numerical cartridges.

The cartridges are consolidated into a combined attention memory.

**Phase 2 — Memory Use**

The model receives question templates.

The source facts are not reintroduced as ordinary prompt text.

The model answers using the consolidated numerical memory.

Rebuilding the memory requires processing the source facts again.

The phrase "read once" therefore applies to the construction of a particular memory build, not to every possible future reconstruction.

---

## 11. Independent Numerical Cartridge Writing

Each cartridge is created from an independent source fact.

The system captures:

- Layer-3 residual representations.
- Attention K/V tensors from layers 0–3.
- Numerical representations needed for consolidation.

The first cartridge includes the established instruction prefix.

Subsequent cartridges contribute their fact-specific memory rows.

The original source text is not retained as an ordinary text document inside the numerical attention-memory payload.

### Numerical Structure

Model:

`Qwen/Qwen2.5-7B-Instruct`

Hidden dimension:

`3584`

Attention K/V tensor structure:

```text
[1, 4, T, 128]
```

Precision:

`bfloat16`

The model contains 28 transformer layers.

The native CUT3 boundary is at layer 3.

---

## 12. Native Append-Only Consolidation

The new experimental engine combines independently written cartridges into one numerical memory.

The first cartridge initializes the memory.

Additional cartridges are appended.

```text
CARTRIDGE A
     |
     v
MEMORY A
     |
     + CARTRIDGE B
     |
     v
MEMORY AB
     |
     + CARTRIDGE C
     |
     v
MEMORY ABC
```

During append operations, the engine computes higher-layer attention states for the newly added memory rows.

Previously written rows remain part of the memory.

The experiment explicitly checks whether those earlier rows remain unchanged.

It also compares the stepwise memory with the memory generated by the original native engine pathway.

### 12.1 AKBASCORE Memory Measurements

Condition:

`ABC · DC_CUT3`

| Cartridge | Source tokens | Added rows | Numerical bytes | Operation |
|---|---:|---:|---:|---|
| A | 24 | 53 | 814,080 | Initialize |
| B | 27 | 27 | 414,720 | Append |
| C | 29 | 29 | 445,440 | Append |

Final consolidated memory:

**109 rows**

**6,250,496 bytes**

Approximately:

**5.96 MiB**

Measured operations:

| Operation | Time |
|---|---:|
| Initialize A | 1,296.1 ms |
| Append B | 177.0 ms |
| Append C | 91.6 ms |

Memory SHA-256:

`9833210806512527df5e549ec16963520c0889c30e87d710323c2fede759816f`

### 12.2 MATRIX Memory Measurements

Condition:

`ABC · DC_CUT3`

| Cartridge | Source tokens | Added rows | Numerical bytes | Operation |
|---|---:|---:|---:|---|
| A | 25 | 54 | 829,440 | Initialize |
| B | 36 | 36 | 552,960 | Append |
| C | 25 | 25 | 384,000 | Append |

Final consolidated memory:

**115 rows**

**6,594,560 bytes**

Approximately:

**6.29 MiB**

Measured operations:

| Operation | Time |
|---|---:|
| Initialize A | 74.7 ms |
| Append B | 90.7 ms |
| Append C | 88.8 ms |

Memory SHA-256:

`d6feb1d33a850332537f647f10e03c4af87615a6d6b0411189ab4e03d164c83e`

### 12.3 Consolidation Verification

The published v1.4 run reports:

| Verification | AKBASCORE | MATRIX |
|---|---|---|
| Earlier rows unchanged | PASS | PASS |
| Layer 0–3 K/V matches cartridge | PASS | PASS |
| Stepwise memory equals native engine memory | PASS | PASS |
| Frozen model weights | PASS | PASS |
| Sentinel unchanged | PASS | PASS |

These results establish the reported numerical consistency of the tested consolidation operations.

They do not establish that every possible memory combination will behave correctly.

---

## 13. Native DC_CUT3 Injection

The consolidated numerical memory is incorporated into the frozen model through the native DC_CUT3 pathway.

The procedure uses:

1. Stored layer-3 residual representations.
2. Numerical attention K/V tensors.
3. Native layer-3 injection.
4. Higher-layer memory computation.
5. Frozen transformer inference.

The original model parameters remain unchanged.

The memory is numerical rather than a retrieved text passage pasted into the answering prompt.

The engine uses the transformer's own attention-memory structures.

This distinction is important.

AKBASCORE MAM does not eliminate attention caches.

It investigates how independently constructed numerical memory can be maintained and introduced into the frozen model's computation.

---

## 14. Demonstration A — Mustafa Akbaş / AKBASCORE

The first v1.4 demonstration uses a fictional knowledge domain.

Three independent cartridges establish a relational sequence.

### Cartridge A — Activation

```text
In the interplanetary data network, Mustafa Akbaş
successfully activates the AKBASCORE MAM architecture.
```

### Cartridge B — Conversion

```text
The AKBASCORE MAM architecture converts dibekgoz
crystal resonance into Morvex energy through
layer-3 tensor injection.
```

### Cartridge C — Effect

```text
When Morvex energy combines in layer-3 tensors,
it causes reversal of time flow in the target
system without modifying its frozen neural weights.
```

### Relational Structure

```text
Mustafa Akbaş
      |
      v
AKBASCORE MAM
      |
      v
Dibekgoz crystal resonance
      |
      v
Morvex energy
      |
      v
Reversal of time flow
```

This is a fictional scenario.

It is not a claim about real-world physics.

### 14.1 Direct Question Results

Condition:

`ABC · DC_CUT3`

| Category | Raw answer | Result |
|---|---|---|
| Activation | AKBASCORE MAM | Correct |
| Conversion | Morvex | Correct |
| Effect | reversal of time flow | Correct |
| Final | reversal of time flow | Correct |

### 14.2 Staged Three-Step Relational Chain

The experiment uses three predefined questions.

Each subsequent question incorporates the preceding model answer.

Observed sequence:

```text
HOP 1
AKBASCORE MAM

HOP 2
Morvex

HOP 3
reversal of time flow
```

Result:

**3/3 expected answers.**

This is an externally specified question sequence.

It is not autonomous planning.

### 14.3 Counterfactual Memory

The experiment substitutes an alternative final effect.

Original:

`reversal of time flow`

Counterfactual:

`temporal stasis`

| Condition | Final answer |
|---|---|
| ABC | reversal of time flow |
| ABC_CF | temporal stasis |

The model produced the expected changed answer.

### 14.4 Free-Form Synthesis

The experiment also requested a longer explanatory answer.

The raw output was:

```text
Mustafa Akbaş produces the final effect in the target
system by converting dib
```

The response reached the 16-token generation limit.

It was therefore classified as:

`UNSCORED`

The published demonstration does not claim successful unrestricted long-form synthesis.

---

## 15. Demonstration B — The Matrix / Neo

The second demonstration uses a fictional scenario inspired by The Matrix.

Three independent cartridges establish the relationship between a cartridge upload, stored knowledge and a resulting ability.

### Cartridge A — Upload

```text
Inside the Matrix, Tank presses the AKBASCORE neural
cartridge injection button and uploads cartridge
MX-71 into Neo.
```

### Cartridge B — Knowledge

```text
Cartridge MX-71 contains Kung Fu knowledge, which
the AKBASCORE MAM interface injects into Neo
through layer-3 tensors without changing the
model weights.
```

### Cartridge C — Ability

```text
When Neo uses the Kung Fu knowledge from cartridge
MX-71 inside the Matrix, Neo gains bullet-dodge
ability.
```

### Relational Structure

```text
Tank
  |
  v
Cartridge MX-71
  |
  v
Kung Fu knowledge
  |
  v
Neo
  |
  v
Bullet-dodge ability
```

### 15.1 Direct Question Results

Condition:

`ABC · DC_CUT3`

| Category | Raw answer | Result |
|---|---|---|
| Activation | Tank | Correct |
| Content | Kung Fu knowledge | Correct |
| Injection layer | 3 | Correct |
| Effect | bullet-dodge ability | Correct |
| Final | bullet-dodge ability | Correct |

### 15.2 Staged Three-Step Relational Chain

Observed sequence:

```text
HOP 1
MX-71

HOP 2
Kung Fu knowledge

HOP 3
bullet-dodge ability
```

Result:

**3/3 expected answers.**

### 15.3 Counterfactual Memory

Original:

`bullet-dodge ability`

Counterfactual:

`slow-motion perception`

| Condition | Final answer |
|---|---|
| ABC | bullet-dodge ability |
| ABC_CF | slow-motion perception |

The model produced the expected changed answer.

### 15.4 Free-Form Synthesis

The raw synthesis output was:

```text
Tank uploads cartridge MX-71 into Neo. The AKBASCORE M
```

The output reached the 16-token limit.

It was classified as:

`UNSCORED`

### 15.5 Information Overlap

Cartridge C explicitly mentions MX-71 and Kung Fu knowledge.

Consequently, some relational answers remain available even when cartridge B is absent.

This is an important limitation.

The demonstration does not establish complete independence between all cartridge contents.

---

## 16. Missing-Cartridge and Counterfactual Controls

The v1.4 demonstration evaluates nine memory conditions.

| Condition | Available memory |
|---|---|
| NONE | No factual cartridge |
| A | Cartridge A |
| B | Cartridge B |
| C | Cartridge C |
| AB | Cartridges A and B |
| AC | Cartridges A and C |
| BC | Cartridges B and C |
| ABC | Complete memory |
| ABC_CF | Complete memory with counterfactual C |

These conditions are evaluated using two pathways:

**DC_CUT3**

Native numerical-memory consolidation.

**TEXT_JOINT**

Text-based comparison control.

The experiments evaluate:

- Direct factual answers.
- Correct abstention.
- Unsupported answers.
- Missing information.
- Complete relational chains.
- Counterfactual final answers.

### 16.1 Complete Memory

Under ABC, both demonstrations produced the expected direct answers and staged three-step chains.

Under ABC_CF, both demonstrations produced the expected counterfactual final answers.

### 16.2 Missing Memory

The NONE condition produced the expected abstentions for the evaluated factual questions.

However, incomplete-memory conditions did not always produce correct abstention.

Examples include:

- AKBASCORE B produced answers unsupported by the available cartridge set.
- AKBASCORE BC completed the relational chain despite missing cartridge A.
- MATRIX AC completed the relational chain despite missing cartridge B.
- MATRIX BC produced a chain whose final result was unsupported under the evaluation rule.

These outcomes are preserved in the published log.

The system does not currently demonstrate perfect missing-information detection.

### 16.3 Audit Categories

The demonstration distinguishes:

| Status | Meaning |
|---|---|
| CORRECT_KNOWN | Expected answer produced from available information |
| CORRECT_ABSTENTION | Appropriate UNKNOWN response |
| UNSUPPORTED | Answer produced despite missing required information |
| MISSED | Available answer not produced |
| WRONG | Different incorrect answer |
| UNSCORED | Not evaluated for correctness |

The audit categories are applied after generation.

Expected answers are evaluation metadata, not intended as generation-time answers.

---

## 17. DC_CUT3 Versus TEXT_JOINT

The new demonstration compares numerical-memory consolidation against a text-based control.

The published Figure 6 reports:

**93/99 identical raw answers between DC_CUT3 and TEXT_JOINT.**

The audit-status totals were also identical between the two methods.

This result is scientifically important.

It demonstrates substantial behavioral agreement between the two pathways on the evaluated panel.

It does not establish that DC_CUT3 produces superior answers.

The experiment instead documents that:

1. The native numerical-memory pathway executes.
2. The numerical memory has a distinct internal representation.
3. The frozen model can answer questions using that representation.
4. The tested behavioral outputs largely agree with the text-based control.
5. Missing-memory behavior remains imperfect.

The numerical and behavioral results must be interpreted separately.

---

## 18. Internal Neural Memory X-Ray

The v1.4 demonstrator includes an internal numerical memory X-Ray.

The X-Ray visualizes actual K/V tensors from the executed experiment.

For the AKBASCORE ABC condition:

| Property | Value |
|---|---:|
| Transformer layers | 28 |
| Memory rows | 109 |
| K/V heads | 4 |
| Head dimension | 128 |
| Precision | bfloat16 |
| K/V memory | 6,250,496 bytes |
| Approximate size | 5.96 MiB |

### Key Heatmap

The first heatmap displays the layerwise structure of attention key representations.

### Value Heatmap

The second heatmap displays the layerwise structure of attention value representations.

The visualization uses per-row vector norms and layerwise display normalization.

### Cartridge Boundaries

The X-Ray distinguishes:

- Instruction prefix.
- Cartridge A.
- Cartridge B.
- Cartridge C.
- Layer-3 injection boundary.
- Lower-layer stored representations.
- Higher-layer consolidated representations.

### Numerical Comparison

A separate graph compares DC_CUT3 memory with TEXT_JOINT memory across transformer layers.

The plotted differences are measured from numerical tensors.

They are not randomly generated illustrative values.

However, numerical representation differences do not independently establish superior reasoning performance.

---

## 19. Seven-Figure Scientific Demonstration

The new v1.4 demonstrator produces seven scientific JPEG figures.

The figures are designed for high-resolution academic presentation.

They document the architecture, measured numerical memory, direct answers, relational chains, experimental controls and verification.

### Figure 1 — Silicon Hippocampus Architecture

**From Source Fact to Source-Free Answer**

The figure documents:

- Independent cartridge writing.
- Numerical memory representation.
- Append-only consolidation.
- Native CUT3 injection.
- Frozen transformer computation.
- Source-free question answering.

### Figure 2 — Cartridge Creation and Consolidation

**Measured Numerical Memory Construction**

The figure documents:

- Independent A/B/C cartridge creation.
- Source token counts.
- Memory row counts.
- Numerical payload sizes.
- Initialization and append times.
- Earlier-row preservation.
- Native engine equivalence.

### Figure 3 — Internal Neural Memory X-Ray

**Layerwise Numerical Memory Structure**

The figure documents:

- Real K/V tensor heatmaps.
- All 28 transformer layers.
- Cartridge boundaries.
- Layer-3 injection boundary.
- Memory size.
- Numerical comparison with TEXT_JOINT.

### Figure 4 — Mustafa Akbaş Experiment

**Three Cartridges and a Relational Chain**

The figure documents:

- Complete cartridge texts.
- Direct questions.
- Raw model answers.
- Audit results.
- Three-stage relational sequence.
- Generation-limit disclosure.

### Figure 5 — The Matrix / Neo Experiment

**Numerical Memory and Frozen Model Answers**

The figure documents:

- Cartridge upload.
- Kung Fu knowledge.
- Layer-3 injection.
- Direct questions.
- Raw model answers.
- Three-stage relational sequence.
- Known information overlap.

### Figure 6 — Causal Controls

**Missing Memory, Counterfactuals and Multi-Hop Evaluation**

The figure documents:

- Nine memory conditions.
- Direct answers.
- Missing-information behavior.
- Unsupported answers.
- Counterfactual replacements.
- Staged relational chains.
- DC_CUT3 versus TEXT_JOINT.
- Separate historical TEST677 results.

### Figure 7 — Verification and Research Record

**Model Integrity and Experimental Provenance**

The figure documents:

- Model configuration.
- GPU and software environment.
- Source SHA-256.
- Model-weight integrity.
- Memory integrity.
- Runtime measurements.
- Experimental limitations.
- Research references.

### Research Output Package

The demonstrator provides research outputs including:

- Seven scientific JPEG figures.
- Raw model answers.
- Experimental measurements.
- Numerical memory metadata.
- Integrity records.
- SHA-256 information.
- Experimental logs.
- A consolidated evidence ZIP.

For academic publication, the original high-resolution JPEG files should be used rather than screenshots.

The public experimental log is available here:

[HIPPOCAMPUS_1.4_log](HIPPOCAMPUS_1.4_log)

---

## 20. Interactive Gradio Research Laboratory

Silicon Hippocampus v1.4 includes an interactive Gradio demonstration.

The interface provides access to the experimental scenarios and memory operations.

### Experimental Scenarios

**Demonstration A**

Mustafa Akbaş / AKBASCORE.

**Demonstration B**

The Matrix / Neo.

### Interface Capabilities

The application includes:

1. Scenario selection.
2. Cartridge text inspection.
3. Cartridge text editing.
4. Scenario reset.
5. Memory-condition selection.
6. Native DC_CUT3 execution.
7. TEXT_JOINT comparison.
8. Direct question answering.
9. Staged relational question sequences.
10. Missing-cartridge controls.
11. Counterfactual evaluation.
12. Numerical memory measurements.
13. Internal K/V tensor visualization.
14. Scientific JPEG generation.
15. Experimental log inspection.
16. Downloadable research records.

The interface is an experimental research tool.

It is not presented as a production-ready general-purpose artificial intelligence system.

### Reproducibility

The application is available as both a three-part implementation and a complete single-file implementation.

#### Three-Part Google Colab Version

Run these files in order within the same runtime:

**Part 1 — Engine and Initialization**

[HIPPOCAMPUS_1.4_part1.py](HIPPOCAMPUS_1.4_part1.py)

**Part 2 — Read-Only Measurement and Experimental Evaluation**

[HIPPOCAMPUS_1.4_part2.py](HIPPOCAMPUS_1.4_part2.py)

**Part 3 — Seven Scientific Figures and Gradio Interface**

[HIPPOCAMPUS_1.4_part3.py](HIPPOCAMPUS_1.4_part3.py)

#### Complete Single-File Version

[HIPPOCAMPUS_1.4_full.py](HIPPOCAMPUS_1.4_full.py)

The full version contains the complete demonstration.

#### Published Experimental Log

[HIPPOCAMPUS_1.4_log](HIPPOCAMPUS_1.4_log)

The published log records the executed A100 experiment.

The temporary Gradio share link depends on the active Colab session.

It is not a permanent hosted service.

---

## 21. v1.4 — Measured Experimental Environment

Published run:

`MAM14-20261014-220212-0001`

Execution date:

`14 October 2026`

GPU:

`NVIDIA A100-SXM4-40GB`

### Environment

| Component | Configuration |
|---|---|
| Model | Qwen/Qwen2.5-7B-Instruct |
| Precision | bfloat16 |
| Attention | SDPA |
| GPU | NVIDIA A100-SXM4-40GB |
| PyTorch | 2.11.0+cu130 |
| Transformers | 5.18.0 |
| Gradio | 6.29.0 |
| Memory mechanism | Native DC_CUT3 |
| Model weights | Frozen |
| LoRA | Disabled |
| Trainable parameters | 0 |

Measured demonstration duration:

**57.6 seconds**

### Model Integrity

Full-weight SHA-256:

`b441b005d826d45f2778291696ac5ae22dfb3421defb37c539904d8d38f93786`

Equal to the TEST575 reference:

`True`

Sentinel unchanged:

`True`

All-parameter guard unchanged:

`True`

All runs frozen:

`True`

### Source Integrity

Embedded v1.4 engine SHA-256:

`fe6ddcc0d61e70f5590a0ce696cb0e4f561f901e79ceafc703d8138598be188b`

TEST664 source SHA-256:

`a6dcd8142446ab82fe7996bd512d59fc34525e6a092a68f069dad07a2a286eb5`

TEST664 panel SHA-256:

`e0686148d6fd88050a097ee5441ce99a4bb068a463408742bc476c83aaba6c05`

TEST566 git blob:

`b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71`

These identifiers document the source chain used by the experiment.

Hash verification establishes file identity and integrity.

It does not independently prove the semantic correctness of model-generated answers.

---

# PART III — RESEARCH COMPARISON AND SCIENTIFIC SCOPE

## 22. Why AKBASCORE MAM Is Not Conventional Text RAG

The original physical-memory architecture and the new consolidation experiments differ from conventional text retrieval followed by prompt augmentation.

| Property | Conventional Text RAG | AKBASCORE MAM |
|---|---|---|
| Stored information | Text passages and embeddings | Numerical attention-memory cartridges |
| Representation | Text and retrieval vectors | Frozen model's own internal numerical states |
| Retrieval | Text or embedding similarity | Native geometric addressing in the physical-memory system |
| Answer-time input | Retrieved text inserted into prompt | Numerical attention memory and question template |
| Encoder | Often a separate embedding model | The same frozen language model |
| Training | Optional | None in the demonstrated memory operations |
| Physical storage | Text/vector database | Independent numerical cartridge files |
| Multi-record use | Retrieved text passages | Experimental numerical cartridge consolidation |
| Internal inspection | Retrieved context | Layerwise K/V tensor measurements |

The system still uses transformer attention-memory structures.

The distinction concerns how the memory is constructed, represented, stored, selected, consolidated and used.

---

## 23. Two Complementary Research Capabilities

The physical-memory architecture and the v1.4 consolidation architecture address different problems.

| Capability | Physical Silicon Hippocampus | Silicon Hippocampus v1.4 |
|---|---|---|
| Primary objective | Retrieve a relevant physical cartridge | Consolidate multiple numerical cartridges |
| Memory organization | Independent SSD files | Combined numerical attention memory |
| Published cartridge count | 1,453 | Three per demonstration |
| Address mechanism | PCA128 + FAISS IVF128 + PEAK50 | Experimental memory-condition selection |
| Main evaluation | Cartridge selection and factual answering | Relational answering and memory controls |
| Question structure | Direct factual questions | Direct and staged relational questions |
| Counterfactual evaluation | Not the main published case-00 objective | Included |
| Missing-memory controls | Address and file-integrity controls | NONE/A/B/C/AB/AC/BC/ABC/ABC_CF |
| Numerical X-Ray | Cartridge-level tensor verification | Layerwise consolidated-memory visualization |
| Model weights | Frozen | Frozen |

These are separate experimental achievements.

Their results must not be combined into a single success percentage.

---

## 24. Scientific Achievements

The published AKBASCORE MAM research demonstrates the following capabilities under the documented experimental conditions.

### Physical Memory

- Independent numerical cartridge construction.
- Physical SSD cartridge storage.
- SHA-256 file verification.
- Native geometric memory addressing.
- PCA128 projection.
- FAISS IVF128 candidate search.
- TOP2 candidate ranking.
- PEAK50 final selection.
- Source-free factual answering.
- Bitwise cartridge reconstruction.
- Frozen model-weight preservation.

### Multi-Cartridge Memory

- Independent A/B/C cartridge writing.
- Native numerical memory consolidation.
- Append-only memory construction.
- Earlier-row preservation in measured runs.
- Native memory equivalence checks.
- Layer-3 residual injection.
- Attention K/V memory construction.
- Direct question answering from consolidated memory.
- Predefined three-step relational question sequences.
- Counterfactual memory replacement.
- Missing-cartridge evaluation.
- Unsupported-answer detection during auditing.
- Layerwise numerical memory X-Ray.
- DC_CUT3 versus TEXT_JOINT comparison.
- Seven-figure scientific visualization.
- Public experimental logs and source code.

These are experimental capabilities, not claims of universal reliability.

---

## 25. Scientific Limitations

The project explicitly distinguishes verified numerical operations from broader claims about reasoning.

### Physical-Memory Limitations

The original 96-question result consists of two wordings over the same 48 synthetic facts.

The PEAK50 rule was fixed after earlier experiments involving those facts.

Therefore, the published acceptance panel is not a blind holdout.

The reported SSD loading times may benefit from operating-system page caching.

Large-scale retrieval behavior beyond the published 1,453-cartridge pool remains untested in this record.

### Multi-Cartridge Limitations

The v1.4 demonstrations use two fictional scenarios.

The three-step relational chains are externally predefined.

They do not establish autonomous planning.

Some incomplete-memory conditions produce unsupported answers.

Cartridge contents are not always informationally independent.

DC_CUT3 and TEXT_JOINT produce largely identical answers on the tested panel.

The experiment does not establish numerical-memory behavioral superiority over the text-based control.

The 16-token generation limit prevents a reliable evaluation of unrestricted long-form synthesis.

### General Limitations

The current research does not establish:

- Artificial general intelligence.
- Human-equivalent memory.
- Autonomous scientific reasoning.
- Perfect missing-information detection.
- Universal cross-model portability.
- General-purpose reasoning superiority.
- Production-scale reliability.
- Successful generalization to arbitrary unseen domains.

These boundaries are part of the research record.

---

## 26. Future Research Directions

The next research stages include:

1. Preregistered blind holdout evaluations.
2. New facts and new question templates.
3. Larger relational-memory benchmarks.
4. Reduced information overlap between cartridges.
5. More reliable missing-information detection.
6. Independent cross-model replication.
7. Larger physical cartridge pools.
8. Cold-SSD latency measurement.
9. Consumer GPU evaluation.
10. Longer-context memory experiments.
11. More complex multi-cartridge relationships.
12. Generalization beyond fictional scenarios.
13. Independent scientific reproduction.
14. Stronger controls for causal attribution.
15. Evaluation of unrestricted natural-language synthesis.

These are future research objectives.

They are not presented as completed results.

---

# PART IV — RUNNING THE DEMONSTRATIONS

## 27. Original Physical Silicon Hippocampus

The original physical-memory demonstration remains available.

### Three-Part Google Colab Version

Run the following files in order:

1. [HIPPOCAMPUS_part1.py](HIPPOCAMPUS_part1.py)
2. [HIPPOCAMPUS_part2.py](HIPPOCAMPUS_part2.py)
3. [HIPPOCAMPUS_part3.py](HIPPOCAMPUS_part3.py)

The three parts must run within the same Colab runtime.

Do not restart the runtime between parts.

### Complete Single-File Version

[HIPPOCAMPUS_full.py](HIPPOCAMPUS_full.py)

### Original Demonstration Workflow

The original demonstration performs:

```text
ENGINE INITIALIZATION
        |
        v
SOURCE VERIFICATION
        |
        v
MODEL LOAD
        |
        v
1,453 CARTRIDGE WRITES
        |
        v
SSD MANIFEST VERIFICATION
        |
        v
PCA128 / FAISS IVF128
        |
        v
PEAK50 ACCEPTANCE
        |
        v
GRADIO INTERFACE
        |
        v
LIVE WRITE / SSD / READ
        |
        v
INTEGRITY CHECKS
        |
        v
FIGURES AND EVIDENCE PACKAGE
```

The original interface rotates through the predefined question panel.

It performs the established integrity and answer-verification operations.

### Original Published Log

[Hippocampus.log](Hippocampus.log)

Related records:

- [627.log](627.log)
- [626.log](626.log)
- [624.log](624.log)

---

## 28. Silicon Hippocampus v1.4

### Recommended Runtime

Google Colab with an NVIDIA A100 GPU.

The published execution used:

`NVIDIA A100-SXM4-40GB`

### Three-Part Version

**Part 1 — Engine and Initialization**

[HIPPOCAMPUS_1.4_part1.py](HIPPOCAMPUS_1.4_part1.py)

**Part 2 — Measurement and Evaluation**

[HIPPOCAMPUS_1.4_part2.py](HIPPOCAMPUS_1.4_part2.py)

**Part 3 — Scientific Figures and Gradio**

[HIPPOCAMPUS_1.4_part3.py](HIPPOCAMPUS_1.4_part3.py)

Execute all three parts in order.

Use the same runtime.

### Complete Version

[HIPPOCAMPUS_1.4_full.py](HIPPOCAMPUS_1.4_full.py)

This file provides the complete demonstration in one Python source.

### Published Experimental Record

[HIPPOCAMPUS_1.4_log](HIPPOCAMPUS_1.4_log)

### Reproduction Principles

For faithful reproduction:

- Preserve the original engine.
- Preserve the native CUT3 operations.
- Preserve the experimental question panels.
- Preserve the numerical precision.
- Preserve the model configuration.
- Preserve the integrity checks.
- Preserve raw model outputs.
- Do not substitute expected answers for generated answers.
- Do not merge historical and current measurements.
- Do not present unexecuted results as measured results.

---

## 29. Outputs and Research Records

### Original Physical-Memory Demonstration

The original demonstration produces:

- Four scientific figures.
- A Turkish version of the architecture figure.
- Raw experimental logs.
- Sealed JSON payloads.
- Forward-pass records.
- SHA-256 manifests.
- Verification summaries.
- An evidence ZIP.

The original figures are generated during execution.

### Silicon Hippocampus v1.4

The new demonstration produces:

- Seven scientific JPEG figures.
- Raw model outputs.
- Numerical memory measurements.
- K/V tensor visualizations.
- Memory-condition comparisons.
- Counterfactual results.
- Experimental metadata.
- Integrity verification.
- SHA-256 records.
- A consolidated research package.

The seven figures correspond to:

1. Silicon Hippocampus architecture.
2. Cartridge creation and consolidation.
3. Internal neural memory X-Ray.
4. Mustafa Akbaş experiment.
5. The Matrix / Neo experiment.
6. Causal controls and relational chains.
7. Verification and research record.

The repository links in this README point to the available source files and logs.

Generated figures are not linked to nonexistent repository paths.

---

## 30. Source Files and Research Chain

### Original Physical-Memory System

| File | Role |
|---|---|
| [HIPPOCAMPUS_part1.py](HIPPOCAMPUS_part1.py) | Engine initialization and physical-memory construction |
| [HIPPOCAMPUS_part2.py](HIPPOCAMPUS_part2.py) | RUN pipeline, verification and figures |
| [HIPPOCAMPUS_part3.py](HIPPOCAMPUS_part3.py) | Self-test and Gradio interface |
| [HIPPOCAMPUS_full.py](HIPPOCAMPUS_full.py) | Complete original demonstration |
| [627.py](627.py) | Locked PEAK50 and physical-memory acceptance |
| [626.py](626.py) | Candidate features and reranking |
| [624.py](624.py) | Physical cartridge pool and FAISS addressing |
| [610.py](610.py) | Frozen Qwen CUT3 baseline |
| [566.py](566.py) | Native CUT3 memory operations |
| [575.py](575.py) | Memory consolidation reference |

### Silicon Hippocampus v1.4

| File | Role |
|---|---|
| [HIPPOCAMPUS_1.4_part1.py](HIPPOCAMPUS_1.4_part1.py) | v1.4 engine and initialization |
| [HIPPOCAMPUS_1.4_part2.py](HIPPOCAMPUS_1.4_part2.py) | Read-only measurement and experimental evaluation |
| [HIPPOCAMPUS_1.4_part3.py](HIPPOCAMPUS_1.4_part3.py) | Seven scientific figures and Gradio interface |
| [HIPPOCAMPUS_1.4_full.py](HIPPOCAMPUS_1.4_full.py) | Complete single-file v1.4 demonstration |
| [HIPPOCAMPUS_1.4_log](HIPPOCAMPUS_1.4_log) | Published A100 experimental record |
| [664.py](664.py) | Verified engine research source |
| [677.py](677.py) | Multi-cartridge and missing-memory experimental controls |

### Research Continuity

The project preserves its chronological experimental history.

The original physical-memory results remain independently documented.

The v1.4 consolidation results are added as a new research stage.

Earlier results are not retroactively replaced by later demonstrations.

---

# PART V — LICENSE AND CITATION

## 31. License

The software and materials in this repository are governed by the:

**AKBASCORE RESEARCH SOFTWARE LICENSE**

Authoritative license text:

[LICENSE](LICENSE)

The LICENSE file is the controlling legal document.

### Research and Educational Use

The license permits research, education, evaluation, modification and derivative works under its stated conditions.

### Commercial Revenue Threshold

Commercial use of covered material is permitted without a separate commercial license when the user, together with its affiliates, has combined consolidated annual gross revenue below:

## USD 50,000,000

**Fifty million United States dollars.**

The threshold is based on combined consolidated annual gross revenue in the most recently completed fiscal year.

It is not based on:

- Company valuation.
- Investment raised.
- Market capitalization.
- Total assets.

At or above USD 50,000,000 in annual gross revenue, commercial use of covered material requires a separate commercial license from the copyright holder.

### License Continuity

This research extension does not change the terms of earlier releases.

The authoritative terms remain those in the repository's LICENSE file.

Third-party models and dependencies remain subject to their respective licenses.

These include:

- Qwen2.5-7B-Instruct.
- PyTorch.
- Transformers.
- FAISS.
- Gradio.

### Commercial Licensing

Mustafa Akbaş  
AkbasCore AI Teknoloji  
Mersin, Türkiye

Commercial licensing inquiries may be directed through this repository.

---

## 32. Citation and Permanent Research Records

Earlier public records in the AKBASCORE MAM research line are preserved.

### AKBASCORE MAM v1.0

**Mistral-7B — DC6**

DOI:

[10.5281/zenodo.23245358](https://doi.org/10.5281/zenodo.23245358)

### AKBASCORE MAM v1.1

**Qwen2.5-7B-Instruct — DC3**

DOI:

[10.5281/zenodo.23257347](https://doi.org/10.5281/zenodo.23257347)

### AKBASCORE MAM v1.2.0

**Qwen2.5-7B — CUT3 + FRAM**

DOI:

[10.5281/zenodo.23282902](https://doi.org/10.5281/zenodo.23282902)

### Physical Silicon Hippocampus

**WRITE · SSD · READ**

Frozen Qwen2.5-7B-Instruct.

1,453 independent numerical memory cartridges.

PCA128 · FAISS IVF128 · PEAK50.

Published experimental record:

[Hippocampus.log](Hippocampus.log)

### Silicon Hippocampus v1.4

**Native Internal Tensor Memory Injection & Multi-Hop Relational Synthesis**

Independent numerical cartridge writing.

Native append-only consolidation.

Frozen transformer inference.

Missing-memory and counterfactual controls.

Internal numerical memory X-Ray.

Seven-figure scientific demonstration.

Published experimental record:

[HIPPOCAMPUS_1.4_log](HIPPOCAMPUS_1.4_log)

A permanent DOI can be added after the corresponding versioned research artifact is deposited.

### BibTeX

```bibtex
@techreport{akbas2026siliconhippocampus,
  author      = {Akbaş, Mustafa},
  title       = {{AKBASCORE MAM — The Silicon Hippocampus:
                 Physical Numerical Memory Cartridges,
                 Native CUT3 Consolidation and
                 Multi-Hop Relational Synthesis}},
  institution = {AkbasCore AI Teknoloji},
  address     = {Mersin, Türkiye},
  type        = {Technical Research Report / Software Experimental Record},
  year        = {2026},
  month       = oct,
  url         = {https://github.com/ceceli33/titan-cognitive-core-v2}
}
```

---

## 33. Research Statement

AKBASCORE MAM investigates whether numerical representations derived from a frozen language model can serve as externally maintained, reusable memory.

The research began with the construction of independent numerical attention-memory cartridges.

The Physical Silicon Hippocampus established an experimental WRITE · SSD · READ pathway, including physical storage, native geometric addressing and source-free factual answering.

The Silicon Hippocampus v1.4 extension investigates a complementary capability: consolidating multiple independently written numerical cartridges and using the resulting memory to answer relational questions.

The project distinguishes five research problems:

**Memory Construction**

How information is transformed into numerical memory.

**Memory Storage**

How numerical memories are maintained independently of the model weights.

**Memory Retrieval**

How the system identifies relevant stored information.

**Memory Consolidation**

How independently written memories can be combined while preserving earlier numerical states.

**Memory-Based Reasoning**

How a frozen model uses available memory to answer factual and relational questions, and how reliably it recognizes missing information.

The published experiments demonstrate specific capabilities under controlled conditions.

They also document limitations, including unsupported answers, incomplete-memory failures and substantial behavioral agreement with text-based controls.

The research does not claim biological equivalence, artificial general intelligence or universal reasoning superiority.

AKBASCORE MAM remains an ongoing investigation into the computational possibilities and boundaries of model-native numerical memory.

---

## AKBASCORE AI TECHNOLOGY

### MAM — THE SILICON HIPPOCAMPUS

**Physical Memory · Native Tensor Injection · Append-Only Consolidation · Relational Synthesis**

Research and development:

**Mustafa Akbaş**

AkbasCore AI Teknoloji  
Mersin, Türkiye

**Copyright © 2026 Mustafa Akbaş**

All rights governed by the AKBASCORE RESEARCH SOFTWARE LICENSE.
