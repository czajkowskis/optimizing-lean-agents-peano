# Pilot: does entropy at the failing line choose a better repair?

## Question

Lean already says which line of a failed proof breaks, so entropy adds nothing to *finding* the error. The open question is whether the model's entropy on that line tells us *how to repair* it:

- **High entropy on the error line:** the model was hesitating between options. Resampling from that line should often pick a different, working option.
- **Low entropy on the error line:** the model was confidently wrong. Resampling from the same spot will likely repeat the mistake, so it is better to back up or restart.

**Entropy earns its place only if an entropy-gated repair beats the same repair without the gate.** If it doesn't, the honest conclusion is that the compiler does the work and entropy is not needed for repair.

## Design in one paragraph

Take failed whole-proof attempts that we already have, with their Lean error line and per-token entropy. For **every** selected attempt, run **every** repair arm, with the same number of samples per arm. The gated policy (C) and its random control (D) are then computed offline, by choosing per attempt which arm's outcome counts. This factorial design costs nothing extra for C and D, makes D exact instead of noisy, and lets us test the core hypothesis directly as an interaction: *the advantage of resampling locally over backing up is larger when the error line has high entropy.*

## Cost ladder: cheapest experiment first

The arms are run in steps, cheapest first. Each step runs only if the previous step's DECISION line says so. Each step reuses everything generated and compiled before it, so nothing is paid for twice.

| Step | Generates | New tokens | GPU time (1 A100) | Answers | Stop here if |
|---|---|---|---|---|---|
| 0 (done, free) | nothing: offline selection and checks on the laptop | 0 | 0 | Are the cut points, prompts and code paths right? | a check fails |
| smoke | 20 attempts × 1 sample of each arm | about 0.4M | minutes | Does the pipeline run on the cluster? Does Lean there agree with Helios? Real token lengths? | Lean disagrees with Helios |
| **1** | LOCAL × 2 samples, all 872 attempts | about 4.4M | **under 1 hour** | **H1:** does resampling from Lean's error line beat fresh attempts per token? Also, descriptively, is LOCAL success higher on high-entropy lines? | LOCAL loses to FRESH per token. Then no choice of repair point can pay off, and steps 2–3 are moot. |
| 2 | adds BACK × 2 samples | about 5M | under 1 hour | **H2 and the crossover:** should entropy choose between LOCAL and BACK? | no interaction, or an interaction without crossover |
| 3 | adds FEEDBACK, and raises every arm to 4 samples | about 58M | about 6–12 hours (4 GPUs: 2–3 hours of wall time) | Confirms with more samples; compares with the strong baseline; tests C-feedback | (final) |

The largest **fixed** cost is setting up Lean on AMU. Downloading prebuilt Mathlib takes minutes. If no prebuilt files exist for the pinned fork, Lean must be built from source, which takes hours of CPU. That cost is paid once, by the smoke test.

**Why step 1 alone is informative:** if fresh attempts are cheaper per success than repairing from the error line, then gating that repair by entropy can't beat fresh sampling. That ends this line of work for under 1 GPU-hour.

**Its limit:** step 1's high-vs-low comparison has no control. High-entropy attempts might be more repairable for reasons unrelated to the line, for example the error kind. BACK in step 2 is that control.

## Arms

All arms use Goedel-Prover-V2-8B, bf16, temperature 1.0, the same vLLM version, and one fixed seed per (attempt, arm, sample). Prefixes are cut from the stored `output_ids`, not from re-tokenized text, so the model continues exactly what it wrote.

| Arm | What is generated | Prefix kept | Extra GPU cost |
|---|---|---|---|
| **FRESH** | a new attempt from scratch (today's method) | prompt only | **none**: the 256 existing attempts per problem give its success rate and tokens per attempt |
| **LOCAL** | resample from the start of Lean's first error line | prompt + output up to the error line | yes |
| **BACK** | resample from 3 accepted lines before the error line (or the start of the proof block, if it is closer) | prompt + output up to that line | yes |
| **FEEDBACK** | Goedel-V2's own correction mode: a new attempt whose prompt contains the failed proof and Lean's error message (`gp_seed_correction` template, already in `prover/`) | prompt with feedback | yes, and the most expensive |

**FEEDBACK** is the baseline reviewers will ask for first: the model was trained to repair from Lean errors. Entropy-guided repair has to be compared with it, not only with resampling.

**Why 3 lines for BACK:** in the existing analysis the within-proof z-score of line NLL starts rising 1–2 lines before the error (−1: +0.22, error line: +0.71). Backing up 3 lines goes past that rise. The number is fixed in advance and not tuned.

### Policies evaluated offline from the arms

| Policy | Rule | Role |
|---|---|---|
| **A** | always FRESH | today |
| **B** | always LOCAL | cheap, obvious baseline from Lean's line |
| **C-back** | LOCAL if the error line has high entropy, else BACK | **the hypothesis** |
| **C-fresh** | LOCAL if high entropy, else FRESH | the "restart when confidently wrong" variant |
| **C-feedback** | LOCAL if high entropy, else FEEDBACK | "ask the model to rethink when confidently wrong" |
| **D** | LOCAL or BACK chosen at random, in the same proportion as C-back | control for C-back; computed exactly as the expectation |
| **E** | always FEEDBACK | strong baseline |

## The entropy gate (fixed before any generation)

- **Feature:** `maxH` of the error line, z-scored within its own proof against the accepted lines before it. This is the feature with the best error-line AUROC in the existing analysis (0.708 on all problems). Z-scoring removes differences in baseline entropy between problems.
- **Threshold:** the median of that z-score over all 12,238 mapped failed attempts, computed once from the existing data and written into the pilot config before anything is generated. It is **not** tuned on repair outcomes.
- **Robustness check (reported, not used to choose):** the same analysis with the threshold computed on the paper's train problems only, evaluated on test problems.

## Which attempts

**Problems (Stage 1):** the 23 problems that the baseline solves at a rate between 0.4% and 64% (1 to 163 of 256). Excluded: the 59 unsolved ones (saved for Stage 2) and the 3 near-saturated ones (`putnam_1988_b1`, `putnam_1977_a3`, `putnam_1986_b1`, at 76–94%), where fresh sampling already wins and repair can't show anything.

**Attempts per problem:** up to 40 failed attempts with a mapped first error, sampled with a fixed seed and **stratified by the gate**: 20 above the threshold and 20 below. This keeps both strata balanced within each problem, so problem difficulty can't masquerade as an entropy effect. That gave **872 attempts** (432 high, 440 low): `putnam_1990_a1` has only 7 high-entropy candidates, and `putnam_2000_a2` has 5 candidates, all high, so it adds nothing to H2. The gate threshold came out at z = 0.826 (0.851 on train problems only).

**Excluded attempts**, the same as in `error_lines85.py`: no Lean block, error in the statement lines, no positioned error, or an error line that isn't in the model's block. Also excluded: attempts that hit the 40,960-token limit (loops), because there is nothing sensible to continue.

**Samples:** 4 per attempt per arm. That gives a per-attempt success estimate and allows pass@k curves.

## Mechanics

1. **Cutting the prefix.** Map Lean's first error to a line of the model's last ```` ```lean4 ```` block (the existing ASCII-only `difflib` mapping). Find the first output token whose text starts that line, cut `output_ids` just before it, and append the prefix to `prompt_ids`. Check that the cut lands on a newline boundary; skip the attempt and count it if not.
2. **Generating.** vLLM offline with `prompt_token_ids`, prefix caching on, so the 4 samples of LOCAL and BACK share their prefill. LOCAL and BACK have a hard cap of **8,192 new tokens** (the remainder of a proof block is typically about 2–3k). FEEDBACK keeps the original 40,960 cap, because it reasons from scratch.
3. **Building the proof to compile.** The compiled proof is the prefix plus the continuation, and the last Lean block is extracted the same way as in the original run. A continuation that never closes the block counts as a failure.
4. **Compiling.** Use the same Lean and Mathlib version (the pinned `mathlib4` submodule) and the same REPL scheduler (`lean_compiler/repl_scheduler.py`, with the same JSON-prefix fix as on Helios) as the run that produced the attempts. Both generating and compiling run on AMU. Before any repair is compiled, a check job recompiles 87 **unchanged** original attempts. If any verdict differs from Helios, everything after it is cancelled.
5. **What is logged per sample:** attempt id, arm, sample index, seed, prefix length, new tokens, finish reason, compile verdict, and the first error, if any.

## Analysis

**Primary outcome:** whether a repair sample passes Lean.

**Primary efficiency metric:** successes per million generated tokens. Prefill is reported separately: it is cheap with caching but not free.

1. **H1, repair is worth it at all:** compare LOCAL and FRESH on successes per million tokens, per problem and pooled. If LOCAL doesn't beat FRESH, the gating question is moot and we report that.
2. **H2, the main test:** a difference in differences. Within each problem, compute LOCAL minus BACK success rate in the high-entropy stratum minus the same in the low-entropy stratum, then average over problems. Both arms run on the same attempts, so each attempt is its own control. Problem-level random effects aren't needed, because the comparison is within problems.
   - **CI:** a bootstrap that resamples problems.
   - **p-value:** a one-sided permutation test that shuffles stratum labels within each problem.
   - **Why not a mixed logistic regression:** it would test the same thing but needs statistics libraries on the cluster, so it's left as an optional check on the downloaded results.
3. **Policy comparison:** C-back against B, D and E on success rate and successes per million tokens, with 95% bootstrap CIs that resample *problems* (attempts within a problem are not independent). C-fresh is reported alongside.
4. **Secondary:** results by error kind ("unsolved goals" is 55% of errors and entropy localizes it worst, AUROC 0.64), by train and test problems, and pass@k for k = 1–4.

**Power, roughly:** with about 435 attempts per stratum, 4 samples each, and clustering by problem, Stage 1 can detect an interaction of about 7 percentage points or more if repair success is around 10%. Smaller effects need Stage 2.

## Decision rule (set now)

**The interaction alone is not enough.** A gate can beat "always LOCAL" only if there is a **crossover**: the low-entropy alternative must beat LOCAL on low-entropy lines. If LOCAL works better on high-entropy lines but is still the best choice on low-entropy ones, entropy *predicts* where repair works but doesn't change *which* repair to use. (The synthetic test of `analyze.py` showed exactly this case.)

**"Interaction"** below means: H2 difference in differences > 0, with the 95% problem-bootstrap CI above 0, **and** permutation p < 0.05.

- **Go to Stage 2** if there is an interaction **and** C-back beats both B and D on successes per million tokens, with the lower end of the 80% problem-bootstrap CI above zero.
- **Go with a different gate** if there is an interaction and C-feedback (LOCAL on high-entropy lines, FEEDBACK on low ones) or C-fresh beats B, but C-back does not.
- **Interaction without crossover:** report that entropy predicts repairability but doesn't choose the repair. Stage 2 only if the predictive use is worth testing, for example ranking which failed attempts to repair first.
- **Stop and write it up as a negative result** if LOCAL ≈ BACK in both strata (interaction CI centered near 0). That would mean entropy doesn't help choose a repair, which is worth a slide and a paragraph.
- **Rethink the framing** if E (FEEDBACK) dominates every resampling policy on successes per million tokens. Then the next question is whether entropy can gate *when to ask for feedback*, not where to resample.

`analyze.py` applies these rules and prints the verdict. Its synthetic tests, with a crossover, an interaction without crossover, and no effect, give GO, INTERACTION WITHOUT CROSSOVER and STOP respectively.

## Stage 2 (only if Stage 1 says go)

- Add the 59 unsolved problems, starting with those whose failed attempts reach the furthest into the proof. A repair that solves a problem that 256 fresh attempts never solved is a headline result.
- Scale up to about 100 attempts per problem.
- Try a second model (Goedel-Prover-V2-32B) on a subset, so the effect isn't specific to one model.

## Budget (to be confirmed by the smoke test)

Estimates assume about 2.5k new tokens per LOCAL sample, 3k per BACK sample and 14k per FEEDBACK sample (the average full attempt in the AMU check set).

| Item | Samples | New tokens |
|---|---|---|
| LOCAL | 3,488 | about 9M |
| BACK | 3,488 | about 10M |
| FEEDBACK | 3,428 (15 attempts have a correction prompt over the prover's 30,720-token limit and are skipped, as the prover would) | about 48M |
| **Total** | 10,404 | **about 67M** |

At about 3k tokens/s per A100 with vLLM, that is **about 7–15 GPU-hours**, which fits under the 30 GPU-hour limit for AMU. FEEDBACK is about 70% of the cost. If needed it can run on half the attempts, without changing the H2 test.

Compiling is about 11k proofs on an AMU CPU node, a few node-hours.

## Smoke test before Stage 1

Run 20 attempts spread over the problems × 1 sample per arm (`run.sh --smoke`). Check that:

- prefixes cut on line boundaries;
- continuations close the Lean block;
- the compile pipeline reproduces the original verdict when given the **unchanged** original attempt (an exact check of the pipeline);
- the real token lengths per arm are known, to replace the budget estimates above.

## Pitfalls already handled

- **Threshold leakage:** the gate is fixed from existing data before generation.
- **Problem difficulty confounding entropy:** strata are balanced within each problem, and H2 compares only within problems.
- **Tokenization drift:** prefixes come from stored token ids.
- **Unequal budgets:** LOCAL and BACK have the same cap, and every comparison is also reported per token.
- **Loops:** attempts truncated at the token limit are excluded.
- **Non-independence:** CIs resample problems, not samples.

## Files

How to run the pilot and what each file does: [README.md](README.md).
