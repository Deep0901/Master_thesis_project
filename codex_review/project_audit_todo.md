# Project Audit TODO

Audit date: 2026-07-20

Scope checked:
- Repository structure and git state
- README and requirements
- Evaluation data/results
- Fuzzy/evaluation tests
- Supervisor red flags from `Final_Revision_Guide.docx`

## Executive Summary

The project is not ready for final submission. The main problem is not the prototype UI; it is the evidence chain. The repository currently cannot reproduce the nonzero thesis Table 7.1 values, the final judgment file appears overwritten or inconsistent, the semantic baseline is still mock by default, and the tests do not guard against these failures.

Work should start with data/evaluation repair, not prose editing.

## Critical: Must Fix First

### 1. Restore and freeze the authoritative ground truth

Current state:
- `evaluation/ground_truth_final.json` currently has 15 queries, 450 judgments, grades 0-3, and annotator `heuristic`.
- `evaluation/data/pooled_candidates.csv` has 150 rows and adjudicated grades 0-2.
- 148 of 150 pooled rows have `judge1_grade == judge2_grade == adjudicated_grade`.
- The supervisor guide says the final file used for Table 7.1 should be the authoritative final judgment file, but the current file no longer matches that story.

What to do:
- Decide the actual authoritative final judgment set.
- If `pooled_candidates.csv` is the final human/consolidated set, regenerate `ground_truth_final.json` from it.
- Use labels such as `single_assessor` or `author_consolidated`, not `heuristic` or `adjudicated`, unless those labels are literally true.
- Document the relevance scale in one place: either 0-2 or 0-3.
- Remove fallback behavior that silently copies `ground_truth_auto.json` or `ground_truth_manual.json` into `ground_truth_final.json`.

### 2. Fix evaluation results: current generated tables are invalid

Current state:
- `evaluation/results/query_metrics.csv` has 90 rows but all metrics are `0.0`.
- `evaluation/results/system_summary.csv` has all metric means as `0.0` and CI values as `nan`.
- `evaluation/results/publication_tables.md` is all zeros/nans.
- This does not match the thesis PDF values, e.g. Portal MAP 0.8469 and Fuzzy MAP 0.7839.

What to do:
- Repair the ground-truth/ranked-document ID matching.
- Rerun the full evaluation from the fixed authoritative data.
- Add a regression test that fails if all systems/all metrics are zero.
- Add a check that regenerated Table 7.1 matches the published values or explicitly updates the thesis values.

### 3. Replace or remove the mock semantic baseline

Current state:
- `code/ranking/ai_semantic_baseline.py` defaults `AISemanticBaseline()` to `MockEmbeddingProvider()`.
- `evaluation/experiment_runner.py` imports the prototype semantic baseline and `AISemanticBaselineAdapter` calls `AISemanticBaseline()` with no real provider.
- `requirements.txt` comments out `sentence-transformers` and `torch`.

What to do:
- Either install/pin a real multilingual sentence-transformer model and revision, or remove the semantic baseline from the benchmark and thesis claims.
- If keeping it, record model name, model revision, library versions, random seeds, embedding cache/checksum, and reproduce results without network access after setup.

### 4. Fix time drift in scoring

Current state:
- Some code uses frozen corpus search, but `evaluation/experiment_runner.py` still contains `datetime.now()` in recency/scoring and timestamps.

What to do:
- Use a fixed evaluation reference date for all recency calculations.
- Keep timestamps only as metadata, not as ranking inputs.
- Add a test that the same command produces identical metrics on different calendar dates.

### 5. Repair statistical outputs

Current state:
- `pairwise_statistics.csv` has `statistic` values of `0.0` and `p_value` as `nan`.
- `evaluation/experiment_runner.py` still writes `statistic: 0.0` after calling SciPy Wilcoxon.
- Confidence interval labelling is inconsistent: t-based summary CI exists, while bootstrap intervals are separate pairwise mean-difference intervals.

What to do:
- Store the actual Wilcoxon statistic.
- Explicitly handle identical vectors and zero differences.
- Decide whether table-level CIs are t intervals or bootstrap intervals and label them accurately.
- Clamp or explain bounded metric intervals if a method can exceed [0, 1].

## High Priority: Tests and Code Health

### 6. Fix the broken test suite

Current test result:
- Command run: `.venv\Scripts\python.exe -m pytest code\tests evaluation\tests -q`
- Result: 13 failed, 19 passed.

Main failures:
- Tests expect old API names such as `mf_type`, `LEFT_SHOULDER`, `RIGHT_SHOULDER`, `TNorm.MIN`, `OGDRuleBase.create_rules()`, and `CalibratedOGDVariables.create_relevance_variable()`.
- Actual production code uses different dataclass fields/enums/methods.
- One integration assertion says the highest-quality dataset should rank highest, but current engine returns a lower score for that case.

What to do:
- Decide whether tests or implementation represent the intended API.
- Update tests to current API or add compatibility aliases if the thesis describes the old API.
- Add evaluation-specific tests that check nonzero metrics, final-ground-truth schema, and no fallback ground-truth copying.

### 7. Remove duplicate/prototype confusion

Current state:
- There are main modules under `code/`.
- There are duplicated modules under `code/prototype/`.
- There is another nested duplicate under `code/prototype/prototype/`.
- `evaluation/experiment_runner.py` imports from `code.prototype.ranking.ai_semantic_baseline`, not the top-level `code.ranking.ai_semantic_baseline`.

What to do:
- Choose one source of truth for each system component.
- Point evaluation imports to production modules only.
- Mark legacy prototype/backup folders clearly or remove them from the submission branch.

### 8. Fix README and project metadata

Current state:
- README still describes the planned study, not the completed one.
- It says user study `15-20 participants, within-subjects design`, while the thesis uses n=10.
- RQs differ from the thesis.
- Clone URL is placeholder `https://github.com/username/swiss-ogd-fuzzy.git`.
- Test command says `pytest tests/ -v`, but tests live in `code/tests` and `evaluation/tests`.
- README has mojibake characters in headings and diagram.

What to do:
- Rewrite README around the final experiment.
- Use the same RQs as the thesis.
- Document actual systems: Portal, BM25, linear weighted baseline, Fuzzy, Semantic if kept.
- Provide one reproducibility command.
- Fix encoding/mojibake and remove placeholder URLs.

## Medium Priority: Thesis-Repository Alignment

### 9. Align methodology with actual data

Current state:
- Code still includes kappa/agreement helpers and final export labels `adjudicated`.
- Thesis PDF still says two independent judges/adjudication/kappa.

What to do:
- If no two independent judges existed, remove the two-judge story from thesis and README.
- Keep kappa code only as unused utility, or remove it to avoid confusion.
- Add a limitation about single-assessor/author-bias.

### 10. Reconcile fuzzy factor names

Current state:
- Implemented fuzzy inputs include resource availability/resources.
- Thesis text also uses dataset quality and organizational quality in places.

What to do:
- Use one term everywhere: `resource availability`, unless code is changed and results rerun.
- Update Appendix B/table captions/prose accordingly.

### 11. Add missing result sections after rerun

Needed:
- Per-query table for 15 queries by systems.
- Compact significance table with statistic, raw p, corrected p, decision.
- ENV-01 coverage/candidate-generation discussion.
- Sensitivity section using the same production configuration.
- Membership-breakpoint sensitivity or deletion of claims promising it.
- Leave-one-rule-out analysis or deletion of claims promising it.

### 12. Make the user-study data auditable

Current issue from supervisor guide:
- Appendix A lists 10 questionnaire items, Table 7.2 reports 21.

What to do:
- Add exact questionnaire wording, order, scale, and item grouping.
- Add de-identified item-level response matrix if allowed.
- If open-ended answers were not collected/analyzed, stop calling it qualitative or mixed-method.

## Final Submission Gate

Before submission, all of these should be true:

- `ground_truth_final.json` is the real authoritative final file and does not get silently overwritten.
- One documented command regenerates Table 7.1 from a frozen snapshot and fixed date.
- `publication_tables.md` has nonzero, thesis-matching metrics.
- Tests pass, including evaluation regression tests.
- Semantic baseline is real/pinned or removed.
- README, thesis, and repository name the same RQs, systems, institution, and participant design.
- Final PDF has no `??`, `[?`, `two independent`, `kappa`, `adjudicated`, `mock`, `500 datasetsg`, `highest MRR`, or `organizational quality` unless actually supported.

## Recommended Work Order

1. Stop editing thesis prose temporarily.
2. Fix `ground_truth_final.json` and remove fallback copying.
3. Fix evaluation ID matching and rerun results until metrics are nonzero and reproducible.
4. Decide semantic baseline: real pinned model or remove.
5. Fix statistical outputs and confidence interval labels.
6. Add regression tests for the evaluation pipeline.
7. Rewrite README.
8. Update thesis tables/prose/limitations from corrected outputs.
9. Do the final PDF search and visual page check.
