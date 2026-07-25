# Supervisor Revision Checklist

Source files checked:
- Thesis PDF: `C:\Users\deeps\Downloads\Human_Centered_Information_Retrieval_for_Swiss_Open_Government_Data_Using_Fuzzy_Logic_Based_Ranking_Thesis_Report (1).pdf`
- Supervisor guide: `C:\Users\deeps\Downloads\Final_Revision_Guide.docx`
- Repository: `C:\thesis`

## Highest Priority: Evidence and Reproducibility

1. Replace or remove the semantic baseline.
   - Current repo issue: `code/ranking/ai_semantic_baseline.py` still defaults `AISemanticBaseline()` to `MockEmbeddingProvider()`.
   - The mock provider uses Python `hash(word)` and pseudo-random vectors, so it is not a defensible AI semantic baseline.
   - Required action: use a real pinned multilingual sentence-transformer model, record model revision/library versions/seeds/checksums, rerun all results; or remove semantic baseline and all comparative claims.

2. Freeze the benchmark and remove live/time drift.
   - Repo has some frozen-corpus logic in `evaluation/experiment_runner.py`, but recency still uses `datetime.now()` in several scoring paths.
   - Required action: use one committed metadata snapshot and one fixed reference date everywhere; describe the experiment as reranking a common candidate pool unless you independently index the full corpus.

3. Repair relevance-judgment provenance.
   - Current files still conflict: `ground_truth_auto.json` and `ground_truth_manual.json` include grades 0-3, while `ground_truth_final.json` includes grades 0-2.
   - `evaluation/data/pooled_candidates.csv` has judge/adjudicated columns that are identical for almost all rows, so the thesis must not claim two independent judges.
   - Required action: document the real single-assessor process, explain any 0-3 to 0-2 mapping, rename/relabel misleading adjudication fields, and make the pipeline fail if final ground truth is missing.

4. Fix statistics before publishing tables.
   - Current repo issue: Wilcoxon statistic is discarded and written as `0.0` in `evaluation/experiment_runner.py`.
   - Current repo issue: `system_summary.csv` currently contains zero/nan rows, so published values must be regenerated from a clean working pipeline.
   - Required action: store Wilcoxon statistic, define zero-difference handling, use correctly labelled confidence intervals, rerun pairwise tests and Holm correction.

5. Make the user study auditable.
   - Current thesis problem: Appendix A lists 10 questionnaire items, but Table 7.2 reports 21 criteria/items.
   - Required action: insert the exact instrument actually used, response scale, item order, open-ended prompts if any, and de-identified item-level responses or a clear restricted-data audit trail.

## Thesis Claim Alignment

6. Use one set of research questions everywhere.
   - Current PDF has different RQs in Chapter 1, Section 7.8, and Chapter 8.
   - Required action: choose one evidence-compatible RQ set and repeat it verbatim in abstract, introduction, methodology, Chapter 7, Chapter 8, and README.

7. Correct false performance claims.
   - Current PDF still says the proposed system achieved the highest MRR.
   - Table 7.1 shows Portal Search and Metadata Ranking at MRR 0.9000, while Fuzzy is 0.8333.
   - Required action: say Fuzzy is competitive or higher than BM25/semantic only where supported, not highest overall.

8. Use the implemented fuzzy factor name consistently.
   - Current PDF mixes `resource availability`, `dataset quality`, and `organizational quality`.
   - Required action: use `resource availability` unless code and results are changed and rerun.

9. Rename the metadata baseline honestly.
   - Supervisor says it is not metadata-only because it includes portal-position relevance plus recency/completeness/resources.
   - Required action: call it a linear weighted-sum or linear multi-criteria baseline, report weights, and discuss why fuzzy adds value beyond that simpler interpretable competitor.

10. Tone down causal/top-grade claims.
   - Current thesis uses language like demonstrates/improves/supports too strongly for a descriptive benchmark plus single-condition questionnaire.
   - Required action: recalibrate abstract and conclusion after rerun; do not claim explanations caused trust or understanding.

## Missing Results and Sections

11. Insert the missing per-query table.
   - Current PDF still contains `Table??` in Section 7.3.2.
   - Required action: include corrected per-query results, or at minimum a readable nDCG@10 table for 15 queries by five systems.

12. Discuss ENV-01 correctly.
   - Supervisor says all systems score zero because the corpus lacks a relevant air-quality dataset.
   - Required action: explain it as corpus coverage/candidate-generation failure, not ranking quality; optionally add a supplementary sensitivity check excluding ENV-01.

13. Insert a compact significance table.
   - Required action: after fixing pipeline, include system pair, metric, Wilcoxon statistic, raw p, Holm-corrected p, and decision.

14. Add or remove promised sensitivity analyses.
   - Required action: add Section 7.3.4 with production-config sensitivity results.
   - Resolve mismatch where sensitivity baseline reported MAP 0.8191 while Table 7.1 reported Fuzzy MAP 0.7839.
   - Either perform membership-function perturbation and leave-one-rule-out analysis, or delete promises that they were done.

15. Delete phantom heuristic evaluation text.
   - Required action: remove roadmap wording that announces a heuristic evaluation section if no real method/results are added.

## Methodology Rewrite

16. Remove false two-judge/adjudication/kappa story.
   - Current PDF still contains `Two independent judges`, `adjudication`, `Cohen's kappa`, and the Cohen reference.
   - Required action: rewrite Sections 6.2, 6.5, 6.8, 6.9, 7.9, and summaries that depend on this protocol.
   - Replace with: single-assessor author labels, consolidation/finalization pass, published files for audit, and limitation about author bias.

17. Make Appendix C traceable.
   - Required action: replace generic explanation wording with one real saved trace containing query, dataset ID/title, matched fields, recency value, resources, completeness, activated rules/strengths, defuzzified score, and final explanation.

## Visible PDF Cleanup

18. Fix visible unresolved/incorrect text.
   - Found in current PDF: `Table??`, `[?5]`, `[?6]`, `500 datasetsg`, missing spaces around citations/words, and duplicated/awkward methodology text.

19. Reconcile participant wording.
   - Current PDF uses both `Bachelor's/Master's students` and `Master's students`.
   - Required action: match the actual participant group exactly.

20. Reconcile institution/front matter.
   - Supervisor notes mismatch between University of Neuchatel in title page and University of Fribourg/Human-IST in README.
   - Required action: use official institutional wording consistently.

21. Update appendices and generated references.
   - Appendix A must match Table 7.2.
   - Appendix B must match implemented membership functions and factor names.
   - Update TOC/list of tables after adding sections.

## Repository and Submission Gate

22. Rewrite README to match the completed study.
   - Required action: same RQs as thesis, actual n=10 formative design, actual systems, correct semantic implementation, real test paths, real project tree, fixed clone URL, supported Python version, pinned dependencies.

23. Provide one fresh-clone reproducibility command.
   - Required action: command must reproduce Table 7.1 from committed snapshot, fixed date, final judgments, real/pinned semantic baseline, and production fuzzy ranker without live network/manual copying.
   - Record checksums for snapshot, judgment file, and generated summary.

24. Final PDF gate before submission.
   - Compile twice.
   - Search final PDF for: `??`, `[?`, `two independent`, `kappa`, `adjudicated`, `mock`, `500 datasetsg`, `highest MRR`, `organizational quality`, old participant wording.
   - Inspect every page at 100% zoom.

## Practical Work Order

1. Fix the evaluation code and data first.
2. Rerun the benchmark and regenerate tables/results.
3. Rewrite methodology/provenance/statistics sections.
4. Align RQs, claims, abstract, conclusion, limitations, and README.
5. Add missing per-query/significance/sensitivity/user-study audit material.
6. Do the visible PDF cleanup and final compile/search gate.
