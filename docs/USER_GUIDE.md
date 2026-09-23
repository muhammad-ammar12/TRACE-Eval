# TRACE-Eval user guide

This guide runs one reference-aligned evaluation from input preparation through expert-adjudicated output export.

## 1. Prepare the environment

Clone the repository and enter its root directory:

```console
git clone https://github.com/muhammad-ammar12/TRACE-Eval.git
cd TRACE-Eval
```

Create and activate a virtual environment:

```console
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the Python environment:

```console
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Supply model credentials

The notebook's supplied configuration uses OpenAI chat and embedding APIs. Set the key in the shell before starting JupyterLab, add `OPENAI_API_KEY` to Google Colab Secrets, or use the notebook's secure prompt:

```powershell
$env:OPENAI_API_KEY="your_openai_api_key_here"
```

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

For optional Voyage embeddings, install the Voyage Python client and set `VOYAGE_API_KEY`. The default workflow does not require it.

```console
python -m pip install voyageai
```

## 3. Prepare an aligned input pair

Each run compares:

1. a reference extraction containing the evidence expected for one study; and
2. a structured extraction produced by the model or pipeline being evaluated.

Keep both files at the same study level. DOCX files may contain paragraphs and tables; table rows are converted to pipe-separated text before item extraction. Plain-text and JSON-like generated extraction files can also be read.

The repository includes 14 example pairs under:

```text
examples/reference/
examples/extracted/
```

The Tanner pair is preconfigured in the notebook.

## 4. Open the notebook

Start JupyterLab from the repository root:

```console
python -m jupyterlab Evaluation_framework.ipynb
```

The working directory must contain `trace_eval.py`, `requirements.txt`, and the `examples/` directory. In Google Colab, upload the repository files or clone the repository, then change into the cloned directory before running the notebook.

## 5. Configure the run

Edit the notebook's configuration cell:

```python
study_id = "tanner"

generated_docx_path = "examples/extracted/tanner_extraction_4o.docx"
groundtruth_docx_path = "examples/reference/tanner.docx"
output_dir = "outputs/tanner"

config = EvalConfig(
    extraction_model="gpt-5.1",
    judge_model="gpt-5.1",
    embedding_provider="openai",
    openai_embedding_model="text-embedding-3-large",
    top_k=5,
    min_candidate_similarity=0.75,
    sparse_min_candidate_similarity=0.50,
    lexical_fallback_top_k=5,
    lexical_min_score=0.50,
)
```

Use a unique `study_id` and output directory for each study. The model fields accept model IDs supported by the configured OpenAI client. The embedding provider can be `openai` or `voyage`; the corresponding model field and API credential must also be set.

The main and sparse similarity thresholds decide which candidates are shown to the judge and expert. Lexical fallback supplements embedding retrieval for short numeric or statistical items. Similarity and lexical scores do not directly enter the final metric formulas.

## 6. Generate reference items and candidate matches

Run the item-level evaluation cell:

```python
result = evaluate_extraction_pair_item_level(
    groundtruth_docx_path=groundtruth_docx_path,
    generated_docx_path=generated_docx_path,
    output_dir=output_dir,
    study_id=study_id,
    config=config,
)
```

This stage uses LLM calls to:

1. convert the reference extraction into clinically meaningful evaluation items or claim groups;
2. convert the generated extraction into context-preserving evidence blocks;
3. retrieve dense and lexical candidates;
4. assign provisional `agree`, `contradict`, or `cannot_verify` labels.

When no candidate passes retrieval, the framework assigns `missing`. The output files from this pass preserve the reference items, extracted blocks, candidate matches, provisional judgements, and review dashboard.

## 7. Review and adjudicate every item

Run the expert-review cell:

```python
review_save_path = f"{output_dir}/{study_id}_adjudicated_review_in_progress.csv"
editable_review_df = launch_expert_review_dashboard(
    review_df,
    save_csv_path=review_save_path,
)
```

For each row, inspect:

- the reference item and required subfacts;
- the retrieved extracted-evidence block;
- retrieval scores and method;
- the provisional label, confidence, rationale, and critical difference;
- the subfact statuses.

Choose one decision:

| Decision | Effect |
| --- | --- |
| `agree_with_llm` | Accept the provisional label. |
| `agree` | Set the final label to `agree`. |
| `contradict` | Set the final label to `contradict`. |
| `cannot_verify` | Set the final label to `cannot_verify`. |

Add an expert comment when the reason for an override or uncertainty should be retained. Save before leaving the notebook. Review every row before producing the final study metrics.

The CSV can also be edited manually. Preserve the existing columns and enter decisions only in `expert_decision` and comments in `expert_comment`.

## 8. Export the final adjudicated results

Run the final export cell:

```python
adjudicated_df = pd.read_csv(review_save_path).fillna("")

final_metrics = compute_final_metrics_from_review(adjudicated_df)
final_paths = save_final_adjudicated_metrics(
    review_df=adjudicated_df,
    output_dir=output_dir,
    study_id=study_id,
)
```

This writes:

```text
{study_id}_final_adjudicated_review.csv
{study_id}_final_adjudicated_metrics.xlsx
{study_id}_final_adjudicated_metrics.json
```

The XLSX workbook contains:

- `summary_metrics`;
- `final_adjudicated_review`;
- `subfact_diagnostics`;
- `label_counts`;
- separate `cannot_verify`, `missing`, `contradiction`, and `agree` item sheets.

## 9. Interpret the outputs

The primary unit is one reference evaluation item or claim group. Let:

- `A` = final `agree` count;
- `C` = final `contradict` count;
- `M` = final `missing` count;
- `U` = final `cannot_verify` count.

Calculate:

```text
Completeness       = A / (A + C + M)
Correctness        = A / (A + C)
F1                 = 2 * completeness * correctness / (completeness + correctness)
Cannot-verify rate = U / (A + C + M + U)
```

The module exports completeness, correctness, detailed counts, and the full item-level record. Use the final adjudicated label counts for F1 and cannot-verify rate reporting.

## 10. Evaluate another study

Add a new aligned pair to local input directories, change the two file paths and `study_id`, and rerun the notebook from the configuration cell onward. Write each study to a separate output directory or use unique study IDs to avoid replacing existing files.

API-backed item extraction, embeddings, and provisional labels may vary when a provider updates a hosted model. Preserve the model identifiers, settings, generated review files, expert decisions, and exact repository commit with each evaluation run.
