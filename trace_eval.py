from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from docx import Document
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential




SourceType = Literal["ground_truth", "generated"]
ItemLabel = Literal["agree", "contradict", "cannot_verify"]
FinalLabel = Literal["agree", "contradict", "cannot_verify", "missing"]
ExpertDecision = Literal["agree_with_llm", "agree", "contradict", "cannot_verify"]
SubfactStatus = Literal["present", "absent", "contradicted", "cannot_verify"]

ALLOWED_CATEGORIES = {
    "study_design",
    "setting",
    "population",
    "sample_size",
    "allocation",
    "blinding",
    "inclusion_criteria",
    "exclusion_criteria",
    "baseline_characteristics",
    "intervention",
    "comparator",
    "outcome",
    "time_point",
    "statistical_method",
    "effect_size",
    "p_value",
    "statistical_result",
    "funding",
    "conclusion",
    "other",
}

SPARSE_CATEGORIES = {
    "p_value",
    "statistical_result",
    "effect_size",
    "sample_size",
    "time_point",
    "outcome",
    "intervention",
}


@dataclass
class EvalConfig:
    """Central settings for item-level GT-focused evaluation."""

    # Models
    extraction_model: str = "gpt-5.1"
    judge_model: str = "gpt-5.1"

    # Embeddings. OpenAI is simplest; Voyage is optional.
    embedding_provider: Literal["openai", "voyage"] = "openai"
    openai_embedding_model: str = "text-embedding-3-large"
    voyage_embedding_model: str = "voyage-4-large"

    # Retrieval. Similarity is used only for retrieval, never as a score.
    top_k: int = 5
    min_candidate_similarity: float = 0.25
    sparse_min_candidate_similarity: float = 0.14
    lexical_fallback_top_k: int = 3
    lexical_min_score: float = 0.18

    # Chunking for long extraction files.
    max_chunk_chars: int = 45_000
    chunk_overlap_chars: int = 1_500

    # LLM behaviour.
    temperature: float = 0.0
    max_rationale_words: int = 70

    # Review/dashboard behaviour.
    low_confidence_threshold: float = 0.60
    flag_low_confidence: bool = True
    flag_gt_unspecified_gen_concrete: bool = True
    fill_missing_display_values: bool = True

    # Metrics. If expert decision is blank, use the LLM/system provisional label.
    allow_metrics_with_incomplete_expert_review: bool = True


@dataclass
class EvaluationItem:
    """A GT evaluation item / claim group.

    This is the primary metric unit. It may contain multiple required subfacts,
    but the main dashboard and primary metrics use one row per item.
    """

    item_id: str
    source: Literal["ground_truth"]
    category: str
    item_name: str
    canonical_fact: str
    required_subfacts: List[str]
    evidence_text: str
    source_section: str = ""
    chunk_id: Optional[str] = None


@dataclass
class GeneratedEvidenceBlock:
    """A generated extraction block/item preserved with context.

    For JSON/table-like generated outputs, each outcome object or field group
    should remain one block. This prevents values such as 'P < 0.05' from being
    separated from the outcome they belong to.
    """

    block_id: str
    source: Literal["generated"]
    category: str
    block_name: str
    evidence_text: str
    extracted_subfacts: List[str]
    source_section: str = ""
    chunk_id: Optional[str] = None


@dataclass
class CandidateMatch:
    """Retrieved generated block for a GT evaluation item."""

    candidate_id: str
    item_id: str
    block_id: str
    item_index: int
    block_index: int
    similarity: float
    lexical_score: float
    retrieval_method: str  # embedding, lexical_fallback, embedding+lexical


@dataclass
class ItemJudgement:
    """LLM provisional item-level judgement."""

    candidate_id: str
    item_id: str
    block_id: str
    label: ItemLabel
    confidence: float
    rationale: str
    critical_difference: str
    subfact_statuses: List[Dict[str, Any]]
    similarity: float
    lexical_score: float
    retrieval_method: str




def get_openai_client() -> OpenAI:
    """Create an OpenAI client from environment variables only."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it as an environment variable or Colab Secret."
        )
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def normalise_space(text: Any) -> str:
    """Collapse repeated whitespace and convert null-like values to an empty string."""
    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def safe_category(category: Any) -> str:
    """Normalise category to the controlled set."""
    c = normalise_space(category).lower().replace(" ", "_")
    return c if c in ALLOWED_CATEGORIES else "other"


def read_docx_text(path: str | Path) -> str:
    """Read paragraphs and table text from a DOCX file."""
    doc = Document(str(path))
    parts: List[str] = []

    for para in doc.paragraphs:
        text = normalise_space(para.text)
        if text:
            parts.append(text)

    for table in doc.tables:
        for row in table.rows:
            cells = [normalise_space(cell.text) for cell in row.cells]
            cells = [c for c in cells if c]
            if cells:
                parts.append(" | ".join(cells))

    return "\n".join(parts).strip()


def read_extraction_text(path: str | Path) -> str:
    """Read extraction text from DOCX or plain-text/JSON-like files."""
    path = Path(path)
    try:
        return read_docx_text(path)
    except Exception:
        pass

    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return path.read_text(encoding=enc).strip()
        except Exception:
            continue

    raise RuntimeError(f"Could not read file as DOCX or text: {path}")


def chunk_text(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    """Split text into overlapping chunks without cutting too aggressively."""
    text = str(text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 1
        if current and current_len + para_len > max_chars:
            chunk = "\n".join(current)
            chunks.append(chunk)
            overlap = chunk[-overlap_chars:] if overlap_chars > 0 else ""
            current = [overlap, para] if overlap else [para]
            current_len = len(overlap) + para_len
        else:
            current.append(para)
            current_len += para_len

    if current:
        chunks.append("\n".join(current))
    return chunks


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity for two embedding matrices."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)))
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a_norm @ b_norm.T


def safe_json_dump(obj: Any, path: str | Path) -> None:
    """Write JSON with readable formatting."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_cell(value: Any, fallback: str = "") -> str:
    """Return a display-safe cell string; no literal NaN in dashboards."""
    text = normalise_space(value)
    return text if text else fallback





@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def llm_json(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Call a chat-completions model and parse a JSON-object response."""
    kwargs: Dict[str, Any] = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception:
        # Some reasoning models reject temperature.
        kwargs.pop("temperature", None)
        response = client.chat.completions.create(**kwargs)

    content = response.choices[0].message.content or "{}"
    return json.loads(content)




GT_ITEM_EXTRACTION_SYSTEM = """
You are an expert scientific information extractor for clinical evidence evaluation.

Task:
Convert the supplied HUMAN GROUND-TRUTH extraction into evaluation items / claim groups.

Important:
The evaluation item is the PRIMARY metric unit. Do not over-atomise into many tiny
rows when a field is naturally one meaningful clinical item, such as an intervention
protocol, outcome set, result set, eligibility criteria set, or methods item.

Rules:
1. Extract all required information from the GT extraction.
2. Group tightly related details into one evaluation item when they belong to one
   clinical extraction field or one natural claim group.
   Examples:
   - Intervention protocol with dose, frequency, PFMT, EMS settings = one item.
   - Urodynamic p-values for MUP, MCUP, PTR% = one item with subfacts.
   - Blinding components may be one item if the source reports them together.
3. Use `required_subfacts` to preserve important details inside each item.
4. The `canonical_fact` should be a concise complete statement describing the item.
5. The `evidence_text` must preserve the FULL original bullet/field/section text
   from the GT extraction. Do not shorten it to only one subfact.
6. If the GT says not specified/not stated/not reported/not provided/not detailed,
   preserve that as an item, because it affects review.
7. Do not infer from outside knowledge. Use only the supplied GT text.
8. Assign exactly one category from the allowed categories.
9. Prefer fewer, clinically meaningful evaluation items over excessive micro-facts,
   while still listing detailed required subfacts.

Allowed categories:
study_design, setting, population, sample_size, allocation, blinding,
inclusion_criteria, exclusion_criteria, baseline_characteristics,
intervention, comparator, outcome, time_point, statistical_method,
effect_size, p_value, statistical_result, funding, conclusion, other.

Return JSON only:
{
  "items": [
    {
      "category": "intervention",
      "item_name": "Intervention protocol",
      "canonical_fact": "The intervention consisted of HT plus PFMT plus EMS, with estriol ovules daily for 2 weeks then twice weekly.",
      "required_subfacts": [
        "The intervention included HT.",
        "The intervention included PFMT.",
        "The intervention included EMS.",
        "Estriol ovules were administered daily for 2 weeks.",
        "Estriol ovules were administered twice per week thereafter."
      ],
      "evidence_text": "Full original GT bullet/field/section text.",
      "source_section": "Nearest heading or section, or empty string."
    }
  ]
}
""".strip()


GENERATED_BLOCK_EXTRACTION_SYSTEM = """
You are an expert scientific information extractor for clinical evidence evaluation.

Task:
Convert the supplied GENERATED extraction into evidence blocks/items for matching
against ground-truth evaluation items.

Important:
Preserve context. Do NOT split structured JSON/table/list outcome objects into
contextless fragments. Numeric/statistical values such as p-values must remain
bound to their outcome description, group, time point, and comparison when present.

Rules:
1. Extract generated evidence blocks at the level they are reported:
   - one intervention paragraph/object = one block,
   - one comparator paragraph/object = one block,
   - one outcome/result JSON object or table row = one block,
   - one methods/risk-of-bias field = one block.
2. `evidence_text` must contain the full original block, field, table row, JSON
   object, or paragraph. Do not shorten it.
3. `extracted_subfacts` should list compact details present in the block for display
   and judge support, but matching should still use the full evidence block.
4. Preserve values exactly: p-values, units, frequencies, sample sizes, dose,
   duration, outcome abbreviations, and group labels.
5. Do not infer beyond the generated extraction.
6. Assign exactly one category from the allowed categories.

Allowed categories:
study_design, setting, population, sample_size, allocation, blinding,
inclusion_criteria, exclusion_criteria, baseline_characteristics,
intervention, comparator, outcome, time_point, statistical_method,
effect_size, p_value, statistical_result, funding, conclusion, other.

Return JSON only:
{
  "blocks": [
    {
      "category": "statistical_result",
      "block_name": "Secondary outcome result: MUP/MUCP/PTR",
      "evidence_text": "Full generated object/row/paragraph containing outcome, time point, and p-value.",
      "extracted_subfacts": [
        "MUP increased.",
        "MUCP increased.",
        "PTR increased.",
        "Comparison of outcomes was P < 0.05."
      ],
      "source_section": "Nearest heading or section, or empty string."
    }
  ]
}
""".strip()




def _dedupe_by_key(items: List[Dict[str, Any]], fields: Iterable[str]) -> List[Dict[str, Any]]:
    """Deduplicate extraction outputs using selected fields."""
    seen = set()
    unique: List[Dict[str, Any]] = []
    for item in items:
        key = " | ".join(normalise_space(item.get(f, "")).lower() for f in fields)
        if not key.strip() or key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def extract_gt_evaluation_items(
    text: str,
    config: EvalConfig,
    client: Optional[OpenAI] = None,
) -> List[EvaluationItem]:
    """Extract grouped GT evaluation items from the reference extraction."""
    client = client or get_openai_client()
    chunks = chunk_text(text, config.max_chunk_chars, config.chunk_overlap_chars)
    raw_items: List[Dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        user = f"""
Source type: ground_truth
Chunk ID: gt_chunk_{i+1:03d}

Ground-truth extraction text:
{chunk}
""".strip()
        data = llm_json(client, config.extraction_model, GT_ITEM_EXTRACTION_SYSTEM, user, config.temperature)
        items = data.get("items", [])
        if not isinstance(items, list):
            items = []
        for item in items:
            if isinstance(item, dict):
                item["chunk_id"] = f"gt_chunk_{i+1:03d}"
                raw_items.append(item)

    raw_items = _dedupe_by_key(raw_items, ["category", "item_name", "canonical_fact", "evidence_text"])

    out: List[EvaluationItem] = []
    for idx, item in enumerate(raw_items, start=1):
        subfacts = item.get("required_subfacts", [])
        if not isinstance(subfacts, list):
            subfacts = []
        subfacts = [normalise_space(s) for s in subfacts if normalise_space(s)]
        canonical = normalise_space(item.get("canonical_fact", ""))
        evidence = normalise_space(item.get("evidence_text", ""))
        if not canonical and evidence:
            canonical = evidence
        if not canonical:
            continue
        out.append(
            EvaluationItem(
                item_id=f"GTI_{idx:04d}",
                source="ground_truth",
                category=safe_category(item.get("category", "other")),
                item_name=normalise_space(item.get("item_name", f"Evaluation item {idx}")),
                canonical_fact=canonical,
                required_subfacts=subfacts,
                evidence_text=evidence,
                source_section=normalise_space(item.get("source_section", "")),
                chunk_id=normalise_space(item.get("chunk_id", "")) or None,
            )
        )
    return out


def extract_generated_evidence_blocks(
    text: str,
    config: EvalConfig,
    client: Optional[OpenAI] = None,
) -> List[GeneratedEvidenceBlock]:
    """Extract context-preserving generated evidence blocks."""
    client = client or get_openai_client()
    chunks = chunk_text(text, config.max_chunk_chars, config.chunk_overlap_chars)
    raw_blocks: List[Dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        user = f"""
Source type: generated
Chunk ID: generated_chunk_{i+1:03d}

Generated extraction text:
{chunk}
""".strip()
        data = llm_json(client, config.extraction_model, GENERATED_BLOCK_EXTRACTION_SYSTEM, user, config.temperature)
        blocks = data.get("blocks", [])
        if not isinstance(blocks, list):
            blocks = []
        for block in blocks:
            if isinstance(block, dict):
                block["chunk_id"] = f"generated_chunk_{i+1:03d}"
                raw_blocks.append(block)

    raw_blocks = _dedupe_by_key(raw_blocks, ["category", "block_name", "evidence_text"])

    out: List[GeneratedEvidenceBlock] = []
    for idx, block in enumerate(raw_blocks, start=1):
        subfacts = block.get("extracted_subfacts", [])
        if not isinstance(subfacts, list):
            subfacts = []
        subfacts = [normalise_space(s) for s in subfacts if normalise_space(s)]
        evidence = normalise_space(block.get("evidence_text", ""))
        if not evidence:
            continue
        out.append(
            GeneratedEvidenceBlock(
                block_id=f"GENB_{idx:04d}",
                source="generated",
                category=safe_category(block.get("category", "other")),
                block_name=normalise_space(block.get("block_name", f"Generated block {idx}")),
                evidence_text=evidence,
                extracted_subfacts=subfacts,
                source_section=normalise_space(block.get("source_section", "")),
                chunk_id=normalise_space(block.get("chunk_id", "")) or None,
            )
        )
    return out





def item_retrieval_text(item: EvaluationItem) -> str:
    """Context-rich retrieval text for a GT evaluation item."""
    return normalise_space(
        "\n".join(
            [
                f"Category: {item.category}",
                f"Section: {item.source_section}",
                f"Evaluation item: {item.item_name}",
                f"Canonical fact: {item.canonical_fact}",
                "Required subfacts: " + "; ".join(item.required_subfacts),
                f"Evidence: {item.evidence_text}",
            ]
        )
    )


def block_retrieval_text(block: GeneratedEvidenceBlock) -> str:
    """Context-rich retrieval text for a generated evidence block."""
    return normalise_space(
        "\n".join(
            [
                f"Category: {block.category}",
                f"Section: {block.source_section}",
                f"Generated block: {block.block_name}",
                f"Evidence block: {block.evidence_text}",
                "Extracted subfacts: " + "; ".join(block.extracted_subfacts),
            ]
        )
    )


def embed_texts_openai(texts: List[str], model: str, client: Optional[OpenAI] = None) -> np.ndarray:
    """Embed texts using OpenAI embeddings."""
    if not texts:
        return np.zeros((0, 1), dtype=float)
    client = client or get_openai_client()
    response = client.embeddings.create(model=model, input=texts)
    return np.array([item.embedding for item in response.data], dtype=float)


def embed_texts_voyage(texts: List[str], model: str) -> np.ndarray:
    """Embed texts using Voyage AI embeddings."""
    if not texts:
        return np.zeros((0, 1), dtype=float)
    if not os.getenv("VOYAGE_API_KEY"):
        raise RuntimeError("VOYAGE_API_KEY is not set.")
    try:
        import voyageai  # type: ignore
    except Exception as exc:
        raise ImportError("Install Voyage client first: pip install voyageai") from exc
    vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    result = vo.embed(texts, model=model, input_type="document")
    return np.array(result.embeddings, dtype=float)


def embed_texts(texts: List[str], config: EvalConfig, client: Optional[OpenAI] = None) -> np.ndarray:
    """Embed texts with the configured provider."""
    if config.embedding_provider == "openai":
        return embed_texts_openai(texts, config.openai_embedding_model, client=client)
    if config.embedding_provider == "voyage":
        return embed_texts_voyage(texts, config.voyage_embedding_model)
    raise ValueError(f"Unsupported embedding provider: {config.embedding_provider}")


def important_tokens(text: str) -> set[str]:
    """Extract lexical tokens useful for fallback retrieval.

    This intentionally keeps abbreviations, p-value forms, numbers, units, and
    all-caps clinical outcome labels such as MUP, MCUP, MUCP, PTR.
    """
    text = normalise_space(text)
    text_upper = text.upper()
    tokens: set[str] = set()

    # Abbreviations and all-caps tokens.
    for tok in re.findall(r"\b[A-Z]{2,}[A-Z0-9%]*\b", text_upper):
        tokens.add(tok)

    # p-value forms.
    for m in re.findall(r"P\s*[<=>≤≥]\s*0?\.\d+", text_upper):
        tokens.add(re.sub(r"\s+", "", m))
    if re.search(r"\bP\b|P-VALUE|PVALUE", text_upper):
        tokens.add("P_VALUE")

    # Numeric values, units, frequencies.
    for tok in re.findall(r"\b\d+(?:\.\d+)?\s*(?:HZ|MS|MA|MG|MIN|MINUTES|MONTHS|WEEKS|DAYS|%|X/WEEK|/WEEK)?\b", text_upper):
        tokens.add(re.sub(r"\s+", "", tok))

    # Useful domain words.
    domain_words = {
        "MUP", "MCUP", "MUCP", "PTR", "PAD", "TEST", "URINARY", "LEAKAGE",
        "ESTRIOL", "OVULE", "OVULES", "PFMT", "EMS", "HT", "SUI", "QUALITY", "LIFE",
        "SIGNIFICANT", "INCREASE", "DECREASE", "IMPROVEMENT", "CONTROL", "INTERVENTION",
    }
    for w in domain_words:
        if re.search(rf"\b{re.escape(w)}\b", text_upper):
            tokens.add(w)

    return tokens


def lexical_overlap_score(query: str, candidate: str) -> float:
    """Compute a conservative lexical overlap score for fallback retrieval."""
    q = important_tokens(query)
    c = important_tokens(candidate)
    if not q or not c:
        return 0.0
    return len(q & c) / max(1, len(q))


def category_threshold(category: str, config: EvalConfig) -> float:
    """Lower embedding threshold for sparse/numeric/statistical categories."""
    return config.sparse_min_candidate_similarity if category in SPARSE_CATEGORIES else config.min_candidate_similarity


def retrieve_candidate_matches(
    gt_items: List[EvaluationItem],
    gen_blocks: List[GeneratedEvidenceBlock],
    config: EvalConfig,
    client: Optional[OpenAI] = None,
) -> Tuple[List[CandidateMatch], np.ndarray, np.ndarray]:
    """Retrieve generated blocks for each GT item.

    Candidate retrieval uses context-rich embeddings plus a lexical fallback for
    short statistical/numeric/abbreviation-heavy items. Similarity and lexical
    scores are retrieval aids only; final metrics never use them directly.
    """
    if not gt_items or not gen_blocks:
        return [], np.zeros((len(gt_items), len(gen_blocks))), np.zeros((len(gt_items), len(gen_blocks)))

    item_texts = [item_retrieval_text(x) for x in gt_items]
    block_texts = [block_retrieval_text(x) for x in gen_blocks]

    item_emb = embed_texts(item_texts, config, client=client)
    block_emb = embed_texts(block_texts, config, client=client)
    sim = cosine_similarity_matrix(item_emb, block_emb)

    lex = np.zeros_like(sim, dtype=float)
    for i, itxt in enumerate(item_texts):
        for j, btxt in enumerate(block_texts):
            lex[i, j] = lexical_overlap_score(itxt, btxt)

    candidates: List[CandidateMatch] = []
    seen: set[Tuple[int, int]] = set()

    for i, item in enumerate(gt_items):
        threshold = category_threshold(item.category, config)
        # Embedding candidates.
        embed_order = np.argsort(sim[i])[::-1][: max(config.top_k, 1)]
        for j in embed_order:
            if sim[i, j] >= threshold:
                seen.add((i, int(j)))
                method = "embedding+lexical" if lex[i, j] >= config.lexical_min_score else "embedding"
                candidates.append(
                    CandidateMatch(
                        candidate_id=f"C_{len(candidates)+1:05d}",
                        item_id=item.item_id,
                        block_id=gen_blocks[int(j)].block_id,
                        item_index=i,
                        block_index=int(j),
                        similarity=float(sim[i, j]),
                        lexical_score=float(lex[i, j]),
                        retrieval_method=method,
                    )
                )

        # Lexical fallback candidates for sparse/statistical/numeric items or when no embedding hit.
        has_candidate_for_item = any(c.item_index == i for c in candidates)
        if item.category in SPARSE_CATEGORIES or not has_candidate_for_item:
            lex_order = np.argsort(lex[i])[::-1][: max(config.lexical_fallback_top_k, 1)]
            for j in lex_order:
                j = int(j)
                if (i, j) in seen:
                    continue
                if lex[i, j] >= config.lexical_min_score:
                    seen.add((i, j))
                    candidates.append(
                        CandidateMatch(
                            candidate_id=f"C_{len(candidates)+1:05d}",
                            item_id=item.item_id,
                            block_id=gen_blocks[j].block_id,
                            item_index=i,
                            block_index=j,
                            similarity=float(sim[i, j]),
                            lexical_score=float(lex[i, j]),
                            retrieval_method="lexical_fallback",
                        )
                    )

    return candidates, sim, lex




ITEM_JUDGE_SYSTEM = """
You are an expert evaluator of clinical information extraction.

Task:
Judge whether a GENERATED extraction block captures a GROUND-TRUTH evaluation item.
The ground truth is the reference. The generated evidence is from the generated
extraction, not from the original study paper. Do not use outside knowledge.

Allowed item-level labels:
- agree: the generated block sufficiently preserves the material meaning of the GT item.
- contradict: the generated block clearly conflicts with the GT item on a material detail.
- cannot_verify: the generated block is related but too ambiguous/incomplete/specific to safely classify as agree or contradict.

Use cannot_verify when:
- GT says not specified/not reported/not stated and generated provides a concrete value;
- the generated block has extra specificity not verifiable from the GT item;
- the generated wording may be compatible but materially ambiguous;
- a p-value/statistical value is present but its scope across outcomes is unclear;
- subfacts are mixed: some present, some absent/uncertain, and the item-level conclusion is not safe.

Use agree only when enough required subfacts are present and no material conflict exists.
Use contradict only when there is a clear factual conflict, not merely missing detail.
If the generated block omits an important required subfact but does not contradict it,
mark that subfact as absent and use cannot_verify or contradict depending on materiality.

Return JSON only:
{
  "label": "agree | contradict | cannot_verify",
  "confidence": 0.0,
  "rationale": "brief rationale, no more than 70 words",
  "critical_difference": "specific material conflict/uncertainty/missing detail, or null",
  "subfact_statuses": [
    {
      "subfact": "required GT subfact text",
      "status": "present | absent | contradicted | cannot_verify",
      "generated_support": "short phrase from generated block or empty string"
    }
  ]
}
""".strip()


def judge_item_candidate(
    item: EvaluationItem,
    block: GeneratedEvidenceBlock,
    candidate: CandidateMatch,
    config: EvalConfig,
    client: Optional[OpenAI] = None,
) -> ItemJudgement:
    """Judge one GT item against one retrieved generated block."""
    client = client or get_openai_client()
    user = f"""
GROUND-TRUTH EVALUATION ITEM
Item ID: {item.item_id}
Category: {item.category}
Item name: {item.item_name}
Canonical fact: {item.canonical_fact}
Required subfacts:
{json.dumps(item.required_subfacts, ensure_ascii=False, indent=2)}
GT evidence span:
{item.evidence_text}

GENERATED EVIDENCE BLOCK
Block ID: {block.block_id}
Category: {block.category}
Block name: {block.block_name}
Generated evidence span:
{block.evidence_text}
Generated extracted subfacts for context:
{json.dumps(block.extracted_subfacts, ensure_ascii=False, indent=2)}

Retrieval metadata, for context only and not scoring:
Embedding similarity: {candidate.similarity:.4f}
Lexical overlap score: {candidate.lexical_score:.4f}
Retrieval method: {candidate.retrieval_method}
""".strip()
    data = llm_json(client, config.judge_model, ITEM_JUDGE_SYSTEM, user, config.temperature)

    label = normalise_space(data.get("label", "cannot_verify")).lower()
    if label not in {"agree", "contradict", "cannot_verify"}:
        label = "cannot_verify"

    try:
        confidence = float(data.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    subfact_statuses = data.get("subfact_statuses", [])
    if not isinstance(subfact_statuses, list):
        subfact_statuses = []
    cleaned_subfacts: List[Dict[str, Any]] = []
    allowed_status = {"present", "absent", "contradicted", "cannot_verify"}
    for sf in subfact_statuses:
        if not isinstance(sf, dict):
            continue
        status = normalise_space(sf.get("status", "cannot_verify")).lower()
        if status not in allowed_status:
            status = "cannot_verify"
        cleaned_subfacts.append(
            {
                "subfact": normalise_space(sf.get("subfact", "")),
                "status": status,
                "generated_support": normalise_space(sf.get("generated_support", "")),
            }
        )

    # Ensure every required subfact appears in diagnostic output.
    existing = {x["subfact"].lower() for x in cleaned_subfacts if x.get("subfact")}
    for sf in item.required_subfacts:
        if sf.lower() not in existing:
            cleaned_subfacts.append({"subfact": sf, "status": "cannot_verify", "generated_support": ""})

    return ItemJudgement(
        candidate_id=candidate.candidate_id,
        item_id=item.item_id,
        block_id=block.block_id,
        label=label,  # type: ignore[arg-type]
        confidence=confidence,
        rationale=normalise_space(data.get("rationale", "")),
        critical_difference=normalise_space(data.get("critical_difference", "")),
        subfact_statuses=cleaned_subfacts,
        similarity=candidate.similarity,
        lexical_score=candidate.lexical_score,
        retrieval_method=candidate.retrieval_method,
    )


def judge_all_candidates(
    gt_items: List[EvaluationItem],
    gen_blocks: List[GeneratedEvidenceBlock],
    candidates: List[CandidateMatch],
    config: EvalConfig,
    client: Optional[OpenAI] = None,
) -> List[ItemJudgement]:
    """Judge all retrieved candidates."""
    client = client or get_openai_client()
    items_by_id = {x.item_id: x for x in gt_items}
    blocks_by_id = {x.block_id: x for x in gen_blocks}
    judgements: List[ItemJudgement] = []

    for cand in candidates:
        item = items_by_id.get(cand.item_id)
        block = blocks_by_id.get(cand.block_id)
        if item is None or block is None:
            continue
        judgements.append(judge_item_candidate(item, block, cand, config, client=client))
    return judgements


def choose_best_judgement(judgements: List[ItemJudgement]) -> Optional[ItemJudgement]:
    """Choose the best provisional judgement for one GT item.

    Prefer clear agreement, then cannot_verify, then contradiction. Within label
    tiers, prioritise confidence, similarity, and lexical score.
    """
    if not judgements:
        return None
    label_rank = {"agree": 3, "cannot_verify": 2, "contradict": 1}
    return sorted(
        judgements,
        key=lambda j: (
            label_rank.get(j.label, 0),
            j.confidence,
            j.similarity,
            j.lexical_score,
        ),
        reverse=True,
    )[0]





UNSPECIFIED_PATTERNS = re.compile(
    r"\b(not specified|not stated|not reported|not provided|not detailed|unclear|nr|n/r)\b",
    flags=re.IGNORECASE,
)


def is_unspecified_text(text: str) -> bool:
    """Detect GT statements indicating absence of reported information."""
    return bool(UNSPECIFIED_PATTERNS.search(normalise_space(text)))


def has_concrete_content(text: str) -> bool:
    """Heuristic: generated block gives a concrete value/detail."""
    text = normalise_space(text)
    if not text:
        return False
    if is_unspecified_text(text):
        return False
    return bool(re.search(r"\d|\b(yes|no|daily|weekly|months|weeks|hz|mg|ma|p\s*[<=>])\b", text, flags=re.I)) or len(text.split()) >= 5


def subfact_summary(subfact_statuses: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count subfact statuses."""
    counts = {"present": 0, "absent": 0, "contradicted": 0, "cannot_verify": 0}
    for sf in subfact_statuses or []:
        status = normalise_space(sf.get("status", "cannot_verify")).lower()
        if status in counts:
            counts[status] += 1
        else:
            counts["cannot_verify"] += 1
    return counts


def row_type_for_item(
    item: EvaluationItem,
    block: Optional[GeneratedEvidenceBlock],
    judgement: Optional[ItemJudgement],
    config: EvalConfig,
) -> str:
    """Assign dashboard row type for one GT item."""
    if block is None or judgement is None:
        return "missing_gt_item"

    if config.flag_gt_unspecified_gen_concrete and is_unspecified_text(item.canonical_fact + " " + item.evidence_text) and has_concrete_content(block.evidence_text):
        return "gt_unspecified_gen_concrete"

    counts = subfact_summary(judgement.subfact_statuses)
    if counts["present"] > 0 and (counts["absent"] > 0 or counts["cannot_verify"] > 0 or counts["contradicted"] > 0):
        return "partial_subfact_coverage"

    if judgement.label == "cannot_verify":
        return "llm_cannot_verify"

    if config.flag_low_confidence and judgement.confidence < config.low_confidence_threshold:
        return "low_confidence_match"

    if judgement.label == "agree":
        return "matched_agree"
    if judgement.label == "contradict":
        return "direct_conflict"
    return "requires_review"


def review_reason_for(row_type: str) -> str:
    """Human-readable explanation for dashboard row type."""
    reasons = {
        "missing_gt_item": "No plausible generated evidence block was found for this GT evaluation item.",
        "gt_unspecified_gen_concrete": "GT states information is unspecified/not reported, but generated extraction gives a concrete value; expert should decide.",
        "partial_subfact_coverage": "Generated block covers some required subfacts but not all; expert should inspect subfact statuses.",
        "llm_cannot_verify": "LLM could not safely classify the generated block as agreement or contradiction.",
        "low_confidence_match": "LLM judgement has low confidence; expert should confirm or override.",
        "matched_agree": "LLM judged this GT evaluation item as captured by the generated extraction.",
        "direct_conflict": "LLM judged this generated block as materially conflicting with the GT evaluation item.",
        "requires_review": "System flagged this row for expert review.",
    }
    return reasons.get(row_type, "Expert should confirm or override.")


def build_item_level_review_dashboard(
    gt_items: List[EvaluationItem],
    gen_blocks: List[GeneratedEvidenceBlock],
    candidates: List[CandidateMatch],
    judgements: List[ItemJudgement],
    config: EvalConfig,
) -> pd.DataFrame:
    """Build exactly one review row per GT evaluation item.

    Generated-only extra blocks are intentionally ignored in the main dashboard.
    """
    judgements_by_item: Dict[str, List[ItemJudgement]] = {}
    for j in judgements:
        judgements_by_item.setdefault(j.item_id, []).append(j)

    blocks_by_id = {b.block_id: b for b in gen_blocks}
    rows: List[Dict[str, Any]] = []

    for idx, item in enumerate(gt_items, start=1):
        best = choose_best_judgement(judgements_by_item.get(item.item_id, []))
        block = blocks_by_id.get(best.block_id) if best else None

        if best is None or block is None:
            llm_label: FinalLabel = "missing"
            llm_confidence: Optional[float] = None
            llm_rationale = "No generated evidence block passed retrieval threshold or lexical fallback."
            critical_difference = "No matching generated evidence block found."
            similarity: Optional[float] = None
            lexical_score: Optional[float] = None
            retrieval_method = "none"
            subfact_statuses = [
                {"subfact": sf, "status": "absent", "generated_support": ""}
                for sf in item.required_subfacts
            ]
        else:
            llm_label = best.label  # type: ignore[assignment]
            llm_confidence = best.confidence
            llm_rationale = best.rationale
            critical_difference = best.critical_difference
            similarity = best.similarity
            lexical_score = best.lexical_score
            retrieval_method = best.retrieval_method
            subfact_statuses = best.subfact_statuses

        counts = subfact_summary(subfact_statuses)
        total_subfacts = sum(counts.values())
        subfact_coverage = counts["present"] / total_subfacts if total_subfacts else None
        row_type = row_type_for_item(item, block, best, config)

        rows.append(
            {
                "review_id": f"R_{idx:05d}",
                "row_type": row_type,
                "category": item.category,
                "gt_item_id": item.item_id,
                "gt_item_name": item.item_name,
                "gt_canonical_fact": item.canonical_fact,
                "gt_required_subfacts": json.dumps(item.required_subfacts, ensure_ascii=False),
                "gt_evidence_text": item.evidence_text,
                "gt_source_section": item.source_section,
                "matched_generated_block_id": block.block_id if block else "",
                "generated_block_name": block.block_name if block else "No generated candidate found",
                "generated_evidence_text": block.evidence_text if block else "No generated candidate found",
                "generated_extracted_subfacts": json.dumps(block.extracted_subfacts, ensure_ascii=False) if block else "[]",
                "generated_source_section": block.source_section if block else "",
                "similarity_retrieval_only": similarity,
                "lexical_score_retrieval_only": lexical_score,
                "retrieval_method": retrieval_method,
                "llm_label": llm_label,
                "llm_confidence": llm_confidence,
                "llm_rationale": llm_rationale,
                "critical_difference": critical_difference,
                "subfact_statuses_json": json.dumps(subfact_statuses, ensure_ascii=False),
                "subfacts_present": counts["present"],
                "subfacts_absent": counts["absent"],
                "subfacts_contradicted": counts["contradicted"],
                "subfacts_cannot_verify": counts["cannot_verify"],
                "subfact_coverage_diagnostic": subfact_coverage,
                "review_reason": review_reason_for(row_type),
                # Human-editable fields.
                "expert_decision": "",
                "expert_comment": "",
                # Filled during adjudication.
                "final_label": "",
                "included_in_completeness": "",
                "included_in_correctness": "",
            }
        )

    df = pd.DataFrame(rows)
    if config.fill_missing_display_values:
        df = df.fillna("")
    return df





ALLOWED_EXPERT_DECISIONS = {"", "agree_with_llm", "agree", "contradict", "cannot_verify"}
ALLOWED_FINAL_LABELS = {"agree", "contradict", "cannot_verify", "missing"}


def normalise_expert_decision(value: Any) -> str:
    """Normalise user-entered expert decisions."""
    value = normalise_space(value).lower()
    aliases = {
        "accept_llm": "agree_with_llm",
        "accept": "agree_with_llm",
        "same_as_llm": "agree_with_llm",
        "llm": "agree_with_llm",
        "agreed": "agree",
        "yes": "agree",
        "correct": "agree",
        "contradiction": "contradict",
        "disagree": "contradict",
        "incorrect": "contradict",
        "cannot verify": "cannot_verify",
        "can't verify": "cannot_verify",
        "uncertain": "cannot_verify",
        "unknown": "cannot_verify",
        "not sure": "cannot_verify",
    }
    return aliases.get(value, value)


def derive_final_label(llm_label: Any, expert_decision: Any) -> FinalLabel:
    """Resolve final label from LLM/system label and optional expert decision."""
    llm = normalise_space(llm_label).lower()
    if llm not in ALLOWED_FINAL_LABELS:
        llm = "cannot_verify"

    decision = normalise_expert_decision(expert_decision)
    if decision == "":
        return llm  # type: ignore[return-value]
    if decision == "agree_with_llm":
        return llm  # type: ignore[return-value]
    if decision in {"agree", "contradict", "cannot_verify"}:
        return decision  # type: ignore[return-value]

    raise ValueError(
        f"Invalid expert_decision={decision!r}. Allowed values: "
        "agree_with_llm, agree, contradict, cannot_verify, or blank."
    )


def adjudicate_review_dashboard(review_df: pd.DataFrame) -> pd.DataFrame:
    """Add final labels and metric-inclusion flags."""
    df = review_df.copy().fillna("")
    final_labels: List[str] = []
    inc_complete: List[bool] = []
    inc_correct: List[bool] = []

    for _, row in df.iterrows():
        final = derive_final_label(row.get("llm_label", "cannot_verify"), row.get("expert_decision", ""))
        final_labels.append(final)
        inc_complete.append(final in {"agree", "contradict", "missing"})
        inc_correct.append(final in {"agree", "contradict"})

    df["final_label"] = final_labels
    df["included_in_completeness"] = inc_complete
    df["included_in_correctness"] = inc_correct
    return df


def compute_final_metrics_from_review(review_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute item-level metrics from expert-adjudicated/provisional dashboard.

    Blank expert decisions are allowed: the LLM/system label is used. cannot_verify
    rows are excluded from metric denominators and reported separately.
    """
    df = adjudicate_review_dashboard(review_df)
    counts = df["final_label"].value_counts(dropna=False).to_dict()
    agree = int(counts.get("agree", 0))
    contradict = int(counts.get("contradict", 0))
    missing = int(counts.get("missing", 0))
    cannot_verify = int(counts.get("cannot_verify", 0))

    completeness_den = agree + contradict + missing
    correctness_den = agree + contradict

    completeness = agree / completeness_den if completeness_den else None
    correctness = agree / correctness_den if correctness_den else None

    expert_completed = int(df["expert_decision"].astype(str).str.strip().ne("").sum())
    total_rows = int(len(df))

    # Diagnostic subfact coverage: not the primary metric.
    subfact_cols = ["subfacts_present", "subfacts_absent", "subfacts_contradicted", "subfacts_cannot_verify"]
    subfact_diag: Dict[str, Any] = {}
    if all(c in df.columns for c in subfact_cols):
        present = int(pd.to_numeric(df["subfacts_present"], errors="coerce").fillna(0).sum())
        absent = int(pd.to_numeric(df["subfacts_absent"], errors="coerce").fillna(0).sum())
        contradicted = int(pd.to_numeric(df["subfacts_contradicted"], errors="coerce").fillna(0).sum())
        sf_cv = int(pd.to_numeric(df["subfacts_cannot_verify"], errors="coerce").fillna(0).sum())
        assessable_sf = present + absent + contradicted
        subfact_diag = {
            "subfacts_present": present,
            "subfacts_absent": absent,
            "subfacts_contradicted": contradicted,
            "subfacts_cannot_verify": sf_cv,
            "subfact_coverage_diagnostic": present / assessable_sf if assessable_sf else None,
            "subfact_coverage_policy": "diagnostic only; not used as primary completeness/correctness metric",
        }

    return {
        "evaluation_mode": "gt_focused_item_level_reference_evaluation",
        "primary_metric_unit": "GT evaluation item / claim group",
        "total_gt_items": total_rows,
        "final_label_counts": {
            "agree": agree,
            "contradict": contradict,
            "missing": missing,
            "cannot_verify": cannot_verify,
        },
        "completeness": completeness,
        "completeness_formula": "agree / (agree + contradict + missing)",
        "completeness_numerator": agree,
        "completeness_denominator": completeness_den,
        "correctness": correctness,
        "correctness_formula": "agree / (agree + contradict)",
        "correctness_numerator": agree,
        "correctness_denominator": correctness_den,
        "cannot_verify_count": cannot_verify,
        "cannot_verify_policy": "excluded from completeness and correctness denominators; reported separately",
        "expert_review_completed_rows": expert_completed,
        "expert_review_total_rows": total_rows,
        "expert_review_completion_rate": expert_completed / total_rows if total_rows else None,
        **subfact_diag,
    }





def dataframe_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataframe for Excel/HTML display; no literal NaN cells."""
    return df.copy().fillna("").replace({np.nan: ""})


def save_evaluation_outputs(
    output_dir: str | Path,
    study_id: str,
    gt_items: List[EvaluationItem],
    gen_blocks: List[GeneratedEvidenceBlock],
    candidates: List[CandidateMatch],
    judgements: List[ItemJudgement],
    review_df: pd.DataFrame,
    config: EvalConfig,
) -> Dict[str, str]:
    """Save JSON, Excel, CSV, and HTML outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = output_dir / study_id
    review_adjudicated = adjudicate_review_dashboard(review_df)
    provisional_metrics = compute_final_metrics_from_review(review_df)

    paths = {
        "json": str(prefix.with_name(prefix.name + "_evaluation_results.json")),
        "excel": str(prefix.with_name(prefix.name + "_evaluation_audit.xlsx")),
        "review_csv": str(prefix.with_name(prefix.name + "_human_review_dashboard.csv")),
        "review_xlsx": str(prefix.with_name(prefix.name + "_human_review_dashboard.xlsx")),
        "html": str(prefix.with_name(prefix.name + "_evaluation_report.html")),
    }

    serialised = {
        "study_id": study_id,
        "evaluation_mode": "gt_focused_item_level_reference_evaluation",
        "config": asdict(config),
        "metrics_provisional_or_expert_adjudicated": provisional_metrics,
        "ground_truth_items": [asdict(x) for x in gt_items],
        "generated_blocks": [asdict(x) for x in gen_blocks],
        "candidate_matches": [asdict(x) for x in candidates],
        "item_judgements": [asdict(x) for x in judgements],
        "human_review_dashboard": review_adjudicated.to_dict(orient="records"),
    }
    safe_json_dump(serialised, paths["json"])

    review_clean = dataframe_for_excel(review_adjudicated)
    review_clean.to_csv(paths["review_csv"], index=False)

    summary_df = pd.DataFrame(
        [
            {"metric": k, "value": json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v}
            for k, v in provisional_metrics.items()
        ]
    )
    gt_df = pd.DataFrame([asdict(x) for x in gt_items])
    gen_df = pd.DataFrame([asdict(x) for x in gen_blocks])
    cand_df = pd.DataFrame([asdict(x) for x in candidates])
    judge_df = pd.DataFrame([asdict(x) for x in judgements])

    for p in [paths["excel"], paths["review_xlsx"]]:
        with pd.ExcelWriter(p, engine="openpyxl") as writer:
            dataframe_for_excel(summary_df).to_excel(writer, sheet_name="summary_metrics", index=False)
            review_clean.to_excel(writer, sheet_name="human_review_dashboard", index=False)
            dataframe_for_excel(gt_df).to_excel(writer, sheet_name="gt_evaluation_items", index=False)
            dataframe_for_excel(gen_df).to_excel(writer, sheet_name="generated_blocks", index=False)
            dataframe_for_excel(cand_df).to_excel(writer, sheet_name="candidate_matches", index=False)
            dataframe_for_excel(judge_df).to_excel(writer, sheet_name="item_judgements", index=False)

    html = f"""
    <html>
    <head>
      <meta charset='utf-8'>
      <title>{study_id} extraction evaluation report</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 28px; }}
        table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
        th, td {{ border: 1px solid #ddd; padding: 6px; vertical-align: top; }}
        th {{ background: #f3f3f3; }}
        .note {{ background: #fff8dc; padding: 10px; border: 1px solid #e0d28a; }}
      </style>
    </head>
    <body>
      <h1>{study_id} extraction evaluation report</h1>
      <div class='note'>Primary metric unit: GT evaluation item / claim group. Generated-only extra facts are ignored by default. cannot_verify rows are excluded from completeness and correctness denominators.</div>
      <h2>Summary metrics</h2>
      {summary_df.to_html(index=False, escape=False)}
      <h2>Human review dashboard</h2>
      {review_clean.to_html(index=False, escape=False)}
    </body>
    </html>
    """
    Path(paths["html"]).write_text(html, encoding="utf-8")

    return paths




def evaluate_extraction_pair_item_level(
    groundtruth_docx_path: str | Path,
    generated_docx_path: str | Path,
    output_dir: str | Path,
    study_id: str,
    config: Optional[EvalConfig] = None,
    client: Optional[OpenAI] = None,
) -> Dict[str, Any]:
    """Run the full GT-focused item-level evaluation.

    There is intentionally no source_docx_path. This framework validates the
    generated extraction against the GT extraction only.
    """
    config = config or EvalConfig()
    client = client or get_openai_client()

    gt_text = read_extraction_text(groundtruth_docx_path)
    gen_text = read_extraction_text(generated_docx_path)

    gt_items = extract_gt_evaluation_items(gt_text, config, client=client)
    gen_blocks = extract_generated_evidence_blocks(gen_text, config, client=client)

    candidates, similarity_matrix, lexical_matrix = retrieve_candidate_matches(
        gt_items, gen_blocks, config, client=client
    )
    judgements = judge_all_candidates(gt_items, gen_blocks, candidates, config, client=client)

    review_df = build_item_level_review_dashboard(
        gt_items, gen_blocks, candidates, judgements, config
    )
    metrics = compute_final_metrics_from_review(review_df)
    paths = save_evaluation_outputs(
        output_dir, study_id, gt_items, gen_blocks, candidates, judgements, review_df, config
    )

    return {
        "study_id": study_id,
        "metrics": metrics,
        "output_paths": paths,
        "ground_truth_items": [asdict(x) for x in gt_items],
        "generated_blocks": [asdict(x) for x in gen_blocks],
        "candidate_matches": [asdict(x) for x in candidates],
        "item_judgements": [asdict(x) for x in judgements],
        "human_review_dashboard": review_df.to_dict(orient="records"),
        "similarity_matrix_shape": list(similarity_matrix.shape),
        "lexical_matrix_shape": list(lexical_matrix.shape),
    }





def load_review_dashboard(path: str | Path) -> pd.DataFrame:
    """Load a human review dashboard from CSV/XLSX."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name="human_review_dashboard").fillna("")
    return pd.read_csv(path).fillna("")


def _metrics_summary_dataframe(metrics: Dict[str, Any]) -> pd.DataFrame:
    """Convert nested metrics dict into a human-readable summary table."""
    rows: List[Dict[str, Any]] = []
    for key, value in metrics.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                rows.append({"metric": f"{key}.{sub_key}", "value": sub_value})
        else:
            rows.append({"metric": key, "value": value})
    return pd.DataFrame(rows)


def _subfact_diagnostics_dataframe(review_df: pd.DataFrame) -> pd.DataFrame:
    """Expand JSON subfact statuses into one diagnostic row per subfact."""
    rows: List[Dict[str, Any]] = []
    df = review_df.copy().fillna("")
    for _, row in df.iterrows():
        raw = row.get("subfact_statuses_json", "[]")
        try:
            subfacts = json.loads(raw) if isinstance(raw, str) and raw.strip() else []
        except Exception:
            subfacts = []
        if not isinstance(subfacts, list):
            subfacts = []
        for sf in subfacts:
            if not isinstance(sf, dict):
                continue
            rows.append(
                {
                    "review_id": row.get("review_id", ""),
                    "category": row.get("category", ""),
                    "gt_item_id": row.get("gt_item_id", ""),
                    "gt_item_name": row.get("gt_item_name", ""),
                    "final_label": row.get("final_label", ""),
                    "subfact": sf.get("subfact", ""),
                    "subfact_status": sf.get("status", ""),
                    "generated_support": sf.get("generated_support", ""),
                }
            )
    return pd.DataFrame(rows)


def save_final_adjudicated_metrics(
    review_df: pd.DataFrame,
    output_dir: str | Path,
    study_id: str,
) -> Dict[str, str]:
    """Save final metrics after expert review using the preferred output contract.

    This function is intended to be called after the expert has edited the review
    dashboard. Blank expert decisions are allowed: the system uses the LLM/system
    provisional label. The saved CSV records all expert changes and the derived
    final labels. The JSON contains the final metrics, label counts, and output
    file locations.

    Returned paths match the earlier GT-focused notebook style:
        {"csv": ..., "excel": ..., "json": ...}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adjudicated = adjudicate_review_dashboard(review_df)
    metrics = compute_final_metrics_from_review(adjudicated)
    label_counts = metrics.get("final_label_counts", {})

    paths = {
        "csv": str(output_dir / f"{study_id}_final_adjudicated_review.csv"),
        "excel": str(output_dir / f"{study_id}_final_adjudicated_metrics.xlsx"),
        "json": str(output_dir / f"{study_id}_final_adjudicated_metrics.json"),
    }

    # Ensure display/export never shows literal NaN.
    adjudicated_clean = dataframe_for_excel(adjudicated)
    adjudicated_clean.to_csv(paths["csv"], index=False)

    summary_df = _metrics_summary_dataframe(metrics)
    label_counts_df = pd.DataFrame(
        [{"label": k, "count": v} for k, v in dict(label_counts).items()]
    )
    subfact_df = _subfact_diagnostics_dataframe(adjudicated_clean)

    cannot_verify_df = adjudicated_clean[adjudicated_clean["final_label"].astype(str).eq("cannot_verify")]
    missing_df = adjudicated_clean[adjudicated_clean["final_label"].astype(str).eq("missing")]
    contradiction_df = adjudicated_clean[adjudicated_clean["final_label"].astype(str).eq("contradict")]
    agree_df = adjudicated_clean[adjudicated_clean["final_label"].astype(str).eq("agree")]

    with pd.ExcelWriter(paths["excel"], engine="openpyxl") as writer:
        dataframe_for_excel(summary_df).to_excel(writer, sheet_name="summary_metrics", index=False)
        adjudicated_clean.to_excel(writer, sheet_name="final_adjudicated_review", index=False)
        dataframe_for_excel(subfact_df).to_excel(writer, sheet_name="subfact_diagnostics", index=False)
        dataframe_for_excel(label_counts_df).to_excel(writer, sheet_name="label_counts", index=False)
        dataframe_for_excel(cannot_verify_df).to_excel(writer, sheet_name="cannot_verify_items", index=False)
        dataframe_for_excel(missing_df).to_excel(writer, sheet_name="missing_items", index=False)
        dataframe_for_excel(contradiction_df).to_excel(writer, sheet_name="contradiction_items", index=False)
        dataframe_for_excel(agree_df).to_excel(writer, sheet_name="agree_items", index=False)

    payload = {
        "study_id": study_id,
        "evaluation_mode": metrics.get("evaluation_mode", "gt_focused_item_level_reference_evaluation"),
        "primary_metric_unit": metrics.get("primary_metric_unit", "GT evaluation item / claim group"),
        "metrics": metrics,
        "label_counts": label_counts,
        "output_files": paths,
        "review_records": adjudicated_clean.to_dict(orient="records"),
    }
    safe_json_dump(payload, paths["json"])

    return paths


def save_adjudicated_metrics(
    review_df: pd.DataFrame,
    output_dir: str | Path,
    study_id: str,
) -> Dict[str, str]:
    """Backward-compatible alias for saving final expert-adjudicated outputs.

    Returns the preferred keys: csv, excel, json.
    """
    return save_final_adjudicated_metrics(review_df=review_df, output_dir=output_dir, study_id=study_id)



def launch_expert_review_dashboard(review_df: pd.DataFrame, save_csv_path: str | Path):
    """Simple row-by-row Colab expert review widget.

    Returns an editable dataframe object. If ipywidgets is unavailable, returns
    the dataframe and prints instructions for editing CSV/XLSX manually.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import clear_output, display
    except Exception:
        print("ipywidgets is not available. Edit the CSV/XLSX dashboard manually instead.")
        return review_df

    df = review_df.copy().fillna("")
    decisions = ["", "agree_with_llm", "agree", "contradict", "cannot_verify"]
    idx_state = {"idx": 0}

    out = widgets.Output()
    dropdown = widgets.Dropdown(options=decisions, description="Decision:")
    comment = widgets.Textarea(description="Comment:", layout=widgets.Layout(width="100%", height="80px"))
    prev_btn = widgets.Button(description="Previous")
    next_btn = widgets.Button(description="Save & Next", button_style="primary")
    save_btn = widgets.Button(description="Save CSV", button_style="success")

    def render():
        with out:
            clear_output()
            i = idx_state["idx"]
            row = df.iloc[i]
            dropdown.value = row.get("expert_decision", "") if row.get("expert_decision", "") in decisions else ""
            comment.value = str(row.get("expert_comment", ""))
            print(f"Review row {i+1} of {len(df)}: {row.get('review_id', '')}")
            print(f"Row type: {row.get('row_type', '')}")
            print(f"Category: {row.get('category', '')}")
            print(f"LLM label: {row.get('llm_label', '')}")
            print(f"LLM confidence: {row.get('llm_confidence', '')}")
            print(f"Review reason: {row.get('review_reason', '')}")
            print("\nGROUND TRUTH ITEM")
            print(row.get("gt_item_name", ""))
            print(row.get("gt_canonical_fact", ""))
            print("\nGROUND TRUTH REQUIRED SUBFACTS")
            try:
                for sf in json.loads(row.get("gt_required_subfacts", "[]")):
                    print(f"- {sf}")
            except Exception:
                print(row.get("gt_required_subfacts", ""))
            print("\nGROUND TRUTH EVIDENCE SPAN")
            print(row.get("gt_evidence_text", ""))
            print("\nGENERATED EVIDENCE BLOCK")
            print(row.get("generated_block_name", ""))
            print(row.get("generated_evidence_text", ""))
            print("\nSUBFACT STATUSES")
            try:
                for sf in json.loads(row.get("subfact_statuses_json", "[]")):
                    print(f"- [{sf.get('status')}] {sf.get('subfact')} :: {sf.get('generated_support')}")
            except Exception:
                print(row.get("subfact_statuses_json", ""))
            print("\nLLM rationale")
            print(row.get("llm_rationale", ""))
            print("\nCritical difference / uncertainty")
            print(row.get("critical_difference", ""))

    def persist_current():
        i = idx_state["idx"]
        df.at[df.index[i], "expert_decision"] = dropdown.value
        df.at[df.index[i], "expert_comment"] = comment.value
        Path(save_csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv_path, index=False)

    def on_next(_):
        persist_current()
        idx_state["idx"] = min(len(df) - 1, idx_state["idx"] + 1)
        render()

    def on_prev(_):
        persist_current()
        idx_state["idx"] = max(0, idx_state["idx"] - 1)
        render()

    def on_save(_):
        persist_current()
        with out:
            print(f"\nSaved dashboard to: {save_csv_path}")

    next_btn.on_click(on_next)
    prev_btn.on_click(on_prev)
    save_btn.on_click(on_save)

    controls = widgets.VBox([out, dropdown, comment, widgets.HBox([prev_btn, next_btn, save_btn])])
    display(controls)
    render()
    return df
