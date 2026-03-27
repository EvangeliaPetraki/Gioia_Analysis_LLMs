"""
WP5 Gioia First- + Second-Order Extractor (Chutes)
--------------------------------------------------
- Reads PDFs from an input folder
- Chunks text
- Calls Chutes LLM per chunk (STRICT JSON for Gioia 1st-order concepts)
- Merges first-order concepts at document level
- Calls Chutes LLM once per document to generate 2nd-order themes from merged 1st-order concepts
- Writes ONE PDF PER INPUT PDF into an output folder
"""

import os
import re
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import requests
import pdfplumber
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
import random
from analysis.env_config import get_bool, get_float, get_int, get_list, get_path, get_str

# ----------------------------
# Config
# ----------------------------
MAX_RETRIES = get_int("GIOIA_SECOND_ORDER_MAX_RETRIES", 8)
BASE_BACKOFF_S = get_float("GIOIA_SECOND_ORDER_BASE_BACKOFF_S", 2.0)
MAX_BACKOFF_S = get_float("GIOIA_SECOND_ORDER_MAX_BACKOFF_S", 60.0)
CHARS_PER_CHUNK = get_int("CHARS_PER_CHUNK", 4000)
SLEEP_BETWEEN_CALLS = get_float("SLEEP_BETWEEN_CALLS", 0.5)

CHUTES_API_KEY = get_str("CHUTES_API_KEY", "")
CHUTES_BASE_URL = get_str("CHUTES_BASE_URL", "https://llm.chutes.ai/v1")
CHUTES_MODEL = get_str(
    "GIOIA_SECOND_ORDER_MODEL",
    get_str("CHUTES_MODEL", "Qwen/Qwen2.5-72B-Instruct"),
)

# Prefer hot / high-availability models for batch jobs
FALLBACK_MODELS = get_list(
    "GIOIA_SECOND_ORDER_FALLBACK_MODELS",
    [
        CHUTES_MODEL,
        "Qwen/Qwen3-32B",
        "mistral/Mistral-Nemo-Instruct-2407",
        "deepseek/DeepSeek-V3-0324",
    ],
)

CAPACITY_STATUS = {429, 500, 503}


# Controls PDF verbosity:
INCLUDE_CHUNK_OUTPUTS = get_bool("GIOIA_SECOND_ORDER_INCLUDE_CHUNK_OUTPUTS", False)
INCLUDE_MERGED_RECORD = get_bool("GIOIA_SECOND_ORDER_INCLUDE_MERGED_RECORD", True)
INCLUDE_SECOND_ORDER = get_bool("GIOIA_SECOND_ORDER_INCLUDE_SECOND_ORDER", True)

# 2nd-order call settings
SECOND_ORDER_MAX_TOKENS = get_int("SECOND_ORDER_MAX_TOKENS", 2500)
SECOND_ORDER_RETRIES = get_int("SECOND_ORDER_RETRIES", 3)

# ----------------------------
# Prompts (Gioia Stage 1)
# ----------------------------
SYSTEM_GIOIA_FIRST_ORDER = """You are assisting a Horizon Europe research project (WP5 – European labour market resilience management). You are acting as a structured qualitative coding assistant. The researcher retains interpretive authority.

We are applying the Gioia methodology (Gioia, Corley & Hamilton, 2013).
In THIS step, you must extract FIRST-ORDER CONCEPTS from the provided policy text.

Definition (operational):
First-order concepts are:
- Close to the language of the policy document
- Descriptive rather than interpretive
- Reflecting how the policy frames problems, objectives, instruments, target groups, governance, or transition challenges
- Not abstract theoretical categories
- Not grouped into broader themes
- Not evaluated or critiqued

Important distinctions:
- Do NOT create second-order themes.
- Do NOT summarise the whole document.
- Each concept must represent one distinct intervention logic or policy-relevant idea.
- Do not artificially split a single intervention into multiple concepts.
- Ignore section headings, figure titles, and numbered section labels.

Each concept must:
- Have a short label (3–10 words)
- Be supported by one short verbatim quote from the text (<=25 words)
- Stay faithful to the document’s meaning

RELEVANCE FILTER (CRITICAL):
Extract first-order concepts ONLY if they relate to:
- Labour market impacts, employment, skills, reskilling, upskilling
- Governance mechanisms for green or digital transition
- Policy instruments, funding tools, regulatory measures
- Social inclusion, inequality, vulnerable groups
- Resilience, adaptability, shock absorption, transition management
- Regional or sectoral transition challenges
- Institutional or regulatory change related to labour markets

DO NOT extract:
- Literature titles or bibliographic references
- Section headings or numbering
- Methodological descriptions of data analysis
- Database descriptions
- Annex information
- Website/contact details
- Copyright or legal disclaimers
- Generic background information without policy relevance

Granularity rule:
If multiple statements refer to the same intervention logic, governance mechanism, or policy instrument, consolidate them into ONE first-order concept.
Do not create multiple concepts that only differ in wording but refer to the same type of action.
Prefer slightly broader first-order labels when several sentences describe the same intervention.
If the text contains repeated ideas, only include meaningfully distinct concepts.

PER-CHUNK DISCIPLINE RULES:
You are analysing ONE CHUNK of a larger policy document.
- Do NOT attempt to infer content outside this chunk.
- Extract only concepts clearly supported within this chunk.
- Do NOT try to anticipate later sections of the document.
- Do NOT aim for exhaustiveness within this chunk.

Concept density control:
- Prefer 5–12 high-quality, non-redundant concepts per chunk.
- If the chunk contains limited policy-relevant material, return fewer concepts.
- If no relevant policy content appears, return an empty list.

ADDITIONAL METADATA EXTRACTION:
Extract document metadata ONLY if explicitly stated; otherwise set to null/unknown.

OUTPUT SCHEMA (STRICT JSON ONLY):
{
  "chunk_id": int,
  "doc_id": string|null,
  "source_pages": {"start": int|null, "end": int|null},
  "document_metadata": {
    "document_type": "policy_instrument"|"policy_report"|"research_paper"|"discussion_paper"|"technical_study"|"case_study"|"other"|"unknown",
    "governance_level": "supranational"|"national"|"regional"|"local"|"sectoral"|"unknown",
    "country": string|null,
    "region": string|null,
    "issuing_body": string|null,
    "year": int|null
  },
  "first_order_concepts": [
    {
      "label": string,
      "type": "objective"|"measure"|"instrument"|"target_group"|"governance"|"funding"|"problem_framing"|"resilience_framing"|"twin_transition_green"|"twin_transition_digital"|"social_inclusion"|"monitoring_evaluation"|"other",
      "quote": string
    }
  ]
}

Hard constraints:
- label max 80 chars
- quote max 240 chars and <=25 words (approx.)
- first_order_concepts max 30
- Output must start with '{' and end with '}'.
"""

USER_GIOIA_TEMPLATE = """chunk_id: {chunk_id}
doc_id: {doc_id}
pages: {page_start}-{page_end}
text:
<<<
{chunk_text}
>>>
"""


# ----------------------------
# Prompts (Gioia Stage 2)
# ----------------------------
SYSTEM_GIOIA_SECOND_ORDER = """You are assisting a Horizon Europe research project (WP5 – European labour market resilience management).

We are applying the Gioia methodology (Gioia, Corley & Hamilton, 2013).

In THIS step, you must derive SECOND-ORDER THEMES from an existing list of FIRST-ORDER CONCEPTS.

Method:
- First-order concepts are document-centric and close-to-text.
- Second-order themes are researcher-centric and capture underlying mechanisms, organising logics, or governance patterns across the first-order concepts.
- Do NOT create aggregate dimensions in this step.

STRICT RULES:
- Use ONLY the first-order concepts provided (do not invent new concepts).
- Each second-order theme must be supported by at least 3 first-order concepts.
- Return 5–12 second-order themes (fewer if only fewer strong clusters exist).
- Do not evaluate or critique policies.
- Do not summarise the whole document.
- Ensure themes are not duplicates and do not overlap heavily.
- Output MUST be valid JSON. Do not include markdown, code fences, or any text outside the JSON object.

WP5 analytical orientation (use these lenses when clustering):
- Governance coordination mechanisms
- Skills intelligence systems
- Curriculum/qualification reform logics
- Employer involvement and social dialogue
- Reskilling/upskilling pathways
- Inclusion/distributional mechanisms
- Transition-management / resilience mechanisms
- Digital + green integration logics

OUTPUT SCHEMA (STRICT JSON ONLY):
{
  "doc_id": string,
  "second_order_themes": [
    {
      "theme_label": string,
      "mechanism_explanation": string,
      "first_order_concept_ids": [int, int, int],
      "first_order_concept_labels": [string, string, string]
    }
  ]
}

Hard constraints:
- theme_label max 80 chars
- mechanism_explanation max 600 chars
- second_order_themes max 12
- Output must start with '{' and end with '}'.
"""

USER_SECOND_ORDER_TEMPLATE = """doc_id: {doc_id}

FIRST-ORDER CONCEPTS (numbered). Each item includes: label | type | quote.
{numbered_concepts}
"""


# ----------------------------
# JSON repair prompt
# ----------------------------
SYSTEM_JSON_REPAIR = """You are a strict JSON repair tool.
You will receive text that is intended to contain ONE JSON object but may be invalid, truncated, or wrapped.
Return a single VALID JSON object only.
Rules:
- Output must start with '{' and end with '}'.
- No markdown, no commentary.
- Fix quoting/escaping issues as needed.
- If multiple JSON objects appear, return the most complete one.
If repair is impossible, return: {"error":"unrepairable_json"}.
"""


# ----------------------------
# JSON parsing helpers
# ----------------------------

def _parse_http_status_from_error(e: Exception) -> Optional[int]:
    m = re.search(r"Chutes HTTP (\d{3})", str(e))
    return int(m.group(1)) if m else None


def _is_capacity_error(e: Exception) -> bool:
    msg = (str(e) or "").lower()
    status = _parse_http_status_from_error(e)

    if status in CAPACITY_STATUS:
        return (
            "maximum capacity" in msg
            or "try again later" in msg
            or "no instances available" in msg
            or "infrastructure" in msg
        )
    return False


def _exp_backoff_with_jitter(attempt_idx: int,
                             base: float = BASE_BACKOFF_S,
                             cap: float = MAX_BACKOFF_S) -> float:
    delay = min(cap, base * (2 ** attempt_idx))
    return delay * (0.7 + 0.6 * random.random())

def _extract_json_obj(text: str) -> Dict[str, Any]:
    """
    Extract the FIRST valid JSON object found anywhere in the text.
    This is more robust than assuming the JSON is the whole response
    or only at the end.
    """
    text = (text or "").strip()

    # Fast path: whole text is JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Scan for balanced {...} blocks and try parsing
    start_positions = [m.start() for m in re.finditer(r"\{", text)]
    for start in start_positions:
        depth = 0
        for end in range(start, len(text)):
            ch = text[end]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:end+1].strip()
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break  # this block isn't parseable; try next '{'
    raise ValueError("No valid JSON object found in model output.")



def safe_parse_json(text: str) -> Dict[str, Any]:
    return _extract_json_obj(text)


# ----------------------------
# Chutes LLM call
# ----------------------------
def llm_raw(
    system_prompt: str,
    user_prompt: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 3500
) -> str:
    if not CHUTES_API_KEY:
        raise RuntimeError("Set CHUTES_API_KEY in your environment.")

    url = f"{CHUTES_BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": model or CHUTES_MODEL,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {CHUTES_API_KEY}",
        "Content-Type": "application/json",
    }

    r = requests.post(url, headers=headers, json=payload, timeout=120)

    if not r.ok:
        with open("last_http_error.txt", "w", encoding="utf-8") as f:
            f.write(f"STATUS: {r.status_code}\n")
            f.write(r.text)
        raise RuntimeError(f"Chutes HTTP {r.status_code}: {r.text[:2000]}")

    data = r.json()
    return data["choices"][0]["message"]["content"]


# def llm_json(
#     system_prompt: str,
#     user_prompt: str,
#     *,
#     model: Optional[str] = None,
#     temperature: float = 0.0,
#     max_tokens: int = 3500,
#     retries: int = 2,
#     sleep_s: float = 0.8
# ) -> Dict[str, Any]:
#     last_err = None

#     for attempt in range(1, retries + 1):
#         text = ""
#         try:
#             text = llm_raw(system_prompt, user_prompt, model=model, temperature=temperature, max_tokens=max_tokens)

#             try:
#                 return safe_parse_json(text)
#             except Exception:
#                 with open("last_bad_output.txt", "w", encoding="utf-8") as f:
#                     f.write(text)

#                 repaired = llm_raw(
#                     SYSTEM_JSON_REPAIR,
#                     f"TEXT:\n<<<\n{text}\n>>>",
#                     model=model,
#                     temperature=0.0,
#                     max_tokens=max_tokens
#                 )
#                 with open("last_repaired_output.txt", "w", encoding="utf-8") as f:
#                     f.write(repaired)
#                 return safe_parse_json(repaired)

#         except Exception as e:
#             last_err = e
#             if text:
#                 with open("last_exception_output.txt", "w", encoding="utf-8") as f:
#                     f.write(text)
#             time.sleep(sleep_s * attempt)

#     raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}")

def llm_json(
    system_prompt: str,
    user_prompt: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 3500,
    retries: int = MAX_RETRIES,
    sleep_s: Optional[float] = None,  # <-- ADD THIS
) -> Dict[str, Any]:

    last_err: Optional[Exception] = None

    # Rotate models across attempts
    models_to_try = [model] if model else []
    models_to_try += [m for m in FALLBACK_MODELS if m and m not in models_to_try]

    attempt = 0

    while attempt < retries:
        chosen_model = models_to_try[attempt % len(models_to_try)]
        text = ""

        try:
            text = llm_raw(
                system_prompt,
                user_prompt,
                model=chosen_model,
                temperature=temperature,
                max_tokens=max_tokens
            )

            try:
                return safe_parse_json(text)

            except Exception:
                # Save bad output
                with open("last_bad_output.txt", "w", encoding="utf-8") as f:
                    f.write(text)

                repaired = llm_raw(
                    SYSTEM_JSON_REPAIR,
                    f"TEXT:\n<<<\n{text}\n>>>",
                    model=chosen_model,
                    temperature=0.0,
                    max_tokens=max_tokens
                )

                with open("last_repaired_output.txt", "w", encoding="utf-8") as f:
                    f.write(repaired)

                return safe_parse_json(repaired)

        except Exception as e:
            last_err = e

            if text:
                with open("last_exception_output.txt", "w", encoding="utf-8") as f:
                    f.write(text)

            # Capacity-aware handling
            if _is_capacity_error(e):
                delay = _exp_backoff_with_jitter(attempt)
                print(
                    f"[WARN] Capacity error on model={chosen_model}. "
                    f"Backing off {delay:.1f}s "
                    f"(attempt {attempt+1}/{retries})"
                )
                time.sleep(delay)
            else:
                delay = min(5.0, 0.8 * (attempt + 1))
                if sleep_s is not None:
                    delay = max(delay, float(sleep_s))
                print(
                    f"[WARN] Non-capacity error on model={chosen_model}. "
                    f"Sleeping {delay:.1f}s "
                    f"(attempt {attempt+1}/{retries})"
                )
                time.sleep(delay)

            attempt += 1

    raise RuntimeError(
        f"LLM call failed after {retries} attempts: {last_err}"
    )

# ----------------------------
# PDF text extraction + chunking
# ----------------------------
def clean_text(t: str) -> str:
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


def extract_pdf_pages(pdf_path: str) -> List[Tuple[int, str]]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            pages.append((i, txt))
    return pages


def chunk_pages(pages: List[Tuple[int, str]], max_chars: int = CHARS_PER_CHUNK):
    buf: List[Tuple[int, str]] = []
    buf_len = 0
    start_page: Optional[int] = None

    for pno, txt in pages:
        txt = clean_text(txt or "")
        if not txt:
            continue

        if start_page is None:
            start_page = pno

        if buf_len + len(txt) > max_chars and buf:
            yield {
                "page_start": start_page,
                "page_end": buf[-1][0],
                "text": "\n\n".join([f"[PAGE {pno}]\n{ptxt}" for (pno, ptxt) in buf]),
            }
            buf, buf_len = [], 0
            start_page = pno

        buf.append((pno, txt))
        buf_len += len(txt)

    if buf:
        yield {
            "page_start": start_page,
            "page_end": buf[-1][0],
            "text": "\n\n".join([f"[PAGE {pno}]\n{ptxt}" for (pno, ptxt) in buf]),
        }


# ----------------------------
# Merge + helpers
# ----------------------------
def merge_chunk_concepts_locally(chunk_jsons: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge chunk outputs into one doc-level object:
    - Merge page ranges (min start, max end)
    - Dedupe concepts by normalized (label + type)
    - Keep first quote seen
    - Preserve document_metadata (from first non-empty metadata)
    """
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip().lower())

    doc_id = None
    min_page = None
    max_page = None

    merged_map: Dict[str, Dict[str, Any]] = {}
    doc_meta: Optional[Dict[str, Any]] = None

    for obj in chunk_jsons:
        if not isinstance(obj, dict):
            continue

        if not doc_id and obj.get("doc_id"):
            doc_id = obj.get("doc_id")

        if doc_meta is None and isinstance(obj.get("document_metadata"), dict):
            # pick first metadata dict we see
            doc_meta = obj.get("document_metadata")

        sp = obj.get("source_pages") or {}
        if isinstance(sp, dict):
            s = sp.get("start")
            e = sp.get("end")
            if isinstance(s, int):
                min_page = s if (min_page is None or s < min_page) else min_page
            if isinstance(e, int):
                max_page = e if (max_page is None or e > max_page) else max_page

        concepts = obj.get("first_order_concepts") or []
        if not isinstance(concepts, list):
            continue

        for c in concepts:
            if not isinstance(c, dict):
                continue
            label = c.get("label") or ""
            ctype = c.get("type") or "other"
            if not str(label).strip():
                continue
            key = f"{norm(label)}||{norm(ctype)}"
            if key not in merged_map:
                merged_map[key] = {
                    "label": str(label)[:80],
                    "type": ctype,
                    "quote": (c.get("quote") or "")[:240],
                }

    merged_list = sorted(
        merged_map.values(),
        key=lambda x: (str(x.get("type") or ""), str(x.get("label") or "").lower())
    )

    if doc_meta is None:
        doc_meta = {
            "document_type": "unknown",
            "governance_level": "unknown",
            "country": None,
            "region": None,
            "issuing_body": None,
            "year": None,
        }

    return {
        "doc_id": doc_id,
        "source_pages": {"start": min_page, "end": max_page},
        "document_metadata": doc_meta,
        "first_order_concepts": merged_list,
    }

def build_numbered_first_order_list(merged: Dict[str, Any]) -> str:
    concepts = merged.get("first_order_concepts") or []
    lines: List[str] = []
    for i, c in enumerate(concepts, start=1):
        label = (c.get("label") or "").strip()
        ctype = (c.get("type") or "other").strip()
        lines.append(f"{i}. {label} | {ctype}")
    return "\n".join(lines)



def generate_second_order_themes(doc_id: str, merged: Dict[str, Any]) -> Dict[str, Any]:
    numbered = build_numbered_first_order_list(merged)
    user_prompt = USER_SECOND_ORDER_TEMPLATE.format(doc_id=doc_id, numbered_concepts=numbered)

    return llm_json(
        SYSTEM_GIOIA_SECOND_ORDER,
        user_prompt,
        max_tokens=SECOND_ORDER_MAX_TOKENS,
        retries=SECOND_ORDER_RETRIES,
        sleep_s=0.9
    )


# ----------------------------
# PDF writing
# ----------------------------
def _wrap_lines(text: str, max_chars: int = 110):
    text = (text or "").strip()
    if not text:
        return []
    text = re.sub(r"\s+", " ", text)
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def write_report_pdf(
    source_pdf_path: Path,
    output_dir: Path,
    doc_id: str,
    merged: Optional[Dict[str, Any]],
    second_order: Optional[Dict[str, Any]],
    chunk_outputs: List[Dict[str, Any]],
    include_merged: bool = True,
    include_second_order: bool = True,
    include_chunks: bool = True,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = output_dir / f"{source_pdf_path.stem}_GIOIA_FIRST_SECOND_ORDER.pdf"

    c = canvas.Canvas(str(out_pdf), pagesize=A4)
    width, height = A4
    x = 2 * cm
    y = height - 2 * cm

    # Header
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, "WP5 Gioia — First- & Second-Order Coding (LLM-assisted)")
    y -= 18

    c.setFont("Helvetica", 10)
    intro = (
        "This report lists first-order concepts (close-to-text) and second-order themes (researcher-centric) "
        "produced using a structured Gioia-method workflow. Quotes provide traceability. "
        "No aggregate dimensions are produced here."
    )
    for ln in _wrap_lines(intro, max_chars=110):
        c.drawString(x, y, ln)
        y -= 12

    y -= 4
    c.drawString(x, y, f"Document ID: {doc_id}")
    y -= 14
    c.drawString(x, y, f"Source file: {source_pdf_path.name}")
    y -= 18

    # Document metadata (from merged)
    meta = (merged or {}).get("document_metadata") if merged else None
    if isinstance(meta, dict):
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x, y, "Document metadata (from model output):")
        y -= 14
        c.setFont("Helvetica", 10)
        lines = [
            f"Document type: {meta.get('document_type')}",
            f"Governance level: {meta.get('governance_level')}",
            f"Country: {meta.get('country') or '—'}",
            f"Region: {meta.get('region') or '—'}",
            f"Issuing body: {meta.get('issuing_body') or '—'}",
            f"Year: {meta.get('year') or '—'}",
        ]
        for ln in lines:
            if y < 4 * cm:
                c.showPage()
                y = height - 2 * cm
                c.setFont("Helvetica", 10)
            c.drawString(x, y, ln[:120])
            y -= 12
        y -= 6

    # 1st-order merged concepts
    if include_merged and merged and isinstance(merged.get("first_order_concepts"), list):
        concepts = merged.get("first_order_concepts") or []
        if y < 4 * cm:
            c.showPage()
            y = height - 2 * cm

        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y, "Document-Level First-Order Concepts (merged across chunks)")
        y -= 16

        c.setFont("Helvetica", 10)
        if not concepts:
            c.drawString(x, y, "No first-order concepts were extracted from this document.")
            y -= 14
        else:
            for idx, cc in enumerate(concepts, start=1):
                if y < 3.5 * cm:
                    c.showPage()
                    y = height - 2 * cm

                label = (cc.get("label") or "").strip()
                ctype = (cc.get("type") or "other").strip()
                quote = (cc.get("quote") or "").strip()

                c.setFont("Helvetica-Bold", 10)
                c.drawString(x, y, f"{idx}. {label}  [{ctype}]")
                y -= 12

                if quote:
                    c.setFont("Helvetica-Oblique", 9)
                    for ln in _wrap_lines(f"Evidence: “{quote}”", max_chars=105):
                        c.drawString(x + 12, y, ln)
                        y -= 11
                y -= 6

    # 2nd-order themes
    if include_second_order:
        if y < 6 * cm:
            c.showPage()
            y = height - 2 * cm

        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y, "Second-Order Themes (derived from merged first-order concepts)")
        y -= 16

        if not second_order or not isinstance(second_order.get("second_order_themes"), list):
            c.setFont("Helvetica", 10)
            c.drawString(x, y, "No second-order themes were generated (or output was invalid).")
            y -= 14
        else:
            themes = second_order.get("second_order_themes") or []
            for t_idx, t in enumerate(themes, start=1):
                if y < 4.5 * cm:
                    c.showPage()
                    y = height - 2 * cm

                label = (t.get("theme_label") or "").strip()
                expl = (t.get("mechanism_explanation") or "").strip()
                ids = t.get("first_order_concept_ids") or []
                lbls = t.get("first_order_concept_labels") or []

                c.setFont("Helvetica-Bold", 10)
                c.drawString(x, y, f"{t_idx}. {label}")
                y -= 12

                if expl:
                    c.setFont("Helvetica", 9)
                    for ln in _wrap_lines(expl, max_chars=105):
                        c.drawString(x + 12, y, ln)
                        y -= 11

                # Supporting concepts
                c.setFont("Helvetica-Oblique", 9)
                support_line = f"Supports (IDs): {', '.join([str(i) for i in ids])}" if ids else "Supports (IDs): —"
                for ln in _wrap_lines(support_line, max_chars=105):
                    c.drawString(x + 12, y, ln)
                    y -= 11

                if lbls:
                    c.setFont("Helvetica", 9)
                    c.drawString(x + 12, y, "Supporting concept labels:")
                    y -= 11
                    c.setFont("Helvetica", 8)
                    for lab in lbls[:12]:
                        if y < 3.0 * cm:
                            c.showPage()
                            y = height - 2 * cm
                            c.setFont("Helvetica", 8)
                        for ln in _wrap_lines(f"- {lab}", max_chars=100):
                            c.drawString(x + 18, y, ln)
                            y -= 10

                y -= 10

    # Chunk appendix
    if include_chunks:
        c.showPage()
        y = height - 2 * cm
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y, "Appendix — Chunk-Level Output (for validation)")
        y -= 16

        c.setFont("Helvetica", 10)
        c.drawString(x, y, "This section prints the raw JSON outputs per chunk.")
        y -= 14

        for i, ch in enumerate(chunk_outputs, start=1):
            if y < 4 * cm:
                c.showPage()
                y = height - 2 * cm

            c.setFont("Helvetica-Bold", 10)
            c.drawString(x, y, f"Chunk {i} | pages {ch['page_start']}-{ch['page_end']}")
            y -= 14

            extracted = ch.get("extracted") or {}
            dump = json.dumps(extracted, ensure_ascii=False, indent=2)

            c.setFont("Courier", 7)
            for ln in dump.splitlines():
                if y < 2.5 * cm:
                    c.showPage()
                    y = height - 2 * cm
                    c.setFont("Courier", 7)
                ln = ln[:160]
                c.drawString(x, y, ln)
                y -= 9

            y -= 10

    c.save()
    return out_pdf


# ----------------------------
# Main processing: one PDF -> one output PDF
# ----------------------------
def process_one_pdf(pdf_path: str, output_folder: str) -> Path:
    source_pdf = Path(pdf_path)
    doc_id = source_pdf.stem
    out_dir = Path(output_folder)

    out_pdf = out_dir / f"{source_pdf.stem}_GIOIA_FIRST_SECOND_ORDER.pdf"
    if out_pdf.exists():
        print(f"Skipping (already exists): {out_pdf.name}")
        return out_pdf

    pages = extract_pdf_pages(str(source_pdf))
    chunks = list(chunk_pages(pages))

    chunk_outputs: List[Dict[str, Any]] = []
    for i, ch in enumerate(chunks, start=1):
        user_prompt = USER_GIOIA_TEMPLATE.format(
            chunk_id=i,
            doc_id=doc_id,
            page_start=ch["page_start"],
            page_end=ch["page_end"],
            chunk_text=ch["text"],
        )

        extracted = llm_json(SYSTEM_GIOIA_FIRST_ORDER, user_prompt, max_tokens=3500)

        # Defensive defaults
        extracted.setdefault("chunk_id", i)
        extracted.setdefault("doc_id", doc_id)
        extracted.setdefault("source_pages", {"start": ch["page_start"], "end": ch["page_end"]})

        if not isinstance(extracted.get("document_metadata"), dict):
            extracted["document_metadata"] = {
                "document_type": "unknown",
                "governance_level": "unknown",
                "country": None,
                "region": None,
                "issuing_body": None,
                "year": None,
            }

        if not isinstance(extracted.get("first_order_concepts"), list):
            extracted["first_order_concepts"] = []

        chunk_outputs.append({
            "page_start": ch["page_start"],
            "page_end": ch["page_end"],
            "extracted": extracted,
        })
        time.sleep(SLEEP_BETWEEN_CALLS)

    extracted_list = [x["extracted"] for x in chunk_outputs]
    merged = merge_chunk_concepts_locally(extracted_list) if INCLUDE_MERGED_RECORD else None

    # 2nd-order themes (one call per document)
    second_order = None
    if INCLUDE_SECOND_ORDER and merged and isinstance(merged.get("first_order_concepts"), list):
        # If there are too few concepts, skip (prevents weak themes)
        if len(merged["first_order_concepts"]) >= 8:
            try:
                second_order = generate_second_order_themes(doc_id, merged)
                time.sleep(SLEEP_BETWEEN_CALLS)
            except Exception as e:
                print(f"[WARN] Second-order failed for {doc_id}: {e}")
                second_order = {"doc_id": doc_id, "second_order_themes": []}
        else:
            second_order = {"doc_id": doc_id, "second_order_themes": []}

    out_pdf = write_report_pdf(
        source_pdf_path=source_pdf,
        output_dir=out_dir,
        doc_id=doc_id,
        merged=merged,
        second_order=second_order,
        chunk_outputs=chunk_outputs,
        include_merged=INCLUDE_MERGED_RECORD,
        include_second_order=INCLUDE_SECOND_ORDER,
        include_chunks=INCLUDE_CHUNK_OUTPUTS,
    )

    print(f"Saved report PDF: {out_pdf}")
    return out_pdf


def process_folder(input_folder: str, output_folder: str):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    pdfs = sorted([p for p in input_folder.iterdir() if p.suffix.lower() == ".pdf"])
    print(f"Found {len(pdfs)} PDFs in {input_folder}")

    for p in pdfs:
        print(f"\nProcessing: {p.name}")
        process_one_pdf(str(p), str(output_folder))


# ----------------------------
# Entrypoint
# ----------------------------
if __name__ == "__main__":
    input_folder = get_path("GIOIA_SECOND_ORDER_INPUT_FOLDER")
    output_folder = get_path("GIOIA_SECOND_ORDER_OUTPUT_FOLDER")
    if input_folder is None or output_folder is None:
        raise RuntimeError(
            "Set GIOIA_SECOND_ORDER_INPUT_FOLDER and GIOIA_SECOND_ORDER_OUTPUT_FOLDER in .env."
        )
    process_folder(str(input_folder), str(output_folder))
