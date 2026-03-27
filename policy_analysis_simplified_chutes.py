"""
WP5 Simplified Extractor (Chutes)
--------------------------------
- Reads PDFs from an input folder
- Chunks text
- Calls Chutes LLM per chunk (JSON extraction)
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
import traceback
from analysis.env_config import get_bool, get_float, get_int, get_path, get_str



# ----------------------------
# Config
# ----------------------------
CHARS_PER_CHUNK = get_int("CHARS_PER_CHUNK", 4000)
SLEEP_BETWEEN_CALLS = get_float("SLEEP_BETWEEN_CALLS", 0.5)

CHUTES_API_KEY = get_str("CHUTES_API_KEY", "")
CHUTES_BASE_URL = get_str("CHUTES_BASE_URL", "https://llm.chutes.ai/v1")
CHUTES_MODEL = get_str(
    "POLICY_ANALYSIS_MODEL",
    get_str("CHUTES_MODEL", "Qwen/Qwen2.5-72B-Instruct"),
)

# Controls PDF verbosity:
INCLUDE_CHUNK_OUTPUTS = get_bool("POLICY_ANALYSIS_INCLUDE_CHUNK_OUTPUTS", False)
INCLUDE_MERGED_RECORD = get_bool("POLICY_ANALYSIS_INCLUDE_MERGED_RECORD", True)
INCLUDE_LLM_MERGE = get_bool("POLICY_ANALYSIS_INCLUDE_LLM_MERGE", False)

# ----------------------------
# Crash-safe checkpointing (JSONL)
# ----------------------------
def _checkpoint_paths(output_folder: str, doc_id: str) -> tuple[Path, Path]:
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_ok = out_dir / f"{doc_id}__chunk_checkpoint.jsonl"
    ckpt_err = out_dir / f"{doc_id}__chunk_errors.jsonl"
    return ckpt_ok, ckpt_err


def load_chunk_checkpoint(ckpt_path: Path) -> dict[int, dict]:
    """
    Returns {chunk_id: record} from a JSONL checkpoint file.
    Safe if file doesn't exist.
    """
    done: dict[int, dict] = {}
    if not ckpt_path.exists():
        return done

    with ckpt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                cid = rec.get("chunk_id")
                if isinstance(cid, int):
                    done[cid] = rec
            except Exception:
                # Ignore any corrupted line rather than breaking resume.
                continue
    return done


def append_jsonl(path: Path, record: dict) -> None:
    """
    Append one JSON line and flush immediately (so it's safe on crash).
    """
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ----------------------------
# Prompts (your WP5 schema)
# ----------------------------
SYSTEM_EXTRACT = """You are an expert EU labour market and skills policy analyst.
Extract ONLY information explicitly present in the provided text. Do NOT infer. ALL ANSWERS MUST BE IN ENGLISH.

Return STRICT JSON ONLY, no markdown, no explanations.

Output schema:
{
  "chunk_id": int,
"doc_id": string|null,
  "source_pages": {"start": int|null, "end": int|null},
  "items": [
    {
      "title": string|null,
      "issuing_body": string|null,
      "policy_level": "eu"|"national"|"regional"|"local"|null,
      "jurisdictions": [string],
      "year": int|null,
      "date_range": [string],
      "policy_type": [string],
      "instruments": [string],
      "twin_transition": {"green":[string], "digital":[string]},
      "objectives": [string],
      "measures": [string],
      "target_groups": [string],
      "sectors": [string],
      "funding": {"budget_amounts":[string], "funding_sources":[string]},
      "social_inclusion": {"inclusion_mechanisms":[string], "equality_dimensions":[string]},
      "monitoring_evaluation": {
        "evaluation_plan_present": boolean|null,
        "metrics_or_kpis":[string],
        "evidence_of_results":[string]
      },
      "quotes":[string]
    }
  ]
}

Rules:
- items = distinct policies/frameworks explicitly mentioned.
- Only include items that are NAMED policy initiatives / strategies / programmes / directives / regulations / packages.
  (Must contain a proper name or official title as written in the text; do NOT create generic "policy implications" items.)
- If the text only discusses policy ideas without naming a policy, return items: [] for that chunk.
- Prefer the exact official name in "title" (e.g., "European Green Deal", "Fit for 55", etc.) IF it appears verbatim in the text.
- HARD RULE: Only create an item if the title appears verbatim somewhere in the provided chunk text (case-insensitive). Otherwise, do NOT create the item.
- Put the most concrete evidence in quotes (verbatim). Include the [PAGE N] marker in the quote when possible.
- If none found, return items: [].
- quotes: max 3, <= 25 words, verbatim from text.
- Each list max 12 items; each string max 240 chars.
- Output must start with '{' and end with '}'.
"""

SYSTEM_DOC_SUMMARY = """You are summarising a policy document for a research work package.

TASK
Produce a short, descriptive summary of the document based ONLY on the extracted policy information provided.

RULES
- Use ONLY the information present in the input.
- Do NOT infer, evaluate, compare, or add external knowledge.
- Do NOT judge effectiveness or quality.
- Do NOT introduce new policy names.
- The summary must describe the document's focus in relation to:
  • labour markets
  • skills, reskilling, or upskilling
  • green and/or digital transitions (if present)

FORMAT
- 1 short paragraph (4–6 sentences).
- Plain, neutral academic language.
- No bullet points.
- No headings.
- No citations.

OUTPUT
Return plain text only.
"""

USER_EXTRACT_TEMPLATE = """chunk_id: {chunk_id}
doc_id: {doc_id}
pages: {page_start}-{page_end}
text:
<<<
{chunk_text}
>>>
"""


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

SYSTEM_MERGE = """You are a strict data-merging tool for policy extraction.
You will receive a JSON array of chunk extractions from the SAME document.
Each chunk extraction has: doc_id, source_pages, items[].

TASK:
- Produce ONE final JSON object with this schema:
{
  "doc_id": string|null,
  "source_pages": {"start": int|null, "end": int|null},
  "items": [ { ...policy object... } ]
}

MERGE RULES:
- Only keep items that are NAMED policies/strategies/programmes/directives/regulations/packages explicitly mentioned.
  (If generic policy ideas with no official name, drop them.)
- Deduplicate items referring to the same policy (normalize by title; if title missing, use issuing_body+year+policy_type).
- For each field: keep the most specific non-empty value; union list fields (cap 12).
- Keep quotes verbatim (cap 3 per item, <=25 words).
- Output STRICT JSON ONLY.
- HARD RULE: Keep an item ONLY if its title appears verbatim in at least one of the provided chunk items' quotes.
  If not, drop it
"""

# ----------------------------
# JSON parsing helpers
# ----------------------------
def _extract_json_obj(text: str) -> Dict[str, Any]:
    """
    Best-effort JSON extraction if the model returns extra text.
    """
    text = (text or "").strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to grab the last {...} block
    m = re.search(r"(\{.*\})\s*$", text, flags=re.S)
    if m:
        return json.loads(m.group(1))

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
    max_tokens: int = 6000
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

def llm_raw_with_autotrim(system_prompt: str, user_prompt: str, *, temperature: float = 0.0, max_tokens: int = 120) -> str:
    """
    Calls llm_raw; if we hit context-length errors, trims user_prompt and retries.
    This prevents 'off by 1 token' crashes.
    """
    prompt = user_prompt

    for _ in range(12):  # multiple trims if needed
        try:
            return llm_raw(system_prompt, prompt, temperature=temperature, max_tokens=max_tokens)
        except RuntimeError as e:
            msg = str(e)

            # Only handle context-length problems
            if ("context length" in msg) or ("maximum input length" in msg) or ("input_tokens" in msg):
                # Trim from end; remove a decent chunk to avoid repeated off-by-1 failures
                cut = 8000  # characters
                if len(prompt) <= cut + 200:
                    # Can't trim meaningfully anymore; re-raise
                    raise
                prompt = prompt[:-cut]
                continue

            # Other errors should still bubble up
            raise

    raise RuntimeError("Failed to fit prompt within context length after repeated trimming.")


def llm_json(
    system_prompt: str,
    user_prompt: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2500,
    retries: int = 2,
    sleep_s: float = 0.8
) -> Dict[str, Any]:
    """
    1 LLM call normally. If JSON parsing fails, 1 repair call.
    """
    last_err = None

    for attempt in range(1, retries + 1):
        text = ""
        try:
            text = llm_raw(system_prompt, user_prompt, model=model, temperature=temperature, max_tokens=max_tokens)

            try:
                return safe_parse_json(text)
            except Exception:
                # Save bad output for debugging
                with open("last_bad_output.txt", "w", encoding="utf-8") as f:
                    # f.write(text)
                    f.write("\n\n" + ("=" * 80) + "\n")
                    f.write(text)

                # Repair call (costs one more request; only happens when needed)
                repaired = llm_raw(
                    SYSTEM_JSON_REPAIR,
                    f"TEXT:\n<<<\n{text}\n>>>",
                    model=model,
                    temperature=0.0,
                    max_tokens=max_tokens
                )
                with open("last_repaired_output.txt", "w", encoding="utf-8") as f:
                    f.write("\n\n" + ("=" * 80) + "\n")
                    f.write(repaired)
                return safe_parse_json(repaired)

        except Exception as e:
            last_err = e
            if text:
                with open("last_exception_output.txt", "w", encoding="utf-8") as f:
                    f.write("\n\n" + ("=" * 80) + "\n")
                    f.write(text)
            time.sleep(sleep_s * attempt)

    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}")


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
    """
    pages: list of (page_number_1based, page_text)
    yields chunks: dict(page_start, page_end, text)
    """
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
                # "text": "\n\n".join([x[1] for x in buf]),
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
# Local merge (no LLM)
# ----------------------------
def merge_chunk_jsons_locally(chunk_jsons: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge chunk extractions into one document-level object:
    - Merge page ranges across chunks (min start, max end)
    - Merge items by normalized title
    - Union list fields with dedupe + cap
    - Fill missing scalars
    """
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip().lower())

    def dedupe_list(xs: List[Any], limit: int = 12) -> List[str]:
        out = []
        seen = set()
        for x in xs:
            if x is None:
                continue
            s = str(x).strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s[:240])
            if len(out) >= limit:
                break
        return out

    def merge_dict(dst: Dict[str, Any], src: Dict[str, Any]):
        for k, v in (src or {}).items():
            if v is None:
                continue
            if isinstance(v, list):
                dst[k] = dedupe_list((dst.get(k) or []) + v, limit=12)
            elif isinstance(v, dict):
                if not isinstance(dst.get(k), dict):
                    dst[k] = {}
                merge_dict(dst[k], v)
            else:
                if not dst.get(k):
                    dst[k] = v

    doc_id = None
    min_page = None
    max_page = None

    merged_map: Dict[str, Dict[str, Any]] = {}

    for obj in chunk_jsons:
        if not isinstance(obj, dict):
            continue

        if not doc_id and obj.get("doc_id"):
            doc_id = obj.get("doc_id")

        sp = obj.get("source_pages") or {}
        if isinstance(sp, dict):
            s = sp.get("start")
            e = sp.get("end")
            if isinstance(s, int):
                min_page = s if (min_page is None or s < min_page) else min_page
            if isinstance(e, int):
                max_page = e if (max_page is None or e > max_page) else max_page

        items = obj.get("items") or []
        if not isinstance(items, list):
            continue

        for it in items:
            if not isinstance(it, dict):
                continue
            title = it.get("title")
            key = norm(title) if title else ""
            if not key:
                continue

            if key not in merged_map:
                merged_map[key] = it
            else:
                cur = merged_map[key]
                merge_dict(cur, it)
                merged_map[key] = cur

    return {
        "doc_id": doc_id,
        "source_pages": {"start": min_page, "end": max_page},
        "items": list(merged_map.values()),
    }

def generate_document_summary(merged: Dict[str, Any]) -> str:
    """
    Generate one descriptive summary per document (1 LLM call).
    """
    if not merged or not merged.get("items"):
        return (
            "This document does not explicitly mention named policy initiatives "
            "related to labour market resilience or the twin transition."
        )

    user_prompt = json.dumps(
        {
            "doc_id": merged.get("doc_id"),
            "source_pages": merged.get("source_pages"),
            "policies": merged.get("items"),
        },
        ensure_ascii=False,
        indent=2,
    )

    summary_text = llm_raw_with_autotrim(
        SYSTEM_DOC_SUMMARY,
        user_prompt,
        temperature=0.0,
        # max_tokens=180,
        max_tokens=120,
    )

    return summary_text.strip()

   
# ----------------------------
# PDF writing (one output PDF per input PDF)
# ----------------------------
def _wrap_lines(text: str, max_chars: int = 110):
    text = (text or "").strip()
    if not text:
        return []
    text = re.sub(r"\s+", " ", text)
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def _draw_kv(c, x, y, key, val, line_h=12, max_chars=110):
    def ensure_space():
        nonlocal y
        if y < 2 * cm:
            c.showPage()
            y = A4[1] - 2 * cm

    ensure_space()

    # Key label (skip label if key is empty string, used for nested rendering)
    if key:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x, y, f"{key}:")
        y -= line_h
    c.setFont("Helvetica", 9)

    # ---- dict ----
    if isinstance(val, dict):
        if not val:
            ensure_space()
            c.drawString(x + 12, y, "(empty)")
            y -= line_h
            return y
        for dk, dv in val.items():
            if isinstance(dv, (dict, list)):
                ensure_space()
                c.setFont("Helvetica-Bold", 9)
                c.drawString(x + 12, y, f"{dk}:")
                y -= line_h
                c.setFont("Helvetica", 9)
                y = _draw_kv(c, x + 12, y, "", dv, line_h=line_h, max_chars=max_chars)
            else:
                for ln in _wrap_lines(f"{dk}: {dv}", max_chars=max_chars):
                    ensure_space()
                    c.drawString(x + 12, y, ln)
                    y -= line_h
        return y

    # ---- list ----
    if isinstance(val, list):
        if not val:
            ensure_space()
            c.drawString(x + 12, y, "(empty)")
            y -= line_h
            return y

        # If list contains dicts (e.g., items: [{...}, {...}]), render them nicely
        if all(isinstance(it, dict) for it in val):
            for idx, it in enumerate(val, start=1):
                ensure_space()
                c.setFont("Helvetica-Bold", 9)
                c.drawString(x + 12, y, f"- item {idx}")
                y -= line_h
                c.setFont("Helvetica", 9)

                # Prefer showing title first if present
                if it.get("title"):
                    for ln in _wrap_lines(f"title: {it.get('title')}", max_chars=max_chars):
                        ensure_space()
                        c.drawString(x + 24, y, ln)
                        y -= line_h

                for dk, dv in it.items():
                    if dk == "title":
                        continue
                    if isinstance(dv, (dict, list)):
                        ensure_space()
                        c.setFont("Helvetica-Bold", 9)
                        c.drawString(x + 24, y, f"{dk}:")
                        y -= line_h
                        c.setFont("Helvetica", 9)
                        y = _draw_kv(c, x + 24, y, "", dv, line_h=line_h, max_chars=max_chars)
                    else:
                        for ln in _wrap_lines(f"{dk}: {dv}", max_chars=max_chars):
                            ensure_space()
                            c.drawString(x + 24, y, ln)
                            y -= line_h
                y -= 4
            return y

        # Otherwise render as bullet list
        for item in val:
            for ln in _wrap_lines(f"- {item}", max_chars=max_chars):
                ensure_space()
                c.drawString(x + 12, y, ln)
                y -= line_h
        return y

    # ---- scalar fallback ----
    s = "" if val is None else str(val)
    lines = _wrap_lines(s, max_chars=max_chars)
    if not lines:
        ensure_space()
        c.drawString(x + 12, y, "(empty)")
        y -= line_h
        return y

    for ln in lines:
        ensure_space()
        c.drawString(x + 12, y, ln)
        y -= line_h
    return y

def _is_empty_value(v):
    if v is None:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    if isinstance(v, list) and len(v) == 0:
        return True
    if isinstance(v, dict) and len(v) == 0:
        return True
    return False

def _non_empty_dict(d: dict) -> dict:
    """Remove empty values recursively (for printing only)."""
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            vv = _non_empty_dict(v)
            if not _is_empty_value(vv):
                out[k] = vv
        elif isinstance(v, list):
            vv = [x for x in v if not _is_empty_value(x)]
            if vv:
                out[k] = vv
        else:
            if not _is_empty_value(v):
                out[k] = v
    return out

FIELD_LABELS = {
    "issuing_body": "Issuing body",
    "policy_level": "Policy level",
    "jurisdictions": "Jurisdictions",
    "year": "Year",
    "date_range": "Date range",
    "policy_type": "Policy type",
    "instruments": "Policy instruments",
    "twin_transition": "Twin transition",
    "objectives": "Objectives",
    "measures": "Measures",
    "target_groups": "Target groups",
    "sectors": "Sectors",
    "funding": "Funding",
    "social_inclusion": "Social inclusion",
    "monitoring_evaluation": "Monitoring & evaluation",
    "quotes": "Direct evidence (quotes)",
}

def draw_policy_cards_all_metadata(c, x, y, items, line_h=12, max_chars=112):
    """
    Render merged policy items as readable bullet cards.
    Prints ALL extracted metadata fields that are non-empty (including nested dicts).
    """
    def ensure_space(min_y=4 * cm):
        nonlocal y
        if y < min_y:
            c.showPage()
            y = A4[1] - 2 * cm

    def draw_kv_line(label, value, indent=12):
        nonlocal y
        if _is_empty_value(value):
            return
        if isinstance(value, list):
            if not value:
                return
            c.setFont("Helvetica-Bold", 9)
            ensure_space()
            c.drawString(x + indent, y, f"{label}:")
            y -= 11
            c.setFont("Helvetica", 9)
            for it in value:
                if _is_empty_value(it):
                    continue
                for ln in _wrap_lines(f"- {str(it)}", max_chars=max_chars):
                    ensure_space()
                    c.drawString(x + indent + 10, y, ln)
                    y -= 11
            return

        if isinstance(value, dict):
            vv = _non_empty_dict(value)
            if not vv:
                return
            c.setFont("Helvetica-Bold", 9)
            ensure_space()
            c.drawString(x + indent, y, f"{label}:")
            y -= 11
            c.setFont("Helvetica", 9)
            for dk, dv in vv.items():
                if isinstance(dv, (dict, list)):
                    draw_kv_line(dk, dv, indent=indent + 10)
                else:
                    for ln in _wrap_lines(f"{dk}: {dv}", max_chars=max_chars):
                        ensure_space()
                        c.drawString(x + indent + 10, y, ln)
                        y -= 11
            return

        # scalar
        c.setFont("Helvetica-Bold", 9)
        ensure_space()
        c.drawString(x + indent, y, f"{label}:")
        y -= 11
        c.setFont("Helvetica", 9)
        for ln in _wrap_lines(str(value), max_chars=max_chars):
            ensure_space()
            c.drawString(x + indent + 10, y, ln)
            y -= 11

    if not items:
        c.setFont("Helvetica", 10)
        ensure_space()
        c.drawString(x, y, "No named policy initiatives, strategies, programmes, or regulations were explicitly mentioned in this document.")
        y -= 14
        return y

    # print in a stable order (title-based)
    def sort_key(it):
        return (it.get("policy_level") or "", (it.get("title") or "").lower())

    items_sorted = sorted(items, key=sort_key)

    for i, it in enumerate(items_sorted, start=1):
        ensure_space()
        it_clean = _non_empty_dict(it)

        title = (it.get("title") or "(no title)").strip()

        c.setFont("Helvetica-Bold", 11)
        c.drawString(x, y, f"{i}. {title}")
        y -= 14

        # Important fields first (then everything else)
        preferred_order = [
            "issuing_body",
            "policy_level",
            "jurisdictions",
            "year",
            "date_range",
            "policy_type",
            "instruments",
            "twin_transition",
            "objectives",
            "measures",
            "target_groups",
            "sectors",
            "funding",
            "social_inclusion",
            "monitoring_evaluation",
            "quotes",
        ]

        # Always show source_pages if present (some extractions keep it per-item)
        if "source_pages" in it_clean:
            draw_kv_line("source_pages", it_clean.get("source_pages"), indent=12)

        # Show preferred fields (if non-empty)

        SECTION_HEADERS = {
        "issuing_body": "Scope and Governance",
        "objectives": "Objectives and Measures",
        "instruments": "Policy Instruments",
        "social_inclusion": "Social and Monitoring Aspects",
        "quotes": "Supporting Evidence",}

        printed_sections = set()
        # for k in preferred_order:
        #     if k in it_clean and k != "title":
        #         draw_kv_line(k, it_clean.get(k), indent=12)

        # # Show any other fields that may exist (future-proof)
        # remaining = [k for k in it_clean.keys() if k not in preferred_order and k not in ("title",)]
        # for k in sorted(remaining):
        #     draw_kv_line(k, it_clean.get(k), indent=12)
        for k in preferred_order:
            if k not in it_clean or k == "title":
                continue

            if k in SECTION_HEADERS and SECTION_HEADERS[k] not in printed_sections:
                ensure_space()
                c.setFont("Helvetica-Bold", 10)
                c.drawString(x + 12, y, SECTION_HEADERS[k])
                y -= 12
                c.setFont("Helvetica", 9)
                printed_sections.add(SECTION_HEADERS[k])

            label = FIELD_LABELS.get(k, k.replace("_", " ").title())
            draw_kv_line(label, it_clean.get(k), indent=12)

        y -= 10

    return y


def write_extraction_pdf(
    source_pdf_path: Path,
    output_dir: Path,
    doc_id: str,
    merged: Optional[Dict[str, Any]],
    chunk_outputs: List[Dict[str, Any]],
    include_merged: bool = True,
    include_chunks: bool = True,
    doc_summary: Optional[str] = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = output_dir / f"{source_pdf_path.stem}_EXTRACTION4.pdf"

    c = canvas.Canvas(str(out_pdf), pagesize=A4)
    width, height = A4
    x = 2 * cm
    y = height - 2 * cm
    line_h = 12

    # Header
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, "WP5 Policy Extraction Report")
    y -= 18
    
    c.setFont("Helvetica", 12)
    # c.drawString(x,y, "This report summarises named policy initiatives explicitly mentioned in the source document.Only information stated verbatim in the text has been extracted. No interpretation or inference has been applied.")
    
    intro = (
    "This report summarises named policy initiatives explicitly mentioned in the source document. "
    "Only information stated verbatim in the text has been extracted. "
    "No interpretation or inference has been applied.")
    for ln in _wrap_lines(intro, max_chars=110):
        c.drawString(x, y, ln)
        y -= 12

    y -= 4
    c.setFont("Helvetica", 10)
    c.drawString(x, y, f"Document ID: {doc_id}")
    y -= 14
    c.drawString(x, y, f"Source file: {source_pdf_path.name}")
    y -= 18

    if doc_summary:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y, "Document Summary")
        y -= 16

        c.setFont("Helvetica", 10)
        for ln in _wrap_lines(doc_summary, max_chars=110):
            c.drawString(x, y, ln)
            y -= 12

        # y -= 18

    c.showPage()
    y = height - 2 * cm

    # Quick human-readable summary (from merged.items if available)
    if include_merged and merged is not None and isinstance(merged.get("items"), list):
        items = merged.get("items") or []
        c.setFont("Helvetica-Bold", 12)
        
        y -= 18

        c.drawString(x, y, "Identified Policies — Executive Summary")
        y -= 16

        if not items:
            c.setFont("Helvetica", 10)
            c.drawString(x, y, "No named policies detected in this document.")
            y -= 14
        else:
            for idx, it in enumerate(items, start=1):
                if y < 4 * cm:
                    c.showPage()
                    y = height - 2 * cm

                title = (it.get("title") or "(no title)").strip()
                issuing = (it.get("issuing_body") or "").strip()
                level = it.get("policy_level")
                year = it.get("year")
                quotes = it.get("quotes") or []

                c.setFont("Helvetica-Bold", 10)
                c.drawString(x, y, f"{idx}. {title}")
                y -= 12

                c.setFont("Helvetica", 9)
                meta = []
                # if issuing: meta.append(f"- Issuing body: {issuing}")
                # if level: meta.append(f"- Level: {level}")
                # if year: meta.append(f"- Year: {year}")

                meta_lines = []
                if issuing: meta_lines.append(f"Issuing body: {issuing}")
                if level: meta_lines.append(f"Policy level: {level}")
                if year: meta_lines.append(f"Year: {year}")

                for mline in meta_lines:
                    for ln in _wrap_lines(f"- {mline}", max_chars=110):
                        c.drawString(x + 12, y, ln)
                        y -= 11

                # for ln in _wrap_lines(" | ".join(meta), max_chars=110):
                #     c.drawString(x + 12, y, ln)
                #     y -= 11

                if quotes:
                    c.setFont("Helvetica-Oblique", 9)
                    for q in quotes[:2]:
                        for ln in _wrap_lines(f'- Direct evidence from the text: "{q}"', max_chars=105):
                            c.drawString(x + 12, y, ln)
                            y -= 11
                    c.setFont("Helvetica", 9)

                y -= 8

        y -= 6

    # Policies (deduplicated) — full metadata cards (no JSON dump)
    if include_merged and merged is not None and isinstance(merged.get("items"), list):
        if y < 4 * cm:
            c.showPage()
            y = height - 2 * cm

        c.showPage()
        y = height - 2 * cm
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y, "Detailed Policy Information (verbatim extraction)")
        y -= 16

        c.setFont("Helvetica", 10)
        note = ("The information below lists all policy-related details explicitly stated in the document. " "Fields not shown were not mentioned in the source text.")
        for ln in _wrap_lines(note, max_chars=110):
            c.drawString(x, y, ln)
            y -= 12
        # c.drawString(x,y, "The information below lists all policy-related details explicitly stated in the document. Fields not shown were not mentioned in the source text.")

        y -= 6
        y = draw_policy_cards_all_metadata(c, x, y, merged.get("items") or [], line_h=line_h)
        y -= 6

    # Chunk outputs section (verbatim model JSON per chunk)
    c.showPage()
    y = height - 2 * cm
    if include_chunks:
        if y < 4 * cm:
            c.showPage()
            y = height - 2 * cm

        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y, "Appendix — Technical Extraction Output (JSON)")
        y -= 16

        c.setFont("Helvetica", 10)
        c.drawString(x,y, "This section is provided for validation and technical review. It is not intended for general users.")
        
        y -= 14

        for i, ch in enumerate(chunk_outputs, start=1):
            if y < 4 * cm:
                c.showPage()
                y = height - 2 * cm

            c.setFont("Helvetica-Bold", 10)
            c.drawString(x, y, f"Chunk {i} | pages {ch['page_start']}-{ch['page_end']}")
            y -= 14

            extracted = ch.get("extracted") or {}
            for k, v in extracted.items():
                y = _draw_kv(c, x, y, k, v, line_h=line_h)
                y -= 2

            y -= 10

    c.save()
    return out_pdf

def filter_items_by_verbatim_title(extracted: Dict[str, Any], chunk_text: str) -> Dict[str, Any]:
    """
    Drop items whose title doesn't appear verbatim in the chunk text (case-insensitive).
    Prevents invented policy names.
    """
    if not isinstance(extracted, dict):
        return extracted

    items = extracted.get("items")
    if not isinstance(items, list) or not items:
        return extracted

    hay = (chunk_text or "").lower()
    kept = []
    for it in items:
        if not isinstance(it, dict):
            continue
        title = (it.get("title") or "").strip()
        if not title:
            continue
        if title.lower() in hay:
            kept.append(it)

    extracted["items"] = kept
    return extracted

# ----------------------------
# Main processing: one PDF -> one output PDF
# ----------------------------
def process_one_pdf(pdf_path: str, output_folder: str) -> Path:
    source_pdf = Path(pdf_path)
    doc_id = source_pdf.stem
    out_dir = Path(output_folder)

    pages = extract_pdf_pages(str(source_pdf))
    chunks = list(chunk_pages(pages))

    # ---- crash-safe resume setup ----
    ckpt_ok, ckpt_err = _checkpoint_paths(output_folder, doc_id)
    done = load_chunk_checkpoint(ckpt_ok)

    chunk_outputs: List[Dict[str, Any]] = []

    # If resuming, pre-fill chunk_outputs from checkpoint (so merge/pdf still works)
    for cid in sorted(done.keys()):
        rec = done[cid]
        chunk_outputs.append({
            "page_start": rec.get("page_start"),
            "page_end": rec.get("page_end"),
            "extracted": rec.get("extracted") or {},
        })

    for i, ch in enumerate(chunks, start=1):
        # Skip chunks already checkpointed
        if i in done:
            continue

        user_prompt = USER_EXTRACT_TEMPLATE.format(
            chunk_id=i,
            doc_id=doc_id,
            page_start=ch["page_start"],
            page_end=ch["page_end"],
            chunk_text=ch["text"],
        )

        try:
            extracted = llm_json(SYSTEM_EXTRACT, user_prompt, max_tokens=2500)
            extracted = filter_items_by_verbatim_title(extracted, ch["text"])

            # Save immediately (safety pin)
            ok_record = {
                "chunk_id": i,
                "doc_id": doc_id,
                "page_start": ch["page_start"],
                "page_end": ch["page_end"],
                "extracted": extracted,
            }
            append_jsonl(ckpt_ok, ok_record)

            chunk_outputs.append({
                "page_start": ch["page_start"],
                "page_end": ch["page_end"],
                "extracted": extracted,
            })

        except Exception as e:
            # Record the failure and continue (so you don't lose everything)
            err_record = {
                "chunk_id": i,
                "doc_id": doc_id,
                "page_start": ch["page_start"],
                "page_end": ch["page_end"],
                "error": repr(e),
                "traceback": traceback.format_exc(),
            }
            append_jsonl(ckpt_err, err_record)

            # OPTIONAL: also checkpoint an empty extraction so merge can proceed
            empty_extracted = {
                "chunk_id": i,
                "doc_id": doc_id,
                "source_pages": {"start": ch["page_start"], "end": ch["page_end"]},
                "items": [],
            }
            ok_record = {
                "chunk_id": i,
                "doc_id": doc_id,
                "page_start": ch["page_start"],
                "page_end": ch["page_end"],
                "extracted": empty_extracted,
            }
            append_jsonl(ckpt_ok, ok_record)

            chunk_outputs.append({
                "page_start": ch["page_start"],
                "page_end": ch["page_end"],
                "extracted": empty_extracted,
            })

        time.sleep(SLEEP_BETWEEN_CALLS)


    merged = None

    extracted_list = [x["extracted"] for x in chunk_outputs]

    if INCLUDE_LLM_MERGE:
        # cheaper merge input: only the essentials
        merge_input = []
        for obj in extracted_list:
            if not isinstance(obj, dict):
                continue
            merge_input.append({
                "doc_id": obj.get("doc_id"),
                "source_pages": obj.get("source_pages"),
                "items": obj.get("items") or [],
            })

        merged = llm_json(
            SYSTEM_MERGE,
            json.dumps(merge_input, ensure_ascii=False),
            max_tokens=2500
        )
    elif INCLUDE_MERGED_RECORD:
        # merged = merge_chunk_jsons_locally([x["extracted"] for x in chunk_outputs])
        merged = merge_chunk_jsons_locally(extracted_list)

    doc_summary = generate_document_summary(merged)
    
    # if INCLUDE_MERGED_RECORD:
    #     merged = merge_chunk_jsons_locally([x["extracted"] for x in chunk_outputs])

    out_pdf = write_extraction_pdf(
        source_pdf_path=source_pdf,
        output_dir=out_dir,
        doc_id=doc_id,
        merged=merged,
        chunk_outputs=chunk_outputs,
        include_merged=INCLUDE_MERGED_RECORD,
        include_chunks=INCLUDE_CHUNK_OUTPUTS,
        doc_summary=doc_summary,
    )

    print(f"Saved extraction PDF: {out_pdf}")
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
    input_folder = get_path("POLICY_ANALYSIS_INPUT_FOLDER")
    output_folder = get_path("POLICY_ANALYSIS_OUTPUT_FOLDER")
    if input_folder is None or output_folder is None:
        raise RuntimeError(
            "Set POLICY_ANALYSIS_INPUT_FOLDER and POLICY_ANALYSIS_OUTPUT_FOLDER in .env."
        )
    process_folder(str(input_folder), str(output_folder))
