"""
Transducer v6 — LLM Schema Re-Ranking + Execution Evidence
============================================================
Lessons from v1-v5:
  - Dense retrieval gets gold in top-5 for 96% of queries
  - Re-embedding per candidate DESTROYS ranking (0.64 → 0.06)
  - Execution probing fails when DBs are empty (Spider)
  - LLM schema reasoning was the ONLY approach that improved Spider R@1
  - LLM scoring gave best BIRD results too (R@1=0.84 in V2)

Architecture (simplified):
  1. Dense retrieval → Top-5 candidates
  2. For populated DBs: run execution probes, include results in LLM prompt
  3. ONE LLM call: pick the best DB from top-5 with full schema context
  4. Return re-ranked list

Key: ONE focused LLM call, not 5-7 scattered calls.
"""

import os
import re
import json
import time
import sqlite3
from typing import List, Dict, Tuple, Optional

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

ROUTER_MODEL = "qwen/qwen3-32b"


class AgenticTransducer:
    def __init__(self, schemas: Dict[str, str], model_name: str = ROUTER_MODEL):
        self.schemas = schemas
        self.model_name = model_name
        self.client = None

        api_key = os.environ.get("GROQ")
        if api_key:
            try:
                self.client = Groq(api_key=api_key)
            except Exception as e:
                print(f"Warning: Groq init failed: {e}")
        else:
            print("Warning: GROQ API Key not found.")

    # -----------------------------------------------------------------------
    # LLM call with retry + <think> stripping
    # -----------------------------------------------------------------------
    def _call_llm(self, prompt: str, system: str = "You are a database expert.",
                  temperature: float = 0.0) -> str:
        if not self.client:
            return ""

        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": prompt}
                    ],
                    model=self.model_name,
                    temperature=temperature,
                    max_completion_tokens=2048,
                )
                text = resp.choices[0].message.content or ""
                # Strip qwen3-32b <think> reasoning — handle both complete and incomplete
                # Complete: <think>...</think>
                text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
                # Incomplete (truncated): <think>... with no closing tag
                text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
                return text.strip()
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    wait = 10 * (2 ** attempt)
                    print(f"[RateLimit] Sleeping {wait}s...")
                    time.sleep(wait)
                    continue
                return f"Error: {e}"
        return ""

    # -----------------------------------------------------------------------
    # Core: LLM picks the best DB from candidates
    # -----------------------------------------------------------------------
    def rerank(self, query: str, candidates: List[str],
               exec_evidence: Dict[str, str] = None) -> List[str]:
        """
        Uses ONE LLM call to re-rank top-5 candidate databases.
        
        For each candidate, provides:
          - Compressed schema (tables + columns, ~200 chars)
          - Execution evidence if available (what data was found)
        
        Returns: re-ranked list of db_ids, best first.
        """
        # Build schema context per candidate
        db_descriptions = []
        for i, db in enumerate(candidates):
            schema = self.schemas.get(db, "")
            # Extract just tables and columns for compact representation
            tables = re.findall(
                r'Table:\s*(\w+),\s*Columns:\s*([^;]+)',
                schema, re.IGNORECASE
            )
            if tables:
                compact = "; ".join(
                    f"{tbl}({', '.join(c.strip() for c in cols.split(','))})"
                    for tbl, cols in tables
                )
            else:
                compact = schema[:300]

            desc = f"[{i+1}] {db}: {compact}"
            
            # Add execution evidence if available
            if exec_evidence and db in exec_evidence:
                ev = exec_evidence[db]
                if ev:
                    desc += f"\n    Evidence: {ev}"

            db_descriptions.append(desc)

        schema_block = "\n".join(db_descriptions)

        prompt = f"""Given a natural language question and {len(candidates)} candidate database schemas, determine which database is the BEST match for answering the question.

Consider:
1. Which database's table/column structure matches the entities in the question?
2. Which database has the specific tables needed (e.g., keyword tables for keyword queries, conference tables for conference queries)?
3. If execution evidence shows data was found in a database, that's strong signal.
4. Prefer databases with MORE relevant linking tables (e.g., publication_keyword links publications to keywords).

Question: {query}

Candidate databases:
{schema_block}

IMPORTANT: Reply with ONLY the database name (e.g., "academic"). Nothing else."""

        response = self._call_llm(prompt, temperature=0.0)

        # Parse response: find which candidate was selected
        response_lower = response.strip().lower()
        
        # Try exact match first
        for db in candidates:
            if db.lower() == response_lower:
                # Move selected to front, keep rest in original order
                result = [db] + [d for d in candidates if d != db]
                return result

        # Try substring match
        for db in candidates:
            if db.lower() in response_lower:
                result = [db] + [d for d in candidates if d != db]
                return result

        # Fallback: keep original order
        return list(candidates)

    # -----------------------------------------------------------------------
    # Execution probe helpers (for populated DBs)
    # -----------------------------------------------------------------------
    def extract_entities(self, query: str) -> List[str]:
        """Extract value-level entities from query."""
        prompt = f"""Extract specific NAMED ENTITIES and VALUES from this question.
These are concrete things to filter/search for — NOT generic SQL words.
Examples: conference names, years, person names, locations, organizations.

Question: {query}

Output ONLY a JSON array. Example: ["VLDB", "2021"]
If none found, output: []"""

        response = self._call_llm(
            prompt,
            system="Extract entities. Output ONLY a JSON array.",
            temperature=0.0
        )

        try:
            return json.loads(response.strip())
        except (json.JSONDecodeError, TypeError):
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except (json.JSONDecodeError, TypeError):
                    pass
        return []

    def generate_probes(self, query: str, entities: List[str],
                        db_id: str) -> List[str]:
        """Generate brute-force probe SQLs for all text columns."""
        schema_text = self.schemas.get(db_id, "")
        tables = re.findall(
            r'Table:\s*(\w+),\s*Columns:\s*([^;]+)',
            schema_text, re.IGNORECASE
        )
        if not tables:
            return []

        TEXT_KW = {'name', 'title', 'keyword', 'abstract', 'description',
                   'label', 'text', 'phrase', 'affiliation', 'shortname',
                   'fullname', 'venuename', 'keyphrasename', 'authorname'}
        NUM_KW = {'year', 'date', 'num', 'count', 'citation', 'reference'}

        text_columns = []
        numeric_columns = []
        for tbl, col_str in tables:
            for col in [c.strip() for c in col_str.split(',')]:
                cl = col.lower()
                if any(kw in cl for kw in TEXT_KW):
                    text_columns.append((tbl, col))
                if any(kw in cl for kw in NUM_KW):
                    numeric_columns.append((tbl, col))

        # Query-noun prioritization
        query_words = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))
        def relevance(tc):
            t = tc[0].lower()
            if t in query_words: return 0
            for w in query_words:
                if w in t or t in w: return 1
            return 2
        text_columns.sort(key=relevance)

        text_ents = [e for e in entities if not re.match(r'^\d{4}$', str(e))]
        year_ents = [e for e in entities if re.match(r'^\d{4}$', str(e))]

        probes = []
        for ent in text_ents[:2]:
            for tbl, col in text_columns:
                probes.append(
                    f"SELECT 1 FROM [{tbl}] WHERE [{col}] LIKE '%{ent}%' LIMIT 1"
                )
        for yr in year_ents[:1]:
            for tbl, col in numeric_columns[:3]:
                probes.append(
                    f"SELECT 1 FROM [{tbl}] WHERE [{col}] = {yr} LIMIT 1"
                )
        if not probes:
            for tbl, _ in tables[:3]:
                probes.append(f"SELECT 1 FROM [{tbl}] LIMIT 1")

        return probes[:20]

    def execute_probes(self, probes: List[str], db_path: str) -> Tuple[float, List[bool]]:
        """Execute probes against SQLite. Returns (score, hit_list)."""
        if not db_path or not os.path.exists(db_path):
            return 0.0, [False] * len(probes)

        results = []
        score = 0.0
        for sql in probes:
            try:
                conn = sqlite3.connect(db_path, timeout=2.0)
                rows = conn.cursor().execute(sql).fetchall()
                conn.close()
                hit = bool(rows)
                results.append(hit)
                if hit:
                    score += 1.0
            except Exception:
                results.append(False)

        return score, results
