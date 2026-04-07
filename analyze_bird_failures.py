#!/usr/bin/env python3
"""
BIRD Failure Taxonomy Analyzer
================================
Parses bird_misclassified_pipeline.jsonl and categorizes every failed query
into one of four failure buckets tailored for the data-rich BIRD dataset.

Categories:
1. Retrieval Funnel Drop (Stage 0/1 Miss)
2. SQL Generator Choke (Syntax/Execution Error)
3. The "Empty Set" Trap (Logic Choke)
4. Populated Ambiguity / Judge Error
"""

import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
JSONL_PATH = os.path.join(RESULTS_DIR, "bird_misclassified_pipeline.jsonl")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "bird_full_failure_taxonomy.txt")


def get_candidate_status(record, db_id):
    """Get the execution_status of a specific db_id from candidate_contexts."""
    for ctx in record.get("candidate_contexts", []):
        if ctx.get("db_id") == db_id:
            return ctx.get("execution_status", "MISSING")
    return "MISSING"


def get_candidate_sql(record, db_id):
    """Get the generated SQL for a specific db_id from candidate_contexts."""
    for ctx in record.get("candidate_contexts", []):
        if ctx.get("db_id") == db_id:
            return ctx.get("sql", "N/A")
    return "N/A"


def classify_failure(record):
    """Classify a failed BIRD query into exactly one taxonomy category."""
    gold_db = record.get("gold_db", "")
    selected_db = record.get("final_selected_db", "")
    plausible = record.get("plausible_dbs_identified", [])

    # Category 1: Retrieval Funnel Drop
    # Gold DB was filtered out by Stage 0 or Stage 1
    if gold_db not in plausible:
        return 1, "Retrieval Funnel Drop"

    # Gold DB IS in plausible — check its execution status
    gold_status = get_candidate_status(record, gold_db)
    selected_status = get_candidate_status(record, selected_db)

    # Category 2: SQL Generator Choke
    # Gold DB is plausible but DeepSeek generated broken SQL for it
    if gold_status == "ERROR":
        return 2, "SQL Generator Choke"

    # Category 3: The "Empty Set" Trap (Logic Choke)
    # DeepSeek wrote valid SQL but the logic was too restrictive —
    # returned EMPTY_SET or ZERO_COUNT on a database that has real data
    if gold_status in ("EMPTY_SET", "ZERO_COUNT"):
        return 3, 'The "Empty Set" Trap'

    # Category 4: Populated Ambiguity / Judge Error
    # Gold DB returned POPULATED data but the Judge still picked wrong
    return 4, "Populated Ambiguity / Judge Error"


def main():
    # Load all records
    records = []
    with open(JSONL_PATH, "r") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))

    total = len(records)
    correct = sum(1 for r in records if r.get("correct"))
    failures = [r for r in records if not r.get("correct")]
    n_fail = len(failures)

    # Classify every failure
    categories = {1: [], 2: [], 3: [], 4: []}
    cat_names = {
        1: "Retrieval Funnel Drop",
        2: "SQL Generator Choke",
        3: 'The "Empty Set" Trap',
        4: "Populated Ambiguity / Judge Error",
    }

    for record in failures:
        cat_id, cat_name = classify_failure(record)
        categories[cat_id].append(record)

    # ===== Build the report =====
    lines = []

    # Section A: Summary Stats
    lines.append("=" * 70)
    lines.append("BIRD MISCLASSIFIED PIPELINE — FAILURE TAXONOMY REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Total queries processed:  {total}")
    lines.append(f"Correct (recovered):      {correct} ({correct/total*100:.1f}%)")
    lines.append(f"Failed:                   {n_fail} ({n_fail/total*100:.1f}%)")
    lines.append("")
    lines.append("-" * 70)
    lines.append("SECTION A: SUMMARY STATS")
    lines.append("-" * 70)
    lines.append("")

    for cat_id in [1, 2, 3, 4]:
        count = len(categories[cat_id])
        pct = count / n_fail * 100 if n_fail > 0 else 0
        lines.append(f"  Category {cat_id}: {cat_names[cat_id]}")
        lines.append(f"    Count: {count}/{n_fail} ({pct:.1f}%)")
        lines.append("")

    # Section B: Detailed Log
    lines.append("")
    lines.append("-" * 70)
    lines.append("SECTION B: DETAILED FAILURE LOG")
    lines.append("-" * 70)

    failure_idx = 0
    for cat_id in [1, 2, 3, 4]:
        lines.append("")
        lines.append(f"{'='*70}")
        lines.append(f"  CATEGORY {cat_id}: {cat_names[cat_id].upper()} ({len(categories[cat_id])} failures)")
        lines.append(f"{'='*70}")

        for record in categories[cat_id]:
            failure_idx += 1
            gold_db = record.get("gold_db", "")
            selected_db = record.get("final_selected_db", "")
            gold_status = get_candidate_status(record, gold_db)
            selected_status = get_candidate_status(record, selected_db)
            gold_sql = get_candidate_sql(record, gold_db)
            selected_sql = get_candidate_sql(record, selected_db)
            query = record.get("user_query", "")
            evidence = record.get("bird_evidence", "")

            lines.append("")
            lines.append(f"[Failure {failure_idx}] - CATEGORY: {cat_names[cat_id]}")
            lines.append(f"Query: {query}")
            if evidence:
                lines.append(f"BIRD Evidence: {evidence}")
            lines.append(f"Gold DB: {gold_db} | Status: {gold_status}")
            lines.append(f"Selected DB: {selected_db} | Status: {selected_status}")
            lines.append(f"Gold SQL: {gold_sql}")
            lines.append(f"Selected SQL: {selected_sql}")
            lines.append("-" * 60)

    report = "\n".join(lines)

    # Write full report to file
    with open(OUTPUT_PATH, "w") as f:
        f.write(report)

    # Print ONLY summary stats to terminal
    print()
    print("=" * 70)
    print("BIRD FAILURE TAXONOMY — SUMMARY")
    print("=" * 70)
    print(f"  Total queries:  {total}")
    print(f"  Correct:        {correct} ({correct/total*100:.1f}%)")
    print(f"  Failed:         {n_fail} ({n_fail/total*100:.1f}%)")
    print()
    for cat_id in [1, 2, 3, 4]:
        count = len(categories[cat_id])
        pct = count / n_fail * 100 if n_fail > 0 else 0
        print(f"  Cat {cat_id}: {cat_names[cat_id]:40s} — {count:>4}/{n_fail} ({pct:.1f}%)")
    print()
    print(f"  Full report saved to: {OUTPUT_PATH}")
    print()


if __name__ == "__main__":
    main()
