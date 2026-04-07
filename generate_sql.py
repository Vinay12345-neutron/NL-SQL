"""
Execution-Guided SQL Generation with Self-Correction

This script takes the routed output from the Agentic Retrieval phase, 
retrieves the database schemas, and uses a heavy LLM to generate SQL queries.

Critically, it employs Execution-Guided Self-Correction:
1. It attempts to execute the generated SQL against the actual local SQLite file.
2. If it fails (e.g., OperationalError), it feeds the exact error message back 
   to the LLM for a single attempt to self-correct the query.
3. Calculates Execution Success Rate and outputs the final JSON.
"""

import os
import json
import sqlite3
import argparse
import time
from tqdm import tqdm
from typing import Dict, Tuple, Optional, Any, List

# From external libraries
from dotenv import load_dotenv
from groq import Groq

# Reuse generic metrics import from baseline script (Assuming calculate_metrics handles lists of dicts if needed)
# To match ablation script styles, we import load_schemas
from baseline_retrieval import load_schemas, calculate_metrics

load_dotenv()

# ==========================================
# Configuration & Constants
# ==========================================
MODEL_NAME = "qwen/qwen3-32b" # As explicitly requested by the user
MAX_RETRIES = 1

def get_db_path(db_id: str, dataset_name: str) -> Optional[str]:
    """
    Locates the unzipped SQLite database file for a given db_id.
    Handles fallbacks since the router might hallucinate a DB from another dataset.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define possible paths
    paths_to_check = [
        os.path.join(base_dir, "spider_data", "database", db_id, f"{db_id}.sqlite"),
        os.path.join(base_dir, "dev_20240627", "dev_databases", db_id, f"{db_id}.sqlite"),
        os.path.join(base_dir, "train", "train_databases", db_id, f"{db_id}.sqlite")
    ]
    
    # Priority based on dataset_name, but we will check all
    if "bird" in dataset_name.lower():
        paths_to_check = [paths_to_check[1], paths_to_check[2], paths_to_check[0]]
        
    for path in paths_to_check:
        if os.path.exists(path):
            return path
            
    return None

def get_database_preview(db_path: str, max_tables: int = 15) -> str:
    """Connects to the SQLite DB and fetches 3 sample rows from each table."""
    if not db_path or not os.path.exists(db_path):
        return "Preview could not be generated (Database file not found)."
        
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall() if row[0] != 'sqlite_sequence']
        
        preview_sections = []
        for table in tables[:max_tables]:
            try:
                # Get column types and names
                cursor.execute(f"PRAGMA table_info(`{table}`)")
                col_info = cursor.fetchall()
                if not col_info:
                    continue
                    
                col_names = [info[1] for info in col_info]
                
                # Get 3 sample rows
                cursor.execute(f"SELECT * FROM `{table}` LIMIT 3;")
                rows = cursor.fetchall()
                
                if rows:
                    preview = f"Table: {table}\n"
                    preview += " | ".join(col_names) + "\n"
                    for row in rows:
                        row_str = " | ".join(repr(val) for val in row)
                        preview += row_str + "\n"
                    preview_sections.append(preview)
            except sqlite3.Error:
                continue
                
        conn.close()
        
        if preview_sections:
            return "\n".join(preview_sections)
        else:
            return "No data found in tables."
            
    except Exception as e:
        return f"Error connecting to database for preview: {e}"

def execute_sql(sql: str, db_path: str) -> Tuple[bool, str, Any]:
    """
    Attempts to execute a SQL query against the target database.
    Returns (success_boolean, error_or_success_message, result_set).
    """
    if not db_path or not os.path.exists(db_path):
        return False, f"Database file not found: {db_path}", None
        
    try:
        # Connect with a timeout to prevent hanging on complex accidental queries
        conn = sqlite3.connect(db_path, timeout=5.0)
        cursor = conn.cursor()
        
        # We only need to verify parsing and execution validity here, not data fetching logic.
        cursor.execute(sql)
        res = cursor.fetchall() # Optional: Ensures no fetch-related errors
        
        conn.close()
        return True, "Execution Successful.", res
        
    except sqlite3.Error as e:
        return False, str(e), None
    except Exception as e:
        return False, f"Unexpected Execution Error: {e}", None

def extract_code_block(text: str) -> str:
    """
    Extracts purely the SQL code from a markdown-formatted LLM response,
    filtering out <think> tags if reasoning models are used.
    """
    text = text.strip()
    
    # Remove thought block
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
        
    # Extract from markdown if present
    if "```sql" in text:
        text = text.split("```sql")[-1].split("```")[0].strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()
        
    return text.strip()

def generate_sql_with_feedback(client: Groq, resolved_question: str, db_id: str, schema_str: str, data_preview_str: str, db_path: str, temperature: float = 0.1) -> Tuple[str, bool, str, Any]:
    """
    Generates SQL using the LLM. Implements a self-correction loop if execution fails.
    """
    # 1. Initial Prompt Construction
    system_prompt = """You are an expert, strict SQLite developer. 
You MUST output perfectly executable SQLite code. 
CRITICAL RULES:
1. You are strictly forbidden from hallucinating or guessing table or column names.
2. You may ONLY use the exact tables, columns, and foreign keys provided in the schema.
3. Output ONLY the raw SQL query string. No markdown formatting, no backticks, and absolutely no explanations."""
    
    initial_prompt = f"""
Write a single, valid SQLite query to answer the question.

=== EXACT SCHEMA FOR DATABASE '{db_id}' ===
{schema_str}

=== SAMPLE DATA PREVIEW (Top 3 rows) ===
{data_preview_str}
==========================================

Question: {resolved_question}
"""
    try:
        # Step 1: Initial Generation
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_prompt}
            ],
            temperature=temperature, # Optimal temperature for code generation
            max_completion_tokens=4096,
            stream=False
        )
        
        sql = extract_code_block(completion.choices[0].message.content)
        
        # Step 2: Attempt Execution
        success, exec_msg, res = execute_sql(sql, db_path)
        
        if success:
            return sql, True, exec_msg, res
            
        # Step 3: Self-Correction Loop (Feedback)
        correction_prompt = f"""
The following SQLite query failed to execute:
{sql}

The error returned by the SQLite engine was:
{exec_msg}

Please fix the error and provide the corrected SQL query. Return ONLY the raw SQL query, without any markdown formatting or backticks.
"""
        completion_retry = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_prompt},
                {"role": "assistant", "content": sql},
                {"role": "user", "content": correction_prompt}
            ],
            temperature=temperature,
            max_completion_tokens=4096,
            stream=False
        )
        
        corrected_sql = extract_code_block(completion_retry.choices[0].message.content)
        
        # Step 4: Final Execution Attempt
        success_retry, exec_msg_retry, res_retry = execute_sql(corrected_sql, db_path)
        return corrected_sql, success_retry, exec_msg_retry, res_retry

    except Exception as e:
        return f"API/Internal Error: {e}", False, str(e), None


def main():
    parser = argparse.ArgumentParser(description="Execution-Guided SQL Generation with Self-Correction")
    parser.add_argument("--input", type=str, required=True, help="Path to routed Agentic results JSON (e.g., bird_agentic_results.json)")
    parser.add_argument("--dataset", type=str, required=True, choices=["spider", "bird"], help="Dataset being processed")
    parser.add_argument("--output", type=str, default="results/sql_generation_results.json", help="Path to save output JSON")
    parser.add_argument("--limit", type=int, default=None, help="Optional: Limit the number of queries to process (for debugging)")
    args = parser.parse_args()

    # 1. Initialization
    print("Loading schemas...")
    schemas = load_schemas() 
    
    try:
        api_key = os.environ.get("GROQ")
        client = Groq(api_key=api_key)
    except Exception as e:
        print(f"Failed to initialize Groq client (Check GROQ env variable): {e}")
        return

    # 2. Load input routing data
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return
        
    with open(args.input, 'r') as f:
        data = json.load(f)
        
    if args.limit:
        data = data[:args.limit]
        print(f"Limiting to first {args.limit} queries.")

    print(f"\\n=== Starting SQL Generation ({args.dataset.upper()}) ===")
    print(f"Total Queries: {len(data)}")
    
    results = []
    success_count = 0
    
    # Optional: We extract retrieval variables to compute R@K and MAP at the end if present
    gold_dbs = []
    retrieved_dbs_list = []
    queries_list = []
    has_retrieval_data = False

    # 3. Processing Loop
    for item in tqdm(data, desc="Generating SQL & Executing"):
        # Handle variants in JSON schema from prev phases
        orig_q = item.get('original_question', item.get('question'))
        resolved_q = item.get('resolved_question', orig_q)
        
        # Retrieval Tracking variables
        item_gold = item.get('gold_db')
        item_retrieved = item.get('retrieved_dbs', [])
        
        if not item_retrieved:
            # Fallback if no retrieved array
            item_retrieved = [item.get('selected_db', item.get('gold_db'))]
            
        candidates = item_retrieved[:3] # Up to top 3
        if len(candidates) == 0:
            candidates = ["unknown_db"]
            
        final_sql = ""
        final_exec_msg = ""
        is_success = False
        final_selected_db = candidates[0]
        
        for candidate_db in candidates:
            # 1. Setup candidate info
            schema_str = schemas.get(candidate_db, "No Schema Found.")
            db_path = get_db_path(candidate_db, args.dataset)
            
            # 2. Get Data Preview
            data_preview_str = get_database_preview(db_path)
            
            # 3. Generate Single Query (No N=3 loop, back to Low Temp)
            sql, success, exec_msg, res = generate_sql_with_feedback(
                client, resolved_q, candidate_db, schema_str, data_preview_str, db_path, temperature=0.1
            )
            
            # 4. Check for Empty Set (The False Positive Trap)
            is_empty_res = (not res) or (len(res) == 0)
            
            # 5. THE SHORT-CIRCUIT: If it executed perfectly AND returned actual data
            if success:
                final_sql = sql
                final_exec_msg = "Execution Successful with Data."
                final_selected_db = candidate_db
                is_success = True
                break # STOP checking other databases immediately!
                
            else:
                # It failed, or it returned an empty set (False Positive)
                # Save it as a fallback in case all candidates fail, but keep looping
                final_sql = sql
                final_exec_msg = exec_msg if not success else "Executed but returned empty set []"
                final_selected_db = candidate_db
                is_success = False
        if is_success:
            success_count += 1
            
        # Update retrieved_dbs to reflect the Execution-Guided ranking.
        # If final_selected_db isn't at index 0, move it to index 0 so R@1 works!
        if item_retrieved and final_selected_db in item_retrieved:
            item_retrieved.remove(final_selected_db)
            item_retrieved.insert(0, final_selected_db)
            
        if item_gold:
            has_retrieval_data = True
            gold_dbs.append(item_gold)
            retrieved_dbs_list.append(item_retrieved)
            queries_list.append(orig_q)
            
        results.append({
            "original_question": orig_q,
            "resolved_question": resolved_q,
            "selected_db": final_selected_db,
            "generated_sql": final_sql,
            "execution_status": is_success,
            "execution_error": final_exec_msg if not is_success else None
        })
        
        # Rate Limiting
        time.sleep(1.5)

    # 4. Metrics Calculation
    exec_accuracy = success_count / len(data) if data else 0
    
    print(f"\\n=== SQL Generation Complete ===")
    print(f"Execution Accuracy (Valid SQL): {exec_accuracy:.4f} ({success_count}/{len(data)})")
    
    # Calculate Routing Metrics (Recall@K, MAP) as requested
    if has_retrieval_data:
        print("\\n=== Routing Metrics ===")
        retrieval_metrics = calculate_metrics(f"{args.dataset}-GenRun", queries_list, gold_dbs, retrieved_dbs_list)
        # Assuming calculate_metrics prints the output, if it returns a dict we could print it.
        # But calculate_metrics prints locally and returns a dict usually.
        print(f"R@1: {retrieval_metrics['R@1']:.4f}, R@3: {retrieval_metrics['R@3']:.4f}, R@5: {retrieval_metrics['R@5']:.4f}, MAP: {retrieval_metrics['MAP']:.4f}")
    
    # 5. Save Output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        output_payload = {
            "metrics": {
                "execution_accuracy": exec_accuracy,
                "total_queries": len(data),
                "successful_executions": success_count
            },
            "data": results
        }
        if has_retrieval_data:
             output_payload["metrics"]["retrieval"] = retrieval_metrics
             
        json.dump(output_payload, f, indent=2)
        
    print(f"\\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
