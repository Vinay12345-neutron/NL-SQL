
import os
import json
import re
from typing import List, Dict, Tuple
from groq import Groq

# Load environment variables if not already loaded
from dotenv import load_dotenv
load_dotenv()

class AgenticTransducer:
    def __init__(self, schemas: Dict[str, str], model_name: str = "llama-3.1-8b-instant"):
        self.schemas = schemas
        self.model_name = model_name
        self.client = None
        
        api_key = os.environ.get("GROQ")
        if api_key:
            try:
                self.client = Groq(api_key=api_key)
            except Exception as e:
                print(f"Warning: Failed to initialize Groq client: {e}")
        else:
            print("Warning: GROQ API Key not found in environment.")

    def _call_llm(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        if not self.client:
            return "Error: Client not initialized"
            
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                temperature=0.1, # Low temp for deterministic logic
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error calling LLM: {e}"

    def _get_context_str(self, candidates: List[str]) -> str:
        context_str = ""
        for db in candidates:
            # Increase limit to 8000 chars to ensure full schema visibility
            s_text = self.schemas.get(db, "")[:8000] 
            context_str += f"Database: {db}\nSchema: {s_text}\n\n"
        return context_str

    def classify(self, query: str, candidates: List[str]) -> Tuple[str, str]:
        """
        Action (a): Classify the question.
        Uses exact instruction from Paper Appendix B.2.
        """
        context_str = self._get_context_str(candidates)

        prompt = f"""
        You are a classifier responsible for analyzing a question strictly based on the provided context.
        Your job is to classify the question into one of the following categories: "Normal", "Incomplete", and "Ambiguous".
        Make your classification solely based on the question and the given context.
        Include a brief explanation for your classification.
        Ensure the response is formatted exactly as:
        Classification: <class label>
        Explanation: <explanation>
        
        Context:
        {context_str}

        Question: {query}
        """
        
        response = self._call_llm(prompt, system_prompt="You are a classifier.")
        
        # Parse Text Format
        # Expected:
        # Classification: Normal
        # Explanation: ...
        
        label = "Normal"
        reasoning = "No reasoning provided."
        
        try:
            lines = response.strip().split('\n')
            for line in lines:
                if line.lower().startswith("classification:"):
                    raw_label = line.split(":", 1)[1].strip().strip("'").strip('"')
                    # Normalize label
                    if "ambiguous" in raw_label.lower():
                        label = "Ambiguous"
                    elif "incomplete" in raw_label.lower():
                        label = "Incomplete"
                    else:
                        label = "Normal"
                elif line.lower().startswith("explanation:"):
                    reasoning = line.split(":", 1)[1].strip()
        except Exception as e:
            print(f"Parse Error in Classify: {e}. Resp: {response}")
            
        return label, reasoning

    def resolve(self, query: str, candidates: List[str], label: str) -> str:
        """
        Action (b): Resolve.
        Uses exact instruction from Paper Appendix B.2.
        """
        context_str = self._get_context_str(candidates)
            
        prompt = f"""
        You are a helpful assistant. Your task is to resolve incomplete or ambiguous questions by rewriting them into clear and complete questions based on the provided context and explanation.
        
        Context:
        {context_str}
        
        Question: {query}
        Classification: {label}
        
        Format your response as:
        Resolved: <resolved question>
        """
        
        response = self._call_llm(prompt, system_prompt="You are a helpful assistant.")
        
        # Parse "Resolved: ..."
        resolved_q = query
        try:
            for line in response.strip().split('\n'):
                if line.lower().startswith("resolved:"):
                    resolved_q = line.split(":", 1)[1].strip()
                    break
            # Fallback if no prefix found, maybe whole response is query?
            if resolved_q == query and len(response) < len(query) * 2:
                # Heuristic: if response is short and no prefix, assume it is the query
                cleaned = response.replace("Resolved:", "").strip()
                if cleaned:
                    resolved_q = cleaned
        except:
            pass
            
        return resolved_q

    def answer(self, query: str, candidates: List[str]) -> str:
        """
        Action (c): Answer (Select Database).
        Uses Groq JSON mode to guarantee precise string matching without regex.
        """
        context_str = self._get_context_str(candidates)
            
        prompt = f"""
        Context:
        {context_str}
        
        Question: {query}
        
        Task: Identify the Database ID that contains the information to answer the question.
        You MUST select exactly one ID from this exact list: {candidates}
        
        Strategy: Focus on exact table/column name matches before semantic coverage.
        
        Respond strictly in JSON format with the key "database_id".
        """
        
        if not self.client:
            print("[Error] Groq client not initialized in answer step.")
            return candidates[0]
            
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a routing agent. You only output valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                temperature=0.0,
                response_format={"type": "json_object"} # Forces perfect JSON output
            )
            
            response_text = chat_completion.choices[0].message.content
            data = json.loads(response_text)
            pred_db = data.get("database_id", candidates[0])
            
            # Verify the LLM didn't hallucinate an ID
            if pred_db in candidates:
                return pred_db
            else:
                print(f"\n[Warn] LLM hallucinated ID: {pred_db}. Falling back.")
                return candidates[0]
                
        except Exception as e:
            print(f"\n[Error] Answer LLM Failed: {e}")
            return candidates[0]

