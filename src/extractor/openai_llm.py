"""OpenAI GPT-4o based Knowledge Graph extractor.

This module implements the LLMExtractor interface using OpenAI's GPT-4o model.
All extraction is performed via LLM reasoning—no regex or pattern matching.

Features:
- Unified prompt matching Ollama/Gemini
- Automatic JSON repair for common LLM mistakes
- Relation type normalization
"""

import json
import os
from typing import Any, Final

from openai import OpenAI

from src.extractor.base import LLMExtractor
from src.schema import KnowledgeGraph

# Valid relation types (same as Ollama/Gemini)
VALID_RELATIONS: Final[set[str]] = {
    "OWNS", "HAS_RISK", "REPORTS_AMOUNT", "OPERATES", "IMPACTED_BY",
    "DECLINED_DUE_TO", "SUPPORTED_BY", "PARTNERED_WITH", "JOINT_VENTURE_WITH",
    "RAISED_CAPITAL", "INVESTED_IN", "COMMITTED_CAPEX", "TARGETS",
    "PLANS_TO", "ON_TRACK_TO", "COMMITTED_TO", "COMPLIES_WITH", "SUBJECT_TO",
}

# Map common LLM mistakes to valid relation types
RELATION_MAPPING: Final[dict[str, str]] = {
    "SUBSIDIARY": "OWNS",
    "SUBSIDIARY_OF": "OWNS",
    "PART_OF": "OWNS",
    "PART_OF_GROUP": "OWNS",
    "BELONGS_TO": "OWNS",
    "JOINT_VENTURE": "JOINT_VENTURE_WITH",
    "JV": "JOINT_VENTURE_WITH",
    "PARTNER": "PARTNERED_WITH",
    "FACES_RISK": "HAS_RISK",
    "REPORTED": "REPORTS_AMOUNT",
    "REVENUE": "REPORTS_AMOUNT",
}

# Unified prompt (same rules as Ollama/Gemini)
# Note: Using {{ and }} for literal braces in f-string compatible format
SYSTEM_PROMPT: Final[str] = """You are an information extraction engine.

TASK
Extract a Knowledge Graph from the text below and return ONLY valid JSON.

ENTITIES:
Company, RiskFactor, DollarAmount

RELATIONS:
REPORTS_AMOUNT, HAS_RISK, OWNS, OPERATES,
PARTNERED_WITH, JOINT_VENTURE_WITH,
IMPACTED_BY, DECLINED_DUE_TO, SUPPORTED_BY,
RAISED_CAPITAL, INVESTED_IN, COMMITTED_CAPEX,
TARGETS, PLANS_TO, ON_TRACK_TO, COMMITTED_TO,
COMPLIES_WITH, SUBJECT_TO

OWNERSHIP RULES:
- Subsidiary or JV cannot own parent.
- If A is subsidiary/JV of B, relation is B → A.
- If unclear, use PARTNERED_WITH or JOINT_VENTURE_WITH.

RULES:
- Keep names exactly as written.
- Extract all monetary values (₹, INR, USD, crore, billion, million).
- Each amount is a separate DollarAmount node.
- Extract risks when text mentions risk, volatility, regulatory,
  geopolitical, legal, compliance, margin, slowdown.
- Deduplicate companies by full legal name.
- IDs: company_1, amount_1, risk_1, etc.
- Every node must include a short context.
- Output ONLY JSON. No text.

EXAMPLE OUTPUT:
{
  "schema_version": "1.0.0",
  "nodes": [
    {"id": "company_1", "type": "Company", "name": "Reliance Industries", "context": "Parent company"},
    {"id": "company_2", "type": "Company", "name": "Reliance Retail", "context": "Subsidiary"},
    {"id": "amount_1", "type": "DollarAmount", "name": "₹10,71,174 crore", "context": "Revenue FY2024"},
    {"id": "risk_1", "type": "RiskFactor", "name": "market volatility", "context": "Impacts margins"}
  ],
  "relationships": [
    {"source": "company_1", "target": "company_2", "relation": "OWNS"},
    {"source": "company_1", "target": "amount_1", "relation": "REPORTS_AMOUNT"},
    {"source": "company_1", "target": "risk_1", "relation": "HAS_RISK"}
  ]
}"""


def _extract_json(content: str) -> str:
    """Extract JSON from response, handling markdown fences."""
    content = content.strip()
    
    # Strip markdown code blocks
    if content.startswith("```"):
        lines = content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    
    # Find JSON object
    first = content.find("{")
    last = content.rfind("}")
    if first != -1 and last > first:
        return content[first:last + 1]
    return content


def _fix_malformed_nodes(data: dict[str, Any]) -> dict[str, Any]:
    """Fix common LLM JSON mistakes in nodes."""
    if "nodes" not in data:
        return data
    
    fixed_nodes = []
    for node in data["nodes"]:
        if isinstance(node, dict):
            fixed_node = {}
            for key, value in node.items():
                if ": " in key and key.startswith("id:"):
                    id_value = key.split(": ", 1)[1].strip()
                    fixed_node["id"] = id_value
                    fixed_node["type"] = value
                else:
                    fixed_node[key] = value
            
            if "id" in fixed_node and "type" in fixed_node:
                if "name" not in fixed_node:
                    fixed_node["name"] = fixed_node.get("context", "Unknown")
                if "context" not in fixed_node:
                    fixed_node["context"] = ""
                fixed_nodes.append(fixed_node)
            elif "id" in node and "type" in node:
                if "context" not in node:
                    node["context"] = ""
                fixed_nodes.append(node)
    
    data["nodes"] = fixed_nodes
    return data


def _normalize_relations(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize relation types to valid values."""
    if "relationships" not in data:
        return data
    
    valid_rels = []
    for rel in data["relationships"]:
        if isinstance(rel, dict) and "relation" in rel:
            relation = rel["relation"].upper().replace(" ", "_")
            
            if relation in VALID_RELATIONS:
                rel["relation"] = relation
                valid_rels.append(rel)
            elif relation in RELATION_MAPPING:
                rel["relation"] = RELATION_MAPPING[relation]
                valid_rels.append(rel)
    
    data["relationships"] = valid_rels
    return data


class OpenAIExtractor(LLMExtractor):
    """OpenAI GPT-4o based Knowledge Graph extractor.

    Uses unified prompt matching Ollama/Gemini for consistent extraction.
    Includes automatic JSON repair and relation normalization.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the OpenAI extractor.

        Args:
            api_key: OpenAI API key. If not provided, reads from
                OPENAI_API_KEY environment variable.

        Raises:
            ValueError: If no API key is provided and OPENAI_API_KEY
                environment variable is not set.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.client = OpenAI(api_key=self.api_key)

    def extract(self, text: str) -> KnowledgeGraph:
        """Extract a Knowledge Graph from raw text using GPT-4o.

        Args:
            text: Raw financial text to extract entities from.

        Returns:
            A validated KnowledgeGraph instance.

        Raises:
            ValueError: If the LLM response is empty or invalid.
        """
        response = self.client.responses.create(
            model="gpt-4o",
            temperature=0,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        )

        content = response.output_text
        if content is None:
            raise ValueError("LLM returned empty response")

        json_str = _extract_json(content)
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {json_str[:200]}") from e

        # Ensure required top-level keys
        if "schema_version" not in data:
            data["schema_version"] = "1.0.0"
        if "nodes" not in data:
            data["nodes"] = []
        if "relationships" not in data:
            data["relationships"] = []

        # Fix common LLM mistakes
        data = _fix_malformed_nodes(data)
        data = _normalize_relations(data)

        return KnowledgeGraph.model_validate(data)
