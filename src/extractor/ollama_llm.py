"""Ollama-based Knowledge Graph extractor.

This module implements the LLMExtractor interface using a local Ollama instance.
All extraction is performed via LLM reasoning—no regex or pattern matching.
"""

import json
from typing import Final

import httpx

from src.extractor.base import LLMExtractor
from src.schema import KnowledgeGraph

# Default Ollama API endpoint
DEFAULT_BASE_URL: Final[str] = "http://localhost:11434"
DEFAULT_MODEL: Final[str] = "llama3:latest"


# System prompt optimized for Ollama models
SYSTEM_PROMPT: Final[str] = """You are a financial document analyst that extracts structured knowledge graphs.

STRICT EXTRACTION RULES (MANDATORY):

1. Extract ONLY information explicitly stated in the text.
2. DO NOT infer, assume, or hallucinate facts.
3. DO NOT omit any explicitly stated companies, risks, amounts, or ownerships.
4. Output ONLY valid JSON. No explanations. No markdown. No code blocks.

ENTITY TYPES:
- Company
- RiskFactor
- DollarAmount

RELATIONSHIP TYPES:
- OWNS
- HAS_RISK
- REPORTS_AMOUNT

OWNERSHIP EXTRACTION RULES (MANDATORY):
Create an OWNS relationship ONLY when the text explicitly states ownership or subsidiary language, including:
- "subsidiary"
- "owns"
- "owned by"
- "wholly owned"
- "parent company"
- "X owns Y"
- "Y is a subsidiary of X"

If the text states that Entity A is a subsidiary of Entity B:
- source = Entity B
- target = Entity A

If no explicit ownership language exists, DO NOT create OWNS.

RELATIONSHIP TYPE CONSTRAINTS (MANDATORY):
- HAS_RISK:
  - source MUST be a Company
  - target MUST be a RiskFactor
  - NEVER link HAS_RISK to a Company
- REPORTS_AMOUNT:
  - source MUST be a Company
  - target MUST be a DollarAmount
- OWNS:
  - source MUST be a Company
  - target MUST be a Company

EXTRACTION COMPLETENESS RULE (MANDATORY):
If the text explicitly mentions:
- risks → you MUST extract RiskFactor nodes and HAS_RISK relationships
- monetary values → you MUST extract DollarAmount nodes and REPORTS_AMOUNT relationships
- subsidiaries or ownership → you MUST extract OWNS relationships

OUTPUT FORMAT:
Return a single JSON object with this exact structure:

{
  "schema_version": "1.0.0",
  "nodes": [
    {"id": "company_1", "type": "Company", "name": "Company Name"},
    {"id": "risk_1", "type": "RiskFactor", "name": "Risk description"},
    {"id": "amount_1", "type": "DollarAmount", "name": "$X"}
  ],
  "relationships": [
    {"source": "company_1", "target": "risk_1", "relation": "HAS_RISK"},
    {"source": "company_1", "target": "amount_1", "relation": "REPORTS_AMOUNT"},
    {"source": "company_1", "target": "company_2", "relation": "OWNS"}
  ]
}

ID RULES:
- Use incremental IDs: company_1, company_2, risk_1, amount_1, etc.
- IDs must be unique.
- Relationships MUST reference valid node IDs.

OPTIONAL CONFIDENCE SCORE:
- You MAY include a 'confidence' field (0.0–1.0) on relationships
- Use higher confidence (≥0.9) only when the relationship is explicitly stated
- Omit the confidence field if unsure
- Absence of confidence is acceptable

Respond with ONLY the JSON object."""


class OllamaExtractor(LLMExtractor):
    """Ollama-based Knowledge Graph extractor.

    This extractor uses a local Ollama instance to extract structured
    Knowledge Graph data from raw financial text. It enforces strict
    JSON-only output with anti-hallucination instructions.

    Attributes:
        base_url: The Ollama API base URL.
        model: The Ollama model name to use.

    Example:
        >>> extractor = OllamaExtractor(model="llama3")
        >>> graph = extractor.extract("Acme Corp reported $1.2 billion revenue.")
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        """Initialize the Ollama extractor.

        Args:
            model: The Ollama model name to use. Defaults to "llama3".
            base_url: The Ollama API base URL.
                Defaults to "http://localhost:11434".
        """
        self.model = model
        self.base_url = base_url.rstrip("/")

    def extract(self, text: str) -> KnowledgeGraph:
        """Extract a Knowledge Graph from raw text using Ollama.

        Sends the input text to a local Ollama instance with a strict
        system prompt that enforces JSON-only output conforming to the
        KnowledgeGraph schema. No post-processing or cleanup is applied.

        Args:
            text: Raw financial text to extract entities and relationships from.

        Returns:
            A validated KnowledgeGraph instance containing the extracted
            nodes and relationships.

        Raises:
            ValueError: If Ollama is not running, returns empty response,
                or returns invalid JSON.
            httpx.ConnectError: If the Ollama server is unreachable.
            json.JSONDecodeError: If the LLM response is not valid JSON.
        """
        prompt = f"{SYSTEM_PROMPT}\n\n---\n\nDocument to analyze:\n\n{text}\n\n---\n\nExtracted JSON:"

        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0,
                    },
                },
                timeout=120.0,
            )
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise ValueError(
                f"Ollama is not running at {self.base_url}. "
                "Please start Ollama with 'ollama serve'."
            ) from e
        except httpx.HTTPStatusError as e:
            raise ValueError(
                f"Ollama API error: {e.response.status_code} - {e.response.text}"
            ) from e

        try:
            response_data = response.json()
        except json.JSONDecodeError as e:
            raise ValueError(f"Ollama returned invalid JSON response: {e}") from e

        content = response_data.get("response")

        if not content:
            raise ValueError("Ollama returned empty response")

        # Strip any markdown code blocks if present
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first line (```json) and last line (```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines)

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM response is not valid JSON: {content[:500]}"
            ) from e

        return KnowledgeGraph.model_validate(parsed)
