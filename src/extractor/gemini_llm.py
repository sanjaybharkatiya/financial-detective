"""Google Gemini based Knowledge Graph extractor.

This module implements the LLMExtractor interface using Google's Gemini 1.5 Pro model.
All extraction is performed via LLM reasoning—no regex or pattern matching.
"""

import json
import os
from typing import Final

from google import genai
from google.genai import types

from src.extractor.base import LLMExtractor
from src.schema import KnowledgeGraph

# System prompt that enforces strict extraction rules (same as OpenAI/Ollama)
SYSTEM_PROMPT: Final[str] = """Extract a Knowledge Graph from the financial text.

OUTPUT FORMAT - COPY THIS EXACT STRUCTURE:
{
  "schema_version": "1.0.0",
  "nodes": [
    {"id": "company_1", "type": "Company", "name": "..."},
    {"id": "risk_1", "type": "RiskFactor", "name": "..."},
    {"id": "amount_1", "type": "DollarAmount", "name": "..."}
  ],
  "relationships": [
    {"source": "company_1", "target": "...", "relation": "..."}
  ]
}

CRITICAL RULES:
1. Output ONLY the JSON above - no wrapper, no nesting
2. Top-level keys must be EXACTLY: schema_version, nodes, relationships
3. Do NOT wrap in "knowledge_graph", "data", "result", or any other key
4. Do NOT add "financials", "boardOfDirectors", "auditors", "entities"
5. All nodes go in ONE flat "nodes" array
6. All relationships go in ONE flat "relationships" array

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

# Default model to use for Gemini
DEFAULT_MODEL: Final[str] = "gemini-2.0-flash"


class GeminiExtractor(LLMExtractor):
    """Google Gemini based Knowledge Graph extractor.

    This extractor uses Google's Gemini models to extract structured
    Knowledge Graph data from raw financial text. It enforces strict
    JSON-only output with anti-hallucination instructions.

    Attributes:
        api_key: The Gemini API key used for authentication.
        model: The Gemini model name to use.
        client: The Gemini client instance.

    Example:
        >>> extractor = GeminiExtractor()
        >>> graph = extractor.extract("Acme Corp reported $1.2 billion revenue.")
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize the Gemini extractor.

        Args:
            api_key: Gemini API key. If not provided, reads from
                GEMINI_API_KEY environment variable.
            model: Gemini model name. Defaults to "gemini-1.5-pro".

        Raises:
            ValueError: If no API key is provided and GEMINI_API_KEY
                environment variable is not set.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")

        self.model = model or DEFAULT_MODEL
        self.client = genai.Client(api_key=self.api_key)

    def extract(self, text: str) -> KnowledgeGraph:
        """Extract a Knowledge Graph from raw text using Gemini.

        Sends the input text to Gemini with a strict system prompt
        that enforces JSON-only output conforming to the KnowledgeGraph schema.
        No post-processing or cleanup is applied to the response.

        Args:
            text: Raw financial text to extract entities and relationships from.

        Returns:
            A validated KnowledgeGraph instance containing the extracted
            nodes and relationships.

        Raises:
            ValueError: If the LLM response is empty or cannot be validated.
            google.api_core.exceptions.GoogleAPIError: If the Gemini API call fails.
            json.JSONDecodeError: If the LLM response is not valid JSON.
        """
        response = self.client.models.generate_content(
            model=self.model,
            contents=text,
            config=types.GenerateContentConfig(
                temperature=0,
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
            ),
        )

        content = response.text
        if content is None:
            raise ValueError("LLM returned empty response")

        # Strip markdown code blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)

        return KnowledgeGraph.model_validate(json.loads(content))
