"""OpenAI GPT-4o based Knowledge Graph extractor.

This module implements the LLMExtractor interface using OpenAI's GPT-4o model.
All extraction is performed via LLM reasoning—no regex or pattern matching.
"""

import json
import os

from openai import OpenAI
from typing import Final

from src.extractor.base import LLMExtractor
from src.schema import KnowledgeGraph

# System prompt that enforces strict extraction rules
SYSTEM_PROMPT: Final[str] = """Extract a Knowledge Graph from the financial text.

OUTPUT FORMAT - COPY THIS EXACT STRUCTURE:
{{
  "schema_version": "1.0.0",
  "nodes": [
    {{"id": "company_1", "type": "Company", "name": "..."}},
    {{"id": "risk_1", "type": "RiskFactor", "name": "..."}},
    {{"id": "amount_1", "type": "DollarAmount", "name": "..."}}
  ],
  "relationships": [
    {{"source": "company_1", "target": "...", "relation": "..."}}
  ]
}}

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

{{
  "schema_version": "1.0.0",
  "nodes": [
    {{"id": "company_1", "type": "Company", "name": "Company Name"}},
    {{"id": "risk_1", "type": "RiskFactor", "name": "Risk description"}},
    {{"id": "amount_1", "type": "DollarAmount", "name": "$X"}}
  ],
  "relationships": [
    {{"source": "company_1", "target": "risk_1", "relation": "HAS_RISK"}},
    {{"source": "company_1", "target": "amount_1", "relation": "REPORTS_AMOUNT"}},
    {{"source": "company_1", "target": "company_2", "relation": "OWNS"}}
  ]
}}

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


def _get_schema_json() -> str:
    """Generate JSON schema representation for the KnowledgeGraph model.

    Returns:
        A JSON string representing the KnowledgeGraph schema.
    """
    return json.dumps(KnowledgeGraph.model_json_schema(), indent=2)


def _build_system_prompt() -> str:
    """Build the complete system prompt with injected schema.

    Returns:
        The system prompt string with the KnowledgeGraph schema embedded.
    """
    return SYSTEM_PROMPT.format(schema=_get_schema_json())


class OpenAIExtractor(LLMExtractor):
    """OpenAI GPT-4o based Knowledge Graph extractor.

    This extractor uses OpenAI's GPT-4o model to extract structured
    Knowledge Graph data from raw financial text. It enforces strict
    JSON-only output with anti-hallucination instructions.

    Attributes:
        api_key: The OpenAI API key used for authentication.
        client: The OpenAI client instance.

    Example:
        >>> extractor = OpenAIExtractor()
        >>> graph = extractor.extract("Acme Corp reported $1.2 billion revenue.")
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

        Sends the input text to OpenAI GPT-4o with a strict system prompt
        that enforces JSON-only output conforming to the KnowledgeGraph schema.
        No post-processing or cleanup is applied to the response.

        Args:
            text: Raw financial text to extract entities and relationships from.

        Returns:
            A validated KnowledgeGraph instance containing the extracted
            nodes and relationships.

        Raises:
            ValueError: If the LLM response is empty or cannot be validated.
            openai.APIError: If the OpenAI API call fails.
            json.JSONDecodeError: If the LLM response is not valid JSON.
        """
        response = self.client.responses.create(
            model="gpt-4o",
            temperature=0,
            input=[
                {"role": "system", "content": _build_system_prompt()},
                {"role": "user", "content": text},
            ],
        )

        content = response.output_text

        if content is None:
            raise ValueError("LLM returned empty response")

        return KnowledgeGraph.model_validate(json.loads(content))

