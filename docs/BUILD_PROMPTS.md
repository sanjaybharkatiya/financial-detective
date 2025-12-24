# Build Prompts for Financial Detective

This document contains all the prompts/tasks used to build the Financial Detective project.

---

## 1. Schema Definition

```
Create src/schema.py.
Requirements:
- Use Pydantic v2
- Define a strict Knowledge Graph schema
Root model:
- schema_version: str
- nodes: List[Node]
- relationships: List[Relationship]
Node:
- id: str
- type: Literal["Company", "RiskFactor", "MonetaryAmount"]
- name: str
Relationship:
- source: str
- target: str
- relation: Literal["OWNS", "HAS_RISK", "REPORTS_AMOUNT"]
Rules:
- No optional fields
- No extra fields allowed
- Add docstrings (Google style)
- Add type hints everywhere
```

---

## 2. Input Loader

```
Create src/input_loader.py.
Requirements:
- Load raw financial text from data/raw_report.txt
- Do not clean, normalize, split, or parse the text
- Preserve formatting and line breaks
- Return the full text as a single string
- Add Google-style module and function docstrings
- Use Python type hints
Expose function: load_raw_text() -> str
```

---

## 3. Extractor (Initial)

```
Create src/extractor.py.
Requirements:
- Use OpenAI GPT-4o via the official Python SDK
- Load API key from environment variable OPENAI_API_KEY
- Temperature must be 0
- Use a strict system prompt that:
  - Forbids regex and pattern matching
  - Forbids hallucination or inference
  - Forces JSON-only output
- Inject the KnowledgeGraph Pydantic schema into the prompt
- Parse the LLM response into a Python dict
- Do not perform any post-processing or cleanup
- Add Google-style docstrings and Python type hints
Expose function: extract_knowledge_graph(text: str) -> dict
```

---

## 4. Validator

```
Create src/validator.py.
Requirements:
- Accept a KnowledgeGraph object
- Validate the following:
  1. All node IDs are unique
  2. Every relationship source and target refers to an existing node ID
- Raise descriptive ValueError exceptions on failure
- Do NOT modify the KnowledgeGraph
- Add Google-style docstrings
- Use Python type hints
Expose function: validate_knowledge_graph(graph: KnowledgeGraph) -> None
```

---

## 5. Visualizer

```
Create src/visualizer.py.
Requirements:
- Accept a validated KnowledgeGraph object
- Build a directed NetworkX graph
- Add nodes with attributes:
  - label = node.name
  - type = node.type
- Color nodes by type:
  - Company: blue
  - RiskFactor: red
  - DollarAmount: green
- Add directed edges with relationship labels
- Render graph using matplotlib
- Save image to visuals/graph.png
- Do not display interactively
- Add Google-style docstrings and Python type hints
Expose function: render_graph(graph: KnowledgeGraph) -> None
```

---

## 6. Main Orchestration

```
Create main.py.
Requirements:
- Load raw text using input_loader.load_raw_text
- Extract KnowledgeGraph using extractor.extract_knowledge_graph
- Validate using validator.validate_knowledge_graph
- Save graph_output.json to data/ directory
- Render graph image using visualizer.render_graph
- Print clear progress logs for each step
- Exit with non-zero status on failure
- Add Google-style docstrings and Python type hints
```

---

## 7. Tests

### test_validator.py
```
Create tests/test_validator.py:
Write pytest tests for validate_knowledge_graph.
Tests:
- Valid graph passes
- Duplicate node IDs fail
- Invalid relationship references fail
```

### test_extractor.py
```
Create tests/test_extractor.py:
Write pytest tests for extract_knowledge_graph.
Requirements:
- Mock OpenAI client responses
- Ensure valid JSON is parsed into KnowledgeGraph
- Ensure invalid JSON raises error
```

### test_visualizer.py
```
Create tests/test_visualizer.py:
Write pytest tests for render_graph.
Requirements:
- Use a temporary directory
- Ensure graph image file is created
```

---

## 8. Configuration Module

```
Create src/config.py.
Requirements:
- Use a Pydantic BaseModel for configuration
- Support LLM_PROVIDER with allowed values: "openai" | "ollama"
- Support OLLAMA_MODEL (default: "llama3")
- Read values from environment variables
- Provide a function load_config() that returns a validated config object
- Include Google-style docstrings
- No business logic here, config only
```

---

## 9. Abstract Base Class for Extractors

```
Create src/extractor/base.py.
Requirements:
- Define an abstract base class named LLMExtractor
- Use abc.ABC
- Define one abstract method: extract(text: str) -> KnowledgeGraph
- Import KnowledgeGraph from src.schema
- Add clear Google-style docstrings explaining the contract
- No implementation logic
```

---

## 10. OpenAI Extractor

```
Create src/extractor/openai_llm.py.
Requirements:
- Implement the LLMExtractor interface
- Move existing GPT-4o extraction logic from extractor.py into this class
- Use OpenAI client (responses.create)
- Keep temperature=0
- Reuse the existing system prompt and schema injection logic
- Validate output using KnowledgeGraph.model_validate
- Raise ValueError if OPENAI_API_KEY is missing
- No regex, no post-processing
- Include full Google-style docstrings
```

---

## 11. Ollama Extractor

```
Create src/extractor/ollama_llm.py.
Requirements:
- Implement the LLMExtractor interface
- Use HTTP POST to Ollama API (default: http://localhost:11434/api/generate)
- Model name must be configurable
- Send the SAME system prompt and schema as OpenAI extractor
- Enforce JSON-only output
- Parse response and validate with KnowledgeGraph.model_validate
- Raise ValueError if Ollama is not running or returns invalid JSON
- No regex, no post-processing
- Add full Google-style docstrings
```

---

## 12. Factory Pattern

```
Create src/extractor/factory.py.
Requirements:
- Load configuration using src.config.load_config
- If llm_provider == "openai", return OpenAIExtractor
- If llm_provider == "ollama", return OllamaExtractor
- Pass required config values into constructors
- Raise ValueError for unsupported providers
- Keep the file small and readable
- Add Google-style docstrings
```

---

## 13. Refactor Extractor

```
Refactor src/extractor.py.
Requirements:
- Remove any direct OpenAI or Ollama logic
- Import create_extractor from src.extractor.factory
- extract_knowledge_graph(text: str) should:
  - create the extractor via factory
  - call extractor.extract(text)
  - return KnowledgeGraph
- Keep the function signature unchanged
- Update docstrings accordingly
- Do not modify main.py
```

---

## 14. Dotenv Support

```
Add dotenv support:
1. Add python-dotenv to requirements.txt
2. In src/config.py, import and call load_dotenv() at module load time
3. Use environment variables as the single source of truth
4. Add .env to .gitignore
5. Create .env.example with example configuration
6. Update README with .env documentation
```

---

## 15. Google Gemini Support

```
Add support for a new LLM provider: gemini.
Requirements:
- Add "gemini" as a valid LLM_PROVIDER in src/config.py
- Add GEMINI_API_KEY configuration (environment variable)
- Create src/extractor/gemini_llm.py implementing LLMExtractor
- Use google-generativeai SDK with model gemini-1.5-pro
- Enforce the SAME strict JSON-only extraction rules used by OpenAI/Ollama
- No schema, validator, or main.py changes
- Update extractor factory to route gemini provider
- Add dependency to requirements.txt
- Keep architecture provider-agnostic and fail-fast
```

---

## 16. Mermaid Visualization

```
Add Mermaid diagram generation as an optional visualization output.
Requirements:
- Do NOT remove or change existing NetworkX visualization
- Add Mermaid as an additional optional output
- Create src/visualizer_mermaid.py with function:
  def render_mermaid(graph: KnowledgeGraph, output_path: Path) -> None
- Use:
  - Company → rectangle
  - RiskFactor → rounded
  - DollarAmount → parallelogram
- Write output to visuals/graph.mmd
- Update main.py to generate both outputs
- Update README.md with Mermaid documentation
```

---

## 17. Optional Confidence Scores

```
Update schema and prompts to support optional confidence scores:
1. Add optional field to Relationship: confidence: float | None
2. Valid range: 0.0 to 1.0
3. Keep extra="forbid"
4. Update prompts to allow optional confidence field
5. Validators must continue to pass if confidence is missing
```

---

## 18. Ownership Extraction Rules

```
Update SYSTEM_PROMPT in both OpenAI and Ollama extractors:
Add OWNERSHIP EXTRACTION RULES (MANDATORY):
- Create OWNS relationship ONLY when text explicitly states ownership
- Keywords: "subsidiary", "owns", "owned by", "wholly owned", "parent company"
- If Entity A is subsidiary of Entity B: source = B, target = A
- If no explicit ownership language, DO NOT create OWNS
```

---

## 19. Relationship Type Constraints

```
Update prompts with RELATIONSHIP TYPE CONSTRAINTS (MANDATORY):
- HAS_RISK: source MUST be Company, target MUST be RiskFactor
- REPORTS_AMOUNT: source MUST be Company, target MUST be DollarAmount
- OWNS: source MUST be Company, target MUST be Company

Add EXTRACTION COMPLETENESS RULE (MANDATORY):
- If text mentions risks → extract RiskFactor nodes and HAS_RISK
- If text mentions monetary values → extract DollarAmount and REPORTS_AMOUNT
- If text mentions subsidiaries → extract OWNS relationships
```

---

## 20. Python Version Detection

```
Enhance main.py to auto-detect Python version at runtime.
If Python >= 3.14 and NetworkX unavailable:
- Print friendly explanatory message
- Suggest using Python 3.11–3.13 for PNG rendering
- Do NOT raise errors or change execution flow
- Keep Mermaid rendering always enabled
```

---

## 21. README Updates

```
Update README.md:
1. Add "Supported Providers" table with OpenAI, Gemini, Ollama
2. Add environment variables table with all config options
3. Add usage examples for each provider
4. Add Design Highlights section
5. Add Mermaid Visualization section
6. Update Prerequisites to be provider-neutral
7. Update Limitations section for multi-provider support
```

---

## 22. Technical Design Document

```
Create/Update docs/TECHNICAL_DESIGN.md:
1. Executive Summary with provider-agnostic architecture
2. High-Level Architecture with extractor package
3. Multi-Provider LLM Extraction Strategy section
4. JSON Schema Design with optional confidence field
5. Visualization Strategy with Mermaid support
6. Limitations & Assumptions (updated)
7. Future Enhancements
8. Appendix A: Module Dependency Graph
9. Appendix B: External Dependencies
```

---

## Document Revision History

| Version | Date | Description |
|---------|------|-------------|
| 1.0 | December 2024 | Initial project build prompts |
