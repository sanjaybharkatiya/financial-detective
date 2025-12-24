# Technical Design Document: The Financial Detective

**Version:** 1.0  
**Date:** December 2024  
**Author:** Architecture Team  
**Status:** Final

---

## 1. Executive Summary

The Financial Detective is an LLM-powered extraction pipeline that transforms unstructured financial text—such as excerpts from public company annual reports and enterprise financial disclosures—into a validated, structured Knowledge Graph. The system addresses the fundamental challenge of extracting entities (companies, risk factors, monetary amounts) and their relationships from natural language financial disclosures without relying on brittle regex or rule-based pattern matching.

The solution leverages pluggable LLM providers (OpenAI GPT-4o, Google Gemini, and local Ollama models) with a carefully designed prompt that enforces JSON-only output conforming to a strict Pydantic schema. Provider selection is driven by environment variables and implemented via a factory pattern, enabling seamless switching between cloud and local inference without code changes. All extraction is performed through LLM reasoning; no post-processing, cleanup, or pattern matching is applied. The output is a validated JSON Knowledge Graph accompanied by visual graph representations (NetworkX PNG and Mermaid diagrams) for human verification.

**Key Constraints:**
- No regex or pattern-based extraction
- All entity and relationship extraction via LLM reasoning
- Strict schema enforcement: optional fields allowed only if explicitly defined in schema (e.g., `confidence`); undocumented fields remain forbidden via `extra="forbid"`
- Fail-fast on invalid data; no silent error suppression

---

## 2. Problem Statement

### The Challenge

Financial reports contain critical structured information embedded in unstructured prose:

- **Ownership structures** between corporate entities
- **Risk disclosures** tied to specific business units
- **Monetary figures** associated with revenue, liabilities, and investments

This information is essential for compliance analysis, investment research, and regulatory reporting. However, it exists in narrative form with:

- Inconsistent terminology across documents
- Complex sentence structures with nested clauses
- Cross-references and coreference (pronouns referring to earlier entities)
- Implicit relationships requiring contextual inference

### Why Regex Fails

Rule-based extraction using regular expressions is fundamentally unsuitable for this domain:

| Challenge | Regex Limitation |
|-----------|------------------|
| **Phrasing Variation** | "$1.2 billion," "1,200 million dollars," "revenue of $9.5 billion" require separate patterns |
| **Contextual Meaning** | "The company" requires coreference resolution to identify the referent |
| **Negation Handling** | "No material risk was identified" creates false positives for naive patterns |
| **Relationship Extraction** | Multi-sentence relationships ("Parent owns Subsidiary. Subsidiary reported...") cannot be captured |
| **Maintenance Burden** | Each new document format requires pattern updates |

Regex approaches exhibit O(n) maintenance cost where n is the number of document variations encountered. This is unsustainable for production financial analysis.

### Design Decision

The system uses LLM-based extraction exclusively. This trades computational cost (API calls) for:

1. Zero maintenance of extraction rules
2. Generalization across phrasing variations
3. Semantic understanding of relationships
4. Contextual disambiguation

---

## 3. High-Level Architecture

### Pipeline Overview

The system follows a strict linear pipeline with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PIPELINE FLOW                                   │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────┐
  │ raw_report  │
  │   .txt      │
  └──────┬──────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                        input_loader.py                          │
  │  • Read file as single string                                   │
  │  • Preserve all formatting                                      │
  │  • No parsing, cleaning, or normalization                       │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                      extractor/ package                         │
  │  ┌───────────────────────────────────────────────────────────┐  │
  │  │  factory.py: Environment-driven provider selection        │  │
  │  │    ├── OpenAI GPT-4o (cloud)                              │  │
  │  │    ├── Google Gemini (cloud)                              │  │
  │  │    └── Ollama (local)                                     │  │
  │  ├───────────────────────────────────────────────────────────┤  │
  │  │  All providers share:                                     │  │
  │  │  • System prompt with schema injection                    │  │
  │  │  • Anti-hallucination instructions                        │  │
  │  │  • JSON-only output enforcement                           │  │
  │  │  • Temperature = 0 for determinism                        │  │
  │  └───────────────────────────────────────────────────────────┘  │
  │  • Parse JSON response                                          │
  │  • Validate against Pydantic schema                             │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │
                                 ▼
                        ┌───────────────────┐
                        │  KnowledgeGraph   │
                        │    (Pydantic)     │
                        └─────────┬─────────┘
                                  │
                                  ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                         validator.py                            │
  │  • Check: at least one node exists                              │
  │  • Check: all node IDs are unique                               │
  │  • Check: all relationship references are valid                 │
  │  • Raise ValueError on any violation                            │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                     │
              ▼                                     ▼
  ┌───────────────────────┐             ┌───────────────────────┐
  │   graph_output.json   │             │     visualizer.py     │
  │                       │             │  • Build NetworkX     │
  │  Serialized via       │             │    DiGraph            │
  │  model_dump_json()    │             │  • Color nodes by     │
  │                       │             │    type               │
  └───────────────────────┘             │  • Render to PNG      │
                                        └───────────┬───────────┘
                                                    │
                                                    ▼
                                        ┌───────────────────────┐
                                        │      graph.png        │
                                        └───────────────────────┘
```

### Component Isolation

Each module has a single responsibility:

| Module | Input | Output | Side Effects |
|--------|-------|--------|--------------|
| `input_loader.py` | File path | Raw string | None |
| `chunker.py` | Raw string | List of chunks | None |
| `graph_merger.py` | List of KnowledgeGraphs | Single KnowledgeGraph | None |
| `extractor/` | Raw string | KnowledgeGraph | LLM API call (provider-dependent) |
| `validator.py` | KnowledgeGraph | None (or raises) | None |
| `visualizer.py` | KnowledgeGraph | None | Writes PNG file (optional on Python 3.14) |
| `visualizer_mermaid.py` | KnowledgeGraph | None | Writes .mmd file (always available) |
| `main.py` | None | Exit code | Orchestrates pipeline |

This separation enables independent testing, clear error attribution, and future component replacement without system-wide changes.

---

## 4. Entity & Relationship Modeling

### Knowledge Graph Model

The domain model captures three entity types and three relationship types, chosen to represent the core information structure of financial disclosures:

```
                    ┌─────────────────┐
                    │     Company     │
                    │  (Organization) │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
     ┌──────────┐     ┌──────────┐     ┌──────────┐
     │   OWNS   │     │ HAS_RISK │     │ REPORTS_ │
     │          │     │          │     │  AMOUNT  │
     └────┬─────┘     └────┬─────┘     └────┬─────┘
          │                │                │
          ▼                ▼                ▼
   ┌──────────┐     ┌──────────┐     ┌──────────┐
   │ Company  │     │RiskFactor│     │  Dollar  │
   │          │     │          │     │  Amount  │
   └──────────┘     └──────────┘     └──────────┘
```

### Node Types

| Type | Description | Examples |
|------|-------------|----------|
| **Company** | Corporate entities, subsidiaries, or business units | "Parent Corporation," "Subsidiary Holdings," "Retail Division" |
| **RiskFactor** | Disclosed risks, uncertainties, or adverse conditions | "Currency fluctuation risk," "Regulatory compliance risk" |
| **DollarAmount** | Monetary values with implicit or explicit currency | "$1.2 billion," "€45 million," "USD 500 million" |

### Relationship Types

| Relationship | Semantics | Example |
|--------------|-----------|---------|
| **OWNS** | Parent-subsidiary or equity ownership | Parent Corp → OWNS → Subsidiary Holdings |
| **HAS_RISK** | Entity exposed to a risk factor | Tech Division → HAS_RISK → Regulatory compliance risk |
| **REPORTS_AMOUNT** | Entity associated with a monetary figure | Parent Corp → REPORTS_AMOUNT → $1.2 billion |

### Rationale for Schema Simplicity

The schema intentionally uses a minimal set of types:

1. **Reduced LLM Ambiguity** — Fewer categories mean clearer classification boundaries
2. **Higher Extraction Accuracy** — Broad categories reduce edge-case misclassification
3. **Extensibility** — New types can be added without breaking existing extractions
4. **Validation Simplicity** — Strict enums enable compile-time type checking

A more granular schema (e.g., distinguishing "Revenue" from "Expense" amounts) would increase extraction errors without proportional analytical benefit.

---

## 5. JSON Schema Design

### Why Pydantic v2

Pydantic v2 was selected for schema definition based on:

| Criterion | Pydantic v2 Advantage |
|-----------|----------------------|
| **Type Safety** | Native Python type hints with runtime validation |
| **JSON Schema Generation** | `model_json_schema()` produces OpenAPI-compatible schema |
| **Strict Mode** | `extra="forbid"` rejects unexpected fields |
| **Serialization** | `model_dump_json()` for consistent output |
| **Ecosystem** | Standard in FastAPI and modern Python tooling |

### Schema Definition

```python
class Node(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    type: Literal["Company", "RiskFactor", "DollarAmount"]
    name: str

class Relationship(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str
    target: str
    relation: Literal["OWNS", "HAS_RISK", "REPORTS_AMOUNT"]
    confidence: float | None = None  # Optional: 0.0-1.0

class KnowledgeGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str
    nodes: list[Node]
    relationships: list[Relationship]
```

### Strictness as Contract

The `extra="forbid"` configuration serves as a contract between the LLM and the system:

1. **No Schema Drift** — LLM cannot introduce unexpected fields
2. **Fail-Fast Validation** — Invalid outputs are rejected immediately
3. **Deterministic Parsing** — Same schema applies to extraction and downstream consumers

This strictness is intentional: we prefer a failed extraction over a partially correct one with unvalidated fields.

### Optional Confidence Scores

The `Relationship` model supports an optional `confidence` field (0.0–1.0) for downstream filtering, analytics, or human review. This field is:

- **Optional** — LLMs may omit it without validation failure
- **Model-dependent** — Not all providers consistently produce confidence scores
- **Future-ready** — Enables confidence-based filtering in future iterations

The `extra="forbid"` constraint still applies; only the explicitly defined `confidence` field is permitted.

---

## 6. Multi-Provider LLM Extraction Strategy

The system supports multiple LLM providers through a factory pattern, enabling environment-driven provider selection without code changes.

### Provider Overview

| Provider | Model | Use Case |
|----------|-------|----------|
| **OpenAI** | GPT-4o | Production-grade cloud extraction with strong instruction following |
| **Google Gemini** | gemini-2.0-flash | Cloud extraction with large context windows (1M+ tokens) |
| **Ollama** | Llama 3, Mistral, etc. | Local inference, offline capable, no API costs |

### OpenAI GPT-4o

GPT-4o was selected as the default cloud provider based on:

| Factor | Consideration |
|--------|---------------|
| **Instruction Following** | Superior adherence to JSON-only output constraints |
| **Context Window** | 128K tokens accommodates large financial documents |
| **Reasoning Quality** | Strong performance on entity disambiguation |
| **Availability** | Production-grade API with SLA |

### Google Gemini

Gemini offers an alternative cloud option with:

| Factor | Consideration |
|--------|---------------|
| **Context Window** | Up to 1M tokens for very large documents |
| **Model Flexibility** | Multiple models (gemini-2.0-flash, gemini-1.5-flash) configurable via environment |
| **JSON Mode** | Native `response_mime_type="application/json"` enforcement |

### Ollama (Local Inference)

Ollama enables fully local, offline extraction:

| Factor | Consideration |
|--------|---------------|
| **Privacy** | Documents never leave the local machine |
| **Cost** | No API fees; unlimited extractions |
| **Latency** | Network-independent; suitable for air-gapped environments |

### Shared Prompt Constraints

All providers share identical prompt constraints and schema validation:

- Anti-hallucination instructions
- JSON-only output enforcement
- Temperature = 0 for determinism
- Pydantic schema injection
- Strict `extra="forbid"` validation

### Prompt Design Principles

The system prompt follows four principles:

#### 1. Anti-Hallucination

```
DO NOT hallucinate or infer information not explicitly stated in the text.
DO NOT add entities or relationships that are not directly mentioned.
ONLY extract information that is explicitly present in the provided text.
```

This explicit prohibition reduces fabrication. The LLM is instructed to err on the side of omission rather than invention.

#### 2. JSON-Only Output

```
Output ONLY valid JSON matching the schema below—no explanations, no markdown.
Respond with ONLY the JSON object. No additional text.
```

Markdown code fences, explanatory text, and commentary are explicitly forbidden. This eliminates the need for output parsing or cleanup.

#### 3. Schema Injection

The complete JSON schema is embedded in the system prompt via `KnowledgeGraph.model_json_schema()`. This provides the LLM with:

- Exact field names and types
- Allowed enum values
- Required vs. optional fields (all required)

The LLM receives the schema as a structured reference, not as natural language description.

#### 4. Deterministic Behavior

```python
temperature=0
```

Temperature is set to zero to maximize output consistency. While not perfectly deterministic (API version changes may introduce variation), this setting minimizes extraction variance across runs.

### No Post-Processing

The system performs no regex, string manipulation, or heuristic cleanup on LLM output:

- JSON is parsed directly via `json.loads()`
- Pydantic validates the parsed dict
- Any parsing or validation failure propagates as an exception

This design choice ensures that extraction quality is entirely attributable to the LLM and prompt—there are no hidden correction layers that mask prompt deficiencies.

---

## 7. Large Document Chunking

The system supports automatic chunking for documents that exceed LLM context windows.

### Architecture

When chunking is enabled and text exceeds the configured chunk size:

```
┌─────────────┐
│  Raw Text   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                         chunker.py                               │
│  • estimate_tokens(text) — chars/4 approximation                │
│  • split_text(text, chunk_size, overlap)                        │
│  • Paragraph-aware splitting with sentence fallback             │
└──────────────────────────────┬──────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        ┌──────────┐     ┌──────────┐     ┌──────────┐
        │ Chunk 1  │     │ Chunk 2  │     │ Chunk N  │
        └────┬─────┘     └────┬─────┘     └────┬─────┘
             │                │                │
             ▼                ▼                ▼
        ┌──────────┐     ┌──────────┐     ┌──────────┐
        │ Extract  │     │ Extract  │     │ Extract  │
        │ Graph 1  │     │ Graph 2  │     │ Graph N  │
        └────┬─────┘     └────┬─────┘     └────┬─────┘
             │                │                │
             └────────────────┼────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       graph_merger.py                            │
│  • merge_graphs(graphs) — concatenate nodes/relationships       │
│  • Preserve schema_version from first graph                     │
│  • No deduplication or entity resolution                        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
                      ┌───────────────────┐
                      │  KnowledgeGraph   │
                      │   (Unified)       │
                      └───────────────────┘
```

### Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `CHUNK_ENABLED` | Enable/disable chunking | `true` |
| `CHUNK_SIZE_TOKENS` | Target tokens per chunk | `4000` |
| `CHUNK_OVERLAP_TOKENS` | Overlap between chunks | `200` |

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Token estimation via chars/4** | Avoids external tokenizer dependency; sufficient for chunking purposes |
| **Paragraph-aware splitting** | Preserves semantic coherence within chunks |
| **Configurable overlap** | Preserves context for entities spanning chunk boundaries |
| **Simple concatenation merge** | No cross-chunk inference per architectural constraints |
| **Fail-fast on any chunk error** | Maintains reliability guarantees |

### Constraints

The chunking implementation adheres to strict constraints:

1. **No schema changes** — KnowledgeGraph model unchanged
2. **No cross-chunk inference** — Chunks processed independently
3. **No entity resolution** — Duplicate entities may exist in merged graph
4. **Backward compatible** — `CHUNK_ENABLED=false` preserves original behavior

---

## 8. Validation & Reliability

### Two-Layer Validation

The system implements validation at two distinct levels:

#### Layer 1: Schema Validation (Pydantic)

Enforces structural correctness:
- All required fields present
- Field types match (str, Literal enum values)
- No extra fields (`extra="forbid"`)

This validation occurs during `KnowledgeGraph.model_validate()`.

#### Layer 2: Graph Integrity Validation

Enforces semantic correctness:
- At least one node exists
- All node IDs are unique
- Every relationship `source` and `target` references an existing node ID

This validation occurs in `validator.py` after Pydantic parsing succeeds.

### Fail-Fast Philosophy

The system surfaces errors immediately:

| Condition | Behavior |
|-----------|----------|
| Missing `OPENAI_API_KEY` | `ValueError` raised before API call |
| LLM returns non-JSON | `json.JSONDecodeError` raised |
| JSON doesn't match schema | Pydantic `ValidationError` raised |
| Duplicate node IDs | `ValueError` with specific IDs listed |
| Invalid relationship reference | `ValueError` with specific references listed |

Errors are never suppressed or logged silently. The pipeline exits with non-zero status on any failure.

### Error Message Quality

Validation errors include specific details:

```
Duplicate node IDs found: ['company_1']
Invalid node references in relationships: ["source 'nonexistent' in relationship ..."]
```

This enables rapid debugging without inspecting intermediate state.

---

## 9. Visualization Strategy

The system produces two visualization outputs: NetworkX PNG (optional) and Mermaid diagrams (always available).

### Technology Selection

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Graph Library** | NetworkX | Standard Python graph library with rich algorithms |
| **PNG Rendering** | matplotlib | Ubiquitous, no external dependencies beyond Python |
| **Mermaid Diagrams** | Native text output | Lightweight, CI-friendly, GitHub-renderable |

### NetworkX PNG Visualization

| Element | Encoding |
|---------|----------|
| **Company nodes** | Blue circles |
| **RiskFactor nodes** | Red circles |
| **DollarAmount nodes** | Green circles |
| **Relationships** | Gray directed edges with labels |
| **Layout** | Spring layout with fixed seed (42) for reproducibility |

**Note:** NetworkX visualization is optional and may be skipped on Python 3.14 due to upstream compatibility issues. The pipeline continues successfully without PNG output.

### Mermaid Diagram Visualization

Mermaid diagrams are always generated as a first-class output:

| Element | Mermaid Encoding |
|---------|------------------|
| **Company nodes** | Rectangle `["label"]` |
| **RiskFactor nodes** | Rounded `("label")` |
| **DollarAmount nodes** | Parallelogram `[/"label"/]` |
| **Relationships** | Labeled arrows `-->|RELATION|` |

Benefits of Mermaid:
- **GitHub-Renderable** — `.mmd` files render natively in GitHub file browser
- **CI-Friendly** — Text-based output is easy to diff and version control
- **Lightweight** — No image rendering dependencies required
- **Portable** — Can be embedded in Markdown, issues, and documentation

### Purpose of Visualization

Both visualization formats serve:

1. **Explainability** — Humans can verify extraction correctness visually
2. **Proof of Work** — Demonstrates pipeline completed successfully
3. **Communication** — Shareable artifacts for non-technical stakeholders

The visualizations are not intended for interactive exploration; they are static verification artifacts.

---

## 10. Testing Strategy

### Coverage Overview

| Module | Test File | Focus Areas |
|--------|-----------|-------------|
| `validator.py` | `test_validator.py` | Valid graphs pass; duplicates/invalid refs fail |
| `extractor.py` | `test_extractor.py` | Mocked OpenAI responses; JSON parsing; error cases |
| `visualizer.py` | `test_visualizer.py` | File creation; PNG validity; directory handling |

### Mocking External Calls

OpenAI API calls are fully mocked in tests:

```python
@patch("src.extractor.OpenAI")
def test_valid_json_returns_knowledge_graph(self, mock_openai_class):
    mock_response = MagicMock()
    mock_response.output_text = json.dumps(valid_response)
    # ...
```

This ensures:
- Tests run without API keys
- No cost incurred during CI/CD
- Deterministic test outcomes

### Failure Scenario Coverage

Tests explicitly verify error handling:

| Scenario | Expected Behavior |
|----------|-------------------|
| Duplicate node IDs | `ValueError` with ID list |
| Invalid relationship source | `ValueError` with reference details |
| Missing API key | `ValueError` raised before API call |
| Malformed JSON response | `json.JSONDecodeError` |
| Empty LLM response | `ValueError` |

### File System Isolation

Visualization tests use pytest's `tmp_path` fixture:

```python
def test_graph_image_is_created(self, tmp_path: Path):
    output_path = tmp_path / "output" / "graph.png"
    render_graph(graph, output_path)
    assert output_path.exists()
```

This ensures tests do not pollute the working directory and are isolated from each other.

---

## 11. Error Handling & Edge Cases

### Error Categories

| Category | Example | Handling |
|----------|---------|----------|
| **Configuration** | Missing `OPENAI_API_KEY` | `ValueError` before any processing |
| **Network** | OpenAI API timeout | `openai.APIError` propagates |
| **LLM Output** | Non-JSON response | `json.JSONDecodeError` |
| **Schema Mismatch** | Invalid node type | Pydantic `ValidationError` |
| **Graph Integrity** | Orphan relationship | `ValueError` from validator |
| **File System** | Unwritable output path | `OSError` from visualizer |

### Design Principle: Surface Errors Early

The system follows the principle of "fail fast, fail loud":

1. **No Silent Defaults** — Missing configuration raises immediately
2. **No Partial Success** — Either the full pipeline succeeds or it fails
3. **Specific Error Messages** — Errors identify the exact failure point
4. **Non-Zero Exit** — `main.py` returns exit code 1 on any error

This design ensures that errors are detected during development and testing, not silently ignored in production.

---

## 12. Limitations & Assumptions

### Assumptions

| Assumption | Implication |
|------------|-------------|
| **Explicit Relationships** | The system only extracts relationships stated in text; implicit relationships are not inferred |
| **English Language** | Prompts and extraction logic assume English-language documents |
| **Single Document** | Pipeline processes one document per run; batch processing is not implemented |
| **OpenAI Availability** | Production use requires reliable internet and API access |
| **Schema Sufficiency** | The three node types and three relationship types capture the relevant domain |

### Limitations

| Limitation | Impact | Mitigation Path |
|------------|--------|-----------------|
| **LLM Variability** | Minor output differences across API versions | Pin API version; monitor for drift |
| **Token Limits** | Documents exceeding context limits are automatically chunked | Chunking enabled by default; configurable via environment variables |
| **No Entity Resolution** | Similar entity names with slight variations may be separate nodes | Post-processing entity resolution layer |
| **Cost per Cloud Extraction** | Each cloud run incurs API cost | Use Ollama for development; caching layer for repeated extractions |
| **No Incremental Updates** | Full re-extraction required for document changes | Delta extraction for document versions |

### Deliberate Non-Goals

The following are explicitly out of scope:

- Real-time streaming extraction
- Multi-document graph merging
- User-facing API or web interface
- Fine-tuned models

These are future enhancement candidates, not current requirements.

---

## 13. Future Enhancements

The following enhancements are candidates for subsequent iterations:

### Near-Term

| Enhancement | Effort | Value |
|-------------|--------|-------|
| **Mermaid → PNG Export** | Low | Generate PNG from Mermaid diagrams for non-GitHub environments |
| **Provider Benchmarking** | Low | Compare extraction quality across OpenAI, Gemini, and Ollama |
| **Confidence-Based Filtering** | Low | Filter relationships by confidence threshold in output |
| **Smart Chunk Deduplication** | Medium | Deduplicate entities extracted across multiple chunks |
| **Caching Layer** | Low | Reduce API costs for repeated extractions |

### Medium-Term

| Enhancement | Effort | Value |
|-------------|--------|-------|
| **Entity Resolution** | Medium | Merge duplicate entities across naming variations |
| **Incremental Extraction** | High | Process document changes without full re-extraction |
| **Batch Document Processing** | Medium | Process multiple documents in a single run |

### Long-Term

| Enhancement | Effort | Value |
|-------------|--------|-------|
| **Graph Database Persistence** | High | Enable querying and relationship traversal at scale |
| **REST API Layer** | Medium | Integrate with external systems |
| **Multi-Document Knowledge Graph** | High | Build cumulative graph across document corpus |

---

## Appendix A: Module Dependency Graph

```
main.py
    │
    ├── src/input_loader.py
    │
    ├── src/extractor/                    # LLM extractor package
    │       ├── __init__.py               # Package entry point (with chunking orchestration)
    │       │       ├── src/chunker.py    # Text chunking
    │       │       ├── src/graph_merger.py  # Graph merging
    │       │       └── src/config.py     # Configuration management
    │       ├── base.py                   # Abstract LLMExtractor interface
    │       ├── factory.py                # Environment-driven provider selection
    │       │       └── src/config.py
    │       ├── openai_llm.py             # OpenAI GPT-4o implementation
    │       ├── gemini_llm.py             # Google Gemini implementation
    │       └── ollama_llm.py             # Ollama local implementation
    │               └── src/schema.py
    │
    ├── src/chunker.py                    # Text chunking utilities
    │
    ├── src/graph_merger.py               # KnowledgeGraph merging
    │       └── src/schema.py
    │
    ├── src/validator.py
    │       └── src/schema.py
    │
    ├── src/visualizer.py                 # NetworkX PNG (optional)
    │       └── src/schema.py
    │
    └── src/visualizer_mermaid.py         # Mermaid diagrams (always available)
            └── src/schema.py
```

All modules depend on `schema.py` as the single source of truth for data structures. Provider selection is driven by `config.py` which reads environment variables. Chunking is orchestrated in `extractor/__init__.py` and is transparent to callers.

---

## Appendix B: External Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| `pydantic` | 2.x | Schema definition and validation |
| `openai` | 1.x | OpenAI GPT-4o API client |
| `google-genai` | 1.x | Google Gemini API client |
| `httpx` | 0.25.x | HTTP client (used by Ollama extractor for local API calls) |
| `python-dotenv` | 1.x | Environment variable loading from .env files |
| `networkx` | 3.x | Graph data structure and algorithms |
| `matplotlib` | 3.x | Graph rendering to PNG |
| `pytest` | 8.x | Testing framework |

### Runtime Dependencies

| Dependency | Required | Purpose |
|------------|----------|---------|
| **Ollama** | When `LLM_PROVIDER=ollama` | Local LLM inference server |

---

## Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | December 2024 | Architecture Team | Initial release |
| 1.1 | December 2024 | Architecture Team | Multi-provider LLM support (OpenAI, Gemini, Ollama); Mermaid visualization; optional confidence scores |
| 1.2 | December 2024 | Architecture Team | Removed company-specific references; made documentation generic |
| 1.3 | December 2024 | Architecture Team | Added automatic document chunking for large documents; simple merge strategy |
| 1.3 | December 2024 | Architecture Team | Added large document chunking support with configurable chunk size and overlap |
