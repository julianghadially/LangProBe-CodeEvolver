# Chain-of-Thought Reasoning for Hover Claim Verification

## Overview

This implementation adds a comprehensive chain-of-thought reasoning module to `langProBe/hover/hover_program.py` that processes retrieved documents to verify claims. The system performs structured, multi-step reasoning to determine whether retrieved evidence SUPPORTS or REFUTES a given claim.

## Architecture

### Components Added

1. **ChainOfThoughtVerifier** - Main reasoning module
2. **HoverProgram** - Complete pipeline (retrieval + verification)
3. **Four reasoning signatures** that guide the chain-of-thought process

## Chain-of-Thought Reasoning Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Extract Key Facts                                       │
│ ────────────────────────                                        │
│ Input:  Claim + Retrieved Passages                             │
│ Output: List of key facts relevant to the claim                │
│ Purpose: Identify specific factual statements from documents   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Gap Analysis (Identify Missing Information)            │
│ ────────────────────────────────────────────                   │
│ Input:  Claim + Key Facts                                      │
│ Output: Missing information + Coverage assessment              │
│ Purpose: Determine what evidence is still lacking              │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Chain Facts Together                                   │
│ ─────────────────────────                                      │
│ Input:  Claim + Key Facts + Missing Info                       │
│ Output: Reasoning chains connecting facts                      │
│ Purpose: Build logical connections toward verification         │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Final Verification Decision                            │
│ ───────────────────────────────────────────                    │
│ Input:  All previous outputs                                   │
│ Output: Decision (SUPPORTS/REFUTES) + Confidence + Justification│
│ Purpose: Make final evidence-based verification decision       │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Details

### 1. ExtractKeyFactsSignature

Extracts specific factual statements from retrieved passages that are relevant to the claim.

**Input:**
- `claim`: The claim to verify
- `passages`: Retrieved documents/passages

**Output:**
- `key_facts`: List of relevant facts extracted from passages

### 2. IdentifyMissingInfoSignature

Performs gap analysis to identify information missing from the evidence.

**Input:**
- `claim`: The claim to verify
- `key_facts`: Extracted facts

**Output:**
- `missing_info`: List of missing pieces of information
- `coverage_assessment`: Assessment of evidence completeness

### 3. ChainFactsSignature

Chains facts together to form logical connections leading to verification.

**Input:**
- `claim`: The claim to verify
- `key_facts`: Extracted facts
- `missing_info`: Identified gaps

**Output:**
- `reasoning_chains`: List of logical reasoning chains

### 4. FinalVerificationSignature

Makes the final verification decision based on all reasoning steps.

**Input:**
- `claim`: The claim to verify
- `key_facts`: Extracted facts
- `missing_info`: Identified gaps
- `reasoning_chains`: Logical connections

**Output:**
- `decision`: "SUPPORTS" or "REFUTES"
- `confidence_score`: Float between 0.0 and 1.0
- `justification`: Explanation of the decision

## Usage Examples

### Standalone Verifier

```python
from langProBe.hover.hover_program import ChainOfThoughtVerifier
import dspy

# Configure DSPy
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key="your-key"))

# Create verifier
verifier = ChainOfThoughtVerifier()

# Verify a claim with passages
claim = "Antonis Fotsis is a player for Ilysiakos B.C."
passages = [
    "Ilysiakos B.C. | A Greek professional basketball club.",
    "Antonis Fotsis | A Greek basketball player for Ilysiakos B.C."
]

result = verifier(claim=claim, passages=passages)

# Access results
print(f"Decision: {result.decision}")                    # "SUPPORTS" or "REFUTES"
print(f"Confidence: {result.confidence_score}")          # 0.0 to 1.0
print(f"Justification: {result.justification}")          # Detailed explanation

# Access intermediate reasoning
print(f"Key Facts: {result.key_facts}")                  # Extracted facts
print(f"Missing Info: {result.missing_info}")            # Gap analysis
print(f"Reasoning Chains: {result.reasoning_chains}")    # Logical connections
```

### Complete Pipeline (Retrieval + Verification)

```python
from langProBe.hover.hover_program import HoverProgram
import dspy

# Configure DSPy with LM and retriever
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini", api_key="your-key"),
    rm=dspy.ColBERTv2(url="your-colbert-url")
)

# Create complete program
program = HoverProgram()

# Process a claim (retrieves documents then verifies)
claim = "Antonis Fotsis is a player for Ilysiakos B.C."
result = program(claim=claim)

# Access results
print(f"Retrieved {len(result.retrieved_docs)} documents")
print(f"Decision: {result.decision}")
print(f"Confidence: {result.confidence_score}")
print(f"Justification: {result.justification}")
```

### Using with HoverMultiHopPipeline

```python
from langProBe.hover.hover_pipeline import HoverMultiHopPipeline
from langProBe.hover.hover_program import HoverProgram

# Configure pipeline
pipeline = HoverMultiHopPipeline()
pipeline.setup_lm("openai/gpt-4o-mini", api_key="your-key")

# Replace the retrieval-only program with verification program
pipeline.program = HoverProgram()

# Run verification
result = pipeline(claim="Your claim here")
```

## Key Features

✅ **Explicit Reasoning**: Each step produces interpretable intermediate outputs
✅ **Gap Analysis**: Identifies what information is missing before making decisions
✅ **Logical Chaining**: Connects facts together before classification
✅ **Confidence Scoring**: Provides confidence based on evidence strength
✅ **Transparent Justification**: Explains decisions with references to facts
✅ **Modular Design**: Can use verifier standalone or in complete pipeline
✅ **Backward Compatible**: HoverMultiHop still works for retrieval-only tasks

## Integration with Existing Code

The implementation maintains backward compatibility:

- **HoverMultiHop**: Original retrieval-only module (unchanged functionality)
- **HoverProgram**: New module that combines retrieval + verification
- **ChainOfThoughtVerifier**: Standalone verifier that can be used independently

## Output Structure

### ChainOfThoughtVerifier Output

```python
dspy.Prediction(
    decision="SUPPORTS" | "REFUTES",
    confidence_score=0.85,  # 0.0 to 1.0
    justification="Detailed explanation...",
    key_facts=["Fact 1", "Fact 2", ...],
    missing_info=["Missing piece 1", ...],
    coverage_assessment="Assessment text...",
    reasoning_chains=["Chain 1", "Chain 2", ...]
)
```

### HoverProgram Output

```python
dspy.Prediction(
    retrieved_docs=["Doc 1", "Doc 2", ...],  # 21 docs max
    decision="SUPPORTS" | "REFUTES",
    confidence_score=0.85,
    justification="Detailed explanation...",
    key_facts=["Fact 1", "Fact 2", ...],
    missing_info=["Missing piece 1", ...],
    coverage_assessment="Assessment text...",
    reasoning_chains=["Chain 1", "Chain 2", ...]
)
```

## Example Reasoning Flow

### Input
**Claim:** "Antonis Fotsis is a player for the club whose name has the starting letter from an alphabet derived from the Phoenician alphabet."

**Retrieved Passages:**
1. "Ilysiakos B.C. | Ilysiakos B.C. is a Greek professional basketball club."
2. "Greek alphabet | The Greek alphabet is derived from the Phoenician alphabet."
3. "Antonis Fotsis | Antonis Fotsis is a Greek basketball player for Ilysiakos B.C."

### Step 1: Extract Key Facts
- "Antonis Fotsis is a basketball player for Ilysiakos B.C."
- "Ilysiakos B.C. is a Greek professional basketball club"
- "The Greek alphabet is derived from the Phoenician alphabet"
- "Ilysiakos starts with the letter 'I' from the Greek alphabet"

### Step 2: Gap Analysis
**Missing Info:**
- Direct confirmation that 'I' is a letter in the Greek alphabet (implicit)

**Coverage Assessment:**
- All key elements covered: player identity, club association, alphabet origin

### Step 3: Chain Facts
1. "Antonis Fotsis plays for Ilysiakos B.C. (Fact 1 + Fact 3)"
2. "Ilysiakos B.C. is a Greek club, meaning its name uses Greek letters (Fact 2)"
3. "Greek alphabet is derived from Phoenician alphabet (Fact 3)"
4. "Therefore, Ilysiakos starts with 'I', a Greek letter derived from Phoenician"

### Step 4: Final Decision
**Decision:** SUPPORTS
**Confidence:** 0.92
**Justification:** "The evidence confirms all elements of the claim. Antonis Fotsis plays for Ilysiakos B.C., which is a Greek club whose name uses the Greek alphabet (starting with 'I'). The Greek alphabet is confirmed to be derived from the Phoenician alphabet. All logical connections are well-supported."

## Testing

A test script is provided at `/workspace/test_hover_verification.py`:

```bash
python test_hover_verification.py
```

This demonstrates:
- Architecture visualization
- Standalone verifier usage
- Complete pipeline usage
- Example claim and reasoning flow

## File Changes

### Modified Files
- `langProBe/hover/hover_program.py`: Added chain-of-thought verification

### New Files
- `/workspace/test_hover_verification.py`: Test and demonstration script
- `/workspace/HOVER_COT_IMPLEMENTATION.md`: This documentation

### Lines of Code
- **4 new signatures**: ~120 lines
- **ChainOfThoughtVerifier class**: ~75 lines
- **HoverProgram class**: ~55 lines
- **Test script**: ~200 lines
- **Total**: ~450 lines of new code

## Dependencies

- `dspy`: Core framework
- `typing.Literal`: For type hints (Python 3.8+)
- All existing dependencies from `langProBe`

## Future Enhancements

Potential improvements:
1. Add multi-label support (beyond SUPPORTS/REFUTES)
2. Implement fact-checking against external knowledge bases
3. Add adversarial robustness testing
4. Integrate with retrieval optimization (MIPROv2, BootstrapFewShot)
5. Add visualization tools for reasoning chains
6. Implement confidence calibration

## Notes

- The verifier uses `dspy.ChainOfThought` for all reasoning steps, ensuring transparent reasoning
- Each step explicitly builds on previous outputs, creating a clear reasoning chain
- The gap analysis step ensures the model considers missing information before making decisions
- Confidence scoring considers evidence completeness, consistency, and reasoning strength
- The implementation maintains backward compatibility with existing Hover benchmarks

## Authors

Implementation by Claude (Anthropic) for the langProBe framework.
