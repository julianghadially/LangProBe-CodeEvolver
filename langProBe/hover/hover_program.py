import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

class ExtractQueryAspects(dspy.Signature):
    """Identify a distinct queryable aspect or entity from the claim that hasn't been covered yet.
    The aspect should be a specific entity, person, place, event, or concept that can be searched for.
    Ensure the aspect is different from the previous aspects already covered."""

    claim = dspy.InputField(desc="The claim to analyze")
    previous_aspects = dspy.InputField(desc="List of aspects already covered in previous hops")

    aspect = dspy.OutputField(desc="A distinct queryable aspect, entity, or concept from the claim")
    reasoning = dspy.OutputField(desc="Explanation of why this aspect is important and different from previous aspects")

class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim. 
    
    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant. 
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 7
        self.extract_aspect = dspy.ChainOfThought(ExtractQueryAspects)
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def forward(self, claim):
        # Track covered aspects for diversity
        covered_aspects = []

        # HOP 1: Extract and query the first aspect directly from the claim
        aspect1_result = self.extract_aspect(
            claim=claim,
            previous_aspects=covered_aspects
        )
        aspect1 = aspect1_result.aspect
        covered_aspects.append(aspect1)
        hop1_docs = self.retrieve_k(aspect1).passages

        # HOP 2: Extract a second distinct aspect different from hop 1
        aspect2_result = self.extract_aspect(
            claim=claim,
            previous_aspects=covered_aspects
        )
        aspect2 = aspect2_result.aspect
        covered_aspects.append(aspect2)
        hop2_docs = self.retrieve_k(aspect2).passages

        # HOP 3: Extract a third aspect or generate a connecting query
        aspect3_result = self.extract_aspect(
            claim=claim,
            previous_aspects=covered_aspects
        )
        aspect3 = aspect3_result.aspect
        hop3_docs = self.retrieve_k(aspect3).passages

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)
