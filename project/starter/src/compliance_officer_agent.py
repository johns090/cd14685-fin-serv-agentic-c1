# Compliance Officer Agent - ReACT Implementation  
# TODO: Implement Compliance Officer Agent using ReACT prompting

"""
Compliance Officer Agent Module

This agent generates regulatory-compliant SAR narratives using ReACT prompting.
It takes risk analysis results and creates structured documentation for 
FinCEN submission.

YOUR TASKS:
- Study ReACT (Reasoning + Action) prompting methodology
- Design system prompt with Reasoning/Action framework
- Implement narrative generation with word limits
- Validate regulatory compliance requirements
- Create proper audit logging and error handling
"""

import json
import openai
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv
from unittest.mock import Mock

# TODO: Import your foundation components
from .foundation_sar import (
    ComplianceOfficerOutput,
    ExplainabilityLogger, 
    CaseData,
    RiskAnalystOutput
)

# Load environment variables
load_dotenv()

class ComplianceOfficerAgent:
    """
    Compliance Officer agent using ReACT prompting framework.
    
    TODO: Implement agent that:
    - Uses Reasoning + Action structured prompting
    - Generates regulatory-compliant SAR narratives
    - Enforces word limits and terminology
    - Includes regulatory citations
    - Validates narrative completeness
    """
    
    def __init__(self, openai_client, explainability_logger, model="gpt-3.5-turbo"): #"gpt-4", "gpt-3.5-turbo"
        """Initialize the Compliance Officer Agent
        
        Args:
            openai_client: OpenAI client instance
            explainability_logger: Logger for audit trails
            model: OpenAI model to use
        """
        # TODO: Initialize agent components
        self.openai_client = openai_client
        self.explainability_logger = explainability_logger
        self.model = model
        
        # TODO: Design ReACT system prompt

        # self.system_prompt = """TODO: Create your ReACT system prompt here   
        
        # Key elements to include:
        # - Agent persona as senior compliance officer
        # - ReACT framework: Reasoning Phase + Action Phase
        # - Narrative constraints (≤120 words)
        # - Regulatory terminology requirements
        # - JSON output format specification
        # - BSA/AML compliance focus

        
        # """

        self.system_prompt = """As a senior compliance officer, your role is to ensure adherence to BSA/AML regulations while utilizing the ReACT framework. 
        
        You are a fraud investigator using the ReACT framework. Follow this pattern:
            OBSERVATION: [State what you observe from the data]
            THOUGHT: [Reason about what the observation means]  
            ACTION: [Describe what investigation step you would take next]

            Continue this cycle 3-4 times, then provide a final assessment.

            Use this systematic approach to analyze the fraud case step by step.
        
        **REASONING Phase:**
        1. Review the risk analyst's findings
        2. Assess regulatory narrative requirements
        3. Identify key compliance elements
        4. Consider narrative structure
        
        **ACTION Phase:**
        1. Draft concise narrative (≤120 words)
        2. Include specific details and amounts
        3. Reference suspicious activity pattern
        4. Ensure regulatory language

        Ensure that the narrative is complete and meets all regulatory requirements before submission. 
        ## Response Format (JSON)
        Your response MUST be valid JSON with this exact structure:
        {
            narrative: str = Field(..., max_length=1000, description="Regulatory narrative text contains max 120 words"),
            narrative_reasoning: str = Field(..., max_length=500, description="Reasoning for narrative construction")
            regulatory_citations: List[str] = Field(..., description="List of relevant regulations")
            completeness_check: bool = Field(..., description="Whether narrative meets all requirements")
        }

        **Key regulatory requirements for SAR "narrative" attribute above:**
        "word_limit": 120,
        "required_elements": [
            "Customer identification",
            "Suspicious activity description", 
            "Transaction amounts and dates",
            "Why activity is suspicious"
        ],
        "terminology": [
            "Suspicious activity",
            "Regulatory threshold",
            "Financial institution",
            "Money laundering",
            "Bank Secrecy Act"
        ],
        "citations": [
            "31 CFR 1020.320 (BSA)",
            "12 CFR 21.11 (SAR Filing)",
            "FinCEN SAR Instructions"
        ]

        """

    def generate_compliance_narrative(self, case_data, risk_analysis) -> 'ComplianceOfficerOutput':
        # TODO: Implement narrative generation that:
        # - Creates ReACT-structured user prompt
        # - Includes risk analysis findings
        # - Makes OpenAI API call with constraints
        # - Validates narrative word count
        # - Parses and validates JSON response
        # - Logs operations for audit
        
        """
        Generate regulatory-compliant SAR narrative using ReACT framework.
        
        Args:
            case_data: Data related to the case
            risk_analysis: Analysis of risks associated with the case
        
        Returns:
            ComplianceOfficerOutput: Structured output containing narrative and citations
        """
        # Create ReACT-structured user prompt

        user_prompt = f"Case Data: {case_data}\nRisk Analysis: {risk_analysis}"
        
        # Record start time and make OpenAI API call with constraints
        start_time = datetime.now()
        response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower temperature for consistency in financial analysis
                max_tokens=1000,  
                response_format={"type": "json_object"}  # Enforce JSON response
            )
        print("Compliance Response:", response)  # Debugging output

        # Parse and validate JSON response
        respones_content = response.choices[0].message.content
        print("response content: ", respones_content)  # Debugging output
        json_response = self._extract_json_from_response(respones_content)
        
        print("extract json from response content: ", json_response)  # Debugging output
        narrative = json_response.get('narrative', '')
        narrative_word_count = len(narrative.split())
        if narrative_word_count > 120:
            raise ValueError(f"Narrative exceeds word limit of 120 words. Current word count: {narrative_word_count}")
        
   
        # Log operations for audit
        execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        self.explainability_logger.log_agent_action(
            agent_type="ComplianceOfficer",
            action="generate_compliance_narrative",
            case_id=case_data.get("customer_id", "UNKNOWN") if isinstance(case_data, dict) else getattr(case_data, "customer_id", "UNKNOWN"),
            input_data={"risk_analysis_summary": str(risk_analysis)[:200]},
            output_data={"narrative_length": len(narrative.split())},
            reasoning="Generated SAR narrative based on risk analysis",
            execution_time_ms=execution_time_ms,
            success=True
        )
        
        return ComplianceOfficerOutput(
            narrative=narrative,
            narrative_reasoning="Generated based on case data and risk analysis.",
            regulatory_citations=json_response.get('citations', []),
            completeness_check=True
        )

    def _extract_json_from_response(self, response_content: str) -> str:
        # """Extract JSON content from LLM response
        
        # TODO: Implement JSON extraction that handles:
        # - JSON in code blocks (```json)
        # - JSON in plain text
        # - Malformed responses
        # - Empty responses
        # """
                
        """
        Extract JSON content from LLM response
        
        Args:
            response_content: The content returned from the LLM response
        
        Returns:
            str: Extracted JSON content
        """
        import json
        
        # Attempt to extract JSON from code blocks
        try:
            if '```json' in response_content:
                json_part = response_content.split('```json')[1].split('```')[0]
            else:
                json_part = response_content
            
            # Load JSON to validate and parse
            return json.loads(json_part)
        except (json.JSONDecodeError, IndexError):
            # Handle malformed or empty responses
            return {"error": "Invalid JSON response"}
        
        return {}

    def _format_risk_analysis_for_prompt(self, risk_analysis) -> str:
        """Format risk analysis results for compliance prompt
        - Classification and confidence
        - Key suspicious indicators
        - Risk level assessment
        - Analyst reasoning
        """
        if not risk_analysis:
            return "No risk analysis provided."
        
        # Assume risk_analysis is a dict or RiskAnalystOutput with relevant fields
        try:
            classification = getattr(risk_analysis, 'classification', None) or risk_analysis.get('classification', 'N/A')
            confidence = getattr(risk_analysis, 'confidence', None) or risk_analysis.get('confidence', 'N/A')
            indicators = getattr(risk_analysis, 'suspicious_indicators', None) or risk_analysis.get('suspicious_indicators', [])
            risk_level = getattr(risk_analysis, 'risk_level', None) or risk_analysis.get('risk_level', 'N/A')
            reasoning = getattr(risk_analysis, 'analyst_reasoning', None) or risk_analysis.get('analyst_reasoning', 'N/A')
        except Exception:
            return str(risk_analysis)
        
        formatted = (
            f"Classification: {classification} (Confidence: {confidence})\n"
            f"Key Indicators: {', '.join(indicators) if indicators else 'None'}\n"
            f"Risk Level: {risk_level}\n"
            f"Analyst Reasoning: {reasoning}"
        )
        return formatted

    def _validate_narrative_compliance(self, narrative: str) -> Dict[str, Any]:
        """Validate narrative meets regulatory requirements
        - Word count (≤120 words)
        - Required elements present
        - Appropriate terminology
        - Regulatory completeness
        """
        requirements = get_regulatory_requirements()
        result = {
            "word_count_valid": validate_word_count(narrative, requirements["word_limit"]),
            "required_elements_present": all(
                elem.lower() in narrative.lower() for elem in requirements["required_elements"]
            ),
            "terminology_used": any(
                term.lower() in narrative.lower() for term in requirements["terminology"]
            ),
            "citations_present": any(
                citation.split()[0] in narrative for citation in requirements["citations"]
            ),
        }
        result["compliant"] = all(result.values())
        return result

# ===== REACT PROMPTING HELPERS =====

def create_react_framework():
    """Helper function showing ReACT structure
    
    TODO: Study this example and adapt for compliance narratives:
    
    **REASONING Phase:**
    1. Review the risk analyst's findings
    2. Assess regulatory narrative requirements
    3. Identify key compliance elements
    4. Consider narrative structure
    
    **ACTION Phase:**
    1. Draft concise narrative (≤120 words)
    2. Include specific details and amounts
    3. Reference suspicious activity pattern
    4. Ensure regulatory language
    """
    return {
        "reasoning_phase": [
            "Review risk analysis findings",
            "Assess regulatory requirements", 
            "Identify compliance elements",
            "Plan narrative structure"
        ],
        "action_phase": [
            "Draft concise narrative",
            "Include specific details",
            "Reference activity patterns",
            "Use regulatory language"
        ]
    }

def get_regulatory_requirements():
    """Key regulatory requirements for SAR narratives
    
    TODO: Use these requirements in your prompts:
    """
    return {
        "word_limit": 120,
        "required_elements": [
            "Customer identification",
            "Suspicious activity description", 
            "Transaction amounts and dates",
            "Why activity is suspicious"
        ],
        "terminology": [
            "Suspicious activity",
            "Regulatory threshold",
            "Financial institution",
            "Money laundering",
            "Bank Secrecy Act"
        ],
        "citations": [
            "31 CFR 1020.320 (BSA)",
            "12 CFR 21.11 (SAR Filing)",
            "FinCEN SAR Instructions"
        ]
    }

# ===== TESTING UTILITIES =====

def test_narrative_generation():
    """Test the agent with sample risk analysis
    
    TODO: Use this function to test your implementation:
    - Create sample risk analysis results
    - Initialize compliance agent
    - Generate narrative
    - Validate compliance requirements
    """
    print("🧪 Testing Compliance Officer Agent")
    # Sample risk analysis
    sample_risk_analysis = {
        "classification": "High Risk",
        "confidence": 0.92,
        "suspicious_indicators": ["Large cash deposits", "Frequent wire transfers"],
        "risk_level": "High",
        "analyst_reasoning": "Pattern of transactions is consistent with money laundering."
    }
    # Sample case data
    sample_case_data = {
        "customer_id": "C12345",
        "activity_description": "Customer made multiple large cash deposits and frequent international wire transfers.",
        "transaction_amounts": ["$50,000", "$75,000"],
        "transaction_dates": ["2026-01-10", "2026-01-15"],
        "suspicious_reason": "Activity exceeds regulatory threshold and matches known laundering patterns."
    }
    # Initialize mock OpenAI client and logger
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '```json\n{"narrative": "Customer C12345 made multiple large cash deposits and frequent wire transfers, exceeding regulatory thresholds. Activity is consistent with money laundering patterns.", "regulatory_citations": ["31 CFR 1020.320 (BSA)", "FinCEN SAR Instructions"]}\n```'
    mock_client.chat.completions.create.return_value = mock_response
    
    logger = ExplainabilityLogger("compliance_test.jsonl")
    agent = ComplianceOfficerAgent(mock_client, logger)
    risk_prompt = agent._format_risk_analysis_for_prompt(sample_risk_analysis)
    print("Risk Analysis Prompt:\n", risk_prompt)
    output = agent.generate_compliance_narrative(sample_case_data, sample_risk_analysis)
    print("Generated Narrative:\n", output.narrative)
    compliance = agent._validate_narrative_compliance(output.narrative)
    print("Compliance Validation:\n", compliance)

def validate_word_count(text: str, max_words: int = 120) -> bool:
    """Helper to validate word count
    
    TODO: Use this utility in your validation:
    """
    word_count = len(text.split())
    return word_count <= max_words

if __name__ == "__main__":
    print("✅ Compliance Officer Agent Module")
    print("ReACT prompting for regulatory narrative generation")
    print("\n📋 TODO Items:")
    print("• Design ReACT system prompt")
    print("• Implement generate_compliance_narrative method")
    print("• Add narrative validation (word count, terminology)")
    print("• Create regulatory citation system")
    print("• Test with sample risk analysis results")
    print("\n💡 Key Concepts:")
    print("• ReACT: Reasoning + Action structured prompting")
    print("• Regulatory Compliance: BSA/AML requirements")
    print("• Narrative Constraints: Word limits and terminology")
    print("• Audit Logging: Complete decision documentation")
