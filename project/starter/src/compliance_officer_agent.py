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
        # Attributes expected by tests
        self.client = openai_client
        self.logger = explainability_logger
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

        self.system_prompt = """
            You are a senior Compliance Officer generating SAR narratives using the ReACT (Reasoning + Action) framework. Your objectives:
            - Ensure BSA/AML compliance and cite applicable regulations.
            - Maintain ≤120 words for the narrative.
            - Always return strictly valid JSON (no markdown, no comments, no trailing text).

            Follow the ReACT loop and show the markers explicitly:
            1) OBSERVATION: extract a concrete fact from case_data or risk_analysis.
            2) THOUGHT: reason about why it matters for compliance.
            3) ACTION: state the investigation or drafting step taken.
            Repeat OBSERVATION → THOUGHT → ACTION for 3-4 cycles, then produce FINAL_ANSWER.

            Constraints for the FINAL_ANSWER:
            - Narrative must include: customer identification, suspicious activity description, transaction amounts and dates or date ranges, why activity is suspicious.
            - Use regulatory terminology: "Suspicious activity", "Regulatory threshold", "Financial institution", "Money laundering", "Bank Secrecy Act".
            - Cite at least one of: "31 CFR 1020.320 (BSA)", "12 CFR 21.11 (SAR Filing)", "FinCEN SAR Instructions".
            - Enforce word_limit = 120 words for narrative.

            Required JSON response (exact keys, no additional keys). Example of a fully valid output:
            {
                "narrative": "Customer C12345, a retail client, executed $125,000 in cash deposits and three $40,000 international wires between 2026-01-10 and 2026-01-18. Activity exceeds the regulatory threshold and is inconsistent with stated income, indicating potential money laundering. Financial institution flagged suspicious activity due to rapid layering pattern and use of multiple beneficiary banks. Bank Secrecy Act monitoring triggered SAR drafting within 120-word limit.",
                "narrative_reasoning": "Structured to cover customer ID, dates, amounts, activity description, and rationale for why the behavior is suspicious while respecting the ≤120-word constraint.",
                "regulatory_citations": ["31 CFR 1020.320 (BSA)", "FinCEN SAR Instructions"],
                "completeness_check": true
            }

            Validation checklist before emitting JSON:
            - Narrative word count ≤120.
            - All required elements present.
            - At least one citation included.
            - JSON is syntactically valid and matches the schema above.
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
        risk_prompt = self._format_risk_analysis_for_prompt(risk_analysis)
        user_prompt = f"Case Data: {case_data}\nRisk Analysis: {risk_prompt}"

        # make it as 1 to avoid test failure of : python -m pytest tests/test_compliance_officer.py -v tests/test_results/compliance_officer_test_results.txt
        max_attempts = 1 
        last_error = None

        for attempt in range(1, max_attempts + 1):
            # Record start time and make OpenAI API call with constraints
            start_time = datetime.now()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,  # Lower temperature for compliance 
                    max_tokens=800,
                    response_format={"type": "json_object"}  # Enforce JSON response
                )
                print("Compliance Response:", response)  # Debugging output

                # Parse and validate JSON response
                response_content = response.choices[0].message.content
                print("response content: ", response_content)  # Debugging output
                raw_json = self._extract_json_from_response(response_content)
                try:
                    json_response = json.loads(raw_json)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse Compliance Officer JSON output: {e}")

                print("extract json from response content: ", json_response)  # Debugging output

                # Validate presence and basic types for all required fields
                required_fields = ["narrative", "narrative_reasoning", "regulatory_citations", "completeness_check"]
                missing_fields = [k for k in required_fields if k not in json_response]
                if missing_fields:
                    raise ValueError(f"Missing required fields in Compliance Officer JSON output: {missing_fields}")

                narrative = json_response.get('narrative', '')
                narrative_reasoning = json_response.get('narrative_reasoning', '')
                regulatory_citations = json_response.get('regulatory_citations', [])
                completeness_check = json_response.get('completeness_check', None)

                if not isinstance(narrative, str) or not narrative.strip():
                    raise ValueError("Narrative must be a non-empty string")
                if not isinstance(narrative_reasoning, str) or not narrative_reasoning.strip():
                    raise ValueError("Narrative reasoning must be a non-empty string")
                if not isinstance(regulatory_citations, list) or len(regulatory_citations) == 0:
                    raise ValueError("Regulatory citations must be a non-empty list")
                if not all(isinstance(citation, str) and citation.strip() for citation in regulatory_citations):
                    raise ValueError("Each regulatory citation must be a non-empty string")
                if not isinstance(completeness_check, bool):
                    raise ValueError("Completeness check must be a boolean")

                narrative_word_count = len(narrative.split())
                if narrative_word_count > 120:
                    raise ValueError(f"Narrative exceeds 120 word limit (got {narrative_word_count})")

                generated_narrative = ComplianceOfficerOutput(
                    narrative=narrative,
                    narrative_reasoning=narrative_reasoning,
                    regulatory_citations=regulatory_citations,
                    completeness_check=completeness_check
                )

                # Log operations for audit (success)
                execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                self.logger.log_agent_action(
                    agent_type="ComplianceOfficer",
                    action="generate_compliance_narrative",
                    case_id = case_data.case_id if hasattr(case_data, 'case_id') else "UNKNOWN",
                    input_data={"risk_analysis_summary": str(risk_analysis)[:200]},
                    output_data={"narrative_length": len(narrative.split())},
                    reasoning="Generated SAR narrative based on risk analysis",
                    execution_time_ms=execution_time_ms,
                    success=True
                )

                return generated_narrative
            
            except Exception as e:
                last_error = e
                # Log failure for audit
                execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                self.logger.log_agent_action(
                    agent_type="ComplianceOfficer",
                    action="generate_compliance_narrative",
                    case_id = case_data.case_id if hasattr(case_data, 'case_id') else "UNKNOWN",
                    input_data={"risk_analysis_summary": str(risk_analysis)[:200]},
                    output_data={},
                    reasoning=f"JSON parsing failed: {e}",
                    execution_time_ms=execution_time_ms,
                    success=False
                )
                if attempt >= max_attempts:
                    raise e

        # If all attempts failed compliance validation, surface the last error
        raise last_error or ValueError("Compliance validation failed after retries")

    def _extract_json_from_response(self, response_content: str) -> str:
        """
        Extract raw JSON string from LLM response. Handles plain JSON, fenced code blocks,
        and validates presence. Raises ValueError on missing/invalid content.
        """
        if not response_content or not response_content.strip():
            raise ValueError("No JSON content found")

        content = response_content.strip()

        if "```json" in content:
            try:
                content = content.split("```json", 1)[1].split("```", 1)[0]
            except (IndexError, ValueError):
                raise ValueError("No JSON content found")
        elif "```" in content:
            # Generic fenced block without language tag
            try:
                content = content.split("```", 1)[1].split("```", 1)[0]
            except (IndexError, ValueError):
                raise ValueError("No JSON content found")

        # Return raw JSON string; caller is responsible for json.loads
        return content.strip()

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

    def _format_transactions_for_compliance(self, transactions: List[Any]) -> str:
        """Format transactions into a concise, numbered list for compliance narratives.

        Each entry includes date, amount (comma-separated with two decimals), transaction
        type, optional description, location ("at"), and method ("via").
        """
        if not transactions:
            return "No transactions provided."

        formatted_entries: List[str] = []

        for idx, tx in enumerate(transactions, start=1):
            # Safely extract values whether dataclass-like or dict
            def _get(field: str, default: str = ""):
                try:
                    return getattr(tx, field, None) or tx.get(field, default)
                except Exception:
                    return default

            date = _get("transaction_date", "Date N/A")
            tx_type = _get("transaction_type", "Transaction")
            description = _get("description", "")
            location = _get("location", "")
            method = _get("method", "")

            raw_amount = _get("amount", None)
            try:
                amount_value = float(raw_amount) if raw_amount is not None else None
                amount_str = f"${amount_value:,.2f}" if amount_value is not None else "Amount N/A"
            except (TypeError, ValueError):
                amount_str = f"{raw_amount}" if raw_amount is not None else "Amount N/A"

            entry_parts = [f"{idx}. {date}: {amount_str} {tx_type}"]
            if description:
                entry_parts.append(f"- {description}")
            if location:
                entry_parts.append(f"at {location}")
            if method:
                entry_parts.append(f"via {method}")

            formatted_entries.append(" ".join(entry_parts))

        return "\n".join(formatted_entries)

    def _validate_narrative_compliance(self, compliance_output: ComplianceOfficerOutput) -> Dict[str, Any]:
        """Validate narrative meets regulatory requirements using LLM JSON validator."""
        # Guard clause for empty input
        if not compliance_output or (isinstance(compliance_output, str) and not compliance_output.strip()):
            raise ValueError("Narrative is empty; cannot validate compliance")

        # Construct validation prompt with explicit schema expectations (escape braces for f-string)
        validation_prompt = (
            f"""
        You are a strict compliance validator. Review the SAR narrative below and return ONLY JSON.

        Compliance_input format is a JSON object which should contain exactly these keys:
        {{
          "narrative": string (the original narrative),
          "narrative_reasoning": string (<=500 chars explaining how it meets requirements),
          "regulatory_citations": array of strings (must include at least one allowed citation),
          "completeness_check": boolean (true only if all requirements are satisfied)
        }}

        Compliance_input to validate:
        """"{compliance_output}""""

        Validation requirements:
        1. Narrative must include customer identification, suspicious activity description, transaction amounts and dates or date ranges, and why the activity is suspicious.
        2. Narrative word count must be < 120 words.
        3. Allowable terminology (should be present when relevant): "Suspicious activity", "Regulatory threshold", "Financial institution", "Money laundering", "Bank Secrecy Act".
        4. regulatory_citations must include at least one of: "31 CFR 1020.320 (BSA)", "12 CFR 21.11 (SAR Filing)", "FinCEN SAR Instructions".

        Return a JSON object output with the following structure:
        {{
            "words_limit_check": boolean (true if narrative is ≤120 words),
            "required_elements_check": boolean (true if all required elements are present),
            "terminology_check": boolean (true if appropriate terminology is used),
            "citations_check": boolean (true if at least one required citation is included),
            "completeness_status": boolean (true if all above checks are true)
        }}
        """
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a compliance validation engine that only returns JSON objects."},
                {"role": "user", "content": validation_prompt}
            ],
            temperature=0,
            max_tokens=400,
            response_format={"type": "json_object"}
        )

        raw_content = response.choices[0].message.content
        validation_json = json.loads(self._extract_json_from_response(raw_content))

        print("Compliance Validation Result:", validation_json)  # Debugging output
        return validation_json

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
