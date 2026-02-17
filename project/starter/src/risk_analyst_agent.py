# Risk Analyst Agent - Chain-of-Thought Implementation
# TODO: Implement Risk Analyst Agent using Chain-of-Thought prompting

"""
Risk Analyst Agent Module

This agent performs suspicious activity classification using Chain-of-Thought reasoning.
It analyzes customer profiles, account behavior, and transaction patterns to identify
potential financial crimes.

YOUR TASKS:
- Study Chain-of-Thought prompting methodology
- Design system prompt with structured reasoning framework
- Implement case analysis with proper error handling
- Parse and validate structured JSON responses
- Create comprehensive audit logging
"""

import json
import openai
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv

# TODO: Import your foundation components
from .foundation_sar import (
    RiskAnalystOutput, 
    ExplainabilityLogger, 
    CaseData
)

# Load environment variables
load_dotenv()

class RiskAnalystAgent:
    """
    Risk Analyst agent using Chain-of-Thought reasoning.
    
    TODO: Implement agent that:
    - Uses systematic Chain-of-Thought prompting
    - Classifies suspicious activity patterns
    - Returns structured JSON output
    - Handles errors gracefully
    - Logs all operations for audit
    """
    
    def __init__(self, openai_client, explainability_logger, model="gpt-3.5-turbo", retry_limit: int = 2): #"gpt-4", "gpt-3.5-turbo"
        """Initialize the Risk Analyst Agent
        
        Args:
            openai_client: OpenAI client instance
            explainability_logger: Logger for audit trails
            model: OpenAI model to use
        """
        # TODO: Initialize agent components
        self.client = openai_client
        self.logger = explainability_logger
        self.model = model
        self.retry_limit = retry_limit
        
        # Build classification categories dynamically from helper function
        classification_categories = get_classification_categories()
        categories_text = "\n".join(
            f"        - **{cat}**: {desc}" 
            for cat, desc in classification_categories.items()
        )

        self.system_prompt = f"""
        You are a Senior Financial Crime Risk Analyst with 15+ years of experience in AML/CFT compliance and SAR (Suspicious Activity Report) processing system. Your role is to analyze suspicious financial activity patterns and classify the underlying risk type.

        **IMPORTANT: Use Chain-of-Thought reasoning. Think step by step, explaining your logic at each stage before reaching conclusions.**

        ## Step-by-Step Analysis Framework

        ### STEP 1: DATA REVIEW
        - Extract and document all relevant transaction details (amounts, frequency, timing, parties involved)
        - Identify account holder information and behavioral baseline
        - Identify relationships between different data entities
        - Note any unusual patterns or deviations from normal activity
        - Flag any red flags in customer profile or transaction metadata
        - **Output**: Structured summary of key facts

        ### STEP 2: PATTERN RECOGNITION
        - Analyze transaction timing and frequency patterns
        - Examine transaction amounts and progressive escalation/de-escalation
        - Identify beneficiaries and transaction counterparties
        - Look for structuring patterns (round amounts, just-below-threshold amounts)
        - Check for layering or circular transaction patterns
        - **Output**: List of identified patterns with confidence levels

        ### STEP 3: REGULATORY MAPPING
        - Map identified patterns to regulatory red flags (31 CFR 1010.320)
        - Consider jurisdictional compliance requirements
        - Identify applicable BSA/AML filing thresholds and triggers
        - Cross-reference with sanctions list indicators
        - **Output**: Mapping of activities to regulatory requirements

        ### STEP 4: RISK QUANTIFICATION
        - Assess likelihood of illicit intent (Low/Medium/High)
        - Evaluate financial impact and AML/CFT risk exposure
        - Consider customer risk profile and beneficial ownership structure
        - Calculate overall risk score (0-100)
        - **Output**: Quantified risk metrics with justification

        ### STEP 5: CLASSIFICATION DECISION
        - Based on all evidence, determine primary classification category
        - Provide alternative classifications if applicable
        - Explain confidence level in primary classification
        - State any mitigating or aggravating factors
        - **Output**: Final classification with reasoning

        ## Classification Categories
        {categories_text}

        ## Classification Guide:
        - **Structuring**: Breaking transactions into smaller amounts to avoid reporting thresholds ($10,000+)
        - **Sanctions**: Activity involving prohibited jurisdictions, entities, or individuals on sanctions lists
        - **Fraud**: Identity theft, account takeover, false statements, or unauthorized transactions
        - **Money_Laundering**: Complex schemes with multiple layers (placement, layering, integration) to obscure illicit fund origins
        - **Other**: Suspicious patterns not fitting the above categories but still requiring investigation

        ## Response Format (JSON)
        Your response MUST be valid JSON with this exact structure:
        {{
            "step_1_data_review": {{
                "key_facts": ["fact1", "fact2", ...],
                "red_flags": ["flag1", "flag2", ...],
                "summary": "Brief summary of customer and transaction details"
            }},
            "step_2_pattern_recognition": {{
                "patterns_identified": [
                    {{"pattern": "description", "confidence": "High/Medium/Low", "reasoning": "why"}}
                ],
                "summary": "Analysis of transaction patterns"
            }},
            "step_3_regulatory_mapping": {{
                "applicable_regulations": ["31 CFR 1010.320", ...],
                "compliance_requirements": ["requirement1", ...],
                "filing_triggers": ["trigger1", ...]
            }},
            "step_4_risk_quantification": {{
                "likelihood_of_illicit_intent": "High/Medium/Low",
                "risk_score": 0,
                "financial_impact": "High/Medium/Low",
                "justification": "Detailed explanation"
            }},
            "step_5_classification": {{
                "classification": "Structuring",
                "confidence_score": 0.85,
                "reasoning": "Step-by-step analysis reasoning",
                "key_indicators": ["indicator1", "indicator2"],
                "risk_level": "High"
            }}
        }}

        ## Constraints and Rules:
        - step_1_data_review.summary: max 250 characters
        - step_2_pattern_recognition.summary: max 250 characters
        - step_4_risk_quantification.likelihood_of_illicit_intent: one of [High, Medium, Low]
        - step_4_risk_quantification.risk_score: integer 0-100
        - step_4_risk_quantification.financial_impact: one of [High, Medium, Low]
        - step_5_classification.classification: one of [Structuring, Sanctions, Fraud, Money_Laundering, Other]
        - step_5_classification.confidence_score: float between 0.0 and 1.0
        - step_5_classification.reasoning: max 300 characters
        - step_5_classification.key_indicators: list of strings (suspicious indicators)
        - step_5_classification.risk_level: one of [Low, Medium, High, Critical]

        """
        
    def analyze_case(self, case_data: CaseData) -> RiskAnalystOutput:
        """
        Perform risk analysis on a case using Chain-of-Thought reasoning.
        
        Workflow:
        - Creates structured user prompt with case details using _format_case_for_prompt
        - Makes OpenAI API call with system prompt
        - Parses and validates JSON response using _extract_json_from_response
        - Handles errors and logs all operations
        - Returns validated RiskAnalystOutput
        
        Args:
            case_data: CaseData object containing customer, account, and transaction information
            
        Returns:
            RiskAnalystOutput object with analysis results
            
        Raises:
            ValueError: If case data is invalid or response parsing fails
            Exception: If OpenAI API call fails
        """
        analysis_start_time = datetime.now()
        case_id = case_data.case_id if hasattr(case_data, 'case_id') else "UNKNOWN"
        
        # try:
        # Step 1: Format case data into structured prompt
        user_prompt = self._format_case_for_prompt(case_data)
        
        raw_response = None
        last_api_error = None
        for attempt in range(1, self.retry_limit + 1):
            try:
                api_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,  # Lower temperature for consistency in financial analysis
                    max_tokens=1000,   # Sufficient for detailed CoT reasoning
                    # response_format={"type": "json_object"}  # Enforce JSON response
                )

                print("llm_response: ", api_response)
                raw_response = api_response.choices[0].message.content
                print("Raw LLM Response: ", raw_response[:500])  # Print first 500 chars for debugging
                break

            except Exception as api_error:
                last_api_error = api_error
                if attempt == self.retry_limit:
                    error_msg = f"OpenAI API call failed after {self.retry_limit} attempts: {str(api_error)}"
                    execution_time_ms = (datetime.now() - analysis_start_time).total_seconds() * 1000
                    self.logger.log_agent_action(
                        agent_type="RiskAnalyst",
                        action="analyze_case",
                        case_id=case_id,
                        input_data={"prompt_length": len(user_prompt)},
                        output_data={},
                        reasoning="OpenAI API call failed",
                        execution_time_ms=execution_time_ms,
                        success=False,
                        error_message=error_msg
                    )
                    raise ValueError(error_msg) from api_error
                else:
                    print(f"API error on attempt {attempt}/{self.retry_limit}: {api_error}. Retrying...")
                    continue
        
        # Step 3: Extract and validate JSON from response (with retries)
        analysis_json = None
        last_json_error = None
        for attempt in range(1, self.retry_limit + 1):
            try:
                json_str = self._extract_json_from_response(raw_response)
                analysis_json = json.loads(json_str)
                break
            except ValueError as json_error:
                last_json_error = json_error
                if attempt == self.retry_limit:
                    error_msg = f"JSON parsing failed after {self.retry_limit} attempts: {str(json_error)}"
                    execution_time_ms = (datetime.now() - analysis_start_time).total_seconds() * 1000
                    
                    self.logger.log_agent_action(
                        agent_type="RiskAnalyst",
                        action="analyze_case",
                        case_id=case_id,
                        input_data={"response_length": len(raw_response) if raw_response else 0},
                        output_data={},
                        reasoning="JSON parsing failed",
                        execution_time_ms=execution_time_ms,
                        success=False,
                        error_message=error_msg
                    )
                    raise ValueError(error_msg) from json_error
                else:
                    print(f"JSON parsing error on attempt {attempt}/{self.retry_limit}: {json_error}. Retrying...")
                    continue
        
        # Step 4: Extract classification and confidence
        # Accept both flat JSON schema and the nested Chain-of-Thought schema
        if "classification" in analysis_json:
            # flat schema returned by some prompts / tests
            classification = analysis_json.get("classification", "Other")
            confidence_score = float(analysis_json.get("confidence_score", 0.0))
        else:
            # nested CoT schema
            classification = analysis_json.get("step_5_classification", {}).get("classification", "Other")
            confidence_score = float(analysis_json.get("step_5_classification", {}).get("confidence_score", 0.0))
        

        # Extract reasoning from all steps
        reasoning_text = f"""
            CHAIN-OF-THOUGHT ANALYSIS SUMMARY:

            # Step 1 - Data Review:
            # {analysis_json.get("step_1_data_review", {}).get("summary", "N/A")}

            # Step 2 - Pattern Recognition:
            # {analysis_json.get("step_2_pattern_recognition", {}).get("summary", "N/A")}

            # Step 3 - Regulatory Mapping:
            # Applicable Regulations: {', '.join(analysis_json.get("step_3_regulatory_mapping", {}).get("applicable_regulations", []))}

            # Step 4 - Risk Quantification:
            # Risk Score: {analysis_json.get("step_4_risk_quantification", {}).get("risk_score", "N/A")}/100
            # Likelihood of Illicit Intent: {analysis_json.get("step_4_risk_quantification", {}).get("likelihood_of_illicit_intent", "N/A")}

            Step 5 - Classification Decision:
            {analysis_json.get("step_5_classification", {}).get("reasoning", "N/A")}
            """
        
        print("Reasoning Text length: ", len(reasoning_text))

        # key indicators and risk level may be in flat or nested schema
        if "key_indicators" in analysis_json:
            key_indicators = analysis_json.get("key_indicators", [])
        else:
            key_indicators = analysis_json.get("step_5_classification", {}).get("key_indicators", [])

        if "risk_level" in analysis_json:
            risk_level = analysis_json.get("risk_level", "Low")
        else:
            risk_level = analysis_json.get("step_5_classification", {}).get("risk_level", "Low")

        
        # Step 5: Create and return RiskAnalystOutput
        output = RiskAnalystOutput(
            classification=classification,
            confidence_score=confidence_score,
            reasoning=reasoning_text,
            key_indicators=key_indicators,
            risk_level=risk_level
        )
        
        # Log successful completion
        execution_time_ms = (datetime.now() - analysis_start_time).total_seconds() * 1000
        
        self.logger.log_agent_action(
            agent_type="RiskAnalyst",
            action="analyze_case",
            case_id=case_id,
            input_data={
                "customer_id": case_data.customer.customer_id,
                "accounts_count": len(case_data.accounts) if hasattr(case_data, 'accounts') else 1,
                "transactions_count": len(case_data.transactions) if hasattr(case_data, 'transactions') else 0
            },
            output_data={
                "classification": classification,
                "confidence_score": confidence_score
            },
            reasoning=f"Risk analysis completed successfully. Classification: {classification}",
            execution_time_ms=execution_time_ms,
            success=True
        )
        
        return output
        

    def _extract_json_from_response(self, response_content: str) -> str:
        """Extract JSON content from LLM response
        
        Handles multiple JSON formats:
        - JSON in code blocks (```json ... ```)
        - JSON in plain text
        - Malformed responses
        - Empty responses
        
        Args:
            response_content: Raw response from LLM
            
        Returns:
            Extracted JSON string
            
        Raises:
            ValueError: If no valid JSON found or response is empty
            json.JSONDecodeError: If extracted content is not valid JSON
        """
        # Handle empty response
        if not response_content or not response_content.strip():
            raise ValueError("No JSON content found")
        
        json_str = None
        
        # Strategy 1: Try to extract from code blocks (```json ... ```)
        if "```json" in response_content:
            try:
                start_idx = response_content.find("```json") + len("```json")
                end_idx = response_content.find("```", start_idx)
                if end_idx > start_idx:
                    json_str = response_content[start_idx:end_idx].strip()
            except Exception:
                pass
        
        # Strategy 2: Try to extract from generic code blocks (``` ... ```)
        if not json_str and "```" in response_content:
            try:
                start_idx = response_content.find("```") + len("```")
                end_idx = response_content.find("```", start_idx)
                if end_idx > start_idx:
                    json_str = response_content[start_idx:end_idx].strip()
            except Exception:
                pass
        
        # Strategy 3: Extract from plain text (find first { and last })
        if not json_str:
            try:
                # Find the first opening brace
                start_idx = response_content.find("{")
                # Find the last closing brace
                end_idx = response_content.rfind("}")
                
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = response_content[start_idx:end_idx + 1]
            except Exception:
                pass
        
        # If still no JSON found, raise error
        if not json_str or not json_str.strip():
            raise ValueError(
                f"Failed to parse Risk Analyst JSON output. Response preview: {response_content[:200]}..."
            )
        
        # Validate extracted JSON by attempting to parse
        try:
            json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Extracted content is not valid JSON: {str(e)}. "
                f"Content preview: {json_str[:200]}..."
            )
        
        return json_str

    def _format_accounts(self, accounts: List[Any]) -> str:
        """Format accounts as space-separated records for prompts.

        Each account will be one line with fields separated by a single space.
        Fields: account_id account_type status current_balance average_monthly_balance opening_date

        Args:
            accounts: list of AccountData-like objects

        Returns:
            A multi-line string where each line describes one account.
        """
        if not accounts:
            return ""

        lines = []
        for acct in accounts:
            acct_id = getattr(acct, 'account_id', 'N/A')
            acct_type = getattr(acct, 'account_type', 'N/A')
            status = getattr(acct, 'status', 'N/A')
            opening = getattr(acct, 'opening_date', 'N/A')

            # Safely format numeric balances
            try:
                curr_bal = getattr(acct, 'current_balance', None)
                curr_bal_str = f"${curr_bal:,.2f}" if curr_bal is not None else "N/A"
            except Exception:
                curr_bal_str = "N/A"

            try:
                avg_bal = getattr(acct, 'average_monthly_balance', None)
                avg_bal_str = f"${avg_bal:,.2f}" if avg_bal is not None else "N/A"
            except Exception:
                avg_bal_str = "N/A"

            # Build a single space-separated record
            # Use single spaces between fields to keep it compact for LLM prompts
            line = f"{acct_id} {acct_type} {status} {curr_bal_str} {avg_bal_str} {opening}"
            lines.append(line)

        return "\n".join(lines)

    def _format_transactions(self, transactions: List[Any]) -> str:
        """Format transactions as a numbered list for prompts.

        Each transaction is one line like:
        1. 2025-01-01: Cash_Deposit $9,900.00 - description | location

        Args:
            transactions: list of TransactionData-like objects

        Returns:
            Multi-line string with numbered transaction entries.
        """
        if not transactions:
            return ""

        # Sort by transaction_date for stable ordering
        try:
            sorted_txs = sorted(transactions, key=lambda x: x.transaction_date)
        except Exception:
            sorted_txs = list(transactions)

        lines = []
        for idx, tx in enumerate(sorted_txs, start=1):
            date = getattr(tx, 'transaction_date', 'N/A')
            tx_type = getattr(tx, 'transaction_type', 'N/A')
            amount = getattr(tx, 'amount', None)
            try:
                amount_str = f"${amount:,.2f}" if amount is not None else "N/A"
            except Exception:
                amount_str = "N/A"

            desc = getattr(tx, 'description', None)
            location = getattr(tx, 'location', None)

            line = f"{idx}. {date}: {tx_type} {amount_str}"
            extras = []
            if desc:
                extras.append(str(desc))
            if location:
                extras.append(str(location))

            if extras:
                line += " - " + " | ".join(extras)

            lines.append(line)

        return "\n".join(lines)

    def _format_case_for_prompt(self, case_data: CaseData) -> str:
        """Format case data for the analysis prompt
        
        Creates a structured, readable format for LLM analysis that includes:
        - Customer profile summary
        - Account information
        - Transaction details with key metrics
        - Financial summary statistics
        
        Args:
            case_data: CaseData object containing customer, account, and transaction info
            
        Returns:
            Formatted string ready for LLM analysis
        """
        # Extract customer data
        customer = case_data.customer
        accounts = case_data.accounts
        transactions = case_data.transactions
        
        # Build prompt string
        prompt = "═" * 80 + "\n"
        prompt += "SUSPICIOUS ACTIVITY CASE FOR ANALYSIS\n"
        prompt += "═" * 80 + "\n\n"
        
        # 1. CUSTOMER PROFILE SUMMARY
        prompt += "📋 CUSTOMER PROFILE\n"
        prompt += "─" * 80 + "\n"
        prompt += f"Name:                 {customer.name}\n"
        prompt += f"Customer ID:          {customer.customer_id}\n"
        prompt += f"Date of Birth:        {customer.date_of_birth}\n"
        prompt += f"SSN (Last 4):         {customer.ssn_last_4}\n"
        prompt += f"Address:              {customer.address}\n"
        if hasattr(customer, 'phone') and customer.phone:
            prompt += f"Phone:                {customer.phone}\n"
        prompt += f"Customer Since:       {customer.customer_since}\n"
        occupation_str = customer.occupation if (hasattr(customer, 'occupation') and customer.occupation is not None) else 'N/A'
        prompt += f"Occupation:           {occupation_str}\n"
        if hasattr(customer, 'annual_income') and customer.annual_income is not None:
            prompt += f"Annual Income:        ${customer.annual_income:,.2f}\n"
        else:
            prompt += "Annual Income:        N/A\n"
        prompt += f"Risk Rating:          {customer.risk_rating}\n\n"
        
        # 2. ACCOUNT INFORMATION
        prompt += "🏦 ACCOUNT INFORMATION\n"
        prompt += "─" * 80 + "\n"
        
        if accounts and len(accounts) > 0:
            prompt += f"Total Accounts:       {len(accounts)}\n"
            prompt += "\n"
            
            for idx, account in enumerate(accounts, 1):
                prompt += f"Account #{idx}:\n"
                prompt += f"  Account ID:         {account.account_id}\n"
                prompt += f"  Account Type:       {account.account_type}\n"
                prompt += f"  Account Status:     {account.status}\n"
                prompt += f"  Current Balance:    ${account.current_balance:,.2f}\n"
                prompt += f"  Avg Monthly Balance: ${account.average_monthly_balance:,.2f}\n"
                prompt += f"  Account Opened:     {account.opening_date}\n"
                if hasattr(account, 'authorized_users') and account.authorized_users:
                    prompt += f"  Authorized Users:   {len(account.authorized_users)}\n"
                prompt += "\n"
        else:
            prompt += "No accounts found in case data.\n\n"
        
        # 3. TRANSACTION SUMMARY STATISTICS
        prompt += "📊 TRANSACTION SUMMARY STATISTICS\n"
        prompt += "─" * 80 + "\n"
        
        if transactions and len(transactions) > 0:
            # Calculate metrics
            total_transactions = len(transactions)
            total_inflow = sum(t.amount for t in transactions if t.amount > 0)
            total_outflow = sum(abs(t.amount) for t in transactions if t.amount < 0)
            total_volume = abs(total_inflow) + abs(total_outflow)
            avg_transaction = total_volume / total_transactions if total_transactions > 0 else 0
            
            # Get date range
            transaction_dates = [t.transaction_date for t in transactions]
            min_date = min(transaction_dates)
            max_date = max(transaction_dates)
            
            prompt += f"Total Transactions:   {total_transactions}\n"
            prompt += f"Date Range:           {min_date} to {max_date}\n"
            prompt += f"Total Inflows:        ${total_inflow:,.2f}\n"
            prompt += f"Total Outflows:       ${total_outflow:,.2f}\n"
            prompt += f"Net Flow:             ${total_inflow - total_outflow:,.2f}\n"
            prompt += f"Total Volume:         ${total_volume:,.2f}\n"
            prompt += f"Average Transaction:  ${avg_transaction:,.2f}\n"
            prompt += f"Max Transaction:      ${max(t.amount for t in transactions):,.2f}\n"
            prompt += f"Min Transaction:      ${min(t.amount for t in transactions):,.2f}\n"
            prompt += "\n"
            
            # 4. TRANSACTION DETAILS WITH KEY METRICS
            prompt += "💳 TRANSACTION DETAILS\n"
            prompt += "─" * 80 + "\n"
            prompt += f"{'Date':<12} {'Type':<20} {'Amount':<15} {'Counterparty':<25} {'Description':<20}\n"
            prompt += "─" * 80 + "\n"
            
            for tx in sorted(transactions, key=lambda x: x.transaction_date):
                counterparty = tx.counterparty if (hasattr(tx, 'counterparty') and tx.counterparty is not None) else "N/A"
                description = tx.description if (hasattr(tx, 'description') and tx.description is not None) else "N/A"
                prompt += f"{tx.transaction_date:<12} {tx.transaction_type:<20} ${tx.amount:>13,.2f} {counterparty:<25} {description:<20}\n"
            
            prompt += "\n"
        else:
            prompt += "No transactions found in case data.\n\n"
        
        # 5. CASE METADATA
        prompt += "📌 CASE METADATA\n"
        prompt += "─" * 80 + "\n"
        if hasattr(case_data, 'case_id'):
            prompt += f"Case ID:              {case_data.case_id}\n"
        if hasattr(case_data, 'filing_institution'):
            prompt += f"Filing Institution:   {case_data.filing_institution}\n"
        if hasattr(case_data, 'observation_period'):
            prompt += f"Observation Period:   {case_data.observation_period}\n"
        if hasattr(case_data, 'suspicious_indicators') and case_data.suspicious_indicators:
            prompt += f"Suspicious Indicators:\n"
            for indicator in case_data.suspicious_indicators:
                prompt += f"  • {indicator}\n"
        prompt += "\n"
        
        prompt += "═" * 80 + "\n"
        prompt += "Please analyze this case using the 5-step Chain-of-Thought framework.\n"
        prompt += "=" * 80 + "\n\n"
        
        return prompt

# ===== PROMPT ENGINEERING HELPERS =====

def create_chain_of_thought_framework():
    """Helper function showing Chain-of-Thought structure
    
    TODO: Study this example and adapt for financial crime analysis:
    
    **Analysis Framework** (Think step-by-step):
    1. **Data Review**: What does the data tell us?
    2. **Pattern Recognition**: What patterns are suspicious?
    3. **Regulatory Mapping**: Which regulations apply?
    4. **Risk Quantification**: How severe is the risk?
    5. **Classification Decision**: What category fits best?
    """
    return {
        "step_1": "Data Review - Examine all available information",
        "step_2": "Pattern Recognition - Identify suspicious indicators", 
        "step_3": "Regulatory Mapping - Connect to known typologies",
        "step_4": "Risk Quantification - Assess severity level",
        "step_5": "Classification Decision - Determine final category"
    }

def get_classification_categories():
    """Standard SAR classification categories
    
    TODO: Use these categories in your prompts:
    """
    return {
        "Structuring": "Transactions designed to avoid reporting thresholds",
        "Sanctions": "Potential sanctions violations or prohibited parties",
        "Fraud": "Fraudulent transactions or identity-related crimes",
        "Money_Laundering": "Complex schemes to obscure illicit fund sources", 
        "Other": "Suspicious patterns not fitting standard categories"
    }

# ===== TESTING UTILITIES =====

def test_agent_with_sample_case():
    """Test the agent with a sample case
    
    TODO: Use this function to test your implementation:
    - Create sample case data
    - Initialize agent
    - Run analysis
    - Validate results
    """
    print("🧪 Testing Risk Analyst Agent")
    print("TODO: Implement test case")

if __name__ == "__main__":
    print("🔍 Risk Analyst Agent Module")
    print("Chain-of-Thought reasoning for suspicious activity classification")
    print("\n📋 TODO Items:")
    print("• Design Chain-of-Thought system prompt")
    print("• Implement analyze_case method")
    print("• Add JSON parsing and validation")
    print("• Create comprehensive error handling")
    print("• Test with sample cases")
    print("\n💡 Key Concepts:")
    print("• Chain-of-Thought: Step-by-step reasoning")
    print("• Structured Output: Validated JSON responses")
    print("• Financial Crime Detection: Pattern recognition")
    print("• Audit Logging: Complete decision trails")
