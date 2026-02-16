# Foundation SAR Tests - Top 10 Essential Tests

"""
Streamlined test suite for foundation_sar.py module focusing on core functionality
"""

import pytest
import os
import sys
from datetime import datetime
from pathlib import Path

# Configure Python path to find src/ module from foundation_sar.py
# This works from any location (notebook, pytest, IDE, etc.)
# project_root = os.path.abspath('..')
# src_path = os.path.join(project_root, 'src')
# tests_path = os.path.join(project_root, 'tests')


try:
    
    from foundation_sar import (
        CustomerData,
        AccountData, 
        TransactionData,
        CaseData,
        ExplainabilityLogger,
        DataLoader
    )
    
    # Test if classes are actually implemented (not just empty pass statements)
    try:
        # Check if CustomerData has proper fields defined, not just an empty pass
        # If the class is just "pass", it won't have any model fields
        if hasattr(CustomerData, 'model_fields') and CustomerData.model_fields:
            # Try to create a simple instance to see if it's properly implemented
            test_customer = CustomerData(
                customer_id="TEST", 
                name="Test", 
                date_of_birth="1990-01-01",
                ssn_last_4="1234",
                address="123 Test St",
                customer_since="2020-01-01",
                risk_rating="Low", 
                annual_income=50000
            )
            # If we get here, the implementation exists and works
            FOUNDATION_IMPLEMENTED = True
        else:
            # No model fields defined - just empty pass statements
            FOUNDATION_IMPLEMENTED = False
    except Exception as e:
        # Any error means implementation is incomplete
        FOUNDATION_IMPLEMENTED = False
        
except ImportError:
    # Graceful fallback when students haven't implemented yet
    FOUNDATION_IMPLEMENTED = False
    print("test_foundation could not import foundation_sar components:")

if not FOUNDATION_IMPLEMENTED:
    print("⚠️  Foundation components not yet implemented - tests will be skipped")
    print("💡 Implement the classes in src/foundation_sar.py to run these tests")

class TestCustomerData:
    """Test CustomerData Pydantic schema"""
    
    @pytest.mark.skipif(not FOUNDATION_IMPLEMENTED, reason="Foundation not implemented yet")
    def test_valid_customer_data(self):
        """Test CustomerData with complete valid inputs"""
        customer_data = {
            "customer_id": "CUST_0001",
            "name": "Allison Hill",
            "date_of_birth": "1958-08-25",
            "ssn_last_4": "2679",
            "address": "600 Jeffery Parkways, New Jamesside, MT 29394",
            "phone": "394.802.6542x351",
            "customer_since": "2016-06-14",
            "risk_rating": "Low",
            "occupation": "Local government officer",
            "annual_income": 48815
        }
        
        customer = CustomerData(**customer_data)
        assert customer.customer_id == "CUST_0001"
        assert customer.name == "Allison Hill"
        assert customer.risk_rating == "Low"
        assert customer.annual_income == 48815

    @pytest.mark.skipif(not FOUNDATION_IMPLEMENTED, reason="Foundation not implemented yet")
    def test_customer_risk_rating_validation(self):
        """Test CustomerData risk rating validation"""
        # Test valid risk ratings
        for rating in ["Low", "Medium", "High"]:
            customer = CustomerData(
                customer_id=f"CUST_{rating}",
                name="Test Customer",
                date_of_birth="1980-01-01",
                ssn_last_4="1234",
                address="123 Test St",
                customer_since="2020-01-01",
                risk_rating=rating
            )
            assert customer.risk_rating == rating

class TestAccountData:
    """Test AccountData Pydantic schema"""
    
    @pytest.mark.skipif(not FOUNDATION_IMPLEMENTED, reason="Foundation not implemented yet")
    def test_valid_account_data(self):
        """Test AccountData with valid inputs"""
        account_data = {
            "account_id": "CUST_0001_ACC_1",
            "customer_id": "CUST_0001",
            "account_type": "Checking",
            "opening_date": "2016-06-14",
            "current_balance": 51690.75,
            "average_monthly_balance": 45000.00,
            "status": "Active"
        }
        
        account = AccountData(**account_data)
        assert account.account_id == "CUST_0001_ACC_1"
        assert account.account_type == "Checking"
        assert account.current_balance == 51690.75
        assert account.status == "Active"

    @pytest.mark.skipif(not FOUNDATION_IMPLEMENTED, reason="Foundation not implemented yet")
    def test_account_balance_validation(self):
        """Test AccountData balance validation"""
        # Test zero balance
        account = AccountData(
            account_id="ACC_TEST",
            customer_id="CUST_TEST",
            account_type="Checking",
            opening_date="2020-01-01",
            current_balance=0.0,
            average_monthly_balance=0.0,
            status="Active"
        )
        assert account.current_balance == 0.0

class TestTransactionData:
    """Test TransactionData Pydantic schema"""
    
    @pytest.mark.skipif(not FOUNDATION_IMPLEMENTED, reason="Foundation not implemented yet")
    def test_valid_transaction_data(self):
        """Test TransactionData with valid inputs"""
        transaction_data = {
            "transaction_id": "TXN_B24455F3",
            "account_id": "CUST_0001_ACC_1",
            "transaction_date": "2025-01-08",
            "transaction_type": "Online_Transfer",
            "amount": 9900.0,
            "description": "ONLINE TRANSFER TO SAVINGS",
            "counterparty": "WELLS FARGO BANK",
            "location": "ONLINE",
            "method": "ACH"
        }
        
        transaction = TransactionData(**transaction_data)
        assert transaction.transaction_id == "TXN_B24455F3"
        assert transaction.amount == 9900.0
        assert transaction.transaction_type == "Online_Transfer"

    @pytest.mark.skipif(not FOUNDATION_IMPLEMENTED, reason="Foundation not implemented yet")
    def test_transaction_amount_validation(self):
        """Test TransactionData amount validation"""
        # Test positive amount
        transaction = TransactionData(
            transaction_id="TXN_TEST",
            account_id="ACC_TEST",
            transaction_date="2025-01-01",
            transaction_type="Deposit",
            amount=100.50,
            description="Test deposit",
            method="ACH"
        )
        assert transaction.amount == 100.50

class TestCaseData:
    """Test CaseData unified schema"""
    
    @pytest.mark.skipif(not FOUNDATION_IMPLEMENTED, reason="Foundation not implemented yet")
    def test_valid_case_creation(self):
        """Test creating a valid case with all components"""
        # Create sample customer
        customer = CustomerData(
            customer_id="CUST_0001",
            name="Test Customer",
            date_of_birth="1980-01-01",
            ssn_last_4="1234",
            address="123 Test St",
            customer_since="2020-01-01",
            risk_rating="Medium",
            annual_income=75000
        )
        
        # Create sample account
        account = AccountData(
            account_id="CUST_0001_ACC_1",
            customer_id="CUST_0001",
            account_type="Checking",
            opening_date="2020-01-01",
            current_balance=10000.0,
            average_monthly_balance=8000.0,
            status="Active"
        )
        
        # Create sample transaction
        transaction = TransactionData(
            transaction_id="TXN_001",
            account_id="CUST_0001_ACC_1",
            transaction_date="2025-01-01",
            transaction_type="Cash_Deposit",
            amount=9900.0,
            description="Cash deposit just under threshold",
            method="Cash"
        )
        
        # Create unified case
        case = CaseData(
            case_id="CASE_001",
            customer=customer,
            accounts=[account],
            transactions=[transaction],
            case_created_at=datetime.now().isoformat(),
            data_sources={
                "customer_source": "test_data",
                "account_source": "test_data", 
                "transaction_source": "test_data"
            }
        )
        
        assert case.case_id == "CASE_001"
        assert case.customer.customer_id == "CUST_0001"
        assert len(case.accounts) == 1
        assert len(case.transactions) == 1

class TestDataLoader:
    """Test DataLoader functionality"""
    
    @pytest.mark.skipif(not FOUNDATION_IMPLEMENTED, reason="Foundation not implemented yet")
    def test_csv_data_loading(self):
        """Test DataLoader can create cases from data"""
        logger = ExplainabilityLogger("test_audit.jsonl")
        loader = DataLoader(logger)
        
        # Sample data with all required fields
        customer_data = {
            "customer_id": "CUST_0001",
            "name": "Test Customer",
            "date_of_birth": "1980-01-01",
            "ssn_last_4": "1234",
            "address": "123 Test St",
            "customer_since": "2020-01-01",
            "risk_rating": "Medium",
            "annual_income": 75000
        }
        
        account_data = [{
            "account_id": "CUST_0001_ACC_1",
            "customer_id": "CUST_0001",
            "account_type": "Checking",
            "opening_date": "2020-01-01",
            "current_balance": 10000.0,
            "average_monthly_balance": 8000.0,
            "status": "Active"
        }]
        
        transaction_data = [{
            "transaction_id": "TXN_001",
            "account_id": "CUST_0001_ACC_1",
            "transaction_date": "2025-01-01",
            "transaction_type": "Cash_Deposit",
            "amount": 9900.0,
            "description": "Test transaction",
            "method": "Cash"
        }]
        
        case = loader.create_case_from_data(customer_data, account_data, transaction_data)
        assert case is not None
        assert case.customer.customer_id == "CUST_0001"
        assert len(case.accounts) == 1
        assert len(case.transactions) == 1
        
        # Cleanup
        if os.path.exists("test_audit.jsonl"):
            os.remove("test_audit.jsonl")

class TestExplainabilityLogger:
    """Test ExplainabilityLogger audit functionality"""
    
    @pytest.mark.skipif(not FOUNDATION_IMPLEMENTED, reason="Foundation not implemented yet")
    def test_log_creation(self):
        """Test ExplainabilityLogger can create log entries"""
        logger = ExplainabilityLogger("test_log.jsonl")
        
        logger.log_agent_action(
            agent_type="TestAgent",
            action="test_action",
            case_id="CASE_001",
            input_data={"test": "input"},
            output_data={"test": "output"},
            reasoning="Test reasoning",
            execution_time_ms=100.0,
            success=True
        )
        
        assert len(logger.entries) == 1
        assert logger.entries[0]["agent_type"] == "TestAgent"
        assert logger.entries[0]["case_id"] == "CASE_001"
        
        # Cleanup
        if os.path.exists("test_log.jsonl"):
            os.remove("test_log.jsonl")

    @pytest.mark.skipif(not FOUNDATION_IMPLEMENTED, reason="Foundation not implemented yet")
    def test_log_file_writing(self):
        """Test ExplainabilityLogger writes to file"""
        log_file = "test_file_log.jsonl"
        logger = ExplainabilityLogger(log_file)
        
        # Log multiple entries
        for i in range(3):
            logger.log_agent_action(
                agent_type="TestAgent",
                action=f"test_action_{i}",
                case_id=f"CASE_{i:03d}",
                input_data={"iteration": i},
                output_data={"result": f"output_{i}"},
                reasoning=f"Test reasoning {i}",
                execution_time_ms=50.0 + i,
                success=True
            )
        
        # Check file exists and has content
        assert os.path.exists(log_file)
        
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 3
        
        # Cleanup
        if os.path.exists(log_file):
            os.remove(log_file)


# ===== MAIN FUNCTION FOR DIRECT TESTING =====

if __name__ == "__main__":
    """Run tests directly when executed as a script"""
    
    print("🧪 Running test_foundation.py tests directly")
    print("=" * 60)
    
    if not FOUNDATION_IMPLEMENTED:
        print("❌ Foundation components not implemented yet!")
        print("💡 Please implement the classes in src/foundation_sar.py first")
        exit(1)
    
    try:
        # Test CustomerData
        print("\n👥 Testing CustomerData...")
        test_customer = TestCustomerData()
        test_customer.test_valid_customer_data()
        print("   ✅ test_valid_customer_data PASSED")
        
        test_customer.test_customer_risk_rating_validation()
        print("   ✅ test_customer_risk_rating_validation PASSED")
        
        # Test AccountData
        print("\n🏦 Testing AccountData...")
        test_account = TestAccountData()
        test_account.test_valid_account_data()
        print("   ✅ test_valid_account_data PASSED")
        
        test_account.test_account_balance_validation()
        print("   ✅ test_account_balance_validation PASSED")
        
        # Test TransactionData
        print("\n💳 Testing TransactionData...")
        test_transaction = TestTransactionData()
        test_transaction.test_valid_transaction_data()
        print("   ✅ test_valid_transaction_data PASSED")
        
        test_transaction.test_transaction_amount_validation()
        print("   ✅ test_transaction_amount_validation PASSED")
        
        # Test CaseData
        print("\n📋 Testing CaseData...")
        test_case = TestCaseData()
        test_case.test_valid_case_creation()
        print("   ✅ test_valid_case_creation PASSED")
        
        # Test DataLoader
        print("\n📊 Testing DataLoader...")
        test_loader = TestDataLoader()
        test_loader.test_csv_data_loading()
        print("   ✅ test_csv_data_loading PASSED")
        
        # Test ExplainabilityLogger
        print("\n📝 Testing ExplainabilityLogger...")
        test_logger = TestExplainabilityLogger()
        test_logger.test_log_creation()
        print("   ✅ test_log_creation PASSED")
        
        test_logger.test_log_file_writing()
        print("   ✅ test_log_file_writing PASSED")
        
        # Summary
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your foundation_sar.py implementation is complete and working.")
        
    except AssertionError as e:
        print(f"\n❌ Test failed with assertion error:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        exit(1)
        
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        exit(1)