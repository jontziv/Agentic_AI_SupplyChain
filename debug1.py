import asyncio
from datetime import datetime
from config import SupplyChainConfig
from workflow import SupplyChainWorkflow

async def test_workflow():
    """Test the workflow to identify the issue"""
    
    print("🧪 Testing workflow...")
    
    try:
        # Initialize configuration
        from config import SupplyChainConfig
        print(f"✅ Config initialized")
        print(f"   - GROQ API Key: {'Present' if SupplyChainConfig.GROQ_API_KEY else 'Missing'}")
        print(f"   - Data folder: {SupplyChainConfig.DATA_FOLDER.exists()}")
        
        # Initialize workflow
        workflow = SupplyChainWorkflow(SupplyChainConfig)
        print(f"✅ Workflow initialized")
        
        # Test data
        trigger_data = {
            "trigger_type": "test",
            "timestamp": datetime.now().isoformat(),
            "user": "test_user"
        }
        
        # Run workflow
        print("🚀 Starting workflow...")
        result = await workflow.run_workflow(trigger_data, "test_thread")
        
        print(f"📊 Workflow result type: {type(result)}")
        
        if result:
            if isinstance(result, dict):
                print(f"📋 Result keys: {list(result.keys())}")
                current_step = result.get("current_step", "unknown")
                requires_review = result.get("requires_human_review", False)
                print(f"📍 Current step: {current_step}")
                print(f"🧑‍💼 Requires human review: {requires_review}")
                
                if requires_review:
                    print("✅ Workflow correctly paused for human review!")
                    print("🎉 This is the expected behavior - workflow is working perfectly!")
                    
                    # Test continuing workflow with dummy human feedback
                    print("\n🧪 Testing workflow continuation with dummy approval...")
                    
                    dummy_feedback = {
                        "decision": "✅ Approve all recommendations",
                        "timestamp": datetime.now().isoformat(),
                        "user": "test_user"
                    }
                    
                    continue_result = await workflow.continue_workflow_after_human_feedback(
                        dummy_feedback, "test_thread"
                    )
                    
                    print(f"📊 Continue result: {continue_result}")
                
            else:
                print(f"⚠️ Result is not a dict: {result}")
        else:
            print("❌ Result is None")
        
        # Cleanup
        await workflow.cleanup_checkpointer()
        print("✅ Cleanup completed")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(f"📋 Full traceback: {traceback.format_exc()}")

def test_basic_imports():
    """Test basic imports"""
    print("🧪 Testing imports...")
    
    try:
        from config import SupplyChainConfig
        print("✅ config imported")
        
        from state import SupplyChainState
        print("✅ state imported")
        
        from workflow import SupplyChainWorkflow
        print("✅ workflow imported")
        
        from agents.data_analyst import DataAnalystAgent
        print("✅ data_analyst imported")
        
        from agents.demand_forecaster import DemandForecasterAgent
        print("✅ demand_forecaster imported")
        
        from agents.inventory_optimizer import InventoryOptimizerAgent
        print("✅ inventory_optimizer imported")
        
        from agents.risk_assessor import RiskAssessorAgent
        print("✅ risk_assessor imported")
        
        from agents.validator import ValidatorAgent
        print("✅ validator imported")
        
        import langgraph
        print("✅ langgraph imported")
        
        import langchain_groq
        print("✅ langchain_groq imported")
        
        print("🎉 All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        print(f"📋 Full traceback: {traceback.format_exc()}")
        return False

def test_config():
    """Test configuration"""
    print("🧪 Testing configuration...")
    
    try:
        from config import SupplyChainConfig
        
        print(f"✅ Configuration loaded")
        print(f"   - GROQ_API_KEY present: {bool(SupplyChainConfig.GROQ_API_KEY)}")
        print(f"   - Root folder: {SupplyChainConfig.ROOT_FOLDER}")
        print(f"   - Data folder: {SupplyChainConfig.DATA_FOLDER}")
        print(f"   - Data folder exists: {SupplyChainConfig.DATA_FOLDER.exists()}")
        print(f"   - Prophet script exists: {SupplyChainConfig.PROPHET_SCRIPT.exists()}")
        print(f"   - Reorderpoint script exists: {SupplyChainConfig.REORDERPOINT_SCRIPT.exists()}")
        
        # Test validation
        issues = SupplyChainConfig.validate_config()
        if issues:
            print("⚠️ Configuration issues:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("✅ No configuration issues")
            
        return len(issues) == 0
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        import traceback
        print(f"📋 Full traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("🔍 Starting debug tests...")
    
    # Test 1: Basic imports
    if not test_basic_imports():
        print("❌ Import test failed - fix imports first")
        exit(1)
    
    # Test 2: Configuration
    if not test_config():
        print("❌ Config test failed - fix configuration first")
        exit(1)
    
    # Test 3: Workflow
    asyncio.run(test_workflow())
    
    print("🎉 Debug tests complete!")