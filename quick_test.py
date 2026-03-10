#!/usr/bin/env python3
"""
Quick test script - test a single query to see if the new approach works
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def quick_test():
    """Quick test of the hybrid LLM"""
    
    # Check API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Please set OPENROUTER_API_KEY in your .env file")
        return False
    
    # Enable comparison mode
    os.environ["COMPARE_LLM_APPROACHES"] = "true"
    
    try:
        from hybrid_llm import HybridPlanningLLM
        
        print("🧪 Quick test of HybridPlanningLLM")
        print("=" * 40)
        
        llm = HybridPlanningLLM()
        
        # Simple test query
        query = "Show me total revenue from last 3 days"
        print(f"📝 Testing query: {query}")
        
        result = llm.invoke(query)
        
        if result.get("success"):
            print("✅ Test PASSED!")
            print(f"🔧 Approach: {result.get('approach_used', 'unknown')}")
            print(f"📊 Query Type: {result.get('plan', {}).get('query_type', 'unknown')}")
            
            if "comparison_metadata" in result:
                print(f"🔍 Comparison: {result['comparison_metadata']}")
            
            return True
        else:
            print(f"❌ Test FAILED: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    if quick_test():
        print("\n🎉 Success! The hybrid LLM is working.")
        print("\n💡 Next steps:")
        print("1. Run: python test_llm_approaches.py")  
        print("2. If satisfied, set USE_TOOL_CALLING=true in .env")
    else:
        print("\n⚠️  Test failed. Check your API key and dependencies.")