"""
Test script to demonstrate reasoning capabilities
"""
import asyncio
import json
from reasoning_agents import create_reasoning_agent
from enhanced_planning_llm import reasoning_planning_llm


async def test_reasoning_capabilities():
    """Test the new reasoning capabilities with example queries"""
    
    print("🧠 Testing Enhanced Reasoning Capabilities")
    print("=" * 60)
    
    # Test queries of varying complexity
    test_queries = [
        {
            "query": "Show me orders from last 7 days",
            "expected_approach": "standard",
            "description": "Simple data retrieval"
        },
        {
            "query": "Why might prepaid orders have higher AOV than COD orders?",
            "expected_approach": "cot", 
            "description": "Analytical reasoning about business patterns"
        },
        {
            "query": "Analyze our top 3 underperforming states and suggest specific action items for each",
            "expected_approach": "react",
            "description": "Complex multi-step analysis requiring tools and strategic thinking"
        },
        {
            "query": "Compare revenue trends between Flipkart and Amazon over the last 3 months, considering seasonal factors and payment mode preferences",
            "expected_approach": "cot",
            "description": "Complex comparative analysis"
        }
    ]
    
    # Test 1: Meta-Reasoning (Approach Selection)
    print("\n🎯 Test 1: Meta-Reasoning - Automatic Approach Selection")
    print("-" * 50)
    
    meta_agent = create_reasoning_agent("meta")
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n{i}. Query: \"{test_case['query']}\"")
        print(f"   Description: {test_case['description']}")
        print(f"   Expected: {test_case['expected_approach']}")
        
        try:
            result = meta_agent.invoke({
                "query": test_case["query"],
                "context": {},
                "available_reasoning_types": ["react", "cot", "standard"]
            })
            
            recommended = result.get("recommended_approach", "unknown")
            justification = result.get("justification", "No justification provided")
            
            print(f"   🤖 Recommended: {recommended}")
            print(f"   📝 Reasoning: {justification}")
            
            # Check if recommendation matches expectation
            match = "✅" if recommended == test_case["expected_approach"] else "⚠️"
            print(f"   {match} Match: {recommended == test_case['expected_approach']}")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    # Test 2: ReACT Reasoning
    print(f"\n\n🔧 Test 2: ReACT (Reasoning + Acting)")
    print("-" * 50)
    
    react_agent = create_reasoning_agent("react")
    react_query = "Find the top 3 states with declining revenue and suggest action items"
    
    print(f"Query: \"{react_query}\"")
    
    try:
        react_result = react_agent.invoke({
            "query": react_query,
            "available_tools": [
                {
                    "name": "get_orders",
                    "description": "Fetch order data",
                    "parameters": {"start_date": "string", "end_date": "string"}
                },
                {
                    "name": "calculate_metrics",
                    "description": "Calculate revenue metrics", 
                    "parameters": {"data": "object", "metrics": "array"}
                }
            ],
            "context": {},
            "goal": f"Answer comprehensively: {react_query}"
        })
        
        reasoning_trace = react_result.get("reasoning_trace", [])
        answer = react_result.get("answer", "No answer provided")
        
        print(f"\n🧠 Reasoning Steps: {len(reasoning_trace)}")
        
        for i, step in enumerate(reasoning_trace[:3], 1):  # Show first 3 steps
            step_type = step.get("type", "unknown")
            content = step.get("content", "")[:150] + "..." if len(step.get("content", "")) > 150 else step.get("content", "")
            print(f"   {i}. {step_type.upper()}: {content}")
        
        if len(reasoning_trace) > 3:
            print(f"   ... and {len(reasoning_trace) - 3} more reasoning steps")
        
        print(f"\n📋 Final Answer: {answer[:200]}..." if len(answer) > 200 else f"\n📋 Final Answer: {answer}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Test 3: Chain of Thought Reasoning  
    print(f"\n\n🔗 Test 3: Chain of Thought Reasoning")
    print("-" * 50)
    
    cot_agent = create_reasoning_agent("cot")
    cot_query = "Why might there be differences in Average Order Value between different payment modes?"
    
    print(f"Query: \"{cot_query}\"")
    
    try:
        cot_result = cot_agent.invoke({
            "query": cot_query,
            "data": {
                "payment_modes": ["Prepaid", "COD"],
                "sample_data": "Mock order data would be here"
            },
            "analysis_type": "causal"
        })
        
        reasoning_steps = cot_result.get("reasoning_steps", [])
        final_answer = cot_result.get("final_answer", "No answer provided")
        
        print(f"\n🧠 Reasoning Chain: {len(reasoning_steps)} steps")
        
        for i, step in enumerate(reasoning_steps, 1):
            step_name = step.get("step", f"Step {i}")
            step_reasoning = step.get("reasoning", "")[:100] + "..." if len(step.get("reasoning", "")) > 100 else step.get("reasoning", "")
            print(f"   {i}. {step_name}")
            print(f"      💭 {step_reasoning}")
        
        print(f"\n📋 Final Synthesis: {final_answer[:300]}..." if len(final_answer) > 300 else f"\n📋 Final Synthesis: {final_answer}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Test 4: Enhanced Planning
    print(f"\n\n📋 Test 4: Enhanced Planning with Reasoning")
    print("-" * 50)
    
    planning_query = "Analyze why our Shopify revenue dropped last month compared to Flipkart"
    
    print(f"Query: \"{planning_query}\"")
    
    try:
        planning_result = reasoning_planning_llm.invoke({
            "query": planning_query,
            "reasoning_mode": "cot",
            "schema": {}
        })
        
        plan = planning_result.get("plan", {})
        metadata = planning_result.get("reasoning_metadata", {})
        
        print(f"\n✅ Planning Success: {planning_result.get('success', False)}")
        print(f"📊 Plan Type: {plan.get('plan_type', 'unknown')}")
        print(f"🎯 Reasoning Strategy: {plan.get('reasoning_strategy', 'Not specified')}")
        
        steps = plan.get("steps", [])
        print(f"\n📝 Execution Steps: {len(steps)}")
        
        for i, step in enumerate(steps[:4], 1):  # Show first 4 steps
            tool = step.get("tool", "unknown")
            purpose = step.get("reasoning_purpose", "No purpose specified")
            print(f"   {i}. {tool}: {purpose}")
        
        if len(steps) > 4:
            print(f"   ... and {len(steps) - 4} more steps")
        
        complexity = metadata.get("complexity_analysis", {})
        if complexity:
            print(f"\n🔍 Complexity Score: {complexity.get('complexity_score', 'unknown')}/10")
            print(f"📈 Reasoning Type: {complexity.get('reasoning_type', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print(f"\n\n{'=' * 60}")
    print("🎉 Testing Complete!")
    print("\nNext Steps:")
    print("1. Start your FastAPI server: python app.py")
    print("2. Visit http://localhost:5000/docs to explore the new reasoning endpoints")
    print("3. Try the reasoning APIs with complex queries")
    print("4. Check the integration guide: REASONING_INTEGRATION_GUIDE.md")


if __name__ == "__main__":
    asyncio.run(test_reasoning_capabilities())