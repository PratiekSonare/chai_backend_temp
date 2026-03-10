"""
Enhanced planning LLM with reasoning capabilities
"""
from llm_providers import PlanningLLM
import json
from typing import Dict, List, Any


class ReasoningEnhancedPlanningLLM(PlanningLLM):
    """
    Enhanced Planning LLM that incorporates reasoning strategies
    """
    
    def invoke(self, params: dict) -> dict:
        """
        Generate enhanced execution plan with reasoning steps
        """
        query = params["query"]
        schema = params.get("schema", {})
        reasoning_mode = params.get("reasoning_mode", "standard")
        
        # First, analyze the query complexity
        complexity_analysis = self._analyze_query_complexity(query)
        
        # Generate reasoning-aware plan based on complexity
        if complexity_analysis["requires_reasoning"]:
            return self._generate_reasoning_plan(query, schema, complexity_analysis, reasoning_mode)
        else:
            # Use standard planning for simple queries
            return super().invoke(params)
    
    def _analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """
        Analyze query complexity to determine if reasoning is needed
        """
        prompt = f"""Analyze the complexity of this query and determine if it requires advanced reasoning.

Query: "{query}"

Analyze the following aspects:
1. **Multiple steps required**: Does this query need multiple sequential analysis steps?
2. **Causal reasoning**: Does it ask "why" or requires understanding cause-effect relationships?
3. **Comparative analysis**: Does it require comparing multiple entities or time periods?
4. **Hypothesis formation**: Does it require forming and testing hypotheses?
5. **External knowledge**: Does it require understanding market context or business domain knowledge?
6. **Ambiguity resolution**: Are there ambiguous terms that need clarification?

Return a JSON object with this structure:
{{
  "requires_reasoning": boolean,
  "complexity_score": 1-10,
  "reasoning_type": "causal" | "comparative" | "analytical" | "exploratory" | "simple",
  "key_challenges": ["challenge1", "challenge2"],
  "recommended_approach": "react" | "cot" | "standard",
  "reasoning_justification": "explanation of why this approach is recommended"
}}

Focus on identifying queries that go beyond simple data retrieval and require analytical thinking."""

        try:
            response = self._call_api([{"role": "user", "content": prompt}], temperature=0.3)
            return json.loads(response)
        except Exception as e:
            # Fallback analysis
            return self._fallback_complexity_analysis(query)
    
    def _fallback_complexity_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback complexity analysis using heuristics"""
        query_lower = query.lower()
        
        complexity_indicators = {
            "why": 3,
            "how": 2,
            "analyze": 3,
            "explain": 3,
            "compare": 2,
            "vs": 2,
            "versus": 2,
            "factors": 3,
            "impact": 3,
            "cause": 3,
            "reason": 3,
            "trend": 2,
            "pattern": 2,
            "insight": 3
        }
        
        score = 0
        challenges = []
        
        for indicator, weight in complexity_indicators.items():
            if indicator in query_lower:
                score += weight
                challenges.append(f"Contains '{indicator}' requiring analytical thinking")
        
        # Additional heuristics
        if len(query.split()) > 15:
            score += 1
            challenges.append("Long query with multiple concepts")
        
        if query.count("and") > 2:
            score += 1
            challenges.append("Multiple conditions requiring sequential analysis")
        
        requires_reasoning = score >= 3
        reasoning_type = "analytical" if requires_reasoning else "simple"
        recommended_approach = "cot" if score >= 5 else ("react" if score >= 3 else "standard")
        
        return {
            "requires_reasoning": requires_reasoning,
            "complexity_score": min(score, 10),
            "reasoning_type": reasoning_type,
            "key_challenges": challenges,
            "recommended_approach": recommended_approach,
            "reasoning_justification": f"Score: {score}, challenges: {len(challenges)}"
        }
    
    def _generate_reasoning_plan(self, query: str, schema: dict, complexity: dict, reasoning_mode: str) -> dict:
        """
        Generate a reasoning-aware execution plan
        """
        # Build enhanced schema description
        schema_desc = self._build_enhanced_schema_description(schema)
        tools_desc = self._get_enhanced_tool_descriptions()
        
        reasoning_guidelines = {
            "react": """
Use ReACT (Reasoning + Acting) approach:
1. Break down the query into reasoning steps
2. For each step, think about what information is needed
3. Take actions to gather that information
4. Observe results and reason about next steps
5. Continue until the query is fully answered
            """,
            "cot": """
Use Chain of Thought reasoning:
1. Identify all sub-questions within the main query
2. Plan step-by-step logical analysis
3. Execute each reasoning step systematically
4. Build insights progressively
5. Synthesize final comprehensive answer
            """,
            "standard": """
Use enhanced standard planning with reasoning awareness:
1. Identify required data and calculations
2. Plan execution sequence with logical dependencies
3. Include verification steps
4. Ensure comprehensive analysis
            """
        }
        
        prompt = f"""You are an advanced query planning assistant with reasoning capabilities.

User Query: "{query}"

Query Complexity Analysis:
- Complexity Score: {complexity['complexity_score']}/10
- Reasoning Type: {complexity['reasoning_type']}
- Key Challenges: {complexity.get('key_challenges', [])}
- Recommended Approach: {reasoning_mode}

{reasoning_guidelines.get(reasoning_mode, reasoning_guidelines['standard'])}

Available Data Schema:
{schema_desc}

Available Tools:
{tools_desc}

Create a comprehensive execution plan that addresses the query's complexity. Include:

1. **Reasoning Strategy**: How will you approach this complex query?
2. **Step-by-Step Plan**: Detailed execution steps with reasoning justification
3. **Dependencies**: How steps build on each other logically
4. **Verification Points**: How to validate intermediate results
5. **Synthesis Plan**: How to combine results into final insights

Return a JSON object with this structure:
{{
  "plan_type": "reasoning_enhanced",
  "reasoning_strategy": "description of overall approach",
  "steps": [
    {{
      "id": "step1",
      "tool": "tool_name", 
      "params": {{}},
      "save_as": "result_name",
      "depends_on": [],
      "reasoning_purpose": "why this step is needed",
      "expected_insight": "what insight this step should provide",
      "verification_criteria": "how to validate this step"
    }}
  ],
  "verification_plan": [
    {{
      "step_id": "step1",
      "verification_method": "how to verify this step",
      "confidence_indicators": ["what indicates success"]
    }}
  ],
  "synthesis_strategy": "how to combine all insights into final answer",
  "expected_output_format": "structure of final response",
  "confidence_estimation": "expected confidence level and why"
}}

Important guidelines:
- Include explicit reasoning steps, not just data retrieval
- Plan for intermediate analysis and insight generation
- Consider multiple perspectives when analyzing complex questions
- Include verification of logical consistency
- Plan for handling potential inconsistencies or missing data
"""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self._call_api(messages, temperature=0.4)
            
            plan_data = json.loads(response)
            
            # Validate and enhance the plan
            enhanced_plan = self._enhance_reasoning_plan(plan_data, query, complexity)
            
            return {
                "success": True,
                "plan": enhanced_plan,
                "summarized_query": self._generate_enhanced_summary(query, complexity),
                "reasoning_metadata": {
                    "complexity_analysis": complexity,
                    "planned_reasoning_mode": reasoning_mode,
                    "plan_type": "reasoning_enhanced"
                }
            }
            
        except Exception as e:
            print(f"Error in reasoning plan generation: {e}")
            # Fallback to standard planning
            return super().invoke({"query": query, "schema": schema})
    
    def _enhance_reasoning_plan(self, plan_data: dict, query: str, complexity: dict) -> dict:
        """
        Enhance the generated plan with additional reasoning structure
        """
        enhanced_plan = plan_data.copy()
        
        # Add reasoning metadata to each step
        for step in enhanced_plan.get("steps", []):
            if "reasoning_purpose" not in step:
                step["reasoning_purpose"] = "Data processing step"
            
            if "expected_insight" not in step:
                step["expected_insight"] = "Intermediate result for analysis"
            
            # Add confidence tracking
            step["confidence_tracking"] = True
            
            # Add error handling strategy
            step["error_handling"] = {
                "retry_strategy": "standard",
                "fallback_approach": "continue with available data"
            }
        
        # Add reasoning checkpoints
        enhanced_plan["reasoning_checkpoints"] = self._generate_reasoning_checkpoints(
            enhanced_plan.get("steps", [])
        )
        
        # Add meta-reasoning questions
        enhanced_plan["meta_reasoning_questions"] = [
            "Are we gathering the right information to answer the query?",
            "Do our intermediate results make logical sense?",
            "Are we missing any important perspectives?",
            "How confident are we in our analysis so far?"
        ]
        
        return enhanced_plan
    
    def _generate_reasoning_checkpoints(self, steps: List[dict]) -> List[dict]:
        """
        Generate reasoning checkpoints for plan validation
        """
        checkpoints = []
        
        # Add checkpoint after data gathering steps
        data_steps = [s for s in steps if s.get("tool") in ["get_all_orders_recent", "apply_filters"]]
        if data_steps:
            checkpoints.append({
                "after_step": data_steps[-1]["id"],
                "checkpoint_type": "data_validation",
                "questions": [
                    "Do we have sufficient data to proceed?",
                    "Is the data quality acceptable?",
                    "Are there any obvious data anomalies?"
                ]
            })
        
        # Add checkpoint after analysis steps  
        analysis_steps = [s for s in steps if any(keyword in s.get("tool", "").lower() 
                                                for keyword in ["calculate", "analyze", "group"])]
        if analysis_steps:
            checkpoints.append({
                "after_step": analysis_steps[-1]["id"],
                "checkpoint_type": "analysis_validation", 
                "questions": [
                    "Do the results align with expectations?",
                    "Are there any surprising findings that need investigation?",
                    "Is the analysis comprehensive enough?"
                ]
            })
        
        return checkpoints
    
    def _build_enhanced_schema_description(self, schema: dict) -> str:
        """Build enhanced schema description with reasoning context"""
        if not schema:
            return "No schema information available"
        
        desc = "Available data fields with their characteristics:\n"
        
        for field, info in schema.items():
            desc += f"\n- **{field}** ({info.get('type', 'unknown')})"
            
            if info.get('is_categorical'):
                enum_values = info.get('enum', [])[:8]  # Show first 8 values
                desc += f"\n  - Categorical values: {enum_values}"
                if len(info.get('enum', [])) > 8:
                    desc += " (and more...)"
            
            if info.get('example'):
                desc += f"\n  - Example: {info['example']}"
            
            # Add reasoning context for common fields
            reasoning_context = self._get_field_reasoning_context(field)
            if reasoning_context:
                desc += f"\n  - Analysis notes: {reasoning_context}"
        
        return desc
    
    def _get_field_reasoning_context(self, field_name: str) -> str:
        """Get reasoning context for common fields"""
        context_map = {
            "payment_mode": "Useful for analyzing payment preferences and conversion patterns",
            "marketplace": "Key for marketplace performance comparison and channel analysis",
            "order_status": "Important for understanding order lifecycle and potential issues",
            "state": "Geographic analysis and regional performance insights",
            "total_amount": "Revenue analysis and order value distribution",
            "order_date": "Temporal trends and seasonality analysis",
            "sku": "Product performance and inventory insights"
        }
        
        return context_map.get(field_name.lower(), "")
    
    def _get_enhanced_tool_descriptions(self) -> str:
        """Get enhanced tool descriptions with reasoning context"""
        return """
Available analysis tools:

1. **get_all_orders_recent**: Fetch order data for date ranges
   - Use for: Gathering base data for analysis
   - Reasoning: Start with this to establish data foundation

2. **apply_filters**: Filter data by conditions  
   - Use for: Narrowing data to specific segments
   - Reasoning: Apply early for performance and focused analysis

3. **calculate_metrics**: Calculate business metrics (revenue, AOV, count)
   - Use for: Quantitative analysis and KPI calculation  
   - Reasoning: Core metrics provide foundation for insights

4. **group_and_aggregate**: Group data by dimensions with aggregation
   - Use for: Comparative analysis across categories
   - Reasoning: Reveals patterns and differences between groups

5. **analyze_trends**: Temporal analysis and trend identification
   - Use for: Understanding changes over time
   - Reasoning: Essential for 'why' questions about performance changes

6. **generate_insights**: LLM-powered insight generation from results
   - Use for: Converting analytical results into business insights
   - Reasoning: Final step to provide actionable conclusions
"""
    
    def _generate_enhanced_summary(self, query: str, complexity: dict) -> str:
        """Generate enhanced summary incorporating complexity analysis"""
        base_summary = self._generate_fallback_summary(query)
        
        complexity_note = f" (Complexity: {complexity['complexity_score']}/10 - {complexity['reasoning_type']})"
        
        return base_summary + complexity_note


# Create instance for import
reasoning_planning_llm = ReasoningEnhancedPlanningLLM()