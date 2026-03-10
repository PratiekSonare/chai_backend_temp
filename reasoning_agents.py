"""
Enhanced reasoning agents with ReACT, Chain of Thought, and multi-step reasoning capabilities
"""
import json
from typing import List, Dict, Any, Tuple
from llm_providers import OpenRouterLLM


class ReACTAgent(OpenRouterLLM):
    """
    ReACT (Reasoning + Acting) Agent
    
    Implements the ReACT paradigm where the agent:
    1. Reasons about the problem (Thought)
    2. Takes an action (Act) 
    3. Observes the result (Observation)
    4. Repeats until goal is achieved
    """
    
    def __init__(self):
        super().__init__()
        self.max_iterations = 10
        self.reasoning_history = []
    
    def invoke(self, params: dict) -> dict:
        """
        Execute ReACT reasoning loop for complex queries
        
        Args:
            params: {
                "query": str,
                "available_tools": list,
                "context": dict,
                "goal": str
            }
        """
        query = params["query"]
        available_tools = params.get("available_tools", [])
        context = params.get("context", {})
        goal = params.get("goal", "Answer the user's query accurately")
        
        self.reasoning_history = []
        iteration = 0
        
        while iteration < self.max_iterations:
            # Reasoning Phase (Thought)
            thought = self._generate_thought(query, context, available_tools, iteration)
            self.reasoning_history.append({"type": "thought", "content": thought, "iteration": iteration})
            
            # Check if reasoning suggests we're done
            if self._is_goal_achieved(thought, goal):
                return self._generate_final_response(query, goal)
            
            # Action Phase (Act)
            action = self._decide_action(thought, available_tools, context)
            self.reasoning_history.append({"type": "action", "content": action, "iteration": iteration})
            
            if action["type"] == "final_answer":
                return self._generate_final_response(query, goal)
            
            # Observation Phase (Observe)
            observation = self._execute_action(action, context)
            self.reasoning_history.append({"type": "observation", "content": observation, "iteration": iteration})
            
            # Update context with new information
            context.update(observation.get("context_updates", {}))
            
            iteration += 1
        
        return {
            "success": False,
            "error": "Max iterations reached without achieving goal",
            "reasoning_trace": self.reasoning_history
        }
    
    def _generate_thought(self, query: str, context: dict, available_tools: list, iteration: int) -> str:
        """Generate reasoning thought for current iteration"""
        
        tools_desc = "\n".join([f"- {tool['name']}: {tool['description']}" for tool in available_tools])
        history_summary = self._summarize_reasoning_history()
        
        prompt = f"""You are an analytical reasoning agent. Think step by step about how to solve this query.

Query: "{query}"

Available Tools:
{tools_desc}

Current Context: {json.dumps(context, indent=2)}

Previous Reasoning History:
{history_summary}

Current Iteration: {iteration + 1}

Think about:
1. What do I know so far?
2. What information do I still need?
3. What's the next logical step?
4. Which tool would be most helpful now?
5. Am I close to having enough information to answer the query?

Provide your reasoning as a clear thought process."""

        messages = [{"role": "user", "content": prompt}]
        return self._call_api(messages, temperature=0.3)
    
    def _decide_action(self, thought: str, available_tools: list, context: dict) -> dict:
        """Decide next action based on reasoning"""
        
        tools_desc = "\n".join([
            f"- {tool['name']}: {tool['description']}\n  Parameters: {tool.get('parameters', {})}"
            for tool in available_tools
        ])
        
        prompt = f"""Based on your reasoning, decide the next action.

Reasoning: "{thought}"

Available Actions:
{tools_desc}
- final_answer: Provide the final answer (use when you have enough information)

Return a JSON object with this structure:
{{
  "type": "tool_name" or "final_answer",
  "parameters": {{"param1": "value1", "param2": "value2"}},
  "justification": "Why this action is the best next step"
}}

If you have enough information to answer the query, use "final_answer".
Otherwise, choose the most appropriate tool to gather the needed information."""

        messages = [{"role": "user", "content": prompt}]
        response = self._call_api(messages, temperature=0.1)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"type": "final_answer", "justification": "Failed to parse action decision"}
    
    def _execute_action(self, action: dict, context: dict) -> dict:
        """Execute the decided action and return observation"""
        tool_name = action.get("type")
        parameters = action.get("parameters", {})
        
        # This would integrate with your existing tool registry
        # For now, returning a mock observation
        return {
            "success": True,
            "result": f"Executed {tool_name} with {parameters}",
            "context_updates": {"last_action": tool_name}
        }
    
    def _is_goal_achieved(self, thought: str, goal: str) -> bool:
        """Check if the goal has been achieved based on reasoning"""
        check_prompt = f"""
Goal: {goal}
Current Thought: {thought}

Based on the reasoning, has the goal been achieved? Answer only "yes" or "no".
"""
        response = self._call_api([{"role": "user", "content": check_prompt}], temperature=0.1)
        return response.lower().strip().startswith("yes")
    
    def _summarize_reasoning_history(self) -> str:
        """Summarize the reasoning history for context"""
        if not self.reasoning_history:
            return "No previous reasoning."
        
        summary = []
        for entry in self.reasoning_history[-6:]:  # Last 6 entries to keep context manageable
            summary.append(f"{entry['type'].upper()}: {entry['content'][:200]}...")
        
        return "\n".join(summary)
    
    def _generate_final_response(self, query: str, goal: str) -> dict:
        """Generate final response based on reasoning history"""
        
        reasoning = self._summarize_reasoning_history()
        
        prompt = f"""Based on your reasoning process, provide a final answer to the query.

Query: "{query}"
Goal: "{goal}"

Your Reasoning Process:
{reasoning}

Provide a comprehensive final answer that addresses the query completely. Include:
1. Direct answer to the question
2. Key insights discovered
3. Supporting evidence from your analysis
4. Any recommendations or next steps

Answer:"""

        messages = [{"role": "user", "content": prompt}]
        final_answer = self._call_api(messages, temperature=0.2)
        
        return {
            "success": True,
            "answer": final_answer,
            "reasoning_trace": self.reasoning_history,
            "reasoning_summary": reasoning
        }


class ChainOfThoughtAgent(OpenRouterLLM):
    """
    Chain of Thought (CoT) Agent
    
    Implements step-by-step reasoning for complex analytical queries
    """
    
    def invoke(self, params: dict) -> dict:
        """
        Execute Chain of Thought reasoning
        
        Args:
            params: {
                "query": str,
                "data": dict,
                "analysis_type": str,
                "steps": list (optional)
            }
        """
        query = params["query"]
        data = params.get("data", {})
        analysis_type = params.get("analysis_type", "general")
        custom_steps = params.get("steps")
        
        if custom_steps:
            reasoning_steps = custom_steps
        else:
            reasoning_steps = self._generate_reasoning_steps(query, analysis_type)
        
        step_results = []
        cumulative_insights = {}
        
        for i, step in enumerate(reasoning_steps):
            step_result = self._execute_reasoning_step(step, query, data, cumulative_insights, i)
            step_results.append(step_result)
            
            # Update cumulative insights
            cumulative_insights.update(step_result.get("insights", {}))
        
        # Synthesize final answer
        final_synthesis = self._synthesize_final_answer(query, step_results)
        
        return {
            "success": True,
            "reasoning_steps": step_results,
            "final_answer": final_synthesis,
            "chain_of_thought_trace": reasoning_steps
        }
    
    def _generate_reasoning_steps(self, query: str, analysis_type: str) -> List[str]:
        """Generate appropriate reasoning steps based on query and analysis type"""
        
        base_steps = {
            "comparison": [
                "Identify what entities are being compared",
                "Determine the comparison criteria and metrics", 
                "Analyze data for each entity separately",
                "Calculate comparative statistics",
                "Identify patterns and differences",
                "Draw insights about the comparison"
            ],
            "trend": [
                "Identify the time period and data points",
                "Calculate baseline metrics",
                "Analyze period-over-period changes",
                "Identify trend patterns (growth, decline, seasonal)",
                "Consider external factors affecting trends",
                "Predict future implications"
            ],
            "distribution": [
                "Identify the dimension being analyzed",
                "Calculate distribution statistics",
                "Analyze concentration and spread",
                "Identify outliers and anomalies", 
                "Compare with expected distributions",
                "Draw insights about the distribution patterns"
            ],
            "general": [
                "Break down the query into sub-questions",
                "Identify required data and calculations",
                "Perform step-by-step analysis",
                "Validate results and check for anomalies",
                "Draw preliminary conclusions",
                "Synthesize final insights"
            ]
        }
        
        return base_steps.get(analysis_type, base_steps["general"])
    
    def _execute_reasoning_step(self, step: str, query: str, data: dict, 
                              cumulative_insights: dict, step_index: int) -> dict:
        """Execute a single reasoning step"""
        
        prompt = f"""Execute this reasoning step carefully:

STEP {step_index + 1}: {step}

Original Query: "{query}"

Available Data Summary:
{self._summarize_data(data)}

Previous Insights:
{json.dumps(cumulative_insights, indent=2)}

For this step, think through:
1. What specific analysis does this step require?
2. What calculations or observations do I need to make?
3. What insights can I derive from this step?
4. How does this step build on previous insights?

Provide your reasoning for this step in a structured format:
- Analysis: What you analyzed
- Calculations: Any calculations performed  
- Observations: What you observed
- Insights: Key insights from this step
"""

        messages = [{"role": "user", "content": prompt}]
        response = self._call_api(messages, temperature=0.3)
        
        return {
            "step": step,
            "step_index": step_index,
            "reasoning": response,
            "insights": self._extract_insights_from_response(response)
        }
    
    def _summarize_data(self, data: dict) -> str:
        """Create a summary of available data for reasoning"""
        if not data:
            return "No data provided"
        
        summary = []
        for key, value in data.items():
            if isinstance(value, list):
                summary.append(f"- {key}: List with {len(value)} items")
            elif isinstance(value, dict):
                summary.append(f"- {key}: Dictionary with {len(value)} keys")
            else:
                summary.append(f"- {key}: {type(value).__name__}")
        
        return "\n".join(summary)
    
    def _extract_insights_from_response(self, response: str) -> dict:
        """Extract structured insights from LLM response"""
        # Simple extraction - could be enhanced with better parsing
        insights = {}
        
        try:
            # Look for key-value patterns in the response
            lines = response.split('\n')
            for line in lines:
                if ':' in line and any(keyword in line.lower() for keyword in ['insight', 'finding', 'conclusion']):
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        insights[key] = value
        except Exception:
            pass
            
        return insights
    
    def _synthesize_final_answer(self, query: str, step_results: List[dict]) -> str:
        """Synthesize final answer from all reasoning steps"""
        
        all_reasoning = "\n\n".join([
            f"Step {result['step_index'] + 1}: {result['step']}\n{result['reasoning']}"
            for result in step_results
        ])
        
        prompt = f"""Synthesize a comprehensive final answer based on the chain of thought reasoning.

Original Query: "{query}"

Complete Reasoning Chain:
{all_reasoning}

Provide a final answer that:
1. Directly addresses the original query
2. Incorporates insights from all reasoning steps
3. Presents a clear, coherent conclusion
4. Includes supporting evidence from the analysis
5. Identifies any limitations or caveats

Final Answer:"""

        messages = [{"role": "user", "content": prompt}]
        return self._call_api(messages, temperature=0.2)


class MetaReasoningAgent(OpenRouterLLM):
    """
    Meta-Reasoning Agent that decides which reasoning approach to use
    """
    
    def invoke(self, params: dict) -> dict:
        """
        Decide the best reasoning approach for a given query
        
        Args:
            params: {
                "query": str,
                "context": dict,
                "available_reasoning_types": list
            }
        """
        query = params["query"]
        context = params.get("context", {})
        available_types = params.get("available_reasoning_types", ["react", "cot", "standard"])
        
        reasoning_choice = self._choose_reasoning_approach(query, context, available_types)
        reasoning_parameters = self._generate_reasoning_parameters(query, reasoning_choice, context)
        
        return {
            "recommended_approach": reasoning_choice,
            "reasoning_parameters": reasoning_parameters,
            "justification": reasoning_parameters.get("justification", "")
        }
    
    def _choose_reasoning_approach(self, query: str, context: dict, available_types: list) -> str:
        """Choose the best reasoning approach"""
        
        types_desc = {
            "react": "ReACT (Reasoning + Acting) - Best for complex queries requiring multiple steps and tool usage",
            "cot": "Chain of Thought - Best for analytical queries requiring step-by-step logical reasoning", 
            "standard": "Standard planning - Best for simple, direct queries"
        }
        
        available_desc = "\n".join([f"- {t}: {types_desc.get(t, 'Unknown type')}" for t in available_types])
        
        prompt = f"""Analyze this query and recommend the best reasoning approach.

Query: "{query}"

Context: {json.dumps(context, indent=2)}

Available Reasoning Approaches:
{available_desc}

Consider:
1. Query complexity (simple vs multi-step)
2. Need for external tools/data
3. Type of reasoning required (logical analysis vs action-oriented)
4. Ambiguity level

Respond with just the approach name: {', '.join(available_types)}"""

        messages = [{"role": "user", "content": prompt}]
        response = self._call_api(messages, temperature=0.1).strip().lower()
        
        # Find best match
        for approach in available_types:
            if approach in response:
                return approach
        
        return "standard"  # Fallback
    
    def _generate_reasoning_parameters(self, query: str, approach: str, context: dict) -> dict:
        """Generate parameters for the chosen reasoning approach"""
        
        if approach == "react":
            return {
                "max_iterations": 8,
                "goal": f"Comprehensively answer: {query}",
                "justification": "Query requires multi-step reasoning with potential tool usage"
            }
        elif approach == "cot":
            analysis_type = self._determine_analysis_type(query)
            return {
                "analysis_type": analysis_type,
                "justification": f"Query requires {analysis_type} analysis with step-by-step reasoning"
            }
        else:
            return {
                "justification": "Query can be handled with standard planning approach"
            }
    
    def _determine_analysis_type(self, query: str) -> str:
        """Determine the type of analysis needed for CoT reasoning"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["compare", "vs", "versus", "difference"]):
            return "comparison"
        elif any(word in query_lower for word in ["trend", "over time", "growth", "decline"]):
            return "trend" 
        elif any(word in query_lower for word in ["distribution", "breakdown", "split", "across"]):
            return "distribution"
        else:
            return "general"


# Factory function to create the appropriate reasoning agent
def create_reasoning_agent(agent_type: str) -> Any:
    """Factory function to create reasoning agents"""
    agents = {
        "react": ReACTAgent,
        "cot": ChainOfThoughtAgent, 
        "meta": MetaReasoningAgent
    }
    
    if agent_type not in agents:
        raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(agents.keys())}")
    
    return agents[agent_type]()