import json
from openai import AsyncOpenAI
import httpx
from typing import Dict, Any, List, Optional
from app.core.config import settings
from app.models.models import LLMPromptTemplate, ProcessingStep
from sqlalchemy.orm import Session

class LLMController:
    def __init__(self):
        self.openai_client = None
        if settings.LLM_PROVIDER == "openai" and settings.OPENAI_API_KEY:
            self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        elif settings.LLM_PROVIDER == "openrouter" and settings.OPENROUTER_API_KEY:
                  return {
            "status": "confirmed",
            "reason": "Basic validation passed",
            "confidence": 0.6,
            "side_effects": [],
            "quality_impact": "positive"
        }
    
    def _enhance_decision_with_context(self, decision: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance LLM decision with dynamic context"""
        # Add dataset-specific adaptations
        if context.get("schema_analysis", {}).get("data_patterns", {}).get("survey_responses"):
            decision["survey_adaptations"] = {
                "preserve_likert_scales": True,
                "validate_response_consistency": True,
                "check_straight_lining": True
            }
        
        # Add complexity-based adjustments
        complexity = context.get("processing_strategy", {}).get("risk_level", "medium")
        if complexity == "high":
            decision["safety_measures"] = {
                "backup_data": True,
                "conservative_parameters": True,
                "step_by_step_validation": True
            }
        
        # Add performance optimizations
        size_category = context.get("schema_analysis", {}).get("basic_categorization", {}).get("size_category", "medium")
        if size_category == "large":
            decision["performance_optimizations"] = {
                "batch_processing": True,
                "memory_efficient": True,
                "progress_tracking": True
            }
        
        return decision
    
    def _extract_dynamic_factors(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key dynamic factors that influenced the decision"""
        factors = {
            "data_size": context.get("metadata", {}).get("basic_info", {}).get("total_rows", 0),
            "missing_data_ratio": self._calculate_missing_ratio(context),
            "data_patterns": context.get("schema_analysis", {}).get("data_patterns", {}),
            "complexity_score": context.get("schema_analysis", {}).get("data_complexity", {}).get("structural_complexity", 0),
            "survey_detected": context.get("schema_analysis", {}).get("data_patterns", {}).get("survey_responses", False)
        }
        
        return factors
    
    def _calculate_missing_ratio(self, context: Dict[str, Any]) -> float:
        """Calculate missing data ratio from context"""
        missing_info = context.get("metadata", {}).get("missing_values", {})
        total_missing = missing_info.get("total_missing", 0)
        basic_info = context.get("metadata", {}).get("basic_info", {})
        total_cells = basic_info.get("total_rows", 1) * basic_info.get("total_columns", 1)
        
        return total_missing / total_cells if total_cells > 0 else 0.0

# Global instance
llm_controller = LLMController()
    
    async def decide_next_step(self, metadata: Dict[str, Any], requirements: Dict[str, Any], 
                             history: List[Dict[str, Any]], db: Session) -> Dict[str, Any]:
        """
        Uses LLM to decide which cleaning step to run next, with dynamic analysis.
        Returns JSON plan with recommended actions based on dataset characteristics.
        """
        # Get dynamic dataset analysis
        from app.modules.ml_processors import ml_processors
        
        # Enhanced analysis context with dynamic characteristics
        analysis_context = {
            "metadata": metadata,
            "requirements": requirements,
            "history": history,
            "available_methods": self._get_available_methods(),
            "dataset_characteristics": metadata.get("dataset_characteristics", {}),
            "data_quality": metadata.get("data_quality", {}),
            "survey_analysis": metadata.get("survey_analysis", {}),
            "dynamic_insights": metadata.get("dynamic_insights", {}),
            "processing_strategy": metadata.get("processing_strategy", {}),
            "schema_analysis": metadata.get("schema_analysis", {})
        }
        
        # Get the dynamic decision prompt template
        prompt_template = self._get_dynamic_prompt_template(db, "decide_step", analysis_context)
        
        # Format the prompt with enhanced context and dynamic adaptation
        prompt = self._format_dynamic_prompt(prompt_template, analysis_context)
        
        try:
            # Call LLM with enhanced context
            response = await self._call_llm(prompt)
            
            # Parse and validate response with dynamic suggestions
            decision = self._parse_dynamic_llm_response(response, analysis_context)
            
            # Enhance decision with dynamic context
            decision = self._enhance_decision_with_context(decision, analysis_context)
            
            # Add reasoning and confidence
            decision["llm_reasoning"] = response.get("reasoning", "")
            decision["confidence"] = response.get("confidence", 0.8)
            decision["analysis_based"] = True
            decision["dynamic_factors"] = self._extract_dynamic_factors(analysis_context)
            
            return decision
            
        except Exception as e:
            # Enhanced fallback with dynamic analysis
            return self._dynamic_fallback_decision(metadata, requirements, history, analysis_context)
    
    async def verify_step_execution(self, pre_meta: Dict[str, Any], post_meta: Dict[str, Any], 
                                  step_info: Dict[str, Any], history: List[Dict[str, Any]], 
                                  db: Session) -> Dict[str, Any]:
        """
        Uses LLM to verify if the intended change happened without negative side effects.
        Returns verification result with status and reasoning.
        """
        # Get the verification prompt template
        prompt_template = self._get_prompt_template(db, "verify_step")
        
        # Prepare context for verification
        context = {
            "pre_metadata": pre_meta,
            "post_metadata": post_meta,
            "step_info": step_info,
            "history": history
        }
        
        # Format the prompt
        prompt = self._format_prompt(prompt_template, context)
        
        try:
            # Call LLM for verification
            response = await self._call_llm(prompt)
            
            # Parse verification response
            verification = {
                "status": response.get("status", "needs_review"),
                "reason": response.get("reason", ""),
                "confidence": response.get("confidence", 0.8),
                "side_effects": response.get("side_effects", []),
                "quality_impact": response.get("quality_impact", "neutral")
            }
            
            return verification
            
        except Exception as e:
            # Fallback verification
            return self._fallback_verification(pre_meta, post_meta, step_info)
    
    def _get_dynamic_prompt_template(self, db: Session, template_type: str, 
                                   analysis_context: Dict[str, Any]) -> str:
        """Get dynamic prompt template based on dataset characteristics"""
        # Try to get custom template from database
        template = db.query(LLMPromptTemplate).filter(
            LLMPromptTemplate.name == template_type
        ).first()
        
        if template:
            return template.template
        
        # Generate dynamic prompt based on dataset characteristics
        if template_type == "decide_step":
            return self._create_dynamic_decision_prompt(analysis_context)
        elif template_type == "verify_step":
            return self._create_dynamic_verification_prompt(analysis_context)
        else:
            return self._get_default_template(template_type)
    
    def _create_dynamic_decision_prompt(self, context: Dict[str, Any]) -> str:
        """Create a dynamic decision prompt based on dataset characteristics"""
        metadata = context.get("metadata", {})
        dataset_chars = metadata.get("dataset_characteristics", {})
        survey_analysis = metadata.get("survey_analysis", {})
        
        base_prompt = """
You are an expert data scientist tasked with cleaning a dataset. Based on the following analysis, 
recommend the next optimal cleaning step.

Dataset Characteristics:
- Size: {size_category}
- Complexity Score: {complexity_score}
- Sparsity Level: {sparsity_level}%
- Data Quality Score: {quality_score}%

Current Data Issues:
{data_issues}

Survey-Specific Context:
{survey_context}

Available Methods:
{available_methods}

Processing History:
{history}

User Requirements:
{requirements}

Based on this analysis, recommend the next cleaning step in JSON format:
{{
    "steps": [
        {{
            "task": "task_name",
            "method": "method_name", 
            "columns": ["col1", "col2"],
            "parameters": {{}},
            "priority": "high|medium|low",
            "description": "explanation of why this step",
            "expected_outcome": "what this will achieve"
        }}
    ],
    "reasoning": "detailed explanation of recommendation",
    "confidence": 0.9
}}
"""
        
        # Format with actual data
        data_issues = self._format_data_issues(metadata)
        survey_context = self._format_survey_context(survey_analysis)
        
        return base_prompt.format(
            size_category=dataset_chars.get("size_category", "unknown"),
            complexity_score=dataset_chars.get("complexity_score", 0),
            sparsity_level=dataset_chars.get("sparsity_level", 0),
            quality_score=metadata.get("quality_score", 0),
            data_issues=data_issues,
            survey_context=survey_context,
            available_methods=json.dumps(context.get("available_methods", {}), indent=2),
            history=json.dumps(context.get("history", []), indent=2),
            requirements=json.dumps(context.get("requirements", {}), indent=2)
        )
    
    def _format_data_issues(self, metadata: Dict[str, Any]) -> str:
        """Format data quality issues for the prompt"""
        issues = []
        
        missing_info = metadata.get("missing_values", {})
        if missing_info.get("total_missing", 0) > 0:
            issues.append(f"Missing values: {missing_info['total_missing']} cells")
        
        duplicates_info = metadata.get("duplicates", {})
        if duplicates_info.get("total_duplicates", 0) > 0:
            issues.append(f"Duplicate rows: {duplicates_info['total_duplicates']}")
        
        outliers_info = metadata.get("outliers", {})
        outlier_cols = [col for col, info in outliers_info.items() 
                       if isinstance(info, dict) and info.get("iqr_outliers", 0) > 0]
        if outlier_cols:
            issues.append(f"Outliers detected in columns: {', '.join(outlier_cols[:5])}")
        
        return "\n".join(f"- {issue}" for issue in issues) if issues else "- No major issues detected"
    
    def _format_survey_context(self, survey_analysis: Dict[str, Any]) -> str:
        """Format survey-specific context for the prompt"""
        context_items = []
        
        weight_vars = survey_analysis.get("weight_variables", [])
        if weight_vars:
            context_items.append(f"Weight variables detected: {', '.join(weight_vars)}")
        
        likert_scales = survey_analysis.get("likert_scales", [])
        if likert_scales:
            context_items.append(f"Likert scales found: {len(likert_scales)} columns")
        
        demo_vars = survey_analysis.get("demographic_variables", [])
        if demo_vars:
            context_items.append(f"Demographic variables: {', '.join(demo_vars[:3])}")
        
        return "\n".join(f"- {item}" for item in context_items) if context_items else "- No survey-specific patterns detected"
    
    def _parse_dynamic_llm_response(self, response: Dict[str, Any], 
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response with dynamic validation"""
        try:
            # Enhanced parsing with context validation
            steps = response.get("steps", [])
            validated_steps = []
            
            for step in steps:
                # Validate step against available methods
                task = step.get("task")
                method = step.get("method")
                
                available_methods = context.get("available_methods", {})
                if task in available_methods and method in available_methods[task]:
                    validated_steps.append(step)
                else:
                    # Try to suggest alternative
                    if task in available_methods:
                        step["method"] = available_methods[task][0]  # Use first available method
                        validated_steps.append(step)
            
            return {
                "steps": validated_steps,
                "original_response": response,
                "validation_applied": len(validated_steps) != len(steps)
            }
            
        except Exception as e:
            # Return safe fallback
            return {"steps": [], "error": str(e)}
    
    def _dynamic_fallback_decision(self, metadata: Dict[str, Any], requirements: Dict[str, Any],
                                 history: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced fallback decision based on dynamic analysis"""
        steps = []
        
        # Priority 1: Handle missing values if significant
        missing_info = metadata.get("missing_values", {})
        total_missing = missing_info.get("total_missing", 0)
        if total_missing > 0:
            cols_with_missing = list(missing_info.get("columns_with_missing", {}).keys())
            if cols_with_missing:
                steps.append({
                    "task": "impute_missing",
                    "method": "median",  # Safe default
                    "columns": cols_with_missing[:5],  # Limit to first 5 columns
                    "priority": "high",
                    "description": f"Address {total_missing} missing values in {len(cols_with_missing)} columns"
                })
        
        # Priority 2: Remove duplicates if found
        duplicates_info = metadata.get("duplicates", {})
        if duplicates_info.get("total_duplicates", 0) > 0:
            steps.append({
                "task": "remove_duplicates",
                "method": "exact",
                "columns": None,
                "priority": "high",
                "description": f"Remove {duplicates_info['total_duplicates']} duplicate rows"
            })
        
        # Priority 3: Handle outliers in numeric columns
        outliers_info = metadata.get("outliers", {})
        outlier_cols = [col for col, info in outliers_info.items() 
                       if isinstance(info, dict) and info.get("iqr_outliers", 0) > 5]
        if outlier_cols:
            steps.append({
                "task": "detect_outliers",
                "method": "iqr",
                "columns": outlier_cols[:3],  # Limit to 3 columns
                "priority": "medium",
                "description": f"Handle outliers in {len(outlier_cols)} numeric columns"
            })
        
        return {
            "steps": steps[:3],  # Limit to 3 steps
            "llm_reasoning": "Fallback rule-based decision due to LLM unavailability",
            "confidence": 0.7,
            "is_fallback": True
        }
    
    async def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """
        Makes the actual LLM API call based on the configured provider.
        """
        if (settings.LLM_PROVIDER in ["openai", "openrouter"]) and self.openai_client:
            return await self._call_openai_compatible(prompt)
        elif settings.LLM_PROVIDER == "local":
            return await self._call_local_llm(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")
    
    async def _call_openai_compatible(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI-compatible API (works for both OpenAI and OpenRouter)"""
        try:
            messages = [
                {"role": "system", "content": "You are an expert data scientist specializing in survey data cleaning. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
            
        except Exception as e:
            raise Exception(f"LLM API error: {e}")
    
    async def _call_openai(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI GPT API (legacy method name for compatibility)"""
        return await self._call_openai_compatible(prompt)
    
    async def _call_local_llm(self, prompt: str) -> Dict[str, Any]:
        """Call local LLM API (e.g., Llama-3)"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{settings.LOCAL_LLM_URL}/generate",
                    json={
                        "prompt": prompt,
                        "max_tokens": 2000,
                        "temperature": 0.3
                    },
                    timeout=60.0
                )
                
                result = response.json()
                return json.loads(result.get("text", "{}"))
                
        except Exception as e:
            raise Exception(f"Local LLM API error: {e}")
    
    def _get_prompt_template(self, db: Session, template_name: str) -> str:
        """Retrieve prompt template from database"""
        template = db.query(LLMPromptTemplate).filter(
            LLMPromptTemplate.name == template_name,
            LLMPromptTemplate.is_active == True
        ).first()
        
        if template:
            return template.template
        else:
            # Return default templates if not found in database
            return self._get_default_template(template_name)
    
    def _get_default_template(self, template_name: str) -> str:
        """Default prompt templates"""
        templates = {
            "decide_step": """
You are the decision brain for a survey dataset cleaning tool. Given the following information, decide which cleaning steps to execute next.

Dataset Metadata:
{metadata}

User Requirements:
{requirements}

Processing History:
{history}

Available Methods:
{available_methods}

Based on this information, decide the next cleaning step(s). Consider:
1. Data quality issues that need addressing
2. User requirements and priorities
3. Logical sequence of operations
4. Risk of data loss or corruption

Respond with a JSON object in this exact format:
{{
  "steps": [
    {{
      "task": "task_name",
      "method": "method_name",
      "columns": ["column1", "column2"],
      "parameters": {{"param1": "value1"}},
      "priority": "high|medium|low",
      "risk_level": "low|medium|high"
    }}
  ],
  "reasoning": "Explanation of why these steps were chosen",
  "confidence": 0.8
}}
""",
            "verify_step": """
You are the quality control agent for a data cleaning operation. Compare the before and after states to verify the operation was successful.

Pre-Step Metadata:
{pre_metadata}

Post-Step Metadata:
{post_metadata}

Step Information:
{step_info}

Processing History:
{history}

Analyze the changes and determine:
1. Did the intended change occur correctly?
2. Were there any unexpected side effects?
3. Is the data quality better or worse?
4. Should this step be approved?

Respond with a JSON object in this exact format:
{{
  "status": "confirmed|needs_review|failed",
  "reason": "Detailed explanation of the verification result",
  "confidence": 0.9,
  "side_effects": ["list", "of", "unexpected", "changes"],
  "quality_impact": "positive|neutral|negative"
}}
"""
        }
        return templates.get(template_name, "")
    
    def _format_prompt(self, template: str, context: Dict[str, Any]) -> str:
        """Format prompt template with context"""
        try:
            return template.format(**context)
        except KeyError as e:
            print(f"Missing context key: {e}")
            return template
    
    def _format_dynamic_prompt(self, template: str, context: Dict[str, Any]) -> str:
        """Format dynamic prompt template with enhanced context"""
        try:
            # Extract specific context elements for dynamic formatting
            metadata = context.get("metadata", {})
            requirements = context.get("requirements", {})
            history = context.get("history", [])
            available_methods = context.get("available_methods", {})
            
            # Create a comprehensive context for formatting
            format_context = {
                "metadata": json.dumps(metadata, indent=2),
                "requirements": json.dumps(requirements, indent=2),
                "history": json.dumps(history, indent=2),
                "available_methods": json.dumps(available_methods, indent=2),
                "dataset_characteristics": json.dumps(metadata.get("dataset_characteristics", {}), indent=2),
                "data_quality": json.dumps(metadata.get("data_quality", {}), indent=2),
                "survey_analysis": json.dumps(metadata.get("survey_analysis", {}), indent=2)
            }
            
            return template.format(**format_context)
        except Exception as e:
            print(f"Error formatting dynamic prompt: {e}")
            # Fallback to basic formatting
            return template.format(
                metadata=json.dumps(context.get("metadata", {}), indent=2),
                requirements=json.dumps(context.get("requirements", {}), indent=2),
                history=json.dumps(context.get("history", []), indent=2),
                available_methods=json.dumps(context.get("available_methods", {}), indent=2)
            )
    
    def _parse_llm_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate LLM response"""
        # Basic validation
        if "steps" not in response:
            raise ValueError("LLM response missing 'steps' field")
        
        # Validate each step
        for step in response["steps"]:
            required_fields = ["task", "method", "columns"]
            for field in required_fields:
                if field not in step:
                    raise ValueError(f"Step missing required field: {field}")
        
        return response
    
    def _get_available_methods(self) -> Dict[str, List[str]]:
        """Return available cleaning methods for each task type"""
        return {
            "remove_duplicates": ["exact_match", "near_match", "fuzzy_match"],
            "impute_missing": ["mean", "median", "mode", "knn", "random_forest", "forward_fill", "backward_fill"],
            "detect_outliers": ["iqr", "zscore", "isolation_forest", "local_outlier_factor"],
            "fix_dtypes": ["auto_convert", "categorical", "numeric", "datetime"],
            "normalize_text": ["lowercase", "remove_special", "standardize_encoding"],
            "handle_categories": ["merge_rare", "create_other", "label_encode"]
        }
    
    def _fallback_decision(self, metadata: Dict[str, Any], requirements: Dict[str, Any], 
                          history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Rule-based fallback when LLM is unavailable"""
        steps = []
        
        # Check for duplicates first
        duplicates_info = metadata.get("duplicates", {})
        if duplicates_info.get("total_duplicates", 0) > 0:
            steps.append({
                "task": "remove_duplicates",
                "method": "exact_match",
                "columns": [],
                "parameters": {},
                "priority": "high",
                "risk_level": "low"
            })
        
        # Check for missing values
        missing_info = metadata.get("missing_values", {})
        if missing_info.get("total_missing", 0) > 0:
            columns_with_missing = list(missing_info.get("columns_with_missing", {}).keys())
            steps.append({
                "task": "impute_missing",
                "method": "median",
                "columns": columns_with_missing[:3],  # Limit to first 3 columns
                "parameters": {},
                "priority": "medium",
                "risk_level": "medium"
            })
        
        return {
            "steps": steps,
            "reasoning": "Fallback rule-based decision due to LLM unavailability",
            "confidence": 0.6
        }
    
    def _fallback_verification(self, pre_meta: Dict[str, Any], post_meta: Dict[str, Any], 
                             step_info: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based fallback verification"""
        # Basic checks
        pre_rows = pre_meta.get("basic_info", {}).get("total_rows", 0)
        post_rows = post_meta.get("basic_info", {}).get("total_rows", 0)
        
        # Check if too much data was lost
        if post_rows < pre_rows * 0.8:  # More than 20% data loss
            return {
                "status": "needs_review",
                "reason": "Significant data loss detected",
                "confidence": 0.9,
                "side_effects": ["major_data_loss"],
                "quality_impact": "negative"
            }
        
        return {
            "status": "confirmed",
            "reason": "Basic validation passed",
            "confidence": 0.7,
            "side_effects": [],
            "quality_impact": "neutral"
        }
    
    def _enhance_decision_with_context(self, decision: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance LLM decision with dynamic context"""
        # Add dataset-specific adaptations
        if context.get("schema_analysis", {}).get("data_patterns", {}).get("survey_responses"):
            decision["survey_adaptations"] = {
                "preserve_likert_scales": True,
                "validate_response_consistency": True,
                "check_straight_lining": True
            }
        
        # Add complexity-based adjustments
        complexity = context.get("processing_strategy", {}).get("risk_level", "medium")
        if complexity == "high":
            decision["safety_measures"] = {
                "backup_data": True,
                "conservative_parameters": True,
                "step_by_step_validation": True
            }
        
        # Add performance optimizations
        size_category = context.get("schema_analysis", {}).get("basic_categorization", {}).get("size_category", "medium")
        if size_category == "large":
            decision["performance_optimizations"] = {
                "batch_processing": True,
                "memory_efficient": True,
                "progress_tracking": True
            }
        
        return decision
    
    def _extract_dynamic_factors(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key dynamic factors that influenced the decision"""
        factors = {
            "data_size": context.get("metadata", {}).get("basic_info", {}).get("total_rows", 0),
            "missing_data_ratio": self._calculate_missing_ratio(context),
            "data_patterns": context.get("schema_analysis", {}).get("data_patterns", {}),
            "complexity_score": context.get("schema_analysis", {}).get("data_complexity", {}).get("structural_complexity", 0),
            "survey_detected": context.get("schema_analysis", {}).get("data_patterns", {}).get("survey_responses", False)
        }
        
        return factors
    
    def _calculate_missing_ratio(self, context: Dict[str, Any]) -> float:
        """Calculate missing data ratio from context"""
        missing_info = context.get("metadata", {}).get("missing_values", {})
        total_missing = missing_info.get("total_missing", 0)
        basic_info = context.get("metadata", {}).get("basic_info", {})
        total_cells = basic_info.get("total_rows", 1) * basic_info.get("total_columns", 1)
        
        return total_missing / total_cells if total_cells > 0 else 0.0

# Global instance
llm_controller = LLMController()
