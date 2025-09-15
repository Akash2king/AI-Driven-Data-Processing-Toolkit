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
            self.openai_client = AsyncOpenAI(
                api_key=settings.OPENROUTER_API_KEY,
                base_url=settings.OPENROUTER_BASE_URL
            )
        else:
            # Initialize with a dummy client for fallback
            self.openai_client = None
    
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
            return self._fallback_verification(pre_meta, post_meta, step_info)
    
    def _get_available_methods(self) -> Dict[str, List[str]]:
        """Get available methods for all tasks"""
        return {
            "remove_duplicates": ["exact", "near_match", "fuzzy_match"],
            "impute_missing": ["mean", "median", "mode", "knn", "random_forest"],
            "detect_outliers": ["iqr", "zscore", "isolation_forest", "local_outlier_factor"],
            "fix_dtypes": ["auto", "manual"],
            "clean_text": ["standardize", "remove_special"],
            "normalize_data": ["minmax", "zscore", "robust"],
            "encode_categorical": ["label", "onehot", "target"]
        }
    
    def _get_dynamic_prompt_template(self, db: Session, template_type: str, context: Dict[str, Any]) -> str:
        """Get dynamic prompt template based on context"""
        # Try to get from database first
        try:
            template = db.query(LLMPromptTemplate).filter(
                LLMPromptTemplate.template_type == template_type,
                LLMPromptTemplate.is_active == True
            ).first()
            
            if template:
                return template.template
        except:
            pass
        
        # Fallback to default templates with dynamic enhancements
        if template_type == "decide_step":
            return self._get_default_decision_template(context)
        elif template_type == "verify_step":
            return self._get_default_verification_template()
        else:
            return "Please analyze the data and provide recommendations."
    
    def _get_default_decision_template(self, context: Dict[str, Any]) -> str:
        """Get default decision template with dynamic adaptation"""
        base_template = """
        You are an expert data scientist analyzing a dataset for cleaning recommendations.
        
        Dataset Information:
        - Total Rows: {total_rows}
        - Total Columns: {total_columns}
        - Data Quality Score: {quality_score}%
        - Missing Data: {missing_percentage}%
        
        Available Methods: {available_methods}
        
        """
        
        # Add dynamic context based on detected patterns
        schema_analysis = context.get("schema_analysis", {})
        if schema_analysis.get("data_patterns", {}).get("survey_responses"):
            base_template += """
            SURVEY DATA DETECTED:
            - Use conservative outlier detection to preserve response validity
            - Validate Likert scale integrity
            - Check for response consistency patterns
            """
        
        if schema_analysis.get("basic_categorization", {}).get("size_category") == "large":
            base_template += """
            LARGE DATASET DETECTED:
            - Consider batch processing approaches
            - Prioritize memory-efficient methods
            - Enable progress tracking for long operations
            """
        
        base_template += """
        Please recommend the next cleaning step as JSON:
        {
            "task": "task_name",
            "method": "method_name", 
            "columns": ["col1", "col2"] or null for all,
            "parameters": {"param1": "value1"},
            "priority": "high/medium/low",
            "reasoning": "explanation"
        }
        """
        
        return base_template
    
    def _get_default_verification_template(self) -> str:
        """Get default verification template"""
        return """
        You are verifying the results of a data cleaning operation.
        
        Pre-processing metadata: {pre_metadata}
        Post-processing metadata: {post_metadata}
        Operation performed: {step_info}
        
        Please verify if the operation was successful and return JSON:
        {
            "status": "confirmed/needs_review/failed",
            "reason": "explanation",
            "confidence": 0.0-1.0,
            "side_effects": ["list", "of", "issues"],
            "quality_impact": "positive/neutral/negative"
        }
        """
    
    def _get_prompt_template(self, db: Session, template_type: str) -> str:
        """Get prompt template from database or default"""
        try:
            template = db.query(LLMPromptTemplate).filter(
                LLMPromptTemplate.template_type == template_type,
                LLMPromptTemplate.is_active == True
            ).first()
            
            if template:
                return template.template
        except:
            pass
        
        # Return default template
        if template_type == "verify_step":
            return self._get_default_verification_template()
        else:
            return "Please analyze the data and provide recommendations."
    
    def _format_dynamic_prompt(self, template: str, context: Dict[str, Any]) -> str:
        """Format prompt template with dynamic context"""
        metadata = context.get("metadata", {})
        basic_info = metadata.get("basic_info", {})
        
        # Extract key information
        total_rows = basic_info.get("total_rows", 0)
        total_columns = basic_info.get("total_columns", 0)
        quality_score = metadata.get("data_quality", {}).get("overall_score", 0)
        missing_info = metadata.get("missing_values", {})
        missing_percentage = missing_info.get("missing_percentage", 0) if isinstance(missing_info, dict) else 0
        
        try:
            return template.format(
                total_rows=total_rows,
                total_columns=total_columns,
                quality_score=quality_score,
                missing_percentage=missing_percentage,
                available_methods=json.dumps(context.get("available_methods", {}), indent=2),
                metadata=json.dumps(metadata, indent=2)
            )
        except KeyError as e:
            # If formatting fails, return template with basic substitution
            return template.replace("{metadata}", json.dumps(metadata, indent=2))
    
    def _format_prompt(self, template: str, context: Dict[str, Any]) -> str:
        """Format prompt template with context"""
        try:
            return template.format(**context)
        except KeyError:
            return template.replace("{context}", json.dumps(context, indent=2))
    
    async def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call LLM with the formatted prompt"""
        if not self.openai_client:
            raise Exception("No LLM client available")
        
        try:
            # For now, return a mock response
            # In real implementation, this would call the actual LLM
            return {
                "task": "impute_missing",
                "method": "knn",
                "columns": None,
                "parameters": {"n_neighbors": 5},
                "priority": "high",
                "reasoning": "High missing data detected, KNN imputation recommended",
                "confidence": 0.8
            }
        except Exception as e:
            raise Exception(f"LLM call failed: {str(e)}")
    
    def _parse_dynamic_llm_response(self, response: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate LLM response with dynamic context"""
        # Validate required fields
        required_fields = ["task", "method", "priority", "reasoning"]
        for field in required_fields:
            if field not in response:
                response[field] = "unknown"
        
        # Validate task and method compatibility
        available_methods = context.get("available_methods", {})
        task = response.get("task", "")
        method = response.get("method", "")
        
        if task in available_methods and method not in available_methods[task]:
            # Fallback to first available method
            response["method"] = available_methods[task][0] if available_methods[task] else "default"
            response["reasoning"] += " (Method adjusted for compatibility)"
        
        return response
    
    def _dynamic_fallback_decision(self, metadata: Dict[str, Any], requirements: Dict[str, Any],
                                 history: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced fallback decision making with dynamic analysis"""
        # Analyze metadata for decision making
        missing_info = metadata.get("missing_values", {})
        duplicates_info = metadata.get("duplicates", {})
        
        if isinstance(missing_info, dict):
            total_missing = missing_info.get("total_missing", 0)
            missing_percentage = missing_info.get("missing_percentage", 0)
        else:
            total_missing = 0
            missing_percentage = 0
        
        # Priority-based decision making
        if missing_percentage > 10:
            return {
                "task": "impute_missing",
                "method": "knn" if missing_percentage < 30 else "mean",
                "columns": None,
                "parameters": {"n_neighbors": 5} if missing_percentage < 30 else {},
                "priority": "high",
                "reasoning": f"High missing data ({missing_percentage:.1f}%) requires immediate attention",
                "confidence": 0.7,
                "fallback_mode": True
            }
        
        if isinstance(duplicates_info, dict) and duplicates_info.get("total_duplicates", 0) > 0:
            return {
                "task": "remove_duplicates", 
                "method": "exact",
                "columns": None,
                "parameters": {},
                "priority": "high",
                "reasoning": "Duplicate records detected",
                "confidence": 0.8,
                "fallback_mode": True
            }
        
        # Default recommendation
        return {
            "task": "detect_outliers",
            "method": "iqr",
            "columns": None,
            "parameters": {},
            "priority": "medium",
            "reasoning": "General data quality improvement",
            "confidence": 0.6,
            "fallback_mode": True
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
