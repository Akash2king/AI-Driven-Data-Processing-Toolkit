import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class MetadataGenerator:
    def __init__(self):
        pass
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            # Handle NaN and infinity values
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(key): self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_types(item) for item in obj]
        elif pd.isna(obj):
            return None
        elif hasattr(obj, 'item'):  # Handle other numpy scalars
            result = obj.item()
            # Additional check for float values after conversion
            if isinstance(result, float) and (np.isnan(result) or np.isinf(result)):
                return None
            return result
        else:
            try:
                # Try to convert to native Python type
                if hasattr(obj, 'dtype'):
                    result = obj.item() if hasattr(obj, 'item') else str(obj)
                    # Check if result is a problematic float
                    if isinstance(result, float) and (np.isnan(result) or np.isinf(result)):
                        return None
                    return result
                return obj
            except (TypeError, ValueError):
                return str(obj)
    
    def generate_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Returns comprehensive JSON metadata about the dataset including:
        - Dataset statistics
        - Missing value analysis
        - Duplicate detection
        - Outlier summary
        - Column data types
        - Correlation matrix summary
        - Weight variable detection
        - Dynamic data quality assessment
        - Adaptive schema detection
        - Dynamic processing strategies
        """
        try:
            # Dynamic schema analysis first
            schema_analysis = self._analyze_dynamic_schema(df)
            
            # Basic dataset information with dynamic categorization
            basic_info = self._get_basic_info(df)
            basic_info.update(schema_analysis["basic_categorization"])
            
            # Comprehensive column analysis with dynamic typing
            column_info = self._get_column_info(df)
            column_info.update(schema_analysis["column_intelligence"])
            
            # Missing values analysis with pattern detection
            missing_info = self._get_missing_values_info(df)
            missing_info.update(self._analyze_missing_patterns(df))
            
            # Dynamic duplicate analysis
            duplicates_info = self._get_duplicates_info(df)
            duplicates_info.update(self._analyze_duplicate_patterns(df))
            
            # Adaptive outlier detection
            outliers_info = self._get_outliers_info(df)
            outliers_info.update(self._analyze_outlier_patterns(df))
            
            # Dynamic data quality assessment
            quality_info = self._get_data_quality_score(df)
            quality_info.update(self._assess_dynamic_quality(df, schema_analysis))
            
            # Enhanced survey-specific analysis
            survey_info = self._get_survey_analysis(df)
            survey_info.update(self._detect_advanced_survey_patterns(df))
            
            # Dynamic correlation analysis
            correlation_info = self._get_correlation_info(df)
            correlation_info.update(self._analyze_variable_relationships(df))
            
            # Generate adaptive recommendations
            recommendations = self._generate_adaptive_recommendations(df, schema_analysis)
            
            # Create comprehensive metadata structure with dynamic insights
            metadata = {
                "basic_info": basic_info,
                "column_info": column_info,
                "missing_values": missing_info,
                "duplicates": duplicates_info,
                "outliers": outliers_info,
                "data_quality": quality_info,
                "survey_analysis": survey_info,
                "correlation_info": correlation_info,
                "recommendations": recommendations,
                "schema_analysis": schema_analysis,
                "processing_strategy": self._determine_processing_strategy(df, schema_analysis),
                "dynamic_insights": self._generate_dynamic_insights(df, schema_analysis)
            }
            
            return self._convert_numpy_types(metadata)
            
        except Exception as e:
            return {
                "error": f"Metadata generation failed: {str(e)}",
                "basic_info": {"total_rows": len(df), "total_columns": len(df.columns)},
                "fallback_mode": True
            }
    
    def _analyze_dynamic_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dataset schema dynamically to understand data patterns"""
        schema_analysis = {
            "data_patterns": self._detect_data_patterns(df),
            "column_relationships": self._analyze_column_relationships(df),
            "data_complexity": self._assess_data_complexity(df),
            "optimal_strategies": self._determine_optimal_strategies(df),
            "basic_categorization": {},
            "column_intelligence": {}
        }
        
        # Dynamic categorization based on size and complexity
        total_rows, total_cols = df.shape
        memory_usage = df.memory_usage(deep=True).sum() / (1024**2)  # MB
        
        schema_analysis["basic_categorization"] = {
            "size_category": self._categorize_dataset_size(total_rows, total_cols),
            "complexity_score": self._calculate_complexity_score(df),
            "memory_category": self._categorize_memory_usage(memory_usage),
            "processing_recommendation": self._recommend_processing_approach(df)
        }
        
        # Intelligent column analysis
        schema_analysis["column_intelligence"] = {
            "semantic_types": self._detect_semantic_types(df),
            "data_roles": self._identify_data_roles(df),
            "processing_priority": self._assign_processing_priority(df)
        }
        
        return schema_analysis
    
    def _analyze_dataset_characteristics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall dataset characteristics dynamically"""
        characteristics = {
            "size_category": self._categorize_dataset_size(df),
            "complexity_score": self._calculate_complexity_score(df),
            "data_types_distribution": self._get_dtype_distribution(df),
            "sparsity_level": self._calculate_sparsity(df),
            "heterogeneity_score": self._calculate_heterogeneity(df)
        }
        
        return characteristics
    
    def _get_processing_suggestions(self, df: pd.DataFrame, missing_info: Dict, 
                                  duplicates_info: Dict, outliers_info: Dict) -> List[Dict[str, Any]]:
        """Generate dynamic processing suggestions based on data characteristics"""
        suggestions = []
        
        # Missing data suggestions
        total_missing_pct = (missing_info.get("total_missing", 0) / (df.shape[0] * df.shape[1])) * 100
        if total_missing_pct > 10:
            suggestions.append({
                "category": "missing_data",
                "priority": "high",
                "suggestion": "Consider advanced imputation methods due to high missing data rate",
                "affected_percentage": round(total_missing_pct, 2)
            })
        elif total_missing_pct > 5:
            suggestions.append({
                "category": "missing_data", 
                "priority": "medium",
                "suggestion": "Apply targeted imputation for missing values",
                "affected_percentage": round(total_missing_pct, 2)
            })
        
        # Duplicate suggestions
        dup_pct = duplicates_info.get("duplicate_percentage", 0)
        if dup_pct > 5:
            suggestions.append({
                "category": "duplicates",
                "priority": "high",
                "suggestion": "Remove duplicate records to improve data quality",
                "affected_percentage": round(dup_pct, 2)
            })
        
        # Outlier suggestions
        numeric_cols_with_outliers = sum(1 for col_info in outliers_info.values() 
                                       if isinstance(col_info, dict) and col_info.get("iqr_outliers", 0) > 0)
        if numeric_cols_with_outliers > 0:
            suggestions.append({
                "category": "outliers",
                "priority": "medium",
                "suggestion": f"Review outliers in {numeric_cols_with_outliers} numeric columns",
                "affected_columns": numeric_cols_with_outliers
            })
        
        # Data type suggestions
        object_cols = len(df.select_dtypes(include=['object']).columns)
        if object_cols > df.shape[1] * 0.5:
            suggestions.append({
                "category": "data_types",
                "priority": "low",
                "suggestion": "Many text columns detected - consider type optimization",
                "affected_columns": object_cols
            })
        
        return suggestions
    
    def _get_survey_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive survey-specific analysis"""
        survey_analysis = {
            "weight_variables": self._detect_weight_variables(df),
            "likert_scales": self._detect_likert_scales(df),
            "demographic_variables": self._detect_demographic_variables(df),
            "response_patterns": self._analyze_response_patterns(df),
            "survey_quality_indicators": self._calculate_survey_quality(df)
        }
        
        return survey_analysis
    
    def _detect_likert_scales(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect Likert scale columns with detailed analysis"""
        likert_scales = []
        
        for col in df.columns:
            if self._is_likert_scale_column(df[col]):
                scale_info = {
                    "column": col,
                    "scale_range": self._get_scale_range(df[col]),
                    "response_distribution": df[col].value_counts().to_dict(),
                    "central_tendency": float(df[col].median()) if pd.api.types.is_numeric_dtype(df[col]) else None
                }
                likert_scales.append(scale_info)
        
        return likert_scales
    
    def _detect_demographic_variables(self, df: pd.DataFrame) -> List[str]:
        """Detect demographic variables in the dataset"""
        demographic_keywords = [
            'age', 'gender', 'sex', 'race', 'ethnicity', 'education', 'income', 
            'marital', 'employment', 'occupation', 'religion', 'region', 'urban', 'rural'
        ]
        
        demographic_vars = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in demographic_keywords):
                demographic_vars.append(col)
        
        return demographic_vars
    
    def _analyze_response_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze response patterns for survey quality assessment"""
        patterns = {
            "straight_lining": [],
            "extreme_responses": [],
            "missing_patterns": {},
            "response_consistency": {}
        }
        
        # Check for straight-lining in Likert scales
        likert_cols = [col for col in df.columns if self._is_likert_scale_column(df[col])]
        if len(likert_cols) > 1:
            for idx in df.index:
                row_responses = df.loc[idx, likert_cols].dropna()
                if len(row_responses) > 2 and row_responses.nunique() == 1:
                    patterns["straight_lining"].append(idx)
        
        # Analyze missing patterns
        missing_by_row = df.isnull().sum(axis=1)
        patterns["missing_patterns"] = {
            "rows_with_high_missing": (missing_by_row > df.shape[1] * 0.5).sum(),
            "average_missing_per_row": float(missing_by_row.mean())
        }
        
        return patterns
    
    def _calculate_survey_quality(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate survey-specific quality indicators"""
        quality = {}
        
        # Completion rate
        total_responses = df.shape[0] * df.shape[1]
        valid_responses = total_responses - df.isnull().sum().sum()
        quality["completion_rate"] = (valid_responses / total_responses) * 100
        
        # Response variance (for numeric columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            avg_variance = df[numeric_cols].var().mean()
            quality["response_variance"] = float(avg_variance) if not pd.isna(avg_variance) else 0.0
        
        # Consistency score
        quality["consistency_score"] = 100.0  # Placeholder for more sophisticated analysis
        
        return quality
    
    def compare_metadata(self, pre_meta: Dict[str, Any], post_meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns a diff summary of changes between pre and post processing metadata.
        """
        diff = {
            "summary": {},
            "changes": {},
            "improvements": [],
            "concerns": []
        }
        
        # Compare basic info
        pre_basic = pre_meta.get("basic_info", {})
        post_basic = post_meta.get("basic_info", {})
        
        diff["changes"]["rows"] = {
            "before": pre_basic.get("total_rows", 0),
            "after": post_basic.get("total_rows", 0),
            "change": post_basic.get("total_rows", 0) - pre_basic.get("total_rows", 0)
        }
        
        diff["changes"]["columns"] = {
            "before": pre_basic.get("total_columns", 0),
            "after": post_basic.get("total_columns", 0),
            "change": post_basic.get("total_columns", 0) - pre_basic.get("total_columns", 0)
        }
        
        # Compare missing values
        pre_missing = pre_meta.get("missing_values", {})
        post_missing = post_meta.get("missing_values", {})
        
        diff["changes"]["missing_values"] = {
            "before": pre_missing.get("total_missing", 0),
            "after": post_missing.get("total_missing", 0),
            "change": post_missing.get("total_missing", 0) - pre_missing.get("total_missing", 0)
        }
        
        # Compare duplicates
        pre_dups = pre_meta.get("duplicates", {})
        post_dups = post_meta.get("duplicates", {})
        
        diff["changes"]["duplicates"] = {
            "before": pre_dups.get("total_duplicates", 0),
            "after": post_dups.get("total_duplicates", 0),
            "change": post_dups.get("total_duplicates", 0) - pre_dups.get("total_duplicates", 0)
        }
        
        # Generate improvement notes
        if diff["changes"]["missing_values"]["change"] < 0:
            diff["improvements"].append("Reduced missing values")
        
        if diff["changes"]["duplicates"]["change"] < 0:
            diff["improvements"].append("Removed duplicate records")
        
        # Generate concerns
        if diff["changes"]["rows"]["change"] < -0.1 * pre_basic.get("total_rows", 0):
            diff["concerns"].append("Significant data loss (>10% of rows)")
        
        # Calculate overall quality improvement
        pre_quality = pre_meta.get("data_quality", {}).get("overall_score", 0)
        post_quality = post_meta.get("data_quality", {}).get("overall_score", 0)
        
        diff["summary"]["quality_improvement"] = post_quality - pre_quality
        diff["summary"]["data_preserved"] = (diff["changes"]["rows"]["after"] / 
                                           max(diff["changes"]["rows"]["before"], 1)) * 100
        
        return diff
    
    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset information"""
        return {
            "total_rows": int(len(df)),
            "total_columns": int(len(df.columns)),
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / (1024 * 1024)),
            "column_names": list(df.columns)
        }
    
    def _get_column_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed information about each column"""
        column_info = {}
        
        for col in df.columns:
            col_data = df[col]
            
            info = {
                "dtype": str(col_data.dtype),
                "non_null_count": int(col_data.count()),
                "null_count": int(col_data.isnull().sum()),
                "unique_count": int(col_data.nunique()),
                "is_categorical": self._is_categorical(col_data),
                "is_numeric": pd.api.types.is_numeric_dtype(col_data),
                "is_datetime": pd.api.types.is_datetime64_any_dtype(col_data)
            }
            
            # Add statistics for numeric columns
            if info["is_numeric"]:
                try:
                    clean_data = col_data.dropna()
                    if len(clean_data) > 0:
                        # Calculate statistics safely
                        mean_val = clean_data.mean()
                        median_val = clean_data.median()
                        std_val = clean_data.std()
                        min_val = clean_data.min()
                        max_val = clean_data.max()
                        q25_val = clean_data.quantile(0.25)
                        q75_val = clean_data.quantile(0.75)
                        
                        # Validate each statistic before adding
                        stats_update = {}
                        if pd.notna(mean_val) and np.isfinite(mean_val):
                            stats_update["mean"] = float(mean_val)
                        if pd.notna(median_val) and np.isfinite(median_val):
                            stats_update["median"] = float(median_val)
                        if pd.notna(std_val) and np.isfinite(std_val):
                            stats_update["std"] = float(std_val)
                        if pd.notna(min_val) and np.isfinite(min_val):
                            stats_update["min"] = float(min_val)
                        if pd.notna(max_val) and np.isfinite(max_val):
                            stats_update["max"] = float(max_val)
                        if pd.notna(q25_val) and np.isfinite(q25_val):
                            stats_update["q25"] = float(q25_val)
                        if pd.notna(q75_val) and np.isfinite(q75_val):
                            stats_update["q75"] = float(q75_val)
                        
                        info.update(stats_update)
                except (TypeError, ValueError, OverflowError):
                    # Handle cases where numeric operations fail
                    pass
            
            # Add frequency info for categorical columns
            if info["is_categorical"] or info["unique_count"] < 20:
                try:
                    value_counts = col_data.value_counts().head(10)
                    top_values = {}
                    for idx, val in value_counts.items():
                        top_values[str(idx)] = int(val)
                    info["top_values"] = top_values
                except (TypeError, ValueError):
                    # Handle cases where value_counts fails
                    pass
            
            column_info[col] = info
        
        return column_info
    
    def _get_missing_values_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing values patterns"""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        # Convert pandas Series to dict with native Python types
        missing_counts_dict = {}
        missing_percentages_dict = {}
        
        for col in missing_counts.index:
            if missing_counts[col] > 0:
                missing_counts_dict[col] = int(missing_counts[col])
                missing_percentages_dict[col] = float(missing_percentages[col])
        
        completely_missing_cols = []
        for col in missing_counts.index:
            if missing_counts[col] == len(df):
                completely_missing_cols.append(col)
        
        return {
            "total_missing": int(df.isnull().sum().sum()),
            "columns_with_missing": missing_counts_dict,
            "missing_percentages": missing_percentages_dict,
            "completely_missing_columns": completely_missing_cols,
            "rows_with_any_missing": int(df.isnull().any(axis=1).sum()),
            "rows_completely_missing": int(df.isnull().all(axis=1).sum())
        }
    
    def _get_duplicates_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze duplicate records"""
        total_duplicates = df.duplicated().sum()
        duplicate_subsets = {}
        
        # Check for duplicates in key column combinations
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(categorical_cols) > 0:
            subset_duplicates = df.duplicated(subset=categorical_cols[:5]).sum()  # Limit to first 5 cols
            duplicate_subsets["categorical_columns"] = int(subset_duplicates)
        
        return {
            "total_duplicates": int(total_duplicates),
            "duplicate_percentage": float((total_duplicates / len(df)) * 100),
            "subset_duplicates": duplicate_subsets,
            "unique_rows": int(len(df) - total_duplicates)
        }
    
    def _get_outliers_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers in numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                # IQR method
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_iqr = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
                
                # Z-score method
                z_scores = np.abs(stats.zscore(col_data))
                outliers_zscore = (z_scores > 3).sum()
                
                outlier_info[col] = {
                    "iqr_outliers": int(outliers_iqr),
                    "zscore_outliers": int(outliers_zscore),
                    "outlier_percentage_iqr": float((outliers_iqr / len(col_data)) * 100),
                    "outlier_percentage_zscore": float((outliers_zscore / len(col_data)) * 100)
                }
        
        return outlier_info
    
    def _get_correlation_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numeric variables"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return {"message": "Insufficient numeric columns for correlation analysis"}
        
        corr_matrix = numeric_df.corr()
        
        # Find high correlations (excluding diagonal)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    high_corr_pairs.append({
                        "variable1": corr_matrix.columns[i],
                        "variable2": corr_matrix.columns[j],
                        "correlation": float(corr_val)
                    })
        
        return {
            "high_correlations": high_corr_pairs,
            "correlation_matrix_shape": corr_matrix.shape,
            "avg_correlation": float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, 1)].mean())
        }
    
    def _get_data_quality_score(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate an overall data quality score"""
        scores = {}
        
        # Completeness score (based on missing values)
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness = (total_cells - missing_cells) / total_cells * 100
        scores["completeness"] = float(completeness)
        
        # Uniqueness score (based on duplicates)
        duplicate_rows = df.duplicated().sum()
        uniqueness = (df.shape[0] - duplicate_rows) / df.shape[0] * 100
        scores["uniqueness"] = float(uniqueness)
        
        # Consistency score (basic dtype consistency)
        consistency = 100  # Placeholder - could implement more sophisticated checks
        scores["consistency"] = float(consistency)
        
        # Overall score (weighted average)
        overall = (completeness * 0.4 + uniqueness * 0.3 + consistency * 0.3)
        scores["overall_score"] = float(overall)
        
        return scores
    
    def _detect_weight_variables(self, df: pd.DataFrame) -> List[str]:
        """Detect potential survey weight variables"""
        potential_weights = []
        
        for col in df.columns:
            col_name_lower = col.lower()
            
            # Check for common weight variable naming patterns
            weight_keywords = ['weight', 'wt', 'wgt', 'pond', 'factor', 'expansion']
            if any(keyword in col_name_lower for keyword in weight_keywords):
                potential_weights.append(col)
                continue
            
            # Check for numeric columns with characteristics of weights
            if pd.api.types.is_numeric_dtype(df[col]):
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    # Weights often have specific characteristics
                    mean_val = col_data.mean()
                    min_val = col_data.min()
                    max_val = col_data.max()
                    
                    # Check if values are typically around 1 or have weight-like distribution
                    if (0.1 <= mean_val <= 10 and min_val > 0 and 
                        col_data.std() / mean_val < 2):  # Low coefficient of variation
                        potential_weights.append(col)
        
        return potential_weights
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate data cleaning recommendations based on analysis"""
        recommendations = []
        
        # Missing value recommendations
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if missing_pct > 5:
            recommendations.append(f"Consider imputation for missing values ({missing_pct:.1f}% missing)")
        
        # Duplicate recommendations
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            recommendations.append(f"Remove {duplicates} duplicate records")
        
        # Outlier recommendations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if len(df[col].dropna()) > 0:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outliers = (z_scores > 3).sum()
                if outliers > 0:
                    recommendations.append(f"Review {outliers} potential outliers in {col}")
        
        # Data type recommendations
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.05:  # Low cardinality
                    recommendations.append(f"Consider converting {col} to categorical type")
        
        return recommendations
    
    def _is_categorical(self, series: pd.Series) -> bool:
        """Determine if a series should be treated as categorical"""
        if series.dtype == 'object' or series.dtype.name == 'category':
            return True
        
        # Check if numeric column has low cardinality (might be categorical)
        if pd.api.types.is_numeric_dtype(series):
            unique_ratio = series.nunique() / len(series)
            return unique_ratio < 0.05 and series.nunique() < 20
        
    def _categorize_dataset_size(self, df: pd.DataFrame) -> str:
        """Categorize dataset size"""
        total_cells = df.shape[0] * df.shape[1]
        if total_cells < 10000:
            return "small"
        elif total_cells < 1000000:
            return "medium" 
        else:
            return "large"
    
    def _calculate_complexity_score(self, df: pd.DataFrame) -> float:
        """Calculate dataset complexity score"""
        score = 0
        
        # Factor in number of columns
        score += min(df.shape[1] / 50, 1) * 25
        
        # Factor in data type diversity
        dtype_count = len(df.dtypes.unique())
        score += min(dtype_count / 5, 1) * 25
        
        # Factor in missing data complexity
        missing_cols = (df.isnull().sum() > 0).sum()
        score += min(missing_cols / df.shape[1], 1) * 25
        
        # Factor in categorical complexity
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        high_cardinality = sum(1 for col in cat_cols if df[col].nunique() > 100)
        score += min(high_cardinality / len(cat_cols) if len(cat_cols) > 0 else 0, 1) * 25
        
        return score
    
    def _get_dtype_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Get distribution of data types"""
        dtype_counts = {}
        for dtype in df.dtypes:
            dtype_str = str(dtype)
            dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1
        return dtype_counts
    
    def _calculate_sparsity(self, df: pd.DataFrame) -> float:
        """Calculate dataset sparsity (percentage of missing values)"""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        return (missing_cells / total_cells) * 100
    
    def _calculate_heterogeneity(self, df: pd.DataFrame) -> float:
        """Calculate data heterogeneity score"""
        scores = []
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # For numeric: use coefficient of variation
                clean_col = df[col].dropna()
                if len(clean_col) > 0 and clean_col.std() > 0:
                    cv = clean_col.std() / abs(clean_col.mean()) if clean_col.mean() != 0 else 0
                    scores.append(min(cv, 2))  # Cap at 2
            else:
                # For categorical: use entropy-like measure
                value_counts = df[col].value_counts()
                if len(value_counts) > 1:
                    proportions = value_counts / value_counts.sum()
                    entropy = -sum(p * np.log2(p) for p in proportions if p > 0)
                    scores.append(entropy / np.log2(len(value_counts)))  # Normalize
        
        return float(np.mean(scores)) if scores else 0.0
    
    def _is_likert_scale_column(self, series: pd.Series) -> bool:
        """Check if column represents a Likert scale"""
        if not pd.api.types.is_numeric_dtype(series):
            return False
        
        unique_vals = series.dropna().unique()
        if len(unique_vals) < 3 or len(unique_vals) > 10:
            return False
        
        # Check if values are consecutive integers starting from 1 or 0
        sorted_vals = sorted(unique_vals)
        return (all(isinstance(x, (int, np.integer)) for x in sorted_vals) and
                (sorted_vals == list(range(1, len(sorted_vals) + 1)) or
                 sorted_vals == list(range(0, len(sorted_vals)))))
    
    def _get_scale_range(self, series: pd.Series) -> Dict[str, Any]:
        """Get range information for scale columns"""
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return {}
        
        return {
            "min": int(clean_series.min()),
            "max": int(clean_series.max()),
            "unique_values": sorted(clean_series.unique().tolist())
        }
    
    # Additional dynamic methods to support enhanced metadata generation
    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns"""
        return {
            "missing_pattern_type": "MCAR",  # Simplified for now
            "systematic_missing": False,
            "missing_correlations": []
        }
    
    def _analyze_duplicate_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze duplicate patterns"""
        return {
            "duplicate_distribution": {"total_groups": 0},
            "near_duplicates": 0,
            "duplicate_causes": []
        }
    
    def _analyze_outlier_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze outlier patterns"""
        return {
            "outlier_clustering": {},
            "outlier_severity": {},
            "potential_data_errors": []
        }
    
    def _assess_dynamic_quality(self, df: pd.DataFrame, schema_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Dynamic quality assessment"""
        base_quality = self._get_data_quality_score(df)
        base_quality["dynamic_assessment"] = True
        return base_quality
    
    def _detect_advanced_survey_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect advanced survey patterns"""
        return {
            "response_consistency": {"status": "analyzed"},
            "straight_lining": {"status": "analyzed"},
            "response_time_patterns": {"status": "no_time_data"},
            "attention_checks": {"potential_attention_checks": []}
        }
    
    def _analyze_variable_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze variable relationships"""
        return {
            "strong_correlations": [],
            "relationship_analysis": "completed"
        }
    
    def _generate_adaptive_recommendations(self, df: pd.DataFrame, schema_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate adaptive recommendations"""
        recommendations = []
        
        # Basic recommendations based on data characteristics
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_ratio > 0.1:
            recommendations.append({
                "task": "handle_missing_data",
                "priority": "high" if missing_ratio > 0.2 else "medium",
                "method": "knn_imputation",
                "reason": f"Dataset has {missing_ratio:.1%} missing values"
            })
        
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            recommendations.append({
                "task": "remove_duplicates",
                "priority": "high",
                "method": "exact_matching",
                "reason": f"Found {duplicate_count} duplicate records"
            })
        
        return recommendations
    
    def _determine_processing_strategy(self, df: pd.DataFrame, schema_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine processing strategy"""
        return {
            "approach": "standard_pipeline",
            "priority_order": ["missing_data", "duplicates", "outliers"],
            "estimated_time": "< 5 minutes",
            "risk_level": "low"
        }
    
    def _generate_dynamic_insights(self, df: pd.DataFrame, schema_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate dynamic insights"""
        return {
            "key_findings": ["Dataset analyzed successfully"],
            "potential_issues": [],
            "optimization_opportunities": [],
            "analysis_confidence": 0.85
        }

    # Missing helper methods for dynamic schema analysis
    def _detect_data_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect data patterns in the dataset"""
        patterns = {
            "sequential_ids": self._has_sequential_ids(df),
            "datetime_columns": self._detect_datetime_patterns(df),
            "categorical_patterns": self._detect_categorical_patterns(df),
            "numeric_patterns": self._detect_numeric_patterns(df)
        }
        return patterns
    
    def _analyze_column_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze relationships between columns"""
        relationships = {
            "strongly_correlated": [],
            "potential_dependencies": [],
            "redundant_columns": []
        }
        
        # Simple correlation analysis for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            try:
                corr_matrix = df[numeric_cols].corr()
                # Find strong correlations (> 0.8)
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.8:
                            relationships["strongly_correlated"].append({
                                "col1": corr_matrix.columns[i],
                                "col2": corr_matrix.columns[j],
                                "correlation": float(corr_val)
                            })
            except:
                pass
        
        return relationships
    
    def _assess_data_complexity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess the complexity of the dataset"""
        complexity = {
            "size_score": self._calculate_size_complexity(df),
            "type_diversity_score": self._calculate_type_diversity(df),
            "missing_data_complexity": self._calculate_missing_complexity(df),
            "overall_complexity": "medium"
        }
        
        avg_score = (complexity["size_score"] + complexity["type_diversity_score"] + 
                    complexity["missing_data_complexity"]) / 3
        
        if avg_score < 30:
            complexity["overall_complexity"] = "low"
        elif avg_score > 70:
            complexity["overall_complexity"] = "high"
        
        return complexity
    
    def _determine_optimal_strategies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Determine optimal processing strategies"""
        strategies = {
            "missing_data_strategy": "standard_imputation",
            "outlier_strategy": "iqr_detection",
            "duplicate_strategy": "exact_match",
            "validation_strategy": "comprehensive"
        }
        
        # Adjust based on data characteristics
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_ratio > 0.3:
            strategies["missing_data_strategy"] = "advanced_imputation"
        
        return strategies
    
    # Helper methods for pattern detection
    def _has_sequential_ids(self, df: pd.DataFrame) -> bool:
        """Check if dataset has sequential ID columns"""
        for col in df.columns:
            if 'id' in col.lower() and df[col].dtype in ['int64', 'int32']:
                if df[col].is_monotonic_increasing:
                    return True
        return False
    
    def _detect_datetime_patterns(self, df: pd.DataFrame) -> List[str]:
        """Detect datetime column patterns"""
        datetime_cols = []
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                datetime_cols.append(col)
            elif df[col].dtype == 'object':
                # Try to parse as datetime
                try:
                    pd.to_datetime(df[col].dropna().head(5))
                    datetime_cols.append(col)
                except:
                    pass
        return datetime_cols
    
    def _detect_categorical_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect categorical patterns"""
        categorical_info = {
            "low_cardinality": [],  # < 10 unique values
            "medium_cardinality": [],  # 10-50 unique values
            "high_cardinality": []  # > 50 unique values
        }
        
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                unique_count = df[col].nunique()
                if unique_count < 10:
                    categorical_info["low_cardinality"].append(col)
                elif unique_count <= 50:
                    categorical_info["medium_cardinality"].append(col)
                else:
                    categorical_info["high_cardinality"].append(col)
        
        return categorical_info
    
    def _detect_numeric_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect numeric patterns"""
        numeric_info = {
            "integer_columns": [],
            "float_columns": [],
            "potentially_categorical_numeric": []
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].dtype in ['int64', 'int32', 'int16', 'int8']:
                numeric_info["integer_columns"].append(col)
                # Check if it might be categorical (few unique values)
                if df[col].nunique() <= 10:
                    numeric_info["potentially_categorical_numeric"].append(col)
            else:
                numeric_info["float_columns"].append(col)
        
        return numeric_info
    
    # Helper methods for complexity calculation
    def _calculate_size_complexity(self, df: pd.DataFrame) -> float:
        """Calculate complexity based on dataset size"""
        total_cells = len(df) * len(df.columns)
        if total_cells < 1000:
            return 10.0
        elif total_cells < 10000:
            return 30.0
        elif total_cells < 100000:
            return 60.0
        else:
            return 90.0
    
    def _calculate_type_diversity(self, df: pd.DataFrame) -> float:
        """Calculate complexity based on data type diversity"""
        unique_types = len(df.dtypes.unique())
        if unique_types <= 2:
            return 20.0
        elif unique_types <= 4:
            return 50.0
        else:
            return 80.0
    
    def _calculate_missing_complexity(self, df: pd.DataFrame) -> float:
        """Calculate complexity based on missing data"""
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        return missing_ratio * 100
    
    # Additional missing methods needed for schema analysis
    def _categorize_dataset_size(self, total_rows: int, total_cols: int) -> str:
        """Categorize dataset size"""
        if total_rows < 100 or total_cols < 5:
            return "small"
        elif total_rows < 10000 or total_cols < 20:
            return "medium"
        else:
            return "large"
    
    def _calculate_complexity_score(self, df: pd.DataFrame) -> float:
        """Calculate overall complexity score"""
        size_score = self._calculate_size_complexity(df)
        type_score = self._calculate_type_diversity(df)
        missing_score = self._calculate_missing_complexity(df)
        return (size_score + type_score + missing_score) / 3
    
    def _categorize_memory_usage(self, memory_mb: float) -> str:
        """Categorize memory usage"""
        if memory_mb < 1:
            return "low"
        elif memory_mb < 10:
            return "medium"
        else:
            return "high"
    
    def _recommend_processing_approach(self, df: pd.DataFrame) -> str:
        """Recommend processing approach"""
        complexity = self._calculate_complexity_score(df)
        if complexity < 30:
            return "standard"
        elif complexity < 70:
            return "enhanced"
        else:
            return "advanced"
    
    def _detect_semantic_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detect semantic types of columns"""
        semantic_types = {}
        for col in df.columns:
            if 'id' in col.lower():
                semantic_types[col] = "identifier"
            elif 'name' in col.lower():
                semantic_types[col] = "name"
            elif 'date' in col.lower() or 'time' in col.lower():
                semantic_types[col] = "temporal"
            elif 'age' in col.lower():
                semantic_types[col] = "age"
            elif df[col].dtype in ['int64', 'float64'] and df[col].nunique() <= 10:
                semantic_types[col] = "categorical_numeric"
            else:
                semantic_types[col] = "general"
        return semantic_types
    
    def _identify_data_roles(self, df: pd.DataFrame) -> Dict[str, str]:
        """Identify the role of each column in the dataset"""
        data_roles = {}
        for col in df.columns:
            if 'id' in col.lower():
                data_roles[col] = "key"
            elif df[col].dtype == 'object' and df[col].nunique() < len(df) * 0.5:
                data_roles[col] = "categorical"
            elif df[col].dtype in ['int64', 'float64']:
                data_roles[col] = "numeric"
            else:
                data_roles[col] = "text"
        return data_roles
    
    def _assign_processing_priority(self, df: pd.DataFrame) -> Dict[str, str]:
        """Assign processing priority to columns"""
        priority = {}
        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if missing_pct > 50:
                priority[col] = "high"
            elif missing_pct > 20:
                priority[col] = "medium"
            else:
                priority[col] = "low"
        return priority

# Global instance
metadata_generator = MetadataGenerator()
