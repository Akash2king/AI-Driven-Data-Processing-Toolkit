import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class MLProcessors:
    def __init__(self):
        self.supported_tasks = {
            "remove_duplicates": self.remove_duplicates,
            "impute_missing": self.impute_missing,
            "detect_outliers": self.detect_outliers,
            "fix_dtypes": self.fix_dtypes,
            "clean_text": self.clean_text_data,
            "normalize_data": self.normalize_numeric_data,
            "encode_categorical": self.encode_categorical_data,
            "handle_surveys": self.handle_survey_specific_cleaning
        }
    
    def get_available_methods(self, task: str) -> List[str]:
        """Get available methods for a specific task"""
        method_map = {
            "remove_duplicates": ["exact", "near_match", "fuzzy_match", "semantic"],
            "impute_missing": ["mean", "median", "mode", "knn", "random_forest", "forward_fill", "backward_fill", "interpolate"],
            "detect_outliers": ["iqr", "zscore", "isolation_forest", "local_outlier_factor", "modified_zscore"],
            "fix_dtypes": ["auto", "manual", "smart_convert"],
            "clean_text": ["standardize", "remove_special", "normalize_case", "remove_stopwords"],
            "normalize_data": ["minmax", "zscore", "robust", "quantile"],
            "encode_categorical": ["label", "onehot", "target", "binary"],
            "handle_surveys": ["weight_analysis", "response_validation", "scale_standardization"]
        }
        return method_map.get(task, [])
    
    def analyze_dataset_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Dynamically analyze dataset structure to suggest optimal cleaning strategies"""
        analysis = {
            "dataset_shape": df.shape,
            "column_analysis": {},
            "data_quality_issues": [],
            "recommended_tasks": [],
            "survey_indicators": {},
            "dynamic_characteristics": self._analyze_dynamic_characteristics(df),
            "adaptive_strategies": self._determine_adaptive_strategies(df),
            "processing_complexity": self._assess_processing_complexity(df)
        }
        
        # Analyze each column dynamically
        for col in df.columns:
            col_analysis = self._analyze_column(df[col], col)
            analysis["column_analysis"][col] = col_analysis
            
            # Identify potential issues dynamically
            if col_analysis["missing_percentage"] > 5:
                analysis["data_quality_issues"].append({
                    "type": "missing_data",
                    "column": col,
                    "severity": "high" if col_analysis["missing_percentage"] > 20 else "medium",
                    "percentage": col_analysis["missing_percentage"]
                })
            
            if col_analysis["is_categorical"] and col_analysis["unique_count"] > 1000:
                analysis["data_quality_issues"].append({
                    "type": "high_cardinality",
                    "column": col,
                    "severity": "medium",
                    "unique_count": col_analysis["unique_count"]
                })
            
            if col_analysis["has_outliers"]:
                analysis["data_quality_issues"].append({
                    "type": "outliers",
                    "column": col,
                    "severity": "low",
                    "outlier_count": col_analysis.get("outlier_count", 0)
                })
        
        # Check for survey-specific patterns
        analysis["survey_indicators"] = self._detect_survey_patterns(df)
        
        # Generate dynamic recommendations
        analysis["recommended_tasks"] = self._generate_dynamic_recommendations(df, analysis)
        
        return analysis
    
    def _analyze_dynamic_characteristics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dynamic characteristics of the dataset"""
        n_rows, n_cols = df.shape
        memory_usage = df.memory_usage(deep=True).sum() / (1024**2)  # MB
        
        characteristics = {
            "size_category": self._categorize_size(n_rows, n_cols),
            "memory_footprint": {
                "size_mb": memory_usage,
                "category": "large" if memory_usage > 100 else "medium" if memory_usage > 10 else "small"
            },
            "data_density": self._calculate_data_density(df),
            "type_diversity": self._analyze_type_diversity(df),
            "complexity_score": self._calculate_complexity_score(df),
            "processing_hints": self._generate_processing_hints(df)
        }
        
        return characteristics
    
    def _categorize_size(self, n_rows: int, n_cols: int) -> str:
        """Categorize dataset size"""
        if n_rows > 100000 or n_cols > 100:
            return "large"
        elif n_rows > 10000 or n_cols > 50:
            return "medium"
        else:
            return "small"
    
    def _calculate_data_density(self, df: pd.DataFrame) -> float:
        """Calculate data density (non-null ratio)"""
        total_cells = df.shape[0] * df.shape[1]
        non_null_cells = df.count().sum()
        return float(non_null_cells / total_cells) if total_cells > 0 else 0.0
    
    def _analyze_type_diversity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze diversity of data types"""
        type_counts = df.dtypes.value_counts().to_dict()
        type_counts = {str(k): int(v) for k, v in type_counts.items()}
        
        return {
            "unique_types": len(type_counts),
            "type_distribution": type_counts,
            "is_homogeneous": len(type_counts) <= 2,
            "dominant_type": max(type_counts, key=type_counts.get) if type_counts else "unknown"
        }
    
    def _calculate_complexity_score(self, df: pd.DataFrame) -> float:
        """Calculate overall complexity score (0-100)"""
        factors = []
        
        # Size factor
        n_rows, n_cols = df.shape
        size_score = min(50, (n_rows / 10000) * 20 + (n_cols / 100) * 30)
        factors.append(size_score)
        
        # Missing data factor
        missing_ratio = df.isnull().sum().sum() / (n_rows * n_cols)
        missing_score = missing_ratio * 30
        factors.append(missing_score)
        
        # Type diversity factor
        type_diversity = len(df.dtypes.unique())
        diversity_score = min(20, type_diversity * 3)
        factors.append(diversity_score)
        
        return float(sum(factors))
    
    def _generate_processing_hints(self, df: pd.DataFrame) -> List[str]:
        """Generate processing hints based on data characteristics"""
        hints = []
        
        n_rows, n_cols = df.shape
        if n_rows > 100000:
            hints.append("Consider batch processing for large dataset")
        
        missing_ratio = df.isnull().sum().sum() / (n_rows * n_cols)
        if missing_ratio > 0.3:
            hints.append("High missing data - consider multiple imputation")
        
        text_cols = df.select_dtypes(include=['object']).columns
        if len(text_cols) > n_cols * 0.5:
            hints.append("Text-heavy dataset - consider text preprocessing")
        
        return hints
    
    def _determine_adaptive_strategies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Determine adaptive cleaning strategies based on data characteristics"""
        strategies = {
            "imputation": self._recommend_imputation_strategy(df),
            "outlier_detection": self._recommend_outlier_strategy(df),
            "duplicate_removal": self._recommend_duplicate_strategy(df),
            "type_conversion": self._recommend_type_strategy(df),
            "validation": self._recommend_validation_strategy(df)
        }
        
        return strategies
    
    def _recommend_imputation_strategy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Recommend optimal imputation strategy"""
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        numeric_ratio = len(df.select_dtypes(include=[np.number]).columns) / len(df.columns)
        
        if missing_ratio < 0.05:
            method = "mean" if numeric_ratio > 0.7 else "mode"
            complexity = "low"
        elif missing_ratio < 0.2:
            method = "knn"
            complexity = "medium"
        else:
            method = "random_forest"
            complexity = "high"
        
        return {
            "recommended_method": method,
            "complexity": complexity,
            "rationale": f"Missing ratio: {missing_ratio:.2%}, Numeric ratio: {numeric_ratio:.2%}"
        }
    
    def _recommend_outlier_strategy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Recommend optimal outlier detection strategy"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {"recommended_method": "none", "rationale": "No numeric columns"}
        
        # Check data distribution characteristics
        sample_col = numeric_cols[0]
        sample_data = df[sample_col].dropna()
        
        if len(sample_data) < 100:
            method = "iqr"
            rationale = "Small sample size"
        elif self._is_survey_like(df):
            method = "modified_zscore"
            rationale = "Survey data detected"
        else:
            method = "isolation_forest"
            rationale = "General purpose detection"
        
        return {
            "recommended_method": method,
            "rationale": rationale,
            "target_columns": list(numeric_cols)
        }
    
    def _recommend_duplicate_strategy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Recommend optimal duplicate removal strategy"""
        text_ratio = len(df.select_dtypes(include=['object']).columns) / len(df.columns)
        
        if text_ratio > 0.5:
            method = "fuzzy_match"
            rationale = "High text content suggests fuzzy matching needed"
        else:
            method = "exact"
            rationale = "Primarily numeric/structured data"
        
        return {
            "recommended_method": method,
            "rationale": rationale,
            "text_ratio": text_ratio
        }
    
    def _recommend_type_strategy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Recommend type conversion strategy"""
        object_cols = df.select_dtypes(include=['object']).columns
        recommendations = []
        
        for col in object_cols:
            sample_values = df[col].dropna().head(100)
            if self._looks_like_numeric(sample_values):
                recommendations.append({
                    "column": col,
                    "current_type": "object",
                    "recommended_type": "numeric",
                    "confidence": "high"
                })
            elif self._looks_like_datetime(sample_values):
                recommendations.append({
                    "column": col,
                    "current_type": "object",
                    "recommended_type": "datetime",
                    "confidence": "medium"
                })
        
        return {
            "conversions": recommendations,
            "total_candidates": len(recommendations)
        }
    
    def _recommend_validation_strategy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Recommend validation strategy"""
        if self._is_survey_like(df):
            return {
                "strategy": "survey_validation",
                "checks": ["response_consistency", "scale_validity", "completion_patterns"]
            }
        elif self._has_financial_data(df):
            return {
                "strategy": "financial_validation", 
                "checks": ["positive_amounts", "currency_consistency", "range_validation"]
            }
        else:
            return {
                "strategy": "general_validation",
                "checks": ["data_types", "range_validation", "consistency_checks"]
            }
    
    def _assess_processing_complexity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall processing complexity"""
        n_rows, n_cols = df.shape
        missing_ratio = df.isnull().sum().sum() / (n_rows * n_cols)
        
        complexity_score = 0
        if n_rows > 100000:
            complexity_score += 30
        if n_cols > 100:
            complexity_score += 20
        if missing_ratio > 0.2:
            complexity_score += 25
        if len(df.dtypes.unique()) > 5:
            complexity_score += 15
        
        if complexity_score > 70:
            level = "high"
            estimated_time = "10-30 minutes"
        elif complexity_score > 40:
            level = "medium" 
            estimated_time = "5-10 minutes"
        else:
            level = "low"
            estimated_time = "1-5 minutes"
        
        return {
            "level": level,
            "score": complexity_score,
            "estimated_time": estimated_time,
            "factors": self._identify_complexity_factors(df)
        }
    
    def _identify_complexity_factors(self, df: pd.DataFrame) -> List[str]:
        """Identify factors contributing to processing complexity"""
        factors = []
        
        n_rows, n_cols = df.shape
        if n_rows > 100000:
            factors.append(f"Large row count: {n_rows:,}")
        if n_cols > 100:
            factors.append(f"High column count: {n_cols}")
        
        missing_ratio = df.isnull().sum().sum() / (n_rows * n_cols)
        if missing_ratio > 0.2:
            factors.append(f"High missing data: {missing_ratio:.1%}")
        
        if len(df.dtypes.unique()) > 5:
            factors.append(f"Diverse data types: {len(df.dtypes.unique())} types")
        
        return factors
    
    def _is_survey_like(self, df: pd.DataFrame) -> bool:
        """Check if dataset appears to be survey data"""
        # Look for Likert scale patterns
        likert_count = 0
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) >= 3 and len(unique_vals) <= 10:
                    # Check if consecutive integers
                    sorted_vals = sorted(unique_vals)
                    if all(isinstance(x, (int, np.integer)) for x in sorted_vals):
                        if sorted_vals == list(range(1, len(sorted_vals) + 1)):
                            likert_count += 1
        
        # Also check column names
        survey_indicators = ['q1', 'q2', 'rating', 'score', 'agree', 'satisfaction']
        name_indicators = sum(1 for col in df.columns if any(ind in col.lower() for ind in survey_indicators))
        
        return likert_count >= 3 or name_indicators >= 5
    
    def _has_financial_data(self, df: pd.DataFrame) -> bool:
        """Check if dataset contains financial data"""
        financial_indicators = ['price', 'cost', 'amount', 'revenue', 'dollar', 'currency']
        return any(ind in col.lower() for col in df.columns for ind in financial_indicators)
    
    def _looks_like_numeric(self, series: pd.Series) -> bool:
        """Check if text column looks like it should be numeric"""
        try:
            # Try to convert a sample to numeric
            sample = series.astype(str).str.replace(',', '').str.replace('$', '')
            pd.to_numeric(sample, errors='raise')
            return True
        except:
            return False
    
    def _looks_like_datetime(self, series: pd.Series) -> bool:
        """Check if text column looks like it should be datetime"""
        try:
            # Try to parse a sample as datetime
            pd.to_datetime(series.head(10), errors='raise')
            return True
        except:
            return False
    
    def remove_duplicates(self, df: pd.DataFrame, method: str = "exact", 
                         columns: Optional[List[str]] = None, 
                         **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Removes duplicates using various methods.
        Returns: (cleaned_dataframe, processing_info)
        """
        original_count = len(df)
        processing_info = {
            "method": method,
            "columns_used": columns,
            "original_count": original_count
        }
        
        try:
            if method == "exact":
                if columns:
                    cleaned_df = df.drop_duplicates(subset=columns, keep='first')
                else:
                    cleaned_df = df.drop_duplicates(keep='first')
            
            elif method == "near_match":
                # For near match, we'll use a simple approach based on string similarity
                # This is a simplified implementation
                cleaned_df = self._remove_near_duplicates(df, columns, kwargs.get('threshold', 0.85))
            
            elif method == "fuzzy_match":
                # Fuzzy matching implementation (simplified)
                cleaned_df = self._remove_fuzzy_duplicates(df, columns, kwargs.get('threshold', 0.8))
            
            else:
                raise ValueError(f"Unknown duplicate removal method: {method}")
            
            duplicates_removed = original_count - len(cleaned_df)
            processing_info.update({
                "duplicates_removed": duplicates_removed,
                "final_count": len(cleaned_df),
                "success": True
            })
            
            return cleaned_df, processing_info
            
        except Exception as e:
            processing_info.update({
                "error": str(e),
                "success": False
            })
            return df, processing_info
    
    def impute_missing(self, df: pd.DataFrame, method: str, 
                      columns: Optional[List[str]] = None, 
                      **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Performs imputation using various methods.
        Returns: (imputed_dataframe, processing_info)
        """
        if columns is None:
            columns = df.columns.tolist()
        
        processing_info = {
            "method": method,
            "columns_processed": columns,
            "original_missing_count": df[columns].isnull().sum().sum()
        }
        
        try:
            df_copy = df.copy()
            
            if method == "mean":
                for col in columns:
                    if pd.api.types.is_numeric_dtype(df_copy[col]):
                        df_copy[col].fillna(df_copy[col].mean(), inplace=True)
            
            elif method == "median":
                for col in columns:
                    if pd.api.types.is_numeric_dtype(df_copy[col]):
                        df_copy[col].fillna(df_copy[col].median(), inplace=True)
            
            elif method == "mode":
                for col in columns:
                    mode_value = df_copy[col].mode()
                    if len(mode_value) > 0:
                        df_copy[col].fillna(mode_value[0], inplace=True)
            
            elif method == "knn":
                df_copy = self._knn_imputation(df_copy, columns, kwargs.get('n_neighbors', 5))
            
            elif method == "random_forest":
                df_copy = self._random_forest_imputation(df_copy, columns)
            
            elif method == "forward_fill":
                df_copy[columns] = df_copy[columns].fillna(method='ffill')
            
            elif method == "backward_fill":
                df_copy[columns] = df_copy[columns].fillna(method='bfill')
            
            else:
                raise ValueError(f"Unknown imputation method: {method}")
            
            final_missing_count = df_copy[columns].isnull().sum().sum()
            processing_info.update({
                "final_missing_count": final_missing_count,
                "values_imputed": processing_info["original_missing_count"] - final_missing_count,
                "success": True
            })
            
            return df_copy, processing_info
            
        except Exception as e:
            processing_info.update({
                "error": str(e),
                "success": False
            })
            return df, processing_info
    
    def detect_outliers(self, df: pd.DataFrame, method: str, 
                       columns: Optional[List[str]] = None, 
                       action: str = "flag", **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Detects and optionally handles outliers.
        Actions: 'flag', 'remove', 'cap'
        Returns: (processed_dataframe, processing_info)
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        processing_info = {
            "method": method,
            "action": action,
            "columns_processed": columns,
            "outliers_detected": {}
        }
        
        try:
            df_copy = df.copy()
            outlier_indices = set()
            
            for col in columns:
                if not pd.api.types.is_numeric_dtype(df_copy[col]):
                    continue
                
                col_outliers = self._detect_column_outliers(df_copy[col], method, **kwargs)
                outlier_indices.update(col_outliers)
                processing_info["outliers_detected"][col] = len(col_outliers)
            
            processing_info["total_outliers"] = len(outlier_indices)
            
            # Apply action
            if action == "flag":
                df_copy['outlier_flag'] = df_copy.index.isin(outlier_indices)
            elif action == "remove":
                df_copy = df_copy.drop(index=list(outlier_indices))
            elif action == "cap":
                df_copy = self._cap_outliers(df_copy, columns, method, **kwargs)
            
            processing_info.update({
                "rows_affected": len(outlier_indices),
                "final_count": len(df_copy),
                "success": True
            })
            
            return df_copy, processing_info
            
        except Exception as e:
            processing_info.update({
                "error": str(e),
                "success": False
            })
            return df, processing_info
    
    def fix_dtypes(self, df: pd.DataFrame, columns: Optional[List[str]] = None, 
                   target_types: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Corrects data type mismatches.
        Returns: (corrected_dataframe, processing_info)
        """
        if columns is None:
            columns = df.columns.tolist()
        
        processing_info = {
            "columns_processed": columns,
            "type_changes": {},
            "errors": {}
        }
        
        try:
            df_copy = df.copy()
            
            for col in columns:
                original_type = str(df_copy[col].dtype)
                
                try:
                    if target_types and col in target_types:
                        target_type = target_types[col]
                    else:
                        # Auto-detect appropriate type
                        target_type = self._detect_optimal_dtype(df_copy[col])
                    
                    if target_type == "numeric":
                        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                    elif target_type == "categorical":
                        df_copy[col] = df_copy[col].astype('category')
                    elif target_type == "datetime":
                        df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                    elif target_type == "string":
                        df_copy[col] = df_copy[col].astype('string')
                    
                    new_type = str(df_copy[col].dtype)
                    if original_type != new_type:
                        processing_info["type_changes"][col] = {
                            "from": original_type,
                            "to": new_type
                        }
                
                except Exception as e:
                    processing_info["errors"][col] = str(e)
            
            processing_info["success"] = True
            return df_copy, processing_info
            
        except Exception as e:
            processing_info.update({
                "error": str(e),
                "success": False
            })
            return df, processing_info
    
    def compute_survey_weighted_stats(self, df: pd.DataFrame, weight_var: str) -> Dict[str, Any]:
        """
        Returns weighted means and proportions for survey data.
        """
        if weight_var not in df.columns:
            raise ValueError(f"Weight variable '{weight_var}' not found in dataset")
        
        weights = df[weight_var]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if weight_var in numeric_cols:
            numeric_cols.remove(weight_var)
        
        weighted_stats = {}
        
        # Weighted means for numeric variables
        for col in numeric_cols:
            valid_mask = ~(df[col].isna() | weights.isna())
            if valid_mask.sum() > 0:
                weighted_mean = np.average(df[col][valid_mask], weights=weights[valid_mask])
                weighted_stats[col] = {
                    "weighted_mean": float(weighted_mean),
                    "valid_n": int(valid_mask.sum())
                }
        
        # Weighted proportions for categorical variables
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols:
            valid_mask = ~(df[col].isna() | weights.isna())
            if valid_mask.sum() > 0:
                col_data = df[col][valid_mask]
                col_weights = weights[valid_mask]
                
                proportions = {}
                for category in col_data.unique():
                    cat_mask = col_data == category
                    if cat_mask.sum() > 0:
                        weighted_prop = col_weights[cat_mask].sum() / col_weights.sum()
                        proportions[str(category)] = float(weighted_prop)
                
                weighted_stats[col] = {
                    "weighted_proportions": proportions,
                    "valid_n": int(valid_mask.sum())
                }
        
        return weighted_stats
    
    def compute_design_effect(self, df: pd.DataFrame, weight_var: str) -> Dict[str, float]:
        """
        Calculates design effect for survey weights.
        """
        if weight_var not in df.columns:
            raise ValueError(f"Weight variable '{weight_var}' not found in dataset")
        
        weights = df[weight_var].dropna()
        
        if len(weights) == 0:
            return {"design_effect": 0.0, "effective_sample_size": 0.0}
        
        # Design effect calculation
        n = len(weights)
        sum_w = weights.sum()
        sum_w_squared = (weights ** 2).sum()
        
        design_effect = (sum_w ** 2) / (n * sum_w_squared)
        effective_sample_size = n * design_effect
        
        return {
            "design_effect": float(design_effect),
            "effective_sample_size": float(effective_sample_size),
            "actual_sample_size": int(n)
        }
    
    def _remove_near_duplicates(self, df: pd.DataFrame, columns: List[str], 
                               threshold: float) -> pd.DataFrame:
        """Remove near-duplicate records based on string similarity"""
        # Simplified implementation - in practice, you'd use more sophisticated methods
        if not columns:
            return df.drop_duplicates()
        
        # For string columns, compare similarity
        text_columns = [col for col in columns if df[col].dtype == 'object']
        if not text_columns:
            return df.drop_duplicates(subset=columns)
        
        # This is a simplified approach - would need proper fuzzy matching library
        return df.drop_duplicates(subset=columns)
    
    def _remove_fuzzy_duplicates(self, df: pd.DataFrame, columns: List[str], 
                                threshold: float) -> pd.DataFrame:
        """Remove fuzzy duplicate records"""
        # Simplified implementation
        return self._remove_near_duplicates(df, columns, threshold)
    
    def _knn_imputation(self, df: pd.DataFrame, columns: List[str], 
                       n_neighbors: int) -> pd.DataFrame:
        """KNN imputation for missing values"""
        df_copy = df.copy()
        numeric_columns = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
        
        if len(numeric_columns) > 1:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df_copy[numeric_columns] = imputer.fit_transform(df_copy[numeric_columns])
        
        return df_copy
    
    def _random_forest_imputation(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Random Forest imputation for missing values"""
        df_copy = df.copy()
        
        for col in columns:
            if df_copy[col].isnull().any():
                # Separate features and target
                feature_cols = [c for c in df_copy.select_dtypes(include=[np.number]).columns 
                               if c != col and not df_copy[c].isnull().all()]
                
                if len(feature_cols) > 0:
                    # Get complete cases for training
                    complete_mask = ~df_copy[col].isnull()
                    missing_mask = df_copy[col].isnull()
                    
                    if complete_mask.sum() > 10:  # Need enough training data
                        X_train = df_copy.loc[complete_mask, feature_cols]
                        y_train = df_copy.loc[complete_mask, col]
                        X_missing = df_copy.loc[missing_mask, feature_cols]
                        
                        # Choose regressor or classifier based on target type
                        if pd.api.types.is_numeric_dtype(df_copy[col]):
                            model = RandomForestRegressor(n_estimators=10, random_state=42)
                        else:
                            model = RandomForestClassifier(n_estimators=10, random_state=42)
                            le = LabelEncoder()
                            y_train = le.fit_transform(y_train.astype(str))
                        
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_missing)
                        
                        if not pd.api.types.is_numeric_dtype(df_copy[col]):
                            predictions = le.inverse_transform(predictions)
                        
                        df_copy.loc[missing_mask, col] = predictions
        
        return df_copy
    
    def _detect_column_outliers(self, series: pd.Series, method: str, **kwargs) -> List[int]:
        """Detect outliers in a single column"""
        outlier_indices = []
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return outlier_indices
        
        if method == "iqr":
            Q1 = clean_series.quantile(0.25)
            Q3 = clean_series.quantile(0.75)
            IQR = Q3 - Q1
            multiplier = kwargs.get('iqr_multiplier', 1.5)
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            outlier_mask = (series < lower_bound) | (series > upper_bound)
            outlier_indices = series[outlier_mask].index.tolist()
        
        elif method == "zscore":
            z_threshold = kwargs.get('z_threshold', 3)
            z_scores = np.abs(stats.zscore(clean_series))
            outlier_indices = clean_series[z_scores > z_threshold].index.tolist()
        
        elif method == "isolation_forest":
            if len(clean_series) > 10:
                clf = IsolationForest(contamination=kwargs.get('contamination', 0.1), 
                                     random_state=42)
                outlier_pred = clf.fit_predict(clean_series.values.reshape(-1, 1))
                outlier_indices = clean_series[outlier_pred == -1].index.tolist()
        
        elif method == "local_outlier_factor":
            if len(clean_series) > 20:
                clf = LocalOutlierFactor(n_neighbors=min(20, len(clean_series)-1))
                outlier_pred = clf.fit_predict(clean_series.values.reshape(-1, 1))
                outlier_indices = clean_series[outlier_pred == -1].index.tolist()
        
        return outlier_indices
    
    def _cap_outliers(self, df: pd.DataFrame, columns: List[str], 
                     method: str, **kwargs) -> pd.DataFrame:
        """Cap outliers at percentile boundaries"""
        df_copy = df.copy()
        
        for col in columns:
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                lower_percentile = kwargs.get('lower_percentile', 5)
                upper_percentile = kwargs.get('upper_percentile', 95)
                
                lower_bound = df_copy[col].quantile(lower_percentile / 100)
                upper_bound = df_copy[col].quantile(upper_percentile / 100)
                
                df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df_copy
    
    def clean_text_data(self, df: pd.DataFrame, method: str = "standardize",
                       columns: Optional[List[str]] = None, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Clean and standardize text data"""
        if columns is None:
            columns = df.select_dtypes(include=['object']).columns.tolist()
        
        processing_info = {
            "method": method,
            "columns_processed": columns,
            "changes_made": {}
        }
        
        try:
            df_copy = df.copy()
            
            for col in columns:
                if df_copy[col].dtype == 'object':
                    original_unique = df_copy[col].nunique()
                    
                    if method == "standardize":
                        df_copy[col] = df_copy[col].astype(str).str.strip().str.lower()
                    elif method == "remove_special":
                        df_copy[col] = df_copy[col].astype(str).str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
                    elif method == "normalize_case":
                        df_copy[col] = df_copy[col].astype(str).str.title()
                    elif method == "remove_stopwords":
                        # Basic stopword removal (can be enhanced with NLTK)
                        stopwords = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
                        df_copy[col] = df_copy[col].astype(str).apply(
                            lambda x: ' '.join([word for word in x.split() if word.lower() not in stopwords])
                        )
                    
                    new_unique = df_copy[col].nunique()
                    processing_info["changes_made"][col] = {
                        "unique_before": original_unique,
                        "unique_after": new_unique,
                        "standardization_effect": original_unique - new_unique
                    }
            
            processing_info["success"] = True
            return df_copy, processing_info
            
        except Exception as e:
            processing_info.update({"error": str(e), "success": False})
            return df, processing_info
    
    def normalize_numeric_data(self, df: pd.DataFrame, method: str = "minmax",
                              columns: Optional[List[str]] = None, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Normalize numeric data using various methods"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        processing_info = {
            "method": method,
            "columns_processed": columns,
            "scaling_parameters": {}
        }
        
        try:
            df_copy = df.copy()
            
            for col in columns:
                if pd.api.types.is_numeric_dtype(df_copy[col]):
                    col_data = df_copy[col].dropna()
                    
                    if method == "minmax":
                        min_val, max_val = col_data.min(), col_data.max()
                        df_copy[col] = (df_copy[col] - min_val) / (max_val - min_val)
                        processing_info["scaling_parameters"][col] = {"min": min_val, "max": max_val}
                    
                    elif method == "zscore":
                        mean_val, std_val = col_data.mean(), col_data.std()
                        df_copy[col] = (df_copy[col] - mean_val) / std_val
                        processing_info["scaling_parameters"][col] = {"mean": mean_val, "std": std_val}
                    
                    elif method == "robust":
                        median_val = col_data.median()
                        mad = np.median(np.abs(col_data - median_val))
                        df_copy[col] = (df_copy[col] - median_val) / mad
                        processing_info["scaling_parameters"][col] = {"median": median_val, "mad": mad}
                    
                    elif method == "quantile":
                        q25, q75 = col_data.quantile(0.25), col_data.quantile(0.75)
                        df_copy[col] = (df_copy[col] - q25) / (q75 - q25)
                        processing_info["scaling_parameters"][col] = {"q25": q25, "q75": q75}
            
            processing_info["success"] = True
            return df_copy, processing_info
            
        except Exception as e:
            processing_info.update({"error": str(e), "success": False})
            return df, processing_info
    
    def encode_categorical_data(self, df: pd.DataFrame, method: str = "label",
                               columns: Optional[List[str]] = None, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Encode categorical data using various methods"""
        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        processing_info = {
            "method": method,
            "columns_processed": columns,
            "encoding_mappings": {}
        }
        
        try:
            df_copy = df.copy()
            
            for col in columns:
                if df_copy[col].dtype in ['object', 'category']:
                    if method == "label":
                        le = LabelEncoder()
                        df_copy[col + '_encoded'] = le.fit_transform(df_copy[col].astype(str).fillna('missing'))
                        processing_info["encoding_mappings"][col] = dict(zip(le.classes_, le.transform(le.classes_)))
                    
                    elif method == "onehot":
                        dummies = pd.get_dummies(df_copy[col], prefix=col)
                        df_copy = pd.concat([df_copy, dummies], axis=1)
                        processing_info["encoding_mappings"][col] = dummies.columns.tolist()
                    
                    elif method == "binary":
                        # Simple binary encoding for high cardinality
                        unique_vals = df_copy[col].unique()
                        n_bits = int(np.ceil(np.log2(len(unique_vals))))
                        for i in range(n_bits):
                            df_copy[f"{col}_bin_{i}"] = 0
                        # This is a simplified binary encoding
                        processing_info["encoding_mappings"][col] = f"{n_bits} binary columns created"
            
            processing_info["success"] = True
            return df_copy, processing_info
            
        except Exception as e:
            processing_info.update({"error": str(e), "success": False})
            return df, processing_info
    
    def handle_survey_specific_cleaning(self, df: pd.DataFrame, method: str = "weight_analysis",
                                       columns: Optional[List[str]] = None, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle survey-specific data cleaning tasks"""
        processing_info = {
            "method": method,
            "survey_analysis": {},
            "cleaning_applied": []
        }
        
        try:
            df_copy = df.copy()
            
            if method == "weight_analysis":
                weight_vars = self._detect_weight_variables(df_copy)
                processing_info["survey_analysis"]["detected_weights"] = weight_vars
                
                for weight_var in weight_vars:
                    if weight_var in df_copy.columns:
                        weight_stats = self.compute_design_effect(df_copy, weight_var)
                        processing_info["survey_analysis"][f"{weight_var}_stats"] = weight_stats
            
            elif method == "response_validation":
                # Validate survey responses for consistency
                response_patterns = {}
                for col in df_copy.columns:
                    if self._is_likert_scale(df_copy[col]):
                        response_patterns[col] = self._analyze_response_pattern(df_copy[col])
                processing_info["survey_analysis"]["response_patterns"] = response_patterns
            
            elif method == "scale_standardization":
                # Standardize different scale formats
                for col in df_copy.columns:
                    if self._is_scale_column(df_copy[col]):
                        df_copy[col] = self._standardize_scale(df_copy[col])
                        processing_info["cleaning_applied"].append(f"Standardized scale: {col}")
            
            processing_info["success"] = True
            return df_copy, processing_info
            
        except Exception as e:
            processing_info.update({"error": str(e), "success": False})
            return df, processing_info
    
    def _analyze_column(self, series: pd.Series, col_name: str) -> Dict[str, Any]:
        """Dynamically analyze a single column"""
        analysis = {
            "name": col_name,
            "dtype": str(series.dtype),
            "count": len(series),
            "non_null_count": series.count(),
            "null_count": series.isnull().sum(),
            "missing_percentage": (series.isnull().sum() / len(series)) * 100,
            "unique_count": series.nunique(),
            "is_numeric": pd.api.types.is_numeric_dtype(series),
            "is_categorical": self._is_categorical(series),
            "is_datetime": pd.api.types.is_datetime64_any_dtype(series),
            "has_outliers": False,
            "potential_issues": []
        }
        
        # Numeric column analysis
        if analysis["is_numeric"]:
            clean_series = series.dropna()
            if len(clean_series) > 0:
                analysis.update({
                    "mean": float(clean_series.mean()),
                    "median": float(clean_series.median()),
                    "std": float(clean_series.std()),
                    "min": float(clean_series.min()),
                    "max": float(clean_series.max()),
                    "skewness": float(clean_series.skew()),
                    "kurtosis": float(clean_series.kurtosis())
                })
                
                # Check for outliers using IQR
                Q1, Q3 = clean_series.quantile(0.25), clean_series.quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((clean_series < Q1 - 1.5 * IQR) | (clean_series > Q3 + 1.5 * IQR)).sum()
                analysis["has_outliers"] = outliers > 0
                analysis["outlier_count"] = int(outliers)
        
        # Categorical column analysis
        elif analysis["is_categorical"]:
            value_counts = series.value_counts().head(10)
            analysis["top_values"] = value_counts.to_dict()
            analysis["cardinality"] = series.nunique()
            
            if analysis["cardinality"] > len(series) * 0.9:
                analysis["potential_issues"].append("Very high cardinality - might be identifier")
        
        # Text data analysis
        if series.dtype == 'object':
            text_lengths = series.astype(str).str.len()
            analysis["avg_text_length"] = float(text_lengths.mean())
            analysis["max_text_length"] = int(text_lengths.max())
            analysis["contains_special_chars"] = series.astype(str).str.contains(r'[^a-zA-Z0-9\s]').any()
        
        return analysis
    
    def _detect_survey_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect survey-specific patterns in the dataset"""
        patterns = {
            "likert_scales": [],
            "weight_variables": [],
            "demographic_vars": [],
            "response_codes": [],
            "survey_metadata": {}
        }
        
        for col in df.columns:
            # Detect Likert scales
            if self._is_likert_scale(df[col]):
                patterns["likert_scales"].append(col)
            
            # Detect weight variables
            if self._is_weight_variable(col, df[col]):
                patterns["weight_variables"].append(col)
            
            # Detect demographic variables
            if self._is_demographic_variable(col, df[col]):
                patterns["demographic_vars"].append(col)
            
            # Detect response codes
            if self._has_response_codes(df[col]):
                patterns["response_codes"].append(col)
        
        return patterns
    
    def _generate_dynamic_recommendations(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate dynamic cleaning recommendations based on dataset analysis"""
        recommendations = []
        
        # Missing value recommendations
        for col, col_analysis in analysis["column_analysis"].items():
            if col_analysis["missing_percentage"] > 5:
                if col_analysis["is_numeric"]:
                    recommendations.append({
                        "task": "impute_missing",
                        "method": "median" if col_analysis.get("has_outliers") else "mean",
                        "columns": [col],
                        "priority": "high" if col_analysis["missing_percentage"] > 20 else "medium",
                        "reason": f"High missing values ({col_analysis['missing_percentage']:.1f}%) in numeric column"
                    })
                elif col_analysis["is_categorical"]:
                    recommendations.append({
                        "task": "impute_missing",
                        "method": "mode",
                        "columns": [col],
                        "priority": "medium",
                        "reason": f"Missing values in categorical column ({col_analysis['missing_percentage']:.1f}%)"
                    })
        
        # Outlier recommendations
        for col, col_analysis in analysis["column_analysis"].items():
            if col_analysis.get("has_outliers", False):
                recommendations.append({
                    "task": "detect_outliers",
                    "method": "iqr",
                    "columns": [col],
                    "priority": "medium",
                    "reason": f"Outliers detected in {col} ({col_analysis.get('outlier_count', 0)} values)"
                })
        
        # Duplicate recommendations
        if df.duplicated().sum() > 0:
            recommendations.append({
                "task": "remove_duplicates",
                "method": "exact",
                "columns": None,
                "priority": "high",
                "reason": f"Found {df.duplicated().sum()} duplicate rows"
            })
        
        # Data type recommendations
        for col, col_analysis in analysis["column_analysis"].items():
            if col_analysis["dtype"] == "object" and col_analysis["unique_count"] < 20:
                recommendations.append({
                    "task": "fix_dtypes",
                    "method": "auto",
                    "columns": [col],
                    "priority": "low",
                    "reason": f"Low cardinality text column might be categorical: {col}"
                })
        
        # Survey-specific recommendations
        survey_indicators = analysis.get("survey_indicators", {})
        if survey_indicators.get("weight_variables"):
            recommendations.append({
                "task": "handle_surveys",
                "method": "weight_analysis",
                "columns": survey_indicators["weight_variables"],
                "priority": "medium",
                "reason": "Survey weight variables detected - analyze for proper weighting"
            })
        
        if survey_indicators.get("likert_scales"):
            recommendations.append({
                "task": "handle_surveys",
                "method": "scale_standardization",
                "columns": survey_indicators["likert_scales"],
                "priority": "low",
                "reason": "Likert scales detected - consider standardization"
            })
        
        return sorted(recommendations, key=lambda x: {"high": 3, "medium": 2, "low": 1}[x["priority"]], reverse=True)
    
    def _is_likert_scale(self, series: pd.Series) -> bool:
        """Check if a column contains Likert scale responses"""
        if not pd.api.types.is_numeric_dtype(series):
            return False
        
        unique_vals = series.dropna().unique()
        if len(unique_vals) < 3 or len(unique_vals) > 10:
            return False
        
        # Check if values are consecutive integers
        sorted_vals = sorted(unique_vals)
        return all(isinstance(x, (int, np.integer)) for x in sorted_vals) and \
               all(sorted_vals[i] == sorted_vals[i-1] + 1 for i in range(1, len(sorted_vals)))
    
    def _is_weight_variable(self, col_name: str, series: pd.Series) -> bool:
        """Check if a column is likely a weight variable"""
        col_name_lower = col_name.lower()
        weight_keywords = ['weight', 'wt', 'wgt', 'pond', 'factor', 'expansion']
        
        if any(keyword in col_name_lower for keyword in weight_keywords):
            return True
        
        if pd.api.types.is_numeric_dtype(series):
            clean_series = series.dropna()
            if len(clean_series) > 0:
                mean_val = clean_series.mean()
                return 0.1 <= mean_val <= 10 and clean_series.min() > 0
        
        return False
    
    def _is_demographic_variable(self, col_name: str, series: pd.Series) -> bool:
        """Check if a column contains demographic information"""
        demo_keywords = ['age', 'gender', 'sex', 'race', 'ethnicity', 'education', 'income', 'marital']
        col_name_lower = col_name.lower()
        
        return any(keyword in col_name_lower for keyword in demo_keywords)
    
    def _has_response_codes(self, series: pd.Series) -> bool:
        """Check if a column contains coded responses (like 1=Yes, 2=No)"""
        if series.dtype == 'object':
            unique_vals = series.dropna().unique()
            return len(unique_vals) < 10 and any(str(val).isdigit() for val in unique_vals)
        return False
    
    def _is_scale_column(self, series: pd.Series) -> bool:
        """Check if a column represents a measurement scale"""
        return self._is_likert_scale(series) or pd.api.types.is_numeric_dtype(series)
    
    def _standardize_scale(self, series: pd.Series) -> pd.Series:
        """Standardize scale values to a common format"""
        if pd.api.types.is_numeric_dtype(series):
            # Normalize to 0-1 scale
            min_val, max_val = series.min(), series.max()
            if max_val > min_val:
                return (series - min_val) / (max_val - min_val)
        return series
    
    def _analyze_response_pattern(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze response patterns in survey data"""
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return {}
        
        value_counts = clean_series.value_counts()
        return {
            "response_distribution": value_counts.to_dict(),
            "most_common_response": value_counts.index[0],
            "response_variance": float(clean_series.var()) if pd.api.types.is_numeric_dtype(clean_series) else None,
            "potential_straight_lining": self._detect_straight_lining(clean_series)
        }
    
    def _detect_straight_lining(self, series: pd.Series) -> bool:
        """Detect if responses show straight-lining pattern"""
        if not pd.api.types.is_numeric_dtype(series):
            return False
        
        # Simple straight-lining detection
        value_counts = series.value_counts()
        if len(value_counts) == 1:
            return True  # All same response
        
        # Check if one response dominates (>80%)
        return value_counts.iloc[0] / len(series) > 0.8

    def _detect_optimal_dtype(self, series: pd.Series) -> str:
        """Detect the optimal data type for a series"""
        # Try to convert to numeric
        try:
            numeric_series = pd.to_numeric(series, errors='coerce')
            if not numeric_series.isnull().all():
                return "numeric"
        except:
            pass
        
        # Try to convert to datetime
        try:
            pd.to_datetime(series, errors='raise')
            return "datetime"
        except:
            pass
        
        # Check if should be categorical
        if series.dtype == 'object':
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.1:  # Less than 10% unique values
                return "categorical"
        
        return "string"

# Global instance
ml_processors = MLProcessors()
