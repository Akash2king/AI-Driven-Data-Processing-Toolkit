import pandas as pd
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
import json
import os
from app.core.config import settings

class ReportGenerator:
    def __init__(self):
        self.reports_dir = Path(settings.UPLOAD_DIR) / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_clean_report(self, history: List[Dict[str, Any]], 
                            final_df: pd.DataFrame,
                            original_metadata: Dict[str, Any],
                            final_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Outputs comprehensive cleaning report with before/after stats and step summary.
        Returns report metadata and file paths.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_id = f"cleaning_report_{timestamp}"
        
        # Generate report content
        report_content = self._generate_report_content(
            history, final_df, original_metadata, final_metadata
        )
        
        # Save as JSON
        json_path = self.reports_dir / f"{report_id}.json"
        with open(json_path, 'w') as f:
            json.dump(report_content, f, indent=2, default=str)
        
        # Generate HTML report
        html_path = self.reports_dir / f"{report_id}.html"
        html_content = self._generate_html_report(report_content)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return {
            "report_id": report_id,
            "json_path": str(json_path),
            "html_path": str(html_path),
            "summary": report_content["summary"],
            "timestamp": timestamp
        }
    
    def export_clean_data(self, df: pd.DataFrame, format: str = "csv", 
                         filename: str = None) -> Dict[str, Any]:
        """
        Exports cleaned dataset to specified format.
        Supported formats: csv, excel, json
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cleaned_data_{timestamp}"
        
        export_info = {
            "filename": filename,
            "format": format,
            "rows": len(df),
            "columns": len(df.columns),
            "export_timestamp": datetime.now().isoformat()
        }
        
        try:
            if format.lower() == "csv":
                file_path = self.reports_dir / f"{filename}.csv"
                df.to_csv(file_path, index=False)
                export_info["file_path"] = str(file_path)
            
            elif format.lower() in ["excel", "xlsx"]:
                file_path = self.reports_dir / f"{filename}.xlsx"
                df.to_excel(file_path, index=False, engine='openpyxl')
                export_info["file_path"] = str(file_path)
            
            elif format.lower() == "json":
                file_path = self.reports_dir / f"{filename}.json"
                df.to_json(file_path, orient='records', indent=2)
                export_info["file_path"] = str(file_path)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            export_info["success"] = True
            export_info["file_size_mb"] = os.path.getsize(export_info["file_path"]) / (1024 * 1024)
            
        except Exception as e:
            export_info["success"] = False
            export_info["error"] = str(e)
        
        return export_info
    
    def _generate_report_content(self, history: List[Dict[str, Any]], 
                               final_df: pd.DataFrame,
                               original_metadata: Dict[str, Any],
                               final_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the main report content"""
        
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "tool_version": "1.0.0",
                "total_steps": len(history)
            },
            "summary": self._generate_summary(original_metadata, final_metadata, history),
            "original_dataset": self._format_metadata_for_report(original_metadata),
            "final_dataset": self._format_metadata_for_report(final_metadata),
            "processing_steps": self._format_steps_for_report(history),
            "quality_assessment": self._assess_final_quality(final_metadata, history),
            "recommendations": self._generate_final_recommendations(final_metadata, history),
            "data_lineage": self._generate_data_lineage(history)
        }
        
        return report
    
    def _generate_summary(self, original_meta: Dict[str, Any], 
                         final_meta: Dict[str, Any], 
                         history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate executive summary of the cleaning process"""
        
        orig_basic = original_meta.get("basic_info", {})
        final_basic = final_meta.get("basic_info", {})
        
        # Calculate key metrics
        rows_change = final_basic.get("total_rows", 0) - orig_basic.get("total_rows", 0)
        cols_change = final_basic.get("total_columns", 0) - orig_basic.get("total_columns", 0)
        
        orig_missing = original_meta.get("missing_values", {}).get("total_missing", 0)
        final_missing = final_meta.get("missing_values", {}).get("total_missing", 0)
        missing_reduction = orig_missing - final_missing
        
        orig_duplicates = original_meta.get("duplicates", {}).get("total_duplicates", 0)
        final_duplicates = final_meta.get("duplicates", {}).get("total_duplicates", 0)
        duplicates_removed = orig_duplicates - final_duplicates
        
        # Quality scores
        orig_quality = original_meta.get("data_quality", {}).get("overall_score", 0)
        final_quality = final_meta.get("data_quality", {}).get("overall_score", 0)
        quality_improvement = final_quality - orig_quality
        
        return {
            "dataset_changes": {
                "rows_before": orig_basic.get("total_rows", 0),
                "rows_after": final_basic.get("total_rows", 0),
                "rows_change": rows_change,
                "columns_before": orig_basic.get("total_columns", 0),
                "columns_after": final_basic.get("total_columns", 0),
                "columns_change": cols_change
            },
            "quality_improvements": {
                "missing_values_filled": missing_reduction,
                "duplicates_removed": duplicates_removed,
                "quality_score_before": round(orig_quality, 2),
                "quality_score_after": round(final_quality, 2),
                "quality_improvement": round(quality_improvement, 2)
            },
            "processing_summary": {
                "total_steps_executed": len(history),
                "successful_steps": len([s for s in history if s.get("status") == "completed"]),
                "failed_steps": len([s for s in history if s.get("status") == "failed"]),
                "llm_verified_steps": len([s for s in history if s.get("verification_status") == "confirmed"])
            },
            "data_preservation": {
                "data_retention_percentage": round((final_basic.get("total_rows", 0) / 
                                                  max(orig_basic.get("total_rows", 0), 1)) * 100, 2),
                "is_acceptable": rows_change >= -0.1 * orig_basic.get("total_rows", 0)  # Less than 10% loss
            }
        }
    
    def _format_metadata_for_report(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Format metadata for inclusion in report"""
        return {
            "basic_info": metadata.get("basic_info", {}),
            "data_quality_score": metadata.get("data_quality", {}),
            "missing_values_summary": {
                "total_missing": metadata.get("missing_values", {}).get("total_missing", 0),
                "columns_with_missing": len(metadata.get("missing_values", {}).get("columns_with_missing", {})),
                "worst_columns": dict(list(metadata.get("missing_values", {}).get("missing_percentages", {}).items())[:5])
            },
            "duplicates_summary": metadata.get("duplicates", {}),
            "potential_weight_variables": metadata.get("potential_weights", [])
        }
    
    def _format_steps_for_report(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format processing steps for report"""
        formatted_steps = []
        
        for i, step in enumerate(history, 1):
            formatted_step = {
                "step_number": i,
                "task": step.get("task", "unknown"),
                "method": step.get("method", "unknown"),
                "columns_affected": step.get("columns", []),
                "execution_status": step.get("status", "unknown"),
                "verification_status": step.get("verification_status", "not_verified"),
                "execution_time_seconds": step.get("execution_time", 0),
                "llm_reasoning": step.get("llm_reasoning", ""),
                "verification_reason": step.get("verification_reason", ""),
                "user_approved": step.get("user_approved", False)
            }
            
            # Add before/after metrics if available
            if "pre_metadata" in step and "post_metadata" in step:
                pre_meta = step["pre_metadata"]
                post_meta = step["post_metadata"]
                
                formatted_step["impact"] = {
                    "rows_before": pre_meta.get("basic_info", {}).get("total_rows", 0),
                    "rows_after": post_meta.get("basic_info", {}).get("total_rows", 0),
                    "missing_values_before": pre_meta.get("missing_values", {}).get("total_missing", 0),
                    "missing_values_after": post_meta.get("missing_values", {}).get("total_missing", 0),
                    "duplicates_before": pre_meta.get("duplicates", {}).get("total_duplicates", 0),
                    "duplicates_after": post_meta.get("duplicates", {}).get("total_duplicates", 0)
                }
            
            formatted_steps.append(formatted_step)
        
        return formatted_steps
    
    def _assess_final_quality(self, final_metadata: Dict[str, Any], 
                            history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess the final data quality"""
        quality_score = final_metadata.get("data_quality", {}).get("overall_score", 0)
        
        assessment = {
            "overall_score": round(quality_score, 2),
            "grade": self._get_quality_grade(quality_score),
            "strengths": [],
            "weaknesses": [],
            "ready_for_analysis": quality_score >= 80
        }
        
        # Assess strengths
        missing_pct = (final_metadata.get("missing_values", {}).get("total_missing", 0) / 
                      max(final_metadata.get("basic_info", {}).get("total_rows", 1) * 
                          final_metadata.get("basic_info", {}).get("total_columns", 1), 1)) * 100
        
        if missing_pct < 5:
            assessment["strengths"].append("Low missing values (<5%)")
        
        if final_metadata.get("duplicates", {}).get("total_duplicates", 0) == 0:
            assessment["strengths"].append("No duplicate records")
        
        # Assess weaknesses
        if missing_pct > 15:
            assessment["weaknesses"].append(f"High missing values ({missing_pct:.1f}%)")
        
        if quality_score < 70:
            assessment["weaknesses"].append("Overall data quality below recommended threshold")
        
        return assessment
    
    def _generate_final_recommendations(self, final_metadata: Dict[str, Any], 
                                      history: List[Dict[str, Any]]) -> List[str]:
        """Generate final recommendations based on the cleaned dataset"""
        recommendations = []
        
        # Quality-based recommendations
        quality_score = final_metadata.get("data_quality", {}).get("overall_score", 0)
        if quality_score < 80:
            recommendations.append("Consider additional cleaning steps to improve data quality")
        
        # Missing values recommendations
        missing_cols = final_metadata.get("missing_values", {}).get("columns_with_missing", {})
        if missing_cols:
            high_missing_cols = [col for col, count in missing_cols.items() if count > 
                               final_metadata.get("basic_info", {}).get("total_rows", 0) * 0.1]
            if high_missing_cols:
                recommendations.append(f"Consider excluding or further imputing high-missing columns: {high_missing_cols}")
        
        # Weight variables recommendations
        weight_vars = final_metadata.get("potential_weights", [])
        if weight_vars:
            recommendations.append(f"Consider using survey weights for analysis: {weight_vars}")
        
        # Analysis readiness
        if quality_score >= 80:
            recommendations.append("Dataset is ready for statistical analysis")
            recommendations.append("Document any exclusions or imputations in your analysis methodology")
        
        return recommendations
    
    def _generate_data_lineage(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate data lineage tracking"""
        lineage = {
            "processing_pipeline": [],
            "data_transformations": {},
            "quality_checkpoints": []
        }
        
        for step in history:
            lineage["processing_pipeline"].append({
                "step": step.get("task", "unknown"),
                "method": step.get("method", "unknown"),
                "timestamp": step.get("execution_timestamp", ""),
                "verified": step.get("verification_status") == "confirmed"
            })
            
            if step.get("verification_status") == "confirmed":
                lineage["quality_checkpoints"].append({
                    "step": step.get("task", "unknown"),
                    "verification_reason": step.get("verification_reason", "")
                })
        
        return lineage
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def _generate_html_report(self, report_content: Dict[str, Any]) -> str:
        """Generate HTML version of the report"""
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Survey Data Cleaning Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 30px 0; padding: 20px; border-left: 4px solid #3498db; background-color: #f8f9fa; }}
        .summary-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
        .metric-box {{ background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .step-list {{ background: white; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .quality-grade {{ font-size: 2em; font-weight: bold; color: {grade_color}; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .success {{ color: #27ae60; }}
        .warning {{ color: #f39c12; }}
        .error {{ color: #e74c3c; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Survey Data Cleaning Report</h1>
        <p>Generated on: {timestamp}</p>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        <div class="summary-grid">
            <div class="metric-box">
                <h3>Dataset Changes</h3>
                <p><strong>Rows:</strong> {rows_before} → {rows_after} ({rows_change:+d})</p>
                <p><strong>Columns:</strong> {columns_before} → {columns_after} ({columns_change:+d})</p>
                <p><strong>Data Retention:</strong> {data_retention:.1f}%</p>
            </div>
            <div class="metric-box">
                <h3>Quality Improvements</h3>
                <p><strong>Missing Values Filled:</strong> {missing_filled}</p>
                <p><strong>Duplicates Removed:</strong> {duplicates_removed}</p>
                <p><strong>Quality Score:</strong> {quality_before:.1f} → {quality_after:.1f}</p>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Final Quality Assessment</h2>
        <div class="quality-grade">Grade: {quality_grade}</div>
        <p><strong>Overall Score:</strong> {quality_score:.1f}/100</p>
        <p><strong>Ready for Analysis:</strong> {ready_for_analysis}</p>
        
        <h4>Strengths:</h4>
        <ul>
            {strengths_list}
        </ul>
        
        <h4>Areas for Attention:</h4>
        <ul>
            {weaknesses_list}
        </ul>
    </div>
    
    <div class="section">
        <h2>Processing Steps</h2>
        {steps_table}
    </div>
    
    <div class="section">
        <h2>Final Recommendations</h2>
        <ul>
            {recommendations_list}
        </ul>
    </div>
</body>
</html>
        """
        
        # Extract data for template
        summary = report_content["summary"]
        quality = report_content["quality_assessment"]
        
        # Format grade color
        grade_colors = {"A": "#27ae60", "B": "#2ecc71", "C": "#f39c12", "D": "#e67e22", "F": "#e74c3c"}
        grade_color = grade_colors.get(quality["grade"], "#7f8c8d")
        
        # Format lists
        strengths_list = "".join([f"<li>{s}</li>" for s in quality["strengths"]])
        weaknesses_list = "".join([f"<li>{w}</li>" for w in quality["weaknesses"]])
        recommendations_list = "".join([f"<li>{r}</li>" for r in report_content["recommendations"]])
        
        # Format steps table
        steps_html = "<table><tr><th>Step</th><th>Task</th><th>Method</th><th>Status</th><th>Verified</th></tr>"
        for step in report_content["processing_steps"]:
            status_class = "success" if step["execution_status"] == "completed" else "error"
            verified_class = "success" if step["verification_status"] == "confirmed" else "warning"
            
            steps_html += f"""
            <tr>
                <td>{step['step_number']}</td>
                <td>{step['task']}</td>
                <td>{step['method']}</td>
                <td class="{status_class}">{step['execution_status']}</td>
                <td class="{verified_class}">{step['verification_status']}</td>
            </tr>
            """
        steps_html += "</table>"
        
        # Fill template
        return html_template.format(
            timestamp=report_content["report_metadata"]["generated_at"],
            rows_before=summary["dataset_changes"]["rows_before"],
            rows_after=summary["dataset_changes"]["rows_after"],
            rows_change=summary["dataset_changes"]["rows_change"],
            columns_before=summary["dataset_changes"]["columns_before"],
            columns_after=summary["dataset_changes"]["columns_after"],
            columns_change=summary["dataset_changes"]["columns_change"],
            data_retention=summary["data_preservation"]["data_retention_percentage"],
            missing_filled=summary["quality_improvements"]["missing_values_filled"],
            duplicates_removed=summary["quality_improvements"]["duplicates_removed"],
            quality_before=summary["quality_improvements"]["quality_score_before"],
            quality_after=summary["quality_improvements"]["quality_score_after"],
            quality_grade=quality["grade"],
            quality_score=quality["overall_score"],
            ready_for_analysis="Yes" if quality["ready_for_analysis"] else "No",
            grade_color=grade_color,
            strengths_list=strengths_list,
            weaknesses_list=weaknesses_list,
            steps_table=steps_html,
            recommendations_list=recommendations_list
        )

# Global instance
report_generator = ReportGenerator()
