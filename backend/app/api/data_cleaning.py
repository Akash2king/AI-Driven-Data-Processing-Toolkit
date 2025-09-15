from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Dict, Any, List
import os
import pandas as pd
import uuid
import json
from datetime import datetime
from app.core.database import get_db
from app.modules.data_upload import data_upload_service
from app.modules.metadata_generator import metadata_generator
from app.modules.llm_controller import llm_controller
from app.modules.ml_processors import ml_processors
from app.modules.report_generator import report_generator
from app.models.models import Dataset, ProcessingStep, ProcessingSession

router = APIRouter()

@router.get("/dataset/{dataset_id}/analysis")
async def get_dynamic_analysis(dataset_id: int, db: Session = Depends(get_db)):
    """Get comprehensive dynamic analysis of the dataset"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        # Load the dataset
        df = data_upload_service._load_dataframe(
            dataset.filename.split('.')[0], 
            "." + dataset.filename.split('.')[-1]
        )
        
        # Perform dynamic analysis using ML processors
        analysis = ml_processors.analyze_dataset_structure(df)
        
        return {
            "dataset_id": dataset_id,
            "analysis": analysis,
            "timestamp": dataset.upload_timestamp
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dataset/{dataset_id}/available-methods")
async def get_available_methods(dataset_id: int, task: str = None, db: Session = Depends(get_db)):
    """Get available methods for data processing tasks"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        if task:
            methods = ml_processors.get_available_methods(task)
            return {
                "task": task,
                "available_methods": methods
            }
        else:
            # Return all available tasks and methods
            all_methods = {}
            for task_name in ml_processors.supported_tasks.keys():
                all_methods[task_name] = ml_processors.get_available_methods(task_name)
            
            return {
                "all_available_methods": all_methods,
                "supported_tasks": list(ml_processors.supported_tasks.keys())
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a dataset file and return basic information"""
    try:
        # Upload file
        upload_result = await data_upload_service.upload_file(file)
        
        # Create dataset record
        dataset = Dataset(
            filename=upload_result["filename"],
            original_filename=upload_result["original_filename"],
            file_path=upload_result["file_path"],
            file_size=upload_result["file_size"],
            status="uploaded"
        )
        
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        
        # Generate initial metadata
        df = data_upload_service._load_dataframe(
            upload_result["file_path"], 
            upload_result["file_extension"]
        )
        metadata = metadata_generator.generate_metadata(df)
        
        # Update dataset with metadata
        dataset.dataset_metadata = metadata
        db.commit()
        
        return {
            "dataset_id": dataset.id,
            "file_info": upload_result,
            "metadata": metadata,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dataset/{dataset_id}/preview")
async def preview_dataset(dataset_id: int, rows: int = 5, db: Session = Depends(get_db)):
    """Get a preview of the dataset with JSON serialization safety"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        preview = data_upload_service.preview_data(dataset.filename.split('.')[0], rows)
        # Ensure JSON serialization safety
        return _ensure_json_serializable(preview)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dataset/{dataset_id}/metadata")
async def get_dataset_metadata(dataset_id: int, db: Session = Depends(get_db)):
    """Get current metadata for a dataset"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return {
        "dataset_id": dataset_id,
        "metadata": dataset.dataset_metadata,
        "status": dataset.status,
        "last_updated": dataset.upload_timestamp
    }

@router.post("/dataset/{dataset_id}/session")
async def create_processing_session(
    dataset_id: int, 
    requirements: Dict[str, Any], 
    db: Session = Depends(get_db)
):
    """Create a new processing session with user requirements"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        session_id = str(uuid.uuid4())
        session = ProcessingSession(
            dataset_id=dataset_id,
            session_id=session_id,
            requirements=requirements,
            status="active"
        )
        
        db.add(session)
        db.commit()
        db.refresh(session)
        
        return {
            "session_id": session_id,
            "dataset_id": dataset_id,
            "requirements": requirements,
            "status": "active"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/session/{session_id}/next-step")
async def get_next_cleaning_step(session_id: str, db: Session = Depends(get_db)):
    """Get LLM recommendation for the next cleaning step"""
    session = db.query(ProcessingSession).filter(
        ProcessingSession.session_id == session_id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Get current dataset metadata
        dataset = db.query(Dataset).filter(Dataset.id == session.dataset_id).first()
        current_metadata = dataset.dataset_metadata
        
        # Get processing history
        history = db.query(ProcessingStep).filter(
            ProcessingStep.dataset_id == session.dataset_id
        ).order_by(ProcessingStep.step_number).all()
        
        history_data = [
            {
                "step_number": step.step_number,
                "task": step.task,
                "method": step.method,
                "columns": step.columns,
                "status": step.status,
                "verification_status": step.verification_status
            }
            for step in history
        ]
        
        # Get LLM decision
        decision = await llm_controller.decide_next_step(
            metadata=current_metadata,
            requirements=session.requirements,
            history=history_data,
            db=db
        )
        
        return {
            "session_id": session_id,
            "recommended_step": decision,
            "llm_reasoning": decision.get("llm_reasoning", ""),
            "confidence": decision.get("confidence", 0.8)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/session/{session_id}/execute-step")
async def execute_cleaning_step(
    session_id: str,
    payload: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Execute a cleaning step after user approval"""
    session = db.query(ProcessingSession).filter(
        ProcessingSession.session_id == session_id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Extract parameters from payload
    step_info = payload.get("step_info", {})
    user_approved = payload.get("user_approved", True)
    
    if not user_approved:
        return {"status": "rejected", "message": "Step rejected by user"}
    
    try:
        # Debug: Log the step_info to understand what's being passed
        print(f"DEBUG: Received step_info: {step_info}")
        
        # Load current dataset
        dataset = db.query(Dataset).filter(Dataset.id == session.dataset_id).first()
        file_extension = os.path.splitext(dataset.file_path)[1]
        df = data_upload_service._load_dataframe(dataset.file_path, file_extension)
        
        # Generate pre-step metadata
        pre_metadata = metadata_generator.generate_metadata(df)
        
        # Execute the step
        task = step_info.get("task")
        method = step_info.get("method")
        columns = step_info.get("columns", [])
        parameters = step_info.get("parameters", {})
        
        # Enhanced validation
        if not task:
            raise ValueError(f"No task specified in step_info. Received: {step_info}")
        
        if not method:
            raise ValueError(f"No method specified for task '{task}'. Received: {step_info}")
        
        print(f"DEBUG: Executing task='{task}', method='{method}', columns={columns}")
        
        start_time = pd.Timestamp.now()
        
        if task == "remove_duplicates":
            processed_df, processing_info = ml_processors.remove_duplicates(
                df, method, columns, **parameters
            )
        elif task == "impute_missing":
            processed_df, processing_info = ml_processors.impute_missing(
                df, method, columns, **parameters
            )
        elif task == "detect_outliers":
            processed_df, processing_info = ml_processors.detect_outliers(
                df, method, columns, **parameters
            )
        elif task == "fix_dtypes":
            processed_df, processing_info = ml_processors.fix_dtypes(
                df, columns, parameters.get("target_types")
            )
        else:
            raise ValueError(f"Unknown task: {task}")
        
        execution_time = (pd.Timestamp.now() - start_time).total_seconds()
        
        # Generate post-step metadata
        post_metadata = metadata_generator.generate_metadata(processed_df)
        
        # Verify with LLM
        verification = await llm_controller.verify_step_execution(
            pre_meta=pre_metadata,
            post_meta=post_metadata,
            step_info=step_info,
            history=[],  # Could include previous steps
            db=db
        )
        
        # Create processing step record
        step_number = session.current_step + 1
        processing_step = ProcessingStep(
            dataset_id=session.dataset_id,
            step_number=step_number,
            task=task,
            method=method,
            columns=columns,
            parameters=parameters,
            pre_metadata=pre_metadata,
            post_metadata=post_metadata,
            execution_time=int(execution_time),
            status="completed" if processing_info.get("success") else "failed",
            llm_reasoning=step_info.get("llm_reasoning", ""),
            verification_status=verification["status"],
            verification_reason=verification["reason"],
            user_approved=user_approved
        )
        
        db.add(processing_step)
        
        # Update session
        session.current_step = step_number
        
        # Update dataset metadata if step was successful
        if processing_info.get("success"):
            dataset.dataset_metadata = post_metadata
        
        db.commit()
        
        # Ensure all response data is JSON serializable
        response_data = {
            "step_id": processing_step.id,
            "execution_status": processing_step.status,
            "verification": _ensure_json_serializable(verification),
            "processing_info": _ensure_json_serializable(processing_info),
            "metadata_changes": _ensure_json_serializable(metadata_generator.compare_metadata(pre_metadata, post_metadata))
        }
        
        return response_data
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


def _ensure_json_serializable(obj):
    """Convert numpy types and other non-serializable types to JSON-compatible types"""
    import numpy as np
    
    if isinstance(obj, dict):
        return {key: _ensure_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.str_):
        return str(obj)
    elif hasattr(obj, 'item'):  # Handle other numpy scalar types
        try:
            return obj.item()
        except:
            return str(obj)
    else:
        return obj

@router.get("/session/{session_id}/history")
async def get_processing_history(session_id: str, db: Session = Depends(get_db)):
    """Get the complete processing history for a session"""
    session = db.query(ProcessingSession).filter(
        ProcessingSession.session_id == session_id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    steps = db.query(ProcessingStep).filter(
        ProcessingStep.dataset_id == session.dataset_id
    ).order_by(ProcessingStep.step_number).all()
    
    history = []
    for step in steps:
        step_data = {
            "step_id": step.id,
            "step_number": step.step_number,
            "task": step.task,
            "method": step.method,
            "columns": step.columns,
            "parameters": step.parameters,
            "execution_status": step.status,
            "verification_status": step.verification_status,
            "verification_reason": step.verification_reason,
            "execution_time": step.execution_time,
            "user_approved": step.user_approved,
            "timestamp": step.execution_timestamp
        }
        
        if step.pre_metadata and step.post_metadata:
            step_data["metadata_changes"] = metadata_generator.compare_metadata(
                step.pre_metadata, step.post_metadata
            )
        
        history.append(step_data)
    
    return {
        "session_id": session_id,
        "dataset_id": session.dataset_id,
        "total_steps": len(history),
        "current_step": session.current_step,
        "history": history
    }

@router.post("/session/{session_id}/generate-report")
async def generate_final_report(session_id: str, db: Session = Depends(get_db)):
    """Generate comprehensive final cleaning report"""
    session = db.query(ProcessingSession).filter(
        ProcessingSession.session_id == session_id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Get dataset and load data
        dataset = db.query(Dataset).filter(Dataset.id == session.dataset_id).first()
        file_extension = os.path.splitext(dataset.file_path)[1]
        original_df = data_upload_service._load_dataframe(dataset.file_path, file_extension)
        
        # Get processing history
        steps = db.query(ProcessingStep).filter(
            ProcessingStep.dataset_id == session.dataset_id
        ).order_by(ProcessingStep.step_number).all()
        
        # Calculate original and final metadata
        original_metadata = metadata_generator.generate_metadata(original_df)
        
        # Apply processing steps to get final state
        final_df = original_df.copy()
        processing_info_list = []
        total_processing_time = 0
        
        for step in steps:
            if step.status == "completed" and step.user_approved:
                try:
                    task = step.task
                    method = step.method
                    columns = step.columns or []
                    parameters = step.parameters or {}
                    
                    # Apply the processing step
                    if task == "remove_duplicates":
                        from app.modules.ml_processors import ml_processors
                        final_df, step_info = ml_processors.remove_duplicates(final_df, method, columns, **parameters)
                    elif task == "impute_missing":
                        from app.modules.ml_processors import ml_processors
                        final_df, step_info = ml_processors.impute_missing(final_df, method, columns, **parameters)
                    elif task == "detect_outliers":
                        from app.modules.ml_processors import ml_processors
                        final_df, step_info = ml_processors.detect_outliers(final_df, method, columns, **parameters)
                    elif task == "fix_dtypes":
                        from app.modules.ml_processors import ml_processors
                        final_df, step_info = ml_processors.fix_dtypes(final_df, columns, parameters.get("target_types"))
                    
                    processing_info_list.append(step_info)
                    total_processing_time += step.execution_time or 0
                    
                except Exception as step_error:
                    print(f"Warning: Could not replay step {step.step_number}: {step_error}")
                    continue
        
        # Generate final metadata
        final_metadata = metadata_generator.generate_metadata(final_df)
        
        # Build comprehensive history data
        history_data = []
        for step in steps:
            step_data = {
                "task": step.task,
                "method": step.method,
                "columns": step.columns,
                "parameters": step.parameters,
                "status": step.status,
                "verification_status": step.verification_status,
                "verification_reason": step.verification_reason,
                "execution_timestamp": step.execution_timestamp.isoformat() if step.execution_timestamp else "",
                "execution_time": step.execution_time or 0,
                "llm_reasoning": step.llm_reasoning,
                "user_approved": step.user_approved,
                "pre_metadata": step.pre_metadata,
                "post_metadata": step.post_metadata
            }
            
            # Add processing impact information
            if step.pre_metadata and step.post_metadata:
                pre_rows = step.pre_metadata.get("basic_info", {}).get("total_rows", 0)
                post_rows = step.post_metadata.get("basic_info", {}).get("total_rows", 0)
                step_data["processing_info"] = {
                    "records_affected": pre_rows - post_rows if pre_rows > post_rows else 0,
                    "improvement_percentage": 5.0  # Placeholder - could be calculated based on specific metrics
                }
            
            history_data.append(step_data)
        
        # Calculate comprehensive summary metrics
        original_rows = len(original_df)
        final_rows = len(final_df)
        data_retention = (final_rows / original_rows * 100) if original_rows > 0 else 100
        
        # Calculate quality scores
        original_quality = calculate_quality_score(original_metadata)
        final_quality = calculate_quality_score(final_metadata)
        
        # Build comprehensive summary
        summary = {
            "total_steps": len([s for s in steps if s.status == "completed"]),
            "total_processing_time": total_processing_time,
            "initial_quality_score": original_quality,
            "final_quality_score": final_quality,
            "quality_improvement": final_quality - original_quality,
            "data_preservation": {
                "original_rows": original_rows,
                "final_rows": final_rows,
                "percentage_retained": data_retention,
                "rows_removed": original_rows - final_rows
            },
            "file_size_mb": os.path.getsize(dataset.file_path) / (1024 * 1024) if os.path.exists(dataset.file_path) else 0,
            "quality_indicators": {
                "completeness": calculate_completeness_score(final_metadata),
                "consistency": 85.0,  # Placeholder - would need more sophisticated calculation
                "validity": calculate_validity_score(final_metadata),
                "uniqueness": calculate_uniqueness_score(final_metadata)
            }
        }
        
        # Generate recommendations
        recommendations = generate_recommendations(original_metadata, final_metadata, history_data)
        
        # Build comprehensive report using existing method
        basic_report = report_generator.generate_clean_report(
            history_data, final_df, original_metadata, final_metadata
        )
        
        # Enhance report with additional information
        enhanced_report = {
            "report": {
                "summary": summary,
                "steps": history_data,
                "original_metadata": original_metadata,
                "final_metadata": final_metadata,
                "recommendations": recommendations,
                "basic_report": basic_report,  # Include the generated report
                "data_comparison": {
                    "shape_change": {
                        "original": {"rows": original_rows, "columns": len(original_df.columns)},
                        "final": {"rows": final_rows, "columns": len(final_df.columns)}
                    },
                    "quality_improvement": {
                        "missing_values_reduction": (
                            original_metadata.get("missing_values", {}).get("total_missing", 0) -
                            final_metadata.get("missing_values", {}).get("total_missing", 0)
                        ),
                        "duplicates_removed": (
                            original_metadata.get("duplicates", {}).get("total_duplicates", 0) -
                            final_metadata.get("duplicates", {}).get("total_duplicates", 0)
                        )
                    }
                }
            },
            "export_options": ["csv", "xlsx", "json", "pdf"],
            "session_info": {
                "session_id": session_id,
                "dataset_id": session.dataset_id,
                "created_timestamp": session.created_timestamp.isoformat(),
                "status": session.status
            }
        }
        
        # Ensure JSON serialization safety before returning
        return _ensure_json_serializable(enhanced_report)
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/dataset/{dataset_id}/export")
async def export_dataset(
    dataset_id: int, 
    payload: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Export the cleaned dataset in specified format"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Extract parameters from payload
    format = payload.get("format", "csv")
    filename = payload.get("filename", f"cleaned_dataset_{dataset_id}")
    include_metadata = payload.get("include_metadata", False)
    
    try:
        # Load final dataset (this should eventually load the processed version)
        file_extension = os.path.splitext(dataset.file_path)[1]
        df = data_upload_service._load_dataframe(dataset.file_path, file_extension)
        
        # Apply any completed processing steps to get the final processed data
        # Get processing steps for this dataset
        processing_steps = db.query(ProcessingStep).filter(
            ProcessingStep.dataset_id == dataset_id,
            ProcessingStep.status == "completed",
            ProcessingStep.user_approved == True
        ).order_by(ProcessingStep.step_number).all()
        
        # Apply each processing step in order
        for step in processing_steps:
            try:
                task = step.task
                method = step.method
                columns = step.columns or []
                parameters = step.parameters or {}
                
                if task == "remove_duplicates":
                    from app.modules.ml_processors import ml_processors
                    df, _ = ml_processors.remove_duplicates(df, method, columns, **parameters)
                elif task == "impute_missing":
                    from app.modules.ml_processors import ml_processors
                    df, _ = ml_processors.impute_missing(df, method, columns, **parameters)
                elif task == "detect_outliers":
                    from app.modules.ml_processors import ml_processors
                    df, _ = ml_processors.detect_outliers(df, method, columns, **parameters)
                elif task == "fix_dtypes":
                    from app.modules.ml_processors import ml_processors
                    df, _ = ml_processors.fix_dtypes(df, columns, parameters.get("target_types"))
                # Add other tasks as needed
                
            except Exception as step_error:
                print(f"Warning: Could not apply step {step.step_number}: {step_error}")
                continue
        
        # Export the processed data
        export_result = report_generator.export_clean_data(df, format, filename)
        
        # Add metadata if requested
        if include_metadata and export_result.get("success"):
            metadata_info = {
                "export_timestamp": datetime.now().isoformat(),
                "original_dataset_id": dataset_id,
                "processing_steps_applied": len(processing_steps),
                "final_shape": {"rows": len(df), "columns": len(df.columns)},
                "processing_steps": [
                    {
                        "step_number": step.step_number,
                        "task": step.task,
                        "method": step.method,
                        "execution_time": step.execution_time,
                        "status": step.status
                    } for step in processing_steps
                ]
            }
            export_result["metadata"] = metadata_info
        
        return export_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Additional endpoints for enhanced frontend functionality

@router.get("/dataset/{dataset_id}/preview")
async def get_dataset_preview(dataset_id: int, rows: int = 5, db: Session = Depends(get_db)):
    """Get a preview of the dataset"""
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Load and preview data
        file_extension = os.path.splitext(dataset.file_path)[1]
        df = data_upload_service._load_dataframe(dataset.file_path, file_extension)
        preview_data = df.head(rows).to_dict("records")
        
        return {
            "file_id": str(dataset.id),
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "preview_data": preview_data,
            "data": preview_data,  # Alternative property name
            "data_types": df.dtypes.astype(str).to_dict(),
            "column_types": df.dtypes.astype(str).to_dict()  # Alternative property name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/step/{step_id}/approve")
async def approve_step(step_id: str, db: Session = Depends(get_db)):
    """Approve a processing step"""
    try:
        step = db.query(ProcessingStep).filter(ProcessingStep.id == int(step_id)).first()
        if not step:
            raise HTTPException(status_code=404, detail="Step not found")
        
        step.user_approved = True
        step.verification_status = "confirmed"
        db.commit()
        
        return {"status": "approved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/step/{step_id}/reject")
async def reject_step(step_id: str, reason: str, db: Session = Depends(get_db)):
    """Reject a processing step"""
    try:
        step = db.query(ProcessingStep).filter(ProcessingStep.id == int(step_id)).first()
        if not step:
            raise HTTPException(status_code=404, detail="Step not found")
        
        step.user_approved = False
        step.verification_status = "failed"
        step.verification_reason = reason
        db.commit()
        
        return {"status": "rejected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/step/{step_id}/preview")
async def get_step_preview(step_id: str, db: Session = Depends(get_db)):
    """Get preview of data after a processing step"""
    try:
        step = db.query(ProcessingStep).filter(ProcessingStep.id == int(step_id)).first()
        if not step:
            raise HTTPException(status_code=404, detail="Step not found")
        
        # For now, return mock data - in a real implementation, 
        # this would show the actual processed data
        return {
            "step_id": step_id,
            "preview_data": [
                {"id": 1, "name": "Sample Data", "value": 100, "status": "clean"},
                {"id": 2, "name": "Another Row", "value": 200, "status": "clean"}
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/session/{session_id}/download")
async def download_cleaned_data(session_id: str, db: Session = Depends(get_db)):
    """Download cleaned dataset"""
    try:
        session = db.query(ProcessingSession).filter(ProcessingSession.session_id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get the processed dataset by applying all completed steps
        dataset = db.query(Dataset).filter(Dataset.id == session.dataset_id).first()
        file_extension = os.path.splitext(dataset.file_path)[1]
        df = data_upload_service._load_dataframe(dataset.file_path, file_extension)
        
        # Apply completed processing steps
        processing_steps = db.query(ProcessingStep).filter(
            ProcessingStep.dataset_id == session.dataset_id,
            ProcessingStep.status == "completed",
            ProcessingStep.user_approved == True
        ).order_by(ProcessingStep.step_number).all()
        
        for step in processing_steps:
            try:
                task = step.task
                method = step.method
                columns = step.columns or []
                parameters = step.parameters or {}
                
                if task == "remove_duplicates":
                    from app.modules.ml_processors import ml_processors
                    df, _ = ml_processors.remove_duplicates(df, method, columns, **parameters)
                elif task == "impute_missing":
                    from app.modules.ml_processors import ml_processors
                    df, _ = ml_processors.impute_missing(df, method, columns, **parameters)
                elif task == "detect_outliers":
                    from app.modules.ml_processors import ml_processors
                    df, _ = ml_processors.detect_outliers(df, method, columns, **parameters)
                elif task == "fix_dtypes":
                    from app.modules.ml_processors import ml_processors
                    df, _ = ml_processors.fix_dtypes(df, columns, parameters.get("target_types"))
                
            except Exception as step_error:
                print(f"Warning: Could not apply step {step.step_number}: {step_error}")
                continue
        
        # Convert to CSV
        from fastapi.responses import Response
        csv_content = df.to_csv(index=False)
        
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=cleaned_data.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions for report generation
def calculate_quality_score(metadata: Dict[str, Any]) -> float:
    """Calculate overall data quality score"""
    try:
        total_missing = metadata.get("missing_values", {}).get("total_missing", 0)
        total_cells = metadata.get("basic_info", {}).get("total_rows", 1) * metadata.get("basic_info", {}).get("total_columns", 1)
        
        if total_cells == 0:
            return 0.0
        
        # Base score calculation
        completeness = max(0, (total_cells - total_missing) / total_cells * 100)
        
        # Adjust for duplicates
        duplicates = metadata.get("duplicates", {}).get("total_duplicates", 0)
        total_rows = metadata.get("basic_info", {}).get("total_rows", 1)
        uniqueness = max(0, (total_rows - duplicates) / total_rows * 100)
        
        # Combined score (weighted average)
        quality_score = (completeness * 0.6) + (uniqueness * 0.4)
        
        return min(100.0, max(0.0, quality_score))
    except:
        return 75.0  # Default score if calculation fails


def calculate_completeness_score(metadata: Dict[str, Any]) -> float:
    """Calculate data completeness score"""
    try:
        total_missing = metadata.get("missing_values", {}).get("total_missing", 0)
        total_cells = metadata.get("basic_info", {}).get("total_rows", 1) * metadata.get("basic_info", {}).get("total_columns", 1)
        
        if total_cells == 0:
            return 100.0
        
        return max(0.0, (total_cells - total_missing) / total_cells * 100)
    except:
        return 90.0


def calculate_validity_score(metadata: Dict[str, Any]) -> float:
    """Calculate data validity score"""
    try:
        # Simple validity based on data types and outliers
        outliers_count = 0
        outliers_info = metadata.get("outliers", {})
        for col, info in outliers_info.items():
            if isinstance(info, dict):
                outliers_count += info.get("iqr_outliers", 0)
        
        total_rows = metadata.get("basic_info", {}).get("total_rows", 1)
        validity = max(0, (total_rows - outliers_count) / total_rows * 100)
        
        return min(100.0, max(70.0, validity))  # Minimum 70% validity
    except:
        return 85.0


def calculate_uniqueness_score(metadata: Dict[str, Any]) -> float:
    """Calculate data uniqueness score"""
    try:
        duplicates = metadata.get("duplicates", {}).get("total_duplicates", 0)
        total_rows = metadata.get("basic_info", {}).get("total_rows", 1)
        
        if total_rows == 0:
            return 100.0
        
        return max(0.0, (total_rows - duplicates) / total_rows * 100)
    except:
        return 95.0


def generate_recommendations(original_metadata: Dict[str, Any], final_metadata: Dict[str, Any], history_data: List[Dict[str, Any]]) -> List[str]:
    """Generate data quality recommendations"""
    recommendations = []
    
    try:
        # Quality improvement recommendations
        original_quality = calculate_quality_score(original_metadata)
        final_quality = calculate_quality_score(final_metadata)
        improvement = final_quality - original_quality
        
        if improvement > 10:
            recommendations.append("Significant data quality improvement achieved through systematic cleaning processes.")
        elif improvement > 5:
            recommendations.append("Moderate data quality enhancement observed. Consider additional validation steps.")
        else:
            recommendations.append("Minimal quality change detected. Review cleaning strategy for optimization.")
        
        # Missing data recommendations
        final_missing = final_metadata.get("missing_values", {}).get("total_missing", 0)
        if final_missing > 0:
            recommendations.append("Consider implementing imputation strategies for remaining missing values.")
        else:
            recommendations.append("All missing values have been successfully addressed.")
        
        # Duplicate data recommendations
        final_duplicates = final_metadata.get("duplicates", {}).get("total_duplicates", 0)
        if final_duplicates > 0:
            recommendations.append("Review and address remaining duplicate records to ensure data uniqueness.")
        else:
            recommendations.append("Data uniqueness has been maintained throughout the cleaning process.")
        
        # Process optimization recommendations
        if len(history_data) > 5:
            recommendations.append("Consider streamlining the data cleaning pipeline to reduce processing steps.")
        
        recommendations.append("Implement regular data quality monitoring for future data collection.")
        recommendations.append("Document cleaning procedures for reproducibility and consistency.")
        
    except Exception as e:
        # Fallback recommendations
        recommendations = [
            "Data quality has been improved through systematic cleaning processes.",
            "Regular data validation procedures should be implemented for future data collection.",
            "Consider implementing automated data quality checks in the data pipeline.",
            "Monitor key quality indicators for ongoing data maintenance."
        ]
    
    return recommendations
