import os
import uuid
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import pyreadstat
from fastapi import UploadFile, HTTPException
from app.core.config import settings
from app.utils.file_utils import get_file_extension, validate_file_size

class DataUploadService:
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    async def upload_file(self, file: UploadFile) -> Dict[str, Any]:
        """
        Accepts CSV/XLSX/SAV files and stores them temporarily.
        Returns file metadata and unique file ID.
        """
        # Validate file extension
        file_ext = get_file_extension(file.filename)
        if file_ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_ext} not supported. Allowed types: {settings.ALLOWED_EXTENSIONS}"
            )
        
        # Validate file size
        content = await file.read()
        await file.seek(0)  # Reset file pointer
        
        if not validate_file_size(content, settings.MAX_FILE_SIZE):
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / (1024*1024):.1f}MB"
            )
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = f"{file_id}{file_ext}"
        file_path = self.upload_dir / filename
        
        # Save file
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
        
        return {
            "file_id": file_id,
            "filename": filename,
            "original_filename": file.filename,
            "file_path": str(file_path),
            "file_size": len(content),
            "file_extension": file_ext
        }
    
    def preview_data(self, file_id: str, rows: int = 5) -> Dict[str, Any]:
        """
        Returns top rows of the dataset for UI preview with JSON serialization safety.
        """
        file_info = self._get_file_info(file_id)
        if not file_info:
            raise HTTPException(status_code=404, detail="File not found")
        
        try:
            df = self._load_dataframe(file_info["file_path"], file_info["file_extension"])
            
            # Get preview data and ensure JSON serialization
            preview_df = df.head(rows)
            
            # Convert preview data to ensure JSON serialization
            preview_records = []
            for _, row in preview_df.iterrows():
                record = {}
                for col in preview_df.columns:
                    value = row[col]
                    # Handle different data types for JSON serialization
                    if pd.isna(value):
                        record[col] = None
                    elif hasattr(value, 'item'):  # numpy types
                        try:
                            converted = value.item()
                            # Check for infinity and NaN in float values
                            if isinstance(converted, float):
                                if np.isinf(converted) or np.isnan(converted):
                                    record[col] = None
                                else:
                                    record[col] = converted
                            else:
                                record[col] = converted
                        except:
                            record[col] = str(value)
                    elif isinstance(value, float):
                        # Handle regular Python floats that might be inf/nan
                        if np.isinf(value) or np.isnan(value):
                            record[col] = None
                        else:
                            record[col] = value
                    else:
                        record[col] = value
                preview_records.append(record)
            
            return {
                "file_id": file_id,
                "total_rows": int(len(df)),  # Ensure int not numpy.int64
                "total_columns": int(len(df.columns)),
                "columns": list(df.columns),
                "preview_data": preview_records,
                "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to preview data: {str(e)}")
    
    def validate_format(self, file_id: str) -> Dict[str, Any]:
        """
        Checks file format and returns validation status.
        """
        file_info = self._get_file_info(file_id)
        if not file_info:
            raise HTTPException(status_code=404, detail="File not found")
        
        try:
            df = self._load_dataframe(file_info["file_path"], file_info["file_extension"])
            
            # Basic validation checks
            validation_results = {
                "file_id": file_id,
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "basic_info": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "memory_usage": df.memory_usage(deep=True).sum(),
                    "has_headers": True  # Assume headers exist for now
                }
            }
            
            # Check for empty dataset
            if len(df) == 0:
                validation_results["errors"].append("Dataset is empty")
                validation_results["is_valid"] = False
            
            # Check for duplicate column names
            if len(df.columns) != len(set(df.columns)):
                validation_results["warnings"].append("Duplicate column names detected")
            
            # Check for unnamed columns
            unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed:')]
            if unnamed_cols:
                validation_results["warnings"].append(f"Unnamed columns detected: {unnamed_cols}")
            
            return validation_results
            
        except Exception as e:
            return {
                "file_id": file_id,
                "is_valid": False,
                "errors": [f"Failed to read file: {str(e)}"],
                "warnings": [],
                "basic_info": None
            }
    
    def _get_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Helper method to get file information by file_id"""
        # In a real implementation, this would query the database
        # For now, we'll search for the file in the upload directory
        for file_path in self.upload_dir.glob(f"{file_id}.*"):
            return {
                "file_id": file_id,
                "file_path": str(file_path),
                "file_extension": file_path.suffix
            }
        return None
    
    def _load_dataframe(self, file_path: str, file_extension: str) -> pd.DataFrame:
        """Helper method to load dataframe based on file extension"""
        if file_extension == ".csv":
            # Try different encodings for CSV files
            encodings = ['utf-8', 'utf-8-sig', 'latin1', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    return pd.read_csv(file_path, encoding=encoding)
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    if encoding == encodings[-1]:  # Last encoding attempt
                        raise e
            raise ValueError(f"Could not decode CSV file with any of the attempted encodings: {encodings}")
        elif file_extension in [".xlsx", ".xls"]:
            try:
                return pd.read_excel(file_path)
            except Exception as e:
                raise ValueError(f"Failed to read Excel file: {str(e)}")
        elif file_extension == ".sav":
            try:
                df, meta = pyreadstat.read_sav(file_path)
                return df
            except Exception as e:
                raise ValueError(f"Failed to read SPSS file: {str(e)}")
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

# Global instance
data_upload_service = DataUploadService()
