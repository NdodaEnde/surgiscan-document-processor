"""
FastAPI application for the SurgiScan Document Processing Microservice.
Production-ready microservice with proper error handling, monitoring, and security.
"""

import os
import time
import uuid
import tempfile
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import uvicorn

from app.core.config import settings
from app.core.logging import setup_logging, get_logger, RequestLogger
from app.api.models import (
    HealthResponse, ErrorResponse, DocumentProcessingResult, 
    BatchProcessingResult, ValidationRequest, ValidationResponse,
    FileUploadValidator, ProcessingModeEnum, ProcessingStatusEnum
)
from app.services.processor import DocumentProcessor
from app.services.database import DatabaseManager
from app.services.storage import StorageManager


# Setup logging
setup_logging()
logger = get_logger(__name__)

# Global service instances
processor: Optional[DocumentProcessor] = None
db_manager: Optional[DatabaseManager] = None
storage_manager: Optional[StorageManager] = None

# Security
security = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    
    # Startup
    logger.info("Starting SurgiScan Document Processing Microservice")
    
    global processor, db_manager, storage_manager
    
    try:
        # Initialize services
        processor = DocumentProcessor()
        db_manager = DatabaseManager(settings.MONGODB_URL, settings.DATABASE_NAME)
        storage_manager = StorageManager()
        
        # Connect to database
        if settings.MONGODB_URL:
            await db_manager.connect()
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        # Continue anyway - some functionality may work
    
    yield
    
    # Shutdown
    logger.info("Shutting down microservice")
    
    if processor:
        await processor.cleanup()
    
    if db_manager:
        await db_manager.close()
    
    logger.info("Microservice shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AI-powered medical document processing microservice for SurgiScan",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

if not settings.DEBUG:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure based on your deployment
    )


# Authentication dependency
async def get_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Validate API key if configured"""
    
    if settings.API_KEY:
        if not credentials or credentials.credentials != settings.API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"}
            )
    return credentials


# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests"""
    
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log API request
    logger.info(
        f"{request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "request_id": request_id,
            "process_time": process_time,
            "status_code": response.status_code
        }
    )
    
    return response


# Error handling
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_code=f"HTTP_{exc.status_code}"
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.error(
        f"Unhandled exception in request {request_id}: {str(exc)}",
        exc_info=True,
        extra={"request_id": request_id}
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR"
        ).model_dump()
    )


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    
    uptime = time.time() - getattr(health_check, 'start_time', time.time())
    if not hasattr(health_check, 'start_time'):
        health_check.start_time = time.time()
    
    return HealthResponse(
        status="healthy",
        service=settings.PROJECT_NAME,
        version=settings.VERSION,
        timestamp=datetime.utcnow(),
        uptime_seconds=uptime,
        mongodb_connected=db_manager.connected if db_manager else False,
        supported_document_types=[
            "certificate_of_fitness",
            "vision_test", 
            "audiometric_test",
            "spirometry_report",
            "consent_form",
            "medical_questionnaire"
        ],
        processing_modes=["smart", "fast", "extract_all", "detect_only"],
        features={
            "batch_processing": True,
            "async_processing": True,
            "database_storage": bool(settings.MONGODB_URL),
            "file_storage": True,
            "validation_workflow": True
        }
    )


@app.post("/api/v1/historic-documents/upload", response_model=DocumentProcessingResult)
async def upload_document(
    file: UploadFile = File(...),
    processing_mode: ProcessingModeEnum = Form(ProcessingModeEnum.SMART),
    save_to_database: bool = Form(True),
    api_key: Optional[HTTPAuthorizationCredentials] = Depends(get_api_key)
):
    """
    Upload and process a single historic document.
    Compatible with SurgiScan Historic Documents component.
    """
    
    request_id = str(uuid.uuid4())
    document_id = str(uuid.uuid4())
    
    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )
    
    if not FileUploadValidator.validate_file_extension(file.filename, settings.ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not allowed. Supported: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )
    
    if file.size and not FileUploadValidator.validate_file_size(file.size, settings.MAX_FILE_SIZE_MB):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE_MB}MB"
        )
    
    # Save temporary file
    temp_path = None
    try:
        with RequestLogger(logger, request_id, "document_processing"):
            # Save uploaded file
            temp_path = await storage_manager.save_temp_file(file)
            
            # Process document
            result = await processor.process_document(
                document_id=document_id,
                file_path=temp_path,
                filename=file.filename,
                mode=processing_mode.value
            )
            
            # Save to database if requested
            db_info = {"status": "not_saved"}
            if save_to_database and db_manager and db_manager.connected:
                try:
                    db_document_id = await db_manager.save_processing_result(
                        document_id, file.filename, result
                    )
                    db_info = {
                        "status": "saved",
                        "document_id": db_document_id,
                        "saved_at": datetime.utcnow()
                    }
                except Exception as e:
                    logger.error(f"Failed to save to database: {e}")
                    db_info = {"status": "save_failed", "error": str(e)}
            
            # Store file permanently if processing succeeded
            file_url = None
            if result.status == ProcessingStatusEnum.COMPLETED:
                try:
                    file_url = await storage_manager.store_file(temp_path, file.filename, document_id)
                except Exception as e:
                    logger.warning(f"Failed to store file permanently: {e}")
            
            return DocumentProcessingResult(
                success=result.status == ProcessingStatusEnum.COMPLETED,
                document_id=document_id,
                filename=file.filename,
                status=ProcessingStatusEnum(result.status),
                document_types_found=[dt for dt in result.extracted_data.keys()],
                extracted_data=result.extracted_data,
                processing_summary=result.processing_summary,
                patient_info=result.patient_info,
                database=db_info,
                created_at=datetime.utcnow(),
                needs_validation=result.needs_validation,
                confidence_score=result.confidence_score
            )
    
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.post("/api/v1/historic-documents/batch-upload", response_model=BatchProcessingResult)
async def batch_upload_documents(
    files: List[UploadFile] = File(...),
    processing_mode: ProcessingModeEnum = Form(ProcessingModeEnum.SMART),
    save_to_database: bool = Form(True),
    api_key: Optional[HTTPAuthorizationCredentials] = Depends(get_api_key)
):
    """
    Upload and process multiple documents in a batch
    """
    
    batch_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided"
        )
    
    # Validate all files first
    for file in files:
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid filename: {file.filename}"
            )
        
        if not FileUploadValidator.validate_file_extension(file.filename, settings.ALLOWED_EXTENSIONS):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type not allowed: {file.filename}"
            )
        
        if file.size and not FileUploadValidator.validate_file_size(file.size, settings.MAX_FILE_SIZE_MB):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File too large: {file.filename}"
            )
    
    with RequestLogger(logger, request_id, f"batch_processing_{len(files)}_files"):
        # Save all files temporarily
        temp_files = []
        documents = []
        
        try:
            for file in files:
                temp_path = await storage_manager.save_temp_file(file)
                document_id = str(uuid.uuid4())
                temp_files.append(temp_path)
                documents.append((document_id, temp_path, file.filename))
            
            # Process all documents
            results = await processor.batch_process_documents(documents, processing_mode.value)
            
            # Convert results to API format
            api_results = []
            successful = 0
            failed = 0
            
            for i, result in enumerate(results):
                document_id, _, filename = documents[i]
                
                # Save to database if requested
                db_info = {"status": "not_saved"}
                if (save_to_database and db_manager and db_manager.connected and 
                    result.status == ProcessingStatusEnum.COMPLETED):
                    try:
                        db_document_id = await db_manager.save_processing_result(
                            document_id, filename, result
                        )
                        db_info = {
                            "status": "saved",
                            "document_id": db_document_id,
                            "saved_at": datetime.utcnow()
                        }
                    except Exception as e:
                        logger.error(f"Failed to save {filename} to database: {e}")
                        db_info = {"status": "save_failed", "error": str(e)}
                
                api_result = DocumentProcessingResult(
                    success=result.status == ProcessingStatusEnum.COMPLETED,
                    document_id=document_id,
                    filename=filename,
                    status=ProcessingStatusEnum(result.status),
                    document_types_found=[dt for dt in result.extracted_data.keys()],
                    extracted_data=result.extracted_data,
                    processing_summary=result.processing_summary,
                    patient_info=result.patient_info,
                    database=db_info,
                    created_at=datetime.utcnow(),
                    needs_validation=result.needs_validation,
                    confidence_score=result.confidence_score
                )
                
                api_results.append(api_result)
                
                if result.status == ProcessingStatusEnum.COMPLETED:
                    successful += 1
                else:
                    failed += 1
            
            return BatchProcessingResult(
                success=True,
                batch_id=batch_id,
                total_files=len(files),
                successful_extractions=successful,
                failed_extractions=failed,
                processing_mode=processing_mode,
                saved_to_database=save_to_database,
                results=api_results,
                created_at=datetime.utcnow()
            )
        
        finally:
            # Clean up temporary files
            for temp_path in temp_files:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)


@app.get("/api/v1/historic-documents/{document_id}/status")
async def get_document_status(
    document_id: str,
    api_key: Optional[HTTPAuthorizationCredentials] = Depends(get_api_key)
):
    """Get processing status of a document"""
    
    if not db_manager or not db_manager.connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )
    
    try:
        document = await db_manager.get_document_by_id(document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        return {
            "document_id": document_id,
            "filename": document.get("filename"),
            "status": document.get("status", "unknown"),
            "upload_date": document.get("created_at"),
            "processing_complete_at": document.get("processed_at"),
            "confidence_score": document.get("confidence_score"),
            "needs_validation": document.get("needs_validation", False),
            "is_validated": document.get("is_validated", False),
            "document_types": document.get("document_types", []),
            "patient_info": document.get("patient_info", {})
        }
        
    except Exception as e:
        logger.error(f"Failed to get document status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document status"
        )


@app.get("/api/v1/historic-documents/{document_id}")
async def get_document(
    document_id: str,
    api_key: Optional[HTTPAuthorizationCredentials] = Depends(get_api_key)
):
    """Get complete document data including extracted content"""
    
    if not db_manager or not db_manager.connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )
    
    try:
        document = await db_manager.get_document_by_id(document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        return {"document": document}
        
    except Exception as e:
        logger.error(f"Failed to get document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document"
        )


@app.put("/api/v1/historic-documents/{document_id}/validate", response_model=ValidationResponse)
async def validate_document(
    document_id: str,
    validation: ValidationRequest,
    api_key: Optional[HTTPAuthorizationCredentials] = Depends(get_api_key)
):
    """Validate and save corrected extracted data"""
    
    if not db_manager or not db_manager.connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )
    
    try:
        # Update document with validated data
        success = await db_manager.update_document_validation(
            document_id, 
            validation.extracted_data,
            validation.validation_notes
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # TODO: Integrate with SurgiScan patient system
        integration_result = {"success": False, "message": "Integration not implemented"}
        
        return ValidationResponse(
            success=True,
            document_id=document_id,
            validation_status="validated",
            integration=integration_result,
            message="Document validation completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to validate document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate document"
        )


@app.get("/api/v1/statistics")
async def get_statistics(
    api_key: Optional[HTTPAuthorizationCredentials] = Depends(get_api_key)
):
    """Get processing statistics"""
    
    if not db_manager or not db_manager.connected:
        return {
            "error": "Database not connected",
            "total_documents": 0,
            "last_updated": datetime.utcnow()
        }
    
    try:
        stats = await db_manager.get_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics"
        )


# =============================================================================
# APPLICATION STARTUP
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=settings.DEBUG,
        reload=settings.DEBUG
    )