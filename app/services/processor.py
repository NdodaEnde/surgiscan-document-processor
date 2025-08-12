"""
Core document processing service.
Refactored from the original Historic_document_processor.py with production improvements.
"""

import asyncio
import uuid
import tempfile
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field
from agentic_doc.parse import parse

from app.core.config import settings, ProcessingMode, ProcessingStatus
from app.core.logging import get_logger, log_processing_start, log_processing_complete, log_processing_error
from app.schemas.medical_documents import DOCUMENT_SCHEMAS, DETECTION_PATTERNS, SimpleDocumentTypeDetection


logger = get_logger(__name__)


class ProcessingResult(BaseModel):
    """Result of document processing"""
    document_id: str
    filename: str
    status: str
    extracted_data: Dict[str, Any] = {}
    processing_summary: Dict[str, Any] = {}
    patient_info: Dict[str, Any] = {}
    error: Optional[str] = None
    confidence_score: Optional[float] = None
    needs_validation: bool = False


class DocumentProcessor:
    """Production-ready document processing service"""
    
    def __init__(self):
        self.schemas = DOCUMENT_SCHEMAS
        self.patterns = DETECTION_PATTERNS
        self.executor = ThreadPoolExecutor(max_workers=settings.MAX_CONCURRENT_PROCESSING)
        
    async def process_document(
        self, 
        document_id: str,
        file_path: str, 
        filename: str,
        mode: str = ProcessingMode.SMART,
        patient_context: Optional[Dict] = None
    ) -> ProcessingResult:
        """
        Process a single document with comprehensive error handling
        """
        
        log_processing_start(logger, document_id, filename, mode)
        start_time = datetime.utcnow()
        
        try:
            # Run processing in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._process_document_sync,
                file_path,
                mode
            )
            
            # Create processing result
            processing_result = ProcessingResult(
                document_id=document_id,
                filename=filename,
                status=ProcessingStatus.COMPLETED,
                extracted_data=result['extracted_data'],
                processing_summary=result['processing_summary'],
                patient_info=self._extract_patient_info(result['extracted_data']),
                confidence_score=self._calculate_confidence_score(result['extracted_data']),
                needs_validation=self._needs_validation(result['extracted_data'])
            )
            
            log_processing_complete(logger, document_id, result)
            return processing_result
            
        except Exception as e:
            log_processing_error(logger, document_id, e)
            return ProcessingResult(
                document_id=document_id,
                filename=filename,
                status=ProcessingStatus.FAILED,
                error=str(e)
            )
    
    def _process_document_sync(self, file_path: str, mode: str) -> Dict[str, Any]:
        """
        Synchronous document processing (runs in thread pool)
        """
        
        start_time = datetime.utcnow()
        
        # Choose document types based on mode
        if mode == ProcessingMode.EXTRACT_ALL:
            doc_types = list(self.schemas.keys())
        elif mode == ProcessingMode.FAST:
            doc_types = ["certificate_of_fitness", "vision_test", "audiometric_test"]
        elif mode == ProcessingMode.DETECT_ONLY:
            doc_types = self._detect_document_types(file_path, use_api_detection=True)
            if not doc_types:
                raise Exception("Document type detection failed")
        else:  # SMART mode
            doc_types = self._detect_document_types(file_path, use_api_detection=True)
            if not doc_types:
                doc_types = ["certificate_of_fitness", "vision_test", "audiometric_test"]
        
        # Extract data from documents
        extraction_results = self._extract_document_data(file_path, doc_types)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        summary = {
            'mode': mode,
            'document_types_attempted': len(doc_types),
            'successful_extractions': len(extraction_results),
            'total_fields_extracted': sum(len(data) for data in extraction_results.values()),
            'processing_time': round(processing_time, 2),
            'api_calls_made': len(doc_types) + (1 if mode in [ProcessingMode.SMART, ProcessingMode.DETECT_ONLY] else 0)
        }
        
        return {
            'extracted_data': extraction_results,
            'processing_summary': summary,
            'file_path': file_path
        }
    
    def _detect_document_types(self, file_path: str, use_api_detection: bool = True) -> List[str]:
        """
        Detect document types using hybrid approach
        """
        
        detected_types = []
        
        if use_api_detection:
            try:
                # Try API detection
                results = parse(file_path, extraction_model=SimpleDocumentTypeDetection)
                
                if results and results[0].extraction:
                    detection = results[0].extraction
                    types_str = detection.document_types_present
                    
                    # Parse comma-separated string
                    detected_types = [t.strip() for t in types_str.split(',') if t.strip()]
                    
                    # Validate detected types
                    detected_types = [t for t in detected_types if t in self.schemas]
                    
                    if detected_types:
                        return detected_types
                        
            except Exception as e:
                logger.warning(f"API detection failed: {str(e)[:100]}...")
        
        # Fallback: Content-based detection
        try:
            basic_results = parse(file_path)
            if basic_results and basic_results[0].markdown:
                content = basic_results[0].markdown.lower()
                
                # Score each document type
                scores = {}
                for doc_type, patterns in self.patterns.items():
                    score = sum(1 for pattern in patterns if pattern in content)
                    if score > 0:
                        scores[doc_type] = score
                
                # Return types with sufficient scores
                detected_types = [doc_type for doc_type, score in scores.items() if score >= 2]
                
                if detected_types:
                    return detected_types
                    
        except Exception as e:
            logger.warning(f"Content detection failed: {str(e)[:100]}...")
        
        # Final fallback
        return ["certificate_of_fitness", "vision_test", "audiometric_test"]
    
    def _extract_document_data(self, file_path: str, doc_types: List[str]) -> Dict[str, Any]:
        """
        Extract data for specified document types
        """
        
        extraction_results = {}
        
        for doc_type in doc_types:
            if doc_type not in self.schemas:
                continue
                
            try:
                schema = self.schemas[doc_type]
                results = parse(file_path, extraction_model=schema)
                
                if results and results[0].extraction:
                    data = results[0].extraction.model_dump()
                    meaningful_data = {k: v for k, v in data.items() 
                                     if v is not None and v != "" and v != []}
                    
                    # Quality threshold
                    min_fields = 3 if doc_type != "consent_form" else 2
                    
                    if len(meaningful_data) >= min_fields:
                        extraction_results[doc_type] = meaningful_data
                        
            except Exception as e:
                logger.warning(f"Extraction failed for {doc_type}: {str(e)[:80]}...")
        
        return extraction_results
    
    def _extract_patient_info(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract common patient information from all document types
        """
        
        patient_info = {}
        
        for doc_type, data in extracted_data.items():
            # Patient name (try multiple field names)
            name_fields = ['patient_name', 'name', 'initials_and_surname', 'employee_name']
            for field in name_fields:
                if field in data and data[field] and not patient_info.get('name'):
                    patient_info['name'] = data[field]
            
            # ID number
            id_fields = ['id_number', 'patient_id']
            for field in id_fields:
                if field in data and data[field] and not patient_info.get('id_number'):
                    patient_info['id_number'] = data[field]
            
            # Company
            company_fields = ['company_name', 'company', 'employer_name']
            for field in company_fields:
                if field in data and data[field] and not patient_info.get('company_name'):
                    patient_info['company_name'] = data[field]
            
            # Latest examination date
            date_fields = ['examination_date', 'test_date', 'consent_date']
            for field in date_fields:
                if field in data and data[field]:
                    patient_info['last_examination_date'] = data[field]
                    break
        
        return patient_info
    
    def _calculate_confidence_score(self, extracted_data: Dict[str, Any]) -> float:
        """
        Calculate confidence score based on extraction quality
        """
        
        if not extracted_data:
            return 0.0
        
        total_score = 0
        total_documents = len(extracted_data)
        
        for doc_type, data in extracted_data.items():
            # Score based on number of fields extracted
            field_count = len([v for v in data.values() if v])
            max_expected_fields = {
                'certificate_of_fitness': 15,
                'vision_test': 12,
                'audiometric_test': 10,
                'spirometry_report': 15,
                'consent_form': 8,
                'medical_questionnaire': 20
            }
            
            expected_fields = max_expected_fields.get(doc_type, 10)
            document_score = min(1.0, field_count / expected_fields)
            total_score += document_score
        
        return round(total_score / total_documents, 2)
    
    def _needs_validation(self, extracted_data: Dict[str, Any]) -> bool:
        """
        Determine if document needs human validation
        """
        
        if not extracted_data:
            return True
        
        # Check confidence score
        confidence = self._calculate_confidence_score(extracted_data)
        if confidence < 0.8:
            return True
        
        # Check for critical patient data
        patient_info = self._extract_patient_info(extracted_data)
        if not patient_info.get('name') or not patient_info.get('id_number'):
            return True
        
        return False
    
    async def batch_process_documents(
        self,
        documents: List[Tuple[str, str, str]],  # (document_id, file_path, filename)
        mode: str = ProcessingMode.SMART
    ) -> List[ProcessingResult]:
        """
        Process multiple documents concurrently
        """
        
        batch_id = str(uuid.uuid4())
        logger.info(f"Starting batch processing: {len(documents)} documents", 
                   extra={'batch_id': batch_id})
        
        # Process documents concurrently
        tasks = []
        for document_id, file_path, filename in documents:
            task = self.process_document(document_id, file_path, filename, mode)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                document_id, _, filename = documents[i]
                error_result = ProcessingResult(
                    document_id=document_id,
                    filename=filename,
                    status=ProcessingStatus.FAILED,
                    error=str(result)
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        successful = len([r for r in processed_results if r.status == ProcessingStatus.COMPLETED])
        failed = len([r for r in processed_results if r.status == ProcessingStatus.FAILED])
        
        logger.info(f"Batch processing completed: {successful} successful, {failed} failed",
                   extra={'batch_id': batch_id, 'successful': successful, 'failed': failed})
        
        return processed_results
    
    async def cleanup(self):
        """
        Cleanup resources
        """
        if self.executor:
            self.executor.shutdown(wait=True)