#!/usr/bin/env python3
"""
REST API for MeiGen-MultiTalk video generation.
Provides async job processing with status tracking and webhook support.
"""

import modal
import os
import uuid
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union
from enum import Enum
import asyncio
import httpx

from fastapi import FastAPI, HTTPException, Depends, Security, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, HttpUrl, validator
import boto3
from botocore.exceptions import ClientError

# Import our existing Modal app and functions
try:
    from app_multitalk_cuda import app as modal_app, generate_video_cuda, generate_multi_person_video
except ImportError:
    # For testing without Modal runtime
    import modal
    modal_app = modal.App("multitalk-api")
    generate_video_cuda = None
    generate_multi_person_video = None

# ==================== Configuration ====================

API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"
MAX_AUDIO_FILES = 5
WEBHOOK_RETRY_ATTEMPTS = 3
WEBHOOK_TIMEOUT = 10

# ==================== Enums ====================

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AudioType(str, Enum):
    ADD = "add"  # Sequential speaking
    PARA = "para"  # Simultaneous speaking

class VideoResolution(str, Enum):
    RES_480P = "480p"
    RES_720P = "720p"

# ==================== Pydantic Models ====================

class S3Location(BaseModel):
    """S3 location specification."""
    bucket: str = Field(..., description="S3 bucket name")
    key: str = Field(..., description="S3 object key")
    
    @validator('bucket')
    def validate_bucket_name(cls, v):
        if not v or len(v) < 3 or len(v) > 63:
            raise ValueError("Bucket name must be between 3 and 63 characters")
        return v

class GenerationOptions(BaseModel):
    """Video generation options."""
    resolution: VideoResolution = Field(
        default=VideoResolution.RES_480P,
        description="Output video resolution"
    )
    sample_steps: int = Field(
        default=20,
        ge=10,
        le=50,
        description="Number of sampling steps (10-50)"
    )
    audio_type: AudioType = Field(
        default=AudioType.ADD,
        description="Audio mode for multi-person videos"
    )
    audio_cfg: float = Field(
        default=4.0,
        ge=3.0,
        le=5.0,
        description="Audio guidance scale (3-5)"
    )
    color_correction: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Color correction strength (0-1)"
    )

class VideoGenerationRequest(BaseModel):
    """Request model for video generation."""
    prompt: str = Field(..., description="Text description of the video content")
    image_s3_url: str = Field(..., description="S3 URL for reference image")
    audio_s3_urls: Union[str, List[str]] = Field(
        ...,
        description="S3 URL(s) for audio file(s)"
    )
    output_s3_bucket: Optional[str] = Field(
        None,
        description="S3 bucket for output (uses default if not specified)"
    )
    output_s3_prefix: str = Field(
        default="outputs/",
        description="S3 prefix for output files"
    )
    webhook_url: Optional[HttpUrl] = Field(
        None,
        description="URL for completion webhook"
    )
    options: GenerationOptions = Field(
        default_factory=GenerationOptions,
        description="Generation options"
    )
    
    @validator('audio_s3_urls')
    def validate_audio_urls(cls, v):
        if isinstance(v, str):
            v = [v]
        if len(v) > MAX_AUDIO_FILES:
            raise ValueError(f"Maximum {MAX_AUDIO_FILES} audio files allowed")
        return v
    
    @validator('image_s3_url', 'audio_s3_urls', pre=True)
    def validate_s3_urls(cls, v):
        def is_valid_s3_url(url):
            return url.startswith('s3://') and len(url.split('/', 3)) >= 4
        
        if isinstance(v, list):
            for url in v:
                if not is_valid_s3_url(url):
                    raise ValueError(f"Invalid S3 URL format: {url}")
        else:
            if not is_valid_s3_url(v):
                raise ValueError(f"Invalid S3 URL format: {v}")
        return v

class JobResponse(BaseModel):
    """Response model for job submission."""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    created_at: datetime = Field(..., description="Job creation timestamp")
    message: str = Field(..., description="Status message")

class JobStatusResponse(BaseModel):
    """Response model for job status checks."""
    job_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    progress: Optional[int] = Field(None, ge=0, le=100, description="Progress percentage")
    result: Optional[Dict] = Field(None, description="Job result (when completed)")
    error: Optional[str] = Field(None, description="Error message (when failed)")
    metadata: Optional[Dict] = Field(None, description="Additional job metadata")

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    request_id: Optional[str] = Field(None, description="Request tracking ID")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy")
    version: str = Field(default=API_VERSION)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class WebhookPayload(BaseModel):
    """Webhook notification payload."""
    job_id: str
    status: JobStatus
    timestamp: datetime
    result: Optional[Dict] = None
    error: Optional[str] = None

# ==================== Job Storage ====================

# In-memory job storage (use PostgreSQL/Redis for production)
jobs_db = modal.Dict.from_name("multitalk-jobs", create_if_missing=True)

class JobManager:
    """Manages job lifecycle and storage."""
    
    @staticmethod
    async def create_job(request_data: dict) -> str:
        """Create a new job and return job ID."""
        job_id = str(uuid.uuid4())
        job_data = {
            "id": job_id,
            "status": JobStatus.PENDING,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "request": request_data,
            "progress": 0,
            "result": None,
            "error": None,
            "metadata": {}
        }
        jobs_db[job_id] = job_data
        return job_id
    
    @staticmethod
    async def get_job(job_id: str) -> Optional[dict]:
        """Get job data by ID."""
        return jobs_db.get(job_id)
    
    @staticmethod
    async def update_job(job_id: str, updates: dict) -> bool:
        """Update job data."""
        job = jobs_db.get(job_id)
        if not job:
            return False
        
        job.update(updates)
        job["updated_at"] = datetime.now(timezone.utc).isoformat()
        jobs_db[job_id] = job
        return True
    
    @staticmethod
    async def list_jobs(limit: int = 100) -> List[dict]:
        """List recent jobs."""
        # In production, implement proper pagination
        all_jobs = []
        
        # Modal Dict doesn't support iteration directly
        # For now, we'll just return an empty list in testing
        # In production, use a proper database
        try:
            # This is a placeholder - in production use proper DB queries
            return []
        except Exception:
            return []

# ==================== Authentication ====================

security = HTTPBearer()

def get_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """Validate API key from Bearer token."""
    api_key = credentials.credentials
    
    # Get valid API keys from environment
    api_keys_env = os.environ.get("API_KEYS", "")
    
    if not api_keys_env:
        # Development mode - accept any key if API_KEYS not set
        print("Warning: API_KEYS not configured, accepting any key")
        return api_key
    
    # Parse valid keys
    valid_api_keys = [key.strip() for key in api_keys_env.split(",") if key.strip()]
    
    if api_key not in valid_api_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return api_key

# ==================== Webhook Delivery ====================

async def deliver_webhook(webhook_url: str, payload: WebhookPayload, attempt: int = 1):
    """Deliver webhook notification with retry logic."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                webhook_url,
                json=payload.dict(),
                timeout=WEBHOOK_TIMEOUT,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            print(f"Webhook delivered successfully to {webhook_url}")
            return True
            
        except Exception as e:
            print(f"Webhook delivery failed (attempt {attempt}): {e}")
            
            if attempt < WEBHOOK_RETRY_ATTEMPTS:
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
                return await deliver_webhook(webhook_url, payload, attempt + 1)
            
            return False

# ==================== S3 Utilities ====================

def parse_s3_url(s3_url: str) -> tuple[str, str]:
    """Parse S3 URL into bucket and key."""
    if not s3_url.startswith("s3://"):
        raise ValueError("Invalid S3 URL format")
    
    parts = s3_url[5:].split("/", 1)
    if len(parts) != 2:
        raise ValueError("Invalid S3 URL format")
    
    return parts[0], parts[1]

def get_s3_client():
    """Get configured S3 client."""
    return boto3.client('s3')

async def generate_presigned_url(bucket: str, key: str, expiration: int = 3600) -> str:
    """Generate presigned URL for S3 object."""
    s3_client = get_s3_client()
    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=expiration
        )
        return url
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"Error generating download URL: {e}")

# ==================== FastAPI App ====================

# Create FastAPI app
web_app = FastAPI(
    title="MeiGen-MultiTalk API",
    description="REST API for audio-driven video generation",
    version=API_VERSION,
    docs_url=f"{API_PREFIX}/docs",
    redoc_url=f"{API_PREFIX}/redoc",
    openapi_url=f"{API_PREFIX}/openapi.json"
)

# ==================== API Endpoints ====================

@web_app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API docs."""
    return RedirectResponse(url=f"{API_PREFIX}/docs")

@web_app.get(f"{API_PREFIX}/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse()

@web_app.post(
    f"{API_PREFIX}/generate",
    response_model=JobResponse,
    tags=["Video Generation"],
    summary="Submit video generation job",
    response_description="Job submission confirmation"
)
async def generate_video(
    request: VideoGenerationRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key)
):
    """
    Submit a video generation job.
    
    The job will be processed asynchronously. Use the returned job_id to check status.
    """
    # Create job
    job_id = await JobManager.create_job(request.dict())
    
    # Schedule background processing
    background_tasks.add_task(process_video_generation, job_id, request)
    
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        created_at=datetime.now(timezone.utc),
        message="Job submitted successfully"
    )

@web_app.get(
    f"{API_PREFIX}/jobs/{{job_id}}",
    response_model=JobStatusResponse,
    tags=["Job Management"],
    summary="Get job status",
    responses={
        404: {"model": ErrorResponse, "description": "Job not found"}
    }
)
async def get_job_status(
    job_id: str,
    api_key: str = Depends(get_api_key)
):
    """Get the status of a video generation job."""
    job = await JobManager.get_job(job_id)
    
    if not job:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error="Job not found",
                detail=f"No job with ID {job_id}",
                request_id=job_id
            ).dict()
        )
    
    return JobStatusResponse(
        job_id=job["id"],
        status=job["status"],
        created_at=datetime.fromisoformat(job["created_at"]),
        updated_at=datetime.fromisoformat(job["updated_at"]),
        progress=job.get("progress"),
        result=job.get("result"),
        error=job.get("error"),
        metadata=job.get("metadata", {})
    )

@web_app.get(
    f"{API_PREFIX}/jobs/{{job_id}}/download",
    tags=["Job Management"],
    summary="Get download URL for completed job",
    responses={
        404: {"model": ErrorResponse, "description": "Job not found"},
        400: {"model": ErrorResponse, "description": "Job not completed"}
    }
)
async def get_download_url(
    job_id: str,
    expiration: int = 3600,
    api_key: str = Depends(get_api_key)
):
    """Get a presigned download URL for the generated video."""
    job = await JobManager.get_job(job_id)
    
    if not job:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(error="Job not found").dict()
        )
    
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="Job not completed",
                detail=f"Job status is {job['status']}"
            ).dict()
        )
    
    result = job.get("result", {})
    s3_output = result.get("s3_output")
    
    if not s3_output:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(error="Output file not found").dict()
        )
    
    # Parse S3 URL and generate presigned URL
    bucket, key = parse_s3_url(s3_output)
    download_url = await generate_presigned_url(bucket, key, expiration)
    
    return {
        "job_id": job_id,
        "download_url": download_url,
        "expires_in": expiration,
        "s3_uri": s3_output
    }

@web_app.delete(
    f"{API_PREFIX}/jobs/{{job_id}}",
    tags=["Job Management"],
    summary="Cancel a job",
    responses={
        404: {"model": ErrorResponse, "description": "Job not found"}
    }
)
async def cancel_job(
    job_id: str,
    api_key: str = Depends(get_api_key)
):
    """Cancel a pending or processing job."""
    job = await JobManager.get_job(job_id)
    
    if not job:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(error="Job not found").dict()
        )
    
    if job["status"] in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="Job cannot be cancelled",
                detail=f"Job is already {job['status']}"
            ).dict()
        )
    
    # Update job status
    await JobManager.update_job(job_id, {"status": JobStatus.CANCELLED})
    
    return {"job_id": job_id, "status": JobStatus.CANCELLED, "message": "Job cancelled"}

@web_app.get(
    f"{API_PREFIX}/jobs",
    tags=["Job Management"],
    summary="List recent jobs"
)
async def list_jobs(
    limit: int = 10,
    api_key: str = Depends(get_api_key)
):
    """List recent video generation jobs."""
    jobs = await JobManager.list_jobs(limit)
    
    return {
        "jobs": [
            {
                "job_id": job["id"],
                "status": job["status"],
                "created_at": job["created_at"],
                "updated_at": job["updated_at"]
            }
            for job in jobs
        ],
        "count": len(jobs)
    }

@web_app.post(
    f"{API_PREFIX}/webhook-test",
    tags=["Testing"],
    summary="Test webhook delivery"
)
async def test_webhook(
    webhook_url: HttpUrl,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key)
):
    """Test webhook delivery to ensure your endpoint is configured correctly."""
    test_payload = WebhookPayload(
        job_id="test-" + str(uuid.uuid4()),
        status=JobStatus.COMPLETED,
        timestamp=datetime.now(timezone.utc),
        result={"message": "This is a test webhook"}
    )
    
    background_tasks.add_task(deliver_webhook, str(webhook_url), test_payload)
    
    return {
        "message": "Test webhook scheduled",
        "webhook_url": str(webhook_url),
        "payload": test_payload.dict()
    }

# ==================== Background Processing ====================

async def process_video_generation(job_id: str, request: VideoGenerationRequest):
    """Process video generation job in the background."""
    try:
        # Update job status to processing
        await JobManager.update_job(job_id, {
            "status": JobStatus.PROCESSING,
            "progress": 10
        })
        
        # Parse S3 URLs
        image_bucket, image_key = parse_s3_url(request.image_s3_url)
        
        # Handle single or multiple audio files
        audio_urls = request.audio_s3_urls
        if isinstance(audio_urls, str):
            audio_urls = [audio_urls]
        
        audio_keys = []
        for audio_url in audio_urls:
            _, audio_key = parse_s3_url(audio_url)
            audio_keys.append(audio_key)
        
        # Determine output bucket
        output_bucket = request.output_s3_bucket
        if not output_bucket:
            output_bucket = os.environ.get('AWS_BUCKET_NAME')
            if not output_bucket:
                raise ValueError("No output bucket specified and AWS_BUCKET_NAME not set")
        
        # Update progress
        await JobManager.update_job(job_id, {"progress": 30})
        
        # Call appropriate Modal function
        if generate_video_cuda is None or generate_multi_person_video is None:
            # Testing mode - simulate successful completion
            await asyncio.sleep(2)
            result = {
                "success": True,
                "s3_output": f"s3://{output_bucket}/{request.output_s3_prefix}test_video_{job_id}.mp4",
                "duration": 5.0,
                "frames": 81,
                "processing_time": 2.0
            }
        elif len(audio_keys) == 1:
            # Single person video
            result = await generate_video_cuda.remote.aio(
                prompt=request.prompt,
                image_key=image_key,
                audio_key=audio_keys[0],
                sample_steps=request.options.sample_steps
            )
        else:
            # Multi-person video
            result = await generate_multi_person_video.remote.aio(
                prompt=request.prompt,
                image_key=image_key,
                audio_keys=audio_keys,
                sample_steps=request.options.sample_steps,
                audio_type=request.options.audio_type,
                audio_cfg=request.options.audio_cfg,
                color_correction=request.options.color_correction
            )
        
        # Check if generation was successful
        if not result.get("success"):
            raise Exception(result.get("error", "Unknown error"))
        
        # Update job with result
        await JobManager.update_job(job_id, {
            "status": JobStatus.COMPLETED,
            "progress": 100,
            "result": result
        })
        
        # Send webhook if configured
        if request.webhook_url:
            webhook_payload = WebhookPayload(
                job_id=job_id,
                status=JobStatus.COMPLETED,
                timestamp=datetime.now(timezone.utc),
                result=result
            )
            await deliver_webhook(str(request.webhook_url), webhook_payload)
        
    except Exception as e:
        # Update job with error
        error_message = str(e)
        await JobManager.update_job(job_id, {
            "status": JobStatus.FAILED,
            "error": error_message
        })
        
        # Send failure webhook
        if request.webhook_url:
            webhook_payload = WebhookPayload(
                job_id=job_id,
                status=JobStatus.FAILED,
                timestamp=datetime.now(timezone.utc),
                error=error_message
            )
            await deliver_webhook(str(request.webhook_url), webhook_payload)
        
        print(f"Job {job_id} failed: {error_message}")

# ==================== Modal App Setup ====================

# Add FastAPI image requirements
api_image = modal.Image.debian_slim().pip_install(
    "fastapi[standard]",
    "pydantic",
    "boto3",
    "httpx",
    "python-multipart"
)

# Create Modal function for the API
@modal_app.function(
    image=api_image,
    secrets=[
        modal.Secret.from_name("aws-secret")
    ]
)
@modal.asgi_app()
def fastapi_app():
    """Modal function that serves the FastAPI app."""
    return web_app

# ==================== Main ====================

if __name__ == "__main__":
    # For local testing with uvicorn
    import uvicorn
    uvicorn.run(web_app, host="0.0.0.0", port=8000)