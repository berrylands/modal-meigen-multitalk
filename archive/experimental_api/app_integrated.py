#!/usr/bin/env python3
"""
Integrated MeiGen-MultiTalk app with REST API and CUDA video generation.
This single app contains both the API endpoints and video generation functions.
"""

import modal
import os
import sys
import uuid
import json
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union
from enum import Enum
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
import httpx
from fastapi import FastAPI, HTTPException, Depends, Security, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, HttpUrl, validator

# Create the integrated app
app = modal.App("multitalk-integrated")

# ==================== Configuration ====================

API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"
MAX_AUDIO_FILES = 5
WEBHOOK_RETRY_ATTEMPTS = 3
WEBHOOK_TIMEOUT = 10

# ==================== Image Definition ====================

# Combined image with all dependencies
integrated_image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:24.06-py3", add_python="3.11")
    .apt_install("ffmpeg", "libsm6", "libxext6", "libgl1", "wget")
    .pip_install(
        # CUDA dependencies
        "torch",
        "torchvision", 
        "transformers",
        "accelerate",
        "diffusers",
        "opencv-python",
        "pillow",
        "numpy<2",
        "av",
        "xformers",
        "spandrel",
        "soundfile",
        "einops",
        "torchao",
        "uuid",
        "scikit-image",
        "scipy",
        "imageio",
        "boto3",
        "botocore",
        "numba",
        # API dependencies
        "fastapi[standard]",
        "pydantic",
        "httpx",
        "python-multipart"
    )
    .run_commands(
        # Install flash-attn
        "pip install ninja",
        "export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE && "
        "export MAX_JOBS=4 && "
        "export TORCH_CUDA_ARCH_LIST='8.0;8.6;8.9;9.0' && "
        "pip install flash-attn==2.6.1 --no-build-isolation",
        gpu="a10g"
    )
)

# ==================== Add Local Files ====================

# Include the CUDA app code and utilities in the image
integrated_image = integrated_image.add_local_file("app_multitalk_cuda.py", "/root/app_multitalk_cuda.py")
integrated_image = integrated_image.add_local_file("s3_utils.py", "/root/s3_utils.py")

# ==================== Shared Models (from api.py) ====================

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AudioType(str, Enum):
    ADD = "add"
    PARA = "para"

class VideoResolution(str, Enum):
    RES_480P = "480p"
    RES_720P = "720p"

class GenerationOptions(BaseModel):
    resolution: VideoResolution = Field(default=VideoResolution.RES_480P)
    sample_steps: int = Field(default=20, ge=10, le=50)
    audio_type: AudioType = Field(default=AudioType.ADD)
    audio_cfg: float = Field(default=4.0, ge=3.0, le=5.0)
    color_correction: float = Field(default=0.7, ge=0.0, le=1.0)

class VideoGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text description of the video content")
    image_s3_url: str = Field(..., description="S3 URL for reference image")
    audio_s3_urls: Union[str, List[str]] = Field(..., description="S3 URL(s) for audio file(s)")
    output_s3_bucket: Optional[str] = Field(None)
    output_s3_prefix: str = Field(default="outputs/")
    webhook_url: Optional[HttpUrl] = Field(None)
    options: GenerationOptions = Field(default_factory=GenerationOptions)
    
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
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    created_at: datetime = Field(..., description="Job creation timestamp")
    message: str = Field(..., description="Status message")

class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    progress: Optional[int] = Field(None, ge=0, le=100)
    result: Optional[Dict] = Field(None)
    error: Optional[str] = Field(None)
    metadata: Optional[Dict] = Field(None)

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None)
    request_id: Optional[str] = Field(None)

class HealthResponse(BaseModel):
    status: str = Field(default="healthy")
    version: str = Field(default=API_VERSION)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class WebhookPayload(BaseModel):
    job_id: str
    status: JobStatus
    timestamp: datetime
    result: Optional[Dict] = None
    error: Optional[str] = None

# ==================== Job Storage ====================

jobs_db = modal.Dict.from_name("multitalk-jobs", create_if_missing=True)
job_ids_db = modal.Dict.from_name("multitalk-job-ids", create_if_missing=True)

class JobManager:
    @staticmethod
    async def create_job(request_data: dict) -> str:
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
        
        # Track job ID
        job_ids_list = job_ids_db.get("job_ids", [])
        job_ids_list.append({
            "id": job_id,
            "created_at": job_data["created_at"]
        })
        if len(job_ids_list) > 1000:
            job_ids_list = job_ids_list[-1000:]
        job_ids_db["job_ids"] = job_ids_list
        
        return job_id
    
    @staticmethod
    async def get_job(job_id: str) -> Optional[dict]:
        return jobs_db.get(job_id)
    
    @staticmethod
    async def update_job(job_id: str, updates: dict) -> bool:
        job = jobs_db.get(job_id)
        if not job:
            return False
        
        job.update(updates)
        job["updated_at"] = datetime.now(timezone.utc).isoformat()
        jobs_db[job_id] = job
        return True
    
    @staticmethod
    async def list_jobs(limit: int = 100) -> List[dict]:
        job_ids_list = job_ids_db.get("job_ids", [])
        job_ids_list.sort(key=lambda x: x["created_at"], reverse=True)
        job_ids_list = job_ids_list[:limit]
        
        all_jobs = []
        for job_info in job_ids_list:
            job = jobs_db.get(job_info["id"])
            if job:
                all_jobs.append(job)
        
        return all_jobs

# ==================== CUDA Video Generation Functions ====================

@app.function(
    image=integrated_image,
    gpu="a10g",
    memory=32768,
    timeout=900,
    secrets=[modal.Secret.from_name("aws-secret")],
)
def generate_video_cuda(
    prompt: str,
    image_key: str,
    audio_key: str,
    sample_steps: int = 20,
    output_prefix: str = "multitalk_single"
):
    """CUDA video generation function for single person."""
    sys.path.append("/root")
    from app_multitalk_cuda import generate_video_cuda as cuda_fn
    return cuda_fn.local(prompt, image_key, audio_key, sample_steps, output_prefix)

@app.function(
    image=integrated_image,
    gpu="a10g",
    memory=32768,
    timeout=900,
    secrets=[modal.Secret.from_name("aws-secret")],
)
def generate_multi_person_video(
    prompt: str,
    image_key: str,
    audio_keys: List[str],
    sample_steps: int = 20,
    audio_type: str = "add",
    audio_cfg: float = 4.0,
    color_correction: float = 0.7,
    output_prefix: str = "multitalk_multi"
):
    """CUDA video generation function for multiple people."""
    sys.path.append("/root")
    from app_multitalk_cuda import generate_multi_person_video as cuda_fn
    return cuda_fn.local(prompt, image_key, audio_keys, sample_steps, 
                        audio_type, audio_cfg, color_correction, output_prefix)

# ==================== API Utilities ====================

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
                await asyncio.sleep(2 ** attempt)
                return await deliver_webhook(webhook_url, payload, attempt + 1)
            
            return False

# ==================== Authentication ====================

security = HTTPBearer()

def get_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """Validate API key from Bearer token."""
    api_key = credentials.credentials
    
    # Get valid API keys from environment
    api_keys_env = os.environ.get("API_KEYS", "")
    
    if not api_keys_env:
        print("Warning: API_KEYS not configured, accepting any key")
        return api_key
    
    valid_api_keys = [key.strip() for key in api_keys_env.split(",") if key.strip()]
    
    if api_key not in valid_api_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return api_key

# ==================== Background Processing ====================

async def process_video_generation_integrated(job_id: str, request: VideoGenerationRequest):
    """Process video generation job with integrated CUDA functions."""
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
        if len(audio_keys) == 1:
            # Single person video
            result = await generate_video_cuda.remote.aio(
                prompt=request.prompt,
                image_key=image_key,
                audio_key=audio_keys[0],
                sample_steps=request.options.sample_steps,
                output_prefix=f"api_job_{job_id}"
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
                color_correction=request.options.color_correction,
                output_prefix=f"api_job_{job_id}"
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
    summary="Submit video generation job"
)
async def generate_video(
    request: VideoGenerationRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key)
):
    """Submit a video generation job."""
    # Create job
    job_id = await JobManager.create_job(request.dict())
    
    # Schedule background processing
    background_tasks.add_task(process_video_generation_integrated, job_id, request)
    
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        created_at=datetime.now(timezone.utc),
        message="Job submitted successfully"
    )

@web_app.get(
    f"{API_PREFIX}/jobs/{{job_id}}",
    response_model=JobStatusResponse,
    tags=["Job Management"]
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
    tags=["Job Management"]
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
    tags=["Job Management"]
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
    tags=["Job Management"]
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
    tags=["Testing"]
)
async def test_webhook(
    webhook_url: HttpUrl,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key)
):
    """Test webhook delivery."""
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

# ==================== Modal App Function ====================

@app.function(
    image=integrated_image,
    gpu="a10g",
    memory=32768,
    secrets=[modal.Secret.from_name("aws-secret")],
    container_idle_timeout=300,
)
@modal.asgi_app()
def fastapi_app():
    """Serve the FastAPI app with integrated video generation."""
    sys.path.append("/root")
    return web_app

if __name__ == "__main__":
    print("Deploy this integrated app with: modal deploy app_integrated.py")