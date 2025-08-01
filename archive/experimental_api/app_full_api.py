#!/usr/bin/env python3
"""
Full MeiGen-MultiTalk app with integrated REST API.
This combines the CUDA video generation with the FastAPI REST endpoints.
"""

import modal
import os
from pathlib import Path

# Create the app
app = modal.App("multitalk-full")

# ==================== Shared Imports ====================

# Import all the API code at the top level
from api import (
    web_app, VideoGenerationRequest, JobManager, JobStatus, 
    parse_s3_url, deliver_webhook, WebhookPayload,
    process_video_generation
)

# ==================== Image Definition ====================

# Combined image with both CUDA and API dependencies
full_image = (
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

# ==================== Mount Files ====================

# Mount the necessary files
code_files = modal.mount.from_local_python_packages("app_multitalk_cuda", "api", "s3_utils")

# ==================== Video Generation Functions ====================

@app.function(
    image=full_image,
    gpu="a10g",
    memory=32768,
    timeout=900,
    secrets=[modal.Secret.from_name("aws-secret")],
    mounts=[code_files],
)
def generate_video_cuda(*args, **kwargs):
    """CUDA video generation function."""
    import sys
    sys.path.append("/root")
    from app_multitalk_cuda import generate_video_cuda as cuda_fn
    return cuda_fn.local(*args, **kwargs)

@app.function(
    image=full_image,
    gpu="a10g", 
    memory=32768,
    timeout=900,
    secrets=[modal.Secret.from_name("aws-secret")],
    mounts=[code_files],
)
def generate_multi_person_video(*args, **kwargs):
    """Multi-person CUDA video generation function."""
    import sys
    sys.path.append("/root")
    from app_multitalk_cuda import generate_multi_person_video as cuda_fn
    return cuda_fn.local(*args, **kwargs)

# ==================== Custom Process Function ====================

async def process_video_generation_integrated(job_id: str, request: VideoGenerationRequest):
    """Process video generation job with direct function calls."""
    from datetime import datetime, timezone
    
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

# ==================== FastAPI App ====================

@app.function(
    image=full_image,
    secrets=[modal.Secret.from_name("aws-secret")],
    mounts=[code_files],
    container_idle_timeout=300,
)
@modal.asgi_app()
def fastapi_app():
    """Serve the FastAPI app with integrated video generation."""
    import sys
    sys.path.append("/root")
    
    # Import API and replace the process function
    import api
    api.process_video_generation = process_video_generation_integrated
    
    # Import the CUDA functions to make them available
    global generate_video_cuda, generate_multi_person_video
    
    return web_app

if __name__ == "__main__":
    print("Deploy this app with: modal deploy app_full_api.py")