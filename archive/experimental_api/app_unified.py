#!/usr/bin/env python3
"""
Unified Modal app that combines the REST API with CUDA video generation.
This ensures the API can call the video generation functions directly.
"""

import modal
from pathlib import Path

# Create the unified app
app = modal.App("multitalk-unified")

# ==================== Image Definitions ====================

# CUDA image for video generation
cuda_image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:24.06-py3", add_python="3.11")
    .apt_install("ffmpeg", "libsm6", "libxext6", "libgl1", "wget")
    .pip_install(
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
        "numba"
    )
    .run_commands(
        # Install flash-attn with proper CUDA architecture settings
        "pip install ninja",
        "export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE && "
        "export MAX_JOBS=4 && "
        "export TORCH_CUDA_ARCH_LIST='8.0;8.6;8.9;9.0' && "
        "pip install flash-attn==2.6.1 --no-build-isolation",
        gpu="a10g"
    )
)

# API image with FastAPI
api_image = modal.Image.debian_slim().pip_install(
    "fastapi[standard]",
    "pydantic",
    "boto3",
    "httpx",
    "python-multipart"
)

# ==================== Mount the code ====================

# Mount all necessary Python files
mounts = [
    modal.Mount.from_local_file("app_multitalk_cuda.py"),
    modal.Mount.from_local_file("api.py"),
    modal.Mount.from_local_file("s3_utils.py"),
]

# ==================== Import the functions ====================

# Import CUDA functions
@app.function(
    image=cuda_image,
    gpu="a10g",
    memory=32768,
    timeout=900,
    secrets=[modal.Secret.from_name("aws-secret")],
    mounts=mounts,
)
def generate_video_cuda(*args, **kwargs):
    """Wrapper for CUDA video generation."""
    from app_multitalk_cuda import generate_video_cuda as cuda_fn
    return cuda_fn.local(*args, **kwargs)

@app.function(
    image=cuda_image,
    gpu="a10g",
    memory=32768,
    timeout=900,
    secrets=[modal.Secret.from_name("aws-secret")],
    mounts=mounts,
)
def generate_multi_person_video(*args, **kwargs):
    """Wrapper for multi-person CUDA video generation."""
    from app_multitalk_cuda import generate_multi_person_video as cuda_fn
    return cuda_fn.local(*args, **kwargs)

# ==================== API Implementation ====================

# Import the API components and modify to use local functions
@app.function(
    image=api_image,
    secrets=[modal.Secret.from_name("aws-secret")],
    mounts=mounts,
)
@modal.asgi_app()
def fastapi_app():
    """Serve the FastAPI app with integrated CUDA functions."""
    import sys
    sys.path.append("/root")
    
    # Import and modify the API to use our wrapped functions
    from api import web_app, process_video_generation
    import api
    
    # Monkey patch the process_video_generation to use our functions
    async def patched_process_video_generation(job_id: str, request):
        """Process video generation job with integrated CUDA functions."""
        try:
            # Import necessary modules
            from api import JobManager, parse_s3_url, JobStatus
            import os
            from datetime import datetime, timezone
            
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
            
            # Call appropriate function (using our wrapped versions)
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
                from api import deliver_webhook, WebhookPayload
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
                from api import deliver_webhook, WebhookPayload
                webhook_payload = WebhookPayload(
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    timestamp=datetime.now(timezone.utc),
                    error=error_message
                )
                await deliver_webhook(str(request.webhook_url), webhook_payload)
            
            print(f"Job {job_id} failed: {error_message}")
    
    # Replace the original function
    api.process_video_generation = patched_process_video_generation
    
    return web_app

if __name__ == "__main__":
    print("Deploy this unified app with: modal deploy app_unified.py")