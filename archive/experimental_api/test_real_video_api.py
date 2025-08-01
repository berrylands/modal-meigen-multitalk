#!/usr/bin/env python3
"""
Test script to submit a real video generation request to the integrated API.
"""

import requests
import json
import time
import sys

# API configuration
API_KEY = "test-api-key"  # Update with your actual API key
BASE_URL = "https://berrylands--multitalk-integrated-fastapi-app.modal.run/api/v1"

# Your S3 files
IMAGE_S3_URL = "s3://760572149-framepack/multi1.png"
AUDIO_S3_URL = "s3://760572149-framepack/1.wav"

def submit_video_generation():
    """Submit a video generation request."""
    print("=== Submitting Real Video Generation Request ===")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "prompt": "A person speaking naturally with clear lip sync",
        "image_s3_url": IMAGE_S3_URL,
        "audio_s3_urls": AUDIO_S3_URL,
        "output_s3_bucket": "760572149-framepack",
        "output_s3_prefix": "outputs/",
        "options": {
            "sample_steps": 20,
            "resolution": "480p"
        }
    }
    
    print(f"Submitting to: {BASE_URL}/generate")
    print(f"Using image: {IMAGE_S3_URL}")
    print(f"Using audio: {AUDIO_S3_URL}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/generate",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        result = response.json()
        job_id = result['job_id']
        print(f"\n✓ Job submitted successfully!")
        print(f"Job ID: {job_id}")
        print(f"Status: {result['status']}")
        
        return job_id
        
    except requests.exceptions.RequestException as e:
        print(f"\n✗ Error submitting job: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        return None

def check_job_status(job_id):
    """Check the status of a job."""
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    
    try:
        response = requests.get(
            f"{BASE_URL}/jobs/{job_id}",
            headers=headers
        )
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"Error checking status: {e}")
        return None

def get_download_url(job_id):
    """Get download URL for completed job."""
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    
    try:
        response = requests.get(
            f"{BASE_URL}/jobs/{job_id}/download",
            headers=headers
        )
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"Error getting download URL: {e}")
        return None

def monitor_job(job_id, timeout=600):
    """Monitor job until completion."""
    print(f"\n=== Monitoring Job {job_id} ===")
    
    start_time = time.time()
    last_progress = -1
    
    while time.time() - start_time < timeout:
        status = check_job_status(job_id)
        
        if not status:
            print("\nFailed to get job status")
            return False
        
        current_progress = status.get('progress', 0)
        if current_progress != last_progress:
            print(f"\rProgress: {current_progress}% - Status: {status['status']}", end='', flush=True)
            last_progress = current_progress
        
        if status['status'] == 'completed':
            print(f"\n\n✓ Job completed successfully!")
            
            # Get download URL
            download_info = get_download_url(job_id)
            if download_info:
                print(f"\n=== Video Ready for Download ===")
                print(f"S3 URI: {download_info['s3_uri']}")
                print(f"Download URL: {download_info['download_url']}")
                print(f"Expires in: {download_info['expires_in']} seconds")
            
            return True
            
        elif status['status'] == 'failed':
            print(f"\n\n✗ Job failed!")
            print(f"Error: {status.get('error', 'Unknown error')}")
            return False
        
        time.sleep(5)  # Check every 5 seconds
    
    print(f"\n\nTimeout reached after {timeout} seconds")
    return False

def main():
    """Main test function."""
    print("MeiGen-MultiTalk Real Video Generation Test")
    print("=" * 50)
    
    # Submit job
    job_id = submit_video_generation()
    
    if not job_id:
        print("\nFailed to submit job. Exiting.")
        return 1
    
    # Monitor job
    success = monitor_job(job_id)
    
    if success:
        print("\n\n=== Test Completed Successfully ===")
        return 0
    else:
        print("\n\n=== Test Failed ===")
        return 1

if __name__ == "__main__":
    sys.exit(main())