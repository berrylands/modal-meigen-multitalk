#!/usr/bin/env python3
"""
Submit real video generation requests using your actual S3 files.
Update the BUCKET_NAME with your actual bucket name.
"""

import requests
import json
import time
import os

# Configuration
API_URL = "https://berrylands--multitalk-api-fastapi-app.modal.run/api/v1"
API_KEY = "test-api-key"  # In production, use a secure API key

# UPDATE THIS WITH YOUR ACTUAL BUCKET NAME
BUCKET_NAME = "your-bucket-name"  # <-- REPLACE THIS

def submit_single_person_video():
    """Submit a single-person video generation request."""
    print("=== Submitting Single-Person Video Request ===")
    
    request_data = {
        "prompt": "A person speaking naturally with clear lip sync",
        "image_s3_url": f"s3://{BUCKET_NAME}/multi1.png",
        "audio_s3_urls": f"s3://{BUCKET_NAME}/1.wav",
        "options": {
            "sample_steps": 20,
            "resolution": "480p",
            "audio_cfg": 4.0
        }
    }
    
    response = requests.post(
        f"{API_URL}/generate",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json=request_data
    )
    
    if response.status_code == 200:
        job_data = response.json()
        print(f"✓ Job submitted successfully!")
        print(f"  Job ID: {job_data['job_id']}")
        return job_data['job_id']
    else:
        print(f"✗ Failed to submit job: {response.status_code}")
        print(f"  Response: {response.text}")
        return None

def submit_multi_person_video():
    """Submit a multi-person conversation video generation request."""
    print("\n=== Submitting Multi-Person Video Request ===")
    
    request_data = {
        "prompt": "Two people having a natural conversation",
        "image_s3_url": f"s3://{BUCKET_NAME}/multi1.png",
        "audio_s3_urls": [
            f"s3://{BUCKET_NAME}/1.wav",
            f"s3://{BUCKET_NAME}/2.wav"
        ],
        "options": {
            "audio_type": "add",  # Sequential speaking
            "sample_steps": 20,
            "resolution": "480p",
            "audio_cfg": 4.0,
            "color_correction": 0.7
        }
    }
    
    response = requests.post(
        f"{API_URL}/generate",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json=request_data
    )
    
    if response.status_code == 200:
        job_data = response.json()
        print(f"✓ Job submitted successfully!")
        print(f"  Job ID: {job_data['job_id']}")
        return job_data['job_id']
    else:
        print(f"✗ Failed to submit job: {response.status_code}")
        print(f"  Response: {response.text}")
        return None

def check_job_status(job_id):
    """Check the status of a job."""
    response = requests.get(
        f"{API_URL}/jobs/{job_id}",
        headers={"Authorization": f"Bearer {API_KEY}"}
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to get job status: {response.status_code}")
        return None

def wait_for_completion(job_id, max_wait=300):
    """Wait for a job to complete."""
    print(f"\nWaiting for job {job_id} to complete...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        status_data = check_job_status(job_id)
        if not status_data:
            return None
        
        status = status_data['status']
        progress = status_data.get('progress', 0)
        
        print(f"  Status: {status}, Progress: {progress}%")
        
        if status == 'completed':
            print("✓ Job completed successfully!")
            return status_data
        elif status in ['failed', 'cancelled']:
            print(f"✗ Job {status}: {status_data.get('error', 'Unknown error')}")
            return None
        
        time.sleep(10)  # Check every 10 seconds
    
    print("✗ Timeout waiting for job completion")
    return None

def get_download_url(job_id):
    """Get download URL for completed video."""
    response = requests.get(
        f"{API_URL}/jobs/{job_id}/download",
        headers={"Authorization": f"Bearer {API_KEY}"},
        params={"expiration": 3600}  # 1 hour expiration
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to get download URL: {response.status_code}")
        return None

def main():
    """Main function to run the video generation examples."""
    
    if BUCKET_NAME == "your-bucket-name":
        print("ERROR: Please update BUCKET_NAME in this script with your actual S3 bucket name!")
        print("Look for the line: BUCKET_NAME = \"your-bucket-name\"")
        return
    
    print(f"Using S3 bucket: {BUCKET_NAME}")
    print(f"API URL: {API_URL}")
    print("-" * 50)
    
    # Submit single-person video
    single_job_id = submit_single_person_video()
    
    # Submit multi-person video
    multi_job_id = submit_multi_person_video()
    
    # Wait for and download results
    jobs = [
        ("Single-Person", single_job_id),
        ("Multi-Person", multi_job_id)
    ]
    
    for job_type, job_id in jobs:
        if job_id:
            print(f"\n{'='*50}")
            print(f"Processing {job_type} Video (Job ID: {job_id})")
            print(f"{'='*50}")
            
            # Wait for completion
            result = wait_for_completion(job_id)
            
            if result and result['status'] == 'completed':
                # Get download URL
                download_data = get_download_url(job_id)
                
                if download_data:
                    print(f"\n✓ {job_type} Video Ready!")
                    print(f"  S3 Location: {download_data['s3_uri']}")
                    print(f"  Download URL: {download_data['download_url']}")
                    print(f"  Expires in: {download_data['expires_in']} seconds")
                    
                    # You could download the file here if needed
                    # response = requests.get(download_data['download_url'])
                    # with open(f"{job_type.lower()}_video.mp4", 'wb') as f:
                    #     f.write(response.content)

if __name__ == "__main__":
    main()