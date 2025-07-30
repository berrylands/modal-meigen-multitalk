#!/usr/bin/env python3
"""
Practical examples of using the MeiGen-MultiTalk REST API.
These examples demonstrate common use cases and patterns.
"""

import os
import time
import json
import requests
from typing import List, Dict, Optional
from datetime import datetime

# Configuration
API_URL = os.environ.get('MULTITALK_API_URL', 'https://your-modal-app.modal.run/api/v1')
API_KEY = os.environ.get('MULTITALK_API_KEY', 'your-api-key')


class MultiTalkAPI:
    """Enhanced client with practical examples."""
    
    def __init__(self, api_url: str = API_URL, api_key: str = API_KEY):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make API request with error handling."""
        url = f"{self.api_url}{endpoint}"
        response = self.session.request(method, url, **kwargs)
        
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            error_data = response.json() if response.content else {}
            print(f"API Error: {error_data.get('error', str(e))}")
            print(f"Details: {error_data.get('detail', 'No details available')}")
            raise
        
        return response.json()
    
    # ========== Example 1: Basic Single Person Video ==========
    
    def example_single_person_video(self):
        """Generate a simple single-person talking video."""
        print("\n=== Example 1: Single Person Video ===")
        
        job = self._request('POST', '/generate', json={
            'prompt': 'Professional presenter explaining product features',
            'image_s3_url': 's3://my-bucket/images/presenter.png',
            'audio_s3_urls': 's3://my-bucket/audio/product-demo.wav',
            'options': {
                'sample_steps': 20,
                'resolution': '480p'
            }
        })
        
        print(f"Job submitted: {job['job_id']}")
        print(f"Status: {job['status']}")
        
        # Wait for completion
        result = self.wait_for_job(job['job_id'])
        print(f"Video ready: {result['result']['s3_output']}")
        
        return result
    
    # ========== Example 2: Multi-Person Conversation ==========
    
    def example_multi_person_conversation(self):
        """Generate a two-person conversation video."""
        print("\n=== Example 2: Multi-Person Conversation ===")
        
        job = self._request('POST', '/generate', json={
            'prompt': 'Two experts discussing artificial intelligence',
            'image_s3_url': 's3://my-bucket/images/two-experts.png',
            'audio_s3_urls': [
                's3://my-bucket/audio/expert1-part1.wav',
                's3://my-bucket/audio/expert2-response.wav',
                's3://my-bucket/audio/expert1-part2.wav'
            ],
            'options': {
                'audio_type': 'add',  # Sequential speaking
                'color_correction': 0.7,
                'sample_steps': 25
            }
        })
        
        print(f"Job submitted: {job['job_id']}")
        
        # Poll with progress updates
        result = self.wait_for_job_with_progress(job['job_id'])
        print(f"Video ready: {result['result']['s3_output']}")
        print(f"Processing time: {result['result']['processing_time']}s")
        
        return result
    
    # ========== Example 3: Batch Processing ==========
    
    def example_batch_processing(self, image_audio_pairs: List[Dict[str, str]]):
        """Process multiple videos in parallel."""
        print("\n=== Example 3: Batch Processing ===")
        
        jobs = []
        
        # Submit all jobs
        for i, pair in enumerate(image_audio_pairs):
            job = self._request('POST', '/generate', json={
                'prompt': f'Speaker {i+1} presentation',
                'image_s3_url': pair['image'],
                'audio_s3_urls': pair['audio'],
                'output_s3_prefix': f'batch/{datetime.now().strftime("%Y%m%d")}/speaker{i+1}/'
            })
            
            jobs.append({
                'job_id': job['job_id'],
                'index': i,
                'submitted_at': datetime.now()
            })
            
            print(f"Submitted job {i+1}/{len(image_audio_pairs)}: {job['job_id']}")
        
        # Wait for all to complete
        results = []
        for job_info in jobs:
            try:
                result = self.wait_for_job(job_info['job_id'])
                results.append({
                    'index': job_info['index'],
                    'job_id': job_info['job_id'],
                    'output': result['result']['s3_output'],
                    'duration': (datetime.now() - job_info['submitted_at']).total_seconds()
                })
                print(f"✓ Job {job_info['index']+1} completed")
            except Exception as e:
                print(f"✗ Job {job_info['index']+1} failed: {e}")
                results.append({
                    'index': job_info['index'],
                    'job_id': job_info['job_id'],
                    'error': str(e)
                })
        
        return results
    
    # ========== Example 4: Webhook Integration ==========
    
    def example_with_webhook(self, webhook_url: str):
        """Submit job with webhook notification."""
        print("\n=== Example 4: Webhook Integration ===")
        
        job = self._request('POST', '/generate', json={
            'prompt': 'CEO announcing quarterly results',
            'image_s3_url': 's3://my-bucket/images/ceo.png',
            'audio_s3_urls': 's3://my-bucket/audio/quarterly-announcement.wav',
            'webhook_url': webhook_url,
            'options': {
                'resolution': '720p',
                'sample_steps': 30
            }
        })
        
        print(f"Job submitted with webhook: {job['job_id']}")
        print(f"Webhook will be called at: {webhook_url}")
        print("You can now process other tasks while waiting for the webhook...")
        
        return job
    
    # ========== Example 5: Error Handling and Retries ==========
    
    def example_with_retry(self, max_retries: int = 3):
        """Demonstrate robust error handling and retry logic."""
        print("\n=== Example 5: Error Handling with Retries ===")
        
        retry_count = 0
        backoff_seconds = 5
        
        while retry_count < max_retries:
            try:
                job = self._request('POST', '/generate', json={
                    'prompt': 'Training video narrator',
                    'image_s3_url': 's3://my-bucket/images/narrator.png',
                    'audio_s3_urls': 's3://my-bucket/audio/training-script.wav'
                })
                
                print(f"Job submitted successfully: {job['job_id']}")
                
                # Wait with timeout
                result = self.wait_for_job(job['job_id'], timeout=300)
                print(f"Video completed: {result['result']['s3_output']}")
                
                return result
                
            except requests.HTTPError as e:
                if e.response.status_code == 429:  # Rate limited
                    retry_count += 1
                    wait_time = backoff_seconds * (2 ** (retry_count - 1))
                    print(f"Rate limited. Waiting {wait_time}s before retry {retry_count}/{max_retries}")
                    time.sleep(wait_time)
                elif e.response.status_code >= 500:  # Server error
                    retry_count += 1
                    print(f"Server error. Retry {retry_count}/{max_retries}")
                    time.sleep(backoff_seconds)
                else:
                    # Client error, don't retry
                    raise
            except Exception as e:
                print(f"Unexpected error: {e}")
                raise
        
        raise Exception(f"Failed after {max_retries} retries")
    
    # ========== Example 6: Download and Local Save ==========
    
    def example_download_video(self, job_id: str, local_path: str):
        """Download completed video to local file."""
        print("\n=== Example 6: Download Video ===")
        
        # Get download URL
        download_info = self._request('GET', f'/jobs/{job_id}/download', 
                                    params={'expiration': 7200})
        
        print(f"Download URL obtained, expires in {download_info['expires_in']}s")
        
        # Download file
        response = requests.get(download_info['download_url'], stream=True)
        response.raise_for_status()
        
        # Save to local file
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        file_size = os.path.getsize(local_path) / (1024 * 1024)  # MB
        print(f"Video saved to: {local_path} ({file_size:.2f} MB)")
        
        return local_path
    
    # ========== Helper Methods ==========
    
    def wait_for_job(self, job_id: str, timeout: int = 600, 
                     poll_interval: int = 10) -> Dict:
        """Wait for job completion with timeout."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self._request('GET', f'/jobs/{job_id}')
            
            if status['status'] == 'completed':
                return status
            elif status['status'] in ['failed', 'cancelled']:
                raise Exception(f"Job {status['status']}: {status.get('error', 'Unknown error')}")
            
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Job {job_id} timed out after {timeout}s")
    
    def wait_for_job_with_progress(self, job_id: str, timeout: int = 600) -> Dict:
        """Wait for job with progress updates."""
        start_time = time.time()
        last_progress = -1
        
        while time.time() - start_time < timeout:
            status = self._request('GET', f'/jobs/{job_id}')
            
            # Show progress updates
            progress = status.get('progress', 0)
            if progress != last_progress:
                print(f"Progress: {progress}%")
                last_progress = progress
            
            if status['status'] == 'completed':
                print("Progress: 100%")
                return status
            elif status['status'] in ['failed', 'cancelled']:
                raise Exception(f"Job {status['status']}: {status.get('error')}")
            
            time.sleep(5)
        
        raise TimeoutError(f"Job {job_id} timed out")
    
    def list_recent_jobs(self, limit: int = 10) -> List[Dict]:
        """List recent jobs."""
        response = self._request('GET', '/jobs', params={'limit': limit})
        return response['jobs']
    
    def cancel_job(self, job_id: str) -> Dict:
        """Cancel a pending or processing job."""
        return self._request('DELETE', f'/jobs/{job_id}')


# ========== Main Examples Runner ==========

def main():
    """Run all examples."""
    
    # Initialize client
    client = MultiTalkAPI()
    
    # Check API health
    try:
        health = client._request('GET', '/health')
        print(f"API Status: {health['status']} (v{health['version']})")
    except Exception as e:
        print(f"API appears to be down: {e}")
        return
    
    # Run examples (comment out any you don't want to run)
    
    # 1. Single person video
    try:
        client.example_single_person_video()
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    # 2. Multi-person conversation
    try:
        client.example_multi_person_conversation()
    except Exception as e:
        print(f"Example 2 failed: {e}")
    
    # 3. Batch processing
    try:
        batch_data = [
            {
                'image': 's3://my-bucket/speakers/person1.png',
                'audio': 's3://my-bucket/audio/speech1.wav'
            },
            {
                'image': 's3://my-bucket/speakers/person2.png',
                'audio': 's3://my-bucket/audio/speech2.wav'
            }
        ]
        results = client.example_batch_processing(batch_data)
        print(f"\nBatch processing completed: {len(results)} videos")
    except Exception as e:
        print(f"Example 3 failed: {e}")
    
    # 4. Webhook example (requires public webhook URL)
    try:
        webhook_url = os.environ.get('WEBHOOK_URL')
        if webhook_url:
            client.example_with_webhook(webhook_url)
        else:
            print("\nSkipping webhook example (set WEBHOOK_URL env var)")
    except Exception as e:
        print(f"Example 4 failed: {e}")
    
    # 5. Error handling with retries
    try:
        client.example_with_retry()
    except Exception as e:
        print(f"Example 5 failed: {e}")
    
    # 6. List recent jobs
    try:
        print("\n=== Recent Jobs ===")
        jobs = client.list_recent_jobs(5)
        for job in jobs:
            print(f"- {job['job_id']}: {job['status']} (created: {job['created_at']})")
    except Exception as e:
        print(f"Failed to list jobs: {e}")


if __name__ == '__main__':
    main()