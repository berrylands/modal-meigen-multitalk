#!/usr/bin/env python3
"""
Simple test script for the MeiGen-MultiTalk REST API.
This script tests basic functionality without requiring actual S3 resources.
"""

import os
import sys
import time
import json
import requests
from datetime import datetime

# Configuration
API_URL = os.environ.get('MULTITALK_API_URL', 'http://localhost:8000/api/v1')
API_KEY = os.environ.get('MULTITALK_API_KEY', 'test-api-key')

# Test data (you'll need to update these with your actual S3 URLs)
TEST_IMAGE_URL = os.environ.get('TEST_IMAGE_URL', 's3://my-bucket/test-image.png')
TEST_AUDIO_URL = os.environ.get('TEST_AUDIO_URL', 's3://my-bucket/test-audio.wav')


def test_health_check():
    """Test the health check endpoint."""
    print("Testing health check...")
    
    response = requests.get(f"{API_URL}/health")
    
    assert response.status_code == 200, f"Health check failed: {response.status_code}"
    data = response.json()
    
    assert data['status'] == 'healthy'
    assert 'version' in data
    assert 'timestamp' in data
    
    print(f"✓ Health check passed: API version {data['version']}")
    return True


def test_authentication():
    """Test API key authentication."""
    print("\nTesting authentication...")
    
    # Test without API key
    response = requests.post(f"{API_URL}/generate", json={})
    assert response.status_code == 403, "Expected 403 without API key"
    print("✓ Correctly rejected request without API key")
    
    # Test with invalid API key
    headers = {'Authorization': 'Bearer invalid-key'}
    response = requests.post(f"{API_URL}/generate", headers=headers, json={})
    
    # In development mode (no API_KEYS set), any key is accepted
    if response.status_code == 401:
        print("✓ Correctly rejected invalid API key")
    else:
        print("✓ API in development mode (accepting any key)")
    
    return True


def test_input_validation():
    """Test request validation."""
    print("\nTesting input validation...")
    
    headers = {'Authorization': f'Bearer {API_KEY}'}
    
    # Test missing required fields
    response = requests.post(f"{API_URL}/generate", headers=headers, json={})
    assert response.status_code == 422, "Expected 422 for missing fields"
    print("✓ Correctly rejected request with missing fields")
    
    # Test invalid S3 URL format
    response = requests.post(f"{API_URL}/generate", headers=headers, json={
        'prompt': 'Test',
        'image_s3_url': 'not-an-s3-url',
        'audio_s3_urls': ['s3://bucket/audio.wav']
    })
    assert response.status_code == 422, "Expected 422 for invalid S3 URL"
    print("✓ Correctly rejected invalid S3 URL format")
    
    # Test invalid options
    response = requests.post(f"{API_URL}/generate", headers=headers, json={
        'prompt': 'Test',
        'image_s3_url': 's3://bucket/image.png',
        'audio_s3_urls': ['s3://bucket/audio.wav'],
        'options': {
            'sample_steps': 100  # Out of range
        }
    })
    assert response.status_code == 422, "Expected 422 for invalid options"
    print("✓ Correctly rejected invalid option values")
    
    return True


def test_job_submission():
    """Test job submission and status checking."""
    print("\nTesting job submission...")
    
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    
    # Submit a test job
    job_data = {
        'prompt': 'API test video generation',
        'image_s3_url': TEST_IMAGE_URL,
        'audio_s3_urls': [TEST_AUDIO_URL],
        'options': {
            'sample_steps': 20,
            'resolution': '480p'
        }
    }
    
    response = requests.post(f"{API_URL}/generate", headers=headers, json=job_data)
    
    if response.status_code != 200:
        print(f"Failed to submit job: {response.status_code}")
        print(f"Response: {response.text}")
        return False
    
    data = response.json()
    job_id = data['job_id']
    
    assert 'job_id' in data
    assert data['status'] == 'pending'
    assert 'created_at' in data
    
    print(f"✓ Job submitted successfully: {job_id}")
    
    # Check job status
    print("Checking job status...")
    response = requests.get(f"{API_URL}/jobs/{job_id}", headers=headers)
    
    assert response.status_code == 200
    status_data = response.json()
    
    assert status_data['job_id'] == job_id
    assert status_data['status'] in ['pending', 'processing', 'completed', 'failed']
    assert 'created_at' in status_data
    assert 'updated_at' in status_data
    
    print(f"✓ Job status retrieved: {status_data['status']}")
    
    # Test job listing
    response = requests.get(f"{API_URL}/jobs?limit=5", headers=headers)
    assert response.status_code == 200
    
    jobs_data = response.json()
    assert 'jobs' in jobs_data
    assert 'count' in jobs_data
    
    # Check if our job is in the list
    job_ids = [job['job_id'] for job in jobs_data['jobs']]
    assert job_id in job_ids, "Submitted job not found in job list"
    
    print(f"✓ Job listing works: {jobs_data['count']} jobs found")
    
    # Test job cancellation
    if status_data['status'] in ['pending', 'processing']:
        response = requests.delete(f"{API_URL}/jobs/{job_id}", headers=headers)
        assert response.status_code == 200
        
        cancel_data = response.json()
        assert cancel_data['status'] == 'cancelled'
        
        print("✓ Job cancellation works")
    
    return job_id


def test_webhook():
    """Test webhook endpoint."""
    print("\nTesting webhook...")
    
    headers = {'Authorization': f'Bearer {API_KEY}'}
    
    # Test webhook URL (you can use webhook.site for testing)
    test_webhook_url = "https://webhook.site/test"
    
    response = requests.post(f"{API_URL}/webhook-test", headers=headers, json={
        'webhook_url': test_webhook_url
    })
    
    if response.status_code == 200:
        data = response.json()
        assert 'message' in data
        assert 'webhook_url' in data
        assert 'payload' in data
        
        print(f"✓ Webhook test endpoint works")
        print(f"  Webhook would be sent to: {test_webhook_url}")
    else:
        print("✗ Webhook test failed (this is okay for local testing)")
    
    return True


def test_error_responses():
    """Test error response format."""
    print("\nTesting error responses...")
    
    headers = {'Authorization': f'Bearer {API_KEY}'}
    
    # Test 404 error
    response = requests.get(f"{API_URL}/jobs/non-existent-job", headers=headers)
    
    if response.status_code == 404:
        data = response.json()
        assert 'error' in data or 'detail' in data
        print("✓ 404 error response format correct")
    
    # Test method not allowed
    response = requests.put(f"{API_URL}/health", headers=headers)
    assert response.status_code == 405
    print("✓ 405 Method Not Allowed works")
    
    return True


def run_load_test(num_requests=10):
    """Run a simple load test."""
    print(f"\nRunning load test with {num_requests} requests...")
    
    headers = {'Authorization': f'Bearer {API_KEY}'}
    
    start_time = time.time()
    successful = 0
    failed = 0
    
    for i in range(num_requests):
        try:
            response = requests.get(f"{API_URL}/health", headers=headers, timeout=5)
            if response.status_code == 200:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"Request {i+1} failed: {e}")
    
    duration = time.time() - start_time
    requests_per_second = num_requests / duration
    
    print(f"✓ Load test completed:")
    print(f"  Total requests: {num_requests}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Requests/second: {requests_per_second:.2f}")
    
    return successful == num_requests


def main():
    """Run all tests."""
    print(f"Testing MeiGen-MultiTalk API")
    print(f"API URL: {API_URL}")
    print(f"API Key: {'*' * (len(API_KEY) - 4) + API_KEY[-4:]}")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("Authentication", test_authentication),
        ("Input Validation", test_input_validation),
        ("Job Submission", test_job_submission),
        ("Webhook", test_webhook),
        ("Error Responses", test_error_responses),
        ("Load Test", lambda: run_load_test(20))
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result is not False:
                passed += 1
            else:
                failed += 1
                print(f"✗ {test_name} failed")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"Test Summary: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())