"""
AWS Lambda function to call MeiGen-MultiTalk REST API.
This example shows how to integrate with the Modal-hosted API from AWS Lambda.
"""

import json
import os
import boto3
import requests
from typing import Dict, Any

# Configuration
API_BASE_URL = os.environ.get('MULTITALK_API_URL', 'https://your-modal-app.modal.run/api/v1')
API_KEY = os.environ.get('MULTITALK_API_KEY')
DEFAULT_S3_BUCKET = os.environ.get('S3_BUCKET')

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler for video generation requests.
    
    Expected event format:
    {
        "prompt": "A person speaking about AI",
        "image_s3_url": "s3://bucket/path/to/image.png",
        "audio_s3_urls": ["s3://bucket/path/to/audio1.wav"],
        "webhook_url": "https://your-webhook.com/callback",
        "options": {
            "sample_steps": 20,
            "audio_type": "add"
        }
    }
    """
    
    try:
        # Validate required fields
        if not event.get('prompt'):
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing required field: prompt'})
            }
        
        if not event.get('image_s3_url'):
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing required field: image_s3_url'})
            }
        
        if not event.get('audio_s3_urls'):
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing required field: audio_s3_urls'})
            }
        
        # Prepare API request
        api_request = {
            "prompt": event['prompt'],
            "image_s3_url": event['image_s3_url'],
            "audio_s3_urls": event['audio_s3_urls'],
            "output_s3_bucket": event.get('output_s3_bucket', DEFAULT_S3_BUCKET),
            "output_s3_prefix": event.get('output_s3_prefix', 'outputs/'),
            "webhook_url": event.get('webhook_url'),
            "options": event.get('options', {})
        }
        
        # Call Modal API
        headers = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json'
        }
        
        response = requests.post(
            f'{API_BASE_URL}/generate',
            json=api_request,
            headers=headers,
            timeout=30
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Return success response
        return {
            'statusCode': 200,
            'body': json.dumps({
                'job_id': result['job_id'],
                'status': result['status'],
                'message': result['message']
            })
        }
        
    except requests.exceptions.RequestException as e:
        # Handle API errors
        error_message = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_data = e.response.json()
                error_message = error_data.get('detail', error_message)
            except:
                pass
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'API request failed',
                'detail': error_message
            })
        }
        
    except Exception as e:
        # Handle unexpected errors
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Internal error',
                'detail': str(e)
            })
        }


def check_job_status(job_id: str) -> Dict[str, Any]:
    """
    Check the status of a video generation job.
    Can be called from another Lambda function or scheduled periodically.
    """
    try:
        headers = {
            'Authorization': f'Bearer {API_KEY}'
        }
        
        response = requests.get(
            f'{API_BASE_URL}/jobs/{job_id}',
            headers=headers,
            timeout=10
        )
        
        response.raise_for_status()
        return response.json()
        
    except Exception as e:
        return {
            'error': 'Failed to check job status',
            'detail': str(e)
        }


def get_download_url(job_id: str, expiration: int = 3600) -> Dict[str, Any]:
    """
    Get a presigned download URL for a completed video.
    """
    try:
        headers = {
            'Authorization': f'Bearer {API_KEY}'
        }
        
        response = requests.get(
            f'{API_BASE_URL}/jobs/{job_id}/download',
            params={'expiration': expiration},
            headers=headers,
            timeout=10
        )
        
        response.raise_for_status()
        return response.json()
        
    except Exception as e:
        return {
            'error': 'Failed to get download URL',
            'detail': str(e)
        }


# Example: Lambda function that processes S3 upload events
def s3_trigger_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda function triggered by S3 uploads.
    Automatically generates videos when audio files are uploaded.
    
    Configure S3 bucket to trigger this function on .wav uploads.
    """
    
    s3 = boto3.client('s3')
    results = []
    
    for record in event.get('Records', []):
        # Extract S3 event details
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        
        # Skip if not a WAV file
        if not key.lower().endswith('.wav'):
            continue
        
        # Derive paths (customize based on your naming convention)
        # Example: audio/session123/audio1.wav -> images/session123/image.png
        path_parts = key.split('/')
        if len(path_parts) < 2:
            continue
            
        session_id = path_parts[-2]
        image_key = f"images/{session_id}/image.png"
        
        # Check if reference image exists
        try:
            s3.head_object(Bucket=bucket, Key=image_key)
        except:
            print(f"Reference image not found: s3://{bucket}/{image_key}")
            continue
        
        # Submit video generation job
        api_request = {
            "prompt": f"Person speaking from session {session_id}",
            "image_s3_url": f"s3://{bucket}/{image_key}",
            "audio_s3_urls": [f"s3://{bucket}/{key}"],
            "output_s3_prefix": f"outputs/{session_id}/",
            "options": {
                "sample_steps": 20
            }
        }
        
        try:
            headers = {
                'Authorization': f'Bearer {API_KEY}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                f'{API_BASE_URL}/generate',
                json=api_request,
                headers=headers,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            results.append({
                'audio_key': key,
                'job_id': result['job_id'],
                'status': 'submitted'
            })
            
        except Exception as e:
            results.append({
                'audio_key': key,
                'error': str(e),
                'status': 'failed'
            })
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'processed': len(results),
            'results': results
        })
    }


# Example: Lambda function for webhook receiver
def webhook_receiver(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda function to receive webhook notifications from MultiTalk API.
    
    Configure API Gateway to forward POST requests to this function.
    """
    
    try:
        # Parse webhook payload
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', {})
        
        job_id = body.get('job_id')
        status = body.get('status')
        timestamp = body.get('timestamp')
        
        print(f"Webhook received for job {job_id}: {status}")
        
        # Handle different statuses
        if status == 'completed':
            # Video generation completed successfully
            result = body.get('result', {})
            s3_output = result.get('s3_output')
            
            # You could trigger additional processing here
            # For example: Send SNS notification, update DynamoDB, etc.
            
            print(f"Video available at: {s3_output}")
            
        elif status == 'failed':
            # Handle failure
            error = body.get('error')
            print(f"Job failed: {error}")
            
            # You could send alerts or retry logic here
        
        # Return success to acknowledge webhook
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Webhook processed'})
        }
        
    except Exception as e:
        print(f"Webhook processing error: {e}")
        # Still return 200 to prevent retries for malformed webhooks
        return {
            'statusCode': 200,
            'body': json.dumps({'error': str(e)})
        }