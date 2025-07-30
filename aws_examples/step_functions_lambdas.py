"""
Lambda functions for AWS Step Functions workflow.
These functions integrate with the MeiGen-MultiTalk REST API.
"""

import json
import os
import boto3
import requests
from typing import Dict, Any, List
from urllib.parse import urlparse

# Configuration
API_BASE_URL = os.environ.get('MULTITALK_API_URL', 'https://your-modal-app.modal.run/api/v1')
API_KEY = os.environ.get('MULTITALK_API_KEY')


def validate_multitalk_input(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda function to validate input for MultiTalk video generation.
    
    Expected event format:
    {
        "prompt": "A person speaking",
        "image_s3_url": "s3://bucket/path/to/image.png",
        "audio_s3_urls": ["s3://bucket/path/to/audio1.wav"],
        "output_s3_bucket": "my-output-bucket",
        "output_s3_prefix": "outputs/",
        "options": {
            "sample_steps": 20,
            "audio_type": "add"
        }
    }
    """
    
    # Required fields
    required_fields = ['prompt', 'image_s3_url', 'audio_s3_urls']
    
    for field in required_fields:
        if field not in event:
            raise Exception(f"ValidationError: Missing required field: {field}")
    
    # Validate S3 URLs
    def is_valid_s3_url(url: str) -> bool:
        return url.startswith('s3://') and len(url.split('/', 3)) >= 4
    
    if not is_valid_s3_url(event['image_s3_url']):
        raise Exception(f"ValidationError: Invalid S3 URL format for image: {event['image_s3_url']}")
    
    audio_urls = event['audio_s3_urls']
    if isinstance(audio_urls, str):
        audio_urls = [audio_urls]
    
    for url in audio_urls:
        if not is_valid_s3_url(url):
            raise Exception(f"ValidationError: Invalid S3 URL format for audio: {url}")
    
    # Validate options if provided
    if 'options' in event:
        options = event['options']
        
        if 'sample_steps' in options:
            steps = options['sample_steps']
            if not isinstance(steps, int) or steps < 10 or steps > 50:
                raise Exception("ValidationError: sample_steps must be between 10 and 50")
        
        if 'audio_type' in options:
            if options['audio_type'] not in ['add', 'para']:
                raise Exception("ValidationError: audio_type must be 'add' or 'para'")
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'validated': True,
            'message': 'Input validation successful'
        })
    }


def check_s3_media(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda function to verify S3 media files exist.
    """
    
    s3 = boto3.client('s3')
    
    def check_s3_object(s3_url: str) -> bool:
        """Check if S3 object exists."""
        parsed = urlparse(s3_url)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        
        try:
            s3.head_object(Bucket=bucket, Key=key)
            return True
        except:
            return False
    
    # Check image
    image_url = event['image_s3_url']
    if not check_s3_object(image_url):
        raise Exception(f"MediaNotFound: Image not found: {image_url}")
    
    # Check audio files
    audio_urls = event['audio_s3_urls']
    if isinstance(audio_urls, str):
        audio_urls = [audio_urls]
    
    for audio_url in audio_urls:
        if not check_s3_object(audio_url):
            raise Exception(f"MediaNotFound: Audio not found: {audio_url}")
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'all_media_exists': True,
            'image_checked': image_url,
            'audio_checked': audio_urls
        })
    }


def submit_multitalk_job(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda function to submit a video generation job to MultiTalk API.
    """
    
    # Prepare API request
    api_request = {
        "prompt": event['prompt'],
        "image_s3_url": event['image_s3_url'],
        "audio_s3_urls": event['audio_s3_urls']
    }
    
    # Add optional fields
    if 'output_s3_bucket' in event:
        api_request['output_s3_bucket'] = event['output_s3_bucket']
    if 'output_s3_prefix' in event:
        api_request['output_s3_prefix'] = event['output_s3_prefix']
    if 'options' in event:
        api_request['options'] = event['options']
    
    # Call Modal API
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(
            f'{API_BASE_URL}/generate',
            json=api_request,
            headers=headers,
            timeout=30
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Also store in DynamoDB for tracking
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('MultitalkJobs')
        
        table.put_item(
            Item={
                'job_id': result['job_id'],
                'status': result['status'],
                'created_at': result['created_at'],
                'request': json.dumps(api_request),
                'state_machine_execution': context.aws_request_id
            }
        )
        
        return result
        
    except requests.exceptions.RequestException as e:
        error_message = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_data = e.response.json()
                error_message = error_data.get('detail', error_message)
            except:
                pass
        
        raise Exception(f"API request failed: {error_message}")


def check_multitalk_status(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda function to check the status of a MultiTalk job.
    """
    
    job_id = event['job_id']
    
    headers = {
        'Authorization': f'Bearer {API_KEY}'
    }
    
    try:
        response = requests.get(
            f'{API_BASE_URL}/jobs/{job_id}',
            headers=headers,
            timeout=10
        )
        
        response.raise_for_status()
        job_data = response.json()
        
        # Update DynamoDB
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('MultitalkJobs')
        
        table.update_item(
            Key={'job_id': job_id},
            UpdateExpression='SET #status = :status, #updated = :updated, #progress = :progress',
            ExpressionAttributeNames={
                '#status': 'status',
                '#updated': 'updated_at',
                '#progress': 'progress'
            },
            ExpressionAttributeValues={
                ':status': job_data['status'],
                ':updated': job_data['updated_at'],
                ':progress': job_data.get('progress', 0)
            }
        )
        
        return job_data
        
    except Exception as e:
        raise Exception(f"Failed to check job status: {str(e)}")


def get_multitalk_download(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda function to get download URL for completed video.
    """
    
    job_id = event['job_id']
    expiration = event.get('expiration', 3600)
    
    headers = {
        'Authorization': f'Bearer {API_KEY}'
    }
    
    try:
        response = requests.get(
            f'{API_BASE_URL}/jobs/{job_id}/download',
            params={'expiration': expiration},
            headers=headers,
            timeout=10
        )
        
        response.raise_for_status()
        download_data = response.json()
        
        # Update DynamoDB with download info
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('MultitalkJobs')
        
        table.update_item(
            Key={'job_id': job_id},
            UpdateExpression='SET #download = :url, #s3uri = :s3',
            ExpressionAttributeNames={
                '#download': 'download_url',
                '#s3uri': 's3_uri'
            },
            ExpressionAttributeValues={
                ':url': download_data['download_url'],
                ':s3': download_data['s3_uri']
            }
        )
        
        return download_data
        
    except Exception as e:
        raise Exception(f"Failed to get download URL: {str(e)}")


# Handler mapping for Lambda function deployment
HANDLER_MAP = {
    'validate-multitalk-input': validate_multitalk_input,
    'check-s3-media': check_s3_media,
    'submit-multitalk-job': submit_multitalk_job,
    'check-multitalk-status': check_multitalk_status,
    'get-multitalk-download': get_multitalk_download
}


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Universal Lambda handler that routes to specific functions.
    Set FUNCTION_NAME environment variable to specify which function to run.
    """
    
    function_name = os.environ.get('FUNCTION_NAME')
    if not function_name or function_name not in HANDLER_MAP:
        raise ValueError(f"Invalid or missing FUNCTION_NAME: {function_name}")
    
    handler = HANDLER_MAP[function_name]
    return handler(event, context)