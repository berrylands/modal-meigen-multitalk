"""
S3 utilities for downloading inputs and uploading outputs.
"""

import boto3
import os
from typing import Optional, Dict, Any
import tempfile
from pathlib import Path

class S3Manager:
    """Manages S3 operations for MultiTalk inputs and outputs."""
    
    def __init__(self, bucket_name: str = None):
        """Initialize S3 client with AWS credentials from environment."""
        self.s3_client = boto3.client('s3')
        # Try bucket name from parameter, then AWS_BUCKET_NAME (Modal secret), then S3_BUCKET_NAME
        self.bucket_name = bucket_name or os.environ.get('AWS_BUCKET_NAME') or os.environ.get('S3_BUCKET_NAME')
        
        if not self.bucket_name:
            raise ValueError("S3 bucket name must be provided or set in AWS_BUCKET_NAME or S3_BUCKET_NAME env var")
    
    def download_file(self, s3_key: str, local_path: str = None) -> str:
        """
        Download a file from S3.
        
        Args:
            s3_key: S3 object key (e.g., 'multi1.png' or 'path/to/file.wav')
            local_path: Local path to save to. If None, saves to temp directory.
            
        Returns:
            Path to the downloaded file
        """
        if local_path is None:
            # Create temp file with same extension
            suffix = Path(s3_key).suffix
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                local_path = tmp.name
        
        print(f"Downloading s3://{self.bucket_name}/{s3_key} to {local_path}")
        self.s3_client.download_file(self.bucket_name, s3_key, local_path)
        
        return local_path
    
    def upload_file(self, local_path: str, s3_key: str = None) -> str:
        """
        Upload a file to S3.
        
        Args:
            local_path: Path to local file to upload
            s3_key: S3 object key. If None, uses basename of local_path
            
        Returns:
            S3 URI of uploaded file
        """
        if s3_key is None:
            s3_key = os.path.basename(local_path)
        
        print(f"Uploading {local_path} to s3://{self.bucket_name}/{s3_key}")
        self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
        
        return f"s3://{self.bucket_name}/{s3_key}"
    
    def download_inputs(self, image_key: str, audio_key: str, 
                       output_dir: str = None) -> Dict[str, str]:
        """
        Download MultiTalk input files from S3.
        
        Args:
            image_key: S3 key for input image
            audio_key: S3 key for input audio
            output_dir: Directory to save files to. If None, uses temp directory.
            
        Returns:
            Dict with 'image_path' and 'audio_path' keys
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="multitalk_inputs_")
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        # Download files
        image_path = self.download_file(
            image_key, 
            os.path.join(output_dir, os.path.basename(image_key))
        )
        audio_path = self.download_file(
            audio_key,
            os.path.join(output_dir, os.path.basename(audio_key))
        )
        
        return {
            "image_path": image_path,
            "audio_path": audio_path,
            "output_dir": output_dir
        }
    
    def upload_output(self, video_path: str, s3_prefix: str = "outputs/") -> str:
        """
        Upload generated video to S3.
        
        Args:
            video_path: Path to generated video file
            s3_prefix: Prefix for S3 key (default: 'outputs/')
            
        Returns:
            S3 URI of uploaded video
        """
        # Generate unique S3 key
        video_name = os.path.basename(video_path)
        s3_key = f"{s3_prefix}{video_name}"
        
        return self.upload_file(video_path, s3_key)
    
    def list_bucket_contents(self, prefix: str = "") -> list:
        """List objects in the S3 bucket with optional prefix."""
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=prefix
        )
        
        if 'Contents' not in response:
            return []
        
        return [obj['Key'] for obj in response['Contents']]


# Helper functions for Modal integration
def download_s3_inputs(bucket_name: str, image_key: str, audio_key: str) -> Dict[str, str]:
    """
    Convenience function to download inputs from S3.
    
    Args:
        bucket_name: S3 bucket name
        image_key: S3 key for image (e.g., 'multi1.png')
        audio_key: S3 key for audio (e.g., '1.wav')
        
    Returns:
        Dict with local paths
    """
    s3_manager = S3Manager(bucket_name)
    return s3_manager.download_inputs(image_key, audio_key)


def upload_s3_output(bucket_name: str, video_path: str, s3_key: str = None) -> str:
    """
    Convenience function to upload output to S3.
    
    Args:
        bucket_name: S3 bucket name
        video_path: Local path to video file
        s3_key: Optional S3 key (uses default if not provided)
        
    Returns:
        S3 URI of uploaded file
    """
    s3_manager = S3Manager(bucket_name)
    return s3_manager.upload_output(video_path)


if __name__ == "__main__":
    # Test S3 operations
    print("S3 utilities for MultiTalk")
    print("Usage:")
    print("  from s3_utils import download_s3_inputs, upload_s3_output")
    print("  ")
    print("  # Download inputs")
    print("  paths = download_s3_inputs('my-bucket', 'multi1.png', '1.wav')")
    print("  ")
    print("  # Upload output")
    print("  s3_uri = upload_s3_output('my-bucket', 'output.mp4')")