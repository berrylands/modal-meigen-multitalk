# MeiGen-MultiTalk REST API Documentation

## Overview

The MeiGen-MultiTalk REST API provides programmatic access to audio-driven video generation capabilities. Generate videos with single or multiple speakers synchronized to audio tracks.

**Base URL**: `https://berrylands--multitalk-api-fastapi-app.modal.run/api/v1`

**Authentication**: Bearer token (API key) required for all endpoints

## Quick Start

```bash
# Submit a single-person video generation job
curl -X POST "https://berrylands--multitalk-api-fastapi-app.modal.run/api/v1/generate" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A person speaking naturally with clear lip sync",
    "image_s3_url": "s3://760572149-framepack/multi1.png",
    "audio_s3_urls": "s3://760572149-framepack/1.wav"
  }'

# Submit a two-person conversation job
curl -X POST "https://berrylands--multitalk-api-fastapi-app.modal.run/api/v1/generate" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Two people having a conversation",
    "image_s3_url": "s3://760572149-framepack/multi1.png",
    "audio_s3_urls": ["s3://760572149-framepack/1.wav", "s3://760572149-framepack/2.wav"],
    "options": {
      "audio_type": "add",
      "sample_steps": 20
    }
  }'

# Check job status
curl "https://berrylands--multitalk-api-fastapi-app.modal.run/api/v1/jobs/{job_id}" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## Authentication

All API requests must include an Authorization header:

```
Authorization: Bearer YOUR_API_KEY
```

## API Reference

### Generate Video

Submit a new video generation job.

**Endpoint**: `POST /api/v1/generate`

**Request Body**:
```json
{
  "prompt": "string (required) - Description of the video content",
  "image_s3_url": "string (required) - S3 URL of reference image",
  "audio_s3_urls": ["string"] | "string" (required) - S3 URL(s) of audio file(s)",
  "output_s3_bucket": "string (optional) - S3 bucket for output",
  "output_s3_prefix": "string (optional) - S3 prefix for output files (default: 'outputs/')",
  "webhook_url": "string (optional) - URL for completion webhook",
  "options": {
    "resolution": "480p" | "720p" (optional, default: "480p"),
    "sample_steps": "integer (optional, 10-50, default: 20)",
    "audio_type": "add" | "para" (optional, default: "add")",
    "audio_cfg": "number (optional, 3-5, default: 4.0)",
    "color_correction": "number (optional, 0-1, default: 0.7)"
  }
}
```

**Response**:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "created_at": "2024-01-20T12:00:00Z",
  "message": "Job submitted successfully"
}
```

**Example - Single Person**:
```bash
curl -X POST "https://berrylands--multitalk-api-fastapi-app.modal.run/api/v1/generate" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A professional explaining cloud computing",
    "image_s3_url": "s3://my-bucket/images/person.png",
    "audio_s3_urls": "s3://my-bucket/audio/explanation.wav",
    "options": {
      "resolution": "720p",
      "sample_steps": 30
    }
  }'
```

**Example - Multi-Person Conversation**:
```bash
curl -X POST "https://berrylands--multitalk-api-fastapi-app.modal.run/api/v1/generate" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Two people having a conversation about AI",
    "image_s3_url": "s3://my-bucket/images/two-people.png",
    "audio_s3_urls": [
      "s3://my-bucket/audio/person1.wav",
      "s3://my-bucket/audio/person2.wav"
    ],
    "options": {
      "audio_type": "add",
      "color_correction": 0.8
    }
  }'
```

### Get Job Status

Retrieve the status and details of a video generation job.

**Endpoint**: `GET /api/v1/jobs/{job_id}`

**Response**:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending" | "processing" | "completed" | "failed" | "cancelled",
  "created_at": "2024-01-20T12:00:00Z",
  "updated_at": "2024-01-20T12:05:00Z",
  "progress": 75,
  "result": {
    "success": true,
    "s3_output": "s3://bucket/outputs/video_123.mp4",
    "duration": 5.2,
    "frames": 81,
    "processing_time": 45.3
  },
  "error": null,
  "metadata": {}
}
```

**Status Values**:
- `pending`: Job queued for processing
- `processing`: Video generation in progress
- `completed`: Successfully generated
- `failed`: Generation failed (see error field)
- `cancelled`: Job was cancelled

### Get Download URL

Get a presigned download URL for a completed video.

**Endpoint**: `GET /api/v1/jobs/{job_id}/download`

**Query Parameters**:
- `expiration`: URL expiration time in seconds (default: 3600, max: 86400)

**Response**:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "download_url": "https://bucket.s3.amazonaws.com/outputs/video_123.mp4?...",
  "expires_in": 3600,
  "s3_uri": "s3://bucket/outputs/video_123.mp4"
}
```

### List Jobs

List recent video generation jobs.

**Endpoint**: `GET /api/v1/jobs`

**Query Parameters**:
- `limit`: Maximum number of jobs to return (default: 10, max: 100)

**Response**:
```json
{
  "jobs": [
    {
      "job_id": "550e8400-e29b-41d4-a716-446655440000",
      "status": "completed",
      "created_at": "2024-01-20T12:00:00Z",
      "updated_at": "2024-01-20T12:05:00Z"
    }
  ],
  "count": 1
}
```

### Cancel Job

Cancel a pending or processing job.

**Endpoint**: `DELETE /api/v1/jobs/{job_id}`

**Response**:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "cancelled",
  "message": "Job cancelled"
}
```

### Health Check

Check API health status.

**Endpoint**: `GET /api/v1/health`

**Response**:
```json
{
  "status": "healthy",
  "version": "v1",
  "timestamp": "2024-01-20T12:00:00Z"
}
```

### Test Webhook

Test webhook delivery configuration.

**Endpoint**: `POST /api/v1/webhook-test`

**Request Body**:
```json
{
  "webhook_url": "https://your-server.com/webhook"
}
```

## Webhooks

When a `webhook_url` is provided, the API will send notifications on job completion.

**Webhook Payload**:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "timestamp": "2024-01-20T12:05:00Z",
  "result": {
    "success": true,
    "s3_output": "s3://bucket/outputs/video_123.mp4",
    "duration": 5.2,
    "frames": 81,
    "processing_time": 45.3
  },
  "error": null
}
```

**Webhook Retry Logic**:
- 3 retry attempts with exponential backoff
- Timeouts after 10 seconds per attempt

## Error Handling

All errors follow a consistent format:

```json
{
  "error": "Error type",
  "detail": "Detailed error message",
  "request_id": "req_123456"
}
```

**Common Error Codes**:
- `400`: Bad Request - Invalid input parameters
- `401`: Unauthorized - Invalid or missing API key
- `404`: Not Found - Job ID doesn't exist
- `429`: Too Many Requests - Rate limit exceeded
- `500`: Internal Server Error

## Rate Limits

Default rate limits (adjustable based on plan):
- 100 requests per minute
- 1000 requests per hour
- 10 concurrent jobs

## Code Examples

### Python

```python
import requests
import time

class MultiTalkClient:
    def __init__(self, api_url, api_key):
        self.api_url = api_url.rstrip('/')
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def generate_video(self, prompt, image_url, audio_urls, **options):
        """Submit a video generation job."""
        data = {
            'prompt': prompt,
            'image_s3_url': image_url,
            'audio_s3_urls': audio_urls
        }
        
        if options:
            data['options'] = options
        
        response = requests.post(
            f'{self.api_url}/generate',
            json=data,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def get_job_status(self, job_id):
        """Get job status."""
        response = requests.get(
            f'{self.api_url}/jobs/{job_id}',
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def wait_for_completion(self, job_id, timeout=600, poll_interval=10):
        """Wait for job to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_job_status(job_id)
            
            if status['status'] == 'completed':
                return status
            elif status['status'] in ['failed', 'cancelled']:
                raise Exception(f"Job {status['status']}: {status.get('error')}")
            
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Job {job_id} timed out after {timeout} seconds")
    
    def get_download_url(self, job_id, expiration=3600):
        """Get download URL for completed video."""
        response = requests.get(
            f'{self.api_url}/jobs/{job_id}/download',
            params={'expiration': expiration},
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

# Usage example
client = MultiTalkClient(
    'https://berrylands--multitalk-api-fastapi-app.modal.run/api/v1',
    'YOUR_API_KEY'
)

# Generate single-person video
job = client.generate_video(
    prompt='CEO presenting quarterly results',
    image_url='s3://my-bucket/ceo-image.png',
    audio_urls='s3://my-bucket/presentation.wav',
    resolution='720p',
    sample_steps=25
)

print(f"Job submitted: {job['job_id']}")

# Wait for completion
result = client.wait_for_completion(job['job_id'])
print(f"Video generated: {result['result']['s3_output']}")

# Get download URL
download = client.get_download_url(job['job_id'])
print(f"Download URL: {download['download_url']}")
```

### Node.js

```javascript
const axios = require('axios');

class MultiTalkClient {
  constructor(apiUrl, apiKey) {
    this.apiUrl = apiUrl.replace(/\/$/, '');
    this.headers = {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    };
  }

  async generateVideo(prompt, imageUrl, audioUrls, options = {}) {
    const response = await axios.post(
      `${this.apiUrl}/generate`,
      {
        prompt,
        image_s3_url: imageUrl,
        audio_s3_urls: audioUrls,
        options
      },
      { headers: this.headers }
    );
    return response.data;
  }

  async getJobStatus(jobId) {
    const response = await axios.get(
      `${this.apiUrl}/jobs/${jobId}`,
      { headers: this.headers }
    );
    return response.data;
  }

  async waitForCompletion(jobId, timeout = 600000, pollInterval = 10000) {
    const startTime = Date.now();
    
    while (Date.now() - startTime < timeout) {
      const status = await this.getJobStatus(jobId);
      
      if (status.status === 'completed') {
        return status;
      } else if (['failed', 'cancelled'].includes(status.status)) {
        throw new Error(`Job ${status.status}: ${status.error}`);
      }
      
      await new Promise(resolve => setTimeout(resolve, pollInterval));
    }
    
    throw new Error(`Job ${jobId} timed out`);
  }

  async getDownloadUrl(jobId, expiration = 3600) {
    const response = await axios.get(
      `${this.apiUrl}/jobs/${jobId}/download`,
      {
        params: { expiration },
        headers: this.headers
      }
    );
    return response.data;
  }
}

// Usage example
(async () => {
  const client = new MultiTalkClient(
    'https://berrylands--multitalk-api-fastapi-app.modal.run/api/v1',
    'YOUR_API_KEY'
  );

  try {
    // Generate multi-person conversation
    const job = await client.generateVideo(
      'Two experts discussing AI ethics',
      's3://my-bucket/two-experts.png',
      [
        's3://my-bucket/expert1-audio.wav',
        's3://my-bucket/expert2-audio.wav'
      ],
      {
        audio_type: 'add',
        color_correction: 0.7
      }
    );

    console.log(`Job submitted: ${job.job_id}`);

    // Wait and get result
    const result = await client.waitForCompletion(job.job_id);
    console.log(`Video ready: ${result.result.s3_output}`);

  } catch (error) {
    console.error('Error:', error.message);
  }
})();
```

### Go

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "net/http"
    "time"
)

type MultiTalkClient struct {
    APIUrl string
    APIKey string
}

type GenerateRequest struct {
    Prompt         string                 `json:"prompt"`
    ImageS3URL     string                 `json:"image_s3_url"`
    AudioS3URLs    interface{}            `json:"audio_s3_urls"`
    Options        map[string]interface{} `json:"options,omitempty"`
}

type JobResponse struct {
    JobID     string    `json:"job_id"`
    Status    string    `json:"status"`
    CreatedAt time.Time `json:"created_at"`
    Message   string    `json:"message"`
}

type JobStatus struct {
    JobID    string                 `json:"job_id"`
    Status   string                 `json:"status"`
    Progress int                    `json:"progress"`
    Result   map[string]interface{} `json:"result"`
    Error    string                 `json:"error"`
}

func (c *MultiTalkClient) GenerateVideo(prompt, imageURL string, audioURLs interface{}, options map[string]interface{}) (*JobResponse, error) {
    req := GenerateRequest{
        Prompt:      prompt,
        ImageS3URL:  imageURL,
        AudioS3URLs: audioURLs,
        Options:     options,
    }

    body, err := json.Marshal(req)
    if err != nil {
        return nil, err
    }

    httpReq, err := http.NewRequest("POST", c.APIUrl+"/generate", bytes.NewBuffer(body))
    if err != nil {
        return nil, err
    }

    httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)
    httpReq.Header.Set("Content-Type", "application/json")

    client := &http.Client{Timeout: 30 * time.Second}
    resp, err := client.Do(httpReq)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var jobResp JobResponse
    if err := json.NewDecoder(resp.Body).Decode(&jobResp); err != nil {
        return nil, err
    }

    return &jobResp, nil
}

func (c *MultiTalkClient) GetJobStatus(jobID string) (*JobStatus, error) {
    httpReq, err := http.NewRequest("GET", c.APIUrl+"/jobs/"+jobID, nil)
    if err != nil {
        return nil, err
    }

    httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)

    client := &http.Client{Timeout: 10 * time.Second}
    resp, err := client.Do(httpReq)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var status JobStatus
    if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
        return nil, err
    }

    return &status, nil
}

func (c *MultiTalkClient) WaitForCompletion(jobID string, timeout time.Duration) (*JobStatus, error) {
    deadline := time.Now().Add(timeout)
    ticker := time.NewTicker(10 * time.Second)
    defer ticker.Stop()

    for {
        select {
        case <-ticker.C:
            status, err := c.GetJobStatus(jobID)
            if err != nil {
                return nil, err
            }

            switch status.Status {
            case "completed":
                return status, nil
            case "failed", "cancelled":
                return nil, fmt.Errorf("job %s: %s", status.Status, status.Error)
            }

            if time.Now().After(deadline) {
                return nil, fmt.Errorf("timeout waiting for job %s", jobID)
            }
        }
    }
}

// Usage example
func main() {
    client := &MultiTalkClient{
        APIUrl: "https://berrylands--multitalk-api-fastapi-app.modal.run/api/v1",
        APIKey: "YOUR_API_KEY",
    }

    // Single person video
    job, err := client.GenerateVideo(
        "Product demo presentation",
        "s3://bucket/presenter.png",
        "s3://bucket/demo-audio.wav",
        map[string]interface{}{
            "resolution": "720p",
            "sample_steps": 25,
        },
    )
    if err != nil {
        panic(err)
    }

    fmt.Printf("Job submitted: %s\n", job.JobID)

    // Wait for completion
    result, err := client.WaitForCompletion(job.JobID, 10*time.Minute)
    if err != nil {
        panic(err)
    }

    fmt.Printf("Video ready: %s\n", result.Result["s3_output"])
}
```

## Best Practices

### Input Preparation

1. **Image Requirements**:
   - Resolution: 896x448 pixels (recommended)
   - Format: PNG or JPEG
   - Clear facial features for best results

2. **Audio Requirements**:
   - Format: WAV (recommended) or MP3
   - Sample rate: 16kHz or higher
   - Clear speech without background noise

3. **Multi-Person Videos**:
   - Use "add" mode for sequential speaking
   - Use "para" mode for simultaneous speaking
   - Ensure audio files don't overlap in "add" mode

### Performance Tips

1. **Batch Processing**:
   ```python
   # Submit multiple jobs in parallel
   jobs = []
   for audio in audio_files:
       job = client.generate_video(prompt, image, audio)
       jobs.append(job['job_id'])
   
   # Wait for all to complete
   results = [client.wait_for_completion(job_id) for job_id in jobs]
   ```

2. **Webhook Integration**:
   ```python
   # Use webhooks instead of polling
   job = client.generate_video(
       prompt, image, audio,
       webhook_url='https://your-server.com/webhook'
   )
   ```

3. **Error Handling**:
   ```python
   try:
       result = client.generate_video(prompt, image, audio)
   except requests.HTTPError as e:
       if e.response.status_code == 429:
           # Rate limited, wait and retry
           time.sleep(60)
       elif e.response.status_code == 400:
           # Invalid input, check parameters
           error = e.response.json()
           print(f"Validation error: {error['detail']}")
   ```

### Security

1. **API Key Management**:
   - Store keys in environment variables
   - Rotate keys regularly
   - Use different keys for dev/prod

2. **S3 Security**:
   - Use presigned URLs with expiration
   - Implement bucket policies
   - Enable server-side encryption

3. **Network Security**:
   - Always use HTTPS
   - Implement request signing if needed
   - Monitor for unusual activity

## Troubleshooting

### Common Issues

**Job Stays in Pending**:
- Check if API is under heavy load
- Verify S3 permissions for input files
- Ensure input files exist and are accessible

**Job Fails Immediately**:
- Validate image dimensions (896x448)
- Check audio file format and encoding
- Verify S3 URLs are properly formatted

**Webhook Not Received**:
- Ensure webhook URL is publicly accessible
- Check for firewall/security group rules
- Verify webhook endpoint returns 2xx status

**Download URL Expired**:
- Request new URL with longer expiration
- Download immediately after generation
- Consider copying to permanent storage

## Support

- API Status: https://status.modal.com
- Documentation: https://github.com/your-org/modal-meigen-multitalk
- Issues: https://github.com/your-org/modal-meigen-multitalk/issues