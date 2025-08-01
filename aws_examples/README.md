# AWS Integration Examples for MeiGen-MultiTalk

This directory contains examples of how to integrate MeiGen-MultiTalk's REST API with various AWS services.

## Overview

The examples show how to:
- Call the Modal-hosted REST API from AWS Lambda
- Orchestrate video generation workflows with Step Functions
- Handle S3 events to automatically process uploaded media
- Receive webhooks for job completion notifications

## Files

### lambda_function.py
Contains Lambda function examples for:
- `lambda_handler` - Basic function to submit video generation jobs
- `check_job_status` - Poll for job completion status
- `get_download_url` - Get presigned URLs for completed videos
- `s3_trigger_handler` - Automatically process S3 uploads
- `webhook_receiver` - Handle completion webhooks

### step_functions_definition.json
A complete Step Functions state machine that:
- Validates input parameters
- Verifies S3 media files exist
- Submits video generation job
- Polls for completion
- Sends notifications via SNS
- Updates DynamoDB tracking table

### step_functions_lambdas.py
Lambda functions designed to work with the Step Functions workflow:
- `validate_multitalk_input` - Input validation
- `check_s3_media` - Verify media files exist
- `submit_multitalk_job` - Submit to REST API
- `check_multitalk_status` - Check job status
- `get_multitalk_download` - Get download URLs

## Setup Instructions

### 1. Deploy the REST API to Modal

First, ensure your Modal app is deployed:

```bash
modal deploy api.py
```

Note the URL of your deployed API (e.g., `https://your-app.modal.run`).

### 2. Set Up Environment Variables

For Lambda functions, set these environment variables:
- `MULTITALK_API_URL`: Your Modal app URL + `/api/v1`
- `MULTITALK_API_KEY`: Your API key
- `S3_BUCKET`: Default S3 bucket for outputs

### 3. Deploy Lambda Functions

#### Option 1: Single Universal Function
Deploy `lambda_function.py` and use different handler methods:
- Handler: `lambda_function.lambda_handler`
- For S3 triggers: `lambda_function.s3_trigger_handler`
- For webhooks: `lambda_function.webhook_receiver`

#### Option 2: Step Functions Integration
Deploy `step_functions_lambdas.py` with:
- Handler: `step_functions_lambdas.lambda_handler`
- Set `FUNCTION_NAME` environment variable for each Lambda to specify which function to run

### 4. Create DynamoDB Table (Optional)

If using the Step Functions workflow, create a DynamoDB table:
- Table name: `MultitalkJobs`
- Partition key: `job_id` (String)

### 5. Deploy Step Functions

1. Replace placeholders in `step_functions_definition.json`:
   - `REGION`: Your AWS region
   - `ACCOUNT`: Your AWS account ID

2. Create the state machine:
```bash
aws stepfunctions create-state-machine \
  --name MultitalkVideoGeneration \
  --definition file://step_functions_definition.json \
  --role-arn arn:aws:iam::ACCOUNT:role/StepFunctionsRole
```

### 6. Configure API Gateway (Optional)

To expose the REST API through API Gateway:

1. Create a new REST API
2. Create a proxy resource: `/{proxy+}`
3. Set up ANY method with Lambda proxy integration
4. Point to a Lambda that forwards requests to Modal

Example Lambda proxy:
```python
import requests

def lambda_handler(event, context):
    modal_url = f"{MODAL_API_URL}{event['path']}"
    
    response = requests.request(
        method=event['httpMethod'],
        url=modal_url,
        headers=event['headers'],
        json=event.get('body'),
        params=event.get('queryStringParameters')
    )
    
    return {
        'statusCode': response.status_code,
        'headers': dict(response.headers),
        'body': response.text
    }
```

## Usage Examples

### Direct Lambda Invocation

```python
import boto3
import json

lambda_client = boto3.client('lambda')

# Submit a video generation job
response = lambda_client.invoke(
    FunctionName='multitalk-submit-job',
    InvocationType='RequestResponse',
    Payload=json.dumps({
        'prompt': 'A person explaining AWS integration',
        'image_s3_url': 's3://my-bucket/images/person.png',
        'audio_s3_urls': ['s3://my-bucket/audio/speech.wav'],
        'output_s3_bucket': 'my-output-bucket',
        'output_s3_prefix': 'videos/'
    })
)

result = json.loads(response['Payload'].read())
job_id = result['job_id']
```

### Step Functions Execution

```python
import boto3
import json

sfn_client = boto3.client('stepfunctions')

response = sfn_client.start_execution(
    stateMachineArn='arn:aws:states:region:account:stateMachine:MultitalkVideoGeneration',
    input=json.dumps({
        'prompt': 'Two people discussing cloud computing',
        'image_s3_url': 's3://my-bucket/images/two-people.png',
        'audio_s3_urls': [
            's3://my-bucket/audio/person1.wav',
            's3://my-bucket/audio/person2.wav'
        ],
        'options': {
            'audio_type': 'add',
            'sample_steps': 20
        }
    })
)

execution_arn = response['executionArn']
```

### S3 Event Configuration

Configure your S3 bucket to trigger the Lambda on `.wav` uploads:

1. Go to S3 bucket properties
2. Create event notification
3. Event type: `PUT`
4. Suffix: `.wav`
5. Destination: Lambda function `s3_trigger_handler`

## Architecture Patterns

### Pattern 1: Simple Request-Response
```
Client -> Lambda -> Modal API -> S3
```

### Pattern 2: Event-Driven Processing
```
S3 Upload -> Lambda Trigger -> Modal API -> S3 Output -> SNS Notification
```

### Pattern 3: Orchestrated Workflow
```
Client -> Step Functions -> Multiple Lambdas -> Modal API -> DynamoDB/SNS
```

### Pattern 4: API Gateway Frontend
```
Client -> API Gateway -> Lambda Proxy -> Modal API
```

## Cost Considerations

- Lambda invocations: ~$0.20 per million requests
- Step Functions: ~$25 per million state transitions  
- API Gateway: ~$3.50 per million API calls
- DynamoDB: ~$0.25 per GB-month storage
- Data transfer: Varies by region and volume

## Security Best Practices

1. **API Keys**: Store in AWS Secrets Manager or Systems Manager Parameter Store
2. **IAM Roles**: Use least-privilege policies
3. **VPC**: Consider VPC endpoints for S3 access
4. **Encryption**: Enable S3 bucket encryption
5. **Monitoring**: Set up CloudWatch alarms for errors

## Troubleshooting

### Common Issues

1. **Timeout Errors**
   - Increase Lambda timeout (max 15 minutes)
   - Use Step Functions for longer workflows

2. **Authentication Failures**
   - Verify API key is set correctly
   - Check Lambda environment variables

3. **S3 Access Denied**
   - Ensure Lambda role has S3 permissions
   - Check bucket policies

4. **Step Functions Stuck**
   - Check CloudWatch logs for Lambda errors
   - Verify state machine IAM role permissions

### Debugging Tips

1. Enable Lambda X-Ray tracing
2. Use CloudWatch Logs Insights for querying
3. Set up CloudWatch dashboards
4. Test functions locally with SAM CLI

## Support

For issues specific to:
- Modal/MultiTalk API: Check Modal logs and documentation
- AWS Integration: Review CloudWatch logs and AWS documentation
- This example code: Open an issue in the GitHub repository