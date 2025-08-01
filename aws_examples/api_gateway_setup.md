# API Gateway Configuration for MeiGen-MultiTalk

This guide walks through setting up AWS API Gateway to expose the Modal-hosted REST API with additional AWS features like request throttling, API keys, and CloudWatch logging.

## Architecture Overview

```
Client -> API Gateway -> Lambda Authorizer (optional) -> Lambda Proxy -> Modal REST API
                |
                ├── CloudWatch Logs
                ├── X-Ray Tracing  
                └── Usage Plans & API Keys
```

## Step-by-Step Setup

### 1. Create REST API

```bash
# Using AWS CLI
aws apigateway create-rest-api \
  --name "MeiGen-MultiTalk-API" \
  --description "API Gateway for MeiGen-MultiTalk video generation" \
  --endpoint-configuration types=REGIONAL
```

Or via Console:
1. Go to API Gateway console
2. Choose "REST API" (not Private)
3. Select "New API"
4. API name: `MeiGen-MultiTalk-API`
5. Endpoint Type: Regional

### 2. Create Lambda Proxy Function

Create `api_gateway_proxy.py`:

```python
import json
import os
import requests
from urllib.parse import urlencode

# Configuration
MODAL_API_URL = os.environ['MODAL_API_URL']  # e.g., https://your-app.modal.run
MODAL_API_KEY = os.environ['MODAL_API_KEY']

def lambda_handler(event, context):
    """
    Lambda function to proxy requests from API Gateway to Modal.
    """
    
    # Extract request details
    path = event.get('path', '/')
    method = event.get('httpMethod', 'GET')
    headers = event.get('headers', {})
    query_params = event.get('queryStringParameters', {})
    body = event.get('body', '')
    
    # Parse body if it's JSON
    if headers.get('Content-Type') == 'application/json' and body:
        try:
            body = json.loads(body)
        except:
            pass
    
    # Build Modal URL
    modal_url = f"{MODAL_API_URL}{path}"
    if query_params:
        modal_url += f"?{urlencode(query_params)}"
    
    # Forward headers (excluding Host and Lambda-specific headers)
    forward_headers = {}
    exclude_headers = ['Host', 'X-Forwarded-For', 'X-Forwarded-Port', 
                      'X-Forwarded-Proto', 'X-Amzn-Trace-Id']
    
    for key, value in headers.items():
        if key not in exclude_headers:
            forward_headers[key] = value
    
    # Add Modal API key
    forward_headers['Authorization'] = f'Bearer {MODAL_API_KEY}'
    
    try:
        # Make request to Modal
        if method == 'GET':
            response = requests.get(modal_url, headers=forward_headers, 
                                  params=query_params, timeout=30)
        elif method == 'POST':
            response = requests.post(modal_url, headers=forward_headers, 
                                   json=body, timeout=30)
        elif method == 'DELETE':
            response = requests.delete(modal_url, headers=forward_headers, 
                                     timeout=30)
        else:
            response = requests.request(method, modal_url, headers=forward_headers,
                                      json=body, timeout=30)
        
        # Build Lambda proxy response
        return {
            'statusCode': response.status_code,
            'headers': {
                'Content-Type': response.headers.get('Content-Type', 'application/json'),
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,X-Api-Key,Authorization',
                'Access-Control-Allow-Methods': 'GET,POST,DELETE,OPTIONS'
            },
            'body': response.text
        }
        
    except requests.exceptions.Timeout:
        return {
            'statusCode': 504,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': 'Gateway timeout'})
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': 'Internal server error', 'detail': str(e)})
        }
```

Deploy this Lambda function with:
- Runtime: Python 3.11
- Memory: 512 MB
- Timeout: 30 seconds
- Environment variables: `MODAL_API_URL`, `MODAL_API_KEY`

### 3. Configure API Gateway Resources

```bash
# Get API ID
API_ID=$(aws apigateway get-rest-apis --query 'items[?name==`MeiGen-MultiTalk-API`].id' --output text)

# Get root resource ID
ROOT_ID=$(aws apigateway get-resources --rest-api-id $API_ID --query 'items[?path==`/`].id' --output text)

# Create /api resource
aws apigateway create-resource \
  --rest-api-id $API_ID \
  --parent-id $ROOT_ID \
  --path-part "api"

# Create proxy resource under /api
API_RESOURCE_ID=$(aws apigateway get-resources --rest-api-id $API_ID --query 'items[?path==`/api`].id' --output text)

aws apigateway create-resource \
  --rest-api-id $API_ID \
  --parent-id $API_RESOURCE_ID \
  --path-part "{proxy+}"
```

### 4. Create Methods

```bash
# Get proxy resource ID
PROXY_ID=$(aws apigateway get-resources --rest-api-id $API_ID --query 'items[?path==`/api/{proxy+}`].id' --output text)

# Create ANY method
aws apigateway put-method \
  --rest-api-id $API_ID \
  --resource-id $PROXY_ID \
  --http-method ANY \
  --authorization-type NONE \
  --api-key-required true
```

### 5. Configure Lambda Integration

```bash
# Set up Lambda integration
aws apigateway put-integration \
  --rest-api-id $API_ID \
  --resource-id $PROXY_ID \
  --http-method ANY \
  --type AWS_PROXY \
  --integration-http-method POST \
  --uri "arn:aws:apigateway:REGION:lambda:path/2015-03-31/functions/arn:aws:lambda:REGION:ACCOUNT:function:multitalk-api-proxy/invocations"
```

### 6. Deploy API

```bash
# Create deployment
aws apigateway create-deployment \
  --rest-api-id $API_ID \
  --stage-name prod \
  --stage-description "Production stage"
```

### 7. Create Usage Plan and API Keys

```bash
# Create usage plan
USAGE_PLAN_ID=$(aws apigateway create-usage-plan \
  --name "MultitalkBasic" \
  --description "Basic usage plan for Multitalk API" \
  --throttle burstLimit=100,rateLimit=50 \
  --quota limit=10000,period=DAY \
  --api-stages apiId=$API_ID,stage=prod \
  --query 'id' --output text)

# Create API key
API_KEY_ID=$(aws apigateway create-api-key \
  --name "multitalk-client-key" \
  --description "API key for Multitalk clients" \
  --enabled \
  --query 'id' --output text)

# Associate API key with usage plan
aws apigateway create-usage-plan-key \
  --usage-plan-id $USAGE_PLAN_ID \
  --key-id $API_KEY_ID \
  --key-type API_KEY
```

## Advanced Configuration

### Custom Authorizer (Optional)

Create a Lambda authorizer for custom authentication:

```python
def lambda_authorizer(event, context):
    """
    Custom authorizer for additional authentication logic.
    """
    
    # Extract token from event
    token = event.get('authorizationToken', '')
    
    # Validate token (implement your logic)
    if not validate_token(token):
        raise Exception('Unauthorized')
    
    # Return policy
    return {
        'principalId': 'user',
        'policyDocument': {
            'Version': '2012-10-17',
            'Statement': [{
                'Action': 'execute-api:Invoke',
                'Effect': 'Allow',
                'Resource': event['methodArn']
            }]
        },
        'context': {
            'userId': 'extracted-user-id'
        }
    }
```

### Request/Response Mapping Templates

For request transformation:

```velocity
#set($inputRoot = $input.path('$'))
{
  "prompt": "$inputRoot.prompt",
  "image_s3_url": "$inputRoot.image_url",
  "audio_s3_urls": ["$inputRoot.audio_url"],
  "options": {
    "sample_steps": #if($inputRoot.steps) $inputRoot.steps #else 20 #end
  }
}
```

### CloudWatch Logging

Enable detailed CloudWatch logs:

```bash
# Create log group
aws logs create-log-group --log-group-name /aws/apigateway/multitalk

# Update stage with logging
aws apigateway update-stage \
  --rest-api-id $API_ID \
  --stage-name prod \
  --patch-operations \
    op=replace,path=/accessLogSettings/destinationArn,value=arn:aws:logs:REGION:ACCOUNT:log-group:/aws/apigateway/multitalk \
    op=replace,path=/accessLogSettings/format,value='$context.requestId'
```

### CORS Configuration

Add CORS support for browser-based clients:

```python
# In Lambda proxy function, add to response headers:
'Access-Control-Allow-Origin': '*',
'Access-Control-Allow-Headers': 'Content-Type,X-Api-Key,Authorization',
'Access-Control-Allow-Methods': 'GET,POST,DELETE,OPTIONS'
```

Also create OPTIONS method for preflight:

```bash
aws apigateway put-method \
  --rest-api-id $API_ID \
  --resource-id $PROXY_ID \
  --http-method OPTIONS \
  --authorization-type NONE
```

## Client Usage Examples

### Using cURL

```bash
# Get your API endpoint
ENDPOINT="https://$API_ID.execute-api.REGION.amazonaws.com/prod"

# Submit a job
curl -X POST "$ENDPOINT/api/v1/generate" \
  -H "x-api-key: YOUR-API-KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A person speaking",
    "image_s3_url": "s3://bucket/image.png",
    "audio_s3_urls": ["s3://bucket/audio.wav"]
  }'

# Check status
curl "$ENDPOINT/api/v1/jobs/JOB-ID" \
  -H "x-api-key: YOUR-API-KEY"
```

### Using Python SDK

```python
import requests

class MultitalkClient:
    def __init__(self, endpoint, api_key):
        self.endpoint = endpoint
        self.headers = {'x-api-key': api_key}
    
    def generate_video(self, prompt, image_url, audio_urls):
        response = requests.post(
            f"{self.endpoint}/api/v1/generate",
            headers=self.headers,
            json={
                'prompt': prompt,
                'image_s3_url': image_url,
                'audio_s3_urls': audio_urls
            }
        )
        return response.json()
    
    def get_status(self, job_id):
        response = requests.get(
            f"{self.endpoint}/api/v1/jobs/{job_id}",
            headers=self.headers
        )
        return response.json()

# Usage
client = MultitalkClient(
    'https://xyz.execute-api.region.amazonaws.com/prod',
    'your-api-key'
)

job = client.generate_video(
    'Person explaining AWS',
    's3://bucket/person.png',
    ['s3://bucket/speech.wav']
)
```

## Monitoring and Alerts

### CloudWatch Metrics

Key metrics to monitor:
- 4XXError: Client errors
- 5XXError: Server errors
- Count: Total API calls
- Latency: Response time
- IntegrationLatency: Backend latency

### Create Alarms

```bash
# High error rate alarm
aws cloudwatch put-metric-alarm \
  --alarm-name multitalk-api-errors \
  --alarm-description "Alert on high API error rate" \
  --metric-name 4XXError \
  --namespace AWS/ApiGateway \
  --statistic Sum \
  --period 300 \
  --threshold 10 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 1
```

### X-Ray Tracing

Enable X-Ray for detailed tracing:

```bash
aws apigateway update-stage \
  --rest-api-id $API_ID \
  --stage-name prod \
  --patch-operations op=replace,path=/tracingEnabled,value=true
```

## Cost Optimization

1. **Caching**: Enable API Gateway caching for GET requests
2. **Throttling**: Set appropriate rate limits
3. **Reserved Capacity**: Consider reserved capacity for predictable workloads
4. **Regional vs Edge**: Use regional endpoints to reduce costs

## Security Best Practices

1. **API Keys**: Rotate regularly
2. **IAM Roles**: Use least privilege
3. **WAF**: Consider AWS WAF for additional protection
4. **Private API**: Use VPC endpoints for internal APIs
5. **Secrets Manager**: Store sensitive configuration

## Troubleshooting

### Common Issues

1. **502 Bad Gateway**
   - Check Lambda function logs
   - Verify Modal API is accessible
   - Check timeout settings

2. **403 Forbidden**
   - Verify API key is included
   - Check usage plan limits
   - Validate IAM permissions

3. **429 Too Many Requests**
   - Check throttling settings
   - Review usage plan quotas

### Debug Tips

1. Enable CloudWatch detailed logs
2. Use X-Ray for request tracing
3. Test with API Gateway test console
4. Check Lambda function logs
5. Verify Modal API directly