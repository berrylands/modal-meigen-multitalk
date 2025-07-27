# Modal Setup Guide

This guide walks through setting up Modal for the MeiGen-MultiTalk project.

## Prerequisites

- Python 3.10+
- Modal account (sign up at https://modal.com)

## 1. Install Modal

```bash
pip install modal
```

Verify installation:
```bash
modal --version
```

## 2. Authenticate with Modal

Run the setup command:
```bash
modal setup
```

This will:
1. Open a browser window for authentication
2. Create a token that's stored locally
3. Allow you to deploy functions to Modal

## 3. Create Modal Secrets

We need to set up secrets for:
- HuggingFace API token (for model downloads)
- AWS credentials (for S3 access)

### Option A: Using Modal Dashboard (Recommended)

1. Go to https://modal.com/settings/secrets
2. Create a new secret called `huggingface-secret`:
   ```
   HUGGINGFACE_TOKEN=your_token_here
   ```
3. Create a new secret called `aws-secret`:
   ```
   AWS_ACCESS_KEY_ID=your_key_id
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_REGION=eu-west-1
   ```

### Option B: Using Modal CLI

```bash
# Create HuggingFace secret
modal secret create huggingface-secret \
  HUGGINGFACE_TOKEN=your_token_here

# Create AWS secret
modal secret create aws-secret \
  AWS_ACCESS_KEY_ID=your_key_id \
  AWS_SECRET_ACCESS_KEY=your_secret_key \
  AWS_REGION=eu-west-1
```

## 4. Test Authentication

Create a test file `test_modal.py`:
```python
import modal

app = modal.App("test-auth")

@app.function(secrets=[modal.Secret.from_name("huggingface-secret")])
def test_secrets():
    import os
    return f"HF Token exists: {'HUGGINGFACE_TOKEN' in os.environ}"

@app.local_entrypoint()
def main():
    result = test_secrets.remote()
    print(result)
```

Run the test:
```bash
modal run test_modal.py
```

## 5. Environment Variables

Create a `.env` file for local development:
```bash
# Modal token (automatically set by modal setup)
MODAL_TOKEN_ID=your_token_id
MODAL_TOKEN_SECRET=your_token_secret

# Optional: set default environment
MODAL_ENVIRONMENT=main
```

## Troubleshooting

### Authentication Issues
- Run `modal token set --token-id YOUR_ID --token-secret YOUR_SECRET`
- Check `~/.modal/config.json` for stored credentials

### Secret Access Issues
- Ensure secrets are created in the correct environment
- Check secret names match exactly (case-sensitive)

### Network Issues
- Modal requires outbound HTTPS access
- Check firewall/proxy settings

## Next Steps

Once authentication is set up:
1. Deploy the test function: `modal deploy test_modal.py`
2. Check the Modal dashboard for your deployed apps
3. Proceed with model setup and deployment