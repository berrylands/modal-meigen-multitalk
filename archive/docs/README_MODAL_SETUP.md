# Modal Setup Instructions

Since Modal requires interactive authentication, please follow these steps manually:

## 1. Authenticate with Modal

Run the following command:
```bash
modal setup
```

This will:
- Open a browser window
- Ask you to log in to Modal (or create an account)
- Save authentication tokens locally

## 2. Verify Authentication

After setup, test your connection:
```bash
python test_simple.py
```

You should see:
```
✅ Success: Hello from Modal!

Modal is properly configured!
```

## 3. Create Secrets (Optional but Recommended)

### Via Modal Dashboard (Easier)
1. Go to https://modal.com/settings/secrets
2. Create `huggingface-secret` with:
   - `HUGGINGFACE_TOKEN`: Your HuggingFace token
3. Create `aws-secret` with:
   - `AWS_ACCESS_KEY_ID`: Your AWS key
   - `AWS_SECRET_ACCESS_KEY`: Your AWS secret
   - `AWS_REGION`: eu-west-1

### Via CLI
```bash
# HuggingFace
modal secret create huggingface-secret HUGGINGFACE_TOKEN=your_token

# AWS
modal secret create aws-secret \
  AWS_ACCESS_KEY_ID=your_key \
  AWS_SECRET_ACCESS_KEY=your_secret \
  AWS_REGION=eu-west-1
```

## 4. Test Full Setup

Run the complete test:
```bash
python test_modal.py
```

## 5. Deploy a Test Function

```bash
modal deploy test_simple.py
```

This will deploy your function to Modal's cloud. You can then see it in the Modal dashboard.

## Troubleshooting

### "Not authenticated" Error
- Run `modal setup` again
- Check `~/.modal/config.json` exists

### Secret Not Found
- Secrets are case-sensitive
- Check the Modal dashboard to verify secret names
- Ensure you're in the right Modal environment

### Network Issues
- Modal requires HTTPS outbound access
- Check proxy/firewall settings

## Next Steps

Once setup is complete:
1. Run `python app.py` to test the main application
2. Deploy with `modal deploy app.py`
3. Continue with model setup (Issue #3)