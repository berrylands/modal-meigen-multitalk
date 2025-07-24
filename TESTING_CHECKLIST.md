# Modal Setup Testing Checklist

Follow these steps to test the Modal setup:

## 1. Quick Setup Check
- [ ] Run `python verify_setup.py`
- [ ] Verify Modal SDK is installed
- [ ] Check for authentication status

## 2. Modal Authentication (if needed)
- [ ] Set MODAL_API_TOKEN in .env file
- [ ] OR run `modal setup` for interactive auth

## 3. Basic Connectivity Test
- [ ] Run `python test_simple.py`
- [ ] Should see connection success message

## 4. Create/Update Secrets
### Using the helper script:
```bash
./create_secrets.sh
```

### Or manually via Modal Dashboard:
- [ ] Go to https://modal.com/settings/secrets
- [ ] Create `huggingface-secret` with HF_TOKEN
- [ ] Create `aws-secret` with AWS credentials

## 5. Complete Environment Test
- [ ] Run `python test_complete_setup.py`
- [ ] Verify all systems:
  - AWS S3 access
  - HuggingFace authentication
  - Modal connection

## 6. Main App Test
- [ ] Run `modal run app.py`
- [ ] Should see health check results
- [ ] Should see model download function status

## Troubleshooting

### Authentication Failed
```bash
# Check config exists
ls ~/.modal/config.json

# Re-run setup
modal setup
```

### Secret Not Found
- Secrets are case-sensitive
- Check exact names in dashboard
- Ensure you're in the right environment

### GPU Not Available
- Normal for local testing
- Will work when deployed

## Success Criteria
✅ `test_simple.py` runs without errors
✅ Modal dashboard shows your account
✅ Can deploy functions
✅ (Optional) Secrets are accessible