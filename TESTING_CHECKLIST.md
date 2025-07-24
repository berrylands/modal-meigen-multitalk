# Modal Setup Testing Checklist

Follow these steps to test the Modal setup:

## 1. Modal Authentication
- [ ] Run `modal setup`
- [ ] Complete browser authentication
- [ ] Verify token is saved (should see success message)

## 2. Basic Connectivity Test
- [ ] Run `python test_simple.py`
- [ ] Should see: `✅ Success: Hello from Modal!`
- [ ] If error, check authentication

## 3. Create Secrets (Optional)
### Option A: Modal Dashboard
- [ ] Go to https://modal.com/settings/secrets
- [ ] Create `huggingface-secret`:
  - HUGGINGFACE_TOKEN=your_token
- [ ] Create `aws-secret`:
  - AWS_ACCESS_KEY_ID=your_key
  - AWS_SECRET_ACCESS_KEY=your_secret
  - AWS_REGION=eu-west-1

### Option B: CLI
```bash
modal secret create huggingface-secret HUGGINGFACE_TOKEN=your_token
modal secret create aws-secret AWS_ACCESS_KEY_ID=key AWS_SECRET_ACCESS_KEY=secret AWS_REGION=eu-west-1
```

## 4. Full Test Suite
- [ ] Run `python test_modal.py`
- [ ] Check all tests pass:
  - Basic function: ✓
  - Secrets access: ✓ (or ✗ if not created)
  - GPU access: Should show GPU info

## 5. Deploy Test Function
- [ ] Run `modal deploy test_simple.py`
- [ ] Check Modal dashboard for deployed function
- [ ] Note the function URL

## 6. Main App Test
- [ ] Run `python app.py`
- [ ] Should see health check results
- [ ] Should see model download function status

## 7. Clean Up (Optional)
- [ ] Remove test deployment: `modal app stop test-simple`

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