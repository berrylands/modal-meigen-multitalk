# Development Guide

This guide covers the development workflow for Modal MeiGen-MultiTalk.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/berrylands/modal-meigen-multitalk.git
cd modal-meigen-multitalk
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Modal:
```bash
modal setup
```

## Project Structure

```
modal-meigen-multitalk/
├── app.py              # Main Modal application
├── models/
│   ├── download.py     # Model download utilities
│   └── pipeline.py     # MultiTalk pipeline wrapper
├── utils/
│   ├── audio.py        # Audio processing utilities
│   ├── video.py        # Video processing utilities
│   └── storage.py      # S3/storage utilities
├── tests/              # Test files
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## Key Components

### Model Management
- Models are stored in Modal Volumes for persistence
- Download scripts handle model acquisition from HuggingFace
- Pipeline wrapper manages model loading and inference

### GPU Optimization
- Use `gpu="a10g"` for standard workloads
- Use `gpu="a100"` for 720p or batch processing
- Flash Attention and xformers for memory efficiency

### Cold Start Mitigation
- Pre-load critical models in container image
- Use Modal's warm pool for frequently accessed endpoints
- Implement health checks to keep containers warm

## Testing

Run tests locally:
```bash
python -m pytest tests/
```

Test Modal deployment:
```bash
modal run app.py::test_function
```

## Deployment

Deploy to Modal:
```bash
modal deploy app.py
```

Monitor deployment:
```bash
modal logs -f
```

## Common Issues

### Out of Memory
- Reduce batch size
- Use lower resolution (480p instead of 720p)
- Enable memory-efficient attention

### Slow Cold Starts
- Increase warm pool size
- Cache more models in image
- Use smaller model variants

### Model Download Failures
- Check HuggingFace tokens
- Verify network connectivity
- Use Modal secrets for credentials