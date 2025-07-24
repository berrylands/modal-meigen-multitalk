# Modal MeiGen-MultiTalk

A serverless implementation of [MeiGen-MultiTalk](https://github.com/MeiGen-AI/MultiTalk) running on [Modal](https://modal.com/).

## Overview

This project provides a serverless API for audio-driven multi-person conversational video generation using the MeiGen-MultiTalk model. It leverages Modal's serverless GPU infrastructure to provide on-demand video generation without maintaining expensive GPU instances.

## Features

- 🎥 Audio-driven talking head video generation
- 🚀 Serverless deployment with automatic scaling
- 💰 Pay-per-use pricing (no idle costs)
- 🎭 Support for single and multi-person scenarios
- 🎨 Works with cartoon characters and singing
- ⚡ GPU-accelerated inference

## Requirements

- Python 3.10+
- Modal account and API key
- ~82GB storage for model weights

## Quick Start

1. Install Modal:
```bash
pip install modal
```

2. Authenticate with Modal:
```bash
modal setup
```

3. Deploy the application:
```bash
modal deploy app.py
```

## Architecture

The project uses:
- Modal Volumes for persistent model storage
- GPU-enabled serverless functions (A10G or better)
- Custom container with PyTorch 2.4.1 and dependencies
- HTTP endpoint for API access

## API Usage

```python
# Example API call (coming soon)
```

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for development guidelines.

## License

This project follows the license terms of the original [MeiGen-MultiTalk](https://github.com/MeiGen-AI/MultiTalk) project.