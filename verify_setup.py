"""Verify Modal setup is complete."""

import os
import sys
from pathlib import Path

def check_modal_installed():
    """Check if Modal is installed."""
    try:
        import modal
        print("✅ Modal SDK installed (version: {})".format(modal.__version__))
        return True
    except ImportError:
        print("❌ Modal SDK not installed")
        print("   Run: pip install modal")
        return False

def check_modal_config():
    """Check if Modal is configured."""
    config_path = Path.home() / ".modal" / "config.json"
    if config_path.exists():
        print("✅ Modal config found at ~/.modal/config.json")
        return True
    else:
        print("❌ Modal not configured")
        print("   Run: modal setup")
        return False

def check_env_file():
    """Check for .env file."""
    if Path(".env").exists():
        print("✅ .env file exists")
        return True
    else:
        print("⚠️  No .env file (optional for local testing)")
        print("   Copy .env.example to .env if needed")
        return True  # Optional

def check_requirements():
    """Check if all requirements are satisfied."""
    try:
        import torch
        print("✅ PyTorch installed")
    except ImportError:
        print("⚠️  PyTorch not installed (will be installed in Modal container)")
    
    return True

def main():
    """Run all checks."""
    print("Modal Setup Verification")
    print("=" * 50)
    
    checks = [
        check_modal_installed(),
        check_modal_config(),
        check_env_file(),
        check_requirements()
    ]
    
    print("\n" + "=" * 50)
    if all(checks[:2]):  # First two are required
        print("✅ Basic setup complete!")
        print("\nNext steps:")
        print("1. Run: python test_simple.py")
        print("2. Create secrets in Modal dashboard (optional)")
        print("3. Run: python test_modal.py")
    else:
        print("❌ Setup incomplete")
        print("\nPlease complete the steps above before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main()