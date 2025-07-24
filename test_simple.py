"""Simple Modal test without secrets."""

import modal

app = modal.App("test-simple")

@app.function()
def hello():
    """Test basic Modal function."""
    return "Hello from Modal!"

@app.local_entrypoint()
def main():
    """Run the test."""
    print("Testing Modal connection...")
    try:
        result = hello.remote()
        print(f"✅ Success: {result}")
        print("\nModal is properly configured!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nPlease run 'modal setup' to authenticate with Modal")

if __name__ == "__main__":
    main()