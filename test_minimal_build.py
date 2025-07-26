"""
Test minimal builds to isolate the issue.
"""

import modal
import os
import time

modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Test 1: Just NumPy
print("Test 1: Building with just NumPy...")
numpy_image = modal.Image.debian_slim(python_version="3.10").pip_install("numpy==1.26.4")

app1 = modal.App("test-numpy-only")

@app1.function(image=numpy_image)
def test_numpy():
    import numpy as np
    return f"NumPy {np.__version__} installed"

try:
    with app1.run():
        print(f"Result: {test_numpy.remote()}")
        print("✅ NumPy image works\n")
except Exception as e:
    print(f"❌ NumPy image failed: {e}\n")

time.sleep(2)

# Test 2: NumPy + Numba
print("Test 2: Building with NumPy + Numba...")
numba_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("numpy==1.26.4")
    .pip_install("numba==0.59.1")
)

app2 = modal.App("test-numpy-numba")

@app2.function(image=numba_image)
def test_numba():
    import numpy as np
    import numba
    return f"NumPy {np.__version__}, Numba {numba.__version__} installed"

try:
    with app2.run():
        print(f"Result: {test_numba.remote()}")
        print("✅ NumPy + Numba image works\n")
except Exception as e:
    print(f"❌ NumPy + Numba image failed: {e}\n")

time.sleep(2)

# Test 3: Large pip install
print("Test 3: Building with many packages at once...")
many_packages_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy==1.26.4",
        "scipy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "pillow",
        "requests",
        "tqdm",
        "pyyaml",
        "h5py",
    )
)

app3 = modal.App("test-many-packages")

@app3.function(image=many_packages_image)
def test_many():
    return "Many packages installed"

try:
    with app3.run():
        print(f"Result: {test_many.remote()}")
        print("✅ Many packages image works\n")
except Exception as e:
    print(f"❌ Many packages image failed: {e}\n")

print("\nMinimal build tests completed!")