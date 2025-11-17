# install_requirements.py

import subprocess
import sys

# Upgrade pip first
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

# Step 1: Install packages prone to timeout separately
hard_packages = [
    "torch==2.9.1",
    "faiss-cpu==1.12.0"
]

for pkg in hard_packages:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", pkg,
        "--timeout", "300",  # increase timeout
        "--retries", "5"
    ])

# Step 2: Install the rest from requirements.txt
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-r", "requirements.txt",
    "--timeout", "300",
    "--retries", "5"
])

print("All packages installed successfully!")