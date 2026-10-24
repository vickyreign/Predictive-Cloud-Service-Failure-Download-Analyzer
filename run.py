"""
Root launcher for CloudPulse.
Usage: python run.py
"""

import sys
from pathlib import Path

# Add cloudpulse directory to Python path
cloudpulse_dir = Path(__file__).resolve().parent / "cloudpulse"
sys.path.insert(0, str(cloudpulse_dir))

from app import app

if __name__ == "__main__":
    print("=" * 60)
    print("Starting CloudPulse: Predictive Failure & Downtime Analyzer")
    print("Serving on: http://localhost:5000")
    print("Demo Admin:   admin   / Admin@123")
    print("Demo Analyst: analyst / Analyst@123")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True)
