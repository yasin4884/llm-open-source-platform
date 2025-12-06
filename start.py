#!/usr/bin/env python3
"""
🚀 HuggingFace Model Platform - Startup Script
Similar to Ollama but for open-source HuggingFace models
"""

import os
import sys
import subprocess
import time

def print_banner():
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   🤗 HuggingFace Model Platform                          ║
    ║   ─────────────────────────────────                      ║
    ║   Like Ollama, but for HuggingFace models!               ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_requirements():
    """Check if all required packages are installed"""
    required = [
        "torch",
        "transformers",
        "fastapi",
        "uvicorn",
        "pydantic"
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("📦 Installing missing packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        print("✅ Packages installed!")
    
    return True

def create_directories():
    """Create necessary directories"""
    dirs = ["models", "templates"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("📁 Directories ready")

def check_files():
    """Check if all required files exist"""
    required_files = {
        "generator.py": "QwenCodeGenerator module",
        "ModelConfigurator.py": "ModelConfigurator module",
        "backend.py": "FastAPI backend"
    }
    
    missing = []
    for file, desc in required_files.items():
        if not os.path.exists(file):
            missing.append(f"  - {file} ({desc})")
    
    if missing:
        print("❌ Missing files:")
        for m in missing:
            print(m)
        return False
    
    print("✅ All modules found")
    return True

def check_frontend():
    """Check if frontend exists"""
    frontend_path = "templates/index.html"
    if not os.path.exists(frontend_path):
        print("⚠️  Frontend not found at templates/index.html")
        print("   The API will still work, but no web UI will be available.")
        return False
    print("✅ Frontend found")
    return True

def start_server(host="0.0.0.0", port=8000, reload=False):
    """Start the FastAPI server"""
    import uvicorn
    
    print(f"""
    ┌─────────────────────────────────────────┐
    │  🌐 Server starting...                  │
    │                                         │
    │  Local:   http://localhost:{port}        │
    │  Network: http://{host}:{port}       │
    │                                         │
    │  Press Ctrl+C to stop                   │
    └─────────────────────────────────────────┘
    """)
    
    uvicorn.run(
        "backend:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

def main():
    print_banner()
    
    print("🔍 Checking requirements...")
    time.sleep(0.5)
    
    # Check requirements
    check_requirements()
    
    # Create directories
    create_directories()
    
    # Check files
    if not check_files():
        print("\n❌ Cannot start: Missing required files")
        sys.exit(1)
    
    # Check frontend
    check_frontend()
    
    print("\n" + "="*50)
    print("🚀 Starting HuggingFace Model Platform...")
    print("="*50 + "\n")
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="HuggingFace Model Platform")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=7000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Start server
    start_server(host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
