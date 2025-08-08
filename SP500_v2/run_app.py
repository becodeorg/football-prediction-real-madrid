"""
Simple script to run the Streamlit app
"""

import subprocess
import sys
import os

def run_streamlit_app():
    """Run the Streamlit application"""
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "app", "app.py")
    
    print("🚀 Starting S&P 500 Trading Dashboard...")
    print("📊 Open your browser and go to: http://localhost:8503")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app/app.py", 
            "--server.address", "localhost",
            "--server.port", "8503",
            "--browser.gatherUsageStats", "false"
        ], cwd=script_dir)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error running the app: {e}")
        print("Make sure streamlit is installed: pip install streamlit")

if __name__ == "__main__":
    run_streamlit_app()
