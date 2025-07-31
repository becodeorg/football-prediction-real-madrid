#!/usr/bin/env python3
"""
Docker Deployment Test Script
Validates Docker deployment functionality
"""

import os
import sys
import json
import time
import subprocess
import requests
from datetime import datetime
from pathlib import Path

def run_command(command, timeout=30):
    """Run a shell command and return result"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def test_docker_available():
    """Test if Docker is available"""
    print("Testing Docker availability...")
    
    success, stdout, stderr = run_command("docker --version")
    if success:
        print(f"✓ Docker available: {stdout.strip()}")
        return True
    else:
        print(f"✗ Docker not available: {stderr}")
        return False

def test_docker_compose_available():
    """Test if Docker Compose is available"""
    print("Testing Docker Compose availability...")
    
    success, stdout, stderr = run_command("docker-compose --version")
    if success:
        print(f"✓ Docker Compose available: {stdout.strip()}")
        return True
    else:
        print(f"✗ Docker Compose not available: {stderr}")
        return False

def test_dockerfile_syntax():
    """Test Dockerfile syntax"""
    print("Testing Dockerfile syntax...")
    
    if not Path("Dockerfile").exists():
        print("✗ Dockerfile not found")
        return False
    
    success, stdout, stderr = run_command("docker build --dry-run -f Dockerfile .", timeout=60)
    if success or "Successfully" in stdout:
        print("✓ Dockerfile syntax valid")
        return True
    else:
        print(f"✗ Dockerfile syntax error: {stderr}")
        return False

def test_docker_compose_syntax():
    """Test docker-compose.yml syntax"""
    print("Testing Docker Compose syntax...")
    
    if not Path("docker-compose.yml").exists():
        print("✗ docker-compose.yml not found")
        return False
    
    success, stdout, stderr = run_command("docker-compose config")
    if success:
        print("✓ Docker Compose syntax valid")
        return True
    else:
        print(f"✗ Docker Compose syntax error: {stderr}")
        return False

def test_environment_setup():
    """Test environment configuration"""
    print("Testing environment setup...")
    
    # Check environment template
    if not Path(".env.production").exists():
        print("✗ .env.production template not found")
        return False
    
    # Check if .env exists
    if Path(".env").exists():
        print("✓ .env configuration file exists")
    else:
        print("⚠️  .env file not found (copy from .env.production)")
    
    # Check data directories
    data_dirs = ["docker_data/data", "docker_data/models", "docker_data/logs", "docker_data/config"]
    missing_dirs = []
    
    for dir_path in data_dirs:
        if Path(dir_path).exists():
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"✗ Missing directory: {dir_path}")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print("Creating missing directories...")
        for dir_path in missing_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"✓ Created: {dir_path}")
    
    return True

def test_image_build():
    """Test Docker image build"""
    print("Testing Docker image build...")
    
    print("Building Docker image (this may take a few minutes)...")
    success, stdout, stderr = run_command("docker build -t sp500-prediction-test .", timeout=300)
    
    if success:
        print("✓ Docker image built successfully")
        return True
    else:
        print(f"✗ Docker image build failed: {stderr}")
        return False

def test_container_run():
    """Test container startup"""
    print("Testing container startup...")
    
    # Start container in test mode
    print("Starting test container...")
    success, stdout, stderr = run_command(
        "docker run --rm -d --name sp500-test -e ENVIRONMENT=test sp500-prediction-test test",
        timeout=60
    )
    
    if not success:
        print(f"✗ Container startup failed: {stderr}")
        return False
    
    # Wait a moment for container to initialize
    time.sleep(5)
    
    # Check if container is running
    success, stdout, stderr = run_command("docker ps --filter name=sp500-test")
    if "sp500-test" in stdout:
        print("✓ Test container running")
        
        # Stop test container
        run_command("docker stop sp500-test")
        return True
    else:
        print("✗ Test container not running")
        return False

def test_health_check():
    """Test health check functionality"""
    print("Testing health check...")
    
    # Run health check in container
    success, stdout, stderr = run_command(
        "docker run --rm sp500-prediction-test python healthcheck.py",
        timeout=30
    )
    
    if success:
        print("✓ Health check passed")
        print(f"Health check output:\n{stdout}")
        return True
    else:
        print(f"✗ Health check failed: {stderr}")
        return False

def test_compose_services():
    """Test Docker Compose services"""
    print("Testing Docker Compose services...")
    
    # Start services in background
    print("Starting services with Docker Compose...")
    success, stdout, stderr = run_command("docker-compose up -d", timeout=120)
    
    if not success:
        print(f"✗ Failed to start services: {stderr}")
        return False
    
    # Wait for services to initialize
    print("Waiting for services to initialize...")
    time.sleep(30)
    
    # Check service status
    success, stdout, stderr = run_command("docker-compose ps")
    if success and "sp500-scheduler" in stdout:
        print("✓ Services started successfully")
        
        # Test dashboard accessibility (if running)
        try:
            response = requests.get("http://localhost:8501/_stcore/health", timeout=10)
            if response.status_code == 200:
                print("✓ Dashboard accessible")
            else:
                print("⚠️  Dashboard not accessible (may still be starting)")
        except:
            print("⚠️  Dashboard not accessible (may still be starting)")
        
        # Stop services
        print("Stopping services...")
        run_command("docker-compose down", timeout=60)
        return True
    else:
        print(f"✗ Services not running properly: {stdout}")
        return False

def cleanup():
    """Cleanup test resources"""
    print("Cleaning up test resources...")
    
    # Stop any running test containers
    run_command("docker stop sp500-test 2>/dev/null")
    run_command("docker-compose down 2>/dev/null")
    
    # Remove test image
    run_command("docker rmi sp500-prediction-test 2>/dev/null")
    
    print("✓ Cleanup completed")

def main():
    """Run all deployment tests"""
    print("S&P 500 Prediction System - Docker Deployment Test")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    print()
    
    tests = [
        ("Docker Available", test_docker_available),
        ("Docker Compose Available", test_docker_compose_available),
        ("Dockerfile Syntax", test_dockerfile_syntax),
        ("Docker Compose Syntax", test_docker_compose_syntax),
        ("Environment Setup", test_environment_setup),
        ("Image Build", test_image_build),
        ("Container Run", test_container_run),
        ("Health Check", test_health_check),
        ("Compose Services", test_compose_services)
    ]
    
    passed = 0
    failed = 0
    
    try:
        for test_name, test_func in tests:
            print(f"\n{'='*60}")
            print(f"TESTING: {test_name}")
            print('='*60)
            
            try:
                if test_func():
                    passed += 1
                    print(f"\n🎉 {test_name}: PASSED")
                else:
                    failed += 1
                    print(f"\n❌ {test_name}: FAILED")
            except Exception as e:
                failed += 1
                print(f"\n💥 {test_name}: ERROR - {e}")
    
    finally:
        cleanup()
    
    print(f"\n{'='*60}")
    print("DEPLOYMENT TEST SUMMARY")
    print('='*60)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {passed/len(tests)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL DEPLOYMENT TESTS PASSED!")
        print("Your Docker deployment is ready for production!")
        print("\nNext steps:")
        print("1. Copy .env.production to .env and configure")
        print("2. Run: docker-compose up -d")
        print("3. Access dashboard at: http://localhost:8501")
        print("4. Monitor logs: docker-compose logs -f")
    else:
        print(f"\n❌ {failed} test(s) failed.")
        print("Please address the issues before deploying.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
