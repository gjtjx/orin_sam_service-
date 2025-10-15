"""
Simple test script for the SAM service.
Run this after starting the service to verify it's working.
"""
import requests
import sys


def test_service():
    """Test the SAM service endpoints."""
    base_url = "http://localhost:8000"
    
    print("Testing Orin SAM Service...")
    print("=" * 50)
    
    # Test 1: Root endpoint
    print("\n1. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        assert response.status_code == 200
        data = response.json()
        print(f"✓ Service: {data.get('service')}")
        print(f"✓ Version: {data.get('version')}")
        print(f"✓ Status: {data.get('status')}")
    except Exception as e:
        print(f"✗ Root endpoint failed: {e}")
        return False
    
    # Test 2: Health check
    print("\n2. Testing health check endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Status: {data.get('status')}")
            print(f"✓ Model loaded: {data.get('model_loaded')}")
            print(f"✓ Model type: {data.get('model_type')}")
        elif response.status_code == 503:
            print("⚠ Service is starting (model not loaded yet)")
        else:
            print(f"✗ Unexpected status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False
    
    # Test 3: OpenAPI docs
    print("\n3. Testing API documentation...")
    try:
        response = requests.get(f"{base_url}/docs")
        assert response.status_code == 200
        print("✓ API documentation is accessible")
    except Exception as e:
        print(f"✗ API docs failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("\nService is running at:", base_url)
    print("API Documentation:", f"{base_url}/docs")
    return True


if __name__ == "__main__":
    try:
        success = test_service()
        sys.exit(0 if success else 1)
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to service. Make sure it's running!")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        sys.exit(1)
