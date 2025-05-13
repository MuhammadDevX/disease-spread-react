import requests
import json
import time

def test_api_connectivity():
    """Test connectivity to the API server"""
    print("\n=== Testing API Connectivity ===")
    try:
        response = requests.get("http://localhost:8000/")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.json()}")
        return True
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to the API server at http://localhost:8000/")
        print("Make sure your FastAPI server is running.")
        return False

def test_simulation_request():
    """Test a basic simulation request"""
    print("\n=== Testing Simulation Request ===")
    
    params = {
        "num_nodes": 100,
        "network_model": "watts_strogatz",
        "model_type": "SIR",
        "transmission_rate": 0.5,
        "recovery_rate": 0.1,
        "mortality_rate": 0.01,
        "latency_period": 5,
        "immunity_duration": 100,
        "healthcare_capacity": 100,
        "initial_infection_rate": 0.01,
        "steps": 50
    }
    
    try:
        print(f"Sending request with parameters: {json.dumps(params, indent=2)}")
        start_time = time.time()
        response = requests.post(
            "http://localhost:8000/simulate", 
            json=params
        )
        duration = time.time() - start_time
        
        print(f"Request completed in {duration:.2f} seconds")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("Success! Response contains:")
            print(f"- {len(data['time_series'])} time series steps")
            print(f"- {len(data['nodes'])} nodes")
            print(f"- {len(data['edges'])} edges")
            return True
        else:
            print(f"Error response: {response.text}")
            return False
    except Exception as e:
        print(f"Error during request: {str(e)}")
        return False

def run_diagnostics():
    """Run all diagnostic tests"""
    print("=== Disease Simulation API Diagnostics ===")
    
    # Test 1: API connectivity
    if not test_api_connectivity():
        print("\n❌ API server is not reachable. Please start the server and try again.")
        return
        
    # Test 2: Simulation endpoint
    if test_simulation_request():
        print("\n✅ Basic simulation request successful!")
    else:
        print("\n❌ Simulation request failed. Check the server logs for more details.")
    
if __name__ == "__main__":
    run_diagnostics()