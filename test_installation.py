import sys
import importlib

def test_import(package_name):
    try:
        importlib.import_module(package_name)
        print(f"✓ {package_name} imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import {package_name}: {e}")
        return False

def main():
    print("Testing package imports...")
    print("-" * 50)
    
    packages = [
        "streamlit",
        "networkx",
        "numpy",
        "pandas",
        "plotly",
        "scipy",
        "community",  # python-louvain
        "joblib",
        "sklearn",
        "matplotlib"
    ]
    
    all_successful = all(test_import(pkg) for pkg in packages)
    
    print("-" * 50)
    if all_successful:
        print("All packages imported successfully!")
        print("You can now run the application with:")
        print("streamlit run app.py")
    else:
        print("Some packages failed to import.")
        print("Please check your installation and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main() 