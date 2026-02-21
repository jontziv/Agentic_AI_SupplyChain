
# test_setup.py - Use this to test your configuration
import os
from dotenv import load_dotenv
from pathlib import Path

def test_environment_setup():
    """Test if environment is set up correctly"""
    print("Testing environment setup...")
    
    # Load .env file
    load_dotenv()
    
    # Check if API key is loaded
    api_key = os.getenv("GROQ_API_KEY")
    
    print(f"Current working directory: {Path.cwd()}")
    
    # Check file structure
    files_to_check = [
        "Prophet.py",
        "Reorderpoint.py", 
        "demand_forecast_dataset.csv",
        "inventory_optimisation_dataset.csv",
        ".env"
    ]
    
    print("\nFile structure check:")
    for file in files_to_check:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
    
    # Check data folder
    data_folder = Path("data")
    if data_folder.exists():
        print(f"✅ data/ folder exists")
        csv_files = list(data_folder.glob("*.csv"))
        if csv_files:
            print(f"   Found {len(csv_files)} CSV files in data/")
            for csv_file in csv_files:
                print(f"   - {csv_file.name}")
    else:
        print(f"❌ data/ folder missing")
    
    if api_key:
        print(f"\n✅ GROQ_API_KEY loaded: {api_key[:8]}...")
        
        # Test GROQ connection
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            print("✅ GROQ client initialized successfully")
            
            # Test a simple API call
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}],
                model="mixtral-8x7b-32768",
                max_tokens=10
            )
            print("✅ GROQ API connection successful")
            return True
            
        except ImportError:
            print("❌ GROQ library not installed. Run: pip install groq")
            return False
        except Exception as e:
            print(f"❌ GROQ API connection failed: {e}")
            return False
    else:
        print("\n❌ GROQ_API_KEY not found")
        print("\nSetup instructions:")
        print("1. Create a .env file in your project directory")
        print("2. Add this line to the .env file:")
        print("   GROQ_API_KEY=your_actual_api_key_here")
        print("3. Get your free API key from https://console.groq.com/")
        return False

if __name__ == "__main__":
    test_environment_setup()