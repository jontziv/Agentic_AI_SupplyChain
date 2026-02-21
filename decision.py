import pandas as pd
import json
from datetime import datetime

class DecisionAgent:
    def __init__(self, config):
        self.config = config
        self.groq_client = None
        self.latest_results = None
        self.recommendations = None
        self.groq_available = False
        
        # Initialize GROQ client with error handling
        self._initialize_groq_client()
    
    def _initialize_groq_client(self):
        """Initialize GROQ client with proper error handling"""
        try:
            if not self.config.GROQ_API_KEY:
                print("⚠️ GROQ API key not found. Decision agent will work in fallback mode.")
                self.groq_available = False
                return
            
            from groq import Groq
            self.groq_client = Groq(api_key=self.config.GROQ_API_KEY)
            self.groq_available = True
            print("✅ GROQ client initialized successfully")
            
        except ImportError:
            print("⚠️ GROQ library not installed. Run: pip install groq")
            self.groq_available = False
        except Exception as e:
            print(f"⚠️ Error initializing GROQ client: {e}")
            self.groq_available = False
        
    def handle_recalibration_signal(self, message):
        """Handle signal from recalibration agent"""
        if message.get("event") == "recalibration_complete":
            print("Decision agent activated - processing new results...")
            self.process_results()
    
    def read_results_data(self):
        """Read the updated CSV files from data folder"""
        try:
            results = {}
            
            # Define the specific result files we expect
            result_files = [
                self.config.DEMAND_FORECAST_RESULTS,
                self.config.INVENTORY_OPTIMIZATION_RESULTS
            ]
            
            # Also check for any other CSV files in the data folder
            if self.config.DATA_FOLDER.exists():
                data_files = list(self.config.DATA_FOLDER.glob("*.csv"))
                result_files.extend([f for f in data_files if f not in result_files])
            
            for file_path in result_files:
                try:
                    if file_path.exists():
                        df = pd.read_csv(file_path)
                        results[file_path.name] = df
                        print(f"Loaded {len(df)} rows from {file_path.name}")
                except Exception as e:
                    print(f"Could not read {file_path}: {e}")
            
            return results
        except Exception as e:
            print(f"Error reading results data: {e}")
            return {}
    
    def generate_recommendations(self, data):
        """Use GROQ LLM to analyze results and generate recommendations"""
        if not self.groq_available:
            return self._generate_fallback_recommendations(data)
        
        try:
            # Prepare data summary for LLM
            data_summary = ""
            for filename, df in data.items():
                data_summary += f"\n{filename}:\n"
                data_summary += f"Columns: {', '.join(df.columns.tolist())}\n"
                data_summary += df.head(10).to_string() + "\n"
                if len(df) > 10:
                    data_summary += f"... (showing first 10 of {len(df)} total rows)\n"
            
            prompt = f"""
            As a supply chain AI assistant, analyze the following inventory optimization and demand forecast results:

            DATA:
            {data_summary}

            Please provide:
            1. A summary table of key metrics (reorder points, MOQ, safety stock levels, demand forecasts)
            2. Risk analysis of the recommended parameters
            3. Specific recommendations for the supply planner
            4. Any concerns or items that need attention
            5. Approval recommendation (Accept/Review/Reject) with reasoning

            Format your response in a clear, structured way that a supply planner can easily understand and act upon.
            """
            
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert supply chain analyst providing actionable insights for inventory optimization and demand forecasting."},
                    {"role": "user", "content": prompt}
                ],
                model=self.config.GROQ_MODEL,
                max_tokens=2000,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating GROQ recommendations: {e}")
            return self._generate_fallback_recommendations(data)
    
    def _generate_fallback_recommendations(self, data):
        """Generate basic recommendations when GROQ is not available"""
        recommendations = "# Supply Chain Analysis (Fallback Mode)\n\n"
        recommendations += "⚠️ **Note**: GROQ AI analysis is not available. Showing basic data summary.\n\n"
        
        for filename, df in data.items():
            recommendations += f"## {filename}\n"
            recommendations += f"- **Total Records**: {len(df)}\n"
            recommendations += f"- **Columns**: {', '.join(df.columns.tolist())}\n"
            
            # Look for common supply chain columns
            supply_chain_cols = ['reorder_point', 'safety_stock', 'moq', 'demand', 'forecast', 'inventory']
            relevant_cols = [col for col in df.columns if any(sc_col in col.lower() for sc_col in supply_chain_cols)]
            
            if relevant_cols:
                recommendations += "- **Key Supply Chain Metrics**:\n"
                for col in relevant_cols[:5]:  # Show first 5 relevant columns
                    if pd.api.types.is_numeric_dtype(df[col]) and not df[col].empty:
                        recommendations += f"  - {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, avg={df[col].mean():.2f}\n"
            
            recommendations += "\n"
        
        recommendations += "## Recommendations\n"
        recommendations += "1. **Review Model Outputs**: Examine the updated demand forecasts and reorder points\n"
        recommendations += "2. **Validate Forecasts**: Check if demand forecasts align with business expectations\n"
        recommendations += "3. **Assess Inventory Levels**: Verify that safety stock levels meet service level targets\n"
        recommendations += "4. **Check Seasonality**: Consider seasonal patterns in demand forecasting\n"
        recommendations += "5. **Supplier Constraints**: Validate MOQ against current supplier agreements\n"
        recommendations += "6. **Risk Assessment**: Review any items with significant parameter changes\n\n"
        recommendations += "**Action Required**: \n"
        recommendations += "- Set up GROQ API key for AI-powered analysis\n"
        recommendations += "- Manual review of all recommendations before implementation\n"
        
        return recommendations
    
    def process_results(self):
        """Process the results and generate recommendations"""
        # Read the updated data from results folder
        self.latest_results = self.read_results_data()
        
        if not self.latest_results:
            print("No results data found to process")
            return
        
        # Generate recommendations using GROQ or fallback
        self.recommendations = self.generate_recommendations(self.latest_results)
        print("Recommendations generated successfully")
    
    def get_recommendations(self):
        """Get the latest recommendations"""
        return {
            "timestamp": datetime.now().isoformat(),
            "data": self.latest_results,
            "recommendations": self.recommendations
        }