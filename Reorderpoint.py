import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class InventoryOptimizer:
    """
    Inventory optimization calculator for determining optimal reorder policies.
    """
    
    def __init__(self, service_level=0.95, ordering_cost=100, holding_cost_rate=0.25):
        """
        Initialize the optimizer with default parameters.
        
        Args:
            service_level (float): Desired service level (0.95 = 95%)
            ordering_cost (float): Cost per order placed ($)
            holding_cost_rate (float): Annual holding cost as % of item value
        """
        self.service_level = service_level
        self.z_score = self._get_z_score(service_level)
        self.ordering_cost = ordering_cost
        self.holding_cost_rate = holding_cost_rate
        
    def _get_z_score(self, service_level):
        """Get Z-score for given service level using common values."""
        z_scores = {
            0.90: 1.28, 0.95: 1.65, 0.97: 1.88, 0.99: 2.33, 0.995: 2.58
        }
        return z_scores.get(service_level, 1.65)  # Default to 95%
    
    def load_data(self, inventory_file='inventory_optimisation_dataset.csv', 
                  forecast_file='data/demand_forecast_results.csv'):
        """
        Load and prepare the datasets.
        
        Args:
            inventory_file (str): Path to historical inventory data
            forecast_file (str): Path to demand forecast data
            
        Returns:
            tuple: (historical_df, forecast_df)
        """
        try:
            # Load historical data
            historical_df = pd.read_csv(inventory_file)
            historical_df['date'] = pd.to_datetime(historical_df['date'])
            
            # Load forecast data
            forecast_df = pd.read_csv(forecast_file)
            
            print(f"Loaded {len(historical_df)} historical records")
            print(f"Loaded {len(forecast_df)} forecast records")
            
            return historical_df, forecast_df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def calculate_monthly_demand(self, historical_df):
        """
        Calculate monthly demand statistics from historical data.
        
        Args:
            historical_df (pd.DataFrame): Historical inventory data
            
        Returns:
            pd.DataFrame: Monthly demand statistics by SKU
        """
        # Add month-year column
        historical_df['month_year'] = historical_df['date'].dt.to_period('M')
        
        # Group by SKU and month to get monthly demand
        monthly_demand = historical_df.groupby(['SKU', 'month_year'])['demand'].sum().reset_index()
        
        # Calculate statistics per SKU
        demand_stats = monthly_demand.groupby('SKU')['demand'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).reset_index()
        
        demand_stats.columns = ['SKU', 'avg_monthly_demand', 'std_monthly_demand', 
                               'min_monthly_demand', 'max_monthly_demand', 'months_of_data']
        
        # Calculate daily averages (assuming 30 days per month)
        demand_stats['avg_daily_demand'] = demand_stats['avg_monthly_demand'] / 30
        demand_stats['std_daily_demand'] = demand_stats['std_monthly_demand'] / np.sqrt(30)
        
        return demand_stats
    
    def get_lead_time_stats(self, historical_df):
        """
        Calculate lead time statistics from historical data.
        
        Args:
            historical_df (pd.DataFrame): Historical inventory data
            
        Returns:
            pd.DataFrame: Lead time statistics by SKU
        """
        lead_time_stats = historical_df.groupby('SKU')['lead_time'].agg([
            'mean', 'std', 'min', 'max'
        ]).reset_index()
        
        lead_time_stats.columns = ['SKU', 'avg_lead_time', 'std_lead_time', 
                                  'min_lead_time', 'max_lead_time']
        
        return lead_time_stats
    
    def calculate_safety_stock(self, avg_daily_demand, std_daily_demand, avg_lead_time):
        """
        Calculate safety stock using standard deviation method.
        
        Formula: Safety Stock = Z × √(σ_demand² × L + μ_demand² × σ_lead_time²)
        Simplified to: Z × σ_daily_demand × √avg_lead_time (when lead time variability is low)
        
        Args:
            avg_daily_demand (float): Average daily demand
            std_daily_demand (float): Standard deviation of daily demand
            avg_lead_time (float): Average lead time in days
            
        Returns:
            float: Safety stock quantity
        """
        # Simplified formula focusing on demand variability
        safety_stock = self.z_score * std_daily_demand * np.sqrt(avg_lead_time)
        return max(0, safety_stock)  # Ensure non-negative
    
    def calculate_reorder_point(self, avg_daily_demand, avg_lead_time, safety_stock):
        """
        Calculate reorder point.
        
        Formula: ROP = (Average Daily Demand × Lead Time) + Safety Stock
        
        Args:
            avg_daily_demand (float): Average daily demand
            avg_lead_time (float): Average lead time in days
            safety_stock (float): Safety stock quantity
            
        Returns:
            float: Reorder point quantity
        """
        lead_time_demand = avg_daily_demand * avg_lead_time
        return lead_time_demand + safety_stock
    
    def calculate_eoq(self, annual_demand, unit_cost=10):
        """
        Calculate Economic Order Quantity.
        
        Formula: EOQ = √(2 × D × S / H)
        Where: D = annual demand, S = ordering cost, H = holding cost per unit per year
        
        Args:
            annual_demand (float): Annual demand in units
            unit_cost (float): Cost per unit (for holding cost calculation)
            
        Returns:
            float: Economic order quantity
        """
        holding_cost_per_unit = unit_cost * self.holding_cost_rate
        eoq = np.sqrt((2 * annual_demand * self.ordering_cost) / holding_cost_per_unit)
        return eoq
    
    def optimize_inventory(self, inventory_file='inventory_optimisation_dataset.csv', 
                          forecast_file='data/demand_forecast_results.csv'):
        """
        Main optimization function that calculates all inventory metrics.
        
        Args:
            inventory_file (str): Path to historical inventory data
            forecast_file (str): Path to demand forecast data
            
        Returns:
            pd.DataFrame: Complete optimization results
        """
        print("Starting inventory optimization...")
        
        # Load data
        historical_df, forecast_df = self.load_data(inventory_file, forecast_file)
        
        # Calculate historical statistics
        print("Calculating historical demand patterns...")
        demand_stats = self.calculate_monthly_demand(historical_df)
        lead_time_stats = self.get_lead_time_stats(historical_df)
        
        # Merge historical stats
        historical_stats = pd.merge(demand_stats, lead_time_stats, on='SKU')
        
        # Prepare results dataframe starting with forecast data
        results = forecast_df.copy()
        results = pd.merge(results, historical_stats, on='SKU', how='left')
        
        # Calculate optimization metrics for each row
        print("Calculating optimization metrics...")
        
        # Calculate annual demand from monthly forecasts
        results['annual_demand'] = results['forecasted_demand'] * 12
        
        # Calculate safety stock
        results['safety_stock'] = results.apply(
            lambda row: self.calculate_safety_stock(
                row['avg_daily_demand'], 
                row['std_daily_demand'], 
                row['avg_lead_time']
            ), axis=1
        )
        
        # Calculate reorder point
        results['reorder_point'] = results.apply(
            lambda row: self.calculate_reorder_point(
                row['avg_daily_demand'], 
                row['avg_lead_time'], 
                row['safety_stock']
            ), axis=1
        )
        
        # Calculate EOQ
        results['eoq'] = results.apply(
            lambda row: self.calculate_eoq(row['annual_demand']), axis=1
        )
        
        # Round values for practical use
        numeric_columns = ['safety_stock', 'reorder_point', 'eoq']
        for col in numeric_columns:
            results[col] = results[col].round(0).astype(int)
        
        # Select final columns for output
        final_columns = ['SKU', 'Month', 'forecasted_demand', 'reorder_point', 
                        'safety_stock', 'eoq']
        
        # Ensure all required columns exist
        if 'Month' not in results.columns:
            results['Month'] = results.get('month', results.get('Month', 'N/A'))
        
        final_results = results[final_columns].copy()
        
        print(f"Optimization complete for {len(final_results)} SKU-month combinations")
        return final_results
    
    def generate_summary_report(self, results_df):
        """
        Generate a summary report of the optimization results.
        
        Args:
            results_df (pd.DataFrame): Optimization results
            
        Returns:
            str: Summary report
        """
        report = []
        report.append("=" * 60)
        report.append("INVENTORY OPTIMIZATION SUMMARY REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Service Level: {self.service_level * 100:.1f}%")
        report.append(f"Z-Score: {self.z_score}")
        report.append("")
        
        # Overall statistics
        report.append("OVERALL STATISTICS:")
        report.append(f"Total SKU-month combinations: {len(results_df)}")
        report.append(f"Unique SKUs: {results_df['SKU'].nunique()}")
        report.append("")
        
        # By SKU summary
        report.append("SUMMARY BY SKU:")
        sku_summary = results_df.groupby('SKU').agg({
            'forecasted_demand': 'sum',
            'reorder_point': 'mean',
            'safety_stock': 'mean',
            'eoq': 'mean'
        }).round(0)
        
        for sku in sku_summary.index:
            row = sku_summary.loc[sku]
            report.append(f"{sku}:")
            report.append(f"  Annual Forecast: {row['forecasted_demand']:,.0f} units")
            report.append(f"  Avg Reorder Point: {row['reorder_point']:,.0f} units")
            report.append(f"  Avg Safety Stock: {row['safety_stock']:,.0f} units")
            report.append(f"  EOQ: {row['eoq']:,.0f} units")
            report.append("")
        
        return "\n".join(report)


def main():
    """
    Main execution function.
    """
    try:
        import os
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Initialize optimizer with production parameters
        optimizer = InventoryOptimizer(
            service_level=0.95,      # 95% service level
            ordering_cost=100,       # $100 per order
            holding_cost_rate=0.25   # 25% annual holding cost rate
        )
        
        # Run optimization
        results = optimizer.optimize_inventory()
        
        # Display results
        print("\n" + "="*80)
        print("INVENTORY OPTIMIZATION RESULTS")
        print("="*80)
        print(results.to_string(index=False))
        
        # Save results to CSV in data folder
        output_file = 'data/inventory_optimization_results.csv'
        results.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        # Generate and display summary report
        summary = optimizer.generate_summary_report(results)
        print("\n" + summary)
        
        # Save summary report in data folder
        report_file = 'data/inventory_optimization_summary.txt'
        with open(report_file, 'w') as f:
            f.write(summary)
        print(f"Summary report saved to: {report_file}")
        
        return results
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    results = main()