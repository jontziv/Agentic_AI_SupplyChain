#!/usr/bin/env python3
"""
Production-ready demand forecasting script using Prophet.
Processes SKU demand data and generates 3-month forecasts with accuracy metrics.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import logging
from typing import Dict, List, Tuple
import sys
from datetime import datetime, timedelta
import multiprocessing as mp
from functools import partial

# Configure logging to reduce debug output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress Prophet warnings and debug messages for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
logging.getLogger('cmdstan').setLevel(logging.ERROR)

class DemandForecaster:
    """Production-grade demand forecasting using Prophet."""
    
    def __init__(self, forecast_months: int = 3):
        self.forecast_months = forecast_months
        self.prophet_params = {
            'growth': 'linear',
            'seasonality_mode': 'multiplicative',
            'yearly_seasonality': True,
            'weekly_seasonality': False,
            'daily_seasonality': False,
            'changepoint_prior_scale': 0.1,  # Conservative for stability
            'seasonality_prior_scale': 1.0,
            'holidays_prior_scale': 1.0,
            'interval_width': 0.8,
            'uncertainty_samples': 100  # Reduced for speed
        }
    
    def load_and_prepare_data(self, filepath: str) -> pd.DataFrame:
        """Load and prepare the demand dataset."""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} records from {filepath}")
            
            # Validate required columns
            required_cols = ['date', 'sku', 'demand']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns. Expected: {required_cols}")
            
            # Convert date and handle parsing
            df['date'] = pd.to_datetime(df['date'])
            df['demand'] = pd.to_numeric(df['demand'], errors='coerce')
            
            # Remove invalid records
            initial_len = len(df)
            df = df.dropna(subset=['date', 'demand', 'sku'])
            if len(df) < initial_len:
                logger.warning(f"Removed {initial_len - len(df)} invalid records")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def aggregate_monthly_demand(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate daily demand to monthly level per SKU."""
        df['year_month'] = df['date'].dt.to_period('M')
        
        monthly_demand = (df.groupby(['sku', 'year_month'])['demand']
                         .sum()
                         .reset_index())
        
        # Convert back to datetime for Prophet
        monthly_demand['ds'] = monthly_demand['year_month'].dt.start_time
        monthly_demand = monthly_demand.rename(columns={'demand': 'y'})
        monthly_demand = monthly_demand[['sku', 'ds', 'y']]
        
        logger.info(f"Aggregated to {len(monthly_demand)} monthly records across {df['sku'].nunique()} SKUs")
        return monthly_demand
    
    def validate_sku_data(self, sku_data: pd.DataFrame, sku: str) -> bool:
        """Validate if SKU has sufficient data for forecasting."""
        if len(sku_data) < 12:  # Need at least 12 months
            logger.warning(f"SKU {sku}: Insufficient data ({len(sku_data)} months). Skipping.")
            return False
        
        if sku_data['y'].sum() == 0:
            logger.warning(f"SKU {sku}: Zero total demand. Skipping.")
            return False
        
        # Check for reasonable variance
        if sku_data['y'].std() == 0:
            logger.warning(f"SKU {sku}: No variance in demand. Skipping.")
            return False
        
        return True
    
    def fit_prophet_model(self, sku_data: pd.DataFrame) -> Prophet:
        """Fit Prophet model with optimized parameters."""
        # Add floor of 0 for demand (can't be negative)
        sku_data = sku_data.copy()
        sku_data['floor'] = 0
        
        # Configure Prophet with growth constraints
        model = Prophet(**self.prophet_params)
        
        # Add monthly seasonality
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        # Fit model
        model.fit(sku_data)
        return model
    
    def calculate_accuracy_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Calculate forecasting accuracy metrics."""
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        
        return {
            'MAE': round(mae, 2),
            'MSE': round(mse, 2),
            'RMSE': round(rmse, 2)
        }
    
    def forecast_sku(self, sku_data_tuple: Tuple[str, pd.DataFrame]) -> List[Dict]:
        """Forecast demand for a single SKU."""
        sku, sku_data = sku_data_tuple
        results = []
        
        try:
            if not self.validate_sku_data(sku_data, sku):
                return results
            
            # Sort by date
            sku_data = sku_data.sort_values('ds').reset_index(drop=True)
            
            # Split data for validation (use last 6 months for testing)
            split_date = sku_data['ds'].iloc[-6] if len(sku_data) > 18 else sku_data['ds'].iloc[-3]
            train_data = sku_data[sku_data['ds'] < split_date].copy()
            test_data = sku_data[sku_data['ds'] >= split_date].copy()
            
            if len(train_data) < 6:  # Need minimum training data
                logger.warning(f"SKU {sku}: Insufficient training data. Skipping.")
                return results
            
            # Fit model on training data
            model = self.fit_prophet_model(train_data)
            
            # Validate on test data
            if len(test_data) > 0:
                # Ensure test data has floor column
                test_data_pred = test_data[['ds']].copy()
                test_data_pred['floor'] = 0
                test_forecast = model.predict(test_data_pred)
                test_metrics = self.calculate_accuracy_metrics(
                    test_data['y'].values, 
                    test_forecast['yhat'].values
                )
            else:
                test_metrics = {'MAE': 0, 'MSE': 0, 'RMSE': 0}
            
            # Refit on full data for final forecast
            model = self.fit_prophet_model(sku_data)
            
            # Generate future dates
            last_date = sku_data['ds'].max()
            future_dates = pd.date_range(
                start=last_date + timedelta(days=32),
                periods=self.forecast_months,
                freq='MS'  # Month start
            )
            
            future_df = pd.DataFrame({
                'ds': future_dates,
                'floor': 0
            })
            
            forecast = model.predict(future_df)
            
            # Prepare results
            for i, row in forecast.iterrows():
                results.append({
                    'SKU': sku,
                    'Month': row['ds'].strftime('%Y-%m'),
                    'forecasted_demand': max(0, round(row['yhat'], 0)),  # Ensure non-negative
                    'MAE': test_metrics['MAE'],
                    'MSE': test_metrics['MSE'],
                    'RMSE': test_metrics['RMSE']
                })
            
            logger.info(f"SKU {sku}: Forecast completed successfully")
            
        except Exception as e:
            logger.error(f"Error forecasting SKU {sku}: {str(e)}")
        
        return results
    
    def run_forecast(self, filepath: str) -> pd.DataFrame:
        """Main method to run the complete forecasting pipeline."""
        logger.info("Starting demand forecasting pipeline")
        start_time = datetime.now()
        
        try:
            # Load and prepare data
            df = self.load_and_prepare_data(filepath)
            monthly_data = self.aggregate_monthly_demand(df)
            
            # Group by SKU
            sku_groups = list(monthly_data.groupby('sku'))
            logger.info(f"Processing {len(sku_groups)} SKUs")
            
            # Process SKUs in parallel for speed
            n_cores = min(mp.cpu_count(), len(sku_groups), 4)  # Limit cores for stability
            
            if n_cores > 1 and len(sku_groups) > 1:
                logger.info(f"Using {n_cores} cores for parallel processing")
                with mp.Pool(n_cores) as pool:
                    all_results = pool.map(self.forecast_sku, sku_groups)
            else:
                logger.info("Using single-threaded processing")
                all_results = [self.forecast_sku(sku_group) for sku_group in sku_groups]
            
            # Flatten results
            results = [result for sublist in all_results for result in sublist]
            
            if not results:
                logger.error("No forecasts generated. Check data quality and requirements.")
                return pd.DataFrame()
            
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Reorder columns
            column_order = ['SKU', 'Month', 'forecasted_demand', 'MAE', 'MSE', 'RMSE']
            results_df = results_df[column_order]
            
            # Sort by SKU and Month
            results_df = results_df.sort_values(['SKU', 'Month']).reset_index(drop=True)
            
            execution_time = datetime.now() - start_time
            logger.info(f"Forecasting completed in {execution_time.total_seconds():.1f} seconds")
            logger.info(f"Generated forecasts for {results_df['SKU'].nunique()} SKUs")
            
            return results_df
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

def main():
    """Main execution function."""
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = 'demand_forecast_dataset.csv'
    
    try:
        forecaster = DemandForecaster(forecast_months=3)
        results = forecaster.run_forecast(filepath)
        
        if not results.empty:
            # Save results to data folder
            import os
            os.makedirs('data', exist_ok=True)
            output_file = os.path.join('data', 'demand_forecast_results.csv')
            results.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
            
            # Display summary
            print("\n" + "="*60)
            print("DEMAND FORECASTING RESULTS SUMMARY")
            print("="*60)
            print(f"Total SKUs processed: {results['SKU'].nunique()}")
            print(f"Total forecasts: {len(results)}")
            print(f"Average MAE: {results['MAE'].mean():.2f}")
            print(f"Average RMSE: {results['RMSE'].mean():.2f}")
            print("\nFirst 10 rows:")
            print(results.head(10).to_string(index=False))
            print(f"\nFull results saved to: {output_file}")
        else:
            print("No results generated. Please check your data.")
    
    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()