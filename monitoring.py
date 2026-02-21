import pandas as pd
import time
import threading
from pathlib import Path
from datetime import datetime
import json

class MonitoringAgent:
    def __init__(self, config):
        self.config = config
        self.is_monitoring = False
        self.initial_row_counts = {}
        self.callbacks = []
        
    def register_callback(self, callback):
        """Register callback for when new data is detected"""
        self.callbacks.append(callback)
        
    def get_row_count(self, file_path):
        """Get current row count of CSV file"""
        try:
            if file_path.exists():
                df = pd.read_csv(file_path)
                return len(df)
            return 0
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return 0
    
    def initialize_baseline(self):
        """Set initial row counts for monitoring"""
        self.initial_row_counts = {
            "demand_forecast": self.get_row_count(self.config.DEMAND_FORECAST_FILE),
            "inventory_optimization": self.get_row_count(self.config.INVENTORY_OPTIMIZATION_FILE)
        }
        print(f"Baseline established: {self.initial_row_counts}")
    
    def check_for_changes(self):
        """Check if new rows have been added to either file"""
        current_counts = {
            "demand_forecast": self.get_row_count(self.config.DEMAND_FORECAST_FILE),
            "inventory_optimization": self.get_row_count(self.config.INVENTORY_OPTIMIZATION_FILE)
        }
        
        changes_detected = False
        change_details = {}
        
        for file_type, current_count in current_counts.items():
            initial_count = self.initial_row_counts.get(file_type, 0)
            if current_count > initial_count:
                changes_detected = True
                change_details[file_type] = {
                    "initial": initial_count,
                    "current": current_count,
                    "new_rows": current_count - initial_count
                }
        
        if changes_detected:
            print(f"Changes detected: {change_details}")
            self.trigger_recalibration(change_details)
            # Update baseline after detection
            self.initial_row_counts = current_counts
            
        return changes_detected
    
    def trigger_recalibration(self, change_details):
        """Notify all registered callbacks about data changes"""
        message = {
            "timestamp": datetime.now().isoformat(),
            "agent": "monitoring",
            "event": "data_changed",
            "details": change_details
        }
        
        for callback in self.callbacks:
            callback(message)
    
    def start_monitoring(self, interval=30):
        """Start monitoring files for changes"""
        self.is_monitoring = True
        self.initialize_baseline()
        
        def monitor_loop():
            while self.is_monitoring:
                self.check_for_changes()
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("Monitoring started...")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        print("Monitoring stopped.")