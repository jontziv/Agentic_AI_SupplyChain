import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

class RecalibrationAgent:
    def __init__(self, config):
        self.config = config
        self.is_running = False
        self.callbacks = []
        
    def register_callback(self, callback):
        """Register callback for when recalibration is complete"""
        self.callbacks.append(callback)
    
    def run_script(self, script_path, script_name):
        """Run a Python script and wait for completion"""
        try:
            print(f"Starting {script_name}...")
            # Run script from the root directory (where all scripts are located)
            result = subprocess.run(
                [sys.executable, str(script_path)], 
                capture_output=True, 
                text=True, 
                check=True,
                cwd=self.config.ROOT_FOLDER  # Run from root directory
            )
            print(f"{script_name} completed successfully")
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Error running {script_name}: {e}")
            print(f"Error output: {e.stderr}")
            return False, e.stderr
        except Exception as e:
            print(f"Unexpected error running {script_name}: {e}")
            return False, str(e)
    
    def handle_monitoring_signal(self, message):
        """Handle signal from monitoring agent"""
        if message.get("event") == "data_changed" and not self.is_running:
            print(f"Recalibration triggered by: {message}")
            self.start_recalibration(message)
    
    def start_recalibration(self, trigger_message):
        """Start the recalibration process"""
        def recalibration_process():
            self.is_running = True
            
            try:
                # Ensure data directory exists
                self.config.DATA_FOLDER.mkdir(exist_ok=True)
                
                # Step 1: Run Prophet.py
                prophet_success, prophet_output = self.run_script(
                    self.config.PROPHET_SCRIPT, 
                    "Prophet.py"
                )
                
                if not prophet_success:
                    raise Exception(f"Prophet.py failed: {prophet_output}")
                
                # Step 2: Run Reorderpoint.py
                reorder_success, reorder_output = self.run_script(
                    self.config.REORDERPOINT_SCRIPT, 
                    "Reorderpoint.py"
                )
                
                if not reorder_success:
                    raise Exception(f"Reorderpoint.py failed: {reorder_output}")
                
                # Step 3: Signal completion to decision agent
                completion_message = {
                    "timestamp": datetime.now().isoformat(),
                    "agent": "recalibration",
                    "event": "recalibration_complete",
                    "trigger": trigger_message,
                    "prophet_output": prophet_output,
                    "reorder_output": reorder_output
                }
                
                for callback in self.callbacks:
                    callback(completion_message)
                
                print("Recalibration completed successfully")
                
            except Exception as e:
                print(f"Recalibration failed: {e}")
                error_message = {
                    "timestamp": datetime.now().isoformat(),
                    "agent": "recalibration",
                    "event": "recalibration_failed",
                    "error": str(e)
                }
                for callback in self.callbacks:
                    callback(error_message)
            
            finally:
                self.is_running = False
        
        # Run recalibration in a separate thread
        recalibration_thread = threading.Thread(target=recalibration_process, daemon=True)
        recalibration_thread.start()
