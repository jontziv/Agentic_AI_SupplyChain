import time
import logging
from groq import Groq

def groq_api_call_with_retry(client, messages, model="deepseek-r1-distill-llama-70b", max_tokens=500, max_retries=3):
    """Wrapper for Groq API calls with rate limit handling"""
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens
            )
            return response
            
        except Exception as e:
            if "rate_limit_exceeded" in str(e):
                # Extract wait time from error message
                if "Please try again in" in str(e):
                    wait_time_str = str(e).split("Please try again in ")[1].split(".")[0]
                    # Parse time (e.g., "32m33" -> seconds)
                    if "m" in wait_time_str:
                        minutes = int(wait_time_str.split("m")[0])
                        wait_time = minutes * 60 + 30  # Add buffer
                    else:
                        wait_time = 300  # Default 5 minutes
                        
                    logging.warning(f"Rate limit hit. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    time.sleep(60 * (2 ** attempt))  # Exponential backoff
            else:
                raise e
    
    raise Exception("Max retries exceeded")