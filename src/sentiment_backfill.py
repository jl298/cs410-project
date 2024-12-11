import subprocess
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    try:
        python_executable = sys.executable
        cmd = [python_executable, "sentiment_inference.py", "--backfill"]
        logging.info("Starting sentiment_inference.py with --backfill option")

        process = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

        logging.info("Successfully completed sentiment_inference.py")
        logging.info(f"Output: {process.stdout}")

    except subprocess.CalledProcessError as e:
        logging.error(f"Error running sentiment_inference.py: {str(e)}")
        logging.error(f"Error output: {e.stderr}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise