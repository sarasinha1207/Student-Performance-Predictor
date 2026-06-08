import logging
import sys

# Configure logging to write exclusively to stdout (console)
# This prevents creating any log folders or files on disk, keeping the project structure clean
logging.basicConfig(
    level=logging.INFO,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
