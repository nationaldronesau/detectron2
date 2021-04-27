"""
The environment variable "ENVIRONMENT" will be available no matter how the service is run.

    Deployment:
    circle ci passes it at the docker build stage as an argument.

    Locally:
    main.py sets it when called directly
    pytest defaults to "development", if it is not already set
"""

import os
import subprocess
import logging


# Get the name of the service from the folder name
SERVICE_NAME = subprocess.getoutput("basename $(pwd)")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Place environment specific variables here
if ENVIRONMENT == "development":
    logging.debug("Running in development mode")
    LOGGING_LEVEL = logging.DEBUG
    SMARTDATA_API_KEY = "80f4a4ac-bfc7-45ec-bcd6-6e1da4c201c9"
    SMARTDATA_ENDPOINT = "https://staging.ndsmartdata.com/api"
    # SMARTDATA_ENDPOINT = "http://0.0.0.0:8000/api"

elif ENVIRONMENT == "staging":
    logging.debug("Running in staging mode")
    LOGGING_LEVEL = logging.INFO
    # This key was copied from inspection-service
    SMARTDATA_API_KEY = "80f4a4ac-bfc7-45ec-bcd6-6e1da4c201c9"
    SMARTDATA_ENDPOINT = "https://staging.ndsmartdata.com/api"

elif ENVIRONMENT == "production":
    logging.debug("Running in production mode")
    LOGGING_LEVEL = logging.INFO
    # This key was copied from inspection-service
    SMARTDATA_API_KEY = "fb3aa83c-549f-448d-826d-cd9658b2e983"
    SMARTDATA_ENDPOINT = "https://app.ndsmartdata.com/api"

else:
    logging.error(f"Could not identify which environment should be used. ENVIRONMENT = {ENVIRONMENT}")
    raise RuntimeError(f"Could not identify which environment should be used. ENVIRONMENT = {ENVIRONMENT}")
