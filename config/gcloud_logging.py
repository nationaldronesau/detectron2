import os
import logging
import google.cloud.logging  # Don't conflict with standard logging
from google.cloud.logging.handlers import CloudLoggingHandler, setup_logging


def setup_logging_client(log_name: str):
    """
    This function will connect to gcluod and post all logging commands there as well as locally
    The "Log Name" can then be set to the name of the service as a filter
    https://console.cloud.google.com/logs?project=development-278003
    :param log_name: The custom name of the log file that will be used in gcloud
    :return:
    """

    if os.path.exists("config/GCLOUD_LOGGING_SERVICE_KEY.json"):
        # The GCLOUD_LOGGING_SERVICE_KEY exists in circle ci, and is passed through to the service
        # There is one for each environment.  eg. development:
        # console.cloud.google.com/iam-admin/serviceaccounts/details/104042617795891603364?project=development-278003
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "config/GCLOUD_LOGGING_SERVICE_KEY.json"

        # Instantiates a client and handler for logging with gcloud
        client = google.cloud.logging.Client()
        handler = CloudLoggingHandler(name=log_name, client=client)
        logging.getLogger().setLevel(logging.INFO)  # defaults to WARN
        setup_logging(handler)

        logging.debug("Logging connected to GCloud")
    else:
        print("No GCLOUD_LOGGING_SERVICE_KEY detected, using native logging.")
