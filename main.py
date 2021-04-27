__version__ = "2.0"

from fastapi import FastAPI
import uvicorn
import os
from config import gcloud_logging
from config.constants import SERVICE_NAME
from model_inference import *
# Import the Fast API Routers
from routers import rust#, solar_panel_defect


# Setup logging
gcloud_logging.setup_logging_client(log_name=SERVICE_NAME)

# load model
load_model()

# Create the FastAPI app object
app = FastAPI(
    title=SERVICE_NAME,
    description="The microservice that handles image detection built from machine learning.",
    version=__version__
)

# FastAPI Routers
app.include_router(rust.router)
# app.include_router(solar_panel_defect.router) # no solar model in detectron2 yet


@app.get("/")
def root() -> str:
    """
    Having something returned here makes for cleaner logs and less "404 Not Found" errors
    :return: The name and version of the service
    """
    return f"{app.title}: version {app.version}"


@app.get("/version")
def get_version() -> str:
    return app.version


if __name__ == '__main__':  # pragma: no cover
    """
    Running the app from the python file allows for easier debugging in IDE's
    """

    os.environ["ENVIRONMENT"] = "development"

    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
