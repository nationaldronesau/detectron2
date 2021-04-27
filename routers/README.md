## API organisation

For most applications, a single file should not be used for all of the API code and routes. \
Having a separate file for each subdirectory of the API, it makes it easier to navigate.

This also matches the [FastAPI Style Guide](https://fastapi.tiangolo.com/tutorial/bigger-applications/)

Additionally, be sure to update the tags used to match the subdirectory. This makes cleaner organisation in http://0.0.0.0:8080/docs
```python
@router.get("/greetings/casual", tags=["greetings"])
```
### Adding a new subdirectory for the API
After creating a new file in the "routers" directory, FastAPI needs to be told about the file.

For example, a file called **routers/users.py** , can be added to the **main.py** file by importing and including \
and including it into the app router. eg.
```python
from fastapi import FastAPI
from routers import users


app = FastAPI(
    title="template-service",
    description="The template-service is used as base for all microservices using python",
    version="1.1.0"
)

app.include_router(users.router)
``` 
The **routers/greetings.py** file exists as an example, and can be deleted.