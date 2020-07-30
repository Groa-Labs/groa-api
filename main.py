import uvicorn  # type: ignore

from groa_ds_api import create_app

application = app = create_app()

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level='info')
