from fastapi import FastAPI, Request, UploadFile, Response, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pdf2image import convert_from_bytes
from PIL import Image

from predict.predict import predict

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/results", StaticFiles(directory="results"), name="results")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request, "id": id})

@app.post("/scan/")
async def scan(file: UploadFile = File(...)):
    content_type = file.content_type
    file_name = file.filename

    # Get Image
    if content_type == 'application/pdf':
        # Get Image from 1st page of pdf
        pages = convert_from_bytes(file.file.read())
        image = pages[0]
        file_name = file_name[:-3] + '.png'
    elif content_type.startswith('image'):
        image = Image.open(file.file)
    else:
        # Return error if Unknown file
        return Response(status_code=422)
    
    # Scan with CUTIE
    xlsx_path = predict(image, file_name)
    return xlsx_path