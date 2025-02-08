from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os

app = FastAPI()

# Set up templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def register(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/data", response_class=HTMLResponse)
async def data(request: Request):
    return templates.TemplateResponse("data.html", {"request": request})

@app.get("/manage", response_class=HTMLResponse)
async def manage(request: Request):
    return templates.TemplateResponse("manage.html", {"request": request, "data_rows": []})

@app.get("/manage/edit/{record_id}", response_class=HTMLResponse)
async def edit_record(request: Request, record_id: int):
    return templates.TemplateResponse("edit_record.html", {
        "request": request,
        "record_id": record_id,
        "existing_word": "example"
    })
