from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import os
import json
from generator import QwenCodeGenerator
from ModelConfigurator import ModelConfigurator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HuggingFace Model Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
qwen_generator = None
models_storage = {}
MODELS_DIR = "./models"
METADATA_FILE = "./models/metadata.json"


class ModelDownloadRequest(BaseModel):
    model_name: str
    model_type: str


class ChatRequest(BaseModel):
    model_name: str
    prompt: str
    max_new_tokens: Optional[int] = 256


class GenerateCodeRequest(BaseModel):
    target_model_name: str
    model_type: str


def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {}


def save_metadata(metadata):
    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)


@app.on_event("startup")
async def startup_event():
    global qwen_generator
    logger.info("🚀 Starting up the platform...")
    
    # Initialize Qwen Generator
    qwen_generator = QwenCodeGenerator()  # اینجا اصلاح شد!
    qwen_generator.load()
    logger.info("✅ Qwen Code Generator loaded successfully")
    
    # Create models directory
    os.makedirs(MODELS_DIR, exist_ok=True)
    logger.info("✅ Platform ready!")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend"""
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend not found. Place index.html in templates/</h1>")


@app.get("/api/info")
async def api_info():
    return {
        "message": "HuggingFace Model Platform API",
        "endpoints": {
            "models": "/models",
            "download": "/models/download",
            "chat": "/chat",
            "generate_code": "/generate-code"
        }
    }


@app.get("/models")
async def list_models():
    metadata = load_metadata()
    return {
        "models": list(metadata.keys()),
        "details": metadata
    }


@app.post("/models/download")
async def download_model(request: ModelDownloadRequest):
    try:
        logger.info(f"📥 Downloading model: {request.model_name} ({request.model_type})")
        
        configurator = ModelConfigurator(save_dir=MODELS_DIR)
        result = configurator.configure(
            model_name=request.model_name,
            model_type=request.model_type
        )

        models_storage[request.model_name] = configurator

        metadata = load_metadata()
        metadata[request.model_name] = {
            "type": request.model_type,
            "path": result["path"],
            "device": result["device"]
        }
        save_metadata(metadata)

        logger.info(f"✅ Model {request.model_name} downloaded successfully")
        
        return {
            "status": "success",
            "model_name": request.model_name,
            "model_type": request.model_type,
            "saved_path": result["path"]
        }

    except Exception as e:
        logger.error(f"❌ Error downloading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_with_model(request: ChatRequest):
    try:
        if request.model_name not in models_storage:
            metadata = load_metadata()
            if request.model_name not in metadata:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model '{request.model_name}' not found. Download it first."
                )

            logger.info(f"🔄 Loading model from storage: {request.model_name}")
            configurator = ModelConfigurator(save_dir=MODELS_DIR)
            configurator.configure(
                model_name=request.model_name,
                model_type=metadata[request.model_name]["type"]
            )
            models_storage[request.model_name] = configurator

        response = models_storage[request.model_name].chat(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens
        )

        return {
            "model": request.model_name,
            "prompt": request.prompt,
            "response": response
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-code")
async def generate_setup_code(request: GenerateCodeRequest):
    try:
        if qwen_generator is None:
            raise HTTPException(status_code=500, detail="Qwen generator not loaded")

        logger.info(f"🔧 Generating code for: {request.target_model_name}")
        
        code = qwen_generator.generate_setup_code(
            target_model_name=request.target_model_name,
            model_type=request.model_type
        )

        return {
            "target_model": request.target_model_name,
            "model_type": request.model_type,
            "generated_code": code
        }

    except Exception as e:
        logger.error(f"❌ Error generating code: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/models/{model_name:path}")
async def delete_model(model_name: str):
    try:
        if model_name in models_storage:
            del models_storage[model_name]

        metadata = load_metadata()
        if model_name in metadata:
            del metadata[model_name]
            save_metadata(metadata)

        logger.info(f"🗑️ Model {model_name} deleted")
        return {"status": "deleted", "model": model_name}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
