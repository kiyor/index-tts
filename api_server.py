#!/usr/bin/env python3
"""
IndexTTS REST API Server
Provides REST API endpoints for IndexTTS text-to-speech system
"""

import asyncio
import json
import os
import sys
import time
import uuid
from typing import Dict, List, Optional, Any
from pathlib import Path
import threading
import glob
import logging
from dataclasses import dataclass, asdict

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import IndexTTS components
from indextts.infer_v2 import IndexTTS2
from gpu_configs import GPUOptimizer
from memory_monitor import GPUMemoryMonitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
tts_instance = None
task_storage = {}  # In-memory task storage
current_tasks = {}  # Currently running tasks

# ==================== Pydantic Models ====================

class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    reference_audio: Optional[str] = Field(None, description="Path to reference audio file")
    emo_audio: Optional[str] = Field(None, description="Path to emotion reference audio file")
    emo_alpha: float = Field(1.0, description="Emotion control weight (0.0-2.0)")
    emo_vector: Optional[List[float]] = Field(None, description="Emotion vector (6 values)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")

class DemoVoiceRequest(BaseModel):
    category: str = Field(..., description="Voice category")
    subcategory: str = Field(..., description="Voice subcategory")
    filename: str = Field(..., description="Audio filename")
    text: str = Field(..., description="Text to synthesize")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str
    result_url: Optional[str] = None
    error: Optional[str] = None
    progress: Optional[float] = None
    estimated_remaining: Optional[float] = None

class TTSResponse(BaseModel):
    success: bool
    task_id: str
    message: str
    audio_url: Optional[str] = None
    inference_time: Optional[float] = None
    audio_duration: Optional[float] = None

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    code: str
    details: Optional[str] = None

class SystemInfo(BaseModel):
    system: Dict[str, Any]
    model: Dict[str, Any]
    capabilities: Dict[str, Any]

class QueueStatus(BaseModel):
    queue_size: int
    current_task: Optional[Dict[str, Any]] = None
    total_completed: int
    average_execution_time: float
    estimated_wait_time: float

class VoiceCategory(BaseModel):
    name: str
    subcategories: List[Dict[str, Any]]

class VoiceInfo(BaseModel):
    filename: str
    path: str
    duration: Optional[float] = None
    sample_rate: Optional[int] = None

class AudioFile(BaseModel):
    filename: str
    path: str
    size: int
    created_time: str
    duration: Optional[float] = None

# ==================== Task Management ====================

@dataclass
class Task:
    id: str
    status: str
    created_time: float
    start_time: float = 0
    end_time: float = 0
    result_path: Optional[str] = None
    error_message: Optional[str] = None
    progress: float = 0
    request_data: Dict[str, Any] = None
    
    def to_dict(self):
        return asdict(self)

def create_task(request_data: Dict[str, Any]) -> str:
    """Create a new task and return task ID"""
    task_id = str(uuid.uuid4())[:8]
    task = Task(
        id=task_id,
        status="queued",
        created_time=time.time(),
        request_data=request_data
    )
    task_storage[task_id] = task
    logger.info(f"Created task {task_id}")
    return task_id

def get_task(task_id: str) -> Optional[Task]:
    """Get task by ID"""
    return task_storage.get(task_id)

def update_task_status(task_id: str, status: str, progress: float = None, error: str = None, result_path: str = None):
    """Update task status"""
    if task_id in task_storage:
        task = task_storage[task_id]
        task.status = status
        if progress is not None:
            task.progress = progress
        if error:
            task.error_message = error
        if result_path:
            task.result_path = result_path
        if status == "running":
            task.start_time = time.time()
        elif status in ["completed", "failed"]:
            task.end_time = time.time()
        logger.info(f"Updated task {task_id}: {status}")

# ==================== Background Tasks ====================

async def process_tts_task(task_id: str, request: TTSRequest):
    """Process TTS generation task in background"""
    try:
        update_task_status(task_id, "running", 0.1)

        # Generate output path
        output_path = f"outputs/api_task_{task_id}_{int(time.time())}.wav"
        os.makedirs("outputs", exist_ok=True)

        # Get reference audio path
        reference_audio = request.reference_audio
        if not reference_audio:
            raise ValueError("Reference audio is required")

        # Process parameters
        params = request.parameters.copy()

        update_task_status(task_id, "running", 0.3)

        # Prepare emotion parameters
        emo_audio_prompt = request.emo_audio
        emo_alpha = request.emo_alpha
        emo_vector = request.emo_vector

        # Run inference with IndexTTS2
        result = tts_instance.infer(
            spk_audio_prompt=reference_audio,
            text=request.text,
            output_path=output_path,
            emo_audio_prompt=emo_audio_prompt,
            emo_alpha=emo_alpha,
            emo_vector=emo_vector,
            verbose=False,
            **params
        )

        # IndexTTS2.infer returns the output path or generator result
        result_path = output_path if os.path.exists(output_path) else result

        update_task_status(task_id, "completed", 1.0, result_path=result_path)
        logger.info(f"Task {task_id} completed successfully")

    except Exception as e:
        error_msg = str(e)
        update_task_status(task_id, "failed", error=error_msg)
        logger.error(f"Task {task_id} failed: {error_msg}")
    finally:
        if task_id in current_tasks:
            del current_tasks[task_id]

async def process_demo_voice_task(task_id: str, request: DemoVoiceRequest):
    """Process demo voice TTS task in background"""
    try:
        update_task_status(task_id, "running", 0.1)

        # Get demo audio path
        demo_audio_path = get_demo_audio_path(request.category, request.subcategory, request.filename)
        if not demo_audio_path:
            raise ValueError(f"Demo audio not found: {request.category}/{request.subcategory}/{request.filename}")

        # Generate output path
        output_path = f"outputs/demo_task_{task_id}_{int(time.time())}.wav"
        os.makedirs("outputs", exist_ok=True)

        update_task_status(task_id, "running", 0.3)

        # Process parameters
        params = request.parameters.copy()

        # Run inference with IndexTTS2
        result = tts_instance.infer(
            spk_audio_prompt=demo_audio_path,
            text=request.text,
            output_path=output_path,
            verbose=False,
            **params
        )

        # IndexTTS2.infer returns the output path or generator result
        result_path = output_path if os.path.exists(output_path) else result

        update_task_status(task_id, "completed", 1.0, result_path=result_path)
        logger.info(f"Demo voice task {task_id} completed successfully")

    except Exception as e:
        error_msg = str(e)
        update_task_status(task_id, "failed", error=error_msg)
        logger.error(f"Demo voice task {task_id} failed: {error_msg}")
    finally:
        if task_id in current_tasks:
            del current_tasks[task_id]

# ==================== Demo Voice Functions ====================

def scan_demos_directory():
    """Scan demos directory and return structure"""
    demos_dir = "demos"
    if not os.path.exists(demos_dir):
        return {}
    
    demos_dict = {}
    try:
        for category in os.listdir(demos_dir):
            category_path = os.path.join(demos_dir, category)
            if not os.path.isdir(category_path) or category.startswith('.'):
                continue
                
            demos_dict[category] = {}
            
            for subcategory in os.listdir(category_path):
                subcategory_path = os.path.join(category_path, subcategory)
                if not os.path.isdir(subcategory_path) or subcategory.startswith('.'):
                    continue
                
                wav_files = glob.glob(os.path.join(subcategory_path, "*.wav"))
                if wav_files:
                    demos_dict[category][subcategory] = []
                    for wav_file in sorted(wav_files):
                        filename = os.path.basename(wav_file)
                        demos_dict[category][subcategory].append({
                            'name': filename,
                            'path': wav_file
                        })
    except Exception as e:
        logger.error(f"Error scanning demos directory: {e}")
    
    return demos_dict

def get_demo_audio_path(category: str, subcategory: str, filename: str) -> Optional[str]:
    """Get demo audio file path"""
    demos_dict = scan_demos_directory()
    if category in demos_dict and subcategory in demos_dict[category]:
        for audio in demos_dict[category][subcategory]:
            if audio['name'] == filename:
                return audio['path']
    return None

# ==================== FastAPI App ====================

app = FastAPI(
    title="IndexTTS API",
    description="REST API for IndexTTS text-to-speech system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# ==================== API Endpoints ====================

@app.on_event("startup")
async def startup_event():
    """Initialize TTS system on startup"""
    global tts_instance
    try:
        logger.info("Initializing IndexTTS2 system...")

        # Get GPU configuration
        gpu_config = GPUOptimizer.get_gpu_config()
        GPUOptimizer.print_gpu_info(gpu_config)

        # Initialize TTS with IndexTTS2
        tts_instance = IndexTTS2(
            model_dir="checkpoints",
            cfg_path="checkpoints/config.yaml",
            use_fp16=gpu_config.get('use_fp16', False),
            use_cuda_kernel=gpu_config.get('use_cuda_kernel', True),
            use_deepspeed=gpu_config.get('use_deepspeed', False),
        )

        logger.info("IndexTTS2 system initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize IndexTTS2: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/api/system/info", response_model=SystemInfo)
async def get_system_info():
    """Get system information"""
    import torch
    
    system_info = {
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }
    
    if torch.cuda.is_available():
        system_info["gpu_name"] = torch.cuda.get_device_name(0)
        system_info["memory_total"] = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024
    
    model_info = {
        "version": "2.0",
        "device": str(tts_instance.device),
        "fp16_enabled": tts_instance.use_fp16,
        "cuda_kernel_enabled": tts_instance.use_cuda_kernel,
    }
    
    capabilities = {
        "fast_inference": True,
        "batch_processing": True,
        "demo_voices": len(scan_demos_directory()) > 0,
        "queue_management": True,
    }
    
    return SystemInfo(
        system=system_info,
        model=model_info,
        capabilities=capabilities
    )

@app.post("/api/tts/generate", response_model=TTSResponse)
async def generate_tts(request: TTSRequest, background_tasks: BackgroundTasks):
    """Generate text-to-speech audio"""
    try:
        # Validate request
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text is required")
        
        if not request.reference_audio:
            raise HTTPException(status_code=400, detail="Reference audio is required")
        
        if not os.path.exists(request.reference_audio):
            raise HTTPException(status_code=404, detail="Reference audio file not found")
        
        # Create task
        task_id = create_task(request.dict())
        
        # Add to current tasks
        current_tasks[task_id] = time.time()
        
        # Start background processing
        background_tasks.add_task(process_tts_task, task_id, request)
        
        return TTSResponse(
            success=True,
            task_id=task_id,
            message="TTS generation started",
            audio_url=f"/api/tts/result/{task_id}"
        )
        
    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tts/status/{task_id}", response_model=TaskResponse)
async def get_task_status(task_id: str):
    """Get TTS task status"""
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Calculate estimated remaining time
    estimated_remaining = None
    if task.status == "running" and task.start_time > 0:
        elapsed = time.time() - task.start_time
        estimated_remaining = max(0, 30 - elapsed)  # Rough estimate
    
    return TaskResponse(
        task_id=task.id,
        status=task.status,
        message=f"Task {task.status}",
        result_url=f"/api/tts/result/{task_id}" if task.result_path else None,
        error=task.error_message,
        progress=task.progress,
        estimated_remaining=estimated_remaining
    )

@app.get("/api/tts/result/{task_id}")
async def get_task_result(task_id: str):
    """Get TTS task result"""
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.status == "completed" and task.result_path:
        if os.path.exists(task.result_path):
            return FileResponse(
                task.result_path,
                media_type="audio/wav",
                filename=os.path.basename(task.result_path)
            )
        else:
            raise HTTPException(status_code=404, detail="Result file not found")
    elif task.status == "failed":
        raise HTTPException(status_code=500, detail=task.error_message or "Task failed")
    else:
        raise HTTPException(status_code=202, detail="Task not completed yet")

@app.get("/api/demo/categories")
async def get_demo_categories():
    """Get available demo voice categories"""
    demos_dict = scan_demos_directory()
    categories = []
    
    for category, subcategories in demos_dict.items():
        subcategory_list = []
        for subcategory, files in subcategories.items():
            subcategory_list.append({
                "name": subcategory,
                "audio_count": len(files)
            })
        
        categories.append({
            "name": category,
            "subcategories": subcategory_list
        })
    
    return {"categories": categories}

@app.get("/api/demo/voices/{category}/{subcategory}")
async def get_demo_voices(category: str, subcategory: str):
    """Get available demo voices for category/subcategory"""
    demos_dict = scan_demos_directory()
    
    if category not in demos_dict:
        raise HTTPException(status_code=404, detail="Category not found")
    
    if subcategory not in demos_dict[category]:
        raise HTTPException(status_code=404, detail="Subcategory not found")
    
    voices = []
    for audio_info in demos_dict[category][subcategory]:
        voices.append({
            "filename": audio_info['name'],
            "path": audio_info['path'],
            "duration": None,  # Could be calculated if needed
            "sample_rate": None
        })
    
    return {
        "category": category,
        "subcategory": subcategory,
        "voices": voices
    }

@app.post("/api/demo/use", response_model=TTSResponse)
async def use_demo_voice(request: DemoVoiceRequest, background_tasks: BackgroundTasks):
    """Use demo voice for TTS generation"""
    try:
        # Validate request
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text is required")
        
        # Check if demo audio exists
        demo_path = get_demo_audio_path(request.category, request.subcategory, request.filename)
        if not demo_path:
            raise HTTPException(status_code=404, detail="Demo audio not found")
        
        # Create task
        task_id = create_task(request.dict())
        
        # Add to current tasks
        current_tasks[task_id] = time.time()
        
        # Start background processing
        background_tasks.add_task(process_demo_voice_task, task_id, request)
        
        return TTSResponse(
            success=True,
            task_id=task_id,
            message="Demo voice TTS generation started",
            audio_url=f"/api/tts/result/{task_id}"
        )
        
    except Exception as e:
        logger.error(f"Demo voice TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/queue/status", response_model=QueueStatus)
async def get_queue_status():
    """Get queue status"""
    running_tasks = len(current_tasks)
    completed_tasks = len([t for t in task_storage.values() if t.status == "completed"])
    
    current_task_info = None
    if current_tasks:
        # Get the oldest running task
        oldest_task_id = min(current_tasks.keys(), key=lambda k: current_tasks[k])
        task = get_task(oldest_task_id)
        if task:
            current_task_info = {
                "id": task.id,
                "text_preview": task.request_data.get("text", "")[:50] + "..." if len(task.request_data.get("text", "")) > 50 else task.request_data.get("text", ""),
                "elapsed_time": time.time() - task.start_time if task.start_time > 0 else 0,
                "estimated_remaining": 30  # Rough estimate
            }
    
    return QueueStatus(
        queue_size=0,  # We process immediately
        current_task=current_task_info,
        total_completed=completed_tasks,
        average_execution_time=15.0,  # Rough estimate
        estimated_wait_time=0
    )

@app.get("/api/audio/recent")
async def get_recent_audio():
    """Get recently generated audio files"""
    try:
        output_files = []
        if os.path.exists("outputs"):
            for file in sorted(glob.glob("outputs/*.wav"), key=os.path.getctime, reverse=True):
                if len(output_files) >= 10:
                    break
                
                filename = os.path.basename(file)
                stat = os.stat(file)
                output_files.append({
                    "filename": filename,
                    "path": file,
                    "size": stat.st_size,
                    "created_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_ctime)),
                    "duration": None
                })
        
        return {"files": output_files}
        
    except Exception as e:
        logger.error(f"Error getting recent audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/audio/download/{filename}")
async def download_audio(filename: str):
    """Download audio file"""
    file_path = os.path.join("outputs", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=filename
    )

# ==================== Main ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="IndexTTS API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7871, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    
    logger.info(f"Starting IndexTTS API server on {args.host}:{args.port}")
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )