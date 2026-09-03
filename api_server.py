# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

"""
A model worker executes the model.
"""
import argparse
import asyncio
import base64
import gc
import logging
import logging.handlers
import os
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import torch
import trimesh
import uvicorn
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover, FaceReducer
from hy3dgen.texgen import Hunyuan3DPaintPipeline

LOGDIR = '.'

PROJECT_ROOT = Path(__file__).resolve().parent
SUPPORTED_OUTPUT_TYPES = {"glb"}
API_ONLY_KEYS = {"face_count", "mesh", "model_profile", "seed", "texture", "type"}


@dataclass(frozen=True)
class ModelProfile:
    label: str
    local_path: str
    repository: str
    subfolder: str
    enable_flashvdm: bool
    default_steps: int


MODEL_PROFILES = {
    "fast": ModelProfile("Fast (Mini Turbo)", "models/Hunyuan3D-2mini", "tencent/Hunyuan3D-2mini",
                         "hunyuan3d-dit-v2-mini-turbo", True, 5),
    "balanced": ModelProfile("Balanced (Full Turbo)", "models/Hunyuan3D-2", "tencent/Hunyuan3D-2",
                             "hunyuan3d-dit-v2-0-turbo", True, 5),
    "quality": ModelProfile("Quality (Full)", "models/Hunyuan3D-2", "tencent/Hunyuan3D-2",
                            "hunyuan3d-dit-v2-0", False, 20),
}


class ApiError(Exception):
    def __init__(self, status_code, code, message, retryable=False, details=None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details or {}


def model_directory(model_path, subfolder):
    path = Path(model_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path / subfolder


def profile_available(profile):
    return model_directory(profile.local_path, profile.subfolder).is_dir()


def resolve_model_config(profile_name, model_path=None, subfolder=None):
    profile = MODEL_PROFILES[profile_name]
    resolved_path = model_path or profile.local_path
    resolved_subfolder = subfolder or profile.subfolder
    target = model_directory(resolved_path, resolved_subfolder)
    if model_path is None and not target.is_dir():
        raise ValueError(f"Profile '{profile_name}' is not installed at {target}")
    if model_path is not None:
        path = Path(model_path).expanduser()
        looks_local = path.is_absolute() or model_path.startswith((".", "models/"))
        if looks_local and not target.is_dir():
            raise ValueError(f"Model subfolder does not exist: {target}")
    if target.is_dir():
        resolved_path = str(target.parent)
    return resolved_path, resolved_subfolder, profile


def error_response(error):
    return JSONResponse({"error": {
        "code": error.code,
        "message": error.message,
        "retryable": error.retryable,
        "details": error.details,
    }}, status_code=error.status_code)

handler = None


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


SAVE_DIR = 'gradio_cache'
os.makedirs(SAVE_DIR, exist_ok=True)

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("controller", f"{SAVE_DIR}/controller.log")


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image, validate=True)))


class ModelWorker:
    def __init__(self, model_path, tex_model_path, subfolder, device, enable_tex,
                 profile_name, enable_flashvdm, default_steps, idle_timeout):
        self.model_path = model_path
        self.tex_model_path = tex_model_path
        self.subfolder = subfolder
        self.device = device
        self.enable_tex = enable_tex
        self.profile_name = profile_name
        self.enable_flashvdm = enable_flashvdm
        self.default_steps = default_steps
        self.idle_timeout = idle_timeout
        self.worker_id = worker_id
        self.pipeline = None
        self.pipeline_tex = None
        self.rembg = BackgroundRemover()
        self.generation_lock = threading.Lock()
        self.idle_timer = None
        self._load_pipelines()
        self._schedule_idle_unload()

    @property
    def models_loaded(self):
        return self.pipeline is not None

    def _load_pipelines(self):
        if self.pipeline is None:
            logger.info("Loading geometry profile %s (%s/%s) on worker %s",
                        self.profile_name, self.model_path, self.subfolder, self.worker_id)
            pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                self.model_path, subfolder=self.subfolder, use_safetensors=True, device=self.device)
            if self.enable_flashvdm:
                pipeline.enable_flashvdm(mc_algo='mc')
            self.pipeline = pipeline
        if self.enable_tex and self.pipeline_tex is None:
            logger.info("Loading texture pipeline %s", self.tex_model_path)
            self.pipeline_tex = Hunyuan3DPaintPipeline.from_pretrained(self.tex_model_path)

    def _cancel_idle_timer(self):
        if self.idle_timer is not None:
            self.idle_timer.cancel()
            self.idle_timer = None

    def _schedule_idle_unload(self):
        self._cancel_idle_timer()
        if self.idle_timeout > 0:
            self.idle_timer = threading.Timer(self.idle_timeout, self._unload_if_idle)
            self.idle_timer.daemon = True
            self.idle_timer.start()

    def _unload_if_idle(self):
        if not self.generation_lock.acquire(blocking=False):
            self._schedule_idle_unload()
            return
        try:
            logger.info("Idle timeout reached; unloading GPU pipelines")
            self.pipeline = None
            self.pipeline_tex = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            self.generation_lock.release()

    def acquire_generation_slot(self):
        if not self.generation_lock.acquire(blocking=False):
            raise ApiError(429, "SERVER_BUSY", "Another GPU generation is already running.", True,
                           {"profile": self.profile_name})
        self._cancel_idle_timer()

    @torch.inference_mode()
    def generate(self, uid, params, slot_acquired=False):
        if not slot_acquired:
            self.acquire_generation_slot()
        try:
            self._validate_request(params)
            self._load_pipelines()
            return self._generate(uid, dict(params))
        finally:
            self.generation_lock.release()
            self._schedule_idle_unload()

    def _validate_request(self, params):
        if 'image' not in params:
            raise ApiError(422, "IMAGE_REQUIRED",
                           "An input image is required because text-to-image is disabled.")
        requested_profile = params.get("model_profile")
        if requested_profile is not None and requested_profile not in MODEL_PROFILES:
            raise ApiError(422, "UNKNOWN_MODEL_PROFILE", f"Unknown model profile: {requested_profile}")
        if requested_profile is not None and requested_profile != self.profile_name:
            raise ApiError(409, "MODEL_PROFILE_FIXED",
                           f"This server is running the '{self.profile_name}' geometry profile.",
                           details={"active_profile": self.profile_name})
        texture = params.get('texture', False)
        if not isinstance(texture, bool):
            raise ApiError(422, "INVALID_TEXTURE", "'texture' must be a boolean.")
        if texture and not self.enable_tex:
            raise ApiError(409, "TEXTURE_DISABLED", "Texture generation is disabled on this server.")
        output_type = params.get('type', 'glb')
        if output_type not in SUPPORTED_OUTPUT_TYPES:
            raise ApiError(422, "UNSUPPORTED_OUTPUT_TYPE", "Only GLB output is supported.")
        octree_resolution = params.get("octree_resolution", 128)
        if not isinstance(octree_resolution, int) or not 16 <= octree_resolution <= 512:
            raise ApiError(422, "INVALID_OCTREE_RESOLUTION", "Octree resolution must be between 16 and 512.")
        face_count = params.get('face_count', 40000)
        if not isinstance(face_count, int) or not 1 <= face_count <= 100000:
            raise ApiError(422, "INVALID_FACE_COUNT", "Face count must be between 1 and 100000.")

    def _generate(self, uid, params):
        try:
            image = load_image_from_base64(params["image"])
            image.load()
        except Exception as exc:
            raise ApiError(422, "INVALID_IMAGE", "The image is not valid base64 image data.") from exc

        texture = params.get('texture', False)
        output_type = params.get('type', 'glb')
        octree_resolution = params.get("octree_resolution", 128)
        face_count = params.get('face_count', 40000)

        image = self.rembg(image)
        if 'mesh' in params:
            try:
                mesh = trimesh.load(BytesIO(base64.b64decode(params["mesh"], validate=True)), file_type='glb')
            except Exception as exc:
                raise ApiError(422, "INVALID_MESH", "The mesh is not valid base64 GLB data.") from exc
        else:
            pipeline_params = {key: value for key, value in params.items() if key not in API_ONLY_KEYS}
            pipeline_params.pop("text", None)
            pipeline_params['image'] = image
            pipeline_params['generator'] = torch.Generator(self.device).manual_seed(params.get("seed", 1234))
            pipeline_params['octree_resolution'] = octree_resolution
            pipeline_params['num_inference_steps'] = params.get("num_inference_steps", self.default_steps)
            pipeline_params['guidance_scale'] = params.get('guidance_scale', 5.0)
            pipeline_params['mc_algo'] = 'mc'
            import time
            start_time = time.time()
            mesh = self.pipeline(**pipeline_params)[0]
            logger.info("Geometry generation took %.2f seconds", time.time() - start_time)

        if texture:
            mesh = FloaterRemover()(mesh)
            mesh = DegenerateFaceRemover()(mesh)
            mesh = FaceReducer()(mesh, max_facenum=face_count)
            mesh = self.pipeline_tex(mesh, image)

        with tempfile.NamedTemporaryFile(suffix=f'.{output_type}', delete=False) as temp_file:
            mesh.export(temp_file.name)
            mesh = trimesh.load(temp_file.name)
            save_path = os.path.join(SAVE_DIR, f'{uid}.{output_type}')
            mesh.export(save_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return save_path, uid


app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 你可以指定允许的来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)


@app.post("/generate")
async def generate(request: Request):
    logger.info("Worker generating...")
    uid = uuid.uuid4()
    try:
        params = await request.json()
        worker.acquire_generation_slot()
        file_path, uid = await asyncio.to_thread(worker.generate, uid, params, True)
        return FileResponse(file_path)
    except ApiError as e:
        return error_response(e)
    except torch.cuda.CudaError as e:
        logger.exception("CUDA generation error")
        return error_response(ApiError(503, "CUDA_ERROR", str(e) or "GPU generation failed.", True,
                                       {"profile": worker.profile_name}))
    except Exception as e:
        logger.exception("Unexpected generation error")
        return error_response(ApiError(500, "GENERATION_FAILED", str(e)))


@app.post("/send")
async def send(request: Request):
    logger.info("Worker send...")
    uid = uuid.uuid4()
    try:
        params = await request.json()
        worker.acquire_generation_slot()
    except ApiError as e:
        return error_response(e)
    except Exception as e:
        return error_response(ApiError(400, "INVALID_REQUEST", str(e)))
    threading.Thread(target=worker.generate, args=(uid, params, True), daemon=True).start()
    ret = {"uid": str(uid)}
    return JSONResponse(ret, status_code=200)


@app.get("/health")
async def health():
    return {"status": "ready"}


@app.get("/v1/config")
async def config():
    return {
        "status": "ready",
        "geometry_profile": worker.profile_name,
        "geometry_subfolder": worker.subfolder,
        "texture_enabled": worker.enable_tex,
        "text_to_image_enabled": False,
        "request_model_selection": False,
        "models_loaded": worker.models_loaded,
        "idle_timeout_seconds": worker.idle_timeout,
        "device": str(worker.device),
    }


@app.get("/v1/models")
async def models():
    return {
        "default": worker.profile_name,
        "dynamic_selection": False,
        "models": [{
            "id": profile_id,
            "label": profile.label,
            "available": profile_available(profile),
            "active": profile_id == worker.profile_name,
        } for profile_id, profile in MODEL_PROFILES.items()],
    }


@app.get("/status/{uid}")
async def status(uid: str):
    save_file_path = os.path.join(SAVE_DIR, f'{uid}.glb')
    print(save_file_path, os.path.exists(save_file_path))
    if not os.path.exists(save_file_path):
        response = {'status': 'processing'}
        return JSONResponse(response, status_code=200)
    else:
        base64_str = base64.b64encode(open(save_file_path, 'rb').read()).decode()
        response = {'status': 'completed', 'model_base64': base64_str}
        return JSONResponse(response, status_code=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--model-profile", choices=MODEL_PROFILES, default="fast")
    parser.add_argument("--model-path", "--model_path", dest="model_path")
    parser.add_argument("--subfolder")
    parser.add_argument("--tex-model-path", "--tex_model_path", dest="tex_model_path",
                        default='models/Hunyuan3D-2')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--idle-timeout", type=float, default=300,
                        help="Unload GPU models after this many idle seconds; 0 disables unloading")
    parser.add_argument('--enable-tex', '--enable_tex', dest='enable_tex', action='store_true')
    args = parser.parse_args()
    if args.idle_timeout < 0:
        parser.error("--idle-timeout must be zero or greater")
    try:
        model_path, subfolder, profile = resolve_model_config(
            args.model_profile, args.model_path, args.subfolder)
    except ValueError as exc:
        parser.error(str(exc))
    tex_model_path = args.tex_model_path
    tex_path = Path(tex_model_path).expanduser()
    if not tex_path.is_absolute():
        local_tex_path = PROJECT_ROOT / tex_path
        if local_tex_path.is_dir():
            tex_model_path = str(local_tex_path)
    logger.info(f"args: {args}")
    worker = ModelWorker(
        model_path=model_path, tex_model_path=tex_model_path, subfolder=subfolder,
        device=args.device, enable_tex=args.enable_tex, profile_name=args.model_profile,
        enable_flashvdm=profile.enable_flashvdm, default_steps=profile.default_steps,
        idle_timeout=args.idle_timeout)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
