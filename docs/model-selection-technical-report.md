# Technical Report: Selectable Geometry Models for the Hunyuan3D API and Blender Add-on

## 1. Executive summary

This report proposes adding selectable geometry-quality profiles to the Hunyuan3D API and Blender add-on while retaining the existing Hunyuan3D-Paint texture pipeline.

The current service loads one geometry model at process startup. The geometry subfolder is fixed to `hunyuan3d-dit-v2-mini-turbo`, so changing `--model_path` alone cannot select a different architecture. The Blender add-on also has no model selector and sends only generation parameters.

The recommended first implementation is **startup-time model selection** through a validated `--model-profile` argument. It is predictable, has no request-time reload penalty, and avoids holding multiple geometry models in a 16 GB GPU. A later implementation can add dynamic request-time selection using a single-model cache, serialized generation, explicit model unloading, and CPU offloading.

Recommended deployment profiles:

| Profile | Geometry checkpoint | Intended use |
|---|---|---|
| `fast` | Hunyuan3D-2mini, `hunyuan3d-dit-v2-mini-turbo` | Interactive iteration and lowest VRAM |
| `balanced` | Hunyuan3D-2, `hunyuan3d-dit-v2-0-turbo` | Better geometry with turbo inference |
| `quality` | Hunyuan3D-2, `hunyuan3d-dit-v2-0` | Highest available local geometry quality |

Texture generation remains independent of the geometry profile and uses Hunyuan3D-Paint-v2-0-Turbo plus Hunyuan3D-Delight.

## 2. Scope

### 2.1 Goals

- Allow an operator to choose a geometry model explicitly.
- Expose understandable presets instead of arbitrary filesystem paths to Blender users.
- Keep image-to-geometry and image-to-textured-geometry behavior consistent.
- Operate safely on an NVIDIA RTX 5060 Ti with 16 GB VRAM.
- Preserve backward compatibility for existing `/generate` clients.
- Return actionable errors rather than the current generic network error.

### 2.2 Non-goals

- Training or fine-tuning models.
- Loading every geometry model into GPU memory simultaneously.
- Enabling text-to-image implicitly. The current `pipeline_t2i` initialization is disabled, so text-only generation is outside the initial model-selection change.
- Treating the custom CUDA rasterizer as a geometry model. It is a texture-rendering and texture-baking component.

## 3. Current system

### 3.1 Geometry initialization

`ModelWorker.__init__` in `api_server.py` accepts a `subfolder` argument, but the CLI does not expose it. The worker is created without passing a subfolder, which applies this default:

```text
hunyuan3d-dit-v2-mini-turbo
```

The effective startup sequence is:

1. Load background removal.
2. Load one shape pipeline.
3. Enable FlashVDM.
4. Optionally load the Delight and Paint-Turbo texture pipelines.
5. Start FastAPI/Uvicorn.

### 3.2 Request behavior

`POST /generate` accepts an image, generation settings, an optional existing mesh, and a `texture` flag. If no mesh is supplied, the shape pipeline creates one. If `texture=true`, the server cleans the mesh, reduces its face count, and passes it to the paint pipeline.

The request currently forwards most JSON keys directly to the shape pipeline. Model-selection metadata must be removed from the inference arguments before calling the pipeline.

### 3.3 Blender behavior

The Blender add-on sends requests to `{api_url}/generate`. It exposes octree resolution, inference steps, guidance scale, and texture generation, but no model profile.

If a mesh is selected and texture is enabled, the add-on sends that mesh for texturing. If no mesh is selected, it requests new geometry.

### 3.4 Current operational constraints

- The Mini-Turbo plus Paint-Turbo service occupies approximately 10.9 GB VRAM while idle in the observed environment.
- Texture generation has already encountered allocation pressure on the 16 GB GPU.
- Text-only requests fail because `self.pipeline_t2i` is not initialized.
- The API maps unrelated exceptions to a generic “network error due to high traffic” response.
- The concurrency limit is represented by a semaphore, but `/generate` does not acquire it around inference. Concurrent GPU requests can therefore cause out-of-memory failures.

## 4. Model options

### 4.1 Fast profile

```yaml
name: fast
model_path: models/Hunyuan3D-2mini
subfolder: hunyuan3d-dit-v2-mini-turbo
flashvdm: true
default_steps: 5
```

Advantages:

- Fastest response.
- Lowest shape-model memory requirement.
- Suitable for Blender iteration.
- Successfully smoke-tested on the installed RTX 5060 Ti.

Trade-off: lower geometry fidelity than the full 1.1B model.

### 4.2 Balanced profile

```yaml
name: balanced
model_path: models/Hunyuan3D-2
subfolder: hunyuan3d-dit-v2-0-turbo
flashvdm: true
default_steps: 5
```

Advantages:

- Full-size geometry network.
- Turbo inference.
- Better expected shape quality than Mini-Turbo.

Trade-offs:

- More GPU memory and model-load time.
- The checkpoint must be downloaded before enabling this profile.

### 4.3 Quality profile

```yaml
name: quality
model_path: models/Hunyuan3D-2
subfolder: hunyuan3d-dit-v2-0
flashvdm: false
default_steps: 20
```

Advantages:

- Highest-fidelity geometry option among the currently available local checkpoints.
- The checkpoint is already present locally.

Trade-offs:

- Slower inference.
- Greater memory pressure when the texture pipeline remains resident.
- FlashVDM and step defaults must be validated for this non-Turbo checkpoint rather than applied unconditionally.

## 5. Proposed architecture

## 5.1 Phase 1: startup-time selection

Add a model-profile registry and a CLI argument:

```text
--model-profile {fast,balanced,quality}
```

Retain advanced overrides:

```text
--model-path PATH_OR_REPO
--subfolder SUBFOLDER
```

Resolution order:

1. Load defaults from `--model-profile`.
2. Override the path if `--model-path` is supplied.
3. Override the subfolder if `--subfolder` is supplied.
4. Validate that the resulting local directory or remote repository configuration is valid.
5. Log the resolved model configuration.

Suggested registry shape:

```python
MODEL_PROFILES = {
    "fast": {
        "model_path": "tencent/Hunyuan3D-2mini",
        "subfolder": "hunyuan3d-dit-v2-mini-turbo",
        "enable_flashvdm": True,
        "default_steps": 5,
    },
    "balanced": {
        "model_path": "tencent/Hunyuan3D-2",
        "subfolder": "hunyuan3d-dit-v2-0-turbo",
        "enable_flashvdm": True,
        "default_steps": 5,
    },
    "quality": {
        "model_path": "tencent/Hunyuan3D-2",
        "subfolder": "hunyuan3d-dit-v2-0",
        "enable_flashvdm": False,
        "default_steps": 20,
    },
}
```

This phase does not require a Blender model dropdown because each API instance has one fixed profile. The active profile should be visible through a status endpoint so Blender can display it.

### Recommended endpoints

Add:

```text
GET /health
GET /v1/models
GET /v1/config
```

Example response from `GET /v1/config`:

```json
{
  "status": "ready",
  "geometry_profile": "fast",
  "geometry_subfolder": "hunyuan3d-dit-v2-mini-turbo",
  "texture_enabled": true,
  "text_to_image_enabled": false,
  "device": "cuda"
}
```

### Example startup commands

Fast textured service:

```bash
python api_server.py \
  --host 0.0.0.0 \
  --port 8188 \
  --model-profile fast \
  --enable_tex
```

Quality geometry service:

```bash
python api_server.py \
  --host 0.0.0.0 \
  --port 8189 \
  --model-profile quality
```

Running fast and quality services simultaneously is not recommended on one 16 GB GPU unless inactive models are CPU-resident.

## 5.2 Phase 2: request-time selection

After startup-time selection is stable, `/generate` may accept:

```json
{
  "image": "<base64>",
  "model_profile": "quality",
  "texture": true
}
```

The server must not keep every profile on the GPU. Use a single active geometry pipeline managed by a `ModelManager`:

```text
request
  -> validate profile
  -> acquire global generation lock
  -> compare requested and active profiles
  -> unload old geometry pipeline if different
  -> clear references and CUDA cache
  -> load requested geometry pipeline
  -> generate mesh
  -> release or offload geometry pipeline
  -> run texture pipeline
  -> return GLB
```

Required manager responsibilities:

- Profile validation.
- One active geometry pipeline at a time.
- Async lock or worker queue covering both model switching and inference.
- Explicit unload and garbage collection.
- CUDA cache cleanup.
- Failure rollback so a partially loaded model is not marked active.
- Metrics for load time, inference time, texture time, and peak memory.

Pseudo-interface:

```python
class ModelManager:
    async def generate(self, profile, params):
        async with self.lock:
            self.ensure_profile(profile)
            return self.pipeline(**params)

    def ensure_profile(self, profile):
        if profile == self.active_profile:
            return
        self.unload_geometry()
        self.load_geometry(profile)
```

Because the existing worker performs synchronous GPU work, either run the whole operation in a dedicated worker thread or use a single background job queue. An asyncio semaphore alone does not make synchronous CUDA inference non-blocking.

## 5.3 Texture memory strategy

The 16 GB target GPU makes memory management central to request-time selection. Recommended order:

1. Serialize all GPU generation requests.
2. Use Mini-Turbo as the default profile.
3. Reduce face count before texturing.
4. Enable Diffusers CPU offloading for Delight and Paint where compatible.
5. For quality requests, unload or move the geometry pipeline to CPU before texture inference.
6. Call `gc.collect()` and `torch.cuda.empty_cache()` only after Python references are released.
7. If full-pipeline residency remains unstable, split shape and texture into sequential worker processes.

A robust low-memory flow is:

```text
load shape -> generate mesh -> unload shape -> load/run texture -> return GLB
```

This has higher latency but a lower peak memory requirement than keeping shape and texture models resident together.

## 6. API contract

### 6.1 Backward-compatible request

If `model_profile` is omitted, use the server default:

```json
{
  "image": "<base64>",
  "texture": false,
  "octree_resolution": 128,
  "num_inference_steps": 5,
  "guidance_scale": 5.0
}
```

### 6.2 Selectable request

```json
{
  "image": "<base64>",
  "model_profile": "quality",
  "texture": true,
  "octree_resolution": 256,
  "num_inference_steps": 20,
  "guidance_scale": 5.5,
  "face_count": 40000,
  "seed": 1234,
  "type": "glb"
}
```

The API must remove `model_profile`, `texture`, `face_count`, and output-format metadata before forwarding keyword arguments to the shape pipeline.

### 6.3 Validation

Reject invalid combinations before inference:

- Unknown profile: HTTP 422.
- Missing image while text-to-image is disabled: HTTP 422 with `IMAGE_REQUIRED`.
- `texture=true` while the texture pipeline is disabled: HTTP 409 with `TEXTURE_DISABLED`.
- Unsupported output type: HTTP 422.
- Excessive octree resolution or face count: HTTP 422.
- Busy queue: HTTP 429 or a job ID, rather than launching concurrent GPU work.

Suggested error format:

```json
{
  "error": {
    "code": "CUDA_OUT_OF_MEMORY",
    "message": "Texture generation exceeded available GPU memory.",
    "retryable": true,
    "details": {
      "profile": "quality",
      "texture": true
    }
  }
}
```

Do not convert model, validation, filesystem, and CUDA errors into a generic network-traffic message.

## 7. Blender integration

Add an enum property:

```text
Model Quality
- Fast (Mini Turbo)
- Balanced (Full Turbo)
- Quality (Full)
```

The Blender add-on should first call `GET /v1/models` and display only profiles available on that server. This prevents the client from requesting missing checkpoints.

Recommended UI behavior:

- Display the server’s active model if dynamic switching is disabled.
- Enable the dropdown only when `request_model_selection=true` is advertised by the server.
- Require an image while text-to-image is unavailable.
- Clearly distinguish “Generate a new mesh” from “Texture selected mesh.”
- Warn when a selected mesh changes the operation into texturing mode.
- Add request timeout and connection-error handling.
- Display structured server errors directly.

Example model-discovery response:

```json
{
  "default": "fast",
  "dynamic_selection": true,
  "models": [
    {"id": "fast", "label": "Fast (Mini Turbo)", "available": true},
    {"id": "balanced", "label": "Balanced (Full Turbo)", "available": false},
    {"id": "quality", "label": "Quality (Full)", "available": true}
  ]
}
```

## 8. Deployment alternatives

### 8.1 One fixed-profile server

Best initial option.

- Lowest complexity.
- Predictable memory use.
- Restart required to change profile.

### 8.2 Multiple server processes

One port per profile:

```text
8188 -> fast
8189 -> quality
```

This is simple for clients but unsuitable if both processes load models onto the same 16 GB GPU. It is viable when services run on separate GPUs or load on demand.

### 8.3 One dynamic server

Best user experience after memory management is implemented.

- One URL.
- Blender dropdown works naturally.
- Model switches add load latency.
- Requires strict serialization and unload logic.

### 8.4 Separate shape and texture workers

Most scalable architecture:

```text
API gateway -> shape worker -> mesh artifact -> texture worker -> GLB
```

Benefits include isolated memory lifecycles, independent scaling, and clearer failure recovery. It is more operationally complex than needed for one workstation.

## 9. Security and reliability

- Bind to `127.0.0.1` unless LAN access is required.
- If bound to `0.0.0.0`, restrict access with a firewall or authentication.
- Replace unrestricted CORS with known Blender/web origins where applicable.
- Limit decoded image and mesh sizes before allocating memory.
- Validate base64 input and image formats.
- Use bounded temporary storage and delete artifacts after delivery.
- Never accept arbitrary model paths from an untrusted request. Requests should select only server-defined profile IDs.
- Add a maximum request duration and cancellation policy.
- Use a single GPU queue on the 16 GB workstation.

## 10. Testing plan

### 10.1 Unit tests

- Profile resolution and CLI overrides.
- Unknown-profile rejection.
- Removal of API-only keys before shape inference.
- Image-required validation.
- Texture-disabled validation.
- Structured exception mapping.

### 10.2 Integration tests

For each installed profile:

1. Start the server.
2. Confirm `/health` and `/v1/config`.
3. Generate an untextured GLB from a fixed image and seed.
4. Validate that the GLB loads and has vertices and faces.
5. Generate a textured GLB.
6. Validate UVs, material assignment, and texture images.
7. Record time and peak VRAM.

### 10.3 Model-switch tests

- Fast to quality and quality to fast.
- Repeated requests without switching.
- Failed model load followed by a valid request.
- Concurrent requests proving serialization.
- Texture generation immediately after a quality-model switch.

### 10.4 Blender acceptance tests

- Discover server capabilities.
- Generate new fast geometry.
- Generate new quality geometry.
- Generate a textured object from an image.
- Texture an existing selected mesh.
- Verify clear errors for text-only submission and unreachable server.

## 11. Rollout plan

### Phase A: configuration foundation

- Add the model registry.
- Add `--model-profile` and `--subfolder`.
- Pass the resolved subfolder to `ModelWorker`.
- Add health/config/model endpoints.
- Improve validation and errors.

### Phase B: Blender fixed-profile awareness

- Display the active server profile.
- Disable unsupported text-only operations.
- Improve operation mode and errors.

### Phase C: dynamic model manager

- Add `model_profile` to `/generate`.
- Introduce a global GPU lock and bounded queue.
- Implement unload/load transitions.
- Add memory metrics and CPU offloading.

### Phase D: Blender model dropdown

- Fetch model capabilities.
- Add quality presets.
- Show model-switch latency and low-memory warnings.

## 12. Recommendation

Implement startup-time selection first and use the following workstation configuration:

- Default service: `fast` with texture enabled for routine Blender work.
- Quality service: launch `quality` only when needed, preferably after stopping the fast service.
- Do not run two fully loaded services on the same 16 GB GPU.
- Before dynamic switching, fix request serialization and structured errors.
- For reliable quality-plus-texture jobs, unload the geometry model before texture inference or enable CPU offloading.

This approach provides immediate model choice with a small code change while establishing a safe path toward a single Blender dropdown and request-time model switching.