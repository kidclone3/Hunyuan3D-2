# API

You could launch an API server locally, which you could post web request for Image/Text to 3D, Texturing existing mesh,
and e.t.c.

Choose one installed geometry profile when starting the server:

```bash
# Mini Turbo geometry; unload GPU models after five idle minutes.
python api_server.py --host 0.0.0.0 --port 8080 \
  --model-profile fast --enable-tex --idle-timeout 300

# Full-quality geometry without the texture pipeline.
python api_server.py --host 0.0.0.0 --port 8080 \
  --model-profile quality
```

The available profiles are `fast`, `balanced`, and `quality`. A profile must be
installed under `models/` before it can be selected. `--model-path` and
`--subfolder` remain available as advanced overrides. Set `--idle-timeout 0` to
keep models resident indefinitely.

The service exposes capability discovery endpoints:

```text
GET /health
GET /v1/config
GET /v1/models
```

Model selection is fixed for the lifetime of the server. Concurrent GPU work is
rejected with HTTP 429. After the idle timeout, geometry and texture pipelines
are unloaded; the next valid request reloads them automatically.

A demo post request for image to 3D without texture.

```bash
img_b64_str=$(base64 -i assets/demo.png)
curl -X POST "http://localhost:8080/generate" \
     -H "Content-Type: application/json" \
     -d '{
           "image": "'"$img_b64_str"'",
         }' \
     -o test2.glb
```
