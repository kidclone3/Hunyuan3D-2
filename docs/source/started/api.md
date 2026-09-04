# API

You could launch an API server locally, which you could post web request for Image/Text to 3D, Texturing existing mesh,
and e.t.c.

Choose one installed geometry profile when starting the server:

```bash
# Mini Turbo geometry; unload GPU models after five idle minutes.
uv run python api_server.py --host 0.0.0.0 --port 8080 \
  --model-profile fast --enable-tex --idle-timeout 300

# Full-quality geometry without the texture pipeline.
uv run python api_server.py --host 0.0.0.0 --port 8080 \
  --model-profile quality
```

Running `uv run python api_server.py` loads defaults from
`config/api_server.yaml`. Explicit command-line arguments override YAML values;
settings omitted from YAML use defaults defined in `api_server.py`.

Use another configuration file with `--config`:

```bash
uv run python api_server.py --config config/quality.yaml --port 8189
```

Both `--enable-tex` and `--disable-tex` are available so the YAML boolean can
be overridden in either direction.

### Authentication

All generation, status, and capability endpoints require a Bearer credential.
Set it through the environment rather than committing it to YAML:

```bash
export HUNYUAN3D_API_KEY="replace-with-a-long-random-credential"
uv run python api_server.py
```

Clients send it as an HTTP header:

```text
Authorization: Bearer replace-with-a-long-random-credential
```

`GET /health` remains unauthenticated for service monitoring. The Blender
add-on has a password-style **API Credential** field and sends this header
automatically.

For systemd, copy `deploy/hunyuan3d.service` to `/etc/systemd/system/` and
`deploy/hunyuan3d-api.env.example` to `/etc/hunyuan3d-api.env`. Replace the
example credential, set the environment file mode to `600`, then enable the
service:

```bash
sudo chmod 600 /etc/hunyuan3d-api.env
sudo systemctl daemon-reload
sudo systemctl enable --now hunyuan3d
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
     -H "Authorization: Bearer $HUNYUAN3D_API_KEY" \
     -d '{
           "image": "'"$img_b64_str"'",
         }' \
     -o test2.glb
```
