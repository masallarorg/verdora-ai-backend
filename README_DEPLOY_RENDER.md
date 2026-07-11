# Verdora AI Backend - Render Deploy

Bu klasör FastAPI + OpenAI görüntü analizi backend'idir.

## Lokal test

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

`.env` içine gerçek anahtar yaz:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_VISION_MODEL=gpt-4o-mini
```

Kontrol:

```text
http://localhost:8000/health
```

## Render ayarları

- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Environment Variables:
  - `OPENAI_API_KEY`: gerçek OpenAI API key
  - `OPENAI_MODEL`: `gpt-4o-mini`
  - `OPENAI_VISION_MODEL`: `gpt-4o-mini`

Render URL örneği:

```text
https://verdora-ai-backend.onrender.com
```

Flutter build/run içinde:

```powershell
--dart-define=AI_BASE_URL=https://verdora-ai-backend.onrender.com
```

## Önemli

OpenAI key yoksa backend sahte bitki sonucu döndürmez. `/health` içinde `openai_enabled: true` görünmelidir.
