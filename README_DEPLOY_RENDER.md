# Verdora AI Backend v9.3 - Render Deploy

Bu sürüm OpenAI görüntü analizinde `application/octet-stream` MIME hatasını düzeltir.

## Render ayarları

Build Command:

```bash
pip install -r requirements.txt
```

Start Command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Environment Variables:

```env
PYTHON_VERSION=3.12.11
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
OPENAI_VISION_MODEL=gpt-4o-mini
```

## Kontrol

```text
https://SENIN-RENDER-URL/health
```

`openai_enabled: true` görünmeli.

## Güncelleme

```powershell
git add .
git commit -m "Fix OpenAI image MIME handling"
git push
```

Render'da: Manual Deploy -> Clear build cache & deploy.
