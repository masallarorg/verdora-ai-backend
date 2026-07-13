# Verdora Backend v9.5

## Yeni endpoint
`POST /ai/analyze-photo-set`

Multipart alan adı: `images`

En fazla 5 fotoğraf birlikte analiz edilir:
- genel bitki
- yaprak yakın plan
- saksı
- toprak
- sorunlu bölge

## Render deploy
```bash
git add .
git commit -m "Verdora v9.5 multi photo premium scan"
git push
```
Render içinde:
- Manual Deploy
- Clear build cache & deploy

## Environment
```env
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
OPENAI_VISION_MODEL=gpt-4o-mini
PYTHON_VERSION=3.12.11
```
