from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(title="Verdora AI Backend", version="0.9.5-openai-multiphoto")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DetectedIssue(BaseModel):
    label: str
    severity: str = Field(pattern="^(low|medium|high)$")
    confidence: float
    description: str


class PlantKnowledge(BaseModel):
    scientificName: str
    family: str
    origin: str
    whatItDoes: str
    benefits: list[str]
    risks: list[str]
    toxicityNote: str
    idealEnvironment: list[str]
    careRecipe: list[str]
    commonProblems: list[str]
    proTips: list[str]


class PlantDiagnosisResponse(BaseModel):
    plant_name: str
    confidence: float
    health_score: int
    summary: str
    detected_issues: list[DetectedIssue]
    recommended_actions: list[str]
    next_photo_check_at: datetime
    knowledge: PlantKnowledge


class LiveFrameResponse(BaseModel):
    plant_name: str
    confidence: float
    health_score: int
    symptoms: list[str]
    has_disease_signal: bool
    is_stable: bool


class NotificationRequest(BaseModel):
    token: str
    is_premium: bool = False
    route: str = "/reminders"
    plant_name: str | None = None


class NotificationResponse(BaseModel):
    ok: bool
    message_id: str | None = None
    mode: str


def _health_check_interval(score: int, issue_count: int) -> datetime:
    now = datetime.now(timezone.utc)
    if score < 65 or issue_count >= 2:
        return now + timedelta(days=2)
    if score < 82 or issue_count == 1:
        return now + timedelta(days=7)
    return now + timedelta(days=14)


def _coerce_ai_json(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end >= 0:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _require_openai_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Gerçek OpenAI analizi için backend_python_fastapi/.env içine OPENAI_API_KEY eklenmeli. Sahte bitki sonucu gösterilmiyor.",
        )
    return api_key




def _detect_image_mime(image_bytes: bytes, supplied_mime: str | None = None) -> str:
    """OpenAI image_url data URI sadece gerçek image MIME kabul eder.
    Telefon/Flutter bazen multipart content-type olarak application/octet-stream gönderir;
    bu durumda dosya imzasından MIME belirlenir.
    """
    mime = (supplied_mime or '').split(';')[0].strip().lower()
    supported = {'image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/gif'}
    if mime in supported:
        return 'image/jpeg' if mime == 'image/jpg' else mime

    head = image_bytes[:16]
    if head.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg'
    if head.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png'
    if head[:4] == b'RIFF' and head[8:12] == b'WEBP':
        return 'image/webp'
    if head.startswith((b'GIF87a', b'GIF89a')):
        return 'image/gif'

    raise HTTPException(
        status_code=400,
        detail='Fotoğraf biçimi okunamadı. Lütfen kamera ile yeni bir JPG/PNG fotoğraf çekip tekrar dene.',
    )



def _localized_plant_name(raw_name: str, scientific_name: str | None = None) -> str:
    combined = f"{raw_name} {scientific_name or ''}".lower()
    mapping = {
        'monstera deliciosa': 'Deve tabanı (Monstera deliciosa)',
        'monstera': 'Deve tabanı (Monstera)',
        'spathiphyllum': 'Barış çiçeği (Spathiphyllum)',
        'peace lily': 'Barış çiçeği (Spathiphyllum)',
        'epipremnum aureum': 'Salon sarmaşığı / Pothos (Epipremnum aureum)',
        'pothos': 'Salon sarmaşığı / Pothos (Epipremnum aureum)',
        'ficus elastica': 'Kauçuk bitkisi (Ficus elastica)',
        'ficus': 'Ficus türü (Ficus)',
        'sansevieria': 'Paşa kılıcı (Sansevieria)',
        'dracaena trifasciata': 'Paşa kılıcı (Dracaena trifasciata)',
        'zamioculcas': 'Zamia / ZZ bitkisi (Zamioculcas zamiifolia)',
        'zz plant': 'Zamia / ZZ bitkisi (Zamioculcas zamiifolia)',
        'chamaedorea': 'Dağ palmiyesi (Chamaedorea elegans)',
        'areca': 'Areka palmiyesi (Dypsis lutescens)',
        'dypsis lutescens': 'Areka palmiyesi (Dypsis lutescens)',
        'calathea': 'Dua çiçeği (Calathea)',
        'maranta': 'Dua çiçeği (Maranta)',
        'aloe vera': 'Aloe vera (Aloe vera)',
        'cactus': 'Kaktüs (Cactaceae)',
        'cactaceae': 'Kaktüs (Cactaceae)',
        'orchid': 'Orkide (Orchidaceae)',
        'phalaenopsis': 'Orkide (Phalaenopsis)',
        'chlorophytum': 'Kurdele çiçeği (Chlorophytum comosum)',
    }
    for key, value in mapping.items():
        if key in combined:
            return value
    raw = raw_name.strip()
    if not raw:
        return 'Olası iç mekân bitkisi'
    lowered = raw.lower()
    if lowered in {'net değil', 'tanımlanamayan bitki', 'net tanımlanamadı'}:
        return 'Olası iç mekân bitkisi'
    # Eğer AI sadece Latin ad verdiyse kullanıcıya olası ifadesiyle göster.
    parts = raw.split()
    looks_latin = len(parts) >= 2 and parts[0][:1].isupper() and parts[1][:1].islower()
    if looks_latin and '(' not in raw:
        return f'Olası: {raw}'
    return raw


def _normalize_knowledge(data: dict[str, Any], plant_name: str) -> PlantKnowledge:
    knowledge = data.get("knowledge") if isinstance(data.get("knowledge"), dict) else data

    def text(*keys: str, fallback: str) -> str:
        for key in keys:
            value = knowledge.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return fallback

    def items(*keys: str, fallback: list[str]) -> list[str]:
        for key in keys:
            value = knowledge.get(key)
            if isinstance(value, list):
                normalized = [str(item).strip() for item in value if str(item).strip()]
                if normalized:
                    return normalized[:7]
        return fallback

    return PlantKnowledge(
        scientificName=text("scientificName", "scientific_name", fallback=plant_name),
        family=text("family", fallback="AI analiziyle belirlenecek"),
        origin=text("origin", fallback="Tür doğrulamasına göre değişebilir"),
        whatItDoes=text("whatItDoes", "what_it_does", fallback="Bu bitki için kullanım ve dekoratif bilgi fotoğraf analizinden üretildi."),
        benefits=items("benefits", fallback=["Yaşam alanına doğal görünüm katar", "Düzenli bakım rutini oluşturur"]),
        risks=items("risks", fallback=["Yanlış sulama sararma veya kök sorunlarına yol açabilir", "Toksisite net değilse çocuk ve evcil hayvanlardan uzak tutulmalıdır"]),
        toxicityNote=text("toxicityNote", "toxicity_note", fallback="Toksisite kesin değilse yenmemeli ve evcil hayvanlardan uzak tutulmalıdır."),
        idealEnvironment=items("idealEnvironment", "ideal_environment", fallback=["Parlak dolaylı ışık", "Drenajlı saksı", "Sulamadan önce toprak kontrolü"]),
        careRecipe=items("careRecipe", "care_recipe", fallback=["Toprak nemini kontrol et", "Sorunlu yaprağı takip et", "7 gün sonra aynı açıdan fotoğraf çek"]),
        commonProblems=items("commonProblems", "common_problems", fallback=["Sararma", "Yaprak ucu kuruması", "Dökülme"]),
        proTips=items("proTips", "pro_tips", fallback=["Fotoğrafları aynı açı ve ışıkta çek", "Tek belirtiye göre kesin karar verme"]),
    )


PLANT_ANALYSIS_PROMPT = """
Sen Verdora AI bitki tanıma ve bakım asistanısın. Verilen görselleri birlikte analiz et ve SADECE geçerli JSON döndür.
ÖNCELİK: Görselde bir bitki görünüyorsa tür adı mutlaka üret. plant_name alanı boş kalamaz ve kullanıcıya önce Türkçe ad gösterilmelidir.
Kesin emin değilsen plant_name alanında "Olası: Deve tabanı (Monstera deliciosa)" gibi Türkçe + bilimsel ad yaz ve confidence değerini düşür.
Sadece görüntüde gerçekten bitki seçilemiyorsa "Tanımlanamayan bitki" yaz; bitki görünüyorsa asla yalnızca Latin ad veya yalnızca "Net değil" yazma.
Birden fazla görsel varsa yaprak, genel görünüm, saksı ve toprak bilgisini birlikte değerlendir.
JSON şeması:
{
  "plant_name": "Türkçe yaygın ad + parantez içinde bilimsel ad, örn: Deve tabanı (Monstera deliciosa)",
  "confidence": 0.0-1.0,
  "health_score": 0-100,
  "summary": "Tür, sağlık ve gözlenen belirti özeti Türkçe",
  "detected_issues": [{"label":"belirti", "severity":"low|medium|high", "confidence":0.0-1.0, "description":"kısa açıklama"}],
  "recommended_actions": ["kısa bakım adımı"],
  "knowledge": {
    "scientificName":"bilimsel ad",
    "family":"familya",
    "origin":"köken",
    "whatItDoes":"ne işe yarar/dekoratif veya kullanım açıklaması",
    "benefits":["fayda 1", "fayda 2", "fayda 3"],
    "risks":["risk/zarar 1", "risk/zarar 2", "risk/zarar 3"],
    "toxicityNote":"çocuk/evcil hayvan/toksisite notu",
    "idealEnvironment":["ışık", "toprak", "nem", "sıcaklık"],
    "careRecipe":["bakım adımı 1", "bakım adımı 2", "bakım adımı 3"],
    "commonProblems":["yaygın sorun 1", "yaygın sorun 2"],
    "proTips":["profesyonel ipucu 1", "profesyonel ipucu 2"]
  }
}
Profesyonel algoritma gibi davran: tür güveni, yaprak şekli, damar yapısı, yaprak rengi, leke dağılımı, solma, sararma, kahverengi uç, gövde duruşu, toprak görünümü, saksı drenajı, ışık koşulu ve fotoğraf kalitesini birlikte değerlendir.
health_score sadece hastalık değil; bakım riski, gözlenen stres ve fotoğraf kalitesine göre dengeli olmalı.
Premium olmayan kullanıcıya bile bitkinin durumu, yapılacak bakım adımları, riskler ve faydalar yeterince açık verilmelidir. Premium farkı; otomatik takvim, agresif bildirim, daha sık kontrol ve gelişim hedefidir. Kullanıcıyı yönlendiren, premium kalitesinde kısa ama zengin bakım planı üret.
İnsan sağlığı veya tıbbi tavsiye verme; yalnızca bitki bakım rehberi olarak konuş.
"""


def _input_images_payload(images: list[tuple[bytes, str]]) -> list[dict[str, str]]:
    payload: list[dict[str, str]] = []
    for image_bytes, mime_type in images:
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        safe_mime_type = _detect_image_mime(image_bytes, mime_type)
        payload.append({"type": "input_image", "image_url": f"data:{safe_mime_type};base64,{encoded}"})
    return payload


async def _analyze_with_openai_images(images: list[tuple[bytes, str]]) -> PlantDiagnosisResponse:
    _require_openai_key()
    if not images:
        raise HTTPException(status_code=400, detail="En az bir fotoğraf gerekli.")
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = os.getenv("OPENAI_VISION_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": PLANT_ANALYSIS_PROMPT},
                        *_input_images_payload(images),
                    ],
                }
            ],
        )
        raw = getattr(response, "output_text", None)
        if not raw:
            raw = str(response)
        data = _coerce_ai_json(raw)
        issues = [DetectedIssue(**item) for item in data.get("detected_issues", []) if isinstance(item, dict)]
        score = int(max(0, min(100, int(data.get("health_score", 0)))))
        plant_name = str(data.get("plant_name", "")).strip()
        if not plant_name:
            raise ValueError("AI plant_name alanı boş döndü")
        knowledge = _normalize_knowledge(data, plant_name)
        plant_name = _localized_plant_name(plant_name, knowledge.scientificName)
        return PlantDiagnosisResponse(
            plant_name=plant_name,
            confidence=float(max(0, min(1, float(data.get("confidence", 0.0))))),
            health_score=score,
            summary=str(data.get("summary", "Analiz tamamlandı.")),
            detected_issues=issues,
            recommended_actions=[str(item) for item in data.get("recommended_actions", []) if str(item).strip()],
            next_photo_check_at=_health_check_interval(score, len(issues)),
            knowledge=knowledge,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail="Fotoğraflar gerçek analiz için işlenemedi. Yaprak, bitkinin genel görünümü ve mümkünse toprak/saksı ile tekrar dene.") from exc


async def _analyze_with_openai(image_bytes: bytes, mime_type: str) -> PlantDiagnosisResponse:
    return await _analyze_with_openai_images([(image_bytes, mime_type)])


def _notification_content(is_premium: bool, plant_name: str | None) -> tuple[str, str]:
    name = plant_name or "bitkin"
    if is_premium:
        return (
            "Verdora AI bakım kontrolü",
            f"{name} için sağlık skoru, fotoğraf kontrolü ve bakım görevleri güncellendi.",
        )
    return (
        "Ücretsiz kontrolün hazır",
        f"{name} için hızlı bakım kontrolü yap. Sararma veya dökülme varsa bugün fotoğraf çek.",
    )


def _send_fcm_notification(payload: NotificationRequest) -> NotificationResponse:
    credentials_path = os.getenv("FIREBASE_ADMIN_CREDENTIALS", "").strip()
    if not credentials_path:
        raise HTTPException(status_code=503, detail="FIREBASE_ADMIN_CREDENTIALS ayarlanmadı.")
    try:
        import firebase_admin
        from firebase_admin import credentials, messaging

        if not firebase_admin._apps:
            cred = credentials.Certificate(credentials_path)
            firebase_admin.initialize_app(cred)
        title, body = _notification_content(payload.is_premium, payload.plant_name)
        message = messaging.Message(
            token=payload.token,
            notification=messaging.Notification(title=title, body=body),
            data={
                "route": payload.route,
                "premium": str(payload.is_premium).lower(),
                "plantName": payload.plant_name or "",
            },
            android=messaging.AndroidConfig(priority="high"),
            apns=messaging.APNSConfig(headers={"apns-priority": "10"}),
        )
        message_id = messaging.send(message)
        return NotificationResponse(ok=True, message_id=message_id, mode="fcm")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Bildirim gönderilemedi: {exc}") from exc


@app.post("/notifications/send", response_model=NotificationResponse)
def send_notification(payload: NotificationRequest) -> NotificationResponse:
    return _send_fcm_notification(payload)


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "service": "verdora-ai-openai",
        "openai_enabled": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "model": os.getenv("OPENAI_VISION_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
    }


@app.post("/ai/analyze-photo", response_model=PlantDiagnosisResponse)
async def analyze_photo(image: Annotated[UploadFile, File(description="High quality captured plant photo")]) -> PlantDiagnosisResponse:
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Fotoğraf dosyası boş.")
    return await _analyze_with_openai(image_bytes=image_bytes, mime_type=image.content_type or "image/jpeg")


@app.post("/ai/analyze-photo-set", response_model=PlantDiagnosisResponse)
async def analyze_photo_set(images: Annotated[list[UploadFile], File(description="Plant, leaves, pot and soil photos")]) -> PlantDiagnosisResponse:
    prepared: list[tuple[bytes, str]] = []
    for image in images[:5]:
        image_bytes = await image.read()
        if image_bytes:
            prepared.append((image_bytes, image.content_type or "image/jpeg"))
    if not prepared:
        raise HTTPException(status_code=400, detail="En az bir fotoğraf yüklenmeli.")
    return await _analyze_with_openai_images(prepared)


@app.post("/ai/live-frame", response_model=LiveFrameResponse)
async def live_frame(image: Annotated[UploadFile, File(description="Low resolution captured frame")]) -> LiveFrameResponse:
    image_bytes = await image.read()
    diagnosis = await _analyze_with_openai(image_bytes=image_bytes, mime_type=image.content_type or "image/jpeg")
    return LiveFrameResponse(
        plant_name=diagnosis.plant_name,
        confidence=diagnosis.confidence,
        health_score=diagnosis.health_score,
        symptoms=[issue.label for issue in diagnosis.detected_issues] or ["Belirgin sorun yok"],
        has_disease_signal=bool(diagnosis.detected_issues),
        is_stable=diagnosis.confidence >= 0.65,
    )
