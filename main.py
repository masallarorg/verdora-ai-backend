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

app = FastAPI(title="Verdora AI Backend", version="0.9.0-openai")

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
Sen Verdora AI bitki tanıma ve bakım asistanısın. Fotoğrafı analiz et ve SADECE geçerli JSON döndür.
Ezbere aynı bitkiyi yazma. Fotoğraf net değilse confidence değerini düşür ve plant_name alanına "Net tanımlanamadı" yaz.
JSON şeması:
{
  "plant_name": "Türün Türkçe veya yaygın adı",
  "confidence": 0.0-1.0,
  "health_score": 0-100,
  "summary": "Tür, sağlık ve gözlenen belirti özeti Türkçe",
  "detected_issues": [{"label":"belirti", "severity":"low|medium|high", "confidence":0.0-1.0, "description":"kısa açıklama"}],
  "recommended_actions": ["kısa bakım adımı"],
  "knowledge": {
    "scientificName":"bilimsel ad veya Net değil",
    "family":"familya veya Net değil",
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
Profesyonel algoritma gibi davran: tür güveni, yaprak rengi, leke dağılımı, solma, sararma, kahverengi uç, gövde duruşu, toprak görünümü, ışık koşulu ve fotoğraf kalitesini birlikte değerlendir.
health_score sadece hastalık değil; bakım riski, gözlenen stres ve fotoğraf kalitesine göre dengeli olmalı.
İnsan sağlığı veya tıbbi tavsiye verme; yalnızca bitki bakım rehberi olarak konuş.
"""


async def _analyze_with_openai(image_bytes: bytes, mime_type: str) -> PlantDiagnosisResponse:
    _require_openai_key()
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = os.getenv("OPENAI_VISION_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:{mime_type or 'image/jpeg'};base64,{encoded}"
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": PLANT_ANALYSIS_PROMPT},
                        {"type": "input_image", "image_url": data_uri},
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
        return PlantDiagnosisResponse(
            plant_name=plant_name,
            confidence=float(max(0, min(1, float(data.get("confidence", 0.0))))),
            health_score=score,
            summary=str(data.get("summary", "Analiz tamamlandı.")),
            detected_issues=issues,
            recommended_actions=[str(item) for item in data.get("recommended_actions", []) if str(item).strip()],
            next_photo_check_at=_health_check_interval(score, len(issues)),
            knowledge=_normalize_knowledge(data, plant_name),
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"OpenAI gerçek sonuç üretemedi: {exc}") from exc


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
