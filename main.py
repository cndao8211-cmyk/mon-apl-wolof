import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

system_prompt = """
Tu es un assistant IA intelligent, poli et serviable qui maîtrise parfaitement le Wolof et le Français.
Réponds toujours dans la langue utilisée par l'utilisateur (en Wolof s'il parle Wolof, en Français s'il parle Français).
Sois clair, naturel et précis.
"""

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MessageUtilisateur(BaseModel):
    historique: list

@app.get("/")
def accueil():
    return {"status": "Serveur IA Wolof en ligne 24h/24"}

@app.post("/chat")
@limiter.limit("15/minute")
def discuter(request: Request, msg: MessageUtilisateur):
    try:
        reponse = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=msg.historique,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt
            )
        )
        return {"reponse": reponse.text}
    except Exception as e:
        return {"reponse": f"Erreur serveur : {str(e)}"}
