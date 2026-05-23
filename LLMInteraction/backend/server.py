import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from dotenv import load_dotenv


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "Set OPENAI_API_KEY"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/voice-token")
def get_voice_token():
    url = "https://api.openai.com/v1/realtime/client_secrets"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    json_body = {
        "session": {
            "type": "realtime",
            "model": "gpt-realtime",
        }
    }
    r = requests.post(url, headers=headers, json=json_body)
    r.raise_for_status()
    data = r.json()
    # docs: ephemeral secret value is at top-level "value" :contentReference[oaicite:1]{index=1}
    return {"client_secret": data["value"]}

class Pose(BaseModel):
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float

class MoveRobotRequest(BaseModel):
    pose: Pose
    speed: float = 0.2

class UnscrewFastenerRequest(BaseModel):
    fastener_id: str
    torque_nm: float = 0.8
    direction: str  # "ccw" or "cw"

@app.post("/api/robot/move")
async def move_robot(req: MoveRobotRequest):
    # TODO: replace these prints with your real robot calls
    print("[ROBOT] move_robot:", req.pose, "speed:", req.speed)
    # e.g. call your controller here
    # robot.move_to_pose(req.pose, speed=req.speed)
    return {"status": "ok"}

@app.post("/api/robot/unscrew")
async def unscrew_fastener(req: UnscrewFastenerRequest):
    print("[ROBOT] unscrew_fastener:", req.fastener_id,
          "torque:", req.torque_nm, "direction:", req.direction)
    # robot.unscrew(req.fastener_id, torque=req.torque_nm, direction=req.direction)
    return {"status": "ok"}