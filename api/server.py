from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from ingestion.pick_recommender import DraftRecommender
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

recommender = DraftRecommender()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DraftRequest(BaseModel):
    blue_team: List[str]
    red_team: List[str]
    role: str
    side: str

@app.post("/recommend")
def recommend(data: DraftRequest):

    baseline_prob, results = recommender.recommend_pick(
        blue_team=data.blue_team,
        red_team=data.red_team,
        role=data.role,
        side=data.side,
        top_n=5
    )

    return {
    "baseline_win_probability": float(baseline_prob),
    "recommendations": results
    }

    return {"recommendations": formatted}
