from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    embeddings: List[List[float]]
class PredictResponse(BaseModel):
    cluster: int
    explanation: str
