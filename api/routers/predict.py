import numpy as np

from fastapi import APIRouter, HTTPException
from ..schemas import PredictRequest, PredictResponse
from xai_face_clustering.pipeline import PipelineRunner

router = APIRouter()
runner = PipelineRunner(...)  # load pretrained

@router.post("/", response_model=PredictResponse, summary="Cluster & explain one sample")
async def predict(req: PredictRequest):
    try:
        emb = np.array(req.embeddings)
        cluster = runner.cluster1.fit_predict(emb.reshape(1,-1))[0]
        explain = runner.explainer.local_waterfall(emb.reshape(1,-1))[0]
        return PredictResponse(cluster=cluster, explanation=explain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
