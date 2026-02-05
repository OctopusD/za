import os
import argparse
from typing import Optional, List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager

from comet import download_model, load_from_checkpoint


# =========================
# Request / Response Schema
# =========================

class ScoreRequest(BaseModel):
    source: str
    mt: str
    ref: Optional[str] = None


class ScoreResponse(BaseModel):
    score: float
    model: str


class ScoreBatchRequest(BaseModel):
    items: List[ScoreRequest]
    batch_size: Optional[int] = None  # optional override


class ScoreBatchResponse(BaseModel):
    scores: List[float]
    model: str


# =========================
# xCOMET Engine
# =========================

class XCometEngine:
    def __init__(self, model_id_or_ckpt: str, device: str, offline: bool):
        self.model_id_or_ckpt = model_id_or_ckpt
        self.device = device
        self.offline = offline

        self.model = None
        self.model_name = None

    def load(self):
        if self.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        # 支持：本地 ckpt 或 HuggingFace 模型名
        if os.path.isfile(self.model_id_or_ckpt) and self.model_id_or_ckpt.endswith(".ckpt"):
            ckpt_path = self.model_id_or_ckpt
            self.model_name = os.path.basename(ckpt_path)
        else:
            ckpt_path = download_model(self.model_id_or_ckpt)
            self.model_name = self.model_id_or_ckpt

        self.model = load_from_checkpoint(ckpt_path)

        if "cuda" in self.device and torch.cuda.is_available():
            self.model.to(self.device)
        else:
            self.model.to("cpu")

        self.model.eval()

    def _extract_scores(self, out, n: int) -> List[float]:
        """
        兼容不同 comet 版本的 predict 返回结构，输出长度为 n 的 list[float]
        """
        # 常见：PredictOutput 对象，带 .scores 属性
        if hasattr(out, "scores"):
            scores = out.scores
            try:
                return [float(x) for x in scores]
            except Exception:
                pass

        # dict: {"scores": [...]}
        if isinstance(out, dict) and "scores" in out:
            return [float(x) for x in out["scores"]]

        # tuple: (scores, ...)
        if isinstance(out, tuple) and len(out) > 0:
            first = out[0]
            try:
                return [float(x) for x in first]
            except Exception:
                pass

        # list: could already be scores
        if isinstance(out, list):
            # list of numbers
            if len(out) == n and all(isinstance(x, (int, float)) for x in out):
                return [float(x) for x in out]
            # list of dicts with "score"
            if len(out) == n and all(isinstance(x, dict) and "score" in x for x in out):
                return [float(x["score"]) for x in out]

        raise RuntimeError(f"Unexpected output type/format: {type(out)}")

    @torch.inference_mode()
    def score(self, source: str, mt: str, ref: Optional[str] = None) -> float:
        data = [{"src": source, "mt": mt}]
        if ref is not None:
            data[0]["ref"] = ref

        out = self.model.predict(data, batch_size=1)
        scores = self._extract_scores(out, n=1)
        return float(scores[0])

    @torch.inference_mode()
    def score_batch(self, items: List[ScoreRequest], batch_size: int = 8) -> List[float]:
        n = len(items)
        data = []
        for it in items:
            d = {"src": it.source, "mt": it.mt}
            if it.ref is not None:
                d["ref"] = it.ref
            data.append(d)

        out = self.model.predict(data, batch_size=batch_size)
        scores = self._extract_scores(out, n=n)

        if len(scores) != n:
            raise RuntimeError(f"Score length mismatch: got {len(scores)}, expected {n}")

        return scores


# =========================
# FastAPI App
# =========================

def create_app(engine: XCometEngine) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # === startup ===
        engine.load()
        yield
        # === shutdown ===
        # 如需释放资源可在此处处理
        # del engine.model

    app = FastAPI(
        title="xCOMET API Server",
        version="1.1",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": engine.model_name,
            "device": str(next(engine.model.parameters()).device),
        }

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest):
        try:
            s = engine.score(req.source, req.mt, req.ref)
            return ScoreResponse(score=s, model=engine.model_name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/score_batch", response_model=ScoreBatchResponse)
    def score_batch(req: ScoreBatchRequest):
        try:
            items = req.items
            if not items:
                return ScoreBatchResponse(scores=[], model=engine.model_name)

            bs = req.batch_size or min(8, len(items))
            scores = engine.score_batch(items, batch_size=bs)
            return ScoreBatchResponse(scores=scores, model=engine.model_name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


# =========================
# Main (argparse style)
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="xCOMET Reward Model API Server (single + batch)")

    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/vjuicefs_ai_gpt/public_data/open_source_data/auto_download/huggingface/11189869/XCOMET-XL/checkpoints/model.ckpt",
        help="HuggingFace model id or local .ckpt path",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=3426, help="Port for API server")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0 / cpu")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force HF offline mode (recommended in production)",
    )

    args = parser.parse_args()

    engine = XCometEngine(
        model_id_or_ckpt=args.model_path,
        device=args.device,
        offline=args.offline,
    )

    app = create_app(engine)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,      # 必须 1，避免模型重复加载
        log_level="info",
    )
