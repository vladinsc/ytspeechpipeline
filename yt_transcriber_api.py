"""Authenticated asynchronous API for the GPU transcription pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl

from batch_pipeline import atomic_write_json, nvidia_smi_snapshot, utc_now, video_label
from speech_pipeline import GPUTranscriber, PipelineConfig, ProsodyPipeline, VocalIsolator, resolve_device

log = logging.getLogger("yt-transcriber-api")
RESULTS_DIR = Path(os.environ.get("YT_TRANSCRIBER_RESULTS_DIR", "/data/results"))
WORK_DIR = Path(os.environ.get("YT_TRANSCRIBER_WORK_DIR", "/data/work"))
JOBS_PATH = RESULTS_DIR / "api_jobs.json"
DEVICE_REQUEST = os.environ.get("YT_TRANSCRIBER_DEVICE", "cuda")
DEMUCS_DEVICE_REQUEST = os.environ.get("YT_TRANSCRIBER_DEMUCS_DEVICE", DEVICE_REQUEST)
ALIGNMENT_DEVICE_REQUEST = os.environ.get("YT_TRANSCRIBER_ALIGNMENT_DEVICE", DEVICE_REQUEST)
WHISPER_MODEL = os.environ.get("YT_TRANSCRIBER_WHISPER_MODEL", "large-v3")
DEMUCS_MODEL = os.environ.get("YT_TRANSCRIBER_DEMUCS_MODEL", "htdemucs")
COMPUTE_TYPE = os.environ.get("YT_TRANSCRIBER_COMPUTE_TYPE", "int8")


class JobRequest(BaseModel):
    url: HttpUrl
    label: Literal["kids", "normal"]
    granularity: Literal["word", "syllable", "both"] = "both"
    pitch_floor: float = 75.0
    pitch_ceiling: float = 600.0


class JobStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()
        self.data = {"schema_version": 1, "updated_at": utc_now(), "summary": {}, "jobs": {}}
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
        for job in self.data.get("jobs", {}).values():
            if job["status"] == "processing":
                job.update(status="queued", stage_status="restart_queued",
                           detail="API restarted; job queued from the beginning.")
        self.save()

    def save(self) -> None:
        with self.lock:
            jobs = list(self.data["jobs"].values())
            self.data["updated_at"] = utc_now()
            self.data["summary"] = {
                "total": len(jobs),
                "queued": sum(job["status"] == "queued" for job in jobs),
                "processing": sum(job["status"] == "processing" for job in jobs),
                "completed": sum(job["status"] == "completed" for job in jobs),
                "failed": sum(job["status"] == "failed" for job in jobs),
            }
            atomic_write_json(self.path, self.data)

    def create(self, request: JobRequest) -> dict:
        url = str(request.url)
        if not url.startswith(("https://www.youtube.com/", "https://youtube.com/", "https://youtu.be/")):
            raise ValueError("Only public youtube.com and youtu.be URLs are accepted")
        job_id = uuid.uuid4().hex[:16]
        stem = f"api_{job_id}_{video_label(url)}"
        job = {
            "job_id": job_id, "url": url, "label": request.label,
            "granularity": request.granularity, "pitch_floor": request.pitch_floor,
            "pitch_ceiling": request.pitch_ceiling, "status": "queued", "stage": "waiting",
            "stage_status": "queued", "detail": "", "created_at": utc_now(),
            "updated_at": utc_now(), "started_at": None, "completed_at": None,
            "output_path": str((RESULTS_DIR / request.label / f"{stem}.json").resolve()),
            "workdir": str((WORK_DIR / request.label / stem).resolve()),
            "record_counts": None, "error": None,
        }
        with self.lock:
            self.data["jobs"][job_id] = job
            self.save()
        return dict(job)

    def get(self, job_id: str) -> dict | None:
        with self.lock:
            job = self.data["jobs"].get(job_id)
            return dict(job) if job else None

    def update(self, job_id: str, **changes) -> None:
        with self.lock:
            self.data["jobs"][job_id].update(changes, updated_at=utc_now())
            self.save()


class Runtime:
    ready = False
    startup_error: str | None = None


runtime = Runtime()


def public_job(job: dict) -> dict:
    return {key: value for key, value in job.items() if key not in {"workdir", "output_path"}}


def count_records(payload: list | dict) -> dict[str, int]:
    if isinstance(payload, list):
        return {"records": len(payload)}
    return {key: len(value) for key, value in payload.items() if isinstance(value, list)}


def process_job(job_id: str) -> None:
    job = runtime.store.get(job_id)
    if job is None:
        return
    runtime.store.update(job_id, status="processing", stage="initializing",
                         stage_status="started", started_at=utc_now(), error=None)

    def progress(stage: str, stage_status: str, detail: str) -> None:
        runtime.store.update(job_id, stage=stage, stage_status=stage_status, detail=detail)

    cfg = PipelineConfig(
        url=job["url"], out_path=Path(job["output_path"]), workdir=Path(job["workdir"]),
        device=runtime.device, whisper_model=WHISPER_MODEL, compute_type=COMPUTE_TYPE,
        demucs_device=runtime.demucs_device, alignment_device=runtime.alignment_device,
        demucs_model=DEMUCS_MODEL, language="en", pitch_floor=float(job["pitch_floor"]),
        pitch_ceiling=float(job["pitch_ceiling"]), keep_workdir=False,
        granularity=job["granularity"], label=job["label"],
    )
    try:
        payload = ProsodyPipeline(cfg, device=runtime.device).run(
            isolator=runtime.isolator, transcriber=runtime.transcriber,
            progress_callback=progress,
        )
        runtime.store.update(
            job_id, status="completed", stage="finished", stage_status="completed",
            detail="Result is available from the result endpoint.",
            record_counts=count_records(payload), completed_at=utc_now(),
        )
    except Exception as exc:
        log.exception("Job %s failed", job_id)
        runtime.store.update(job_id, status="failed", stage_status="failed",
                             error=f"{type(exc).__name__}: {exc}")


async def worker() -> None:
    while True:
        job_id = await runtime.queue.get()
        try:
            await asyncio.to_thread(process_job, job_id)
        finally:
            runtime.queue.task_done()


@asynccontextmanager
async def lifespan(_: FastAPI):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    runtime.store = JobStore(JOBS_PATH)
    runtime.queue = asyncio.Queue()
    try:
        runtime.device = resolve_device(DEVICE_REQUEST)
        runtime.demucs_device = DEMUCS_DEVICE_REQUEST
        runtime.alignment_device = ALIGNMENT_DEVICE_REQUEST
        gpu = nvidia_smi_snapshot()
        runtime.store.data["gpu"] = gpu
        runtime.store.save()
        if runtime.device == "cuda" and not gpu["available"]:
            raise RuntimeError(f"nvidia-smi unavailable: {gpu.get('error')}")
        log.info(
            "Stage devices: WhisperX=%s, Demucs=%s, alignment=%s, Silero/Praat=cpu",
            runtime.device, runtime.demucs_device, runtime.alignment_device,
        )
        runtime.isolator = await asyncio.to_thread(
            VocalIsolator, runtime.demucs_device, DEMUCS_MODEL
        )
        runtime.transcriber = await asyncio.to_thread(
            GPUTranscriber,
            runtime.device,
            WHISPER_MODEL,
            COMPUTE_TYPE,
            1,
            "en",
            runtime.alignment_device,
        )
        runtime.worker_task = asyncio.create_task(worker())
        for job in runtime.store.data["jobs"].values():
            if job["status"] == "queued":
                runtime.queue.put_nowait(job["job_id"])
        runtime.ready = True
        yield
    except Exception as exc:
        runtime.startup_error = f"{type(exc).__name__}: {exc}"
        log.exception("API startup failed")
        raise
    finally:
        runtime.ready = False
        if hasattr(runtime, "worker_task"):
            runtime.worker_task.cancel()


ROOT_PATH = os.environ.get("YT_TRANSCRIBER_ROOT_PATH", "").rstrip("/")
app = FastAPI(title="yt-transcriber", version="1.0.0", root_path=ROOT_PATH, lifespan=lifespan)


@app.get("/")
def root() -> dict:
    return {"service": "yt-transcriber", "docs": f"{ROOT_PATH}/docs",
            "health": f"{ROOT_PATH}/health"}


@app.get("/health")
def health() -> JSONResponse:
    body = {"service": "yt-transcriber", "ready": runtime.ready,
            "device": getattr(runtime, "device", None), "error": runtime.startup_error}
    return JSONResponse(body, status_code=200 if runtime.ready else 503)


@app.post("/v1/jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_job(request: JobRequest) -> dict:
    if not runtime.ready:
        raise HTTPException(status_code=503, detail="Models are not ready")
    if not 40 <= request.pitch_ceiling <= 1200 or not 20 <= request.pitch_floor < request.pitch_ceiling:
        raise HTTPException(status_code=422, detail="Invalid pitch range")
    try:
        job = runtime.store.create(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    await runtime.queue.put(job["job_id"])
    return public_job(job)


@app.get("/v1/jobs")
def list_jobs() -> dict:
    jobs = [public_job(job) for job in runtime.store.data["jobs"].values()]
    return {"summary": runtime.store.data["summary"], "jobs": jobs}


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = runtime.store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return public_job(job)


@app.get("/v1/jobs/{job_id}/result")
def get_result(job_id: str):
    job = runtime.store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"Job is {job['status']}")
    path = Path(job["output_path"])
    if not path.exists():
        raise HTTPException(status_code=410, detail="Result file is missing")
    return json.loads(path.read_text(encoding="utf-8"))
