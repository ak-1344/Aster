"""ASTER FastAPI application entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from backend.app.query_manager.query_manager import execute_query
from backend.app.query_manager.dataset_manager import process_upload
from backend.app.api.dashboard import router as dashboard_router


app = FastAPI(title="ASTER")
app.include_router(dashboard_router)


class QueryRequest(BaseModel):
	query: str = Field(..., min_length=1, description="Natural-language analytical query")
	dataset_id: str | None = Field(None, description="Optional dataset ID from a previous upload")
	clarification_response: str | None = Field(
		None,
		description="Optional user reply when a prior response returned clarification_needed",
	)

@app.post("/query")
def post_query(body: QueryRequest) -> dict[str, Any]:
	"""Run the canonical pipeline for a natural-language query."""

	dataset_path = None
	if body.dataset_id:
		dataset_path = Path("backend/data/raw") / f"{body.dataset_id}.csv"
		if not dataset_path.exists():
			raise HTTPException(status_code=400, detail=f"Dataset {body.dataset_id} not found")

	return execute_query(
		body.query,
		dataset_path=dataset_path,
		clarification_response=body.clarification_response,
	)

@app.post("/upload")
def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
	"""Upload a CSV dataset and run initial feature engineering."""
	
	try:
		return process_upload(file.file, file.filename or "upload.csv")
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except RuntimeError as e:
		raise HTTPException(status_code=422, detail=str(e))


def _frontend_html() -> str:
	"""Load the temporary demo page from the frontend folder."""
	return (Path(__file__).resolve().parents[3] / "frontend" / "index.html").read_text(encoding="utf-8")

def _frontend_dashboard_html() -> str:
	"""Load the dashboard page from the frontend folder."""
	return (Path(__file__).resolve().parents[3] / "frontend" / "dashboard.html").read_text(encoding="utf-8")

@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
	"""Serve the temporary demo page."""
	return HTMLResponse(_frontend_html())

@app.get("/dashboard-ui", response_class=HTMLResponse)
def dashboard_ui() -> HTMLResponse:
	"""Serve the dashboard page."""
	return HTMLResponse(_frontend_dashboard_html())

