"""ASTER FastAPI application entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from backend.app.query_manager.query_manager import execute_query
from backend.app.query_manager.dataset_manager import process_upload


app = FastAPI(title="ASTER")


class QueryRequest(BaseModel):
	query: str = Field(..., min_length=1, description="Natural-language analytical query")

@app.post("/query")
def post_query(body: QueryRequest) -> dict[str, Any]:
	"""Run the canonical pipeline for a natural-language query."""

	return execute_query(body.query)

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


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
	"""Serve the temporary demo page."""

	return HTMLResponse(_frontend_html())

