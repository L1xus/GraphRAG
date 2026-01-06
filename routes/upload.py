from fastapi import APIRouter, UploadFile, File, HTTPException
import tempfile
from services.neo4j_service import Neo4jService
from core.docs_load import load_pipeline
from core.models import UploadResponse

router = APIRouter()
neo4j_service = Neo4jService()

@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        result = load_pipeline(
            pdf_path=tmp_path,
            filename=file.filename,
            neo4j_service=neo4j_service
        )

        return UploadResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
