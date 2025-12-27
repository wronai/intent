import os
import sys

# Add parent to path to import intentforge
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from intentforge import Intent, IntentForge, IntentType, TargetPlatform

# Initialize FastAPI
app = FastAPI(title="IntentForge API Server")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize IntentForge
# We use Ollama by default as per dev workflow
provider = os.getenv("LLM_PROVIDER", "ollama")
model = os.getenv("LLM_MODEL", "llama3.1:8b")

print(f"Initializing IntentForge with Provider: {provider}, Model: {model}")

forge = IntentForge(
    enable_auto_deploy=True,  # We want to execute the code
    sandbox_mode=True,  # Safely
    provider=provider,
    model=model,
)


class IntentRequest(BaseModel):
    description: str
    intent_type: str = "workflow"  # Default to workflow/generic
    context: dict[str, Any] = {}


class IntentResponse(BaseModel):
    success: bool
    message: str
    result: Any | None = None
    original_intent: str


@app.post("/api/intent", response_model=IntentResponse)
async def process_intent(request: IntentRequest):
    """
    Process a natural language intent
    """
    print(f"Received intent: {request.description}")

    # Map string type to Enum
    try:
        if request.intent_type == "workflow":
            i_type = IntentType.WORKFLOW
        elif request.intent_type == "api":
            i_type = IntentType.API_ENDPOINT
        else:
            i_type = IntentType.WORKFLOW

        # Create Intent
        intent = Intent(
            description=request.description,
            intent_type=i_type,
            target_platform=TargetPlatform.GENERIC_PYTHON,
            context=request.context,
        )

        # Process
        result = await forge.process_intent(intent)

        if not result.success:
            return IntentResponse(
                success=False,
                message=f"Generation failed: {result.validation_errors}",
                original_intent=request.description,
            )

        # Even if generation succeeded, execution might fail or be empty if auto_deploy is off
        # However, we enabled auto_deploy=True.
        # The result.execution_result should contain the return value of the executed code

        exec_res = result.execution_result

        if exec_res and not exec_res.success:
            print("❌ Execution Failed")
            print("FAILED CODE:")
            print("---")
            print(result.generated_code)
            print("---")
            print(f"Error: {exec_res.error}")

        return IntentResponse(
            success=True,
            message="Intent processed successfully",
            result=exec_res,
            original_intent=request.description,
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Serve static files for the frontend example
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8085)
