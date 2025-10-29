import uvicorn
from app import app


def main():
    """Run the FastAPI server"""
    print("Starting Waste Segregation API server...")
    print("API will be available at: http://localhost:8000")
    print("Documentation at: http://localhost:8000/docs")

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


if __name__ == "__main__":
    main()
