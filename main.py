import uvicorn
from src.app import server

if __name__ == "__main__":
    uvicorn.run(server, host="0.0.0.0", port=5002)