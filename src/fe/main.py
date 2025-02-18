import asyncio
import base64
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
import grpc
from streaming_pb2 import PushTextRequest, PopImageRequest
from streaming_pb2_grpc import StreamingStub

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

def get_grpc_stub():
    channel = grpc.insecure_channel('localhost:50051')
    return StreamingStub(channel)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    stub = get_grpc_stub()
    
    pop_image_response = stub.PopImage(PopImageRequest(time_stamp=""))

    try:
        while True:
            # Receive text from the client
            text = await websocket.receive_text()
            
            # Push text to gRPC service
            push_image_response = stub.PushText(PushTextRequest(text=text, time_stamp=""))
            
            # Start receiving images
            for response in pop_image_response:
                if response.image:
                    # Convert bytes to base64
                    base64_image = base64.b64encode(response.image).decode('utf-8')
                    print("Show image in fe")
                    await websocket.send_text(f"data:image/jpeg;base64,{base64_image}")
    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)