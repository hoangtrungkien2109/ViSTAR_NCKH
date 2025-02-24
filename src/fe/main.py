import asyncio
import base64
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
import grpc.aio
from streaming_pb2 import PushTextRequest, PopImageRequest
from streaming_pb2_grpc import StreamingStub

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

async def get_grpc_stub():
    channel = grpc.aio.insecure_channel('localhost:50051')
    return StreamingStub(channel)

async def handle_text(websocket: WebSocket, stub: StreamingStub):
    try:
        while True:
            text = await websocket.receive_text()
            await stub.PushText(PushTextRequest(text=text, time_stamp=""))
    except Exception as e:
        print(f"Text handling error: {e}")
        raise

async def handle_images(websocket: WebSocket, stub: StreamingStub):
    try:
        async for response in stub.PopImage(PopImageRequest(time_stamp="")):
            if response.image:
                base64_image = base64.b64encode(response.image).decode('utf-8')
                await websocket.send_text(f"data:image/jpeg;base64,{base64_image}")
    except Exception as e:
        print(f"Image handling error: {e}")
        raise

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    stub = await get_grpc_stub()

    text_task = asyncio.create_task(handle_text(websocket, stub))
    image_task = asyncio.create_task(handle_images(websocket, stub))

    try:
        done, pending = await asyncio.wait(
            [text_task, image_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"Error cancelling task: {e}")
        for task in done:
            if task.exception():
                print(f"Task failed: {task.exception()}")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
