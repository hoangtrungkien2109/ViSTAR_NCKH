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

async def handle_text(websocket: WebSocket, stub):
    loop = asyncio.get_event_loop()
    try:
        while True:
            text = await websocket.receive_text()
            # Run blocking gRPC call in executor
            await loop.run_in_executor(
                None,
                stub.PushText,
                PushTextRequest(text=text, time_stamp="")
            )
    except Exception as e:
        print(f"Text handling error: {e}")
        raise

async def handle_images(websocket: WebSocket, stub):
    loop = asyncio.get_event_loop()
    try:
        # Get blocking gRPC stream
        pop_image_stream = await loop.run_in_executor(
            None,
            stub.PopImage,
            PopImageRequest(time_stamp="")
        )
        
        while True:
            try:
                # Get next item from blocking stream
                response = await loop.run_in_executor(None, next, pop_image_stream)
                if response.image:
                    base64_image = base64.b64encode(response.image).decode('utf-8')
                    await websocket.send_text(f"data:image/jpeg;base64,{base64_image}")
            except StopIteration:
                break  # Stream ended
            except grpc.RpcError as e:
                print(f"gRPC error: {e}")
                break
    except Exception as e:
        print(f"Image handling error: {e}")
        raise

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    stub = get_grpc_stub()

    text_task = asyncio.create_task(handle_text(websocket, stub))
    image_task = asyncio.create_task(handle_images(websocket, stub))

    try:
        done, pending = await asyncio.wait(
            [text_task, image_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel remaining tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"Error cancelling task: {e}")

        # Check for exceptions
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