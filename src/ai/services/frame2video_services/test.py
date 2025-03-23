import time
import random
from queue import Queue
from threading import Thread

# Simulate a real-time data source
def data_source(queue: Queue):
    while True:
        data = {"timestamp": time.time(), "value": random.randint(1, 100)}
        queue.put(data)
        time.sleep(1)  # Simulate real-time data arrival


# Data processing stage
def process_data(input_queue: Queue, output_queue: Queue):
    while True:
        if not input_queue.empty():
            raw_data = input_queue.get()
            processed_data = {
                "timestamp": raw_data["timestamp"],
                "value": raw_data["value"],
                "processed_value": raw_data["value"] * 2,  # Example transformation
            }
            output_queue.put(processed_data)


# Save the processed data (sink)
def save_data(output_queue: Queue):
    while True:
        if not output_queue.empty():
            data = output_queue.get()
            print(f"Saved: {data}")
            # Simulate saving to a database or file
            time.sleep(0.5)


# Main function to create the pipeline
if __name__ == "__main__":
    # Queues for data flow
    raw_data_queue = Queue()
    processed_data_queue = Queue()

    # Start threads for each stage
    source_thread = Thread(target=data_source, args=(raw_data_queue,))
    processing_thread = Thread(target=process_data, args=(raw_data_queue, processed_data_queue))
    sink_thread = Thread(target=save_data, args=(processed_data_queue,))

    # Start the threads
    source_thread.start()
    processing_thread.start()
    sink_thread.start()

    # Keep the main thread alive
    source_thread.join()
    processing_thread.join()
    sink_thread.join()
