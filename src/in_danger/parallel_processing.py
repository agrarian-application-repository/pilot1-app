import cv2
import queue
import threading
import concurrent.futures

from src.in_danger.in_danger_v2_utils import perform_detection, perform_segmentation


def detection_process(detector, frame, detection_args):
    classes, boxes_centers, boxes_wh, boxes_corner1, boxes_corner2 = perform_detection(detector, frame, detection_args)
    return classes, boxes_centers, boxes_wh, boxes_corner1, boxes_corner2


def segmentation_process(segmenter, frame, segmentation_args):
    segment_danger_mask = perform_segmentation(segmenter, frame, segmentation_args)
    return segment_danger_mask


def georeferenced_data_handling_process(flight_data_file, frame_id):
    flight_frame_data = parse_drone_frame(flight_data_file, frame_id)
    return "result3"


def combine_results(result1, result2, result3):
    return f"Combined: {result1}, {result2}, {result3}"


def async_final_process(frame, combined_result):
    # For example, overlay text on the frame.
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, combined_result, (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


def writer_worker(writer_queue, video_writer):
    while True:
        item = writer_queue.get()
        if item is None:  # Sentinel value to exit.
            break
        frame, combined_result = item
        processed_frame = async_final_process(frame, combined_result)
        video_writer.write(processed_frame)
        writer_queue.task_done()


def main():
    input_video_path = 'input.mp4'
    output_video_path = 'output.mp4'

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    writer_queue = queue.Queue(maxsize=10)
    writer_thread = threading.Thread(target=writer_worker, args=(writer_queue, video_writer))
    writer_thread.start()

    # The executor creates a pool of persistent processes.
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Submit the three operations in parallel.
            future1 = executor.submit(detection_process, detector, frame, detection_args)
            future2 = executor.submit(segmentation_process, segmenter, frame, segmentation_args)
            future3 = executor.submit(georeferenced_data_handling_process, frame)

            # Wait for the three results.
            classes, boxes_centers, boxes_wh, boxes_corner1, boxes_corner2 = future1.result()
            segment_danger_mask = future2.result()
            combined_dem_mask_over_frame = future3.result()

            # create safety mask
            # check intersections

            combined = combine_results(result1, result2, result3)
            writer_queue.put((frame, combined))

    writer_queue.put(None)
    writer_thread.join()
    cap.release()
    video_writer.release()


if __name__ == '__main__':
    main()
