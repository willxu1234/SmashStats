from imageai.Detection import VideoObjectDetection  # pylint: disable=import-error
import os

execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(os.path.join(
    execution_path, "yolo-tiny.h5"))
detector.loadModel()

video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "traffic.mp4"), output_file_path=os.path.join(
    execution_path, "traffic_detected"), frames_per_second=20, log_progress=True)

print(video_path)
