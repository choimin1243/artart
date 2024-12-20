import cv2 as cv
import numpy as np
import mediapipe as mp
from utils import CvFpsCalc
import os
import sys
from datetime import datetime
import copy
import math

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument('--static_image_mode', action='store_true')
    parser.add_argument("--model_complexity", type=int, default=1)
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    args = parser.parse_args()
    return args

def main(img):
    args = get_args()
    cap_device = args.device

    # Get the screen resolution
    cv.namedWindow("Fullscreen", cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty("Fullscreen", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    screen_width, screen_height = cv.getWindowImageRect("Fullscreen")[2:]
    cv.destroyWindow("Fullscreen")

    cap_width, cap_height = screen_width, screen_height

    static_image_mode = args.static_image_mode
    model_complexity = args.model_complexity
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    background_image = img
    if background_image is None:
        print("Background image not found.")
        return

    background_image = cv.resize(background_image, (cap_width, cap_height))

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # 실행 파일이 있는 디렉토리 경로를 가져옵니다.
    if getattr(sys, 'frozen', False):
        application_path = os.path.dirname(sys.executable)
    else:
        application_path = os.path.dirname(os.path.abspath(__file__))

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f'output_{current_time}.mp4'
    output_path = os.path.join(application_path, output_filename)

    # 비디오 저장을 위한 설정
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    fps = 20.0
    out = cv.VideoWriter(output_path, fourcc, fps, (cap_width, cap_height))

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Create a named window and set it to fullscreen
    cv.namedWindow('Pose with Background', cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty('Pose with Background', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    try:
        while cap.isOpened():
            display_fps = cvFpsCalc.get()

            ret, image = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting...")
                break

            image = cv.flip(image, 1)
            debug_image = copy.deepcopy(background_image)

            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks is not None:
                debug_image = draw_pose(
                    debug_image,
                    results.pose_landmarks,
                )

            debug_image = cv.resize(debug_image, (cap_width, cap_height))
            out.write(debug_image)

            cv.putText(debug_image, "FPS:" + str(display_fps), (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
            cv.imshow('Pose with Background', debug_image)

            key = cv.waitKey(1)
            if key == 27:  # ESC
                print("ESC pressed. Saving video...")
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        print("Cleaning up resources...")
        cap.release()
        out.release()
        cv.destroyAllWindows()
        print(f"Video saved successfully at {output_path}")

    return output_path

def draw_pose(image, landmarks, visibility_th=0.5):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []
    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_point.append([index, landmark.visibility, (landmark_x, landmark_y), landmark_z])

    right_leg = landmark_point[23]
    left_leg = landmark_point[24]
    leg_x = int((right_leg[2][0] + left_leg[2][0]) / 2)
    leg_y = int((right_leg[2][1] + left_leg[2][1]) / 2)

    landmark_point[23][2] = (leg_x, leg_y)
    landmark_point[24][2] = (leg_x, leg_y)

    sorted_landmark_point = sorted(landmark_point, reverse=True, key=lambda x: x[3])

    (face_x, face_y), face_radius = min_enclosing_face_circle(landmark_point)

    face_x = int(face_x)
    face_y = int(face_y)
    face_radius = int(face_radius * 1.5)

    stick_radius01 = int(face_radius * (4 / 5))
    stick_radius02 = int(stick_radius01 * (3 / 4))
    stick_radius03 = int(stick_radius02 * (3 / 4))

    draw_list = [11, 12, 23, 24]

    for landmark_info in sorted_landmark_point:
        index = landmark_info[0]

        if index in draw_list:
            point01 = [p for p in landmark_point if p[0] == index][0]
            point02 = [p for p in landmark_point if p[0] == (index + 2)][0]
            point03 = [p for p in landmark_point if p[0] == (index + 4)][0]

            if point01[1] > visibility_th and point02[1] > visibility_th:
                image = draw_stick(
                    image,
                    point01[2],
                    stick_radius01,
                    point02[2],
                    stick_radius02,
                )
            if point02[1] > visibility_th and point03[1] > visibility_th:
                image = draw_stick(
                    image,
                    point02[2],
                    stick_radius02,
                    point03[2],
                    stick_radius03,
                )

    cv.circle(image, (face_x, face_y), face_radius, (0, 0, 0), -1)

    return image

def draw_stick(image, point01, point01_radius, point02, point02_radius):
    color = (0, 0, 0)
    cv.circle(image, point01, point01_radius, color, -1)
    cv.circle(image, point02, point02_radius, color, -1)

    draw_list = []
    for index in range(2):
        rad = math.atan2(point02[1] - point01[1], point02[0] - point01[0])

        rad = rad + (math.pi / 2) + (math.pi * index)
        point_x = int(point01_radius * math.cos(rad)) + point01[0]
        point_y = int(point01_radius * math.sin(rad)) + point01[1]

        draw_list.append([point_x, point_y])

        point_x = int(point02_radius * math.cos(rad)) + point02[0]
        point_y = int(point02_radius * math.sin(rad)) + point02[1]

        draw_list.append([point_x, point_y])

    points = np.array((draw_list[0], draw_list[1], draw_list[3], draw_list[2]))
    cv.fillConvexPoly(image, points=points, color=color)

    return image

def min_enclosing_face_circle(landmark_point):
    landmark_array = np.empty((0, 2), int)

    index_list = [1, 4, 7, 8, 9, 10]
    for index in index_list:
        np_landmark_point = [np.array((landmark_point[index][2][0], landmark_point[index][2][1]))]
        landmark_array = np.append(landmark_array, np_landmark_point, axis=0)

    center, radius = cv.minEnclosingCircle(points=landmark_array)

    return center, radius

if __name__ == '__main__':
    main()