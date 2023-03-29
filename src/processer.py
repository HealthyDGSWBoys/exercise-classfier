import math
import pandas as pd
import numpy as np
import os
import cv2
import tqdm
import csv
import sys
import mediapipe as mp
mp_pose = mp.solutions.pose


class AngleProcesser:
    angles = [["left_elbow", [11, 13, 15]],
        ["right_elbow", [12, 14, 16]],
        ["left_knee", [23, 25, 27]],
        ["right_knee", [24, 26, 28]],
        ["left_hip_y", [25, 23, 24]],
        ["right_hip_y", [26, 24, 23]],
        ["left_hip_x", [11, 23, 25]],
        ["right_hip_x", [12, 24, 26]],
        ["left_shoulder_x", [13, 11, 23]],
        ["right_shoulder_x", [14, 12, 24]],
        ["left_shoulder_y", [13, 11, 12]],
        ["right_shoulder_y", [14, 12, 11]],
        ["left_ankle", [25, 27, 31]],
        ["right_ankle", [26, 28, 32]]]
    
    
    def __init__(self) -> None:
        pass

    def __call__(self, landmark):
        return np.array([round(AngleProcesser.three_angle(
            landmark[item[1][0]], 
            landmark[item[1][1]], 
            landmark[item[1][2]]), 5) for item in AngleProcesser.angles], dtype=np.float32) / math.pi

    @staticmethod
    def three_angle(a, b, c):
        ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
        bc = [c[0] - b[0], c[1] - b[1], c[2] - b[2]]
        abVec = math.sqrt(ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2])
        bcVec = math.sqrt(bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2])
        abNorm = [ab[0] / abVec, ab[1] / abVec, ab[2] / abVec]
        bcNorm = [bc[0] / bcVec, bc[1] / bcVec, bc[2] / bcVec]
        res = abNorm[0] * bcNorm[0] + abNorm[1] * bcNorm[1] + abNorm[2] * bcNorm[2]
        return math.pi - math.acos(res)
    
    @staticmethod
    def get_angle_names():
        return [item[0] for item in AngleProcesser.angles]
    

class DistanceProcesser:
    _landmark_names = [
        'nose',
        'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear',
        'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_pinky_1', 'right_pinky_1',
        'left_index_1', 'right_index_1',
        'left_thumb_2', 'right_thumb_2',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
        'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index',
    ]
    def __init__(self):
        pass
    def __call__(self, landmarks):
        return np.array([
            self._get_distance(
                self._get_average_by_names(
                    landmarks, 'left_hip', 'right_hip'
                ),
                self._get_average_by_names(
                    landmarks, 'left_shoulder', 'right_shoulder'
                )
            ),
            self._get_distance_by_names(
                landmarks, 'left_shoulder', 'left_elbow'
            ),
            self._get_distance_by_names(
                landmarks, 'right_shoulder', 'right_elbow'
            ),
            self._get_distance_by_names(
                landmarks, 'left_elbow', 'left_wrist'
            ),
            self._get_distance_by_names(
                landmarks, 'right_elbow', 'right_wrist'
            ),
            self._get_distance_by_names(
                landmarks, 'left_hip', 'left_knee'
            ),
            self._get_distance_by_names(
                landmarks, 'right_hip', 'right_knee'
            ),
            self._get_distance_by_names(
                landmarks, 'left_knee', 'left_ankle'
            ),
            self._get_distance_by_names(
                landmarks, 'right_knee', 'right_ankle'
            ),
            self._get_distance_by_names(
                landmarks, 'left_shoulder', 'left_wrist'
            ),
            self._get_distance_by_names(
                landmarks, 'right_shoulder', 'right_wrist'
            ),
            self._get_distance_by_names(
                landmarks, 'left_hip', 'left_ankle'
            ),
            self._get_distance_by_names(
                landmarks, 'right_hip', 'right_ankle'
            ),
            self._get_distance_by_names(
                landmarks, 'left_hip', 'left_wrist'
            ),
            self._get_distance_by_names(
                landmarks, 'right_hip', 'right_wrist'
            ),
            self._get_distance_by_names(
                landmarks, 'left_shoulder', 'left_ankle'
            ),
            self._get_distance_by_names(
                landmarks, 'right_shoulder', 'right_ankle'
            ),
            self._get_distance_by_names(
                landmarks, 'left_hip', 'left_wrist'
            ),
            self._get_distance_by_names(
                landmarks, 'right_hip', 'right_wrist'
            ),
            self._get_distance_by_names(
                landmarks, 'left_elbow', 'right_elbow'
            ),
            self._get_distance_by_names(
                landmarks, 'left_knee', 'right_knee'
            ),
            self._get_distance_by_names(
                landmarks, 'left_wrist', 'right_wrist'
            ),
            self._get_distance_by_names(
                landmarks, 'left_ankle', 'right_ankle'
            ),

            # 신체 꺾임의 각도.

            # self._get_distance(
            #     self._get_average_by_names(landmarks, 'left_wrist', 'left_ankle'),
            #     landmarks[self._landmark_names.index('left_hip')]),
            # self._get_distance(
            #     self._get_average_by_names(landmarks, 'right_wrist', 'right_ankle'),
            #     landmarks[self._landmark_names.index('right_hip')]),
        ]).flatten() / 3000
    def _get_average_by_names(self, landmarks: np.ndarray, name_from: str, name_to: str) -> np.ndarray:
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return (lmk_from + lmk_to) / 2
    def _get_distance_by_names(self, landmarks: np.ndarray, name_from: str, name_to: str) -> np.ndarray:
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return self._get_distance(lmk_from, lmk_to)
    def _get_distance(self, lmk_from: np.ndarray, lmk_to: np.ndarray) -> np.ndarray:
        return lmk_to - lmk_from
    
class DistanceProcesser2:
    _landmark_names = [
        'nose',
        'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear',
        'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_pinky_1', 'right_pinky_1',
        'left_index_1', 'right_index_1',
        'left_thumb_2', 'right_thumb_2',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
        'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index',
    ]
    def __init__(self):
        pass
    def __call__(self, landmarks):
        return np.array([
            self._get_distance(
                self._get_average_by_names(
                    landmarks, 'left_hip', 'right_hip'
                ),
                self._get_average_by_names(
                    landmarks, 'left_shoulder', 'right_shoulder'
                )
            ),
            self._get_distance_by_names(
                landmarks, 'left_shoulder', 'left_elbow'
            ),
            self._get_distance_by_names(
                landmarks, 'right_shoulder', 'right_elbow'
            ),
            self._get_distance_by_names(
                landmarks, 'left_elbow', 'left_wrist'
            ),
            self._get_distance_by_names(
                landmarks, 'right_elbow', 'right_wrist'
            ),
            self._get_distance_by_names(
                landmarks, 'left_hip', 'left_knee'
            ),
            self._get_distance_by_names(
                landmarks, 'right_hip', 'right_knee'
            ),
            self._get_distance_by_names(
                landmarks, 'left_knee', 'left_ankle'
            ),
            self._get_distance_by_names(
                landmarks, 'right_knee', 'right_ankle'
            ),
            self._get_distance_by_names(
                landmarks, 'left_shoulder', 'left_wrist'
            ),
            self._get_distance_by_names(
                landmarks, 'right_shoulder', 'right_wrist'
            ),
            self._get_distance_by_names(
                landmarks, 'left_hip', 'left_ankle'
            ),
            self._get_distance_by_names(
                landmarks, 'right_hip', 'right_ankle'
            ),
            self._get_distance_by_names(
                landmarks, 'left_hip', 'left_wrist'
            ),
            self._get_distance_by_names(
                landmarks, 'right_hip', 'right_wrist'
            ),
            self._get_distance_by_names(
                landmarks, 'left_shoulder', 'left_ankle'
            ),
            self._get_distance_by_names(
                landmarks, 'right_shoulder', 'right_ankle'
            ),
            self._get_distance_by_names(
                landmarks, 'left_hip', 'left_wrist'
            ),
            self._get_distance_by_names(
                landmarks, 'right_hip', 'right_wrist'
            ),
            self._get_distance_by_names(
                landmarks, 'left_elbow', 'right_elbow'
            ),
            self._get_distance_by_names(
                landmarks, 'left_knee', 'right_knee'
            ),
            self._get_distance_by_names(
                landmarks, 'left_wrist', 'right_wrist'
            ),
            self._get_distance_by_names(
                landmarks, 'left_ankle', 'right_ankle'
            ),

            # 신체 꺾임의 각도.

            # self._get_distance(
            #     self._get_average_by_names(landmarks, 'left_wrist', 'left_ankle'),
            #     landmarks[self._landmark_names.index('left_hip')]),
            # self._get_distance(
            #     self._get_average_by_names(landmarks, 'right_wrist', 'right_ankle'),
            #     landmarks[self._landmark_names.index('right_hip')]),
        ]).flatten() / 3000
    def _get_average_by_names(self, landmarks: np.ndarray, name_from: str, name_to: str) -> np.ndarray:
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return (lmk_from + lmk_to) / 2
    def _get_distance_by_names(self, landmarks: np.ndarray, name_from: str, name_to: str) -> np.ndarray:
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return self._get_distance(lmk_from, lmk_to)
    def _get_distance(self, lmk_from: np.ndarray, lmk_to: np.ndarray) -> np.ndarray:
        a = lmk_to[0] - lmk_from[0]    
        b = lmk_to[1] - lmk_from[1] 
        c = lmk_to[2] - lmk_from[2]     
        return math.sqrt(a * a + b * b + c * c)


class MultiProcesser:
    def __init__(self, processers) -> None:
        self.processers = processers
        pass
    def __call__(self, landmarks):
        data = []
        for processer in self.processers:
            data.append(processer(landmarks))
        
        return np.concatenate((data[0], data[1]))

class CSVLoader:
    def __init__(self) -> None:
        pass

    def __call__(self, filename):
        data = pd.read_csv(filename)
        return np.array([item.reshape((33, 3)) for item in np.array([item for idx, item in data.iterrows()], dtype=np.float32)])


class BootstrapHelper(object):
    """분류를 위해 이미지를 부트스트랩하고 포즈 샘플을 필터링하는 데 도움을 줍니다."""

    def __init__(self, pose_class_name):
        self._pose_class_name = pose_class_name
        self.pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )

    def bootstrap(self, images):
        # CSV를 위한 폴더를 만든다.
        if not os.path.exists(self._csvs_out_folder):
            os.makedirs(self._csvs_out_folder)

        pose_class_name = self._pose_class_name
        print('Bootstrapping ', pose_class_name, file=sys.stderr)

        # 포즈 클래스의 경로
        images_in_folder = os.path.join(self._images_in_folder, pose_class_name)
        images_out_folder = os.path.join(self._images_out_folder, pose_class_name)
        csv_out_path = os.path.join(self._csvs_out_folder, pose_class_name + '.csv')

        if not os.path.exists(images_out_folder): # out 폴더가 존재하지 않을 경우
            os.makedirs(images_out_folder) # 자동 생성

        with open(csv_out_path, 'w') as csv_out_file:
            csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)

            # 모든 이미지를 Bootstrap 한다.
            for image in tqdm.tqdm(images):
                # 이미지를 불러온다.
                input_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # 새 포즈 트래커를 초기화하고 실행한다.
                result = self.pose.process(image=input_frame)
                pose_landmarks = result.pose_landmarks

                # 포즈가 감지된 경우 랜드마크를 저장한다.
                if pose_landmarks is not None:
                    # 랜드마크를 얻는다.
                    frame_height, frame_width = input_frame.shape[0], input_frame.shape[1]
                    pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width] for lmk in pose_landmarks.landmark], dtype=np.float32)
                    assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)
                    csv_out_writer.writerow(pose_landmarks.flatten().astype(np.str_).tolist())

    def analyze_outliers(self, outliers):
        """이상치를 찾기 위해 각 표본을 다른 모든 표본과 비교하여 분류합니다.

        샘플이 원래 클래스와 다르게 분류된 경우 -> 삭제하거나 유사한 샘플을 추가해야 합니다.
        """
        for outlier in outliers:
            image_path = os.path.join(self._images_out_folder, outlier.sample.class_name, outlier.sample.name)

            print('Outlier')
            print('  sample path =    ', image_path)
            print('  sample class =   ', outlier.sample.class_name)
            print('  detected class = ', outlier.detected_class)
            print('  all classes =    ', outlier.all_classes)

            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def remove_outliers(self, outliers):
        """이미지 폴더에서 이상치를 제거합니다."""
        for outlier in outliers:
            image_path = os.path.join(self._images_out_folder, outlier.sample.class_name, outlier.sample.name)
            os.remove(image_path)

    def print_images_in_statistics(self):
        """입력 이미지 폴더에서 통계를 출력합니다."""
        self._print_images_statistics(self._images_in_folder, self._pose_class_names)

    def print_images_out_statistics(self):
        """출력 이미지 폴더에서 통계를 출력합니다."""
        self._print_images_statistics(self._images_out_folder, self._pose_class_names)

    def _print_images_statistics(self, images_folder, pose_class_names):
        print('Number of images per pose class:')
        for pose_class_name in pose_class_names:
            n_images = len([n for n in os.listdir(os.path.join(images_folder, pose_class_name)) if not n.startswith('.')])
            print('  {}: {}'.format(pose_class_name, n_images))