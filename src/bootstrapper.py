import argparse
import os
import cv2
from pathlib import Path
import mediapipe as mp
import csv
import tqdm
import numpy as np
import glob

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

parser = argparse.ArgumentParser()
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
parser.add_argument('dirname', type=dir_path, help="부트스트랩할 비디오가 들어있는 디렉토리를 지정받습니다")
parser.add_argument('--out', '-o', help='부트스트래핑한 CSV파일을 내보낼 디렉토리를 입력받습니다')
parser.add_argument('--complexity', '-c', help='모델 복잡도를 설정합니다', default=1)

args = parser.parse_args()

workspace_dir = args.dirname

file_list = os.listdir(workspace_dir)

# shutil.rmtree(os.path.join(workspace_dir, "dataset"))
os.makedirs(os.path.join(workspace_dir, "dataset"))

for video_file in file_list:
    IMAGE_FILES = []
    BG_COLOR = (192, 192, 192) # gray
    if video_file[0] != '.':
        video_file = os.path.join(workspace_dir, video_file)
        if os.path.isdir(video_file):
            # for item in os.listdir(video_file):
            #     IMAGE_FILES.append(cv2.imread(item))
            continue
        elif os.path.isfile(video_file):
            if os.path.splitext(video_file)[1] != ".mp4": continue
            vcap = cv2.VideoCapture(video_file)
            while True:
                success, image = vcap.read()
                if success:
                    IMAGE_FILES.append(image)
                else:
                    break
            vcap.release()
            video_file = Path(video_file).stem
        else:
            print("올바르지 않은 파일 명입니다")
            continue
    else: continue

    file = open(os.path.join(workspace_dir, "dataset", video_file) + ".csv", 'w')
    writer = csv.writer(file)
    
    name = []

    for landmark in mp_pose.PoseLandmark:
        landmark_name = str(landmark).split('.')[1]
        name.append(landmark_name + "_x")
        name.append(landmark_name + "_y")
        name.append(landmark_name + "_z")
    writer.writerow(name)
    with tqdm.tqdm(total=len(IMAGE_FILES), position=0, leave=False) as pbar:
        with mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5
            ) as pose:
                for idx, image in enumerate(IMAGE_FILES):
                    result = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    if not result.pose_world_landmarks:
                        continue
                    pose_landmarks = result.pose_world_landmarks
                    frame_height, frame_width = image.shape[0], image.shape[1]

                    pose_landmarks = np.array([[lmk.x, lmk.y, lmk.z] for lmk in pose_landmarks.landmark], dtype=np.float32) * 100
                    assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)
                    writer.writerow(pose_landmarks.flatten().astype(np.str_).tolist())

                    # annotated_image = image.copy()
                    # mp_drawing.draw_landmarks(
                    # annotated_image,
                    # result.pose_landmarks,
                    # mp_pose.POSE_CONNECTIONS,
                    # landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    # cv2.imshow("A",annotated_image)
                    # cv2.waitKey(1)
                    
                    pbar.update()

    file.close()
        
path = os.path.join(workspace_dir, "dataset/")
merge_path = os.path.join(workspace_dir, "dataset.csv")

file_list = glob.glob(path + '*')

with open(merge_path, 'w') as f:
    is_first = True
    for file in file_list:
        with open(file ,'r') as f2:
            if is_first:
                is_first = False
            else:
                f2.readline()
            while True:
                line = f2.readline()
                if not line:
                    break
                f.write(line)
        file_name = file.split('\\')[-1]

print('>>> All file merge complete...')