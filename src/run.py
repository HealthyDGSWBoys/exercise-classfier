from keras import models
from keras.optimizers import SGD
import numpy as np
import mediapipe as mp
import processer as util
import cv2 as cv
import tqdm

mp_pose = mp.solutions.pose

# pose_angle = PoseAngle()
pose = mp_pose.Pose(
            model_complexity=1,
            static_image_mode=True,
            min_detection_confidence=0.5
        )
processer = util.MultiProcesser(
    [
        util.AngleProcesser(),
        util.DistanceProcesser2(),
    ]
)
# processer = util.AngleProcesser()
model = models.load_model('./dist/temp.h5')
model.summary()
cap = cv.VideoCapture('./test/test.mp4')
visualizer = util.PoseClassificationVisualizer('up')


width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
frame = cap.get(cv.CAP_PROP_FPS)

fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('./test/output.mp4', fourcc, frame, (int(width), int(height)))

while True:
    ret, frame = cap.read()
    if ret:
        
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = pose.process(image)
        if result.pose_world_landmarks is not None:
            landmarks = []
            for landmark in result.pose_world_landmarks.landmark:
                landmarks.append([
                    landmark.x * 100,
                    landmark.y * 100,
                    landmark.z * 100
                ])
            res = model.predict(np.array([processer(np.array(landmarks))]))[0].tolist()

            max_val = res.index(max(res))
            print(res)
            if max_val == 0:
                name = "left"
            elif max_val == 1:
                name = "stand"
            else:
                name = "right"

            frame = visualizer(
                frame=frame,
                dataset=res
            )
            cv.putText(frame, name, (50, 300), cv.FONT_HERSHEY_PLAIN, 7, (0, 0, 255), 10, cv.LINE_AA)
            out.write(frame)
        key = cv.waitKey(1000 // 60)
        if key == 113:
            break

    else:
        break

cap.release()
out.release()