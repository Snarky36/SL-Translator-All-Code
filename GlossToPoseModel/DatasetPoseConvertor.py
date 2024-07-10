import os

import cv2
import mediapipe as mp
import numpy as np

class DatasetPoseConvertor():
    def __init__(self, folder_path, show_image=False):
        self.folder_path = folder_path
        self.show_image = show_image
        self.mp_pose = mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
    def detect_pose(self, image_path):
        sign_image = cv2.imread(image_path)
        if sign_image is None:
            raise Exception("Imaginea nu a fost gasita")
        converted_pose = self.pose.process(cv2.cvtColor(sign_image, cv2.COLOR_BGR2RGB))

        return sign_image, converted_pose


    def show_pose(self, image_path):
        image, converted_pose = self.detect_pose(image_path=image_path)

        if converted_pose.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, converted_pose.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

        cv2.imshow("Pose Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def convert_to_pose(self, image_path, pose_file_path):

        image, converted_pose = self.detect_pose(image_path=image_path)

        if converted_pose.pose_landmarks:
            # Save the pose landmarks to a .pose file
            with open(pose_file_path, 'w') as file:
                for landmark in converted_pose.pose_landmarks.landmark:
                    file.write(f"{landmark.x} {landmark.y} {landmark.z} {landmark.visibility}\n")
            print(f"Pose saved to {pose_file_path}")

    def convert_all_images_to_pose(self):
        pose_folder_path = self.folder_path + "/poses/"
        os.makedirs(pose_folder_path, exist_ok=True)

        for filename in os.listdir(self.folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(self.folder_path, filename)
                pose_file_name = os.path.splitext(filename)[0] + '.pose'
                pose_file_path = os.path.join(pose_folder_path, pose_file_name)
                self.convert_to_pose(image_path, pose_file_path)

    def show_pose_only(self, image_path):
        image, converted_pose = self.detect_pose(image_path=image_path)
        if converted_pose.pose_landmarks:
            blank_image = np.zeros_like(image)
            self.mp_drawing.draw_landmarks(
                blank_image, converted_pose.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))
            cv2.imshow("Pose Only", blank_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No pose landmarks detected.")



if __name__ == "__main__":
    pose_file_path = 'output.pose'
    image_path = '../images0033.png'

    pose_convertor = DatasetPoseConvertor("D:/Licenta/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/dev/01April_2010_Thursday_heute-6697", False)

    pose_convertor.convert_all_images_to_pose()


