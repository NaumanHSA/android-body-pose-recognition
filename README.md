# Introduction
This is an android application for body pose recognition using MediaPipe AAR in Android Studio with Gradle. The steps to build and use MediaPipe AAR is documented in MediaPipe's documentation.

The mediapipe body pose landmarks detection app has been buid using AAR and has been integrated in Android Studio. The landmarks have been passed through a classifier to recognise a specific gesture.

# Dataset
Mediapipe pose detection generated 33 normalized landmarks points in 3 dimensions (x, y, z) with an extra parameters for visibility. A CSV dataset with 4520 records and 132 features ;num_landmarks * landmark_parameters (33 * 4) flatten in as below:

            x1, y1, z1, v1, .... x33, y33, z33, v33

The 133th column in the dataset contains the class. Total number of classes in the dataset are 9. The dataset CSV can be downloaded from the link given in the last section.

# Training Classifier
A deep learning classifier has been trained on the dataset for a 100 epochs bearing training accuracy of 99.45% while validation accuracy of 98.04%. The point of choosing deep learning classifier instead of traditional classifier was to convert the classifier into tflite model and integrate in Android Studio. The converted tflite model gave validation accuracy of 98% which is quite the same as the original model.

The jupyter notebook for training the classifier can be found in my Google Drive link given in the last section..

# How to run the Application
1. Clone the repository. Then download the my_pose_tracking.aar file from my Google Drive link given in the last section. Paste it in the app/libs folder in the repository.

2. Open the Android Studio and import project. Wait for the gradle build and run the application while connecting your phone to the PC considering USB debugging is on inside your phone.

# References
1. [Project Files Google Drive](https://drive.google.com/drive/folders/1Vk_WdC_G0nZAXPB7XnLjTQfsYt4hsoNI)

2. [MediaPipe Android Archive](https://google.github.io/mediapipe/getting_started/android_archive_library.html)

3. [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html)
