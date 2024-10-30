import cv2
import mediapipe as mp
import pyautogui

x1 = y1 = x2 = y2 = 0
# Initialize webcam and MediaPipe Hands
webcam = cv2.VideoCapture(0)
my_hands = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

while True:
    # Capture frame from webcam
    _, image = webcam.read()
    image = cv2.flip(image,1)
    frame_height, frame_width, _ = image.shape

    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = my_hands.process(rgb_image)
    hands = output.multi_hand_landmarks

    # If hands are detected
    if hands:
        for hand in hands:
            # Draw landmarks and connections
            drawing_utils.draw_landmarks(
                image,
                hand,
                mp.solutions.hands.HAND_CONNECTIONS,
                drawing_utils.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                drawing_utils.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
            )
            # Extract landmarks for the detected hand
            landmarks = hand.landmark
            for id, landmark in enumerate(landmarks):
                # Convert normalized landmarks to pixel values
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                
                # Highlight thumb tip and index finger tip
                if id == 8:  # Index finger tip
                    cv2.circle(image, (x, y), 8, (255, 130, 180), 6)
                    x1 = x
                    y1 = y
                if id == 4:  # Thumb tip
                    cv2.circle(image, (x, y), 8, (255, 0, 180), 6)
                    x2 = x
                    y2 = y
        dist = ((x2-x1)**2) + ((y2-y1)**2) ** (0.5)//4
        cv2.line(image,(x1,y1),(x2,y2),(0,255,0),5)
        if dist > 50 :
            pyautogui.press("volumeup")
        else :
            pyautogui.press("volumedown")        
    # Display the output
    cv2.imshow("Hand Volume Control", image)

    # Exit on pressing ESC key
    key = cv2.waitKey(10)
    if key == 27:
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()



