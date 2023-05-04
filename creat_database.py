import cv2

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('InShot_20220403_121921259.mp4')
count = 0
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        frame = cv2.resize(frame, (800, 600)) #600:400 fr 01
        frame = frame[100:600, :]

        if count < 450:
            cv2.imwrite('train03c/test{}.jpg'.format(count), frame)
            print('test{}.jpg'.format(count))
            count=count+1
    cv2.imshow("live", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

