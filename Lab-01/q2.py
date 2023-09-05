import cv2

vid_capture = cv2.VideoCapture('vid.mp4')
if(vid_capture.isOpened() == False):
    print('Error opening video file')
else:
    fps = vid_capture.get(5)
    print('Frames per second: ',fps)

    frame_count = vid_capture.get(7)
    print('Frame count: ',frame_count)

while(vid_capture.isOpened()):
    ret, frame = vid_capture.read()
    if ret == True:
        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        cv2.imshow('Frame',frame)
        key = cv2.waitKey(5000)
        if key == ord('q'):
            break
        else:
            break
vid_capture.release()
cv2.destroyAllWindows()