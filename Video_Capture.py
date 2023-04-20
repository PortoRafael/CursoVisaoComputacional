import cv2

detector_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.Videopture(0)

while True:
  ok, frame = video_capture.read()
  
  imagem_cinza = cv2.cvtColor(frame,cv2.COLOR_BGRRGRAY)
  
  dtc = detector_face.detectMuiltiScale(imagem_cinza)
  
  for x,y,w,h in dtc:
    print(w, h)
    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
  cv2.imshow('Video',frame)
  
  if cv2.waitKey(1) & 0xFF = ord('q'):
    break

video_capture.release()
cv2.destroyAllWindows()
