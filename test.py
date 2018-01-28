import cv2
import numpy as np

cap = cv2.VideoCapture("overpass.mov")
i = 0
ret, frame = cap.read()
for i in range(12):
    print(cap.get(i))


# print (cap.get(3))
# print (cap.get(4))
# print (cap.get(7))
# while ret:
#
#     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite('img/img' + str(i) + '.jpg', gray)
#     cv2.imshow('frame', frame)
#     i += 1
#     print np.array(frame, np.float32).shape
#     print cap.get(3)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     ret, frame = cap.read()
#
# cap.release()
# cv2.destroyAllWindows()
