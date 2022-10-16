import cv2 as cv


def get_face(fileName='lady.jpeg'):
    image_path = fileName
    cascPath = "haarcascade_frontalface_default.xml"

    faceCascade = cv.CascadeClassifier(cascPath)

    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)


    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv.CASCADE_SCALE_IMAGE
    )

    print(f"Found {len(faces)} faces!")
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face = image[y:y + h, x:x + w]
            return face
    else:
        return []



# out = get_face('115.jpg')
# if len(out) > 0:
#     cv.imshow("Faces found", out)
#     cv.waitKey(0)

