import os
import face_recognition

for j in os.listdir('./MMAFEDB/train'):

    if j[0] == ".":
        continue

    for i in os.listdir('./MMAFEDB/train/' + j):

        if i[0] == ".":
            continue

        file = './MMAFEDB/train/'+ j + '/' + i
        image = face_recognition.load_image_file(file)

        if (len(face_recognition.face_locations(image)) == 0):
            print('delete face:' + j + i)
            os.remove(file)
            