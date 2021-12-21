import os
import heapq
import pickle
import face_recognition as fr

def SimilarFaces(n,path):
    dir_list = os.listdir(path)

    conocidas = []
    names_in_order = []
    break_fg = False
    it = 0
    for file_path in dir_list:
        path_tmp = path + "/" + file_path
        img_list = os.listdir(path_tmp)

        for file_name in img_list:
            path_aux = path_tmp + "/" + file_name
            img = fr.load_image_file(path_aux)

            unknown_face_encodings = fr.face_encodings(img)
            for elem in unknown_face_encodings:
                if it == n:
                    break_fg = True
                    break
                names_in_order.append(path_aux)
                conocidas.append(elem)
                it = it + 1

            if break_fg:
                break
        if break_fg:
            break
    return conocidas,names_in_order