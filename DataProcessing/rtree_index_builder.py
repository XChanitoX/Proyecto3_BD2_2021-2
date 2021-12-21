import os
import face_recognition as fr
from rtree import index

def process_collection(rtree_name, n):
    path = "/mnt/c/Users/Sebastian/C_2021-2/BD2/Projects/P3/DataProcessing/Collection/lfw"
    dir_list = os.listdir(path)

    p = index.Property()
    p.dimension = 128
    p.buffering_capacity = 10
    rtree_idx = index.Index(rtree_name, properties=p)

    c = 0
    break_fg = False
    images_list = []
    for file_path in dir_list:
        path_tmp = path + "/" + file_path

        img_list = os.listdir(path_tmp)

        for file_name in img_list:
            path_aux = path_tmp + "/" + file_name
            img = fr.load_image_file(path_aux)

#     Get face encodings for any faces in the uploaded image
            unknown_face_encodings = fr.face_encodings(img)

            for elem in unknown_face_encodings:

                if c == n:
                    break_fg = True
                    break

                coor_tmp = list(elem)
                for coor_i in elem:
                    coor_tmp.append(coor_i)
                tmp_obj = {"path": path_tmp, "name": file_name}
                rtree_idx.insert(c, coor_tmp, tmp_obj)
                images_list.append((c, path_aux))
                c = c + 1

            if break_fg:
                break

        if break_fg:
            break
    rtree_idx.close()

    print(str(c) + " images processed")
    return rtree_idx


rtree_name = 'rtreeFile'
process_collection(rtree_name, 600)
