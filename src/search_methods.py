import os
import heapq
import face_recognition as fr
from rtree import index


def KNN_Seq(k, query, n, path):

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

    distances = fr.face_distance(conocidas, query)
    res = []

    for i in range(it):
        res.append((distances[i], names_in_order[i]))
    heapq.heapify(res)
    result = heapq.nsmallest(k, res)

    return result


def KNN_rtree(k, to_search):
    path = "/mnt/d/UTEC/2021-2/BD2/Proyecto/Proyecto3_BD2_2021-2/DataProcessing/"
    rtree_name = path + 'rtreeFile'

    query = to_search
    p = index.Property()
    p.dimension = 128  # D
    p.buffering_capacity = 10  # M

    rtreeidx = index.Rtree(rtree_name, properties=p)
    query_list = list(query)
    for query_i in query:
        query_list.append(query_i)

    return rtreeidx.nearest(coordinates=query_list, num_results=k, objects='raw')

def Seq_testing():
    # KNN Seq Testing
    collection_path = '/mnt/d/UTEC/2021-2/BD2/Proyecto/Proyecto3_BD2_2021-2/DataProcessing/Collection/lfw/'
    img_path = "/mnt/d/UTEC/2021-2/BD2/Proyecto/Proyecto3_BD2_2021-2/DataProcessing/Collection/set_1/"
    img = fr.load_image_file(img_path+'foto1.jpg')
    query = fr.face_encodings(img)[0]
    result = KNN_Seq(10, query, 500, collection_path)
    print(result)

def RTree_testing():
    # KNN RTree testing
    path = '/mnt/d/UTEC/2021-2/BD2/Proyecto/Proyecto3_BD2_2021-2/DataProcessing/Collection/set_1/'
    img = fr.load_image_file(path+'foto1.jpg')
    query = fr.face_encodings(img)[0]
    result = KNN_rtree(10,query)
    print(list(result))

# Seq_testing()
# RTree_testing()