import os
import face_recognition as fr
from rtree import index
import heapq

base_path = os.getcwd()

"""
Utility for Experimentation and testing
"""
def KNN_rtree_exp(n,k, to_search):
    path = base_path + "/DataProcessing/rtree_files/"
    rtree_name = path + 'rtreeFile' + str(n)

    query = to_search
    p = index.Property()
    p.dimension = 128  # D
    p.buffering_capacity = 10  # M

    rtreeidx = index.Rtree(rtree_name, properties=p)
    query_list = list(query)
    for query_i in query:
        query_list.append(query_i)

    return rtreeidx.nearest(coordinates=query_list, num_results=k, objects='raw')

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

from timeit import default_timer as timer
def Seq_testing(n):
    # KNN Seq Testing
    collection_path = base_path + '/DataProcessing/Collection/lfw/'
    img_path = base_path + "/DataProcessing/Collection/set_1/"
    img = fr.load_image_file(img_path+'foto1.jpg')
    query = fr.face_encodings(img)[0]
    start = timer()
    KNN_Seq(8, query, n, collection_path)
    end = timer()
    return (end - start)

def RTree_testing(n):
    # KNN RTree testing
    path = base_path + '/DataProcessing/Collection/set_1/'
    img = fr.load_image_file(path+'foto3.jpg')
    query = fr.face_encodings(img)[0]
    start = timer()
    KNN_rtree_exp(n,8,query)
    end = timer()
    return (end - start)


"""
Experimentation
"""
def Experimentation():
    N = [100,200,400,800,1600,3200,6400,12800]
    times = []

    for n in N:
        times.append(RTree_testing(n))
        # times.append(Seq_testing(n))
    print('Times:\n',times)

Experimentation()