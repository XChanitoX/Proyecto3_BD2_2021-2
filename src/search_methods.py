import os
import heapq
import face_recognition as fr
from rtree import index
import pickle

def KNN_Seq(k, query,n):
    base_path = "/mnt/d/UTEC/2021-2/BD2/Proyecto/Proyecto3_BD2_2021-2"
    path = base_path + '/DataProcessing/seq_files/'
    faces = pickle.load(open(path+'faces'+str(n)+'.dat','rb'))
    order = pickle.load(open(path+'order'+str(n)+'.dat','rb'))
    
    distances = fr.face_distance(faces, query)
    res = []

    for i in range(n):
        res.append((distances[i], order[i]))
    heapq.heapify(res)
    result = heapq.nsmallest(k, res)

    return result


def KNN_rtree(k, to_search):
    path = "/mnt/d/UTEC/2021-2/BD2/Proyecto/Proyecto3_BD2_2021-2/DataProcessing/rtree_files/"
    rtree_name = path + 'rtreeFile12800'

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