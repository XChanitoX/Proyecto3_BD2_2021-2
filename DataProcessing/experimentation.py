import os
import face_recognition as fr
from rtree import index
import heapq
import pickle

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

def KNN_Seq(k, query,n):
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

from timeit import default_timer as timer
def Seq_testing(n):
    # KNN Seq Testing
    img_path = base_path + "/DataProcessing/Collection/set_1/"
    img = fr.load_image_file(img_path+'foto1.jpg')
    query = fr.face_encodings(img)[0]
    start = timer()
    KNN_Seq(8, query, n)
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
        # times.append(RTree_testing(n))
        times.append(Seq_testing(n))
        print(n,'DONE')
    print('Times:\n',times)

Experimentation()