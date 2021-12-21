import os
import heapq
import pickle
import face_recognition as fr

base_path = os.getcwd()

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

def Seq_Serializer(n,a,b):
    path = base_path + '/DataProcessing/seq_files/'
    data = [a,b]
    filenames = ['faces','order']
    it = 0
    for d in data:
        file = open(path+filenames[it] + str(n) + '.dat','wb+')
        pickle.dump(d,file)
        file.close()
        it += 1

"""
Testing phase - Checks out
"""
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

def testing():
    collection_path = base_path + '/DataProcessing/Collection/lfw/'
    faces, order = SimilarFaces(100,collection_path)
    Seq_Serializer(100,faces,order)
    img_path = base_path + "/DataProcessing/Collection/set_1/"
    img = fr.load_image_file(img_path+'foto1.jpg')
    query = fr.face_encodings(img)[0]
    data = KNN_Seq(8,query,200)
    print(data)

"""
Creation of all files for the given N collection
"""

def files_creation():
    N = [100,200,400,800,1600,3200,6400,12800]
    collection_path = base_path + '/DataProcessing/Collection/lfw/'
    for n in N:
        faces, order = SimilarFaces(n, collection_path)
        Seq_Serializer(n,faces,order)
        print(n,'DONE')

files_creation()