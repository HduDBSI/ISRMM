import math
import os
import csv
import numpy as np
import keras
from keras import Input, Model
from keras.optimizer_v1 import SGD
from keras.optimizers import adam_v2
from keras.layers import Multiply, Subtract, Concatenate, Dense, Dropout, multiply, Flatten
import numpy as np
from keras.models import load_model

import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

from data3_mtask_shuffle_api1.node2vecembedding_shuffle_api import havaembedding

import heapq

parentdic = os.path.dirname(os.path.dirname(os.getcwd()))

def str_list_to_float(str_list):
    return [float(item) for item in str_list]

def AP(ranked_list, ground_truth):

    p_at_k = np.zeros(len(ranked_list))
    c = 0
    for i in range(1, len(ranked_list) + 1):
        rel = 0
        if ranked_list[i - 1] in ground_truth:
            rel = 1
            c += 1
        p_at_k[i - 1] = rel * c / i
    if c == 0:
        return 0.0
    else:

        return np.sum(p_at_k) / min(len(ground_truth),len(ranked_list))

def IDCG(n):
    idcg = 0
    for i in range(n):
        idcg += 1 / math.log(i+2, 2)
    return idcg

def NGCD(ranked_list, ground_truth):
    dcg = 0
    idcg = IDCG(min(len(ranked_list),len(ground_truth)))
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id not in ground_truth:
            continue
        rank = i + 1
        dcg += 1 / math.log(rank + 1, 2)
    return dcg / idcg


def findmashupapiemb(data):#获取mashup和api的结构embedding，使用node2vec获得

    mashupembdict, apiembdict=havaembedding(data)

    return mashupembdict,apiembdict


def findapides():

    apidesdict=dict()
    with open(parentdic + "/data/data_api.csv", 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for data in reader:
            id = data[0]
            api_name = data[1].strip()
            api_des = data[3]
            cat=data[4].strip()
            apitags = data[6]
            provider = data[12]
            listprovider = provider.split(",")
            apidesdict.__setitem__(api_name,api_des)

    return apidesdict

def findapitag():

    apitagdict=dict()
    with open(parentdic + "/data/data_api.csv", 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for data in reader:
            id = data[0]
            api_name = data[1].strip()
            api_des = data[3]
            cat=data[4].strip()
            apitags = data[6]

            apitagdict.__setitem__(api_name,",".join(apitags.split("###")))

    return apitagdict


def findmashupdes():

    apidesdict=findapides()

    mashupdesdict=dict()
    newapidesdict=dict()

    apiset=set()
    mashupset=set()

    mashupapidict=dict()

    with open(parentdic + "/data/data3_mtask_shuffle/mashup_train.csv", 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for data in reader:
            id = data[0]
            mashup_name = data[1].strip()
            mashupset.add(mashup_name)
            mashup_link = data[2]
            apis = data[6]
            mashup_des = data[3]
            tags = data[4]
            listapis = apis.split("###")

            mashupdesdict.__setitem__(mashup_name.strip(),mashup_des)

            for i in listapis:
                apiname = i.strip()
                i = i.strip()

                if (i in apidesdict.keys()):
                    apiset.add(apiname)
                    newapidesdict.__setitem__(apiname,apidesdict[i])
                    if(mashupapidict.__contains__(mashup_name)):
                        mashupapidict[mashup_name]=mashupapidict[mashup_name]+"###"+apiname
                    else:
                        mashupapidict.__setitem__(mashup_name,apiname)
                elif (i + " API" in apidesdict.keys()):
                    apiset.add(apiname)
                    newapidesdict.__setitem__(apiname,apidesdict[i+" API"])
                    if (mashupapidict.__contains__(mashup_name)):
                        mashupapidict[mashup_name] = mashupapidict[mashup_name] + "###" + apiname
                    else:
                        mashupapidict.__setitem__(mashup_name, apiname)
                else:
                    pass

    return mashupdesdict,newapidesdict,mashupapidict,apiset,mashupset


def finddesemb():

    trainmashupdesembdict=dict()
    apidesembdict=dict()

    with open(parentdic + "/data/data3_mtask_shuffle/electra/trainmashupdesemb.csv", "r") as f:
        reader = csv.reader(f)
        for data in reader:
            train_mashup_name= data[0]
            train_mashup_des_emb=[float(i) for i in (data[1].split(" "))]
            trainmashupdesembdict.__setitem__(train_mashup_name,train_mashup_des_emb)

    with open(parentdic + "/data/data3_mtask_shuffle/electra/apidesemb.csv", "r") as f:
        reader = csv.reader(f)
        for data in reader:
            api_name=data[0]
            api_des_emb=[float(i) for i in (data[1].split(" "))]
            apidesembdict.__setitem__(api_name,api_des_emb)

    return trainmashupdesembdict,apidesembdict

def findtraintagemb():


    trainmashuptagembdict=dict()
    apitagembdict=dict()

    with open(parentdic + "/data/data3_mtask_shuffle/electra/trainmashuptagemb.csv", "r") as f:
        reader = csv.reader(f)
        for data in reader:
            train_mashup_name= data[0]
            train_mashup_des_emb=[float(i) for i in (data[1].split(" "))]
            trainmashuptagembdict.__setitem__(train_mashup_name,train_mashup_des_emb)

    with open(parentdic + "/data/data3_mtask_shuffle/electra/apitagemb.csv", "r") as f:
        reader = csv.reader(f)
        for data in reader:
            api_name=data[0]
            api_des_emb=[float(i) for i in (data[1].split(" "))]
            apitagembdict.__setitem__(api_name,api_des_emb)

    return trainmashuptagembdict,apitagembdict

def findtesttagemb():


    testmashuptagembdict=dict()

    with open(parentdic + "/data/data3_mtask_shuffle/electra/testmashuptagemb.csv", "r") as f:
        reader = csv.reader(f)
        for data in reader:
            train_mashup_name= data[0]
            train_mashup_des_emb=[float(i) for i in (data[1].split(" "))]
            testmashuptagembdict.__setitem__(train_mashup_name,train_mashup_des_emb)

    return testmashuptagembdict



def getmashuptag2id():

    mashuptagdict=dict()
    tagset=set()
    with open(parentdic + "/data/data3_mtask_shuffle/mashuptag2id.csv", "r") as f:
        reader = csv.reader(f)
        for data in reader:
            tag = data[0]
            id=data[1]
            tagset.add(id)
            mashuptagdict.__setitem__(tag,id)
    #print(len(tagset))

    mashuptag2iddict=dict()

    with open(parentdic + "/data/data3_mtask_shuffle/mashup_train.csv", "r") as f:
        reader = csv.reader(f)
        for data in reader:
            id = data[0]
            mashup_name = data[1].strip()
            mashup_link = data[2]
            apis = data[6]
            mashup_des = data[3]
            tags = data[4]
            listtags=tags.split("###")
            tagslist=[]
            for i in listtags:
                tagname=i.strip()
                tagid=mashuptagdict[tagname]
                tagslist.append(int(tagid))
            #print(tagslist)
            f = [0] * len(tagset)
            for i in tagslist:
                f[i] = 1

            mashuptag2iddict.__setitem__(mashup_name,f)

    return mashuptag2iddict

def getapi2id():

    apidict = dict()
    with open(parentdic + "/data/data3_mtask_shuffle/api2id.csv", "r") as f:
        reader = csv.reader(f)
        for data in reader:
            apiname = data[0]
            id = data[1]
            apidict.__setitem__(apiname, id)
    return apidict


def getid2api():

    id2apidict = dict()
    with open(parentdic + "/data/data3_mtask_shuffle/api2id.csv", "r") as f:
        reader = csv.reader(f)
        for data in reader:
            apiname = data[0]
            id = data[1]
            id2apidict.__setitem__(int(id), apiname)
    return id2apidict


def gettag2id():

    tag2iddict=dict()
    with open(parentdic + "/data/data3_mtask_shuffle/apitag2id.csv", "r") as f:
        reader = csv.reader(f)
        for data in reader:
            apiname = data[0]
            id=data[1]
            tag2iddict.__setitem__(apiname,id)
    return tag2iddict

def gettrainapitagdict(): #得到apitag的多类分布0-1

    tag2iddict=gettag2id()
    tagset=set()
    for k,v in tag2iddict.items():
        tagset.add(k)

    trainapitagdict=dict()
    with open(parentdic + "/data/data3_mtask_shuffle/trainapitag.csv", "r") as f:
        reader = csv.reader(f)
        for data in reader:
            apiname = data[0]
            tags=data[1]
            listtags=tags.split("###")

            apitagslist=[]
            for i in listtags:
                tagid=tag2iddict[i.strip()]
                apitagslist.append(int(tagid))

            f = [0] * len(tagset)
            for i in apitagslist:
                f[i] = 1

            trainapitagdict.__setitem__(apiname,f)

    return trainapitagdict

def buildmodel():

    trainapitagdict=gettrainapitagdict()

    mashuptagdict=getmashuptag2id() #一共有302个分类

    apidict=getapi2id()


    mashupdesdict, newapidesdict, mashupapidict, apiset, mashupset = findmashupdes()

    mashupembdict, apiembdict=findmashupapiemb() #是结构的emb,长度为16

    trainmashupdesembdict, apidesembdict=finddesemb() #是文本的emb,长度为768

    mashupstru = []
    mashupdesemd=[]

    taglabel=[]
    apilabel=[]

    apitaglabel = []

    gruapides = []
    gruapistru = []

    des_model = load_model(parentdic + "/data/data3_mtask_shuffle/model_api1_gru/des_gru.h5")
    stru_model = load_model(parentdic + "/data/data3_mtask_shuffle/model_api1_gru/stru_model.h5")

    des_lstm_model = load_model(parentdic + "/data/data3_mtask_shuffle/model_api1_gru/gru.h5")
    stru_lstm_model = load_model(parentdic + "/data/data3_mtask_shuffle/model_api1_gru/stru_gru.h5")

    for k,v in mashupapidict.items():
        mashup_name=k
        listapis=v.split("###")

        for api in range(0,len(listapis)):

            mashupemb = mashupembdict[mashup_name]  # 结构的embedding

            mashupdesemd.append(trainmashupdesembdict[mashup_name])  # des的emb长度为768

            gruapides.append([apidesembdict[listapis[api].strip()]])

            mashupstru.append(mashupemb)
            gruapistru.append([apiembdict[listapis[api].strip()]])

            taglabel.append(mashuptagdict[mashup_name])
            apiidlist = []
            for i in [i for i in listapis if i not in listapis[api]]:
                apiid = apidict[i]
                # print(apiid)
                apiidlist.append(int(apiid))
            f = [0] * len(apiset)
            for i in apiidlist:
                f[i] = 1

            apilabel.append(f)
            apitaglabel.append(trainapitagdict[listapis[api].strip()])

    stru_predict = stru_lstm_model.predict(np.array(gruapistru))
    apilstmstruemd = []
    for i in stru_predict:
        apilstmstruemd.append(i.tolist())

    des_predict = des_lstm_model.predict(np.array(gruapides))
    apilstmdesemd = []
    for i in des_predict:
        apilstmdesemd.append(i.tolist())

    apilstmdesemd = np.array(apilstmdesemd)
    apilstmstruemd = np.array(apilstmstruemd)

    mashupstru = np.array(mashupstru)

    mashupdesemd=np.array(mashupdesemd)


    taglabel=np.array(taglabel)
    apilabel=np.array(apilabel)
    apitaglabel=np.array(apitaglabel)

    #neighborapilabel=np.array(neighborapilabel)

    des_intermediate_layer_model = Model(inputs=des_model.input, outputs=des_model.get_layer("layer_dense_2").output)
    x = des_intermediate_layer_model.predict([mashupdesemd,apilstmdesemd])

    stru_intermediate_layer_model = Model(inputs=stru_model.input, outputs=stru_model.get_layer("layer_dense_2").output)
    y = stru_intermediate_layer_model.predict([mashupstru,apilstmstruemd])

    #mashupdes_input = Input(shape=(768,), dtype='float32', name='mashupdes_input')  # x

    des_input = Input(shape=(50,), dtype='float32', name='des_input') #x
    stru_input = Input(shape=(50,), dtype='float32', name='stru_input')#y


    mf_vector = Concatenate(name='fea_concatenate')([des_input,stru_input])


    mf_vector = Dropout(0.2)(mf_vector)

    mf_vector = Dense(256, activation='relu', name='layer_dense_1')(mf_vector)

    mf_vector = Dropout(0.2)(mf_vector)

    mf_vector = Dense(128, activation='relu', name='layer_dense_2')(mf_vector)

    mf_vector = Dropout(0.2)(mf_vector)

    mf_vector = Dense(64, activation='relu', name='layer_dense_3')(mf_vector)

    mf_vector = Dropout(0.2)(mf_vector)

    mf_vector = Dense(32, activation='relu', name='layer_dense_4')(mf_vector)

    #mf_vector = Dropout(0.2)(mf_vector)

    # des_mf_vector = Dropout(0.5)(des_mf_vector)
    prediction1 = Dense(1062, activation='sigmoid', name="final_prediction1")(mf_vector)

    prediction2 = Dense(311, activation='sigmoid', name="final_prediction2")(mf_vector)

    prediction3 = Dense(311, activation='sigmoid', name="final_prediction3")(mf_vector)

    des_model = Model(inputs=[des_input, stru_input], outputs=[prediction1, prediction2,prediction3])
    #des_model = Model(inputs=[des_input, stru_input], outputs=prediction1)

    des_model.compile(optimizer=adam_v2.Adam(learning_rate=0.0001),
                      loss="binary_crossentropy",
                      metrics=['accuracy'],
                      )

    des_model.summary()
    des_model.fit(x=[x, y],
                  #y={"final_prediction1": apilabel},
                  y={"final_prediction1": apilabel, "final_prediction2": taglabel,"final_prediction3": apitaglabel},  # y=apilabel,#
                  epochs=300,
                  batch_size=128
                  )
    des_model.save(parentdic + "/data/data3_mtask_shuffle/model_api1_gru/new_fusion_model.h5")

buildmodel()

def findmashuptagssimi():

    trainmashuptagdict=dict()
    with open(parentdic + "/data/data3_mtask_shuffle/mashup_train.csv", 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for data in reader:
            id = data[0]
            mashup_name = data[1].strip()

            apis = data[6]
            mashup_des = data[3]
            tags = data[4]
            trainmashuptagdict.__setitem__(mashup_name,tags)

    testtainmashupsimidict=dict()
    with open(parentdic + "/data/data3_mtask_shuffle/mashup_test.csv", 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for data in reader:
            id = data[0]
            mashup_name = data[1].strip()

            apis = data[6]
            mashup_des = data[3]
            tags = data[4]
            listtags=tags.split("###")

            for k,v in trainmashuptagdict.items():
                trainmashupname=k
                listtrainmashuptags=v.split("###")
                #print(listtags,listtrainmashuptags)
                sim=(2*len(set(listtags)&set(listtrainmashuptags)))/(len(set(listtags))+len(set(listtrainmashuptags)))
                #print(sim)
                testtainmashupsimidict.__setitem__(mashup_name+"###"+trainmashupname,sim)
            #print("----------------")

    return testtainmashupsimidict

def findtestapidesemb():

    testapidesemb=dict()
    with open(parentdic + "/data/data3_mtask_shuffle/electra/testapidesemb.csv", "r") as f:
        reader = csv.reader(f)
        for data in reader:
            api_name=data[0]
            api_des_emb=[float(i) for i in (data[1].split(" "))]
            testapidesemb.__setitem__(api_name,api_des_emb)

    return testapidesemb

def predict():

    apidict = getapi2id()  # 把apiname映射成id

    id2apidict=getid2api()

    testapidesembdict = findtestapidesemb()

    des_model = load_model(parentdic + "/data/data3_mtask_shuffle/model_api1_gru/des_gru.h5")
    stru_model = load_model(parentdic + "/data/data3_mtask_shuffle/model_api1_gru/stru_model.h5")

    des_lstm_model = load_model(parentdic + "/data/data3_mtask_shuffle/model_api1_gru/gru.h5")
    stru_lstm_model = load_model(parentdic + "/data/data3_mtask_shuffle/model_api1_gru/stru_gru.h5")

    fusion_model=load_model(parentdic + "/data/data3_mtask_shuffle/model_api1_gru/new_fusion_model.h5")

    mashupembdict, apiembdict = findmashupapiemb()  # 是结构的emb,长度为16

    trainmashupdesembdict, apidesembdict = finddesemb()  # 是文本的emb,长度为768


    #得到testmashup的文本的embedding
    testmashupdesembdict=dict()
    with open(parentdic + "/data/data3_mtask_shuffle/electra/testmashupdesemb.csv", "r") as f:
        reader = csv.reader(f)
        for data in reader:
            test_mashup_name = data[0]
            test_mashup_des_emb = [float(i) for i in (data[1].split(" "))]
            testmashupdesembdict.__setitem__(test_mashup_name, test_mashup_des_emb)



    #先构建出test_mashup的stru的embedding
    #从文本中找出相似邻居，构建
    neighor_num=20
    test_mashup_stru_emb_dict = dict()
    test_mashup_neighbor_apis=dict()
    for k, v in testmashupdesembdict.items():
        test_mashupname = k
        simdict = dict()
        for i, j in trainmashupdesembdict.items():
            train_mashup_name = i
            sim1 = 1 - cosine(v, j)  # train和test的des相似度
            #sim2=testtainmashupsimidict[test_mashupname+"###"+train_mashup_name]
            simdict.__setitem__(train_mashup_name, sim1)#0.5*sim1+0.5*sim2
        # 排序，选取出最高的topn个

        simdict = sorted(simdict.items(), key=lambda x: x[1], reverse=True)
        topntrainmashup = simdict[:neighor_num]


        totalsimilar=0.0

        #neighborapislist=list()
        for i in topntrainmashup:
            similar = i[1]
            totalsimilar=totalsimilar+similar

        test_sum = np.array(0)
        for i in topntrainmashup:
            trainmashupsimilar = i[0]
            similar = i[1]/totalsimilar
            #print(similar)
            #print(mashupembdict[trainmashupsimilar])
            trainmashupstruemb = [j * similar for j in
                                  mashupembdict[trainmashupsimilar]]  # 这个是和testmashup相似的trainmashup的stru embedding
            test_sum = test_sum + np.array(trainmashupstruemb)

        #print("----------------------")
        test_mashup_stru_emb = test_sum
        #print(test_mashup_stru_emb)
        test_mashup_stru_emb_dict.__setitem__(test_mashupname, test_mashup_stru_emb)
        #break

    topn = [1, 2, 3, 4, 5]
    totalprecision = [0.0, 0.0, 0.0, 0.0, 0.0]
    totalrecall = [0.0, 0.0, 0.0, 0.0, 0.0]
    totalf1 = [0.0, 0.0, 0.0, 0.0, 0.0]
    totalmap = [0.0, 0.0, 0.0, 0.0, 0.0]
    totalngcd = [0.0, 0.0, 0.0, 0.0, 0.0]
    count = 0

    with open(parentdic + "/data/data3_mtask_shuffle/mashup_test.csv", 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for data in reader:
            id = data[0]
            mashup_name = data[1].strip()
            mashup_des = data[3]

            listapis=data[6].split("###")

            #for api in range(0,len(listapis)):

            selectapiname=listapis[2].strip()

            testmashupdes = []
            testmashupstru = []


            test_mashup_stru_emb = test_mashup_stru_emb_dict[mashup_name]
            testmashupstru.append(test_mashup_stru_emb)

            test_mashup_des_emb = testmashupdesembdict[mashup_name]
            testmashupdes.append(test_mashup_des_emb)

            stru_predict = stru_lstm_model.predict(np.array([[apiembdict[selectapiname]]]))
            apilstmstruemd = []
            for i in stru_predict:
                apilstmstruemd.append(i.tolist())

            des_predict = des_lstm_model.predict(np.array([[testapidesembdict[selectapiname]]]))
            apilstmdesemd = []
            for i in des_predict:
                apilstmdesemd.append(i.tolist())

            # neighborapislabel.append(test_mashup_neighbor_apis[mashup_name])

            testmashupdes = np.array(testmashupdes)
            testmashupstru = np.array(testmashupstru)

            apilstmstruemd = np.array(apilstmstruemd)
            apilstmdesemd = np.array(apilstmdesemd)

            des_intermediate_layer_model = Model(inputs=des_model.input,
                                                 outputs=des_model.get_layer("layer_dense_2").output)
            x = des_intermediate_layer_model.predict([testmashupdes, apilstmdesemd])

            stru_intermediate_layer_model = Model(inputs=stru_model.input,
                                                  outputs=stru_model.get_layer("layer_dense_2").output)
            y = stru_intermediate_layer_model.predict([testmashupstru, apilstmstruemd])

            predict = fusion_model.predict([x, y])
            # predict1 = des_model.predict([testmashupdes, apides])
            # predict2 = stru_model.predict([testmashupstru, apistru])

            # print(type(predict1))
            # predict=np.array(predict1[0]) + np.array(predict2[0])

            predictlist = predict[0].reshape(-1)

            if (selectapiname in apidict.keys()):

                selectindex = int(apidict[selectapiname])

                predictlist[selectindex] = 0.0
            else:
                pass

            testapisset = set()
            for i in [i for i in listapis if i not in listapis[2]]:
                i = i.strip()
                if (i == "none"):
                    continue
                testapisset.add(i)

            for i in range(0, len(topn)):


                re = map(list(predictlist).index, heapq.nlargest(topn[i], predictlist))

                recommendset = set()
                ranked_list = list()
                for key in re:
                    recommendset.add(id2apidict[int(key)])
                    ranked_list.append(id2apidict[int(key)])

                #####print(recommendset, testapisset, listapis[1])

                commonset = recommendset & testapisset

                # print(recommendset)

                # map = AP(ranked_list, testapisset)

                precision = len(commonset) / topn[i]
                recall = len(commonset) / float(len(testapisset))
                # f1 = (2 * len(commonset)) / (topn[i] + len(newtestapiset))
                if (precision == 0.0 or recall == 0.0):
                    f1 = 0.0
                else:
                    f1 = 2 * precision * recall / (precision + recall)
                ap = AP(ranked_list, testapisset)
                ngcd = NGCD(ranked_list, testapisset)
                # print(precision)

                totalprecision[i] = totalprecision[i] + precision
                totalrecall[i] = totalrecall[i] + recall
                totalf1[i] = totalf1[i] + f1
                totalmap[i] = totalmap[i] + ap
                totalngcd[i] = totalngcd[i] + ngcd

            count = count + 1

    totalprecision = [i / count for i in totalprecision]
    totalrecall = [i / count for i in totalrecall]
    totalf1 = [i / count for i in totalf1]
    totalmap = [i / count for i in totalmap]
    totalngcd = [i / count for i in totalngcd]

    print()
    # print(totalprecision)
    print("precision", '\t'.join([("%.8f" % x).ljust(12) for x in totalprecision]))
    print("totalrecall", '\t'.join([("%.8f" % x).ljust(12) for x in totalrecall]))
    print("totalf1", '\t'.join([("%.8f" % x).ljust(12) for x in totalf1]))
    print("totalmap", '\t'.join([("%.8f" % x).ljust(12) for x in totalmap]))
    print("totalngcd", '\t'.join([("%.8f" % x).ljust(12) for x in totalngcd]))


predict()



