
from numpy import *
import sys
import re
import matplotlib.pyplot as plt
import itertools as it
import matplotlib as mpl
import operator


depth=0


"""
this function  provides the normalized data
 for the logistic regression part 
 data array consists of normalized float numbers
 label array contains the value of 0 or 1 which 
 represents whether book is written by Arthur Conan Doyle
 or Herman Melville
"""
def loadDataSet():
    data = []
    label= []
    filename = 'normalize.txt'
    with open(filename) as f:
        for line in f:
            line = line.strip().split()
            temp=[]
            for index in range(0,len(line)-1):
                temp.append( float(line[index]))
            data.append(temp)
            label.append(int(line[-1]))
    return data,label


"""
the main part of activation function is sigmoid function
x can be a float number or an array
"""
def sigmoid(x):
    return 1.0/(1+exp(-x))


"""
the train part of my code
stochastic gradient descent function
values which are in weights array can be set randomly, 
for the sake of convenience, 
set each element in  weights array to be 1 and 
set the learning rate to be 0.0001
set the number of iterations to be 1000  
"""
def stocGradDescent0(dataMat, labelMat):
    dataMatrix=mat(dataMat)
    classLabels=labelMat
    m,n=shape(dataMatrix)
    alpha=0.0001
    maxCycles = 1000
    weights=ones((n,1))

    for k in range(maxCycles):
        for i in range(m):
            h = sigmoid(sum(dataMatrix[i] * weights))
            error = classLabels[i] - h
            weights = weights + alpha * error * dataMatrix[i].transpose()
    return weights


"""
since there are many irrelevant informations in the text
I use regular expression to remove the "noise" in the paragraph
these noises are shown as the punctuations as followed
"""
def Punctuation(string):
    # punctuation marks
    punctuations = '''()-[]{};:'"\,<>./@#$%^&*_~'''

    # traverse the given string and if any punctuation
    # marks occur replace it with null
    for x in string:
        if x in punctuations:
            string = string.replace(x, "")
    return string

"""
in order to calculate the logistic regression
we should do the normalization part which also means "scale"
I choose to use Min-max normalization to process
the data
"""
def Normalization(x):
    return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]


"""
the purpose of extract function is to process data 
from  Project Gutenberg website 
"""
def extract():
    g = open("result.txt", "a")
    g.write("the")
    g.write("\t")
    g.write("exclamation")
    g.write("\t")
    g.write("question")
    g.write("\t")
    g.write("ed")
    g.write("\t")
    g.write("ly")
    g.write("\t")
    g.write("name")
    g.write("\t")
    g.write("type")
    g.write("\t")
    g.write("he")
    g.write("\t")
    g.write("she")
    g.write("\t")
    g.write("ing")
    g.write("\t")
    g.write("who")
    g.write("\n")
    g.close()
    filename ='4signature.txt'
    TheArray=[]
    ExclamationArray=[]
    QuestionArray=[]
    EdArray=[]
    LyArray=[]
    NameArray=[]
    TypeArray=[]
    HeArray=[]
    SheArray=[]
    IngArray=[]
    label=[]
    count=0
    countThe=0
    countExclamation=0
    countQuestion=0
    countEd=0
    countLy=0
    countName=0
    countType=0
    countHe=0
    countShe=0
    countIng=0
    """
    process the text from Arthur Conan Doyle's 10 books
    For each book, I extract a certain part of contents
    """
    with open(filename) as f:
        for line in f:
            line=line.strip()
            if len(line)==0:
                continue
            line=Punctuation(line)
            data=line.split(" ")
            for i in range(0,len(data)):
                count+=1
                """
                for every 250 words in 10 books, count the total number of 
                10 features as one sample 
                I choose to write these samples into a separate txt file
                so I use write() function
                """
                if count==250:
                    h = open("result.txt", "a")
                    h.write(str(countThe))
                    TheArray.append(countThe)
                    h.write("\t")
                    h.write(str(countExclamation))
                    ExclamationArray.append(countExclamation)
                    h.write("\t")
                    h.write(str(countQuestion))
                    QuestionArray.append(countQuestion)
                    h.write("\t")
                    h.write(str(countEd))
                    EdArray.append(countEd)
                    h.write("\t")
                    h.write(str(countLy))
                    LyArray.append(countLy)
                    h.write("\t")
                    h.write(str(countName))
                    NameArray.append(countName)
                    h.write("\t")
                    h.write(str(countType))
                    TypeArray.append(countType)
                    h.write("\t")
                    h.write(str(countHe))
                    HeArray.append(countHe)
                    h.write("\t")
                    h.write(str(countShe))
                    SheArray.append(countShe)
                    h.write("\t")
                    h.write(str(countIng))
                    IngArray.append(countIng)
                    h.write("\t")
                    """
                    for each sample, mark 1 as
                    Arthur Conan Doyle's book in the logistic regression part
                    """
                    h.write(str(1))
                    label.append(str(1))
                    h.write("\n")
                    h.close()
                    count = 0
                    countThe = 0
                    countExclamation = 0
                    countQuestion = 0
                    countEd = 0
                    countLy = 0
                    countName=0
                    countType=0
                    countHe=0
                    countShe=0
                    countIng=0
                """
                count all the features that appear in the books
                """
                if (data[i]).lower()=="the":
                    countThe+=1
                elif "!" in data[i]:
                    countExclamation+=1
                elif "?" in data[i]:
                    countQuestion+=1
                elif "ed" in data[i] or "ive" in data[i]:
                    countEd+=1
                elif "ly" in data[i]:
                    countLy+=1
                elif (data[i]).lower()=="sherlock":
                    countName+=1
                elif (data[i]).lower()=="holmes":
                    countType+=1
                elif (data[i]).lower()=="he":
                    countHe+=1
                elif (data[i]).lower()=="she":
                    countShe+=1
                elif "ing" in data[i]:
                    countIng+=1

    filename1 ='whale.txt'
    count = 0
    countThe = 0
    countExclamation = 0
    countQuestion = 0
    countEd = 0
    countLy = 0
    countName = 0
    countType = 0
    countHe = 0
    countShe = 0
    countIng = 0
    """
    process the text from Herman Melville's 10 books
    For each book, I extract a certain part of contents
    """
    with open(filename1) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            line = Punctuation(line)
            data = line.split(" ")
            for i in range(0, len(data)):
                count += 1
                """
                for every 250 words in 10 books, count the total number of 
                10 features as one sample 
                I choose to write these samples into a separate txt file
                so I use write() function
                """
                if count == 250:
                    h = open("result.txt", "a")
                    h.write(str(countThe))
                    TheArray.append(countThe)
                    h.write("\t")
                    h.write(str(countExclamation))
                    ExclamationArray.append(countExclamation)
                    h.write("\t")
                    h.write(str(countQuestion))
                    QuestionArray.append(countQuestion)
                    h.write("\t")
                    h.write(str(countEd))
                    EdArray.append(countEd)
                    h.write("\t")
                    h.write(str(countLy))
                    LyArray.append(countLy)
                    h.write("\t")
                    h.write(str(countName))
                    NameArray.append(countName)
                    h.write("\t")
                    h.write(str(countType))
                    TypeArray.append(countType)
                    h.write("\t")
                    h.write(str(countHe))
                    HeArray.append(countHe)
                    h.write("\t")
                    h.write(str(countShe))
                    SheArray.append(countShe)
                    h.write("\t")
                    h.write(str(countIng))
                    IngArray.append(countIng)
                    h.write("\t")
                    """
                    for each sample, mark 0 as
                    Herman Melville's book in the logistic regression part
                    """
                    h.write(str(0))
                    label.append(str(0))
                    h.write("\n")
                    h.close()
                    count = 0
                    countThe = 0
                    countExclamation = 0
                    countQuestion = 0
                    countEd = 0
                    countLy = 0
                    countName = 0
                    countType = 0
                    countHe = 0
                    countShe = 0
                    countIng = 0
                """
                count all the features that appear in the books
                """
                if (data[i]).lower()=="the":
                    countThe+=1
                elif "!" in data[i]:
                    countExclamation+=1
                elif "?" in data[i]:
                    countQuestion+=1
                elif "ed" in data[i] or "ive" in data[i]:
                    countEd+=1
                elif "ly" in data[i]:
                    countLy+=1
                elif (data[i]).lower()=="sherlock":
                    countName+=1
                elif (data[i]).lower()=="holmes":
                    countType+=1
                elif (data[i]).lower()=="he":
                    countHe+=1
                elif (data[i]).lower()=="she":
                    countShe+=1
                elif "ing" in data[i]:
                    countIng+=1

    """
    for every sample, Min-max normalization process
    should be done
    """
    the=Normalization(TheArray)
    exclamation=Normalization(ExclamationArray)
    question=Normalization(QuestionArray)
    ed=Normalization(EdArray)
    ly=Normalization(LyArray)
    name=Normalization(NameArray)
    type=Normalization(TypeArray)
    he=Normalization(HeArray)
    she=Normalization(SheArray)
    ing=Normalization(IngArray)


    """
    after the Min-max normalization,
    I open a new normalize txt file to store the data
    so there are two files named result.txt and normalize.txt 
    which represent the data before and after
    normalization process
    """
    k=open("normalize.txt", "a")
    for i in range(0,len(the)):
        k.write(str(the[i]))
        k.write("\t")
        k.write(str(exclamation[i]))
        k.write("\t")
        k.write(str(question[i]))
        k.write("\t")
        k.write(str(ed[i]))
        k.write("\t")
        k.write(str(ly[i]))
        k.write("\t")
        k.write(str(name[i]))
        k.write("\t")
        k.write(str(type[i]))
        k.write("\t")
        k.write(str(he[i]))
        k.write("\t")
        k.write(str(she[i]))
        k.write("\t")
        k.write(str(ing[i]))
        k.write("\t")
        k.write(str(label[i]))
        k.write("\n")
    k.close()

"""
the predict part of my code
"""
def determineResult(weights):
    #test file in the command line
    filename = sys.argv[3] + '.txt'
    test=[]
    countThe = 0
    countExclamation = 0
    countQuestion = 0
    countEd = 0
    countLy = 0
    countName=0
    countType=0
    countHe=0
    countShe=0
    countIng=0
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            line = Punctuation(line)
            data = line.split(" ")
            """
            for 250 words paragraph
            count total number of 10 features and 
            then normalize 
            """
            for i in range(0,len(data)):
                if (data[i]).lower()=="the":
                    countThe+=1
                elif "!" in data[i]:
                    countExclamation+=1
                elif "?" in data[i]:
                    countQuestion+=1
                elif "ed" in data[i] or "ive" in data[i]:
                    countEd+=1
                elif "ly" in data[i]:
                    countLy+=1
                elif (data[i]).lower()=="sherlock":
                    countName+=1
                elif (data[i]).lower()=="holmes":
                    countType+=1
                elif (data[i]).lower()=="he":
                    countHe+=1
                elif (data[i]).lower()=="she":
                    countShe+=1
                elif "ing" in data[i]:
                    countIng+=1
    test.append(countThe)
    test.append(countExclamation)
    test.append(countQuestion)
    test.append(countEd)
    test.append(countLy)
    test.append(countName)
    test.append(countType)
    test.append(countHe)
    test.append(countShe)
    test.append(countIng)
    """
    normalization must be done
    """
    test=Normalization(test)
    """
    first do dot operation
    then use sigmoid function  
    to calculate final result
    """
    print("The output is shown as followed: ")
    print(sigmoid(dot(array(test), array(weights))))

"""
-------------------------------------------------------------------------------------------------------------------
the following part is decision tree
"""
def preProcess():
    the = []
    exclamation = []
    question = []
    ed = []
    ly = []
    name=[]
    type=[]
    he=[]
    she=[]
    ing=[]
    with open("result.txt") as f:
        h = open("decision.txt", "a")
        label=f.readline()
        h.write(label)
        h.close()
        for line in f:
            data=line.strip().split()
            the.append(float(data[0]))
            exclamation.append(float(data[1]))
            question.append(float(data[2]))
            ed.append(float(data[3]))
            ly.append(float(data[4]))
            name.append(float(data[5]))
            type.append(float(data[6]))
            he.append(float(data[7]))
            she.append(float(data[8]))
            ing.append(float(data[9]))
    """
    since the values in logistic regression part are continuous
    we should convert these continuous variables to discrete variables
    so for each feature of total 10 features
    first I calculate the mean of each feature in result.txt file
    """
    the_mean=mean(the)
    exclamation_mean=mean(exclamation)
    question_mean=mean(question)
    ed_mean=mean(ed)
    ly_mean=mean(ly)
    name_mean=mean(name)
    type_mean=mean(type)
    he_mean=mean(he)
    she_mean=mean(she)
    ing_mean=mean(ing)
    with open("result.txt") as f:
        f.readline()
        for line in f:
            data=line.strip().split()
            g = open("decision.txt", "a")
            """
            if each feature value from one sample is greater than average, 
            I set this value as "Y"
            otherwise as "N" 
            so we can get a decision.txt file which contains only "Y" and "N"
            """
            if float(data[0])>=the_mean:
                g.write("Y")
                g.write("\t")
            else:
                g.write("N")
                g.write("\t")
            if float(data[1])>=exclamation_mean:
                g.write("Y")
                g.write("\t")
            else:
                g.write("N")
                g.write("\t")
            if float(data[2])>=question_mean:
                g.write("Y")
                g.write("\t")
            else:
                g.write("N")
                g.write("\t")
            if float(data[3])>=ed_mean:
                g.write("Y")
                g.write("\t")
            else:
                g.write("N")
                g.write("\t")
            if float(data[4])>=ly_mean:
                g.write("Y")
                g.write("\t")
            else:
                g.write("N")
                g.write("\t")
            if float(data[5])>=name_mean:
                g.write("Y")
                g.write("\t")
            else:
                g.write("N")
                g.write("\t")
            if float(data[6])>=type_mean:
                g.write("Y")
                g.write("\t")
            else:
                g.write("N")
                g.write("\t")
            if float(data[7])>=he_mean:
                g.write("Y")
                g.write("\t")
            else:
                g.write("N")
                g.write("\t")
            if float(data[8])>=she_mean:
                g.write("Y")
                g.write("\t")
            else:
                g.write("N")
                g.write("\t")
            if float(data[9])>=ing_mean:
                g.write("Y")
                g.write("\t")
            else:
                g.write("N")
                g.write("\t")
            """
            for Arthur Conan Doyle's book, we mark it as "Y"
            for Herman Melville's book, we mark it as "N"
            """
            if int(data[-1])==1:
                g.write("Y")
                g.write("\n")
            else:
                g.write("N")
                g.write("\n")
            g.close()
    return the_mean,exclamation_mean,question_mean,ed_mean,ly_mean,name_mean,type_mean,he_mean,she_mean,ing_mean



"""
the purpose of this function is to process data
from decision.txt
dataSet array returns every features
labels array returns which author wrote the book
"""
def createDataSet():
    dataSet=[]
    filename = "decision.txt"
    with open(filename) as f:
        labels=f.readline().split()
        labels=labels[:len(labels)-1]
        for line in f:
            dataSet.append(line.split())
    return dataSet,labels


"""
to calculate entropy in the decision tree
"""
def calcShannonEnt(dataSet):
    """
    use dictionary to count total number of
    "Y" and "N" that appear
    """
    dic={}
    for i in range(0,len(dataSet)):
        if dataSet[i][-1] not in dic:
            dic[dataSet[i][-1]]=0
        dic[dataSet[i][-1]] +=1
    res = 0.0
    for key in dic.keys():
        res-=float(dic[key])/len(dataSet)*math.log(dic[key]/len(dataSet),2)
    return res

"""
every time when the best feature is chosen
we need to split the data set into two parts
"""
def splitDataSet(dataSet,axis,value):
    res=[]
    for i in range(0,len(dataSet)):
        if dataSet[i][axis]==value:
            temp=dataSet[i][:axis]+dataSet[i][axis+1:]
            res.append(temp)
    return res

"""
for each feature, we calculate the gain
finally we pick up the maximal gain
and then choose the maximal gain feature 
to split
"""
def chooseBestFeatureToSplit(dataSet):
    baseEntropy=calcShannonEnt(dataSet)
    bestInforGain = -1
    bestFeature = -1

    #calculate each column which is in the dataSet
    for i in range(0,len(dataSet[0])-1):
        column=[dataSet[j][i] for j in range(0,len(dataSet))]
        s = set()
        for k in range(0,len(column)):
            s.add(column[k])
        newEntropy=0
        for element in s:
            subset=splitDataSet(dataSet,i,element)
            entropy=calcShannonEnt(subset)
            newEntropy+=entropy*len(subset)/len(dataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain>bestInforGain:
            bestInforGain = infoGain
            bestFeature = i
    return bestFeature

"""
if there are no more features 
if the number of "Y" is greater than "N", we return "Y"
as the result and vice versa

for the cutoff part, if the tree reaches certain depth,
we should also do this,
if the number of "Y" is greater than "N", we return "Y"
as the result and vice versa
"""
def majority(classList):
    dic={}
    for i in range(0,len(classList)):
        if classList[i] not in dic:
            dic[classList[i]]=0
        dic[classList[i]]+=1

    sortedClassCount = sorted(dic.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet,labels):

    classList=[dataSet[i][-1] for i in range(0,len(dataSet))]
    """
    if there are all "Y" , we return "Y"
    if there are all "N", we return "N"
    """
    if classList.count(classList[-1])==len(classList):
        return classList[-1]
    """
    no more features to split
    return majority
    """
    if len(dataSet[0])==1:
        return majority(classList)
    """
    reach certain depth
    return majority
    """
    global depth
    depth += 1
    if depth > 20:
        return majority(classList)

    bestFeat=chooseBestFeatureToSplit(dataSet)
    mark=labels[bestFeat]
    temp=[]
    temp.extend(labels[:bestFeat]+labels[bestFeat+1:])

    """
    use a dictionary to store tree infomations
    also can hard code dictionary easily 
    """
    tree={mark:{}}
    s=set([dataSet[i][bestFeat] for i in range(0,len(dataSet))])

    """
    add up "missing" infomation at the leaf node
    """
    if len(s) == 1:
        if "Y" not in s:
            tree[mark]["Y"] = "N"
        if "N" not in s:
            tree[mark]["N"] = "N"
    """
    recursive call
    """
    for value in s:
        subset=splitDataSet(dataSet,bestFeat,value)
        tree[mark][value]=createTree(subset,temp)
    return tree

def testDic(read_dictionary,the_mean,ex_mean,qu_mean,ed_mean,ly_mean,name_mean,type_mean,he_mean,she_mean,ing_mean):
    filename = sys.argv[2]
    test = []
    countThe = 0
    countExclamation = 0
    countQuestion = 0
    countEd = 0
    countLy = 0
    countName=0
    countType=0
    countHe=0
    countShe=0
    countIng=0
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            line = Punctuation(line)
            data = line.split(" ")
            for i in range(0, len(data)):
                """
                for 250 words paragraph
                count total number of 10 features and 
                for decision tree, normalization is no longer needed
                """
                if (data[i]).lower()=="the":
                    countThe+=1
                elif "!" in data[i]:
                    countExclamation+=1
                elif "?" in data[i]:
                    countQuestion+=1
                elif "ed" in data[i] or "ive" in data[i]:
                    countEd+=1
                elif "ly" in data[i]:
                    countLy+=1
                elif (data[i]).lower()=="sherlock":
                    countName+=1
                elif (data[i]).lower()=="holmes":
                    countType+=1
                elif (data[i]).lower()=="he":
                    countHe+=1
                elif (data[i]).lower()=="she":
                    countShe+=1
                elif "ing" in data[i]:
                    countIng+=1
    test.append(countThe)
    test.append(countExclamation)
    test.append(countQuestion)
    test.append(countEd)
    test.append(countLy)
    test.append(countName)
    test.append(countType)
    test.append(countHe)
    test.append(countShe)
    test.append(countIng)

    """
    here I use a dictionay to store the 250 word
    infomation
    """
    dic = {}
    if test[0]>=the_mean:
        dic["the"]="Y"
    else:
        dic["the"]="N"
    if test[1]>=ex_mean:
        dic["exclamation"]="Y"
    else:
        dic["exclamation"]="N"
    if test[2]>=qu_mean:
        dic["question"]="Y"
    else:
        dic["question"]="N"
    if test[3]>=ed_mean:
        dic["ed"]="Y"
    else:
        dic["ed"]="N"
    if test[4]>=ly_mean:
        dic["ly"]="Y"
    else:
        dic["ly"]="N"
    if test[5]>=name_mean:
        dic["name"]="Y"
    else:
        dic["name"]="N"
    if test[6]>=type_mean:
        dic["type"]="Y"
    else:
        dic["type"]="N"
    if test[7]>=he_mean:
        dic["he"]="Y"
    else:
        dic["he"]="N"
    if test[8]>=she_mean:
        dic["she"]="Y"
    else:
        dic["she"]="N"
    if test[9]>=ing_mean:
        dic["ing"]="Y"
    else:
        dic["ing"]="N"

    """
    here I use a funtion to reach the leaf node 
    of the decision tree I created to get the final 
    result
    """
    # print(dic)
    norm=read_dictionary.copy()
    while norm.keys() !="Y" or "N":
        for key in norm.keys():
            norm=norm[key][dic[key]]
            if norm=="Y"or norm=="N":
                return norm



def main():
    if sys.argv[1]=="train":
        if sys.argv[2]=="regression":
            extract()
            dataMat, labelMat = loadDataSet()
            weights=stocGradDescent0(dataMat, labelMat).getA()
            print("weights array is shown as followed: ")
            print(weights)


        if sys.argv[2] == "tree":
            extract()
            the_mean,ex_mean,qu_mean,ed_mean,ly_mean,name_mean,type_mean,he_mean,she_mean,ing_mean=preProcess()
            dataSet, labels = createDataSet()
            mydic=createTree(dataSet, labels)
            print("The decision tree model is shown as followed: ")
            print(mydic)
            print("mean is shown as followed")
            print([the_mean,ex_mean,qu_mean,ed_mean,ly_mean,name_mean,type_mean,he_mean,she_mean,ing_mean])

            #save dictionary
            save('my_file.npy', mydic)

    elif sys.argv[1]=="predict":
            # Load dictionary
            read_dictionary = {'type': {'N': {'ing': {'N': {'ly': {'N': {'he': {'N': {'the': {'N': {'name': {'N': {'exclamation': {'N': {'ed': {'N': {'question': {'N': {'she': {'N': 'Y', 'Y': 'Y'}}, 'Y': {'she': {'N': 'Y', 'Y': 'Y'}}}}, 'Y': {'question': {'N': {'she': {'N': 'N', 'Y': 'N'}}, 'Y': {'she': {'N': 'N', 'Y': 'N'}}}}}}, 'Y': {'ed': {'N': {'question': {'N': {'she': {'Y': 'N', 'N': 'Y'}}, 'Y': {'she': {'N': 'N', 'Y': 'N'}}}}, 'Y': {'question': {'N': 'N', 'Y': {'she': {'N': 'Y', 'Y': 'Y'}}}}}}}}, 'Y': 'Y'}}, 'Y': 'N'}}, 'Y': 'Y'}}, 'Y': 'N'}}, 'Y': 'N'}}, 'Y': 'Y'}}
            print("the imported decision tree model is shown as followed: ")
            print(read_dictionary)
            the_mean=14.123067010309278
            ex_mean=1.053479381443299
            qu_mean=1.419458762886598
            ed_mean=11.918170103092784
            ly_mean=4.072809278350515
            name_mean=0.0663659793814433
            type_mean=0.389819587628866
            he_mean=2.9510309278350517
            she_mean=0.4420103092783505
            ing_mean=7.468427835051546
            print("The output is shown as followed: ")
            print(testDic(read_dictionary, the_mean,ex_mean,qu_mean,ed_mean,ly_mean,name_mean,type_mean,he_mean,she_mean,ing_mean))


if __name__=='__main__':
    main()
