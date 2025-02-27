# Tool library for CRF model
# Hsin-Min Lu (luim@ntu.edu.tw) 2023/12/22
# For non-commercial use only

import re
import pycrfsuite
import pycrfsuite as pysuite
from nltk import tokenize
import numpy as np
import sys


target = ['B1','B1A','B1B','B2','B3','B4','B5','B6','B7','B7A','B8','B9','B9A','B9B',
              'B10','B11','B12','B13','B14','B15']
core = [0, 1, 4, 6, 8, 9]


def recommendTag(text):
    preN = text[0:15]
    criteria = re.findall(r'ITEM[s]?\s*[I0-9]*[(]?[A-Za-z]?[)]?[.]?',preN,re.IGNORECASE)
    if len(criteria) == 1:
        sign = re.findall(r'[0-9]+[()]?[A-Za-z]?[)]?',criteria[0])
        if len(sign) != 0:
            return('B'+sign[0])
        else:
            return('O')
    else:
        return('O')

def itemShow(text):
    criteria = re.findall(r'^\s*item',text[0:15],re.IGNORECASE)
    if len(criteria) == 1 :
        return(True)
    else:
        return(False)

def checkSpecialContent(text):
    criteria1 = re.findall(r'incorporated[\s\w]*by[\s\w]*reference',text[0:300],re.IGNORECASE)
    criteria2 = re.findall(r'set[\s\w]*forth[\s\w]*page',text[0:300],re.IGNORECASE)
    if len(criteria1) > 0 or len(criteria2) > 0:
        return(True)
    else:
        return(False)

def checkSignature(text):
    criteria = re.findall(r'signature[s]',text,re.IGNORECASE)
    if len(criteria) != 0:
        return(True)
    else:
        return(False)

def unigram(text, n = 15, lower = True):
    # firstN = text.split()
    # filtered_words = [word for word in firstN][0:8]
    if lower == True:
        text = text.strip().lower()
    else:
        text = text.strip()
    filtered_words = tokenize.word_tokenize(text)[0:n]    
    return(filtered_words)

def bigram(ugram, lower = True):
    bigram = []
    for i, atok in enumerate(ugram):
        if i == 0:
            continue
        if lower == True:
            bigram.append(ugram[i-1].lower() + "_" + ugram[i].lower())
        else:
            bigram.append(ugram[i-1] + "_" + ugram[i])
    return bigram

def itemNameCheck(prefix, text):
    alltasks = {'item 1': r'^item[\s\w.-]*business',
                'item 1A': r'^item[\s\w.-]*risk\s+factor',
                'item 1B': r'^item\[\s\w.-]*unresolved\s+staff\s+comment',
                'item 2': r'^item[\s\w.-]*propert[yies]',
                'item 3': r'^item[\s\w.-]*legal[\s]*proceedings',
                'item 4': r'^item[\s\w.-]*submission[\s\w]*'
                           'matters|item[\s\w.-－]mine[\s]*safety[\s]*disclosure',
                'item 5': r'^item[\s\w.-]*market[\s\w]*registrant[’''\ss]*common[\s]*equity',
                'item 6': r'^item[\s\w.-]*select[ed][\s\w]*financial\s+data',
                'item 7': r'^item[\s\w.-]*management[\s\'s’'']*discussion[\s\w]*analysis',
                'item 7A': r'^item[\s\w.-]*quantitative[\s\w]*qualitative[\s]*disclosure',
                'item 8': r'^item[\s\w.-]*financial[\s]*statement[\s\w]*supplementary',
                'item 9': r'^item[\s\w.-]*changes[\s\w]*disagreement[s]',
                'item 9A': r'^item[\s\w.-]*cntrol[\s\w]*procedure[s]',
                'item 9B': r'^item[\s\w.-]*other[\s]*information',
                'item 10': r'^item[\s\w.-]*director[,\w\s]*executive[,\w\s]*officer',
                'item 11': r'^item[\s\w.-]*executive[\s]*compensation',
                'item 12': r'^item[\s\w.-]*security[\s]*ownership[,\w\s]*certain[\s]*beneficial',
                'item 13': r'^item[\s\w.-]*certain[\s]*relationship'
                            '[,\w\s]*related[\s]*transaction',
                'item 14': r'^item[\s\w.-]*principal[\s]*account[\s\w]*fees[\s\w]*services',
                'item 15': r'^item\s*15[.\s-]*exhibit[\w\s]*schedules'}
    value = {}
    text = text.strip()
    for item, condition in alltasks.items():
        criteria = re.findall(condition, text, re.IGNORECASE)
        if len(criteria) == 1:
            value[prefix+item] = True
        #else:        
        #    value[prefix+item] = False
    return(value)


def itemNameCheckLose(prefix, text):
    alltasks = {'item 1': r'^\s*item[\s\\.-]+1(?!\.A)\b',
                'item 1A': r'^\s*item[\s\\.-]+1\.?A\b',
                'item 1B': r'^\s*item[\s\\.-]+1B\b',
                'item 2': r'^\s*item[\s\\.-]+2\b',
                'item 3': r'^\s*item[\s\\.-]+3\b',
                'item 4': r'^\s*item[\s\\.-]+4\b',
                'item 5': r'^\s*item[\s\\.-]+5\b',
                'item 6': r'^\s*item[\s\\.-]+6\b',
                'item 7': r'^\s*item[\s\\.-]+7\b',
                'item 7A': r'^\s*item[\s\\.-]+7A\b',
                'item 8': r'^\s*item[\s\\.-]+8\b',
                'item 9': r'^\s*item[\s\\.-]+9\b',
                'item 9A': r'^\s*item[\s\\.-]+9A\b',
                'item 9B': r'^\s*item[\s\\.-]+9B\b',
                'item 10': r'^\s*item[\s\\.-]+10\b',
                'item 11': r'^\s*item[\s\\.-]+11\b',
                'item 12': r'^\s*item[\s\\.-]+12\b',
                'item 13': r'^\s*item[\s\\.-]+13\b',
                'item 14': r'^\s*item[\s\\.-]+14\b',
                'item 15': r'^\s*item[\s\\.-]+15\b'}
    value = {}
    text = text.strip()
    for item, condition in alltasks.items():
        criteria = re.findall(condition, text, re.IGNORECASE)
        if len(criteria) == 1:
            value[prefix+item] = True
        #else:        
        #    value[prefix+item] = False
    return(value)


# provide the accumulated item count
def item_acc_vector(doc):
    acc_state = []
    this_state = {'item 1': 0, 'item 1A': 0, 'item 1B': 0, 
                  'item 2': 0, 'item 3': 0, 'item 4': 0,
                  'item 5': 0, 'item 6': 0, 'item 7': 0,
                  'item 7A': 0, 'item 8': 0, 'item 9': 0, 'item 9A': 0,
                  'item 9B': 0, 'item 10': 0, 'item 11': 0, 
                  'item 12': 0, 'item 13': 0, 'item 14': 0,
                  'item 15': 0}
    for i, sentence in enumerate(doc[2]):
        tmp1 = itemNameCheckLose("", sentence)
        #print(i, tmp1)
        for aitem in tmp1:
            this_state[aitem] += 1
        acc_state.append(this_state.copy())
        
    # max_nitem = np.median(np.array(list(this_state.values())))
    # if max_nitem <= 0.0:
    #     max_nitem = 1.0
    
    for aitem in acc_state:
        for key, value in aitem.items():
            max_nitem = this_state[key]
            if max_nitem <= 0.0:
                max_nitem = 1.0            
            aitem[key] = value / max_nitem
    
    return acc_state



def preprocessing(text):
    temp = re.sub(r'[,.&()\/\\?:!";\'\"$%]', '', text)
    return(' '.join(temp.split()).lower())

def sentence2features(doc, acc_state, i):
    sentence = doc[2][i].strip()
    filtedSent = preprocessing(sentence)
    #unigrmList = unigram(filtedSent)
    pureUnigram = unigram(sentence)
    pureBigram = bigram(pureUnigram)
    sentSign = itemShow(sentence)    
    senthead = sentence.strip()
    sentheadupper = False
    if len(senthead) > 0:        
        sentheadupper = senthead[0].isupper()
    
    forwpos1 = i/len(doc[2]) if len(doc[2]) != 0 else 0.
    backpos1 = (len(doc[2])-i)/len(doc[2]) if len(doc[2]) != 0 else 0.
    headthreshold = 50.0
    headpos = min(i, headthreshold) / headthreshold * 1.0
    wordlen = len(tokenize.word_tokenize(sentence))
    wordlenmax = 20.0
    wordlen = min(wordlen, wordlenmax) / wordlenmax * 1.0
    featureSet = {
        'sentence.isupperWithItem' : True if sentSign and sentheadupper else False,
        'sentence.forwardPosition' : forwpos1,
        'sentence.backwardPosition' : backpos1,
        'sentence.headpos': headpos,        
        #'sentence.headposraw': i,
        'sentence.wordlen': wordlen,
        #'sentence.unorder': pureUnigram,
        #'sentence.bigrams': pureBigram,
        'sentence.special_content':checkSpecialContent(sentence)
    }
    
    for atoken in pureUnigram:
        featureSet['sentence.unigram.' + atoken] = 1.
    for atoken in pureBigram:
        featureSet['sentence.bigram.' + atoken] = 1.
    
    #for j in range(len(unigrmList)):
    #    featureSet['unigrm'+str(j)] = unigrmList[j]
    value = itemNameCheck('sentence.', sentence)
    for k, k_value in value.items():
        featureSet[k] = k_value
        
    value = itemNameCheckLose('sentence.itemlose.', sentence)
    for k, k_value in value.items():
        featureSet[k] = k_value    
        
    for key, value in acc_state[i].items():
        featureSet['accitem_' + key] = value
        
    if i > 0:
        sentence1 = doc[2][i-1]
        # filtedSent1 = preprocessing(sentence1)
        # unigrmList1 = unigram(filtedSent1)
        pureUnigram1 = unigram(sentence1)
        pureBigram1 = bigram(pureUnigram1)
        sentSign1 = itemShow(sentence1)
        
        featureSet.update({
            '-1:sentence.isupperWithItem' : True if sentSign1 and pureUnigram1[0].isupper() else False,
            #'-1:sentence.forwardPosition' : i/len(doc[2]) if len(doc[2]) != 0 else 0,
            #'-1:sentence.backwardPosition' : (len(doc[2])-i)/len(doc[2]) if len(doc[2]) != 0 else 0,
            #'-1:sentence.headpos': min(i, headthreshold) / headthreshold * 1.0,
            #'-1:sentence.charlen': len(sentence.strip()), 
            #'-1:sentence.unorder': pureUnigram1,
            '-1:sentence.special_content':checkSpecialContent(sentence1)
        })
        #for j in range(len(unigrmList1)):
        #    featureSet['-1:unigrm'+str(j)] = unigrmList1[j]
        for atoken in pureUnigram1:
            featureSet['-1:sentence.unigram.' + atoken] = 1.
        for atoken in pureBigram1:
            featureSet['-1:sentence.bigram.' + atoken] = 1.
        
        value1 = itemNameCheck('-1:sentence.', sentence1)
        for k, k_value in value1.items():
            featureSet[k] = k_value            
        value1 = itemNameCheckLose('-1:sentence.itemlose.', sentence1)
        for k, k_value in value1.items():
            featureSet[k] = k_value
    else:
        sentSign1 = False
        featureSet['BOD']=True
    if i < (len(doc[2])-1):
        sentence1 = doc[2][i+1]
        # filtedSent1 = preprocessing(sentence1)
        unigram1 = unigram(sentence1)
        sentSign2 = itemShow(sentence1)
        pureUnigram1 = unigram(sentence1)
        pureBigram1 = bigram(pureUnigram1)
        
        featureSet.update({
            '+1:sentence.isupperWithItem' : True if sentSign2 and pureUnigram1[0].isupper() else False,
            #'+1:sentence.forwardPosition' : i/len(doc[2]) if len(doc[2]) != 0 else 0,
            #'+1:sentence.backwardPosition' : (len(doc[2])-i)/len(doc[2]) if len(doc[2]) != 0 else 0,
            #'+1:sentence.headpos': min(i, headthreshold) / headthreshold * 1.0,
            #'+1:sentence.charlen': len(sentence1.strip()), 
            #'+1:sentence.unorder': pureUnigram1,
            '+1:sentence.special_content':checkSpecialContent(sentence1)
        })
        #for j in range(len(unigram1)):
        #    featureSet['+1:unigrm'+str(j)] = unigram1[j]
        for atoken in pureUnigram1:
            featureSet['+1:sentence.unigram.' + atoken] = 1.
        for atoken in pureBigram1:
            featureSet['+1:sentence.bigram.' + atoken] = 1.
        value1 = itemNameCheck('+1:sentence.', sentence1)
        for k, k_value in value1.items():
            featureSet[k] = k_value
        value1 = itemNameCheckLose('+1:sentence.itemlose.', sentence1)
        for k, k_value in value1.items():
            featureSet[k] = k_value
    else:
        sentSign2 = False
        featureSet['EOD'] =True
    if i < (len(doc[2])-2):
        if sentSign and (not sentSign1) and (not sentSign2):
            tag = recommendTag(sentence)
            featureSet.update({'sentence.windowmid':tag})
        if sentSign and (not sentSign2) and (not itemShow(doc[2][i+2])):
            tag = recommendTag(sentence)
            featureSet.update({'sentence.windowfront':tag})
        sentence = doc[2][i+2]
        # featureSet.update({'+2:sentence.charlen': len(sentence.strip())})
    if i > 2:
        if sentSign and (not sentSign1) and (not itemShow(doc[2][i-2])):
            tag = recommendTag(sentence)
            featureSet.update({'sentence.windowback':tag})
        sentence = doc[2][i-2]
        # featureSet.update({'-2:sentence.charlen': len(sentence.strip())})
    return(featureSet)


def doc2features(doc):
    acc_state = item_acc_vector(doc)
    return [sentence2features(doc, acc_state, i) for i in range(len(doc[2]))]


def sent2labels(doc):
    return([doc[1][label_index] for label_index in range(len(doc[1]))])

def doc2tokens(doc):
    return([ doc[index] for index in range(len(doc))])

def metricCal(y_pred, y_true, target):
    pre_list = set([index for index, value in enumerate(y_pred) if value in target.values()])
    true_list = set([index for index, value in enumerate(y_true) if value in target.values()])
    precision = len(pre_list.intersection(true_list))/len(pre_list)
    recall = len(pre_list.intersection(true_list))/len(true_list)
    if precision + recall ==0:
        f1 = 0
    else:
        f1 = 2*precision*recall/(precision + recall)
    return(precision,recall,f1)


def trec_val_cross(y_pred, y_true):
    alltasks = {'0':{'beg':'B1','end':'I1'},'1':{'beg':'B1A','end':'I1A'},'2':{'beg':'B1B','end':'I1B'},
                '3':{'beg':'B2','end':'I2'},'4':{'beg':'B3','end':'I3'},'5':{'beg':'B4','end':'I4'},
                '6':{'beg':'B5','end':'I5'},'7':{'beg':'B6','end':'I6'},'8':{'beg':'B7','end':'I7'},
                '9':{'beg':'B7A','end':'I7A'},'10':{'beg':'B8','end':'I8'},'11':{'beg':'B9','end':'I9'},
                '12':{'beg':'B9A','end':'I9A'},'13':{'beg':'B9B','end':'I9B'},'14':{'beg':'B10','end':'I10'},
                '15':{'beg':'B11','end':'I11'},'16':{'beg':'B12','end':'I12'},'17':{'beg':'B13','end':'I13'},
                '18':{'beg':'B14','end':'I14'},'19':{'beg':'B15','end':'I15'}}
    allitem = {'0':{'pre':-1,'recall':-1,'f1':-1}, '1':{'pre':-1,'recall':-1,'f1':-1}, '2':{'pre':-1,'recall':-1,'f1':-1},
               '3':{'pre':-1,'recall':-1,'f1':-1}, '4':{'pre':-1,'recall':-1,'f1':-1}, '5':{'pre':-1,'recall':-1,'f1':-1},
               '6':{'pre':-1,'recall':-1,'f1':-1}, '7':{'pre':-1,'recall':-1,'f1':-1}, '8':{'pre':-1,'recall':-1,'f1':-1},
               '9':{'pre':-1,'recall':-1,'f1':-1}, '10':{'pre':-1,'recall':-1,'f1':-1}, '11':{'pre':-1,'recall':-1,'f1':-1},
               '12':{'pre':-1,'recall':-1,'f1':-1}, '13':{'pre':-1,'recall':-1,'f1':-1}, '14':{'pre':-1,'recall':-1,'f1':-1},
               '15':{'pre':-1,'recall':-1,'f1':-1}, '16':{'pre':-1,'recall':-1,'f1':-1}, '17':{'pre':-1,'recall':-1,'f1':-1},
               '18':{'pre':-1,'recall':-1,'f1':-1}, '19':{'pre':-1,'recall':-1,'f1':-1}}
    for key, value in alltasks.items():
        pre_index = True if value['beg'] in y_pred or value['end'] in y_pred else False
        true_index = True if value['beg'] in y_true or value['end'] in y_true else False
        
        
        if not true_index:
            # if the item does not exist
            if pre_index:
                # if model labeled non-exist items, set performance to 0.0
                allitem[key]['pre'] = 0.0
                allitem[key]['recall'] = 0.0
                allitem[key]['f1'] = 0.0            
            else:
                # if model did not label non-exist items, set performance to 1.0
                allitem[key]['pre'] = 1.0
                allitem[key]['recall'] = 1.0
                allitem[key]['f1'] = 1.0
        else:
            # if the item does exist in the document
            if pre_index:
                # if the model do make prediction
                allitem[key]['pre'], allitem[key]['recall'], allitem[key]['f1'] = metricCal(y_pred, y_true, value)
            else:
                # if the model does not make prediction on this item, set performance to 0.0
                allitem[key]['pre'] = 0
                allitem[key]['recall'] = 0
                allitem[key]['f1'] = 0
    return(allitem)

def trainCRF(X_train, y_train, filename, c1, c2, max_iterations = 500, minfreq = 50, possible_transitions = True):
    trainer = pycrfsuite.Trainer(verbose=True)
    trainer.select(algorithm='lbfgs')
    trainer.set_params({
        'c1': c1,   # coefficient for L1 penalty
        'c2': c2,  # coefficient for L2 penalty
        'max_iterations': max_iterations,  # stop earlier
        'feature.minfreq': minfreq,
        # include transitions that are possible, but not observed
        'feature.possible_transitions': possible_transitions
    })
    trainer.params()
    print("    Loading training data to trainer...", flush = True)
    for xseq, yseq in zip(X_train, y_train):
        try:
            trainer.append(xseq, yseq)
        except:
            print(" Error encountered when loading data to crf trainer.")
            print(xseq)
            print(yseq)
    print("    Data loading completed...", flush = True)
    print("    Model training started...", flush = True)
    trainer.train('./crfsuite/'+filename+'.crfsuite')
    print('Finished Training')
    return(trainer)



# original in [ana] ana/edgar_10kseg/sample_2021annotation.ipynb
def pred_10k(lines, tagger):
    nrow = len(lines)
    # print("    There are %d lines (before removing empty lines)" % nrow, flush = True)
    
    # remove empty lines and keep the mapping
    seqkeep = 0 # sequence line no for keeped lines
    linekeep = []  # keeped lines
    seqmap = dict()  #map from keeped line no. to original line no.
    for i, aline in enumerate(lines):
        aline = aline.strip()
        if len(aline) > 0:
            linekeep.append(aline)
            seqmap[seqkeep] = i
            seqkeep += 1
            
    tt = []
    tt.append('0')
    tt.append('0')
    tt.append(linekeep)        
    pred = tagger.tag(doc2features(tt))
    
    # map the predicted tags back to original line sequence
    pred_ext = ['X'] * len(lines)
    for i, tag in enumerate(pred):
        i2 = seqmap[i]
        pred_ext[i2] = tag
    
    ## use simple "carry to next blank" rule
    #last_tag = 'O'
    #for i, tag in enumerate(pred_ext):
    #    if tag == 'X':
    #        pred_ext[i] = last_tag
    #    last_tag = tag
    
    
    # fix the problem related to B tags
    last_tag = 'O'
    N = len(pred_ext)
    for i, tag in enumerate(pred_ext):
        if tag == 'X':
            if last_tag[0] == 'B':
                # find next predicted tag
                if i+1 < N:
                    next_ptag = pred_ext[i+1]
                    step = 2
                    while next_ptag == 'X' and i + step < N:
                        next_ptag = pred_ext[i+step]
                        step += 1
                    if next_ptag == 'X':
                        # in case we reach the end of the list
                        next_ptag = 'O'
                    elif next_ptag[0] == 'B':
                        # will not carry future B tags
                        # next_ptag = last_tag
                        next_ptag = "I" + last_tag[1:]
                else:
                    next_ptag = 'O'
                pred_ext[i] = next_ptag
            else:
                pred_ext[i] = last_tag
        last_tag = tag
    return pred_ext