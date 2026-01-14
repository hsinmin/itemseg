# Tool library for CRF model
# Hsin-Min Lu (luim@ntu.edu.tw) 2023/12/22
# For non-commercial use only

import re
import pycrfsuite
import pycrfsuite as pysuite
from nltk import tokenize
import numpy as np
import sys


# Target labels representing beginning tags for each SEC 10-K item section
# B = Beginning tag, followed by item number (e.g., B1 = Item 1, B1A = Item 1A)
target = ['B1','B1A','B1B','B2','B3','B4','B5','B6','B7','B7A','B8','B9','B9A','B9B',
              'B10','B11','B12','B13','B14','B15']
# Core item indices considered most important for analysis
core = [0, 1, 4, 6, 8, 9]


def recommendTag(text):
    """
    Recommends a tag (label) for a line of text based on ITEM pattern matching.

    Args:
        text: Input text string to analyze

    Returns:
        str: Tag in format 'B' + item number (e.g., 'B1', 'B1A') if ITEM pattern found,
             otherwise 'O' (Outside/Other)
    """
    preN = text[0:15]  # Extract first 15 characters
    # Search for ITEM patterns like "ITEM 1", "ITEM 1A", "ITEM 1.", etc.
    criteria = re.findall(r'ITEM[s]?\s*[I0-9]*[(]?[A-Za-z]?[)]?[.]?',preN,re.IGNORECASE)
    if len(criteria) == 1:
        # Extract the item number and letter (e.g., "1A", "2", "7A")
        sign = re.findall(r'[0-9]+[()]?[A-Za-z]?[)]?',criteria[0])
        if len(sign) != 0:
            return('B'+sign[0])  # Return beginning tag
        else:
            return('O')  # No item number found
    else:
        return('O')  # No unique ITEM pattern found

def itemShow(text):
    """
    Checks if the text starts with the word "ITEM".

    Args:
        text: Input text string to check

    Returns:
        bool: True if text begins with "ITEM" (case-insensitive), False otherwise
    """
    # Search for "item" at the beginning of the text (first 15 chars)
    criteria = re.findall(r'^\s*item',text[0:15],re.IGNORECASE)
    if len(criteria) == 1 :
        return(True)
    else:
        return(False)

def checkSpecialContent(text):
    """
    Detects special boilerplate phrases common in SEC filings.

    Args:
        text: Input text string to check (uses first 300 characters)

    Returns:
        bool: True if text contains "incorporated by reference" or "set forth page",
              False otherwise
    """
    # Check for common SEC filing reference phrases
    criteria1 = re.findall(r'incorporated[\s\w]*by[\s\w]*reference',text[0:300],re.IGNORECASE)
    criteria2 = re.findall(r'set[\s\w]*forth[\s\w]*page',text[0:300],re.IGNORECASE)
    if len(criteria1) > 0 or len(criteria2) > 0:
        return(True)
    else:
        return(False)

def checkSignature(text):
    """
    Checks if the text contains the word "signature" or "signatures".

    Args:
        text: Input text string to check

    Returns:
        bool: True if "signature" or "signatures" found (case-insensitive), False otherwise
    """
    criteria = re.findall(r'signature[s]',text,re.IGNORECASE)
    if len(criteria) != 0:
        return(True)
    else:
        return(False)

def unigram(text, n = 15, lower = True):
    """
    Extracts unigrams (individual words) from text using NLTK tokenization.

    Args:
        text: Input text string to tokenize
        n: Maximum number of tokens to return (default: 15)
        lower: Whether to convert text to lowercase (default: True)

    Returns:
        list: List of up to n word tokens
    """
    # Alternative simpler approaches (commented out):
    # firstN = text.split()
    # filtered_words = [word for word in firstN][0:8]
    if lower == True:
        text = text.strip().lower()
    else:
        text = text.strip()
    filtered_words = tokenize.word_tokenize(text)[0:n]
    return(filtered_words)

def bigram(ugram, lower = True):
    """
    Generates bigrams (consecutive word pairs) from a list of unigrams.

    Args:
        ugram: List of unigram tokens
        lower: Whether to convert tokens to lowercase (default: True)

    Returns:
        list: List of bigrams in format "word1_word2"
    """
    bigram = []
    for i, atok in enumerate(ugram):
        if i == 0:
            continue  # Skip first token (no previous token to pair with)
        if lower == True:
            bigram.append(ugram[i-1].lower() + "_" + ugram[i].lower())
        else:
            bigram.append(ugram[i-1] + "_" + ugram[i])
    return bigram

def itemNameCheck(prefix, text):
    """
    Checks for specific SEC 10-K item sections by matching full item descriptions.
    Uses content-based patterns (e.g., "business", "risk factor") for robust identification.

    Args:
        prefix: String prefix to add to matched item names in output dictionary
        text: Input text to check for item patterns

    Returns:
        dict: Dictionary with keys like "{prefix}item 1", "{prefix}item 1A", etc.
              Only matched items are included (value=True)
    """
    # Pattern dictionary mapping item names to their full description regex patterns
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
            value[prefix+item] = True  # Only add matched items to output
        #else:
        #    value[prefix+item] = False  # Commented: don't add non-matches
    return(value)


def itemNameCheckLose(prefix, text):
    """
    Checks for SEC 10-K item sections using loose number-only matching.
    Less strict than itemNameCheck - only matches "ITEM" followed by number/letter.

    Args:
        prefix: String prefix to add to matched item names in output dictionary
        text: Input text to check for item patterns

    Returns:
        dict: Dictionary with keys like "{prefix}item 1", "{prefix}item 1A", etc.
              Only matched items are included (value=True)
    """
    # Loose patterns: just "ITEM" + number/letter (e.g., "ITEM 1", "ITEM 7A")
    alltasks = {'item 1': r'^\s*item[\s\\.-]+1(?!\.A)\b',  # Negative lookahead to avoid matching "1A"
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


def item_acc_vector(doc):
    """
    Generates normalized accumulated item counts for each sentence in the document.
    Provides positional information about where each item type appears.

    Args:
        doc: Document tuple where doc[2] contains list of sentences

    Returns:
        list: List of dictionaries (one per sentence) containing normalized accumulated
              counts for each item type. Values range from 0.0 to 1.0, normalized by
              the total count of that item type in the document.
    """
    acc_state = []  # Will store accumulated state for each sentence
    # Initialize counter for all possible item types
    this_state = {'item 1': 0, 'item 1A': 0, 'item 1B': 0,
                  'item 2': 0, 'item 3': 0, 'item 4': 0,
                  'item 5': 0, 'item 6': 0, 'item 7': 0,
                  'item 7A': 0, 'item 8': 0, 'item 9': 0, 'item 9A': 0,
                  'item 9B': 0, 'item 10': 0, 'item 11': 0,
                  'item 12': 0, 'item 13': 0, 'item 14': 0,
                  'item 15': 0}

    # First pass: accumulate raw counts for each sentence position
    for i, sentence in enumerate(doc[2]):
        tmp1 = itemNameCheckLose("", sentence)  # Check for item patterns
        #print(i, tmp1)
        for aitem in tmp1:
            this_state[aitem] += 1  # Increment count for found items
        acc_state.append(this_state.copy())  # Save snapshot of accumulated counts

    # Alternative normalization approach (commented out):
    # max_nitem = np.median(np.array(list(this_state.values())))
    # if max_nitem <= 0.0:
    #     max_nitem = 1.0

    # Second pass: normalize accumulated counts by total count per item type
    for aitem in acc_state:
        for key, value in aitem.items():
            max_nitem = this_state[key]  # Total count for this item type
            if max_nitem <= 0.0:
                max_nitem = 1.0  # Avoid division by zero
            aitem[key] = value / max_nitem  # Normalize to [0.0, 1.0]

    return acc_state



def preprocessing(text):
    """
    Cleans text by removing punctuation and normalizing whitespace.

    Args:
        text: Input text string to preprocess

    Returns:
        str: Cleaned text with punctuation removed, whitespace normalized, and converted to lowercase
    """
    # Remove common punctuation marks
    temp = re.sub(r'[,.&()\/\\?:!";\'\"$%]', '', text)
    # Normalize whitespace and convert to lowercase
    return(' '.join(temp.split()).lower())

def sentence2features(doc, acc_state, i):
    """
    Extracts comprehensive feature set for a single sentence for CRF training/prediction.
    Includes current sentence features, previous/next sentence context, and positional info.

    Args:
        doc: Document tuple where doc[2] contains list of sentences
        acc_state: Accumulated item count state from item_acc_vector()
        i: Index of current sentence in doc[2]

    Returns:
        dict: Feature dictionary with keys like 'sentence.unigram.word', '-1:sentence.bigram.word1_word2',
              '+1:sentence.itemlose.item 7A', 'sentence.forwardPosition', etc.
    """
    sentence = doc[2][i].strip()
    filtedSent = preprocessing(sentence)
    #unigrmList = unigram(filtedSent)  # Alternative: preprocessed unigrams
    pureUnigram = unigram(sentence)  # Unigrams from original text
    pureBigram = bigram(pureUnigram)
    sentSign = itemShow(sentence)  # Does sentence start with "ITEM"?
    senthead = sentence.strip()
    sentheadupper = False
    if len(senthead) > 0:
        sentheadupper = senthead[0].isupper()  # First character is uppercase?

    # Positional features
    forwpos1 = i/len(doc[2]) if len(doc[2]) != 0 else 0.  # Relative position from start [0.0, 1.0]
    backpos1 = (len(doc[2])-i)/len(doc[2]) if len(doc[2]) != 0 else 0.  # Relative position from end
    headthreshold = 50.0
    headpos = min(i, headthreshold) / headthreshold * 1.0  # Position from start, capped at 50
    wordlen = len(tokenize.word_tokenize(sentence))
    wordlenmax = 20.0
    wordlen = min(wordlen, wordlenmax) / wordlenmax * 1.0  # Normalized word length [0.0, 1.0]

    # Initialize feature set with basic sentence features
    featureSet = {
        'sentence.isupperWithItem' : True if sentSign and sentheadupper else False,
        'sentence.forwardPosition' : forwpos1,
        'sentence.backwardPosition' : backpos1,
        'sentence.headpos': headpos,
        #'sentence.headposraw': i,  # Alternative: raw position index
        'sentence.wordlen': wordlen,
        #'sentence.unorder': pureUnigram,  # Alternative: unigram list
        #'sentence.bigrams': pureBigram,  # Alternative: bigram list
        'sentence.special_content':checkSpecialContent(sentence)
    }
    
    # Add unigram features (bag-of-words representation)
    for atoken in pureUnigram:
        featureSet['sentence.unigram.' + atoken] = 1.

    # Add bigram features
    for atoken in pureBigram:
        featureSet['sentence.bigram.' + atoken] = 1.

    # Alternative positional unigram features (commented out):
    #for j in range(len(unigrmList)):
    #    featureSet['unigrm'+str(j)] = unigrmList[j]

    # Add content-based item matching features (e.g., "ITEM 1. Business")
    value = itemNameCheck('sentence.', sentence)
    for k, k_value in value.items():
        featureSet[k] = k_value

    # Add loose number-based item matching features (e.g., "ITEM 7A")
    value = itemNameCheckLose('sentence.itemlose.', sentence)
    for k, k_value in value.items():
        featureSet[k] = k_value

    # Add accumulated item count features (position within item sequence)
    for key, value in acc_state[i].items():
        featureSet['accitem_' + key] = value
        
    # Features from PREVIOUS sentence (i-1) - provides context
    if i > 0:
        sentence1 = doc[2][i-1]
        # filtedSent1 = preprocessing(sentence1)  # Alternative: preprocessed text
        # unigrmList1 = unigram(filtedSent1)
        pureUnigram1 = unigram(sentence1)
        pureBigram1 = bigram(pureUnigram1)
        sentSign1 = itemShow(sentence1)

        # Add previous sentence features with '-1:' prefix
        featureSet.update({
            '-1:sentence.isupperWithItem' : True if sentSign1 and pureUnigram1[0].isupper() else False,
            #'-1:sentence.forwardPosition' : i/len(doc[2]) if len(doc[2]) != 0 else 0,  # Commented: positional features
            #'-1:sentence.backwardPosition' : (len(doc[2])-i)/len(doc[2]) if len(doc[2]) != 0 else 0,
            #'-1:sentence.headpos': min(i, headthreshold) / headthreshold * 1.0,
            #'-1:sentence.charlen': len(sentence.strip()),
            #'-1:sentence.unorder': pureUnigram1,  # Commented: unigram list
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
        # First sentence in document - no previous context
        sentSign1 = False
        featureSet['BOD']=True  # Beginning of document marker
    # Features from NEXT sentence (i+1) - provides forward context
    if i < (len(doc[2])-1):
        sentence1 = doc[2][i+1]
        # filtedSent1 = preprocessing(sentence1)  # Alternative: preprocessed text
        unigram1 = unigram(sentence1)
        sentSign2 = itemShow(sentence1)
        pureUnigram1 = unigram(sentence1)
        pureBigram1 = bigram(pureUnigram1)

        # Add next sentence features with '+1:' prefix
        featureSet.update({
            '+1:sentence.isupperWithItem' : True if sentSign2 and pureUnigram1[0].isupper() else False,
            #'+1:sentence.forwardPosition' : i/len(doc[2]) if len(doc[2]) != 0 else 0,  # Commented: positional features
            #'+1:sentence.backwardPosition' : (len(doc[2])-i)/len(doc[2]) if len(doc[2]) != 0 else 0,
            #'+1:sentence.headpos': min(i, headthreshold) / headthreshold * 1.0,
            #'+1:sentence.charlen': len(sentence1.strip()),
            #'+1:sentence.unorder': pureUnigram1,  # Commented: unigram list
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
        # Last sentence in document - no next context
        sentSign2 = False
        featureSet['EOD'] =True  # End of document marker
    # Window-based features - detect isolated ITEM headers
    if i < (len(doc[2])-2):
        # Current sentence has "ITEM", but prev and next don't: likely section header
        if sentSign and (not sentSign1) and (not sentSign2):
            tag = recommendTag(sentence)
            featureSet.update({'sentence.windowmid':tag})
        # Current sentence has "ITEM", but next two don't: likely section header
        if sentSign and (not sentSign2) and (not itemShow(doc[2][i+2])):
            tag = recommendTag(sentence)
            featureSet.update({'sentence.windowfront':tag})
        sentence = doc[2][i+2]  # Sentence at i+2
        # featureSet.update({'+2:sentence.charlen': len(sentence.strip())})  # Commented: char length feature

    if i > 2:
        # Current sentence has "ITEM", but prev two don't: likely section header
        if sentSign and (not sentSign1) and (not itemShow(doc[2][i-2])):
            tag = recommendTag(sentence)
            featureSet.update({'sentence.windowback':tag})
        sentence = doc[2][i-2]  # Sentence at i-2
        # featureSet.update({'-2:sentence.charlen': len(sentence.strip())})  # Commented: char length feature

    return(featureSet)


def doc2features(doc):
    """
    Converts an entire document to feature vectors for CRF training/prediction.

    Args:
        doc: Document tuple where doc[2] contains list of sentences

    Returns:
        list: List of feature dictionaries, one for each sentence
    """
    acc_state = item_acc_vector(doc)  # Calculate accumulated item counts
    return [sentence2features(doc, acc_state, i) for i in range(len(doc[2]))]


def sent2labels(doc):
    """
    Extracts label sequence from a document.

    Args:
        doc: Document tuple where doc[1] contains labels

    Returns:
        list: List of labels for each sentence
    """
    return([doc[1][label_index] for label_index in range(len(doc[1]))])

def doc2tokens(doc):
    """
    Extracts tokens/sentences from a document.

    Args:
        doc: Document tuple

    Returns:
        list: List of document elements
    """
    return([ doc[index] for index in range(len(doc))])

def metricCal(y_pred, y_true, target):
    """
    Calculates precision, recall, and F1 score for predicted labels.

    Args:
        y_pred: List of predicted labels
        y_true: List of true/gold labels
        target: Dictionary with 'beg' and 'end' keys containing target label values

    Returns:
        tuple: (precision, recall, f1) scores
    """
    # Find indices where predicted/true labels match target values
    pre_list = set([index for index, value in enumerate(y_pred) if value in target.values()])
    true_list = set([index for index, value in enumerate(y_true) if value in target.values()])

    # Calculate precision: (true positives) / (predicted positives)
    precision = len(pre_list.intersection(true_list))/len(pre_list)
    # Calculate recall: (true positives) / (actual positives)
    recall = len(pre_list.intersection(true_list))/len(true_list)

    # Calculate F1 score (harmonic mean of precision and recall)
    if precision + recall ==0:
        f1 = 0
    else:
        f1 = 2*precision*recall/(precision + recall)
    return(precision,recall,f1)


def trec_val_cross(y_pred, y_true):
    """
    Calculates per-item performance metrics across all 10-K items.
    Uses IOB tagging scheme: B = Beginning, I = Inside.

    Args:
        y_pred: List of predicted labels
        y_true: List of true/gold labels

    Returns:
        dict: Dictionary mapping item IDs ('0'-'19') to their precision, recall, and F1 scores
    """
    # Define all 20 possible 10-K items with their BIO tags
    # B = Beginning of item, I = Inside item
    alltasks = {'0':{'beg':'B1','end':'I1'},'1':{'beg':'B1A','end':'I1A'},'2':{'beg':'B1B','end':'I1B'},
                '3':{'beg':'B2','end':'I2'},'4':{'beg':'B3','end':'I3'},'5':{'beg':'B4','end':'I4'},
                '6':{'beg':'B5','end':'I5'},'7':{'beg':'B6','end':'I6'},'8':{'beg':'B7','end':'I7'},
                '9':{'beg':'B7A','end':'I7A'},'10':{'beg':'B8','end':'I8'},'11':{'beg':'B9','end':'I9'},
                '12':{'beg':'B9A','end':'I9A'},'13':{'beg':'B9B','end':'I9B'},'14':{'beg':'B10','end':'I10'},
                '15':{'beg':'B11','end':'I11'},'16':{'beg':'B12','end':'I12'},'17':{'beg':'B13','end':'I13'},
                '18':{'beg':'B14','end':'I14'},'19':{'beg':'B15','end':'I15'}}

    # Initialize performance metrics for all items
    allitem = {'0':{'pre':-1,'recall':-1,'f1':-1}, '1':{'pre':-1,'recall':-1,'f1':-1}, '2':{'pre':-1,'recall':-1,'f1':-1},
               '3':{'pre':-1,'recall':-1,'f1':-1}, '4':{'pre':-1,'recall':-1,'f1':-1}, '5':{'pre':-1,'recall':-1,'f1':-1},
               '6':{'pre':-1,'recall':-1,'f1':-1}, '7':{'pre':-1,'recall':-1,'f1':-1}, '8':{'pre':-1,'recall':-1,'f1':-1},
               '9':{'pre':-1,'recall':-1,'f1':-1}, '10':{'pre':-1,'recall':-1,'f1':-1}, '11':{'pre':-1,'recall':-1,'f1':-1},
               '12':{'pre':-1,'recall':-1,'f1':-1}, '13':{'pre':-1,'recall':-1,'f1':-1}, '14':{'pre':-1,'recall':-1,'f1':-1},
               '15':{'pre':-1,'recall':-1,'f1':-1}, '16':{'pre':-1,'recall':-1,'f1':-1}, '17':{'pre':-1,'recall':-1,'f1':-1},
               '18':{'pre':-1,'recall':-1,'f1':-1}, '19':{'pre':-1,'recall':-1,'f1':-1}}

    # Evaluate each item type
    for key, value in alltasks.items():
        # Check if item appears in predictions or ground truth
        pre_index = True if value['beg'] in y_pred or value['end'] in y_pred else False
        true_index = True if value['beg'] in y_true or value['end'] in y_true else False


        if not true_index:
            # Item does not exist in ground truth
            if pre_index:
                # Model incorrectly predicted non-existent item: false positive
                allitem[key]['pre'] = 0.0
                allitem[key]['recall'] = 0.0
                allitem[key]['f1'] = 0.0
            else:
                # Model correctly did not predict non-existent item: true negative
                allitem[key]['pre'] = 1.0
                allitem[key]['recall'] = 1.0
                allitem[key]['f1'] = 1.0
        else:
            # Item exists in ground truth
            if pre_index:
                # Model made predictions for this item: calculate metrics
                allitem[key]['pre'], allitem[key]['recall'], allitem[key]['f1'] = metricCal(y_pred, y_true, value)
            else:
                # Model failed to predict existing item: false negative
                allitem[key]['pre'] = 0
                allitem[key]['recall'] = 0
                allitem[key]['f1'] = 0
    return(allitem)

def trainCRF(X_train, y_train, filename, c1, c2, max_iterations = 500, minfreq = 50, possible_transitions = True):
    """
    Trains a CRF model using pycrfsuite with L-BFGS optimization.

    Args:
        X_train: List of feature sequences (each sequence is a list of feature dicts)
        y_train: List of label sequences (each sequence is a list of labels)
        filename: Name for saved model file (saved to './crfsuite/{filename}.crfsuite')
        c1: L1 regularization coefficient (higher = more regularization)
        c2: L2 regularization coefficient (higher = more regularization)
        max_iterations: Maximum training iterations (default: 500)
        minfreq: Minimum feature frequency to include (default: 50)
        possible_transitions: Include unobserved but possible transitions (default: True)

    Returns:
        trainer: Trained pycrfsuite.Trainer object
    """
    trainer = pycrfsuite.Trainer(verbose=True)
    trainer.select(algorithm='lbfgs')  # L-BFGS optimization algorithm
    trainer.set_params({
        'c1': c1,   # L1 regularization coefficient (for feature selection)
        'c2': c2,  # L2 regularization coefficient (for weight smoothing)
        'max_iterations': max_iterations,  # Maximum optimization iterations
        'feature.minfreq': minfreq,  # Minimum frequency for feature inclusion
        # Include transitions that are possible but not observed in training
        'feature.possible_transitions': possible_transitions
    })
    trainer.params()
    print("    Loading training data to trainer...", flush = True)
    # Load all training sequences
    for xseq, yseq in zip(X_train, y_train):
        try:
            trainer.append(xseq, yseq)
        except:
            print(" Error encountered when loading data to crf trainer.")
            print(xseq)
            print(yseq)
    print("    Data loading completed...", flush = True)
    print("    Model training started...", flush = True)
    # Train and save model
    trainer.train('./crfsuite/'+filename+'.crfsuite')
    print('Finished Training')
    return(trainer)



def pred_10k(lines, tagger):
    """
    Predicts 10-K item labels for a document using a trained CRF model.
    Handles empty lines and applies post-processing rules.

    Original implementation from: [ana] ana/edgar_10kseg/sample_2021annotation.ipynb

    Args:
        lines: List of text lines from the document
        tagger: Trained pycrfsuite.Tagger object

    Returns:
        list: Predicted labels for each line (including 'X' for empty lines)
    """
    nrow = len(lines)
    # print("    There are %d lines (before removing empty lines)" % % nrow, flush = True)

    # Remove empty lines and maintain mapping to original line numbers
    seqkeep = 0  # Sequence number for non-empty lines
    linekeep = []  # List of non-empty lines
    seqmap = dict()  # Map from kept line index to original line index
    for i, aline in enumerate(lines):
        aline = aline.strip()
        if len(aline) > 0:
            linekeep.append(aline)
            seqmap[seqkeep] = i
            seqkeep += 1

    # Create document structure for feature extraction
    tt = []
    tt.append('0')  # Document ID placeholder
    tt.append('0')  # Labels placeholder
    tt.append(linekeep)  # Non-empty lines
    pred = tagger.tag(doc2features(tt))  # Generate predictions
    
    # Map predicted tags back to original line sequence
    # Empty lines are marked with 'X' (placeholder)
    pred_ext = ['X'] * len(lines)
    for i, tag in enumerate(pred):
        i2 = seqmap[i]  # Get original line index
        pred_ext[i2] = tag

    # Alternative simple approach (commented out):
    # Carry forward last tag to blank lines
    #last_tag = 'O'
    #for i, tag in enumerate(pred_ext):
    #    if tag == 'X':
    #        pred_ext[i] = last_tag
    #    last_tag = tag

    # Post-processing: Handle empty lines ('X') with context-aware rules
    # Propagate tags intelligently based on BIO tagging scheme
    last_tag = 'O'
    N = len(pred_ext)
    for i, tag in enumerate(pred_ext):
        if tag == 'X':  # Empty line needs tag assignment
            if last_tag[0] == 'B':  # Previous line was Beginning of item
                # Look ahead to find next non-empty predicted tag
                if i+1 < N:
                    next_ptag = pred_ext[i+1]
                    step = 2
                    while next_ptag == 'X' and i + step < N:
                        next_ptag = pred_ext[i+step]
                        step += 1
                    if next_ptag == 'X':
                        # Reached end of list: use Outside tag
                        next_ptag = 'O'
                    elif next_ptag[0] == 'B':
                        # Next is new item Beginning: convert current to Inside
                        # Don't carry forward future B tags
                        # next_ptag = last_tag  # Alternative: keep B tag
                        next_ptag = "I" + last_tag[1:]  # Convert to Inside tag
                else:
                    next_ptag = 'O'
                pred_ext[i] = next_ptag
            else:
                # Previous wasn't Beginning: just carry forward previous tag
                pred_ext[i] = last_tag
        last_tag = tag
    return pred_ext