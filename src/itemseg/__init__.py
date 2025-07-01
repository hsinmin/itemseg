#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import platform
import sys, re, os, collections
import re
import time
import os, random
import requests
import pandas as pd
from inscriptis import get_text
import html
# import unicodedata
# import nltk
import pycrfsuite
from nltk import tokenize
import glob
import json
import gensim
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import numpy as np
from itemseg import lib_10kq_seg_v1 as lib10kq
from itemseg import crf_feature_lib_v8 as crf_feature
from itemseg import gpt4itemSeg
from argparse import ArgumentParser
import urllib.parse
import pathlib
from sentence_transformers import SentenceTransformer
html2txt_type = "inscriptis"

sentence_bert_model = SentenceTransformer('stsb-mpnet-base-v2')
# Bi-LSTM
class BiLSTM2(nn.Module):

    # def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
    def __init__(self, input_dim, tag_to_ix, hidden_dim, device, batch_size=4, num_layers=1, dropout=0.5):  # tag_to_ix就是label_mapping
        '''
        parameters:
            tag_to_ix: 標籤對應標號的字典
            hidden dimension: BILSM 隱藏層的神經元數量
        '''
        super(BiLSTM2, self).__init__()
        self.hidden_dim = hidden_dim
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix) # the output size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device
        # self.embedding_dim = embedding_dim
        # self.vocab_size = vocab_size
        # self.adjust_input_dim = nn.Linear(input_dim, 803)  # 將 768 調整為 803
        ## dimension: batch_size, num_line, dim        
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2,  # we uses input_dim instead of embedding_dim
                            num_layers = num_layers, bidirectional = True, 
                            batch_first = True, dropout = dropout)


        # 將 LSTM 的輸出映射到標籤空間
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.dropout = nn.Dropout(dropout)
    def init_hidden(self, batch_size=0):
        # (num_layers, nums_directions, minibatch_size, hidden_dim, device)
        # 實際上初始化的 h0, c0
        if batch_size == 0:
            # use default batch size if not passed to us; 
            batch_size = self.batch_size
        return (torch.randn(2 * self.num_layers, batch_size, self.hidden_dim // 2, device=self.device),
                torch.randn(2 * self.num_layers, batch_size, self.hidden_dim // 2, device=self.device))
    
    def forward(self, sentence):
        # sentence = self.adjust_input_dim(sentence)
        # check the sentence is packed or not
        if isinstance(sentence, PackedSequence):
            batch_size = sentence.unsorted_indices.shape[0] 
        else:
            batch_size = sentence.shape[0]    
        self.hidden = self.init_hidden(batch_size=batch_size) # initial state and hidden state
        lstm_out, self.hidden = self.lstm(sentence, self.hidden) # 最後輸出結果和最後的隱藏狀態

        if isinstance(sentence, PackedSequence):
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        lstm_feats = self.hidden2tag(self.dropout(lstm_out))
        return lstm_feats

class batch_shuffler:
    def __init__(self, total_size, batch_size, shuffle=True):
        self.total_size = total_size
        self.batch_size = batch_size
        self.shuf_idxlist = list(range(self.total_size))
        if shuffle: random.shuffle(self.shuf_idxlist)
        self.startpos = 0
        self.first = True
    def __iter__(self):
        return self
    def __next__(self):
        if self.startpos+self.batch_size >= self.total_size:
            raise StopIteration
        else:            
            if self.first == False:
                self.startpos += self.batch_size
            else:
                self.first = False
            # print(self.startpos)
            return self.shuf_idxlist[self.startpos:(self.startpos+self.batch_size)]


def get_resource(dest="__home__", check_only=False, verbose=1, 
                 url0 = "http://www.im.ntu.edu.tw/~lu/data/itemseg/"):
    files = ['crf8f6_m5000c2_1f_200f06c1_0.00c2_1.00_m5000.crfsuite',
             'word2vecmodel_10kq3a_epoch_5',
             'word2vecmodel_10kq3a_epoch_5.syn1neg.npy',
             'word2vecmodel_10kq3a_epoch_5.wv.vectors.npy',
             'tag2023_v1_labelidmap.pkl',
             'tag2021_v3_labelidmap.pkl',  # for bert
             'bert_model/bert_model.pth',
             'lstm_model/h256len100lay2lr3complete_args.json',
             'lstm_model/h256len100lay2lr3complete_e020_vac97.31_vce0.08639.pth']
    
    if dest == "__home__":
        # replace with real home path
        dest = str(pathlib.Path.home()) + "/itemseg/resource/"
        
    if check_only == False:
        print(f"Download resource to {dest}")
        if not os.path.exists(dest):
            os.makedirs(dest)
            os.makedirs(os.path.join(dest, "lstm_model"))
            os.makedirs(os.path.join(dest, "bert_model"))
        # start download files
        err_count = 0
        for atarget in files:
            url = url0 + "resource/" + atarget
            outfn = os.path.join(dest, atarget)
            if verbose >= 1:
                print(f"Getting {url}")

            r = requests.get(url, allow_redirects=True)
            if r.status_code == 200:            
                open(outfn, 'wb').write(r.content)
            else:
                err_count += 1
            
        if err_count == 0: print("Resource download completed")
        

def main():
    parser = ArgumentParser()
    
    parser.add_argument("--get_resource", dest="get_resource", 
                        action="store_true",
                        help="Download resource files")
    default_resource_url = "http://www.im.ntu.edu.tw/~lu/data/itemseg/"
    parser.add_argument("--resource_url", dest="resource_url", type=str,
                        default=default_resource_url,
                        help=f"Set URL to download resource files. Default: {default_resource_url}")
    # input options
    # currently does not support local file yet
    parser.add_argument("--input", dest="input", type=str,
                        # default='',
                        # required=True,
                        help="EDGAR filing URL; e.g. https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/0000320193-23-000106.txt")
    parser.add_argument("--input_type", dest="input_type", type=str,
                         # default='auto',
                         help="[raw|html|native_text|cleaned_text] \n" 
                              "    raw: Complete submission text file. See sample file at https://www.sec.gov/Archives/edgar/data/789019/000156459020034944/0001564590-20-034944.txt\n"
                              "    html: HTML report. See sample file at https://www.sec.gov/ix?doc=/Archives/edgar/data/789019/000156459020034944/msft-10k_20200630.htm\n"
                              "native_text: text report. See sample file at https://www.sec.gov/Archives/edgar/data/789019/000103221001501099/d10k.txt\n"
                               "cleaned_text: 10-K report converted to the pure text formated with tables removed.")
    parser.add_argument("--user_agent_str", dest="user_agent_str", type=str,
                         default='N/A',
                         help="User Agent String per SEC's request. E.g. 'Sample Company Name AdminContact@<sample company domain>.com'")

    # output options
    parser.add_argument("--outputdir", dest="outputdir", type=str,
                        default="./segout01/",
                        help="model output dir")
    parser.add_argument("--outfn_prefix", dest="outfn_prefix", type=str,
                        default="AUTO",
                        help="output filename prefix (AUTO=let the script decide)")
    parser.add_argument("--outfn_type", dest="outfn_type", type=str,
                        default="csv,item1,item1a,item3,item7",
                        help="output file type; csv=line-by-line prediction and text; itemx=per item text in a single file")

    # model options
    parser.add_argument("--method", dest="method", type=str,
                        default='crf',
                        help="[crf|lstm|bert|chatgpt] Item segmentation method; crf: conditional random field; lstm: Bi-directional long short-term memory; bert: bert encoder coupled with bi-lstm; chatgpt: use openai api and line-id-based prompting.")
    parser.add_argument("--word2vec", dest="word2vec", type=str,
                        default='./resource/word2vecmodel_10kq3a_epoch_5',
                        help="File name of the word2vec model (gensim trained)")
    parser.add_argument("--lstmpath", dest="lstmpath", type=str,
                        default="./resource/lstm_model",
                        help="lstm model (path) for inference")
    parser.add_argument("--crfpath", dest="crfpath", type=str,
                        default="./resource/crf8f6_m5000c2_1f_200f06c1_0.00c2_1.00_m5000.crfsuite",
                        help="CRF model (path) for inference")
    # todo: default to 'AUTO'                        
    parser.add_argument("--labelid_map", dest="labelid_map", type=str,
                        default='AUTO',
                        help="labelid mapping file; a dictionary of two maps (for lstm and bert)")
    parser.add_argument("--verbose", dest="verbose", default = 1, type = int,
                        help="verbose level=0, 1, or 2; 0=silent, 2=many messages")
    parser.add_argument("--debug", dest="debug", 
                        action="store_true",
                        help="save in-progress files for debugging")
    # For BERT model
    # parser.add_argument("--optimizer", dest="optimizer", type=str,
    #                 default="Adam",
    #                 help="Optimizer; Adam or SGD")                         
    # parser.add_argument("--lr", dest="lr",  
    #                     type = float, default=0.0001,
    #                     help="Learning rate")
    # parser.add_argument("--weight_decay", dest="weight_decay",  
    #                 type = float, default=0.0,
    #                 help="Weight decay; default=0.0")  
    # parser.add_argument("--num_layers", dest="num_layers",  
    #                 type = int, default=2,
    #                 help="Bi-LSTM hidden dimension")   
    # parser.add_argument("--hidden_dim", dest="hidden_dim",  
    #                 type = int, default=256,
    #                 help="Bi-LSTM hidden dimension")
    parser.add_argument('--bertpath', dest='bertpath', type=str,
                        default='./resource/bert_model/bert_model.pth',
                        help="BERT model (path) for inference")
    # For chatgpt model start
    parser.add_argument('--apikey', dest='apikey', type=str,
                        default=None,
                        help='Your own openai api key for using chatgpt model.')
    # For chatgpt model end

    args = parser.parse_args()
    args.hostname = platform.node()

    # set  user-invisiable parameters
    args.optimizer = "Adam"
    args.lr = 0.0001
    args.weight_decay = 0.0
    args.num_layers = 2
    args.hidden_dim = 256

    # test dynamic html page
    # args = parser.parse_args(args=['--input', 
    #                                "https://www.sec.gov/ix?doc=/Archives/edgar/data/789019/000095017023035122/msft-20230630.htm",
    #                                '--debug'])

    # test raw file
    # args = parser.parse_args(args=['--input', 
    #                                "https://www.sec.gov/Archives/edgar/data/789019/000095017023035122/0000950170-23-035122.txt",
    #                                ])
    
    
    # test raw file (pure text)
    # https://www.sec.gov/Archives/edgar/data/789019/000103221099001375/0001032210-99-001375.txt
    # args = parser.parse_args(args=['--input', 
    #                                "https://www.sec.gov/Archives/edgar/data/789019/000103221099001375/0001032210-99-001375.txt",
    #                                ])
    
    # test crf method + raw file
    # args = parser.parse_args(args=['--input', 
    #                                "https://www.sec.gov/Archives/edgar/data/789019/000095017023035122/0000950170-23-035122.txt",
    #                                "--method", "crf"])

    # local file
    # args = parser.parse_args(args=['--input', 
    #                                "rawdata/6404287.txt",
    #                                "--method", "lstm"])
    
    
    if args.verbose >=1:
        print("itemseg: A tool for 10-K Item Segmentation")
        print("    Free to use for non-commercial purpose.")
        print("    Maintained by: Hsin-Min Lu (luim@ntu.edu.tw)")
        # todo: add project URL
        print("    Please cite our work (https://arxiv.org/abs/2502.08875) "
              "if you use this tool in your research.")

    if args.verbose >=2:
        print("Arguments:", args)
        
    if (args.input is None) and (args.get_resource == False):
        parser.error("Need either --input or --get_resource")
    
    if args.get_resource:
        get_resource()
        sys.exit(0)
    
    # now let's check input_type
    if args.input_type is None:
        parser.error("Need to specify input_type")

    legal_input_type = ['raw', 'html', 'native_text', 'cleaned_text']
    if args.input_type not in legal_input_type:
        parser.error(f"Illegal input type. Need to be one of these {legal_input_type}")

    method = args.method

    rdnseed = 52345

    resource_prefix = str(pathlib.Path.home()) + "/itemseg/"

    # crf_model_fn = "resource/crf8f6_m5000c2_1f_200f06c1_0.00c2_1.00_m5000.crfsuite"
    crf_model_fn = os.path.join(resource_prefix, args.crfpath)
    # lstm_model_fn = "resource/lstm_model"
    lstm_model_fn = os.path.join(resource_prefix, args.lstmpath)
    # word2vec_fn = "resource/word2vecmodel_10kq3a_epoch_5"
    word2vec_fn = os.path.join(resource_prefix, args.word2vec)

    # label2id_fn = "resource/tag2023_v1_labelidmap.pkl"
    # ./resource/tag2023_v1_labelidmap.pkl
    if args.labelid_map == "AUTO":
        if args.method in ["lstm"]:
            label2id_fn = os.path.join(resource_prefix, "resource/tag2023_v1_labelidmap.pkl")
        elif args.method in ['crf', 'bert']:
            label2id_fn = os.path.join(resource_prefix, "resource/tag2021_v3_labelidmap.pkl")
        else:
            print(f"Unknown method {args.method} for automatic labelid_map assignment")
            sys.exit(105)
    else:
        label2id_fn = os.path.join(resource_prefix, args.labelid_map)

    bert_model_fn = os.path.join(resource_prefix, args.bertpath)        
    
    res_files = [crf_model_fn, 
                 lstm_model_fn,
                 word2vec_fn,
                 label2id_fn]
    for ares in res_files:
        if os.path.exists(ares) == False:
            print(f"Cannot find resource file {crf_model_fn}.\n" 
                   "Did you foreget to download resource files with '--get_resource'?")
            sys.exit(300)

    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    if method == "lstm":    
        # the new model with proper tokenization
        if args.verbose >= 2:
            print(f"Loading word2vec model from {word2vec_fn}")
        word2vec_model = gensim.models.Word2Vec.load(word2vec_fn)

        myseed = rdnseed
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(myseed)
        torch.manual_seed(myseed)  # for CPU
        if torch.cuda.is_available():
            torch.cuda.manual_seed(myseed)
            torch.cuda.manual_seed_all(myseed)  # for GPU


        # label2id_fn = args.labelid_map
        if args.verbose >= 2:
            print(f"Loading labelid_map from {label2id_fn}")    
        with open(label2id_fn, 'rb') as f:
            labelid_map = pickle.load(f)

        label_mapping  = labelid_map['label2id']
        reverse_label_mapping = {v: k for k, v in label_mapping.items()}

        tmpmax = max(label_mapping.values())
        if args.verbose >= 2:
            print(f"    max id for label is {tmpmax}; going to add two more")
        START_TAG = "<START>"
        STOP_TAG = "<STOP>"
        label_mapping[START_TAG] = tmpmax + 1
        label_mapping[STOP_TAG] = tmpmax + 2    

        # Model setting; 
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if args.verbose >= 2:
            print("Using device", device)

        hyperparameters_rnn = {            
            'batch_size': 1,  # one line per batch;             
            'gamma': 0,          # L2 loss punishment
            'hidden_dim': 256, # will be overwritten
            'remove_punc': False, 
            'add_spaces_between_punc': False,
            'to_lower_case': False
        }

        if args.verbose >= 2:
            print(f"==== Reading lstm model files in {lstm_model_fn}")
        # args.inference_only = True 
        fns = glob.glob(lstm_model_fn + "/*_args.json")
        fns = sorted(fns)
        if args.verbose >= 2:
            print(f"     Using model setting in {fns[0]}")

        # updating model parameters to args
        with open(fns[0], "r") as fp:
            model_param = json.load(fp) 

        for akey in model_param:
            # skip some of the settings (command line has the priority)
            if akey not in ['model_outdir', 'outdir_wfoldid']:            
                vars(args)[akey] = model_param[akey]

        # pick the best model, 
        # currently using a simple rule (the last one)
        fns2 = glob.glob(lstm_model_fn + "/*.pth")
        fns2 = sorted(fns2)
        best_model_name = fns2[-1]
        if args.verbose >= 2:
            print(f"     Using model file {best_model_name}")
    elif method == "crf":
        #load tagger
        tagger = pycrfsuite.Tagger()
        tagger.open(crf_model_fn)
    elif method == "chatgpt":
        pass
    elif method == "bert":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if args.verbose >= 2:
            print("Using device", device)
        hyperparameters_rnn = {
            'optimizer': args.optimizer,
            'optim_hparas': {
                'lr': args.lr, 
                'weight_decay': args.weight_decay        
            },
            'hidden_dim': args.hidden_dim,
        }

        # current_dir = os.path.dirname(__file__)
        # 現有路徑
        # label2id_bert = os.path.join(current_dir, 'tag2021_v3_labelidmap.pkl') 
        with open(label2id_fn, 'rb') as f:
            labelid_map = pickle.load(f)
        label_mapping  = labelid_map['label2id']
        reverse_label_mapping = {v: k for k, v in label_mapping.items()}

        tmpmax = max(label_mapping.values())
        print(f"[BERT] max id for label is {tmpmax}; going to add two more")
        START_TAG = "<START>"
        STOP_TAG = "<STOP>"
        PADDING_TAG = "<PADDING>"
        label_mapping[START_TAG] = tmpmax + 1
        label_mapping[STOP_TAG] = tmpmax + 2
        label_mapping[PADDING_TAG] = tmpmax + 3

    else:
        print(f"Unknonwn method {method}. Stop")
        sys.exit(103)
    
    if method == "lstm":
        input_dim = len(word2vec_model.wv['a']) + 3
        if args.verbose >=2:
            print(f"   LSTM input dim = {input_dim}")
            print(f"   label_mapping = {label_mapping}")
            print(f"   tag set size = {len(label_mapping)}")
            print(f"   hidden_dim = {args.hidden_dim}")
            print(f"   device = {device}")
            print(f"   attention_method = {args.attention_method}")
            print(f"   num_layers = {args.num_layers}")
        model_lstm = lib10kq.BiLSTM_Tok(input_dim, 
                                    label_mapping, 
                                    args.hidden_dim, 
                                    device,
                                    attention_method=args.attention_method,
                                    num_layers=args.num_layers).to(device)
        model_lstm = model_lstm.float()
        
        # load model
        if device == "cpu":
            ckpt = torch.load(best_model_name, torch.device('cpu'))
        else:
            ckpt = torch.load(best_model_name)  
        model_lstm.load_state_dict(ckpt)


    # to be continued here...
    # ====== 2025/6/27 ====
    if args.input.find("http") >=0:
        src_type = "url"
    else:
        src_type = "fn"    

    # src_type = "txt"
    # src_type = "url"
    if args.verbose >= 2:
        print("Source type is", src_type)


    if src_type == "fn":
        srcfn = args.input
        with open(srcfn, "r") as fh:
            rawtext = fh.read()
    elif src_type == "url":        
        srcurl = args.input

        # do sec url translate
        if srcurl.find("sec.gov/") < 0:
            if args.verbose >= 1:
                print("Warning: this is not a sec.gov URL.")
        if srcurl.find("sec.gov/ix?doc=/") >= 0:        
            srcurl = srcurl.replace("ix?doc=/", "")
            if args.verbose >= 1:
                print("EDGAR dynamic URL detected. Apply URL translation.")
                print(f"    Accessing {srcurl} instead")

        print(f"Getting raw file from {srcurl}")

        # "Host": "www.sec.gov",
        # user_agent_str = args.user_agent_str
        if args.user_agent_str == "N/A":
            print("You need to specify user_agent_str per SEC's rule.")
            print("cf. https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data")
            sys.exit(200)

        # user_agent_str = "National Taiwan University, luim@ntu.edu.tw"
        # todo: setup cli parameter!!!!
        headers = {        
            "User-Agent": args.user_agent_str,
            "Accept-Encoding": "gzip, deflate",
            "host": "www.sec.gov"
            }

        r = requests.get(srcurl, headers=headers)
        if args.verbose >= 2:
            print(f"URL respond code = {r.status_code}")

        if args.debug:
            with open(os.path.join(args.outputdir, "rawfile.txt"), 
                      "w", encoding="utf-8") as my_file:
                my_file.write(r.text)

        if(r.text == None):            
            print(f"No response from target URL. Stop (response code = {r.status_code})")            
            sys.exit(100)
        elif len(r.text)< 50:            
            print(f"The length of filed text is too small (len(r.text)). Stop.") 
            urltype = "HTML"
            sys.exit(101)
        elif r.text.find("Your Request Originates from an Undeclared Automated Tool") >= 0:
            print(f"Error: SEC denied undeclared automated tool!")
            print(f"Header: {headers}")       
            sys.exit(102)
        else:
            rawtext = r.text
    # ---- Now we get rawtext; either from local file or url
    # The next step is to verify and preprocess based on rawtext    #         


    par1 = re.compile('(<SEC-DOCUMENT>.*?</SEC-HEADER>)(.*)', re.M | re.S)
    par1m1=par1.findall(rawtext)
    if len(par1m1) == 0:
        print("Cannot find header tags (SEC-DOCUMENT to SEC-HEADER). Assume to be user specified type.")
        if args.input_type == "raw":
            print(f"User specified input_type={args.input_type} but the header does not exist. Stop")
            sys.exit(191)
        # urltype = "HTML"         
        urltype = args.input_type
    else:
        sec_header = par1m1[0][0]
        html1 = par1m1[0][1]
        urltype = "raw"
        if args.input_type != "raw":
            print(f"User specified input_type={args.input_type} but header exists. Stop")
            sys.exit(190)
      
    if args.verbose >= 1:
        print(f"(based on header tag) URL Type = {urltype} "
               "(raw=EDGAR Complete submission text file; html=10-K in HTML format;"
               " native_text=10-K in its original text format;"
               " cleaned_text=10-K in text format with tables removed )")

    # triangulate with user specified type
    # arg.input_type need to be [raw|html|native_text|cleaned_text
    #                                    input_type
    #   urltype == raw  (with header)    raw (ok)       html|native_text|cleaned_text (not ok)
    #   urltype == HTML (no header)      raw (not ok)   html|native_text|cleaned_text (ok)
    #    


    # prase sec_header
    if urltype == "raw":
        header_info = lib10kq.parse_edgar_header(sec_header)
        
        if args.verbose >= 1:
            print(f"Company Name = {header_info['cname']}")
            print(f"File type = {header_info['ftype']}")
            print(f"Confirmed period of report = {header_info['cpr']}")
            print(f"Industry: {header_info['sic_desc']} - {header_info['sic_code']}")

    if urltype == "raw":
        #now, split by document
        par2 = re.compile('(<DOCUMENT>.*?</DOCUMENT>)', re.M | re.S)
        par2m1= par2.findall(html1)
        get_target = 0

        if args.verbose >= 2:
            print("# of document component:", len(par2m1))

        for adoc in par2m1:
            par3=re.compile('<TYPE>(\S+)')
            par3m1 = par3.findall(adoc)
            doc_type = par3m1[0]

            #<FILENAME>body10k.htm
            par3a=re.compile('<FILENAME>(.*)')
            par3am1 = par3a.findall(adoc)
            if len(par3am1) > 0:
                doc_fn = par3am1[0].strip()
                ext1 = doc_fn.split('.')
                ext2 = ext1[-1].lower()
                if(ext1[-1].lower() == "pdf"):
                  continue
            else:
                ext1=['nofile', 'txt']
                ext2 = ext1[-1].lower()
                doc_fn="nofilename.txt"

            if(get_target > 0):
                break

            if args.verbose >= 2:
                print("      type in db: %s -- doc_type:%s" % (header_info['ftype'], doc_type))

            if doc_type in (header_info['ftype']):
                get_target = 1            

                if(ext2 == "txt"):                    
                    clean_text = lib10kq.strip_tags(adoc)
                    clean_text = html.unescape(clean_text)
                    # remove html comment
                    html_com1 = re.compile('(<!--.*?-->)', re.M | re.S)
                    htmp1 = html_com1.subn('', clean_text)
                    clean_text = htmp1[0]
                    clean_text = lib10kq.translate2ascii(clean_text.encode('utf-8'))                    
                else:
                    orig1 = "<!DOCTYPE HTML PUBLIC'-//W3C//DTD HTML 3.2//EN\">"
                    replace1 = "<!DOCTYPE HTML PUBLIC\"-//W3C//DTD HTML 3.2//EN\">"

                    adoc=adoc.replace(orig1, replace1)

                    if(html2txt_type == "lynx"):                
                        raise(Exception("unsupported method: lynx"))
                        p = subprocess.Popen(['lynx', '-nolist', '--dump', '-width=2024000', '-stdin'],
                            stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
                        p.stdin.write(adoc.encode('utf-8'))
                        clean_text = p.communicate()[0]
                        p.wait()
                        # clean_text = unicodedata.normalize("NFKD", unicode(clean_text, 'ascii', errors='ignore')).encode('ascii', 'ignore')
                        clean_text = html.unescape(clean_text)
                        clean_text = translate2ascii(clean_text)
                    elif html2txt_type == "inscriptis":
                        clean_text = get_text(adoc)
                        clean_text = html.unescape(clean_text)
                        clean_text = lib10kq.translate2ascii(clean_text.encode('utf-8'))

                        if args.debug:
                            # fn1 = outprefix + "urlfile" + "_clean.htm.txt"                
                            fn1 = os.path.join(args.outputdir, "urlfile" + "_clean.htm.txt" )
                            with open(fn1, 'w', encoding = 'utf-8') as fh1:
                                fh1.write(clean_text)
                    else:
                        raise(Exception("unsupported html2txt conversion method %s" % html2txt_type))

                pure_text2 = lib10kq.pretty_text(clean_text)
                if args.debug:
                    fn1 = os.path.join(args.outputdir, "urlfile" + "_clean2.htm.txt")
                    with open(fn1, 'w', encoding = 'utf-8') as fh1:
                        fh1.write(pure_text2)
    elif urltype == "html":
        # HTML
        if args.verbose >= 2:
            print("Processing html file")
        # if src_type == "url":
        adoc = rawtext
        
        # (new method)
        if args.debug:
            # fn1 = outprefix + "%s_%s.htm" % ("urlfile", "webhtml")
            fn1 = os.path.join(args.outputdir, "%s_%s.htm" % ("urlfile", "webhtml"))        
            print("    Saving temp file %s" % fn1)
            fh1 = open(fn1, 'w', encoding = 'UTF-8')
            fh1.write(adoc)
            fh1.close()

        clean_text = get_text(adoc)
        clean_text = html.unescape(clean_text)
        clean_text = lib10kq.translate2ascii(clean_text.encode('utf-8'))
        pure_text2 = lib10kq.pretty_text(clean_text)
        if args.debug:
            # fn1 = outprefix + "urlfile" + "_clean.htm.txt"                
            fn1 = os.path.join(args.outputdir, "urlfile" + "_clean2.htm.txt")
            with open(fn1, 'w', encoding = 'utf-8') as fh1:
                fh1.write(pure_text2)
    elif urltype == "native_text":  
        clean_text = lib10kq.translate2ascii(rawtext.encode('utf-8'))
        pure_text2 = lib10kq.pretty_text(clean_text)
        if args.debug:
            # fn1 = outprefix + "urlfile" + "_clean.htm.txt"                
            fn1 = os.path.join(args.outputdir, "urlfile" + "_clean2.htm.txt")
            with open(fn1, 'w', encoding = 'utf-8') as fh1:
                fh1.write(pure_text2)
    elif urltype == "cleaned_text":
        pure_text2 = rawtext
    else:
        print(f"Unknown input_type {urltype}")
        sys.exit(195)

    if src_type == "url":
        if args.outfn_prefix == "AUTO":
            tmp1 = urllib.parse.urlparse(srcurl)
            lastpart = tmp1[2].split("/")[-1]
            args.outfn_prefix = lastpart
    else:
        # fn
        if args.outfn_prefix == "AUTO":
            args.outfn_prefix = os.path.basename(args.input)

    rawtext = pure_text2
    srcfn = "urlfn"

    lines = rawtext.split("\n")
    nrow = len(lines)
    if args.verbose >= 2:
        print("    There are %d lines (before removing empty lines)" % nrow, flush = True)  

    if method == "lstm":
        model_lstm.eval()
        nrow = len(lines)
        seqkeep = 0 # sequence line no for kept lines
        linekeep = []  # keeped lines
        seqmap = dict()  #map from kept line no. to original line no.
        for i, aline in enumerate(lines):
            aline = aline.strip()
            if len(aline) > 0:
                linekeep.append(aline)
                seqmap[seqkeep] = i
                seqkeep += 1

        x, doc_mask = lib10kq.gen_doc_feature(linekeep, word2vec_model=word2vec_model)
        x = torch.tensor(np.array(x))
        x = x.float()
        x = x.to(device)

        with torch.no_grad():     
            tmp_pred = model_lstm(x, doc_mask)
            # ce_loss = criterion(tmp_pred, y)
            # total_loss += ce_loss.cpu().item()
            max_pred = tmp_pred.argmax(dim = 1)
            max_pred = max_pred.cpu()
            max_pred = max_pred.tolist()
            pred = [reverse_label_mapping[tmp_pred] for tmp_pred in max_pred]    
            # preds.append(pred_label)

        # map the predicted tags back to original line sequence
        pred_ext = ['X'] * len(lines)
        for i, tag in enumerate(pred):
            i2 = seqmap[i]
            pred_ext[i2] = tag

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

        outdf = pd.DataFrame({'pred': pred_ext, 'sentence': lines})
        # csvstr = outdf.to_csv(index=False)
        if args.outfn_type.find("csv") >= 0:
            outdf.to_csv(os.path.join(args.outputdir, "%s.csv" % args.outfn_prefix), index = False)  


    if method == "crf":        
        pred_ext = crf_feature.pred_10k(lines, tagger)
        outdf = pd.DataFrame({'pred': pred_ext, 'sentence': lines})
        # csvstr = outdf.to_csv(index=False)
        if args.outfn_type.find("csv") >= 0:
            # outdf.to_csv(outprefix + "%s.csv" % os.path.basename(srcfn), index = False)
            outdf.to_csv(os.path.join(args.outputdir, "%s.csv" % args.outfn_prefix), index = False)

    # For chatgpt model start
    if method == 'chatgpt':
        if args.apikey is None:
            print("[Error] Please provide your openAI api key. \n\nUsing: \n        python3 -m itemseg --apikey YOUR_API_KEY \n\nto set up your api key.")
            sys.exit(1)        

        apikey = args.apikey

        # 處理要喂進 chatgpt model 的輸入
        text_final = gpt4itemSeg.preprocess_doc(args, lines)
        # 喂進 chatgpt model
        response = gpt4itemSeg.openai(text_final, apikey)
        # 取得與每行句子對應的預測tag
        pred_ext = gpt4itemSeg.map_lines_to_tags(response, lines)

        outdf = pd.DataFrame({'pred': pred_ext, 'sentence': lines})
        if args.outfn_type.find("csv") >= 0:
            outdf.to_csv(os.path.join(args.outputdir, "%s.csv" % args.outfn_prefix), index = False)        

    # For chatgpt model end

    # BERT
    if method == 'bert':        
        # load model
        # 組合路徑到 `bert_model.pth`
        # model_path = os.path.join(current_dir, args.bertpath)
        # bert_model_fn        
        print(f"Loading BERT model from {bert_model_fn}")

        # embeddings = sentence_bert_model.encode(lines, convert_to_tensor=True)  # (batch_size, embedding_dim)
        nrow = len(lines)
        seqkeep = 0 # sequence line no for keeped lines
        linekeep = []  # keeped lines
        seqmap = dict()  #map from keeped line no. to original line no.
        for i, aline in enumerate(lines):
            aline = aline.strip()
            if len(aline) > 0:
                linekeep.append(aline)
                seqmap[seqkeep] = i
                seqkeep += 1

        # Block start - 這個 block 是替換 line number 745, 757, 772 的部分
        total_features_df = createFeatures(linekeep)
        # print(f"total_features_df = {total_features_df}")        
        # raise(Exception("here"))

        doc_data = total_features_df.to_numpy()
        embeddings = sentence_bert_model.encode(linekeep)  # (batch_size, embedding_dim)
        final_data = np.hstack((doc_data, embeddings))
        final_data = torch.from_numpy(final_data).float().to(device)
        input_dim = final_data.shape[1]
        # tmp_batchx = final_data.unsqueeze(1)
        tmp_batchx = final_data.unsqueeze(0)
        # Block end
        # input_dim = embeddings.shape[1]

        if args.verbose >=2:
            print(f"final_data shape: {final_data.shape}")
            print(f"tmp_batchx shape: {tmp_batchx.shape}")
            print(f"input_dim: {input_dim}")
            print(f"label_mapping: {label_mapping}")
            print(f"hidden_dim: {hyperparameters_rnn['hidden_dim']}")
            print(f"num_layers: {args.num_layers}")
            print(f"device: {device}")
        

        model_bert = BiLSTM2(input_dim, 
                         label_mapping, 
                         hyperparameters_rnn['hidden_dim'], 
                         device, 
                         num_layers=args.num_layers).to(device)
        model_bert = model_bert.float() 

        ckpt = torch.load(bert_model_fn) 
        model_bert.load_state_dict(ckpt)
        # model_bert = model_lstm_crf   

        model_bert.eval()

        # tmp_batchx = embeddings.to(device).unsqueeze(1)  # (batch_size, 1, embedding_dim)

        # 初始化 hidden state
        batch_size = tmp_batchx.shape[0]
        model_bert.hidden = model_bert.init_hidden(batch_size)

        with torch.no_grad():
            tmp_pred = model_bert(tmp_batchx) 
            max_pred = tmp_pred.squeeze(0).argmax(dim=1)
            max_pred = max_pred.cpu().tolist()
            pred = [reverse_label_mapping[tmp_pred] for tmp_pred in max_pred]    

 
        # map the predicted tags back to original line sequence
        pred_ext = ['X'] * len(lines)
        for i, tag in enumerate(pred):
            i2 = seqmap[i]
            pred_ext[i2] = tag

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

        outdf = pd.DataFrame({'pred': pred_ext, 'sentence': lines})
        
        if args.outfn_type.find("csv") >= 0:
            outdf.to_csv(os.path.join(args.outputdir, "%s.csv" % args.outfn_prefix), index = False)  


    if args.verbose >= 1:
        print(f"Output files to {args.outputdir}/{args.outfn_prefix}*")
    lib10kq.write_item_file(args, lines, pred_ext)
    return 0
    
if __name__ == "__main__":
    sys.exit(main())
    

def createFeatures(doc):
    sen_pos_percentile = []
    back_sen_pos_percentile = []
    first_is_item = []
    headpos = []
    wordlen_list = []
    signature_list = []
    part_two_list = []
    content_1 = []
    content_2 = []
    re_name_list = []  # regular_expression name
    re_number_list = []  # regular_expression number    
    for sen_i, sen in enumerate(doc):

        # sen_pos_percentile
        sen_pos_percentile.append(sen_i / len(doc))

        # back sen_pos_percentile
        back_sen_pos_percentile.append((len(doc)-sen_i) / len(doc))

        # starts with item
        if itemShow(sen):
            first_is_item.append(1)
        else:
            first_is_item.append(0)

        # head_pos: 越大越可能不是head (超過threshold就是1) 代替裡面各種threshold
        headthreshold_ratio = 0.3  # 0 等於不看
        headthreshold = headthreshold_ratio * len(doc)
        headpos.append(min(sen_i, headthreshold) / headthreshold * 1.0)

        # wordlen
        wordlenmax = 20.0
        wordlen = len(sen.split())
        wordlen_list.append(min(wordlen, wordlenmax) / wordlenmax * 1.0)
        
        # signatureCheck
        signature_sign = signatureCheck(sen)
        signature_list.append(signature_sign)
        
        # partTwoCheck
        # part_two_sign = partTwoCheck(sen)
        # part_two_list.append(part_two_sign)

        # regular_expression name
        prefix, suffix = 'contain_', '_name'
        trans_row = itemNameCheck(prefix, suffix, sen)
        re_name_list.append(trans_row)

        # regular_expression by number
        prefix_, suffix_ = 'contain_', '_number'
        trans_row_ = itemNumberCheck(prefix_, suffix_, sen)
        re_number_list.append(trans_row_)

        # content_1
        if sen_i+10 < len(doc):
            if len(re.findall(r'ITEM[s]?\s*[0-9]+[(]?[A-Za-z]?[)]?[.]?\s?',doc[sen_i-1][0:150],re.IGNORECASE)) !=0 or \
               len(re.findall(r'ITEM[s]?\s*[0-9]+[(]?[A-Za-z]?[)]?[.]?\s?',doc[sen_i-2][0:150],re.IGNORECASE)) !=0 or \
               len(re.findall(r'ITEM[s]?\s*[0-9]+[(]?[A-Za-z]?[)]?[.]?\s?',doc[sen_i+1][0:150],re.IGNORECASE)) !=0 or \
               len(re.findall(r'ITEM[s]?\s*[0-9]+[(]?[A-Za-z]?[)]?[.]?\s?',doc[sen_i+2][0:150],re.IGNORECASE)) !=0:
                content_1.append(1)
            else:
                content_1.append(0)
        else:
            content_1.append(0)

        # content_2
        if sen_i+10 < len(doc):
            if len(re.findall(r'ITEM[s]?\s*[0-9]+[(]?[A-Za-z]?[)]?[.]?\s?',doc[sen_i+1][0:150],re.IGNORECASE)) !=0 or \
            len(re.findall(r'ITEM[s]?\s*[0-9]+[(]?[A-Za-z]?[)]?[.]?\s?',doc[sen_i+2][0:150],re.IGNORECASE)) !=0 or \
            len(re.findall(r'ITEM[s]?\s*[0-9]+[(]?[A-Za-z]?[)]?[.]?\s?',doc[sen_i+3][0:150],re.IGNORECASE)) !=0 or \
            len(re.findall(r'ITEM[s]?\s*[0-9]+[(]?[A-Za-z]?[)]?[.]?\s?',doc[sen_i+4][0:150],re.IGNORECASE)) !=0 or \
            len(re.findall(r'ITEM[s]?\s*[0-9]+[(]?[A-Za-z]?[)]?[.]?\s?',doc[sen_i+5][0:150],re.IGNORECASE)) !=0 or \
            len(re.findall(r'ITEM[s]?\s*[0-9]+[(]?[A-Za-z]?[)]?[.]?\s?',doc[sen_i+6][0:150],re.IGNORECASE)) !=0 or \
            len(re.findall(r'ITEM[s]?\s*[0-9]+[(]?[A-Za-z]?[)]?[.]?\s?',doc[sen_i+7][0:150],re.IGNORECASE)) !=0 or \
            len(re.findall(r'ITEM[s]?\s*[0-9]+[(]?[A-Za-z]?[)]?[.]?\s?',doc[sen_i+8][0:150],re.IGNORECASE)) !=0 or \
            len(re.findall(r'ITEM[s]?\s*[0-9]+[(]?[A-Za-z]?[)]?[.]?\s?',doc[sen_i+9][0:150],re.IGNORECASE)) !=0 or \
            len(re.findall(r'ITEM[s]?\s*[0-9]+[(]?[A-Za-z]?[)]?[.]?\s?',doc[sen_i+10][0:150],re.IGNORECASE)) !=0:
                content_2.append(1)
            else:
                content_2.append(0)
        else:
            content_2.append(0)
            

        
        # 我在baseline(regularEX)判斷時設了不少threshold(content, signature, part two)，但是在這個情況下是不是先不用
        # 設threshold了，有統一擺了一個 headthreshold

        # unigram and biagram 顏秀
        # sentence bert
        

    total_features = list(zip(sen_pos_percentile, 
                              back_sen_pos_percentile, 
                              first_is_item,
                              wordlen_list, 
                              signature_list, 
                              headpos, 
                              content_1, 
                              content_2))
    total_features_df = pd.DataFrame(total_features, 
              columns=['sen_pos_percentile', 
                       'back_sen_pos_percentile', 
                       'first_is_item',
                       'wordlen', 
                       'signature',  
                       'headpos', 
                       'content_1', 
                       'content_2'])
    re_features_df = pd.DataFrame(re_name_list)
    re_number_df = pd.DataFrame(re_number_list)
    total_features_df = total_features_df.join(re_features_df)
    total_features_df = total_features_df.join(re_number_df)
    
    return total_features_df

def itemShow(text):
    criteria = re.findall(r'^\s*item',text[0:15],re.IGNORECASE)
    if len(criteria) == 1 :
        return(1)
    else:
        return(0)
        

def signatureCheck(text):
    criteria = re.findall(r'signature',text,re.IGNORECASE)
    if len(criteria) != 0:
        return(1)
    else:
        return(0)

def itemNameCheck(prefix, suffix, text):
    alltasks = {
        'item 1': r'^item[\s\w.-]*business',
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
        'item 15': r'^item\s*15[.\s-]*exhibit[\w\s]*schedules'
        }
    item_name_features = {}  # 這就是 sentence 轉換後的特徵，一筆資料
    text = text.strip()
    for item, condition in alltasks.items():
        criteria = re.findall(condition, text, re.IGNORECASE)  # 抓回符合規則的那段string
        # print(criteria)
        if len(criteria) == 1:  # 多過一個就不要，因為 item 那句只會出現一次，也不會重複出現
            item_name_features[prefix+item+suffix] = 1
        else:        
            item_name_features[prefix+item+suffix] = 0
    return item_name_features

def itemNumberCheck(prefix, suffix, text):
    tag_list = ["1", "1A", "2", "3", "4", "5", "6"]
    pre150 = text[0:50]
    item_number_features = {}
    criteria = re.findall(r'ITEM[s]?\s*[0-9]+[(]?[A-Za-z]?[)]?[.]?\s?',pre150,re.IGNORECASE) #  For most cases
    if len(criteria) !=0:
        # print(criteria[0]) # ex.ITEM 1.
        sign = re.findall(r'[0-9]+[()]?[A-Za-z]?[)]?',criteria[0])
        recommend_tag = sign[0].upper()
        for tag in tag_list:
            if tag == recommend_tag:
                item_number_features[prefix+tag+suffix] = 1
            else:
                item_number_features[prefix+tag+suffix] = 0
    else:
        for tag in tag_list:
            item_number_features[prefix+tag+suffix] = 0
    return item_number_features
