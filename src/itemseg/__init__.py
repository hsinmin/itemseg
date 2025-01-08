#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import platform
import sys, re, os, collections
import re
import time
import os
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
import numpy as np
from itemseg import lib_10kq_seg_v1 as lib10kq
from itemseg import crf_feature_lib_v8 as crf_feature
from argparse import ArgumentParser
import urllib.parse
import pathlib
html2txt_type = "inscriptis"


def get_resource(dest="__home__", check_only=False, verbose=1, 
                 url0 = "http://www.im.ntu.edu.tw/~lu/data/itemseg/"):
    files = ['crf8f6_m5000c2_1f_200f06c1_0.00c2_1.00_m5000.crfsuite',
             'word2vecmodel_10kq3a_epoch_5',
             'word2vecmodel_10kq3a_epoch_5.syn1neg.npy',
             'word2vecmodel_10kq3a_epoch_5.wv.vectors.npy',
             'tag2023_v1_labelidmap.pkl',
             'isla_model/h256len100lay2lr3complete_args.json',
             'isla_model/h256len100lay2lr3complete_e020_vac97.31_vce0.08639.pth']
    
    if dest == "__home__":
        # replace with real home path
        dest = str(pathlib.Path.home()) + "/itemseg/resource/"
        
    if check_only == False:
        print(f"Download resource to {dest}")
        if not os.path.exists(dest):
            os.makedirs(dest)
            os.makedirs(os.path.join(dest, "isla_model"))
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
    parser.add_argument("--resource_url", dest="resource_url", type=str,
                        default="http://www.im.ntu.edu.tw/~lu/data/itemseg/",
                        help="URL to download resource files")
    # input options
    # currently does not support local file yet
    parser.add_argument("--input", dest="input", type=str,
                        # default='',
                        # required=True,
                        help="EDGAR filing URL; e.g. https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/0000320193-23-000106.txt")
    parser.add_argument("--input_type", dest="input_type", type=str,
                         default='auto',
                         help="Input type. auto: determine automatically; fn: file name; url: obtain web page by requests")

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
                        default='isla',
                        help="Item segmentation method; isla or crf")
    parser.add_argument("--word2vec", dest="word2vec", type=str,
                        default='./resource/word2vecmodel_10kq3a_epoch_5',
                        help="File name of the word2vec model (gensim trained)")
    parser.add_argument("--islapath", dest="islapath", type=str,
                        default="./resource/isla_model",
                        help="ISLA model (path) for inference")
    parser.add_argument("--crfpath", dest="crfpath", type=str,
                        default="./resource/crf8f6_m5000c2_1f_200f06c1_0.00c2_1.00_m5000.crfsuite",
                        help="CRF model (path) for inference")
    parser.add_argument("--labelid_map", dest="labelid_map", type=str,
                        default='./resource/tag2023_v1_labelidmap.pkl',
                        help="labelid mapping file; a dictionary of two maps (for ISLA)")
    parser.add_argument("--verbose", dest="verbose", default = 1, type = int,
                        help="verbose level=0, 1, or 2; 0=silent, 2=many messages")
    parser.add_argument("--debug", dest="debug", 
                        action="store_true",
                        help="save in-progress files for debugging")

    args = parser.parse_args()
    args.hostname = platform.node()

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
    #                                "--method", "isla"])
    
    
    if args.verbose >=1:
        print("itemseg: Item Segmentation with Line-based Attention (ISLA)")
        print("    A 10-K Item Segmentation Tool")
        print("    Free to use for non-commercial purpose.")
        print("    Maintained by: Hsin-Min Lu (luim@ntu.edu.tw)")
        # todo: add project URL
        print("    Please cite our work if you use this tool in your research.")

    if args.verbose >=2:
        print("Arguments:", args)
        
    if (args.input is None) and (args.get_resource == False):
        parser.error("Need either --input or --get_resource")

    
    if args.get_resource:
        get_resource()
        sys.exit(0)
    
    method = args.method

    rdnseed = 52345

    resource_prefix = str(pathlib.Path.home()) + "/itemseg/"

    # crf_model_fn = "resource/crf8f6_m5000c2_1f_200f06c1_0.00c2_1.00_m5000.crfsuite"
    crf_model_fn = os.path.join(resource_prefix, args.crfpath)
    # isla_model_fn = "resource/isla_model"
    isla_model_fn = os.path.join(resource_prefix, args.islapath)
    # word2vec_fn = "resource/word2vecmodel_10kq3a_epoch_5"
    word2vec_fn = os.path.join(resource_prefix, args.word2vec)
    # label2id_fn = "resource/tag2023_v1_labelidmap.pkl"
    label2id_fn = os.path.join(resource_prefix, args.labelid_map)
    
    res_files = [crf_model_fn, 
                 isla_model_fn,
                 word2vec_fn,
                 label2id_fn]
    for ares in res_files:
        if os.path.exists(ares) == False:
            print(f"Cannot find resource file {crf_model_fn}.\n" 
                   "Did you foreget to download resource files with '--get_resource'?")
            sys.exit(300)

    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    if method == "isla":    
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
            print(f"==== Reading ISLA model files in {isla_model_fn}")
        # args.inference_only = True 
        fns = glob.glob(isla_model_fn + "/*_args.json")
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
        fns2 = glob.glob(isla_model_fn + "/*.pth")
        fns2 = sorted(fns2)
        best_model_name = fns2[-1]
        if args.verbose >= 2:
            print(f"     Using model file {best_model_name}")
    elif method == "crf":
        #load tagger
        tagger = pycrfsuite.Tagger()
        tagger.open(crf_model_fn)
    else:
        print(f"Unknonwn method {method}. Stop")
        sys.exit(103)

    
    if method == "isla":
        input_dim = len(word2vec_model.wv['a']) + 3
        if args.verbose >=2:
            print(f"   LSTM input dim = {input_dim}")
        model_isla = lib10kq.BiLSTM_Tok(input_dim, 
                                    label_mapping, 
                                    args.hidden_dim, 
                                    device,
                                    attention_method=args.attention_method,
                                    num_layers=args.num_layers).to(device)
        model_isla = model_isla.float()
        
        # load model
        if device == "cpu":
            ckpt = torch.load(best_model_name, torch.device('cpu'))
        else:
            ckpt = torch.load(best_model_name)  
        model_isla.load_state_dict(ckpt)

    if args.input_type == "auto":
        if args.input.find("http") >=0:
            src_type = "url"
        else:
            src_type = "fn"
            # urltype = "fn"
    else:
        src_type = args.input_type

    # src_type = "txt"
    # src_type = "url"
    if args.verbose >= 2:
        print("Input type is", src_type)


    if src_type == "fn":
        srcfn = "rawdata/6404287.txt"
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
        headers = {        
            "User-Agent": "Item Segmentation with Line-Based Attention",
            "Accept-Encoding": "gzip, deflate" 
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
        else:
            rawtext = r.text
            
            
    par1 = re.compile('(<SEC-DOCUMENT>.*?</SEC-HEADER>)(.*)', re.M | re.S)
    par1m1=par1.findall(rawtext)
    if len(par1m1) == 0:
        print("Cannot find critial tags (SEC-DOCUMENT to SEC-HEADER). Assume to be HTML file.")
        urltype = "HTML"                
    else:
        sec_header = par1m1[0][0]
        html1 = par1m1[0][1]
        urltype = "RAW"
      
    if args.verbose >= 1:
        print(f"URL Type = {urltype} (HTML=Ordinary HTML; RAW=EDGAR Complete submission text file)")    

    # prase sec_header
    if urltype == "RAW":
        header_info = lib10kq.parse_edgar_header(sec_header)
        
        if args.verbose >= 1:
            print(f"Company Name = {header_info['cname']}")
            print(f"File type = {header_info['ftype']}")
            print(f"Confirmed period of report = {header_info['cpr']}")
            print(f"Industry: {header_info['sic_desc']} - {header_info['sic_code']}")

    if urltype == "RAW":
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
                        fh1.write(clean_text)
    elif urltype == "HTML":
        # HTML
        if args.verbose >= 2:
            print("Processing html file")
        if src_type == "url":
            adoc = r.text
        
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
                fh1.write(clean_text)

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


    if method == "isla":    
        model_isla.eval()        
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

        x, doc_mask = lib10kq.gen_doc_feature(linekeep, word2vec_model=word2vec_model)
        x = torch.tensor(np.array(x))
        x = x.float()
        x = x.to(device)

        with torch.no_grad():     
            tmp_pred = model_isla(x, doc_mask)
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

    if args.verbose >= 1:
        print(f"Output files to {args.outputdir}/{args.outfn_prefix}*")
    lib10kq.write_item_file(args, lines, pred_ext)
    return 0
    
if __name__ == "__main__":
    sys.exit(main())
    