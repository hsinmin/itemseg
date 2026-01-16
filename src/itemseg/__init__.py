#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10-K Item Segmentation Tool

This module provides functionality to segment SEC 10-K filings into their constituent items
using various machine learning approaches (CRF, LSTM, BERT, ChatGPT).
"""

# Standard library imports
import platform
import sys
import re
import os
import glob
import json
import pickle
import pathlib
import urllib.parse
import html

# Third-party imports
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pycrfsuite
import gensim
from inscriptis import get_text
from argparse import ArgumentParser

# Local imports
from itemseg import lib_10kq_seg_v1 as lib10kq
from itemseg import crf_feature_lib_v8 as crf_feature
from itemseg import gpt4itemSeg

# Configuration: HTML to text conversion method
html2txt_type = "inscriptis"

def get_resource(dest="__home__", check_only=False, verbose=1,
                 url0="http://nebula.lu.im.ntu.edu.tw/itemseg/"):
    """
    Download required model resource files from remote server.

    Args:
        dest (str): Destination directory for downloaded files.
                   "__home__" will resolve to ~/itemseg/resource/
        check_only (bool): If True, only check for resources without downloading
        verbose (int): Verbosity level (0=silent, 1=normal, 2=detailed)
        url0 (str): Base URL for downloading resource files

    Returns:
        None. Downloads files to the specified destination directory.
    """
    # List of required model files to download
    files = ['crf8f6_m5000c2_1f_200f06c1_0.00c2_1.00_m5000.crfsuite',  # CRF model
             'word2vecmodel_10kq3a_epoch_5',  # Word2Vec model base
             'word2vecmodel_10kq3a_epoch_5.syn1neg.npy',  # Word2Vec negative sampling weights
             'word2vecmodel_10kq3a_epoch_5.wv.vectors.npy',  # Word2Vec word vectors
             'tag2023_v1_labelidmap.pkl',  # Label mapping for LSTM
             'tag2021_v3_labelidmap.pkl',  # Label mapping for BERT/CRF
             'bert_model/bert_model.pth',  # BERT model weights
             'lstm_model/h256len100lay2lr3complete_args.json',  # LSTM model config
             'lstm_model/h256len100lay2lr3complete_e020_vac97.31_vce0.08639.pth']  # LSTM weights

    # Resolve destination path
    if dest == "__home__":
        # Use user's home directory
        dest = os.path.join(str(pathlib.Path.home()), "itemseg", "resource")

    if check_only == False:
        # Download mode: fetch all resource files
        print(f"Download resource to {dest}")

        # Create directory structure if it doesn't exist
        if not os.path.exists(dest):
            os.makedirs(dest)
            os.makedirs(os.path.join(dest, "lstm_model"))
            os.makedirs(os.path.join(dest, "bert_model"))

        # Download each resource file
        err_count = 0
        for atarget in files:
            url = url0 + "resource/" + atarget
            outfn = os.path.join(dest, atarget)

            if verbose >= 1:
                print(f"Getting {url}")

            # Fetch file from remote server
            r = requests.get(url, allow_redirects=True)
            if r.status_code == 200:
                print(f"   write to {outfn}")
                open(outfn, 'wb').write(r.content)
            else:
                print(f"   Error {r.status_code}: Cannot access {url}")
                err_count += 1

        if err_count == 0:
            print("Resource download completed")
        

def main():
    """
    Main entry point for the 10-K item segmentation tool.

    This function:
    1. Parses command-line arguments
    2. Downloads or verifies resource files
    3. Loads input 10-K filing (from URL or local file)
    4. Preprocesses the document (HTML/text conversion, cleaning)
    5. Loads the specified ML model (CRF/LSTM/BERT/ChatGPT)
    6. Performs item segmentation
    7. Outputs results to CSV and individual item files

    Returns:
        int: Exit code (0 for success, non-zero for errors)
    """
    parser = ArgumentParser()

    # Resource management arguments
    parser.add_argument("--get_resource", dest="get_resource",
                        action="store_true",
                        help="Download resource files (models, embeddings, etc.)")
    # default_resource_url = "http://nebula.lu.im.ntu.edu.tw/itemseg/"
    default_resource_url = "https://www.im.ntu.edu.tw/~lu/data/itemseg/"
    parser.add_argument("--resource_url", dest="resource_url", type=str,
                        default=default_resource_url,
                        help=f"Set URL to download resource files. Default: {default_resource_url}")
    # Input source arguments
    parser.add_argument("--input", dest="input", type=str,
                        help="Path to local input file or EDGAR filing URL; "
                             "e.g. https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/0000320193-23-000106.txt")
    parser.add_argument("--input_type", dest="input_type", type=str,
                         help="Input file format type:\n"
                              "  raw: Complete submission text file with SEC headers\n"
                              "  html: HTML report from EDGAR\n"
                              "  native_text: Native text format 10-K\n"
                              "  cleaned_text: Pre-processed text with tables removed")
    parser.add_argument("--user_agent_str", dest="user_agent_str", type=str,
                         default='N/A',
                         help="User Agent string required by SEC for automated requests. "
                              "Format: 'Company Name Contact@domain.com'")

    # Output configuration arguments
    parser.add_argument("--outputdir", dest="outputdir", type=str,
                        default="./segout01/",
                        help="Directory where output files will be saved")
    parser.add_argument("--outfn_prefix", dest="outfn_prefix", type=str,
                        default="AUTO",
                        help="Prefix for output filenames (AUTO=derive from input filename)")
    parser.add_argument("--outfn_type", dest="outfn_type", type=str,
                        default="csv,item1,item1a,item3,item7",
                        help="Output file types (comma-separated): "
                             "csv=line-by-line predictions; itemX=extract specific items to separate files")

    # Model selection and configuration arguments
    parser.add_argument("--method", dest="method", type=str,
                        default='crf',
                        help="Item segmentation method:\n"
                             "  crf: Conditional Random Field (CPU-friendly, no GPU required)\n"
                             "  lstm: Bidirectional LSTM (requires GPU for best performance)\n"
                             "  bert: BERT encoder + BiLSTM (requires GPU)\n"
                             "  chatgpt: OpenAI API with prompt-based segmentation")
    parser.add_argument("--word2vec", dest="word2vec", type=str,
                        default='./resource/word2vecmodel_10kq3a_epoch_5',
                        help="Path to Word2Vec model file (gensim format, used by LSTM)")
    parser.add_argument("--lstmpath", dest="lstmpath", type=str,
                        default="./resource/lstm_model",
                        help="Path to LSTM model directory containing weights and config")
    parser.add_argument("--crfpath", dest="crfpath", type=str,
                        default="./resource/crf8f6_m5000c2_1f_200f06c1_0.00c2_1.00_m5000.crfsuite",
                        help="Path to trained CRF model file")
    parser.add_argument("--labelid_map", dest="labelid_map", type=str,
                        default='AUTO',
                        help="Path to label-to-ID mapping pickle file (AUTO=auto-select based on method)")
    parser.add_argument("--verbose", dest="verbose", default=1, type=int,
                        help="Verbosity level: 0=silent, 1=normal, 2=detailed debug info")
    parser.add_argument("--debug", dest="debug",
                        action="store_true",
                        help="Save intermediate processing files for debugging")
    parser.add_argument('--bertpath', dest='bertpath', type=str,
                        default='./resource/bert_model/bert_model.pth',
                        help="Path to BERT model weights file")
    parser.add_argument('--apikey', dest='apikey', type=str,
                        default=None,
                        help='OpenAI API key (required for chatgpt method)')
    
    # Parse command-line arguments
    args = parser.parse_args()
    args.hostname = platform.node()

    # Set internal parameters (used by some models)
    # Note: These are mostly legacy parameters, not exposed to CLI
    args.optimizer = "Adam"
    args.lr = 0.0001  # Learning rate (not used in inference)
    args.weight_decay = 0.0
    args.num_layers = 2  # LSTM/BERT layers
    args.hidden_dim = 256  # Hidden dimension for LSTM/BERT

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
    
    
    # Display banner and version info
    if args.verbose >= 1:
        print("itemseg: A tool for 10-K Item Segmentation")
        print("    Free to use for non-commercial purpose.")

    if args.verbose >= 2:
        print("Arguments:", args)

    # Validate command-line arguments
    if (args.input is None) and (args.get_resource == False):
        parser.error("Need either --input or --get_resource")

    # Handle resource download mode
    if args.get_resource:
        get_resource(url0=args.resource_url)
        sys.exit(0)

    # Validate input_type parameter
    if args.input_type is None:
        parser.error("Need to specify input_type")

    legal_input_type = ['raw', 'html', 'native_text', 'cleaned_text']
    if args.input_type not in legal_input_type:
        parser.error(f"Illegal input type. Must be one of: {legal_input_type}")

    # Configure method and paths
    method = args.method
    rdnseed = 52345  # Random seed for reproducibility
    resource_prefix = str(pathlib.Path.home()) + "/itemseg/"

    # Construct full paths to model resource files
    crf_model_fn = os.path.join(resource_prefix, args.crfpath)
    lstm_model_fn = os.path.join(resource_prefix, args.lstmpath)
    word2vec_fn = os.path.join(resource_prefix, args.word2vec)

    # Auto-select appropriate label mapping based on method
    if args.labelid_map == "AUTO":
        if args.method in ["lstm"]:
            label2id_fn = os.path.join(resource_prefix, "resource/tag2023_v1_labelidmap.pkl")
        elif args.method in ['crf', 'bert', 'chatgpt']:    # need to verify the chatgpt case
            label2id_fn = os.path.join(resource_prefix, "resource/tag2021_v3_labelidmap.pkl")
        else:
            print(f"Unknown method {args.method} for automatic labelid_map assignment")
            sys.exit(105)
    else:
        label2id_fn = os.path.join(resource_prefix, args.labelid_map)

    bert_model_fn = os.path.join(resource_prefix, args.bertpath)

    # Verify that all required resource files exist
    res_files = [crf_model_fn,
                 lstm_model_fn,
                 word2vec_fn,
                 label2id_fn,
                 bert_model_fn]
    for ares in res_files:
        if os.path.exists(ares) == False:
            print(f"Cannot find resource file {ares}.\n"
                   "Did you forget to download resource files with '--get_resource'?")
            sys.exit(300)

    # Create output directory if it doesn't exist
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    # Determine compute device (GPU if available, otherwise CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ================================
    # STEP 1: Load the 10-K filing
    # ================================

    # Determine if input is a URL or local file
    if args.input.find("http") >= 0:
        src_type = "url"
    else:
        src_type = "fn"  # filename

    if args.verbose >= 2:
        print("Source type is", src_type)

    # Load input from local file
    if src_type == "fn":
        with open(args.input, "r") as fh:
            rawtext = fh.read()

    # Load input from URL
    elif src_type == "url":
        srcurl = args.input

        # Validate SEC.gov URL
        if srcurl.find("sec.gov/") < 0:
            if args.verbose >= 1:
                print("Warning: this is not a sec.gov URL.")

        # Handle EDGAR interactive viewer URLs (convert to direct file URLs)
        if srcurl.find("sec.gov/ix?doc=/") >= 0:
            srcurl = srcurl.replace("ix?doc=/", "")
            if args.verbose >= 1:
                print("EDGAR dynamic URL detected. Apply URL translation.")
                print(f"    Accessing {srcurl} instead")

        print(f"Getting input file from {srcurl}")

        # SEC requires User-Agent header for automated access
        if args.user_agent_str == "N/A":
            print("You need to specify user_agent_str per SEC's rule.")
            print("cf. https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data")
            sys.exit(200)

        # Prepare HTTP headers for SEC compliance
        headers = {
            "User-Agent": args.user_agent_str,
            "Accept-Encoding": "gzip, deflate",
            "host": "www.sec.gov"
            }

        # Fetch the filing from SEC
        r = requests.get(srcurl, headers=headers)
        if args.verbose >= 2:
            print(f"URL respond code = {r.status_code}")

        # Save raw response for debugging if requested
        if args.debug:
            with open(os.path.join(args.outputdir, "rawfile.txt"),
                      "w", encoding="utf-8") as my_file:
                my_file.write(r.text)

        # Validate response
        if(r.text == None):
            print(f"No response from target URL. Stop (response code = {r.status_code})")
            sys.exit(100)
        elif len(r.text) < 50:
            print(f"The length of filed text is too small ({len(r.text)} chars). Stop.")
            sys.exit(101)
        elif r.text.find("Your Request Originates from an Undeclared Automated Tool") >= 0:
            print(f"Error: SEC denied undeclared automated tool!")
            print(f"Header: {headers}")
            sys.exit(102)
        else:
            rawtext = r.text

    # ================================
    # STEP 2: Detect and validate input type
    # ================================         

    # Try to detect SEC submission header (present in "raw" type files)
    par1 = re.compile(r'(<SEC-DOCUMENT>.*?</SEC-HEADER>)(.*)', re.M | re.S)
    par1m1 = par1.findall(rawtext)

    if len(par1m1) == 0:
        # No SEC header found - must be HTML, native_text, or cleaned_text
        print("Cannot find header tags (SEC-DOCUMENT to SEC-HEADER). Assume to be user specified type.")
        if args.input_type == "raw":
            print(f"User specified input_type={args.input_type} but the header does not exist. Stop")
            sys.exit(191)
        urltype = args.input_type
    else:
        # SEC header found - this is a "raw" submission file
        sec_header = par1m1[0][0]  # Header portion
        html1 = par1m1[0][1]  # Document portion
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
        par2 = re.compile(r'(<DOCUMENT>.*?</DOCUMENT>)', re.M | re.S)
        par2m1= par2.findall(html1)
        get_target = 0

        if args.verbose >= 2:
            print("# of document component:", len(par2m1))

        for adoc in par2m1:
            par3=re.compile(r'<TYPE>(\S+)')
            par3m1 = par3.findall(adoc)
            doc_type = par3m1[0]

            #<FILENAME>body10k.htm
            par3a=re.compile(r'<FILENAME>(.*)')
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
                    html_com1 = re.compile(r'(<!--.*?-->)', re.M | re.S)
                    htmp1 = html_com1.subn('', clean_text)
                    clean_text = htmp1[0]
                    clean_text = lib10kq.translate2ascii(clean_text.encode('utf-8'))                    
                else:
                    orig1 = "<!DOCTYPE HTML PUBLIC'-//W3C//DTD HTML 3.2//EN\">"
                    replace1 = "<!DOCTYPE HTML PUBLIC\"-//W3C//DTD HTML 3.2//EN\">"

                    adoc=adoc.replace(orig1, replace1)

                    if(html2txt_type == "lynx"):                
                        raise(Exception("unsupported method: lynx"))
                    elif html2txt_type == "inscriptis":
                        clean_text = get_text(adoc)
                        if args.debug:                            
                            fn1 = os.path.join(args.outputdir, "urlfile" + "_clean00.htm.txt" )
                            with open(fn1, 'w', encoding = 'utf-8') as fh1:
                                fh1.write(clean_text)
                        clean_text = html.unescape(clean_text)
                        if args.debug:
                            fn1 = os.path.join(args.outputdir, "urlfile" + "_clean01.htm.txt" )
                            with open(fn1, 'w', encoding = 'utf-8') as fh1:
                                fh1.write(clean_text)
                        clean_text = lib10kq.translate2ascii(clean_text.encode('utf-8'))
                        if args.debug:                            
                            fn1 = os.path.join(args.outputdir, "urlfile" + "_clean02.htm.txt" )
                            with open(fn1, 'w', encoding = 'utf-8') as fh1:
                                fh1.write(clean_text)

                        if args.debug:
                            # fn1 = outprefix + "urlfile" + "_clean.htm.txt"                
                            fn1 = os.path.join(args.outputdir, "urlfile" + "_clean03.htm.txt" )
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
        if args.debug:
            # fn1 = outprefix + "urlfile" + "_clean.htm.txt"                
            fn1 = os.path.join(args.outputdir, "urlfile" + "_clean00.htm.txt")
            with open(fn1, 'w', encoding = 'utf-8') as fh1:
                fh1.write(clean_text)
        clean_text = html.unescape(clean_text)
        if args.debug:
            # fn1 = outprefix + "urlfile" + "_clean.htm.txt"                
            fn1 = os.path.join(args.outputdir, "urlfile" + "_clean01.htm.txt")
            with open(fn1, 'w', encoding = 'utf-8') as fh1:
                fh1.write(clean_text)
        clean_text = lib10kq.translate2ascii(clean_text.encode('utf-8'))
        if args.debug:
            # fn1 = outprefix + "urlfile" + "_clean.htm.txt"                
            fn1 = os.path.join(args.outputdir, "urlfile" + "_clean02.htm.txt")
            with open(fn1, 'w', encoding = 'utf-8') as fh1:
                fh1.write(clean_text)
        pure_text2 = lib10kq.pretty_text(clean_text)
        if args.debug:
            # fn1 = outprefix + "urlfile" + "_clean.htm.txt"                
            fn1 = os.path.join(args.outputdir, "urlfile" + "_clean03.htm.txt")
            with open(fn1, 'w', encoding = 'utf-8') as fh1:
                fh1.write(pure_text2)
        if args.debug:
            # fn1 = outprefix + "urlfile" + "_clean.htm.txt"                
            fn1 = os.path.join(args.outputdir, "urlfile" + "_clean04.htm.txt")
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
    # srcfn = "urlfn"

    lines = rawtext.split("\n")
    nrow = len(lines)
    if args.verbose >= 2:
        print("    There are %d lines (before removing empty lines)" % nrow, flush = True)  

    # ====================
    # input file is ready, now we can start to prepare the model

    # prepare the model
    if method == "lstm":    
        # prepare the lstm model
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
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if args.verbose >= 2:
            print("Using device", device)

        
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
        from sentence_transformers import SentenceTransformer
        # this will trigger internet access. Consider to add option to supress this behavior
        sentence_bert_model = SentenceTransformer('stsb-mpnet-base-v2')

        
        if args.verbose >= 2:
            print("Using device", device)
        
        # current_dir = os.path.dirname(__file__)        
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

        pred_ext = lib10kq.expand_pred_to_lines (pred, seqmap, lines)     
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

        # chatgpt model input
        text_final = gpt4itemSeg.preprocess_doc(args, lines)
        # submit to chatgpt model
        response = gpt4itemSeg.openai(text_final, apikey)
        # get predicted tag
        pred_ext = gpt4itemSeg.map_lines_to_tags(response, lines)

        outdf = pd.DataFrame({'pred': pred_ext, 'sentence': lines})
        if args.outfn_type.find("csv") >= 0:
            outdf.to_csv(os.path.join(args.outputdir, "%s.csv" % args.outfn_prefix), index = False)        

    # For chatgpt model end

    # BERT
    if method == 'bert':        
        # load model
        # `bert_model.pth`
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

        total_features_df = lib10kq.createFeatures(linekeep)
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
            print(f"hidden_dim: {args.hidden_dim}")
            print(f"num_layers: {args.num_layers}")
            print(f"device: {device}")
        

        model_bert = lib10kq.BiLSTM2(input_dim, 
                         label_mapping, 
                         args.hidden_dim, 
                         device, 
                         num_layers=args.num_layers).to(device)
        model_bert = model_bert.float() 

        if device == "cpu":
            ckpt = torch.load(bert_model_fn, map_location=torch.device('cpu'))
        else:
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
        pred_ext = lib10kq.expand_pred_to_lines (pred, seqmap, lines)        
        outdf = pd.DataFrame({'pred': pred_ext, 'sentence': lines})
        
        if args.outfn_type.find("csv") >= 0:
            outdf.to_csv(os.path.join(args.outputdir, "%s.csv" % args.outfn_prefix), index = False)  

    if args.verbose >= 1:
        print(f"Output files to {args.outputdir}/{args.outfn_prefix}*")
    lib10kq.write_item_file(args, lines, pred_ext)
    return 0
    
if __name__ == "__main__":
    sys.exit(main())
    

