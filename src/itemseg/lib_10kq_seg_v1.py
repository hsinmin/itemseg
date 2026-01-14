"""
Library for 10-K/10-Q SEC Filing Item Segmentation (Version 1)

This module provides functionality for processing and segmenting SEC 10-K and 10-Q
filings into their constituent items (Items 1-15).

Key Components:
- Text preprocessing: HTML stripping, Unicode-to-ASCII conversion, table/noise filtering
- EDGAR header parsing: Extract filing metadata (company name, dates, SIC codes)
- Feature engineering: Position-based, word-based, and regex-based features for item detection
- Neural network models: PyTorch sequence tagging models with attention mechanisms
- Training & evaluation: Training loops, validation, metrics computation (precision, recall, F1)
- Item extraction: Extract and save specific items from segmented filings
- Performance utilities: Timer and progress tracking for long-running operations
"""

from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import unicodedata
import nltk
import re
import collections
from nltk import tokenize
import os
from html.parser import HTMLParser
from io import StringIO
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import pandas as pd

# Global tokenizer configuration
word_tokenizer = nltk.tokenize.wordpunct_tokenize  # Standard word tokenizer

# English punctuation marks for text processing
eng_punc = ['~', '\\', '>', '<', '@', '|', '+', '.', '?', '!', ':','=', '*', '-', ',', '(', ')', '[', ']', '{', '}', '/', '$', '%', '&', ';', '"', "'"]

# Regex patterns for text cleaning
re_float = re.compile('([+-]?\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?')  # Matches floating point numbers
punc_splus = re.compile(r'[=_\s.-]{2,}')  # Matches 2+ repeated punctuation/whitespace

class MLStripper(HTMLParser):
    """
    HTML tag stripper that removes all HTML markup from text.

    Extends HTMLParser to extract only the text content from HTML,
    discarding all tags, attributes, and formatting.

    OBSOLETE: Do not use. This implementation may be replaced in future versions.
    """
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True  # Convert HTML entities to unicode
        self.text = StringIO()  # Buffer to accumulate text content

    def handle_data(self, d):
        """Handle text data between HTML tags."""
        self.text.write(d)

    def get_data(self):
        """Return the accumulated text content."""
        return self.text.getvalue()

def strip_tags(html):
    """
    Strip all HTML tags from input text, keeping only text content.
    This function is only used in simple case (pure text filings).
    Do not use this function on HTML filings.

    Args:
        html: String containing HTML markup

    Returns:
        String with all HTML tags removed

    OBSOLETE: Do not use. This function may be replaced in future versions.
    """
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def write_item_file(args, lines, pred_ext):
    """
    Extract items from predicted tags and write to separate output files.

    This function extracts specific items (1, 1A, 2-14) from segmented filing text
    based on predicted sequence tags and writes each item to a separate file.

    Args:
        args: Arguments object containing:
            - outfn_type: Comma-separated list of items to output (e.g., "item1,item2")
            - outputdir: Output directory for item files
            - outfn_prefix: Prefix for output filenames
        lines: List of text lines from the filing
        pred_ext: List of predicted tags (e.g., ['B1', 'I1', 'O', 'B2', ...])

    Output:
        Writes files named "{prefix}_1.txt", "{prefix}_1A.txt", etc. to outputdir
    """
    # Extract all items using Begin/Inside tags (B/I notation)
    b1_str, b1_nl = item_extract(lines, pred_ext, ['B1', 'I1'])    
    b1_wlen = word_len(b1_str)
    b1a_str, b1a_nl = item_extract(lines, pred_ext, ['B1A', 'I1A'])
    b1a_wlen = word_len(b1a_str)
    b2_str, b2_nl = item_extract(lines, pred_ext, ['B2', 'I2'])
    b2_wlen = word_len(b2_str)
    b3_str, b3_nl = item_extract(lines, pred_ext, ['B3', 'I3'])
    b3_wlen = word_len(b3_str)
    b4_str, b4_nl = item_extract(lines, pred_ext, ['B4', 'I4'])
    b4_wlen = word_len(b4_str)
    b5_str, b5_nl = item_extract(lines, pred_ext, ['B5', 'I5'])
    b5_wlen = word_len(b5_str)
    b6_str, b6_nl = item_extract(lines, pred_ext, ['B6', 'I6'])
    b6_wlen = word_len(b6_str)
    b7_str, b7_nl = item_extract(lines, pred_ext, ['B7', 'I7'])
    b7_wlen = word_len(b7_str)
    b7a_str, b7a_nl = item_extract(lines, pred_ext, ['B7A', 'I7A'])
    b7a_wlen = word_len(b7a_str)
    b8_str, b8_nl = item_extract(lines, pred_ext, ['B8', 'I8'])
    b8_wlen = word_len(b8_str)

    b9_str, b9_nl = item_extract(lines, pred_ext, ['B9', 'I9'])
    b9_wlen = word_len(b9_str)
    b9a_str, b9a_nl = item_extract(lines, pred_ext, ['B9A', 'I9A'])
    b9a_wlen = word_len(b9a_str)
    b10_str, b10_nl = item_extract(lines, pred_ext, ['B10', 'I10'])
    b10_wlen = word_len(b10_str)
    b11_str, b11_nl = item_extract(lines, pred_ext, ['B11', 'I11'])
    b11_wlen = word_len(b11_str)
    b12_str, b12_nl = item_extract(lines, pred_ext, ['B12', 'I12'])
    b12_wlen = word_len(b12_str)
    b13_str, b13_nl = item_extract(lines, pred_ext, ['B13', 'I13'])
    b13_wlen = word_len(b13_str)
    b14_str, b14_nl = item_extract(lines, pred_ext, ['B14', 'I14'])
    b14_wlen = word_len(b14_str)    
    
    
    outtype = args.outfn_type.split(",")
    outtype = [x.strip() for x in outtype]

    if "item1" in outtype:
        with open(os.path.join(args.outputdir, "%s_1.txt" % args.outfn_prefix), 'w') as fh1:
            fh1.write(b1_str)

    if "item1a" in outtype:
        with open(os.path.join(args.outputdir, "%s_1A.txt" % args.outfn_prefix), 'w') as fh1:
            fh1.write(b1a_str)

    if "item2" in outtype:        
        with open(os.path.join(args.outputdir, "%s_2.txt" % args.outfn_prefix), 'w') as fh1:
            fh1.write(b2_str)    

    if "item3" in outtype:        
        with open(os.path.join(args.outputdir, "%s_3.txt" % args.outfn_prefix), 'w') as fh1:
            fh1.write(b3_str)

    if "item4" in outtype:        
        with open(os.path.join(args.outputdir, "%s_4.txt" % args.outfn_prefix), 'w') as fh1:
            fh1.write(b4_str)

    if "item5" in outtype:        
        with open(os.path.join(args.outputdir, "%s_5.txt" % args.outfn_prefix), 'w') as fh1:
            fh1.write(b5_str)

    if "item6" in outtype:        
        with open(os.path.join(args.outputdir, "%s_6.txt" % args.outfn_prefix), 'w') as fh1:
            fh1.write(b6_str)

    if "item7" in outtype:        
        with open(os.path.join(args.outputdir, "%s_7.txt" % args.outfn_prefix), 'w') as fh1:
            fh1.write(b7_str)

    if "item8" in outtype:        
        with open(os.path.join(args.outputdir, "%s_8.txt" % args.outfn_prefix), 'w') as fh1:
            fh1.write(b8_str)

    if "item9" in outtype:        
        with open(os.path.join(args.outputdir, "%s_9.txt" % args.outfn_prefix), 'w') as fh1:
            fh1.write(b9_str)

    if "item10" in outtype:        
        with open(os.path.join(args.outputdir, "%s_10.txt" % args.outfn_prefix), 'w') as fh1:
            fh1.write(b10_str)

    if "item11" in outtype:        
        with open(os.path.join(args.outputdir, "%s_11.txt" % args.outfn_prefix), 'w') as fh1:
            fh1.write(b11_str)

    if "item12" in outtype:        
        with open(os.path.join(args.outputdir, "%s_12.txt" % args.outfn_prefix), 'w') as fh1:
            fh1.write(b12_str)

    if "item13" in outtype:        
        with open(os.path.join(args.outputdir, "%s_13.txt" % args.outfn_prefix), 'w') as fh1:
            fh1.write(b13_str)

    if "item14" in outtype:        
        with open(os.path.join(args.outputdir, "%s_14.txt" % args.outfn_prefix), 'w') as fh1:
            fh1.write(b14_str)    


def parse_edgar_header(sec_header):
    """
    Parse EDGAR SEC filing header to extract metadata.

    Extracts company information and filing details from the standardized
    header section of SEC filings.

    Args:
        sec_header: String containing the EDGAR header section

    Returns:
        Dictionary with keys:
            - cname: Company conformed name
            - ftype: Filing type (e.g., '10-K', '10-Q')
            - cpr: Conformed period of report (YYYYMMDD)
            - sic_desc: Standard Industrial Classification description
            - sic_code: SIC code (integer, -1 if not found)
    """
    # Regular expressions for extracting header fields
    re0 = re.compile('CONFORMED SUBMISSION TYPE:\s+(\S+)')
    re1 = re.compile('CONFORMED PERIOD OF REPORT:\s+(\d+)')
    re2 = re.compile('FILED AS OF DATE:\s+(\d+)')
    re3 = re.compile('DATE AS OF CHANGE:\s+(\d+)')
    re4 = re.compile('STANDARD INDUSTRIAL CLASSIFICATION:\s*(.*?)\s+\\[(\d+)\\]')
    re5 = re.compile('COMPANY CONFORMED NAME:(.*)')
    m0 = re0.findall(sec_header)
    m1 = re1.findall(sec_header)
    m2 = re2.findall(sec_header)
    m3 = re3.findall(sec_header)
    if len(m1) != 1:
      date1 = '19000101'
    else:
      date1 = m1[0]
    if len(m0) != 1:
        thetype = "Unknown"
    else:
        thetype = m0[0]
    if len(m2) != 1:
      date2 = '19000101'
    else:
      date2 = m2[0]
    if len(m3) != 1:
      date3 = '19000101'
    else:
      date3 = m3[0]
    m4=re4.findall(sec_header)
    if len(m4) < 1:
      sich_desc = "N/A"
      sich = -1
    else:
      sich_desc = m4[0][0]
      try:
        sich = int(m4[0][1])
      except:
        sich= -1
    m5 = re5.findall(sec_header)
    if len(m5) < 1:
        cname = "Unknown"
    else:
        cname = m5[0]
        cname = cname.strip()
    # if args.verbose >= 1:
    #     print(f"Company Name = {cname}")
    #     print(f"File type = {thetype}")
    #     print(f"Confirmed period of report = {date1}")
    #    print(f"Industry: {sich_desc} - {sich}")
        
    return {'cname': cname, 
            'ftype': thetype,
            'cpr': date1,
            'sic_desc': sich_desc,
            'sic_code': sich}    
        
def item_extract(lines, tags, target_tags):
    """
    Extract lines that match specified target tags.

    Args:
        lines: List of text lines
        tags: List of tags corresponding to each line
        target_tags: List of tags to extract (e.g., ['B1', 'I1'])

    Returns:
        Tuple of (extracted_text, num_lines):
            - extracted_text: Newline-joined string of matching lines
            - num_lines: Number of lines extracted
    """
    outtext = []
    for i, atag in enumerate(tags):
        if atag in target_tags:
            outtext.append(lines[i])
    return "\n".join(outtext), len(outtext)

def word_len(text):
    """
    Count the number of meaningful words in text (excluding pure punctuation).

    Args:
        text: Input string

    Returns:
        Integer count of words containing at least one letter or digit
    """
    nonPunct = re.compile('.*[A-Za-z0-9].*')  # must contain a letter or digit
    filtered = [w for w in tokenize.word_tokenize(text) if nonPunct.match(w)]
    wlen = len(filtered)
    return wlen

def translate2ascii(text):
    """
    Convert Unicode text to ASCII by replacing common special characters.

    Handles various Unicode characters commonly found in SEC filings:
    - Smart quotes to straight quotes
    - Em/en dashes to hyphens
    - Accented characters to base letters
    - Trademark symbols to (TM), (R), (C)
    - Non-ASCII punctuation and symbols

    Args:
        text: Byte string containing Unicode text

    Returns:
        ASCII-compatible string with special chars replaced
    """
    # to string
    clean_text2 = text.decode('utf-8')
    clean_text2 = clean_text2.replace(u"’", "'")
    clean_text2 = clean_text2.replace(u"“", '"')
    clean_text2 = clean_text2.replace(u"”", '"')
    clean_text2 = clean_text2.replace(u"•", '*')
    clean_text2 = clean_text2.replace(u"§", 'SS')
    clean_text2 = clean_text2.replace(u"—", '-')
    clean_text2 = clean_text2.replace(u"–", '-')
    clean_text2 = clean_text2.replace(u"‐", '-')
    clean_text2 = clean_text2.replace(u"®", '(R)')
    clean_text2 = clean_text2.replace(u"°", ' ')
    clean_text2 = clean_text2.replace(u"€", '$')
    clean_text2 = clean_text2.replace(u"†", '+')
    clean_text2 = clean_text2.replace(u"¨", '..')
    clean_text2 = clean_text2.replace(u"þ", ' ')
    clean_text2 = clean_text2.replace(u"‘", "'")
    clean_text2 = clean_text2.replace(u"£", " ")
    clean_text2 = clean_text2.replace(u"·", "*")
    clean_text2 = clean_text2.replace(u"©", "(C)")
    
    clean_text2 = clean_text2.replace(u"¾", "3/4")        
    clean_text2 = clean_text2.replace(u"½", "1/2")
    clean_text2 = clean_text2.replace(u"¢", "c/")
    
    clean_text2 = clean_text2.replace(u"\u0080", "(E)")
    clean_text2 = clean_text2.replace(u"\u0086", "+")
    clean_text2 = clean_text2.replace(u"\u0091", "'")
    clean_text2 = clean_text2.replace(u"\u0092", "'")
    clean_text2 = clean_text2.replace(u"\u0093", '"')
    clean_text2 = clean_text2.replace(u"\u0094", '"')
    clean_text2 = clean_text2.replace(u"\u0095", '*')
    clean_text2 = clean_text2.replace(u"\u0096", '-')
    clean_text2 = clean_text2.replace(u"\u0097", '-')
    clean_text2 = clean_text2.replace(u"\u0098", '~')
    clean_text2 = clean_text2.replace(u"\u0099", 'TM')
    
    clean_text2 = clean_text2.replace(u"\u2010", '-')
    clean_text2 = clean_text2.replace(u"\u2011", '-')
    clean_text2 = clean_text2.replace(u"\u2012", '-')
    clean_text2 = clean_text2.replace(u"\u2013", '-')
    clean_text2 = clean_text2.replace(u"­", '-')
    
    LATIN_LETTERS = {
        u'\N{LATIN SMALL LETTER DOTLESS I}': 'i',
        u'\N{LATIN SMALL LETTER S WITH CEDILLA}': 's',
        u'\N{LATIN SMALL LETTER C WITH CEDILLA}': 'c',
        u'\N{LATIN SMALL LETTER G WITH BREVE}': 'g',
        u'\N{LATIN SMALL LETTER O WITH DIAERESIS}': 'o',
        u'\N{LATIN SMALL LETTER U WITH DIAERESIS}': 'u',
        u'\N{LATIN SMALL LETTER A WITH GRAVE}' : 'a',
        u'\N{LATIN SMALL LETTER A WITH ACUTE}' : 'a',
        u'\N{LATIN SMALL LETTER A WITH CIRCUMFLEX}' : 'a',
        u'\N{LATIN SMALL LETTER A WITH TILDE}' : 'a',
        u'\N{LATIN SMALL LETTER A WITH DIAERESIS}' : 'a',
        u'\N{LATIN SMALL LETTER A WITH RING ABOVE}' : 'a',
        u'\N{LATIN SMALL LETTER A WITH MACRON}': 'a',
        u'\N{LATIN SMALL LETTER A WITH BREVE}': 'a',
        u'\N{LATIN SMALL LETTER AE}' : 'ae',
        u'\N{LATIN SMALL LETTER E WITH GRAVE}' : 'e',
        u'\N{LATIN SMALL LETTER E WITH ACUTE}' : 'e',
        u'\N{LATIN SMALL LETTER E WITH CIRCUMFLEX}' : 'e',
        u'\N{LATIN SMALL LETTER E WITH DIAERESIS}' : 'e',
        u'\N{LATIN SMALL LETTER I WITH GRAVE}' : 'i',
        u'\N{LATIN SMALL LETTER I WITH ACUTE}' : 'i',
        u'\N{LATIN SMALL LETTER I WITH CIRCUMFLEX}' : 'i',
        u'\N{LATIN SMALL LETTER I WITH DIAERESIS}' : 'i',
        u'\N{LATIN SMALL LETTER N WITH TILDE}' : 'n',
        u'\N{LATIN SMALL LETTER O WITH GRAVE}' : 'o',
        u'\N{LATIN SMALL LETTER O WITH ACUTE}' : 'o',
        u'\N{LATIN SMALL LETTER O WITH CIRCUMFLEX}' : 'o',
        u'\N{LATIN SMALL LETTER O WITH TILDE}' : 'o',
        u'\N{LATIN SMALL LETTER O WITH STROKE}': 'o',
        u'\N{LATIN SMALL LETTER U WITH GRAVE}': 'u',
        u'\N{LATIN SMALL LETTER U WITH ACUTE}': 'u',
        u'\N{LATIN SMALL LETTER U WITH CIRCUMFLEX}': 'u',
        u'\N{LATIN SMALL LETTER Y WITH ACUTE}': 'y',
        u'\N{LATIN SMALL LETTER Y WITH DIAERESIS}': 'y'
    }
    CAPITAL_LATIN_LETTERS = {
        u'\N{LATIN CAPITAL LETTER I WITH DOT ABOVE}': 'I',
        u'\N{LATIN CAPITAL LETTER S WITH CEDILLA}': 'S',
        u'\N{LATIN CAPITAL LETTER C WITH CEDILLA}': 'C',
        u'\N{LATIN CAPITAL LETTER G WITH BREVE}': 'G',
        u'\N{LATIN CAPITAL LETTER O WITH DIAERESIS}': 'O',
        u'\N{LATIN CAPITAL LETTER U WITH DIAERESIS}': 'U',
        u'\N{LATIN CAPITAL LETTER A WITH GRAVE}' : 'A',
        u'\N{LATIN CAPITAL LETTER A WITH ACUTE}' : 'A',
        u'\N{LATIN CAPITAL LETTER A WITH CIRCUMFLEX}' : 'A',
        u'\N{LATIN CAPITAL LETTER A WITH TILDE}' : 'A',
        u'\N{LATIN CAPITAL LETTER A WITH DIAERESIS}' : 'A',
        u'\N{LATIN CAPITAL LETTER A WITH RING ABOVE}' : 'A',
        u'\N{LATIN CAPITAL LETTER A WITH MACRON}': 'A',
        u'\N{LATIN CAPITAL LETTER A WITH BREVE}': 'A',
        u'\N{LATIN CAPITAL LETTER AE}' : 'AE',
        u'\N{LATIN CAPITAL LETTER E WITH GRAVE}' : 'E',
        u'\N{LATIN CAPITAL LETTER E WITH ACUTE}' : 'E',
        u'\N{LATIN CAPITAL LETTER E WITH CIRCUMFLEX}' : 'E',
        u'\N{LATIN CAPITAL LETTER E WITH DIAERESIS}' : 'E',
        u'\N{LATIN CAPITAL LETTER I WITH GRAVE}' : 'I',
        u'\N{LATIN CAPITAL LETTER I WITH ACUTE}' : 'I',
        u'\N{LATIN CAPITAL LETTER I WITH CIRCUMFLEX}' : 'I',
        u'\N{LATIN CAPITAL LETTER I WITH DIAERESIS}' : 'I',
        u'\N{LATIN CAPITAL LETTER N WITH TILDE}' : 'N',
        u'\N{LATIN CAPITAL LETTER O WITH GRAVE}' : 'O',
        u'\N{LATIN CAPITAL LETTER O WITH ACUTE}' : 'O',
        u'\N{LATIN CAPITAL LETTER O WITH CIRCUMFLEX}' : 'O',
        u'\N{LATIN CAPITAL LETTER O WITH TILDE}' : 'O',
        u'\N{LATIN CAPITAL LETTER O WITH STROKE}': 'O',
        u'\N{LATIN CAPITAL LETTER U WITH GRAVE}': 'U',
        u'\N{LATIN CAPITAL LETTER U WITH ACUTE}': 'U',
        u'\N{LATIN CAPITAL LETTER U WITH CIRCUMFLEX}': 'U',
        u'\N{LATIN CAPITAL LETTER Y WITH ACUTE}': 'Y',
        u'\N{LATIN CAPITAL LETTER Y WITH DIAERESIS}': 'Y'
    }
    for key, value in LATIN_LETTERS.items():
        # print(f"{key} --> {value}")
        clean_text2 = clean_text2.replace(key, value)
    for key, value in CAPITAL_LATIN_LETTERS.items():
        # print(f"{key} --> {value}")
        clean_text2 = clean_text2.replace(key, value)
    # last resort
    # clean_text2 = unicodedata.normalize('NFKD', clean_text2).encode('ascii', 'replace')
    clean_text2 = unicodedata.normalize('NFKD', clean_text2).encode('ascii', 'xmlcharrefreplace')
    # replace all remaining non-ascii char to space, format example: &#1086;
    clean_text2 = re.sub('&#\d+;', ' ', clean_text2.decode('ascii'))
    # for debugging only, easy identification of missed unicode characters
    # clean_text2 = re.sub('&#\d+;', '????', clean_text2.decode('ascii'))
    return clean_text2
    
# import pdb

def pretty_text(pure_text1):
    """
    Clean and filter text by removing table fragments, noise, and repetitive lines.

    This function applies heuristic filters to remove non-content lines such as:
    - Table rows with excessive spaces/punctuation
    - Lines with many uppercase characters (likely headers/tables)
    - Lines with high ratio of numbers (likely tables)
    - Frequently repeated lines (likely boilerplate)

    Args:
        pure_text1: Input text string with newlines

    Returns:
        Cleaned text string with noise lines replaced by empty lines
    """
    lines = pure_text1.split('\n')
    line_count = 0
    all_lines_clean = []

    empty_linm1 = 0  # Track consecutive empty lines
    for aline in lines:
        aline_orig = aline.strip()
        

        aline = aline.strip().lower()
        if len(aline) > 2:
            line_count = line_count + 1
            # lineword = word_tokenizer.tokenize(aline)
            lineword = word_tokenizer(aline)
            wordlen = 0
            floatlen = 0
            zero_rate = (aline.count(' ') + aline.count('-') + aline.count('=') + aline.count('_') + 0.0) / len(aline)
            # item_count = aline.count('item')
            # should not delete lines containing item or none.
            item_count = aline.count('item') + aline.count('none')
            lead_lower_digit = 0
            if aline_orig[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                                               'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                                               'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                                               'w', 'x', 'y', 'z']:
                lead_lower_digit = 1

            
            m1 = punc_splus.findall(aline_orig)
            uppercount = sum(1 for elem in aline_orig if elem.isupper())
            # plen is numbe of surplus space (and other punctuations)
            plen = 0
            for tmp1 in m1:
                # print(tmp1)
                plen += (len(tmp1)-1)

            lineword_clean = []
            for aword in lineword:
                #if (len(aword) < 25) and (aword not in eng_punc) and (aword not in eng_stop):
                if (len(aword) < 25):
                    #looks like a word
                    rem1 = re_float.findall(aword)
                    wordlen = wordlen + 1
                    if len(rem1) > 0:
                        #we have a float number
                        floatlen = floatlen + 1
                        lineword_clean.append("NumWord")
                    elif aword in eng_punc:
                        # floatlen = floatlen + 1
                        wordlen = wordlen - 1
                    else:
                        pass

            #print "  Summary: seq=%d; %d (float) out of %d words" % (seq[0], floatlen, wordlen)
            # if aline_orig.find("The Company has a large") >= 0:
            #     print(f" -- {aline_orig}")
            #     pdb.set_trace()

            if wordlen > 0:
                float_ratio = float(floatlen) / wordlen
            else:
                float_ratio = 0.0

            # keep short lines (wordlen <=4); 2021-8-25 (side effect is many table residual, not doing this for now
            if (item_count == 0) and ((plen >= 10) or (uppercount >= 25)):
                # if many surplus space or many uppercase characters, reject the line
                if (wordlen > 50):
                    all_lines_clean.append(aline_orig)
                    empty_linm1 = 0
                else:
                    all_lines_clean.append("")
                    # print(f"rejected {aline_orig}")
                    empty_linm1+=1
            elif plen == 0:
                # no extra space, accept
                all_lines_clean.append(aline_orig)
                empty_linm1=0
            elif (plen <= 3) and (uppercount < 3):
                # if no or few surplus space and few uppercase characters, accept the line
                all_lines_clean.append(aline_orig)
                empty_linm1=0
            else:
                # if something in between in terms of surplus space and uppercase character, look at other features. 
                if (item_count > 0) or (lead_lower_digit > 0 and len(aline_orig) <= 20) or ((float_ratio < 0.5 and wordlen > 0) and (zero_rate < 0.3)):
                    all_lines_clean.append(aline_orig)
                    empty_linm1=0
                    #print "zero-rate=%f" % zero_rate
                else:
                    # print ("reject (%d/%d; zero=%f): %s (%s)" % (floatlen, wordlen, zero_rate, aline.strip(), lineword))
                    # print(f"rejected {aline_orig}")
                    if(empty_linm1==0):
                        all_lines_clean.append("")
                    empty_linm1+=1

            #if float_ratio > 0.5:
            #    raw_input("Look at this line!")
        else:
            #the case when this line does not have much...
            if(empty_linm1==0):
                all_lines_clean.append("")
                # print(f"rejected {aline_orig}")
            empty_linm1 +=1
            #pass

    # find duplicated lines in all_lines_clean
    lcount = collections.Counter(all_lines_clean)
    lcount2 = lcount.most_common(50)
    line2remove = []
    for k, v in lcount2:
        if len(k) > 1 and k.lower().find("none") < 0 and k.lower().find("item") < 0 and k.lower().find("not applicable") < 0 and v >= 15:
            line2remove.append(k)
    # print("High frequency lines: ", lcount2[0:10])    
    # print("line2remove = ", line2remove)
    
    line2remove = set(line2remove)
    all_lines_clean2 = []
    for aline in all_lines_clean:
        if aline not in line2remove:
            all_lines_clean2.append(aline)
        else:
            all_lines_clean2.append("")

    pure_text2= "\n".join(all_lines_clean2)
    pure_text2 = pure_text2.strip()
    return(pure_text2)


class BiLSTM_Tok(nn.Module):
    """
    Bidirectional LSTM with attention for sequence tagging of SEC filing items.

    This model processes word embeddings through a BiLSTM and applies attention
    mechanisms to aggregate token-level representations into line-level representations.

    Architecture:
        Input -> BiLSTM -> Attention -> Linear -> Output tags

    Supports three attention methods:
        - "complete": Full attention mechanism with learned weights
        - "simple": Use only the first token (linehead) of each line
        - "simple2": Use the second token of each line
    """
    def __init__(self, input_dim, tag_to_ix, hidden_dim,
                 device, attention_method="complete", num_layers=1):
        """
        Initialize the BiLSTM model with attention.

        Args:
            input_dim: Dimension of input features (word embeddings + position features)
            tag_to_ix: Dictionary mapping tags to indices (label vocabulary)
            hidden_dim: Hidden dimension size for LSTM (will be split bidirectionally)
            device: PyTorch device (cuda/cpu)
            attention_method: Attention type - "complete", "simple", or "simple2"
            num_layers: Number of LSTM layers (default=1)
        """
        super(BiLSTM_Tok, self).__init__()
        self.hidden_dim = hidden_dim
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix) # the output size
        self.att_method = attention_method
        self.num_layers = num_layers
        self.device = device
        
        # print(f"    BiLSTM_Tok input dim={input_dim}")
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2,  # we uses input_dim instead of embedding_dim
                            num_layers=num_layers, bidirectional=True)
        
        # map LSTM output to label space
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # attention net parameters        
        self.w_omega = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim)) # torch.Size([256, 256])
        self.u_omega = nn.Parameter(torch.Tensor(hidden_dim, 1))# u_omega([256, 1])
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)        
    def attention_net(self, x, doc_mask):
        """
        Apply attention mechanism to aggregate token representations into line representations.

        Args:
            x: Token-level LSTM outputs, shape [total_doc_tokens, 1, hidden_dim]
            doc_mask: Tuple of indices marking end positions of each line

        Returns:
            Line-level representations, shape [num_lines, hidden_dim]

        Methods:
            - "complete": Full attention with learned weights (softmax over tokens in each line)
            - "simple": Take only the first token (linehead) of each line
            - "simple2": Take the second token of each line
        """
        if self.att_method == "complete":
            # x: torch.Size([26833, 1, 128])
            # w_omega: # torch.Size([128, 128])
            u = torch.tanh(torch.matmul(x, self.w_omega))  # torch.Size([26833, 1, 128])       #[batch, seq_len, hidden_dim] # torch.Size([6040, 1, 256])
            att = torch.matmul(u, self.u_omega)   # torch.Size([26833, 1, 1])                #[batch, seq_len, 1]  #[batch, seq_len, 1] # torch.Size([6040, 1, 1])
            att_score = []
            for index in range(len(doc_mask)):
                if index == 0:
                    att_score.append(F.softmax(att[0: doc_mask[index]], dim=0) )
                else:
                    att_score.append(F.softmax(att[doc_mask[index-1]: doc_mask[index]], dim=0) )
            # the last line
            att_score.append( F.softmax(att[doc_mask[-1]:], dim=0))
            att_score = torch.cat(att_score)  # torch.Size([26830, 1, 1])
            assert att_score.shape[0] == x.shape[0]
            # before: [216 2(word len in each sentence) 1] after: [6040 1 1] do softmax line by line, should be the same as by document
            
            # x (doc_word_len,1,hidden_dim)   att_score(doc_word_len,1,1)
            scored_x = x * att_score  #[batch, seq_len, hidden_dim] 6040 1 256
            scored_x_2 = torch.tensor_split(scored_x, doc_mask) # 216 2 1 256

            context =[]
            for i in scored_x_2:
                context.append(torch.sum(i.squeeze(dim=1), dim=0)) # dim=1 or may happen unexpected error # 256
            context = torch.stack(context)
            # print(f"attention_net: context.shape = {context.shape}") # torch.Size([2165, 128])
            
            return context #[batch, hidden_dim][216 256]
        elif self.att_method == "simple":
            # just take the linehead
            # print(f"attention_net: doc_mask len = {len(doc_mask)}")
            # print(f"attention_net: x.shape = {x.shape}") # torch.Size([26833, 1, 128])
            scored_x_2 = torch.tensor_split(x, doc_mask)
            # print(f"attention_net: len of scored_x_2 = {len(scored_x_2)}") # len of scored_x_2 = 2166
            # print(f"attention_net: scored_x_2[0].shape = {scored_x_2[0].shape}") # torch.Size([3, 1, 128])

            context =[]
            for i in scored_x_2:
                # print(f"attention_net: i shape = {i.shape}")
                context.append(i[0]) 
            # context = torch.stack(context)
            context = torch.squeeze(torch.stack(context))
            # print(f"attention_net: context.shape = {context.shape}") # context.shape = torch.Size([2165, 1, 128])
            
            return context 
        elif self.att_method == "simple2":
            # Take the first token next to the linehead
            # print(f"attention_net: doc_mask len = {len(doc_mask)}")
            # print(f"attention_net: x.shape = {x.shape}") # torch.Size([26833, 1, 128])
            scored_x_2 = torch.tensor_split(x, doc_mask)
            # print(f"attention_net: len of scored_x_2 = {len(scored_x_2)}") # len of scored_x_2 = 2166
            # print(f"attention_net: scored_x_2[0].shape = {scored_x_2[0].shape}") # torch.Size([3, 1, 128])

            context =[]
            for i in scored_x_2:
                # print(f"attention_net: i shape = {i.shape}")
                # print(f"attention_net: i[1] shape = {i[1].shape}")
                # print(f"attention_net: i[0] shape = {i[0].shape}")
                context.append(i[1]) 
            # print(f" len of context is {len(context)}")
            
            # print(f"after stack attention_net: context.shape = {context.shape}")
            
            context = torch.squeeze(torch.stack(context))
            # print(f"after sequeeze attention_net: context.shape = {context.shape}") # context.shape = torch.Size([2165, 1, 128])
            return context        
        else:
            raise(Exception(f"attention_net: method {method} not defined"))
    
    def init_hidden(self):
        # (num_layers, nums_directions,minibatch_size, hidden_dim, device)
        
        return (torch.randn(2 * self.num_layers, 1, self.hidden_dim // 2, device=self.device),
                torch.randn(2 * self.num_layers, 1, self.hidden_dim // 2, device=self.device))
    
    
    def forward(self, sentence, doc_mask):
        # handle the dimension issue
        sentence = sentence.view(sentence.size()[0],1,sentence.size()[1]) # torch.Size([document length, 1, features])
        self.hidden = self.init_hidden() # initial hidden state
        # embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(sentence, self.hidden) # last hidden state
        lstm_out = lstm_out.squeeze() # word * hidden size
        lstm_out = lstm_out.unsqueeze(1)
    
        attn_output = self.attention_net(lstm_out, doc_mask) #lstm_out: [batch, seq_len, hidden_dim*2] 
        #      # [batch, seq_len, hidden_dim*2] 
        #        #     should be # of lines; word length; hidden; but seq length is not fixed; handle this at att_net        #      
        # lstm_out = attn_output # [batch, hidden_dim]

        lstm_feats = self.hidden2tag(attn_output) # lstm_out [seq_len, hidden_dim]
        return lstm_feats
        
def my_word_tokenizer(cur_sent, method="split",
                      trun_line_len=0, addheader="_LINEHEAD_"):
    """
    Tokenize a sentence with optional header addition and truncation.

    Args:
        cur_sent: Input sentence string
        method: Tokenization method - "split" (whitespace) or "word_tokenizer" (NLTK)
        trun_line_len: Truncate to this many tokens (0 = no truncation)
        addheader: Header token to prepend (empty string = no header)

    Returns:
        List of tokens
    """
    if addheader != "":
        cur_sent = addheader + " " + cur_sent

    if method == "word_tokenizer":
        alltoks = word_tokenize(cur_sent)
    elif method == "split":
        alltoks = cur_sent.split()
    else:
        raise(Exception(f"unsupported tokenizer method {method}"))

    if trun_line_len > 0:
        alltoks = alltoks[0:trun_line_len]

    return alltoks


def gen_doc_feature(lines, word2vec_model):
    """
    Generate word embedding features for a document with position information.

    Converts lines to sequences of word embeddings augmented with positional features.

    Args:
        lines: List of text lines
        word2vec_model: Gensim Word2Vec model with .wv attribute

    Returns:
        Tuple of (doc_embed, doc_mask):
            - doc_embed: List of embeddings, shape [total_tokens, embedding_dim+3]
                         Each embedding includes [sent_pos, sent_pos_back, tok_pos, word_vec...]
            - doc_mask: Tuple of indices marking the end position of each line
    """
    doc_embed = []
    doc_mask = []

    nline_doc = len(lines)
    # sen_pos_percentile
    # sen_pos_percentile.append(int(100 * sen_i / len(doc)))
    # back sen_pos_percentile
    # back_sen_pos_percentile.append(int(100 * (len(doc)-sen_i) / len(doc)))
    
    for sent_index, cur_sent in enumerate(lines):
        sentence_embed = []
        # if hyperparameters['remove_punc']:
        #     cur_sent = cur_sent.translate(str.maketrans('', '', string.punctuation))
        # if hyperparameters['add_spaces_between_punc']:
        #     cur_sent = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", cur_sent)
        # if hyperparameters['to_lower_case']:
        #      cur_sent = cur_sent.lower()
        
        sent_pos = sent_index / nline_doc
        sent_pos_back = 1 - sent_pos
        
        
        # word_tokenize
        # alltoks = word_tokenize(cur_sent)
        alltoks = my_word_tokenizer(cur_sent, trun_line_len = 100)
        # for cur_index, cur_word in enumerate(cur_sent.split()):
        for cur_index, cur_word in enumerate(alltoks):
            if cur_word in word2vec_model.wv:
                # print(cur_word + " ")
                word_embed = word2vec_model.wv[cur_word]
                tok_pos = min(cur_index / 20, 2)
                ext_word_embed = np.concatenate(([sent_pos, sent_pos_back, tok_pos], word_embed))
                sentence_embed.append(ext_word_embed)
        
        doc_embed.extend(sentence_embed)
        doc_mask.append(len(doc_embed)) # record the ending position, e.g. 5,20,30,..., the last word is also a ending position                
    x = doc_embed
    # remove the last one so that tensor_split can function correctly
    doc_mask = doc_mask[0:-1]
    doc_mask = tuple(doc_mask)
    
    return x, doc_mask
            
    

# Bi-LSTM
class BiLSTM2(nn.Module):
    """
    Bidirectional LSTM for sequence tagging with batch processing support.
    (To be coupled with BERT)

    This model processes batches of sequences through a BiLSTM and applies
    a linear layer for tag prediction. Supports packed sequences for efficient
    handling of variable-length inputs.

    Architecture:
        Input -> BiLSTM (with dropout) -> Linear -> Output tags
    """
    def __init__(self, input_dim, tag_to_ix, hidden_dim, device, batch_size=4, num_layers=1, dropout=0.5):
        """
        Initialize the BiLSTM model.

        Args:
            input_dim: Dimension of input features
            tag_to_ix: Dictionary mapping tags to indices (label vocabulary)
            hidden_dim: Hidden dimension size for LSTM (split bidirectionally)
            device: PyTorch device (cuda/cpu)
            batch_size: Default batch size (default=4)
            num_layers: Number of LSTM layers (default=1)
            dropout: Dropout probability (default=0.5)
        """
        super(BiLSTM2, self).__init__()
        self.hidden_dim = hidden_dim
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix) # the output size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device
        # self.embedding_dim = embedding_dim
        # self.vocab_size = vocab_size
        # self.adjust_input_dim = nn.Linear(input_dim, 803)  # adjust 768 to 803
        ## dimension: batch_size, num_line, dim        
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2,  # we uses input_dim instead of embedding_dim
                            num_layers = num_layers, bidirectional = True, 
                            batch_first = True, dropout = dropout)


        # Map LSTM output to label space
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
        lstm_out, self.hidden = self.lstm(sentence, self.hidden) # final output

        if isinstance(sentence, PackedSequence):
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        lstm_feats = self.hidden2tag(self.dropout(lstm_out))
        return lstm_feats





# criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)
criterion = nn.CrossEntropyLoss()

def train(subtrain_x, subtrain_y, valid_x, valid_y,
          model, hyperparameters, device,
          model_outdir="./segmodels/bilstm_crf_att_wordebd_0/"):
    """
    Train the sequence tagging model with validation and early stopping.

    Trains the model for multiple epochs, validates after each epoch, saves the best model,
    and implements early stopping based on validation accuracy.

    Args:
        subtrain_x: List of training documents (each is a list of lines)
        subtrain_y: List of training labels (each is a list of tags per line)
        valid_x: List of validation documents
        valid_y: List of validation labels
        model: PyTorch model to train
        hyperparameters: Dict containing:
            - n_epochs: Maximum number of training epochs
            - optimizer: Optimizer name (e.g., 'Adam')
            - optim_hparas: Optimizer parameters (e.g., {'lr': 0.001})
            - early_stop: Early stopping patience
        device: PyTorch device (cuda/cpu)
        model_outdir: Directory to save model checkpoints

    Returns:
        Tuple of (best_valid_acc, best_model_filename, best_status, avg_epoch_time)
    """
    n_epochs = hyperparameters['n_epochs']

    # optimizer
    optimizer = getattr(torch.optim, hyperparameters['optimizer'])(
        model.parameters(), **hyperparameters['optim_hparas'])
    print(f"--- Optimizer is: {optimizer}")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 
                   factor=0.5, patience=2, verbose=True) # looking at micro-averaging accuracy
    
    # min_ce = float('inf')
    max_valid_acc = -1
    best_epoch = -1
    best_status = []
    loss_record = {'train': [], 'valid': []}  # for recording training, valid loss
    early_stop_count = 0
    epoch = 0
    running_loss = 0.0
    running_loss_all = 0.0
    running_loss_count = 0
    # ts_train_start = datetime.now()
    status_record = [] # list of dict
    gt_epoch = lib10kq.gclock(total_laps = n_epochs, prefix = "Epoch Timer")    
    while epoch < n_epochs:
        ts_train_epoch_start = datetime.now()
        gt_intra = lib10kq.gclock(total_laps = len(subtrain_x), prefix = "    Batch timer")
        
        model.train()
        # count = len(subtrain_x)
        for batch_i, (x, y) in enumerate(zip(subtrain_x, subtrain_y)):
            
            if batch_i > 0 and batch_i % 300 == 0:
                gt_intra.mark_lap(increment=300)
                gt_intra.report()
                print(f"        Avg running loss = {running_loss/running_loss_count:.4f}")
                running_loss = 0.0
                running_loss_count = 0

            optimizer.zero_grad()
            
            # !!! convert to a function: gen_doc_feature()
            # processing x to word2vec type (if needed)
            x, doc_mask = gen_doc_feature(x)
            
            x, y = torch.tensor(np.array(x)), torch.tensor(np.array(y))
            x, y = x.float(), y   
            x, y = x.to(device), y.to(device=device, dtype=torch.int64)
            model.zero_grad()

            lstm_feat = model(x, doc_mask)
            loss = criterion(lstm_feat, y)
            # Step 4. Compute the loss, gradients, and update the parameters by
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_loss_all += loss.item()
            running_loss_count += 1
            

        # train_ce_all = valid_and_test(subtrain_x, subtrain_y, model, hyperparameters, device)
        # train_ce_all = 1 # to reduce computational time
        # loss_record['train'].append(train_ce_all)
        running_loss_all = running_loss_all / len(subtrain_x)
        print(f"    ** E{epoch} Training loss (ce) = {running_loss_all:.4f}")
        
        gt_valid = lib10kq.gclock(total_laps = 1, prefix = "    Valid Performance Timer")
        print("    -- Start doing validation...")
        # do this every epoch
        # print(f"   {datetime.now()}: training completed")
        # valid_ce = valid_and_test(valid_x, valid_y, model, hyperparameters, device)
        valid_acc, valid_acc_macro, \
            valid_ce, core_f1, all_f1, \
                preds, targets, tmpperf = actual_vs_pred(valid_x, valid_y, model, 
                                                       hyperparameters, device)
        scheduler.step(valid_acc)
        tmp_lr = optimizer.param_groups[0]['lr']
        print(f"    New learning rate = {tmp_lr}")
        print(f"    ** E{epoch} Valid: Micro acc = {valid_acc:.4f}; Macro Acc = {valid_acc_macro:.4f}; Valid Cross Entropy = {valid_ce:.8f}")
        print(f"    ** E{epoch} Valid: Core f1 = {core_f1:.4f}; All f1 = {all_f1:.4f}")
        
        gt_valid.mark_lap()
        gt_valid.report()
        del gt_valid

        # record the status
        tmp_status = {'expname': args.mn_prefix, 
                      'epoch': epoch,
                      'train_ce': running_loss_all,
                      'valid_ce': valid_ce,
                      'macro_acc': valid_acc_macro,
                      'micro_acc': valid_acc,
                      'core_f1': core_f1,
                      'all_f1': all_f1,
                      'timestamp': datetime.now()}
        status_record.append(tmp_status)
        # output to csv
        status_outfn = os.path.join(model_outdir, f"{args.mn_prefix}_train_status.csv")
        print(f"   Write status to {status_outfn}")
        status_pd = pd.DataFrame(status_record)
        status_pd.to_csv(status_outfn)



        # epoch_time_elapsed = (datetime.now() - ts_train_start).total_seconds() / 60 # converted to mins
        # epoch_total_est = epoch_time_elapsed / (epoch + 1) * n_epochs
        # epoch_time_remain = epoch_total_est - epoch_time_elapsed
        # print(f"    Elapsed time = {epoch_time_elapsed:.2f} mins")
        # print(f"    Estimated total time = {epoch_total_est:.2f} mins")
        # print(f"    Estimated time remaining = {epoch_time_remain:.2f} mins")
        # print(f"    Elapsed time/Est. Total time/ Est. remain time = "\
        #       f"{epoch_time_elapsed:.2f}/{epoch_total_est:.2f}/{epoch_time_remain:.2f} mins")
        
        if valid_acc > max_valid_acc:
            # Save model if your model improved
            max_valid_acc = valid_acc
            best_epoch = epoch
            # if epoch % 5 == 0:
            # valid_acc, prei, tari = actual_vs_pred(valid_x, valid_y, model, hyperparameters, device)            
            # outfn = os.path.join(model_outdir, 'word_base_word2vec_LSTM' + str(valid_ce) + '.pth')
            outfn = os.path.join(model_outdir, f"{args.mn_prefix}_e{epoch:03d}_vac{valid_acc*100:.2f}_vce{valid_ce:.5f}.pth")
            print(f"    Saving model to {outfn}")
            torch.save(model.state_dict(), outfn)  # Save model
            early_stop_count = 0
            best_model_fn = outfn
            best_status = tmp_status
        else:
            early_stop_count += 1
            # print(f"   minimal_ce = {min_ce} while this ce = {valid_ce}")
            print(f'No improvement! Best performance at Epoch = {best_epoch:03d} with Micro acc = {max_valid_acc:.4f}; early stop count = {early_stop_count}')
            

        epoch += 1
        gt_epoch.mark_lap()
        tmp = gt_epoch.report()
        # loss_record['valid'].append(valid_ce)
        if early_stop_count > hyperparameters['early_stop']:
            print("Early stopping criteria matched.")
            break

    print('train finished')
    return max_valid_acc, best_model_fn, best_status, tmp['avg_time']

def valid_and_test(valid_x, valid_y, model, hyperparameters, device):
    model.eval()
    total_loss = 0
    for x, y in zip(valid_x, valid_y):
        # processing x to word2vec type
        x, doc_mask = gen_doc_feature(x)
        
        x, y = torch.tensor(np.array(x)), torch.tensor(np.array(y))
        x, y = x.float(), y
        x, y = x.to(device), y.to(device=device, dtype=torch.int64)
        with torch.no_grad():                  # disable gradient calculation
            ce_loss = model.neg_log_likelihood(x, y, doc_mask).to(device)
        total_loss += ce_loss.detach().cpu().item()  # * len(x) # accumulate loss
    total_loss = total_loss / len(valid_x)      # compute averaged loss
    return total_loss        

def actual_vs_pred(valid_x, valid_y, model, hyperparameters, device):
    """
    Evaluate model predictions against ground truth labels.

    Computes per-document and aggregate performance metrics including accuracy,
    precision, recall, and F1 scores for each item type.

    Args:
        valid_x: List of validation documents (each is a list of lines)
        valid_y: List of validation labels (each is a list of tags per line)
        model: PyTorch model to evaluate
        hyperparameters: Dictionary of hyperparameters
        device: PyTorch device (cuda/cpu)

    Returns:
        Tuple of (micro_acc, macro_acc, total_loss, core_f1, all_f1, preds, targets, perf_details):
            - micro_acc: Micro-averaged accuracy across all lines
            - macro_acc: Macro-averaged accuracy across documents
            - total_loss: Average cross-entropy loss
            - core_f1: F1 score for core items (1, 1A, 3, 5, 7, 7A, 8, 10, 11)
            - all_f1: F1 score averaged across all items
            - preds: List of predicted tag sequences
            - targets: List of ground truth tag sequences
            - perf_details: List of per-document performance dictionaries
    """
    model.eval()
    preds, targets = [], []
    total_counts = 0
    total_loss = 0
    numtesting = []
    cc = 0
    for x, y in zip(valid_x, valid_y):
        tmpuid = test_uid[cc]
        cc += 1
        # processing x to word2vec type
        yorig = y
        x, doc_mask = gen_doc_feature(x)           
        x, y = torch.tensor(np.array(x)), torch.tensor(np.array(y))
        x, y = x.float(), y
        x, y = x.to(device), y.to(device=device, dtype=torch.int64)
        # total_counts += 1
        with torch.no_grad(): 
            # path_score, pred = model(x, doc_mask)
            tmp_pred = model(x, doc_mask)
            ce_loss = criterion(tmp_pred, y)
            total_loss += ce_loss.cpu().item()
            max_pred = tmp_pred.argmax(dim = 1)
            max_pred = max_pred.cpu()
        max_pred = max_pred.tolist()
        pred_label = [reverse_label_mapping[tmp_pred] for tmp_pred in max_pred]    
        preds.append(pred_label) 
        gt_label = [reverse_label_mapping[tmp_gt] for tmp_gt in yorig]
        targets.append(gt_label)
        oneperf = lib10kq.trec_val_cross(pred_label, gt_label)
        oneperf['numline'] = len(yorig)
        oneperf['uid'] = tmpuid
        numtesting.append(oneperf)
    total_loss = total_loss / len(valid_x)
    tmpperf = []
    for adoc in numtesting:
        perf2 = collections.OrderedDict()
        # let's add this back
        perf2['uid'] = adoc['uid']
        perf2['numline'] = adoc['numline']
        for key, value in adoc.items():
            if key in ["uid", 'numline']:
                continue
            for key2, value2 in value.items():
                perf2[f"{key}_{key2}"] = value2
        tmpperf.append(perf2)
    tmppd1 = pd.DataFrame(tmpperf)
    fold_perf = tmppd1.mean()
    fold_perf = fold_perf.rename('performance')
    core_f1, all_f1 = lib10kq.f1_perf(fold_perf)
    # macro averaging

    acc_list = []
    for tmp_pred, tmp_actual in zip(preds, targets):
        acc_list.append(np.mean(np.array(tmp_pred) == np.array(tmp_actual)))
    acc_macro = mean(acc_list)

    flat_preds = np.array([p for sub in preds for p in sub])
    flat_targets = np.array([t for sub in targets for t in sub])
    acc_ = sum(flat_preds == flat_targets) / len(flat_targets)
    # total_loss = total_loss / total_counts
    return acc_, acc_macro, total_loss, core_f1, all_f1, preds, targets, tmpperf

def train_valid_test_uid_text(valid_fold, use_ntrain_fold=0):
    """
    Split data into train/valid/test sets by fold, returning UIDs and text.

    Uses 10-fold cross-validation scheme where:
    - valid_fold: Used for validation
    - test_fold: valid_fold + 1 (wraps to 0 if > 9)
    - Remaining folds: Used for training

    Args:
        valid_fold: Fold index for validation (0-9)
        use_ntrain_fold: Limit training to first N folds (0 = use all available)

    Returns:
        Tuple of (train_uid, valid_uid, test_uid, train_doc, valid_doc, test_doc)
        where each is a list of UIDs or document texts

    Note: Requires global variables copy_df and fold_index to be defined
    """
    test_fold = valid_fold + 1
    allfold = set(list(range(10)))
    if test_fold > 9: test_fold = test_fold % 10
    train_folds = allfold - set([valid_fold, test_fold])
    if use_ntrain_fold > 0:
        train_folds = list(train_folds)[0:use_ntrain_fold]
        train_folds = set(train_folds)
    print(f"Valid fold = {valid_fold}; test_fold = {test_fold}; train_folds = {train_folds}")
    
    all_text_data = copy_df['text data']
    all_uid = copy_df['uid']
    
    train_uid, valid_uid, test_uid = [], [], []
    train_doc, valid_doc, test_doc = [], [], []
    
    for aid in fold_index[valid_fold]:
        valid_uid.append(all_uid[aid])
        valid_doc.append(all_text_data[aid])
        
    for aid in fold_index[test_fold]:
        test_uid.append(all_uid[aid])
        test_doc.append(all_text_data[aid])
        
    for atrain_fold in train_folds:
        for aid in fold_index[atrain_fold]:            
            train_uid.append(all_uid[aid])
            train_doc.append(all_text_data[aid])
    return train_uid, valid_uid, test_uid, train_doc, valid_doc, test_doc


def train_valid_test(valid_fold, use_ntrain_fold=0):
    """
    Split data into train/valid/test sets by fold, returning X and Y data.

    Uses 10-fold cross-validation scheme where:
    - valid_fold: Used for validation
    - test_fold: valid_fold + 1 (wraps to 0 if > 9)
    - Remaining folds: Used for training

    Args:
        valid_fold: Fold index for validation (0-9)
        use_ntrain_fold: Limit training to first N folds (0 = use all available)

    Returns:
        Tuple of (subtrain_x, subtrain_y, valid_x, valid_y, test_x, test_y)
        where each is a list of documents/labels

    Note: Requires global variables X, Y, and fold_index to be defined
    """
    test_fold = valid_fold + 1
    allfold = set(list(range(10)))
    if test_fold > 9: test_fold = test_fold % 10
    train_folds = allfold - set([valid_fold, test_fold])
    if use_ntrain_fold > 0:
        train_folds = list(train_folds)[0:use_ntrain_fold]
        train_folds = set(train_folds)
    print(f"Valid fold = {valid_fold}; test_fold = {test_fold}; train_folds = {train_folds}")


    subtrain_x, subtrain_y = [], []
    valid_x, valid_y = [], []
    test_x, test_y = [], []


    for aid in fold_index[valid_fold]:
        valid_x.append(X[aid])
        valid_y.append(Y[aid])
        # this is incorrect
        # else:
        #     valid_x.append(X[aid][0:trun_line_len])
        #     valid_y.append(Y[aid][0:trun_line_len])

    for aid in fold_index[test_fold]:
        test_x.append(X[aid])
        test_y.append(Y[aid])
        
    for atrain_fold in train_folds:
        for aid in fold_index[atrain_fold]:            
            subtrain_x.append(X[aid])
            subtrain_y.append(Y[aid])            
    
    return subtrain_x, subtrain_y, valid_x, valid_y, test_x, test_y
    
    
    
    

class gclock():
    """
    Generic timer for tracking progress and estimating time to completion.

    Tracks elapsed time, average time per lap, and estimates total time
    when total_laps is provided. Useful for monitoring long-running operations
    like training loops.
    """
    def __init__(self, total_laps=0, prefix="GTimer", keep_all_marks=False, stime=None):
        """
        Initialize the timer.

        Args:
            total_laps: Total number of laps/iterations expected (0 for unknown)
            prefix: Prefix string for output messages
            keep_all_marks: Whether to keep all lap timestamps (default False)
            stime: Start time (default is current time)
        """
        self.prefix = prefix
        if stime == None:
            self.stime = datetime.now()
        else:
            self.stime = stime
        self.mtime = stime
        self.total_laps = total_laps
        self.lcount = 0  # Lap count
        self.avg_ltime = 0.0  # Average time per lap
        self.lmarks = []  # List of lap timestamps (if keepall=True)
        self.keepall = keep_all_marks
        self.est_total = 0  # Estimated total time
    def mark_lap(self, increment=1, mtime=None):
        """
        Mark completion of one or more laps and update timing statistics.

        Args:
            increment: Number of laps completed (default=1)
            mtime: Mark time (default is current time)
        """
        if mtime == None:
            self.mtime = datetime.now()
        else:
            self.mtime = mtime
        self.et = (self.mtime - self.stime).total_seconds() / 60  # in mins
        self.lcount += increment
        self.avg_ltime = self.et / self.lcount
        if self.keepall:
            self.lmarks.append(mtime)
        if self.total_laps > 0:
            self.est_total = self.avg_ltime * self.total_laps

    def report(self):
        """
        Print timing report with average time, elapsed time, remaining time, and total time.

        Returns:
            Dictionary with keys 'avg_time' and 'total_time'
        """
        print(f"{self.prefix}: AT/ET/RT/TT="\
              f"{self.avg_ltime:.2f}/{self.et:.2f}/"\
              f"{self.est_total-self.et:.2f}/{self.est_total:.2f} mins "\
              f"({self.et/self.est_total*100:.1f}%) or {self.lcount}/{self.total_laps}")
        return {'avg_time': self.avg_ltime, 'total_time': self.est_total}
        


def metricCal(y_pred, y_true, target):
    """
    Calculate precision, recall, and F1 for a specific item/target.

    Args:
        y_pred: List of predicted tags
        y_true: List of ground truth tags
        target: Dictionary with 'beg' and 'end' tags (e.g., {'beg': 'B1', 'end': 'I1'})

    Returns:
        Tuple of (precision, recall, f1)
    """
    pre_list = set([index for index, value in enumerate(y_pred) if value in target.values()])
    true_list = set([index for index, value in enumerate(y_true) if value in target.values()])
    precision = len(pre_list.intersection(true_list)) / len(pre_list)
    recall = len(pre_list.intersection(true_list)) / len(true_list)
    if precision + recall ==0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return(precision, recall, f1)

def metricCal2(y_pred, y_true, target):
    """
    Calculate precision, recall, F1, and accuracy for a specific item/target.

    Args:
        y_pred: List of predicted tags
        y_true: List of ground truth tags
        target: Dictionary with 'beg' and 'end' tags (e.g., {'beg': 'B1', 'end': 'I1'})

    Returns:
        Tuple of (precision, recall, f1, accuracy)
    """
    pre_list = set([index for index, value in enumerate(y_pred) if value in target.values()])
    true_list = set([index for index, value in enumerate(y_true) if value in target.values()])
    precision = len(pre_list.intersection(true_list)) / len(pre_list)
    recall = len(pre_list.intersection(true_list)) / len(true_list)
    
    y_pred_bin = []
    for vv in y_pred:
        if vv in target.values():
            y_pred_bin.append(1.)
        else:
            y_pred_bin.append(0.)
    y_pred_bin = np.array(y_pred_bin)
    
    y_true_bin = []
    for vv in y_true:
        if vv in target.values():
            y_true_bin.append(1.)
        else:
            y_true_bin.append(0.)
    y_true_bin = np.array(y_true_bin)
            
    accuracy = np.sum(y_pred_bin == y_true_bin) / y_true_bin.shape[0]
    # print(np.sum(y_pred_bin == y_true_bin),  y_true_bin.shape[0], target.values(), y_pred_bin)
    
    if precision + recall ==0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return(precision, recall, f1, accuracy)

def trec_val_cross(y_pred, y_true):
    """
    Calculate performance metrics for all item types in 10-K/10-Q filings.

    Computes precision, recall, F1, and accuracy for each item (1-15, including A/B variants).
    Handles cases where items are not present in the document.

    Args:
        y_pred: List of predicted tags for a document
        y_true: List of ground truth tags for a document

    Returns:
        Dictionary mapping item names to performance dicts with keys:
            'pre': Precision (or 1.0/0.0 for non-existent items)
            'recall': Recall (or 1.0/0.0 for non-existent items)
            'f1': F1 score (or 1.0/0.0 for non-existent items)
            'acc': Accuracy (or 1.0/0.0 for non-existent items)
    """
    alltasks = {'Item1':{'beg':'B1','end':'I1'},'Item1A':{'beg':'B1A','end':'I1A'},'Item1B':{'beg':'B1B','end':'I1B'},
                'Item2':{'beg':'B2','end':'I2'},'Item3':{'beg':'B3','end':'I3'},'Item4':{'beg':'B4','end':'I4'},
                'Item5':{'beg':'B5','end':'I5'},'Item6':{'beg':'B6','end':'I6'},'Item7':{'beg':'B7','end':'I7'},
                'Item7A':{'beg':'B7A','end':'I7A'},'Item8':{'beg':'B8','end':'I8'},'Item9':{'beg':'B9','end':'I9'},
                'Item9A':{'beg':'B9A','end':'I9A'},'Item9B':{'beg':'B9B','end':'I9B'},'Item10':{'beg':'B10','end':'I10'},
                'Item11':{'beg':'B11','end':'I11'},'Item12':{'beg':'B12','end':'I12'},'Item13':{'beg':'B13','end':'I13'},
                'Item14':{'beg':'B14','end':'I14'},'Item15':{'beg':'B15','end':'I15'}}
    # allitem = {'0':{'pre':-1,'recall':-1,'f1':-1}, '1':{'pre':-1,'recall':-1,'f1':-1}, '2':{'pre':-1,'recall':-1,'f1':-1},
    #            '3':{'pre':-1,'recall':-1,'f1':-1}, '4':{'pre':-1,'recall':-1,'f1':-1}, '5':{'pre':-1,'recall':-1,'f1':-1},
    #            '6':{'pre':-1,'recall':-1,'f1':-1}, '7':{'pre':-1,'recall':-1,'f1':-1}, '8':{'pre':-1,'recall':-1,'f1':-1},
    #            '9':{'pre':-1,'recall':-1,'f1':-1}, '10':{'pre':-1,'recall':-1,'f1':-1}, '11':{'pre':-1,'recall':-1,'f1':-1},
    #            '12':{'pre':-1,'recall':-1,'f1':-1}, '13':{'pre':-1,'recall':-1,'f1':-1}, '14':{'pre':-1,'recall':-1,'f1':-1},
    #            '15':{'pre':-1,'recall':-1,'f1':-1}, '16':{'pre':-1,'recall':-1,'f1':-1}, '17':{'pre':-1,'recall':-1,'f1':-1},
    #            '18':{'pre':-1,'recall':-1,'f1':-1}, '19':{'pre':-1,'recall':-1,'f1':-1}}
    # initialize values
    allitem = {}
    for key, value in alltasks.items():
        allitem[key] = {'pre': -1.0, 'recall': -1.0, 'f1': -1.0}
    
    for key, value in alltasks.items():
        # the model predict the focal item
        pre_index  = True if value['beg'] in y_pred or value['end'] in y_pred else False
        true_index = True if value['beg'] in y_true or value['end'] in y_true else False
        
        if not true_index:
            # if the item does not exist
            if pre_index:
                # if model labeled non-exist items, set performance to 0.0
                allitem[key]['pre'] = 0.0
                allitem[key]['recall'] = 0.0
                allitem[key]['f1'] = 0.0
                allitem[key]['acc'] = 0.0
            else:
                # if model did not label non-exist items, set performance to 1.0
                allitem[key]['pre'] = 1.0
                allitem[key]['recall'] = 1.0
                allitem[key]['f1'] = 1.0
                allitem[key]['acc'] = 1.0
        else:
            # if the item does exist in the document
            if pre_index:
                # if the model do make prediction
                allitem[key]['pre'], allitem[key]['recall'], allitem[key]['f1'], allitem[key]['acc'] = metricCal2(y_pred, y_true, value)
            else:
                # if the model does not make prediction on this item, set performance to 0.0
                allitem[key]['pre'] = 0.0
                allitem[key]['recall'] = 0.0
                allitem[key]['f1'] = 0.0
                allitem[key]['acc'] = 0.0
    return(allitem)

import re

def f1_perf(fold_perf):
    """
    Extract core and overall F1 scores from performance metrics.

    Core items are the most important sections: 1, 1A, 3, 5, 7, 7A, 8, 10, 11.

    Args:
        fold_perf: Pandas Series or dict containing performance metrics
                   with keys like 'Item1_f1', 'Item2_f1', etc.

    Returns:
        Tuple of (mean_core_f1, mean_all_f1):
            - mean_core_f1: Average F1 score for core items
            - mean_all_f1: Average F1 score for all items
    """
    core_f1 = []
    all_f1 = []
    # coreitem = ['Item1', 'Item1A', 'Item3', 'Item5', 'Item7', 'Item7A', 'Item8', 'Item10', 'Item11']
    rep1 = re.compile('^Item1_|Item1A_|Item3_|Item5_|Item7_|Item7A_|Item8_|Item10_|Item11_')
    for key, value in fold_perf.to_dict().items():
        tstr = "_f1"        
        if key[-len(tstr):] == tstr:
            all_f1.append(value)
            if len(rep1.findall(key)) > 0:
                # print(f"{key} = {value}")
                core_f1.append(value)
    
    core_f1 = np.array(core_f1)
    all_f1 = np.array(all_f1)
    mean_core_f1 = core_f1.mean()
    mean_all_f1 = all_f1.mean()
    return mean_core_f1, mean_all_f1

def createFeatures(doc):
    """
    Create hand-crafted features for each line in a document.

    Extracts various features useful for item segmentation:
    - Position features (forward/backward percentile)
    - Item markers (starts with "ITEM")
    - Length features (word count)
    - Signature detection
    - Regex-based item name and number matching
    - Content proximity features (nearby ITEM mentions)

    Args:
        doc: List of text lines (strings)

    Returns:
        Pandas DataFrame with feature columns for each line
    """
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

        # head_pos: 
        headthreshold_ratio = 0.3  # 0 = ignore
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
            

        # unigram and biagram Yen-shou
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
    """
    Check if line starts with "ITEM" (case-insensitive).

    Args:
        text: Text line to check

    Returns:
        1 if line starts with "ITEM", 0 otherwise
    """
    criteria = re.findall(r'^\s*item',text[0:15],re.IGNORECASE)
    if len(criteria) == 1 :
        return(1)
    else:
        return(0)


def signatureCheck(text):
    """
    Check if line contains the word "signature" (case-insensitive).

    Args:
        text: Text line to check

    Returns:
        1 if "signature" found, 0 otherwise
    """
    criteria = re.findall(r'signature',text,re.IGNORECASE)
    if len(criteria) != 0:
        return(1)
    else:
        return(0)

def itemNameCheck(prefix, suffix, text):
    """
    Check if line matches item name patterns using regex.

    Detects items by their textual descriptions (e.g., "ITEM 1. Business").

    Args:
        prefix: Prefix for feature names (e.g., "contain_")
        suffix: Suffix for feature names (e.g., "_name")
        text: Text line to check

    Returns:
        Dictionary mapping feature names to binary values (0 or 1)
        Keys are formatted as "{prefix}{item}{suffix}"
    """
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
    item_name_features = {}  # sentence feature, one instance
    text = text.strip()
    for item, condition in alltasks.items():
        criteria = re.findall(condition, text, re.IGNORECASE)  # get what is matched
        # print(criteria)
        if len(criteria) == 1:  # item only show once. 
            item_name_features[prefix+item+suffix] = 1
        else:        
            item_name_features[prefix+item+suffix] = 0
    return item_name_features

def itemNumberCheck(prefix, suffix, text):
    """
    Check if line contains item numbers using regex.

    Detects items by their numeric identifiers (e.g., "ITEM 1", "ITEM 1A").

    Args:
        prefix: Prefix for feature names (e.g., "contain_")
        suffix: Suffix for feature names (e.g., "_number")
        text: Text line to check

    Returns:
        Dictionary mapping feature names to binary values (0 or 1)
        Keys are formatted as "{prefix}{tag}{suffix}" for tags 1, 1A, 2-6
    """
    tag_list = ["1", "1A", "2", "3", "4", "5", "6"]
    pre150 = text[0:50]  # Check only first 50 characters
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


def expand_pred_to_lines(pred, seqmap, lines):
    """
    Map predicted tags back to the original line sequence and fill gaps.

    This function handles cases where some lines were not processed (marked as 'X')
    by propagating tags from surrounding predicted lines.

    Args:
        pred: List of predicted tags (length = number of processed lines)
        seqmap: List mapping pred indices to original line indices
        lines: Original list of all lines

    Returns:
        List of tags (length = len(lines)) with gaps filled
        - 'X' markers are replaced based on surrounding tag context
        - If preceded by a 'B' tag, propagates appropriate 'I' tag
    """
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
    return pred_ext

