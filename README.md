# itemseg

![](https://raw.githubusercontent.com/hsinmin/itemseg/main/ITEMSEG%20LOGO1%20SMALL.jpg)


10-K Item Segmentation with Line-based Attention (ISLA) is a tool to process
EDGAR 10-K reports and extract item-specific text. 


[![PyPI - Version](https://img.shields.io/pypi/v/itemseg.svg)](https://pypi.org/project/itemseg)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/itemseg.svg)](https://pypi.org/project/itemseg)

-----

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Installation

```console
pip3 install itemseg
```

### Download resource file
```console
python3 -m itemseg --get_resource
```

### Download nltk data

Launch python3 console
```console
>>> import nltk
>>> nltk.download('punkt')
```

### Segment items in a 10-K file
Using Apple 10-K (2023) as an example:
```console
python3 -m itemseg --input https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/0000320193-23-000106.txt
```

See the results in ./segout01/

The *.csv file contain line-by-line prediction for items in a Begin-Inside-Outside (BIO) style tags. Other files contain item-sepcific text. Change output file types via `--outfn_type`.


### About 10-K files. 
A 10-K report is an annual report filed by publicly traded companies with the U.S. Securities and Exchange Commission (SEC). It provides a comprehensive overview of the company's financial performance and is more detailed than an annual report. Key items of a 10-K report include:

* Item 1 (Business): Describes the company's main operations, products, and services.
* Item 1A (Risk Factors): Outlines risks that could affect the company's business, financial condition, or operating results. 
* Item 3 (Legal Proceedings)
* Item 7 (Management’s Discussion and Analysis of Financial Condition and Results of Operations; MD&A): Offers management's perspective on the financial results, including discussion of liquidity, capital resources, and results of operations.

You can search and read 10-K reports through the [EDGAR web interface](https://www.sec.gov/edgar/search-and-access). The itemseg module takes the URL of the `Complete submission text file`, convert the HTML to formated txt file, and segment the txt file by items. 

As an example, the AMAZON 10-K report page for [fiscal year 2022](https://www.sec.gov/Archives/edgar/data/1018724/000101872423000004/0001018724-23-000004-index.htm) shows the link to the HTML 10-K report and a `Complete submission text file` [0001018724-23-000004.txt](https://www.sec.gov/Archives/edgar/data/1018724/000101872423000004/0001018724-23-000004.txt). Pass this link (https://www.sec.gov/Archives/edgar/data/1018724/000101872423000004/0001018724-23-000004.txt) to the itemseg module, and it will retrive the file and segment items for you. 

```console
python3 -m itemseg --input https://www.sec.gov/Archives/edgar/data/1018724/000101872423000004/0001018724-23-000004.txt
```

The default setting is to output line-by-line tag (BIO style) in a csv file, together with Item 1, Item 1A, Item 3, and Item 7 in separate files (--outfn_type "csv,item1,item1a,item3,item7"). You can change output file type combination with --outfn_type. For example, if you only want to output Item 1A and Item 7, then set --outfn_type "item1a,item7". 

If you are trying to process large amounts of 10-K files, a good starting point is the master index (https://www.sec.gov/Archives/edgar/full-index/), which lists all available files and provides a convenient venue to construct a comprehensive list of target files.

The module also comes with a script file that allow you to run the module via `itemseg` command. The default location (for Ubuntu) is at ~/.local/bin. Add this location to your path to enable `itemseg` command. 


## License

`itemseg` is distributed under the terms of the [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/) license.
