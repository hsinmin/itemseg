# itemseg

![](https://raw.githubusercontent.com/hsinmin/itemseg/main/ITEMSEG%20LOGO1%20SMALL.jpg)

Itemseg is a 10-K item segmentation tool for processing 10-K reports and extracting item-specific text.

Itemseg supports the following input formats (--input_type):
* **raw**: Complete submission text file. See example at [SEC Website](https://www.sec.gov/Archives/edgar/data/789019/000156459020034944/0001564590-20-034944.txt)
* **html**: 10-K report in HTML format. See example at [SEC Website](https://www.sec.gov/Archives/edgar/data/789019/000156459020034944/msft-10k_20200630.htm)
* **native_text**: 10-K report in pure text format. See example at [SEC Website](https://www.sec.gov/Archives/edgar/data/789019/000103221001501099/d10k.txt)
* **cleaned_text**: 10-K report converted to pure text format with tables removed.

The input (`--input`) can be either a local file or a URL pointing to the SEC website.

Itemseg supports the following item segmentation approaches (--method):
* **crf**: Conditional Random Field (default method). Recommended for machines without a GPU.
* **lstm**: Bi-directional Long Short-Term Memory.
* **bert**: BERT4ItemSeg; BERT encoder coupled with Bi-LSTM.
* **chatgpt**: GPT4ItemSeg; Uses OpenAI API and line-id-based prompting.

Both **lstm** and **bert** require a GPU to work at a reasonable speed. You will need to setup the GPU hardware and driver before using these approaches. You can still use itemseg to process 10-K reports without GPUs by selecting the **crf** approach. 

[![PyPI - Version](https://img.shields.io/pypi/v/itemseg.svg)](https://pypi.org/project/itemseg)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/itemseg.svg)](https://pypi.org/project/itemseg)

-----

**Table of Contents**

- [Installation](#installation)
- [Itemseg Example Usage](#itemseg-example-usage)
- [License](#license)
- [Citation](#citation)

## Installation

We recommend installing itemseg in a separate environment created by `virtualenv` to prevent library version conflicts. The instructions below have been tested with Ubuntu 22 LTS and macOS 15.5.

### Setup virtualenv
Install `virtualenv` first if it is not already installed. 

For Ubuntu 22 LTS:
```console
sudo apt install python3-venv
```

For macOS:
```console
pip3 install virtualenv
```

The next step is to setup the virtualenv.

Ubuntu 22 LTS and macOS:
```console
python3 -m venv env_itemseg
```

Activate the virtual environment:
```console
source env_itemseg/bin/activate
```

Now we can install itemseg
```console
pip3 install itemseg
```

### Download resource files
You will need to download resource files first before start using the tool.
```console
python3 -m itemseg --get_resource
```

### Download NLTK data
```console
python3 -m nltk.downloader punkt punkt_tab
```

## Itemseg Example Usage

### Segment items in a 10-K file
Using Apple 10-K (2023) as an example (adjust --user_agent according to your affiliation):
```console
python3 -m itemseg --input_type raw --input https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/0000320193-23-000106.txt --user_agent "Some University johndow@someuniversity.edu"
```

See the results in `./segout01/`.

The `*.csv` file contains line-by-line predictions for items in Begin-Inside-Outside (BIO) style tags. Other files contain item-specific text.

### About 10-K reports
A 10-K report is an annual report filed by publicly traded companies with the U.S. Securities and Exchange Commission (SEC). It provides a comprehensive overview of the company's financial performance and is more detailed than an annual report. Key items of a 10-K report include:

* **Item 1 (Business)**: Describes the company's main operations, products, and services.
* **Item 1A (Risk Factors)**: Outlines risks that could affect the company's business, financial condition, or operating results.
* **Item 3 (Legal Proceedings)**
* **Item 7 (Management’s Discussion and Analysis of Financial Condition and Results of Operations; MD&A)**: Offers management's perspective on the financial results, including discussion of liquidity, capital resources, and results of operations.

You can search and read 10-K reports through the [EDGAR web interface](https://www.sec.gov/edgar/search-and-access).  For **raw** input type, Itemseg takes the URL of the `Complete submission text file`, converts the HTML to formatted text, removing tables with numerical content, and segments the text file by items.

As an example, the Amazon 10-K report page for [fiscal year 2022](https://www.sec.gov/Archives/edgar/data/1018724/000101872423000004/0001018724-23-000004-index.htm) shows the link to the HTML 10-K report and a `Complete submission text file` [0001018724-23-000004.txt](https://www.sec.gov/Archives/edgar/data/1018724/000101872423000004/0001018724-23-000004.txt). Pass this link to the itemseg module, and it will retrieve the file and segment items for you. Remember to adjust --user_agent according to your affiliation. 

```console
python3 -m itemseg --input_type raw --input https://www.sec.gov/Archives/edgar/data/1018724/000101872423000004/0001018724-23-000004.txt --user_agent "Some University johedoe@someuniv.edu"

```

The default setting outputs line-by-line tags (BIO style) in a CSV file, together with Item 1, Item 1A, Item 3, and Item 7 in separate files (`--outfn_type "csv,item1,item1a,item3,item7"`). You can change the output file type combination with `--outfn_type`. For example, if you only want to output Item 1A and Item 7, set `--outfn_type "item1a,item7"`.

If you are trying to process large amounts of 10-K files, a good starting point is the [master index](https://www.sec.gov/Archives/edgar/full-index/), which lists all available files and provides a convenient way to construct a comprehensive list of target files.

<!-- The module also comes with a script that allows you to run the module via the `itemseg` command. The default location (for Ubuntu) is at `~/.local/bin`. Add this location to your PATH to enable the `itemseg` command. -->

## License

`itemseg` is distributed under the terms of the [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/) license.

## Citation

Please cite our work if you use **itemseg** in your research. 

Lu, H.-M., Chien, Y.-T., Yen, H.-H., and Chen, Y.-H. (2025). Utilizing Pre-trained and Large Language Models for 10-K Items Segmentation. arXiv preprint arxiv:2502.08875 (https://arxiv.org/abs/2502.08875).

We extend our special thanks to Chia-Tai Li and I-Chen Tsai for their valuable support in managing the dataset, as well as merging and refactoring the project.