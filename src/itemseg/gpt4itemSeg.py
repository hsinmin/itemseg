import nltk
from openai import OpenAI
import pickle
import os

# Few shots (five shots)
instruction = """I am an excellent financial professional. \
The task is to identify the starting lines of items \
in 10-K report. 

A 10-K report may contain the following items:
Item 1. Business
Item 1A. Risk Factors
Item 1B. Unresolved Staff Comments
Item 2. Properties
Item 3. Legal Proceedings
Item 4. Mine Safety Disclosures
Item 5. Market for Registrant’s Common Equity, Related \
Stockholder Matters and Issuer Purchases of Equity Securities
Item 6. Selected Financial Data
Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations
Item 7A. Quantitative and Qualitative Disclosures About Market Risk
Item 8. Financial Statements and Supplementary Data
Item 9. Changes in and Disagreements With Accountants on Accounting and Financial Disclosure
Item 9A. Controls and Procedures
Item 9B. Other Information
Item 9C. Disclosure Regarding Foreign Jurisdictions that Prevent Inspections
Item 10. Directors, Executive Officers and Corporate Governance
Item 11. Executive Compensation
Item 12. Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters
Item 13. Certain Relationships and Related Transactions, and Director Independence
Item 14. Principal Accountant Fees and Services
Item 15. Exhibit and Financial Statement Schedules

Each item may start with a title, followed by the content. \
Each line contains a line ID, followed by its content. \
Extract the line ID of Item 1, Item 1A, Item 2, Item 3, \
Item 4, Item 5, Item 6, Item 7, Item 7A, Item 8, Item 9, \
Item 9A, Item 10, Item 11, Item 12, Item 13, Item 14, Item 15. \
If the item is not available, print NA. \
The beginning of a report may contain a table of contents \
that also lists the item heading but is irrelevant.
Below are some examples. 

Example 1:
=====
1	10-K 1 knowles20181231-10xk.htm 10-K
2	UNITED STATES
4	Washington, D.C. 20549
6	FORM 10-K
7	
8	(Mark One)
9	
10	For the fiscal year ended December 31, 2018 .
11	
12	OF 1934
13	For the transition period from to
14	
15	Commission File Number: 001-36102
16	
17	Knowles Corporation
18	(Exact name of registrant as specified in its charter)
19	
20	(State or other jurisdiction of incorporation or organization)    (I.R.S. Employer Identification No.)
21	
22	1151 Maplewood Drive
23	
24	(630) 250-5100
25	(Registrant's telephone number, including area code)
26	
27	Securities registered pursuant to Section 12(b) of the Act:
29	Securities registered pursuant to Section 12(g) of the Act: None
30	
31	Indicate by check mark if the registrant is a well-known seasoned issuer, as defined in Rule 405 of the Securities Act.
32	Yes   No o
33	Indicate by check mark if the registrant is not required to file reports pursuant to Section 13 or 15(d) of the Act. Yes o No
34	
35	Indicate by check mark whether the registrant: (1) has filed all reports required to be filed by Section 13 or 15(d) 
36	Yes   No o
37	
38	Indicate by check mark whether the registrant has submitted electronically every Interactive Data File required to be 
39	Yes   No o
40	
41	Indicate by check mark if disclosure of delinquent filers pursuant to Item 405 of Regulation S-K (SS229.405) is not 
43	Indicate by check mark whether the registrant is a large accelerated filer, an accelerated filer, a non-accelerated filer, 
45	Non-accelerated filer o      (Do not check if a smaller reporting company)    Smaller reporting company o
46	Emerging growth company o
47	
48	If an emerging growth company, indicate by check mark if the registrant has elected not to use the extended transition period
50	Indicate by check mark whether the registrant is a shell company (as defined in Rule 12b-2 of the Act). Yes o No
51	
52	The aggregate market value of the voting and non-voting common stock held by non-affiliates of the registrant as of 
54	Certain information contained in the registrant's Proxy Statement for its 2019 Annual Meeting of Stockholders is incorporated 
55	
56	
57	
58	Page
59	
60	Item 1.       Business                                                                                                        3
61	Item 1A.      Risk Factors                                                                                                    7
62	Item 1B.      Unresolved Staff Comments                                                                                       16
63	Item 2.       Properties                                                                                                      17
64	Item 3.       Legal Proceedings                                                                                               17
65	Item 4.       Mine Safety Disclosures                                                                                         17
67	Item 5.       Market for Registrant's Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities    20
68	Item 6.       Selected Financial Data                                                                                         22
69	Item 7.       Management's Discussion and Analysis of Financial Condition and Results of Operations                           24
70	Item 7A.      Quantitative and Qualitative Disclosures About Market Risk                                                      46
71	Item 8.       Financial Statements and Supplementary Data                                                                     47
72	Item 9.       Changes in and Disagreements with Accountants on Accounting and Financial Disclosure                            93
73	Item 9A.      Controls and Procedures                                                                                         93
74	Item 9B.      Other Information                                                                                               94
76	Item 10.      Directors, Executive Officers and Corporate Governance                                                          94
77	Item 11.      Executive Compensation                                                                                          95
78	Item 12.      Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters                  95
79	Item 13.      Certain Relationships and Related Transactions, and Director Independence                                       96
80	Item 14.      Principal Accountant Fees and Services                                                                          96
82	Item 15.      Exhibits and Financial Statement Schedules                                                                      96
83	Item 16.      Form 10-K Summary                                                                                               100
84	
85	
86	
87	PART I
88	ITEM 1. BUSINESS
89	
90	Unless the context otherwise requires, references in this Annual Report on Form 10-K to "Knowles," the "Company," "we," "our," 
91	
92	Our Company
93	
94	We are a market leader and global provider of advanced micro-acoustic, audio processing, and precision device solutions, 
95	
96	Our Strategy
=====

Output:
Item 1,87
Item 1A,NA
Item 2,NA
Item 3,NA 
Item 4,NA
Item 5,NA
Item 6,NA
Item 7,NA
Item 7A,NA
Item 8,NA
Item 9,NA
Item 9A,NA
Item 10,NA
Item 11,NA
Item 12,NA
Item 13,NA
Item 14,NA
Item 15,NA



Example 2:
=====
987	Consolidated Balance Sheet Data:
988	As of January 31,
989	
990	The Company adopted Accounting Standards Update ("ASU") 2016-02, Leases (Topic 842) on February 1, 2019 and ASU 2014-09, Revenue from Contracts with Customers (Topic 606) on 
991	
992	Ooma | FY2021 Form 10-K | 42
993	ITEM 7. Management's Discussion and Analysis of Financial Condition and Results of Operations
994	
995	The following discussion should be read in conjunction with our consolidated financial statements and the related notes to those statements included elsewhere in this Form 10-K. 
996	
997	This section of this Form 10-K generally discusses fiscal 2021 and 2020 items and year-to-year comparisons between fiscal 2021 and 2020. 
998	
999	Executive Overview
1000	
1001	Ooma creates powerful connected experiences for businesses and consumers. Our smart SaaS and UCaaS platforms serve as a communications hub, which offers cloud-based communications solutions, 
1002	
1003	We generate subscription and services revenue by selling subscriptions and other services for our communications services, 
1004	
1005	We refer to Ooma Office and Ooma Enterprise collectively as Ooma Business. Ooma Residential includes Ooma Telo basic and premier services as well as our smart security solutions. 
1006	
1007	Fiscal 2021 Financial Performance
=====

Output:
Item 1,NA
Item 1A,NA
Item 2,NA
Item 3,NA 
Item 4,NA
Item 5,NA
Item 6,NA
Item 7,993
Item 7A,NA
Item 8,NA
Item 9,NA
Item 9A,NA
Item 10,NA
Item 11,NA
Item 12,NA
Item 13,NA
Item 14,NA
Item 15,NA


Example 3:
=====
121	As of December 26, 2020, we employed approximately 241 individuals on a full-time equivalent basis compared to approximately 
122	Government Regulations
123	
124	Benefit Plans
125	ENGlobal sponsors a 401(k) retirement plan for its employees. The Company, at the direction of the Board of Directors, 
126	
127	ITEM 1A. RISK FACTORS
128	Set forth below and elsewhere in this Report and in other documents that we file with the SEC are risks and uncertainties 
129	
130	Economic downturns and the volatility and level of oil and natural gas prices could have a negative impact on our businesses. 
131	Prices and expectations about future prices of oil and natural gas;
132	Domestic and foreign supply of and demand for oil and natural gas;
133	The cost of exploring for, developing, producing and delivering oil and natural gas;
134	Weather conditions, such as hurricanes, which may affect our clients' ability to produce oil and natural gas;
135	Available pipeline, storage and other transportation capacity;
136	Federal, state and local regulation of oilfield activities;
=====

Output:
Item 1,NA
Item 1A,127
Item 2,NA
Item 3,NA 
Item 4,NA
Item 5,NA
Item 6,NA
Item 7,NA
Item 7A,NA
Item 8,NA
Item 9,NA
Item 9A,NA
Item 10,NA
Item 11,NA
Item 12,NA
Item 13,NA
Item 14,NA
Item 15,NA


Example 4:
=====
1	10-K 1 knowles20181231-10xk.htm 10-K
2	UNITED STATES
3	
4	Washington, D.C. 20549
5	
6	FORM 10-K
7	
8	(Mark One)
9	
10	For the fiscal year ended December 31, 2018 .
11	
12	OF 1934
13	For the transition period from to
14	
15	Commission File Number: 001-36102
16	
17	Knowles Corporation
18	(Exact name of registrant as specified in its charter)
19	
20	(State or other jurisdiction of incorporation or organization)    (I.R.S. Employer Identification No.)
21	
22	1151 Maplewood Drive
23	
24	(630) 250-5100
25	(Registrant's telephone number, including area code)
26	
27	Securities registered pursuant to Section 12(b) of the Act:
28	
29	Securities registered pursuant to Section 12(g) of the Act: None
30	
31	Indicate by check mark if the registrant is a well-known seasoned issuer, as defined in Rule 405 of the Securities Act.
32	Yes   No o
33	Indicate by check mark if the registrant is not required to file reports pursuant to Section 13 or 15(d) of the Act. Yes o No
34	
35	Indicate by check mark whether the registrant: (1) has filed all reports required to be filed by 
36	Yes   No o
37	
38	Indicate by check mark whether the registrant has submitted electronically every Interactive Data File required to be submitted and posted pursuant to Rule 405 of Regulation S-T (SS232.405 of this chapter) during the preceding 12 months (or for such shorter period that the registrant was required to submit such files).
39	Yes   No o
40	
41	Indicate by check mark if disclosure of delinquent filers pursuant to Item 405 of Regulation S-K (SS229.405) 
43	Indicate by check mark whether the registrant is a large accelerated filer, an accelerated filer, a non-accelerated filer
44	
53	
54	Certain information contained in the registrant's Proxy Statement for its 2019 Annual Meeting of 
57	
58	Page
59	
60	Item 1.       Business                                                                   3
61	Item 1A.      Risk Factors                                      7
62	Item 1B.      Unresolved Staff Comments                                              16
63	Item 2.       Properties                                                              17
64	Item 3.       Legal Proceedings                                                                   17
65	Item 4.       Mine Safety Disclosures                                                    17
66	
67	Item 5.       Market for Registrant's Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities    20
68	Item 6.       Selected Financial Data                                       22
69	Item 7.       Management's Discussion and Analysis of Financial Condition and Results of Operations                 24
70	Item 7A.      Quantitative and Qualitative Disclosures About Market Risk               46
71	Item 8.       Financial Statements and Supplementary Data                   47
72	Item 9.       Changes in and Disagreements with Accountants on Accounting and Financial Disclosure         93
73	Item 9A.      Controls and Procedures                                                      93
74	Item 9B.      Other Information                                                   94
75	
76	Item 10.      Directors, Executive Officers and Corporate Governance                     94
77	Item 11.      Executive Compensation                                               95
78	Item 12.      Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters        95
79	Item 13.      Certain Relationships and Related Transactions, and Director Independence         96
80	Item 14.      Principal Accountant Fees and Services                               96
81	
82	Item 15.      Exhibits and Financial Statement Schedules                                        96
83	Item 16.      Form 10-K Summary                                                                100
84	
85	
86	
87	PART I
88	ITEM 1. BUSINESS
89	
90	Unless the context otherwise requires, references in this Annual Report on Form 10-K to "Knowles," the "Company," "we," "our," or "us" refer to Knowles 
91	
92	Our Company
93	
94	We are a market leader and global provider of advanced micro-acoustic, audio processing, 
95	
96	Our Strategy
97	
98	Our primary focus has been to position the Company to benefit from the positive 
99	
100	We have been focused on strategically positioning the business to support these 
101	
102	In our Precision Devices segment, we continue to drive higher sales growth and 
103	
104	Our Business Segments
105	
198	
199	Employees
200	
201	We currently employ approximately 8,000 persons across our facilities in 11 countries. Approximately 78% of our 
202	
203	Other Information
204	
205	Knowles was incorporated in Delaware on June 12, 2013. The address of our principal executive offices is 1151 Maplewood Drive, 
206	
207	ITEM 1A. RISK FACTORS
208	
209	Cautionary Statement Concerning Forward-Looking Statements
210	
211	This Annual Report on Form 10-K contains certain statements regarding business strategies, market potential, future financial performance, future action, results, 
212	
213	In particular, information included under the sections entitled "Business," "Risk Factors," 
214	
215	Readers are cautioned that the matters discussed in these forward-looking statements are subject to risks, uncertainties, assumptions, and other factors that are difficult to predict and which could 
216	
217	You should consider each of the following factors as well as the other information in this Annual Report on Form 10-K, 
221	Risks Related To Our Business
222	
223	We depend on the mobile handset market for a significant portion of our revenues, and any downturn or slower than expected growth in this market could significantly reduce our revenues and adversely impact our operating results.
224	
237	
238	Certain of our businesses rely on highly specialized suppliers or foundries for critical materials, components, 
239	
240	Global markets for our products are highly competitive and subject to rapid technological change. If we are unable to develop new products and compete effectively in these markets, our financial condition and operating results could be materially adversely affected.
241	
242	We compete in highly competitive, technology-based, industries that are highly dynamic as new technologies 
243	
244	We operate in the highly competitive mobile handset industry, which requires us to invest significant capital in developing, qualifying, and ramping production of new products without any assurance of product sales which could negatively impact our operating results and profits.
245	
246	A significant portion of our consolidated revenues are derived from acoustic components and audio solutions, 
247	
248	In addition, the time required and costs incurred by us to ramp-up production for new products can be significant. 
249	
250	
251	
252	Our foreign operations, supply chain, and footprint optimization strategies are each subject to various risks that could materially adversely impact our results of operations and financial position.
253	
254	Many of our manufacturing operations, research and development operations, vendors, and suppliers are located outside the United States and if we are unable to successfully manage the risks associated with our global operations, our results of operations and financial position could be negatively impacted. These risks include:
262	Given that many of our manufacturing operations are located outside the United States, a border tax, if enacted, could have a material adverse effect on our operating results.
366	Each of our certificate of incorporation, our by-laws, and Delaware law, as currently in effect, contain provisions 
367	
368	o    the inability of our stockholders to call a special meeting or act by written consent;
369	o    rules regarding how stockholders may present proposals or nominate directors for election at stockholder meetings;
370	o    the right of our Board of Directors to issue preferred stock without stockholder approval;
371	o    the classification of our Board of Directors and a provision that stockholders may only remove directors for cause, in each case until our 2021 annual meeting of stockholders;
372	o    the ability of our directors, without a stockholder vote, to fill vacancies on our Board of Directors (including those resulting from an enlargement of the Board of Directors); and
373	o    the requirement that stockholders holding at least 80% of our voting stock are required to amend certain provisions in our certificate of incorporation and our by-laws.
374	
375	In addition, current Delaware law includes provisions which limit the ability of persons that, without prior board approval, acquire 
376	
377	In light of present circumstances, we believe these provisions taken as a whole protect our stockholders from coercive 
378	
379	ITEM 1B. UNRESOLVED STAFF COMMENTS
380	
381	None.
382	
383	
384	
385	ITEM 2. PROPERTIES
386	
387	Our corporate headquarters is located in Itasca, Illinois. We maintain technical customer support offices and operating 
388	
389	The number, type, location, and size of the properties used by our continuing operations as of December 31, 2018 are shown in the following chart:
390	
391	Total
392	Number and nature of facilities:
393	
394	Other Facilities (principally sales, research and development, and headquarters)      13
395	
396	Square footage (in 000s):
397	
398	Locations:
399	
400	(1) Expiration dates on leased facilities range from 1 to 9 years.
401	
402	We believe that our owned and leased facilities are well-maintained and suitable for our operations.
403	
404	ITEM 3. LEGAL PROCEEDINGS
405	
406	From time to time, we are involved in various legal proceedings and claims arising in the ordinary course of our business, including those related to intellectual property, which may be owned by us or others. 
407	
408	ITEM 4. MINE SAFETY DISCLOSURES
409	
410	Not applicable.
411	
412	
413	
414	The following sets forth information regarding our executive officers, as of February 19, 2019.
415	
416	Air A. Bastarrica, Jr.    39     Vice President, Controller
417	
421	
422	PART II
423	
424	ITEM 5. MARKET FOR REGISTRANT'S COMMON EQUITY, RELATED STOCKHOLDER MATTERS, AND ISSUER PURCHASES OF EQUITY SECURITIES
425	
426	Market Information
427	
428	Our common stock is listed on the New York Stock Exchange ("NYSE") under the ticker symbol "KN".
429	
430	Dividends
431	
432	Since our common stock began trading on the NYSE, we have not paid cash dividends and we do not anticipate paying a 
433	
434	Holders
435	
436	The number of holders of record of our common stock as of February 14, 2019 was approximately 982 .
437	
438	Recent Sales of Unregistered Securities
439	
440	None.
441	
442	Issuer Purchases of Equity Securities
443	
444	None.
445	
446	
447	
448	Performance Graph
449	This performance graph does not constitute soliciting material, is not deemed filed with the SEC, 
450	
451	Data Source: NYSE
452	*Total return assumes reinvestment of dividends.
453	
454	
455	
456	ITEM 6. SELECTED FINANCIAL DATA
457	
458	The following table presents selected financial data on a continuing operations basis as derived 
459	
460	The selected financial data includes costs of Knowles' businesses, which include the allocation of certain corporate expenses 
461	
462	
463	Statement of Earnings Data (1)
464	
465	As of December 31,
466	Balance Sheet Data
467	
468	
469	Other Data (1)
470	
471	(1)    On July 7, 2016, the Company completed the sale of its Speaker and Receiver Product Line. 
472	
473	(2)    On July 1, 2015, the Company completed its acquisition of all of the outstanding shares of common stock of Audience. The Consolidated Statements of Earnings and Consolidated Balance Sheets include the results of operations, net assets acquired, and depreciation and amortization expense related to Audience since the date of acquisition.
474	
483	(6)    Also includes current portion of long-term debt and capital lease obligations.
484	
485	(7)    We use the term "EBIT" throughout this Annual Report on Form 10-K, defined as net earnings plus (i) interest expense
486	
487	
488	
489	ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS
490	
491	The discussion and analysis presented below refer to and should be read in conjunction with our 
492	
493	Management's discussion and analysis, which we refer to as "MD&A," of our results of operations, financial condition, 
494	
495	As discussed in Note 2. Disposed and Discontinued Operations to our audited Consolidated Financial Statements under Item 8
496	
497	Our Business
498	
499	We are a market leader and global provider of advanced micro-acoustic, audio processing, and precision device solutions
500	
501	Our Business Segments
=====

Output:
Item 1,88
Item 1A,207
Item 2,385
Item 3,404 
Item 4,408
Item 5,424
Item 6,456
Item 7,489
Item 7A,NA
Item 8,NA
Item 9,NA
Item 9A,NA
Item 10,NA
Item 11,NA
Item 12,NA
Item 13,NA
Item 14,NA
Item 15,NA


Example 5:
=====
1787	PricewaterhouseCoopers LLP, the Company's independent registered public accounting firm, has audited the effectiveness of 
1788	
1789	(c) Changes in Internal Control Over Financial Reporting
1790	
1791	There has been no change in our internal control over financial reporting that occurred during the fourth quarter of 2018 
1792	
1793	(d) Inherent Limitations on Effectiveness of Controls
1794	
1795	Our management, including the CEO and CFO, do not expect that our disclosure controls or our internal control over 
1796	
1797	ITEM 9B. OTHER INFORMATION
1798	
1799	Not applicable.
1800	
1801	PART III
1802	
1803	ITEM 10. DIRECTORS, EXECUTIVE OFFICERS, AND CORPORATE GOVERNANCE
1804	
1805	The information with respect to the directors and the board committees of the Company and other corporate governance 
1806	
1807	The information with respect to the executive officers of the Company required to be included pursuant to this Item 10 is 
1808	
1809	The information with respect to Section 16(a) reporting compliance required to be included in this Item 10
=====

Output:
Item 1,NA
Item 1A,NA
Item 2,NA
Item 3,NA 
Item 4,NA
Item 5,NA
Item 6,NA
Item 7,NA
Item 7A,NA
Item 8,NA
Item 9,NA
Item 9A,NA
Item 10,1803
Item 11,NA
Item 12,NA
Item 13,NA
Item 14,NA
Item 15,NA


Below is a 10-K report.
List the result in a table format. \
The first column is the item ID. The second column is the Line ID. \
Use comma (",") to separate the two columns. Include no additional white space. 

=====
"""

# for the estimation of token length
avg_tok_per_word = 1.25
prompt_max_len = 120000

text_final_pre = instruction
trun_len = 30

def truncate_line(intext, ntok=100):
    # intext = "This document covers the company's compliance with accounting standards, including fair value measurements, derivative instruments, income taxes, and legal contingencies. It also includes notes that provide additional context and detail on the financial statements, methodologies used in financial measurements, and significant accounting policies."
    intexttok = intext.split(" ")
    outtext = " ".join(intexttok[0:ntok])
    return outtext

# truncate_line("this is  a test sentence.", ntok=100)


def preprocess_doc(args, lines):
    this_trun_len = trun_len
    nline = len(lines)

    # 這裡為 doc 添加行號，並且檢查 prompt token 數量是否超過限制，最多跑五次
    for _ in range(5):
        text_final = text_final_pre

        for i in range(nline):
            aline = lines[i]
            if len(aline) == 0:
                continue

            text_final += f'{i} {truncate_line(aline, ntok=this_trun_len)}\n'
        
        if args.verbose > 0:
            print(f'prompt Character length = {len(text_final)}')
        words = nltk.word_tokenize(text_final)
        est_token = len(words) * avg_tok_per_word
        if args.verbose > 0:
            print(f"prompt nltk token count and est. token len = {len(words)} / {est_token}")

        # 假設預估的 tokens 數量超過最大限制，則縮減 trun_len 的長度再跑一次
        if est_token > prompt_max_len:
            this_trun_len -= 5
            if args.verbose > 0:
                print(f"    prompt too long, reduce max line len to {this_trun_len}")
            if this_trun_len < 3:
                if args.verbose > 0:
                    print(f" trun_len < 3 ; ({this_trun_len}); stop")
        else:
            break
    
    return text_final

def openai(text_final, apikey):
    msg = [{"role": "user", "content": f"{text_final}\n =====\nOutput:"}]

    print('--Contact openai api...')
    client = OpenAI(api_key=apikey)
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=msg,
    )

    return response

def map_lines_to_tags(response, lines):
    items = response.choices[0].message.content.split('\n')
    itemsegid = []
    for a_item in items:
        tmp1 = a_item.split(',')
        if len(tmp1) < 2:
            continue

        key = tmp1[0].strip()
        key = key.upper()
        key = key.replace('.', '')
        key = key.replace('ITEM', '')
        key = key.strip()

        try:
            value = int(tmp1[1].strip())
            if value >= len(lines): # unrealistic case, just skip
                continue
        except:
            continue

        itemsegid.append([key, value])
    
    if len(itemsegid) == 0:
        # No Item Found!!! Set all lines to 'O'
        predlabel = ['O'] * len(lines)
    else:
        itemsegid2 = sorted(itemsegid, key=lambda x: x[1])

        predlabel = ['O'] * len(lines)
        lastline = len(lines)
        for a_item in itemsegid2[::-1]:
            predlabel[a_item[1]] = 'B' + a_item[0]
            
            for tmpid in range(a_item[1]+1, lastline):
                predlabel[tmpid] = 'I' + a_item[0]
            lastline = a_item[1]

    return predlabel