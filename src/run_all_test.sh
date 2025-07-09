#!/bin/bash

echo "This is a test script for itemseg."
echo "It will execute the following routine:"
echo "* Delete existing virtualenv"
echo "* Create a new virtualenv"
echo "* Activate the virtualenv"
echo "* Run unittest script ['raw', 'html', 'native_text', 'cleaned_text'] X 
                      ['crf', 'lstm', 'bert']"
echo "* Run chatgpt api test"
echo "* Exit the virtualenv"
echo "    ============"



# virtualenv name
VENAME=utvenvitemseg

if [ -d "$VENAME" ]; then
    echo $VENAME exists. Press Enter to delete the directory or Ctrl-c to abort.
    read tmp
    rm -rf $VENAME
fi


echo "Create virtualenv $VENAME"
python3 -m venv $VENAME
echo "Activate the virtualenv"
source $VENAME/bin/activate

echo "Install required packages"
pip3 install  'requests <= 2.32.4'
pip3 install  'pandas <= 2.3.0'
pip3 install  'inscriptis <= 2.6.0' 'python-crfsuite >= 0.9.11'
pip3 install  'nltk <= 3.9.1' 'gensim <= 4.3.3' 'torch <= 2.7.1' 'scikit-learn <= 1.7.0'
pip3 install  'scipy <= 1.13.1' 'numpy <= 1.26.4' 'urllib3 <= 2.5.0' 'openai <= 1.91.0'
pip3 install  'sentence_transformers <= 4.1.0'
# check gensim numpy dependency (2.0.2 not good?)

echo "=== Install depednency completed"
# python3 -m itemseg --help

echo "=== Run get_resource to download the required resources"
python3 -m itemseg --get_resource
if [ $? -ne 0 ]; then
    echo "Error: Get resource failed. Is the host reachable?"
    exit 1
fi

echo "=== Get nltk resources"
python3 -m nltk.downloader punkt punkt_tab
if [ $? -ne 0 ]; then
    echo "Error: Get nltk resources failed."
    exit 1
fi

echo "=== Now its time to run the test scripts ==="

setname=raw_crf_online
echo "=== Run test script for $setname"
rm -rf segout01
python3 -m itemseg --input_type raw \
       --input https://www.sec.gov/Archives/edgar/data/354950/000035495020000015/0000354950-20-000015.txt \
       --method crf \
       --user_agent "Some University johndoe@someuniv.edu" \
       --outfn_prefix homedepot_raw.txt 
diff segout01 ../testdata/out_homedepot_raw_crf
if [ $? -ne 0 ]; then
    echo "$setname Error: raw output different from prestored result."     
    exit 1
else
    echo "$setname OK: output matches prestored result."
fi




setname=raw_crf
echo "=== Run test script for $setname"
rm -rf segout01
python3 -m itemseg --input_type raw --input ../testdata/homedepot_raw.txt --method crf
diff segout01 ../testdata/out_homedepot_raw_crf
if [ $? -ne 0 ]; then
    echo "$setname Error: raw output different from prestored result."     
    exit 1
else
    echo "$setname OK: output matches prestored result."
fi

setname=raw_lstm
echo "=== Run test script for $setname"
rm -rf segout01
python3 -m itemseg --input_type raw --input ../testdata/homedepot_raw.txt --method lstm
diff segout01 ../testdata/out_homedepot_raw_lstm
if [ $? -ne 0 ]; then
    echo "$setname Error: raw output different from prestored result."     
    exit 1
else
    echo "$setname OK: output matches prestored result."
fi

setname=raw_bert
echo "=== Run test script for $setname"
rm -rf segout01
python3 -m itemseg --input_type raw --input ../testdata/homedepot_raw.txt --method bert
diff segout01 ../testdata/out_homedepot_raw_bert
if [ $? -ne 0 ]; then
    echo "$setname Error: raw output different from prestored result."     
    exit 1
else
    echo "$setname OK: output matches prestored result."
fi

# ==== html tests ====
setname=html_crf
echo "=== Run test script for $setname"
rm -rf segout01
python3 -m itemseg --input_type html --input ../testdata/homedepot_html.htm --method crf
diff segout01 ../testdata/out_homedepot_htm_crf
if [ $? -ne 0 ]; then
    echo "$setname Error: html output different from prestored result."     
    exit 1
else
    echo "$setname OK: output matches prestored result."
fi 

setname=html_lstm
echo "=== Run test script for $setname"
rm -rf segout01
python3 -m itemseg --input_type html --input ../testdata/homedepot_html.htm --method lstm
diff segout01 ../testdata/out_homedepot_htm_lstm
if [ $? -ne 0 ]; then
    echo "$setname Error: html output different from prestored result."     
    exit 1
else
    echo "$setname OK: output matches prestored result."
fi

setname=html_bert
echo "=== Run test script for $setname"
rm -rf segout01 
python3 -m itemseg --input_type html --input ../testdata/homedepot_html.htm --method bert
diff segout01 ../testdata/out_homedepot_htm_bert
if [ $? -ne 0 ]; then
    echo "$setname Error: html output different from prestored result."     
    exit 1     
else
    echo "$setname OK: output matches prestored result."
fi

# ==== native_text tests ====
setname=native_text_crf
echo "=== Run test script for $setname"
rm -rf segout01
python3 -m itemseg --input_type native_text --input ../testdata/homedepot_ntext.txt --method crf
diff segout01 ../testdata/out_homedepot_ntext_crf
if [ $? -ne 0 ]; then
    echo "$setname Error: native_text output different from prestored result."     
    exit 1
else        
    echo "$setname OK: output matches prestored result."
fi 

setname=native_text_lstm
echo "=== Run test script for $setname"
rm -rf segout01
python3 -m itemseg --input_type native_text --input ../testdata/homedepot_ntext.txt --method lstm
diff segout01 ../testdata/out_homedepot_ntext_lstm
if [ $? -ne 0 ]; then
    echo "$setname Error: native_text output different from prestored result."     
    exit 1
else    
    echo "$setname OK: output matches prestored result."
fi  

setname=native_text_bert
echo "=== Run test script for $setname"
rm -rf segout01     
python3 -m itemseg --input_type native_text --input ../testdata/homedepot_ntext.txt --method bert
diff segout01 ../testdata/out_homedepot_ntext_bert
if [ $? -ne 0 ]; then
    echo "$setname Error: native_text output different from prestored result."     
    exit 1
else
    echo "$setname OK: output matches prestored result."
fi  