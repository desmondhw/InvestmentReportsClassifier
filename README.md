<img width="632" alt="Screenshot 2019-08-23 at 17 38 06" src="https://user-images.githubusercontent.com/34176396/63583060-d17ad880-c5cc-11e9-9cfe-787106cd5ff0.png">

# Classifying Investment Research Reports by Sectors and Market Sentiment using Text Analytics

Investment analysts at large buy-side financial institutions receive about 2,000 research reports daily from research institutions and other banks. Manually sorting and analyzing thousands of reports daily is time consuming, if not impossible.

We attempted to develop a solution that will (1) cluster research reports into their respective industry sectors, (2) carry out sentiment analysis to determine the investment sentiment of each report - 'Bullish' or 'Bearish'.

Contributed by: Kevin Moe, Desmond Ho, Lim Hui Ting, Catherine Leo, Ong Hui Lin and Siddhant Agrawal.<br> 
Pls contact desmondhw@gmail.com for any questions.

**Data**:<br>

Research reports from a prominent investment bank was used. The data are in pickle format - banking.pkl, healthcare.pkl, realestate.pkl, energy.pkl.

**Codes**:
Follow these steps sequentially for runnings Python files in the Codes folder:<br>

1) Run 'functions.py' first to load functions.<br>

2) Download all the pickle files - 'banking.pkl', 'healthcare.pkl', 'realestate.pkl', 'energy.pkl', 'labels.pkl', 'labels_files.pkl'.<br>
3) Download 'stopwords.txt'. This file contains the list of stopwords the team came up with manually that was more suited for our buisness problem<br>

4) Download and <br>
Run 'TF_best.py' for  clustering using TF vectorization.<br>
Run 'TFIDF_best.py' for clustering using TFIDF vectorization.<br>
Run 'word2vec_best.py' for clustering using word embeddings.<br>
Run 'findSentiment.py' first which will generate 'sentiment_forInputs.xlsx'. Then 'Sentiment_Analysis.py' for Logistic Regression Classifier of invesment sentiments.<br><br>

Both TFIDF and word2vec clustering both gave good purities (96-97%)
Opted to work with the TFIDF clustering model as it was simpler, and had fewer parameters to optimize.


**Report**:
Our full report can be found in: 'Text Analytics Project_Report_Group5_final.docx'

**Requirements:** <br>

* Python 3.7 <br>

**Python Libraries Required:**<br>

* numpy<br>
* pandas<br>
* scikit-learn<br>
* matplotlib.pyplot<br>
* nltk<br>
* gensim<br>

Python libraries above can be installed via `pip`. Note that we only tested with the versions above, newer versions might not work.

