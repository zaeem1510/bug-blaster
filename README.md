# bug-blaster
Bug classification and similar bug retrieval through JIRA tickets.

**Background:**
Software deveopement teams often encounter numerous bugs during the develpment and maintenance phases. Identifying the types of bugs and efficiently resolving them is crucial for the delivering high-quality software. Manual bug classification and searching for solutions from the existing tickets can be time consuming and error prone.

**Objective:**
The objectove was to develop an intelligent machine learning model that automates the bug classification process and make the retrieval of similar bugs faster and hassle free. The model inputs bug summaries and accurately classifies the bugs into predefined categories or clusters. Additionally, the system provides a feature to retrieve similar bugs based on the input summary, assisting developers in finding relevant solutions and accelerating the debugging process.

**Note:** Different excel files have been used for training the model. Please change the path of the read_csv/read_excel to your file.

1) **Data Collection and Preprocessing:** Gathering a JIRA dataset with bugs' tikcets and correspoding bug summaries. Preprocessing techniques were applied to clean the data. The dimensions of the dataest is a prone errors as the training, testing and prediction datasets have to be dimensionally coherrent. NaN values must also be removed to avoid errors due to incompatibility of data types.

2) **Model developement:** Designing and training the supervised ML model to classify the bugs. Logistic regression and K-Means clustering were the different algorithms used for training the model.

3) **Similar bug retrieval:** Implementing a search mechanism by returning the most similar bug based on the similarity of the summary.

**Logistic Regression Model**(Refer to Bug_Prediction.ipynb in the main)

**Note**: It is recommended that the notebook be run on an online console for minimizing errors while importing libraries.

A detailed description of all the sections of code has been given in the notebook above each code snippet. 
**Libraries used:**

1) NumPy: NumPy is the fundamental package for scientific computing in Python. It is a Python library that provides a multidimensional array object, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more.
2) pandas: pandas is a Python package providing fast, flexible, and expressive data structures designed to make working with “relational” or “labeled” data both easy and intuitive. It aims to be the fundamental high-level building block for doing practical, real-world data analysis in Python.
3) genism: Gensim is a Python library for topic modelling, document indexing and similarity retrieval with large corpora. Target audience is the natural language processing (NLP) and information retrieval (IR) community.
4) sklearn: Scikit-learn is an open source machine learning library that supports supervised and unsupervised learning. It also provides various tools for model fitting, data preprocessing, model selection, model evaluation, and many other utilities
5) matplotlib: Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.

**Note:** genism library needs to be manually installed on the machine to execute the functionality of NLP through it. If the notebook is run on an online console, the installation will happen autoatically through a dedicated section meant to download genism as a plugin/api.

**Reading the file:**
The file is read to the variable dataset through the read_excel method. This may be changed to read_csv according the file type that needs to be read.

**Vectorization and Tokenization:**
Fundamentals idea for executing NLP is to convert all the summries in the given dataset to tokens and then convert them to vectors. Two respective functions spacy_tokenizer and sent_vec hae been defined for this purpose.

**Training and testing:**
The test size may be manipulated according to the size of the input dataset. As the input dataset grows, it is advised to decrease the test size slighlty to avoid underfitting. Underfitting is amore likely scenario than overfitting in regards to this dataset as the summaries are highly diverse in nature after tokenization.

**Accuracy Metrics:**
Accuracy can be measured using accuracy_score or f1_score metrics but they pose error while converting float to string data types and hence cannot be used. 


**K-Means Clustering:**(Refer to bug_prediction_kmeans.ipynb)

**Note**: The model formed clusters of the training datset but was unable to predict an output due to It is recommended that the notebook be run on an online console for minimizing errors while importing libraries.

**Libraries used:**

1) NumPy: NumPy is the fundamental package for scientific computing in Python. It is a Python library that provides a multidimensional array object, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more.
2) pandas: pandas is a Python package providing fast, flexible, and expressive data structures designed to make working with “relational” or “labeled” data both easy and intuitive. It aims to be the fundamental high-level building block for doing practical, real-world data analysis in Python.
3) nltk: It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries. **Note:** nltk will be downloaded automatically if the notebook is run on an online console but it needs to be installed manually on the machine if the notebook is run locally. 
4) sklearn: Scikit-learn is an open source machine learning library that supports supervised and unsupervised learning. It also provides various tools for model fitting, data preprocessing, model selection, model evaluation, and many other utilities.
5) matplotlib: Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.

**Reading the file:**
The file is read to the variable dataset through the read_excel method. This may be changed to read_csv according the file type that needs to be read.

**Stemming and tokenizing:**
Two respective functions were defined to tokeize and stem the summaries and only tokenize the summaries. Then, the stems of the tokens were appended to a list _totalvocab_tokenized_.

**Vectorization:**
Term Frequency-Inverse Document Frequency was used to implement vectorization of the trainig dataset. Here, ngram_range takes unigrams, bigrams and trigrams into consideration. fit_transform method was called to scale the training dataset after finding the mean and standard deviation.

**Note:** Here, it must be noted that only the x_train data was fitted and transformed (wihtout the y_train) and stored in matrix. This matrix was used in the following section to for modelling. This might be the possible source of error which does not allow the model to perform bug retrieval.

Similarity is calculated here on the basis of cosine similarity rather than euclidean distance because the length of each summary is different. The cosine similarity is calculated on the basis of angle where lesser the angle, higher the similarity(always in the range of 0 to 1)
**K-means Clustering:**
10 clusters were predetermined to classify the tickets into types. The number of clusters has a linear relationship with overfitting hence, to avoid overfitting, the number of clusters must be reduced and vice versa for underfitting.
**Note:** The model was pickled to allow reloading the model by storing it in km. This step is not necessary but included to improve modularity.

**Multidimensional Scaling:**
Multidimensional scaling is used to convert the distance calculated earlier (through cosine similarity) into a 2-dimensioanl array containing the cooradinates of the tickets that can be plotted.

**Visualization:**
Cluster names and colors are defined in respective dictionaries. The tickets were clustered based on the assigned colours. Tickets with higher similarity are clustered together and the ones that have least similarity are the farthest. **Note:** The plot visualizes the distance between the vectorized tickets and hence has no dependence on the training of the model.
