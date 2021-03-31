import os
import random
import numpy as np
import pandas as pd
# import igraph
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn import preprocessing
import nltk
from nltk.tokenize import word_tokenize
import csv
import networkx as nx
from tqdm import tqdm

parent_dir = os.getcwd()

nltk.download('punkt') # for tokenization
nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()

testing_set_path = os.path.join(parent_dir, "testing_set.txt")
with open(testing_set_path, "r") as f:
    reader = csv.reader(f)
    testing_set  = list(reader)

testing_set = [element[0].split(" ") for element in testing_set]

###################
# random baseline #
###################

random_predictions = np.random.choice([0, 1], size=len(testing_set))
random_predictions = zip(range(len(testing_set)),random_predictions)

random_predictions_path = os.path.join(parent_dir, "random_predictions.csv")
with open(random_predictions_path,"w") as pred:
    csv_out = csv.writer(pred)
    for row in random_predictions:
        csv_out.writerow(row)
        
# note: Kaggle requires that you add "ID" and "category" column headers

###############################
# beating the random baseline #
###############################

# the following script gets an F1 score of approximately 0.66

# data loading and preprocessing 

# the columns of the data frame below are: 
# (1) paper unique ID (integer)
# (2) publication year (integer)
# (3) paper title (string)
# (4) authors (strings separated by ,)
# (5) name of journal (optional) (string)
# (6) abstract (string) - lowercased, free of punctuation except intra-word dashes

training_set_path = os.path.join(parent_dir, "training_set.txt")
with open(training_set_path, "r") as f:
    reader = csv.reader(f)
    training_set  = list(reader)

training_set = [element[0].split(" ") for element in training_set]

node_information_path = os.path.join(parent_dir, "node_information.csv")
with open(node_information_path, "r") as f:
    reader = csv.reader(f)
    node_info  = list(reader)

IDs = [element[0] for element in node_info]

# compute TFIDF vector of each paper
corpus = [element[5] for element in node_info]
vectorizer = TfidfVectorizer(stop_words="english")
# each row is a node in the order of node_info
features_TFIDF = vectorizer.fit_transform(corpus)

name_cols = ['id', 'year', 'title', 'authors', 'journal', 'abstract']
node_information = pd.read_csv(node_information_path, header=None, names=name_cols)

def stem_text(text):
    text_list = word_tokenize(text)
    text_list = [stemmer.stem(token) for token in text_list]
    return ' '.join(text_list)

node_information['stem_title'] = node_information['title'].apply(stem_text)
features_title = TfidfVectorizer().fit_transform(node_information['stem_title'])

def generate_graph(path):
    X = np.genfromtxt(path, delimiter=' ').astype(int)
    G = nx.Graph()

    for row in tqdm(X):
        node1, node2, edge = tuple(row)
        G.add_node(node1)
        G.add_node(node2)
        if edge:
            G.add_edge(node1, node2)
    return G
    
graph = generate_graph(training_set_path)

def extract_features(source, 
                     target, 
                     node_information=node_information, 
                     features_TFIDF=features_TFIDF, 
                     features_title=features_title,
                     graph=graph):

    source_index = node_information[node_information['id'] == source].index[0]
    target_index = node_information[node_information['id'] == target].index[0]
    
    sim_abstract = features_TFIDF[source_index].toarray().reshape(-1) @ features_TFIDF[target_index].toarray().reshape(-1)
    
    sim_title = features_title[source_index].toarray().reshape(-1) @ features_title[target_index].toarray().reshape(-1)
    
    plus_or_minus = lambda boolean : 1 if boolean else -1
    same_journal = plus_or_minus(node_information['journal'][source_index] == node_information['journal'][target_index])
    
    set_neigh_source = set(nx.neighbors(graph, source))
    set_neigh_target = set(nx.neighbors(graph, target))
    
    shared_neigh = len(set_neigh_source & set_neigh_target) # / len(set_neigh_source | set_neigh_target)
    
    auth_source = node_information[node_information['id'] == source]['authors'].values[0]
    auth_target = node_information[node_information['id'] == target]['authors'].values[0]
    
    if type(auth_source) == str:
        source_auth = auth_source.split(","); source_auth = set([auth.strip(' ') for auth in source_auth])
    else:
        source_auth = set()
    if type(auth_target) == str:
        target_auth = auth_target.split(","); target_auth = set([auth.strip(' ') for auth in target_auth])
    else:
        target_auth = set()
    
    shared_authors = len(source_auth & target_auth)
    
    return sim_abstract, sim_title, same_journal, shared_neigh, shared_authors
    
    


# for each training example we need to compute features
# in this baseline we will train the model on only 5% of the training set

# randomly select 5% of training set
to_keep = random.sample(range(len(training_set)), k=int(round(len(training_set)*0.05)))
training_set_reduced = [training_set[i] for i in to_keep]

# we will use three basic features:

training_features = []

counter = 0
for i in range(len(training_set_reduced)):
    source = int(training_set_reduced[i][0])
    target = int(training_set_reduced[i][1])
    
    sim_abstract, sim_title, same_journal, shared_neigh, shared_authors = extract_features(source, target)
    
    row = [sim_abstract, sim_title, same_journal, shared_neigh, shared_authors]
    
    training_features.append(row)
   
    counter += 1
    if counter % 1000 == True:
        print(counter, "training examples processsed")

# convert list of lists into array
# documents as rows, unique words as columns (i.e., example as rows, features as columns)
training_features = np.array(training_features)

# scale
training_features = preprocessing.scale(training_features)

# convert labels into integers then into column array
labels = [int(element[2]) for element in training_set_reduced]
labels = list(labels)
labels_array = np.array(labels)

# initialize basic SVM
classifier = svm.LinearSVC()

# train
classifier.fit(training_features, labels_array)

# test
# we need to compute the features for the testing set

testing_features = []
   
counter = 0
for i in range(len(testing_set)):
    source = int(testing_set[i][0])
    target = int(testing_set[i][1])
    
    sim_abstract, sim_title, same_journal, shared_neigh, shared_authors = extract_features(source, target)
    
    row = [sim_abstract, sim_title, same_journal, shared_neigh, shared_authors]
    testing_features.append(row)
    
    counter += 1
    if counter % 1000 == True:
        print(counter, "testing examples processsed")
        
# convert list of lists into array
# documents as rows, unique words as columns (i.e., example as rows, features as columns)
testing_features = np.array(testing_features)

# scale
testing_features = preprocessing.scale(testing_features)

# issue predictions
predictions_SVM = list(classifier.predict(testing_features))

# write predictions to .csv file suitable for Kaggle (just make sure to add the column names)
predictions_SVM = zip(range(len(testing_set)), predictions_SVM)

improved_predictions_path = os.path.join(parent_dir, "improved_predictions.csv")

idx = list(range(len(predictions_SVM)))
preds_data = np.array([idx, predictions_SVM]).T
preds = pd.DataFrame(data=preds_data, columns=['id', 'category'])
preds.to_csv(improved_predictions_path, index=False)
