import math
import nltk
import os
import numpy as np
from collections import Counter
import xml.etree.ElementTree as ET
from nltk import word_tokenize, sent_tokenize, FreqDist
from nltk.util import ngrams
from os import listdir, system
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import linear_model, ensemble


## return: list of excerpts
def load_excerpts(file_path):
	excerpts = []
	f = open(file_path)
	for line in f:
		excerpts.append(line.decode('utf-8'))
	f.close()
	return excerpts

def load_train_score(file_path):
	train_score_list = []
	f = open(file_path)
	for line in f:
		train_score_list.append(int(line.decode('utf-8')))
	f.close()
	return train_score_list

## return: list of list of words
def tokenize_excerpt(excerpt):
	return [word for sent in sent_tokenize(excerpt) for word in word_tokenize(sent)]


## features
# vocab_size, frac_freq, frac_rare, median, average
def get_unigram_feat(input_string):
	word_list = tokenize_excerpt(input_string)
	count_dict = FreqDist(word_list)

	# type vocab_size
	vocab_size = len(count_dict)
	lst = [vocab_size]
	
	freq_count = 0
	rare_count = 0
	for word in count_dict:
		if count_dict[word] > 5:
			freq_count += 1
		elif count_dict[word] == 1:
			rare_count += 1
	# frac_freq frac_rare
	lst.append(float(freq_count)/vocab_size)
	lst.append(float(rare_count)/vocab_size)

	# median_word average_word
	len_list = []
	for word in word_list:
		len_list.append(len(word))
	lst.append(np.median(len_list))
	lst.append(np.mean(len_list))
	return lst



###################### CoreNPL from HW3
###### 4 ######################################################################################

def extract_pos_tags(xml_directory):
	# Param:    - (str) xml directory
	# Return:   - (list of str) all the unique pos tags from all documents in xml directory, sorted alphabetically
	tag_set = set()
	for xml_file in listdir(xml_directory):
		tree = ET.parse(xml_directory + xml_file)
		root = tree.getroot()
		for sentence in root[0][0]:
			for token in sentence[0]:
				tag_set.add(token.find('POS').text)
	return sorted(list(tag_set))


def map_pos_tags(xml_filename, pos_tag_list):
	# Param:   - (str) xml_filename: an xml file path
	# Param:   - (list of str) pos_tag_list: list of known POS tags
	# Return:  - (list of float) a vector in the feature space of the known POS tag list
	pos_tags = []
	tree = ET.parse(xml_filename)
	root = tree.getroot()
	for sentence in root[0][0]:
		for token in sentence[0]:
			pos_tags.append(token.find('POS').text)
	token_num = len(pos_tags)
	tag_counter = Counter(pos_tags)
	return [1.0 * tag_counter.get(tag, 0) / token_num for tag in pos_tag_list]


def map_universal_tags(ptb_pos_feat_vector,
                       pos_tag_list, ptb_google_mapping, universal_tag_list):
	# Param:   - (list of float) ptb_pos_feat_vector: output vector from section 4.2
	# Param:   - (list of str) pos_tag_list: output from section 4.1
	# Param:   - (dict: str --> str) ptb_google_mapping: a mapping from PTB tags to Google universal tags
	# Param:   - (list of str) universal_tag_list: list of Google universal tags sorted in alphabetical order
	# Return:  - (list of float) a vector in the feature space of the universal tag list
	uni_dict = {}
	for i in range(len(pos_tag_list)):
		pos_tag = pos_tag_list[i]
		uni_tag = ptb_google_mapping[pos_tag]
		uni_dict[uni_tag] = uni_dict.get(uni_tag, 0) + ptb_pos_feat_vector[i]
	return [uni_dict.get(tag, 0.0) for tag in universal_tag_list]

###### 5 ######################################################################################

def extract_ner_tags(xml_directory):
	# Param:   - (str) xml directory
	# Return:  - (list of str) all the unique NER tags from all documents in xml directory, sorted alphabetically
	tag_set = set()
	for xml_file in listdir(xml_directory):
		if xml_file.endswith(".xml"):
			tree = ET.parse(xml_directory + xml_file)
			root = tree.getroot()
			for sentence in root[0][0]:
				for token in sentence[0]:
					tag_set.add(token.find('NER').text)
	return sorted(list(tag_set))

def map_named_entity_tags(xml_filename, entity_list):
	# Param:   - (str) xml_filename: an xml file path
	# Param:   - (list of str) entity_list: list of named entity classes (output from 5.1)
	# Return:  - (list of float) a vector in the feature space of the Named Entity list
	ner_tags = []
	tree = ET.parse(xml_filename)
	root = tree.getroot()
	for sentence in root[0][0]:
		for token in sentence[0]:
			ner_tags.append(token.find('NER').text)
	token_num = len(ner_tags)
	tag_counter = Counter(ner_tags)
	return [1.0 * tag_counter.get(tag, 0) / token_num for tag in entity_list]


###### 6 ######################################################################################

def extract_dependencies(xml_directory):
	# Param:   - (str) xml directory
	# Return:  - (list of str) all the unique NER tags from all documents in xml directory, sorted alphabetically
	dep_set = set()
	for xml_file in listdir(xml_directory):
		if xml_file.endswith(".xml"):
			tree = ET.parse(xml_directory + xml_file)
			root = tree.getroot()
			for sentence in root[0][0]:
				for dependency in sentence[2]:
					dep_set.add(dependency.attrib.get("type"))
	return sorted(list(dep_set))

def map_dependencies(xml_filename, dependency_list):
	# Param:   - (str) xml_filename: an xml file path
	# Param:   - (list of str) dependency_list: the list of dependency types (output in 6.1) as input
	# Return:  - (list of float) Each element takes the value of the number of times the corresponding dependency in  
	# 			dependency list appeared in the xml input file normalized by the number of all dependencies in the text
	dependencies = []
	tree = ET.parse(xml_filename)
	root = tree.getroot()
	for sentence in root[0][0]:
		for dependency in sentence[2]:
			dependencies.append(dependency.attrib.get("type"))
	dep_num = len(dependencies)
	dep_counter = Counter(dependencies)
	if dep_num == 0:
		return [0.0 for dep in dependency_list]
	else:
		return [1.0 * dep_counter.get(dep, 0) / dep_num for dep in dependency_list]


###### 7 ######################################################################################

# Helper function: to parse a single production rule recursively
def tree_parser(rule):
	# Param:   - (str) rule
	rule = rule.strip()
	if rule.startswith("(") and rule.endswith(")"):
		return tree_parser(rule[1:-1])
	elif "(" not in rule or ")" not in rule:
		return []
	# find all '(', ')' at the level of current child
	left_par_list = []
	right_par_list = []
	num_of_unpaired = 0
	for i in range(len(rule)):
		if rule[i] == "(":
			num_of_unpaired += 1
			if num_of_unpaired == 1:
				left_par_list.append(i)
		elif rule[i] == ")":
			num_of_unpaired -= 1
			if num_of_unpaired == 0:
				right_par_list.append(i)
	
	parent = [rule[0:left_par_list[0]].strip()]
	for i in range(len(left_par_list)):
		parent.append(rule[left_par_list[i]+1:right_par_list[i]].split()[0])
	result = ["_".join(parent)]
	for i in range(len(left_par_list)):
		child = tree_parser(rule[left_par_list[i]+1:right_par_list[i]])
		if len(child) > 0:
			result += child
	return result

def extract_prod_rules(xml_directory):
	# Param:   - (str) xml directory
	# Return:  - (list of str) all the unique syntactic production rules for the parse trees in xml directory, sorted alphabetically
	rule_set = set()
	for xml_file in listdir(xml_directory):
		if xml_file.endswith(".xml"):
			tree = ET.parse(xml_directory + xml_file)
			root = tree.getroot()
			for sentence in root[0][0]:
				for rule in tree_parser(sentence.find('parse').text):
					rule_set.add(rule)
	return sorted(list(rule_set))


def map_prod_rules(xml_filename, rules_list):
	# Param:   - (str) xml_filename: an xml file path
	# Param:   - (list of list of str) rules_list: a list of rules in the format specified above
	# Return:  - (list of 0/1 integer) The element in the list takes the value 1 if its corresponding rule in rules list 
	# 				appeared in the xml input file, 0 otherwise.
	file_rule_set = set()
	tree = ET.parse(xml_filename)
	root = tree.getroot()
	for sentence in root[0][0]:
		for rule in tree_parser(sentence.find('parse').text):
			file_rule_set.add(rule)
	return [1 if rule in file_rule_set else 0 for rule in rules_list]


###### 8 ######################################################################################

# Helper function for Brown Clustering
def generate_cluster_codes(brown_file_path):
	# Param:   - (str) brown_file_path: brown cluster file path
	# Return:  - (list of str) a list of unique cluster names/codes present in the file
	code_list = []
	f = open(brown_file_path, 'r')
	for line in f:
		code = line.split()[0].strip()
		if len(code_list) == 0 or code != code_list[len(code_list) -1]:
			code_list.append(code)
	f.close()
	code_list.append("8888") # for words not in Brown clusters
	return code_list

# Helper function for Brown Clustering
def generate_word_cluster_mapping(brown_file_path):
	# Param:   - (str) brown_file_path: path to the brown cluster file
	# Return:  - (dict: str -> str) a dict containing a mapping from words occurring in the brown cluster files to their cluster codes
	cluster_dict = {}
	f = open(brown_file_path, 'r')
	for line in f:
		split_list = line.split()
		word = split_list[1].strip()
		code = split_list[0].strip()
		cluster_dict[word] = code
	f.close()
	return cluster_dict


def map_brown_clusters(xml_file_path, cluster_code_list, word_cluster_mapping):
	# Param:   - (str) xml_file_path
	# Param:   - (list of str) cluster_code_list: output of the helper function
	# Param:   - (dict: str -> str) word_cluster_mapping: output of the helper function
	# Return:  - (list of float) a representation that reflects the normalized frequency of each known Brown cluster in the given text
	word_list = []
	tree = ET.parse(xml_file_path)
	root = tree.getroot()
	for sentence in root[0][0]:
		for token in sentence[0]:
			word_list.append(token.find('word').text)
	word_num = len(word_list)
	cluster_list = [word_cluster_mapping.get(word, "8888") for word in word_list]
	cluster_counter = Counter(cluster_list)
	return [1.0 * cluster_counter.get(code, 0) / word_num for code in cluster_code_list]


###### 9 ######################################################################################

def createPOSFeat(xml_dir, pos_tag_list):
	file_list = sorted(listdir(xml_dir), key = lambda x : int(x.split('.')[0].split('_')[-1]))
	result = []
	for xml_file in file_list:
		result.append(map_pos_tags(xml_dir + xml_file, pos_tag_list))
	return np.array(result)


def createUniversalPOSFeat(pos_feat_2D_array, pos_tag_list, ptb_google_mapping, universal_tag_list):
	result = []
	for feat_array in pos_feat_2D_array:
		## TODO: check if np.array can work as list
		result.append(map_universal_tags(feat_array, pos_tag_list, ptb_google_mapping, universal_tag_list))
	return np.array(result)


def createNERFeat(xml_dir, entity_list):
	file_list = sorted(listdir(xml_dir), key = lambda x : int(x.split('.')[0].split('_')[-1]))
	result = []
	for xml_file in file_list:
		result.append(map_named_entity_tags(xml_dir + xml_file, entity_list))
	return np.array(result)


def createDependencyFeat(xml_dir, dependency_list):
	file_list = sorted(listdir(xml_dir), key = lambda x : int(x.split('.')[0].split('_')[-1]))
	result = []
	for xml_file in file_list:
		result.append(map_dependencies(xml_dir + xml_file, dependency_list))
	return np.array(result)


def createSyntaticProductionFeat(xml_dir, rules_list):
	file_list = sorted(listdir(xml_dir), key = lambda x : int(x.split('.')[0].split('_')[-1]))
	result = []
	for xml_file in file_list:
		result.append(map_prod_rules(xml_dir + xml_file, rules_list))
	return np.array(result)


def createBrownClusterFeat(xml_dir, cluster_code_list, word_cluster_mapping):
	file_list = sorted(listdir(xml_dir), key = lambda x : int(x.split('.')[0].split('_')[-1]))
	result = []
	for xml_file in file_list:
		result.append(map_brown_clusters(xml_dir + xml_file, cluster_code_list, word_cluster_mapping))
	return np.array(result)


def run_classifier(X_train, y_train, X_test, predicted_labels_file):
	clf = SVC()
	clf.fit(X_train, y_train)

	y_test = clf.predict(X_test)
	f = open(predicted_labels_file, 'w')
	for y in y_test:
		f.write(str(y) + '\n')
	f.close()
	return


###### 10 #####################################################################################

## unigram
def extract_top_k_unigrams(doc_directory, k):
	# Param:    - (str) doc directory
	token_list = []
	for file in listdir(doc_directory):
		f = open(doc_directory + file, 'r')
		for sent in sent_tokenize(f.read().decode('utf-8')):
			token_list.extend(word_tokenize(sent))
		f.close()
	return [pair[0] for pair in Counter(token_list).most_common(k)]

def map_k_top_unigrams(doc_filename, top_k_unigrams_list):
	tokens = []
	f = open(doc_filename, 'r')
	for sent in sent_tokenize(f.read().decode('utf-8')):
		tokens.extend(word_tokenize(sent))
	token_num = len(tokens)
	unigram_counter = Counter(tokens)
	return [1.0 * unigram_counter.get(unigram, 0) / token_num for unigram in top_k_unigrams_list]

def createUnigramFeat(doc_dir, top_k_unigrams_list):
    file_list = sorted(listdir(doc_dir), key = lambda x : int(x.split('.')[0].split('_')[-1]))
    result = []
    for file in file_list:
        result.append(map_k_top_unigrams(doc_dir + file, top_k_unigrams_list))
    return np.array(result)


## bigram
def extract_top_k_bigrams(doc_directory, k):
	# Param:    - (str) doc directory
	bigram_list = []
	for file in listdir(doc_directory):
		f = open(doc_directory + file)
		for sent in sent_tokenize(f.read().decode('utf-8')):
			curt_list = list(ngrams(word_tokenize(sent), 2, pad_left=True, pad_right=True))
			bigram_list.extend(curt_list)
		f.close()
	top_k = Counter(bigram_list).most_common(k)
	if k > len(top_k):
		print top_k[len(top_k) -1]
	else:
		print top_k[k-1]
	return [pair[0] for pair in top_k]


def map_k_top_bigrams(doc_filename, top_k_bigrams_list):
	bigrams = []
	f = open(doc_filename, 'r')
	for sent in sent_tokenize(f.read().decode('utf-8')):
		curt_list = list(ngrams(word_tokenize(sent), 2, pad_left=True, pad_right=True))
		bigrams.extend(curt_list)
	f.close()
	bigram_num = len(bigrams)
	bigram_counter = Counter(bigrams)
	if bigram_num == 0:
		return [0.0 for bigram in top_k_bigrams_list]
	return [1.0 * bigram_counter.get(bigram, 0) / bigram_num for bigram in top_k_bigrams_list]


def createBigramFeat(doc_dir, top_k_bigrams_list):
	file_list = sorted(listdir(doc_dir), key = lambda x : int(x.split('.')[0].split('_')[-1]))
	result = []
	for file in file_list:
		result.append(map_k_top_bigrams(doc_dir + file, top_k_bigrams_list))
	return np.array(result)


if __name__ == "__main__":
	# input_dir = "/home1/c/cis530/project/data/"
	# output_dir = "/home1/b/boyihe/CIS530/pro/"

	input_dir = "./data/"

	train_excerpt_list = load_excerpts(input_dir + "project_train.txt")
	train_score_list = load_train_score(input_dir + "project_train_scores.txt")
	test_list = load_excerpts(input_dir + "project_test.txt")

	# ############# vocab_size, frac_freq, frac_rare, median, average (hw2)
	# train_lst = []
	# for excerpt in train_excerpt_list:
	# 	train_lst.append(get_unigram_feat(excerpt))

	############# NPL (hw3)
	hw_path = "./"
	train_path = "./train_split/"
	test_path = "./test_split/"

	train_xml_path = hw_path + "train_xml/"
	test_xml_path = hw_path + "test_xml/"

	brown_file_path = hw_path + "brown-rcv1.clean.tokenized-CoNLL03.txt-c100-freq1.txt"

	pos_tag_list = extract_pos_tags(train_xml_path)

	ptb_google_mapping = {}
	f = open("./en-ptb.map")
	for line in f:
		ptb, uni = line.split('\t')
		ptb_google_mapping[ptb] = uni[:-1]	# remove the ending '\n' from uni tag
	f.close()
	universal_tag_list = sorted(list(set(ptb_google_mapping.values())))

	entity_list = extract_ner_tags(train_xml_path)
	dependency_list = extract_dependencies(train_xml_path)
	rules_list = extract_prod_rules(train_xml_path)
	cluster_code_list = generate_cluster_codes(brown_file_path)
	word_cluster_mapping = generate_word_cluster_mapping(brown_file_path)

	# get features for train
	pos_feat_train = createPOSFeat(train_xml_path, pos_tag_list)
	uni_feat_train = createUniversalPOSFeat(pos_feat_train, pos_tag_list, ptb_google_mapping, universal_tag_list)
	ner_feat_train = createNERFeat(train_xml_path, entity_list)
	dep_feat_train = createDependencyFeat(train_xml_path, dependency_list)
	syn_feat_train = createSyntaticProductionFeat(train_xml_path, rules_list)
	clu_feat_train = createBrownClusterFeat(train_xml_path, cluster_code_list, word_cluster_mapping)

	# get features for test
	# pos_feat_test = createPOSFeat(test_xml_path, pos_tag_list)
	# uni_feat_test = createUniversalPOSFeat(pos_feat_test, pos_tag_list, ptb_google_mapping, universal_tag_list)
	# ner_feat_test = createNERFeat(test_xml_path, entity_list)
	# dep_feat_test = createDependencyFeat(test_xml_path, dependency_list)
	# syn_feat_test = createSyntaticProductionFeat(test_xml_path, rules_list)
	# clu_feat_test = createBrownClusterFeat(test_xml_path, cluster_code_list, word_cluster_mapping)

	# X_train = np.concatenate(
 #        (pos_feat_train, uni_feat_train, ner_feat_train, dep_feat_train, syn_feat_train, clu_feat_train), axis = 1)
	# X_test = np.concatenate(
 #        (pos_feat_test, uni_feat_test, ner_feat_test, dep_feat_test, syn_feat_test, clu_feat_test), axis = 1)

	X_train = ner_feat_train

	train_file_list = sorted(listdir(train_xml_path), key = lambda x : int(x.split('.')[0].split('_')[-1]))

	## serilize
	# np.save("./hw3_npy/pos_train", pos_feat_train)
	# np.save("./hw3_npy/pos_test", pos_feat_test)
	# np.save("./hw3_npy/uni_train", uni_feat_train)
	# np.save("./hw3_npy/uni_test", uni_feat_test)
	# np.save("./hw3_npy/ner_train", ner_feat_train)
	# np.save("./hw3_npy/ner_test", ner_feat_test)
	# np.save("./hw3_npy/dep_train", dep_feat_train)
	# np.save("./hw3_npy/dep_test", dep_feat_test)
	# np.save("./hw3_npy/syn_train", syn_feat_train)
	# np.save("./hw3_npy/syn_test", syn_feat_test)
	# np.save("./hw3_npy/clu_train", clu_feat_train)
	# np.save("./hw3_npy/clu_test", clu_feat_test)

	############# unigram & bigram (hw3)
	# top_k_unigrams_list = extract_top_k_unigrams(train_path, 5000)
	# unigram_feat_train = createUnigramFeat(train_path, top_k_unigrams_list)
	# unigram_feat_test = createUnigramFeat(test_path, top_k_unigrams_list)
	# np.save("unigram_train", unigram_feat_train)

	## cross validate
	# clf = linear_model.LogisticRegression(penalty = 'l1', C = 100, max_iter = 300, solver = 'liblinear', multi_class = 'ovr')
	clf = SVC(C = 5, kernel = 'linear')
	scores = cross_val_score(clf, X_train, train_score_list, cv = 5)

	# # print scores
	print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))





