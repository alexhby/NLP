import math
import nltk
import os
import numpy as np
import xml.etree.ElementTree as ET
from collections import Counter
from nltk import word_tokenize, sent_tokenize, FreqDist
from nltk.util import ngrams
from operator import itemgetter 
from os import listdir, system
from scipy import stats
from sklearn import linear_model
from sklearn.linear_model import BayesianRidge, Lasso, LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC, SVR
from random import sample
import pickle
import operator
from collections import defaultdict


####################### PART 1 - PREPARATION ############################


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


def load_optional_score(file_path):
    optional_file_list = []
    optional_score_list = []
    f = open(file_path)
    for line in f:
        parts = line.split()
        optional_file_list.append(parts[0])
        optional_score_list.append(-float(parts[1]))
    f.close()
    return optional_file_list, optional_score_list


def load_optional_excerpts(folder_path, file_list):
    excerpts = []
    for file in file_list:
        f = open(folder_path + file + ".txt")
        excerpt = ""
        for line in f:
            if line != "\r\n":
                excerpt += line
        f.close()
        excerpts.append(excerpt.decode('utf-8'))    
    return excerpts



####################### PART 2 - EVALUATION FUNCTION ####################
def compute_spearman_correlation(ground_truth, predictions):
    return stats.spearmanr(ground_truth, predictions).correlation



####################### PART 3 - CROSS VALIDATION #######################
def split_list(input_list, cv):
	rand_list = copy(input_list)
	shuffle(rand_list)
	split_list = []
	length = len(rand_list)
	sublength = int(math.ceil(float(length)/cv))
	idx = 0
	for i in xrange(cv - 1):
		sublist = []
		for j in xrange(sublength):
			sublist.append(rand_list[idx + j])
		split_list.append(sublist)
		idx += sublength
	## append last sublists
	sublist = []
	while idx < length:
		sublist.append(rand_list[idx])
		idx += 1
	split_list.append(sublist)
	return split_list


def cross_valid_with_spearman(x_train, y_train, cv = 5):
	x_split_list = split_list(x_train, cv)
	y_split_list = split_list(y_train, cv)

	reg = BayesianRidge()
	scores = []
	for i in xrange(cv):
		x_train_sample = copy(x_split_list)
		x_valid = x_train_sample[i]
		del x_train_sample[i]
		x_train_sample = [item for sublist in x_train_sample for item in sublist]   ## flatten

		y_train_sample = copy(y_split_list)
		y_valid = y_train_sample[i]
		del y_train_sample[i]
		y_train_sample = [item for sublist in y_train_sample for item in sublist]

		reg.fit(x_train_sample, y_train_sample)
		estimate = reg.predict(x_valid)
		scores.append(compute_spearman_correlation(y_valid, estimate).correlation)
		reg.fit(x_valid, y_valid)
		estimate = reg.predict(x_train_sample)
		scores.append(compute_spearman_correlation(y_train_sample, estimate).correlation)
	return scores



####################### PART 4 - CLASSICAL FEATURES #####################
def tokenize_excerpt(excerpt):
	return [word for sent in sent_tokenize(excerpt) for word in word_tokenize(sent)]


def get_vocab_size(input_string):
    token_list = tokenize_excerpt(input_string)
    count_dict = FreqDist(token_list)
    vocab_size = len(count_dict)
    return vocab_size


def get_frac_freq(input_string):
    token_list = tokenize_excerpt(input_string)
    count_dict = FreqDist(token_list)
    vocab_size = len(count_dict)
    freq_count = 0
    for word in count_dict:
        if count_dict[word] >= 7:
            freq_count += 1
    frac_freq = float(freq_count) / vocab_size
    return frac_freq


def get_frac_rare(input_string):
    token_list = tokenize_excerpt(input_string)
    count_dict = FreqDist(token_list)
    vocab_size = len(count_dict)
    rare_count = 0
    for word in count_dict:
        if count_dict[word] <= 3:
            rare_count += 1
    frac_rare = float(rare_count) / vocab_size
    return frac_rare


def get_median_length(input_string):
    token_list = tokenize_excerpt(input_string)
    len_list = [len(token) for token in token_list]
    median_length = np.median(len_list)
    return median_length


def get_average_length(input_string):
    token_list = tokenize_excerpt(input_string)
    len_list = [len(token) for token in token_list]
    average_length = np.mean(len_list)
    return average_length


def get_average_sentence_length(input_string):
	sentence_list = sent_tokenize(input_string)
	len_list = [len(word_tokenize(sentence)) for sentence in sentence_list]
	return np.mean(len_list)


def get_type_token_ratio(input_string):
    token_list = tokenize_excerpt(input_string)
    count_dict = FreqDist(token_list)
    vocab_size = len(count_dict)
    return vocab_size * 1.0 / len(token_list)



####################### PART 5 - SYNTACTIC FEATURES #####################
def extract_pos_tags(xml_directory):
	tag_set = set()
	for xml_file in listdir(xml_directory):
		tree = ET.parse(xml_directory + xml_file)
		root = tree.getroot()
		for sentence in root[0][0]:
			for token in sentence[0]:
				tag_set.add(token.find('POS').text)
	return sorted(list(tag_set))


def map_pos_tags(xml_filename, pos_tag_list):
	pos_tags = []
	tree = ET.parse(xml_filename)
	root = tree.getroot()
	for sentence in root[0][0]:
		for token in sentence[0]:
			pos_tags.append(token.find('POS').text)
	token_num = len(pos_tags)
	tag_counter = Counter(pos_tags)
	return [1.0 * tag_counter.get(tag, 0) / token_num for tag in pos_tag_list]


def map_universal_tags(ptb_pos_feat_vector, pos_tag_list, ptb_google_mapping, universal_tag_list):
	uni_dict = {}
	for i in range(len(pos_tag_list)):
		pos_tag = pos_tag_list[i]
		uni_tag = ptb_google_mapping[pos_tag]
		uni_dict[uni_tag] = uni_dict.get(uni_tag, 0) + ptb_pos_feat_vector[i]
	return [uni_dict.get(tag, 0.0) for tag in universal_tag_list]


def extract_ner_tags(xml_directory):
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
	ner_tags = []
	tree = ET.parse(xml_filename)
	root = tree.getroot()
	for sentence in root[0][0]:
		for token in sentence[0]:
			ner_tags.append(token.find('NER').text)
	token_num = len(ner_tags)
	tag_counter = Counter(ner_tags)
	return [1.0 * tag_counter.get(tag, 0) / token_num for tag in entity_list]


def extract_dependencies(xml_directory):
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


def tree_parser(rule):
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
	file_rule_set = set()
	tree = ET.parse(xml_filename)
	root = tree.getroot()
	for sentence in root[0][0]:
		for rule in tree_parser(sentence.find('parse').text):
			file_rule_set.add(rule)
	return [1 if rule in file_rule_set else 0 for rule in rules_list]


def generate_cluster_codes(brown_file_path):
	code_list = []
	f = open(brown_file_path, 'r')
	for line in f:
		code = line.split()[0].strip()
		if len(code_list) == 0 or code != code_list[len(code_list) -1]:
			code_list.append(code)
	f.close()
	code_list.append("8888")
	return code_list


def generate_word_cluster_mapping(brown_file_path):
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


def createPOSFeat(xml_dir, pos_tag_list):
	file_list = sorted(listdir(xml_dir), key = lambda x : int(x.split('.')[0].split('_')[-1]))
	result = []
	for xml_file in file_list:
		result.append(map_pos_tags(xml_dir + xml_file, pos_tag_list))
	return np.array(result)


def createUniversalPOSFeat(pos_feat_2D_array, pos_tag_list, ptb_google_mapping, universal_tag_list):
	result = []
	for feat_array in pos_feat_2D_array:
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


####################### unigram bigram from HW3

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


####################### tf-idf

def flatten(listoflists):
	return [item for sublist in listoflists for item in sublist]

## sample: a list of standardized excerpts
def get_tf(sample):
	return dict(FreqDist(flatten(sample))) # return tf dict

## corpus: a list of standardized excerpts
def get_idf(corpus):
	df_dict = defaultdict(int)
	for excerpt in corpus:
		for word in set(excerpt):
			df_dict[word] += 1
	n = float(len(corpus))
	idf_dict = {}
	for word in df_dict:
		idf_dict[word] = np.log(n / df_dict[word])
	return idf_dict

def get_tfidf(tf_dict, idf_dict):
	d = {}
	for word in tf_dict:
		if word in idf_dict:
			d[word] = tf_dict[word] * idf_dict[word]
	return d

def get_tfidf_weights_topk(tf_dict, idf_dict, k):
	d = get_tfidf(tf_dict, idf_dict)
	lst = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
	return lst[0:k]

## pickle
def save_idf(idf_dict):
	pickle.dump(idf_dict, open("idf.p", "wb"), True)
	return

def load_idf(corpus):
	if os.path.isfile("idf.p") and os.stat("idf.p").st_size != 0:
		idf_dict = pickle.load(open("idf.p", "rb")) # if already pickled, deserialize
	else:
		idf_dict = get_idf(corpus)
		save_idf(idf_dict)
	return idf_dict

def create_feature_space(wordlist):
	index = 0
	d = {}
	for word in wordlist:
		if word not in d:
			d[word] = index
			index += 1
	return d

def vectorize_tfidf(feature_space, idf_dict, sample):
	tfidf_dict = get_tfidf(get_tf(sample), idf_dict)
	vector = [0] * len(feature_space)
	for word in feature_space:
		if word in tfidf_dict:
			vector[feature_space[word]] = tfidf_dict[word]
	return vector

## create the feature matrix
def create_tfidf_feat(excerpt_list):
	corpus = []
	for excerpt in excerpt_list:
		corpus.append(tokenize_excerpt(excerpt))
	corpus_idf_dict = load_idf(corpus)
	corpus_tf_dict = get_tf(corpus)
	corpus_tfidf_list = get_tfidf_weights_topk(corpus_tf_dict, corpus_idf_dict, 1000)
	
	corpus_wordlist = []
	for item in corpus_tfidf_list:
		corpus_wordlist.append(item[0])
	matrix = []
	for excerpt in excerpt_list:
		sample = []
		for sent in sent_tokenize(excerpt):
			sample.append(word_tokenize(sent))
		matrix.append(vectorize_tfidf(create_feature_space(corpus_wordlist), corpus_idf_dict, sample))
	return matrix


####################### mi

def get_word_probs(sample):
	d = get_tf(sample)	# get tf_dict
	total = sum(d.itervalues())
	for word in d:
		d[word] = float(d[word]) / total
	return d

## Consider only words that appear at least 5 times in the corpus.
def get_word_general_probs(corpus):
	d = get_tf(corpus)	# get tf_dict
	total = sum(d.itervalues())
	for word in d.keys():
		if d[word] >= 5:
			d[word] = float(d[word]) / total
		else:
			del d[word]
	return d

def get_mi(sample, corpus):
	d = get_word_probs(sample)
	corpus_dict = get_word_general_probs(corpus)
	for word in d.keys():
		if word in corpus_dict:
			d[word] = np.log(d[word] / corpus_dict[word])
		else:
			del d[word]
	return d

## sample: a list of standardized excerpts
## corpus: a list of standardized excerpts
def get_mi_topk(sample, corpus, k):
	d = get_mi(sample, corpus)
	lst = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
	return lst[0:k]

def vectorize_mi(feature_space, word_probs, sample):
	mi_dict = {}
	cond_probs = get_word_probs(sample)
	for word in cond_probs:
		if word in word_probs:
			mi_dict[word] = np.log(cond_probs[word] / word_probs[word])
	vector = [0] * len(feature_space)
	for word in feature_space:
		if word in mi_dict:
			vector[feature_space[word]] = mi_dict[word]
	return vector

## create the feature matrix
def create_mi_feat(excerpt_list):
	corpus = []
	for excerpt in excerpt_list:
		corpus.append(tokenize_excerpt(excerpt))
	word_probs = get_word_general_probs(corpus)

	corpus_idf_dict = load_idf(corpus)
	corpus_tf_dict = get_tf(corpus)
	corpus_tfidf_list = get_tfidf_weights_topk(corpus_tf_dict, corpus_idf_dict, 1000)
	corpus_wordlist = []
	for item in corpus_tfidf_list:
		corpus_wordlist.append(item[0])

	matrix = []
	for excerpt in excerpt_list:
		sample = []
		for sent in sent_tokenize(excerpt):
			sample.append(word_tokenize(sent))
		matrix.append(vectorize_mi(create_feature_space(corpus_wordlist), word_probs, sample))
	return matrix

###########################################################################

if __name__ == "__main__":
	train_xml_path  = "./train_xml/"
	test_xml_path   = "./test_xml/"
	brown_file_path = "./brown-rcv1.clean.tokenized-CoNLL03.txt-c100-freq1.txt"

	train_excerpt_list = load_excerpts("./data/project_train.txt")
	train_score_list = load_train_score("./data/project_train_scores.txt")
	optional_file_list, optional_score_list = load_optional_score("./data/optional_training/optional_project_train_scores.txt")
	optional_excerpt_list = load_optional_excerpts("./data/optional_training/", optional_file_list)
	test_excerpt_list = load_excerpts("./data/project_test.txt")

	# feat_vocab_size_train       = [[get_vocab_size(excerpt)]              for excerpt in train_excerpt_list]
	# feat_frac_freq_train        = [[get_frac_freq(excerpt)]               for excerpt in train_excerpt_list]
	# feat_frac_rare_train        = [[get_frac_rare(excerpt)]               for excerpt in train_excerpt_list]
	# feat_median_length_train    = [[get_median_length(excerpt)]           for excerpt in train_excerpt_list]
	# feat_average_length_train   = [[get_average_length(excerpt)]          for excerpt in train_excerpt_list]
	# feat_average_sentence_train = [[get_average_sentence_length(excerpt)] for excerpt in train_excerpt_list]
	# feat_type_token_ratio_train = [[get_type_token_ratio(excerpt)]        for excerpt in train_excerpt_list]

	# ptb_google_mapping = dict()
	# file = open("./en-ptb.map")
	# for line in file.readlines():
	# 	ptb, uni = line.split("\t")
	# 	uni = uni[:-1]
	# 	ptb_google_mapping[ptb] = uni
	# universal_tag_list = sorted(list(set(ptb_google_mapping.values())))

	# pos_tag_list = extract_pos_tags(train_xml_path)
	# entity_list = extract_ner_tags(train_xml_path)
	# dependency_list = extract_dependencies(train_xml_path)
	# rules_list = extract_prod_rules(train_xml_path)
	# cluster_code_list = generate_cluster_codes(brown_file_path)
	# word_cluster_mapping = generate_word_cluster_mapping(brown_file_path)

	# pos_feat_train = createPOSFeat(train_xml_path, pos_tag_list)
	# uni_feat_train = createUniversalPOSFeat(pos_feat_train, pos_tag_list, ptb_google_mapping, universal_tag_list)
	# ner_feat_train = createNERFeat(train_xml_path, entity_list)
	# dep_feat_train = createDependencyFeat(train_xml_path, dependency_list)
	# syn_feat_train = createSyntaticProductionFeat(train_xml_path, rules_list)
	# clu_feat_train = createBrownClusterFeat(train_xml_path, cluster_code_list, word_cluster_mapping)

	npy_folder = "./npy_files/"
	## serilize
	# np.save(npy_folder + "pos_train", pos_feat_train)
	# np.save(npy_folder + "pos_test", pos_feat_test)
	# np.save(npy_folder + "uni_train", uni_feat_train)
	# np.save(npy_folder + "uni_test", uni_feat_test)
	# np.save(npy_folder + "ner_train", ner_feat_train)
	# np.save(npy_folder + "ner_test", ner_feat_test)
	# np.save(npy_folder + "dep_train", dep_feat_train)
	# np.save(npy_folder + "dep_test", dep_feat_test)
	# np.save(npy_folder + "syn_train", syn_feat_train)
	# np.save(npy_folder + "syn_test", syn_feat_test)
	# np.save(npy_folder + "clu_train", clu_feat_train)
	# np.save(npy_folder + "clu_test", clu_feat_test)

	## deserialize
	# pos_feat_train = np.load(npy_folder + "pos_train.npy")
	# uni_feat_train = np.load(npy_folder + "uni_train.npy")
	# ner_feat_train = np.load(npy_folder + "ner_train.npy")
	# dep_feat_train = np.load(npy_folder + "dep_train.npy")
	# syn_feat_train = np.load(npy_folder + "syn_train.npy")
	# clu_feat_train = np.load(npy_folder + "clu_train.npy")

	############## unigram
	train_path = "./train_split/"
	test_path = "./test_split/"

	# top_k_unigrams_list = extract_top_k_unigrams(train_path, 2000)
	# unigram_feat_train = createUnigramFeat(train_path, top_k_unigrams_list)
	# unigram_feat_test = createUnigramFeat(test_path, top_k_unigrams_list)
	# np.save("./npy_files/unigram_train", unigram_feat_train)
	# np.save("./npy_files/unigram_test", unigram_feat_test)

	# top_k_bigrams_list = extract_top_k_bigrams(train_path, 4500)
	# bigram_feat_train = createBigramFeat(train_path, top_k_bigrams_list)
	# bigram_feat_test = createBigramFeat(test_path, top_k_bigrams_list)
	# np.save("./npy_files/bigram_train", bigram_feat_train)
	# np.save("./npy_files/bigram_test", bigram_feat_test)


	############## tf-idf
	train_file_path = "./data/project_train.txt"
	#tfidf_feat_train = create_tfidf_feat(train_excerpt_list)
	mi_feat_train = create_mi_feat(train_excerpt_list)
	############## Cross valid
	feat_train = mi_feat_train
	# feat_train = np.concatenate((syn_feat_train, clu_feat_train), axis = 1)


	# reg = SVR(C = 5, kernel = 'linear')
	# reg = LogisticRegression(penalty = 'l1', C = 1, max_iter = 300, solver = 'liblinear', multi_class = 'ovr') 

	# reg = BayesianRidge()
	# reg = BayesianRidge(normalize = True)
	reg = SVR(C = 10, kernel = 'rbf')

	scores = []
	total_num = 461
	train_num = 231
	for _ in range(40):
		for _ in range(5):
			train_idx = sample(range(total_num), train_num)
			X_train = [feat_train[i] for i in range(total_num) if i in train_idx]
			y_train = [train_score_list[i] for i in range(total_num) if i in train_idx]
			X_valid = [feat_train[i] for i in range(total_num) if i not in train_idx]
			y_valid = [train_score_list[i] for i in range(total_num) if i not in train_idx]
			reg.fit(X_train, y_train)
			estimate = reg.predict(X_valid)
			scores.append(compute_spearman_correlation(y_valid, estimate))
			reg.fit(X_valid, y_valid)
			estimate = reg.predict(X_train)
			scores.append(compute_spearman_correlation(y_train, estimate))

	print "Mean: ", np.mean(scores)
	# print "Max:  ", max(scores)
	# print "Min:  ", min(scores)
	# print scores




