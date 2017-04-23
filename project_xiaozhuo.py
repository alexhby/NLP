import nltk
import os
import numpy as np
import xml.etree.ElementTree as ET
from collections import Counter
from nltk import word_tokenize, sent_tokenize, FreqDist
from operator import itemgetter 
from os import listdir, system
from scipy import stats
from sklearn import linear_model
from sklearn.linear_model import BayesianRidge, Lasso, LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC, SVR
from random import sample

####################### PART 1 - PREPARATION #######################


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



####################### PART 2 - EVALUATION FUNCTION #######################
def compute_spearman_correlation(ground_truth, predictions):
    return stats.spearmanr(ground_truth, predictions).correlation



####################### PART 3 - CLASSICAL FEATURES #######################
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



####################### PART 4 - SYNTACTIC FEATURES #######################
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



###########################################################################

if __name__ == "__main__":
	train_xml_path = "./train_xml/"
	test_xml_path  = "./test_xml/"

	train_excerpt_list = load_excerpts("project_train.txt")
	train_score_list = load_train_score("project_train_scores.txt")
	optional_file_list, optional_score_list = load_optional_score("project_optional_scores.txt")
	optional_excerpt_list = load_optional_excerpts("./optional_training/", optional_file_list)
	test_excerpt_list = load_excerpts("project_test.txt")

	# feat_vocab_size_train       = [[get_vocab_size(excerpt)]              for excerpt in train_excerpt_list]
	# feat_frac_freq_train        = [[get_frac_freq(excerpt)]               for excerpt in train_excerpt_list]
	# feat_frac_rare_train        = [[get_frac_rare(excerpt)]               for excerpt in train_excerpt_list]
	# feat_median_length_train    = [[get_median_length(excerpt)]           for excerpt in train_excerpt_list]
	# feat_average_length_train   = [[get_average_length(excerpt)]          for excerpt in train_excerpt_list]
	# feat_average_sentence_train = [[get_average_sentence_length(excerpt)] for excerpt in train_excerpt_list]
	# feat_type_token_ratio_train = [[get_type_token_ratio(excerpt)]        for excerpt in train_excerpt_list]

	ptb_google_mapping = dict()
	file = open("./en-ptb.map")
	for line in file.readlines():
		ptb, uni = line.split("\t")
		uni = uni[:-1]
		ptb_google_mapping[ptb] = uni
	universal_tag_list = sorted(list(set(ptb_google_mapping.values())))

	pos_tag_list = extract_pos_tags(train_xml_path)
	pos_feat_train = createPOSFeat(train_xml_path, pos_tag_list)
	uni_feat_train = createUniversalPOSFeat(pos_feat_train, pos_tag_list, ptb_google_mapping, universal_tag_list)

	feat_train = uni_feat_train

	# reg = SVR(C = 5, kernel = 'linear')
	# reg = LogisticRegression(penalty = 'l1', C = 1, max_iter = 300, solver = 'liblinear', multi_class = 'ovr') 
	reg = BayesianRidge()

	scores = []
	total_num = 461
	train_num = 231
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
	print "Max:  ", max(scores)
	print "Min:  ", min(scores)
	print scores




