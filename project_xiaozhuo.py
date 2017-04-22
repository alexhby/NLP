import nltk
import numpy as np
from nltk import word_tokenize, sent_tokenize, FreqDist
from scipy import stats
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

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
        if count_dict[word] > 5:
            freq_count += 1
    frac_freq = float(freq_count) / vocab_size
    return frac_freq


def get_frac_rare(input_string):
    token_list = tokenize_excerpt(input_string)
    count_dict = FreqDist(token_list)
    vocab_size = len(count_dict)
    rare_count = 0
    for word in count_dict:
        if count_dict[word] == 1:
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




if __name__ == "__main__":
    train_excerpt_list = load_excerpts("project_train.txt")
    train_score_list = load_train_score("project_train_scores.txt")
    optional_file_list, optional_score_list = load_optional_score("project_optional_scores.txt")
    optional_excerpt_list = load_optional_excerpts("./optional_training/", optional_file_list)
    test_excerpt_list = load_excerpts("project_test.txt")
    
    feat_vocab_size_train    = [[get_vocab_size(excerpt)] for excerpt in train_excerpt_list]
    feat_vocab_size_optional = [[get_vocab_size(excerpt)] for excerpt in optional_excerpt_list]


    # clf = SVC(C = 5, kernel = 'linear')
    # clf = LogisticRegression(penalty = 'l1', C = 1, max_iter = 300, solver = 'liblinear', multi_class = 'ovr') 

    reg = BayesianRidge()
    reg.fit(feat_vocab_size_optional, optional_score_list)
    estimate = reg.predict(feat_vocab_size_train)

    print compute_spearman_correlation(train_score_list, estimate)




