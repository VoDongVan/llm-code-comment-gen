import sys
import os
import json
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import nltk
from nltk.translate import meteor
from nltk import word_tokenize
import evaluate
from evaluate import load
from sentence_transformers import SentenceTransformer, SimilarityFunction
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from numpy import dot
from numpy.linalg import norm

nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('omw-1.4')

parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)

bertscore = load("bertscore")
prompt_methods = ['zero-shot', 'few-shot-project', 'few-shot-bm25', 'few-shot-codeBERT', 'cot', 'critique', 'expert']

def compute_bleu(res, data, method='zero-shot'):
    candidate_list = []
    reference_list = []
    project_scores = []
    chencherry = SmoothingFunction()
    for i in range(0, len(res)):
        candidate_list_project = []
        reference_list_project = []
        for j in range(0, len(res[i]['functions_res'])):
            candidate = word_tokenize(res[i]['functions_res'][j][method])#.split()
            reference = word_tokenize(data[i]['functions'][j]['docstring'])#.split()
            candidate_list.append(candidate)
            reference_list.append([reference])
            candidate_list_project.append(candidate)
            reference_list_project.append([reference])
        project_scores.append(corpus_bleu(reference_list_project, candidate_list_project, smoothing_function=chencherry.method0))
    bleu_score = corpus_bleu(reference_list, candidate_list, smoothing_function=chencherry.method0)
    bleu_score_per_project = sum(project_scores) / len(res)
    print(f"BLEU Score ({method}):", bleu_score, ", average by project:", bleu_score_per_project)
    return bleu_score, bleu_score_per_project

def compute_meteor(res, data, method="zero-shot"):
    scores = []
    project_scores = []
    for i in range(0, len(res)):
        cur_scores = []
        for j in range(0, len(res[i]['functions_res'])):
            candidate = word_tokenize(res[i]['functions_res'][j][method])#.split()
            reference = word_tokenize(data[i]['functions'][j]['docstring'])#.split()
            score = round(meteor([reference], candidate), 4)
            scores.append(score)
            cur_scores.append(score)
        cur_scores = sum(cur_scores) / len(cur_scores)
        project_scores.append(cur_scores)
    meteor_score = sum(scores) / len(scores)
    meteor_score_per_project = sum(project_scores) / len(res)
    print(f"METEOR Score ({method}):", meteor_score, ", average by project:", meteor_score_per_project)
    return meteor_score, meteor_score_per_project

def compute_rougel(res, data, method="zero-shot"):
    #CODE BASED ON RENCOS: https://github.com/zhangj111/rencos/blob/master/evaluation/rouge/rouge.py
    def my_lcs(string, sub):
        """
        Calculates longest common subsequence for a pair of tokenized strings
        :param string : list of str : tokens from a string split using whitespace
        :param sub : list of str : shorter string, also split using whitespace
        :returns: length (list of int): length of the longest common subsequence between the two strings

        Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
        """
        if(len(string)< len(sub)):
            sub, string = string, sub

        lengths = [[0 for i in range(0,len(sub)+1)] for j in range(0,len(string)+1)]

        for j in range(1,len(sub)+1):
            for i in range(1,len(string)+1):
                if(string[i-1] == sub[j-1]):
                    lengths[i][j] = lengths[i-1][j-1] + 1
                else:
                    lengths[i][j] = max(lengths[i-1][j] , lengths[i][j-1])

        return lengths[len(string)][len(sub)]

    def calc_score(candidate, refs, beta=1.2):
        """
        Compute ROUGE-L score given one candidate and references for an image
        :param candidate: str : candidate sentence to be evaluated
        :param refs: list of str : COCO reference sentences for the particular image to be evaluated
        :returns score: int (ROUGE-L score for the candidate evaluated against references)
        """
        assert(len(candidate)==1)
        assert(len(refs)>0)         
        prec = []
        rec = []

        # split into tokens
        token_c = candidate[0].split(" ")

        for reference in refs:
            # split into tokens
            token_r = reference.split(" ")
            # compute the longest common subsequence
            lcs = my_lcs(token_r, token_c)
            prec.append(lcs/float(len(token_c)))
            rec.append(lcs/float(len(token_r)))

        prec_max = max(prec)
        rec_max = max(rec)

        if(prec_max!=0 and rec_max !=0):
            score = ((1 + beta**2)*prec_max*rec_max)/float(rec_max + beta**2*prec_max)
        else:
            score = 0.0
        return score
    
    scores = []
    project_scores = []
    for i in range(0, len(res)):
        cur_scores = []
        for j in range(0, len(res[i]['functions_res'])):
            candidate = res[i]['functions_res'][j][method]
            reference = data[i]['functions'][j]['docstring']
            score = round(calc_score([candidate], [reference]), 4)
            scores.append(score)
            cur_scores.append(score)
        cur_scores = sum(cur_scores) / len(cur_scores)
        project_scores.append(cur_scores)
    rougel_score = sum(scores) / len(scores)
    rougel_score_per_project = sum(project_scores) / len(project_scores)
    print(f"ROUGE-L Score ({method}):", rougel_score, ", average by project:", rougel_score_per_project)
    return rougel_score, rougel_score_per_project

def compute_bertscore(res, data, method="zero-shot"):
    candidate_list = []
    reference_list = []
    project_scores = []

    #GLOBAL
    for i in range(0, len(res)):
        for j in range(0, len(res[i]['functions_res'])):
            candidate = res[i]['functions_res'][j][method]
            reference = data[i]['functions'][j]['docstring']
            candidate_list.append(candidate)
            reference_list.append(reference)
    bert_score = bertscore.compute(predictions=candidate_list, references=reference_list, lang="en")
    f1 = sum(bert_score['f1']) / len(bert_score['f1'])
    # PER PROJECT
    k = 0
    for i in range(0, len(res)):
        score = []
        for j in range(0, len(res[i]['functions_res'])):
            score.append(bert_score['f1'][k])
            k += 1
        score = sum(score) / len(score)
        project_scores.append(score)
    bert_score_per_project = sum(project_scores) / len(project_scores)
    
    print(f"BERT Score F1 ({method}):", f1, ", average by project:", bert_score_per_project)
    return bert_score

def compute_sentencebert(res, data, method="zero-shot"):
    candidate_list = []
    reference_list = []
    project_scores_cos = []
    project_scores_euc = []
    for i in range(0, len(res)):
        for j in range(0, len(res[i]['functions_res'])):
            candidate = res[i]['functions_res'][j][method]
            reference = data[i]['functions'][j]['docstring']
            candidate_list.append(candidate)
            reference_list.append(reference)
    #COSINE SIMILARITY
    model = SentenceTransformer("all-mpnet-base-v2", similarity_fn_name=SimilarityFunction.COSINE)
    candidate_embeddings = model.encode(candidate_list)
    reference_embeddings = model.encode(reference_list)
    # PER PROJECT COSINE SIMILARITY
    k = 0
    for i in range(0, len(res)):
        candidate_embeddings_project = []
        reference_embeddings_project = []
        for j in range(0, len(res[i]['functions_res'])):
            candidate_embeddings_project.append(candidate_embeddings[k])
            reference_embeddings_project.append(reference_embeddings[k])
            k += 1
        cos_sim_pairwise = model.similarity(candidate_embeddings_project, reference_embeddings_project)
        cos_sim = 0
        for i in range(0, len(res[i]['functions_res'])):
            cos_sim += cos_sim_pairwise[i, i]
        cos_sim /= len(candidate_embeddings_project)
        project_scores_cos.append(cos_sim)
    # GLOBAL COSINE SIMILARITY
    cos_sim_pairwise = model.similarity(candidate_embeddings, reference_embeddings)
    cos_sim = 0

    # EUCLIDEAN
    model = SentenceTransformer("all-mpnet-base-v2", similarity_fn_name=SimilarityFunction.EUCLIDEAN)
    # PER PROJECT EUCLIDEAN SIMILARITY
    k = 0
    for i in range(0, len(res)):
        candidate_embeddings_project = []
        reference_embeddings_project = []
        for j in range(0, len(res[i]['functions_res'])):
            candidate_embeddings_project.append(candidate_embeddings[k])
            reference_embeddings_project.append(reference_embeddings[k])
            k += 1
        euclidean_sim_pairwise = model.similarity(candidate_embeddings_project, reference_embeddings_project)
        euclidean_sim = 0
        for i in range(0, len(res[i]['functions_res'])):
            euclidean_sim += euclidean_sim_pairwise[i, i]
        euclidean_sim /= len(candidate_embeddings_project)
        project_scores_euc.append(euclidean_sim)

    # GLOBAL EUCLIDEAN SIMILARITY
    euclidean_sim_pairwise = model.similarity(candidate_embeddings, reference_embeddings)
    euclidean_sim = 0
    for i in range(0, len(candidate_list)):
        euclidean_sim += euclidean_sim_pairwise[i,i]
        cos_sim += cos_sim_pairwise[i,i]
    euclidean_sim /= len(candidate_list)
    cos_sim /= len(candidate_list)
    euclidean_sim_per_project = sum(project_scores_euc) / len(project_scores_euc)
    cos_sim_per_project = sum(project_scores_cos) / len(project_scores_cos)
    print(f"SentenceBert euclidean similarity ({method}):", euclidean_sim.item(), ", average by project:", euclidean_sim_per_project.item())
    print(f"SentenceBert cosine similarity ({method}):", cos_sim.item(), ", average by project:", cos_sim_per_project.item())
    return euclidean_sim, cos_sim, euclidean_sim_per_project, cos_sim_per_project 

def compute_USE(res, data, method='zero-shot'):
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(module_url)
    def embed(input):
      return model(input)
    compute_cos_sim = lambda a, b: dot(a, b)/(norm(a)*norm(b))

    candidate_list = []
    reference_list = []
    project_scores = []
    for i in range(0, len(res)):
        for j in range(0, len(res[i]['functions_res'])):
            candidate = res[i]['functions_res'][j][method]
            reference = data[i]['functions'][j]['docstring']
            candidate_list.append(candidate)
            reference_list.append(reference)
    candidate_embedding = np.array(embed(candidate_list)).tolist()
    reference_embedding = np.array(embed(reference_list)).tolist()

    k = 0
    for i in range(0, len(res)):
        cur_score = 0
        for j in range(0, len(res[i]['functions_res'])):
            candidate = candidate_embedding[k]
            reference = reference_embedding[k]
            cur_score += compute_cos_sim(candidate, reference)
            k += 1
        cur_score /= len(res[i]['functions_res'])
        project_scores.append(cur_score)
    use_score_per_project = sum(project_scores) / len(res)

    use_score = 0
    for i in range(0, len(candidate_list)):
        candidate = candidate_embedding[i]
        reference = reference_embedding[i]
        use_score += compute_cos_sim(candidate, reference)
    use_score /= len(candidate_list)
    print(f"Universal Sentence Encoder Cosine Similarity: ({method}):", use_score, ", average by project:", use_score_per_project)
    return use_score, use_score_per_project