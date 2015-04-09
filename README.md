# Reranking
Implements Reranking techniques for Machine Translation
pro.py: implements PRO. It uses bleu_smoooth and files from data/ folder. It outputs a file "output" which can be given as input to compute-bleu to obtain accuracy score.
rerank_5feat.py: is a modified rerank- with all 5 features for evaluation. They can be given as command line argument. 
The options are: -l -1 -t -0.875 -s -0.58 -n 1.042 -u -6.6 to get the best performance
compute-bleu: computes accuracy on outputs sentences from rerank and pro.
output: results from pro.py. Use as: ./compute-bleu < output
5features_manual: results from rerank_5feat.py with given weights. Use as the file output.
bleu_smooth and bleu: compute BLEU+1 and BLEU scores respectively. Used by pro.py and compute-bleu respectively.

Dependencies: Expects megam classifier. Searches in /usr/local/bin/

