#!/usr/bin/env python
import optparse
import sys
import bleu_smooth
from nltk.classify.megam import call_megam, parse_megam_weights,config_megam
optparser = optparse.OptionParser()
optparser.add_option("-k", "--kbest-list", dest="train", default="data/train.100best", help="100-best translation lists")
optparser.add_option("-d", "--dev_kbest-list", dest="dev", default="data/dev+test.100best", help="100-best translation lists")
optparser.add_option("-r", "--reference", dest="reference", default="data/train.ref", help="Target language reference sentences")
(opts, _) = optparser.parse_args()
lm = tm1 = -0.92
tm2 = -1
megam_features = []
samples = []
sent_features = []
sign = lambda x: (1, -1)[x<0]
config_megam("/usr/local/bin/")
#Read Reference Translation for Training
ref = [line.strip().split() for line in open(opts.reference)]
#Read Candidate Translations for Training
all_hyps = [pair.split(' ||| ') for pair in open(opts.train)]
num_sents = len(all_hyps) / 100
bleu_score_per_sent = []
for s in xrange(0, num_sents):
  del bleu_score_per_sent[:]
  del sent_features[:]
  del samples[:]
  empty=0
  hyps_for_one_sent = all_hyps[s * 100:s * 100 + 100]
  #compute BLEU+1 for label and read/compute feature values
  for (num, hyp, feats) in hyps_for_one_sent:
      untranslated=0
      temp_feat = []
      h = hyp.strip().split()
      stats = [0 for i in xrange(10)]
      stats = [sum(scores) for scores in zip(stats, bleu_smooth.bleu_stats(h,ref[s]))]
      x=bleu_smooth.bleu(stats)
      bleu_score_per_sent.append(x)
      for feat in feats.split(' '):
        (k, v) = feat.split('=')
        temp_feat.append(float(v))
      temp_feat.append(len(h))
      for word in h:
        try:
		   word.decode('ascii')
        except UnicodeDecodeError:
           untranslated+=1
      temp_feat.append(untranslated)
      sent_features.append(temp_feat)
  #Create pairs
  for elem1 in range(0,100):
      for elem2 in range(elem1+1,100):
          diff = abs(bleu_score_per_sent[elem1]-bleu_score_per_sent[elem2])
          #For removing threshold
          #samples.append((diff, elem1,elem2))
          ##Standard PRO
          if(diff>=0.05):
              samples.append((diff, elem1,elem2))
          else:
              empty+=1
  #Reduce Threshold if fewer pairs are obtained
  if(len(samples)<50):
      diff_lim=0.05
      while(len(samples)<50):
          del samples[:]
          empty=0
          diff_lim =diff_lim/5.0;
          for elem1 in range(0,100):
              for elem2 in range(elem1+1,100):
                  diff = abs(bleu_score_per_sent[elem1]-bleu_score_per_sent[elem2])
                  if(diff>=diff_lim):
                      samples.append((diff, elem1,elem2))
                  else:
                      empty+=1
  #Sort is decreasing order of BLEU+1
  samples.sort(key = lambda x:x[0],reverse=True)
  #For Random Pair Selection
  '''rands=[]
  for i in range(50):
      rands.append(random.randint(0,len(samples)-1))
  for pair_cnt in rands:'''
  #For Standard PRO
  for pair_cnt in range(50):
      #Read top 50 pairs
      temp =[sent_features[samples[pair_cnt][1]][ind] - sent_features[samples[pair_cnt][2]][ind] for \
                              ind in range(len(sent_features[samples[pair_cnt][1]]))]
      temp_sign= sign(bleu_score_per_sent[samples[pair_cnt][1]]-bleu_score_per_sent[samples[pair_cnt][2]])
      #Store a feature vector for both + and -
      megam_features.append((temp,temp_sign))
      megam_features.append(([-1*x for x in temp],-1*temp_sign))
  #Print Progress
  if(len(megam_features)%5000 ==0):
    sys.stderr.write("%d " % len(megam_features))
#Write features to file for input to megam
train_file = open("train_feat_vectors","w")
for f_vec in megam_features:
    train_file.write(str(f_vec[1])+"\t"+"0 "+str(f_vec[0][0])+" 1 "+str(f_vec[0][1])+" 2 "+str(f_vec[0][2])+" 3 "+str(f_vec[0][3])+" 4 "+str(f_vec[0][4])+"\n")
train_file.close()
#Train using Megam
st=call_megam(["-fvals","-nobias","-quiet","binary","train_feat_vectors"])
wts = parse_megam_weights(st,5)
sys.stderr.write("\n%s\n " % str(wts))
#Store new weights in dictionary
weights = {'p(e)'       : float(wts[0]) ,
           'p(e|f)'     : float(wts[1]),
           'p_lex(f|e)' : float(wts[2]),
           'len'        : float(wts[3]),
           'untr'       : float(wts[4])}
#Write output sentences to file
sent_file = open("output","w")
#From rerank- use new weights to predict test
all_hyps = [pair.split(' ||| ') for pair in open(opts.dev)]
num_sents = len(all_hyps) / 100
for s in xrange(0, num_sents):
  hyps_for_one_sent = all_hyps[s * 100:s * 100 + 100]
  (best_score, best) = (-1e300, '')
  for (num, hyp, feats) in hyps_for_one_sent:
    score = 0.0
    untranslated=0.0
    for feat in feats.split(' '):
      (k, v) = feat.split('=')
      score += weights[k] * float(v)
    score+= weights['len']*float(len(hyp.strip().split()))
    for word in hyp.strip().split():
        try:
		   word.decode('ascii')
        except UnicodeDecodeError:
           untranslated+=1
    score+=weights['untr'] *float(untranslated)
    if score > best_score:
      (best_score, best) = (score, hyp)
  try:
    #sys.stdout.write("%s\n" % best)
    sent_file.write("%s\n" % best)
  except (Exception):
    sys.exit(1)
sent_file.close()


