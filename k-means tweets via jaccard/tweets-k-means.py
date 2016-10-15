import sys, string, nltk
import pandas as pd
import numpy as np
from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from itertools import chain
from sklearn.preprocessing import LabelEncoder
from kmeans.JaccardKMeans import JaccardKMeans
#nltk.download("stopwords")	### stopwords corpus here; download if code doesn't run

def main(argv):
	
	
	if len(argv) is not 4:
		print("Expecting four arguments: <cluster #> <seeds_file> <tweets_json> <output_file>")
		sys.exit()
	
	k = argv[0]
	seeds = argv[1]
	tweets = argv[2]
	output = argv[3]
	
	'''
	k = 5
	seeds = 'InitialSeeds.txt'
	tweets = 'Tweets.json'
	'''
	
	
	# get some seeds
	with open(seeds) as s:
		seeds = s.readlines()
	seeds = [seed.replace(',\n', '') for seed in seeds]
	seeds = pd.DataFrame(seeds, columns = ['id'])
	seeds['id'] = seeds['id'].astype(np.int64)
	seeds = seeds.sample(int(k))
	
	# load data
	with open(tweets) as f:
		lines = f.readlines()
	df = pd.read_json(lines[0])
	for i in range(1, len(lines)):
		new_line = pd.read_json(lines[i], dtype = {'id': 'category'})
		df = df.append(new_line)
		
	# keep only id and tweets		################### CHECK DROPPED COLUMNS HERE
	for col in df.columns:						####### YOU LOST THE CORRECT IDs SOMEWHERE
		if col != 'id' and col != 'text':
			df = df.drop(col, 1)
	
	# clean a little and tokenize
	tweet_tokenizer = TweetTokenizer(reduce_len = True, preserve_case = False, strip_handles = True)
	tweets = df['text']
	tweet_tokens = [tweet_tokenizer.tokenize(tweet) for tweet in tweets]
	new_tweets = [" ".join(tokens) for tokens in tweet_tokens]
	
	# clean more
	tweet_tokenizer = TweetTokenizer(reduce_len = True)
	punctuations = list(string.punctuation)
	punctuations.append('rt')
	tweet_tokens = [tweet_tokenizer.tokenize(tweet) for tweet in new_tweets]
	tweet_tokens = [[token for token in tokens if token not in punctuations] for tokens in tweet_tokens]
	tweet_tokens = [[token for token in tokens if token.find('http') == -1] for tokens in tweet_tokens]
	tweet_tokens = [[token for token in tokens if token.find('#') == -1] for tokens in tweet_tokens]
	tweet_tokens = [[token for token in tokens if token.find('...') == -1] for tokens in tweet_tokens]
	tweet_tokens = [[token for token in tokens if token.find('@') == -1] for tokens in tweet_tokens]
	
	# more cleaning
	for digit in string.digits:
		tweet_tokens = [[token for token in tokens if token.find(digit) == -1] 
						 for tokens in tweet_tokens]
	for single_char in string.ascii_lowercase.replace('i', ''):
		tweet_tokens = [[token for token in tokens if token != single_char]
						 for tokens in tweet_tokens]
	tweet_tokens = [[token for token in tokens if len(token) != 1 or token == 'i']
					 for tokens in tweet_tokens]
					 
	# stemming and stop words removal
	stops = list(stopwords.words('english'))
	tweet_tokens = [[token for token in tokens if token not in stops] 
					 for tokens in tweet_tokens]
	stemmer = SnowballStemmer("english")
	tweet_tokens = [[stemmer.stem(token) for token in tokens]
					 for tokens in tweet_tokens]

	# encode to numeric labels
	just_tokens = list(chain.from_iterable(tweet_tokens))
	encoder = LabelEncoder()
	encoder.fit(just_tokens)
	tweet_labels = [encoder.transform(tweet) for tweet in tweet_tokens]
	
	# now we have sets for computing Jaccard
	tweet_sets = [set(tweet) for tweet in tweet_labels]
	
	# prepare for kmeans algorithm:
	df = df.drop('text', 1)
	df['tweets'] = tweet_sets
	
	# kmeans magic
	km = JaccardKMeans(df, seeds)
	km.output_clusters(output)
	
	
if __name__ == "__main__":
	main(sys.argv[1:])
	
	
	
	
	
	
	
