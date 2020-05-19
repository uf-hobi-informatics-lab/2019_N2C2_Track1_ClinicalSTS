import pandas as pd
import argparse
import os
import re
from functools import partial

SYMBOLS = {',', '?', '!', ':', '\'', '"', '(', ')', ';', '@', '^', '^', '&', '&', '$', '$', '£',
           '[', ']', '{', '}', '<', '>', '+', '-', "*", "#", "%", "=", "~", '/', "_"}

def to_tsv(data, filename):
	os.makedirs(os.path.dirname(filename), exist_ok=True)
	temp = '{}\t{}\t{}\n'
	with open(filename, 'w') as file:
		file.write(temp.format('score', 'sent1', 'sent2'))
		for line in data:
			file.write(temp.format(line[0], line[1], line[2]))

def text_normalization(text):
	txt = sent_tokenizer(text).strip().replace("\n", " ")
	# print(txt)
	tokens = txt.lower().split(" ")
	ntokens = []
	for token in tokens:
		ntk = token.split(" ")
		ntokens.extend(ntk)
	return ntokens

def preprocess(d, row):
	sent1 = row[0]
	sent2 = row[1]
	score = float(row[2])
	s1 = text_normalization(sent1)
	s2 = text_normalization(sent2)
	d.append([score, s1, s2])

def split_sentence(text):
	preprocessed_text_list = []
	text = text.lower()
	text = re.sub('\t|_{4,}|-{4,}|\*{6,}|={4,}', ' ', text)
	chs = []
	for ch in text:
		if ch in SYMBOLS:	
			chs.extend([' ', ch, ' '])
		else:
			chs.append(ch)
	line = ''.join(chs)
	line = re.sub('[ ]{2,}', ' ', line).strip()
	if len(line) > 0:
		preprocessed_text_list.append(line)
	
	return preprocessed_text_list

def sent_tokenizer(text):
	token_list = split_sentence(text)
	return ''.join(token_list)
	

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--data_dir",
		default=None,
		type=str,
		required=True
	)
	parser.add_argument(
		"--output_dir",
		default=None,
		type=str,
		required=True
	)

	args = parser.parse_args()
	
	dfile = args.data_dir
	df = pd.read_csv(args.data_dir, sep='\t', header=None, dtype=object)
	d = list()
	df.apply(partial(preprocess, d), axis=1)
	# print(d[:10])
	nd = []
	for each in d:
		score, s1, s2 = each
		# nd.append([score, s1, s2])
		nd.append([score, " ".join(s1).replace("\n", " "), " ".join(s2).replace("\n", " ")])
	
	to_tsv(nd, args.output_dir)


if __name__ == '__main__':
	main()