# This script is used to generate data for five fold cross validation
import os
import argparse
import numpy as np
from itertools import combinations

def readfile(path):
	arr = []
	with open(path, 'r') as r:
		lines = r.readlines()
		for i, line in enumerate(lines):
			if i == 0:
				continue
			arr.append(line)
	return arr

def to_tsv(data, filename):
    import os
    temp = "{}\t{}\t{}\n"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(temp.format("score", "sent1", "sent2"))
        for each in data:
            f.write(each)

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

	nd = readfile(args.data_dir)
	nd.extend([nd[0], nd[800], nd[1600]])
	nd = np.array(nd)
	parts = np.split(nd, 5)
	
	comb = [0,1,2,3,4]
	for each in combinations(comb, 4):
		b = list(set(comb) - set(each))[0]
		m = []
		for k in each:
			m.extend(parts[k])
		
		outdir = os.path.join(args.output_dir, f"sample{b}")
		if not os.path.isdir(outdir):
			os.makedirs(os.path.dirname(outdir), exist_ok=True)
		to_tsv(m, os.path.join(outdir, 'train.tsv'))
		to_tsv(parts[b], os.path.join(outdir, 'dev.tsv'))

if __name__ == '__main__':
	main()