import sys
import pandas as pd
from kmeans.KMeans import KMeans

def main(argv):
	if len(argv) is not 3:
		print("Please enter k, input, and output file names only")
		sys.exit()
	
	k = int(argv[0])
	data = pd.read_table(argv[1])
	output = argv[2]
	
	km = KMeans(data, k)
	km.output_clusters(output)
	
	
if __name__ == "__main__":
	main(sys.argv[1:])
