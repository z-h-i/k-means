import pandas as pd
import numpy as np
from numpy.linalg import norm
from functools import reduce

class KMeans:
	
	# intialize centroids, ids, and other etc. relevant information
	def __init__(self, df, k):
		self.k = k
		self.df = df
		self.centroid_info = df.sample(k)
		self.centroids = self.centroid_info[['x', 'y']].as_matrix()
		self.points = df[['x', 'y']].as_matrix()
		self.centroid_ids = list(range(1, k + 1))
		
	# one iteration of cluster assignment and relocation
	def cluster_search(self):
		
		centroid_assignments = []
		for point in self.points:
			
			distances = []
			for centroid in self.centroids:
				distances.append(norm(centroid - point))
			
			centroid_assignment = [i + 1 for i, val in enumerate(distances) if val == min(distances)]
			centroid_assignments.append(centroid_assignment[0])
		
		self.df['centroids'] = centroid_assignments
		
		for centroid_id in self.centroid_ids:
			new_location = self.df[self.df['centroids'] == centroid_id][['x', 'y']].mean().as_matrix()
			self.centroids[centroid_id - 1] = new_location

	# compute SSE
	def get_SSE(self):

		points_per_centroid = [self.df[self.df['centroids'] == centroid_id][['x', 'y']].as_matrix() 
							   for centroid_id in self.centroid_ids]
		SSE = sum([norm(point - self.centroids[i]) ** 2 for i, points in enumerate(points_per_centroid)
				   for point in points])

		return SSE

	# output into file
	def output_clusters(self, output_file):
		
		for i in range(25):
			self.cluster_search()
			
		SSE = self.get_SSE()
		new_df = pd.DataFrame(self.centroid_ids)
		points_ids = [reduce(lambda x, y: str(x) + ', ' + str(y),
					  self.df[self.df['centroids'] == centroid_id].id) for centroid_id in
					  self.centroid_ids]
		new_df['points_ids'] = points_ids
		new_df = new_df.rename(columns = {0: 'cluster_id'})
		new_df.to_csv(output_file, sep = '\t', index = False)
		
		with open(output_file, 'a') as f:
			f.write('SSE = ' + str(SSE))

