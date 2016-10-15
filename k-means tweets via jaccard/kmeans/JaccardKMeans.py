import pandas as pd
import numpy as np
from functools import reduce

class JaccardKMeans:
	
	# intialize centroids, ids, and other etc. relevant information
	def __init__(self, df, seeds):
		self.k = len(seeds)
		self.df = df
		self.centroid_info = seeds.merge(df, how = 'left', on = 'id')
		self.centroids = self.centroid_info['tweets']
		self.points = df['tweets']
		self.centroid_ids = list(range(1, self.k + 1))
		self.old_centroids = self.centroids
		self.new_centroids = None
		
	# comput Jaccard distance
	def jaccard(self, A, B):
		AuB = len(A.union(B))
		AnB = len(A.intersection(B))
		distance = 1 - (float(AnB) / AuB)
		return distance
		
	# one iteration of cluster assignment and relocation
	def cluster_search(self):
		
		self.old_centroids = self.centroids
		centroid_assignments = []
		for point in self.points:
			
			distances = []
			for centroid in self.centroids:
				distances.append(self.jaccard(point, centroid))
			
			centroid_assignment = [i + 1 for i, val in enumerate(distances) if val == min(distances)]
			centroid_assignments.append(centroid_assignment[0])
		
		self.df['centroids'] = centroid_assignments
		
		for centroid_id in self.centroid_ids:
			
			centroid_points = self.df[self.df['centroids'] == centroid_id].tweets
			min_mean_distance = 2
			for point_A in centroid_points:
				
				if len(centroid_points) == 1:
					self.centroids.iloc[centroid_id - 1] = point_A
					
				else:
					total_distance = sum([self.jaccard(point_A, point_B) for point_B in centroid_points])
					mean_distance = float(total_distance) / (len(centroid_points) - 1)
					
					if mean_distance < min_mean_distance:
						min_mean_distance = mean_distance
						self.centroids.iloc[centroid_id - 1] = point_A
						
		self.new_centroids = self.centroids

	# compute SSE
	def get_SSE(self):

		points_per_centroid = [self.df[self.df['centroids'] == centroid_id].tweets
							   for centroid_id in self.centroid_ids]
		SSE = 0
		for centroid_id in self.centroid_ids:
			centroid_points = self.df[self.df['centroids'] == centroid_id].tweets
			SSE = SSE + sum([self.jaccard(self.centroids.iloc[centroid_id - 1], point) ** 2
							 for point in centroid_points])
		
		return SSE

	# output into file
	def output_clusters(self, output_file):
		
		while not self.old_centroids.equals(self.new_centroids):
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

