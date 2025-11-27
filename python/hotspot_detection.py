from sklearn.cluster import DBSCAN
import numpy as np

coords = acc[['Latitude','Longitude']].to_numpy()
kms_per_radian = 6371.0088
epsilon = 0.5 / kms_per_radian  # 0.5 km
db = DBSCAN(eps=epsilon, min_samples=5, algorithm='ball_tree', metric='haversine')
labels = db.fit_predict(np.radians(coords))
acc['cluster'] = labels
hotspots = acc[acc['cluster'] != -1].groupby('cluster').agg({'Latitude':'mean','Longitude':'mean','AccidentID':'count'})
