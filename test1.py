import numpy as np
from numpy import vstack,array
from scipy.cluster.vq import kmeans,vq
from cluster import KMeansClustering
a = np.array([0.916666666666667, 0.683333333333333, 0.566666666666667, 0.533333333333333, 0.533333333333333, 0.0, 0.0, 0.0, 0.0, 0.0333333333333333, 0.183333333333333, 0.316666666666667, 0.466666666666667, 0.5, 0.616666666666667, 0.65, 0.666666666666667, 0.833333333333333, 0.633333333333333, 0.516666666666667, 0.716666666666667, 0.866666666666667, 0.916666666666667, 1.05, 1.1, 1.05, 1.11666666666667, 1.08333333333333, 1.18333333333333, 0.95, 0.966666666666667, 1.03333333333333, 1.01666666666667, 1.11666666666667, 1.26666666666667, 1.25, 1.15, 1.11666666666667, 1.15, 1.05, 1.2, 1.31666666666667, 1.33333333333333, 1.36666666666667, 1.48333333333333, 1.46666666666667, 1.38333333333333, 1.41666666666667, 1.35, 1.46666666666667, 1.31666666666667, 1.26666666666667, 1.18333333333333, 0.75, 0.65, 0.633333333333333, 0.516666666666667, 0.133333333333333, 0.0166666666666667, 0.0, 0.0, 0.05, 0.0166666666666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0166666666666667, 0.166666666666667, 0.3, 0.35, 0.383333333333333, 0.5, 0.6, 0.716666666666667, 0.866666666666667, 0.85, 0.433333333333333, 0.516666666666667, 0.566666666666667, 0.45, 0.483333333333333, 0.566666666666667, 0.666666666666667, 0.616666666666667, 0.666666666666667, 0.683333333333333, 0.75, 0.616666666666667, 0.733333333333333, 0.75, 0.716666666666667, 0.466666666666667, 0.483333333333333, 0.516666666666667, 0.566666666666667, 0.55, 0.466666666666667, 0.45, 0.466666666666667, 0.483333333333333, 0.6, 0.75, 0.766666666666667, 0.733333333333333, 0.533333333333333, 0.35, 0.166666666666667, 0.2, 0.133333333333333, 0.0, 0.0, 0.0166666666666667, 0.0833333333333333, 0.216666666666667, 0.266666666666667, 0.0, 0.0, 0.0, 0.616666666666667, 0.666666666666667, 0.716666666666667, 0.85, 0.75, 0.85, 1.01666666666667, 1.08333333333333, 1.03333333333333, 1.05, 1.18333333333333, 1.05, 1.05, 0.333333333333333, 0.633333333333333, 0.666666666666667, 0.75, 0.75, 0.716666666666667, 0.75, 0.8, 0.916666666666667, 0.916666666666667, 1.0, 1.0, 1.05, 1.03333333333333, 1.06666666666667, 1.06666666666667, 1.15, 0.9, 0.816666666666667, 0.916666666666667, 0.633333333333333, 0.783333333333333, 0.816666666666667, 0.983333333333333, 1.03333333333333, 1.11666666666667, 1.08333333333333, 1.1, 1.23333333333333, 1.1, 1.05, 1.08333333333333, 1.18333333333333, 1.21666666666667, 1.05, 0.866666666666667, 0.9, 0.816666666666667, 0.833333333333333, 0.65, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.15, 0.383333333333333, 0.55, 0.733333333333333, 0.85, 0.8, 0.683333333333333, 0.85, 0.816666666666667, 1.0, 0.683333333333333, 0.583333333333333, 0.733333333333333, 0.85, 0.166666666666667, 0.866666666666667, 0.85, 0.85, 0.433333333333333, 0.55, 0.616666666666667, 1.53333333333333, 0.583333333333333, 0.75, 0.916666666666667, 1.08333333333333, 1.18333333333333, 1.18333333333333, 1.1, 1.01666666666667, 1.11666666666667, 0.9, 0.916666666666667, 1.01666666666667, 0.966666666666667, 0.966666666666667, 0.95, 0.983333333333333, 1.11666666666667, 0.85, 0.833333333333333, 0.9, 0.866666666666667, 0.616666666666667, 0.6, 0.516666666666667, 0.416666666666667, 0.1, 0.0666666666666667, 0.0833333333333333, 0.1, 0.166666666666667, 0.233333333333333, 0.3, 0.433333333333333, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0666666666666667, 0.233333333333333, 0.383333333333333, 0.45, 0.483333333333333, 0.45, 0.45, 0.55, 0.55, 0.783333333333333, 0.95, 1.05, 1.2, 1.1, 0.966666666666667, 0.75, 0.783333333333333, 0.683333333333333, 0.7, 0.65, 0.25, 0.766666666666667, 0.533333333333333, 0.6, 0.7, 0.866666666666667, 1.03333333333333, 0.816666666666667, 1.0, 1.15, 1.18333333333333, 1.33333333333333, 1.45, 1.61666666666667, 1.78333333333333, 1.91666666666667, 1.96666666666667, 2.11666666666667, 2.15, 2.13333333333333, 2.1, 2.2, 2.36666666666667, 2.43333333333333, 2.5, 2.51666666666667, 2.21666666666667, 2.0, 2.0, 1.83333333333333, 1.56666666666667, 1.26666666666667, 1.15, 1.26666666666667, 1.03333333333333, 0.933333333333333, 0.716666666666667, 0.583333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0666666666666667, 0.333333333333333, 0.233333333333333, 0.333333333333333, 0.233333333333333, 0.366666666666667, 0.516666666666667, 0.583333333333333, 0.55, 0.65, 0.75, 0.866666666666667, 1.03333333333333, 1.06666666666667, 0.916666666666667, 0.5, 0.566666666666667, 0.45, 0.316666666666667, 0.483333333333333, 0.4, 0.566666666666667, 0.733333333333333, 0.866666666666667, 0.933333333333333, 1.06666666666667, 0.983333333333333, 1.03333333333333, 1.01666666666667, 0.916666666666667, 0.833333333333333, 0.966666666666667, 1.11666666666667, 1.23333333333333, 1.21666666666667, 1.11666666666667, 1.0, 0.933333333333333, 1.1, 1.26666666666667, 1.25, 1.35, 1.3, 1.43333333333333, 1.43333333333333, 1.4, 1.55, 1.58333333333333, 1.56666666666667, 1.6, 1.55, 1.4, 1.36666666666667, 1.05, 0.933333333333333, 1.01666666666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0333333333333333, 0.133333333333333, 0.316666666666667, 0.433333333333333, 0.6, 0.766666666666667, 0.966666666666667, 1.13333333333333, 1.25, 1.31666666666667, 1.43333333333333, 1.08333333333333, 0.9, 1.03333333333333, 1.13333333333333, 1.3, 1.31666666666667, 1.36666666666667, 1.03333333333333, 1.15, 1.18333333333333, 1.28333333333333, 1.38333333333333, 1.48333333333333, 1.58333333333333, 1.71666666666667, 1.85, 1.96666666666667, 1.95, 2.0, 1.73333333333333, 1.8, 1.95, 1.85, 1.8, 1.46666666666667, 1.48333333333333, 1.48333333333333, 1.45, 1.51666666666667, 1.51666666666667, 1.5, 1.48333333333333, 1.4, 1.28333333333333, 1.25, 1.31666666666667, 1.18333333333333, 0.9, 0.916666666666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.266666666666667, 0.25, 0.166666666666667, 0.0, 0.0333333333333333, 0.0666666666666667, 0.233333333333333, 0.4, 0.566666666666667, 0.566666666666667, 0.733333333333333, 0.9, 1.06666666666667, 1.16666666666667, 1.21666666666667, 1.23333333333333, 1.41666666666667, 1.5, 1.66666666666667, 1.83333333333333, 1.68333333333333, 1.66666666666667, 1.71666666666667, 1.65, 1.35, 1.38333333333333, 1.4, 1.45, 1.61666666666667, 1.75, 1.95, 2.11666666666667, 2.06666666666667, 2.2, 2.35, 2.3, 2.46666666666667, 2.63333333333333, 2.8, 2.86666666666667, 3.0, 2.95, 2.95, 2.65, 2.76666666666667, 2.76666666666667, 1.96666666666667, 1.66666666666667, 1.6, 0.983333333333333, 0.783333333333333, 0.683333333333333, 0.583333333333333, 0.733333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0666666666666667, 0.216666666666667, 0.35, 0.483333333333333, 0.533333333333333, 0.616666666666667, 0.566666666666667, 0.75, 0.533333333333333, 0.2, 0.366666666666667, 0.533333333333333, 0.633333333333333, 0.8, 0.966666666666667, 1.13333333333333, 1.3, 1.46666666666667, 1.48333333333333, 1.31666666666667, 0.983333333333333, 0.866666666666667, 0.966666666666667, 0.733333333333333, 0.9, 1.06666666666667, 1.13333333333333, 1.21666666666667, 1.33333333333333, 1.4, 1.56666666666667, 1.71666666666667, 1.86666666666667, 1.45, 1.41666666666667, 1.7, 1.86666666666667, 1.96666666666667, 0.233333333333333, 2.2, 2.11666666666667, 2.1, 2.18333333333333, 2.26666666666667, 2.21666666666667, 2.08333333333333, 2.25, 2.05, 0.833333333333333, 0.716666666666667, 0.716666666666667, 0.616666666666667, 0.583333333333333, 0.5, 0.416666666666667, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0166666666666667, 0.133333333333333, 0.283333333333333, 0.45, 0.55, 0.616666666666667, 0.6, 0.75, 0.65, 0.8, 0.966666666666667, 1.13333333333333, 1.28333333333333, 1.46666666666667, 1.43333333333333, 1.46666666666667, 1.31666666666667, 1.45, 1.46666666666667, 1.51666666666667, 1.38333333333333, 1.4, 1.38333333333333, 1.23333333333333, 1.4, 1.56666666666667, 1.73333333333333, 1.8, 1.73333333333333, 1.88333333333333, 1.6, 1.15, 1.05, 1.01666666666667, 0.966666666666667, 1.06666666666667, 1.01666666666667, 0.816666666666667, 0.616666666666667, 0.283333333333333, 0.0666666666666667, 0.133333333333333, 0.3, 0.466666666666667, 0.516666666666667, 0.45, 0.266666666666667, 0.05, 0.0, 0.05, 0.183333333333333, 0.35, 0.433333333333333, 0.516666666666667, 0.633333333333333, 0.7, 0.666666666666667, 0.55, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0666666666666667, 0.216666666666667, 0.35, 0.416666666666667, 0.2, 0.25, 0.416666666666667, 0.316666666666667, 0.45, 0.6, 0.75, 0.9, 1.06666666666667, 1.23333333333333, 1.21666666666667, 1.28333333333333, 1.45, 1.51666666666667, 1.28333333333333, 1.45, 1.45, 1.28333333333333, 1.13333333333333, 1.13333333333333, 1.21666666666667, 1.36666666666667, 1.36666666666667, 1.51666666666667, 1.65, 1.81666666666667, 1.96666666666667, 1.88333333333333, 2.0, 1.78333333333333, 1.36666666666667, 1.28333333333333, 1.33333333333333, 1.5, 1.61666666666667, 1.8, 1.91666666666667, 2.08333333333333, 2.18333333333333, 2.05, 1.66666666666667, 1.2, 1.05, 1.0, 0.766666666666667, 0.916666666666667, 0.966666666666667, 1.11666666666667, 0.883333333333333, 0.9, 0.566666666666667, 0.566666666666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.316666666666667, 0.483333333333333, 0.65, 0.816666666666667, 0.966666666666667, 1.3, 1.45, 1.6, 1.76666666666667, 1.93333333333333, 2.03333333333333, 2.2, 2.2, 2.18333333333333, 2.3, 2.36666666666667, 2.51666666666667, 2.56666666666667, 2.61666666666667, 2.78333333333333, 2.95, 2.73333333333333, 2.56666666666667, 2.18333333333333, 1.85, 1.5, 1.43333333333333, 1.35, 0.5, 0.133333333333333, 0.15, 0.0, 0.0166666666666667, 0.0, 0.116666666666667, 0.25, 0.416666666666667, 0.583333333333333, 0.75, 0.666666666666667, 0.566666666666667, 0.416666666666667, 0.566666666666667, 0.716666666666667, 0.866666666666667, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.216666666666667, 0.383333333333333, 0.533333333333333, 0.7, 0.75, 0.6, 0.683333333333333, 0.366666666666667, 0.5, 0.466666666666667, 0.0166666666666667, 0.166666666666667, 0.3, 0.45, 0.583333333333333, 0.516666666666667, 0.516666666666667, 0.266666666666667, 0.35, 0.516666666666667, 0.683333333333333, 0.8, 0.716666666666667, 0.783333333333333, 0.85, 1.01666666666667, 1.03333333333333, 1.03333333333333, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 0.85, 0.766666666666667, 0.266666666666667, 0.833333333333333, 0.8, 0.55, 0.166666666666667, 0.0666666666666667, 0.116666666666667, 0.0, 0.15, 0.3, 0.283333333333333, 0.3, 0.466666666666667, 0.4, 0.333333333333333, 0.05, 0.133333333333333, 0.05, 0.1, 0.25, 0.366666666666667, 0.383333333333333, 0.533333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.316666666666667, 0.133333333333333, 0.3, 0.466666666666667, 0.633333333333333, 0.8, 0.966666666666667, 1.13333333333333, 1.3, 1.36666666666667, 1.53333333333333, 1.66666666666667, 1.83333333333333, 1.96666666666667, 2.13333333333333, 0.333333333333333, 1.95, 1.95, 1.5, 1.56666666666667, 1.75, 1.9, 2.06666666666667, 2.23333333333333, 2.33333333333333, 2.5, 1.95, 1.65, 1.73333333333333, 1.13333333333333, 0.35, 0.133333333333333, 0.3, 0.433333333333333, 0.533333333333333, 0.4, 0.533333333333333, 0.7, 0.866666666666667, 0.966666666666667, 1.01666666666667, 1.06666666666667, 1.0, 1.03333333333333, 0.816666666666667, 0.65, 0.483333333333333, 0.6, 0.683333333333333, 0.666666666666667, 0.633333333333333, 0.533333333333333, 0.583333333333333, 0.683333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0166666666666667, 0.183333333333333, 0.266666666666667, 0.4, 0.533333333333333, 0.7, 0.866666666666667, 1.0, 1.06666666666667, 1.18333333333333, 0.8, 0.733333333333333, 0.566666666666667, 0.733333333333333, 0.65, 0.683333333333333, 0.8, 0.733333333333333, 0.383333333333333, 0.5, 0.633333333333333, 0.8, 0.95, 1.05, 0.85, 0.833333333333333, 0.866666666666667, 0.766666666666667, 0.816666666666667, 0.783333333333333, 0.716666666666667, 0.883333333333333, 0.516666666666667, 0.433333333333333, 0.5, 0.633333333333333, 0.783333333333333, 0.733333333333333, 0.9, 0.766666666666667, 0.683333333333333, 0.8, 0.716666666666667, 0.833333333333333, 0.933333333333333, 0.95, 0.716666666666667, 0.766666666666667, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.166666666666667, 0.316666666666667, 0.483333333333333, 0.616666666666667, 0.683333333333333, 0.833333333333333, 0.983333333333333, 1.08333333333333, 1.13333333333333, 1.3, 1.21666666666667, 1.26666666666667, 1.43333333333333, 1.58333333333333, 1.58333333333333, 1.63333333333333, 1.2, 0.8, 0.466666666666667, 0.433333333333333, 0.15, 0.0, 0.133333333333333, 0.166666666666667, 0.1, 0.1, 0.0666666666666667, 0.116666666666667, 0.15, 0.183333333333333, 0.3, 0.3, 0.366666666666667, 0.416666666666667, 0.45, 0.516666666666667, 0.633333333333333, 0.633333333333333, 0.0, 0.566666666666667, 0.733333333333333, 0.9, 1.06666666666667, 1.23333333333333, 1.4, 1.56666666666667, 1.6, 1.76666666666667, 1.93333333333333, 1.78333333333333, 1.76666666666667, 1.68333333333333, 0.0166666666666667, 1.16666666666667, 0.8, 0.683333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0333333333333333, 0.2, 0.316666666666667, 0.133333333333333, 0.166666666666667, 0.183333333333333, 0.0333333333333333, 0.0, 0.0166666666666667, 0.0333333333333333, 0.0833333333333333, 0.25, 0.416666666666667, 0.583333333333333, 0.75, 0.816666666666667, 0.55, 0.383333333333333, 0.116666666666667, 0.233333333333333, 0.216666666666667, 0.383333333333333, 0.433333333333333, 0.283333333333333, 0.45, 0.616666666666667, 0.65, 0.433333333333333, 0.583333333333333, 0.683333333333333, 0.85, 0.566666666666667, 0.616666666666667, 0.466666666666667, 0.566666666666667, 0.633333333333333, 0.633333333333333, 0.516666666666667, 0.416666666666667, 0.466666666666667, 0.666666666666667, 0.816666666666667, 0.933333333333333, 0.65, 0.7, 0.8, 0.966666666666667, 1.08333333333333, 1.18333333333333, 1.11666666666667, 1.13333333333333, 1.08333333333333, 1.21666666666667, 1.21666666666667, 1.25, 1.03333333333333, 1.05, 0.883333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0833333333333333, 0.233333333333333, 0.35, 0.4, 0.0, 0.0333333333333333, 0.166666666666667, 0.283333333333333, 0.483333333333333, 0.616666666666667, 0.783333333333333, 0.933333333333333, 0.916666666666667, 0.766666666666667, 0.9, 1.0, 1.05, 0.933333333333333, 0.383333333333333, 0.0333333333333333, 0.133333333333333, 0.0833333333333333, 0.05, 0.0, 0.133333333333333, 0.3, 0.466666666666667, 0.633333333333333, 0.8, 0.666666666666667, 0.75, 0.933333333333333, 0.966666666666667, 1.13333333333333, 1.25, 1.41666666666667, 0.916666666666667, 0.5, 0.633333333333333, 0.833333333333333, 0.233333333333333, 0.383333333333333, 0.533333333333333, 0.633333333333333, 0.8, 0.966666666666667, 1.01666666666667, 1.0, 1.11666666666667, 1.21666666666667, 0.466666666666667, 0.1, 0.266666666666667, 0.383333333333333, 0.466666666666667, 0.433333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.216666666666667, 0.266666666666667, 0.416666666666667, 0.5, 0.383333333333333, 0.55, 0.633333333333333, 0.6, 0.766666666666667, 0.816666666666667, 0.95, 0.65, 0.4, 0.0, 0.3, 0.45, 0.45, 0.616666666666667, 0.716666666666667, 0.883333333333333, 1.05, 1.2, 1.33333333333333, 1.48333333333333, 1.46666666666667, 1.26666666666667, 1.26666666666667, 1.21666666666667, 1.03333333333333, 1.1, 0.966666666666667, 0.666666666666667, 0.733333333333333, 0.666666666666667, 0.816666666666667, 0.783333333333333, 0.516666666666667, 0.483333333333333, 0.35, 0.516666666666667, 0.683333333333333, 0.85, 0.833333333333333, 0.75, 0.883333333333333, 0.983333333333333, 1.06666666666667, 1.0, 0.816666666666667, 0.966666666666667, 0.8, 0.65, 0.75, 0.65, 0.616666666666667, 0.683333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.216666666666667, 0.383333333333333, 0.55, 0.7, 0.866666666666667, 1.03333333333333, 1.2, 1.36666666666667, 1.53333333333333, 1.85, 2.0, 2.13333333333333, 2.18333333333333, 2.31666666666667, 2.48333333333333, 2.41666666666667, 2.58333333333333, 2.68333333333333, 2.83333333333333, 2.9, 2.76666666666667, 2.58333333333333, 2.66666666666667, 2.46666666666667, 2.33333333333333, 1.68333333333333, 1.76666666666667, 0.883333333333333, 0.633333333333333, 0.783333333333333, 0.2, 1.11666666666667, 0.333333333333333, 0.5, 0.666666666666667, 0.816666666666667, 0.95, 0.75, 0.716666666666667, 0.866666666666667, 0.983333333333333, 1.11666666666667, 0.183333333333333, 0.116666666666667, 0.216666666666667, 0.316666666666667, 0.366666666666667, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.133333333333333, 0.283333333333333, 0.366666666666667, 0.366666666666667, 0.416666666666667, 0.433333333333333, 0.566666666666667, 0.583333333333333, 0.75, 0.9, 0.966666666666667, 1.11666666666667, 1.18333333333333, 1.11666666666667, 1.11666666666667, 0.966666666666667, 0.666666666666667, 0.7, 0.533333333333333, 0.683333333333333, 0.816666666666667, 0.833333333333333, 0.766666666666667, 0.933333333333333, 1.08333333333333, 1.13333333333333, 1.18333333333333, 1.23333333333333, 1.25, 1.21666666666667, 1.26666666666667, 1.4, 1.5, 1.5, 1.46666666666667, 1.58333333333333, 1.3, 1.46666666666667, 1.63333333333333, 1.75, 1.85, 1.71666666666667, 1.76666666666667, 1.7, 1.18333333333333, 1.25, 0.85, 0.583333333333333, 0.333333333333333, 0.433333333333333, 0.6, 0.716666666666667, 0.766666666666667, 0.9, 0.616666666666667, 0.616666666666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0333333333333333, 0.2, 0.35, 0.5, 0.566666666666667, 0.4, 0.0166666666666667, 0.15, 0.3, 0.466666666666667, 0.583333333333333, 0.75, 1.01666666666667, 0.866666666666667, 0.983333333333333, 1.06666666666667, 1.08333333333333, 0.716666666666667, 0.633333333333333, 0.7, 0.5, 0.633333333333333, 0.6, 0.766666666666667, 0.9, 1.06666666666667, 1.16666666666667, 1.26666666666667, 1.31666666666667, 1.43333333333333, 1.6, 1.36666666666667, 1.51666666666667, 1.53333333333333, 1.38333333333333, 1.46666666666667, 1.43333333333333, 1.35, 1.1, 1.26666666666667, 1.43333333333333, 1.46666666666667, 1.63333333333333, 1.71666666666667, 1.53333333333333, 1.2, 1.28333333333333, 1.23333333333333, 0.383333333333333, 0.366666666666667, 0.0833333333333333, 0.183333333333333, 0.316666666666667, 0.416666666666667, 0.566666666666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.216666666666667, 0.366666666666667, 0.533333333333333, 0.7, 0.866666666666667, 1.03333333333333, 1.18333333333333, 1.33333333333333, 1.5, 1.63333333333333, 1.71666666666667, 0.0, 1.53333333333333, 1.7, 1.58333333333333, 1.56666666666667, 1.4, 1.51666666666667, 1.68333333333333, 1.81666666666667, 1.58333333333333, 1.53333333333333, 1.66666666666667, 1.8, 1.51666666666667, 1.46666666666667, 1.03333333333333, 0.916666666666667, 0.95, 1.08333333333333, 1.15, 1.2, 1.05, 1.21666666666667, 1.38333333333333, 1.55, 1.7, 1.81666666666667, 1.95, 1.5, 0.583333333333333, 0.583333333333333, 0.483333333333333, 0.566666666666667, 0.7, 0.633333333333333, 0.566666666666667, 0.683333333333333, 0.233333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.116666666666667, 0.0, 0.133333333333333, 0.3, 0.45, 0.6, 0.75, 0.916666666666667, 0.783333333333333, 0.883333333333333, 0.966666666666667, 0.916666666666667, 0.983333333333333, 0.9, 0.516666666666667, 0.383333333333333, 0.0, 0.133333333333333, 0.266666666666667, 0.366666666666667, 0.533333333333333, 0.566666666666667, 0.55, 0.7, 0.866666666666667, 0.866666666666667, 1.06666666666667, 1.13333333333333, 1.28333333333333, 1.36666666666667, 1.33333333333333, 1.48333333333333, 1.51666666666667, 1.21666666666667, 1.13333333333333, 0.75, 0.75, 0.466666666666667, 0.533333333333333, 0.583333333333333, 0.75, 0.916666666666667, 0.966666666666667, 0.85, 0.733333333333333, 0.533333333333333, 0.25, 0.3, 0.3, 0.466666666666667, 0.633333333333333, 0.7, 0.766666666666667, 0.716666666666667, 0.516666666666667, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.2, 0.35, 0.5, 0.583333333333333, 0.4, 0.566666666666667, 0.733333333333333, 0.9, 0.883333333333333, 1.05, 1.21666666666667, 1.35, 1.48333333333333, 1.5, 1.6, 1.41666666666667, 1.51666666666667, 1.68333333333333, 0.0, 0.716666666666667, 1.25, 1.11666666666667, 1.28333333333333, 1.1, 1.25, 1.41666666666667, 1.08333333333333, 1.1, 1.01666666666667, 1.15, 1.08333333333333, 1.15, 1.26666666666667, 1.13333333333333, 0.916666666666667, 0.6, 0.683333333333333, 0.533333333333333, 0.583333333333333, 0.533333333333333, 0.633333333333333, 0.783333333333333, 0.95, 1.0, 1.05, 1.18333333333333, 1.0, 0.683333333333333, 0.65, 0.633333333333333, 0.766666666666667, 0.633333333333333, 0.766666666666667, 0.7, 0.85, 0.8, 0.65, 0.733333333333333, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.216666666666667, 0.366666666666667, 0.516666666666667, 0.583333333333333, 0.75, 0.7, 0.566666666666667, 0.566666666666667, 0.566666666666667, 0.566666666666667, 0.566666666666667, 0.566666666666667, 0.566666666666667, 0.566666666666667, 0.566666666666667, 0.566666666666667, 0.566666666666667, 0.566666666666667, 1.41666666666667, 1.1, 0.983333333333333, 1.08333333333333, 1.13333333333333, 1.18333333333333, 1.13333333333333, 0.316666666666667, 0.4, 0.566666666666667, 0.733333333333333, 0.866666666666667, 1.03333333333333, 1.2, 1.26666666666667, 1.25, 1.3, 1.36666666666667, 1.53333333333333, 1.45, 1.18333333333333, 1.28333333333333, 1.21666666666667, 1.08333333333333, 1.15, 1.28333333333333, 1.45, 1.61666666666667, 1.76666666666667, 1.88333333333333, 2.0, 2.03333333333333, 1.76666666666667, 1.65, 1.31666666666667, 1.03333333333333, 0.8, 0.483333333333333, 0.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.116666666666667, 0.25, 0.383333333333333, 0.233333333333333, 0.233333333333333, 0.383333333333333, 0.55, 0.633333333333333, 0.8, 0.733333333333333, 0.733333333333333, 0.9, 1.06666666666667, 1.2, 1.3, 1.35, 1.51666666666667, 1.68333333333333, 1.81666666666667, 1.9, 1.86666666666667, 1.91666666666667, 1.08333333333333, 0.916666666666667, 0.966666666666667, 0.816666666666667, 0.766666666666667, 0.816666666666667, 0.883333333333333, 1.01666666666667, 1.18333333333333, 1.35, 1.33333333333333, 1.41666666666667, 1.16666666666667, 0.583333333333333, 0.133333333333333, 0.166666666666667, 0.233333333333333, 0.25, 0.35, 0.416666666666667, 0.55, 0.55, 0.683333333333333, 0.85, 0.75, 0.383333333333333, 0.316666666666667, 0.183333333333333, 0.116666666666667, 0.05, 0.2, 0.233333333333333, 0.4, 0.516666666666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.216666666666667, 0.383333333333333, 0.566666666666667, 0.666666666666667, 0.85, 0.95, 1.11666666666667, 1.23333333333333, 1.2, 1.15, 1.15, 1.18333333333333, 1.33333333333333, 1.33333333333333, 1.43333333333333, 1.33333333333333, 1.45, 1.43333333333333, 1.51666666666667, 1.33333333333333, 1.28333333333333, 1.3, 1.36666666666667, 1.53333333333333, 1.58333333333333, 1.75, 1.1, 1.06666666666667, 0.55, 0.533333333333333, 0.666666666666667, 0.816666666666667, 0.983333333333333, 1.15, 1.35, 1.18333333333333, 0.983333333333333, 0.816666666666667, 0.933333333333333, 0.766666666666667, 0.7, 0.566666666666667, 0.483333333333333, 0.6, 0.666666666666667, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0166666666666667, 0.0833333333333333, 0.116666666666667, 0.0, 0.0, 0.133333333333333, 0.266666666666667, 0.433333333333333, 0.6, 0.733333333333333, 0.85, 1.0, 1.16666666666667, 1.21666666666667, 1.28333333333333, 1.18333333333333, 1.3, 1.23333333333333, 0.783333333333333, 0.683333333333333, 0.666666666666667, 0.683333333333333, 0.316666666666667, 0.233333333333333, 0.116666666666667, 0.15, 0.0333333333333333, 0.0, 0.0, 0.0666666666666667, 0.0, 0.133333333333333, 0.3, 0.466666666666667, 0.516666666666667, 0.533333333333333, 0.45, 0.616666666666667, 0.766666666666667, 0.383333333333333, 0.35, 0.466666666666667, 0.433333333333333, 0.2, 0.316666666666667, 0.483333333333333, 0.4, 0.216666666666667, 0.0166666666666667, 0.15, 0.283333333333333, 0.416666666666667, 0.133333333333333, 0.2, 0.25, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.166666666666667, 0.0166666666666667, 0.0, 0.0166666666666667, 0.15, 0.233333333333333, 0.4, 0.45, 0.183333333333333, 0.183333333333333, 0.35, 0.516666666666667, 0.616666666666667, 0.716666666666667, 0.8, 0.616666666666667, 0.916666666666667, 0.633333333333333, 0.566666666666667, 0.416666666666667, 0.433333333333333, 0.583333333333333, 0.75, 0.766666666666667, 0.916666666666667, 1.01666666666667, 1.15, 1.25, 1.1, 1.16666666666667, 1.33333333333333, 1.41666666666667, 1.51666666666667, 1.56666666666667, 1.65, 1.6, 1.71666666666667, 1.91666666666667, 1.98333333333333, 2.05, 1.8, 0.183333333333333, 1.53333333333333, 1.7, 1.8, 1.63333333333333, 1.56666666666667, 1.58333333333333, 1.68333333333333, 1.51666666666667, 1.35, 0.0833333333333333, 1.06666666666667, 0.833333333333333, 0.766666666666667, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0666666666666667, 0.183333333333333, 0.383333333333333, 0.533333333333333, 0.7, 0.866666666666667, 1.03333333333333, 1.13333333333333, 1.26666666666667, 1.28333333333333, 0.733333333333333, 0.883333333333333, 1.05, 1.21666666666667, 1.36666666666667, 0.916666666666667, 0.75, 0.8, 0.966666666666667, 1.1, 1.18333333333333, 1.3, 1.38333333333333, 1.33333333333333, 1.46666666666667, 1.63333333333333, 1.53333333333333, 1.55, 1.21666666666667, 1.28333333333333, 1.01666666666667, 1.18333333333333, 1.33333333333333, 1.16666666666667, 0.766666666666667, 0.8, 0.583333333333333, 0.7, 0.866666666666667, 1.03333333333333, 1.2, 1.31666666666667, 1.48333333333333, 1.41666666666667, 1.46666666666667, 1.36666666666667, 1.43333333333333, 1.53333333333333, 1.65, 1.5, 0.916666666666667, 0.933333333333333, 1.0, 1.08333333333333, 1.06666666666667, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.216666666666667, 0.366666666666667, 0.466666666666667, 0.55, 0.366666666666667, 0.483333333333333, 0.65, 0.816666666666667, 0.983333333333333, 1.01666666666667, 1.03333333333333, 1.11666666666667, 1.26666666666667, 1.31666666666667, 1.46666666666667, 1.18333333333333, 1.35, 1.43333333333333, 1.43333333333333, 0.966666666666667, 0.75, 0.816666666666667, 0.55, 0.3, 0.3, 0.466666666666667, 0.383333333333333, 0.2, 0.166666666666667])
#a= [1,2,3,4,5,6,1,2,3,4,510]
np.vstack(a)
centroids,_ = kmeans(a,5)
idx,_ = vq(a,centroids)
print idx