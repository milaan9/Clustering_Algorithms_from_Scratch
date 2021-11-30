<p align="center"> 
<a href="https://github.com/milaan9"><img src="https://img.shields.io/static/v1?logo=github&label=maintainer&message=milaan9&color=ff3300" alt="Last Commit"/></a> 
<a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fmilaan9%2FClustering_Algorithms_from_Scratch%2Ftree%2Fmain%2F01_MATLAB&count_bg=%231DC92C&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=views&edge_flat=false"/></a>
</p> 
<!--<img src="https://badges.pufler.dev/contributors/milaan9/01_Python_Introduction?size=50&padding=5&bots=true" alt="milaan9"/>-->
 
 # Clustering Algorithms with MATLAB

## 1. Clustering Algorithms
- **K-means**
    - **K-means** algorithm performs the division of data points into 'K' clusters that share similarities and are dissimilar to the objects belonging to another cluster where, each data point belongs to the cluster with the nearest mean (cluster centers or cluster centroids), serving as a prototype of the cluster. The term 'K' is a number. You need to tell the system how many clusters you need to create. For example, K = 2 refers to two clusters.

- **K-means++**
    - Generally speaking, **K-means++** algorithm is similar to **K-means**;
    - Unlike classic K-means randomly choosing initial centroids, a better initialization procedure is integrated into **K-means++**, where observations far from existing centroids have higher probabilities of being chosen as the next centroid.
    - The initializeation procedure can be achieved using Fitness Proportionate Selection.

- **ISODATA (Iterative Self-Organizing Data Analysis)**
    - To be brief, **ISODATA** introduces two additional operations: Splitting and Merging;
    - When the number of observations within one class is less than one pre-defined threshold, **ISODATA** merges two classes with minimum between-class distance; 
    - When the within-class variance of one class exceeds one pre-defined threshold, **ISODATA** splits this class into two different sub-classes.

- **Mean Shift**
	- For each point *x*, find neighbors, calculate mean vector *m*, update *x = m*, until *x == m*;
	- Non-parametric model, no need to specify the number of classes;
	- No structure priori.

- **DBSCAN (Density-Based Spatial Clustering of Application with Noise)**
	- Starting with pre-selected core objects, DBSCAN extends each cluster based on the connectivity between data points;
	- DBSCAN takes noisy data into consideration, hence robust to outliers;
	- Choosing good parameters can be hard without prior knowledge;
- **Gaussian Mixture Model (GMM)**
- **LVQ (Learning Vector Quantization)**

## 2. Subspace Clustering Algorithms
- **Subspace K-means**
    - This algorithm directly extends **K-means** to Subspace Clustering through multiplying each dimension *d<sub>j</sub>* by one weight *m<sub>j</sub>* (s.t. sum(*m<sub>j</sub>*)=1, *j*=1,2,...,*p*);
    - It can be efficiently sovled in an Expectation-Maximization (EM) fashion. In each E-step, it updates weights, centroids using Lagrange Multiplier;
    - This rough algorithm suffers from the problem on its favor of using just a few dimensions when clustering sparse data;

- **Entropy-Weighting Subspace K-means**
    - Generally speaking, this algorithm is similar to **Subspace K-means**;
    - In addition, it introduces one regularization item related to weight entropy into the objective function, in order to mitigate the aforementioned problem in **Subspace K-means**.
    - Apart from its succinctness and efficiency, it works well on a broad range of real-world datasets.
