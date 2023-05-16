# READ ME

CorEx implementation with the S&P500 example.  

Tested on python 3.9.0

It is a re-implementation of https://github.com/gregversteeg/CorEx.  
The references are : 

Discovering Structure in High-Dimensional Data Through Correlation Explanation
Greg Ver Steeg and Aram Galstyan, NIPS 2014, http://arxiv.org/abs/1406.1222

Some theoretical developments are described here:
Maximally Informative Hierarchical Representions of High-Dimensional Data
Greg Ver Steeg and Aram Galstyan, AISTATS 2015, http://arxiv.org/abs/1410.7404

The folder is organised with following files and folder:

- corex.py: original numpy code for CorEx algorithm from https://github.com/gregversteeg/CorEx.  
- corex_torch.py: CorEx algorithm with torch implementation.
- corex_visualization.py: plotly based tools to show results in form of graph.
- data: folder containing the dataset of S&P500 components.
- Corex example on SP500.ipynb: Jupyter notebook with an example of using CorEx for clustering the assets. 
- requirements.txt: list of dependecy for pip installation

*corex.py* and *corex_torch.py* have same structure.  
