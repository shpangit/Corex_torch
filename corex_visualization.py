import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.colors as pltcolor

def softmax(x):
	return np.exp(x)/np.sum(np.exp(x))

### Visualsiation tools with netorkx

def get_edges(corex,dist = 'inv_log',no_alpha = False):
	mis = corex.mis.cpu()
	alpha = torch.tensor(1)
	if not no_alpha:
		alpha = corex.alpha.cpu()

	edges,argmax_misalpha = torch.max(mis*alpha,dim = 0)
	argmax_misalpha = argmax_misalpha.cpu().numpy().reshape(-1)
	edges = edges.cpu().numpy().reshape(-1)
	test = np.array_equal(argmax_misalpha,corex.clusters)
	if not test:
		print('!!! clusters and distances do not match')
	
	# Correction for MIS > 1 (since it is normalized mis it should be betwwen 0 and 1)
	# By doing this, the maximum distance is bounded
	edges = np.where(edges>1,torch.tensor(1e-5),edges)
	

	if dist == 'softmax':
		distances = np.log(1/softmax(edges))
	else: #"log_inv"
		distances = np.log(1/edges)
		
	
	return distances

def get_links(corex):
	return corex.clusters


def plot_clusters_graph(vis_nodes,lat_nodes,linked,edges,color):
	# plt.close()
	G = nx.Graph()
	
	for i,y_node in enumerate(lat_nodes):
		y_node_name = "Y"+str(y_node)
		G.add_node(y_node_name,color = 'green')
		if i>=1 : #ADD EDGES BETWEEN CLUSTERS
			G.add_edge(y_node_name,y_node_prec,weight = 2)
		y_node_prec = "Y"+str(y_node)
	
	if not isinstance(color,(list,np.ndarray)):
		color = [color]*len(linked)
	
	for i,l in enumerate(linked):
		G.add_node(vis_nodes[i],color=color[i])
		if l not in lat_nodes:
			raise Exception("cluster number is not in latent nodes")
		G.add_edge(vis_nodes[i],'Y'+str(l),weight = edges[i])

	pos = nx.kamada_kawai_layout(G)

	nx.draw(G,pos,node_color = list(nx.get_node_attributes(G,'color').values()),with_labels = True)
	plt.show()

def plot_corex_clusters(corex,color = 'blue'):
	edges = get_edges(corex)
	linked = get_links(corex)
	lat_nodes = list(range(corex.n_hidden))
	vis_nodes = list(range(corex.n_visible))

	plot_clusters_graph(vis_nodes,lat_nodes,linked,edges,color)


# Update 18/11/2020 Plotly use graph :

def make_plotly_corex(list_of_corex,color_bar = False,dist='inv_log',no_alpha = False):
	E = [] # list of edge_value for each layer
	L = [] # List of Latent nodes for each layer
	V = [] # List of visible nodes for each layer
	Links = [] # list of links (visibmel-latent) for each layer
	CS = list_of_corex# List of Corex. Should be sorteb by layer number

	# Extract nodes and edges information from Corex'
	for i,cor in enumerate(CS):
	#Getting nodes,linked,edged value
	## nodes
		latent_prefix = 'Y' * i
		if i==0: # if we are at the first layer : vsibiel - latent variables
			vis_nodes = list((range(cor.n_visible)))
		else:
			vis_nodes = [latent_prefix+str(i) for i in range(cor.n_visible)]
		lat_nodes = [latent_prefix+'Y'+str(i) for i in range(cor.n_hidden)]
		#Edges value
		edges = get_edges(cor,dist)
		linked = get_links(cor)
	
		E.append(edges)
		L.append(lat_nodes)
		V.append(vis_nodes)
		Links.append(linked)
		
	# Create Networkx Graphe
	G = nx.Graph()
	G.add_nodes_from(V[0])
	for l in L:
		G.add_nodes_from(l)

	#Add edges
	for i,(vis_nodes,link,edges) in enumerate(zip(V,Links,E)):
		for j,l in enumerate(link):
			prefix = 'Y' * (i+1)
			G.add_edge(vis_nodes[j],prefix+str(l),weight = edges[j])

	# Distancing the last clusters
	last_latent = L[-1]
	for i,y_node in enumerate(last_latent):
		if i==1:
			pass
		y_node_prec = last_latent[i-1]
		G.add_edge(y_node,y_node_prec,weight = 5)

	pos = nx.kamada_kawai_layout(G) # for now we suppose only this layout
	nx.set_node_attributes(G,pos,'pos')

	#Plotly Config code:
	edge_x = []
	edge_y = []
	for edge in G.edges():
		e0,e1 = edge
		if isinstance(e0,str) and isinstance(e1,str):
			if e0.startswith('Y') and e1.startswith('Y'):
				continue
			
		x0, y0 = G.nodes[edge[0]]['pos']
		x1, y1 = G.nodes[edge[1]]['pos']
		
		edge_x.append(x0)
		edge_x.append(x1)
		edge_x.append(None)
		edge_y.append(y0)
		edge_y.append(y1)
		edge_y.append(None)

	edge_trace = go.Scatter(
		x=edge_x, y=edge_y,
		line=dict(width=0.5, color='#888'),
		hoverinfo='text',
		mode='lines')

	vis_node_x = []
	vis_node_y = []
	lat_node_x = []
	lat_node_y = []
	for node in G.nodes():
		x, y = G.nodes[node]['pos']
		if isinstance(node,int):
			vis_node_x.append(x)
			vis_node_y.append(y)
		else:
			lat_node_x.append(x)
			lat_node_y.append(y)


	lat_node_trace = go.Scatter(
		x=lat_node_x, y=lat_node_y,
		mode='markers',
		hoverinfo='text',
		marker=dict(
			showscale=color_bar,
			# colorscale options
			#'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
			#'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
			#'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
			# colorscale='YlGnBu',
			colorscale='Reds',
			reversescale=False,
			color=[],
			size=15,
			colorbar=dict(
				thickness=15,
				title='TC',
				xanchor='left',
				titleside='right',
			),
			line=dict(
                color='DarkSlateGrey',
                width=1
            )))

	vis_node_trace = go.Scatter(
		x=vis_node_x, y=vis_node_y,
		mode='markers',
		hoverinfo='text',
		marker = dict(
			size = 8,
			color = []
		),
		line_width=1)

	# Seting attributes to nodes and edges
	vis_node_text = []
	vis_node_color = []
	lat_node_val = []
	lat_node_text = []
	lat_node_size = []
	for node in G.nodes:
		if isinstance(node,int) : # Visible nodes are named as integer
			vis_node_text.append('Var '+str(node))
			vis_node_color.append("#888")
		else: # latent nodes are named starting with 'Y'
			c_number = int(node.replace('Y',''))
			layer_number = node.count('Y')
			cor = CS[layer_number - 1]
			tc = cor.tcs.cpu().numpy()[c_number]
			lat_node_val.append(tc)
			lat_node_text.append("Layer "+str(layer_number)+"<br>Cluster node "+str(c_number)+ "<br>TC : "+str(tc))
			lat_node_size.append(layer_number*5+10)
			
	vis_node_trace.text = vis_node_text
	vis_node_trace.marker.color = vis_node_color
	lat_node_trace.text = lat_node_text
	lat_node_trace.marker.color = lat_node_val 
	lat_node_trace.marker.size = lat_node_size

	res = {
				'vis_nodes':vis_node_trace,
				'lat_nodes':lat_node_trace,
				'edge':edge_trace

		}
	return res

def make_fig(vis_node_trace,edge_trace,lat_node_trace,title=None,color_bar=False,plot_var_names=False,plot_z_names = False):
	fig = go.Figure(data=[vis_node_trace,edge_trace,lat_node_trace],
			 layout=go.Layout(
				title=title,
				titlefont_size=16,
				showlegend=False,
				hovermode='closest',
				margin=dict(b=20,l=5,r=5,t=40),
				xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
				yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
				)

	# Plot variable names in the graph and the Z associated
	if plot_var_names:
		fig.data[0]['mode'] = 'markers+text'
		# fig.data[0]['textposition'] = 'bottom center'
	
	if plot_z_names:
		fig.data[2]['mode'] = 'markers+text'
		fig.data[2]['hovertext'] = fig.data[2]['text']
		fig.data[2]['text'] = ["$Z_{"+str(i+1)+"}$" for i in range(len(fig.data[2]['text']))]
		# fig.data[2]['textposition'] = 'bottom center'

	return fig

def plotly_corex(list_of_corex,color_bar=False,dist='inv_log',no_alpha=False):
	pt = make_plotly_corex(list_of_corex,color_bar,dist,no_alpha)
	vis_node,lat_node,edge = pt['vis_nodes'],pt['lat_nodes'],pt['edge']
	fig = make_fig(vis_node,edge,lat_node)
	return fig

# Update for change markers symbol for depending the real calss of the the input
def plot_corex_with_vnames(corex,col_names,dist = 'inv_log',color_bar = False,title=None,
							plot_var_names=False,plot_z_names=False,
							vis_node_colors=None):
	"""
	vis_node_colors : list, depending on the value, the colors of visible nodes 'Xi' cahnge colors 
	"""
	pt = make_plotly_corex([corex],color_bar=color_bar,dist=dist,no_alpha=True)
	vis_node,lat_node,edges = pt['vis_nodes'],pt['lat_nodes'],pt['edge']
	vis_node.text = col_names

	fig = make_fig(vis_node,edges,lat_node,title=title,color_bar=color_bar,plot_var_names=plot_var_names,plot_z_names=plot_z_names)
	if isinstance(vis_node_colors,(list,np.ndarray)):
		fig.data[2]['marker']['symbol'] = 0 # symbols for z is circle
		fig.data[0]['marker']['symbol'] = 1 # Symbols for x is a square

		# Adding colors
		c_list = np.array(pltcolor.qualitative.Dark24)
		vis_node_colors = c_list[vis_node_colors]
		fig.data[0]['marker']['color'] = vis_node_colors

	return fig

# Update Dec.2022
# Adding a funciton for reconnect plotly in the notebook
from plotly.offline import init_notebook_mode

def reload_plotly():
	init_notebook_mode(connected=True)  