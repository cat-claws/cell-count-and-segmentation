import random
import numpy as np
import networkx as nx

def getNeighbours(inst_map):
	neighboursUD = inst_map[:-1]!= inst_map[1:]
	neighboursLR = inst_map[:, :-1]!= inst_map[:, 1:]
	neighbours = np.concatenate((np.stack((inst_map[:-1][neighboursUD], inst_map[1:][neighboursUD]), axis = 1),
								 np.stack((inst_map[:, :-1][neighboursLR], inst_map[:, 1:][neighboursLR]), axis = 1)), axis = 0)
	return neighbours


def getConstrainedMap(inst_map):	
	graph = nx.Graph()
	graph.add_nodes_from(np.unique(inst_map))
	graph.add_edges_from(getNeighbours(inst_map))
	graph.remove_node(0)

	transdict = nx.coloring.greedy_color(graph, strategy="random_sequential")

	labels = list(range(5)) # 5 is the maximum neighbour number here
	np.random.shuffle(labels)
	
	for k in transdict:
		transdict[k] = labels[transdict[k]] + 1
	transdict[0] = 0

	return replace_with_dict(inst_map, transdict)

def getConstrainedMapNeg(inst_map):	
	graph = nx.Graph()
	graph.add_nodes_from(np.unique(inst_map))
	graph.add_edges_from(getNeighbours(inst_map))
	graph.remove_node(0)

	if graph.number_of_edges() < 1 or graph.number_of_nodes() < 2:
		return None

	else:
		if graph.number_of_edges() > 1:
			graph.remove_edges_from(random.sample(list(graph.edges()), np.random.randint(1, graph.number_of_edges())))
		u, v = random.choice(list(graph.edges()))
		graph = nx.contracted_nodes(graph, u, v)

		transdict = nx.coloring.greedy_color(graph, strategy="random_sequential")
		transdict[v] = transdict[u]

		labels = list(range(5)) # 5 is the maximum neighbour number here
		np.random.shuffle(labels)

		for k in transdict:
			transdict[k] = labels[transdict[k]] + 1
		transdict[0] = 0

		return replace_with_dict(inst_map, transdict)

# https://www.py4u.net/discuss/169577
def replace_with_dict(ar, dic):
	# Extract out keys and values
	k = np.array(list(dic.keys()))
	v = np.array(list(dic.values()))

	# Get argsort indices
	sidx = k.argsort()

	# Drop the magic bomb with searchsorted to get the corresponding
	# places for a in keys (using sorter since a is not necessarily sorted).
	# Then trace it back to original order with indexing into sidx
	# Finally index into values for desired output.
	return v[sidx[np.searchsorted(k,ar,sorter=sidx)]]
