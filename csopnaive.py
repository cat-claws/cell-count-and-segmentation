import torch
import numpy as np
import networkx as nx
import torch.nn.functional as F
from constraint import *

def getColoredIndices(insts):
	# insts [H, W]
	return [x for x in torch.unique(insts) if x != 0]

def getCostDict(preds, insts, num = 5):
	# preds [channel, H, W]
	# insts [H, W]
	with torch.no_grad():
		indices = getColoredIndices(insts)
		costs = {}
		for index in indices:
			label = (insts.unsqueeze(0) == index).long()
			for k in range(1, num + 1):
				costs[(index.item(), k)] = F.cross_entropy(preds.unsqueeze(0), label * k, ignore_index = 0).item()
	return costs


def getNeighbours(inst_map):
	# inst_map [H, W]
	neighboursUD = inst_map[:-1]!= inst_map[1:]
	neighboursLR = inst_map[:, :-1]!= inst_map[:, 1:]
	neighbours = np.concatenate((np.stack((inst_map[:-1][neighboursUD], inst_map[1:][neighboursUD]), axis = 1),
								 np.stack((inst_map[:, :-1][neighboursLR], inst_map[:, 1:][neighboursLR]), axis = 1)), axis = 0)
	return neighbours

def getGraph(inst_map):	
	# inst_map [H, W]
	graph = nx.Graph()
	graph.add_nodes_from(np.unique(inst_map))
	graph.add_edges_from(getNeighbours(inst_map))
	if 0 in graph:
		graph.remove_node(0)
	return graph

def findAllSolutionsCSP(graph, num = 5):
	problem = Problem()
	problem.addVariables(graph.nodes(), range(1, num + 1))
	for edge in graph.edges():
		problem.addConstraint(lambda u, v: u != v, edge)
	return problem.getSolutionIter()

def getDictCSOP(preds, insts, num = 5):
	# preds [channel, H, W]
	# insts [H, W]
	costs = getCostDict(preds, insts, num)
	G = getGraph(insts.cpu())
	solution = {}
	for x in (G.subgraph(c) for c in nx.connected_components(G)):
		subsolutions = findAllSolutionsCSP(x)
		mincost = float('inf')
		for subsol in subsolutions:
			temp = sum([costs[(k, v)] for k, v in subsol.items()])
			if temp < mincost:
				mincost = temp
				subsolution = subsol
		solution.update(subsolution)
	return solution

def replace_with_dict(ar, dic):
	ar = ar.cpu().numpy()
	# Extract out keys and values
	k = np.array(list(dic.keys()))
	v = np.array(list(dic.values()))

	# Get argsort indices
	sidx = k.argsort()

	# Drop the magic bomb with searchsorted to get the corresponding
	# places for a in keys (using sorter since a is not necessarily sorted).
	# Then trace it back to original order with indexing into sidx
	# Finally index into values for desired output.
	return torch.from_numpy(v[sidx[np.searchsorted(k,ar,sorter=sidx)]])

def adjust_labels(preds, labels, colors = 5):
	with torch.no_grad():
		for i in range(preds.shape[0]):
			solution = getDictCSOP(preds[i], labels[i], num = colors)
			solution[0] = 0
			labels[i] = replace_with_dict(labels[i], solution).to(preds.device)
	return labels
