#coding=utf-8
import numpy as np
import networkx as nx
import random
import logging

class Graph():

	def __init__(self, nx_G, is_directed, p, q):
		self.G = nx_G
		self.is_directed = is_directed
		self.p = p
		self.q = q

	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges
		walk = [start_node]
		while len(walk) < walk_length:
			cur = walk[-1]
			# 求当前结点的邻居结点
			cur_nbrs = sorted(G.neighbors(cur))
			# 如果存在邻居结点
			if len(cur_nbrs) > 0:
				# 如果序列中仅有一个结点，即第一次游走
				if len(walk) == 1:
					"""
					结合cur_nbrs = sorted(G.neighbor(cur)) 和 alias_nodes/alias_edges的序号，才能确定节点的ID。
					所以路径上的每个节点在确定下一个节点是哪个的时候，都要经过sorted(G.neighbors(cur))这一步。"""

					'''
					由于提前把概率设置并储存在图数据中，这里调用alias_draw（）即可获得。
					alias_draw实现的就是Alias Method采样方法。
					'''
					walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					# 如果序列中有多个结点 # 找到当前游走结点的前一个结点
					prev = walk[-2]
					next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],
											   alias_edges[(prev, cur)][1])]
					walk.append(next)
			else:
				break
		return walk

	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		"""
		对每个结点，根据num_walks得出其多条随机游走路径
		"""
		logging.info("Repeatedly simulate random walks from each node...")
		G = self.G
		walks = []
		logging.info("all nodes to list")
		nodes = list(G.nodes())
		print('Walk iteration:')
		for walk_iter in range(num_walks):
			print(str(walk_iter + 1), '/', str(num_walks))
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))
		return walks

	def get_alias_edge(self, src, dst):
		'''
		Get the alias edge setup lists for a given edge.
		:param src:  随机游走序列中的上一个结点
        :param dst:  当前结点
		'''
		G = self.G
		p = self.p
		q = self.q
		unnormalized_probs = []
		# 这里可以进行优化，默认是选取所有的邻居结点
		# 三种情况
		for dst_nbr in sorted(G.neighbors(dst)):
			# 返回源结点
			if dst_nbr == src:
				unnormalized_probs.append(G[dst][dst_nbr]['weight'] / p)
			# 源结点和这个目标结点的邻居结点之间有直连边
			elif G.has_edge(dst_nbr, src):
				unnormalized_probs.append(G[dst][dst_nbr]['weight'])
			# 没有直连边
			else:
				unnormalized_probs.append(G[dst][dst_nbr]['weight'] / q)
		norm_const = sum(unnormalized_probs)
		try:
			# 概率归一化
			normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
		except:
			normalized_probs = [0.0 for u_prob in unnormalized_probs]
		return alias_setup(normalized_probs)

	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		logging.info("Start Preprocessing of transition probabilities for guiding the random walks.")
		G = self.G
		is_directed = self.is_directed
		# 存储每个结点对应的两个采样列表
		alias_nodes = {}
		# G.nodes()返回一个结点列表
		for node in G.nodes():
			# 得到当前结点的邻居结点(有直连关系)的权值列表，[1,1,1,1...]
			unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
			# 权重求和
			norm_const = sum(unnormalized_probs)
			try:
				# 求每个权重的占的比重，权重大的占的比重就大
				normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
			except:
				print(node)
				normalized_probs = [0.0 for u_prob in unnormalized_probs]
			alias_nodes[node] = alias_setup(normalized_probs)
		alias_edges = {}
		triads = {}
		if is_directed:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		else:
			# G.edges()返回一个列表元组，列表里面是边关系，形如[(1,2), (1,3), ...]
			# (1,2)代表结点1和结点2之间有一条边
			for edge in G.edges():
				# 先构建(1,2)，再构建(2,1)
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])
		# alias_nodes形式为{1:(J, q), 2:(J,q)...},1和2代表结点id
		# alias_edges形式为{(1,2):(J,q), (2,1):(J,q),(1,3):(J,q)...} (1,2)代表一条边
		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges
		return

def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	alias_setup的作用是根据二阶random walk输出的概率变成每个节点对应两个数，被后面的alias_draw函数所进行抽样
    :param probs: 结点之间权重所占比例向量，是一个列表
    :return: 输入概率，得到对应的两个列表，
             一个是在原始的prob数组[0.4,0.8,0.6,1]，
	'''
	# J和q数组和probs数组大小一致
	# probs长度由当前结点的邻居节点数量决定
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)
	# 将数据分类为具有概率的结果 大于或者小于1 / K.
	# 这两个列表里存放的是结点的下标
	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
		q[kk] = K * prob
		if q[kk] < 1.0:
			smaller.append(kk)
		else:
			larger.append(kk)
	# 然后循环并创建少量二元混合分布
	# 在整个均匀混合分布中适当地分配更大的结果。
	# 假如每条边权重都为1，实际上这里的while循环不会执行，因为每条边概率都是一样的，相当于不需要采样
	while len(smaller) > 0 and len(larger) > 0:
		small = smaller.pop() # smaller自己也会减少最右边的值
		large = larger.pop()
		# 在代码中的实现思路： 构建方法：
		# 1.找出其中面积小于等于1的列，如i列，这些列说明其一定要被别的事件矩形填上，所以在Prab[i]中填上其面积
		# 2.然后从面积大于1的列中，选出一个，比如j列，用它将第i列填满，然后Alias[i] = j，第j列面积减去填充用掉的面积。
		J[small] = large
		q[large] = q[large] + q[small] - 1.0
		if q[large] < 1.0:
			smaller.append(large)
		else:
			larger.append(large)
	return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	抽样函数
    使用alias采样从一个非均匀离散分布中采样
	'''
	K = len(J)
	# 从整体均匀混合分布中采样
	kk = int(np.floor(np.random.rand() * K))
	# 从二元混合中采样，要么保留较小的，要么选择更大的
	if np.random.rand() < q[kk]:
		return kk
	else:
		return J[kk]

