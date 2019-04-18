# degree discount
"""
N(v):= v的neighbour
Star(v) :=由 v以及它的邻居组成的子图
t_v 是 t的所有邻居中已经被激活的 node集合的size
d_v 出度

d_v = O(1/p)
t_v = O(1/p)

公式 1： 1 +(d_v - 2t_v - (d_v - t_v)*t_v*p+O(t_v)) *p

在IC模型with 概率p


"""
import sys
import numpy as np
import copy
import time
import heapq
import random

# TODO 改成用collection.deque来优化
node_list = None
begin = time.time()
end = None


class ddv:
    def __init__(self, d, dd, t, i):
        self.d = d
        self.dd = dd
        self.t = t
        self.index = i

    def __lt__(self, other):
        return self.dd > other.dd  # 逆序


class Node:
    def __init__(self, i):
        self.mg1 = 0
        self.flag = 0
        self.t_mg1 = 0
        self.index = i

    def __lt__(self, other):
        return self.mg1 > other.mg1  # 逆序


def degree_discount_IC(k, out_list, weight):
    S = list()
    Q = []
    V = dict()
    for key in out_list:
        d = sum([weight[key][i] for i in out_list[key]])
        V[key] = ddv(d, d, 0, key)
        heapq.heappush(Q, V[key])
    while len(S) != k:
        u = heapq.heappop(Q).index
        S.append(u)
        for node in out_list[u]:
            v = V[node]
            v.t += 1
            v.dd = v.d - 2 * v.t - (v.d - v.t) * v.t * weight[u][v.index]
    return S


def read_args():
    social_network = sys.argv[2]
    seed_set = sys.argv[4]
    model = sys.argv[6]
    time_limit = int(sys.argv[8])
    return social_network, seed_set, model, time_limit


def read_graph(fileName):
    edges = []
    with open(fileName, 'r') as f:
        meta = next(f).split()
        node_num = int(meta[0])
        edge_num = int(meta[1])
        for line_i in range(edge_num):
            tmp = next(f).split()
            tmp2 = list(map(int, tmp[:-1]))
            tmp2.append(float(tmp[-1]))
            edges.append(tmp2)
    return node_num, edge_num, edges


def ISE_IC(activity_set, out_nei, weight):
    """
    neighbor: adjacent list
    weight:   numpy.ndarray
    activity_set: set
    """
    all_active = set()
    count = len(activity_set)
    while activity_set:
        for node in activity_set:
            all_active.add(node)
        new_activity_set = set()
        for seed in activity_set:
            neis = out_nei[seed]
            for nei in neis:
                if nei not in all_active:
                    if random.random() < weight[seed][nei]:  # active
                        new_activity_set.add(nei)
        count += len(new_activity_set)
        activity_set = new_activity_set
    return count


def ISE_LT(a_set, out_list, weight, G, in_list):
    """
    G: (node, threshold)
    """
    activity_set = copy.deepcopy(a_set)
    all_active = set()
    for key in G:
        if G[key] == 0:
            activity_set.add(key)
    count = len(activity_set)
    while activity_set:
        for node in activity_set:
            all_active.add(node)
        new_activity_set = set()
        for seed in activity_set:
            neis = out_list[seed]
            for nei in neis:
                if nei not in all_active:
                    a_nei_w = [weight[node][nei] for node in in_list[nei] if node in all_active]
                    if sum(a_nei_w) > G[nei]: # active
                        new_activity_set.add(nei)
        count += len(new_activity_set)
        activity_set = new_activity_set
    return count


def random_threshold():
    global node_list
    G = dict()
    random.shuffle(node_list)
    for i in node_list:
        G[i] = random.random()
    return G


def ISE_LTexpected(activity_set, out_list, weight, in_list):
    """
    G: (node, threshold)
    """
    loc_sum = 0
    N = 1000
    for _ in range(N):
        if time.time()>end:
            return 0
        # print(time.time() - begin)
        G = random_threshold()
        one_sample = ISE_LT(activity_set, out_list, weight, G, in_list)
        loc_sum += one_sample
    return loc_sum / N


def ISE_ICexpected(activity_set, neighbor, weight):
    loc_sum = 0
    N = 1000
    for _ in range(N):
        if time.time()>end:
            return 0
        # print(time.time() - begin)
        one_sample = ISE_IC(activity_set, neighbor, weight)
        loc_sum += one_sample
    return loc_sum / N


def build_graph(edges):
    """

    :return: 邻接列表， numpy.ndarray
    """
    global node_list
    in_ad_list = dict()
    out_ad_list = dict()
    weight_m = dict()
    node_list = set()
    for edge in edges:
        out_ad_list[edge[0]] = list()
        out_ad_list[edge[1]] = list()
        in_ad_list[edge[1]]  = list()
        in_ad_list[edge[0]]  = list()
        weight_m[edge[0]] = {}
        node_list.add(edge[0])
        node_list.add(edge[1])
    node_list = list(node_list)
    for edge in edges:
        out_ad_list[edge[0]].append(edge[1])
        in_ad_list[edge[1]].append(edge[0])
        weight_m[edge[0]][edge[1]] = edge[2]
    return in_ad_list, out_ad_list, weight_m


def CELF_IC(k, out_list, weight):
    """
    # TODO 先不做优化， 后期再优化
    k: seed_num
    Q 所有的增删操作都应该heapq
    G: contain the state of nodes
       edges
    V: vertexes are tuples
    """
    # step 1
    global node_list
    S = list()
    Q = list()
    V = dict()
    for i in node_list:
        V[i] = Node(i)
    # step 2
    for key in V:
        u = V[key]
        if time.time() >end:
            return None, None
        u.mg1 = ISE_ICexpected({u.index}, out_list, weight)
        u.t_mg1 = u.mg1
        # step 4
        heapq.heappush(Q, u)
    # step 5
    while len(S) < k:
        if time.time() > end:
            return None, None
        u = heapq.heappop(Q)  # step 6
        if u.flag == len(S):  # u.flag == |S|  step 7
            # step 8
            # TODO debug
            # print("append", u.index, u.t_mg1, u.mg1)
            S.append(u.index)
            continue
        else:
            # step 13
            tmp = copy.deepcopy(S)
            tmp.append(u.index)
            u.t_mg1 = ISE_ICexpected(tmp, out_list, weight)
            u.mg1 = u.t_mg1 - V[S[-1]].t_mg1
            # TODO debug
            # print('refresh', u.index, u.t_mg1, u.mg1, V[S[-1]].t_mg1)
        # step 14
        u.flag = len(S)
        heapq.heappush(Q, u)
    return S, V[S[-1]].t_mg1   # return marginal gain


def CELF_LT(k, out_list, weight, in_list):
    """
        # TODO 先不做优化， 后期再优化
        k: seed_num
        Q 所有的增删操作都应该heapq
        G: contain the state of nodes
           edges
        V: vertexes are objects
        """
    global node_list
    # step 1
    S = list()
    Q = list()
    V = dict()
    for i in node_list:
        V[i] = Node(i)
    # step 2
    for key in V:
        u = V[key]
        if time.time() > end:
            return None, None
        u.mg1 = ISE_LTexpected({u.index}, out_list, weight, in_list)
        u.t_mg1 = u.mg1
        # step 4
        heapq.heappush(Q, u)
    # step 5
    while len(S) < k:
        if time.time() > end:
            return None, None
        u = heapq.heappop(Q)  # step 6
        if u.flag == len(S):  # u.flag == |S|  step 7
            # step 8
            # TODO debug
            # print("append", u.index, u.t_mg1, u.mg1)
            S.append(u.index)
            continue
        else:
            # step 13
            tmp = copy.deepcopy(S)
            tmp.append(u.index)
            u.t_mg1 = ISE_LTexpected(tmp, out_list, weight, in_list)
            u.mg1 = u.t_mg1 - V[S[-1]].t_mg1
            # TODO debug
            # print('refresh', u.index, u.t_mg1, u.mg1, V[S[-1]].t_mg1)
        # step 14
        u.flag = len(S)
        heapq.heappush(Q, u)
    return S, V[S[-1]].t_mg1  # return marginal gain


if __name__ == '__main__':
    social_network, seed_size, model, time_limit = read_args()
    end = begin + time_limit - 2
    seed_size = int(seed_size)
    node_num, edge_num, edges = read_graph(social_network)
    in_list, out_list, wm = build_graph(edges)
    d_seed_set = degree_discount_IC(seed_size, out_list, wm)
    ic_seed_set, count = CELF_IC(seed_size, out_list, wm)
    lt_seed_set = None
    if model != 'IC' and time.time() - begin + 10 < time_limit:
        lt_seed_set, count = CELF_LT(seed_size, out_list, wm, in_list)
    seed_set = lt_seed_set or ic_seed_set or d_seed_set
    # print(lt_seed_set)
    # print(ic_seed_set)
    # print(d_seed_set)
    for i in seed_set:
        print(i)
    # print(time.time() - begin)