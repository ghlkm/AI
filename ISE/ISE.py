"""
1. 读参数
2. 读图
3.
"""
import sys
import numpy as np
import copy
import time


def read_args():
    social_network = sys.argv[2]
    seed_set = sys.argv[4]
    model = sys.argv[6]
    time_limit = sys.argv[8]
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


def read_seed(fileName):
    seed_set = set()
    with open(fileName, 'r') as f:
        for line in f:
            seed_set.add(int(line.strip()))
    return seed_set


import random
def ISE_IC(activity_set, neighbor, weight):
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
            neis = neighbor[seed]
            for nei in neis:
                if nei not in all_active:
                    if random.random() < weight[seed][nei]:  # active
                        new_activity_set.add(nei)
        count += len(new_activity_set)
        activity_set = new_activity_set
    return count


def ISE_LT(a_set, neighbor, weight, G, in_list):
    """
    G: (node, threshold)
    """
    activity_set = copy.deepcopy(a_set)
    all_active = set()
    for node in G:
        if node[1] == 0:
            activity_set.add(node[0])
    count = len(activity_set)
    while activity_set:
        for node in activity_set:
            all_active.add(node)
        new_activity_set = set()
        for seed in activity_set:
            neis = neighbor[seed]
            for nei in neis:
                if nei not in all_active:
                    a_nei_w = [weight[node][nei] for node in in_list[nei] if node in all_active]
                    if sum(a_nei_w) > G[nei][1]: # active
                        new_activity_set.add(nei)
        count += len(new_activity_set)
        activity_set = new_activity_set
    return count


def random_threshold(node_num):
    node_num += 1
    G = np.ndarray(dtype=tuple, shape=(1, node_num))
    G = G[0]
    for i in range(node_num):
        G[i] = (i, random.random())
    return G


def ISE_LTexpected(activity_set, out_list, weight, in_list):
    """
    G: (node, threshold)
    """
    loc_sum = 0
    N = 1000
    j = 0
    for i in range(N):
        G = random_threshold(len(out_list) - 1)
        one_sample = ISE_LT(activity_set, out_list, weight, G, in_list)
        loc_sum += one_sample
        if time.time() < end:
            j = i
            break
    return loc_sum / (j+1)


def ISE_ICexpected(activity_set, neighbor, weight):
    loc_sum = 0
    N = 1000
    j = 0
    for i in range(N):
        one_sample = ISE_IC(activity_set, neighbor, weight)
        loc_sum += one_sample
        if time.time() < end:
            j = i
            break
    return loc_sum / (j+1)


def build_graph(edges, node_num):
    """

    :return: 邻接列表， numpy.ndarray
    """
    in_ad_list = np.ndarray(dtype=list, shape=(1, node_num+1))
    in_ad_list = in_ad_list[0]
    out_ad_list = np.ndarray(dtype=list, shape=(1, node_num + 1))
    out_ad_list = out_ad_list[0]
    weight_m = np.zeros(shape=(node_num+1, node_num+1))
    for i in range(len(in_ad_list)):
        in_ad_list[i] = list()
        out_ad_list[i] = list()
    for edge in edges:
        out_ad_list[edge[0]].append(edge[1])
        in_ad_list[edge[1]].append(edge[0])
        weight_m[edge[0], edge[1]] = edge[2]
    return in_ad_list, out_ad_list, weight_m


if __name__ == '__main__':
    begin = time.time()
    social_network, seed_file, model, time_limit = read_args()
    end = begin + time_limit - 2
    node_num, edge_num, edges = read_graph(social_network)
    seed_set = read_seed(seed_file)
    in_list, out_list, wm = build_graph(edges, node_num)
    if model == 'IC':
        count = ISE_ICexpected(seed_set, out_list, wm)
    else:
        count = ISE_LTexpected(seed_set, out_list, wm, in_list)
    print(count)
    # print(time.time() - begin)
    # print(node_num)
    # print(edge_num)
    # print(edges[0])
    # print(seed_set)