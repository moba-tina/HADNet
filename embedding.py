import torch
import torch.nn as nn
import dgl
from dgl.nn import GraphConv
import numpy as np
import copy
import networkx as nx
import pandas as pd
# from ganer_flow_dwt import FlowVAE
# from ganer_energy import EnergyModel
from markov import generate_counterfactual_data_and_save
from sklearn.model_selection import train_test_split

# def conformal_projection(v1, v2):
#     norm_v1 = torch.norm(v1)
#     norm_v2 = torch.norm(v2)
#     factor = 2 * torch.dot(v1, v2) / (norm_v1**2 + norm_v2**2)
#     return factor * v1

class EvolveGCN(nn.Module):
    def __init__(self, in_feats, out_feats, num_time_steps):
        super(EvolveGCN, self).__init__()
        self.num_time_steps = num_time_steps
        self.gcn_layers = nn.ModuleList(
            [GraphConv(in_feats, out_feats, allow_zero_in_degree=True) for _ in range(num_time_steps)])

    def forward(self, g_list, features):
        diff_features = []
        for i in range(self.num_time_steps):
            g = g_list[i]
            if i < 1:
                features[i] = self.gcn_layers[i](g, features[i])
                diff_features.append(features[i])
            else:
                features[i][:g_list[i - 1].number_of_nodes()] = features[i - 1]
                features[i] = self.gcn_layers[i](g, features[i])
                diff = features[i].clone()
                for j in np.arange(g_list[i - 1].number_of_nodes()):
                    diff[j] = diff[j] - features[i - 1][j]
                diff_features.append(diff)
        return features, diff_features

def getgraph(first_list_left_node, first_list_right_node):
    node = copy.deepcopy(first_list_left_node)
    node.extend(first_list_right_node)
    all_node = list(set(node))
    all_edges = []
    for i in np.arange(len(first_list_left_node)):
        all_edges.append([int(first_list_left_node[i]), int(first_list_right_node[i])])
    graph = nx.Graph()
    for nodei in all_node:
        graph.add_node(nodei)
    for edge in all_edges:
        graph.add_edge(edge[0], edge[1])
    dglgraph = dgl.from_networkx(graph)
    return dglgraph

def select_Node_emebdding(G, node_features, first_node, first_ano, influence_dict=None):
    ano = []
    nodeset = []
    featureset = []
    influenceset = []
    first_node = list(set(first_node))
    for node in G.nodes():
        features = node_features[node]
        if node in first_node:
            nodeset.append(node)
            featureset.append(features.tolist())
            if node in first_ano:
                ano.append(1)
            else:
                ano.append(0)
            if influence_dict is not None:
                inf = influence_dict.get(node, {'degree':0, 'betweenness':0, 'closeness':0, 'pagerank':0})
                influenceset.append([inf['degree'], inf['betweenness'], inf['closeness'], inf['pagerank']])
    if influence_dict is not None:
        return nodeset, featureset, ano, influenceset
    else:
        return nodeset, featureset, ano

def original_ano(first_ano_label, first_featureset):
    zero_indices = np.where(first_ano_label == 1)[0]
    first_original_ano_features = first_featureset[zero_indices]
    return first_original_ano_features

def count_difference(lst):
    count_0 = np.sum(lst == 0)
    count_1 = np.sum(lst == 1)
    difference = abs(count_0 - count_1)
    return difference

if __name__ == '__main__':

    in_feats = 128
    out_feats = 128
    encoding_dim_autoE = 32
    n = 4
    # autoencoder = FlowVAE(in_feats, encoding_dim_autoE)
    dataname = "reddit"
    data = pd.read_csv("data/" + dataname + ".csv", header=None)
    start = min(data[2])
    end = max(data[2])
    interval = (end - start) / n
    time_points = [start + (i + 1) * interval for i in range(n)]
    time_ids = [i for i in np.arange(len(time_points))]

    for time_id in time_ids:
        new_column_name = f'DF_Column_{time_id}'
        data[new_column_name] = data[2].apply(lambda x: 1 if x < time_points[time_id] else 0)
        if time_id == 0:
            new_snapshot_name = f'DF_snapshot_{time_id}'
            data[new_snapshot_name] = data[2].apply(lambda x: 1 if x <= time_points[time_id] else 0)
        else:
            new_snapshot_name = f'DF_snapshot_{time_id}'
            data[new_snapshot_name] = data[2].apply(lambda x: 1 if (
                    time_points[time_id] >= x > time_points[time_id - 1]) else 0)
        new_ano_name = f'DF_ano_{time_id}'
        data[new_ano_name] = data.apply(lambda row: row[1] if (row[2] <= time_points[time_id] and row[3] == 1) else -1,
                                        axis=1)
    pd.set_option('display.max_columns', None)
    left_node_lists = {time_id: [] for time_id in time_ids}
    right_node_lists = {time_id: [] for time_id in time_ids}
    ano_node_lists = {time_id: [] for time_id in time_ids}
    snapshot_lists = {time_id: [] for time_id in time_ids}
    graph_list = []
    for time_id in time_ids:
        condition = data[f'DF_Column_{time_id}'] == 1
        condition_ano = data[f'DF_ano_{time_id}'] != -1
        condition_snap = data[f'DF_snapshot_{time_id}'] == 1
        left_node_lists[time_id] = data.loc[condition, 0].tolist()
        right_node_lists[time_id] = data.loc[condition, 1].tolist()
        ano_node_lists[time_id] = data.loc[condition_ano, f'DF_ano_{time_id}'].tolist()
        snapshot_lists[time_id] = data.loc[condition_snap, 0].tolist() + data.loc[condition_snap, 1].tolist()
        graph = getgraph(left_node_lists[time_id], right_node_lists[time_id])
        graph_list.append(graph)

    # 计算每个快照的影响力标签
    influence_labels = {time_id: {} for time_id in time_ids}
    for time_id in time_ids:
        nx_graph = graph_list[time_id].to_networkx().to_undirected()
        degree = nx.degree_centrality(nx_graph)
        betweenness = nx.betweenness_centrality(nx_graph)
        closeness = nx.closeness_centrality(nx_graph)
        pagerank = nx.pagerank(nx_graph)
        for node in nx_graph.nodes():
            influence_labels[time_id][node] = {
                'degree': degree.get(node, 0),
                'betweenness': betweenness.get(node, 0),
                'closeness': closeness.get(node, 0),
                'pagerank': pagerank.get(node, 0)
            }

    num_nodes = [g.number_of_nodes() for g in graph_list]
    features = [torch.randn(num, in_feats) for num in num_nodes]
    model = EvolveGCN(in_feats, out_feats, n)
    output_features, diff_features = model(graph_list, features)
    node_set_lists = {time_id: [] for time_id in time_ids}
    feature_set_lists = {time_id: [] for time_id in time_ids}
    ano_label_lists = {time_id: [] for time_id in time_ids}
    generate_label_lists = {time_id: [] for time_id in time_ids}
    timestamp_label_lists = {time_id: [] for time_id in time_ids}
    original_ano_features_lists = {time_id: [] for time_id in time_ids}
    num_samples_lists = {time_id: [] for time_id in time_ids}
    generated_features_lists = {time_id: [] for time_id in time_ids}
    generated_ano_list = {time_id: [] for time_id in time_ids}
    generated_timestamp_lists = {time_id: [] for time_id in time_ids}
    feature_combined_lists = {time_id: [] for time_id in time_ids}
    generated_label_lists = {time_id: [] for time_id in time_ids}
    ano_all_label_lists = {time_id: [] for time_id in time_ids}
    timestamp_all_label_lists = {time_id: [] for time_id in time_ids}
    generated_all_label_lists = {time_id: [] for time_id in time_ids}
    timestamp_add_feature_lists = {time_id: [] for time_id in time_ids}
    generate_add_timestamp_add_feature_lists = {time_id: [] for time_id in time_ids}
    influence_set_lists = {time_id: [] for time_id in time_ids}

    for time_id in time_ids:
        nodes, featureset, anos, influences = select_Node_emebdding(
            graph_list[time_id], diff_features[time_id], snapshot_lists[time_id], ano_node_lists[time_id], influence_labels[time_id])
        node_set_lists[time_id] = nodes
        feature_set_lists[time_id] = np.array(featureset)
        ano_label_lists[time_id] = np.array(anos)
        influence_set_lists[time_id] = np.array(influences)
        generate_label_lists[time_id] = np.array([0 for i in nodes])
        timestamp_label_lists[time_id] = np.array([time_id for i in nodes])
        if time_id != time_ids[-1]:
            original_ano_features_lists[time_id] = original_ano(ano_label_lists[time_id], feature_set_lists[time_id])
            if original_ano_features_lists[time_id].size==0:
                feature_combined_lists[time_id] = feature_set_lists[time_id]
                ano_all_label_lists[time_id] = np.array(ano_label_lists[time_id])
                timestamp_all_label_lists[time_id] = np.array(timestamp_label_lists[time_id])
                generated_all_label_lists[time_id] = np.array(generate_label_lists[time_id])
                influence_set_lists[time_id] = influence_set_lists[time_id]
            else:
                num_samples_lists[time_id] = count_difference(ano_label_lists[time_id])
                # autoencoder.train(original_ano_features_lists[time_id], num_epochs=100, learning_rate=0.001)
                generated_features_lists[time_id] = generate_counterfactual_data_and_save(original_ano_features_lists[time_id],num_samples=num_samples_lists[time_id])
                feature_combined_lists[time_id] = np.vstack(
                    (feature_set_lists[time_id], generated_features_lists[time_id]))
                generated_ano_list[time_id] = np.array([1] * generated_features_lists[time_id].shape[0])
                generated_label_lists[time_id] = np.array([1] * generated_features_lists[time_id].shape[0])
                generated_timestamp_lists[time_id] = np.array([time_id] * generated_features_lists[time_id].shape[0])
                ano_all_label_lists[time_id] = np.concatenate((ano_label_lists[time_id], generated_ano_list[time_id]))
                timestamp_all_label_lists[time_id] = np.concatenate(
                    (timestamp_label_lists[time_id], generated_timestamp_lists[time_id]))
                generated_all_label_lists[time_id] = np.concatenate(
                    (generate_label_lists[time_id], generated_label_lists[time_id]))
                influence_set_lists[time_id] = np.vstack([
                    influence_set_lists[time_id],
                    np.zeros((generated_features_lists[time_id].shape[0], influence_set_lists[time_id].shape[1]))
                ])
        else:
            feature_combined_lists[time_id] = feature_set_lists[time_id]
            ano_all_label_lists[time_id] = np.array(ano_label_lists[time_id])
            timestamp_all_label_lists[time_id] = np.array(timestamp_label_lists[time_id])
            generated_all_label_lists[time_id] = np.array(generate_label_lists[time_id])
            influence_set_lists[time_id] = influence_set_lists[time_id]
        timestamp_all_label_lists[time_id] = timestamp_all_label_lists[time_id][:, np.newaxis]
        generated_all_label_lists[time_id] = generated_all_label_lists[time_id][:, np.newaxis]
        timestamp_add_feature_lists[time_id] = np.concatenate(
            [timestamp_all_label_lists[time_id], feature_combined_lists[time_id]], axis=1)
        generate_add_timestamp_add_feature_lists[time_id] = np.concatenate(
            [generated_all_label_lists[time_id], np.array(timestamp_add_feature_lists[time_id])], axis=1)
        generate_add_timestamp_add_feature_lists[time_id] = np.concatenate(
            [influence_set_lists[time_id], generate_add_timestamp_add_feature_lists[time_id]], axis=1)

    my_list = []
    for time_id, feature in generate_add_timestamp_add_feature_lists.items():
        my_list.append(feature)
    train_feature = np.vstack(my_list[:-1])
    my_ano_all_label_list = []
    for time_id, feature in ano_all_label_lists.items():
        feature=feature[:, np.newaxis]
        my_ano_all_label_list.append(feature)
    train_ano_label = np.vstack(my_ano_all_label_list[:-1])
    last_snapshot_feature = generate_add_timestamp_add_feature_lists[time_ids[-1]]
    last_snapshot_ano_label = ano_all_label_lists[time_ids[-1]]
    train_last_infor, test_infor, train_last_label, test_label = train_test_split(
        last_snapshot_feature, last_snapshot_ano_label,
        test_size=1 - 0.1)
    train_feature = np.concatenate([train_feature, train_last_infor], axis=0)
    train_last_label=train_last_label[:,np.newaxis]
    train_ano_label = np.concatenate([train_ano_label, train_last_label], axis=0)
    train_infor, valid_infor, train_label, valid_label = train_test_split(train_feature, train_ano_label,
                                                                        test_size=0.2)
    with open("data_hmm/"+dataname + 'train'+str(n)+'.txt', 'w') as f:
        for i in np.arange(len(train_label)):
            label = str(int(train_label[i]))
            embedding = ','.join([str(x) for x in train_infor[i]])
            f.write(f"{label},{embedding}\n")
    with open("data_hmm/"+dataname + 'valid'+str(n)+'.txt', 'w') as f:
        for i in np.arange(len(valid_label)):
            label = str(int(valid_label[i]))
            embedding = ','.join([str(x) for x in valid_infor[i]])
            f.write(f"{label},{embedding}\n")
    with open("data_hmm/"+dataname + 'test'+str(n)+'.txt', 'w') as f:
        for i in np.arange(len(test_label)):
            label = str(int(test_label[i]))
            embedding = ','.join([str(x) for x in test_infor[i]])
            f.write(f"{label},{embedding}\n")