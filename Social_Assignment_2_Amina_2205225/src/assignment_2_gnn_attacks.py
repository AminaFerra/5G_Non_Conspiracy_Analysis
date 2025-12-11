import networkx as nx
import pandas as pd
import numpy as np
import random
import community as community_louvain 
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, accuracy_score
import warnings
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GCNConv
from matplotlib.patches import Patch
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

FEATURE_COLUMNS = ['degree', 'clustering_coeff', 'betweenness_centrality', 'eigenvector_centrality', 'community_id']
METRICS_FOLDER = 'results/metrics'
VIS_FOLDER = 'results/visualizations'
os.makedirs(METRICS_FOLDER, exist_ok=True)
os.makedirs(VIS_FOLDER, exist_ok=True)
BOT_COLOR = '#ff8a5d'      
HUMAN_COLOR = '#4da6ff'    
TARGET_COLOR = 'black'     
ATTACK_HIGHLIGHT_COLOR = '#ffe600' 

# Data Loading and Graph Building
def load_facebook_graph(file_path="Data/facebook_combined.txt"):
    try:
        graph = nx.read_edgelist(file_path, nodetype=int)
        if graph.is_directed():
            graph = graph.to_undirected()
        if not nx.is_connected(graph):
            graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
        print(f"Graph loaded successfully (LCC used): {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")
        return graph
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Returning an empty graph.")
        return nx.Graph() 

# Calculate and Save Descriptive Stats
def calculate_and_save_stats(df_features, feature_cols, scenario_name):
    stats_df = df_features[feature_cols].agg(['mean', 'min', 'max']).T
    descriptive_stats = df_features.describe().T
    print(f"\nDescriptive Statistics for {scenario_name} Graph Metrics:")
    print(stats_df.round(6).to_markdown(numalign="left", stralign="left"))
    save_dataframe_to_csv(descriptive_stats.round(6), METRICS_FOLDER, f'03_{scenario_name.lower().replace(" ", "_")}_features_stats.csv')
    return stats_df

# Compute Graph Metrics and Synthetic Labels
def compute_graph_features(graph):
    print("Computing 5 graph metrics (Degree, Clustering, Betweenness, Eigenvector, Community)...")
    degree = dict(graph.degree())
    clustering = nx.clustering(graph)
    k_nodes = min(graph.number_of_nodes(), 500)
    betweenness = nx.betweenness_centrality(graph, k=k_nodes, seed=42, normalized=True) 
    try:
        eigenvector = nx.eigenvector_centrality(graph, max_iter=200, tol=1e-05)
    except nx.PowerIterationFailedConvergence:
        print("Warning: Eigenvector Centrality failed to converge. Using fallback uniform values.")
        eigenvector = {node: 1.0 / graph.number_of_nodes() for node in graph.nodes()}
    partition = community_louvain.best_partition(graph, random_state=42)
    features_df = pd.DataFrame({
        'degree': pd.Series(degree),
        'clustering_coeff': pd.Series(clustering),
        'betweenness_centrality': pd.Series(betweenness),
        'eigenvector_centrality': pd.Series(eigenvector), 
        'community_id': pd.Series(partition),
    }).fillna(0)
    all_nodes = list(graph.nodes())
    features_df = features_df.reindex(all_nodes, fill_value=0)
    print(f"\nMetric Output Sample (Node IDs are indices):\n{features_df.head().round(6)}\n")
    return features_df

def create_synthetic_labels(graph, features_df, bot_percentage=0.05):
    N = graph.number_of_nodes()
    bot_count = max(1, int(bot_percentage * N))
    sorted_by_degree = features_df['degree'].sort_values()
    low_degree_bots = sorted_by_degree.head(bot_count // 2).index.tolist()
    high_degree_bots = sorted_by_degree.tail(bot_count - bot_count // 2).index.tolist()
    bot_nodes = set(low_degree_bots + high_degree_bots)
    labels = pd.Series(0, index=graph.nodes())
    labels.loc[list(bot_nodes)] = 1
    print(f"Synthetic labels created: {len(bot_nodes)} bots (1). Total nodes: {N}.")
    return features_df.join(labels.rename('label')), bot_nodes

#Functions
def get_nx_graph_from_pyg_data(data, idx_to_node):
    new_graph = nx.Graph()
    unique_pyg_edges = data.edge_index.t().unique(dim=0).tolist()   
    nx_edges = []
    seen_edges = set()
    for u_idx, v_idx in unique_pyg_edges:
        u, v = idx_to_node[u_idx], idx_to_node[v_idx]
        if u != v:
            #undirected graph
            edge_tuple = tuple(sorted((u, v)))
            if edge_tuple not in seen_edges:
                nx_edges.append((u, v))
                seen_edges.add(edge_tuple)
    new_graph.add_edges_from(nx_edges)
    return new_graph

def map_indices_to_nodes(indices, idx_to_node):
    return [idx_to_node[i] for i in indices]

def save_dataframe_to_csv(df, folder, filename):
    full_path = os.path.join(folder, filename)
    df.to_csv(full_path, index=True)
    print(f"Saved output to: {full_path}")

# GNN Model Definitions
class GCNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Training and Evaluation Functions
def train_gnn(model, data, epochs=100, model_name=""):
    train_labels = data.y[data.train_mask]
    bot_count = (train_labels == 1).sum().item()
    human_count = (train_labels == 0).sum().item()
    class_weight = human_count / bot_count if bot_count > 0 else 1.0
    weight_tensor = torch.tensor([1.0, class_weight], device=data.x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    print(f"Training {model_name}... (Bot Class Weight: {class_weight:.2f}, Epochs: {epochs})")
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask], weight=weight_tensor)
        loss.backward()
        optimizer.step()
    # Calculate final training metrics
    train_metrics, _ = evaluate_gnn(model, data, data.train_mask, "Train")
    return train_metrics['Accuracy'], train_metrics['F1-Score (Bot)']

def evaluate_gnn(model, data, mask, name=""):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)        
        y_true = data.y[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            # Bot (Label 1) Metrics
            'F1-Score (Bot)': f1_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0),
            'Recall (Bot)': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
            # Human (Label 0) Metrics
            'F1-Score (Human)': f1_score(y_true, y_pred, average='binary', pos_label=0, zero_division=0),
            'Recall (Human)': recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        }
    print(f"--- {name} Performance ({mask.sum().item()} nodes) ---")
    print(f"Accuracy: {metrics['Accuracy']:.4f}, F1 (Bot): {metrics['F1-Score (Bot)']:.4f}, Recall (Bot): {metrics['Recall (Bot)']:.4f}")
    print(f"                         F1 (Human): {metrics['F1-Score (Human)']:.4f}, Recall (Human): {metrics['Recall (Human)']:.4f}")
    return metrics, pred

# Attack Implementation
def structural_evasion_attack_nx(original_graph, target_indices_pg, data_df_baseline, idx_to_node, test_human_indices_pg, max_changes=3):
    evasion_graph = original_graph.copy()
    human_degree_mean = data_df_baseline[data_df_baseline['label'] == 0]['degree'].mean()
    target_nodes_nx = map_indices_to_nodes(target_indices_pg, idx_to_node)
    test_human_nodes_nx = map_indices_to_nodes(test_human_indices_pg, idx_to_node) 
    total_links_added = 0
    for bot_node in target_nodes_nx:
        current_degree = evasion_graph.degree(bot_node)
        degree_diff = int(human_degree_mean - current_degree)
        if degree_diff > 0: 
            potential_neighbors = [
                n for n in evasion_graph.nodes() 
                if n in test_human_nodes_nx and not evasion_graph.has_edge(bot_node, n)
            ]
            num_changes = min(degree_diff, max_changes, len(potential_neighbors))
            if num_changes > 0:
                edges_to_add = random.sample(potential_neighbors, num_changes)
                evasion_graph.add_edges_from([(bot_node, n) for n in edges_to_add])
                total_links_added += num_changes
    print(f"Structural Evasion Attack: Added {total_links_added} links to {len(target_nodes_nx)} bots.")
    return evasion_graph

def graph_poisoning_attack_nx(original_graph, train_bot_indices_pg, train_human_indices_pg, idx_to_node, num_edges_to_add=50):
    poison_graph = original_graph.copy()
    train_bots_nx = map_indices_to_nodes(train_bot_indices_pg, idx_to_node)
    train_humans_nx = map_indices_to_nodes(train_human_indices_pg, idx_to_node)
    links_added = 0
    for _ in range(num_edges_to_add):
        if not train_bots_nx or not train_humans_nx:
            break
        bot = random.choice(train_bots_nx)
        human = random.choice(train_humans_nx)
        if not poison_graph.has_edge(bot, human):
            poison_graph.add_edge(bot, human)
            links_added += 1
    print(f"Graph Poisoning Attack: Added {links_added} links between training bots and humans.")
    return poison_graph

# Enhanced Visualization Function
def find_best_viz_target(indices_pg, data_pg, min_degree=5):
    for idx in indices_pg:
        adj_indices = data_pg.edge_index[1][data_pg.edge_index[0] == idx]
        if adj_indices.shape[0] >= min_degree:
            return idx
    return indices_pg[0] 

def visualize_attack_comparison_full(graph_baseline, data_evasion, data_poison, all_bot_nodes_nx, 
                                     idx_to_node, target_node_evasion, target_node_poison, 
                                     evasion_targets_pg, train_bot_indices_pg):
    fig_full, ax_full = plt.subplots(figsize=(10, 8))
    ax_full.set_title('Baseline Graph: Network Structure Visualization', fontsize=16)
    try:
        center_node = max(graph_baseline.degree(), key=lambda item: item[1])[0]
        full_subgraph = nx.ego_graph(graph_baseline, center_node, radius=2) 
    except:
        full_subgraph = graph_baseline.subgraph(list(graph_baseline.nodes)[:min(graph_baseline.number_of_nodes(), 400)]) 
    full_pos = nx.spring_layout(full_subgraph, seed=42, iterations=50)
    full_node_colors = [BOT_COLOR if node in all_bot_nodes_nx else HUMAN_COLOR for node in full_subgraph.nodes()] 
    nx.draw_networkx_nodes(full_subgraph, full_pos, node_size=50, node_color=full_node_colors, ax=ax_full, edgecolors='gray', linewidths=0.5)
    nx.draw_networkx_edges(full_subgraph, full_pos, width=0.5, alpha=0.3, edge_color='gray', ax=ax_full)
    full_legend_elements = [
        Patch(facecolor=BOT_COLOR, label=f'Bots ({sum(1 for n in full_subgraph.nodes() if n in all_bot_nodes_nx)})'),
        Patch(facecolor=HUMAN_COLOR, label=f'Humans ({sum(1 for n in full_subgraph.nodes() if n not in all_bot_nodes_nx)})')
    ]
    ax_full.legend(handles=full_legend_elements, loc='upper right', title="Legend")
    ax_full.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_FOLDER, '01_baseline_full_graph.png'))
    plt.close(fig_full)
    print(f"Baseline full graph visualization saved to: {os.path.join(VIS_FOLDER, '01_baseline_full_graph.png')}")
    # 2.Network Comparison Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Ego-Network Visualization of Bot Nodes Before and After Attacks', fontsize=16)
    graph_evasion = get_nx_graph_from_pyg_data(data_evasion, idx_to_node)
    graph_poison = get_nx_graph_from_pyg_data(data_poison, idx_to_node)
    evasion_targets_nx = map_indices_to_nodes(evasion_targets_pg, idx_to_node)
    poison_targets_nx = map_indices_to_nodes(train_bot_indices_pg, idx_to_node)
    # Plot 1: Baseline (Original)
    visualize_egonet_single(
        graph_baseline, target_node_evasion, axes[0], 
        f'1. Baseline Structure ', all_bot_nodes_nx,
        highlight_color=TARGET_COLOR,
        attack_highlight_color=ATTACK_HIGHLIGHT_COLOR 
    )
    # Plot 2: Structural Evasion
    visualize_egonet_single(
        graph_evasion, target_node_evasion, axes[1], 
        '2. Structural Evasion ', all_bot_nodes_nx, 
        highlight_color=TARGET_COLOR,
        attacked_nodes=evasion_targets_nx,
        attack_highlight_color=ATTACK_HIGHLIGHT_COLOR 
    )
    # Plot 3: Graph Poisoning
    visualize_egonet_single(
        graph_poison, target_node_poison, axes[2], 
        f'3. Graph Poisoning ', all_bot_nodes_nx,
        highlight_color=TARGET_COLOR,
        attacked_nodes=poison_targets_nx,
        attack_highlight_color=ATTACK_HIGHLIGHT_COLOR 
    )
    legend_elements = [
        Patch(facecolor=BOT_COLOR, label='Bot Node'),
        Patch(facecolor=HUMAN_COLOR, label='Human Node'),
        Patch(facecolor=TARGET_COLOR, label='Target Bot (Center)'),
        Patch(facecolor=ATTACK_HIGHLIGHT_COLOR, label='Attacked Neighbor/Target')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=4)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(VIS_FOLDER, 'enhanced_attack_comparison.png'))
    plt.close(fig)
    print(f"Ego-Network comparison saved to: {os.path.join(VIS_FOLDER, 'enhanced_attack_comparison.png')}")

def visualize_egonet_single(graph, target_node, ax, title, all_bot_nodes_nx, highlight_color, attacked_nodes=None, attack_highlight_color=ATTACK_HIGHLIGHT_COLOR):
    if target_node not in graph.nodes():
        ax.set_title(f"{title}\nNode {target_node} not found.")
        ax.axis('off')
        return
    ego_net = nx.ego_graph(graph, target_node, radius=1)
    node_colors = []
    node_sizes = []
    for node in ego_net.nodes():
        size = 200
        color = HUMAN_COLOR  # Default to Human Color
        if node in all_bot_nodes_nx:
            color = BOT_COLOR  # Set to Bot Color
        # Check if the neighbor is one of the nodes involved in the attack
        is_attacked_neighbor = attacked_nodes is not None and node != target_node and node in attacked_nodes
        if is_attacked_neighbor:
            color = attack_highlight_color 
            size = 250
        if node == target_node:
            color = highlight_color
            size = 350
        node_colors.append(color)
        node_sizes.append(size)
    pos = nx.spring_layout(ego_net, seed=42, iterations=50) 
    # Draw nodes and edges
    nx.draw_networkx_nodes(ego_net, pos, node_size=node_sizes, node_color=node_colors, ax=ax, edgecolors='gray', linewidths=0.5)
    nx.draw_networkx_edges(ego_net, pos, width=1.0, alpha=0.5, ax=ax)
    ax.set_title(title, fontsize=12)
    ax.axis('off')
# MAIN
if __name__ == "__main__":
    # --- Setup ---
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    # --- Initial Data Processing (Baseline) ---
    graph_baseline = load_facebook_graph()
    if graph_baseline.number_of_nodes() == 0:
        print("Cannot proceed: Graph failed to load.")
        exit()
    features_df_baseline = compute_graph_features(graph_baseline)
    data_df_baseline, all_bot_nodes_nx = create_synthetic_labels(graph_baseline, features_df_baseline)
    # Save Baseline Features and calculate stats
    features_df_baseline_only = data_df_baseline.drop(columns=['label'])
    save_dataframe_to_csv(features_df_baseline_only, METRICS_FOLDER, '01_baseline_features.csv')
    calculate_and_save_stats(features_df_baseline_only, FEATURE_COLUMNS, "Baseline")
    # --- PyG Data Preparation (Baseline) ---
    X_features = data_df_baseline[FEATURE_COLUMNS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    x = torch.tensor(X_scaled, dtype=torch.float)
    y = torch.tensor(data_df_baseline['label'].values, dtype=torch.long)
    node_list = data_df_baseline.index.tolist()
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    indexed_edges = [(node_to_idx[u], node_to_idx[v]) for u, v in graph_baseline.edges()]
    if len(indexed_edges) == 0:
        print("Warning: Baseline graph has no edges. Adding self-loops.")
        indexed_edges = [(i, i) for i in range(len(node_list))]
    edge_index = torch.tensor(indexed_edges, dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip([0])], dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    data_pg = Data(x=x, edge_index=edge_index, y=y)
    all_indices = np.arange(data_pg.num_nodes)
    train_indices, test_indices = train_test_split(all_indices, test_size=0.2, stratify=data_pg.y, random_state=42)
    train_mask = torch.zeros(data_pg.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data_pg.num_nodes, dtype=torch.bool)
    train_mask[train_indices] = True
    test_mask[test_indices] = True
    data_pg.train_mask = train_mask
    data_pg.test_mask = test_mask
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    test_bot_indices = [i for i in test_indices if data_pg.y[i] == 1]
    test_human_indices = [i for i in test_indices if data_pg.y[i] == 0] 
    train_bot_indices = [i for i in train_indices if data_pg.y[i] == 1]
    train_human_indices = [i for i in train_indices if data_pg.y[i] == 0]
    IN_CHANNELS = data_pg.num_node_features
    HIDDEN_CHANNELS = 16
    OUT_CHANNELS = 2
    all_metrics = []
    baseline_train_metrics = {}
    print("\n" + "="*70 + "\n1. BASELINE: TRAINING AND TESTING ON CLEAN DATA")
    
    # GraphSAGE Baseline
    gnn_model_sage_baseline = GraphSAGEModel(IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS)
    train_acc_sage, train_f1_sage = train_gnn(gnn_model_sage_baseline, data_pg, model_name="GraphSAGE Baseline")
    test_metrics_sage, _ = evaluate_gnn(gnn_model_sage_baseline, data_pg, data_pg.test_mask, "GraphSAGE Baseline (Test/Clean)")
    baseline_train_metrics['GraphSAGE'] = {'Train_Acc': train_acc_sage, 'Train_F1': train_f1_sage}
    all_metrics.append({**test_metrics_sage, 'Model': 'GraphSAGE', 'Scenario': 'Baseline', 'Train_Acc': train_acc_sage, 'Train_F1': train_f1_sage})
    
    # GCN Baseline
    gnn_model_gcn_baseline = GCNModel(IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS)
    train_acc_gcn, train_f1_gcn = train_gnn(gnn_model_gcn_baseline, data_pg, model_name="GCN Baseline")
    test_metrics_gcn, _ = evaluate_gnn(gnn_model_gcn_baseline, data_pg, data_pg.test_mask, "GCN Baseline (Test/Clean)")
    baseline_train_metrics['GCN'] = {'Train_Acc': train_acc_gcn, 'Train_F1': train_f1_gcn}
    all_metrics.append({**test_metrics_gcn, 'Model': 'GCN', 'Scenario': 'Baseline', 'Train_Acc': train_acc_gcn, 'Train_F1': train_f1_gcn})

    print("\n" + "="*70 + "\n2. EVASION ATTACK: EVALUATING BASELINE MODELS ON ATTACKED DATA")
    attack_targets_pg = test_bot_indices[:len(test_bot_indices) // 2] 
    evasion_graph = structural_evasion_attack_nx(
        graph_baseline, attack_targets_pg, data_df_baseline, 
        idx_to_node, test_human_indices, max_changes=3
    )

    features_evasion = compute_graph_features(evasion_graph) 
    features_df_evasion_only = features_evasion[FEATURE_COLUMNS].reindex(features_df_baseline.index, fill_value=0)
    save_dataframe_to_csv(features_df_evasion_only, METRICS_FOLDER, '04_evasion_features.csv')
    calculate_and_save_stats(features_df_evasion_only, FEATURE_COLUMNS, "Evasion Attack")
    X_evasion_features = features_df_evasion_only.values
    X_evasion_scaled = scaler.transform(X_evasion_features)
    x_evasion = torch.tensor(X_evasion_scaled, dtype=torch.float)
    evasion_indexed_edges = [(node_to_idx[u], node_to_idx[v]) for u, v in evasion_graph.edges()]
    
    if len(evasion_indexed_edges) == 0:
        print("Warning: Evasion graph has no edges. Adding self-loops.")
        evasion_indexed_edges = [(i, i) for i in range(len(node_list))]    
    evasion_edge_index = torch.tensor(evasion_indexed_edges, dtype=torch.long).t().contiguous()
    evasion_edge_index = torch.cat([evasion_edge_index, evasion_edge_index.flip([0])], dim=1)
    evasion_edge_index = torch.unique(evasion_edge_index, dim=1)
    # data_evasion contains clean features but the ATTACKED graph structure
    data_evasion = Data(x=x_evasion, edge_index=evasion_edge_index, y=data_pg.y, 
                         train_mask=data_pg.train_mask, test_mask=data_pg.test_mask)
    # Evaluate the BASELINE SAGE model on the EVASION data
    evasion_metrics_sage, _ = evaluate_gnn(gnn_model_sage_baseline, data_evasion, data_evasion.test_mask, "GraphSAGE (Evasion Attack Test)")
    all_metrics.append({**evasion_metrics_sage, 'Model': 'GraphSAGE', 'Scenario': 'Evasion Attack', 
                        'Train_Acc': baseline_train_metrics['GraphSAGE']['Train_Acc'], 
                        'Train_F1': baseline_train_metrics['GraphSAGE']['Train_F1']})
    
    # Evaluate the BASELINE GCN model on the EVASION data
    evasion_metrics_gcn, _ = evaluate_gnn(gnn_model_gcn_baseline, data_evasion, data_evasion.test_mask, "GCN (Evasion Attack Test)")
    all_metrics.append({**evasion_metrics_gcn, 'Model': 'GCN', 'Scenario': 'Evasion Attack', 
                        'Train_Acc': baseline_train_metrics['GCN']['Train_Acc'], 
                        'Train_F1': baseline_train_metrics['GCN']['Train_F1']})

    # Evaluate the EVASION model on CLEAN TEST DATA
    evasion_clean_metrics_sage, _ = evaluate_gnn(gnn_model_sage_baseline, data_pg, data_pg.test_mask, "GraphSAGE (Evasion Model/Clean Test)")
    all_metrics.append({**evasion_clean_metrics_sage, 'Model': 'GraphSAGE', 'Scenario': 'Evasion Attack (on Clean Data)', 
                        'Train_Acc': baseline_train_metrics['GraphSAGE']['Train_Acc'], 
                        'Train_F1': baseline_train_metrics['GraphSAGE']['Train_F1']})

    # Same for GCN
    evasion_clean_metrics_gcn, _ = evaluate_gnn(gnn_model_gcn_baseline, data_pg, data_pg.test_mask, "GCN (Evasion Model/Clean Test)")
    all_metrics.append({**evasion_clean_metrics_gcn, 'Model': 'GCN', 'Scenario': 'Evasion Attack (on Clean Data)', 
                        'Train_Acc': baseline_train_metrics['GCN']['Train_Acc'], 
                        'Train_F1': baseline_train_metrics['GCN']['Train_F1']})

    #3. GRAPH POISONING ATTACK (Train-Time)
    print("\n" + "="*70 + "\n3. POISONING ATTACK: TRAINING NEW MODELS ON CORRUPTED DATA")
    
    poison_graph = graph_poisoning_attack_nx(
        graph_baseline, train_bot_indices, train_human_indices, 
        idx_to_node, num_edges_to_add=50
    )
    features_poison = compute_graph_features(poison_graph) 
    features_df_poison_only = features_poison[FEATURE_COLUMNS].reindex(features_df_baseline.index, fill_value=0)
    save_dataframe_to_csv(features_df_poison_only, METRICS_FOLDER, '05_poisoning_features.csv')
    calculate_and_save_stats(features_df_poison_only, FEATURE_COLUMNS, "Poisoning Attack")
    X_poison_features = features_df_poison_only.values
    X_poison_scaled = scaler.transform(X_poison_features)
    x_poison = torch.tensor(X_poison_scaled, dtype=torch.float)
    poison_indexed_edges = [(node_to_idx[u], node_to_idx[v]) for u, v in poison_graph.edges()]
    if len(poison_indexed_edges) == 0:
        print("Warning: Poisoning graph has no edges. Adding self-loops.")
        poison_indexed_edges = [(i, i) for i in range(len(node_list))]
    poison_edge_index = torch.tensor(poison_indexed_edges, dtype=torch.long).t().contiguous()
    poison_edge_index = torch.cat([poison_edge_index, poison_edge_index.flip([0])], dim=1)
    poison_edge_index = torch.unique(poison_edge_index, dim=1)
    # data_poison contains POISONED graph structure and corrupted features
    data_poison = Data(x=x_poison, edge_index=poison_edge_index, y=data_pg.y, 
                       train_mask=data_pg.train_mask, test_mask=data_pg.test_mask)

    # Train NEW GraphSAGE model on the POISONED data
    gnn_model_sage_poisoned = GraphSAGEModel(IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS)
    train_acc_sage, train_f1_sage = train_gnn(gnn_model_sage_poisoned, data_poison, model_name="GraphSAGE (Poisoned Train)")
    # 3. Evaluate on POISONED TEST DATA (Default behavior)
    poisoning_metrics_sage, _ = evaluate_gnn(gnn_model_sage_poisoned, data_poison, data_poison.test_mask, "GraphSAGE (Poisoning Attack Test/Poisoned)")
    all_metrics.append({**poisoning_metrics_sage, 'Model': 'GraphSAGE', 'Scenario': 'Poisoning Attack (on Poisoned Data)', 'Train_Acc': train_acc_sage, 'Train_F1': train_f1_sage})

    # Evaluate poisoned model on CLEAN TEST DATA
    # data_pg contains the clean features and clean edge index for testing
    clean_metrics_sage, _ = evaluate_gnn(gnn_model_sage_poisoned, data_pg, data_pg.test_mask, "GraphSAGE (Poisoned Model/Clean Test)")
    all_metrics.append({**clean_metrics_sage, 'Model': 'GraphSAGE', 'Scenario': 'Poisoning Attack (on Clean Data)', 'Train_Acc': train_acc_sage, 'Train_F1': train_f1_sage})

    # Train NEW GCN model on the POISONED data
    gnn_model_gcn_poisoned = GCNModel(IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS)
    train_acc_gcn, train_f1_gcn = train_gnn(gnn_model_gcn_poisoned, data_poison, model_name="GCN (Poisoned Train)")
    # 3. Evaluate on POISONED TEST DATA (Default behavior)
    poisoning_metrics_gcn, _ = evaluate_gnn(gnn_model_gcn_poisoned, data_poison, data_poison.test_mask, "GCN (Poisoning Attack Test/Poisoned)")
    all_metrics.append({**poisoning_metrics_gcn, 'Model': 'GCN', 'Scenario': 'Poisoning Attack (on Poisoned Data)', 'Train_Acc': train_acc_gcn, 'Train_F1': train_f1_gcn})
    
    # Evaluate poisoned model on CLEAN TEST DATA
    # data_pg contains the clean features and clean edge index for testing
    clean_metrics_gcn, _ = evaluate_gnn(gnn_model_gcn_poisoned, data_pg, data_pg.test_mask, "GCN (Poisoned Model/Clean Test)")
    all_metrics.append({**clean_metrics_gcn, 'Model': 'GCN', 'Scenario': 'Poisoning Attack (on Clean Data)', 'Train_Acc': train_acc_gcn, 'Train_F1': train_f1_gcn})

    #4. FINAL COMPARISON & VISUALIZATION
    final_df = pd.DataFrame(all_metrics)
    display_df = final_df[['Model', 'Scenario', 'Accuracy', 'F1-Score (Bot)', 'Recall (Bot)', 'F1-Score (Human)', 'Recall (Human)', 'Train_Acc', 'Train_F1']]
    display_df = display_df.rename(columns={'F1-Score (Bot)': 'F1 (Bot)', 'F1-Score (Human)': 'F1 (Human)', 'Recall (Bot)': 'Recall (Bot)', 'Recall (Human)': 'Recall (Human)', 'Train_Acc': 'Train Acc', 'Train_F1': 'Train F1 (Bot)'})

    # Print the table
    print("\n" + "="*115 + "\nFINAL PERFORMANCE COMPARISON: ALL MODELS & ATTACKS (Includes Human Metrics)")
    print(display_df.round(4).to_markdown(index=False))
    print("="*115)
    
    save_dataframe_to_csv(final_df.round(4), METRICS_FOLDER, '02_performance_comparison.csv')

    # Visualizations
    vis_target_idx_evasion = find_best_viz_target(test_bot_indices, data_pg, min_degree=5)
    vis_target_idx_poison = find_best_viz_target(train_bot_indices, data_pg, min_degree=5)

    vis_target_node_evasion = idx_to_node[vis_target_idx_evasion]
    vis_target_node_poison = idx_to_node[vis_target_idx_poison]

    visualize_attack_comparison_full(
        graph_baseline, data_evasion, data_poison, all_bot_nodes_nx, 
        idx_to_node, vis_target_node_evasion, vis_target_node_poison, 
        attack_targets_pg, train_bot_indices
    )