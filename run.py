import os
import shutil
import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import EditedNearestNeighbours
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import metrics
from typing import Any, Optional, Tuple
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import warnings
import time
import pywt

warnings.filterwarnings("ignore")

class Config:
    def __init__(self):
        self.dataset = 'eucore3'
        self.network_numbers = 5
        self.train_data = f'data_hmm/{self.dataset}train{self.network_numbers}.txt'
        self.valid_data = f'data_hmm/{self.dataset}valid{self.network_numbers}.txt'
        self.test_data = f'data_hmm/{self.dataset}test{self.network_numbers}.txt'
        self.in_dim = 128
        self.batch_size = 128
        self.initial_learning_rate = 0.0001
        self.epochs = 5
        self.repeats = 10
        self.model_directory = f'output_hmm/{self.dataset}{self.network_numbers}_model'
        self.coeff = 10

def divide_infor_label(data):
    link_label = data[:, 0]
    influence_labels = data[:, 1:5]  # 4个影响力标签
    infor = data[:, 5:]
    return link_label, influence_labels, infor

def divide_network_node(data):
    generate_label = data[:, :, 0]
    timestamp_label = data[:, :, 1]
    node = data[:, :, 2:]
    return timestamp_label, generate_label, node

def get_train_test(train_data, test_data, valid_data, batch_size=64):
    train = pd.read_csv(train_data, header=None, sep=',')
    test = pd.read_csv(test_data, header=None, sep=',')
    valid = pd.read_csv(valid_data, header=None, sep=',')
    train = np.array(train)
    test = np.array(test)
    valid = np.array(valid)
    train_link_label, train_influence, train = divide_infor_label(train)
    test_link_label, test_influence, test = divide_infor_label(test)
    valid_link_label, valid_influence, valid = divide_infor_label(valid)
    train = torch.from_numpy(train).unsqueeze(dim=1).float()
    train_link_label = torch.from_numpy(train_link_label).unsqueeze(dim=1).long()
    train_influence = torch.from_numpy(train_influence).float()
    train_set = TensorDataset(train, train_link_label, train_influence)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test = torch.from_numpy(test).unsqueeze(dim=1).float()
    test_link_label = torch.from_numpy(test_link_label).unsqueeze(dim=1).long()
    test_influence = torch.from_numpy(test_influence).float()
    test_set = TensorDataset(test, test_link_label, test_influence)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,drop_last=True)
    valid = torch.from_numpy(valid).unsqueeze(dim=1).float()
    valid_link_label = torch.from_numpy(valid_link_label).unsqueeze(dim=1).long()
    valid_influence = torch.from_numpy(valid_influence).float()
    valid_set = TensorDataset(valid, valid_link_label, valid_influence)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, test_loader, valid_loader

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

def grad_reverse(x, coeff):
    return GradReverse.apply(x, coeff)

class AANet_influent(nn.Module):
    def __init__(self, in_dim, network_numbers):
        super(AANet_influent, self).__init__()
        self.generality_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.target_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.wave_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_dim//2, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()

        )
        self.sample_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )
        self.weight_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_dim + 2, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.weight_softmax = nn.Sequential(
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        )
        self.link_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )
        self.network_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, network_numbers),
            nn.Softmax(dim=1)
        )
        self.residual1 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.residual2 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.influence_predictor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
    def forward(self, node_embedding, weight_input, coeff=10):
        wavelet_coeffs = []
        for i in range(node_embedding.size(0)):
            coeffs = pywt.wavedec(node_embedding[i].cpu().numpy(), 'db1', level=1)
            wavelet_coeffs.append(torch.tensor(coeffs[0]))  # Extract the first high-frequency component

        wavelet_coeffs = torch.stack(wavelet_coeffs).to(node_embedding.device)

        wavelet_feature = wavelet_coeffs.permute(0, 2, 1)
        wave_feature = self.wave_conv(wavelet_feature)
        node_embedding = node_embedding.permute(0, 2, 1)
        device = node_embedding.device
        node_embedding = node_embedding + 0.3 * wave_feature
        generality_feature = self.generality_conv(node_embedding)
        generality_feature = self.residual1(generality_feature) + generality_feature
        generality_feature = generality_feature.view(generality_feature.size(0), -1)
        target_feature = self.target_conv(node_embedding)
        target_feature = self.residual2(target_feature) + target_feature
        target_feature = target_feature.view(target_feature.size(0), -1)
        weight_input = weight_input.permute(0, 2, 1)
        weight_out = self.weight_conv(weight_input)
        weight_out = weight_out.view(weight_out.size(0), -1)
        weight_out = self.weight_softmax(weight_out)
        feature = torch.zeros_like(target_feature, device=device)
        for i in range(feature.shape[0]):
            feature[i] = generality_feature[i] * weight_out[i][0] + target_feature[i] * weight_out[i][1]
        link_output = self.link_classifier(feature)
        network_output = self.network_classifier(feature)
        sample_output = self.sample_classifier(feature)
        influence_pred = self.influence_predictor(grad_reverse(feature, coeff))
        return link_output, network_output, sample_output, feature, influence_pred

def get_pred(out):
    out = out.argmax(dim=1)
    one = torch.ones_like(out)
    zero = torch.zeros_like(out)
    out = torch.where(out == 1, one, zero)
    return out

def train_AANet_influent_Model(dataset, train_loader, valid_loader, model, criterion, initial_learning_rate, epochs, coeff):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model_path = f'output_hmm/{dataset}{config.network_numbers}_model/'
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.mkdir(model_path)
    total_start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate)
    for epoch in range(epochs + 1):
        model.train()
        loss_vec = []
        auc_vec = []
        for data in train_loader:
            infor, link_label, influence_label = data
            network_label, sample_label, node = divide_network_node(infor)
            infor, node, link_label, network_label, sample_label, influence_label = (
                infor.to(device), node.to(device), link_label.to(device),
                network_label.to(device), sample_label.to(device), influence_label.to(device)
            )
            optimizer.zero_grad()
            link_out, _, sample_out, feature, influence_pred = model(node, infor, coeff)
            link_loss = criterion(link_out, link_label.squeeze(1).long())
            # sample_loss = criterion(sample_out, sample_label.squeeze(1).long())
            # influence_loss = nn.MSELoss()(influence_pred, influence_label)
            loss = link_loss
            loss_vec.append(loss.detach().cpu().numpy())
            auc = metrics.roc_auc_score(link_label.cpu().numpy(), link_out.detach().cpu().numpy()[:, 1])
            auc_vec.append(auc)
            loss.backward(retain_graph=True)
            optimizer.step()
        loss = np.mean(loss_vec)
        auc = np.mean(auc_vec)
        valid_auc_vec = []
        for valid_data in valid_loader:
            valid_infor, valid_link_label, valid_influence_label = valid_data
            _, valid_sample_label, valid_node = divide_network_node(valid_infor)
            valid_infor, valid_node, valid_link_label, valid_sample_label, valid_influence_label = (
                valid_infor.to(device), valid_node.to(device), valid_link_label.to(device),
                valid_sample_label.to(device), valid_influence_label.to(device)
            )
            with torch.no_grad():
                valid_link_out, _, valid_sample_out, feature, valid_influence_pred = model(valid_node, valid_infor, coeff)
                # valid_link_loss = criterion(valid_link_out, valid_link_label.squeeze(1).long())
                # valid_sample_loss = criterion(valid_sample_out, valid_sample_label.squeeze(1).long())
                # valid_influence_loss = nn.MSELoss()(valid_influence_pred, valid_influence_label)
                # meta_loss = valid_link_loss
            link_out_np = valid_link_out.detach().cpu().numpy()
            link_label_np = valid_link_label.cpu().numpy()
            try:
                valid_auc = metrics.roc_auc_score(link_label_np, link_out_np[:, 1])
                valid_auc_vec.append(valid_auc)
            except ValueError:
                pass
        valid_auc = np.mean(valid_auc_vec)
        torch.save(model.state_dict(), model_path + f'model{epoch}.pkl')
        print(
            f'Model Epoch: [{epoch}/{epochs}], learning rate:{initial_learning_rate:.6f}, train loss:{loss:.4f}, train auc:{auc:.4f}')
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    print(f'Total training time: {total_elapsed_time:.2f} seconds')
    return model_path + f'model{epochs}.pkl'

def test_AANet_influent_Model(test_loader, AANet_influent_model, best_valid_dir, coeff):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AANet_influent_model.load_state_dict(torch.load(best_valid_dir, map_location=device))
    AANet_influent_model = AANet_influent_model.to(device)
    AANet_influent_model.eval()
    acc_vec = []
    precision_vec = []
    f1_vec = []
    auc_vec = []
    aupr_vec = []
    for data in test_loader:
        infor, link_label, influence_label = data
        _, _, node = divide_network_node(infor)
        infor, node, link_label = infor.to(device), node.to(device), link_label.to(device)
        with torch.no_grad():
            AANet_influent_out, _, _, feature, influence_pred = AANet_influent_model(node, infor, coeff)
            pred = get_pred(AANet_influent_out).cpu()
            link_label = link_label.squeeze(1).long().cpu()
        acc = (pred == link_label).float().mean()
        acc_vec.append(acc.detach().cpu().numpy())
        precision = metrics.precision_score(link_label, pred, average='weighted')
        f1 = metrics.f1_score(link_label, pred, average='weighted')
        precision_vec.append(precision)
        f1_vec.append(f1)
        try:
            auc = metrics.roc_auc_score(link_label, AANet_influent_out.detach().cpu().numpy()[:, 1])
            auc_vec.append(auc)
        except ValueError:
            pass
        try:
            aupr = metrics.average_precision_score(link_label, AANet_influent_out.detach().cpu().numpy()[:, 1])
            aupr_vec.append(aupr)
        except ValueError:
            pass
    auc = np.mean(auc_vec)
    precision = np.mean(precision_vec)
    accuracy = np.mean(acc_vec)
    f1_score = np.mean(f1_vec)
    aupr = np.mean(aupr_vec)
    return auc, precision, accuracy, f1_score, aupr

if __name__ == "__main__":
    config = Config()
    acc_t = []
    precision_t = []
    recall_t = []
    f1_t = []
    auc_t = []
    aupr_t = []
    for repeat in range(config.repeats):
        train_loader, test_loader, valid_loader = get_train_test(config.train_data, config.valid_data, config.test_data, config.batch_size)
        AANet_influent_model = AANet_influent(in_dim=config.in_dim, network_numbers=config.network_numbers)
        criterion = nn.CrossEntropyLoss()
        best_valid_dir = train_AANet_influent_Model(config.dataset, train_loader, valid_loader, AANet_influent_model, criterion, config.initial_learning_rate, config.epochs, config.coeff)
        auc, precision, acc, f1, aupr = test_AANet_influent_Model(test_loader, AANet_influent_model, best_valid_dir, config.coeff)
        print(f'repeat:{repeat + 1}, ROC-AUC:{auc:.4f}, Precision:{precision:.4f}, F1_score:{f1:.4f}, AUPR:{aupr:.4f}')
        acc_t.append(acc)
        precision_t.append(precision)
        f1_t.append(f1)
        auc_t.append(auc)
        aupr_t.append(aupr)