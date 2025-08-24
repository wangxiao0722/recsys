import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score


def print_model_params_torch(model):
    """Print weights and biases of each layer in the PyTorch model."""
    print("Model parameters: ")
    for name, param in model.named_parameters():
        print(f"{name} shape: {tuple(param.shape)}")
        print(param.data.cpu().numpy())


def compute_auc(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    return auc


class EmbeddingDNN(nn.Module):
    def __init__(self, field_dims, hidden_dims=(512, 256, 32)):
        """
        简单的 3 层 DNN 模型（带 Embedding 输入）
        :param field_dims: dict {field_name: num_embeddings}
        :param hidden_dims: 隐藏层维度 (三层)
        """
        super().__init__()
        # 每个 field 一个 embedding，输出维度设为 8（可以调）
        self.embeddings = nn.ModuleDict({
            field: nn.Embedding(num_embeddings, 32) for field, num_embeddings in field_dims.items()
        })

        input_dim = len(field_dims) * 32  # 拼接后的维度

        # 三层 MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(hidden_dims[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.LayerNorm(hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], 1)   # 输出层
        )

    def forward(self, x):
        # 拼接所有 field 的 embedding
        emb_list = [self.embeddings[field](x[field]) for field in self.embeddings]
        x_emb = torch.cat(emb_list, dim=1)  # [batch_size, num_fields*8]

        logits = self.mlp(x_emb).squeeze(1)
        prob = torch.sigmoid(logits)
        return prob


def preprocess_data_with_ordinal_encoder(train_df, test_df, cat_features):
    """
    Encode categorical features using OrdinalEncoder.
    Unknown categories in test set are assigned to a special OOV index.
    field_dims: dict mapping feature name to number of unique values (+1 for OOV).
    """
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1)
    train_df[cat_features] = encoder.fit_transform(
        train_df[cat_features].astype(str))
    test_df[cat_features] = encoder.transform(
        test_df[cat_features].astype(str))
    # +1 for 0-indexing, +1 for OOV
    field_dims = {feat: int(train_df[feat].max()) + 2 for feat in cat_features}
    # Replace unknown values in test set with OOV index
    for feat in cat_features:
        test_df[feat] = test_df[feat].replace(-1, field_dims[feat] - 1)
    return train_df, test_df, field_dims

def embedding_lr_train_predict(
    x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame, field_dims,
    emb_dim=1, epochs=1, batch_size=64, lr=0.001
) -> np.ndarray:
    """
    Train EmbeddingLR model and predict on test data.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmbeddingDNN(field_dims).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    train_tensor = {field: torch.LongTensor(
        x_train[field].values).to(device) for field in field_dims}
    test_tensor = {field: torch.LongTensor(
        x_test[field].values).to(device) for field in field_dims}
    n_samples = len(x_train)
    all_preds = []
    all_labels = []
    for epoch in range(epochs):
        epoch_loss = 0
        permutation = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = {field: val[indices]
                       for field, val in train_tensor.items()}
            batch_y = y_train_tensor[indices]
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            epoch_loss += loss.item() * len(batch_y)
            loss.backward()
            optimizer.step()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(batch_y.detach().cpu().numpy())
        epoch_auc = roc_auc_score(all_labels, all_preds)
        epoch_loss /= n_samples
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.6f}, AUC: {epoch_auc:.6f}")

    print("Predicting on test data...")
    # print_model_params_torch(model)
    with torch.no_grad():
        pred = model(test_tensor).cpu().numpy().flatten()
        print("Prediction completed.")
    return pred

def main():
    # Data preparation
    dtype_map = {
        'id': str,                       # 如果有 id 列
        'click': 'int8',                  # 标签
        'hour': 'int32',                  # 时间，可以读作整数
        # 离散 / 类别特征
        'C1': 'category',
        'banner_pos': 'category',
        'site_id': 'category',
        'site_domain': 'category',
        'site_category': 'category',
        'app_id': 'category',
        'app_domain': 'category',
        'app_category': 'category',
        'device_id': 'category',
        'device_ip': 'category',
        'device_model': 'category',
        'device_type': 'category',
        'device_conn_type': 'category',
        'C14': 'category',
        'C15': 'category',
        'C16': 'category',
        'C17': 'category',
        'C18': 'category',
        'C19': 'category',
        'C20': 'category',
        'C21': 'category'
    }

    all_features = [
        'hour',        # 时间特征
        'C1',          # 匿名类别
        'banner_pos',
        'site_id',
        'site_domain',
        'site_category',
        'app_id',
        'app_domain',
        'app_category',
        'device_id',
        'device_ip',
        'device_model',
        'device_type',
        'device_conn_type',
        'C14','C15','C16','C17','C18','C19','C20','C21'  # 匿名类别
    ]

    cat_features = [
        'C1',
        'banner_pos',
        'site_id',
        'site_domain',
        'site_category',
        'app_id',
        'app_domain',
        'app_category',
        'device_id',
        'device_ip',
        'device_model',
        'device_type',
        'device_conn_type',
        'C14','C15','C16','C17','C18','C19','C20','C21'
    ]

    dense_features = [
        'hour'
    ]

    train = pd.read_csv(
        "data/train_sample.csv",
        dtype=dtype_map,
        usecols = all_features + ['click']
    )

    test = pd.read_csv(
        "data/test_sample.csv",
        dtype=dtype_map,
        usecols=all_features + ['id']
    )

    x_train = train[cat_features].copy()
    x_test = test[cat_features].copy()
    y_train = train['click'].values
    x_train, x_test, field_dims = preprocess_data_with_ordinal_encoder(
        x_train, x_test, cat_features)

    print("Train sample:")
    print(x_train.sample(3))
    print("Test sample:")
    print(x_test.sample(3))
    print("field_dims : ", field_dims)
    print("Label sample: ")
    print(y_train[:3])

    predictions = embedding_lr_train_predict(
        x_train, y_train, x_test, field_dims,
        emb_dim=1, epochs=1, batch_size=1024, lr=0.001
    )
    print('predictions: \n')
    print(predictions)
    test['click'] = predictions
    test[['id', 'click']].to_csv('data/result.csv', index=False)
    print("Results saved to result.csv")

if __name__ == "__main__":
    main()
