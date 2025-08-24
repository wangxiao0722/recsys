import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
import time
from functools import wraps
from torch.utils.data import Dataset, DataLoader


def print_model_params_torch(model):
    """Print weights and biases of each layer in the PyTorch model."""
    print("Model parameters: ")
    for name, param in model.named_parameters():
        print(f"{name} shape: {tuple(param.shape)}")
        print(param.data.cpu().numpy())


def compute_auc(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    return auc


def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        running_time_minutes = (end_time - start_time) / 60
        print(f"函数 {func.__name__} 运行时间: {running_time_minutes:.6f} 分钟")
        return result
    return wrapper


class CTRIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, csv_file, cat_features, label_col=None, chunksize=100_000):
        self.csv_file = csv_file
        self.cat_features = cat_features
        self.label_col = label_col
        self.chunksize = chunksize

    def __iter__(self):
        for chunk in pd.read_csv(self.csv_file, chunksize=self.chunksize):
            x = {col: torch.from_numpy(chunk[col].values.astype("int64")) for col in self.cat_features}
            if self.label_col:
                y = torch.from_numpy(chunk[self.label_col].values.astype("float32"))
                for i in range(len(chunk)):
                    yield {col: x[col][i] for col in x}, y[i]
            else:
                for i in range(len(chunk)):
                    yield {col: x[col][i] for col in x}



class EmbeddingDNN(nn.Module):
    def __init__(self, field_dims, hidden_dims=(32, 16, 8), emb_dim=8):
        """
        简单的 3 层 DNN 模型（带 Embedding 输入）
        :param field_dims: dict {field_name: num_embeddings}
        :param hidden_dims: 隐藏层维度 (三层)
        """
        super().__init__()
        # 每个 field 一个 embedding，输出维度设为 8（可以调）
        self.embeddings = nn.ModuleDict({
            field: nn.Embedding(num_embeddings, emb_dim) for field, num_embeddings in field_dims.items()
        })

        input_dim = len(field_dims) * emb_dim  # 拼接后的维度

        layers = []
        dim_in = input_dim

        for dim_out in hidden_dims:
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(nn.LayerNorm(dim_out))
            layers.append(nn.ReLU())
            dim_in = dim_out
        
        self.mlp = nn.Sequential(*layers)
        self.fc = nn.Linear(dim_in, 1)


    def forward(self, x):
        # 拼接所有 field 的 embedding
        emb_list = [self.embeddings[field](x[field]) for field in self.embeddings]
        x_emb = torch.cat(emb_list, dim=1)  # [batch_size, num_fields*8]
        hidden = self.mlp(x_emb)
        logits = self.fc(hidden).squeeze(1)
        prob = torch.sigmoid(logits)
        return prob

@timing_decorator
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


@timing_decorator
def embedding_lr_train_predict(
    x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame, field_dims,
    cat_features, epochs=1, batch_size=64, lr=0.001
) -> np.ndarray:
    """
    Train EmbeddingLR model and predict on test data.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmbeddingDNN(field_dims).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
   
    train_dataset = CTRIterableDataset("data/train.csv", cat_features, label_col="click", chunksize=100000)
    train_loader = DataLoader(train_dataset, batch_size=1024,
                              num_workers=0, pin_memory=True)

    for epoch in range(epochs):
        model.train()
        all_preds = []
        all_labels = []
        epoch_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = {k: v.to(device) for k, v in batch_x.items()}
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_y.size(0)
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(batch_y.detach().cpu().numpy())

        epoch_loss /= len(train_dataset)
        epoch_auc = roc_auc_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.6f}, AUC: {epoch_auc:.6f}")

    print("Predicting on test data...")
    model.eval()
    test_dataset = CTRDataset(x_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    preds = []
    with torch.no_grad():
        for batch_x in test_loader:
            batch_x = {k: v.to(device) for k, v in batch_x.items()}
            outputs = model(batch_x)
            preds.append(outputs.cpu().numpy())
    preds = np.concatenate(preds, axis=0).flatten()
    return preds

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
        "data/test.csv",
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
        x_train, y_train, x_test, field_dims, cat_features,
        epochs=10, batch_size=64, lr=0.001
    )
    print('predictions: \n')
    print(predictions)
    test['click'] = predictions
    test[['id', 'click']].to_csv('data/result.csv', index=False)
    print("Results saved to result.csv")

if __name__ == "__main__":
    main()
