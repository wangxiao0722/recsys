import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
from torchviz import make_dot


def print_model_params_torch(model):
    """Print weights and biases of each layer in the PyTorch model."""
    print("Model params: ")
    for name, param in model.named_parameters():
        print(f"{name} shape: {tuple(param.shape)}")
        print(param.data.cpu().numpy())


class EmbeddingLR(nn.Module):
    def __init__(self, field_dims):
        """
        用 Embedding lookup 方式实现逻辑回归，emb_dim 设置为 1
        """
        super().__init__()
        self.embeddings = nn.ModuleDict({
            field: nn.Embedding(num_embeddings, 1) for field, num_embeddings in field_dims.items()
        })
        with torch.no_grad():
            for _, emb in self.embeddings.items():
                emb.weight.fill_(1.0)
        print('Embeddings: ')
        print(self.embeddings)
        print("Embeddings weights: ")
        print(self.embeddings['device_id'].weight.data)
        # LR 的输入是特征数量
        input_dim = 1 * len(field_dims)
        print('imput_dim: ', input_dim)
        self.linear = nn.Linear(input_dim, 1, bias=True)
        with torch.no_grad():
            self.linear.weight.fill_(1.0)
            self.linear.bias.fill_(0.0)  # 初始化偏置为0
        self.linear.weight.requires_grad = False  # 不训练线性层权重

    def forward(self, x):
        # x: 一堆索引，所以会拿到一堆 [[1,1,1],[1,1]]
        # squeeze -1(最后一个维度) 后: [1,1,1,1,1]
        emb_list = []
        for field in self.embeddings:
            print("Self embeddings field: \n")
            print(self.embeddings[field])
            print("x[field] : \n")
            print(x[field])
            print("type x[field] \n")
            print(type(x[field]))
            emb = self.embeddings[field](x[field])  # [batch_size, 1]
            emb_list.append(emb)
        x_emb = torch.cat(emb_list, dim=1)  # [batch_size, num_fields]
        logits = self.linear(x_emb).squeeze(1)
        prob = torch.sigmoid(logits)
        return prob


def preprocess_data_with_ordinal_encoder(train_df, test_df, cat_features):
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1)
    train_df[cat_features] = encoder.fit_transform(
        train_df[cat_features].astype(str))
    test_df[cat_features] = encoder.transform(
        test_df[cat_features].astype(str))
    # +1 for 0-index, +1 for OOV
    field_dims = {feat: int(train_df[feat].max()) + 2 for feat in cat_features}
    # OOV 这类未见过的特征，设为最大编号。
    for feat in cat_features:
        test_df[feat] = test_df[feat].replace(-1, field_dims[feat] - 1)
    return train_df, test_df, field_dims


def embedding_lr_train_predict(
    x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame, field_dims,
    emb_dim=1, epochs=1, batch_size=64, lr=0.001
) -> np.ndarray:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmbeddingLR(field_dims).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    y_train_tensor = torch.FloatTensor(y_train).to(device)
    train_tensor = {field: torch.LongTensor(
        x_train[field].values).to(device) for field in field_dims}
    test_tensor = {field: torch.LongTensor(
        x_test[field].values).to(device) for field in field_dims}
    n_samples = len(x_train)
    print("Starting training...")
    for epoch in range(epochs):
        permutation = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = {field: val[indices]
                       for field, val in train_tensor.items()}
            batch_y = y_train_tensor[indices]
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        model.train()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    # 构造示例输入，batch_size=1
    example_input = {field: torch.LongTensor([0]).to(
        next(model.parameters()).device) for field in field_dims}
    output = model(example_input)
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.format = 'png'
    dot.render('embedding_lr_model')
    print("Predicting on test data...")
    print_model_params_torch(model)
    with torch.no_grad():
        pred = model(test_tensor).cpu().numpy().flatten()
    print("Prediction completed.")
    # output = model(test_tensor)

    return pred


def main():
    # Preprepare Data start
    dtype_map = {
        'id': str,
        'device_id': 'category',
        'site_id': 'category',
        'app_id': 'category',
        'click': 'int8',
    }

    train = pd.read_csv(
        "data/train.csv",
        dtype=dtype_map,
        usecols=['device_id', 'site_id', 'app_id', 'click']
    )

    test = pd.read_csv(
        "data/test.csv",
        dtype=dtype_map,
        usecols=['id', 'device_id', 'site_id', 'app_id']
    )

    cat_features = ['device_id', 'site_id', 'app_id']

    x_train = train[cat_features].copy()
    x_test = test[cat_features].copy()
    y_train = train['click'].values
    x_train, x_test, field_dims = preprocess_data_with_ordinal_encoder(
        x_train, x_test, cat_features)

    # Preprepare Data End
    print("Train sampe:")
    print(x_train.sample(3))
    print("Test sample:")
    print(x_test.sample(3))
    print("field_dims : ", field_dims)
    print("Label sample: ")
    print(y_train[:3])

    predictions = embedding_lr_train_predict(
        x_train, y_train, x_test, field_dims,
        emb_dim=1, epochs=10, batch_size=1024, lr=0.001
    )
    test['click'] = predictions
    test[['id', 'click']].to_csv('data/result.csv', index=False)
    print("Results saved to result.csv")


if __name__ == "__main__":
    main()
