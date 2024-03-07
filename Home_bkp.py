import streamlit as st
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import gdown
from tensorflow.keras.models import load_model
from pathlib import Path

import json

from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch import nn
from transformers import BertModel

from faknow.model.layers.layer import TextCNNLayer
from faknow.model.model import AbstractModel

from faknow.data.dataset.text import TextDataset
from faknow.data.process.text_process import TokenizerFromPreTrained
from faknow.evaluate.evaluator import Evaluator

import torch
from torch.utils.data import DataLoader

import pandas as pd


@st.cache_resource
def load_model_h5(path):
    return load_model(path, compile=False)


@st.cache(allow_output_mutation=True)
def get_model():
    model5 = "last-epoch-model-2024-02-27-15_22_42_3.pth"
    f_checkpoint = Path(f"assets/models//{model5}")
    # if verify_checkpoint(model5, f_checkpoint, "1klOgwmAUsjkVtTwMi9Cqyheednf_U18n"):
    if verify_checkpoint(model5, f_checkpoint, "1DUzROEnqM6aaBXXnuVOvH2__8wBsEJa9"):
        MODEL5 = load_model_h5(f_checkpoint)


def verify_checkpoint(model_name, f_checkpoint, gID):
    if not f_checkpoint.exists():
        load_model_from_gd(model_name, gID)
    return f_checkpoint.exists()


def load_model_from_gd(model_name, gID):
    save_dest = Path("models")
    save_dest.mkdir(exist_ok=True)
    output = f"assets/models/{model_name}"
    # f_checkpoint = Path(f"models//{model_name}")
    with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
        gdown.download(id=gID, output=output, quiet=False)
        # gdown.download(f"https://drive.google.com/uc?id=1klOgwmAUsjkVtTwMi9Cqyheednf_U18n", output)


class _MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dims: List[int],
        dropout_rate: float,
        output_layer=True,
    ):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(nn.Linear(input_dim, embed_dim))
            layers.append(nn.BatchNorm1d(embed_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """

        Args:
            x (Tensor): shared feature from domain and text, shape=(batch_size, embed_dim)

        """
        return self.mlp(x)


class _MaskAttentionLayer(torch.nn.Module):
    """
    Compute attention layer
    """

    def __init__(self, input_size: int):
        super(_MaskAttentionLayer, self).__init__()
        self.attention_layer = torch.nn.Linear(input_size, 1)

    def forward(
        self, inputs: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        weights = self.attention_layer(inputs).view(-1, inputs.size(1))
        if mask is not None:
            weights = weights.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(weights, dim=-1).unsqueeze(1)
        outputs = torch.matmul(weights, inputs).squeeze(1)
        return outputs, weights


class MDFEND(AbstractModel):
    r"""
    MDFEND: Multi-domain Fake News Detection, CIKM 2021
    paper: https://dl.acm.org/doi/10.1145/3459637.3482139
    code: https://github.com/kennqiang/MDFEND-Weibo21
    """

    def __init__(
        self,
        pre_trained_bert_name: str,
        domain_num: int,
        mlp_dims: Optional[List[int]] = None,
        dropout_rate=0.2,
        expert_num=5,
    ):
        """

        Args:
            pre_trained_bert_name (str): the name or local path of pre-trained bert model
            domain_num (int): total number of all domains
            mlp_dims (List[int]): a list of the dimensions in MLP layer, if None, [384] will be taken as default, default=384
            dropout_rate (float): rate of Dropout layer, default=0.2
            expert_num (int): number of experts also called TextCNNLayer, default=5
        """
        super(MDFEND, self).__init__()
        self.domain_num = domain_num
        self.expert_num = expert_num
        self.bert = BertModel.from_pretrained(pre_trained_bert_name)
        self.embedding_size = self.bert.config.hidden_size
        self.loss_func = nn.BCELoss()
        if mlp_dims is None:
            mlp_dims = [384]

        filter_num = 64
        filter_sizes = [1, 2, 3, 5, 10]
        experts = [
            TextCNNLayer(self.embedding_size, filter_num, filter_sizes)
            for _ in range(self.expert_num)
        ]
        self.experts = nn.ModuleList(experts)

        self.gate = nn.Sequential(
            nn.Linear(self.embedding_size * 2, mlp_dims[-1]),
            nn.ReLU(),
            nn.Linear(mlp_dims[-1], self.expert_num),
            nn.Softmax(dim=1),
        )

        self.attention = _MaskAttentionLayer(self.embedding_size)

        self.domain_embedder = nn.Embedding(
            num_embeddings=self.domain_num, embedding_dim=self.embedding_size
        )
        self.classifier = _MLP(320, mlp_dims, dropout_rate)

    def forward(self, token_id: Tensor, mask: Tensor, domain: Tensor) -> Tensor:
        """

        Args:
            token_id (Tensor): token ids from bert tokenizer, shape=(batch_size, max_len)
            mask (Tensor): mask from bert tokenizer, shape=(batch_size, max_len)
            domain (Tensor): domain id, shape=(batch_size,)

        Returns:
            FloatTensor: the prediction of being fake, shape=(batch_size,)
        """
        text_embedding = self.bert(token_id, attention_mask=mask).last_hidden_state
        attention_feature, _ = self.attention(text_embedding, mask)

        domain_embedding = self.domain_embedder(domain.view(-1, 1)).squeeze(1)

        gate_input = torch.cat([domain_embedding, attention_feature], dim=-1)
        gate_output = self.gate(gate_input)

        shared_feature = 0
        for i in range(self.expert_num):
            expert_feature = self.experts[i](text_embedding)
            shared_feature += expert_feature * gate_output[:, i].unsqueeze(1)

        label_pred = self.classifier(shared_feature)

        return torch.sigmoid(label_pred.squeeze(1))

    def calculate_loss(self, data) -> Tensor:
        """
        calculate loss via BCELoss

        Args:
            data (dict): batch data dict

        Returns:
            loss (Tensor): loss value
        """

        token_ids = data["text"]["token_id"]
        masks = data["text"]["mask"]
        domains = data["domain"]
        labels = data["label"]
        output = self.forward(token_ids, masks, domains)
        return self.loss_func(output, labels.float())

    def predict(self, data_without_label) -> Tensor:
        """
        predict the probability of being fake news

        Args:
            data_without_label (Dict[str, Any]): batch data dict

        Returns:
            Tensor: one-hot probability, shape=(batch_size, 2)
        """

        token_ids = data_without_label["text"]["token_id"]
        masks = data_without_label["text"]["mask"]
        domains = data_without_label["domain"]

        # shape=(n,), data = 1 or 0
        round_pred = torch.round(self.forward(token_ids, masks, domains)).long()
        # after one hot: shape=(n,2), data = [0,1] or [1,0]
        one_hot_pred = torch.nn.functional.one_hot(round_pred, num_classes=2)
        return one_hot_pred


def main():
    max_len, bert = 178, "dccuchile/bert-base-spanish-wwm-uncased"
    tokenizer = TokenizerFromPreTrained(max_len, bert)

    # dataset
    batch_size = 64
    domain_num = 12

    # get_model()
    st.markdown("<h2 style='text-align: center; color:white;'>Omdena: El-Salvador chapter</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color:white;'>BERT local deployment test</h3>", unsafe_allow_html=True)

    MODEL_SAVE_PATH = f"assets/models/last-epoch-model-2024-02-27-15_22_42_6.pth"
    MDFEND_MODEL = MDFEND(bert, domain_num, expert_num=15, mlp_dims=[2024, 1012, 606])
    MDFEND_MODEL.load_state_dict(
        torch.load(f=MODEL_SAVE_PATH, map_location=torch.device("cpu"))
    )

    testing_path = "sample.json"
    user_input = st.text_area("Enter Text to Analyze")
    button = st.button("Analyze")
    if user_input and button:
        dummy = {}
        dummy["text"] = "dummy text"
        dummy["domain"] = 8
        dummy["label"] = 0

        user_data = {}
        user_data["text"] = user_input
        user_data["domain"] = 8
        user_data["label"] = 0

        user_data = pd.DataFrame([dummy, user_data])
        user_data.to_json(testing_path)
        testing_set = TextDataset(testing_path, ["text"], tokenizer)
        testing_loader = DataLoader(testing_set, batch_size, shuffle=False)

        for batch_data in testing_loader:
            output = MDFEND_MODEL.predict(batch_data)
            st.write("Output: ", output)


main()
