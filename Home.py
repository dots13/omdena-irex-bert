import streamlit as st
import numpy as np
import pandas as pd
import json
from pathlib import Path

from helper_functions import *

def load_model(f_checkpoint):
    f_checkpoint = f_checkpoint
    with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
        gdown.download(id='1-4NIx36LmRF2R5T8Eu5Zku_-CGvV07VE', quiet=True, use_cookies=False)

if __name__ == "__main__":
    max_len, bert = 178, "dccuchile/bert-base-spanish-wwm-uncased"
    tokenizer = TokenizerFromPreTrained(max_len, bert)
    model5 = "last-epoch-model-2024-02-27-15_22_42_6.pth"
    f_checkpoint = Path(f"models//{model5}")
    if not f_checkpoint.exists():
        load_model(f_checkpoint)
    path_to_model = f_checkpoint
    domain_num = 12
    max_len, bert = 178 , 'dccuchile/bert-base-spanish-wwm-uncased'
    MDFEND_MODEL = MDFEND(bert, domain_num , expert_num=15 , mlp_dims = [2024 ,1012 ,606])
    MDFEND_MODEL.load_state_dict(torch.load(f=path_to_model , map_location=torch.device('cpu')))

    # dataset
    batch_size = 64
    domain_num = 12

    # get_model()
    st.markdown(
        "<h2 style='text-align: center; color:white;'>Omdena: El-Salvador chapter</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h3 style='text-align: center; color:white;'>BERT: Misinformation Classification</h3>",
        unsafe_allow_html=True,
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
            st.write("Output: ", output[1])
