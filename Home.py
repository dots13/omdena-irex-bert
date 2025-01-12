import streamlit as st
import numpy as np
import pandas as pd
import json
from pathlib import Path

from helper_functions import *

@st.cache_resource
def model_init(MODEL_SAVE_PATH):
    print('model_init')
    MDFEND_MODEL = MDFEND(bert, domain_num, expert_num=15, mlp_dims=[2024, 1012, 606])
    MDFEND_MODEL.load_state_dict(
        torch.load(f=MODEL_SAVE_PATH, map_location=torch.device("cpu")))
    return MDFEND_MODEL

def load_model(f_checkpoint):
    f_checkpoint = f_checkpoint
    with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
        gdown.download(id='1-4NIx36LmRF2R5T8Eu5Zku_-CGvV07VE', quiet=True, use_cookies=False)

if __name__ == "__main__":
    max_len, bert = 178, "dccuchile/bert-base-spanish-wwm-uncased"
    tokenizer = TokenizerFromPreTrained(max_len, bert)

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

    model5 = "last-epoch-model-2024-02-27-15_22_42_6.pth"
    f_checkpoint = Path(f"assets/models//{model5}")
    # https://drive.google.com/file/d/1S4c77GF9PA29QOJyQOceZVb6zyBSbETG/view?usp=sharing
    if verify_checkpoint(model5, f_checkpoint, "1S4c77GF9PA29QOJyQOceZVb6zyBSbETG"):
        print('verify_checkpoint')
        MODEL_SAVE_PATH = f"assets/models/last-epoch-model-2024-02-27-15_22_42_6.pth"
        MDFEND_MODEL = model_init(f_checkpoint)

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
