import pandas as pd
import numpy as np
import re
import os
import sys
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import gradio as gr


def is_false_alarm(code_text):

    code_text = re.sub('\/\*[\S\s]*\*\/', '', code_text)
    code_text = re.sub('\/\/.*', '', code_text)
    code_text = re.sub('(\\\\n)+', '\\n', code_text)

    # 1. CFA-CodeBERTa-small.pt -> CodeBERTa-small-v1 finetunig model
    path = os.getcwd() + '\models\CFA-CodeBERTa-small.pt'
    tokenizer = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
    input_ids = tokenizer.encode(
        code_text, max_length=512, truncation=True, padding='max_length')
    input_ids = torch.tensor([input_ids])
    model = RobertaForSequenceClassification.from_pretrained(
        path, num_labels=2)
    model.to('cpu')
    pred_1 = model(input_ids)[0].detach().cpu().numpy()[0]
    # model(input_ids)[0].argmax().detach().cpu().numpy().item()

    # 2. CFA-codebert-c.pt -> codebert-c finetuning model
    path = os.getcwd() + '\models\CFA-codebert-c.pt'
    tokenizer = AutoTokenizer.from_pretrained(path)
    input_ids = tokenizer(code_text, padding=True, max_length=512,
                          truncation=True, return_token_type_ids=True)['input_ids']
    input_ids = torch.tensor([input_ids])
    model = AutoModelForSequenceClassification.from_pretrained(
        path, num_labels=2)
    model.to('cpu')
    pred_2 = model(input_ids)[0].detach().cpu().numpy()[0]

    # 3. CFA-codebert-c-v2.pt -> undersampling + codebert-c finetuning model
    path = os.getcwd() + '\models\CFA-codebert-c-v2.pt'
    tokenizer = RobertaTokenizer.from_pretrained(path)
    input_ids = tokenizer(code_text, padding=True, max_length=512,
                          truncation=True, return_token_type_ids=True)['input_ids']
    input_ids = torch.tensor([input_ids])
    model = RobertaForSequenceClassification.from_pretrained(
        path, num_labels=2)
    model.to('cpu')
    pred_3 = model(input_ids)[0].detach().cpu().numpy()

    # 4. codeT5 finetuning model
    path = os.getcwd() + '\models\CFA-codeT5'
    model_params = {
        # model_type: t5-base/t5-large
        "MODEL": path,
        "TRAIN_BATCH_SIZE": 8,  # training batch size
        "VALID_BATCH_SIZE": 8,  # validation batch size
        "VAL_EPOCHS": 1,  # number of validation epochs
        "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
        "MAX_TARGET_TEXT_LENGTH": 3,  # max length of target text
        "SEED": 2022,  # set seed for reproducibility
    }
    data = pd.DataFrame({'code': [code_text]})
    pred_4 = T5Trainer(
        dataframe=data,
        source_text="code",
        model_params=model_params
    )
    pred_4 = int(pred_4[0])

    # ensemble
    tot_result = (pred_1 * 0.1 + pred_2 * 0.1 +
                  pred_3 * 0.7 + pred_4 * 0.1).argmax()
    if tot_result == 0:
        return "false positive !!"
    else:
        return "true positive !!"


# codeT5
class YourDataSetClass(Dataset):

    def __init__(
            self, dataframe, tokenizer, source_len, source_text):

        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        # self.summ_len = target_len
        # self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        return len(self.source_text)

    def __getitem__(self, index):

        source_text = str(self.source_text[index])
        source_text = " ".join(source_text.split())
        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
        }


def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=150,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )

            preds = [tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            if ((preds != '0') | (preds != '1')):
                preds = '0'

            predictions.extend(preds)
    return predictions


def T5Trainer(dataframe, source_text, model_params, step="test",):

    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = model.to('cpu')

    dataframe = dataframe[[source_text]]

    val_dataset = dataframe
    val_set = YourDataSetClass(
        val_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"],  source_text)

    val_params = {
        'batch_size': model_params["VALID_BATCH_SIZE"],
        'shuffle': False,
        'num_workers': 0
    }

    val_loader = DataLoader(val_set, **val_params)

    for epoch in range(model_params["VAL_EPOCHS"]):
        predictions = validate(epoch, tokenizer, model, 'cpu', val_loader)

    return predictions


#################################################################################

'''demo = gr.Interface(
    fn = greet,
    inputs = "text",
    outputs= "number")
demo.launch(share=True)
'''
with gr.Blocks() as demo1:
    gr.Markdown(
        """
    <h1 align="center">
    False-Alarm-Detector
    </h1>
    """)

    gr.Markdown(
        """
    정적 분석기를 통해 오류라고 보고된 C언어 코드의 함수를 입력하면,
    오류가 True-positive 인지 False-positive 인지 분류 해 주는 프로그램입니다.
    """)

    '''
    with gr.Accordion(label='모델에 대한 설명 ( 여기를 클릭 하시오. )',open=False):
        gr.Markdown(
        """
        총 3개의 모델을 사용하였다.
        1. codeBERTa-small-v1
        - codeBERTa-small-v1 설명
        2. codeBERT - C
        - codeBERT - C 설명
        3. codeT5
        - codeT5 설명
        """
        )
    '''
    with gr.Row():
        with gr.Column():
            inputs = gr.Textbox(
                lines=10, placeholder="코드를 입력하시오.", label='Code')
            with gr.Row():
                btn = gr.Button("결과 출력")
        with gr.Column():
            output = gr.Text(label='Result')
    btn.click(fn=is_false_alarm, inputs=inputs, outputs=output)

if __name__ == "__main__":
    demo1.launch()
