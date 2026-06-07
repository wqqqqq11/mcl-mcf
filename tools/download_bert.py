from transformers import BertTokenizer, BertModel, BertConfig

# 指定保存路径
save_path = "D:/PythonWorkplace/MCL-MCF-main/MCL-MCF/src/bert"

# 下载并保存模型文件
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)

# 保存到本地
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
config.save_pretrained(save_path)

print(f"BERT模型已成功下载到 {save_path}")