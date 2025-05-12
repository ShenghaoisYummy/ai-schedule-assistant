import os
import torch
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification

from config import INTENT_MODEL_PATH, LABEL_ENCODER_PATH, NER_MODEL_PATH, BERT_MODEL_NAME

class ModelManager:
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """加载所有需要的模型"""
        try:
            # 加载意图分类器
            with open(INTENT_MODEL_PATH, "rb") as f:
                self.models["intent_classifier"] = pickle.load(f)
            
            with open(LABEL_ENCODER_PATH, "rb") as f:
                self.models["label_encoder"] = pickle.load(f)
            
            # 加载NER模型
            self.models["ner_tokenizer"] = AutoTokenizer.from_pretrained(NER_MODEL_PATH)
            self.models["ner_model"] = AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH)
            self.models["ner_model"].eval()
            
            # 获取NER标签映射
            self.models["id2label"] = self.models["ner_model"].config.id2label
            
            # 加载BERT模型用于提取嵌入
            self.models["bert_tokenizer"] = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
            self.models["bert_model"] = AutoModel.from_pretrained(BERT_MODEL_NAME)
            
            print("所有模型加载成功")
            return True
        except Exception as e:
            print(f"加载模型时出错: {e}")
            return False
    
    def get_embedding(self, text):
        """从文本中提取BERT嵌入向量"""
        tokenizer = self.models["bert_tokenizer"]
        model = self.models["bert_model"]
        
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 使用[CLS]标记表示
        embedding = outputs.last_hidden_state[:, 0, :].numpy()
        return embedding
    
    def predict_intent(self, text):
        """预测文本的意图"""
        embedding = self.get_embedding(text)
        
        classifier = self.models["intent_classifier"]
        label_encoder = self.models["label_encoder"]
        
        # 获取预测概率
        probabilities = classifier.predict_proba(embedding)[0]
        
        # 获取预测类别索引
        prediction = classifier.predict(embedding)[0]
        
        # 获取意图名称
        intent = label_encoder.inverse_transform([prediction])[0]
        
        # 获取置信度（预测类别的概率）
        confidence = float(probabilities[prediction])
        
        # 获取所有类别的概率并排序
        all_probs = []
        for i, prob in enumerate(probabilities):
            intent_name = label_encoder.inverse_transform([i])[0]
            all_probs.append({"name": intent_name, "confidence": float(prob)})
        
        all_probs.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "name": intent,
            "confidence": confidence,
            "all_intents": all_probs[:3]  # 只返回前3个最可能的意图
        }
    
    def predict_ner(self, text):
        """预测文本中的命名实体"""
        # 分词
        words = text.split()
        
        ner_model = self.models["ner_model"]
        ner_tokenizer = self.models["ner_tokenizer"]
        id2label = self.models["id2label"]
        
        # 使用模型的分词器处理
        inputs = ner_tokenizer(words, truncation=True, is_split_into_words=True, return_tensors="pt")
        
        # 预测
        with torch.no_grad():
            outputs = ner_model(**inputs)
        
        # 获取预测标签
        predictions = torch.argmax(outputs.logits, dim=2).squeeze().tolist()
        
        # 将预测与单词对齐
        word_ids = inputs.word_ids()
        predicted_labels = []
        current_word = None
        
        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            
            if word_id != current_word:
                current_word = word_id
                predicted_labels.append(id2label[predictions[idx]])
        
        # 将结果格式化为实体列表
        ner_results = list(zip(words, predicted_labels))
        entities = []
        current_entity = None
        
        for i, (word, label) in enumerate(ner_results):
            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                entity_type = label[2:]  # 移除 "B-"
                current_entity = {"type": entity_type, "text": word, "start": i}
            elif label.startswith("I-") and current_entity:
                current_entity["text"] += " " + word
            elif label == "O":
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities