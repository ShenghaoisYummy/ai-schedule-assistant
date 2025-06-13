import os
import torch
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification

from config import INTENT_MODEL_PATH, LABEL_ENCODER_PATH, NER_MODEL_PATH, BERT_MODEL_NAME

class ModelManager:
    def __init__(self):
        self.models = {}
        self.models_loaded = False
        self.sklearn_models_loaded = False
        self.transformers_models_loaded = False
        self.load_models()
    
    def load_models(self):
        """加载所有需要的模型"""
        print("🚀 开始加载模型...")
        
        # Try to load sklearn models first
        self._load_sklearn_models()
        
        # Try to load transformer models
        self._load_transformer_models()
        
        if self.sklearn_models_loaded or self.transformers_models_loaded:
            self.models_loaded = True
            print("✅ 至少部分模型加载成功，应用可以运行")
        else:
            print("⚠️ 所有模型都无法加载，将使用智能回退功能")
            self.models_loaded = False
        
        return True  # Always return True so app can start
    
    def _load_sklearn_models(self):
        """尝试加载sklearn模型"""
        try:
            print("📦 尝试加载sklearn模型...")
            
            # 检查文件是否存在
            if not os.path.exists(INTENT_MODEL_PATH):
                print(f"❌ 意图分类器文件不存在: {INTENT_MODEL_PATH}")
                return False
                
            if not os.path.exists(LABEL_ENCODER_PATH):
                print(f"❌ 标签编码器文件不存在: {LABEL_ENCODER_PATH}")
                return False
            
            # 尝试加载sklearn模型
            print("📄 加载意图分类器...")
            with open(INTENT_MODEL_PATH, "rb") as f:
                self.models["intent_classifier"] = pickle.load(f)
            print("✅ 意图分类器加载成功")
            
            with open(LABEL_ENCODER_PATH, "rb") as f:
                self.models["label_encoder"] = pickle.load(f)
            print("✅ 标签编码器加载成功")
            
            self.sklearn_models_loaded = True
            print("🎯 sklearn模型加载完成")
            return True
            
        except Exception as e:
            print(f"❌ sklearn模型加载失败: {e}")
            print("💡 提示: 这通常是sklearn版本不兼容导致的")
            self.sklearn_models_loaded = False
            return False
    
    def _load_transformer_models(self):
        """尝试加载transformer模型"""
        try:
            print("🤖 尝试加载transformer模型...")
            
            if not os.path.exists(NER_MODEL_PATH):
                print(f"❌ NER模型目录不存在: {NER_MODEL_PATH}")
                return False
            
            # 加载NER模型
            print("🔤 加载NER模型...")
            self.models["ner_tokenizer"] = AutoTokenizer.from_pretrained(NER_MODEL_PATH)
            self.models["ner_model"] = AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH)
            self.models["ner_model"].eval()
            print("✅ NER模型加载成功")
            
            # 获取NER标签映射
            self.models["id2label"] = self.models["ner_model"].config.id2label
            print("✅ NER标签映射获取成功")
            
            # 加载BERT模型用于提取嵌入
            print("🧠 加载BERT模型...")
            self.models["bert_tokenizer"] = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
            self.models["bert_model"] = AutoModel.from_pretrained(BERT_MODEL_NAME)
            print("✅ BERT模型加载成功")
            
            self.transformers_models_loaded = True
            print("🎯 transformer模型加载完成")
            return True
            
        except Exception as e:
            print(f"❌ transformer模型加载失败: {e}")
            self.transformers_models_loaded = False
            return False
    
    def get_embedding(self, text):
        """从文本中提取BERT嵌入向量"""
        if not self.transformers_models_loaded or "bert_tokenizer" not in self.models:
            # 返回随机嵌入作为fallback
            print("⚠️ BERT模型未加载，使用随机嵌入")
            return np.random.rand(1, 768)  # DistilBERT的嵌入维度
            
        try:
            tokenizer = self.models["bert_tokenizer"]
            model = self.models["bert_model"]
            
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # 使用[CLS]标记表示
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
            return embedding
        except Exception as e:
            print(f"BERT嵌入提取失败: {e}")
            return np.random.rand(1, 768)
    
    def predict_intent(self, text):
        """预测文本的意图"""
        # 智能意图识别 - 即使没有模型也能工作
        if self.sklearn_models_loaded:
            try:
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
            except Exception as e:
                print(f"模型意图预测出错: {e}")
        
        # 智能回退：基于关键词的意图识别
        return self._smart_intent_fallback(text)
    
    def _smart_intent_fallback(self, text):
        """智能意图识别回退方案"""
        text_lower = text.lower()
        
        # 定义关键词模式
        add_keywords = ['schedule', 'book', 'add', 'create', 'set', 'plan', 'arrange', 'meeting', 'appointment']
        delete_keywords = ['cancel', 'delete', 'remove', 'clear']
        update_keywords = ['change', 'update', 'modify', 'move', 'reschedule']
        query_keywords = ['when', 'what', 'show', 'list', 'find', 'search', 'tell', 'check']
        
        # 计算匹配分数
        add_score = sum(1 for keyword in add_keywords if keyword in text_lower)
        delete_score = sum(1 for keyword in delete_keywords if keyword in text_lower)
        update_score = sum(1 for keyword in update_keywords if keyword in text_lower)
        query_score = sum(1 for keyword in query_keywords if keyword in text_lower)
        
        # 确定最高分数的意图
        scores = {
            'add': add_score,
            'delete': delete_score, 
            'update': update_score,
            'query': query_score
        }
        
        best_intent = max(scores, key=scores.get)
        confidence = min(0.9, max(0.6, scores[best_intent] * 0.2))  # 基于匹配数量的置信度
        
        # 如果没有明确匹配，默认为添加
        if scores[best_intent] == 0:
            best_intent = 'add'
            confidence = 0.5
        
        all_intents = [{"name": intent, "confidence": score * 0.2} for intent, score in scores.items()]
        all_intents.sort(key=lambda x: x["confidence"], reverse=True)
        
        print(f"💡 智能回退识别意图: {best_intent} (置信度: {confidence:.2f})")
        
        return {
            "name": best_intent,
            "confidence": confidence,
            "all_intents": all_intents[:3]
        }
    
    def predict_ner(self, text):
        """预测文本中的命名实体"""
        if self.transformers_models_loaded:
            try:
                return self._transformer_ner(text)
            except Exception as e:
                print(f"transformer NER预测出错: {e}")
        
        # 智能回退：基于规则的实体识别
        return self._smart_ner_fallback(text)
    
    def _transformer_ner(self, text):
        """使用transformer模型进行NER"""
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
    
    def _smart_ner_fallback(self, text):
        """智能实体识别回退方案"""
        import re
        
        words = text.split()
        entities = []
        
        # 时间模式
        time_patterns = [
            r'\b(\d{1,2}):(\d{2})\s*(am|pm|AM|PM)?\b',  # 2:30 PM
            r'\b(\d{1,2})\s*(am|pm|AM|PM)\b',           # 2 PM
            r'\b(at|from|to)\s+(\d{1,2}):?(\d{2})?\s*(am|pm|AM|PM)?\b'
        ]
        
        # 日期关键词
        date_words = ['today', 'tomorrow', 'yesterday', 'next week', 'monday', 'tuesday', 
                     'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        
        # 动作关键词
        action_words = ['schedule', 'book', 'add', 'create', 'cancel', 'delete', 'update', 'modify']
        
        # 检测时间
        for pattern in time_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    "type": "TIME", 
                    "text": match.group(),
                    "start": match.start()
                })
        
        # 检测日期词汇
        text_lower = text.lower()
        for date_word in date_words:
            if date_word in text_lower:
                start_pos = text_lower.find(date_word)
                entities.append({
                    "type": "DATE",
                    "text": date_word,
                    "start": start_pos
                })
        
        # 检测动作词汇
        for action_word in action_words:
            if action_word in text_lower:
                start_pos = text_lower.find(action_word)
                entities.append({
                    "type": "ACTION",
                    "text": action_word,
                    "start": start_pos
                })
        
        # 检测可能的标题（大写开头的连续词汇）
        words = text.split()
        for i, word in enumerate(words):
            if word[0].isupper() and word.lower() not in [w.lower() for w in action_words + date_words]:
                # 可能是事件标题
                title_words = [word]
                j = i + 1
                while j < len(words) and (words[j][0].isupper() or words[j].lower() in ['with', 'and', 'for']):
                    title_words.append(words[j])
                    j += 1
                
                if len(title_words) >= 1:
                    entities.append({
                        "type": "TITLE",
                        "text": " ".join(title_words),
                        "start": i
                    })
        
        print(f"💡 智能回退识别实体: {len(entities)}个实体")
        return entities