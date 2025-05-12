import os

# 基础目录路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型路径
INTENT_MODEL_PATH = os.path.join(BASE_DIR, 'models/intent_classifier/intent_classifier_SVM.pkl')
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, 'models/intent_classifier/label_encoder.pkl')
NER_MODEL_PATH = os.path.join(BASE_DIR, 'models/mobilebert-ner')

# BERT模型配置
BERT_MODEL_NAME = "distilbert-base-uncased"

# 服务器配置
DEBUG = True
HOST = '0.0.0.0'
PORT = 8888