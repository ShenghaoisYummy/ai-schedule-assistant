from flask import Flask, request, jsonify
import traceback
import time
from models import ModelManager
from utils import generate_response
from config import DEBUG, HOST, PORT
from flask_cors import CORS


app = Flask(__name__)
CORS(app) 
model_manager = ModelManager()

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({"status": "healthy", "timestamp": time.time()})

@app.route('/analyze', methods=['POST'])
def analyze():
    """分析用户输入的文本"""
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data['text']
        
        # 预测意图
        intent = model_manager.predict_intent(text)
        
        # 预测实体
        entities = model_manager.predict_ner(text)
        
        # 生成响应
        response = generate_response(intent, entities)
        
        # 返回结果
        result = {
            "text": text,
            "intent": intent,
            "entities": entities,
            "response": response
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"处理请求时出错: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=DEBUG, host=HOST, port=PORT)
    
#使用curl测试api    
# 测试健康检查端点
# curl http://localhost:8888/health

# 测试分析端点
# curl -X POST http://localhost:8888/analyze \
#  -H "Content-Type: application/json" \
#  -d '{"text": "Schedule a meeting with John tomorrow at 2pm"}'