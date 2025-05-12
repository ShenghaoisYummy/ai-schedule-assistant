def generate_response(intent, entities):
    """根据意图和实体生成回复"""
    
    # 提取各种实体
    action = next((e['text'] for e in entities if e['type'] == 'action'), '')
    title = next((e['text'] for e in entities if e['type'] == 'title'), '')
    date = next((e['text'] for e in entities if e['type'] == 'date'), '')
    location = next((e['text'] for e in entities if e['type'] == 'location'), '')
    start_time = next((e['text'] for e in entities if e['type'] == 'startTime'), '')
    end_time = next((e['text'] for e in entities if e['type'] == 'endTime'), '')
    description = next((e['text'] for e in entities if e['type'] == 'description'), '')
    
    intent['name'] = intent['name'].lower()
    # 根据意图生成响应
    if intent['name'] == 'add':
        response = f"I've scheduled an event"
        if title:
            response += f" titled '{title}'"
        if date:
            response += f" on {date}"
        if start_time:
            response += f" starting at {start_time}"
        if end_time:
            response += f" ending at {end_time}"
        if location:
            response += f" at {location}"
        if description:
            response += f". Notes: {description}"
        response += "."
        
        # 如果没有足够的信息
        if not title and not date and not start_time:
            response = "I'd like to help you schedule an event. Could you provide more details like title, date or time?"
        
        return response
    
    elif intent['name'] == 'delete':
        if title or date or start_time:
            response = f"I've deleted"
            if title:
                response += f" the event '{title}'"
            else:
                response += " the event"
            if date:
                response += f" on {date}"
            if start_time:
                response += f" at {start_time}"
            response += "."
            return response
        else:
            return "Which event would you like me to delete? Please provide the title, date or time."
    
    elif intent['name'] == 'update':
        if title or date or start_time:
            response = f"I've updated"
            if title:
                response += f" the event '{title}'"
            else:
                response += " the event"
            
            changes = []
            if date:
                changes.append(f"date to {date}")
            if start_time:
                changes.append(f"start time to {start_time}")
            if end_time:
                changes.append(f"end time to {end_time}")
            if location:
                changes.append(f"location to {location}")
            if description:
                changes.append(f"description to '{description}'")
                
            if changes:
                response += " with " + ", ".join(changes)
            response += "."
            response = "Sorry, we do not support upgrade. We can only do add/delete/query."
            return response
        else:
            return "Which event would you like me to update, and what changes should I make?"
    
    elif intent['name'] == 'chitchat':
        # 根据实体或上下文生成闲聊回复
        greetings = ["Hi", "Hello", "Hey"]
        if any(word in action.lower() for word in ["hi", "hello", "hey", "greetings"]):
            return f"{greetings[0]} there! How can I help with your schedule today?"
        elif any(word in action.lower() for word in ["how are you", "doing", "feeling"]):
            return "I'm doing well, thanks for asking! How can I assist with your calendar?"
        elif any(word in action.lower() for word in ["thank", "thanks"]):
            return "You're welcome! Is there anything else I can help you with?"
        elif any(word in action.lower() for word in ["goodbye", "bye", "see you"]):
            return "Goodbye! Have a great day!"
        else:
            return "I'm here to help with your schedule. You can ask me to add, delete, or search for events."
    
    elif intent['name'] == 'query':
        if date:
            if title:
                return f"I'll check for events titled '{title}' on {date}."
            else:
                return f"I'll check your schedule for {date}."
        elif title:
            return f"I'll check for events titled '{title}'."
        else:
            return "What date or event are you asking about?"
    # 默认回复
    return "I'm not sure I understand. Would you like to schedule, delete, or update an event?"