// lib/models/message_model.dart
import 'dart:convert';

class MessageModel {
  final int id;
  final String text;
  final bool isUser;
  final DateTime timestamp;
  final String? intent;
  final List<dynamic>? entities;

  MessageModel({
    required this.id,
    required this.text,
    required this.isUser,
    required this.timestamp,
    this.intent,
    this.entities,
  });

  // Create an instance from database Map
  factory MessageModel.fromMap(Map<String, dynamic> map) {
    // Process timestamp
    DateTime parsedTimestamp;
    try {
      parsedTimestamp = DateTime.parse(map['timestamp']);
    } catch (e) {
      parsedTimestamp = DateTime.now();
      print('Error parsing timestamp: $e');
    }
    
    // Process optional fields
    String? intent;
    List<dynamic>? entities;
    
    if (map.containsKey('intent') && map['intent'] != null) {
      intent = map['intent'];
    }
    
    if (map.containsKey('entities') && map['entities'] != null) {
      try {
        if (map['entities'] is String) {
          entities = jsonDecode(map['entities']);
        } else if (map['entities'] is List) {
          entities = map['entities'];
        }
      } catch (e) {
        print('Error parsing entities: $e');
      }
    }
    
    return MessageModel(
      id: map['id'],
      text: map['text'],
      isUser: map['isUser'] == 1,
      timestamp: parsedTimestamp,
      intent: intent,
      entities: entities,
    );
  }

  // Convert to database Map
  Map<String, dynamic> toMap() {
    final Map<String, dynamic> map = {
      'id': id,
      'text': text,
      'isUser': isUser ? 1 : 0,
      'timestamp': timestamp.toIso8601String(),
    };
    
    // These extra fields may not be stored in the database
    // but are used in the memory model
    if (intent != null) {
      map['intent'] = intent;
    }
    
    if (entities != null) {
      map['entities'] = jsonEncode(entities);
    }
    
    return map;
  }

  @override
  String toString() {
    return 'Message{id: $id, text: $text, isUser: $isUser, timestamp: $timestamp}';
  }
}