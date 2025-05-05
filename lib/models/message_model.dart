// lib/models/message_model.dart
class MessageModel {
  final int id;
  final String text;
  final bool isUser;
  final DateTime timestamp;

  MessageModel({
    required this.id,
    required this.text,
    required this.isUser,
    required this.timestamp,
  });

  // From JSON map to model
  factory MessageModel.fromMap(Map<String, dynamic> map) {
    return MessageModel(
      id: map['id'],
      text: map['text'],
      isUser: map['isUser'] == 1,
      timestamp: DateTime.parse(map['timestamp']),
    );
  }

  // From model to JSON map
  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'text': text,
      'isUser': isUser ? 1 : 0,
      'timestamp': timestamp.toIso8601String(),
    };
  }
}