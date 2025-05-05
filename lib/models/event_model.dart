// lib/models/event_model.dart
class EventModel {
  final int id;
  final String title;
  final String description;
  final DateTime date;      // This stores both date and time
  final DateTime startTime; // Specific start time
  final DateTime endTime;   // Specific end time
  final String location;    // Optional location field
  final bool isAllDay;      // Flag for all-day events

  EventModel({
    required this.id,
    required this.title,
    required this.description,
    required this.date,
    required this.startTime,
    required this.endTime,
    this.location = '',
    this.isAllDay = false,
  });

  // From JSON map to model
  factory EventModel.fromMap(Map<String, dynamic> map) {
    return EventModel(
      id: map['id'],
      title: map['title'],
      description: map['description'],
      date: DateTime.parse(map['date']),
      startTime: DateTime.parse(map['startTime']),
      endTime: DateTime.parse(map['endTime']),
      location: map['location'] ?? '',
      isAllDay: map['isAllDay'] == 1,
    );
  }

  // From model to JSON map
  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'title': title,
      'description': description,
      'date': date.toIso8601String(),
      'startTime': startTime.toIso8601String(),
      'endTime': endTime.toIso8601String(),
      'location': location,
      'isAllDay': isAllDay ? 1 : 0,
    };
  }

  // Create a copy of this event with modified fields
  EventModel copyWith({
    int? id,
    String? title,
    String? description,
    DateTime? date,
    DateTime? startTime,
    DateTime? endTime,
    String? location,
    bool? isAllDay,
  }) {
    return EventModel(
      id: id ?? this.id,
      title: title ?? this.title,
      description: description ?? this.description,
      date: date ?? this.date,
      startTime: startTime ?? this.startTime,
      endTime: endTime ?? this.endTime,
      location: location ?? this.location,
      isAllDay: isAllDay ?? this.isAllDay,
    );
  }
}