// lib/models/event_model.dart
class EventModel {
  final int id;
  final String title;
  final String description;
  final String date;  // ISO format date string YYYY-MM-DD
  final String startTime;  // ISO format date-time string
  final String endTime;  // ISO format date-time string
  final String location;
  final bool isAllDay;

  EventModel({
    required this.id,
    required this.title,
    required this.description,
    required this.date,
    required this.startTime,
    required this.endTime,
    required this.location,
    required this.isAllDay,
  });

  // Create an instance from database Map
  factory EventModel.fromMap(Map<String, dynamic> map) {
    return EventModel(
      id: map['id'],
      title: map['title'],
      description: map['description'] ?? '',
      date: map['date'],
      startTime: map['startTime'],
      endTime: map['endTime'],
      location: map['location'] ?? '',
      isAllDay: map['isAllDay'] == 1,
    );
  }

  // Convert to database Map
  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'title': title,
      'description': description,
      'date': date,
      'startTime': startTime,
      'endTime': endTime,
      'location': location,
      'isAllDay': isAllDay ? 1 : 0,
    };
  }

  @override
  String toString() {
    return 'Event{id: $id, title: $title, date: $date}';
  }
}