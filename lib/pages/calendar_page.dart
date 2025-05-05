// lib/pages/calendar_page.dart
import 'package:flutter/material.dart';
import 'package:table_calendar/table_calendar.dart';
import 'package:intl/intl.dart';
import 'package:calendar_chatbot/models/event_model.dart';
import 'package:calendar_chatbot/database/database_helper.dart';

class CalendarPage extends StatefulWidget {
  const CalendarPage({Key? key}) : super(key: key);

  @override
  _CalendarPageState createState() => _CalendarPageState();
}

class _CalendarPageState extends State<CalendarPage> {
  CalendarFormat _calendarFormat = CalendarFormat.week;
  DateTime _focusedDay = DateTime.now();
  DateTime? _selectedDay;
  Map<DateTime, List<EventModel>> _events = {};
  bool _isLoading = true;

  // Time format
  final timeFormat = DateFormat('h:mm a');
  final dateFormat = DateFormat('EEE, MMM d, yyyy');

  @override
  void initState() {
    super.initState();
    _selectedDay = _focusedDay;
    _loadEvents();
  }

  // Load events from database
  Future<void> _loadEvents() async {
    setState(() {
      _isLoading = true;
    });

    final events = await DatabaseHelper.instance.getAllEvents();
    final Map<DateTime, List<EventModel>> eventMap = {};

    for (var event in events) {
      final day = DateTime(
        event.date.year,
        event.date.month,
        event.date.day,
      );

      if (eventMap[day] != null) {
        eventMap[day]!.add(event);
      } else {
        eventMap[day] = [event];
      }
    }

    setState(() {
      _events = eventMap;
      _isLoading = false;
    });
  }

  List<EventModel> _getEventsForDay(DateTime day) {
    return _events[DateTime(day.year, day.month, day.day)] ?? [];
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Calendar'),
        actions: [
          IconButton(
            icon: const Icon(Icons.add),
            onPressed: () => _showAddEventDialog(),
          ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : Column(
              children: [
                TableCalendar(
                  firstDay: DateTime.utc(2020, 1, 1),
                  lastDay: DateTime.utc(2030, 12, 31),
                  focusedDay: _focusedDay,
                  calendarFormat: _calendarFormat,
                  eventLoader: _getEventsForDay,
                  startingDayOfWeek: StartingDayOfWeek.monday,
                  calendarStyle: const CalendarStyle(
                    outsideDaysVisible: false,
                    markersMaxCount: 3,
                  ),
                  selectedDayPredicate: (day) {
                    return isSameDay(_selectedDay, day);
                  },
                  onDaySelected: (selectedDay, focusedDay) {
                    if (!isSameDay(_selectedDay, selectedDay)) {
                      setState(() {
                        _selectedDay = selectedDay;
                        _focusedDay = focusedDay;
                      });
                    }
                  },
                  onFormatChanged: (format) {
                    if (_calendarFormat != format) {
                      setState(() {
                        _calendarFormat = format;
                      });
                    }
                  },
                  onPageChanged: (focusedDay) {
                    _focusedDay = focusedDay;
                  },
                  availableCalendarFormats: const {
                    CalendarFormat.month: 'Month view',
                    CalendarFormat.week: 'Week view',
                  },
                ),
                const SizedBox(height: 8.0),
                Expanded(
                  child: _buildEventList(),
                ),
              ],
            ),
    );
  }

  Widget _buildEventList() {
    final events = _getEventsForDay(_selectedDay!);

    // Sort events by start time
    events.sort((a, b) => a.startTime.compareTo(b.startTime));

    return events.isEmpty
        ? const Center(
            child: Text('No events for today'),
          )
        : ListView.builder(
            itemCount: events.length,
            itemBuilder: (context, index) {
              final event = events[index];
              return Card(
                margin: const EdgeInsets.symmetric(
                  horizontal: 12.0,
                  vertical: 4.0,
                ),
                child: ListTile(
                  leading: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(
                        event.isAllDay ? Icons.calendar_today : Icons.access_time,
                        color: Colors.blue,
                      ),
                      if (!event.isAllDay)
                        Text(
                          timeFormat.format(event.startTime),
                          style: TextStyle(
                            fontSize: 12,
                            color: Colors.grey[600],
                          ),
                        ),
                    ],
                  ),
                  title: Text(event.title),
                  subtitle: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(event.description),
                      const SizedBox(height: 4),
                      if (!event.isAllDay)
                        Text(
                          '${timeFormat.format(event.startTime)} - ${timeFormat.format(event.endTime)}',
                          style: TextStyle(
                            fontSize: 12,
                            color: Colors.grey[600],
                          ),
                        ),
                      if (event.location.isNotEmpty)
                        Text(
                          'Location: ${event.location}',
                          style: TextStyle(
                            fontSize: 12,
                            color: Colors.grey[600],
                          ),
                        ),
                    ],
                  ),
                  trailing: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      IconButton(
                        icon: const Icon(Icons.edit, color: Colors.blue),
                        onPressed: () => _showEditEventDialog(event),
                      ),
                      IconButton(
                        icon: const Icon(Icons.delete, color: Colors.red),
                        onPressed: () => _deleteEvent(event),
                      ),
                    ],
                  ),
                ),
              );
            },
          );
  }

  void _showAddEventDialog() {
    final titleController = TextEditingController();
    final descriptionController = TextEditingController();
    final locationController = TextEditingController();
    
    DateTime selectedDate = _selectedDay!;
    TimeOfDay startTime = TimeOfDay.now();
    TimeOfDay endTime = TimeOfDay(
      hour: startTime.hour + 1,
      minute: startTime.minute,
    );
    bool isAllDay = false;

    showDialog(
      context: context,
      builder: (context) => StatefulBuilder(
        builder: (context, setState) => AlertDialog(
          title: const Text('Add New Event'),
          content: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                TextField(
                  controller: titleController,
                  decoration: const InputDecoration(labelText: 'Title'),
                ),
                TextField(
                  controller: descriptionController,
                  decoration: const InputDecoration(labelText: 'Description'),
                  maxLines: 3,
                ),
                TextField(
                  controller: locationController,
                  decoration: const InputDecoration(labelText: 'Location (optional)'),
                ),
                const SizedBox(height: 16),
                Row(
                  children: [
                    const Text('Date: '),
                    TextButton(
                      onPressed: () async {
                        final pickedDate = await showDatePicker(
                          context: context,
                          initialDate: selectedDate,
                          firstDate: DateTime(2020),
                          lastDate: DateTime(2030),
                        );
                        if (pickedDate != null) {
                          setState(() {
                            selectedDate = pickedDate;
                          });
                        }
                      },
                      child: Text(dateFormat.format(selectedDate)),
                    ),
                  ],
                ),
                Row(
                  children: [
                    Checkbox(
                      value: isAllDay,
                      onChanged: (value) {
                        setState(() {
                          isAllDay = value ?? false;
                        });
                      },
                    ),
                    const Text('All-day event'),
                  ],
                ),
                if (!isAllDay) ...[
                  Row(
                    children: [
                      const Text('Start Time: '),
                      TextButton(
                        onPressed: () async {
                          final pickedTime = await showTimePicker(
                            context: context,
                            initialTime: startTime,
                          );
                          if (pickedTime != null) {
                            setState(() {
                              startTime = pickedTime;
                              // If end time is earlier than start time, adjust it
                              if (_timeToMinutes(endTime) <= _timeToMinutes(startTime)) {
                                endTime = TimeOfDay(
                                  hour: (startTime.hour + 1) % 24,
                                  minute: startTime.minute,
                                );
                              }
                            });
                          }
                        },
                        child: Text(_formatTimeOfDay(startTime)),
                      ),
                    ],
                  ),
                  Row(
                    children: [
                      const Text('End Time: '),
                      TextButton(
                        onPressed: () async {
                          final pickedTime = await showTimePicker(
                            context: context,
                            initialTime: endTime,
                          );
                          if (pickedTime != null) {
                            setState(() {
                              endTime = pickedTime;
                            });
                          }
                        },
                        child: Text(_formatTimeOfDay(endTime)),
                      ),
                    ],
                  ),
                ],
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Cancel'),
            ),
            TextButton(
              onPressed: () async {
                if (titleController.text.isNotEmpty) {
                  // Create start and end DateTime objects
                  final startDateTime = _combineDateAndTime(selectedDate, startTime);
                  final endDateTime = _combineDateAndTime(selectedDate, endTime);
                  
                  final newEvent = EventModel(
                    id: 0, // Auto-increment
                    title: titleController.text,
                    description: descriptionController.text,
                    date: selectedDate,
                    startTime: isAllDay ? 
                      DateTime(selectedDate.year, selectedDate.month, selectedDate.day) : 
                      startDateTime,
                    endTime: isAllDay ? 
                      DateTime(selectedDate.year, selectedDate.month, selectedDate.day, 23, 59) : 
                      endDateTime,
                    location: locationController.text,
                    isAllDay: isAllDay,
                  );

                  await DatabaseHelper.instance.insertEvent(newEvent);
                  await _loadEvents(); // Reload events
                  Navigator.pop(context);
                }
              },
              child: const Text('Save'),
            ),
          ],
        ),
      ),
    );
  }

  void _showEditEventDialog(EventModel event) {
    final titleController = TextEditingController(text: event.title);
    final descriptionController = TextEditingController(text: event.description);
    final locationController = TextEditingController(text: event.location);
    
    DateTime selectedDate = event.date;
    TimeOfDay startTime = TimeOfDay(
      hour: event.startTime.hour,
      minute: event.startTime.minute,
    );
    TimeOfDay endTime = TimeOfDay(
      hour: event.endTime.hour,
      minute: event.endTime.minute,
    );
    bool isAllDay = event.isAllDay;

    showDialog(
      context: context,
      builder: (context) => StatefulBuilder(
        builder: (context, setState) => AlertDialog(
          title: const Text('Edit Event'),
          content: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                TextField(
                  controller: titleController,
                  decoration: const InputDecoration(labelText: 'Title'),
                ),
                TextField(
                  controller: descriptionController,
                  decoration: const InputDecoration(labelText: 'Description'),
                  maxLines: 3,
                ),
                TextField(
                  controller: locationController,
                  decoration: const InputDecoration(labelText: 'Location (optional)'),
                ),
                const SizedBox(height: 16),
                Row(
                  children: [
                    const Text('Date: '),
                    TextButton(
                      onPressed: () async {
                        final pickedDate = await showDatePicker(
                          context: context,
                          initialDate: selectedDate,
                          firstDate: DateTime(2020),
                          lastDate: DateTime(2030),
                        );
                        if (pickedDate != null) {
                          setState(() {
                            selectedDate = pickedDate;
                          });
                        }
                      },
                      child: Text(dateFormat.format(selectedDate)),
                    ),
                  ],
                ),
                Row(
                  children: [
                    Checkbox(
                      value: isAllDay,
                      onChanged: (value) {
                        setState(() {
                          isAllDay = value ?? false;
                        });
                      },
                    ),
                    const Text('All-day event'),
                  ],
                ),
                if (!isAllDay) ...[
                  Row(
                    children: [
                      const Text('Start Time: '),
                      TextButton(
                        onPressed: () async {
                          final pickedTime = await showTimePicker(
                            context: context,
                            initialTime: startTime,
                          );
                          if (pickedTime != null) {
                            setState(() {
                              startTime = pickedTime;
                              // If end time is earlier than start time, adjust it
                              if (_timeToMinutes(endTime) <= _timeToMinutes(startTime)) {
                                endTime = TimeOfDay(
                                  hour: (startTime.hour + 1) % 24,
                                  minute: startTime.minute,
                                );
                              }
                            });
                          }
                        },
                        child: Text(_formatTimeOfDay(startTime)),
                      ),
                    ],
                  ),
                  Row(
                    children: [
                      const Text('End Time: '),
                      TextButton(
                        onPressed: () async {
                          final pickedTime = await showTimePicker(
                            context: context,
                            initialTime: endTime,
                          );
                          if (pickedTime != null) {
                            setState(() {
                              endTime = pickedTime;
                            });
                          }
                        },
                        child: Text(_formatTimeOfDay(endTime)),
                      ),
                    ],
                  ),
                ],
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Cancel'),
            ),
            TextButton(
              onPressed: () async {
                if (titleController.text.isNotEmpty) {
                  // Create start and end DateTime objects
                  final startDateTime = _combineDateAndTime(selectedDate, startTime);
                  final endDateTime = _combineDateAndTime(selectedDate, endTime);
                  
                  final updatedEvent = EventModel(
                    id: event.id,
                    title: titleController.text,
                    description: descriptionController.text,
                    date: selectedDate,
                    startTime: isAllDay ? 
                      DateTime(selectedDate.year, selectedDate.month, selectedDate.day) : 
                      startDateTime,
                    endTime: isAllDay ? 
                      DateTime(selectedDate.year, selectedDate.month, selectedDate.day, 23, 59) : 
                      endDateTime,
                    location: locationController.text,
                    isAllDay: isAllDay,
                  );

                  await DatabaseHelper.instance.updateEvent(updatedEvent);
                  await _loadEvents(); // Reload events
                  Navigator.pop(context);
                }
              },
              child: const Text('Save'),
            ),
          ],
        ),
      ),
    );
  }

  void _deleteEvent(EventModel event) async {
    // Show confirmation dialog
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete Event'),
        content: Text('Are you sure you want to delete "${event.title}"?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () async {
              await DatabaseHelper.instance.deleteEvent(event.id);
              await _loadEvents(); // Reload events
              Navigator.pop(context);
            },
            child: const Text('Delete'),
          ),
        ],
      ),
    );
  }

  // Helper methods for time handling
  String _formatTimeOfDay(TimeOfDay time) {
    final now = DateTime.now();
    final dateTime = DateTime(
      now.year,
      now.month,
      now.day,
      time.hour,
      time.minute,
    );
    return timeFormat.format(dateTime);
  }

  int _timeToMinutes(TimeOfDay time) {
    return time.hour * 60 + time.minute;
  }

  DateTime _combineDateAndTime(DateTime date, TimeOfDay time) {
    return DateTime(
      date.year,
      date.month,
      date.day,
      time.hour,
      time.minute,
    );
  }
}