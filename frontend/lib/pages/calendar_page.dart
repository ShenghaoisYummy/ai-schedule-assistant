// lib/pages/calendar_page.dart
import 'package:flutter/material.dart';
import 'package:calendar_chatbot/models/event_model.dart';
import 'package:calendar_chatbot/database/database_helper.dart';
import 'package:intl/intl.dart';

class CalendarPage extends StatefulWidget {
  const CalendarPage({Key? key}) : super(key: key);

  @override
  _CalendarPageState createState() => _CalendarPageState();
}

class _CalendarPageState extends State<CalendarPage> {
  DateTime _selectedDate = DateTime.now();
  List<EventModel> _events = [];
  bool _isLoading = true;
  final DatabaseHelper _dbHelper = DatabaseHelper.instance;
  final DateFormat _dateFormat = DateFormat('yyyy-MM-dd');
  final DateFormat _displayDateFormat = DateFormat('EEE, MMM d');
  final DateFormat _timeFormat = DateFormat('h:mm a');

  @override
  void initState() {
    super.initState();
    _loadEvents();
  }

  Future<void> _loadEvents() async {
    setState(() {
      _isLoading = true;
    });

    try {
      final events = await _dbHelper.getAllEvents();
      setState(() {
        _events = events;
        _isLoading = false;
      });
    } catch (e) {
      print('Error loading events: $e');
      setState(() {
        _isLoading = false;
      });
    }
  }

  DateTime _getDateTimeFromEvent(EventModel event) {
    // Convert string date to DateTime object
    return DateTime.parse(event.date);
  }

  DateTime _getStartTimeFromEvent(EventModel event) {
    // Convert string time to DateTime object
    return DateTime.parse(event.startTime);
  }

  DateTime _getEndTimeFromEvent(EventModel event) {
    // Convert string time to DateTime object
    return DateTime.parse(event.endTime);
  }

  List<EventModel> _getEventsForSelectedDate() {
    final String selectedDateStr = _dateFormat.format(_selectedDate);
    return _events.where((event) => event.date == selectedDateStr).toList();
  }

  @override
  Widget build(BuildContext context) {
    final eventsForSelectedDate = _getEventsForSelectedDate();

    return Scaffold(
      appBar: AppBar(
        title: Text('Calendar'),
        actions: [
          IconButton(
            icon: Icon(Icons.refresh),
            onPressed: _loadEvents,
            tooltip: 'Refresh Events',
          ),
        ],
      ),
      body: Column(
        children: [
          _buildCalendarHeader(),
          Expanded(
            child: _isLoading
                ? Center(child: CircularProgressIndicator())
                : eventsForSelectedDate.isEmpty
                    ? _buildNoEventsMessage()
                    : _buildEventsList(eventsForSelectedDate),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () => _showAddEventDialog(context),
        child: Icon(Icons.add),
        tooltip: 'Add Event',
      ),
    );
  }

  Widget _buildCalendarHeader() {
    return Container(
      padding: EdgeInsets.all(16.0),
      decoration: BoxDecoration(
        color: Theme.of(context).primaryColor,
      ),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              IconButton(
                icon: Icon(Icons.chevron_left, color: Colors.white),
                onPressed: () {
                  setState(() {
                    _selectedDate = _selectedDate.subtract(Duration(days: 1));
                  });
                },
              ),
              Text(
                _displayDateFormat.format(_selectedDate),
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 20.0,
                  fontWeight: FontWeight.bold,
                ),
              ),
              IconButton(
                icon: Icon(Icons.chevron_right, color: Colors.white),
                onPressed: () {
                  setState(() {
                    _selectedDate = _selectedDate.add(Duration(days: 1));
                  });
                },
              ),
            ],
          ),
          SizedBox(height: 8.0),
          GestureDetector(
            onTap: () => _selectDate(context),
            child: Container(
              padding: EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.2),
                borderRadius: BorderRadius.circular(20.0),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.calendar_today, color: Colors.white, size: 16.0),
                  SizedBox(width: 8.0),
                  Text(
                    'Tap to select date',
                    style: TextStyle(color: Colors.white),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildNoEventsMessage() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.event_busy,
            size: 64.0,
            color: Colors.grey[400],
          ),
          SizedBox(height: 16.0),
          Text(
            'No events for this day',
            style: TextStyle(
              fontSize: 18.0,
              color: Colors.grey[600],
            ),
          ),
          SizedBox(height: 8.0),
          TextButton.icon(
            icon: Icon(Icons.add_circle_outline),
            label: Text('Add Event'),
            onPressed: () => _showAddEventDialog(context),
          ),
        ],
      ),
    );
  }

  Widget _buildEventsList(List<EventModel> events) {
    // Sort events by time
    events.sort((a, b) {
      if (a.isAllDay && !b.isAllDay) return -1;
      if (!a.isAllDay && b.isAllDay) return 1;
      return _getStartTimeFromEvent(a).compareTo(_getStartTimeFromEvent(b));
    });

    return ListView.builder(
      padding: EdgeInsets.all(8.0),
      itemCount: events.length,
      itemBuilder: (context, index) {
        final event = events[index];
        return _buildEventCard(event);
      },
    );
  }

  Widget _buildEventCard(EventModel event) {
    bool isAllDay = event.isAllDay;
    
    return Card(
      margin: EdgeInsets.symmetric(vertical: 8.0, horizontal: 4.0),
      elevation: 2.0,
      child: ListTile(
        contentPadding: EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
        leading: Container(
          padding: EdgeInsets.all(8.0),
          decoration: BoxDecoration(
            color: Theme.of(context).primaryColor.withOpacity(0.1),
            borderRadius: BorderRadius.circular(8.0),
          ),
          child: Icon(
            isAllDay ? Icons.calendar_today : Icons.access_time,
            color: Theme.of(context).primaryColor,
          ),
        ),
        title: Text(
          event.title,
          style: TextStyle(
            fontWeight: FontWeight.bold,
            fontSize: 16.0,
          ),
        ),
        subtitle: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(height: 4.0),
            if (isAllDay)
              Text(
                'All day',
                style: TextStyle(
                  color: Colors.grey[600],
                ),
              )
            else
              Text(
                '${_timeFormat.format(_getStartTimeFromEvent(event))} - ${_timeFormat.format(_getEndTimeFromEvent(event))}',
                style: TextStyle(
                  color: Colors.grey[600],
                ),
              ),
            if (event.location.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(top: 4.0),
                child: Row(
                  children: [
                    Icon(Icons.location_on, size: 14.0, color: Colors.grey[600]),
                    SizedBox(width: 4.0),
                    Expanded(
                      child: Text(
                        event.location,
                        style: TextStyle(
                          color: Colors.grey[600],
                          fontSize: 13.0,
                        ),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                  ],
                ),
              ),
            if (event.description.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(top: 4.0),
                child: Text(
                  event.description,
                  style: TextStyle(
                    fontSize: 13.0,
                  ),
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
          ],
        ),
        trailing: PopupMenuButton<String>(
          onSelected: (value) {
            if (value == 'edit') {
              _showEditEventDialog(context, event);
            } else if (value == 'delete') {
              _showDeleteConfirmation(context, event);
            }
          },
          itemBuilder: (context) => [
            PopupMenuItem(
              value: 'edit',
              child: ListTile(
                leading: Icon(Icons.edit),
                title: Text('Edit'),
                contentPadding: EdgeInsets.zero,
              ),
            ),
            PopupMenuItem(
              value: 'delete',
              child: ListTile(
                leading: Icon(Icons.delete, color: Colors.red),
                title: Text('Delete', style: TextStyle(color: Colors.red)),
                contentPadding: EdgeInsets.zero,
              ),
            ),
          ],
        ),
        onTap: () => _showEventDetails(context, event),
      ),
    );
  }

  Future<void> _selectDate(BuildContext context) async {
    final DateTime? picked = await showDatePicker(
      context: context,
      initialDate: _selectedDate,
      firstDate: DateTime(2020),
      lastDate: DateTime(2030),
    );

    if (picked != null && picked != _selectedDate) {
      setState(() {
        _selectedDate = picked;
      });
    }
  }

  void _showAddEventDialog(BuildContext context) {
    final TextEditingController titleController = TextEditingController();
    final TextEditingController descriptionController = TextEditingController();
    final TextEditingController locationController = TextEditingController();
    
    DateTime selectedDate = _selectedDate;
    TimeOfDay startTime = TimeOfDay(hour: 9, minute: 0);
    TimeOfDay endTime = TimeOfDay(hour: 10, minute: 0);
    bool isAllDay = false;

    showDialog(
      context: context,
      builder: (BuildContext context) {
        return StatefulBuilder(
          builder: (context, setState) {
            return AlertDialog(
              title: Text('Add Event'),
              content: SingleChildScrollView(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    TextField(
                      controller: titleController,
                      decoration: InputDecoration(
                        labelText: 'Title',
                        hintText: 'Enter event title',
                      ),
                    ),
                    SizedBox(height: 16.0),
                    Row(
                      children: [
                        Icon(Icons.calendar_today, size: 20.0),
                        SizedBox(width: 8.0),
                        Text(
                          _displayDateFormat.format(selectedDate),
                          style: TextStyle(fontSize: 16.0),
                        ),
                        Spacer(),
                        TextButton(
                          onPressed: () async {
                            final DateTime? picked = await showDatePicker(
                              context: context,
                              initialDate: selectedDate,
                              firstDate: DateTime(2020),
                              lastDate: DateTime(2030),
                            );
                            if (picked != null) {
                              setState(() {
                                selectedDate = picked;
                              });
                            }
                          },
                          child: Text('Change'),
                        ),
                      ],
                    ),
                    SwitchListTile(
                      title: Text('All day'),
                      value: isAllDay,
                      onChanged: (value) {
                        setState(() {
                          isAllDay = value;
                        });
                      },
                      contentPadding: EdgeInsets.zero,
                    ),
                    if (!isAllDay) ...[
                      Row(
                        children: [
                          Icon(Icons.access_time, size: 20.0),
                          SizedBox(width: 8.0),
                          Text('Start: ${startTime.format(context)}'),
                          Spacer(),
                          TextButton(
                            onPressed: () async {
                              final TimeOfDay? picked = await showTimePicker(
                                context: context,
                                initialTime: startTime,
                              );
                              if (picked != null) {
                                setState(() {
                                  startTime = picked;
                                  // If end time is before start time, adjust it
                                  if (_timeToMinutes(endTime) < _timeToMinutes(startTime)) {
                                    endTime = TimeOfDay(
                                      hour: (startTime.hour + 1) % 24,
                                      minute: startTime.minute,
                                    );
                                  }
                                });
                              }
                            },
                            child: Text('Change'),
                          ),
                        ],
                      ),
                      Row(
                        children: [
                          Icon(Icons.access_time, size: 20.0),
                          SizedBox(width: 8.0),
                          Text('End: ${endTime.format(context)}'),
                          Spacer(),
                          TextButton(
                            onPressed: () async {
                              final TimeOfDay? picked = await showTimePicker(
                                context: context,
                                initialTime: endTime,
                              );
                              if (picked != null) {
                                setState(() {
                                  endTime = picked;
                                });
                              }
                            },
                            child: Text('Change'),
                          ),
                        ],
                      ),
                    ],
                    SizedBox(height: 16.0),
                    TextField(
                      controller: locationController,
                      decoration: InputDecoration(
                        labelText: 'Location',
                        hintText: 'Enter location (optional)',
                      ),
                    ),
                    SizedBox(height: 16.0),
                    TextField(
                      controller: descriptionController,
                      decoration: InputDecoration(
                        labelText: 'Description',
                        hintText: 'Enter description (optional)',
                      ),
                      maxLines: 3,
                    ),
                  ],
                ),
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.of(context).pop(),
                  child: Text('Cancel'),
                ),
                ElevatedButton(
                  onPressed: () async {
                    if (titleController.text.trim().isEmpty) {
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(content: Text('Please enter a title')),
                      );
                      return;
                    }

                    final startDateTime = _combineDateAndTime(selectedDate, startTime);
                    final endDateTime = _combineDateAndTime(selectedDate, endTime);

                    // Create event model - Note that dates and times use string format
                    final event = EventModel(
                      id: 0, // Database will assign ID
                      title: titleController.text.trim(),
                      description: descriptionController.text.trim(),
                      date: _dateFormat.format(selectedDate), // Convert to string format
                      startTime: isAllDay 
                          ? _combineDateAndTime(selectedDate, TimeOfDay(hour: 0, minute: 0)).toIso8601String()
                          : startDateTime.toIso8601String(),
                      endTime: isAllDay 
                          ? _combineDateAndTime(selectedDate, TimeOfDay(hour: 23, minute: 59)).toIso8601String()
                          : endDateTime.toIso8601String(),
                      location: locationController.text.trim(),
                      isAllDay: isAllDay,
                    );

                    await _dbHelper.insertEvent(event);
                    Navigator.of(context).pop();
                    _loadEvents();
                  },
                  child: Text('Save'),
                ),
              ],
            );
          },
        );
      },
    );
  }

  void _showEditEventDialog(BuildContext context, EventModel event) {
    final TextEditingController titleController = TextEditingController(text: event.title);
    final TextEditingController descriptionController = TextEditingController(text: event.description);
    final TextEditingController locationController = TextEditingController(text: event.location);
    
    // Convert string date to DateTime
    DateTime selectedDate = DateTime.parse(event.date);
    
    // Extract TimeOfDay from string time
    DateTime startDateTime = DateTime.parse(event.startTime);
    TimeOfDay startTime = TimeOfDay(
      hour: startDateTime.hour,
      minute: startDateTime.minute,
    );
    
    DateTime endDateTime = DateTime.parse(event.endTime);
    TimeOfDay endTime = TimeOfDay(
      hour: endDateTime.hour,
      minute: endDateTime.minute,
    );
    
    bool isAllDay = event.isAllDay;

    showDialog(
      context: context,
      builder: (BuildContext context) {
        return StatefulBuilder(
          builder: (context, setState) {
            return AlertDialog(
              title: Text('Edit Event'),
              content: SingleChildScrollView(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    TextField(
                      controller: titleController,
                      decoration: InputDecoration(
                        labelText: 'Title',
                        hintText: 'Enter event title',
                      ),
                    ),
                    SizedBox(height: 16.0),
                    Row(
                      children: [
                        Icon(Icons.calendar_today, size: 20.0),
                        SizedBox(width: 8.0),
                        Text(
                          _displayDateFormat.format(selectedDate),
                          style: TextStyle(fontSize: 16.0),
                        ),
                        Spacer(),
                        TextButton(
                          onPressed: () async {
                            final DateTime? picked = await showDatePicker(
                              context: context,
                              initialDate: selectedDate,
                              firstDate: DateTime(2020),
                              lastDate: DateTime(2030),
                            );
                            if (picked != null) {
                              setState(() {
                                selectedDate = picked;
                              });
                            }
                          },
                          child: Text('Change'),
                        ),
                      ],
                    ),
                    SwitchListTile(
                      title: Text('All day'),
                      value: isAllDay,
                      onChanged: (value) {
                        setState(() {
                          isAllDay = value;
                        });
                      },
                      contentPadding: EdgeInsets.zero,
                    ),
                    if (!isAllDay) ...[
                      Row(
                        children: [
                          Icon(Icons.access_time, size: 20.0),
                          SizedBox(width: 8.0),
                          Text('Start: ${startTime.format(context)}'),
                          Spacer(),
                          TextButton(
                            onPressed: () async {
                              final TimeOfDay? picked = await showTimePicker(
                                context: context,
                                initialTime: startTime,
                              );
                              if (picked != null) {
                                setState(() {
                                  startTime = picked;
                                  // If end time is before start time, adjust it
                                  if (_timeToMinutes(endTime) < _timeToMinutes(startTime)) {
                                    endTime = TimeOfDay(
                                      hour: (startTime.hour + 1) % 24,
                                      minute: startTime.minute,
                                    );
                                  }
                                });
                              }
                            },
                            child: Text('Change'),
                          ),
                        ],
                      ),
                      Row(
                        children: [
                          Icon(Icons.access_time, size: 20.0),
                          SizedBox(width: 8.0),
                          Text('End: ${endTime.format(context)}'),
                          Spacer(),
                          TextButton(
                            onPressed: () async {
                              final TimeOfDay? picked = await showTimePicker(
                                context: context,
                                initialTime: endTime,
                              );
                              if (picked != null) {
                                setState(() {
                                  endTime = picked;
                                });
                              }
                            },
                            child: Text('Change'),
                          ),
                        ],
                      ),
                    ],
                    SizedBox(height: 16.0),
                    TextField(
                      controller: locationController,
                      decoration: InputDecoration(
                        labelText: 'Location',
                        hintText: 'Enter location (optional)',
                      ),
                    ),
                    SizedBox(height: 16.0),
                    TextField(
                      controller: descriptionController,
                      decoration: InputDecoration(
                        labelText: 'Description',
                        hintText: 'Enter description (optional)',
                      ),
                      maxLines: 3,
                    ),
                  ],
                ),
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.of(context).pop(),
                  child: Text('Cancel'),
                ),
                ElevatedButton(
                  onPressed: () async {
                    if (titleController.text.trim().isEmpty) {
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(content: Text('Please enter a title')),
                      );
                      return;
                    }

                    final startDateTime = _combineDateAndTime(selectedDate, startTime);
                    final endDateTime = _combineDateAndTime(selectedDate, endTime);

                    // Update event model - Note that dates and times use string format
                    final updatedEvent = EventModel(
                      id: event.id,
                      title: titleController.text.trim(),
                      description: descriptionController.text.trim(),
                      date: _dateFormat.format(selectedDate), // Convert to string format
                      startTime: isAllDay 
                          ? _combineDateAndTime(selectedDate, TimeOfDay(hour: 0, minute: 0)).toIso8601String()
                          : startDateTime.toIso8601String(),
                      endTime: isAllDay 
                          ? _combineDateAndTime(selectedDate, TimeOfDay(hour: 23, minute: 59)).toIso8601String()
                          : endDateTime.toIso8601String(),
                      location: locationController.text.trim(),
                      isAllDay: isAllDay,
                    );

                    await _dbHelper.updateEvent(updatedEvent);
                    Navigator.of(context).pop();
                    _loadEvents();
                  },
                  child: Text('Save'),
                ),
              ],
            );
          },
        );
      },
    );
  }

  void _showDeleteConfirmation(BuildContext context, EventModel event) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text('Delete Event'),
          content: Text('Are you sure you want to delete "${event.title}"?'),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: Text('Cancel'),
            ),
            TextButton(
              onPressed: () async {
                await _dbHelper.deleteEvent(event.id);
                Navigator.of(context).pop();
                _loadEvents();
              },
              child: Text('Delete', style: TextStyle(color: Colors.red)),
            ),
          ],
        );
      },
    );
  }

  void _showEventDetails(BuildContext context, EventModel event) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(16.0)),
      ),
      builder: (BuildContext context) {
        return Container(
          padding: EdgeInsets.all(16.0),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Flexible(
                    child: Text(
                      event.title,
                      style: TextStyle(
                        fontSize: 20.0,
                        fontWeight: FontWeight.bold,
                      ),
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  IconButton(
                    icon: Icon(Icons.close),
                    onPressed: () => Navigator.of(context).pop(),
                  ),
                ],
              ),
              Divider(),
              SizedBox(height: 8.0),
              Row(
                children: [
                  Icon(Icons.calendar_today, color: Theme.of(context).primaryColor),
                  SizedBox(width: 8.0),
                  Text(
                    _displayDateFormat.format(DateTime.parse(event.date)),
                    style: TextStyle(fontSize: 16.0),
                  ),
                ],
              ),
              SizedBox(height: 16.0),
              Row(
                children: [
                  Icon(Icons.access_time, color: Theme.of(context).primaryColor),
                  SizedBox(width: 8.0),
                  Text(
                    event.isAllDay
                        ? 'All day'
                        : '${_timeFormat.format(DateTime.parse(event.startTime))} - ${_timeFormat.format(DateTime.parse(event.endTime))}',
                    style: TextStyle(fontSize: 16.0),
                  ),
                ],
              ),
              if (event.location.isNotEmpty) ...[
                SizedBox(height: 16.0),
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(Icons.location_on, color: Theme.of(context).primaryColor),
                    SizedBox(width: 8.0),
                    Expanded(
                      child: Text(
                        event.location,
                        style: TextStyle(fontSize: 16.0),
                      ),
                    ),
                  ],
                ),
              ],
              if (event.description.isNotEmpty) ...[
                SizedBox(height: 16.0),
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(Icons.notes, color: Theme.of(context).primaryColor),
                    SizedBox(width: 8.0),
                    Expanded(
                      child: Text(
                        event.description,
                        style: TextStyle(fontSize: 16.0),
                      ),
                    ),
                  ],
                ),
              ],
              SizedBox(height: 24.0),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  OutlinedButton.icon(
                    icon: Icon(Icons.edit),
                    label: Text('Edit'),
                    onPressed: () {
                      Navigator.of(context).pop();
                      _showEditEventDialog(context, event);
                    },
                  ),
                  OutlinedButton.icon(
                    icon: Icon(Icons.delete, color: Colors.red),
                    label: Text('Delete', style: TextStyle(color: Colors.red)),
                    onPressed: () {
                      Navigator.of(context).pop();
                      _showDeleteConfirmation(context, event);
                    },
                  ),
                ],
              ),
              SizedBox(height: 16.0),
            ],
          ),
        );
      },
    );
  }

  // Helper method: Combine date and time into DateTime object
  DateTime _combineDateAndTime(DateTime date, TimeOfDay time) {
    return DateTime(
      date.year,
      date.month,
      date.day,
      time.hour,
      time.minute,
    );
  }

  // Helper method: Convert TimeOfDay to minutes for comparison
  int _timeToMinutes(TimeOfDay time) {
    return time.hour * 60 + time.minute;
  }
}