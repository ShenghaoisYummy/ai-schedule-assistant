// lib/utils/chatbot_helper.dart
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:calendar_chatbot/database/database_helper.dart';
import 'package:calendar_chatbot/models/event_model.dart';
import 'package:calendar_chatbot/models/message_model.dart';
import 'package:intl/intl.dart';
import 'package:flutter/material.dart' show TimeOfDay;
import 'package:calendar_chatbot/api_config.dart';


class ChatbotHelper {
  // API server address - Use relative URL in production
  // static String apiBaseUrl = kIsWeb && kDebugMode
  //     ? 'http://localhost:8080/api'  // Web platform development
  //     : ApiConfig.baseUrl;  // Production or mobile
  static String apiBaseUrl = ApiConfig.baseUrl
  // Database helper
  static final DatabaseHelper _dbHelper = DatabaseHelper.instance;

  // Analyze text and get intent and entities
  static Future<Map<String, dynamic>> analyzeText(String text) async {
    try {
      final response = await http.post(
        Uri.parse('${apiBaseUrl}/analyze'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'text': text}),
      ).timeout(Duration(seconds: 10)); // Set timeout
      
      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        print('API error: ${response.statusCode} - ${response.body}');
        throw Exception('Failed to analyze text: ${response.statusCode}');
      }
    } catch (e) {
      print('Error analyzing text: $e');
      throw Exception('Network error: $e');
    }
  }

  // Health check - Test if API connection is working
  static Future<bool> checkHealth() async {
    try {
      final response = await http.get(
        Uri.parse('${apiBaseUrl}/health'),
      ).timeout(Duration(seconds: 5));
      
      return response.statusCode == 200;
    } catch (e) {
      print('Health check failed: $e');
      return false;
    }
  }

  // Parse response and extract key information
  static Map<String, dynamic> parseResponse(Map<String, dynamic> apiResponse) {
    final intent = apiResponse['intent']['name'] as String;
    final entities = apiResponse['entities'] as List<dynamic>;
    final response = apiResponse['response'] as String;
    
    // Extract key entities
    String? action, title, date, location, startTime, endTime, description;
    
    for (var entity in entities) {
      String type = entity['type'] as String;
      // Remove B- or I- prefix
      if (type.startsWith('B-')) {
        type = type.substring(2);
      } else if (type.startsWith('I-')) {
        continue; // Skip internal tags, already handled by B tag
      }
      
      String text = entity['text'] as String;
      
      switch(type.toLowerCase()) {
        case 'action':
          action = text;
          break;
        case 'title':
          title = text;
          break;
        case 'date':
          date = text;
          break;
        case 'location':
          location = text;
          break;
        case 'starttime':
          startTime = text;
          break;
        case 'endtime':
          endTime = text;
          break;
        case 'description':
          description = text;
          break;
      }
    }
    
    return {
      'intent': intent,
      'response': response,
      'entities': entities,
      'action': action,
      'title': title,
      'date': date,
      'location': location,
      'startTime': startTime,
      'endTime': endTime,
      'description': description,
    };
  }

  // Process different types of commands based on intent
  static Future<bool> processIntent(String intent, Map<String, dynamic> parsedData) async {
    try {
      // Execute different actions based on intent
      switch (intent) {
        case 'add':
          // Create calendar event
          await _createCalendarEvent(parsedData);
          return true;
        case 'delete':
          // Delete calendar event
          await _deleteCalendarEvent(parsedData);
          return true;
        case 'update':
          // Update calendar event
          await _updateCalendarEvent(parsedData);
          return true;
        case 'query':
          await _queryCalendarEvents(parsedData);
          return true;
        case 'chitchat':
          // Chitchat doesn't require any calendar operations
          return true;
        default:
          print('Unknown intent: $intent');
          return false;
      }
    } catch (e) {
      print('Error processing intent: $e');
      return false;
    }
  }
  
  // Save message history
  static Future<int> saveMessage(String text, bool isUser) async {
    final message = MessageModel(
      id: 0, // Database will automatically assign ID
      text: text,
      isUser: isUser,
      timestamp: DateTime.now(),
    );
    
    return await _dbHelper.insertMessage(message);
  }
  
  // Get all message history
  static Future<List<MessageModel>> getAllMessages() async {
    return await _dbHelper.getAllMessages();
  }
  
  // Clear message history
  static Future<int> clearMessages() async {
    try {
      final count = await _dbHelper.clearAllMessages();
      return 0;
    } catch (e) {
      print('Error in clearMessages: $e');
      return 0;
    }
  }

  // Create calendar event
  static Future<void> _createCalendarEvent(Map<String, dynamic> data) async {
    print('Creating calendar event with data: $data');
    
    // Parse title
    final String title = data['title'] ?? 'New Event';
    
    // Parse date
    DateTime eventDate;
    try {
      if (data['date'] != null) {
        eventDate = _parseDate(data['date']);
      } else {
        eventDate = DateTime.now(); // Default to today
      }
    } catch (e) {
      print('Error parsing date: ${data['date']} - $e');
      eventDate = DateTime.now();
    }
    
    // Parse time
    TimeOfDay startTimeOfDay;
    TimeOfDay endTimeOfDay;
    
    try {
      if (data['startTime'] != null) {
        startTimeOfDay = _parseTime(data['startTime']);
      } else {
        startTimeOfDay = TimeOfDay(hour: 9, minute: 0); // Default 9 AM
      }
      
      if (data['endTime'] != null) {
        endTimeOfDay = _parseTime(data['endTime']);
      } else {
        // Default duration 1 hour
        endTimeOfDay = TimeOfDay(
          hour: (startTimeOfDay.hour + 1) % 24,
          minute: startTimeOfDay.minute
        );
      }
    } catch (e) {
      print('Error parsing time: $e');
      startTimeOfDay = TimeOfDay(hour: 9, minute: 0);
      endTimeOfDay = TimeOfDay(hour: 10, minute: 0);
    }
    
    // Combine date and time
    final DateTime startDateTime = DateTime(
      eventDate.year,
      eventDate.month,
      eventDate.day,
      startTimeOfDay.hour,
      startTimeOfDay.minute,
    );
    
    final DateTime endDateTime = DateTime(
      eventDate.year,
      eventDate.month,
      eventDate.day,
      endTimeOfDay.hour,
      endTimeOfDay.minute,
    );
    
    // Create event object
    final event = EventModel(
      id: 0, // Database will automatically assign ID
      title: title,
      description: data['description'] ?? '',
      date: eventDate.toIso8601String().split('T')[0], // Keep only date part
      startTime: startDateTime.toIso8601String(),
      endTime: endDateTime.toIso8601String(),
      location: data['location'] ?? '',
      isAllDay: false, // Default not an all-day event
    );
    
    // Save to database
    await _dbHelper.insertEvent(event);
    print('Event created: ${event.title} on ${event.date}');
  }
  
  // Delete calendar event
  static Future<void> _deleteCalendarEvent(Map<String, dynamic> data) async {
    print('Deleting calendar event with criteria: $data');
    
    // Get all events
    final List<EventModel> allEvents = await _dbHelper.getAllEvents();
    List<EventModel> matchingEvents = [];
    
    // Matching criteria: title, date, time
    if (data['title'] != null) {
      // Match by title
      final String title = data['title'].toLowerCase();
      matchingEvents = allEvents.where((event) => 
        event.title.toLowerCase().contains(title)
      ).toList();
    } else if (data['date'] != null) {
      // Match by date
      try {
        final DateTime parsedDate = _parseDate(data['date']);
        final String dateString = parsedDate.toIso8601String().split('T')[0];
        
        matchingEvents = allEvents.where((event) => 
          event.date == dateString
        ).toList();
        
        // If time condition exists, filter further
        if (data['startTime'] != null && matchingEvents.isNotEmpty) {
          final TimeOfDay parsedTime = _parseTime(data['startTime']);
          
          matchingEvents = matchingEvents.where((event) {
            final DateTime eventStartTime = DateTime.parse(event.startTime);
            return eventStartTime.hour == parsedTime.hour && 
                   eventStartTime.minute == parsedTime.minute;
          }).toList();
        }
      } catch (e) {
        print('Error parsing date/time for deletion: $e');
      }
    } else {
      // Not enough criteria for matching
      print('Not enough criteria to identify events for deletion');
      return;
    }
    
    // Delete matching events
    if (matchingEvents.isNotEmpty) {
      print('Found ${matchingEvents.length} matching events to delete');
      
      for (var event in matchingEvents) {
        await _dbHelper.deleteEvent(event.id);
        print('Deleted event: ${event.title}');
      }
    } else {
      print('No matching events found for deletion');
    }
  }
  
  // Update calendar event
  static Future<void> _updateCalendarEvent(Map<String, dynamic> data) async {
    print('Updating calendar event with data: $data');
    
    // Get all events
    final List<EventModel> allEvents = await _dbHelper.getAllEvents();
    List<EventModel> matchingEvents = [];
    
    // Matching criteria: title, date
    if (data['title'] != null) {
      // Match by title
      final String title = data['title'].toLowerCase();
      matchingEvents = allEvents.where((event) => 
        event.title.toLowerCase().contains(title)
      ).toList();
    } else if (data['date'] != null) {
      // Match by date
      try {
        final DateTime parsedDate = _parseDate(data['date']);
        final String dateString = parsedDate.toIso8601String().split('T')[0];
        
        matchingEvents = allEvents.where((event) => 
          event.date == dateString
        ).toList();
      } catch (e) {
        print('Error parsing date for update: $e');
      }
    } else {
      // Not enough criteria for matching
      print('Not enough criteria to identify events for update');
      return;
    }
    
    // Update matching events
    if (matchingEvents.isNotEmpty) {
      print('Found ${matchingEvents.length} matching events to update');
      
      // Only update the first matching event, since there might be multiple events with the same name
      final EventModel eventToUpdate = matchingEvents.first;
      
      // Prepare updated values (only update provided fields)
      String updatedTitle = eventToUpdate.title;
      String updatedDescription = eventToUpdate.description;
      String updatedDate = eventToUpdate.date;
      String updatedStartTime = eventToUpdate.startTime;
      String updatedEndTime = eventToUpdate.endTime;
      String updatedLocation = eventToUpdate.location;
      
      // If new title provided
      if (data['title'] != null && data['title'] != eventToUpdate.title) {
        updatedTitle = data['title'];
      }
      
      // If new description provided
      if (data['description'] != null) {
        updatedDescription = data['description'];
      }
      
      // If new date provided
      if (data['date'] != null) {
        try {
          final DateTime parsedDate = _parseDate(data['date']);
          updatedDate = parsedDate.toIso8601String().split('T')[0];
          
          // Keep times unchanged, only update date part
          final DateTime oldStart = DateTime.parse(eventToUpdate.startTime);
          final DateTime oldEnd = DateTime.parse(eventToUpdate.endTime);
          
          updatedStartTime = DateTime(
            parsedDate.year, parsedDate.month, parsedDate.day,
            oldStart.hour, oldStart.minute
          ).toIso8601String();
          
          updatedEndTime = DateTime(
            parsedDate.year, parsedDate.month, parsedDate.day,
            oldEnd.hour, oldEnd.minute
          ).toIso8601String();
        } catch (e) {
          print('Error updating date: $e');
        }
      }
      
      // If new start time provided
      if (data['startTime'] != null) {
        try {
          final TimeOfDay parsedTime = _parseTime(data['startTime']);
          final DateTime dateTime = DateTime.parse(updatedStartTime);
          
          updatedStartTime = DateTime(
            dateTime.year, dateTime.month, dateTime.day,
            parsedTime.hour, parsedTime.minute
          ).toIso8601String();
        } catch (e) {
          print('Error updating start time: $e');
        }
      }
      
      // If new end time provided
      if (data['endTime'] != null) {
        try {
          final TimeOfDay parsedTime = _parseTime(data['endTime']);
          final DateTime dateTime = DateTime.parse(updatedEndTime);
          
          updatedEndTime = DateTime(
            dateTime.year, dateTime.month, dateTime.day,
            parsedTime.hour, parsedTime.minute
          ).toIso8601String();
        } catch (e) {
          print('Error updating end time: $e');
        }
      }
      
      // If new location provided
      if (data['location'] != null) {
        updatedLocation = data['location'];
      }
      
      // Create updated event object
      final updatedEvent = EventModel(
        id: eventToUpdate.id,
        title: updatedTitle,
        description: updatedDescription,
        date: updatedDate,
        startTime: updatedStartTime,
        endTime: updatedEndTime,
        location: updatedLocation,
        isAllDay: eventToUpdate.isAllDay,
      );
      
      // Save to database
      await _dbHelper.updateEvent(updatedEvent);
      print('Event updated: ${updatedEvent.title}');
    } else {
      print('No matching events found for update');
    }
  }


  // Query events method
  static Future<List<EventModel>> _queryCalendarEvents(Map<String, dynamic> data) async {
    print('Querying events with data: $data');
    
    // Initialize database helper
    final dbHelper = DatabaseHelper.instance;
    
    // Extract query parameters
    String? dateStr = data['date'];
    String? title = data['title'];
    
    // Query conditions
    Map<String, dynamic> whereClause = {};

    //print("Check!!");
    
    // If date parameter exists, convert to database date format
    // Parse and format date
    if (dateStr != null && dateStr.isNotEmpty) {
      try {
        // Remove question mark and clean up string
        dateStr = dateStr.replaceAll('?', '').trim();
        print('Processing date string: $dateStr');
        
        // Use existing _parseDate method
        final DateTime parsedDate = _parseDate(dateStr);
        print('Successfully parsed date: $parsedDate');
        
        // Format for database query (YYYY-MM-DD)
        final String formattedDate = DateFormat('yyyy-MM-dd').format(parsedDate);
        print('Formatted date for query: $formattedDate');
        
        // Add to query conditions
        whereClause['date'] = formattedDate;
      } catch (e) {
        print('Error parsing date for query: $e');
        // Error handling...
      }
    }
    
    // If title parameter exists, add title filter
    if (title != null && title.isNotEmpty) {
      whereClause['title'] = title;
    }
    
    // Execute query
    try {
      List<EventModel> events;
      if (whereClause.isEmpty) {
        // If no filter conditions, get the 10 most recent events
        events = await dbHelper.getRecentEvents(10);
      } else {
        // Query based on filter conditions
        events = await dbHelper.queryEvents(whereClause);
      }
      
      print('Query returned ${events.length} events');
      return events;
    } catch (e) {
      print('Error querying events: $e');
      return [];
    }
  }

  // Add this public method to ChatbotHelper.dart
  static Future<List<EventModel>> queryCalendarEvents(Map<String, dynamic> data) async {
    return await _queryCalendarEvents(data);
  }
  
  // Helper method: Parse date string
  static DateTime _parseDate(String dateStr) {
    dateStr = dateStr.toLowerCase();
    final DateTime now = DateTime.now();
    
    // Check common date terms
    if (dateStr.contains('today') || dateStr.contains('now')) {
      return now;
    } else if (dateStr.contains('tomorrow')) {
      return now.add(Duration(days: 1));
    } else if (dateStr.contains('day after tomorrow')) {
      return now.add(Duration(days: 2));
    } else if (dateStr.contains('next week')) {
      return now.add(Duration(days: 7));
    } else if (dateStr.contains('next month')) {
      return DateTime(now.year, now.month + 1, now.day);
    }
    
    // Try to parse common date formats
    try {
      // Try to match month name (e.g. "May 10")
      final RegExp monthDayPattern = RegExp(r'(\w+)\s+(\d{1,2})(?:st|nd|rd|th)?');
      final match = monthDayPattern.firstMatch(dateStr);
      
      if (match != null) {
        final String month = match.group(1)!;
        final int day = int.parse(match.group(2)!);
        
        // Convert month name to month number
        int monthNum;
        switch (month.toLowerCase()) {
          case 'january': case 'jan': monthNum = 1; break;
          case 'february': case 'feb': monthNum = 2; break;
          case 'march': case 'mar': monthNum = 3; break;
          case 'april': case 'apr': monthNum = 4; break;
          case 'may': monthNum = 5; break;
          case 'june': case 'jun': monthNum = 6; break;
          case 'july': case 'jul': monthNum = 7; break;
          case 'august': case 'aug': monthNum = 8; break;
          case 'september': case 'sep': monthNum = 9; break;
          case 'october': case 'oct': monthNum = 10; break;
          case 'november': case 'nov': monthNum = 11; break;
          case 'december': case 'dec': monthNum = 12; break;
          default: monthNum = now.month; // Default to current month
        }
        
        return DateTime(now.year, monthNum, day);
      }
      
      // Try to parse expressions like "next Monday"
      final RegExp nextWeekdayPattern = RegExp(r'next\s+(\w+)');
      final nextWeekdayMatch = nextWeekdayPattern.firstMatch(dateStr);
      
      if (nextWeekdayMatch != null) {
        final String weekday = nextWeekdayMatch.group(1)!.toLowerCase();
        int targetWeekday;
        
        switch (weekday) {
          case 'monday': case 'mon': targetWeekday = DateTime.monday; break;
          case 'tuesday': case 'tue': targetWeekday = DateTime.tuesday; break;
          case 'wednesday': case 'wed': targetWeekday = DateTime.wednesday; break;
          case 'thursday': case 'thu': targetWeekday = DateTime.thursday; break;
          case 'friday': case 'fri': targetWeekday = DateTime.friday; break;
          case 'saturday': case 'sat': targetWeekday = DateTime.saturday; break;
          case 'sunday': case 'sun': targetWeekday = DateTime.sunday; break;
          default: targetWeekday = now.weekday + 1; // Default to tomorrow
        }
        
        // Calculate next target weekday
        int daysUntilTarget = targetWeekday - now.weekday;
        if (daysUntilTarget <= 0) {
          daysUntilTarget += 7; // If target weekday already passed, calculate for next week
        }
        
        return now.add(Duration(days: daysUntilTarget));
      }
      
      // Try to parse standard date formats
      try {
        return DateFormat('yyyy-MM-dd').parse(dateStr);
      } catch (e) {
        try {
          return DateFormat('MM/dd/yyyy').parse(dateStr);
        } catch (e) {
          try {
            return DateFormat('dd/MM/yyyy').parse(dateStr);
          } catch (e) {
            // Finally try to convert weekday names
            switch (dateStr.toLowerCase()) {
              case 'monday': case 'mon':
                return _getNextWeekday(DateTime.monday);
              case 'tuesday': case 'tue':
                return _getNextWeekday(DateTime.tuesday);
              case 'wednesday': case 'wed':
                return _getNextWeekday(DateTime.wednesday);
              case 'thursday': case 'thu':
                return _getNextWeekday(DateTime.thursday);
              case 'friday': case 'fri':
                return _getNextWeekday(DateTime.friday);
              case 'saturday': case 'sat':
                return _getNextWeekday(DateTime.saturday);
              case 'sunday': case 'sun':
                return _getNextWeekday(DateTime.sunday);
              default:
                throw FormatException('Unable to parse date: $dateStr');
            }
          }
        }
      }
    } catch (e) {
      print('Failed to parse date: $dateStr - $e');
      return now; // Return current date on failure
    }
  }
  
  // Helper method: Get next specific weekday
  static DateTime _getNextWeekday(int targetWeekday) {
    final DateTime now = DateTime.now();
    int daysUntilTarget = targetWeekday - now.weekday;
    
    if (daysUntilTarget <= 0) {
      daysUntilTarget += 7; // If today or past, calculate for next week
    }
    
    return now.add(Duration(days: daysUntilTarget));
  }
  
  // Helper method: Parse time string
  static TimeOfDay _parseTime(String timeStr) {
    timeStr = timeStr.toLowerCase().replaceAll(' ', '');
    
    // Support common time expressions
    if (timeStr.contains('noon')) {
      return TimeOfDay(hour: 12, minute: 0);
    } else if (timeStr.contains('midnight')) {
      return TimeOfDay(hour: 0, minute: 0);
    } else if (timeStr.contains('morning')) {
      return TimeOfDay(hour: 9, minute: 0); // Default 9 AM
    } else if (timeStr.contains('afternoon')) {
      return TimeOfDay(hour: 14, minute: 0); // Default 2 PM
    } else if (timeStr.contains('evening')) {
      return TimeOfDay(hour: 18, minute: 0); // Default 6 PM
    } else if (timeStr.contains('night')) {
      return TimeOfDay(hour: 20, minute: 0); // Default 8 PM
    }
    
    // Try to match hour:minute format (12-hour or 24-hour)
    final RegExp timePattern = RegExp(r'(\d{1,2})(?::(\d{2}))?(?:\s*([ap]\.?m\.?))?');
    final match = timePattern.firstMatch(timeStr);
    
    if (match != null) {
      int hour = int.parse(match.group(1)!);
      int minute = 0;
      
      if (match.group(2) != null) {
        minute = int.parse(match.group(2)!);
      }
      
      // Handle AM/PM
      final String? ampm = match.group(3);
      if (ampm != null) {
        if (ampm.startsWith('p') && hour < 12) {
          hour += 12; // Convert to 24-hour format
        } else if (ampm.startsWith('a') && hour == 12) {
          hour = 0; // 12 AM = 0 hours
        }
      } else {
        // Smart handling when AM/PM not specified
        if (hour < 12 && timeStr.contains('evening') || timeStr.contains('night')) {
          hour += 12;
        }
      }
      
      return TimeOfDay(hour: hour % 24, minute: minute);
    }
    
    // If unable to parse, default to 9:00
    return TimeOfDay(hour: 9, minute: 0);
  }
}

// Encapsulate TimeOfDay class, simplify time handling
class TimeOfDay {
  final int hour;
  final int minute;
  
  TimeOfDay({required this.hour, required this.minute});
  
  @override
  String toString() {
    final String hourString = hour.toString().padLeft(2, '0');
    final String minuteString = minute.toString().padLeft(2, '0');
    return '$hourString:$minuteString';
  }
}