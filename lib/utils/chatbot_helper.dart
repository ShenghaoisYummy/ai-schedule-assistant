// lib/utils/chatbot_helper.dart
import 'package:calendar_chatbot/models/event_model.dart';
import 'package:calendar_chatbot/database/database_helper.dart';
import 'package:intl/intl.dart';

class ChatbotHelper {
  static Future<String> processMessage(String message) async {
    // Convert message to lowercase for easier comparison
    final lowerMessage = message.toLowerCase();
    
    // Format for displaying time
    final timeFormat = DateFormat('h:mm a');
    final dateFormat = DateFormat('EEEE, MMMM d');
    
    // Check for calendar-related queries
    if (lowerMessage.contains('today') && 
        (lowerMessage.contains('schedule') || 
         lowerMessage.contains('agenda') || 
         lowerMessage.contains('events'))) {
      return await _getTodaySchedule(timeFormat, dateFormat);
    } else if ((lowerMessage.contains('week') || lowerMessage.contains('weekly')) && 
               (lowerMessage.contains('schedule') || 
                lowerMessage.contains('agenda') || 
                lowerMessage.contains('events'))) {
      return await _getWeekSchedule(timeFormat, dateFormat);
    } else if (lowerMessage.contains('tomorrow') && 
               (lowerMessage.contains('schedule') || 
                lowerMessage.contains('agenda') || 
                lowerMessage.contains('events'))) {
      return await _getTomorrowSchedule(timeFormat, dateFormat);
    } else if ((lowerMessage.contains('remind') || lowerMessage.contains('tell')) && 
               lowerMessage.contains('about') && 
               lowerMessage.contains('next')) {
      return await _getNextEvent(timeFormat, dateFormat);
    }
    
    // Default response for non-calendar queries
    return "I'm your calendar assistant. You can ask me about your schedule for today, this week, or tomorrow. How can I help you?";
  }
  
  // Get today's schedule
  static Future<String> _getTodaySchedule(DateFormat timeFormat, DateFormat dateFormat) async {
    final today = DateTime.now();
    final todayStart = DateTime(today.year, today.month, today.day);
    final todayEnd = DateTime(today.year, today.month, today.day, 23, 59, 59);
    
    final events = await DatabaseHelper.instance.getAllEvents();
    final todayEvents = events.where((event) => 
      event.date.isAtSameMomentAs(todayStart) ||
      (event.date.isAfter(todayStart) && event.date.isBefore(todayEnd)) ||
      (event.startTime.isAfter(todayStart) && event.startTime.isBefore(todayEnd))
    ).toList();
    
    if (todayEvents.isEmpty) {
      return "You have no events scheduled for today (${dateFormat.format(today)}).";
    }
    
    // Sort events by start time
    todayEvents.sort((a, b) => a.startTime.compareTo(b.startTime));
    
    // Build response
    final buffer = StringBuffer();
    buffer.write("Here's your schedule for today (${dateFormat.format(today)}):\n\n");
    
    for (var event in todayEvents) {
      if (event.isAllDay) {
        buffer.write("• All day: ${event.title}");
      } else {
        buffer.write("• ${timeFormat.format(event.startTime)} - ${timeFormat.format(event.endTime)}: ${event.title}");
      }
      
      if (event.location.isNotEmpty) {
        buffer.write(" (${event.location})");
      }
      
      buffer.write("\n");
    }
    
    return buffer.toString();
  }
  
  // Get this week's schedule
  static Future<String> _getWeekSchedule(DateFormat timeFormat, DateFormat dateFormat) async {
    final today = DateTime.now();
    
    // Find the start of the week (Monday)
    final weekStart = today.subtract(Duration(days: today.weekday - 1));
    final weekStartDate = DateTime(weekStart.year, weekStart.month, weekStart.day);
    
    // Find the end of the week (Sunday)
    final weekEnd = weekStartDate.add(const Duration(days: 6, hours: 23, minutes: 59));
    
    final events = await DatabaseHelper.instance.getAllEvents();
    final weekEvents = events.where((event) => 
      (event.date.isAfter(weekStartDate) && event.date.isBefore(weekEnd)) ||
      event.date.isAtSameMomentAs(weekStartDate) ||
      (event.startTime.isAfter(weekStartDate) && event.startTime.isBefore(weekEnd))
    ).toList();
    
    if (weekEvents.isEmpty) {
      return "You have no events scheduled for this week (${dateFormat.format(weekStartDate)} - ${dateFormat.format(weekEnd)}).";
    }
    
    // Sort events by date and start time
    weekEvents.sort((a, b) {
      final dateComparison = a.date.compareTo(b.date);
      if (dateComparison != 0) return dateComparison;
      return a.startTime.compareTo(b.startTime);
    });
    
    // Group events by day
    final eventsByDay = <DateTime, List<EventModel>>{};
    for (var event in weekEvents) {
      final eventDate = DateTime(event.date.year, event.date.month, event.date.day);
      if (!eventsByDay.containsKey(eventDate)) {
        eventsByDay[eventDate] = [];
      }
      eventsByDay[eventDate]!.add(event);
    }
    
    // Build response
    final buffer = StringBuffer();
    buffer.write("Here's your schedule for this week (${dateFormat.format(weekStartDate)} - ${dateFormat.format(weekEnd)}):\n\n");
    
    final sortedDates = eventsByDay.keys.toList()..sort();
    for (var date in sortedDates) {
      buffer.write("${dateFormat.format(date)}:\n");
      
      for (var event in eventsByDay[date]!) {
        if (event.isAllDay) {
          buffer.write("• All day: ${event.title}");
        } else {
          buffer.write("• ${timeFormat.format(event.startTime)} - ${timeFormat.format(event.endTime)}: ${event.title}");
        }
        
        if (event.location.isNotEmpty) {
          buffer.write(" (${event.location})");
        }
        
        buffer.write("\n");
      }
      
      buffer.write("\n");
    }
    
    return buffer.toString();
  }
  
  // Get tomorrow's schedule
  static Future<String> _getTomorrowSchedule(DateFormat timeFormat, DateFormat dateFormat) async {
    final today = DateTime.now();
    final tomorrow = today.add(const Duration(days: 1));
    final tomorrowStart = DateTime(tomorrow.year, tomorrow.month, tomorrow.day);
    final tomorrowEnd = DateTime(tomorrow.year, tomorrow.month, tomorrow.day, 23, 59, 59);
    
    final events = await DatabaseHelper.instance.getAllEvents();
    final tomorrowEvents = events.where((event) => 
      event.date.isAtSameMomentAs(tomorrowStart) ||
      (event.date.isAfter(tomorrowStart) && event.date.isBefore(tomorrowEnd)) ||
      (event.startTime.isAfter(tomorrowStart) && event.startTime.isBefore(tomorrowEnd))
    ).toList();
    
    if (tomorrowEvents.isEmpty) {
      return "You have no events scheduled for tomorrow (${dateFormat.format(tomorrow)}).";
    }
    
    // Sort events by start time
    tomorrowEvents.sort((a, b) => a.startTime.compareTo(b.startTime));
    
    // Build response
    final buffer = StringBuffer();
    buffer.write("Here's your schedule for tomorrow (${dateFormat.format(tomorrow)}):\n\n");
    
    for (var event in tomorrowEvents) {
      if (event.isAllDay) {
        buffer.write("• All day: ${event.title}");
      } else {
        buffer.write("• ${timeFormat.format(event.startTime)} - ${timeFormat.format(event.endTime)}: ${event.title}");
      }
      
      if (event.location.isNotEmpty) {
        buffer.write(" (${event.location})");
      }
      
      buffer.write("\n");
    }
    
    return buffer.toString();
  }
  
  // Get the next upcoming event
  static Future<String> _getNextEvent(DateFormat timeFormat, DateFormat dateFormat) async {
    final now = DateTime.now();
    
    final events = await DatabaseHelper.instance.getAllEvents();
    if (events.isEmpty) {
      return "You don't have any upcoming events scheduled.";
    }
    
    // Find the next event that hasn't ended yet
    final upcomingEvents = events.where((event) => 
      event.endTime.isAfter(now)
    ).toList();
    
    if (upcomingEvents.isEmpty) {
      return "You don't have any upcoming events scheduled.";
    }
    
    // Sort by start time to find the next one
    upcomingEvents.sort((a, b) => a.startTime.compareTo(b.startTime));
    final nextEvent = upcomingEvents.first;
    
    // Calculate how soon the event is
    final timeUntil = nextEvent.startTime.difference(now);
    String timeUntilText;
    
    if (timeUntil.inDays > 0) {
      timeUntilText = "${timeUntil.inDays} ${timeUntil.inDays == 1 ? 'day' : 'days'}";
    } else if (timeUntil.inHours > 0) {
      timeUntilText = "${timeUntil.inHours} ${timeUntil.inHours == 1 ? 'hour' : 'hours'}";
    } else if (timeUntil.inMinutes > 0) {
      timeUntilText = "${timeUntil.inMinutes} ${timeUntil.inMinutes == 1 ? 'minute' : 'minutes'}";
    } else {
      timeUntilText = "less than a minute";
    }
    
    // Build response
    final buffer = StringBuffer();
    buffer.write("Your next event is ");
    buffer.write("\"${nextEvent.title}\" ");
    
    if (nextEvent.isAllDay) {
      buffer.write("(all day) ");
    } else {
      buffer.write("at ${timeFormat.format(nextEvent.startTime)} ");
    }
    
    buffer.write("on ${dateFormat.format(nextEvent.date)}");
    
    if (nextEvent.location.isNotEmpty) {
      buffer.write(" at ${nextEvent.location}");
    }
    
    buffer.write(". It starts in $timeUntilText.");
    
    if (nextEvent.description.isNotEmpty) {
      buffer.write("\n\nDescription: ${nextEvent.description}");
    }
    
    return buffer.toString();
  }
}