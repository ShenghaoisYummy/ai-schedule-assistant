// lib/database/database_helper.dart
import 'package:flutter/foundation.dart';
import 'dart:convert';
import 'package:calendar_chatbot/models/event_model.dart';
import 'package:calendar_chatbot/models/message_model.dart';

// For web platform
import 'dart:html' as html;

// For mobile platforms
import 'package:path/path.dart' as path;
import 'package:sqflite/sqflite.dart' as sqflite;

class DatabaseHelper {
  // Singleton pattern
  DatabaseHelper._privateConstructor();
  static final DatabaseHelper instance = DatabaseHelper._privateConstructor();

  static sqflite.Database? _database;
  static bool _initialized = false;

  // Get database instance (works for both web and mobile)
  Future<dynamic> get database async {
    if (_initialized) {
      return kIsWeb ? true : _database;
    }
    
    if (kIsWeb) {
      _initialized = true;
      print('Initialized web storage');
      return true; // Just a flag for web, we'll use localStorage directly
    } else {
      _database = await _initSqliteDatabase();
      _initialized = true;
      print('Initialized SQLite database');
      return _database;
    }
  }

  // Initialize SQLite database (for mobile only)
  Future<sqflite.Database> _initSqliteDatabase() async {
    final String dbPath = path.join(await sqflite.getDatabasesPath(), 'calendar_chatbot.db');
    return await sqflite.openDatabase(
      dbPath,
      version: 2, // Using version 2 for the updated schema
      onCreate: _onCreate,
      onUpgrade: _onUpgrade,
    );
  }

  // Create tables (for mobile only)
  Future<void> _onCreate(sqflite.Database db, int version) async {
    // Create events table with time support
    await db.execute('''
      CREATE TABLE events(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        date TEXT NOT NULL,
        startTime TEXT NOT NULL,
        endTime TEXT NOT NULL,
        location TEXT,
        isAllDay INTEGER NOT NULL DEFAULT 0
      )
    ''');

    // Create messages table
    await db.execute('''
      CREATE TABLE messages(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        isUser INTEGER NOT NULL,
        timestamp TEXT NOT NULL
      )
    ''');
  }

  // Handle database upgrades (for mobile only)
  Future<void> _onUpgrade(sqflite.Database db, int oldVersion, int newVersion) async {
    print('Upgrading database from $oldVersion to $newVersion');
    
    if (oldVersion < 2) {
      // This handles upgrading from version 1 to 2
      print('Adding new columns to events table');
      
      // Add new columns to events table
      await db.execute('ALTER TABLE events ADD COLUMN startTime TEXT');
      await db.execute('ALTER TABLE events ADD COLUMN endTime TEXT');
      await db.execute('ALTER TABLE events ADD COLUMN location TEXT DEFAULT ""');
      await db.execute('ALTER TABLE events ADD COLUMN isAllDay INTEGER DEFAULT 0');
      
      // Update existing records to set startTime and endTime
      final List<Map<String, dynamic>> events = await db.query('events');
      for (var event in events) {
        final DateTime date = DateTime.parse(event['date']);
        final DateTime startDateTime = DateTime(date.year, date.month, date.day, 9, 0); // Default 9:00 AM
        final DateTime endDateTime = DateTime(date.year, date.month, date.day, 10, 0); // Default 10:00 AM
        
        await db.update(
          'events',
          {
            'startTime': startDateTime.toIso8601String(),
            'endTime': endDateTime.toIso8601String(),
          },
          where: 'id = ?',
          whereArgs: [event['id']],
        );
      }
      print('Database migration completed');
    }
  }

  // Event-related operations
  Future<int> insertEvent(EventModel event) async {
    await database; // Ensure database is initialized
    
    if (kIsWeb) {
      // For web, store in localStorage
      print('Storing event in localStorage: ${event.title}');
      final events = await getAllEvents();
      
      // Generate new ID
      int maxId = 0;
      for (var e in events) {
        if (e.id > maxId) maxId = e.id;
      }
      
      final newEvent = EventModel(
        id: maxId + 1,
        title: event.title,
        description: event.description,
        date: event.date,
        startTime: event.startTime,
        endTime: event.endTime,
        location: event.location,
        isAllDay: event.isAllDay,
      );
      
      events.add(newEvent);
      final String eventsJson = jsonEncode(events.map((e) => e.toMap()).toList());
      html.window.localStorage['events'] = eventsJson;
      
      return newEvent.id;
    } else {
      // For mobile, use SQLite
      print('Storing event in SQLite: ${event.title}');
      final db = await database as sqflite.Database;
      return await db.insert('events', event.toMap());
    }
  }

  Future<List<EventModel>> getAllEvents() async {
    await database; // Ensure database is initialized
    
    if (kIsWeb) {
      // For web, retrieve from localStorage
      print('Retrieving events from localStorage');
      final String? eventsJson = html.window.localStorage['events'];
      
      if (eventsJson == null || eventsJson.isEmpty) {
        print('No events found in localStorage');
        return [];
      }
      
      try {
        final List<dynamic> eventsList = jsonDecode(eventsJson);
        return eventsList.map((e) => EventModel.fromMap(e)).toList();
      } catch (e) {
        print('Error parsing events from localStorage: $e');
        return [];
      }
    } else {
      // For mobile, use SQLite
      print('Retrieving events from SQLite');
      final db = await database as sqflite.Database;
      final List<Map<String, dynamic>> maps = await db.query('events');
      return List.generate(maps.length, (i) => EventModel.fromMap(maps[i]));
    }
  }

  Future<int> updateEvent(EventModel event) async {
    await database; // Ensure database is initialized
    
    if (kIsWeb) {
      // For web, update in localStorage
      print('Updating event in localStorage: ${event.title}');
      final events = await getAllEvents();
      
      final index = events.indexWhere((e) => e.id == event.id);
      if (index != -1) {
        events[index] = event;
        final String eventsJson = jsonEncode(events.map((e) => e.toMap()).toList());
        html.window.localStorage['events'] = eventsJson;
        return 1; // Success
      }
      
      return 0; // Not found
    } else {
      // For mobile, use SQLite
      print('Updating event in SQLite: ${event.title}');
      final db = await database as sqflite.Database;
      return await db.update(
        'events',
        event.toMap(),
        where: 'id = ?',
        whereArgs: [event.id],
      );
    }
  }

  Future<int> deleteEvent(int id) async {
    await database; // Ensure database is initialized
    
    if (kIsWeb) {
      // For web, delete from localStorage
      print('Deleting event from localStorage, ID: $id');
      final events = await getAllEvents();
      
      final originalLength = events.length;
      events.removeWhere((e) => e.id == id);
      
      if (events.length < originalLength) {
        final String eventsJson = jsonEncode(events.map((e) => e.toMap()).toList());
        html.window.localStorage['events'] = eventsJson;
        return 1; // Success
      }
      
      return 0; // Not found
    } else {
      // For mobile, use SQLite
      print('Deleting event from SQLite, ID: $id');
      final db = await database as sqflite.Database;
      return await db.delete(
        'events',
        where: 'id = ?',
        whereArgs: [id],
      );
    }
  }

  // Message-related operations
  Future<int> insertMessage(MessageModel message) async {
    await database; // Ensure database is initialized
    
    if (kIsWeb) {
      // For web, store in localStorage
      print('Storing message in localStorage');
      final messages = await getAllMessages();
      
      // Generate new ID
      int maxId = 0;
      for (var m in messages) {
        if (m.id > maxId) maxId = m.id;
      }
      
      final newMessage = MessageModel(
        id: maxId + 1,
        text: message.text,
        isUser: message.isUser,
        timestamp: message.timestamp,
      );
      
      messages.add(newMessage);
      final String messagesJson = jsonEncode(messages.map((m) => m.toMap()).toList());
      html.window.localStorage['messages'] = messagesJson;
      
      return newMessage.id;
    } else {
      // For mobile, use SQLite
      print('Storing message in SQLite');
      final db = await database as sqflite.Database;
      return await db.insert('messages', message.toMap());
    }
  }

  Future<List<MessageModel>> getAllMessages() async {
    await database; // Ensure database is initialized
    
    if (kIsWeb) {
      // For web, retrieve from localStorage
      print('Retrieving messages from localStorage');
      final String? messagesJson = html.window.localStorage['messages'];
      
      if (messagesJson == null || messagesJson.isEmpty) {
        print('No messages found in localStorage');
        return [];
      }
      
      try {
        final List<dynamic> messagesList = jsonDecode(messagesJson);
        final messages = messagesList.map((m) => MessageModel.fromMap(m)).toList();
        
        // Sort by timestamp
        messages.sort((a, b) => a.timestamp.compareTo(b.timestamp));
        return messages;
      } catch (e) {
        print('Error parsing messages from localStorage: $e');
        return [];
      }
    } else {
      // For mobile, use SQLite
      print('Retrieving messages from SQLite');
      final db = await database as sqflite.Database;
      final List<Map<String, dynamic>> maps = await db.query(
        'messages',
        orderBy: 'timestamp ASC',
      );
      return List.generate(maps.length, (i) => MessageModel.fromMap(maps[i]));
    }
  }

  Future<int> clearAllMessages() async {
    await database; // Ensure database is initialized
    
    if (kIsWeb) {
      // For web, clear messages from localStorage
      print('Clearing all messages from localStorage');
      html.window.localStorage['messages'] = '[]';
      return 1; // Success
    } else {
      // For mobile, use SQLite
      print('Clearing all messages from SQLite');
      final db = await database as sqflite.Database;
      return await db.delete('messages');
    }
  }
}