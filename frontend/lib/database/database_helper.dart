// lib/database/database_helper.dart

import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'package:sqflite/sqflite.dart';
import 'package:calendar_chatbot/models/event_model.dart';
import 'package:calendar_chatbot/models/message_model.dart';
import 'package:shared_preferences/shared_preferences.dart'; // Add this import

class DatabaseHelper {
  static final DatabaseHelper instance = DatabaseHelper._init();
  static Database? _database;

  DatabaseHelper._init();

  Future<Database> get database async {
    if (_database != null) return _database!;
    
    if (kIsWeb) {
      // Web platform doesn't need an actual SQLite database initialization
      print('Initialized web storage');
      return _database!;
    } else {
      _database = await _initDB('calendar_chatbot.db');
      return _database!;
    }
  }

  Future<Database> _initDB(String filePath) async {
    final dbPath = await getDatabasesPath();
    final path = join(dbPath, filePath);

    return await openDatabase(
      path, 
      version: 1,
      onCreate: _createDB
    );
  }

  Future _createDB(Database db, int version) async {
    // Create events table
    await db.execute('''
      CREATE TABLE events(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT,
        date TEXT NOT NULL,
        startTime TEXT,
        endTime TEXT,
        location TEXT,
        isAllDay INTEGER
      )
    ''');
    
    // Create messages table
    await db.execute('''
      CREATE TABLE messages(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        isUser INTEGER NOT NULL,
        timestamp TEXT NOT NULL,
        intent TEXT,
        entities TEXT
      )
    ''');
  }

  // Insert event
  Future<int> insertEvent(EventModel event) async {
    try {
      if (kIsWeb) {
        // Web storage implementation
        final prefs = await SharedPreferences.getInstance();
        int nextId = (prefs.getInt('next_event_id') ?? 0) + 1;
        
        // Update event ID
        final updatedEvent = EventModel(
          id: nextId,
          title: event.title,
          description: event.description,
          date: event.date,
          startTime: event.startTime,
          endTime: event.endTime,
          location: event.location,
          isAllDay: event.isAllDay,
        );
        
        // Save event
        await prefs.setString('event_$nextId', jsonEncode(updatedEvent.toMap()));
        await prefs.setInt('next_event_id', nextId);
        
        print('Saved event to web storage: ${updatedEvent.title}');
        return nextId;
      } else {
        // SQLite implementation
        final db = await database;
        final id = await db.insert('events', event.toMap());
        print('Saved event to SQLite: ${event.title} with ID $id');
        return id;
      }
    } catch (e) {
      print('Error inserting event: $e');
      return -1;
    }
  }

  // Get all events
  Future<List<EventModel>> getAllEvents() async {
    try {
      if (kIsWeb) {
        // Web storage implementation
        final prefs = await SharedPreferences.getInstance();
        final List<EventModel> events = [];
        
        // Get all keys and filter for events
        final keys = prefs.getKeys();
        final eventKeys = keys.where((key) => key.startsWith('event_')).toList();
        
        for (String key in eventKeys) {
          final String? eventJson = prefs.getString(key);
          if (eventJson != null) {
            try {
              final Map<String, dynamic> map = jsonDecode(eventJson);
              events.add(EventModel.fromMap(map));
            } catch (e) {
              print('Error parsing event $key: $e');
            }
          }
        }
        
        return events;
      } else {
        // SQLite implementation
        final db = await database;
        final result = await db.query('events');
        
        return result.map((map) => EventModel.fromMap(map)).toList();
      }
    } catch (e) {
      print('Error getting all events: $e');
      return [];
    }
  }

  // Update event
  Future<int> updateEvent(EventModel event) async {
    try {
      if (kIsWeb) {
        // Web storage implementation
        final prefs = await SharedPreferences.getInstance();
        await prefs.setString('event_${event.id}', jsonEncode(event.toMap()));
        return 1; // Indicates success
      } else {
        // SQLite implementation
        final db = await database;
        return await db.update(
          'events',
          event.toMap(),
          where: 'id = ?',
          whereArgs: [event.id]
        );
      }
    } catch (e) {
      print('Error updating event: $e');
      return 0;
    }
  }

  // Delete event
  Future<int> deleteEvent(int id) async {
    try {
      if (kIsWeb) {
        // Web storage implementation
        final prefs = await SharedPreferences.getInstance();
        await prefs.remove('event_$id');
        return 1; // Indicates success
      } else {
        // SQLite implementation
        final db = await database;
        return await db.delete(
          'events',
          where: 'id = ?',
          whereArgs: [id]
        );
      }
    } catch (e) {
      print('Error deleting event: $e');
      return 0;
    }
  }

  // Insert message
  Future<int> insertMessage(MessageModel message) async {
    try {
      if (kIsWeb) {
        // Web storage implementation
        final prefs = await SharedPreferences.getInstance();
        int nextId = (prefs.getInt('next_message_id') ?? 0) + 1;
        
        // Update message ID
        final Map<String, dynamic> messageMap = message.toMap();
        messageMap['id'] = nextId;
        
        // Save message
        await prefs.setString('message_$nextId', jsonEncode(messageMap));
        await prefs.setInt('next_message_id', nextId);
        
        print('Stored message in web storage');
        return nextId;
      } else {
        // SQLite implementation
        final db = await database;
        return await db.insert('messages', message.toMap());
      }
    } catch (e) {
      print('Error inserting message: $e');
      return -1;
    }
  }

  // Get all messages
  Future<List<MessageModel>> getAllMessages() async {
    try {
      if (kIsWeb) {
        // Web storage implementation
        final prefs = await SharedPreferences.getInstance();
        final List<MessageModel> messages = [];
        
        // Get all keys and filter for messages
        final keys = prefs.getKeys();
        final messageKeys = keys.where((key) => key.startsWith('message_')).toList();
        
        // Sort keys to ensure messages are returned in ID order
        messageKeys.sort((a, b) {
          final idA = int.tryParse(a.split('_')[1]) ?? 0;
          final idB = int.tryParse(b.split('_')[1]) ?? 0;
          return idA.compareTo(idB);
        });
        
        for (String key in messageKeys) {
          final String? messageJson = prefs.getString(key);
          if (messageJson != null) {
            try {
              final Map<String, dynamic> map = jsonDecode(messageJson);
              messages.add(MessageModel.fromMap(map));
            } catch (e) {
              print('Error parsing message $key: $e');
            }
          }
        }
        
        return messages;
      } else {
        // SQLite implementation
        final db = await database;
        final result = await db.query(
          'messages', 
          orderBy: 'id ASC'
        );
        
        return result.map((map) => MessageModel.fromMap(map)).toList();
      }
    } catch (e) {
      print('Error getting all messages: $e');
      return [];
    }
  }

  // Query events method
  Future<List<EventModel>> queryEvents(Map<String, dynamic> whereClause) async {
    final List<EventModel> events = [];
    
    try {
      if (kIsWeb) {
        // Web storage implementation
        final prefs = await SharedPreferences.getInstance();
        
        // Get all keys and filter for events
        final keys = prefs.getKeys();
        final eventKeys = keys.where((key) => key.startsWith('event_')).toList();
        
        for (String key in eventKeys) {
          final String? eventJson = prefs.getString(key);
          if (eventJson != null) {
            try {
              final Map<String, dynamic> map = jsonDecode(eventJson);
              final event = EventModel.fromMap(map);
              
              // Apply filter conditions
              bool matches = true;
              whereClause.forEach((key, value) {
                // For title, use contains matching
                if (key == 'title') {
                  if (!event.title.toLowerCase().contains(value.toLowerCase())) {
                    matches = false;
                  }
                } else if (map[key] != value) {
                  matches = false;
                }
              });
              
              if (matches) {
                events.add(event);
              }
            } catch (e) {
              print('Error parsing event $key: $e');
            }
          }
        }
      } else {
        // SQLite implementation
        final db = await database;
        String whereString = '';
        List<dynamic> whereArgs = [];
        
        int i = 0;
        whereClause.forEach((key, value) {
          if (i > 0) whereString += ' AND ';
          
          // For title, use LIKE query
          if (key == 'title') {
            whereString += '$key LIKE ?';
            whereArgs.add('%$value%');
          } else {
            whereString += '$key = ?';
            whereArgs.add(value);
          }
          i++;
        });
        
        final List<Map<String, dynamic>> maps = await db.query(
          'events',
          where: whereString.isNotEmpty ? whereString : null,
          whereArgs: whereArgs.isNotEmpty ? whereArgs : null,
        );
        
        for (var map in maps) {
          events.add(EventModel.fromMap(map));
        }
      }
      
      return events;
    } catch (e) {
      print('Error querying events: $e');
      return [];
    }
  }

  // Get recent events method
  Future<List<EventModel>> getRecentEvents(int limit) async {
    final List<EventModel> events = [];
    
    try {
      if (kIsWeb) {
        // Web storage implementation
        final prefs = await SharedPreferences.getInstance();
        
        // Get all keys and filter for events
        final keys = prefs.getKeys();
        final eventKeys = keys.where((key) => key.startsWith('event_')).toList();
        
        List<EventModel> allEvents = [];
        for (String key in eventKeys) {
          final String? eventJson = prefs.getString(key);
          if (eventJson != null) {
            try {
              final Map<String, dynamic> map = jsonDecode(eventJson);
              allEvents.add(EventModel.fromMap(map));
            } catch (e) {
              print('Error parsing event $key: $e');
            }
          }
        }
        
        // Sort by date
        allEvents.sort((a, b) {
          final DateTime dateA = DateTime.parse(a.date);
          final DateTime dateB = DateTime.parse(b.date);
          return dateA.compareTo(dateB);
        });
        
        // Get the most recent events
        events.addAll(allEvents.take(limit));
      } else {
        // SQLite implementation
        final db = await database;
        final List<Map<String, dynamic>> maps = await db.query(
          'events',
          orderBy: 'date ASC',
          limit: limit,
        );
        
        for (var map in maps) {
          events.add(EventModel.fromMap(map));
        }
      }
      
      return events;
    } catch (e) {
      print('Error getting recent events: $e');
      return [];
    }
  }

  // Clear all messages
  Future<void> clearAllMessages() async {
    try {
      if (kIsWeb) {
        // Web storage implementation
        final prefs = await SharedPreferences.getInstance();
        
        // Get all keys and filter for messages
        final keys = prefs.getKeys();
        final messageKeys = keys.where((key) => key.startsWith('message_')).toList();
        
        // Delete all messages
        for (String key in messageKeys) {
          await prefs.remove(key);
        }
        
        // Reset message ID counter
        await prefs.setInt('next_message_id', 0);
        
        print('Cleared all messages from web storage');
      } else {
        // SQLite implementation
        final db = await database;
        await db.delete('messages');
        print('Cleared all messages from SQLite database');
      }
    } catch (e) {
      print('Error clearing messages: $e');
    }
  }

  // Close database
  Future close() async {
    if (!kIsWeb) {
      final db = await instance.database;
      db.close();
    }
  }
}