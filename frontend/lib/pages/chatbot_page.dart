// lib/pages/chatbot_page.dart
import 'package:flutter/material.dart';
import 'package:calendar_chatbot/utils/chatbot_helper.dart';
import 'package:calendar_chatbot/models/message_model.dart';
import 'package:calendar_chatbot/models/event_model.dart';
import 'package:calendar_chatbot/pages/calendar_page.dart';
import 'package:intl/intl.dart';
import 'package:http/http.dart' as http;
import 'package:path/path.dart';
import 'dart:convert';
import '../api_config.dart'; // Import the new config file

class ChatbotPage extends StatefulWidget {
  const ChatbotPage({Key? key}) : super(key: key); 
  @override
  _ChatbotPageState createState() => _ChatbotPageState();
}

class _ChatbotPageState extends State<ChatbotPage> {
  final TextEditingController _textController = TextEditingController();
  List<MessageModel> _messages = [];
  bool _isLoading = false;
  bool _isConnected = true;
  final ScrollController _scrollController = ScrollController();
  
  @override
  void initState() {
    super.initState();
    _initializeChat();
  }
  
  Future<void> _initializeChat() async {
    setState(() {
      _isLoading = true;
    });
    
    // Check API connection
    final isHealthy = await ChatbotHelper.checkHealth();
    
    // Load historical messages
    final messages = await ChatbotHelper.getAllMessages();
    
    setState(() {
      _isConnected = isHealthy;
      _messages = messages;
      _isLoading = false;
    });
    
    if (_messages.isEmpty) {
      // If no historical messages, add welcome message
      _addBotMessage(
        'Hi! I\'m your schedule assistant. How can I help you today?',
        intent: 'chitchat',
      );
    }
    
    // If API connection failed, show warning
    if (!isHealthy) {
      _addBotMessage(
        'I\'m having trouble connecting to the server. Your messages might not be processed correctly.',
        intent: 'error',
      );
    }
    
    _scrollToBottom();
  }
  
  void _handleSubmitted(String text) async {
    if (text.isEmpty) return;
    
    _textController.clear();
    
    // Add user message to UI
    _addUserMessage(text);
    
    // If not connected to API, add error message
    if (!_isConnected) {
      _addBotMessage(
        'Sorry, I\'m not connected to the server. Please check your network connection.',
        intent: 'error',
      );
      return;
    }
    
    setState(() {
      _isLoading = true;
    });
    
    try {
      // Use the new configuration to build the URL
      final response = await http.post(
        Uri.parse(ApiConfig.analyzeUrl),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'text': text}),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        final botResponse = data['response'] ?? 'Sorry, I could not understand.';
        _addBotMessage(
          botResponse,
          intent: data['intent'],
          entities: data['entities'],
        );
        
        // Process intent (e.g., add to calendar)
        final success = await ChatbotHelper.processIntent(data['intent'], data);
        
        // If ADD intent and successful, add confirmation message
        if (success && data['intent'] == 'add') {
          final title = _getEntityValue(data['entities'], 'title') ?? 'meeting';
          final date = _getEntityValue(data['entities'], 'date') ?? 'specified date';
          final time = _getEntityValue(data['entities'], 'startTime') ?? '';
          
          final confirmMessage = "I've added '$title' ${time.isNotEmpty ? 'at $time' : ''} on $date to your calendar.";
          _addBotMessage(confirmMessage, intent: 'confirmation');
        }
        
        // If calendar operation successful, add confirmation and option to view calendar
        if (success && data['intent'] != 'chitchat') {
          WidgetsBinding.instance.addPostFrameCallback((_) {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Text('Calendar updated successfully!'),
                action: SnackBarAction(
                  label: 'View Calendar',
                  onPressed: () {
                    Navigator.push(
                      context, 
                      MaterialPageRoute(builder: (context) => CalendarPage()),
                    );
                  },
                ),
                duration: Duration(seconds: 5),
              ),
            );
          });
        }
      } else {
        setState(() {
          _messages.insert(0, {'text': 'Error: Server connection failed.', 'isUser': false});
        });
      }
    } catch (e) {
      setState(() {
        _messages.insert(0, {'text': 'Error: ${e.toString()}', 'isUser': false});
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
      _scrollToBottom();
    }
  }

  // Helper method: Get entity value of specific type from entity list
  String? _getEntityValue(List<dynamic>? entities, String type) {
    if (entities == null) return null;
    
    for (var entity in entities) {
      if (entity['type'] == type) {
        return entity['text'];
      }
    }
    
    return null;
  }

  // Method to display query results
  void _displayQueryResults(List<EventModel> events, Map<String, dynamic> queryParams) {
    print('=== DISPLAYING QUERY RESULTS: ${events.length} events ==='); // Significant log
    
    if (events.isEmpty) {
      // Message for no events found
      String message = "I couldn't find any events";
      final date = queryParams['date'];
      if (date != null) {
        message += " on $date";
      }
      
      _addBotMessage(message, intent: "query_empty");
      return;
    }
    
    // Message for events found
    String message = "I found ${events.length} event${events.length > 1 ? 's' : ''}";
    final date = queryParams['date'];
    if (date != null) {
      message += " on $date";
    }
    message += ":";
    
    _addBotMessage(message, intent: "query_results");
    
    // Add detailed information for each event
    for (var event in events) {
      try {
        // Convert time format
        String timeStr = "All day";
        if (!event.isAllDay) {
          if (event.startTime.isNotEmpty && event.endTime.isNotEmpty) {
            try {
              final startTime = DateTime.parse(event.startTime);
              final endTime = DateTime.parse(event.endTime);
              timeStr = "${DateFormat('h:mm a').format(startTime)} - ${DateFormat('h:mm a').format(endTime)}";
            } catch (e) {
              print('Error formatting time: $e');
              timeStr = "${event.startTime} - ${event.endTime}";
            }
          }
        }
        
        // Format date
        String dateStr = event.date;
        try {
          final date = DateTime.parse(event.date);
          dateStr = DateFormat('EEE, MMM d').format(date);
        } catch (e) {
          print('Error formatting date: $e');
        }
        
        // Create event details message
        String eventDetails = "📅 *${event.title}*\n";
        eventDetails += "📆 $dateStr\n";
        eventDetails += "⏰ $timeStr";
        
        if (event.location.isNotEmpty) {
          eventDetails += "\n📍 ${event.location}";
        }
        
        print('Adding event details to UI: $eventDetails');
        _addBotMessage(eventDetails, intent: "event_details");
      } catch (e) {
        print('Error displaying event: $e');
        // Simplified display in error case
        _addBotMessage("📅 *${event.title}*", intent: "event_details");
      }
    }
    
    // Add action prompt
    //if (events.isNotEmpty) {
    //  _addBotMessage(
    //    "You can tap on the calendar icon to view more details.",
    //    intent: "suggestion"
    //  );
    //}
  }
  
  void _addUserMessage(String text) {
    // Create message model
    final message = MessageModel(
      id: 0, // Database will assign ID
      text: text,
      isUser: true,
      timestamp: DateTime.now(),
    );
    
    // Add to UI
    setState(() {
      _messages.add(message);
    });
    
    // Save to database
    ChatbotHelper.saveMessage(text, true);
    
    _scrollToBottom();
  }
  
  void _addBotMessage(String text, {String? intent, List<dynamic>? entities}) {
    // Create message model
    final message = MessageModel(
      id: 0, // Database will assign ID
      text: text,
      isUser: false,
      timestamp: DateTime.now(),
      intent: intent,
      entities: entities,
    );
    
    // Add to UI
    setState(() {
      _messages.add(message);
    });
    
    // Save to database (excluding intent and entities, as these are not in base message model)
    ChatbotHelper.saveMessage(text, false);
    
    _scrollToBottom();
  }
  
  void _scrollToBottom() {
   // Ensure scrolling to bottom after build completion
   WidgetsBinding.instance!.addPostFrameCallback((_) {
     if (_scrollController.hasClients) {
       _scrollController.animateTo(
         _scrollController.position.maxScrollExtent,
         duration: Duration(milliseconds: 300),
         curve: Curves.easeOut,
       );
     }
   });
 }
 
 void _clearChat() {
   showDialog(
     context: context,
     builder: (context) => AlertDialog(
       title: Text('Clear Conversation'),
       content: Text('Are you sure you want to clear all messages? This cannot be undone.'),
       actions: [
         TextButton(
           onPressed: () => Navigator.pop(context),
           child: Text('Cancel'),
         ),
         TextButton(
           onPressed: () async {
             Navigator.pop(context);
             await ChatbotHelper.clearMessages();
             setState(() {
               _messages = [];
             });
             _addBotMessage(
               'Hi! I\'m your schedule assistant. How can I help you today?',
               intent: 'chitchat',
             );
           },
           child: Text('Clear'),
         ),
       ],
     ),
   );
 }

 @override
 Widget build(BuildContext context) {
   return Scaffold(
     appBar: AppBar(
       title: Text('Schedule Assistant'),
       actions: [
         // Connection status indicator
         Container(
           padding: EdgeInsets.all(16.0),
           child: Center(
             child: Container(
               width: 12,
               height: 12,
               decoration: BoxDecoration(
                 shape: BoxShape.circle,
                 color: _isConnected ? Colors.green : Colors.red,
               ),
             ),
           ),
         ),
         // Clear conversation option
         IconButton(
           icon: Icon(Icons.delete_outline),
           onPressed: _clearChat,
           tooltip: 'Clear Conversation',
         ),
         // View calendar option
         IconButton(
           icon: Icon(Icons.calendar_today),
           onPressed: () {
             Navigator.push(
               context,
               MaterialPageRoute(builder: (context) => CalendarPage()),
             );
           },
           tooltip: 'View Calendar',
         ),
       ],
     ),
     body: Column(
       children: [
         Expanded(
           child: _messages.isEmpty
               ? Center(child: Text('No messages yet'))
               : ListView.builder(
                   controller: _scrollController,
                   padding: EdgeInsets.all(8.0),
                   itemCount: _messages.length,
                   itemBuilder: (context, index) => _buildMessage(_messages[index]),
                 ),
         ),
         if (_isLoading) LinearProgressIndicator(),
         Divider(height: 1.0),
         _buildMessageComposer(),
       ],
     ),
   );
 }

 Widget _buildMessage(MessageModel message) {
   return Container(
     margin: EdgeInsets.symmetric(vertical: 10.0, horizontal: 8.0),
     child: Row(
       mainAxisAlignment: 
           message.isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
       crossAxisAlignment: CrossAxisAlignment.start,
       children: [
         if (!message.isUser) 
           CircleAvatar(
             backgroundColor: Theme.of(context).primaryColor,
             child: Icon(Icons.smart_toy, color: Colors.white),
           ),
         SizedBox(width: message.isUser ? 0 : 8.0),
         Flexible(
           child: Column(
             crossAxisAlignment: 
                 message.isUser ? CrossAxisAlignment.end : CrossAxisAlignment.start,
             children: [
               Container(
                 padding: EdgeInsets.symmetric(horizontal: 16.0, vertical: 10.0),
                 decoration: BoxDecoration(
                   color: message.isUser 
                       ? Theme.of(context).primaryColor 
                       : (message.intent == 'error' ? Colors.red[100] : Colors.grey[200]),
                   borderRadius: BorderRadius.circular(20),
                 ),
                 child: Column(
                   crossAxisAlignment: CrossAxisAlignment.start,
                   children: [
                     Text(
                       message.text,
                       style: TextStyle(
                         color: message.isUser 
                             ? Colors.white 
                             : (message.intent == 'error' ? Colors.red[900] : Colors.black87),
                       ),
                     ),
                     if (!message.isUser && message.timestamp != null)
                       Padding(
                         padding: const EdgeInsets.only(top: 4.0),
                         child: Text(
                           DateFormat('HH:mm').format(message.timestamp),
                           style: TextStyle(
                             fontSize: 10.0,
                             color: Colors.grey[600],
                           ),
                         ),
                       ),
                   ],
                 ),
               ),
               // Only display intent and entities in non-user messages
               if (!message.isUser && message.intent != null && message.intent != 'error')
                 Padding(
                   padding: const EdgeInsets.only(top: 4.0, left: 8.0),
                   child: Text(
                     'Intent: ${message.intent}',
                     style: TextStyle(
                       fontSize: 11.0,
                       color: Colors.grey[600],
                     ),
                   ),
                 ),
               if (!message.isUser && message.entities != null && message.entities!.isNotEmpty)
                 Padding(
                   padding: const EdgeInsets.only(top: 2.0, left: 8.0),
                   child: _buildEntityChips(message.entities!),
                 ),
             ],
           ),
         ),
         SizedBox(width: message.isUser ? 8.0 : 0),
         if (message.isUser)
           CircleAvatar(
             backgroundColor: Colors.grey[300],
             child: Icon(Icons.person, color: Colors.grey[700]),
           ),
       ],
     ),
   );
 }

 Widget _buildEntityChips(List<dynamic> entities) {
   // Create a map to store processed entities
   Map<String, String> entityMap = {};
   
   for (var entity in entities) {
     String type = entity['type'] as String;
     String text = entity['text'] as String;
     
     // Handle B- and I- prefixes
     if (type.startsWith('B-')) {
       type = type.substring(2);
       entityMap[type] = text;
     } else if (type.startsWith('I-')) {
       type = type.substring(2);
       if (entityMap.containsKey(type)) {
         entityMap[type] = '${entityMap[type]!} $text';
       } else {
         entityMap[type] = text;
       }
     } else {
       entityMap[type] = text;
     }
   }
   
   return Wrap(
     spacing: 4.0,
     runSpacing: 4.0,
     children: entityMap.entries.map((entry) {
       return Chip(
         materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
         label: Text(
           '${entry.key}: ${entry.value}',
           style: TextStyle(fontSize: 10.0),
         ),
         backgroundColor: _getEntityColor(entry.key),
         padding: EdgeInsets.all(0),
         labelPadding: EdgeInsets.symmetric(horizontal: 8.0),
       );
     }).toList(),
   );
 }

 Color _getEntityColor(String type) {
   switch (type.toLowerCase()) {
     case 'action':
       return Colors.blue[100]!;
     case 'title':
       return Colors.purple[100]!;
     case 'date':
       return Colors.green[100]!;
     case 'location':
       return Colors.orange[100]!;
     case 'starttime':
       return Colors.teal[100]!;
     case 'endtime':
       return Colors.indigo[100]!;
     case 'description':
       return Colors.pink[100]!;
     default:
       return Colors.grey[200]!;
   }
 }

 Widget _buildMessageComposer() {
   return Container(
     padding: EdgeInsets.symmetric(horizontal: 8.0, vertical: 4.0),
     decoration: BoxDecoration(
       color: Theme.of(context).cardColor,
       boxShadow: [
         BoxShadow(
           blurRadius: 3,
           color: Colors.black12,
           offset: Offset(0, -2),
         ),
       ],
     ),
     child: SafeArea(
       child: Row(
         children: [
           IconButton(
             icon: Icon(Icons.mic),
             onPressed: _isConnected ? () {
               // Voice input feature (optional implementation)
               ScaffoldMessenger.of(context).showSnackBar(
                 SnackBar(content: Text('Voice input not implemented yet')),
               );
             } : null,
             color: Theme.of(context).primaryColor,
           ),
           Expanded(
             child: TextField(
               controller: _textController,
               onSubmitted: _isConnected ? _handleSubmitted : null,
               decoration: InputDecoration(
                 hintText: _isConnected 
                     ? 'Ask about scheduling...' 
                     : 'Unable to connect to server...',
                 border: OutlineInputBorder(
                   borderRadius: BorderRadius.circular(20.0),
                   borderSide: BorderSide.none,
                 ),
                 filled: true,
                 fillColor: Colors.grey[200],
                 contentPadding: EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
               ),
               enabled: _isConnected,
             ),
           ),
           IconButton(
             icon: Icon(Icons.send),
             onPressed: _isConnected && !_isLoading
                 ? () => _handleSubmitted(_textController.text)
                 : null,
             color: Theme.of(context).primaryColor,
           ),
         ],
       ),
     ),
   );
 }
}