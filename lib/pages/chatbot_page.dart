// lib/pages/chatbot_page.dart
import 'package:flutter/material.dart';
import 'package:calendar_chatbot/models/message_model.dart';
import 'package:calendar_chatbot/database/database_helper.dart';
import 'package:calendar_chatbot/utils/chatbot_helper.dart';

class ChatbotPage extends StatefulWidget {
  const ChatbotPage({Key? key}) : super(key: key);

  @override
  _ChatbotPageState createState() => _ChatbotPageState();
}

class _ChatbotPageState extends State<ChatbotPage> {
  final TextEditingController _controller = TextEditingController();
  List<MessageModel> _messages = [];
  final ScrollController _scrollController = ScrollController();
  bool _isLoading = true;
  bool _isTyping = false;

  @override
  void initState() {
    super.initState();
    _loadMessages();
  }

  Future<void> _loadMessages() async {
    setState(() {
      _isLoading = true;
    });

    final messages = await DatabaseHelper.instance.getAllMessages();
    
    setState(() {
      _messages = messages;
      _isLoading = false;
    });

    // Add welcome message if there are no messages
    if (_messages.isEmpty) {
      _addBotMessage("Hello! I'm your calendar assistant. You can ask me about your schedule for today, this week, or about upcoming events. How can I help you?");
    }

    // Scroll to bottom after loading messages
    if (_messages.isNotEmpty) {
      _scrollToBottom();
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  Future<void> _addBotMessage(String text) async {
    final botMsg = MessageModel(
      id: 0, // Auto-increment
      text: text,
      isUser: false,
      timestamp: DateTime.now(),
    );
    
    await DatabaseHelper.instance.insertMessage(botMsg);
    await _loadMessages();
    _scrollToBottom();
  }

  void _handleSubmitted(String text) async {
    if (text.trim().isEmpty) return;
    
    _controller.clear();
    
    // Add user message and save to database
    final userMsg = MessageModel(
      id: 0, // Auto-increment
      text: text,
      isUser: true,
      timestamp: DateTime.now(),
    );
    
    await DatabaseHelper.instance.insertMessage(userMsg);
    
    // Reload messages to get correct ID
    await _loadMessages();
    
    // Auto-scroll to bottom
    _scrollToBottom();
    
    // Show typing indicator
    setState(() {
      _isTyping = true;
    });
    
    // Process the message and generate a response
    try {
      final response = await ChatbotHelper.processMessage(text);
      
      // Simulate typing delay based on response length
      await Future.delayed(Duration(milliseconds: 500 + (response.length ~/ 10) * 50));
      
      await _addBotMessage(response);
    } catch (e) {
      print('Error processing message: $e');
      await _addBotMessage("I'm sorry, I encountered an error while processing your request.");
    } finally {
      setState(() {
        _isTyping = false;
      });
    }
  }

  void _scrollToBottom() {
    Future.delayed(const Duration(milliseconds: 100), () {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Calendar Assistant'),
        actions: [
          IconButton(
            icon: const Icon(Icons.delete),
            onPressed: () => _showClearChatDialog(),
          ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : Column(
              children: [
                Expanded(
                  child: ListView.builder(
                    controller: _scrollController,
                    padding: const EdgeInsets.all(8.0),
                    itemCount: _messages.length + (_isTyping ? 1 : 0),
                    itemBuilder: (context, index) {
                      if (index < _messages.length) {
                        final message = _messages[index];
                        return _buildChatMessage(message);
                      } else {
                        // Typing indicator
                        return _buildTypingIndicator();
                      }
                    },
                  ),
                ),
                const Divider(height: 1.0),
                Container(
                  decoration: BoxDecoration(
                    color: Theme.of(context).cardColor,
                  ),
                  child: _buildTextComposer(),
                ),
              ],
            ),
    );
  }

  Widget _buildTypingIndicator() {
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 10.0, horizontal: 16.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          CircleAvatar(
            child: const Text('Bot'),
            backgroundColor: Colors.grey[300],
          ),
          const SizedBox(width: 8.0),
          Container(
            padding: const EdgeInsets.all(12.0),
            decoration: BoxDecoration(
              color: Colors.grey[200],
              borderRadius: BorderRadius.circular(8.0),
            ),
            child: Row(
              children: [
                SizedBox(
                  width: 40,
                  child: _DotLoadingIndicator(),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildChatMessage(MessageModel message) {
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 10.0, horizontal: 16.0),
      child: Row(
        mainAxisAlignment: message.isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (!message.isUser) ...[
            CircleAvatar(
              child: const Text('Bot'),
              backgroundColor: Colors.grey[300],
            ),
            const SizedBox(width: 8.0),
          ],
          Flexible(
            child: Container(
              padding: const EdgeInsets.all(12.0),
              decoration: BoxDecoration(
                color: message.isUser ? Colors.blue[100] : Colors.grey[200],
                borderRadius: BorderRadius.circular(8.0),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    message.text,
                    style: const TextStyle(fontSize: 14.0),
                  ),
                  const SizedBox(height: 4.0),
                  Text(
                    _formatTime(message.timestamp),
                    style: TextStyle(
                      fontSize: 10.0,
                      color: Colors.grey[600],
                    ),
                  ),
                ],
              ),
            ),
          ),
          if (message.isUser) ...[
            const SizedBox(width: 8.0),
            CircleAvatar(
              child: const Text('Me'),
              backgroundColor: Colors.blue[200],
            ),
          ],
        ],
      ),
    );
  }

  String _formatTime(DateTime time) {
    return '${time.hour.toString().padLeft(2, '0')}:${time.minute.toString().padLeft(2, '0')}';
  }

  Widget _buildTextComposer() {
    return IconTheme(
      data: IconThemeData(color: Theme.of(context).primaryColor),
      child: Container(
        margin: const EdgeInsets.symmetric(horizontal: 8.0),
        padding: const EdgeInsets.symmetric(horizontal: 8.0, vertical: 4.0),
        child: Row(
          children: [
            Flexible(
              child: TextField(
                controller: _controller,
                decoration: const InputDecoration(
                  hintText: 'Ask about your schedule...',
                  border: InputBorder.none,
                ),
                onSubmitted: _isTyping ? null : _handleSubmitted,
                enabled: !_isTyping,
              ),
            ),
            IconButton(
              icon: const Icon(Icons.send),
              onPressed: _isTyping ? null : () => _handleSubmitted(_controller.text),
            ),
          ],
        ),
      ),
    );
  }

  void _showClearChatDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Clear Chat History'),
        content: const Text('Are you sure you want to clear all chat history? This action cannot be undone.'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () async {
              await DatabaseHelper.instance.clearAllMessages();
              await _loadMessages();
              Navigator.pop(context);
            },
            child: const Text('Clear'),
          ),
        ],
      ),
    );
  }
}

// Custom typing indicator animation
class _DotLoadingIndicator extends StatefulWidget {
  @override
  _DotLoadingIndicatorState createState() => _DotLoadingIndicatorState();
}

class _DotLoadingIndicatorState extends State<_DotLoadingIndicator> with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  int _dotCount = 0;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 500),
    )..addListener(() {
        setState(() {
          _dotCount = (_controller.value * 3).floor() + 1;
        });
      });
    _controller.repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    String dots = '';
    for (int i = 0; i < _dotCount; i++) {
      dots += '•';
    }
    return Text(
      dots,
      style: TextStyle(
        fontSize: 20,
        fontWeight: FontWeight.bold,
        color: Colors.grey[700],
      ),
    );
  }
}