import 'dart:convert';
import 'package:serious_python/serious_python.dart';

class PythonTokenizerService {
  static bool _initialized = false;

  static Future<void> initialize() async {
    if (_initialized) return;
    _initialized = true;
  }

  static Future<Map<String, dynamic>> _sendCommand(String command, List<dynamic> args) async {
    if (!_initialized) {
      await initialize();
    }
    
    final input = json.encode({
      "command": command,
      "args": args,
    });
    
    try {
      // Correct usage - await the Future<String?> and handle null
      final String? result = await SeriousPython.run(
        'tokenizer_service.py',
        appFileName: 'tokenizer_service.py',
        environmentVariables: {'PYTHONPATH': 'assets/python'},
      );
      
      if (result == null) {
        throw Exception('Python script returned null');
      }
      
      return json.decode(result);
    } catch (e) {
      throw Exception('Python command failed: $e');
    }
  }

  static Future<void> loadModel() async {
    final modelPath = 'assets/models/tagalog_to_cebuano';
    final result = await _sendCommand("init", [modelPath]);
    if (result.containsKey('error')) {
      throw Exception(result['error']);
    }
  }

  static Future<Map<String, dynamic>> tokenize(String text) async {
    final result = await _sendCommand("tokenize", [text]);
    if (result.containsKey('error')) {
      throw Exception(result['error']);
    }
    return result;
  }
}