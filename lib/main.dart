import 'package:flutter/material.dart';
import 'package:serious_python/serious_python.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Tagalog → Cebuano Translator',
      home: TranslationPage(),
    );
  }
}

class TranslationPage extends StatefulWidget {
  @override
  _TranslationPageState createState() => _TranslationPageState();
}

class _TranslationPageState extends State<TranslationPage> {
  final TextEditingController _controller = TextEditingController();
  String _translation = '';
  bool _pythonFailed = false;

  @override
  void initState() {
    super.initState();
    _initPython();
  }

  Future<void> _initPython() async {
    try {
      // 1️⃣ Initialize Python environment (optional, if needed)
      await SeriousPython.run(
        'assets/pyTranslator',
        appFileName: 'translator_entry.py',
        modulePaths: ['assets/pyTranslator'],
        environmentVariables: {
          'COMMAND': 'init',
          'MODEL_PATH': 'assets/pyTranslator/tagalog_to_cebuano', // path to your model
        },
        sync: true,
      );

      // ✅ Now Python is initialized, no need to call init again
    } catch (_) {
      setState(() => _pythonFailed = true);
    }
  }


  Future<void> _translateText() async {
    if (_controller.text.isEmpty) return;

    try {
      final result = await SeriousPython.run(
        'assets/pyTranslator',
        appFileName: 'translator_entry.py',
        modulePaths: ['assets/pyTranslator'],
        environmentVariables: {
          'COMMAND': 'translate',        // command: init or translate
          'USER_TEXT': _controller.text, // text to translate
          'MODEL_PATH': 'assets/pyTranslator/tagalog_to_cebuano', // only needed for init
        },
        sync: true,
      );

      setState(() => _translation = result ?? '');
    } catch (_) {
      setState(() {
        _translation = "Python translation failed";
        _pythonFailed = true;
      });
    }
  }


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Tagalog → Cebuano Translator')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            TextField(
              controller: _controller,
              decoration: InputDecoration(
                labelText: _pythonFailed
                    ? 'Python failed. Type fallback text'
                    : 'Enter Tagalog text',
              ),
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: _translateText,
              child: Text('Translate'),
            ),
            SizedBox(height: 20),
            Text('Translation: $_translation'),
          ],
        ),
      ),
    );
  }
}
