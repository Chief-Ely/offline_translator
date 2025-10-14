import 'package:flutter/material.dart';
import 'onnx_translation.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ONNX Translation Test',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        useMaterial3: true,
      ),
      home: const TranslationTestPage(),
    );
  }
}

class TranslationTestPage extends StatefulWidget {
  const TranslationTestPage({super.key});

  @override
  State<TranslationTestPage> createState() => _TranslationTestPageState();
}

class _TranslationTestPageState extends State<TranslationTestPage> {
  final OnnxModel _model = OnnxModel();
  bool _isInitialized = false;
  bool _isLoading = false;
  
  final TextEditingController _inputController = TextEditingController();
  final TextEditingController _outputController = TextEditingController();
  final TextEditingController _tokenizationController = TextEditingController();
  final TextEditingController _tokenizationOutputController = TextEditingController();
  final TextEditingController _vocabDebugController = TextEditingController();
  
  String _statusMessage = 'Press "Initialize Model" to start';
  final ScrollController _scrollController = ScrollController();

  @override
  void initState() {
    super.initState();
    _initializeModel();
  }

  Future<void> _initializeModel() async {
    setState(() {
      _isLoading = true;
      _statusMessage = 'Loading model...';
      _vocabDebugController.text = 'Initializing...';
    });

    try {
      await _model.init(
        modelBasePath: 'assets/models/tagalog_to_cebuano',
      );
      
      // Test vocabulary after initialization using the new debug method
      _testVocabulary();
      
      setState(() {
        _isInitialized = true;
        _isLoading = false;
        _statusMessage = 'Model initialized successfully!';
      });
      
      // Set some example text
      _inputController.text = "maayong adlaw sa paggawas sa balay";
      _tokenizationController.text = "Kamusta ka";
      
    } catch (e) {
      setState(() {
        _isLoading = false;
        _statusMessage = 'Error initializing model: $e';
        _vocabDebugController.text = 'Error: $e';
      });
    }
  }

  void _testVocabulary() {
    try {
      // Use the new debugVocabulary method from OnnxModel
      final debugOutput = _model.debugVocabulary();
      
      setState(() {
        _vocabDebugController.text = debugOutput;
      });
    } catch (e) {
      setState(() {
        _vocabDebugController.text = 'Error testing vocabulary: $e';
      });
    }
  }

  Future<void> _translateText() async {
    if (!_isInitialized || _inputController.text.isEmpty) return;
    
    setState(() {
      _isLoading = true;
      _statusMessage = 'Translating...';
    });

    try {
      final translated = await _model.runModel(
        _inputController.text,
        maxNewTokens: 40,
      );
      
      setState(() {
        _outputController.text = translated;
        _isLoading = false;
        _statusMessage = 'Translation completed!';
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
        _statusMessage = 'Translation error: $e';
      });
    }
  }

  void _testTokenization() {
    if (!_isInitialized || _tokenizationController.text.isEmpty) return;
    
    try {
      final tokenIds = _model.tokenize(_tokenizationController.text);
      final tokens = tokenIds.map((id) => _model.reverseVocab[id] ?? '<unk>').toList();
      
      final result = '''
=== TOKENIZATION RESULTS ===
Input: "${_tokenizationController.text}"
Token IDs: $tokenIds
Tokens: $tokens
Detokenized: "${_model.detokenize(tokenIds)}"

=== TOKEN DETAILS ===
${_getTokenDetails(tokenIds)}

=== EXPECTED PYTHON OUTPUT ===
For "Kamusta ka": [921, 34088, 86, 17, 0]
Tokens: ['▁Ka', 'must', 'a', '▁ka', '</s>']
''';
      
      setState(() {
        _tokenizationOutputController.text = result;
        _statusMessage = 'Tokenization test completed!';
      });
    } catch (e) {
      setState(() {
        _tokenizationOutputController.text = 'Tokenization error: $e';
        _statusMessage = 'Tokenization test failed!';
      });
    }
  }

  String _getTokenDetails(List<int> tokenIds) {
    final details = StringBuffer();
    for (final id in tokenIds) {
      final token = _model.reverseVocab[id] ?? '<unk>';
      details.writeln('  $id: "$token"');
    }
    return details.toString();
  }

  void _clearAll() {
    setState(() {
      _inputController.clear();
      _outputController.clear();
      _tokenizationController.clear();
      _tokenizationOutputController.clear();
      _statusMessage = 'Cleared all fields';
    });
  }

  void _refreshVocabularyTest() {
    if (_isInitialized) {
      _testVocabulary();
      setState(() {
        _statusMessage = 'Vocabulary test refreshed!';
      });
    }
  }

  @override
  void dispose() {
    _model.release();
    _inputController.dispose();
    _outputController.dispose();
    _tokenizationController.dispose();
    _tokenizationOutputController.dispose();
    _vocabDebugController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('ONNX Translation Test'),
        backgroundColor: Colors.blue,
        foregroundColor: Colors.white,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _isLoading ? null : _initializeModel,
            tooltip: 'Reinitialize Model',
          ),
          IconButton(
            icon: const Icon(Icons.bug_report),
            onPressed: _isLoading ? null : _refreshVocabularyTest,
            tooltip: 'Refresh Vocabulary Test',
          ),
          IconButton(
            icon: const Icon(Icons.clear_all),
            onPressed: _clearAll,
            tooltip: 'Clear All',
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: SingleChildScrollView(
          controller: _scrollController,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Status Panel
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(12.0),
                  child: Row(
                    children: [
                      Icon(
                        _isInitialized ? Icons.check_circle : Icons.error,
                        color: _isInitialized ? Colors.green : Colors.orange,
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          _statusMessage,
                          style: TextStyle(
                            color: _isInitialized ? Colors.green : Colors.orange,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                      if (_isLoading)
                        const SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        ),
                    ],
                  ),
                ),
              ),
              
              const SizedBox(height: 16),
              
              // Vocabulary Debug Section
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      const Row(
                        children: [
                          Icon(Icons.bug_report, size: 18),
                          SizedBox(width: 8),
                          Text(
                            'Vocabulary Debug',
                            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                          ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      const Text(
                        'This shows if your vocabulary matches Python tokenizer IDs',
                        style: TextStyle(fontSize: 12, color: Colors.grey),
                      ),
                      const SizedBox(height: 12),
                      SizedBox(
                        height: 250, // Increased height for better visibility
                        child: TextField(
                          controller: _vocabDebugController,
                          decoration: const InputDecoration(
                            labelText: 'Vocabulary Comparison Results',
                            border: OutlineInputBorder(),
                            alignLabelWithHint: true,
                          ),
                          maxLines: null,
                          expands: true,
                          readOnly: true,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              
              const SizedBox(height: 16),
              
              // Translation Section
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      const Text(
                        'Translation Test',
                        style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(height: 12),
                      TextField(
                        controller: _inputController,
                        decoration: const InputDecoration(
                          labelText: 'Input Text to Translate',
                          border: OutlineInputBorder(),
                          hintText: 'Enter text in source language...',
                        ),
                        maxLines: 3,
                        minLines: 2,
                      ),
                      const SizedBox(height: 12),
                      ElevatedButton(
                        onPressed: _isInitialized && !_isLoading ? _translateText : null,
                        child: const Text('Translate'),
                      ),
                      const SizedBox(height: 12),
                      TextField(
                        controller: _outputController,
                        decoration: const InputDecoration(
                          labelText: 'Translation Output',
                          border: OutlineInputBorder(),
                          hintText: 'Translated text will appear here...',
                        ),
                        maxLines: 4,
                        minLines: 3,
                        readOnly: true,
                      ),
                    ],
                  ),
                ),
              ),
              
              const SizedBox(height: 16),
              
              // Tokenization Test Section
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      const Text(
                        'Tokenization Test',
                        style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(height: 12),
                      TextField(
                        controller: _tokenizationController,
                        decoration: const InputDecoration(
                          labelText: 'Text to Tokenize',
                          border: OutlineInputBorder(),
                          hintText: 'Enter text to test tokenization...',
                        ),
                        maxLines: 2,
                        minLines: 1,
                      ),
                      const SizedBox(height: 12),
                      ElevatedButton(
                        onPressed: _isInitialized ? _testTokenization : null,
                        child: const Text('Test Tokenization'),
                      ),
                      const SizedBox(height: 12),
                      SizedBox(
                        height: 250, // Increased height for better visibility
                        child: TextField(
                          controller: _tokenizationOutputController,
                          decoration: const InputDecoration(
                            labelText: 'Tokenization Results',
                            border: OutlineInputBorder(),
                            alignLabelWithHint: true,
                          ),
                          maxLines: null,
                          expands: true,
                          readOnly: true,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              
              const SizedBox(height: 16),
            ],
          ),
        ),
      ),
    );
  }
}