// import 'dart:convert';
// import 'dart:math';
// import 'package:flutter/services.dart';
// import 'package:onnxruntime/onnxruntime.dart';
// import 'dart:typed_data';
// import 'python_tokenizer_service.dart';

// class TokenizeResult {
//   final List<int> tokenIds;
//   final String debugInfo;

//   TokenizeResult(this.tokenIds, this.debugInfo);
// }

// class OnnxModel {
//   late OrtSession _encoderSession;
//   late OrtSession _decoderSession;
//   late OrtSession _decoderWithPastSession;
//   late Map<String, int> _vocab;
//   late Map<int, String> reverseVocab;
  
//   late int eosTokenId;
//   late int padTokenId;
//   late int unkTokenId;
  
//   bool _pythonTokenizerReady = false;

//   OnnxModel();

//   Future<void> initialize({
//     String? modelBasePath,
//     String? encoderAsset,
//     String? decoderAsset,
//     String? decoderWithPastAsset,
//     String? vocabAsset,
//     String? tokenizerConfigAsset,
//     String? generationConfigAsset,
//   }) async {
//     // Set default paths
//     String basePath = modelBasePath ?? 'assets/models/tagalog_to_cebuano';
    
//     encoderAsset ??= '$basePath/encoder_model.onnx';
//     decoderAsset ??= '$basePath/decoder_model.onnx';
//     decoderWithPastAsset ??= '$basePath/decoder_with_past_model.onnx';
//     vocabAsset ??= '$basePath/vocab.json';
//     tokenizerConfigAsset ??= '$basePath/tokenizer_config.json';
//     generationConfigAsset ??= '$basePath/generation_config.json';

//     // Initialize ONNX runtime
//     OrtEnv.instance.init();

//     // Load vocabulary
//     final vocabStr = await rootBundle.loadString(vocabAsset);
//     final vocabJson = jsonDecode(vocabStr) as Map<String, dynamic>;
//     _vocab = vocabJson.map((k, v) => MapEntry(k, (v as num).toInt()));
//     reverseVocab = Map.fromEntries(_vocab.entries.map((e) => MapEntry(e.value, e.key)));

//     // Initialize token IDs
//     _initializeTokenIds();

//     // Load tokenizer configs
//     await _loadTokenizerConfig(tokenizerConfigAsset);
//     await _loadGenerationConfig(generationConfigAsset);

//     // Try to initialize Python tokenizer
//     try {
//       await PythonTokenizerService.initialize();
//       await PythonTokenizerService.loadModel();
//       _pythonTokenizerReady = true;
//     } catch (e) {
//       _pythonTokenizerReady = false;
//     }

//     // Load ONNX models
//     await _loadModels(
//       encoderAsset: encoderAsset,
//       decoderAsset: decoderAsset,
//       decoderWithPastAsset: decoderWithPastAsset,
//     );
//   }

//   void _initializeTokenIds() {
//     eosTokenId = reverseVocab.entries
//         .firstWhere(
//           (e) => e.value == '</s>',
//           orElse: () => MapEntry(0, '</s>'),
//         )
//         .key;
//     unkTokenId = reverseVocab.entries
//         .firstWhere(
//           (e) => e.value == '<unk>',
//           orElse: () => MapEntry(1, '<unk>'),
//         )
//         .key;
//     padTokenId = reverseVocab.entries
//         .firstWhere(
//           (e) => e.value == '<pad>',
//           orElse: () => MapEntry(
//               (_vocab['<pad>'] ?? -1) >= 0 ? _vocab['<pad>']! : 0, '<pad>'),
//         )
//         .key;
//   }

//   Future<void> _loadTokenizerConfig(String tokenizerConfigAsset) async {
//     try {
//       final configStr = await rootBundle.loadString(tokenizerConfigAsset);
//       final config = jsonDecode(configStr) as Map<String, dynamic>;
      
//       if (config.containsKey('pad_token') && _vocab.containsKey(config['pad_token'])) {
//         padTokenId = _vocab[config['pad_token']!]!;
//       }
//       if (config.containsKey('unk_token') && _vocab.containsKey(config['unk_token'])) {
//         unkTokenId = _vocab[config['unk_token']!]!;
//       }
//     } catch (_) {
//       // Ignore if config file is missing
//     }
//   }

//   Future<void> _loadGenerationConfig(String generationConfigAsset) async {
//     try {
//       final configStr = await rootBundle.loadString(generationConfigAsset);
//       final config = jsonDecode(configStr) as Map<String, dynamic>;
      
//       if (config.containsKey('eos_token_id')) {
//         eosTokenId = (config['eos_token_id'] as num).toInt();
//       } else if (config.containsKey('eos_token') && _vocab.containsKey(config['eos_token'])) {
//         eosTokenId = _vocab[config['eos_token']!]!;
//       }
//     } catch (_) {
//       // Ignore if config file is missing
//     }

//     // Final fallback values
//     eosTokenId = eosTokenId >= 0 ? eosTokenId : 0;
//     padTokenId = padTokenId >= 0 ? padTokenId : eosTokenId;
//     unkTokenId = unkTokenId >= 0 ? unkTokenId : 1;
//   }

//   Future<void> _loadModels({
//     required String encoderAsset,
//     required String decoderAsset,
//     required String decoderWithPastAsset,
//   }) async {
//     final encData = await rootBundle.load(encoderAsset);
//     final decData = await rootBundle.load(decoderAsset);
//     final decPastData = await rootBundle.load(decoderWithPastAsset);
    
//     final encBytes = encData.buffer.asUint8List();
//     final decBytes = decData.buffer.asUint8List();
//     final decPastBytes = decPastData.buffer.asUint8List();

//     final sessionOptions = OrtSessionOptions();
//     _encoderSession = OrtSession.fromBuffer(encBytes, sessionOptions);
//     _decoderSession = OrtSession.fromBuffer(decBytes, sessionOptions);
//     _decoderWithPastSession = OrtSession.fromBuffer(decPastBytes, sessionOptions);
//   }

//   // Public getters
//   int get vocabSize => _vocab.length;

//   int getTokenId(String token) {
//     return _vocab[token] ?? unkTokenId;
//   }

//   // Tokenization methods
//   Future<List<int>> tokenize(String text) async {
//     if (_pythonTokenizerReady) {
//       try {
//         final result = await PythonTokenizerService.tokenize(text);
//         return List<int>.from(result['input_ids']);
//       } catch (e) {
//         _pythonTokenizerReady = false;
//         // Fall through to fallback tokenization
//       }
//     }
    
//     // Fallback tokenization
//     return _fallbackTokenize(text);
//   }

//   Future<TokenizeResult> tokenizeWithDebug(String text) async {
//     List<int> tokenIds;
//     String debugInfo;
    
//     if (_pythonTokenizerReady) {
//       try {
//         final result = await PythonTokenizerService.tokenize(text);
//         tokenIds = List<int>.from(result['input_ids']);
//         final tokens = result['tokens'] as List<dynamic>;
        
//         debugInfo = '''
// === PYTHON TOKENIZATION ===
// Input: "$text"
// Tokens: $tokens
// Token IDs: $tokenIds
// Status: ACTIVE
// ''';
//       } catch (e) {
//         tokenIds = await _fallbackTokenize(text);
//         debugInfo = '''
// === TOKENIZATION FALLBACK ===
// Input: "$text"
// Token IDs: $tokenIds
// Status: PYTHON TOKENIZER FAILED - USING FALLBACK
// Error: $e
// ''';
//       }
//     } else {
//       tokenIds = await _fallbackTokenize(text);
//       debugInfo = '''
// === FALLBACK TOKENIZATION ===
// Input: "$text"
// Token IDs: $tokenIds
// Status: PYTHON TOKENIZER NOT AVAILABLE
// ''';
//     }
    
//     debugInfo += '\nTokens: ${tokenIds.map((id) => reverseVocab[id] ?? "<unk>").toList()}';
//     debugInfo += '\nDetokenized: "${detokenize(tokenIds)}"';
    
//     return TokenizeResult(tokenIds, debugInfo);
//   }

//   Future<List<int>> _fallbackTokenize(String text) async {
//     final normalized = _normalizeText(text);
//     if (normalized.isEmpty) return [eosTokenId];

//     final tokenIds = <int>[];
    
//     // Simple word-based tokenization as fallback
//     final words = normalized.split(' ');
    
//     for (final word in words) {
//       if (word.isEmpty) continue;
      
//       int position = 0;
//       while (position < word.length) {
//         String? bestMatch;
//         int bestMatchLength = 0;

//         for (final token in _vocab.keys) {
//           if (token.isEmpty) continue;

//           final isFirstTokenInWord = (position == 0);
//           final hasPrefix = token.startsWith('▁');

//           if (isFirstTokenInWord != hasPrefix) continue;

//           final compareToken = hasPrefix ? token.substring(1) : token;

//           if (position + compareToken.length <= word.length) {
//             final substring = word.substring(position, position + compareToken.length);
//             if (substring.toLowerCase() == compareToken.toLowerCase()) {
//               if (compareToken.length > bestMatchLength) {
//                 bestMatch = token;
//                 bestMatchLength = compareToken.length;
//               }
//             }
//           }
//         }

//         if (bestMatch != null) {
//           tokenIds.add(_vocab[bestMatch]!);
//           position += bestMatchLength;
//         } else {
//           tokenIds.add(unkTokenId);
//           position++;
//         }
//       }
//     }

//     tokenIds.add(eosTokenId);
//     return tokenIds;
//   }

//   String _normalizeText(String text) {
//     return text.replaceAll(RegExp(r'\s+'), ' ').trim();
//   }

//   String debugVocabulary() {
//     final output = StringBuffer();
//     output.writeln('=== VOCABULARY INFO ===');
//     output.writeln('Vocabulary size: $vocabSize');
//     output.writeln('EOS Token: $eosTokenId');
//     output.writeln('PAD Token: $padTokenId');
//     output.writeln('UNK Token: $unkTokenId');
//     output.writeln('Python Tokenizer: ${_pythonTokenizerReady ? "READY" : "NOT AVAILABLE"}');
    
//     return output.toString();
//   }

//   // Text processing methods
//   String detokenize(List<int> tokenIds) {
//     final tokens = tokenIds.map((id) => reverseVocab[id] ?? '<unk>').toList();
//     final buffer = StringBuffer();

//     for (int i = 0; i < tokens.length; i++) {
//       final token = tokens[i];

//       if (token == '</s>') continue;

//       if (token.startsWith('<') && token.endsWith('>')) {
//         if (buffer.isNotEmpty) buffer.write(' ');
//         buffer.write(token);
//         continue;
//       }

//       if (token.startsWith('▁')) {
//         if (buffer.isNotEmpty) buffer.write(' ');
//         buffer.write(token.substring(1));
//       } else if (_isPunctuation(token)) {
//         buffer.write(token);
//       } else {
//         buffer.write(token);
//       }
//     }

//     return buffer.toString().trim();
//   }

//   bool _isPunctuation(String token) {
//     const punctuations = {
//       '.', ',', '!', '?', ':', ';', '-', '—', '(', ')', '[', ']', '"', '\''
//     };
//     return punctuations.contains(token);
//   }

//   // Model execution
//   Future<String> translate(
//     String inputText, {
//     String? languageToken,
//     int maxNewTokens = 50,
//   }) async {
//     try {
//       // Prepare input text
//       String textToTokenize = inputText.trim();
//       if (languageToken != null && languageToken.isNotEmpty) {
//         textToTokenize = '$languageToken $textToTokenize';
//       }

//       // Tokenize input
//       final inputIds = await tokenize(textToTokenize);
//       final sequenceLength = inputIds.length;
//       final attentionMask = List<int>.filled(sequenceLength, 1);

//       // Run encoder
//       final inputTensor = OrtValueTensor.createTensorWithDataList(
//         Int64List.fromList(inputIds),
//         [1, sequenceLength],
//       );
//       final attentionMaskTensor = OrtValueTensor.createTensorWithDataList(
//         Int64List.fromList(attentionMask),
//         [1, sequenceLength],
//       );

//       final encoderInputs = {
//         'input_ids': inputTensor,
//         'attention_mask': attentionMaskTensor,
//       };

//       final encoderOutputs = await _encoderSession.runAsync(
//         OrtRunOptions(),
//         encoderInputs,
//       );

//       if (encoderOutputs == null || encoderOutputs.isEmpty) {
//         throw Exception('Encoder returned no outputs');
//       }

//       final encoderHiddenStates = encoderOutputs[0];

//       // Generate translation
//       final generatedIds = <int>[];
//       var decoderInputIds = [padTokenId];

//       for (int step = 0; step < maxNewTokens; step++) {
//         final decoderInputTensor = OrtValueTensor.createTensorWithDataList(
//           Int64List.fromList(decoderInputIds),
//           [1, decoderInputIds.length],
//         );

//         final decoderInputs = {
//           'input_ids': decoderInputTensor,
//           'encoder_hidden_states': encoderHiddenStates!,
//           'encoder_attention_mask': attentionMaskTensor,
//         };

//         final decoderOutputs = await _decoderSession.runAsync(
//           OrtRunOptions(),
//           decoderInputs,
//         );

//         if (decoderOutputs == null || decoderOutputs.isEmpty) {
//           throw Exception('Decoder returned no outputs at step $step');
//         }

//         final logitsTensor = decoderOutputs[0];
//         final logits = _extractLastStepLogits(logitsTensor!);

//         if (logits.isEmpty) break;

//         final nextToken = _sampleToken(logits);

//         if (nextToken == eosTokenId) break;

//         generatedIds.add(nextToken);
//         decoderInputIds.add(nextToken);

//         if (generatedIds.length >= maxNewTokens) break;
//       }

//       return detokenize(generatedIds).trim();
//     } catch (e) {
//       rethrow;
//     }
//   }

//   // Helper methods for model execution
//   List<double> _extractLastStepLogits(OrtValue logitsTensor) {
//     try {
//       final raw = logitsTensor.value;
      
//       if (raw is List && raw.isNotEmpty) {
//         if (raw[0] is List) {
//           final batch = raw[0] as List<dynamic>;
//           if (batch.isNotEmpty) {
//             final lastStep = batch.last;
//             if (lastStep is List) {
//               return lastStep.map((e) => (e as num).toDouble()).toList();
//             }
//           }
//         }
//       }
      
//       return List.filled(56824, 0.0);
//     } catch (e) {
//       return List.filled(56824, 0.0);
//     }
//   }

//   int _sampleToken(List<double> logits) {
//     if (logits.isEmpty) return eosTokenId;
    
//     final probabilities = softmax(logits);
//     double maxProbability = probabilities[0];
//     int maxIndex = 0;
    
//     for (int i = 1; i < probabilities.length; i++) {
//       if (probabilities[i] > maxProbability) {
//         maxProbability = probabilities[i];
//         maxIndex = i;
//       }
//     }
    
//     return maxIndex;
//   }

//   List<double> softmax(List<double> logits) {
//     final maxValue = logits.reduce(max);
//     final exponentials = logits.map((logit) => exp(logit - maxValue)).toList();
//     final sumExponentials = exponentials.fold<double>(0.0, (a, b) => a + b);
//     return exponentials.map((exp) => exp / sumExponentials).toList();
//   }

//   void dispose() {
//     try {
//       _encoderSession.release();
//       _decoderSession.release();
//       _decoderWithPastSession.release();
//       OrtEnv.instance.release();
//     } catch (_) {
//       // Ignore release errors
//     }
//   }
// }

import 'dart:convert';
import 'dart:math';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'dart:typed_data';
import 'python_tokenizer_service.dart';

class TokenizeResult {
  final List<int> tokenIds;
  final String debugInfo;

  TokenizeResult(this.tokenIds, this.debugInfo);
}

class OnnxModel {
  late OrtSession _encoderSession;
  late OrtSession _decoderSession;
  late OrtSession _decoderWithPastSession;
  late Map<String, int> _vocab;
  late Map<int, String> reverseVocab;
  
  late int eosTokenId;
  late int padTokenId;
  late int unkTokenId;
  
  bool _pythonTokenizerReady = false;
  bool _isInitialized = false;

  OnnxModel();

  // Align with main.dart: change from initialize to init
  Future<void> init({
    required String modelBasePath,
    String? encoderAsset,
    String? decoderAsset,
    String? decoderWithPastAsset,
    String? vocabAsset,
    String? tokenizerConfigAsset,
    String? generationConfigAsset,
  }) async {
    return initialize(
      modelBasePath: modelBasePath,
      encoderAsset: encoderAsset,
      decoderAsset: decoderAsset,
      decoderWithPastAsset: decoderWithPastAsset,
      vocabAsset: vocabAsset,
      tokenizerConfigAsset: tokenizerConfigAsset,
      generationConfigAsset: generationConfigAsset,
    );
  }

  Future<void> initialize({
    String? modelBasePath,
    String? encoderAsset,
    String? decoderAsset,
    String? decoderWithPastAsset,
    String? vocabAsset,
    String? tokenizerConfigAsset,
    String? generationConfigAsset,
  }) async {
    // Set default paths
    String basePath = modelBasePath ?? 'assets/models/tagalog_to_cebuano';
    
    encoderAsset ??= '$basePath/encoder_model.onnx';
    decoderAsset ??= '$basePath/decoder_model.onnx';
    decoderWithPastAsset ??= '$basePath/decoder_with_past_model.onnx';
    vocabAsset ??= '$basePath/vocab.json';
    tokenizerConfigAsset ??= '$basePath/tokenizer_config.json';
    generationConfigAsset ??= '$basePath/generation_config.json';

    // Initialize ONNX runtime
    OrtEnv.instance.init();

    // Load vocabulary
    final vocabStr = await rootBundle.loadString(vocabAsset);
    final vocabJson = jsonDecode(vocabStr) as Map<String, dynamic>;
    _vocab = vocabJson.map((k, v) => MapEntry(k, (v as num).toInt()));
    reverseVocab = Map.fromEntries(_vocab.entries.map((e) => MapEntry(e.value, e.key)));

    // Initialize token IDs
    _initializeTokenIds();

    // Load tokenizer configs
    await _loadTokenizerConfig(tokenizerConfigAsset);
    await _loadGenerationConfig(generationConfigAsset);

    // Try to initialize Python tokenizer
    try {
      await PythonTokenizerService.initialize();
      await PythonTokenizerService.loadModel();
      _pythonTokenizerReady = true;
    } catch (e) {
      _pythonTokenizerReady = false;
    }

    // Load ONNX models
    await _loadModels(
      encoderAsset: encoderAsset,
      decoderAsset: decoderAsset,
      decoderWithPastAsset: decoderWithPastAsset,
    );

    _isInitialized = true;
  }

  void _initializeTokenIds() {
    eosTokenId = reverseVocab.entries
        .firstWhere(
          (e) => e.value == '</s>',
          orElse: () => MapEntry(0, '</s>'),
        )
        .key;
    unkTokenId = reverseVocab.entries
        .firstWhere(
          (e) => e.value == '<unk>',
          orElse: () => MapEntry(1, '<unk>'),
        )
        .key;
    padTokenId = reverseVocab.entries
        .firstWhere(
          (e) => e.value == '<pad>',
          orElse: () => MapEntry(
              (_vocab['<pad>'] ?? -1) >= 0 ? _vocab['<pad>']! : 0, '<pad>'),
        )
        .key;
  }

  Future<void> _loadTokenizerConfig(String tokenizerConfigAsset) async {
    try {
      final configStr = await rootBundle.loadString(tokenizerConfigAsset);
      final config = jsonDecode(configStr) as Map<String, dynamic>;
      
      if (config.containsKey('pad_token') && _vocab.containsKey(config['pad_token'])) {
        padTokenId = _vocab[config['pad_token']!]!;
      }
      if (config.containsKey('unk_token') && _vocab.containsKey(config['unk_token'])) {
        unkTokenId = _vocab[config['unk_token']!]!;
      }
    } catch (_) {
      // Ignore if config file is missing
    }
  }

  Future<void> _loadGenerationConfig(String generationConfigAsset) async {
    try {
      final configStr = await rootBundle.loadString(generationConfigAsset);
      final config = jsonDecode(configStr) as Map<String, dynamic>;
      
      if (config.containsKey('eos_token_id')) {
        eosTokenId = (config['eos_token_id'] as num).toInt();
      } else if (config.containsKey('eos_token') && _vocab.containsKey(config['eos_token'])) {
        eosTokenId = _vocab[config['eos_token']!]!;
      }
    } catch (_) {
      // Ignore if config file is missing
    }

    // Final fallback values
    eosTokenId = eosTokenId >= 0 ? eosTokenId : 0;
    padTokenId = padTokenId >= 0 ? padTokenId : eosTokenId;
    unkTokenId = unkTokenId >= 0 ? unkTokenId : 1;
  }

  Future<void> _loadModels({
    required String encoderAsset,
    required String decoderAsset,
    required String decoderWithPastAsset,
  }) async {
    final encData = await rootBundle.load(encoderAsset);
    final decData = await rootBundle.load(decoderAsset);
    final decPastData = await rootBundle.load(decoderWithPastAsset);
    
    final encBytes = encData.buffer.asUint8List();
    final decBytes = decData.buffer.asUint8List();
    final decPastBytes = decPastData.buffer.asUint8List();

    final sessionOptions = OrtSessionOptions();
    _encoderSession = OrtSession.fromBuffer(encBytes, sessionOptions);
    _decoderSession = OrtSession.fromBuffer(decBytes, sessionOptions);
    _decoderWithPastSession = OrtSession.fromBuffer(decPastBytes, sessionOptions);
  }

  // Public getters
  int get vocabSize => _vocab.length;

  int getTokenId(String token) {
    return _vocab[token] ?? unkTokenId;
  }

  // Tokenization methods - align with main.dart expectations
  TokenizeResult tokenize(String text) {
    // For synchronous tokenization as expected by main.dart
    final tokenIds = _fallbackTokenizeSync(text);
    final debugInfo = '''
=== TOKENIZATION RESULT ===
Input: "$text"
Token IDs: $tokenIds
Tokens: ${tokenIds.map((id) => reverseVocab[id] ?? "<unk>").toList()}
Detokenized: "${detokenize(tokenIds)}"
''';
    
    return TokenizeResult(tokenIds, debugInfo);
  }

  List<int> _fallbackTokenizeSync(String text) {
    final normalized = _normalizeText(text);
    if (normalized.isEmpty) return [eosTokenId];

    final tokenIds = <int>[];
    
    // Simple word-based tokenization as fallback
    final words = normalized.split(' ');
    
    for (final word in words) {
      if (word.isEmpty) continue;
      
      int position = 0;
      while (position < word.length) {
        String? bestMatch;
        int bestMatchLength = 0;

        for (final token in _vocab.keys) {
          if (token.isEmpty) continue;

          final isFirstTokenInWord = (position == 0);
          final hasPrefix = token.startsWith('▁');

          if (isFirstTokenInWord != hasPrefix) continue;

          final compareToken = hasPrefix ? token.substring(1) : token;

          if (position + compareToken.length <= word.length) {
            final substring = word.substring(position, position + compareToken.length);
            if (substring.toLowerCase() == compareToken.toLowerCase()) {
              if (compareToken.length > bestMatchLength) {
                bestMatch = token;
                bestMatchLength = compareToken.length;
              }
            }
          }
        }

        if (bestMatch != null) {
          tokenIds.add(_vocab[bestMatch]!);
          position += bestMatchLength;
        } else {
          tokenIds.add(unkTokenId);
          position++;
        }
      }
    }

    tokenIds.add(eosTokenId);
    return tokenIds;
  }

  // Align with main.dart: change from translate to runModel
  Future<String> runModel(
    String inputText, {
    int maxNewTokens = 50,
  }) async {
    return translate(inputText, maxNewTokens: maxNewTokens);
  }

  Future<String> translate(
    String inputText, {
    String? languageToken,
    int maxNewTokens = 50,
  }) async {
    try {
      // Prepare input text
      String textToTokenize = inputText.trim();
      if (languageToken != null && languageToken.isNotEmpty) {
        textToTokenize = '$languageToken $textToTokenize';
      }

      // Tokenize input
      final tokenizeResult = tokenize(textToTokenize);
      final inputIds = tokenizeResult.tokenIds;
      final sequenceLength = inputIds.length;
      final attentionMask = List<int>.filled(sequenceLength, 1);

      // Run encoder
      final inputTensor = OrtValueTensor.createTensorWithDataList(
        Int64List.fromList(inputIds),
        [1, sequenceLength],
      );
      final attentionMaskTensor = OrtValueTensor.createTensorWithDataList(
        Int64List.fromList(attentionMask),
        [1, sequenceLength],
      );

      final encoderInputs = {
        'input_ids': inputTensor,
        'attention_mask': attentionMaskTensor,
      };

      final encoderOutputs = await _encoderSession.runAsync(
        OrtRunOptions(),
        encoderInputs,
      );

      if (encoderOutputs == null || encoderOutputs.isEmpty) {
        throw Exception('Encoder returned no outputs');
      }

      final encoderHiddenStates = encoderOutputs[0];

      // Generate translation
      final generatedIds = <int>[];
      var decoderInputIds = [padTokenId];

      for (int step = 0; step < maxNewTokens; step++) {
        final decoderInputTensor = OrtValueTensor.createTensorWithDataList(
          Int64List.fromList(decoderInputIds),
          [1, decoderInputIds.length],
        );

        final decoderInputs = {
          'input_ids': decoderInputTensor,
          'encoder_hidden_states': encoderHiddenStates!,
          'encoder_attention_mask': attentionMaskTensor,
        };

        final decoderOutputs = await _decoderSession.runAsync(
          OrtRunOptions(),
          decoderInputs,
        );

        if (decoderOutputs == null || decoderOutputs.isEmpty) {
          throw Exception('Decoder returned no outputs at step $step');
        }

        final logitsTensor = decoderOutputs[0];
        final logits = _extractLastStepLogits(logitsTensor!);

        if (logits.isEmpty) break;

        final nextToken = _sampleToken(logits);

        if (nextToken == eosTokenId) break;

        generatedIds.add(nextToken);
        decoderInputIds.add(nextToken);

        if (generatedIds.length >= maxNewTokens) break;
      }

      return detokenize(generatedIds).trim();
    } catch (e) {
      rethrow;
    }
  }

  // Vocabulary debug method as expected by main.dart
  String debugVocabulary() {
    final output = StringBuffer();
    output.writeln('=== VOCABULARY INFO ===');
    output.writeln('Vocabulary size: $vocabSize');
    output.writeln('EOS Token: $eosTokenId (${reverseVocab[eosTokenId] ?? "unknown"})');
    output.writeln('PAD Token: $padTokenId (${reverseVocab[padTokenId] ?? "unknown"})');
    output.writeln('UNK Token: $unkTokenId (${reverseVocab[unkTokenId] ?? "unknown"})');
    output.writeln('Python Tokenizer: ${_pythonTokenizerReady ? "READY" : "NOT AVAILABLE"}');
    output.writeln('Model Initialized: $_isInitialized');
    
    // Test specific tokens mentioned in main.dart
    output.writeln('\n=== SPECIFIC TOKEN TEST ===');
    final testTokens = ['▁Ka', 'must', 'a', '▁ka', '</s>'];
    for (final token in testTokens) {
      final id = _vocab[token];
      output.writeln('  "$token": $id');
    }
    
    // Test the example from main.dart
    output.writeln('\n=== EXAMPLE TOKENIZATION TEST ===');
    final exampleText = "Kamusta ka";
    final exampleResult = tokenize(exampleText);
    output.writeln('Input: "$exampleText"');
    output.writeln('Token IDs: ${exampleResult.tokenIds}');
    output.writeln('Tokens: ${exampleResult.tokenIds.map((id) => reverseVocab[id] ?? "<unk>").toList()}');
    
    return output.toString();
  }

  String _normalizeText(String text) {
    return text.replaceAll(RegExp(r'\s+'), ' ').trim();
  }

  // Text processing methods
  String detokenize(List<int> tokenIds) {
    final tokens = tokenIds.map((id) => reverseVocab[id] ?? '<unk>').toList();
    final buffer = StringBuffer();

    for (int i = 0; i < tokens.length; i++) {
      final token = tokens[i];

      if (token == '</s>') continue;

      if (token.startsWith('<') && token.endsWith('>')) {
        if (buffer.isNotEmpty) buffer.write(' ');
        buffer.write(token);
        continue;
      }

      if (token.startsWith('▁')) {
        if (buffer.isNotEmpty) buffer.write(' ');
        buffer.write(token.substring(1));
      } else if (_isPunctuation(token)) {
        buffer.write(token);
      } else {
        buffer.write(token);
      }
    }

    return buffer.toString().trim();
  }

  bool _isPunctuation(String token) {
    const punctuations = {
      '.', ',', '!', '?', ':', ';', '-', '—', '(', ')', '[', ']', '"', '\''
    };
    return punctuations.contains(token);
  }

  // Helper methods for model execution
  List<double> _extractLastStepLogits(OrtValue logitsTensor) {
    try {
      final raw = logitsTensor.value;
      
      if (raw is List && raw.isNotEmpty) {
        if (raw[0] is List) {
          final batch = raw[0] as List<dynamic>;
          if (batch.isNotEmpty) {
            final lastStep = batch.last;
            if (lastStep is List) {
              return lastStep.map((e) => (e as num).toDouble()).toList();
            }
          }
        }
      }
      
      return List.filled(56824, 0.0);
    } catch (e) {
      return List.filled(56824, 0.0);
    }
  }

  int _sampleToken(List<double> logits) {
    if (logits.isEmpty) return eosTokenId;
    
    final probabilities = softmax(logits);
    double maxProbability = probabilities[0];
    int maxIndex = 0;
    
    for (int i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > maxProbability) {
        maxProbability = probabilities[i];
        maxIndex = i;
      }
    }
    
    return maxIndex;
  }

  List<double> softmax(List<double> logits) {
    final maxValue = logits.reduce(max);
    final exponentials = logits.map((logit) => exp(logit - maxValue)).toList();
    final sumExponentials = exponentials.fold<double>(0.0, (a, b) => a + b);
    return exponentials.map((exp) => exp / sumExponentials).toList();
  }

  // Align with main.dart: change from dispose to release
  void release() {
    dispose();
  }

  void dispose() {
    try {
      _encoderSession.release();
      _decoderSession.release();
      _decoderWithPastSession.release();
      OrtEnv.instance.release();
      _isInitialized = false;
    } catch (_) {
      // Ignore release errors
    }
  }
}