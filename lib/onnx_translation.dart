import 'dart:convert';
import 'dart:math';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'dart:typed_data';

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
  late Map<int, String> reverseVocab; // Made public for the UI
  
  late int eosTokenId;
  late int padTokenId;
  late int unkTokenId;
  
  final RegExp _specialTokenRegex = RegExp(r'<[^>]+>');

  OnnxModel();

  Future<void> init({
    String? modelBasePath,
    String? encoderAsset,
    String? decoderAsset,
    String? decoderWithPastAsset,
    String? vocabAsset,
    String? tokenizerConfigAsset,
    String? generationConfigAsset,
  }) async {
    // Path composition
    encoderAsset ??= modelBasePath != null
        ? '${modelBasePath.endsWith('/') ? modelBasePath : '$modelBasePath/'}encoder_model.onnx'
        : 'assets/onnx_model/encoder_model.onnx';

    decoderAsset ??= modelBasePath != null
        ? '${modelBasePath.endsWith('/') ? modelBasePath : '$modelBasePath/'}decoder_model.onnx'
        : 'assets/onnx_model/decoder_model.onnx';

    decoderWithPastAsset ??= modelBasePath != null
        ? '${modelBasePath.endsWith('/') ? modelBasePath : '$modelBasePath/'}decoder_with_past_model.onnx'
        : 'assets/onnx_model/decoder_with_past_model.onnx';

    vocabAsset ??= modelBasePath != null
        ? '${modelBasePath.endsWith('/') ? modelBasePath : '$modelBasePath/'}vocab.json'
        : 'assets/onnx_model/vocab.json';

    tokenizerConfigAsset ??= modelBasePath != null
        ? '${modelBasePath.endsWith('/') ? modelBasePath : '$modelBasePath/'}tokenizer_config.json'
        : 'assets/onnx_model/tokenizer_config.json';

    generationConfigAsset ??= modelBasePath != null
        ? '${modelBasePath.endsWith('/') ? modelBasePath : '$modelBasePath/'}generation_config.json'
        : 'assets/onnx_model/generation_config.json';

    OrtEnv.instance.init();

    // Load vocabulary JSON
    final vocabStr = await rootBundle.loadString(vocabAsset);
    final vocabJson = jsonDecode(vocabStr) as Map<String, dynamic>;
    _vocab = vocabJson.map((k, v) => MapEntry(k, (v as num).toInt()));
    reverseVocab = Map.fromEntries(_vocab.entries.map((e) => MapEntry(e.value, e.key)));

    // DEBUG: Print vocabulary information
    print('=== VOCABULARY DEBUG INFO ===');
    print('Vocabulary size: ${_vocab.length}');
    
    // Print first few entries to check format
    print('First 10 vocabulary entries:');
    final sampleEntries = _vocab.entries.take(10).toList();
    for (final entry in sampleEntries) {
      print('  "${entry.key}" -> ${entry.value}');
    }

    // Check critical tokens
    final criticalTokens = ['▁Ka', 'must', 'a', '▁ka'];
    print('Critical token check:');
    for (final token in criticalTokens) {
      if (_vocab.containsKey(token)) {
        print('  ✅ "$token" -> ${_vocab[token]}');
      } else {
        print('  ❌ "$token" NOT FOUND in vocabulary!');
      }
    }

    // Initialize token IDs with sane defaults
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

    // Attempt to load tokenizer_config.json for pad/unk tokens
    try {
      final tcfg = await rootBundle.loadString(tokenizerConfigAsset);
      final tc = jsonDecode(tcfg) as Map<String, dynamic>;
      if (tc.containsKey('pad_token')) {
        final padTok = tc['pad_token'] as String;
        if (_vocab.containsKey(padTok)) padTokenId = _vocab[padTok]!;
      }
      if (tc.containsKey('unk_token')) {
        final unkTok = tc['unk_token'] as String;
        if (_vocab.containsKey(unkTok)) unkTokenId = _vocab[unkTok]!;
      }
    } catch (_) {
      // Ignore if tokenizer_config.json missing
    }

    // Attempt to load generation_config.json for eos token id or token string
    try {
      final gcfg = await rootBundle.loadString(generationConfigAsset);
      final gc = jsonDecode(gcfg) as Map<String, dynamic>;
      if (gc.containsKey('eos_token_id')) {
        eosTokenId = (gc['eos_token_id'] as num).toInt();
      } else if (gc.containsKey('eos_token')) {
        final tok = gc['eos_token'] as String;
        if (_vocab.containsKey(tok)) eosTokenId = _vocab[tok]!;
      }
    } catch (_) {
      // Ignore if generation_config.json missing
    }

    // Final sanity fallback
    eosTokenId = eosTokenId >= 0 ? eosTokenId : 0;
    padTokenId = padTokenId >= 0 ? padTokenId : eosTokenId;
    unkTokenId = unkTokenId >= 0 ? unkTokenId : 1;

    print('Final token IDs - EOS: $eosTokenId, PAD: $padTokenId, UNK: $unkTokenId');

    // Load ONNX models
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

  // Add these getters and methods:
  int get vocabSize => _vocab.length;

  int getTokenId(String token) {
    return _vocab[token] ?? unkTokenId;
  }

  /// Debug method to check specific tokens
  String debugVocabulary() {
    final output = StringBuffer();
    output.writeln('=== VOCABULARY DEBUG ===');
    output.writeln('Total vocabulary size: ${_vocab.length}');
    output.writeln('Special tokens - EOS: $eosTokenId, PAD: $padTokenId, UNK: $unkTokenId');
    output.writeln('');
    
    // Test tokens from Python output
    final testTokens = ['▁Ka', 'must', 'a', '▁ka', '▁Kamusta'];
    output.writeln('=== TOKEN ID CHECK ===');
    
    for (final token in testTokens) {
      final foundId = _vocab[token];
      final expectedId = _getExpectedTokenId(token);
      output.writeln('Token: "$token"');
      output.writeln('  Expected: $expectedId, Found: $foundId, Match: ${foundId == expectedId}');
      if (foundId == null) {
        output.writeln('  ⚠️  TOKEN NOT FOUND IN VOCABULARY!');
      } else if (foundId != expectedId) {
        output.writeln('  ❌ ID MISMATCH!');
      } else {
        output.writeln('  ✅ Correct');
      }
    }
    
    return output.toString();
  }

  /// Expected token IDs from Python output
  int? _getExpectedTokenId(String token) {
    final expected = {
      '▁Ka': 921,
      'must': 34088,
      'a': 86,
      '▁ka': 17,
      '▁Kamusta': null, // We don't have this from Python output
    };
    return expected[token];
  }

  /// Tokenizes input text into token IDs with debug info
  // TokenizeResult tokenize(String text) {
  //   final normalized = text.replaceAll(RegExp(r'\s+'), ' ').trim();
  //   if (normalized.isEmpty) {
  //     return TokenizeResult([eosTokenId], 'Empty input');
  //   }

  //   final tokenIds = <int>[];
  //   final debugInfo = StringBuffer();
  //   final words = normalized.split(' ');

  //   debugInfo.writeln('=== TOKENIZATION DEBUG ===');
  //   debugInfo.writeln('Input: "$text"');
  //   debugInfo.writeln('Words: $words');
  //   debugInfo.writeln('');

  //   for (int wordIndex = 0; wordIndex < words.length; wordIndex++) {
  //     final word = words[wordIndex];
  //     if (word.isEmpty) continue;

  //     debugInfo.writeln('Word: "$word"');
  //     int pos = 0;

  //     while (pos < word.length) {
  //       String? bestMatch;
  //       int bestMatchLength = 0;

  //       for (final token in _vocab.keys) {
  //         if (token.isEmpty) continue;

  //         final isFirstTokenInWord = (pos == 0);
  //         final hasPrefix = token.startsWith('▁');

  //         if (isFirstTokenInWord != hasPrefix) continue;

  //         final compareToken = hasPrefix ? token.substring(1) : token;

  //         if (pos + compareToken.length <= word.length) {
  //           final substring = word.substring(pos, pos + compareToken.length);
  //           if (substring == compareToken) {
  //             if (compareToken.length > bestMatchLength) {
  //               bestMatch = token;
  //               bestMatchLength = compareToken.length;
  //             }
  //           }
  //         }
  //       }

  //       if (bestMatch != null) {
  //         final tokenId = _vocab[bestMatch]!;
  //         tokenIds.add(tokenId);
  //         debugInfo.writeln('  POS $pos: "$bestMatch" -> $tokenId');
  //         pos += bestMatchLength;
  //       } else {
  //         tokenIds.add(unkTokenId);
  //         debugInfo.writeln('  POS $pos: No match -> UNK');
  //         pos++;
  //       }
  //     }
  //   }

  //   tokenIds.add(eosTokenId);
    
  //   debugInfo.writeln('');
  //   debugInfo.writeln('=== FINAL RESULT ===');
  //   debugInfo.writeln('Token IDs: $tokenIds');
  //   debugInfo.writeln('Tokens: ${tokenIds.map((id) => reverseVocab[id] ?? "<unk>").toList()}');

  //   return TokenizeResult(tokenIds, debugInfo.toString());
  // }
  /// Tokenizes input text into token IDs - Python-compatible version
  TokenizeResult tokenize(String text) {
    final normalized = text.replaceAll(RegExp(r'\s+'), ' ').trim();
    if (normalized.isEmpty) {
      return TokenizeResult([eosTokenId], 'Empty input');
    }

    final tokenIds = <int>[];
    final debugInfo = StringBuffer();

    // Add space at beginning to help with '▁' token matching (like Python does)
    final textWithSpace = ' $normalized';
    int pos = 1;

    debugInfo.writeln('=== TOKENIZATION DEBUG ===');
    debugInfo.writeln('Input: "$text"');
    debugInfo.writeln('Processing: "$textWithSpace"');
    debugInfo.writeln('');

    while (pos < textWithSpace.length) {
      if (textWithSpace[pos] == ' ') {
        pos++;
        continue;
      }

      String? bestMatch;
      int bestMatchLength = 0;

      for (final token in _vocab.keys) {
        if (token.isEmpty) continue;

        // Exact matching like Python tokenizer
        if (pos + token.length <= textWithSpace.length) {
          final substring = textWithSpace.substring(pos, pos + token.length);
          if (substring == token) {
            if (token.length > bestMatchLength) {
              bestMatch = token;
              bestMatchLength = token.length;
            }
          }
        }
      }

      if (bestMatch != null) {
        final tokenId = _vocab[bestMatch]!;
        tokenIds.add(tokenId);
        debugInfo.writeln('  POS $pos: "$bestMatch" -> $tokenId');
        pos += bestMatchLength;
      } else {
        tokenIds.add(unkTokenId);
        debugInfo.writeln('  POS $pos: No match -> UNK');
        pos++;
      }
    }

    tokenIds.add(eosTokenId);

    debugInfo.writeln('');
    debugInfo.writeln('=== FINAL RESULT ===');
    debugInfo.writeln('Token IDs: $tokenIds');
    debugInfo.writeln(
      'Tokens: ${tokenIds.map((id) => reverseVocab[id] ?? "<unk>").toList()}',
    );

    return TokenizeResult(tokenIds, debugInfo.toString());
  }
  
  /// Checks if a token string is punctuation.
  bool _isPunctuation(String token) {
    const punctuations = {
      '.', ',', '!', '?', ':', ';', '-', '—', '(', ')', '[', ']', '"', '\''
    };
    return punctuations.contains(token);
  }

  /// Converts token IDs back into a readable string.
  String detokenize(List<int> tokenIds) {
    final tokens = tokenIds.map((id) => reverseVocab[id] ?? '<unk>').toList();
    final buffer = StringBuffer();

    for (int i = 0; i < tokens.length; i++) {
      final tok = tokens[i];

      if (tok == '</s>') continue;

      if (tok.startsWith('<') && tok.endsWith('>')) {
        if (buffer.isNotEmpty) buffer.write(' ');
        buffer.write(tok);
        continue;
      }

      if (tok.startsWith('▁')) {
        if (buffer.isNotEmpty) buffer.write(' ');
        buffer.write(tok.substring(1));
      } else if (_isPunctuation(tok)) {
        buffer.write(tok);
      } else {
        buffer.write(tok);
      }
    }

    return buffer.toString().trim();
  }

  /// Applies softmax function on the list of logits.
  List<double> softmax(List<double> logits) {
    final maxValue = logits.reduce(max);
    final exps = logits.map((l) => exp(l - maxValue)).toList();
    final sumExp = exps.fold<double>(0.0, (a, b) => a + b);
    return exps.map((e) => e / sumExp).toList();
  }

  //Simple fix
  Future<String> runModel(
    String inputText, {
    String? initialLangToken,
    int maxNewTokens = 50,
  }) async {
    try {
      print('=== STARTING TRANSLATION ===');
      print('Input: "$inputText"');

      // 1️⃣ Tokenize input
      String textToTokenize = inputText.trim();
      if (initialLangToken != null && initialLangToken.isNotEmpty) {
        textToTokenize = '$initialLangToken $textToTokenize';
      }

      final result = tokenize(textToTokenize);
      final inputIds = result.tokenIds;
      final seqLen = inputIds.length;
      final attentionMask = List<int>.filled(seqLen, 1);

      print('Token IDs: $inputIds');
      print('Sequence length: $seqLen');

      // 2️⃣ Run encoder - MATCH EXACT SPECIFICATIONS
      final inputTensor = OrtValueTensor.createTensorWithDataList(
        Int64List.fromList(inputIds), // int64 for input_ids
        [1, seqLen], // batch_size, encoder_sequence_length
      );
      final attentionMaskTensor = OrtValueTensor.createTensorWithDataList(
        Int64List.fromList(attentionMask), // int64 for attention_mask
        [1, seqLen], // batch_size, encoder_sequence_length
      );

      final encoderInputs = {
        'input_ids': inputTensor,
        'attention_mask': attentionMaskTensor,
      };

      print('Running encoder...');
      final encoderOutputs = await _encoderSession.runAsync(
        OrtRunOptions(),
        encoderInputs,
      );

      if (encoderOutputs == null || encoderOutputs.isEmpty) {
        throw Exception('Encoder returned no outputs');
      }

      final encoderHiddenStates =
          encoderOutputs[0]; // last_hidden_state: float32
      print('Encoder completed - hidden states shape: [1, $seqLen, 512]');

      // 3️⃣ Generate translation using decoder
      final generatedIds = <int>[];
      var decoderInputIds = [padTokenId];

      print('Starting generation with decoder...');

      for (int step = 0; step < maxNewTokens; step++) {
        // Create decoder inputs - MATCH EXACT SPECIFICATIONS
        final decInputTensor = OrtValueTensor.createTensorWithDataList(
          Int64List.fromList(decoderInputIds), // int64 for input_ids
          [1, decoderInputIds.length], // batch_size, decoder_sequence_length
        );

        // Note: decoder expects 'encoder_attention_mask' not 'attention_mask'
        final decoderInputs = {
          'input_ids': decInputTensor,
          'encoder_hidden_states': encoderHiddenStates!, // float32
          'encoder_attention_mask': attentionMaskTensor, // int64 - CORRECT NAME
        };

        final decoderOutputs = await _decoderSession.runAsync(
          OrtRunOptions(),
          decoderInputs,
        );

        if (decoderOutputs == null || decoderOutputs.isEmpty) {
          throw Exception('Decoder returned no outputs at step $step');
        }

        // Extract logits - first output is logits: float32
        final logitsTensor = decoderOutputs[0];
        final logits = _extractLastStepLogits(logitsTensor!);

        if (logits.isEmpty) {
          print('Warning: Empty logits at step $step');
          break;
        }

        // Sample next token
        final nextToken = _sampleToken(logits);
        final tokenText = reverseVocab[nextToken] ?? '<unk>';

        print('Step $step: $nextToken ("$tokenText")');

        // Stop conditions
        if (nextToken == eosTokenId) {
          print('Generated EOS token, stopping.');
          break;
        }

        generatedIds.add(nextToken);
        decoderInputIds.add(nextToken);

        // Additional stop condition for very long sequences
        if (generatedIds.length >= maxNewTokens) {
          print('Reached max token limit, stopping.');
          break;
        }
      }

      final translation = detokenize(generatedIds).trim();
      print('=== TRANSLATION COMPLETE ===');
      print('Final: "$translation"');
      print('Generated IDs: $generatedIds');

      return translation;
    } catch (e) {
      print('=== TRANSLATION FAILED ===');
      print('Error: $e');
      print('Stack trace: ${e.toString()}');
      rethrow;
    }
  }

  List<double> _extractLastStepLogits(OrtValue logitsTensor) {
    try {
      final raw = logitsTensor.value;
      
      if (raw is List && raw.isNotEmpty) {
        // Handle 3D tensor: [batch_size, decoder_sequence_length, vocab_size]
        if (raw[0] is List) {
          final batch = raw[0] as List<dynamic>; // batch_size = 1
          if (batch.isNotEmpty) {
            // Get the last token in the sequence
            final lastStep = batch.last;
            if (lastStep is List) {
              return lastStep.map((e) => (e as num).toDouble()).toList();
            }
          }
          // If we can't get last step, try first step as fallback
          if (batch.isNotEmpty && batch[0] is List) {
            return (batch[0] as List<dynamic>).map((e) => (e as num).toDouble()).toList();
          }
        }
      }
      
      print('Warning: Could not extract logits properly, using fallback');
      return List.filled(56824, 0.0); // vocab_size = 56824
      
    } catch (e) {
      print('Error extracting logits: $e');
      return List.filled(56824, 0.0);
    }
  }

  /// Sample token using greedy decoding
  int _sampleToken(List<double> logits) {
    if (logits.isEmpty) return eosTokenId;
    
    final probs = softmax(logits);
    double maxProb = probs[0];
    int maxIndex = 0;
    
    for (int i = 1; i < probs.length; i++) {
      if (probs[i] > maxProb) {
        maxProb = probs[i];
        maxIndex = i;
      }
    }
    
    return maxIndex;
  }

  void release() {
    try {
      _encoderSession.release();
      _decoderSession.release();
      _decoderWithPastSession.release();
      OrtEnv.instance.release();
    } catch (_) {}
  }
}