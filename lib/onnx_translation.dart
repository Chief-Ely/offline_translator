import 'dart:convert';
import 'dart:math';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'dart:typed_data';

// class OnnxModel {
//   late OrtSession _encoderSession;
//   late OrtSession _decoderSession;
//   late OrtSession _decoderWithPastSession;
//   late Map<String, int> _vocab;
//   late Map<int, String> reverseVocab;
  
//   late int eosTokenId;
//   late int padTokenId;
//   late int unkTokenId;
  
//   final RegExp _specialTokenRegex = RegExp(r'<[^>]+>');

//   OnnxModel();

//   Future<void> init({
//     String? modelBasePath,
//     String? encoderAsset,
//     String? decoderAsset,
//     String? decoderWithPastAsset,
//     String? vocabAsset,
//     String? tokenizerConfigAsset,
//     String? generationConfigAsset,
//   }) async {
//     // Path composition (same as before)
//     encoderAsset ??= modelBasePath != null
//         ? '${modelBasePath.endsWith('/') ? modelBasePath : '$modelBasePath/'}encoder_model.onnx'
//         : 'assets/onnx_model/encoder_model.onnx';

//     decoderAsset ??= modelBasePath != null
//         ? '${modelBasePath.endsWith('/') ? modelBasePath : '$modelBasePath/'}decoder_model.onnx'
//         : 'assets/onnx_model/decoder_model.onnx';

//     decoderWithPastAsset ??= modelBasePath != null
//         ? '${modelBasePath.endsWith('/') ? modelBasePath : '$modelBasePath/'}decoder_with_past_model.onnx'
//         : 'assets/onnx_model/decoder_with_past_model.onnx';

//     vocabAsset ??= modelBasePath != null
//         ? '${modelBasePath.endsWith('/') ? modelBasePath : '$modelBasePath/'}vocab.json'
//         : 'assets/onnx_model/vocab.json';

//     // Load vocabulary and configs (same as before)
//     final vocabStr = await rootBundle.loadString(vocabAsset);
//     final vocabJson = jsonDecode(vocabStr) as Map<String, dynamic>;
//     _vocab = vocabJson.map((k, v) => MapEntry(k, (v as num).toInt()));
//     reverseVocab = Map.fromEntries(_vocab.entries.map((e) => MapEntry(e.value, e.key)));

//     // Initialize token IDs (same as before)
//     eosTokenId = reverseVocab.entries.firstWhere((e) => e.value == '</s>', orElse: () => MapEntry(0, '</s>')).key;
//     unkTokenId = reverseVocab.entries.firstWhere((e) => e.value == '<unk>', orElse: () => MapEntry(1, '<unk>')).key;
//     padTokenId = reverseVocab.entries.firstWhere((e) => e.value == '<pad>', orElse: () => MapEntry((_vocab['<pad>'] ?? -1) >= 0 ? _vocab['<pad>']! : 0, '<pad>')).key;

//     // Load config files and override token IDs if available...

//     // Load ONNX models
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

//   // tokenize and detokenize methods remain the same...

//   Future<String> runModel(
//     String inputText, {
//     String? initialLangToken,
//     int maxNewTokens = 50,
//   }) async {
//     // 1️⃣ Prepare and tokenize input
//     String textToTokenize = inputText.trim();
//     if (initialLangToken != null && initialLangToken.isNotEmpty) {
//       textToTokenize = '$initialLangToken $textToTokenize';
//     }

//     final inputIds = tokenize(textToTokenize);
//     final seqLen = inputIds.length;
//     final attentionMask = List<int>.filled(seqLen, 1);

//     // 2️⃣ Run encoder
//     final inputTensor = OrtValueTensor.createTensorWithDataList(
//         Int64List.fromList(inputIds), [1, seqLen]);
//     final attentionMaskTensor = OrtValueTensor.createTensorWithDataList(
//         Int64List.fromList(attentionMask), [1, seqLen]);

//     final encoderInputs = {
//       'input_ids': inputTensor,
//       'attention_mask': attentionMaskTensor,
//     };

//     final encoderOutputs = await _encoderSession.runAsync(OrtRunOptions(), encoderInputs);
//     final encoderHiddenStates = encoderOutputs![0];

//     // 3️⃣ Initialize decoder state
//     final generatedIds = <int>[];
//     var decoderInputIds = [padTokenId]; // Start with pad token
    
//     // Store past key values for each layer
//     List<OrtValue?> pastKeyValues = List.filled(12, null); // 6 layers * 2 (key + value)

//     // 4️⃣ First decoder step (using decoder_model.onnx)
//     var decInputTensor = OrtValueTensor.createTensorWithDataList(
//         Int64List.fromList(decoderInputIds), [1, decoderInputIds.length]);

//     var decoderInputs = {
//       'input_ids': decInputTensor,
//       'encoder_hidden_states': encoderHiddenStates!,
//       'encoder_attention_mask': attentionMaskTensor,
//     };

//     var decoderOutputs = await _decoderSession.runAsync(OrtRunOptions(), decoderInputs);
    
//     // Extract logits and past key values from first step
//     final firstLogitsTensor = decoderOutputs![0];
//     // Store past key values (outputs 1-12 are the past key values)
//     for (int i = 0; i < 12; i++) {
//       pastKeyValues[i] = decoderOutputs[i + 1];
//     }

//     // 5️⃣ Get first token
//     final firstLogits = _extractLastStepLogits(firstLogitsTensor!);
//     int nextToken = _sampleToken(firstLogits);
    
//     if (nextToken != eosTokenId) {
//       generatedIds.add(nextToken);
//       decoderInputIds.add(nextToken);
//     } else {
//       return detokenize(generatedIds);
//     }

//     // 6️⃣ Subsequent steps using decoder_with_past_model.onnx
//     for (int step = 1; step < maxNewTokens; step++) {
//       // Prepare inputs for decoder_with_past
//       final currentInputTensor = OrtValueTensor.createTensorWithDataList(
//           Int64List.fromList([nextToken]), [1, 1]);

//       final pastDecoderInputs = {
//         'input_ids': currentInputTensor,
//         'encoder_attention_mask': attentionMaskTensor,
//         'encoder_hidden_states': encoderHiddenStates,
//       };

//       // Add past key values to inputs
//       for (int i = 0; i < 12; i++) {
//         pastDecoderInputs['past_key_values.$i.decoder.key'] = pastKeyValues[i]!;
//         pastDecoderInputs['past_key_values.$i.decoder.value'] = pastKeyValues[i + 12]!;
//         pastDecoderInputs['past_key_values.$i.encoder.key'] = pastKeyValues[i + 24]!;
//         pastDecoderInputs['past_key_values.$i.encoder.value'] = pastKeyValues[i + 36]!;
//       }

//       final pastDecoderOutputs = await _decoderWithPastSession.runAsync(
//           OrtRunOptions(), pastDecoderInputs);

//       // Update past key values
//       final newLogitsTensor = pastDecoderOutputs![0];
//       for (int i = 0; i < 12; i++) {
//         pastKeyValues[i] = pastDecoderOutputs[i + 1];
//       }

//       // Sample next token
//       final logits = _extractLastStepLogits(newLogitsTensor!);
//       nextToken = _sampleToken(logits);

//       if (nextToken == eosTokenId) break;
      
//       generatedIds.add(nextToken);
//     }

//     return detokenize(generatedIds).trim();
//   }

//   /// Helper method to extract logits from the last token position
//   List<double> _extractLastStepLogits(OrtValue logitsTensor) {
//     final raw = logitsTensor.value;
    
//     if (raw is List && raw.isNotEmpty) {
//       // Handle 3D tensor: [batch_size, sequence_length, vocab_size]
//       if (raw[0] is List && (raw[0] as List).isNotEmpty) {
//         final batch = raw[0] as List<dynamic>;
//         final lastStep = batch.last as List<dynamic>;
//         return lastStep.map((e) => (e as num).toDouble()).toList();
//       }
//       // Handle 2D tensor: [batch_size, vocab_size]  
//       else if (raw[0] is num) {
//         return raw.map((e) => (e as num).toDouble()).toList();
//       }
//     }
    
//     // Fallback: return empty list
//     return [];
//   }

//   /// Sample token using greedy decoding (same as your softmax approach)
//   int _sampleToken(List<double> logits) {
//     if (logits.isEmpty) return eosTokenId;
    
//     final probs = softmax(logits);
//     double maxProb = probs[0];
//     int maxIndex = 0;
    
//     for (int i = 1; i < probs.length; i++) {
//       if (probs[i] > maxProb) {
//         maxProb = probs[i];
//         maxIndex = i;
//       }
//     }
    
//     return maxIndex;
//   }

//   // softmax method remains the same...
//   List<double> softmax(List<double> logits) {
//     final maxValue = logits.reduce(max);
//     final exps = logits.map((l) => exp(l - maxValue)).toList();
//     final sumExp = exps.fold<double>(0.0, (a, b) => a + b);
//     return exps.map((e) => e / sumExp).toList();
//   }

//   void release() {
//     try {
//       _encoderSession.release();
//       _decoderSession.release();
//       _decoderWithPastSession.release();
//       OrtEnv.instance.release();
//     } catch (_) {}
//   }
// }

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

  // COPY THESE FROM YOUR ORIGINAL IMPLEMENTATION:
  
  /// Tokenizes input text into token IDs.
  List<int> tokenize(String text) {
    final normalized = text.replaceAll(RegExp(r'\s+'), ' ').trim();
    if (normalized.isEmpty) return [eosTokenId];

    final tokenIds = <int>[];
    final tokens = _vocab.keys.toList()
      ..sort((a, b) => b.length.compareTo(a.length));

    int pos = 0;
    while (pos < normalized.length) {
      final match = _specialTokenRegex.matchAsPrefix(normalized, pos);
      if (match != null) {
        final specialTok = match.group(0)!;
        if (_vocab.containsKey(specialTok)) {
          tokenIds.add(_vocab[specialTok]!);
          pos += specialTok.length;
          while (pos < normalized.length && normalized[pos] == ' ') {
            pos++;
          }
          continue;
        }
      }

      bool matched = false;
      for (final tok in tokens) {
        if (tok.length > normalized.length - pos) continue;
        final substr = normalized.substring(pos, pos + tok.length);

        if (substr == tok) {
          tokenIds.add(_vocab[tok]!);
          pos += tok.length;
          matched = true;
          break;
        }
        if (tok.startsWith('▁')) {
          final plainTok = tok.substring(1);
          if ((pos == 0 || normalized[pos - 1] == ' ') &&
              normalized.substring(pos, pos + plainTok.length) == plainTok) {
            tokenIds.add(_vocab[tok]!);
            pos += plainTok.length;
            matched = true;
            break;
          }
        }
      }

      if (!matched) {
        tokenIds.add(unkTokenId);
        pos += 1;
      } else {
        while (pos < normalized.length && normalized[pos] == ' ') {
          pos++;
        }
      }
    }

    tokenIds.add(eosTokenId);
    return tokenIds;
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

  // ... (rest of the new methods: runModel, _extractLastStepLogits, _sampleToken, release)
  // Include the updated runModel method and helper methods from my previous response

  Future<String> runModel(
    String inputText, {
    String? initialLangToken,
    int maxNewTokens = 50,
  }) async {
    // 1️⃣ Prepare and tokenize input
    String textToTokenize = inputText.trim();
    if (initialLangToken != null && initialLangToken.isNotEmpty) {
      textToTokenize = '$initialLangToken $textToTokenize';
    }

    final inputIds = tokenize(textToTokenize);
    final seqLen = inputIds.length;
    final attentionMask = List<int>.filled(seqLen, 1);

    // 2️⃣ Run encoder
    final inputTensor = OrtValueTensor.createTensorWithDataList(
        Int64List.fromList(inputIds), [1, seqLen]);
    final attentionMaskTensor = OrtValueTensor.createTensorWithDataList(
        Int64List.fromList(attentionMask), [1, seqLen]);

    final encoderInputs = {
      'input_ids': inputTensor,
      'attention_mask': attentionMaskTensor,
    };

    final encoderOutputs = await _encoderSession.runAsync(OrtRunOptions(), encoderInputs);
    final encoderHiddenStates = encoderOutputs![0];

    // 3️⃣ Initialize decoder state
    final generatedIds = <int>[];
    var decoderInputIds = [padTokenId]; // Start with pad token
    
    // Store past key values for each layer
    List<OrtValue?> pastKeyValues = List.filled(12, null); // 6 layers * 2 (key + value)

    // 4️⃣ First decoder step (using decoder_model.onnx)
    var decInputTensor = OrtValueTensor.createTensorWithDataList(
        Int64List.fromList(decoderInputIds), [1, decoderInputIds.length]);

    var decoderInputs = {
      'input_ids': decInputTensor,
      'encoder_hidden_states': encoderHiddenStates!,
      'encoder_attention_mask': attentionMaskTensor,
    };

    var decoderOutputs = await _decoderSession.runAsync(OrtRunOptions(), decoderInputs);
    
    // Extract logits and past key values from first step
    final firstLogitsTensor = decoderOutputs![0];
    // Store past key values (outputs 1-12 are the past key values)
    for (int i = 0; i < 12; i++) {
      pastKeyValues[i] = decoderOutputs[i + 1];
    }

    // 5️⃣ Get first token
    final firstLogits = _extractLastStepLogits(firstLogitsTensor!);
    int nextToken = _sampleToken(firstLogits);
    
    if (nextToken != eosTokenId) {
      generatedIds.add(nextToken);
      decoderInputIds.add(nextToken);
    } else {
      return detokenize(generatedIds);
    }

    // 6️⃣ Subsequent steps using decoder_with_past_model.onnx
    for (int step = 1; step < maxNewTokens; step++) {
      // Prepare inputs for decoder_with_past
      final currentInputTensor = OrtValueTensor.createTensorWithDataList(
          Int64List.fromList([nextToken]), [1, 1]);

      final pastDecoderInputs = {
        'input_ids': currentInputTensor,
        'encoder_attention_mask': attentionMaskTensor,
        'encoder_hidden_states': encoderHiddenStates,
      };

      // Add past key values to inputs
      for (int i = 0; i < 12; i++) {
        pastDecoderInputs['past_key_values.$i.decoder.key'] = pastKeyValues[i]!;
        pastDecoderInputs['past_key_values.$i.decoder.value'] = pastKeyValues[i + 12]!;
        pastDecoderInputs['past_key_values.$i.encoder.key'] = pastKeyValues[i + 24]!;
        pastDecoderInputs['past_key_values.$i.encoder.value'] = pastKeyValues[i + 36]!;
      }

      final pastDecoderOutputs = await _decoderWithPastSession.runAsync(
          OrtRunOptions(), pastDecoderInputs);

      // Update past key values
      final newLogitsTensor = pastDecoderOutputs![0];
      for (int i = 0; i < 12; i++) {
        pastKeyValues[i] = pastDecoderOutputs[i + 1];
      }

      // Sample next token
      final logits = _extractLastStepLogits(newLogitsTensor!);
      nextToken = _sampleToken(logits);

      if (nextToken == eosTokenId) break;
      
      generatedIds.add(nextToken);
    }

    return detokenize(generatedIds).trim();
  }

  /// Helper method to extract logits from the last token position
  List<double> _extractLastStepLogits(OrtValue logitsTensor) {
    final raw = logitsTensor.value;
    
    if (raw is List && raw.isNotEmpty) {
      // Handle 3D tensor: [batch_size, sequence_length, vocab_size]
      if (raw[0] is List && (raw[0] as List).isNotEmpty) {
        final batch = raw[0] as List<dynamic>;
        final lastStep = batch.last as List<dynamic>;
        return lastStep.map((e) => (e as num).toDouble()).toList();
      }
      // Handle 2D tensor: [batch_size, vocab_size]  
      else if (raw[0] is num) {
        return raw.map((e) => (e as num).toDouble()).toList();
      }
    }
    
    // Fallback: return empty list
    return [];
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