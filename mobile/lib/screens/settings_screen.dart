import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

import '../services/config_service.dart';

/// Full settings screen accessible from welcome screen.
/// Includes server URL (required for mobile) and all agent settings.
class SettingsScreen extends StatefulWidget {
  final ConfigService configService;
  final VoidCallback onSave;

  const SettingsScreen({
    super.key,
    required this.configService,
    required this.onSave,
  });

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _formKey = GlobalKey<FormState>();
  final _serverUrlController = TextEditingController();

  bool _loading = false;
  bool _saving = false;
  String _saveStatus = '';
  String? _error;
  bool _serverConnected = false;

  // Agent settings fields
  String _agentName = 'Cal';
  List<String> _wakeGreetings = ["Hey, what's up?", "What's up?", 'How can I help?'];

  // Prompt settings
  String _promptMode = 'default';
  final _promptController = TextEditingController();

  // Provider settings
  String _llmProvider = 'ollama';
  String _ollamaHost = 'http://localhost:11434';
  String _ollamaModel = '';
  String _groqApiKey = '';
  String _groqModel = '';
  String _ttsProvider = 'kokoro';
  String _ttsVoiceKokoro = 'am_puck';
  String _ttsVoicePiper = 'speaches-ai/piper-en_US-ryan-high';

  // Integration settings
  bool _hassEnabled = false;
  String _hassHost = '';
  String _hassToken = '';
  String _hassAgentId = 'conversation.home_assistant';
  List<Map<String, String>> _hassAgents = [];
  bool _n8nEnabled = false;
  String _n8nUrl = '';
  String _n8nToken = '';
  bool _fridayEnabled = false;
  String _fridayHost = '';
  String _fridayToken = '';

  // LLM settings
  double _temperature = 0.7;
  int _numCtx = 8192;
  int _maxTurns = 20;
  int _toolCacheSize = 3;

  // Turn detection
  bool _allowInterruptions = true;
  double _minEndpointingDelay = 0.5;

  // Noise suppression
  bool _noiseSuppressionEnabled = false;
  double _noiseSuppressionAttenDb = 100.0;

  // Energy gate
  bool _energyGateEnabled = true;
  double _energyGateThresholdDb = -35.0;

  // Wake word
  bool _wakeWordEnabled = false;
  String _wakeWordModel = 'models/hey_cal.onnx';
  double _wakeWordThreshold = 0.5;
  double _wakeWordTimeout = 3.0;

  // TV/Media rejection
  bool _tvRejectionEnabled = true;
  double _tvRejectionMinLiveness = 0.2;
  bool _mediaNoiseMusicDetection = true;
  bool _mediaNoiseStereoDetection = true;
  bool _mediaNoisePlaybackVoiceDetection = true;

  // Speaker recognition
  bool _speakerRecognitionEnabled = false;
  double _speakerRecognitionThreshold = 0.75;
  bool _speakerRecognitionRequired = false;
  List<Map<String, dynamic>> _speakers = [];
  bool _speakersLoading = false;

  // Available options
  List<String> _voices = [];
  List<String> _ollamaModels = [];
  List<String> _groqModels = [];
  List<String> _wakeWordModels = [];

  // Test states
  bool _testingOllama = false;
  bool _ollamaConnected = false;
  String? _ollamaError;
  bool _testingGroq = false;
  bool _groqConnected = false;
  String? _groqError;
  bool _testingHass = false;
  bool _hassConnected = false;
  String? _hassError;
  String? _hassInfo;
  bool _testingN8n = false;
  bool _n8nConnected = false;
  String? _n8nError;
  bool _testingFriday = false;
  bool _fridayConnected = false;
  String? _fridayError;

  // Text controllers
  final _wakeGreetingsController = TextEditingController();

  String get _webhookUrl {
    final serverUrl = _serverUrlController.text.trim();
    if (serverUrl.isEmpty) return '';
    final uri = Uri.tryParse(serverUrl);
    if (uri == null) return '';
    return 'http://${uri.host}:8889';
  }

  @override
  void initState() {
    super.initState();
    _serverUrlController.text = widget.configService.serverUrl;
    // Try to load settings if server is configured
    if (widget.configService.isConfigured) {
      unawaited(_loadSettings());
    }
  }

  @override
  void dispose() {
    _serverUrlController.dispose();
    _wakeGreetingsController.dispose();
    _promptController.dispose();
    super.dispose();
  }

  Future<void> _testConnection() async {
    final webhookUrl = _webhookUrl;
    if (webhookUrl.isEmpty) {
      setState(() {
        _error = 'Enter a valid server URL first';
        _serverConnected = false;
      });
      return;
    }

    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final response = await http.get(Uri.parse('$webhookUrl/settings')).timeout(
            const Duration(seconds: 5),
          );
      if (response.statusCode == 200) {
        setState(() {
          _serverConnected = true;
        });
        await _loadSettings();
      } else {
        setState(() {
          _serverConnected = false;
          _error = 'Server returned ${response.statusCode}';
        });
      }
    } catch (e) {
      setState(() {
        _serverConnected = false;
        _error = 'Could not connect to server';
      });
    } finally {
      setState(() {
        _loading = false;
      });
    }
  }

  Future<void> _loadSettings() async {
    final webhookUrl = _webhookUrl;
    if (webhookUrl.isEmpty) return;

    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      // First load settings to get the correct tts_provider
      final settingsRes = await http.get(Uri.parse('$webhookUrl/settings'));

      String ttsProvider = _ttsProvider;
      if (settingsRes.statusCode == 200) {
        final data = jsonDecode(settingsRes.body);
        final settings = data['settings'] ?? {};
        ttsProvider = settings['tts_provider'] ?? _ttsProvider;
      }

      // Now fetch voices with the correct provider, plus wake word models
      final results = await Future.wait([
        http.get(Uri.parse('$webhookUrl/voices?provider=$ttsProvider')),
        http.get(Uri.parse('$webhookUrl/wake-word/models')),
      ]);

      final voicesRes = results[0];
      final wakeWordModelsRes = results[1];

      if (settingsRes.statusCode == 200) {
        final data = jsonDecode(settingsRes.body);
        final settings = data['settings'] ?? {};

        // Load prompt content
        final promptContent = data['prompt_content'] ?? '';

        setState(() {
          _serverConnected = true;
          // Agent
          _agentName = settings['agent_name'] ?? _agentName;
          _wakeGreetings = List<String>.from(settings['wake_greetings'] ?? _wakeGreetings);
          _wakeGreetingsController.text = _wakeGreetings.join('\n');

          // Prompt
          _promptMode = settings['prompt'] ?? 'default';
          _promptController.text = promptContent;

          // Providers
          _llmProvider = settings['llm_provider'] ?? _llmProvider;
          _ollamaHost = settings['ollama_host'] ?? _ollamaHost;
          _ollamaModel = settings['ollama_model'] ?? _ollamaModel;
          _groqModel = settings['groq_model'] ?? _groqModel;
          _ttsProvider = settings['tts_provider'] ?? _ttsProvider;
          _ttsVoiceKokoro = settings['tts_voice_kokoro'] ?? _ttsVoiceKokoro;
          _ttsVoicePiper = settings['tts_voice_piper'] ?? _ttsVoicePiper;

          // Integrations
          _hassEnabled = settings['hass_enabled'] ?? _hassEnabled;
          _hassHost = settings['hass_host'] ?? _hassHost;
          _hassToken = settings['hass_token'] ?? _hassToken;
          _hassAgentId = settings['hass_agent_id'] ?? _hassAgentId;
          _n8nEnabled = settings['n8n_enabled'] ?? _n8nEnabled;
          _n8nUrl = settings['n8n_url'] ?? _n8nUrl;
          _n8nToken = settings['n8n_token'] ?? _n8nToken;
          _fridayEnabled = settings['friday_enabled'] ?? _fridayEnabled;
          _fridayHost = settings['friday_host'] ?? _fridayHost;
          _fridayToken = settings['friday_token'] ?? _fridayToken;

          // LLM settings
          _temperature = (settings['temperature'] ?? _temperature).toDouble();
          _numCtx = settings['num_ctx'] ?? _numCtx;
          _maxTurns = settings['max_turns'] ?? _maxTurns;
          _toolCacheSize = settings['tool_cache_size'] ?? _toolCacheSize;

          // Turn detection
          _allowInterruptions = settings['allow_interruptions'] ?? _allowInterruptions;
          _minEndpointingDelay = (settings['min_endpointing_delay'] ?? _minEndpointingDelay).toDouble();

          // Noise suppression
          _noiseSuppressionEnabled = settings['noise_suppression_enabled'] ?? _noiseSuppressionEnabled;
          _noiseSuppressionAttenDb = (settings['noise_suppression_atten_db'] ?? _noiseSuppressionAttenDb).toDouble();

          // Energy gate
          _energyGateEnabled = settings['energy_gate_enabled'] ?? _energyGateEnabled;
          _energyGateThresholdDb = (settings['energy_gate_threshold_db'] ?? _energyGateThresholdDb).toDouble();

          // Wake word
          _wakeWordEnabled = settings['wake_word_enabled'] ?? _wakeWordEnabled;
          _wakeWordModel = settings['wake_word_model'] ?? _wakeWordModel;
          _wakeWordThreshold = (settings['wake_word_threshold'] ?? _wakeWordThreshold).toDouble();
          _wakeWordTimeout = (settings['wake_word_timeout'] ?? _wakeWordTimeout).toDouble();

          // TV/Media rejection
          _tvRejectionEnabled = settings['tv_rejection_enabled'] ?? _tvRejectionEnabled;
          _tvRejectionMinLiveness = (settings['tv_rejection_min_liveness'] ?? _tvRejectionMinLiveness).toDouble();
          _mediaNoiseMusicDetection = settings['media_noise_music_detection'] ?? _mediaNoiseMusicDetection;
          _mediaNoiseStereoDetection = settings['media_noise_stereo_detection'] ?? _mediaNoiseStereoDetection;
          _mediaNoisePlaybackVoiceDetection = settings['media_noise_playback_voice_detection'] ?? _mediaNoisePlaybackVoiceDetection;

          // Speaker recognition
          _speakerRecognitionEnabled = settings['speaker_recognition_enabled'] ?? _speakerRecognitionEnabled;
          _speakerRecognitionThreshold = (settings['speaker_recognition_threshold'] ?? _speakerRecognitionThreshold).toDouble();
          _speakerRecognitionRequired = settings['speaker_recognition_required'] ?? _speakerRecognitionRequired;
        });
      }

      if (voicesRes.statusCode == 200) {
        final data = jsonDecode(voicesRes.body);
        setState(() {
          _voices = List<String>.from(data['voices'] ?? []);
        });
      }

      if (wakeWordModelsRes.statusCode == 200) {
        final data = jsonDecode(wakeWordModelsRes.body);
        setState(() {
          _wakeWordModels = List<String>.from(data['models'] ?? []);
        });
      }
    } catch (e) {
      setState(() {
        _error = 'Failed to load settings: $e';
      });
    } finally {
      setState(() {
        _loading = false;
      });
    }
  }

  Future<void> _fetchVoices(String provider) async {
    final webhookUrl = _webhookUrl;
    if (webhookUrl.isEmpty) return;

    try {
      final res = await http.get(Uri.parse('$webhookUrl/voices?provider=$provider'));
      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);
        setState(() {
          _voices = List<String>.from(data['voices'] ?? []);
        });
      }
    } catch (e) {
      // Silently fail - voices list will remain as-is
    }
  }

  Future<void> _loadSpeakers() async {
    final webhookUrl = _webhookUrl;
    if (webhookUrl.isEmpty || !_speakerRecognitionEnabled) return;

    setState(() {
      _speakersLoading = true;
    });

    try {
      final res = await http.get(Uri.parse('$webhookUrl/speakers'));
      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);
        setState(() {
          _speakers = List<Map<String, dynamic>>.from(data['speakers'] ?? []);
        });
      }
    } catch (e) {
      // Silently fail
    } finally {
      setState(() {
        _speakersLoading = false;
      });
    }
  }

  void _handleTtsProviderChange(String provider) {
    if (provider == _ttsProvider) return;
    setState(() {
      _ttsProvider = provider;
    });
    unawaited(_fetchVoices(provider));
  }

  Future<void> _testOllama() async {
    if (_ollamaHost.isEmpty) return;
    setState(() {
      _testingOllama = true;
      _ollamaError = null;
    });

    try {
      final res = await http.post(
        Uri.parse('$_webhookUrl/setup/test-ollama'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'host': _ollamaHost}),
      );
      final result = jsonDecode(res.body);

      if (result['success'] == true) {
        setState(() {
          _ollamaConnected = true;
          _ollamaModels = List<String>.from(result['models'] ?? []);
          if (_ollamaModel.isEmpty && _ollamaModels.isNotEmpty) {
            _ollamaModel = _ollamaModels.first;
          }
        });
      } else {
        setState(() {
          _ollamaConnected = false;
          _ollamaError = result['error'] ?? 'Connection failed';
        });
      }
    } catch (e) {
      setState(() {
        _ollamaConnected = false;
        _ollamaError = 'Failed to connect';
      });
    } finally {
      setState(() {
        _testingOllama = false;
      });
    }
  }

  Future<void> _testGroq() async {
    if (_groqApiKey.isEmpty) return;
    setState(() {
      _testingGroq = true;
      _groqError = null;
    });

    try {
      final res = await http.post(
        Uri.parse('$_webhookUrl/setup/test-groq'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'api_key': _groqApiKey}),
      );
      final result = jsonDecode(res.body);

      if (result['success'] == true) {
        setState(() {
          _groqConnected = true;
          _groqModels = List<String>.from(result['models'] ?? []);
          if (_groqModel.isEmpty && _groqModels.isNotEmpty) {
            final preferred = 'llama-3.3-70b-versatile';
            _groqModel = _groqModels.contains(preferred) ? preferred : _groqModels.first;
          }
        });
      } else {
        setState(() {
          _groqConnected = false;
          _groqError = result['error'] ?? 'Invalid API key';
        });
      }
    } catch (e) {
      setState(() {
        _groqConnected = false;
        _groqError = 'Failed to validate';
      });
    } finally {
      setState(() {
        _testingGroq = false;
      });
    }
  }

  Future<void> _fetchHassAgents() async {
    if (_hassHost.isEmpty || _hassToken.isEmpty) return;
    try {
      final res = await http.post(
        Uri.parse('$_webhookUrl/setup/hass-agents'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'host': _hassHost, 'token': _hassToken}),
      );
      final result = jsonDecode(res.body);

      if (result['success'] == true && result['agents'] != null) {
        setState(() {
          _hassAgents = (result['agents'] as List)
              .map((a) => {'id': a['id'] as String, 'name': a['name'] as String})
              .toList();
          // Set default agent if not already set
          if (_hassAgentId.isEmpty && _hassAgents.isNotEmpty) {
            _hassAgentId = _hassAgents.first['id']!;
          }
        });
      }
    } catch (e) {
      // Silently fail
    }
  }

  Future<void> _testHass() async {
    if (_hassHost.isEmpty || _hassToken.isEmpty) return;
    setState(() {
      _testingHass = true;
      _hassError = null;
      _hassInfo = null;
    });

    try {
      final res = await http.post(
        Uri.parse('$_webhookUrl/setup/test-hass'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'host': _hassHost, 'token': _hassToken}),
      );
      final result = jsonDecode(res.body);

      if (result['success'] == true) {
        setState(() {
          _hassConnected = true;
          _hassInfo = 'Connected - ${result['device_count']} entities';
        });
        // Fetch available agents on successful connection
        unawaited(_fetchHassAgents());
      } else {
        setState(() {
          _hassConnected = false;
          _hassError = result['error'] ?? 'Connection failed';
          _hassAgents = [];
        });
      }
    } catch (e) {
      setState(() {
        _hassConnected = false;
        _hassError = 'Failed to connect';
        _hassAgents = [];
      });
    } finally {
      setState(() {
        _testingHass = false;
      });
    }
  }

  String _getN8nMcpUrl(String host) {
    if (host.isEmpty) return '';
    final baseUrl = host.replaceAll(RegExp(r'/$'), '');
    if (baseUrl.contains('/mcp-server')) return baseUrl;
    return '$baseUrl/mcp-server/http';
  }

  Future<void> _testN8n() async {
    if (_n8nUrl.isEmpty || _n8nToken.isEmpty) return;
    setState(() {
      _testingN8n = true;
      _n8nError = null;
    });

    try {
      final mcpUrl = _getN8nMcpUrl(_n8nUrl);
      final res = await http.post(
        Uri.parse('$_webhookUrl/setup/test-n8n'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'url': mcpUrl, 'token': _n8nToken}),
      );
      final result = jsonDecode(res.body);

      if (result['success'] == true) {
        setState(() {
          _n8nConnected = true;
        });
      } else {
        setState(() {
          _n8nConnected = false;
          _n8nError = result['error'] ?? 'Connection failed';
        });
      }
    } catch (e) {
      setState(() {
        _n8nConnected = false;
        _n8nError = 'Failed to connect';
      });
    } finally {
      setState(() {
        _testingN8n = false;
      });
    }
  }

  Future<void> _testFriday() async {
    if (_fridayHost.isEmpty || _fridayToken.isEmpty) return;
    setState(() {
      _testingFriday = true;
      _fridayError = null;
    });

    try {
      final res = await http.post(
        Uri.parse('$_webhookUrl/setup/test-friday'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'host': _fridayHost, 'token': _fridayToken}),
      );
      final result = jsonDecode(res.body);

      if (result['success'] == true) {
        setState(() {
          _fridayConnected = true;
        });
      } else {
        setState(() {
          _fridayConnected = false;
          _fridayError = result['error'] ?? 'Connection failed';
        });
      }
    } catch (e) {
      setState(() {
        _fridayConnected = false;
        _fridayError = 'Failed to connect';
      });
    } finally {
      setState(() {
        _testingFriday = false;
      });
    }
  }

  String get _currentVoice {
    return _ttsProvider == 'piper' ? _ttsVoicePiper : _ttsVoiceKokoro;
  }

  void _setCurrentVoice(String voice) {
    setState(() {
      if (_ttsProvider == 'piper') {
        _ttsVoicePiper = voice;
      } else {
        _ttsVoiceKokoro = voice;
      }
    });
  }

  Future<void> _save() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() {
      _saving = true;
      _saveStatus = 'Saving...';
      _error = null;
    });

    try {
      // Save server URL first
      await widget.configService.setServerUrl(_serverUrlController.text.trim());

      // If connected, save agent settings
      if (_serverConnected) {
        final greetings =
            _wakeGreetingsController.text.split('\n').where((g) => g.trim().isNotEmpty).toList();

        final settings = {
          // Agent
          'agent_name': _agentName,
          'wake_greetings': greetings,
          'prompt': _promptMode,

          // Providers
          'llm_provider': _llmProvider,
          'ollama_host': _ollamaHost,
          'ollama_model': _ollamaModel,
          'groq_model': _groqModel,
          'tts_provider': _ttsProvider,
          'tts_voice_kokoro': _ttsVoiceKokoro,
          'tts_voice_piper': _ttsVoicePiper,

          // Integrations
          'hass_enabled': _hassEnabled,
          'hass_host': _hassHost,
          'hass_token': _hassToken,
          'hass_agent_id': _hassAgentId,
          'n8n_enabled': _n8nEnabled,
          'n8n_url': _n8nEnabled ? _getN8nMcpUrl(_n8nUrl) : _n8nUrl,
          'n8n_token': _n8nToken,
          'friday_enabled': _fridayEnabled,
          'friday_host': _fridayHost,
          'friday_token': _fridayToken,

          // LLM settings
          'temperature': _temperature,
          'num_ctx': _numCtx,
          'max_turns': _maxTurns,
          'tool_cache_size': _toolCacheSize,

          // Turn detection
          'allow_interruptions': _allowInterruptions,
          'min_endpointing_delay': _minEndpointingDelay,

          // Noise suppression
          'noise_suppression_enabled': _noiseSuppressionEnabled,
          'noise_suppression_atten_db': _noiseSuppressionAttenDb,

          // Energy gate
          'energy_gate_enabled': _energyGateEnabled,
          'energy_gate_threshold_db': _energyGateThresholdDb,

          // Wake word
          'wake_word_enabled': _wakeWordEnabled,
          'wake_word_model': _wakeWordModel,
          'wake_word_threshold': _wakeWordThreshold,
          'wake_word_timeout': _wakeWordTimeout,

          // TV/Media rejection
          'tv_rejection_enabled': _tvRejectionEnabled,
          'tv_rejection_min_liveness': _tvRejectionMinLiveness,
          'media_noise_music_detection': _mediaNoiseMusicDetection,
          'media_noise_stereo_detection': _mediaNoiseStereoDetection,
          'media_noise_playback_voice_detection': _mediaNoisePlaybackVoiceDetection,

          // Speaker recognition
          'speaker_recognition_enabled': _speakerRecognitionEnabled,
          'speaker_recognition_threshold': _speakerRecognitionThreshold,
          'speaker_recognition_required': _speakerRecognitionRequired,
        };

        final res = await http.post(
          Uri.parse('$_webhookUrl/settings'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({'settings': settings}),
        );

        if (res.statusCode != 200) {
          setState(() {
            _error = 'Failed to save agent settings: ${res.statusCode}';
          });
          return;
        }

        // Save Groq API key if using Groq and key was entered
        if (_llmProvider == 'groq' && _groqApiKey.isNotEmpty) {
          await http.post(
            Uri.parse('$_webhookUrl/setup/complete'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({
              'llm_provider': 'groq',
              'groq_api_key': _groqApiKey,
              'groq_model': _groqModel,
            }),
          );
        }

        // Save custom prompt if using custom mode
        if (_promptMode == 'custom' && _promptController.text.isNotEmpty) {
          await http.post(
            Uri.parse('$_webhookUrl/prompt'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({'content': _promptController.text}),
          );
        }

        // Download Piper model if using Piper
        if (_ttsProvider == 'piper' && _ttsVoicePiper.isNotEmpty) {
          setState(() {
            _saveStatus = 'Downloading voice model...';
          });
          try {
            await http.post(
              Uri.parse('$_webhookUrl/download-piper-model'),
              headers: {'Content-Type': 'application/json'},
              body: jsonEncode({'model_id': _ttsVoicePiper}),
            );
          } catch (e) {
            // Non-critical - model can be downloaded later
          }
        }
      }

      widget.onSave();
    } catch (e) {
      setState(() {
        _error = 'Failed to save: $e';
      });
    } finally {
      if (mounted) {
        setState(() {
          _saving = false;
          _saveStatus = '';
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final isFirstSetup = !widget.configService.isConfigured;

    return Scaffold(
      backgroundColor: const Color(0xFF1A1A1A),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1A1A1A),
        leading: isFirstSetup
            ? null
            : IconButton(
                icon: const Icon(Icons.close, color: Colors.white),
                onPressed: () => Navigator.of(context).pop(),
              ),
        title: Text(
          isFirstSetup ? 'CAAL Setup' : 'Settings',
          style: const TextStyle(color: Colors.white),
        ),
        actions: [
          if (_saving)
            Padding(
              padding: const EdgeInsets.all(16),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      valueColor: AlwaysStoppedAnimation(Color(0xFF45997C)),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    _saveStatus.isNotEmpty ? _saveStatus : 'Saving...',
                    style: const TextStyle(
                      color: Color(0xFF45997C),
                      fontSize: 12,
                    ),
                  ),
                ],
              ),
            )
          else
            TextButton(
              onPressed: _save,
              child: const Text(
                'Save',
                style: TextStyle(
                  color: Color(0xFF45997C),
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
        ],
      ),
      body: SafeArea(
        child: Form(
          key: _formKey,
          child: ListView(
            padding: const EdgeInsets.all(16),
            children: [
              if (_error != null) ...[
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.red.withValues(alpha: 0.1),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    _error!,
                    style: const TextStyle(color: Colors.red),
                  ),
                ),
                const SizedBox(height: 16),
              ],

              // Connection Section
              _buildSectionHeader('Connection', Icons.link),
              _buildCard([
                _buildLabel('Server URL'),
                Row(
                  children: [
                    Expanded(
                      child: TextFormField(
                        controller: _serverUrlController,
                        keyboardType: TextInputType.url,
                        autocorrect: false,
                        style: const TextStyle(color: Colors.white),
                        decoration: _inputDecoration(hint: 'http://192.168.1.100:3000'),
                        validator: (value) {
                          if (value == null || value.trim().isEmpty) {
                            return 'Server URL is required';
                          }
                          final uri = Uri.tryParse(value.trim());
                          if (uri == null || !uri.hasScheme || !uri.hasAuthority) {
                            return 'Enter a valid URL';
                          }
                          return null;
                        },
                      ),
                    ),
                    const SizedBox(width: 8),
                    TextButton(
                      onPressed: _loading ? null : _testConnection,
                      style: TextButton.styleFrom(
                        backgroundColor:
                            _serverConnected ? const Color(0xFF45997C) : const Color(0xFF2A2A2A),
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
                      ),
                      child: _loading
                          ? const SizedBox(
                              width: 16,
                              height: 16,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                valueColor: AlwaysStoppedAnimation(Colors.white),
                              ),
                            )
                          : Text(_serverConnected ? '✓' : 'Test'),
                    ),
                  ],
                ),
                const SizedBox(height: 4),
                Text(
                  _serverConnected ? 'Connected to CAAL server' : 'Your CAAL server address',
                  style: TextStyle(
                    fontSize: 12,
                    color: _serverConnected ? const Color(0xFF45997C) : Colors.white54,
                  ),
                ),
              ]),

              // Settings (only show if connected)
              if (_serverConnected) ...[
                // Agent Section
                const SizedBox(height: 24),
                _buildSectionHeader('Agent', Icons.smart_toy_outlined),
                _buildCard([
                  _buildTextField(
                    label: 'Agent Name',
                    value: _agentName,
                    onChanged: (v) => setState(() => _agentName = v),
                  ),
                  _buildLabel('Wake Greetings'),
                  TextFormField(
                    controller: _wakeGreetingsController,
                    maxLines: 3,
                    style: const TextStyle(color: Colors.white),
                    decoration: _inputDecoration(hint: 'One greeting per line'),
                  ),
                ]),

                // Prompt Section
                const SizedBox(height: 24),
                _buildSectionHeader('System Prompt', Icons.description_outlined),
                _buildCard([
                  _buildLabel('Prompt Mode'),
                  Row(
                    children: [
                      Expanded(
                        child: GestureDetector(
                          onTap: () => setState(() => _promptMode = 'default'),
                          child: Container(
                            padding: const EdgeInsets.symmetric(vertical: 12),
                            decoration: BoxDecoration(
                              color: _promptMode == 'default'
                                  ? const Color(0xFF45997C).withValues(alpha: 0.2)
                                  : const Color(0xFF2A2A2A),
                              border: Border.all(
                                color: _promptMode == 'default'
                                    ? const Color(0xFF45997C)
                                    : Colors.white.withValues(alpha: 0.1),
                              ),
                              borderRadius: const BorderRadius.only(
                                topLeft: Radius.circular(8),
                                bottomLeft: Radius.circular(8),
                              ),
                            ),
                            child: Center(
                              child: Text(
                                'Default',
                                style: TextStyle(
                                  color: _promptMode == 'default' ? Colors.white : Colors.white70,
                                  fontWeight: FontWeight.w500,
                                ),
                              ),
                            ),
                          ),
                        ),
                      ),
                      Expanded(
                        child: GestureDetector(
                          onTap: () => setState(() => _promptMode = 'custom'),
                          child: Container(
                            padding: const EdgeInsets.symmetric(vertical: 12),
                            decoration: BoxDecoration(
                              color: _promptMode == 'custom'
                                  ? const Color(0xFF45997C).withValues(alpha: 0.2)
                                  : const Color(0xFF2A2A2A),
                              border: Border.all(
                                color: _promptMode == 'custom'
                                    ? const Color(0xFF45997C)
                                    : Colors.white.withValues(alpha: 0.1),
                              ),
                              borderRadius: const BorderRadius.only(
                                topRight: Radius.circular(8),
                                bottomRight: Radius.circular(8),
                              ),
                            ),
                            child: Center(
                              child: Text(
                                'Custom',
                                style: TextStyle(
                                  color: _promptMode == 'custom' ? Colors.white : Colors.white70,
                                  fontWeight: FontWeight.w500,
                                ),
                              ),
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  _buildLabel('Prompt Content'),
                  TextFormField(
                    controller: _promptController,
                    maxLines: 8,
                    enabled: _promptMode == 'custom',
                    style: TextStyle(
                      color: _promptMode == 'custom' ? Colors.white : Colors.white54,
                      fontFamily: 'monospace',
                      fontSize: 12,
                    ),
                    decoration: InputDecoration(
                      hintText: _promptMode == 'default'
                          ? 'Using default system prompt'
                          : 'Enter custom system prompt...',
                      hintStyle: const TextStyle(color: Colors.white38),
                      filled: true,
                      fillColor: _promptMode == 'custom'
                          ? const Color(0xFF2A2A2A)
                          : const Color(0xFF1E1E1E),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(8),
                        borderSide: BorderSide.none,
                      ),
                      contentPadding: const EdgeInsets.all(12),
                    ),
                  ),
                  if (_promptMode == 'default')
                    const Padding(
                      padding: EdgeInsets.only(top: 8),
                      child: Text(
                        'Select "Custom" to edit the system prompt',
                        style: TextStyle(color: Colors.white38, fontSize: 11),
                      ),
                    ),
                ]),

                // Providers Section
                const SizedBox(height: 24),
                _buildSectionHeader('Providers', Icons.cloud_outlined),
                _buildCard([
                  // LLM Provider
                  _buildLabel('LLM Provider'),
                  _buildProviderToggle(
                    options: ['ollama', 'groq'],
                    labels: ['Ollama', 'Groq'],
                    subtitles: ['Local, private', 'Fast cloud'],
                    selected: _llmProvider,
                    onChanged: (v) => setState(() => _llmProvider = v),
                  ),
                  const SizedBox(height: 16),

                  // Ollama config
                  if (_llmProvider == 'ollama') ...[
                    _buildLabel('Ollama Host'),
                    Row(
                      children: [
                        Expanded(
                          child: TextFormField(
                            initialValue: _ollamaHost,
                            style: const TextStyle(color: Colors.white),
                            decoration: _inputDecoration(hint: 'http://localhost:11434'),
                            onChanged: (v) => setState(() => _ollamaHost = v),
                          ),
                        ),
                        const SizedBox(width: 8),
                        _buildTestButton(
                          testing: _testingOllama,
                          connected: _ollamaConnected,
                          onPressed: _testOllama,
                        ),
                      ],
                    ),
                    if (_ollamaError != null)
                      Padding(
                        padding: const EdgeInsets.only(top: 4),
                        child: Text(_ollamaError!, style: const TextStyle(color: Colors.red, fontSize: 12)),
                      ),
                    if (_ollamaConnected)
                      Padding(
                        padding: const EdgeInsets.only(top: 4),
                        child: Text('${_ollamaModels.length} models available',
                            style: const TextStyle(color: Color(0xFF45997C), fontSize: 12)),
                      ),
                    const SizedBox(height: 12),
                    if (_ollamaModels.isNotEmpty || _ollamaModel.isNotEmpty)
                      _buildDropdown(
                        label: 'Model',
                        value: _ollamaModel.isNotEmpty ? _ollamaModel : (_ollamaModels.isNotEmpty ? _ollamaModels.first : ''),
                        options: _ollamaModels.isNotEmpty ? _ollamaModels : [_ollamaModel],
                        onChanged: (v) => setState(() => _ollamaModel = v ?? _ollamaModel),
                      ),
                  ],

                  // Groq config
                  if (_llmProvider == 'groq') ...[
                    _buildLabel('API Key'),
                    Row(
                      children: [
                        Expanded(
                          child: TextFormField(
                            initialValue: _groqApiKey,
                            obscureText: true,
                            style: const TextStyle(color: Colors.white),
                            decoration: _inputDecoration(hint: _groqModel.isNotEmpty ? '••••••••••••••••' : 'gsk_...'),
                            onChanged: (v) => setState(() => _groqApiKey = v),
                          ),
                        ),
                        const SizedBox(width: 8),
                        _buildTestButton(
                          testing: _testingGroq,
                          connected: _groqConnected,
                          onPressed: _testGroq,
                        ),
                      ],
                    ),
                    if (_groqError != null)
                      Padding(
                        padding: const EdgeInsets.only(top: 4),
                        child: Text(_groqError!, style: const TextStyle(color: Colors.red, fontSize: 12)),
                      ),
                    if (_groqConnected)
                      Padding(
                        padding: const EdgeInsets.only(top: 4),
                        child: Text('${_groqModels.length} models available',
                            style: const TextStyle(color: Color(0xFF45997C), fontSize: 12)),
                      ),
                    if (!_groqConnected && _groqApiKey.isEmpty && _groqModel.isNotEmpty)
                      const Padding(
                        padding: EdgeInsets.only(top: 4),
                        child: Text('API key configured (enter new key to change)',
                            style: TextStyle(color: Color(0xFF45997C), fontSize: 12)),
                      ),
                    const SizedBox(height: 12),
                    if (_groqModels.isNotEmpty || _groqModel.isNotEmpty)
                      _buildDropdown(
                        label: 'Model',
                        value: _groqModel.isNotEmpty ? _groqModel : (_groqModels.isNotEmpty ? _groqModels.first : ''),
                        options: _groqModels.isNotEmpty ? _groqModels : [_groqModel],
                        onChanged: (v) => setState(() => _groqModel = v ?? _groqModel),
                      ),
                  ],

                  const Divider(color: Colors.white24, height: 32),

                  // TTS Provider
                  _buildLabel('TTS Provider'),
                  _buildProviderToggle(
                    options: ['kokoro', 'piper'],
                    labels: ['Kokoro', 'Piper'],
                    subtitles: ['GPU neural TTS', 'CPU lightweight'],
                    selected: _ttsProvider,
                    onChanged: _handleTtsProviderChange,
                  ),
                  const SizedBox(height: 16),
                  _buildDropdown(
                    label: 'Voice',
                    value: _currentVoice,
                    options: _voices.isNotEmpty ? _voices : [_currentVoice],
                    onChanged: (v) => _setCurrentVoice(v ?? _currentVoice),
                  ),
                ]),

                // Integrations Section
                const SizedBox(height: 24),
                _buildSectionHeader('Integrations', Icons.extension_outlined),

                // Home Assistant
                _buildCard([
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text('Home Assistant',
                          style: TextStyle(color: Colors.white, fontWeight: FontWeight.w600)),
                      Switch(
                        value: _hassEnabled,
                        onChanged: (v) => setState(() => _hassEnabled = v),
                        activeTrackColor: const Color(0xFF45997C),
                      ),
                    ],
                  ),
                  if (_hassEnabled) ...[
                    const SizedBox(height: 12),
                    _buildLabel('Host URL'),
                    TextFormField(
                      initialValue: _hassHost,
                      style: const TextStyle(color: Colors.white),
                      decoration: _inputDecoration(hint: 'http://homeassistant.local:8123'),
                      onChanged: (v) => setState(() => _hassHost = v),
                    ),
                    const SizedBox(height: 12),
                    _buildLabel('Access Token'),
                    Row(
                      children: [
                        Expanded(
                          child: TextFormField(
                            initialValue: _hassToken,
                            obscureText: true,
                            style: const TextStyle(color: Colors.white),
                            decoration: _inputDecoration(hint: 'eyJ0eX...'),
                            onChanged: (v) => setState(() => _hassToken = v),
                          ),
                        ),
                        const SizedBox(width: 8),
                        _buildTestButton(
                          testing: _testingHass,
                          connected: _hassConnected,
                          onPressed: _testHass,
                        ),
                      ],
                    ),
                    if (_hassError != null)
                      Padding(
                        padding: const EdgeInsets.only(top: 4),
                        child: Text(_hassError!, style: const TextStyle(color: Colors.red, fontSize: 12)),
                      ),
                    if (_hassInfo != null)
                      Padding(
                        padding: const EdgeInsets.only(top: 4),
                        child: Text(_hassInfo!, style: const TextStyle(color: Color(0xFF45997C), fontSize: 12)),
                      ),
                    if (_hassAgents.isNotEmpty) ...[
                      const SizedBox(height: 12),
                      _buildLabel('AI Assistant'),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 12),
                        decoration: BoxDecoration(
                          color: const Color(0xFF1E1E1E),
                          borderRadius: BorderRadius.circular(8),
                          border: Border.all(color: const Color(0xFF333333)),
                        ),
                        child: DropdownButton<String>(
                          value: _hassAgentId.isNotEmpty ? _hassAgentId : null,
                          isExpanded: true,
                          dropdownColor: const Color(0xFF1E1E1E),
                          style: const TextStyle(color: Colors.white),
                          underline: const SizedBox(),
                          hint: const Text('Select assistant', style: TextStyle(color: Colors.grey)),
                          items: _hassAgents.map((agent) {
                            return DropdownMenuItem<String>(
                              value: agent['id'],
                              child: Text(agent['name']!),
                            );
                          }).toList(),
                          onChanged: (v) => setState(() => _hassAgentId = v ?? ''),
                        ),
                      ),
                      const Padding(
                        padding: EdgeInsets.only(top: 4),
                        child: Text(
                          'Choose which AI assistant handles smart home requests',
                          style: TextStyle(color: Colors.grey, fontSize: 11),
                        ),
                      ),
                    ],
                  ],
                ]),

                const SizedBox(height: 12),

                // n8n
                _buildCard([
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text('n8n', style: TextStyle(color: Colors.white, fontWeight: FontWeight.w600)),
                      Switch(
                        value: _n8nEnabled,
                        onChanged: (v) => setState(() => _n8nEnabled = v),
                        activeTrackColor: const Color(0xFF45997C),
                      ),
                    ],
                  ),
                  if (_n8nEnabled) ...[
                    const SizedBox(height: 12),
                    _buildLabel('Host URL'),
                    TextFormField(
                      initialValue: _n8nUrl,
                      style: const TextStyle(color: Colors.white),
                      decoration: _inputDecoration(hint: 'http://n8n:5678'),
                      onChanged: (v) => setState(() => _n8nUrl = v),
                    ),
                    const Padding(
                      padding: EdgeInsets.only(top: 4),
                      child: Text('/mcp-server/http will be appended automatically',
                          style: TextStyle(color: Colors.white38, fontSize: 11)),
                    ),
                    const SizedBox(height: 12),
                    _buildLabel('Access Token'),
                    Row(
                      children: [
                        Expanded(
                          child: TextFormField(
                            initialValue: _n8nToken,
                            obscureText: true,
                            style: const TextStyle(color: Colors.white),
                            decoration: _inputDecoration(hint: 'n8n_api_...'),
                            onChanged: (v) => setState(() => _n8nToken = v),
                          ),
                        ),
                        const SizedBox(width: 8),
                        _buildTestButton(
                          testing: _testingN8n,
                          connected: _n8nConnected,
                          onPressed: _testN8n,
                        ),
                      ],
                    ),
                    if (_n8nError != null)
                      Padding(
                        padding: const EdgeInsets.only(top: 4),
                        child: Text(_n8nError!, style: const TextStyle(color: Colors.red, fontSize: 12)),
                      ),
                    if (_n8nConnected)
                      const Padding(
                        padding: EdgeInsets.only(top: 4),
                        child: Text('Connected', style: TextStyle(color: Color(0xFF45997C), fontSize: 12)),
                      ),
                  ],
                ]),

                const SizedBox(height: 12),

                // Friday Assistant
                _buildCard([
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text('Friday Assistant', style: TextStyle(color: Colors.white, fontWeight: FontWeight.w600)),
                      Switch(
                        value: _fridayEnabled,
                        onChanged: (v) => setState(() => _fridayEnabled = v),
                        activeTrackColor: const Color(0xFF45997C),
                      ),
                    ],
                  ),
                  if (_fridayEnabled) ...[
                    const SizedBox(height: 12),
                    _buildLabel('Host URL'),
                    TextFormField(
                      initialValue: _fridayHost,
                      style: const TextStyle(color: Colors.white),
                      decoration: _inputDecoration(hint: 'http://10.0.0.55:18789'),
                      onChanged: (v) => setState(() => _fridayHost = v),
                    ),
                    const Padding(
                      padding: EdgeInsets.only(top: 4),
                      child: Text('Clawdbot API endpoint for advanced AI assistance',
                          style: TextStyle(color: Colors.white38, fontSize: 11)),
                    ),
                    const SizedBox(height: 12),
                    _buildLabel('API Token'),
                    Row(
                      children: [
                        Expanded(
                          child: TextFormField(
                            initialValue: _fridayToken,
                            obscureText: true,
                            style: const TextStyle(color: Colors.white),
                            decoration: _inputDecoration(hint: 'your_api_token'),
                            onChanged: (v) => setState(() => _fridayToken = v),
                          ),
                        ),
                        const SizedBox(width: 8),
                        _buildTestButton(
                          testing: _testingFriday,
                          connected: _fridayConnected,
                          onPressed: _testFriday,
                        ),
                      ],
                    ),
                    if (_fridayError != null)
                      Padding(
                        padding: const EdgeInsets.only(top: 4),
                        child: Text(_fridayError!, style: const TextStyle(color: Colors.red, fontSize: 12)),
                      ),
                    if (_fridayConnected)
                      const Padding(
                        padding: EdgeInsets.only(top: 4),
                        child: Text('Connected', style: TextStyle(color: Color(0xFF45997C), fontSize: 12)),
                      ),
                  ],
                ]),

                const SizedBox(height: 24),
                _buildSectionHeader('LLM Settings', Icons.tune),
                _buildCard([
                  Row(
                    children: [
                      Expanded(
                        child: _buildNumberField(
                          label: 'Temperature',
                          value: _temperature,
                          min: 0.0,
                          max: 2.0,
                          decimals: 1,
                          onChanged: (v) => setState(() => _temperature = v),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: _buildIntField(
                          label: 'Context Size',
                          value: _numCtx,
                          min: 1024,
                          max: 131072,
                          step: 1024,
                          onChanged: (v) => setState(() => _numCtx = v),
                        ),
                      ),
                    ],
                  ),
                  Row(
                    children: [
                      Expanded(
                        child: _buildIntField(
                          label: 'Max Turns',
                          value: _maxTurns,
                          min: 1,
                          max: 100,
                          onChanged: (v) => setState(() => _maxTurns = v),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: _buildIntField(
                          label: 'Tool Cache',
                          value: _toolCacheSize,
                          min: 0,
                          max: 10,
                          onChanged: (v) => setState(() => _toolCacheSize = v),
                        ),
                      ),
                    ],
                  ),
                  const Divider(color: Colors.white24, height: 24),
                  // Turn Detection section
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Allow Interruptions',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 14,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                          Text(
                            'Interrupt the agent while speaking',
                            style: TextStyle(
                              color: Colors.white54,
                              fontSize: 12,
                            ),
                          ),
                        ],
                      ),
                      Switch(
                        value: _allowInterruptions,
                        onChanged: (v) => setState(() => _allowInterruptions = v),
                        activeTrackColor: const Color(0xFF45997C),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  _buildNumberField(
                    label: 'Endpointing Delay (s)',
                    value: _minEndpointingDelay,
                    min: 0.1,
                    max: 1.0,
                    decimals: 1,
                    onChanged: (v) => setState(() => _minEndpointingDelay = v),
                  ),
                  const Text(
                    'How long to wait after you stop speaking',
                    style: TextStyle(color: Colors.white38, fontSize: 11),
                  ),
                  const Divider(color: Colors.white24, height: 24),
                  // Noise Suppression section
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Noise Suppression',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 14,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                          Text(
                            'Filter background noise (DeepFilterNet)',
                            style: TextStyle(
                              color: Colors.white54,
                              fontSize: 12,
                            ),
                          ),
                        ],
                      ),
                      Switch(
                        value: _noiseSuppressionEnabled,
                        onChanged: (v) => setState(() => _noiseSuppressionEnabled = v),
                        activeTrackColor: const Color(0xFF45997C),
                      ),
                    ],
                  ),
                  if (_noiseSuppressionEnabled) ...[
                    const SizedBox(height: 12),
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            const Text(
                              'Suppression Strength',
                              style: TextStyle(
                                color: Colors.white70,
                                fontSize: 13,
                              ),
                            ),
                            Text(
                              '${_noiseSuppressionAttenDb.toInt()}dB',
                              style: const TextStyle(
                                color: Colors.white70,
                                fontSize: 13,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        SliderTheme(
                          data: SliderTheme.of(context).copyWith(
                            activeTrackColor: const Color(0xFF45997C),
                            thumbColor: const Color(0xFF45997C),
                            inactiveTrackColor: Colors.white24,
                          ),
                          child: Slider(
                            value: _noiseSuppressionAttenDb,
                            min: 20,
                            max: 100,
                            divisions: 8,
                            onChanged: (v) => setState(() => _noiseSuppressionAttenDb = v),
                          ),
                        ),
                        const Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text('Conservative', style: TextStyle(color: Colors.white38, fontSize: 11)),
                            Text('Aggressive', style: TextStyle(color: Colors.white38, fontSize: 11)),
                          ],
                        ),
                        const SizedBox(height: 4),
                        const Text(
                          'Higher values remove more noise but may affect voice quality',
                          style: TextStyle(color: Colors.white38, fontSize: 11),
                        ),
                      ],
                    ),
                  ],
                  const Divider(color: Colors.white24, height: 24),
                  // Energy Gate section
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Energy Gate',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 14,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                          Text(
                            'Filter quiet/distant sounds (TV)',
                            style: TextStyle(
                              color: Colors.white54,
                              fontSize: 12,
                            ),
                          ),
                        ],
                      ),
                      Switch(
                        value: _energyGateEnabled,
                        onChanged: (v) => setState(() => _energyGateEnabled = v),
                        activeTrackColor: const Color(0xFF45997C),
                      ),
                    ],
                  ),
                  if (_energyGateEnabled) ...[
                    const SizedBox(height: 12),
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            const Text(
                              'Threshold',
                              style: TextStyle(
                                color: Colors.white70,
                                fontSize: 13,
                              ),
                            ),
                            Text(
                              '${_energyGateThresholdDb.toInt()}dB',
                              style: const TextStyle(
                                color: Colors.white70,
                                fontSize: 13,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        SliderTheme(
                          data: SliderTheme.of(context).copyWith(
                            activeTrackColor: const Color(0xFF45997C),
                            thumbColor: const Color(0xFF45997C),
                            inactiveTrackColor: Colors.white24,
                          ),
                          child: Slider(
                            value: _energyGateThresholdDb,
                            min: -50,
                            max: -20,
                            divisions: 30,
                            onChanged: (v) => setState(() => _energyGateThresholdDb = v),
                          ),
                        ),
                        const Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text('More Sensitive', style: TextStyle(color: Colors.white38, fontSize: 11)),
                            Text('Less Sensitive', style: TextStyle(color: Colors.white38, fontSize: 11)),
                          ],
                        ),
                        const SizedBox(height: 4),
                        const Text(
                          '-40dB = quiet room, -35dB = speech, -30dB = loud speech',
                          style: TextStyle(color: Colors.white38, fontSize: 11),
                        ),
                      ],
                    ),
                  ],
                ]),

                const SizedBox(height: 24),
                _buildSectionHeader('Wake Word', Icons.hearing),
                _buildCard([
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Server-Side Wake Word',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 14,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                          Text(
                            'Activate with wake phrase',
                            style: TextStyle(
                              color: Colors.white54,
                              fontSize: 12,
                            ),
                          ),
                        ],
                      ),
                      Switch(
                        value: _wakeWordEnabled,
                        onChanged: (v) => setState(() => _wakeWordEnabled = v),
                        activeTrackColor: const Color(0xFF45997C),
                      ),
                    ],
                  ),
                  if (_wakeWordEnabled) ...[
                    const SizedBox(height: 12),
                    _buildWakeWordModelDropdown(),
                    Row(
                      children: [
                        Expanded(
                          child: _buildNumberField(
                            label: 'Threshold',
                            value: _wakeWordThreshold,
                            min: 0.1,
                            max: 1.0,
                            decimals: 1,
                            onChanged: (v) => setState(() => _wakeWordThreshold = v),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: _buildNumberField(
                            label: 'Timeout (s)',
                            value: _wakeWordTimeout,
                            min: 1.0,
                            max: 30.0,
                            decimals: 1,
                            onChanged: (v) => setState(() => _wakeWordTimeout = v),
                          ),
                        ),
                      ],
                    ),
                  ],
                ]),

                // TV/Media Rejection Section
                const SizedBox(height: 24),
                _buildSectionHeader('Audio Filter', Icons.filter_alt_outlined),
                _buildCard([
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'TV / Media Rejection',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 14,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                          Text(
                            'Filter TV, radio, and video playback',
                            style: TextStyle(
                              color: Colors.white54,
                              fontSize: 12,
                            ),
                          ),
                        ],
                      ),
                      Switch(
                        value: _tvRejectionEnabled,
                        onChanged: (v) => setState(() => _tvRejectionEnabled = v),
                        activeTrackColor: const Color(0xFF45997C),
                      ),
                    ],
                  ),
                  if (_tvRejectionEnabled) ...[
                    const SizedBox(height: 16),
                    const Text(
                      'Uses AI-powered audio analysis to distinguish live speech from media playback.',
                      style: TextStyle(color: Colors.white54, fontSize: 12),
                    ),
                    const SizedBox(height: 12),
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            const Text(
                              'Liveness Threshold',
                              style: TextStyle(
                                color: Colors.white70,
                                fontSize: 13,
                              ),
                            ),
                            Text(
                              _tvRejectionMinLiveness.toStringAsFixed(2),
                              style: const TextStyle(
                                color: Colors.white70,
                                fontSize: 13,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        SliderTheme(
                          data: SliderTheme.of(context).copyWith(
                            activeTrackColor: const Color(0xFF45997C),
                            thumbColor: const Color(0xFF45997C),
                            inactiveTrackColor: Colors.white24,
                          ),
                          child: Slider(
                            value: _tvRejectionMinLiveness,
                            min: 0.1,
                            max: 0.5,
                            divisions: 8,
                            onChanged: (v) => setState(() => _tvRejectionMinLiveness = v),
                          ),
                        ),
                        const Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text('More Permissive', style: TextStyle(color: Colors.white38, fontSize: 11)),
                            Text('Stricter', style: TextStyle(color: Colors.white38, fontSize: 11)),
                          ],
                        ),
                      ],
                    ),
                    const Divider(color: Colors.white24, height: 24),
                    // Music Detection
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        const Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'Music Detection',
                              style: TextStyle(color: Colors.white70, fontSize: 13),
                            ),
                            Text(
                              'Detect radio and video music',
                              style: TextStyle(color: Colors.white38, fontSize: 11),
                            ),
                          ],
                        ),
                        Switch(
                          value: _mediaNoiseMusicDetection,
                          onChanged: (v) => setState(() => _mediaNoiseMusicDetection = v),
                          activeTrackColor: const Color(0xFF45997C),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    // Stereo Detection
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        const Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'Stereo Detection',
                              style: TextStyle(color: Colors.white70, fontSize: 13),
                            ),
                            Text(
                              'Detect wide stereo (media vs live)',
                              style: TextStyle(color: Colors.white38, fontSize: 11),
                            ),
                          ],
                        ),
                        Switch(
                          value: _mediaNoiseStereoDetection,
                          onChanged: (v) => setState(() => _mediaNoiseStereoDetection = v),
                          activeTrackColor: const Color(0xFF45997C),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    // Playback Voice Detection
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        const Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'Playback Voice Detection',
                              style: TextStyle(color: Colors.white70, fontSize: 13),
                            ),
                            Text(
                              'Detect voices from video/podcast',
                              style: TextStyle(color: Colors.white38, fontSize: 11),
                            ),
                          ],
                        ),
                        Switch(
                          value: _mediaNoisePlaybackVoiceDetection,
                          onChanged: (v) => setState(() => _mediaNoisePlaybackVoiceDetection = v),
                          activeTrackColor: const Color(0xFF45997C),
                        ),
                      ],
                    ),
                  ],
                ]),

                // Speaker Recognition Section
                const SizedBox(height: 12),
                _buildCard([
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Speaker Recognition',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 14,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                          Text(
                            'Voice biometrics like Voice Match',
                            style: TextStyle(
                              color: Colors.white54,
                              fontSize: 12,
                            ),
                          ),
                        ],
                      ),
                      Switch(
                        value: _speakerRecognitionEnabled,
                        onChanged: (v) {
                          setState(() => _speakerRecognitionEnabled = v);
                          if (v) unawaited(_loadSpeakers());
                        },
                        activeTrackColor: const Color(0xFF45997C),
                      ),
                    ],
                  ),
                  if (_speakerRecognitionEnabled) ...[
                    const SizedBox(height: 12),
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: Colors.blue.withValues(alpha: 0.1),
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(color: Colors.blue.withValues(alpha: 0.3)),
                      ),
                      child: const Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Requires: resemblyzer package',
                            style: TextStyle(color: Colors.blue, fontSize: 12, fontWeight: FontWeight.w500),
                          ),
                          SizedBox(height: 2),
                          Text(
                            'pip install resemblyzer',
                            style: TextStyle(color: Colors.blue, fontSize: 11),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 16),
                    // Enrolled Speakers
                    const Text(
                      'Enrolled Speakers',
                      style: TextStyle(color: Colors.white, fontSize: 13, fontWeight: FontWeight.w500),
                    ),
                    const SizedBox(height: 8),
                    if (_speakersLoading)
                      const Padding(
                        padding: EdgeInsets.symmetric(vertical: 8),
                        child: Row(
                          children: [
                            SizedBox(
                              width: 16,
                              height: 16,
                              child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white54),
                            ),
                            SizedBox(width: 8),
                            Text('Loading speakers...', style: TextStyle(color: Colors.white54, fontSize: 12)),
                          ],
                        ),
                      )
                    else if (_speakers.isEmpty)
                      const Text(
                        'No speakers enrolled. Enroll speakers via web UI.',
                        style: TextStyle(color: Colors.white54, fontSize: 12),
                      )
                    else
                      Column(
                        children: _speakers.map((speaker) {
                          return Container(
                            margin: const EdgeInsets.only(bottom: 8),
                            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                            decoration: BoxDecoration(
                              color: const Color(0xFF2A2A2A),
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                Text(
                                  speaker['name'] ?? 'Unknown',
                                  style: const TextStyle(color: Colors.white, fontSize: 13),
                                ),
                                Text(
                                  '${speaker['enrollment_samples'] ?? 0} samples',
                                  style: const TextStyle(color: Colors.white54, fontSize: 11),
                                ),
                              ],
                            ),
                          );
                        }).toList(),
                      ),
                    const Divider(color: Colors.white24, height: 24),
                    // Threshold
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            const Text(
                              'Verification Threshold',
                              style: TextStyle(color: Colors.white70, fontSize: 13),
                            ),
                            Text(
                              _speakerRecognitionThreshold.toStringAsFixed(2),
                              style: const TextStyle(color: Colors.white70, fontSize: 13),
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        SliderTheme(
                          data: SliderTheme.of(context).copyWith(
                            activeTrackColor: const Color(0xFF45997C),
                            thumbColor: const Color(0xFF45997C),
                            inactiveTrackColor: Colors.white24,
                          ),
                          child: Slider(
                            value: _speakerRecognitionThreshold,
                            min: 0.5,
                            max: 0.9,
                            divisions: 8,
                            onChanged: (v) => setState(() => _speakerRecognitionThreshold = v),
                          ),
                        ),
                        const Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text('More Lenient', style: TextStyle(color: Colors.white38, fontSize: 11)),
                            Text('Stricter', style: TextStyle(color: Colors.white38, fontSize: 11)),
                          ],
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),
                    // Require Recognized Speaker
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        const Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'Require Recognized Speaker',
                              style: TextStyle(color: Colors.white70, fontSize: 13),
                            ),
                            Text(
                              'Only respond to enrolled speakers',
                              style: TextStyle(color: Colors.white38, fontSize: 11),
                            ),
                          ],
                        ),
                        Switch(
                          value: _speakerRecognitionRequired,
                          onChanged: (v) => setState(() => _speakerRecognitionRequired = v),
                          activeTrackColor: const Color(0xFF45997C),
                        ),
                      ],
                    ),
                  ],
                ]),

                const SizedBox(height: 16),
                const Text(
                  'Note: Model, context size, and wake word changes take effect on next session.',
                  style: TextStyle(color: Colors.white38, fontSize: 12),
                  textAlign: TextAlign.center,
                ),
              ] else if (!isFirstSetup) ...[
                const SizedBox(height: 48),
                const Center(
                  child: Column(
                    children: [
                      Icon(Icons.cloud_off, size: 48, color: Colors.white38),
                      SizedBox(height: 16),
                      Text(
                        'Connect to server to configure agent settings',
                        style: TextStyle(color: Colors.white54),
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                ),
              ],

              const SizedBox(height: 32),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildProviderToggle({
    required List<String> options,
    required List<String> labels,
    required List<String> subtitles,
    required String selected,
    required ValueChanged<String> onChanged,
  }) {
    return Row(
      children: [
        for (int i = 0; i < options.length; i++)
          Expanded(
            child: GestureDetector(
              onTap: () => onChanged(options[i]),
              child: Container(
                padding: const EdgeInsets.symmetric(vertical: 12),
                decoration: BoxDecoration(
                  color: selected == options[i]
                      ? const Color(0xFF45997C).withValues(alpha: 0.2)
                      : const Color(0xFF2A2A2A),
                  border: Border.all(
                    color: selected == options[i]
                        ? const Color(0xFF45997C)
                        : Colors.white.withValues(alpha: 0.1),
                  ),
                  borderRadius: BorderRadius.only(
                    topLeft: i == 0 ? const Radius.circular(8) : Radius.zero,
                    bottomLeft: i == 0 ? const Radius.circular(8) : Radius.zero,
                    topRight: i == options.length - 1 ? const Radius.circular(8) : Radius.zero,
                    bottomRight: i == options.length - 1 ? const Radius.circular(8) : Radius.zero,
                  ),
                ),
                child: Column(
                  children: [
                    Text(
                      labels[i],
                      style: TextStyle(
                        color: selected == options[i] ? Colors.white : Colors.white70,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      subtitles[i],
                      style: TextStyle(
                        color: Colors.white.withValues(alpha: 0.5),
                        fontSize: 10,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
      ],
    );
  }

  Widget _buildTestButton({
    required bool testing,
    required bool connected,
    required VoidCallback onPressed,
  }) {
    return TextButton(
      onPressed: testing ? null : onPressed,
      style: TextButton.styleFrom(
        backgroundColor: connected ? const Color(0xFF45997C) : const Color(0xFF2A2A2A),
        foregroundColor: Colors.white,
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      ),
      child: testing
          ? const SizedBox(
              width: 16,
              height: 16,
              child: CircularProgressIndicator(
                strokeWidth: 2,
                valueColor: AlwaysStoppedAnimation(Colors.white),
              ),
            )
          : Text(connected ? '✓' : 'Test'),
    );
  }

  Widget _buildSectionHeader(String title, IconData icon) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        children: [
          Icon(icon, color: const Color(0xFF45997C), size: 20),
          const SizedBox(width: 8),
          Text(
            title,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 16,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCard(List<Widget> children) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF242424),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withValues(alpha: 0.1)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: children,
      ),
    );
  }

  Widget _buildLabel(String label) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Text(
        label,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 14,
          fontWeight: FontWeight.w500,
        ),
      ),
    );
  }

  InputDecoration _inputDecoration({String? hint}) {
    return InputDecoration(
      hintText: hint,
      hintStyle: const TextStyle(color: Colors.white38),
      filled: true,
      fillColor: const Color(0xFF2A2A2A),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(8),
        borderSide: BorderSide.none,
      ),
      contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
    );
  }

  Widget _buildTextField({
    required String label,
    required String value,
    required ValueChanged<String> onChanged,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildLabel(label),
        TextFormField(
          initialValue: value,
          style: const TextStyle(color: Colors.white),
          decoration: _inputDecoration(),
          onChanged: onChanged,
        ),
        const SizedBox(height: 16),
      ],
    );
  }

  Widget _buildDropdown({
    required String label,
    required String value,
    required List<String> options,
    required ValueChanged<String?> onChanged,
  }) {
    final safeValue = options.contains(value) ? value : (options.isNotEmpty ? options.first : value);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildLabel(label),
        DropdownButtonFormField<String>(
          initialValue: safeValue,
          style: const TextStyle(color: Colors.white),
          dropdownColor: const Color(0xFF2A2A2A),
          decoration: _inputDecoration(),
          items: options.map((o) => DropdownMenuItem(value: o, child: Text(o))).toList(),
          onChanged: onChanged,
        ),
        const SizedBox(height: 16),
      ],
    );
  }

  Widget _buildNumberField({
    required String label,
    required double value,
    required double min,
    required double max,
    required int decimals,
    required ValueChanged<double> onChanged,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildLabel(label),
        TextFormField(
          initialValue: value.toStringAsFixed(decimals),
          style: const TextStyle(color: Colors.white),
          keyboardType: const TextInputType.numberWithOptions(decimal: true),
          decoration: _inputDecoration(),
          onChanged: (v) {
            final parsed = double.tryParse(v);
            if (parsed != null && parsed >= min && parsed <= max) {
              onChanged(parsed);
            }
          },
        ),
        const SizedBox(height: 16),
      ],
    );
  }

  Widget _buildIntField({
    required String label,
    required int value,
    required int min,
    required int max,
    int step = 1,
    required ValueChanged<int> onChanged,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildLabel(label),
        TextFormField(
          initialValue: value.toString(),
          style: const TextStyle(color: Colors.white),
          keyboardType: TextInputType.number,
          decoration: _inputDecoration(),
          onChanged: (v) {
            final parsed = int.tryParse(v);
            if (parsed != null && parsed >= min && parsed <= max) {
              onChanged(parsed);
            }
          },
        ),
        const SizedBox(height: 16),
      ],
    );
  }

  String _formatModelName(String path) {
    return path
        .replaceAll('models/', '')
        .replaceAll('.onnx', '')
        .replaceAll('_', ' ')
        .split(' ')
        .map((w) => w.isNotEmpty ? '${w[0].toUpperCase()}${w.substring(1)}' : '')
        .join(' ');
  }

  Widget _buildWakeWordModelDropdown() {
    final options = _wakeWordModels.isNotEmpty ? _wakeWordModels : [_wakeWordModel];
    final safeValue = options.contains(_wakeWordModel) ? _wakeWordModel : options.first;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Wake Word Model',
          style: TextStyle(color: Colors.white54, fontSize: 12),
        ),
        const SizedBox(height: 4),
        DropdownButtonFormField<String>(
          initialValue: safeValue,
          style: const TextStyle(color: Colors.white),
          dropdownColor: const Color(0xFF2A2A2A),
          decoration: _inputDecoration(),
          items: options
              .map((m) => DropdownMenuItem(
                    value: m,
                    child: Text(_formatModelName(m)),
                  ))
              .toList(),
          onChanged: (v) => setState(() => _wakeWordModel = v ?? _wakeWordModel),
        ),
        const SizedBox(height: 12),
      ],
    );
  }
}
