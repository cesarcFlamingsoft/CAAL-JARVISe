import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:livekit_client/livekit_client.dart' as sdk;

class BarVisualizer extends StatefulWidget {
  final sdk.Participant? participant;
  final int barCount;
  final Color color;
  final double barWidth;
  final double minHeight;
  final double maxHeight;

  const BarVisualizer({
    super.key,
    required this.participant,
    this.barCount = 5,
    this.color = const Color(0xFF00FFFF), // Cyan default
    this.barWidth = 8.0,
    this.minHeight = 8.0,
    this.maxHeight = 60.0,
  });

  @override
  State<BarVisualizer> createState() => _BarVisualizerState();
}

class _BarVisualizerState extends State<BarVisualizer> with TickerProviderStateMixin {
  late List<AnimationController> _controllers;
  late List<Animation<double>> _animations;
  
  @override
  void initState() {
    super.initState();
    _initializeAnimations();
    
    if (widget.participant != null) {
      widget.participant!.addListener(_onParticipantChanged);
    }
  }

  void _initializeAnimations() {
    _controllers = List.generate(
      widget.barCount,
      (index) => AnimationController(
        vsync: this,
        duration: Duration(milliseconds: 100 + (index * 20)),
      ),
    );

    _animations = _controllers.map((controller) {
      return Tween<double>(begin: widget.minHeight, end: widget.maxHeight)
          .animate(CurvedAnimation(parent: controller, curve: Curves.easeOut));
    }).toList();
  }

  @override
  void didUpdateWidget(BarVisualizer oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.participant != widget.participant) {
      oldWidget.participant?.removeListener(_onParticipantChanged);
      widget.participant?.addListener(_onParticipantChanged);
    }
  }

  @override
  void dispose() {
    widget.participant?.removeListener(_onParticipantChanged);
    for (var controller in _controllers) {
      controller.dispose();
    }
    super.dispose();
  }

  void _onParticipantChanged() {
    if (mounted) {
      final audioLevel = widget.participant?.audioLevel ?? 0.0;
      _updateBars(audioLevel);
    }
  }

  void _updateBars(double audioLevel) {
    // Boost audio level for better visibility
    final boostedLevel = (audioLevel * 3.0).clamp(0.0, 1.0);
    
    for (int i = 0; i < _controllers.length; i++) {
      // Add some variation between bars for visual effect
      final variation = math.sin((i * math.pi / _controllers.length) + DateTime.now().millisecondsSinceEpoch / 200);
      final targetValue = boostedLevel * (0.8 + variation * 0.2);
      
      _controllers[i].animateTo(
        targetValue.clamp(0.0, 1.0),
        duration: const Duration(milliseconds: 100),
        curve: Curves.easeOut,
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      crossAxisAlignment: CrossAxisAlignment.center,
      children: List.generate(widget.barCount, (index) {
        return Padding(
          padding: EdgeInsets.symmetric(horizontal: widget.barWidth * 0.3),
          child: AnimatedBuilder(
            animation: _animations[index],
            builder: (context, child) {
              return Container(
                width: widget.barWidth,
                height: _animations[index].value,
                decoration: BoxDecoration(
                  color: widget.color,
                  borderRadius: BorderRadius.circular(widget.barWidth / 2),
                ),
              );
            },
          ),
        );
      }),
    );
  }
}
