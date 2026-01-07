import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:livekit_client/livekit_client.dart' as sdk;

class JarvisVisualizer extends StatefulWidget {
  final sdk.Participant? participant;
  final double size;
  final Color color;

  const JarvisVisualizer({
    super.key,
    required this.participant,
    this.size = 300,
    this.color = const Color(0xFF00FFFF), // Cyan default
  });

  @override
  State<JarvisVisualizer> createState() => _JarvisVisualizerState();
}

class _JarvisVisualizerState extends State<JarvisVisualizer> with TickerProviderStateMixin {
  late AnimationController _rotationController;
  
  @override
  void initState() {
    super.initState();
    _rotationController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 10),
    )..repeat();
    
    if (widget.participant != null) {
      widget.participant!.addListener(_onParticipantChanged);
    }
  }

  @override
  void didUpdateWidget(JarvisVisualizer oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.participant != widget.participant) {
      oldWidget.participant?.removeListener(_onParticipantChanged);
      widget.participant?.addListener(_onParticipantChanged);
    }
  }

  @override
  void dispose() {
    widget.participant?.removeListener(_onParticipantChanged);
    _rotationController.dispose();
    super.dispose();
  }

  void _onParticipantChanged() {
    if (mounted) {
      setState(() {});
    }
  }

  @override
  Widget build(BuildContext context) {
    // Get audio level (0.0 to 1.0)
    final double rawAudioLevel = widget.participant?.audioLevel ?? 0.0;
    
    // Target scale calculation:
    // Boost low levels significantly so even whispers are visible.
    // More aggressive scaling for prominent animation.
    final double targetScale = 1.0 + (rawAudioLevel * 8.0).clamp(0.0, 2.5);

    return SizedBox(
      width: widget.size,
      height: widget.size,
      child: TweenAnimationBuilder<double>(
        tween: Tween<double>(begin: 1.0, end: targetScale),
        duration: const Duration(milliseconds: 100),
        curve: Curves.easeOut,
        builder: (context, smoothedScale, child) {
          return AnimatedBuilder(
            animation: _rotationController,
            builder: (context, child) {
              // Calculate dynamic rotation speed based on activity
              // When speaking, rotate significantly faster for prominent effect
              final double activityBoost = (smoothedScale - 1.0) * 1.5;
              final double currentRotation = _rotationController.value * 2 * math.pi + (activityBoost * 4);

              return CustomPaint(
                painter: JarvisPainter(
                  rotation: currentRotation,
                  scale: smoothedScale,
                  color: widget.color,
                  waveTime: _rotationController.value,
                  audioLevel: rawAudioLevel,
                ),
              );
            },
          );
        },
      ),
    );
  }
}

class JarvisPainter extends CustomPainter {
  final double rotation;
  final double scale;
  final Color color;
  final double waveTime;
  final double audioLevel;

  JarvisPainter({
    required this.rotation,
    required this.scale,
    required this.color,
    required this.waveTime,
    required this.audioLevel,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final maxRadius = size.width / 2;
    
    final Paint mainPaint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    // Use current scale to determine opacity intensity
    // More dramatic intensity range for prominent visual feedback
    final double intensity = ((scale - 1.0) * 1.5).clamp(0.0, 1.0);
    
    final Paint glowPaint = Paint()
      ..color = color.withValues(alpha: 0.2 + (0.6 * intensity))
      ..style = PaintingStyle.fill
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 35);

    // --- Core Reactor ---
    
    // Dynamic radius responding to audio
    // Base radius slightly larger
    final double baseRadius = maxRadius * 0.25; 
    // More dramatic pulse effect
    final double currentRadius = baseRadius * (0.7 + (scale * 0.3));
    
    // Central core glow (the light source) - larger and more prominent
    canvas.drawCircle(center, currentRadius * 2.0, glowPaint); 
    canvas.drawCircle(center, currentRadius * 1.2, glowPaint..color = color.withValues(alpha: 0.3 + (0.5 * intensity))); 
    
    // Inner solid core ring - thicker and brighter
    canvas.drawCircle(
      center, 
      baseRadius * 0.6, 
      mainPaint..color = color.withValues(alpha: 0.95)..strokeWidth = 4
    );
    
    // Rotating Triangle/Reactor Structure
    canvas.save();
    canvas.translate(center.dx, center.dy);
    canvas.rotate(-rotation * 1.5); // Faster counter-rotation
    _drawPoly(canvas, 3, baseRadius * 0.5, mainPaint..strokeWidth = 2);
    canvas.restore();
    
    // --- Middle Rings (Multiple Wavy Layers - Speech Reactive) ---
    
    canvas.save();
    canvas.translate(center.dx, center.dy);
    
    // Base ring radius that expands with speech
    final double midRingRadius = maxRadius * 0.5 + (intensity * 25); 
    
    // Multiple wave layers at different frequencies for rich animation
    // Each layer reacts differently to create speech-like motion
    
    // Outer wave layer - slow, subtle
    canvas.save();
    canvas.rotate(rotation * 0.8);
    _drawWavyRing(
      canvas, 
      midRingRadius * 1.15,
      96,
      audioLevel * 6, // Subtle amplitude
      waveTime * 5, 
      4, // 4 waves
      mainPaint..strokeWidth = 1.5..color = color.withValues(alpha: 0.4 + (0.3 * intensity))
    );
    canvas.restore();
    
    // Mid wave layer - medium speed, responds to speech peaks
    canvas.save();
    canvas.rotate(rotation * -0.6);
    _drawWavyRing(
      canvas, 
      midRingRadius,
      96,
      audioLevel * 8, // More amplitude for speech reactivity
      waveTime * 8, 
      5, // 5 waves
      mainPaint..strokeWidth = 2..color = color.withValues(alpha: 0.6 + (0.3 * intensity))
    );
    canvas.restore();
    
    // Inner wave layer - fast, speech-reactive
    canvas.save();
    canvas.rotate(rotation * 1.2);
    _drawWavyRing(
       canvas,
       midRingRadius * 0.85,
       96,
       audioLevel * 7, // Dynamic amplitude
       waveTime * 12, // Faster wave animation
       6, // 6 waves
       mainPaint..strokeWidth = 1.8..color = color.withValues(alpha: 0.5 + (0.4 * intensity))
    );
    canvas.restore();
    
    // Innermost wave - very fast, like vocal vibrations
    canvas.save();
    canvas.rotate(rotation * -1.5);
    _drawWavyRing(
       canvas,
       midRingRadius * 0.7,
       80,
       audioLevel * 5, // Subtle but fast
       waveTime * 15,
       7, // 7 waves
       mainPaint..strokeWidth = 1.2..color = color.withValues(alpha: 0.4 + (0.3 * intensity))
    );
    canvas.restore();
    
    canvas.restore();

    // --- Outer Interface (Stable but complex) ---
    final double outerRadius = maxRadius * 0.85;

    canvas.save();
    canvas.translate(center.dx, center.dy);
    
    // Outer wave layers - subtle but speech-reactive
    canvas.save();
    canvas.rotate(rotation * 0.1); 
    _drawWavyRing(
      canvas,
      outerRadius,
      120,
      audioLevel * 4, // Very subtle wave
      waveTime * 4,
      3, // Fewer waves for outer ring
      mainPaint..strokeWidth = 1.5..color = color.withValues(alpha: 0.5 + (0.2 * intensity))
    );
    canvas.restore();
    
    // Outer thin wavy line
    canvas.save();
    canvas.rotate(rotation * -0.15);
    _drawWavyRing(
      canvas,
      maxRadius * 0.92,
      100,
      audioLevel * 3, // Very subtle
      waveTime * -3,
      4,
      mainPaint..strokeWidth = 1..color = color.withValues(alpha: 0.35 + (0.2 * intensity))
    );
    canvas.restore();
    
    // Tick marks - subtle rotation
    canvas.rotate(-rotation * 0.2);
    _drawTickMarks(canvas, maxRadius * 0.95, 48, mainPaint..strokeWidth = 1..color = color.withValues(alpha: 0.4 + (0.2 * intensity)));
    
    canvas.restore();
  }

  void _drawPoly(Canvas canvas, int sides, double radius, Paint paint) {
    final path = Path();
    final angle = (math.pi * 2) / sides;
    
    path.moveTo(
      radius * math.cos(0.0), 
      radius * math.sin(0.0)
    );
    
    for (int i = 1; i <= sides; i++) {
      path.lineTo(
        radius * math.cos(angle * i),
        radius * math.sin(angle * i)
      );
    }
    path.close();
    canvas.drawPath(path, paint);
  }

  void _drawWavyRing(Canvas canvas, double radius, int points, double waveAmplitude, double waveTime, int waveCount, Paint paint) {
    final path = Path();
    final double angleStep = (2 * math.pi) / points;
    
    for (int i = 0; i <= points; i++) {
      final double angle = i * angleStep;
      // Create wave effect using sine function
      final double wave = math.sin(angle * waveCount + waveTime) * waveAmplitude;
      final double currentRadius = radius + wave;
      
      final double x = currentRadius * math.cos(angle);
      final double y = currentRadius * math.sin(angle);
      
      if (i == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }
    
    path.close();
    canvas.drawPath(path, paint..style = PaintingStyle.stroke);
  }
  
  void _drawTechRing(Canvas canvas, double radius, int segments, Paint paint) {
    final double step = (2 * math.pi) / segments;
    final double gap = 0.4;
    
    for (int i = 0; i < segments; i++) {
        canvas.drawArc(
            Rect.fromCircle(center: Offset.zero, radius: radius),
            i * step + gap/2,
            step - gap,
            false,
            paint
        );
    }
  }
  
  void _drawSegmentedRing(Canvas canvas, double radius, int segments, Paint paint) {
      final double step = (2 * math.pi) / segments;
      final double gap = 0.1;

      for (int i = 0; i < segments; i++) {
          canvas.drawArc(
              Rect.fromCircle(center: Offset.zero, radius: radius),
              i * step + gap/2,
              step - gap,
              false,
              paint
          );
      }
  }

  void _drawTickMarks(Canvas canvas, double radius, int count, Paint paint) {
      final double step = (2 * math.pi) / count;
      for (int i = 0; i < count; i++) {
          final double angle = i * step;
          final double innerR = radius - 5;
          final double outerR = radius;
          
          canvas.drawLine(
              Offset(innerR * math.cos(angle), innerR * math.sin(angle)),
              Offset(outerR * math.cos(angle), outerR * math.sin(angle)),
              paint
          );
      }
  }

  @override
  bool shouldRepaint(JarvisPainter oldDelegate) => 
      oldDelegate.rotation != rotation || 
      oldDelegate.scale != scale || 
      oldDelegate.color != color ||
      oldDelegate.waveTime != waveTime ||
      oldDelegate.audioLevel != audioLevel;
}
