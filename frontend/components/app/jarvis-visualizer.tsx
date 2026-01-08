'use client';

import React, { useCallback, useEffect, useMemo, useRef } from 'react';
import type { TrackReference } from '@livekit/components-react';

interface JarvisVisualizerProps {
  trackRef?: TrackReference;
  color?: string;
  className?: string;
}

function hexToRgb(hex: string): { r: number; g: number; b: number } | null {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16),
      }
    : null;
}

export function JarvisVisualizer({
  trackRef,
  color = '#00FFFF',
  className,
}: JarvisVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const requestRef = useRef<number | undefined>(undefined);
  const rotationRef = useRef<number>(0);
  const smoothedScaleRef = useRef(1.0);
  const rgb = useMemo(() => hexToRgb(color) || { r: 0, g: 255, b: 255 }, [color]);

  const draw = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      width: number,
      height: number,
      rawAudioLevel: number,
      baseRotation: number,
      rgb: { r: number; g: number; b: number }
    ) => {
      ctx.clearRect(0, 0, width, height);

      const center = { x: width / 2, y: height / 2 };
      const maxRadius = width / 2; // Assuming square canvas for calculation

      // Smooth scale
      const targetScale = 1.0 + Math.min(Math.max(rawAudioLevel * 8.0, 0.0), 2.5);
      // Simple lerp: current = current + (target - current) * 0.1
      smoothedScaleRef.current += (targetScale - smoothedScaleRef.current) * 0.1;
      const scale = smoothedScaleRef.current;

      // Derived values
      const activityBoost = (scale - 1.0) * 1.5;
      const currentRotation = baseRotation + activityBoost * 4;
      const waveTime = baseRotation; // reusing rotation value as time factor since it's time-based
      const intensity = Math.min(Math.max((scale - 1.0) * 1.5, 0.0), 1.0);

      // Helper for colors
      const getStrokeStyle = (alpha: number) => `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`;
      const getFillStyle = (alpha: number) => `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`;

      // --- Core Reactor ---
      const baseRadius = maxRadius * 0.25;
      const currentRadius = baseRadius * (0.7 + scale * 0.3);

      // Glow
      ctx.save();
      ctx.filter = 'blur(20px)'; // Reduced blur for web performance? Dart had 35. Start with 20.
      ctx.fillStyle = getFillStyle(0.2 + 0.6 * intensity);
      ctx.beginPath();
      ctx.arc(center.x, center.y, currentRadius * 2.0, 0, 2 * Math.PI);
      ctx.fill();

      ctx.fillStyle = getFillStyle(0.3 + 0.5 * intensity);
      ctx.beginPath();
      ctx.arc(center.x, center.y, currentRadius * 1.2, 0, 2 * Math.PI);
      ctx.fill();
      ctx.restore();

      // Inner solid core ring
      ctx.strokeStyle = getStrokeStyle(0.95);
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.arc(center.x, center.y, baseRadius * 0.6, 0, 2 * Math.PI);
      ctx.stroke();

      // Rotating Triangle
      ctx.save();
      ctx.translate(center.x, center.y);
      ctx.rotate(-currentRotation * 1.5);
      drawPoly(ctx, 3, baseRadius * 0.5, 2, getStrokeStyle(1.0)); // Paint not fully specified in snippet, assume solid
      ctx.restore();

      // --- Middle Rings ---
      ctx.save();
      ctx.translate(center.x, center.y);
      const midRingRadius = maxRadius * 0.5 + intensity * 25;

      // Outer wave layer
      ctx.save();
      ctx.rotate(currentRotation * 0.8);
      drawWavyRing(
        ctx,
        midRingRadius * 1.15,
        96,
        rawAudioLevel * 6 * 5,
        // Scaling amplitude (Dart was audioLevel*6 but pixels might differ).
        // Dart visualizer size 300. Web visualizer 90px?
        // Wait, tile size is 90px in closed state, but might be larger.
        // We should scale amplitude relative to radius maybe?
        // Dart code used constant factors with size 300.
        // Let's use proportional factors.
        // audioLevel*6 is roughly 6 pixels if audio is 1.0.
        // 6/150 (radius) = 4%.
        waveTime * 5,
        4,
        1.5,
        getStrokeStyle(0.4 + 0.3 * intensity)
      );
      ctx.restore();

      // Mid wave layer
      ctx.save();
      ctx.rotate(currentRotation * -0.6);
      drawWavyRing(
        ctx,
        midRingRadius,
        96,
        rawAudioLevel * 8 * 5,
        waveTime * 8,
        5,
        2,
        getStrokeStyle(0.6 + 0.3 * intensity)
      );
      ctx.restore();

      // Inner wave layer
      ctx.save();
      ctx.rotate(currentRotation * 1.2);
      drawWavyRing(
        ctx,
        midRingRadius * 0.85,
        96,
        rawAudioLevel * 7 * 5,
        waveTime * 12,
        6,
        1.8,
        getStrokeStyle(0.5 + 0.4 * intensity)
      );
      ctx.restore();

      // Innermost wave
      ctx.save();
      ctx.rotate(currentRotation * -1.5);
      drawWavyRing(
        ctx,
        midRingRadius * 0.7,
        80,
        rawAudioLevel * 5 * 5,
        waveTime * 15,
        7,
        1.2,
        getStrokeStyle(0.4 + 0.3 * intensity)
      );
      ctx.restore();

      ctx.restore(); // End Middle Rings translate

      // --- Outer Interface ---
      const outerRadius = maxRadius * 0.85;

      ctx.save();
      ctx.translate(center.x, center.y);

      // Outer wave layers
      ctx.save();
      ctx.rotate(currentRotation * 0.1);
      drawWavyRing(
        ctx,
        outerRadius,
        120,
        rawAudioLevel * 4 * 5,
        waveTime * 4,
        3,
        1.5,
        getStrokeStyle(0.5 + 0.2 * intensity)
      );
      ctx.restore();

      // Outer thin wavy line
      ctx.save();
      ctx.rotate(currentRotation * -0.15);
      drawWavyRing(
        ctx,
        maxRadius * 0.92,
        100,
        rawAudioLevel * 3 * 5,
        waveTime * -3,
        4,
        1,
        getStrokeStyle(0.35 + 0.2 * intensity)
      );
      ctx.restore();

      // Tick marks
      ctx.save();
      ctx.rotate(-currentRotation * 0.2);
      drawTickMarks(ctx, maxRadius * 0.95, 48, 1, getStrokeStyle(0.4 + 0.2 * intensity));
      ctx.restore();

      ctx.restore(); // End Outer Interface translate
    },
    [smoothedScaleRef]
  );

  const drawPoly = (
    ctx: CanvasRenderingContext2D,
    sides: number,
    radius: number,
    lineWidth: number,
    strokeStyle: string
  ) => {
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    const angleStep = (Math.PI * 2) / sides;
    ctx.moveTo(radius * Math.cos(0), radius * Math.sin(0));
    for (let i = 1; i <= sides; i++) {
      ctx.lineTo(radius * Math.cos(angleStep * i), radius * Math.sin(angleStep * i));
    }
    ctx.closePath();
    ctx.stroke();
  };

  const drawWavyRing = (
    ctx: CanvasRenderingContext2D,
    radius: number,
    points: number,
    amplitude: number,
    time: number,
    waves: number,
    lineWidth: number,
    strokeStyle: string
  ) => {
    // Check for unreasonably small radius to prevent weird artifacts
    if (radius <= 0) return;

    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();

    const angleStep = (2 * Math.PI) / points;

    // We'll iterate one extra point to close the loop smoothly if we relied on lineTo,
    // but since we compute i=0 and i=points (which is 2PI), they should match.
    for (let i = 0; i <= points; i++) {
      const angle = i * angleStep;
      const wave = Math.sin(angle * waves + time) * amplitude;
      const r = radius + wave;
      const x = r * Math.cos(angle);
      const y = r * Math.sin(angle);
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.closePath();
    ctx.stroke();
  };

  const drawTickMarks = (
    ctx: CanvasRenderingContext2D,
    radius: number,
    count: number,
    lineWidth: number,
    strokeStyle: string
  ) => {
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = lineWidth;
    const step = (2 * Math.PI) / count;
    for (let i = 0; i < count; i++) {
      const angle = i * step;
      const innerR = radius - 5;
      const outerR = radius;

      ctx.beginPath();
      ctx.moveTo(innerR * Math.cos(angle), innerR * Math.sin(angle));
      ctx.lineTo(outerR * Math.cos(angle), outerR * Math.sin(angle));
      ctx.stroke();
    }
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Handle High DPI displays
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.scale(dpr, dpr);

    const animate = (time: number) => {
      const seconds = time / 1000;
      rotationRef.current = ((seconds % 10) / 10) * 2 * Math.PI;

      const rawAudioLevel = trackRef?.participant?.audioLevel ?? 0.0;

      draw(ctx, rect.width, rect.height, rawAudioLevel, rotationRef.current, rgb);

      requestRef.current = requestAnimationFrame(animate);
    };

    requestRef.current = requestAnimationFrame(animate);

    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [trackRef, draw, rgb]);

  return <canvas ref={canvasRef} className={className} style={{ width: '100%', height: '100%' }} />;
}
