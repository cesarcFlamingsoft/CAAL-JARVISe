/**
 * The two artifacts hand recognition runs on, pinned and served by this
 * origin.
 *
 * The runtime is the MediaPipe Tasks Vision WebAssembly build, copied
 * verbatim from the pinned npm package into public/hands/wasm (loader and
 * binary for both the SIMD and the non-SIMD variant, whichever the resolver
 * picks for the browser). The model is the MediaPipe hand_landmarker task
 * file, float16, version 1 (Apache-2.0), taken from
 * storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/
 * and kept in public/hands/models, pinned by size and SHA-256.
 *
 * Nothing is loaded from anywhere else: the resolver is pointed at
 * HAND_RUNTIME_PATH and the model at HAND_MODEL_PATH, both same-origin paths.
 * The runtime version matters beyond features: releases from 1.0.0 onwards
 * send usage metrics to Google, and 0.10.35 is the newest whose bundle names
 * no server at all. The guard test checks the served files against these
 * pins and the installed bundle against that promise, so bump with care.
 */
export const HAND_RUNTIME_PACKAGE = '@mediapipe/tasks-vision';
export const HAND_RUNTIME_VERSION = '0.10.35';
/** Directory under public/ holding the runtime loader and binary. */
export const HAND_RUNTIME_PATH = '/hands/wasm';
export const HAND_RUNTIME_FILES: readonly string[] = [
  'vision_wasm_internal.js',
  'vision_wasm_internal.wasm',
  'vision_wasm_nosimd_internal.js',
  'vision_wasm_nosimd_internal.wasm',
];
export const HAND_MODEL_PATH = '/hands/models/hand_landmarker.task';
export const HAND_MODEL_BYTES = 7819105;
export const HAND_MODEL_SHA256 = 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1';
