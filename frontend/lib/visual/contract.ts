/** The first slice permits only this non-sensitive, non-conversational caption. */
export const VISUAL_PROMPT =
  'Briefly describe what is visible in this camera view. Include a visible brand, model, or text only when clear; never guess.';

export function isVisualPrompt(value: unknown): value is string {
  return value === VISUAL_PROMPT;
}
