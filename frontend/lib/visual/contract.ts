/** The first slice permits only this non-sensitive, non-conversational caption. */
export const VISUAL_PROMPT = 'Briefly describe what is visible in this camera view.';

export function isVisualPrompt(value: unknown): value is string {
  return value === VISUAL_PROMPT;
}
