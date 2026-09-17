/** Reply-language selection: a personal choice, parsed the same way on both sides. */
export const languages = ['auto', 'en', 'es'] as const;
export type Language = (typeof languages)[number];

export interface LanguageView {
  language: Language;
  source: 'personal' | 'default';
  applies_to: 'new_sessions';
}

export const languageLabels: Record<Language, string> = {
  auto: 'Automatic — follow how I speak',
  en: 'English',
  es: 'Español',
};

export function parseLanguageView(value: unknown): LanguageView | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const v = value as Record<string, unknown>;
  if (
    !languages.includes(v.language as Language) ||
    !['personal', 'default'].includes(v.source as string) ||
    v.applies_to !== 'new_sessions'
  )
    return null;
  return {
    language: v.language as Language,
    source: v.source as LanguageView['source'],
    applies_to: 'new_sessions',
  };
}
