'use client';

import { useEffect, useState } from 'react';

export type VisualizationType = 'jarvis' | 'soundbars';

const STORAGE_KEY = 'visualization_type';

/**
 * Hook to fetch and track the current visualization type setting.
 * Uses localStorage for immediate updates and listens for custom events.
 */
export function useVisualizationType(): VisualizationType {
  const [visualizationType, setVisualizationType] = useState<VisualizationType>(() => {
    // Initialize from localStorage if available
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored === 'jarvis' || stored === 'soundbars') {
        return stored;
      }
    }
    return 'jarvis';
  });

  useEffect(() => {
    // Fetch from API on mount
    fetch('/api/settings')
      .then((res) => res.json())
      .then((data) => {
        const type = data.settings?.visualization_type;
        if (type === 'jarvis' || type === 'soundbars') {
          setVisualizationType(type);
          localStorage.setItem(STORAGE_KEY, type);
        }
      })
      .catch(() => {
        console.warn('Failed to fetch visualization type, using default');
      });

    // Listen for settings changes
    const handleSettingsChange = (event: Event) => {
      const customEvent = event as CustomEvent<{ visualization_type: VisualizationType }>;
      const newType = customEvent.detail.visualization_type;
      if (newType === 'jarvis' || newType === 'soundbars') {
        setVisualizationType(newType);
        localStorage.setItem(STORAGE_KEY, newType);
      }
    };

    window.addEventListener('settings-updated', handleSettingsChange);

    return () => {
      window.removeEventListener('settings-updated', handleSettingsChange);
    };
  }, []);

  return visualizationType;
}
