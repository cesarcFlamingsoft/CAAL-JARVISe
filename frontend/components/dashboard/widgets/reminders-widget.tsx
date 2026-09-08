'use client';

import { Button } from '@/components/livekit/button';
import type { CapabilitiesState } from '@/hooks/useDashboardCapabilities';
import {
  WidgetBlocked,
  WidgetEmpty,
  WidgetError,
  WidgetLoading,
  WidgetSignIn,
} from '../widget-notice';

const PROVIDER_LABEL = { local: 'Local reminders', apple: 'Apple Reminders' } as const;

interface RemindersWidgetProps {
  capabilities: CapabilitiesState & { reload: () => void };
  passwordLogin: boolean;
  onOpenSettings: () => void;
}

export function RemindersWidget({
  capabilities,
  passwordLogin,
  onOpenSettings,
}: RemindersWidgetProps) {
  if (capabilities.status === 'loading') {
    return <WidgetLoading label="Checking reminder sources…" />;
  }
  if (capabilities.status === 'unauthorized') {
    return <WidgetSignIn passwordLogin={passwordLogin} what="your reminders" />;
  }
  if (capabilities.status === 'error') {
    return (
      <WidgetError
        title="Reminders are unavailable"
        detail="The backend did not answer."
        onRetry={capabilities.reload}
      />
    );
  }

  const { reminders, alarms } = capabilities.data;
  if (!reminders.configured || !reminders.provider) {
    return (
      <WidgetEmpty
        title="No reminder provider"
        detail="Choose local or Apple Reminders under Settings → Integrations."
        action={
          <Button variant="outline" size="sm" onClick={onOpenSettings}>
            Open settings
          </Button>
        }
      />
    );
  }

  return (
    <div className="space-y-3">
      <ul className="flex flex-wrap gap-1.5">
        <li className="bg-muted rounded-full px-2.5 py-1 text-xs font-medium">
          {PROVIDER_LABEL[reminders.provider]}
        </li>
        <li className="bg-muted text-muted-foreground rounded-full px-2.5 py-1 text-xs">
          Alarms {alarms.enabled ? 'on' : 'off'}
        </li>
      </ul>

      <WidgetBlocked
        title="Open reminders"
        detail="JARVIS can create and list reminders by voice, but the list is not exposed to the dashboard yet."
        endpoint={reminders.listEndpoint}
      />
    </div>
  );
}
