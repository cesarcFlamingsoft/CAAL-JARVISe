/** Browser-safe durable work projection. Shared by the BFF and browser. */
export const WORK_STATUS = {
  queued: { label: 'Queued for FRIDAY', detail: 'FRIDAY will start when a worker is free' },
  running: { label: 'FRIDAY is working', detail: 'I’m actively working on this now' },
  succeeded: { label: 'Completed', detail: null },
  failed: { label: 'Couldn’t complete', detail: null },
  cancelled: { label: 'Cancelled', detail: null },
  interrupted: { label: 'Interrupted — retry if needed', detail: null },
} as const;
export type WorkStatus = keyof typeof WORK_STATUS;
export interface WorkItem {
  title: string;
  status: WorkStatus;
  created_at: number;
  updated_at: number;
  started_at: number | null;
  finished_at: number | null;
}
export interface WorkFeed {
  generated_at: number;
  items: WorkItem[];
}
function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}
function timestamp(value: unknown): value is number {
  // Unix seconds within the browser Date range.
  return (
    typeof value === 'number' &&
    Number.isSafeInteger(value) &&
    value >= 0 &&
    value <= 8_640_000_000_000
  );
}
export function browserWork(value: unknown): WorkFeed | null {
  if (
    !record(value) ||
    !timestamp(value.generated_at) ||
    !Array.isArray(value.items) ||
    value.items.length > 12
  )
    return null;
  const items: WorkItem[] = [];
  for (const item of value.items) {
    if (
      !record(item) ||
      typeof item.title !== 'string' ||
      item.title.length > 240 ||
      !item.title.trim() ||
      Array.from(item.title).length > 120 ||
      typeof item.status !== 'string' ||
      !Object.hasOwn(WORK_STATUS, item.status) ||
      !timestamp(item.created_at) ||
      !timestamp(item.updated_at) ||
      !(item.started_at === null || timestamp(item.started_at)) ||
      !(item.finished_at === null || timestamp(item.finished_at))
    )
      return null;
    items.push({
      title: item.title,
      status: item.status as WorkStatus,
      created_at: item.created_at,
      updated_at: item.updated_at,
      started_at: item.started_at,
      finished_at: item.finished_at,
    });
  }
  return { generated_at: value.generated_at, items };
}
export function workRefreshInterval(value: unknown): number {
  return browserWork(value)?.items.some(
    (item) => item.status === 'queued' || item.status === 'running'
  )
    ? 15_000
    : 60_000;
}
