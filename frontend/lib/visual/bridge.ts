/** Ephemeral commands only. No transcript, storage, model, or camera ownership. */
export type VisionAnalysis = (signal: AbortSignal) => Promise<string>;
export type VisionReanalysis = (question: string, signal: AbortSignal) => Promise<string>;
export interface VisionBinding {
  user: string;
  room: string;
  participant: string;
  agent: string;
  epoch: string;
}
export class VisionCommandHandler {
  private binding: VisionBinding | null;
  private analyze: VisionAnalysis | null;
  private reanalyze: VisionReanalysis | null;
  private send: ((packet: Record<string, unknown>) => Promise<void>) | null;
  private sequence = 0;
  private request: AbortController | null = null;
  constructor(
    binding: VisionBinding,
    analyze: VisionAnalysis | null,
    send: (packet: Record<string, unknown>) => Promise<void>,
    reanalyze: VisionReanalysis | null = null
  ) {
    this.binding = binding;
    this.analyze = analyze;
    this.reanalyze = reanalyze;
    this.send = send;
  }
  close() {
    this.request?.abort();
    this.request = null;
    this.binding = null;
    this.analyze = null;
    this.reanalyze = null;
    this.send = null;
  }
  async receive(value: unknown, sender: string) {
    const b = this.binding;
    if (!b || sender !== b.agent || !value || typeof value !== 'object') return;
    const p = value as Record<string, unknown>;
    if (
      p.action === 'vision.cancel' &&
      p.user === b.user &&
      p.room === b.room &&
      p.epoch === b.epoch &&
      Number.isSafeInteger(p.seq) &&
      Number(p.seq) >= this.sequence
    ) {
      this.sequence = Number(p.seq);
      this.request?.abort();
      this.request = null;
      return;
    }
    const isAnalyze = p.action === 'vision.analyze';
    const isReanalyze = p.action === 'vision.reanalyze';
    const expectedKeys = isAnalyze
      ? 'action,epoch,expires,room,seq,user'
      : 'action,epoch,expires,question,room,seq,user';
    if (
      Object.keys(p).sort().join(',') !== expectedKeys ||
      (!isAnalyze && !isReanalyze) ||
      p.user !== b.user ||
      p.room !== b.room ||
      p.epoch !== b.epoch ||
      !Number.isSafeInteger(p.seq) ||
      Number(p.seq) <= this.sequence ||
      typeof p.expires !== 'number' ||
      p.expires <= Date.now() ||
      p.expires > Date.now() + 31000
    )
      return;
    if (
      isReanalyze &&
      (typeof p.question !== 'string' ||
        p.question.length < 1 ||
        p.question.length > 240 ||
        /[\x00-\x1f\x7f]/.test(p.question))
    )
      return;
    this.sequence = Number(p.seq);
    this.request?.abort();
    const controller = new AbortController();
    this.request = controller;
    const timer = setTimeout(() => controller.abort(), p.expires - Date.now());
    let description = '';
    try {
      description = isAnalyze
        ? this.analyze
          ? await this.analyze(controller.signal)
          : ''
        : this.reanalyze
          ? await this.reanalyze(p.question as string, controller.signal)
          : '';
      if (description.length > 1200 || /[\x00-\x1f\x7f]/.test(description)) description = '';
      if (!controller.signal.aborted && this.binding === b) {
        await this.send?.({ ...p, action: 'vision.result', description });
      }
    } catch (cause) {
      if (cause instanceof Error && cause.message === 'not_signed_in') {
        await this.send?.({
          action: 'vision.close',
          user: b.user,
          room: b.room,
          epoch: b.epoch,
        }).catch(() => undefined);
        this.close();
        return;
      }
      if (!controller.signal.aborted && this.binding === b) {
        await this.send?.({ ...p, action: 'vision.result', description: '' }).catch(
          () => undefined
        );
      }
    } finally {
      clearTimeout(timer);
      description = '';
      if (this.request === controller) this.request = null;
    }
  }
}
