/**
 * What the hand pipeline is allowed to do to the Monitor surface.
 *
 * The dashboard implements this; the hand layer calls it. Points are viewport
 * pixels. Every method is safe to call at frame rate and when nothing is
 * under the hand.
 */
import type { Point } from './gesture';

export interface HandTarget {
  id: string;
  title: string;
  /** False when the target exists but cannot be moved right now (stacked layout, for one). */
  movable: boolean;
}

export interface HandSurfaceController {
  /** The open hand is over this point; highlight whatever is there. */
  hover(point: Point): HandTarget | null;
  /** No hand is tracked any more. */
  hoverEnd(): void;
  /** A fist closed here: select what is under it. */
  select(point: Point): HandTarget | null;
  /** The fist was held: start moving the target, if it can move. */
  grab(point: Point): HandTarget | null;
  /** The grabbed target follows the hand. */
  move(point: Point): void;
  /** The hand opened: settle the layout. */
  release(): void;
  /** The hand vanished or control was switched off: put things back. */
  cancel(): void;
}
