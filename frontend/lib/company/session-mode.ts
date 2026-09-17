/**
 * Asking for a company-private session, from the browser.
 *
 * A session's kind is decided once, when the room is created and the
 * room-bound principal is signed, and it never changes inside the session.
 * That is deliberate: a conversation that has already read a contract must not
 * be able to become a cloud-assisted conversation partway through. So entering
 * and leaving are both navigations, and both start a genuinely new session.
 *
 * This module only reads the request out of the URL. Whether it is granted is
 * decided twice more: the BFF signs it into the principal only for a verified
 * user, and the voice worker honours it only for the provisioned owner of the
 * company library.
 */

/** Where "Start a company session" goes. A fresh load, so a fresh room. */
export const COMPANY_SESSION_HREF = '/?company=1';

/** Where "Leave the company session" goes. A fresh load, so a fresh room. */
export const ORDINARY_SESSION_HREF = '/';

/** Whether this page load is asking for a company-private session. */
export function companySessionRequested(search: unknown): boolean {
  if (typeof search !== 'string' || search.length === 0) return false;
  try {
    return new URLSearchParams(search.startsWith('?') ? search.slice(1) : search).get('company') === '1';
  } catch {
    return false;
  }
}
