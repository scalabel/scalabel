import { nanoid } from 'nanoid'

/**
 * Generate unique 16-bit ID with alphabet
 * 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz-
 * ~5 million years is needed in order to have a 1% probability of at
 * least one collision.
 */
export function uid (): string {
  return nanoid(16)
}
