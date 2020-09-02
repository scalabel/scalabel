import { customAlphabet } from "nanoid"

const ALPHABET =
  "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
const nanoid = customAlphabet(ALPHABET, 16)

/**
 * Generate unique 16-bit ID with alphabet
 * 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
 * ~4 million years is needed in order to have a 1% probability of at
 * least one collision.
 * -_ are removed from the default alphabet because the string with - or _ may
 * be treated as multiple words instead of 1 in some text editors
 */
export function uid(): string {
  return nanoid()
}
