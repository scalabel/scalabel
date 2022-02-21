import { TrackType } from "../types/state"

/**
 * Check if tracks overlap in items
 *
 * @param tracks
 */
export function tracksOverlapping(tracks: TrackType[]): boolean {
  const itemIndices = new Set<number>()
  for (const track of tracks) {
    const trackIndices = [...new Set(Object.keys(track.labels))]
    for (const key of trackIndices) {
      const index = Number(key)
      if (itemIndices.has(index)) {
        return true
      }
      itemIndices.add(index)
    }
  }
  return false
}
