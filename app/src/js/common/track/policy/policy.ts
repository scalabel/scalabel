import { TrackPolicyType } from '../../types'
import { Track } from '../track'

/** Abstract parent class for all track policies */
export class TrackPolicy {
  /** Associated track */
  protected _track: Track

  constructor (track: Track) {
    this._track = track
  }

  /** Get policy type */
  public get type (): TrackPolicyType {
    return TrackPolicyType.NONE
  }

  /** Update track when labels/shapes are changed */
  public update (_itemIndex: number): void {
    return
  }

  /** Get next manual index from some index */
  protected getNearestManualIndices (index: number): [number, number] {
    const itemIndices = this._track.indices.sort((a, b) => a - b)
    const itemArrayUpdatedIndex = itemIndices.indexOf(index)

    if (itemArrayUpdatedIndex < 0) {
      return [-1, -1]
    }

    let lastManualIndex = -1
    for (let i = itemArrayUpdatedIndex - 1; i >= 0; i -= 1) {
      const validIndex = itemIndices[i]
      const label = this._track.getLabel(validIndex)
      if (label && label.manual) {
        lastManualIndex = validIndex
        break
      }
    }

    let nextManualIndex = -1
    for (let i = itemArrayUpdatedIndex + 1; i < itemIndices.length; i += 1) {
      const validIndex = itemIndices[i]
      const label = this._track.getLabel(validIndex)
      if (label && label.manual) {
        nextManualIndex = validIndex
        break
      }
    }

    return [lastManualIndex, nextManualIndex]
  }
}
