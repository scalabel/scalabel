import { TrackPolicyType } from '../../../types'
import { Track } from '../../track'
import { TrackPolicy } from '../policy'

/** Parent class for linear interpolation policy types */
export abstract class LinearInterpolationPolicy extends TrackPolicy {
  constructor (track: Track) {
    super(track)
  }

  /** Get policy type */
  public get type () {
    return TrackPolicyType.LINEAR_INTERPOLATION
  }

  /** Update */
  public update (itemIndex: number): void {
    const indices = this._track.indices
    const currentManualIndices = this.getNearestManualIndices(itemIndex)

    // Backward
    if (currentManualIndices[0] >= 0) {
      this.interpolate(currentManualIndices[0], itemIndex)
    } else {
      this.copyShapes(indices[0], itemIndex, itemIndex)
    }

    // Forward
    if (currentManualIndices[1] >= 0) {
      this.interpolate(itemIndex, currentManualIndices[1])
    } else {
      this.copyShapes(itemIndex, indices[indices.length - 1], itemIndex)
    }
  }

  /** Copy shapes */
  protected copyShapes (start: number, end: number, source: number) {
    if (!this._track.getLabel(source)) {
      return
    }
    console.log('copy', start, end, source)

    const sourceShapes = this._track.getShapes(source)
    for (let i = start + 1; i < end; i++) {
      const label = this._track.getLabel(i)
      if (!label) {
        continue
      }

      const shapes = this._track.getShapes(i)
      const newShapes = []
      for (let shapeIndex = 0; shapeIndex < sourceShapes.length; shapeIndex++) {
        newShapes.push({
          ...sourceShapes[shapeIndex],
          id: shapes[shapeIndex].id
        })
      }

      this._track.setShapes(i, newShapes)
      this._track.addUpdatedIndex(i)
    }
  }

  /** Linear interpolation */
  protected interpolate (start: number, end: number) {
    this.calculateDeltas(start, end)
    for (let i = start + 1; i < end; i++) {
      if (this._track.getLabel(i)) {
        this.interpolateIndex(start, i)
        this._track.addUpdatedIndex(i)
      }
    }
  }

  /** Calculate deltas and save in instance variables */
  protected abstract calculateDeltas (start: number, end: number): void

  /** Function for linear interpolating at certain index */
  protected abstract interpolateIndex (start: number, index: number): void
}
