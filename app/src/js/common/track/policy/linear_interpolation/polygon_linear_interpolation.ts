import _ from 'lodash'
import { Track } from '../../track'
import { LinearInterpolationPolicy } from './linear_interpolation_policy'

/**
 * Class for linear interpolating polygon's
 */
export class PolygonLinearInterpolationPolicy extends
  LinearInterpolationPolicy {
  constructor (track: Track) {
    super(track)
  }

  /** No deltas needed */
  protected calculateDeltas () {
    return
  }

  /** Linear interpolate shapes at index */
  protected interpolateIndex (
    start: number,
    index: number
  ): void {
    this.copyShapes(index - 1, index + 1, start)
  }
}
