import _ from 'lodash'
import { Track } from './track'
/**
 * Class for linear interpolating polygon's
 */
export class CustomLabel2DTrack extends Track {
  constructor () {
    super()
  }

  /** Linear interpolate shapes at index */
  protected linearInterpolateIndex (
    itemIndex: number,
    previousIndex: number
  ): void {
    this._shapes[itemIndex][0].type = this._shapes[previousIndex][0].type
    this._shapes[itemIndex][0].shape =
      _.cloneDeep(this._shapes[previousIndex][0].shape)
  }
}
