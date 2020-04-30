import { IdType } from 'aws-sdk/clients/workdocs'
import { getStateFunc } from '../../../js/common/simple_store'
import { findNewTracksFromState } from './util'

/**
 * Collect the states from the current state
 */
export class TrackCollector extends Array<IdType> {

  /** access the state */
  private _getState: getStateFunc

  constructor (getState: getStateFunc) {
    super()
    this._getState = getState
  }

  /** Collect the latest tracks from the state */
  public collect () {
    const trackIds = findNewTracksFromState(this._getState(), this)
    this.push(...trackIds)
  }
}
