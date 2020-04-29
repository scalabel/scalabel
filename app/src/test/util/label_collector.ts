import { getStateFunc } from '../../js/common/simple_store'
import { IdType } from '../../js/functional/types'
import { findNewLabelsFromState } from '../server/util/util'

/**
 * Collect the states from the current state
 */
export class LabelCollector extends Array<IdType> {

  /** access the state */
  private _getState: getStateFunc

  constructor (getState: getStateFunc) {
    super()
    this._getState = getState
  }

  /** Collect the latest tracks from the state */
  public collect () {
    const state = this._getState()
    const trackIds = findNewLabelsFromState(
      state, state.user.select.item, this)
    this.push(...trackIds)
  }
}
