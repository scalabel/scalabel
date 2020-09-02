import { GetStateFunc } from "../../src/common/simple_store"
import { IdType } from "../../src/types/state"
import { findNewLabelsFromState } from "../util/state"

/**
 * Collect the states from the current state
 */
export class LabelCollector extends Array<IdType> {
  /** access the state */
  private readonly _getState: GetStateFunc

  /**
   * Constructor
   *
   * @param getState
   */
  constructor(getState: GetStateFunc) {
    super()
    this._getState = getState
  }

  /** Collect the latest tracks from the state */
  public collect(): number {
    const state = this._getState()
    const trackIds = findNewLabelsFromState(state, state.user.select.item, this)
    this.push(...trackIds)
    return this.length
  }
}
