import { IdType } from "aws-sdk/clients/workdocs"

import { GetStateFunc } from "../../src/common/simple_store"
import { findNewTracksFromState } from "../util/state"

/**
 * Collect the states from the current state
 */
export class TrackCollector extends Array<IdType> {
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
  public collect(): void {
    const trackIds = findNewTracksFromState(this._getState(), this)
    this.push(...trackIds)
  }
}
