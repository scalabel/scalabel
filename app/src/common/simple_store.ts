import { ActionType } from "../types/action"
import { State } from "../types/state"

export type GetStateFunc = () => State
export type DispatchFunc = (actoin: ActionType) => void

/**
 * Simple store wrapper class
 */
export class SimpleStore {
  /** get state function */
  public getState: GetStateFunc
  /** dispatch action function */
  public dispatch: DispatchFunc

  /**
   * Constructor
   *
   * @param getState
   * @param dispatch
   */
  constructor(getState: GetStateFunc, dispatch: DispatchFunc) {
    this.getState = getState
    this.dispatch = dispatch
  }

  /**
   * Get the standalone state accessor function
   */
  public getter(): GetStateFunc {
    return this.getState.bind(this)
  }

  /**
   * Get the standalone state dispatch function
   */
  public dispatcher(): DispatchFunc {
    return this.dispatch.bind(this)
  }
}
