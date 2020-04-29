import { ActionType } from '../action/types'
import { State } from '../functional/types'

export type getStateFunc = () => State
export type dispatchFunc = (actoin: ActionType) => void

/**
 * Simple store wrapper class
 */
export class SimpleStore {
  /** get state function */
  public getState: getStateFunc
  /** dispatch action function */
  public dispatch: dispatchFunc

  constructor (getState: getStateFunc, dispatch: dispatchFunc) {
    this.getState = getState
    this.dispatch = dispatch
  }

  /**
   * Get the standalone state accessor function
   */
  public getter (): getStateFunc {
    return this.getState.bind(this)
  }

  /**
   * Get the standalone state dispatch function
   */
  public dispatcher (): dispatchFunc {
    return this.dispatch.bind(this)
  }
}
