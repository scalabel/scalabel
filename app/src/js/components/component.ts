import * as React from 'react'
import Session from '../common/session'
import { State as StateType } from '../functional/types'

/**
 * Root class of our components
 */
export abstract class Component<Props> extends
  React.Component<Props, StateType> {
    /** flag to check if a component is mounted, preventing possible memory
     * leak from rendering un-mounted component
     */
  private _isMounted = false
  /**
   * General constructor
   * @param props: component props
   */
  constructor (props: Readonly<Props>) {
    super(props)
    Session.subscribe(this.onStateUpdated.bind(this))
    this.state = Session.getState()
  }

  /**
   * after mounting, set flag to allow for rendering and state updates
   */
  public componentDidMount () {
    this._isMounted = true
  }

  /**
   * after unmounting, set flag so no state updates are possible
   */
  public componentWillUnmount () {
    this._isMounted = false
  }

  /**
   * Callback for updated state
   * When the global state is updated, this component should be updated
   */
  private onStateUpdated (): void {
    if (this._isMounted) {
      this.setState(Session.getState())
    }
  }
}
