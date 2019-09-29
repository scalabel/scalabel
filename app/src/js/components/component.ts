import * as React from 'react'
import Session from '../common/session'
import { State as StateType } from '../functional/types'

/**
 * Root class of our components
 */
export abstract class Component<Props> extends
  React.Component<Props, StateType> {
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
   * Callback for updated state
   * When the global state is updated, this component should be updated
   */
  private onStateUpdated (): void {
    this.setState(Session.getState())
  }
}
