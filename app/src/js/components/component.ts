import * as React from 'react'
import Session from '../common/session'
import { State as StateType } from '../functional/types'

interface State {
  /** state of the session */
  session: StateType
}

/**
 * Root class of our components
 */
export abstract class Component<Props> extends React.Component<Props, State> {
  /**
   * General constructor
   * @param props: component props
   */
  constructor (props: Readonly<Props>) {
    super(props)
    Session.subscribe(this.onStateUpdated.bind(this))
    this.state = {
      session: Session.getState()
    }
  }

  /**
   * Callback for updated state
   * When the global state is updated, this component should be updated
   */
  private onStateUpdated (): void {
    this.setState({
      session: Session.getState()
    })
  }
}
