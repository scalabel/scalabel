import Session, { ConnectionStatus } from '../common/session'
import { State } from '../functional/types'
import { Component } from './component'

/**
 * Abstract class for Canvas
 */
export abstract class Viewer<Props> extends Component<Props> {
  /**
   * General constructor
   * @param props: component props
   */
  protected constructor (props: Readonly<Props>) {
    super(props)
  }

  /**
   * Execute when component state is updated
   */
  public componentDidUpdate () {
    this.updateState(this.state)
    this.redraw()
  }

  /**
   * Checks whether to freeze interface
   */
  public checkFreeze () {
    return Session.status === ConnectionStatus.RECONNECTING
  }

  /**
   * Redraw function for canvas
   * It should always fetch the current state from this.state
   * instead of Session
   */
  public abstract redraw (): boolean

  /**
   * notify state is updated
   */
  protected abstract updateState (state: State): void
}
