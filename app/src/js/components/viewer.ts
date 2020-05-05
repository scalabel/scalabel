import { ReduxState } from '../common/configure_store'
import { shouldCanvasFreeze } from '../functional/selector'
import { State } from '../functional/types'
import { Component } from './component'

export interface DrawableProps {
  /** Whether the canvas should freeze */
  shouldFreeze: boolean
}

export const mapStateToDrawableProps = (
  state: ReduxState): DrawableProps => {
  return {
    shouldFreeze: shouldCanvasFreeze(state)
  }
}

/**
 * Abstract class for Canvas
 */
export abstract class DrawableCanvas<
  Props extends DrawableProps> extends Component<Props> {
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
    return this.props.shouldFreeze
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
