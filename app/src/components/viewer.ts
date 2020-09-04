import { shouldCanvasFreeze } from "../functional/selector"
import { ReduxState } from "../types/redux"
import { State } from "../types/state"
import { Component } from "./component"

export interface DrawableProps {
  /** Whether the canvas should freeze */
  shouldFreeze: boolean
  /** Whether tracking is enabled */
  tracking: boolean
}

export const mapStateToDrawableProps = (state: ReduxState): DrawableProps => {
  return {
    shouldFreeze: shouldCanvasFreeze(state),
    tracking: state.present.task.config.tracking
  }
}

/**
 * Abstract class for Canvas
 */
export abstract class DrawableCanvas<
  Props extends DrawableProps
> extends Component<Props> {
  /**
   * General constructor
   *
   * @param props: component props
   * @param props
   */
  protected constructor(props: Readonly<Props>) {
    super(props)
  }

  /**
   * Execute when component state is updated
   */
  public componentDidUpdate(): void {
    this.updateState(this.state)
    this.redraw()
  }

  /**
   * Checks whether to freeze interface
   */
  public checkFreeze(): boolean {
    return this.props.shouldFreeze
  }

  /**
   * Redraw function for canvas
   * It should always fetch the current state from this.state
   * instead of Session
   */
  public abstract redraw(): boolean

  /**
   * notify state is updated
   */
  protected abstract updateState(state: State): void
}
