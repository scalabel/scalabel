import { Component } from './component';

/**
 * Abstract class for Canvas
 */
export abstract class Canvas<Props> extends Component<Props> {
  /**
   * General constructor
   * @param props: component props
   */
  protected constructor(props: Readonly<Props>) {
    super(props);
  }

  /**
   * Execute when component state is updated
   */
  public componentDidUpdate() {
    this.redraw();
  }

  /**
   * Redraw function for canvas
   * It should always fetch the current state from this.state
   * instead of Session
   */
  protected abstract redraw(): boolean;
}
