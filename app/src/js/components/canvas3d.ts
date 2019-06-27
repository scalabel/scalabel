import { Canvas } from './canvas';

/**
 * Abstract class for 3d canvas
 */
export abstract class Canvas3d<Props> extends Canvas<Props> {
  protected constructor(props: Readonly<Props>) {
    super(props);
  }
}
