import { Canvas } from './canvas'

/**
 * Abstract class for 2d canvas
 */
export abstract class Canvas2d<Props> extends Canvas<Props> {
  protected constructor (props: Readonly<Props>) {
    super(props)
  }
}
