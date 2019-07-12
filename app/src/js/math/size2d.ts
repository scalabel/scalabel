import { Vector } from './vector'

/** 2D size */
export class Size2D extends Vector {
  constructor (w: number = 0, h: number = 0) {
    super(2)
    this[0] = w
    this[1] = h
  }

  /** get x */
  public get width (): number {
    return this[0]
  }

  /** set x */
  public set width (v: number) {
    this[0] = v
  }

  /** get y */
  public get height (): number {
    return this[1]
  }

  /** set x */
  public set height (v: number) {
    this[1] = v
  }
}
