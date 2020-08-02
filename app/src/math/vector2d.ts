import { Vector } from './vector'

/** 2D vector */
export class Vector2D extends Vector {
  constructor (x: number = 0, y: number = 0) {
    super(2)
    this[0] = x
    this[1] = y
  }

  /** get x */
  public get x (): number {
    return this[0]
  }

  /** set x */
  public set x (v: number) {
    this[0] = v
  }

  /** get y */
  public get y (): number {
    return this[1]
  }

  /** set x */
  public set y (v: number) {
    this[1] = v
  }

  /** area of the vector */
  public area (): number {
    return this.prod()
  }
}
