/** general vector class */
export class Vector extends Array<number> {
  /**
   * Constructor
   * @param dim
   */
  constructor(dim: number = 0) {
    super(dim)
  }

  /**
   * Set the values of this vector
   * @param {number[]} args: list of numbers to fill
   */
  public set(...args: number[]): this {
    for (let i = 0; i < this.length && i < args.length; i += 1) {
      this[i] = args[i]
    }
    return this
  }

  /** clone the current vector */
  public clone(): this {
    const newVector = Object.create(this)
    Object.assign(newVector, Array.from(this))
    return newVector
  }

  /** negate the values */
  public negate(): this {
    this.forEach((v, i, arr) => (arr[i] = -v))
    return this
  }

  /** scale the whole vector */
  public scale(s: number): this {
    this.forEach((v, i, arr) => (arr[i] = v * s))
    return this
  }

  /** add a number or vector */
  public add(s: number | Vector): this {
    if (typeof s === "number") {
      this.forEach((v, i, arr) => (arr[i] = v + s))
    } else {
      this.forEach((v, i, arr) => (arr[i] = v + s[i]))
    }
    return this
  }

  /** subtract a number or vector */
  public subtract(s: number | Vector): this {
    if (typeof s === "number") {
      this.forEach((v, i, arr) => (arr[i] = v - s))
    } else {
      this.forEach((v, i, arr) => (arr[i] = v - s[i]))
    }
    return this
  }

  /** dot product with another vector */
  public dot(vector: Vector): this {
    this.forEach((v, i, arr) => (arr[i] = v * vector[i]))
    return this
  }

  /** calculate the absolute values of the vector elements */
  public abs(): this {
    this.forEach((v, i, arr) => (arr[i] = Math.abs(v)))
    return this
  }

  /** product of the elements */
  public prod(): number {
    let p = 1
    this.forEach((v) => (p *= v))
    return p
  }
}
