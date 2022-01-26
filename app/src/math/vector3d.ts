import * as THREE from "three"

import { Vector3Type } from "../types/state"
import { Vector } from "./vector"

/** 2D vector */
export class Vector3D extends Vector {
  /**
   * Constructor
   *
   * @param x
   * @param y
   * @param z
   */
  constructor(x: number = 0, y: number = 0, z: number = 0) {
    super(3)
    this[0] = x
    this[1] = y
    this[2] = z
  }

  /** get x */
  public get x(): number {
    return this[0]
  }

  /** set x */
  public set x(v: number) {
    this[0] = v
  }

  /** get y */
  public get y(): number {
    return this[1]
  }

  /** set y */
  public set y(v: number) {
    this[1] = v
  }

  /** get z */
  public get z(): number {
    return this[2]
  }

  /** set z */
  public set z(v: number) {
    this[2] = v
  }

  /**
   * convert from the vector in THREE
   *
   * @param v
   */
  public fromThree(v: THREE.Vector3): this {
    this[0] = v.x
    this[1] = v.y
    this[2] = v.z
    return this
  }

  /** Convert to ThreeJS Vector3 */
  public toThree(): THREE.Vector3 {
    return new THREE.Vector3(this[0], this[1], this[2])
  }

  /** Convert to ThreeJS Euler */
  public toThreeEuler(): THREE.Euler {
    return new THREE.Euler(this[0], this[1], this[2])
  }

  /** convert to raw 3D type */
  public toState(): Vector3Type {
    return { x: this[0], y: this[1], z: this[2] }
  }

  /**
   * Set value from a raw 3d vector
   *
   * @param v
   */
  public fromState(v: Vector3Type): this {
    this[0] = v.x
    this[1] = v.y
    this[2] = v.z
    return this
  }

  /**
   * Copy from other Vector3D
   *
   * @param v
   */
  public copy(v: Vector3D): this {
    this[0] = v[0]
    this[1] = v[1]
    this[2] = v[2]
    return this
  }

  /**
   * Multiply all values by a scalar
   *
   * @param s
   */
  public multiplyScalar(s: number): void {
    this[0] *= s
    this[1] *= s
    this[2] *= s
  }

  /**
   * Divide all values by a scalar
   *
   * @param s
   */
  public divideScalar(s: number): void {
    this[0] /= s
    this[1] /= s
    this[2] /= s
  }

  /**
   * Cross product with another vector
   *
   * @param v
   */
  public cross(v: Vector3D): this {
    const x = this[1] * v[2] - this[2] * v[1]
    const y = this[2] * v[0] - this[0] * v[2]
    const z = this[0] * v[1] - this[1] * v[0]
    this[0] = x
    this[1] = y
    this[2] = z
    return this
  }

  /**
   * Distance to another vector
   *
   * @param v
   */
  public distanceTo(v: Vector3D): number {
    const x = this[0] - v[0]
    const y = this[1] - v[1]
    const z = this[2] - v[2]
    return Math.sqrt(x * x + y * y + z * z)
  }

  /** Magnitude of vector */
  public magnitude(): number {
    return Math.sqrt(this[0] * this[0] + this[1] * this[1] + this[2] * this[2])
  }

  /** Calculate unit vector */
  public normalize(): this {
    const m = this.magnitude()
    if (m !== 0) {
      this.divideScalar(m)
    }
    return this
  }

  /** Absolute value */
  public abs(): this {
    this[0] = Math.abs(this[0])
    this[1] = Math.abs(this[1])
    this[2] = Math.abs(this[2])
    return this
  }
}
