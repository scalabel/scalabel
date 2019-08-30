import * as THREE from 'three'
import { Vector3Type } from '../functional/types'
import { Vector } from './vector'

/** 2D vector */
export class Vector3D extends Vector {
  constructor (x: number = 0, y: number = 0, z: number = 0) {
    super(3)
    this[0] = x
    this[1] = y
    this[2] = z
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

  /** set y */
  public set y (v: number) {
    this[1] = v
  }

  /** get z */
  public get z (): number {
    return this[2]
  }

  /** set z */
  public set z (v: number) {
    this[2] = v
  }

  /** convert from the vector in THREE */
  public fromThree (v: THREE.Vector3): this {
    this[0] = v.x
    this[1] = v.y
    this[2] = v.z
    return this
  }

  /** Convert to ThreeJS Vector3 */
  public toThree (): THREE.Vector3 {
    return new THREE.Vector3(this[0], this[1], this[2])
  }

  /** Convert to ThreeJS Euler */
  public toThreeEuler (): THREE.Euler {
    return new THREE.Euler(this[0], this[1], this[2])
  }

  /** convert to raw 3D type */
  public toObject (): Vector3Type {
    return { x: this[0], y: this[1], z: this[2] }
  }

  /**
   * Set value from a raw 3d vector
   */
  public fromObject (v: Vector3Type): this {
    this[0] = v.x
    this[1] = v.y
    this[2] = v.z
    return this
  }

  /**
   * Copy from other Vector3D
   * @param v
   */
  public copy (v: Vector3D): this {
    this[0] = v[0]
    this[1] = v[1]
    this[2] = v[2]
    return this
  }

  /**
   * Multiply all values by a scalar
   */
  public multiplyScalar (s: number) {
    this[0] *= s
    this[1] *= s
    this[2] *= s
  }
}
