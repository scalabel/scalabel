import { ExtrinsicsType, IntrinsicsType, SensorType } from "../types/state"
import * as THREE from "three"
import { DataType } from "../const/common"

/**
 * Object representation of a sensor
 */
export class Sensor {
  /** id */
  private readonly _id: number
  /** name */
  private readonly _name: string
  /** type */
  private readonly _type: string
  /** intrinsic parameters */
  private readonly _intrinsics: IntrinsicsType | null
  /** extrinsic parameters */
  private readonly _extrinsics: ExtrinsicsType | null
  /** projection matrix */
  private _projectionMatrix: THREE.Matrix4 | null
  /** Transformation matrix */
  private _transformationMatrix: THREE.Matrix4 | null
  /** Inverse transformation matrix */
  private _inverseTransformationMatrix: THREE.Matrix4 | null
  /** up direction for sensor coordinates */
  private readonly _up: THREE.Vector3
  /** forward direction for sensor coordinates */
  private readonly _forward: THREE.Vector3
  /** Height above ground */
  private readonly _height: number | null

  /**
   * Constructor
   *
   * @param id
   * @param name
   * @param type
   * @param intrinsics
   * @param extrinsics
   * @param image
   * @param height
   */
  constructor(
    id: number,
    name: string,
    type: string,
    intrinsics?: IntrinsicsType,
    extrinsics?: ExtrinsicsType,
    image?: HTMLImageElement,
    height?: number
  ) {
    this._id = id
    this._name = name
    this._type = type
    this._intrinsics = intrinsics ?? null
    this._extrinsics = extrinsics ?? null
    this._projectionMatrix = null
    this._transformationMatrix = null
    this._inverseTransformationMatrix = null
    this._height = height ?? null

    if (image !== undefined) {
      this.calculateProjectionMatrix(image)
    }
    this.calculateTransformationMatrix()

    // Pass in as config variable later
    if (this._type === DataType.POINT_CLOUD) {
      this._up = new THREE.Vector3(0, 0, 1)
      this._forward = new THREE.Vector3(1, 0, 0)
    } else {
      this._up = new THREE.Vector3(0, -1, 0)
      this._forward = new THREE.Vector3(0, 0, 1)
    }
  }

  /**
   * Calculate projection matrix, based on sensor intrinsics
   *
   * @param image
   */
  private calculateProjectionMatrix(image: HTMLImageElement): void {
    if (this._intrinsics !== null) {
      const width = image.width
      const height = image.height
      this._projectionMatrix = new THREE.Matrix4()
      this._projectionMatrix.set(
        this._intrinsics.focalLength.x / width,
        0,
        this._intrinsics.focalCenter.x / width,
        0,
        0,
        this._intrinsics.focalLength.y / height,
        this._intrinsics.focalCenter.y / height,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0
      )
    }
  }

  /**
   * Calculate transformation matrix, based on sensor extrinsics
   */
  private calculateTransformationMatrix(): void {
    if (this._extrinsics !== null) {
      const matrix = new THREE.Matrix4().makeRotationFromEuler(
        new THREE.Euler(
          -this._extrinsics.rotation.x,
          -this._extrinsics.rotation.y,
          -this._extrinsics.rotation.z
        )
      )
      const position = new THREE.Vector3(
        this._extrinsics.translation.x,
        this._extrinsics.translation.y,
        this._extrinsics.translation.z
      )
      position.applyMatrix4(matrix).multiplyScalar(-1)
      matrix.setPosition(position)
      matrix.invert()
      this._transformationMatrix = matrix
      this._inverseTransformationMatrix = matrix.clone().invert()
    }
  }

  /**
   * Create instance from SensorType
   *
   * @param s
   * @param image
   */
  public static fromSensorType(
    s: SensorType,
    image?: HTMLImageElement
  ): Sensor {
    return new Sensor(
      s.id,
      s.name,
      s.type,
      s.intrinsics,
      s.extrinsics,
      image,
      s.height
    )
  }

  /**
   * Transform from global to sensor coordinates using extrinsics
   *
   * @param v
   */
  public transform(v: THREE.Vector3): THREE.Vector3 {
    if (this._transformationMatrix !== null) {
      return v.clone().applyMatrix4(this._transformationMatrix)
    } else {
      return v
    }
  }

  /**
   * Transform from sensor to global coordinates using extrinsics
   *
   * @param v
   */
  public inverseTransform(v: THREE.Vector3): THREE.Vector3 {
    if (this._inverseTransformationMatrix !== null) {
      return v.clone().applyMatrix4(this._inverseTransformationMatrix)
    } else {
      return v
    }
  }

  /**
   * Rotate vector with extrinsic rotation
   *
   * @param v
   */
  public rotate(v: THREE.Vector3): THREE.Vector3 {
    if (this._transformationMatrix !== null) {
      const rotationMatrix = new THREE.Matrix4().extractRotation(
        this._transformationMatrix
      )
      return v.clone().applyMatrix4(rotationMatrix)
    } else {
      return v
    }
  }

  /**
   * Inverse rotate vector with extrinsic rotation
   *
   * @param v
   */
  public inverseRotate(v: THREE.Vector3): THREE.Vector3 {
    if (this._inverseTransformationMatrix !== null) {
      const rotationMatrix = new THREE.Matrix4().extractRotation(
        this._inverseTransformationMatrix
      )
      return v.clone().applyMatrix4(rotationMatrix)
    } else {
      return v
    }
  }

  /**
   * Project point in sensor coordinates into image coordinates using intrinsics
   *
   * @param v
   */
  public project(v: THREE.Vector3): THREE.Vector3 {
    if (this._projectionMatrix !== null) {
      return v.clone().applyMatrix4(this._projectionMatrix)
    } else {
      return v
    }
  }

  /**
   * Up direction for sensor
   */
  public get up(): THREE.Vector3 {
    return this._up
  }

  /**
   * Forward direction for sensor
   */
  public get forward(): THREE.Vector3 {
    return this._forward
  }

  /**
   * Id of the sensor
   */
  public get id(): number {
    return this._id
  }

  /**
   * Name of the sensor
   */
  public get name(): string {
    return this._name
  }

  /**
   * Type of the sensor
   */
  public get type(): string {
    return this._type
  }

  /**
   * Does the sensor have extrinsic parameters
   */
  public hasExtrinsics(): boolean {
    return this._extrinsics !== null
  }

  /**
   * Does the sensor have intrinsic parameters
   */
  public hasIntrinsics(): boolean {
    return this._intrinsics !== null
  }

  /**
   * Height of sensor above ground
   */
  public get height(): number | null {
    return this._height
  }
}
