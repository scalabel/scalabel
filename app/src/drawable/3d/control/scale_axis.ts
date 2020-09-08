import * as THREE from "three"

import Label3D from "../label3d"
import { ControlUnit } from "./controller"

/**
 * Unit for scaling object in certain direction
 */
export class ScaleAxis extends THREE.Group implements ControlUnit {
  /** axis to scale */
  private readonly _axis: "x" | "y" | "z"
  /** direction for scaling */
  private readonly _direction: THREE.Vector3
  /** line */
  private readonly _line: THREE.Line
  /** guideline */
  private readonly _guideline: THREE.Line
  /** box */
  private readonly _box: THREE.Mesh
  /** side length */
  private readonly _sideLength: number

  /**
   * Constructor
   *
   * @param axis
   * @param negate
   * @param color
   * @param sideLength
   */
  constructor(
    axis: "x" | "y" | "z",
    negate: boolean,
    color: number,
    sideLength: number = 0.4
  ) {
    super()
    this._axis = axis
    this._direction = new THREE.Vector3()
    switch (this._axis) {
      case "x":
        this._direction.x = 1
        break
      case "y":
        this._direction.y = 1
        break
      case "z":
        this._direction.z = 1
        break
    }

    if (negate) {
      this._direction.multiplyScalar(-1)
    }

    const lineGeometry = new THREE.BufferGeometry()
    lineGeometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute([0, 0, 0, 0, 0, 1], 3)
    )
    this._line = new THREE.Line(
      lineGeometry,
      new THREE.LineBasicMaterial({ color, transparent: true })
    )
    this.add(this._line)

    const guidelineGeometry = new THREE.BufferGeometry()
    guidelineGeometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute([0, 0, -10, 0, 0, 10], 3)
    )
    this._guideline = new THREE.Line(
      guidelineGeometry,
      new THREE.LineBasicMaterial({ color, transparent: true })
    )
    this._guideline.scale.set(1, 1, 1)

    this._box = new THREE.Mesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshBasicMaterial({ color, transparent: true })
    )
    this.add(this._box)

    this._sideLength = sideLength
    this._line.scale.set(1, 1, 1 - this._sideLength)

    this._box.scale.set(this._sideLength, this._sideLength, this._sideLength)
    this._box.position.z = 1

    const quaternion = new THREE.Quaternion()
    quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), this._direction)
    this.applyQuaternion(quaternion)

    this.setHighlighted()
    this.layers.enableAll()
    this._line.layers.enableAll()
    this._box.layers.enableAll()
  }

  /**
   * get update vectors: [translation, rotation, scale, new intersection]
   *
   * @param oldIntersection
   * @param newProjection
   * @param dragPlane
   * @param labels
   * @param bounds
   * @param local
   */
  public transform(
    oldIntersection: THREE.Vector3,
    newProjection: THREE.Ray,
    dragPlane: THREE.Plane,
    labels: Label3D[],
    bounds: THREE.Box3,
    local: boolean
  ): THREE.Vector3 {
    const direction = new THREE.Vector3()
    direction.copy(this._direction)

    const worldDirection = new THREE.Vector3()
    worldDirection.copy(this._direction)

    if (this.parent !== null) {
      const worldQuaternion = new THREE.Quaternion()
      this.parent.getWorldQuaternion(worldQuaternion)
      worldDirection.applyQuaternion(worldQuaternion)
    }

    const translationCoplanar = new THREE.Vector3()
    translationCoplanar.crossVectors(dragPlane.normal, worldDirection)
    const translationNormal = new THREE.Vector3()
    translationNormal.crossVectors(translationCoplanar, worldDirection)
    const translationPlane = new THREE.Plane()
    translationPlane.setFromNormalAndCoplanarPoint(
      translationNormal,
      oldIntersection
    )

    const newIntersection = new THREE.Vector3()
    newProjection.intersectPlane(translationPlane, newIntersection)

    const mouseDelta = new THREE.Vector3()
    mouseDelta.copy(newIntersection)
    mouseDelta.sub(oldIntersection)

    const projectionLength = mouseDelta.dot(worldDirection)
    const worldDelta = new THREE.Vector3()
    worldDelta.copy(worldDirection)
    worldDelta.multiplyScalar(projectionLength)

    const nextIntersection = new THREE.Vector3()
    nextIntersection.copy(oldIntersection)
    nextIntersection.add(worldDelta)

    const scaleDelta = new THREE.Vector3()
    scaleDelta.copy(this._direction)
    scaleDelta.x = Math.abs(scaleDelta.x)
    scaleDelta.y = Math.abs(scaleDelta.y)
    scaleDelta.z = Math.abs(scaleDelta.z)
    scaleDelta.multiplyScalar(projectionLength)

    const dimensions = new THREE.Vector3()
    dimensions.copy(bounds.max)
    dimensions.sub(bounds.min)

    const scaleFactor = new THREE.Vector3()
    scaleFactor.copy(dimensions)
    scaleFactor.add(scaleDelta)
    scaleFactor.divide(dimensions)
    scaleFactor.x = Math.abs(scaleFactor.x)
    scaleFactor.y = Math.abs(scaleFactor.y)
    scaleFactor.z = Math.abs(scaleFactor.z)

    const center = new THREE.Vector3()
    bounds.getCenter(center)

    const anchor = new THREE.Vector3()
    anchor.copy(direction)
    anchor.multiply(dimensions)
    anchor.divideScalar(-2.0)
    anchor.add(center)

    if (local) {
      anchor.sub(labels[0].center)
      anchor.applyQuaternion(labels[0].orientation)
      anchor.add(labels[0].center)
    }

    for (const label of labels) {
      label.scale(scaleFactor, anchor, local)
    }

    return nextIntersection
  }

  /**
   * Set highlighted
   *
   * @param object
   * @param intersection
   */
  public setHighlighted(intersection?: THREE.Intersection): boolean {
    ;(this._box.material as THREE.Material).needsUpdate = true
    ;(this._line.material as THREE.Material).needsUpdate = true

    if (
      intersection !== undefined &&
      (intersection.object === this ||
        intersection.object === this._line ||
        intersection.object === this._box)
    ) {
      ;(this._box.material as THREE.Material).opacity = 0.9
      ;(this._line.material as THREE.Material).opacity = 0.9

      this.add(this._guideline)
      return true
    } else {
      ;(this._box.material as THREE.Material).opacity = 0.65
      ;(this._line.material as THREE.Material).opacity = 0.65

      this.remove(this._guideline)
      return false
    }
  }

  /**
   * Set faded when another object is highlighted
   */
  public setFaded(): void {
    ;(this._box.material as THREE.Material).needsUpdate = true
    ;(this._line.material as THREE.Material).needsUpdate = true
    ;(this._box.material as THREE.Material).opacity = 0.25
    ;(this._line.material as THREE.Material).opacity = 0.25
  }

  /**
   * Override ThreeJS raycast to intersect with box
   *
   * @param raycaster
   * @param intersects
   */
  public raycast(
    raycaster: THREE.Raycaster,
    intersects: THREE.Intersection[]
  ): void {
    this._line.raycast(raycaster, intersects)
    this._box.raycast(raycaster, intersects)
  }

  /**
   * Update scale according to world scale
   *
   * @param worldScale
   */
  public updateScale(worldScale: THREE.Vector3): void {
    if (this.parent !== null) {
      const direction = new THREE.Vector3()
      direction.copy(this._direction)
      // Direction.applyQuaternion(worldQuaternion.inverse())

      const newScale = (Math.abs(direction.dot(worldScale)) * 3) / 4
      this._line.scale.set(1, 1, newScale)
      this._box.position.z = newScale
    }
  }
}
