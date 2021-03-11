import * as THREE from "three"

import Label3D from "../label3d"
import { ControlUnit } from "./controller"

/**
 * ThreeJS object used for moving parent object along certain axis
 */
export class TranslationAxis extends THREE.Group implements ControlUnit {
  /** Translation direction (180 degree symmetric) */
  private readonly _direction: THREE.Vector3
  /** cone size */
  private readonly _coneSize: number
  /** line */
  private readonly _line: THREE.Line
  /** guideline */
  private readonly _guideline: THREE.Line
  /** cone */
  private readonly _cone: THREE.Mesh

  /**
   * Constructor
   *
   * @param direction
   * @param color
   * @param coneSize
   */
  constructor(direction: THREE.Vector3, color: number, coneSize: number = 0.3) {
    super()
    this._coneSize = coneSize

    this._direction = new THREE.Vector3()
    this._direction.copy(direction)
    this._direction.normalize()

    const lineGeometry = new THREE.BufferGeometry()
    lineGeometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute([0, 0, 0, 0, 1, 0], 3)
    )
    this._line = new THREE.Line(
      lineGeometry,
      new THREE.LineBasicMaterial({ color, transparent: true })
    )
    this.add(this._line)

    const guidelineGeometry = new THREE.BufferGeometry()
    guidelineGeometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute([0, -10, 0, 0, 10, 0], 3)
    )
    this._guideline = new THREE.Line(
      guidelineGeometry,
      new THREE.LineBasicMaterial({ color, transparent: true })
    )
    this._guideline.scale.set(1, 1, 1)

    this._cone = new THREE.Mesh(
      new THREE.ConeGeometry(1, 1.2),
      new THREE.MeshBasicMaterial({ color, transparent: true })
    )
    this.add(this._cone)

    this._line.scale.set(1, 0.75 - this._coneSize, 1)

    this._cone.scale.set(this._coneSize, this._coneSize, this._coneSize)
    this._cone.position.y = 0.75

    const quaternion = new THREE.Quaternion()
    quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), this._direction)
    this.applyQuaternion(quaternion)

    this.setHighlighted()
    this.layers.enableAll()
    this._line.layers.enableAll()
    this._cone.layers.enableAll()
  }

  /**
   * Mouse movement while mouse down on box (from raycast)
   *
   * @param oldIntersection
   * @param newProjection
   * @param dragPlane
   * @param labels
   * @param _bounds
   * @param local
   */
  public transform(
    oldIntersection: THREE.Vector3,
    newProjection: THREE.Ray,
    dragPlane: THREE.Plane,
    labels: Label3D[],
    _bounds: THREE.Box3,
    local: boolean
  ): THREE.Vector3 {
    const direction = new THREE.Vector3()
    direction.copy(this._direction)

    if (local) {
      direction.applyQuaternion(labels[0].orientation)
    }

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

    const projectionLength = mouseDelta.dot(direction)
    const positionDelta = new THREE.Vector3()
    positionDelta.copy(direction)
    positionDelta.multiplyScalar(projectionLength)

    const intersectionDelta = new THREE.Vector3()
    intersectionDelta.copy(worldDirection)
    intersectionDelta.multiplyScalar(projectionLength)

    const nextIntersection = new THREE.Vector3()
    nextIntersection.copy(oldIntersection)
    nextIntersection.add(intersectionDelta)

    for (const label of labels) {
      label.translate(positionDelta)
    }

    return newIntersection
  }

  /**
   * Set highlighted
   *
   * @param object
   * @param intersection
   */
  public setHighlighted(intersection?: THREE.Intersection): boolean {
    ;(this._line.material as THREE.Material).needsUpdate = true
    ;(this._cone.material as THREE.Material).needsUpdate = true

    if (
      intersection !== undefined &&
      (intersection.object === this ||
        intersection.object === this._line ||
        intersection.object === this._cone)
    ) {
      ;(this._line.material as THREE.Material).opacity = 0.9
      ;(this._cone.material as THREE.Material).opacity = 0.9

      this.add(this._guideline)
      return true
    } else {
      ;(this._line.material as THREE.Material).opacity = 0.65
      ;(this._cone.material as THREE.Material).opacity = 0.65

      this.remove(this._guideline)
      return false
    }
  }

  /**
   * Set faded when another object is highlighted
   */
  public setFaded(): void {
    ;(this._line.material as THREE.Material).needsUpdate = true
    ;(this._cone.material as THREE.Material).needsUpdate = true
    ;(this._line.material as THREE.Material).opacity = 0.25
    ;(this._cone.material as THREE.Material).opacity = 0.25
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
    this._cone.raycast(raycaster, intersects)
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

      this._line.scale.set(1, newScale, 1)

      this._cone.position.y = newScale
    }
  }
}
