import * as THREE from "three"

import Label3D from "../label3d"
import { ControlUnit } from "./controller"

/**
 * Single rotation ring
 */
export class RotationRing extends THREE.Mesh implements ControlUnit {
  /** normal */
  private readonly _normal: THREE.Vector3
  /** guideline */
  private readonly _guideline: THREE.Line
  /** intersection point on highlight */
  private readonly _highlightIntersection: THREE.Vector3

  /**
   * Constructor
   *
   * @param normal
   * @param color
   */
  constructor(normal: THREE.Vector3, color: number) {
    super(
      new THREE.TorusGeometry(1, 0.07, 32, 24),
      new THREE.MeshBasicMaterial({ color, transparent: true })
    )
    this._normal = normal

    const lineGeometry = new THREE.BufferGeometry()
    lineGeometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute([0, 0, -10, 0, 0, 10], 3)
    )
    this._guideline = new THREE.Line(
      lineGeometry,
      new THREE.LineBasicMaterial({ color, transparent: true })
    )
    this._guideline.scale.set(1, 1, 1)

    const quaternion = new THREE.Quaternion()
    quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), this._normal)
    this.quaternion.copy(quaternion)

    this._highlightIntersection = new THREE.Vector3()
    this.setHighlighted()
    this.layers.enableAll()
  }

  /**
   * Set highlighted
   *
   * @param object
   * @param intersection
   */
  public setHighlighted(intersection?: THREE.Intersection): boolean {
    ;(this.material as THREE.Material).needsUpdate = true

    if (intersection !== undefined && intersection.object === this) {
      ;(this.material as THREE.Material).opacity = 0.9

      this.add(this._guideline)
      this._highlightIntersection.copy(intersection.point)
      return true
    } else {
      ;(this.material as THREE.Material).opacity = 0.65

      this.remove(this._guideline)
      return false
    }
  }

  /**
   * Set faded when another object is highlighted
   */
  public setFaded(): void {
    ;(this.material as THREE.Material).needsUpdate = true
    ;(this.material as THREE.Material).opacity = 0.25
  }

  /**
   * Translate input labels
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
    const newIntersection = new THREE.Vector3()
    newProjection.intersectPlane(dragPlane, newIntersection)

    const normal = new THREE.Vector3()
    normal.copy(this._normal)

    if (local) {
      normal.applyQuaternion(labels[0].orientation)
    }

    const delta = new THREE.Vector3()
    delta.copy(newIntersection)
    delta.sub(oldIntersection)

    const dragDirection = new THREE.Vector3()
    dragDirection.crossVectors(dragPlane.normal, normal)
    dragDirection.normalize()

    const centerToCamera = new THREE.Vector3()
    this.getWorldPosition(centerToCamera)
    centerToCamera.sub(newProjection.origin)

    const intersectionToCamera = new THREE.Vector3()
    intersectionToCamera.copy(this._highlightIntersection)
    intersectionToCamera.sub(newProjection.origin)

    // If the intersection is further away than the center, reverse direction
    if (intersectionToCamera.length() - centerToCamera.length() > 1e-3) {
      dragDirection.negate()
    }

    const dragAmount = delta.dot(dragDirection)
    const rotation = new THREE.Quaternion()
    rotation.setFromAxisAngle(normal, dragAmount)

    for (const label of labels) {
      label.rotate(rotation)
    }

    return newIntersection
  }

  /**
   * Update scale according to world scale
   *
   * @param worldScale
   */
  public updateScale(worldScale: THREE.Vector3): void {
    if (this.parent !== null) {
      const newScale = Math.max(
        1,
        0.45 *
          Math.min(
            Math.abs(worldScale.x),
            Math.abs(worldScale.y),
            Math.abs(worldScale.z)
          )
      )
      this.scale.set(newScale, newScale, newScale)
    }
  }
}
