import * as THREE from "three"

import Label3D from "../label3d"
import { ControlUnit } from "./controller"

/**
 * Translate along plane
 */
export class TranslationPlane extends THREE.Mesh implements ControlUnit {
  /** normal direction */
  private readonly _normal: THREE.Vector3

  /**
   * Constructor
   *
   * @param normal
   * @param color
   */
  constructor(normal: THREE.Vector3, color: number) {
    super(
      new THREE.PlaneGeometry(0.5, 0.5),
      new THREE.MeshBasicMaterial({
        color,
        side: THREE.DoubleSide,
        transparent: true
      })
    )
    this._normal = new THREE.Vector3()
    this._normal.copy(normal)
    this._normal.normalize()

    const quaternion = new THREE.Quaternion()
    quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), this._normal)
    this.quaternion.copy(quaternion)
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

      return true
    } else {
      ;(this.material as THREE.Material).opacity = 0.65

      return false
    }
  }

  /**
   * Set not highlighted when another object is highlighted
   *
   * @param object
   */
  public setFaded(): void {
    ;(this.material as THREE.Material).needsUpdate = true
    ;(this.material as THREE.Material).opacity = 0.25
  }

  /**
   * Get translation delta
   *
   * @param oldIntersection
   * @param newProjection
   * @param dragPlane
   * @param _dragPlane
   * @param labels
   * @param _bounds
   * @param local
   */
  public transform(
    oldIntersection: THREE.Vector3,
    newProjection: THREE.Ray,
    _dragPlane: THREE.Plane,
    labels: Label3D[],
    _bounds: THREE.Box3,
    local: boolean
  ): THREE.Vector3 {
    const normal = new THREE.Vector3()
    normal.copy(this._normal)

    const toLocal = new THREE.Matrix4()
    if (this.parent !== null) {
      toLocal.getInverse(this.parent.matrixWorld)
    }

    const localIntersection = new THREE.Vector3()
    localIntersection.copy(oldIntersection)
    localIntersection.applyMatrix4(toLocal)

    const localProjection = new THREE.Ray()
    localProjection.copy(newProjection)
    localProjection.applyMatrix4(toLocal)

    const plane = new THREE.Plane()
    plane.setFromNormalAndCoplanarPoint(normal, localIntersection)
    const newIntersection = new THREE.Vector3()
    localProjection.intersectPlane(plane, newIntersection)

    const delta = new THREE.Vector3()
    delta.copy(newIntersection)
    delta.sub(localIntersection)

    if (local) {
      delta.applyQuaternion(labels[0].orientation)
    }

    if (this.parent !== null) {
      newIntersection.applyMatrix4(this.parent.matrixWorld)
    }

    for (const label of labels) {
      label.translate(delta)
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
        2,
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
