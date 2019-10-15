import * as THREE from 'three'
import { ControlUnit } from './controller'

/**
 * Translate along plane
 */
export class TranslationPlane extends THREE.Mesh
  implements ControlUnit {
  /** normal direction */
  private _normal: THREE.Vector3

  constructor (normal: THREE.Vector3, color: number) {
    super(
      new THREE.PlaneGeometry(0.5, 0.5),
      new THREE.MeshBasicMaterial({
        color, side: THREE.DoubleSide, transparent: true
      })
    )
    this._normal = new THREE.Vector3()
    this._normal.copy(normal)
    this._normal.normalize()

    const quaternion = new THREE.Quaternion()
    quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), this._normal)
    this.quaternion.copy(quaternion)
  }

  /**
   * Set highlighted
   * @param object
   */
  public setHighlighted (intersection ?: THREE.Intersection): boolean {
    { (this.material as THREE.Material).needsUpdate = true }
    if (intersection && intersection.object === this) {
      { (this.material as THREE.Material).opacity = 0.9 }
      return true
    } else {
      { (this.material as THREE.Material).opacity = 0.65 }
      return false
    }
  }

  /**
   * Set not highlighted when another object is highlighted
   * @param object
   */
  public setFaded (): void {
    { (this.material as THREE.Material).needsUpdate = true }
    { (this.material as THREE.Material).opacity = 0.25 }
  }

  /**
   * Get translation delta
   * @param oldIntersection
   * @param newProjection
   * @param dragPlane
   */
  public getDelta (
    oldIntersection: THREE.Vector3,
    newProjection: THREE.Ray,
    _dragPlane: THREE.Plane,
    object?: THREE.Object3D
  ): [THREE.Vector3, THREE.Quaternion, THREE.Vector3, THREE.Vector3] {
    const normal = new THREE.Vector3()
    normal.copy(this._normal)

    const toLocal = new THREE.Matrix4()
    if (this.parent) {
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

    if (object) {
      delta.applyQuaternion(object.quaternion)
    }

    if (this.parent) {
      newIntersection.applyMatrix4(this.parent.matrixWorld)
    }

    return [
      delta,
      new THREE.Quaternion(0, 0, 0, 1),
      new THREE.Vector3(),
      newIntersection
    ]
  }

  /**
   * Update scale according to world scale
   * @param worldScale
   */
  public updateScale (worldScale: THREE.Vector3) {
    if (this.parent) {
      const newScale = Math.max(2, Math.min(
        Math.abs(worldScale.x),
        Math.abs(worldScale.y),
        Math.abs(worldScale.z)
      ))
      this.scale.set(newScale, newScale, newScale)
    }
  }
}
