import * as THREE from 'three'
import { ControlUnit } from './controller'

/**
 * Single rotation ring
 */
export class RotationRing extends THREE.Mesh implements ControlUnit {
  /** normal */
  private _normal: THREE.Vector3
  /** guideline */
  private _guideline: THREE.Line

  constructor (normal: THREE.Vector3, color: number) {
    super(
      new THREE.TorusGeometry(1, .02, 32, 24),
      new THREE.MeshBasicMaterial({ color, transparent: true })
   )
    this._normal = normal

    const lineGeometry = new THREE.BufferGeometry()
    lineGeometry.addAttribute(
      'position',
      new THREE.Float32BufferAttribute([ 0, 0, -10, 0, 0, 10 ], 3)
    )
    this._guideline = new THREE.Line(
      lineGeometry,
      new THREE.LineBasicMaterial({ color, transparent: true })
    )
    this._guideline.scale.set(1, 1, 1)

    const quaternion = new THREE.Quaternion()
    quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), this._normal)
    this.quaternion.copy(quaternion)

    this.setHighlighted()
  }

  /**
   * Set highlighted
   * @param object
   */
  public setHighlighted (intersection ?: THREE.Intersection): boolean {
    { (this.material as THREE.Material).needsUpdate = true }
    if (intersection && intersection.object === this) {
      { (this.material as THREE.Material).opacity = 0.9 }
      this.add(this._guideline)
      return true
    } else {
      { (this.material as THREE.Material).opacity = 0.65 }
      this.remove(this._guideline)
      return false
    }
  }

  /**
   * Set faded when another object is highlighted
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
    dragPlane: THREE.Plane,
    object?: THREE.Object3D
  ): [THREE.Vector3, THREE.Quaternion, THREE.Vector3, THREE.Vector3] {

    const newIntersection = new THREE.Vector3()
    newProjection.intersectPlane(dragPlane, newIntersection)

    const normal = new THREE.Vector3()
    normal.copy(this._normal)

    if (object) {
      normal.applyQuaternion(object.quaternion)
    }

    const delta = new THREE.Vector3()
    delta.copy(newIntersection)
    delta.sub(oldIntersection)

    const dragDirection = new THREE.Vector3()
    dragDirection.crossVectors(dragPlane.normal, normal)
    dragDirection.normalize()

    // toLocal is matrix that coverts world to object frame
    const toLocal = new THREE.Matrix4()
    if (this.parent) {
      toLocal.getInverse(this.parent.matrixWorld)
    }

    // local projection is the projection in the object frame
    const localProjection = new THREE.Ray()
    localProjection.copy(newProjection).applyMatrix4(toLocal)

    // torusPos is the position of the center of the torus, (0, 0, 0)
    const torusPos = new THREE.Vector3()
    // torusPlane is aligned with the torus
    const torusPlane = new THREE.Plane()
    torusPlane.setFromNormalAndCoplanarPoint(this._normal, torusPos)

    // torus intersection is where the projection intersects the torus plane
    const torusIntersection = new THREE.Vector3()
    localProjection.intersectPlane(torusPlane, torusIntersection)

    // camPos is camera position in object frame
    const camPos = new THREE.Vector3()
    camPos.copy(newProjection.origin).applyMatrix4(toLocal)

    const camToTorusCenterDistance = camPos.distanceTo(torusPos)
    const camToTorusIntersectionDistance = camPos.distanceTo(torusIntersection)

    // if the intersection is further away than the center, reverse direction
    if (camToTorusIntersectionDistance > camToTorusCenterDistance) {
      dragDirection.negate()
    }

    const dragAmount = delta.dot(dragDirection)
    const rotation = new THREE.Quaternion()
    rotation.setFromAxisAngle(normal, dragAmount)

    return [
      new THREE.Vector3(0, 0, 0),
      rotation,
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
      const newScale = Math.max(1, .45 * Math.min(
        Math.abs(worldScale.x),
        Math.abs(worldScale.y),
        Math.abs(worldScale.z)
      ))
      this.scale.set(newScale, newScale, newScale)
    }
  }
}
