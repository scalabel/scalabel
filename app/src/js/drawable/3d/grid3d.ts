import * as THREE from 'three'
import { ShapeTypeName } from '../../common/types'
import { Plane3DType, ShapeType } from '../../functional/types'
import { Vector3D } from '../../math/vector3d'
import Label3D from './label3d'
import { Shape3D } from './shape3d'

/**
 * ThreeJS class for rendering grid
 */
export class Grid3D extends Shape3D {
  /** grid lines */
  private _lines: THREE.GridHelper

  constructor (label: Label3D) {
    super(label)
    this._lines = new THREE.GridHelper(1, 6, 0xffffff, 0xffffff)
    this._lines.rotation.x = -Math.PI / 2
    this.add(this._lines)
    this.scale.x = 6
    this.scale.y = 6
  }

  /** Get lines object */
  public get lines (): THREE.GridHelper {
    return this._lines
  }

  /** Get shape type name */
  public get typeName () {
    return ShapeTypeName.GRID
  }

  /** Get center */
  public get center (): Vector3D {
    return (new Vector3D()).fromThree(this.position)
  }

  /** Set center */
  public set center (center: Vector3D) {
    this.position.copy(center.toThree())
  }

  /** Get normal */
  public get normal (): Vector3D {
    const normal = new THREE.Vector3(0, 0, 1)
    this.getWorldDirection(normal)
    return (new Vector3D()).fromThree(normal)
  }

  /**
   * Add to scene for rendering
   * @param scene
   */
  public render (scene: THREE.Scene): void {
    scene.add(this)
  }

  /** Do not highlight plane for now */
  public setHighlighted (intersection?: THREE.Intersection) {
    if (intersection && intersection.object === this._lines) {
      { (this._lines.material as THREE.LineBasicMaterial).color.set(0xff0000) }
      (this._lines.material as THREE.LineBasicMaterial).needsUpdate = true
    } else {
      { (this._lines.material as THREE.LineBasicMaterial).color.set(0xffffff) }
      (this._lines.material as THREE.LineBasicMaterial).needsUpdate = true
    }
  }

  /**
   * Object representation
   */
  public toState (): ShapeType {
    return{
      center: (new Vector3D()).fromThree(this.position).toState(),
      orientation:
        (new Vector3D()).fromThree(this.rotation.toVector3()).toState()
    }
  }

  /**
   * Override ThreeJS raycast
   * @param raycaster
   * @param intersects
   */
  public raycast (
    raycaster: THREE.Raycaster,
    intersects: THREE.Intersection[]
  ) {
    if (this.label.selected || this.label.anyChildSelected()) {
      const ray = raycaster.ray
      const normal = new THREE.Vector3(0, 0, 1)
      normal.applyEuler(this.rotation)
      const plane = new THREE.Plane()
      plane.setFromNormalAndCoplanarPoint(normal, this.position)
      const target = new THREE.Vector3()
      const intersection = ray.intersectPlane(plane, target)
      if (intersection) {
        // const worldToPlane = new THREE.Matrix4()
        // worldToPlane.getInverse(this.matrixWorld)
        // const intersectionPlane = new THREE.Vector3()
        // intersectionPlane.copy(intersection)
        // intersectionPlane.applyMatrix4(worldToPlane)
        // if (
        //   intersectionPlane.x <= 0.5 &&
        //   intersectionPlane.x >= -0.5 &&
        //   intersectionPlane.y >= -0.5 &&
        //   intersectionPlane.y >= -0.5
        // ) {
        //   const difference = new THREE.Vector3()
        //   difference.copy(intersection)
        //   difference.sub(ray.origin)
        //   const distance = difference.length()
        //   if (distance < raycaster.far && distance > raycaster.near) {
        //     intersects.push({
        //       distance,
        //       point: intersection,
        //       object: this._lines
        //     })
        //   }
        // }
        const difference = new THREE.Vector3()
        difference.copy(intersection)
        difference.sub(ray.origin)
        const distance = difference.length()
        if (distance < raycaster.far && distance > raycaster.near) {
          intersects.push({
            distance,
            point: intersection,
            object: this._lines
          })
        }
      }
    } else {
      this._lines.raycast(raycaster, intersects)
    }

    for (const child of this.children) {
      child.raycast(raycaster, intersects)
    }
  }

  /** update parameters */
  public updateState (
    shape: ShapeType, id: number, _activeCamera?: THREE.Camera
  ) {
    super.updateState(shape, id)
    const newShape = shape as Plane3DType
    this.position.copy(
      (new Vector3D()).fromState(newShape.center).toThree()
    )
    this.rotation.setFromVector3(
      (new Vector3D()).fromState(newShape.orientation).toThree()
    )
  }
}
