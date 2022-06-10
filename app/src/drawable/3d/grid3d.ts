import * as THREE from "three"

import { ShapeTypeName } from "../../const/common"
import { makePlane } from "../../functional/states"
import { Vector3D } from "../../math/vector3d"
import { IdType, Plane3DType, ShapeType } from "../../types/state"
import Label3D from "./label3d"
import { Shape3D } from "./shape3d"

/**
 * ThreeJS class for rendering grid
 */
export class Grid3D extends Shape3D {
  /** grid lines */
  private readonly _lines: THREE.GridHelper
  /** internal shape state */
  private _planeShape: Plane3DType

  /**
   * Constructor
   *
   * @param label
   */
  constructor(label: Label3D) {
    super(label)
    this._lines = new THREE.GridHelper(
      1,
      6,
      new THREE.Color().fromArray(label.color),
      new THREE.Color().fromArray(label.color)
    )
    this._lines.rotation.x = -Math.PI / 2
    this.add(this._lines)
    this.scale.x = 3
    this.scale.y = 3
    this._planeShape = makePlane()
  }

  /** get the shape id */
  public get shapeId(): IdType {
    return this._planeShape.id
  }

  /** Get lines object */
  public get lines(): THREE.GridHelper {
    return this._lines
  }

  /** Get shape type name */
  public get typeName(): string {
    return ShapeTypeName.GRID
  }

  /** Get center */
  public get center(): Vector3D {
    return new Vector3D().fromThree(this.position)
  }

  /** Set center */
  public set center(center: Vector3D) {
    this.position.copy(center.toThree())
  }

  /** Get normal */
  public get normal(): Vector3D {
    const normal = new THREE.Vector3(0, 0, 1)
    this.getWorldDirection(normal)
    return new Vector3D().fromThree(normal)
  }

  /**
   * Add to scene for rendering
   *
   * @param scene
   */
  public render(scene: THREE.Scene): void {
    scene.add(this)
  }

  /**
   * Do not highlight plane for now
   *
   * @param intersection
   */
  public setHighlighted(intersection?: THREE.Intersection): void {
    if (intersection !== undefined && intersection.object === this._lines) {
      ;(this._lines.material as THREE.LineBasicMaterial).color.set(0xff0000)
      ;(this._lines.material as THREE.LineBasicMaterial).needsUpdate = true
    } else {
      ;(this._lines.material as THREE.LineBasicMaterial).color.set(
        new THREE.Color().fromArray(this.label.color)
      )
      ;(this._lines.material as THREE.LineBasicMaterial).needsUpdate = true
    }
  }

  /**
   * Object representation
   */
  public toState(): ShapeType {
    const plane = this._planeShape
    plane.center = new Vector3D().fromThree(this.center.toThree()).toState()
    plane.orientation = new Vector3D()
      .fromThree(this.rotation.toVector3())
      .toState()
    return plane
  }

  /**
   * Override ThreeJS raycast
   *
   * @param raycaster
   * @param intersects
   */
  public raycast(
    raycaster: THREE.Raycaster,
    intersects: THREE.Intersection[]
  ): void {
    const ray = raycaster.ray
    const normal = new THREE.Vector3(0, 0, 1)
    normal.applyEuler(this.rotation)
    const plane = new THREE.Plane()
    plane.setFromNormalAndCoplanarPoint(normal, this.position)
    const target = new THREE.Vector3()
    const intersection = ray.intersectPlane(plane, target)
    if (intersection !== null) {
      // Check for intersection within bounds
      const worldToPlane = new THREE.Matrix4()
      worldToPlane.copy(this.matrixWorld.clone().invert())
      const intersectionPlane = new THREE.Vector3()
      intersectionPlane.copy(intersection)
      intersectionPlane.applyMatrix4(worldToPlane)
      if (
        intersectionPlane.x <= 0.5 &&
        intersectionPlane.x >= -0.5 &&
        intersectionPlane.y >= -0.5 &&
        intersectionPlane.y <= 0.5
      ) {
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
    }

    for (const child of this.children) {
      child.raycast(raycaster, intersects)
    }
  }

  /**
   * update parameters
   *
   * @param shape
   * @param id
   * @param _activeCamera
   */
  public updateState(
    shape: ShapeType,
    id: IdType,
    _activeCamera?: THREE.Camera
  ): void {
    super.updateState(shape, id)
    const newShape = shape as Plane3DType
    this.position.copy(new Vector3D().fromState(newShape.center).toThree())
    this.rotation.copy(
      new Vector3D().fromState(newShape.orientation).toThreeEuler()
    )
    // Also update the _planeShape
    this._planeShape = newShape
  }

  /**
   * Set visibility for viewer
   *
   * @param viewerId
   * @param v
   */
  public setVisible(viewerId: number, v: boolean = true): void {
    super.setVisible(viewerId, v)
    if (v) {
      this._lines.layers.enable(viewerId)
    } else {
      this._lines.layers.disable(viewerId)
    }
  }
}
