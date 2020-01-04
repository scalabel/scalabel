import * as THREE from 'three'
import Label3D from '../label3d'

export interface ControlUnit extends THREE.Object3D {
  /** get update vectors: [translation, rotation, scale, new intersection] */
  transform: (
    oldIntersection: THREE.Vector3,
    newProjection: THREE.Ray,
    dragPlane: THREE.Plane,
    labels: Label3D[],
    bounds: THREE.Box3,
    local: boolean
  ) => THREE.Vector3
  /** set highlight */
  setHighlighted: (intersection ?: THREE.Intersection) => boolean
  /** set faded */
  setFaded: () => void
  /** Update scale according to world scale */
  updateScale: (worldScale: THREE.Vector3) => void
}

/**
 * Super class for all controllers
 */
export abstract class Controller extends THREE.Object3D {
  /** translation axes */
  protected _controlUnits: ControlUnit[]
  /** current axis being dragged */
  protected _highlightedUnit: ControlUnit | null
  /** local or world */
  protected _local: boolean
  /** original intersection point */
  protected _intersectionPoint: THREE.Vector3
  /** Plane of intersection point w/ camera direction */
  protected _dragPlane: THREE.Plane
  /** previous projection */
  protected _projection: THREE.Ray
  /** labels to transform */
  protected _labels: Label3D[]
  /** bounds of the labels */
  protected _bounds: THREE.Box3

  constructor (labels: Label3D[], bounds: THREE.Box3) {
    super()
    this._controlUnits = []
    this._local = true
    this._intersectionPoint = new THREE.Vector3()
    this._highlightedUnit = null
    this._dragPlane = new THREE.Plane()
    this._projection = new THREE.Ray()
    this._labels = labels
    this._bounds = bounds
  }

  /** Returns whether this is highlighted */
  public get highlighted (): boolean {
    return this._highlightedUnit !== null
  }

  /** highlight function */
  public setHighlighted (intersection?: THREE.Intersection) {
    this._highlightedUnit = null
    for (const axis of this._controlUnits) {
      if (axis.setHighlighted(intersection) && intersection) {
        this._highlightedUnit = axis
        this._intersectionPoint = intersection.point
        for (const nonAxis of this._controlUnits) {
          if (nonAxis !== axis) {
            nonAxis.setFaded()
          }
        }
        break
      }
    }
  }

  /** mouse down */
  public onMouseDown (camera: THREE.Camera) {
    if (this._highlightedUnit) {
      const normal = new THREE.Vector3()
      camera.getWorldDirection(normal)
      this._dragPlane.setFromNormalAndCoplanarPoint(
        normal,
        this._intersectionPoint
      )
      return true
    }
    return false
  }

  /** mouse move */
  public onMouseMove (projection: THREE.Ray) {
    if (this._highlightedUnit && this._dragPlane) {
      const newIntersection = this._highlightedUnit.transform(
        this._intersectionPoint,
        projection,
        this._dragPlane,
        this._labels,
        this._bounds,
        this._local
      )

      this._intersectionPoint.copy(newIntersection)
      this._projection.copy(projection)
      return true
    }
    this._projection.copy(projection)
    return false
  }

  /** mouse up */
  public onMouseUp () {
    return false
  }

  /** raycast */
  public raycast (
    raycaster: THREE.Raycaster, intersects: THREE.Intersection[]
  ) {
    for (const unit of this._controlUnits) {
      unit.raycast(raycaster, intersects)
    }
  }

  /** Return true if working in local frame */
  public get local (): boolean {
    return this._local
  }

  /** Toggle local/world */
  public toggleFrame () {
    if (this._labels.length === 1) {
      this._local = !this._local
    } else {
      this._local = false
    }
  }

  /** Update scales of control units */
  public updateScale (scale: THREE.Vector3) {
    for (const unit of this._controlUnits) {
      unit.updateScale(scale)
    }
  }
}
