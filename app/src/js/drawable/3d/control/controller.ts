import * as THREE from 'three'
import { projectionFromNDC } from '../../../view_config/point_cloud'
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

  /** mouse move */
  public drag (dx: number, dy: number, camera: THREE.Camera) {
    if (this._highlightedUnit) {
      const normal = new THREE.Vector3()
      camera.getWorldDirection(normal)

      const dragPlane = new THREE.Plane()
      dragPlane.setFromNormalAndCoplanarPoint(
        normal,
        this._intersectionPoint
      )

      const previousCoord =
        (new THREE.Vector3()).copy(this._intersectionPoint).project(camera)
      const projection =
        projectionFromNDC(previousCoord.x + dx, previousCoord.y + dy, camera)

      const newIntersection = this._highlightedUnit.transform(
        this._intersectionPoint,
        projection,
        dragPlane,
        this._labels,
        this._bounds,
        this._local
      )

      this._intersectionPoint.copy(newIntersection)
      return true
    }
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
