import * as THREE from 'three'

export interface ControlUnit extends THREE.Object3D {
  /** get update vectors: [translation, rotation, scale, new intersection] */
  getDelta: (
    oldIntersection: THREE.Vector3,
    newProjection: THREE.Ray,
    dragPlane: THREE.Plane,
    object?: THREE.Object3D
  ) => [THREE.Vector3, THREE.Quaternion, THREE.Vector3, THREE.Vector3]
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
  /** Object to modify */
  protected _object: THREE.Object3D | null
  /** local or world */
  protected _local: boolean
  /** original intersection point */
  protected _intersectionPoint: THREE.Vector3
  /** Plane of intersection point w/ camera direction */
  protected _dragPlane: THREE.Plane
  /** previous projection */
  protected _projection: THREE.Ray

  constructor () {
    super()
    this._controlUnits = []
    this._object = null
    this._local = true
    this._intersectionPoint = new THREE.Vector3()
    this._highlightedUnit = null
    this._dragPlane = new THREE.Plane()
    this._projection = new THREE.Ray()
    this.matrixAutoUpdate = false
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
    if (this._highlightedUnit && this._object) {
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
    if (this._highlightedUnit && this._dragPlane && this._object) {
      const object = (this._local) ? this._object : undefined
      const [
        translationDelta,
        quaternionDelta,
        scaleDelta,
        newIntersection
      ] = this._highlightedUnit.getDelta(
        this._intersectionPoint,
        projection,
        this._dragPlane,
        object
      )

      const newScale = new THREE.Vector3()
      newScale.copy(this._object.scale)
      newScale.add(scaleDelta)

      if (Math.min(newScale.x, newScale.y, newScale.z) > 0.01) {
        this._object.position.add(translationDelta)
        this._object.applyQuaternion(quaternionDelta)
        this._object.scale.add(scaleDelta)
      }

      this._intersectionPoint.copy(newIntersection)
      this.refreshDisplayParameters()
      this._projection.copy(projection)
      return true
    }
    this.refreshDisplayParameters()
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

  /** attach to object */
  public attach (object: THREE.Object3D) {
    this._object = object
    this.updateMatrix()
    this.updateMatrixWorld(true)
    this.refreshDisplayParameters()
  }

  /** detach */
  public detach () {
    this._object = null
  }

  /** Toggle local/world */
  public toggleFrame () {
    if (this._object) {
      this._local = !this._local
      this.attach(this._object)
    }
  }

  /** Refresh display params */
  protected refreshDisplayParameters () {
    if (this._object) {
      // Isolate child from parent transformations first
      this._object.updateMatrixWorld(true)
      this.matrix.getInverse(this._object.matrix)

      this.matrix.setPosition(new THREE.Vector3())
      if (this._local) {
        // Move back to _object frame of reference, but do not apply scaling

        this.matrix.multiply(
          (new THREE.Matrix4()).makeRotationFromQuaternion(
            this._object.quaternion
          )
        )
      }

      const worldScale = new THREE.Vector3()
      this._object.getWorldScale(worldScale)

      for (const unit of this._controlUnits) {
        unit.updateScale(worldScale)
      }
    }
  }
}
