import * as THREE from 'three'
import { CubeType } from '../../functional/types'
import { Vector2D } from '../../math/vector2d'
import { Vector3D } from '../../math/vector3d'
import { projectionFromNDC } from '../../view_config/point_cloud'
import { getColorById } from '../util'
import { TransformationControl } from './control/transformation_control'
import { Grid3D } from './grid3d'
import { Plane3D } from './plane3d'

const faceNormals = [
  new THREE.Vector3(1, 0, 0),
  new THREE.Vector3(-1, 0, 0),
  new THREE.Vector3(0, 1, 0),
  new THREE.Vector3(0, -1, 0),
  new THREE.Vector3(0, 0, 1),
  new THREE.Vector3(0, 0, -1)
]

/**
 * Shape for Box3D label
 */
export class Cube3D extends THREE.Group {
  /** Box faces */
  private _box: THREE.Mesh
  /** Outline ThreeJS object */
  private _outline: THREE.LineSegments
  /** Id of corresponding Box3D */
  private _id: number
  /** Color */
  private _color: number[]
  /** Anchor corner index */
  private _anchorIndex: number
  /** Redux state */
  private _center: Vector3D
  /** Redux state */
  private _size: Vector3D
  /** Redux state */
  private _orientation: Vector3D
  /** controls */
  private _control: TransformationControl | null
  /** Normal of the closest face */
  private _closestFaceNormal: THREE.Vector3
  /** Control points */
  private _controlSpheres: THREE.Mesh[]
  /** Highlighted control point */
  private _highlightedSphere: THREE.Mesh | null
  /** whether highlighted */
  private _highlighted: boolean
  /** Plane shape */
  private _grid: Readonly<Grid3D> | null
  /** Id of surface */
  private _surfaceId: number
  /** First corner for temp init */
  private _firstCorner: Vector2D | null

  /**
   * Make box with assigned id
   * @param id
   */
  constructor (id: number) {
    super()
    this._box = new THREE.Mesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshBasicMaterial({
        color: 0xffffff,
        vertexColors: THREE.FaceColors,
        transparent: true,
        opacity: 0.5
      })
    )

    this._outline = new THREE.LineSegments(
      new THREE.EdgesGeometry(this._box.geometry),
      new THREE.LineBasicMaterial({ color: 0xffffff })
    )

    this._id = id

    this._color = []

    this._anchorIndex = 0

    this.add(this._box)
    this.add(this._outline)

    this._center = new Vector3D()
    this._size = new Vector3D()
    this._orientation = new Vector3D()

    this._control = null

    this._closestFaceNormal = new THREE.Vector3()
    this._controlSpheres = []
    for (let i = 0; i < 4; i += 1) {
      this._controlSpheres.push(new THREE.Mesh(
        new THREE.SphereGeometry(0.05, 16, 12),
        new THREE.MeshBasicMaterial({
          color: 0xffffff,
          transparent: true,
          opacity: 0.3
        })
      ))
      this.add(this._controlSpheres[i])
      this._controlSpheres[i].visible = false
    }
    this._highlightedSphere = this._controlSpheres[0]
    this._highlightedSphere.position.x = 0
    this._highlightedSphere = null

    this._highlighted = false

    this ._grid = null

    this._surfaceId = -1

    this.setHighlighted()

    this._firstCorner = null
  }

  /**
   * Set size
   * @param size
   */
  public setSize (size: Vector3D): void {
    this.scale.copy(size.toThree())
    this._size.copy(size)
  }

  /**
   * Get size
   */
  public getSize (): Vector3D {
    return (new Vector3D()).fromThree(this.scale)
  }

  /**
   * Set center position
   * @param center
   */
  public setCenter (center: Vector3D): void {
    this.position.copy(center.toThree())
    this._center.copy(center)
  }

  /**
   * Get center position
   */
  public getCenter (): Vector3D {
    return (new Vector3D()).fromThree(this.position)
  }

  /**
   * Set orientation as euler
   * @param orientation
   */
  public setOrientation (orientation: Vector3D): void {
    this.rotation.setFromVector3(orientation.toThree())
    this._orientation.copy(orientation)
  }

  /**
   * Get orientation as euler
   */
  public getOrientation (): Vector3D {
    return (new Vector3D()).fromThree(this.rotation.toVector3())
  }

  /**
   * set id of associated label
   * @param id
   */
  public setId (id: number): void {
    this._id = id
    this._color = getColorById(id)
    this._color = this._color.map((v) => v / 255.)
  }

  /**
   * Get index
   */
  public getId (): number {
    return this._id
  }

  /**
   * Get ThreeJS box
   */
  public getBox (): THREE.Mesh {
    return this._box
  }

  /**
   * Set surface id
   * @param id
   */
  public setSurfaceId (id: number) {
    this._surfaceId = id
  }

  /**
   * Convert to state representation
   */
  public toCube (): CubeType {
    return {
      center: this.getCenter().toObject(),
      size: this.getSize().toObject(),
      orientation: this.getOrientation().toObject(),
      anchorIndex: this._anchorIndex,
      surfaceId: this._surfaceId
    }
  }

  /**
   * move anchor to next corner
   */
  public incrementAnchorIndex (): void {
    this._anchorIndex = (this._anchorIndex + 1) % 8
  }

  /**
   * attach to plane
   * @param plane
   */
  public attachToPlane (plane: Plane3D) {
    this._grid = plane.shapes()[0]
    this._grid.add(this)
  }

  /**
   * attach to plane
   * @param plane
   */
  public detachFromPlane () {
    if (this._grid) {
      this._grid.remove(this)
    }
    this._grid = null
  }

  /**
   * Add to scene for rendering
   * @param scene
   */
  public render (scene: THREE.Scene,
                 camera: THREE.Camera): void {
    if (this._highlighted) {
      (this._outline.material as THREE.LineBasicMaterial).color.set(0xff0000)
      for (const sphere of this._controlSpheres) {
        sphere.visible = true
        sphere.scale.set(
          1. / this.scale.x, 1. / this.scale.y, 1. / this.scale.z
        )
      }

      this.setControlSpheres(camera)
    } else {
      (this._outline.material as THREE.LineBasicMaterial).color.set(0xffffff)
      for (const sphere of this._controlSpheres) {
        sphere.visible = false
      }
    }

    // Check if shape already in scene
    for (const child of scene.children) {
      if (child === this) {
        return
      }
    }

    if (!this._grid) {
      scene.add(this)
    }
  }

  /**
   * Add/remove controls
   * @param control
   * @param show
   */
  public setControl (control: TransformationControl, show: boolean) {
    if (show) {
      this.add(control)
      this._control = control
      this._control.attach(this)
    } else if (this._control) {
      this._control.detach()
      this.remove(control)
      this._control = null
    }
  }

  /** Set highlighted */
  public setHighlighted (intersection?: THREE.Intersection) {
    for (const sphere of this._controlSpheres) {
      { (sphere.material as THREE.Material).opacity = 0.3 }
      { (sphere.material as THREE.Material).needsUpdate = true }
    }
    this._highlightedSphere = null
    if (intersection) {
      this._highlighted = true
      for (const sphere of this._controlSpheres) {
        if (intersection.object === sphere) {
          this._highlightedSphere = sphere
          { (sphere.material as THREE.Material).opacity = 0.8 }
          break
        }
      }
    } else {
      this._highlighted = false
    }
  }

  /**
   * Override ThreeJS raycast to intersect with box
   * @param raycaster
   * @param intersects
   */
  public raycast (
    raycaster: THREE.Raycaster,
    intersects: THREE.Intersection[]
  ) {
    const newIntersects: THREE.Intersection[] = []
    if (this._control) {
      this._control.raycast(raycaster, newIntersects)
    }

    for (const sphere of this._controlSpheres) {
      sphere.raycast(raycaster, newIntersects)
    }

    if (newIntersects.length === 0) {
      this._box.raycast(raycaster, intersects)
    } else {
      for (const intersect of newIntersects) {
        intersects.push(intersect)
      }
    }
  }

  /**
   * Init params for click creation
   * @param x
   * @param y
   * @param camera
   */
  public clickInit (x: number, y: number, camera: THREE.Camera) {
    if (this._grid) {
      this._firstCorner = new Vector2D(x, y)
      const projection = projectionFromNDC(x, y, camera)

      const normal = new THREE.Vector3(0, 0, 1)
      normal.applyQuaternion(this._grid.quaternion)

      const plane = new THREE.Plane()
      plane.setFromNormalAndCoplanarPoint(normal, this._grid.position)

      const newPosition = new THREE.Vector3()
      projection.intersectPlane(plane, newPosition)

      const toGrid = new THREE.Matrix4()
      toGrid.getInverse(this._grid.matrixWorld)

      newPosition.applyMatrix4(toGrid)
      this.position.copy(newPosition)

      this.updateMatrixWorld(true)

      this.visible = false
    }
  }

  /**
   * Drag to mouse
   * @param projection
   */
  public drag (x: number, y: number, camera: THREE.Camera) {
    const projection = projectionFromNDC(x, y, camera)

    this.updateMatrixWorld(true)

    const toLocal = new THREE.Matrix4()
    toLocal.getInverse(this.matrixWorld)

    const localProjection = new THREE.Ray()
    localProjection.copy(projection)
    localProjection.applyMatrix4(toLocal)

    const highlightedPlaneNormal = new THREE.Vector3()
    highlightedPlaneNormal.copy(this._closestFaceNormal)

    const highlightedPlane = new THREE.Plane(highlightedPlaneNormal)

    if (this._firstCorner && this._grid) {
      this.setControlSpheres(camera)

      const delta = new THREE.Vector2(
        x - this._firstCorner.x,
        y - this._firstCorner.y
      )

      if (delta.length() < 0.01) {
        return false
      }

      const normal = new THREE.Vector3()
      normal.copy(this._closestFaceNormal)
      const planePoint = new THREE.Vector3()
      planePoint.copy(this._controlSpheres[0].position)

      const plane = new THREE.Plane()
      plane.setFromNormalAndCoplanarPoint(normal, planePoint)

      const initialIntersect = new THREE.Vector3()

      localProjection.intersectPlane(highlightedPlane, initialIntersect)

      let closestDistance = Infinity
      this._highlightedSphere = this._controlSpheres[0]

      for (const sphere of this._controlSpheres) {
        const initialDelta = new THREE.Vector3()
        initialDelta.copy(sphere.position)
        initialDelta.sub(initialIntersect)
        const dist = initialDelta.length()
        if (dist < closestDistance) {
          closestDistance = dist
          this._highlightedSphere = sphere
        }
      }

      if (this._highlightedSphere.position.z < 0) {
        return false
      }

      const oppositeCornerInit = new THREE.Vector3()
      oppositeCornerInit.copy(this._highlightedSphere.position)
      oppositeCornerInit.multiplyScalar(-1)

      if (this._closestFaceNormal.x !== 0) {
        oppositeCornerInit.x *= -1
      } else if (this._closestFaceNormal.y !== 0) {
        oppositeCornerInit.y *= -1
      } else {
        oppositeCornerInit.z *= -1
      }

      oppositeCornerInit.applyMatrix4(this.matrix)

      this.position.add(this.position)
      this.position.sub(oppositeCornerInit)

      this.updateMatrixWorld(true)

      this._firstCorner = null

      this.visible = true
    }

    if (!this._highlightedSphere) {
      return false
    }

    highlightedPlane.setFromNormalAndCoplanarPoint(
      highlightedPlaneNormal,
      this._highlightedSphere.position
    )

    const intersection = new THREE.Vector3()

    const startingPosition = new THREE.Vector3()
    startingPosition.copy(this._highlightedSphere.position)

    if (this._grid && this._highlightedSphere.position.z < 0) {
      const gridNormal = new THREE.Vector3()
      gridNormal.z = 1
      const gridPlane = new THREE.Plane()
      gridPlane.setFromNormalAndCoplanarPoint(
        gridNormal, this._highlightedSphere.position
      )

      const newHighlightedLocalPosition = new THREE.Vector3()
      localProjection.intersectPlane(gridPlane, newHighlightedLocalPosition)

      const highlightedPositionDelta = new THREE.Vector3()
      highlightedPositionDelta.copy(newHighlightedLocalPosition)
      highlightedPositionDelta.sub(this._highlightedSphere.position)

      const noTranslationMatrix = new THREE.Matrix3()
      noTranslationMatrix.setFromMatrix4(this.matrix)
      highlightedPositionDelta.applyMatrix3(noTranslationMatrix)

      const oppositeCorner = new THREE.Vector3()
      oppositeCorner.copy(this._highlightedSphere.position)
      oppositeCorner.multiplyScalar(-1)

      if (this._closestFaceNormal.x !== 0) {
        oppositeCorner.x *= -1
      } else if (this._closestFaceNormal.y !== 0) {
        oppositeCorner.y *= -1
      } else {
        oppositeCorner.z *= -1
      }

      const oppositeLocal = new THREE.Vector3()
      oppositeLocal.copy(oppositeCorner)

      oppositeCorner.applyMatrix4(this.matrixWorld)

      this.position.add(highlightedPositionDelta)
      this.updateMatrixWorld(true)

      const rayDirection = new THREE.Vector3()
      rayDirection.copy(oppositeCorner)
      rayDirection.sub(camera.position)
      rayDirection.normalize()

      const cameraPosition = new THREE.Vector3()
      cameraPosition.copy(camera.position)

      const rayToCorner = new THREE.Ray(cameraPosition, rayDirection)
      toLocal.getInverse(this.matrixWorld, true)
      rayToCorner.applyMatrix4(toLocal)

      rayToCorner.intersectPlane(highlightedPlane, intersection)

      startingPosition.copy(oppositeLocal)
    } else {
      localProjection.intersectPlane(highlightedPlane, intersection)
    }

    const scaleDelta = new THREE.Vector3()
    const positionDelta = new THREE.Vector3()

    scaleDelta.copy(intersection)
    scaleDelta.sub(startingPosition)
    scaleDelta.multiply(this.scale)

    const worldQuaternion = new THREE.Quaternion()
    this.getWorldQuaternion(worldQuaternion)

    positionDelta.copy(scaleDelta)
    positionDelta.multiplyScalar(0.5)
    positionDelta.applyQuaternion(this.quaternion)

    scaleDelta.multiply(startingPosition)
    scaleDelta.multiplyScalar(2)

    const newScale = new THREE.Vector3()
    newScale.copy(this.scale)
    newScale.add(scaleDelta)

    if (newScale.x < 0 || newScale.y < 0 || newScale.z < 0) {
      return true
    }

    this.position.add(positionDelta)
    this.scale.add(scaleDelta)

    return true
  }

  /**
   * Returns true if control sphere is highlighted
   */
  public shouldDrag (): boolean {
    return this._highlightedSphere !== null || this._firstCorner !== null
  }

  /**
   * Set sphere positions from normal
   * @param normal
   */
  private setControlSpheres (camera: THREE.Camera) {
    // Find normal closest to camera
    const worldQuaternion = new THREE.Quaternion()
    this.getWorldQuaternion(worldQuaternion)
    const cameraDirection = new THREE.Vector3()
    camera.getWorldDirection(cameraDirection)
    cameraDirection.applyQuaternion(worldQuaternion.inverse())
    let maxCloseness = 0
    for (const normal of faceNormals) {
      const closeness = -normal.dot(cameraDirection)
      if (closeness > maxCloseness) {
        this._closestFaceNormal.copy(normal)
        maxCloseness = closeness
      }
    }

    for (let i = 0; i < this._controlSpheres.length; i += 1) {
      const firstSign = (i % 2 === 0) ? -1 : 1
      const secondSign = (Math.floor(i / 2) === 0) ? -1 : 1
      if (this._closestFaceNormal.x !== 0) {
        this._controlSpheres[i].position.set(
          this._closestFaceNormal.x, firstSign, secondSign
        )
      } else if (this._closestFaceNormal.y !== 0) {
        this._controlSpheres[i].position.set(
           firstSign, this._closestFaceNormal.y, secondSign
        )
      } else {
        this._controlSpheres[i].position.set(
           firstSign, secondSign, this._closestFaceNormal.z
        )
      }
      this._controlSpheres[i].position.multiplyScalar(0.5)
    }
  }
}
