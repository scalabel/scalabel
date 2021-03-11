import * as THREE from "three"

import { ShapeTypeName } from "../../const/common"
import { makeCube } from "../../functional/states"
import { Vector2D } from "../../math/vector2d"
import { Vector3D } from "../../math/vector3d"
import { CubeType, IdType, ShapeType } from "../../types/state"
import { projectionFromNDC } from "../../view_config/point_cloud"
import { Grid3D } from "./grid3d"
import Label3D from "./label3d"
import { Plane3D } from "./plane3d"
import { Shape3D } from "./shape3d"

const faceNormals = [
  new THREE.Vector3(1, 0, 0),
  new THREE.Vector3(-1, 0, 0),
  new THREE.Vector3(0, 1, 0),
  new THREE.Vector3(0, -1, 0),
  new THREE.Vector3(0, 0, 1),
  new THREE.Vector3(0, 0, -1)
]

const DISTANCE_SCALE_CORRECTION = 5

/**
 * Shape for Box3D label
 */
export class Cube3D extends Shape3D {
  /** Box faces */
  private readonly _box: THREE.Mesh
  /** Outline ThreeJS object */
  private readonly _outline: THREE.LineSegments
  /** Color */
  private _color: number[]
  /** Anchor corner index */
  private _anchorIndex: number
  /** Normal of the closest face */
  private readonly _closestFaceNormal: THREE.Vector3
  /** Control points */
  private readonly _controlSpheres: THREE.Mesh[]
  /** Highlighted control point */
  private _highlightedSphere: THREE.Mesh | null
  /** Plane shape */
  private _grid: Readonly<Grid3D> | null
  /** First corner for temp init */
  private _firstCorner: Vector2D | null
  /** internal shape state */
  private _cubeShape: CubeType

  /**
   * Make box with assigned id
   *
   * @param id
   * @param label
   */
  constructor(label: Label3D) {
    super(label)
    this._box = new THREE.Mesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshBasicMaterial({
        color: 0xffffff,
        vertexColors: THREE.FaceColors,
        transparent: true,
        opacity: 0.35
      })
    )
    this.add(this._box)
    this._box.geometry.computeBoundingBox()

    this._outline = new THREE.LineSegments(
      new THREE.EdgesGeometry(this._box.geometry),
      new THREE.LineBasicMaterial({ color: 0xffffff })
    )
    this.add(this._outline)

    this._color = []

    this._anchorIndex = 0

    this._closestFaceNormal = new THREE.Vector3()
    this._controlSpheres = []
    for (let i = 0; i < 4; i += 1) {
      this._controlSpheres.push(
        new THREE.Mesh(
          new THREE.SphereGeometry(0.05, 16, 12),
          new THREE.MeshBasicMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.3
          })
        )
      )
      this.add(this._controlSpheres[i])
      this._controlSpheres[i].visible = false
    }
    this._highlightedSphere = this._controlSpheres[0]
    this._highlightedSphere.position.x = 0
    this._highlightedSphere = null

    this._grid = null

    this._firstCorner = null

    this._cubeShape = makeCube()

    this.setHighlighted()
  }

  /** Get shape type name */
  public get typeName(): string {
    return ShapeTypeName.CUBE
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
      this._box.layers.enable(viewerId)
      this._outline.layers.enable(viewerId)
      for (const sphere of this._controlSpheres) {
        sphere.layers.enable(viewerId)
      }
    } else {
      this._box.layers.disable(viewerId)
      this._outline.layers.disable(viewerId)
      for (const sphere of this._controlSpheres) {
        sphere.layers.disable(viewerId)
      }
    }
  }

  /**
   * Set color
   * TODO: make consistent convention for color type, either int or 0~1 float
   *
   * @param color
   */
  public set color(color: number[]) {
    this._color = color.map((v) => v / 255)
  }

  /**
   * Get color
   */
  public get color(): number[] {
    return this._color
  }

  /**
   * Get ThreeJS box
   */
  public get box(): THREE.Mesh {
    return this._box
  }

  /** get the shape id */
  public get shapeId(): IdType {
    return this._cubeShape.id
  }

  /**
   * Convert to state representation
   */
  public toState(): ShapeType {
    const worldCenter = new THREE.Vector3()
    this.getWorldPosition(worldCenter)
    const worldSize = new THREE.Vector3()
    this.getWorldScale(worldSize)
    const worldQuaternion = new THREE.Quaternion()
    this.getWorldQuaternion(worldQuaternion)
    const worldOrientation = new THREE.Euler()
    worldOrientation.setFromQuaternion(worldQuaternion)
    if (this._grid !== null) {
      const inverseRotation = new THREE.Quaternion()
      inverseRotation.copy(this._grid.quaternion)
      inverseRotation.inverse()

      const gridCenter = new THREE.Vector3()
      gridCenter.copy(worldCenter)
      gridCenter.sub(this._grid.position)
      gridCenter.applyQuaternion(inverseRotation)
      gridCenter.z = 0.5 * worldSize.z
      worldCenter.copy(gridCenter)
      worldCenter.applyQuaternion(this._grid.quaternion)
      worldCenter.add(this._grid.position)

      const gridRotation = new THREE.Quaternion()
      gridRotation.copy(this.quaternion)
      gridRotation.multiply(inverseRotation)
      const gridEuler = new THREE.Euler()
      gridEuler.setFromQuaternion(gridRotation)
      gridEuler.x = 0
      gridEuler.y = 0
      worldQuaternion.setFromEuler(gridEuler)
      worldQuaternion.multiply(this._grid.quaternion)
      worldOrientation.setFromQuaternion(worldQuaternion)
    }
    const cube = this._cubeShape
    cube.center = new Vector3D().fromThree(worldCenter).toState()
    cube.size = new Vector3D().fromThree(worldSize).toState()
    cube.orientation = new Vector3D()
      .fromThree(worldOrientation.toVector3())
      .toState()
    cube.anchorIndex = this._anchorIndex

    return cube
  }

  /**
   * move anchor to next corner
   */
  public incrementAnchorIndex(): void {
    this._anchorIndex = (this._anchorIndex + 1) % 8
  }

  /**
   * attach to plane
   *
   * @param plane
   */
  public attachToPlane(plane: Plane3D): void {
    this._grid = plane.internalShapes()[0] as Grid3D
  }

  /**
   * attach to plane
   *
   * @param plane
   */
  public detachFromPlane(): void {
    this._grid = null
  }

  /**
   * update parameters
   *
   * @param shape
   * @param id
   */
  public updateState(shape: ShapeType, id: IdType): void {
    const geometry = this._box.geometry as THREE.Geometry
    for (const face of geometry.faces) {
      face.color.fromArray(this._color)
    }
    super.updateState(shape, id)
    const cube = shape as CubeType
    this.position.copy(new Vector3D().fromState(cube.center).toThree())
    this.rotation.copy(
      new Vector3D().fromState(cube.orientation).toThreeEuler()
    )
    this.scale.copy(new Vector3D().fromState(cube.size).toThree())
    // Also update the _cubeShape
    this._cubeShape = cube
  }

  /**
   * Add to scene for rendering
   *
   * @param scene
   * @param camera
   */
  public render(scene: THREE.Scene, camera: THREE.Camera): void {
    if (this._highlighted) {
      ;(this._outline.material as THREE.LineBasicMaterial).color.set(0xff0000)
      for (const sphere of this._controlSpheres) {
        sphere.visible = true
        sphere.scale.set(1 / this.scale.x, 1 / this.scale.y, 1 / this.scale.z)
      }

      this.setControlSpheres(camera)
    } else if (this._label.selected) {
      ;(this._outline.material as THREE.LineBasicMaterial).color.set(0xffff00)
    } else {
      ;(this._outline.material as THREE.LineBasicMaterial).color.set(0xffffff)
      for (const sphere of this._controlSpheres) {
        sphere.visible = false
      }
    }

    const geometry = this._box.geometry as THREE.Geometry
    for (const face of geometry.faces) {
      face.color.fromArray(this._color)
    }

    // Check if shape already in scene
    for (const child of scene.children) {
      if (child === this) {
        return
      }
    }

    if (this._grid === null) {
      scene.add(this)
    }
  }

  /**
   * Set highlighted
   *
   * @param intersection
   */
  public setHighlighted(intersection?: THREE.Intersection): void {
    for (const sphere of this._controlSpheres) {
      ;(sphere.material as THREE.Material).opacity = 0.3
      ;(sphere.material as THREE.Material).needsUpdate = true

      sphere.visible = true
    }
    this._highlightedSphere = null
    if (intersection !== undefined) {
      ;(this._outline.material as THREE.LineBasicMaterial).color.set(0xff0000)
      this._highlighted = true

      for (const sphere of this._controlSpheres) {
        if (intersection.object === sphere) {
          this._highlightedSphere = sphere
          ;(sphere.material as THREE.Material).opacity = 0.8

          break
        }
      }
    } else {
      ;(this._outline.material as THREE.LineBasicMaterial).color.set(0xffffff)
      this._highlighted = false

      for (const sphere of this._controlSpheres) {
        sphere.visible = false
      }
    }
  }

  /**
   * Override ThreeJS raycast to intersect with box
   *
   * @param raycaster
   * @param intersects
   */
  public raycast(
    raycaster: THREE.Raycaster,
    intersects: THREE.Intersection[]
  ): void {
    const newIntersects: THREE.Intersection[] = []

    for (const sphere of this._controlSpheres) {
      sphere.raycast(raycaster, newIntersects)
    }

    if (newIntersects.length > 0) {
      for (const intersect of newIntersects) {
        intersects.push(intersect)
      }
      return
    }

    if (this.label.selected) {
      this.label.labelList.control.raycast(raycaster, newIntersects)
      if (newIntersects.length > 0) {
        for (const intersect of newIntersects) {
          intersects.push(intersect)
        }
        return
      }
    }

    this._box.raycast(raycaster, intersects)
  }

  /**
   * Init params for click creation
   *
   * @param x
   * @param y
   * @param camera
   */
  public clickInit(x: number, y: number, camera: THREE.Camera): void {
    if (this._grid !== null) {
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

      // NewPosition.applyMatrix4(toGrid)
      this.position.copy(newPosition)
      this.quaternion.copy(this._grid.quaternion)

      // This._grid.attach(this)

      // This.updateMatrixWorld(true)

      // This.visible = false
    }
  }

  // TODO: remove this disable
  /* eslint-disable max-lines-per-function,max-statements */
  /**
   * Drag to mouse
   *
   * @param projection
   * @param x
   * @param y
   * @param camera
   */
  public drag(x: number, y: number, camera: THREE.Camera): boolean {
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

    if (this._firstCorner !== null && this._grid !== null) {
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

    if (this._highlightedSphere === null) {
      return false
    }

    highlightedPlane.setFromNormalAndCoplanarPoint(
      highlightedPlaneNormal,
      this._highlightedSphere.position
    )

    const intersection = new THREE.Vector3()

    const startingPosition = new THREE.Vector3()
    startingPosition.copy(this._highlightedSphere.position)

    if (this._grid !== null && this._highlightedSphere.position.z < 0) {
      const gridNormal = new THREE.Vector3()
      gridNormal.z = 1
      const gridPlane = new THREE.Plane()
      gridPlane.setFromNormalAndCoplanarPoint(
        gridNormal,
        this._highlightedSphere.position
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

    for (const sphere of this._controlSpheres) {
      sphere.scale.set(1 / this.scale.x, 1 / this.scale.y, 1 / this.scale.z)
    }

    return true
  }

  /**
   * Returns true if control sphere is highlighted
   */
  public shouldDrag(): boolean {
    return this._highlightedSphere !== null || this._firstCorner !== null
  }

  /**
   * Set sphere positions from normal
   *
   * @param normal
   * @param camera
   */
  public setControlSpheres(camera: THREE.Camera): void {
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

    const scaleVector = new THREE.Vector3()
    const scaleFactor =
      scaleVector.subVectors(this.position, camera.position).length() /
      DISTANCE_SCALE_CORRECTION

    for (let i = 0; i < this._controlSpheres.length; i += 1) {
      const firstSign = i % 2 === 0 ? -1 : 1
      const secondSign = Math.floor(i / 2) === 0 ? -1 : 1
      if (this._closestFaceNormal.x !== 0) {
        this._controlSpheres[i].position.set(
          this._closestFaceNormal.x,
          firstSign,
          secondSign
        )
      } else if (this._closestFaceNormal.y !== 0) {
        this._controlSpheres[i].position.set(
          firstSign,
          this._closestFaceNormal.y,
          secondSign
        )
      } else {
        this._controlSpheres[i].position.set(
          firstSign,
          secondSign,
          this._closestFaceNormal.z
        )
      }
      this._controlSpheres[i].position.multiplyScalar(0.5)

      this._controlSpheres[i].scale.set(
        1 / this.scale.x,
        1 / this.scale.y,
        1 / this.scale.z
      )
      this._controlSpheres[i].scale.multiplyScalar(scaleFactor)
    }
  }
}
