import * as THREE from 'three'
import { CubeType } from '../functional/types'
import { Vector3D } from '../math/vector3d'
import { getColorById } from './util'

export enum DrawMode {
  STANDBY,
  SELECTED,
  MOVING,
  SCALING,
  ROTATING,
  EXTRUDING
}

/**
 * Shape for Box3D label
 */
export class Cube3D extends THREE.Group {

  /**
   * Set drawing mode
   * @param drawMode
   */
  public set drawMode (drawMode: DrawMode) {
    this._drawMode = drawMode
  }

  /**
   * Get drawing mode
   */
  public get drawMode (): DrawMode {
    return this._drawMode
  }
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
  /** ThreeJS Axes for visualization anchor position */
  private _axes: THREE.Group
  /** drawing mode */
  private _drawMode: DrawMode
  /** Redux state */
  private _center: Vector3D
  /** Redux state */
  private _size: Vector3D
  /** Redux state */
  private _orientation: Vector3D

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

    const origin = new THREE.Vector3()
    this._axes = new THREE.Group()
    this._axes.add(new THREE.ArrowHelper(
      new THREE.Vector3(0, 0, 1), origin, 1.1, 0xffffff
    ))
    this._axes.add(new THREE.ArrowHelper(
      new THREE.Vector3(0, 1, 0), origin, 1.1, 0xffffff
    ))
    this._axes.add(new THREE.ArrowHelper(
      new THREE.Vector3(1, 0, 0), origin, 1.1, 0xffffff
    ))
    this._axes.add(new THREE.Mesh(
      new THREE.SphereGeometry(0.02),
      new THREE.MeshBasicMaterial({ color: 0xffffff })
    ))
    this._axes.position.x = -0.5
    this._axes.position.y = -0.5
    this._axes.position.z = -0.5

    this.add(this._box)
    this.add(this._outline)
    this.add(this._axes)

    this._drawMode = DrawMode.STANDBY

    this._center = new Vector3D()
    this._size = new Vector3D()
    this._orientation = new Vector3D()
  }

  /**
   * Set size
   * @param size
   */
  public setSize (size: Vector3D): void {
    this.scale.copy(size.toThree())
    this._axes.scale.x = 1. / size[0]
    this._axes.scale.y = 1. / size[1]
    this._axes.scale.z = 1. / size[2]
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
   * Convert to state representation
   */
  public toCube (): CubeType {
    return {
      center: this.getCenter(),
      size: this.getSize(),
      orientation: this.getOrientation(),
      anchorIndex: this._anchorIndex
    }
  }

  /**
   * move anchor to next corner
   */
  public incrementAnchorIndex (): void {
    this._anchorIndex = (this._anchorIndex + 1) % 8
  }

  /**
   * Add to scene for rendering
   * @param scene
   */
  public render (scene: THREE.Scene,
                 highlighted: boolean): void {
    if (highlighted) {
      (this._outline.material as THREE.LineBasicMaterial).color.set(0xff0000)
    } else {
      (this._outline.material as THREE.LineBasicMaterial).color.set(0xffffff)
    }

    switch (this._drawMode) {
      case DrawMode.STANDBY:
        this.setColorFromRGB(this._color)
        break
      case DrawMode.MOVING:
      case DrawMode.ROTATING:
      case DrawMode.SELECTED:
        this.setColor(0xff0000)
        break
      case DrawMode.SCALING:
        this.setColor(0x00ff00, [8, 9, 10, 11])
        break
      case DrawMode.EXTRUDING:
        this.setColor(0x00ff00, [0, 1, 2, 3, 4, 5, 6, 7])
        break
    }

    this._axes.position.z = Math.floor(this._anchorIndex / 4) - 0.5
    this._axes.position.y = Math.floor(this._anchorIndex / 2) % 2 - 0.5
    this._axes.position.x = this._anchorIndex % 2 - 0.5

    // Check if shape already in scene
    for (const child of scene.children) {
      if (child === this) {
        return
      }
    }

    scene.add(this)
  }

  /**
   * Move box along view plane
   * @param viewPlaneNormal
   * @param projection
   * @param cameraPosition
   * @param cameraToOriginalPosition
   * @param centerToIntersectionOffset
   */
  public moveAlongViewPlane (
    projection: THREE.Vector3,
    viewPlaneNormal: THREE.Vector3,
    cameraPosition: THREE.Vector3,
    intersectionToCamera: THREE.Vector3,
    centerToIntersectionOffset: THREE.Vector3
  ) {
    // Get distance from camera to new intersection between projection & plane
    const dist = -intersectionToCamera.dot(viewPlaneNormal) /
      projection.dot(viewPlaneNormal)

    const newProjection = new THREE.Vector3()
    newProjection.copy(projection)
    newProjection.multiplyScalar(dist)

    // Set box position to point
    newProjection.add(cameraPosition)
    newProjection.sub(centerToIntersectionOffset)

    this.position.copy(newProjection)
  }

  /**
   * Scale box relative to anchor
   * @param projection
   * @param boxIntersectionPoint
   * @param cameraPosition
   */
  public scaleToProjection (
    projection: THREE.Vector3,
    boxIntersectionPoint: THREE.Vector3,
    cameraPosition: THREE.Vector3
  ) {
    const worldToModel = new THREE.Matrix4()
    worldToModel.getInverse(this.matrixWorld)

    const intersectionModel = new THREE.Vector3()
    intersectionModel.copy(boxIntersectionPoint)
    intersectionModel.applyMatrix4(worldToModel)

    const boxPlaneNormal = new THREE.Vector3(0, 0, 1)
    const boxPlane = new THREE.Plane(boxPlaneNormal,
      -Math.sign(Number(intersectionModel.z > 0)) * 0.5)

    const cameraPositionModel = new THREE.Vector3()
    cameraPositionModel.copy(cameraPosition)
    cameraPositionModel.applyMatrix4(worldToModel)

    const projectionModel = new THREE.Vector3()
    projectionModel.copy(projection)
    projectionModel.transformDirection(worldToModel)

    const mouseRay = new THREE.Ray(cameraPositionModel, projectionModel)

    const intersectionPoint = new THREE.Vector3()
    mouseRay.intersectPlane(boxPlane, intersectionPoint)

    // Set closest box corner to be at the point
    if (Math.abs(intersectionPoint.x - this._axes.position.x) >
        Math.abs(intersectionPoint.y - this._axes.position.y)) {
      const boxBase = new THREE.Vector3()
      boxBase.x = this._axes.position.x
      boxBase.applyMatrix4(this.matrixWorld)

      let newScale = (intersectionPoint.x - this._axes.position.x) *
        this.scale.x

      if (this._axes.position.x > 0) {
        newScale = Math.min(newScale, -0.1)
      } else {
        newScale = Math.max(newScale, 0.1)
      }

      const horizontal = new THREE.Vector3()
      horizontal.x = 1
      horizontal.transformDirection(this.matrixWorld)
      horizontal.multiplyScalar(newScale / 2)

      this.position.copy(boxBase)
      this.position.add(horizontal)

      this.scale.x = Math.abs(newScale)
      this._axes.scale.x = Math.abs(1. / newScale)
    } else {
      const boxBase = new THREE.Vector3()
      boxBase.y = this._axes.position.y
      boxBase.applyMatrix4(this.matrixWorld)

      let newScale = (intersectionPoint.y - this._axes.position.y) *
        this.scale.y

      if (this._axes.position.y > 0) {
        newScale = Math.min(newScale, -0.1)
      } else {
        newScale = Math.max(newScale, 0.1)
      }

      const horizontal = new THREE.Vector3()
      horizontal.y = 1
      horizontal.transformDirection(this.matrixWorld)
      horizontal.multiplyScalar(newScale / 2)

      this.position.copy(boxBase)
      this.position.add(horizontal)

      this.scale.y = Math.abs(newScale)
      this._axes.scale.y = Math.abs(1. / newScale)
    }

    this.updateMatrix()
    this.updateMatrixWorld()
  }

  /**
   * Extrude relative to anchor
   * @param projection
   * @param cameraPosition
   */
  public extrudeToProjection (
    projection: THREE.Vector3,
    cameraPosition: THREE.Vector3
  ) {
    const worldToModel = new THREE.Matrix4()
    worldToModel.getInverse(this.matrixWorld)

    const cameraPositionModel = new THREE.Vector3()
    cameraPositionModel.copy(cameraPosition)
    cameraPositionModel.applyMatrix4(worldToModel)

    const projectionModel = new THREE.Vector3()
    projectionModel.copy(projection)
    projectionModel.transformDirection(worldToModel)

    const det = 1 - projectionModel.z * projectionModel.z
    const originDiff = new THREE.Vector3()
    originDiff.copy(this._axes.position)
    originDiff.sub(cameraPositionModel)

    projectionModel.multiplyScalar((originDiff.dot(projectionModel) -
      originDiff.z * projectionModel.z) / det)

    const closestPoint = new THREE.Vector3()
    closestPoint.copy(cameraPositionModel)
    closestPoint.add(projectionModel)

    let newScale = (closestPoint.z - this._axes.position.z) *
      this.scale.z

    if (this._axes.position.z > 0) {
      newScale = Math.min(newScale, -0.1)
    } else {
      newScale = Math.max(newScale, 0.1)
    }

    const boxBase = new THREE.Vector3()
    boxBase.z = this._axes.position.z
    boxBase.applyMatrix4(this.matrixWorld)

    const vertical = new THREE.Vector3()
    vertical.z = 1
    vertical.transformDirection(this.matrixWorld)
    vertical.multiplyScalar(newScale / 2)

    const newBoxPosition = new THREE.Vector3()
    newBoxPosition.copy(boxBase)
    newBoxPosition.add(vertical)

    this.position.copy(newBoxPosition)

    // Scale
    this.scale.z = Math.abs(newScale)
    this._axes.scale.z = Math.abs(1. / newScale)

    this.updateMatrix()
    this.updateMatrixWorld()
  }

  /**
   * Rotate about anchor
   * @param projection
   * @param initialProjection
   * @param cameraPosition
   * @param cameraWorldDirection
   */
  public rotateToProjection (
    projection: THREE.Vector3,
    viewPlaneNormal: THREE.Vector3,
    boxIntersectionPoint: THREE.Vector3,
    cameraPosition: THREE.Vector3
  ) {
    const newRay = new THREE.Ray(cameraPosition, projection)

    const initialDirection = new THREE.Vector3()
    initialDirection.subVectors(
      boxIntersectionPoint, cameraPosition
    ).normalize()
    const initialRay = new THREE.Ray(cameraPosition, initialDirection)

    const initialRotation = new THREE.Quaternion()
    initialRotation.setFromEuler(this._orientation.toThreeEuler())

    const initialTransformation = new THREE.Matrix4()
    initialTransformation.compose(
      this._center.toThree(),
      initialRotation,
      this._size.toThree()
    )

    const anchor = new THREE.Vector3()
    anchor.copy(this._axes.position)
    anchor.applyMatrix4(initialTransformation)
    const anchorDirection = new THREE.Vector3()
    anchorDirection.subVectors(
      anchor, cameraPosition
    ).normalize()
    const anchorRay = new THREE.Ray(cameraPosition, anchorDirection)

    const viewPlane = new THREE.Plane(viewPlaneNormal,
      viewPlaneNormal.dot(anchor))

    let anchorHit = new THREE.Vector3()
    let initialHit = new THREE.Vector3()
    let newHit = new THREE.Vector3()

    anchorHit = anchorRay.intersectPlane(viewPlane, anchorHit)
    initialHit = initialRay.intersectPlane(viewPlane, initialHit)
    newHit = newRay.intersectPlane(viewPlane, newHit)

    if (initialHit && newHit && anchorHit) {
      this.quaternion.copy(initialRotation)
      this.rotation.setFromQuaternion(this.quaternion)

      const initialOffset = new THREE.Vector3()
      const newOffset = new THREE.Vector3()
      initialOffset.subVectors(initialHit, anchorHit)
      newOffset.subVectors(newHit, anchorHit)

      let angle = initialOffset.angleTo(newOffset)

      const posCross = new THREE.Vector3()
      posCross.copy(initialOffset)
      posCross.cross(newOffset)
      if (posCross.dot(viewPlaneNormal) < 0) {
        angle = -angle
      }

      // Set to original rotation at start of edit
      const q = new THREE.Quaternion()
      q.setFromAxisAngle(viewPlaneNormal, angle)
      this.quaternion.premultiply(q)
      this.rotation.setFromQuaternion(this.quaternion)

      const newTransformation = new THREE.Matrix4()
      newTransformation.compose(
        this._center.toThree(), this.quaternion,
        this._size.toThree()
      )

      const newAnchor = new THREE.Vector3()
      newAnchor.copy(this._axes.position)
      newAnchor.applyMatrix4(newTransformation)

      const delta = new THREE.Vector3()
      delta.copy(newAnchor)
      delta.sub(anchor)

      this.position.copy(this._center.toThree())
      this.position.sub(delta)

      this.updateMatrix()
      this.updateMatrixWorld(true)
    }
  }

  /**
   * Set bbox face colors
   * @param color
   * @param faces
   */
  private setColor (
    color: number,
    faces: number[] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  ) {
    const geometry = this._box.geometry as THREE.BoxGeometry
    for (const i of faces) {
      geometry.faces[i].color.set(color)
    }

    geometry.colorsNeedUpdate = true
  }

  /**
   * Set bbox face colors from RGB array
   * @param color
   * @param faces
   */
  private setColorFromRGB (
    color: number[],
    faces: number[] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  ) {
    const geometry = this._box.geometry as THREE.BoxGeometry
    for (const i of faces) {
      geometry.faces[i].color.setRGB(color[0], color[1], color[2])
    }

    geometry.colorsNeedUpdate = true
  }
}
