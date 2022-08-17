import * as THREE from "three"

import { selectLabel, selectLabel3dType } from "../../action/select"
import {
  activateSpan,
  deactivateSpan,
  pauseSpan,
  resetSpan,
  resumeSpan,
  undoSpan
} from "../../action/span3d"
import Session from "../../common/session"
import {
  DataType,
  Key,
  LabelTypeName,
  ViewerConfigTypeName
} from "../../const/common"
import {
  getCurrentViewerConfig,
  isItemLoaded
} from "../../functional/state_util"
import { makePointCloudViewerConfig, makeSensor } from "../../functional/states"
import { Vector3D } from "../../math/vector3d"
import {
  CubeType,
  INVALID_ID,
  PointCloudViewerConfigType,
  SensorType,
  State,
  ViewerConfigType
} from "../../types/state"
import { commitLabels } from "../states"
import { Label3D } from "./label3d"
import { alert } from "../../common/alert"
import { Severity } from "../../types/common"
import { Plane3D } from "./plane3d"
import { Span3D } from "./box_span/span3d"
import {
  calculatePlaneCenter,
  calculatePlaneRotation,
  getMainSensor,
  transformPointCloud,
  estimateGroundPlane,
  getClosestPoint,
  getPointsFromVertices
} from "../../common/util"
import { createBox3dLabel, createPlaneLabel } from "./label3d_list"
import { projectionFromNDC } from "../../view_config/point_cloud"

/**
 * Handles user interactions with labels
 */
export class Label3DHandler {
  /** highlighted label */
  private _highlightedLabel: Label3D | null
  /** whether mouse is down on the selected box */
  private _mouseDownOnSelection: boolean
  /** The hashed list of keys currently down */
  private _keyDownMap: { [key: string]: boolean }
  /** viewer config */
  private _viewerConfig: ViewerConfigType
  /** index of selected item */
  private _selectedItemIndex: number
  /** Sensors that are currently in use */
  private _sensorIds: number[]
  /** Current sensor */
  private _sensor: SensorType
  /** camera */
  private _camera: THREE.Camera
  /** timer for throttling committing effects of key presses to state */
  private _keyThrottleTimer: ReturnType<typeof setTimeout> | null
  /** Whether tracking is enabled */
  private readonly _tracking: boolean
  /** Recorded state of last update */
  private _state: State

  /**
   * Constructor
   *
   * @param camera
   * @param tracking
   */
  constructor(camera: THREE.Camera, tracking: boolean) {
    this._highlightedLabel = null
    this._mouseDownOnSelection = false
    this._keyDownMap = {}
    this._viewerConfig = makePointCloudViewerConfig(-1)
    this._selectedItemIndex = -1
    this._sensorIds = []
    this._sensor = makeSensor(-1, "", DataType.POINT_CLOUD)
    this._camera = camera
    this._keyThrottleTimer = null
    this._tracking = tracking
    this._state = Session.getState()
  }

  /** Set camera */
  public set camera(camera: THREE.Camera) {
    this._camera = camera
  }

  /**
   * Get camera
   */
  public get camera(): THREE.Camera {
    return this._camera
  }

  /**
   * Update handler params when state updated
   *
   * @param state
   * @param itemIndex
   * @param viewerId
   */
  public updateState(state: State, itemIndex: number, viewerId: number): void {
    this._state = state
    this._selectedItemIndex = itemIndex
    this._viewerConfig = getCurrentViewerConfig(state, viewerId)
    this._sensorIds = Object.keys(state.task.sensors).map((key) => Number(key))
    if (this._viewerConfig.sensor in state.task.sensors) {
      this._sensor = state.task.sensors[this._viewerConfig.sensor]
    }
  }

  /**
   * Handle double click, select label for editing. Move ground plane if highlighted.
   *
   * @param x
   * @param y
   * @returns true if consumed, false otherwise
   */
  public onDoubleClick(x: number, y: number): boolean {
    const selectedLabel = Session.label3dList.selectedLabel
    const state = this._state
    const mainSensor = getMainSensor(state)
    if (
      this._highlightedLabel === null &&
      selectLabel !== null &&
      selectedLabel?.label.type === LabelTypeName.PLANE_3D
    ) {
      // Add to hist shapes for undo
      const shape = Session.label3dList.getCurrentShape()
      Session.label3dList.addShapeToHistShapes(shape)
      const planeLabel = selectedLabel as Plane3D
      // Get interception with plane
      let planeCenter = planeLabel.center
      if (mainSensor.height !== null) {
        const { up } = this.getAxes()
        planeCenter = up
          .toThree()
          .normalize()
          .multiplyScalar(-1)
          .multiplyScalar(mainSensor.height)
      }
      const plane = new THREE.Plane()
      const normal = new THREE.Vector3(0, 0, 1)
      normal.applyQuaternion(planeLabel.orientation)
      plane.setFromNormalAndCoplanarPoint(normal, planeCenter)
      const projection = projectionFromNDC(x, y, this._camera)
      const point3d = new THREE.Vector3()
      projection.intersectPlane(plane, point3d)

      // If point cloud, move to closest point
      const sensors = Object.values(state.task.sensors)
      const hasPointCloud = sensors.reduce(
        (prev: boolean, curr) => prev || curr.type === DataType.POINT_CLOUD,
        false
      )
      if (hasPointCloud) {
        const itemIndex = state.user.select.item
        const pointCloud = new THREE.Points(
          Session.pointClouds[itemIndex][mainSensor.id]
        )
        const closestPoint = getClosestPoint(pointCloud, projection)
        if (closestPoint !== null) {
          point3d.copy(closestPoint)
        }
      }
      // Move plane center
      planeLabel.move(point3d)

      commitLabels(
        [...Session.label3dList.updatedLabels.values()],
        this._tracking
      )
      return true
    } else {
      this.selectHighlighted()
      return this._highlightedLabel !== null
    }
  }

  /**
   * Fit ground plane to nearby points
   */
  private fitGroundPlane(): void {
    const range = 2 // look at points within 2m of center
    const groundPlane = Session.label3dList.getItemGroundPlane(
      this._selectedItemIndex
    )
    const state = this._state
    const mainSensor = getMainSensor(state)
    const itemIndex = state.user.select.item
    if (
      groundPlane !== null &&
      Session.pointClouds[itemIndex][mainSensor.id] !== undefined
    ) {
      const center = groundPlane.center
      const pointCloud = new THREE.Points(
        Session.pointClouds[itemIndex][mainSensor.id]
      )
      const vertices = Array.from(
        pointCloud.geometry.getAttribute("position").array
      )
      const points = getPointsFromVertices(vertices)
      const filteredPoints = points.filter((p) => p.distanceTo(center) < range)
      const estimatedPlane = estimateGroundPlane(filteredPoints)
      const newCenter = new THREE.Vector3()
      estimatedPlane.projectPoint(center, newCenter)
      const { up, left } = this.getAxes()
      const rotation = calculatePlaneRotation(
        up.toThree(),
        left.toThree(),
        estimatedPlane.normal
      )
      groundPlane.rotation.copy(new THREE.Euler().setFromVector3(rotation))
      groundPlane.move(newCenter)
      commitLabels(
        [...Session.label3dList.updatedLabels.values()],
        this._tracking
      )
    }
  }

  /**
   * Duplicate box
   */
  private duplicateBox(): void {
    const selectedLabel = Session.label3dList.selectedLabel
    const { forward } = this.getAxes()
    if (
      selectedLabel !== null &&
      selectedLabel.label.type === LabelTypeName.BOX_3D
    ) {
      const shape = selectedLabel.internalShapes()[0].toState() as CubeType
      const center = new Vector3D(
        shape.center.x,
        shape.center.y,
        shape.center.z
      ).subtract(forward)
      const dimension = new Vector3D(shape.size.x, shape.size.y, shape.size.z)
      const orientation = new Vector3D(
        shape.orientation.x,
        shape.orientation.y,
        shape.orientation.z
      )
      const label = createBox3dLabel(
        Session.label3dList,
        this._selectedItemIndex,
        this._sensorIds,
        Session.label3dList.currentCategory,
        center,
        dimension,
        orientation
      )
      commitLabels([label], this._tracking)
    }
  }

  /**
   * Process mouse down action
   *
   * @param x
   * @param y
   */
  public onMouseDown(x: number, y: number): boolean {
    if (Session.label3dList.control.highlighted) {
      this._mouseDownOnSelection = true
      if (Session.label3dList.selectedLabel !== null) {
        const shape = Session.label3dList.getCurrentShape()
        Session.label3dList.addShapeToHistShapes(shape)
      }
      Session.label3dList.control.onMouseDown(this._camera)
      return false
    }

    if (this._highlightedLabel !== null) {
      const consumed = this._highlightedLabel.onMouseDown(x, y, this._camera)
      if (consumed) {
        this._mouseDownOnSelection = true
        if (Session.label3dList.selectedLabel !== null) {
          const shape = Session.label3dList.getCurrentShape()
          Session.label3dList.addShapeToHistShapes(shape)
        }
        return false
      }
    }

    return false
  }

  /**
   * Process mouse up action
   */
  public onMouseUp(): boolean {
    let consumed = false
    if (Session.label3dList.control.visible) {
      consumed = Session.label3dList.control.onMouseUp()
    }
    if (!consumed && Session.label3dList.selectedLabel !== null) {
      Session.label3dList.selectedLabel.onMouseUp()
    }
    commitLabels(
      [...Session.label3dList.updatedLabels.values()],
      this._tracking
    )
    Session.label3dList.clearUpdatedLabels()
    // Set current label as selected label
    if (
      this._mouseDownOnSelection &&
      this._highlightedLabel !== Session.label3dList.selectedLabel
    ) {
      this.selectHighlighted()
    }
    this._mouseDownOnSelection = false
    return false
  }

  /**
   * Process mouse move action
   *
   * @param x NDC
   * @param y NDC
   * @param camera
   * @param raycastIntersection
   */
  public onMouseMove(
    x: number,
    y: number,
    raycastIntersection?: THREE.Intersection
  ): boolean {
    if (
      this._mouseDownOnSelection &&
      Session.label3dList.selectedLabel !== null &&
      Session.label3dList.control.visible &&
      Session.label3dList.control.highlighted
    ) {
      const consumed = Session.label3dList.control.onMouseMove(
        x,
        y,
        this._camera
      )
      if (consumed) {
        return true
      }
    }
    if (this._mouseDownOnSelection && this._highlightedLabel !== null) {
      this._highlightedLabel.onMouseMove(x, y, this._camera)
      return true
    } else {
      this.highlight(raycastIntersection)
    }

    return false
  }

  /**
   * Get axes for viewer type. Returns axes in order: up, forward, right
   */
  private getAxes(): { up: Vector3D; forward: Vector3D; left: Vector3D } {
    const mainSensor = getMainSensor(this._state)
    const forward = new Vector3D().fromThree(mainSensor.forward).normalize()
    const up = new Vector3D().fromThree(mainSensor.up).normalize()
    const left = up.clone().cross(forward).normalize()
    return {
      up,
      forward,
      left
    }
  }

  /**
   * Calculate box span rotation, given box, plane, and viewer type
   *
   * @param boxSpan
   * @param groundPlane
   */
  private getBoxSpanRotation(
    boxSpan: Span3D,
    groundPlane: Plane3D
  ): THREE.Vector3 {
    const { up, left } = this.getAxes()
    const rotation = groundPlane.orientation.clone()
    const defaultPlaneForward = new THREE.Vector3(0, 0, 1)
    const planePitch = up.toThree().angleTo(defaultPlaneForward)
    const planePitchEuler = new THREE.Euler().setFromVector3(
      left.toThree().clone().normalize().multiplyScalar(planePitch)
    )
    rotation.multiply(new THREE.Quaternion().setFromEuler(planePitchEuler))

    const leftOnPlane = left.toThree().applyQuaternion(rotation)
    const sideVec = boxSpan.sideEdge.clone().normalize()
    const yaw = Math.PI / 2 - leftOnPlane.angleTo(sideVec)
    const yawVec = up.clone().toThree().multiplyScalar(yaw)
    const yawEuler = new THREE.Euler().setFromVector3(yawVec)
    const yawRotation = new THREE.Quaternion().setFromEuler(yawEuler)
    rotation.multiply(yawRotation)

    return new THREE.Euler().setFromQuaternion(rotation).toVector3()
  }

  /**
   * Create new box3D label
   */
  private createLabel(): void {
    const center = new Vector3D()
    if (this._viewerConfig.type === ViewerConfigTypeName.POINT_CLOUD) {
      center.fromState(
        (this._viewerConfig as PointCloudViewerConfigType).target
      )
    } else {
      center.copy(new Vector3D(0, 0, 10))
      if (this._sensor.extrinsics !== undefined) {
        const worldDirection = new THREE.Vector3()
        this._camera.getWorldDirection(worldDirection)
        worldDirection.normalize()
        worldDirection.multiplyScalar(5)
        center.fromState(this._sensor.extrinsics.translation)
        center.add(new Vector3D().fromThree(worldDirection))
      }
    }
    const dimension = new Vector3D(1, 1, 1)
    const orientation = new Vector3D()
    const boxSpan = this._state.session.info3D.boxSpan
    const groundPlane = Session.label3dList.getItemGroundPlane(
      this._selectedItemIndex
    )
    if (boxSpan?.complete === true && groundPlane !== null) {
      const { up, forward, left } = this.getAxes()
      center.fromThree(boxSpan.center)
      dimension.fromThree(boxSpan.dimensions(up, forward, left))
      orientation.fromThree(this.getBoxSpanRotation(boxSpan, groundPlane))
      Session.dispatch(deactivateSpan())
    }

    const label = createBox3dLabel(
      Session.label3dList,
      this._selectedItemIndex,
      this._sensorIds,
      Session.label3dList.currentCategory,
      center,
      dimension,
      orientation
    )
    commitLabels([label], this._tracking)

    alert(Severity.SUCCESS, "Box successfully created")
  }

  /**
   * Toggle selecting ground plane.
   */
  private toggleGroundPlane(): void {
    const groundPlane = Session.label3dList.getItemGroundPlane(
      this._selectedItemIndex
    )
    if (groundPlane !== null) {
      const selectedLabel = Session.label3dList.selectedLabel
      const groundPlaneSelected = selectedLabel?.labelId === groundPlane.labelId
      if (groundPlaneSelected) {
        Session.dispatch(
          selectLabel(Session.label3dList.selectedLabelIds, -1, INVALID_ID)
        )
      } else {
        Session.dispatch(
          selectLabel(
            Session.label3dList.selectedLabelIds,
            this._selectedItemIndex,
            groundPlane.labelId
          )
        )
      }
    } else {
      const center = new Vector3D(0, 1.5, 10)
      const label = createPlaneLabel(
        Session.label3dList,
        this._selectedItemIndex,
        Session.label3dList.currentCategory,
        center,
        undefined,
        this._sensorIds
      )
      commitLabels([label], this._tracking)
    }
  }

  /**
   * Change heading of 3d box, if currently selected
   */
  private changeBoxHeading(): void {
    const { up } = this.getAxes()
    const label = Session.label3dList.selectedLabel
    if (label !== null && label.type === LabelTypeName.BOX_3D) {
      const shape = Session.label3dList.getCurrentShape() as CubeType
      Session.label3dList.addShapeToHistShapes(shape)

      // Swap width and length
      const newShape: CubeType = { ...shape }
      const size = shape.size
      newShape.size =
        up.y !== 0
          ? { x: size.z, y: size.y, z: size.x }
          : { x: size.y, y: size.x, z: size.z }

      label.setShape(newShape)

      // Rotate box
      const normal = up.toThree()
      normal.applyQuaternion(label.orientation)
      const rotation = new THREE.Quaternion().setFromAxisAngle(
        normal,
        Math.PI / 2
      )
      label.rotate(rotation)
      commitLabels(
        [...Session.label3dList.updatedLabels.values()],
        this._tracking
      )
    }
  }

  /**
   * Create ground plane
   *
   * @param itemIndex
   */
  public createGroundPlane(itemIndex: number): void {
    const state = this._state
    const mainSensor = getMainSensor(state)
    const item = state.task.items[itemIndex]
    const isLoaded = isItemLoaded(state, item.index)
    const hasGroundPlane =
      Object.values(item.labels).filter(
        (label) => label.type === LabelTypeName.PLANE_3D
      ).length > 0
    const config = this._viewerConfig as PointCloudViewerConfigType
    const sensorIdx = config.sensor
    const sensor = state.task.sensors[sensorIdx]
    const isPointCloud = sensor.type === DataType.POINT_CLOUD
    if (isLoaded && !hasGroundPlane) {
      // estimate ground plane
      let center = new THREE.Vector3(0, mainSensor.height ?? 1.5, 10)
      let rotation = new THREE.Vector3(Math.PI / 2, 0, 0)
      if (isPointCloud) {
        const rawGeometry = Session.pointClouds[item.index][sensorIdx]
        const geometry = transformPointCloud(rawGeometry, sensorIdx, state)
        const pointCloud = Array.from(
          new THREE.Points(geometry).geometry.getAttribute("position").array
        )
        const points = getPointsFromVertices(pointCloud)
        const estimatedPlane = estimateGroundPlane(points)
        const target = new Vector3D().fromState(config.target)
        const { up, left } = this.getAxes()
        const down = up.toThree().clone().multiplyScalar(-1)
        center = calculatePlaneCenter(estimatedPlane, target.toThree(), down)
        rotation = calculatePlaneRotation(
          up.toThree(),
          left.toThree(),
          estimatedPlane.normal
        )
      }
      const label = createPlaneLabel(
        Session.label3dList,
        item.index,
        Session.label3dList.currentCategory,
        new Vector3D().fromThree(center),
        new Vector3D().fromThree(rotation),
        Object.keys(state.task.sensors).map((key) => Number(key))
      )
      commitLabels([label], this._tracking)
    }
  }

  /**
   * Handle keyboard events
   *
   * @param {KeyboardEvent} e
   * @returns true if consumed, false otherwise
   */
  public onKeyDown(e: KeyboardEvent): boolean {
    const state = Session.getState()
    // TODO: break the cases into functions
    switch (e.key) {
      case Key.G_UP:
      case Key.G_LOW: {
        this.toggleGroundPlane()
        return true
      }
      case Key.C_UP:
      case Key.C_LOW: {
        this.duplicateBox()
        return true
      }
      case Key.P_UP:
      case Key.P_LOW: {
        this.fitGroundPlane()
        return true
      }
      case Key.SPACE: {
        if (
          state.session.info3D.boxSpan !== null &&
          !state.session.info3D.boxSpan.complete
        ) {
          break
        } else if (!state.session.info3D.isBoxSpan) {
          Session.dispatch(activateSpan())
          return true
        } else {
          this.createLabel()
          return true
        }
      }
      case Key.BACKSPACE:
        if (state.session.info3D.isBoxSpan) {
          Session.dispatch(deactivateSpan())
        }
        return true
      case Key.ESCAPE: {
        Session.dispatch(
          selectLabel(Session.label3dList.selectedLabelIds, -1, INVALID_ID)
        )
        const boxSpan = state.session.info3D.boxSpan
        if (state.session.info3D.isBoxSpan && boxSpan !== null) {
          if (boxSpan.numPoints > 0) {
            Session.dispatch(resetSpan())
          } else {
            Session.dispatch(deactivateSpan())
          }
        }
        return true
      }
      case Key.ENTER:
        Session.dispatch(
          selectLabel(Session.label3dList.selectedLabelIds, -1, INVALID_ID)
        )
        return true
      case Key.N_UP:
      case Key.N_LOW:
        this.createLabel()
        return true
      case Key.B_UP:
      case Key.B_LOW:
        Session.dispatch(selectLabel3dType(LabelTypeName.BOX_3D))
        return true
      case Key.T_UP:
      case Key.T_LOW:
        if (this.isKeyDown(Key.SHIFT)) {
          if (Session.label3dList.selectedLabel !== null) {
            const shape = Session.label3dList.getCurrentShape()
            Session.label3dList.addShapeToHistShapes(shape)

            const target = (this._viewerConfig as PointCloudViewerConfigType)
              .target
            Session.label3dList.selectedLabel.move(
              new Vector3D().fromState(target).toThree()
            )
            commitLabels(
              [...Session.label3dList.updatedLabels.values()],
              this._tracking
            )
            Session.label3dList.clearUpdatedLabels()
          }
        }
        break
      // Change box heading
      case Key.F_UP:
      case Key.F_LOW: {
        this.changeBoxHeading()
        break
      }
      case Key.Z_UP:
      case Key.Z_LOW:
        if (this.isKeyDown(Key.META) || this.isKeyDown(Key.CONTROL)) {
          const shape = Session.label3dList.getLastShape()
          if (shape !== null && Session.label3dList.selectedLabel !== null) {
            Session.label3dList.selectedLabel.setShape(shape)
            commitLabels(
              [...Session.label3dList.updatedLabels.values()],
              this._tracking
            )
          }
        }
        break
      case Key.Q_LOW:
        if (state.session.info3D.isBoxSpan) {
          Session.dispatch(pauseSpan())
        } else if (state.session.info3D.boxSpan !== null) {
          Session.dispatch(resumeSpan())
        }
        return true
      case Key.U_LOW:
        if (state.session.info3D.isBoxSpan) {
          Session.dispatch(undoSpan())
        }
        break
      default:
        break
    }
    if (Session.label3dList.selectedLabel !== null && !this.isKeyDown(e.key)) {
      const consumed = Session.label3dList.control.onKeyDown(e, this._camera)
      if (consumed) {
        const shape = Session.label3dList.getCurrentShape()
        Session.label3dList.addShapeToHistShapes(shape)

        this._keyDownMap[e.key] = true
        if (this._keyThrottleTimer !== null) {
          window.clearTimeout(this._keyThrottleTimer)
        }
        this.timedRepeat(() => {
          Session.label3dList.control.onKeyDown(e, this._camera)
          Session.label3dList.onDrawableUpdate()
        }, e.key)
        return true
      }
    }
    this._keyDownMap[e.key] = true
    return false
  }

  /**
   * Handle key up
   *
   * @param e
   */
  public onKeyUp(e: KeyboardEvent): boolean {
    // TODO: make _keyDownMap a Map
    // eslint-disable-next-line @typescript-eslint/no-dynamic-delete
    delete this._keyDownMap[e.key]
    if (Session.label3dList.selectedLabel !== null) {
      Session.label3dList.control.onKeyUp(e)
    }
    this._keyThrottleTimer = setTimeout(() => {
      if (Session.label3dList.updatedLabels.size > 0) {
        commitLabels(
          [...Session.label3dList.updatedLabels.values()],
          this._tracking
        )
        Session.label3dList.clearUpdatedLabels()
      }
    }, 200)
    return false
  }

  /**
   * Highlight label if ray from mouse is intersecting a label
   *
   * @param object
   * @param point
   * @param intersection
   */
  private highlight(intersection?: THREE.Intersection): void {
    if (this._highlightedLabel !== null) {
      this._highlightedLabel.setHighlighted()
    }
    this._highlightedLabel = null

    if (intersection !== undefined) {
      const object = intersection.object
      const label = Session.label3dList.getLabelFromRaycastedObject3D(object)

      if (label !== null) {
        label.setHighlighted(intersection)
        this._highlightedLabel = label
      }
    }
    Session.label3dList.control.setHighlighted(intersection)
  }

  /**
   * Whether a specific key is pressed down
   *
   * @param {string} key - the key to check
   * @return {boolean}
   */
  private isKeyDown(key: string): boolean {
    return this._keyDownMap[key]
  }

  /**
   * Select highlighted label
   */
  private selectHighlighted(): void {
    if (this._highlightedLabel !== null) {
      if (
        (this.isKeyDown(Key.CONTROL) || this.isKeyDown(Key.META)) &&
        this._highlightedLabel !== Session.label3dList.selectedLabel
      ) {
        Session.dispatch(
          selectLabel(
            Session.label3dList.selectedLabelIds,
            this._selectedItemIndex,
            this._highlightedLabel.labelId,
            this._highlightedLabel.category[0],
            this._highlightedLabel.attributes,
            true
          )
        )
      } else {
        Session.dispatch(
          selectLabel(
            Session.label3dList.selectedLabelIds,
            this._selectedItemIndex,
            this._highlightedLabel.labelId,
            this._highlightedLabel.category[0],
            this._highlightedLabel.attributes
          )
        )
      }
    }
  }

  /**
   * Repeat function as long as key is held down
   *
   * @param fn
   * @param key
   * @param timeout
   */
  private timedRepeat(fn: () => void, key: string, timeout: number = 30): void {
    if (this.isKeyDown(key)) {
      fn()
      setTimeout(() => this.timedRepeat(fn, key, timeout), timeout)
    }
  }
}
