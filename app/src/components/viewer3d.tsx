import { Box, IconButton } from "@material-ui/core"
import Tooltip from "@mui/material/Tooltip"
import Fade from "@mui/material/Fade"
import ImportExportIcon from "@material-ui/icons/ImportExport"
import LockIcon from "@material-ui/icons/Lock"
import SyncIcon from "@material-ui/icons/Sync"
import ThreeDRotationSharpIcon from "@material-ui/icons/ThreeDRotationSharp"
import ThreeSixtyIcon from "@material-ui/icons/ThreeSixty"
import TripOriginIcon from "@material-ui/icons/TripOrigin"
import ColorLensIcon from "@material-ui/icons/ColorLens"
import { withStyles } from "@material-ui/styles"
import React from "react"
import * as THREE from "three"

import { changeViewerConfig, toggleSynchronization } from "../action/common"
import {
  alignToAxis,
  CameraLockState,
  CameraMovementParameters,
  lockedToSelection,
  moveCameraAndTarget,
  toggleRotation,
  updateLockStatus
} from "../action/point_cloud"
import Session from "../common/session"
import * as types from "../const/common"
import { Vector3D } from "../math/vector3d"
import { viewerStyles } from "../styles/viewer"
import {
  ColorSchemeType,
  PointCloudViewerConfigType,
  State
} from "../types/state"
import {
  DrawableViewer,
  ViewerClassTypes,
  ViewerProps
} from "./drawable_viewer"
import Label3dCanvas from "./label3d_canvas"
import PointCloudCanvas from "./point_cloud_canvas"
import Tag3dCanvas from "./tag_3d_canvas"
import { isCurrentItemLoaded } from "../functional/state_util"
import { getMainSensor } from "../common/util"
import { Sensor } from "../common/sensor"

interface ClassType extends ViewerClassTypes {
  /** camera z lock */
  camera_y_lock_icon: string
  /** camera x lock */
  camera_x_lock_icon: string
  /** button */
  viewer_button: string
}

export interface Props extends ViewerProps {
  /** classes */
  classes: ClassType
  /** id of the viewer, for referencing viewer config in state */
  id: number
}

/**
 * Conditionally wrap box with bottom border
 *
 * @param element
 * @param underline
 */
function underlineElement(
  element: React.ReactElement,
  underline: boolean = false
): React.ReactElement {
  if (underline) {
    return (
      <Box borderBottom={1} borderColor="grey.500">
        {element}
      </Box>
    )
  }
  return element
}

/**
 * Calculate forward vector
 *
 * @param viewerConfig
 * @param state
 */
function calculateForward(
  viewerConfig: PointCloudViewerConfigType,
  state: State
): THREE.Vector3 {
  const mainSensor = getMainSensor(state)
  const up = mainSensor.up
  const origin = new THREE.Vector3(0, 0, 0)
  const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(up, origin)
  const target = new THREE.Vector3(
    viewerConfig.target.x,
    viewerConfig.target.y,
    viewerConfig.target.z
  )
  const camPosition = new THREE.Vector3(
    viewerConfig.position.x,
    viewerConfig.position.y,
    viewerConfig.position.z
  )
  const camDirection = target.clone().sub(camPosition)
  const forward = new THREE.Vector3()
  plane.projectPoint(camDirection, forward)
  forward.normalize().multiplyScalar(CameraMovementParameters.MOVE_AMOUNT)
  return forward
}

/**
 * Calculate left vector
 *
 * @param viewerConfig
 * @param forward
 * @param state
 */
function calculateLeft(forward: THREE.Vector3, state: State): THREE.Vector3 {
  // Get vector pointing up
  const mainSensor = getMainSensor(state)
  const vertical = mainSensor.up

  // Handle movement in three dimensions
  const left = new THREE.Vector3()
  left.crossVectors(vertical, forward)
  left.normalize()
  left.multiplyScalar(CameraMovementParameters.MOVE_AMOUNT)
  return left
}

/**
 * Viewer for images and 2d labels
 */
class Viewer3D extends DrawableViewer<Props> {
  /** Camera for math */
  private _camera: THREE.PerspectiveCamera
  /** Raycaster */
  private _raycaster: THREE.Raycaster
  /** Target position */
  private readonly _target: THREE.Vector3
  /** Current point cloud */
  private _pointCloud: THREE.Points | null
  /** Timer for scroll function */
  private _scrollTimer: ReturnType<typeof setTimeout> | null
  /** Whether the camera is being moved */
  private _movingCamera: boolean
  /** Whether the camera has been transformed */
  private cameraTransformed: boolean

  /**
   * Constructor
   *
   * @param {Object} props: react props
   * @param props
   */
  constructor(props: Props) {
    super(props)
    this._camera = new THREE.PerspectiveCamera(45, 1, 0.2, 1000)
    this._raycaster = new THREE.Raycaster()
    this._target = new THREE.Vector3()
    this._pointCloud = null
    this._scrollTimer = null
    this._movingCamera = false
    this.cameraTransformed = false
  }

  /** Called when component updates */
  public componentDidUpdate(): void {
    const state = this.state
    if (!isCurrentItemLoaded(state)) {
      return
    }
    if (this._viewerConfig !== undefined) {
      this.updateCamera(this._viewerConfig as PointCloudViewerConfigType)

      if (Session.activeViewerId === this.props.id) {
        Session.label3dList.setActiveCamera(this._camera)
      }

      this._camera.layers.set(this.props.id)
      const item = this.state.user.select.item
      const sensor = this.state.user.viewerConfigs[this.props.id].sensor
      this._pointCloud = new THREE.Points(Session.pointClouds[item][sensor])
    }
  }

  /**
   * Render function
   *
   * @return {React.Fragment} React fragment
   */
  protected getDrawableComponents(): React.ReactElement[] {
    const views: React.ReactElement[] = []
    if (this._viewerConfig !== undefined) {
      views.push(
        <PointCloudCanvas
          key={`pointCloudCanvas${this.props.id}`}
          display={this._container}
          id={this.props.id}
          camera={this._camera}
        />
      )
      views.push(
        <Label3dCanvas
          key={`label3dCanvas${this.props.id}`}
          display={this._container}
          id={this.props.id}
          camera={this._camera}
        />
      )

      views.push(
        <Tag3dCanvas
          key={`tag3dCanvas${this.props.id}`}
          display={this._container}
          id={this.props.id}
          camera={this._camera}
        />
      )
    }

    return views
  }

  /** Return menu buttons */
  protected getMenuComponents(): [] | JSX.Element[] {
    if (this._viewerConfig !== undefined) {
      const config = this._viewerConfig as PointCloudViewerConfigType

      return [
        this.getYLockButton(config),
        this.getXLockButton(config),
        this.getXAxisButton(config),
        this.getYAxisButton(config),
        this.getZAxisButton(config),
        this.getFlipButton(config),
        this.getSynchronizationButton(config),
        this.getSelectionLockButton(config),
        this.getOriginButton(config),
        this.getViewerConfigButton(config),
        this.getColorSchemeButton(config)
      ]
    }

    return []
  }

  /**
   * y-axis lock button
   *
   * @param config
   */
  protected getYLockButton(config: PointCloudViewerConfigType): JSX.Element {
    return (
      <Tooltip
        key={`yLockButton${this.props.id}`}
        title="Lock Y-axis"
        enterDelay={500}
        TransitionComponent={Fade}
        TransitionProps={{ timeout: 600 }}
        arrow
      >
        <IconButton
          onClick={() => this.toggleCameraLock(CameraLockState.Y_LOCKED)}
          className={this.props.classes.viewer_button}
        >
          {underlineElement(
            <div className={this.props.classes.camera_y_lock_icon}>
              <ThreeSixtyIcon />
            </div>,
            config.lockStatus === CameraLockState.Y_LOCKED
          )}
        </IconButton>
      </Tooltip>
    )
  }

  /**
   * x-axis lock button
   *
   * @param config
   */
  protected getXLockButton(config: PointCloudViewerConfigType): JSX.Element {
    return (
      <Tooltip
        key={`xLockButton${this.props.id}`}
        title="Lock X-axis"
        enterDelay={500}
        TransitionComponent={Fade}
        TransitionProps={{ timeout: 600 }}
        arrow
      >
        <IconButton
          onClick={() => this.toggleCameraLock(CameraLockState.X_LOCKED)}
          className={this.props.classes.viewer_button}
          edge={"start"}
        >
          {underlineElement(
            <div className={this.props.classes.camera_x_lock_icon}>
              <ThreeSixtyIcon />
            </div>,
            config.lockStatus === CameraLockState.X_LOCKED
          )}
        </IconButton>
      </Tooltip>
    )
  }

  /**
   * snap to x-axis button
   *
   * @param config
   */
  protected getXAxisButton(config: PointCloudViewerConfigType): JSX.Element {
    return (
      <Tooltip
        key={`xAxisButton${this.props.id}`}
        title="Snap to X-axis"
        enterDelay={500}
        TransitionComponent={Fade}
        TransitionProps={{ timeout: 600 }}
        arrow
      >
        <IconButton
          className={this.props.classes.viewer_button}
          onClick={() =>
            Session.dispatch(alignToAxis(this.props.id, config, 0))
          }
          edge={"start"}
        >
          {underlineElement(
            <span style={{ color: "#ff7700" }}>X</span>,
            config.lockStatus === CameraLockState.SELECTION_X
          )}
        </IconButton>
      </Tooltip>
    )
  }

  /**
   * snap to x-axis button
   *
   * @param config
   */
  protected getYAxisButton(config: PointCloudViewerConfigType): JSX.Element {
    return (
      <Tooltip
        key={`yAxisButton${this.props.id}`}
        title="Snap to Y-axis"
        enterDelay={500}
        TransitionComponent={Fade}
        TransitionProps={{ timeout: 600 }}
        arrow
      >
        <IconButton
          className={this.props.classes.viewer_button}
          onClick={() =>
            Session.dispatch(alignToAxis(this.props.id, config, 1))
          }
        >
          {underlineElement(
            <span style={{ color: "#77ff77" }}>Y</span>,
            config.lockStatus === CameraLockState.SELECTION_Y
          )}
        </IconButton>
      </Tooltip>
    )
  }

  /**
   * snap to x-axis button
   *
   * @param config
   */
  protected getZAxisButton(config: PointCloudViewerConfigType): JSX.Element {
    return (
      <Tooltip
        key={`zAxis${this.props.id}`}
        title="Snap to Z-axis"
        enterDelay={500}
        TransitionComponent={Fade}
        TransitionProps={{ timeout: 600 }}
        arrow
      >
        <IconButton
          className={this.props.classes.viewer_button}
          onClick={() =>
            Session.dispatch(alignToAxis(this.props.id, config, 2))
          }
        >
          {underlineElement(
            <span style={{ color: "#0088ff" }}>Z</span>,
            config.lockStatus === CameraLockState.SELECTION_Z
          )}
        </IconButton>
      </Tooltip>
    )
  }

  /**
   * flip button
   *
   * @param config
   */
  protected getFlipButton(config: PointCloudViewerConfigType): JSX.Element {
    return (
      <Tooltip
        key={`flipButton${this.props.id}`}
        title="Flip"
        enterDelay={500}
        TransitionComponent={Fade}
        TransitionProps={{ timeout: 600 }}
        arrow
      >
        <IconButton
          className={this.props.classes.viewer_button}
          onClick={() => {
            const newConfig = {
              ...config,
              flipAxis: !config.flipAxis
            }
            Session.dispatch(changeViewerConfig(this.props.id, newConfig))
          }}
        >
          {underlineElement(<ImportExportIcon />, config.flipAxis)}
        </IconButton>
      </Tooltip>
    )
  }

  /**
   * synchronize views button
   *
   * @param config
   */
  protected getSynchronizationButton(
    config: PointCloudViewerConfigType
  ): JSX.Element {
    return (
      <Tooltip
        key={`synchronizationButton${this.props.id}`}
        title="Synchronize views"
        enterDelay={500}
        TransitionComponent={Fade}
        TransitionProps={{ timeout: 600 }}
        arrow
      >
        <IconButton
          className={this.props.classes.viewer_button}
          onClick={() => {
            Session.dispatch(toggleSynchronization(this._viewerId, config))
          }}
        >
          {underlineElement(<SyncIcon />, config.synchronized)}
        </IconButton>
      </Tooltip>
    )
  }

  /**
   * lock selection button
   *
   * @param config
   */
  protected getSelectionLockButton(
    config: PointCloudViewerConfigType
  ): JSX.Element {
    return (
      <Tooltip
        key={`selectionLockButton${this.props.id}`}
        title="Lock selection"
        enterDelay={500}
        TransitionComponent={Fade}
        TransitionProps={{ timeout: 600 }}
        arrow
      >
        <IconButton
          className={this.props.classes.viewer_button}
          onClick={() => {
            Session.dispatch(toggleRotation(this._viewerId, config))
          }}
        >
          {underlineElement(<LockIcon />, lockedToSelection(config))}
        </IconButton>
      </Tooltip>
    )
  }

  /**
   * view origin button
   *
   * @param config
   */
  protected getOriginButton(config: PointCloudViewerConfigType): JSX.Element {
    return (
      <Tooltip
        key={`originButton${this.props.id}`}
        title="View origin"
        enterDelay={500}
        TransitionComponent={Fade}
        TransitionProps={{ timeout: 600 }}
        arrow
      >
        <IconButton
          className={this.props.classes.viewer_button}
          onClick={() => {
            const newConfig = {
              ...config,
              position: { x: 0, y: 0, z: 0 },
              target: { x: 10, y: 0, z: 0 }
            }
            Session.dispatch(changeViewerConfig(this.props.id, newConfig))
          }}
        >
          <TripOriginIcon />
        </IconButton>
      </Tooltip>
    )
  }

  /**
   * viewer config button
   *
   * @param config
   */
  protected getViewerConfigButton(
    config: PointCloudViewerConfigType
  ): JSX.Element {
    return (
      <Tooltip
        key={`viewerConfigButton${this.props.id}`}
        title="Viewer config"
        enterDelay={500}
        TransitionComponent={Fade}
        TransitionProps={{ timeout: 600 }}
        arrow
      >
        <IconButton
          className={this.props.classes.viewer_button}
          onClick={() => {
            Session.dispatch(toggleRotation(this.props.id, config))
          }}
          edge={"start"}
        >
          {underlineElement(
            <ThreeDRotationSharpIcon />,
            config.cameraRotateDir === true
          )}
        </IconButton>
      </Tooltip>
    )
  }

  /**
   * color scheme button
   *
   * @param config
   */
  protected getColorSchemeButton(
    config: PointCloudViewerConfigType
  ): JSX.Element {
    return (
      <Tooltip
        key={`colorSchemeButton${this.props.id}`}
        title="Color scheme"
        enterDelay={500}
        TransitionComponent={Fade}
        TransitionProps={{ timeout: 600 }}
        arrow
      >
        <IconButton
          className={this.props.classes.viewer_button}
          onClick={() => {
            const colorSchemes = Object.values(ColorSchemeType)
            const itemType = this.state.task.config.itemType
            if (itemType !== types.ItemTypeName.IMAGE) {
              const imageIndex = colorSchemes.indexOf(ColorSchemeType.IMAGE)
              colorSchemes.splice(imageIndex, 1)
            }
            const currentIndex = colorSchemes.indexOf(config.colorScheme)
            const nextIndex = (currentIndex + 1) % colorSchemes.length
            const newConfig = {
              ...config,
              colorScheme: colorSchemes[nextIndex]
            }
            Session.dispatch(changeViewerConfig(this.props.id, newConfig))
          }}
          edge={"start"}
        >
          <ColorLensIcon />
        </IconButton>
      </Tooltip>
    )
  }

  /**
   * Mouse enter
   *
   * @param e
   */
  protected onMouseEnter(e: React.MouseEvent): void {
    super.onMouseEnter(e)
    const state = this.state
    if (!isCurrentItemLoaded(state)) {
      return
    }
    Session.label3dList.setActiveCamera(this._camera)
  }

  /**
   * Handle mouse move
   *
   * @param e
   */
  protected onMouseMove(e: React.MouseEvent): void {
    const oldX = this._mX
    const oldY = this._mY
    super.onMouseMove(e)
    const dX = this._mX - oldX
    const dY = this._mY - oldY
    if (this._mouseDown && this._viewerConfig !== undefined) {
      this._movingCamera = true
      const lockSelection = lockedToSelection(
        this._viewerConfig as PointCloudViewerConfigType
      )
      if (this._mouseButton === 2) {
        const delta = this.dragCamera(dX, dY)
        if (lockSelection && Session.label3dList.selectedLabel !== null) {
          Session.label3dList.selectedLabel.translate(delta)
        }
      } else {
        if (lockSelection && Session.label3dList.selectedLabel !== null) {
          if (
            this.isKeyDown(types.Key.CONTROL) ||
            this.isKeyDown(types.Key.META)
          ) {
            const quaternion = this.rotateCameraViewDirection(dX)
            Session.label3dList.selectedLabel.rotate(quaternion)
          }
        } else {
          this.rotateCameraSpherical(dX, dY)
        }

        Session.label3dList.onDrawableUpdate()
      }
    }
  }

  /**
   * Handle mouse up
   *
   * @param e
   */
  protected onMouseUp(e: React.MouseEvent): void {
    super.onMouseUp(e)
    this._movingCamera = false
    this.commitCamera()
  }

  /**
   * Handle double click
   *
   * @param e
   */
  protected onDoubleClick(e: React.MouseEvent): void {
    if (this._container === undefined || this._viewerConfig === undefined) {
      return
    }

    const normalized = this.normalizeCoordinates(e.clientX, e.clientY)

    const NDC = this.convertMouseToNDC(normalized[0], normalized[1])

    this._raycaster.params = {
      ...this._raycaster.params,
      Line: { threshold: 0.2 }
    }
    this._raycaster.setFromCamera(
      new THREE.Vector2(NDC[0], NDC[1]),
      this._camera
    )

    if (this._pointCloud !== null) {
      const intersects = this._raycaster.intersectObject(this._pointCloud)

      if (intersects.length > 0) {
        const newTarget = intersects[0].point
        const viewerConfig = this._viewerConfig as PointCloudViewerConfigType
        Session.dispatch(
          moveCameraAndTarget(
            new Vector3D(
              viewerConfig.position.x - viewerConfig.target.x + newTarget.x,
              viewerConfig.position.y - viewerConfig.target.y + newTarget.y,
              viewerConfig.position.z - viewerConfig.target.z + newTarget.z
            ),
            new Vector3D(newTarget.x, newTarget.y, newTarget.z),
            this._viewerId,
            viewerConfig
          )
        )
      }
    }
  }

  /**
   * Handle mouse leave
   *
   * @param e
   */
  protected onMouseLeave(): void {}

  /**
   * Handle mouse wheel
   *
   * @param e
   */
  protected onWheel(e: WheelEvent): void {
    e.preventDefault()
    if (this._scrollTimer !== null) {
      window.clearTimeout(this._scrollTimer)
    }
    this.zoomCamera(e.deltaY)
    this._movingCamera = true
    this._scrollTimer = setTimeout(() => {
      this._movingCamera = false
      this.commitCamera()
    }, 30)
  }

  /**
   * Override key handler
   *
   * @param e
   */
  protected onKeyDown(e: KeyboardEvent): void {
    if (this.isKeyDown(e.key)) {
      return
    }

    super.onKeyDown(e)
    if (Session.activeViewerId !== this.props.id) {
      return
    }

    switch (e.key) {
      case types.Key.PERIOD:
        this._movingCamera = true
        this.timedRepeat(this.moveUp.bind(this), e.key)
        break
      case types.Key.SLASH:
        this._movingCamera = true
        this.timedRepeat(this.moveDown.bind(this), e.key)
        break
      case types.Key.S_LOW:
      case types.Key.S_UP:
        this._movingCamera = true
        this.timedRepeat(this.moveBackward.bind(this), e.key)
        break
      case types.Key.W_LOW:
      case types.Key.W_UP:
        this._movingCamera = true
        this.timedRepeat(this.moveForward.bind(this), e.key)
        break
      case types.Key.A_LOW:
      case types.Key.A_UP:
        this._movingCamera = true
        this.timedRepeat(this.moveLeft.bind(this), e.key)
        break
      case types.Key.D_LOW:
      case types.Key.D_UP:
        this._movingCamera = true
        this.timedRepeat(this.moveRight.bind(this), e.key)
        break
    }
  }

  /**
   * Override on key up
   *
   * @param e
   */
  protected onKeyUp(e: KeyboardEvent): void {
    if (this.isKeyDown(e.key) && Session.activeViewerId === this.props.id) {
      if (this._movingCamera) {
        this._movingCamera = false
        this.commitCamera()
      }
    }
    super.onKeyUp(e)
  }

  /**
   * update camera parameters with config
   *
   * @param config
   */
  private updateCamera(config: PointCloudViewerConfigType): void {
    if (this._viewerConfig === undefined || this._movingCamera) {
      return
    }

    this._target.set(config.target.x, config.target.y, config.target.z)
    const mainSensor = getMainSensor(this.state)
    this.cameraTransformed = config.cameraTransformed

    if (
      lockedToSelection(config) &&
      Session.label3dList.selectedLabel !== null
    ) {
      const up = new THREE.Vector3()
      const forward = new THREE.Vector3()
      const lockStatus = config.lockStatus

      switch (lockStatus) {
        case CameraLockState.SELECTION_X:
          up.z = 1
          forward.x = 1
          break
        case CameraLockState.SELECTION_Y:
          up.z = 1
          forward.y = 1
          break
        case CameraLockState.SELECTION_Z:
          up.x = 1
          forward.z = 1
          break
      }

      if (config.flipAxis) {
        forward.multiplyScalar(-1)
      }

      const position = new Vector3D().fromState(config.position).toThree()
      const target = new Vector3D().fromState(config.target).toThree()

      const offset = new THREE.Vector3().copy(position)
      offset.sub(target)

      target.copy(Session.label3dList.selectedLabel.center)
      this._target.copy(target)

      up.applyQuaternion(Session.label3dList.selectedLabel.orientation)
      forward.applyQuaternion(Session.label3dList.selectedLabel.orientation)
      forward.multiplyScalar(offset.length())

      this._camera.position.copy(target)
      this._camera.position.add(forward)
      this._camera.up = up
      this._camera.lookAt(target)
    } else {
      this._camera.up.copy(mainSensor.up)
      this._camera.position.x = config.position.x
      this._camera.position.y = config.position.y
      this._camera.position.z = config.position.z

      if (
        this.state.task.config.itemType === types.ItemTypeName.IMAGE &&
        !this.cameraTransformed
      ) {
        const sensorId = this.state.user.viewerConfigs[this.props.id].sensor
        this.transformCamera(sensorId)
        this.commitCamera()
      }
      this._camera.lookAt(this._target)
    }

    if (this._container !== null) {
      const oldAspect = this._camera.aspect
      this._camera.aspect =
        this._container.offsetWidth / this._container.offsetHeight
      this._camera.updateProjectionMatrix()
      if (Math.abs(this._camera.aspect - oldAspect) > 1e-3) {
        this.forceUpdate()
      }
    }
  }

  /**
   * Transform points with sensor extrinsics
   *
   * @param sensorId
   */
  private transformCamera(sensorId: number): void {
    const sensorType = this.state.task.sensors[sensorId]
    const sensor = Sensor.fromSensorType(sensorType)
    const mainSensor = getMainSensor(this.state)

    this._target.copy(
      mainSensor.inverseTransform(sensor.transform(this._target))
    )
    this._camera.position.copy(
      mainSensor.inverseTransform(sensor.transform(this._camera.position))
    )
    this._camera.up.copy(
      mainSensor.inverseTransform(sensor.transform(this._camera.up))
    )
    this.cameraTransformed = true
  }

  /**
   * Normalize coordinates to display
   *
   * @param mX
   * @param mY
   */
  private convertMouseToNDC(mX: number, mY: number): number[] {
    if (this._container !== null) {
      let x = mX / this._container.offsetWidth
      let y = mY / this._container.offsetHeight
      x = 2 * x - 1
      y = -2 * y + 1

      return [x, y]
    }
    return [0, 0]
  }

  /**
   * Toggle locked state
   *
   * @param targetState
   */
  private toggleCameraLock(targetState: number): void {
    const config = this._viewerConfig as PointCloudViewerConfigType
    const newLockState =
      config.lockStatus === targetState ? CameraLockState.UNLOCKED : targetState
    Session.dispatch(updateLockStatus(this.props.id, config, newLockState))
  }

  /**
   * Rotate camera along camera view axis. Returns the rotation applied
   *
   * @param amount
   */
  private rotateCameraViewDirection(amount: number): THREE.Quaternion {
    const axis = new THREE.Vector3()
    this._camera.getWorldDirection(axis)

    const quaternion = new THREE.Quaternion()
    quaternion.setFromAxisAngle(
      axis,
      amount / CameraMovementParameters.MOUSE_CORRECTION_FACTOR
    )

    this._camera.applyQuaternion(quaternion)

    return quaternion
  }

  /**
   * Rotate camera around target
   *
   * @param dx
   * @param dy
   */
  private rotateCameraSpherical(dx: number, dy: number): void {
    if (this._viewerConfig === undefined) {
      return
    }
    const viewerConfig = this._viewerConfig as PointCloudViewerConfigType
    if (viewerConfig.lockStatus === CameraLockState.X_LOCKED) {
      dx = 0
    } else if (viewerConfig.lockStatus === CameraLockState.Y_LOCKED) {
      dy = 0
    }
    const target = new THREE.Vector3(
      this._target.x,
      this._target.y,
      this._target.z
    )
    const offset = new THREE.Vector3(
      this._camera.position.x,
      this._camera.position.y,
      this._camera.position.z
    )

    const sensor = getMainSensor(this.state)
    const up = sensor.up

    offset.sub(target)
    // Rotate so that positive y-axis is vertical
    const rotVertQuat = new THREE.Quaternion().setFromUnitVectors(
      up,
      new THREE.Vector3(0, 1, 0)
    )
    offset.applyQuaternion(rotVertQuat)

    // Convert to spherical coordinates
    const spherical = new THREE.Spherical()
    spherical.setFromVector3(offset)

    // Apply rotations
    // TODO(julin): make this movement customizable
    if (viewerConfig.cameraRotateDir === false) {
      spherical.theta -= dx / CameraMovementParameters.MOUSE_CORRECTION_FACTOR
      spherical.phi -= dy / CameraMovementParameters.MOUSE_CORRECTION_FACTOR
    } else {
      spherical.theta += dx / CameraMovementParameters.MOUSE_CORRECTION_FACTOR
      spherical.phi += dy / CameraMovementParameters.MOUSE_CORRECTION_FACTOR
    }

    spherical.phi = Math.max(0, Math.min(Math.PI, spherical.phi))

    spherical.makeSafe()

    // Convert to Cartesian
    offset.setFromSpherical(spherical)

    // Rotate back to original coordinate space
    const quatInverse = rotVertQuat.clone().invert()
    offset.applyQuaternion(quatInverse)
    offset.add(target)

    this._camera.position.copy(offset)
    this._camera.lookAt(this._target)
    this._camera.up.copy(up)
  }

  /**
   * Drag camera, returns translation
   *
   * @param dx
   * @param dy
   */
  private dragCamera(dx: number, dy: number): THREE.Vector3 {
    const dragVector = new THREE.Vector3(
      (-dx / CameraMovementParameters.MOUSE_CORRECTION_FACTOR) * 2,
      (dy / CameraMovementParameters.MOUSE_CORRECTION_FACTOR) * 2,
      0
    )
    dragVector.applyQuaternion(this._camera.quaternion)

    this._camera.position.add(dragVector)
    this._target.add(dragVector)

    return dragVector
  }

  /**
   * Zoom camera
   *
   * @param dY
   */
  private zoomCamera(dY: number): void {
    if (this._viewerConfig !== undefined) {
      const target = this._target
      const offset = new THREE.Vector3().copy(this._camera.position)
      offset.sub(target)

      const spherical = new THREE.Spherical()
      spherical.setFromVector3(offset)

      // Decrease distance from origin by amount specified
      let newRadius = spherical.radius
      if (dY > 0) {
        newRadius *= CameraMovementParameters.ZOOM_SPEED
      } else {
        newRadius /= CameraMovementParameters.ZOOM_SPEED
      }
      // Limit zoom to not be too close
      if (newRadius > 0.1 && newRadius < 500) {
        spherical.radius = newRadius

        offset.setFromSpherical(spherical)

        offset.add(target)

        this._camera.position.copy(offset)
        Session.label3dList.onDrawableUpdate()
      }
    }
  }

  /** Move camera up */
  private moveUp(): void {
    if (this._viewerConfig !== undefined) {
      this._camera.position.z += CameraMovementParameters.MOVE_AMOUNT
      this._target.z += CameraMovementParameters.MOVE_AMOUNT
      Session.label3dList.onDrawableUpdate()
    }
  }

  /** Move camera down */
  private moveDown(): void {
    if (this._viewerConfig !== undefined) {
      this._camera.position.z -= CameraMovementParameters.MOVE_AMOUNT
      this._target.z -= CameraMovementParameters.MOVE_AMOUNT
      Session.label3dList.onDrawableUpdate()
    }
  }

  /** Move camera forward */
  private moveForward(): void {
    if (this._viewerConfig !== undefined) {
      const forward = calculateForward(
        this._viewerConfig as PointCloudViewerConfigType,
        this.state
      )
      this._camera.position.add(forward)
      this._target.add(forward)
      Session.label3dList.onDrawableUpdate()
    }
  }

  /** Move camera backward */
  private moveBackward(): void {
    if (this._viewerConfig !== undefined) {
      const forward = calculateForward(
        this._viewerConfig as PointCloudViewerConfigType,
        this.state
      )
      this._camera.position.sub(forward)
      this._target.sub(forward)
      Session.label3dList.onDrawableUpdate()
    }
  }

  /** Move camera left */
  private moveLeft(): void {
    if (this._viewerConfig !== undefined) {
      const forward = calculateForward(
        this._viewerConfig as PointCloudViewerConfigType,
        this.state
      )
      const left = calculateLeft(forward, this.state)
      this._camera.position.add(left)
      this._target.add(left)
      Session.label3dList.onDrawableUpdate()
    }
  }

  /** Move camera right */
  private moveRight(): void {
    if (this._viewerConfig !== undefined) {
      const forward = calculateForward(
        this._viewerConfig as PointCloudViewerConfigType,
        this.state
      )
      const left = calculateLeft(forward, this.state)
      this._camera.position.sub(left)
      this._target.sub(left)
      Session.label3dList.onDrawableUpdate()
    }
  }

  /** Commit camera to state */
  private commitCamera(): void {
    const newConfig = {
      ...(this._viewerConfig as PointCloudViewerConfigType),
      position: new Vector3D().fromThree(this._camera.position).toState(),
      target: new Vector3D().fromThree(this._target).toState(),
      cameraTransformed: this.cameraTransformed
    }
    Session.dispatch(changeViewerConfig(this._viewerId, newConfig))
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

export default withStyles(viewerStyles)(Viewer3D)
