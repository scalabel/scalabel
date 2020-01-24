import { Box, IconButton } from '@material-ui/core'
import LockIcon from '@material-ui/icons/Lock'
import SyncIcon from '@material-ui/icons/Sync'
import ThreeSixtyIcon from '@material-ui/icons/ThreeSixty'
import { withStyles } from '@material-ui/styles'
import React from 'react'
import * as THREE from 'three'
import { changeViewerConfig, toggleSynchronization } from '../action/common'
import { alignToAxis, CameraLockState, lockedToSelection, MOUSE_CORRECTION_FACTOR, moveBack, moveCameraAndTarget, moveDown, moveForward, moveLeft, moveRight, moveUp, toggleSelectionLock, updateLockStatus, zoomCamera } from '../action/point_cloud'
import Session from '../common/session'
import * as types from '../common/types'
import { PointCloudViewerConfigType } from '../functional/types'
import { Vector3D } from '../math/vector3d'
import { viewerStyles } from '../styles/viewer'
import { DrawableViewer, ViewerClassTypes, ViewerProps } from './drawable_viewer'
import Label3dCanvas from './label3d_canvas'
import PointCloudCanvas from './point_cloud_canvas'

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

/** Conditionally wrap box with bottom border */
function underlineElement (element: React.ReactElement, underline?: boolean) {
  if (underline) {
    return (
      <Box borderBottom={1} borderColor='grey.500'>
        {element}
      </Box>
    )
  }
  return element
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
  private _target: THREE.Vector3

  /**
   * Constructor
   * @param {Object} props: react props
   */
  constructor (props: Props) {
    super(props)
    this._camera = new THREE.PerspectiveCamera(45, 1, 0.2, 1000)
    this._raycaster = new THREE.Raycaster()
    this._target = new THREE.Vector3()
  }

  /** Called when component updates */
  public componentDidUpdate () {
    if (this._viewerConfig) {
      this.updateCamera(this._viewerConfig as PointCloudViewerConfigType)

      if (Session.activeViewerId === this.props.id) {
        Session.label3dList.setActiveCamera(this._camera)
      }

      // this._camera.layers.set(
      //   this.props.id - Math.min(
      //     ...Object.keys(this.state.user.viewerConfigs).map(
      //       (key) => Number(key)
      //     )
      //   )
      // )
    }
  }

  /**
   * Render function
   * @return {React.Fragment} React fragment
   */
  protected getDrawableComponents () {
    const views: React.ReactElement[] = []
    if (this._viewerConfig) {
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

    }

    return views
  }

  /** Return menu buttons */
  protected getMenuComponents () {
    if (this._viewerConfig) {
      const config = this._viewerConfig as PointCloudViewerConfigType
      const yLockButton = (
        <IconButton
          onClick={() => this.toggleCameraLock(CameraLockState.Y_LOCKED)}
          className={this.props.classes.viewer_button}
        >
            {
              underlineElement(
                <div className={this.props.classes.camera_y_lock_icon}>
                  <ThreeSixtyIcon />
                </div>,
                config.lockStatus === CameraLockState.Y_LOCKED
              )
            }
        </IconButton>
      )
      const xLockButton = (
        <IconButton
          onClick={() => this.toggleCameraLock(CameraLockState.X_LOCKED)}
          className={this.props.classes.viewer_button}
          edge={'start'}
        >
            {
              underlineElement(
                <div className={this.props.classes.camera_x_lock_icon}>
                  <ThreeSixtyIcon />
                </div>,
                config.lockStatus === CameraLockState.X_LOCKED
              )
            }
        </IconButton>
      )
      const xAxisButton = (
        <IconButton
          className={this.props.classes.viewer_button}
          onClick={() => Session.dispatch(alignToAxis(
            this.props.id,
            config,
            0
          ))}
          edge={'start'}
        >
          {
            underlineElement(
              <span style={{ color: '#ff7700' }}>X</span>,
              config.lockStatus === CameraLockState.SELECTION_X
            )
          }
        </IconButton>
      )
      const yAxisButton = (
        <IconButton
          className={this.props.classes.viewer_button}
          onClick={() => Session.dispatch(alignToAxis(
            this.props.id,
            config,
            1
          ))}
        >
          {
            underlineElement(
              <span style={{ color: '#77ff77' }}>Y</span>,
              config.lockStatus === CameraLockState.SELECTION_Y
            )
          }
        </IconButton>
      )
      const zAxisButton = (
        <IconButton
          className={this.props.classes.viewer_button}
          onClick={() => Session.dispatch(alignToAxis(
            this.props.id,
            config,
            2
          ))}
        >
          {
            underlineElement(
              <span style={{ color: '#0088ff' }}>Z</span>,
              config.lockStatus === CameraLockState.SELECTION_Z
            )
          }
        </IconButton>
      )
      const synchronizationButton = (
        <IconButton
          className={this.props.classes.viewer_button}
          onClick={() => {
            if (this._viewerConfig) {
              Session.dispatch(toggleSynchronization(
                this._viewerId, config
              ))
            }
          }}
        >
          {underlineElement(<SyncIcon />, config.synchronized)}
        </IconButton >
      )
      const selectionLockButton = (
        <IconButton
          className={this.props.classes.viewer_button}
          onClick={() => {
            if (this._viewerConfig) {
              Session.dispatch(toggleSelectionLock(
                this._viewerId,
                config
              ))
            }
          }}
        >
          {underlineElement(
            <LockIcon />,
            lockedToSelection(config)
          )}
        </IconButton>
      )

      return [
        yLockButton,
        xLockButton,
        xAxisButton,
        yAxisButton,
        zAxisButton,
        synchronizationButton,
        selectionLockButton
      ]
    }
    return []
  }

  /** Mouse enter */
  protected onMouseEnter (e: React.MouseEvent) {
    super.onMouseEnter(e)
    Session.label3dList.setActiveCamera(this._camera)
  }

  /**
   * Handle mouse move
   * @param e
   */
  protected onMouseMove (e: React.MouseEvent) {
    const oldX = this._mX
    const oldY = this._mY
    super.onMouseMove(e)
    const dX = this._mX - oldX
    const dY = this._mY - oldY
    if (this._mouseDown && this._viewerConfig) {
      const lockSelection =
        lockedToSelection(this._viewerConfig as PointCloudViewerConfigType)
      if (this._mouseButton === 2) {
        const delta = this.dragCamera(dX, dY)
        if (lockSelection && Session.label3dList.selectedLabel) {
          Session.label3dList.selectedLabel.translate(delta)
        }
      } else {
        if (lockSelection && Session.label3dList.selectedLabel) {
          const quaternion = this.rotateCameraViewDirection(dX)
          Session.label3dList.selectedLabel.rotate(quaternion)
        } else {
          this.rotateCameraSpherical(dX, dY)
        }

        Session.label3dList.onDrawableUpdate()
      }
    }
  }

  /** Handle mouse up */
  protected onMouseUp (e: React.MouseEvent) {
    super.onMouseUp(e)
    this.commitCamera()
  }

  /**
   * Handle double click
   * @param e
   */
  protected onDoubleClick (e: React.MouseEvent) {
    if (!this._container || !this._viewerConfig) {
      return
    }

    const normalized = this.normalizeCoordinates(e.clientX, e.clientY)

    const NDC = this.convertMouseToNDC(
      normalized[0],
      normalized[1]
    )

    this._raycaster.linePrecision = 0.2
    this._raycaster.setFromCamera(
      new THREE.Vector2(NDC[0], NDC[1]), this._camera
    )
    const pointCloud =
      Session.pointClouds[this._item][this._viewerConfig.sensor]

    const intersects = this._raycaster.intersectObject(pointCloud)

    if (intersects.length > 0) {
      const newTarget = intersects[0].point
      const viewerConfig = this._viewerConfig as PointCloudViewerConfigType
      Session.dispatch(moveCameraAndTarget(
        new Vector3D(
          viewerConfig.position.x - viewerConfig.target.x + newTarget.x,
          viewerConfig.position.y - viewerConfig.target.y + newTarget.y,
          viewerConfig.position.z - viewerConfig.target.z + newTarget.z
        ),
        new Vector3D(
          newTarget.x,
          newTarget.y,
          newTarget.z
        ),
        this._viewerId,
        viewerConfig
      ))
    }
  }

  /**
   * Handle mouse leave
   * @param e
   */
  protected onMouseLeave () {
    return
  }

  /**
   * Handle mouse wheel
   * @param e
   */
  protected onWheel (e: WheelEvent) {
    e.preventDefault()
    if (this._viewerConfig) {
      const pointCloudZoomAction = zoomCamera(
        e.deltaY,
        this._viewerId,
        this._viewerConfig as PointCloudViewerConfigType
      )
      if (pointCloudZoomAction) {
        Session.dispatch(pointCloudZoomAction)
      }
    }
  }

  /** Override key handler */
  protected onKeyDown (e: KeyboardEvent): void {
    const viewerConfig = this._viewerConfig as PointCloudViewerConfigType
    if (lockedToSelection(viewerConfig)) {
      return
    }
    switch (e.key) {
      case types.Key.PERIOD:
        Session.dispatch(moveUp(this._viewerId, viewerConfig))
        break
      case types.Key.SLASH:
        Session.dispatch(moveDown(this._viewerId, viewerConfig))
        break
      case types.Key.S_LOW:
      case types.Key.S_UP:
        Session.dispatch(moveBack(this._viewerId, viewerConfig))
        break
      case types.Key.W_LOW:
      case types.Key.W_UP:
        Session.dispatch(moveForward(this._viewerId, viewerConfig))
        break
      case types.Key.A_LOW:
      case types.Key.A_UP:
        Session.dispatch(moveLeft(this._viewerId, viewerConfig))
        break
      case types.Key.D_LOW:
      case types.Key.D_UP:
        Session.dispatch(moveRight(this._viewerId, viewerConfig))
        break
    }
  }

  /** update camera parameters with config */
  private updateCamera (config: PointCloudViewerConfigType) {
    if (!this._viewerConfig) {
      return
    }

    if (this._container) {
      this._camera.aspect = this._container.offsetWidth /
        this._container.offsetHeight
      this._camera.updateProjectionMatrix()
    }

    this._target.set(
      config.target.x,
      config.target.y,
      config.target.z
    )

    if (
      lockedToSelection(config) &&
      Session.label3dList.selectedLabel
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

      const position = (new Vector3D()).fromObject(config.position).toThree()
      const target = (new Vector3D()).fromObject(config.target).toThree()

      const offset = (new THREE.Vector3()).copy(position)
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
      this._camera.up.x = config.verticalAxis.x
      this._camera.up.y = config.verticalAxis.y
      this._camera.up.z = config.verticalAxis.z
      this._camera.position.x = config.position.x
      this._camera.position.y = config.position.y
      this._camera.position.z = config.position.z
      this._camera.lookAt(this._target)
    }
  }

  /**
   * Normalize coordinates to display
   * @param mX
   * @param mY
   */
  private convertMouseToNDC (mX: number, mY: number) {
    if (this._container) {
      let x = mX / this._container.offsetWidth
      let y = mY / this._container.offsetHeight
      x = 2 * x - 1
      y = -2 * y + 1

      return [x, y]
    }
    return [0, 0]
  }

  /** Toggle locked state */
  private toggleCameraLock (targetState: number) {
    const config = this._viewerConfig as PointCloudViewerConfigType
    const newLockState =
      (config.lockStatus === targetState) ?
        CameraLockState.UNLOCKED : targetState
    Session.dispatch(updateLockStatus(
      this.props.id,
      config,
      newLockState
    ))
  }

  /** Rotate camera along camera view axis. Returns the rotation applied */
  private rotateCameraViewDirection (amount: number): THREE.Quaternion {
    const axis = new THREE.Vector3()
    this._camera.getWorldDirection(axis)

    const quaternion = new THREE.Quaternion()
    quaternion.setFromAxisAngle(axis, amount / MOUSE_CORRECTION_FACTOR)

    this._camera.applyQuaternion(quaternion)

    return quaternion
  }

  /** Rotate camera around target */
  private rotateCameraSpherical (dx: number, dy: number) {
    if (!this._viewerConfig) {
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

    offset.sub(target)

    // Rotate so that positive y-axis is vertical
    const rotVertQuat = new THREE.Quaternion().setFromUnitVectors(
      new THREE.Vector3(0, 0, 1),
      new THREE.Vector3(0, 1, 0))
    offset.applyQuaternion(rotVertQuat)

    // Convert to spherical coordinates
    const spherical = new THREE.Spherical()
    spherical.setFromVector3(offset)

    // Apply rotations
    spherical.theta += dx / MOUSE_CORRECTION_FACTOR
    spherical.phi += dy / MOUSE_CORRECTION_FACTOR

    spherical.phi = Math.max(0, Math.min(Math.PI, spherical.phi))

    spherical.makeSafe()

    // Convert to Cartesian
    offset.setFromSpherical(spherical)

    // Rotate back to original coordinate space
    const quatInverse = rotVertQuat.clone().inverse()
    offset.applyQuaternion(quatInverse)
    offset.add(target)

    this._camera.position.copy(offset)
    this._camera.lookAt(this._target)
  }

  /** Drag camera, returns translation */
  private dragCamera (dx: number, dy: number): THREE.Vector3 {
    const dragVector = new THREE.Vector3(
      -dx / MOUSE_CORRECTION_FACTOR * 2,
      dy / MOUSE_CORRECTION_FACTOR * 2,
      0
    )
    dragVector.applyQuaternion(this._camera.quaternion)

    this._camera.position.add(dragVector)
    this._target.add(dragVector)

    return dragVector
  }

  /** Commit camera to state */
  private commitCamera () {
    const newConfig = {
      ...this._viewerConfig as PointCloudViewerConfigType,
      position: (new Vector3D()).fromThree(this._camera.position).toObject(),
      target: (new Vector3D()).fromThree(this._target).toObject()
    }
    Session.dispatch(changeViewerConfig(
      this._viewerId, newConfig
    ))
  }
}

export default withStyles(viewerStyles)(Viewer3D)
